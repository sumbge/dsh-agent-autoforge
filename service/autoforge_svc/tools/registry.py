"""The tool registry: hot-swappable, self-policing, observable.

Three properties the framework needs that a plain dict of functions lacks:

1. **Hot swap** — tools can be added/removed while the agent runs. The
   registry owns the schema list handed to the LLM, so a newly forged tool is
   usable on the very next turn.
2. **Policing** — every call is recorded in the spec's ledger. A tool that
   degrades below threshold is auto-quarantined; a quarantined tool is hidden
   from context but still executable via `force=True`.
3. **Observability** — `report()` gives a full picture of tool health, which
   is what makes "did my agent get better or worse" an answerable question.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from .spec import ToolSpec, ToolState


class ToolAborted(Exception):
    """A call gave up because the operator asked for the floor.

    Not every `Exception` out of a tool is a tool failure. "The person running
    this said something and wants the turn" is a decision *about* the call, not
    evidence about the tool, and the two must not arrive at the ledger wearing
    the same clothes — that is how three operator interruptions retire a
    perfectly good tool. A runner that can be cut short raises this; the
    registry reports it as `aborted=True` and leaves `spec.stats` alone.
    """


@dataclass
class ToolResult:
    name: str
    ok: bool
    output: str
    error: str | None = None
    duration_ms: float = 0.0
    quarantined: bool = False
    awaiting_confirmation: bool = False   # stopped at the autonomy gate
    aborted: bool = False                 # the operator asked for the floor

    def to_json(self) -> str:
        return json.dumps(
            {"ok": self.ok, "output": self.output, "error": self.error},
            ensure_ascii=False,
        )


class ToolRegistry:
    def __init__(
        self,
        *,
        min_calls_for_judgement: int = 3,
        quarantine_success_rate: float = 0.5,
        quarantine_consecutive_failures: int = 3,
        auto_quarantine: bool = True,
        visible_states: set[ToolState] | None = None,
        policy: Any = None,
        confirmer: Any = None,
        on_state_change: Callable[[ToolSpec], None] | None = None,
        controls: Any = None,
    ) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.min_calls_for_judgement = min_calls_for_judgement
        self.quarantine_success_rate = quarantine_success_rate
        self.quarantine_consecutive_failures = quarantine_consecutive_failures
        self.auto_quarantine = auto_quarantine
        # -- the autonomy gate (see autonomy/confirm.py) -----------------
        # `call` is the one place every tool actually runs, which is why the
        # confirmation gate lives here rather than in each tool: a freedom the
        # policy switched off stops the call and asks the operator. Both are
        # None by default, and None means no gate — a bare registry behaves
        # exactly as it did before this existed.
        self.policy = policy
        self.confirmer = confirmer
        # Where a gate decision goes so that it outlives the process. Optional,
        # and defaulted lazily rather than imported at module scope: `registry`
        # is imported by the sandbox's own tests, and reaching for the audit
        # package from here on import would make a tool registry depend on
        # sqlite. `None` means "use the process plane", which is itself inert
        # until something binds a ledger to it.
        self.controls = controls
        # Told whenever a tool's *state* moves, so a decision made here can
        # outlive the process. A registry alone persists nothing: it is a dict
        # of specs, and retire/quarantine/promote are memory writes. That was
        # invisible while nothing read the state back — but the store does, as
        # the first thing the next session does, and it can only read a value
        # some transition wrote. So a tool retired as harmful came back
        # `active` on the next start, and the ledger said "retired" while the
        # registry said "fine": two records of the same decision disagreeing,
        # with nothing comparing them. `None` leaves a bare registry as it was.
        self.on_state_change = on_state_change
        # Which states get injected into the LLM's context by default.
        # Trusted-only is the production stance. Verification harnesses widen
        # this to include DRAFT so a tool can be tested before it is trusted.
        self.visible_states = visible_states or {ToolState.PROBATION, ToolState.ACTIVE}
        self._events: list[dict[str, Any]] = []
        #: Where "when was this last called" comes from. A callable rather than
        #: a field because the answer outlives this process: the registry's own
        #: `spec.stats.last_called` is only as good as the last `save_tool`, and
        #: a call is not a state change, so most calls are never written back.
        #: The ledger has every call; this is how the ledger gets asked. None
        #: falls back to `spec.stats.last_called`, which is the right answer for
        #: a bare registry with no store behind it (tests, embedders).
        self.last_called_source: Callable[[str], float | None] | None = None
        #: Calls made through this registry, which is what makes the exposure
        #: cache below safe to keep: the thing the exposure depends on -- how
        #: recently each tool was used -- can only change when a call happens.
        #: How many days of silence demote a tool from "carried in full" to
        #: "named on a list". A week, because the ledger says that is roughly
        #: the working set: 40 of 204 tools on this machine at 7 days, 143 at
        #: 14. Read by the exposure cache's key and by the block that explains
        #: the demotion to the model, so the sentence and the behaviour cannot
        #: disagree.
        self.window_days = 7.0
        #: Names to keep hot whatever the ledger says, set by the caller per
        #: run. A *past* record cannot answer whether a tool is wanted for
        #: *this* task, and the run that needs a tool it has not used this week
        #: is exactly the run whose first turn would otherwise not carry it.
        self.protect: set[str] = set()
        self._calls_seen = 0
        self._exposure_cache: tuple[Any, ...] | None = None
        self._exposure_result: tuple[list[dict[str, Any]], list[str],
                                     dict[str, int]] | None = None

    @property
    def _plane(self) -> Any:
        """The control plane, resolved on first use.

        Not resolved in `__init__`: a registry built inside a fresh process --
        which is every test -- would then import the audit layer whether or not
        it ever gated anything. Lazy keeps the cost where the decision is.
        """
        if self.controls is not None:
            return self.controls
        from ..autonomy.controls import plane

        # Always the process plane, never a null one chosen by guessing from the
        # environment: an unbound plane is already inert (it counts nothing,
        # writes nothing, and says `writing: False`), and a registry that picked
        # the null plane because an env var was absent would go on recording
        # nothing after `bind()` had attached a ledger. Which one to use is a
        # question about the ledger, and the plane can answer it; the registry
        # cannot.
        self.controls = plane()
        return self.controls

    # -- registration -------------------------------------------------
    def register(self, spec: ToolSpec, *, replace: bool = True) -> ToolSpec:
        if spec.name in self._tools and not replace:
            raise ValueError(f"tool {spec.name!r} already registered")
        self._tools[spec.name] = spec
        self._log("register", spec.name, {"state": spec.state.value, "hash": spec.hash})
        return spec

    def unregister(self, name: str) -> None:
        if name in self._tools:
            del self._tools[name]
            self._log("unregister", name, {})

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    # -- schema exposure ----------------------------------------------
    def schemas(self, *, include: set[ToolState] | None = None) -> list[dict[str, Any]]:
        """Schemas to hand the LLM.

        By default only `visible_states` (trusted) are exposed: DRAFT is
        unverified, QUARANTINED is degraded, RETIRED is gone. Pass `include`
        to widen for one call (e.g. a verification harness that must see the
        tool it is testing).
        """
        if include is None:
            include = self.visible_states
        return [s.schema for s in self._tools.values() if s.state in include]

    # -- exposure: what rides in every request, and what is only listed -----
    def days_since_called(self, name: str) -> float | None:
        """How long ago this tool was last called, or None if never.

        The ledger first, `spec.stats` second. The order matters: a call is not
        a state change, so `stats.last_called` only survives processes where
        something happened to save the tool -- which is to say, almost never.
        The ledger answers for every call ever made on this machine, which is
        the only source with enough rows to demote anything honestly.
        """
        if self.last_called_source is not None:
            try:
                ts = self.last_called_source(name)
            except Exception:                             # noqa: BLE001
                ts = None
            if ts:
                return max(0.0, (time.time() - float(ts)) / 86400.0)
        spec = self._tools.get(name)
        if spec is None or not spec.stats.last_called:
            return None
        return max(0.0, (time.time() - spec.stats.last_called) / 86400.0)

    def exposure(self, *, window_days: float | None = None,
                 protect: "Iterable[str] | None" = None,
                 include: "set[ToolState] | None" = None
                 ) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
        """(hot schemas, cold lines, counts) for the next request.

        Cached until a call happens or the hour turns over, because those are
        the only two things that can change the answer. Without the cache this
        would run a query per tool per turn, which is a real cost paid to save a
        real cost; with it, the prompt it builds is byte-identical between two
        turns that made no call, which is what the provider's prefix cache needs.

        The hour is in the key because a call is *not* the only thing that moves
        the answer -- time does. A tool called yesterday is hot for seven days
        and must go cold on the eighth without anybody calling anything. A cache
        keyed only on the call counter gets that right for every session that
        restarts, and wrong for one that stays up: measured from the counter
        alone, a long-lived session would keep yesterday's working set for as
        long as it ran. One bucket per hour bounds the error at an hour and
        re-bills a prompt prefix once an hour, which is nothing against a window
        measured in days.
        """
        from . import exposure as exposure_mod

        if window_days is None:
            window_days = self.window_days
        if protect is None:
            protect = self.protect
        key = (round(window_days, 4), tuple(sorted(protect)),
               tuple(sorted(s.value for s in (include or self.visible_states))),
               self._calls_seen, len(self._tools),
               int(time.time() // 3600))
        if self._exposure_cache == key and self._exposure_result is not None:
            return self._exposure_result
        result = exposure_mod.split(
            self.schemas(include=include) if include is not None else self.schemas(),
            self.days_since_called,
            window_days=window_days,
            protect=protect,
        )
        self._exposure_cache = key
        self._exposure_result = result
        return result

    # -- invocation ----------------------------------------------------
    def call(self, name: str, arguments: dict[str, Any], *, force: bool = False) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(name, False, "", error=f"unknown tool {name!r}")

        if spec.state in (ToolState.QUARANTINED, ToolState.RETIRED) and not force:
            spec.stats.record_call(False, "blocked: quarantined")
            self._log("blocked", name, {"state": spec.state.value})
            return ToolResult(
                name, False, "",
                error=f"tool {name!r} is {spec.state.value}; pass force=True to run anyway",
                quarantined=True,
            )

        blocked = self._gate(spec, name, arguments)
        if blocked is not None:
            return blocked

        # Only here: after the state check and after the gate. A tool that was
        # quarantined or that the operator refused did not run, and counting it
        # as used would keep a tool hot on the strength of calls it never made.
        self._calls_seen += 1
        spec.stats.last_called = time.time()

        started = time.perf_counter()
        try:
            if spec.runner is not None:
                out = spec.runner(name, arguments)
            else:
                out = spec.fn(**arguments)
            # A runner may answer with a ToolResult of its own. `spec.runner`
            # is documented as `ToolResult-ish` and an out-of-process tool
            # knows its own failure semantics better than we can infer them
            # -- an MCP server's `isError` is not an exception, and a sandbox
            # timeout is not an empty success. Without this branch the object
            # fell through to `json.dumps`, so the registry reported ok=True
            # and an `output` of the dataclass repr: a failure laundered into
            # a success with the real error buried in a string.
            if isinstance(out, ToolResult):
                out.duration_ms = (time.perf_counter() - started) * 1000
                spec.stats.record_call(out.ok, out.error)
                if self.auto_quarantine:
                    self._maybe_quarantine(spec)
                return out
            if out is None:
                out = ""
            elif not isinstance(out, str):
                out = json.dumps(out, ensure_ascii=False, default=str)
            ok, err = True, None
        except Exception as exc:  # noqa: BLE001 - tool boundary must never crash the loop
            out = ""
            ok, err = False, f"{type(exc).__name__}: {exc}"
            self._log("error", name, {"error": err, "tb": traceback.format_exc()[-2000:]})

        duration = (time.perf_counter() - started) * 1000
        spec.stats.record_call(ok, err)
        if self.auto_quarantine:
            self._maybe_quarantine(spec)
        return ToolResult(name, ok, out, err, duration)

    # -- the autonomy gate ---------------------------------------------
    def _gate(self, spec: ToolSpec, name: str,
              arguments: dict[str, Any]) -> ToolResult | None:
        """Ask before running a tool whose declared scope needs a freedom the
        policy has switched off. Returns None when the call may proceed.

        The confirmer answers True (run it), False (refuse it) or None (nobody
        to ask — a job, a thread, a closed stdin). The three are kept apart on
        purpose: "an operator said no" and "there was no operator" are different
        facts, and a gate that reports the second as the first is lying about
        who decided.

        Three things this deliberately does NOT do:

        * It does not guess. A tool that declares no scope needs every freedom,
          so it is gated too — an unlabelled tool is not evidence of innocence.
        * It does not count against the tool. A refusal is the operator's
          decision, not a failure, so `spec.stats` is left alone. Otherwise
          three refusals in a row would auto-quarantine a working tool.
        * It does not read silence as consent. No answer, no run.
        """
        if self.policy is None or not self.policy.denied:
            return None   # nothing is switched off, so nothing can be gated

        from ..autonomy.confirm import describe, disabled_freedoms

        needed = disabled_freedoms(getattr(spec, "effect_signature", ""), self.policy)
        if not needed:
            return None

        question = describe(needed, name, arguments)
        answer: bool | None = None
        if self.confirmer is not None:
            try:
                answer = self.confirmer(name, arguments, needed)
            except Exception as exc:  # noqa: BLE001 - a broken confirmer must not run the tool
                self._log("confirm", name, {"needed": needed, "outcome": "error",
                                            "error": f"{type(exc).__name__}: {exc}"})
                self._plane.gate(tool=name, needed=needed, outcome="error",
                                 question=question, error=f"{type(exc).__name__}: {exc}")
                return ToolResult(
                    name, False, "",
                    error=(f"Denied: the confirmer raised {type(exc).__name__}: {exc}. "
                           f"A gate that cannot be asked is a gate that says no."),
                    awaiting_confirmation=True,
                )

        if answer is None:
            self._log("confirm", name, {"needed": needed, "outcome": "no_operator"})
            # Recorded as a denial with its own outcome: a tool that did not run
            # because nobody could answer looks exactly like a tool that ran and
            # did nothing, and the ledger is where that difference has to survive.
            self._plane.gate(tool=name, needed=needed, outcome="no_operator",
                             question=question)
            return ToolResult(
                name, False, "",
                error=(f"Denied: {question} There is nobody to ask in this run "
                       f"(no operator attached to the confirmation gate), so the "
                       f"answer is no. Re-enable {', '.join(needed)}, or run where "
                       f"an operator can answer."),
                awaiting_confirmation=True,
            )

        self._log("confirm", name,
                  {"needed": needed, "outcome": "confirmed" if answer else "refused"})
        self._plane.gate(tool=name, needed=needed,
                         outcome="confirmed" if answer else "refused",
                         question=question)
        if answer:
            return None
        return ToolResult(
            name, False, "", error=f"Denied by operator: {question}",
            awaiting_confirmation=True,
        )

    # -- health management ---------------------------------------------
    def _maybe_quarantine(self, spec: ToolSpec) -> None:
        st = spec.stats
        if st.calls < self.min_calls_for_judgement:
            return
        degraded = (
            st.success_rate < self.quarantine_success_rate
            or st.consecutive_failures >= self.quarantine_consecutive_failures
        )
        if degraded and spec.state in (ToolState.ACTIVE, ToolState.PROBATION):
            # Through the funnel, not straight to the attribute: this is the
            # safety valve, and a safety valve that re-opens on restart is not
            # one. A tool quarantined for failing three calls in a row used to
            # be back on active duty the next time the agent started, because
            # the store was never told.
            self._set_state(spec, ToolState.QUARANTINED, "auto_quarantine",
                            {
                                "success_rate": round(st.success_rate, 3),
                                "consecutive_failures": st.consecutive_failures,
                                "calls": st.calls,
                            })

    def _set_state(self, spec: ToolSpec, state: ToolState, kind: str,
                   extra: dict[str, Any] | None = None) -> None:
        """The one place a tool's state moves, and the one place it is announced.

        Every transition routes through here for the same reason the agent
        funnels its events through `_record`: the alternative is a transition
        that moves the state and forgets to tell anyone, and that is not a
        hypothetical — retirement and quarantine both moved the state and told
        only the in-memory event log, which is how a judgement about a harmful
        tool came to expire with the process that made it.
        """
        spec.state = state
        self._log(kind, spec.name, extra or {})
        if self.on_state_change is not None:
            try:
                self.on_state_change(spec)
            except Exception:                # noqa: BLE001
                # The state has already moved in memory; a store that is
                # locked, full or closed must not turn that into a failed
                # call. The cost of swallowing this is a decision that does
                # not survive the restart — visible, and cheaper than a call
                # that raises after it already took effect.
                pass

    def rehab(self, name: str, *, require_clean: int = 2) -> ToolSpec:
        """Bring a quarantined tool back on trial after it proves itself again."""
        spec = self._tools[name]
        spec.stats.consecutive_failures = 0
        spec.stats.calls = 0
        spec.stats.successes = 0
        spec.stats.failures = 0
        self._set_state(spec, ToolState.PROBATION, "rehab",
                        {"require_clean": require_clean})
        return spec

    def retire(self, name: str) -> None:
        if name in self._tools:
            self._set_state(self._tools[name], ToolState.RETIRED, "retire")

    def quarantine(self, name: str, reason: str = "manual") -> None:
        if name in self._tools:
            self._set_state(self._tools[name], ToolState.QUARANTINED,
                            "quarantine", {"reason": reason})

    def promote(self, name: str) -> ToolSpec:
        spec = self._tools[name]
        self._set_state(spec, ToolState.ACTIVE, "promote")
        return spec

    # -- reporting -------------------------------------------------------
    def report(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        for s in self._tools.values():
            by_state[s.state.value] = by_state.get(s.state.value, 0) + 1
        return {
            "total": len(self._tools),
            "by_state": by_state,
            "events": len(self._events),
            "tools": [s.to_dict() for s in self._tools.values()],
        }

    def events(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._events[-limit:]

    def _log(self, kind: str, name: str, extra: dict[str, Any]) -> None:
        self._events.append({"t": time.time(), "kind": kind, "tool": name, **extra})

    # -- scoping -----------------------------------------------------------
    def scoped(self, allowed, *, role: str = "",
               deny: Callable[[ToolSpec], str] | None = None) -> "ToolRegistry":
        """A view of this registry that can only see and call what it permits.

        Two ways to narrow, and they compose:

        `allowed` is a name whitelist. Empty means no name restriction, and with
        no `deny` either, the registry itself is returned — so a node that names
        no whitelist behaves exactly as before and this stays a pure widening of
        what roles can express.

        `deny` is a predicate over the whole `ToolSpec`, returning "" to permit
        or the reason to refuse. A name whitelist cannot express a role like
        "may read but may not write", because a critic's tool set depends on what
        each tool *does* rather than what it is called — and forged tools are
        named at runtime. The predicate lets the role's ceiling live in one
        place (`autonomy/roles.py`) while the enforcement stays here.
        """
        allowed = set(allowed or ())
        if not allowed and deny is None:
            return self
        return ScopedRegistry(self, allowed, role=role, deny=deny)


class ScopedRegistry(ToolRegistry):
    """Restricts a child agent to a role's whitelist and ceiling, for real.

    Exists because `tools_whitelist` used to be parsed, stored, round-tripped
    through the store and asserted in tests — and consulted by nothing. A role
    that does not restrict anything is a label, not a role. This makes it bite
    at both ends: the permitted subset is all the child's model can *see*
    (`schemas`), and all it can *run* (`call`). Both matter — hiding the rest
    stops the model asking, refusing stops it succeeding if it asks anyway.

    State lives on the parent, so stats, events and any tool a child forges
    still land in the one real registry. This is a lens, not a fork.
    """

    def __init__(self, base: ToolRegistry, allowed: set[str], *, role: str = "",
                 deny: Callable[[ToolSpec], str] | None = None) -> None:
        # Deliberately not calling super().__init__: this view owns no tools of
        # its own. Every registration is the parent's.
        self._base = base
        self._allowed = set(allowed)
        self._deny = deny
        self.role = role
        self.refusals: list[str] = []

    def _refusal(self, name: str) -> str:
        """Why this view will not run `name`. "" means permitted.

        One decision point for `allows`, `names`, `schemas` and `call`, so the
        model's view and the model's reach cannot disagree — a tool visible in
        the schema list but refused at call time trains the model to ask for
        things it cannot have.
        """
        if self._allowed and name not in self._allowed:
            label = f" ({self.role})" if self.role else ""
            return (f"Denied: '{name}' is outside this agent's role{label}. "
                    f"Allowed here: {', '.join(sorted(self._allowed)) or 'nothing'}.")
        if self._deny is not None:
            spec = self._base.get(name)
            if spec is not None:
                return self._deny(spec)
        return ""

    def allows(self, name: str) -> bool:
        return not self._refusal(name)

    def names(self) -> list[str]:
        return [n for n in self._base.names() if self.allows(n)]

    def schemas(self, *, include: set[ToolState] | None = None) -> list[dict[str, Any]]:
        return [
            s for s in self._base.schemas(include=include)
            if self.allows(s.get("function", {}).get("name"))
        ]

    def call(self, name: str, arguments: dict[str, Any], *,
             force: bool = False) -> ToolResult:
        why = self._refusal(name)
        if why:
            self.refusals.append(name)
            return ToolResult(name, False, "", error=why)
        return self._base.call(name, arguments, force=force)

    def __getattr__(self, item: str) -> Any:
        # Everything else — register, unregister, report, the ledger — belongs
        # to the parent. __getattr__ only fires when normal lookup fails, and
        # _base/_allowed/role/refusals are instance attributes, so they do not
        # reach here.
        return getattr(self._base, item)


__all__ = ["ToolRegistry", "ScopedRegistry", "ToolResult"]
