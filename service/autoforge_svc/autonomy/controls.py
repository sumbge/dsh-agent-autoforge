"""The control plane's receipts: every judgement the machinery makes, on the chain.

Why this exists
---------------
Three things in this package decide something and then forget they decided:

  * `tools/registry.py::_gate` — the operator was asked, and answered yes, no,
    or was not there. The outcome went to `ToolRegistry._events`, a list that
    lives in memory and dies with the process.
  * `forge/manifest.py::CapabilityManifest.admit` — a declaration was checked
    against the ceiling. Nothing has ever called it, so nothing has ever
    recorded that a declaration was judged.
  * `autonomy/selfmod.py::SelfModifier.amend` — a change to the agent was
    accepted, refused for want of a rationale, refused as a no-op, or vetoed.
    `amendments` is a list on an object that is rebuilt every session.

The ledger is the one durable record this project has. A decision that is real
enough to stop a tool from running is real enough to be a row in it — otherwise
"who allowed this, and what did they see" is answerable only from memory, and
memory is exactly what the next session does not have.

What this is not
----------------
It is not a second log. Every write goes through `audit.record`, which goes
through `chaining.append_event`, so these rows share the chain, the writer id
and `verify_chain` with every forge and run. A control plane with its own file
would be a second hash chain, and two chains is the same as none.

It is not a gate either. This observes; it never returns a verdict, and no
caller branches on it. A recorder that could refuse would need a policy, and
"the thing that decides" and "the thing that remembers what was decided" are
easier to trust when they are not the same object.

One plane instance per agent, held as `agent.controls`, and optional at every
call site: with no plane attached, `null_plane` swallows the receipt and the
behaviour is byte for byte what it was before. What it must never do is fail
*silently* — a control plane that cannot write says so, on the object, and in
`summary()`.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

__all__ = ["ControlPlane", "plane", "bind", "null_plane"]

#: Outcomes the autonomy gate can reach, named once so the ledger vocabulary and
#: the tests cannot drift. `no_operator` is kept apart from `refused` for the
#: same reason `_gate` returns them apart: "an operator said no" and "there was
#: nobody to ask" are different facts about the same non-event, and a record
#: that merges them accuses somebody who was not there.
GATE_OUTCOMES = ("confirmed", "refused", "no_operator", "error")


class ControlPlane:
    """Records what the control layer decided, on the existing audit chain."""

    def __init__(self, *, conn: Any = None, agent: str = "autoforge",
                 alert: Any = None, enabled: bool = True) -> None:
        self.conn = conn
        self.agent = agent
        self.alert = alert
        self.enabled = enabled
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._failures: list[dict[str, Any]] = []

    # -- wiring ----------------------------------------------------------

    def bind(self, conn: Any) -> "ControlPlane":
        """Attach a ledger. Before this, receipts are counted but not written."""
        self.conn = conn
        return self

    @property
    def writing(self) -> bool:
        return bool(self.conn is not None and self.enabled)

    # -- the write path --------------------------------------------------

    def _record(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """One row, counted, and never allowed to break the caller.

        A control plane on the path between an operator's answer and the tool
        that answer authorises must not be able to turn a working run into a
        broken one. But a swallowed failure is how a log becomes decorative, so
        every failure is kept on `self._failures` and shows up in `summary()`.
        """
        kind = payload.get("event_type", "?")
        # The lock is held across the *write*, not merely across the counter.
        # Measured: 16 threads appending through one shared sqlite connection
        # landed 5 rows and raised 11 `another row available` errors — sqlite
        # refuses a second statement on a connection that is mid-iteration, and
        # `chaining.append_event` iterates to read the head hash before it
        # inserts. So two gate decisions made at the same moment were not two
        # rows: they were one row and one lost refusal. Serialising here fixes
        # it for every writer that goes through this plane; the same hazard
        # remains in `store.log_event` for callers that bypass it.
        with self._lock:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            if not self.writing:
                return None
            try:
                from .. import audit

                return audit.record(self.conn, payload, writer=self.agent)
            except Exception as exc:                          # noqa: BLE001
                self._failures.append({"event_type": kind, "ts": time.time(),
                                       "error": "%s: %s" % (type(exc).__name__, exc)})
                return None

    # -- 1. the autonomy gate -------------------------------------------

    def gate(self, *, tool: str, needed: list[str], outcome: str,
             question: str = "", error: str = "", extra: str = "") -> dict[str, Any] | None:
        """The operator was asked and here is what came back.

        `allowed` is False for every outcome but `confirmed` — including
        `no_operator`. The receipt has to be readable by someone asking "did
        this run", and the three refusals are told apart by `outcome`, not by
        pretending one of them ran.
        """
        assert outcome in GATE_OUTCOMES, outcome
        allowed = outcome == "confirmed"
        if allowed:
            reason = "operator confirmed %s" % tool
        elif outcome == "refused":
            reason = "operator refused %s" % tool
        elif outcome == "no_operator":
            reason = ("nobody was available to answer for %s, and silence is not "
                      "consent" % tool)
        else:
            reason = "the confirmer raised for %s: %s" % (tool, error or "?")
        payload = {
            "event_type": "gate_allow" if allowed else "gate_deny",
            "allowed": allowed,
            "dimension": "tool",
            "resource": tool,
            "rule": "autonomy.gate: %s" % outcome,
            "reason": reason,
            "agent": self.agent,
            "extra": extra or ", ".join(needed),
            "ts": time.time(),
            "gate_outcome": outcome,
            "freedoms": list(needed),
            "question": question[:600],
        }
        if not allowed and self.alert is not None:
            try:                                              # a notifier that
                self.alert(payload)                           # throws must not
            except Exception:                                 # eat the refusal
                pass
        return self._record(payload)

    # -- 2. the manifest ------------------------------------------------

    def declaration(self, manifest: Any, *, source: str = "",
                    run_id: str = "", refused: str = "") -> dict[str, Any] | None:
        """A run declared its envelope, and here is how it compared.

        `admit()` is what computes the breaches, so the receipt cannot disagree
        with the refusal: `rule` says `ceiling` when the declaration was judged
        against a limit, and `intent` when it was judged for saying nothing.
        A declaration that was never checked — `manifest.admit()` exists and
        nobody calls it — is the failure this receipt is meant to make visible,
        which is why the rule names the check rather than only the outcome.
        """
        try:
            breaches = list(manifest.ceiling_breaches())
            general = bool(getattr(manifest, "general_purpose", False))
            intent = str(getattr(manifest, "intent", "") or "")
        except Exception:                                     # noqa: BLE001
            breaches, general, intent = [], False, ""
        allowed = general or (not breaches and bool(intent.strip()))
        if refused:
            # The caller already enforced the rule and is about to raise. The
            # receipt is written first, and `allowed` is forced False rather than
            # recomputed, so a refusal can never be recorded as an admission by a
            # breach list that happens not to reproduce the caller's reason.
            allowed, breaches = False, breaches or [refused]
            rule = "ceiling: refused before any code ran"
            reason = refused[:300]
            return self._record({
                "event_type": "declaration_deny", "allowed": False,
                "dimension": "policy",
                "resource": (source or intent or "declaration")[:200],
                "rule": rule, "reason": reason, "agent": self.agent,
                "extra": json.dumps({k: getattr(manifest, k, None) for k in
                                     ("memory_mb", "max_processes", "cpu_seconds", "wall_s")},
                                    default=str),
                "ts": time.time(), "source": source, "run_id": run_id,
                "intent": intent[:300],
            })
        if general:
            rule = "ceiling: not applied (general_purpose)"
            reason = "run declares itself general-purpose; the ceiling is a forged tool's budget"
        elif breaches:
            rule = "ceiling: %s" % breaches[0]
            reason = "declaration refused: " + "; ".join(breaches)
        elif not intent.strip():
            rule = "intent: required"
            reason = ("declaration refused: no intent line, so there is nothing "
                      "to reconcile the run against afterwards")
        else:
            rule = "ceiling: within limits"
            reason = "declaration admitted: %s" % intent[:200]
        numbers = {k: getattr(manifest, k, None) for k in
                   ("memory_mb", "max_processes", "cpu_seconds", "wall_s")}
        return self._record({
            "event_type": "declaration_allow" if allowed else "declaration_deny",
            "allowed": allowed,
            "dimension": "policy",   # a declaration is judged about the agent, not a resource
            "resource": (source or intent or "declaration")[:200],
            "rule": rule,
            "reason": reason,
            "agent": self.agent,
            "extra": json.dumps(numbers, default=str),
            "ts": time.time(),
            "source": source,
            "run_id": run_id,
            "intent": intent[:300],
        })

    # -- 3. self-modification -------------------------------------------

    def amendment(self, amendment: Any, *, checked: int = 0,
                  reason: str = "") -> dict[str, Any] | None:
        """A change to the agent, accepted or refused, with what was checked.

        `checked` is the number of preconditions that ran before the verdict —
        rationale required, veto, no-op. It is on the receipt because "the
        change was rejected" and "nothing was consulted before rejecting it"
        are different states of the machinery, and the second one is how a
        guard rail quietly stops running.
        """
        accepted = bool(getattr(amendment, "accepted", False))
        target = str(getattr(amendment, "target", "?"))
        rejected = getattr(amendment, "rejected_reason", None) or reason
        payload = {
            "event_type": "selfmod_allow" if accepted else "selfmod_deny",
            "allowed": accepted,
            "dimension": "policy",
            "resource": target,
            "rule": ("selfmod.accepted" if accepted
                     else "selfmod.rejected: %s" % (rejected or "unspecified")),
            "reason": (getattr(amendment, "rationale", "") or "")[:400] if accepted
                      else "refused: %s" % (rejected or "unspecified"),
            "agent": self.agent,
            "extra": (getattr(amendment, "diff_summary", lambda: "")() or "")[:400],
            "ts": time.time(),
            "checked": int(checked),
        }
        return self._record(payload)

    # -- reporting -------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """What the control layer did, including what it failed to record."""
        with self._lock:
            return {
                "writing": self.writing,
                "counts": dict(sorted(self._counts.items())),
                "record_failures": len(self._failures),
                "failures": list(self._failures[-5:]),
                "agent": self.agent,
            }

    def failures(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._failures)


class _NullPlane(ControlPlane):
    """A plane that records nothing, for a registry or verifier built bare.

    Not the same as a plane with `conn=None`: this one does not even count, so
    a five-line test that constructs a `ToolRegistry()` gets no state it did not
    ask for. Presence of the object, not its contents, is what callers check.
    """

    def __init__(self) -> None:
        super().__init__(conn=None, agent="", enabled=False)

    def _record(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        return None

    def gate(self, **kwargs: Any) -> None:
        return None

    def declaration(self, manifest: Any, **kwargs: Any) -> None:
        return None

    def amendment(self, amendment: Any, **kwargs: Any) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        return {"writing": False, "counts": {}, "record_failures": 0,
                "failures": [], "agent": ""}


null_plane = _NullPlane()

_PLANE: ControlPlane | None = None


def plane() -> ControlPlane:
    """The process-wide plane, so a bare call site still records somewhere."""
    global _PLANE
    if _PLANE is None:
        _PLANE = ControlPlane()
    return _PLANE


def bind(conn: Any, *, agent: str = "autoforge", alert: Any = None) -> ControlPlane:
    """Attach the process-wide plane to a ledger."""
    global _PLANE
    _PLANE = ControlPlane(conn=conn, agent=agent, alert=alert)
    return _PLANE
