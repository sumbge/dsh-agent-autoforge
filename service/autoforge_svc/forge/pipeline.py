"""The forge pipeline: the three-stage closed loop.

    FORGE ────────> VERIFY ────────> SEAL
    (generate)      (exec + trigger)   (register in lifecycle)

and the living half:

    OBSERVE ──────> JUDGE ─────────> ACT
    (ledger)        (degrade?)        (quarantine / rehab / retire)

Why a pipeline object rather than loose functions: the stages have policy —
how many tries before giving up, whether to promote on pass, whether to
re-verify on drift. Policy belongs somewhere inspectable, so it lives here
with a full audit log of every forge attempt.

The design bet (DESIGN.md §2): most frameworks stop at FORGE. The failures
that actually hurt show up in the living half — a tool that never fires, or
one that fired well and quietly rotted. autoforge closes both loops.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..autonomy.policy import AutonomyPolicy
from ..core.llm import LLMAborted, LLMClient
from ..tools.registry import ToolRegistry
from ..tools.spec import ToolSpec, ToolState
from .generator import GeneratedTool, TemplateGenerator, UnrecoverableGeneration
from .sandbox import Sandbox
from .verifier import ToolVerifier, VerificationReport


class ToolGenerator(Protocol):  # pragma: no cover - structural
    def generate(self, need: str, context: str = "") -> GeneratedTool: ...


@dataclass
class ForgeAttempt:
    need: str
    round: int
    generated: GeneratedTool | None = None
    report: VerificationReport | None = None
    error: str | None = None
    duration_ms: float = 0.0
    accepted: bool = False


@dataclass
class ForgeResult:
    need: str
    spec: ToolSpec | None
    attempts: list[ForgeAttempt] = field(default_factory=list)
    #: Ended early because the operator spoke. Distinct from a failed forge: the
    #: tool was not judged, so nothing about it is known -- which is why this
    #: must never be reported as "the forge failed".
    aborted: bool = False
    #: Set when the candidate was built and verified but its *name* was already
    #: taken. The tool did not fail; it was refused, and the difference decides
    #: what the caller says. Without this the caller read `ok=False` and reported
    #: the need as unservable, when the truth was "a tool beside this one already
    #: exists" -- the opposite advice.
    replace_conflict: str = ""

    @property
    def ok(self) -> bool:
        return self.spec is not None

    @property
    def rounds(self) -> int:
        return len(self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "need": self.need,
            "ok": self.ok,
            "rounds": self.rounds,
            "tool": self.spec.to_dict() if self.spec else None,
            "attempts": [
                {
                    "round": a.round,
                    "error": a.error,
                    "accepted": a.accepted,
                    "report": a.report.to_dict() if a.report else None,
                }
                for a in self.attempts
            ],
        }


@dataclass
class ForgeConfig:
    max_rounds: int = 3              # generate→verify retries (feeds failure back)
    promote_on_pass: bool = True     # ACTIVE on a clean verification
    sample_args: dict[str, Any] | None = None
    require_execution: bool = True
    require_trigger: bool = True
    require_negative: bool = True
    #: The claim gate: the tool is run more than once, on its own declared
    #: sample, and its answer must survive a repeat, a row permutation, and a
    #: shuffled-label control. On by default, because a tool that returns a
    #: number nobody checked is the failure this framework exists to prevent --
    #: `audit_predictive_claim_csv` passed every check it had on 2026-10-02
    #: while returning PASS on a label-leaked dataset.
    require_claim: bool = True


class ForgePipeline:
    def __init__(
        self,
        generator: ToolGenerator,
        verifier: ToolVerifier,
        registry: ToolRegistry,
        *,
        config: ForgeConfig | None = None,
        sandbox: Sandbox | None = None,
        policy: AutonomyPolicy | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.generator = generator
        self.verifier = verifier
        self.config = config or ForgeConfig()
        # `config.require_claim` is the switch the pipeline owns; the verifier
        # owns the switch it consults. Binding them here means a caller who
        # turned the gate off in ForgeConfig cannot still get a claim verdict
        # from a verifier built with the default -- and, the way it actually
        # bit, a caller who built the verifier *before* the pipeline (the CLI
        # does exactly that) cannot silently stay on the old battery. Explicit
        # argument wins over config when the verifier was built with one.
        if hasattr(verifier, "run_claim_check"):
            verifier.run_claim_check = bool(self.config.require_claim)
        self.registry = registry
        self.sandbox = sandbox or verifier.sandbox
        # None means "no policy attached": every gate below fails open, which
        # is what the forge pipeline did before the policy existed. Attaching
        # a policy is how a caller makes the verdicts binding.
        self.policy = policy
        self.on_event = on_event
        self.log: list[dict[str, Any]] = []

    def _may_promote(self) -> bool:
        """Whether a clean verification may seal a tool as ACTIVE.

        Turning may_promote_tools off does NOT stop the tool being forged or
        called — creation stays free and PROBATION tools are already in the
        model's context. It only withholds ACTIVE, the state that means
        "trusted without further evidence".
        """
        if not self.config.promote_on_pass:
            return False
        return self.policy is None or self.policy.may_promote_tools

    # -- the loop --------------------------------------------------------
    def forge(self, need: str, context: str = "",
              should_abort: Callable[[], bool] | None = None,
              replace: bool = False) -> ForgeResult:
        """Run the loop until a tool is sealed, the rounds run out, or the
        operator says something.

        `should_abort` is the operator's line reaching into the middle of a
        round. A forge is the longest thing this framework does -- a model call
        per round, then a stack of sandbox samples -- and without this the only
        chance to hear them was at a round boundary, minutes away. It is asked
        at each boundary *and* by the sandbox itself while a child is running,
        so a correction lands mid-sample rather than after it.
        """
        result = ForgeResult(need, None)
        feedback = ""
        existing = ", ".join(self.registry.names()) or "(none)"

        # Announce up front: round 1 starts with a model call that can take a
        # while, and without this the very first thing a user sees is silence.
        self._emit("forge_start", {
            "need": need, "max_rounds": self.config.max_rounds,
        })

        # The sandbox outlives this call -- the verifier owns it and hands the
        # same instance to the adversary and the fuzzer -- so the predicate is
        # restored rather than just dropped. A stale one here would let a long
        # gone operator abort somebody else's forge.
        previous_check = self.sandbox.abort_check
        self.sandbox.abort_check = should_abort
        # And to the model clients, for the same reason and with the same
        # restore. The sandbox only covers the tool under test: the calls that
        # write it, attack it and probe it are made by clients owned by the
        # verifier and the generator, and each of those is a place a round can
        # spend two minutes deaf to the operator. `_model_clients` collects
        # them rather than listing them, so a stage added later is covered by
        # construction instead of by remembering.
        previous_model_checks = [
            (client, client.abort_check) for client in self._model_clients()
        ]
        for client, _ in previous_model_checks:
            client.abort_check = should_abort
        try:
            for round_no in range(1, self.config.max_rounds + 1):
                if should_abort is not None and should_abort():
                    result.aborted = True
                    break
                started = time.perf_counter()
                attempt = ForgeAttempt(need, round_no)
                # Set when retrying cannot possibly change the outcome, so the loop
                # stops after recording the attempt instead of spending another
                # round's budget to observe the same wall.
                futile = False
                try:
                    prompt_need = need if not feedback else self._repair_prompt(need, feedback)
                    generated = self.generator.generate(prompt_need, context or existing)
                    attempt.generated = generated
                    # Between the model call and the sandbox: the answer just
                    # arrived, and if the operator spoke during it, verifying it
                    # spends sandbox time proving a tool that is about to be
                    # replaced.
                    if should_abort is not None and should_abort():
                        result.aborted = True
                        break
                    spec = self._to_spec(generated)

                    # The name is known the moment the candidate exists, and
                    # this is the only point on the path where the duplicate
                    # can be refused *exactly* and *cheaply*. Exactly, because
                    # two names are compared rather than a request against a
                    # label; cheaply, because verification -- a sandbox run,
                    # a trigger probe, a negative probe -- has not happened
                    # yet. Replayed over the ledger's 117 accepted forges this
                    # is 8/8 duplicates caught and 0/109 legitimate forges
                    # blocked. The pre-forge lexical veto that used to sit in
                    # `Arbiter` was 7/8 and 95/109; see `route/arbiter.py` for
                    # that table. Order matters here, so it is asserted in
                    # `tests/test_forge_arbitration.py` rather than assumed.
                    taken = (not replace
                             and self.registry.get(spec.name) is not None)
                    if taken:
                        attempt.accepted = False
                        result.replace_conflict = (
                            "%r is already in this library" % spec.name)
                        attempt.error = (
                            "refused before verification: this library already "
                            "holds a tool named %r, and a forge is not a "
                            "deliberate replacement. Evolve it, or forge under "
                            "a name that says what is different." % spec.name)
                        self._emit("name_taken", {
                            "need": need, "name": spec.name,
                            "round": attempt.round,
                        })
                        futile = True
                        # Fall through to the attempt record at the bottom of
                        # the loop rather than `continue` or `break`. Both jump
                        # past that record, and a refusal with no
                        # `forge_attempt` row reads on the ledger as a forge
                        # that ran and produced nothing -- the opposite of what
                        # happened. `futile` ends the loop after the record.
                        report = None
                    else:
                        report = self.verifier.verify(spec, self.config.sample_args)
                    attempt.report = report
                    if report is not None:
                        spec.verification = report.to_dict()

                    if report is not None and report.passed:
                        if self._may_promote():
                            spec.state = ToolState.ACTIVE
                        else:
                            # PROBATION tools are already visible to the model, so
                            # nothing is lost but the seal. Record why, so a tool
                            # sitting on probation is never a mystery.
                            spec.state = ToolState.PROBATION
                            self._emit("promote_withheld", {
                                "tool": spec.name,
                                "reason": (
                                    "may_promote_tools is off"
                                    if self.policy is not None and not self.policy.may_promote_tools
                                    else "promote_on_pass is off"
                                ),
                            })
                        # `replace=False`: a forge is not a deliberate
                        # replacement. This call used to inherit
                        # `register(replace=True)`, so a candidate that happened
                        # to be named like a tool already in the library
                        # overwrote it -- no refusal, no mention in the reply.
                        # The ledger shows 8 accepted forges landing on a name
                        # already taken, and `chrome_cdp_drive` reaching version
                        # 3 that way. Evolve is the path that replaces on
                        # purpose; it judges the old against the new first.
                        try:
                            self.registry.register(spec, replace=replace)
                        except ValueError as exc:
                            attempt.accepted = False
                            # Not a failed round: the tool was built and
                            # verified. Re-asking the model to "repair" an
                            # already-working tool would spend the rest of the
                            # budget on the wrong problem, so this stops the
                            # loop and hands the fact to the caller.
                            futile = True
                            result.replace_conflict = str(exc)
                            attempt.error = (
                                "refused: %s -- the tool was verified, but this "
                                "library already holds that name. Evolve it, or "
                                "forge under a name that says what is different."
                                % exc)
                        else:
                            attempt.accepted = True
                            result.spec = spec
                    elif report is not None:
                        spec.state = ToolState.DRAFT
                        feedback = self._feedback(report)
                    # `report is None` is the name-taken refusal: there is no
                    # verdict to feed back and no repair to ask for, because
                    # nothing about the candidate was judged -- its name was
                    # taken before verification ever ran. `futile` already
                    # ended the loop; `feedback` is left untouched.
                except LLMAborted:
                    # The operator's line, arriving inside a model call. Not a
                    # failure and not a round: nothing about the candidate was
                    # judged, so there is no verdict to record and no error to
                    # feed the next round's prompt -- retrying here would ask
                    # the model to repair a tool that was never found wanting,
                    # which is how one interruption turned into a second run of
                    # the same forge. The attempt is not appended either, so
                    # `result.rounds` counts rounds that produced something.
                    result.aborted = True
                    break
                except UnrecoverableGeneration as exc:
                    # Nothing about this failure is round-specific: the model, not
                    # the attempt, cannot produce an answer. Record it and stop.
                    futile = True
                    attempt.error = f"{type(exc).__name__}: {exc}"
                    self._emit("forge_error", {
                        "round": round_no,
                        "error": attempt.error,
                        "futile": True,
                        "traceback": traceback.format_exc(),
                    })
                except Exception as exc:  # noqa: BLE001
                    attempt.error = f"{type(exc).__name__}: {exc}"
                    # A framework whose whole point is judging generated code cannot
                    # afford to swallow its own tracebacks. Keep them in the log.
                    self._emit("forge_error", {
                        "round": round_no,
                        "error": attempt.error,
                        "traceback": traceback.format_exc(),
                    })
                    feedback = attempt.error

                attempt.duration_ms = (time.perf_counter() - started) * 1000
                result.attempts.append(attempt)
                self._emit("forge_attempt", {
                    "need": need, "round": round_no, "accepted": attempt.accepted,
                    "error": attempt.error,
                    # How long the round took. A reader cannot tell "the need is
                    # hard" from "one round took eleven minutes" without it.
                    "duration_ms": round(attempt.duration_ms, 1),
                    "report": attempt.report.to_dict() if attempt.report else None,
                })
                if attempt.accepted or futile:
                    break
        finally:
            self.sandbox.abort_check = previous_check
            for client, previous in previous_model_checks:
                client.abort_check = previous

        if result.aborted:
            # Not `forge_done` with ok=False. Nothing was judged, so there is no
            # verdict to report -- and a caller that reads a failed verdict here
            # would record a tool as bad because a human interrupted.
            self._emit("forge_aborted", {
                "need": need, "rounds": result.rounds,
                "reason": "the operator said something",
            })
            return result

        self._emit("forge_done", {
            "need": need, "ok": result.ok, "rounds": result.rounds,
            "name": getattr(result.spec, "name", None),
            # The claim verdict on the terminal event, where a reader looks for
            # "what happened to this forge". `forge_attempt` carries the whole
            # report, but the attempt is per-round and the claim verdict is the
            # one line somebody actually wants: did the tool that came out of
            # this measure anything, or does it only look like it does.
            "claim": (self.verifier.claim_report.to_dict()
                      if getattr(self.verifier, "claim_report", None) else None),
            **({"replace_conflict": result.replace_conflict}
               if result.replace_conflict else {}),
        })
        return result

    # -- helpers ---------------------------------------------------------
    def _model_clients(self) -> list[LLMClient]:
        """Every model client this pipeline can reach, deduplicated.

        Asked of the parts rather than handed in: a stage that holds a client
        is a stage that makes calls, and a stage added later will be collected
        here without anyone updating a list. The verifier's probe agents and
        the adversary's attacker are built from `verifier.llm`, so the same
        instance covers them.
        """
        found: list[LLMClient] = []
        owners = (self.generator, self.verifier, getattr(self.verifier, "adversary", None))
        for owner in owners:
            client = getattr(owner, "llm", None)
            if client is not None and all(client is not seen for seen in found):
                found.append(client)
        return found

    def _to_spec(self, g: GeneratedTool) -> ToolSpec:
        """Wrap a generated tool as a spec whose execution goes through the sandbox.

        The `fn` is a tripwire: any in-process call is a bug, so it raises
        loudly rather than silently running untrusted code in the agent loop.
        The `runner` is the real path — it shells out to the sandbox.
        """
        sandbox = self.sandbox

        def _bridge(name: str, args: dict[str, Any]) -> Any:
            res = sandbox.run(g.code, g.entry, args)
            if res.timed_out:
                raise TimeoutError(f"tool {name} timed out after {sandbox.timeout}s")
            if not res.ok:
                raise RuntimeError(res.error or "sandbox failure")
            return res.output if res.output is not None else (res.stdout or "")

        def _tripwire(**_: Any) -> Any:
            raise RuntimeError(
                "generated tool attempted in-process execution; use the sandbox runner"
            )

        return ToolSpec(
            name=g.name,
            description=g.description,
            parameters=g.parameters,
            fn=_tripwire,
            runner=_bridge,
            code=g.code,
            source="generated",
            probes=g.probes,
            effect_signature=g.effect_signature,
            # Both of these were dropped here until 2026-10-02, and dropping
            # them made the whole positive-case apparatus inert on the live
            # forge path. `check_execution` reads `spec.sample_call` and does
            # nothing with `g.sample_call`; the generator was required by its
            # own prompt to emit one, the field existed, the check consulted it
            # -- and the translation between them was missing. Measured: every
            # one of the 141 tools on this machine's ledger was verified with no
            # positive example at all, which is exactly the vacuity the
            # `check_execution` docstring says was fixed. It was fixed in the
            # check and never connected to the pipeline. Found by the claim gate,
            # which reported "un-probed" on a tool whose generator had supplied
            # a perfectly good sample.
            sample_call=dict(g.sample_call or {}),
            sample_expect=g.sample_expect or "",
            tags=g.tags,
            state=ToolState.DRAFT,
        )

    @staticmethod
    def _feedback(report: VerificationReport) -> str:
        lines = [f"- {c.name}: {c.detail}" for c in report.failed]
        # A claim-gate failure is a different kind of message from "it crashed".
        # The tool will not be fixed by making it more robust; it will be fixed
        # by connecting the answer to the inputs, or by not claiming to measure
        # something it does not measure. Saying so here is the difference between
        # three rounds of hardening a broken instrument and one round of repair.
        for check in report.failed:
            if check.name != "claim":
                continue
            findings = (check.evidence or {}).get("findings") or []
            for finding in findings:
                if finding.get("ok"):
                    continue
                lines.append(f"  control {finding.get('name')}: {finding.get('detail')}")
            lines.append(
                "  The battery above asks whether the code runs; this control asks "
                "whether its answer depends on its inputs. Fix the dependency, or "
                "stop returning a number you did not derive.")
        return "Failed checks:\n" + "\n".join(lines)

    @staticmethod
    def _repair_prompt(original: str, feedback: str) -> str:
        return (
            f"{original}\n\n"
            f"Your previous tool did not pass verification:\n{feedback}\n"
            "Fix the root cause. Keep the same behaviour, make it robust."
        )

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        entry = {"t": time.time(), "kind": kind, **payload}
        self.log.append(entry)
        if self.on_event:
            self.on_event(kind, payload)

    # -- the living half -------------------------------------------------
    def judge(self, name: str) -> dict[str, Any]:
        """Inspect a tool's ledger and act if it has degraded."""
        spec = self.registry.get(name)
        if spec is None:
            return {"tool": name, "action": "unknown"}
        st = spec.stats
        before = spec.state
        if spec.state in (ToolState.ACTIVE, ToolState.PROBATION):
            self.registry._maybe_quarantine(spec)
        action = "none"
        if spec.state != before:
            action = f"{before.value}->{spec.state.value}"
        return {
            "tool": name,
            "action": action,
            "state": spec.state.value,
            "success_rate": round(st.success_rate, 3),
            "calls": st.calls,
        }


__all__ = [
    "ForgePipeline",
    "ForgeConfig",
    "ForgeResult",
    "ForgeAttempt",
    "TemplateGenerator",
    "Sandbox",
    "LLMClient",
]
