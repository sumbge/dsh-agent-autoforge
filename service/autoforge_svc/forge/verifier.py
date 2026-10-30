"""Verification: the gate between "forged" and "trusted".

Five orthogonal checks, in order of increasing cost:

  A. execution     鈥?does the code run on a standard input without crashing?
  B. robustness    鈥?does it survive edge-case inputs? (deterministic fuzzer)
  C. adversarial   鈥?can it survive an LLM attacker? (adversarial gate)
  D. trigger       鈥?when the need arises, will the agent call this tool?
  E. negative      鈥?when the need is absent, will it stay quiet?
  F. claim         鈥?is the number it returns a fact about the world, or a fact
                     about the data it was handed? (reproducibility, order
                     invariance, and a known-answer control on its own labels)

Question D+E is the "Constraint Tax" and "over-triggering" problems. Question B
guards against the exact bug found in the live demo (ISBN prefix contamination).
Question C is unique to autoforge 鈥?no framework runs an attacker against its
own tools before trusting them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..core.agent import Agent
from ..core.llm import LLMClient
from ..tools.registry import ToolRegistry
from ..tools.spec import ToolSpec, ToolState, normalise_parameters
from .adversary import AdversarialGate, AdversarialReport
from .fuzzer import RobustnessResult, run_robustness_checks
from .sandbox import Sandbox
from .manifest import (apply_declaration, intent_for, reconcile,
                       CapabilityManifest, ManifestRefused)
from .claimgate import ClaimReport, check_claim


def _plane():
    """The process control plane, if the control layer is importable.

    Guarded because `forge` is used by tools that do not need it, and a missing
    recorder is not a reason to fail a verification battery. The plane, when it
    exists, is inert until a ledger is bound to it 鈥?see autonomy/controls.py.
    """
    try:
        from ..autonomy.controls import plane

        return plane()
    except Exception:                                        # noqa: BLE001
        from ..autonomy.controls import null_plane

        return null_plane


@dataclass
class CheckResult:
    """One check's verdict, in THREE states rather than two.

    `passed` alone cannot express "this check had nothing to judge", and
    collapsing that case into `passed=True` is how a whole battery goes
    vacuous: six green ticks, none of which ever compared an answer to a
    question. `claimgate` already learned this -- it carries `applicable` and
    says "not applicable (no rail declared)" instead of silently passing --
    and this is the same idea applied to every check.

    The distinction the three states make:

      passed=True,  unprobed=False   measured, and it held
      passed=False, unprobed=False   measured, and it failed
      passed=?,     unprobed=True    nothing was measured; this is not a pass

    `unprobed=True` still sets `passed=False` on purpose. A report is a claim
    that the tool was tested; a check that could not test is not evidence for
    that claim, so the tool must not be sealed on the strength of it. The
    reason is carried in `detail` and `evidence["unprobed"]`, so a caller can
    tell "we looked and it was wrong" from "we never looked".
    """

    name: str
    passed: bool
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    #: True when the check had nothing to judge. Never true together with
    #: `passed=True`: see the class docstring for why that pairing is refused.
    unprobed: bool = False
    #: True when the check ran, found a defect, and the defect is in the TEST
    #: rather than the tool -- a negative probe about the same subject as the
    #: positive one, say. A waiver is not a pass: it is excluded from the
    #: aggregate verdict, and the reason travels in `detail` so the gap stays
    #: visible instead of being absorbed.
    waived: bool = False

    @property
    def judged(self) -> bool:
        """Whether this check actually measured anything."""
        return not self.unprobed and not self.waived

    def __post_init__(self) -> None:
        # The one invariant worth enforcing in code rather than in a comment:
        # a check that did not run must not report success. Making this raise
        # means a future check cannot quietly reintroduce the vacuity by
        # forgetting to pass `unprobed`.
        if self.waived and self.passed:
            raise ValueError(
                f"check {self.name!r} cannot be both waived and passed: a"
                f" waived check found a defect in its own setup, so it is"
                f" neither evidence for nor against the tool")
        if self.unprobed and self.passed:
            raise ValueError(
                f"check {self.name!r} cannot be both unprobed and passed: "
                f"nothing was measured, so there is nothing to pass on"
            )


@dataclass
class VerificationReport:
    tool: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    #: The manifest the run was admitted under, and what was measured against
    #: it. A verdict of "passed" says the checks went green; this says what
    #: the tool was allowed to be while it did.
    declaration: CapabilityManifest | None = None
    reconciliation: dict[str, Any] | None = None

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def waived(self) -> list[CheckResult]:
        """Checks whose own setup was defective."""
        return [c for c in self.checks if c.waived]

    @property
    def unprobed(self) -> list[CheckResult]:
        """Checks that had nothing to judge.

        Kept separate from `failed` because the two ask for different work: a
        failed check wants the tool repaired, an unprobed one wants the *test*
        repaired -- a sample call, a rail, a probe. Reporting them together is
        how "we never looked" gets read as "we looked and it was wrong".
        """
        return [c for c in self.checks if c.unprobed]

    def summary(self) -> str:
        ok = sum(1 for c in self.checks if c.passed)
        line = f"{self.tool}: {ok}/{len(self.checks)} checks passed"
        blank = self.unprobed
        if blank:
            # Named rather than folded into the ratio. "5/6 passed" invites the
            # reader to assume the sixth failed; it did not run at all.
            line += (f"; {len(blank)} unprobed "
                     f"({', '.join(c.name for c in blank)})")
        waived = self.waived
        if waived:
            # Named apart from `unprobed`, because the work they ask for differs:
            # an unprobed check wants the tool's sample call or its rail, a
            # waived one wants a better probe. Same empty slot in the ratio,
            # different next step.
            line += (f"; {len(waived)} waived "
                     f"({', '.join(c.name for c in waived)})")
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "passed": self.passed,
            "unprobed": [c.name for c in self.unprobed],
            "checks": [
                {"name": c.name, "passed": c.passed, "unprobed": c.unprobed,
                 "detail": c.detail, "evidence": c.evidence}
                for c in self.checks
            ],
            "declaration": self.declaration.to_dict() if self.declaration else None,
            "reconciliation": self.reconciliation,
        }


def _strip_quoted(text: str) -> str:
    """The text outside any quoted span.

    A probe's payload is its argument; the frame is what surrounds it. Keeping
    the payload in the shape comparison made two identical requests look
    different purely because they asked about different strings, which is the
    one thing the shape is meant to ignore.
    """
    out: list[str] = []
    depth = 0
    for ch in text:
        if ch in "'\"":
            depth = 0 if depth else 1
            out.append(" ")
            continue
        out.append(" " if depth else ch)
    return "".join(out)


class ToolVerifier:
    """Runs the full check battery against a freshly forged tool.

    Checks (in order of increasing cost):
      A. execution 鈥?does the code run without crashing on a standard input?
      B. robustness 鈥?does it survive edge-case inputs? (deterministic fuzzer)
      C. adversarial 鈥?can it survive an LLM attacker?
      D. trigger 鈥?does the agent call it when appropriate?
      E. negative 鈥?does it stay quiet when inappropriate?
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        sandbox: Sandbox | None = None,
        run_execution_check: bool = True,
        run_robustness_check: bool = True,
        run_adversarial_check: bool = True,
        run_trigger_check: bool = True,
        run_negative_check: bool = True,
        run_claim_check: bool = True,
        #: How many draws the positive probe gets. One was too few for the
        #: same reason negative_trials was: measured 2026-10-26, square2
        #: reported 'fired 0/1' on two runs and passed on the next two, so a
        #: working tool alternated between ACTIVE and DRAFT.
        trigger_trials: int = 3,
        #: How many draws the negative probe gets. Three is the smallest odd
        #: number that can express a majority, and it is enough to tell a
        #: coin-flip boundary from a consistent one: measured 5/8 quiet on a
        #: near-miss probe becomes 2-1 or 3-0 rather than a single lucky draw.
        #: Raising it costs one model call per draw, so this stays at the
        #: smallest number that works rather than at the largest affordable.
        negative_trials: int = 3,
        adversary: AdversarialGate | None = None,
        require_robustness_rate: float = 0.8,
    ) -> None:
        self.llm = llm
        self.sandbox = sandbox or Sandbox()
        #: Built at construction, before a line of any tool runs, so the
        #: declaration cannot be written afterwards to match what happened.
        self.declaration: CapabilityManifest = intent_for(name=None)
        #: Peak bytes / CPU / processes the kernel charged the last run, filled
        #: by check_execution and reconciled into the report.
        self.measurements: dict[str, Any] = {}
        #: Filled by the claim gate; None when it did not run.
        self.claim_report: ClaimReport | None = None
        self.run_execution_check = run_execution_check
        self.run_robustness_check = run_robustness_check
        self.run_adversarial_check = run_adversarial_check
        self.run_trigger_check = run_trigger_check
        self.run_negative_check = run_negative_check
        self.run_claim_check = run_claim_check
        self.trigger_trials = trigger_trials
        self.negative_trials = negative_trials
        self.adversary = adversary
        self.require_robustness_rate = require_robustness_rate

    # -- individual checks (A鈥揈) ----------------------------------------
    @staticmethod
    def _infer_args(spec: ToolSpec) -> dict[str, Any]:
        """Generate plausible sample arguments from the parameter schema."""
        props = spec.parameters.get("properties") or {}
        args = {}
        for name, schema in props.items():
            t = schema.get("type", "string")
            if t == "string":
                args[name] = "978-0-306-40615-7"
            elif t in ("number", "integer"):
                args[name] = 1
            elif t == "boolean":
                args[name] = True
            elif t == "object":
                args[name] = {}
            elif t == "array":
                args[name] = []
            else:
                args[name] = "test"
        return args

    def check_execution(self, spec: ToolSpec, sample_args: dict[str, Any] | None = None) -> CheckResult:
        """A. Run it out-of-process on a *valid* call, and on schema-shaped args.

        This check used to pass on `result.ok` alone, and that made the whole
        battery vacuous. `_infer_args` invents arguments from the schema -- for
        a string parameter it always supplied the literal "978-0-306-40615-7",
        an ISBN left over from the tool this framework was first built for. The
        generator is separately instructed to answer "INVALID: <reason>" rather
        than raise. Put together: every forged tool was handed input it could
        not process, answered "INVALID", exited 0, and was marked verified.

        The consequence was not a missed bug, it was an inverted incentive. A
        tool that did nothing scored highest: it survived every fuzz probe by
        refusing all of them, and it passed here by exiting cleanly. Meanwhile
        a tool that tried to do the job and failed on a real input looked
        *worse*. The agent then took "verified" at its word, blamed the
        environment when the tool failed in use, and spent ninety minutes
        diagnosing a network that was never broken.

        So the positive case is now the primary one. When the generator
        declared a `sample_call`, the tool is run on real arguments and the
        answer must be an answer -- not a refusal, not None. An unparseable or
        absent sample is reported as un-probed rather than quietly passing,
        because "we could not test it" and "it works" must not print the same.
        """
        args = dict(sample_args) if sample_args else dict(spec.sample_call or {})

        if args:
            result = self.sandbox.run(spec.code, spec.name, args)
            self.measurements = dict(getattr(result, "accounting", None) or {})
            evidence = {
                "sample_call": args,
                "output": str(result.output)[:300],
                "duration_ms": round(result.duration_ms, 1),
            }
            if result.timed_out:
                return CheckResult(
                    "execution", False,
                    f"timed out on the valid sample call after {self.sandbox.timeout}s",
                    evidence,
                )
            if not result.ok:
                return CheckResult(
                    "execution", False,
                    f"failed on the valid sample call: {result.error or 'no result'}",
                    evidence,
                )
            output = "" if result.output is None else str(result.output).strip()
            if not output:
                return CheckResult(
                    "execution", False,
                    f"produced nothing for the valid sample call {args!r}; a tool "
                    f"that returns None or only prints has not been shown to work",
                    evidence,
                )
            # The generator is told to signal "cannot process this" with an
            # INVALID: prefix. Getting one back for a call the generator itself
            # certified as valid means the tool does not do the job it was
            # written for -- exactly the case that used to pass.
            if output.upper().startswith("INVALID"):
                return CheckResult(
                    "execution", False,
                    f"rejected its own valid sample call {args!r}: {output[:200]}",
                    evidence,
                )
            if spec.sample_expect and spec.sample_expect not in output:
                return CheckResult(
                    "execution", False,
                    f"answer for {args!r} does not contain {spec.sample_expect!r}: "
                    f"{output[:200]}",
                    evidence,
                )
            # Only now, having been right about something, does the tool earn a
            # look at whether it is also total. The synthetic probe is run too,
            # and reported only as a warning: failing on invented garbage is a
            # robustness problem, not evidence that the tool is wrong.
            probe = self._infer_args(spec)
            probe_result = self.sandbox.run(spec.code, spec.name, probe) if probe else None
            probe_note = ""
            if probe_result is not None and not probe_result.ok:
                probe_note = (f"; warning: raised on synthetic probe "
                              f"{probe!r}: {probe_result.error}")
            return CheckResult(
                "execution", True,
                f"answered {args!r} with {output[:80]!r}{probe_note}",
                evidence,
            )

        # No valid example: this check had nothing to judge.
        #
        # The comment here used to say "An un-probed tool must not read as a
        # pass" and then return `result.ok` -- which IS a pass whenever the code
        # runs. Measured consequence: a tool that answered 212F as 180C (correct
        # is 100C) and declared no sample was sealed ACTIVE in 4 of 6 runs, the
        # only thing stopping it being an LLM attacker that happened to guess the
        # boundary. The prose was right; the return value was not.
        #
        # Now the `unprobed` flag carries that judgement in the data, and
        # `passed=False` follows from it -- see `CheckResult`. The code is still
        # run, because "it does not even execute" is a different and cheaper
        # finding worth reporting first.
        probe = self._infer_args(spec)
        result = self.sandbox.run(spec.code, spec.name, probe)
        if result.timed_out:
            return CheckResult("execution", False, "timed out",
                               {"timeout": self.sandbox.timeout, "probe": probe},
                               unprobed=True)
        if not result.ok:
            return CheckResult("execution", False, result.error or "failed",
                               {"probe": probe}, unprobed=True)
        return CheckResult(
            "execution", False,
            "UNPROBED: no valid sample call was declared, so correctness was "
            "never examined; all this shows is that the code runs. A tool that "
            "answers every input with the same constant passes this too. Declare "
            "sample_call (a real input and the answer it should produce) and run "
            "evaluate_tool again.",
            {"output": str(result.output)[:200], "probe": probe,
             "duration_ms": round(result.duration_ms, 1), "unprobed": True},
            unprobed=True,
        )

    def check_robustness(self, spec: ToolSpec) -> CheckResult:
        """B. Deterministic edge-case fuzzing 鈥?survives the ISBN prefix bug?"""
        result = run_robustness_checks(
            spec, sandbox=self.sandbox, require_survival_rate=self.require_robustness_rate,
        )
        # Name the probes that failed. Without this the retry only sees "0/19"
        # and cannot learn which inputs broke it, so the feedback loop is inert.
        detail = result.summary()
        if result.failures:
            shown = "; ".join(
                f"{f.get('label', '?')} -> {str(f.get('error', ''))[:60]}"
                for f in result.failures[:5]
            )
            detail += f". Failing probes: {shown}"
        return CheckResult(
            "robustness", result.passed,
            detail,
            {"survival_rate": round(result.survival_rate, 3),
             "survived": result.survived, "total": result.total,
             "invariance": (result.invariance.to_dict()
                            if result.invariance is not None else None)},
        )

    def check_adversarial(self, spec: ToolSpec, sample_args: dict[str, Any] | None = None) -> CheckResult:
        """C. An LLM attacker tries to break the tool."""
        gate = self.adversary or AdversarialGate(self.llm, execution_sandbox=self.sandbox)
        report = gate.attack(spec)
        return CheckResult(
            "adversarial", report.passed,
            report.summary(),
            {"survived": report.survived, "total": report.total_attacks},
        )

    def check_trigger(self, spec: ToolSpec, query: str) -> CheckResult:
        """D. Register the tool alone, ask the query, see if it fires."""
        registry = self._probe_registry()
        registry.register(spec)
        hits = 0
        seen: list[str] = []
        for _ in range(self.trigger_trials):
            # No terminate tool here: the probe measures whether THIS tool
            # fires, so the context must contain nothing else that competes.
            agent = Agent(self.llm, registry, max_turns=2, allow_self_terminate=False)
            res = agent.run(query)
            seen.extend(res.tool_calls)
            if spec.name in res.tool_calls:
                hits += 1
        # Any hit, and the asymmetry with `negative` is deliberate rather than
        # an oversight. The two questions are not mirror images:
        #
        #   trigger  asks "CAN this tool be reached at all". One reach proves
        #            it; a tool that fires on 1 of 3 draws is a tool the model
        #            will use, and the other two draws say something about the
        #            model's phrasing, not about the tool's reachability.
        #   negative asks "does it stay OUT of unrelated work". That one needs
        #            a majority, because a single quiet draw proves nothing
        #            about a boundary the model crosses at random.
        #
        # Using the same rule for both was the mistake measured on 2026-10-26:
        # `factorial` reported `fired 1/3` and failed -- sending a working tool
        # back to DRAFT because the model happened to phrase the probe
        # differently twice. Reachability is existential; restraint is not.
        passed = hits > 0
        return CheckResult(
            "trigger", passed,
            f"fired {hits}/{self.trigger_trials} on the positive probe",
            {"query": query, "tool_calls": seen},
        )

    #: Overlap at or above this is a probe about the same subject, not an
    #: unrelated request. Chosen from the library as it actually stands
    #: (measured 2026-10-26, every `negatives[0]` the battery uses):
    #:
    #:   ov=5  count_vowels   "How many consonants are in the word 'banana'"
    #:   ov=5  celsius_..._formula  "Convert 25 degrees Fahrenheit to Celsius"
    #:   ov=4  string_length  "Reverse the word 'banana'"
    #:   ov=4  kilometres_to_miles  "Convert 10 miles to kilometres"
    #:   ov=3  word_count     "Count the number of vowels in 'hello world'"
    #:   ov=1  factorial      "What is the square root of 5?"
    #:   ov=0  sum_numbers_in_text  (a Chinese sentence, no shared tokens)
    #:
    #: The two populations do not overlap, so 3 is the smallest value that
    #: catches every measured near miss without touching a real one. It cannot
    #: be lower: 1 and 2 are the probes that behave.
    NEAR_PROBE_OVERLAP = 3

    def check_negative(self, spec: ToolSpec, query: str,
                       trials: int = 0) -> CheckResult:
        """E. Ask an unrelated-but-adjacent query; the tool must stay quiet.

        Sampled more than once, because one draw is not a measurement. Measured
        on 2026-10-26 with `is_palindrome` and its own negative probe ("What is
        the string length of 'hello'?"): eight runs, five quiet and three that
        called the tool. A single-trial check therefore reported PASS or FAIL for
        the same tool, same code, same library -- and the pipeline seals on
        `report.passed`, so a good tool was sent back to DRAFT roughly one time
        in eight for a coin flip.

        Two things are worth separating here, and only the first is this check's:

          1. Sampling noise. The model is not deterministic, so a boundary this
             close cannot be judged from one observation. Fixed by trials and a
             majority verdict: `passed` is "most runs stayed quiet", and the
             exact counts travel in the evidence so a caller can see a 3-2 split
             for what it is rather than seeing "stayed quiet".

          2. A badly-chosen probe. "What is the string length of 'hello'?" is
             one word away from the positive probe for `is_palindrome`, so
             calling the tool is a reasonable reading and the tool is not at
             fault. That is a defect in the *generated probe*, not in the tool,
             and it is reported as a warning rather than by failing the battery
             -- the same split `check_execution` makes for the synthetic probe.
             Failing a tool for a near-miss probe is how a good library loses
             tools to its own test suite.

        The distinction is drawn from the two queries' words, which is a proxy
        and is documented as one: it can miss a paraphrase and it can flag an
        incidental overlap. It decides only what the DETAIL says; the pass/fail
        line above is sampling-only.
        """
        runs = trials or self.negative_trials
        registry = self._probe_registry()
        registry.register(spec)
        fired = 0
        calls: list[str] = []
        for _ in range(runs):
            agent = Agent(self.llm, registry, max_turns=2, allow_self_terminate=False)
            res = agent.run(query)
            calls.extend(res.tool_calls)
            if spec.name in res.tool_calls:
                fired += 1
        quiet = runs - fired
        # Majority, not unanimity: one draw out of five is noise, three out of
        # five is a pattern. `>` rather than `>=` so a 2-2 split on four trials
        # does not pass by rounding.
        majority_quiet = quiet * 2 > runs

        # A probe that shares most of its content with the positive one cannot
        # fail the tool, because it is not testing "does this tool stay out of
        # unrelated work" -- it is testing "does the model phrase the same
        # request differently". Measured on 2026-10-26 with `word_count`:
        # `Count the number of vowels in 'hello world'` shares three content
        # words with `How many words are in the sentence ...`, and the model
        # called `word_count` in 2 of 3 runs. Swapping in any genuinely
        # unrelated probe (celsius, square root, reverse) made it 3/3 quiet.
        # The tool was fine; the probe was near.
        #
        # So a near probe unprobes the check rather than failing it. This is the
        # same call `claimgate` makes for a rail it cannot perturb, and it is
        # deliberately NOT a rescue for a genuinely failing tool: the threshold
        # is high (5), and the evidence keeps the raw counts so a 0/3-quiet run
        # is still visible as such even when it lands as `unprobed`.
        overlap = self._probe_overlap(spec, query)
        near_probe = overlap >= self.NEAR_PROBE_OVERLAP

        if near_probe:
            detail = (f"WAIVED: the probe shares {overlap} content word(s) with "
                      f"the positive one, so it is not an unrelated request -- "
                      f"the model called the tool in {fired}/{runs} runs, which "
                      f"reads as adjacency rather than over-firing. This does not "
                      f"count for or against the tool; give it a probe from a "
                      f"different subject to make this check mean something.")
            return CheckResult(
                "negative", False, detail,
                {"query": query, "runs": runs, "fired": fired, "quiet": quiet,
                 "tool_calls": calls, "probe_overlap": overlap, "waived": True},
                waived=True,
            )

        detail = f"stayed quiet in {quiet}/{runs} runs"
        if fired:
            detail += f"; called the tool in {fired}/{runs}"
        return CheckResult(
            "negative", majority_quiet, detail,
            {"query": query, "runs": runs, "fired": fired, "quiet": quiet,
             "tool_calls": calls, "probe_overlap": overlap},
        )

    @staticmethod
    def _probe_overlap(spec: ToolSpec, negative: str) -> int:
        """How much the negative probe shares with the positive one.

        Counts content words plus a penalty for sharing the same *quoted
        payload*, because the words alone miss the case that actually caused
        this: `Check if 'racecar' is a palindrome` vs `Reverse the characters in
        'racecar'` shares one content word (`characters` vs `palindrome` are
        different), yet both ask about the same literal string, so reaching for
        the tool is reasonable. Without the payload term that pair scored 1 and
        looked "far".

        This is a proxy and is used as one: it only decides what the DETAIL
        says, never the pass/fail line, so an overestimate costs a warning where
        a failure was deserved -- the cheaper of the two mistakes.
        """
        stop = {"a", "an", "the", "is", "of", "in", "to", "and", "or", "what",
                "how", "many", "much", "this", "that", "for", "with", "on",
                "it", "its", "s", "string", "word", "text", "does", "do",
                "read", "same", "check"}
        #: The nouns a counting/measuring request cycles through. They are
        #: the SLOT in "how many X are in Y", so two probes that differ only
        #: here are the same request and must not be treated as unrelated.
        COUNT_HEADS = {"words", "word", "characters", "character", "letters",
                       "letter", "vowels", "vowel", "consonants", "consonant",
                       "lines", "line", "digits", "digit", "spaces", "space",
                       "items", "item", "elements", "element", "entries"}

        def words(text: str) -> set[str]:
            return {w for w in "".join(
                c.lower() if c.isalnum() else " " for c in str(text)
            ).split() if w and w not in stop}

        def quoted(text: str) -> set[str]:
            """The literal payloads a probe asks about, if any.

            Bare numbers are excluded: `Compute 5 + 3` is not sharing a payload
            with `Compute 0!` just because both contain digits, and counting it
            as one pushed a genuinely distant probe to a near-miss score.
            """
            out: set[str] = set()
            for q in ("'", '"'):
                parts = str(text).split(q)
                # Odd indices are between a pair of quotes.
                for i in range(1, len(parts), 2):
                    piece = parts[i].strip().lower()
                    if piece and not piece.replace(" ", "").isdigit():
                        out.add(piece)
            return out

        def shape(text: str) -> set[str]:
            """The sentence frame with its subject removed.

            `How many words are in 'x'?` and `How many characters are in 'y'?`
            become the same skeleton, which is the point: the request is
            identical and only the thing being counted changed, so the model
            reaching for the counting tool is reading the sentence correctly.

            Built from the stopword-and-punctuation skeleton, collapsing runs of
            placeholders so the frame is compared rather than the vocabulary.
            """
            # The quoted payload is the ARGUMENT, not the frame: strip it
            # before reading the shape, or `How many words are in 'a b c'
            # and `How many characters are in 'x y'` look different when
            # they are the same request. Measured: without this strip the
            # pair scored 1 and the flake survived.
            stripped = _strip_quoted(str(text)).lower()
            tokens: list[str] = []
            for chunk in stripped.replace("'", " ").replace('"', " ").split():
                cleaned = "".join(c for c in chunk if c.isalnum())
                if not cleaned:
                    continue
                # Nouns in a "how many X" frame are the slot being
                # filled, not shared vocabulary: `How many WORDS are in
                # ...` and `How many CHARACTERS are in ...` are the same
                # request. The function word plus a count-shaped head is
                # what makes it a frame; anything after it is content.
                tokens.append("_" if cleaned in stop or cleaned in COUNT_HEADS else cleaned)
            # Collapse consecutive placeholders: "how many _ are in _" is one
            # frame whether the subject is one word or two.
            collapsed: list[str] = []
            for tok in tokens:
                if tok == "_" and collapsed and collapsed[-1] == "_":
                    continue
                collapsed.append(tok)
            return {" ".join(collapsed)} if collapsed else set()

        positive: set[str] = set()
        payload: set[str] = set()
        skeleton_positive: set[str] = set()
        for probe in spec.probes:
            if probe.query:
                positive |= words(probe.query)
                payload |= quoted(probe.query)
                skeleton_positive |= shape(probe.query)
        shared_words = len(positive & words(negative))
        # A shared payload counts for more than any single word: the tool was
        # asked about the same thing, which is adjacency, not over-firing.
        shared_payload = len(payload & {p.split()[0] for p in quoted(negative) if p.split()})
        # And a shared SHAPE counts for more still, because it is the case that
        # beat every word-level measure: `How many words are in 'the quick brown
        # fox jumps'?` versus `How many characters are in 'hello world'?` shares
        # one content word and scored 1 -- yet the model called `count_words` in
        # 3/3 runs, because the two sentences are the same request with one noun
        # swapped. Lexical overlap cannot see that; the frame can.
        shared_shape = len(skeleton_positive & shape(negative))
        return shared_words + (3 * shared_payload) + (3 * shared_shape)

    @staticmethod
    def _probe_registry() -> ToolRegistry:
        return ToolRegistry(
            auto_quarantine=False,
            visible_states={ToolState.DRAFT, ToolState.PROBATION, ToolState.ACTIVE},
        )

    # -- battery --------------------------------------------------------
    def verify(self, spec: ToolSpec, sample_args: dict[str, Any] | None = None) -> VerificationReport:
        # Settle the parameter shape before any check reads it. The generator
        # and the store already normalise at their trust boundaries; a spec
        # built by hand still reaches here raw, and every check below reads
        # `schema.get("type", ...)` per property. The judge establishes its own
        # preconditions rather than dying three frames deep.
        spec.parameters = normalise_parameters(spec.parameters)

        # Declare, then run under the declaration. The ordering is the point:
        # a manifest written after the fact is a description, and this one has
        # to be a constraint to be worth anything.
        self.declaration = intent_for(name=spec.name, code=spec.code,
                                      description=getattr(spec, "description", "") or "")
        # A declaration is admitted, or it is refused, and either way it is a row.
        # `admit()` has existed since the manifest was written and nothing ever
        # called it: the ceiling was computed and never asked, which is the
        # difference between a budget and a decoration.
        #
        # The refusal is enforced *and* recorded, in that order of authority: the
        # receipt is written first only because once `ManifestRefused` is raised
        # there is no later frame in which to write it. The raise is not softened
        # into a failed check 鈥?"refused before any code ran" is the promise the
        # module makes in its own error message, and a verifier that ran the code
        # anyway to be able to report on it would break exactly that promise.
        try:
            self.declaration.admit()
        except ManifestRefused as exc:
            _plane().declaration(self.declaration, source=spec.name, refused=str(exc))
            raise
        _plane().declaration(self.declaration, source=spec.name)
        self.sandbox = apply_declaration(self.sandbox, self.declaration)
        checks: list[CheckResult] = []

        if self.run_execution_check:
            checks.append(self.check_execution(spec, sample_args))

        if self.run_robustness_check and spec.code:
            checks.append(self.check_robustness(spec))

        if self.run_adversarial_check and spec.code:
            checks.append(self.check_adversarial(spec, sample_args))

        if self.run_trigger_check:
            positive = [p for p in spec.probes if p.expect == "call"]
            if not positive:
                checks.append(CheckResult(
                    "trigger", False,
                    "UNPROBED: no positive probe was declared, so nothing asked "
                    "this tool to fire; a tool that can never be reached passes "
                    "an empty test. Add a probe whose query should trigger it.",
                    {"unprobed": True}, unprobed=True))
            else:
                for probe in positive[:2]:
                    checks.append(self.check_trigger(spec, probe.query))

        if self.run_negative_check:
            negatives = [p.negative_query for p in spec.probes if p.negative_query]
            if negatives:
                checks.append(self.check_negative(spec, negatives[0]))

        # F. The claim gate. Last because it is the only check that has to
        # *run the tool more than once*, and first in importance because the
        # other five can all be green while the answer is wrong. A tool with no
        # declared rail is recorded as not-applicable rather than passed: the
        # battery must never let "we did not look" print as "it is fine".
        if self.run_claim_check and spec.code:
            claim = check_claim(spec, self.sandbox, sample_args=sample_args)
            self.claim_report = claim
            # `applicable=False` is a legitimate N/A, not a gap: this tool
            # declares no rail (no enum, no boolean, no prediction column), so
            # there is nothing whose influence could be tested. Treating that as
            # `unprobed` was wrong and it sealed nothing -- it failed every
            # correct single-argument tool, which is the opposite defect. The
            # three-state distinction is for checks that SHOULD have judged and
            # did not; a check with nothing in its remit stays a pass and says so
            # in its detail ("not applicable (no rail declared)").
            #
            # A rail that IS declared and cannot be exercised is a different
            # matter, and `claimgate` already answers it: it sets `applicable`
            # True and `passed` False with a `un_probed` finding, which lands
            # here as an ordinary failure.
            checks.append(CheckResult(
                "claim", bool(claim.passed), claim.summary(), claim.to_dict()))
        else:
            self.claim_report = None

        # The two absences are NOT the same, and conflating them was a real bug
        # caught by the sealing test on 2026-10-26:
        #
        #   unprobed  the check had something to judge and could not. That is a
        #             hole in the evidence, so the report cannot be a pass --
        #             a report is a claim that the tool was tested.
        #   waived    the check judged its own setup defective. It is neither
        #             evidence for nor against, so it is simply removed from the
        #             verdict rather than counted as a failure.
        #
        # The first version filtered on `judged` (which excludes both) and then
        # asked `all(c.passed for c in judged)`. With one unprobed `execution`
        # and one passing `claim`, `judged` was non-empty and every member
        # passed, so a tool with no sample call was sealed ACTIVE -- exactly the
        # defect the unprobed state exists to prevent, reintroduced by the fix
        # for a different one.
        counted = [c for c in checks if not c.waived]
        passed = bool(checks) and all(c.passed for c in counted)
        report = VerificationReport(spec.name, passed, checks)
        report.declaration = self.declaration
        # Declared against measured. A finding here does not fail the battery
        # -- see manifest.reconcile for why -- it goes into the record so that
        # a *pattern* of under-declaration is visible later.
        report.reconciliation = reconcile(self.declaration, None, self.measurements)
        return report


def register_if_verified(
    registry: ToolRegistry,
    spec: ToolSpec,
    verifier: ToolVerifier,
    *,
    sample_args: dict[str, Any] | None = None,
    promote_on_pass: bool = False,
) -> VerificationReport:
    """Verify, then place the tool in the lifecycle accordingly."""
    report = verifier.verify(spec, sample_args)
    spec.verification = report.to_dict()
    if report.passed:
        spec.state = ToolState.ACTIVE if promote_on_pass else ToolState.PROBATION
    else:
        spec.state = ToolState.DRAFT
    registry.register(spec)
    return report


__all__ = [
    "ToolVerifier",
    "VerificationReport",
    "CheckResult",
    "register_if_verified",
    "json",
]