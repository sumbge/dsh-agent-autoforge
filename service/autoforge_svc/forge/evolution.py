"""Tool evolution engine: mutation, crossover, and natural selection.

This is the mechanism that makes a tool library *alive*. Instead of just fixing
a broken tool, the framework spawns a population of mutant offspring, tests
all of them, and keeps the one that survives the verification gauntlet.

Core idea (inspired by genetic programming, grounded in LLM generation):

  RETRY is not REPAIR. When a tool fails, retrying the same generator prompt
  (even with failure feedback) usually produces near-identical output. Real
  improvement needs *exploration* — multiple divergent candidates that the
  verifier can select among.

The flow:
  1. A tool reaches its failure threshold (or the user requests improvement).
  2. `evolve()` spawns `population_size` mutants by perturbing the tool's
     code, probes, parameters, and description through an LLM.
  3. Each mutant runs through the verifier (execution + trigger + negative).
  4. The highest-scoring surviving mutant is kept.
  5. If the best mutant beats the original's fitness, it replaces the original.

`evolve_from_scratch()` is the crossover path: given two parent tools (or a
tool + a failure report), it creates hybrids that combine their capabilities.

The design bet: a population of 3-5 mutants, tested in parallel, costs about
the same as 3-5 sequential retries with no diversity guarantee — and usually
produces a working tool on the first batch.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.llm import LLMClient
from ..core.message import Message
from ..forge.generator import GeneratedTool, GENERATOR_SYSTEM, extract_json
from ..forge.validity import FrozenBaseline, ValidityGate, ValidityReport
from ..forge.verifier import CheckResult, ToolVerifier, VerificationReport
from ..tools.spec import ToolSpec, ToolState, TriggerProbe

# The check classes the battery is *supposed* to contain. A mutant that omits
# one of these does not get a smaller denominator — a missing class counts as
# failed. Without this, deleting your own guardrail probe raises your score.
REQUIRED_CHECK_CLASSES: tuple[str, ...] = (
    "execution",
    "robustness",
    "adversarial",
    "trigger",
    "negative",
)

# Probe-mass needed to earn the full evidence multiplier. Calibrated to the
# fuzzer's default 50-probe surface so a 6-probe "exam" cannot match a real
# one. Surviving MORE probes is better; the term is capped, not unbounded.
_REFERENCE_PROBE_MASS = 20.0

_MUTATE_SYSTEM = (
    "You are a tool breeder. You will be given a tool's current code and its\n"
    "verification failure report. Your job: generate a DIVERGENT variant that\n"
    "fixes the root cause. Do NOT reproduce the same approach — try a\n"
    "structurally different solution.\n\n"
    "Emit in the same JSON envelope as a normal tool generator:\n"
    "{name, description, code, entry, parameters, probes, effect_signature, tags, rationale}\n\n"
    "Rules:\n"
    "- Keep the same `name` as the original so it can replace it.\n"
    "- The code must be self-contained Python, stdlib only.\n"
    "- Include at least 2 probes. At least one must have a negative_query.\n"
    "- If the original failed because of a missing input guard, ADD that guard.\n"
    "- If the original failed because of a too-strict guard, LOOSEN it.\n"
    "- If the original timed out, optimise the algorithm.\n"
)

_CROSSOVER_SYSTEM = (
    "You are a tool breeder. You are given TWO parent tools that each solve\n"
    "part of a problem. Generate a hybrid that combines their capabilities\n"
    "into a single coherent tool.\n\n"
    "Emit in the JSON envelope format:\n"
    "{name, description, code, entry, parameters, probes, ...}\n\n"
    "Rules:\n"
    "- The hybrid must do what BOTH parents did, choosing a name that covers both.\n"
    "- The implementation must merge the logic cleanly, not shell out to them.\n"
    "- Code must be self-contained stdlib Python.\n"
    "- Include probes and negative_query for the combined capability.\n"
)

_ANALYSE_SYSTEM = (
    "You are a tool forensics analyst. Given a tool's code and the error it\n"
    "produced, identify the ROOT CAUSE (one sentence) and what must change.\n"
    "Output STRICT JSON only:\n"
    '{"root_cause": "...", "required_change": "...", "change_type": "guard|loosen|logic|perf"}'
)

_population_prompt = """Original tool: {name}
Code:
```python
{code}
```

Failure report:
{failures}

Generate {n} DIFFERENT approaches to fixing this tool.
Number each approach FIX-1 through FIX-{n}.
Each must be structurally different from the others.
Wrap each in the standard JSON envelope."""


@dataclass
class Mutant:
    generated: GeneratedTool
    report: VerificationReport | None = None
    fitness: float = 0.0
    duration_ms: float = 0.0
    vector: tuple[float, float, float] = (0.0, 0.0, 0.0)
    validity: ValidityReport | None = None
    rejected_by: str = ""            # which gate vetoed it, if any

    @property
    def admissible(self) -> bool:
        return self.rejected_by == "" and (self.validity is None or self.validity.admissible)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitness": round(self.fitness, 3),
            "vector": list(self.vector),
            "passed": self.report.passed if self.report else False,
            "admissible": self.admissible,
            "rejected_by": self.rejected_by,
            "root_cause": "",
        }


@dataclass
class EvolutionResult:
    tool: str
    original_version: str
    best_mutant: Mutant | None = None
    mutants: list[Mutant] = field(default_factory=list)
    kept_existing: bool = False
    rounds: int = 0

    @property
    def improved(self) -> bool:
        return self.best_mutant is not None and self.best_mutant.report is not None and self.best_mutant.report.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "improved": self.improved,
            "kept_existing": self.kept_existing,
            "rounds": self.rounds,
            "mutants_tried": len(self.mutants),
        }


# ---------------------------------------------------------------------------
class EvolutionEngine:
    """Breed better tools through mutation, crossover, and selection."""

    def __init__(
        self,
        llm: LLMClient,
        verifier: ToolVerifier,
        *,
        population_size: int = 3,
        parallel_verification: bool = True,
        mutate_model: str | None = None,
        validity_gate: ValidityGate | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.llm = llm
        self.verifier = verifier
        self.population_size = population_size
        self.parallel_verification = parallel_verification
        self.mutate_model = mutate_model
        # Default ON: a population that competes without an independent gate
        # optimises its own exam. Opting out is a deliberate, visible choice.
        self.validity_gate = validity_gate if validity_gate is not None else ValidityGate()
        self.on_event = on_event
        self.log: list[dict[str, Any]] = []

    # -- mutation ---------------------------------------------------------
    def analyse_failure(self, tool: ToolSpec, error: str) -> dict[str, str]:
        """Ask the LLM to root-cause a tool failure."""
        prompt = f"Tool: {tool.name}\nCode:\n```python\n{tool.code}\n```\nError:\n{error}\n"
        resp = self.llm.chat(
            [Message.system(_ANALYSE_SYSTEM), Message.user(prompt)],
            tools=None,
        )
        data = extract_json(resp.content)
        if data is None or not isinstance(data, dict):
            return {"root_cause": "unknown", "required_change": "retry", "change_type": "retry"}
        return {
            "root_cause": str(data.get("root_cause", "unknown")),
            "required_change": str(data.get("required_change", "retry")),
            "change_type": str(data.get("change_type", "retry")),
        }

    def evolve(
        self,
        tool: ToolSpec,
        failure_report: str,
        *,
        existing_names: str = "",
        baseline: FrozenBaseline | None = None,
        context: str = "internal",
    ) -> EvolutionResult:
        """Spawn mutants, run the independent gate, then verify the survivors.

        Order matters: the gate runs BEFORE verification, so a mutant that
        deleted a guardrail is dropped without ever competing on fitness. The
        veto is not a penalty it can out-earn.
        """
        result = EvolutionResult(
            tool=tool.name,
            original_version=tool.hash,
        )

        # 1. Spawn population
        prompt = _population_prompt.format(
            name=tool.name, code=tool.code,
            failures=failure_report, n=self.population_size,
        )
        context = f"Existing tools: {existing_names}" if existing_names else ""

        # We ask the model to generate all N in one response (cheaper),
        # then split by FIX-N markers.
        combined_prompt = (
            f"{prompt}\n\n"
            f"Context: {context}\n" if context else ""
        )
        # Use a temporary "for all N" generator
        resp = self.llm.chat(
            [
                Message.system(_MUTATE_SYSTEM),
                Message.user(combined_prompt),
            ],
            tools=None,
        )
        candidates = self._parse_population(resp.content)
        if not candidates:
            result.kept_existing = True
            return result

        # 2. Verify each in turn. Gate first: an inadmissible mutant never
        #    reaches the scoreboard, so no fitness value can rescue it.
        gate = self.validity_gate
        mutants: list[Mutant] = []
        for generated in candidates[:self.population_size]:
            started = time.perf_counter()
            spec = self._to_spec(generated)

            verdict: ValidityReport | None = None
            rejected = ""
            if gate is not None:
                verdict = gate.evaluate(spec, baseline=baseline, context=context)
                if not verdict.admissible:
                    rejected = verdict.violated[0].gate

            if rejected:
                mutants.append(Mutant(
                    generated, None, 0.0,
                    (time.perf_counter() - started) * 1000,
                    (0.0, 0.0, 0.0), verdict, rejected,
                ))
                self._emit_veto(generated.name, rejected, verdict)
                continue

            report = self.verifier.verify(spec)
            duration = (time.perf_counter() - started) * 1000
            mutants.append(Mutant(
                generated, report,
                self._compute_fitness(report), duration,
                self._fitness_vector(report), verdict,
            ))

        admissible = [m for m in mutants if m.admissible]
        result.mutants = mutants
        result.rounds = 1

        # 3. Select among admissible mutants only, via the Pareto front.
        if not admissible:
            result.kept_existing = True
            return result

        best = self._pareto_front(admissible)[0]
        result.best_mutant = best
        result.kept_existing = not (best.report and best.report.passed)

        return result

    def _emit_veto(self, name: str, gate: str, verdict: ValidityReport | None) -> None:
        entry = {
            "t": time.time(), "kind": "mutant_vetoed",
            "tool": name, "gate": gate,
            "detail": verdict.violated[0].detail if verdict and verdict.violated else "",
        }
        self.log.append(entry)
        if self.on_event:
            self.on_event("mutant_vetoed", entry)

    # -- crossover --------------------------------------------------------
    def crossover(self, parent_a: ToolSpec, parent_b: ToolSpec) -> GeneratedTool:
        """Breed a hybrid from two parent tools."""
        prompt = (
            f"Parent A ({parent_a.name}):\n```python\n{parent_a.code}\n```\n\n"
            f"Parent B ({parent_b.name}):\n```python\n{parent_b.code}\n```\n\n"
            "Generate a hybrid tool that merges both capabilities. "
            "Name it something that covers both functions."
        )
        resp = self.llm.chat(
            [Message.system(_CROSSOVER_SYSTEM), Message.user(prompt)],
            tools=None,
        )
        data = extract_json(resp.content)
        if data is None:
            raise ValueError(f"crossover produced no parseable JSON: {resp.content[:300]}")
        probes = [
            TriggerProbe(
                query=p.get("query", ""),
                expect=p.get("expect", "call"),
                negative_query=p.get("negative_query"),
            )
            for p in data.get("probes", [])
            if p.get("query")
        ]
        return GeneratedTool(
            name=data.get("name", "hybrid").strip(),
            description=data.get("description", "").strip(),
            code=data.get("code", "").strip(),
            parameters=data.get("parameters") or {"type": "object", "properties": {}},
            entry=data.get("entry") or data.get("name", ""),
            probes=probes,
            effect_signature=data.get("effect_signature", "pure"),
            tags=data.get("tags") or ["hybrid"],
            rationale=data.get("rationale", ""),
        )

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _compute_fitness(report: VerificationReport) -> float:
        """Score a mutant WITHOUT letting it define its own exam.

        The old formula was `passed / len(report.checks)`, which handed the
        mutant two levers: drop a check it would have failed and the numerator
        *and* the denominator both move in its favour. Replaced with:

          cleared / len(REQUIRED_CHECK_CLASSES)     <- denominator is fixed

        A missing check class is a failed check class. A mutant that omits its
        negative probe scores as if the negative probe failed, which is the
        truth. The probe-mass multiplier then rewards surviving more evidence
        but is capped, so it cannot be farmed by inflating a trivial probe set.
        """
        if not report.checks:
            return 0.0
        by_class: dict[str, list[CheckResult]] = {}
        for c in report.checks:
            by_class.setdefault(c.name, []).append(c)

        cleared = sum(
            1 for cls in REQUIRED_CHECK_CLASSES
            if (rs := by_class.get(cls)) and all(r.passed for r in rs)
        )
        base = cleared / len(REQUIRED_CHECK_CLASSES)
        mass = min(EvolutionEngine._survived_mass(report), _REFERENCE_PROBE_MASS)
        return base * (1.0 + 0.25 * mass / _REFERENCE_PROBE_MASS)

    @staticmethod
    def _survived_mass(report: VerificationReport) -> float:
        """How much evidence actually survived. Reads counts, not class names."""
        mass = 0.0
        for c in report.checks:
            if not c.passed:
                continue
            ev = getattr(c, "evidence", None)
            n = ev.get("survived") if isinstance(ev, dict) else None
            mass += float(n) if isinstance(n, (int, float)) else 1.0
        return mass

    @staticmethod
    def _fitness_vector(report: VerificationReport) -> tuple[float, float, float]:
        """(task, safety, robustness) — kept apart so no axis can buy another.

        paste_5's point: a single scalar lets a mutant pay for safety with
        capability. A vector means `high task, low safety` is dominated by any
        mutant that is at least as good on all three.
        """
        by_class: dict[str, list[CheckResult]] = {}
        for c in report.checks:
            by_class.setdefault(c.name, []).append(c)

        def rate(cls: str) -> float:
            rs = by_class.get(cls) or []
            return sum(1 for r in rs if r.passed) / len(rs) if rs else 0.0

        task = (rate("execution") + rate("trigger")) / 2
        safety = (rate("negative") + rate("adversarial")) / 2
        robustness = rate("robustness")
        return (round(task, 4), round(safety, 4), round(robustness, 4))

    @staticmethod
    def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
        """Pareto dominance: no worse anywhere, strictly better somewhere."""
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

    @classmethod
    def _pareto_front(cls, mutants: list["Mutant"]) -> list["Mutant"]:
        """The non-dominated survivors, best fitness first."""
        front = [
            m for m in mutants
            if not any(cls._dominates(o.vector, m.vector)
                       for o in mutants if o is not m)
        ]
        return sorted(front or mutants, key=lambda m: m.fitness, reverse=True)

    @staticmethod
    def _parse_population(text: str) -> list[GeneratedTool]:
        """Parse a multi-mutant response into individual GeneratedTools."""
        results: list[GeneratedTool] = []
        # Strategy: find all JSON blocks, collect them
        blocks = text.split("FIX-")
        for block in blocks:
            data = extract_json(block)
            if data is None or not isinstance(data, dict):
                continue
            if not data.get("name") or not data.get("code"):
                continue
            probes = [
                TriggerProbe(
                    query=p.get("query", ""),
                    expect=p.get("expect", "call"),
                    negative_query=p.get("negative_query"),
                )
                for p in data.get("probes", [])
                if p.get("query")
            ]
            results.append(GeneratedTool(
                name=data.get("name", "").strip(),
                description=data.get("description", "").strip(),
                code=data.get("code", "").strip(),
                parameters=data.get("parameters") or {"type": "object", "properties": {}},
                entry=data.get("entry") or data.get("name", ""),
                probes=probes,
                effect_signature=data.get("effect_signature", "pure"),
                tags=data.get("tags") or [],
                rationale=data.get("rationale", ""),
            ))
        return results

    @staticmethod
    def _to_spec(g: GeneratedTool) -> ToolSpec:
        return ToolSpec(
            name=g.name,
            description=g.description,
            parameters=g.parameters,
            fn=lambda **_: (_ for _ in ()).throw(RuntimeError("evolution stub")),
            code=g.code,
            source="evolved",
            probes=g.probes,
            effect_signature=g.effect_signature,
            tags=g.tags,
            state=ToolState.DRAFT,
        )


__all__ = [
    "EvolutionEngine",
    "EvolutionResult",
    "Mutant",
    "GeneratedTool",
    "ToolVerifier",
    "REQUIRED_CHECK_CLASSES",
]