"""The search: propose a kernel, prove it right, measure it, keep the winner.

The loop is generate → compile → verify → measure → mutate, and the ordering is
the method rather than an implementation detail:

  VERIFY BEFORE MEASURE. Every candidate is run against the reference in an
  isolated child *first*, and a candidate that fails is never timed. This is
  what makes the mutation catalogue safe to be reckless with: a transform that
  breaks the kernel does not corrupt the search, it just does not get a number.
  A loop that times first and checks afterwards will select for the kernel that
  skipped the most work, because skipping work is the most reliable way to be
  fast.

  THE BASELINE IS FORCED FAST. Beating `-O2` with `-O3` is not a result, it is
  a build flag — and it is the single most common way a "4x faster kernel"
  claim turns out to be empty. So the baseline is built twice: once at the
  package default and once with the strongest flags available on this machine.
  `measure.Comparison` refuses to phrase a win against the second one as a
  claim unless it beat the first, and the report prints both ratios so the
  reader can see which one the number came from.

  MUTATIONS ARE LABELLED BY WHAT THEY CHANGE. Every variant records the one
  thing it is testing, so the history reads as a set of hypotheses rather than
  a list of opaque source diffs. When the best result is "add `restrict`", that
  is a finding worth keeping; when it is "flag soup", that is worth knowing too.

What this deliberately does not do: call a model to write kernels. The mutation
catalogue is syntactic and small on purpose, because the value here is the
*discipline* — the verify-before-measure loop, the isolated runner, the honest
baseline — and a catalogue that can be read in one sitting is what makes a
result attributable. Swapping in generated candidates later is a change to one
function; the guarantees around them stay.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..measure import Comparison, DEFAULT_REPS, DEFAULT_WARMUP, BenchResult
from ..timing import Duration
from . import ops as ops_mod
from . import safety
from .kernel import CompiledKernel, KernelSource, compile_kernel
from .probe import probe

#: How the baseline is built when we want the strongest thing this machine can
#: do, so "we beat the baseline" cannot be an artefact of optimisation level.
FAST_BASELINE_FLAGS: tuple[str, ...] = ("-O3", "-funroll-loops", "-funsafe-math-optimizations")

#: Hard ceiling on candidates, so a search cannot run away from the agent.
DEFAULT_MAX_CANDIDATES = 24

#: Wall-clock ceiling for the whole search, seconds.
DEFAULT_BUDGET_S = 300.0

#: The main loop of a kernel, as it appears in the shapes this codebase forges.
_MAIN_LOOP_RE = re.compile(r"^([ \t]*)(for\s*\([^)]*\))\s*\{?", re.MULTILINE)

#: A pointer parameter, for the `restrict` mutation.
_PTR_PARAM_RE = re.compile(r"\b(const\s+)?(float|double|int|long|unsigned)\s*\*\s*"
                           r"([A-Za-z_][A-Za-z0-9_]*)")


# -- mutations --------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    """A candidate source and the single thing it is testing."""

    source: KernelSource
    change: str
    kind: str = "flags"                      # flags | source | parallel | shape


def flag_variants(source: KernelSource, info: Any = None) -> list[Variant]:
    """The cheapest mutations: change nothing but the build.

    Ordered from most-likely-to-help to least, because the search stops as soon
    as it has an answer it can defend and burning the budget on flag soup after
    that is how a 20-minute search happens.
    """
    out: list[Variant] = []
    base = set(source.flags)

    def add(flags: tuple[str, ...], change: str) -> None:
        merged = tuple(f for f in flags if f not in base)
        if merged and merged != source.flags:
            out.append(Variant(KernelSource(source.name, source.code,
                                            source.target, merged,
                                            source.openmp), change))

    add(("-O3",), "raise the optimisation level to -O3")
    add(("-O3", "-funroll-loops"), "unroll the loops (-funroll-loops)")
    if info is None or info.has("AVX2"):
        add(("-march=native",), "build for this machine's instruction set")
    add(("-O3", "-flto"), "link-time optimisation across the shim")
    add(("-O3", "-fno-tree-vectorize"), "disable auto-vectorisation (a control: "
                                        "if this is faster, the vectoriser is "
                                        "hurting, which means the loop is "
                                        "aliasing-bound)")
    return out


def source_variants(source: KernelSource, info: Any = None) -> list[Variant]:
    """Mutations that change the C, each one labelled with its hypothesis.

    All of them are guesses about the compiler, and all of them are cheap to
    refute because a wrong guess simply fails to verify or fails to be faster.
    """
    out: list[Variant] = []
    code = source.code

    if "restrict" not in code:
        # The hypothesis: the compiler cannot prove the two pointers do not
        # overlap, so it emits a scalar loop plus a runtime alias check. This is
        # the single most common reason a plain C kernel is 3-4x off the mark.
        rewritten = _PTR_PARAM_RE.sub(
            lambda m: f"{m.group(1) or ''}{m.group(2)} *__restrict {m.group(3)}",
            code)
        if rewritten != code:
            out.append(Variant(
                KernelSource(source.name, rewritten, source.target, source.flags,
                             source.openmp),
                "mark the pointer parameters __restrict so the compiler may "
                "vectorise (aliasing is the usual reason one of these is slow)",
                "source"))
        out.append(Variant(
            KernelSource(source.name, _inject_before_loop(code, "#pragma GCC ivdep"),
                         source.target, source.flags, source.openmp),
            "assert no loop-carried dependency with `#pragma GCC ivdep`",
            "source"))

    if "#pragma GCC unroll" not in code:
        out.append(Variant(
            KernelSource(source.name, _inject_before_loop(code, "#pragma GCC unroll 8"),
                         source.target, source.flags, source.openmp),
            "unroll the main loop by 8",
            "source"))

    if not source.openmp:
        out.append(Variant(
            KernelSource(source.name, _inject_before_loop(code, "#pragma omp parallel for"),
                         source.target, source.flags, True),
            "parallelise the main loop with OpenMP (works only if the "
            "iterations are independent — verification decides)",
            "parallel"))

    if "#include <omp.h>" not in code and info is not None and info.openmp:
        out.append(Variant(
            KernelSource(source.name, _inject_before_loop(code, "#pragma omp simd"),
                         source.target, source.flags, True),
            "ask for SIMD explicitly with `#pragma omp simd`",
            "parallel"))
    return out


def _inject_before_loop(code: str, pragma: str) -> str:
    """Put a pragma on the line before the first `for` in the source.

    Syntactic, and honest about it: on a kernel whose first loop is not the hot
    one this lands in the wrong place, and the compiler then either errors or
    the variant fails to be faster. Both outcomes are handled by the loop.
    """
    m = _MAIN_LOOP_RE.search(code)
    if not m:
        return code
    indent = m.group(1)
    at = m.start()
    return f"{code[:at]}{indent}{pragma}\n{code[at:]}"


def variants(source: KernelSource, info: Any = None) -> list[Variant]:
    """The whole catalogue, flags first because they are free."""
    return flag_variants(source, info) + source_variants(source, info)


# -- search -----------------------------------------------------------------
@dataclass
class Candidate:
    """One attempt: what it was, what happened, and the number if it earned one."""

    change: str
    kind: str
    source: KernelSource
    compiled: bool = False
    verified: bool = False
    median_ms: float | None = None
    spread_pct: float | None = None
    timings_s: list[float] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    error: str = ""
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.compiled and self.verified and self.median_ms is not None

    def bench(self) -> BenchResult:
        """The full sample set, not just the median.

        Kept because the claim machinery needs the spread: a comparison built
        from a single number has a spread of zero, reads as a measurement with no
        noise, and gets downgraded for being too tidy to believe — which is the
        right verdict for a made-up sample and the wrong one for a real median.
        """
        return BenchResult(timings=[Duration.from_seconds(s) for s in self.timings_s],
                           label=self.change)

    def to_dict(self) -> dict[str, Any]:
        return {
            "change": self.change, "kind": self.kind, "compiled": self.compiled,
            "verified": self.verified, "median_ms": self.median_ms,
            "spread_pct": self.spread_pct, "blocked": list(self.blocked),
            "error": self.error, "note": self.note,
            "target": self.source.resolved_target(),
            "flags": list(self.source.build_command),
        }

    def line(self) -> str:
        if self.blocked:
            return f"BLOCKED   {self.change}  [{', '.join(self.blocked)}]"
        if not self.compiled:
            return f"NO BUILD  {self.change}  ({_first_line(self.error)})"
        if not self.verified:
            return f"WRONG     {self.change}  ({_first_line(self.error)})"
        spread = f"±{self.spread_pct:.2f}%" if self.spread_pct is not None else ""
        return f"OK        {self.median_ms:9.4f} ms {spread:>9s}  {self.change}"


def _first_line(text: str, limit: int = 110) -> str:
    """The most informative line of a multi-line failure.

    Not simply the first: a compiler error's first line is the include stack
    ("In file included from ..."), and truncating there reports a build failure
    with no reason in it. The line carrying `error:` is the one that says what
    to change, so it is preferred when present.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "no diagnostics"
    pick = next((ln for ln in lines if "error" in ln.lower()), lines[0])
    return pick[:limit]


@dataclass
class SearchResult:
    """The winner, the history, and an honest statement of what was proven."""

    problem: Any
    baseline: BenchResult
    baseline_fast: BenchResult
    winner: Candidate | None
    history: list[Candidate] = field(default_factory=list)
    stopped_because: str = ""
    wall: Duration = field(default_factory=lambda: Duration.from_seconds(0.0))
    n_candidates: int = 0

    @property
    def comparison_fast(self) -> Comparison | None:
        """The claim: ours against the strongest baseline on this machine."""
        if self.winner is None or self.winner.median_ms is None:
            return None
        return Comparison(label="the -O3 baseline", ours=self.winner.bench(),
                          baseline=self.baseline_fast, baseline_forced_fast=True,
                          scope=f"{self.problem.kind}, "
                                f"{self.problem.spec.size_arg()} elements")

    @property
    def comparison_default(self) -> Comparison | None:
        if self.winner is None or self.winner.median_ms is None:
            return None
        return Comparison(label="the default-flag baseline", ours=self.winner.bench(),
                          baseline=self.baseline, baseline_forced_fast=False,
                          scope=f"{self.problem.kind}")

    def report(self) -> str:
        lines = [
            f"search over {self.n_candidates} candidates for "
            f"{self.problem.kind} ({self.problem.spec.describe()})",
            f"  baseline (default flags) : {self._ms(self.baseline)}",
            f"  baseline (-O3, -ffast)   : {self._ms(self.baseline_fast)}",
            "",
        ]
        for c in self.history:
            lines.append("  " + c.line())
        lines.append("")
        if self.winner is not None:
            lines.append(f"  winner: {self.winner.change}")
            lines.append(f"          {self._ms(self.baseline)} -> "
                         f"{self.winner.median_ms:.4f} ms")
        else:
            lines.append("  no candidate beat the baseline and verified")
        for cmp_ in (self.comparison_default, self.comparison_fast):
            if cmp_ is not None:
                lines.append(f"  claim: {cmp_.claim}")
        if self.stopped_because:
            lines.append(f"  stopped: {self.stopped_because} "
                         f"(after {self.wall.format('ms')})")
        return "\n".join(lines)

    @staticmethod
    def _ms(bench: BenchResult) -> str:
        return bench.median.format("ms") if bench.ok else "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "baseline_ms": self.baseline.median.ms if self.baseline.ok else None,
            "baseline_fast_ms": self.baseline_fast.median.ms if self.baseline_fast.ok else None,
            "winner": self.winner.to_dict() if self.winner else None,
            "history": [c.to_dict() for c in self.history],
            "stopped_because": self.stopped_because,
            "wall_s": self.wall.seconds_,
            "n_candidates": self.n_candidates,
            "claim": self.comparison_fast.claim if self.comparison_fast else "",
        }


def baseline_source(code: str, name: str, *, fast: bool = False) -> KernelSource:
    """The reference implementation at a stated optimisation level.

    `fast=True` builds with the strongest flags this package will use, which is
    the baseline a claim has to beat. `fast=False` is the default build, which
    is the baseline a *result* has to beat to be worth reporting at all.
    """
    flags = FAST_BASELINE_FLAGS if fast else ()
    return KernelSource(name, code, "native", flags)


def tune(
    problem: Any,
    code: str,
    *,
    name: str | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    budget_s: float = DEFAULT_BUDGET_S,
    reps: int = DEFAULT_REPS,
    warmup: int = DEFAULT_WARMUP,
    timeout: float | None = None,
    on_step: Callable[[Candidate], None] | None = None,
    catalogue: Iterable[Variant] | None = None,
) -> SearchResult:
    """Search for a faster correct kernel, and say what was actually shown.

    `problem` must come from `ops.problem`, so a reference exists; the search
    has no way to accept a workload it cannot check.
    """
    started = time.perf_counter()
    info = probe()
    name = name or problem.spec.fn

    base_src = baseline_source(code, name)
    fast_src = baseline_source(code, name, fast=True)

    baseline = _measure(base_src, problem, reps, warmup, timeout)
    baseline_fast = _measure(fast_src, problem, reps, warmup, timeout)

    result = SearchResult(problem=problem, baseline=baseline,
                          baseline_fast=baseline_fast, winner=None)
    if not baseline.ok:
        result.stopped_because = (f"the baseline itself did not verify: "
                                  f"{baseline.error or baseline.verdict.get('reason', '')}")
        return result

    best = baseline.median.ms
    cands = list(catalogue) if catalogue is not None else variants(base_src, info)

    for i, var in enumerate(cands, 1):
        if i > max_candidates:
            result.stopped_because = "candidate ceiling reached"
            break
        if time.perf_counter() - started > budget_s:
            result.stopped_because = "wall-clock budget reached"
            break

        cand = _attempt(var, problem, reps, warmup, timeout)
        result.history.append(cand)
        result.n_candidates += 1
        if on_step is not None:
            on_step(cand)

        if cand.ok and cand.median_ms is not None and cand.median_ms < best:
            best = cand.median_ms
            result.winner = cand
            result.stopped_because = "found an improvement" if not result.stopped_because \
                else result.stopped_because

    if not result.stopped_because:
        result.stopped_because = "catalogue exhausted"
    result.wall = Duration.from_seconds(time.perf_counter() - started)
    return result


def _attempt(var: Variant, problem: Any, reps: int, warmup: int,
             timeout: float | None) -> Candidate:
    """Preflight, compile, verify, measure — in that order, stopping on failure."""
    cand = Candidate(change=var.change, kind=var.kind, source=var.source)

    pf = safety.preflight(var.source, problem=problem)
    cand.blocked = [f.code for f in pf.blocking]
    if pf.blocking:
        cand.error = "; ".join(str(f) for f in pf.blocking)
        return cand

    try:
        kernel = compile_kernel(var.source)
    except Exception as exc:                                 # noqa: BLE001
        cand.error = str(exc)
        return cand
    cand.compiled = True

    guard = safety.run_isolated(kernel, problem, reps=reps, warmup=warmup,
                                timeout=timeout)
    if guard.timed_out:
        cand.error = "timed out — an unbounded loop, most likely"
        return cand
    if guard.crashed:
        cand.error = guard.fault or f"crashed (exit {guard.returncode})"
        return cand
    if not guard.ok:
        cand.error = guard.verdict.get("reason", guard.error or "not verified")
        bad = [k for k, v in (guard.verdict.get("checks") or {}).items()
               if not v.get("passed")]
        if bad:
            first = (guard.verdict["checks"] or {})[bad[0]]
            cand.note = (f"output {bad[0]}: {first.get('reason', '')} "
                         f"(max|Δ|={first.get('max_abs_diff')})")
        return cand

    cand.verified = True
    timings = list(guard.verdict.get("timings_s", []))
    bench = BenchResult(timings=[Duration.from_seconds(s) for s in timings],
                        label=var.change)
    if not bench.ok:
        cand.error = "verified but produced no timings"
        cand.verified = False
        return cand
    cand.timings_s = timings
    cand.median_ms = bench.median.ms
    cand.spread_pct = bench.spread_pct
    if not bench.is_signal:
        cand.note = bench.verdict
    return cand


def _measure(source: KernelSource, problem: Any, reps: int, warmup: int,
             timeout: float | None) -> BenchResult:
    """Build and measure one source, returning a BenchResult that says why not."""
    try:
        kernel = compile_kernel(source)
    except Exception as exc:                                 # noqa: BLE001
        return BenchResult(timings=[], label=f"baseline {source.name}",
                           error=str(exc))
    guard = safety.run_isolated(kernel, problem, reps=reps, warmup=warmup,
                                timeout=timeout)
    if not guard.ok:
        return BenchResult(timings=[], label=f"baseline {source.name}",
                           error=guard.fault or guard.error
                           or guard.verdict.get("reason", "not verified"))
    return BenchResult(timings=[Duration.from_seconds(s)
                                for s in guard.verdict.get("timings_s", [])],
                       label=f"baseline {source.name}")


# -- reference kernels ------------------------------------------------------
#: Naive C sources for the problems in `ops.problem`. These are the starting
#: points the search improves on, and they are deliberately the obvious version:
#: a search that starts from an already-tuned kernel has nothing to find, and a
#: baseline that is already good hides the transforms that matter.
NAIVE: dict[str, str] = {
    "saxpy": """
#include <stddef.h>
void saxpy(float *y, const float *x, float a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}
""",
    "vector_add": """
#include <stddef.h>
void vector_add(float *out, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}
""",
    "reduce_sum": """
#include <stddef.h>
float reduce_sum(const float *x, size_t n) {
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++) {
        acc += x[i];
    }
    return acc;
}
""",
    "relu": """
#include <stddef.h>
void relu(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}
""",
    "softmax": """
#include <stddef.h>
#include <math.h>
void softmax(float *out, const float *x, size_t n) {
    float m = x[0];
    for (size_t i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        out[i] = expf(x[i] - m);
        sum += out[i];
    }
    for (size_t i = 0; i < n; i++) out[i] /= sum;
}
""",
    "matmul": """
#include <stddef.h>
void matmul(float *C, const float *A, const float *B, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < n; k++) {
                acc += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = acc;
        }
    }
}
""",
}


def naive(kind: str) -> str:
    """The starting source for a problem kind."""
    if kind not in NAIVE:
        raise KeyError(f"no naive kernel for {kind!r}; known: {', '.join(NAIVE)}")
    return NAIVE[kind]


def tune_kind(
    kind: str,
    *,
    size: int = 1 << 20,
    n: int = 256,
    dtype: str = "f32",
    seed: int = 0,
    **kwargs: Any,
) -> SearchResult:
    """Build the problem and the starting kernel, then search. The one call a
    tool handler needs."""
    prob = ops_mod.problem(kind, size=size, n=n, dtype=dtype, seed=seed)
    return tune(prob, naive(kind), name=kind, **kwargs)


__all__ = [
    "FAST_BASELINE_FLAGS", "DEFAULT_MAX_CANDIDATES", "DEFAULT_BUDGET_S",
    "Variant", "Candidate", "SearchResult", "tune", "tune_kind",
    "variants", "flag_variants", "source_variants", "naive", "NAIVE",
    "baseline_source",
]
