"""Measuring a kernel without fooling yourself.

Hardware-agnostic on purpose: a GPU kernel and a CPU kernel are wrong in the
same four ways, and the ways are all in the numbers rather than in the code.
Shared by both kernel layers so a fix to the discipline lands on both at once.

  1. UNITS. `do_bench` returns milliseconds, `perf_counter` returns seconds.
     Scale either one wrong and every figure is 1000x off while the label still
     reads "ms". `timing.Duration` makes that structurally hard; this module
     never formats a duration itself — it only ever emits `Duration`, which
     prints its own unit.
  2. NOISE. A single measurement is not a measurement. Every result carries a
     spread, and the caller is told whether the spread is small enough for the
     difference to mean anything.
  3. A SLOW TARGET. Beating a baseline that silently took a slow path means
     nothing. `Comparison` requires the caller to state whether the baseline was
     forced down its fast path, and labels an unforced win as unproven. On the
     CPU this is the difference between `-O0` and `-O3`, or between a naive
     loop and BLAS, and it is the single most common way a "5x faster kernel"
     turns out to be a build flag.
  4. SKIPPED WORK. A kernel that drops the mask is fast and wrong. `verify()`
     demands a reference and reports the max absolute difference, so an
     identity kernel cannot pass as an optimisation.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .timing import Duration, UnitError, from_do_bench

# Spread thresholds, as a percentage of the median. These are the skill's
# numbers, kept in one place so a result's verdict is not a judgement call made
# separately at each call site.
SPREAD_SIGNAL_PCT = 1.0      # below this, a difference is real
SPREAD_NOISE_PCT = 5.0       # above this, the measurement is a coin flip

DEFAULT_WARMUP = 3
DEFAULT_REPS = 5


@dataclass
class BenchResult:
    """Durations, a spread, and an explicit statement of what is not known."""

    timings: list[Duration]
    label: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.timings)

    @property
    def n(self) -> int:
        return len(self.timings)

    def _seconds(self) -> list[float]:
        return [d.seconds_ for d in self.timings]

    @property
    def fastest(self) -> Duration:
        return Duration(min(self._seconds()))

    @property
    def median(self) -> Duration:
        return Duration(statistics.median(self._seconds()))

    @property
    def spread_pct(self) -> float:
        """(max-min)/median, in percent. The honest noise figure.

        Uses max-min rather than stdev because a single slow outlier — a
        dispatch hiccup, another process on the machine — is exactly what the
        reader needs to see, and stdev hides it.
        """
        secs = self._seconds()
        if len(secs) < 2:
            return 0.0
        med = statistics.median(secs)
        if med <= 0:
            return 0.0
        return (max(secs) - min(secs)) / med * 100.0

    @property
    def verdict(self) -> str:
        if not self.ok:
            return "no measurement"
        if self.n < 2:
            return f"single sample ({self.median.format('ms')}) — not a measurement"
        s = self.spread_pct
        if s <= SPREAD_SIGNAL_PCT:
            return f"spread {s:.2f}% — signal"
        if s >= SPREAD_NOISE_PCT:
            return f"spread {s:.2f}% — noise; increase reps before concluding"
        return f"spread {s:.2f}% — marginal; more reps would settle it"

    @property
    def is_signal(self) -> bool:
        return self.ok and self.n >= 2 and self.spread_pct <= SPREAD_SIGNAL_PCT

    def summary(self) -> str:
        if not self.ok:
            return f"{self.label or 'benchmark'}: FAILED — {self.error or 'no samples'}"
        head = (f"{self.label or 'benchmark'}: {self.median.format('ms')} "
                f"(min {self.fastest.format('ms')}, n={self.n})")
        out = [head, f"  {self.verdict}"]
        for note in self.notes:
            out.append(f"  note: {note}")
        return "\n".join(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "ok": self.ok,
            "n": self.n,
            "median_ms": self.median.ms if self.ok else None,
            "min_ms": self.fastest.ms if self.ok else None,
            "spread_pct": self.spread_pct if self.ok else None,
            "spread_level": ("signal" if self.spread_pct <= SPREAD_SIGNAL_PCT
                             else "noise" if self.spread_pct >= SPREAD_NOISE_PCT
                             else "marginal") if self.ok and self.n >= 2 else "n/a",
            "verdict": self.verdict,
            "error": self.error,
            "notes": list(self.notes),
            "meta": dict(self.meta),
        }


def benchmark_kernel(
    fn: Callable[[], Any],
    *,
    label: str = "",
    warmup: int = DEFAULT_WARMUP,
    reps: int = DEFAULT_REPS,
    on_first_call: Callable[[Any], None] | None = None,
) -> BenchResult:
    """Time `fn` over several repetitions and report the spread.

    `fn` returns either a Duration, or a float which is interpreted as
    MILLISECONDS — the convention the GPU timers use, and the one value this
    codebase accepts without a unit attached. The interpretation is stated here
    and nowhere else; every other place carries a Duration.

    Warmup calls are never timed. A cold first call pays compilation and
    algorithm-selection costs that are real but are not what is being measured,
    and including it is how a 7% fake spread appears.
    """
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    if warmup < 0:
        raise ValueError(f"warmup cannot be negative, got {warmup}")

    notes: list[str] = []
    if reps < 3:
        notes.append(f"only {reps} repetitions; a spread needs more to mean anything")

    timings: list[Duration] = []
    try:
        for _ in range(warmup):
            fn()
        for i in range(reps):
            started = time.perf_counter()
            value = fn()
            wall = time.perf_counter() - started
            if isinstance(value, Duration):
                timings.append(value)
            elif isinstance(value, (int, float)):
                timings.append(from_do_bench(float(value)))
            else:
                # No timer returned. Fall back to the wall clock around the
                # call, and say so, because a wall-clock number around an async
                # launch measures the launch, not the kernel. On the CPU the
                # same caveat applies for a different reason: the call may
                # return before the work is done if it queued anything (a
                # thread pool, a library that dispatches asynchronously).
                timings.append(Duration.from_seconds(wall))
                if "wall-clock fallback" not in " ".join(notes):
                    notes.append(
                        "fn returned no Duration; timed with the wall clock "
                        "around the call, which measures the call rather than "
                        "the work if anything it started outlives it"
                    )
        if on_first_call is not None and timings:
            on_first_call(None)
    except Exception as exc:                                 # noqa: BLE001
        return BenchResult(timings=timings, label=label,
                           error=f"{type(exc).__name__}: {exc}", notes=notes)

    return BenchResult(timings=timings, label=label, notes=notes)


@dataclass
class Comparison:
    """A speedup claim, with the honesty it requires attached."""

    label: str
    ours: BenchResult
    baseline: BenchResult
    baseline_forced_fast: bool
    scope: str = ""

    @property
    def ratio(self) -> float:
        """ours/baseline in time. <1 means we are faster.

        Safe against a constant unit error because both sides carry their own
        unit — which is the one comparison the do_bench bug cannot corrupt.
        """
        if not self.ours.ok or not self.baseline.ok:
            raise UnitError("cannot compare an unsuccessful benchmark")
        return self.ours.median / self.baseline.median

    @property
    def claim(self) -> str:
        if not self.ours.ok or not self.baseline.ok:
            return "no claim — a side of the comparison failed"
        r = self.ratio
        faster = r < 1.0
        if not faster:
            return f"{1 / r:.3f}x slower than {self.label}"
        if not self.baseline_forced_fast:
            return (f"faster than {self.label}, but the baseline was NOT forced "
                    f"down its fast path — this is not yet a claim")
        if not self.ours.is_signal:
            return (f"{1 / r:.3f}x faster than {self.label}, but our own spread "
                    f"({self.ours.spread_pct:.2f}%) is too wide to assert it")
        scope = f" at {self.scope}" if self.scope else ""
        return f"{1 / r:.3f}x of {self.label}{scope}"

    def summary(self) -> str:
        lines = [self.ours.summary(), self.baseline.summary(), f"  -> {self.claim}"]
        if self.scope:
            lines.append(f"  scope: {self.scope}")
        base = f"{self.label} was forced to its fast backend" if self.baseline_forced_fast \
            else f"{self.label} was NOT forced to its fast backend"
        lines.append(f"  ({base})")
        return "\n".join(lines)


def verify(
    actual: Sequence[float],
    reference: Sequence[float],
    *,
    tolerance: float = 1e-4,
    label: str = "",
) -> dict[str, Any]:
    """Compare against a reference, so a fast-but-wrong kernel cannot pass.

    Returns a dict rather than a bool: the caller needs `max_abs_diff` to judge
    whether a failure is a real bug or accumulated floating-point error, and a
    bare False would take that away.

    Two specific readings worth knowing, both from real kernels:
      max_diff == 0.0   -> you measured the identity; the kernel did nothing
      max_diff ~ 2e-4   -> correct fp16 accumulation
      max_diff > 1e-1   -> a real bug, not precision
    """
    if len(actual) != len(reference):
        return {
            "label": label, "passed": False, "max_abs_diff": None,
            "reason": f"length mismatch: {len(actual)} vs {len(reference)}",
        }
    if not actual:
        return {"label": label, "passed": False, "max_abs_diff": None,
                "reason": "empty output"}

    max_diff = 0.0
    worst = 0
    for i, (a, b) in enumerate(zip(actual, reference)):
        d = abs(float(a) - float(b))
        if d > max_diff:
            max_diff, worst = d, i

    exact = max_diff == 0.0
    return {
        "label": label,
        "passed": max_diff <= tolerance,
        "max_abs_diff": max_diff,
        "worst_index": worst,
        "tolerance": tolerance,
        "reason": (
            "identical to the reference — did the kernel do any work?"
            if exact else
            "within tolerance" if max_diff <= tolerance else
            "exceeds tolerance: this is a correctness bug, not precision"
        ),
    }


def bandwidth_gbps(bytes_moved: int, duration: Duration) -> float:
    """Effective bandwidth in GB/s.

    Exists so nobody hand-writes this. The classic bug is computing it from a
    value that was already scaled by 1000, which yields a number three orders
    of magnitude off — and the cross-check that catches it (does this match the
    hardware's peak?) only works if the arithmetic is in one place.

    GB/s here is 10^9 bytes per second, matching vendor spec sheets. On a CPU
    this is the number that explains a disappointing multi-threaded kernel: a
    loop over a large array stops scaling at the point it saturates the memory
    bus, and the byte count is the only way to see that.
    """
    if duration.seconds_ <= 0:
        raise UnitError("bandwidth of a zero-length duration is undefined")
    return (bytes_moved / 1e9) / duration.seconds_


def check_units_by_wallclock(
    synth_fn: Callable[[], float],
    *,
    expected_seconds: float,
    tolerance: float = 0.5,
) -> dict[str, Any]:
    """Settle "is this value ms or s?" by measurement, not by documentation.

    `synth_fn` must return a duration claim (a float) for work whose real
    wall-clock cost is known — e.g. a `time.sleep`, or a kernel of known length.
    We time the call with perf_counter and ask which interpretation of the
    returned value matches.

    tolerance is a fraction: 0.5 means the reading must be within 50% of the
    true wall time, which is loose enough for a busy machine and tight enough to
    separate ms from s by a factor of 1000.
    """
    started = time.perf_counter()
    claimed = float(synth_fn())
    wall = time.perf_counter() - started
    if wall <= 0:
        return {"settled": False, "reason": "wall clock did not advance"}
    if expected_seconds <= 0:
        return {"settled": False, "reason": "expected_seconds must be positive"}

    as_ms = claimed / 1000.0
    as_s = claimed
    ms_err = abs(as_ms - expected_seconds) / expected_seconds
    s_err = abs(as_s - expected_seconds) / expected_seconds

    if ms_err <= tolerance and ms_err < s_err:
        unit = "ms"
    elif s_err <= tolerance:
        unit = "s" if s_err < ms_err else "ms"
    else:
        return {
            "settled": False,
            "claimed": claimed, "wall_s": wall, "expected_s": expected_seconds,
            "reason": "neither interpretation matches the wall clock",
        }
    return {
        "settled": True,
        "unit": unit,
        "claimed": claimed,
        "claimed_as_s": claimed if unit == "s" else claimed / 1000.0,
        "wall_s": wall,
        "expected_s": expected_seconds,
        "relative_error": ms_err if unit == "ms" else s_err,
    }


__all__ = [
    "SPREAD_SIGNAL_PCT", "SPREAD_NOISE_PCT", "DEFAULT_WARMUP", "DEFAULT_REPS",
    "BenchResult", "Comparison", "benchmark_kernel", "verify",
    "bandwidth_gbps", "check_units_by_wallclock",
]
