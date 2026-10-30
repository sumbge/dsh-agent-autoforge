"""Measuring a CUDA kernel without fooling yourself.

The measurement discipline — units, spread, the baseline-that-must-be-forced,
and verification against a reference — lives in `autoforge.measure`, because a
CPU kernel is wrong in the same four ways. Re-exported here so everything
already written against `gpu.bench` keeps working.

The GPU-specific part is the note that matters on this side: a wall-clock
number around an asynchronous launch measures the launch, not the kernel.
"""
from __future__ import annotations

from ..measure import (
    DEFAULT_REPS,
    DEFAULT_WARMUP,
    SPREAD_NOISE_PCT,
    SPREAD_SIGNAL_PCT,
    BenchResult,
    Comparison,
    bandwidth_gbps,
    benchmark_kernel,
    check_units_by_wallclock,
    verify,
)

__all__ = [
    "SPREAD_SIGNAL_PCT", "SPREAD_NOISE_PCT", "DEFAULT_WARMUP", "DEFAULT_REPS",
    "BenchResult", "Comparison", "benchmark_kernel", "verify",
    "bandwidth_gbps", "check_units_by_wallclock",
]
