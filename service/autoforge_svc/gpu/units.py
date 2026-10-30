"""Timing units for the CUDA layer.

The type and the lints live in `autoforge.timing`, because the bug they prevent
is arithmetical rather than hardware-specific — the CPU layer is exposed to the
mirror image of it. Re-exported here so that everything already written against
`gpu.units.Duration` keeps working and there is exactly one `Duration` in the
process rather than two that can drift.

What is genuinely CUDA-shaped stays in this module: `from_cuda_event`, because
`cudaEventElapsedTime` is a CUDA API with a fixed millisecond contract.
"""
from __future__ import annotations

from ..timing import (
    Duration,
    MsOffence,
    UnitError,
    audit_ms_from_seconds,
    audit_ms_scale,
    audit_source_tree,
    from_do_bench,
    from_wallclock,
)


def from_cuda_event(value: float, unit: str = "ms") -> Duration:
    """Wrap a CUDA-event elapsed_time result. cudaEventElapsedTime is ms."""
    if unit.lower() in ("ms", "millisecond", "milliseconds"):
        return Duration.from_ms(value)
    if unit.lower() in ("s", "sec", "second", "seconds"):
        return Duration.from_seconds(value)
    raise UnitError(f"cuda event elapsed_time is in ms; got unit={unit!r}")


__all__ = [
    "Duration", "UnitError", "MsOffence",
    "from_do_bench", "from_wallclock", "from_cuda_event",
    "audit_ms_scale", "audit_ms_from_seconds", "audit_source_tree",
]
