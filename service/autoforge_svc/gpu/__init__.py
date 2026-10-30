"""CUDA kernel-layer control.

Hermes (and every other agent framework) reaches a GPU the way it reaches
anything else: by shelling out to `nvcc`, or by asking torch to do it. That is
process-level access. This package is a layer below that — the agent's tools
speak in terms of *kernels*, *launch geometry*, *occupancy* and *event timing*,
and it learns the result of a change in the same units the hardware reports.

Why it belongs in autoforge specifically: autoforge's whole claim is that a
forged tool is not trusted until it passes a verification battery. A GPU kernel
is the case where that discipline matters most, because a wrong kernel does not
crash — it returns plausible numbers fast. So the same way `ToolVerifier`
refuses to seal an unverified tool, `gpu.verify` refuses to report a kernel
number that has not been checked against a reference, for units, and for noise.

The split, and why each piece exists:

  probe    what silicon is here. Never raises; available=False is an answer.
  units    Duration, so a number cannot be printed with the wrong unit. The
           do_bench-returns-ms bug is structurally prevented, not documented.
  kernel   sources, a content-addressed compile cache, launch geometry,
           occupancy (estimated and measured).
  safety   pre-flight refusals and a hang guard. Bounds the blast radius
           without capping what the agent may attempt.
  bench    measurement with a spread, a verdict, and a baseline that must be
           forced down its fast path before a speedup is a claim.
  ops      the launch path and the references to verify against.

Nothing here requires a GPU to import. `probe()` returning available=False is a
first-class answer, and the static checks in `units.py` run on a CPU-only
laptop by design.
"""
from __future__ import annotations

from .bench import (
    SPREAD_NOISE_PCT,
    SPREAD_SIGNAL_PCT,
    BenchResult,
    Comparison,
    bandwidth_gbps,
    benchmark_kernel,
    check_units_by_wallclock,
    verify,
)
from .backend_triton import (
    TritonSignature,
    compile_auto,
    compile_triton,
    triton_available,
    triton_kernel_report,
)
from .kernel import (
    MAX_THREADS_PER_BLOCK,
    CompiledKernel,
    KernelCache,
    KernelSource,
    KernelUnavailable,
    LaunchConfig,
    Occupancy,
    cache_stats,
    compile_kernel,
    estimate_occupancy,
    measured_occupancy,
)
from .ops import (
    LaunchOutcome,
    bench_kernel,
    cpu_reference,
    launch,
    torch_baseline,
    torch_reference,
)
from .probe import DeviceInfo, arch_limits, probe, probe_all
from .safety import (
    DEFAULT_TIME_BUDGET_S,
    DevicePoisoned,
    HangGuard,
    Refusal,
    guard,
    preflight,
    reset_guard,
)
from .units import (
    Duration,
    MsOffence,
    UnitError,
    audit_ms_scale,
    audit_source_tree,
    from_cuda_event,
    from_do_bench,
)

__all__ = [
    # probe
    "DeviceInfo", "probe", "probe_all", "arch_limits", "ARCH_LIMITS",
    # units
    "Duration", "UnitError", "MsOffence", "audit_ms_scale", "audit_source_tree",
    "from_do_bench", "from_cuda_event",
    # kernel
    "KernelSource", "KernelUnavailable", "LaunchConfig", "Occupancy",
    "CompiledKernel", "KernelCache", "compile_kernel", "cache_stats",
    "estimate_occupancy", "measured_occupancy", "MAX_THREADS_PER_BLOCK",
    # safety
    "Refusal", "preflight", "HangGuard", "DevicePoisoned", "guard",
    "reset_guard", "DEFAULT_TIME_BUDGET_S",
    # bench
    "BenchResult", "Comparison", "benchmark_kernel", "verify",
    "bandwidth_gbps", "check_units_by_wallclock",
    "SPREAD_SIGNAL_PCT", "SPREAD_NOISE_PCT",
    # ops
    "LaunchOutcome", "launch", "bench_kernel", "cpu_reference",
    "torch_reference", "torch_baseline",
    # triton backend
    "TritonSignature", "compile_triton", "compile_auto", "triton_available",
    "triton_kernel_report", "measure_registers",
]
