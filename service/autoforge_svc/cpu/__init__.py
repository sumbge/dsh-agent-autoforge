"""Forging CPU kernels: probe the toolchain, build, prove, measure, search.

The sibling of `autoforge.gpu`, and the two are meant to read the same way — the
same `KernelSource` / `CompiledKernel` / `KernelCache` vocabulary, the same
`Duration`, the same verify-before-measure rule. The hardware differs, and the
parts that genuinely differ are called out rather than smoothed over:

    probe   what silicon and toolchain is here, and whether an artefact built
            for a target can run on this machine (the SIGILL check)
    kernel  C source to a loadable shared library, content-addressed on target,
            compiler version and flags
    safety  a source lint, and an isolated child process so a segfault or an
            infinite loop is data rather than the end of the session
    ops     a call described as data, with a mandatory reference answer, so
            "verify it" is not a step that can be skipped
    tune    the search: generate, compile, verify, measure, mutate, keep the
            winner — with a baseline strong enough that a win means something

The order of these is the argument. `probe` before `kernel` because a build for
a target this machine cannot run is worse than no build. `safety` before `ops`
because the first call to an unproven kernel must not be in the agent's own
process. And in `tune`, verification before measurement, always — a search that
times first selects for the kernel that skipped the most work, which is the one
failure mode a fast hardware layer cannot detect on its own.
"""
from __future__ import annotations

from .kernel import (
    DEFAULT_OPT,
    OMP_FLAG,
    CompiledKernel,
    KernelCache,
    KernelSource,
    KernelUnavailable,
    cache_stats,
    compile_kernel,
    library_suffix,
)
from .ops import (
    DTYPES,
    TOLERANCES,
    TYPECODES,
    Buffer,
    CallError,
    CallSpec,
    Problem,
    RunResult,
    Scalar,
    close,
    evaluate,
    invoke,
    marshal,
    problem,
    run,
    size_t,
)
from .probe import CpuInfo, SIGILL_RISK, clear_probe_cache, probe, runs_here
from .safety import (
    DEFAULT_TIMEOUT,
    Finding,
    GuardResult,
    Preflight,
    preflight,
    run_isolated,
)
from .tune import (
    DEFAULT_BUDGET_S,
    DEFAULT_MAX_CANDIDATES,
    FAST_BASELINE_FLAGS,
    NAIVE,
    Candidate,
    SearchResult,
    Variant,
    flag_variants,
    naive,
    source_variants,
    tune,
    tune_kind,
    variants,
)

__all__ = [
    # probe
    "CpuInfo", "probe", "runs_here", "clear_probe_cache", "SIGILL_RISK",
    # kernel
    "KernelSource", "CompiledKernel", "KernelCache", "KernelUnavailable",
    "compile_kernel", "cache_stats", "library_suffix", "DEFAULT_OPT", "OMP_FLAG",
    # safety
    "Finding", "Preflight", "preflight", "GuardResult", "run_isolated",
    "DEFAULT_TIMEOUT",
    # ops
    "DTYPES", "TOLERANCES", "TYPECODES", "CallError", "Buffer", "Scalar",
    "CallSpec", "RunResult", "Problem", "run", "close", "problem", "evaluate",
    "size_t", "marshal", "invoke",
    # tune
    "Variant", "Candidate", "SearchResult", "tune", "tune_kind", "variants",
    "flag_variants", "source_variants", "naive", "NAIVE",
    "FAST_BASELINE_FLAGS", "DEFAULT_MAX_CANDIDATES", "DEFAULT_BUDGET_S",
]
