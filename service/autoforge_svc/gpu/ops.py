"""Running kernels, and the reference implementations to judge them against.

The launch wrapper is thin on purpose. Everything interesting lives in the
guards (`safety.preflight`, `safety.HangGuard`) and in the measurement
(`bench`); this module's job is to route a launch through both so that no
caller can accidentally bypass them.

`torch_reference()` and the CPU references exist for one reason: `verify()`
needs something to compare against, and the alternative — trusting the kernel
that is being tested — is what makes a fast-but-wrong kernel ship.

Everything degrades to `KernelUnavailable`/None on a machine with no GPU. The
reference paths are pure Python so they run everywhere, including CI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .backend_triton import TritonSignature, compile_auto
from .bench import BenchResult, benchmark_kernel, verify
from .kernel import (
    CompiledKernel,
    KernelSource,
    KernelUnavailable,
    LaunchConfig,
    compile_kernel,
)
from .probe import probe
from .safety import DEFAULT_TIME_BUDGET_S, HangGuard, Refusal, guard, preflight
from .units import Duration


@dataclass
class LaunchOutcome:
    """What a launch attempt produced: a result, or a refusal, or nothing."""

    ok: bool
    elapsed: Duration | None = None
    refusal: Refusal | None = None
    error: str = ""
    kernel: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    occupied: str = ""

    def summary(self) -> str:
        if self.refusal is not None:
            return f"refused: {self.refusal}"
        if not self.ok:
            return f"failed: {self.error}"
        bits = [f"launched in {self.elapsed.format('us')}"
                if self.elapsed else "launched"]
        if self.occupied:
            bits.append(self.occupied)
        return "  ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "elapsed_ms": self.elapsed.ms if self.elapsed else None,
            "refusal": ({"code": self.refusal.code, "reason": self.refusal.reason,
                         "detail": self.refusal.detail} if self.refusal else None),
            "error": self.error,
            "kernel": self.kernel,
            "config": self.config,
            "occupied": self.occupied,
        }


def launch(
    source: KernelSource,
    config: LaunchConfig,
    *,
    entry: str = "",
    args: Sequence[Any] | None = None,
    sig: TritonSignature | None = None,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    cache: Any = None,
    _runner: Callable[[CompiledKernel, LaunchConfig, list[Any]], Any] | None = None,
) -> LaunchOutcome:
    """Compile (cached), guard, launch, time.

    Order matters and is the point of this function:

      1. probe      — is there a device? refuse early, cheaply.
      2. preflight  — is this geometry launchable at all? refuse with arithmetic.
      3. guard.check— has this device already hung this session? refuse.
      4. compile    — cached by content, backend chosen by `compile_auto`, so a
                      box with no nvcc still compiles via Triton.
      5. guard.run  — launch under a wall-clock budget, in a daemon thread.

    `_runner` is the injection point for tests: it stands in for the driver
    call, so the guard logic can be exercised on a machine with no GPU. It is
    underscore-prefixed because no production caller should pass it.
    """
    device = probe()
    if not device.available:
        return LaunchOutcome(
            ok=False,
            refusal=Refusal(code="no-device",
                            reason="no CUDA device is available",
                            detail={"notes": list(device.notes)}),
        )

    entries = source.entry_points()
    target = entry or (entries[0] if entries else "")
    refusal = preflight(config, device=device, kernel_code=source.code,
                        entry=target, time_budget_s=time_budget_s)
    if refusal is not None:
        return LaunchOutcome(ok=False, refusal=refusal)

    g: HangGuard = guard()
    try:
        g.check()
    except Exception as exc:                                 # noqa: BLE001
        return LaunchOutcome(ok=False, refusal=Refusal(
            code="device-poisoned", reason=str(exc)))

    try:
        compiled = compile_auto(source, sig, entry=target, cache=cache)
    except KernelUnavailable as exc:
        return LaunchOutcome(ok=False, error=str(exc))

    runner = _runner or _default_runner
    try:
        _value, elapsed, completed = g.run(
            lambda: runner(compiled, config, list(args or [])),
            budget_s=time_budget_s,
        )
    except Exception as exc:                                 # noqa: BLE001
        return LaunchOutcome(ok=False, error=f"{type(exc).__name__}: {exc}",
                             kernel=compiled.to_dict(), config=config.to_dict())

    if not completed:
        return LaunchOutcome(
            ok=False,
            elapsed=Duration.from_seconds(elapsed),
            error=(f"did not complete within {time_budget_s:.0f}s; device "
                   f"poisoned for this session"),
            kernel=compiled.to_dict(), config=config.to_dict(),
        )

    return LaunchOutcome(
        ok=True,
        elapsed=Duration.from_seconds(elapsed),
        kernel=compiled.to_dict(),
        config=config.to_dict(),
    )


def _default_runner(compiled: CompiledKernel, config: LaunchConfig,
                    args: list[Any]) -> Any:
    """The real driver call. Raises KernelUnavailable when there is no binding."""
    raise KernelUnavailable(
        "no CUDA runtime binding is installed to actually launch this kernel",
        remedy="use the torch path (torch_baseline) or install cuda-python on "
               "the GPU host",
    )


def bench_kernel(
    source: KernelSource,
    config: LaunchConfig,
    *,
    entry: str = "",
    label: str = "",
    sig: TritonSignature | None = None,
    warmup: int = 3,
    reps: int = 5,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    _fn: Callable[[], Any] | None = None,
) -> BenchResult:
    """Benchmark a kernel's launch, under the same guards as `launch`.

    A benchmark is `launch` repeated, so it would be a mistake to let it skip
    the guards. It does not: the guards run first, and a refusal aborts the
    benchmark with the refusal as the error and no timings at all — an empty
    timing list is the only honest report for "this was never launched".

    `_fn` replaces the launch call itself, for testing the guard path on a
    machine with no GPU.
    """
    device = probe()
    if not device.available:
        return BenchResult(
            timings=[], label=label or source.name,
            error="no CUDA device is available, so nothing was benchmarked",
        )
    entries = source.entry_points()
    refusal = preflight(config, device=device, kernel_code=source.code,
                        entry=entry or (entries[0] if entries else ""),
                        time_budget_s=time_budget_s)
    if refusal is not None:
        return BenchResult(timings=[], label=label or source.name,
                           error=str(refusal))

    if _fn is None:
        outcome = launch(source, config, entry=entry, sig=sig,
                         time_budget_s=time_budget_s)
        if not outcome.ok:
            return BenchResult(
                timings=[], label=label or source.name,
                error=(str(outcome.refusal) if outcome.refusal else outcome.error),
            )

        def one() -> Any:
            return launch(source, config, entry=entry, sig=sig,
                          time_budget_s=time_budget_s).elapsed
    else:
        def one() -> Any:
            return _fn()

    return benchmark_kernel(one, label=label or source.name,
                            warmup=warmup, reps=reps)


# -- references -------------------------------------------------------------
def cpu_reference(kind: str, *arrays: Sequence[float], **kw: Any) -> list[float]:
    """Pure-Python references, so `verify` works with no GPU and no torch.

    These are deliberately the slow, obvious version. They are the definition
    of correct that a fast kernel is measured against, and an optimised
    reference would be a second thing to debug.
    """
    k = kind.lower()
    if k == "add":
        a, b = arrays[0], arrays[1]
        return [float(x) + float(y) for x, y in zip(a, b)]
    if k == "mul":
        a, b = arrays[0], arrays[1]
        return [float(x) * float(y) for x, y in zip(a, b)]
    if k == "scale":
        a = arrays[0]
        s = float(kw["s"])
        return [float(x) * s for x in a]
    if k == "relu":
        a = arrays[0]
        return [x if x > 0 else 0.0 for x in a]
    if k == "softmax":
        a = [float(x) for x in arrays[0]]
        if not a:
            return []
        m = max(a)
        exps = [math.exp(x - m) for x in a]
        total = sum(exps)
        return [e / total for e in exps]
    if k == "matmul":
        a, b = arrays[0], arrays[1]
        n = int(kw["n"])
        out = []
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for p in range(n):
                    acc += float(a[i * n + p]) * float(b[p * n + j])
                out.append(acc)
        return out
    raise ValueError(f"no cpu reference for {kind!r}")


def torch_reference(kind: str, *arrays: Any, **kw: Any) -> Any:
    """The torch equivalent, when torch is present.

    Returns None rather than raising when torch is absent, because "no torch
    here" is a normal state on the CPU-only laptop this is developed on.
    """
    try:
        import torch
    except Exception:                                        # noqa: BLE001
        return None
    k = kind.lower()
    ts = [torch.as_tensor(a, dtype=torch.float32) for a in arrays]
    if k == "add":
        return ts[0] + ts[1]
    if k == "mul":
        return ts[0] * ts[1]
    if k == "scale":
        return ts[0] * float(kw["s"])
    if k == "relu":
        return torch.clamp(ts[0], min=0)
    if k == "softmax":
        return torch.softmax(ts[0], dim=-1)
    if k == "matmul":
        n = int(kw["n"])
        return ts[0].reshape(n, n) @ ts[1].reshape(n, n)
    raise ValueError(f"no torch reference for {kind!r}")


def torch_baseline(kind: str, n: int, *, reps: int = 5, warmup: int = 3,
                   label: str = "", **kw: Any) -> BenchResult:
    """Time torch's own implementation of the same operation.

    This is the baseline a hand-written kernel must beat to be worth keeping,
    and the honest framing matters: on a laptop with no CUDA torch this returns
    an error result rather than a CPU timing presented as a GPU baseline.
    """
    try:
        import torch
    except Exception as exc:                                 # noqa: BLE001
        return BenchResult(timings=[], label=label or kind,
                           error=f"torch is not installed ({exc})")
    if not torch.cuda.is_available():
        return BenchResult(
            timings=[], label=label or kind,
            error=("torch reports no CUDA device; a CPU timing would not be a "
                   "GPU baseline, so none is reported"),
        )

    def one() -> Duration:
        dev = torch.device("cuda")
        if kind == "softmax":
            x = torch.randn(n, device=dev)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            torch.softmax(x, dim=-1)
            e.record()
            torch.cuda.synchronize()
            return Duration.from_ms(s.elapsed_time(e))
        a = torch.randn(n * n, device=dev)
        b = torch.randn(n * n, device=dev)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        if kind == "matmul":
            a.reshape(n, n) @ b.reshape(n, n)
        elif kind == "add":
            a + b
        elif kind == "relu":
            torch.clamp(a, min=0)
        e.record()
        torch.cuda.synchronize()
        return Duration.from_ms(s.elapsed_time(e))

    return benchmark_kernel(one, label=label or kind, warmup=warmup, reps=reps)


__all__ = [
    "LaunchOutcome",
    "launch",
    "bench_kernel",
    "cpu_reference",
    "torch_reference",
    "torch_baseline",
    "verify",
]
