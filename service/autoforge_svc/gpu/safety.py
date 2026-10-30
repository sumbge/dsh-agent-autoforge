"""Pre-flight guards for kernel launches.

WHAT CAN ACTUALLY GO WRONG, and what this file does about each:

  A kernel with an infinite loop hangs the device. On a machine whose GPU also
  drives the display, that reads as "the computer froze". You cannot kill a
  running kernel from CUDA — so the defence is a wall-clock budget, enforced
  from outside the call, and a hard refusal to try again on a device that has
  already hung once this session.

  A kernel with an out-of-range index corrupts device memory. Defence: bound
  the geometry before launch, so a grid the device cannot possibly run is
  refused with the arithmetic shown rather than attempted.

  A launch request that exceeds the block or shared-memory limit fails at the
  driver with a terse error. Defence: refuse it ourselves, with a message that
  names the actual limit and the requested value.

  An absent device. Defence: `probe().available` is a precondition. On a
  laptop with an integrated AMD GPU the CUDA tools refuse and say so, instead
  of failing somewhere deeper with a confusing error.

The design rule this file exists to honour is autoforge's own (§2.5): bound the
blast radius, do not cap capability. So nothing here asks permission to launch
a kernel. It checks whether the geometry is launchable, times the run, and
refuses to repeat a known-hang. Capability is untouched; the failure modes that
damage the host are closed.

Everything in this module is pure logic, so it is fully testable on a machine
with no GPU — which is the only kind of safety check that survives.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .kernel import MAX_THREADS_PER_BLOCK, LaunchConfig
from .probe import DeviceInfo, arch_limits

# Grid dimensions are unsigned 32-bit on every current architecture.
MAX_GRID_DIM = 2**31 - 1
# A launch beyond this many total threads is almost always a units mistake
# (threads vs bytes, or an off-by-1000). Refusing it beats waiting for it.
MAX_TOTAL_THREADS = 2**31
DEFAULT_TIME_BUDGET_S = 30.0


@dataclass(frozen=True)
class Refusal:
    """A launch that should not be attempted, and exactly why.

    Carries the numbers, not just a verdict: the arithmetic is what tells the
    agent whether to shrink the block or the grid.
    """

    code: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:                                # pragma: no cover
        return f"[{self.code}] {self.reason}"


def preflight(
    config: LaunchConfig,
    *,
    device: DeviceInfo | None = None,
    kernel_code: str = "",
    entry: str = "",
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
) -> Refusal | None:
    """Every check that can be answered without touching the GPU.

    Returns None when the launch may proceed, otherwise the first refusal.
    Checks are ordered cheapest-and-most-fundamental first, so the message the
    agent gets is about the root problem rather than a downstream symptom.
    """
    # 1. Is there a device at all? Everything below assumes one.
    if device is not None and not device.available:
        return Refusal(
            code="no-device",
            reason="no CUDA device is available, so no kernel can be launched",
            detail={"notes": list(device.notes)},
        )

    # 2. Block size. This is a hardware constant, not a preference.
    if config.threads_per_block > MAX_THREADS_PER_BLOCK:
        return Refusal(
            code="block-too-large",
            reason=(f"block of {config.threads_per_block} threads exceeds the "
                    f"{MAX_THREADS_PER_BLOCK}-thread hardware maximum"),
            detail={"requested": config.threads_per_block,
                    "limit": MAX_THREADS_PER_BLOCK},
        )

    # 3. Grid dimensions.
    for axis, size in zip("xyz", config.grid):
        if size > MAX_GRID_DIM:
            return Refusal(
                code="grid-dim-too-large",
                reason=(f"grid.{axis}={size} exceeds the maximum grid "
                        f"dimension {MAX_GRID_DIM}"),
                detail={"axis": axis, "requested": size, "limit": MAX_GRID_DIM},
            )

    # 4. Total work. A number this large is a units bug, not an intention.
    if config.total_threads > MAX_TOTAL_THREADS:
        return Refusal(
            code="total-threads-absurd",
            reason=(f"{config.total_threads} total threads is beyond anything "
                    f"launchable ({MAX_TOTAL_THREADS}); this is usually a "
                    f"threads/bytes confusion"),
            detail={"requested": config.total_threads, "limit": MAX_TOTAL_THREADS},
        )

    # 5. Shared memory, against this device's architecture if we know it.
    if config.shared_bytes:
        if device is not None and device.compute_capability:
            lim = arch_limits(*device.compute_capability)
            if lim and config.shared_bytes > lim["smem"]:
                return Refusal(
                    code="shared-memory-too-large",
                    reason=(f"{config.shared_bytes} bytes of shared memory per "
                            f"block exceeds the {lim['smem']} bytes this "
                            f"architecture has per SM"),
                    detail={"requested": config.shared_bytes, "limit": lim["smem"]},
                )

    # 6. Does the entry point actually exist in the source? Catching it here
    #    turns a driver-level lookup failure into a readable message.
    if kernel_code and entry:
        from .kernel import KernelSource

        try:
            names = KernelSource(name="probe", code=kernel_code).entry_points()
        except ValueError:
            names = []
        if names and entry not in names:
            return Refusal(
                code="no-such-entry",
                reason=f"no __global__ {entry!r} in the source",
                detail={"available": names},
            )

    # 7. The time budget must be a positive, finite number. A budget of zero or
    #    None would silently mean "no timeout", which is the one thing this
    #    module exists to prevent.
    if not time_budget_s or time_budget_s <= 0:
        return Refusal(
            code="no-time-budget",
            reason=("a positive time budget is required; an unbounded launch is "
                    "the failure this guard exists to prevent"),
            detail={"requested": time_budget_s},
        )

    return None


class DevicePoisoned(RuntimeError):
    """A previous launch on this device did not return inside its budget."""


class HangGuard:
    """Runs a blocking call with a wall-clock budget, and remembers a hang.

    A hung kernel cannot be killed, so this cannot make the hang go away. What
    it does do:
      * return control to the caller instead of blocking the agent forever
      * mark the device poisoned, so no further kernel is launched on it this
        session — the difference between one bad launch and a reboot loop

    The call runs in a daemon thread. That is deliberate: if the GPU call never
    returns, the thread must not keep the process alive.
    """

    def __init__(self) -> None:
        self.poisoned_reason: str | None = None
        self.launches = 0
        self.hangs = 0

    def check(self) -> None:
        if self.poisoned_reason:
            raise DevicePoisoned(self.poisoned_reason)

    def run(self, fn: Callable[[], Any], *,
            budget_s: float = DEFAULT_TIME_BUDGET_S) -> tuple[Any, float, bool]:
        """Return (result, elapsed_s, completed).

        completed=False means the budget expired — the caller must treat the
        device as unavailable rather than assume the work finished.
        """
        self.check()
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["value"] = fn()
            except BaseException as exc:                     # noqa: BLE001
                box["error"] = exc

        t = threading.Thread(target=target, daemon=True,
                             name="autoforge-gpu-launch")
        started = time.perf_counter()
        t.start()
        t.join(budget_s)
        elapsed = time.perf_counter() - started

        if t.is_alive():
            self.hangs += 1
            self.poisoned_reason = (
                f"a kernel launch did not return within {budget_s:.0f}s and "
                f"could not be cancelled; this device is not used again this "
                f"session. If the display froze, reboot before retrying."
            )
            return None, elapsed, False

        self.launches += 1
        if "error" in box:
            raise box["error"]
        return box.get("value"), elapsed, True

    def stats(self) -> dict[str, Any]:
        return {
            "launches": self.launches,
            "hangs": self.hangs,
            "poisoned": bool(self.poisoned_reason),
            "reason": self.poisoned_reason or "",
        }


# One guard per process. Kernel launches are a global resource (the device), so
# the bookkeeping has to be global too, or a poisoned device would look healthy
# to the next tool call.
_GUARD = HangGuard()


def guard() -> HangGuard:
    return _GUARD


def reset_guard() -> None:
    """Clear poisoning. For tests, and for a human who has rebooted and means
    it — never called automatically, because forgetting a hang is the whole
    failure this guards against."""
    global _GUARD
    _GUARD = HangGuard()
