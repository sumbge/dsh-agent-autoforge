"""Device discovery: what silicon is actually here.

`probe()` never raises. `available=False` is a first-class answer, not an
error, because the same code path has to run on a laptop with an integrated
AMD GPU and on a rented RTX 4080. A layer that only works when a GPU is
present cannot be tested anywhere, so it never gets tested.

Backends are tried in order of how much they tell us per unit of setup:

  torch.cuda   — richest (name, cc, SM count, memory, driver/runtime) and the
                 most likely to be present on a rented box
  cupy         — same, when cupy is the installed path
  nvidia-smi   — works with no Python GPU library at all, which is the state a
                 fresh container is often in
  nvcc         — last resort: tells us a toolkit exists, not that a device does

Every field is optional. A backend that answers half the questions is still
worth recording, because the honest report is "we know this much".
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

# Static per-architecture limits, used only by estimate_occupancy() when the
# real cudaOccupancy call is unavailable. Keyed by (major, minor).
# Values are threads/SM, blocks/SM, registers/SM, shared memory/SM in bytes.
# Sourced from the NVIDIA CUDA C Programming Guide's compute-capabilities
# table; treat as an estimate and prefer the measured value when present.
ARCH_LIMITS: dict[tuple[int, int], dict[str, int]] = {
    (7, 0): {"threads": 1024, "blocks": 32, "regs": 65536, "smem": 98304},
    (7, 5): {"threads": 1024, "blocks": 16, "regs": 65536, "smem": 65536},
    (8, 0): {"threads": 2048, "blocks": 32, "regs": 65536, "smem": 167936},
    (8, 6): {"threads": 1536, "blocks": 16, "regs": 65536, "smem": 102400},
    (8, 9): {"threads": 1536, "blocks": 24, "regs": 65536, "smem": 102400},
    (9, 0): {"threads": 2048, "blocks": 32, "regs": 65536, "smem": 233472},
    (10, 0): {"threads": 2048, "blocks": 32, "regs": 65536, "smem": 233472},
    (12, 0): {"threads": 2048, "blocks": 32, "regs": 65536, "smem": 233472},
}


def arch_limits(major: int, minor: int) -> dict[str, int] | None:
    """Limits for a compute capability, or None if we have never seen it.

    None is deliberate: inventing limits for an unknown architecture would
    produce a confident occupancy number that is simply wrong.
    """
    return ARCH_LIMITS.get((major, minor))


@dataclass
class DeviceInfo:
    """What we could find out, and what we could not.

    `notes` is not decoration — it is how a caller tells "no GPU in this box"
    apart from "there is a GPU but I could not talk to it", which need
    different fixes.
    """

    available: bool = False
    backend: str = ""                    # torch | cupy | nvidia-smi | nvcc
    count: int = 0
    name: str = ""
    compute_capability: tuple[int, int] | None = None
    sm_count: int = 0
    total_memory_bytes: int = 0
    driver_version: str = ""
    runtime_version: str = ""
    nvcc_version: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def smem_per_sm(self) -> int:
        lim = arch_limits(*self.compute_capability) if self.compute_capability else None
        return lim["smem"] if lim else 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "available": self.available,
            "backend": self.backend,
            "count": self.count,
            "name": self.name,
            "sm_count": self.sm_count,
            "total_memory_bytes": self.total_memory_bytes,
            "driver_version": self.driver_version,
            "runtime_version": self.runtime_version,
            "nvcc_version": self.nvcc_version,
            "notes": list(self.notes),
        }
        d["compute_capability"] = (
            f"{self.compute_capability[0]}.{self.compute_capability[1]}"
            if self.compute_capability else None
        )
        if self.compute_capability and not arch_limits(*self.compute_capability):
            d["notes"].append(
                f"no static limits for sm_{self.compute_capability[0]}"
                f"{self.compute_capability[1]}; occupancy estimate unavailable"
            )
        return d

    def summary(self) -> str:
        if not self.available:
            why = "; ".join(self.notes) if self.notes else "no CUDA device found"
            return f"No CUDA device. {why}"
        cc = f"sm_{self.compute_capability[0]}{self.compute_capability[1]}" \
            if self.compute_capability else "unknown arch"
        mem = f"{self.total_memory_bytes / 2**30:.1f} GiB" if self.total_memory_bytes else "?"
        return (f"{self.name} x{self.count} ({cc}, {self.sm_count} SMs, {mem}) "
                f"via {self.backend}; driver {self.driver_version or '?'}, "
                f"runtime {self.runtime_version or '?'}")


def probe() -> DeviceInfo:
    """First device, or an unavailable DeviceInfo explaining why."""
    return probe_all()[0]


def probe_all() -> list[DeviceInfo]:
    """Devices from the first backend that answers, plus a no-device entry.

    Returns a single-element list [DeviceInfo(available=False, ...)] when
    nothing is found, so callers can always index [0].
    """
    for attempt in (_from_torch, _from_cupy, _from_nvidia_smi):
        info = attempt()
        if info is not None:
            return [info]
    # Nothing claimed a device. Record what toolchain exists anyway — "there is
    # no nvcc either" is a materially different situation from "nvcc is here,
    # the driver is not".
    notes: list[str] = []
    nvcc = _nvcc_version()
    if nvcc:
        notes.append(f"nvcc {nvcc} present but no device backend answered "
                     f"(driver missing or no GPU)")
    else:
        notes.append("no CUDA backend answered and no nvcc on PATH")
    return [DeviceInfo(available=False, nvcc_version=nvcc, notes=notes)]


# -- backends ---------------------------------------------------------------
def _from_torch() -> DeviceInfo | None:
    try:
        import torch
    except Exception:                                        # noqa: BLE001
        return None
    try:
        if not torch.cuda.is_available():
            # A CPU-only torch build is extremely common on laptops and is
            # worth distinguishing from "torch absent" for the notes.
            return None
        count = torch.cuda.device_count()
        if count < 1:
            return None
        props = torch.cuda.get_device_properties(0)
        cc = (props.major, props.minor)
        info = DeviceInfo(
            available=True,
            backend="torch",
            count=count,
            name=props.name,
            compute_capability=cc,
            sm_count=getattr(props, "multi_processor_count", 0),
            total_memory_bytes=getattr(props, "total_memory", 0),
            driver_version=getattr(props, "driver_version", "") or "",
            runtime_version=getattr(torch.version, "cuda", "") or "",
            nvcc_version=_nvcc_version(),
        )
        if not arch_limits(*cc):
            info.notes.append(f"unrecognised compute capability {cc[0]}.{cc[1]}")
        return info
    except Exception as exc:                                 # noqa: BLE001
        return DeviceInfo(available=False, backend="torch",
                          notes=[f"torch present but cuda enumeration failed: "
                                 f"{type(exc).__name__}: {exc}"])


def _from_cupy() -> DeviceInfo | None:
    try:
        import cupy
    except Exception:                                        # noqa: BLE001
        return None
    try:
        count = cupy.cuda.runtime.getDeviceCount()
        if count < 1:
            return None
        props = cupy.cuda.runtime.getDeviceProperties(0)
        cc = (int(props["major"]), int(props["minor"]))
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        info = DeviceInfo(
            available=True,
            backend="cupy",
            count=count,
            name=name,
            compute_capability=cc,
            sm_count=int(props.get("multiProcessorCount", 0)),
            total_memory_bytes=int(props.get("totalGlobalMem", 0)),
            runtime_version=str(cupy.cuda.runtime.runtimeGetVersion()),
            nvcc_version=_nvcc_version(),
        )
        if not arch_limits(*cc):
            info.notes.append(f"unrecognised compute capability {cc[0]}.{cc[1]}")
        return info
    except Exception:                                        # noqa: BLE001
        return None


def _from_nvidia_smi() -> DeviceInfo | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,compute_cap,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:                                        # noqa: BLE001
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    first = [p.strip() for p in lines[0].split(",")]
    name = first[0] if first else "unknown"
    cc: tuple[int, int] | None = None
    if len(first) > 1 and re.match(r"^\d+\.\d+$", first[1]):
        major, minor = first[1].split(".")
        cc = (int(major), int(minor))
    mem = 0
    if len(first) > 2 and first[2].isdigit():
        mem = int(first[2]) * 2**20            # MiB -> bytes
    return DeviceInfo(
        available=True,
        backend="nvidia-smi",
        count=len(lines),
        name=name,
        compute_capability=cc,
        total_memory_bytes=mem,
        driver_version=first[3] if len(first) > 3 else "",
        nvcc_version=_nvcc_version(),
        notes=["nvidia-smi only: SM count and runtime version unavailable "
               "without a Python GPU library"],
    )


_NVCC_RE = re.compile(r"release\s+(\d+\.\d+)")


def _nvcc_version() -> str:
    exe = shutil.which("nvcc")
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, "--version"], capture_output=True,
                             text=True, timeout=15)
    except Exception:                                        # noqa: BLE001
        return ""
    m = _NVCC_RE.search(out.stdout or "")
    return m.group(1) if m else "present (version unparsed)"
