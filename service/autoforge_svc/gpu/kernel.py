"""Kernel sources, the compile cache, launch geometry, and occupancy.

The layer below "run a command that happens to invoke nvcc". Here a kernel is
a value with an identity — source text, target architecture, compiler flags —
and its compiled artefact is content-addressed by that identity. Compiling the
same kernel twice is a cache lookup, which matters because kernel iteration is
a loop of small edits and a cold nvcc is 10-30 s.

Two things are deliberately split, because conflating them is how a
performance claim becomes unfalsifiable:

  estimate_occupancy()  pure arithmetic from the architecture limits. Runs
                        anywhere, no GPU. Fast, approximate, and labelled as
                        such.
  measured_occupancy()  asks the driver. Needs a device. This is the number
                        you quote.

When the estimate and the measurement disagree, the measurement wins and the
estimate is recorded as having been wrong — that disagreement is the only
signal that the static limits table is stale for a new architecture.

Nothing here raises on import. On a machine with no toolkit, `compile_kernel`
raises `KernelUnavailable` with the reason attached, which is a value the agent
can report rather than a crash it has to swallow.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .probe import arch_limits, probe


class KernelUnavailable(RuntimeError):
    """No compiler or no driver. Carries the reason, never a bare failure."""

    def __init__(self, reason: str, *, remedy: str = "") -> None:
        self.reason = reason
        self.remedy = remedy
        msg = reason if not remedy else f"{reason} ({remedy})"
        super().__init__(msg)


# -- sources ----------------------------------------------------------------
@dataclass(frozen=True)
class KernelSource:
    """A kernel plus everything that changes its compiled output.

    `arch` and `flags` are part of the identity, not metadata: a cubin built
    for sm_75 must never be served to an sm_89 request. Getting that wrong
    produces a load error at best and a silently-wrong run at worst, so the
    cache key covers it.
    """

    name: str
    code: str                              # CUDA C, may contain many __global__
    arch: tuple[int, int] = (8, 9)
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", self.name):
            raise ValueError(
                f"kernel name {self.name!r} must be a C identifier — it becomes "
                f"part of a filename and a symbol lookup"
            )
        if not self.code.strip():
            raise ValueError("kernel code is empty")

    @property
    def arch_flag(self) -> str:
        return f"sm_{self.arch[0]}{self.arch[1]}"

    @property
    def sm_flag(self) -> str:
        return f"-arch={self.arch_flag}"

    def cache_key(self) -> str:
        """sha256 over everything that affects the artefact."""
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(b"\0")
        h.update(self.arch_flag.encode())
        h.update(b"\0")
        h.update("\0".join(self.flags).encode())
        h.update(b"\0")
        # Normalise trailing whitespace so a re-indent is a cache hit but a
        # real edit is not.
        h.update("\n".join(ln.rstrip() for ln in self.code.splitlines()).encode())
        return h.hexdigest()

    def entry_points(self) -> list[str]:
        """`__global__` symbols declared in this source.

        Parsed rather than asked for, so the agent can be told what it is about
        to launch without a compiler in the loop.
        """
        return re.findall(
            r"__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)", self.code
        )

    def has_entry(self, name: str) -> bool:
        return name in self.entry_points()


# -- launch geometry --------------------------------------------------------
@dataclass(frozen=True)
class LaunchConfig:
    """Grid and block geometry, plus the shared memory the launch requests."""

    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (256, 1, 1)
    shared_bytes: int = 0

    def __post_init__(self) -> None:
        for label, dims in (("grid", self.grid), ("block", self.block)):
            if len(dims) != 3:
                raise ValueError(f"{label} must be 3 dimensions, got {dims!r}")
            if any(int(d) < 1 for d in dims):
                raise ValueError(f"{label} dimensions must all be >= 1, got {dims!r}")
        if self.shared_bytes < 0:
            raise ValueError("shared_bytes cannot be negative")

    @property
    def threads_per_block(self) -> int:
        return self.block[0] * self.block[1] * self.block[2]

    @property
    def total_blocks(self) -> int:
        return self.grid[0] * self.grid[1] * self.grid[2]

    @property
    def total_threads(self) -> int:
        return self.threads_per_block * self.total_blocks

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": list(self.grid),
            "block": list(self.block),
            "shared_bytes": self.shared_bytes,
            "threads_per_block": self.threads_per_block,
            "total_blocks": self.total_blocks,
            "total_threads": self.total_threads,
        }


# Threads per SM is the divisor for occupancy; block sizes above this can never
# launch at all, on any current architecture.
MAX_THREADS_PER_BLOCK = 1024


@dataclass
class Occupancy:
    """Occupancy, with a flag saying whether it was measured or derived."""

    blocks_per_sm: int
    threads_per_sm: int
    max_threads_per_sm: int
    limiting_factor: str
    measured: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        if not self.max_threads_per_sm:
            return 0.0
        return self.threads_per_sm / self.max_threads_per_sm

    def summary(self) -> str:
        how = "measured" if self.measured else "estimated"
        return (f"{self.ratio * 100:.1f}% {how} "
                f"({self.blocks_per_sm} blocks/SM x {self.threads_per_sm // max(1, self.blocks_per_sm)} "
                f"threads, limited by {self.limiting_factor})")


def estimate_occupancy(
    config: LaunchConfig,
    *,
    arch: tuple[int, int] = (8, 9),
    regs_per_thread: int = 0,
    regs_granularity: int = 8,
) -> Occupancy | None:
    """Arithmetic occupancy from the architecture's static limits.

    Returns None for an architecture whose limits we do not have. That is the
    honest answer: inventing limits for an unseen SM would yield a confident
    occupancy figure with nothing behind it.

    Register granularity is modelled because it materially changes the answer —
    the hardware allocates registers per warp in units (typically 8 for the
    threads-per-SM product), so 33 registers costs the same as 40. Skipping
    that rounding is the most common way a hand occupancy calculation is
    optimistic.

    This is an estimate. `measured_occupancy()` is the number to publish.
    """
    lim = arch_limits(*arch)
    if lim is None:
        return None

    threads = config.threads_per_block
    if threads > MAX_THREADS_PER_BLOCK:
        raise ValueError(
            f"block of {threads} threads exceeds the {MAX_THREADS_PER_BLOCK} "
            f"hardware maximum — it cannot launch on any current architecture"
        )

    by_threads = lim["threads"] // threads

    by_smem = lim["blocks"] if not config.shared_bytes else (
        lim["smem"] // config.shared_bytes
    )

    if regs_per_thread:
        # Round up to the allocation granularity before dividing: the hardware
        # gives a warp blocks of registers, not individuals.
        rounded = -(-regs_per_thread // regs_granularity) * regs_granularity
        per_block = rounded * threads
        by_regs = lim["regs"] // per_block if per_block else lim["blocks"]
    else:
        by_regs = lim["blocks"]

    candidates = {
        "threads per SM": by_threads,
        "shared memory per SM": by_smem,
        "registers per SM": by_regs,
        "resident blocks per SM": lim["blocks"],
    }
    blocks = max(0, min(candidates.values()))
    # Which limit to blame when two tie: report the one that would move first if
    # it were relaxed. Ties broken by this order so the answer is deterministic.
    limiting = min(candidates, key=lambda k: (candidates[k], k))
    return Occupancy(
        blocks_per_sm=blocks,
        threads_per_sm=blocks * threads,
        max_threads_per_sm=lim["threads"],
        limiting_factor=limiting,
        measured=False,
        detail={"candidates": candidates,
                "regs_per_thread_rounded": (
                    -(-regs_per_thread // regs_granularity) * regs_granularity
                    if regs_per_thread else 0)},
    )


def measured_occupancy(config: LaunchConfig, kernel: "CompiledKernel") -> Occupancy | None:
    """Ask the driver. Returns None when no runtime is available.

    Calls cudaOccupancyMaxActiveBlocksPerMultiprocessor through whichever
    binding is present. Kept next to the estimator so the two can be compared:
    a persistent disagreement means ARCH_LIMITS needs updating for that SM.
    """
    try:
        from cuda import cuda as _cuda          # cuda-python
    except Exception:                            # noqa: BLE001
        _cuda = None
    if _cuda is None:
        return None
    info = probe()
    lim = arch_limits(*info.compute_capability) if info.compute_capability else None
    if lim is None:
        return None
    try:
        bound = kernel._module_function()         # raises if not loadable
        err, blocks = _cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            bound, config.threads_per_block, config.shared_bytes
        )
        if err != _cuda.CUresult.CUDA_SUCCESS:
            return None
        return Occupancy(
            blocks_per_sm=int(blocks),
            threads_per_sm=int(blocks) * config.threads_per_block,
            max_threads_per_sm=lim["threads"],
            limiting_factor="driver",
            measured=True,
        )
    except KernelUnavailable:
        return None
    except Exception:                            # noqa: BLE001
        return None


# -- compilation ------------------------------------------------------------
@dataclass
class CompiledKernel:
    """A compiled artefact plus the identity that produced it."""

    source: KernelSource
    artefact: Path
    cache_key: str
    from_cache: bool
    compiler: str                            # nvcc | nvrtc
    cubin: bool = True
    _handle: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.source.name,
            "arch": self.source.arch_flag,
            "compiler": self.compiler,
            "cache_key": self.cache_key[:16],
            "from_cache": self.from_cache,
            "artefact": str(self.artefact),
            "entry_points": self.source.entry_points(),
        }

    def _module_function(self):
        raise KernelUnavailable(
            "no CUDA driver binding available to load this module",
            remedy="install cuda-python, pycuda or cupy on the GPU host",
        )


class KernelCache:
    """Content-addressed store for compiled kernels.

    Disk-backed on purpose: the agent's kernel iteration loop crosses process
    boundaries (a tool call compiles, a later tool call launches), so an
    in-memory-only cache would miss every time in exactly the workflow it
    exists to speed up.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else Path(
            os.environ.get("AUTOFORGE_KERNEL_CACHE")
            or (Path(tempfile.gettempdir()) / "autoforge_kernels")
        )
        self.hits = 0
        self.misses = 0

    def path_for(self, source: KernelSource, suffix: str = ".cubin") -> Path:
        return self.root / source.arch_flag / f"{source.name}_{source.cache_key()[:16]}{suffix}"

    def get(self, source: KernelSource) -> Path | None:
        p = self.path_for(source)
        if p.exists() and p.stat().st_size > 0:
            self.hits += 1
            return p
        self.misses += 1
        return None

    def put(self, source: KernelSource, data: bytes) -> Path:
        p = self.path_for(source)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write then rename, so a concurrent reader never sees a partial cubin.
        tmp = p.with_suffix(p.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(p)
        return p

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "root": str(self.root),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
        }


_DEFAULT_CACHE = KernelCache()


def compile_kernel(
    source: KernelSource,
    *,
    cache: KernelCache | None = None,
    nvcc: str | None = None,
    timeout: float = 120.0,
) -> CompiledKernel:
    """Compile to a cubin, or explain precisely why not.

    Never returns a CompiledKernel that was not actually produced. A cache hit
    is verified by the artefact existing and being non-empty; a miss must be a
    successful nvcc run, and the cubin's presence is re-checked rather than
    assumed from the exit code.
    """
    cache = cache or _DEFAULT_CACHE
    exe = nvcc or shutil.which("nvcc")

    cached = cache.get(source)
    if cached is not None:
        return CompiledKernel(source=source, artefact=cached,
                              cache_key=source.cache_key(), from_cache=True,
                              compiler="cache")

    if not exe:
        raise KernelUnavailable(
            "no nvcc on PATH and no cached artefact",
            remedy="install the CUDA toolkit on the GPU host, or point "
                   "AUTOFORGE_KERNEL_CACHE at a populated cache directory",
        )

    with tempfile.TemporaryDirectory(prefix="autoforge_kernel_") as td:
        cu = Path(td) / f"{source.name}.cu"
        cu.write_text(source.code, encoding="utf-8")
        out = Path(td) / f"{source.name}.cubin"
        cmd = [exe, source.sm_flag, "-cubin", "-o", str(out), str(cu),
               "-lineinfo", *source.flags]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise KernelUnavailable(
                f"nvcc timed out after {timeout:.0f}s"
            ) from exc
        if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            # nvcc reports the diagnostic the agent actually needs, so it is
            # carried through rather than summarised away.
            detail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            raise KernelUnavailable(
                f"nvcc failed for {source.name!r} ({source.arch_flag}):\n{detail}",
                remedy="fix the kernel source or the arch flag",
            )
        data = out.read_bytes()

    p = cache.put(source, data)
    return CompiledKernel(source=source, artefact=p,
                          cache_key=source.cache_key(), from_cache=False,
                          compiler="nvcc")


def cache_stats(cache: KernelCache | None = None) -> dict[str, Any]:
    return (cache or _DEFAULT_CACHE).stats()
