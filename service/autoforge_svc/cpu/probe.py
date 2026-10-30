"""What CPU silicon and toolchain is actually here.

`probe()` never raises. `available=False` is a first-class answer, not an error,
for the same reason the GPU layer does it: this code has to run on a laptop with
no compiler and on a rented box with three of them.

`available` on this side means something slightly different from the GPU side,
and the difference is worth stating. A GPU probe asks "is there hardware?"
Here it asks "can I build?" — because on a CPU the hardware is never in doubt
(you are running on it) and the thing that is actually missing is the compiler.
So `available` is a statement about the toolchain, and `notes` says which half
was found.

The feature list is read out of the compiler rather than out of the operating
system. That is deliberate: what matters to a kernel forge is not what the CPU
can do, it is what the toolchain will *emit* for it, and those are different
questions. `gcc -march=native -dM -E` answers the second one, and it is the same
answer the build will get. A /proc/cpuinfo flag list answers the first and can
disagree.

The two hazards this module exists to surface:

  CACHE. A CPU kernel that tiles for the wrong cache size is slower than a naive
  one, and the kernel author cannot see the size from inside the kernel. Reporting
  L1/L2/L3 here is what makes the tiling decision derivable instead of guessed.

  `-march=native`. The CPU analogue of building a cubin for the wrong SM, except
  worse: a cubin for a newer SM fails to load, while a native binary for a newer
  CPU *loads and then dies with SIGILL* on an older one. `native` is recorded as
  the target it actually resolved to, and `runs_here()` refuses an artefact built
  wider than this machine.
"""
from __future__ import annotations

import functools
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Compilers, in preference order. gcc first because it is the one that exists on
# a stock MSYS2 or a Linux box; clang when gcc is absent; cl.exe is listed
# because a Windows dev box will have it and its flags are entirely different,
# which is why `flavour` is tracked alongside the path rather than sniffed.
COMPILERS: tuple[tuple[str, str], ...] = (
    ("gcc", "gnu"), ("cc", "gnu"), ("clang", "clang"), ("cl", "msvc"), ("tcc", "tcc"),
)

#: Cache identity of an artefact, so a build can be refused on a machine that is
#: narrower than the one it was built for.
SIGILL_RISK = (
    "an artefact built for a wider instruction set loads fine and then dies with "
    "SIGILL on a narrower CPU; the target is part of the compile identity for "
    "that reason"
)


@dataclass
class CpuInfo:
    """The machine and the toolchain, with every field optional.

    A probe that answers half the questions is still worth recording, because
    the honest report is "we know this much".
    """

    available: bool = False                # a compiler was found
    name: str = ""
    logical_cores: int = 0
    physical_cores: int = 0
    features: list[str] = field(default_factory=list)
    arch: str = ""                         # the -march value 'native' resolves to
    cache_bytes: dict[str, int] = field(default_factory=dict)
    compiler: str = ""                     # path
    compiler_flavour: str = ""             # gnu | clang | msvc | tcc
    compiler_version: str = ""
    openmp: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def threads(self) -> int:
        """How many threads a kernel may use before it stops helping.

        Logical cores, because that is what the scheduler will actually give
        you. A compute-bound kernel wants `physical_cores`; a memory-bound one
        wants neither, and `tune` is where that gets decided by measurement
        rather than here by assertion.
        """
        return self.logical_cores or os.cpu_count() or 1

    def has(self, feature: str) -> bool:
        want = feature.strip().upper()
        return any(f.upper() == want for f in self.features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "name": self.name,
            "logical_cores": self.logical_cores,
            "physical_cores": self.physical_cores,
            "features": list(self.features),
            "arch": self.arch,
            "cache_bytes": dict(self.cache_bytes),
            "compiler": self.compiler,
            "compiler_flavour": self.compiler_flavour,
            "compiler_version": self.compiler_version,
            "openmp": self.openmp,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        if not self.available:
            why = "; ".join(self.notes) if self.notes else "no C compiler found"
            return f"No CPU toolchain usable for forging. {why}"
        cores = f"{self.physical_cores}C/{self.logical_cores}T" \
            if self.physical_cores else f"{self.logical_cores}T"
        cache = ", ".join(f"{k} {v // 1024}K" for k, v in self.cache_bytes.items())
        simd = ",".join(f for f in self.features
                        if f.upper() in ("AVX2", "AVX512F", "FMA", "SSE4.2")) or "baseline"
        return (f"{self.name or 'CPU'} ({cores}, {simd}"
                f"{', ' + cache if cache else ''}) via "
                f"{Path(self.compiler).name} {self.compiler_version}"
                f"{' +openmp' if self.openmp else ''}"
                f"; -march=native resolves to {self.arch or 'unknown'}")


def probe() -> CpuInfo:
    """Best effort, never raises, always returns something.

    Memoised. Every field costs a subprocess — the registry read, the powershell
    core count, three compiler invocations — and `KernelSource.cache_key()` asks
    for it once per candidate. Re-probing inside a search loop would spend more
    time in `gcc -###` than in the kernels being searched.

    The returned object is a copy, so a caller that annotates its own info (a
    test forcing a feature list) cannot poison the memo for everyone else.
    """
    return _copy(_probe_uncached())


def clear_probe_cache() -> None:
    """Drop the memo. For tests, and for a tool that has just installed a
    compiler and wants the new one found without restarting the process."""
    _probe_uncached.cache_clear()


def _copy(info: CpuInfo) -> CpuInfo:
    import copy as _copy_mod
    clone = _copy_mod.copy(info)
    clone.features = list(info.features)
    clone.notes = list(info.notes)
    clone.cache_bytes = dict(info.cache_bytes)
    return clone


@functools.lru_cache(maxsize=1)
def _probe_uncached() -> CpuInfo:
    info = CpuInfo()
    _identify(info)
    _count_cores(info)
    _caches(info)
    _find_compiler(info)
    if info.available:
        # Only meaningful once a compiler is known: this asks the compiler what
        # it will emit, which is the only feature list that matches the build.
        _probe_features_from_compiler(info)
    return info


def runs_here(target: str, info: CpuInfo | None = None) -> tuple[bool, str]:
    """May an artefact built for `target` be run on this machine?

    `target` is the -march value it was compiled with. `native` is treated as
    "the machine that built it", which is unknowable from the name alone, so the
    answer is a qualified yes with the qualifier spelled out rather than a
    silent pass. Everything else is compared as an ordered feature level, which
    is the honest approximation: it is enough to catch the mistake that actually
    happens (v3 artefact deployed to a v1 host) without pretending to model
    every CPU in existence.
    """
    info = info or probe()
    if target in ("", "x86-64", "generic"):
        return True, ""
    if target == "native":
        return True, ("built with -march=native, so it is specific to the "
                      "machine that built it; if this is a different machine "
                      "expect SIGILL, and rebuild rather than retry")
    if target == info.arch:
        # This is the machine's own resolved target, so it runs here by
        # construction. Checked before the lookup so that a CPU newer than
        # _ARCH_LEVELS does not refuse to load kernels it just built.
        return True, ""
    want = _arch_level(target)
    have = _arch_level(info.arch)
    if want is None or have is None:
        # Fail closed. An unknown name is most likely a target *newer* than this
        # table — `znver5`, `raptorlake` — which is precisely the dangerous
        # case, and the guard exists to prevent a SIGILL, so "I cannot prove it
        # runs" has to mean no. The cost of being wrong here is one rebuild; the
        # cost of the other direction is a dead interpreter.
        unknown = target if want is None else info.arch
        return False, (
            f"cannot place {unknown!r} on the microarchitecture scale, so there "
            f"is no way to prove a {target} artefact runs on {info.arch}; "
            f"rebuild for this machine (or add the name to _ARCH_LEVELS)")
    if want > have:
        return False, (f"built for {target} (level {want}) on a machine at level "
                       f"{have} ({info.arch}); this is the {SIGILL_RISK}")
    return True, ""


# Ordered x86-64 microarchitecture levels, which is the only portable
# comparison available across gcc's -march names.
_ARCH_LEVELS: dict[str, int] = {
    "x86-64": 1, "x86-64-v2": 2, "x86-64-v3": 3, "x86-64-v4": 4,
    "nehalem": 2, "westmere": 2, "sandybridge": 2, "ivybridge": 2,
    "haswell": 3, "broadwell": 3, "skylake": 3, "skylake-avx512": 4,
    "zen": 2, "zen2": 3, "zen3": 3, "zen4": 4,
    "znver1": 2, "znver2": 3, "znver3": 3, "znver4": 4,
    "core2": 1, "pentium4": 1, "atom": 1,
}


def _arch_level(name: str) -> int | None:
    n = (name or "").strip().lower()
    if n in _ARCH_LEVELS:
        return _ARCH_LEVELS[n]
    for token, level in _ARCH_LEVELS.items():
        if n.startswith(token):
            return level
    return None


# -- identification ----------------------------------------------------------
def _identify(info: CpuInfo) -> None:
    info.name = _cpu_name()
    info.arch = platform.machine()
    if sys.platform == "darwin":
        info.notes.append(
            "Apple silicon: `arch -arm64` and Rosetta change what `machine` "
            "means, so a forged kernel's target is checked at run time, not now"
        )


def _cpu_name() -> str:
    """The marketing name, per platform, with a fallback that is never wrong.

    Windows needs the registry: `platform.processor()` returns a family/model
    string there, which is useless in a report. Linux has /proc/cpuinfo. macOS
    has sysctl. Each is wrapped because a probe that raises is a probe nobody
    can call from a tool.
    """
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            with key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if value:
                    return str(value).strip()
        except Exception:                                    # noqa: BLE001
            pass
        return platform.processor()
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
        out = _run(["sysctl", "-n", "hw.model"])
        return out.strip() if out else platform.processor()
    try:
        text = Path("/proc/cpuinfo").read_text(errors="replace")
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


# -- cores and cache ---------------------------------------------------------
def _count_cores(info: CpuInfo) -> None:
    info.logical_cores = os.cpu_count() or 0
    try:
        info.logical_cores = len(os.sched_getaffinity(0)) or info.logical_cores
    except AttributeError:
        pass                                          # windows has no affinity API here
    if info.logical_cores:
        # cgroup/container limit, which is the number that actually binds on a
        # rented box and which os.cpu_count over-reports by an order of magnitude.
        quota = _cgroup_cpu_quota()
        if quota and quota < info.logical_cores:
            info.notes.append(
                f"cgroup allows {quota} CPUs but the host reports "
                f"{info.logical_cores}; using {quota} as the thread bound")
            info.logical_cores = quota
    info.physical_cores = _physical_cores(info.logical_cores)


def _cgroup_cpu_quota() -> int | None:
    """v2 then v1. None when unlimited, absent, or unreadable."""
    for path, scale in ((Path("/sys/fs/cgroup/cpu.max"), None),
                        (Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), 100_000)):
        try:
            text = path.read_text().split()
        except OSError:
            continue
        if path.name == "cpu.max":
            if len(text) == 2 and text[0] != "max":
                try:
                    return max(1, int(text[0]) // int(text[1]))
                except (ValueError, ZeroDivisionError):
                    return None
        else:
            try:
                quota = int(text[0])
            except (IndexError, ValueError):
                continue
            if quota > 0:
                return max(1, quota // scale)
    return None


def _physical_cores(logical: int) -> int:
    """Real cores, not hyperthreads. 0 when it cannot be determined."""
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor).NumberOfCores"], timeout=20)
        try:
            return int(out.strip().splitlines()[0])
        except (ValueError, IndexError, AttributeError):
            return 0
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.physicalcpu"])
        try:
            return int(out.strip())
        except (ValueError, AttributeError):
            return 0
    try:
        text = Path("/proc/cpuinfo").read_text(errors="replace")
        cores = set()
        phys = core = None
        for line in text.splitlines() + [""]:
            if line.startswith("physical id"):
                phys = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core = line.split(":", 1)[1].strip()
            elif not line.strip() and phys is not None and core is not None:
                cores.add((phys, core))
                phys = core = None
        return len(cores)
    except OSError:
        return 0


def _caches(info: CpuInfo) -> None:
    """L1d/L2/L3 in bytes, keyed the way a tiler wants to read them.

    Absent keys are absent on purpose: a kernel built for a guessed L2 is worse
    than one built for an unknown L2, because the guess is silent.
    """
    if sys.platform == "linux":
        base = Path("/sys/devices/system/cpu/cpu0/cache")
        try:
            for entry in sorted(base.glob("index*")):
                kind = (entry / "level").read_text().strip()
                what = (entry / "type").read_text().strip()
                size = (entry / "size").read_text().strip()
                if what.lower() == "instruction":
                    continue
                m = re.match(r"^(\d+)([KMG])$", size.upper())
                if not m:
                    continue
                mult = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[m.group(2)]
                info.cache_bytes[f"L{kind}"] = int(m.group(1)) * mult
            return
        except Exception:                                    # noqa: BLE001
            pass
    if sys.platform == "darwin":
        for key, label in (("hw.l1dcachesize", "L1"), ("hw.l2cachesize", "L2"),
                           ("hw.l3cachesize", "L3")):
            out = _run(["sysctl", "-n", key])
            if out and out.strip().isdigit():
                info.cache_bytes[label] = int(out.strip())
        return
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor) | "
                    "ForEach-Object { \"$($_.L2CacheSize) $($_.L3CacheSize)\" }"],
                   timeout=20)
        try:
            l2, l3 = out.strip().splitlines()[0].split()
            if l2.isdigit() and int(l2):
                info.cache_bytes["L2"] = int(l2) * 1024
            if l3.isdigit() and int(l3):
                info.cache_bytes["L3"] = int(l3) * 1024
        except Exception:                                    # noqa: BLE001
            pass


# -- toolchain ---------------------------------------------------------------
def _find_compiler(info: CpuInfo) -> None:
    for name, flavour in COMPILERS:
        path = shutil.which(name)
        if not path:
            continue
        info.compiler = path
        info.compiler_flavour = flavour
        info.available = True
        info.compiler_version = _compiler_version(path, flavour)
        info.openmp = _openmp_available(path, flavour)
        return
    info.notes.append(
        "no C compiler on PATH (looked for "
        + ", ".join(n for n, _ in COMPILERS)
        + "); install gcc or clang to forge CPU kernels"
    )


def _compiler_version(path: str, flavour: str) -> str:
    flag = "/?" if flavour == "msvc" else "--version"
    out = _run([path, flag], timeout=20)
    if flavour == "msvc":
        return "present (version needs the developer prompt)" if out else "present"
    first = (out or "").strip().splitlines()
    if not first:
        return ""
    m = re.search(r"(\d+\.\d+\.\d+)", first[0])
    return m.group(1) if m else first[0][:40]


def _openmp_available(path: str, flavour: str) -> bool:
    """Compile a one-line OpenMP probe. Asked, not assumed.

    Assuming it is how a kernel gets written that quietly runs single-threaded
    on the machine that has no libgomp, which then reads as "the parallel
    version was not faster".
    """
    if flavour not in ("gnu", "clang"):
        return flavour == "msvc"
    flag = "-fopenmp" if flavour == "gnu" else "-fopenmp"
    with _scratch() as td:
        src = td / "omp.c"
        src.write_text("#include <omp.h>\n"
                       "int main(void){return omp_get_max_threads()<0;}\n")
        out = td / "omp.exe"
        proc = _spawn([path, flag, str(src), "-o", str(out)], timeout=60)
        return bool(proc and proc.returncode == 0 and out.exists())


def _probe_features_from_compiler(info: CpuInfo) -> None:
    """Ask the compiler what it emits for this machine.

    `-dM -E` dumps the preprocessor's macro table, which contains exactly the
    `__AVX2__`-style defines the build will see. This is the ground truth a
    kernel author needs, and it comes from the same tool that will compile the
    kernel, so the two cannot disagree.
    """
    if info.compiler_flavour == "msvc":
        info.notes.append(
            "MSVC does not report its target ISA through the preprocessor; "
            "focus on /arch:AVX2 and /openmp rather than a feature list")
        return
    with _scratch() as td:
        empty = td / "empty.c"
        empty.write_text("")
        proc = _spawn([info.compiler, "-march=native", "-dM", "-E", str(empty)],
                      timeout=60)
        if not proc or proc.returncode != 0:
            info.notes.append(
                "-march=native is not accepted by this compiler; targets must "
                "be named explicitly (a portable build is still possible)")
            return
        defines = dict(re.findall(r"#define\s+(__[A-Za-z0-9_]+__)\s+(\S+)",
                                  proc.stdout or ""))
    if "__AVX2__" in defines:
        info.features.append("AVX2")
    if "__AVX512F__" in defines:
        info.features.append("AVX512F")
    if "__FMA__" in defines:
        info.features.append("FMA")
    if "__SSE4_2__" in defines:
        info.features.append("SSE4.2")
    if "__AVX__" in defines and "AVX2" not in info.features:
        info.features.append("AVX")
    info.arch = _native_arch(info) or info.arch


def _native_arch(info: CpuInfo) -> str:
    """The -march name `native` resolves to, per ISA vendor.

    The whole point is to replace the word "native" — which means "the machine
    that built this" and therefore nothing at all on the machine that runs it —
    with a name that can be compared.

    Two reads, because the first one has a trap in it. `-###` prints the driver's
    own command line before the cc1 line it expands to, and the driver's line
    says `-march=native` — the very word being replaced. Taking the first match
    yields "native", so matches are collected and the literal is skipped.
    """
    resolved = _march_from_verbose(info)
    if resolved:
        return resolved
    # `-Q --help=target` prints the resolved value with no command lines around
    # it, which is a cleaner read when -### output changes shape between gcc
    # versions. It is the fallback rather than the first choice because clang
    # accepts -### and not -Q.
    with _scratch() as td:
        empty = td / "e.c"
        empty.write_text("")
        proc = _spawn([info.compiler, "-march=native", "-Q", "--help=target"],
                      timeout=60)
    if proc and proc.returncode == 0:
        for line in (proc.stdout or "").splitlines():
            m = re.match(r"\s+-march=\s+(\S+)\s*$", line)
            if m and m.group(1).lower() not in ("native", ""):
                return m.group(1)
    if "Ryzen" in (info.name or "") or "AMD" in (info.name or ""):
        # A Ryzen without a resolvable -march is at least v3 by name, but that
        # is an inference and is labelled as one by being the last resort only.
        return "x86-64-v3" if info.has("AVX2") else ""
    return ""


def _march_from_verbose(info: CpuInfo) -> str:
    with _scratch() as td:
        empty = td / "e.c"
        empty.write_text("")
        proc = _spawn([info.compiler, "-march=native", "-###", "-E", str(empty)],
                      timeout=60)
    text = (proc.stderr or "") if proc else ""
    for candidate in re.findall(r"-march=([A-Za-z0-9_.+-]+)", text):
        if candidate.lower() != "native":
            return candidate
    return ""


# -- helpers -----------------------------------------------------------------
def _run(cmd: list[str], timeout: float = 15) -> str:
    proc = _spawn(cmd, timeout=timeout)
    return (proc.stdout or "") if proc and proc.returncode == 0 else ""


def _spawn(cmd: list[str], timeout: float = 15):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, errors="replace")
    except Exception:                                        # noqa: BLE001
        return None


class _scratch:
    """A temporary directory that cleans up even when the compiler dies."""

    def __enter__(self) -> Path:
        import tempfile
        self._td = tempfile.TemporaryDirectory(prefix="autoforge_cpu_probe_")
        return Path(self._td.name)

    def __exit__(self, *exc: Any) -> None:
        self._td.cleanup()


__all__ = ["CpuInfo", "probe", "runs_here", "COMPILERS", "SIGILL_RISK"]
