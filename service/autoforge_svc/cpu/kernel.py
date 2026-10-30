"""Compiling a C kernel and loading it back.

The shape mirrors `autoforge.gpu.kernel` deliberately: same `KernelSource` /
`CompiledKernel` / `KernelCache` vocabulary, same content-addressed store, same
rule that a `CompiledKernel` is never returned unless a real artefact was
produced. A kernel forge whose two hardware layers say different things is two
forges to learn, so only the parts that genuinely differ are different:

  ARTEFACT. A shared library (`.dll` / `.so` / `.dylib`) rather than a cubin,
  because the CPU has no driver to load code into a device — the OS loader is
  the driver, and `ctypes` is the binding.

  IDENTITY. There is no `arch=sm_89` to key on, and the thing that replaces it
  is more dangerous, not less. A cubin for the wrong SM fails to load loudly. A
  binary built with `-march=native` on a newer CPU loads fine on an older one
  and then dies with SIGILL at the first wide instruction. So the resolved
  target and the compiler version are both in the cache key, and `probe.runs_here`
  is the check that keeps a cached artefact from being served to a machine that
  cannot execute it.

  FLAGS. `-O3` is the reason most "faster kernel" results exist, so the flags
  are part of the identity and are reported with every artefact. An unoptimised
  baseline is not a baseline, and `measure.Comparison` refuses to call a win
  against one a real claim.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import probe as probe_mod

#: Optimisation that is on unless a caller actively removes it. `-O2` rather
#: than `-O3` because `-O3` on a memory-bound loop mostly buys code size, and
#: the search in `tune` will find `-O3` on its own when it actually helps.
DEFAULT_OPT = ("-O2", "-funroll-loops", "-fno-math-errno")

#: Turned on only by explicit request, per kernel. OpenMP changes the artefact
#: and needs a runtime, so it is never silently added.
OMP_FLAG = "-fopenmp"


class KernelUnavailable(RuntimeError):
    """No compiler or no artefact. Carries the reason, never a bare failure."""

    def __init__(self, reason: str, *, remedy: str = "") -> None:
        self.reason = reason
        self.remedy = remedy
        super().__init__(reason if not remedy else f"{reason} ({remedy})")


#: Handles returned by `os.add_dll_directory`, held for the process lifetime.
#: Windows drops the search path again when the handle is garbage collected, so
#: a local variable here would silently undo itself at the next collection.
_DLL_DIR_HANDLES: list[Any] = []


def _add_compiler_dll_dir() -> str:
    """Put the compiler's `bin` directory on the DLL search path. Returns it, or "".

    Needed for any artefact linked against a runtime the compiler ships rather
    than the system: `libgomp-1.dll` for `-fopenmp`, and on mingw also the
    `libwinpthread` and gcc support DLLs. Without it an OpenMP kernel fails to
    load on a path that exists, and Windows names the missing *dependency* by
    reporting the module that could not be loaded — so the error points at our
    own freshly-built DLL instead of at libgomp.

    Linux needs something equivalent (an `RPATH` or `LD_LIBRARY_PATH`), but a
    `-fopenmp` build there finds libgomp through the standard search path, so
    this stays a Windows-only patch for a Windows-only trap.
    """
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return ""
    exe = getattr(probe_mod.probe(), "compiler", "") or ""
    if not exe:
        return ""
    bindir = Path(exe).parent
    if not bindir.is_dir():
        return ""
    try:
        _DLL_DIR_HANDLES.append(os.add_dll_directory(str(bindir)))
    except (OSError, AttributeError):                        # pragma: no cover
        return ""
    return str(bindir)


# -- sources ----------------------------------------------------------------
@dataclass(frozen=True)
class KernelSource:
    """C source plus everything that changes the compiled output.

    `target` is the `-march` value. `native` is allowed and recorded as-is, but
    `resolved_target()` is what goes into the cache key, because the word
    "native" describes a build machine rather than an instruction set and only
    looks like an identity.
    """

    name: str
    code: str
    target: str = "native"
    flags: tuple[str, ...] = ()
    openmp: bool = False

    def __post_init__(self) -> None:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", self.name):
            raise ValueError(
                f"kernel name {self.name!r} must be a C identifier — it becomes "
                f"part of a filename and a symbol lookup"
            )
        if not self.code.strip():
            raise ValueError("kernel code is empty")
        if self.openmp and OMP_FLAG not in self.effective_flags:
            # Written into the flags below; checked here so the dataclass stays
            # frozen and the identity cannot drift from the build command.
            pass

    @property
    def effective_flags(self) -> tuple[str, ...]:
        base = tuple(DEFAULT_OPT) + tuple(self.flags)
        if self.openmp and OMP_FLAG not in base:
            base = base + (OMP_FLAG,)
        return base

    @property
    def march_flag(self) -> str:
        return f"-march={self.target}"

    @property
    def build_command(self) -> tuple[str, ...]:
        """The flags a build will actually use, as a readable tuple."""
        return (self.march_flag,) + self.effective_flags

    def resolved_target(self) -> str:
        """The instruction set this will really be built for.

        `native` is replaced by the name the compiler resolves it to, so the
        cache key says `znver3` rather than `native`. Unresolvable stays honest
        and keys on "unresolved", which costs a cache hit but never a bad one.
        """
        if self.target != "native":
            return self.target
        return probe_mod.probe().arch or "unresolved"

    def cache_key(self) -> str:
        """sha256 over everything that affects the artefact.

        The compiler version is in here. It is the difference between a cache
        that survives a toolchain upgrade and one that silently serves a binary
        built by a different gcc — and the second is how a bug reappears after
        being fixed.
        """
        info = probe_mod.probe()
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(b"\0")
        h.update(self.resolved_target().encode())
        h.update(b"\0")
        h.update(self.compiler_fingerprint().encode())
        h.update(b"\0")
        h.update("\0".join(self.effective_flags).encode())
        h.update(b"\0")
        # Normalise trailing whitespace so a re-indent is a hit and a real edit
        # is not.
        h.update("\n".join(ln.rstrip() for ln in self.code.splitlines()).encode())
        del info
        return h.hexdigest()

    def compiler_fingerprint(self) -> str:
        info = probe_mod.probe()
        return f"{info.compiler_flavour}:{info.compiler_version or 'unknown'}"

    def entry_points(self) -> list[str]:
        """Functions this source exports, parsed without a compiler.

        Static functions and `main` are excluded because neither can be called
        through `ctypes`; everything else that looks like a definition is
        listed. Parsed rather than asked for so the agent can be told what it
        may call before anything is built.
        """
        found: list[str] = []
        for m in re.finditer(
            r"^[ \t]*(?!static\b|typedef\b|struct\b|union\b|enum\b)"
            r"([A-Za-z_][A-Za-z0-9_ \t*]*?)[ \t*]+([A-Za-z_][A-Za-z0-9_]*)"
            r"[ \t]*\([^;{]*\)[ \t]*\{",
            self.code, re.MULTILINE,
        ):
            name = m.group(2)
            if name not in ("main",) and name not in found:
                found.append(name)
        return found

    def has_entry(self, name: str) -> bool:
        return name in self.entry_points()


#: Suffix for the artefact on this platform.
def library_suffix() -> str:
    if sys.platform == "win32":
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


# -- compilation ------------------------------------------------------------
@dataclass
class CompiledKernel:
    """A built artefact plus the identity that produced it."""

    source: KernelSource
    artefact: Path
    cache_key: str
    from_cache: bool
    compiler: str                            # gcc | clang | cache
    stderr: str = ""
    _lib: Any = None

    @property
    def build_flags(self) -> tuple[str, ...]:
        return self.source.build_command

    def load(self):
        """`ctypes` handle, or a reason. Loaded once and memoised.

        The running-machine check is here rather than at compile time because
        this is the first moment the answer matters: a cached artefact built on
        a different box is fine to keep on disk and fatal to load.

        On Windows the compiler's own directory is added to the DLL search path
        first, and that is not a nicety — it is the difference between an OpenMP
        kernel running and not. `-fopenmp` links the artefact against
        `libgomp-1.dll`, which ships in the compiler's `bin` directory and is not
        on Python's search path, and Windows reports the missing *dependency* by
        naming the module that failed to load. So the error reads "Could not find
        module <our kernel>.dll" for a kernel that is sitting right there on
        disk, which sends you looking for the wrong file entirely.
        """
        if self._lib is not None:
            return self._lib
        ok, why = probe_mod.runs_here(self.source.resolved_target())
        if not ok:
            raise KernelUnavailable(f"cannot load {self.artefact.name}: {why}",
                                    remedy="rebuild on this machine")
        import ctypes
        _add_compiler_dll_dir()
        try:
            self._lib = ctypes.CDLL(str(self.artefact))
        except OSError as exc:
            raise KernelUnavailable(
                f"the loader refused {self.artefact.name}: {exc}",
                remedy=("check that the compiler and Python agree on word size"
                        + ("; for a parallel kernel also check that the OpenMP "
                           "runtime was found — the message names our kernel "
                           "even when the missing file is libgomp"
                           if self.source.openmp else "")),
            ) from exc
        return self._lib

    def call(self, name: str, *args: Any, restype: Any = None):
        """Call an exported function, with the symbol checked first.

        A `ctypes` call to a missing symbol raises `AttributeError` with no
        mention of the kernel, which is useless in a search loop. Naming the
        available exports is what turns it into a diagnosis.
        """
        if not self.source.has_entry(name) and name not in self.source.entry_points():
            raise KernelUnavailable(
                f"{name!r} is not exported by {self.source.name!r}; "
                f"available: {', '.join(self.source.entry_points()) or 'none'}"
            )
        lib = self.load()
        try:
            fn = getattr(lib, name)
        except AttributeError as exc:
            raise KernelUnavailable(
                f"{name!r} is declared in the source but not exported by the "
                f"artefact — it is probably `static`"
            ) from exc
        import ctypes
        fn.restype = restype
        return fn(*args)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.source.name,
            "target": self.source.resolved_target(),
            "flags": list(self.build_flags),
            "compiler": self.compiler,
            "cache_key": self.cache_key[:16],
            "from_cache": self.from_cache,
            "artefact": str(self.artefact),
            "entry_points": self.source.entry_points(),
            "openmp": self.source.openmp,
        }


class KernelCache:
    """Content-addressed store for compiled kernels.

    Disk-backed for the same reason the GPU one is: the search loop crosses
    process boundaries, and an in-memory cache would miss on every iteration of
    the thing it exists to speed up. Keyed by target *and* compiler, so one
    cache directory can serve several machines without ever serving one a
    binary it cannot run.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else Path(
            os.environ.get("AUTOFORGE_CPU_CACHE")
            or (Path(tempfile.gettempdir()) / "autoforge_cpu_kernels")
        )
        self.hits = 0
        self.misses = 0

    def path_for(self, source: KernelSource, suffix: str = "") -> Path:
        suffix = suffix or library_suffix()
        tag = source.compiler_fingerprint().replace(":", "-").replace(" ", "")
        return self.root / source.resolved_target() / tag / \
            f"{source.name}_{source.cache_key()[:16]}{suffix}"

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
        # Write then rename, so a concurrent reader never sees a half-written
        # library — and on Windows, never sees a file another process has open
        # for writing.
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
    cc: str | None = None,
    timeout: float = 120.0,
) -> CompiledKernel:
    """Compile to a shared library, or explain precisely why not.

    Never returns a `CompiledKernel` that was not actually produced: a cache hit
    must be a non-empty file, a miss must be a successful compiler run, and the
    library's existence is re-checked rather than inferred from the exit code.
    """
    cache = cache or _DEFAULT_CACHE
    info = probe_mod.probe()
    exe = cc or info.compiler

    cached = cache.get(source)
    if cached is not None:
        return CompiledKernel(source=source, artefact=cached,
                              cache_key=source.cache_key(), from_cache=True,
                              compiler="cache")

    if not exe:
        raise KernelUnavailable(
            "no C compiler on PATH and no cached artefact",
            remedy="install gcc or clang, or point AUTOFORGE_CPU_CACHE at a "
                   "populated cache directory",
        )

    # A target the toolchain was not asked about is a build that "works" and
    # then dies at run time. Checked before invoking, so the error names the
    # feature rather than the signal.
    ok, why = probe_mod.runs_here(source.resolved_target(), info)
    if not ok:
        raise KernelUnavailable(f"cannot build for {source.target}: {why}",
                               remedy="build with -march=native on this machine")

    suffix = library_suffix()
    with tempfile.TemporaryDirectory(prefix="autoforge_cpu_build_") as td:
        c = Path(td) / f"{source.name}.c"
        c.write_text(source.code, encoding="utf-8")
        out = Path(td) / f"{source.name}{suffix}"
        cmd = [exe, "-shared", "-fPIC", "-o", str(out), str(c),
               *source.build_command]
        if source.openmp and info.compiler_flavour in ("gnu", "clang"):
            cmd.append(OMP_FLAG)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, errors="replace")
        except subprocess.TimeoutExpired as exc:
            raise KernelUnavailable(
                f"{Path(exe).name} timed out after {timeout:.0f}s building "
                f"{source.name!r}"
            ) from exc
        if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            # The compiler's own diagnostic is what the agent needs, so it is
            # carried through rather than summarised away.
            detail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            raise KernelUnavailable(
                f"{Path(exe).name} failed for {source.name!r} "
                f"({' '.join(source.build_command)}):\n{detail}",
                remedy="fix the kernel source or the flags",
            )
        data = out.read_bytes()
        warnings = (proc.stderr or "").strip()

    p = cache.put(source, data)
    return CompiledKernel(source=source, artefact=p,
                          cache_key=source.cache_key(), from_cache=False,
                          compiler=Path(exe).name, stderr=warnings)


def cache_stats(cache: KernelCache | None = None) -> dict[str, Any]:
    return (cache or _DEFAULT_CACHE).stats()


__all__ = [
    "KernelUnavailable", "KernelSource", "CompiledKernel", "KernelCache",
    "compile_kernel", "cache_stats", "library_suffix",
    "DEFAULT_OPT", "OMP_FLAG",
]
