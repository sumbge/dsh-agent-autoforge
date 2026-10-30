"""The Triton compile backend.

WHY THIS EXISTS — found by testing on the actual rented box, not by design.

`kernel.py` compiles CUDA C with `nvcc`. On the AutoDL image that this layer
was built to run against, **there is no nvcc**: the toolkit is not installed,
only the runtime that PyTorch ships. Triton, however, bundles its own `ptxas`
and can JIT a kernel with no toolkit at all. So on the exact hardware this
layer targets, the nvcc path is dead and the Triton path is the only one that
compiles anything.

That is a real gap, not a hypothetical: verified on an RTX 4090 D where
`shutil.which("nvcc")` is None and Triton compiles fine.

Triton kernels are Python, not CUDA C, so this is a separate front end rather
than a flag on the other one. A Triton source is:

  * exec'd to obtain the `@triton.jit` function object (Triton's AST front end
    needs the function, not the text)
  * compiled through `triton.compile`, which runs its own ptxas
  * cached by content, in the same store the nvcc path uses

The artefact is the cubin Triton produced, plus the PTX and the register
count Triton reports — that last number is what feeds `kernel.estimate_occupancy`
so the estimate is anchored to something the compiler actually said rather than
to a guess.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .kernel import (
    CompiledKernel,
    KernelCache,
    KernelSource,
    KernelUnavailable,
)


def triton_available() -> tuple[bool, str]:
    """(available, version_or_reason). Never raises."""
    try:
        import triton                                  # noqa: F401
    except Exception as exc:                           # noqa: BLE001
        return False, f"triton is not installed ({type(exc).__name__}: {exc})"
    try:
        import triton.language  # noqa: F401
        from triton.compiler import compile as _c     # noqa: F401
    except Exception as exc:                           # noqa: BLE001
        return False, f"triton is installed but not importable ({exc})"
    import triton as _t
    return True, getattr(_t, "__version__", "unknown")


@dataclass(frozen=True)
class TritonSignature:
    """The argument types and constexprs a Triton kernel is specialised on.

    Part of the kernel's identity, exactly like `arch` on the CUDA side: a
    Triton kernel compiled for N=1024 is a different binary from the same
    source compiled for N=4096, and serving one for the other silently gives
    wrong answers. So this goes into the cache key.
    """

    signature: dict[str, str]                    # name -> "fp16" | "*fp32" | "i32"
    constexprs: dict[str, int | bool | float]
    num_warps: int = 4
    num_stages: int = 3

    def cache_suffix(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self.signature):
            h.update(f"{k}:{self.signature[k]};".encode())
        h.update(b"|")
        for k in sorted(self.constexprs):
            h.update(f"{k}={self.constexprs[k]};".encode())
        h.update(f"|w{self.num_warps}s{self.num_stages}".encode())
        return h.hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": dict(self.signature),
            "constexprs": dict(self.constexprs),
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


_JIT_FN_RE = re.compile(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _is_jit_function(obj: Any) -> bool:
    """Is this a Triton JIT kernel?

    Triton's JITFunction has moved module and name across versions, so this
    checks for the attribute Triton's own compiler uses to identify one
    (`cache_key` plus `run`), rather than importing a private path that a
    version bump would break.
    """
    return callable(obj) and hasattr(obj, "cache_key") and hasattr(obj, "run")


def _load_jit_function(code: str, name: str):
    """Return the source's `@triton.jit` function.

    FOUND ON THE REAL BOX: Triton refuses a kernel that was `exec`'d from a
    string. Its front end calls `inspect.getsource` on the function to build the
    AST, and that requires a real file on disk:

        ValueError: @jit functions should be defined in a Python file

    So the source is written to a temp module and imported, rather than exec'd.
    The temp file is what Triton reads; it is cleaned up immediately after,
    because Triton has already captured the AST and the compiled artefact is
    cached by content.

    Selection order, chosen so ambiguity is always an error rather than a
    silent pick:
      1. an explicit `name` that exists in the module namespace
      2. exactly one JIT function in the namespace
      3. otherwise refuse, listing the candidates
    """
    import importlib.util
    import tempfile

    ns: dict[str, Any] = {}
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(code)
            tmp_path = fh.name
        spec = importlib.util.spec_from_file_location(
            f"_autoforge_triton_{abs(hash(code)) & 0xFFFFFF}", tmp_path)
        if spec is None or spec.loader is None:          # pragma: no cover
            raise KernelUnavailable("could not build a module spec for the source")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ns = vars(mod)
    except KernelUnavailable:
        raise
    except Exception as exc:                           # noqa: BLE001
        raise KernelUnavailable(
            f"Triton source failed to import: {type(exc).__name__}: {exc}",
            remedy="the source must be valid Python defining a @triton.jit kernel",
        ) from exc
    finally:
        if tmp_path:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if name and name in ns and _is_jit_function(ns[name]):
        return ns[name]

    jit_fns = {k: v for k, v in ns.items() if _is_jit_function(v)}
    if len(jit_fns) == 1:
        return next(iter(jit_fns.values()))
    if not jit_fns:
        defs = sorted(k for k, v in ns.items()
                      if callable(v) and not k.startswith("_"))
        raise KernelUnavailable(
            f"no @triton.jit function in Triton source (module-level callables: "
            f"{defs or 'none'})",
            remedy="decorate the kernel with @triton.jit, or name it in `entry`",
        )
    raise KernelUnavailable(
        f"Triton source defines {len(jit_fns)} kernels "
        f"({', '.join(sorted(jit_fns))}); pass `entry` to choose one",
        remedy="ambiguous entry point",
    )


def compile_triton(
    source: KernelSource,
    sig: TritonSignature,
    *,
    entry: str = "",
    cache: KernelCache | None = None,
    target: Any = None,
) -> CompiledKernel:
    """Compile Triton Python source to a cubin via Triton's own ptxas.

    Returns a real artefact on success. Raises KernelUnavailable with the
    compiler's own diagnostic on failure — the same contract as the nvcc path,
    so a caller cannot tell which backend ran except by the `compiler` field.
    """
    ok, why = triton_available()
    if not ok:
        raise KernelUnavailable(why, remedy="pip install triton")

    from triton.compiler import compile as triton_compile
    from triton.compiler.compiler import ASTSource

    cache = cache or KernelCache()
    src_key = source.cache_key()
    sig_key = sig.cache_suffix()
    key = f"{src_key}_{sig_key}"
    # Both halves must survive into the filename. Taking key[:16] here would
    # keep only the source hash — the source is a 16-char digest, so the
    # signature suffix would be silently truncated away and two different
    # constexpr specialisations (BLOCK=1024 vs 2048) would share one artefact,
    # serving the wrong binary. Caught on the real box, not in review.
    artefact_path = (cache.root / source.arch_flag /
                     f"{source.name}_{src_key[:8]}_{sig_key[:8]}.cubin")
    meta_path = artefact_path.with_suffix(".json")

    if artefact_path.exists() and artefact_path.stat().st_size > 0:
        return CompiledKernel(
            source=source, artefact=artefact_path, cache_key=key,
            from_cache=True, compiler="triton",
        )

    fn = _load_jit_function(source.code, entry or source.name)
    ast = ASTSource(fn=fn, signature=sig.signature, constexprs=sig.constexprs)

    kwargs: dict[str, Any] = {
        "options": {"num_warps": sig.num_warps, "num_stages": sig.num_stages}
    }
    if target is not None:
        kwargs["target"] = target

    try:
        compiled = triton_compile(ast, **kwargs)
    except Exception as exc:                           # noqa: BLE001
        detail = str(exc)
        raise KernelUnavailable(
            f"Triton compile failed for {source.name!r} "
            f"(warps={sig.num_warps}, stages={sig.num_stages}):\n{detail[-2000:]}",
            remedy="check the signature/constexpr keys match the kernel's params",
        ) from exc

    asm = getattr(compiled, "asm", {}) or {}
    cubin = asm.get("cubin")
    if not cubin:
        raise KernelUnavailable(
            f"Triton returned no cubin for {source.name!r} "
            f"(asm keys: {sorted(asm)})",
            remedy="this triton build may only emit ptx",
        )

    artefact_path.parent.mkdir(parents=True, exist_ok=True)
    artefact_path.write_bytes(cubin)

    # The metadata Triton itself reports. n_regs is the number the static
    # occupancy estimate needs to be anchored to reality.
    md = getattr(compiled, "metadata", None)
    import json
    meta: dict[str, Any] = {"signature": sig.to_dict()}
    if md is not None:
        for field in ("num_warps", "num_stages", "shared", "name",
                      "n_regs", "n_spills", "group_size_0"):
            val = getattr(md, field, None)
            if val is not None:
                try:
                    meta[field] = int(val)
                except (TypeError, ValueError):
                    meta[field] = str(val)
    if "n_regs" not in meta and hasattr(compiled, "n_regs"):
        meta["n_regs"] = int(getattr(compiled, "n_regs"))
    try:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError:
        pass

    return CompiledKernel(
        source=source, artefact=artefact_path, cache_key=key,
        from_cache=False, compiler="triton",
    )


def compile_auto(
    source: KernelSource,
    sig: TritonSignature | None = None,
    *,
    entry: str = "",
    cache: KernelCache | None = None,
) -> CompiledKernel:
    """Compile with whichever backend this machine actually has.

    Order, and the reason for it:

      1. Triton, when the source looks like Triton (it imports `triton`) and
         Triton is installed. On a rented box this is the ONLY working path —
         Triton bundles ptxas, and there is usually no nvcc.
      2. nvcc, for CUDA C.

    If neither can run, the error names both reasons instead of only the one
    the last attempt produced. That matters because on a rented box the
    actionable advice is "use the Triton path" — and an error that only says
    "nvcc not found" would hide that.
    """
    looks_triton = "triton" in source.code and sig is not None
    if looks_triton:
        return compile_triton(source, sig, entry=entry, cache=cache)

    import shutil

    have_nvcc = shutil.which("nvcc") is not None
    if have_nvcc:
        from .kernel import compile_kernel

        return compile_kernel(source, cache=cache)

    t_ok, t_why = triton_available()
    reasons = ["no nvcc on PATH, so the CUDA C path cannot run"]
    if sig is None:
        reasons.append(
            "no Triton signature was supplied, so the Triton path cannot be used "
            "either — pass a TritonSignature with the kernel's arg types"
        )
    elif not t_ok:
        reasons.append(f"Triton is unusable: {t_why}")
    raise KernelUnavailable(
        "nothing on this machine can compile a kernel: " + "; ".join(reasons),
        remedy=("on a rented GPU box use the Triton path: install triton "
                "(usually present with torch) and supply a TritonSignature"),
    )


def measure_registers(source: KernelSource, sig: TritonSignature,
                      entry: str = "") -> dict[str, Any]:
    """The register count Triton's ptxas actually chose, for this source+sig.

    WHY THIS IS SEPARATE FROM `compile_triton`: the artefact on disk is a cubin,
    and the register count is not recoverable from it by reading bytes. It lives
    on Triton's in-memory compiled object, and on most Triton versions it is only
    populated once the module handles are initialised — i.e. after the kernel has
    been loaded onto a device. So this deliberately compiles again and loads,
    which requires a live CUDA context. That is why it is a reporting call and
    not part of the hot compile path: `compile_triton` must stay usable at
    compile time on a machine that will ship the cubin elsewhere.

    Returns {"measured": bool, "n_regs": int, "n_spills": int, "reason": str}.
    Never raises — an unmeasurable register count is a fact to report, not an
    error, because the occupancy estimate degrades to the threads limit and
    says so.
    """
    ok, why = triton_available()
    if not ok:
        return {"measured": False, "n_regs": 0, "n_spills": 0, "reason": why}
    try:
        from triton.compiler import compile as triton_compile
        from triton.compiler.compiler import ASTSource
    except Exception as exc:                           # noqa: BLE001
        return {"measured": False, "n_regs": 0, "n_spills": 0,
                "reason": f"triton compiler API unavailable: {exc}"}
    try:
        fn = _load_jit_function(source.code, entry or source.name)
        ast = ASTSource(fn=fn, signature=sig.signature, constexprs=sig.constexprs)
        compiled = triton_compile(
            ast, options={"num_warps": sig.num_warps,
                          "num_stages": sig.num_stages})
    except Exception as exc:                           # noqa: BLE001
        return {"measured": False, "n_regs": 0, "n_spills": 0,
                "reason": f"compile failed: {type(exc).__name__}: {exc}"}

    n_regs = getattr(compiled, "n_regs", None)
    if n_regs is None:
        # Populate the handles: this is where ptxas' register allocation becomes
        # readable. Guarded because it needs a device and varies by version.
        init = getattr(compiled, "_init_handles", None)
        if callable(init):
            try:
                init()
            except Exception as exc:                   # noqa: BLE001
                return {"measured": False, "n_regs": 0, "n_spills": 0,
                        "reason": (f"register count needs a live device and "
                                   f"loading the module failed: {exc}")}
        n_regs = getattr(compiled, "n_regs", None)

    if n_regs is None:
        return {"measured": False, "n_regs": 0, "n_spills": 0,
                "reason": "this Triton build does not expose n_regs"}
    return {
        "measured": True,
        "n_regs": int(n_regs),
        "n_spills": int(getattr(compiled, "n_spills", 0) or 0),
        "reason": "",
    }


def triton_kernel_report(source: KernelSource, sig: TritonSignature,
                         entry: str = "") -> dict[str, Any]:
    """What the kernel looks like before and after compilation.

    Used by the tool layer so the agent sees the register count and shared
    memory Triton decided on — the two numbers that turn a hand-waved
    occupancy argument into an arithmetic one.
    """
    ok, why = triton_available()
    if not ok:
        return {"available": False, "reason": why}
    try:
        k = compile_triton(source, sig, entry=entry)
    except KernelUnavailable as exc:
        return {"available": True, "compiled": False, "error": str(exc)}
    md_path = k.artefact.with_suffix(".json")
    meta: dict[str, Any] = {}
    if md_path.exists():
        import json
        try:
            meta = json.loads(md_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    return {
        "available": True,
        "compiled": True,
        "cache_key": k.cache_key,
        "from_cache": k.from_cache,
        "artefact": str(k.artefact),
        "bytes": k.artefact.stat().st_size if k.artefact.exists() else 0,
        "metadata": meta,
        "registers": measure_registers(source, sig, entry=entry),
    }
