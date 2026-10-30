"""Not letting a kernel take the process down with it.

This is the layer the GPU side does not need and the CPU side cannot do
without, and the reason is structural. A bad CUDA kernel fails at a driver call
that returns an error code. A bad C kernel does one of three things the agent
cannot survive:

  IT SEGFAULTS. A missing bounds check writes past the end of a buffer and the
  interpreter dies with the agent's whole session in it. No exception is raised,
  no finally block runs, nothing is logged.

  IT NEVER RETURNS. `while (i <= n)` with an off-by-one instead of `<` is an
  infinite loop over a buffer, and in-process there is no way to interrupt it
  that does not also kill the agent.

  IT CORRUPTS THE HEAP. The failure surfaces somewhere unrelated, minutes later,
  in code that is not the kernel.

So a kernel is never called in the agent's process until it has been called in
somebody else's. `run_isolated` is that somebody: a fresh `python -m` child with
a hard timeout, whose death — by signal, by exit code, by timeout — is reported
as data. The parent then decides whether to try again, and never dies of it.

`preflight` is the cheap half: a source scan that catches the handful of things
worth knowing before spending a compile on them. It is a lint, not a sandbox,
and it says so — the isolation is what makes running an unchecked kernel
survivable, and preflight is what makes reading the result less surprising.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..timing import Duration
from . import ops as ops_mod

#: Default wall-clock ceiling for one isolated run, in seconds. Generous enough
#: for a slow build of a big problem, tight enough that a hang does not look
#: like a hung agent. Overridable per call and by AUTOFORGE_CPU_TIMEOUT.
DEFAULT_TIMEOUT = 120.0

#: Windows reports a fault as an NTSTATUS value in the exit code rather than as
#: a negative signal number, so a crash there reads as `exit code 3221225477`
#: with no hint of what happened. These are the ones worth naming.
WINDOWS_FAULTS: dict[int, str] = {
    0xC0000005: "access violation (segfault: a read or write outside a buffer)",
    0xC00000FD: "stack overflow (unbounded recursion, or a huge stack array)",
    0xC0000374: "heap corruption (a write past the end of a malloc'd buffer)",
    0xC000001D: "illegal instruction (an artefact built for a wider CPU than "
                "this one — SIGILL)",
    0xC0000094: "integer division by zero",
    0xC0000096: "privileged instruction",
    0xC0000135: "a required DLL was not found",
}


@dataclass
class Finding:
    """One thing wrong with a kernel source, with the fix attached."""

    code: str
    message: str
    severity: str = "warn"                   # block | warn | info
    line: int = 0
    remedy: str = ""

    def __str__(self) -> str:
        where = f" (line {self.line})" if self.line else ""
        why = f" — {self.remedy}" if self.remedy else ""
        return f"[{self.severity}] {self.code}: {self.message}{where}{why}"


@dataclass
class Preflight:
    """The findings, plus whether anything should stop the build."""

    findings: list[Finding] = field(default_factory=list)

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "block"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warn"]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def summary(self) -> str:
        if not self.findings:
            return "preflight: nothing to flag"
        lines = [f"preflight: {len(self.blocking)} blocking, "
                 f"{len(self.warnings)} warning(s)"]
        lines += [f"  {f}" for f in self.findings]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "blocking": [f.code for f in self.blocking],
            "findings": [{"code": f.code, "severity": f.severity,
                          "message": f.message, "line": f.line,
                          "remedy": f.remedy} for f in self.findings],
        }


#: Patterns worth refusing or flagging, each with the reason a kernel author
#: would not have guessed. Ordered, because the first hit on a line is the
#: interesting one.
_RULES: tuple[tuple[str, str, str, str, str], ...] = (
    # code, severity, pattern, message, remedy
    ("forks", "block", r"\b(fork|vfork|system|popen|exec[lv]p?e?)\s*\(",
     "a kernel that spawns processes is not a kernel",
     "move the orchestration into the harness, not the C"),
    ("unsafe_str", "warn", r"\b(gets|strcpy|strcat|sprintf)\s*\(",
     "unbounded string function; a kernel taking a length should use the "
     "bounded variant",
     "use the _s or n-taking form"),
    ("unbounded_loop", "warn", r"(while\s*\(\s*1\s*\)|while\s*\(\s*true\s*\)"
                              r"|for\s*\(\s*;\s*;\s*\))",
     "deliberate infinite loop; if it can be reached with a bad n the run only "
     "ends by timeout",
     "bound it, or make sure the exit condition cannot be skipped"),
    ("fast_math", "warn", r"-ffast-math|-funsafe-math-optimizations",
     "-ffast-math is not IEEE: it reassociates the sum a reference computes "
     "differently, so a correct kernel can fail verification",
     "drop it, or verify with an equally reassociated reference"),
    ("nondeterministic", "warn", r"\b(rand|srand|time)\s*\(",
     "reads the clock or the global RNG, so two runs on identical input "
     "disagree; a reference comparison then fails intermittently",
     "take the value as an argument"),
    ("openmp_missing", "warn", r"#pragma\s+omp",
     "an OpenMP pragma without the flag is silently ignored and the kernel runs "
     "single-threaded, which reads as 'parallel was not faster'",
     "set openmp=True on the KernelSource"),
    ("print_in_kernel", "info", r"\b(printf|fprintf|puts)\s*\(",
     "I/O in the kernel body dominates the measurement it is inside",
     "remove it, or time a variant without it"),
    ("unchecked_alloc", "info", r"\bmalloc\s*\(",
     "heap allocation inside a timed region is usually the dominant cost, and "
     "the size is not checked",
     "check the pointer, or hoist the allocation out of the loop"),
    ("no_fma", "info", r"__m?256|immintrin",
     "intrinsics bypass the compiler's vectoriser; the search can find this "
     "itself, so start from the plain loop",
     "keep the portable version as the baseline"),
)

#: `int n` where the problem is large: a 32-bit element count overflows for a
#: buffer over 2G elements and the kernel then walks off the end. Flagged by
#: comparing the declared parameter type against the problem's size.
_INT_PARAM_RE = re.compile(r"\bint\s+([A-Za-z_][A-Za-z0-9_]*)\s*[,)]")

#: 2^31 elements, past which a signed 32-bit count is wrong.
INT_OVERFLOW_ELEMS = 1 << 31


def preflight(source: Any, *, problem: Any = None, info: Any = None) -> Preflight:
    """Scan a kernel source for the things worth knowing before compiling.

    `problem` is optional and only sharpens the advice: given one, the scan can
    say whether the declared parameter types can hold the problem's size, which
    it cannot know from the source alone.
    """
    pf = Preflight()
    code = getattr(source, "code", "") or ""
    lines = code.splitlines()

    for name, severity, pattern, message, remedy in _RULES:
        rx = re.compile(pattern)
        for i, line in enumerate(lines, 1):
            if line.lstrip().startswith(("*", "//", "/*")):
                continue                          # a comment is not a finding
            if rx.search(line):
                # -ffast-math is a *build* flag, so it is only ever present in
                # the source as a leftover; the flag list is checked separately.
                pf.findings.append(Finding(name, message, severity, i, remedy))
                break

    if getattr(source, "openmp", False):
        if not any(re.search(r"#pragma\s+omp", ln) for ln in lines):
            pf.findings.append(Finding(
                "openmp_unused", "OpenMP was requested but the source has no "
                "#pragma omp, so the flag buys nothing", "info",
                remedy="add a pragma, or drop openmp=True"))

    for flag in getattr(source, "effective_flags", ()) or ():
        if flag in _FLAG_HAZARDS:
            pf.findings.append(Finding(*_FLAG_HAZARDS[flag]))

    if info is not None and getattr(info, "openmp", False) is False \
            and any(re.search(r"#pragma\s+omp", ln) for ln in lines):
        pf.findings.append(Finding(
            "no_openmp_runtime", "the compiler on this machine reports no "
            "OpenMP support, so the pragmas will not parallelise", "warn",
            remedy="install libgomp or use explicit threads"))

    declared = _declared_int_params(code)
    size = getattr(problem, "spec", None)
    n_elems = size.size_arg() if size is not None else 0
    if n_elems >= INT_OVERFLOW_ELEMS:
        for param in declared:
            pf.findings.append(Finding(
                "int_count", f"`int {param}` is 32-bit but the problem has "
                f"{n_elems} elements, past 2^31", "block",
                remedy="take size_t, or use SIZE_CTYPE via ops.size_t()"))

    seen: set[str] = set()
    unique: list[Finding] = []
    for f in pf.findings:
        if f.code in seen:
            continue
        seen.add(f.code)
        unique.append(f)
    pf.findings = unique
    return pf


_FLAG_HAZARDS: dict[str, tuple[str, str, str, str]] = {
    "-ffast-math": ("fast_math", "-ffast-math is not IEEE: it reassociates the "
                    "sum a reference computes differently", "warn",
                    "drop it, or verify against an equally reassociated reference"),
    "-Ofast": ("ofast", "-Ofast implies -ffast-math and is not IEEE", "warn",
               "use -O3 unless the reassociation is intended"),
    "-march=native": ("native_flag", "native is an identity-free target: the "
                      "artefact is specific to this CPU", "info",
                      "record the resolved target, which this package does"),
}


def _declared_int_params(code: str) -> list[str]:
    return sorted({m.group(1) for m in _INT_PARAM_RE.finditer(code)})


# -- isolation --------------------------------------------------------------
@dataclass
class GuardResult:
    """What a process boundary observed, including how the child died."""

    ok: bool = False
    timed_out: bool = False
    crashed: bool = False
    returncode: int | None = None
    fault: str = ""                          # a named crash, when recognised
    error: str = ""
    verdict: dict[str, Any] = field(default_factory=dict)
    wall: Duration = field(default_factory=lambda: Duration.from_seconds(0.0))

    @property
    def timings_ms(self) -> list[float]:
        return [s * 1000.0 for s in self.verdict.get("timings_s", [])]

    def summary(self) -> str:
        head = f"guarded run in {self.wall.format('ms')}"
        if self.timed_out:
            return (f"{head}: TIMED OUT — the kernel did not return. This is "
                    f"almost always an unbounded loop; there is no partial "
                    f"result to trust.")
        if self.crashed:
            reason = self.fault or f"exit code {self.returncode}"
            return (f"{head}: CRASHED — {reason}. The agent survived because "
                    f"the call was isolated.")
        if self.error:
            return f"{head}: {self.error}"
        tail = "" if self.ok else " — see verdict"
        state = "verified" if self.ok else "wrong answer"
        return f"{head}: {state}{tail}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "timed_out": self.timed_out, "crashed": self.crashed,
            "returncode": self.returncode, "fault": self.fault,
            "error": self.error, "verdict": self.verdict,
            "wall_s": self.wall.seconds_,
        }


def _payload_for(kernel: Any, prob: Any, reps: int, warmup: int) -> dict[str, Any]:
    """What the child needs to rebuild this run: the artefact, the recipe, the knobs.

    The reference answer is deliberately absent. It is *derived* — `problem()`
    reproduces it exactly from `params`, because every fill comes from a seeded
    RNG — so shipping it would be shipping a description's output rather than the
    description. It is also what stops the child being cheap to start: 16M floats
    of JSON is around 330 MB and thirteen seconds of `json.dumps` per candidate,
    which arrives as "why does the search take a minute per mutation" rather than
    as an obvious marshalling cost.
    """
    if not prob.params:
        raise ValueError(
            f"{prob.kind} has no params to rebuild from; build problems with "
            f"ops.problem() rather than constructing Problem directly"
        )
    return {
        "artefact": str(kernel.artefact),
        "source": {
            "name": kernel.source.name,
            "code": kernel.source.code,
            "target": kernel.source.target,
            "flags": list(kernel.source.flags),
            "openmp": kernel.source.openmp,
        },
        "params": prob.params,
        "reps": reps,
        "warmup": warmup,
        "nbytes": prob.nbytes,
    }


def run_isolated(
    kernel: Any,
    prob: Any,
    *,
    reps: int = 5,
    warmup: int = 3,
    timeout: float | None = None,
) -> GuardResult:
    """Verify and time a kernel in a child process, surviving whatever it does.

    The child is a `python -m` subprocess rather than a `multiprocessing`
    process for one concrete reason: on Windows, `spawn` requires an importable
    `__main__`, and the agent's own entry point is not always a file (a REPL, a
    `python -c`, a notebook). A subprocess has no such requirement, is killable
    by the OS on a hard deadline, and yields a return code that distinguishes a
    crash from a timeout — which is the difference between "fix the loop bound"
    and "fix the index" in the report the agent reads.
    """
    if timeout is None:
        timeout = float(os.environ.get("AUTOFORGE_CPU_TIMEOUT", DEFAULT_TIMEOUT))
    if not Path(kernel.artefact).exists():
        return GuardResult(error=f"artefact missing: {kernel.artefact}")

    with tempfile.TemporaryDirectory(prefix="autoforge_cpu_run_") as td:
        payload = Path(td) / "job.json"
        out = Path(td) / "result.json"
        payload.write_text(json.dumps(_payload_for(kernel, prob, reps, warmup)),
                           encoding="utf-8")
        env = dict(os.environ)
        # The child must be able to import this package without inheriting the
        # parent's sys.path, which a `-m` invocation does not guarantee.
        pkg_parent = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = pkg_parent + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        # numpy's BLAS is the harness's, not the kernel's, and on this host it
        # can kill the child before a single instruction of the kernel runs.
        # Measured on a 16-core Windows box: OpenBLAS allocating its default
        # one-thread-per-core aborts the process with
        #   "OpenBLAS error: Memory allocation still failed after 10 retries"
        # -- no traceback, no return code to interpret, just a dead child. That
        # turns 18 tests red and, in production, reports a perfectly good kernel
        # as unrunnable. 4 threads is comfortable here and OpenBLAS is not what
        # is being measured.
        #
        # OPENBLAS_* only. OMP_NUM_THREADS is deliberately left alone: a kernel
        # written with `#pragma omp parallel` is timed on its own threads, and
        # pinning those would change the number the guard reports.
        env.setdefault("OPENBLAS_NUM_THREADS", "4")
        env.setdefault("OPENBLAS_DEFAULT_NUM_THREADS", "4")
        # The module name is derived, not hardcoded. Upstream this file lives at
        # `autoforge.cpu.safety`, but the service hosting it imports the same
        # source as `autoforge_svc.cpu.safety` -- and a hardcoded
        # `-m autoforge.cpu.safety` made every guarded run die with
        #   ModuleNotFoundError: No module named 'autoforge'
        # which the guard then reported as `crashed`, i.e. as a defect in the
        # kernel rather than in the harness. `__name__` is the name this module
        # was actually imported under, so `-m` resolves it the same way.
        module = __name__
        if module == "__main__":                            # pragma: no cover
            module = "autoforge.cpu.safety"
        cmd = [sys.executable, "-m", module,
               "--payload", str(payload), "--out", str(out)]

        started = __import__("time").perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, env=env,
                                  errors="replace", cwd=pkg_parent)
        except subprocess.TimeoutExpired as exc:
            wall = Duration.from_seconds(timeout)
            return GuardResult(
                timed_out=True, returncode=None, wall=wall,
                error=f"killed after {timeout:.0f}s",
                verdict={"reason": "timeout",
                         "stdout": (exc.stdout or "")[-500:] if exc.stdout else "",
                         "stderr": (exc.stderr or "")[-500:] if exc.stderr else ""},
            )
        wall = Duration.from_seconds(__import__("time").perf_counter() - started)

        if out.exists():
            try:
                result = json.loads(out.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return GuardResult(
                    returncode=proc.returncode, wall=wall,
                    error=f"the child wrote unreadable output: {exc}",
                    verdict={"stderr": (proc.stderr or "")[-800:]},
                )
            return GuardResult(
                ok=bool(result.get("ok")), returncode=proc.returncode,
                wall=wall, verdict=result,
                error="" if result.get("ok") else result.get("reason", "not verified"),
            )

        # No result file: the child died before it could write one. Name the
        # death rather than reporting an empty failure.
        fault = _name_fault(proc.returncode)
        return GuardResult(
            crashed=True, returncode=proc.returncode, fault=fault, wall=wall,
            error=fault or f"the child exited {proc.returncode} with no result",
            verdict={"stderr": (proc.stderr or "")[-800:],
                     "stdout": (proc.stdout or "")[-400:]},
        )


def _name_fault(returncode: int | None) -> str:
    """Turn a death into a sentence, on both platforms."""
    if returncode is None:
        return ""
    if sys.platform == "win32":
        # Windows reports NTSTATUS as an unsigned 32-bit exit code.
        return WINDOWS_FAULTS.get(returncode & 0xFFFFFFFF,
                                  WINDOWS_FAULTS.get(returncode, ""))
    if returncode < 0:
        import signal as sig
        try:
            return f"killed by {sig.Signals(-returncode).name} ({sig.strsignal(-returncode)})"
        except (ValueError, AttributeError):
            return f"killed by signal {-returncode}"
    return ""


# -- child entry ------------------------------------------------------------
def _child(payload_path: str, out_path: str) -> int:
    """The child half. Any failure is written to `out_path`, never raised.

    Written this way because the parent's whole job is to distinguish outcomes,
    and an uncaught exception here would be indistinguishable from a crash.

    The problem is rebuilt from its recipe rather than received, so the child
    computes the reference with the same code the parent would have — the
    guarantee is reproducibility, not transmission.
    """
    from .kernel import CompiledKernel, KernelSource
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    try:
        src = KernelSource(**payload["source"])
        kernel = CompiledKernel(source=src, artefact=Path(payload["artefact"]),
                                cache_key="", from_cache=True, compiler="cache")
        recipe = dict(payload["params"])
        prob = ops_mod.problem(recipe.pop("kind"), **recipe)
        result = ops_mod.evaluate(kernel, prob, reps=payload["reps"],
                                  warmup=payload["warmup"])
        result.setdefault("nbytes", payload.get("nbytes"))
        Path(out_path).write_text(json.dumps(result), encoding="utf-8")
        return 0
    except BaseException as exc:                             # noqa: BLE001
        Path(out_path).write_text(json.dumps({
            "ok": False, "verified": False, "reason": f"{type(exc).__name__}: {exc}",
        }), encoding="utf-8")
        return 0


def _main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="autoforge.cpu.safety",
                                 description="isolated kernel runner (internal)")
    ap.add_argument("--payload", required=True)
    ap.add_argument("--out", required=True)
    ns = ap.parse_args(argv)
    return _child(ns.payload, ns.out)


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "DEFAULT_TIMEOUT", "WINDOWS_FAULTS", "Finding", "Preflight", "GuardResult",
    "preflight", "run_isolated",
]
