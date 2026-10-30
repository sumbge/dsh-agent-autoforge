"""Running a forged kernel, and knowing whether it was right.

A benchmark says a kernel is fast. This module is the half that says whether it
is fast *and correct*, which is the only combination worth having — and the half
that a search loop gets wrong when it is written in a hurry, because the fastest
kernel in the search is reliably the one that skipped the most work.

Three ideas, one per hazard:

  A CALL IS DATA. `CallSpec` describes a call in terms a fresh process can
  reconstruct: dtypes, shapes, fills. Data rather than a closure because the
  isolated runner (`safety.run_isolated`) has to rebuild the call on the far
  side of a process boundary — and because a description can be shown to the
  agent, logged, and hashed into a cache key, while a closure cannot.

  A REFERENCE IS MANDATORY. `problem()` returns the inputs *and* the expected
  output, computed here in Python or numpy. There is no way to ask for a
  problem without a reference, so "verify against something" is not a step a
  caller can forget.

  CLOSENESS IS RELATIVE. `measure.verify` reports absolute difference, which is
  the right primitive for a single element and the wrong test for a reduction
  scaled by n. `close()` adds the relative term, so a matmul is not failed for
  the crime of being 256 terms long. Both numbers are reported, because "10x
  tolerance" and "one element is wrong" are different bugs and the pair
  distinguishes them.
"""
from __future__ import annotations

import os

import array
import ctypes
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

from ..measure import DEFAULT_REPS, DEFAULT_WARMUP
from ..timing import Duration

#: ctype name and width, for the dtypes a kernel is realistically written in.
DTYPES: dict[str, tuple[str, int]] = {
    "f32": ("c_float", 4),
    "f64": ("c_double", 8),
    "i32": ("c_int", 4),
    "i64": ("c_longlong", 8),
    "u32": ("c_uint", 4),
    "u64": ("c_ulonglong", 8),
}

#: Passed for `size_t n` parameters, which every C kernel takes.
SIZE_CTYPE = "c_size_t"

#: `array.array` type codes, for filling a ctypes block through the buffer
#: protocol. Kept beside DTYPES because the two must agree on width: a mismatch
#: silently truncates every element instead of failing.
TYPECODES: dict[str, str] = {
    "f32": "f", "f64": "d", "i32": "i", "i64": "q", "u32": "I", "u64": "Q",
}

# Standard relative/absolute tolerances, per dtype, in the numpy sense.
# f32 accumulates error in a long reduction; f64 does not, and holding f64 to
# f32's tolerance would hide a real bug.
TOLERANCES: dict[str, tuple[float, float]] = {
    "f32": (1e-4, 1e-4),
    "f64": (1e-10, 1e-12),
    "i32": (0.0, 0.0),
    "i64": (0.0, 0.0),
    "u32": (0.0, 0.0),
    "u64": (0.0, 0.0),
}


class CallError(RuntimeError):
    """A call specification that cannot be turned into ctypes arguments."""


# -- describing a call ------------------------------------------------------
@dataclass(frozen=True)
class Buffer:
    """An array argument, described rather than instantiated.

    `fill` is one of "ones", "zeros", "index" (i/N, well-conditioned for a
    reduction), "random" (seeded), or a number. The named fills exist so a
    problem is reproducible from its description alone: "random" without a seed
    is a flaky test, and flaky is indistinguishable from a real bug at 3am.
    """

    dtype: str = "f32"
    size: int = 1 << 20
    fill: Any = "ones"
    name: str = "buf"
    seed: int = 0
    _template: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise CallError(f"unknown dtype {self.dtype!r}; known: {', '.join(DTYPES)}")
        if self.size < 1:
            raise CallError(f"buffer size must be >= 1, got {self.size}")

    @property
    def nbytes(self) -> int:
        return self.size * DTYPES[self.dtype][1]

    def values(self) -> list[Any]:
        """The concrete array, built deterministically.

        Notes on the two non-obvious fills, both learned from reductions:
          "index" gives i/(N) in [0,1) so a sum is O(1) rather than O(N) and
            does not lose precision in f32 at large N;
          "random" is drawn from a seeded `random.Random`, not the module-level
            RNG, so two problems built in different orders still agree.
        """
        n, dt = self.size, self.dtype
        if isinstance(self.fill, (int, float)):
            return [self.fill] * n
        f = str(self.fill).lower()
        if f in ("ones", "one"):
            return [1] * n if dt.startswith(("i", "u")) else [1.0] * n
        if f in ("zeros", "zero"):
            return [0] * n
        if f == "index":
            return [i / n for i in range(n)]
        if f == "random":
            rng = random.Random(self.seed)
            if dt.startswith("f"):
                return [rng.random() for _ in range(n)]
            return [rng.randrange(1, 100) for _ in range(n)]
        raise CallError(f"no such fill: {self.fill!r}")

    def to_ctypes(self) -> Any:
        """A fresh ctypes array holding this buffer's *initial* values.

        Fresh on every call, and that is the point rather than an inefficiency:
        a repetition that starts from the previous repetition's output is not
        measuring the same thing, and for an in-place kernel like saxpy the
        output grows without bound across reps until the numbers have stopped
        resembling the workload.

        The values are converted to ctypes once and cached, because building an
        array from a million Python floats costs about a second — some four
        orders of magnitude more than the kernel being measured — and this is
        called once per repetition. After the first call the copy is a memmove
        over a contiguous block, which is the same thing the kernel is about to
        do to it anyway.
        """
        if self._template is None:
            cname, _ = DTYPES[self.dtype]
            ctype = getattr(ctypes, cname)
            # Via `array.array`, not `(ctype * n)(*values())`. Star-unpacking a
            # list of N floats builds N arguments one at a time — for a 16M
            # element buffer that is tens of seconds, which then presents as a
            # slow *search* rather than a slow marshaller. `array.array` fills a
            # contiguous block through the buffer protocol instead.
            packed = array.array(TYPECODES[self.dtype], self.values())
            template = (ctype * self.size).from_buffer_copy(packed)
            object.__setattr__(self, "_template", template)
        cname, _ = DTYPES[self.dtype]
        arr = (getattr(ctypes, cname) * self.size)()
        ctypes.memmove(arr, self._template, self.nbytes)
        return arr

    @classmethod
    def from_values(cls, name: str, values: list[Any]) -> "Buffer":
        """A buffer wrapping an already-computed array, for the result side."""
        dt = "f64" if any(isinstance(v, float) for v in values) else "i64"
        b = cls(dtype=dt, size=max(1, len(values)), fill=0, name=name)
        object.__setattr__(b, "_values", list(values))
        return b

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "buffer", "name": self.name, "dtype": self.dtype,
                "size": self.size, "fill": self.fill, "seed": self.seed,
                "nbytes": self.nbytes}


@dataclass(frozen=True)
class Scalar:
    """A scalar argument, with the C type it must be passed as.

    The type is explicit because `ctypes` defaults an int to 32 bits, and a
    kernel taking `size_t n` that is handed a C int reads garbage in its high
    half and runs off the end. That failure looks exactly like a kernel bug.
    """

    dtype: str = "i32"
    value: Any = 0
    ctype: str = ""

    def to_ctypes(self) -> Any:
        """The ctypes object for this argument — same protocol as `Buffer`."""
        name = self.ctype or DTYPES[self.dtype][0]
        cast = getattr(ctypes, name)
        if name in ("c_float", "c_double"):
            return cast(float(self.value))
        return cast(int(self.value))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "scalar", "dtype": self.dtype, "value": self.value,
                "ctype": self.ctype or DTYPES[self.dtype][0]}


@dataclass(frozen=True)
class CallSpec:
    """A whole call: which symbol, which arguments, which come back.

    `outputs` lists the argument indices that are read after the call. `returns`
    names the dtype of the return value when the kernel returns one instead of
    writing an output buffer — a reduction does, and a spec that could only
    describe out-parameters would make every reduction unverifiable, which is
    exactly the class of kernel where a wrong answer is easiest to ship.

    Return values are captured under the index -1, so `outputs=()` with
    `returns="f32"` is a legal and complete description of `float reduce_sum(...)`.
    """

    fn: str
    args: tuple[Any, ...]
    outputs: tuple[int, ...] = ()
    returns: str = ""
    label: str = ""

    @property
    def buffers(self) -> list[Buffer]:
        return [a for a in self.args if isinstance(a, Buffer)]

    def size_arg(self) -> int:
        """The element count, taken from the first buffer.

        Kernels in this codebase are passed a flat n; taking it from the data
        rather than a separate field means the two cannot disagree.
        """
        for a in self.args:
            if isinstance(a, Buffer):
                return a.size
        return 0

    @property
    def restype(self) -> Any:
        if not self.returns:
            return None
        if self.returns not in DTYPES:
            raise CallError(f"unknown return dtype {self.returns!r}")
        return getattr(ctypes, DTYPES[self.returns][0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "fn": self.fn,
            "label": self.label,
            "args": [a.to_dict() for a in self.args],
            "outputs": list(self.outputs),
            "returns": self.returns,
        }

    def describe(self) -> str:
        parts = []
        for a in self.args:
            if isinstance(a, Buffer):
                parts.append(f"{a.name}:{a.dtype}[{a.size}]")
            elif isinstance(a, Scalar):
                parts.append(f"{a.value}")
            else:
                parts.append(str(a))
        out = ",".join(str(i) for i in self.outputs) or "-"
        ret = f" -> {self.returns}" if self.returns else ""
        return f"{self.fn}({', '.join(parts)}){ret} [out: args[{out}]]"


#: How a size_t parameter is described, so a spec never hand-writes a ctype.
def size_t(value: int) -> Scalar:
    return Scalar(dtype="u64", value=int(value), ctype=SIZE_CTYPE)


# -- running ----------------------------------------------------------------
@dataclass
class RunResult:
    """What came back, and how long it took."""

    outputs: dict[int, list[Any]] = field(default_factory=dict)
    wall: Duration = field(default_factory=lambda: Duration.from_seconds(0.0))
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs": {str(k): list(v) for k, v in self.outputs.items()},
            "wall_s": self.wall.seconds_,
            "error": self.error,
        }


def marshal(spec: CallSpec) -> list[Any]:
    """The ctypes arguments for a spec. Untimed by contract.

    Kept separate from the call so the measurement covers the kernel and not the
    marshalling — a distinction that decides the result. Building the arguments
    for a million-element saxpy takes about a second, so a timer wrapped around
    `run` reports a thousand milliseconds for a kernel that takes half of one,
    and the search then selects for whichever candidate happens to marshal
    fastest. That is the same mistake as reporting a GPU benchmark in the wrong
    unit: the number is not wrong, it is measuring something else.
    """
    return [a.to_ctypes() for a in spec.args]


def invoke(kernel: Any, spec: CallSpec, c_args: list[Any]) -> Any:
    """Call the kernel on already-marshalled arguments. Just the call."""
    return kernel.call(spec.fn, *c_args, restype=spec.restype)


def run(kernel: Any, spec: CallSpec, *, c_args: list[Any] | None = None) -> RunResult:
    """Call the kernel once, in this process, and read back its outputs.

    In-process is the wrong place for an untrusted kernel and the right place
    for a verified one: the caller (`safety.run_isolated`) has already paid for
    a process boundary to survive a crash or a hang, and paying for a second one
    per repetition would make the benchmark measure process startup.

    `c_args` lets a caller reuse arguments it has already built (untimed). Pass
    the same list to time a run without the marshalling inside the timer.
    """
    if c_args is None:
        c_args = marshal(spec)
    started = time.perf_counter()
    try:
        ret = invoke(kernel, spec, c_args)
    except Exception as exc:                                 # noqa: BLE001
        return RunResult(error=f"{type(exc).__name__}: {exc}",
                         wall=Duration.from_seconds(time.perf_counter() - started))
    wall = Duration.from_seconds(time.perf_counter() - started)
    outputs: dict[int, list[Any]] = {}
    for idx in spec.outputs:
        if idx >= len(c_args):
            return RunResult(error=f"output index {idx} is out of range "
                                   f"({len(c_args)} arguments)", wall=wall)
        outputs[idx] = list(c_args[idx])
    if spec.returns:
        # -1 is the return slot, so a reduction is describable in the same
        # structure as an out-parameter kernel.
        outputs[-1] = [ret]
    return RunResult(outputs=outputs, wall=wall)


# -- comparing --------------------------------------------------------------
def close(
    actual: list[Any],
    reference: list[Any],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """numpy-style closeness: |a-r| <= atol + rtol*|r|, with all the numbers.

    Both the absolute and the relative worst case are reported. The pair is what
    makes a failure diagnosable: a huge absolute difference on a large reference
    is accumulated error and probably fine; the same absolute difference on a
    small reference is an off-by-one in an index.
    """
    if len(actual) != len(reference):
        return {"passed": False, "max_abs_diff": None, "max_rel_diff": None,
                "reason": f"length mismatch: {len(actual)} vs {len(reference)}"}
    if not actual:
        return {"passed": False, "max_abs_diff": None, "max_rel_diff": None,
                "reason": "empty output"}
    max_abs = 0.0
    max_rel = 0.0
    worst_abs = worst_rel = 0
    nan_seen = 0
    for i, (a, r) in enumerate(zip(actual, reference)):
        a, r = float(a), float(r)
        if math.isnan(a) or math.isnan(r):
            nan_seen += 1
            continue
        d = abs(a - r)
        if d > max_abs:
            max_abs, worst_abs = d, i
        denom = abs(r)
        rel = d / denom if denom > 0 else (0.0 if d == 0 else math.inf)
        if rel > max_rel:
            max_rel, worst_rel = rel, i
    if nan_seen:
        return {"passed": False, "max_abs_diff": max_abs, "max_rel_diff": max_rel,
                "worst_index": worst_abs, "nan_count": nan_seen,
                "reason": f"{nan_seen} NaN(s) in the output — a NaN is never a "
                          f"rounding error"}
    passed = max_abs <= (atol + rtol * abs(float(reference[worst_abs])))
    tolerances_off = max_abs / (atol + rtol * abs(float(reference[worst_abs]))) \
        if (atol + rtol * abs(float(reference[worst_abs]))) > 0 else (
            math.inf if max_abs > 0 else 1.0)
    return {
        "passed": passed,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "worst_index": worst_abs,
        "worst_rel_index": worst_rel,
        "rtol": rtol,
        "atol": atol,
        "tolerance_headroom": tolerances_off,
        "reason": ("identical to the reference — did the kernel do any work?"
                   if max_abs == 0.0 else
                   "within tolerance" if passed else
                   f"{tolerances_off:.1f}x over tolerance — a correctness bug, "
                   f"not precision"),
    }


# -- problems: a verified workload ------------------------------------------
@dataclass
class Problem:
    """A call plus the answer, plus how the answer was computed."""

    kind: str
    spec: CallSpec
    inputs: dict[int, list[Any]]
    expected: dict[int, list[Any]]
    rtol: float = 1e-4
    atol: float = 1e-4
    backend: str = "python"                  # numpy | python
    note: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for b in self.spec.buffers)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "call": self.spec.describe(),
                "spec": self.spec.to_dict(), "rtol": self.rtol, "atol": self.atol,
                "backend": self.backend, "note": self.note,
                "nbytes": self.nbytes, "params": dict(self.params)}


def _numpy():
    """numpy, with its BLAS thread count bounded before the import.

    Why this is not premature: importing numpy initialises OpenBLAS, which sizes
    its thread pool from the core count and allocates for it. On this host --
    16 cores, Windows -- that allocation fails inside the process and OpenBLAS
    does not raise, it *aborts*:

        OpenBLAS error: Memory allocation still failed after 10 retries, giving up.

    No traceback, no exception for a caller to catch, and the process is gone.
    Measured: 16 threads dies, 8 dies, 4 and below is fine. The visible symptom
    was 18 tests in tests/test_cpu.py going red with the pytest process dying
    mid-file, which reads exactly like "the isolated-execution layer is broken"
    when nothing in the layer is involved.

    Set before `import numpy`, and with `setdefault` so a caller who knows their
    machine can override it. OPENBLAS_* only -- OMP_NUM_THREADS governs the
    kernel under test, and bounding *that* would change the number the guard
    reports.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_DEFAULT_NUM_THREADS", "4")
    try:
        import numpy
        return numpy
    except Exception:                                        # noqa: BLE001
        return None


def _build_problem(
    kind: str,
    *,
    size: int = 1 << 20,
    n: int = 256,
    dtype: str = "f32",
    seed: int = 0,
    fill: Any = "random",
) -> Problem:
    """A named workload with its reference answer.

    Each kind is a shape the search loop can actually improve on, chosen because
    a naive version of it is beatable: saxpy is bandwidth-bound, matmul wants
    tiling and blocking, reduce wants unrolling and multiple accumulators,
    softmax wants a single pass with an online maximum.
    """
    np = _numpy()
    rtol, atol = TOLERANCES.get(dtype, (1e-4, 1e-4))

    if kind == "saxpy":
        a = Buffer(dtype, size, fill, "x", seed)
        b = Buffer(dtype, size, fill, "y", seed + 1)
        alpha = 2.5
        xv, yv = a.values(), b.values()
        exp = [alpha * xv[i] + yv[i] for i in range(size)]
        return Problem(kind, CallSpec("saxpy",
                                      (Buffer(dtype, size, fill, "y", seed + 1),
                                       Buffer(dtype, size, fill, "x", seed),
                                       Scalar("f32", alpha), size_t(size)),
                                      (0,), label=f"saxpy n={size}"),
                       {0: yv, 1: xv}, {0: exp}, rtol, atol,
                       "numpy" if np else "python")

    if kind == "vector_add":
        a = Buffer(dtype, size, fill, "a", seed)
        b = Buffer(dtype, size, fill, "b", seed + 1)
        av, bv = a.values(), b.values()
        exp = [av[i] + bv[i] for i in range(size)]
        return Problem(kind, CallSpec("vector_add",
                                      (Buffer(dtype, size, fill, "out", seed),
                                       Buffer(dtype, size, fill, "a", seed),
                                       Buffer(dtype, size, fill, "b", seed + 1),
                                       size_t(size)),
                                      (0,), label=f"vector_add n={size}"),
                       {1: av, 2: bv}, {0: exp}, rtol, atol,
                       "numpy" if np else "python")

    if kind == "reduce_sum":
        a = Buffer(dtype, size, "index", "x", seed)
        xv = a.values()
        if np:
            total = float(np.sum(np.array(xv, dtype=np.float32 if dtype == "f32"
                                             else np.float64)))
        else:
            total = math.fsum(float(v) for v in xv)
        return Problem(kind, CallSpec("reduce_sum",
                                      (Buffer(dtype, size, "index", "x", seed),
                                       size_t(size)),
                                      (), returns=dtype,
                                      label=f"reduce_sum n={size}"),
                       {0: xv}, {-1: [total]}, rtol, atol,
                       "numpy" if np else "python",
                       note="returns by value, so the check is on the return "
                            "slot (-1) rather than a buffer")

    if kind == "matmul":
        a = Buffer(dtype, n * n, fill, "A", seed)
        b = Buffer(dtype, n * n, fill, "B", seed + 1)
        av, bv = a.values(), b.values()
        if np:
            A = np.array(av, dtype=np.float32 if dtype == "f32" else np.float64
                         ).reshape(n, n)
            B = np.array(bv, dtype=A.dtype).reshape(n, n)
            exp = (A @ B).reshape(-1).tolist()
        else:
            exp = [math.fsum(av[i * n + k] * bv[k * n + j] for k in range(n))
                   for i in range(n) for j in range(n)]
        return Problem(kind, CallSpec("matmul",
                                      (Buffer(dtype, n * n, 0, "C", 0),
                                       Buffer(dtype, n * n, fill, "A", seed),
                                       Buffer(dtype, n * n, fill, "B", seed + 1),
                                       size_t(n)),
                                      (0,), label=f"matmul {n}x{n}"),
                       {1: av, 2: bv}, {0: exp}, 1e-3, 1e-3,
                       "numpy" if np else "python",
                       note="a wider tolerance because a k-term dot product in "
                            "f32 legitimately loses precision")

    if kind == "relu":
        a = Buffer(dtype, size, fill, "x", seed)
        xv = a.values()
        exp = [max(0.0, float(v)) for v in xv]
        return Problem(kind, CallSpec("relu",
                                      (Buffer(dtype, size, fill, "x", seed),
                                       size_t(size)),
                                      (0,), label=f"relu n={size}"),
                       {0: xv}, {0: exp}, rtol, atol,
                       "numpy" if np else "python")

    if kind == "softmax":
        a = Buffer(dtype, size, fill, "x", seed)
        xv = [float(v) for v in a.values()]
        m = max(xv)
        ex = [math.exp(v - m) for v in xv]
        tot = math.fsum(ex)
        exp = [e / tot for e in ex]
        return Problem(kind, CallSpec("softmax",
                                      (Buffer(dtype, size, 0, "out", 0),
                                       Buffer(dtype, size, fill, "x", seed),
                                       size_t(size)),
                                      (0,), label=f"softmax n={size}"),
                       {1: xv}, {0: exp}, 1e-3, 1e-5,
                       "python",
                       note="the max subtraction is the whole point: without it "
                            "exp overflows and the output is all-NaN")

    raise CallError(
        f"no such problem {kind!r}; known: saxpy, vector_add, reduce_sum, "
        f"matmul, relu, softmax"
    )


def problem(
    kind: str,
    *,
    size: int = 1 << 20,
    n: int = 256,
    dtype: str = "f32",
    seed: int = 0,
    fill: Any = "random",
) -> Problem:
    """Build a workload, stamping the recipe that produced it.

    The stamp is what makes the problem portable across a process boundary. A
    problem is fully determined by its kind and these five arguments — the fills
    are drawn from seeded RNGs and the reference is computed from them — so
    "which workload was this" is a *description*, not the 16 million floats of
    its answer. Sending the answer instead costs hundreds of megabytes of JSON
    per candidate and dominates everything else the search does; see
    `safety.run_isolated`.
    """
    prob = _build_problem(kind, size=size, n=n, dtype=dtype, seed=seed, fill=fill)
    prob.params = {"kind": kind, "size": size, "n": n, "dtype": dtype,
                   "seed": seed, "fill": fill}
    return prob


def evaluate(kernel: Any, prob: Problem, *, reps: int = DEFAULT_REPS,
             warmup: int = DEFAULT_WARMUP) -> dict[str, Any]:
    """Verify, then time. In that order, and never the other way round.

    The order is the whole method. Timing an unverified kernel produces a number
    that the search loop will then select for, and the fastest wrong kernel is
    both the most likely to win and the most expensive to discover later.
    """
    if not prob.expected:
        return {"ok": False, "reason": f"{prob.kind} has no output to verify",
                "verified": False}
    verdict = run(kernel, prob.spec)
    if not verdict.ok:
        return {"ok": False, "verified": False, "reason": verdict.error}

    checks: dict[str, Any] = {}
    for idx, want in prob.expected.items():
        got = verdict.outputs.get(idx)
        if got is None:
            checks[str(idx)] = {"passed": False, "reason": "output not captured"}
            continue
        checks[str(idx)] = close(got, want, rtol=prob.rtol, atol=prob.atol)
    passed = all(c.get("passed") for c in checks.values())

    timings: list[float] = []
    if passed:
        try:
            for _ in range(max(0, warmup)):
                run(kernel, prob.spec)
            for _ in range(max(1, reps)):
                # Marshal before the timer, never inside it. This is also what
                # resets the working set to its initial values between reps: a
                # repetition that starts from the last repetition's output is a
                # different workload, and for an accumulating kernel it drifts
                # until the numbers no longer resemble anything.
                c_args = marshal(prob.spec)
                started = time.perf_counter()
                invoke(kernel, prob.spec, c_args)
                timings.append(time.perf_counter() - started)
        except Exception as exc:                             # noqa: BLE001
            return {"ok": False, "verified": True, "checks": checks,
                    "reason": f"timing failed after a correct run: {exc}"}

    return {
        "ok": passed,
        "verified": passed,
        "checks": checks,
        "timings_s": timings,
        "unit": "s",
        "nbytes": prob.nbytes,
        "call": prob.spec.describe(),
        "backend": prob.backend,
        "reason": "verified" if passed else "output does not match the reference",
    }


__all__ = [
    "DTYPES", "TOLERANCES", "CallError", "Buffer", "Scalar", "CallSpec",
    "RunResult", "Problem", "run", "close", "problem", "evaluate", "size_t",
    "marshal", "invoke",
]
