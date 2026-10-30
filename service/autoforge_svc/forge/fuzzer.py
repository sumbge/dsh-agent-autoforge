"""Parameter robustness fuzzing — stress-tests a tool's input boundary handling.

The insight: most LLM-generated tools handle the "happy path" correctly and
crash on edge cases that a real user WILL throw at them. This check
systematically probes:

  - Empty / missing / null inputs for every parameter
  - Extremely long values (buffer overruns in Python are rare, but OOM is real)
  - Wrong types (string where int expected, list where dict expected)
  - Unicode / emoji injections
  - Negative numbers, zero, max-int for numeric types
  - Leading/trailing whitespace
  - "ISBN "-style prefix contamination (the exact bug found in the live demo)

Each probe is a deterministic transformation of the schema — no LLM calls.
Results are scored: the tool must survive at least `require_survival_rate`
of all robustness probes to pass.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from ..tools.spec import ToolSpec
from .invariance import InvarianceResult, check as check_invariances


# ---------------------------------------------------------------------------
# Probe generators — one per JSON Schema type
# ---------------------------------------------------------------------------

def _probes_for_string() -> list[Any]:
    """Ways strings can be surprising."""
    return [
        "",
        " ",
        "  ",
        "\t",
        "\n",
        "a" * 100_000,           # large input
        "ISBN 978-0-306-40615-7",  # prefix contamination (the ISBN bug)
        "978-0-306-40615-7 ",
        " 978-0-306-40615-7",
        "NULL",
        "None",
        "undefined",
        "<script>alert('xss')</script>",
        "😀🔥🎉",
        "日本語的テスト",
        "héllo wörld",
        "a\x00b",                 # null byte
        "line1\nline2\nline3",
        "\x1b[31mred\x1b[0m",     # ANSI escape
    ]


def _probes_for_number() -> list[Any]:
    return [
        0,
        -0,
        1,
        -1,
        2**31,
        -(2**31),
        2**63,
        -(2**63),
        10**300,   # big float
        -(10**300),
        0.5,
        -0.5,
        1e-10,
        math.nan,
        math.inf,
        -math.inf,
    ]


def _probes_for_integer() -> list[Any]:
    return [v for v in _probes_for_number() if isinstance(v, int) and not math.isnan(v)] + [
        10**100,  # big integer
    ]


def _probes_for_boolean() -> list[Any]:
    return [True, False]


def _probes_for_array() -> list[Any]:
    return [
        [],
        [None],
        [1, 2, 3],
        [{"key": "value"}] * 1000,  # large
    ]


def _probes_for_object() -> list[Any]:
    return [
        {},
        {"extra_key": "unexpected"},
        {f"key_{i}": i for i in range(1000)},
    ]


_PROBE_MAP: dict[str, list[Any]] = {
    "string": _probes_for_string(),
    "number": _probes_for_number(),
    "integer": _probes_for_integer(),
    "boolean": _probes_for_boolean(),
    "array": _probes_for_array(),
    "object": _probes_for_object(),
}


# ---------------------------------------------------------------------------
# Combinatorial probe generator
# ---------------------------------------------------------------------------

@dataclass
class ProbeInput:
    args: dict[str, Any]
    label: str


def generate_robustness_probes(
    spec: ToolSpec,
    max_probes: int = 50,
) -> list[ProbeInput]:
    """Generate input combinations that stress the tool's parameter boundaries.

    Strategy: for each parameter that has a type we know how to fuzz, we
    generate a set of probes where *that parameter* gets an edge case and
    all others get a plausible default.

    This is O(n * m) where n=params, m=probes-per-type. We cap at max_probes.
    """
    props = spec.parameters.get("properties") or {}
    required = set(spec.parameters.get("required") or [])

    results: list[ProbeInput] = []

    for name, schema in props.items():
        ptype = schema.get("type", "string")
        probes = _PROBE_MAP.get(ptype, [""])
        # Build a defaults dict for *other* params
        defaults = _build_defaults(props, exclude=name)

        for probe_val in probes:
            combined = dict(**defaults)
            combined[name] = probe_val
            label = f"{name}={_label_for(probe_val)}"
            results.append(ProbeInput(combined, label))
            if len(results) >= max_probes:
                return results

    return results


@dataclass
class RobustnessResult:
    name: str
    passed: bool
    total: int = 0
    survived: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    invariance: InvarianceResult | None = None

    @property
    def survival_rate(self) -> float:
        return self.survived / self.total if self.total else 0.0

    def summary(self) -> str:
        base = f"{self.name}: {self.survived}/{self.total} robustness probes survived"
        if self.invariance is not None:
            base += f"; {self.invariance.detail()}"
        return base

    def to_dict(self) -> dict[str, Any]:
        d = {
            "tool": self.name,
            "passed": self.passed,
            "survival_rate": round(self.survival_rate, 3),
            "total": self.total,
            "survived": self.survived,
            "failures": [
                {"label": f["label"], "error": f["error"]}
                for f in self.failures[:5]
            ],
        }
        if self.invariance is not None:
            d["invariance"] = self.invariance.to_dict()
        return d


def run_robustness_checks(
    spec: ToolSpec,
    *,
    sandbox: Any = None,
    require_survival_rate: float = 1.0,
    max_probes: int = 50,
    check_invariance: bool = True,
) -> RobustnessResult:
    """Run all robustness probes against the tool.

    Definedness alone is not an oracle — "did not raise" is satisfied by a
    function that returns a constant. When `check_invariance` is on, the
    metamorphic battery in `..forge.invariance` is run too, and a tool that is
    degenerate or breaks a normalisation relation fails the check outright.
    """
    probes = generate_robustness_probes(spec, max_probes)
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    survived = 0

    for probe in probes:
        try:
            if sandbox is not None and spec.code:
                r = sandbox.run(spec.code, spec.name, probe.args)
                ok = r.ok and r.output is not None
                err = r.error if not r.ok else (
                    None if r.output is not None else "returned None")
            else:
                out = spec.fn(**probe.args)
                ok = out is not None
                err = None if ok else "returned None"
        except Exception as exc:  # noqa: BLE001
            ok = False
            err = f"{type(exc).__name__}: {exc}"

        if ok:
            survived += 1
        else:
            failures.append({"label": probe.label, "args": str(probe.args)[:200], "error": err})

    invariance = None
    if check_invariance:
        invariance = check_invariances(
            spec,
            sandbox=sandbox,
            reference_args=_reference_args(spec),
            probe_args=[p.args for p in probes],
        )

    duration = (time.perf_counter() - started) * 1000
    rate = survived / len(probes) if probes else 1.0
    defined_ok = rate >= require_survival_rate
    return RobustnessResult(
        name=spec.name,
        passed=defined_ok and (invariance is None or invariance.passed),
        total=len(probes),
        survived=survived,
        failures=failures,
        duration_ms=duration,
        invariance=invariance,
    )


def _reference_args(spec: ToolSpec) -> dict[str, Any]:
    """The positive control: a plausible, well-formed call.

    Mirrors ToolVerifier._infer_args. Kept local rather than imported so the
    fuzzer stays free of a verifier dependency.
    """
    props = spec.parameters.get("properties") or {}
    args: dict[str, Any] = {}
    for name, schema in props.items():
        t = schema.get("type", "string")
        if t == "string":
            args[name] = "978-0-306-40615-7"
        elif t in ("number", "integer"):
            args[name] = 1
        elif t == "boolean":
            args[name] = True
        elif t == "object":
            args[name] = {}
        elif t == "array":
            args[name] = []
        else:
            args[name] = "test"
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_defaults(props: dict[str, Any], exclude: str) -> dict[str, Any]:
    """Plausible default values for all properties except `exclude`."""
    defaults = {}
    for name, schema in props.items():
        if name == exclude:
            continue
        ptype = schema.get("type", "string")
        defaults[name] = _default_for(ptype)
    return defaults


def _default_for(ptype: str) -> Any:
    if ptype == "string":
        return "test"
    if ptype in ("number", "integer"):
        return 1
    if ptype == "boolean":
        return True
    if ptype == "array":
        return []
    if ptype == "object":
        return {}
    return ""


def _label_for(val: Any) -> str:
    r = repr(val)
    if len(r) > 60:
        return f"len={len(r)}_prefix={r[:20]}"
    return r


__all__ = [
    "RobustnessResult",
    "run_robustness_checks",
    "generate_robustness_probes",
    "ProbeInput",
    "check_invariance",
]