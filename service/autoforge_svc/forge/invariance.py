"""Metamorphic oracles for the robustness layer.

The gap this closes
-------------------
`run_robustness_checks` scored a probe as "survived" iff the tool did not raise
(`ok = out is not None`). So wrong-but-total functions scored 100%: a function
returning the constant `"nope"` passed every probe, and a validator that ignores
an `"ISBN "` prefix passed all 19 — while being wrong on exactly the input the
probe existed to test.

A probe only means something if there is an oracle behind it. This module
supplies oracles of two kinds, and none of them is authored by the tool being
scored:

  computed  — true of any honest implementation, so the verifier derives them
              rather than asking:

    defined        f(x) does not raise and does not return None
    deterministic  f(x) is stable across repeated calls
    non_degenerate f does not emit one constant across valid AND garbage input

  declared  — semantic obligations that need domain knowledge the verifier does
              not have (is `"ISBN "` part of the value or noise around it?).
              The generator asserts these at birth; `FrozenBaseline` then keeps
              them, so a later mutant cannot quietly drop one.

There is no exact oracle here and there cannot be: nothing labels the correct
output for a novel input. Metamorphic relations constrain how outputs must
*relate* to each other, which is checkable without labels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..tools.spec import ToolSpec

# ---------------------------------------------------------------------------
# Computed relations — no declaration needed, derived by the verifier
# ---------------------------------------------------------------------------
COMPUTED_RELATIONS: tuple[str, ...] = ("defined", "deterministic", "non_degenerate")


# ---------------------------------------------------------------------------
# Token normalisers — what counts as noise around a value
#
# Conservative by design: only tokens whose alphabetic decoration is
# unambiguously not part of the value. `title` and `text` are absent on purpose
# — for those, whitespace and casing are content.
# ---------------------------------------------------------------------------
_STRIP_PREFIXES = {
    "isbn": ("isbn-13:", "isbn-10:", "isbn:", "isbn"),
    "doi": ("doi:", "https://doi.org/", "http://doi.org/"),
    "issn": ("issn:", "issn"),
    "orcid": ("orcid:", "https://orcid.org/", "orcid"),
    "pmid": ("pmid:", "pmid"),
    "arxiv": ("arxiv:", "arxiv"),
}

_TOKEN_PARAMS: dict[str, tuple[str, ...]] = {
    "isbn": ("isbn", "isbn13", "isbn_13", "isbn10", "isbn_10"),
    "doi": ("doi",),
    "issn": ("issn",),
    "orcid": ("orcid",),
    "pmid": ("pmid",),
    "arxiv": ("arxiv", "arxiv_id", "arxivid"),
}

# Case-insensitive token types: their canonical form folds case.
_CASE_FOLDABLE = frozenset({"isbn", "doi", "issn", "orcid", "pmid", "arxiv"})


def _surround(value: Any) -> Any:
    """Pad with whitespace — never content-bearing for a token-like value."""
    return f"  {value}  " if isinstance(value, str) else value


def _case(value: Any) -> Any:
    """Upper-case — tokens fold case, they are not free text."""
    return value.upper() if isinstance(value, str) else value


def _prefixer(token: str) -> Callable[[Any], Any]:
    """Decorate a value the way a human would, e.g. `ISBN 978-0-306-40615-7`."""
    label = token.upper()

    def transform(value: Any) -> Any:
        return f"{label} {value}" if isinstance(value, str) else value

    return transform


def normalisers_for(
    param: str, declared: Iterable[str] = ()
) -> list[tuple[str, Callable[[Any], Any]]]:
    """Metamorphic transforms that must not change the output for `param`.

    Everything here is scoped to *token-like* parameters — names that denote a
    value with a canonical form. Free text (`title`, `text`, `body`) gets
    nothing, because for those the whitespace and casing ARE the content: a
    character counter must not be told that `"  x  "` equals `"x"`.
    """
    tokens = {_token_for(param)} | {d.strip().lower() for d in declared}
    tokens.discard("")

    out: list[tuple[str, Callable[[Any], Any]]] = []
    for token in sorted(tokens):
        if token not in _TOKEN_PARAMS:
            continue
        # Whitespace padding around a token is never part of the value.
        out.append(("surrounding_whitespace", _surround))
        if _STRIP_PREFIXES.get(token):
            out.append((f"{token}_prefix", _prefixer(token)))
        if token in _CASE_FOLDABLE:
            out.append((f"{token}_case", _case))
    return out


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class InvarianceResult:
    """Outcome of the metamorphic battery."""

    name: str
    passed: bool = True
    total: int = 0
    checked: int = 0
    violations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.checked / self.total if self.total else 1.0

    def summary(self) -> str:
        return f"{self.name}: {self.detail()}"

    def detail(self) -> str:
        """The verdict without the tool name, for embedding in a wider report.

        RobustnessResult already names the tool; repeating it here stutters in
        the operator's terminal and spends tokens twice in the retry prompt.
        """
        if self.passed and not self.violations:
            return f"{self.checked}/{self.total} invariances hold"
        head = (
            f"{len(self.violations)} invariance violation(s) "
            f"({self.checked}/{self.total} checked)"
        )
        return head + "".join(f". {self._describe(v)}" for v in self.violations[:4])

    @staticmethod
    def _describe(v: dict[str, Any]) -> str:
        """One violated relation, with the evidence a retry prompt can act on.

        The bare count ("2 invariance violation(s)") teaches the model nothing:
        the next round reproduces the same code and burns the same round. Name
        the relation and show the pair of calls that disagreed.
        """
        text = f"{v.get('relation', '?')}: {v.get('detail', '')}"
        if "clean_input" in v:
            text += (
                f" [{v['clean_input']!r} -> {v['clean_output']!r}"
                f" but {v['mutated_input']!r} -> {v['mutated_output']!r}]"
            )
        elif "first" in v:
            text += f" [{v['first']!r} != {v['second']!r}]"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "passed": self.passed,
            "checked": self.checked,
            "total": self.total,
            "violations": self.violations[:5],
        }


# ---------------------------------------------------------------------------
# Calling
# ---------------------------------------------------------------------------
def _call(spec: ToolSpec, sandbox: Any, args: dict[str, Any]) -> tuple[bool, Any, str]:
    """Invoke the tool. Returns (ok, output, error)."""
    if sandbox is not None and spec.code:
        r = sandbox.run(spec.code, spec.name, args)
        return r.ok, r.output, (r.error or "")
    try:
        out = spec.fn(**args)
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, out, ""


def _same(a: Any, b: Any) -> bool:
    """Equality that tolerates the JSON round-trip the sandbox imposes."""
    if a == b:
        return True
    try:
        return str(a) == str(b) and type(a).__name__ == type(b).__name__
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------
def derive(spec: ToolSpec) -> list[str]:
    """Relations applicable to this spec, computed relations first.

    Declared relations live on `ToolSpec.invariances` (token names such as
    `"isbn"`); they are folded in here so callers see one list.
    """
    declared = [f"normalise:{t}" for t in (getattr(spec, "invariances", None) or [])]
    return list(COMPUTED_RELATIONS) + declared


def check(
    spec: ToolSpec,
    *,
    sandbox: Any = None,
    reference_args: dict[str, Any] | None = None,
    probe_args: list[dict[str, Any]] | None = None,
) -> InvarianceResult:
    """Run every applicable relation. `passed` is False if any is violated."""
    result = InvarianceResult(spec.name)
    if reference_args is None:
        reference_args = _infer_args(spec)
    probe_args = list(probe_args or [])

    def violation(relation: str, detail: str, **extra: Any) -> None:
        entry = {"relation": relation, "detail": detail}
        entry.update(extra)
        result.violations.append(entry)
        result.passed = False

    # -- defined + deterministic on the reference call --------------------
    ok, out, err = _call(spec, sandbox, reference_args)
    result.total += 2
    if ok and out is not None:
        result.checked += 1
    else:
        violation("defined", err or "returned None on the reference input")

    ok2, out2, _ = _call(spec, sandbox, reference_args)
    if ok and ok2 and _same(out, out2):
        result.checked += 1
    else:
        violation("deterministic", "two identical calls disagreed",
                  first=str(out)[:80], second=str(out2)[:80])

    # -- non_degenerate ---------------------------------------------------
    # Meaningful only against a contrast set. With no probe inputs there is
    # one output and nothing to compare it to, so the relation is not
    # evaluated rather than failed — absence of evidence is not evidence.
    if probe_args:
        result.total += 1
        outputs: list[Any] = [out] if ok else []
        for args in probe_args:
            pok, pout, _ = _call(spec, sandbox, args)
            if pok:
                outputs.append(pout)
        distinct = {_freeze(o) for o in outputs}
        if len(distinct) > 1:
            result.checked += 1
        else:
            violation(
                "non_degenerate",
                f"returned one constant value ({next(iter(distinct), None)!r}) across "
                f"valid and garbage inputs — the check is satisfiable without doing work",
                samples=len(outputs),
            )

    # -- declared normalisation relations ---------------------------------
    for transform_label, transform, param in _applicable_transforms(spec, reference_args):
        result.total += 1
        mutated = dict(reference_args)
        mutated[param] = transform(reference_args[param])
        mok, mout, merr = _call(spec, sandbox, mutated)
        if not mok:
            # A mutation that crashes the tool is a failure of the relation:
            # the decorated form is a legitimate way to write the value.
            violation(f"normalise:{transform_label}",
                      merr or "raised on a normalised variant", param=param)
            continue
        if _same(mout, out):
            result.checked += 1
        else:
            violation(
                f"normalise:{transform_label}",
                f"output changed when {param!r} was written differently — the "
                f"tool is matching on decoration, not on the value",
                param=param,
                clean_input=str(reference_args[param])[:60],
                mutated_input=str(mutated[param])[:60],
                clean_output=str(out)[:60],
                mutated_output=str(mout)[:60],
            )

    return result


def _freeze(value: Any) -> Any:
    """Hashable projection so near-identical outputs collapse together."""
    if isinstance(value, (dict, list, set, tuple)):
        try:
            import json
            return json.dumps(value, sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            return str(value)
    return value


def _applicable_transforms(
    spec: ToolSpec, reference_args: dict[str, Any]
) -> list[tuple[str, Callable[[Any], Any], str]]:
    """(label, transform, param) for every relation this spec must satisfy."""
    declared = [t for t in (getattr(spec, "invariances", None) or [])]
    out: list[tuple[str, Callable[[Any], Any], str]] = []
    seen: set[tuple[str, str]] = set()
    for param, value in reference_args.items():
        if not isinstance(value, str):
            continue
        for label, transform in normalisers_for(param, declared):
            if (label, param) in seen:
                continue
            seen.add((label, param))
            out.append((label, transform, param))
    return out


def _token_for(param: str) -> str:
    key = (param or "").strip().lower()
    for token, names in _TOKEN_PARAMS.items():
        if key in names:
            return token
    return ""


def _infer_args(spec: ToolSpec) -> dict[str, Any]:
    """Mirror of ToolVerifier._infer_args — kept local to avoid a cycle."""
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


__all__ = [
    "COMPUTED_RELATIONS",
    "InvarianceResult",
    "check",
    "derive",
    "normalisers_for",
]
