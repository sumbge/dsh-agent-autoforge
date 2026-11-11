"""The claim gate: a sixth check, and the first one that is about a *claim*.

A-E ask "does this tool run, survive, resist attack, fire when needed, and stay
quiet when not". None of them asks the question that actually costs a week:

    is the number this tool hands back a fact about the world, or is it a fact
    about the data it happened to be handed?

That question has a name in the experimental sciences -- it is what a control
is for -- and it has been answered the same way since long before software:
    (1) do the same thing twice and get the same answer,
    (2) do it to the same thing in a different order and get the same answer,
    (3) feed it a case whose right answer you already know and check that it
        says so.

This module runs those three against a freshly forged tool, using nothing but
the tool's own declared inputs. It reads the tool's parameter schema for a
*rail*: a parameter with two or more declared values that the tool is supposed
to be sensitive to (an enum, or a boolean flag). A tool with no rail is not
audited here -- and says so, rather than passing quietly.

Two shapes are recognised, because they fail in opposite directions:

  predict  f(rail=A) and f(rail=B) must differ (it must be reading its own
           input), and must each be (1) reproducible and (2) order-invariant.

  audit    the tool judges *someone else's* claim (it takes a target column and
           a prediction column). Here the rail is the control: permute the
           target column and the verdict MUST collapse. An auditor that still
           says PASS after its labels have been shuffled is not measuring
           anything, and it says PASS just as confidently on the case it was
           built to catch.

That second shape is not hypothetical. It is the exact failure this module was
written after: `audit_predictive_claim_csv`, forged on 2026-10-02, returned
VERDICT: PASS on a dataset whose prediction column was partly derived from the
label. Nine checks went green and the tool was wrong in the one way it existed
to prevent. The battery asked whether the code ran; nothing asked whether the
judgement was real.

Scope, stated so it is not overread: this is a *behavioural* gate. It runs the
tool and perturbs its inputs. It does not read the code, and a tool that
hard-codes its answers to survive exactly these perturbations would pass. That
is a real limit, and it is the same limit every test suite has.
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import re
import shutil
import string
import tempfile
from dataclasses import dataclass, field
from typing import Any

from ..tools.spec import ToolSpec

#: Values a rail parameter is allowed to take when the schema does not say.
_BOOL_TRUE = ("true", "yes", "on", "1")


@dataclass
class ClaimFinding:
    """One perturbation, and whether the tool behaved like a measurement."""

    name: str
    ok: bool
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok,
                "detail": self.detail, "evidence": self.evidence}


@dataclass
class ClaimReport:
    tool: str
    kind: str = "none"            # predict | audit | none
    applicable: bool = False
    passed: bool = True
    rail: str = ""
    #: Something was asked and nothing could be measured. Same meaning as in
    #: `verifier.CheckResult`: un-probed is not passed, and a report that says
    #: otherwise is a claim the gate cannot support.
    unprobed: bool = False
    #: Set only when `rail` came from this gate's own difference probe rather
    #: than from the tool's schema. Explicit rather than inferred from
    #: `kind == "none"`: the cosmetic-rail branch also runs with kind "none"
    #: and takes its rail name straight out of the declaration, so inferring
    #: printed "synthesised" about a rail the tool itself had declared.
    rail_synthesised: bool = False
    #: Set when the tool declared the varying argument as `x-claim: path-only`,
    #: i.e. it says its answer is a function of the path and not of the file's
    #: bytes. The file probe is then not the right instrument -- it holds the
    #: path still on purpose -- so a different pair of checks runs instead, and
    #: both the summary and the ledger have to say which instrument was used.
    path_only: bool = False
    findings: list[ClaimFinding] = field(default_factory=list)

    @property
    def failed(self) -> list[ClaimFinding]:
        return [f for f in self.findings if not f.ok]

    def summary(self) -> str:
        if self.unprobed:
            return f"{self.tool}: claim gate un-probed ({self.kind}, out of scope)"
        if not self.applicable:
            return f"{self.tool}: claim gate not applicable (no rail declared)"
        ok = sum(1 for f in self.findings if f.ok)
        if self.path_only and self.rail:
            return (f"{self.tool}: claim gate {ok}/{len(self.findings)} "
                    f"(path-axis probe, rail={self.rail!r} declared path-only)")
        # A rail this gate synthesised from a file-like argument is not the
        # tool's own declaration, and the summary has to say which it was.
        # Otherwise `kind=none` next to `rail=path` reads as a contradiction.
        if self.rail_synthesised and self.rail:
            return (f"{self.tool}: claim gate {ok}/{len(self.findings)} "
                    f"(difference probe, rail={self.rail!r} synthesised)")
        return f"{self.tool}: claim gate {ok}/{len(self.findings)} ({self.kind}, rail={self.rail})"

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "kind": self.kind, "applicable": self.applicable,
                "passed": self.passed, "rail": self.rail, "unprobed": self.unprobed,
                "rail_synthesised": self.rail_synthesised,
                "path_only": self.path_only,
                "findings": [f.to_dict() for f in self.findings]}


# -- reading the tool's own declarations ----------------------------------

def _is_cosmetic(schema: dict[str, Any]) -> bool:
    """Does the tool declare this parameter as one nothing may depend on?

    A flag that only affects formatting is a real design, and a gate that
    insists every boolean moves the answer would be inventing a requirement.
    But that is the *tool's* claim to make, in a way a machine can read, not
    something the gate may assume on its behalf -- assuming it is what turned
    the sensitivity check into a comment. `"x-claim": "cosmetic"` is that
    declaration.
    """
    value = schema.get("x-claim", schema.get("x_claim", ""))
    return str(value).strip().lower() == "cosmetic"


def _is_path_only(schema: dict[str, Any]) -> bool:
    """Does the tool declare that its answer is about the path, not the file?

    Measured 2026-11-01, `chk_pathaxis.py`: a tool answering `{"len": len(p),
    "stem": ...}` moves when the *path* changes and not when the file's bytes
    change, and the file probe had no way to say so -- it failed with "reporting
    a constant about itself", which is false. Both a path measurer and a
    constant return the same three answers when the probe holds the path still.

    Rather than guess, this follows the same route `x-claim: cosmetic` already
    takes: the tool says in machine-readable form what its answer is a function
    of, and the gate holds it to that. Declaring it is not free -- the file
    probe is waived, so the answer must move with the path instead, and the
    waiver is written into the report and the ledger.
    """
    value = schema.get("x-claim", schema.get("x_claim", ""))
    return str(value).strip().lower() in ("path-only", "path_only", "path")


def find_rails(spec: ToolSpec) -> list[tuple[str, list[Any], bool]]:
    """Every declared rail: (name, values, declared_cosmetic), in schema order."""
    props = (spec.parameters or {}).get("properties") or {}
    out: list[tuple[str, list[Any], bool]] = []
    for name, schema in props.items():
        if not isinstance(schema, dict):
            continue
        enum = schema.get("enum")
        if isinstance(enum, list) and len(enum) >= 2:
            out.append((name, list(enum), _is_cosmetic(schema)))
        elif schema.get("type") == "boolean":
            out.append((name, [True, False], _is_cosmetic(schema)))
    return out


def find_rail(spec: ToolSpec) -> tuple[str, list[Any]]:
    """The parameter the tool is supposed to be sensitive to, and its values.

    An enum is a declaration with two or more named values; a boolean is the
    two-valued degenerate case. First match wins, in schema order, so the
    choice is stable across runs rather than dependent on dict ordering.

    A parameter the tool declares cosmetic (`"x-claim": "cosmetic"`) is not a
    rail: it is the tool saying in advance that nothing may depend on it, which
    is a statement this gate can hold it to.
    """
    for name, values, cosmetic in find_rails(spec):
        if not cosmetic:
            return name, values
    return "", []


def _names(spec: ToolSpec) -> str:
    props = (spec.parameters or {}).get("properties") or {}
    return " ".join([spec.name, getattr(spec, "description", "") or "",
                     " ".join(props.keys())]).lower()


def find_kind(spec: ToolSpec) -> str:
    """predict / audit / none, from what the tool declares about itself.

    "audit" is claimed only when the tool takes both a truth column and a
    prediction column: that pair is what makes it a judge of somebody else's
    claim rather than a producer of its own.
    """
    text = _names(spec)
    target_like = any(k in text for k in ("target_col", "label_col", "truth", "ground_truth"))
    pred_like = any(k in text for k in ("pred_col", "prediction_col", "pred", "score_col", "model_score"))
    if target_like and pred_like:
        return "audit"
    if find_rail(spec)[0]:
        return "predict"
    return "none"


# -- running the tool under perturbation ----------------------------------

_VERDICT_RE = re.compile(r"(VERDICT\s*[:=]\s*)([A-Za-z_]+)", re.I)
_NUM_RE = re.compile(r"-?\d+\.\d+|\b\d+\.\d+\b|-?\d{2,}")


def normalise(out: Any) -> str:
    s = str(out).strip()
    s = re.sub(r"\s+", " ", s)
    return s.replace("-0.000", "0.000").replace("-0.0 ", "0.0 ")


def verdict_of(out: Any) -> str:
    m = _VERDICT_RE.search(str(out))
    return m.group(2).upper() if m else ""


def first_number(out: Any) -> float | None:
    m = _NUM_RE.search(str(out))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _csv_arg(args: dict[str, Any]) -> str | None:
    """Which argument, if any, carries a CSV the tool will read."""
    for k, v in args.items():
        if not isinstance(v, str):
            continue
        low = k.lower()
        if low in ("csv", "csv_text", "data", "data_csv", "table") and "," in v and "\n" in v:
            return k
        if low in ("path", "file", "csv_path", "filepath") and v.lower().endswith(".csv") and os.path.exists(v):
            return k
    return None


def _read_csv_text(value: str) -> str:
    if "," in value and "\n" in value:
        return value
    with open(value, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _text_args(spec: ToolSpec, args: dict[str, Any]) -> list[str]:
    """REQUIRED string parameters this gate may vary, most binding first.

    Required only, and the reason is asymmetric cost. A tool that refuses to run
    without a parameter has promised to read it, so an answer that does not move
    when that parameter changes is evidence against it. An OPTIONAL parameter is
    a different promise: the tool has a default and may legally prefer it,
    ignoring whatever the caller handed over. Accusing a tool on that basis
    would be inventing a requirement -- the same mistake that produced the
    `x-claim: path-only` false accusation next door.

    The price of being wrong here is bounded and knows its direction: a tool
    whose only string argument is optional keeps the verdict it had before this
    instrument existed (`un-probed`, which fails). So no tool can be newly
    accused by this probe; the tools whose verdict changes are exactly the ones
    that were previously condemned for having nothing measurable.

    Also excludes `x-claim: cosmetic` (the tool says nothing may depend on it)
    and `x-claim: path-only` (a different instrument judges those).

    Measured 2026-11-04: before this existed, a tool taking `{"text": ...}`
    matched `_path_args` nowhere, so the gate reported "nothing it can perturb"
    and three tools with three different behaviours -- honest, constant,
    length-measuring -- came back with one identical verdict. Also measured that
    day: the first version of THIS function walked optional parameters too,
    which its own docstring had already argued against. The code lost that
    argument and the docstring was right.
    """
    params = spec.parameters or {}
    props = params.get("properties") or {}
    required = [str(r) for r in (params.get("required") or []) if str(r) in props]
    out: list[str] = []
    for name in required:
        schema = props.get(name)
        if not isinstance(schema, dict) or schema.get("type") != "string":
            continue
        if _is_cosmetic(schema) or _is_path_only(schema):
            continue
        value = args.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        out.append(name)
    return out


# Only letters and digits, and every character maps to a DIFFERENT one, so a
# rewrite of the same length keeps every length-derived answer still and moves
# every content-derived one.
_TEXT_SWAPS = str.maketrans(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "bcdefghijklmnopqrstuvwxyzaBCDEFGHIJKLMNOPQRSTUVWXYZA1234567890")


def _text_variants(raw: str) -> list[tuple[str, str, str]]:
    """(axis, label, text) triples to try against the tool's own sample.

    Two axes, and the gate needs both. A counter of a specific character is
    unmoved by a rotation that preserves the count, and a length measurer is
    unmoved by any rewrite that keeps the length; requiring both would fail an
    honest tool of either kind. One axis moving is enough to prove the answer
    is about the input, which is all this probe claims.

    Nothing here is random on purpose: a probe whose inputs differ run to run
    makes a failure impossible to reproduce.
    """
    out: list[tuple[str, str, str]] = []
    rotated = raw.translate(_TEXT_SWAPS)
    if rotated and rotated != raw:
        out.append(("difference", "same length, every letter shifted", rotated))
    reversed_ = raw[::-1]
    if reversed_ and reversed_ != raw:
        out.append(("difference", "same length, reversed", reversed_))
    doubled = raw + raw
    out.append(("length", "the same text repeated", doubled))
    for suffix in ("abcdef", "0123456789"):
        out.append(("length", f"longer by {len(suffix)} characters", raw + suffix))
    seen: set[str] = set()
    unique: list[tuple[str, str, str]] = []
    for axis, label, value in out:
        if value in seen or value == raw:
            continue
        seen.add(value)
        unique.append((axis, label, value))
    return unique


def _number_args(spec: ToolSpec, args: dict[str, Any]) -> list[tuple[str, Any]]:
    """REQUIRED numeric parameters this gate may vary, with their sample values.

    Same rule as `_text_args` and for the same reason: required only. A
    parameter the tool refuses to run without is one it has promised to read,
    so an answer that does not move when that parameter changes is evidence
    against it. An optional numeric parameter is a promise the tool never made
    -- it has a default and may legally prefer it.

    Excludes `x-claim: cosmetic` (the tool says nothing may depend on it) and
    `x-claim: path-only` (a different instrument judges those). Booleans are
    excluded on purpose: `find_rails` already turns a required boolean into a
    two-valued rail, so the rail-sensitivity probe owns them.

    Measured 2026-11-05, before this existed: three tools whose only required
    parameter was `{"n": {"type": "integer"}}` -- `n * 2`, `return 42`, and
    `len(str(n))` -- all came back `applicable=False unprobed=True` with the
    single finding `no_rail`. Identical verdicts because no measurement
    happened, not because the tools were alike. `_path_args` matches file-like
    names and `_text_args` matches `type == "string"`; an integer fell between
    the two instruments.
    """
    params = spec.parameters or {}
    props = params.get("properties") or {}
    required = [str(r) for r in (params.get("required") or []) if str(r) in props]
    out: list[tuple[str, Any]] = []
    for name in required:
        schema = props.get(name)
        if not isinstance(schema, dict):
            continue
        kind = schema.get("type")
        if kind not in ("integer", "number"):
            continue
        if _is_cosmetic(schema) or _is_path_only(schema):
            continue
        value = args.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if kind == "integer" and not isinstance(value, int):
            continue
        out.append((name, value))
    return out


def _number_variants(raw: Any, *, integer: bool) -> list[tuple[str, str, Any]]:
    """(axis, label, value) triples to try against the tool's own sample.

    Two axes, for the reason the text probe has two. A tool reading the VALUE
    is moved by a nearby value; a tool reading only the WIDTH (a digit count, a
    log-scale bucket) is moved only by a scaling. Requiring both would fail an
    honest tool of either kind, so one axis moving is enough -- which is all
    this probe claims: the answer is about the input, not a constant about the
    code.

    Values stay in the declared type: an `integer` parameter never receives a
    float from here, because a tool that does `int(k["n"])` would raise on one
    and the gate would record a refusal as unusable rather than as evidence.

    Nothing here is random: a probe whose inputs differ run to run makes a
    failure impossible to reproduce.
    """
    cands: list[tuple[str, str, Any]] = []
    for delta in (1, -1, 2):
        cands.append(("difference", "shifted by %+d" % delta, raw + delta))
    for factor in (2, 3):
        cands.append(("magnitude", "multiplied by %d" % factor, raw * factor))
    cands.append(("difference", "negated", -raw))
    if not integer:
        cands.append(("magnitude", "halved", raw / 2.0))

    seen: set[Any] = set()
    unique: list[tuple[str, str, Any]] = []
    for axis, label, value in cands:
        if isinstance(value, bool):
            continue
        if integer and not isinstance(value, int):
            continue
        if value == raw or value in seen:
            continue
        seen.add(value)
        unique.append((axis, label, value))
    return unique

def _path_args(spec: ToolSpec) -> list[str]:
    """Parameter names that plausibly name a file the tool is supposed to read."""
    props = (spec.parameters or {}).get("properties") or {}
    out: list[str] = []
    for name, schema in props.items():
        if not isinstance(schema, dict) or schema.get("type") not in (None, "string"):
            continue
        low = name.lower()
        if low in ("path", "file", "filepath", "file_path", "target", "source",
                   "filename", "file_name", "dir", "directory", "folder"):
            out.append(name)
    return out


def _strip_paths(out: Any, *paths: str) -> str:
    """Remove the probe identifiers from an answer before comparing it.

    Two probes must be distinguishable *by their contents*, not by their
    names. Without this, a tool that merely echoes the path it was given --
    or dumps its whole argument dict -- produces two different strings for
    the trivial reason that the paths differ, and the difference check
    would clear a tool that measured nothing. Measured 2026-10-31: the
    constant-returning liar was cleared by exactly that, because its dict
    carried `path` in it.
    """
    s = str(out)
    # Longest needle first, and every needle in both raw and repr-escaped form.
    #
    # Measured 2026-11-01: a tool that returned nothing but the path it was
    # handed -- `{"ok": True, "path": path}` -- PASSED this gate. The probes
    # live in three separate directories (`_probe_dir`), so their directory
    # names differ; substituting dirname, then basename, then the whole path
    # left the random directory name in place, because the earlier basename
    # pass had already destroyed the whole-path needle. Three probes duly
    # produced three different strings and the gate read that as "the answer
    # moved with the file". Replacing the full path before its parts, and
    # handling the doubled backslashes of a repr, removes the only thing that
    # made those three
    # outputs differ.
    pairs: list[tuple[str, str]] = []
    for p in paths:
        if not p:
            continue
        for form in {p, p.replace("\\", "\\\\")}:
            pairs.append((form, "<probe>/<file>"))
            d = os.path.dirname(form)
            if d:
                pairs.append((d, "<probe>"))
                # The *last* component of the directory, not just the whole
                # path. Measured 2026-11-01: a tool answering
                # `{'d': basename(dirname(path)), 'bytes': 123456, 'lines': 789}`
                # -- constant bytes, constant lines, one path-derived string --
                # passed 3/3. The three probe files live in three `_probe_dir`
                # directories, so `d` differed across them, and the gate read
                # that as "the answer moved with the file". A directory's own
                # name is as much a property of the path as the path is.
                dd = os.path.basename(d)
                if dd:
                    pairs.append((dd, "<probe>"))
        base = os.path.basename(p)
        if base:
            pairs.append((base, "<file>"))
        # Every remaining path component, so that no fragment of a probe path
        # survives to be echoed back as (fake) evidence of measurement. Drive
        # and root tokens are left alone: they are two characters and would
        # match inside ordinary answers.
        for part in re.split(r"[\\/]+", p):
            if len(part) >= 4 and not part.endswith(":"):
                pairs.append((part, "<probe-part>"))
    for needle, repl in sorted(pairs, key=lambda kv: len(kv[0]), reverse=True):
        if needle:
            s = s.replace(needle, repl)
    return s


def _rows(text: str) -> list[list[str]]:
    return [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]


def _permuted_csv(text: str, *, mode: str, column: str = "", seed: int = 11) -> str:
    """mode='rows' shuffles the data rows; mode='column' shuffles one column
    independently, which is the label-permutation control."""
    rows = _rows(text)
    if len(rows) < 3:
        return text
    head, body = rows[0], rows[1:]
    rnd = random.Random(seed)
    if mode == "rows":
        rnd.shuffle(body)
    else:
        idx = None
        for i, h in enumerate(head):
            if h.strip().lower() == column.strip().lower():
                idx = i
                break
        if idx is None:
            return text
        col = [r[idx] if idx < len(r) else "" for r in body]
        rnd.shuffle(col)
        for r, v in zip(body, col):
            if idx < len(r):
                r[idx] = v
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(head)
    w.writerows(body)
    return buf.getvalue()


def _known_answer_csv(text: str, target_col: str, pred_col: str, *, mode: str,
                      seed: int = 17) -> str:
    """Rewrite only the prediction column, leaving columns, order and size alone."""
    rows = _rows(text)
    if len(rows) < 3:
        return text
    head, body = rows[0], rows[1:]
    ti = pi = None
    low = [h.strip().lower() for h in head]
    for i, h in enumerate(low):
        if h == target_col.strip().lower():
            ti = i
        if h == pred_col.strip().lower():
            pi = i
    if ti is None or pi is None:
        return text
    rnd = random.Random(seed)
    vals = [(r[ti] if ti < len(r) else "0") for r in body]
    if mode == "noise":
        # Keep the prediction column's own marginal, destroy its link to the
        # answer: a shuffle of the predictions against the rows.
        preds = [(r[pi] if pi < len(r) else "0") for r in body]
        rnd.shuffle(preds)
        vals = preds
    for r, v in zip(body, vals):
        while len(r) <= pi:
            r.append("")
        r[pi] = v
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(head)
    w.writerows(body)
    return buf.getvalue()


_R_RE = re.compile(r"\br\s*[=:]\s*(-?\d+(?:\.\d+)?)", re.I)


def _reported_r(out: Any) -> float | None:
    """The correlation the tool itself reports, if it reports one."""
    m = _R_RE.search(str(out))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return first_number(out)


def _spike_csv(text: str, target_col: str, pred_col: str, *, alpha: float = 0.5) -> str:
    """pred := (1-alpha)*pred + alpha*target, with the target rescaled onto the
    prediction's own range so the mixture is meaningful whatever the units are."""
    rows = _rows(text)
    if len(rows) < 3:
        return text
    head, body = rows[0], rows[1:]
    low = [h.strip().lower() for h in head]
    ti = pi = None
    for i, h in enumerate(low):
        if h == target_col.strip().lower():
            ti = i
        if h == pred_col.strip().lower():
            pi = i
    if ti is None or pi is None:
        return text
    def nums(idx):
        out = []
        for r in body:
            try:
                out.append(float(r[idx]))
            except Exception:                                     # noqa: BLE001
                out.append(None)
        return out
    tv, pv = nums(ti), nums(pi)
    tvv = [x for x in tv if x is not None]
    pvv = [x for x in pv if x is not None]
    if not tvv or not pvv:
        return text
    lo_p, hi_p = min(pvv), max(pvv)
    lo_t, hi_t = min(tvv), max(tvv)
    span = (hi_t - lo_t) or 1.0
    for i, r in enumerate(body):
        if tv[i] is None or pv[i] is None:
            continue
        scaled = lo_p + (tv[i] - lo_t) / span * ((hi_p - lo_p) or 1.0)
        while len(r) <= pi:
            r.append("")
        r[pi] = repr(round((1 - alpha) * pv[i] + alpha * scaled, 6))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(head)
    w.writerows(body)
    return buf.getvalue()


def _target_column(args: dict[str, Any], spec: ToolSpec) -> str:
    """The truth column the tool was pointed at, taken from the call itself."""
    for k, v in args.items():
        if isinstance(v, str) and k.lower() in ("target_col", "label_col", "truth", "ground_truth"):
            return v
    props = (spec.parameters or {}).get("properties") or {}
    for name in props:
        if name.lower() in ("target_col", "label_col", "truth", "ground_truth"):
            return str((props.get(name) or {}).get("default") or "")
    return ""


class ClaimGate:
    """Runs the three controls against a tool's own declared behaviour."""

    def __init__(self, sandbox, *, run: bool = True, timeout_headroom: float = 2.0) -> None:
        self.sandbox = sandbox
        self.run = run

    # -- one call, with the perturbation applied --------------------------
    def _call(self, spec: ToolSpec, args: dict[str, Any]):
        sandbox = self.sandbox
        if sandbox is None:
            return None
        return sandbox.run(spec.code, spec.name, args)

    def _no_rail_findings(self, spec: ToolSpec, rep: ClaimReport,
                          sample_args: dict[str, Any] | None = None) -> None:
        """What to do with a tool that declares no enum/boolean rail.

        Before 2026-10-31 this branch appended `("no_rail", True, ...)` and
        returned: the whole claim gate printed as satisfied for every tool whose
        parameters were plain strings. Measured consequence on that date --
        a tool that never opened a file returned `bytes=123456` for
        `C:\\Windows\\win.ini` (truly 92) and scored 5/6 checks green, the only
        failure being `trigger`, which asks whether the tool gets *used* rather
        than whether it *measures*.

        A string parameter is not "nothing to perturb". It is an input, and an
        input can be varied: give the tool two files whose contents differ
        while their names, sizes and formats do not, and a tool that actually
        reads its argument must answer differently. That is the same
        rail-sensitivity question, asked of a rail the schema cannot express.

        When there is no file argument either -- an arithmetic tool on two
        numbers, a formatter -- this still does not invent a verdict: it reports
        the check as un-probed, and un-probed does not pass (the `unprobed`
        rule in `verifier` says exactly this).
        """
        names = _path_args(spec)
        # `sample_args` first, for the same reason `gate()` prefers it: the
        # caller (verifier.verify, service.evaluate_tool) passes the sample it
        # actually intends to run, and a spec's stored sample_call may be empty.
        # Measured 2026-11-01: this line read `dict(spec.sample_call or {})`,
        # so every string-parameter tool reaching this branch with sample_args
        # supplied -- i.e. every one that goes through verify() -- reported
        # "no file-like argument to vary" and was scored un-probed. The probe
        # existed; this line was looking in the wrong place for its input.
        args = dict(sample_args or spec.sample_call or {})
        texts = _text_args(spec, args)
        # One instrument is enough: if a text argument can be varied, the
        # answer it produces is already under test. Numbers come second only
        # so that a tool taking both keeps the verdict it had before this
        # instrument existed.
        numbers = _number_args(spec, args)
        if not names or not args:
            if texts:
                self._text_axis_findings(spec, rep, args, texts[0], args[texts[0]])
                return
            if numbers:
                _n, _v = numbers[0]
                _props = (spec.parameters or {}).get("properties") or {}
                _schema = _props.get(_n)
                _int = isinstance(_schema, dict) and _schema.get("type") == "integer"
                self._number_axis_findings(spec, rep, args, _n, _v, integer=_int)
                return
            rep.findings.append(ClaimFinding(
                "no_rail", False,
                "no declared rail (an enum or boolean parameter, or a "
                "target/prediction column pair), no file-like argument, no "
                "required string argument, and no required numeric argument "
                "carrying a usable sample value, so this gate has nothing it "
                "can perturb. Un-probed is not "
                "passed: nothing here establishes that the answer is about "
                "the input rather than about the code that formatted it"))
            rep.unprobed = True
            return

        name = names[0]
        raw = args.get(name)
        if not isinstance(raw, str) or not raw:
            # A file-like NAME with no file in the sample is not the end of the
            # gate: if the tool also takes text, that instrument still applies.
            # Ordered this way because a file probe on a tool that never opened
            # the path would be judging the wrong argument.
            if texts:
                self._text_axis_findings(spec, rep, args, texts[0], args[texts[0]])
                return
            if numbers:
                _n, _v = numbers[0]
                _props = (spec.parameters or {}).get("properties") or {}
                _schema = _props.get(_n)
                _int = isinstance(_schema, dict) and _schema.get("type") == "integer"
                self._number_axis_findings(spec, rep, args, _n, _v, integer=_int)
                return
            rep.findings.append(ClaimFinding(
                "no_rail", False,
                f"file-like parameter {name!r} carries no usable sample value "
                f"({raw!r}), so the difference probe could not be built. "
                f"Un-probed is not passed"))
            rep.unprobed = True
            return

        # A tool that says its answer is a function of the path rather than of
        # the file is making a different claim, and the file probe cannot judge
        # it: that probe deliberately holds the path still, so a path measurer
        # and a constant give the same three answers. Judging it with the file
        # probe and calling the result "a constant about itself" is a false
        # accusation -- measured 2026-11-01 in `chk_pathaxis.py`, where
        # `{"len": len(p)}` and `{"bytes": 123456}` landed on the same verdict.
        #
        # So the declaration is honoured and paid for: the file probe is waived
        # in writing, and the answer must instead move with the path. A constant
        # still fails that, and a file reader that mis-declares itself fails it
        # too (its answer does not move when only the path does).
        props = (spec.parameters or {}).get("properties") or {}
        if isinstance(props.get(name), dict) and _is_path_only(props[name]):
            self._path_axis_findings(spec, rep, args, raw, name)
            return

        made: list[str] = []
        try:
            pair = self._twin_files(raw)
            if pair is None:
                rep.findings.append(ClaimFinding(
                    "no_rail", False,
                    f"could not build two same-size, different-content probes "
                    f"for {name!r}: its declared sample ({raw!r}) is neither a "
                    f"readable file nor a writable directory. Un-probed is not "
                    f"passed -- forge with a real path in sample_call to make "
                    f"this check mean something"))
                rep.unprobed = True
                return
            a_path, b_path, c_path = pair
            # The probe exists and is about to run, so this gate IS judging this
            # tool. Saying "not applicable (no rail declared)" from here on was
            # a label contradicting its own findings -- measured 2026-11-01: a
            # file-reading tool printed `claim: not applicable` directly above
            # `claim/difference: answers moved with the file`. The work had been
            # done and the summary said it had not.
            rep.applicable = True
            rep.rail = name
            rep.rail_synthesised = True
            # rmtree wants the directories `_twin_files` made, not the files
            # inside them: the whole probe tree goes, so a failing tool cannot
            # leave litter in the temp dir it was pointed at.
            made = [os.path.dirname(p) for p in (a_path, b_path, c_path)]

            a_args = dict(args)
            a_args[name] = a_path
            b_args = dict(args)
            b_args[name] = b_path
            c_args = dict(args)
            c_args[name] = c_path

            ok_a, out_a = self._meaningful(self._call(spec, a_args))
            if not ok_a:
                rep.findings.append(ClaimFinding(
                    "baseline", False, f"probe A on {os.path.basename(a_path)}: {out_a}"))
                return
            rep.findings.append(ClaimFinding(
                "baseline", True,
                f"ran on a filled probe file ({os.path.basename(a_path)})",
                {"output": out_a[:200]}))

            ok_b, out_b = self._meaningful(self._call(spec, b_args))
            if not ok_b:
                rep.findings.append(ClaimFinding(
                    "difference", False,
                    f"probe B failed where probe A succeeded: {out_b}. A tool that "
                    f"answers one file and not an identically-shaped one is not "
                    f"reading its argument reliably"))
                return

            # What a difference here can and cannot prove.
            #
            # Two probes: same byte length, same basename, same extension,
            # different bytes. So the *only* thing that distinguishes them is
            # content. A tool that measures the file may legitimately answer
            # the same for both -- a size tool reports 16 and 16 and is right
            # to. A tool that measures nothing answers the same for both and
            # for the wrong reason. One probe pair cannot separate those two.
            #
            # A third answer settles it: a file of a DIFFERENT size. If the
            # tool's output moves between "same size, other bytes" and
            # "different size", it is reading the file. That is a pass, and it
            # is a pass for the honest size tool and the honest hash tool
            # alike, while a constant answers all three identically and fails.
            ok_c, out_c = self._meaningful(self._call(spec, c_args))
            if not ok_c:
                rep.findings.append(ClaimFinding(
                    "size_sensitivity", False,
                    f"probe C failed where A and B succeeded: {out_c}"))
                return
            norm_a = normalise(_strip_paths(out_a, a_path, b_path, c_path))
            norm_b = normalise(_strip_paths(out_b, a_path, b_path, c_path))
            norm_c = normalise(_strip_paths(out_c, a_path, b_path, c_path))

            moved_on_size = norm_a != norm_c
            moved_on_content = norm_a != norm_b
            rep.findings.append(ClaimFinding(
                "difference", bool(moved_on_size or moved_on_content),
                # What this check proves, and what it does not.
                #
                # The old wording said "this is a measurement of the input, not
                # a constant about the code" -- and that is more than the
                # evidence supports. Measured 2026-11-01, `chk_diag_small.py`:
                # a tool that reports the true size under 100 bytes and
                # `123456` above it answered `92` for the declared win.ini,
                # `46` for the shrunken copy, and `123456` for the grown one.
                # The answer DID move, so this check passed it -- on the axis
                # where it lies. A difference is necessary for honesty and not
                # sufficient for it: a liar whose falsehood is a step function
                # moves its answer too. What the gate can honestly say is that
                # the output is not a constant; the values being right is what
                # `sample_expect` and the falsification battery are for.
                (f"the answer is not a constant: it moved with the file "
                 f"({', '.join(k for k, v in (('size', moved_on_size), ('content', moved_on_content)) if v)}). "
                 f"Necessary for an honest measurement, not sufficient -- this "
                 f"does not say the values are correct"
                 if (moved_on_size or moved_on_content) else
                 (f"three probe files -- two of equal size with different bytes, "
                  f"one of a different size -- produced the identical answer "
                  f"({norm_a[:120]!r}). A tool whose answer does not move when "
                  f"the file's size and contents do is not measuring the file; "
                  f"it is reporting a constant about itself. This is the check "
                  f"that a hard-coded `bytes=123456` cannot pass")),
                {"same_size_different_bytes": norm_a != norm_b,
                 "different_size": norm_a != norm_c,
                 "A": norm_a[:120], "B": norm_b[:120], "C": norm_c[:120]}))

            # The generality check: does it behave the same on a file it did
            # not create?
            #
            # The three probes above are built by this gate and therefore share
            # properties no ordinary file has. Measured 2026-11-01, four liars
            # that PASSED -- each answered correctly for the probes and
            # `bytes=123456` for every real file -- by branching on exactly
            # those properties: the path being under `tempfile.gettempdir()`,
            # the parent directory being `tmp`+8 chars, the file being under
            # 100 bytes, and the file being printable ASCII. Each of those is a
            # one-line test a tool could carry by accident, too.
            #
            # A real file the caller already trusts, measured the same way and
            # deposited in the same ledger, closes all four at once: what a liar
            # detects in `probe.txt` is not what it sees in `win.ini`. Only the
            # DECLARED sample is used, so this adds no new read to the host --
            # the tool was going to open it anyway.
            self._generality_probe(spec, rep, args, raw, name)
        finally:
            for p in made:
                shutil.rmtree(p, ignore_errors=True)

    def _text_axis_findings(self, spec: ToolSpec, rep: ClaimReport,
                            args: dict[str, Any], name: str, raw: str) -> None:
        """Judge a tool whose required input is a string it is meant to read.

        The file probe cannot do this: it decides whether to run by matching
        the parameter NAME against a list of file-like names, so `text` never
        reaches it. Measured 2026-11-04, three tools taking `{"text": ...}` --
        an honest vowel counter, one that returns a hard-coded 3, and one that
        returns `len(text)` -- all came back `passed=False applicable=False
        unprobed=True` with the single finding `no_rail`. The verdict was
        identical because no measurement happened, not because the tools were
        alike.

        The probe is the file probe's shape with the input varied directly:

          baseline    it answers its own declared sample
          difference  a same-length rewrite with different letters moves it
          length      a longer input moves it

        and the verdict is `difference or length`, because neither axis alone
        can be required of every honest tool: a counter of one character is
        unmoved by a rotation that preserves the count, and a length measurer
        is unmoved by any rewrite that keeps the length.

        What a pass proves is bounded, and said so in the finding text: the
        answer is not a constant about the code. It does NOT say the answer is
        correct -- a liar that answers `len(text) % 7` moves on both axes. The
        values being right is what `sample_expect` and the falsification
        battery are for, and the same limit is already written into the file
        probe's `difference` wording.
        """
        rep.applicable = True
        rep.rail = name
        rep.rail_synthesised = True

        ok_base, out_base = self._meaningful(self._call(spec, args))
        if not ok_base:
            rep.findings.append(ClaimFinding(
                "baseline", False,
                f"probe on the declared sample for {name!r} did not answer: {out_base}"))
            return
        rep.findings.append(ClaimFinding(
            "baseline", True,
            f"ran on its own declared sample ({name!r}, {len(raw)} characters)",
            {"output": out_base[:200], "chars": len(raw)}))

        norm_base = normalise(out_base)
        moved_by: list[str] = []
        tried: list[str] = []
        unusable: list[str] = []
        for axis, label, value in _text_variants(raw):
            probe = dict(args)
            probe[name] = value
            ok, out = self._meaningful(self._call(spec, probe))
            if not ok:
                # The tool refused an input this gate invented. That is not
                # evidence about the tool -- a parser may legitimately reject a
                # string that is not the shape it documents -- so it is recorded
                # and skipped rather than counted as a failure.
                unusable.append(f"{label}: {out}")
                continue
            tried.append(label)
            if normalise(out) != norm_base:
                moved_by.append(f"{label} -> {normalise(out)[:80]!r}")

        if not tried:
            rep.findings.append(ClaimFinding(
                "difference", False,
                f"no varied input for {name!r} was accepted, so nothing was "
                f"measured: {len(unusable)} probe(s) came back unusable "
                f"({'; '.join(unusable[:3])}). Un-probed is not passed"))
            rep.unprobed = True
            return

        rep.findings.append(ClaimFinding(
            "difference", bool(moved_by),
            (f"the answer is not a constant: it moved when the text changed "
             f"({'; '.join(moved_by[:3])}). Necessary for an honest reading of "
             f"the input, not sufficient -- this does not say the values are "
             f"correct"
             if moved_by else
             f"{len(tried)} rewritten input(s) -- same length, longer, and the "
             f"same text repeated -- all produced the identical answer "
             f"({norm_base[:120]!r}). A tool whose answer does not move when "
             f"its required {name!r} argument changes is not reading that "
             f"argument; it is reporting a constant about itself"),
            {"sample": norm_base[:120], "moved_by": moved_by,
             "tried": tried, "unusable": unusable[:5]}))

    def _number_axis_findings(self, spec: ToolSpec, rep: ClaimReport,
                              args: dict[str, Any], name: str, raw: Any,
                              *, integer: bool) -> None:
        """Judge a tool whose required input is a number it is meant to read.

        The same shape as the text probe, over the other common parameter type:

          baseline    it answers its own declared sample
          difference  a nearby value moves the answer
          magnitude   a scaled value moves the answer

        and the verdict is `difference or magnitude`. What a pass proves is
        bounded and stated in the finding text: the answer is not a constant
        about the code. It does NOT say the answer is correct -- `n * 2 + 1`
        moves on both axes and is wrong. Correctness is `sample_expect` and the
        falsification battery.
        """
        rep.applicable = True
        rep.rail = name
        rep.rail_synthesised = True

        ok_base, out_base = self._meaningful(self._call(spec, args))
        if not ok_base:
            rep.findings.append(ClaimFinding(
                "baseline", False,
                f"probe on the declared sample for {name!r} did not answer: {out_base}"))
            return
        rep.findings.append(ClaimFinding(
            "baseline", True,
            f"ran on its own declared sample ({name!r} = {raw!r})",
            {"output": out_base[:200], "value": raw}))

        norm_base = normalise(out_base)
        moved_by: list[str] = []
        tried: list[str] = []
        unusable: list[str] = []
        for axis, label, value in _number_variants(raw, integer=integer):
            probe = dict(args)
            probe[name] = value
            ok, out = self._meaningful(self._call(spec, probe))
            if not ok:
                # The tool refused a value this gate invented. A parser may
                # legitimately reject an out-of-range number, so this is
                # recorded and skipped rather than counted against the tool.
                unusable.append(f"{label}: {out}")
                continue
            tried.append(label)
            if normalise(out) != norm_base:
                moved_by.append(f"{label} -> {normalise(out)[:80]!r}")

        if not tried:
            seen_unusable = "; ".join(unusable[:3])
            rep.findings.append(ClaimFinding(
                "difference", False,
                f"no varied value for {name!r} was accepted, so nothing was "
                f"measured: {len(unusable)} probe(s) came back unusable "
                f"({seen_unusable}). Un-probed is not passed"))
            rep.unprobed = True
            return

        seen_moved = "; ".join(moved_by[:3])
        rep.findings.append(ClaimFinding(
            "difference", bool(moved_by),
            (f"the answer is not a constant: it moved when the number changed "
             f"({seen_moved}). Necessary for an honest reading of "
             f"the input, not sufficient -- this does not say the values are "
             f"correct"
             if moved_by else
             f"{len(tried)} rewritten value(s) -- shifted, scaled, negated -- "
             f"all produced the identical answer ({norm_base[:120]!r}). A tool "
             f"whose answer does not move when its required {name!r} argument "
             f"changes is not reading that argument; it is reporting a "
             f"constant about itself"),
            {"sample": norm_base[:120], "moved_by": moved_by,
             "tried": tried, "unusable": unusable[:5], "value": raw}))

    def _path_axis_findings(self, spec: ToolSpec, rep: ClaimReport,
                            args: dict[str, Any], raw: str, name: str) -> None:
        """Judge a tool whose declared subject is the path, not the file.

        The file probe cannot do this, and using it here was a false accusation.
        That probe holds the path still on purpose -- one drawn directory-name
        length for all three probes, one drawn basename -- so a path measurer
        and a bare constant answer all three probes identically. Measured
        2026-11-01 in `chk_pathaxis.py`: `{"len": len(p)}` and
        `{"bytes": 123456}` land on the same verdict there, which is a fact
        about the instrument and not about the tool.

        So the axis is turned around, and the declaration is held to BOTH halves
        of what it claims:

          path_axis      the answer MUST move when only the path moves: same
                         bytes, same basename, same pinned mtime, one sibling
                         directory of equal length and one of a different
                         length. A constant fails here.
          content_blind  the answer MUST NOT move when the file at that same
                         path is rewritten -- different length, different bytes.
                         A tool that reads the file and calls itself path-only
                         fails here, whatever it reads.

        The second check is what makes the declaration cost something. Without
        it, `x-claim: path-only` would simply buy an exemption from the file
        probe, and anything reporting bytes would take it.

        Every dimension this check is not varying is held still: the payloads
        are byte-identical, the basename is one drawn name reused three times,
        and mtime is pinned to the same instant on every probe because the files
        are written one after another. Measured 2026-11-01: an unpinned
        `st_mtime` made an honest size tool look path-sensitive in the first
        version of `chk_pathaxis.py`. An unpinned dimension is measured noise.
        """
        rep.applicable = True
        rep.rail = name
        rep.rail_synthesised = True
        rep.path_only = True

        try:
            is_dir = os.path.isdir(raw)
        except OSError:
            is_dir = False

        ext = os.path.splitext(raw)[1]
        if not ext or len(ext) > 8 or not all(c.isalnum() or c == "." for c in ext):
            ext = ".txt"
        alphabet = string.ascii_lowercase + string.digits
        base_name = "".join(random.choice(alphabet) for _ in range(10)) + ext
        seed_name = "seed.txt"

        dir_len = random.randint(6, 17)
        other_len = dir_len + random.randint(2, 6)

        pinned = 1_600_000_000.0
        text_a = b"probe payload alpha 0123456789\nsecond line here\nthird\n"
        # The same-path copy must move BOTH length and content, and the reason
        # is a hole this check had for one revision. A size-reading tool that
        # mis-declared itself `path-only` answered the same for `text_a` and a
        # same-length rewrite of it, so `content_blind` passed and only
        # `path_axis` caught the lie -- and only because a size answer does not
        # move with a path. Reuse the path with a DIFFERENT LENGTH and a
        # different character mix: now a size reader moves, a digest reader
        # moves, a line counter moves, and a measurement of the path stays
        # still, which is exactly the claim being tested.
        #
        # Letters change case, digits become a letter, and the tail is doubled,
        # so every class a reader could count changes at once.
        text_b = bytes(
            0x0A if b == 0x0A else
            b - 0x20 if 0x61 <= b <= 0x7A else
            ord("x") if 0x30 <= b <= 0x39 else
            b ^ 0x01
            for b in text_a) + b"tail appended to change the length\n"
        if len(text_b) == len(text_a):   # pragma: no cover - the transform is fixed
            text_b += b"!"

        made: list[str] = []
        try:
            probe_base = self._probe_home()
            try:
                d1 = self._probe_dir(probe_base, length=dir_len)
                made.append(d1)
                d2 = self._probe_dir(probe_base, length=dir_len)
                made.append(d2)
                d3 = self._probe_dir(probe_base, length=other_len)
                made.append(d3)
            except OSError as exc:
                rep.findings.append(ClaimFinding(
                    "path_axis", False,
                    f"could not build three probe paths under {probe_base!r}: "
                    f"{exc}. Un-probed is not passed"))
                rep.unprobed = True
                return

            if is_dir:
                # A directory sample: the probes ARE the paths, and "different
                # contents at the same path" means one more entry inside it.
                for d in (d1, d2, d3):
                    with open(os.path.join(d, seed_name), "w", encoding="utf-8") as fh:
                        fh.write("seed\n")
                p1, p2, p3 = d1, d2, d3
            else:
                p1 = os.path.join(d1, base_name)
                p2 = os.path.join(d2, base_name)
                p3 = os.path.join(d3, base_name)
                for p in (p1, p2, p3):
                    with open(p, "wb") as fh:
                        fh.write(text_a)
                    os.utime(p, (pinned, pinned))

            def run_at(where: str):
                call_args = dict(args)
                call_args[name] = where
                return self._meaningful(self._call(spec, call_args))

            ok_1, out_1 = run_at(p1)
            if not ok_1:
                rep.findings.append(ClaimFinding(
                    "path_axis", False, f"probe A on {p1!r}: {out_1}"))
                return
            rep.findings.append(ClaimFinding(
                "baseline", True, f"ran on {p1!r}", {"output": out_1[:200]}))

            ok_2, out_2 = run_at(p2)
            if not ok_2:
                rep.findings.append(ClaimFinding(
                    "path_axis", False,
                    f"the same call on a same-length sibling path failed: {out_2}"))
                return
            ok_3, out_3 = run_at(p3)
            if not ok_3:
                rep.findings.append(ClaimFinding(
                    "path_axis", False,
                    f"the same call on a longer path failed: {out_3}"))
                return

            # Deliberately NOT `_strip_paths`: the path is the thing under
            # measurement here, so erasing it before comparing would be erasing
            # the evidence. The file probe does the opposite for the same
            # reason.
            n1 = normalise(out_1)
            n2 = normalise(out_2)
            n3 = normalise(out_3)
            moved_on_name = n1 != n2
            moved_on_length = n1 != n3

            rep.findings.append(ClaimFinding(
                "path_axis", bool(moved_on_name or moved_on_length),
                (f"the answer moved with the path alone -- same bytes, same "
                 f"basename, same mtime, and the path changed in "
                 f"{', '.join(k for k, v in (('name', moved_on_name), ('length', moved_on_length)) if v)}. "
                 f"This is what `x-claim: path-only` promises and the one thing "
                 f"the file probe cannot see"
                 if (moved_on_name or moved_on_length) else
                 f"three paths holding the same bytes under the same basename "
                 f"produced the identical answer ({n1[:120]!r}), although the "
                 f"path changed both in name and in length. A tool that declares "
                 f"its answer is about the path and answers the same for every "
                 f"path is reporting a constant about itself"),
                {"moved_on_name": moved_on_name, "moved_on_length": moved_on_length,
                 "A": n1[:120], "B": n2[:120], "C": n3[:120]}))

            # Same path, different contents: a measurement of the path must not
            # notice, and a measurement of the file must.
            if is_dir:
                with open(os.path.join(p1, "extra.txt"), "w", encoding="utf-8") as fh:
                    fh.write("extra\n")
            else:
                with open(p1, "wb") as fh:
                    fh.write(text_b)
                os.utime(p1, (pinned, pinned))
            ok_4, out_4 = run_at(p1)
            if not ok_4:
                rep.findings.append(ClaimFinding(
                    "content_blind", False,
                    f"the same path failed once its contents changed: {out_4}"))
                return
            n4 = normalise(out_4)
            rep.findings.append(ClaimFinding(
                "content_blind", n4 == n1,
                (f"unchanged when only the contents at that same path changed, "
                 f"which is what a measurement of the path rather than of the "
                 f"file has to do"
                 if n4 == n1 else
                 f"the answer moved when only the contents at the same path "
                 f"changed ({n1[:120]!r} -> {n4[:120]!r}). This tool reads the "
                 f"file, so `x-claim: path-only` is a mis-declaration: the "
                 f"declaration waives the file probe, and it has to be true"),
                {"same_path_same_answer": n4 == n1,
                 "before": n1[:120], "after": n4[:120]}))
        finally:
            for p in made:
                shutil.rmtree(p, ignore_errors=True)

    def _generality_probe(self, spec: ToolSpec, rep: ClaimReport,
                          args: dict[str, Any], raw: str, name: str) -> None:
        """Does the tool answer a real file by the same rule as the probes?

        The three synthetic probes are all this gate's own construction, so
        they share properties no ordinary file has. A tool can branch on those
        properties -- and four did, measurably, on 2026-11-01: each returned the
        true size for the probes and `bytes=123456` for every real file, each
        by testing one thing (`under tempfile.gettempdir()`, parent name is
        `tmp`+8 chars, size < 100, contents are printable ASCII), and all four
        PASSED.

        The check here cannot assume what the tool measures -- that is the whole
        problem, and assuming it is how the first version of this function
        failed three honest tools on the same day: it required the answer to
        contain one of {byte size, sha256, crc32, line count}, and a tool that
        counts digits, counts words, or reports the first character legitimately
        contains none of those.

        So the question asked here is not "is the number right" but "does the
        RULE still hold off the probe". The declared sample is answered once as
        it is, and then against several copies that each move ONE observable
        attribute -- size, line structure, or character composition -- while
        keeping the name, extension and directory kind. A tool passes if ANY
        copy moves its answer.

        One copy is not enough, and getting that wrong once already cost three
        honest tools on 2026-11-01: the first twin flipped every byte by XOR 1,
        which maps `'0'`+1 to `'1'` and `'a'`+1 to `'b'`, so it changed the
        content digest while leaving the digit count, the letter count, the
        word count, the longest line and the entropy all exactly as they were.
        A digit-counting tool was then failed for answering the same thing --
        correctly. Multiple copies, one axis each, is the same rule the probes
        above use, and for the same reason: no single perturbation can clear
        every honest tool, and exempting tools is worse than measuring them.
        """
        try:
            n = os.path.getsize(raw)
            with open(raw, "rb") as fh:
                # The WHOLE file, not a head: `_twin_copies` builds its
                # size-changing copies out of this, and a truncated read caps
                # every copy below the original's real length. Measured
                # 2026-11-01: with a 64 KB head, a tool that answered truly
                # only under 100 KB PASSED -- the declared sample was 360 KB
                # and every copy built from its first 64 KB stayed under the
                # threshold. Reading it all costs one file that the tool is
                # about to read anyway.
                material = fh.read()
        except OSError as exc:
            rep.findings.append(ClaimFinding(
                "generality", False,
                f"declared sample {raw!r} could not be measured "
                f"({type(exc).__name__}), so the probe results cannot be checked "
                f"against a real file. Un-probed is not passed"))
            rep.unprobed = True
            return

        g_args = dict(args)
        g_args[name] = raw
        ok_g, out_g = self._meaningful(self._call(spec, g_args))
        if not ok_g:
            rep.findings.append(ClaimFinding(
                "generality", False,
                f"failed on the declared sample {raw!r}: {out_g}. Passing on "
                f"synthetic probes while failing on the real file it was given "
                f"is the pattern of a tool tuned to the probe"))
            return
        flat_g = normalise(_strip_paths(out_g, raw))

        twins = self._twin_copies(material, n)
        twin_dir = ""
        try:
            twin_dir = self._probe_dir(self._probe_home())
            ext = os.path.splitext(raw)[1] or ".dat"
            base_name = os.path.basename(raw)
            moved: list[str] = []
            seen: list[tuple[str, str]] = []
            for axis, data in twins:
                if data == material:
                    continue
                p = os.path.join(twin_dir, base_name)
                with open(p, "wb") as fh:
                    fh.write(data)
                t_args = dict(args)
                t_args[name] = p
                ok_t, out_t = self._meaningful(self._call(spec, t_args))
                if not ok_t:
                    # Failing on a legal file of the same kind is itself a
                    # finding, but not the one this check is about; the
                    # robustness checks own that.
                    continue
                flat_t = normalise(_strip_paths(out_t, p))
                seen.append((axis, flat_t))
                if flat_t != flat_g:
                    moved.append(axis)
                os.remove(p)
        except OSError as exc:
            rep.findings.append(ClaimFinding(
                "generality", False,
                f"could not build copies of the declared sample "
                f"({type(exc).__name__}), so the probe results could not be "
                f"checked off the probe directory. Un-probed is not passed"))
            rep.unprobed = True
            return
        finally:
            if twin_dir:
                shutil.rmtree(twin_dir, ignore_errors=True)

        if moved:
            rep.findings.append(ClaimFinding(
                "generality", True,
                # Scoped the same way as `difference`, for the same measured
                # reason: `e_small` declared on the 92-byte win.ini answers
                # `123456` for the grown copy, and that move is what this check
                # credits. "It moved off the probe" is the claim; "it is right
                # off the probe" is not, and saying so would be the gate
                # testifying to something it never measured.
                f"the rule follows the file off the probe: copies of {raw!r} "
                f"keeping its name, extension and directory kind but moving "
                f"{', '.join(moved)} were answered differently from the original. "
                f"Not a statement that the new answers are correct",
                {"axes": moved, "declared": flat_g[:140],
                 "twins": {a: v[:120] for a, v in seen}}))
            return
        if not seen:
            rep.findings.append(ClaimFinding(
                "generality", False,
                f"no copy of the declared sample {raw!r} could be answered at "
                f"all, so nothing off the probe was corroborated. Un-probed is "
                f"not passed"))
            rep.unprobed = True
            return
        rep.findings.append(ClaimFinding(
            "generality", False,
            f"answered the declared sample {raw!r} and every copy of it "
            f"identically ({flat_g[:140]!r}) while moving "
            f"{', '.join(a for a, _ in seen)}. A tool whose answer does not "
            f"move when a real file's size, structure and contents do is "
            f"reporting something about itself, not about the file -- the same "
            f"failure the probes detect, here confirmed where no probe is in "
            f"sight"))

    def _twin_copies(self, material: bytes, n: int) -> list[tuple[str, bytes]]:
        """Copies of a real file, each moving exactly one observable attribute.

        `size` changes the byte count with the shape kept. `lines` keeps the
        byte count and moves the newlines. `composition` keeps the byte count
        and the newline count and rewrites the character classes, so a tool
        counting digits, letters, spaces or punctuation sees a different number
        without size or line structure moving at all. `content` keeps every
        count that is easy to compute and changes what the bytes are.
        """
        body = material if material else b"\n"
        out: list[tuple[str, bytes]] = []

        # size: same shape, different length -- in BOTH directions.
        #
        # Measured 2026-11-01: a tool that answered truly only for files under
        # 100 KB PASSED, because every copy built here was smaller than the
        # declared sample and therefore still under its threshold. A copy that
        # grows as well as one that shrinks costs nothing and closes the case
        # where the threshold sits above the real file rather than below it.
        if len(body) > 8:
            out.append(("size_down", body[: max(1, len(body) // 2)]))
        else:
            out.append(("size_down", body + b"\nrealfile"))
        out.append(("size_up", body + body))

        # lines: same byte count, different newline count.
        lines = bytearray(body)
        if b"\n" in body[:64]:
            for i, b in enumerate(lines[:64]):
                if b == 0x0A:
                    lines[i] = 0x20
                    break
        else:
            at = min(8, len(lines) - 1)
            if at > 0:
                lines[at] = 0x0A
        out.append(("lines", bytes(lines)))

        # composition: same byte count, same newline count, different classes.
        comp = bytearray(body)
        for i, b in enumerate(comp):
            if 0x61 <= b <= 0x7A:          # a-z -> A-Z
                comp[i] = b - 0x20
            elif 0x30 <= b <= 0x39:        # 0-9 -> x
                comp[i] = 0x78
            elif b == 0x20:                # space -> _
                comp[i] = 0x5F
            elif b == 0x0A:
                comp[i] = 0x0A             # newline count preserved
        out.append(("composition", bytes(comp)))

        # content: digest moves without the cheap counts necessarily moving.
        cont = bytearray(body)
        for i in range(len(cont) - 1, -1, -1):
            if cont[i] not in (0x0A,):
                cont[i] = cont[i] ^ 0xFF
                break
        out.append(("content", bytes(cont)))

        # mtime: an otherwise identical copy written now.
        out.append(("mtime", body))

        # Every one of the copies above keeps the original's first bytes, so a
        # tool that reads only the head answers them all identically. Two copies
        # that move the head close that gap: `head` rewrites the leading bytes
        # in place, `head_grow` prepends, and both keep size/name/extension.
        if len(body) >= 2:
            head = bytearray(body)
            for i in range(min(2, len(head))):
                head[i] = 0x5A if head[i] != 0x5A else 0x51   # 'Z' / 'Q'
            out.append(("head", bytes(head)))
        if len(body) >= 8:
            out.append(("head_shift", body[1:] + body[:1]))
        return out

    def _probe_dir(self, parent: str, length: int | None = None) -> str:
        """A fresh directory whose *name* carries no signature of the probe.

        `tempfile.mkdtemp` was used here until 2026-11-01 and it was a leak:
        it always names the directory `tmp` + 8 random characters, so
        `basename(dirname(path))` was always 11 characters long and always
        started with "tmp" -- even after `_probe_home` moved the probes out of
        the temp tree. Measured that day: `chk_evade2.py`'s `e_tmpname`, which
        branches on `d.startswith("tmp") and len(d) == 11`, PASSED the gate
        while answering `bytes=123456` for every real file. Moving the probes
        changed the parent and left the signature, because the signature was in
        the naming scheme, not the location.

        So the name is drawn here: a random length in 6..17 from a random
        alphabet, so neither "length 11" nor "starts with tmp" nor a fixed
        character class can identify a probe directory.

        `length` pins the name length for a caller that needs several probe
        directories to be indistinguishable *by length*. That is not a detail:
        the first version of this helper drew a fresh length per directory, and
        the difference probe immediately started clearing `l_echo_stem` -- a
        tool whose whole answer is the path's basename and `len(path)`. Three
        random lengths made the three probe paths three different lengths, so
        a tool that measured nothing produced three different answers and the
        gate read that as a measurement. Measured 2026-11-01: `chk_both.py`
        went from 0/14 wrong to 1/14 wrong (and `chk_gate.py` echoed it) the
        moment this helper returned random lengths. The length still varies
        *per run*, so it is not a signature; it must only not vary *between
        the probes being compared*.
        """
        alphabet = string.ascii_lowercase + string.digits + "._-"
        # Characters the filesystem eats at the END of a name. Windows strips
        # trailing dots and spaces from a path component *silently*: measured
        # 2026-11-03, `os.mkdir("wmv6ax90..")` returns None, `os.path.isdir()`
        # on the returned path answers True, and there is no such directory --
        # the disk kept `wmv6ax90`. `os.listdir` is the only call that tells the
        # truth, which is why the verification below asks it rather than resting
        # on mkdir having returned.
        #
        # This crashed `chk_pathonly.py` roughly one run in four hundred, with
        # `FileNotFoundError: [Errno 2] ... 'wmv6ax90..\\upsm940449.ini'` -- and
        # only on a name ending in "..", because one trailing dot is stripped
        # back to a working directory while two change how the path is parsed.
        # The old fixup guarded the *start* of the name (a leading "." reads as
        # a dotfile, a leading "_"/"-" as a scratch dir -- signatures) and said
        # nothing about the end.
        # Only "." is in this set: measured the same day, a trailing "-" and a
        # trailing "_" are both kept verbatim by the filesystem. Listing them
        # anyway would be a preference dressed as a platform constraint, and it
        # would shrink the alphabet for a reason that does not exist.
        eaten_at_end = "."
        for _ in range(8):
            want = length if length is not None else random.randint(6, 17)
            stem = "".join(random.choice(alphabet) for _ in range(want))
            # A leading "." would make it dotfile-ish on this platform and a
            # leading "_"/"-" reads as a scratch dir; both are signatures again.
            if stem[0] in "._-":
                stem = random.choice(string.ascii_lowercase) + stem[1:]
            # Same repair at the end, and it has to *replace* rather than drop
            # the character: `length` may be pinned so that sibling probes are
            # indistinguishable by length, and shortening the name would smuggle
            # the difference back in. "_" is used as the replacement because it
            # is already in the alphabet, so it adds no new character class.
            if stem[-1] in eaten_at_end:
                stem = stem[:-1] + "_"
            p = os.path.join(parent, stem)
            try:
                os.mkdir(p)
            except FileExistsError:
                continue
            # Success is not evidence. Ask the disk what it kept.
            try:
                stored = set(os.listdir(parent))
            except OSError:
                return p
            if stem not in stored:
                try:
                    os.rmdir(p)
                except OSError:
                    pass
                continue
            return p
        raise OSError("could not find an unused probe directory name")

    def _probe_home(self) -> str:
        """A writable directory that does not look like a temp dir.

        Measured 2026-11-01: `path.startswith(tempfile.gettempdir())` and
        `basename(dirname(path)) == "tmp"+8 chars` each let a tool answer
        honestly for probes and `bytes=123456` for everything else while
        PASSING. `~` is tried first because it is writable without elevation and
        is not what a temp-dir detector looks at; the temp dir is the fallback.
        """
        try:
            home = os.path.expanduser("~")
            tmp = os.path.abspath(tempfile.gettempdir()).lower()
            if home and os.path.isdir(home) and os.access(home, os.W_OK) \
                    and not os.path.abspath(home).lower().startswith(tmp):
                return home
        except OSError:
            pass
        return tempfile.gettempdir()

    def _twin_files(self, raw: str) -> tuple[str, str, str] | None:
        """Three probe files that a file-reading tool must distinguish.

        A and B: equal byte length, different bytes. C: a different length.
        Same basename, same extension throughout, and written in different
        directories so no two can share a path.

        The design has to answer one question honestly: a size-reporting tool
        legitimately answers A and B identically, and that is not dishonesty.
        So size is varied exactly once (in C) and content exactly once (B,
        against A). A tool is cleared if its answer moves across EITHER axis;
        a tool that answers a constant moves across neither and is caught.

        The probes are written to the temp directory, never next to `raw`.
        Measured 2026-10-31: when `raw` was `C:\\Windows\\win.ini`,
        `os.access(dirname, W_OK)` answered True and `mkdtemp` there raised
        PermissionError, so the whole check collapsed to un-probed for every
        tool whose sample pointed at a system file -- including the honest ones.
        A gate that cannot run its probe has to say so, but it must not be
        unable to run it for a reason as avoidable as that.
        """
        base = tempfile.gettempdir()
        try:
            if os.path.isdir(raw) and os.access(raw, os.W_OK):
                # A directory sample really is the tool's working area, so a
                # probe inside it exercises the same code path the caller would.
                base = raw
        except OSError:
            base = tempfile.gettempdir()

        seed = random.Random()

        def payload(size: int, newlines: int, classes: list[str]) -> bytes:
            """`size` bytes total, exactly `newlines` of them b"\\n".

            The remaining slots are filled by cycling `classes`, a list of
            character sets, so every class that appears is present a known
            number of times and can be varied deliberately.
            """
            newlines = max(0, min(newlines, size))
            body = size - newlines
            chars: list[str] = []
            i = 0
            while len(chars) < body:
                pool = classes[i % len(classes)]
                chars.append(seed.choice(pool))
                i += 1
            seed.shuffle(chars)
            slots = bytearray(b"".join(c.encode("ascii") for c in chars))
            if newlines:
                # Spread the newlines so a line-counting tool sees a real
                # structure rather than one trailing blank line.
                for i in range(newlines):
                    at = (i + 1) * size // (newlines + 1)
                    slots.insert(min(at, len(slots)), 0x0A)
            return bytes(slots[:size])

        # Four axes, and every probe pair varies a different one, because each
        # of them is a legitimate thing for a tool to measure and no single
        # axis can clear every honest tool.
        #
        #   A vs B -- same byte length, same newline count, and DISJOINT
        #            character classes: A is lowercase+digits, B is
        #            uppercase+space+punctuation. So a tool that counts digits,
        #            uppercase, spaces, punctuation, letters or alphanumerics
        #            sees every one of those counts change. Content tools
        #            (sha256, entropy) see it change too.
        #   A vs C -- different byte length AND different newline count, paying
        #            for the two things A vs B deliberately holds equal: a size
        #            tool and a line-counting tool move here and nowhere else.
        #
        # Measured 2026-11-01, two rounds. Round one: the payloads were three
        # newline-free lowercase tokens, so an honest line counter answered 0
        # for all three probes and was failed as "reporting a constant about
        # itself". Round two added newlines and fixed that, but kept the
        # lowercase-only alphabet -- and then honest `count digits`,
        # `count uppercase` and `count spaces` tools each answered 0 to all
        # three probes and were failed identically. Three of eight honest
        # tools, wrong. The gate was wrong every time; the fix is to vary the
        # attribute a tool might measure, not to exempt tools from being
        # measured.
        lower = "abcdefghijklmnopqrstuvwxyz"
        upper = lower.upper()
        digits = "0123456789"
        punct = "!@#$%^&*()-_=+[]{};:,.<>?/|~"
        a_data = payload(24, 3, [lower, digits])
        b_data = payload(24, 3, [upper, " ", punct])
        c_data = payload(32, 7, [lower, upper, digits, " ", punct])
        while b_data == a_data or c_data in (a_data, b_data):
            a_data = payload(24, 3, [lower, digits])
            b_data = payload(24, 3, [upper, " ", punct])
            c_data = payload(32, 7, [lower, upper, digits, " ", punct])
        payloads = [a_data, b_data, c_data]

        # The probe's *names* must not be a signature of the probe.
        #
        # Measured 2026-11-01: with the fixed prefix `claimprobe_` and the fixed
        # basename `probe.txt`, three liars that each read one attribute of the
        # path and branched on it -- `basename == "probe.txt"`,
        # `"claimprobe_" in path`, and `basename == "probe.txt" and under
        # tempfile.gettempdir()` -- all PASSED while returning `bytes=123456`
        # for every real file. The gate was handing out the string to detect it
        # by. So the directory prefix is dropped entirely (the probe directory
        # comes from `_probe_dir`, which draws its own name) and the basename is
        # randomised per probe, keeping the sample's extension when it has one
        # so a tool that filters on extension still gets a plausible file.
        ext = os.path.splitext(raw)[1]
        if not ext or len(ext) > 8 or not all(c.isalnum() or c == "." for c in ext):
            ext = ".txt"
        stem_alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

        # One drawn length for all three probe directories, not one each.
        # `_probe_dir` can draw 6..17 characters per directory, and letting it
        # do so here made the three probe *paths* differ in length -- a real
        # difference, in a place this check is not about. `l_echo_stem`, whose
        # entire answer is the basename plus `len(path)`, then produced three
        # different answers and was cleared. The basename is already a fixed 10
        # characters for the same reason. Measured 2026-11-01.
        dir_len = random.randint(6, 17)

        # Where the probes live is itself a signal a liar can branch on.
        # `_probe_home` prefers a real user directory over the temp tree for
        # that reason; see its docstring for the measurement.
        candidates: list[str] = []
        home_probe = self._probe_home()
        for c in (home_probe, base, tempfile.gettempdir()):
            if c not in candidates:
                candidates.append(c)

        for probe_base in candidates:
            made: list[str] = []
            try:
                paths = []
                for data in payloads:
                    d = self._probe_dir(probe_base, length=dir_len)
                    made.append(d)
                    stem = "".join(seed.choice(stem_alphabet) for _ in range(10))
                    p = os.path.join(d, stem + ext)
                    with open(p, "wb") as fh:
                        fh.write(data)
                    paths.append(p)
                return paths[0], paths[1], paths[2]
            except OSError:
                for d in made:
                    shutil.rmtree(d, ignore_errors=True)
                if probe_base == tempfile.gettempdir():
                    return None
        return None

    def _meaningful(self, result) -> tuple[bool, str]:
        """Did the call actually produce an answer? "INVALID" is not an answer.

        Same rule as check_execution: a tool that refuses everything must not
        read the same as a tool that works. Here it is reported as un-probed,
        which fails, because a claim gate that cannot run its perturbation has
        no evidence and "no evidence" is not "fine".
        """
        if result is None:
            return False, "sandbox unavailable"
        if getattr(result, "timed_out", False):
            return False, "timed out"
        if not getattr(result, "ok", False):
            return False, f"error: {str(getattr(result, 'error', ''))[:160]}"
        out = str(getattr(result, "output", ""))
        if out.strip().upper().startswith("INVALID"):
            return False, "returned INVALID on its own declared sample"
        if not out.strip():
            # An empty answer is not an answer. Measured 2026-10-02 while
            # hardening the rail check: a predict tool that returned "" for one
            # rail value and "" for the other compared equal, so "same call
            # twice gave the same answer" and the rails above it went green on a
            # tool that had stopped answering at all. No output is not output.
            return False, "returned nothing at all, which is not an answer"
        return True, out

    def gate(self, spec: ToolSpec, sample_args: dict[str, Any] | None = None) -> ClaimReport:
        kind = find_kind(spec)
        rail, values = find_rail(spec)
        rep = ClaimReport(spec.name, kind=kind)
        if not self.run or not spec.code:
            rep.findings.append(ClaimFinding("gated", True, "claim gate disabled or no code"))
            return rep
        if kind == "none":
            rails = find_rails(spec)
            if rails:
                # Every rail it declares is marked cosmetic. That is a tool with
                # no rail wearing the costume of one, and it must not print as
                # "nothing to check here" -- un-probed is not passed.
                rep.applicable = True
                rep.rail = rails[0][0]
                rep.passed = False
                rep.findings.append(ClaimFinding(
                    "no_binding_rail", False,
                    "every declared rail (%s) is marked `x-claim: cosmetic`, so this "
                    "gate has nothing it is allowed to perturb. A tool whose only "
                    "inputs are ones it says cannot matter has no rail; that is not "
                    "the same as having nothing to measure, and it is not passed."
                    % ", ".join(r[0] for r in rails)))
                return rep
            # No enum/boolean rail. That is not "nothing to measure": a string
            # parameter naming a file is still an input, and inputs can be
            # varied. `_no_rail_findings` builds two same-size files with
            # different contents and requires the answer to move.
            #
            # A string parameter that names TEXT rather than a file is an input
            # too, and until 2026-11-04 the gate did not think so -- measured
            # that day, an honest vowel counter, a hard-coded constant and a
            # `len(text)` measurer all came back with one identical verdict
            # (`unprobed`, single finding `no_rail`), because `_path_args`
            # matches file-like NAMES and none of them had one. The observed
            # cost was a forge that could only seal after the generator added an
            # `include_y` boolean nobody asked for. `_text_axis_findings` is the
            # text-shaped twin of the file probe and answers the same question
            # with the same evidence.
            #
            # Only when there is neither a file-like argument nor a required
            # string one does this report un-probed, and un-probed does not
            # pass -- the same rule `verifier` applies to any check that had
            # nothing to judge.
            #
            # What this replaces, and why: until 2026-10-31 this branch appended
            # `("no_rail", True, ...)`. Measured that day, a tool that never
            # opened a file reported `bytes=123456` for `C:\Windows\win.ini`
            # (truly 92 bytes) and the gate printed "not applicable" -- 5/6
            # checks green, the only failure being `trigger`, which asks whether
            # a tool gets *used*, not whether it *measures*.
            self._no_rail_findings(spec, rep, sample_args)
            rep.passed = all(f.ok for f in rep.findings)
            return rep
        args = dict(sample_args or spec.sample_call or {})
        if not args:
            rep.applicable = True
            rep.rail = rail
            rep.passed = False
            rep.findings.append(ClaimFinding(
                "un_probed", False,
                "a rail is declared but no sample call was given, so the rail could not "
                "be exercised -- un-probed is not passed"))
            return rep
        rep.applicable = True
        rep.rail = rail
        if kind == "audit":
            self._audit_rails(spec, args, rep)
        else:
            self._predict_rails(spec, args, rail, values, rep)
        rep.passed = all(f.ok for f in rep.findings)
        return rep

    # -- rails for a tool that produces its own number --------------------
    def _rail_witness(self, spec, args, rail, a, b):
        """Find an input on which this rail visibly matters.

        Returns `(input, answer_a, answer_b, inputs_tried)`, with `input` None
        when nothing tried moved the answer. `inputs_tried` is reported either
        way: "the sample could not show it and neither could these" is a
        different sentence from "the sample could not show it", and the second
        one was, for three rounds, a false accusation.

        Bounded on purpose. Each candidate costs two sandbox calls (~65 ms each
        on this host), so the variants are capped; a tool whose rail needs a
        shape none of these guesses reach stays unproven, which the finding
        then says in as many words.
        """
        token = _rail_token(rail)
        tried = 0
        for name in _varyable_strings(spec, rail):
            base = args.get(name)
            declared = "" if base is None else str(base)
            for variant in _string_variants(base, token)[:6]:
                if variant == declared:
                    continue
                candidate = dict(args)
                candidate[name] = variant
                candidate[rail] = a
                ok_a, out_a = self._meaningful(self._call(spec, candidate))
                if not ok_a:
                    continue
                flipped = dict(candidate)
                flipped[rail] = b
                ok_b, out_b = self._meaningful(self._call(spec, flipped))
                tried += 1
                if ok_b and normalise(out_a) != normalise(out_b):
                    return candidate, out_a, out_b, tried
        return None, None, None, tried

    def _predict_rails(self, spec, args, rail, values, rep):
        ok, first = self._meaningful(self._call(spec, args))
        if not ok:
            rep.findings.append(ClaimFinding("baseline", False, first))
            return
        rep.findings.append(ClaimFinding("baseline", True, "ran on its own declared sample",
                                         {"output": first[:200]}))
        # 1. deterministic: the same call twice is the same answer.
        ok2, second = self._meaningful(self._call(spec, args))
        same = ok2 and normalise(first) == normalise(second)
        rep.findings.append(ClaimFinding(
            "deterministic", bool(same),
            "same call twice gave the same answer" if same else
            "same call twice gave different answers: this is a measurement that "
            "cannot be reproduced, so nothing built on it can be trusted",
            {"first": first[:160], "second": str(second)[:160]}))
        # 2. the rail must matter: flipping it must change the answer.
        a, b = values[0], values[1]
        alt = dict(args)
        alt[rail] = b if args.get(rail) == a else a
        ok3, third = self._meaningful(self._call(spec, alt))
        moved = ok3 and normalise(third) != normalise(first)
        evidence = {"from": first[:160], "to": str(third)[:160]}
        tried = 0
        if not moved:
            # The declared sample may simply be unable to exercise this rail.
            # Measured 2026-11-04 on a forged `count_vowels(text, count_y=False)`
            # whose sample was `banana`: the tool reads `count_y` correctly, and
            # flipping it still answered 3, because `banana` holds no `y`. Read as
            # "the parameter is not read", that convicted a correct tool in three
            # consecutive rounds. Before concluding the tool ignores its own rail,
            # find an input that can show the difference; only "nothing we tried
            # moved it" is evidence about the tool.
            witness, out_a, out_b, tried = self._rail_witness(spec, args, rail, a, b)
            if witness is not None:
                moved = True
                evidence = {
                    "from": str(out_a)[:160],
                    "to": str(out_b)[:160],
                    "input": {k: str(v)[:100] for k, v in witness.items()},
                    "declared_sample_moved": False,
                }
        sens = ClaimFinding(
            "rail_sensitivity", bool(moved),
            f"changing {rail!r} changed the answer" if moved else
            (f"changing {rail!r} left the answer bit-identical on the tool's own "
             f"sample and on {tried} synthesised input(s): nothing tried made this "
             f"parameter matter. A declared rail the tool never reads is a claim "
             f"about itself that the tool contradicts -- and this was the one check "
             f"standing between a constant-returning predict tool and a verified "
             f"seal, so it is not advisory."),
            evidence)
        # Binding since 2026-10-02. It was advisory until then, on the theory
        # that some flags are cosmetic -- and the measured cost of that theory
        # is in the paragraph above: a predict tool that returns "42.00" every
        # time scored 4/4 findings green and passed the gate, because a constant
        # is deterministic, order-stable, and its unread rail was forgiven. The
        # theory is still served, by the tool's own `x-claim: cosmetic`
        # declaration rather than by the gate's guess.
        rep.findings.append(sens)
        # 3. order invariance, when the tool reads a table.
        key = _csv_arg(args)
        if key:
            self._order_rail(spec, args, key, rep, kind="predict", baseline=first)

    # -- rails for a tool that judges someone else's claim ----------------
    def _audit_rails(self, spec, args, rep):
        ok, first = self._meaningful(self._call(spec, args))
        if not ok:
            rep.findings.append(ClaimFinding("baseline", False, first))
            return
        v1 = verdict_of(first)
        if not v1:
            rep.findings.append(ClaimFinding(
                "machine_readable", False,
                "an auditor must return a verdict token (VERDICT: PASS / REJECT) so the "
                "gate can compare runs; this one returned prose only, which cannot be "
                "checked by anyone but a human reading it"))
            return
        rep.findings.append(ClaimFinding("baseline", True, f"verdict on its own sample: {v1}",
                                         {"output": first[:200]}))
        ok2, second = self._meaningful(self._call(spec, args))
        same = ok2 and normalise(first) == normalise(second)
        rep.findings.append(ClaimFinding(
            "deterministic", bool(same),
            "same call twice gave the same verdict and numbers" if same else
            "same call twice gave different answers: an auditor whose verdict moves "
            "between identical runs is reporting noise",
            {"first": first[:200], "second": str(second)[:200]}))
        key = _csv_arg(args)
        if not key:
            rep.findings.append(ClaimFinding(
                "order_invariance", False,
                "an auditor must be handed a table; no CSV-like argument was found, "
                "so the control could not be run"))
            return
        self._order_rail(spec, args, key, rep, kind="audit", baseline=first)
        self._control_rail(spec, args, key, rep, baseline=first)
        self._control_pair(spec, args, key, rep, baseline=first)
        self._spike_recovery(spec, args, key, rep, baseline=first)

    def _order_rail(self, spec, args, key, rep, *, kind, baseline):
        """Row order must not decide the verdict."""
        try:
            text = _read_csv_text(args[key])
            permuted = _permuted_csv(text, mode="rows")
            tmp = self._materialise(permuted, suffix="_rows.csv")
            alt = dict(args)
            alt[key] = tmp if not ("," in args[key] and "\n" in args[key]) else permuted
            ok, out = self._meaningful(self._call(spec, alt))
            if not ok:
                rep.findings.append(ClaimFinding("order_invariance", False, out))
                return
            if kind == "audit":
                sv = verdict_of(out)
                good = sv == verdict_of(baseline)
                detail = (f"verdict stable under row order ({sv or 'no verdict'})" if good else
                          f"verdict flipped on row order: {verdict_of(baseline)} -> {sv or 'none'}. "
                          f"A conclusion that depends on the order the rows arrived in is not a "
                          f"conclusion about the data.")
                ev = {"baseline": verdict_of(baseline), "permuted": sv}
            else:
                good = normalise(out) == normalise(baseline)
                detail = ("answer stable under row order" if good else
                          "answer changed when the same rows arrived in a different order")
                ev = {"baseline": baseline[:160], "permuted": out[:160]}
            rep.findings.append(ClaimFinding("order_invariance", bool(good), detail, ev))
        except Exception as exc:                                  # noqa: BLE001
            rep.findings.append(ClaimFinding("order_invariance", False,
                                             f"could not run the control: {type(exc).__name__}: {exc}"))

    def _control_rail(self, spec, args, key, rep, *, baseline):
        """The known-answer control: shuffle the truth and the verdict must die."""
        col = _target_column(args, spec)
        if not col:
            rep.findings.append(ClaimFinding(
                "label_permutation", False,
                "no target/label column name could be read from the call, so the "
                "known-answer control could not be run -- un-probed is not passed"))
            return
        try:
            base_v = verdict_of(baseline)
            base_n = first_number(baseline)
            text = _read_csv_text(args[key])
            permuted = _permuted_csv(text, mode="column", column=col)
            alt = dict(args)
            alt[key] = permuted if ("," in args[key] and "\n" in args[key]) else \
                self._materialise(permuted, suffix="_labels.csv")
            ok, out = self._meaningful(self._call(spec, alt))
            if not ok:
                rep.findings.append(ClaimFinding("label_permutation", False, out))
                return
            v = verdict_of(out)
            n = first_number(out)
            # The requirement is *movement*, not a direction. An earlier version
            # of this check demanded the verdict flip to REJECT, and it failed an
            # honest auditor whose baseline sample was already REJECT -- the gate
            # dictating a threshold the tool never declared, which is the very
            # mistake it exists to prevent, committed by the gate itself. What is
            # actually being tested is whether the label column reaches the
            # reading at all: shuffle the answers and the output must not be
            # bit-identical.
            verdict_moved = bool(v) and bool(base_v) and v != base_v
            number_moved = (base_n is not None and n is not None
                            and abs(n - base_n) >= 0.05)
            good = verdict_moved or number_moved
            detail = (
                f"labels shuffled -> {v or 'no verdict'}"
                f" ({base_n if base_n is not None else '?'} -> {n if n is not None else '?'}): "
                f"the label column reaches the reading" if good else
                f"labels shuffled and NOTHING moved: {base_v} -> {v or 'none'} "
                f"({base_n if base_n is not None else '?'} -> {n if n is not None else '?'}). "
                f"An auditor whose output is identical after its answers have been "
                f"shuffled is not reading them -- and it will report the same thing "
                f"on exactly the case it was forged to catch.")
            rep.findings.append(ClaimFinding(
                "label_permutation", bool(good), detail,
                {"baseline_verdict": base_v, "permuted_verdict": v,
                 "baseline_number": base_n, "permuted_number": n,
                 "verdict_moved": verdict_moved, "number_moved": number_moved}))
        except Exception as exc:                                  # noqa: BLE001
            rep.findings.append(ClaimFinding("label_permutation", False,
                                             f"could not run the control: {type(exc).__name__}: {exc}"))

    def _spike_recovery(self, spec, args, key, rep, *, baseline):
        """Add a known dose of the thing the tool claims to detect, and see
        whether the reading moves.

        This is spike recovery out of analytical chemistry: you cannot measure
        an instrument's sensitivity absolutely, so you add a known amount of
        analyte and check that the instrument recovers it. Here the analyte is
        label information. The prediction column is rewritten as

            half genuine prediction, half the answer itself

        which is a known, unmissable dose -- if the tool is a leakage auditor,
        its own reported correlation between prediction and target must rise
        substantially. A tool whose reading does not move has a dead channel,
        and a dead channel reads "everything is fine" on every dataset.

        Deliberately a large dose. A small dose would make this a sensitivity
        *threshold*, and picking that number would be the gate inventing a
        requirement the tool never declared -- it would fail tools that are
        merely insensitive while claiming (falsely) to have proved them broken.
        What is refused here is the qualitative fact: no response at all.
        """
        tcol = _target_column(args, spec)
        pcol = ""
        for k, v in args.items():
            if isinstance(v, str) and k.lower() in ("pred_col", "prediction_col", "score_col", "model_score"):
                pcol = v
        if not tcol or not pcol:
            rep.findings.append(ClaimFinding(
                "spike_recovery", False,
                "spike control skipped: no target/prediction column named in the call"))
            return
        try:
            text = _read_csv_text(args[key])
            spiked = _spike_csv(text, tcol, pcol, alpha=0.5)
            alt = dict(args)
            alt[key] = spiked if ("," in args[key] and "\n" in args[key]) else \
                self._materialise(spiked, suffix="_spike.csv")
            ok, out = self._meaningful(self._call(spec, alt))
            if not ok:
                rep.findings.append(ClaimFinding("spike_recovery", False, out))
                return
            before = _reported_r(baseline)
            after = _reported_r(out)
            if before is None or after is None:
                rep.findings.append(ClaimFinding(
                    "spike_recovery", False,
                    "the tool reports no extractable number, so a known injected dose "
                    "cannot be shown to have registered (r= or a numeric reading)"))
                return
            # Ceiling: a reading already at the top of its scale cannot register
            # any dose. That is a property of the sample, not a defect in the
            # tool, so it is recorded as un-informative rather than failed --
            # the same discipline this file applies everywhere else. A tool that
            # fakes a saturated reading to dodge the spike is caught by the two
            # controls above it, which demand that shuffling the *labels* moves
            # something.
            if abs(before) >= 0.95:
                rep.findings.append(ClaimFinding(
                    "spike_recovery", True,
                    f"not informative: the reading is already at ceiling "
                    f"({before:.3f}), so a known dose has no headroom to register",
                    {"before": before, "after": after, "dose": 0.5, "ceiling": True}))
                return
            moved = after - before
            good = moved >= 0.05
            rep.findings.append(ClaimFinding(
                "spike_recovery", bool(good),
                (f"injected half-the-answer into the prediction: reading {before:.3f} -> "
                 f"{after:.3f} (+{moved:.3f}), the channel is alive") if good else
                (f"injected half-the-answer into the prediction and the reading did not move: "
                 f"{before:.3f} -> {after:.3f} ({moved:+.3f}). A dose this large is unmissable "
                 f"to anything that measures prediction quality, so whatever this tool reports "
                 f"is not connected to the quantity it claims to measure."),
                {"before": before, "after": after, "delta": round(moved, 4),
                 "dose": 0.5, "verdict_before": verdict_of(baseline),
                 "verdict_after": verdict_of(out)}))
        except Exception as exc:                                  # noqa: BLE001
            rep.findings.append(ClaimFinding("spike_recovery", False,
                                             f"could not run the control: {type(exc).__name__}: {exc}"))

    def _control_pair(self, spec, args, key, rep, *, baseline):
        """The pair a lab would insist on: a sample whose answer is known to be
        positive, and one known to be negative, both fed to the same assay.

        Built from the tool's own sample by rewriting only the prediction column,
        so the columns, roles and size stay exactly what the tool expects:

            tautology  prediction := target. A perfect predictor. An auditor
                       that cannot see this is better than noise is not reading
                       the prediction column at all.
            noise      prediction := an independent shuffle of itself, so it
                       carries no information about the target. An auditor that
                       calls this good is not reading the target column.

        What the gate requires is only that the two verdicts DIFFER. It does not
        dictate which is which -- that is the tool's own thresholds, and a gate
        that graded thresholds would be inventing a requirement the tool never
        declared. But an assay that reports the same thing on a known-positive
        and a known-negative sample is not an assay, whatever it prints.
        """
        tcol = _target_column(args, spec)
        pcol = ""
        for k, v in args.items():
            if isinstance(v, str) and k.lower() in ("pred_col", "prediction_col", "score_col", "model_score"):
                pcol = v
        if not tcol or not pcol:
            rep.findings.append(ClaimFinding(
                "control_pair", False,
                "known-answer control skipped: the call names no target column and "
                "no prediction column, so a known-positive/known-negative pair cannot "
                "be built without guessing what the tool reads"))
            return
        try:
            text = _read_csv_text(args[key])
            pos = _known_answer_csv(text, tcol, pcol, mode="tautology")
            neg = _known_answer_csv(text, tcol, pcol, mode="noise")
            outs = {}
            for name, csv_text in (("known_positive", pos), ("known_negative", neg)):
                alt = dict(args)
                alt[key] = csv_text if ("," in args[key] and "\n" in args[key]) else \
                    self._materialise(csv_text, suffix=f"_{name}.csv")
                ok, out = self._meaningful(self._call(spec, alt))
                if not ok:
                    rep.findings.append(ClaimFinding("control_pair", False,
                                                     f"{name}: {out}"))
                    return
                outs[name] = (verdict_of(out), first_number(out), out)
            vp, vn = outs["known_positive"][0], outs["known_negative"][0]
            good = bool(vp or vn) and vp != vn
            rep.findings.append(ClaimFinding(
                "control_pair", bool(good),
                (f"known-positive -> {vp or 'none'}, known-negative -> {vn or 'none'}: "
                 f"the audit tells a perfect predictor from a useless one") if good else
                (f"known-positive -> {vp or 'none'}, known-negative -> {vn or 'none'}: "
                 f"SAME VERDICT on a prediction that is a copy of the answer and one "
                 f"that is independent noise. Whatever this tool is computing, it is "
                 f"not the quality of a prediction -- and it will say the same thing "
                 f"about the case it was forged for."),
                {"known_positive": outs["known_positive"][1],
                 "known_negative": outs["known_negative"][1],
                 "sample": outs["known_positive"][2][:200]}))
        except Exception as exc:                                  # noqa: BLE001
            rep.findings.append(ClaimFinding("control_pair", False,
                                             f"could not run the control: {type(exc).__name__}: {exc}"))

    @staticmethod
    def _materialise(text: str, suffix: str) -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        return path


# -- inputs that can exercise a rail the declared sample cannot -------------

_RAIL_PREFIXES = ("count_", "include_", "with_", "use_", "enable_", "respect_",
                  "treat_", "strip_", "keep_", "no_", "flag_")


def _rail_token(rail: str) -> str:
    """What the rail's name says it toggles: `count_y` -> `y`, `no_csv` -> `csv`.

    A prefix list is a guess, so an unrecognised prefix returns "" and the
    candidate builder falls back to bare letters rather than inventing a token.
    """
    name = (rail or "").strip().lower()
    for prefix in _RAIL_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix):]
    return ""


def _varyable_strings(spec: ToolSpec, rail: str) -> list[str]:
    """String arguments worth moving when the declared sample cannot.

    Cosmetic and path-only parameters are skipped for the same reason they are
    skipped elsewhere: the tool has said nothing may depend on them, and a
    witness built on one would be a witness about the wrong thing.
    """
    props = (spec.parameters or {}).get("properties") or {}
    out: list[str] = []
    for name, schema in props.items():
        if name == rail or not isinstance(schema, dict):
            continue
        if _is_cosmetic(schema) or _is_path_only(schema):
            continue
        if schema.get("type") == "string":
            out.append(name)
    return out


def _string_variants(base: Any, token: str) -> list[str]:
    """The declared value plus versions carrying the rail's own subject matter.

    `banana` cannot show whether `count_y` is read; `bananay` can. The token
    comes from the parameter's name because that is the only information the
    gate has about what the flag means; the single letters are a fallback for
    flags named without one.
    """
    text = "" if base is None else str(base)
    seen: set[str] = set()
    out: list[str] = []

    def add(value: str) -> None:
        if value and value not in seen:
            seen.add(value)
            out.append(value)

    if token:
        add(text + token * 3)
        add(text + token)
        add(token * 3)
        add(text + " " + token * 2)
    for letter in ("y", "a", "e", "o", "u", "1"):
        add(text + letter)
    return out

def check_claim(spec: ToolSpec, sandbox, *, sample_args: dict[str, Any] | None = None,
                run: bool = True) -> ClaimReport:
    """Convenience entry point used by the verifier."""
    return ClaimGate(sandbox, run=run).gate(spec, sample_args)


__all__ = ["ClaimFinding", "ClaimReport", "ClaimGate", "check_claim",
           "find_rail", "find_rails", "find_kind"]
