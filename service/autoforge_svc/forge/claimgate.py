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
    findings: list[ClaimFinding] = field(default_factory=list)

    @property
    def failed(self) -> list[ClaimFinding]:
        return [f for f in self.findings if not f.ok]

    def summary(self) -> str:
        if not self.applicable:
            return f"{self.tool}: claim gate not applicable (no rail declared)"
        ok = sum(1 for f in self.findings if f.ok)
        return f"{self.tool}: claim gate {ok}/{len(self.findings)} ({self.kind}, rail={self.rail})"

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "kind": self.kind, "applicable": self.applicable,
                "passed": self.passed, "rail": self.rail,
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
            rep.findings.append(ClaimFinding(
                "no_rail", True,
                "no declared rail (an enum or boolean parameter, or a target/prediction "
                "column pair): nothing to perturb, so nothing is claimed here"))
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
        alt = dict(args)
        a, b = values[0], values[1]
        alt[rail] = b if args.get(rail) == a else a
        ok3, third = self._meaningful(self._call(spec, alt))
        moved = ok3 and normalise(third) != normalise(first)
        sens = ClaimFinding(
            "rail_sensitivity", bool(moved),
            f"changing {rail!r} changed the answer" if moved else
            (f"changing {rail!r} left the answer bit-identical: the parameter is "
             f"declared, not marked cosmetic, and not read. A declared rail the tool "
             f"ignores is a claim about itself that the tool contradicts -- and this "
             f"was the one check standing between a constant-returning predict tool "
             f"and a verified seal, so it is not advisory."),
            {"from": first[:160], "to": str(third)[:160]})
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


def check_claim(spec: ToolSpec, sandbox, *, sample_args: dict[str, Any] | None = None,
                run: bool = True) -> ClaimReport:
    """Convenience entry point used by the verifier."""
    return ClaimGate(sandbox, run=run).gate(spec, sample_args)


__all__ = ["ClaimFinding", "ClaimReport", "ClaimGate", "check_claim",
           "find_rail", "find_rails", "find_kind"]
