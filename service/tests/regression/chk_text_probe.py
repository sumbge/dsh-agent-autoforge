"""Does the claim gate measure a tool whose only input is text?

The bug this file exists for, measured 2026-11-04 (`m40_text_gap.py`):

    honest counter     passed=False applicable=False unprobed=True  no_rail
    constant liar      passed=False applicable=False unprobed=True  no_rail
    length measurer    passed=False applicable=False unprobed=True  no_rail

Three tools with three different behaviours, one identical verdict, because
`_path_args` matches file-like NAMES and none of them had one.  The verdict was
identical because no measurement happened -- not because the tools were alike.

The observed cost, measured the same day in `m39_forge_three.py` run 1: round 1
declared `['text']` and died on `claim gate un-probed (none, out of scope)`, and
round 2 only sealed after the generator added an `include_y` boolean nobody
asked for.  A gate a correct tool can only satisfy by inventing a parameter is
not filtering liars; it is teaching the generator to add noise.

So this file checks the instrument that answers the same question the file probe
answers, asked of text:

    honest counter     moves on a rewrite      -> passes
    constant           never moves             -> fails
    length measurer    moves on a longer input -> passes (it does read its input)

and, at the end, the property that makes the fix safe: this can only turn a
previously un-probed tool into a judged one.  Nothing that passed before can
fail now.  That is asserted by running every shape twice in one process, with
the instrument monkeypatched off for the first pass -- an A/B of the actual code
paths, not a comparison against what I remember the old code doing.

Each sandbox call costs ~65 ms on this host and a text probe is five calls, so
the tool list here is short on purpose.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICE = os.path.dirname(os.path.dirname(HERE))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc.forge import claimgate as CG          # noqa: E402
from autoforge_svc.forge.claimgate import ClaimGate      # noqa: E402
from autoforge_svc.forge.sandbox import Sandbox          # noqa: E402
from autoforge_svc.tools.spec import ToolSpec            # noqa: E402

FAILS: list[str] = []

# How many rewritten inputs one text argument yields. Kept in step with
# `claimgate._text_variants` and asserted against it: a count only one file
# knows is a count that drifts.
VARIANTS = 5


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print("  OK  " + label)
    else:
        text = str(detail)
        print("FAIL  " + label + ("  <- " + text if text else ""))
        FAILS.append(label)


def finding(rep, name: str):
    """The finding called `name`, or a stand-in that answers Nones.

    Never index `rep.findings` directly in an assertion: the mutation specs for
    this file each delete or rename a finding, and `findings[0]` would then
    raise instead of failing, killing the suite before it prints its verdict.
    (Cost of learning this: three separate suites in this directory.)
    """
    for f in rep.findings:
        if f.name == name:
            return f

    class Missing:
        ok = None
        detail = ""
        evidence: dict = {}

    return Missing()


def truth_of(rep, name: str):
    return finding(rep, name).ok


def detail_of(rep, name: str) -> str:
    return str(finding(rep, name).detail)


def evidence_of(rep, name: str) -> dict:
    ev = finding(rep, name).evidence
    return ev if isinstance(ev, dict) else {}


HONEST = """
def count_vowels(text):
    s = text.strip()
    if not s:
        return "INVALID: empty text"
    return sum(1 for ch in s if ch in 'aeiouAEIOU')
"""

# Reads nothing but its own source. The shape the gate must catch.
CONSTANT = """
def count_vowels(text):
    return 3
"""

# Does not count vowels -- but it does read its argument, which is the only
# question this probe asks. Whether 6 is the right answer for "banana" is what
# `sample_expect` and the falsification battery are for.
LENGTH = """
def count_vowels(text):
    return len(text)
"""

# Rejects everything except the exact word it was shown. Every invented input
# comes back INVALID, so nothing can be measured -- and that must read as
# un-probed, not as a pass and not as a lie.
PICKY = """
def count_vowels(text):
    if text != 'banana':
        return "INVALID: exactly the word banana, please"
    return 3
"""

READER = """
def reader(path):
    with open(path, "r", encoding="utf-8") as fh:
        return len(fh.read())
"""


def spec_for(name: str, code: str, sample: dict, *,
             extra: dict | None = None, required: list[str] | None = None) -> ToolSpec:
    props = {k: {"type": "string"} for k in sample}
    props.update(extra or {})
    return ToolSpec(
        name=name, description="d",
        parameters={"type": "object", "properties": props,
                    "required": list(required if required is not None else sample)},
        fn=lambda **kw: None, code=code, effect_signature="pure",
        sample_call=dict(sample))


def main() -> int:
    gate = ClaimGate(Sandbox())

    # -- the three shapes that used to be one verdict ----------------------
    print("-- three tools, three behaviours, three verdicts")
    honest = spec_for("count_vowels", HONEST, {"text": "banana"})
    constant = spec_for("count_vowels", CONSTANT, {"text": "banana"})
    length = spec_for("count_vowels", LENGTH, {"text": "banana"})

    rh = gate.gate(honest, sample_args={"text": "banana"})
    rc = gate.gate(constant, sample_args={"text": "banana"})
    rl = gate.gate(length, sample_args={"text": "banana"})

    check("the honest counter passes", rh.passed is True, rh.to_dict())
    check("the constant fails", rc.passed is False, rc.to_dict())
    check("the two verdicts are now different", rh.passed != rc.passed,
          (rh.passed, rc.passed))
    check("the instrument ran on all three",
          all(r.applicable and not r.unprobed for r in (rh, rc, rl)),
          [(r.applicable, r.unprobed) for r in (rh, rc, rl)])
    check("and it says the rail was its own, not the tool's",
          all(r.rail == "text" and r.rail_synthesised for r in (rh, rc, rl)),
          [(r.rail, r.rail_synthesised) for r in (rh, rc, rl)])
    check("the summary names the text argument",
          "rail='text'" in rh.summary() and "synthesised" in rh.summary(),
          rh.summary())

    # -- what each verdict rests on ---------------------------------------
    print("-- what the honest pass rests on")
    check("it ran on its own declared sample", truth_of(rh, "baseline") is True,
          rh.to_dict())
    check("and the answer moved when the text did",
          truth_of(rh, "difference") is True, rh.to_dict())
    ev = evidence_of(rh, "difference")
    check("the finding records which rewrites moved it",
          isinstance(ev.get("moved_by"), list) and ev["moved_by"], ev)
    check("and which inputs were tried", isinstance(ev.get("tried"), list) and ev["tried"], ev)
    check("nothing was rejected as unusable", ev.get("unusable") == [], ev)
    check("the sample answer is in the evidence", str(ev.get("sample")) == "3", ev)

    print("-- what the constant's failure rests on")
    check("it also ran on its own sample", truth_of(rc, "baseline") is True,
          rc.to_dict())
    check("and nothing moved", truth_of(rc, "difference") is False, rc.to_dict())
    evc = evidence_of(rc, "difference")
    check("the evidence claims no movement at all",
          evc.get("moved_by") == [], evc)
    check("and names how many rewrites were tried",
          len(evc.get("tried") or []) == VARIANTS, evc.get("tried"))
    check("the detail says the answer is a constant about the code",
          "not reading that argument" in detail_of(rc, "difference"),
          detail_of(rc, "difference"))

    print("-- the length measurer reads its input, so it passes")
    check("its answer moved on a longer input", truth_of(rl, "difference") is True,
          rl.to_dict())
    moved = evidence_of(rl, "difference").get("moved_by") or []
    check("and no same-length rewrite moved it -- that is what makes it a "
          "length measurer",
          not any("same length" in m for m in moved), moved)
    check("the pass says what it does not prove",
          "does not say the values are correct" in detail_of(rh, "difference"),
          detail_of(rh, "difference"))

    # -- a tool that rejects every invented input -------------------------
    print("-- a parser that refuses invented inputs is un-probed, not passed")
    rp = gate.gate(spec_for("count_vowels", PICKY, {"text": "banana"}),
                   sample_args={"text": "banana"})
    check("it does not pass", rp.passed is False, rp.to_dict())
    check("and it is marked un-probed rather than accused",
          rp.unprobed is True, rp.to_dict())
    check("the detail distinguishes 'tried and failed' from 'never tried'",
          "no varied input" in detail_of(rp, "difference"),
          detail_of(rp, "difference"))
    check("and it counts the rejections",
          "probe(s) came back unusable" in detail_of(rp, "difference"),
          detail_of(rp, "difference"))

    # -- required-only, the asymmetry that keeps this from accusing --------
    print("-- an optional string is not a promise")
    optional = ToolSpec(
        name="count_vowels", description="d",
        parameters={"type": "object", "properties": {"text": {"type": "string"}},
                    "required": []},
        fn=lambda **kw: None, code=HONEST, effect_signature="pure",
        sample_call={"text": "banana"})
    check("_text_args ignores a parameter the tool may legally default",
          CG._text_args(optional, {"text": "banana"}) == [],
          CG._text_args(optional, {"text": "banana"}))
    ro = gate.gate(optional, sample_args={"text": "banana"})
    check("so such a tool keeps the verdict it had before this existed",
          ro.passed is False and ro.unprobed is True, ro.to_dict())

    print("-- and a mixed declaration keeps only the required string")
    mixed = ToolSpec(
        name="mixed", description="d",
        parameters={"type": "object", "properties": {
            "text": {"type": "string"},
            "label": {"type": "string", "x-claim": "cosmetic"},
            "where": {"type": "string", "x-claim": "path-only"},
            "flag": {"type": "boolean"},
        }, "required": ["text", "label", "where"]},
        fn=lambda **kw: None, code=HONEST, effect_signature="pure",
        sample_call={"text": "banana", "label": "x", "where": "y"})
    names = CG._text_args(mixed, {"text": "banana", "label": "x", "where": "y"})
    check("exactly the required plain string is varyable", names == ["text"], names)
    check("a cosmetic parameter is excluded", "label" not in names, names)
    check("a path-only parameter is excluded", "where" not in names, names)

    print("-- a number is not text, and since 2026-11-05 it is not un-probed either")
    numeric = ToolSpec(
        name="count_digits", description="d",
        parameters={"type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"]},
        fn=lambda **kw: None, code="def count_digits(value):\n    return 7\n",
        effect_signature="pure", sample_call={"value": 5})
    rn = gate.gate(numeric, sample_args={"value": 5})
    # This used to assert `passed is False and unprobed is True`, and the detail
    # was the `no_rail` sentence about having nothing to perturb. That was a
    # true description of a gap, not a property worth preserving: a required
    # integer fell between `_path_args` (file-like names) and `_text_args`
    # (`type == "string"`), so three different tools -- `n * 2`, `return 42`,
    # `len(str(n))` -- shared one verdict. `_number_args` closed that gap, so
    # this tool is now judged by the numeric instrument and fails for a reason
    # that is about the tool: its answer never moves.
    check("a number is not text",
          rn.rail == "value" and rn.rail_synthesised is True, rn.to_dict())
    check("and it is judged rather than set aside",
          rn.applicable is True and rn.unprobed is False, rn.to_dict())
    check("it still fails -- the answer is a constant about the code",
          rn.passed is False and detail_of(rn, "no_rail") == "", rn.to_dict())

    # -- the variants themselves ------------------------------------------
    print("-- what the rewrites are")
    variants = CG._text_variants("banana")
    check("one sample yields a fixed number of rewrites",
          len(variants) == VARIANTS, variants)
    check("every rewrite is different from the sample",
          all(v != "banana" for _, _, v in variants), variants)
    check("no rewrite is offered twice",
          len({v for _, _, v in variants}) == len(variants), variants)
    check("nothing here is random, so a failure can be reproduced",
          variants == CG._text_variants("banana"), variants)

    print("-- the two axes are distinguishable on purpose")
    shifted = [v for a, _, v in variants if a == "difference"]
    grown = [v for a, _, v in variants if a == "length"]
    check("there is at least one same-length rewrite", bool(shifted), variants)
    check("and every same-length rewrite really is the same length",
          all(len(v) == len("banana") for v in shifted),
          [len(v) for v in shifted])
    check("there is at least one longer input", bool(grown), variants)
    check("and every one of those really is longer",
          all(len(v) > len("banana") for v in grown), [len(v) for v in grown])

    print("-- the shift changes letters, not just their order")
    rotation = [v for a, _, v in variants if a == "difference"][0]
    check("no character keeps its position and its value",
          all(x != y for x, y in zip("banana", rotation)),
          (("banana"), rotation))
    check("non-letters are left alone rather than dropped",
          "!?".translate(CG._TEXT_SWAPS) == "!?", CG._TEXT_SWAPS)

    # -- the property that makes the fix safe -----------------------------
    print("-- this can only add verdicts, never remove one")
    shapes = [
        ("honest", honest, {"text": "banana"}),
        ("constant", constant, {"text": "banana"}),
        ("length", length, {"text": "banana"}),
        ("picky", spec_for("count_vowels", PICKY, {"text": "banana"}),
         {"text": "banana"}),
        ("optional-only", optional, {"text": "banana"}),
        ("numeric", numeric, {"value": 5}),
    ]
    tmp = os.path.join(os.environ.get("TEMP", "."), "m45_reader.txt")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("hello")
    shapes.append(("file-reader", spec_for("reader", READER, {"path": tmp}),
                   {"path": tmp}))

    real = CG._text_args
    before: dict[str, bool] = {}
    try:
        CG._text_args = lambda spec, args: []          # the code before 2026-11-04
        for label, s, sample in shapes:
            before[label] = gate.gate(s, sample_args=dict(sample)).passed
    finally:
        CG._text_args = real

    after: dict[str, bool] = {}
    for label, s, sample in shapes:
        after[label] = gate.gate(s, sample_args=dict(sample)).passed

    regressed = [k for k in before if before[k] and not after[k]]
    check("no tool that passed before fails now", regressed == [], regressed)
    changed = [k for k in before if before[k] != after[k]]
    check("and something did change, or this instrument is not connected",
          bool(changed), (before, after))
    # The first version of this assertion said `changed == ["constant",
    # "honest"]` and was wrong in a way worth keeping: `length` flips too, and
    # it should. That tool does read its argument -- it just reads its size
    # instead of its letters -- so the old gate's "nothing measurable" verdict
    # on it was the same false accusation that condemned the honest counter.
    # The property is about WHAT the newly-passing tools have in common, not
    # about a list I typed from memory.
    newly_passing = sorted(k for k in after if after[k] and not before[k])
    check("the tools that newly pass are the ones that read their input",
          newly_passing == ["honest", "length"], newly_passing)
    check("and the constant is not one of them", after["constant"] is False,
          after["constant"])
    check("a tool the gate could not measure before still cannot be measured "
          "into a pass", after["picky"] is False and after["numeric"] is False,
          (after["picky"], after["numeric"]))
    check("the file probe still answers its own shape unchanged",
          before["file-reader"] is True and after["file-reader"] is True,
          (before["file-reader"], after["file-reader"]))

    try:
        os.remove(tmp)
    except OSError:
        pass

    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
