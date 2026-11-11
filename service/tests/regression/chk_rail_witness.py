"""Does `rail_sensitivity` still convict a tool that ignores its rail?

The bug this file exists for, measured 2026-11-04:

    def count_vowels(text, count_y=False):
        vowels = set('aeiouAEIOU')
        if count_y:
            vowels |= set('yY')
        return sum(1 for ch in text.strip() if ch in vowels)

`sample_call` was `{"text": "banana"}`. Flipping `count_y` left the answer at 3
because `banana` holds no `y`. The gate read that as "the parameter is declared,
not marked cosmetic, and not read" and failed the tool -- three rounds running,
so the forge never sealed a correct tool. The tool was right; the check was
asking the tool's own sample to prove a property of the tool.

`happy` vs `banana` settles which of the two was wrong:

    banana  count_y=False -> 3   count_y=True -> 3    (the sample cannot show it)
    happy   count_y=False -> 1   count_y=True -> 2    (the rail is read)

The fix gives the check a bounded witness search: before concluding the tool
ignores its rail, try inputs that could exercise it. Nothing else changed, so
the set of tools that pass can only grow -- a check that could lose power here
would be a worse bug than the one being fixed, and the mutation specs for this
file each try to make it lose power.

Every check below runs the real sandbox. Each call costs ~65 ms on this host, so
the file stays small on purpose.
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

# How many synthesised inputs per string argument the gate may run. Kept in
# step with `claimgate._rail_witness`, and asserted against it below: a cap
# that only one of the two files knows is a cap that drifts.
CAP = 6


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print("  OK  " + label)
    else:
        text = str(detail)
        print("FAIL  " + label + ("  <- " + text if text else ""))
        FAILS.append(label)


HONEST = """
def count_vowels(text, count_y=False):
    if not isinstance(text, str):
        return "INVALID: text must be a string"
    s = text.strip()
    if not s:
        return "INVALID: empty text"
    vowels = set('aeiouAEIOU')
    if count_y:
        vowels |= set('yY')
    return sum(1 for ch in s if ch in vowels)
"""

# Reads the flag, decides what to do with it -- and then does the same thing
# either way. The honest shape minus one line, which is exactly the shape the
# check is supposed to catch.
IGNORES = """
def count_vowels(text, count_y=False):
    s = text.strip()
    vowels = set('aeiouAEIOU')
    if count_y:
        pass
    return sum(1 for ch in s if ch in vowels)
"""

CONSTANT = """
def count_vowels(text, count_y=False):
    return 42
"""

PARAMS = {
    "type": "object",
    "properties": {
        "text": {"type": "string",
                 "description": "The word or phrase whose vowels should be counted."},
        "count_y": {"type": "boolean",
                    "description": "If true, treat 'y'/'Y' as a vowel as well."},
    },
    "required": ["text"],
}


def spec_for(code: str, sample: dict | None = None, name: str = "count_vowels"):
    return ToolSpec(
        name=name,
        description="Count the vowels in a word.",
        parameters=PARAMS,
        fn=lambda **kw: None,
        code=code,
        effect_signature="pure",
        sample_call=sample if sample is not None else {"text": "banana"},
    )


def finding(report, name: str):
    for f in report.findings:
        if f.name == name:
            return f
    return None


def detail_of(report, name: str) -> str:
    """Never index a finding that may be absent -- an IndexError here would end
    the file before its verdict line, and a crash is not a failure report."""
    f = finding(report, name)
    return "" if f is None else f.detail


def truth_of(report, name: str) -> bool | None:
    f = finding(report, name)
    return None if f is None else bool(f.ok)


def main() -> int:
    gate = ClaimGate(Sandbox())

    # -- the tool that was convicted wrongly, and still measures -----------
    print("-- a correct tool whose declared sample cannot exercise its rail")
    honest = spec_for(HONEST)
    rep = gate.gate(honest, sample_args={"text": "banana"})
    check("the honest tool now passes the gate", rep.passed is True,
          [f.name for f in rep.failed])
    check("rail_sensitivity is the finding that flipped",
          truth_of(rep, "rail_sensitivity") is True, rep.to_dict())
    ev = (finding(rep, "rail_sensitivity").evidence or {})
    check("and the evidence says the sample was not what moved it",
          ev.get("declared_sample_moved") is False, ev)
    check("the evidence names the input that did move it",
          isinstance(ev.get("input"), dict) and ev["input"].get("text"),
          ev)
    check("the witness carries the rail's own subject matter",
          "y" in str((ev.get("input") or {}).get("text", "")),
          (ev.get("input") or {}).get("text"))
    check("the two answers it reports are actually different",
          str(ev.get("from")) != str(ev.get("to")),
          (ev.get("from"), ev.get("to")))

    # -- the two shapes the check exists to convict ------------------------
    print("-- a tool that declares a rail it never reads")
    ignores = spec_for(IGNORES)
    rep2 = gate.gate(ignores, sample_args={"text": "banana"})
    check("it still fails", rep2.passed is False, rep2.to_dict())
    check("rail_sensitivity is why", truth_of(rep2, "rail_sensitivity") is False,
          rep2.to_dict())
    check("and the detail says the search was exhausted, not that none ran",
          "synthesised input" in detail_of(rep2, "rail_sensitivity"),
          detail_of(rep2, "rail_sensitivity"))

    print("-- a tool that returns a constant")
    const = spec_for(CONSTANT)
    rep3 = gate.gate(const, sample_args={"text": "banana"})
    check("it still fails", rep3.passed is False, rep3.to_dict())
    check("rail_sensitivity is why", truth_of(rep3, "rail_sensitivity") is False,
          rep3.to_dict())
    ev3 = (finding(rep3, "rail_sensitivity").evidence or {})
    check("a constant never produces a witness", "input" not in ev3, ev3)
    check("and its two answers are reported as identical",
          str(ev3.get("from")) == str(ev3.get("to")),
          (ev3.get("from"), ev3.get("to")))

    # -- bounds: the search must not become the slow part ------------------
    # Counted on a tool that NEVER finds a witness, because a search that stops
    # at the first hit cannot show you its cap. (Mutation W10 removed the cap
    # and this file still passed: for the honest tool the witness is the first
    # candidate, so `tried` was 1 with or without a limit.)
    print("-- the witness search is bounded")
    tried_flip = (finding(rep, "rail_sensitivity").evidence or {}).get("input")
    check("a witness was found without walking every variant",
          tried_flip is not None, tried_flip)
    capped = gate._rail_witness(honest, {"text": "banana"}, "count_y", True, False)
    check("_rail_witness returns four values", len(capped) == 4, capped)
    check("and tries a finite number of inputs", 0 < capped[3] <= 12, capped[3])
    per_arg = len(CG._string_variants("banana", "y"))
    check("one string argument yields a finite variant list", 0 < per_arg <= 20, per_arg)
    check("more variants exist than the cap allows",
          per_arg > CAP, (per_arg, CAP))
    exhausted = gate._rail_witness(spec_for(IGNORES), {"text": "banana"},
                                   "count_y", True, False)
    check("a search that finds nothing still stops", exhausted[0] is None, exhausted)
    check("and it stopped at the cap, not after every variant",
          exhausted[3] == CAP, (exhausted[3], CAP))

    # -- the token the rail's name supplies --------------------------------
    print("-- what the rail's name contributes")
    check("count_y names the letter y", CG._rail_token("count_y") == "y",
          CG._rail_token("count_y"))
    check("include_header names header",
          CG._rail_token("include_header") == "header",
          CG._rail_token("include_header"))
    check("an unknown prefix yields no token, not a guess",
          CG._rail_token("mode") == "", CG._rail_token("mode"))
    check("the token appears among the variants",
          any("y" in v for v in CG._string_variants("banana", "y")),
          CG._string_variants("banana", "y"))
    check("bare letters are offered when there is no token",
          any("a" in v for v in CG._string_variants("q", "")),
          CG._string_variants("q", ""))

    # -- string arguments that must not be used as witnesses ---------------
    print("-- which parameters may be varied for a witness")
    props = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "label": {"type": "string", "x-claim": "cosmetic"},
            "path": {"type": "string", "x-claim": "path-only"},
            "count_y": {"type": "boolean"},
        },
        "required": ["text"],
    }
    mixed = ToolSpec(name="mixed", description="d", parameters=props,
                     fn=lambda **kw: None, code=HONEST, effect_signature="pure",
                     sample_call={"text": "banana"})
    names = CG._varyable_strings(mixed, "count_y")
    check("a plain string parameter is varyable", names == ["text"], names)
    check("a cosmetic parameter is not", "label" not in names, names)
    check("a path-only parameter is not", "path" not in names, names)

    # -- a tool with no string argument at all -----------------------------
    print("-- nothing to vary is reported as nothing tried")
    numeric = ToolSpec(
        name="count_digits",
        description="Count the digits in a number.",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "integer"},
                "count_y": {"type": "boolean"},
            },
            "required": ["value"],
        },
        fn=lambda **kw: None,
        code="def count_digits(value, count_y=False):\n    return 7\n",
        effect_signature="pure",
        sample_call={"value": 5},
    )
    rep4 = gate.gate(numeric, sample_args={"value": 5})
    check("a tool with no varyable input still fails on an unread rail",
          truth_of(rep4, "rail_sensitivity") is False, rep4.to_dict())
    check("and the detail says zero inputs were tried",
          "on 0 synthesised input(s)" in detail_of(rep4, "rail_sensitivity"),
          detail_of(rep4, "rail_sensitivity"))

    # -- the token is load-bearing, not decoration -------------------------
    # Every other tool here has a single-letter token, and the bare-letter
    # fallback list happens to contain all six single letters used. So those
    # tools find their witness through the fallback and a broken token path
    # still looks fine. (Mutation W5 disabled the token candidates and this
    # file passed.) This tool can only be witnessed through its token.
    print("-- the rail's name is what makes some witnesses reachable")
    token_only = """
def count_zz(text, count_zz=False):
    if not count_zz:
        return 0
    return text.count('zz')
"""
    shape = {
        "type": "object",
        "properties": {
            "text": {"type": "string",
                     "description": "The text to look in."},
            "count_zz": {"type": "boolean",
                         "description": "If true, count 'zz' occurrences."},
        },
        "required": ["text"],
    }
    zspec = ToolSpec(name="count_zz", description="Count 'zz' occurrences.",
                     parameters=shape, fn=lambda **kw: None, code=token_only,
                     effect_signature="pure", sample_call={"text": "banana"})
    check("the rail name yields a two-letter token",
          CG._rail_token("count_zz") == "zz", CG._rail_token("count_zz"))
    zrep = gate.gate(zspec, sample_args={"text": "banana"})
    check("a witness reachable only through the token is still found",
          truth_of(zrep, "rail_sensitivity") is True, zrep.to_dict())
    zinput = ((finding(zrep, "rail_sensitivity").evidence or {}).get("input") or {})
    check("and that witness carries the token",
          "zz" in str(zinput.get("text", "")), zinput)
    check("the bare-letter fallback alone would not have found it",
          all("zz" not in v for v in CG._string_variants("banana", "")),
          CG._string_variants("banana", ""))

    # -- the property that makes this fix safe -----------------------------
    print("-- the fix can only add passes, never remove one")
    for label, code, sample in (
        ("honest/banana", HONEST, {"text": "banana"}),
        ("honest/happy", HONEST, {"text": "happy"}),
        ("ignores/banana", IGNORES, {"text": "banana"}),
        ("constant/banana", CONSTANT, {"text": "banana"}),
    ):
        s = spec_for(code, sample)
        args = dict(sample)
        rail, values = CG.find_rail(s)
        ok_a, out_a = gate._meaningful(gate._call(s, dict(args, **{rail: values[0]})))
        node = {"text": sample["text"], rail: values[1]}
        ok_b, out_b = gate._meaningful(gate._call(s, node))
        sample_moved = ok_a and ok_b and CG.normalise(out_a) != CG.normalise(out_b)
        r = gate.gate(s, sample_args=sample)
        final = truth_of(r, "rail_sensitivity")
        if sample_moved:
            check(f"{label}: moved on its own sample, so the search was not needed",
                  final is True, r.to_dict())
        else:
            check(f"{label}: the verdict is the search's answer, not the sample's",
                  isinstance(final, bool), r.to_dict())

    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
