"""The claim gate's numeric instrument: it must be able to fail, and only on liars.

Measured 2026-11-05, before `_number_args` existed. Three tools whose only
required parameter was `{"n": {"type": "integer"}}`:

    n * 2            passed=False applicable=False unprobed=True  [('no_rail', False)]
    return 42        passed=False applicable=False unprobed=True  [('no_rail', False)]
    len(str(n))      passed=False applicable=False unprobed=True  [('no_rail', False)]

Identical because nothing was measured. `_path_args` matches file-like NAMES,
`_text_args` matches `type == "string"`, and an integer fell between the two
instruments. Same shape as the text gap fixed the day before, one type further
out. A required boolean was never in this gap: `find_rails` already turns
`type == "boolean"` into a two-valued rail.

What the fix must do, and what it must not:

  honest `n * 2`      PASS   -- its answer moves with n
  `return 42`         FAIL   -- nothing moves, and the finding says so
  `len(str(n))`       PASS   -- it really does read n, just not its value
  bucket (`n > 100`)  FAIL   -- sample 7 and every variant stay in one bucket;
                              this gate varies the input, it does not search
  `n / 3.0` (number)  PASS   -- floats work, and never receive an int-only value

The `len(str(n))` pass is not a hole. A width reader is reading its argument;
the thing that would catch it lying about WHICH property it uses is a
different instrument, and this file says so rather than pretending otherwise.

The A/B at the end is the discriminating part: with `_number_args` neutered the
honest tool and the constant get the same verdict again, which is exactly the
state this patch was written to end.
"""
import os
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)

FAILS = []


def check(label, cond, detail=""):
    if cond:
        print("OK   %s" % label)
    else:
        FAILS.append(label)
        print("FAIL %s  <- %s" % (label, detail))


INT_PARAMS = {"type": "object",
              "properties": {"n": {"type": "integer"}},
              "required": ["n"]}
NUM_PARAMS = {"type": "object",
              "properties": {"n": {"type": "number"}},
              "required": ["n"]}
MIXED = {"type": "object",
         "properties": {"n": {"type": "integer"}, "label": {"type": "string"},
                        "loud": {"type": "boolean"}},
         "required": ["n", "label"]}
OPTIONAL_ONLY = {"type": "object",
                 "properties": {"n": {"type": "integer"}},
                 "required": []}

CASES = {
    "honest_int": (
        "def honest_int(**k):\n    return int(k['n']) * 2\n", INT_PARAMS, {"n": 7}),
    "constant_int": (
        "def constant_int(**k):\n    return 42\n", INT_PARAMS, {"n": 7}),
    "width_only": (
        "def width_only(**k):\n    return len(str(k['n']))\n", INT_PARAMS, {"n": 7}),
    "bucket_only": (
        "def bucket_only(**k):\n    n = int(k['n'])\n"
        "    return 'big' if n > 100 else 'small'\n", INT_PARAMS, {"n": 7}),
    "honest_float": (
        "def honest_float(**k):\n    return round(float(k['n']) / 3.0, 6)\n",
        NUM_PARAMS, {"n": 9.0}),
}


def spec_of(name, code, params):
    from autoforge_svc.tools.spec import ToolSpec
    return ToolSpec(name=name, description="d", parameters=params,
                    fn=lambda **k: 0, code=code, effect_signature="pure")


def verdicts(sandbox):
    from autoforge_svc.forge.claimgate import check_claim
    out = {}
    for name, (code, params, sample) in CASES.items():
        rep = check_claim(spec_of(name, code, params), sandbox, sample_args=sample)
        out[name] = rep
    return out


def detail_of(rep, name):
    for f in getattr(rep, "findings", []) or []:
        if f.name == name:
            return f.detail or ""
    return ""


def ok_of(rep, name):
    for f in getattr(rep, "findings", []) or []:
        if f.name == name:
            return bool(f.ok)
    return None


def main():
    os.environ["AUTOFORGE_HOME"] = tempfile.mkdtemp(prefix="chk_number_probe_")
    import autoforge_svc.service as S
    from autoforge_svc.service import Service
    from autoforge_svc.forge import claimgate as CG

    svc = Service(db_path=os.path.join(os.environ["AUTOFORGE_HOME"], "p.db"))
    reps = verdicts(svc.sandbox)

    # -- the three shapes that started this ------------------------------
    check("the honest integer tool passes",
          reps["honest_int"].passed is True, detail_of(reps["honest_int"], "difference"))
    check("the constant liar fails",
          reps["constant_int"].passed is False, detail_of(reps["constant_int"], "difference"))
    check("and its failure names what did not move",
          "all produced the identical answer" in detail_of(reps["constant_int"], "difference"),
          detail_of(reps["constant_int"], "difference"))
    check("the width reader passes -- it does read its argument",
          reps["width_only"].passed is True, detail_of(reps["width_only"], "difference"))

    # -- the gate reaches them at all now --------------------------------
    for name in ("honest_int", "constant_int", "width_only", "bucket_only", "honest_float"):
        check("%s is judged, not set aside" % name,
              reps[name].applicable is True and reps[name].unprobed is False,
              "applicable=%r unprobed=%r" % (reps[name].applicable, reps[name].unprobed))

    # -- a coarse reader is not an honest reader of the VALUE ------------
    check("a bucket-only reader fails: no variant crosses its threshold",
          reps["bucket_only"].passed is False,
          detail_of(reps["bucket_only"], "difference"))
    check("and the finding says the answer is a constant about itself",
          "constant about itself" in detail_of(reps["bucket_only"], "difference"),
          detail_of(reps["bucket_only"], "difference"))

    # -- a float parameter works, and never gets an integer-only value ---
    check("a float tool passes", reps["honest_float"].passed is True,
          detail_of(reps["honest_float"], "difference"))
    check("the rail it names is the numeric parameter",
          reps["honest_int"].rail == "n", repr(reps["honest_int"].rail))
    check("and the probe was synthesised, not declared",
          reps["honest_int"].rail_synthesised is True)

    # -- the variant table ----------------------------------------------
    int_variants = CG._number_variants(7, integer=True)
    check("integer variants never leave the declared type",
          all(isinstance(v, int) and not isinstance(v, bool) for _a, _l, v in int_variants),
          int_variants)
    check("integer variants never repeat the sample",
          all(v != 7 for _a, _l, v in int_variants), int_variants)
    # The repeat check only fires for a sample whose candidates can collide with
    # it. For 7 they cannot (the candidates are 8, 6, 9, 14, 21, -7), so the
    # line above is a wish, not a guard. Zero is the input that makes it visible:
    # 0*2, 0*3 and -0 all equal the sample, and a variant equal to the sample
    # measures nothing while looking like a measurement.
    zero_variants = CG._number_variants(0, integer=True)
    check("a zero sample yields no variant equal to itself",
          all(v != 0 for _a, _l, v in zero_variants), zero_variants)
    check("and it is not silently emptied either",
          len(zero_variants) >= 3, zero_variants)
    check("they are deterministic -- no randomness to hide behind",
          int_variants == CG._number_variants(7, integer=True))
    check("both axes are present",
          {a for a, _l, _v in int_variants} == {"difference", "magnitude"},
          sorted({a for a, _l, _v in int_variants}))
    float_variants = CG._number_variants(9.0, integer=False)
    check("a float parameter additionally gets a halving",
          any(label == "halved" for _a, label, _v in float_variants), float_variants)
    # A `number` parameter is declared to accept a float, but `raw // 2` is a
    # float here by accident (Python keeps the operand type), and `int(raw / 2)`
    # is what a careless writer produces. Assert the value, not the label.
    halved = [v for _a, label, v in float_variants if label == "halved"]
    check("and the halved value is a float, not a truncated int",
          halved == [4.5], halved)
    check("and no float variant is a bool",
          all(not isinstance(v, bool) for _a, _l, v in float_variants), float_variants)

    # -- which parameters it is allowed to touch -------------------------
    check("only REQUIRED numeric parameters are candidates",
          [n for n, _v in CG._number_args(spec_of("x", "", OPTIONAL_ONLY), {"n": 3})] == [],
          CG._number_args(spec_of("x", "", OPTIONAL_ONLY), {"n": 3}))
    check("a required string is not a numeric candidate",
          [n for n, _v in CG._number_args(spec_of("x", "", MIXED),
                                          {"n": 3, "label": "abc"})] == ["n"],
          CG._number_args(spec_of("x", "", MIXED), {"n": 3, "label": "abc"}))
    check("a declared-cosmetic number is left alone",
          CG._number_args(
              spec_of("x", "", {"type": "object", "required": ["n"], "properties": {
                  "n": {"type": "integer", "x-claim": "cosmetic"}}}), {"n": 3}) == [])
    check("a bool sample is not treated as a number",
          CG._number_args(spec_of("x", "", INT_PARAMS), {"n": True}) == [])

    # -- a text argument still wins, so nothing that passed before changes
    both = spec_of("both", "def both(**k):\n    return len(str(k['n'])) + len(k['t'])\n", MIXED)
    check("a tool taking both keeps the text instrument",
          CG._text_args(both, {"n": 3, "label": "abc"}) == ["label"],
          CG._text_args(both, {"n": 3, "label": "abc"}))

    # -- A/B: neuter the instrument and the old blindness returns --------
    real = CG._number_args
    try:
        CG._number_args = lambda spec, args: []
        before = verdicts(svc.sandbox)
    finally:
        CG._number_args = real
    sample = "+".join(sorted(spec_of("x", "", INT_PARAMS).parameters["properties"]))
    check("without the instrument the honest tool is set aside again",
          before["honest_int"].applicable is False and before["honest_int"].unprobed is True,
          "applicable=%r unprobed=%r" % (before["honest_int"].applicable,
                                         before["honest_int"].unprobed))
    check("and it is indistinguishable from the constant liar",
          before["honest_int"].passed == before["constant_int"].passed,
          "honest=%r constant=%r" % (before["honest_int"].passed,
                                     before["constant_int"].passed))
    check("the instrument is what separates them",
          reps["honest_int"].passed is True and reps["constant_int"].passed is False)
    print("note: the A/B names the integer parameter through %r, nothing more" % sample)

    # -- what this instrument cannot see ---------------------------------
    print("note: a liar that reads the value and answers it wrong "
          "(`n * 2 + 1`) moves on both axes and passes. Correctness is "
          "`sample_expect` and the falsification battery, not this file.")
    print("note: a coarse reader (`n > 100`) fails here because the sample and "
          "every variant stay in one bucket. That is this gate's reach, not a "
          "claim that the tool is a liar.")

    print("TOTAL FAILURES: %d %s" % (len(FAILS), FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
