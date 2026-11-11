"""A docstring may not name a mechanism the code contradicts.

The two docstrings in `forge/sandbox.py` claimed `effective_env()` drops TEMP and
TMP, and used that as the reason the child could resolve a different temp dir.
Both names are in `env_allow`. The claim was false for as long as it stood, and
nothing caught it -- because nothing was checking prose against behaviour.

This suite checks the three cheap, decisive facts:

  1. `TEMP` and `TMP` ARE allow-listed. If that ever stops being true the
     docstrings may say so again; until then the sentence must not be there.
  2. The false sentence is absent from the whole file (not just the two blocks).
  3. The replacement states the route that does separate the two filesystems
     (`isolate=True`) and is honest that it could not be exercised here.

It also proves the checks can fail: a copy of the module with the sentence put
back must fail #2. Without that, a passing run is indistinguishable from a
search that never ran.
"""

import os
import re
import shutil
import sys
import tempfile

# This file is service/tests/regression/, so the package root is three up.
SVC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SBX = os.path.join(SVC, "autoforge_svc", "forge", "sandbox.py")

FAILS = []
OKS = 0


def check(label, cond, detail=""):
    global OKS
    if cond:
        OKS += 1
        print("  OK   %s" % label)
    else:
        FAILS.append(label)
        print("  FAIL %s%s" % (label, ("  <- " + repr(detail)) if detail != "" else ""))


def main(**kwargs):
    sys.path.insert(0, SVC)
    from autoforge_svc.forge.sandbox import Sandbox

    text = open(SBX, encoding="utf-8").read()
    allow = tuple(Sandbox.env_allow)

    print("-- the allow-list is the fact the docstrings were arguing about")
    check("TEMP is allow-listed", "TEMP" in allow, allow)
    check("TMP is allow-listed", "TMP" in allow, allow)

    print("-- the false cause is gone from the whole file")
    check("no block claims effective_env drops TEMP",
          "effective_env()` drops TMP" not in text)
    check("the old measured example is gone", "C:\\\\Temp outside" not in text)
    check("the old probe sentence is gone", "here `tempfile` gives" not in text)

    print("-- the replacement names the route that does separate them")
    check("the isolation route is named", "isolate=True" in text)
    check("the four constructions are recorded",
          "Four constructions were measured" in text)
    check("it says the route could not be exercised here",
          "could not be exercised here" in text)
    check("it says wsl lists no distribution", "lists no distribution" in text)

    print("-- and the checks can fail on a module that puts the claim back")
    tmp = tempfile.mkdtemp(prefix="chk_temp_claim_")
    try:
        broken = os.path.join(tmp, "sandbox.py")
        with open(broken, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text.replace(
                "        wrong and is corrected: `TEMP` and `TMP` ARE in `env_allow`, so",
                "        wrong and is corrected, but the old claim returns here:"
                " `effective_env()` drops TMP", 1))
        broken_text = open(broken, encoding="utf-8").read()
        check("a module that puts the claim back is detected",
              "effective_env()` drops TMP" in broken_text,
              "the mutation did not land")
        check("...and the healthy file does not carry it",
              "effective_env()` drops TMP" not in text)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    print("TOTAL FAILURES:", len(FAILS), FAILS if FAILS else "")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
