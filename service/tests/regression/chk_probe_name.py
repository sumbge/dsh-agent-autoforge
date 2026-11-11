"""A probe directory must be a name the filesystem actually kept.

Found 2026-11-03 the way the others were found -- not by reading the code but
by a check failing intermittently and being chased instead of re-run:

    chk_pathonly.py   run 1: rc=1, 19 lines, no verdict
                      runs 2-4: rc=0, "TOTAL WRONG 0 of 11"

The flake was:

    FileNotFoundError: [Errno 2] No such file or directory:
      'C:\\Users\\someone\\wmv6ax90..\\upsm940449.ini'
                                     ^^ two trailing dots

`_probe_dir` draws its name from `string.ascii_lowercase + string.digits +
"._-"` and fixes up a *leading* `.`/`_`/`-` -- because those are signatures --
but says nothing about the *end* of the name. Windows strips trailing dots and
spaces from a path component, so `os.mkdir("wmv6ax90..")` **succeeds** and
answers a path that names the directory `wmv6ax90`; `os.path.isdir()` on the
returned path also answers True, because every later lookup strips again. Only
the *child* fails, and only when the component ends in `..`, where the trailing
dots change how the path is parsed at all.

Measured on this machine (Windows 11, Python 3.13.14):

    asked          mkdir   isdir(asked)   open child
    plainok1       ok      True           ok
    trailone.      ok      True           ok          <- created as "trailone"
    trailtwo..     ok      True           FileNotFoundError
    trailhyph-     ok      True           ok
    trailunder_    ok      True           ok
    dot.mid        ok      True           ok

Two things follow, and the second is the reason this file exists.

1. `_probe_dir` can hand back a path that cannot hold a file. That is the
   intermittent crash.

2. **Success is not evidence.** `os.mkdir` returned None, `os.path.isdir`
   returned True, and there was still nothing there under the name we asked
   for. A probe that reports on the name it *requested* rather than the name
   the disk *kept* is measuring its own intention.

So the contract asserted here is about the disk, not the call: the basename
`_probe_dir` returns must appear **verbatim** in `os.listdir(parent)`, and a
child written through it must exist.

The fix must not reintroduce a signature, which was the whole reason this
helper stopped using `mkdtemp`. So the test also pins the randomness: lengths
still vary across runs, and the odd characters are still *used* -- they are
only required never to land where the filesystem would eat them. Deleting `.`
from the alphabet would pass every check in this file except that one, and it
would quietly narrow the name space this helper exists to widen.

Run with the real interpreter, not through the agent sandbox (30 s CPU cap):

    $env:PYTHONPATH = '<repo>/service'
    python chk_probe_name.py
"""
from __future__ import annotations

import os
import random
import string
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autoforge_svc.forge.claimgate import ClaimGate  # noqa: E402
from autoforge_svc.forge.sandbox import Sandbox  # noqa: E402

FAILS: list[str] = []

DRAWS = 300


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print(f"  OK   {label}")
    else:
        print(f"  FAIL {label}  <- {detail}")
        FAILS.append(label)


def gate() -> ClaimGate:
    return ClaimGate(Sandbox())


def old_rule_letters() -> str:
    """The pre-fix alphabet, verbatim, for the fault injection below.

    Only the alphabet is reused. The pre-fix *helper* is not called, because it
    would have to be duplicated to be called, and a duplicate is exactly the
    thing that stops being true when the product changes -- the same failure
    this file is about, one level up.
    """
    return string.ascii_lowercase + string.digits + "._-"


def names_that_survived() -> None:
    print("-- the name the disk kept, not the name we asked for")
    tmp = tempfile.mkdtemp(prefix="m17_name_")
    g = gate()
    kept, unkept, unwritable = [], [], []
    lengths, odd = set(), 0
    try:
        for _ in range(DRAWS):
            p = g._probe_dir(tmp)
            base = os.path.basename(p)
            lengths.add(len(base))
            if any(c in base for c in "._-"):
                odd += 1
            stored = os.listdir(tmp)
            if base not in stored:
                unkept.append({"returned": base, "stored": sorted(stored)[:4]})
                continue
            kept.append(base)
            try:
                with open(os.path.join(p, "seed.txt"), "w", encoding="utf-8") as fh:
                    fh.write("x")
            except OSError as exc:
                unwritable.append({"name": base, "error": f"{type(exc).__name__}: {exc.strerror}"})
            if base[0] in "._-":
                unkept.append({"returned": base, "stored": "leading punctuation"})

        check("the probe names were drawn at all", len(kept) + len(unkept) == DRAWS,
              (len(kept), len(unkept)))
        check("every name _probe_dir returns is the name the disk kept",
              not unkept, unkept[:3])
        check("and a file can be written through it", not unwritable, unwritable[:3])
        check("no name starts with a character that reads as a scratch dir or dotfile",
              all(b[0] not in "._-" for b in kept), [b for b in kept if b[0] in "._-"][:3])

        print("-- the repair must not become a narrower alphabet")
        check("lengths still vary between draws", len(lengths) >= 5, sorted(lengths))
        check("the odd characters are still in use, only kept off the end",
              odd > DRAWS // 10, f"{odd} of {DRAWS} names contain . _ or -")
        # Only "." is dangerous at the end: measurement above shows a trailing
        # "." is stripped by the filesystem while a trailing "-" or "_" is kept
        # verbatim. Asserting all three off the end would be my preference
        # dressed as a contract, and it would narrow the alphabet for no reason
        # -- which is the failure mode this helper exists to avoid.
        check("and no name ends in '.', the one ending the filesystem eats",
              all(b[-1] != "." for b in kept), [b for b in kept if b[-1] == "."][:5])
    finally:
        for base in os.listdir(tmp):
            child = os.path.join(tmp, base)
            try:
                for f in os.listdir(child):
                    os.remove(os.path.join(child, f))
                os.rmdir(child)
            except OSError:
                pass
        os.rmdir(tmp)


def fault_injection() -> None:
    print("-- the old rule, run until it draws a name the disk renames")
    # Not "40 draws and hope". The failing name has to END in "." and the odds
    # are 1 in 19 per trailing character, so 40 draws mostly proves nothing --
    # the first version of this section reported "0 of 40 names broke" and I
    # nearly read that as "the bug is not real". Draw until the shape appears,
    # with a bound that raises instead of passing quietly.
    tmp = tempfile.mkdtemp(prefix="m17_old_")
    alphabet = old_rule_letters()
    broken: list[str] = []
    draws = 0
    try:
        while not broken and draws < 20000:
            draws += 1
            stem = "".join(random.choice(alphabet) for _ in range(random.randint(6, 17)))
            if stem[0] in "._-" or stem[-1] != ".":
                continue
            p = os.path.join(tmp, stem)
            try:
                os.mkdir(p)
            except FileExistsError:
                continue
            if stem not in os.listdir(tmp):
                broken.append(stem)
        check("a name ending in '.' is created under a different name than requested",
              bool(broken), f"{draws} draws without reproducing -- the check cannot fail here")
        print(f"       reproduced after {draws} draws: asked {broken[0]!r}"
              if broken else f"       not reproduced in {draws} draws")
    finally:
        for base in os.listdir(tmp):
            try:
                os.rmdir(os.path.join(tmp, base))
            except OSError:
                pass
        os.rmdir(tmp)


def trailing_dot_ends_in_dotdot() -> None:
    print("-- the specific shape that crashed the harness: a name ending in '..'")
    tmp = tempfile.mkdtemp(prefix="m17_dd_")
    try:
        p = os.path.join(tmp, "trailtwo..")
        os.mkdir(p)
        crashed = False
        try:
            with open(os.path.join(p, "probe.ini"), "wb") as fh:
                fh.write(b"x")
        except OSError:
            crashed = True
        check("a directory name ending in '..' still mkdirs (so only the write reveals it)",
              os.path.isdir(os.path.join(tmp, "trailtwo")), sorted(os.listdir(tmp)))
        check("and writing a child through it fails -- this is the FileNotFoundError seen in the wild",
              crashed, "the write succeeded here; the platform may have changed")
    finally:
        for base in os.listdir(tmp):
            child = os.path.join(tmp, base)
            try:
                for f in os.listdir(child):
                    os.remove(os.path.join(child, f))
                os.rmdir(child)
            except OSError:
                pass
        os.rmdir(tmp)


def main() -> int:
    names_that_survived()
    fault_injection()
    trailing_dot_ends_in_dotdot()
    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
