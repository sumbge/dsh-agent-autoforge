"""The skill surface: four methods, one name rule, seven refusals that are refusals.

What this suite exists for
==========================

Four defects were found here on 2026-11-06, all measured, all in the same small
surface, and three of them had the same root: **the rule for a skill's name was
written three times and the three copies disagreed.**

1. Refusals were returned as values.  `skill_view` for a missing name answered
   `{"ok": true, "result": {"error": "not found"}}` -- one reply, two `ok`s, the
   outer one claiming the call succeeded.  `skill_write` did it four times
   (bad name, no description, empty body, no skill directory), `skill_forget`
   once, `skill_port` once.  Six refusals wearing a success envelope.  That is
   the fourth time this exact shape has been found in this package -- after
   `verify_ledger`, the three `schedule_*` methods, and `install_system_task`.

2. `skill_view` ignored the `name:` a package declares in its frontmatter, while
   the listing used it.  On `alpha/dir_named_pkg/SKILL.md` declaring
   `name: declared_name`, the listing said `declared_name` and `skill_view`
   answered only to `dir_named_pkg`.  Both were "right" by their own rule.

3. A package whose frontmatter did not repeat its own name disappeared from the
   listing entirely.  `path.stem` for `SKILL.md` is the string `"SKILL"`, which
   `valid_name` rejects for being upper case, so the file fell into `errors` and
   was never listed -- while `skill_view` served it happily under its directory
   name.  The listing said "there is no such skill" about a file it had just
   read.

4. `skill_view` could not count a load for a skill that had never been listed.
   `store.skill_load` returns 0 when the row is missing, and rows are created by
   `SkillLibrary.scan()`.  So `skill_write` then `skill_view` left `loads = 0`,
   and a skill the agent used every turn reported "never used" until something
   else happened to list the library.

The flat layout agreed with itself, which is why none of this was noticed: the
flat layout is the only one the agent writes itself, and the only one where all
three rules coincide.

Note the fifth defect this suite would have caught had it existed an hour
earlier, because it is the same disease one layer down: `skill_forget` first
resolved through one `SkillLibrary` and then archived through a *second* one,
which had never scanned, so it refused a skill the listing had just shown.
Resolving and acting have to happen on one instance -- see the
`skill_forget` assertion below.

What it does NOT claim
======================

* Not that `SkillLibrary` itself is correct in every layout -- it is exercised
  here through the wire, with the three layouts that were measured.
* Not that concurrent writers are safe.  Every write here goes through one
  service process, as the wire does.
* Not that `skill_port` copies anything useful: only its refusal path is
  checked here.  Porting real procedures is `autoforge_skill_port`'s own
  business and needs a source that exists.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

# Bootstrap the package from this file's own location. It used to rely on the
# caller exporting PYTHONPATH, which meant a fresh clone failed here with
# `ModuleNotFoundError: No module named 'autoforge_svc'` -- a message about the
# code when the problem was the environment. Measured 2026-11-11 on a copy of
# this tree in a temp directory with PYTHONPATH unset.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import autoforge_svc

# The service directory, derived from the package that is actually imported.
#
# Not a literal path.  `m20_mutate.py` copies the tree to a temp directory and
# points PYTHONPATH at the copy; a hard-coded SERVICE would make every child
# process below talk to the real, healthy service while the mutated copy sat
# unread -- which is exactly how four mutants escaped `chk_schedule.py` earlier
# today.  Same rule, same reason, recorded in
# `autoforge-suite-must-derive-its-own-service-path`.
SERVICE = os.path.dirname(os.path.dirname(os.path.abspath(autoforge_svc.__file__)))
PY = sys.executable

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print(f"OK   {label}")
    else:
        FAILS.append(label)
        print(f"FAIL {label} <- {detail!r}")


FLAT_ONE = """---
name: flat_one
description: a flat file, the layout the agent writes itself
---
body of the flat one
"""

DECLARED = """---
name: declared_name
description: the package declares a name unlike its directory
---
body of the declared one
"""

UNDECLARED = """---
description: the package never repeats its own name
---
body of the undeclared one
"""

# Runs inside the child service before its dispatcher starts.  It replaces
# `builtins.open` for one file only: the path is read normally by `scan` (which
# asks for it via `Path.read_text`, i.e. `io.open`, not `builtins.open`) and
# refused later by `skill_view` (which calls `builtins.open` directly).  That
# asymmetry is the point -- it puts the suite on the `except OSError` branch
# deliberately instead of hoping for a race.
UNREADABLE = r'''
import builtins, io, sys
from autoforge_svc import service as S

_real_open = builtins.open
TARGET = "unreadable_pkg"


def _open(file, *args, **kwargs):
    if TARGET in str(file):
        raise PermissionError(13, "permission denied by the suite")
    return _real_open(file, *args, **kwargs)


builtins.open = _open
raise SystemExit(S.main([]))
'''


def skill_tree(root: str) -> dict[str, str]:
    """Every file left under the skills directory, relative and sorted."""
    found: dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            found[os.path.relpath(full, root)] = open(full, encoding="utf-8").read()
    return dict(sorted(found.items()))


def wire(home: str, skills: str, requests: list[dict],
         extra_env: dict[str, str] | None = None,
         prelude: str | None = None) -> dict[int, dict]:
    """Send requests to a real `python -m autoforge_svc` and read the envelopes.

    `prelude` runs inside the child before `main([])` and is how a branch that
    only a race can reach becomes reachable on purpose -- the `except OSError`
    in `skill_view` fires only when the file is readable at scan time and
    unreadable a moment later, which no fixture can arrange by timing.  Same
    mechanism as `chk_wire_meter.py`'s broken meter.
    """
    env = dict(os.environ)
    env["AUTOFORGE_HOME"] = home
    env["AUTOFORGE_SKILLS_DIRS"] = skills
    env["PYTHONPATH"] = SERVICE
    env.pop("AUTOFORGE_MODEL", None)
    if extra_env:
        env.update(extra_env)
    stdin = "\n".join(json.dumps(r) for r in requests) + "\n"
    argv = ([PY, "-P", "-c", prelude] if prelude
            else [PY, "-P", "-m", "autoforge_svc"])
    proc = subprocess.run(argv, input=stdin,
                          text=True, capture_output=True, env=env, timeout=240)
    out: dict[int, dict] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            reply = json.loads(line)
        except ValueError:
            continue
        out[reply.get("id")] = reply
    if not out:
        raise SystemExit(f"the service answered nothing; stderr: {proc.stderr[-600:]}")
    return out


def names_of(reply: dict) -> list[str]:
    """The names in a listing reply, or [] if the reply is not a listing.

    Guarded rather than indexed: when a mutant makes the listing a dict, an
    unguarded comprehension raises, the suite dies before printing a verdict,
    and the driver records a crash instead of a catch -- a crash is not a
    catch.  Learned the hard way on `chk_mission.py`'s M8.
    """
    result = reply.get("result")
    if not isinstance(result, list):
        return []
    return [row.get("name") for row in result if isinstance(row, dict)]


def main() -> int:
    root = tempfile.mkdtemp(prefix="chk_skill_")
    try:
        home = os.path.join(root, "home")
        skills = os.path.join(root, "skills")
        os.makedirs(home)
        for rel, body in (("alpha/dir_named_pkg/SKILL.md", DECLARED),
                          ("alpha/undeclared_pkg/SKILL.md", UNDECLARED),
                          ("flat_one.md", FLAT_ONE)):
            full = os.path.join(skills, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as handle:
                handle.write(body)

        requests = [
            {"id": 1, "method": "skills", "params": {}},
            {"id": 2, "method": "skill_view", "params": {"name": "declared_name"}},
            {"id": 3, "method": "skill_view", "params": {"name": "dir_named_pkg"}},
            {"id": 4, "method": "skill_view", "params": {"name": "undeclared_pkg"}},
            {"id": 5, "method": "skill_view", "params": {"name": "FLAT_ONE"}},
            {"id": 6, "method": "skill_write",
             "params": {"name": "fresh_one", "description": "written then read",
                        "body": "body of the fresh one"}},
            {"id": 7, "method": "skill_view", "params": {"name": "fresh_one"}},
            {"id": 8, "method": "skills", "params": {}},
            {"id": 9, "method": "skill_forget", "params": {"name": "DECLARED_NAME"}},
            {"id": 10, "method": "skill_view", "params": {"name": "declared_name"}},
            {"id": 11, "method": "skill_view", "params": {"name": "no_such_skill"}},
            {"id": 12, "method": "skill_write",
             "params": {"name": "Bad Name", "description": "x", "body": "y"}},
            {"id": 13, "method": "skill_write",
             "params": {"name": "ok_name", "description": "", "body": "y"}},
            {"id": 14, "method": "skill_write",
             "params": {"name": "ok_name", "description": "d", "body": "   "}},
            {"id": 15, "method": "skill_port", "params": {"source": "nonsense"}},
            {"id": 16, "method": "report", "params": {}},
            # A listing *after* the forget.  The first version of this suite
            # compared against the listing from before it and failed while the
            # product was right -- the assertion was asking the wrong message.
            {"id": 17, "method": "skills", "params": {}},
        ]
        replies = wire(home, skills, requests)

        print("\n-- one name rule --")
        listed = names_of(replies[1])
        check("the listing shows every file on disk",
              sorted(listed) == ["declared_name", "flat_one", "undeclared_pkg"], listed)
        check("a declared name wins over the directory name",
              "declared_name" in listed and "dir_named_pkg" not in listed, listed)
        check("a package that never declares its name is listed under its directory",
              "undeclared_pkg" in listed, listed)

        body = replies[2].get("result") or {}
        check("skill_view answers to the declared name",
              replies[2].get("ok") is True and body.get("name") == "declared_name", body)
        check("...and only to that name, not to the directory",
              replies[3].get("ok") is False
              and "no skill named" in (replies[3].get("error") or ""),
              replies[3].get("error"))
        check("skill_view serves the package that declares no name",
              replies[4].get("ok") is True
              and (replies[4].get("result") or {}).get("name") == "undeclared_pkg",
              replies[4])
        check("a differently-cased request reaches the same skill",
              replies[5].get("ok") is True
              and (replies[5].get("result") or {}).get("name") == "flat_one",
              replies[5])

        print("\n-- write, read, count --")
        check("skill_write answers with the four fields it promises",
              replies[6].get("ok") is True
              and set(replies[6].get("result") or {}) == {"created", "name", "path", "updated"},
              replies[6].get("result"))
        check("a written skill can be read straight back",
              replies[7].get("ok") is True
              and (replies[7].get("result") or {}).get("name") == "fresh_one",
              replies[7].get("result"))
        # The counter is the reason `_skill_named` scans: without a scan there is
        # no row, `skill_load` returns 0, and a skill read every turn still says
        # "never used".
        report = replies[16].get("result") or {}
        check("a skill that has never been listed is still listed after one write",
              "fresh_one" in names_of(replies[8]), names_of(replies[8]))
        # `skills_never_loaded` is a COUNT, not a list.  The first version of
        # this assertion wrote `(report.get(...) or [])` and iterated it: with a
        # healthy service the count is 0, `0 or []` is `[]`, the loop is empty
        # and the check passed -- an assertion that only held because the number
        # it was given happened to be zero.  A mutant that inflated the count
        # crashed the suite with `TypeError: 'int' object is not iterable`, and
        # a crash is not a catch.  Assert the number, and assert its type.
        never = report.get("skills_never_loaded")
        check("the report counts skills nobody has read",
              isinstance(never, int) and not isinstance(never, bool),
              never)
        check("the skill just read is not counted as never loaded",
              never == 0, never)
        check("the report knows how many skills exist",
              report.get("skills") == 3, report.get("skills"))

        print("\n-- forget moves the file, under the name that exists --")
        forgot = replies[9].get("result") or {}
        check("skill_forget succeeds for a differently-cased request",
              replies[9].get("ok") is True, replies[9].get("error"))
        check("...and answers with the name that exists, not the one it was asked",
              forgot.get("name") == "declared_name", forgot)
        check("...and the archive path is the file that moved",
              str(forgot.get("archived", "")).endswith("SKILL.md")
              and ".archive" in str(forgot.get("archived", "")), forgot.get("archived"))
        check("the forgotten skill is gone from the listing",
              "declared_name" not in names_of(replies[17]), names_of(replies[17]))
        check("reading it afterwards is a refusal",
              replies[10].get("ok") is False
              and "no skill named" in (replies[10].get("error") or ""),
              replies[10].get("error"))

        print("\n-- every refusal is a refusal --")
        # The wording each refusal actually uses.  The first version expected
        # "no such skill" and failed against "no skill named 'no_such_skill'":
        # an assertion about a sentence I had imagined rather than the one the
        # product says.  Quoting the product is the point of the line below.
        refusing = {11: "no skill named", 12: "invalid name", 13: "description",
                    14: "body", 15: "unknown source"}
        for rid, expected in refusing.items():
            reply = replies[rid]
            check(f"the wire says the call failed ({expected})",
                  reply.get("ok") is False, reply.get("ok"))
            check(f"...with no result key to mistake for an answer ({expected})",
                  "result" not in reply, list(reply))
            check(f"...and the reason names the cause ({expected})",
                  expected in (reply.get("error") or "").lower(), reply.get("error"))

        print("\n-- the tree on disk --")
        on_disk = skill_tree(skills)
        check("nothing was deleted, only archived",
              set(on_disk) == {os.path.join("alpha", "dir_named_pkg", ".archive", "SKILL.md"),
                               os.path.join("alpha", "undeclared_pkg", "SKILL.md"),
                               "flat_one.md", "fresh_one.md"},
              sorted(on_disk))
        check("the archived file is still readable",
              "body of the declared one" in on_disk.get(
                  os.path.join("alpha", "dir_named_pkg", ".archive", "SKILL.md"), ""),
              list(on_disk))

        print("\n-- a skill that is there and cannot be read --")
        # The `except OSError` in `skill_view` is reachable only when the file is
        # readable at scan time and unreadable a moment later.  No fixture can
        # arrange that by timing, so the child is told to refuse one path: that
        # is how a branch nobody's test ever walks gets walked.  Without this
        # section, a mutant that turned the refusal back into `{"error": "not
        # found"}` escaped -- the suite had no request that could reach the line.
        broken = os.path.join(root, "broken")
        os.makedirs(os.path.join(broken, "unreadable_pkg"))
        with open(os.path.join(broken, "unreadable_pkg", "SKILL.md"), "w",
                  encoding="utf-8") as handle:
            handle.write("---\nname: unreadable_pkg\ndescription: readable at scan time\n---\nb\n")
        third = wire(home, broken,
                     [{"id": 1, "method": "skills", "params": {}},
                      {"id": 2, "method": "skill_view",
                       "params": {"name": "unreadable_pkg"}}],
                     prelude=UNREADABLE)
        check("the scan could read it, so the listing shows it",
              names_of(third[1]) == ["unreadable_pkg"], names_of(third[1]))
        unreadable = third[2]
        check("reading it afterwards is a refusal",
              unreadable.get("ok") is False, unreadable.get("ok"))
        check("...and the reason says it exists but could not be read",
              "could not be read" in (unreadable.get("error") or ""),
              unreadable.get("error"))
        check("...and it is not reported as a skill that is not there",
              "no skill named" not in (unreadable.get("error") or ""),
              unreadable.get("error"))

        print("\n-- the directories come from the environment --")
        other = os.path.join(root, "elsewhere")
        os.makedirs(other)
        with open(os.path.join(other, "env_only.md"), "w", encoding="utf-8") as handle:
            handle.write("---\nname: env_only\ndescription: only in the environment dir\n---\nb\n")
        second = wire(home, other, [{"id": 1, "method": "skills", "params": {}}])
        check("AUTOFORGE_SKILLS_DIRS decides where skills are read from",
              names_of(second[1]) == ["env_only"], names_of(second[1]))

        print("")
        print("TOTAL FAILURES:", len(FAILS), FAILS)
        return 1 if FAILS else 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
