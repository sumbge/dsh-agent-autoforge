"""The two skill lines the brief cannot get from the menu alone.

Why this file exists
--------------------
`_skill_lines()` names the procedures. That is the menu, and it answers "what
do I have" for the obvious case. It does not answer two questions that change
what the agent does next, and upstream answers both in `_self_report`
(`agent.py:4178`):

* **Borrowed.** Nine procedures in this library are not the agent's own -- they
  were ported from Intelligence Indeed and are tagged `ported`. A skill it did
  not write is one it has less reason to trust and less reason to edit, and the
  brief said nothing about which was which. Upstream's line is explicit that
  the count is "of what is actually in the library now -- not a claim that a
  checkout exists somewhere", because a file in `vendor/` is a file; the number
  that matters is how many it can load by name this turn.
* **Skipped.** `SkillLibrary.scan()` collects two kinds of casualty and never
  raises: files it could not read, and skills shadowed by a more specific
  directory. Upstream renders "N unusable file(s), M shadowed by a more
  specific directory -- `skill_errors` says which and why". This branch said
  nothing at all, so a library where four of forty skills are broken looks
  exactly like one where none are. That is the failure mode `skill_errors`
  exists to prevent, defused by the brief never mentioning it.

What is deliberately NOT here
-----------------------------
Upstream's `_self_report` also carries `loads_total` and `never_loaded`. Those
were left out on purpose: they move whenever any skill is read, and this block
is a prompt *prefix*, so a changing integer re-bills every character behind it.
The counts are still available -- `skill_list` prints them, `skill_errors`
enumerates the casualties. The prefix keeps the part that does not churn.
`chk_brief_probe.py` already guards that property for the probe block; the last
assertion here guards it for this one.

What is checked
---------------
1. A borrowed count appears, and it equals the number tagged `ported`.
2. It is a *measurement*: with no ported skills the line is absent, with two it
   says two.
3. A skipped line appears only when something was actually skipped.
4. It names the file count and points at `skill_errors`.
5. Neither line carries a number that changes per turn (loads, calls).
6. The whole skill block still fits its budget, with the new lines included.
"""
import os
import re
import shutil
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)

from autoforge_svc.service import Service  # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print(f"  OK   {label}")
    else:
        print(f"  FAIL {label}  <- {detail}")
        FAILS.append(label)


def write_skill(root: str, name: str, tags: str, *, description: str = "does a thing") -> str:
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "SKILL.md")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"when_to_use: when {name} is the thing to do\n"
            f"tags: {tags}\n"
            "---\n\n"
            f"# {name}\n\nBody of {name}.\n")
    return p


tmp = tempfile.mkdtemp(prefix="m24_skl_")
skills_dir = os.path.join(tmp, "skills")
os.makedirs(skills_dir, exist_ok=True)
os.environ["AUTOFORGE_SKILLS_DIRS"] = skills_dir
db = os.path.join(tmp, "s.db")
svc = Service(db_path=db)

print("-- a library with nothing borrowed says nothing about borrowing")
write_skill(skills_dir, "own-one", "own")
write_skill(skills_dir, "own-two", "own,housekeeping")
lines = svc._skill_lines()
text = "\n".join(lines)
check("the menu names the skills", "own-one" in text and "own-two" in text, text[:200])
check("no borrowed line when nothing is borrowed", "Borrowed" not in text, text[-200:])
check("no skipped line when nothing was skipped", "skipped" not in text.lower(), text[-200:])

print("-- borrowing is counted from the library, not asserted")
write_skill(skills_dir, "tars-skill-alpha", "ported")
write_skill(skills_dir, "tars-skill-beta", "ported,other")
text2 = "\n".join(svc._skill_lines())
found = re.search(r"Borrowed[:,]\s*(\d+)", text2)
check("there is a borrowed line now", bool(found), text2[:400])
check("and it counts the tagged ones", found is not None and found.group(1) == "2",
      found.group(1) if found else "absent")
check("it says where they came from", "Intelligence Indeed" in text2, text2[:400])
check("and which tool lists them", "tars_skills" in text2, text2[:400])

print("-- what could not be read is reported, not swallowed")
bad = os.path.join(skills_dir, "broken-pack")
os.makedirs(bad, exist_ok=True)
with open(os.path.join(bad, "SKILL.md"), "w", encoding="utf-8", newline="\n") as fh:
    # No frontmatter description at all: `scan()` records this as unusable.
    fh.write("---\nname: broken-pack\ntags: own\n---\n\n# broken\n")
text3 = "\n".join(svc._skill_lines())
check("a skipped line appears once something is skipped", "skipped" in text3.lower(),
      text3[-300:])
check("it gives the count", re.search(r"\b[1-9]\d*\b", text3.split("skipped", 1)[-1]) is not None,
      text3[-200:])
check("and it points at the tool that explains", "skill_errors" in text3, text3[-300:])

print("-- the skipped line must not claim a reason it does not have")
# `scan()` keeps the FIRST skill it meets for a name and records the others in
# `self.shadowed`. Whether the winner is "more specific" depends on directory
# order, not on specificity: two packages named the same under one root are
# decided by walk order. The line said "shadowed by a more specific directory"
# anyway -- inherited wording, false in that case -- and `skill_errors`, the
# tool it points at, said nothing about shadowing at all.
shadow_root = skills_dir
os.makedirs(os.path.join(shadow_root, "zeta", "twice"), exist_ok=True)
os.makedirs(os.path.join(shadow_root, "alpha", "twice"), exist_ok=True)
for where, tag in (("zeta", "zeta"), ("alpha", "alpha")):
    with open(os.path.join(shadow_root, where, "twice", "SKILL.md"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("---\nname: twice\ndescription: from %s\n---\n\nbody\n" % tag)
shadow_text = "\n".join(svc._skill_lines())
check("a shadowed name is counted", re.search(r"\bshadowed\b", shadow_text) is not None,
      shadow_text[-300:])
check("the line does not claim the winner was more specific",
      "more specific" not in shadow_text.split("skipped", 1)[-1],
      shadow_text.split("skipped", 1)[-1][:200])
# Asked inside a guard on purpose. The bug this replaced WAS a raise, so an
# unguarded call reports it as a traceback that ends the file -- which says the
# run could not finish, not which claim is false. A guard turns it into a FAIL.
raised = ""
try:
    reasons = svc.skill_errors()
except Exception as exc:                                   # noqa: BLE001
    reasons = []
    raised = "%s: %s" % (type(exc).__name__, exc)
check("skill_errors answers instead of raising on a file it cannot parse",
      not raised, raised)
# The row type is the field that says what it is. Filtering on the error
# TEXT would tie this test to a sentence that is allowed to change.
shadow_rows = [r for r in reasons if r.get("kind") == "shadowed"]
check("skill_errors really does report the shadowed copy", bool(shadow_rows), reasons[-3:])
check("and it names the file that was ignored",
      any("twice" in str(r.get("path", "")) for r in shadow_rows), shadow_rows[:2])
check("and the file that won, as a separate field",
      any(r.get("winner") and r.get("winner") != r.get("path") for r in shadow_rows),
      shadow_rows[:2])
check("both paths point at copies of the same name",
      all("twice" in str(r.get("path", "")) and "twice" in str(r.get("winner", ""))
          for r in shadow_rows), shadow_rows[:2])

print("-- the lines do not move when a skill is read")
# The property, not a word order: this text is a prompt *prefix*, so a figure
# that changes when any skill is viewed re-bills every character behind it.
# Checking for the shape "N loads" would miss `loads: N`, and checking for the
# noun `loads` would fail on the menu's own heading ("skill_view loads one").
# So the block is rendered twice, with a load recorded in between, and the two
# renderings must be byte-identical.
before = "\n".join(svc._skill_lines())
svc.store.skill_load("own-one")
svc.store.skill_load("own-one")
after = "\n".join(svc._skill_lines())
check("a recorded load does not change the block", before == after,
      [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b][:2])
check("but the load really was recorded",
      any(r.get("loads", 0) >= 2 for r in svc.skills()), "no skill reports a load")

print("-- and the block still fits")
check("the skill block stays inside its budget",
      len(text3) <= svc.SKILL_BUDGET_CHARS + 120,
      f"{len(text3)} vs budget {svc.SKILL_BUDGET_CHARS}")

print("-- the brief as a whole still assembles")
brief = svc.context_brief()
b = str(brief.get("text") or "")
check("the brief has it all", "Borrowed" in b and "SKILLS" in b, b[-400:])
check("and fits its own budget", int(brief.get("chars") or 0) <= 4000, brief.get("chars"))

svc.store.close()
svc.missions.close()
os.environ.pop("AUTOFORGE_SKILLS_DIRS", None)
shutil.rmtree(tmp, ignore_errors=True)

print("TOTAL FAILURES:", len(FAILS), FAILS)
sys.exit(0 if not FAILS else 1)
