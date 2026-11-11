"""The host probe's two false statements, and the store it cannot find.

Why this file exists
--------------------
`tars_port.probe_host()` is the only place in this package that reports on
itself in the first person, and it said two things that were not true.

**1. "32/32 match upstream -- MISMATCH: README.md, ..."** on one line.

    lines.append("vendor files verified: " + (
        str(check["checked"]) + "/" + str(check["listed"]) + " match upstream"
        + ("" if check["ok"] else " -- MISMATCH: " + ", ".join(...))))

`checked` counts *files read*, not *files that matched*, and it was being
printed as the numerator of "match upstream" right beside the list of files
that did not match. Measured 2026-11-03: `verify_vendor()` returned
`{"ok": false, "checked": 32, "listed": 32, "mismatched": [32 items]}`, and the
sentence built from it announced `32/32 match upstream` and then, in the same
breath, listed all thirty-two as mismatches. A reader skimming would take the
first half as the answer.

**2. `store: not created yet`** on a host whose store holds eight thousand rows.

    home = Path(os.environ.get("LOCALAPPDATA") or ~) / "autoforge"
    db = home / "autoforge.db"

The real store is at `<LOCALAPPDATA>/autoforge/data/autoforge.db` -- `_home()`
in `service.py` adds the `data` segment (and honours `AUTOFORGE_HOME`). So the
probe looked one directory up from the store, found nothing, and reported the
absence as a fact about the machine rather than about its own arithmetic.

Both are the same failure this project keeps finding: a report that is not
lying on purpose, built from a number that answers a different question than
the one the sentence asks. So the checks below are about *the sentence*, not
about the code path that produces it.

What is checked
---------------
1. When everything matches, the sentence says so and names no mismatches.
2. When one file differs, the sentence does not claim any file matches, and it
   names the file that differs.
3. The numerator counts matches, so with two of three files matching it reads
   `1/3` (one matched) -- not `3/3`.
4. The store line names the file that exists on this host, when one does.
5. With no store anywhere, it says so in a way that cannot be read as "the
   machine has no store" -- the path it looked in is part of the sentence.
6. The probe's own cost is bounded: the market check must not be able to spend
   a second and a half of a per-turn block's budget.
"""
import os
import shutil
import sqlite3
import sys
import tempfile
import time

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)

from autoforge_svc import tars_port  # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print(f"  OK   {label}")
    else:
        print(f"  FAIL {label}  <- {detail}")
        FAILS.append(label)


def vendor_line(lines: list[str]) -> str:
    for ln in lines:
        if ln.startswith("vendor files verified:"):
            return ln
    return ""


def store_line(lines: list[str]) -> str:
    for ln in lines:
        if ln.startswith("store:"):
            return ln
    return ""


print("-- the vendor sentence, against a fake vendor tree")
# A throwaway copy of the real vendor directory so the fixture cannot drift away
# from the thing it describes. Three files, one of them altered.
root = tempfile.mkdtemp(prefix="m21_vendor_")
src = tars_port._VENDOR
names = sorted(tars_port.VENDOR_SHA256)[:3]
check("the vendored manifest is not empty", len(names) == 3, len(names))
for rel in names:
    dst = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(src, rel.replace("/", os.sep)), dst)

# The manifest is built from the bytes actually on disk here, not copied from the
# shipped one. The shipped manifest records the upstream LF digests while this
# checkout holds CRLF copies, so reusing it would make every check below measure
# the line-ending defect (a separate item) instead of the sentence this file is
# about. A fixture that cannot be right would make a passing test meaningless.
import hashlib  # noqa: E402

real_VENDOR, real_SHA = tars_port._VENDOR, dict(tars_port.VENDOR_SHA256)
try:
    tars_port._VENDOR = type(tars_port._VENDOR)(root)
    # -- all matching
    tars_port.VENDOR_SHA256 = {
        rel: hashlib.sha256(
            open(os.path.join(root, rel.replace("/", os.sep)), "rb").read()
        ).hexdigest()
        for rel in names
    }
    got = tars_port.verify_vendor()
    check("three copied files verify", got.get("ok") is True, got)
    line = vendor_line(tars_port.probe_host(home=root, timeout=0.05))
    check("the sentence says all of them match", line.startswith("vendor files verified: 3/3"), line)
    check("and names no mismatch", "differs" not in line and "MISMATCH" not in line, line)

    # -- one differing
    victim = names[0]
    with open(os.path.join(root, victim.replace("/", os.sep)), "ab") as fh:
        fh.write(b"\n# changed\n")
    got = tars_port.verify_vendor()
    check("the altered file is caught", got.get("ok") is False, got.get("ok"))
    check("and it is the only one", got.get("mismatched") == [victim], got.get("mismatched"))
    line = vendor_line(tars_port.probe_host(home=root, timeout=0.05))
    check("the sentence does not claim a match it does not have",
          "3/3" not in line.split("--")[0], line)
    check("it reports how many matched", "2/3" in line, line)
    check("and it names the file that differs", victim in line, line)
    check("the two halves of the sentence agree",
          ("differs:" in line) == (not got.get("ok")), line)
finally:
    tars_port._VENDOR, tars_port.VENDOR_SHA256 = real_VENDOR, real_SHA
    shutil.rmtree(root, ignore_errors=True)

print("-- the store line, when a store exists")
home = tempfile.mkdtemp(prefix="m21_home_")
# `home=` means "the directory holding autoforge.db" -- the same thing
# `service._home()` returns, which is `<LOCALAPPDATA>/autoforge/data`. So the
# store goes directly here; adding a `data` segment in the fixture would be
# asserting a layout the caller never promised.
db = os.path.join(home, "autoforge.db")
con = sqlite3.connect(db)
con.execute("CREATE TABLE t (x INTEGER)")
con.commit()
con.close()
try:
    line = store_line(tars_port.probe_host(home=home, timeout=0.05))
    check("the store line exists", bool(line), line)
    check("it finds the store it was pointed at", "not created yet" not in line, line)
    check("and it names the path it found", "autoforge.db" in line, line)
    check("it reports the size, not merely the presence", "bytes" in line, line)
finally:
    shutil.rmtree(home, ignore_errors=True)

print("-- and the default resolution is the one the service uses")
# With neither an argument nor AUTOFORGE_HOME, the probe must look where the
# service looks. This used to assert the literal `<LOCALAPPDATA>/autoforge/data`
# -- and on 2026-11-06 that literal turned out to be the *defect*: `_home_dir`
# derived its own home while `store.home_dir()` answered `<LOCALAPPDATA>/autoforge`,
# so the probe reported on a directory the service did not use. Pinning a literal
# made the test side with the wrong half.
#
# Two things are now checked separately, because they are two different claims:
# the probe agrees with the store (whatever the store says), and the store's
# answer is the one `service._home()` gives. If the default moves again, both
# stay true without an edit.
saved = {k: os.environ.get(k) for k in ("LOCALAPPDATA", "AUTOFORGE_HOME")}
base = tempfile.mkdtemp(prefix="m21_local_")
try:
    os.environ["LOCALAPPDATA"] = base
    os.environ.pop("AUTOFORGE_HOME", None)
    resolved = tars_port._home_dir(None)
    from autoforge_svc.store import home_dir as store_home
    check("the probe looks where the store looks",
          os.path.normcase(str(resolved)) == os.path.normcase(store_home()),
          "%r vs %r" % (str(resolved), store_home()))
    check("and that is the directory service._home() reports",
          os.path.normcase(store_home()) == os.path.normcase(
              os.path.join(os.environ["LOCALAPPDATA"], "autoforge")),
          store_home())
    check("the preset's data segment is a preset, not the default",
          not str(resolved).replace(os.sep, "/").endswith("/autoforge/data"),
          str(resolved))
    os.environ["AUTOFORGE_HOME"] = os.path.join(base, "elsewhere")
    check("AUTOFORGE_HOME overrides it, because the service honours it",
          tars_port._home_dir(None) == type(resolved)(os.path.join(base, "elsewhere")),
          str(tars_port._home_dir(None)))
finally:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    shutil.rmtree(base, ignore_errors=True)

print("-- and when there is no store at all, the sentence says where it looked")
home2 = tempfile.mkdtemp(prefix="m21_empty_")
try:
    line = store_line(tars_port.probe_host(home=home2, timeout=0.05))
    check("the line still exists", bool(line), line)
    check("it does not present absence as a fact about the machine",
          "not created yet" not in line or home2 in line, line)
    check("the path it searched is part of the answer",
          "autoforge.db" in line, line)
finally:
    shutil.rmtree(home2, ignore_errors=True)

print("-- the probe cannot spend a whole turn's budget on a dead port")
home3 = tempfile.mkdtemp(prefix="m21_cost_")
try:
    started = time.perf_counter()
    tars_port.probe_host(home=home3, timeout=0.05)
    elapsed_ms = (time.perf_counter() - started) * 1000
    check("the probe finishes well inside a per-turn block", elapsed_ms < 800, f"{elapsed_ms:.0f} ms")
    check("it still reports every line it used to",
          len(tars_port.probe_host(home=home3, timeout=0.05)) >= 8,
          len(tars_port.probe_host(home=home3, timeout=0.05)))
finally:
    shutil.rmtree(home3, ignore_errors=True)

print("-- the real vendor tree matches its own manifest")
# This is the check the sentence above is *about*. Before the line endings were
# normalised it failed 32 of 32, and the failure was invisible because the
# sentence announced success in its first half.
real = tars_port.verify_vendor()
check("every vendored file verifies", real.get("ok") is True,
      {"checked": real.get("checked"), "mismatched": list(real.get("mismatched") or [])[:5]})
check("all of them are accounted for",
      int(real.get("checked") or 0) == int(real.get("listed") or -1) > 0,
      f"{real.get('checked')}/{real.get('listed')}")
check("nothing is missing", not real.get("missing"), list(real.get("missing") or []))
manifest = os.path.join(tars_port._VENDOR, "MANIFEST.sha256")
raw = open(manifest, "rb").read()
check("the manifest itself is LF, like the files it describes", b"\r\n" not in raw,
      raw.count(b"\r\n"))
check("and no vendored file carries a CRLF",
      not any(b"\r\n" in open(os.path.join(tars_port._VENDOR, rel.replace("/", os.sep)), "rb").read()
              for rel in tars_port.VENDOR_SHA256),
      "a Windows checkout rewrote them; .gitattributes must prevent that")
check("a .gitattributes records the rule",
      os.path.isfile(os.path.join(os.path.dirname(SERVICE), ".gitattributes")),
      "no .gitattributes at the plugin root -- the next checkout will undo this")

print("-- the skill lines answer about the skill directories, not about `home`")
# The third false statement, same shape as the second: `probe_host` resolved the
# skills directory as `home / "skills"`, but the library puts skills beside
# `home`, not under it -- the preset sets `home = <LOCALAPPDATA>\autoforge\data`
# and `skills = <LOCALAPPDATA>\autoforge\skills`. So on this machine the probe
# printed `skills: MISSING` while the directory held a skill file. `home` is
# the *store's* directory; the skills directories have their own resolution
# (`skills.default_skill_dirs`, which also honours `AUTOFORGE_SKILLS_DIRS`).
#
# A one-level `*.md` glob would also undercount: every skill shipped as a
# package is `<category>/<name>/SKILL.md`. So the fixture below is a package
# layout plus one flat file, and the expected count is two.
skillroot = tempfile.mkdtemp(prefix="m21_skills_")
pack = os.path.join(skillroot, "cat", "pack")
os.makedirs(pack, exist_ok=True)
with open(os.path.join(pack, "SKILL.md"), "w", encoding="utf-8") as fh:
    fh.write("# a packaged procedure\n")
with open(os.path.join(skillroot, "flat.md"), "w", encoding="utf-8") as fh:
    fh.write("# a flat procedure\n")

from autoforge_svc import skills as skills_mod  # noqa: E402

check("the fixture holds a package a flat glob cannot see",
      len(list(skills_mod.iter_skill_files(skillroot))) == 2
      and len([f for f in os.listdir(skillroot) if f.endswith(".md")]) == 1,
      len(list(skills_mod.iter_skill_files(skillroot))))

home4 = tempfile.mkdtemp(prefix="m21_home4_")
saved_dirs = os.environ.get("AUTOFORGE_SKILLS_DIRS")
os.environ["AUTOFORGE_SKILLS_DIRS"] = skillroot
try:
    lines4 = tars_port.probe_host(home=home4, timeout=0.05)
finally:
    if saved_dirs is None:
        os.environ.pop("AUTOFORGE_SKILLS_DIRS", None)
    else:
        os.environ["AUTOFORGE_SKILLS_DIRS"] = saved_dirs
    shutil.rmtree(home4, ignore_errors=True)

skill_lines = [ln for ln in lines4 if ln.startswith("skills")]
check("the probe says something about skills", bool(skill_lines), lines4)
check("no skill line calls an existing directory missing",
      not any("MISSING" in ln for ln in skill_lines), skill_lines)
check("it names the directory it counted",
      any(skillroot in ln for ln in skill_lines), skill_lines)
check("it counts skill files, not top-level entries",
      any("2 skill(s)" in ln for ln in skill_lines), skill_lines)
check("AUTOFORGE_SKILLS_DIRS wins over anything guessed from `home`",
      any(skillroot in ln for ln in skill_lines)
      and not any("no skill directory" in ln or "MISSING" in ln for ln in skill_lines),
      skill_lines)

archive_lines = [ln for ln in lines4 if ln.startswith("archive") or ln.startswith(".archive")]
check("the archive has a line of its own", bool(archive_lines), lines4)
check("an archive that was never used is not reported as missing",
      not any("MISSING" in ln for ln in archive_lines), archive_lines)
check("and the line says when it will appear",
      any("first use" in ln or "not created yet" in ln for ln in archive_lines),
      archive_lines)

# And when one does exist, the count is reported.
os.makedirs(os.path.join(skillroot, ".archive"), exist_ok=True)
with open(os.path.join(skillroot, ".archive", "old.md"), "w", encoding="utf-8") as fh:
    fh.write("# archived\n")
os.environ["AUTOFORGE_SKILLS_DIRS"] = skillroot
home5 = tempfile.mkdtemp(prefix="m21_home5_")
try:
    lines5 = tars_port.probe_host(home=home5, timeout=0.05)
finally:
    if saved_dirs is None:
        os.environ.pop("AUTOFORGE_SKILLS_DIRS", None)
    else:
        os.environ["AUTOFORGE_SKILLS_DIRS"] = saved_dirs
    shutil.rmtree(home5, ignore_errors=True)
    shutil.rmtree(skillroot, ignore_errors=True)
arch5 = [ln for ln in lines5 if ln.startswith("archive") or ln.startswith(".archive")]
check("an existing archive is counted", any("1 file(s)" in ln for ln in arch5), arch5)

print("TOTAL FAILURES:", len(FAILS), FAILS)
sys.exit(0 if not FAILS else 1)
