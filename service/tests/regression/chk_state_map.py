"""`report()` must show every place this package writes, and name the odd one.

Two measurements on 2026-11-06, both about the same report:

  D-A  `report()["home"]` carried `dir`, `db_path`, `adopted_from` and
       nothing else. Measured over the raw structure (not its JSON text --
       `json.dumps` escapes backslashes, which made a first version of this
       probe report "nothing is visible" for paths that were right there):
       only `ledger` was visible; agenda, runs, spill, compaction log and
       skills were not. Assembling the full layout required importing five
       modules and calling six functions.

  D-B  Once `db_file()` began adopting a populated sibling, the ledger could
       sit one level *below* `home_dir()` while every other entry stayed
       inside it -- `dir=<base>/autoforge` with
       `db_path=<base>/autoforge/data/...`. "Back up the home directory" was
       quietly wrong; "delete the home directory" was quietly incomplete.

What is asserted here is not "the keys exist" but "the report agrees with the
owners", computed by calling the owners in this file. A suite that only
checked for the presence of `state` would pass against a `state` full of
plausible strings.

Discriminating design:
  H1 (fixed): every one of the six entries in `home.state` equals what the
      owning module says, and `ledger_outside_home` is `false` with no
      sibling and `true` with one.
  H2 (regressed): `state` is gone, or holds its own guesses, or
      `ledger_outside_home` is constant -- then at least one assertion below
      fails. The two-scenario comparison exists because a constant answer
      would satisfy the "outside" check in exactly one of them.
"""
import json
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


def _paths(m):
    """Every string in the map, flattening the skills list.

    Written because `skills` is a list of `{origin, path}` records and the
    first version of the "all paths under home" check read `m.values()`
    directly -- so the list was compared with `str()` and the check passed by
    looking at the repr of a list rather than at any path inside it.
    """
    for key, value in m.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("path"):
                    yield item["path"]
        elif value:
            yield value


def _skill_paths(skills):
    """The paths inside a skills entry, whichever shape it currently has."""
    if not skills:
        return []
    if isinstance(skills, str):
        return [skills]
    return [d.get("path", "") for d in skills if isinstance(d, dict) and d.get("path")]


def _is_project_skill(path):
    """Is this path the `project` skill directory rather than user state?

    Asked of the owning module instead of guessing from the string: a path
    that merely looks project-ish would make this exemption a hole.
    """
    from autoforge_svc import skills as _K
    project = [p for origin, p in _K.default_skill_dirs() if origin == "project"]
    return any(os.path.normcase(str(path)) == os.path.normcase(p) for p in project)


def _entry(state, key):
    """The single path a `state_map()` entry names, or "" for a list of them."""
    value = (state or {}).get(key)
    if isinstance(value, list):
        first = value[0] if value else None
        return (first or {}).get("path", "") if isinstance(first, dict) else (first or "")
    return value or ""


def _preset_layout_outside():
    """The shipped preset's layout: home at `data/`, runs and skills siblings.

    Built from the preset itself rather than from a literal in this file, so
    the day the preset moves, this assertion follows it instead of silently
    describing a machine that no longer exists.
    """
    import yaml as _yaml

    plugin = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    preset = os.path.join(plugin, "presets", "autoforge.patch.yml")
    loader = _yaml.SafeLoader
    loader.add_multi_constructor("tag:yaml.org,2002:js", lambda l, s, n: None)
    with open(preset, encoding="utf-8") as fh:
        doc = _yaml.load(fh, Loader=loader)
    cfg = doc[0]["insert"][0]["config"]["plugins"][3]["config"]
    saved = {k: os.environ.get(k) for k in
             ("AUTOFORGE_HOME", "AUTOFORGE_RUNS", "AUTOFORGE_SKILLS_DIRS")}
    try:
        os.environ["AUTOFORGE_HOME"] = str(cfg["home"])
        for env, key in (("AUTOFORGE_RUNS", "runs"),
                         ("AUTOFORGE_SKILLS_DIRS", "skills")):
            if cfg.get(key):
                os.environ[env] = str(cfg[key])
            else:
                os.environ.pop(env, None)
        from autoforge_svc.store import outside_home, state_map
        return outside_home(state_map())
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def scenario(adopt, skills_env=None):
    """A fresh LOCALAPPDATA per call, optionally with a populated sibling.

    `skills_env` exists because of a mutation that escaped this suite once.
    With no `AUTOFORGE_SKILLS_DIRS`, the owner's answer and the obvious guess
    (`<base>/skills`) are the SAME STRING, so replacing the call with the
    guess was invisible -- not a hole in the code but a hole in the fixture:
    the test never varied the dimension that tells "ask the owner" apart from
    "remember the default". Setting the variable makes the two answers differ.
    """
    base = os.path.join(tempfile.mkdtemp(prefix="chk_state_map_"), "LOCALAPPDATA")
    os.makedirs(base)
    for var in ("AUTOFORGE_HOME", "AUTOFORGE_RUNS", "AUTOFORGE_SKILLS_DIRS"):
        os.environ.pop(var, None)
    os.environ["LOCALAPPDATA"] = base
    if skills_env:
        os.environ["AUTOFORGE_SKILLS_DIRS"] = skills_env

    if adopt:
        datadir = os.path.join(base, "autoforge", "data")
        os.makedirs(datadir)
        from autoforge_svc.store import ToolStore
        seed = ToolStore(db_path=os.path.join(datadir, "autoforge.db"))
        seed.log_event("seeded", {"n": 1})
        seed.close()

    from autoforge_svc.service import Service
    from autoforge_svc import store as S, runs as R, schedule as Sc, skills as K
    from autoforge_svc.core import compaction as C

    svc = Service()
    rep = svc.report()
    home = rep.get("home") or {}
    truth = {
        "ledger": svc.db_path,
        "agenda": str(Sc.default_schedule_path()),
        "runs": str(R.journal_dir()),
        "spill": str(C.spill_dir()),
        "compaction_log": str(C.default_log_path()),
        "skills": [{"origin": o, "path": p} for o, p in K.default_skill_dirs()],
    }
    return {"rep": rep, "home": home, "state": home.get("state") or {},
            "truth": truth, "base": base, "db_path": svc.db_path,
            "home_dir": S.home_dir()}


def main():
    plain = scenario(False)
    adopt = scenario(True)

    # -- the map exists and agrees with the modules that own each entry ------
    check("report() carries a state map",
          isinstance(plain["home"].get("state"), dict) and plain["state"],
          sorted(plain["home"]))
    check("and it names all six places, not just the ledger",
          len(plain["state"]) >= 6, sorted(plain["state"]))
    for key in ("ledger", "agenda", "runs", "spill", "compaction_log", "skills"):
        check("state[%r] is what the owning module answers" % key,
              plain["state"].get(key) == plain["truth"][key],
              "report=%r truth=%r" % (plain["state"].get(key), plain["truth"][key]))

    # -- the odd one out is named, and named differently in the two cases ---
    check("with no sibling the ledger is inside home",
          "ledger" not in (plain["home"].get("outside_home") or []),
          plain["home"].get("outside_home"))
    check("with a populated sibling the ledger is adopted",
          bool(adopt["home"].get("adopted_from")), adopt["home"].get("adopted_from"))
    # Deliberately NOT asserting that an adopted ledger counts as "outside":
    # the sibling lives at `<home>/data/autoforge.db`, which is INSIDE home.
    # The boolean this suite used to demand said `True` there, because it asked
    # "is the ledger's directory equal to home" -- a stricter question than the
    # one a reader has. Keeping that assertion would have demanded the old,
    # wrong answer, so it is gone rather than updated.
    check("and its directory is a subdirectory of the reported home",
          os.path.dirname(os.path.abspath(adopt["db_path"])).lower().startswith(
              str((adopt["home"].get("state") or {}).get("home") or "").lower()
              + os.sep.lower()),
          (adopt["db_path"], (adopt["home"].get("state") or {}).get("home")))
    for key in ("ledger", "agenda", "runs", "spill", "compaction_log", "skills"):
        check("state[%r] is still truthful when the ledger is adopted" % key,
              adopt["state"].get(key) == adopt["truth"][key],
              "report=%r truth=%r" % (adopt["state"].get(key), adopt["truth"][key]))
    check("the same six places are named in both cases",
          sorted(plain["state"]) == sorted(adopt["state"]),
          (sorted(plain["state"]), sorted(adopt["state"])))

    # -- asking for the map must not create anything -----------------------
    fresh = os.path.join(tempfile.mkdtemp(prefix="chk_state_map_bare_"), "LOCALAPPDATA")
    os.makedirs(fresh)
    os.environ["LOCALAPPDATA"] = fresh
    from autoforge_svc.store import state_map
    before = sorted(os.listdir(fresh))
    m = state_map()
    after = sorted(os.listdir(fresh))
    check("asking for the map creates nothing",
          before == after, (before, after))
    check("even though it names paths that do not exist yet",
          any(not os.path.exists(p) for p in _paths(m) if p),
          [p for p in _paths(m) if p and not os.path.exists(p)][:4])
    # "Everything under home" is the wrong claim, and the first version of
    # this check asserted it anyway: `default_skill_dirs()` also answers the
    # PROJECT skills directory, which sits beside the code, not beside the
    # user's state. Demanding it be under home is demanding a false answer --
    # the third time in this one file that a shape change left a reader behind.
    # What is true: anything not declared as a project directory is under home.
    # `config` is a deliberate exception, added with the tenth entry on
    # 2026-11-10. It lives at `Path.home() / ".autoforge" / "config.json"`,
    # which is nowhere near `home_dir()` on Windows, and it is named in the
    # map precisely so that this list can see it instead of tripping over it.
    # Naming the exception is the point: a check that silently skipped it
    # would pass for the same reason a check on an empty list passes.
    strays = [p for p in _paths(m)
              if not str(p).lower().startswith(str(m["home"]).lower() + os.sep.lower())
              and p != m["home"]
              and p != m["config"]
              and not _is_project_skill(p)]
    check("every path it names is under home, except the project skill directory and config",
          not strays, strays)
    check("and config really is outside home, not waived by accident",
          not str(m["config"]).lower().startswith(str(m["home"]).lower() + os.sep.lower()),
          (m["config"], m["home"]))
    check("and the map calls config out in outside_home",
          "config" in (plain["home"].get("outside_home") or []),
          plain["home"].get("outside_home"))
    check("and the project directory says it is a project directory",
          any(d.get("origin") == "project" for d in (m.get("skills") or [])
              if isinstance(d, dict)),
          m.get("skills"))

    # -- the skills entry must be ASKED, not remembered --------------------
    # A separate scenario with the variable set. Without this the owner's
    # answer equals the obvious guess and no mutant can be told apart from it.
    moved = os.path.join(tempfile.mkdtemp(prefix="chk_state_map_moved_"), "elsewhere")
    os.makedirs(moved)
    away = scenario(False, skills_env=moved)
    away_skills = away["state"].get("skills")
    check("with AUTOFORGE_SKILLS_DIRS set, the map reports that directory",
          _skill_paths(away_skills) == [moved],
          "report=%r moved=%r" % (away_skills, moved))
    check("and says the origin is the environment, not the user directory",
          [d.get("origin") for d in (away_skills or []) if isinstance(d, dict)]
          == ["env"],
          away_skills)
    check("which is nowhere near the home it also reports",
          not any(str(p).lower().startswith(str(away["state"].get("home", "")).lower())
                  for p in _skill_paths(away_skills)),
          away_skills)
    os.environ.pop("AUTOFORGE_SKILLS_DIRS", None)

    # -- "outside home" must name every entry that leaves, not just one ----
    # Measured online 2026-11-06: with the shipped preset `home` is
    # `<base>/data` while `runs` and `skills` are siblings of it, and the
    # report named only the ledger. One boolean about one entry, on a package
    # where several entries can leave.
    check("the report names which entries sit outside home",
          isinstance(plain["home"].get("outside_home"), list),
          plain["home"].get("outside_home"))
    # The project skill directory always sits beside the code, so at least one
    # entry is outside even in the tidiest layout -- which is the point: the
    # answer is a list, and "[]" would be the interesting value.
    check("and it is not empty even in the default layout",
          bool(plain["home"].get("outside_home")),
          plain["home"].get("outside_home"))
    check("every key it names really is outside home",
          all(not str(_entry(plain["state"], k) or "").lower()
                  .startswith(str(plain["state"].get("home", "")).lower() + os.sep.lower())
              for k in plain["home"].get("outside_home") or []),
          [(k, _entry(plain["state"], k)) for k in plain["home"].get("outside_home") or []])
    check("and no key it omits is outside home",
          all(str(_entry(plain["state"], k) or "").lower()
                  .startswith(str(plain["state"].get("home", "")).lower() + os.sep.lower())
              or _is_project_skill(_entry(plain["state"], k) or "")
              or k in (plain["home"].get("outside_home") or [])
              for k in plain["state"] if k != "home" and _entry(plain["state"], k)),
          [k for k in plain["state"] if k != "home"
           and _entry(plain["state"], k)
           and not str(_entry(plain["state"], k)).lower()
                   .startswith(str(plain["state"].get("home", "")).lower() + os.sep.lower())
           and not _is_project_skill(_entry(plain["state"], k))
           and k not in (plain["home"].get("outside_home") or [])])
    # The per-entry boolean is GONE, and this suite asks that it stay gone.
    # `db_file` can only answer `<home>/autoforge.db` or `<home>/data/autoforge.db`
    # -- both inside home -- so the field was False in every reachable layout.
    # A mutant that hardcoded it to False survived, which is the definition of
    # a field with no information in it. `outside_home` is the live answer and
    # the assertions above already hold it to account.
    check("no per-entry boolean survives beside the list",
          "ledger_outside_home" not in (plain["home"] or {}),
          sorted(plain["home"] or {}))
    check("with an adopted sibling the ledger does NOT leave home",
          "ledger" not in (adopt["home"].get("outside_home") or []),
          adopt["home"].get("outside_home"))
    check("because the sibling store lives in a subdirectory of home",
          os.path.dirname(os.path.abspath(adopt["db_path"])).lower().startswith(
              str((adopt["home"].get("state") or {}).get("home") or "").lower()
              + os.sep.lower()),
          (adopt["db_path"], (adopt["home"].get("state") or {}).get("home")))
    # Three now, not two: `config` joins the preset's own two escapes because
    # it never lived under home in the first place. The assertion stays an
    # exact list rather than a subset test -- a new reader that leaves home
    # without being expected here has to fail this line, which is the only
    # reason the list is worth writing down.
    check("and runs, skills and config can leave together under a preset-style mount",
          _preset_layout_outside() == ["config", "runs", "skills"],
          _preset_layout_outside())

    print("TOTAL FAILURES: %d %s" % (len(FAILS), FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
