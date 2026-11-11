"""Every "where does state live" answer must come from one place.

Measured 2026-11-05, before the fix, with no AUTOFORGE_HOME set:

    store._default_home()              <LOCALAPPDATA>/autoforge
    runs.journal_dir()                 <LOCALAPPDATA>/autoforge/runs    (delegates)
    skills user dir                    <LOCALAPPDATA>/autoforge/skills  (delegates)
    service._home()                    <LOCALAPPDATA>/autoforge
    schedule.default_schedule_path()   ~/.autoforge/schedule.jsonl      <-- its own rule
    compaction.spill_dir()             ~/.autoforge/spill               <-- its own rule
    compaction.default_log_path()      ~/.autoforge/compaction.log.md   <-- its own rule
    service._peer_market_from_file()   ~/.autoforge/peers.json          <-- its own rule

and the consequence was on disk: two schedule files, 868 bytes and 2,667 bytes,
holding disjoint sets of tasks. A task added by the CLI was invisible to the
service and the reverse.

Why "all seven agree" is not enough as an assertion. They agreed in pairs
before the fix too -- four said one thing, five said another, and every single
one of them was self-consistent. A test that only compares them would have
passed on the broken tree as long as all seven were broken the same way. So this
file checks three separate things:

  1. they agree with each other (the symptom),
  2. each one agrees with `store.home_dir()` (the authority),
  3. the source of each contains no second fallback of its own (the cause) --
     i.e. the body names `home_dir` and mentions neither `LOCALAPPDATA` nor
     `expanduser("~")` nor a hand-built `.autoforge` path.

Point 3 is checked on COMMENTS-STRIPPED source via `ast`, because the docstrings
explaining this change quote the old code. That is the same mistake as the
`chk_ledger.py` citation: searching prose for a string and finding your own
explanation of it. Ask the parser, not the text.
"""
import ast
import inspect
import os
import sys

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)

FAILS = []


def check(label, cond, detail=""):
    if cond:
        print("OK   %s" % label)
    else:
        FAILS.append(label)
        print("FAIL %s  <- %s" % (label, detail))


def norm(p):
    return os.path.normcase(os.path.normpath(str(p)))


def body_source(fn):
    """Function body with docstrings and comments gone."""
    src = inspect.getsource(fn)
    tree = ast.parse(_dedent(src))
    # Drop the docstring so a docstring quoting the old rule cannot fail us.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _dedent(src):
    import textwrap
    return textwrap.dedent(src)


def main():
    os.environ.pop("AUTOFORGE_HOME", None)
    os.environ.pop("AUTOFORGE_RUNS", None)

    from autoforge_svc import runs, schedule, service as svc, skills, store
    from autoforge_svc.core import compaction

    base = store.home_dir()

    # -- 1 + 2: the answers -------------------------------------------------
    # Each entry is the HOME that answer implies, not the leaf it returns:
    # `spill_dir()` is `<home>/spill` and the skills dir is `<home>/skills`, so
    # comparing the full path against `<home>` would be a test demanding a wrong
    # answer. That mistake cost four assertions the first time this file ran.
    user_dirs = [p for src, p in skills.default_skill_dirs() if src == "user"]
    answers = {
        "store.home_dir": store.home_dir(),
        "service._home": svc._home(),
        "schedule.default_schedule_path": os.path.dirname(str(schedule.default_schedule_path())),
        "compaction.spill_dir": os.path.dirname(compaction.spill_dir()),
        "compaction.default_log_path": os.path.dirname(compaction.default_log_path()),
        "runs.journal_dir": os.path.dirname(runs.journal_dir()),
        "skills user dir": os.path.dirname(user_dirs[0]) if user_dirs else "",
    }

    for name, value in answers.items():
        check("%s answers with the store's directory" % name,
              norm(value) == norm(base), "got %r, store says %r" % (value, base))

    distinct = {norm(v) for v in answers.values()}
    check("all seven answers are one directory, not two",
          len(distinct) == 1, "distinct answers: %r" % (sorted(distinct),))

    # -- 3: nobody re-derives it -------------------------------------------
    CALLERS = [
        ("service._home", svc._home),
        ("schedule.default_schedule_path", schedule.default_schedule_path),
        ("compaction.spill_dir", compaction.spill_dir),
        ("compaction.default_log_path", compaction.default_log_path),
    ]
    for name, fn in CALLERS:
        body = body_source(fn)
        check("%s calls home_dir()" % name, "home_dir()" in body, body[:220])
        for forbidden in ("LOCALAPPDATA", "expanduser", "Path.home", '"autoforge"', "'autoforge'"):
            check("%s does not re-derive its own fallback (%s)" % (name, forbidden),
                  forbidden not in body, body[:220])

    # `_peer_market_from_file` reaches the directory through a local import, so
    # its body is read the same way but the module it imports from is the point.
    peers = body_source(svc.Service._peer_market_from_file)
    check("the peers lookup calls home_dir()", "home_dir()" in peers, peers[:220])
    check("and it does not build a .autoforge path itself",
          "autoforge\"" not in peers and "expanduser" not in peers, peers[:220])

    # -- the authority itself must not be a second copy ---------------------
    hi = body_source(store.home_dir)
    check("store.home_dir delegates rather than re-implementing",
          "_default_home()" in hi and "LOCALAPPDATA" not in hi, hi[:220])
    dh = body_source(store._default_home)
    check("store._default_home is the only place the base is computed",
          "LOCALAPPDATA" in dh or "XDG_DATA_HOME" in dh, dh[:220])
    check("and it keeps the dotted fallback the shadowing incident earned",
          "\\\".autoforge\\\"" in repr(dh) or ".autoforge" in dh, dh[:220])

    # -- a fresh store lands where the seven say it will --------------------
    import tempfile
    tmp = tempfile.mkdtemp(prefix="chk_home_")
    s = store.ToolStore(db_path=os.path.join(tmp, "x.db"))
    check("a store told to use another directory still reports it",
          norm(os.path.dirname(s.db_path)) == norm(tmp), s.db_path)
    check("home_dir is unaffected by where a particular store was put",
          norm(store.home_dir()) == norm(base), store.home_dir())

    # -- set, it still wins -------------------------------------------------
    saved = os.environ.get("AUTOFORGE_HOME")
    os.environ["AUTOFORGE_HOME"] = tmp
    try:
        check("AUTOFORGE_HOME still overrides the schedule path",
              norm(os.path.dirname(str(schedule.default_schedule_path()))) == norm(tmp),
              str(schedule.default_schedule_path()))
        check("AUTOFORGE_HOME still overrides the spill directory",
              norm(os.path.dirname(compaction.spill_dir())) == norm(tmp),
              compaction.spill_dir())
        check("AUTOFORGE_HOME still overrides the service's home",
              norm(svc._home()) == norm(tmp), svc._home())
        check("and the log path follows it too",
              norm(os.path.dirname(compaction.default_log_path())) == norm(tmp),
              compaction.default_log_path())
    finally:
        if saved is None:
            os.environ.pop("AUTOFORGE_HOME", None)
        else:
            os.environ["AUTOFORGE_HOME"] = saved

    print("TOTAL FAILURES: %d %s" % (len(FAILS), FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
