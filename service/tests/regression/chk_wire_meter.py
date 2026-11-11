"""The meter that says which wire methods have actually been pressed.

Why this suite exists
---------------------
Measured 2026-11-06: 40 of the 76 methods in the dispatch table had no suite
that called them, and the ledger could not answer the question either --
`kind='call'` records forged-tool invocations, never wire methods. So "has this
edge of the surface ever run" had no source at all, and a missing test looked
exactly like an unused feature.

Three design claims this suite holds to account, each of which could be got
wrong in a way that still passes a smoke test:

  1. The meter lives at `main()` -- the one door -- and NOT in `_call_method`.
     If it lived one level down, this repository's own tests (which call
     `svc._call_method(...)` directly, precisely to skip the wire) would be
     counted as production traffic, and the number would be a lie about use.
  2. The meter is a counter, not evidence. It must not write to the hash chain:
     traffic is not a decision, and putting it on the chain would make the
     anchor cadence fire on requests.
  3. `never` is computed from the dispatch table, not from the counter. A
     counter cannot know what it has not seen; only `Service` knows the whole
     list, so `pressed + len(never) == declared` has to hold in one report.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICE = os.path.dirname(os.path.dirname(HERE))
# Same directory, put on the path, because `fresh()` imports the package. The
# `env["PYTHONPATH"]` further down only reaches the child processes.
sys.path.insert(0, SERVICE)
PY = sys.executable

FAILS = []


def check(label, cond, detail=""):
    if cond:
        print("OK   %s" % label)
    else:
        FAILS.append(label)
        print("FAIL %s  <- %s" % (label, detail))


def fresh(home):
    os.environ["AUTOFORGE_HOME"] = home
    for var in ("AUTOFORGE_RUNS", "AUTOFORGE_SKILLS_DIRS"):
        os.environ.pop(var, None)
    import autoforge_svc.service as S
    from autoforge_svc.store import ToolStore
    return S, ToolStore, S.Service(db_path=os.path.join(home, "autoforge.db"))


def run_wire(home, requests, prelude=""):
    """Drive a real `main()` over stdin, like the bridge does.

    A fresh import in this process would prove the table exists; only a
    subprocess with a request written to its stdin proves the meter runs.

    `prelude` is executed before `main()` is called, which is how a broken-meter
    scenario is set up. It exists because renaming the table in the database
    file does not work: the subprocess's own `Service.__init__` runs the DDL,
    and `CREATE TABLE IF NOT EXISTS wire_calls` puts it straight back. Measured
    2026-11-06 -- the mutant that makes the failure branch raise survived that
    fixture with exit=0, because the branch never ran.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = SERVICE
    env["PYTHONIOENCODING"] = "utf-8"
    env["AUTOFORGE_HOME"] = home
    for var in ("AUTOFORGE_RUNS", "AUTOFORGE_SKILLS_DIRS"):
        env.pop(var, None)
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    if prelude:
        driver = os.path.join(home, "_drive.py")
        with open(driver, "w", encoding="utf-8") as fh:
            fh.write("import sys\n"
                     "sys.path.insert(0, %r)\n" % SERVICE
                     + prelude
                     + "\nfrom autoforge_svc.service import main\n"
                       "raise SystemExit(main([]))\n")
        argv = [PY, "-P", driver]
    else:
        argv = [PY, "-P", "-m", "autoforge_svc"]
    p = subprocess.run(argv, input=payload, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env,
                       cwd=SERVICE, timeout=180)
    replies = []
    for line in (p.stdout or "").splitlines():
        try:
            replies.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return p.returncode, replies, (p.stderr or "")[-400:]


def main():
    tmp = tempfile.mkdtemp(prefix="chk_wire_meter_")
    home = os.path.join(tmp, "home")
    os.makedirs(home)
    S, ToolStore, svc = fresh(home)

    # -- 1. the table is declared, not "undeclared" ------------------------
    view = svc.store.table_view()
    check("wire_calls is a table this store created",
          "wire_calls" in view, sorted(view))
    entry = view.get("wire_calls") or {}
    check("the store declares what its zero means, rather than inheriting a verdict",
          entry.get("verdict") == "never-triggered",
          entry)
    check("and says what it is for",
          "wire method" in (entry.get("role") or ""), entry.get("role"))

    # -- 2. a fresh store reports every declared method as never pressed ---
    rep = svc.report()
    wc = rep.get("wire_calls") or {}
    check("report() carries the meter", "never" in wc, wc)
    check("nothing has been pressed yet", wc.get("pressed") == 0, wc.get("pressed"))
    check("every declared method is listed as never pressed",
          sorted(wc.get("never") or []) == sorted(svc.methods),
          (len(wc.get("never") or []), len(svc.methods)))
    check("pressed + never adds up to declared -- the complement is computed, not guessed",
          wc.get("pressed", -1) + len(wc.get("never") or []) == wc.get("declared"),
          (wc.get("pressed"), len(wc.get("never") or []), wc.get("declared")))
    check("and nothing arrived under a name this build does not have",
          wc.get("unknown") == [], wc.get("unknown"))

    # -- 3. counting a call, and counting a failure, are two numbers -------
    svc.store.record_wire_call("report", ok=True, ms=2.0)
    svc.store.record_wire_call("report", ok=False, ms=6.0)
    v = svc.store.wire_call_view()["pressed"]["report"]
    check("two calls were counted", v["calls"] == 2, v)
    check("one of them failed", v["failures"] == 1, v)
    check("total_ms accumulates", abs(v["total_ms"] - 8.0) < 1e-6, v["total_ms"])
    check("last_ms remembers only the latest", abs(v["last_ms"] - 6.0) < 1e-6, v["last_ms"])

    # -- 4. an unknown name is remembered UNDER ITS OWN NAME --------------
    # This is the `anchor_ledger` defect class: a method registered in the
    # bridge but absent from the dispatch table answered `unknown method`, and
    # nothing kept the name afterwards.
    svc.store.record_wire_call("never_heard_of_this", ok=False, ms=0.1, known=False)
    wc2 = svc.report()["wire_calls"]
    check("an unknown method is remembered by name",
          "never_heard_of_this" in (wc2.get("unknown") or []), wc2.get("unknown"))
    check("and it does not inflate the pressed count",
          wc2.get("pressed") == 1, wc2.get("pressed"))
    check("and it is not counted among the declared-but-unpressed",
          "never_heard_of_this" not in (wc2.get("never") or []), wc2.get("unknown"))

    # -- 5. the meter is NOT on the ledger chain --------------------------
    kinds = {r["kind"] for r in svc.store.get_events(200)}
    check("counting a call writes nothing to the hash chain",
          not any(k.startswith("wire_") for k in kinds), sorted(kinds))

    # -- 6. the real door increments; the bypassing door does not ---------
    tmp2 = tempfile.mkdtemp(prefix="chk_wire_meter_wire_")
    rc, replies, err = run_wire(tmp2, [
        {"id": 1, "method": "ping", "params": {}},
        {"id": 2, "method": "no_such_method_at_all", "params": {}},
    ])
    check("the service answered both requests over stdio", rc == 0 and len(replies) == 2,
          (rc, len(replies), err))
    S2, _TS2, svc2 = fresh(tmp2)
    wc3 = svc2.report()["wire_calls"]
    check("a request that came through main() is counted",
          (svc2.store.wire_call_view()["pressed"].get("ping", {}) or {}).get("calls") == 1,
          svc2.store.wire_call_view()["pressed"])
    check("a request for a method this build does not have is counted too",
          "no_such_method_at_all" in (wc3.get("unknown") or []), wc3.get("unknown"))
    check("and the unknown one is marked as not known",
          (svc2.store.wire_call_view()["unknown"].get("no_such_method_at_all") or {}
           ).get("known") is False,
          svc2.store.wire_call_view()["unknown"])

    before = dict(svc2.store.wire_call_view()["pressed"])
    svc2._call_method("report", {})
    after = svc2.store.wire_call_view()["pressed"]
    check("calling _call_method directly is NOT counted -- the meter is at the door",
          after == before, (sorted(after), sorted(before)))

    # -- 7. the meter reports its own failures ----------------------------
    check("report() carries the meter's own failure count",
          "meter_failures" in wc3, sorted(wc3))
    check("and it is a number, so a stop in counting is visible",
          isinstance(wc3.get("meter_failures"), int), wc3.get("meter_failures"))

    # -- 8. a store that cannot count does not fail the request -----------
    S3, ToolStore3, svc3 = fresh(tempfile.mkdtemp(prefix="chk_wire_meter_broken_"))
    svc3.store._conn.execute("ALTER TABLE wire_calls RENAME TO wire_calls_gone")
    svc3.store._conn.commit()
    broken = svc3.report().get("wire_calls") or {}
    check("a broken meter is reported as an error, not raised at the caller",
          "error" in broken, broken)
    check("and the rest of the report still arrives",
          "tools" in svc3.report(), sorted(svc3.report()))

    # 8b. The design rule that matters most: a counter must never break a
    # finished call. Broken with a patched method rather than a renamed table,
    # because `Service.__init__` runs `CREATE TABLE IF NOT EXISTS wire_calls`
    # and would silently rebuild what the fixture removed -- measured
    # 2026-11-06: the mutant that turns this `except` into a `raise` survived a
    # table-renaming fixture with exit=0, because the branch never ran.
    tmp3 = tempfile.mkdtemp(prefix="chk_wire_meter_break_")
    prelude = (
        "from autoforge_svc.store import ToolStore as _T\n"
        "def _boom(self, method, *, ok, ms, known=True):\n"
        "    raise RuntimeError('the meter is broken on purpose')\n"
        "_T.record_wire_call = _boom\n"
    )
    rc3, replies3, err3 = run_wire(tmp3, [{"id": 1, "method": "ping", "params": {}}],
                                   prelude=prelude)
    check("a request still answers when the meter cannot write",
          rc3 == 0 and len(replies3) == 1 and replies3[0].get("ok") is True,
          (rc3, replies3, err3))
    check("the caller is not told about the meter's problem",
          "broken on purpose" not in json.dumps(replies3)
          and "RuntimeError" not in json.dumps(replies3), replies3)

    print("TOTAL FAILURES: %d %s" % (len(FAILS), FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
