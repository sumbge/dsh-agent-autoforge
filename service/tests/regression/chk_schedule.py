"""The four schedule methods, held to what they answer.

Written from measurement, not from the docstrings. Everything asserted here was
first read off a live service in a child process on 2026-11-06, because a test
that encodes what I *think* the shape is just moves the guessing.

What the probes found:

  schedule_list   -> {"report": <text>}          (always, empty or not)
  schedule_tick   -> {"due": [...], "count": n}  (due rows carry overdue_by)
  schedule_add    -> {"id","text","due_at","repeat_seconds","line"}
  schedule_run_due-> {"due","ran","failed","results","elapsed_s"}
                     ... and {"due","ran","failed","results","note"} when idle
  refusals        -> transport `ok: false`, error text on the envelope

That last line is the one this suite exists for. Before 2026-11-06 the three
mutating methods answered `{"error": ...}` inside a reply whose envelope said
`ok: true`, so a machine reading the envelope learned that cancelling a task
that does not exist had succeeded.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

# Bootstrap the package from this file's own location. It used to rely on the
# caller exporting PYTHONPATH, which meant a fresh clone failed here with
# `ModuleNotFoundError: No module named 'autoforge_svc'` -- a message about the
# code when the problem was the environment. Measured 2026-11-11 on a copy of
# this tree in a temp directory with PYTHONPATH unset.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import autoforge_svc

# Derived, not hardcoded. The mutation driver copies the tree, edits the copy,
# points PYTHONPATH at it and runs this file -- so a hardcoded path would have
# this suite spawn children against the REAL code while the driver believed it
# was testing a mutant. Measured: the first version hardcoded the plugin path
# and four of eight mutants escaped with `fails=0`, because the wire half of
# this suite was talking to the healthy service the whole time.
SERVICE = os.path.dirname(os.path.dirname(os.path.abspath(autoforge_svc.__file__)))
# `sys.executable` is set whenever python runs a script; a literal fallback
# would only ever fire in a case that cannot happen, and would name a machine.
PY = sys.executable
if not PY:
    raise SystemExit("cannot locate the interpreter running this suite")

FAILED: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print("OK   %s" % label)
    else:
        print("FAIL %s <- %s" % (label, detail))
        FAILED.append(label)


def wire(home: str, requests: list[dict], *, timeout: int = 240) -> list[dict]:
    env = dict(os.environ)
    env["PYTHONPATH"] = SERVICE
    env["AUTOFORGE_HOME"] = home
    env.pop("AUTOFORGE_RUNS", None)
    env.pop("AUTOFORGE_SKILLS_DIRS", None)
    body = "\n".join(json.dumps(r) for r in requests) + "\n"
    proc = subprocess.run(
        [PY, "-P", "-m", "autoforge_svc"], input=body, capture_output=True,
        text=True, encoding="utf-8", errors="replace", env=env, cwd=SERVICE,
        timeout=timeout,
    )
    out = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> int:
    home = tempfile.mkdtemp(prefix="chk_schedule_")
    reqs = [
        {"id": 1, "method": "schedule_list", "params": {}},          # empty agenda
        {"id": 2, "method": "schedule_tick", "params": {}},
        {"id": 3, "method": "schedule_add", "params": {"text": "Reply with exactly: ok", "when": "+1h"}},
        {"id": 4, "method": "schedule_add", "params": {"text": "second", "when": "+2s"}},
        {"id": 5, "method": "schedule_list", "params": {}},
        {"id": 6, "method": "schedule_tick", "params": {}},          # one is due by now? (no: +2s)
        {"id": 7, "method": "schedule_add", "params": {"text": "bad", "when": "not-a-time"}},
        {"id": 8, "method": "schedule_cancel", "params": {"task_id": "no-such-task"}},
        {"id": 9, "method": "schedule_done", "params": {"task_id": "no-such-task", "note": "x"}},
    ]
    got = {r["id"]: r for r in wire(home, reqs)}

    # -- shapes ---------------------------------------------------------
    check("an empty agenda still answers a report",
          set(got[1]["result"]) == {"report"}, sorted(got[1]["result"]))
    check("and the report says so in words",
          "Nothing scheduled" in got[1]["result"]["report"],
          got[1]["result"]["report"][:80])

    tick = got[2]["result"]
    check("schedule_tick answers {due, count}", set(tick) == {"due", "count"}, sorted(tick))
    check("with nothing due, count is zero and due is empty",
          tick["count"] == 0 and tick["due"] == [], tick)

    add = got[3]["result"]
    check("schedule_add answers the five keys it promises",
          set(add) == {"id", "text", "due_at", "repeat_seconds", "line"}, sorted(add))
    check("the id is the one the line quotes",
          isinstance(add["id"], str) and add["id"] in add["line"], (add["id"], add["line"]))
    check("a one-shot task has no repeat interval", add["repeat_seconds"] == 0.0, add["repeat_seconds"])

    listing = got[5]["result"]["report"]
    check("the agenda now names the task it holds", add["id"] in listing, listing[:120])
    second_id = got[4]["result"]["id"]
    check("and so does the second one", second_id in listing, listing[:120])

    # -- refusals are refusals -----------------------------------------
    bad_when = got[7]
    check("a bad `when` is refused at the envelope",
          bad_when.get("ok") is False, bad_when)
    check("and the reason names the error class",
          "ScheduleError" in (bad_when.get("error") or ""), bad_when.get("error"))
    check("a refused add carries no result at all",
          "result" not in bad_when, sorted(bad_when))

    cancel = got[8]
    check("cancelling an unknown id is refused at the envelope",
          cancel.get("ok") is False, cancel)
    check("and it does not smuggle an `error` field into a result",
          "result" not in cancel, sorted(cancel))
    done = got[9]
    check("recording a run on an unknown id is refused the same way",
          done.get("ok") is False, done)
    check("and that refusal also names the class",
          "ScheduleError" in (done.get("error") or ""), done.get("error"))

    # -- the two verdicts cannot be confused ----------------------------
    # The defect this suite was written for: `{"ok": true, "result":
    # {"error": ...}}` -- a successful envelope around a refused operation.
    # A caller that only reads the envelope must never see the two the same.
    ok_shapes = [r for r in (got[3], got[5]) if r.get("ok") is True]
    bad_shapes = [r for r in (got[7], got[8], got[9]) if r.get("ok") is False]
    check("a success and a refusal never share an envelope status",
          len(ok_shapes) == 2 and len(bad_shapes) == 3,
          (len(ok_shapes), len(bad_shapes)))
    check("no successful reply carries a top-level result.error",
          not any("error" in (r.get("result") or {}) for r in ok_shapes),
          [r.get("result") for r in ok_shapes])

    # -- a real run, with the model stubbed out ------------------------
    # Deliberately NOT through the wire. A fresh child process has an empty
    # `_TOOL_READING`, so `_tool_capable_model` probes up to five candidates at
    # 90s each before running anything -- measured: the first attempt at this
    # section timed out at 300s with no task run. That cost belongs to
    # `chk_route.py`, which already measures it; what this suite is about is
    # whether a due task is picked up, run, recorded and retired.
    #
    # So `_run_task` is replaced in-process with a stub that returns a known
    # `TaskOutcome`. Everything else -- `due()`, the budget arithmetic, the row
    # shape, `complete()`, the agenda afterwards -- is the real code.
    import autoforge_svc.service as S

    # Two separate settings, and getting either wrong is silently a different
    # test: `Service(db_path)` takes a *file*, while the agenda is found by
    # `Schedule()` through `$AUTOFORGE_HOME`. Passing the directory to the
    # constructor raised `sqlite3.OperationalError: unable to open database
    # file`, and leaving the environment alone would have written to the real
    # agenda on this machine -- which is what the previous version of this
    # section did before it timed out.
    home2 = tempfile.mkdtemp(prefix="chk_schedule_run_")
    real_home_env = os.environ.get("AUTOFORGE_HOME")
    os.environ["AUTOFORGE_HOME"] = home2
    try:
        svc = S.Service(os.path.join(home2, "autoforge.db"))
        task = svc.schedule_add("Reply with exactly: ok", "+1s")
        task_id = task["id"]
        time.sleep(1.5)

        tick2 = svc.schedule_tick()
        check("the task that is now due shows up in tick",
              tick2["count"] == 1 and tick2["due"][0]["id"] == task_id, tick2)
        check("and a due row carries overdue_by as a number",
              isinstance(tick2["due"][0]["overdue_by"], float), tick2["due"][0])

        real_run_task = S.Service._run_task
        calls: list[str] = []

        def fake_run_task(self, text, timeout_s=600.0):  # noqa: ANN001
            calls.append(text)
            return S.TaskOutcome("ok", ok=True, turns=1)

        S.Service._run_task = fake_run_task
        try:
            run = svc.schedule_run_due(budget_s=120)
        finally:
            S.Service._run_task = real_run_task

        check("schedule_run_due answers the five keys",
              set(run) == {"due", "ran", "failed", "results", "elapsed_s"}, sorted(run))
        check("it asked the runner to do exactly the one due task",
              calls == ["Reply with exactly: ok"], calls)
        check("and it claims to have run exactly that one",
              run["due"] == 1 and run["ran"] == 1 and run["failed"] == 0, run)
        check("the row carries the outcome fields, not just a sentence",
              isinstance(run["results"], list) and run["results"]
              and {"ok", "note", "id", "turns"} <= set(run["results"][0]),
              run["results"])

        after = svc.schedule_list()["report"]
        check("a one-shot task leaves the agenda once it has run",
              task_id not in after, after[:160])
        shown = svc._scheduler()._tasks.get(task_id)
        check("and it is recorded as having run once, without a failure",
              shown is not None and shown.runs == 1 and shown.failures == 0,
              None if shown is None else (shown.runs, shown.failures))

        # -- a failing run is counted as a failure ---------------------
        home3 = tempfile.mkdtemp(prefix="chk_schedule_fail_")
        os.environ["AUTOFORGE_HOME"] = home3
        svc2 = S.Service(os.path.join(home3, "autoforge.db"))
        fail_id = svc2.schedule_add("do the impossible", "+1s")["id"]
        time.sleep(1.5)

        def fake_fail(self, text, timeout_s=600.0):  # noqa: ANN001
            return S.TaskOutcome("(turn cap of 12 reached before completion)",
                                 ok=False, hit_turn_cap=True, turns=12)

        S.Service._run_task = fake_fail
        try:
            bad = svc2.schedule_run_due(budget_s=120)
        finally:
            S.Service._run_task = real_run_task
        check("a run that did not finish is counted as failed",
              bad["failed"] == 1 and bad["ran"] == 0, bad)
        # The defect `TaskOutcome` was created for: `ok` used to be True
        # whenever nothing raised, so the cap apology went into free text and
        # `failures` stayed at zero for a task that had never finished.
        stale = svc2._scheduler()._tasks.get(fail_id)
        check("and the task's own failure count moved",
              stale is not None and stale.failures == 1,
              None if stale is None else stale.failures)
        check("with the reason recorded where the next run will read it",
              stale is not None and "turn cap" in (stale.notes[-1] if stale.notes else ""),
              None if stale is None else stale.notes)
    finally:
        if real_home_env is None:
            os.environ.pop("AUTOFORGE_HOME", None)
        else:
            os.environ["AUTOFORGE_HOME"] = real_home_env

    # -- idle is not a failure ------------------------------------------
    home3 = tempfile.mkdtemp(prefix="chk_schedule_idle_")
    idle = wire(home3, [{"id": 1, "method": "schedule_run_due", "params": {"budget_s": 10}}])[0]
    check("nothing due is a successful answer, not a refusal",
          idle.get("ok") is True, idle)
    check("and it says so with a note instead of pretending to work",
          idle["result"].get("note") == "nothing was due"
          and idle["result"]["ran"] == 0, idle["result"])

    print()
    if FAILED:
        print("TOTAL FAILURES: %d %s" % (len(FAILED), FAILED))
        return 1
    print("TOTAL FAILURES: 0 []")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
