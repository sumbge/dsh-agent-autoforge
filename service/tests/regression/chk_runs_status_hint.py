"""`runs_status` says what is past its TTL. It does not delete anything.

The decision, recorded as a test rather than as a paragraph:

  `runs.prune` is reachable and stays manual. Nothing calls it on a schedule.
  Upstream's CLI offers `runs list|resume|drop|prune` with no timer either.
  A record removed by a machine nobody was watching cannot be argued about
  afterwards -- and telling a repeated failure from a task nobody ever started
  is the only thing these records are for.

What the reader gets instead is a count and a name: `past_ttl`, `ttl_days`,
and a `hint` sentence that says what would remove them. A hint that deletes is
not a hint, so this file also asserts the opposite: after calling
`runs_status`, every file is still on disk.

Discriminating design. Measured 2026-11-05, before the change, on a journal
holding one 30-day-old record and one fresh one:
    runs_status -> {"count": 2, "resumable": [...], and no field at all about age}
An operator had to compute `time.time() - updated_at` themselves, or run
`prune` and find out by whether the record survived. So:
  H1 (old): no `past_ttl` key, or a `past_ttl` that never names the old record
  H2 (new): `past_ttl == [the old one]`, `ttl_days == 7.0`, `hint` names
      `runs_prune`, and the file is still there afterwards
"""
import os
import sys
import json
import time
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


def plant(root, run_id, _, age_days, task="planted by chk_runs_status_hint"):
    """Write one record `age_days` old, in the shape `runs.Mark` writes."""
    from autoforge_svc import runs as R
    now = time.time()
    stamp = now - age_days * 86400.0
    mark = R.Mark(run_id=run_id, task=task, cwd=os.getcwd(), session="probe",
                  started_at=stamp, updated_at=stamp, writer_pid=999999,
                  turns=1, messages=[], schema=R.SCHEMA)
    path = R._path(run_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    R._atomic_write(path, json.dumps(mark.to_dict(), ensure_ascii=False))
    return path


def main():
    os.environ["AUTOFORGE_HOME"] = tempfile.mkdtemp(prefix="chk_runs_hint_")
    from autoforge_svc import runs as R
    from autoforge_svc.service import Service

    root = R.journal_dir()
    root.mkdir(parents=True, exist_ok=True)

    old = plant(root, "oldrecord01", 0, 30.0, "a month old")
    fresh = plant(root, "newrecord02", 0, 0.001, "just written")

    svc = Service()
    out = svc.runs_status()

    check("runs_status names the TTL it is judging against",
          out.get("ttl_days") == 7.0, out.get("ttl_days"))
    check("it names the records past that TTL",
          out.get("past_ttl") == ["oldrecord01"], out.get("past_ttl"))
    check("and it does not name the fresh one",
          "newrecord02" not in (out.get("past_ttl") or []), out.get("past_ttl"))
    hint = out.get("hint") or ""
    check("the hint says what would remove them", "runs_prune" in hint, hint)
    check("the hint says nothing does it on its own",
          "on its own" in hint or "on their own" in hint, hint)

    # -- the part that matters: reading did not delete --------------------
    check("reading the status deleted nothing", old.exists() and fresh.exists(),
          "old=%s fresh=%s" % (old.exists(), fresh.exists()))
    again = svc.runs_status()
    check("and a second read still sees both",
          again.get("count") == 2 and again.get("past_ttl") == ["oldrecord01"],
          "%s / %s" % (again.get("count"), again.get("past_ttl")))

    # -- empty journal: no hint, no noise --------------------------------
    empty_home = tempfile.mkdtemp(prefix="chk_runs_hint_empty_")
    real_home = os.environ["AUTOFORGE_HOME"]
    os.environ["AUTOFORGE_HOME"] = empty_home
    try:
        svc2 = Service()
        blank = svc2.runs_status()
        check("an empty journal reports no records", blank.get("count") == 0, blank.get("count"))
        check("and says nothing about a TTL it has no records for",
              blank.get("hint") == "" and blank.get("past_ttl") == [],
              "%r / %r" % (blank.get("hint"), blank.get("past_ttl")))
    finally:
        os.environ["AUTOFORGE_HOME"] = real_home

    # -- the count is not a second walk ---------------------------------
    # `past_ttl` is derived from the rows `scan()` already returned. If a later
    # edit re-walks the directory, entries that vanished between the two walks
    # make `count` and the sum disagree. Assert the arithmetic holds.
    check("past_ttl is a subset of the rows that were returned",
          set(out.get("past_ttl") or []) <= {r["run_id"] for r in out.get("runs") or []},
          "%s vs %s" % (out.get("past_ttl"),
                        [r["run_id"] for r in out.get("runs") or []]))
    ages = {r["run_id"]: r["age_s"] for r in out.get("runs") or []}
    check("and the threshold it applied is the one it reported",
          all((ages[rid] > out.get("ttl_days", 0) * 86400.0)
              for rid in out.get("past_ttl") or []),
          ages)

    print("TOTAL FAILURES: %d %s" % (len(FAILS), FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
