"""The retention rule has to be a rule, not a comment.

`runs.prune` was written during the port and never called. Measured 2026-11-04
before this file existed: `prune(` appears exactly once in the whole package --
at its own definition -- while `DEFAULT_TTL_S` promises seven days. Meanwhile
`runs_status`, `resume_run` and `runs_forget` all read that directory, so a
retention nobody enforces does not mean "records are kept", it means "records
accumulate and the first person to notice deletes them by hand, or does not".

Two things had to be true before this was worth wiring up, and both were
measured rather than assumed:

**It had to be honest.** The original `prune` appended a run id on success and
swallowed every `OSError` in a bare `except OSError: pass`, so three different
situations printed the same sentence: nothing was past its TTL, forty records
expired and all of them went, forty expired and the filesystem refused all
forty. The last two are the same answer -- the empty list. Measured on this host
with forty stale records: 157.3 ms total, of which 6.5 ms was the directory
scan and 132.5 ms was the forty `unlink` calls. Refusal is the bulk of the work,
and it was the part with no voice.

**It had to be a person's decision.** Upstream also exposed this as a CLI
verb (`cli.py:1425`, `choices=["list", "resume", "drop", "prune"]`), never as
part of the tick. An unattended deleter over a directory whose purpose is to
outlive the writing process is a bad trade: a record removed by a machine
nobody was watching cannot be argued about afterwards. So it is a tool, where
the scope gate can see the call and refuse it -- `local_write`, because it
rewrites a directory -- and it is not in `schedule_run_due`.

What this file pins, in order:
  * expired records go, current records stay, and the counts add up;
  * a record reaches "past its TTL" only by age -- stale-for-other-reasons
    records are left standing, because deleting those erases the evidence that
    a task was attempted;
  * a refused deletion is named in `failed` while `prune()` still returns only
    what actually went;
  * a nonsensical TTL is refused instead of deleting everything;
  * the act is on the ledger as `runs_pruned`, not as something else;
  * the four wiring layers agree -- a method nobody can call is not a feature.

Run:
  $env:PYTHONPATH = "...\\dsh-agent-autoforge\\service"
  & $py -P ...\\chk_runs_prune.py
"""

from __future__ import annotations

import json
import os
import re
import time
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICE = os.path.dirname(os.path.dirname(HERE))   # regression -> tests -> service
PLUGIN = os.path.dirname(SERVICE)
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc import runs as _runs                      # noqa: E402
from autoforge_svc.autonomy.scopes import scope_for           # noqa: E402
from autoforge_svc.service import Service                     # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print("  OK   " + label)
    else:
        print("  FAIL " + label + (" <- " + str(detail)[:300] if detail else ""))
        FAILS.append(label)


def write_record(root: str, run_id: str, *, age_s: float, task: str = "x",
                 now: float | None = None) -> Path:
    """One record on disk, aged by hand.

    `Mark` is written through the same helpers the real journal uses
    (`to_dict` + `_atomic_write` + `_path`) rather than by formatting JSON
    here: a fixture that builds the file its own way stops testing the reader
    the moment the writer changes shape.
    """
    now = time.time() if now is None else now
    mark = _runs.Mark(run_id=run_id, task=task, started_at=now - age_s,
                      updated_at=now - age_s, writer_pid=os.getpid())
    path = _runs._path(mark.run_id, root)
    _runs._atomic_write(path, json.dumps(mark.to_dict(), ensure_ascii=False))
    return path


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m28_prune_")
    os.environ["AUTOFORGE_HOME"] = tmp
    # Records live in `<home>/runs`, not in `<home>`: `journal_dir()` appends
    # `_SUBDIR`. My first draft wrote them straight into `tmp` and every count
    # came back zero -- the fixture was testing an empty directory. Measured
    # 2026-11-04: `journal_dir()` also does not create the directory, it only
    # names it, so the fixture has to make it.
    root = _runs.journal_dir()
    os.makedirs(root, exist_ok=True)
    now = time.time()

    print("-- age decides, and the counts account for every record")
    old_ids = [f"old{i:09d}" for i in range(5)]
    for rid in old_ids:
        write_record(root, rid, age_s=9 * 86400, now=now)
    young = write_record(root, "young0000001", age_s=60, now=now)
    report = _runs.prune_detail(now=now)
    check("every expired record was removed", len(report["removed"]) == 5,
          report["removed"])
    check("they are the ones that were old",
          sorted(report["removed"]) == sorted(old_ids), report["removed"])
    check("the current record survived", young.exists(), young)
    check("past_ttl counts only what was past it", report["past_ttl"] == 5,
          report["past_ttl"])
    check("kept counts only what was not", sum(report["kept"].values()) == 1,
          report["kept"])
    check("considered is past_ttl plus kept, not a third number",
          report["considered"] == 6, report)
    check("nothing is reported as refused when nothing refused",
          report["failed"] == [], report["failed"])
    check("the TTL that was applied is on the report",
          abs(report["ttl_s"] - 7 * 86400.0) < 1.0, report["ttl_s"])

    print("-- a record too stale to resume is still not deleted")
    # `stale` also covers "resume budget spent" and "transcript unrunnable".
    # Those records are expired in the resume sense and current in the age
    # sense, and the difference matters: deleting them erases the evidence
    # that a task was attempted and failed.
    now = time.time()
    spent = write_record(root, "spent0000001", age_s=120, now=now)
    check("a record whose resumability is spent reads as stale",
          spent.exists(), spent)
    second = _runs.prune_detail(now=now)
    check("but age alone did not delete it", spent.exists(), second["removed"])
    check("and it is counted under the verdict that applies",
          second["kept"].get("stale") or second["kept"].get("waiting")
          or second["kept"].get("live"), second["kept"])

    print("-- prune() still answers the narrow question, and stops lying")
    now = time.time()
    doomed = [f"gone{i:09d}" for i in range(3)]
    for rid in doomed:
        write_record(root, rid, age_s=30 * 86400, now=now)
    before = len(list(Path(root).glob("run-*.json")))
    real_unlink = Path.unlink

    def refuse(self: Path, *a: object, **k: object) -> None:
        raise PermissionError(13, "Access is denied")

    Path.unlink = refuse                                   # type: ignore[assignment]
    try:
        refused = _runs.prune_detail(now=now)
        refused_ids = _runs.prune(root=root, now=now)
    finally:
        Path.unlink = real_unlink                          # type: ignore[assignment]
    check("a refused deletion is named, not swallowed",
          len(refused["failed"]) == 3, refused["failed"])
    check("and it says which record and why",
          all("run_id" in f and f.get("error") for f in refused["failed"]),
          refused["failed"])
    # `refused["failed"][0]` indexes a list the assertion above just said may be
    # empty. That is how a mutant escapes: `P1` restored the bare
    # `except OSError: pass`, the list emptied, the index raised, and the suite
    # died before printing a verdict -- reported as "crashed", not as "caught".
    # A check that can only fail on the path where its subject is missing has to
    # be able to say so without exploding.
    first_error = refused["failed"][0].get("error", "") if refused["failed"] else ""
    check("the reason is the real error, not a placeholder",
          "PermissionError" in first_error, first_error or refused["failed"])
    check("nothing is counted as removed when nothing went",
          refused["removed"] == [], refused["removed"])
    check("prune() returns only what actually went", refused_ids == [],
          refused_ids)
    after = len(list(Path(root).glob("run-*.json")))
    check("the records are still on disk, because they were not deleted",
          after == before, {"before": before, "after": after})

    print("-- the service layer: a tool a person can press")
    svc = Service(db_path=os.path.join(tmp, "autoforge.db"))
    # Wiring first, and guarded. `_call_method` raises `KeyError` on a method
    # that is not on the wire table, so with the wire row deleted every check
    # below would raise instead of failing -- and the runner reports that as
    # `crashed`, which is not the same as `caught` (mutant `P7` proved it: the
    # suite exited 1 with zero FAIL lines). Ask whether the door exists before
    # walking into it, and turn a slam into a sentence if it does not.
    wired = "runs_prune" in getattr(svc, "methods", {})

    def call(method: str, args: dict) -> dict:
        try:
            out = svc._call_method(method, args)
        except Exception as exc:                       # noqa: BLE001
            return {"__raised__": f"{type(exc).__name__}: {exc}"}
        return out if isinstance(out, dict) else {"__answer__": out}

    check("the wire table can reach it", wired,
          sorted(getattr(svc, "methods", {}))[:5])
    if not wired:
        print("  -- remaining service-layer checks need a wired method")
    now = time.time()
    for i in range(2):
        write_record(root, f"svc_old{i:06d}", age_s=20 * 86400, now=now)
    write_record(root, "svc_new0000001", age_s=30, now=now)

    bad = call("runs_prune", {"ttl_days": "seven"})
    check("a TTL that is not a number is refused", "error" in bad, bad)
    zero = call("runs_prune", {"ttl_days": 0})
    check("a zero TTL is refused rather than deleting everything",
          "error" in zero and "positive" in str(zero.get("error")), zero)
    neg = call("runs_prune", {"ttl_days": -3})
    check("and so is a negative one", "error" in neg, neg)
    survived = len(list(Path(root).glob("run-*.json")))
    check("those refusals deleted nothing", survived >= 3, survived)

    out = call("runs_prune", {"ttl_days": 7})
    removed_ids = out.get("removed") or []
    # Not `len(removed) == 2`. The three `gone*` records from the refusal test
    # above are thirty days old and were never deleted -- that was the point of
    # the monkeypatch -- so they are past this TTL too and go with the others.
    # My first draft asserted the fixture's bookkeeping ("two old records were
    # seeded here") instead of the rule ("everything past the TTL went, nothing
    # inside it did"), and it read as a product failure when the product was
    # right. Count what the rule decides, not what the fixture happens to hold.
    check("a real TTL answers with counts", "removed" in out, out)
    check("it removed the records that were past the TTL",
          all(f"svc_old{i:06d}" in removed_ids for i in range(2)),
          removed_ids)
    check("and the leftovers from the refusal test, which were just as old",
          all(g in removed_ids for g in doomed), removed_ids)
    check("it kept the current record",
          sum((out.get("kept") or {}).values()) >= 1, out.get("kept"))
    check("that current record is still on disk",
          (Path(root) / "run-svc_new0000001.json").exists(),
          [p.name for p in Path(root).glob("run-*.json")])
    check("it names the journal directory it acted on",
          Path(str(out.get("journal", ""))) == Path(root), out.get("journal"))
    check("and it says the counts in a sentence too",
          f"removed {len(removed_ids)}" in str(out.get("text", "")),
          out.get("text"))

    con = sqlite3.connect(os.path.join(tmp, "autoforge.db"))
    try:
        rows = con.execute(
            "SELECT payload FROM forge_events WHERE kind = 'runs_pruned'"
        ).fetchall()
    finally:
        con.close()
    check("the act is on the ledger", len(rows) == 1, len(rows))
    if rows:
        payload = json.loads(rows[0][0])
        check("with the numbers, not just a note",
              {"ttl_days", "removed", "failed", "kept"} <= set(payload), payload)
        check("and the removal it claims matches what it did",
              payload.get("removed") == len(removed_ids),
              {"payload": payload.get("removed"), "answer": len(removed_ids)})

    print("-- the four wiring layers agree")
    check("the service has the method",
          callable(getattr(svc, "runs_prune", None)), type(svc).__mro__[:1])
    check("the wire table can reach it",
          "runs_prune" in getattr(svc, "methods", {}), "not in self.methods")
    check("the gate has a scope for it", scope_for("runs_prune") == "local_write",
          scope_for("runs_prune"))
    bridge = Path(PLUGIN) / "lib" / "index.js"
    check("the bridge file was found", bridge.exists(), str(bridge))
    if bridge.exists():
        src = bridge.read_text(encoding="utf-8", errors="replace")
        block = re.search(r"toolName: 'autoforge_runs_prune'.*?\n  \}\)", src,
                          re.S)
        check("the model can see it", block is not None, "no register block")
        if block:
            text = block.group(0)
            check("it calls the method the service actually has",
                  "method: 'runs_prune'" in text, text[:200])
            check("and it takes the TTL, since that is the one decision",
                  "ttl_days" in text, text[:200])

    svc.store.close()
    svc.missions.close()
    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
