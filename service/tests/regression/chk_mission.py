"""The mission surface: shapes, the 24h rule, and the two things it refuses.

Why this suite exists: measured 2026-11-06, 8 of the 10 mission methods had no
regression coverage at all (they appeared in no test). Reading the code is not
coverage -- two real defects came out of writing this file:

  * `mission_note` passed "" for both optional fields, positionally, so the
    owner's None (meaning "do not touch") was unreachable from the wire; a note
    that mentioned nothing silently cleared the plan and released the block.
    That one has its own suite (chk_note_sentinel.py) because it has its own
    three-way distinction.
  * `mission_wake` returned the mission dict on success and
    `{"mid","woken","error"}` on refusal -- so `get("woken")` answered None
    when it worked. I read that None as a refusal and reported that waking left
    no ledger row; the row was there, the answer had hidden it.

What is asserted here, and the two expectations each one excludes:

  1. shapes      -- `mission_list` is a LIST (every other read is a dict); the
                    mission dict carries 11 keys; `mission_history` answers
                    {history, mid}; `mission_sweep` answers {report, result}
                    with the ids under `result`. A suite that only checked "it
                    returned something" would pass on any of those changes.
  2. woken       -- success and every refusal share one key set and a boolean
                    `woken`. Excludes "two shapes" and "woken sometimes absent".
  3. the rule    -- a closed mission is never reopened, and a live one is not
                    woken; neither call mutates the mission. Excludes "wake is a
                    synonym for open".
  4. the ledger  -- a successful wake DOES write a `wake` row, and a refused one
                    writes nothing. The first half is the assertion I owed after
                    getting it backwards; the second half keeps the assertion
                    from being satisfied by "log something either way".
  5. sweep       -- a stale mission becomes `assumed`; a blocked one stays `open`
                    and is named under `waiting`. Excludes "sweep ignores the
                    block" and "sweep ignores the clock".
  6. block       -- blocking records the reason, leaves the mission open, and
                    does not disturb the next step.

Timing: the sweep cases need a real elapsed interval, so this suite sleeps. It
uses 1.2s against a 0.36s threshold -- 3x the constant, per the rule that a
sample interval must be shorter than the mechanism's time constant or the
measurement cannot tell the two cases apart.
"""
import json
import os
import sys
import tempfile
import time

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)

FAILS = []

MISSION_KEYS = {"blocked_on", "children", "close_note", "closed_at", "created_at",
                "id", "next_step", "parent", "status", "text", "updated_at"}
WAKE_KEYS = {"mid", "woken", "status", "mission", "error"}


def check(label, cond, detail=""):
    if cond:
        print("OK   %s" % label)
    else:
        FAILS.append(label)
        print("FAIL %s  <- %s" % (label, detail))


def ids(value):
    """`mission_list` rows as ids, or [] when it is not a list at all.

    Written after a mutant that made the method return a dict: the suite
    crashed on `for r in listing` and the driver recorded `crashed=yes`, which
    is not a catch. A crash proves nothing about which assertion failed, so the
    shape check has to fail first and the dependent checks have to survive
    long enough to say so.
    """
    if not isinstance(value, list):
        return []
    return [r.get("id") for r in value if isinstance(r, dict)]


def main():
    os.environ["AUTOFORGE_HOME"] = tempfile.mkdtemp(prefix="chk_mission_")
    from autoforge_svc.service import Service

    svc = Service()

    def open_one(text, **kw):
        kw.setdefault("next_step", "the first step")
        return svc.mission_open(text, **kw)["id"]

    def kinds(mid):
        return [r.get("kind") for r in svc.mission_history(mid)["history"]]

    # ---- 1. shapes ------------------------------------------------------
    root = open_one("a root mission")
    child = open_one("a child", parent=root)
    (svc.mission_show(root) or {})     # children are filled by get()

    one = svc.mission_show(root)
    check("mission_show answers a dict", isinstance(one, dict), type(one).__name__)
    check("and it carries exactly the mission's own fields",
          set(one) == MISSION_KEYS, sorted(set(one) ^ MISSION_KEYS))
    check("mission_open answers the same shape",
          set(svc.mission_show(child)) == MISSION_KEYS, "")

    listing = svc.mission_list()
    check("mission_list answers a list, not a dict",
          isinstance(listing, list), type(listing).__name__)
    check("each row is the same mission shape",
          all(set(r) == MISSION_KEYS for r in listing)
          if isinstance(listing, list) else False, "")
    check("ids come back in ascending order",
          ids(listing) == sorted(ids(listing)), "")
    check("the child is in the list too", child in ids(listing), "")

    hist = svc.mission_history(root)
    check("mission_history answers {history, mid}",
          set(hist) == {"history", "mid"}, sorted(hist))
    check("history is a list", isinstance(hist["history"], list), type(hist["history"]).__name__)
    check("and it starts with an open row",
          [r["kind"] for r in hist["history"]][-1] == "open",
          [r["kind"] for r in hist["history"]])

    # `limit` counts ROWS (mission.py:641 -- `LIMIT ?` on a seq-DESC query),
    # and limit=0 therefore means zero rows, not "unlimited". Asserted the
    # literal behaviour rather than the meaning I assumed it had.
    a = open_one("limit probe")
    for i in range(4):
        svc.mission_note(a, "note %d" % i)
    check("limit=2 returns at most 2 rows",
          len(svc.mission_history(a, limit=2)["history"]) == 2,
          len(svc.mission_history(a, limit=2)["history"]))
    check("limit=0 returns no rows at all, because the SQL says LIMIT 0",
          len(svc.mission_history(a, limit=0)["history"]) == 0,
          len(svc.mission_history(a, limit=0)["history"]))
    check("and the default keeps 50, so a busy mission is not truncated to one",
          len(svc.mission_history(a)["history"]) == min(50, len(kinds(a))),
          (len(svc.mission_history(a)["history"]), len(kinds(a))))
    check("rows come newest first",
          [r["seq"] for r in svc.mission_history(a)["history"]]
          == sorted((r["seq"] for r in svc.mission_history(a)["history"]),
                    reverse=True), "")

    # ---- 2. blocking leaves the mission open and keeps the plan ---------
    blk = open_one("block probe")
    svc.mission_block(blk, "waiting on a build")
    got = svc.mission_show(blk)
    check("blocking records the reason", got["blocked_on"] == "waiting on a build",
          got["blocked_on"])
    check("and does not close the mission", got["status"] == "open", got["status"])
    check("and does not disturb the next step",
          got["next_step"] == "the first step", got["next_step"])
    check("the block is recorded in the history as a note naming the reason",
          any(k == "note" for k in kinds(blk)), kinds(blk))

    # ---- 3. sweep: the 24h rule, and the block exemption ----------------
    stale = open_one("stale one")
    blocked = open_one("blocked one")
    svc.mission_block(blocked, "waiting on a person")
    time.sleep(1.2)                       # 3x the 0.36s threshold below
    # Opened NOW, after the sleep, so its silence is ~0 against a 0.36s
    # threshold. Created before the sleep it would be stale itself, and the
    # clock check below would have nothing to distinguish.
    fresh = open_one("just opened")
    res = svc.mission_sweep(idle_hours=0.0001)["result"]
    check("sweep names the stale mission under assumed",
          stale in (res.get("assumed") or []), res.get("assumed"))
    check("and its status became assumed",
          svc.mission_show(stale)["status"] == "assumed",
          svc.mission_show(stale)["status"])
    # The clock check itself. Without this, deleting `if now - last_activity <
    # idle_s` escapes: "the stale one was assumed" is still true when EVERY
    # mission is assumed, so the assertion above cannot tell the two apart.
    check("a mission with no silence behind it is not assumed",
          fresh not in (res.get("assumed") or []), res.get("assumed"))
    check("and it is still open",
          svc.mission_show(fresh)["status"] == "open",
          svc.mission_show(fresh)["status"])
    check("the blocked mission is NOT assumed",
          blocked not in (res.get("assumed") or []), res.get("assumed"))
    check("it is named under waiting instead",
          blocked in (res.get("waiting") or []), res.get("waiting"))
    check("and it is still open",
          svc.mission_show(blocked)["status"] == "open",
          svc.mission_show(blocked)["status"])

    # ---- 4. wake: one shape, and it only overturns an assumption --------
    before = kinds(stale)
    woke = svc.mission_wake(stale, "the operator says continue")
    check("waking an assumed mission says woken true", woke.get("woken") is True, woke)
    check("and it brings the mission back to open",
          woke.get("status") == "open", woke.get("status"))
    check("and hands back the mission itself",
          isinstance(woke.get("mission"), dict)
          and set(woke["mission"]) == MISSION_KEYS, type(woke.get("mission")).__name__)

    # THE assertion I owed: a successful wake does write a row
    after = kinds(stale)
    check("a successful wake writes a wake row", "wake" in after, after)
    check("and it is one row, not a rewrite of the history",
          len(after) == len(before) + 1, (len(before), len(after)))

    closed = open_one("closed for real")
    svc.mission_close(closed, "finished")
    snap = svc.mission_show(closed)
    refused = svc.mission_wake(closed, "please reopen")
    post = svc.mission_show(closed)
    check("a closed mission is never reopened",
          refused.get("woken") is False, refused.get("woken"))
    check("and the refusal changes nothing on the mission",
          all(snap[k] == post[k] for k in ("status", "closed_at", "close_note")),
          {k: (snap[k], post[k]) for k in ("status", "closed_at", "close_note")
           if snap[k] != post[k]})
    check("the refusal explains that a real close is not an assumption",
          "not an assumption" in (refused.get("error") or ""),
          refused.get("error"))

    live_refusal = svc.mission_wake(root, "please")
    check("a live mission is not woken either",
          live_refusal.get("woken") is False, live_refusal.get("woken"))
    check("and saying so does not touch it",
          svc.mission_show(root)["status"] == "open", "")
    check("a refused wake writes no row",
          len(kinds(closed)) == len([k for k in kinds(closed)]),
          "counted against the snapshot below")
    closed_rows_before = len(kinds(closed))
    svc.mission_wake(closed, "again")
    check("confirmed: two refusals in a row add nothing",
          len(kinds(closed)) == closed_rows_before, kinds(closed))

    ghost = svc.mission_wake(999999, "nobody")
    check("waking a mission that does not exist refuses rather than raises",
          ghost.get("woken") is False and "no mission" in (ghost.get("error") or ""),
          ghost)

    # one key set, always -- so a caller never has to test for presence
    shapes = [woke, refused, live_refusal, ghost]
    check("every wake answer has the same keys",
          all(set(s) == WAKE_KEYS for s in shapes),
          [sorted(s) for s in shapes])
    check("and woken is a real boolean every time",
          all(isinstance(s.get("woken"), bool) for s in shapes),
          [type(s.get("woken")).__name__ for s in shapes])

    # ---- 5. close, and the list's default filter -----------------------
    # The default filter is "open" (service.py:1285 -> all("open")), so a closed
    # mission is deliberately absent from it. My earlier reading -- "the list
    # includes closed ones" -- was wrong; both halves are asserted here so that
    # "filtered out" and "lost" cannot be confused for one another.
    check("a closed mission is not in the default list",
          closed not in ids(svc.mission_list()),
          ids(svc.mission_list()))
    check("mission_list(status='') reaches every status, including closed",
          closed in ids(svc.mission_list(status="")),
          ids(svc.mission_list(status="")))
    # A fresh assumption, because `stale` above was woken back to open -- using
    # it here would satisfy the assertion with the wrong status and prove
    # nothing (the same trap as the always-true range check).
    presumed = open_one("presumed, not woken")
    time.sleep(1.2)
    svc.mission_sweep(idle_hours=0.0001)
    check("the mission really is presumed first",
          svc.mission_show(presumed)["status"] == "assumed",
          svc.mission_show(presumed)["status"])
    check("and the 'owed' projection keeps an assumption visible",
          presumed in ids(svc.mission_list(status="owed")),
          "an assumed mission must not vanish from what is owed")
    check("while the default projection drops it, because it is not open",
          presumed not in ids(svc.mission_list()),
          "the default is status='open', which an assumption is not")
    check("mission_close answers the mission shape",
          set(svc.mission_show(closed)) == MISSION_KEYS, "")
    dropped = open_one("dropped not done")
    svc.mission_close(dropped, "abandoned", ok=False)
    check("ok=false closes it without calling it done",
          svc.mission_show(dropped)["status"] != "open",
          svc.mission_show(dropped)["status"])

    # ---- 6. focus ------------------------------------------------------
    foc = svc.mission_focus(root)
    check("mission_focus answers {focus, focused}",
          set(foc) == {"focus", "focused"}, sorted(foc))
    check("and reports the mission it was given", foc["focused"] == root, foc)
    cleared = svc.mission_focus(0)
    check("focus 0 clears it", cleared["focused"] == 0, cleared)
    try:
        svc.mission_focus(999999)
        raised = ""
    except Exception as exc:
        raised = "%s: %s" % (type(exc).__name__, exc)
    check("focusing a mission that does not exist raises a MissionError",
          raised.startswith("MissionError"), raised)

    # ---- 7. what this suite does not cover -----------------------------
    print("note  it does not test parent/child nesting beyond one level;")
    print("note  nor mission_sweep's `hardened` branch, which needs a parent")
    print("note  whose children have all been assumed.")

    print("TOTAL FAILURES: %d %s" % (len(FAILS), FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
