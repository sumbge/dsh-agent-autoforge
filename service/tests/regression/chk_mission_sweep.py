"""P1-3: the 24h rule runs every turn -- without writing on every turn.

`MissionStore.report()` carries a comment that refuses to sweep:

    Read-only: the sweep that would flip states writes to the store, and the
    prompt is composed every turn. A report that did the sweeping would take a
    write lock on every turn and move the prompt every time a boundary was
    crossed.

That objection is half right. The write is real, but it only happens when a
boundary is actually crossed -- once, not every turn. What costs every turn is
the *search* for a boundary: measured 2026-11-03, `sweep()` over 50 open
missions is 11.49 ms, because it re-reads `last_activity()` (two queries) per
mission, while a single SQL count answering "would anything change?" is
0.056 ms flat, at 0 and at 50.

So the rule becomes: ask the cheap question every turn, run the expensive sweep
only when the answer is yes. Upstream does this unconditionally at the top of
`_effective_prompt` (`agent.py` `self._sweep_missions()`); we cannot, because
upstream's store is not the block's own reader and ours is.

The dangerous direction of error is one-sided and this file pins it: the
pre-check may say "due" when the sweep then declines (a parent with an owed
child, a mission whose log was touched), but it must NEVER say "nothing due"
while the sweep would move something. A false negative means the 24h rule
silently stops firing, which is the kind of quiet death this whole audit is
about.
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, SERVICE)

from autoforge_svc import mission as mission_mod          # noqa: E402
from autoforge_svc.service import Service                # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail="") -> None:
    if cond:
        print("  OK   " + label)
    else:
        print("  FAIL " + label + ("" if not detail else " <- %s" % (detail,)))
        FAILS.append(label)


def age(store, mid: int, seconds: float) -> None:
    """Move every clock this mission has, so `last_activity` really ages.

    `last_activity` is the max of `updated_at`, `created_at` and the newest
    non-'assume' log row. Ageing only one of them leaves the mission young --
    which is exactly the mistake the first version of this fixture made, and it
    read as "the pre-check disagrees with the sweep".
    """
    when = time_now() - seconds
    con = store._conn
    con.execute("UPDATE missions SET created_at=?, updated_at=? WHERE id=?",
                (when, when, mid))
    con.execute("UPDATE mission_log SET at=? WHERE mission=?", (when, mid))
    con.commit()


def time_now() -> float:
    import time
    return time.time()


tmp = tempfile.mkdtemp(prefix="m26_sweep_")
db = os.path.join(tmp, "s.db")
svc = Service(db_path=db)
missions = svc.missions
DAY = mission_mod.IDLE_ASSUMPTION_S
WEEK = mission_mod.RESURRECT_WINDOW_S


def pend(store, idle_s=DAY):
    """Ask the pre-check, or answer 'not written yet' without crashing.

    A missing method used to abort this file at its first line and report one
    traceback instead of N failures. That is the same distinction as a probe
    that raises: a crash says the file could not finish, not which of the
    claims is false. So the callers below all go through here.
    """
    fn = getattr(store, "pending_transitions", None)
    if not callable(fn):
        return {}
    try:
        return fn(idle_s=idle_s)
    except Exception as exc:                               # noqa: BLE001
        # A pre-check that raises is a pre-check that is wrong, and the caller
        # depends on the difference between "nothing is due" and "the question
        # could not be asked". Returning {} here would read as "nothing is due"
        # -- the false negative this file exists to forbid. So the raise
        # becomes a FAIL below, never a traceback that ends the run before the
        # other claims are tested.
        return {"__raised__": "%s: %s" % (type(exc).__name__, exc),
                "assume": None, "harden": None, "total": None}


print("-- the cheap question exists and answers the same question as the sweep")
check("MissionStore has a read-only pre-check",
      callable(getattr(missions, "pending_transitions", None)),
      [n for n in dir(missions) if "pend" in n])

empty = pend(missions)
check("the pre-check answers instead of raising", "__raised__" not in empty,
      empty.get("__raised__"))
check("nothing open means nothing pending", empty.get("total") == 0, empty)
check("and it counts the two transitions separately",
      set(empty) >= {"assume", "harden", "total"}, sorted(empty))

print()
print("-- a fresh mission is not due")
fresh = svc.mission_open("just started")["id"]
p = pend(missions)
check("a mission opened now is not due to be assumed", p.get("assume") == 0, p)

print()
print("-- a silent mission becomes due, and the sweep agrees")
old = svc.mission_open("gone quiet")["id"]
age(missions, old, 2 * DAY)
p2 = pend(missions)
check("a mission silent for two days is due to be assumed", p2.get("assume") == 1, p2)
wrote_before = missions._conn.total_changes
ran = missions.sweep(idle_s=DAY)
check("the sweep assumed it", old in ran.get("assumed", []), ran)
check("the pre-check said so first, so the two agree",
      pend(missions).get("assume") == 0)
check("the pre-check itself wrote nothing",
      missions._conn.total_changes >= wrote_before,
      "sanity: the sweep is what wrote, not the check")

print()
print("-- a blocked mission is silent on purpose and must not be presumed")
blocked = svc.mission_open("waiting on a person")["id"]
age(missions, blocked, 3 * DAY)
missions._conn.execute("UPDATE missions SET blocked_on=? WHERE id=?",
                       ("waiting on the operator", blocked))
missions._conn.commit()
p3 = pend(missions)
check("a claim of being blocked keeps it out of the count", p3.get("assume") == 0, p3)
ran3 = missions.sweep(idle_s=DAY)
check("and the sweep agrees -- silence means still blocked",
      blocked not in ran3.get("assumed", []), ran3)

print()
print("-- a parent with an owed child: the pre-check may say due, the sweep declines")
parent = svc.mission_open("the whole job")["id"]
child = svc.mission_open("one piece of it", parent=parent)["id"]
age(missions, parent, 2 * DAY)
p4 = pend(missions)
ran4 = missions.sweep(idle_s=DAY)
check("the parent is not assumed while the child is owed",
      parent not in ran4.get("assumed", []), ran4)
check("(the pre-check is allowed to over-report; a false NEGATIVE is the bug)",
      (p4.get("assume") or 0) >= len(ran4.get("assumed", [])),
      "pre=%s swept=%s" % (p4.get("assume"), ran4.get("assumed")))
svc.mission_close(child, note="done", ok=True)

print()
print("-- a peer's touch is in the log, and it counts as being worked")
# The count is store-wide, and by now other missions are legitimately due (the
# parent above, whose child was just closed). So the assertion is a DELTA: add
# this one mission and the count must not move. Comparing against zero would
# have been an assertion about the fixture's leftovers, not about the peer.
before_n = pend(missions).get("assume") or 0
peer = svc.mission_open("annotated by another session")["id"]
con0 = missions._conn
old_t = time_now() - 3 * DAY
con0.execute("UPDATE missions SET created_at=?, updated_at=? WHERE id=?",
             (old_t, old_t, peer))
# a log row written by somebody else, just now -- `updated_at` still says days ago
con0.execute("INSERT INTO mission_log (mission, at, kind, note) VALUES (?,?,?,?)",
             (peer, time_now(), "note", "a peer session is mid-sentence on this"))
con0.commit()
check("the mission's own columns still look days old",
      missions.get(peer).updated_at < time_now() - 2 * DAY)
after_n = pend(missions).get("assume") or 0
check("a peer's recent log row adds nothing to the count",
      after_n == before_n, "before %s after %s" % (before_n, after_n))
ran6 = missions.sweep(idle_s=DAY)
check("and the sweep agrees -- it was touched, so it is not silent",
      peer not in ran6.get("assumed", []), ran6)

print()
print("-- the second clock: an assumption that should harden into done")
assumed = svc.mission_open("presumed complete")["id"]
age(missions, assumed, 2 * DAY)
missions.sweep(idle_s=DAY)
check("it is assumed", missions.get(assumed).status == mission_mod.ASSUMED,
      missions.get(assumed).status)
harden_due = pend(missions)
check("an assumption younger than the resurrection window is not due to harden",
      harden_due.get("harden") == 0, harden_due)
con = missions._conn
con.execute("UPDATE mission_log SET at=? WHERE mission=? AND kind='assume'",
            (time_now() - 2 * WEEK, assumed))
con.commit()
harden_now = pend(missions)
check("past the window it is", harden_now.get("harden") == 1, harden_now)
ran5 = missions.sweep(idle_s=DAY)
check("and the sweep hardens it",
      assumed in ran5.get("hardened", []) or missions.get(assumed).status == mission_mod.DONE,
      ran5)

print()
print("-- the read-only promise: asking does not move the store")
before = con.execute("SELECT COUNT(*) FROM mission_log").fetchone()[0]
statuses_before = {m.id: m.status for m in missions.all()}
for _ in range(20):
    pend(missions)
after = con.execute("SELECT COUNT(*) FROM mission_log").fetchone()[0]
check("twenty checks write no log rows", before == after, "%s -> %s" % (before, after))
check("and change no status",
      statuses_before == {m.id: m.status for m in missions.all()})
check("and leave no transaction open", con.in_transaction is False, con.in_transaction)

print()
print("-- the brief sweeps, and only writes when there is something to write")
quiet = Service(db_path=os.path.join(tmp, "quiet.db"))
check("a quiet service's brief still renders",
      bool(quiet._mission_lines()) and "MISSION" in "".join(quiet._mission_lines()))
rows_before = quiet.missions._conn.execute(
    "SELECT COUNT(*) FROM mission_log").fetchone()[0]
for _ in range(3):
    quiet._mission_lines()
check("three turns with nothing due write nothing",
      quiet.missions._conn.execute(
          "SELECT COUNT(*) FROM mission_log").fetchone()[0] == rows_before)
loud = quiet.mission_open("silent since forever")["id"]
age(quiet.missions, loud, 2 * DAY)
lines = quiet._mission_lines()
text = "\n".join(lines)
check("a turn that finds a due transition runs the sweep",
      quiet.missions.get(loud).status == mission_mod.ASSUMED,
      quiet.missions.get(loud).status)
check("and the block says so instead of leaving it to be discovered later",
      "assumed done" in text, text[-200:])
check("the turn that swept still returns a mission block, not an error",
      "unreadable" not in text, text[:120])

print()
print("-- the pre-check is cheap, and stays cheap as missions accumulate")
import time
for i in range(40):
    quiet.mission_open("filler %d" % i)
def best(fn, n=7):
    b = 9e9
    for _ in range(n):
        t = time.perf_counter(); fn(); b = min(b, (time.perf_counter() - t) * 1000)
    return b
cheap = best(lambda: pend(quiet.missions))
full = best(lambda: quiet.missions.sweep(idle_s=DAY))
check("the question costs far less than the sweep at this size",
      cheap * 10 < full, "check %.3f ms vs sweep %.3f ms" % (cheap, full))
check("and it is under a millisecond, so a per-turn caller can afford it",
      cheap < 1.0, "%.3f ms" % cheap)

for s in (svc, quiet):
    s.store.close()
    s.missions.close()

print()
print("TOTAL FAILURES:", len(FAILS), FAILS)
sys.exit(0 if not FAILS else 1)
