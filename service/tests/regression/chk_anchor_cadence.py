"""The anchor has a cadence, and the uncovered tail is a number.

Measured 2026-11-05 on the live ledger: 8,657 rows, newest anchor at up_to_id
8,054 -- 604 rows (7.0%) that no anchor reached, including all 8 review
receipts. `chaining.anchor` existed and worked; the only caller was
`Service.anchor_ledger`, a tool nobody called. A boundary that is taken when
somebody remembers is a boundary that is not taken.

So `ToolStore.log_event` now anchors after a write, on either of two triggers:

  * a decision row (`_ANCHOR_NOW_KINDS`) -- immediately, because the question
    a later reader asks is "was this decided before or after the evidence
    moved", and an interval cannot answer it;
  * `_ANCHOR_EVERY` rows past the newest anchor -- a bound on how much a later
    editor could re-sign without `anchored_head_missing` noticing.

And `chaining.summarize` reports `newest_anchor` / `rows_beyond_anchor`, so the
gap is countable from `report()` instead of being invisible.

What this file has to be able to catch, in order:
  1. the interval fires exactly at the boundary and not one row early;
  2. a decision anchors without waiting for the interval;
  3. ordinary chatter right after a decision does NOT anchor again
     (otherwise every busy turn writes two anchors and the table becomes noise);
  4. `AUTOFORGE_ANCHOR_EVERY=0` turns the interval off but NOT the decision
     trigger -- turning that off would restore the state this change ends;
  5. a failed anchor is counted and swallowed, never raised: the row it
     follows is already committed;
  6. the chain still verifies, and `rows_beyond_anchor` goes to zero after a
     boundary is declared;
  7. no recursion: `anchor()` writes `chain_anchors`, so a write that anchors
     cannot ask for another anchor.
"""
import importlib
import json
import os
import shutil
import sqlite3
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


def fresh(tmp, every=None):
    """A store in its own directory, with the interval set as asked."""
    if every is None:
        os.environ.pop("AUTOFORGE_ANCHOR_EVERY", None)
    else:
        os.environ["AUTOFORGE_ANCHOR_EVERY"] = str(every)
    os.environ["AUTOFORGE_HOME"] = tmp
    import autoforge_svc.store as S
    importlib.reload(S)          # the knob is read at import time
    import autoforge_svc.chaining as C
    importlib.reload(C)
    return S, S.ToolStore(db_path=os.path.join(tmp, "t.db"))


def anchors(store):
    return store._conn.execute("SELECT COUNT(*) FROM chain_anchors").fetchone()[0]


def newest(store):
    return store._conn.execute(
        "SELECT COALESCE(MAX(up_to_id), 0) FROM chain_anchors").fetchone()[0]


def note_of(store):
    """The newest anchor's note, or "" when there is no anchor at all.

    Written as a helper because the first version indexed straight into
    `fetchone()`. When a mutant stopped the anchor from being taken, there was
    no row, `fetchone()` returned None, and the suite died with a TypeError
    instead of reporting a failure -- which the mutation driver scores as
    "escaped", not "caught". A test that cannot survive the absence it is
    testing for cannot testify about it.
    """
    row = store._conn.execute(
        "SELECT note FROM chain_anchors ORDER BY id DESC LIMIT 1").fetchone()
    return str(row[0] or "") if row else ""


def main():
    print("== 1. the interval fires at the boundary and not one row early ==")
    tmp = tempfile.mkdtemp(prefix="chk_anchor_")
    try:
        S, st = fresh(tmp, every=10)
        check("the knob is read from the environment", S._ANCHOR_EVERY == 10,
              "got %r" % (S._ANCHOR_EVERY,))
        check("a fresh store holds no anchors", anchors(st) == 0, str(anchors(st)))
        for i in range(9):
            st.log_event("chatter", {"i": i})
        check("nine rows past the start is not yet due", anchors(st) == 0,
              "anchors after 9 rows: %d" % anchors(st))
        st.log_event("chatter", {"i": 9})
        check("the tenth row takes the anchor", anchors(st) == 1, str(anchors(st)))
        check("and it freezes the row it followed", newest(st) == 10, str(newest(st)))
        st.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("== 2. a decision anchors without waiting for the interval ==")
    tmp = tempfile.mkdtemp(prefix="chk_anchor_")
    try:
        S, st = fresh(tmp, every=1000)
        for i in range(3):
            st.log_event("chatter", {"i": i})
        check("three rows with a long interval: no anchor yet", anchors(st) == 0,
              str(anchors(st)))
        st.log_event("review", {"tool": "x", "ok": False})
        check("a review row anchors immediately", anchors(st) == 1, str(anchors(st)))
        check("its note names the kind, not a row count",
              "decision row review" in note_of(st), note_of(st))
        st.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("== 3. chatter after a decision does not anchor again ==")
    tmp = tempfile.mkdtemp(prefix="chk_anchor_")
    try:
        S, st = fresh(tmp, every=1000)
        st.log_event("tool_defined", {"name": "t"})
        first = anchors(st)
        for i in range(20):
            st.log_event("chatter", {"i": i})
        check("a decision anchored once", first == 1, str(first))
        check("twenty ordinary rows add nothing", anchors(st) == first,
              "%d -> %d" % (first, anchors(st)))
        st.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("== 4. the interval can be switched off; the decision trigger cannot ==")
    tmp = tempfile.mkdtemp(prefix="chk_anchor_")
    try:
        S, st = fresh(tmp, every=0)
        for i in range(50):
            st.log_event("chatter", {"i": i})
        check("interval 0 means no anchors from chatter", anchors(st) == 0,
              str(anchors(st)))
        st.log_event("retire", {"name": "gone"})
        check("but a decision still takes one", anchors(st) == 1, str(anchors(st)))
        check("the note says it was a decision",
              "decision row retire" in note_of(st), note_of(st))
        st.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("== 5. a failed anchor is counted, never raised ==")
    tmp = tempfile.mkdtemp(prefix="chk_anchor_")
    try:
        S, st = fresh(tmp, every=5)
        for i in range(4):
            st.log_event("chatter", {"i": i})
        real = S.chaining.anchor

        def exploded(conn, note="", writer=None):
            raise sqlite3.OperationalError("database is locked")

        S.chaining.anchor = exploded
        try:
            st.log_event("chatter", {"i": 4})
            raised = None
        except Exception as exc:                               # noqa: BLE001
            raised = exc
        finally:
            S.chaining.anchor = real
        check("the write that triggered it still succeeds", raised is None, repr(raised))
        check("and the failed anchor is counted", st._anchor_skips == 1,
              "skips: %d" % st._anchor_skips)
        check("while the successful ones are counted separately", st._anchor_runs == 0,
              "runs: %d" % st._anchor_runs)
        st.log_event("chatter", {"i": 5})
        check("the next write anchors normally", anchors(st) == 1, str(anchors(st)))
        check("and the run counter moved, not the skip counter",
              (st._anchor_runs, st._anchor_skips) == (1, 1),
              "runs=%d skips=%d" % (st._anchor_runs, st._anchor_skips))
        st.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("== 6. the gap is countable, and a boundary closes it ==")
    tmp = tempfile.mkdtemp(prefix="chk_anchor_")
    try:
        S, st = fresh(tmp, every=7)
        for i in range(20):
            st.log_event("chatter", {"i": i})
        s = st.chain_summary()
        check("the summary names the newest anchor", isinstance(s.get("newest_anchor"), int),
              str(s.get("newest_anchor")))
        check("it names how many rows are past it",
              isinstance(s.get("rows_beyond_anchor"), int),
              str(s.get("rows_beyond_anchor")))
        # The exact number, not a range. The first version asserted
        # `0 <= rows_beyond_anchor < 7`, and a mutant that always answered 0
        # sailed through it: an inequality that admits zero cannot tell a
        # measurement from an empty placeholder. With interval 7 and 20 rows,
        # anchors land on rows 7 and 14, so exactly 6 rows sit past the newest.
        check("the count of rows past the newest anchor is the number, not a range",
              s["rows_beyond_anchor"] == 6, "got %r, expected 6" % (s["rows_beyond_anchor"],))
        check("and 20 rows took two anchors", s.get("anchors") == 2, str(s.get("anchors")))
        v = st.verify_chain()
        check("the chain still verifies", v.get("ok") is True, str(v.get("breaks")))
        check("no breaks were invented", (v.get("breaks") or []) == [], str(v.get("breaks")))
        check("every row is chained", v.get("gaps") == [], str(v.get("gaps")))
        st.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("== 7. anchoring does not recurse ==")
    tmp = tempfile.mkdtemp(prefix="chk_anchor_")
    try:
        S, st = fresh(tmp, every=1)
        # interval 1 means every write anchors; if anchor() wrote to
        # forge_events this would not terminate in bounded time.
        for i in range(6):
            st.log_event("chatter", {"i": i})
        rows = st._conn.execute("SELECT COUNT(*) FROM forge_events").fetchone()[0]
        check("every write still lands exactly one row", rows == 6, str(rows))
        check("and each one took its boundary", anchors(st) == 6, str(anchors(st)))
        v = st.verify_chain()
        check("the chain still verifies at interval 1", v.get("ok") is True,
              str(v.get("breaks")))
        st.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
