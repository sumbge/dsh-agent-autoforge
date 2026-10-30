"""Missions: what I owe, kept where losing sight of it is visible.

The failure this exists for is not "I cannot remember a fact". It is that a
session with many tasks loses the one sentence saying what all of them are
FOR, and then every new request silently replaces it -- the work still gets
done, and the mission it was for is never closed, or never named. A schedule
entry is something that happens at a time; a kept fact is something I know; a
mission is something I owe. Kept apart on purpose, so that "what am I for"
answers with a mission instead of with a fact about a directory.

Storage is the same sqlite file as the tool store, so a mission survives a
restart for the same reason the tools do, and the ledger and the mission list
cannot drift apart by living in two places.

Two rules the shape of this store enforces:

* A mission is never deleted by `close` -- it is closed, with a note. "I
  stopped tracking it" and "it is finished" must not look the same afterwards.
* A new request does not overwrite a mission. It either becomes a new mission
  or a line under an existing one (`parent`), which is why the report prints
  the tree and not a flat list.
* Silence is not evidence of work, and it is not evidence of idleness either.
  So a mission with no activity for 24 hours is ASSUMED COMPLETE rather than
  left owed forever -- see `sweep`. The assumption is a third state, not a
  quiet close: it is printed in the report as an assumption, the operator can
  overturn it by touching the mission, and only seven more days of silence turn
  it into a real close whose note says it was assumed. The alternative -- a
  list that only ever grows -- is one nobody reads, and an unread list is the
  same as no list.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

#: Marks a close that came from silence rather than from work. A prefix on the
#: close note keeps the state readable by `Mission.assumed_close` without adding
#: a column two sessions on the same file would have to agree about.
ASSUMED_CLOSE_PREFIX = "assumed: "

_DEFAULT_DB = os.path.join(
    os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
    "autoforge", "autoforge.db",
)

OPEN = "open"
ASSUMED = "assumed"
DONE = "done"
DROPPED = "dropped"
STATUSES = (OPEN, ASSUMED, DONE, DROPPED)

#: How long a mission may sit untouched before it is presumed finished. The
#: operator's rule, in his words: "if a task has not been asked to continue for
#: more than 24 hours, assume it is complete". A full day is the unit because it
#: is the smallest interval that contains a night's sleep on both sides of it --
#: an assumption that fires while the operator is asleep is an assumption about
#: nothing.
IDLE_ASSUMPTION_S = 24 * 3600.0

#: How long an assumption stands before it hardens into a close. Long enough
#: that a week's absence does not erase work the operator still wants, short
#: enough that the owed list stays readable. This second clock exists because a
#: single clock would make every wrong assumption permanent: the mission would
#: vanish from the report the moment it was presumed done, and a presumption
#: nobody can see is a deletion.
RESURRECT_WINDOW_S = 7 * 24 * 3600.0

# The report is re-sent on every turn, so it is bounded like the kept-facts
# block is: an unbounded mission list would price the agenda at the cost of the
# work, and the agent would have no way to see that happen.
REPORT_BUDGET_CHARS = 1100

_DDL = """
CREATE TABLE IF NOT EXISTS missions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    parent      INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'open',
    next_step   TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    closed_at   REAL,
    close_note  TEXT NOT NULL DEFAULT '',
    blocked_on  TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS mission_log (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    mission INTEGER NOT NULL,
    at      REAL NOT NULL,
    kind    TEXT NOT NULL,
    note    TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS mission_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class MissionError(Exception):
    """Asked for a mission id that is not there, or an illegal move."""


@dataclass
class Mission:
    id: int
    text: str
    parent: int = 0
    status: str = OPEN
    next_step: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    closed_at: float | None = None
    close_note: str = ""
    #: What this is waiting on, in the operator's own words. Non-empty means
    #: "silence here is not idleness, it is a dependency": the sweep reads it as
    #: the operator having claimed the mission aloud, which is the test the 24h
    #: rule is only a proxy for. Without it the rule would presume complete every
    #: mission simply waiting for a download, a credential or a person.
    blocked_on: str = ""
    children: list["Mission"] = field(default_factory=list)

    @property
    def assumed_close(self) -> bool:
        """Closed by silence rather than by work. The two must not read alike.

        The close note is the machine-readable mark -- `done` alone would make
        "the work finished" and "nobody spoke for eight days" the same sentence
        in the record, which is exactly the conflation this store exists to stop.
        """
        return self.status == DONE and self.close_note.startswith(ASSUMED_CLOSE_PREFIX)

    def line(self, indent: int = 0) -> str:
        pad = "  " * indent
        if self.assumed_close:
            mark = "[done by assumption]"
        elif self.status == ASSUMED:
            mark = "[open, assumed done]"
        else:
            mark = {"open": "[open]", "done": "[done]",
                    "dropped": "[dropped]"}.get(self.status, f"[{self.status}]")
        out = f"{pad}M{self.id} {mark} {self.text}"
        if self.status in (OPEN, ASSUMED) and self.next_step:
            out += f"\n{pad}    next: {self.next_step}"
        if self.status in (OPEN, ASSUMED) and self.blocked_on:
            out += (f"\n{pad}    waiting on: {self.blocked_on}"
                    f"  (so silence here is NOT taken as done)")
        if self.status == ASSUMED:
            # One line, and no clock in it: this text rides in the prompt on
            # every turn, and an hour count that changes every hour would move
            # the prompt prefix and re-bill the whole conversation behind it.
            # The clock is in mission_show, where it is read on purpose.
            out += (f"\n{pad}    no activity for a day: presumed complete. Say the "
                    f"word and it is owed again (mission_note M{self.id}); closed "
                    f"for real after {int(RESURRECT_WINDOW_S // 86400)} days of silence.")
        elif self.close_note:
            out += f"\n{pad}    closed: {self.close_note}"
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "text": self.text, "parent": self.parent,
            "status": self.status, "next_step": self.next_step,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "closed_at": self.closed_at, "close_note": self.close_note,
        }


class MissionStore:
    """A small, durable list of what is owed. Append-minded: close, never erase.

    Deliberately not a task scheduler. `autoforge/schedule.py` answers "when
    should this run"; this answers "what is this all for", and the answers must
    stay queryable at different moments -- which is exactly the distinction a
    session with many tasks loses first.
    """

    def __init__(self, db_path: str | None = None,
                 conn: sqlite3.Connection | None = None) -> None:
        """Open the store, or adopt a connection the caller already holds.

        The `conn` path exists because of a failure this store caused the first
        time it ran: a ToolStore and a MissionStore on the same sqlite file are
        two connections, and the second one's schema setup hit
        `database is locked` while the first was alive. That is not a rare
        interleaving -- the agent holds both at once by design, since a mission
        and the ledger entry for the work it caused should live in one file so
        they cannot drift apart.

        So the default wiring is one file, one connection. `db_path` is still
        honoured for the standalone reader (`auto mission`), which has no tool
        store to borrow from.
        """
        self._owns_conn = conn is None
        self.db_path = str(db_path or _DEFAULT_DB)
        if not os.path.isabs(self.db_path):
            self.db_path = os.path.abspath(self.db_path)
        if conn is not None:
            self._conn = conn
        else:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            # A second reader (the `auto mission` CLI, or a peer session on the
            # same store) must wait rather than fail: the failure it would
            # otherwise produce is "the mission list is unreadable", which is
            # the one thing this store exists to prevent.
            self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_DDL)
        self._ensure_columns()
        self._conn.commit()

    def close(self) -> None:
        """Release the connection -- but never one it borrowed.

        Closing a connection it did not open would take the tool store's
        connection down with it, and the agent would lose its ledger mid-run.
        """
        if self._owns_conn:
            self._conn.close()

    # -- writes -----------------------------------------------------------
    def open(self, text: str, parent: int = 0, next_step: str = "",
             blocked_on: str = "") -> Mission:
        text = " ".join((text or "").split())
        if not text:
            raise MissionError("a mission needs a sentence; empty is not a mission")
        if parent:
            row = self._conn.execute(
                "SELECT status FROM missions WHERE id=?", (parent,)).fetchone()
            if row is None:
                raise MissionError(f"no mission M{parent} to hang this under")
            if row["status"] != OPEN:
                raise MissionError(
                    f"M{parent} is {row['status']}; open a new mission instead of "
                    f"reviving a closed one, so the record of closing it stays true")
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO missions (text, parent, status, next_step, created_at,"
            " updated_at, blocked_on) VALUES (?,?,?,?,?,?,?)",
            (text, int(parent), OPEN, next_step, now, now, (blocked_on or "").strip()))
        self._conn.commit()
        mid = int(cur.lastrowid)
        self._log(mid, "open", text)
        return self.get(mid)

    def note(self, mid: int, note: str = "", next_step: str | None = None,
             blocked_on: str | None = None) -> Mission:
        m = self.get(mid)
        if m.status == ASSUMED:
            # Touching it IS the request to continue, so the assumption is
            # overturned by the same act that carries the news. Requiring a
            # separate `wake` first would mean the honest move -- "here is where
            # it got to" -- silently left the mission presumed complete.
            self.wake(mid, note or "touched, so the assumption is overturned")
        now = time.time()
        sets, args = ["updated_at=?"], [now]
        if next_step is not None:
            sets.append("next_step=?")
            args.append(next_step)
        if blocked_on is not None:
            # Empty string clears it, on purpose: "it is no longer waiting on
            # anything" has to be sayable, or the exemption would be a one-way
            # door and every blocked mission would become permanent.
            sets.append("blocked_on=?")
            args.append(blocked_on.strip())
        args.append(mid)
        self._conn.execute(f"UPDATE missions SET {', '.join(sets)} WHERE id=?", args)
        if note:
            self._log(mid, "note", note)
        self._conn.commit()
        return self.get(mid)

    def finish(self, mid: int, note: str = "", ok: bool = True,
               assumed: bool = False) -> Mission:
        m = self.get(mid)
        if m.status not in (OPEN, ASSUMED):
            raise MissionError(
                f"M{mid} is already {m.status}"
                + (f" (a close by assumption, note: {m.close_note!r})"
                   if m.assumed_close else ""))
        kids = [c for c in self.owed_children(mid)]
        if kids:
            raise MissionError(
                "M%d still has open sub-missions: %s. Close those first, or they "
                "disappear from the report while still being owed. A child that "
                "was only PRESUMED done counts here too, so that an assumption "
                "about a piece can never be laundered into a real close of the "
                "whole." % (mid, ", ".join(f"M{k.id}" for k in kids)))
        status = DONE if ok else DROPPED
        if assumed and status == DONE:
            # The prefix is what makes this close tellable apart from a real one
            # a week later. "It is done" and "nobody said anything for eight
            # days" must not be the same row.
            note = ASSUMED_CLOSE_PREFIX + (note or "")
        now = time.time()
        self._conn.execute(
            "UPDATE missions SET status=?, closed_at=?, close_note=?, updated_at=?"
            " WHERE id=?", (status, now, note or "", now, mid))
        self._log(mid, "close", f"{status}: {note}" if note else status)
        self._conn.commit()
        if self.focused() == mid:
            self.set_focus(0)
        return self.get(mid)

    def drop(self, mid: int, note: str) -> Mission:
        return self.finish(mid, note=note, ok=False)

    def wake(self, mid: int, note: str = "") -> Mission:
        """Overturn an assumption of completion. Only an assumption, never a close.

        A real close stays closed -- that rule is the spine of this store. But a
        presumption of completion is a guess about the operator's silence, and a
        guess the operator contradicts must yield. Refusing here would make the
        24h rule a one-way door, which is how a helpful default becomes data
        loss.
        """
        m = self.get(mid)
        if m.status != ASSUMED:
            raise MissionError(
                f"M{mid} is {m.status}, not an assumption"
                + (" (that close was real, and a real close is not reopened; "
                   "open a new mission for the work that remains)"
                   if m.status == DONE else ""))
        now = time.time()
        self._conn.execute(
            "UPDATE missions SET status=?, updated_at=? WHERE id=?", (OPEN, now, mid))
        self._log(mid, "wake", note or "assumption overturned")
        self._conn.commit()
        return self.get(mid)

    def set_focus(self, mid: int) -> str:
        if mid:
            m = self.get(mid)
            if m.status != OPEN:
                raise MissionError(f"M{mid} is {m.status}, so it cannot be the focus")
        self._conn.execute(
            "INSERT INTO mission_state (key, value) VALUES ('focus', ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(int(mid)),))
        self._conn.commit()
        return f"focus is now M{mid}" if mid else "focus cleared"

    def focused(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM mission_state WHERE key='focus'").fetchone()
        return int(row["value"]) if row else 0

    #: Columns added to a table that already shipped. Two sessions and the
    #: market server share this file, so one of them may be running code older
    #: than the other's schema. Read first, write only when something is
    #: genuinely missing: an unconditional ALTER takes a write lock on every
    #: open, which is the failure `ToolStore._ensure_columns` already documents.
    _ADDED_COLUMNS = (
        ("missions", "blocked_on", "TEXT NOT NULL DEFAULT ''"),
    )

    def _ensure_columns(self) -> None:
        """Add columns that predate the running code, idempotently."""
        for table, column, dtype in self._ADDED_COLUMNS:
            try:
                have = {d[1] for d in self._conn.execute(f"PRAGMA table_info({table})")}
            except sqlite3.Error:
                continue
            if not have or column in have:
                continue                 # nothing to do: no write, no lock
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")
                self._conn.commit()
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise

    # -- the silence rule -------------------------------------------------
    def last_activity(self, mid: int) -> float:
        """When this mission was last touched by anything, including a peer.

        Not `updated_at` alone: a mission annotated by another session on the
        same file is being worked, and this process would otherwise presume it
        finished while somebody else was mid-sentence on it. The log is the only
        place a second writer's touch is visible, so silence is measured from
        the later of the two.
        """
        row = self._conn.execute(
            "SELECT MAX(at) AS a FROM mission_log WHERE mission=? AND kind!='assume'",
            (mid,)).fetchone()
        m = self.get(mid)
        stamps = [m.updated_at, m.created_at,
                  (float(row["a"]) if row and row["a"] else 0.0)]
        return max(t for t in stamps if t)

    def assumed_at(self, mid: int) -> float | None:
        """When this mission was presumed complete, or None if it never was."""
        row = self._conn.execute(
            "SELECT MAX(at) AS a FROM mission_log WHERE mission=? AND kind='assume'",
            (mid,)).fetchone()
        return float(row["a"]) if row and row["a"] else None

    def owed_children(self, mid: int) -> list[Mission]:
        """Children still on the projection -- open OR presumed complete.

        Presumed counts. A child is the piece of the parent this store exists to
        keep visible, and letting an *assumption* about the child unlock an
        assumption about the parent would collapse a two-step judgement into one
        silent cascade the first time a session was away for two days.
        """
        return [m for m in self.all("owed") if m.parent == mid]

    def sweep(self, idle_s: float = 0.0, now: float | None = None,
              harden: bool = True) -> dict[str, Any]:
        """Apply the 24h rule. Safe to run often: it is idempotent at one instant.

        Two transitions, and the asymmetry between them is the design:

          * open + silent for `idle_s`  ->  ASSUMED. Still in the report, still
            counted as owed, marked as an assumption, and one `mission_note`
            away from being owed outright. Nothing is lost by this step, which
            is what makes it safe to take on a single day of silence.
          * assumed + the resurrection window elapsed since the ASSUMPTION
            ->  DONE, close note prefixed `assumed:`. The row stays; it leaves
            only the projection. Nothing is lost here either, and the second
            clock is why: with one clock, a mission silent for eight days would
            be assumed and hardened in the same breath and the operator would
            never see the assumption to contradict it.

        A mission with an owed child is never presumed: the child is the piece
        being watched.

        `now` is injected so the clock is testable without waiting a day, and
        `idle_s` so the rule can be tightened or loosened without a code change.
        `updated_at` is deliberately NOT rewritten on assumption: that column
        means "someone last spoke", and overwriting it with a clock would make
        every later report of silence a lie. The assumption's own timestamp goes
        in the log, where events belong.
        """
        idle_s = float(idle_s or IDLE_ASSUMPTION_S)
        now = float(now if now is not None else time.time())
        out: dict[str, Any] = {"assumed": [], "hardened": [], "skipped_parents": [],
                               "waiting": []}

        for m in self.all(OPEN):
            if (m.blocked_on or "").strip():
                # Somebody claimed this aloud -- the operator wrote down what it
                # is waiting for. Silence means "still blocked", and presuming it
                # complete would be the rule eating exactly the missions most
                # likely to be genuinely unfinished.
                out["waiting"].append(m.id)
                continue
            if self.owed_children(m.id):
                # Presumed children block too, on purpose. A child nobody spoke
                # about is not a finished child, and letting an assumption about
                # the piece authorise an assumption about the whole would let one
                # day of silence collapse a two-level judgment into a single
                # silent cascade.
                out["skipped_parents"].append(m.id)
                continue
            if now - self.last_activity(m.id) < idle_s:
                continue
            self._conn.execute(
                "UPDATE missions SET status=? WHERE id=?", (ASSUMED, m.id))
            self._log(m.id, "assume",
                      f"no activity for {(now - self.last_activity(m.id)) / 3600.0:.1f}h: "
                      f"presumed complete", at=now)
            out["assumed"].append(m.id)

        if harden:
            for m in self.all(ASSUMED):
                if (m.blocked_on or "").strip():
                    continue
                born = self.assumed_at(m.id)
                if born is None or (now - born) < RESURRECT_WINDOW_S:
                    continue
                self._conn.execute(
                    "UPDATE missions SET status=?, closed_at=?, close_note=?, updated_at=?"
                    " WHERE id=?",
                    (DONE, now,
                     ASSUMED_CLOSE_PREFIX + "silent for a week after being presumed "
                     "complete. This is an assumption from silence, not a "
                     "confirmation that the work was finished.",
                     now, m.id))
                self._log(m.id, "close",
                          f"assumed: hardened {(now - born) / 86400.0:.1f} days after "
                          f"the assumption", at=now)
                out["hardened"].append(m.id)

        if out["assumed"] or out["hardened"]:
            self._conn.commit()
        return out

    def assumption_report(self, idle_s: float = 0.0,
                          now: float | None = None) -> list[dict[str, Any]]:
        """The presumed-complete missions with the clock made explicit.

        Read by a person or by `mission_show`, never by the prompt: it carries an
        hours count, and an hours count changes every hour -- putting one in the
        prompt would move the cached prefix and re-bill the whole conversation
        behind it, every hour, forever.
        """
        idle_s = float(idle_s or IDLE_ASSUMPTION_S)
        now = float(now if now is not None else time.time())
        out = []
        for m in self.all(ASSUMED):
            born = self.assumed_at(m.id) or now
            out.append({
                "id": m.id, "text": m.text,
                "idle_hours": round((now - self.last_activity(m.id)) / 3600.0, 1),
                "assumed_hours_ago": round((now - born) / 3600.0, 1),
                "days_until_closed": round(max(0.0, RESURRECT_WINDOW_S - (now - born))
                                           / 86400.0, 1),
                "next_step": m.next_step,
            })
        return out

    # -- reads ------------------------------------------------------------
    def get(self, mid: int) -> Mission:
        row = self._conn.execute("SELECT * FROM missions WHERE id=?", (mid,)).fetchone()
        if row is None:
            raise MissionError(f"no mission M{mid}")
        return self._row(row)

    def open_children(self, mid: int) -> list[Mission]:
        """Children with status 'open' exactly -- not the presumed ones.

        Distinct from `owed_children` on purpose: code that asks "is work
        actually still moving under this parent" wants this one, and code that
        asks "may this parent leave the projection" wants the other.
        """
        rows = self._conn.execute(
            "SELECT * FROM missions WHERE parent=? AND status='open' ORDER BY id",
            (mid,)).fetchall()
        return [self._row(r) for r in rows]

    def all(self, status: str | None = None) -> list[Mission]:
        if status == "owed":            # the projection: open and presumed so
            return sorted(self.all(OPEN) + self.all(ASSUMED), key=lambda m: m.id)
        if status:
            rows = self._conn.execute(
                "SELECT * FROM missions WHERE status=? ORDER BY id", (status,)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM missions ORDER BY id").fetchall()
        return [self._row(r) for r in rows]

    def tree(self, status: str | None = OPEN) -> list[Mission]:
        """Roots with their open children attached -- the report's real shape.

        `open` here means "still on the projection": open AND presumed complete.
        Filtering to OPEN alone is what would make the 24h rule a deletion --
        the mission would leave the report the moment it became an assumption,
        and a presumption nobody can see is a silent close.
        """
        rows = self.all(status)
        if status == OPEN:
            # `all(OPEN)` has already filtered to status='open', so filtering
            # again here would be a no-op and the presumed missions would vanish
            # from the report -- which is the silent close this rule is supposed
            # not to be. Ask for the projection, then keep both states.
            rows = self.all("owed")
        by_id = {m.id: m for m in rows}
        roots: list[Mission] = []
        for m in rows:
            if m.parent and m.parent in by_id:
                by_id[m.parent].children.append(m)
            else:
                roots.append(m)
        return roots

    def history(self, mid: int, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT seq, at, kind, note FROM mission_log WHERE mission=?"
            " ORDER BY seq DESC LIMIT ?", (mid, limit)).fetchall()
        return [dict(r) for r in rows]

    # -- the block that rides in the prompt -------------------------------
    def report(self, budget: int = REPORT_BUDGET_CHARS,
               idle_s: float = 0.0) -> str:
        # Read-only: the sweep that would flip states writes to the store, and
        # the prompt is composed every turn. A report that did the sweeping
        # would take a write lock on every turn and move the prompt every time a
        # boundary was crossed. `sweep` is driven from the places that already
        # write -- run start, and the mission tools -- so the state is current
        # where it is acted on, and this stays a pure read.
        whoosh = (f"\n    (the {int((idle_s or IDLE_ASSUMPTION_S) // 3600)}h rule: a mission "
                  f"silent that long is presumed complete, not deleted -- "
                  f"mission_note M<id> takes it back)")
        roots = self.tree(OPEN)
        focused = self.focused()
        head = ("MISSION — what I owe. Read this before acting, and update it when "
                "the answer changes:")
        closed = len(self.all(DONE)) + len(self.all(DROPPED))
        if not roots:
            # The closed count stays in the empty case on purpose. "Nothing is
            # owed" and "this store was never written to" are different facts,
            # and the first version of this printed the same sentence for both
            # -- the same conflation, in miniature, as the bug being fixed.
            return (head
                    + "\n    (nothing open — if the operator asked for "
                      "something, that request IS a mission: record it with "
                      "mission_open before working on it)"
                    + (f"; {closed} closed and on the record" if closed else "")
                    + whoosh)
        lines: list[str] = []
        used = 0
        hidden = 0
        for m in roots:
            chunk = m.line(0) + "".join("\n" + c.line(1) for c in m.children)
            marker = "  <== focus" if m.id == focused else ""
            chunk += marker
            if lines and used + len(chunk) > budget:
                hidden += 1
                continue
            lines.append(chunk)
            used += len(chunk)
        if hidden:
            lines.append(f"    (+{hidden} more open — mission_list reads them)")
        waiting = [m for m in self.all(OPEN) if (m.blocked_on or "").strip()]
        presumed = [m for m in self.all(ASSUMED)]
        assumed_line = ""
        if presumed:
            assumed_line = ("\n    assumed done (no word for a day): "
                            + ", ".join(f"M{m.id}" for m in presumed))
        if waiting:
            assumed_line += ("\n    waiting on something, so not assumed: "
                             + ", ".join(f"M{m.id}" for m in waiting))
        tail = (f"    {len(roots)} open root(s), {closed} closed. "
                f"mission_note updates, mission_close ends one. A new request is a "
                f"line under this, not a replacement for it.{assumed_line}{whoosh}")
        return "\n".join([head, *lines, tail])

    # -- internals --------------------------------------------------------
    def _log(self, mid: int, kind: str, note: str = "",
             at: float | None = None) -> None:
        """Append to the mission's own log. `at` is injectable for the sweep.

        The sweep runs on a clock it may be handed rather than the wall clock, so
        an assumption recorded at a synthetic time has to be written down at that
        same synthetic time -- otherwise the hardening clock in the tests, and in
        any replay, would disagree with the decision that was made.
        """
        self._conn.execute(
            "INSERT INTO mission_log (mission, at, kind, note) VALUES (?,?,?,?)",
            (mid, float(at if at is not None else time.time()), kind, note))

    @staticmethod
    def _row(r: sqlite3.Row) -> Mission:
        keys = set(r.keys())
        return Mission(
            id=int(r["id"]), text=r["text"], parent=int(r["parent"]),
            status=r["status"], next_step=r["next_step"],
            created_at=float(r["created_at"]), updated_at=float(r["updated_at"]),
            closed_at=(float(r["closed_at"]) if r["closed_at"] is not None else None),
            close_note=r["close_note"],
            # Guarded because a store written by an older process on this shared
            # file may not have the column yet, and reading the mission list must
            # never be the thing that breaks.
            blocked_on=(r["blocked_on"] if "blocked_on" in keys else ""),
        )


__all__ = ["Mission", "MissionStore", "MissionError", "OPEN", "ASSUMED",
           "DONE", "DROPPED", "IDLE_ASSUMPTION_S", "RESURRECT_WINDOW_S",
           "ASSUMED_CLOSE_PREFIX",
           "REPORT_BUDGET_CHARS"]
