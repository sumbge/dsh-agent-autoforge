"""Hash-chained event log over the sqlite ledger.



The append-only event log recorded *what happened*. It could not answer a

different question: was any of it rewritten afterwards? Nothing in a row said

so, and 4535 rows were already on disk before anything could. This module adds

the field that makes each row self-witnessing.



Three columns are added to forge_events -- prev_hash, payload_hash, writer_id --

and nothing else about the table changes. Old code that still writes

`INSERT INTO forge_events (timestamp, kind, payload)` keeps working: those rows

land unchained and are reported as such, never silently trusted.



The region that already existed cannot be retroactively chained: the writer of

row 2000 was not recording prev_hash, and no amount of hashing now recovers

what it should have been. So the past is *anchored*, not chained. An anchor

freezes a digest of every legacy row as it stands at anchor time; if any legacy

row is edited later, that digest stops matching and verify() says so. The

distinction between "chained" and "anchored" is the whole point and is never

glossed: anchored rows are detectable-after-the-fact, chained rows are

detectable-individually.



Deliberately not a general log store: no query language, no compaction, no

pluggable backends. append, verify, anchor, lineage_of -- four verbs. Anything

resembling SQL belongs in the sqlite layer underneath.

"""

from __future__ import annotations



import hashlib

import json

import os

import sqlite3

import threading

import time

from typing import Any



GENESIS = "0" * 64

_CHAIN_COLUMNS = ("prev_hash", "payload_hash", "writer_id")

#: One lock per connection object, so two threads sharing a connection
#: serialise instead of corrupting each other's cursors. Not a file lock and
#: not per-database: the thing that is not re-entrant is the connection, so
#: that is what gets the lock. Keyed by id() because a caller may hand this
#: module a connection it got from anywhere -- the CLI opens one, tests open
#: their own -- and every one of them has the same hazard.
_LOCKS: "dict[int, tuple]" = {}
_LOCKS_GUARD = threading.Lock()


#: How long a writer waits for the file lock before giving up.
#: Two agent sessions and the market server share this sqlite file on this
#: machine, and a contended write is a normal event, not an error. Measured
#: with the sqlite default (busy_timeout = 0): three processes appending 120
#: events landed 91, and the 29 that went missing were raised as
#: OperationalError("database is locked") -- lost refusals, silently, which is
#: the one thing an audit log cannot do. With a wait, the same run lands 120.
_BUSY_TIMEOUT_MS = 15000


def _configure(conn, state):
    """Give this connection a wait-on-lock, once, unless the caller chose one.

    Set here rather than at every connect() because the callers are not one
    caller: the CLI opens a connection, tests open their own, and a tool may
    hand this module a connection from anywhere. A shared default of "abandon
    the write immediately" is wrong for a ledger every process on the host
    writes to, and the failure it produces is invisible -- the event is simply
    gone, and verify_chain still says ok.

    A caller who set a timeout explicitly keeps it: PRAGMA reports 0 for an
    untouched connection, and anything else is a decision, not a default.
    """
    if state.get("busy_configured"):
        return
    state["busy_configured"] = True
    try:
        current = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        if not current:
            # WAL first: readers should not block the writer at all, which is
            # the other half of the same problem on this host.
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass        # read-only or in-memory; the timeout still applies
            conn.execute("PRAGMA busy_timeout=%d" % _BUSY_TIMEOUT_MS)
    except sqlite3.DatabaseError:
        # Not worth losing an event over: a connection that cannot answer a
        # PRAGMA will fail on the INSERT with a better message.
        pass


def _write_lock(conn):
    """The lock for this connection (and its one-time state), made once and shared.

    Kept here rather than added to the sqlite3.connect call sites because
    the hazard is not a property of any one caller: measured on this machine,
    16 threads appending through one shared connection landed 5 rows and
    raised 11 exceptions, and `verify_chain` still reported ok -- a silent
    loss, not a detectable corruption. The rows that go missing are mostly
    the refusals, which is the worst thing for an audit log to lose.

    Held only around one append. BEGIN IMMEDIATE below still does the
    cross-process work; this does the in-process work that BEGIN cannot,
    because two threads on one connection are one transaction, not two.
    """
    key = id(conn)
    ent = _LOCKS.get(key)
    # (lock, the connection) both, because id() is only unique among live
    # objects: a connection that is garbage-collected frees its key for the
    # next allocation, and the next allocation could be a new connection.
    # Identity is checked, not assumed, so a reused address makes a new lock
    # instead of two connections sharing one.
    if ent is None or ent[1] is not conn:
        with _LOCKS_GUARD:
            ent = _LOCKS.get(key)
            if ent is None or ent[1] is not conn:
                ent = (threading.RLock(), conn, {})
                _LOCKS[key] = ent
    return ent[0], ent[2]





def rows_of(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT and hand back dicts, whatever the connection's row_factory is.

    This exists because the rest of this module indexes rows by column name, and
    a plain `sqlite3.connect()` returns tuples -- measured on this host, calling
    `verify_chain` on a connection someone else opened raised
    `TypeError: tuple indices must be integers or slices, not str`. For most
    functions that is a bug; for this one it is worse, because the question it
    answers is "was the log rewritten?" and a traceback is not a verdict.
    ToolStore sets `row_factory`, but nothing in this module's contract says a
    caller has one, and an audit log that can only be read by its own writer is
    not an audit log.

    `description` is the names sqlite actually returned, so this works for any
    SELECT without the caller listing columns twice.
    """
    cur = conn.execute(sql, params)
    names = [d[0] for d in (cur.description or ())]
    if not names:
        return []
    if hasattr(cur, "fetchall"):
        raw = cur.fetchall()
    else:                                                     # pragma: no cover
        raw = list(cur)
    out: list[dict] = []
    for r in raw:
        try:
            out.append({n: r[n] for n in names})              # sqlite3.Row
        except (TypeError, IndexError):
            out.append(dict(zip(names, r)))                   # tuple
    return out


def _j(obj: Any) -> str:

    return json.dumps(obj, ensure_ascii=False, default=str, sort_keys=True)





def writer_id(session: str | None = None) -> str:

    """Identify the writer process, not the agent name.



    Two sessions on this host both call themselves "autoforge" and share one

    store; a name is not an identity here. session+pid is.

    """

    sid = session or os.environ.get("AUTOFORGE_SESSION") or "sessionless"

    return f"{sid}@pid{os.getpid()}"





def _digest(kind: str, payload_json: str, timestamp: float, prev_hash: str,

            writer: str) -> str:

    """The exact bytes a row commits to.



    Sorted keys and a fixed separator mean two processes that build the same

    event agree on its hash. Anything not in this string is not covered, which

    is why the writer id is in it: "who wrote it" is part of the claim.

    """

    material = "\x1f".join([

        format(float(timestamp), ".6f"),

        kind,

        payload_json,

        prev_hash,

        writer,

    ])

    return hashlib.sha256(material.encode("utf-8")).hexdigest()





def ensure_schema(conn: sqlite3.Connection) -> list[str]:

    """Add the chain columns if absent. Returns what it added.



    Additive only, on purpose: another session may be running older code

    against this same file right now, and a rewrite of forge_events would

    break it without warning.

    """

    added: list[str] = []

    have = {r[1] for r in conn.execute("PRAGMA table_info(forge_events)")}

    for col in _CHAIN_COLUMNS:

        if col not in have:

            try:

                conn.execute(f"ALTER TABLE forge_events ADD COLUMN {col} TEXT DEFAULT ''")

            except sqlite3.OperationalError as e:

                # Two processes can pass the `have` check at the same moment and

                # both try to add the column; the loser gets "duplicate column".

                # That is the schema converging, not a failure -- and it was

                # being raised straight out of a WRITE path, so a concurrent

                # first use of a fresh ledger turned into a lost event.

                if "duplicate column" not in str(e).lower():

                    raise

            added.append(col)

    conn.execute(

        "CREATE TABLE IF NOT EXISTS chain_anchors ("

        " id INTEGER PRIMARY KEY AUTOINCREMENT,"

        " created_at REAL NOT NULL,"

        " up_to_id INTEGER NOT NULL,"

        " rows INTEGER NOT NULL,"

        " baseline_digest TEXT NOT NULL,"

        " head_hash TEXT NOT NULL,"

        " writer TEXT NOT NULL DEFAULT '',"

        " note TEXT NOT NULL DEFAULT ''"

        ")"

    )

    conn.commit()

    return added





def _row_material(row: dict) -> str:

    return "\x1f".join([str(row["id"]), format(float(row["timestamp"]), ".6f"),

                        row["kind"], row["payload"]])





def _baseline_digest(conn: sqlite3.Connection, up_to_id: int) -> tuple[str, int]:

    """One digest over every legacy row, in id order.



    Not a per-row chain: it cannot say *which* legacy row moved, only that the

    set it was taken over no longer matches. That is the honest limit of

    anchoring, and it is still strictly more than the table had before, which

    was nothing.

    """

    h = hashlib.sha256()

    n = 0

    for row in rows_of(

        conn,

        "SELECT id, timestamp, kind, payload FROM forge_events "

        "WHERE id <= ? AND (payload_hash IS NULL OR payload_hash = '') "

        "ORDER BY id",

        (up_to_id,),

    ):

        h.update(_row_material(row).encode("utf-8"))

        n += 1

    return h.hexdigest(), n





def anchor(conn: sqlite3.Connection, note: str = "", writer: str | None = None) -> dict:

    """Freeze the current state as a baseline. Idempotent in effect, not in rows."""

    ensure_schema(conn)

    up_to = conn.execute("SELECT COALESCE(MAX(id), 0) FROM forge_events").fetchone()[0]

    digest, rows = _baseline_digest(conn, up_to)

    head = _head_hash(conn)

    conn.execute(

        "INSERT INTO chain_anchors (created_at, up_to_id, rows, baseline_digest,"

        " head_hash, writer, note) VALUES (?,?,?,?,?,?,?)",

        (time.time(), up_to, rows, digest, head, writer or writer_id(), note),

    )

    conn.commit()

    return {"up_to_id": up_to, "rows": rows, "baseline_digest": digest, "head_hash": head}





def _head_hash(conn: sqlite3.Connection) -> str:

    """The hash the next chained row must point at.



    Falls back through: last chained row -> newest anchor's head -> GENESIS.

    """

    row = conn.execute(

        "SELECT payload_hash FROM forge_events WHERE payload_hash IS NOT NULL"

        " AND payload_hash != '' ORDER BY id DESC LIMIT 1"

    ).fetchone()

    if row and row[0]:

        return row[0]

    row = conn.execute(

        "SELECT head_hash FROM chain_anchors ORDER BY id DESC LIMIT 1"

    ).fetchone()

    if row and row[0]:

        return row[0]

    return GENESIS





def append_event(conn: sqlite3.Connection, kind: str, payload: dict,

                 session: str | None = None, writer: str | None = None) -> dict:

    """Append one chained event. The only write path into the log.



    BEGIN IMMEDIATE because reading the head and writing the row must not be

    interleaved with another process doing the same -- the hash of my row is a

    claim about the row before it, and a claim made against a stale head is a

    fork, not a chain.

    """

    lock, state = _write_lock(conn)
    with lock:
        _configure(conn, state)
        ensure_schema(conn)

        w = writer or writer_id(session)

        payload_json = _j(payload)

        ts = time.time()

        conn.execute("BEGIN IMMEDIATE")

        try:

            prev = _head_hash(conn)

            digest = _digest(kind, payload_json, ts, prev, w)

            conn.execute(

                "INSERT INTO forge_events (timestamp, kind, payload, prev_hash,"

                " payload_hash, writer_id) VALUES (?,?,?,?,?,?)",

                (ts, kind, payload_json, prev, digest, w),

            )

            conn.commit()

        except Exception:

            conn.rollback()

            raise

        return {"timestamp": ts, "kind": kind, "prev_hash": prev,

                "payload_hash": digest, "writer_id": w}





def verify_chain(conn: sqlite3.Connection) -> dict:
    """Walk the whole log and report what is provable, row by row.

    Two kinds of bad news, deliberately kept apart because they mean different
    things and only one of them is a failure:

    breaks -- a row's contents do not match its own hash, a link does not point
        at its predecessor, or the anchored legacy region changed. Any of these
        is evidence the log was rewritten, and nothing about this store's
        multi-process setup excuses them.

    gaps -- a row with no hash at all, written after chaining began. On this
        host that is the expected state while another session is still running
        code from before the change, and conflating it with a break is what
        makes a verifier into noise nobody reads. Reported, counted, not failed.

    ok therefore means "nothing was rewritten", not "every row is chained".
    """
    ensure_schema(conn)
    # Anchors as dicts, via rows_of rather than a row_factory this module does
    # not control. See rows_of for why that matters here.
    anchors = rows_of(conn, "SELECT id, created_at, up_to_id, rows,"
                            " baseline_digest, head_hash FROM chain_anchors"
                            " ORDER BY id")
    breaks: list[dict] = []
    gaps: list[dict] = []
    chained = 0
    checked_legacy = 0
    unchained: list[int] = []
    prev = ""
    started = False

    # Ordered, named, and read through rows_of: the walk below is the thing
    # that decides whether this log is trustworthy, so it must not depend on
    # the caller having passed a connection with the right row_factory.
    for row in rows_of(
        conn,
        "SELECT id, timestamp, kind, payload, prev_hash, payload_hash,"
        " writer_id FROM forge_events ORDER BY id",
    ):
        if not row["payload_hash"]:
            unchained.append(row["id"])
            if started:
                gaps.append({"row": row["id"], "kind": "unchained_row",
                             "writer": row["writer_id"] or "",
                             "detail": "written after chaining began, carries no hash"})
            continue
        if not started:
            # The first chained row starts from whatever it points at. Only two
            # things are legitimate: genesis, for a log with nothing before it,
            # or the head of an anchor that was already frozen. Anything else is
            # a row inserted with a fabricated starting point, which is a break.
            started = True
            head = row["prev_hash"] or ""
            allowed = {GENESIS, ""} | {a["head_hash"] for a in anchors}
            if head not in allowed:
                breaks.append({"row": row["id"], "kind": "orphan_chain_start",
                               "writer": row["writer_id"] or "",
                               "detail": "first chained row does not start at genesis"
                                         " or at any anchor head"})
            expected_prev = head
        else:
            expected_prev = prev
        expect = _digest(row["kind"], row["payload"], row["timestamp"],
                         row["prev_hash"] or "", row["writer_id"] or "")
        if expect != row["payload_hash"]:
            breaks.append({"row": row["id"], "kind": "payload_hash_mismatch",
                           "writer": row["writer_id"] or "",
                           "detail": "row contents do not match the hash stored on the row"})
        if (row["prev_hash"] or "") != expected_prev:
            breaks.append({"row": row["id"], "kind": "broken_link",
                           "writer": row["writer_id"] or "",
                           "detail": "prev_hash does not equal the previous row's payload_hash"})
        prev = row["payload_hash"]
        chained += 1

    # Every anchor's baseline is the digest over the legacy rows up to it, and
    # the anchors' bounds only ever move forward -- so all of them are prefixes
    # of one ordered walk. Asking for each in a separate query hashed the same
    # rows again and again: measured on this ledger, 74 anchors over 5,085
    # legacy rows produced 371,045 `_row_material` calls, and the whole walk
    # took 3.6 seconds *per request*, because the self-report rendered into
    # every request's system prompt reads the store. The rows are identical
    # every time, so the cost bought nothing.
    #
    # `hashlib` can copy its own state, so one pass is enough: walk the legacy
    # rows in id order, snapshotting the digest at each anchor's bound. Same
    # rows, same order, same function -- the digests are bit-for-bit the ones
    # the per-anchor queries produced, and a test holds that claim down.
    order = sorted(range(len(anchors)), key=lambda i: anchors[i]["up_to_id"])
    taken: dict[int, tuple[str, int]] = {}
    h = hashlib.sha256()
    seen = 0
    it = iter(rows_of(
        conn,
        "SELECT id, timestamp, kind, payload FROM forge_events"
        " WHERE payload_hash IS NULL OR payload_hash = ''"
        " ORDER BY id",
    ))
    nxt = next(it, None)
    for i in order:
        bound = anchors[i]["up_to_id"]
        while nxt is not None and nxt["id"] <= bound:
            h.update(_row_material(nxt).encode("utf-8"))
            seen += 1
            nxt = next(it, None)
        taken[i] = (h.copy().hexdigest(), seen)
    for i, a in enumerate(anchors):
        digest, n = taken[i]
        checked_legacy += n
        if n and digest != a["baseline_digest"]:
            breaks.append({"row": a["up_to_id"], "kind": "legacy_region_modified",
                           "writer": "",
                           "detail": f"rows anchored at id {a['up_to_id']} differ from the"
                                     f" baseline taken there"})

    # Truncation: a chain walking forward cannot see its own tail being cut,
    # because whatever row is last always looks like a valid last row. An anchor
    # is what makes that visible -- it recorded a head, and a head that is no
    # longer in the log means rows after that point were removed. One anchor,
    # then, is not ceremony; it is the only defence against a silent truncate.
    for a in anchors:
        if a["head_hash"] and a["head_hash"] != GENESIS:
            still_there = conn.execute(
                "SELECT 1 FROM forge_events WHERE payload_hash=?", (a["head_hash"],)
            ).fetchone()
            if still_there is None:
                breaks.append({"row": a["up_to_id"], "kind": "anchored_head_missing",
                               "writer": "",
                               "detail": "the head recorded by an anchor is no longer in"
                                         " the log: rows were removed after it was taken"})

    writers = sorted({r[0] for r in conn.execute(
        "SELECT DISTINCT writer_id FROM forge_events"
        " WHERE payload_hash != '' AND writer_id != ''")})
    return {
        "ok": not breaks and chained > 0,
        "chained_rows": chained,
        "legacy_rows_anchored": checked_legacy,
        "anchors": len(anchors),
        "unchained_rows": len(unchained),
        "unchained_after_chaining_began": len(gaps),
        "chained_writers": writers,
        "head_hash": _head_hash(conn),
        "breaks": breaks,
        "gaps": gaps,
        "findings": breaks + gaps,
    }


def lineage_of(conn: sqlite3.Connection, tool: str, depth: int = 3) -> dict:

    """The derivation edges recorded in the log for one tool.



    Only what the writers actually wrote down: forge_done says what was built,

    an evolve event says what it was bred from, a veto says who refused it.

    Nothing is inferred from name or need-text similarity -- that would be a

    guess wearing the word "lineage".

    """

    edges: list[dict] = []

    nodes = {tool}

    frontier = {tool}

    for _ in range(max(1, depth)):

        nxt: set[str] = set()

        for t in frontier:

            for row in rows_of(

                conn,

                "SELECT id, timestamp, kind, payload FROM forge_events"

                " WHERE payload LIKE ? ORDER BY id",

                (f'%"{t}"%',),

            ):

                try:

                    p = json.loads(row["payload"])

                except Exception:

                    continue

                for key in ("derived_from", "parent_version", "derives_from"):

                    if p.get("tool") == t or p.get("name") == t:

                        src = p.get(key)

                        if isinstance(src, str) and src:

                            edges.append({"from": src, "to": t, "kind": row["kind"],

                                          "row": row["id"], "rel": key})

                            nxt.add(src)

                if p.get("tool") == t and p.get("vetoed"):

                    for v in p["vetoed"] or []:

                        edges.append({"from": str(v), "to": t, "kind": "vetoed",

                                      "row": row["id"], "rel": "vetoed_by"})

                        nxt.add(str(v))

        if not nxt:

            break

        nodes |= nxt

        frontier = nxt - nodes | nxt

    return {"tool": tool, "nodes": sorted(nodes), "edges": edges}

