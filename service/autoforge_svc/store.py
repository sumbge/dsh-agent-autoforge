"""SQLite-backed persistence for the tool registry.

Every tool version, every forge attempt, every dependency link — survives
restarts. Auto-creates the DB on first use.

Schema is a flat tool table plus an append-only event log. JSON columns for
structured fields avoid schema migrations while keeping the critical fields
(first-class columns) queryable.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import functools
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools.spec import ToolSpec, ToolState, ToolStats, TriggerProbe, normalise_parameters
from . import chaining

_DEFAULT_DB = "autoforge.db"

#: How long a writer waits for the ledger's write lock before giving up.
#: Two agent sessions and a market server share this sqlite file on this machine,
#: and a contended write is a normal event, not an error.
_SQLITE_BUSY_TIMEOUT_S = 15.0


# -- column-level constants (used by schema generation) -------------
_COLUMNS = dict(
    name="TEXT NOT NULL",
    version="INTEGER DEFAULT 1",
    description="TEXT NOT NULL DEFAULT ''",
    parameters="TEXT NOT NULL DEFAULT '{}'",
    code="TEXT NOT NULL DEFAULT ''",
    entry="TEXT NOT NULL DEFAULT ''",
    source="TEXT NOT NULL DEFAULT 'human'",
    generator="TEXT NOT NULL DEFAULT ''",
    probes="TEXT NOT NULL DEFAULT '[]'",
    #: The concrete valid call the tool declared for itself. Persisted so a gate
    #: that runs *after* the forge -- a re-audit, or the claim gate when a tool
    #: is re-verified on load -- has something real to run it on. Its absence on
    #: every row written before 2026-10-02 is why the claim gate reports those
    #: tools as un-probed rather than passing them: `_infer_args` invents
    #: arguments, and re-running a real gate on invented arguments would seal a
    #: verdict nobody earned.
    sample_call="TEXT NOT NULL DEFAULT '{}'",
    #: The substring a correct answer to `sample_call` must contain. Stored
    #: beside the call because the two are ONE exam: a persisted call with no
    #: expectation can only be re-run, never re-judged, and a re-audit that can
    #: only ask "did it run" is exactly the vacuity `check_execution` exists to
    #: end. Absent on every row written before 2026-10-24, which is why those
    #: read as un-probed rather than passed.
    sample_expect="TEXT NOT NULL DEFAULT ''",
    effect_signature="TEXT NOT NULL DEFAULT ''",
    state="TEXT NOT NULL DEFAULT 'draft'",
    tags="TEXT NOT NULL DEFAULT '[]'",
    cost_hint="TEXT NOT NULL DEFAULT 'cheap'",
    created_at="REAL NOT NULL",
    verification="TEXT NOT NULL DEFAULT '{}'",
    stats="TEXT NOT NULL DEFAULT '{}'",
    old_versions="TEXT NOT NULL DEFAULT '[]'",
    updated_at="REAL NOT NULL",
)
_DDL = (
    "CREATE TABLE IF NOT EXISTS tools (\n"
    + "\n".join(f"    {name} {dtype}," for name, dtype in _COLUMNS.items())
    + "\n    PRIMARY KEY (name)\n);"
    "\nCREATE TABLE IF NOT EXISTS forge_events ("
    "\n    id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "\n    timestamp REAL NOT NULL,"
    "\n    kind TEXT NOT NULL,"
    "\n    payload TEXT NOT NULL DEFAULT '{}'"
    "\n);"
    # Version history, content-addressed. Two tables because they answer two
    # different questions: version_blobs holds each distinct body of code once,
    # keyed by its sha256, so identical code saved under two names is stored
    # once and provably identical; tool_versions holds the timeline -- which
    # sha a tool was on, from which parent, when. old_versions on the tools row
    # stays for backward compatibility, but it only ever filled on an evolve
    # win, which is why 114 tools had 0 archived versions: a version that was
    # live and got replaced outside evolve left no trace at all.
    "\nCREATE TABLE IF NOT EXISTS version_blobs ("
    "\n    sha TEXT NOT NULL PRIMARY KEY,"
    "\n    code TEXT NOT NULL,"
    "\n    created_at REAL NOT NULL"
    "\n);"
    "\nCREATE TABLE IF NOT EXISTS tool_versions ("
    "\n    tool TEXT NOT NULL,"
    "\n    version INTEGER NOT NULL,"
    "\n    blob_sha TEXT NOT NULL,"
    "\n    parent_version INTEGER,"
    "\n    verification TEXT NOT NULL DEFAULT '{}',"
    "\n    archived_at REAL NOT NULL,"
    # Who wrote this version. Added after the table existed, so it is applied by
    # `_ensure_columns` rather than by editing the DDL alone: CREATE TABLE IF NOT
    # EXISTS does nothing to a table that is already there, which is exactly how
    # a migration gets believed without ever running. No foreign key and no
    # DEFAULT beyond '': a row written before this column existed is *unknown*,
    # not someone else's, and every reader here treats '' as unknown.
    "\n    session TEXT NOT NULL DEFAULT '',"
    "\n    PRIMARY KEY (tool, version)"
    "\n);"
    "\nCREATE TABLE IF NOT EXISTS dependencies ("
    "\n    tool TEXT NOT NULL,"
    "\n    depends_on TEXT NOT NULL,"
    "\n    PRIMARY KEY (tool, depends_on)"
    "\n);"
    "\nCREATE TABLE IF NOT EXISTS baselines ("
    "\n    tool TEXT NOT NULL PRIMARY KEY,"
    "\n    frozen_at REAL NOT NULL,"
    "\n    baseline TEXT NOT NULL DEFAULT '{}'"
    "\n);"
    "\nCREATE TABLE IF NOT EXISTS topologies ("
    "\n    id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "\n    timestamp REAL NOT NULL,"
    "\n    task TEXT NOT NULL DEFAULT '',"
    "\n    topology TEXT NOT NULL DEFAULT '{}',"
    "\n    fitness REAL NOT NULL DEFAULT 0.0,"
    "\n    trials INTEGER NOT NULL DEFAULT 0"
    "\n);"
    # Deliberate long-term memory: facts the agent chooses to keep, as opposed
    # to the event log, which records what happened whether it meant to or not.
    # A memory table without this distinction would just be a second ledger.
    "\nCREATE TABLE IF NOT EXISTS memory ("
    "\n    key TEXT NOT NULL PRIMARY KEY,"
    "\n    value TEXT NOT NULL DEFAULT '',"
    "\n    tags TEXT NOT NULL DEFAULT '[]',"
    "\n    created_at REAL NOT NULL,"
    "\n    updated_at REAL NOT NULL,"
    "\n    recalls INTEGER NOT NULL DEFAULT 0"
    "\n);"
    # Skill usage. The markdown files own the *content* -- they are what the
    # agent reads and what a human edits. This table owns the *history*: how
    # often each skill was actually loaded, which is the signal the router
    # scores on. Two sources of truth by design, split along that line:
    # re-scanning the directory refreshes content and never resets counters.
    "\nCREATE TABLE IF NOT EXISTS skills ("
    "\n    name TEXT NOT NULL PRIMARY KEY,"
    "\n    path TEXT NOT NULL DEFAULT '',"
    "\n    source TEXT NOT NULL DEFAULT 'user',"
    "\n    description TEXT NOT NULL DEFAULT '',"
    "\n    when_to_use TEXT NOT NULL DEFAULT '',"
    "\n    tags TEXT NOT NULL DEFAULT '[]',"
    "\n    loads INTEGER NOT NULL DEFAULT 0,"
    "\n    last_loaded REAL,"
    "\n    created_at REAL NOT NULL,"
    "\n    updated_at REAL NOT NULL"
    "\n);"
)


def _now() -> float:
    return time.time()


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _unjson(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _probes_from_list(items: list[dict]) -> list[TriggerProbe]:
    return [
        TriggerProbe(
            query=p.get("query", ""),
            expect=p.get("expect", "call"),
            negative_query=p.get("negative_query"),
        )
        for p in items
        if isinstance(p, dict) and p.get("query")
    ]


# ---------------------------------------------------------------------------
@dataclass
class ToolRecord:
    """Flat persisted representation. Converted to/from ToolSpec."""

    name: str
    version: int
    description: str
    parameters: dict[str, Any]
    code: str
    entry: str
    source: str
    generator: str
    probes: list[TriggerProbe]
    effect_signature: str
    #: The valid call the tool declared, persisted so a gate that runs after the
    #: forge has something real to run it on. Defaulted rather than required:
    #: every row written before 2026-10-02 has none, and a missing field must
    #: read as "un-probed", never as "no sample needed".
    sample_call: dict[str, Any]
    #: What a correct answer to `sample_call` must contain. Kept as its own
    #: field, not folded into `verification`: this is the *declaration*, and
    #: `verification` is what the battery concluded from it. Losing the
    #: declaration means a later re-audit cannot re-judge, only re-run.
    #: Defaulted rather than required, for the rows written before it existed.
    sample_expect: str
    state: ToolState
    tags: list[str]
    cost_hint: str
    created_at: float
    verification: dict[str, Any]
    stats: dict[str, Any]
    old_versions: list[dict[str, Any]]
    updated_at: float

    @classmethod
    def from_spec(cls, spec: ToolSpec) -> "ToolRecord":
        return cls(
            name=spec.name,
            version=1,
            description=spec.description,
            parameters=spec.parameters,
            code=spec.code,
            entry="",
            source=spec.source,
            generator=spec.generator,
            probes=spec.probes,
            effect_signature=spec.effect_signature,
            sample_call=dict(spec.sample_call or {}),
            sample_expect=spec.sample_expect or "",
            state=spec.state,
            tags=spec.tags,
            cost_hint=spec.cost_hint,
            created_at=spec.created_at,
            verification=spec.verification,
            stats=spec.stats.to_dict(),
            old_versions=[],
            updated_at=_now(),
        )

    def to_spec(self, fn=None, runner=None) -> ToolSpec:
        st = ToolStats()
        for k, v in self.stats.items():
            # `hasattr` is not enough: `success_rate` and `trigger_rate` are
            # read-only properties, so `hasattr` says yes and `setattr` raises
            # "property has no setter". `to_dict` writes them (they are useful
            # in a report) and the constructor cannot take them back, so the
            # round trip is asymmetric -- and the failure lands here, on load,
            # where it silently costs every persisted tool. Check that the
            # attribute is actually assignable.
            if not hasattr(st, k):
                continue
            if isinstance(getattr(type(st), k, None), property):
                continue
            setattr(st, k, v)
        spec = ToolSpec(
            name=self.name,
            description=self.description,
            parameters=normalise_parameters(self.parameters),
            fn=fn or (lambda **_: ""),
            runner=runner,
            code=self.code,
            source=self.source,
            generator=self.generator,
            probes=self.probes,
            effect_signature=self.effect_signature,
            sample_call=dict(self.sample_call or {}),
            sample_expect=self.sample_expect or "",
            state=self.state,
            stats=st,
            verification=self.verification,
            tags=self.tags,
            cost_hint=self.cost_hint,
            created_at=self.created_at,
        )
        return spec


def _default_home() -> str:
    """Where state lives when AUTOFORGE_HOME is unset.

    Per-user and absolute, not the current directory. A cwd-relative database
    means the agent's memory silently resets whenever it is launched from a
    different directory, which reads to the agent as "I have no memory" rather
    than "I looked in the wrong place". The env var still overrides.

    The fallback is dotted, and that is load-bearing rather than cosmetic.
    `LOCALAPPDATA` and `XDG_DATA_HOME` are outside every directory Python
    searches for imports, so a data directory named `autoforge` under them is
    only a data directory. The home directory is not: a process whose cwd (or
    sys.path[0]) is `~` imports this package *by that name*, and a plain
    `~/autoforge` is then a namespace package that shadows the real one --
    importing successfully, as an empty directory, with no `__init__.py`.
    Measured on 2026-09-16: the sandbox's env_allow dropped LOCALAPPDATA, so a
    contained job that built a store created `~/autoforge`; from then on
    `pythonw -m autoforge` died with `cannot import name '__version__'` and the
    CREATE_NO_WINDOW hook in `__init__.py` never ran, which is why every
    subprocess had been flashing a console window. `.autoforge` cannot shadow
    anything, so the failure cannot come back this way.
    """
    env = os.environ.get("AUTOFORGE_HOME")
    if env:
        return env
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    if base:
        return os.path.join(base, "autoforge")
    return os.path.join(os.path.expanduser("~"), ".autoforge")


# ---------------------------------------------------------------------------
class ToolStore:
    """SQLite-backed tool registry persistence."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = str(db_path or _DEFAULT_DB)
        home = _default_home()
        if not os.path.isabs(self.db_path):
            self.db_path = os.path.join(home, self.db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # One connection, used from several threads. That is deliberate -- a
        # store lives inside an agent that calls it from tool threads -- but it
        # needs two things sqlite does not do for you:
        #
        #   * a lock, because a sqlite3.Connection is not re-entrant. Its
        #     cursors are shared state: while one thread is iterating a SELECT
        #     (the head-hash read in chaining.append_event), a second thread
        #     issuing another statement on the same connection makes the first
        #     one fail. Measured before this lock existed: 16 threads writing
        #     this ledger landed 4 rows and raised 12 exceptions, and the rows
        #     that vanished were mostly the *refusals* -- the worst thing to
        #     lose from an audit log, and the thing an audit log exists for;
        #   * WAL with a busy timeout, so a second process (another agent
        #     session, toolmarket on the same box) waits for the write lock
        #     instead of failing at the moment something is being recorded.
        self._lock = threading.RLock()
        # tool name -> session, for writes this process made. The persisted copy
        # is the `session` column; this is only the in-process fast path.
        self._last_writer: dict[str, str] = {}
        self._last_writer_at: dict[str, float] = {}
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False,
                                     timeout=float(_SQLITE_BUSY_TIMEOUT_S))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=%d"
                           % int(_SQLITE_BUSY_TIMEOUT_S * 1000))
        # Order matters, and so does the retry. Switching the journal mode needs
        # a lock the busy handler does not always wait on: on a runner where two
        # interpreters open a brand new file at the same instant, the second one
        # raised "database is locked" from this very statement with a 15s busy
        # timeout already set. Retried here, briefly, because losing the race to
        # create the database is not a reason to lose the write.
        self._set_wal_with_retry()
        self._conn.executescript(_DDL)
        self._ensure_columns()

    def _set_wal_with_retry(self, attempts: int = 8) -> None:
        """`journal_mode=WAL`, waiting out another process creating the file.

        Only this pragma is retried: it is the one statement in the constructor
        that takes an exclusive lock. Everything else is a normal write and the
        busy timeout already covers it.
        """
        for i in range(attempts):
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or i == attempts - 1:
                    raise
                time.sleep(0.05 * (i + 1))

    #: Columns added to a table that already shipped. SQLite has no
    #: "ADD COLUMN IF NOT EXISTS", so each is attempted and the duplicate-column
    #: error is the success path. Kept as data rather than as a migration script
    #: so a store created before the column and a store created after converge on
    #: the same shape -- two sessions on this machine are reading this file right
    #: now, and one of them may well be older code than the other.
    _ADDED_COLUMNS = (
        ("tool_versions", "session", "TEXT NOT NULL DEFAULT ''"),
        # The declared valid call, so a gate that runs after the forge has real
        # arguments to run on. Applied here rather than by editing the DDL alone:
        # CREATE TABLE IF NOT EXISTS does nothing to a table that is already
        # there, and this table is 131 rows deep on this machine.
        ("tools", "sample_call", "TEXT NOT NULL DEFAULT '{}'"),
        # The expectation that makes `sample_call` an exam rather than a demo.
        # Same reason as the row above: an existing table is not touched by
        # `CREATE TABLE IF NOT EXISTS`, and the rows already in it have a call
        # with nothing to judge it against.
        ("tools", "sample_expect", "TEXT NOT NULL DEFAULT ''"),
    )

    def _ensure_columns(self) -> None:
        """Add columns that predate the running code, idempotently.

        Read first, write only if something is actually missing. The obvious
        version -- attempt the ALTER and treat "duplicate column" as success --
        makes EVERY open of the store take a write lock, and this store is opened
        by two sessions and a market server on the same file. Measured: with that
        version, `test_two_processes_on_one_file` failed roughly one run in five
        with `database is locked` raised from `PRAGMA journal_mode=WAL` at open.
        A migration that has already run costs nothing and must cost nothing.
        """
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

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- tools ------------------------------------------------------------
    def tools_written_by_other_sessions(self) -> list[str]:
        """Tool names whose CURRENT version was written by a session that is not mine.

        Reads the `session` stamp on the version timeline. Two things it is
        deliberately careful about:

          * only the LATEST version per tool counts. A tool I rewrote after a peer
            touched it is mine now, and reporting it as theirs would send the
            agent to negotiate over something it already owns;
          * session='' is unknown, not foreign. Every tool forged before the
            column existed carries it, so treating it as someone else's would
            make the entire existing shelf -- 114 rows on this host -- look like a
            peer's property.

        This is a report, not the gate. The gate is inside `save_tool`. Every
        earlier version of this method was a stub that queried a table that does
        not exist and returned [] -- which looked like "no collisions" rather
        than like a broken query, and that is the failure mode this file keeps
        running into.
        """
        mine = os.environ.get("AUTOFORGE_SESSION") or ""
        if not mine:
            return []
        try:
            rows = self._conn.execute(
                "SELECT t.tool AS tool, t.session AS session FROM tool_versions t"
                " WHERE t.version = (SELECT MAX(x.version) FROM tool_versions x"
                "                     WHERE x.tool = t.tool)").fetchall()
        except sqlite3.Error:
            return []
        return sorted({str(r["tool"]) for r in rows
                       if str(r["session"] or "") and str(r["session"]) != mine})

    def _foreign_write_age(self, tool: str, mine: str, body: str = "") -> float | None:
        """Seconds since another session last wrote this tool, or None.

        The backstop for the gap a live lane cannot cover: a lane is held only
        while a write is in flight, so two sessions a few seconds apart never
        contend for it. Measured end to end before this existed: session A saved
        a tool, released the lane, session B saved the same name, and B replaced
        A's work with no error anywhere.

        The ROW is the truth and is always read; the in-process cache is only a
        fallback for a table this code cannot read (a store older than the
        column). An earlier version of this consulted the cache first and
        returned on a hit -- and since the cache holds this process's own writes,
        it short-circuited the query that would have seen the other session. A
        row with session='' is a write from before the column existed: unknown,
        not someone else's, so it refuses nothing.
        """
        who, when = "", None
        try:
            row = self._conn.execute(
                "SELECT session, archived_at FROM tool_versions WHERE tool=?"
                " ORDER BY version DESC LIMIT 1", (tool,)).fetchone()
            if row is not None:
                who, when = str(row["session"] or ""), row["archived_at"]
        except sqlite3.Error:
            row = None
        if not who and tool in self._last_writer:
            who, when = self._last_writer[tool], self._last_writer_at.get(tool)
        if not who or who == mine or when is None:
            return None
        if body:
            # Same body, nobody is losing anything: re-saving identical code under
            # the same name is a no-op and refusing it would only push a caller
            # into allow_overwrite for no reason. A different body is the case
            # that silently destroyed a peer's work.
            import hashlib
            sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
            try:
                blob = self._conn.execute(
                    "SELECT blob_sha FROM tool_versions WHERE tool=?"
                    " ORDER BY version DESC LIMIT 1", (tool,)).fetchone()
            except sqlite3.Error:
                blob = None
            if blob is not None and blob["blob_sha"] == sha:
                return None
        return max(0.0, _now() - float(when))

    #: Seconds after which another session's write stops refusing this one. It is
    #: not a lock timeout: the point is to make *contention* visible, not to
    #: forbid a deliberate override, and the ledger keeps every version anyway.
    FOREIGN_WRITE_GRACE_S = 900.0

    def save_stats(self, spec: ToolSpec) -> None:
        """Write back a tool's counters and nothing else.

        `save_tool` is a judgement: it can refuse an overwrite, it archives a
        version, it replaces code. A call is not a judgement -- it is one row of
        usage that the listing reads. Using save_tool for it meant every call
        rewrote the whole row with whatever this process happened to hold in
        memory, so two hosts on one database could flip a tool's `state` back
        to whatever they last loaded. Measured 2026-10-25: two autoforge_svc
        processes (desktop host and the web host) sharing .data/autoforge.db,
        with a retire landing in one of them.

        So: one UPDATE, counters only, and only when a counter actually moved.
        A no-op write is not free -- it takes the write lock and it bumps
        `updated_at`, which is what a human reads to see when a tool last
        changed.
        """
        try:
            self._conn.execute(
                "UPDATE tools SET stats=? WHERE name=?",
                (json.dumps(spec.stats.to_dict(), ensure_ascii=False), spec.name))
            self._conn.commit()
        except sqlite3.Error:
            pass

    def save_tool(self, spec: ToolSpec, *, allow_overwrite: bool = False) -> None:
        """Record a tool.

        Refuses, by default, to replace a tool another session just wrote. The
        insert below is an INSERT OR REPLACE, so before this check a second
        session forging a colliding name silently destroyed the first session's
        tool: no error, no row left to compare against, and the ledger recording
        two forges that each looked successful. Two sessions on this machine
        independently forged the same tools more than once (see
        docs/DUPLICATE_NEEDS.md), so this is not a hypothetical collision.

        A refusal is a refusal, not a merge: choosing between two tools with the
        same name is a judgement about behaviour, and the place to make it is
        `evolve_tool` or an explicit `allow_overwrite=True` by a session that has
        looked at both. Hiding that choice inside an INSERT was the bug.
        """
        from .lanes import LaneRefused, LaneMissing, claim, release, resource

        mine = os.environ.get("AUTOFORGE_SESSION") or ""
        held = None
        if not allow_overwrite:
            # The backstop first: a live lane only covers a write in flight, and
            # two sessions that save the same name a few seconds apart never
            # contend for one. This is the check that closing that gap required
            # the `session` column for.
            age = self._foreign_write_age(spec.name, mine, spec.code or "")
            if age is not None and age <= self.FOREIGN_WRITE_GRACE_S:
                raise ValueError(
                    f"refusing to save tool {spec.name!r}: another session wrote "
                    f"that name {age:.0f}s ago and its version is the one on "
                    f"record. Evolve it, save under another name, or pass "
                    f"allow_overwrite=True to replace it deliberately.")
            try:
                held = claim(resource(f"tool/{spec.name}"), name="store",
                             why=f"forging tool {spec.name}")
            except LaneRefused as exc:
                raise ValueError(
                    f"refusing to save tool {spec.name!r}: {exc}. A different "
                    f"session is writing that name; evolve it or save under "
                    f"another name") from exc
            except LaneMissing:
                held = None
            except Exception:                # noqa: BLE001 - a lane disabled on
                held = None                  # this host must not block saving
        try:
            record = ToolRecord.from_spec(spec)
            self._conn.execute(
                """INSERT OR REPLACE INTO tools
                (name, version, description, parameters, code, entry,
                 source, generator, probes, effect_signature, sample_call,
                 sample_expect,
                 state, tags, cost_hint, created_at, verification, stats,
                 old_versions, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.name, record.version, record.description,
                    _j(record.parameters), record.code, record.entry,
                    record.source, record.generator, _j([
                        {"query": p.query, "expect": p.expect, "negative_query": p.negative_query}
                        for p in record.probes
                    ]),
                    record.effect_signature, _j(record.sample_call or {}),
                    record.sample_expect or "",
                    record.state.value,
                    _j(record.tags), record.cost_hint,
                    record.created_at, _j(record.verification),
                    _j(record.stats), _j(record.old_versions),
                    record.updated_at,
                ),
            )
            self._conn.commit()
        finally:
            if held is not None:
                # Released immediately: this lane marks the write in progress,
                # not the tool's existence forever. A permanent claim would make
                # the first forge of a name block every later evolve of it, which
                # is the opposite of what the lane is for.
                release(held.target, session=held.session)
        # Every save is a version. Doing it here rather than at each call site is
        # the difference between "history is what someone remembered to archive"
        # and "history is what happened".
        self.record_version(record.name, record.code, record.verification,
                            session=mine or None)

    def load_all_tools(self) -> dict[str, ToolRecord]:
        rows = self._conn.execute(
            "SELECT * FROM tools ORDER BY name"
        ).fetchall()
        result: dict[str, ToolRecord] = {}
        for r in rows:
            try:
                result[r["name"]] = ToolRecord(
                    name=r["name"],
                    version=r["version"],
                    description=r["description"],
                    parameters=_unjson(r["parameters"]),
                    code=r["code"],
                    entry=r["entry"],
                    source=r["source"],
                    generator=r["generator"],
                    probes=_probes_from_list(_unjson(r["probes"])),
                    effect_signature=r["effect_signature"],
                    sample_call=_unjson(r["sample_call"]) if "sample_call" in r.keys() else {},
                    # `r.keys()` rather than a direct read: a store opened by
                    # older code (or a row read before the migration ran) has
                    # no such column, and a missing expectation must read as
                    # "un-probed" rather than raise on load and cost the tool.
                    sample_expect=(r["sample_expect"] or "") if "sample_expect" in r.keys() else "",
                    state=ToolState(r["state"]),
                    tags=_unjson(r["tags"]),
                    cost_hint=r["cost_hint"],
                    created_at=r["created_at"],
                    verification=_unjson(r["verification"]),
                    stats=_unjson(r["stats"]),
                    old_versions=_unjson(r["old_versions"]),
                    updated_at=r["updated_at"],
                )
            except Exception:  # noqa: BLE001
                continue  # skip corrupted rows
        return result
    def delete_tool(self, name: str) -> None:
        self._conn.execute("DELETE FROM tools WHERE name=?", (name,))
        self._conn.execute("DELETE FROM dependencies WHERE tool=? OR depends_on=?",
                           (name, name))
        self._conn.commit()

    # -- versions ---------------------------------------------------------
    def archive_version(self, name: str, code: str, verification: dict[str, Any]) -> None:
        """Push the current code+verification into old_versions for rollback."""
        row = self._conn.execute("SELECT old_versions, version FROM tools WHERE name=?",
                                 (name,)).fetchone()
        if row is None:
            return
        versions = _unjson(row["old_versions"])
        versions.append({
            "version": row["version"],
            "code": code,
            "verification": verification,
            "archived_at": _now(),
        })
        # Record the outgoing body at the version it actually held, before the
        # bump. In the evolve path save_tool already recorded this exact body at
        # this number, so this is a no-op there; it matters for the caller that
        # archives a body the store never saw, which is the case that used to
        # end with the code in a JSON column and nowhere else.
        self.record_version(name, code, verification, version=row["version"])
        self._conn.execute(
            "UPDATE tools SET old_versions=?, version=version+1, updated_at=? WHERE name=?",
            (_j(versions), _now(), name),
        )
        self._conn.commit()

    # -- versions (content-addressed) --------------------------------------
    def record_version(self, tool: str, code: str, verification: dict[str, Any] | None = None,
                       parent_version: int | None = None,
                       version: int | None = None,
                       session: str | None = None) -> dict[str, Any]:
        """Record that `tool` held `code`, and give it the next number if it is new.

        Bodies are stored once by sha256, so history costs a row per distinct
        body rather than a full copy per tool per save. The number comes from the
        timeline, never from the caller: a ToolSpec carrying version=1 through a
        breed would otherwise reset the counter, which is how a tool reaches
        version 1 four times and its history becomes unreadable.

        Called on every save_tool. That placement is the fix for 114 tools with
        0 archived versions -- before, only the evolve path archived anything, so
        a body replaced any other way left no trace anywhere.
        """
        sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
        self._conn.execute(
            "INSERT OR IGNORE INTO version_blobs (sha, code, created_at) VALUES (?,?,?)",
            (sha, code, _now()),
        )
        newest = self._conn.execute(
            "SELECT version, blob_sha FROM tool_versions WHERE tool=?"
            " ORDER BY version DESC LIMIT 1", (tool,)).fetchone()
        if version is None:
            if newest is not None and newest["blob_sha"] == sha:
                self._conn.commit()
                return {"tool": tool, "version": newest["version"], "blob_sha": sha,
                        "parent_version": None, "new": False}
            if newest is not None:
                version = newest["version"] + 1
            else:
                row = self._conn.execute(
                    "SELECT version FROM tools WHERE name=?", (tool,)).fetchone()
                version = ((row["version"] if row is not None else 0) or 1)
        known = self._conn.execute(
            "SELECT blob_sha FROM tool_versions WHERE tool=? AND version=?",
            (tool, version)).fetchone()
        if known is None:
            if parent_version is None:
                prev = self._conn.execute(
                    "SELECT MAX(version) FROM tool_versions WHERE tool=? AND version < ?",
                    (tool, version)).fetchone()
                parent_version = prev[0] if prev and prev[0] is not None else None
            who = session or os.environ.get("AUTOFORGE_SESSION") or ""
            self._conn.execute(
                "INSERT INTO tool_versions (tool, version, blob_sha, parent_version,"
                " verification, archived_at, session) VALUES (?,?,?,?,?,?,?)",
                (tool, version, sha, parent_version, _j(verification or {}), _now(),
                 who),
            )
            # The timeline goes into the chained log as well, so "when did this
            # tool's body change" is answerable from the same tamper-evident
            # history as everything else rather than a second, weaker ledger.
            chaining.append_event(self._conn, "version_recorded", {
                "tool": tool, "version": version, "blob_sha": sha,
                "parent_version": parent_version, "session": who,
            })
            self._last_writer[tool] = who
            self._last_writer_at[tool] = _now()
        self._conn.commit()
        return {"tool": tool, "version": version, "blob_sha": sha,
                "parent_version": parent_version, "new": known is None}
    def versions_of(self, tool: str) -> list[dict[str, Any]]:
        """The recorded timeline for one tool, oldest first. Empty is an answer."""
        rows = self._conn.execute(
            "SELECT v.version, v.blob_sha, v.parent_version, v.archived_at,"
            " v.verification, LENGTH(b.code) AS size"
            " FROM tool_versions v JOIN version_blobs b ON b.sha = v.blob_sha"
            " WHERE v.tool=? ORDER BY v.version", (tool,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["verification"] = json.loads(d["verification"] or "{}")
            except (json.JSONDecodeError, TypeError):
                d["verification"] = {}
            out.append(d)
        return out
    def version_of(self, tool: str, version: int | None = None) -> dict[str, Any] | None:
        """The body of one version, by number (default: the newest recorded)."""
        if version is None:
            row = self._conn.execute(
                "SELECT * FROM tool_versions WHERE tool=? ORDER BY version DESC LIMIT 1",
                (tool,)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM tool_versions WHERE tool=? AND version=?",
                (tool, version)).fetchone()
        if row is None:
            return None
        blob = self._conn.execute(
            "SELECT code FROM version_blobs WHERE sha=?", (row["blob_sha"],)).fetchone()
        return {"tool": tool, "version": row["version"], "blob_sha": row["blob_sha"],
                "parent_version": row["parent_version"], "archived_at": row["archived_at"],
                "verification": _unjson(row["verification"]),
                "code": blob["code"] if blob else None}
    def version_graph(self) -> dict[str, Any]:
        """Every recorded version and parent edge, for the lineage view."""
        edges = [{"from": f"{r['tool']}@{r['parent_version']}",
                  "to": f"{r['tool']}@{r['version']}"}
                 for r in self._conn.execute(
                     "SELECT tool, version, parent_version FROM tool_versions"
                     " WHERE parent_version IS NOT NULL ORDER BY tool, version")]
        n = self._conn.execute("SELECT COUNT(*) FROM tool_versions").fetchone()[0]
        blobs = self._conn.execute("SELECT COUNT(*) FROM version_blobs").fetchone()[0]
        return {"versions": n, "distinct_bodies": blobs, "edges": edges}

    # -- dependencies -----------------------------------------------------
    def save_deps(self, tool: str, depends_on: list[str]) -> None:
        self._conn.execute("DELETE FROM dependencies WHERE tool=?", (tool,))
        for dep in depends_on:
            if dep == tool:
                continue
            self._conn.execute(
                "INSERT OR IGNORE INTO dependencies (tool, depends_on) VALUES (?,?)",
                (tool, dep),
            )
        self._conn.commit()
    def get_deps(self, tool: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT depends_on FROM dependencies WHERE tool=? ORDER BY depends_on",
            (tool,),
        ).fetchall()
        return [r["depends_on"] for r in rows]
    def get_reverse_deps(self, tool: str) -> list[str]:
        """Everything that depends on `tool` (for cascade checks)."""
        rows = self._conn.execute(
            "SELECT tool FROM dependencies WHERE depends_on=? ORDER BY tool",
            (tool,),
        ).fetchall()
        return [r["tool"] for r in rows]

    # -- frozen baselines -------------------------------------------------
    def save_baseline(self, baseline: Any) -> None:
        """Persist a tool's frozen exam. Upsert: the row is the obligation set."""
        self._conn.execute(
            "INSERT INTO baselines (tool, frozen_at, baseline) VALUES (?,?,?)"
            " ON CONFLICT(tool) DO UPDATE SET"
            " frozen_at=excluded.frozen_at, baseline=excluded.baseline",
            (baseline.tool, float(baseline.frozen_at), _j(baseline.to_dict())),
        )
        self._conn.commit()
    def load_baseline(self, tool: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT baseline FROM baselines WHERE tool = ?", (tool,)
        ).fetchone()
        return _unjson(row["baseline"]) if row else None

    # -- topologies -------------------------------------------------------
    def save_topology(self, topology: Any, task: str = "") -> None:
        """Persist a designed multi-agent topology (nodes, edges, rationale)."""
        self._conn.execute(
            "INSERT INTO topologies (timestamp, task, topology, fitness, trials)"
            " VALUES (?,?,?,?,?)",
            (
                _now(),
                (task or "")[:500],
                _j(topology.to_dict()),
                float(getattr(topology, "fitness", 0.0) or 0.0),
                int(getattr(topology, "trials", 0) or 0),
            ),
        )
        self._conn.commit()
    def load_topologies(self, limit: int = 20) -> list[dict[str, Any]]:
        """Most recent topologies first, newest at index 0."""
        rows = self._conn.execute(
            "SELECT * FROM topologies ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "task": r["task"],
                "fitness": r["fitness"],
                "trials": r["trials"],
                "topology": _unjson(r["topology"]),
            }
            for r in rows
        ]

    # -- events -----------------------------------------------------------
    def log_event(self, kind: str, payload: dict[str, Any]) -> None:
        """Append one event. Delegates to the chain so every row self-witnesses.

        The write path used to be a bare INSERT, which recorded what happened
        and nothing about whether it was later rewritten. chaining.append_event
        adds prev_hash/payload_hash/writer_id under BEGIN IMMEDIATE, so the
        hash of this row is a claim about the row before it and cannot be made
        against a stale head.
        """
        with self._lock:
            return chaining.append_event(self._conn, kind, payload)

    def verify_chain(self) -> dict[str, Any]:
        """Walk the log and report what is provable about it, row by row."""
        return chaining.verify_chain(self._conn)
    def anchor_chain(self, note: str = "") -> dict[str, Any]:
        """Freeze everything logged so far as a baseline."""
        return chaining.anchor(self._conn, note=note)
    def lineage_of(self, tool: str, depth: int = 3) -> dict[str, Any]:
        """Derivation edges actually recorded for a tool -- not inferred ones."""
        return chaining.lineage_of(self._conn, tool, depth=depth)
    def last_called_by_tool(self) -> dict[str, float]:
        """Newest `call` event per tool, from the ledger.

        The ledger is the only complete record of what was used: `call` events
        are appended for every tool the loop ran, while a tool's own
        `stats.last_called` is only written back when something saves the tool,
        which is rare. That asymmetry is why the exposure decision can be made
        from the ledger and not from the registry.

        One pass over one indexed column. The result is held by the registry
        until a call happens, so this runs once per tool call, not once per
        turn, and the prompt does not move between turns that call nothing.
        """
        newest: dict[str, float] = {}
        try:
            rows = self._conn.execute(
                "SELECT timestamp, payload FROM forge_events"
                # Both kinds, on purpose. The service wrote `tool_called` for a
                # long time while this query asked for `call`, so the two names
                # are one history split in half: reading only the newer one
                # would make every call before the fix look like it never
                # happened, which is the same lie in the other direction.
                " WHERE kind IN ('call', 'tool_called') ORDER BY id"
            ).fetchall()
        except Exception:                                     # noqa: BLE001
            return newest
        for r in rows:
            try:
                name = _unjson(r["payload"]).get("tool")
            except Exception:                                 # noqa: BLE001
                continue
            if name:
                newest[name] = float(r["timestamp"])
        return newest

    def get_events(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM forge_events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for r in reversed(rows):
            entry: dict[str, Any] = {"id": r["id"], "timestamp": r["timestamp"], "kind": r["kind"]}
            entry.update(_unjson(r["payload"]))
            results.append(entry)
        return results


    # -- deliberate memory ------------------------------------------------
    def remember(self, key: str, value: str, tags: list[str] | None = None) -> None:
        """Keep one fact across sessions, replacing any earlier value for `key`."""
        now = _now()
        self._conn.execute(
            "INSERT INTO memory (key, value, tags, created_at, updated_at, recalls)"
            " VALUES (?,?,?,?,?,0)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
            " tags=excluded.tags, updated_at=excluded.updated_at",
            (key, value, _j(tags or []), now, now),
        )
        self._conn.commit()
    def forget(self, key: str) -> bool:
        """Drop one memory. Returns whether it was there to drop."""
        cur = self._conn.execute("DELETE FROM memory WHERE key=?", (key,))
        self._conn.commit()
        return cur.rowcount > 0
    def recall(self, query: str = "", limit: int = 20) -> list[dict[str, Any]]:
        """Read memories back, most recently touched first.

        An empty query returns everything, which is what a fresh session wants:
        what did I know last time. A non-empty query is a case-insensitive
        substring match against key and value.
        """
        like = "%" + query.lower() + "%"
        rows = self._conn.execute(
            "SELECT * FROM memory WHERE lower(key) LIKE ? OR lower(value) LIKE ?"
            " ORDER BY updated_at DESC LIMIT ?",
            (like, like, limit),
        ).fetchall()
        out = [
            {
                "key": r["key"],
                "value": r["value"],
                "tags": _unjson(r["tags"]) or [],
                "updated_at": r["updated_at"],
                "recalls": r["recalls"],
            }
            for r in rows
        ]
        if out:
            self._conn.executemany(
                "UPDATE memory SET recalls = recalls + 1 WHERE key=?",
                [(o["key"],) for o in out],
            )
            self._conn.commit()
        return out
    def memory_for_injection(self, limit: int = 20) -> list[dict[str, Any]]:
        """Read kept facts for the per-turn block — without counting a recall.

        Kept apart from `recall` on purpose. `recall` is the agent *choosing* to
        look something up; this is the prompt carrying what it already decided
        to keep. Folding the two together would bump the recall counter on every
        turn, turning `recalls` into "how many turns has this entry existed" and
        erasing the only signal for which facts the agent actually reaches for.

        Order: most-recalled first, then most-recent. `recalls` starts at zero
        for everything, so a fresh store degrades cleanly to "what did I know
        last time" — and once the agent starts reaching for a fact on purpose,
        that fact climbs.
        """
        rows = self._conn.execute(
            "SELECT key, value, tags, updated_at, recalls FROM memory"
            " ORDER BY recalls DESC, updated_at DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [
            {
                "key": r["key"],
                "value": r["value"],
                "tags": _unjson(r["tags"]) or [],
                "updated_at": r["updated_at"],
                "recalls": r["recalls"],
            }
            for r in rows
        ]

    # -- skills (usage history; the content lives in markdown on disk) -----
    def upsert_skill(self, name: str, path: str, source: str, description: str,
                     when_to_use: str, tags: list[str]) -> None:
        """Refresh a skill's metadata, preserving the counters.

        Called on every scan of the skills directories. The counters are the
        reason this table exists, so a re-scan must never reset them: a skill
        loaded forty times is a different candidate from one that has never
        run, and the router scores that difference. Content fields are
        overwritten — the file is what the agent and the human edit — while
        `loads` and `last_loaded` survive untouched.

        A scan whose fields all match what is already stored writes nothing.
        Without that guard this ran on every scan, and a scan happens on every
        `context_brief` -- once per five seconds, over every skill -- so
        `updated_at` advanced 8 rows continuously while the content never
        changed. That turns the column into "when the last scan ran", and a
        timestamp that is always a few seconds old is not a modification
        history; it is the same trap `mission.py` documents for its own table
        ("Not `updated_at` alone: a mission annotated by another session on the
        same day would look untouched by this one").

        Compared exactly, not by normalising: `_j` is the only serialiser
        writing this column (it produces ``["dsh", "autoforge"]``, comma and
        space), and the scanned values are the same Python types on the way
        back in. `""` and NULL are kept distinct on purpose -- "this skill
        declares no when_to_use" and "this row has no such field" are
        different facts, and only the first is what a file can say.
        """
        tags_json = _j(list(tags))
        description = description or ""
        when_to_use = when_to_use or ""
        source = source or ""
        existing = self._conn.execute(
            "SELECT path, source, description, when_to_use, tags"
            " FROM skills WHERE name=?", (name,)).fetchone()
        if existing is not None and (
            existing["path"] == path
            and (existing["source"] or "") == source
            and (existing["description"] or "") == description
            and (existing["when_to_use"] or "") == when_to_use
            and (existing["tags"] or "") == tags_json
        ):
            return

        now = _now()
        self._conn.execute(
            "INSERT INTO skills (name, path, source, description, when_to_use,"
            " tags, loads, last_loaded, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET"
            " path=excluded.path, source=excluded.source,"
            " description=excluded.description,"
            " when_to_use=excluded.when_to_use,"
            " tags=excluded.tags, updated_at=excluded.updated_at",
            (name, path, source, description, when_to_use, tags_json, now, now),
        )
        self._conn.commit()
    def skill_rows(self) -> list[dict[str, Any]]:
        """Every known skill, most-loaded first — the order the router blends."""
        rows = self._conn.execute(
            "SELECT * FROM skills ORDER BY loads DESC, name ASC").fetchall()
        return [
            {
                "name": r["name"], "path": r["path"], "source": r["source"],
                "description": r["description"], "when_to_use": r["when_to_use"],
                "tags": _unjson(r["tags"]) or [], "loads": r["loads"],
                "last_loaded": r["last_loaded"], "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]
    def skill_load(self, name: str) -> int:
        """Count one load. Returns the new count, or 0 if there is no such skill.

        A skill the agent never loads and a skill that was never written look
        identical from the outside; this counter is how they are told apart,
        so it is bumped by the load itself rather than inferred later.
        """
        row = self._conn.execute(
            "SELECT loads FROM skills WHERE name=?", (name,)).fetchone()
        if row is None:
            return 0
        n = int(row["loads"]) + 1
        self._conn.execute(
            "UPDATE skills SET loads=?, last_loaded=? WHERE name=?",
            (n, _now(), name))
        self._conn.commit()
        return n
    def forget_skill_row(self, name: str) -> bool:
        """Drop a skill's row. The caller archives the file; usage goes with it.

        Kept separate from archiving because they can diverge: a skill file
        that a human deleted out from under the agent should lose its row on
        the next scan without any archiving happening.
        """
        cur = self._conn.execute("DELETE FROM skills WHERE name=?", (name,))
        self._conn.commit()
        return cur.rowcount > 0

    # -- self-report ------------------------------------------------------
    def report(self) -> dict[str, Any]:
        """What this store actually persists — facts, for the agent's self-model.

        Exists because an agent asked "do you remember across sessions?" should
        not answer from its prior. The ledger is on disk; ask it what it holds.
        """
        def _one(sql: str) -> Any:
            row = self._conn.execute(sql).fetchone()
            return row[0] if row else None

        tools = _one("SELECT COUNT(*) FROM tools") or 0
        states = {
            r["state"]: r["n"]
            for r in self._conn.execute(
                "SELECT state, COUNT(*) n FROM tools GROUP BY state ORDER BY state")
        }
        events = _one("SELECT COUNT(*) FROM forge_events") or 0
        kinds = {
            r["kind"]: r["n"]
            for r in self._conn.execute(
                "SELECT kind, COUNT(*) n FROM forge_events GROUP BY kind ORDER BY kind")
        }
        # The agent's own bookkeeping is in the same table as the record of what
        # happened, so an unfiltered count answers "how many events" with a
        # number the reader cannot interpret. A version_recorded row is written
        # by the store on every save, which makes it the loudest kind quickly.
        bookkeeping = {"version_recorded"}
        agent_events = {
            r["kind"]: r["n"]
            for r in self._conn.execute(
                "SELECT kind, COUNT(*) n FROM forge_events"
                " WHERE kind NOT IN ('version_recorded') GROUP BY kind ORDER BY kind")
        }
        return {
            "backend": "sqlite",
            "db_path": os.path.abspath(self.db_path),
            "survives_restart": True,
            "tools": tools,
            "tool_states": states,
            "events": events,
            "events_agent": sum(agent_events.values()),
            "bookkeeping_kinds": sorted(bookkeeping),
            "event_kinds_agent_only": agent_events,
            "event_kinds": kinds,
            "versions": _one("SELECT COUNT(*) FROM tool_versions") or 0,
            "distinct_bodies": _one("SELECT COUNT(*) FROM version_blobs") or 0,
            "chain": self.verify_chain(),
            "memories": _one("SELECT COUNT(*) FROM memory") or 0,
            "skills": _one("SELECT COUNT(*) FROM skills") or 0,
            "skill_loads": _one("SELECT SUM(loads) FROM skills") or 0,
            "skills_never_loaded": _one(
                "SELECT COUNT(*) FROM skills WHERE loads = 0") or 0,
            "since": _one("SELECT MIN(timestamp) FROM forge_events"),
            "until": _one("SELECT MAX(timestamp) FROM forge_events"),
        }


__all__ = ["ToolStore", "ToolRecord"]


# -- thread safety, applied at one choke point -------------------------------
#
# The lock has to cover whole method bodies, not individual statements: the
# failure is a cursor being iterated while another thread executes on the same
# connection, and no per-statement lock can see that. Wrapping each method here,
# once, is also why this is a loop and not 28 hand-indented bodies -- a hand-
# edited docstring indent is exactly the kind of change that compiles and then
# behaves differently from the version the tests were written against.
#
# `__init__` is excluded: it is the method that creates the lock.
def _serialized(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


for _name in [n for n in dir(ToolStore) if not n.startswith("__")]:
    _fn = getattr(ToolStore, _name)
    if isinstance(_fn, (staticmethod, classmethod)) or not callable(_fn):
        continue
    if getattr(ToolStore, _name).__name__ != _name:
        continue
    setattr(ToolStore, _name, _serialized(_fn))