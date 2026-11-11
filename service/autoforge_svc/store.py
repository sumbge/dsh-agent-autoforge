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

#: Attempts at a contended write, on top of the busy timeout. Measured
#: 2026-10-31: a writer that another process held could not be entered for 25
#: seconds straight despite the 15s timeout, because sqlite's busy handler
#: retries internally but does not always outlive a writer that is itself
#: stalled. A few explicit retries with growing gaps cover that case; a truly
#: dead holder is still reported rather than waited on forever.
_WRITE_RETRIES = 4
_WRITE_BACKOFF_S = 0.12

#: The WAL is truncated back to zero once its file exceeds this many bytes.
#:
#: Twice-patched, because the first two versions of this bound were both dead
#: code and the measurements say so:
#:
#:   * It started as `_WAL_CHECKPOINT_PAGES = 2000` on the strength of "4.12 MB
#:     with a sequence that stopped advancing". 2000 pages is 8 MB, so the bound
#:     could never fire on the very failure it was written for -- the wedged WAL
#:     measured 4,120,032 bytes, i.e. 1006 pages. A limit above the observed
#:     fault is not a limit.
#:   * It then gated on `log_frames` from `PRAGMA wal_checkpoint(PASSIVE)`.
#:     Measured 2026-11-01 by writing 400 KB rows into a fresh store: the file
#:     plateaus at 539,752 bytes (131.8 pages) and stays there for hundreds of
#:     writes, with `log_frames` pinned at 101. SQLite's own auto-checkpoint
#:     (`wal_autocheckpoint = 1000`) recycles frames in place, so a frame count
#:     never reaches a large threshold either. The frame count measures how much
#:     is *behind*, not how big the file *is*, and the file is what fills the
#:     disk.
#:
#: So the only honest trigger is the sidecar's own size, and the only honest
#: number is one strictly between the healthy plateau and the observed fault:
#: 1 MiB is ~2x above the measured plateau and ~4x below the wedged file.
_WAL_CHECKPOINT_BYTES = 1 << 20

#: Event kinds the service writes about itself rather than about the work.
#: Kept on the ledger (they are how a host's wiring is verified) but not
#: counted as history.
_BOOKKEEPING_KINDS = ("brief_served",)

# How many rows may accumulate past the newest anchor before an ordinary
# write takes another one. 250 bounds how much a later editor could
# re-sign without the anchor noticing; it is not a claim of coverage.
# Measured 2026-11-05: the tail that went uncovered was 604 rows over 16
# days of ordinary use, because nothing pressed the button.
_ANCHOR_EVERY = int(os.environ.get("AUTOFORGE_ANCHOR_EVERY") or 250)

# Rows that deserve their own boundary: the ones a later reader asks
# "was this decided before or after the evidence moved" about. Ordinary
# chatter waits for the interval; a decision does not.
_ANCHOR_NOW_KINDS = frozenset({
    "tool_defined", "evolve", "retire", "review", "state_change",
    "runs_pruned", "tables_dropped", "ledger_repair", "counters_repaired",
    "model_substituted", "memory_recalls_reset",
})

#: What each table is for, and what a zero in it means. Declared data, not a
#: renderer's guess.
#:
#: Written 2026-11-03 because `report()` had no table-level view at all, and
#: that made two different findings wear the same face: a table whose writer is
#: broken and a table whose button nobody has ever pressed both report zero
#: rows. Five tables were zero that day and they were **three different
#: things** -- `dependencies` and `topologies` had no reachable writer anywhere
#: (deleted in the same pass), `baselines` had a live writer that could not fire
#: because `evolve_tool` has never won, and `mission_state`/`chain_anchors` were
#: fully working features this agent had simply never used. Collapsing those
#: into "0 rows" is how a working system gets deleted for looking dead.
#:
#: `zero` is one of:
#:   ``never-triggered``  the code path exists and would write here; it has not
#:                        run yet. Absence of use, not absence of function.
#:   ``empty``            an ordinary empty state -- rows accumulate with use.
#:
#: A table with no entry here reports ``undeclared`` rather than inheriting a
#: default, so a table added tomorrow shows up as a question on its first run.
_TABLE_NOTES: dict[str, dict[str, str]] = {
    "tools": {
        "role": "one row per tool the registry knows, with state, stats and code",
        "zero": "empty",
    },
    "bridge_tools": {
        "role": "the host's tool catalogue as last reported by the JS bridge",
        "zero": "never-triggered",
    },
    "forge_events": {
        "role": "the append-only, hash-chained ledger of everything that happened",
        "zero": "empty",
    },
    "memory": {
        "role": "deliberate long-term facts the agent chose to keep",
        "zero": "empty",
    },
    "skills": {
        "role": "skill usage counters; the procedure text itself lives in markdown",
        "zero": "empty",
    },
    "version_blobs": {
        "role": "code bodies addressed by sha256, so two names can share one body",
        "zero": "empty",
    },
    "tool_versions": {
        "role": "the version timeline: which sha a tool was on, from which parent, when",
        "zero": "empty",
    },
    "baselines": {
        "role": "a tool's frozen exam, written when `evolve_tool` wins",
        "zero": "never-triggered",
    },
    "missions": {
        "role": "things owed, open and closed",
        "zero": "empty",
    },
    "mission_log": {
        "role": "one row per mission open, note, close, wake or sweep",
        "zero": "empty",
    },
    "mission_state": {
        "role": "the single `focus` marker naming which mission this session is on",
        "zero": "never-triggered",
    },
    "chain_anchors": {
        "role": "frozen checkpoints of the ledger chain, written by anchor_chain()",
        "zero": "never-triggered",
    },
    "wire_calls": {
        "role": ("one row per wire method ever pressed, with call count, failure "
                 "count and the last duration -- the meter that answers "
                 "'has this edge of the surface ever actually run'"),
        "zero": "never-triggered",
    },
}


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
    # `dependencies` and `topologies` used to be created here. Removed
    # 2026-11-03, not because the tables were empty but because nothing could
    # ever have filled them: `save_deps`/`get_deps`/`get_reverse_deps`,
    # `save_topology`/`load_topologies` and the whole of `tools/composition.py`
    # had zero call sites, `TopologyDesigner` had zero instantiations, and the
    # only *producer* of a dependency edge would have been a model writing
    # `uses:<tool>` in `effect_signature` -- a syntax no prompt ever offered it.
    # A table is not a feature; it is where a feature would keep its result.
    "\nCREATE TABLE IF NOT EXISTS baselines ("
    "\n    tool TEXT NOT NULL PRIMARY KEY,"
    "\n    frozen_at REAL NOT NULL,"
    "\n    baseline TEXT NOT NULL DEFAULT '{}'"
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
    # The bridge's own tool catalog -- the 66 `autoforge_*` tools it registers
    # with DSH. They are not forged tools and never enter `tools`, but
    # `describe_tool` promised to answer for any tool by name, and the manifest
    # text tells the model to call it "for its full parameter schema first".
    # Measured 2026-11-02: `describe_tool('autoforge_run_python')` answered
    # "no tool named ... the library holds: fake_file_size, file_facts, ..." --
    # true, and useless, for 60 of the 66 names the model can actually call.
    #
    # Written by the bridge at mount time and read back here, because the
    # bridge is the only place those schemas exist: they are JavaScript object
    # literals in `lib/index.js`, and re-deriving them in Python would be a
    # second copy that drifts. `reported_at` is stored rather than implied so a
    # stale catalog can be described as stale instead of being believed.
    "\nCREATE TABLE IF NOT EXISTS bridge_tools ("
    "\n    name TEXT NOT NULL PRIMARY KEY,"
    "\n    description TEXT NOT NULL DEFAULT '',"
    "\n    parameters TEXT NOT NULL DEFAULT '{}',"
    "\n    reported_at REAL NOT NULL,"
    "\n    schema TEXT NOT NULL DEFAULT '{}'"
    "\n);"
    # Which wire methods have ever been pressed, and whether they worked.
    # Measured 2026-11-06: 40 of 76 dispatch-table methods had no suite that
    # called them, and the ledger could not answer the question -- `kind='call'`
    # records forged-tool invocations, never wire methods, so "has this part of
    # the surface ever run" had no source at all.
    #
    # A counter, not evidence: deliberately NOT in the hash chain. The chain
    # exists so a claim about a tool can be contradicted; "was mission_wake ever
    # pressed" wants a tally, and putting traffic on the chain would make the
    # anchor cadence fire on requests instead of on decisions.
    #
    # `known` is stored rather than implied. A bridge calling a name the service
    # does not have is the `anchor_ledger` defect class -- a method, a bridge
    # registration and eight passing assertions, and the wire answering
    # `unknown method` -- and the request is the only thing that still remembers
    # the name afterwards.
    "\nCREATE TABLE IF NOT EXISTS wire_calls ("
    "\n    method TEXT NOT NULL PRIMARY KEY,"
    "\n    known INTEGER NOT NULL DEFAULT 1,"
    "\n    calls INTEGER NOT NULL DEFAULT 0,"
    "\n    failures INTEGER NOT NULL DEFAULT 0,"
    "\n    total_ms REAL NOT NULL DEFAULT 0,"
    "\n    last_ms REAL NOT NULL DEFAULT 0,"
    "\n    last_at REAL NOT NULL DEFAULT 0"
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


def _tag_list(raw: Any) -> list[str]:
    """One shape for memory tags, whatever wrote the row.

    The column is declared `tags TEXT NOT NULL DEFAULT '[]'` and its declared
    Python type is `list[str]`. Measured 2026-11-06: 37 of 38 rows in the
    production database hold `'"test,probe"'` -- a JSON *string* holding the
    tags -- so `_unjson` hands back `str` and a reader gets a different type
    depending on which writer made the row. Nothing had broken yet only because
    no reader iterates the value; the first one that did would iterate
    characters, and `"beta" in "alpha,beta"` is True by substring, not by
    membership. `Service.remember` takes the comma-separated string a human
    types, so that shape lands here too.

    Only the two memory readers go through this. A tool row and a skill row
    spell their own `tags` differently and are written with a list already.
    """
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    if isinstance(raw, str):
        parsed = _unjson(raw)
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if str(t).strip()]
        if isinstance(parsed, str):
            return [t.strip() for t in parsed.split(",") if t.strip()]
        return [t.strip() for t in raw.split(",") if t.strip()]
    return []


def _sidecar_bytes(db_path: str, suffix: str) -> int:
    """Size of a sqlite sidecar (`-wal`, `-shm`), or 0 when it does not exist.

    The `-wal` file is where a wedged writer shows up: it grows and then stops,
    and no row in the database records that. Read through `stat`, and treat any
    error as "not there" -- a missing sidecar is a normal state, not a fault.
    """
    try:
        return os.stat(os.path.abspath(db_path) + suffix).st_size
    except OSError:
        return 0


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


def home_dir() -> str:
    """The one answer to "where does state live".

    Public because six other modules needed it and five of them wrote their own
    version instead. Measured 2026-11-05: `schedule.default_schedule_path`,
    both path helpers in `core/compaction`, and `Service._peer_market_from_file`
    all answered `<home>/.autoforge` while this function answered
    `$LOCALAPPDATA/autoforge`, so a task added through one reader was invisible
    to the other and two `schedule.jsonl` files existed on the same machine.

    Delegating rather than re-exporting a constant is the point: the fallback
    below carries a reason (the namespace-package shadowing measured on
    2026-09-16) and a second copy would drift away from it.
    """
    return _default_home()


def db_file(home: str | None = None, name: str = "autoforge.db") -> tuple[str, str]:
    """Which file holds the ledger, and whether it was inherited.

    Returns `(path, adopted_from)`. `home_dir()` answers a *directory* and this
    answers the *file*, because those stopped being the same question: a store
    created by the shipped preset lives at `<home>/data/autoforge.db` while
    `home_dir()` answers `<home>`. Measured 2026-11-06, with no
    `AUTOFORGE_HOME`: three rows sat in `<home>/data/autoforge.db`, and a service
    built with the defaults created a second, empty ledger at
    `<home>/autoforge.db` without mentioning the first.

    So the default location wins when it holds a file; otherwise a populated
    sibling is adopted and reported. Creating an empty ledger beside a full one
    is the one outcome nobody discovers afterwards -- the rows are still on disk,
    but no reader is pointed at them.

    An empty or schema-only sibling is *not* adopted: a file is not a ledger, and
    moving to one would only relocate the confusion.

    Two callers ask this: `Service.__init__` and `tars_port.probe_host`. They
    disagreed once already, in the opposite direction, and the fix for that is
    not another copy of the rule.
    """
    base = home or home_dir()
    default = os.path.join(base, name)
    if os.path.exists(default):
        return default, ""
    sibling = os.path.join(base, "data", name)
    if not os.path.exists(sibling) or os.path.getsize(sibling) <= 0:
        return default, ""
    try:
        with sqlite3.connect("file:%s?mode=ro" % sibling.replace("\\", "/"),
                             uri=True) as con:
            rows = con.execute("SELECT COUNT(*) FROM forge_events").fetchone()[0]
    except Exception:                                     # noqa: BLE001
        return default, ""
    if not rows:
        return default, ""
    return sibling, sibling


def state_map(home: str | None = None) -> dict[str, Any]:
    """Every directory and file this package would write to, in one answer.

    Asked of each owner rather than re-derived here: `home_dir` for the
    directory, `db_file` for the ledger (which may be a sibling one level
    down), `schedule`/`runs`/`compaction`/`skills` for their own defaults.
    A copy of their rules in this function would be the ninth derivation.

    Why it exists. Measured 2026-11-06: `report()` carried `home.dir` and
    `home.db_path` and nothing else, so five of the six places this package
    writes -- agenda, runs, spill, compaction log, skills -- were invisible
    to a reader who only had the report. And after `db_file()` began
    adopting a populated sibling, the ledger could sit *outside* the
    directory `home_dir()` names while everything else stayed inside it,
    which made "back up the home directory" quietly wrong and "delete the
    home directory" quietly incomplete.

    Nothing here creates a directory; every entry is a path the package
    would use, not one it has used.

    `config` is the tenth entry, added 2026-11-10. It was the one
    reader of durable state that answered outside `home_dir()` --
    `Path.home() / ".autoforge" / "config.json"` -- and because it
    had no key here it was also the one path `outside_home()` could
    not name. A list that exists to say what left the home directory
    has to know about every reader, or it reports an empty answer for
    a file that is not empty. Worth knowing when reading this entry:
    nothing in this tree writes that file (`configfile.save` has no
    caller) and `notify` is the only reader, so the path is reported
    as "where it would go", exactly like the other nine.
    """
    from . import configfile as _configfile
    from . import runs as _runs
    from . import schedule as _schedule
    from . import skills as _skills
    from . import store as _store
    from .core import compaction as _compaction

    base = home or _store.home_dir()
    ledger, adopted = _store.db_file(base)
    # Every directory the reader would actually search, as `(origin, path)`,
    # not only the `user` one. Measured 2026-11-06: with
    # `$AUTOFORGE_SKILLS_DIRS` set the origins are `env`, `env1`, ... and the
    # old `origin == "user"` filter matched nothing, so this answered `""` --
    # "no skill directory" for a process that reads exactly the directories
    # the variable named. An empty string also fell out of the
    # all-paths-are-under-home check below it, because an empty value does
    # not fail a predicate, it disappears from one. A list, because the
    # variable takes more than one directory and reporting the first would be
    # a quieter version of the same lie.
    skill_dirs = [
        {"origin": origin, "path": path}
        for origin, path in _skills.default_skill_dirs()
    ]
    return {
        "home": base,
        "ledger": ledger,
        "ledger_adopted_from": adopted,
        "cache": os.path.join(base, "cache"),
        "agenda": str(_schedule.default_schedule_path()),
        "runs": str(_runs.journal_dir()),
        "spill": str(_compaction.spill_dir()),
        "compaction_log": str(_compaction.default_log_path()),
        "skills": skill_dirs,
        "config": str(_configfile.config_path()),
    }


def outside_home(state: dict[str, Any]) -> list[str]:
    """Which entries of `state_map()` are not inside `state["home"]`.

    Written because the first version of this report named only the ledger.
    Measured online 2026-11-06 with the shipped preset: `home` is
    `<base>/data` while `runs` and `skills` are `<base>/runs` and
    `<base>/skills`, so two entries were outside home and neither was named --
    the same defect as the one the ledger flag fixed, one entry narrower.

    `ledger` is checked like any other key rather than special-cased. Measured
    2026-11-06: it never leaves -- `db_file` answers either
    `<home>/autoforge.db` or `<home>/data/autoforge.db`, both inside. That is
    why the per-entry boolean next to this list was deleted: a field that can
    only ever say False carries no information, and a mutant that hardcoded it
    to False survived the suite unchanged.
    """
    home = os.path.abspath(str(state.get("home") or ""))
    if not home:
        return []
    prefix = home.rstrip("\\") + os.sep
    out = []
    for key, value in state.items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            path = item.get("path") if isinstance(item, dict) else item
            if not path:
                continue
            absolute = os.path.abspath(str(path))
            if absolute != home and not absolute.startswith(prefix):
                out.append(key)
                break
    return sorted(out)


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
        # Counted, not swallowed: a checkpoint that could not complete is the
        # other way this store can leak disk while looking healthy. The ledger's
        # own drop counter lives on the service (`_ledger_locked_drops`), which
        # is the level that catches the exception -- do not add a second one
        # here, or "how many writes were lost" gets two answers.
        self._checkpoint_skips = 0
        self._checkpoint_runs = 0
        self._anchor_runs = 0
        self._anchor_skips = 0
        # How far the last full chain walk got, in this process only. It answers
        # "has anything been written since I last looked", which is a fact about
        # this process's own reading -- a value loaded from a previous run would
        # claim a walk this process never performed.
        self._chain_verified: dict[str, Any] = {"up_to_id": 0, "at": 0.0}
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
        # The ledger's chain columns and `chain_anchors` are created here, at
        # open, instead of being left to the first writer or reader that needs
        # them. They used to be created by `chaining.ensure_schema`, which is
        # called from `append_event` *and from `verify_chain`* -- and
        # `verify_chain` is what `report()` calls. So the first `report()` on a
        # fresh file measured the schema and then changed it in the same dict
        # literal: view 1 said `chain_anchors` was absent, view 2 said it was
        # present, identical code both times. A report that changes what it
        # reports is not a measurement, and a getter that runs DDL is not a
        # getter. Schema is an open-time responsibility; this is the one place
        # per process where it can be done once and be true afterwards.
        chaining.ensure_schema(self._conn)
        # Indexes come last, and the ORDER is load-bearing. Measured
        # 2026-11-04: created before `chaining.ensure_schema`, the partial
        # index on the legacy shape raised `no such column: payload_hash`,
        # because that column is added by the chain schema, not by `_DDL`.
        # The error was swallowed and a fresh store silently shipped with two
        # of its three indexes -- visible only because a cost measurement
        # refused to move.
        self._ensure_indexes()

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
        # What the model is actually SENT for a bridge tool, as opposed to the
        # descriptor the bridge declares. `defineTool` compiles the descriptor
        # into JSON Schema before the schema reaches a provider, and the two
        # differ by an order of magnitude: measured 2026-11-02, all 66
        # descriptors total 132 characters and their compiled schemas total
        # 11,823. Any decision that compares a cost against a saving has to use
        # this column, or it compares a small number to nothing.
        ("bridge_tools", "schema", "TEXT NOT NULL DEFAULT '{}'"),
    )

    #: Indexes the queries in this file actually need. Data rather than
    #: DDL because `CREATE TABLE IF NOT EXISTS` never adds one to a file
    #: that already exists, and every store on this machine predates these.
    #: Each entry names the statement that pays for it, because an index
    #: nobody can connect to a query is an index nobody will maintain.
    _INDEXES = (
        # report(): `SELECT COUNT(*)`, `GROUP BY kind`, and the same group-by
        # with `WHERE kind NOT IN (...)`. Measured at 12000 rows: COUNT 2.94 ->
        # 0.01 ms (the index is covering, so the count walks it), group-by
        # 5.51 -> 0.60, filtered group-by 6.01 -> 1.38.
        ("forge_events", "forge_events_kind", "kind"),
        # report(): `SELECT MIN(timestamp)` and `MAX(timestamp)`, the ledger's
        # own "when did this start and stop". Measured at 12000 rows: 3.43 ->
        # 0.01 ms each. Not for ordering: `get_events` already returns in id
        # order, which is the primary key and needs nothing.
        ("forge_events", "forge_events_timestamp", "timestamp"),
        # chaining._counts(): `COUNT(*) WHERE payload_hash IS NULL OR
        # payload_hash = ''` -- the legacy, pre-chain rows. Partial, and on
        # `id` alone, because the count never reads a payload and the normal
        # answer is zero: an index over an empty set costs nothing to scan.
        # Measured at 12000 rows: that count goes from a 4.15 ms full scan of
        # the table (measured together with the two other aggregates it used
        # to share a statement with) to 0.01 ms.
        ("forge_events", "forge_events_legacy", "id",
         "payload_hash IS NULL OR payload_hash = ''"),
    )

    def _ensure_indexes(self) -> None:
        """Create the indexes in `_INDEXES`, if this file does not have them.

        Read `sqlite_master` first, for the reason `_ensure_columns` gives:
        `CREATE INDEX IF NOT EXISTS` on an existing index still takes the
        schema lock, and this store is opened by several sessions on one
        file. Measured 2026-11-04 before adding these, `report()` on a
        12000-row ledger took 31.1 ms, of which 29.9 ms was five scans of
        `forge_events`; with both indexes it takes 6.99 ms.
        """
        for entry in self._INDEXES:
            table, name, column = entry[0], entry[1], entry[2]
            predicate = entry[3] if len(entry) > 3 else ""
            try:
                have = self._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
                    (name,)).fetchone()
            except sqlite3.Error:
                continue
            if have:
                continue                 # nothing to do: no write, no lock
            try:
                where = f" WHERE {predicate}" if predicate else ""
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {name} ON {table}({column}){where}")
                self._conn.commit()
            except sqlite3.Error as exc:
                # Only a busy file is survivable here. A file another session
                # is writing may refuse the schema change, and the next open
                # retries; but `no such column` is a bug in this file, and
                # swallowing it is how this method shipped two indexes for a
                # day while its docstring claimed three. Measured 2026-11-04:
                # the swallow hid exactly that error.
                said = str(exc).lower()
                if "locked" not in said and "busy" not in said:
                    raise
                # failing to open the store because a cache could not be
                # built is the worse trade -- but only for a lock.

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

    def connection(self) -> sqlite3.Connection:
        """The live connection, for a writer that must share this file's chain.

        Exposed because the audit layer writes through `chaining.append_event`,
        which needs *this* connection to share the write lock, the busy timeout
        and the `PRAGMA journal_mode=WAL` that were configured here. Opening a
        second connection to the same file looks equivalent and is not: measured
        2026-11-03, a control plane opened its own connection and every receipt
        it wrote was invisible to this store's `verify_chain` until the process
        exited, because the reader's snapshot predated the writer's commit.

        Deliberately a method rather than a public attribute: `_reopen()` replaces
        `self._conn` after a wedge, and a caller that captured the old object
        would go on writing to a closed handle.
        """
        return self._conn


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

    # -- connection hygiene ----------------------------------------------

    def _end_transaction(self) -> bool:
        """Make sure this connection is not sitting in an open transaction.

        Returns True when it is clean. This exists because a swallowed
        `rollback()` is not a harmless detail: measured 2026-11-01 on this
        machine, one failed rollback left the shared connection inside a write
        transaction for good. The process then idled at 0% CPU while holding
        the database's write lock, every other writer -- including its own
        later calls -- got "database is locked" against a 15s busy timeout, and
        the WAL stopped advancing for nine minutes. Reads kept working, which
        is exactly what made it look like contention rather than a wedge.

        A connection that cannot be brought back to autocommit is closed and
        reopened. That is the only recovery that does not depend on the broken
        object cooperating.
        """
        conn = self._conn
        try:
            if not getattr(conn, "in_transaction", False):
                return True
        except Exception:                                    # noqa: BLE001
            pass
        for _ in range(2):
            try:
                conn.rollback()
            except Exception:                                # noqa: BLE001
                pass
            try:
                if not getattr(conn, "in_transaction", False):
                    return True
            except Exception:                                # noqa: BLE001
                pass
        # Still inside a transaction: the connection is not recoverable in
        # place. Replace it rather than hold the write lock forever.
        try:
            conn.close()
        except Exception:                                    # noqa: BLE001
            pass
        self._reopen()
        return False

    def _maybe_anchor(self, kind: str) -> None:
        """Declare a boundary after a write, when one is due.

        Called from `log_event` AFTER the row is committed, so nothing here
        can cost the write that just succeeded. Two triggers, for two
        different questions:

          * `_ANCHOR_NOW_KINDS` -- a decision. The question a later reader
            asks is "was this taken before or after the evidence moved",
            and an interval cannot answer it. Measured 2026-11-05: all 8
            review receipts on this ledger sat past the newest anchor.
          * `_ANCHOR_EVERY` rows -- everything else. A bound on the
            re-signable tail rather than a promise of coverage.

        Silent in outcome, counted in fact: `_anchor_skips` records every
        failed attempt, and a failure here never raises. A busy reader or a
        locked file costs one missing boundary, not a broken turn.

        Setting `AUTOFORGE_ANCHOR_EVERY=0` turns the interval off. It does
        not turn the decision trigger off: that would restore exactly the
        state this method exists to end.
        """
        if _ANCHOR_EVERY <= 0 and kind not in _ANCHOR_NOW_KINDS:
            return
        try:
            due = kind in _ANCHOR_NOW_KINDS
            if not due:
                total = self._conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM forge_events"
                ).fetchone()[0] or 0
                newest = self._conn.execute(
                    "SELECT COALESCE(MAX(up_to_id), 0) FROM chain_anchors"
                ).fetchone()[0] or 0
                due = (int(total) - int(newest)) >= _ANCHOR_EVERY
            if not due:
                return
            note = ("auto: decision row %s" % kind if kind in
                    _ANCHOR_NOW_KINDS else "auto: %d rows past the last anchor"
                    % (int(total) - int(newest)))
            chaining.anchor(self._conn, note=note)
            self._anchor_runs += 1
        except Exception:                              # noqa: BLE001
            # The row is already committed. Nothing here is worth raising
            # over, but it is worth counting -- the same trade and the same
            # reason as _maybe_checkpoint above.
            self._anchor_skips += 1

    def _maybe_checkpoint(self) -> None:
        """Truncate the WAL once its file has grown past `_WAL_CHECKPOINT_BYTES`.

        Measured on this machine, a wedged writer left `-wal` at 4.12 MB and
        frozen while the database itself still read fine -- nothing in any row
        recorded it, and the file only ever shrank again by hand. This is the
        write path, which is where that growth is produced, so the bound lives
        here.

        Two decisions, both forced by measurement rather than taste:

          * The trigger is the sidecar's `stat` size, not the frame count
            `wal_checkpoint(PASSIVE)` reports. A fresh store plateaus at
            539,752 bytes / 131.8 pages while `log_frames` sits at 101 for
            hundreds of writes, because SQLite recycles frames in place -- the
            frame count can honestly stay small while the file is the thing
            filling the disk.
          * It issues `TRUNCATE`, not `PASSIVE`. Measured on the real database
            at 4,120,032 bytes: PASSIVE returned `(0, 789, 789)` and left the
            file byte-for-byte unchanged; TRUNCATE returned `(0, 0, 0)` and took
            it to 0 bytes. PASSIVE reclaims frames, only TRUNCATE reclaims space.

        A busy result is not an error -- a reader may be holding a frame -- so it
        is counted and skipped, never raised. Failing to checkpoint costs disk;
        raising here would cost the write that just succeeded.

        Reading the size is also cheap (`os.stat`, no query), so this costs one
        syscall per ledger write in the common case.
        """
        if _WAL_CHECKPOINT_BYTES <= 0:
            return
        if _sidecar_bytes(self.db_path, "-wal") < _WAL_CHECKPOINT_BYTES:
            return
        try:
            got = self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        except Exception:                                    # noqa: BLE001
            self._checkpoint_skips += 1
            return
        # TRUNCATE reports its own busy flag; a reader that appeared between the
        # measurement and the truncation turns this into a no-op, which is fine.
        if got is not None:
            try:
                if int(got[0]):
                    self._checkpoint_skips += 1
                    return
            except (TypeError, IndexError, ValueError):
                pass
        self._checkpoint_runs += 1

    def _reopen(self) -> None:
        """Rebuild `self._conn` after a wedge, with the same settings."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False,
                                     timeout=float(_SQLITE_BUSY_TIMEOUT_S))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=%d" % int(_SQLITE_BUSY_TIMEOUT_S * 1000))
        self._set_wal_with_retry()

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
            # Retried on the same policy as `log_event`: a save that loses the
            # write lock is a tool the next session will not have, and losing
            # that race was measured on 2026-10-31 (see _WRITE_RETRIES).
            last: Exception | None = None
            for attempt in range(_WRITE_RETRIES):
                try:
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
                                {"query": p.query, "expect": p.expect,
                                 "negative_query": p.negative_query}
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
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                        raise
                    last = exc
                    # `_end_transaction`, not a bare rollback: a rollback that
                    # fails must not leave the connection holding the write
                    # lock. See its docstring for the measured cost of exactly
                    # that on 2026-11-01.
                    self._end_transaction()
                    time.sleep(_WRITE_BACKOFF_S * (2 ** attempt))
            else:
                self._end_transaction()
                raise last if last is not None else sqlite3.OperationalError(
                    "tool write failed")
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
        #
        # The number it returns is written back onto the row, because until
        # 2026-11-01 it was not: `save_tool` inserted `record.version` -- always
        # the spec's default 1 -- and `record_version` numbered the timeline
        # 1,2,3 beside it. Measured that day on `file_facts`: `tool_versions`
        # held v1/v2/v3 with the current body at v3 while `tools.version` still
        # said 1, so the only column a reader consults for "which version is
        # this" disagreed with the timeline that actually defines it. Two
        # answers to one question is the same defect the framework exists to
        # catch; the timeline is authoritative here because it is what
        # `record_version` derives and what `versions_of` replays.
        stamped = self.record_version(record.name, record.code, record.verification,
                                      session=mine or None)
        got = stamped.get("version")
        if isinstance(got, int) and got != record.version:
            self._conn.execute(
                "UPDATE tools SET version=?, updated_at=? WHERE name=?",
                (got, record.updated_at, record.name))
            self._conn.commit()
            record.version = got

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
    # `save_topology` / `load_topologies` were here until 2026-11-03 and had
    # zero call sites. Kept as a note rather than as code because the table
    # itself is gone: `TopologyDesigner` (the only producer of the `topology`
    # and `fitness` values these methods needed) is never instantiated anywhere
    # in the package, so the writer could not have been reached even if called.

    # -- events -----------------------------------------------------------
    def log_event(self, kind: str, payload: dict[str, Any]) -> None:
        """Append one event. Delegates to the chain so every row self-witnesses.

        The write path used to be a bare INSERT, which recorded what happened
        and nothing about whether it was later rewritten. chaining.append_event
        adds prev_hash/payload_hash/writer_id under BEGIN IMMEDIATE, so the
        hash of this row is a claim about the row before it and cannot be made
        against a stale head.

        Retried, because the busy timeout alone is not enough across processes.
        Measured 2026-10-31 on this machine: twelve `BEGIN IMMEDIATE` attempts
        over 25 seconds all failed with "database is locked" while another
        process held the writer, and the event ledger gained zero rows in that
        window. Every one of those failures was an *audit record* that vanished
        -- and the vanished ones are precisely the ones worth keeping, which is
        the lesson this file already carries in the constructor comment about
        sixteen threads landing four rows.

        The retry is bounded and it does not invent success: after the last
        attempt this raises, and the caller decides. Silence is the one
        outcome it will not produce.
        """
        with self._lock:
            last: Exception | None = None
            for attempt in range(_WRITE_RETRIES):
                try:
                    written = chaining.append_event(self._conn, kind, payload)
                    self._maybe_checkpoint()
                    self._maybe_anchor(kind)
                    return written
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                        raise
                    last = exc
                    # See `_end_transaction`: chaining does its own rollback,
                    # but a rollback that itself fails leaves the shared
                    # connection holding the write lock forever.
                    self._end_transaction()
                    time.sleep(_WRITE_BACKOFF_S * (2 ** attempt))
            # Out of attempts. Re-raise rather than swallow: a caller that
            # cannot write to the ledger needs to know, and the tools that call
            # this already handle the failure explicitly.
            self._end_transaction()
            raise last if last is not None else sqlite3.OperationalError("write failed")

    def verify_chain(self) -> dict[str, Any]:
        """Walk the log and report what is provable about it, row by row.

        Expensive by nature -- one sha256 per row -- so it is a *tool call*, not
        something a report does. See `chaining.summarize` for the cheap half.

        Records how far the walk got, so a later `report()` can say whether
        anything has been written since. The mark is set only after a walk that
        actually completed: an exception leaves the previous mark alone rather
        than advancing it over rows nobody checked.
        """
        result = chaining.verify_chain(self._conn)
        self._remember_chain_check(result)
        return result

    def chain_summary(self) -> dict[str, Any]:
        """The chain's shape without the walk, plus whether a walk is due.

        `needs_full_check` is the whole point. A summary that reports no breaks
        is not reporting a clean chain -- it has not looked -- so the answer it
        gives has to be "there is something to check", not "nothing to see".
        Measured 2026-11-03: `report()` spent 95% of its 46 ms in the walk, and
        the walk is linear in rows (515 ms at 100k). Making the cheap path also
        the reassuring path is how a verifier becomes a rubber stamp.
        """
        summary = chaining.summarize(self._conn)
        mark = self._chain_verified
        summary["verified_up_to"] = mark.get("up_to_id", 0)
        summary["verified_at"] = mark.get("at", 0.0)
        # The last walk's verdict travels with it, as history with a bound rather
        # than as a present-tense claim. Leaving it out was worse than it looks:
        # a run that found a broken row would report `needs_full_check: false`
        # afterwards (the walk did cover everything) and no breaks -- so the one
        # moment the store knows something is wrong is the moment it stops
        # mentioning it. Measured 2026-11-03 on a fixture with one tampered row.
        summary["last_full_check"] = dict(mark.get("verdict") or {})
        summary["needs_full_check"] = (
            int(summary.get("up_to_id") or 0) != int(mark.get("up_to_id") or 0)
        )
        summary["note"] = (
            "counts and head only; run verify_ledger for the per-row proof"
            if summary["needs_full_check"] else
            "counts and head only; the last full walk covered every row to date"
        )
        return summary

    def _remember_chain_check(self, result: dict[str, Any]) -> None:
        """Note that a full walk just happened, how far it got, and what it said.

        Recorded in memory, not on disk: it answers "has anything been written
        since I last looked", which is a fact about this process's own reading,
        and a value loaded from a previous run would claim a walk that this
        process never performed.
        """
        breaks = result.get("breaks") or []
        gaps = result.get("gaps") or []
        self._chain_verified = {
            "up_to_id": int(result.get("verified_up_to") or 0),
            "at": time.time(),
            "verdict": {
                "ok": bool(result.get("ok")),
                "checked_at": time.time(),
                "up_to_id": int(result.get("verified_up_to") or 0),
                "breaks": len(breaks),
                "gaps": len(gaps),
                # The first break, named, so the summary is actionable without a
                # second call. One is enough: the walk finds what it finds, and
                # a list here would grow into a second copy of the report.
                "first_break": (breaks[0] if breaks else None),
            },
        }
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

    def save_bridge_tools(self, catalog: list[dict[str, Any]]) -> dict[str, Any]:
        """Record the bridge's tool catalog, wholesale.

        Wholesale, not incremental, and that is the point: the bridge reports
        what it registers right now, so a name that disappeared from
        `lib/index.js` must disappear here too. A merge would keep answering
        for a tool nobody can call any more -- a second copy of the same lie
        this table exists to fix.

        The cost of wholesale is that a failed read must not be mistaken for
        an empty catalog, so an empty `catalog` is refused rather than obeyed.
        """
        if not catalog:
            return {"ok": False, "written": 0,
                    "error": "refusing to record an empty catalog: the bridge "
                             "always registers tools, so an empty list is a "
                             "failed read, not a library that became empty"}
        now = _now()
        rows = []
        for entry in catalog:
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            rows.append((
                name,
                str(entry.get("description") or ""),
                _j(entry.get("parameters") or {}),
                now,
                _j(entry.get("schema") or {}),
            ))
        if not rows:
            return {"ok": False, "written": 0,
                    "error": "every entry was missing a name"}
        with self._lock:
            try:
                self._conn.execute("DELETE FROM bridge_tools")
                self._conn.executemany(
                    "INSERT OR REPLACE INTO bridge_tools"
                    " (name, description, parameters, reported_at, schema)"
                    " VALUES (?, ?, ?, ?, ?)", rows)
                self._conn.commit()
            except sqlite3.Error as exc:
                self._end_transaction()
                return {"ok": False, "written": 0,
                        "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "written": len(rows), "reported_at": now}

    def bridge_tool(self, name: str) -> dict[str, Any] | None:
        """One catalog entry, or None. `None` means "not in the catalog",
        which is a different answer from "no catalog was ever reported"."""
        try:
            row = self._conn.execute(
                "SELECT name, description, parameters, reported_at, schema"
                " FROM bridge_tools WHERE name = ?", (name,)).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        return {"name": row["name"], "description": row["description"],
                "parameters": _unjson(row["parameters"]),
                "schema": _unjson(row["schema"]),
                "reported_at": float(row["reported_at"])}

    def bridge_catalog(self) -> dict[str, Any]:
        """The whole catalog plus when it was last reported."""
        try:
            rows = self._conn.execute(
                "SELECT name, description, parameters, reported_at, schema"
                " FROM bridge_tools ORDER BY name").fetchall()
        except sqlite3.Error:
            return {"tools": [], "reported_at": 0.0}
        return {
            "tools": [{"name": r["name"], "description": r["description"],
                       "parameters": _unjson(r["parameters"]),
                       "schema": _unjson(r["schema"])} for r in rows],
            "reported_at": max((float(r["reported_at"]) for r in rows), default=0.0),
        }

    def calls_of(self, tool: str) -> dict[str, Any]:
        """How many times the ledger saw this tool called, and how many worked.

        `ToolStats` counts calls in memory and is only written back when a tool
        row is saved, so `stats.calls` and the ledger disagree -- and the
        ledger is the one that cannot be lost with the process. `describe_tool`
        reports both rather than one, because a caller asking "how has this
        behaved" needs to know when the two numbers differ.

        Both spellings of the kind are counted for the reason given in
        `last_called_by_tool`: `call` and `tool_called` are one history that was
        written under two names.
        """
        out: dict[str, Any] = {"calls": 0, "ok": 0, "failed": 0, "last": 0.0}
        try:
            rows = self._conn.execute(
                "SELECT timestamp, payload FROM forge_events"
                " WHERE kind IN ('call', 'tool_called') ORDER BY id").fetchall()
        except Exception:                                     # noqa: BLE001
            return out
        for r in rows:
            try:
                payload = _unjson(r["payload"])
            except Exception:                                 # noqa: BLE001
                continue
            if payload.get("tool") != tool:
                continue
            out["calls"] += 1
            if payload.get("ok"):
                out["ok"] += 1
            else:
                out["failed"] += 1
            out["last"] = float(r["timestamp"])
        return out

    def get_events(self, limit: int = 100, kind: str = "",
                   exclude_bookkeeping: bool = False) -> list[dict[str, Any]]:
        """Newest events, oldest last.

        `kind` filters in SQL rather than in the caller. Measured 2026-10-31:
        `my_history(kind="forge")` fetched the newest 30 rows and then filtered
        them in Python, and every one of those 30 was `brief_served` -- so the
        answer was "0 events" for a session that had in fact just forged a tool.
        Filtering after the LIMIT is the same defect as reading a page of a
        sorted file and calling it the file. Asking for a kind must mean
        "the newest N of THAT kind", and that can only be expressed before the
        LIMIT.

        `exclude_bookkeeping` drops the self-monitoring kinds (`brief_served`)
        from the result. They are worth keeping but they are not history of
        the work; a log that is 99.7% heartbeat makes "what did this agent do"
        unanswerable by page size alone.
        """
        sql = "SELECT * FROM forge_events"
        where: list[str] = []
        args: list[Any] = []
        if kind:
            where.append("kind = ?")
            args.append(kind)
        if exclude_bookkeeping:
            where.append("kind NOT IN (%s)" % ",".join("?" * len(_BOOKKEEPING_KINDS)))
            args.extend(_BOOKKEEPING_KINDS)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        rows = self._conn.execute(sql, tuple(args)).fetchall()
        results = []
        for r in reversed(rows):
            entry: dict[str, Any] = {"id": r["id"], "timestamp": r["timestamp"], "kind": r["kind"]}
            entry.update(_unjson(r["payload"]))
            results.append(entry)
        return results

    def event_kinds(self) -> list[tuple[str, int]]:
        """Every kind actually present, with its count. Sorted, biggest last.

        Exists so a caller does not have to guess the vocabulary: `my_history`
        used to accept a `kind` argument with no way to ask what kinds there
        were, and filtering on a plausible-looking name ("forge") returned
        nothing even though `tool_defined` and `version_recorded` had been
        written. A filter nobody can enumerate is a filter that lies by silence.
        """
        rows = self._conn.execute(
            "SELECT kind, COUNT(*) AS n FROM forge_events GROUP BY kind"
        ).fetchall()
        return [(r["kind"], int(r["n"])) for r in rows]


    # -- deliberate memory ------------------------------------------------
    def remember(self, key: str, value: str,
                 tags: list[str] | str | None = None) -> None:
        """Keep one fact across sessions, replacing any earlier value for `key`.

        `tags` accepts the comma-separated string the wire takes as well as a
        list, because there is one column and it has one shape. Before this,
        `Service.remember` passed its string straight into `_j`, writing a JSON
        string where the declared type is a list -- 37 of 38 production rows were
        that shape on 2026-11-06.
        """
        tags = _tag_list(tags)
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

        Only a non-empty query counts as a recall. `recalls` is what orders
        `memory_for_injection`, and that method's docstring states the hazard
        plainly: counting a listing would turn the counter into "how many turns
        has this entry existed" and erase the only signal for which facts the
        agent actually reaches for. Measured 2026-11-02: a single `recall('')`
        listing bumped all 15 stored entries at once, and the block that is
        meant to lead with the most-reached fact led with whichever entry had
        merely been listed most often. The comment described a hazard the line
        beside it committed; the fix is to count the reach, not the glance.

        A query that happens to match everything still counts, because the agent
        typed words to find something and that is the behaviour worth measuring.
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
                "tags": _tag_list(r["tags"]),
                "updated_at": r["updated_at"],
                "recalls": r["recalls"],
            }
            for r in rows
        ]
        if out and query.strip():
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
                "tags": _tag_list(r["tags"]),
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
    def record_wire_call(self, method: str, *, ok: bool, ms: float,
                         known: bool = True) -> None:
        """Count one wire request. Best-effort, and it says when it failed.

        Called from `main()` -- the one door every request comes through -- and
        not from `_call_method`. The distinction matters: a suite that calls
        `svc._call_method("report")` is not a wire call, and if the meter lived
        one level down then this repository's own tests would show up as
        production traffic.

        Never raises. A counter that turns a finished call into a failed one is
        worse than no counter; `Service._wire_call_failures` counts the losses
        the same way `_ledger_locked_drops` counts ledger losses.
        """
        self._conn.execute(
            "INSERT INTO wire_calls (method, known, calls, failures, total_ms,"
            " last_ms, last_at) VALUES (?, ?, 1, ?, ?, ?, ?)"
            " ON CONFLICT(method) DO UPDATE SET"
            "   known = excluded.known,"
            "   calls = calls + 1,"
            "   failures = failures + excluded.failures,"
            "   total_ms = total_ms + excluded.total_ms,"
            "   last_ms = excluded.last_ms,"
            "   last_at = excluded.last_at",
            (str(method), 1 if known else 0, 0 if ok else 1,
             max(0.0, float(ms)), max(0.0, float(ms)), _now()))
        self._conn.commit()

    def wire_call_view(self) -> dict[str, Any]:
        """Which methods were pressed, and which arrived unknown.

        A counter cannot answer "which are missing" -- it has never seen them --
        so this returns only what it has. `Service.report()` supplies the
        complement from the dispatch table, which is the only object that knows
        the full list.
        """
        rows = self._conn.execute(
            "SELECT method, known, calls, failures, total_ms, last_ms, last_at"
            " FROM wire_calls ORDER BY method").fetchall()
        pressed, unknown = {}, {}
        for r in rows:
            entry = {
                "known": bool(r["known"]),
                "calls": int(r["calls"]),
                "failures": int(r["failures"]),
                "total_ms": round(float(r["total_ms"]), 3),
                "last_ms": round(float(r["last_ms"]), 3),
                "last_at": float(r["last_at"]),
            }
            (pressed if entry["known"] else unknown)[r["method"]] = entry
        return {"pressed": pressed, "unknown": unknown}

    def table_view(self) -> dict[str, dict[str, Any]]:
        """Every table in this file, what it holds, and which kind of zero it is.

        Two things kept apart on purpose: `rows` is measured with `COUNT(*)` and
        nothing else, while `role` and `verdict` come from `_TABLE_NOTES` -- a
        declaration written by a person. The measurement is the fact; the
        verdict is the claim about what the fact means, and mixing them into one
        number is how "0 rows" came to mean both "broken" and "unused".

        A table sqlite created for itself (`sqlite_sequence`) is not this
        store's business and is not listed. A table of ours that `_TABLE_NOTES`
        does not describe is listed anyway with `verdict == "undeclared"`: a
        new table must arrive as a question, not inherit an answer.

        Read-only, deliberately: this method must not create anything. It used
        to be called first in `report()` while `verify_chain()` -- a few lines
        later in the same dict literal -- ran `CREATE TABLE IF NOT EXISTS
        chain_anchors` through `chaining.ensure_schema`. So the first report of
        a fresh file described a schema that the report itself then changed, and
        the second report disagreed with the first about which tables exist.
        Measured 2026-11-03 on a throwaway store: `chain_anchors` absent from
        view 1, present in view 2, both from identical code. A report that
        changes the thing it reports is not a measurement.
        """
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            name = r["name"]
            if name.startswith("sqlite_"):
                continue
            # The name comes from sqlite_master, not from a caller, so it cannot
            # carry a quote; quoting it here is belt and braces for a table
            # named by a future migration rather than by a user.
            quoted = '"' + name.replace('"', '""') + '"'
            count = self._conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            note = _TABLE_NOTES.get(name)
            if note is None:
                entry: dict[str, Any] = {
                    "rows": int(count),
                    "role": "",
                    "verdict": "undeclared",
                    "zero": "",
                }
            else:
                zero_kind = note.get("zero", "empty")
                entry = {
                    "rows": int(count),
                    "role": note.get("role", ""),
                    "zero": zero_kind,
                    # A declaration says what an *empty* table means; once rows
                    # exist the only honest verdict is "ok" -- the declaration
                    # has been satisfied, whatever it said absence would mean.
                    "verdict": "ok" if count else ("never-triggered" if zero_kind == "never-triggered" else "empty"),
                }
            out[name] = entry
        return out

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
        # Two different questions, and until 2026-11-01 this function answered
        # both with one hardcoded name and got the split backwards: it called
        # `version_recorded` bookkeeping (2 rows) and left `brief_served` (7896
        # rows, one per context brief) in `events_agent`, so a reader asking
        # "how much has the agent actually done" saw 7901. They are separate
        # sets and both are now named for what they are:
        #
        #   bookkeeping  written by this store on its own behalf, on every
        #                save. Says something about the database, not the work.
        #   ambient      written because the host asked for a context brief.
        #                Real activity, but it is the host's cadence, not a
        #                decision the agent made.
        bookkeeping = {"version_recorded"}
        ambient = set(_BOOKKEEPING_KINDS)
        excluded = tuple(sorted(bookkeeping | ambient))
        agent_events = {
            r["kind"]: r["n"]
            for r in self._conn.execute(
                "SELECT kind, COUNT(*) n FROM forge_events"
                " WHERE kind NOT IN (%s) GROUP BY kind ORDER BY kind"
                % ",".join("?" * len(excluded)), excluded)
        }
        return {
            "backend": "sqlite",
            "db_path": os.path.abspath(self.db_path),
            "survives_restart": True,
            "tables": self.table_view(),
            "tools": tools,
            "tool_states": states,
            "events": events,
            "events_agent": sum(agent_events.values()),
            "bookkeeping_kinds": sorted(bookkeeping),
            "ambient_kinds": sorted(ambient),
            "event_kinds_agent_only": agent_events,
            "event_kinds": kinds,
            "versions": _one("SELECT COUNT(*) FROM tool_versions") or 0,
            "distinct_bodies": _one("SELECT COUNT(*) FROM version_blobs") or 0,
            # `chain` used to be `self.verify_chain()` -- a full sha256 walk of
            # every row, paid by anyone who asked "what do you hold". Measured
            # 2026-11-03 on the production ledger: 43.9 ms of this function's
            # 46.4 ms, i.e. 95%. It is linear in rows (5.2 us each), so the same
            # call would cost 515 ms at 100k rows. A report is a cheap answer to
            # a cheap question; the proof is a tool call, and `needs_full_check`
            # is what points at it.
            "chain": self.chain_summary(),
            "memories": _one("SELECT COUNT(*) FROM memory") or 0,
            "skills": _one("SELECT COUNT(*) FROM skills") or 0,
            "skill_loads": _one("SELECT SUM(loads) FROM skills") or 0,
            "skills_never_loaded": _one(
                "SELECT COUNT(*) FROM skills WHERE loads = 0") or 0,
            "since": _one("SELECT MIN(timestamp) FROM forge_events"),
            "until": _one("SELECT MAX(timestamp) FROM forge_events"),
            # The WAL's size is a fact about this database that nothing else
            # reported. It is the number that went to 4.12 MB and froze while a
            # wedged connection held the write lock, and the number a checkpoint
            # is supposed to bring back down -- so measure it here instead of
            # trusting that `_maybe_checkpoint` is working.
            "wal_bytes": _sidecar_bytes(self.db_path, "-wal"),
            "wal_checkpoints": self._checkpoint_runs,
            "wal_checkpoint_skips": self._checkpoint_skips,
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