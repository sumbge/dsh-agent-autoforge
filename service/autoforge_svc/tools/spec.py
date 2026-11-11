"""Tool specification, lifecycle state, and the reliability ledger.

This is where the framework's thesis lives. A tool is not a function you
register; it is a *claim* with a contract:

    ToolSpec = implementation + schema + trigger probes + effect signature
               + provenance + verification record + live reliability ledger

States form a lifecycle, not a boolean flag:

    DRAFT ──verify──> PROBATION ──earn──> ACTIVE ──decay──> QUARANTINED
                         ^                    │                  │
                         └──────rehab─────────┴──────────────────┘
                                            └──retire──> RETIRED

Free to create (DRAFT, PROBATION are immediately callable if `strict=False`),
but ACTIVE status — the thing that earns context budget and trust — must be
earned by passing probes and then *kept* by a track record.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable


#: How many calls are remembered per tool.
#:
#: Not a round number picked for tidiness: the window has to be long enough to
#: show a pattern (a tool that works, then degrades, then gets quarantined) and
#: short enough that the serialised row stays a row. At fifty entries the
#: `tools.stats` column measures a few kilobytes and one save stays under 3 ms
#: (measured 2026-11-02 -- see `ToolStats`); at twenty thousand it was one
#: megabyte and 52 ms per call, for a table nothing ever prunes.
#:
#: `schedule.py` set the precedent with `KEEP_HISTORY = 20` for task runs. This
#: is larger because a tool's history is the evidence a quarantine decision is
#: reviewed against, and twenty is not enough calls to distinguish "broken" from
#: "unlucky".
KEEP_CALL_HISTORY = 50

#: Slots inside that window reserved for failures.
#:
#: The auto-quarantine rule is `success_rate < 0.5` or three consecutive
#: failures (`registry.quarantine_consecutive_failures`), so ten failures is
#: several distinct episodes -- enough to see whether a repair held. Reserved
#: rather than counted on arrival, because a plain newest-N window evicts the
#: one failure the moment the tool starts succeeding again.
KEEP_CALL_FAILURES = 10

#: Longest error string a history entry keeps.
#:
#: Errors here are usually one line from a tool's own `raise`, but a sandbox
#: timeout carries a traceback and an MCP server can return a stack. Capped at
#: the same order as the shortest genuinely useful message (a Python exception
#: line with its location is ~120 characters) with room to spare.
ERROR_CHARS = 600


class ToolState(str, Enum):
    DRAFT = "draft"            # generated, not yet verified
    PROBATION = "probation"    # verified, on trial — callable but flagged
    ACTIVE = "active"          # earned trust, injected into context by default
    QUARANTINED = "quarantined"  # degraded — hidden by default, still executable
    RETIRED = "retired"        # removed from service


@dataclass
class TriggerProbe:
    """A test of the *trigger* question, not the *execution* question.

    `query` should demand this tool. `expect` states what a well-behaved agent
    should do. `negative_query` (optional) should NOT trigger it — a tool that
    fires on everything is worse than a tool that never fires.
    """

    query: str
    expect: str = "call"
    negative_query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolStats:
    """Append-only ledger of how a tool behaves in the wild.

    *Bounded* ledger, though, and the bound is the interesting part.

    `history` used to grow without limit, and every save re-serialised the whole
    list into the `tools.stats` column. Measured 2026-11-02 against a real
    sqlite file, one `save_tool`:

        history=   100   2.43 ms      5,629 chars
        history=  1000   6.20 ms     54,011 chars
        history=  5000  13.90 ms    268,955 chars
        history= 20000  52.30 ms  1,074,890 chars

    Linear, so a tool with twenty thousand calls rewrites a megabyte every time
    it runs -- and the table it rewrites is an append-only ledger that never
    shrinks.

    Two things must survive the trim, and "keep the newest N" loses both:

    * **The failures.** The entries that explain a quarantine are the oldest
      thing in the list by the time anyone asks, so a newest-N window deletes
      exactly the evidence. `KEEP_CALL_FAILURES` slots are reserved for them.
    * **The count.** `calls`, `successes` and `failures` are counters, not
      `len(history)`; trimming the list must not rewrite the tally, or
      `success_rate` becomes "the rate over the last N calls" while still
      being presented as the tool's record.

    `history_dropped` says how many entries are no longer shown, so a window
    is never presented as a life. It is a count and not a timestamp because
    what a reader needs to know is whether the list they are looking at is all
    of it.
    """

    calls: int = 0
    successes: int = 0
    failures: int = 0
    trigger_hits: int = 0
    trigger_misses: int = 0
    consecutive_failures: int = 0
    first_seen: float = field(default_factory=time.time)
    last_called: float = 0.0
    last_failure: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    history_dropped: int = 0
    #: How many calls the counters no longer include, because `rehab` zeroed them.
    #:
    #: `registry.rehab()` zeroes `calls`/`successes`/`failures` so a tool comes
    #: back on trial with a clean rate, but it deliberately leaves `history`
    #: alone -- the calls really happened. The result is a tool showing
    #: `calls: 0` beside six entries, which reads as corruption unless something
    #: says why. Measured 2026-11-02: `fake_file_size` sat in exactly that state
    #: on the live database, and the only way to explain it was to read
    #: `registry.rehab`. `rehab` now adds the old count here before zeroing, so
    #: `calls + calls_not_in_counters` is the true lifetime call count.
    #:
    #: A scalar and not a synthetic history entry, because a marker inside
    #: `history` would be counted, sorted and trimmed like a call it never was.
    calls_not_in_counters: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0

    @property
    def trigger_rate(self) -> float:
        n = self.trigger_hits + self.trigger_misses
        return self.trigger_hits / n if n else 0.0

    def record_call(self, ok: bool, error: str | None = None) -> None:
        self.calls += 1
        self.last_called = time.time()
        if ok:
            self.successes += 1
            self.consecutive_failures = 0
        else:
            self.failures += 1
            self.consecutive_failures += 1
            self.last_failure = time.time()
        # An error string is untrusted text from a tool (or from a sandbox, or
        # from an MCP server). Unbounded, one pathological failure could put a
        # megabyte of traceback into every future save of this tool.
        if error is not None and len(error) > ERROR_CHARS:
            error = error[:ERROR_CHARS]
        self.history.append(
            {"t": self.last_called, "ok": ok, "error": error}
        )
        if len(self.history) > KEEP_CALL_HISTORY:
            self._trim_history()

    def _trim_history(self) -> None:
        """Keep the newest window, and force the newest failures into it.

        The naive rule -- newest N by position -- loses the evidence. A tool
        that fails once and then succeeds five hundred times has one
        interesting entry and it is the oldest one, so a positional window
        deletes exactly the call that explains a quarantine.

        The rule here: take the newest `KEEP_CALL_HISTORY` entries, then make
        sure the newest `KEEP_CALL_FAILURES` failures are among them, trading
        the oldest *success* for each one that is missing. A failure never
        evicts a failure, so the window cannot fill with repeated copies of the
        same error while older distinct ones fall out.

        Written as a repair rather than as an apportionment ("ten slots for
        failures, forty for successes") because an apportionment is a ceiling
        read backwards: a tool that has failed sixty times in a row would keep
        only ten of them and report forty phantom drops, throwing away history
        it had no reason to throw away.
        """
        if len(self.history) <= KEEP_CALL_HISTORY:
            return
        start = len(self.history) - KEEP_CALL_HISTORY
        keep = self.history[start:]
        fail_at = [i for i, e in enumerate(self.history) if not e["ok"]]
        # Failures older than the window that are still worth forcing back in.
        # They can only be missing when the window is not already all
        # failures: an entry at `i < start` is older than everything in `keep`,
        # so if `keep` were entirely failures they would already be the older
        # sort -- and the newest ten would be inside the window regardless.
        wanted = [i for i in fail_at[-KEEP_CALL_FAILURES:] if i < start]
        if not wanted:
            self.history_dropped += start
            self.history = keep
            return
        out: list[dict[str, Any]] = []
        queue = [self.history[i] for i in wanted]
        for entry in keep:
            if queue and entry["ok"]:
                out.append(queue.pop(0))
            else:
                out.append(entry)
        # A failure that found no success to replace stays out: the window is
        # full of newer failures, which is the case this repair exists to
        # preserve, not to undo.
        out.sort(key=lambda e: e["t"])
        self.history_dropped += len(self.history) - len(out)
        self.history = out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["success_rate"] = round(self.success_rate, 4)
        d["trigger_rate"] = round(self.trigger_rate, 4)
        return d


_JSON_TYPES = frozenset(
    {"string", "number", "integer", "boolean", "object", "array", "null"}
)


def normalise_parameters(params: Any) -> dict[str, Any]:
    """Force a JSON-Schema object whose ``properties`` values are schemas.

    Two habits — one from models, one from hand-written JSON — break every
    downstream consumer with ``AttributeError: 'str' object has no attribute
    'get'``, raised three modules away from the cause::

        {"properties": {"isbn": "string"}}    # bare type name, not a schema
        {"properties": "isbn: string"}        # properties as free text

    The verifier, the fuzzer and the adversarial gate each read
    ``schema.get("type", ...)`` per property, so the shape is settled once, at
    the trust boundary where untrusted parameters enter (LLM output, stored
    JSON) — not in ``ToolSpec`` itself, whose many construction sites and
    hash-dependent lifecycle want exactly what the caller passed.
    """
    if not isinstance(params, dict) or not params:
        return {"type": "object", "properties": {}}

    out = dict(params)
    out.setdefault("type", "object")

    props = out.get("properties")
    props = props if isinstance(props, dict) else {}
    fixed: dict[str, Any] = {}
    for name, schema in props.items():
        if isinstance(schema, str):
            token = schema.strip().lower()
            schema = {"type": token if token in _JSON_TYPES else "string"}
        elif not isinstance(schema, dict):
            schema = {"type": "string"}
        # A *list* of types is legal JSON Schema and means "any of these":
        # {"type": ["number", "null"]} is how a nullable parameter is written,
        # and MCP servers emit it routinely. Every downstream consumer does
        # `schema.get("type", "string")` and then uses that value as a dict key
        # -- `_PROBE_MAP.get(ptype, ...)`, `ptype in ("number", "integer")` --
        # so a list arrives at a hashing site and dies with
        # `TypeError: unhashable type: 'list'`, four modules away from the type
        # that caused it. Reported by a peer session 2026-09-21 as a forge-pipeline
        # crash for any tool whose parameter was declared as a list.
        #
        # Normalised here rather than at each consumer: this is the trust
        # boundary where untrusted parameters enter, which is exactly what this
        # function exists for. The FIRST non-"null" member is the real type --
        # the nullable case is `["T", "null"]`, and picking the last member
        # would call a nullable number a null, which is worse than a crash
        # because it is silent. Dropped, not kept: downstream only asks for one
        # type, and keeping the list would leave the same landmine for the next
        # reader. The original is preserved beside it so nothing is lost.
        elif isinstance(schema, dict):
            declared = schema.get("type")
            if isinstance(declared, (list, tuple, set, frozenset)):
                members = [str(m) for m in declared]
                # Prefer the first member that is not "null". A union of only
                # "null" is degenerate but legal, and "null" IS a JSON type, so
                # it is kept rather than swapped for an invented "string" --
                # the one thing this function must not do is fabricate a type
                # the schema never declared. A member that is not a JSON type
                # at all ("widget") degrades to "string", which is the same
                # convention already applied to a bare string.
                chosen = next(
                    (m for m in members if m != "null"),
                    members[0] if members else "string")
                schema = dict(schema)
                schema["type"] = chosen if chosen in _JSON_TYPES else "string"
                schema["type_union"] = members
        fixed[str(name)] = schema
    out["properties"] = fixed

    required = out.get("required")
    if required is not None:
        out["required"] = (
            [str(r) for r in required] if isinstance(required, list)
            else [str(required)] if isinstance(required, str) else []
        )
    return out


@dataclass
class ToolSpec:
    """A self-made tool, contract and all."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    code: str = ""
    source: str = "human"          # human | generated | composed
    generator: str = ""            # model that wrote it
    probes: list[TriggerProbe] = field(default_factory=list)
    # A concrete, *valid* call: the arguments a real use would pass, and what a
    # correct answer looks like. This is the field whose absence made the whole
    # verification battery vacuous -- see `ToolVerifier.check_execution`. Every
    # other check fed the tool garbage, so a tool that answered "INVALID" to
    # everything scored perfectly. Without a positive example there is nothing
    # to be right about.
    sample_call: dict[str, Any] = field(default_factory=dict)
    sample_expect: str = ""        # substring a correct result must contain, if known
    effect_signature: str = ""     # what it touches; used for re-verification
    invariances: list[str] = field(default_factory=list)  # token types whose
    # decoration must not change the answer (e.g. "isbn"). Declared at birth by
    # the generator, then frozen into FrozenBaseline — a later mutant cannot
    # drop one without failing the regression gate.
    state: ToolState = ToolState.DRAFT
    stats: ToolStats = field(default_factory=ToolStats)
    verification: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    cost_hint: str = "cheap"       # cheap | moderate | expensive
    created_at: float = field(default_factory=time.time)
    # Optional out-of-process executor. When set, the registry calls
    # `runner(name, args) -> ToolResult-ish` instead of `fn(**args)`. Forged
    # tools set this to a sandbox bridge so untrusted code never runs in-loop.
    runner: Callable[..., Any] | None = None
    #: Whether a call that is *already running* can be asked to give up.
    #:
    #: Declared by the tool rather than inferred from its name, because the
    #: answer is a fact about its executor and not about what it does. A
    #: `runner` that shells out to a sandbox can kill the child; an in-process
    #: `fn` cannot be interrupted at all — Python offers no safe way to stop a
    #: frame, so pretending otherwise means either a thread left running
    #: against a history that has moved on, or a `kill` that takes the agent
    #: with it. The registry only offers the abort question to a tool that said
    #: yes, and `interruptible=False` is the honest label: the operator's line
    #: waits for this call to return, and `/status` says so while they wait.
    interruptible: bool = False

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @property
    def hash(self) -> str:
        blob = json.dumps(
            {
                "name": self.name,
                "code": self.code,
                "parameters": self.parameters,
                "description": self.description,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def is_trusted(self) -> bool:
        return self.state in (ToolState.PROBATION, ToolState.ACTIVE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "state": self.state.value,
            "source": self.source,
            "generator": self.generator,
            "hash": self.hash,
            "tags": self.tags,
            "effect_signature": self.effect_signature,
            "stats": self.stats.to_dict(),
            "verification": self.verification,
            "probes": [p.to_dict() for p in self.probes],
            "sample_call": self.sample_call,
            "sample_expect": self.sample_expect,
        }


__all__ = ["ToolState", "ToolStats", "TriggerProbe", "ToolSpec"]
