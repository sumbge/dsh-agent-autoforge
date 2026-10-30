"""Audit of reach decisions — the policy engine's receipts, on the chain.

Ported in shape from `siyad01/agentbox` `internal/audit/logger.go`, vendored at
`D:/Users/china/Desktop/项目_开发/_vendor/agentbox`. What is taken from it:

  * one entry per decision, classified allow/deny *by dimension* rather than as
    a single boolean — `filesystem_deny` and `tool_deny` are different events
    because they are different problems
  * the fields a reader needs to re-judge the call without re-running it:
    `resource`, `rule`, `reason`, and a free `extra`
  * limit breaches logged next to policy denials, since "it was stopped" and "it
    was refused" are the two ways an agent loses reach and a reader should not
    have to look in two places
  * `alert_on`: a manifest can name the event types worth interrupting someone
    for

What is deliberately NOT taken
------------------------------
The Go version writes its own NDJSON file and does its own chaining. autoforge
already has a chain — `chaining.py`, with anchors, writer ids and `verify_chain`
— and 969 of its rows were already chained when this was written. A second
appender would be a second hash chain, and two chains is the same as none: the
question "was this log edited" would have two answers and no reason to prefer
either. So this module is a *vocabulary and a query layer* over the existing
ledger: every write goes through `chaining.append_event`, so audit rows land in
the same chain as every forge, run and amendment.

The one capability that was missing and is added here: the ledger recorded what
happened, but nothing answered "what was refused, and by which rule" — which is
the question the policy engine's receipts exist to make answerable.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "EVENT_TYPES", "DIMENSIONS", "dimension_of", "entry_from_decision", "record", "record_decision",
    "record_limit_breach", "denials", "by_rule", "summary", "export_ndjson",
    "should_alert",
]

#: The five dimensions the policy engine judges, times two verdicts, plus the
#: two events that are not verdicts. Names match `internal/audit/logger.go` so an
#: entry from either implementation reads the same to a person.
EVENT_TYPES = (
    "filesystem_allow", "filesystem_deny",
    "network_allow", "network_deny",
    "tool_allow", "tool_deny",
    "credential_allow", "credential_deny",
    "limit_breached", "policy_alert",
    # The control layer's own decisions: the autonomy gate, the declared
    # envelope, and a change to the agent itself. Not dimension verdicts --
    # they answer "was this allowed" about the machinery rather than about a
    # file or a host -- which is why they carry dimension "policy" and why they
    # are named here rather than folded into one of the ten above.
    "gate_allow", "gate_deny",
    "declaration_allow", "declaration_deny",
    "selfmod_allow", "selfmod_deny",
    # not a dimension verdict: the two spellings for a decision that did not say
    # which question it was answering. Kept in the vocabulary so a reader sees
    # the ambiguity instead of an event type that looks authoritative.
    "policy_allow", "policy_deny",
)

KIND = "policy"          # the `kind` column value for everything written here

DIMENSIONS = ("filesystem", "network", "tool", "credential")


def dimension_of(decision: Any) -> str:
    """Which question was asked.

    Strictly the decision's own answer, with one fallback: a decision object from
    an older caller, or from another engine, that carries only a rule string.
    Deriving it from that string is a *lossy* guess -- `policy: default_deny` is
    the shared rule for all four dimensions, and reading it as "policy" files a
    network denial where nobody will look for it. So the guess is kept, named,
    and only used when the caller did not say.
    """
    said = str(getattr(decision, "dimension", "") or "")
    if said in DIMENSIONS:
        return said
    head = (getattr(decision, "rule", "") or "").split(":", 1)[0].strip()
    candidate = head.split(".")[0]
    return candidate if candidate in DIMENSIONS else "policy"


def entry_from_decision(decision: Any, *, resource: str = "",
                        agent: str = "", extra: str = "") -> dict[str, Any]:
    """Turn a `capability_policy.Decision` into one audit entry.

    Accepts anything with `.allowed/.rule/.reason`, so a caller that wraps the
    engine in its own decision type does not have to convert by hand first.
    """
    rule = getattr(decision, "rule", "") or ""
    dim = dimension_of(decision)
    allowed = bool(getattr(decision, "allowed", False))
    if dim == "policy":
        # A verdict whose dimension nobody stated is recorded as a policy event,
        # not dropped: a decision that happened and left no trace is the failure
        # this module is for.
        event_type = "policy_allow" if allowed else "policy_deny"
    else:
        event_type = "%s_%s" % (dim, "allow" if allowed else "deny")
    return {
        "event_type": event_type,
        "allowed": allowed,
        "resource": resource,
        "rule": rule,
        "reason": getattr(decision, "reason", "") or "",
        "agent": agent,
        "extra": extra,
        "ts": time.time(),
    }


def record(conn: sqlite3.Connection, entry: dict[str, Any],
           writer: str | None = None) -> dict[str, Any]:
    """Append one entry to the existing chain. Returns the chain row."""
    from .. import chaining
    payload = dict(entry)
    payload.setdefault("ts", time.time())
    return chaining.append_event(conn, KIND, payload, writer=writer)


def record_decision(conn: sqlite3.Connection, decision: Any, *,
                    resource: str = "", agent: str = "",
                    extra: str = "", writer: str | None = None) -> dict[str, Any]:
    """The ordinary call: judge, then write the receipt. One line at the call site."""
    return record(conn, entry_from_decision(
        decision, resource=resource, agent=agent, extra=extra), writer=writer)


def record_limit_breach(conn: sqlite3.Connection, limit_type: str, value: Any, *,
                        agent: str = "", writer: str | None = None) -> dict[str, Any]:
    """A limit that bit is an audit event, not just a debug line.

    Kept beside the policy denials on purpose: from the outside, "the sandbox
    killed it" and "the policy refused it" both look like the tool not running,
    and telling them apart is the first thing anyone does afterwards.
    """
    return record(conn, {
        "event_type": "limit_breached", "allowed": False, "resource": "",
        "rule": "limit: %s" % limit_type,
        "reason": "%s limit breached: %r" % (limit_type, value),
        "agent": agent, "extra": "", "ts": time.time(),
    }, writer=writer)


def should_alert(engine: Any, event_type: str) -> bool:
    """Delegate to the engine if it can answer, so `alert_on` lives in one place."""
    fn = getattr(engine, "should_alert", None)
    return bool(fn(event_type)) if callable(fn) else False


# -- reading it back -----------------------------------------------------

def _rows(conn: sqlite3.Connection, since: float | None = None,
          event_type: str | None = None,
          only_denied: bool = False) -> list[dict[str, Any]]:
    """Decode policy rows in order.

    Filtering happens in Python rather than in SQL because the discriminating
    fields live inside the JSON payload — a `LIKE` over that text would match a
    rule name inside a *reason* string and call it a different event.
    """
    # `ToolStore`'s own DDL creates `forge_events` without the chain columns;
    # `chaining.ensure_schema` adds them, idempotently, and every write path
    # calls it. A *reader* did not, so "what was refused" raised `no such column:
    # prev_hash` on a store where nothing had been written yet — the one case
    # where the honest answer is "nothing", and the case that produced every
    # first-run support question. Reading a chain that does not exist yet is not
    # an error.
    from .. import chaining

    chaining.ensure_schema(conn)
    sql = ("SELECT id, timestamp, payload, prev_hash, payload_hash FROM forge_events"
           " WHERE kind = ? ORDER BY id")
    out = []
    for row in conn.execute(sql, (KIND,)):
        try:
            payload = json.loads(row[2] or "{}")
        except ValueError:
            continue
        if since is not None and (row[1] or 0) < since:
            continue
        if event_type and payload.get("event_type") != event_type:
            continue
        if only_denied and payload.get("allowed"):
            continue
        payload = dict(payload)
        payload["id"] = row[0]
        payload["chained"] = bool(row[4])
        out.append(payload)
    return out


def denials(conn: sqlite3.Connection, *, since: float | None = None,
            limit: int = 200) -> list[dict[str, Any]]:
    rows = _rows(conn, since=since, only_denied=True)
    return rows[-limit:]


def by_rule(conn: sqlite3.Connection, *, since: float | None = None) -> list[dict[str, Any]]:
    """Denials grouped by the rule that caused them, most frequent first.

    This is the report that turns receipts into a manifest edit: the top rule is
    the one whose allow list is actually wrong.
    """
    counts: dict[str, dict[str, Any]] = {}
    for row in _rows(conn, since=since, only_denied=True):
        rule = row.get("rule") or "(no rule)"
        slot = counts.setdefault(rule, {"rule": rule, "count": 0,
                                        "event_type": row.get("event_type"),
                                        "example": row.get("resource") or row.get("reason")})
        slot["count"] += 1
    return sorted(counts.values(), key=lambda d: (-d["count"], d["rule"]))


def summary(conn: sqlite3.Connection, *, since: float | None = None) -> dict[str, Any]:
    rows = _rows(conn, since=since)
    allowed = sum(1 for r in rows if r.get("allowed"))
    by_type: dict[str, int] = {}
    for row in rows:
        key = row.get("event_type") or "?"
        by_type[key] = by_type.get(key, 0) + 1
    return {"entries": len(rows), "allowed": allowed,
            "denied": len(rows) - allowed, "by_event_type": by_type,
            "chained": sum(1 for r in rows if r.get("chained"))}


def export_ndjson(conn: sqlite3.Connection, path: Path | str, *,
                  since: float | None = None) -> dict[str, Any]:
    """Write the entries in the Go implementation's field names.

    The vendored tools read that shape, and an audit that cannot be handed to
    the tool that produced the policy is half an audit. `prev_hash` is carried
    through, so a reader comparing this file against the ledger's own
    `verify_chain` can see they are the same chain rather than two logs that
    happen to agree.
    """
    rows = _rows(conn, since=since)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps({
                "id": row.get("id"),
                "timestamp": row.get("ts"),
                "agent": row.get("agent", ""),
                "event_type": row.get("event_type"),
                "allowed": bool(row.get("allowed")),
                "resource": row.get("resource", ""),
                "rule": row.get("rule", ""),
                "reason": row.get("reason", ""),
                "extra": row.get("extra", ""),
                "chained": bool(row.get("chained")),
            }, ensure_ascii=False, sort_keys=True) + "\n")
    return {"path": str(out), "entries": len(rows)}
