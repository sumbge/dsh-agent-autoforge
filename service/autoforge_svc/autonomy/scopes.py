"""What each thing this service can run is allowed to touch.

Two vocabularies have to meet here, and for a long time they did not.

`forge/validity.py` knows seven scope names and what effect labels each may
exercise. `autonomy/confirm.py` knows how to turn a scope into the policy
freedoms it depends on. Nothing in this package ever *stated* a scope for
anything, so `effect_signature` was the empty string everywhere and
`declared_capabilities('')` fell through to `undeclared` -- which is all eight
capabilities. Measured 2026-11-03 on the real ledger: all six forged tools had
`effect_signature = ''`, so under any policy that switches off one freedom,
every one of them was treated as able to do everything.

Upstream fills this gap with `BUILTIN_SCOPES`, a 79-row dict in `agent.py`
applied inside `_add()` -- the function every builtin tool passes through on
its way into the registry. This package has no such function: its 71 wire
methods are a hand-written dispatch dict and never enter the registry at all
(measured: on the real database `registry.names()` lists the six forged tools
and no builtin). Copying the upstream table would therefore have produced
decoration -- a dict nothing reads. So this module provides two things, and
they only make sense together:

  `METHOD_SCOPES` / `scope_for`     -- the 71 wire methods, by name.
  `derived_scope`                   -- what a forged tool's own code says.

`Service.gate_method` reads the first; `Service.tool_define` and the
load-at-startup loop use the second. `chk_scopes.py` refuses to accept either
half alone.

The upstream table could not be copied verbatim for a second reason: 25 of its
names do not exist in this service (`browser_*`, `mcp_*`, `spawn_agent`,
`design_team`, `see`, `draw`, `animate`, `vision_status`, `media_status`,
`amend_self`, `set_autonomy`, `import_package`, `tars_*`, plus `skill_list`
which here is called `skills`), and 13 names here do not exist there
(`anchor_ledger`, `describe`, `events`, `log_event`, `mission_history`,
`ping`, `rehab_tool`, `report`, `system_task_command`, `tool_call`,
`tool_define`, `tool_list`, `tool_usage`). The granularity, though, is
upstream's and worth keeping: these labels are answers to the gate's four
questions (does it read, write, reach the network, install), not a taxonomy of
what the tool is about. That is why `cpu_probe` is `read_only` even though it
really does start a process -- it cannot wedge a machine, and saying `system`
would make it demand execution rights it does not need.
"""
from __future__ import annotations

from typing import Any

#: The wire methods, by what they touch.
#:
#: Sorted by scope then name in the literal so a reader can see at a glance
#: which methods make the largest claim. Every key must exist in
#: `Service.methods`; `chk_scopes.py` fails on either a missing or a stale
#: entry, because a table that drifts from the dispatch dict is a table that
#: answers about a service that no longer exists.
METHOD_SCOPES: dict[str, str] = {
    # -- read_only: reports on state, touches nothing --------------------
    "ping": "read_only",
    "describe": "read_only",
    "context_brief": "read_only",
    "recall": "read_only",
    "mission_list": "read_only",
    "mission_show": "read_only",
    "mission_history": "read_only",
    "skill_errors": "read_only",
    "skills": "read_only",
    "skill_view": "read_only",
    "my_history": "read_only",
    "verify_ledger": "read_only",
    "tool_usage": "read_only",
    "list_tools": "read_only",
    "tool_list": "read_only",
    "describe_tool": "read_only",
    # Reads a spec and its own source; writes nothing but a `review` ledger
    # row. Filed under read_only rather than local_write because the one
    # thing it writes is the receipt for having read -- and a reviewer that
    # needs write permission to report what it found is a reviewer nobody
    # can afford to ask.
    "review_tool": "read_only",
    "report_bridge_tools": "read_only",
    "tool_exposure": "read_only",
    "events": "read_only",
    "report": "read_only",
    "find_gaps": "read_only",
    "schedule_list": "read_only",
    "schedule_tick": "read_only",
    "runs_status": "read_only",
    "system_task_command": "read_only",
    "my_capabilities": "read_only",
    "notify_channels": "read_only",
    "tars_probe": "read_only",        # one read-only pass over the vendored tree
    "tars_skills": "read_only",       # reads a borrowed procedure's text
    "gpu_probe": "read_only",
    "gpu_occupancy": "read_only",
    "gpu_units_audit": "read_only",
    "gpu_verify": "read_only",
    "cpu_probe": "read_only",
    "cpu_runs_here": "read_only",
    "cpu_preflight": "read_only",
    "cpu_units_audit": "read_only",
    "cpu_cache_stats": "read_only",
    # -- local_write: changes this service's own store or files ----------
    "remember": "local_write",
    "forget": "local_write",
    "mission_open": "local_write",
    "mission_note": "local_write",
    "mission_close": "local_write",
    "mission_focus": "local_write",
    "mission_block": "local_write",
    "mission_sweep": "local_write",
    "mission_wake": "local_write",
    "log_event": "local_write",
    "anchor_ledger": "local_write",
    "skill_write": "local_write",
    "skill_forget": "local_write",
    "skill_port": "local_write",
    # Nine files appear under the skills directory. Split from `tars_skills` on
    # purpose: a tool named like a read that mints files would make its
    # effect_signature a lie, and the signature is what the gate reads.
    "tars_port_skills": "local_write",
    "retire_tool": "local_write",
    "rehab_tool": "local_write",
    "runs_forget": "local_write",
    # Deleting a record rewrites the directory, so this is not read_only. It is
    # also why it is a tool and not part of the tick: the gate can see a call
    # somebody made and refuse it, and cannot see a timer.
    "runs_prune": "local_write",
    "resume_run": "local_write",
    "schedule_add": "local_write",
    "schedule_done": "local_write",
    "schedule_cancel": "local_write",
    # -- network: leaves the machine -------------------------------------
    "notify_send": "network",
    # -- system: runs code, starts processes, executes a task ------------
    "run_python": "system",
    "tool_define": "system",          # runs the new body once, in a subprocess
    "tool_call": "system",            # runs whatever the named tool is
    "forge_tool": "system",
    "evaluate_tool": "system",
    "evolve_tool": "system",
    "schedule_run_due": "system",     # hands the task text to a live agent
    "install_system_task": "system",
    "gpu_compile": "system",
    "gpu_bench": "system",
    "cpu_compile": "system",
    "cpu_run_isolated": "system",
    "cpu_tune": "system",
}


def scope_for(method: str) -> str:
    """The declared scope of a wire method, or `""` when it has none.

    Empty is not a default of `read_only`: an unlabelled method has not said
    what it touches, and `confirm.declared_capabilities` reads that as
    `undeclared` -- every capability. Guessing `read_only` here would be this
    module asserting innocence on behalf of code it has not looked at.
    """
    return METHOD_SCOPES.get(method, "")


#: Scopes ordered by how much they permit, so `derived_scope` can pick the
#: smallest one that covers the labels it found.
_BY_SIZE: tuple[str, ...] = (
    "read_only",
    "local_write",
    # Without this entry `derived_scope` over `os.remove(p)` fell through
    # every scope and answered `undeclared` -- all eight capabilities for a
    # tool whose only visible effect is a deletion. Measured 2026-11-05.
    "destructive",
    "network",
    "system",
    "privileged",
)


def derived_scope(code: str) -> str:
    """The narrowest scope whose allowances cover what `code` demonstrably does.

    The floor is `read_only`, never `pure`. `validity.audit_effects` can see a
    write, a subprocess, the network, `eval` and unpickling; it cannot see a
    read, because `open(path)` with no mode and `os.stat` look like nothing at
    all to a regex. Returning `pure` for code the audit could not label would
    be claiming innocence on the strength of not having looked -- the exact
    shape of defect this package exists to catch. `read_only` costs nothing
    under a policy that permits reading, and is gated under one that does not.

    A scope that turns out too narrow is caught by `validity._scope_gate`,
    which compares the labels it can see against what the declared scope
    permits and fails the tool for exceeding its declaration. So erring narrow
    is fail-closed in both directions.
    """
    from ..forge.validity import SCOPE_ALLOWANCES, audit_effects

    labels = {f.label for f in audit_effects(code or "")}
    if not labels:
        return "read_only"
    for scope in _BY_SIZE:
        if labels <= set(SCOPE_ALLOWANCES.get(scope, frozenset())):
            return scope
    return "undeclared"
