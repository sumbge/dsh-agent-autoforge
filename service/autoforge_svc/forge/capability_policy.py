"""Default-deny reach policy — the executable form of the Intent Manifest.

Ported from `siyad01/agentbox` `internal/policy/engine.go`, vendored at
`D:\\Users\\china\\Desktop\\项目_开发\\_vendor\\agentbox` (Go, 1 star). That file is a
working implementation of the AgenticOS idea (arXiv 2606.21129): the agent states
its intent, the system admits only what that intent requires, and every check
returns a *rule receipt* naming the rule that decided — so a denial can be read
back months later without re-running anything.

What is faithful to the Go source
---------------------------------
  * default deny in all four dimensions — filesystem read, filesystem write,
    network, tool — and credentials allowed by enumeration only
  * deny rules consulted before allow rules, in every dimension
  * the receipt: every Decision carries (allowed, reason, rule), and the rule
    string is what an audit reader greps for
  * the same three matchers: path (exact, subtree, glob), host (exact,
    `*.suffix`, `*`), tool (exact, `prefix*`, `*suffix`, `*`)
  * a port that keeps the colon port out of the host comparison

Three deliberate differences, each because the original is wrong here
--------------------------------------------------------------------
  * `expandPath` in the Go source resolves `~` with `filepath.Abs("~")`, which
    is the *working directory*, not the home directory: `~/x` becomes
    `$PWD/~/x`. A faithful copy of a wrong answer is not a port. `~` is the home
    directory here.
  * Rules are written with `/` and targets are normalised to `/` before
    comparison. Manifests copied from the Linux sources would otherwise never
    match under `nt`.
  * Windows-only honesty: this is an *in-process* gate. It decides, and can
    refuse, but it does not stop a process that never asks. Kernel-level
    enforcement on this host is the job object in `forge/containment.py`, which
    bounds memory, processes and CPU and cannot be declined. `enforcement()`
    says which dimensions are which rather than letting a reader assume they are
    all equally hard.
"""
from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

__all__ = ["Decision", "Permissions", "PolicyEngine", "path_matches",
           "host_matches", "tool_matches", "from_manifest", "enforcement"]


# -- the receipt ---------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """The verdict, the rule that produced it, and the dimension it belongs to.

    `rule` is not decoration. "denied" without a rule name is a fact nobody can
    act on; "filesystem.write: D:/repos/other" says which list to edit.

    `dimension` is the call site saying *which* question it asked
    (filesystem/network/tool/credential). The Go source gets this for free --
    `NetworkEvent` and `ToolEvent` are different functions there -- so the rule
    string never has to carry it. Inferring it here from `"policy: default_deny"`
    is not possible: that rule is shared by all four dimensions and means the
    same thing in each. An audit row that guessed would file a network denial
    under "policy" and the reader would go and look at the wrong list.
    """

    allowed: bool
    reason: str
    rule: str
    dimension: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason,
                "rule": self.rule, "dimension": self.dimension}

    def __bool__(self) -> bool:                      # `if engine.check_tool(t):`
        return self.allowed


def _allow(reason: str, rule: str, dimension: str = "") -> Decision:
    return Decision(True, reason, rule, dimension)


def _deny(reason: str, rule: str, dimension: str = "") -> Decision:
    return Decision(False, reason, rule, dimension)


# -- what the agent declares it intends ----------------------------------

@dataclass
class Permissions:
    """One intent's reach. Empty list means *nothing*, never *everything*.

    That is the whole design: a manifest that forgets to mention the network has
    no network, so forgetting is safe. An allow-everything default would make
    every omission a grant.
    """

    filesystem_read: list[str] = field(default_factory=list)
    filesystem_write: list[str] = field(default_factory=list)
    filesystem_deny: list[str] = field(default_factory=list)
    network_allow: list[str] = field(default_factory=list)
    network_deny: list[str] = field(default_factory=list)
    tools_allow: list[str] = field(default_factory=list)
    tools_deny: list[str] = field(default_factory=list)
    credentials: list[str] = field(default_factory=list)
    alert_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filesystem": {"read": list(self.filesystem_read),
                           "write": list(self.filesystem_write),
                           "deny": list(self.filesystem_deny)},
            "network": {"allow": list(self.network_allow),
                        "deny": list(self.network_deny)},
            "tools": {"allow": list(self.tools_allow),
                      "deny": list(self.tools_deny)},
            "credentials": list(self.credentials),
            "audit": {"alert_on": list(self.alert_on)},
        }


# -- matchers, ported one for one ---------------------------------------

def _normalise(path: str) -> str:
    """Compare in one separator, and without a trailing slash.

    A manifest written on Linux says `D:/x/`; a Windows caller passes `D:\\x`.
    Both mean the same directory, and a policy engine that cannot see that would
    deny a path its own manifest allows.
    """
    return path.replace("\\", "/").rstrip("/") or "/"


def expand_path(path: str, home: str | None = None) -> str:
    """`~` means the home directory.

    The Go original resolves it with `filepath.Abs("~")`, which is the working
    directory -- see the module docstring.
    """
    if path == "~" or path.startswith("~/"):
        base = home or str(Path.home())
        rest = path[2:] if path.startswith("~/") else ""
        return os.path.join(base, rest) if rest else base
    return path


def path_matches(target: str, rule: str, home: str | None = None) -> bool:
    """Exact, subtree, or glob — in that order, first hit wins."""
    t = _normalise(expand_path(target, home))
    r = _normalise(expand_path(rule, home))
    if t == r:
        return True
    if t.startswith(r + "/"):
        return True
    return fnmatch.fnmatch(t, r)


def host_matches(host: str, rule: str) -> bool:
    """`*`, exact, or `*.suffix` — the two wildcard forms a manifest needs.

    A bare `*` is anchors-on: it says "any host", which is a deliberate grant
    rather than an accident, because the default when no rule matches is deny.
    """
    if rule == "*":
        return True
    if rule == host:
        return True
    if rule.startswith("*."):
        return host.endswith(rule[1:])
    return False


def tool_matches(tool: str, pattern: str) -> bool:
    if pattern == "*":
        return True
    if pattern == tool:
        return True
    if pattern.endswith("*"):
        return tool.startswith(pattern[:-1])
    if pattern.startswith("*"):
        return tool.endswith(pattern[1:])
    return False


# -- the engine ----------------------------------------------------------

class PolicyEngine:
    """Answer four questions, deny by default, and say which rule decided."""

    def __init__(self, permissions: Permissions, *, home: str | None = None):
        self.permissions = permissions
        self.home = home

    # -- filesystem ------------------------------------------------------

    def check_filesystem(self, op: str, path: str) -> Decision:
        p = self.permissions
        for rule in p.filesystem_deny:
            if path_matches(path, rule, self.home):
                return _deny(f"path {path!r} is explicitly denied",
                             f"filesystem.deny: {rule}", "filesystem")
        if op == "read":
            allow = p.filesystem_read
        elif op == "write":
            allow = p.filesystem_write
        else:
            return _deny(f"unknown filesystem operation {op!r}",
                         "policy: unknown_op", "filesystem")
        for rule in allow:
            if path_matches(path, rule, self.home):
                return _allow(f"path {path!r} matches allow rule",
                              f"filesystem.{op}: {rule}", "filesystem")
        return _deny(f"path {path!r} not in {op} allow list",
                     "policy: default_deny", "filesystem")

    # -- network ---------------------------------------------------------

    def check_network(self, host: str) -> Decision:
        p = self.permissions
        bare = host.rsplit(":", 1)[0] if ":" in host else host
        for rule in p.network_deny:
            if host_matches(bare, rule):
                return _deny(f"host {bare!r} is denied", f"network.deny: {rule}", "network")
        if not p.network_allow:
            return _deny(f"host {bare!r}: no network allow rules defined",
                         "policy: default_deny", "network")
        for rule in p.network_allow:
            if host_matches(bare, rule):
                return _allow(f"host {bare!r} matches allow rule",
                              f"network.allow: {rule}", "network")
        return _deny(f"host {bare!r} not in network allow list",
                     "policy: default_deny", "network")

    # -- tools -----------------------------------------------------------

    def check_tool(self, tool: str) -> Decision:
        p = self.permissions
        for pattern in p.tools_deny:
            if tool_matches(tool, pattern):
                return _deny(f"tool {tool!r} matches deny pattern",
                             f"tool.deny: {pattern}", "tool")
        for pattern in p.tools_allow:
            if tool_matches(tool, pattern):
                return _allow(f"tool {tool!r} matches allow pattern",
                              f"tool.allow: {pattern}", "tool")
        return _deny(f"tool {tool!r} not in allow list", "policy: default_deny", "tool")

    # -- credentials -----------------------------------------------------

    def check_credential(self, name: str) -> Decision:
        """Enumeration only: there is no wildcard form for a secret.

        A pattern that can match a credential you never thought about is a
        leak with a rule name attached.
        """
        if name in self.permissions.credentials:
            return _allow(f"credential {name!r} is in manifest",
                          f"credential: {name}", "credential")
        return _deny(f"credential {name!r} is not declared in manifest",
                     "policy: default_deny", "credential")

    # -- audit -----------------------------------------------------------

    def should_alert(self, event_type: str) -> bool:
        return event_type in self.permissions.alert_on


# -- binding it to the manifest that already exists ----------------------

def from_manifest(manifest: Any, *, roots: Iterable[str] = ()) -> PolicyEngine:
    """Derive reach from the manifest `forge/manifest.py` already declares.

    `manifest.py` answers "how much machine" (memory, processes, CPU, wall);
    this module answers "which paths, hosts, tools, secrets". They are one
    intent, so they are read from one object: a second declaration would be a
    second thing to keep in sync, and the one that drifts is the one nobody
    re-reads.

    The translation is deliberately conservative. `network=True` becomes an
    empty allow list, not `["*"]`: the resource manifest can only say *whether*
    the network is wanted, never to whom, so anything wider than nothing would
    be this module inventing a grant. A caller that needs an egress list passes
    one in.
    """
    perms = Permissions()
    for root in roots:
        perms.filesystem_read.append(str(root))
    if getattr(manifest, "writes", False):
        for root in roots:
            perms.filesystem_write.append(str(root))
    if getattr(manifest, "network", False):
        perms.network_allow = []           # see docstring: not widened
    if getattr(manifest, "general_purpose", False):
        perms.tools_allow = ["*"]
    perms.alert_on = ["limit_breached", "policy_denied"]
    return PolicyEngine(perms)


# -- honesty about hardness ---------------------------------------------

def enforcement() -> dict[str, str]:
    """Which dimensions are refused by the kernel and which by convention.

    Same posture as `manifest.py`'s note on `network`: recorded, not pretended.
    A table that called all four "enforced" would read as a stronger guarantee
    than this host can give, and the difference matters exactly when someone is
    deciding whether to run untrusted code.
    """
    return {
        "filesystem_read": "gated: refused in-process before the read",
        "filesystem_write": "gated: refused in-process before the write",
        "network": "gated at the call sites that route through the engine",
        "tool": "enforced: a denied tool is not dispatched",
        "credential": "enforced: undeclared names are not decryptable here",
        "memory_cpu_processes": "enforced by the kernel: job object, "
                                "forge/containment.py",
    }
