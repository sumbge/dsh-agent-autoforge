"""The single door out: every outbound call goes through one admitted host check.

Port of upstream `autoforge/egress.py` (267 lines, commit c58a854), adapted in
three places and otherwise left alone -- see "What changed" at the bottom.

Why a gateway rather than seven checks
--------------------------------------
This branch had no egress decision anywhere. Seven call sites opened sockets
wherever they were told: `notify.py`'s webhook, `service.py`'s
`_market_search` / `_peer_lookup` / `_discover_upstream` / `_prelookup_market`,
`tars_port.py`'s toolmarket probe, and `core/llm.py`'s POSTs. A rule nobody
consults is a comment, and the count of places to remember is the count of
places that will be forgotten.

What it is, and what it is not
------------------------------
*It is*: an admit decision per outbound host, on the same four-part receipt as
every other policy check, with the verdict and the rule written to the audit
chain. A denied host does not get a connection, and the reason is greppable
afterwards.

*It is not* a kernel firewall. Code that calls `socket` directly -- a forged
tool, a vendored dependency, anything not routed through here -- is not stopped
by this. On Windows that gap can only be closed by the OS, and this module says
so in `coverage()` rather than implying otherwise. What it does buy is the thing
that was actually missing: the agent's own reach is *decided*, not incidental,
and the decision is recorded.

Default posture: `allow`, explicitly
------------------------------------
An empty host list here means **allow**, and that is a deliberate choice with a
reason rather than a convenience. The agent's whole job is fetching, forging and
reaching the network; a default-deny gateway that shipped as the only path out
would have to be disabled to get any work done, and a gate that is always off is
worse than a gate that is honestly open. So the field starts absent and a
deployment that wants a closed boundary sets `AUTOFORGE_EGRESS=closed` -- and
can see, from `coverage()` and from the audit rows, exactly which hosts were
admitted.

What changed from upstream
--------------------------
1. `from autoforge import audit` -> `from . import audit` (this package is
   `autoforge_svc`; upstream's absolute import is the one thing that never
   survives the move).
2. `SITES` is rewritten for this branch: `mcp` and `webtools` do not exist here,
   `llm.chat` and `notify` do and are now routed, and the four `service.py`
   fetch helpers are added. `covered` is still set by the wiring, and
   `chk_egress.py` checks it against the source rather than trusting the table.
3. `bind()` is called from `Service.__init__` instead of from an agent's
   constructor, because that is where this branch opens its store.
"""
from __future__ import annotations

import os
import socket
import time
import urllib.parse
from typing import Any

__all__ = ["GatewayDenied", "Gateway", "gateway", "reset", "admit", "coverage",
           "bind", "DEFAULT_ALLOW", "ALLOW_ALL", "SITES"]

#: Hosts the agent needs to function, used only when `AUTOFORGE_EGRESS=closed`.
#: Loopback is in the list because the local market, the local MCP servers and
#: the bus all live there; a boundary that cut them off would be measuring the
#: wrong thing entirely.
DEFAULT_ALLOW = ("api.aiping.cn", "127.0.0.1", "localhost", "::1",
                 "*.openai.com", "*.anthropic.com")

ALLOW_ALL = "*"

_TRUTHY = {"closed", "deny", "strict", "1", "on", "true", "yes"}
_FALSEY = {"open", "allow", "off", "0", "false", "no"}


class GatewayDenied(RuntimeError):
    """The host was refused. Carries the rule, so the caller can name it.

    A separate exception from the policy module's `Decision` on purpose: callers
    catch this to produce a sentence, and they must not accidentally catch a
    `KeyError` or a socket error under the same name.
    """

    def __init__(self, host: str, rule: str, reason: str):
        super().__init__("egress denied for %r: %s (%s)" % (host, reason, rule))
        self.host = host
        self.rule = rule
        self.reason = reason


def _host_of(url: str) -> str:
    """The host, and only the host: scheme, userinfo, port and path dropped.

    `user@host` matters -- `http://api.aiping.cn@evil.test/` has `evil.test` as
    its host, and a naive split on `@` or on `/` gets the allow list to admit the
    wrong name.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.hostname:
        return parsed.hostname
    if parsed.scheme and parsed.path:
        return parsed.path.split("/")[0].split(":")[0]
    return (url or "").split("/")[0].split(":")[0]


def parse_hosts(value: str) -> list[str]:
    return [h.strip() for h in (value or "").split(",") if h.strip()]


class Gateway:
    """Admit or refuse outbound hosts, and remember that it did.

    A process-wide instance (`gateway()`) because the alternative -- threading a
    policy object through every fetch, client and tool -- means the one place
    somebody forgot is the one that leaks, and nobody would know.
    """

    def __init__(self, *, allow: tuple[str, ...] | list[str] | None = None,
                 mode: str | None = None, engine: Any = None, conn: Any = None,
                 name: str = "autoforge"):
        self.mode = (mode or os.environ.get("AUTOFORGE_EGRESS", "open")).strip().lower()
        self.allow = tuple(allow) if allow is not None else self._from_env()
        if self.closed and ALLOW_ALL in self.allow:
            # Contradictory configuration is refused, not interpreted. One
            # field says "there is a boundary here"; the list says "there is
            # none". Either reading would be a guess about which one the
            # operator meant, and a boundary that guesses is not one -- so this
            # raises, and the message names both fields.
            raise ValueError(
                "egress: mode is %r (closed) but the allow list contains %r. "
                "Closed means the list decides; * means there is no list. "
                "Remove one -- AUTOFORGE_EGRESS_ALLOW, or AUTOFORGE_EGRESS=closed."
                % (self.mode, ALLOW_ALL))
        self.engine = engine
        self.conn = conn
        self.name = name

    # -- configuration ---------------------------------------------------

    def _from_env(self) -> tuple[str, ...]:
        hosts = parse_hosts(os.environ.get("AUTOFORGE_EGRESS_ALLOW", ""))
        if hosts:
            return tuple(hosts)
        if self.closed:
            return tuple(DEFAULT_ALLOW)
        return (ALLOW_ALL,)

    @property
    def closed(self) -> bool:
        mode = self.mode
        if mode in _TRUTHY:
            return True
        if mode in _FALSEY:
            return False
        return False          # an unreadable setting is not a silently closed door

    def rules(self) -> list[str]:
        return list(self.allow)

    # -- the decision ----------------------------------------------------

    def check(self, url: str) -> dict[str, Any]:
        """Judge one URL. Returns a receipt; does not raise and does not connect."""
        host = _host_of(url)
        mode = "closed" if self.closed else "open"
        base = {"host": host, "dimension": "network", "mode": mode,
                "ts": time.time()}

        if not self.allow:
            # An empty list in *either* mode is a refusal, and the two cases are
            # reported apart: closed-with-nothing-listed is a boundary somebody
            # configured; open-with-nothing-listed is a bug in the caller, and
            # saying "denied" for both would hide which one it is.
            return dict(base, allowed=False, rule="egress.deny: empty allow list",
                        reason="no egress allow rules, in %s mode" % mode)

        if not self.closed and ALLOW_ALL in self.allow:
            return dict(base, allowed=True, rule="egress.allow: *",
                        reason="egress is open: host %r admitted" % host)

        # `_host_matches` only accepts an exact host, a `*.suffix` form or `*`.
        # An IP literal can therefore never sneak in through a `*.example.com`
        # rule -- `93.184.216.34` does not end in `.example.com` -- so no special
        # case for addresses is needed, and adding one was the bug in the first
        # draft of this method.
        # `_first_match` would happily match `*`, so the branch above is not a
        # guard on its own -- it is a shortcut. The guard has to be here, where
        # the list is actually consulted: in closed mode a wildcard entry is
        # skipped rather than honoured. (Found by my own test: dropping the
        # branch above changed nothing, because this line admitted everything
        # anyway. Upstream has the same hole; a guard whose effect is produced
        # by the code it guards is not a guard.)
        rules = self.allow
        if self.closed:
            rules = tuple(r for r in self.allow if r != ALLOW_ALL)
        rule = _first_match(host, rules)
        if rule:
            return dict(base, allowed=True, rule="egress.allow: %s" % rule,
                        reason="host %r matches allow rule" % host)
        return dict(base, allowed=False, rule="egress.deny: no allow rule",
                    reason="host %r is not in the egress allow list" % host)

    def admit(self, url: str, *, tool: str = "") -> dict[str, Any]:
        """Check, record, and raise only if refused.

        Called at every covered call site, and callable by anything that wants
        the check without the connection -- so a bad host is reported at
        configuration time rather than at first use.
        """
        receipt = self.check(url)
        receipt["tool"] = tool
        recorded = None
        if self.conn is not None:
            recorded = self._record(receipt)
        if not receipt["allowed"]:
            raise GatewayDenied(receipt["host"], receipt["rule"], receipt["reason"])
        receipt["recorded"] = recorded
        return receipt

    def _record(self, receipt: dict[str, Any]) -> Any:
        """Write the receipt to the audit chain. A failure here must not eat the
        call: losing the log is bad, losing the work because the log is bad is
        worse -- and the fact that it failed is itself reported to the caller."""
        try:
            from . import audit
            event = "network_allow" if receipt["allowed"] else "network_deny"
            # `kind="egress"`, not the default "policy": this row records what
            # the agent DID, not what it was allowed to do. Filing an admitted
            # host under `kind='policy'` made a successful network call look
            # like a policy verdict to anything counting them.
            return audit.record(self.conn, {
                "event_type": event, "allowed": receipt["allowed"],
                "resource": receipt["host"], "rule": receipt["rule"],
                "reason": receipt["reason"], "agent": self.name,
                "extra": receipt.get("tool", ""), "ts": receipt["ts"],
            }, kind="egress")
        except Exception as exc:                              # noqa: BLE001
            return {"recorded": False, "error": str(exc)[:200]}

    # -- reporting -------------------------------------------------------

    def coverage(self) -> dict[str, Any]:
        """What this gate does and does not cover, in words.

        `enforcement()`: the difference between a boundary and a policy is
        exactly whether an uncooperative caller is stopped, and a table that
        blurred it would be read as a firewall.
        """
        return {
            "mode": "closed" if self.closed else "open",
            "allow": list(self.allow),
            "bound": self.conn is not None,
            "gateway": "in-process, at the call sites routed through admit()",
            "covered_sites": [s["name"] for s in SITES if s["covered"]],
            "known_unrouted": [s["name"] for s in SITES if not s["covered"]],
            "not_covered": ("code that opens a socket directly -- a forged tool, "
                            "a vendored library -- is not stopped by this; on "
                            "this host only the OS could, and it is not"),
        }


def _host_matches(host: str, rule: str) -> bool:
    if rule == ALLOW_ALL:
        return True
    if rule == host:
        return True
    if rule.startswith("*."):
        return host.endswith(rule[1:])
    return False


def _first_match(host: str, rules) -> str:
    for rule in rules:
        if _host_matches(host, rule):
            return rule
    return ""


def _is_ip(host: str) -> bool:
    try:
        socket.inet_aton(host)
        return True
    except OSError:
        return ":" in host and all(c in "0123456789abcdef:" for c in host.lower())


# -- the process-wide instance -------------------------------------------

#: Every outbound call site, and whether it is routed through `admit()`.
#:
#: A **covered** entry says which FUNCTION the claim is about (`func`), which
#: class owns it when the name is not unique (`cls`), and how many `admit()`
#: calls it accounts for (`calls`). All three are asserted against the
#: source with `ast`.
#:
#: `cls` is not decoration. `llm.py` defines five methods named `chat` --
#: one per client class -- and my first version picked whichever `ast`
#: walk reached first and landed on the 7-line abstract stub, so the
#: assertion reported `0 routed calls in chat(), 1 claimed` about the wrong
#: function entirely. A name is not an identity here.
#:
#: An **unrouted** entry makes a different claim: that this code path does
#: not go through the gate. It carries no count, because a count of zero in
#: a file that routes something else would be measuring the file rather
#: than the path. What is asserted about it is the admission itself -- the
#: reason has to be a real sentence, not a placeholder, or "known_unrouted"
#: becomes a place to put things people stop reading.
#: This is a *ledger of omissions* as much as a feature list. An egress boundary
#: whose documentation names only what it covers invites the reader to assume the
#: rest is covered too; naming the unrouted ones makes the gap a task instead of
#: a surprise. `covered` is set by the wiring itself -- `chk_egress.py` reads
#: each file and fails if a site marked covered does not call `admit`.
SITES: tuple[dict[str, Any], ...] = (
    {"name": "llm._post_watchable", "module": "autoforge/core/llm.py",
     "func": "_post_watchable", "calls": 1,
     "why": "the model endpoint: the connection that carries a credential, "
            "and the one an operator is most likely to point somewhere else",
     "covered": True},
    {"name": "llm.chat (streaming)", "module": "autoforge/core/llm.py",
     "func": "chat", "cls": "DSHGatewayClient", "calls": 1,
     "why": "the streaming path: a second way to the same endpoint, and a "
            "gate that covered only the first would be a hole the size of "
            "the path actually used",
     "covered": True},
    {"name": "notify._send_webhook", "module": "autoforge/notify.py", "func": "_send_webhook", "calls": 1,
     "why": "a URL the operator configured; still a URL, still outbound",
     "covered": True},
    {"name": "service._market_search", "module": "autoforge/service.py", "func": "_market_search", "calls": 1,
     "why": "the tool market: a configurable base URL, and the response is "
            "parsed, not merely displayed",
     "covered": True},
    {"name": "service._peer_lookup", "module": "autoforge/service.py", "func": "_peer_lookup", "calls": 2,
     "why": "a peer address taken from an environment variable",
     "covered": True},
    {"name": "service._discover_upstream", "module": "autoforge/service.py", "func": "_discover_upstream", "calls": 1,
     "why": "asks a peer what it can do; same trust boundary as _peer_lookup",
     "covered": True},
    {"name": "service._prelookup_market", "module": "autoforge/service.py", "func": "_prelookup_market", "calls": 1,
     "why": "the market again, on a short budget, before a forge",
     "covered": True},
    {"name": "service._gateway_models", "module": "autoforge/service.py", "func": "_gateway_models", "calls": 1,
     "why": "asks the model gateway what it serves -- the answer decides which "
            "model id the forge pipeline and every scheduled task will use, so "
            "it is a request whose result changes behaviour, not a health check",
     "covered": True},
    {"name": "service._reads_tool_results", "module": "autoforge/service.py", "func": "_reads_tool_results", "calls": 1,
     "why": "a model probe that POSTs to /chat/completions with a hand-written "
            "payload; it carries the same gateway credential as a real turn, "
            "and its answer decides which model the forge is allowed to use. "
            "Leaving it out of SITES would make the capability check the one "
            "outbound call an operator could not refuse",
     "covered": True},
    {"name": "tars_port.probe_host", "module": "autoforge/tars_port.py", "func": "probe_host", "calls": 2,
     "why": "loopback and a DNS resolution -- the host probe, not a fetch",
     "covered": True},
    {"name": "sandbox.run (child reach probe)", "module": "autoforge/forge/sandbox.py",
     "why": "runs inside the child process, whose whole purpose is to test what "
            "the child can reach; gating it would mean the probe reports this "
            "process's policy rather than the sandbox's reach",
     "covered": False},
    {"name": "forged tool code", "module": "autoforge/forge/sandbox.py",
     "why": "agent-authored code calls whatever it likes in its own child "
            "process; only the OS can stop it, and on this host it does not",
     "covered": False},
    {"name": "vendored tars code", "module": "autoforge/tars_port.py",
     "why": "third-party code, vendored byte-for-byte; it is not edited to "
            "route through here",
     "covered": False},
)


_GATEWAY: Gateway | None = None


def gateway() -> Gateway:
    """The one gateway. Built once per process, from the environment."""
    global _GATEWAY
    if _GATEWAY is None:
        _GATEWAY = Gateway()
    return _GATEWAY


def reset(**kwargs) -> Gateway:
    """Replace it. For tests and for a process that has just read new settings."""
    global _GATEWAY
    _GATEWAY = Gateway(**kwargs) if kwargs else Gateway()
    return _GATEWAY


def admit(url: str, *, tool: str = "") -> dict[str, Any]:
    return gateway().admit(url, tool=tool)


def coverage() -> dict[str, Any]:
    return gateway().coverage()


def bind(conn: Any, *, name: str = "autoforge", allow=None, mode: str | None = None) -> Gateway:
    """Point the gateway at a ledger so its receipts are recorded.

    Called once, where the store is opened. Without it the gateway still decides
    -- refusing is not conditional on having somewhere to write -- but the
    decision leaves no trace, and a boundary with no record of what it admitted
    is a boundary nobody can audit. Same posture as `chaining.py`: the write is
    best-effort, and a failure to write is reported rather than swallowed into a
    false negative.
    """
    return reset(conn=conn, name=name, allow=allow, mode=mode)
