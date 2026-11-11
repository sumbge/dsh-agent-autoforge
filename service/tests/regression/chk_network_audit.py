"""A tool that dials out must be labelled as doing so.

Measured 2026-11-05, before the network rule was widened. `_EFFECT_PATTERNS`
named `socket.socket` and nothing else from that module:

    socket.create_connection((h, 443), timeout=5)   -> []                 read_only
    socket.gethostbyname(h)                         -> []                 read_only
    socket.getaddrinfo(h, 443)                      -> []                 read_only
    socket.socketpair()                             -> []                 read_only
    socket.create_server(("0.0.0.0", 80))           -> []                 read_only
    socket.socket()                                 -> ['network_egress'] network
    urllib.request.urlopen(u)                       -> ['network_egress'] network

The constructor is the rare spelling. So the audit said a tool whose only
visible effect was dialing out touches nothing, `derived_scope` answered
`read_only`, and `review_tool` had nothing to object to when such a tool
declared `pure`.

What this file does NOT claim: that the gate now stops the connection. It does
not. `egress.admit` is in-process and a sandboxed child inherits none of it --
measured the same day, `AUTOFORGE_EGRESS=closed` with a one-host allow list
still returned `{"dns": "198.18.0.18", "tcp": "connected", "http": "status 200,
64 bytes"}` from inside the sandbox. This is the AUDIT half: the label is
correct, so a lie about scope becomes visible. The runtime half needs an OS
filter and `report()["egress"]["not_covered"]` keeps saying so.

The over-match guards matter as much as the under-match ones: a comment and a
bare string are how a widened regex starts accusing honest tools.
"""
import os
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


#: (label, code, should the audit see network_egress)
REACHES = [
    ("socket.create_connection", "import socket\nsocket.create_connection(('h', 443), timeout=5)", True),
    ("socket.gethostbyname", "import socket\nreturn socket.gethostbyname('h')", True),
    ("socket.gethostbyname_ex", "import socket\nreturn socket.gethostbyname_ex('h')", True),
    ("socket.gethostbyaddr", "import socket\nreturn socket.gethostbyaddr('1.2.3.4')", True),
    ("socket.getaddrinfo", "import socket\nreturn socket.getaddrinfo('h', 443)", True),
    ("socket.getfqdn", "import socket\nreturn socket.getfqdn('h')", True),
    ("socket.getnameinfo", "import socket\nreturn socket.getnameinfo(sa, 0)", True),
    ("socket.socketpair", "import socket\nreturn socket.socketpair()", True),
    ("socket.create_server", "import socket\nreturn socket.create_server(('0.0.0.0', 80))", True),
    ("socket.fromfd", "import socket\nreturn socket.fromfd(3, 2, 1)", True),
    ("socket.inet_aton", "import socket\nreturn socket.inet_aton('1.2.3.4')", True),
    ("socket.socket", "import socket\nreturn socket.socket()", True),
    ("spaced attribute access", "import socket\nreturn socket . create_connection(p)", True),
    ("sendall", "s.sendall(b'x')", True),
    ("sendto", "s.sendto(b'x', addr)", True),
    ("urllib.request", "import urllib.request\nreturn urllib.request.urlopen(u)", True),
    ("urlopen bare", "return urlopen(u)", True),
    ("requests", "import requests\nreturn requests.get(u)", True),
    ("ftplib", "import ftplib\nreturn ftplib.FTP(h)", True),
    ("smtplib", "import smtplib\nreturn smtplib.SMTP(h)", True),
    ("paramiko", "import paramiko\nreturn paramiko.SSHClient()", True),
    ("xmlrpc.client", "import xmlrpc.client\nreturn xmlrpc.client.ServerProxy(h)", True),
]

QUIET = [
    ("a comment about dialing out", "# socket.create_connection is what we would use\nreturn 1"),
    ("no network at all", "return sum(range(10))"),
    ("urllib.parse", "import urllib.parse\nreturn urllib.parse.urlsplit(u).hostname"),
    ("a local file read", "with open(p) as fh:\n    return fh.read()"),
    ("socket.socketname -- a different word", "return socket.socketname"),
]

#: Known limit, measured 2026-11-05 and NOT introduced by the widening.
#: `audit_effects` strips `#` comments and never string literals, so a docstring
#: that merely mentions a socket call is labelled as making one. Verified against
#: the pre-widening rule: `socket.socket` and `urlopen` inside a docstring were
#: already `network_egress` before today. Recorded here rather than asserted
#: quiet -- an assertion that was never true is not a guard, it is a wish.
OVER_MATCHES = [
    ("docstring mentioning socket.socket", "'''we never call socket.socket here'''"),
    ("docstring mentioning urlopen", "'''this does not urlopen anything'''"),
]


def main():
    os.environ["AUTOFORGE_HOME"] = tempfile.mkdtemp(prefix="chk_network_audit_")
    from autoforge_svc.forge.validity import audit_effects, scope_of_signature
    from autoforge_svc.autonomy.scopes import derived_scope

    print("-- forms that really do reach the network")
    seen = 0
    for name, code, _ in REACHES:
        labels = {f.label for f in audit_effects(code)}
        check("%s carries network_egress" % name, "network_egress" in labels, sorted(labels))
        seen += "network_egress" in labels
    check("every one of them, not most", seen == len(REACHES), "%d of %d" % (seen, len(REACHES)))

    print("-- and the derived scope says so")
    for name, code in (
        ("socket.create_connection", "import socket\nsocket.create_connection(('h', 443), timeout=5)"),
        ("socket.gethostbyname", "import socket\nreturn socket.gethostbyname('h')"),
        ("socket.create_server", "import socket\nreturn socket.create_server(('0.0.0.0', 80))"),
    ):
        check("%s derives 'network', not 'read_only'" % name,
              derived_scope(code) == "network", derived_scope(code))

    print("-- a widened regex must not start accusing honest tools")
    for name, code in QUIET:
        labels = {f.label for f in audit_effects(code)}
        check("%s stays quiet" % name, "network_egress" not in labels, sorted(labels))
        check("and %s still derives 'read_only'" % name,
              derived_scope(code) == "read_only", derived_scope(code))

    print("-- a limit the widening did NOT create, recorded so nobody calls it a regression")
    for name, code in OVER_MATCHES:
        labels = {f.label for f in audit_effects(code)}
        check("%s is still over-matched, as it was before today" % name,
              "network_egress" in labels, sorted(labels))
    check("this list is small and named -- not a place to put surprises",
          len(OVER_MATCHES) == 2, len(OVER_MATCHES))

    print("-- a declaration that contradicts the code is now objectable")
    from autoforge_svc.forge.validity import review_tool
    from autoforge_svc.tools.spec import ToolSpec

    def spec(name, sig, code):
        return ToolSpec(name=name, description="d",
                        parameters={"type": "object", "properties": {},
                                    "required": []},
                        fn=lambda **kw: 0, code=code, effect_signature=sig)

    liar = review_tool(spec("dialer_declared_pure", "pure",
                            "import socket\nreturn socket.create_connection(('h', 443))"))
    check("a tool that declares 'pure' and dials out is objected to",
          liar.ok is False, liar.summary())
    check("and the objection names network_egress",
          "network_egress" in details(liar), details(liar))
    honest = review_tool(spec("dialer_declared_network", "network",
                              "import socket\nreturn socket.create_connection(('h', 443))"))
    check("the same code declaring 'network' passes",
          honest.ok is True, honest.summary())
    check("and its scope reads 'network'", honest.scope == "network", honest.scope)

    print("-- the audit half is not the runtime half (stated, not hidden)")
    from autoforge_svc import egress
    cov = egress.coverage()
    check("report coverage still admits sockets are not stopped",
          "socket" in (cov.get("not_covered") or "").lower(), cov.get("not_covered"))
    check("and the unrouted list still names forged tool code",
          any("forged" in str(x).lower() for x in cov.get("known_unrouted") or []),
          cov.get("known_unrouted"))

    print("TOTAL FAILURES: %d %s" % (len(FAILS), FAILS))
    return 1 if FAILS else 0


def details(verdict):
    return "; ".join(f.detail for f in getattr(verdict, "findings", []) or [] if not f.ok)


if __name__ == "__main__":
    raise SystemExit(main())
