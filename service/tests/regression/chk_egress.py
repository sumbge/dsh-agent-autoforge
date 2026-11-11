"""One door out, and an honest list of the doors that are not it.

Before this file there was no egress gate at all. Seven call sites opened
sockets wherever they were told: `notify.py:252` (webhook), `service.py`'s
`_market_search` / `_peer_lookup` / `_discover_upstream` / `_prelookup_market`,
`tars_port.py`'s toolmarket probe, and `core/llm.py`'s three POSTs. A rule
nobody consults is a comment, and the count of places to remember is the count
of places that will be forgotten.

The gate itself is upstream's, ported. What this file insists on is the part
that is easy to skip when you are porting: `coverage()` must name what it does
NOT cover. A boundary whose documentation lists only what it covers invites the
reader to assume the rest is covered too, and the reader is right to assume it
-- that is what a list of coverage means. So `SITES` in the module is a ledger
of omissions as much as a feature list, and `covered` is asserted against the
wiring rather than written down and hoped for.

Two postures worth stating, because both are load-bearing and neither is the
obvious default:

* **Open is the default, explicitly.** An empty allow list in open mode admits
  everything. The agent's job is reaching the network; a gate that shipped
  closed would have to be disabled to get any work done, and a gate that is
  always off is worse than a gate that is honestly open. What the gate buys is
  that the decision is *made* and *recorded*, not that it is refused.
* **A deny is still a deny with nowhere to write it.** `bind()` points the
  gateway at the ledger; without it the gateway still refuses. Refusing is not
  conditional on having somewhere to report it, and a gate that only works when
  its logger works is a gate that fails open.

Run: see autoforge-claimgate-regression-howto.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICE = os.path.dirname(os.path.dirname(HERE))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

import autoforge_svc.egress as eg                     # noqa: E402
from autoforge_svc import chaining                    # noqa: E402
from autoforge_svc.service import Service             # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print("  OK   " + label)
    else:
        print("  FAIL " + label + ("" if detail == "" else " <- " + str(detail)[:220]))
        FAILS.append(label)


def gw(**kw):
    """A gateway that cannot see the process-wide one or the operator's env."""
    saved = {k: os.environ.get(k) for k in
             ("AUTOFORGE_EGRESS", "AUTOFORGE_EGRESS_ALLOW")}
    for k in saved:
        os.environ.pop(k, None)
    try:
        return eg.Gateway(**kw)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


print("-- the host is the host, whatever the URL dresses itself up as")
# `user@host` is the classic way to get an allow list to admit the wrong name.
for url, want in (
    ("https://api.example.com/v1/chat", "api.example.com"),
    ("http://api.example.com@evil.test/x", "evil.test"),
    ("https://user:pw@good.test:8443/p", "good.test"),
    ("http://127.0.0.1:8000/search", "127.0.0.1"),
    ("https://sub.deep.example.com/", "sub.deep.example.com"),
):
    check("host of %-40s is %s" % (url, want), eg._host_of(url) == want, eg._host_of(url))

print()
print("-- open mode admits, and says which rule did it")
g = gw()
r = g.check("https://anything.test/x")
check("open mode admits an unknown host", r["allowed"] is True, r)
check("and the rule names the wildcard", r["rule"].endswith("*"), r["rule"])
check("the receipt carries the host it judged", r["host"] == "anything.test", r)
check("and the mode it judged in", r["mode"] == "open", r)

print()
print("-- closed mode refuses what is not listed, and names the reason")
g = gw(mode="closed")
r = g.check("https://anything.test/x")
check("closed mode refuses a host outside the list", r["allowed"] is False, r)
check("the rule says it was a missing allow rule", "no allow rule" in r["rule"], r["rule"])
check("the reason repeats the host so it is greppable", "anything.test" in r["reason"], r["reason"])
r = g.check("https://api.aiping.cn/v1")
check("a default-allowed host still passes", r["allowed"] is True, r)
r = g.check("http://127.0.0.1:8000/search")
check("loopback passes, because the market and the bus live there", r["allowed"] is True, r)

print()
print("-- a wildcard entry cannot be used to sneak an address in")
g = gw(mode="closed", allow=("*.example.com",))
check("a subdomain matches", g.check("https://a.example.com/")["allowed"] is True)
check("the bare suffix does not match itself",
      g.check("https://example.com/")["allowed"] is False)
# this is the one an allow list written as a substring would get wrong
check("an IP literal does not pass a suffix rule",
      g.check("http://93.184.216.34/")["allowed"] is False)

print()
print("-- an empty list is a refusal in both modes, and the two are told apart")
for mode in ("open", "closed"):
    r = gw(mode=mode, allow=()).check("https://x.test/")
    check("empty allow list refuses in %s mode" % mode, r["allowed"] is False, r)
    check("and the reason names the mode", mode in r["reason"], r["reason"])

print()
print("-- closed mode and a wildcard in the list is a contradiction, and it raises")
# The first version of this section asserted a *behaviour* -- "closed mode
# refuses even with * in the list" -- and it failed, because the guard in
# `check()` was a shortcut and `_first_match` admitted everything two lines
# later. Upstream has the same hole. The fix is not a better reading of the
# contradiction but a refusal to have one: mode=closed says the list decides,
# `*` says there is no list, and a boundary that guesses is not a boundary.
try:
    gw(mode="closed", allow=("*",))
    check("closed with a wildcard is refused at construction", False,
          "it built a gateway instead of raising")
except ValueError as exc:
    check("closed with a wildcard is refused at construction", True)
    check("and the message names both settings", "AUTOFORGE_EGRESS" in str(exc), str(exc)[:120])
    check("and says what to remove", "Remove one" in str(exc), str(exc)[-90:])
# A constructed gateway whose list is edited afterwards still has to hold: the
# `*` is skipped where the list is consulted, not only where it is read.
g = gw(mode="closed", allow=("ok.test",))
g.allow = ("ok.test", "*")
r = g.check("https://anything.test/")
check("a wildcard added after construction does not open a closed gate",
      r["allowed"] is False, r)
check("and a listed host still passes", g.check("https://ok.test/x")["allowed"] is True)
check("the same list in open mode admits, so the modes really differ",
      gw(mode="open", allow=("*",)).check("https://anything.test/")["allowed"] is True)

print()
print("-- admit() raises only when refused, and refuses without a logger")
g = gw(mode="closed")
try:
    g.admit("https://nope.test/", tool="unit")
    check("admit refuses a denied host", False, "it returned instead of raising")
except eg.GatewayDenied as exc:
    check("admit refuses a denied host", True)
    check("the exception carries the rule", bool(exc.rule), exc.rule)
    check("and the host", exc.host == "nope.test", exc.host)
g = gw(mode="closed", allow=("ok.test",))
receipt = g.admit("https://ok.test/x", tool="unit")
check("admit returns a receipt when it allows", receipt["allowed"] is True, receipt)
check("with no logger there is nothing recorded, and that is not an error",
      "recorded" not in receipt or receipt["recorded"] is None, receipt)

print()
print("-- the receipts land in the ledger, on the same four-part shape")
tmp = tempfile.mkdtemp(prefix="m28_egress_")
db = os.path.join(tmp, "e.db")
# A `ToolStore`, not a bare sqlite3 handle: `chaining.ensure_schema` migrates a
# chain that exists, it does not create one (`no such table: forge_events` if
# you hand it a fresh file). The receipt has to land in the same store the
# service uses, so this uses that store.
from autoforge_svc.store import ToolStore                # noqa: E402
estore = ToolStore(db_path=db)
con = estore.connection()
g = gw(mode="closed", allow=("yes.test",), conn=con, name="chk")
g.admit("https://yes.test/x", tool="unit-allow")
try:
    g.admit("https://no.test/x", tool="unit-deny")
except eg.GatewayDenied:
    pass
rows = [dict(r) for r in con.execute(
        "SELECT kind, prev_hash, payload_hash, payload FROM forge_events ORDER BY id")]
import json as _json
payloads = [_json.loads(r["payload"]) for r in rows]
deny = [p for p in payloads if p.get("event_type") == "network_deny"]
allow = [p for p in payloads if p.get("event_type") == "network_allow"]
check("a refusal is written", len(deny) == 1, len(deny))
check("an admission is written", len(allow) == 1, len(allow))
# `event_type` is the sentence; `kind` is which drawer it is filed in. This
# check was missing, and the gap was caught by `chk_schedule_due` instead:
# an admitted host was filed under `kind='policy'`, so a suite asserting
# "nothing was refused while the policy was fully permissive" went red on a
# call that had succeeded. A test that reads only the payload cannot see a
# filing error, and "where would a reader look for this" is the whole point
# of a ledger.
kinds = sorted({r["kind"] for r in rows})
check("both receipts are filed as egress, not as policy verdicts",
      kinds == ["egress"], kinds)
check("and neither is in the policy drawer",
      not any(p.get("event_type", "").startswith("policy_") for p in payloads),
      [p.get("event_type") for p in payloads])
check("the denial carries the host", deny and deny[0]["resource"] == "no.test", deny[:1])
check("and the rule", deny and "no allow rule" in deny[0]["rule"], deny[:1])
check("and the caller that asked", deny and deny[0]["extra"] == "unit-deny", deny[:1])
check("the rows are chained, not loose", all(r["payload_hash"] for r in rows), len(rows))
estore.close()

print()
print("-- the list of what is NOT covered is the point of the module")
cov = gw().coverage()
check("coverage names a mode", "open" in (cov["mode"], "closed"), cov.get("mode"))
check("and lists the sites it does cover", isinstance(cov["covered_sites"], list), cov.keys())
check("and, separately, the ones it does not", isinstance(cov["known_unrouted"], list), cov.keys())
check("the unrouted list is not empty -- this is not a kernel firewall",
      len(cov["known_unrouted"]) >= 3, cov["known_unrouted"])
check("it says in words what it cannot stop",
      "socket" in cov["not_covered"], cov["not_covered"])
check("both lists together are the whole ledger",
      len(cov["covered_sites"]) + len(cov["known_unrouted"]) == len(eg.SITES), 
      (len(cov["covered_sites"]), len(cov["known_unrouted"]), len(eg.SITES)))

print()
print("-- every site names a real file, and the count of routed calls is checked")
# `covered: True` must mean "this function calls admit", and the check reads the
# source, not the table: a table asserting its own truth is a comment.
#
# Per FUNCION, not per file. service.py holds four of these sites, so a
# file-level count of five admits would satisfy every one of them -- and two of
# my own mutants proved that a file-level check passes when the call that
# matters is deleted. The site entry carries a `func` for exactly this: the
# body is sliced out with `ast` and the admits inside it are counted.
SVC = os.path.join(SERVICE, "autoforge_svc")


def body_of(module_path: str, func: str, cls: str = "") -> str:
    """The source of one function, or "" if it is not there.

    `cls` disambiguates a name that is reused across classes. `llm.py` has five
    methods called `chat`; without the class, `ast.walk` returns whichever it
    reached first -- which was the 7-line abstract stub in `LLMClient`, so the
    assertion complained about a function that was never meant to make a
    request. A name is not an identity.
    """
    import ast
    text = open(module_path, "rb").read().decode("utf-8", "replace")
    tree = ast.parse(text)
    lines = text.splitlines()
    if cls:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and node.name == cls):
                continue
            for inner in node.body:
                if (isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and inner.name == func):
                    end = getattr(inner, "end_lineno", None) or inner.body[-1].lineno
                    return "\n".join(lines[inner.lineno - 1:end])
        return ""
    found = ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func:
            end = getattr(node, "end_lineno", None) or node.body[-1].lineno
            found = "\n".join(lines[node.lineno - 1:end])
            break
    return found


for site in eg.SITES:
    rel = site["module"].replace("autoforge/", "")
    path = os.path.join(SVC, rel.replace("/", os.sep))
    check("site %-28s names a real file" % site["name"], os.path.isfile(path), path)
    if not os.path.isfile(path):
        continue
    func = site.get("func", "")
    cls = site.get("cls", "")
    src_all = open(path, "rb").read().decode("utf-8", "replace")
    src = body_of(path, func, cls) if func else src_all
    label = ("%s.%s" % (cls, func)) if cls else (func or "<file>")
    check("  and names a function that exists", bool(src) if func else True, label)
    if not src:
        continue
    bound: list[str] = []
    for m in re.finditer(r"from\s+\.{1,2}egress\s+import\s+([^\n(]+)", src_all):
        for part in m.group(1).split(","):
            name = part.strip().split(" as ")[-1].strip().split(" ")[0]
            if name:
                bound.append(name)
    # Dedupe the bound names. `llm.py` imports `admit as _admit` inside each of
    # two functions, so the same local name is collected twice, and summing per
    # name then counted one call as two and reported "2 routed calls, 1 claimed"
    # -- a checker inflating its own evidence.
    hits = sum(len(re.findall(r"(?<![\w.])" + re.escape(n) + r"\s*\(", src))
               for n in sorted(set(bound)))
    want = int(site.get("calls", 1))
    if site["covered"]:
        check("  and it calls admit exactly where it says", hits == want,
              "%d routed calls in %s(), %d claimed" % (hits, label, want))
    else:
        # An unrouted entry claims the opposite thing, so it is checked the
        # opposite way: no count to satisfy, but the reason has to be a
        # sentence somebody wrote. `why` is the whole value of this half of the
        # table -- a placeholder there turns "known_unrouted" into a place to
        # put things people stop reading, and then the gap is invisible again.
        why = str(site.get("why", ""))
        check("  and it says why it is not routed", len(why) > 60, (site["name"], len(why)))
        check("  and it makes no count claim it cannot keep",
              "calls" not in site and "func" not in site, sorted(site))
print()
print("-- the service binds the gateway to its own ledger at open")
tmp2 = tempfile.mkdtemp(prefix="m28_egress_svc_")
os.environ["AUTOFORGE_HOME"] = tmp2
svc = Service(db_path=os.path.join(tmp2, "s.db"))
try:
    from autoforge_svc import egress as eg2
    live = eg2.gateway()
    check("the process gateway is bound", live.conn is not None, live.conn)
    check("and it writes under the service's name", live.name == "autoforge", live.name)
    check("the service reports coverage instead of pretending",
          hasattr(svc, "report") and "egress" in svc.report(), list(svc.report())[:8])
finally:
    svc.store.close()
    svc.missions.close()
    eg.reset()

print()
print("TOTAL FAILURES:", len(FAILS), FAILS)
sys.exit(0 if not FAILS else 1)
