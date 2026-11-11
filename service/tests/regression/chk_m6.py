"""Behaviour test for the m6/persona-alignment changes, on a throwaway database.

Ninth round. The persona claimed four things that the code did not do:
`describe_tool` existed, `force=True` reached the registry, `rehab` could lift
a quarantine, and `evolve_tool` was the only overwrite path. Each assertion
below is one of those claims being either true or false.

Order matters: this walks the paths a real caller meets them in.

  1. define a tool                                   -> registered, probation
  2. define the same name again, no replace          -> REFUSED, body untouched
  3. define the same name with replace=True          -> REFUSED TOO: no such path
  4. describe_tool(name)                             -> schema + record + versions
  5. describe_tool(unknown)                          -> ok=False, names the library
  6. tool_call on a quarantined tool                 -> refused
  7. tool_call(force=True) on the same tool          -> runs anyway, still quarantined
  8. rehab_tool                                      -> back to probation
  9. evaluate_tool on a quarantined tool that passes -> rehab branch, not promote
 10. evaluate_tool on a quarantined tool that fails  -> stays quarantined
 11. evaluate_tool(promote=False)                    -> reports, moves nothing

Run it with the real interpreter, not the store stub:

    $env:PYTHONPATH = '<plugin>\\service'
    & 'C:\\Users\\sumbger\\AppData\\Local\\Programs\\Python\\Python313\\python.exe' chk_m6.py

Expected: TOTAL FAILURES: 0
"""
import os
import sys
import tempfile
import types

# service/tests/regression/ -> the plugin root is three levels up (regression ->
# tests -> service -> plugin). It used to say four, which pointed one directory
# above the plugin and made this file die on `import autoforge_svc` the moment
# PYTHONPATH was not exported. The env var still wins, because pointing this at
# a broken copy is how the mutation run proves the assertions can fail at all.
_HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.environ.get(
    "AUTOFORGE_PLUGIN",
    os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, os.path.join(PLUGIN, "service"))

from autoforge_svc.service import Service          # noqa: E402
from autoforge_svc.tools.spec import ToolState     # noqa: E402

bad = 0


def check(label, ok, detail=""):
    global bad
    if not ok:
        bad += 1
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"   | {detail}" if detail else ""))


BODY_A = "def probe_tool(**kw):\n    return {'v': 1}\n"
BODY_B = "def probe_tool(**kw):\n    return {'v': 2}\n"

# A throwaway database. Never the real one: this test defines, quarantines and
# redefines a tool called probe_tool, and doing that to the live library would
# leave debris in a ledger that is supposed to be append-only evidence.
tmp = tempfile.mkdtemp(prefix="chk_m6_")
db = os.path.join(tmp, "scratch.db")
svc = Service(db)
print("db:", db)

# 1 -- define
r1 = svc.tool_define(
    "probe_tool", BODY_A, description="returns a constant",
    parameters={"type": "object", "properties": {}},
    sample_call={}, sample_expect="'v': 1")
check("1 define accepted", r1.get("ok") and r1.get("registered"),
      f"state={r1.get('state')}")
check("1 state is probation", r1.get("state") == "probation", r1.get("state"))

# 2 -- redefine without replace must be refused, and must not overwrite
r2 = svc.tool_define(
    "probe_tool", BODY_B, description="a different body",
    parameters={"type": "object", "properties": {}},
    sample_call={}, sample_expect="'v': 2")
check("2 redefine refused", r2.get("ok") is False and r2.get("registered") is False,
      f"ok={r2.get('ok')} registered={r2.get('registered')}")
# The refusal has to name the one path that *does* overwrite. When the message
# said "pass replace=true", that sentence was the documentation for a second
# overwrite path, and the library's own rule stopped being true.
check("2 refusal names evolve_tool", "evolve_tool" in str(r2.get("error", "")),
      str(r2.get("error", ""))[:90])
check("2 refusal does not offer a taking flag",
      "pass replace=true" not in str(r2.get("error", "")).lower(),
      str(r2.get("error", ""))[:90])
check("2 refusal names the state", "probation" in str(r2.get("error", "")))
spec_now = svc.registry.get("probe_tool")
check("2 body untouched", spec_now.code == BODY_A,
      "v1" if spec_now.code == BODY_A else "OVERWRITTEN")
check("2 description untouched", spec_now.description == "returns a constant",
      spec_now.description)
# The output is JSON, so the key is double-quoted. Writing `'v': 1` here -- the
# Python repr -- makes this assertion fail against a perfectly good tool. That
# happened once; it cost two bogus FAILs before the third one turned out real.
c0 = svc.tool_call("probe_tool", {})
check("2 original still runs", str(c0.get("output", "")).find('"v": 1') >= 0,
      str(c0.get("output", ""))[:60])

# 3 -- replace=True is REFUSED as well. This is the assertion that keeps the
# rule true rather than merely documented: a flag that works is a second
# overwrite path no matter what the persona says. `replace` survives only so a
# caller who believes overwriting is available is told which call actually does
# it, instead of silently getting the old body or a bare "already exists".
r3 = svc.tool_define(
    "probe_tool", BODY_B, description="a different body",
    parameters={"type": "object", "properties": {}},
    sample_call={}, sample_expect="'v': 2", replace=True)
check("3 replace is refused too",
      r3.get("ok") is False and r3.get("registered") is False,
      f"ok={r3.get('ok')} registered={r3.get('registered')}")
check("3 refusal is the replace-specific one", r3.get("replace_refused") is True,
      str(r3.get("replace_refused")))
check("3 refusal says no value of it changes the answer",
      "no value of it changes the answer" in str(r3.get("error", "")),
      str(r3.get("error", ""))[:110])
spec_now = svc.registry.get("probe_tool")
check("3 body still the original", spec_now.code == BODY_A,
      "v1" if spec_now.code == BODY_A else "OVERWRITTEN")
c1 = svc.tool_call("probe_tool", {})
check("3 still runs the original", str(c1.get("output", "")).find('"v": 1') >= 0,
      str(c1.get("output", ""))[:60])

# 4 -- describe_tool
d = svc.describe_tool("probe_tool")
check("4 describe ok", d.get("ok") is True)
check("4 has schema", isinstance(d.get("schema"), dict)
      and d["schema"].get("function", {}).get("name") == "probe_tool")
check("4 has record with state", d.get("state") == "probation", str(d.get("state")))
check("4 has stats", isinstance(d.get("stats"), dict),
      str(list((d.get("stats") or {}).keys()))[:80])
# One body, because two refusals wrote nothing. A version count that grew here
# would mean a refused define had still recorded something.
check("4 versions stayed at one", len(d.get("versions") or []) == 1,
      f"{len(d.get('versions') or [])} versions")
check("4 has ledger_calls", isinstance(d.get("ledger_calls"), dict)
      and d["ledger_calls"].get("calls", 0) >= 2,
      str(d.get("ledger_calls")))
# The point of giving both numbers is that a disagreement is visible. The two
# calls above are both on the same body, so they must agree exactly -- an
# in-memory counter that outruns the ledger is a counter nobody can audit.
check("4 memory stats and ledger agree on the call count",
      (d.get("stats") or {}).get("calls") == (d.get("ledger_calls") or {}).get("calls"),
      f"stats={((d.get('stats') or {}).get('calls'))} "
      f"ledger={((d.get('ledger_calls') or {}).get('calls'))}")

# 5 -- unknown name
d2 = svc.describe_tool("no_such_tool")
check("5 unknown refused", d2.get("ok") is False)
check("5 unknown lists the library", "probe_tool" in str(d2.get("error", "")),
      str(d2.get("error", ""))[:90])

# 6/7 -- quarantine, then force
svc.registry.quarantine("probe_tool", "test: pretending it misbehaved")
check("6 state is quarantined",
      svc.registry.get("probe_tool").state is ToolState.QUARANTINED)
c2 = svc.tool_call("probe_tool", {})
check("6 quarantined call refused", c2.get("ok") is False,
      str(c2.get("error", ""))[:90])
check("6 refusal names force", "force=True" in str(c2.get("error", "")))
c3 = svc.tool_call("probe_tool", {}, force=True)
check("7 force runs it", c3.get("ok") is True, str(c3.get("output", ""))[:60])
check("7 force did not lift the quarantine",
      svc.registry.get("probe_tool").state is ToolState.QUARANTINED,
      svc.registry.get("probe_tool").state.value)

# 8 -- manual rehab
r8 = svc.rehab_tool("probe_tool")
check("8 rehab ok", r8.get("ok") is True, str(r8.get("text", ""))[:80])
check("8 back to probation",
      svc.registry.get("probe_tool").state is ToolState.PROBATION)
check("8 counters cleared", svc.registry.get("probe_tool").stats.calls == 0,
      str(svc.registry.get("probe_tool").stats.calls))
r8b = svc.rehab_tool("no_such_tool")
check("8 rehab of unknown refused", r8b.get("ok") is False)

# 9 -- the automatic path: a passing battery on a QUARANTINED tool rehabs it
svc.registry.quarantine("probe_tool", "test: quarantine again")
before = svc.registry.get("probe_tool").state
moved = svc._settle_state_after_evaluation(
    svc.registry.get("probe_tool"),
    types.SimpleNamespace(passed=True), True)
after = svc.registry.get("probe_tool").state
check("9 passing battery moves quarantined", before is ToolState.QUARANTINED
      and after is ToolState.PROBATION,
      f"{before.value} -> {after.value}")
check("9 move is named rehab", (moved or {}).get("from") == "quarantined"
      and (moved or {}).get("to") == "probation", str(moved)[:100])
# A quarantine was earned by real call failures. One clean battery does not
# erase that; it only puts the tool back on trial.
check("9 not promoted straight to active", after is not ToolState.ACTIVE)

# 10 -- a failing battery must leave a quarantine alone
svc.registry.quarantine("probe_tool", "test: third time")
moved2 = svc._settle_state_after_evaluation(
    svc.registry.get("probe_tool"),
    types.SimpleNamespace(passed=False), True)
check("10 failing battery leaves it quarantined",
      svc.registry.get("probe_tool").state is ToolState.QUARANTINED,
      str(moved2))
check("10 no move reported", moved2 is None, str(moved2))

# 11 -- promote=False still only reports
svc.registry.quarantine("probe_tool", "test: fourth")
moved3 = svc._settle_state_after_evaluation(
    svc.registry.get("probe_tool"),
    types.SimpleNamespace(passed=True), False)
check("11 promote=False does not move",
      moved3 is None and svc.registry.get("probe_tool").state is ToolState.QUARANTINED)

print()
print(f"TOTAL FAILURES: {bad}")

# Take the scratch database with us. A test that leaves a database behind on
# every run turns into a slow leak, and an unexplained scratch.db in %TEMP% is
# exactly the kind of thing a later session mistakes for evidence.
for conn_owner in ("missions", "store"):
    owner = getattr(svc, conn_owner, None)
    conn = getattr(owner, "_conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:                                  # noqa: BLE001
            pass
import shutil
shutil.rmtree(tmp, ignore_errors=True)
print("cleaned up:", not os.path.exists(tmp))
sys.exit(1 if bad else 0)
