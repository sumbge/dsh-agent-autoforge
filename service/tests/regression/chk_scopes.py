"""P0-1: a tool that does not say what it touches needs every freedom.

The gate in `tools/registry.py` asks `autonomy/confirm.py` what a tool's
`effect_signature` requires. An empty signature is read as `undeclared`, and
`SCOPE_ALLOWANCES["undeclared"]` is all eight capabilities -- so under any
policy that switches off a single freedom, *every* unlabelled tool is treated
as able to do everything, and every call is gated.

Measured 2026-11-03 on the real ledger: all six forged tools carried
`effect_signature = ''`. Not one of them had ever been asked what it touches,
because nothing in this package ever filled the field in.

Two halves, and they are not the same half:

  * Forged tools DO reach `registry.call`, which is the only place the gate
    runs. Their scope has to be filled at define time. Nothing can be read off
    them, so it is derived from the code and recorded -- `derived_scope`.
  * The 71 wire methods never enter the registry at all (verified: on the real
    database `svc.registry.names()` lists the six forged tools and no
    builtin). A table of their scopes would be decoration unless the dispatch
    loop consults it -- so the table and the call site are one change, and
    this file refuses to accept either one alone.

The upstream table (`agent.py:587 BUILTIN_SCOPES`, 79 rows) cannot be copied
verbatim for a second reason: 25 of its names do not exist here, and 13 names
here do not exist there. The table has to be this service's own.
"""
from __future__ import annotations

import os
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)

from autoforge_svc.service import Service                      # noqa: E402
from autoforge_svc.autonomy.policy import SUPERVISED, FULL_FREEDOM   # noqa: E402
from autoforge_svc.autonomy.confirm import disabled_freedoms   # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print(f"  OK   {label}")
    else:
        print(f"  FAIL {label} <- {detail}")
        FAILS.append(label)


tmp = tempfile.mkdtemp(prefix="m19_scopes_")
svc = Service(db_path=os.path.join(tmp, "scopes.db"))

print("-- the vocabulary the table has to speak")
LEGAL = {"pure", "read_only", "local_write", "destructive", "network",
         "system", "privileged"}
from autoforge_svc.forge.validity import SCOPE_ALLOWANCES
check("the scope table covers every name the gate knows",
      LEGAL <= set(SCOPE_ALLOWANCES) - {"undeclared"},
      sorted(LEGAL - set(SCOPE_ALLOWANCES)))

print()
print("-- half one: every wire method says what it touches")
from autoforge_svc.autonomy.scopes import METHOD_SCOPES, scope_for
methods = sorted(svc.methods)
check("the table is not empty", len(METHOD_SCOPES) >= 60, len(METHOD_SCOPES))
missing = [m for m in methods if m not in METHOD_SCOPES]
check("every wire method has a declared scope", not missing, missing)
stale = sorted(set(METHOD_SCOPES) - set(methods))
check("no entry names a method this service does not have", not stale, stale)
bad = sorted({v for v in METHOD_SCOPES.values()} - LEGAL)
check("every declared scope is a legal one", not bad, bad)
check("unknown names declare nothing rather than guessing", scope_for("no_such") == "",
      scope_for("no_such"))

print()
print("-- half two: the dispatch loop asks before it runs")
read_only_free = disabled_freedoms(METHOD_SCOPES["describe"], SUPERVISED)
check("a read-only method needs nothing under SUPERVISED", read_only_free == [],
      read_only_free)
sys_free = disabled_freedoms(METHOD_SCOPES["run_python"], SUPERVISED)
check("an executing method does need a freedom under SUPERVISED", bool(sys_free), sys_free)
check("and under full freedom nothing is gated",
      disabled_freedoms(METHOD_SCOPES["run_python"], FULL_FREEDOM) == [])

from autoforge_svc.autonomy.confirm import describe as _describe
check("the question names the switch, the tool and the arguments",
      "may_access_network" in _describe(["may_access_network"], "t", {"a": 1}))

print()
print("-- half three: a forged tool derives a scope from its own code")
from autoforge_svc.autonomy.scopes import derived_scope
# The floor is `read_only`, not `pure`, and that is a decision rather than an
# oversight. `audit_effects` can see a write, a subprocess, the network, `eval`
# and unpickling; it cannot see a read, because `open(p)` with no mode looks
# like nothing at all to a regex. Answering `pure` for code the audit could not
# label would be claiming innocence on the strength of not having looked --
# and a `pure` tool under a policy that forbids reading would run with a
# freedom nobody granted it. `read_only` costs nothing when reading is allowed
# and is gated when it is not, so erring narrow is fail-closed in both
# directions. Erring too narrow is caught on the other side by
# `validity._scope_gate`, which fails a tool whose code exceeds its declared
# scope.
cases = {
    'def main(**k):\n    return 1\n': "read_only",
    'def main(**k):\n    return open("x").read()\n': "read_only",
    'def main(**k):\n    return open("x").read() and 1\n': "read_only",
    'def main(**k):\n    open("x","w").write("1")\n    return 1\n': "local_write",
    'def main(**k):\n    import subprocess\n    return subprocess.run(["x"])\n': "system",
}
for body, want in cases.items():
    got = derived_scope(body)
    check(f"code that amounts to {want!r} derives {want!r}", got == want, got)

check("the derivation never answers `undeclared` for readable code",
      derived_scope("def main(**k):\n    return 1\n") in LEGAL)
check("nor `pure`, which would be a claim the audit cannot support",
      derived_scope("def main(**k):\n    return 1\n") != "pure")
check("empty code derives the floor rather than everything", derived_scope("") == "read_only",
      derived_scope(""))

# Added 2026-11-05 with the `destructive` scope. Before it, `derived_scope`
# fell through every entry in `_BY_SIZE` for code whose only visible effect is
# a deletion and answered `undeclared` -- all eight capabilities, for a tool
# that removes a file. That is the hole the translator's known-limit note used
# to describe; this is the assertion that closes it.
deleting = {
    'def main(**k):\n    import os\n    os.remove(k["p"])\n    return 1\n': "destructive",
    'def main(**k):\n    from pathlib import Path\n    Path(k["p"]).unlink()\n    return 1\n': "destructive",
    'def main(**k):\n    import shutil\n    shutil.rmtree(k["p"])\n    return 1\n': "destructive",
    'def main(**k):\n    import os\n    open(k["p"],"w").write("1")\n    os.remove(k["p"])\n    return 1\n': "privileged",
}
for body, want in deleting.items():
    got = derived_scope(body)
    check(f"code that deletes derives {want!r}", got == want, got)
from autoforge_svc.forge.validity import SCOPE_CONTEXT_CEILING
check("and a destructive tool carries the strictest reuse ceiling",
      SCOPE_CONTEXT_CEILING.get("destructive") == 0,
      SCOPE_CONTEXT_CEILING.get("destructive"))

# Every scope except `pure` can read, and it says so in two tables that have to
# agree: `SCOPE_ALLOWANCES` lists the labels the audit can see, `_SCOPE_EXTRA`
# adds the one label it cannot. A scope present in the first and missing from
# the second loses its reading capability silently -- and nothing else in this
# file would notice, because reading is free either way. That is exactly how
# `destructive` could have shipped half-added.
from autoforge_svc.autonomy.confirm import declared_capabilities
unreadable = sorted(s for s in LEGAL if s != "pure"
                    and "filesystem_read" not in declared_capabilities(s))
check("every scope that may read says so in both tables", not unreadable, unreadable)
check("and `destructive` carries exactly delete plus read, nothing else",
      declared_capabilities("destructive") == frozenset(
          {"filesystem_delete", "filesystem_read"}),
      sorted(declared_capabilities("destructive")))

print()
print("-- a reading tool is not labelled a writer (the regex this nearly shipped)")
# `open("x")` was a write under the old pattern: `[^)]*['"][waxr]...` matched
# the PATH when it was a one-character string, and `x` is a legal mode. That
# made `read_only` impossible to declare for any tool that opened a file.
from autoforge_svc.forge.validity import audit_effects
reads = [
    'def main(**k):\n    return open("x").read()\n',
    'def main(**k):\n    return open(p).read()\n',
    'def main(**k):\n    return open(p, "rt").read()\n',
    'def main(**k):\n    return open(p, "rb").read()\n',
]
writes = [
    'def main(**k):\n    open("x","w").write("1")\n',
    'def main(**k):\n    open(p, mode="w")\n',
    'def main(**k):\n    open(p, "r+")\n',
    'def main(**k):\n    open(p, "a")\n',
    'def main(**k):\n    open(p, "x")\n',
]
for body in reads:
    labels = [f.label for f in audit_effects(body)]
    check(f"a read is not called a write: {body.splitlines()[-1].strip()[:40]!r}",
          "filesystem_write" not in labels, labels)
for body in writes:
    labels = [f.label for f in audit_effects(body)]
    check(f"a write is still called a write: {body.splitlines()[-1].strip()[:40]!r}",
          "filesystem_write" in labels, labels)

print()
print("-- and the gate is on the only path a request can take")
svc2 = Service(db_path=os.path.join(tmp, "gate.db"))
seen = {"ran": 0}
svc2.methods["m19_probe"] = lambda: seen.__setitem__("ran", seen["ran"] + 1) or "ran"
from autoforge_svc.autonomy.scopes import METHOD_SCOPES as _MS
_MS["m19_probe"] = "network"

svc2.registry.policy = None
out = svc2._call_method("m19_probe")
check("with no policy the method runs", out == "ran" and seen["ran"] == 1, out)

from autoforge_svc.autonomy.policy import SUPERVISED as _SUP
svc2.registry.policy = _SUP
out = svc2._call_method("m19_probe")
check("under SUPERVISED a network method is refused", isinstance(out, dict)
      and out.get("ok") is False, out)
check("and it says what it needed", isinstance(out, dict)
      and out.get("needed") == ["may_access_network"], (out or {}).get("needed"))
check("and it did not run", seen["ran"] == 1, seen)

out = svc2._call_method("describe")
check("a read-only method still runs under the same policy",
      isinstance(out, dict) and "methods" in out, list(out)[:6] if isinstance(out, dict) else out)
check("an unknown method is not invented", svc2._call_method("no_such_method") is None)
try:
    svc2.store.close()
    svc2.missions.close()
except Exception:
    pass

print()
print("-- and the field actually gets filled at define time")
spec_src = 'def main(**k):\n    return open("y").read()\n'
res = svc.tool_define(name="m19_reads", code=spec_src, description="reads a file")
got = svc.registry.get("m19_reads")
check("a defined tool is registered", got is not None)
check("and it carries a derived scope, not an empty string",
      got is not None and got.effect_signature != "",
      got.effect_signature if got else None)
check("the derived scope matches what the code does",
      got is not None and got.effect_signature == "read_only",
      got.effect_signature if got else None)
check("the define call reported the scope it settled on",
      isinstance(res, dict) and res.get("effect_signature") == "read_only",
      (res or {}).get("effect_signature"))

svc.store.close()
svc.missions.close()
print()
print("TOTAL FAILURES:", len(FAILS), FAILS)
raise SystemExit(0 if not FAILS else 1)
