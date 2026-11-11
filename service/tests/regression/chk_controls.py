"""A refusal that leaves no row: the control plane was built and never bound.

Why this file exists
--------------------
`autonomy/controls.py` is 320 lines of careful reasoning about why every
judgement the machinery makes belongs on the chain, with a `bind(conn)` method
that says exactly what it is for:

    "Attach a ledger. Before this, receipts are counted but not written."

Measured 2026-11-03 against this package: `controls.bind` had **zero** call
sites. So every instance came up `ControlPlane(writing=False, conn=None)`, and
the module's own test of honesty -- "a control plane that cannot write says so,
on the object, and in `summary()`" -- was true of a plane that could never say
anything else. A refused call was counted in `_counts` (in memory) and the
ledger stayed empty.

That is the shape this project keeps catching: the mechanism works, the wire is
missing. Same as `anchor_ledger` (method written, dispatch table not) and the
same as `rehab` (persona promised a name nobody registered). So the test lives
at the door: not "does the plane work", but "is the plane attached to the
ledger this service actually writes to".

What is checked
---------------
1. A service comes up with a *writing* plane, pointed at its own database.
2. Two services in one process do not share a plane. A process-global plane
   would make the second service's refusals land in the first service's
   database -- which reads as success and is worse than an empty ledger.
3. A method refused at the wire gate leaves a row: `kind='policy'`,
   `event_type='gate_deny'`, `gate_outcome='no_operator'`, the method named as
   the resource, and the freedom it wanted in `freedoms`.
4. A *tool* refused at the registry gate leaves a row too -- both gates, one
   vocabulary. `registry._gate` and `Service._gate_method` refuse for the same
   reason and a reader should not have to learn two table names.
5. The receipt is not a second chain: the stitched ledger still verifies after
   these rows exist, and `audit.denials()` finds them.
6. Nothing is written when nothing was refused. A plane that logs its own
   existence on every startup would bury the refusals it exists to preserve.

What it deliberately does not check: that the row is *pretty*. `reason` and
`question` are prose for a person; asserting on prose makes a wording change
look like a regression.
"""
import json
import os
import sqlite3
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)

from autoforge_svc.autonomy.policy import SUPERVISED  # noqa: E402
from autoforge_svc.service import Service  # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print(f"  OK   {label}")
    else:
        print(f"  FAIL {label}  <- {detail}")
        FAILS.append(label)


def policy_rows(db: str) -> list[dict]:
    """Everything the control layer wrote, read straight from sqlite.

    Read from the file rather than through `audit._rows` on purpose: if the
    module's own reader were broken, a test routed through it would agree with
    the bug.
    """
    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            "SELECT id, kind, payload FROM forge_events WHERE kind='policy' ORDER BY id"
        ).fetchall()
    finally:
        con.close()
    out = []
    for rid, kind, payload in rows:
        try:
            out.append({"id": rid, "kind": kind, **json.loads(payload or "{}")})
        except ValueError:
            out.append({"id": rid, "kind": kind, "payload": payload})
    return out


tmp = tempfile.mkdtemp(prefix="m20_ctl_")
db_a = os.path.join(tmp, "a.db")
db_b = os.path.join(tmp, "b.db")
svc = Service(db_path=db_a)
svc2 = Service(db_path=db_b)

print("-- the plane is attached to this service's own ledger")
plane = svc.registry._plane
check("the service has a control plane", plane is not None, type(plane).__name__)
check("it is writing, not inert", plane.writing is True, getattr(plane, "summary", lambda: {})())
check("it points at this service's connection", plane.conn is not None, plane.conn)
check(
    "and not at some other process-wide one",
    plane is not svc2.registry._plane,
    "two services in one process shared a plane -- the second service's refusals "
    "would land in the first service's database",
)
check(
    "nothing was written just by starting up",
    policy_rows(db_a) == [],
    policy_rows(db_a),
)

print("-- a refused wire method leaves a row")
svc.registry.policy = SUPERVISED
denied = svc._call_method("notify_send", {"text": "x"})
check("the call was refused", isinstance(denied, dict) and denied.get("ok") is False, denied)
check("it says it is waiting for a human", denied.get("awaiting_confirmation") is True, denied)
rows = policy_rows(db_a)
check("a receipt exists", len(rows) == 1, rows)
row = rows[0] if rows else {}
check("it is a denial", row.get("event_type") == "gate_deny", row.get("event_type"))
check("the outcome is 'nobody to ask', not 'the operator said no'",
      row.get("gate_outcome") == "no_operator", row.get("gate_outcome"))
check("it names the method", row.get("resource") == "notify_send", row.get("resource"))
check("it names the freedom that was missing",
      row.get("freedoms") == ["may_access_network"], row.get("freedoms"))
check("it is marked not-allowed", row.get("allowed") is False, row.get("allowed"))
check("it carries the dimension a reader filters on",
      row.get("dimension") == "tool", row.get("dimension"))

print("-- a read-only method under the same policy runs and leaves nothing")
before = len(policy_rows(db_a))
allowed = svc._call_method("report")
check("the read-only call ran", isinstance(allowed, dict) and "tables" in allowed,
      list(allowed)[:6] if isinstance(allowed, dict) else allowed)
check("and wrote no receipt", len(policy_rows(db_a)) == before, len(policy_rows(db_a)))

print("-- a refused *tool* leaves a row in the same vocabulary")
svc.registry.policy = None
registered = []
# A forged tool with a scope the policy will object to. Registered directly so
# this test does not depend on the forge pipeline or on a sandbox.
from autoforge_svc.tools.spec import ToolSpec, ToolState  # noqa: E402

spec = ToolSpec(
    name="m20_net",
    description="a tool that claims the network",
    parameters={},
    fn=lambda **k: "ran",
    effect_signature="network",
    state=ToolState.ACTIVE,
)
svc.registry.register(spec, replace=False)
svc.registry.policy = SUPERVISED
result = svc.registry.call("m20_net", {})
check("the tool call was refused", result.ok is False, result.error)
check("and it never ran", result.output == "" and "Denied" in (result.error or ""), result.error)
rows2 = policy_rows(db_a)
check("a second receipt exists", len(rows2) == 2, len(rows2))
row2 = rows2[-1] if len(rows2) >= 2 else {}
check("same event type as the wire gate", row2.get("event_type") == "gate_deny",
      row2.get("event_type"))
check("same outcome", row2.get("gate_outcome") == "no_operator", row2.get("gate_outcome"))
check("it names the tool", row2.get("resource") == "m20_net", row2.get("resource"))

print("-- the receipts are on the chain, not beside it")
verified = svc.store.verify_chain()
check("the ledger still verifies", verified.get("ok") is True, verified.get("breaks"))
check("the rows are chained", verified.get("unchained_after_chaining_began") == 0,
      verified.get("unchained_after_chaining_began"))
from autoforge_svc import audit  # noqa: E402

denials = audit.denials(svc.store.connection())
check("the module's own reader finds them", len(denials) == 2, len(denials))
check("and it names the rule", all("autonomy.gate" in (d.get("rule") or "") for d in denials),
      [d.get("rule") for d in denials])
summary = plane.summary()
check("the plane reports what it recorded", summary.get("counts", {}).get("gate_deny") == 2,
      summary.get("counts"))
check("and no write failures", summary.get("record_failures") == 0, summary.get("failures"))

print("-- the second service's refusals stayed in its own file")
check("service B's ledger is untouched", policy_rows(db_b) == [], policy_rows(db_b))

svc.store.close()
svc.missions.close()
svc2.store.close()
svc2.missions.close()
print("TOTAL FAILURES:", len(FAILS), FAILS)
sys.exit(0 if not FAILS else 1)
