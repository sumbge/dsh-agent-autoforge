"""The one line-protocol method with no tool: `schedule_run_due`.

`schedule_tick` REPORTS what is due. `schedule_run_due` RUNS it. Only the first
one was on the bridge, so from a DSH turn the schedule could be read and could
not be fired -- the button that actually does the work was the one missing.

    lib/index.js methods: 67 registered, 4 absent:
      context_brief     taken directly by the assemble hook, never a tool
      report_bridge_tool internal bookkeeping
      tool_exposure     taken by the same hook
      schedule_run_due  <-- the only real gap

(`schedule_run_due` is reachable another way: `python -m autoforge_svc tick`,
which is what an OS scheduler calls. That path works and is why this went
unnoticed -- the method was tested, the button was not.)

What is worth asserting here is not "the name is in a dict". It is:
  * the tool exists on the bridge, with a parameter the model can use to bound
    a run it starts;
  * calling it with nothing due does NOT reach for a model, and says so;
  * calling it with something due actually runs the task and the ledger records
    it -- and the failure path is exercised without needing a gateway key, by
    pointing a task at a model route that cannot work;
  * the tool is gated like every other `system`-scope method.

Run:
  $env:PYTHONPATH = "...\\dsh-agent-autoforge\\service"
  & $py -P ...\\chk_schedule_due.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICE = os.path.dirname(os.path.dirname(HERE))   # regression -> tests -> service
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc.service import Service          # noqa: E402
from autoforge_svc.autonomy.policy import SUPERVISED  # noqa: E402

LIB = os.path.join(os.path.dirname(SERVICE), "lib", "index.js")

FAILS: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print("  OK   " + label)
    else:
        print("  FAIL " + label + (" <- " + str(detail)[:300] if detail else ""))
        FAILS.append(label)


def bridge_blocks() -> list[dict[str, str]]:
    """Every `register({...})` block in the shipped bridge, as name->method."""
    src = open(LIB, encoding="utf-8").read()
    out = []
    for hit in re.finditer(r"register\(\{(.*?)\n  \}\)", src, re.S):
        body = hit.group(1)
        name = re.search(r"toolName:\s*'([^']+)'", body)
        method = re.search(r"method:\s*'([^']+)'", body)
        if name and method:
            out.append({"tool": name.group(1), "method": method.group(1),
                        "body": body})
    return out


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m24_due_")
    db = os.path.join(tmp, "autoforge.db")
    # The schedule lives at $AUTOFORGE_HOME/schedule.jsonl, not beside the db,
    # so point it at this temp dir rather than at the operator's real agenda:
    # a test that fires somebody's real scheduled task is not a test.
    os.environ["AUTOFORGE_HOME"] = tmp
    svc = Service(db_path=db)

    print("-- the button exists, and it says what it bounds")
    blocks = bridge_blocks()
    due = [b for b in blocks if b["method"] == "schedule_run_due"]
    check("the bridge registers a tool for schedule_run_due", len(due) == 1,
          [b["tool"] for b in blocks if "due" in b["method"]])
    if due:
        body = due[0]["body"]
        check("its tool name is namespaced like the rest",
              due[0]["tool"].startswith("autoforge_"), due[0]["tool"])
        check("it takes a budget, so a model-started run cannot hang forever",
              "budget_s" in body, body[:200])
        check("its description keeps the tick/run_due distinction",
              "report" in body.lower() and ("tick" in body.lower()), body[:200])

    print("-- nothing due is not a model call, and it says so")
    before = svc.my_history(kind="schedule_run_due")["count"]
    res = svc._call_method("schedule_run_due")
    check("it answers", isinstance(res, dict), type(res).__name__)
    check("it reports zero due", res.get("due") == 0, res)
    check("and it names the reason rather than reporting silence",
          "nothing was due" in str(res.get("note", "")), res.get("note"))
    check("no ledger row is written when nothing happened",
          svc.my_history(kind="schedule_run_due")["count"] == before,
          svc.my_history(kind="schedule_run_due")["count"])

    print("-- something due runs, fails loudly, and lands on the ledger")
    added = svc.schedule_add(text="say something", when="+1s")
    check("a task was scheduled", "id" in added, added)
    import time
    time.sleep(1.2)
    # The failure is INJECTED, not inherited from a broken default.
    #
    # This block used to say "no gateway key in this environment, so the agent
    # cannot be built: that is the failure path, exercised for free". That was
    # wrong twice over. The host has a key, so the run got as far as the model
    # lookup; and once the model id was fixed the run would succeed, turning
    # this suite red for the one reason a test must never go red -- the product
    # started working. A test that leans on a defect is a test that fails when
    # the defect is repaired.
    #
    # `_forge_route` is a MODULE-level function called by `_run_task`, so
    # patching the instance attribute (as `chk_run_journal.py` still did) has no
    # effect at all. Patching the module attribute is the only version that
    # reaches the code under test.
    import autoforge_svc.service as _svc_mod
    import autoforge_svc.core.llm as _llm_mod

    real_route = _svc_mod._forge_route
    _svc_mod._forge_route = lambda: ("no-such-model-anywhere", "http://127.0.0.1:9/v1", "key")
    try:
        ran = svc._call_method("schedule_run_due", {"budget_s": 5})
    finally:
        _svc_mod._forge_route = real_route
    check("it ran something", ran.get("due") == 1, ran)
    check("and reports the failure rather than a success",
          ran.get("failed") == 1 and ran.get("ran") == 0, ran)
    # First draft of this assertion looked for the words "credential"/"api"/
    # "key", on the theory that a missing gateway key would be the failure. It
    # was not: this host HAS a key, and the run failed later, at the model
    # lookup (`LLMError: ... HTTP 404 ... model ... could not be resolved`).
    # Asserting a vocabulary I guessed is how a test passes on the day the
    # failure moves. Assert the shape instead: a non-empty note that names an
    # exception rather than the bare word "failed".
    notes = [str(r.get("note", "")) for r in ran.get("results", [])]
    check("the task carries a reason, not just a verdict",
          bool(notes) and all(len(n) > 20 for n in notes), notes)
    check("and the reason names the error it hit",
          any(":" in n and any(c.isalpha() for c in n.split(":")[0]) for n in notes),
          notes)
    after = svc.my_history(kind="schedule_run_due")["count"]
    check("the run is on the ledger", after == before + 1, after)

    print("-- and it is gated like every other system-scope method")
    check("its scope is system, not read_only",
          __import__("autoforge_svc.autonomy.scopes", fromlist=["scope_for"])
          .scope_for("schedule_run_due") == "system",
          __import__("autoforge_svc.autonomy.scopes", fromlist=["scope_for"])
          .scope_for("schedule_run_due"))
    row = svc.store.connection().execute(
        "SELECT COUNT(*) FROM forge_events WHERE kind='policy'").fetchone()[0]
    check("no refusal was recorded while the policy was full freedom", row == 0, row)

    svc.store.close()
    svc.missions.close()
    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
