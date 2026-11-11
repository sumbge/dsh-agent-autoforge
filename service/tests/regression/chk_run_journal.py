"""A scheduled run that did not finish must not be filed as one that did.

Two defects met in the same place, and this file pins both.

**Defect A -- the turn cap was recorded as success.** Measured 2026-11-04,
before the fix:

    _run_task   ->  return getattr(result, "content", "") or ""
    caller      ->  ok = True            # whenever nothing raised
    result row  ->  {'ok': True, 'note': '(turn cap of 12 reached before
                     completion)'}
    task on disk -> runs 1 / failures 0

The apology existed, in a field nothing read. `AgentResult` already carried
`turns`, `tool_calls`, `self_terminated` and `termination_reason` so a caller
could tell a finished run from a truncated one; the caller passed the result
object through `content` and dropped the rest. `failures` is the only counter
that makes "this task is too big for one run" distinguishable from "this task
is quietly working", and it stayed at zero for both.

**Defect B -- `runs.py` had no writers.** 708 lines of resume machinery, three
`Agent(...)` constructions in the package, none passing `journal`. Measured the
same day: `%LOCALAPPDATA%\\autoforge\\runs` exists and holds nothing, and the
ledger has 0 rows for `resume`, `runs_status` and `runs_forget`. Upstream built
the `Journal` in `cli.py`; the port to `service.py` lost that line. So
`runs_status` answered "nothing journalled" forever and `resume_run` had
nothing to resume -- not because nothing was interruptible, but because nothing
was ever written down.

What is asserted here, in that order:
  * running out of turns is filed as a failure, with the reason on the task;
  * the fields survive the trip from `AgentResult` to the tick's result row;
  * a clean ending deletes its record, and a raising ending leaves it standing
    -- the second one is the whole point of writing it;
  * the record on disk parses and carries the task text, so a later process
    could actually act on it.

Run:
  $env:PYTHONPATH = "...\\dsh-agent-autoforge\\service"
  & $py -P ...\\chk_run_journal.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICE = os.path.dirname(os.path.dirname(HERE))   # regression -> tests -> service
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc import runs as _runs                      # noqa: E402
from autoforge_svc.core import llm as _llm                  # noqa: E402
from autoforge_svc.core.llm import LLMError, LLMResponse, MockLLMClient  # noqa: E402
from autoforge_svc.core.message import ToolCall              # noqa: E402
from autoforge_svc.service import Service, TaskOutcome       # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print("  OK   " + label)
    else:
        print("  FAIL " + label + (" <- " + str(detail)[:300] if detail else ""))
        FAILS.append(label)


class SpyJournal(_runs.Journal):
    """A journal that says whether it was used, and whether it cleaned up.

    Subclassed rather than reimplemented: this has to exercise the real
    `beat`/`finish`, and a stand-in would only prove that a stand-in works.
    """

    made: list["SpyJournal"] = []

    def __init__(self, task: str, **kw: object) -> None:
        super().__init__(task, **kw)                 # type: ignore[arg-type]
        self.beats = 0
        self.finished = False
        SpyJournal.made.append(self)

    def beat(self, msgs: object, **kw: object) -> None:
        self.beats += 1
        return super().beat(msgs, **kw)              # type: ignore[arg-type]

    def finish(self) -> None:
        self.finished = True
        return super().finish()


def always_calls_a_tool(_msgs: object, _tools: object = None, **_kw: object) -> LLMResponse:
    """One tool call per turn, forever -- the shape that reaches the cap."""
    return LLMResponse(content="thinking", tool_calls=[
        ToolCall(id="c1", name="no_such_tool", arguments={}),
    ])


def blows_up(_msgs: object, _tools: object = None, **_kw: object) -> LLMResponse:
    raise LLMError("provider said no")


def wire(svc: Service, handler: object) -> None:
    """Point `_run_task` at a mock client without touching the real one.

    `_forge_route` is a MODULE-level function and `_run_task` calls it by that
    name, resolved against the module's globals at call time. So
    `svc._forge_route = ...` -- which is what this line used to do -- set an
    attribute nothing ever reads, and the test passed only because the real
    route happened to carry a working key. It would have gone red for the
    wrong reason the day the key moved.
    """
    _llm.OpenAICompatClient = lambda **kw: MockLLMClient(handler=handler)  # type: ignore[assignment]
    import autoforge_svc.service as _svc_mod
    _svc_mod._forge_route = lambda: ("mock", "http://unused", "key")  # type: ignore[assignment]


def task_on_disk(svc: Service, task_id: str) -> dict:
    """The task as the scheduler reads it back, as a plain dict.

    Read through `Task`'s attributes rather than a `to_dict()`: `Task` has no
    such method, and my first draft called `task.to_dict() if hasattr(...)`
    and fell through to `{}` -- so three assertions compared `None` to `1` and
    reported the product as broken. A helper that answers `{}` for a task that
    exists is a helper that turns a missing accessor into a false accusation.
    """
    task = svc._scheduler().get(task_id)
    if task is None:
        return {}
    return {
        "id": task.id,
        "runs": task.runs,
        "failures": task.failures,
        "cancelled": task.cancelled,
        "history": list(task.history),
    }


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m25_journal_")
    # $AUTOFORGE_HOME decides both the db and the journal directory, so this
    # also keeps the operator's own runs/ untouched.
    os.environ["AUTOFORGE_HOME"] = tmp
    svc = Service(db_path=os.path.join(tmp, "autoforge.db"))
    wire(svc, always_calls_a_tool)
    _runs.Journal = SpyJournal                         # type: ignore[misc]
    SpyJournal.made.clear()

    print("-- a run that spends its turns is not a run that finished")
    added = svc.schedule_add(text="loop forever", when="+0s")
    tid = added["id"]
    ran = svc._call_method("schedule_run_due", {"budget_s": 60})
    check("the task ran", ran.get("due") == 1, ran)
    check("and it is counted as a failure", ran.get("failed") == 1, ran)
    check("not as a success", ran.get("ran") == 0, ran)

    rows = ran.get("results") or [{}]
    row = rows[0]
    check("the row says so in a field, not only in a sentence",
          row.get("hit_turn_cap") is True, row)
    check("and its ok is False", row.get("ok") is False, row.get("ok"))
    check("it names the bounding reason", "cap" in str(row.get("why", "")),
          row.get("why"))
    check("it reports how many turns it used",
          row.get("turns") == 12, row.get("turns"))
    # Exact, not `"cap" in note`. First draft asserted the word "cap" appears,
    # and the mutant `note = (outcome.content or "done")` escaped it: the capped
    # run's *content* is `"(turn cap of 12 reached before completion)"`, so the
    # word is present in both readings and the assertion could not tell the
    # reason from the apology. The contract is that the note is the reason.
    check("the note is the reason, not whatever the run happened to say",
          str(row.get("note", "")) == str(row.get("why", ""))[:300],
          {"note": row.get("note"), "why": row.get("why")})

    disk = task_on_disk(svc, tid)
    check("the task file agrees", disk.get("failures") == 1, disk.get("failures"))
    check("its counters do not claim a success",
          disk.get("runs") == 1 and disk.get("failures") == 1,
          {"runs": disk.get("runs"), "failures": disk.get("failures")})
    history = disk.get("history") or []
    check("the history row is a failure",
          bool(history) and history[-1].get("ok") is False, history[-1:] if history else [])

    print("-- the journal was used, and it cleaned up after itself")
    check("a record was opened for the run", len(SpyJournal.made) == 1,
          len(SpyJournal.made))
    if SpyJournal.made:
        spy = SpyJournal.made[0]
        check("the loop beat at least once", spy.beats > 0, spy.beats)
        check("a run that ended was finished", spy.finished is True, spy.finished)
        left = list(_runs.journal_dir().glob("run-*.json"))
        check("and its file is gone from disk", left == [], [p.name for p in left])

    print("-- a run that raised leaves the record standing")
    SpyJournal.made.clear()
    wire(svc, blows_up)
    added2 = svc.schedule_add(text="explode", when="+0s")
    tid2 = added2["id"]
    ran2 = svc._call_method("schedule_run_due", {"budget_s": 60})
    check("it is reported as a failure", ran2.get("failed") == 1, ran2)
    check("a second record was opened", len(SpyJournal.made) == 1,
          len(SpyJournal.made))
    if SpyJournal.made:
        check("but it was never finished",
              SpyJournal.made[0].finished is False, SpyJournal.made[0].finished)
    left = list(_runs.journal_dir().glob("run-*.json"))
    check("the record is still on disk -- that is the only evidence it started",
          len(left) == 1, [p.name for p in left])
    if left:
        try:
            data = json.loads(left[0].read_text(encoding="utf-8"))
        except Exception as exc:                       # noqa: BLE001
            data = {}
            check("the record parses", False, f"{type(exc).__name__}: {exc}")
        check("it carries the task text a later process would need",
              data.get("task") == "explode", data.get("task"))
        check("and it names the schema it was written by",
              isinstance(data.get("schema"), int), data.get("schema"))
    disk2 = task_on_disk(svc, tid2)
    check("the raising run is a failure on the task too",
          disk2.get("failures") == 1, disk2.get("failures"))

    print("-- the folding rule itself, away from the loop")
    ended = TaskOutcome.from_result(SimpleNamespace(
        content="done", turns=3, tool_calls=["a"], self_terminated=True,
        termination_reason="finished"))
    check("an agent that decided it was done is a success", ended.ok is True,
          ended.to_dict())
    check("and that is not the same claim as hitting the cap",
          ended.hit_turn_cap is False, ended.to_dict())
    capped = TaskOutcome.from_result(SimpleNamespace(
        content="", turns=12, tool_calls=[], hit_turn_cap=True,
        termination_reason="turn cap of 12 reached"))
    check("a capped run is never ok", capped.ok is False, capped.to_dict())
    stopped = TaskOutcome.from_result(SimpleNamespace(
        content="", turns=4, stopped_by_operator=True))
    check("a run a person stopped is not ok either", stopped.ok is False,
          stopped.to_dict())
    bare = TaskOutcome.from_result(SimpleNamespace())
    check("a result object with nothing in it reads as finished", bare.ok is True,
          bare.to_dict())
    check("and asking for the dict twice does not mutate the outcome",
          capped.to_dict() == capped.to_dict(), capped.to_dict())
    check("the dict returns a copy of the tool list, not the list",
          (lambda d: d["tools"].append("x") or d["tools"])(ended.to_dict())
          and ended.tools == ["a"], ended.tools)

    svc.store.close()
    svc.missions.close()
    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
