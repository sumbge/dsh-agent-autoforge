"""Can a tool be reviewed without being re-verified, and does a brief reach a model?

Under test: `forge/validity.py`'s `review_tool` / `ReviewVerdict` and
`core/agent.py`'s `Agent.apply_role`, plus the `Spawner` path that calls the
hook.

Two separate questions, both asked because both were answered "no" in the code
this file was written against:

**The reviewer.** `ValidityGate` answers a question about *change*: is this
mutant worse than the baseline it was bred from? Its strongest scope checks
(`scope_creep`, and `require_scope_declaration`) mean nothing without a
`FrozenBaseline`, and the second one defaults to False -- measured 2026-11-04, a
tool declaring `pure` that writes a file passes the gate on a first approval.
So "review the tool you just forged" had no entry point at all: the only
verdict available was the battery's, which asks whether the tool *works*, not
whether it told the truth about itself.

**The brief.** `autonomy/spawn.py:146` has read
`hook = getattr(child, "apply_role", None)` since it was written. `core/agent.py`
never defined it. So a `SpawnRecord` reported an honest `brief_chars: 265` for a
265-character directive that reached a dataclass and stopped there. `roles.py`
warns about precisely this -- "A hint the child never sees is a note to the file"
-- and the code beside the warning was the note.

The tests are written to fail in the direction that matters. A reviewer that
passes everything is worse than none: it converts "unreviewed" into "clean". So
the objections are tested harder than the passes, and every shape the reviewer
*cannot* judge is required to say so rather than return silence.

Run from the service directory:

    python tests/regression/chk_review.py

The database is a throwaway under %TEMP%; the production one is never opened.
"""
from __future__ import annotations

import os
import sys

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc.autonomy.roles import ceiling, role_brief            # noqa: E402
from autoforge_svc.autonomy.spawn import Spawner                       # noqa: E402
from autoforge_svc.core.agent import Agent, DEFAULT_SYSTEM            # noqa: E402
from autoforge_svc.core.llm import LLMResponse, MockLLMClient          # noqa: E402
from autoforge_svc.forge.validity import review_tool                   # noqa: E402
from autoforge_svc.tools.registry import ToolRegistry                  # noqa: E402
from autoforge_svc.tools.spec import ToolSpec, ToolState               # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    """One assertion. `detail` takes anything and is stringified on the way out.

    It used to be typed `str`, and callers passed lists anyway -- which was
    invisible while those assertions passed, because the detail is only
    formatted on failure. Measured 2026-11-04: a mutant that filed the review
    receipt under the wrong kind made this raise
    `TypeError: can only concatenate str (not "list") to str`, killing the run
    after one FAIL and before the verdict line. **A helper that can only fail on
    the failing path is a helper that hides failures.**
    """
    if cond:
        print("  OK   " + label)
        return
    shown = "" if detail == "" else "  <- " + str(detail)
    print("  FAIL " + label + shown)
    FAILS.append(label)


def spec(name: str, signature: str, code: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        fn=lambda **kw: 1,
        code=code,
        parameters={},
        state=ToolState.ACTIVE,
        effect_signature=signature,
    )


READ_ONE_FILE = "def main(path):\n    with open(path) as fh:\n        return fh.read()\n"
WRITE_ONE_FILE = "def main(path):\n    with open(path, 'w') as fh:\n        fh.write('x')\n"
REACH_NETWORK = (
    "def main():\n"
    "    import urllib.request\n"
    "    return urllib.request.urlopen('http://example.com').read()\n"
)
DELETE_ONE_FILE = "def main(path):\n    import os\n    os.remove(path)\n"


def detail_of(verdict, name: str) -> str:
    """The detail of one finding, or "" when that finding is absent.

    Written because the first version of this file indexed `objectors[0]`
    directly, and a mutant that removed the objection turned the suite into an
    `IndexError` at line 102 -- the run died with two honest FAILs printed and no
    verdict line at all. **A crash is not a failure**: it says the file stopped,
    not which claims are untrue, and it silently skips every assertion after it.
    """
    for f in getattr(verdict, "objectors", []):
        if f.name == name:
            return f.detail
    return ""


def main() -> int:
    print("-- the reviewer answers a question the battery does not ask")
    honest = review_tool(spec("pure_reader", "reads:the file the operator names", READ_ONE_FILE))
    check("an honest reader is not objected to", honest.ok, honest.summary())
    check("and the review actually ran", honest.reviewed, honest.summary())
    check("the scope it read is the one the gate will use",
          honest.scope == "read_only", repr(honest.scope))
    check("the effects it found are the ones the code exercises",
          honest.effects == [], repr(honest.effects))

    print("-- the objection: a signature the code contradicts")
    liar = review_tool(spec("pure_liar", "pure", WRITE_ONE_FILE))
    check("a tool that claims pure and writes is objected to", not liar.ok, liar.summary())
    check("the objection is the contract, not the signature",
          [f.name for f in liar.objectors] == ["contract"],
          repr([f.name for f in liar.objectors]))
    check("and it names the capability it was not permitted",
          "filesystem_write" in detail_of(liar, "contract"), detail_of(liar, "contract"))
    net = review_tool(spec("net_liar", "pure", REACH_NETWORK))
    check("the same holds for the network", not net.ok, net.summary())
    check("which it also names",
          "network_egress" in detail_of(net, "contract"), detail_of(net, "contract"))

    print("-- a claim that is too small is an objection; one that is big enough is not")
    honest_net = review_tool(spec("honest_net", "network", REACH_NETWORK))
    check("a tool that admits it reaches the network is not objected to",
          honest_net.ok, honest_net.summary())
    check("a tool that admits it writes is not objected to -- the review is "
          "about the claim, not about the capability",
          review_tool(spec("honest_writer", "writes:file", WRITE_ONE_FILE)).ok)
    check("a write tool declaring merely read_only is objected to",
          not review_tool(spec("half_honest", "reads:file", WRITE_ONE_FILE)).ok)

    print("-- what the reviewer must refuse to judge")
    unreadable = review_tool(spec("mute", "", READ_ONE_FILE))
    check("an unreadable signature is objected to under its own name",
          [f.name for f in unreadable.objectors] == ["signature"],
          repr([f.name for f in unreadable.objectors]))
    check("and the objection says what the gate will do with it",
          "undeclared" in detail_of(unreadable, "signature"), detail_of(unreadable, "signature"))
    check("a spec with no code is not reviewed", not review_tool(spec("blank", "read_only", "")).reviewed)
    check("and it is not reported as clean either", not review_tool(spec("blank", "read_only", "")).ok)
    check("code that is only whitespace counts as no code",
          not review_tool(spec("spaces", "read_only", "   \n\t\n")).reviewed)

    print("-- the review is readable as data, not only as prose")
    d = honest.to_dict()
    check("the dict carries the verdict", d.get("ok") is True, repr(d.get("ok")))
    check("the dict carries every finding", len(d.get("findings") or []) == 3,
          repr(len(d.get("findings") or [])))
    check("a finding that carries evidence carries it",
          detail_of(liar, "contract") != "" and
          any(f.get("name") == "contract" and "data" in f
              for f in liar.to_dict()["findings"]),
          repr(liar.to_dict()["findings"]))

    print("-- a reviewer that raises reports that, it does not pass silently")
    broken = spec("broken", "read_only", READ_ONE_FILE)
    broken.code = 12345                      # audit_effects wants a str
    verdict = review_tool(broken)
    check("the crash is reported as an error rather than a clean verdict",
          not verdict.ok and bool(verdict.error), verdict.summary())
    check("and the record says it was not reviewed", not verdict.reviewed)

    print("-- apply_role: the hook spawn.py has always looked for")
    llm = MockLLMClient(script=[LLMResponse(content="done")])
    agent = Agent(llm, ToolRegistry(), max_turns=2)
    check("an Agent has the attribute the spawner asks for",
          callable(getattr(agent, "apply_role", None)))

    def apply(target, brief: str, role: str) -> bool:
        """Call the hook, and let a missing hook be a FAIL instead of a crash.

        Measured: with `apply_role` renamed, the attribute check above failed
        honestly and then line 172 raised `AttributeError`, so the suite stopped
        with 1 FAIL printed and 20 assertions never run -- including all six
        about whether a brief reaches a model. A suite that dies when the thing
        it tests is absent is a suite that cannot report absence.
        """
        hook = getattr(target, "apply_role", None)
        if not callable(hook):
            check(f"apply_role exists (needed to call it with {brief!r})", False,
                  "no callable apply_role on this object")
            return False
        try:
            hook(brief, role)
            return True
        except Exception as exc:                              # noqa: BLE001
            check(f"apply_role({brief!r}, {role!r}) does not raise", False,
                  f"{type(exc).__name__}: {exc}")
            return False

    base = agent.system_prompt
    check("the base prompt is what the child was built with",
          base == DEFAULT_SYSTEM or bool(base))
    apply(agent, "You are a gate. Refuse writes.", "critic")
    check("the brief is in the prompt now",
          "You are a gate. Refuse writes." in agent.system_prompt)
    check("with a heading that names the role",
          "## Your role: critic" in agent.system_prompt)
    check("and the base prompt survived underneath it",
          agent.system_prompt.startswith(base), agent.system_prompt[:80])

    apply(agent, "A second brief entirely.", "worker")
    check("a second brief replaces the first role section",
          agent.system_prompt.count("You are a gate") == 0)
    check("the new one is present", "A second brief entirely." in agent.system_prompt)
    check("and the base prompt was not repeated a third time",
          agent.system_prompt.count(base.strip()[:40]) == 1)

    before = agent.system_prompt
    apply(agent, "", "")
    check("an empty brief changes nothing", agent.system_prompt == before)

    # The failure this whole item exists to prevent is a brief that reaches a
    # dataclass and stops. Assert on what the model is handed, not on the field.
    #
    # And assert against the brief `roles.py` actually publishes, not against a
    # sentence written here: the first version of this test looked for "You are
    # a gate" -- the words a hand-written probe had used -- while the real
    # `role_brief("critic")` opens "You are a critic". It failed for the wrong
    # reason, and the fix was to read the real text rather than to retarget the
    # assertion, because a test that asserts its own fixture proves nothing
    # about the code that ships.
    real_brief = role_brief("critic")
    marker = real_brief.split(".")[0]
    llm2 = MockLLMClient(script=[LLMResponse(content="ok")])
    seen: list[str] = []

    def recorder(msgs, tools=None, **kw):
        # The handler's signature is the client's own (`chat` forwards `tools` as
        # a positional). A narrower lambda raises inside the loop, and the test
        # then fails with a traceback instead of a verdict -- measured here on
        # the first run.
        seen.append("\n".join(m.content for m in msgs if m.role == "system"))
        return LLMResponse(content="ok")

    llm2.handler = recorder
    agent2 = Agent(llm2, ToolRegistry(), max_turns=2)
    if apply(agent2, real_brief, "critic"):
        try:
            agent2.run("review something")
        except Exception as exc:                              # noqa: BLE001
            check("the agent runs with a role applied", False,
                  f"{type(exc).__name__}: {exc}")
    check("the brief reached the request the model was given",
          bool(seen) and marker in seen[0],
          (seen[0][:160] if seen else "<no request recorded>"))

    print("-- actually through a Spawner, not just by calling the method")
    built: list[Agent] = []

    def factory(_spawner, registry):
        child = Agent(MockLLMClient(script=[LLMResponse(content="child done")]),
                      registry, max_turns=2)
        built.append(child)
        return child

    reg = ToolRegistry()
    spawner = Spawner(registry=reg, agent_factory=factory)
    rec = spawner.spawn("review the file_facts tool", role="critic")
    check("the spawn produced a child", bool(built), rec.error or "")
    check("the record still reports the brief it carried", len(rec.brief) > 100,
          str(len(rec.brief)))
    check("and the child really was briefed -- this is the line that used to be dead",
          bool(built) and marker in built[0].system_prompt,
          (built[0].system_prompt[-200:] if built else ""))
    check("the brief arrived under a heading that names the role it was given",
          bool(built) and "## Your role: critic" in built[0].system_prompt,
          (built[0].system_prompt[-200:] if built else ""))
    check("the critic child was scoped, not given the whole registry",
          bool(built) and built[0].registry is not reg)
    check("no error was recorded", rec.error is None, str(rec.error))

    print("-- a child that cannot be briefed is not a failure")
    class Deaf:
        def __init__(self, **_kw):
            self.registry = _kw.get("registry")

        def run(self, task):
            class R:
                content = "deaf but fine"
            return R()

    spawner2 = Spawner(registry=ToolRegistry(),
                       agent_factory=lambda s, r: Deaf(registry=r))
    rec2 = spawner2.spawn("do a thing")
    check("a child without the hook still runs",
          rec2.result == "deaf but fine", str(rec2.result))
    check("and no error is invented for the absent hook", rec2.error is None)

    print("-- the critic's ceiling is still a ceiling")
    check("a critic may not take a write tool",
          ceiling("critic") is not None)
    writer = spec("writer_probe", "writes:file", WRITE_ONE_FILE)
    check("the ceiling refuses the write", bool(ceiling("critic")(writer)),
          str(ceiling("critic")(writer)))
    reader = spec("reader_probe", "reads:file", READ_ONE_FILE)
    check("and permits the read", ceiling("critic")(reader) == "",
          str(ceiling("critic")(reader)))

    print("-- the service layer files a receipt and moves nothing")
    import json as _json
    import sqlite3 as _sqlite3
    import tempfile

    from autoforge_svc.service import Service

    tmp = tempfile.mkdtemp(prefix="m19_review_")
    db = os.path.join(tmp, "review.db")
    svc = Service(db_path=db)
    try:
        liar = spec("svc_liar", "pure", WRITE_ONE_FILE)
        svc.registry.register(liar, replace=True)
        check("registering the tool under review works", "svc_liar" in svc.registry.names(),
              sorted(svc.registry.names()))
        stored = svc.registry.get("svc_liar")
        check("the registry holds it as a spec, not None", stored is not None, repr(stored))
        before_state = stored.state if stored is not None else None
        res = svc.review_tool("svc_liar")
        check("the service answers a review", res.get("ok") is False, str(res)[:160])
        check("it carries the verdict shape, not just a sentence",
              all(k in res for k in ("tool", "scope", "effects", "findings", "reviewed", "ok")),
              sorted(res))
        check("the text is the verdict's own summary", res.get("text", "").startswith("review of svc_liar"),
              str(res.get("text"))[:80])
        after = svc.registry.get("svc_liar")
        check("reviewing does not change the tool's state",
              after is not None and after.state == before_state,
              f"{before_state} -> {getattr(after, 'state', None)}")

        miss = svc.review_tool("no_such_tool_here")
        check("an unknown name is refused rather than invented", miss.get("ok") is False, str(miss)[:120])
        check("and the refusal says what the library does hold", "library holds" in miss.get("error", ""),
              miss.get("error", ""))

        svc.review_tool("svc_liar")
        svc.review_tool("no_such_tool_here")
    finally:
        svc.store.close()
        svc.missions.close()

    con = _sqlite3.connect(db)
    try:
        rows = con.execute("SELECT kind, payload FROM forge_events WHERE kind IN ('review','policy')").fetchall()
    finally:
        con.close()
    reviews = [r for r in rows if r[0] == "review"]
    check("each answered review is on the ledger, and the empty miss is not",
          len(reviews) == 2, str(len(reviews)))
    check("the receipt is filed as a review, not as a policy verdict",
          {r[0] for r in rows} == {"review"}, sorted({r[0] for r in rows}))
    payload = _json.loads(reviews[0][1]) if reviews else {}
    check("the receipt names what was objected to",
          payload.get("objected") == ["contract"], str(payload.get("objected")))
    check("and it records that a verdict was actually reached",
          payload.get("reviewed") is True, str(payload.get("reviewed")))

    print("-- what this file cannot see")
    check("the reviewer does not check that the code runs -- that is the battery's job",
          not hasattr(review_tool(spec("x", "pure", "raise SystemExit(1)\n")), "ran"))
    # Until 2026-11-05 this read "deletion still widens to privileged: the table
    # has no scope for it". The table has one now, so the reviewer no longer has
    # to over-permit on a deleter's behalf -- and this assertion is the place
    # that would notice if the scope were removed again.
    check("deletion is reviewed as destructive, not as everything",
          review_tool(spec("deleter", "deletes:file", DELETE_ONE_FILE)).scope == "destructive",
          review_tool(spec("deleter", "deletes:file", DELETE_ONE_FILE)).scope)
    check("and a tool that writes and deletes still widens: no scope covers both",
          review_tool(spec("both", "deletes:file, writes:file",
                           DELETE_ONE_FILE + "def w(**k):\n    open(1,\"w\")\n"))
          .scope == "privileged",
          review_tool(spec("both", "deletes:file, writes:file",
                           DELETE_ONE_FILE + "def w(**k):\n    open(1,\"w\")\n")).scope)

    print()
    if FAILS:
        print(f"TOTAL FAILURES: {len(FAILS)} {sorted(set(FAILS))}")
        return 1
    print("TOTAL FAILURES: 0 []")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
