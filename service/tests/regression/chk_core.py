"""Do the eight never-tested core methods tell the truth?

Under test: `evaluate_tool`, `find_gaps`, `forge_tool`, `notify_channels`,
`resume_run`, `retire_tool`, `runs_forget`, `tool_usage`.

Why these eight. Measured 2026-11-07 by pulling `self.methods` out of
`service.py` with `ast` (76 names) and then grepping all 64 suite files for
three shapes of reference (`.name(`, `.name\\b`, `'name'`): 55 were touched by
something, 13 more were referenced somewhere, and exactly these eight appeared
nowhere. They are the half of the surface that had "it exists on the wire" as
its only evidence.

FIVE DEFECTS FOUND WHILE WRITING THIS SUITE, all fixed, all pinned below:

1. `forge_tool("")` had no empty-need guard. `run_python` refuses empty code in
   0.0 ms and `tool_define` refuses empty code, but `forge_tool` ran on: measured
   4124.6 ms, `_prelookup_market` alone 4132.8 ms of network to a tool market
   that is down on this machine, and only then `Could not forge a working tool
   for:  (0 rounds)`. A guard that runs after the side effect is not a guard.

2. `find_gaps` called every failure "Meta-cognition is disabled." -- and there
   is no disabled state in that method at all: no flag, no config read, nothing
   that can be switched off. With a pipeline that raises on any attribute
   access the reply was that sentence plus `RuntimeError: pipeline is broken on
   purpose`. `text` is the half a model reads; it said "off" when the feature
   had tried and broken.

3. `tool_define(code="")` registered the name first and probed second, so an
   unusable tool stayed in the library with `state: "probation"` and
   `error: entry 'X' not found or not callable` -- listed by `list_tools` and
   callable by nobody.

4. `log_event("")` recorded an event with no kind; `event_kinds()` then
   answered `[['', 1]]`, a group no reader can act on.

5. `remember("")` kept a fact with no key, the per-turn brief rendered it as
   `    : a value with no key` (one of twenty slots, unusable), and
   `forget("")` answered `forgotten: true` for it. `runs_forget`'s five
   refusals also returned bare `{"error": ...}` with no discriminator, while its
   success path answers `{"archived": <run id>, ...}` -- so a caller reading
   `.get("archived")` got `None` from a refusal and from a malformed answer
   alike.

A note on envelopes, because it was measured before it was assumed: a wire
reply of `{"ok": true, "result": {"ok": false, ...}}` is NOT laundering. The
outer `ok` means "the request was handled"; the inner one means "the operation
was refused". That is a legitimate two-layer design, and my first metric
counted eleven of them as defects when all eleven carried a discriminator. What
is a defect is a result with no discriminator at all, which is why (5) is on
this list and the other eleven are not.

  what this suite CAN check

    * every one of the eight answers with a dict and never raises;
    * the four refusals that used to be values are refusals now, and the two
      that legitimately return a menu (`notify_channels`) or a lookup
      (`resume_run` says `resumed: false` in so many words) still read
      honestly;
    * `forge_tool("")` refuses in under 200 ms and touches no network;
    * `find_gaps` names the phase it failed in, never claims a setting, and
      writes the fault to the ledger;
    * a forged tool can be evaluated, called, counted by `tool_usage`, and
      retired -- the success side of four methods that only had refusal
      evidence;
    * `runs_forget` answers the same key set whether it archived or refused.

  what it does NOT claim

    * that `forge_tool` can forge. Forging needs a model; `_tool_capable_model`
      probes up to five candidates at 90 s each, which is minutes, not a test
      budget. The forge pipeline is stubbed where a test needs it and the
      success path of `forge_tool` is listed as unverified rather than skipped
      quietly.
    * that `resume_run` can resume. It says in its own docstring that it does
      not; this checks the lookup, not a resume.
    * that `notify_send` delivers. No channel is configured on this machine, so
      only its refusal is checkable.

Run from the service directory:
    python tests/regression/chk_core.py

Nothing here touches the production database: every case runs in a fresh temp
home, and the network assertions fail closed (a recorded call is a failure).
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

# A temp db_path redirects the database and NOTHING ELSE. Measured 2026-11-07:
# a Service built on `<tmp>/main.db` still answered `resume_run("")` from
# `C:\Users\sumbger\AppData\Local\autoforge\runs` -- the production journal --
# because `runs.journal_dir()` follows the home, not the db. The suite therefore
# read live state while claiming to be isolated, and its own assertion caught it
# only because a real run happened to be there. Set the home before anything
# imports or constructs, and assert the isolation below rather than trusting it.
_HOME = os.environ.get("AUTOFORGE_HOME")
if not _HOME:
    import tempfile as _tf
    _HOME = _tf.mkdtemp(prefix="m217_core_home_")
    os.environ["AUTOFORGE_HOME"] = _HOME

import json                                                      # noqa: E402

from autoforge_svc.service import Service                        # noqa: E402
from autoforge_svc import runs as runs_mod                       # noqa: E402
from autoforge_svc.core.message import Message                   # noqa: E402

FAILS: list[str] = []

BODY = ("def {name}(**kwargs):\n"
        "    return {{'n': kwargs.get('n', 7)}}\n")
PARAMS = {"type": "object", "properties": {"n": {"type": "integer"}},
          "required": []}


def check(label: str, cond: bool, detail: object = "") -> None:
    # `detail: object` + repr, for the reason `chk_gpu.py` records: a failure
    # path that formats its own evidence must not be able to end the suite. A
    # suite that dies before printing a verdict scores as "crashed", and a
    # crashed suite proves nothing.
    print(("  OK   " if cond else "  FAIL ") + label
          + (("  <- " + repr(detail)) if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


def unverified(label: str, why: str) -> None:
    """Printed, never counted -- the same rule `chk_gpu.skipped` follows.

    A skipped line that looked like OK would be the same lie as counting a look
    nobody took.
    """
    print("  UNVERIFIED " + label + "  (" + why + ")")


def answered(fn, *args, **kwargs):
    """Call the product and turn an exception into the answer it did not give."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:                                   # noqa: BLE001
        return "<raised %s: %s>" % (type(exc).__name__, exc)


def fresh(tmp: str, name: str) -> Service:
    """A Service on its own database, never the production one."""
    return Service(db_path=os.path.join(tmp, name + ".db"))


# ---- a scripted model, so three of these are testable after all ----------
#
# Measured 2026-11-11: these three were listed as UNVERIFIED for a week with the
# reason "needs a model". The reason was wrong. `Service._forge_pipeline()`
# builds an `OpenAICompatClient` and caches it in `self._pipeline`; replacing
# that attribute with a pipeline over `MockLLMClient` gives the same code path
# with a scripted answer -- no network, no gateway, no 5x90 s probing. The first
# end-to-end run took 1.0 s and eleven `chat()` calls, all enumerable.
#
# The same trick covers `find_gaps`: it reads `pipeline.verifier.llm`, so the
# analyser is scripted by the same handler.

_VOWEL_CODE = ("def vowel_count(word: str = \"\"):\n"
               "    return sum(1 for ch in str(word).lower() if ch in 'aeiou')\n")


def _vowel_envelope(name: str = "vowel_count") -> str:
    """The generator's answer: one tool, with the two fields that decide a verdict.

    `sample_call` and `sample_expect` are not optional decoration. Measured
    2026-11-11: without them `execution` reports "UNPROBED: no valid sample call
    was declared", an unprobed check is `passed=False`, and `verify()` requires
    every check -- so a tool with no sample can never pass, whatever its code
    does. `probes` is the same story for `trigger`.
    """
    return "FIX-1\n" + json.dumps({
        "name": name,
        "description": "Count the vowels in a word.",
        "code": _VOWEL_CODE.replace("vowel_count", name),
        "parameters": {"type": "object",
                       "properties": {"word": {"type": "string"}},
                       "required": ["word"]},
        "entry": name,
        "sample_call": {"word": "banana"},
        "sample_expect": "3",
        "effect_signature": "pure",
        "probes": [{"query": "How many vowels are in 'banana'?",
                    "expect": "call",
                    "negative_query": "What is the square root of 5?"}],
        "rationale": "counting vowels comes up whenever text is classified",
        "tags": ["text"],
    })


def _gap_envelope() -> str:
    """The analyser's answer: one gap, whose proposed tool the pipeline then forges."""
    # A name of its own, deliberately: an earlier assertion in this suite
    # forges `vowel_count`, and `ForgePipeline.forge` refuses a name the library
    # already holds ("a forge is not a deliberate replacement"). Measured
    # 2026-11-11: reusing the name made `find_gaps` answer `forged: 0`, which
    # looks like the analyser failing to forge and is in fact the duplicate guard
    # working. A fixture that collides with itself tests the guard, not the gap.
    return json.dumps({"gaps": [{
        "need": "Count the vowels in a word",
        "rationale": "text classification keeps needing it",
        "proposed_tool": {"name": "vowel_counter",
                          "description": "Count the vowels in a word.",
                          "parameters": {"type": "object",
                                         "properties": {"word": {"type": "string"}},
                                         "required": ["word"]},
                          "tags": ["text"]},
    }]})


def _scripted_handler(messages, tools=None, **_kw):
    """Answer by role: analyser, generator, or the agent a verifier builds.

    The three callers share `tools=None` for the first two, so the system prompt
    is what tells them apart -- and that distinction is the fixture's job, not
    the product's. Getting it wrong shows up as "the model answered with an
    envelope instead of JSON", which is exactly how it first failed.
    """
    from autoforge_svc.core.llm import LLMResponse, tool_call
    system = messages[0].content if messages else ""
    if tools is not None:
        last = messages[-1].content if messages else ""
        if "square root" in last:
            return LLMResponse(content="not my job")
        # The offered tools, not a name this fixture likes: the analyser forges
        # `vowel_counter` while the earlier assertion forged `vowel_count`, and a
        # scripted agent that always names one of them scores the other as
        # unreachable -- a verdict about the fixture reported as a verdict about
        # the product.
        # `tools` arrives in the OpenAI wire shape --
        # `[{"type": "function", "function": {"name": ...}}]` -- not as a flat
        # spec. Reading `t["name"]` off that gives `None`, and `tool_call(None)`
        # builds a `ToolCall` whose name is `None`, which is what
        # `agent.py:645`'s `", ".join(tc.name for tc in resp.tool_calls)` then
        # dies on: `TypeError: sequence item 0: expected str instance,
        # NoneType found`. Measured 2026-11-11 -- three forge rounds ate it as
        # `forge_error` and the reply said "could not forge a working tool",
        # which blamed the model for a fixture that could not read its own
        # envelope. `forge_tool` succeeded the moment this line was fixed.
        offered = [
            (t.get("function") or {}).get("name")
            for t in (tools or [])
            if isinstance(t, dict) and isinstance(t.get("function"), dict)
        ]
        offered = [n for n in offered if isinstance(n, str) and n]
        name = offered[0] if offered else "vowel_count"
        return LLMResponse(content="", tool_calls=[tool_call(name, {"word": "banana"})])
    if "meta-cognitive architect" in system:
        return LLMResponse(content=_gap_envelope())
    # Name the tool after the prompt's own library listing. `find_gaps` answers
    # with `proposed_tool.name` and then calls `pipeline.forge(need)` -- the
    # generated name, not the proposed one, is what gets forged. Measured
    # 2026-11-11: a fixture that always answered `vowel_count` collided with the
    # tool an earlier assertion had already forged, `ForgePipeline.forge`
    # refused the duplicate name, and `find_gaps` reported `forged: 0` -- which
    # reads as the analyser failing to forge and is in fact the duplicate guard
    # doing its job. The prompt states the taken names; the fixture reads them
    # the same way a model is asked to.
    last = messages[-1].content if messages else ""
    taken = "vowel_count" in last and "do not duplicate" in last
    return LLMResponse(content=_vowel_envelope("vowel_counter" if taken else "vowel_count"))


def scripted_forge(svc) -> object:
    """Point `svc` at a pipeline over the scripted model and reset the throttle.

    `_metacog` is cleared because `MetaCognition` refuses to analyse twice within
    300 s and caches `_last_run`; a fixture that ran once would make the second
    call answer "no gaps" for a reason that has nothing to do with the code.
    """
    from autoforge_svc.autonomy.policy import AutonomyPolicy
    from autoforge_svc.core.llm import MockLLMClient
    from autoforge_svc.forge.generator import LLMToolGenerator
    from autoforge_svc.forge.pipeline import ForgeConfig, ForgePipeline
    from autoforge_svc.forge.verifier import ToolVerifier

    llm = MockLLMClient(handler=_scripted_handler)
    verifier = ToolVerifier(llm, sandbox=svc.sandbox,
                            run_execution_check=True, run_trigger_check=True,
                            run_negative_check=True, run_claim_check=True)
    pipeline = ForgePipeline(LLMToolGenerator(llm, max_tokens=2048), verifier,
                             svc.registry, config=ForgeConfig(), sandbox=svc.sandbox,
                             policy=AutonomyPolicy(), on_event=lambda k, p: None)
    svc._pipeline = pipeline
    svc._metacog = None
    return llm


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m217_core_")
    svc = fresh(tmp, "main")
    try:
        # ---- the suite is actually isolated (asked, not assumed) ------------
        journal = str(runs_mod.journal_dir())
        check("the run journal is the one this suite created, not the live one",
              journal.lower().startswith(_HOME.lower()), journal)
        # ---- all eight answer with a dict, none raises -----------------------
        cases = [
            ("evaluate_tool", lambda: svc.evaluate_tool("no_such_tool")),
            ("notify_channels", lambda: svc.notify_channels()),
            ("retire_tool", lambda: svc.retire_tool("no_such_tool", "because")),
            ("runs_forget", lambda: svc.runs_forget("no-such-run", "why")),
            ("tool_usage", lambda: svc.tool_usage()),
            ("resume_run", lambda: svc.resume_run("")),
        ]
        for name, fn in cases:
            got = answered(fn)
            check("%s answers with a dict" % name,
                  isinstance(got, dict), got if not isinstance(got, dict) else "")

        # `forge_tool("")` and `find_gaps` are timed/stubbed below, so they are
        # not in the loop above: one has a budget to measure and the other needs
        # a broken pipeline to say anything interesting.

        # ---- a missing name is answered before any model is asked ----------
        #
        # Measured 2026-11-12: this loop passed only while a model could be
        # found. `evaluate_tool` looked the name up through `self._registry()`,
        # which builds the forge pipeline -- and building it probes the gateway
        # for a model that can read tool results. So on a machine where the
        # configured model is not served, asking about a tool that does not
        # exist raised RuntimeError about the gateway. A name that is not in the
        # library cannot be helped by any model, so the lookup has to answer
        # first; the assertion below pins that, not just the dict shape.
        missing = answered(svc.evaluate_tool, "no_such_tool")
        check("a missing tool is refused by name", missing.get("ok") is False, missing)
        check("...and no pipeline was built to answer it",
              getattr(svc, "_pipeline", None) is None, type(getattr(svc, "_pipeline", None)).__name__)

        # ---- the scripted pipeline goes in FIRST, before any forge is touched
        #
        # Measured 2026-11-11, and it cost a 300-second hang: the empty-need
        # probe used to run before this install, so when a mutant removed the
        # guard, `forge_tool("")` fell through to the live gateway and spent
        # three real model rounds dying on a 502. The probe then reported a
        # timeout instead of the missing guard -- it was measuring the network.
        # Anything below that reaches a forge has to reach this one.
        llm = scripted_forge(svc)

        # ---- 1. forge_tool refuses an empty need before doing anything ------
        touched: list[str] = []
        real_prelookup = svc._prelookup_market
        svc._prelookup_market = lambda *a, **k: touched.append("prelookup")
        try:
            t0 = time.perf_counter()
            empty_forge = answered(svc.forge_tool, "")
            forge_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            svc._prelookup_market = real_prelookup
        check("forge_tool('') answers with a dict", isinstance(empty_forge, dict),
              empty_forge)
        check("...and refuses", empty_forge.get("ok") is False, empty_forge)
        check("...saying the need was empty",
              "need was empty" in str(empty_forge.get("error", "")), empty_forge)
        # The measurement that named the defect: 4124.6 ms, nearly all of it in
        # `_prelookup_market` reaching a tool market that is down.
        check("...in well under the 4.1 s the network used to cost",
              forge_ms < 200.0, round(forge_ms, 1))
        check("...without reaching the tool market at all", touched == [], touched)

        # ---- 2. find_gaps names the phase and never claims a setting --------
        class Boom:
            def __getattr__(self, name):
                raise RuntimeError("pipeline is broken on purpose")

        broken = fresh(tmp, "broken")
        try:
            broken._forge_pipeline = lambda *a, **k: Boom()
            gaps = answered(broken.find_gaps)
            check("a broken pipeline is not reported as a successful analysis",
                  isinstance(gaps, dict) and gaps.get("ok") is False, gaps)
            check("...the failing phase is named",
                  gaps.get("phase") in ("import", "pipeline", "analyser",
                                        "analysis"), gaps.get("phase"))
            check("...and it is the analysing phase, since the pipeline built",
                  gaps.get("phase") == "analyser", gaps.get("phase"))
            text = str(gaps.get("text", ""))
            # The old sentence claimed a setting. There is no setting; the
            # method has no flag and reads no config.
            check("...and the sentence says fault, not setting",
                  "fault, not a setting" in text, text[:160])
            check("...and it does not blame a switch",
                  "is disabled" not in text, text[:160])
            check("...while the underlying exception is still reported",
                  "pipeline is broken on purpose" in str(gaps.get("error", "")),
                  gaps.get("error"))
            check("...and the fault is written to the ledger",
                  any(k == "find_gaps_failed" for k, _ in broken.store.event_kinds()),
                  broken.store.event_kinds())
        finally:
            broken.store.close()
            broken.missions.close()

        # ---- 3. tool_define with no code takes no name ----------------------
        define = answered(svc.tool_define, "empty_body_probe", "",
                          "a body that is not there", PARAMS)
        check("tool_define refuses an empty body", isinstance(define, dict), define)
        check("...without registering the name",
              define.get("registered") is False, define)
        check("...and says nothing was defined",
              "nothing was defined" in str(define.get("error", "")) or
              "code was empty" in str(define.get("error", "")), define)
        check("...so the library does not hold it",
              svc.registry.get("empty_body_probe") is None,
              svc.registry.names())

        # The control: a real body does register, so the guard is not refusing
        # everything.
        good = answered(svc.tool_define, "core_probe", BODY.format(name="core_probe"),
                        "returns n", PARAMS)
        check("a real body still registers", good.get("registered") is True, good)
        check("...and runs", good.get("ok") is True, good)

        # ---- 4. log_event refuses an empty kind -----------------------------
        logged = answered(svc.log_event, "", {})
        check("log_event refuses an empty kind", logged.get("ok") is False, logged)
        check("...and logs nothing", logged.get("logged") is None, logged)
        check("...so no empty group appears among the recorded kinds",
              "" not in [k for k, _ in svc.store.event_kinds()],
              svc.store.event_kinds())
        named = answered(svc.log_event, "core_probe_event", {"n": 1})
        check("a named kind still lands", named.get("logged") == "core_probe_event",
              named)

        # ---- the success side of four methods that only had refusals --------
        usage_before = svc.tool_usage()
        check("tool_usage answers with a dict", isinstance(usage_before, dict),
              usage_before)
        check("...under a last_called key",
              "last_called" in usage_before, sorted(usage_before))
        called = answered(svc.tool_call, "core_probe", {"n": 5})
        check("a freshly forged tool can be called", called.get("ok") is True, called)
        usage_after = svc.tool_usage()
        last = (usage_after.get("last_called") or {})
        check("tool_usage then knows it was called", "core_probe" in last, last)
        check("...with a timestamp a reader can act on",
              isinstance(last.get("core_probe"), (int, float)), last)

        evaluated = answered(svc.evaluate_tool, "core_probe")
        check("evaluate_tool answers about a real tool", isinstance(evaluated, dict),
              evaluated)
        check("...and the answer is about this tool",
              "core_probe" in repr(evaluated), evaluated)

        retired = answered(svc.retire_tool, "core_probe", "it was a probe")
        check("retire_tool retires a real tool", retired.get("ok") is True, retired)
        # Retirement is a state, not a deletion -- measured 2026-11-07: the name
        # stays in `names()` and `get()` still answers, with `state: "retired"`,
        # while `schemas()` stops offering it. My first assertion demanded
        # absence and was wrong: a retired tool that vanished from the registry
        # could not be rehabilitated, and `rehab_tool` is the documented way
        # back. What a caller must be able to rely on is that it is no longer
        # offered to the model.
        still = svc.registry.get("core_probe")
        check("...leaving it in the library, marked retired",
              still is not None and getattr(still, "state", "") == "retired",
              getattr(still, "state", None))
        offered = [s.get("name") for s in svc.registry.schemas()]
        check("...and no longer offering it to the model",
              "core_probe" not in offered, offered)

        # The rationale guard is before the lookup, on purpose: a retirement
        # nobody wrote down is indistinguishable from one that never happened.
        no_reason = answered(svc.retire_tool, "core_probe", "")
        check("retiring without a reason is refused", no_reason.get("ok") is False,
              no_reason)
        check("...in a sentence that names the missing reason",
              "rationale" in str(no_reason.get("text", "")).lower(), no_reason)

        # ---- the builtin refusal, made reachable on purpose -----------------
        # Nothing in this package can register a builtin (every Agent is built
        # with `allow_self_terminate=False`), so the branch had never run and
        # its sentence had never been read. It told the caller to "Disable it
        # with set_autonomy instead" -- a method that does not exist among the
        # 76 wire methods, is not an attribute on Service, and is not exported
        # by the autonomy module. A refusal whose advice cannot be followed is
        # worse than a plain no. A spec is registered here so the branch runs
        # and the sentence is checked rather than assumed.
        from autoforge_svc.tools.spec import ToolSpec, ToolState
        built = ToolSpec(name="core_builtin", description="a stand-in builtin",
                         parameters={"type": "object", "properties": {}},
                         fn=lambda **kw: 1,
                         code="def core_builtin(**kw):\n    return 1\n",
                         source="builtin", state=ToolState.ACTIVE)
        svc.registry.register(built)
        check("a builtin spec can be put in the library for this test",
              svc.registry.get("core_builtin") is not None, svc.registry.names())
        refused = answered(svc.retire_tool, "core_builtin", "it is a builtin")
        check("retiring a builtin is refused", refused.get("ok") is False, refused)
        btext = str(refused.get("text", ""))
        check("...saying builtins are not retired",
              "builtins are not retired" in btext, btext[:200])
        # The defect: advice naming a method that does not exist.
        check("...and not naming a method this service does not have",
              "set_autonomy" not in btext, btext[:200])
        check("...but stating plainly that no alternative exists",
              "no alternative call to offer" in btext, btext[:200])
        check("...and the builtin is still there afterwards",
              svc.registry.get("core_builtin") is not None, svc.registry.names())
        svc.registry.unregister("core_builtin")

        # ---- 5. the nameless fact -------------------------------------------
        nameless = answered(svc.remember, "", "a value with no key")
        check("remember refuses a fact with no key", nameless.get("ok") is False,
              nameless)
        check("...and keeps nothing", svc.recall("") == [], svc.recall(""))
        forgot = answered(svc.forget, "")
        check("forget refuses an empty key rather than claiming a removal",
              forgot.get("forgotten") is False, forgot)
        kept = answered(svc.remember, "core_named_fact", "kept")
        check("a named fact is still kept", kept.get("key") == "core_named_fact",
              kept)
        check("...and can be read back",
              any(r.get("key") == "core_named_fact" for r in svc.recall("core_named")),
              svc.recall("core_named"))
        check("...and forgotten by name", svc.forget("core_named_fact").get("forgotten")
              is True, svc.forget("core_named_fact"))

        # ---- runs_forget answers the same key set either way ----------------
        # Before the fix the five refusals were bare `{"error": ...}` while the
        # success path answered `{"archived": <run id>, ...}`. A caller reading
        # `.get("archived")` could not tell a refusal from a malformed answer.
        #
        # Each case also names the sentence it expects, and that half is not
        # decoration. Measured 2026-11-07 with the `why` guard deleted: the
        # no-reason case fell through to the match count and answered
        # `'anything' matched 0 run(s)` -- still a dict, still `archived: None`,
        # still `ok: False`. Every shape assertion passed and the mutant escaped.
        # A refusal that names the wrong reason is a different fact from a
        # refusal, and only the reason can tell them apart.
        refusals = [
            ("no run id", ("", "because"), "run_id is required"),
            ("no reason", ("anything", ""), "a reason is required"),
            ("no match", ("no-such-run-anywhere", "why"), "matched 0 run(s)"),
        ]
        for label, args, expected in refusals:
            reply = answered(svc.runs_forget, *args)
            check("runs_forget refuses (%s) with a dict" % label,
                  isinstance(reply, dict), reply)
            check("...and carries the archived key (%s)" % label,
                  "archived" in reply, sorted(reply))
            check("...which is None, not a run id (%s)" % label,
                  reply.get("archived") is None, reply.get("archived"))
            check("...and says it is not ok (%s)" % label,
                  reply.get("ok") is False, reply.get("ok"))
            check("...blaming the thing that is actually missing (%s)" % label,
                  expected in str(reply.get("error", "")), reply.get("error"))

        # ---- notify_channels is a menu, and says which channels are broken --
        channels = svc.notify_channels()
        check("notify_channels answers with channels and problems",
              set(channels) >= {"channels", "problems"}, sorted(channels))
        check("...both lists", isinstance(channels.get("channels"), list)
              and isinstance(channels.get("problems"), list), channels)
        # It has no `ok` key and should not: it is a read-only menu, not an
        # operation. Asserting an `ok` here would be demanding a verdict about
        # nothing.
        sent = answered(svc.notify_send, "hello")
        check("notify_send with no channel configured refuses",
              sent.get("ok") is False, sent)
        check("...naming the missing configuration",
              "no notification channels" in str(sent.get("error", "")), sent)
        # Naming the *path* is the part that was missing until 2026-11-10. Both
        # refusals used to say "in the config file" and name no file; the file
        # meant is `configfile.config_path()`, which is outside home_dir() and
        # which nothing in this tree writes. A reader told "add it to the config
        # file" without a path has to guess, and on Windows the obvious guess is
        # not where the package looks.
        from autoforge_svc import configfile as _cfg
        _hint_path = str(_cfg.config_path())
        check("...and it names the file to edit, not just 'the config file'",
              _hint_path in str(sent.get("error", "")), sent)
        check("...and says this package does not create it",
              "nothing in this package writes it" in str(sent.get("error", "")), sent)
        # The menu path (describe) must agree with the send path: two sentences
        # about the same absence that disagree would be the older defect again.
        from autoforge_svc.notify import Notifier as _N
        check("the empty menu names the same file the refusal does",
              _hint_path in _N([]).describe(), _N([]).describe())
        check("...and neither of them still says only 'the config file'",
              "in the config file" not in str(sent.get("error", ""))
              and "in the config file" not in _N([]).describe(),
              (str(sent.get("error", ""))[:80], _N([]).describe()[:80]))

        # ---- resume_run says what it is ------------------------------------
        resume = svc.resume_run("")
        check("resume_run answers with a dict", isinstance(resume, dict), resume)
        check("...and does not claim to have resumed anything",
              resume.get("resumed") is False, resume.get("resumed"))
        # Two shapes, both honest: with no journal it says where it looked; with
        # one it hands back the evidence and still says `resumed: false`. This
        # temp home is empty, so the first is what must appear -- and if it does
        # not, the suite is reading a journal it did not create.
        err = str(resume.get("error", ""))
        check("...and either names the empty journal or returns a run it found",
              ("no runs journalled" in err and _HOME.lower() in err.lower())
              or resume.get("run_id"), (err[:120], resume.get("run_id")))

        # ---- the three that used to be "needs a model", now measured -------
        #
        # Kept as one block so the reader can see they were answered together:
        # the same scripted pipeline drives all three (installed near the top of
        # this function), and none of them opens a socket. If any of these
        # fails, the reason is in `text` -- the suite prints the whole reply
        # rather than the field it expected.

        forged = answered(svc.forge_tool, "Count the vowels in a word.")
        check("forge_tool forges a working tool",
              isinstance(forged, dict) and forged.get("ok") is True, forged)
        check("...and says the tool is active, not merely defined",
              isinstance(forged, dict) and forged.get("state") == "active",
              (forged or {}).get("state") if isinstance(forged, dict) else forged)
        check("...in one round, so the first answer was accepted",
              isinstance(forged, dict) and forged.get("rounds") == 1,
              (forged or {}).get("rounds") if isinstance(forged, dict) else forged)
        forged_spec = svc.registry.get("vowel_count")
        check("...and the library now holds it", forged_spec is not None,
              svc.registry.names())
        # The tool itself, not just its registration: a spec that answers the
        # wrong number with an `ok: True` beside it is the failure mode this
        # whole suite exists for. Called the way the model calls it, because a
        # forged spec's `fn` is a stub that refuses in-process execution on
        # purpose ("generated tool attempted in-process execution; use the
        # sandbox runner") -- measured 2026-11-11, when calling `fn` directly
        # produced that refusal and I first read it as a forge failure.
        called = answered(svc.tool_call, "vowel_count", {"word": "banana"})
        check("...and it computes the thing it was asked for",
              "3" in str(called), called)

        gaps = answered(svc.find_gaps)
        check("find_gaps proposes real gaps",
              isinstance(gaps, dict) and gaps.get("ok") is True, gaps)
        check("...at least one, not a shrug",
              isinstance(gaps, dict) and (gaps.get("gaps_found") or 0) >= 1, gaps)
        check("...and forges what it proposed rather than only naming it",
              isinstance(gaps, dict) and (gaps.get("forged") or 0) >= 1, gaps)
        check("...naming the gap it found, not just a count",
              isinstance(gaps, dict) and "Count the vowels in a word" in str(gaps.get("text", "")),
              (gaps or {}).get("text") if isinstance(gaps, dict) else gaps)

        # `resume_run` needed an interrupted journal, which is a fixture, not a
        # model. The writer pid is one that cannot be running, and the two
        # timestamps are far enough apart that `claim()` has settled.
        now = time.time()
        mark = runs_mod.Mark(
            run_id="cafebabef00d",
            task="Count the vowels in 'banana'",
            started_at=now - 120.0,
            updated_at=now - 60.0,
            writer_pid=999999,
            turns=3,
            last_action="about to call vowel_count",
            messages=[Message.user("Count the vowels in 'banana'"),
                      Message.assistant("I will call vowel_count.")],
        )
        runs_mod._atomic_write(runs_mod._path(mark.run_id),
                              json.dumps(mark.to_dict(), ensure_ascii=False))
        found = answered(svc.resume_run, "")
        check("resume_run returns a transcript",
              isinstance(found, dict) and found.get("run_id") == "cafebabef00d", found)
        check("...with the turns and the message count, not the messages",
              isinstance(found, dict) and found.get("turns") == 3
              and found.get("messages") == 2,
              (found.get("turns"), found.get("messages")) if isinstance(found, dict) else found)
        check("...and a prompt that carries the last thing the run was doing",
              isinstance(found, dict) and "vowel_count" in str(found.get("prompt", "")),
              str((found or {}).get("prompt", ""))[:200] if isinstance(found, dict) else found)
        check("...while still refusing to claim it re-executed anything",
              isinstance(found, dict) and found.get("resumed") is False
              and "nothing has been re-executed" in str(found.get("next_step", "")), found)

        print()
        print("TOTAL FAILURES:", len(FAILS), FAILS)
        return 0 if not FAILS else 1
    finally:
        for s in (svc,):
            try:
                s.store.close()
                s.missions.close()
            except Exception:                                  # noqa: BLE001
                pass
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
