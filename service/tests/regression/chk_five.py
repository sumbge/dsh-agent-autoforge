"""The five methods no suite had ever executed.

Under test: `tool_list`, `events`, `my_capabilities`, `run_python`,
`evolve_tool`.

Why these five, and why "five" is a smaller number than it looks. Measured
2026-11-07 with two independent instruments, because each one alone was wrong:

  * A static scan (word-boundary regex over all 61 suite files) said every one
    of the 76 wire methods was "touched" -- 56 of the `never_on_the_wire` set
    appeared in at least one suite file.
  * A dynamic census (wrapping all 76 `Service` methods, then running every
    suite in-process) said 64 of 76 were really called, 371 calls in total.

The nine-method gap between those two counts is the whole point: being named in
a file is not being executed. Reading the actual lines closed it:

  * `events` -- four mentions, all prose. `chk_home_adoption.py:108` is a
    comment; `:119` asserts about a fixture's payload, not about `svc.events`;
    `chk_tables.py:7` lists table names in a docstring. No call anywhere.
  * `evolve_tool` -- two mentions: a docstring, and one check that the string
    "evolve_tool" appears inside a refusal message.
  * `run_python` -- four suites, but every mention is the TOOL name
    `autoforge_run_python` (exposure lists, scope table), never the method.
  * `my_capabilities` -- one mention, in a docstring.
  * `tool_list` -- zero mentions of any kind.

Three static instruments were wrong along the way, each in a different shape,
and knowing them is why this suite is written the way it is: `.name(`
misses a bare reference (`answered(svc.find_gaps)`), a regex looking for a
`{"method": "x"}` string misses a Python dict literal (it is two `ast.Constant`
nodes, never one string), and a suite-local `call("name", args)` helper hides
the name inside a positional argument. The dynamic census had its own defect:
a plain wrapper rewrites `inspect.signature`, which made three suites fail for
a reason that had nothing to do with the product. It survives only with
`functools.wraps`.

  what this suite CAN check

    * `tool_list` names every registered tool, and its rows carry the same
      state string the registry reports -- measured against `schemas()`,
      because a listing that disagrees with the thing it lists is a second
      opinion nobody asked for;
    * `events(n)` really is "the newest n, oldest last": ascending ids, and the
      highest id is the ledger's highest -- the ordering claim is checkable
      without trusting the label;
    * a limit of 0 asks for nothing and gets nothing (the honest empty answer,
      not "here is everything");
    * `my_capabilities` answers three keys and does NOT restate what the
      per-turn brief already says -- zero shared lines, measured, because the
      duplication this split exists to remove was four lines shared verbatim;
    * `run_python` runs code in a child process and brings the return value
      back, refuses empty code before starting anything, refuses a non-JSON
      `args` string, and refuses an entry that is not a plain identifier;
    * `evolve_tool` refuses a missing tool by name, and its refusal is a
      refusal (no `improved: true` sitting beside it).

  what it does NOT claim

    * that `evolve_tool` breeds better on a real gateway. The success path IS
      checked, but against a scripted model injected into `_pipeline`: that
      proves the Service, the engine, the verifier and the registry agree, and
      it proves nothing about whether this machine's gateway model writes good
      mutants. (Measured 2026-11-11: the path had never been executed at all,
      and executing it once found that `evolution._to_spec` dropped
      `sample_call`/`sample_expect`, which made `improved` structurally
      unreachable whatever the breeder wrote.)
    * that the live self-test probe is exhaustive. `probe=True` is checked --
      it must add a `probe` block, and the block must confirm itself against the
      parent's own view of the file the child wrote. What is NOT checked is
      whether every network claim in that block is true on some other machine;
      the numbers it reports are this machine's, read back on this machine.
      (2026-11-07: this used to say `chk_env.py` owns that probe. It does not --
      `grep my_capabilities chk_env.py` was zero lines. A pointer to a suite
      that never mentions the subject is worse than no pointer, because it
      reads as coverage.)
    * that `run_python`'s sandbox limits hold under load. This checks the
      round trip and four refusals, not memory ceilings or the 30 s wall.

Run from the service directory:
    python tests/regression/chk_five.py

Isolation: the home is set before `Service` is imported, and the first
assertion asks the run journal where it lives rather than assuming -- a
`db_path` redirects the database and nothing else.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

_HOME = os.environ.get("AUTOFORGE_HOME")
if not _HOME:
    _HOME = tempfile.mkdtemp(prefix="m235_five_home_")
    os.environ["AUTOFORGE_HOME"] = _HOME

from autoforge_svc.service import Service                        # noqa: E402
from autoforge_svc import runs as runs_mod                       # noqa: E402
from autoforge_svc.forge import sandbox as sb
from autoforge_svc.forge import evolution as evo_mod                    # noqa: E402

FAILS: list[str] = []

BODY = ("def {name}(**kwargs):\n"
        "    return {{'n': kwargs.get('n', 7)}}\n")
PARAMS = {"type": "object", "properties": {}, "required": []}


def check(label: str, cond: bool, detail: object = "") -> None:
    print(("  OK   " if cond else "  FAIL ") + label
          + (("  <- " + repr(detail)) if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


def unverified(label: str, why: str) -> None:
    """Printed, never counted: a skip that looked like OK would be a lie."""
    print("  UNVERIFIED " + label + "  (" + why + ")")


def answered(fn, *args, **kwargs):
    """Call the product; an exception becomes the answer it did not give."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:                                   # noqa: BLE001
        return "<raised %s: %s>" % (type(exc).__name__, exc)


# ---- the breed fixture ------------------------------------------------
#
# `evolve_tool` rebuilds an `EvolutionEngine` over `pipeline.verifier.llm` on
# every call, so pointing `svc._pipeline` at a pipeline over `MockLLMClient`
# scripts the whole breed step: the mutate call, the validity gate's own model
# calls if any, and every check the verifier runs. Measured 2026-11-11: the
# first end-to-end breed took 9.2 s and eleven `chat()` calls, all enumerable,
# zero network -- the same trick `chk_core.py` uses for a forge.
#
# What the mutant must carry, and why it is not decoration: `sample_call` and
# `sample_expect` decide `execution`, and `probes` decides `trigger`. A check
# that never runs is `unprobed`, an unprobed check is `passed=False`, and
# `verify()` requires every check -- so a mutant without them can never be
# `improved`, however good its code. That is not a theory: it was the defect.

_BREED_ORIGINAL = ("def breed_probe(word: str = \"\"):\n"
                   "    return None  # ORIGINAL: does not count anything\n")

_BREED_MARKER = "# MUTANT:"


def _breed_envelope() -> str:
    """The breeder's answer: one FIX-N block whose code carries the marker."""
    code = ("def breed_probe(word: str = \"\"):\n"
            "    " + _BREED_MARKER + " counts vowels\n"
            "    return sum(1 for ch in str(word).lower() if ch in 'aeiou')\n")
    return "FIX-1\n" + json.dumps({
        "name": "breed_probe",
        "description": "Count the vowels in a word.",
        "code": code,
        "entry": "breed_probe",
        "parameters": {"type": "object",
                       "properties": {"word": {"type": "string"}},
                       "required": ["word"]},
        "sample_call": {"word": "banana"},
        "sample_expect": "3",
        "effect_signature": "pure",
        "probes": [{"query": "How many vowels are in 'banana'?",
                    "expect": "call",
                    "negative_query": "What is the square root of 5?"}],
        "rationale": "the original returned None instead of counting",
        "tags": ["text"],
    })


def _breed_handler(messages, tools=None, **_kw):
    """Answer by role: breeder, or the agent a verifier builds."""
    from autoforge_svc.core.llm import LLMResponse, tool_call as make_call
    system = messages[0].content if messages else ""
    if tools is not None:
        last = messages[-1].content if messages else ""
        if "square root" in last:
            return LLMResponse(content="not my job")
        offered = [(t.get("function") or {}).get("name")
                   for t in (tools or [])
                   if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        offered = [n for n in offered if isinstance(n, str) and n]
        return LLMResponse(content="",
                           tool_calls=[make_call(offered[0] if offered else
                                                 "breed_probe",
                                                 {"word": "banana"})])
    if "tool breeder" in system:
        return LLMResponse(content=_breed_envelope())
    return LLMResponse(content="{}")


def _breed_forge(svc) -> object:
    """Point `svc` at a pipeline over the scripted breeder; return the model."""
    from autoforge_svc.autonomy.policy import AutonomyPolicy
    from autoforge_svc.core.llm import MockLLMClient
    from autoforge_svc.forge.generator import LLMToolGenerator
    from autoforge_svc.forge.pipeline import ForgeConfig, ForgePipeline
    from autoforge_svc.forge.verifier import ToolVerifier

    llm = MockLLMClient(handler=_breed_handler)
    verifier = ToolVerifier(llm, sandbox=svc.sandbox)
    svc._pipeline = ForgePipeline(
        LLMToolGenerator(llm), verifier, svc.registry,
        config=ForgeConfig(), sandbox=svc.sandbox,
        policy=AutonomyPolicy(), on_event=lambda k, p: None,
    )
    svc._metacog = None
    return llm


def fresh(tmp: str, name: str) -> Service:
    return Service(db_path=os.path.join(tmp, name + ".db"))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m235_five_")
    svc = fresh(tmp, "main")
    try:
        journal = str(runs_mod.journal_dir())
        check("the run journal is the one this suite created, not the live one",
              journal.lower().startswith(_HOME.lower()), journal)

        # ---- tool_list ----------------------------------------------------
        print("-- tool_list names what the registry holds")
        empty = answered(svc.tool_list)
        # Asked before anything is defined, so an empty answer is the honest
        # one -- and the type assertion keeps a dict from passing as a list.
        check("an empty library lists nothing at all",
              isinstance(empty, list) and empty == [], empty)

        svc.tool_define(name="five_probe", description="a probe",
                        parameters=PARAMS, code=BODY.format(name="five_probe"))
        listed = answered(svc.tool_list)
        names_listed = [r.get("name") for r in listed] if isinstance(listed, list) else None
        check("a defined tool appears in the listing",
              names_listed == ["five_probe"], names_listed)
        row = listed[0] if isinstance(listed, list) and listed else {}
        check("each row carries a name, a description and a state",
              set(row) == {"name", "description", "state"}, sorted(row))
        check("the description is the one that was given",
              row.get("description") == "a probe", row.get("description"))
        # The listing and the exposure projection are two readers of one
        # registry. If they disagree, one of them is answering about a
        # different library.
        #
        # `schemas()` answers in the OpenAI function-calling shape --
        # `{"type": "function", "function": {"name": ...}}`, measured
        # 2026-11-07 -- so the name is one level down. Reading `.get("name")`
        # off the top returned `None` and made this assertion fail against
        # healthy code; the nesting is the contract, not a detail.
        schemas = [s.get("function", {}).get("name") for s in svc.registry.schemas()]
        check("...and the listing agrees with what the registry would send",
              "five_probe" in schemas, schemas)
        check("the state string matches the registry's own value",
              row.get("state") == str(getattr(svc.registry.get("five_probe"), "state").value),
              row.get("state"))

        # ---- events -------------------------------------------------------
        print("-- events(n) is the newest n, oldest last")
        many = answered(svc.events, 3)
        check("a limit of 0 asks for nothing and gets nothing",
              answered(svc.events, 0) == [], answered(svc.events, 0))
        check("events answers a list", isinstance(many, list), type(many).__name__)
        ids = [e.get("id") for e in many if isinstance(e, dict)]
        check("the rows come back oldest-last, so ids ascend",
              ids == sorted(ids), ids)
        # The ordering claim is worth nothing unless the page is the top of the
        # ledger. Compare against the maximum, not against another page.
        top = answered(svc.store.get_events, 1)
        top_id = top[-1]["id"] if isinstance(top, list) and top else None
        check("the newest row in that page is the ledger's newest row",
              bool(ids) and ids[-1] == top_id, (ids, top_id))
        check("...and a smaller limit returns no more than it was asked for",
              len(answered(svc.events, 2)) == 2, answered(svc.events, 2))
        row0 = many[0] if many else {}
        check("each event carries id, timestamp, kind and its payload",
              {"id", "timestamp", "kind"} <= set(row0), sorted(row0))
        check("the payload keys are flattened onto the row, not nested",
              "payload" not in row0, sorted(row0))

        # ---- my_capabilities ----------------------------------------------
        print("-- my_capabilities reports the specifics, not the brief again")
        caps = answered(svc.my_capabilities, False)
        check("it answers a dict", isinstance(caps, dict), type(caps).__name__)
        check("...with exactly text, reach and store",
              isinstance(caps, dict) and set(caps) == {"text", "reach", "store"},
              sorted(caps) if isinstance(caps, dict) else caps)
        text = caps.get("text") if isinstance(caps, dict) else None
        check("the text is prose with lines, not a one-liner",
              isinstance(text, str) and len(text.splitlines()) >= 8,
              len(text.splitlines()) if isinstance(text, str) else text)
        check("...and it names the database it is counting",
              isinstance(text, str) and str(caps.get("store", {}).get("db_path")) in text,
              "db_path not in text")
        # The split this block exists for: it must not repeat the brief. One
        # shared line out of twenty is how a reader learns to skim both.
        brief = answered(svc.context_brief)
        brief_text = brief.get("text") if isinstance(brief, dict) else ""
        if isinstance(brief_text, str) and isinstance(text, str):
            shared = sorted(set(x.strip() for x in brief_text.splitlines() if x.strip())
                            & set(x.strip() for x in text.splitlines() if x.strip()))
            check("it shares no line with the per-turn brief", shared == [], shared)
        else:
            check("it shares no line with the per-turn brief", False,
                  (type(brief_text).__name__, type(text).__name__))
        # probe=False is a table lookup; probe=True actually dials and writes.
        # The contract is the block: absent when it was not asked for, present
        # and self-confirming when it was.
        reach_off = caps.get("reach") if isinstance(caps, dict) else None
        check("the unprobed table carries no probe block",
              isinstance(reach_off, dict) and "probe" not in reach_off,
              sorted(reach_off) if isinstance(reach_off, dict) else reach_off)
        probed = answered(svc.my_capabilities, True)
        check("probe=True still answers the same three keys",
              isinstance(probed, dict) and set(probed) == {"text", "reach", "store"},
              sorted(probed) if isinstance(probed, dict) else probed)
        reach_on = probed.get("reach") if isinstance(probed, dict) else None
        block = reach_on.get("probe") if isinstance(reach_on, dict) else None
        check("...and it adds a probe block the unprobed answer did not have",
              isinstance(block, dict), type(block).__name__)
        check("...with the six keys a reader would need to judge it",
              isinstance(block, dict) and set(block) == {
                  "host_filesystem", "dns_intercepted", "dns", "network",
                  "host_filesystem_confirmed_by_parent",
                  "host_filesystem_seen_at"},
              sorted(block) if isinstance(block, dict) else block)
        # The one claim in that block that can be checked without trusting it:
        # the child says where it wrote; the parent says whether it is there.
        check("the filesystem claim is confirmed by the parent, not by itself",
              isinstance(block, dict)
              and block.get("host_filesystem_confirmed_by_parent") is True,
              block.get("host_filesystem_confirmed_by_parent") if isinstance(block, dict) else block)
        check("...and the parent saw it at the path the child named",
              isinstance(block, dict)
              and str((block.get("host_filesystem") or {}).get("detail"))
              == str(block.get("host_filesystem_seen_at")),
              (str((block.get("host_filesystem") or {}).get("detail")),
               str(block.get("host_filesystem_seen_at"))) if isinstance(block, dict) else block)
        for label, key in (("dns", "dns"), ("tls", "network")):
            part = block.get(key) if isinstance(block, dict) else None
            check("the %s claim carries an evidence word, not just ok" % label,
                  isinstance(part, dict) and isinstance(part.get("evidence"), str)
                  and bool(part.get("evidence")),
                  part if not isinstance(part, dict) else part.get("evidence"))
        dns_part = block.get("dns") if isinstance(block, dict) else None
        check("...and when the dns claim says resolved it names what it resolved",
              isinstance(dns_part, dict)
              and dns_part.get("evidence") in ("resolved", "intercepted")
              and (dns_part.get("evidence") != "resolved" or bool(dns_part.get("detail"))),
              dns_part)

        # The confirmation step is only worth having if it can DISAGREE with the
        # child, and on this host it never does: both processes resolve the same
        # temp dir, so "the path the child named" and "the path I would have
        # chosen" are the same string. Two mutants survived here for exactly
        # that reason -- one hard-coded confirmed=True, one reported its own
        # path instead of what it found. Neither changed any observable, because
        # there was nothing to observe. The three cases below construct the
        # disagreement instead of waiting for a machine that has one.
        print("-- the parent confirmation can disagree with the child")
        child = os.path.join("C:", os.sep, "child-temp", ".autoforge_reach_probe")
        parent_dir = os.path.join("C:", os.sep, "parent-temp")

        agree = sb.Sandbox._confirm_write(child, parent_dir, ".autoforge_reach_probe",
                                          exists=lambda p: p == child)
        check("when both see the child's file, the child's path is what was found",
              agree["found"] == child, agree)
        check("...and the write counts as confirmed", agree["confirmed"] is True, agree)

        child_only = sb.Sandbox._confirm_write(child, parent_dir, ".autoforge_reach_probe",
                                               exists=lambda p: p == child)
        check("a child that wrote somewhere this process cannot reach is still found",
              child_only["found"] == child, child_only)

        parent_only = sb.Sandbox._confirm_write(child, parent_dir, ".autoforge_reach_probe",
                                                exists=lambda p: p != child)
        check("...and when only the parent's own path exists, that is what gets reported",
              parent_only["found"] != child and parent_only["found"] is not None,
              parent_only)
        check("...and it is named as the parent's, not silently called the child's",
              str(parent_only["found"]).startswith(parent_dir), parent_only)

        nowhere = sb.Sandbox._confirm_write(child, parent_dir, ".autoforge_reach_probe",
                                           exists=lambda p: False)
        check("when neither path exists the write is NOT confirmed", nowhere["confirmed"] is False,
              nowhere)
        check("...and nothing is reported as the place it was seen", nowhere["found"] is None,
              nowhere)

        both = sb.Sandbox._confirm_write(child, parent_dir, ".autoforge_reach_probe",
                                        exists=lambda p: True)
        check("two candidate paths are always considered, in that order",
              both["candidates"] == [child, os.path.join(parent_dir, ".autoforge_reach_probe")],
              both)
        check("...so a parent path identical to the child's is not listed twice",
              sb.Sandbox._confirm_write(child, os.path.dirname(child), ".autoforge_reach_probe",
                                        exists=lambda p: False)["candidates"] == [child],
              sb.Sandbox._confirm_write(child, os.path.dirname(child), ".autoforge_reach_probe",
                                        exists=lambda p: False))
        check("...and a child that named nothing still leaves one path to look at",
              sb.Sandbox._confirm_write(None, parent_dir, ".autoforge_reach_probe",
                                        exists=lambda p: False)["candidates"]
              == [os.path.join(parent_dir, ".autoforge_reach_probe")],
              sb.Sandbox._confirm_write(None, parent_dir, ".autoforge_reach_probe",
                                        exists=lambda p: False))

        # ---- run_python ---------------------------------------------------
        print("-- run_python runs code and brings the value back")
        ok = answered(svc.run_python,
                      "def main(**kw):\n    return {'n': kw.get('n', 0) + 1}\n",
                      args={"n": 41})
        check("a plain round trip succeeds", ok.get("ok") is True, ok)
        check("...and the returned value crosses the process boundary intact",
              ok.get("output") == {"n": 42}, ok.get("output"))
        check("...with the four keys that say how it went",
              {"ok", "output", "stdout", "stderr", "error", "timed_out",
               "duration_ms"} <= set(ok), sorted(ok))
        check("nothing was written to stderr on a clean run",
              ok.get("stderr") == "", ok.get("stderr"))
        check("...and it did not time out", ok.get("timed_out") is False, ok.get("timed_out"))
        # Refusals, each before any work starts.
        empty_code = answered(svc.run_python, "")
        check("empty code is refused at the door",
              empty_code.get("ok") is False and "empty" in str(empty_code.get("error")),
              empty_code)
        bad_args = answered(svc.run_python,
                            "def main(**kw):\n    return 1\n", args="{not json")
        check("a non-JSON args string is refused, not guessed at",
              bad_args.get("ok") is False and "JSON" in str(bad_args.get("error")),
              bad_args)
        not_object = answered(svc.run_python,
                              "def main(**kw):\n    return 1\n", args="[1, 2]")
        check("...and a JSON array is refused too, since args are named",
              not_object.get("ok") is False and "object" in str(not_object.get("error")),
              not_object)
        bad_entry = answered(svc.run_python,
                             "def main(**kw):\n    return 1\n", entry="has space")
        check("an entry that is not a plain identifier is refused",
              bad_entry.get("ok") is False and "identifier" in str(bad_entry.get("error")),
              bad_entry)
        missing_entry = answered(svc.run_python,
                                 "def something_else(**kw):\n    return 1\n",
                                 entry="main")
        check("code that does not define the entry fails instead of pretending",
              missing_entry.get("ok") is False, missing_entry)

        # ---- evolve_tool --------------------------------------------------
        print("-- evolve_tool refuses what it cannot breed")
        gone = answered(svc.evolve_tool, "no_such_tool_here")
        check("a missing tool is refused by name", gone.get("ok") is False, gone)
        check("...and the refusal names the tool that was asked for",
              "no_such_tool_here" in str(gone.get("text")), gone.get("text"))
        check("...without claiming anything improved beside it",
              "improved" not in gone or gone.get("improved") is False, sorted(gone))
        # A tool that exists but cannot be bred. Measured 2026-11-07: this took
        # 40.3 s (the engine really asks a model) and used to answer "no
        # admissible mutant beat the original (0 tried)" -- a verdict on a
        # comparison in which nothing was compared. The sentence has to say
        # which of the two happened, so the assertion is about that distinction
        # rather than about the wording of a refusal.
        # The breed step asks a model and can fail outright; measured 2026-11-07,
        # the same call answered both ways across ten runs of this file. Whatever it
        # answers, the detail below carries the whole dict so a failure names its cause
        # instead of only repeating the sentence the caller already saw.
        # What the breed step does with a model is a question about a model,
        # measured twice on 2026-11-07: ten standalone runs of this file gave
        # zero failures once and three failures another time, and the same call
        # inside the mutation driver once exceeded its 300 s per-suite budget.
        # A suite about return shapes cannot hold a language model still, so the
        # fix it found is asserted directly: the parser refuses a non-string
        # field by name, and the service answers `tried: None` when the breed
        # step itself fails rather than reporting a count it never measured.
        #
        # The defect these replace (kept here so the reason survives):
        #   FAIL ...and the answer says how many candidates there were
        #     <- {"ok": false, "text": "Could not evolve 'five_probe'."}
        #     and underneath it, AttributeError: 'dict' object has no
        #     attribute 'strip' raised by evolution._parse_population.
        text_field = evo_mod._text_field
        # Every call goes through `answered`. A mutation that makes `_text_field`
        # raise -- a plain `.strip()` on None, say -- used to abort the suite
        # before it printed a verdict, and a crash is not a caught mutant: the
        # driver cannot tell it from a broken harness. Asking the same questions
        # without ever letting an exception escape keeps the report a report.
        check("a model answering a field with a string still gets it trimmed",
              answered(text_field, {"code": "  x = 1  "}, "code") == "x = 1",
              answered(text_field, {"code": "  x = 1  "}, "code"))
        check("an omitted field takes the default rather than raising",
              answered(text_field, {}, "code", "fallback") == "fallback",
              answered(text_field, {}, "code", "fallback"))
        check("an explicit null takes the default too",
              answered(text_field, {"code": None}, "code", "fallback") == "fallback",
              answered(text_field, {"code": None}, "code", "fallback"))
        # `_text_field` raises on purpose: a caller that stored a dict as a
        # tool name would be worse than one that fails loudly. So the assertion
        # is about the message an operator would read, not about a return value.
        bad = answered(text_field, {"name": {"role": "x"}}, "name")
        check("a field answered with an object is refused, not stringified",
              "with dict" in str(bad), bad)
        check("...and the refusal names the field it read",
              "'name'" in str(bad), bad)
        check("...and it does not pretend the model said nothing",
              "instead of a string" in str(bad), bad)
        number = answered(text_field, {"code": 7}, "code")
        check("a number is refused the same way, by type",
              "with int" in str(number), number)
        # The answer shape a failed breed owes a caller: a count it did not
        # measure is None, and None is not 0. A pipeline with no `evolution`
        # attribute stands in for the breed step failing, which is exactly what
        # the measured AttributeError did.
        # `tmp` is a str in this suite (`tempfile.mkdtemp`), not a Path, so the
        # join has to be os.path.join; `tmp / "m268.db"` raised
        # TypeError: unsupported operand type(s) for /: 'str' and 'str' on every
        # run, and rc=1 with zero FAIL lines is exactly the shape a reader
        # cannot tell from a real failure.
        quiet = Service(db_path=os.path.join(tmp, "m268.db"))
        # The breed step fails here for a constructed reason: a pipeline with no
        # `evolution` gives an AttributeError. The assertions below ask whether
        # the *answer* admits it measured nothing. Written against a tool that
        # really exists, because `registry.get("five_probe")` is answered from
        # the other Service instance's library and this one is empty.
        own = quiet.tool_define(
            name="breed_probe",
            code="def breed_probe(n=1):\n    return {'n': n}\n",
            description="a tool to fail to breed",
            parameters={"type": "object", "properties": {"n": {"type": "integer"}}},
        )
        try:
            if not own.get("registered"):
                unverified("a named breed tool for the failure shape",
                           "tool_define did not register %r: %s"
                           % ("breed_probe", own.get("error")))
            else:
                quiet._pipeline = object()
                failed = answered(quiet.evolve_tool, "breed_probe", "irrelevant")
                check("a breed failure answers a dict, not an exception",
                      isinstance(failed, dict), failed)
                check("...with tried present and None, because nothing was counted",
                      failed.get("tried", "missing") is None, failed)
                check("...and it is not reported as zero candidates",
                      failed.get("tried") != 0, failed)
                check("...and the text does not claim a comparison happened",
                      "beat the original" not in str(failed.get("text", "")), failed)
        finally:
            try:
                quiet.store.close()
            except Exception:                                  # noqa: BLE001
                pass
        # ---- the success path, which had never run ----
        #
        # `evolve_tool` builds an `EvolutionEngine` over `pipeline.verifier.llm`
        # every call, so replacing `svc._pipeline` scripts the whole breed step
        # the same way `chk_core.py` scripts a forge: no gateway, no 90 s probe,
        # no network. Written because the three failure shapes were the only
        # thing ever executed here and one of them hid a real defect.
        print("-- evolve_tool breeds a better version when there is one")
        breed = Service(db_path=os.path.join(tmp, "breed.db"))
        try:
            seeded = breed.tool_define(
                name="breed_probe",
                code=_BREED_ORIGINAL,
                description="Count the vowels in a word.",
                parameters={"type": "object",
                            "properties": {"word": {"type": "string"}},
                            "required": ["word"]},
            )
            if not seeded.get("registered"):
                unverified("evolve_tool breeds a better version",
                           "could not seed the tool: %s" % seeded.get("error"))
            else:
                llm = _breed_forge(breed)
                evolved = answered(breed.evolve_tool, "breed_probe", "returns None")
                check("a bred version answers ok", evolved.get("ok") is True, evolved)
                check("...and says it improved", evolved.get("improved") is True,
                      evolved)
                check("...naming the tool", evolved.get("name") == "breed_probe",
                      evolved)
                check("...with a fitness number, not a string",
                      isinstance(evolved.get("fitness"), float), evolved)
                check("...and the text agrees it is active",
                      "new version active" in str(evolved.get("text", "")), evolved)
                check("the breed actually called the model",
                      len(llm.calls) >= 3, len(llm.calls))
                live = breed.registry.get("breed_probe")
                check("the registry holds the mutant, not the original",
                      live is not None and _BREED_MARKER in str(live.code), live)
                # `ToolStore` has `get_events` / `event_kinds`, not `events`
                # -- that name belongs to the Service method. Writing
                # `breed.store.events(40)` raised AttributeError after seven
                # green assertions, which is rc=1 with no FAIL line: the shape a
                # reader cannot tell from a real failure.
                kinds = [row.get("kind") for row in breed.store.get_events(40)]
                check("...and the ledger records the evolve",
                      "evolve" in kinds, kinds[-8:])
        finally:
            try:
                breed.store.close()
            except Exception:                              # noqa: BLE001
                pass

        # ---- the five are on the wire, which is where they were before ----
        print("-- all five are reachable over the wire, not only in process")
        for name in ("tool_list", "events", "my_capabilities", "run_python",
                     "evolve_tool"):
            check("the wire table knows " + name, name in svc.methods,
                  sorted(svc.methods)[:5])
    finally:
        try:
            svc.store.close()
        except Exception:                                      # noqa: BLE001
            pass

    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
