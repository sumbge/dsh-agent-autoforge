"""The model id both model-using paths actually send.

This service had a constant: `xdpool-global/deepseek-v4.1-flash`. It was wrong
on the machine it shipped to. Measured 2026-11-04 against the live gateway:

    GET  /v1/models              -> 35 ids; `xdpool-global/deepseek-v4.1-flash`
                                    is NOT among them (`-sg` is)
    POST /chat/completions       -> HTTP 404 "model ... could not be resolved:
                                    pi-ai provider \\"xdpool-global\\" has no
                                    configured model \\"deepseek-v4.1-flash\\""

Both callers of `_forge_route()` -- `_forge_pipeline` and `_run_task` -- were
dead on arrival. `forge_tool` would have failed at its first model call, and
every scheduled task failed the same way. Nothing caught it because the two
suites that exercise those paths supplied their own route, and one of them
(`chk_run_journal.py`) supplied it in a way that had no effect at all.

What is asserted here:

  * the resolved id is not a constant in this file -- it is read from a profile
    that actually configures one, and $AUTOFORGE_MODEL outranks it;
  * the resolution names the file it read, so a wrong answer is traceable;
  * nothing is guessed when nothing is configured: the callers raise, naming
    where they looked, instead of sending a plausible id and failing later;
  * WHEN a gateway answers, the resolved id is one it serves. Skipped, with a
    printed reason, when no gateway answers -- a skip that says why, not a pass
    that was never earned.

Run:
  $env:PYTHONPATH = "...\\dsh-agent-autoforge\\service"
  & $py -P ...\\chk_route.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICE = os.path.dirname(os.path.dirname(HERE))   # regression -> tests -> service
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc import service as S             # noqa: E402
from autoforge_svc.autonomy.scopes import scope_for  # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    """`detail` is `object`, not `str`.

    Same lesson the review suite already paid for: a helper that can only
    format a detail on the failing path is a helper that hides failures. Here a
    dict is the natural detail, and `str()` of it is exactly what a reader
    wants when something goes wrong.
    """
    if cond:
        print("  OK   " + label)
    else:
        print("  FAIL " + label + (" <- " + str(detail)[:300] if detail else ""))
        FAILS.append(label)


def fake_home(tmp: str, entries: dict[str, str]) -> str:
    """A $DSH_HOME whose profiles/*/cordis.patch.yml hold the given text."""
    for name, body in entries.items():
        folder = os.path.join(tmp, "profiles", name)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "cordis.patch.yml"), "w", encoding="utf-8") as fh:
            fh.write(body)
    return tmp


def with_env(env: dict[str, str]) -> None:
    for key, value in env.items():
        os.environ[key] = value


def scratch_env() -> None:
    """A clean slate: no override, no inherited DSH_HOME."""
    for key in ("AUTOFORGE_MODEL", "AUTOFORGE_BASE_URL", "AUTOFORGE_API_KEY",
                "DSH_HOME", "DSH_OPENAI_GATEWAY_API_KEY"):
        os.environ.pop(key, None)


def default_entry(model: str, provider: str = "xdpool") -> str:
    return (
        "- id: agent-default-model\n"
        "  name: \"@deepseek-ai/dsh-agent-default-model\"\n"
        "  config:\n"
        "    provider: %s\n"
        "    model: %s\n"
        "    reasoningEffort: high\n" % (provider, model)
    )


def main() -> int:
    real_dsh_home = S._dsh_home

    print("-- the old constant is gone from the source, not just from the answer")
    src = open(os.path.join(SERVICE, "autoforge_svc", "service.py"), encoding="utf-8").read()
    # A grep for the literal would also hit this file's docstring, so the
    # assertion is on the *assignment* form: a default value cannot hide behind
    # a comment or an example.
    check("no assignment in service.py defaults the model to the old id",
          not re.search(r'"xdpool-global/deepseek-v4\.1-flash"', src),
          re.findall(r'.{60}xdpool-global.{60}', src))
    check("_forge_route consults the DSH profile rather than a constant",
          "_dsh_model()" in src.split("def _forge_route")[1][:600],
          src.split("def _forge_route")[1][:300])

    print("-- the profile is read, and the answer says which file it came from")
    tmp = tempfile.mkdtemp(prefix="m30_route_")
    scratch_env()
    os.environ["DSH_HOME"] = fake_home(tmp, {"desktop": default_entry("a-flash")})
    S._dsh_home = lambda: tmp                                # type: ignore[assignment]
    model, where = S._dsh_model()
    check("a provider/model pair is assembled from two separate fields",
          model == "xdpool/a-flash", model)
    check("and the answer names the file it read", where.endswith("cordis.patch.yml"), where)
    check("the file it names is the one under the profile it chose",
          os.path.basename(os.path.dirname(where)) == "desktop", where)

    print("-- a model that already carries a provider is taken as written")
    tmp2 = tempfile.mkdtemp(prefix="m30_route2_")
    os.environ["DSH_HOME"] = fake_home(tmp2, {"web": default_entry("trae/deepseek-v4.1-flash")})
    S._dsh_home = lambda: tmp2                               # type: ignore[assignment]
    check("a slashed id is not prefixed a second time",
          S._dsh_model()[0] == "trae/deepseek-v4.1-flash", S._dsh_model())

    print("-- the enabled-model lists are the second source, not the first")
    tmp3 = tempfile.mkdtemp(prefix="m30_route3_")
    os.environ["DSH_HOME"] = fake_home(tmp3, {"acp": (
        "- id: llm-xdpool\n"
        "  name: dsh-xdpool\n"
        "  config:\n"
        "    modelSelectionCn:\n"
        "      enabledModelIds:\n"
        "        - deepseek-v4.1-flash\n"
    )})
    S._dsh_home = lambda: tmp3                               # type: ignore[assignment]
    check("with no agent-default-model, the enabled list still answers",
          S._dsh_model()[0] == "deepseek-v4.1-flash", S._dsh_model())

    print("-- YAML DSH legitimately writes does not make the file unreadable")
    tmp4 = tempfile.mkdtemp(prefix="m30_route4_")
    # `!!js` is real in these files; a plain safe_load refuses the whole
    # document over it, and the error names the tag rather than the setting.
    os.environ["DSH_HOME"] = fake_home(tmp4, {"desktop": (
        "- id: someone-elses-plugin\n"
        "  name: x\n"
        "  config:\n"
        "    when: !!js (() => 'later')\n"
        "- id: agent-default-model\n"
        "  config:\n"
        "    provider: buddy\n"
        "    model: kimi-k3-1\n"
    )})
    S._dsh_home = lambda: tmp4                               # type: ignore[assignment]
    check("a !!js tag elsewhere in the file does not hide the model",
          S._dsh_model()[0] == "buddy/kimi-k3-1", S._dsh_model())

    print("-- nothing configured is answered with nothing, not with a guess")
    tmp5 = tempfile.mkdtemp(prefix="m30_route5_")
    os.environ["DSH_HOME"] = tmp5
    S._dsh_home = lambda: tmp5                               # type: ignore[assignment]
    model5, where5 = S._dsh_model()
    check("no profile -> empty model", model5 == "", model5)
    check("and the reason names where it looked",
          "no profile configured a model" in where5, where5)

    print("-- an explicit override outranks the profile")
    os.environ["AUTOFORGE_MODEL"] = "chosen/by-hand"
    routed = S._forge_route()
    check("$AUTOFORGE_MODEL wins", routed[0] == "chosen/by-hand", routed[0])
    del os.environ["AUTOFORGE_MODEL"]
    check("and removing it returns to the profile's answer",
          S._forge_route()[0] == "", S._forge_route()[0])

    print("-- both callers refuse in words a person can act on")
    tmp6 = tempfile.mkdtemp(prefix="m30_route6_")
    os.environ["DSH_HOME"] = fake_home(tmp6, {})
    S._dsh_home = lambda: tmp6                               # type: ignore[assignment]
    svc = S.Service(db_path=os.path.join(tmp6, "autoforge.db"))
    # A key so the "no credential" branch is not the one taken: the point is
    # that a missing MODEL is refused on its own terms.
    os.environ["AUTOFORGE_API_KEY"] = "key"
    try:
        svc._forge_pipeline()
        check("the forge pipeline refuses to start without a model", False, "it built one")
    except RuntimeError as exc:
        text = str(exc)
        check("the forge pipeline refuses to start without a model",
              "no model" in text, text)
        check("and its refusal says what to set and where to look",
              "AUTOFORGE_MODEL" in text and "/models" in text, text)
    except Exception as exc:                                 # noqa: BLE001
        check("the forge pipeline refuses to start without a model", False,
              f"{type(exc).__name__}: {exc}")
    ran = svc._call_method("schedule_run_due", {"budget_s": 5})
    # Nothing is due, so this must not reach the model at all -- and that is
    # itself the guard: a route check that fired on an idle tick would make
    # every quiet tick an error.
    check("an idle tick still costs nothing when no model is configured",
          ran.get("due") == 0, ran)
    svc.store.close()
    svc.missions.close()

    print("-- a model the gateway does not serve is never handed back as usable")
    os.environ.pop("AUTOFORGE_API_KEY", None)
    S._dsh_home = real_dsh_home                              # type: ignore[assignment]
    scratch_env()
    if os.path.exists(os.path.join(real_dsh_home(), "openai-gateway", "api-key")):
        model7, base7, key7 = S._forge_route()
        served = S._gateway_models(base7, key7)
        if served:
            # This block used to assert that `_forge_route()`'s answer is in
            # `served`, and it passed for months because the profile on this
            # machine happened to name a served model. On 2026-11-07 the profile
            # named `qodercn/qfmodel`, the gateway served seven other ids, and
            # that assertion went red -- correctly, but about the wrong thing:
            # `_forge_route` reports what is CONFIGURED, and replacing an
            # unusable id is the next function's job. Pinning the environment's
            # current taste is not a contract; the contract is that an id the
            # gateway does not serve never comes back as the model to use.
            used7, why7 = S._tool_capable_model(model7, base7, key7)
            if used7 in served:
                check("the model that will be used is one the gateway serves",
                      True, {"configured": model7, "used": used7, "why": why7})
            else:
                check("...or the route refused rather than using an unserved id",
                      why7 == "no alternative",
                      {"configured": model7, "used": used7, "why": why7})
            check("...and a model outside the list is never described as merely "
                  "unverified",
                  not (model7 not in served and why7 == "unverified"),
                  {"configured": model7, "served": len(served), "why": why7})
            check("and when it was replaced, the note says why",
                  "substituted" not in why7
                  or S._ROUTE_NOTE.get("reason") in
                  ("not served by the gateway", "cannot read tool results"),
                  S._ROUTE_NOTE)
        else:
            print("  SKIP no gateway answered at %s, so no claim is made about "
                  "which ids it serves" % base7)
    else:
        print("  SKIP no gateway key at %s, so the live check cannot run"
              % os.path.join(real_dsh_home(), "openai-gateway", "api-key"))

    print("-- and the network call it makes is on the egress ledger")
    eg = __import__("autoforge_svc.egress", fromlist=["SITES", "bind", "admit", "gateway"])
    check("_gateway_models is a declared egress site",
          any(s.get("func") == "_gateway_models" for s in eg.SITES))
    # Declaring the site is not routing through it. The first version of this
    # suite only read the SITES table, and a mutant that deleted the actual
    # `_admit_egress(...)` call escaped -- because the table said the site
    # existed while nothing consulted it. The check below asks the ledger, which
    # only fills if the call site really calls `admit()`.
    probe_dir = tempfile.mkdtemp(prefix="m30_route_egress_")
    store = S.ToolStore(db_path=os.path.join(probe_dir, "autoforge.db"))
    saved = eg.gateway()
    try:
        eg.bind(store.connection(), name="route_probe", allow=("127.0.0.1",), mode="closed")
        S._MODELS_CACHE.clear()
        got = S._gateway_models("http://not-allowed.example:1/v1", "k")
        check("a gateway the egress gate refuses yields no model list", got == [], got)
        rows = [json.loads(r[0]) for r in store.connection().execute(
            "SELECT payload FROM forge_events WHERE kind = 'egress'").fetchall()]
        mine = [r for r in rows if r.get("extra") == "service._gateway_models"]
        check("and the refusal reached the ledger under this site's own name",
              len(mine) == 1, {"rows": len(rows), "mine": mine[:2]})
        if mine:
            check("recorded as a denial", mine[0].get("allowed") is False, mine[0])
            check("naming the host it refused",
                  mine[0].get("resource") == "not-allowed.example", mine[0])
    finally:
        eg.reset()
        if saved.conn is not None:
            eg.bind(saved.conn, name=saved.name, allow=saved.allow, mode=saved.mode)
        store.close()
    check("its scope is the ordinary read/write one, not privileged",
          scope_for("report") in ("read_only", "local_write"), scope_for("report"))

    print("-- the report can answer 'is the configured id even served'")
    # `report()` must stay cheap, so it never pays for the /models call. But a
    # report that names a model without saying whether the gateway lists it
    # leaves the reader unable to tell "this will work" from "nothing will ever
    # answer for this id" -- measured 2026-11-07, when the profile named
    # `qodercn/qfmodel` and the gateway served seven other ids.
    rep_dir = tempfile.mkdtemp(prefix="m30_route_report_")
    rep_svc = S.Service(db_path=os.path.join(rep_dir, "autoforge.db"))
    try:
        S._MODELS_CACHE.clear()
        S._TOOL_READING.clear()
        cold = rep_svc.report().get("model") or {}
        check("with nothing asked yet, the served question is left unanswered",
              cold.get("served_by_gateway") is None, cold)
        check("...and the field exists rather than being absent",
              "served_by_gateway" in cold, sorted(cold))
        # Fill the cache the way a real probe would, rather than making the
        # call here: this asserts what `report()` says given an answer, and a
        # suite that needs a live gateway to run its own assertions is a suite
        # that reports on the network instead of on the code.
        S._MODELS_CACHE.clear()
        S._MODELS_CACHE.update({"stamp": ("http://x/v1", 1), "at": 1.0,
                                "models": ["served/one", "served/two"]})
        S._ROUTE_NOTE.clear()
        warm = rep_svc.report().get("model") or {}
        check("after a probe, the report answers from the cache", 
              "served_by_gateway" in warm, warm)
        # The report reads the configured id, which on this machine is whatever
        # the profile names; the cache above deliberately contains neither of
        # those two ids being the configured one is not guaranteed, so assert
        # the field agrees with the list rather than asserting a fixed value.
        configured = warm.get("id")
        check("...and it agrees with the cached serving list",
              warm.get("served_by_gateway") ==
              (configured in ["served/one", "served/two"]),
              {"configured": configured, "field": warm.get("served_by_gateway")})
    finally:
        rep_svc.store.close()
        rep_svc.missions.close()
        S._MODELS_CACHE.clear()
        S._ROUTE_NOTE.clear()
        S._TOOL_READING.clear()

    print("-- an unserved id is replaced on that fact alone, with no live gateway")
    # The block above needs this machine's gateway to answer, so it can only
    # witness the state of the network. This one stubs both outbound helpers and
    # pins the decision table, which is the part that was wrong: measured
    # 2026-11-07, a profile naming `qodercn/qfmodel` while the gateway served
    # seven other ids produced `unverified` -- "could not be asked" -- and the
    # unusable id was handed back to every caller. The four rows below are the
    # four answers `_tool_capable_model` is allowed to give.
    real_models, real_read = S._gateway_models, S._reads_tool_results
    try:
        def stub(served_list, reads):
            S._gateway_models = lambda base, key, **_k: list(served_list)
            S._reads_tool_results = (
                lambda model, base, key, **_k: reads.get(model, None))

        # 1. the configured id is not served, and a served one works
        stub(["good/a", "good/b"], {"good/a": True})
        got = S._tool_capable_model("stale/x", "http://x/v1", "k")
        check("an id the gateway does not serve is not handed back",
              got[1] == "substituted: good/a" and got[0] == "good/a", got)
        check("...and the note says the reason was the serving list, not tool "
              "reading", S._ROUTE_NOTE.get("reason") == "not served by the gateway",
              S._ROUTE_NOTE)
        check("...and records that the configured id was not listed",
              S._ROUTE_NOTE.get("listed") is False, S._ROUTE_NOTE)
        # The note is what `_tool_read_note` reads out to an operator, and it
        # prints `configured` as "the configured model". A mutant that wrote the
        # substitute into that field was caught by nothing here until this line
        # existed -- every other field of the note stayed truthful, so the note
        # read "the configured model (good/a) ..." about an id nobody configured.
        check("...and still reports the model that was configured, not the one "
              "it chose", S._ROUTE_NOTE.get("configured") == "stale/x",
              S._ROUTE_NOTE)

        # 2. the configured id is not served and nothing served works
        stub(["blind/a"], {"blind/a": False})
        got = S._tool_capable_model("stale/x", "http://x/v1", "k")
        check("an unserved id with no working alternative is a refusal",
              got[1] == "no alternative", got)
        # The old wording claimed the model "cannot see tool results", which
        # asserts something about a model the gateway never ran.
        note2 = S._tool_read_note("A forge", "http://x/v1")
        check("...whose sentence names the serving list, not tool blindness",
              "is not one this gateway serves" in note2, note2)
        check("...and does not accuse it of failing a probe it never took",
              "cannot see tool results" not in note2, note2)

        # 3. it IS served and blind -- the old defect, still handled
        stub(["stale/x", "good/c"], {"stale/x": False, "good/c": True})
        got = S._tool_capable_model("stale/x", "http://x/v1", "k")
        check("a served but blind model is still replaced",
              got[1] == "substituted: good/c", got)
        check("...and that reason is tool reading, not the serving list",
              S._ROUTE_NOTE.get("reason") == "cannot read tool results",
              S._ROUTE_NOTE)

        # 4. it is served, blind, and nothing else works
        stub(["stale/x"], {"stale/x": False})
        S._tool_capable_model("stale/x", "http://x/v1", "k")
        note4 = S._tool_read_note("A forge", "http://x/v1")
        check("a served blind model keeps the tool-result sentence",
              "cannot see tool results" in note4, note4)

        # 5. the gateway list is empty -- "could not ask", not "nothing served"
        stub([], {})
        got = S._tool_capable_model("stale/x", "http://x/v1", "k")
        check("an unaskable gateway keeps the configured model as unverified",
              got == ("stale/x", "unverified"), got)
        check("...and does not claim the id is unserved",
              S._ROUTE_NOTE.get("reason") is None, S._ROUTE_NOTE)

        # 6. a working model the list omits is kept, with the disagreement noted
        stub(["other/y"], {"stale/x": True})
        got = S._tool_capable_model("stale/x", "http://x/v1", "k")
        check("a model that answered correctly is kept even when unlisted",
              got == ("stale/x", "configured"), got)
        check("...and the omission is recorded rather than corrected",
              S._ROUTE_NOTE.get("listed") is False, S._ROUTE_NOTE)
        # 6b. the retry is real, and the note says why the first ask failed.
        # Both of these lived only in the source until the mutants below were
        # written: a retry that does nothing and a note that forgets the cause
        # both leave every other assertion in this block green, because under a
        # stub the retry answers the same thing twice. So the stub counts.
        tries = []

        def flaky(served_list, first, second):
            def reader(model, base, key, **_k):
                tries.append(model)
                return first if len(tries) == 1 else second
            S._gateway_models = lambda base, key, **_k: list(served_list)
            S._reads_tool_results = reader

        tries.clear()
        flaky(["stale/x"], None, True)
        got = S._tool_capable_model("stale/x", "http://x/v1", "k")
        check("a probe that could not ask is asked a second time",
              len(tries) >= 2, tries)
        check("...and the second answer is the one that decides",
              got == ("stale/x", "configured"), got)

        tries.clear()
        flaky(["stale/x"], None, None)
        S._TOOL_READING_WHY[("stale/x", "http://x/v1")] = "URLError: refused"
        got = S._tool_capable_model("stale/x", "http://x/v1", "k")
        check("when asking keeps failing, the answer is unverified",
              got == ("stale/x", "unverified"), got)
        check("...and the note carries the reason asking failed",
              S._ROUTE_NOTE.get("probe") == "URLError: refused", S._ROUTE_NOTE)
        S._TOOL_READING_WHY.clear()


        # 6c. the reason is written by the failing call, not by the assertion.
        # Mutant P4 (drop `reason = ...`) escaped everything above, because every
        # assertion there wrote `_TOOL_READING_WHY` by hand -- so the line that
        # fills it was never on trial. This dials a port nothing listens on and
        # reads the record the probe made for itself.
        dead = ("probe/dead", "http://127.0.0.1:1/v1")
        S._TOOL_READING.clear()
        S._UNASKED_AT.clear()
        S._TOOL_READING_WHY.clear()
        refused = real_read(*dead, key="k", timeout=3.0)
        said = S._TOOL_READING_WHY.get(dead, "")
        check("a probe that cannot connect answers None", refused is None, refused)
        check("...and records what went wrong in its own words",
              bool(said) and said != "the gateway returned no answer", said)
        check("...naming the exception it hit",
              ("Error" in said or "Exception" in said), said)
        S._TOOL_READING.clear()
        S._UNASKED_AT.clear()
        S._TOOL_READING_WHY.clear()

        # 7. "could not ask" is an accident, and an accident is not a verdict.
        # Measured 2026-11-11: one probe of `xdpool/hy4free` ran to 6.7 s and
        # was remembered as None. Every caller afterwards got "unverified",
        # which this module treats as permission to run real work -- on a model
        # that answers every tool turn with "No result provided". The three
        # assertions below are what stops one slow call becoming a permanent
        # alibi; they need no gateway, because they are about the memory.
        # `real_read`, not `S._reads_tool_results`: the stub above is still
        # installed at this point, and it answers from a dict without ever
        # touching the cache these four lines are about. Calling through the
        # module name here measured the stub -- three assertions stayed red
        # until that was noticed.
        stamp = ("probe/cache", "http://probe/v1")
        S._TOOL_READING.clear()
        S._UNASKED_AT.clear()
        S._TOOL_READING_WHY.clear()
        S._TOOL_READING[stamp] = None
        S._UNASKED_AT[stamp] = time.time()
        S._TOOL_READING_WHY[stamp] = "synthetic failure"
        inside = real_read(*stamp, key="k", timeout=2.0)
        check("a failed probe is remembered inside the retry window",
              inside is None and S._TOOL_READING_WHY.get(stamp) ==
              "synthetic failure", inside)
        S._UNASKED_AT[stamp] = time.time() - (S._UNASKED_TTL_S + 1.0)
        after = real_read(*stamp, key="k", timeout=2.0)
        check("...and asked again once the window has passed",
              S._TOOL_READING_WHY.get(stamp) != "synthetic failure",
              S._TOOL_READING_WHY.get(stamp))
        S._TOOL_READING_WHY[stamp] = "synthetic failure"
        S._UNASKED_AT[stamp] = time.time()
        forced = real_read(*stamp, key="k", timeout=2.0, again=True)
        check("...and `again=True` re-asks even inside the window",
              S._TOOL_READING_WHY.get(stamp) != "synthetic failure", forced)
        S._TOOL_READING[stamp] = True
        S._UNASKED_AT[stamp] = time.time() - 10_000.0
        kept = real_read(*stamp, key="k", timeout=2.0)
        check("a real verdict is kept no matter how old", kept is True, kept)
    finally:
        S._gateway_models, S._reads_tool_results = real_models, real_read
        S._ROUTE_NOTE.clear()
        S._MODELS_CACHE.clear()
        S._TOOL_READING.clear()
        S._UNASKED_AT.clear()
        S._TOOL_READING_WHY.clear()

    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
