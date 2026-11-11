"""Does exposure classify by the rule, and does the rule hold on real shapes?

Under test: `Service.tool_exposure` and the `exposure` module it delegates to.
The bridge-side projection is covered by `chk_exposure.mjs`; this file covers the
decision -- which names are hot, what a cold line says, and the cost rule that
makes the whole thing worth doing.

Run from the service directory:
    python tests/regression/chk_exposure_service.py

The database is a throwaway under %TEMP%; the production one is never opened.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc.service import Service                      # noqa: E402
from autoforge_svc.tools import exposure as exposure_mod      # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    # `detail: object`, not `detail: str`: a failure path that formats its own
    # evidence must not be able to crash the suite. Measured 2026-11-06 -- a
    # mutant reached a check with a list and the suite died with
    # `TypeError: can only concatenate str (not "list") to str` before printing
    # any verdict, which the mutation driver correctly scores as "crashed",
    # not "caught". An assertion that cannot report its evidence is not there.
    print(("  OK   " if cond else "  FAIL ") + label
          + (("  <- " + repr(detail)) if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


def fat(prefix: str, n: int = 8) -> dict:
    """A descriptor whose full form is expensive enough that cold is worth it.

    Written out at length on purpose. A cold line costs about 200 characters, so
    a tool whose entire schema is smaller than that CANNOT be made cheaper by
    being described in prose -- and the rule under test is that such a tool stays
    hot. Fixtures with three short arguments were what made the first version of
    this test wrong: it demanded a projection that would have cost MORE.
    """
    return {f"{prefix}_{i}": {"type": "string", "required": True,
                              "description": f"the {prefix} argument number {i} used for this call"}
            for i in range(n)}


def cost_of(name: str, description: str, descriptor: dict) -> int:
    """Name + description + the parameter map as the provider receives it."""
    return len(name) + len(description) + len(json.dumps(descriptor, separators=(",", ":")))


def main() -> int:
    home = tempfile.mkdtemp(prefix="chkexp_")
    os.environ["AUTOFORGE_HOME"] = home
    svc = Service(db_path=os.path.join(home, "t.db"))
    try:
        big_desc = "Compile and time a kernel."
        big_params = fat("axis")
        small_desc = "Check that the service is reachable."
        small_params = {"why": {"type": "string"}}

        catalog = [
            {"name": "autoforge_ping", "description": small_desc, "parameters": small_params},
            {"name": "autoforge_run_python", "description": "Run a snippet.",
             "parameters": {"code": {"type": "string", "required": True}}},
            {"name": "autoforge_gpu_bench", "description": big_desc, "parameters": big_params},
            {"name": "autoforge_cpu_tune", "description": big_desc, "parameters": fat("knob")},
        ]

        print("== 1. no catalog yet ==")
        v = svc.tool_exposure(refresh=True)
        check("source is registry-only", v["source"] == "registry-only", v["source"])
        check("still ok", v["ok"] is True)

        print("== 2. cost rule: cold only when cold is cheaper ==")
        r = svc.report_bridge_tools(catalog)
        check("report accepted", r.get("ok") is True, str(r))
        check("all four written", r.get("written") == 4, str(r))
        v = svc.tool_exposure(refresh=True)
        check("source is now the ledger", v["source"] == "ledger", v["source"])
        check("total is four", v["stats"].get("total") == 4, str(v["stats"]))
        # A tally that describes a different set than the lists beneath it is
        # worse than no tally, because it is the one a reader trusts first.
        # Measured 2026-11-02: the reply said "3 hot, 0 cold of 0 bridge tools"
        # because `hot`/`cold` covered both populations while `total` counted
        # only the bridge's.
        check("the tally counts what the lists contain",
              v["stats"]["hot"] == len(v["hot"]) and v["stats"]["cold"] == len(v["cold"]),
              f"stats={v['stats']} hot={len(v['hot'])} cold={len(v['cold'])}")
        check("the sentence and the tally agree",
              f"{len(v['hot'])} hot, {len(v['cold'])} cold of {v['stats']['total']} tools" in v["text"],
              v["text"])

        cheap_full = cost_of("autoforge_ping", small_desc, small_params)
        cold_line_text = exposure_mod.cold_description(
            "autoforge_ping",
            {"type": "function", "function": {"name": "autoforge_ping",
                                              "description": small_desc,
                                              "parameters": small_params}},
            7.0, desc_override=small_desc)
        cheap_cold = len("autoforge_ping") + len(cold_line_text) + exposure_mod.COLD_STUB_CHARS
        check("the small tool would cost more cold", cheap_cold >= cheap_full,
              f"full={cheap_full} cold={cheap_cold}")
        check("so it stays hot", "autoforge_ping" in v["hot"], str(sorted(v["cold"])))
        check("and it is not in the cold map", "autoforge_ping" not in v["cold"])

        check("the fat tool goes cold", "autoforge_gpu_bench" in v["cold"], str(v["hot"]))
        check("its sibling goes cold too", "autoforge_cpu_tune" in v["cold"])

        print("== 3. the floor is honoured regardless of cost ==")
        check("run_python is hot because it is the floor", "autoforge_run_python" in v["hot"], str(v["hot"]))
        check("the floor is in the hot list even though it is cheap",
              "autoforge_run_python" not in v["cold"])

        print("== 4. the cold line carries the argument names ==")
        line = v["cold"].get("autoforge_gpu_bench", "")
        check("it names parameters", "Params:" in line, line[:140])
        check("it names the first four", "axis_0, axis_1, axis_2, axis_3" in line, line[:160])
        check("it elides the rest", "..." in line, line[:160])
        check("it points at describe_tool", "autoforge_describe_tool('autoforge_gpu_bench')" in line)
        check("it is one line", "\n" not in line, repr(line))

        print("== 5. a descriptor with a parameter named like a schema keyword ==")
        # `autoforge_tool_define` really does take one called `description`. The
        # first implementation guessed the shape from a keyword blocklist and
        # answered "Takes no arguments." for it -- the one line whose entire job
        # is to name the arguments.
        kw = {"description": {"type": "string"}, "code": {"type": "string"}}
        names = exposure_mod._names({"function": {"name": "x", "parameters": kw}})
        check("a keyword-named parameter is still a parameter", names == ["description", "code"], str(names))
        real = exposure_mod._names({"function": {"name": "x", "parameters": {"type": "object",
                                                                            "properties": {"a": {}, "b": {}}}}})
        check("a JSON Schema still reads by its properties", real == ["a", "b"], str(real))

        print("== 6. a call makes a cold tool hot, in either spelling ==")
        svc.store.log_event("call", {"tool": "autoforge_gpu_bench", "ok": True})
        v = svc.tool_exposure(refresh=True)
        check("the called tool is now hot", "autoforge_gpu_bench" in v["hot"], str(v["hot"]))
        check("and left the cold map", "autoforge_gpu_bench" not in v["cold"])
        check("its sibling is still cold", "autoforge_cpu_tune" in v["cold"])

        svc.store.log_event("call", {"tool": "cpu_tune", "ok": True})
        v = svc.tool_exposure(refresh=True)
        check("a short ledger name reaches the prefixed tool", "autoforge_cpu_tune" in v["hot"], str(v["hot"]))

        print("== 7. refresh means refresh ==")
        before = svc.tool_exposure()
        svc.store.log_event("call", {"tool": "autoforge_ping", "ok": True})
        within = svc.tool_exposure()
        check("within the TTL the answer is reused", within["hot"] == before["hot"],
              f'{before["hot"]} vs {within["hot"]}')
        forced = svc.tool_exposure(refresh=True)
        check("refresh bypasses both caches", "autoforge_ping" in forced["hot"], str(forced["hot"]))

        print("== 8. protect ==")
        v = svc.tool_exposure(protect=["autoforge_gpu_bench"], refresh=True)
        check("protect keeps a name hot", "autoforge_gpu_bench" in v["hot"], str(v["hot"]))
    finally:
        try:
            svc.store.close()
            svc.missions.close()
        finally:
            shutil.rmtree(home, ignore_errors=True)

    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
