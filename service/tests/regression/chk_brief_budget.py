"""A per-turn read must not build the forge pipeline.

Measured 2026-11-04. `context_brief()` and `list_tools()` used to call
`self._registry().report()`. `_registry()` calls `self._forge_pipeline()` as a
side effect; after the tool-capability probe was added, pipeline construction
makes REAL gateway HTTP requests (up to 5 candidate models, timeout=90s).

The consequences, in order:
  m80  fresh process: probe_lines 519.8ms, but context_brief 65350.2ms
  m81  brief-first over stdio wedges tool_exposure behind it: both ~12269ms
       (the service is single-threaded over stdio; the JS mount fires the brief
       at mount and tool_exposure at assemble, sharing one 1500ms deadline)
  m82  helper timing: _registry 10112.0ms of a 10491.5ms context_brief
  m83  S0 unpatched 11356.7ms / pipeline_built true
       S1 _tool_capable_model patched to a no-op -> 326.2ms
       S2 self.registry.report() directly -> 0.0ms / pipeline_built false
       S3 one _reads_tool_results probe -> 4719.4ms
  -> the model probe is the whole cost; a pure read needs none of it.

On the wire this showed up as `chk_exposure.mjs`: 27413 -> 27413 chars
(0.0% saved) and "memory brief did not answer within 1500ms". Hot/cold
projection was silently off, not broken -- `applyExposure` simply never got an
answer inside the budget.

The fix: pure reads use `self.registry` (fully populated in `__init__` from
`load_all_tools()`); `_registry()` is kept for the heavy forge/evaluate paths
and carries a comment saying so.

Discriminating design (this file has to be able to FAIL):
  H1 (fixed): context_brief / list_tools build no pipeline and never ask the
      gateway about a model -- both counters stay 0.
  H2 (regressed): a read routes through `_registry()` -- the pipeline counter
      becomes 1. The `trap_works` block proves the counter actually observes
      `_registry()`, so a 0 cannot be an empty measurement.
"""
import os
import sys
import ast
import time
import inspect
import textwrap
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)

FAILS = []


def check(label, cond, detail=""):
    if cond:
        print("OK   %s" % label)
    else:
        FAILS.append(label)
        print("FAIL %s  <- %s" % (label, detail))


def calls_in(src):
    """The dotted names actually CALLED, not merely written in a comment.

    A substring test on source text is how the first version of this file
    failed: the fix's own comment says `never self._registry()`, and the string
    check read that comment as a live call. Ask the parser, not the prose.
    """
    tree = ast.parse(textwrap.dedent(src))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            try:
                out.add(ast.unparse(node.func))
            except Exception:
                pass
    return out


def main():
    os.environ["AUTOFORGE_HOME"] = tempfile.mkdtemp(prefix="chk_brief_budget_")
    import autoforge_svc.service as S
    from autoforge_svc.service import Service

    calls = {"pipeline": 0, "probe": 0}
    real_pipeline = S.Service._forge_pipeline
    real_probe = S._tool_capable_model

    def fake_pipeline(self):
        # Stand in for the whole construction. A real one would reach the
        # gateway; the point of this test is to count the REACH, not to pay it.
        calls["pipeline"] += 1
        self._pipeline = object()
        return self._pipeline

    def fake_probe(model, base, key):
        calls["probe"] += 1
        return (model, "configured")

    S.Service._forge_pipeline = fake_pipeline
    S._tool_capable_model = fake_probe
    try:
        # -- the two per-turn reads -----------------------------------------
        svc = Service()
        calls["pipeline"] = calls["probe"] = 0
        t0 = time.time()
        brief = svc.context_brief()
        brief_ms = (time.time() - t0) * 1000.0
        check("context_brief does not build the forge pipeline",
              calls["pipeline"] == 0, "pipeline builds: %d" % calls["pipeline"])
        check("and it never asks the gateway whether a model reads tool results",
              calls["probe"] == 0, "model probes: %d" % calls["probe"])
        check("and it still returns a real brief",
              isinstance(brief, dict) and bool((brief.get("text") or "").strip()),
              "got %r" % (brief,))
        check("and it stays well inside the bridge's 1500ms budget",
              brief_ms < 1500.0, "%.1f ms" % brief_ms)

        svc2 = Service()
        calls["pipeline"] = calls["probe"] = 0
        listed = svc2.list_tools()
        check("list_tools does not build the forge pipeline either",
              calls["pipeline"] == 0, "pipeline builds: %d" % calls["pipeline"])
        check("and it neither probes a model",
              calls["probe"] == 0, "model probes: %d" % calls["probe"])
        check("and it still reports the library",
              isinstance(listed, dict) and "total" in listed,
              "got %r" % (listed,))

        # -- the trap must be observable, or a 0 above proves nothing -------
        svc3 = Service()
        calls["pipeline"] = calls["probe"] = 0
        reg = svc3._registry()
        check("the counter observes _registry() -- so a zero above is real",
              calls["pipeline"] == 1 and reg is svc3.registry,
              "pipeline builds: %d" % calls["pipeline"])

        # -- source shape: no silent re-routing in a future edit ------------
        # Call sites only. Comments explaining the fix mention _registry by
        # name; a text-contains test would call those mentions a regression.
        cb_calls = calls_in(inspect.getsource(S.Service.context_brief))
        lt_calls = calls_in(inspect.getsource(S.Service.list_tools))
        reg_calls = calls_in(inspect.getsource(S.Service._registry))
        check("context_brief calls self.registry.report(), never self._registry()",
              "self.registry.report" in cb_calls and "self._registry" not in cb_calls,
              "calls: %s" % sorted(cb_calls))
        check("list_tools calls self.registry.report(), never self._registry()",
              "self.registry.report" in lt_calls and "self._registry" not in lt_calls,
              "calls: %s" % sorted(lt_calls))
        check("_registry still builds the pipeline for the heavy paths",
              "self._forge_pipeline" in reg_calls,
              "calls: %s" % sorted(reg_calls))
    finally:
        S.Service._forge_pipeline = real_pipeline
        S._tool_capable_model = real_probe

    print("TOTAL FAILURES: %d %s" % (len(FAILS), FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
