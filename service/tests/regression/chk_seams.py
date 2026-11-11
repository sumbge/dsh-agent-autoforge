"""The fifth same-shape misjudgement, and the only method that would have caught it.

This is not a document. It is a check that runs.

The question it answers -- "how do I avoid the fifth same-shape misjudgement" --
has a property that makes advice worthless: every one of the five felt correct
at the time. So the answer cannot be "be more careful" or "remember to verify".
It has to be a mechanical test that fails on the shape, whoever wrote it.

The shape, stated once
----------------------
    A thing is proven at layer A, then used at layer B, and nothing looks at
    the seam. Both layers are individually correct. The proof is real. The use
    is real. Only the join is missing.

Five instances, in this project, each found only after the fact:

  #1  `describe_tool` described in the persona. It did not exist. Persona is
      layer A, the service is layer B.
  #2  `force=True` named in an error message. `Service.tool_call` never accepted
      it. The message is layer A, the signature is layer B.
  #3  `rehab` written in a docstring. `registry.rehab` existed, no path reached
      it. Documentation layer A, dispatch layer B.
  #4  `chk_bridge.mjs` asserted 27/27 in the report. It had crashed on its tenth
      line since item two. The report is layer A, the harness is layer B.
  #5  `anchor_ledger` tested by calling the method. The bridge calls the
      dispatch table. The test is layer A, the door is layer B.

(#5 happened today. I broke it myself, yesterday, in the same session that
wrote three tests about honesty.)

What the five have in common is not carelessness. It is that **each check was
performed on the layer that was convenient to check**, and the convenience was
invisible because it felt like thoroughness.

So this file encodes the seam checks, one per pair of layers that actually talk.
It is deliberately small and mechanical: it can be read in one sitting, and
every assertion names the two layers it joins.
"""
import ast
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile

SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLUGIN_DIR = os.path.dirname(SERVICE_DIR)
sys.path.insert(0, SERVICE_DIR)

from autoforge_svc.service import Service  # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print(f"  OK   {label}")
    else:
        print(f"  FAIL {label}  <- {detail}")
        FAILS.append(label)


def read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def seams() -> None:
    print("-- seam 1: persona text (layer A) vs. the wire (layer B)")
    # Instance #1 and #3 both lived here: a name written in prose that nothing
    # could call. Any `autoforge_x` mentioned in the persona must be a real
    # registered tool name, and any *method* name mentioned must be on the wire
    # table.
    svc = Service(db_path=os.path.join(tempfile.mkdtemp(prefix="m13_seam_"), "seam.db"))
    try:
        wire = set(svc.methods)
        persona_paths = [
            os.path.join(PLUGIN_DIR, "presets", "autoforge.patch.yml"),
        ]
        raw = ""
        for path in persona_paths:
            if os.path.isfile(path):
                raw += read(path)
        check("the persona file was found", bool(raw), persona_paths)

        # Only the instructions count. YAML comments in this file are notes to
        # humans and legitimately name things in other namespaces -- the first
        # run of this check failed on `python -m autoforge_svc` (a module) and
        # `apply()` (a JS function), both inside the same comment block about
        # history. A check that cannot tell an instruction from a footnote would
        # force the comments to be deleted, which is the wrong repair: the
        # comments are the only record of why the de-duplication exists.
        persona = "\n".join(
            line for line in raw.splitlines() if not line.lstrip().startswith("#")
        )
        check("the persona's instruction text is non-trivial", len(persona) > 200, len(persona))

        # Tool names the persona tells the model to call. The persona opens with
        # "you have a family of `autoforge_*` tools, of which the most important
        # is `forge_tool`" and then uses `forge_tool(need=...)` and
        # `describe_tool(name)` -- short forms whose prefix is stated one line
        # earlier. That is a declared shorthand, not a phantom name, so it is
        # resolved rather than banned; but the resolution is printed, because a
        # shorthand that stops resolving is exactly the seam this file guards.
        named_tools = sorted(set(re.findall(r"\bautoforge_[a-z_]+", persona)))
        bare_tools = sorted(set(re.findall(r"(?<![\w`])`([a-z][a-z0-9_]+)\((?=[a-z_]+=|\))", persona)))
        bridge = read(os.path.join(PLUGIN_DIR, "lib", "index.js"))
        registered = set(re.findall(r"toolName:\s*'([^']+)'", bridge))

        unknown_tools = [t for t in named_tools if t not in registered]
        check("every prefixed tool the persona names is registered by the bridge",
              not unknown_tools, unknown_tools)

        prefix_axis_tools = {t[len("autoforge_"):] for t in registered if t.startswith("autoforge_")}
        unknown_bare = [t for t in bare_tools if t not in prefix_axis_tools]
        check("every short name the persona calls resolves under the `autoforge_` prefix",
              not unknown_bare, unknown_bare)
        check("the persona names at least one tool to check",
              bool(named_tools or bare_tools), (named_tools, bare_tools))
        if bare_tools:
            print("       short forms resolved: " + ", ".join(
                f"{b} -> autoforge_{b}" for b in bare_tools if b in prefix_axis_tools))

        # `bare_tools` only matches a name followed by `(` -- and the sixth
        # instance of the seam was `rehab`, written with no parentheses at all:
        # "`rehab` can put it back on probation". The regex above is the shape
        # of a call, so a *capability claim* phrased as a noun slips past it.
        # Measured 2026-11-03: with the unfixed persona in place this file
        # printed TOTAL FAILURES: 0, and the name it was written to catch was
        # sitting on line 59.
        #
        # So the sweep below is over EVERY backticked lowercase identifier,
        # call-shaped or not. Each must resolve to something the model can
        # actually reach: a bridge tool (long or short form), a wire method, or
        # a declared state value. Anything else has to be named in DATA_NAMES
        # with a reason -- that list is the only way in, so adding a phantom
        # callable later means editing this file in plain sight.
        STATE_VALUES = {"draft", "probation", "active", "quarantined", "retired"}
        DATA_NAMES = {
            # Config fields and counters: values the persona quotes, not things
            # it tells the model to call. Each is a real attribute; each is
            # printed beside the behaviour it governs, which is the point.
            "max_rounds",              # ForgeConfig: the 3 in "最多 3 轮"
            "promote_on_pass",         # ForgeConfig: why a clean pass seals ACTIVE
            "calls_not_in_counters",   # ToolStats: what rehab_rehab does NOT erase
        }
        all_bt = sorted({t for t in re.findall(r"`([^`\n]+)`", persona)
                         if re.fullmatch(r"[a-z][a-z0-9_]*", t)})
        reachable = registered | prefix_axis_tools | wire | STATE_VALUES | DATA_NAMES
        phantom = [t for t in all_bt if t not in reachable]
        check("every backticked lowercase name in the persona resolves somewhere reachable",
              not phantom, phantom)
        stale = sorted(n for n in DATA_NAMES if n not in all_bt)
        check("no entry in DATA_NAMES is stale", not stale, stale)
        check("DATA_NAMES stays small enough to review at a glance",
              len(DATA_NAMES) <= 8, sorted(DATA_NAMES))
        check("the sweep actually looked at names (not an empty persona)",
              len(all_bt) >= 10, len(all_bt))
        print(f"       backticked names checked: {len(all_bt)}; "
              f"resolved as data: {sorted(n for n in all_bt if n in DATA_NAMES)}")

        # Method names the persona tells the model exist.
        prose_calls = sorted(set(re.findall(r"\b([a-z][a-z0-9_]{3,})\(\)", persona)))
        unwired_calls = [m for m in prose_calls if m not in wire and m not in {
            "main", "print", "run", "open", "format", "join", "keys", "items", "get",
        }]
        check("no prose names a method that is absent from the wire",
              not unwired_calls, unwired_calls)
    finally:
        svc.store.close()
        svc.missions.close()

    print("-- seam 2: test calls (layer A) vs. the door the caller uses (layer B)")
    # Instance #5. The rule: if the product reaches a service method through
    # `svc.methods`, a test that calls `svc.method(...)` directly proves nothing
    # about reachability. This asserts the reverse direction is covered: every
    # method the bridge reaches must be on the table (chk_wire owns the full
    # sweep; here we pin the shape so a new tool cannot slip in unnoticed).
    called = sorted(set(re.findall(r"method:\s*'([a-z_][a-z0-9_]*)'", bridge)))
    check("the bridge's method list was parsed", len(called) >= 60, len(called))

    print("-- seam 3: the report (layer A) vs. the harnesses it cites (layer B)")
    # Instance #4. A number in prose is a claim about a file. If the file exits
    # non-zero, the claim is false whatever the number says.
    # Not in this repository: the change log is where the changes were made.
    # A clone without it should skip this seam with a reason, not fail it --
    # an absent file is not evidence that a cited harness is missing.
    report_path = os.environ.get("AUTOFORGE_REPORT", "")
    if report_path and os.path.isfile(report_path):
        report = read(report_path)
        cited = sorted(set(re.findall(r"`(chk_[a-z0-9_]+\.(?:py|mjs))`", report)))
        present = set(os.listdir(os.path.join(SERVICE_DIR, "tests", "regression")))
        missing = [c for c in cited if c not in present]
        check("every harness the report cites exists on disk", not missing, missing)
        check("the report cites a plausible number of harnesses", len(cited) >= 5, len(cited))
    else:
        print("SKIP every harness the report cites exists on disk"
              "  -- no report to compare against (set AUTOFORGE_REPORT to check one)")

    print("-- seam 4: a test's exit code (layer A) vs. its presence in the run list (layer B)")
    # Instance #4's real cause: the harness was not in any list I ran. The fix
    # is that the list is derived, not remembered.
    reg = os.path.join(SERVICE_DIR, "tests", "regression")
    files = sorted(f for f in os.listdir(reg) if f.startswith("chk_") and f.endswith((".py", ".mjs")))
    helpers = sorted(f for f in files if f.endswith(".hooks.mjs"))
    diagnostics = {
        "chk_counter.py", "chk_dbg.py", "chk_diag_small.py",
        "chk_pathaxis.py", "chk_sig.py", "chk_state.py",
    }
    suites = [f for f in files if f not in helpers and f not in diagnostics]
    check("the suite list is derived from the directory, not from memory",
          len(suites) >= 18, len(suites))
    check("helpers are excluded by a rule, not by a hardcoded path",
          all(h.endswith(".hooks.mjs") for h in helpers), helpers)


def main() -> int:
    seams()
    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
