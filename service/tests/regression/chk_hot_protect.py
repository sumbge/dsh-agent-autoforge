"""The tools this turn is about must ride in full, whatever the ledger says.

`exposure.classify` decides hot/cold from one input: how recently a tool was
called. That is the right question for a cost decision and the wrong one for a
task -- it is a question about the *past*. A turn whose text says "compile a
CUDA kernel" needs `gpu_compile` in full on its first step, and the ledger has
never seen that name, so the one turn that needs the tool is the one turn that
cannot read its arguments.

`autoforge` upstream solved this in `agent.py::_hot_protect`: take the task
text, match tool names on word boundaries, hand the survivors to the classifier
as `protect`. The mechanism came over to this branch in
`tool_exposure(protect=[...])` and in `exposure.classify(protect=...)`, and then
nothing ever filled the list -- the bridge calls `tool_exposure` with `{}` on
every assembly. Measured 2026-11-03: `registry.protect` was assigned nowhere in
the package, and the argument was accepted, tested, and always empty.

What this file pins:

1. the matcher itself -- whole tokens and contiguous runs of tokens, never
   fragments (upstream's own docstring records the bug: a substring test kept
   `see` hot on every task containing "seen");
2. that a protected name really does come back hot from the service, not just
   that the word survived a regex;
3. that protection is *narrow* -- a task naming nothing must protect nothing, or
   the setting is a no-op dressed as a feature.
"""
from __future__ import annotations

import os
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)

from autoforge_svc.tools import exposure as exposure_mod  # noqa: E402
from autoforge_svc.service import Service                    # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print("  OK   " + label)
    else:
        print("  FAIL " + label + " <- " + str(detail))
        FAILS.append(label)


def main() -> int:
    print("-- a name in the task is protected, a fragment is not")
    names = ["gpu_compile", "run_python", "see", "run", "browser_open"]
    tool = getattr(exposure_mod, "protect_from_task", None)
    check("exposure exposes a task-to-names matcher", callable(tool), type(tool).__name__)
    if not callable(tool):
        print("TOTAL FAILURES:", len(FAILS), FAILS)
        return 1

    hit = tool("compile the kernel with gpu_compile please", names)
    check("an exact name in the text is protected", "gpu_compile" in hit, sorted(hit))
    check("a name that is not in the text is not", "run_python" not in hit, sorted(hit))

    # The whole point of boundaries. `see` inside "seen" and `run` inside
    # "running" are the two shapes that made upstream's first draft useless.
    frag = tool("I have seen it running before", names)
    check("'seen' does not protect 'see'", "see" not in frag, sorted(frag))
    check("'running' does not protect 'run'", "run" not in frag, sorted(frag))
    check("and nothing else is protected either", frag == set(), sorted(frag))

    words = tool("please run the python part", ["run", "run_python"])
    check("a single-word name matches a whole token", "run" in words, sorted(words))
    check("a multi-word name does not match two separate words",
          "run_python" not in words, sorted(words))
    joined = tool("now call run python", ["run_python"])
    check("a multi-word name matches a contiguous run of its own words",
          "run_python" in joined, sorted(joined))
    apart = tool("run this, then python later", ["run_python"])
    check("but not when the words are separated by other text",
          "run_python" not in apart, sorted(apart))

    case = tool("Use GPU_COMPILE now", names)
    check("matching is case-insensitive", "gpu_compile" in case, sorted(case))
    empty = tool("", names)
    check("no task text protects nothing", empty == set(), sorted(empty))
    noname = tool("some ordinary sentence about nothing", names)
    check("an unrelated sentence protects nothing", noname == set(), sorted(noname))

    print("-- and the service really promotes what the matcher names")
    tmp = tempfile.mkdtemp(prefix="m22_hot_")
    svc = Service(db_path=os.path.join(tmp, "hot.db"))
    try:
        # A tool with a fat description and one parameter: cheap enough to be
        # demoted on the ledger's evidence, expensive enough that keeping it hot
        # is a visible difference in the answer.
        svc.tool_define(
            name="fat_unused",
            code="def main(**k):\n    return 1\n",
            description="x" * 400,
            parameters={"p": {"type": "string", "required": True}},
        )
        plain = svc.tool_exposure(refresh=True)
        check("with no protection the unused tool is cold", "fat_unused" not in plain["hot"],
              plain["hot"])
        guarded = svc.tool_exposure(protect=["fat_unused"], refresh=True)
        check("naming it protects it", "fat_unused" in guarded["hot"], guarded["hot"])
        check("and it leaves the cold list", "fat_unused" not in guarded["cold"],
              sorted(guarded["cold"]))
        check("the protection is per call, not sticky",
              "fat_unused" not in svc.tool_exposure(refresh=True)["hot"],
              svc.tool_exposure(refresh=True)["hot"])

        short = svc.tool_exposure(protect=["unused"], refresh=True)
        check("a short name protects nothing, because the real name carries the prefix",
              "fat_unused" not in short["hot"], short["hot"])

        # -- the text path, which is the one the bridge uses ------------------
        #
        # `protect` is a name list, and the bridge is JavaScript with no opinion
        # about tool names. So the wire carries `task` -- the turn's own words --
        # and the service derives the names. Measured 2026-11-03: this path was
        # called with `{}` on every assembly, so the argument existed, was
        # tested in isolation, and was never filled on the model's path.
        by_task = svc.tool_exposure(task="please read the fat_unused table", refresh=True)
        check("a task that names a tool keeps it hot", "fat_unused" in by_task["hot"],
              by_task["hot"])
        quiet = svc.tool_exposure(task="some ordinary sentence about nothing", refresh=True)
        check("a task that names nothing protects nothing", "fat_unused" not in quiet["hot"],
              quiet["hot"])
        # A matcher failure must degrade to the ledger's answer, not to a crash
        # and not to "protect everything".
        failures = svc.my_history(kind="exposure_task_failed", limit=5)
        check("the text path raised nothing", int(failures.get("count") or 0) == 0,
              failures.get("events"))

        # -- the cache has to include what was asked --------------------------
        #
        # The 60s TTL was keyed on time alone. Measured 2026-11-03: with an
        # empty `protect` cached, the next three calls with a non-empty list
        # returned the empty answer for a full minute -- the first turn of a new
        # task is exactly the turn where the argument changes, so the one turn
        # that needed protection was the one turn that could not get it.
        svc.tool_exposure(task="about nothing at all", refresh=True)     # cache a "no" answer
        warmed = svc.tool_exposure(task="now read the fat_unused table")  # no refresh
        check("a different question is not answered from the previous cache",
              "fat_unused" in warmed["hot"], warmed["hot"])
        back = svc.tool_exposure(task="about nothing at all")
        check("and going back to the older question re-answers it",
              "fat_unused" not in back["hot"], back["hot"])
        same = svc.tool_exposure(task="now read the fat_unused table")
        check("an identical question inside the TTL still uses the cache",
              "fat_unused" in same["hot"], same["hot"])

        # Order is not a different question: the same names in another order are
        # the same set, and a cache that treats them as different would rebuild
        # for no reason on every caller.
        one = svc.tool_exposure(protect=["fat_unused", "nothing_here"], refresh=True)
        two = svc.tool_exposure(protect=["nothing_here", "fat_unused"])
        check("the same names in a different order are the same question",
              one["stats"] == two["stats"] and one["hot"] == two["hot"],
              [one["stats"], two["stats"]])

        # End to end: text in, hot name out. This is the seam that was missing --
        # every piece existed and no caller joined them.
        names_now = [n for n in svc.registry.names()]
        found = exposure_mod.protect_from_task("read the fat_unused table", names_now)
        check("the matcher sees the forged tool's real name", "fat_unused" in found, sorted(found))
        via_text = svc.tool_exposure(protect=sorted(found), refresh=True)
        check("and that is enough to keep it hot", "fat_unused" in via_text["hot"], via_text["hot"])
    finally:
        svc.store.close()
        svc.missions.close()

    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
