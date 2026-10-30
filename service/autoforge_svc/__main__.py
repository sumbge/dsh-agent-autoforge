"""`python -m autoforge_svc ...` -- the invocation the OS scheduler is given.

Two things live here and nothing else, because both are reachable without a DSH
session and neither belongs behind one.

**The stdio service** (no arguments) is what the DSH plugin spawns: JSON-RPC on
stdin, one answer per line. That is the mode everything else in this package is
written for.

**`tick`** is what `schedule.install_system_task` registers, and it exists
because the alternative is worse than it looks. The upstream `wake_command`
emits `python -m autoforge tick --quiet`, which cannot work here for two
independent reasons -- the package is `autoforge_svc`, not `autoforge`, and this
service had no `tick` entry point at all. The failure modes are not equivalent:

  * `python -m autoforge tick` exits with `No module named autoforge`, so the
    task fails visibly-but-silently every fifteen minutes.
  * `python -m autoforge_svc tick` used to treat `tick` as the DATABASE PATH
    (`Service(argv[0])`) and answer `{"ok": true, "db": "tick"}` on a stream
    nobody was reading. A task that appears to succeed while doing nothing is
    worse than one that errors, because nothing ever prompts a look.

So `tick` is a real subcommand here: it takes the due tasks, runs them, records
their outcome, and exits. An argument that is neither `tick` nor empty is now a
usage error rather than a database path -- the ambiguity was the bug.
"""
from __future__ import annotations

import sys

USAGE = """autoforge_svc -- the autoforge service for DSH

usage:
  python -m autoforge_svc [starts the stdio service]
  python -m autoforge_svc tick [--quiet] [--budget SECONDS]
      attend to whatever the schedule says is due, then exit.
      This is what `install_system_task` registers.
"""


def _tick(argv: list[str]) -> int:
    """Run whatever is due, print a line per task unless --quiet, then exit."""
    quiet = "--quiet" in argv or "-q" in argv
    budget = 600.0
    for i, arg in enumerate(argv):
        if arg == "--budget" and i + 1 < len(argv):
            try:
                budget = float(argv[i + 1])
            except ValueError:
                print(f"--budget wants a number of seconds, not {argv[i + 1]!r}",
                      file=sys.stderr)
                return 2

    from .service import Service

    try:
        svc = Service()
    except Exception as exc:                                # noqa: BLE001
        print(f"could not open the service: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    try:
        out = svc.schedule_run_due(budget_s=budget)
    except Exception as exc:                                # noqa: BLE001
        print(f"tick failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not quiet:
        due = out.get("due", 0)
        if not due:
            print("Nothing is due.")
        else:
            print(f"{due} task(s) due:")
            for row in out.get("results", []):
                flag = ("skipped" if row.get("skipped")
                        else "done" if row.get("ok") else "FAILED")
                print(f"  {row.get('id')}: {flag}: {row.get('note', '')[:200]}")
            print(f"\n{out.get('ran', 0)} ran, {out.get('failed', 0)} failed "
                  f"in {out.get('elapsed_s', 0)}s")
    # A failed task is reported through the exit code so the OS scheduler's own
    # log shows it. Silence plus exit 0 is what made the broken path invisible.
    return 1 if out.get("failed") else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if argv and argv[0] == "tick":
        return _tick(argv[1:])
    if argv and not argv[0].startswith("-"):
        # Anything else is a mistake worth naming. Treating it as a database
        # path, as the first version did, is how `tick` came to answer "ok".
        print(f"unknown command {argv[0]!r}\n\n{USAGE}", file=sys.stderr)
        return 2

    # No arguments: the stdio service the DSH plugin speaks to.
    from .service import main as serve
    return serve(argv)


if __name__ == "__main__":
    raise SystemExit(main())
