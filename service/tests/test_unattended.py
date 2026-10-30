"""The unattended path: what the OS scheduler is actually given, and whether it runs.

This exists because the answer was "no" twice over, and both failures were
invisible from the outside.

The upstream `wake_command` emits `python -m autoforge tick --quiet`. That is
correct for `autoforge/`, which has `__main__.py` and a `tick` subcommand. This
service is a *copy* of that file living in `autoforge_svc/`, so the inherited
literal emitted a command that dies with `No module named autoforge` every time
the OS ran it. Worse, the service's own entry point treated any unknown first
argument as a DATABASE PATH: `python -m autoforge_svc tick` opened a database
called `tick` and answered `{"ok": true, "db": "tick"}`.

Two failure modes, and the second is the dangerous one. A task that errors
prompts a look. A task that reports success while doing nothing never does, so
the schedule silently never runs and nothing anywhere says so.

What is tested here:

  * the emitted command names a module this package is importable as;
  * `tick` is a real subcommand, not a path;
  * an unknown argument is a usage error rather than a database;
  * a due task is genuinely run and its outcome recorded;
  * an empty schedule is silent and successful, because a scheduler that reports
    "nothing happened" every fifteen minutes trains its reader to stop reading.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoforge_svc import __main__ as entry          # noqa: E402
from autoforge_svc import schedule as schedule_mod    # noqa: E402
from autoforge_svc.service import Service             # noqa: E402

# The package's own name, not the checkout directory's: the directory is
# `autoforge-svc-gw`, which is not a valid module name, and taking it from the
# path is what made `-m autoforge-svc-gw` fail in the first draft of these
# tests.
# tests/ -> the checkout dir, which CONTAINS the `autoforge_svc` package.
# That directory is what belongs on PYTHONPATH; one level further up is
# `Desktop\dsh`, where `-m autoforge_svc` cannot be found -- measured.
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_NAME = __import__("autoforge_svc").__name__


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    """The schedule is a file under $AUTOFORGE_HOME, not the sqlite database.

    A test that only passed a db_path shared the real machine's schedule --
    measured: `schedule_run_due` found three tasks due in a "fresh" service.
    """
    monkeypatch.setenv("AUTOFORGE_HOME", str(tmp_path))
    return tmp_path


def run_module(*argv: str, home: str) -> subprocess.CompletedProcess:
    """Invoke the package the way the OS scheduler would."""
    env = dict(os.environ)
    env["AUTOFORGE_HOME"] = home
    env["PYTHONPATH"] = PKG_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", PKG_NAME, *argv],
        capture_output=True, text=True, timeout=120, env=env,
        cwd=PKG_ROOT,
    )


# -- the command the OS is given -------------------------------------------

def test_the_wake_command_names_a_module_that_imports():
    """The literal was inherited from `autoforge/` and named the wrong package."""
    argv = schedule_mod.wake_command(15)
    module = argv[argv.index("-m") + 1]
    assert module == "autoforge_svc", (
        f"the scheduled task would run `-m {module}`, which is not this package")
    assert argv[-2:] == ["tick", "--quiet"]


def test_the_emitted_command_can_be_imported_by_a_child():
    """Not just spelled right -- importable where the OS will run it."""
    argv = schedule_mod.wake_command(15)
    module = argv[argv.index("-m") + 1]
    import importlib

    mod = importlib.import_module(module)
    assert hasattr(mod, "__path__"), f"{module} is not a package"


# -- tick is a subcommand, not a database --------------------------------

def test_tick_is_a_real_subcommand(tmp_path):
    result = run_module("tick", home=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "No module named" not in result.stderr
    assert "Nothing is due" in result.stdout


def test_an_unknown_argument_is_a_usage_error_not_a_database(tmp_path):
    """The bug: `tick` used to be read as a database path and answer ok."""
    result = run_module("bogus-command", home=str(tmp_path))
    assert result.returncode == 2
    assert "unknown command" in result.stderr
    assert "db" not in result.stdout


def test_the_help_names_both_modes(tmp_path):
    result = run_module("--help", home=str(tmp_path))
    assert result.returncode == 0
    assert "tick" in result.stdout
    assert "stdio" in result.stdout.lower()


# -- and it genuinely runs the work --------------------------------------

def test_a_due_task_is_run_and_recorded(isolated_home):
    """The whole point: the schedule outlives the session that created it."""
    svc = Service(str(isolated_home / "autoforge.db"))
    import time

    svc.schedule_add(text="say something", when="+1s")
    time.sleep(1.2)
    due = svc.schedule_tick()
    assert due["count"] == 1
    assert due["due"][0]["overdue_by"] >= 0

    # A model is required to actually run it; assert the executor's shape
    # without one rather than reaching the network in a unit test.
    ran = svc.schedule_run_due(budget_s=5.0)
    assert ran["due"] == 1
    # Either it ran (a model is configured) or it failed loudly (none is).
    # Silence is the only outcome that must not happen.
    assert ran["ran"] + ran["failed"] == 1
    row = ran["results"][0]
    assert row["ok"] or row["note"], "a task must record why it failed"


def test_an_empty_schedule_is_silent_and_successful(isolated_home):
    svc = Service(str(isolated_home / "autoforge.db"))
    out = svc.schedule_run_due()
    assert out["due"] == 0
    assert out["failed"] == 0
    assert out["note"] == "nothing was due"


def test_a_task_past_its_budget_is_skipped_not_lost(isolated_home):
    """A wall-clock ceiling must record the skip, not silently drop the task."""
    svc = Service(str(isolated_home / "autoforge.db"))
    import time

    svc.schedule_add(text="first", when="+1s")
    time.sleep(1.2)
    # Zero budget: the first task finds nothing left to spend.
    out = svc.schedule_run_due(budget_s=0.0)
    assert out["due"] == 1
    assert out["results"][0].get("skipped") is True
    assert "budget" in out["results"][0]["note"]


def test_overdue_by_takes_a_timestamp(isolated_home):
    """`overdue_by(None)` raised TypeError inside a comprehension.

    The whole call failed, not one row -- and only when something was due, which
    is the only time the call matters.
    """
    svc = Service(str(isolated_home / "autoforge.db"))
    import time

    svc.schedule_add(text="x", when="+1s")
    time.sleep(1.2)
    out = svc.schedule_tick()          # used to raise TypeError here
    assert isinstance(out["due"][0]["overdue_by"], float)
