"""The two system-task methods, which had no suite at all until 2026-11-06.

What is asserted here, and why each one earned its place:

  1. `install_system_task` raises instead of returning an inner `ok`. It used
     to hand back `{"ok": True, "result": ...}` and the dispatcher wrapped that
     again, so a refusal arrived as `{"ok": true, "result": {"ok": false}}`.
  2. `system_task_command` refuses an interval it cannot honour. It used to
     clamp 0 and -5 up to 1 and report success in terms of the number it was
     handed. Three real Windows tasks were registered that way from a probe
     that only meant to read a shape (deleted the same hour; see memory
     autoforge-probing-install-system-task-registers-real-tasks).
  3. Neither method is allowed to reach the operating system in this file.
     `subprocess.run` is replaced with a recording stub and the suite asserts
     the stub was called with the argv it expects -- so the test proves what
     WOULD be run without running it.

The stub is the point, not a convenience: a suite for a method whose whole job
is to mutate the machine must be able to fail without mutating the machine.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("AUTOFORGE_HOME", tempfile.mkdtemp(prefix="chk_systask_"))

# Put the package on the path before the first import of it. This used to
# `import autoforge_svc` first and then derive SERVICE from the result, which
# only worked when the caller had already exported PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import autoforge_svc  # noqa: E402

SERVICE = os.path.dirname(os.path.dirname(os.path.abspath(autoforge_svc.__file__)))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc import schedule as schedule_mod  # noqa: E402
from autoforge_svc.service import Service  # noqa: E402

FAILS: list[tuple[str, object]] = []


def check(label: str, condition: object, detail: object = "") -> bool:
    ok = bool(condition)
    print(("OK   " if ok else "FAIL ") + label + ("" if ok else f" <- {detail}"))
    if not ok:
        FAILS.append((label, detail))
    return ok


class _Stub:
    """Stands in for subprocess.run inside schedule.install_system_task."""

    def __init__(self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, argv, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append(list(argv))

        class _P:
            returncode = self.returncode
            stdout = self.stdout
            stderr = self.stderr

        return _P()


def main() -> dict:
    real_run = schedule_mod.subprocess.run

    # -- 1. the interval guard, no OS involvement at all --------------------
    refused_zero = refused_negative = None
    for value in (0, -5):
        try:
            schedule_mod.system_task_command(value, "chk-name", "win32", "")
            if value == 0:
                refused_zero = False
            else:
                refused_negative = False
        except schedule_mod.ScheduleError as exc:
            if value == 0:
                refused_zero = str(exc)
            else:
                refused_negative = str(exc)

    check("an interval of zero is refused, not clamped",
          isinstance(refused_zero, str) and "at least 1 minute" in refused_zero,
          refused_zero)
    check("the refusal quotes the number it was given",
          isinstance(refused_zero, str) and "got 0" in refused_zero, refused_zero)
    check("a negative interval is refused too",
          isinstance(refused_negative, str) and "got -5" in refused_negative,
          refused_negative)

    built = schedule_mod.system_task_command(15, "chk-name", "win32", "")
    check("a workable interval still builds a command",
          built[built.index("/MO") + 1] == "15", built)
    check("and the task name travels with it", "chk-name" in built, built)

    defaults = schedule_mod.system_task_command.__defaults__ or ()
    check("the documented default interval is still 30", defaults[0] == 30, defaults)

    # Linux shares the guard, so it cannot be bypassed by asking for cron.
    try:
        schedule_mod.system_task_command(0, "chk-name", "linux", "")
        linux_refused = False
    except schedule_mod.ScheduleError:
        linux_refused = True
    check("the guard is on the interval, not on the Windows branch", linux_refused)

    # -- 2. install_system_task raises; its inner ok is gone ----------------
    sig = inspect.signature(Service.install_system_task)
    check("the service method still takes an interval and a name",
          list(sig.parameters) == ["self", "interval_minutes", "name"], list(sig.parameters))

    stub_ok = _Stub(returncode=0, stdout=b"SUCCESS: created\r\n")
    schedule_mod.subprocess.run = stub_ok
    try:
        text = Service.__new__(Service).install_system_task(15, "chk-name")
    finally:
        schedule_mod.subprocess.run = real_run

    check("a successful install answers with a sentence, not an envelope",
          isinstance(text, str) and "chk-name" in text, text)
    check("it names the task it registered", "chk-name" in str(text), text)
    check("the stub saw exactly one invocation", len(stub_ok.calls) == 1, stub_ok.calls)
    check("the argv handed to the OS carries /Create and the interval",
          bool(stub_ok.calls) and "/Create" in stub_ok.calls[0]
          and stub_ok.calls[0][stub_ok.calls[0].index("/MO") + 1] == "15",
          stub_ok.calls)

    # A refused registration must raise. This is the whole point of removing the
    # try/except: the dispatcher turns an exception into ok:false, and an inner
    # {"ok": False} into a success envelope.
    stub_bad = _Stub(returncode=1, stderr=b"ERROR: Access is denied.\r\n")
    schedule_mod.subprocess.run = stub_bad
    raised: BaseException | None = None
    returned: object = None
    try:
        returned = Service.__new__(Service).install_system_task(15, "chk-name")
    except BaseException as exc:  # noqa: BLE001
        raised = exc
    finally:
        schedule_mod.subprocess.run = real_run

    check("a refused registration raises rather than returning a dict",
          isinstance(raised, schedule_mod.ScheduleError) and returned is None,
          {"raised": type(raised).__name__ if raised else None, "returned": returned})
    check("the exception carries what the scheduler said",
          "Access is denied" in str(raised), str(raised))

    # The service method must not build an envelope of its own.
    src = inspect.getsource(Service.install_system_task)
    body = src.split('"""')[-1] if '"""' in src else src
    check("the method body returns the module call directly",
          'return schedule_mod.install_system_task' in body, body.strip()[-120:])
    check("and it no longer manufactures an inner ok",
          '{"ok": True, "result"' not in body and '{"ok": False' not in body,
          body.strip()[-120:])

    # -- 3. neither helper touches the OS on its own -----------------------
    check("system_task_command is pure: it builds, it does not run",
          "subprocess.run" not in inspect.getsource(schedule_mod.system_task_command))

    # -- 4. the wire, with the stub still in place -------------------------
    # Done in-process on purpose: a child process would inherit the real
    # subprocess module and could actually register a task.
    svc = Service(os.path.join(tempfile.mkdtemp(prefix="chk_systask_db_"), "a.db"))
    stub_wire = _Stub(returncode=0, stdout=b"SUCCESS\r\n")
    schedule_mod.subprocess.run = stub_wire
    try:
        good = svc.install_system_task(20, "chk-wire")
        bad_raised: BaseException | None = None
        try:
            svc.install_system_task(0, "chk-wire")
        except BaseException as exc:  # noqa: BLE001
            bad_raised = exc
    finally:
        schedule_mod.subprocess.run = real_run

    check("the service builds a command for the interval it was given",
          isinstance(good, str) and "20 minute" in good, good)
    check("an unworkable interval never reaches the OS through the service",
          isinstance(bad_raised, schedule_mod.ScheduleError)
          and len(stub_wire.calls) == 1,
          {"raised": type(bad_raised).__name__ if bad_raised else None,
           "os_calls": len(stub_wire.calls)})

    cmd = svc.system_task_command(15, "chk-wire")
    check("system_task_command answers with a command, unchanged",
          isinstance(cmd, dict) and "command" in cmd, cmd)

    print()
    print("TOTAL FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else "[]"))
    # The exit code is half the contract. Without it this file prints a page of
    # FAIL lines, exits 0, and every driver that trusts exit status -- including
    # m20_mutate.py -- reads the run as clean. Measured 2026-11-06: all eight
    # mutants "escaped" for exactly this reason while each one printed its
    # failures correctly.
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
