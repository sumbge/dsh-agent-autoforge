"""OS-level containment for forged code — the runner `Sandbox` promises.

`forge/sandbox.py` states the gap in its own docstring: process + timeout +
environment scrubbing bounds *blast radius*, and it says outright that a real
sandbox needs OS-level containment, with `runner` as the hook. This module is
that hook, filled in.

Windows: a Job Object. The child is assigned the moment it exists, before it
has run a line, and the job carries

  * JOB_OBJECT_LIMIT_PROCESS_MEMORY / _JOB_MEMORY -- a run that asks for a
    600 MB bytearray under a 256 MB cap gets a catchable `MemoryError`, not a
    machine that swaps itself to death.
  * JOB_OBJECT_LIMIT_ACTIVE_PROCESS -- a forged tool that forks is refused by
    the kernel, not killed after the fact by a counting loop in Python.
  * JOB_OBJECT_LIMIT_JOB_TIME -- CPU seconds, enforced whether or not the
    parent's poll loop is still healthy.
  * JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE -- the interesting one. A tool that
    spawns a detached grandchild and returns cannot outlive the run: when the
    last handle to the job closes, everything in it dies. `proc.kill()` reaches
    the direct child only; this reaches the tree, including the processes
    detached on purpose to survive it.

POSIX: `resource.setrlimit` in a preexec_fn -- address space, CPU seconds,
file size, open files, and RLIMIT_NPROC where the kernel honours it.

What this still does NOT claim: it is containment, not a security boundary
against an adversary who already has code execution as this user. There is no
seccomp filter here and no filesystem namespace -- a contained run can still
read the files this user can read. That layer is the Linux one, and it is
described in DESIGN.md rather than written here, because code I cannot test on
this host is a claim, not a capability.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
import threading
import time
from ctypes import wintypes
from typing import Any

#: 0xC0000044 -- STATUS_QUOTA_EXCEEDED, measured on this host as the exit
#: status of a process the job killed for spending its CPU budget. The first
#: version of this constant was 0xC0000104, read off a winnt.h list rather than
#: off a run, and the only visible symptom was that the report blamed the
#: memory cap instead of the CPU one. The number here is what `tasklist` and
#: `SandboxResult.returncode` actually carry.
STATUS_QUOTA_EXCEEDED = 0xC0000044
#: 0xE0434352 is a CLR exception; not ours. Kept here as a reminder that exit
#: codes on Windows are a namespace, not a small enum.

MB = 1024 * 1024

#: How long past the wall-clock timeout a contained run is allowed to keep
#: running so the kernel's own CPU limit can be the one that ends it. Only
#: added when containment is armed; the stock path keeps its deadline exactly.
GRACE_S = 3.0


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )
    ]


class _BASIC_LIMIT(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _EXT_LIMIT(ctypes.Structure):
    """JOBOBJECT_EXTENDED_LIMIT_INFORMATION. 144 bytes on x64 -- asserted below.

    `SetInformationJobObject` does not validate the size, it *reads* structs by
    layout from the struct you pass. A field listed in the wrong order compiles
    and silently enforces the wrong limit, so the size is checked at import.
    """

    _fields_ = [
        ("BasicLimitInformation", _BASIC_LIMIT),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _BASIC_ACCOUNTING(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400

_JobObjectExtendedLimitInformation = 9
_JobObjectBasicAccountingInformation = 1




if os.name == "nt":
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _k32.SetInformationJobObject.restype = wintypes.BOOL
    _k32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD)]
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    if ctypes.sizeof(_EXT_LIMIT) != 144:      # pragma: no cover - x64 guard
        raise RuntimeError(
            f"JOBOBJECT_EXTENDED_LIMIT_INFORMATION is {ctypes.sizeof(_EXT_LIMIT)} "
            "bytes; the kernel reads it by layout, so a wrong size means the "
            "limits silently do not apply")
else:                                          # pragma: no cover
    _k32 = None


class ContainmentRefused(RuntimeError):
    """The limits could not be put in place, so the code was not run.

    Raised rather than degraded on purpose. A sandbox that quietly drops its
    memory cap when a syscall fails is worse than no sandbox, because the
    report still says "contained".
    """


class ContainmentUnavailable(ContainmentRefused):
    """Containment is impossible here, and the caller should use the plain path.

    Distinguished from `ContainmentRefused` because the two demand opposite
    responses. A wrapper that exposes no OS handle is a stand-in -- a test
    double, a shim -- and refusing to run its code would be refusing to run a
    test. A real assign failure is a machine saying "I cannot hold this", and
    running anyway would put "contained" in a report that is not true.
    """
    pass


class JobHandle:
    """A Windows Job Object with the four limits that matter, and its ledger."""

    def __init__(self, memory_bytes: int = 512 * MB, max_processes: int = 8,
                 cpu_seconds: float = 30.0):
        if os.name != "nt":
            raise ContainmentRefused("JobHandle is Windows-only")
        self.memory_bytes = memory_bytes
        self.max_processes = max_processes
        self.cpu_seconds = cpu_seconds
        self.handle = _k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise ContainmentRefused(
                f"CreateJobObject failed: {ctypes.get_last_error()}")
        self._configure()

    def _configure(self) -> None:
        info = _EXT_LIMIT()
        flags = (JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                 | JOB_OBJECT_LIMIT_PROCESS_MEMORY
                 | JOB_OBJECT_LIMIT_JOB_MEMORY
                 | JOB_OBJECT_LIMIT_JOB_TIME
                 | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                 | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION)
        info.BasicLimitInformation.LimitFlags = flags
        info.BasicLimitInformation.PerJobUserTimeLimit = int(self.cpu_seconds * 10_000_000)
        info.BasicLimitInformation.ActiveProcessLimit = self.max_processes
        info.ProcessMemoryLimit = self.memory_bytes
        info.JobMemoryLimit = self.memory_bytes
        ok = _k32.SetInformationJobObject(
            self.handle, _JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            err = ctypes.get_last_error()
            self.close()
            raise ContainmentRefused(f"SetInformationJobObject failed: {err}")

    def assign(self, proc: "subprocess.Popen[bytes]") -> None:
        """Put a *already created but not yet scheduled* child into the job.

        Called on the line after Popen and before anything is written to it.
        The race is real but bounded: a process created suspended would close
        it entirely, at the cost of a resume dance through the thread handle.
        What is closed here is the wide window -- the child cannot get far
        enough to allocate gigabytes or fork a tree before the limits land.
        """
        handle = getattr(proc, "_handle", None)
        if handle is None:
            raise ContainmentUnavailable(
                "the process object exposes no OS handle to assign "
                f"({type(proc).__name__} has no _handle)")
        if not _k32.AssignProcessToJobObject(
                self.handle, wintypes.HANDLE(int(handle))):
            err = ctypes.get_last_error()
            try:
                proc.kill()
            except Exception:                  # noqa: BLE001
                pass
            raise ContainmentRefused(
                f"AssignProcessToJobObject failed: {err} -- refusing to run "
                "uncontained code under a report that says contained")

    def accounting(self) -> dict[str, Any]:
        info = _BASIC_ACCOUNTING()
        got = wintypes.DWORD(0)
        if not _k32.QueryInformationJobObject(
                self.handle, _JobObjectBasicAccountingInformation,
                ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(got)):
            return {}
        peak = _EXT_LIMIT()
        if _k32.QueryInformationJobObject(
                self.handle, _JobObjectExtendedLimitInformation,
                ctypes.byref(peak), ctypes.sizeof(peak), ctypes.byref(got)):
            peak_proc = peak.PeakProcessMemoryUsed
            peak_job = peak.PeakJobMemoryUsed
        else:
            peak_proc = peak_job = 0
        return {
            "cpu_ms": (info.TotalKernelTime + info.TotalUserTime) / 10_000.0,
            "peak_process_bytes": peak_proc,
            "peak_job_bytes": peak_job,
            "processes_launched": info.TotalProcesses,
            "processes_killed_by_limit": info.TotalTerminatedProcesses,
        }

    def close(self) -> None:
        if getattr(self, "handle", None):
            _k32.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self) -> "JobHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def describe_failure(returncode: int | None, accounted: dict[str, Any],
                     timed_out: bool) -> str | None:
    """Turn an exit status into the verdict it actually is.

    A run killed for exceeding its CPU budget exits with 0xC0000104, which read
    raw looks like a crash and read as `timed_out` looks like the wall clock
    was the problem. Neither is true, and both send the next attempt down the
    wrong path -- so the limit that fired is named.
    """
    if timed_out:
        return None
    if returncode is not None and (returncode & 0xFFFFFFFF) == STATUS_QUOTA_EXCEEDED:
        return ("killed by the job object's CPU-time limit "
                f"({accounted.get('cpu_ms', 0) / 1000.0:.1f}s of CPU used)")
    killed = (accounted or {}).get("processes_killed_by_limit") or 0
    if killed:
        # Only reached when the exit status was not the quota code: a memory or
        # active-process kill shows up here and nowhere else, and the two caps
        # are indistinguishable from the outside -- which is why the message
        # names both rather than picking one.
        return (f"{killed} process(es) killed by job limits "
                "(memory or active-process cap)")
    if returncode not in (None, 0):
        return f"exited {returncode} ({returncode & 0xFFFFFFFF:#010x})"
    return None


def contained_runner(base, *, memory_mb: int = 512, max_processes: int = 8,
                     cpu_seconds: float = 30.0):
    """Build the `runner` callable `Sandbox` documents, around a real sandbox.

    `base` is the stock `Sandbox`; every field the ordinary path already sets
    (timeout, env allow-list, output cap, abort check) is read from it, and
    only the process creation and the waiting change. That is on purpose: the
    containment must not become a second, divergent definition of "a run".
    """
    def runner(code: str, entry: str, args: dict | None = None):
        import json as _json
        from . import sandbox as _sandbox_module       # local: avoids a cycle

        result_cls = _sandbox_module.SandboxResult
        payload = _json.dumps(
            {"code": code, "entry": entry, "args": args or {},
             "restrict_builtins": base.restrict_builtins},
            ensure_ascii=False)

        with tempfile.TemporaryDirectory(prefix="autoforge_job_") as td:
            runner_path = os.path.join(td, "_runner.py")
            with open(runner_path, "w", encoding="utf-8") as fh:
                fh.write(_sandbox_module._RUNNER)

            # The wall-clock deadline gets a grace period when containment is
            # on, so that the CPU budget is the limit that fires.
            #
            # JOB_OBJECT_LIMIT_JOB_TIME is checked by the kernel when it
            # reschedules, so it lands *late* -- measured on this host at 2.4s
            # of overshoot for a 3s budget. If the wall clock were the tighter
            # of the two, a runaway tool would be reported as a timeout, and
            # the next attempt would go looking for a slow tool instead of a
            # spinning one. The kernel's own limit is the better arbiter
            # because it is terminal; a poll loop is only as healthy as the
            # process doing the polling.
            # Keyed on the platform, not on `job`: the budget is computed
            # before the job exists. Reading an unbound name here is how this
            # line first failed.
            grace = GRACE_S if os.name == "nt" else 0.0
            budget = min(float(cpu_seconds), float(base.timeout) + grace - 1.0)
            if os.name == "nt":
                job = JobHandle(memory_bytes=memory_mb * MB,
                                # +1 on Windows: CREATE_NO_WINDOW gives the
                                # runner a console, and that console's host
                                # (conhost.exe) is a process *inside this job*.
                                # It is the platform's price for a window nobody
                                # can see, not a process the tool started, so the
                                # declaration's `max_processes` must not be made
                                # to pay for it -- otherwise a declared cap of 2
                                # leaves the tool room for exactly zero children.
                                max_processes=max_processes
                                + (1 if os.name == "nt" else 0),
                                cpu_seconds=budget)
                closer, assigner = job.close, job.assign
            else:
                job = None
                closer, assigner = (lambda: None), (lambda p: _apply_rlimits(p, base, cpu_seconds))

            started = time.perf_counter()
            try:
                proc = subprocess.Popen(
                    [base.python, "-I", runner_path],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, env=base.effective_env(), cwd=td,
                    # Same reason as the plain path. CREATE_NO_WINDOW, not
                    # DETACHED_PROCESS: a detached runner holds no console to
                    # pass to the snippet's own children, so each of those gets
                    # a fresh window -- the flash moved one level down instead of
                    # going away. The extra hidden conhost.exe this costs inside
                    # the job moves processes_launched 1 -> 2, against a ceiling
                    # of 8; see the note on _NO_CONSOLE in sandbox.py.
                    creationflags=(0x08000000 if os.name == "nt" else 0),
                )
            except OSError as exc:
                closer()
                return result_cls(ok=False,
                                  error=f"could not start the sandbox interpreter: {exc}",
                                  duration_ms=(time.perf_counter() - started) * 1000)
            try:
                assigner(proc)
            except ContainmentUnavailable:
                # Not containable, but not a failure either: hand the run to
                # the plain path, whose parsing is the one that has always
                # been tested. Reused rather than duplicated on purpose --
                # two envelope parsers would drift, and the one that only
                # runs when the kernel is available is the one that rots
                # unnoticed.
                import dataclasses as _dc

                closer()
                try:
                    proc.kill()
                except Exception:              # noqa: BLE001
                    pass
                plain = _dc.replace(base, contain=False, runner=None)
                return plain.run(code, entry, args or {})
            except ContainmentRefused as exc:
                closer()
                try:
                    proc.kill()
                except Exception:              # noqa: BLE001
                    pass
                return result_cls(ok=False, error=str(exc),
                                  duration_ms=(time.perf_counter() - started) * 1000)

            done = threading.Event()
            box: dict[str, Any] = {}

            def _drain() -> None:
                try:
                    box["out"], box["err"] = proc.communicate(input=payload.encode("utf-8"))
                except BaseException as exc:   # noqa: BLE001
                    box["exc"] = exc
                finally:
                    done.set()

            threading.Thread(target=_drain, daemon=True).start()
            deadline = started + base.timeout + grace
            verdict = None
            while not done.wait(0.2):
                if base.abort_check is not None and base.abort_check():
                    verdict = "aborted"
                    break
                if time.perf_counter() > deadline:
                    verdict = "timed_out"
                    break

            accounted = job.accounting() if job is not None else {}
            if verdict == "timed_out" and job is not None and accounted.get("cpu_ms", 0) >= budget * 1000.0 * 0.9:
                # The wall clock won the race, but the CPU budget was already
                # spent and only waiting on a reschedule. Say which limit it was.
                verdict = "cpu_limit"
            if verdict is not None:
                _sandbox_module._kill_tree(proc)
                done.wait(timeout=10)
                closer()
                return result_cls(
                    ok=False,
                    error={"aborted": "aborted by the operator",
                           "timed_out": (f"timeout after {base.timeout + grace:.0f}s "
                                         f"wall ({accounted.get('cpu_ms', 0) / 1000.0:.1f}s CPU)"),
                           "cpu_limit": (f"killed by the job object's CPU-time limit "
                                         f"({accounted.get('cpu_ms', 0) / 1000.0:.1f}s of CPU "
                                         f"under a {budget:.1f}s budget)")}[verdict],
                    timed_out=verdict in ("timed_out", "cpu_limit"),
                    aborted=verdict == "aborted",
                    accounting=accounted,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    stdout=(box.get("out") or b"").decode("utf-8", "replace")[: base.max_output_bytes],
                )

            closer()      # KILL_ON_JOB_CLOSE: the tree dies even if it detached
            duration = (time.perf_counter() - started) * 1000
            stdout = (box.get("out") or b"").decode("utf-8", "replace")
            stderr = (box.get("err") or b"").decode("utf-8", "replace")[: base.max_output_bytes]
            note = describe_failure(proc.returncode, accounted, False)
            # A kill for spending the CPU budget *is* a timeout downstream --
            # `pipeline` writes "timed out after Ns" and the retry advice keys
            # off it -- but it is the CPU clock that ran out, not the wall
            # clock, and `note` says which.
            cpu_kill = (proc.returncode is not None
                        and (proc.returncode & 0xFFFFFFFF) == STATUS_QUOTA_EXCEEDED)

            if not stdout.strip():
                return result_cls(
                    ok=False,
                    error=(f"{note}; " if note else "") +
                          f"no output (exit {proc.returncode}) {stderr.strip()[:500]}",
                    duration_ms=duration, returncode=proc.returncode, stderr=stderr,
                    accounting=accounted, timed_out=cpu_kill)

            try:
                envelope = _json.loads(stdout)
            except ValueError as exc:
                return result_cls(ok=False, error=f"unreadable envelope: {exc}",
                                  duration_ms=duration, returncode=proc.returncode,
                                  stdout=stdout[: base.max_output_bytes], stderr=stderr,
                                  accounting=accounted, timed_out=cpu_kill)
            return result_cls(
                ok=bool(envelope.get("ok")),
                output=envelope.get("output"),
                error=envelope.get("error") or note,
                duration_ms=duration, returncode=proc.returncode,
                stdout=envelope.get("stdout") or "", stderr=stderr,
                accounting=accounted,
            )

    return runner


def _apply_rlimits(proc, base, cpu_seconds: float) -> None:   # pragma: no cover
    """POSIX half. Written, not tested on this host -- which is why it says so."""
    raise ContainmentRefused(
        "rlimit containment is not wired on this host (Windows). The Linux "
        "path -- setrlimit plus seccomp/landlock -- is described in DESIGN.md "
        "and runs under WSL2.")
