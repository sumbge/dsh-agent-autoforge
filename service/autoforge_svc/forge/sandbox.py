"""Process-isolated execution for forged tools.

Design stance (see DESIGN.md §2.5): a sandbox bounds *blast radius*, it does
not cap *capability*. We isolate by process + timeout + environment scrubbing
rather than by crippling builtins, because an agent that cannot use the
standard library cannot forge useful tools. The host is protected; the tool is
not castrated.

What this buys:
  * a tool that `os._exit()`s or segfaults cannot kill the agent
  * an infinite loop is reaped by timeout
  * a tool that sprays stdout does not corrupt the agent's protocol

What this does NOT claim:
  * it is not a security boundary against adversarial code. `restrict_builtins`
    narrows the surface for that case, but a真 sandbox needs OS-level
    containment (namespaces / job objects). Hook `runner` for that.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: Windows: hide the child's console. CREATE_NO_WINDOW, not DETACHED_PROCESS.
#:
#: DETACHED_PROCESS was the earlier value here, and the reason was a job-object
#: count: CREATE_NO_WINDOW still creates a console, which shows up as an extra
#: hidden conhost.exe inside this sandbox's job object, and that moved the
#: measured `processes_launched` from 1 to 2 (found by test_containment, not by
#: reading). The trade was wrong, because DETACHED_PROCESS creates no console
#: *for the run to pass down*: a snippet that starts a console-subsystem child
#: of its own is then a console-less parent's child and gets a fresh window on
#: the operator's desktop, which is the whole thing being prevented. The extra
#: processes_launched count is a conhost, not the tool's own descendant, and the
#: manifest ceiling that reads it is 8 (32 for the ceiling probe) against a
#: measured 2 -- the headroom absorbs it.
#:
#: Why it matters at all: the agent is started by a scheduled task through
#: pythonw, which has no console, and a console-subsystem child of a
#: console-less parent gets a *fresh* console window on the desktop.
_NO_CONSOLE = 0x08000000 if os.name == "nt" else 0

def _dedent_all(text):
    """Strip four spaces from every line, blank ones included.

    `textwrap.dedent` picks the common prefix from the NON-blank lines only,
    so a literal whose first line begins at column 0 keeps four spaces on
    every line after it. The runner below is exactly that shape and the
    result was an IndentationError on every call -- reported as "no output",
    which names neither the runner nor the indent. Done per line here.
    """
    out = []
    for line in text.split("\n"):
        out.append(line[4:] if line.startswith("    ") else line)
    return "\n".join(out)


_RUNNER = _dedent_all('''    
    import io, json, os, sys, contextlib
    
    # No window for anything this code starts. The runner's console is hidden
    # (CREATE_NO_WINDOW, inherited by everything the snippet spawns), so a
    # console-subsystem child of a snippet is not treated as a console-less
    # parent's child and given a fresh window. Patched here rather than at the
    # call sites, because the call sites are arbitrary snippet code: the number
    # of windows tracked the number of subprocesses a run started, and a run
    # that investigated the flapping was the run that flapped most.
    #
    # Not DETACHED_PROCESS: that creates no console, so the snippet's own
    # children would each get one. The flag is measured, not read. Enumerating
    # visible top-level windows by pid: flags=0 -> one visible
    # PseudoConsoleWindow, CREATE_NO_WINDOW -> none (and DETACHED_PROCESS ->
    # none for the runner itself, while re-creating the problem one level down).
    # `GetConsoleWindow()` is the wrong probe -- it answers non-zero even under
    # DETACHED_PROCESS, where no window is shown.
    if os.name == "nt":
        import subprocess as _sp
        _NO_WINDOW = 0x08000000                     # CREATE_NO_WINDOW
    
        class _QuietPopen(_sp.Popen):
            def __init__(self, *a, **kw):
                kw["creationflags"] = int(kw.get("creationflags") or 0) | _NO_WINDOW
                super().__init__(*a, **kw)
    
        _sp.Popen = _QuietPopen
    
    # The real stdout, captured before anything can replace sys.stdout.
    _RAW_STDOUT = sys.stdout.buffer

    def _emit(obj):
        # Write UTF-8 bytes, not text. See the payload comment below: `-I`
        # makes PYTHONIOENCODING inert, so sys.stdout would encode with the
        # locale codec and blow up (or mojibake) on any non-ASCII result.
        # Envelope goes to the real stdout, one line, always: the snippet's
        # own prints are redirected below and must never reach this stream.
        _RAW_STDOUT.write(
            json.dumps(obj, default=str, ensure_ascii=False).encode("utf-8") + b"\\n")
        _RAW_STDOUT.flush()
    
    def _main():
        try:
            # Read raw bytes and decode UTF-8 ourselves. `-I` (isolated mode)
            # makes PYTHONIOENCODING inert, so `sys.stdin.read()` would use the
            # locale codec -- cp936 on a Chinese Windows host -- to decode a
            # payload the parent wrote as UTF-8. Any non-ASCII argument (a path
            # under 项目_开发, a CJK string) then mangles into a bogus escape
            # and the call dies with "bad payload: Invalid \\escape".
            payload = json.loads(sys.stdin.buffer.read().decode("utf-8") or "{}")
        except Exception as e:
            _emit({"ok": False, "error": f"bad payload: {e}"})
            return
        code = payload["code"]
        entry = payload["entry"]
        args = payload.get("args") or {}
        restrict = payload.get("restrict_builtins", False)
        ns = {"__name__": "__forged__"}
        try:
            if restrict:
                import builtins
                safe = {k: getattr(builtins, k) for k in (
                    "abs","all","any","bool","dict","enumerate","float","int",
                    "len","list","max","min","range","round","set","sorted",
                    "str","sum","tuple","zip","print","Exception","ValueError",
                    "TypeError","KeyError","IndexError","ZeroDivisionError",
                ) if hasattr(builtins, k)}
                ns["__builtins__"] = safe
            # Module-level code runs inside the redirect too. It used to sit
            # outside it, so a print at module scope went to the real stdout
            # and broke the parent's parse with "Extra data". Nothing the
            # snippet prints may reach the wire.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<forged>", "exec"), ns)
            fn = ns.get(entry)
            if fn is None or not callable(fn):
                _emit({
                    "ok": False,
                    "error": f"entry {entry!r} not found or not callable",
                })
                return
            with contextlib.redirect_stdout(buf):
                out = fn(**args)
            _emit({"ok": True, "output": out, "stdout": buf.getvalue()})
        except BaseException as e:  # noqa: BLE001
            _emit({"ok": False, "error": f"{type(e).__name__}: {e}"})
    
    _main()
    ''').strip()


def _kill_tree(proc: "subprocess.Popen[bytes]") -> None:
    """Kill the child *and everything it started*.

    ``proc.kill()`` reaches the direct child only. A forged tool that shells
    out -- or a verification sample that spawns a worker -- leaves those
    grandchildren running, holding the pipes we are about to read from and the
    temp directory we are about to delete. On Windows that is exactly the
    lock that turns a clean abort into a hang, so the tree is killed by pid.
    """
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=10,
            )
        else:
            proc.kill()
    except Exception:                     # noqa: BLE001 - killing is best effort
        pass
    try:
        proc.wait(timeout=5)
    except Exception:                     # noqa: BLE001
        pass


@dataclass
class SandboxResult:
    ok: bool
    output: Any = None
    stdout: str = ""
    error: str | None = None
    timed_out: bool = False
    #: Killed because the operator spoke, not because the tool was slow or
    #: wrong. Kept separate from `timed_out` on purpose: a timeout is a verdict
    #: on the code, and writing it into a tool's failure ledger would punish the
    #: tool for a decision the *person* made.
    aborted: bool = False
    duration_ms: float = 0.0
    returncode: int | None = None
    stderr: str = ""
    #: What the kernel charged this run -- peak bytes, CPU, processes launched.
    #: Empty on the uncontained path, where nothing is measuring. It travels on
    #: the result rather than being read back from the job afterwards because
    #: the job is closed by then (KILL_ON_JOB_CLOSE), and because a result that
    #: can be read by anything is worth more than a handle that must be.
    accounting: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"ok": self.ok, "output": self.output, "error": self.error},
            ensure_ascii=False, default=str,
        )


@dataclass
class Sandbox:
    """Runs forged code out-of-process."""

    timeout: float = 10.0
    python: str = field(default_factory=lambda: sys.executable)
    restrict_builtins: bool = False
    max_output_bytes: int = 200_000
    env_allow: tuple[str, ...] = (
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP",
        "HOME", "USERPROFILE", "LANG", "LC_ALL", "PYTHONPATH",
        # Where a child's own data belongs. Left out, `ToolStore` falls back to
        # `~` and mkdirs `~/autoforge` -- a directory named exactly like the
        # package -- and a home-cwd import of `autoforge` then resolves to that
        # empty directory as a *namespace package*: no `__init__.py`, so the
        # CREATE_NO_WINDOW hook installed there never runs and every subprocess
        # flashes a console window. Measured 2026-09-16; see tests/test_store.py
        # for the pin.
        "APPDATA", "LOCALAPPDATA",
        # Where installed programs live. Left out, `nvidia-smi` starts and then
        # exits 255 with `Failed to initialize NVML: Unknown Error` -- it loads
        # the vendor DLLs through ProgramFiles, and NVML reports a missing path
        # as an unknown hardware error rather than as a missing variable. A/B
        # measured 2026-11-02: the stripped environment fails, adding this one
        # name back makes it answer; `ProgramW6432` (same value) also works,
        # `ProgramFiles(x86)`/`ProgramData`/`SystemDrive` do not. The old note in
        # the report called this a platform limitation -- it was this list.
        "ProgramFiles", "ProgramW6432",
    )
    runner: Callable[[str, str, dict[str, Any]], SandboxResult] | None = None
    #: Asked while a child is running: ``True`` means kill it now and report
    #: `aborted`. It lives on the sandbox rather than in `run`'s signature
    #: because the calls come from deep inside the verifier -- five checks, a
    #: fuzzer's worth of samples each -- and threading a parameter through all
    #: of them would mean touching every call site to say the same thing. Set
    #: it for the span of a run and clear it after.
    abort_check: Callable[[], bool] | None = None
    #: OS-level containment -- a Job Object on Windows, rlimits on POSIX. On by
    #: default where the kernel API exists, because the agent's own reach being
    #: bounded is the point, not an option. `contain=False` is the old path:
    #: process + timeout + environment, which bounds blast radius and nothing
    #: else -- a tool that detaches a grandchild outlives the run.
    contain: bool = field(default_factory=lambda: os.name == "nt")
    #: Run on the Linux side instead, under a real kernel boundary: namespaces,
    #: a tmpfs for the whole visible filesystem, no network, NO_NEW_PRIVS and a
    #: seccomp filter. Off unless asked for, because it needs the WSL distro to
    #: be running and is slower; when it is on and the distro is down, the run
    #: is REFUSED rather than quietly downgraded to the Windows path. A caller
    #: who asked for a boundary and silently got none has been lied to, and
    #: "the tool ran fine" is exactly how that lie reads afterwards.
    isolate: bool = False
    #: Which distro to isolate inside, when `isolate` is on.
    distro: str = "Ubuntu"
    #: The three numbers the job enforces. Read by `reach()` so the report
    #: states limits that were set, not limits that were intended.
    memory_mb: int = 512
    max_processes: int = 8
    cpu_seconds: float = 30.0

    def effective_env(self) -> dict[str, str]:
        """The environment forged code actually gets — not the agent's own.

        Names missing from `env_allow` are dropped, so a command that resolves
        for the agent process (a git-bash `ps`, say) can be unreachable from
        inside the sandbox. Everything the prompt claims about this host is
        built from *this* mapping, never from `os.environ`.

        Dropping a name is not neutral: a program can start, find its own
        executable, and still fail because the *variable* was the thing it
        needed. `nvidia-smi` did exactly that without `ProgramFiles` -- exit
        255 and `Failed to initialize NVML: Unknown Error`, which reads like a
        hardware fault. Adding a name here is cheap; explaining that error is
        not. See the comment in `env_allow`.
        """
        # Case-insensitively, because the names above are written in the shape
        # a person writes them and Windows stores them upper-cased: a literal
        # `k in self.env_allow` silently dropped `ProgramFiles`, and the only
        # symptom was `nvidia-smi` exiting 255 with a NVML hardware error.
        # Measured 2026-11-02. An allow-list that is quietly stricter than it
        # reads is the same defect this project keeps finding; the comparison
        # belongs in one place, spelled out.
        wanted = {n.lower() for n in self.env_allow}
        env = {k: v for k, v in os.environ.items() if k.lower() in wanted}
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def resolve(self, names) -> dict[str, str | None]:
        """For each name, the path it resolves to from inside the sandbox.

        `reachable` answers the weaker question — "is this name on PATH". The
        question that actually matters is the stronger one: can the binary it
        points at answer the question I am about to ask it? On Windows those
        two answers disagree about `ps`: git-bash ships an MSYS `ps` that is on
        PATH and lists MSYS processes only, so a probe built on it reports a
        nearly empty machine and the emptiness reads as "no such process". The
        path is what tells a blind lister from a working one, so the path is
        what this returns.
        """
        import shutil

        path = self.effective_env().get("PATH")
        return {n: shutil.which(n, path=path) for n in names}

    def reachable(self, names) -> tuple[list[str], list[str]]:
        """Split `names` into (resolvable, not) from inside the sandbox."""
        found = self.resolve(names)
        return ([n for n in names if found.get(n)],
                [n for n in names if not found.get(n)])

    def run(self, code: str, entry: str, args: dict[str, Any] | None = None) -> SandboxResult:
        if self.isolate:
            # Checked before `runner`, because `isolate` is the stronger
            # promise: a caller that set both meant the boundary, not the hook.
            from .wsl_isolation import WslUnavailable, runner_for

            try:
                return runner_for(self, distro=self.distro)(code, entry, args or {})
            except WslUnavailable as exc:
                return SandboxResult(
                    ok=False, error=str(exc),
                    duration_ms=0.0, returncode=None,
                )
        if self.runner is not None:
            return self.runner(code, entry, args or {})
        if self.contain:
            # Imported here, not at module scope: the containment module talks
            # to the kernel, and a Sandbox with contain=False must not require
            # that API to exist.
            from .containment import contained_runner

            return contained_runner(
                self, memory_mb=self.memory_mb,
                max_processes=self.max_processes,
                cpu_seconds=self.cpu_seconds,
            )(code, entry, args or {})

        import time

        payload = json.dumps(
            {
                "code": code,
                "entry": entry,
                "args": args or {},
                "restrict_builtins": self.restrict_builtins,
            },
            ensure_ascii=False,
        )

        # ignore_cleanup_errors: a forged tool that detaches a grandchild
        # leaves that child sitting in this directory with its cwd held
        # open, and Windows then refuses the rmdir. Without this the
        # PermissionError escapes `run` -- a tool that followed the rules
        # and returned correctly is reported as a crash of the sandbox.
        # Found by A/B against the contained path, not by reading.
        with tempfile.TemporaryDirectory(
                prefix="autoforge_", ignore_cleanup_errors=True) as td:
            runner_path = os.path.join(td, "_runner.py")
            with open(runner_path, "w", encoding="utf-8") as fh:
                fh.write(_RUNNER)

            env = self.effective_env()

            started = time.perf_counter()
            try:
                proc = subprocess.Popen(
                    [self.python, "-I", runner_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=td,
                    # All three stdio handles are pipes, so the child needs no
                    # console -- and on Windows a console-subsystem child of a
                    # parent that has no console gets a *new* one, with a
                    # visible window. Every sandboxed call would flash one.
                    creationflags=_NO_CONSOLE,
                )
            except OSError as exc:
                return SandboxResult(
                    ok=False,
                    error=f"could not start the sandbox interpreter: {exc}",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )

            # The child is fed and drained on a worker thread; only the *wait*
            # is polled here. That inversion is the whole point: the wait is the
            # long part, and it is the only part the person at the terminal is
            # locked out by. `communicate` is what makes reading both pipes safe
            # without deadlocking on a full buffer, so it stays -- it just no
            # longer owns the thread while it does its job.
            done = threading.Event()
            box: dict[str, Any] = {}

            def _drain() -> None:
                try:
                    box["out"], box["err"] = proc.communicate(
                        input=payload.encode("utf-8"))
                except BaseException as exc:      # noqa: BLE001
                    box["exc"] = exc
                finally:
                    done.set()

            threading.Thread(target=_drain, daemon=True).start()
            deadline = started + self.timeout
            verdict: str | None = None
            while not done.wait(0.2):
                if self.abort_check is not None and self.abort_check():
                    verdict = "aborted"
                    break
                if time.perf_counter() > deadline:
                    verdict = "timed_out"
                    break

            if verdict is not None:
                _kill_tree(proc)
                done.wait(timeout=10)
                return SandboxResult(
                    ok=False,
                    error=("aborted by the operator" if verdict == "aborted"
                           else f"timeout after {self.timeout}s"),
                    timed_out=verdict == "timed_out",
                    aborted=verdict == "aborted",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    stdout=(box.get("out") or b"").decode(
                        "utf-8", "replace")[: self.max_output_bytes],
                )

            duration = (time.perf_counter() - started) * 1000
            stdout = (box.get("out") or b"").decode("utf-8", "replace")
            stderr = (box.get("err") or b"").decode(
                "utf-8", "replace")[: self.max_output_bytes]

            if not stdout.strip():
                return SandboxResult(
                    ok=False,
                    error=f"no output (exit {proc.returncode}) {stderr.strip()[:500]}",
                    duration_ms=duration,
                    returncode=proc.returncode,
                    stderr=stderr,
                )
            # The runner writes exactly one envelope as its last line, but the
            # child is untrusted: a stray module-level statement, a thread, or a
            # print that escaped the redirect can land after it. A bare JSON
            # scalar on the final line used to reach ``data.get("ok")`` and die
            # with ``AttributeError: 'str' object has no attribute 'get'`` --
            # a message naming neither the tool nor the output. Take the last
            # line that is an *object*; otherwise say what actually arrived.
            data: dict[str, Any] | None = None
            for line in reversed(stdout.strip().splitlines()):
                try:
                    cand = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(cand, dict) and "ok" in cand:
                    data = cand
                    break
            if data is None:
                return SandboxResult(
                    ok=False,
                    error=(
                        f"no result envelope in output (exit {proc.returncode}); "
                        f"raw={stdout.strip()[:300]}"
                    ),
                    duration_ms=duration,
                    returncode=proc.returncode,
                    stderr=stderr,
                )
            return SandboxResult(
                ok=bool(data.get("ok")),
                output=data.get("output"),
                stdout=data.get("stdout", ""),
                error=data.get("error"),
                duration_ms=duration,
                returncode=proc.returncode,
                stderr=stderr,
            )


    def reach(self, probe: bool = False) -> dict[str, Any]:
        """What forged code can actually touch — stated, and optionally measured.

        The thing this exists to prevent: an agent that infers its own reach
        from its tool list ("I have no read_file or run_shell, so I cannot touch
        the machine") and then explains the gap with a story about isolation.
        That story is false. Forged code is a subprocess of *this* process, so
        it inherits this machine's filesystem and network. The host is not over
        there behind a wall; it is the ground the sandbox stands on.

        So the honest split is:

          * reach    — a real filesystem and a real network stack, i.e. the
                       things the agent tends to deny it has
          * bounds   — a fresh cwd per call, a scrubbed environment and a
                       timeout, i.e. what limits blast radius without capping
                       capability

        `probe=True` runs an actual round-trip (write a file outside cwd and
        read it back; resolve a hostname) so the report is evidence rather than
        an assertion. Callers that want speed over proof leave it off.
        """
        # Report the environment the child will actually see, not the length of
        # the allow-list. Those two numbers disagree on any host where the
        # agent's own process does not define every allow-listed name (`HOME`,
        # `LANG`, `LC_ALL`, `PYTHONPATH` are routinely absent on Windows), and
        # a child that counts its own `os.environ` then gets a different answer
        # from the one this report promised. Measured 2026-11-01 on this host:
        # `env_allow` was 13 names, `effective_env()` was 10 -- the brief said
        # 13 and every forged tool that measured got 10, which reads as the
        # sandbox having dropped something it never had. State both, so the
        # smaller number is explained rather than contradicted.
        env_effective = self.effective_env()
        # Same case-folding as `effective_env`, for the same reason. Written the
        # obvious way (`n not in env_effective`) this said `ProgramFiles` and
        # `ProgramW6432` were "allow-listed but unset here" while the child had
        # them as `PROGRAMFILES`/`PROGRAMW6432`, and counted those two as names
        # the sandbox injected. The count still came out right by luck; the two
        # lists naming the same variables on both sides of the ledger would not
        # have stayed lucky. Measured 2026-11-02.
        present = {k.lower() for k in env_effective}
        allowed = {n.lower() for n in self.env_allow}
        injected = tuple(k for k in env_effective if k.lower() not in allowed)
        absent = tuple(n for n in self.env_allow if n.lower() not in present)
        defined = len(env_effective) - len(injected)
        env_line = (f"{len(env_effective)} vars reach the child "
                    f"({defined} of {len(self.env_allow)} allow-listed names "
                    "the parent defines"
                    + (f", +{', '.join(injected)}" if injected else "")
                    + "), not the full environment")
        if absent:
            env_line += f"; allow-listed but unset here: {', '.join(absent)}"
        facts: dict[str, Any] = {
            "host": "this machine — the sandbox is a subprocess of the agent",
            "isolated_from_host": False,
            "filesystem": "read-write, whole host (not confined to cwd)",
            "network": "outbound — DNS and sockets",
            "cwd": "a fresh empty temp dir, per call",
            "env": env_line,
            "env_names_present": sorted(env_effective),
            "env_names_absent": list(absent),
            "restrict_builtins": self.restrict_builtins,
            "timeout_s": self.timeout,
            "contained": self._containment_line(),
            # Where the answering code lives. The persona promises that
            # `my_capabilities` reports "射程/记忆/自身源码位置". For two rounds
            # it reported the first two and not the third: a promise in the
            # persona that nothing in the code was checking, which is the same
            # defect class as every other one in this file. Read from the
            # module that knows, so a moved install moves the answer with it.
            "source": self._source_line(),
        }
        if probe:
            facts["probe"] = self._probe()
        return facts

    @staticmethod
    def _source_line() -> str:
        """Where this service's own code is, so the answer survives a move.

        `__file__` of the module that computes reach, plus its mtime. The mtime
        is the part that matters in practice: the question this answers is not
        "where is the code" but "is the code I am reading the code that is
        running", and a timestamp compared against `Get-CimInstance
        Win32_Process CreationDate` settles that in one look.
        """
        import datetime
        here = Path(__file__).resolve()
        try:
            stamp = datetime.datetime.fromtimestamp(here.stat().st_mtime)
            when = stamp.strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            when = "mtime unavailable"
        return f"{here} (mtime {when}); service package {here.parent.parent.name}"

    def _containment_line(self) -> str:
        """What the kernel is holding the run to, said in the same breath as reach.

        Kept next to `isolated_from_host: False` on purpose. Containment does
        not make the host disappear -- a contained run still reads the files
        this user can read. What changes is that it cannot take the machine's
        memory, its process table, or its CPU with it, and it cannot leave a
        detached grandchild behind to outlive the answer it gave.
        """
        if self.runner is not None:
            return "a custom runner (its limits are the runner's business)"
        if not self.contain:
            return ("none -- process + timeout only; a detached grandchild "
                    "outlives the run")
        if os.name == "nt":
            return (f"job object: {self.memory_mb} MB memory cap, "
                    f"{self.max_processes} processes, {self.cpu_seconds}s CPU, "
                    "kill-on-close reaches the whole tree")
        return "rlimits (address space, CPU, file size, open files)"

    @staticmethod
    def _confirm_write(child_detail: Any, parent_tempdir: str, name: str,
                       *, exists: Any = None) -> dict[str, Any]:
        """Which of the two candidate paths this process can actually see.

        Extracted from `_probe` because it could not be observed where it was.
        On this host the child's `tempfile.gettempdir()` and this process's are
        the same string, so the two branches of the old inline code -- "the
        path the child named" and "the path I would have chosen" -- produced the
        identical answer, and two mutants survived for exactly that reason: one
        that hard-coded `confirmed = True`, one that reported `mine` instead of
        what it found. Neither could be seen, because on this machine there was
        nothing to see. `exists` is injectable for the same reason: the
        interesting case is the one where they DISAGREE, and that case has to be
        constructible rather than waited for.

        The disagreement is not hypothetical, but the reason once written here was
        wrong and is corrected: `TEMP` and `TMP` ARE in `env_allow`, so
        `effective_env()` does not drop them. Four constructions were measured on
        2026-11-08 -- as shipped, both names removed, TEMP pointed at a missing
        directory, TEMP pointed at a real second directory -- and the parent and the
        child answered the same path in all four. The mechanism that really
        separates them is `isolate=True`, where the child is not on this filesystem
        at all and answers /tmp while this process answers a Windows path; that
        route could not be exercised here (`wsl -l -q` returns rc=0 with an empty
        distro list). A child that writes into its own directory and reports
        success proves it could write; it does not prove the write landed anywhere
        this process can look -- and under isolation it landed somewhere this
        process cannot look at all.
        """
        probe = exists if exists is not None else os.path.exists
        candidates: list[str] = []
        if isinstance(child_detail, str) and child_detail:
            candidates.append(child_detail)
        mine = os.path.join(parent_tempdir, name)
        if os.path.abspath(mine) not in [os.path.abspath(c) for c in candidates]:
            candidates.append(mine)
        found = next((c for c in candidates if probe(c)), None)
        return {
            "candidates": candidates,
            "found": found,
            "confirmed": bool(found),
        }

    def _probe(self) -> dict[str, Any]:
        """A real round-trip, so `reach` can be checked instead of believed.

        The file is written outside the per-call cwd — writing inside cwd would
        prove nothing about escaping it — and it is written into the CHILD's
        temp dir, resolved inside the child.

        That last part is the fix for a self-report that contradicted itself:
        handing the child the parent's path made it write to its own directory,
        report ok, and leave this process looking at an empty path in another
        directory -- an answer that said "filesystem OK" and "the file is not
        there" in the same breath.

        The cause once named here -- `effective_env()` dropping TMP/TEMP -- is not
        the cause: both names are in `env_allow`, and four environment
        constructions on 2026-11-08 all left the two processes agreeing. The route
        that does separate them is `isolate=True` (a Linux child answers /tmp),
        which this host cannot run: `wsl -l -q` lists no distribution. So the
        child's own reported path is used first because the isolation route exists,
        not because a measured environment difference exists on Windows.

        And the claim is confirmed from OUT HERE. A child that deletes its own
        file and reports success proves it could write; it does not prove the
        write landed where this process can see it. So the child leaves the
        file, the parent looks for it, records what it found, and removes it.
        """
        name = ".autoforge_reach_probe"
        r = self.run(
            "def probe(name):\n"
            "    import os, socket, tempfile, ssl, uuid\n"
            "    out = {}\n"
            "    home = os.path.join(tempfile.gettempdir(), name)\n"
            "    try:\n"
            "        with open(home, 'w', encoding='utf-8') as fh:\n"
            "            fh.write('reach')\n"
            "        back = open(home, encoding='utf-8').read()\n"
            "        out['host_filesystem'] = {'ok': back == 'reach', 'detail': home}\n"
            "    except Exception as e:\n"
            "        out['host_filesystem'] = {'ok': False, 'detail': type(e).__name__ + ': ' + str(e)}\n"
            "    # A resolution is not evidence of a network on a host whose DNS is\n"
            "    # answered by a wildcard interceptor: measured here, a name that\n"
            "    # cannot exist still resolves to 198.18.0.x. So resolve a name that\n"
            "    # NOBODY could have registered and see whether the answer is a lie.\n"
            "    try:\n"
            "        bogus = 'autoforge-probe-' + uuid.uuid4().hex + '.invalid'\n"
            "        socket.gethostbyname(bogus)\n"
            "        dns_intercepted = True\n"
            "    except Exception:\n"
            "        dns_intercepted = False\n"
            "    out['dns_intercepted'] = dns_intercepted\n"
            "    try:\n"
            "        ip = socket.gethostbyname('pypi.org')\n"
            "        out['dns'] = {'ok': True, 'detail': ip,\n"
            "                      'evidence': 'intercepted' if dns_intercepted else 'resolved'}\n"
            "    except Exception as e:\n"
            "        out['dns'] = {'ok': False, 'detail': type(e).__name__ + ': ' + str(e)}\n"
            "    # The claim that survives an intercepted resolver: a TCP connection\n"
            "    # and a real certificate carrying the expected name.\n"
            "    try:\n"
            "        ctx = ssl.create_default_context()\n"
            "        with socket.create_connection(('pypi.org', 443), timeout=8) as sock:\n"
            "            with ctx.wrap_socket(sock, server_hostname='pypi.org') as tls:\n"
            "                cert = tls.getpeercert()\n"
            "                ver = tls.version()\n"
            "        names = [v for k, v in cert.get('subjectAltName', ()) if k == 'DNS']\n"
            "        out['network'] = {'ok': 'pypi.org' in names,\n"
            "                          'detail': str(ver) + ', SAN ' + ', '.join(names[:3]),\n"
            "                          'evidence': 'tcp+tls certificate'}\n"
            "    except Exception as e:\n"
            "        out['network'] = {'ok': False, 'detail': type(e).__name__ + ': ' + str(e)}\n"
            "    return out\n",
            "probe",
            {"name": name},
        )
        if not r.ok:
            return {"error": r.error or "probe did not run"}
        out = r.output if isinstance(r.output, dict) else {"error": f"unexpected {r.output!r}"}

        # Confirm the write from the parent, in every temp dir either side
        # could have meant, and clean up whatever is found. The decision
        # itself lives in `_confirm_write`, where the disagreement between
        # the two temp dirs can be constructed instead of waited for.
        child_detail = (out.get("host_filesystem") or {}).get("detail") \
            if isinstance(out.get("host_filesystem"), dict) else None
        built = self._confirm_write(child_detail, tempfile.gettempdir(), name)
        out["host_filesystem_confirmed_by_parent"] = built["confirmed"]
        out["host_filesystem_seen_at"] = built["found"]
        for cand in built["candidates"]:
            try:
                if os.path.exists(cand):
                    os.remove(cand)
            except OSError:
                pass
        if not built["confirmed"]:
            out["host_filesystem_note"] = (
                "the child reported a successful write but no file was visible "
                "from this process -- treat that report as unproven")
        return out


__all__ = ["Sandbox", "SandboxResult"]
