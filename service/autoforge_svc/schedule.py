"""Scheduling: work that outlives the conversation that asked for it.

The hard part here is not storing a timestamp. It is not lying about what a
schedule can do, because the obvious implementation lies in a way that is easy
to miss:

    A scheduler inside a process can only fire while that process is running.

An agent that accepts "remind me tomorrow at nine" and stores a row it will
never look at has taken an instruction it cannot keep, and the human on the
other end will believe it. So this module draws the line explicitly, and the
tools built on it say which side they are on:

  * `Schedule` is a durable task table. It does nothing on its own. `due()` is
    a *query*: "what is overdue as of now". The agent calls it when it runs.
  * `tick` is that call. Whatever the agent decides to do with the overdue
    tasks is the agent's business, but nothing fires behind its back.
  * `install_system_task` is how the process gets to run at all when no human
    is present — it hands the problem to the operating system (Task Scheduler
    on Windows, cron elsewhere), which genuinely can start a process no one
    asked for. It is a separate, explicit, privileged act, and it prints what
    it registered rather than doing it quietly.

The distinction between the first two and the third is the whole point. A
`Schedule` row is a promise the agent can keep only while it is alive; a system
task is what makes it alive. Conflating them is how an "autonomous" agent ends
up being a to-do list nobody reads.

Missed runs are reported as missed, not skipped. If a task was due at 03:00 and
the agent next runs at 11:00, `due()` says so and by how much: silently
advancing the schedule would erase the evidence that the agent was not running,
which is exactly the fact worth knowing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

#: How many runs of a repeating task are remembered as history.
KEEP_HISTORY = 20

PERIODS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


class ScheduleError(RuntimeError):
    """The request could not be turned into a task."""


def default_schedule_path() -> Path:
    """`$AUTOFORGE_HOME/schedule.jsonl`, alongside the other durable state."""
    home = os.environ.get("AUTOFORGE_HOME")
    base = Path(home) if home else Path.home() / ".autoforge"
    return base / "schedule.jsonl"


def parse_when(when: str | float | int, now: float | None = None) -> float:
    """A due time as an epoch float.

    Accepts what a person or a model actually writes:
      * `+90s`, `+10m`, `+2h`, `+1d`, `+1w` — relative to now
      * `90m` — relative too; the `+` is optional, because it gets omitted
      * `2026-09-14T09:00:00` — naive local, and `...Z` / `+02:00` for explicit
      * a bare epoch number

    Local time for the naive case is deliberate: "nine tomorrow morning" from a
    person means nine where they are, not nine UTC.
    """
    now = time.time() if now is None else now
    if isinstance(when, (int, float)) and not isinstance(when, bool):
        return float(when)

    text = str(when or "").strip()
    if not text:
        raise ScheduleError("no time was given")

    lowered = text.lower().lstrip("+ ")
    if len(lowered) > 1 and lowered[-1] in PERIODS:
        try:
            amount = float(lowered[:-1])
        except ValueError:
            raise ScheduleError(f"could not read {text!r} as a duration") from None
        if amount < 0:
            raise ScheduleError(f"a duration cannot be negative: {text!r}")
        return now + amount * PERIODS[lowered[-1]]

    iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        raise ScheduleError(
            f"could not read {text!r} as a time. Use ISO 8601 "
            f"(2026-09-14T09:00:00) or a duration (10m, 2h, 1d).") from None
    if parsed.tzinfo is None:
        return parsed.astimezone().timestamp()
    return parsed.timestamp()


def as_clock(when: float) -> str:
    """A due time in local time, for anything a human reads."""
    return datetime.fromtimestamp(when).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Task:
    """One scheduled thing. `text` is what the agent should attend to."""

    id: str
    text: str
    due_at: float
    created_at: float
    repeat: float = 0.0            # seconds; 0 means once
    created_by: str = "agent"
    last_run: float = 0.0
    runs: int = 0
    failures: int = 0
    cancelled: bool = False
    notes: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return "repeating" if self.repeat > 0 else "once"

    def overdue_by(self, now: float) -> float:
        return max(0.0, now - self.due_at)

    def line(self, now: float | None = None) -> str:
        now = time.time() if now is None else now
        late = ""
        if now > self.due_at:
            late = f" — OVERDUE by {human_delta(now - self.due_at)}"
        repeat = f", repeats every {human_delta(self.repeat)}" if self.repeat else ""
        return (f"{self.id} [{self.kind}{repeat}] due {as_clock(self.due_at)}{late}: "
                f"{self.text}")


def human_delta(seconds: float) -> str:
    seconds = max(0.0, seconds)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            amount = seconds / size
            return f"{amount:.0f}{unit}" if abs(amount - round(amount)) < 0.05 else f"{amount:.1f}{unit}"
    return f"{seconds:.0f}s"


class Schedule:
    """A durable task table. Stores; never fires."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else default_schedule_path()
        self._tasks: dict[str, Task] = {}
        self.load_error: str = ""
        self.load()

    # -- persistence ---------------------------------------------------
    def load(self) -> None:
        """Read the table. A damaged line is skipped, not fatal.

        One bad line should cost its own task. Refusing to load the whole file
        would mean a crash while appending takes every future task with it.
        """
        self._tasks = {}
        self.load_error = ""
        if not self.path.exists():
            return
        bad = 0
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            self.load_error = f"could not read {self.path}: {exc}"
            return
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                task = Task(**data)
            except (ValueError, TypeError):
                bad += 1
                continue
            self._tasks[task.id] = task
        if bad:
            self.load_error = f"{bad} unreadable line(s) in {self.path} were skipped"

    def save(self) -> None:
        """Rewrite the file atomically: a crash mid-write must not lose the table."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        text = "".join(json.dumps(asdict(t), ensure_ascii=False) + "\n"
                       for t in self._tasks.values())
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)

    # -- reading -------------------------------------------------------
    def all(self) -> list[Task]:
        return sorted(self._tasks.values(), key=lambda t: t.due_at)

    def active(self) -> list[Task]:
        return [t for t in self.all() if not t.cancelled]

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def due(self, now: float | None = None) -> list[Task]:
        """Overdue and not cancelled, oldest first. A query, not a trigger."""
        now = time.time() if now is None else now
        return [t for t in self.active() if t.due_at <= now]

    def next_due(self, now: float | None = None) -> Task | None:
        now = time.time() if now is None else now
        upcoming = [t for t in self.active() if t.due_at > now]
        return upcoming[0] if upcoming else None

    # -- writing -------------------------------------------------------
    def add(self, text: str, when: str | float | int, repeat: str | float | int = 0,
            created_by: str = "agent", now: float | None = None) -> Task:
        if not str(text or "").strip():
            raise ScheduleError("a task needs something to do — the text was empty")
        now = time.time() if now is None else now
        due_at = parse_when(when, now)
        interval = 0.0
        if repeat not in (0, "0", "", None):
            interval = parse_when(repeat, now) - now
            if interval <= 0:
                raise ScheduleError(f"a repeat interval must be positive, got {repeat!r}")
        task = Task(id=uuid.uuid4().hex[:8], text=str(text).strip(), due_at=due_at,
                    created_at=now, repeat=interval, created_by=created_by)
        self._tasks[task.id] = task
        self.save()
        return task

    def complete(self, task_id: str, ok: bool = True, note: str = "",
                 now: float | None = None) -> Task:
        """Record a run. A repeating task is advanced; a one-shot is closed.

        `note` is the point: the next run wants to know what happened last time,
        and "it failed with this error" is the only thing that makes a repeated
        failure distinguishable from a long silence.
        """
        now = time.time() if now is None else now
        task = self._tasks.get(task_id)
        if task is None:
            raise ScheduleError(f"no task {task_id!r}")
        task.runs += 1
        task.last_run = now
        if not ok:
            task.failures += 1
        if note:
            task.notes.append(note[:500])
            del task.notes[:-10]
        task.history.append({"at": now, "ok": ok, "note": note[:200]})
        del task.history[:-KEEP_HISTORY]

        if task.repeat > 0:
            # Advance from the *scheduled* time, not from now. Advancing from
            # now would let a late run drift the whole series later and later,
            # so "every morning at nine" becomes midday over a few weeks. Skip
            # whole periods if several were missed, but never spin past now
            # without landing on a real point in the series.
            steps = max(1, int((now - task.due_at) // task.repeat) + 1)
            task.due_at += steps * task.repeat
        else:
            task.cancelled = True
        self.save()
        return task

    def cancel(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise ScheduleError(f"no task {task_id!r}")
        task.cancelled = True
        self.save()
        return task

    def forget(self, task_id: str) -> bool:
        """Remove a task entirely. Returns whether it was there."""
        gone = self._tasks.pop(task_id, None) is not None
        if gone:
            self.save()
        return gone

    def report(self, now: float | None = None) -> str:
        now = time.time() if now is None else now
        lines: list[str] = []
        if self.load_error:
            lines.append(f"({self.load_error})")
        upcoming = self.active()
        if not upcoming:
            lines.append("Nothing scheduled.")
            return "\n".join(lines)

        overdue = [t for t in upcoming if t.due_at <= now]
        if overdue:
            lines.append(f"{len(overdue)} task(s) OVERDUE:")
            lines += [f"  {t.line(now)}" for t in overdue]
        later = [t for t in upcoming if t.due_at > now]
        if later:
            lines.append(f"{len(later)} upcoming:")
            lines += [f"  {t.line(now)}" for t in later]
        lines.append(
            "A task only fires while I am running: schedule_tick is what reads "
            "these, and install_system_task is what starts me when nobody has."
        )
        return "\n".join(lines)


# ----------------------------------------------------------------------
# handing the problem to the operating system
# ----------------------------------------------------------------------
#: The command the OS should run to wake the agent up. Kept as an argv list so
#: nothing here depends on a shell's quoting rules.
def wake_command(interval_minutes: int = 30, script: str | None = None) -> list[str]:
    """What the OS should run. `auto` with no arguments is enough."""
    exe = sys.executable
    if script:
        return [exe, script]
    # The module name has to be the one THIS package is importable as. The
    # upstream value "autoforge" was correct for `autoforge/`, and this file is
    # a copy of it living in `autoforge_svc/`, so the inherited literal emitted
    # a command that dies with "No module named autoforge" every time the OS ran
    # it. Measured 2026-10-26 on a registered task; `__package__` is what keeps
    # a future copy from repeating the mistake.
    return [exe, "-m", __package__ or "autoforge_svc", "tick", "--quiet"]


def system_task_command(interval_minutes: int = 30, name: str = "autoforge-tick",
                        platform: str | None = None, script: str | None = None,
                        ) -> list[str]:
    """The command that registers a recurring run, for this platform.

    Returned rather than run, so the caller can show it before doing something
    persistent to the machine. `install_system_task` is the thing that runs it.
    """
    plat = (platform or sys.platform).lower()
    interval = max(1, int(interval_minutes))
    argv = wake_command(interval, script)

    if plat.startswith("win"):
        quoted = " ".join(f'"{a}"' for a in argv)
        return ["schtasks", "/Create", "/TN", name, "/SC", "MINUTE",
                "/MO", str(interval), "/TR", quoted, "/F"]
    if plat.startswith(("linux", "darwin")):
        # cron cannot express sub-minute intervals, and its granularity is
        # minutes anyway, which is what /MO is for above.
        step = f"*/{interval}" if interval < 60 else "0"
        hour = "*" if interval < 60 else f"*/{max(1, interval // 60)}"
        line = (f"{step if interval < 60 else '0'} {hour} * * * "
                f"{' '.join(argv)} # {name}")
        return ["crontab", "-l"], line
    raise ScheduleError(f"no system-task mechanism is known for platform {plat!r}")


def _console_text(raw: bytes | str | None) -> str:
    """Text out of an OS command whose encoding is whatever the console uses.

    Windows tools answer in the console code page (cp936 on a Chinese install,
    cp850 elsewhere), and a POSIX locale can hand back bytes in anything. The
    text is only for a human to read, so the right move on an undecodable byte
    is a replacement character, not an exception three frames from here.
    """
    if not raw:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    for encoding in ("utf-8", "mbcs", "cp936", "latin-1"):
        try:
            return raw.decode(encoding).strip()
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace").strip()


def install_system_task(interval_minutes: int = 30, name: str = "autoforge-tick",
                        platform: str | None = None) -> str:
    """Register the recurring run with the OS. Says what it did, or why not.

    This is the one action in the module that changes the machine's
    configuration, so it is explicit, reversible, and reported. It is also the
    only way an agent can be said to run "unattended" — everything else here
    needs the process to already be alive.
    """
    plat = platform or sys.platform
    command = system_task_command(interval_minutes, name, plat)

    if plat.lower().startswith("win"):
        try:
            # Bytes, not `text=True`. schtasks answers in the console's code
            # page, which on a Chinese Windows is GBK -- decoding it as UTF-8
            # raises UnicodeDecodeError inside subprocess's reader thread, so
            # the failure that reaches us is a decode crash rather than "the
            # scheduler refused". Replace undecodable bytes instead: the exit
            # code is the verdict, and the text is only ever shown to a human.
            proc = subprocess.run(command, capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ScheduleError(
                f"could not register the task with Task Scheduler: "
                f"{type(exc).__name__}: {exc}") from exc
        if proc.returncode != 0:
            detail = _console_text(proc.stderr or proc.stdout)[:400]
            raise ScheduleError(
                f"Task Scheduler refused the task (exit {proc.returncode}): {detail}")
        return (f"Registered the Windows scheduled task {name!r} to run every "
                f"{interval_minutes} minute(s):\n  {' '.join(command)}\n"
                f"Remove it with: schtasks /Delete /TN {name} /F")

    if plat.lower().startswith(("linux", "darwin")):
        # Editing a crontab is not something to do blind: the only safe move is
        # to print the line and let the operator add it, because a botched
        # rewrite takes their other jobs with it. The honest version of this
        # action is a command, not a mutation.
        _, line = command
        return (f"Add this line to your crontab (`crontab -e`) to run every "
                f"{interval_minutes} minute(s):\n  {line}\n"
                f"(Not installed automatically: rewriting a crontab in place "
                f"risks the jobs already in it.)")

    raise ScheduleError(f"no system-task mechanism is known for platform {plat!r}")


__all__ = [
    "Schedule", "ScheduleError", "Task", "parse_when", "as_clock", "human_delta",
    "default_schedule_path", "wake_command", "system_task_command",
    "install_system_task", "PERIODS",
]
