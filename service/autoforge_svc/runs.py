"""A run that survives the wire going down.

The failure this exists for, stated as it happened: a task is under way, the
model call in flight comes back `503`, and the run dies. Everything about that
outcome is wrong in a way no other module here can fix.

* The retry ladder is measured in seconds. `OpenAICompatClient` retries a
  transient status `max_attempts` times with a 2x backoff capped at 30s, which
  is about thirty seconds of riding out a burst. The outages this machine
  actually sees -- aiping.cn 503s, a dropped route to github.com -- last
  minutes. Riding out a burst and surviving an outage are different jobs, and
  the ladder can only do the first.
* The failover chain needs somewhere to fail over *to*. On this host both the
  primary and the configured fallback are the same gateway, and when the
  gateway itself is the thing that is down, a second endpoint on it is not a
  second endpoint.
* Nothing remembers where the loop was. `ForgeAgent.run` builds a fresh
  `Agent` per task, and the message list lives in a local variable inside
  `_run`. When the process raises, or is killed, or the machine reboots, the
  transcript, the turn count and the pending task all die with it. The operator
  is left holding a task that was half-done and an agent that has no record it
  was ever started.

So this module writes down where the loop was, often enough that a resume is
cheap and rarely enough that it costs nothing: one small json file per in-flight
run, rewritten atomically at the turn boundary -- the one point in the loop
where the message list is guaranteed to hold whole groups (an assistant message
and every result answering it) and is therefore a request that can be replayed
rather than a request that would be malformed.

Three decisions worth their reasons:

*The checkpoint records, it does not decide.* `claim` only ever answers "may I
  take this over, and on what grounds"; it does not know how a model is called
  or what a task is. Keeping the judgement in one place is what makes it
  testable without a network, and it is why `scan` can be run by a process that
  has no model at all.

*Staleness is judged on the writer's process, not on the clock.* The same rule
  `lanes` and `bus.presence` already use here: a heartbeat is a claim about a
  process, and `lanes.pid_alive` is the only thing on this host that tells
  "quiet" from "dead" correctly (on Windows `os.kill(pid, 0)` checks rather
  than signals). A timer alone would either steal a live run that was thinking
  for three minutes, or wait forever on a run whose process was killed in the
  first second.

*A resume is bounded.* A run that is resumed and dies again, three times, is
  not a run waiting for the network -- it is a run that fails for a reason the
  network has nothing to do with, and re-entering it forever is a loop that
  spends tokens to reach the same wall. After `MAX_RESUMES` the record is
  reported `stale` with the count in the reason, which is the one fact that
  separates a repeated failure from a long silence. That is the same
  distinction `schedule.Schedule.complete` keeps a note for.

What this is not: not a scheduler (`schedule.py` answers "when should this
run"), not the record of what is owed (`mission.py`), and not a job queue. It
is the answer to one question -- where was the loop, this second -- and it is
deliberately small enough to be read by eye.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .core.message import Message

#: Bumped when the on-disk shape changes in a way an older reader would
#: misread. A record from a newer schema is refused rather than guessed at:
#: resuming on a misread transcript is worse than not resuming, because the
#: model is handed a conversation that never happened.
SCHEMA = 1

#: How many times one task may be picked up and re-entered. See the module
#: docstring -- past this, a failure is a property of the task, not of the wire.
MAX_RESUMES = 3

#: How long after the writer stops before its record may be taken over. Not a
#: wait for the retry ladder -- the ladder lives inside the writer process and
#: died with it, so there is nothing left to race. This is a debounce against
#: two resumers, which is a real hazard rather than a theoretical one: the OS
#: task and an operator at the terminal are two processes looking at the same
#: directory, and both would otherwise see the same stranded run and start it.
#: It is a partial answer on its own; `Journal.adopt` is the rest of it, since
#: a takeover rewrites the record with the new writer's pid before any work
#: begins and the second scanner then reads it as live.
SETTLE_S = 5.0

#: How long a writer may look alive and still fail to beat before it is called
#: wedged. Alive is not the same as progressing: the failure this module is for
#: includes a thread parked on a socket that will never answer, and deferring to
#: a pid forever is how a run that needs resuming never gets one. Generous
#: because the cost of being wrong is a task running twice, and a beat is one
#: small write per turn -- a genuinely working run is nowhere near this quiet.
WEDGED_AFTER_S = 300.0

#: A record this old is not a resume candidate whatever else is true of it. A
#: week matches `mission.RESURRECT_WINDOW_S`, on purpose: the two stores answer
#: questions about the same work, and a run that outlives the mission it was
#: for is a run nobody is waiting for.
DEFAULT_TTL_S = 7 * 24 * 3600.0

#: Where the records live, relative to `store._default_home()`. A directory
#: rather than one file, because two sessions on this host run tasks at the
#: same time and a shared append-only file would make "my run" a question
#: nobody can answer without parsing everyone else's.
_SUBDIR = "runs"


def _home() -> str:
    """The same home the store, the bus and the schedule use.

    Borrowed from `store._default_home` rather than re-derived, and imported
    inside the function because that name is private and `store` pulls in the
    tool spec machinery -- this module is read by a resume path that may be
    running when something else is already broken, so it keeps its own import
    list to the three things it cannot do without.
    """
    try:
        from .store import _default_home

        return _default_home()
    except Exception:                                     # noqa: BLE001
        return os.path.join(os.path.expanduser("~"), ".autoforge")


def journal_dir(root: str | os.PathLike[str] | None = None) -> Path:
    """The directory holding the records. `$AUTOFORGE_RUNS` wins."""
    if root is not None:
        return Path(root)
    override = os.environ.get("AUTOFORGE_RUNS")
    if override:
        return Path(override)
    return Path(_home()) / _SUBDIR


def _path(run_id: str, root: str | os.PathLike[str] | None = None) -> Path:
    return journal_dir(root) / f"run-{run_id}.json"


def _atomic_write(path: Path, text: str) -> None:
    """Write the whole record or leave the old one alone.

    `os.replace` is atomic on this platform, so a reader never sees half a
    record and a crash mid-write leaves the previous heartbeat standing rather
    than a truncated file that parses as nothing. The obvious alternative --
    truncate and rewrite -- has exactly the failure this module exists to
    survive, one level down.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:                                   # a pipe, not a disk
            pass
    os.replace(tmp, path)


def _read(path: Path) -> dict[str, Any] | None:
    """Parse one record, or None. A damaged record is skipped, never fatal.

    Same rule the schedule table follows: one unreadable entry must cost its
    own task and not the whole list, or a crash while writing takes every
    future resume with it.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _messages(record: dict[str, Any]) -> list[Message]:
    raw = record.get("messages")
    if not isinstance(raw, list):
        return []
    return [Message.from_dict(m) for m in raw]


def repair(msgs: Sequence[Message]) -> tuple[list[Message], int]:
    """Trim a trailing group that was interrupted mid-flight. Returns (kept, dropped).

    A transcript is a sequence of whole groups -- an assistant message and every
    result answering it -- and a request built from a *partial* group is not a
    shorter request, it is a malformed one. OpenAI-compatible endpoints reject
    an assistant `tool_calls` entry with no matching `role: tool` message, and
    the ones that do not reject it answer with the results invented, which is
    worse: it looks like progress.

    So the tail is trimmed back to the last whole group. This is not a loss of
    care but the only correct move: the work in the dropped group did not
    finish, so asking the model to continue from it would be asking it to
    continue from a step whose result nobody knows. At most one group goes, and
    the count is kept so the resume prompt can say a step has to be redone
    rather than pretending the interruption cost nothing.

    Trimming at the *reader* rather than at the writer is deliberate: the writer
    is the code that is dying, and a repair step on the way out is exactly the
    code that does not get to run.
    """
    kept = [m for m in msgs if m.role != "system"]
    dropped = 0
    while kept:
        last = kept[-1]
        if last.role == "user":
            break                                  # a task: always replayable
        if last.role == "assistant" and not last.tool_calls:
            break                                  # a finished answer
        if last.role == "assistant":
            kept.pop(); dropped += 1               # asked for tools, got none
            continue
        if last.role == "tool":
            # Walk back over the run of results to the assistant that asked.
            ids = set()
            i = len(kept) - 1
            while i >= 0 and kept[i].role == "tool":
                if kept[i].tool_call_id:
                    ids.add(kept[i].tool_call_id)
                i -= 1
            if i >= 0 and kept[i].role == "assistant":
                asked = {tc.id for tc in kept[i].tool_calls}
                # Both directions, because both are rejections: a call with no
                # result, and a result for a call that was never made. The
                # second cannot arise from this loop, but it can arise from a
                # record somebody edited, and a resume is not the place to
                # discover it.
                if asked and asked <= ids:
                    for j in range(len(kept) - 1, i, -1):
                        if kept[j].tool_call_id not in asked:
                            kept.pop(j); dropped += 1
                    break                          # a whole group: safe to send
            while kept and kept[-1].role == "tool":
                kept.pop(); dropped += 1
            continue
        kept.pop(); dropped += 1                   # anything unrecognised
    return kept, dropped


def _continuable(msgs: Sequence[Message]) -> bool:
    """Whether this transcript holds a task to continue, once repaired.

    The only shape that is genuinely unrunnable is one with no user message in
    it: there is then no task to be in the middle of, and "resuming" would be
    starting, which is a different thing pretending to be this one.
    """
    kept, _ = repair(msgs)
    return any(m.role == "user" for m in kept)


@dataclass
class Mark:
    """One in-flight run, as written down.

    A dataclass rather than a dict at the boundary so that a field added later
    cannot be silently read as its default by a caller that meant the real
    thing -- `from_dict` is the only place that tolerates a missing key.
    """

    run_id: str
    task: str
    started_at: float
    updated_at: float
    writer_pid: int
    turns: int = 0
    attempts: int = 0
    schema: int = SCHEMA
    session: str = ""
    cwd: str = ""
    last_action: str = ""
    prior_run_id: str = ""
    messages: list[Message] = field(default_factory=list)
    #: How many messages were trimmed off the tail by `repair` on the way in.
    #: Not stored on disk -- it is a property of this reading, not of the
    #: record -- which is why it is not in `to_dict`.
    dropped_steps: int = 0

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "task": self.task,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "writer_pid": self.writer_pid,
            "turns": self.turns,
            "attempts": self.attempts,
            "session": self.session,
            "cwd": self.cwd,
            "last_action": self.last_action,
            "prior_run_id": self.prior_run_id,
            # The system prompt is deliberately NOT here. It is re-rendered by
            # the agent that resumes, because this machine's prompt carries the
            # mission list, the kept facts and the host probe -- all of which are
            # recomputed every turn on purpose. Storing it would resume a run
            # against a three-day-old agenda, which is the one thing the mission
            # block exists to prevent, and it would put a megabyte of text into
            # every heartbeat. What is stored is the conversation from the first
            # user message onward: the part that cannot be re-derived.
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mark":
        # Repaired here, at the one boundary where a record written by another
        # process is interpreted, so that everything downstream can assume it
        # holds whole groups. A reader that repaired later would have to be
        # every reader.
        kept, dropped = repair(_messages(data))
        return cls(
            run_id=str(data.get("run_id") or ""),
            task=str(data.get("task") or ""),
            started_at=float(data.get("started_at") or 0.0),
            updated_at=float(data.get("updated_at") or time.time()),
            writer_pid=int(data.get("writer_pid") or 0),
            turns=int(data.get("turns") or 0),
            attempts=int(data.get("attempts") or 0),
            schema=int(data.get("schema") or SCHEMA),
            session=str(data.get("session") or ""),
            cwd=str(data.get("cwd") or ""),
            last_action=str(data.get("last_action") or ""),
            prior_run_id=str(data.get("prior_run_id") or ""),
            messages=kept,
            dropped_steps=dropped,
        )

    # -- questions the reader actually has ------------------------------
    @property
    def age(self) -> float:
        return max(0.0, time.time() - self.updated_at)

    @property
    def owner_alive(self) -> bool:
        """Whether the process that wrote this heartbeat still exists.

        `lanes.pid_alive` rather than a timestamp comparison, and imported here
        rather than re-implemented: the first version of that check on this host
        treated *any* exception from `os.kill(pid, 0)` as dead, which reads a
        live process owned by another user as a corpse.
        """
        if self.writer_pid <= 0:
            return False
        try:
            from .lanes import pid_alive

            return bool(pid_alive(self.writer_pid))
        except Exception:                                 # noqa: BLE001
            # Unanswerable is not the same as dead. Refusing to take over a run
            # whose owner cannot be checked costs a resume; stealing a live
            # run's task costs two runs editing one workspace.
            return True

    def claim(self, *, now: float | None = None, ttl_s: float = DEFAULT_TTL_S,
              settle_s: float = SETTLE_S,
              wedged_after_s: float = WEDGED_AFTER_S,
              max_resumes: int = MAX_RESUMES) -> tuple[str, str]:
        """May this record be taken over, and on what grounds?

        Returns `(status, reason)`. Nothing here is a side effect, which is what
        makes it callable from a scan over every record and from a test with no
        network and no process table.

          resumable -- the writer is gone or wedged, the transcript is
                       grounded, and the resume budget is not spent.
          live      -- the writer is alive and beating. Leave it alone.
          waiting   -- the writer has just stopped. A second resumer may be
                       starting on the same record; see `SETTLE_S`.
          stale     -- the resume budget is spent, or the record is past its
                       TTL, or the transcript cannot be continued.

        Order matters. The three `stale` answers are decided first because they
        are properties of the record itself and no amount of waiting changes
        them -- and a caller told "waiting" about a record that will never be
        resumable would keep looking.
        """
        now = time.time() if now is None else now
        age = max(0.0, now - self.updated_at)

        if age > ttl_s:
            return ("stale", f"abandoned {_human(age)} ago, past the "
                             f"{_human(ttl_s)} limit for a resume")
        if self.attempts >= max_resumes:
            return ("stale", f"picked up and re-entered {self.attempts} time(s) "
                             "already and did not get through; this is failing "
                             "for a reason the wire is not part of")
        if not _continuable(self.messages):
            return ("stale", "the transcript holds no task to continue from")

        if not self.owner_alive:
            if age < settle_s:
                return ("waiting", f"the writer stopped {_human(age)} ago -- "
                                   "letting the record settle so two resumers "
                                   "do not start the same run")
            return ("resumable", f"the writer (pid {self.writer_pid}) is gone; "
                                 f"stopped {_human(age)} ago after "
                                 f"{self.turns} turn(s)")

        if age < wedged_after_s:
            return ("live", f"pid {self.writer_pid} is running this, last beat "
                            f"{_human(age)} ago")
        return ("resumable", f"pid {self.writer_pid} is alive but has not beaten "
                             f"for {_human(age)}, so it is treated as wedged")

def _human(seconds: float) -> str:
    seconds = max(0.0, seconds)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            amount = seconds / size
            return f"{amount:.1f}{unit}" if amount % 1 else f"{amount:.0f}{unit}"
    return f"{seconds:.0f}s"


# ---------------------------------------------------------------------------
class Journal:
    """One task's place in the conversation, kept where losing it is visible.

    The write pattern is deliberately dull: `beat` is called at the turn
    boundary, rewrites the whole small record atomically, and returns nothing.
    A checkpoint that required the caller to decide what to save would be
    called inconsistently, and the call that got forgotten would be the one
    covering the expensive turn.
    """

    def __init__(self, task: str, *, root: str | os.PathLike[str] | None = None,
                 session: str = "", cwd: str = "") -> None:
        self.root = root
        self.task = task
        self.session = session
        self.cwd = cwd
        self.mark = Mark(
            run_id=uuid.uuid4().hex[:12],
            task=task,
            started_at=time.time(),
            updated_at=time.time(),
            writer_pid=os.getpid(),
            session=session,
            cwd=cwd,
        )

    # -- writing --------------------------------------------------------
    def beat(self, msgs: Sequence[Message], *, turn: int = 0,
             last_action: str = "") -> Mark:
        """Write down where the loop is. Called at every turn boundary.

        `msgs` is the *whole* live list including its system message, because
        that is what the caller holds; the system message is dropped here and
        not by every caller, since "which messages are the conversation" is a
        rule that belongs with the format that stores it.
        """
        self.mark.messages = [m for m in msgs if m.role != "system"]
        self.mark.turns = int(turn)
        self.mark.updated_at = time.time()
        self.mark.writer_pid = os.getpid()
        if last_action:
            self.mark.last_action = str(last_action)[:120]
        _atomic_write(_path(self.mark.run_id, self.root),
                      json.dumps(self.mark.to_dict(), ensure_ascii=False))
        return self.mark

    def adopt(self, mark: Mark, *, attempt_of: str = "") -> "Journal":
        """Take over a stranded record, and rewrite it as mine before working.

        Two jobs, and the second is the reason this exists rather than a bare
        assignment. It carries the transcript and the turn count forward -- a
        resume that started from an empty conversation would be a retry wearing
        a resume's clothes -- and it *claims the record*, by writing the new
        writer pid and a fresh heartbeat immediately. From that write on, any
        other scanner reads this record as live and leaves it alone. Without it
        the debounce in `SETTLE_S` would be the only protection, and two
        resumers that start within the same few seconds would both pass it.

        `attempt_of` names the record this one takes over, so the chain of
        attempts is readable in the new record. The chain is why `attempts`
        increments: a task that has been entered three times and failed three
        times is not waiting for a network, and this is the count that says so.
        """
        self.mark = Mark(
            run_id=uuid.uuid4().hex[:12],
            # The task this Journal was constructed with wins over the one in
            # the record. That is the whole point of the constructor argument at
            # this call site: an operator who has since said something different
            # must not be faithfully resumed onto an abandoned instruction --
            # and the record is the older of the two by definition.
            task=self.task or mark.task,
            started_at=mark.started_at or time.time(),
            updated_at=time.time(),
            writer_pid=os.getpid(),
            turns=mark.turns,
            attempts=mark.attempts + 1,
            session=self.session or mark.session,
            cwd=self.cwd or mark.cwd,
            last_action=mark.last_action,
            prior_run_id=attempt_of or mark.run_id,
            messages=list(mark.messages),
        )
        # Written now, not at the first beat: the claim has to be on disk
        # before the work starts, or it is not a claim.
        _atomic_write(_path(self.mark.run_id, self.root),
                      json.dumps(self.mark.to_dict(), ensure_ascii=False))
        # And the record it replaces is removed, because a takeover that left
        # the original standing would not have taken anything over: the next
        # scan would read the same stranded record and resume the same task a
        # second time, in parallel, on one workspace. The attempt count and the
        # prior_run_id are the chain that survives the deletion, which is why
        # they are carried on the new record rather than left behind on the old.
        # Nothing is lost that a reader needs: what this module stores is a note
        # to the *next* process, and the durable record of what happened is the
        # ledger entry every run already writes.
        if mark.run_id and mark.run_id != self.mark.run_id:
            try:
                _path(mark.run_id, self.root).unlink()
            except OSError:
                pass
        return self

    def finish(self) -> None:
        """Remove the record: this run is not a resume candidate any more.

        Deleted rather than marked, unlike a mission or a schedule entry, and
        the difference is the point. A mission is a claim about what is owed and
        must stay queryable after it is done; this is a note to *this* process
        about where it was, and a finished task's note is noise in every later
        scan. What survives the run is the ledger entry `ForgeAgent` already
        writes on finish, so "it completed" is still provable -- just not from
        here.
        """
        try:
            _path(self.mark.run_id, self.root).unlink()
        except OSError:
            pass

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # A clean exit closes the record; an exception leaves it standing,
        # because that is exactly the case a resume exists for. Returning False
        # so nothing about the exception is swallowed here -- a checkpoint that
        # ate the error it was recording would be the worst of both.
        if exc_type is None:
            self.finish()
        return False


def unfinished(root: str | os.PathLike[str] | None = None) -> list[Mark]:
    """Every record on disk, newest heartbeat first.

    A record too new to parse is skipped. A record written by a SCHEMA this
    code does not know is *kept* in the list and reported by `scan` as stale
    with that reason, rather than dropped -- a resume that refuses is a fact the
    operator can act on, and a resume that silently forgets is not.
    """
    out: list[Mark] = []
    for path in sorted(journal_dir(root).glob("run-*.json")):
        data = _read(path)
        if data is None:
            continue
        mark = Mark.from_dict(data)
        if not mark.run_id:
            continue
        out.append(mark)
    out.sort(key=lambda m: m.updated_at, reverse=True)
    return out


def scan(root: str | os.PathLike[str] | None = None, *, now: float | None = None,
         ttl_s: float = DEFAULT_TTL_S,
         settle_s: float = SETTLE_S) -> list[dict[str, Any]]:
    """Every record with the decision `claim` makes about it, as data.

    The reader is a different process from the writer by construction, so the
    answer has to be printable by something that has no model and no task: a
    tick that finds nothing to do, or an operator asking why last night's work
    stopped halfway.
    """
    rows: list[dict[str, Any]] = []
    for mark in unfinished(root):
        status, reason = mark.claim(
            now=now, ttl_s=ttl_s, settle_s=settle_s)
        if mark.schema != SCHEMA:
            status, reason = ("stale",
                              f"written by schema {mark.schema}, this build reads "
                              f"{SCHEMA}")
        rows.append({
            "run_id": mark.run_id,
            "task": mark.task,
            "status": status,
            "reason": reason,
            "turns": mark.turns,
            "attempts": mark.attempts,
            "age_s": round(mark.age, 1),
            "writer_pid": mark.writer_pid,
            "last_action": mark.last_action,
            "messages": len(mark.messages),
            "next_step": _next_step(mark),
        })
    return rows


def _next_step(mark: Mark) -> str:
    """What the resumed loop would actually be asked, in one line.

    Worth deriving rather than storing: the transcript already says it, and a
    stored copy is a second answer that can disagree with the first.
    """
    if not mark.messages:
        return "restart the task (no transcript was written)"
    last = mark.messages[-1]
    if last.role == "tool":
        return f"read the result of {last.name or 'the last tool'} and carry on"
    if last.role == "user":
        return f"continue from: {' '.join(last.content.split())[:90]}"
    return "continue the task"


def resumable(root: str | os.PathLike[str] | None = None, *,
              now: float | None = None, settle_s: float = SETTLE_S) -> list[Mark]:
    """The records worth picking up right now, oldest first.

    Oldest first on purpose: the run that has been stranded longest is the one
    whose operator has been waiting longest, and a scan that resumes newest
    first would starve it behind a steady trickle of fresh failures.
    """
    marks = [m for m in unfinished(root)
             if m.claim(now=now, settle_s=settle_s)[0] == "resumable"]
    marks.sort(key=lambda m: m.updated_at)
    return marks


def resume_prompt(mark: Mark) -> str:
    """What the resumed loop is told, and why it is not just the old task text.

    A resume is not a retry. A retry asks the same question again and hopes;
    this loop is handed a conversation that already contains work, and a model
    told only "do X" will happily do X from the beginning -- re-reading the
    file it already read, re-running the tool it already ran. So the prompt
    names the interruption, says the work above happened, and asks for the next
    step rather than the first.
    """
    where = (f" The last thing it did was {mark.last_action}."
             if mark.last_action else "")
    cut = (" The interrupted step was cut off mid-way and its result is not "
           "recorded, so it has to be redone rather than assumed."
           if mark.dropped_steps else "")
    return (
        f"[resuming an interrupted run] The task you were given was: {mark.task}\n"
        f"You were {mark.turns} turn(s) into it when the connection to the model "
        f"dropped.{where}{cut} The work recorded above already happened -- do not "
        "repeat a step that is already done. Continue from where it stopped, or "
        "say plainly what is still missing and why."
    )


def prune(root: str | os.PathLike[str] | None = None, *,
          now: float | None = None, ttl_s: float = DEFAULT_TTL_S) -> list[str]:
    """Delete records past their TTL. Returns the run ids removed.

    Nothing else is deleted: a record that is merely stale -- the resume budget
    spent, the transcript unrunnable -- is left for `scan` to report. Deleting
    it would erase the evidence that a task was attempted and failed, which is
    the only thing that lets a later reader tell a repeated failure from a task
    nobody ever started.
    """
    now = time.time() if now is None else now
    gone: list[str] = []
    for mark in unfinished(root):
        if now - mark.updated_at > ttl_s:
            try:
                _path(mark.run_id, root).unlink()
                gone.append(mark.run_id)
            except OSError:
                pass
    return gone


def report(root: str | os.PathLike[str] | None = None) -> str:
    """A one-glance account of what is unfinished, for a terminal."""
    rows = scan(root)
    if not rows:
        return "no unfinished runs."
    lines = [f"{len(rows)} unfinished run(s):"]
    for r in rows:
        note = f"  [{r['status']}] {r['run_id']} {_human(r['age_s'])} ago, " \
               f"{r['turns']} turn(s), {r['messages']} message(s)"
        lines.append(note)
        lines.append(f"      task: {r['task'][:100]}")
        if r["last_action"]:
            # Named separately from the reason: what the run was *doing* is the
            # fact that decides whether a resume is cheap or expensive, and it
            # is the one an operator would otherwise have to open the file for.
            lines.append(f"      was:  {r['last_action']}")
        lines.append(f"      {r['reason']}")
        lines.append(f"      next: {r['next_step']}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_TTL_S", "Journal", "MAX_RESUMES", "Mark", "SETTLE_S", "SCHEMA",
    "WEDGED_AFTER_S", "journal_dir", "prune", "repair", "report", "resumable",
    "resume_prompt", "scan", "unfinished",
]
