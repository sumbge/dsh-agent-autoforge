"""Who may edit which file, in chunks of time -- enforced, not remembered.

The failure this exists to prevent, stated as it happened: two sessions on this
machine both worked on `forge/sandbox.py` and `forge/wsl_isolation.py` at the
same time. One declared its lane on the bus at 14:51:22. The other had already
written and committed those exact files by 14:50:52. Both were doing the right
thing by the rules of the day -- the bus printed "declare your lane before
editing a shared file" at startup and accepted a claim whenever anyone got round
to it. The rules were advisory, and advice is exactly what does not arrive in
time: a warning delivered after the file is written is not coordination, it is
an obituary.

So this module makes the lane a thing you cannot have by remembering:

  * a lane is a json file per target path, holding the session that holds it and
    when it expires;
  * a competing session sees the holder's NAME and its AGE, and is refused
    while the lane is live -- and told the one fact that makes the refusal
    actionable, which is how long it has to wait or who to ask;
  * a lane that has not been renewed is EXPIRED, not deleted. History on a
    shared disk may not be rewritten -- the same rule the bus follows for its
    board -- so it is written once and read as stale thereafter;
  * if the holder's process is no longer in the process table, the lane is
    released at once. A session that was killed cannot renew, and a lock whose
    owner is dead and which still blocks is how a working directory turns into
    a place nobody can work.

Scope is deliberate and stated rather than implied: this guards *this module's
callers* -- the self-edit path and anything else that asks it to. It is not a
filesystem-level lock; `run_python` and a hand-edited file go around it. A guard
that claimed otherwise would be worse than none, because the next session would
trust it.

One boundary found by running it, which changes how the lane should be taken: a
lane's liveness is judged on the process that CLAIMED it, so a lane claimed from
a short-lived helper dies with that helper. Measured: a claim made inside one
`run_python` subprocess read back as `holder-gone` from the next process, because
the claiming process had already exited. That is right for the self-edit script
-- its lane covers exactly the window of one edit, which is the window that
matters -- and it is wrong for a session that wants a lane for a long piece of
work: that session must claim from the session process (`auto <pid>-<ts>`), or
the lane will look abandoned to whoever looks next.

Killed-holder evidence is the process table, not the timestamp, for the same
reason `presence` in the bus module is: `pid_alive` is what tells "quiet" from
"dead", and `pid_alive` is correct on this host's semantics (on Windows
`os.kill(pid, 0)` does not signal, it checks).

Every write here goes through one `os.replace` of a temp file, so a reader never
sees half a lane, and a crash mid-write leaves the previous holder intact rather
than nobody holding it.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

#: How long a lane lasts without being renewed. Ten minutes is one long test run
#: plus margin -- shorter and a live session gets interrupted for being slow,
#: longer and a killed one blocks its peers for the rest of the session.
DEFAULT_TTL_S = 600.0

def tempfile_dir() -> str:
    import tempfile
    return tempfile.gettempdir()


def free_prefixes() -> tuple[str, ...]:
    """Directories nobody needs a lane for: scratch space, resolved per call.

    Per call and not at import, because this host's environment is not stable
    across the places this module runs -- a scrubbed subprocess has no TEMP set
    and a normal shell does, and a constant computed at import would then answer
    differently depending on who imported it. Deduplicated because both
    candidates resolve to the same directory here, and a duplicate is harmless
    but reads like a mistake.
    """
    cands = [tempfile_dir(), os.environ.get("TEMP"),
             os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp")]
    out: list[str] = []
    for c in cands:
        if not c:
            continue
        c = os.path.normcase(os.path.abspath(c)) + os.sep
        if c not in out:
            out.append(c)
    return tuple(out)


def bus_dir() -> Path:
    """Where lanes live: beside the bus, so one directory answers "who is here"."""
    env = os.environ.get("AUTOFORGE_BUS_DIR")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or os.path.join(
        os.path.expanduser("~"), "AppData", "Local")
    # The bus resolves its own home the same way; duplicating the rule here
    # would let the two drift, so this is kept to the one candidate that the bus
    # picks on this host and the module says so.
    return Path(base) / "hermes" / "agent-bus"


def lanes_dir() -> Path:
    return bus_dir() / "lanes"


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def pid_alive(pid: int) -> bool:                                     # noqa: D401
    """Is that process still running? Borrowed from the bus, not re-derived.

    Re-derived here once already, and the copy was wrong on Windows: it called
    `os.kill(pid, 0)` inside a try and treated *any* exception as dead, which is
    right for ESRCH and wrong for EPERM -- a live process owned by another user
    raises EPERM and would have read as dead.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    except Exception:                                                # noqa: BLE001
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
        except Exception:                                            # noqa: BLE001
            return False


def own_session_id() -> str:
    return os.environ.get("AUTOFORGE_SESSION") or "auto-%d-%d" % (
        os.getpid(), int(time.time()))


@dataclass
class Lane:
    target: str
    session: str
    name: str
    pid: int
    taken_at: float
    expires_at: float
    why: str = ""
    renewed: list[float] = field(default_factory=list)

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.taken_at)

    def live(self, now: float | None = None) -> bool:
        return (now or time.time()) < self.expires_at

    def holder_alive(self) -> bool:
        return pid_alive(self.pid)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["age_s"] = round(self.age_s, 1)
        d["state"] = self.state()
        return d

    def state(self, now: float | None = None) -> str:
        if not self.holder_alive():
            return "holder-gone"
        if not self.live(now):
            return "expired"
        return "live"


def lane_file(target: str, root: str | os.PathLike[str] | None = None) -> Path:
    """One file per target, named by a digest so any target is representable."""
    import hashlib
    root = Path(root) if root is not None else lanes_dir()
    key = hashlib.sha256(_normalise(target).encode("utf-8", "replace")).hexdigest()[:16]
    return root / f"{key}.json"


#: A lane target that is not a file. Forging writes no .py to disk -- a tool's
#: source goes into a row of the shared sqlite store -- and the collision that
#: matters there is not two writers on a file but two sessions claiming the same
#: NAME. `save_tool` is an INSERT OR REPLACE, so the second writer does not fail
#: loudly: it silently replaces a peer's freshly forged tool with its own. A
#: path-shaped target would be a lie about what is being held, so the target is
#: named instead, and nothing here ever touches the filesystem with it.
RESOURCE_PREFIX = "resource://"


def resource(name: str) -> str:
    """The lane target for a shared named thing: a tool row, a market entry."""
    name = str(name).strip()
    if not name:
        raise ValueError("a resource lane needs a name")
    return RESOURCE_PREFIX + name


def _normalise(target: str) -> str:
    """Paths are normalised (case, separators, ..); resource names are not.

    A resource name is an opaque key: lowercasing it would merge two tools whose
    names differ only in case, which is a silent aliasing bug of exactly the kind
    this module exists to stop.
    """
    target = str(target)
    if target.startswith(RESOURCE_PREFIX):
        return target
    return os.path.normcase(os.path.abspath(target))


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)


def read_lane(target: str, root: str | os.PathLike[str] | None = None) -> Lane | None:
    path = lane_file(target, root)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return Lane(target=d.get("target", target), session=d["session"],
                    name=d.get("name", "?"), pid=int(d.get("pid") or 0),
                    taken_at=float(d.get("taken_at") or 0),
                    expires_at=float(d.get("expires_at") or 0),
                    why=d.get("why", ""), renewed=list(d.get("renewed") or []))
    except (KeyError, TypeError, ValueError):
        return None


class LaneMissing(RuntimeError):
    """No lane on record for that target -- a different situation from refusal.

    Kept separate because the two want different fixes: a refusal means talk to
    the holder or wait; this means nobody holds it, so just take it. Collapsing
    them into one exception sent the caller a stand-in whose age printed as 1.79
    billion seconds -- which is what a placeholder does when it is asked a
    question only a real object can answer.
    """

    def __init__(self, target: str):
        self.target = target
        super().__init__(f"no lane on record for {target}")


class LaneRefused(RuntimeError):
    """Raised when a live lane is held by someone else. Carries the facts."""

    def __init__(self, lane: Lane):
        self.lane = lane
        super().__init__(
            f"{lane.name} holds a live lane on {lane.target} "
            f"({lane.age_s:.0f}s old, expires in "
            f"{max(0.0, lane.expires_at - time.time()):.0f}s); "
            f"ask that session or wait rather than editing over it")


def is_free(target: str, scratch: tuple[str, ...] | None = None) -> bool:
    """Does this path need no lane? `scratch` decides, and it is policy, not OS fact.

    Nobody coordinates over a temp file -- but "temp" is a policy about which
    paths are nobody's business, not a property of the filesystem. It became a
    parameter after a test run made the point: pytest's `tmp_path` lives *inside*
    the temp directory, so a test that wanted to exercise the guarded path got
    the free path instead and eleven assertions failed on a module whose logic
    was right. The same confusion reaches production the other way: a session
    whose scratch space is not `%TEMP%` would have its own files guarded, and a
    caller with a stricter notion of scratch should be able to say so.

    The predicate was also inverted in the first version -- `not any(...)` -- so
    every path outside the temp directory was judged free and the guard said
    nothing about the files it exists to protect. A guard that never refuses is
    indistinguishable from no guard, which is why the first test written for this
    module asserts a project file is *not* free.
    """
    target = str(target)
    if target.startswith(RESOURCE_PREFIX):
        # A named resource is shared by definition: there is no scratch copy of
        # "the tool called read_magic_value".
        return False
    p = _normalise(target)
    roots = free_prefixes() if scratch is None else tuple(
        os.path.normcase(os.path.abspath(s)) + os.sep for s in scratch)
    return any(p.startswith(pre) for pre in roots)


def claim(target: str, *, session: str = "", name: str = "autoforge",
          why: str = "", ttl_s: float = DEFAULT_TTL_S,
          root: str | os.PathLike[str] | None = None,
          scratch: tuple[str, ...] | None = None) -> Lane:
    """Take the lane on `target`, or refuse with who holds it.

    Refusal is by exception rather than a False return, because the caller that
    forgets to check a False return is the caller this module exists for. The
    exception carries the holder, so the message can name a session to ask.

    There is deliberately no `force` argument. The first version had one, meant
    to override a *stale* lane -- and stale lanes are already taken over
    automatically, so all it could actually do was override a live one, which is
    the advisory rule again with an extra step. Measured: with `force=True` a
    second session walked straight through a live lane. Breaking a live lane is
    a real need, but it is a different act: `break_lane` below, which demands a
    reason and leaves the reason on record.
    """
    session = session or own_session_id()
    target = str(target)
    if not target.startswith(RESOURCE_PREFIX):
        target = os.path.abspath(target)
    if is_free(target, scratch):
        return Lane(target=target, session=session, name=name, pid=os.getpid(),
                    taken_at=time.time(), expires_at=time.time() + ttl_s, why=why)

    existing = read_lane(target, root)
    if existing is not None and existing.session != session:
        state = existing.state()
        if state == "live":
            raise LaneRefused(existing)
        # expired or holder-gone: take it. The takeover is appended to
        # `renewed`, so the history shows the lane changed hands and when --
        # and `why` on the new lane is where the reason for the takeover goes.
    now = time.time()
    renewed = list(existing.renewed) if existing else []
    if existing is not None and existing.session != session:
        renewed.append(now)
    lane = Lane(target=target, session=session, name=name, pid=os.getpid(),
                taken_at=now, expires_at=now + ttl_s, why=why, renewed=renewed)
    _write_atomic(lane_file(target, root), lane.to_dict())
    return lane


def break_lane(target: str, *, session: str = "", name: str = "autoforge",
               why: str, ttl_s: float = DEFAULT_TTL_S,
               root: str | os.PathLike[str] | None = None) -> Lane:
    """Take a LIVE lane from someone else, with the reason written down.

    The escape hatch that a live lane needs: a peer can be stuck, wrong about
    which file it is editing, or simply the process that never renews. What it
    is NOT is silent -- `why` is required, so the record answers "who went
    through whose lane and why" instead of only showing a lane that changed
    hands for no visible reason.
    """
    lane = read_lane(target, root)
    if lane is not None and lane.session == session:
        return lane
    now = time.time()
    renewed = list(lane.renewed) if lane else []
    if lane is not None:
        renewed.append(now)
    target = str(target)
    out = Lane(target=(target if target.startswith(RESOURCE_PREFIX)
                       else os.path.abspath(target)),
               session=session or own_session_id(),
               name=name, pid=os.getpid(), taken_at=now,
               expires_at=now + ttl_s, why=why, renewed=renewed)
    _write_atomic(lane_file(out.target, root), out.to_dict())
    return out


def renew(target: str, *, session: str = "", ttl_s: float = DEFAULT_TTL_S,
          root: str | os.PathLike[str] | None = None) -> Lane:
    """Extend my own lane. Refuses to renew someone else's -- that is theft."""
    session = session or own_session_id()
    lane = read_lane(target, root)
    if lane is None:
        raise LaneMissing(target)
    if lane.session != session:
        raise LaneRefused(lane)
    lane.expires_at = time.time() + ttl_s
    lane.renewed.append(time.time())
    _write_atomic(lane_file(target, root), lane.to_dict())
    return lane


def release(target: str, *, session: str = "",
            root: str | os.PathLike[str] | None = None) -> bool:
    """Give up my lane. Someone else's cannot be given up by me."""
    session = session or own_session_id()
    lane = read_lane(target, root)
    if lane is None or lane.session != session:
        return False
    lane.expires_at = 0.0          # expired now; the file stays as history
    lane.renewed.append(time.time())
    _write_atomic(lane_file(target, root), lane.to_dict())
    return True


def holders(root: str | os.PathLike[str] | None = None) -> list[Lane]:
    """Every lane on record, live ones first. For a startup report."""
    root = Path(root) if root is not None else lanes_dir()
    out: list[Lane] = []
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            out.append(Lane(target=d.get("target", "?"),
                            session=d.get("session", "?"),
                            name=d.get("name", "?"), pid=int(d.get("pid") or 0),
                            taken_at=float(d.get("taken_at") or 0),
                            expires_at=float(d.get("expires_at") or 0),
                            why=d.get("why", ""),
                            renewed=list(d.get("renewed") or [])))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    live = [l for l in out if l.state() == "live"]
    dead = [l for l in out if l.state() != "live"]
    live.sort(key=lambda l: -l.age_s)
    dead.sort(key=lambda l: -l.taken_at)
    return live + dead


def report(root: str | os.PathLike[str] | None = None) -> str:
    """The line a session prints at startup: what is taken, and by whom."""
    ls = holders(root)
    if not ls:
        return "lanes: none held"
    live = [l for l in ls if l.state() == "live"]
    head = f"lanes: {len(live)} live of {len(ls)} on record"
    lines = [head]
    for l in ls[:12]:
        lines.append(f"  {l.state():12} {os.path.basename(l.target):34} "
                     f"{l.name:10} {l.age_s:7.0f}s  {l.why[:40]}")
    return "\n".join(lines)
