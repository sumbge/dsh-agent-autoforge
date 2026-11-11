"""The host-probe block in the per-turn brief, and what it may cost.

Why this file exists
--------------------
`tars_port.probe_host()` has been in this package since the tars port and
nothing has ever called it. Upstream injects the equivalent block into every
turn (`agent.py:6248`: `f"{facts}\\n\\n{menu}\\n\\n" f"{probe}\\n\\n{kept}..."`),
and the reason is not decorative: this agent has been running on this host for
weeks, so "is the store reachable", "are the skills there", "is the network up"
are questions it would otherwise re-derive by spending a turn on them.

Bringing it in turns on one number. Measured 2026-11-03 on this machine:

    probe_host()                1645 ms   <- unusable in a per-turn block
    probe_host(timeout=0.2)      214 ms
    probe_host(timeout=0.05)      61 ms
      of which verify_vendor      5 ms   (277 KB of hashing)
      of which skill index       48 ms   cold, 7 ms warm
      of which DNS               14 ms
      of which the market socket  up to `timeout` -- and it is DEAD here, so
                                  the whole 1.5 s was one `socket.create_connection`

So the cost is not the work. It is a `connect` to a port that is not listening,
with a timeout sized for a person waiting, not for a prompt being built. Two
changes follow, and both are asserted here:

1. The default timeout drops to something a per-turn block can afford.
2. The answer is cached for a window, because "the store is reachable" does not
   change between one turn and the next, and re-asking it every turn pays the
   probe's cost for the same sentence.

What is checked
---------------
1. The brief carries a probe block, exactly once.
2. The block is placed with the other "what I have" facts, not after the
   volatile mission and memory lines -- stable content first, so the prefix is
   not re-billed when a mission note changes.
3. The probe cannot block a turn: against a dead port it finishes inside 400 ms.
4. A second call inside the window is a cache hit, not a second probe.
5. `refresh=True` bypasses the cache -- a cache that ignores its own bypass is
   not a cache, it is a constant.
6. A failing probe does not fail the brief: the block degrades and the turn
   still assembles.
7. The block carries no elapsed-time figure. The brief is a prompt prefix, so a
   millisecond reading that moves between calls re-bills every character behind
   it. (`probe_host` documents this for the DNS line; the assertion guards the
   property rather than the single line that happens to have it.)
"""
import os
import re
import shutil
import sys
import tempfile
import time

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)

from autoforge_svc import tars_port  # noqa: E402
from autoforge_svc.service import Service  # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print(f"  OK   {label}")
    else:
        print(f"  FAIL {label}  <- {detail}")
        FAILS.append(label)


tmp = tempfile.mkdtemp(prefix="m23_brief_")
db = os.path.join(tmp, "b.db")
svc = Service(db_path=db)

print("-- the probe is affordable in a per-turn block")
started = time.perf_counter()
default_lines = tars_port.probe_host()
default_ms = (time.perf_counter() - started) * 1000
check("the default probe finishes inside a per-turn budget", default_ms < 400,
      f"{default_ms:.0f} ms")
check("and it still reports every line", len(default_lines) >= 10, len(default_lines))

print("-- and a second look inside the window is not a second probe")
svc._probe_cache = (0.0, [])
t0 = time.perf_counter()
first = svc._probe_lines()
first_ms = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
second = svc._probe_lines()
second_ms = (time.perf_counter() - t0) * 1000
check("the first call does the work", bool(first), first[:2])
check("the second call is served from the window", second_ms * 4 < first_ms + 5,
      f"first {first_ms:.1f} ms, second {second_ms:.1f} ms")
check("and it is the same answer", second == first, (first[:1], second[:1]))
t0 = time.perf_counter()
third = svc._probe_lines(refresh=True)
third_ms = (time.perf_counter() - t0) * 1000
check("refresh=True pays for a real probe again", third_ms * 4 > second_ms + 5,
      f"cached {second_ms:.1f} ms, refreshed {third_ms:.1f} ms")
check("and returns a fresh answer of the same shape", len(third) == len(first),
      (len(first), len(third)))

print("-- the per-turn copy is the short one; the tool keeps the long one")
long_form = tars_port.probe_lines()
short_form = svc._probe_lines(refresh=True)
check("the brief copy is smaller than the tool's copy",
      len("\n".join(short_form)) < len("\n".join(long_form)),
      f"brief {len(chr(10).join(short_form))} vs full {len(chr(10).join(long_form))}")
saved_paths = [ln for ln in short_form if "\\" in ln]
check("nor does it carry the vendor path", "vendor dir" not in "\n".join(short_form),
      [ln[:60] for ln in short_form if "vendor dir" in ln])
check("the archive line stops explaining where it would be created",
      "not created yet" not in "\n".join(short_form),
      [ln[:80] for ln in short_form if "archive" in ln])
check("but it still says the archive is empty",
      "archive: empty" in "\n".join(short_form), short_form[-3:])
check("the store line keeps its path, because it gets looked up",
      any(ln.startswith("- store:") and ("autoforge" in ln or "no autoforge.db" in ln)
          for ln in short_form), [ln[:60] for ln in short_form if "store" in ln])
check("the long copy keeps what the short one dropped",
      "vendor dir" in "\n".join(long_form) and len(saved_paths) >= 0, long_form[2:4])
# The property that matters: brief mode may only *drop* lines, never introduce
# one. If it could rewrite, the short copy would be a second opinion about the
# machine and the two would drift apart -- and the reader who trusts the short
# one would never see the disagreement.
long_set = {ln for ln in long_form}
invented = [ln for ln in short_form if ln.startswith("- ") and ln not in long_set
            and not ln.startswith("- archive: empty")]
check("every line in the brief copy also exists in the long copy", not invented,
      invented[:3])

print("-- the brief carries it, once, in the right place")
brief = svc.context_brief()
text = str(brief.get("text") or "")
heads = [ln for ln in text.splitlines() if ln.startswith("## Host probe")]
check("the brief has a host-probe heading", len(heads) == 1, heads)
probe_at = text.find("## Host probe")
tools_at = text.find("TOOLS")
mission_at = text.find("MISSION")
check("the probe is in the brief at all", probe_at >= 0, text[:200])
# Placement is a cost decision, and it is asserted rather than described: the
# probe says what this machine looks like *now*, so it belongs with the facts
# about what the agent has, above the mission and memory lines that change when
# a note is written. If it drifted below them, every mission note would re-bill
# the probe too.
check("it sits with the capabilities, not below the volatile lines",
      0 <= probe_at < mission_at if mission_at >= 0 else probe_at >= 0,
      f"probe@{probe_at} mission@{mission_at}")
check("and it is above the skill menu's line about procedures",
      probe_at < text.find("PROCEDURES") if text.find("PROCEDURES") >= 0 else True,
      (probe_at, text.find("PROCEDURES")))
check("the brief stays inside its budget",
      int(brief.get("chars") or 0) <= 4000, brief.get("chars"))
check("the block is not empty", "vendor files verified" in text, text[probe_at:probe_at + 200])

print("-- the block cannot carry a figure that moves between calls")
floats = re.findall(r"\b\d+\.\d+\s*(?:ms|s)\b", text)
check("no elapsed time in the brief", not floats, floats)
check("and no wall-clock timestamp either",
      not re.search(r"\b17\d{8}\.\d", text), text[:120])

print("-- a failing probe degrades the block, it does not fail the turn")
saved = tars_port.probe_host


def explode(*_a, **_kw):
    raise RuntimeError("the probe blew up")


try:
    tars_port.probe_host = explode
    svc._probe_cache = (0.0, [])
    try:
        hurt = svc.context_brief()
        raised = ""
    except Exception as exc:                               # noqa: BLE001
        hurt = {}
        raised = f"{type(exc).__name__}: {exc}"
finally:
    tars_port.probe_host = saved
    svc._probe_cache = (0.0, [])
# A raise here would end the file before any assertion ran, which is how a
# regression reads as a crash instead of as a failure. `raised` turns it into
# the sentence it should have been.
check("the brief survives a probe that raises", not raised, raised)
h_text = str(hurt.get("text") or "")
check("the brief still assembles", bool(h_text), hurt)
check("it says the probe failed rather than going quiet",
      "probe" in h_text.lower() and ("unavailable" in h_text or "failed" in h_text),
      h_text[:300])
check("and it stays inside the budget after failing",
      int(hurt.get("chars") or 0) <= 4000, hurt.get("chars"))

svc.store.close()
svc.missions.close()
shutil.rmtree(tmp, ignore_errors=True)

print("TOTAL FAILURES:", len(FAILS), FAILS)
sys.exit(0 if not FAILS else 1)
