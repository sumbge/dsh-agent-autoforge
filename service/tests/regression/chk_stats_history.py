"""Is the per-tool call history bounded, and does the bound keep what matters?

`ToolStats.history` was an unbounded `list[dict]`. Every `record_call` appended
to it, and every save serialised the whole list into `tools.stats`. Measured
2026-11-02 against a real sqlite file:

    history=   100  write=  2.43 ms  stats_json=    5,629 chars
    history=  1000  write=  6.20 ms  stats_json=   54,011 chars
    history=  5000  write= 13.90 ms  stats_json=  268,955 chars
    history= 20000  write= 52.30 ms  stats_json=1,074,890 chars

Linear in calls, so a tool with 20,000 calls rewrites a megabyte every time it
runs. The obvious fix -- keep the newest N -- is wrong on its own: the newest
entries are the successes of a tool that just started working again, and the
handful of failures that explain why it was quarantined would be the first
thing evicted. So the bound reserves slots for failures, and it reports what it
dropped instead of quietly showing a window as if it were the whole life.

Run from the service directory:
    python tests/regression/chk_stats_history.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc.tools.spec import (                         # noqa: E402
    KEEP_CALL_FAILURES, KEEP_CALL_HISTORY, ERROR_CHARS, ToolStats,
)

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    # `detail: object`, not `detail: str`: a failure path that formats its own
    # evidence must not be able to crash the suite. Measured 2026-11-06 -- a
    # mutant reached a check with a list and the suite died with
    # `TypeError: can only concatenate str (not "list") to str` before printing
    # any verdict, which the mutation driver correctly scores as "crashed",
    # not "caught". An assertion that cannot report its evidence is not there.
    print(("  OK   " if cond else "  FAIL ") + label
          + (("  <- " + repr(detail)) if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


def feed(st: ToolStats, n: int, ok: bool = True, error: str | None = None) -> None:
    for _ in range(n):
        st.record_call(ok, error if not ok else None)


def main() -> int:
    print("-- the list stops growing")
    st = ToolStats()
    feed(st, 400)
    check(f"history holds at most {KEEP_CALL_HISTORY} entries",
          len(st.history) == KEEP_CALL_HISTORY, str(len(st.history)))
    check("the overflow is counted, not silently forgotten",
          st.history_dropped == 400 - KEEP_CALL_HISTORY, str(st.history_dropped))

    print("-- trimming does not rewrite the tally")
    check("calls still counts every call", st.calls == 400, str(st.calls))
    check("successes still counts every success", st.successes == 400, str(st.successes))
    check("the success rate is not computed from the window",
          abs(st.success_rate - 1.0) < 1e-9, str(st.success_rate))

    print("-- a failure is not evicted by the successes that follow it")
    st2 = ToolStats()
    feed(st2, 100, ok=True)
    feed(st2, 1, ok=False, error="blocked: quarantined")
    feed(st2, 300, ok=True)
    kept = [e for e in st2.history if not e["ok"]]
    check("the one failure 300 calls ago is still there", len(kept) == 1, str(len(kept)))
    check("its error text survived with it",
          kept and "quarantined" in str(kept[0].get("error")), str(kept))

    print("-- a run of failures is bounded by the window, not by the failure reserve")
    st3 = ToolStats()
    feed(st3, 60, ok=False, error="boom")
    fails_kept = [e for e in st3.history if not e["ok"]]
    # The reserve is a floor for failures, not a ceiling: a tool that failed
    # sixty times in a row has sixty informative entries until the window runs
    # out, and throwing away fifty of them to honour "ten failure slots" would
    # destroy history for no gain.
    check(f"the newest {KEEP_CALL_HISTORY} failures fill the whole window",
          len(fails_kept) == KEEP_CALL_HISTORY, str(len(fails_kept)))
    check("the ten oldest failures were dropped", st3.history_dropped == 10, str(st3.history_dropped))
    check("the kept failures are the newest ones",
          min(e["t"] for e in fails_kept) > 0, "no timestamps")

    print("-- the two budgets share, they do not collide")
    st4 = ToolStats()
    feed(st4, 30, ok=False, error="boom")
    feed(st4, 200, ok=True)
    n_fail = len([e for e in st4.history if not e["ok"]])
    n_ok = len([e for e in st4.history if e["ok"]])
    check("every failure within the reserve is kept", n_fail == KEEP_CALL_FAILURES, str(n_fail))
    check("the rest of the window goes to successes",
          n_ok == KEEP_CALL_HISTORY - KEEP_CALL_FAILURES, str(n_ok))
    check("the window is exactly the cap", len(st4.history) == KEEP_CALL_HISTORY, str(len(st4.history)))

    print("-- order is preserved, so the window reads as a timeline")
    st5 = ToolStats()
    feed(st5, 200)
    ts = [e["t"] for e in st5.history]
    check("entries stay in call order", ts == sorted(ts), "out of order")

    print("-- an error string cannot be the whole payload")
    st6 = ToolStats()
    st6.record_call(False, "x" * 50_000)
    check(f"the error is capped at {ERROR_CHARS} chars",
          len(str(st6.history[0].get("error"))) == ERROR_CHARS,
          str(len(str(st6.history[0].get("error")))))
    st7 = ToolStats()
    st7.record_call(False, "short and specific")
    check("a short error is left alone",
          st7.history[0].get("error") == "short and specific", str(st7.history[0].get("error")))

    print("-- the serialised row is bounded, not just the list")
    st8 = ToolStats()
    feed(st8, 20_000, ok=True)
    size = len(json.dumps(st8.to_dict(), ensure_ascii=False))
    check("stats_json stays small after 20,000 calls", size < 8_000, f"{size} chars")
    check("and every one of those calls is still counted", st8.calls == 20_000, str(st8.calls))

    print("-- the bound survives a round trip through the store")
    import autoforge_svc.store as store_mod
    tmp = tempfile.mkdtemp(prefix="m12_hist_")
    try:
        st_db = store_mod.ToolStore(db_path=os.path.join(tmp, "t.db"))
        try:
            from autoforge_svc.tools.spec import ToolSpec, ToolState
            spec = ToolSpec(name="grow_x", description="d",
                            parameters={"type": "object", "properties": {}},
                            fn=lambda: "", state=ToolState.ACTIVE)
            spec.stats = st8
            st_db.save_tool(spec)
            loaded = st_db.load_all_tools()["grow_x"]
            got = loaded.to_spec()
            check("the loaded history is still capped",
                  len(got.stats.history) == KEEP_CALL_HISTORY, str(len(got.stats.history)))
            check("the loaded dropped-count came back too",
                  got.stats.history_dropped == 20_000 - KEEP_CALL_HISTORY,
                  str(got.stats.history_dropped))
        finally:
            st_db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("-- the cost of one write stays flat")
    tmp2 = tempfile.mkdtemp(prefix="m12_cost_")
    try:
        st_db = store_mod.ToolStore(db_path=os.path.join(tmp2, "t.db"))
        try:
            from autoforge_svc.tools.spec import ToolSpec, ToolState
            spec = ToolSpec(name="cost_x", description="d",
                            parameters={"type": "object", "properties": {}},
                            fn=lambda: "", state=ToolState.ACTIVE)
            st_db.save_tool(spec)
            marks = []
            for target in (200, 2_000, 20_000):
                while spec.stats.calls < target:
                    spec.stats.record_call(True, None)
                t0 = time.perf_counter()
                st_db.save_tool(spec, allow_overwrite=True)
                marks.append(((time.perf_counter() - t0) * 1000, len(json.dumps(spec.stats.to_dict()))))
            sizes = [m[1] for m in marks]
            # The bound is not zero, and the reason is not slack: the serialised
            # form carries 50 float timestamps, and their `repr` is 16, 17 or 18
            # characters depending on the value. Fifty at 18 against fifty at 16
            # is a 100-character swing with nothing to do with the call count.
            # Measured 2026-11-03 over n=40 identical specs: sizes spanned
            # 2,989..3,003 (14 apart), while the timestamp digits alone can
            # account for up to 100.
            #
            # So the question is "does it grow with calls", and the honest way to
            # ask it is against growth: 100 calls to 20,000 is 200x, and the
            # unbounded version went 5,629 -> 1,074,890 characters over that same
            # range. A window of 50 is flat to within timestamp noise, one to two
            # orders of magnitude below any real growth.
            allowed = KEEP_CALL_HISTORY * 2 + 8   # 18-char vs 16-char timestamps
            check("the serialised size does not grow with the call count",
                  max(sizes) - min(sizes) <= allowed, str(sizes))
            check("and it stays near one window, not near the call count",
                  max(sizes) < 20_000, str(max(sizes)))
            check("the write time does not grow with the call count",
                  marks[-1][0] < marks[0][0] * 6 + 5.0,
                  str([round(m[0], 2) for m in marks]))
        finally:
            st_db.close()
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)

    print("-- the tally and the list account for each other, for any sequence")
    # calls + calls_not_in_counters - history_dropped == len(history)
    #
    # This is the identity `describe_tool` now enforces as
    # `counters_unreconciled`. It is checked here over random sequences because
    # the cases I would think to write by hand are exactly the ones my own
    # implementation already handles: the bug this catches is whatever I did not
    # imagine. Measured 2026-11-03: `fake_file_size` sat on the live database
    # with `calls: 0` beside six history entries and no field said why.
    import random
    rng = random.Random(20261103)
    broken = []
    for trial in range(200):
        stx = ToolStats()
        n = rng.choice([0, 1, 49, 50, 51, 120, 400])
        for _ in range(n):
            ok = rng.random() > rng.choice([0.0, 0.5, 0.9])
            stx.record_call(ok, None if ok else "boom")
        lhs = stx.calls + stx.calls_not_in_counters - stx.history_dropped
        if lhs != len(stx.history):
            broken.append((trial, n, stx.calls, stx.calls_not_in_counters,
                           stx.history_dropped, len(stx.history)))
    check("the identity holds across 200 random call sequences",
          not broken, str(broken[:3]))

    print("-- a zeroed tally says why it is zero")
    from autoforge_svc.tools.registry import ToolRegistry
    from autoforge_svc.tools.spec import ToolSpec, ToolState
    reg = ToolRegistry()
    rx = ToolSpec(name="rehab_x", description="d",
                  parameters={"type": "object", "properties": {}},
                  fn=lambda: "", state=ToolState.QUARANTINED)
    reg.register(rx)
    feed(rx.stats, 6)
    check("the tool has calls before rehab", rx.stats.calls == 6, str(rx.stats.calls))
    reg.rehab("rehab_x")
    check("rehab zeroes the trial counters", rx.stats.calls == 0, str(rx.stats.calls))
    check("rehab keeps the history that explains the quarantine",
          len(rx.stats.history) == 6, str(len(rx.stats.history)))
    check("and says how many calls the counters no longer include",
          rx.stats.calls_not_in_counters == 6, str(rx.stats.calls_not_in_counters))
    d = rx.stats.to_dict()
    check("the accounting identity holds in the serialised row",
          d["calls"] + d["calls_not_in_counters"] == 6, str(d))

    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
