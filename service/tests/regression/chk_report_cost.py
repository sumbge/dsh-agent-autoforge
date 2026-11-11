"""Can a reader ask "what does the store hold" without paying for a full audit?

Under test: the split between `report()` and `verify_ledger()`.

Item seven was written down as "`brief_served` is 99.3% of the ledger, archive
it". Measuring before moving anything changed the question twice:

  * Deleting heartbeat rows is not available. On a copy of the production file,
    removing one `brief_served` row made the *next* row a `broken_link`, and
    removing the 7,863 legacy ones produced five breaks including
    `orphan_chain_start`. The chain is the whole reason the rows are there; a
    log you can prune is a log that cannot prove it was not pruned.

  * The heartbeat rows are not expensive to read. `get_events(20)` takes 0.1 ms
    with or without them and `event_kinds()` takes 1.4 ms. Their payload totals
    180 KB out of a 1.8 MB file.

What is expensive is that `report()` -- the answer to "what do you hold", asked
by `my_capabilities` and by any curious turn -- ran `verify_chain()`, a walk of
every row with a sha256 per row. Measured 2026-11-03 on the production ledger:
`report()` 46.4 ms of which `verify_chain()` is 43.9 ms, i.e. 95%. The walk is
linear in rows (5.2 microseconds each on this host), so the same call costs
103 ms at 20k rows and 515 ms at 100k. The heartbeat is what made the row count
large enough to notice; it is not the cause.

So the contract this file pins down:

  report()          cheap, bounded, and says whether a full check is due
  verify_ledger()   still walks every row, still proves what it proved before

The temptation to be avoided is the fix that makes the number look good by
checking less while saying the same thing. `report()["chain"]` must therefore
never present a stale or partial walk as a current one: when it is not the full
walk, it has to say so in a field a reader cannot miss.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc import chaining                                # noqa: E402
from autoforge_svc.service import Service                         # noqa: E402
from autoforge_svc.store import ToolStore                         # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    # `detail: object` and `repr`: a failure path that formats its own evidence
    # must not be able to crash the suite before printing a verdict.
    print(("  OK   " if cond else "  FAIL ") + label
          + (("  <- " + repr(detail)) if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


def grow(store: ToolStore, n: int) -> None:
    """Append n cheap chained rows. 2.5 ms each, so keep n modest."""
    for i in range(n):
        chaining.append_event(store._conn, "call", {"tool": "t%d" % i, "ok": True})


def best_ms(fn, n: int = 5) -> float:
    best = 1e9
    for _ in range(n):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1000.0


def median_ms(fn, rounds: int = 5, per_round: int = 5) -> tuple:
    """(median, max) of `rounds` independent best-of-`per_round` timings.

    Measured 2026-11-07, because a ratio of two sub-millisecond numbers is a
    ratio of two noises and this suite flaked on exactly that: five independent
    best-of-5 timings of `store.report()` at 400 rows came out 0.33 / 0.44 /
    0.52 / 0.53 / 0.33 ms -- the measurement is stable to about 1.6x against
    itself, while the assertion compared a 4x size step and demanded under
    3x. One best-of-5 sample cannot tell 1.3 (what 400 -> 1600 actually costs)
    from 2.9 (what one unlucky draw produced). A median over rounds is the
    cheapest thing that can.
    """
    samples = [best_ms(fn, per_round) for _ in range(rounds)]
    samples.sort()
    return samples[len(samples) // 2], samples[-1]


def min_ms(fn, rounds: int = 7, per_round: int = 9) -> float:
    """The fastest single run over `rounds` independent best-of-`per_round`
    timings.

    A median answers "what does this cost here"; a minimum answers "how fast
    can it go at all", and that is the question a complexity assertion is
    asking. Measured 2026-11-11, after this suite flaked 1 in 5: twelve pairs
    of `chaining.summarize` timings at 400 and 1600 rows had a median ratio of
    1.03 -- flat, as designed -- but two of the twelve draws landed at 4.8x and
    5.1x. The shape of the distribution is the point: those were the same code
    taking 0.25 ms instead of 0.05 ms, which is what an operating system does
    to a sub-millisecond measurement, not what a per-row read looks like. The
    shipped assertion compared those medians against a 2x factor with 25%
    headroom, so it passed 9 of 12. Taking the minimum over more repetitions
    removes the scheduling tail instead of widening the threshold that has to
    absorb it.
    """
    return min(best_ms(fn, per_round) for _ in range(rounds))


def _timed_report(tmp: str, rows: int) -> tuple:
    """(report_ms, walk_ms) for a store of exactly `rows` chained rows."""
    path = os.path.join(tmp, "sized_%d.db" % rows)
    store = ToolStore(db_path=path)
    try:
        grow(store, rows)
        return median_ms(store.report)[0], median_ms(store.verify_chain)[0]
    finally:
        store.close()


def _timed_summary(tmp: str, rows: int) -> float:
    """Median milliseconds for chaining.summarize() on a sized store."""
    path = os.path.join(tmp, "sum_%d.db" % rows)
    store = ToolStore(db_path=path)
    try:
        grow(store, rows)
        return median_ms(lambda: chaining.summarize(store.connection()))[0]
    finally:
        store.close()


def _min_summary(tmp: str, rows: int) -> float:
    """Fastest milliseconds for chaining.summarize() on a sized store.

    Same construction as `_timed_summary`, different estimator -- see `min_ms`
    for why the assertion that compares two sizes uses the minimum.
    """
    path = os.path.join(tmp, "sum_min_%d.db" % rows)
    store = ToolStore(db_path=path)
    try:
        grow(store, rows)
        return min_ms(lambda: chaining.summarize(store.connection()))
    finally:
        store.close()


def _report_plans(store: ToolStore) -> dict:
    """EXPLAIN QUERY PLAN for each statement the cheap report issues."""
    out = {}
    statements = {
        "count": "SELECT COUNT(*) FROM forge_events",
        "kinds": "SELECT kind, COUNT(*) FROM forge_events GROUP BY kind",
        "agent": ("SELECT kind, COUNT(*) FROM forge_events "
                  "WHERE kind NOT IN ('brief_served') GROUP BY kind"),
        "since": "SELECT MIN(timestamp) FROM forge_events",
        "until": "SELECT MAX(timestamp) FROM forge_events",
    }
    for name, sql in statements.items():
        rows = store.connection().execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        out[name] = [str(r[-1]) for r in rows]
    return out


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m11_cost_")
    svc = Service(db_path=os.path.join(tmp, "c.db"))
    try:
        grow(svc.store, 1200)

        print("-- report() no longer pays for a full walk")
        # Order matters here, and getting it wrong is what my first version did:
        # it timed `verify_chain` before asking whether a check was due, so the
        # walk it took for comparison had already cleared the flag it then
        # asserted on. Measure the cheap path first, while nothing has walked.
        rep_ms = median_ms(svc.report)[0]
        walk_ms = median_ms(svc.store.verify_chain)[0]
        # This used to compare `Service.report` against `store.verify_chain` --
        # two different layers -- and it started failing on 2026-11-06 when
        # `report()` grew a `state_map()` call: a measured 1.18 ms of fixed
        # work (seven owners asked where they keep their state). At 1200 rows
        # the service report is 2.3 ms against a 5.4 ms walk, so the 4x ratio
        # stopped holding -- and it had never measured what it claimed. Fixed
        # cost, at a size where the walk is still small. Measured that day:
        # 300 rows report 2.50 / walk 1.40, 1200 rows 2.22 / 5.37, 4800 rows
        # 2.98 / 24.19. `report` moved 1.19x for 16x the rows; the walk moved
        # 17x. The claim worth pinning is about the store, so compare store to
        # store, and hold the service to a budget of its own.
        # Both sides are medians for the same reason as below: at 1200 rows
        # store.report() is around half a millisecond, and one best-of-5 draw
        # of a half-millisecond measurement can move by 1.6x on its own.
        store_rep_ms = median_ms(svc.store.report)[0]
        check("the store's report() is much cheaper than the full walk",
              store_rep_ms * 4 < walk_ms,
              "store.report=%.1fms walk=%.1fms" % (store_rep_ms, walk_ms))
        # The service-level answer adds fixed work (state map, wire meter,
        # controls) that must stay fixed, not become per-row. A walk folded
        # back in would add the walk's own 5.4 ms here, so the budget is stated
        # in absolute terms rather than as a ratio to something it does not own.
        check("and the service's report() stays cheap in absolute terms",
              rep_ms < 10.0, "Service.report=%.1fms at 1200 rows" % rep_ms)

        # The ratio above is a smell test, not the contract: at 300 rows it
        # passes by 11%. What separates a counted report from a per-row walk
        # is how each one grows.
        #
        # This pair used to be a ratio of two ratios built from one best-of-5
        # sample each, and on 2026-11-07 that flaked: two failures in five
        # plain runs, one of them "report 1.83x walk 2.89x (0.62ms -> 1.14ms
        # vs 5.38ms -> 15.52ms)". Neither ratio was wrong; both were noisy.
        # Five independent best-of-5 timings of store.report() at 400 rows
        # measured 0.33 / 0.33 / 0.44 / 0.52 / 0.53 ms -- 1.6x against
        # itself -- and the walk's small sample happened to be slow that run,
        # so the ratio it was divided into collapsed. A ratio of two
        # sub-millisecond numbers is a ratio of two noises.
        #
        # So the numbers are medians now (see median_ms), and the assertion
        # worth making is stated in the units the claim is actually about:
        # rows. Measured 2026-11-07 at 400 -> 1600 rows: report 0.44 -> 0.58
        # ms (+0.13 ms for +1200 rows), walk 2.36 -> 11.77 ms (+9.41 ms). A
        # fixed cost and a per-row cost differ by two orders of magnitude in
        # marginal terms, which no clock noise can bridge, and that is what
        # the next two checks pin.
        small = _timed_report(tmp, 400)
        large = _timed_report(tmp, 1600)
        rep_growth = large[0] / max(small[0], 1e-6)
        walk_growth = large[1] / max(small[1], 1e-6)
        rep_marginal = large[0] - small[0]
        walk_marginal = large[1] - small[1]
        # 1200 extra rows. The honest tool pays a fixed cost for them; a tool
        # that reads the ledger pays for every one. The bracket is one
        # millisecond against five, and both observed values sit far outside
        # it in the right direction.
        check("1200 extra rows cost the report a fraction of a millisecond",
              rep_marginal < 1.0,
              "report +%.2fms for +1200 rows (%.2f -> %.2f)"
              % (rep_marginal, small[0], large[0]))
        check("...while they cost the full walk several",
              walk_marginal > 5.0,
              "walk +%.2fms for +1200 rows (%.2f -> %.2f)"
              % (walk_marginal, small[1], large[1]))
        # Kept as a smell test, on medians. Four times the rows must not cost
        # three times the time; measured 1.30x. The margin exists to catch a
        # walk folded back in (which would land near 4x), not to police a
        # tenth of a millisecond.
        check("and four times the rows is still under three times the cost",
              rep_growth < 3.0,
              "report grew %.2fx for 4x the rows (%.2f -> %.2f ms)"
              % (rep_growth, small[0], large[0]))

        # The chain summary answers the same question at any size, so its cost
        # must not track the ledger. Measured 2026-11-04: 0.055 ms flat from
        # 300 to 12000 rows once the counts are three index-backed queries;
        # as one aggregate that reads every row it was 4.15 ms at 12000. The
        # case that matters is the flat one, so this asserts flatness and the
        # absence of a per-row read, in that order.
        sum_small = _min_summary(tmp, 400)
        sum_large = _min_summary(tmp, 1600)
        # Four times the rows may not cost three times the time. Measured on
        # minima 2026-11-11: 0.141 -> 0.049 ms, i.e. flat. A summary that read
        # every row would land near 4x. The factor is generous on purpose --
        # what it has to separate is "flat" from "per-row", and the estimator
        # above is what keeps the second value from being a scheduling tail.
        check("the chain summary does not get more expensive with rows",
              sum_large < max(sum_small * 3.0, 0.30),
              "%.3fms at 400 rows, %.3fms at 1600" % (sum_small, sum_large))
        check("and it is cheap in absolute terms",
              sum_large < 1.5, "%.3fms at 1600 rows" % (sum_large,))

        # What the two estimators promise, pinned without a clock. Replacing
        # `min` with a median above is invisible to the timing check on a quiet
        # machine -- both land near the healthy value -- so the difference has
        # to be asserted where it is deterministic: feed both a known spread of
        # samples and read which one each returns.
        # The names here are module-level, not `C.`-qualified: this file defines
        # them itself. Patched through the module it would raise NameError, and
        # that crash arrived after every assertion above had already printed OK.
        real_best = globals()["best_ms"]
        try:
            triples = [[5.0, 1.0, 3.0]]
            globals()["best_ms"] = lambda fn, per_round=9: triples[0].pop(0)
            low = globals()["min_ms"](lambda: None, rounds=3, per_round=1)
            triples[0] = [5.0, 1.0, 3.0]
            mid = globals()["median_ms"](lambda: None, rounds=3, per_round=1)[0]
        finally:
            globals()["best_ms"] = real_best
        check("min_ms returns the fastest sample it saw", low == 1.0, low)
        check("...while median_ms returns the middle one", mid == 3.0, mid)
        check("...so the two are not interchangeable", low != mid, (low, mid))

        # No clock in this one: the plans must be index-backed, line by line.
        # Checking the joined string is what my first version did, and on a
        # joined line `SCAN forge_events | ... USING INDEX` reads as indexed:
        # a lookahead that sees any later `USING` excuses an earlier scan.
        plans = _report_plans(svc.store)
        scanning = sorted(
            "%s: %s" % (name, line)
            for name, lines in plans.items() for line in lines
            if re.match(r"^SCAN\s+forge_events\b", line) and "USING" not in line)
        check("every statement report() leans on is index-backed",
              not scanning, str(scanning))
        check("the plan check found the statements it is about",
              len(plans) == 5 and all(plans.get(k) for k in plans),
              str(sorted(plans)))

        # Fresh service, so no walk has happened in this process at all.
        fresh = Service(db_path=os.path.join(tmp, "fresh.db"))
        try:
            grow(fresh.store, 40)
            first = fresh.report().get("chain") or {}
            check("a process that has never walked says a walk is due",
                  first.get("needs_full_check") is True, str(first.get("needs_full_check")))
            check("and reports no verdict of its own",
                  not first.get("last_full_check"), str(first.get("last_full_check")))
            fresh.verify_ledger()
            second = fresh.report().get("chain") or {}
            check("after walking, nothing further is due",
                  second.get("needs_full_check") is False, str(second.get("needs_full_check")))
            check("and the verdict it carries is ok",
                  (second.get("last_full_check") or {}).get("ok") is True,
                  str(second.get("last_full_check")))
            # One more write must make it due again -- that is the whole signal.
            chaining.append_event(fresh.store._conn, "call", {"tool": "later", "ok": True})
            third = fresh.report().get("chain") or {}
            check("one new row makes another check due",
                  third.get("needs_full_check") is True, str(third.get("needs_full_check")))
            check("and the old verdict is kept, bounded by the row it covered",
                  (third.get("last_full_check") or {}).get("ok") is True
                  and third.get("verified_up_to", 0) < third.get("up_to_id", 0),
                  "verified_up_to=%s up_to_id=%s" % (third.get("verified_up_to"),
                                                     third.get("up_to_id")))
        finally:
            fresh.store.close()
            fresh.missions.close()

        print("-- but it says whether the walk is overdue")
        rep = svc.report()
        chain = rep.get("chain") or {}
        check("report() still carries a chain block", isinstance(chain, dict), type(chain).__name__)
        # The point is not which words are used but that the cheap answer is
        # marked as partial. Testing for a specific key name was my own bad
        # assertion: it passed on a spelling and would have missed the meaning.
        # What has to be true is that a reader cannot mistake this for a walk.
        check("the block admits it has not walked every row",
              bool(chain.get("note")) and "walk" in str(chain.get("note")).lower(),
              str(chain.get("note")))
        check("and it does not present a verdict as if freshly computed",
              "ok" not in chain, str(sorted(chain)))
        # Whatever it says about coverage, it must not claim a fresh full walk
        # it did not perform. That is the one lie available here.
        check("a full check is offered, not silently skipped",
              chain.get("needs_full_check") in (True, False), str(chain.get("needs_full_check")))
        # (Whether one is due is tested above, on a service that has never walked.
        # Asserting it here would be asserting on a service this test walked on
        # purpose two lines earlier -- my own contradiction, not a defect.)

        print("-- the real verifier still walks every row")
        ver = svc.verify_ledger()
        check("verify_ledger answers ok", ver.get("ok") is True, str(ver)[:160])
        result = ver.get("result") or {}
        rows = svc.store._conn.execute("SELECT COUNT(*) FROM forge_events").fetchone()[0]
        check("and it walked all of them", result.get("chained_rows") == rows,
              "%s vs %s" % (result.get("chained_rows"), rows))

        print("-- the honest split survives a broken chain")
        # A row whose hash no longer matches its contents. report() must not
        # quietly stay green just because it stopped looking.
        victim = svc.store._conn.execute(
            "SELECT id FROM forge_events ORDER BY id LIMIT 1 OFFSET 600").fetchone()[0]
        svc.store._conn.execute(
            "UPDATE forge_events SET payload='{\"tampered\": true}' WHERE id=?", (victim,))
        svc.store._conn.commit()

        broken = svc.verify_ledger()
        check("verify_ledger sees the tampered row", broken.get("ok") is False, str(broken)[:160])
        check("and names it", any(b.get("row") == victim
                                  for b in (broken.get("result") or {}).get("breaks") or []),
              str(broken.get("result", {}).get("breaks"))[:200])

        after = svc.report().get("chain") or {}
        # This is the point of the split: a cheap report may be behind, but it
        # must never be *wrong in the reassuring direction*. Two ways it could
        # be: showing no verdict at all (reader assumes clean), or showing the
        # last verdict without its bad news.
        check("the cheap report keeps the last walk's bad news",
              (after.get("last_full_check") or {}).get("ok") is False,
              str(after.get("last_full_check")))
        check("and says when that walk happened",
              bool((after.get("last_full_check") or {}).get("checked_at")),
              str(after.get("last_full_check")))
        check("and names the first break it found",
              bool((after.get("last_full_check") or {}).get("first_break")),
              str(after.get("last_full_check")))
        # Once the walk covers everything, nothing new is due -- but the verdict
        # it carries is still 'not ok', so "nothing to check" cannot be confused
        # with "nothing wrong".
        check("a completed walk leaves no further check due",
              after.get("needs_full_check") is False, str(after.get("needs_full_check")))
        print("     (report says: needs_full_check=%s, last_full_check=%s)"
              % (after.get("needs_full_check"),
                 {k: (after.get("last_full_check") or {}).get(k)
                  for k in ("ok", "breaks", "gaps")}))
    finally:
        svc.store.close()
        svc.missions.close()
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
