"""Can a reader of `report()` tell a broken table from an unused one?

Under test: `ToolStore.report()["tables"]`.

The problem this file pins down is not "five tables are empty". It is that on
2026-11-03 the report had **no table-level view at all**: it listed tools,
events, versions, chain, memories and skills -- each picked by hand -- and said
nothing about `dependencies`, `baselines`, `topologies`, `mission_state` or
`chain_anchors`. A reader who noticed those were empty had no way to ask the
store about them, and "this table is broken" and "nobody has ever pressed this
button" produced the same evidence: zero rows.

So the view has to do two separate things and never blur them:

  measured  rows, straight from `COUNT(*)`, for every table in the file
  declared  what the table is for and what a zero means, kept as data next to
            the DDL and reported as a *claim* beside the measurement

A table with no declaration must say so out loud (`undeclared`) rather than
inherit a default, because the failure being fixed is exactly a default that
looked like an answer. A new table added tomorrow should show up here as
`undeclared` on the first run -- that is the check on this check.

Run from the service directory:
    python tests/regression/chk_tables.py

The database is a throwaway under %TEMP%; the production one is never opened.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc.store import ToolStore                        # noqa: E402
from autoforge_svc.service import Service                       # noqa: E402

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


def real_tables(store: ToolStore) -> set[str]:
    """Every table sqlite itself owns is not this store's business."""
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r["name"] for r in rows if not r["name"].startswith("sqlite_")}


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m10_tbl_")
    store = ToolStore(os.path.join(tmp, "t.db"))
    try:
        print("-- the view exists and covers the file")
        rep = store.report()
        view = rep.get("tables")
        check("report() carries a tables view", isinstance(view, dict), type(view).__name__)
        view = view or {}

        seen = real_tables(store)
        check("every table in the file is in the view",
              seen == set(view), f"missing={sorted(seen - set(view))} extra={sorted(set(view) - seen)}")
        check("sqlite's own bookkeeping table is not listed",
              not any(t.startswith("sqlite_") for t in view), str(sorted(view)))

        print("-- the number is measured, not restated")
        for name in sorted(view):
            direct = store._conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            got = view[name].get("rows")
            check(f"{name}: reported rows match a direct COUNT(*)", got == direct, f"{got} vs {direct}")

        print("-- the label is declared, and its absence is visible")
        # Guarded on a non-empty view on purpose. `undeclared == []` is true of
        # an empty dict, so without this line the check passes hardest when the
        # feature is missing entirely -- which is the shape of failure this
        # whole file exists to catch, and I hit it on the first run.
        check("the view is not empty to begin with", len(view) >= 8, str(len(view)))
        undeclared = sorted(n for n, e in view.items() if e.get("verdict") == "undeclared")
        check("no table is left without a declared meaning", undeclared == [], str(undeclared))
        for name in sorted(view):
            entry = view[name]
            check(f"{name}: carries a role", bool(entry.get("role")), str(entry))

        print("-- a zero says which kind of zero it is")
        # These two are the two different kinds, and they must not read alike.
        if "mission_state" in view:
            check("mission_state is 'never-triggered', not 'dormant'",
                  view["mission_state"].get("verdict") == "never-triggered",
                  str(view["mission_state"].get("verdict")))
        if "chain_anchors" in view:
            check("chain_anchors is 'never-triggered', not 'dormant'",
                  view["chain_anchors"].get("verdict") == "never-triggered",
                  str(view["chain_anchors"].get("verdict")))

        print("-- the tables deleted in this pass are gone from a new file")
        # Asked of sqlite, not of the view: "X is not in the view" is true of an
        # empty view, so testing it there would confirm nothing.
        check("dependencies is not created", "dependencies" not in seen, str(sorted(seen)))
        check("topologies is not created", "topologies" not in seen, str(sorted(seen)))

        print("-- a write moves a table from empty to ok")
        store.remember("k1", "v1", tags="t")
        view2 = store.report()["tables"]
        check("memory counts the row", view2["memory"]["rows"] == 1, str(view2["memory"]))
        check("memory is 'ok' once written", view2["memory"]["verdict"] == "ok",
              str(view2["memory"].get("verdict")))

        print("-- a table nobody declared shows up as undeclared")
        store._conn.execute("CREATE TABLE stray (x INTEGER)")
        store._conn.commit()
        view3 = store.report()["tables"]
        check("the stray table appears", "stray" in view3, str(sorted(view3)))
        check("and it is labelled undeclared", view3["stray"]["verdict"] == "undeclared",
              str(view3.get("stray")))
        check("its zero rows are still reported", view3["stray"]["rows"] == 0, str(view3["stray"]))
    finally:
        store.close()
        shutil.rmtree(tmp, ignore_errors=True)

    # A fresh file on purpose: the checks above have already called report()
    # several times, so a file that report() itself mutates has long since
    # settled and would look stable. This is the only place the first call can
    # be observed against the second.
    print("-- a report does not change the thing it reports")
    tmp2 = tempfile.mkdtemp(prefix="m10_tbl2_")
    fresh = ToolStore(os.path.join(tmp2, "t.db"))
    try:
        first = fresh.report()["tables"]
        second = fresh.report()["tables"]
        check("two consecutive reports describe the same tables",
              sorted(first) == sorted(second),
              f"only in the first={sorted(set(first) - set(second))}"
              f" only in the second={sorted(set(second) - set(first))}")
        same_rows = all(first[n]["rows"] == second[n]["rows"] for n in set(first) & set(second))
        check("and the same row counts", same_rows, str(
            {n: (first[n]["rows"], second[n]["rows"])
             for n in set(first) & set(second) if first[n]["rows"] != second[n]["rows"]}))
    finally:
        fresh.close()
        shutil.rmtree(tmp2, ignore_errors=True)

    # `chain_anchors` was the one table in this pass whose write path existed,
    # worked, and had no caller: `chaining.anchor()` -> `store.anchor_chain()`
    # -> (nothing). Leaving it "never-triggered" forever would have made the
    # verdict technically true and practically useless, so the button exists now
    # and this is what proves it moves the row.
    print("-- the button exists and it moves the table")
    tmp3 = tempfile.mkdtemp(prefix="m10_tbl3_")
    svc = Service(db_path=os.path.join(tmp3, "t.db"))
    try:
        before = svc.report()["tables"]["chain_anchors"]
        check("chain_anchors starts at never-triggered", before["verdict"] == "never-triggered",
              str(before))
        res = svc.anchor_ledger(note="m10 check")
        check("anchor_ledger answers ok", res.get("ok") is True, str(res)[:200])
        body = res.get("result") or {}
        check("it reports which row it froze", isinstance(body.get("up_to_id"), int), str(body))
        after = svc.report()["tables"]["chain_anchors"]
        check("the table now holds a row", after["rows"] == 1, str(after))
        check("and reads 'ok', not 'empty'", after["verdict"] == "ok", str(after))
        # The anchor is recorded twice on purpose: once in `chain_anchors` (the
        # fact) and once on the ledger (the index into it). If only the ledger
        # row existed, `verify_ledger` would not know a boundary had been
        # declared; if only the anchor existed, nothing would say who asked.
        ev = svc.my_history(kind="ledger_anchored")
        check("the act is on the ledger too", ev.get("count") == 1, str(ev)[:200])
        # A verifier reading the anchored file must still call it clean: an
        # anchor is a boundary, not a repair, so a break-free chain stays
        # break-free and `legacy_rows_anchored` becomes non-zero.
        ver = svc.verify_ledger()
        check("the ledger still verifies after anchoring", ver.get("ok") is True, str(ver)[:200])
        check("and the verifier sees the anchor",
              (ver.get("result") or {}).get("anchors") == 1,
              str((ver.get("result") or {}).get("anchors")))
    finally:
        svc.store.close()
        svc.missions.close()
        shutil.rmtree(tmp3, ignore_errors=True)

    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
