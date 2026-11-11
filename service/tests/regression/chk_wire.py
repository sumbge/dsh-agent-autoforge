"""The wire surface: three lists that must agree, checked where the caller stands.

Why this file exists
--------------------
`anchor_ledger` was written, tested and shipped, and it did not work.

`Service.anchor_ledger` existed. `store.anchor_chain` existed. `lib/index.js`
registered `autoforge_anchor_ledger` pointing at method `anchor_ledger`. The
test `chk_tables.py` called `svc.anchor_ledger(...)` **directly** and passed.
And the bridge, calling over stdio, got:

    Error: unknown method: anchor_ledger

Because `main()` does not dispatch to attributes on the Service -- it dispatches
to `svc.methods`, a hand-written dict of seventy names living at
`service.py:236`. I added the method and never added the line, and every test I
had called the method rather than the door.

This is the fifth time in this project that a thing was proven at one layer and
used at another. So the check has to live at the door.

What is compared
----------------
1. **Every public Service method is on the wire table.** If somebody adds a
   method and forgets the dict entry, it fails here -- not in a bridge call
   three days later.
2. **Every name on the wire table is a real method.** A typo in the dict is a
   method that answers 500 lines later with `AttributeError`, and only when
   somebody happens to call it.
3. **Every method the shipped bridge calls is on the wire table.** This is the
   direction the actual failure went. It reads `lib/index.js` the way the plugin
   does, by pattern, so it fails when a tool is registered against a method that
   does not exist -- including the "wired in JS, never wired in Python" case.

4. **Every wire method that is meant for the model has a bridge tool.** This
   direction was left out at first, with the reasoning that a 1:1 mapping would
   force fake tools into the catalogue. Measured 2026-11-04: the reasoning was
   right about *some* methods and wrong about the cost. Deleting
   `autoforge_review_tool` from `lib/index.js` -- a tool that had just been
   written, reviewed and covered by 45 assertions -- changed nothing here,
   because nothing asked the reverse question. That is `anchor_ledger` again
   from the other side: proven at one layer, absent at another, and the suite
   was silent. The fix is not 1:1; it is an explicit, reasoned exemption list
   (`NOT_FOR_THE_MODEL`) and the assertion that `table - bridged` equals it
   exactly. An exemption has to be written down to be exempt.

Measured 2026-11-03 before the fix: `public methods 71, wire table 70,
callable but unwired ['anchor_ledger']`. After: both 71, and all 67 methods the
bridge calls resolve.
"""
import inspect
import os
import re
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLUGIN = os.path.dirname(SERVICE)
sys.path.insert(0, SERVICE)

from autoforge_svc.service import Service  # noqa: E402

#: Wire methods that are deliberately not offered to the model as tools, with
#: the reason each one is exempt. `table - bridged` must equal this set exactly:
#: a new wire method with no tool fails here until somebody either registers one
#: or writes down why it has none, and a stale entry fails here too.
#:
#: Every entry is a method the *harness* calls, not the model:
#:   context_brief       the brief is assembled into the system prompt by the
#:                       host hook; a tool that returns the prompt it is already
#:                       inside would be a tool the model could use to read its
#:                       own instructions mid-turn.
#:   report_bridge_tools the bridge reports its own catalogue upward on mount;
#:                       the reader is the service, not the model.
#:   tool_exposure       the hot/cold projection is computed per turn by the same
#:                       hook, from the turn text. A model calling it would be
#:                       asking to be shown a view of tools it already has.
NOT_FOR_THE_MODEL: dict[str, str] = {
    "context_brief": "assembled into the system prompt by the host hook",
    "report_bridge_tools": "the bridge reports its catalogue upward on mount",
    "tool_exposure": "computed per turn from the turn text by the same hook",
}

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print(f"  OK   {label}")
    else:
        print(f"  FAIL {label}  <- {detail}")
        FAILS.append(label)


def public_methods() -> set[str]:
    return {
        name
        for name, value in inspect.getmembers(Service, inspect.isfunction)
        if not name.startswith("_")
    }


def bridge_methods() -> list[str]:
    path = os.path.join(PLUGIN, "lib", "index.js")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    # `method: 'name'` is how every registration names its wire method. Any
    # future spelling that keeps that property keeps working here.
    return sorted(set(re.findall(r"method:\s*'([a-z_][a-z0-9_]*)'", src)))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m12_wire_")
    # The constructor itself is a step where wiring can be wrong: `self.methods`
    # is built inside `__init__`, so a method renamed or deleted makes the
    # Service fail to exist rather than fail a check. Measured 2026-11-04:
    # renaming `review_tool` produced `AttributeError` at construction, zero
    # FAIL lines, no TOTAL line, and the mutation read as "not caught" when in
    # truth the suite never got as far as stating anything.
    try:
        svc = Service(db_path=os.path.join(tmp, "wire.db"))
    except Exception as exc:                                   # noqa: BLE001
        check("the service can be constructed at all", False,
              f"{type(exc).__name__}: {exc}")
        print()
        print("TOTAL FAILURES:", len(FAILS), FAILS)
        return 1
    try:
        table = set(svc.methods)
        declared = public_methods()

        print("-- the wire table covers every method the service offers")
        # An empty table would satisfy "nothing is missing" trivially, so the
        # size is asserted first: this is the same shape of hole the table test
        # had on its first run, where `undeclared == []` was true of an empty
        # view and the check passed hardest with the feature absent.
        check("the wire table is populated", len(table) >= 60, str(len(table)))
        check("every public method is wired", not (declared - table), sorted(declared - table))
        check("nothing is wired that is not a method", not (table - declared), sorted(table - declared))

        print("-- and the shipped bridge only calls methods that are on it")
        called = bridge_methods()
        # Same guard, other side: an unreadable or moved plugin would yield an
        # empty list and every "missing" assertion below would pass while
        # checking nothing.
        check("the bridge registers a substantial tool set", len(called) >= 60, str(len(called)))
        check("every method the bridge calls exists on the wire", not [m for m in called if m not in table],
              [m for m in called if m not in table])
        check("the plugin file was actually found", os.path.isfile(os.path.join(PLUGIN, "lib", "index.js")),
              PLUGIN)

        print("-- and every wire method meant for the model is reachable from it")
        unbridged = sorted(table - set(called))
        check("no wire method is missing a tool except the ones written down",
              unbridged == sorted(NOT_FOR_THE_MODEL),
              [f"{m}: {NOT_FOR_THE_MODEL.get(m, 'NOT EXEMPT')}" for m in unbridged])
        check("every exemption says why it is one",
              all(reason.strip() for reason in NOT_FOR_THE_MODEL.values()),
              NOT_FOR_THE_MODEL)
        check("the exemption list is small -- this is not a place to park tools",
              len(NOT_FOR_THE_MODEL) <= 4, str(len(NOT_FOR_THE_MODEL)))
        check("review_tool is exposed to the model, not parked here",
              "review_tool" in called and "review_tool" not in NOT_FOR_THE_MODEL,
              sorted(called)[:5])

        print("-- the specific one that was broken")
        check("anchor_ledger is on the wire", "anchor_ledger" in table, sorted(table)[:5])
        check("and the bridge calls it by that name", "anchor_ledger" in called, called[:5])

        print("-- the dispatch door really answers it")
        # Calling the method proves nothing (that is what passed while the tool
        # was dead). Going through the same lookup `main()` uses is the point.
        fn = svc.methods.get("anchor_ledger")
        check("methods['anchor_ledger'] resolves", callable(fn), repr(fn))
        if callable(fn):
            res = fn(note="chk_wire: reached through the dispatch table, not the attribute")
            check("and it returns ok through that door", res.get("ok") is True, res)
            check("it froze a row", int((res.get("result") or {}).get("up_to_id") or 0) >= 0, res)
    finally:
        svc.store.close()
        svc.missions.close()

    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
