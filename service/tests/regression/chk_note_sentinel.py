"""The wire must be able to say "leave that field alone".

The distinction this suite pins down exists at the owner and was thrown away at
the door:

    MissionStore.note(..., next_step=None, blocked_on=None)
        None -> do not touch     "" -> clear it
    Service.mission_note(..., next_step="", blocked_on="")   # before
        both always passed positionally, so `None` was unreachable

Measured 2026-11-06, two fresh missions, one call each carrying only a note:

    wire  path: next_step ""            blocked_on ""
    owner path: both preserved

So "here is where it got to" silently cleared the plan and released the block.
`mission.py:266-274` argues at length that `""` must clear a block -- the
exemption would otherwise be a one-way door -- and that argument is only true
if `None` also exists. The Service signature erased half of it.

Three independent questions, because agreement alone proves nothing:
  1. an omitted field survives a note that does not mention it;
  2. an explicit "" still clears (the escape hatch the owner documents);
  3. changing one field leaves the other exactly as it was.

A suite that only asserted (1) would pass on a method that ignores both
parameters forever.
"""
import json
import os
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)

FAILS = []


def check(label, cond, detail=""):
    if cond:
        print("OK   %s" % label)
    else:
        FAILS.append(label)
        print("FAIL %s  <- %s" % (label, detail))


def main():
    home = tempfile.mkdtemp(prefix="chk_note_sentinel_")
    os.environ["AUTOFORGE_HOME"] = home
    from autoforge_svc.service import Service

    svc = Service()

    def fresh(text):
        m = svc.mission_open(text, next_step="the original next step",
                             blocked_on="waiting on the operator")
        return int(m["id"])

    # -- 1. omission is not a deletion ------------------------------------
    a = fresh("note only")
    svc.mission_note(a, "news only")
    got = svc.mission_show(a)
    check("a note that mentions no field changes no field",
          got.get("next_step") == "the original next step",
          "next_step=%r" % (got.get("next_step"),))
    check("and the block it was under survives too",
          got.get("blocked_on") == "waiting on the operator",
          "blocked_on=%r" % (got.get("blocked_on"),))
    check("the note itself was recorded",
          any(r.get("kind") == "note" for r in
              (svc.mission_history(a) or {}).get("history") or []),
          "no note row")

    # -- 2. the documented escape hatch still works -----------------------
    b = fresh("explicit clear")
    svc.mission_note(b, "released", blocked_on="")
    cleared = svc.mission_show(b)
    check("an explicit empty string still clears the block",
          cleared.get("blocked_on") == "",
          "blocked_on=%r" % (cleared.get("blocked_on"),))
    check("and it did not take the next step with it",
          cleared.get("next_step") == "the original next step",
          "next_step=%r" % (cleared.get("next_step"),))

    # -- 3. the two fields move independently -----------------------------
    c = fresh("one field")
    svc.mission_note(c, "plan changed", next_step="the new plan")
    moved = svc.mission_show(c)
    check("setting next_step leaves the block alone",
          moved.get("blocked_on") == "waiting on the operator",
          "blocked_on=%r" % (moved.get("blocked_on"),))
    check("and next_step really is the new one",
          moved.get("next_step") == "the new plan",
          "next_step=%r" % (moved.get("next_step"),))

    d = fresh("clear then move")
    svc.mission_note(d, "released", blocked_on="")
    svc.mission_note(d, "and planned", next_step="the plan after release")
    both = svc.mission_show(d)
    check("a block once cleared stays cleared through a later note",
          both.get("blocked_on") == "",
          "blocked_on=%r" % (both.get("blocked_on"),))
    check("and the later note's next_step landed",
          both.get("next_step") == "the plan after release",
          "next_step=%r" % (both.get("next_step"),))

    # -- 4. the signature itself carries the distinction ------------------
    import inspect
    sig = inspect.signature(Service.mission_note)
    for name in ("next_step", "blocked_on"):
        default = sig.parameters[name].default
        check("%s defaults to None, which is the owner's 'do not touch'" % name,
              default is None, "default=%r" % (default,))
    owner = inspect.signature(type(svc.missions).note)
    check("the door and the owner agree on both sentinels",
          all(sig.parameters[n].default is owner.parameters[n].default
              for n in ("next_step", "blocked_on")),
          "service=%r owner=%r" % (
              [sig.parameters[n].default for n in ("next_step", "blocked_on")],
              [owner.parameters[n].default for n in ("next_step", "blocked_on")]))

    # -- 5. what this suite does not claim -------------------------------
    print("note  it does not test that the bridge describes both parameters;")
    print("note  that is a sentence in lib/index.js and no assertion here reads it.")

    print("TOTAL FAILURES: %d %s" % (len(FAILS), FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
