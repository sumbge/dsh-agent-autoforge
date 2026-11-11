"""The ported code has to be *callable*, or it is a file in a tree.

`vendor/tars` holds 32 verbatim files from Intelligence Indeed (Apache-2.0) and
`tars_port.py` reads them, verifies them and can build a `PortedSkillSystem`
from them. Measured 2026-11-03: nothing on the wire could reach any of it. The
only tars-shaped method was `skill_port(source="tars")`, which *writes* nine
files into the skills directory -- so the one tars operation this branch
exposed was the one with a side effect, and the read-only half had no entry
point at all.

That is the shape this file exists to catch, and it is the same shape as
`anchor_ledger` (item 6) and `controls.bind` (P0-2): a capability that is fully
implemented, fully tested as a function, and unreachable from the model.

Three tools, split the way upstream splits them, because the split is the
honest part:

    tars_probe         read-only: what was borrowed and does it work HERE
    tars_skills        read-only: the index, or one body in full
    tars_port_skills   writes nine files, so its name has to say so

A tool named like a read that quietly mints nine files would make its
`effect_signature` a lie, and that signature is what the autonomy gate keys off.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc.autonomy.confirm import disabled_freedoms          # noqa: E402
from autoforge_svc.autonomy.policy import SUPERVISED                 # noqa: E402
from autoforge_svc.autonomy.scopes import METHOD_SCOPES, scope_for   # noqa: E402
from autoforge_svc.service import Service                            # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    if cond:
        print("  OK   " + label)
    else:
        print("  FAIL " + label + " <- " + str(detail))
        FAILS.append(label)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m23_tars_")
    home = os.path.join(tmp, "home")
    os.makedirs(home, exist_ok=True)
    svc = Service(db_path=os.path.join(home, "autoforge.db"))
    svc.store.close()
    svc.missions.close()
    # Re-open with the skills directory pointed at the fixture, so a port writes
    # nine files into the tempdir instead of into the real library.
    os.environ["AUTOFORGE_SKILLS_DIRS"] = os.path.join(home, "skills")
    os.environ["AUTOFORGE_HOME"] = home
    svc = Service(db_path=os.path.join(home, "autoforge.db"))
    try:
        print("-- the three methods exist on the wire")
        for name in ("tars_probe", "tars_skills", "tars_port_skills"):
            check(f"{name} is on the dispatch table", name in svc.methods,
                  sorted(k for k in svc.methods if "tars" in k))

        print("-- and each declares what it touches")
        check("tars_probe is read_only", scope_for("tars_probe") == "read_only",
              scope_for("tars_probe"))
        check("tars_skills is read_only", scope_for("tars_skills") == "read_only",
              scope_for("tars_skills"))
        # The whole reason the split exists: one of the three writes files, and
        # a signature that says otherwise would let it through a gate that was
        # switched on to stop exactly this.
        check("tars_port_skills is not read_only",
              scope_for("tars_port_skills") == "local_write",
              scope_for("tars_port_skills"))
        check("a read-only one passes a SUPERVISED gate",
              disabled_freedoms(scope_for("tars_probe"), SUPERVISED) == [],
              disabled_freedoms(scope_for("tars_probe"), SUPERVISED))

        print("-- tars_probe answers with evidence, not a claim")
        probe = svc.tars_probe()
        check("tars_probe answers ok", bool(probe.get("ok")), json.dumps(probe)[:200])
        text = str(probe.get("text") or "")
        check("it names the upstream it came from", "Intelligence Indeed" in text
              or "tars" in text.lower(), text[:120])
        check("it reports the verbatim check", "copy-time digest" in text or
              "MANIFEST" in text, text[:200])
        check("it reports the host probe", "store:" in text, text[-300:])
        check("and it carries the probe lines as a list", isinstance(probe.get("probe"), list),
              type(probe.get("probe")).__name__)
        check("the vendor check is actually clean",
              bool((probe.get("vendor") or {}).get("ok")),
              json.dumps(probe.get("vendor"))[:200])

        print("-- tars_skills discloses progressively")
        index = svc.tars_skills()
        check("no argument returns the index", bool(index.get("ok")), json.dumps(index)[:160])
        names = index.get("names") or []
        check("the index lists the borrowed procedures", len(names) >= 5, names)
        check("and each is named the way my library will name it",
              all(n.startswith("tars-") for n in names), names[:4])
        first = str(names[0])[len("tars-"):]
        one = svc.tars_skills(name=first)
        check("a name returns that body", bool(one.get("ok")), json.dumps(one)[:160])
        check("the body is longer than its index line", len(str(one.get("text") or "")) > 200,
              len(str(one.get("text") or "")))
        missing = svc.tars_skills(name="no-such-procedure")
        check("an unknown name says so instead of guessing",
              missing.get("ok") is False, json.dumps(missing)[:160])
        check("and it suggests nothing rather than everything",
              len(missing.get("names") or []) <= 5, missing.get("names"))

        print("-- tars_port_skills writes, and is idempotent")
        skills_dir = os.path.join(home, "skills")
        before = len([f for f in os.listdir(skills_dir) if f.endswith(".md")]) \
            if os.path.isdir(skills_dir) else 0
        ported = svc.tars_port_skills()
        check("it reports ok", bool(ported.get("ok")), json.dumps(ported)[:200])
        written = ported.get("written") or []
        check("it wrote the procedures", len(written) >= 5, written)
        after = len([f for f in os.listdir(skills_dir) if f.endswith(".md")])
        check("they are on disk", after == before + len(written), f"{before} -> {after}")
        check("each landed under the tars- prefix",
              all(str(w).startswith("tars-") for w in written), written[:4])
        again = svc.tars_port_skills()
        # `overwrite_body` defaults to False here, so the second run writes
        # nothing and skips nine. My first draft asserted "nine written, none
        # skipped", which is the overwrite=True behaviour -- it demanded the
        # data loss the default exists to prevent. The property that matters is
        # the file count on disk: idempotent means the second run adds no files.
        after_again = len([f for f in os.listdir(skills_dir) if f.endswith(".md")])
        check("re-running is not a second copy", after_again == after,
              f"{after} -> {after_again}")
        check("and it says what it skipped rather than reporting silence",
              len(again.get("skipped") or []) == len(written),
              json.dumps(again)[:200])
        check("nothing was overwritten without being asked",
              len(again.get("written") or []) == 0, again.get("written"))
        body = open(os.path.join(skills_dir, str(written[0]) + ".md"), encoding="utf-8").read()
        check("the body carries its provenance in the file itself",
              "Intelligence Indeed" in body or "Apache" in body, body[:160])

        print("-- and the writing one is gated where the reading one is not")
        # The split is only worth anything if the gate can tell them apart. Use
        # a policy that refuses filesystem writes: SUPERVISED does not (it keeps
        # `may_write_filesystem`), which is why my first draft of this block was
        # a tautology about a constant rather than a test of my code.
        from dataclasses import replace as _dc_replace
        from autoforge_svc.autonomy.policy import AutonomyPolicy
        no_write = _dc_replace(SUPERVISED, may_write_filesystem=False)
        svc.registry.policy = no_write
        refused = svc._call_method("tars_port_skills")
        check("tars_port_skills is refused when writes are off",
              isinstance(refused, dict) and refused.get("awaiting_confirmation") is True,
              json.dumps(refused)[:200])
        check("and the refusal names the freedom it wanted",
              (refused.get("needed") or []) == ["may_write_filesystem"],
              refused.get("needed"))
        allowed = svc._call_method("tars_skills")
        check("tars_skills still answers under the same policy",
              isinstance(allowed, dict) and allowed.get("ok") is True,
              json.dumps(allowed)[:120])
        svc.registry.policy = None

        print("-- every tars method is on the bridge's callable list")
        lib = os.path.join(os.path.dirname(SERVICE), "lib", "index.js")
        src = open(lib, encoding="utf-8").read()
        for name in ("tars_probe", "tars_skills", "tars_port_skills"):
            check(f"lib/index.js registers autoforge_{name}",
                  f"method: '{name}'" in src, name)
    finally:
        svc.store.close()
        svc.missions.close()
        for key in ("AUTOFORGE_SKILLS_DIRS", "AUTOFORGE_HOME"):
            os.environ.pop(key, None)
        shutil.rmtree(tmp, ignore_errors=True)

    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
