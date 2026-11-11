"""Does the per-turn memory block tell the truth about itself?

Under test: `Service._memory_lines` and `ToolStore.{recall, memory_for_injection}`.

Three things go wrong quietly here, and all three were measured on 2026-11-02:

1. `recall('')` -- "show me what I have" -- bumped the recall counter of every
   entry it displayed. But `memory_for_injection` orders by that counter, and
   its own docstring says folding the two together "turn[s] `recalls` into
   'how many turns has this entry existed' and eras[es] the only signal for
   which facts the agent actually reaches for". The comment describes a hazard
   the code beside it commits.
2. An entry longer than `MEMORY_ENTRY_CHARS` was cut with a bare `…` and nothing
   else: no count, no way to get the rest. The cut lands at a fixed offset, and
   the offset is usually just before the part that says what to do.
3. A fact can go stale while staying in the prompt. Counting that is not this
   file's job; making a stale count detectable is.

Added 2026-11-06, when the `tags` column turned out to hold two shapes:

4. `tags` came back as a `str` from a row written over the wire and as a `list`
   from a row written by a caller holding a list -- one column, declared
   `list[str]`, with 37 of 38 production rows holding a JSON string. No reader
   iterated it, which is the only reason nothing had broken; the first one that
   did would walk characters. The suite now pins one shape for all three writers
   (wire string, caller list, no tags) *and* for rows written before the
   normaliser existed, because a fix that only covers new writes leaves the old
   rows to bite the next reader.
5. `forget` answers `{"forgotten": bool}`, which is honest: "it was not there"
   is exactly what the caller asked. The suite pins that reading, and pins that
   a padded key is a different key.

Run from the service directory:
    python tests/regression/chk_memory.py

The database is a throwaway under %TEMP%; the production one is never opened.
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

from autoforge_svc.service import Service                      # noqa: E402

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


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m9_mem_")
    svc = Service(db_path=os.path.join(tmp, "t.db"))
    try:
        long_value = "claimgate 回归怎么跑：" + ("很长的说明。" * 60) + "最后这句才是能执行的那句。"
        svc.remember(key="alpha-long", value=long_value, tags="t")
        svc.remember(key="beta-short", value="很短。", tags="t")
        svc.remember(key="gamma-other", value="另一样东西。", tags="t")

        # Reach for one fact on purpose. Only that one should climb.
        hit = svc.recall("gamma", 10)
        check("a query finds the one entry it names",
              [r["key"] for r in hit] == ["gamma-other"], str([r["key"] for r in hit]))

        before = {r["key"]: r["recalls"] for r in svc.recall("", 50)}
        # The listing above already ran once, so take the baseline AFTER it and
        # ask again: a second plain listing must move nothing.
        before = {r["key"]: r["recalls"] for r in svc.recall("", 50)}
        svc.recall("", 50)
        after = {r["key"]: r["recalls"] for r in svc.recall("", 50)}
        grew = {k: (before[k], after[k]) for k in after if after[k] != before[k]}
        check("listing what I have does not count as reaching for it",
              grew == {}, f"these climbed: {grew}")

        # And a real reach still climbs. Reach for alpha twice so it is
        # genuinely the most-reached entry -- the fixture already reached for
        # gamma once above, and with equal counters the tie breaks on recency
        # (gamma was written last), which would test nothing about reaching.
        svc.recall("alpha", 10)
        svc.recall("alpha", 10)
        now = {r["key"]: r["recalls"] for r in svc.recall("", 50)}
        check("a real reach still climbs", now["alpha-long"] > after["alpha-long"],
              f"{after['alpha-long']} -> {now['alpha-long']}")
        check("alpha-long is now the most-reached fact",
              now["alpha-long"] > max(v for k, v in now.items() if k != "alpha-long"),
              str(now))

        # The block injected into every prompt.
        lines = svc._memory_lines()
        block = "\n".join(lines)
        check("the block is within its own budget, heading included",
              len(block) <= Service.MEMORY_BUDGET_CHARS + 400, f"{len(block)} chars")

        alpha_line = next((ln for ln in lines if "alpha-long:" in ln), "")
        check("the long entry is present", bool(alpha_line), block[:200])
        check("it was cut, because it is longer than one slot",
              len(alpha_line) < len(long_value), f"{len(alpha_line)} vs {len(long_value)}")
        check("the cut says how to get the rest",
              "recall this key" in alpha_line or "recall('alpha-long')" in alpha_line,
              alpha_line[-160:])
        check("the cut says how much is missing",
              "(+" in alpha_line and "chars" in alpha_line, alpha_line[-160:])
        check("the cut is visible rather than silent", "…" in alpha_line or "..." in alpha_line,
              alpha_line[-160:])

        # The short entry is not cut and gains no pointer noise.
        beta_line = next((ln for ln in lines if "beta-short:" in ln), "")
        check("a short entry is shown whole", beta_line.rstrip().endswith("很短。"), repr(beta_line))

        # Ordering: the entry actually reached for comes first among equals.
        body = [ln for ln in lines[1:] if not ln.strip().startswith("…and")]
        keys = [ln.strip().split(":", 1)[0] for ln in body]
        check("the most-reached entry leads the block", keys and keys[0] == "alpha-long", str(keys))

        # ---- the tags column has one shape, whatever wrote the row --------
        #
        # Measured 2026-11-06: 37 of 38 rows in the production database held
        # `'"test,probe"'` -- a JSON *string* -- because `Service.remember` took
        # the comma-separated form a human types and handed it straight to `_j`,
        # while the column is declared `list[str]`. A row written by a caller
        # holding a list held a list. Nothing had broken, and the only reason is
        # that no reader iterates the value: the first one that did would walk
        # characters, and `"beta" in "alpha,beta"` is True by substring rather
        # than by membership. The reader below is the one that would have done it.
        svc.store.remember("tag-string", "kept from a string", "alpha,beta")
        svc.store.remember("tag-list", "kept from a list", ["alpha", "beta"])
        svc.store.remember("tag-none", "kept with none")
        shaped = {r["key"]: r for r in svc.store.recall("", 50)}
        for name in ("tag-string", "tag-list", "tag-none"):
            check("%s comes back as a list" % name,
                  isinstance(shaped[name]["tags"], list),
                  "%r (%s)" % (shaped[name]["tags"], type(shaped[name]["tags"]).__name__))
        check("a string written over the wire lands as two names",
              shaped["tag-string"]["tags"] == ["alpha", "beta"], shaped["tag-string"]["tags"])
        check("a list written by a caller lands the same way",
              shaped["tag-list"]["tags"] == ["alpha", "beta"], shaped["tag-list"]["tags"])
        check("no tags is an empty list, not a missing key",
              shaped["tag-none"]["tags"] == [], shaped["tag-none"]["tags"])
        check("membership is by name, not by substring",
              "beta" in shaped["tag-string"]["tags"], shaped["tag-string"]["tags"])

        # What is actually in the column, not what a reader makes of it.
        #
        # Without this the factory defect is invisible: the reader normalises, so
        # a row holding `"alpha,beta"` and a row holding `["alpha","beta"]` both
        # come back as the list. The two rows are still not the same row -- an
        # audit script, a `sqlite3` prompt, or the next language to read this
        # database sees a string where the schema says list. That is how the
        # defect was found on 2026-11-06, and the fix has to be visible from the
        # same seat.
        raw_string = svc.store._conn.execute(
            "SELECT tags FROM memory WHERE key='tag-string'").fetchone()[0]
        raw_list = svc.store._conn.execute(
            "SELECT tags FROM memory WHERE key='tag-list'").fetchone()[0]
        check("a string written over the wire is stored as a JSON list",
              json.loads(raw_string) == ["alpha", "beta"], raw_string)
        check("...and both writers leave the same bytes in the column",
              raw_string == raw_list, "%r vs %r" % (raw_string, raw_list))

        # A row written before the normaliser existed. A fix that only applies to
        # new writes leaves exactly these rows to bite the next reader, and the
        # production database is 37 of them.
        svc.store._conn.execute(
            "UPDATE memory SET tags=? WHERE key='tag-list'", (json.dumps("legacy,x"),))
        svc.store._conn.commit()
        legacy = {r["key"]: r for r in svc.store.recall("", 50)}["tag-list"]
        check("a row holding a JSON string of tags reads back as a list",
              legacy["tags"] == ["legacy", "x"],
              "%r (%s)" % (legacy["tags"], type(legacy["tags"]).__name__))
        injected = [r for r in svc.store.memory_for_injection() if r["key"] == "tag-list"][0]
        check("the brief path reads that row the same way",
              injected["tags"] == ["legacy", "x"], injected["tags"])

        # ---- forget answers honestly --------------------------------------
        #
        # `{"forgotten": False}` is a real answer to a real question, not a
        # refusal dressed as success -- a missing key is exactly what the caller
        # asked about. What would be a defect is a refusal wearing `ok: true`;
        # the wire shape is checked by the suite that owns the wire.
        check("forget says whether it was there",
              svc.forget("tag-none") == {"forgotten": True},
              svc.forget("tag-none"))
        check("...and says so honestly the second time",
              svc.forget("tag-none") == {"forgotten": False})
        check("forget keys match exactly, the way remember writes them",
              svc.forget("  tag-string  ") == {"forgotten": False},
              "a padded key is a different key")
        check("the fact is still there after a near miss",
              any(r["key"] == "tag-string" for r in svc.store.recall("", 50)), None)

        # A key that is a *prefix* of another key. This is the case a loose match
        # actually gets wrong, and the padded-key probe above does not reach it:
        # `LIKE '%  tag-string  %'` contains spaces that no stored key has, so a
        # substring mutant survives that probe untouched. Measured 2026-11-06 --
        # it did, and this is the assertion that caught it.
        svc.remember("tag", "the short one", "t")
        svc.remember("tag-string-longer", "the long one", "t")
        check("forgetting the short key leaves the longer one alone",
              svc.forget("tag") == {"forgotten": True}
              and any(r["key"] == "tag-string-longer" for r in svc.store.recall("", 50)),
              sorted(r["key"] for r in svc.store.recall("", 50)))

        print()
        print("TOTAL FAILURES:", len(FAILS), FAILS)
        return 0 if not FAILS else 1
    finally:
        try:
            svc.store.close()
            svc.missions.close()
        except Exception:                                      # noqa: BLE001
            pass
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
