"""A default Service() must not silently ignore the ledger sitting beside it.

Measured 2026-11-06, with no `AUTOFORGE_HOME`:

    preset ledger   <base>/autoforge/data/autoforge.db   3 rows    8192 bytes
    Service() db    <base>/autoforge/autoforge.db        empty   110592 bytes
    report() keys   29, none of them mentioning the neighbour

Every read afterwards goes through `self.db_path`, and nothing else looks. So a
session started without the preset opened a second, empty ledger next to a full
one and reported on the empty one -- which is exactly how the five stray rows on
this machine came to exist. The docstring on `store._default_home` warns about a
different version of this (a directory shadowing the package); the same shape,
one layer up.

The rule this file defends
--------------------------
The default location wins when it holds a file. When it holds nothing and a
populated sibling exists, the sibling is adopted **and the adoption is recorded**
(a `home_adopted` ledger row and `report()["home"]["adopted_from"]`), because
"this service lives in data/" and "this service adopted the ledger in data/" are
the same sentence about different facts. An empty sibling is never adopted: it
is a file, not a ledger, and moving to it would only relocate the confusion.

What this file can and cannot see
--------------------------------
It checks the decision and the receipt. It does not prove that adoption is the
right policy -- an operator who wanted a fresh ledger would call `Service(path)`
and say so, and that path bypasses all of this by construction (checked below).
"""
import os
import sys
import tempfile
import shutil

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)

FAILS = []


def check(label, cond, detail=""):
    if cond:
        print("OK   %s" % label)
    else:
        FAILS.append(label)
        print("FAIL %s  <- %s" % (label, detail))


def fresh():
    """A private `LOCALAPPDATA` with `<base>/autoforge` already there."""
    base = tempfile.mkdtemp(prefix="chk_adopt_")
    root = os.path.join(base, "autoforge")
    os.makedirs(root)
    return base, root


def seed(path, rows=3):
    """A real ledger at `path`, built by the class that owns the schema."""
    from autoforge_svc.store import ToolStore

    os.makedirs(os.path.dirname(path), exist_ok=True)
    st = ToolStore(path)
    for i in range(rows):
        st.log_event("seed_row", {"i": i})
    return st


def main():
    keep = {k: os.environ.get(k) for k in ("LOCALAPPDATA", "AUTOFORGE_HOME")}
    os.environ.pop("AUTOFORGE_HOME", None)
    made = []
    try:
        from autoforge_svc.store import home_dir, db_file
        from autoforge_svc.service import Service, _adopt_sibling_ledger

        # -- 1. nothing anywhere: the default location is used --------------
        base, root = fresh()
        made.append(base)
        os.environ["LOCALAPPDATA"] = base
        want = os.path.join(root, "autoforge.db")
        got, why = _adopt_sibling_ledger(home_dir())
        check("with no ledger anywhere the default location is chosen",
              os.path.normcase(got) == os.path.normcase(want), got)
        check("and nothing is claimed to have been adopted", why == "", why)

        # -- 2. an empty sibling is not a ledger ---------------------------
        base, root = fresh()
        made.append(base)
        os.environ["LOCALAPPDATA"] = base
        os.makedirs(os.path.join(root, "data"))
        open(os.path.join(root, "data", "autoforge.db"), "wb").close()
        got, why = _adopt_sibling_ledger(home_dir())
        check("an empty sibling file is not adopted", why == "", why)
        # `want` is recomputed, not reused: it was built from the first block's
        # `root`, and a fixture that outlives its temp dir turns a passing
        # assertion into a comparison between two different directories -- which
        # is what this line did on its first run.
        want_here = os.path.join(root, "autoforge.db")
        check("and the default location is still the answer",
              os.path.normcase(got) == os.path.normcase(want_here),
              "%r vs %r" % (got, want_here))

        # -- 2b. a schema-only sibling is not history either ---------------
        # The case above (a zero-byte file) cannot distinguish the size guard
        # from the row-count guard: sqlite answers `no such table` for an empty
        # file and both paths fall back to the default location. A *valid* ledger
        # with zero events is the fixture that separates them, and it is the one
        # a real store leaves behind when it is opened and nothing happens. Found
        # by mutation D3 escaping the first version of this file.
        from autoforge_svc.store import ToolStore

        base, root = fresh()
        made.append(base)
        os.environ["LOCALAPPDATA"] = base
        os.makedirs(os.path.join(root, "data"))
        ToolStore(os.path.join(root, "data", "autoforge.db"))   # schema, no rows
        got, why = _adopt_sibling_ledger(home_dir())
        check("a sibling with a schema but no events is not adopted", why == "", why)
        check("and the default location is still the answer",
              os.path.normcase(got) == os.path.normcase(
                  os.path.join(root, "autoforge.db")), got)

        # -- 3. a populated sibling wins, and says so ----------------------
        base, root = fresh()
        made.append(base)
        os.environ["LOCALAPPDATA"] = base
        sib = os.path.join(root, "data", "autoforge.db")
        seed(sib)
        got, why = _adopt_sibling_ledger(home_dir())
        check("a populated sibling is adopted",
              os.path.normcase(got) == os.path.normcase(sib), got)
        check("and the reason names the same file", why == sib, why)

        # -- 4. the service acts on it, and files a receipt ----------------
        before = len([1 for _ in os.scandir(root)])
        svc = Service()
        check("Service() lands on the populated sibling",
              os.path.normcase(svc.db_path) == os.path.normcase(sib), svc.db_path)
        check("and does not create a second ledger beside it",
              not os.path.exists(os.path.join(root, "autoforge.db")),
              "a second ledger appeared")
        home = svc.report().get("home") or {}
        check("report() says where the state is", home.get("dir") == home_dir(),
              "%r" % (home,))
        check("and that this one was adopted", home.get("adopted_from") == sib,
              "%r" % (home,))
        kinds = [r[0] for r in svc.store._conn.execute(
            "SELECT kind FROM forge_events WHERE kind = 'home_adopted'").fetchall()]
        check("the adoption is on the ledger, not only in the reply",
              len(kinds) == 1, "%d home_adopted rows" % len(kinds))

        # -- 5. an existing default is never overridden --------------------
        base, root = fresh()
        made.append(base)
        os.environ["LOCALAPPDATA"] = base
        seed(os.path.join(root, "autoforge.db"), rows=1)
        seed(os.path.join(root, "data", "autoforge.db"), rows=5)
        got, why = _adopt_sibling_ledger(home_dir())
        check("an existing default is not overridden by a fuller sibling",
              why == "" and os.path.normcase(got) == os.path.normcase(
                  os.path.join(root, "autoforge.db")), got)

        # -- 6. an explicit path bypasses the whole question ---------------
        base, root = fresh()
        made.append(base)
        os.environ["LOCALAPPDATA"] = base
        seed(os.path.join(root, "data", "autoforge.db"))
        chosen = os.path.join(base, "elsewhere", "mine.db")
        svc2 = Service(db_path=chosen)
        check("an explicit db_path is obeyed, adoption or not",
              os.path.normcase(svc2.db_path) == os.path.normcase(chosen),
              svc2.db_path)
        check("and it is not reported as adopted",
              (svc2.report().get("home") or {}).get("adopted_from") == "",
              "%r" % (svc2.report().get("home"),))

        # -- 7. the probe asks the same question (2026-11-06 defect) -------
        from autoforge_svc import tars_port

        base, root = fresh()
        made.append(base)
        os.environ["LOCALAPPDATA"] = base
        check("tars_port asks the store rather than deriving its own home",
              os.path.normcase(str(tars_port._home_dir(None))) == os.path.normcase(
                  home_dir()),
              "%r vs %r" % (str(tars_port._home_dir(None)), home_dir()))
        check("a caller-supplied home still wins",
              str(tars_port._home_dir(os.path.join(base, "set"))) ==
              os.path.join(base, "set"),
              str(tars_port._home_dir(os.path.join(base, "set"))))

        # -- 8. the probe reports the file the service uses, not a directory --
        # Measured 2026-11-06: after `_home_dir` started delegating, the probe
        # said `no autoforge.db under <home>` while the service was reading
        # `<home>/data/autoforge.db`. Two readers of the same question, one
        # answer each. `store.db_file` now owns it and both ask.
        base, root = fresh()
        made.append(base)
        os.environ["LOCALAPPDATA"] = base
        sib2 = os.path.join(root, "data", "autoforge.db")
        seed(sib2, rows=2)
        svc3 = Service()
        line = [ln for ln in tars_port.probe_host(timeout=0.05)
                if ln.startswith("store:")][0]
        check("the probe reports the file the service is reading",
              os.path.normcase(svc3.db_path) in os.path.normcase(line),
              "%r vs %r" % (svc3.db_path, line))
        check("and says the file was inherited, not merely found",
              "inherited" in line, line)
        check("db_file and the service agree on the path",
              os.path.normcase(db_file()[0]) == os.path.normcase(svc3.db_path),
              "%r vs %r" % (db_file()[0], svc3.db_path))
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for b in made:
            shutil.rmtree(b, ignore_errors=True)

    print("TOTAL FAILURES: %d %s" % (len(FAILS), FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
