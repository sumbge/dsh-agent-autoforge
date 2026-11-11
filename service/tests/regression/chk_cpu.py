"""Does the CPU half tell the truth on a machine that cannot compile?

Under test: the eight `cpu_*` wire methods and the four lints they lean on
(`audit_ms_scale`, `audit_ms_from_seconds`, `audit_source_tree`, `preflight`).

The measured boundary first, because it decides what this file is allowed to
claim. This machine has **no C compiler**: `cpu_probe()` answers
`available: false` with the note "no C compiler on PATH (looked for gcc, cc,
clang, cl, tcc)", and gcc / cc / clang / cl / tcc / clang-cl / g++ / clang++ are
absent from PATH, from `C:\\Program Files\\LLVM\\bin`, from MSVC 2019 and 2022,
and from every MinGW spelling. So:

  what this suite CAN check
    * `cpu_probe` reports the absence instead of guessing, and its eleven keys
      are always there;
    * every refusal on the compile path says *which* thing is missing;
    * `cache_stats` counts the misses a failed compile leaves behind, so a
      machine that never built anything is distinguishable from one whose
      cache stopped working;
    * the four lints are pure source analysis and compile nothing.

  what it does NOT claim
    * that a kernel compiles, runs isolated, or that tuning finds a winner --
      none of those paths can be reached here, and a suite that asserted them
      would fail on this machine for a reason outside its subject;
    * that `preflight` can tell valid C from invalid C. It is a regex scanner
      over nine known-dangerous shapes and never parses the language; syntax
      errors are outside its range, which is a range and not a defect;
    * that `audit_ms_from_seconds` sees a value that arrives through a
      variable. That false negative is documented in its docstring and pinned
      below as *known*, not as passing.

Four defects were found and fixed while writing this suite; each is pinned so it
cannot come back:

  1. `audit_ms_from_seconds` had never fired. Its gate required a name matching
     `(?<![A-Za-z_])ms(?![A-Za-z_])` -- which the underscore in `from_ms`
     rejects -- *and* a name containing `from`/`as`/`to`. The two conditions
     are mutually exclusive, so all eight plausible door names failed. The one
     line its docstring promises to catch, `Duration.from_ms(perf_counter() -
     start)`, returned zero offences. Nothing in the package was hitting it:
     a silent sentinel, not a silent incident.
  2. `audit_ms_scale` flagged a correct conversion and reported one bug three
     times, because `ast.walk` hands out `ClassDef`/`FunctionDef` as statements
     and `_mentions_ms` walks the whole subtree -- a docstring saying "in ms"
     inside a def labelled every line of that body.
  3. `audit_source_tree` read files as `utf-8`, so a BOM survived as U+FEFF and
     `ast.parse` refused it. Thirteen files under `tests/regression/` were
     silently unauditable.
  4. A path that does not exist was skipped, so `cpu_units_audit(path="typo.py")`
     answered "No thousand-factor-against-ms offences in ..." -- the same
     sentence as having read the file and found it clean. An empty `source`
     did the same. "I did not look" and "I looked and it was clean" have to be
     different answers.

Run from the service directory:
    python tests/regression/chk_cpu.py

Nothing here touches the production database or the real task scheduler.
"""
from __future__ import annotations

import ast
import os
import shutil
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc import timing as T                            # noqa: E402
from autoforge_svc.service import Service                        # noqa: E402

FAILS: list[str] = []

# A kernel with a real entry point: `saxpy` is neither static nor `main`, so
# `entry_points()` finds it. Without one, the compile path refuses before it
# ever asks for a compiler and the refusal under test would be the wrong one.
SAXPY = (
    "void saxpy(const float *x, float *y, float a, long n) {\n"
    "    for (long i = 0; i < n; ++i) y[i] = a * x[i] + y[i];\n"
    "}\n"
)

PROBE_KEYS = {
    "available", "name", "logical_cores", "physical_cores", "arch", "compiler",
    "compiler_version", "openmp", "cache_bytes", "notes", "text",
}


def check(label: str, cond: bool, detail: object = "") -> None:
    # `detail: object` + repr, not `detail: str`: a failure path that formats
    # its own evidence must not be able to crash the suite. Measured elsewhere
    # in this tree -- a list in `detail` killed a suite before it printed any
    # verdict, which the mutation driver correctly scores as "crashed", not
    # "caught". An assertion that cannot report its evidence is not there.
    print(("  OK   " if cond else "  FAIL ") + label
          + (("  <- " + repr(detail)) if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


def skipped(label: str, why: str) -> None:
    """Not a pass and not a failure: a check this machine cannot reach.

    Printed, never counted. A skipped line that looked like OK would be the
    same lie as counting a look nobody took.
    """
    print("  SKIP " + label + "  (" + why + ")")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m9_cpu_")
    svc = Service(db_path=os.path.join(tmp, "t.db"))
    try:
        # ---- who is this machine -------------------------------------------
        probe = svc.cpu_probe()
        check("the probe answers with the eleven declared keys",
              set(probe) == PROBE_KEYS, sorted(set(probe) ^ PROBE_KEYS))
        check("available is a real boolean",
              isinstance(probe["available"], bool), type(probe["available"]).__name__)
        check("the CPU is named", bool(str(probe["name"]).strip()), probe["name"])
        check("the core counts are numbers a caller can use",
              isinstance(probe["logical_cores"], int) and probe["logical_cores"] >= 1
              and isinstance(probe["physical_cores"], int)
              and probe["physical_cores"] >= 1,
              (probe["logical_cores"], probe["physical_cores"]))
        check("physical cores never exceed logical ones",
              probe["physical_cores"] <= probe["logical_cores"],
              (probe["physical_cores"], probe["logical_cores"]))
        # The honesty check that matters on both kinds of machine: whichever
        # answer it gives, the *other* fields have to agree with it.
        if probe["available"]:
            check("a machine that says it can compile names a compiler",
                  bool(str(probe["compiler"]).strip()), probe["compiler"])
        else:
            said = " ".join(probe["notes"])
            check("a machine that says it cannot compile says what is missing",
                  "no C compiler" in said, probe["notes"])
            check("...and does not name a compiler it does not have",
                  not str(probe["compiler"]).strip(), probe["compiler"])
            check("...and the summary carries the same sentence",
                  "no C compiler" in str(probe["text"]), probe["text"][:160])

        # ---- refusals that hold on any machine ------------------------------
        #
        # These three are decided before a compiler is ever consulted, so they
        # are the same answer here as on a machine with gcc.
        no_entry = svc.cpu_compile("int main(void){return 0;}")
        check("source with nothing callable is refused by name",
              "no callable entry point" in no_entry, no_entry[:120])
        unplaceable = svc.cpu_compile(SAXPY, target="x86-64-v9")
        check("a target this CPU cannot be placed on is refused before the build",
              "Refusing to compile for x86-64-v9" in unplaceable, unplaceable[:140])
        bad_problem = svc.cpu_run_isolated(SAXPY, kind="no_such_problem")
        check("an unknown problem kind is refused, not attempted",
              "No such problem" in bad_problem, bad_problem[:120])

        # ---- refusals that only exist without a compiler --------------------
        if not probe["available"]:
            miss = svc.cpu_compile(SAXPY)
            check("a kernel with an entry point still cannot build here",
                  "Could not compile" in miss, miss[:120])
            check("...and the refusal names the missing compiler",
                  "no C compiler" in miss, miss[:160])
            check("...and names the way out (install, or point at a cache)",
                  "install gcc or clang" in miss and "AUTOFORGE_CPU_CACHE" in miss,
                  miss[-160:])
            run_miss = svc.cpu_run_isolated(SAXPY)
            check("isolated run refuses for the same stated reason",
                  "Could not compile" in run_miss and "no C compiler" in run_miss,
                  run_miss[:160])
            here = svc.cpu_runs_here("native")
            check("runs_here refuses to give a verdict instead of inventing one",
                  here.startswith("No verdict without a compiler"), here[:140])
            tuned = svc.cpu_tune("saxpy", size=64, n=1)
            check("tuning reports the search it could not perform",
                  "0 candidates" in tuned or "no candidate" in tuned, tuned[:160])
        else:
            skipped("no-compiler refusals", "this machine has a compiler")

        # ---- the cache counts what happened ---------------------------------
        #
        # A failed build is a miss, and a miss has to be visible: a cache that
        # stopped working and a machine that never built anything both read as
        # "0 hits" otherwise.
        before = svc.cpu_cache_stats()
        svc.cpu_compile(SAXPY)                     # no-op if it cannot build
        after = svc.cpu_cache_stats()
        check("cache_stats answers with a count, not an error",
              "hit(s)" in after or "Nothing compiled" in after, after[:120])
        check("a build attempt that cannot succeed is still counted",
              before != after or "0 hit(s), 0 miss(es)" not in after,
              (before[:60], after[:60]))

        # ---- the millisecond door, judged by name ---------------------------
        #
        # Defect 1. Two mutually exclusive gates meant this never fired.
        yes = ["from_ms", "to_ms", "as_ms", "ms", "fromMs", "_ms", "__ms__", "ms_from"]
        no = ["fromMillis", "millis", "params", "items", "timestamp", "msx", ""]
        wrong_true = [n for n in yes if not T._is_ms_door(n)]
        wrong_false = [n for n in no if T._is_ms_door(n)]
        check("every name that means milliseconds is read as one",
              wrong_true == [], wrong_true)
        check("...and no name that merely contains the letters is",
              wrong_false == [], wrong_false)

        # ---- the lint that never fired --------------------------------------
        caught = T.audit_ms_from_seconds(
            "def f(start):\n"
            "    return Duration.from_ms(perf_counter() - start)\n")
        check("a seconds timer handed to an ms constructor is caught",
              len(caught) == 1, [o.reason[:60] for o in caught])
        check("...and the offence says the value is 1000x low",
              caught and "1000x low" in caught[0].reason,
              caught[0].reason if caught else None)
        check("...and names the door that should have been used",
              caught and "from_wallclock" in caught[0].reason,
              caught[0].reason if caught else None)
        clean = T.audit_ms_from_seconds(
            "def f(start):\n"
            "    return Duration.from_seconds(perf_counter() - start)\n")
        check("the correct constructor is not flagged", clean == [], clean)
        # Stated, not swept: a value that passes through a variable is invisible
        # to this check. Pinned as a known false negative so a future reader
        # finds the limit written down rather than discovering it as a bug.
        via_var = T.audit_ms_from_seconds(
            "def f(start):\n"
            "    elapsed = perf_counter() - start\n"
            "    return Duration.from_ms(elapsed)\n")
        check("a value arriving through a variable is a known blind spot",
              via_var == [], "documented false negative, not a passing case")

        # ---- defects 2 and 3: compound statements, and the BOM --------------
        #
        # `def ms` documenting "in ms" and returning `seconds * 1000.0` is
        # doing the conversion correctly. It was flagged three times over.
        correct = (
            "class Duration:\n"
            "    def ms(self):\n"
            '        """This duration in ms, for a log line."""\n'
            "        return self.seconds * 1000.0\n"
        )
        check("a correct conversion is not reported as the bug",
              T.audit_ms_scale(correct) == [],
              [o.line for o in T.audit_ms_scale(correct)])
        one_bug = (
            "class Harness:\n"
            "    def report(self):\n"
            '        """Timings in ms."""\n'
            "        return f'{(t1 - t0) * 1e3:.3f} ms'\n"
        )
        found = T.audit_ms_scale(one_bug)
        check("a real offence is reported exactly once",
              len(found) == 1, [o.line_no for o in found])
        check("...and points at the line that does the arithmetic",
              found and found[0].line_no == 4,
              found[0].line_no if found else None)

        # A file saved with a BOM parses for Python, so it has to parse for the
        # auditor too. Thirteen files in this tree were unauditable instead.
        bom_dir = os.path.join(tmp, "bomdir")
        os.makedirs(bom_dir, exist_ok=True)
        bom_file = os.path.join(bom_dir, "with_bom.py")
        with open(bom_file, "wb") as fh:
            fh.write(b"\xef\xbb\xbf" + b'x = f"{(a - b) * 1e3:.2f} ms"\n')
        check("the fixture really does start with a BOM",
              open(bom_file, "rb").read(3) == b"\xef\xbb\xbf", None)
        bom_hits = T.audit_source_tree([bom_file])
        first = list(bom_hits.values())[0][0] if bom_hits else None
        check("a BOM file is audited rather than declared unparseable",
              first is not None and first.line_no == 1,
              first.reason if first else "no offence at all")
        check("...and does not answer 'source does not parse'",
              first is not None and "does not parse" not in first.reason,
              first.reason if first else None)

        # ---- defect 4: three answers for three situations -------------------
        missing = T.audit_source_tree([os.path.join(tmp, "never_existed.py")])
        reason = list(missing.values())[0][0].reason if missing else ""
        check("a path that is not there is reported, not skipped",
              "no such file or directory" in reason, reason[:110])
        check("...and is not reported as having been read clean",
              "offences in" not in reason, reason[:110])
        check("an offence with no line prints its reason rather than 'line 0'",
              T.audit_source_tree([os.path.join(tmp, "gone.py")])
              and list(T.audit_source_tree(
                  [os.path.join(tmp, "gone.py")]).values())[0][0].line == "", None)

        # The wire methods, both halves. A CPU reader and a GPU reader of the
        # same lint must not be able to disagree.
        # A path that exists and cannot be read is the third situation, and it
        # needs a fixture that actually lands on the read: a *directory* named
        # like a module, inside a tree being audited. `rglob("*.py")` yields
        # directories as well as files, so the loop reaches `read_text` on it
        # and gets `IsADirectoryError` -- an `OSError`, which is what the
        # branch under test catches. Measured 2026-11-06: without this fixture
        # the mutant that skips an unreadable entry escaped, because nothing in
        # the suite ever handed the audit something it could not open.
        tree = os.path.join(tmp, "tree")
        os.makedirs(tree, exist_ok=True)
        with open(os.path.join(tree, "ok.py"), "w", encoding="utf-8") as fh:
            fh.write("x = 1\n")
        os.makedirs(os.path.join(tree, "deep.py"), exist_ok=True)
        check("the unreadable fixture is a directory matching '*.py'",
              os.path.isdir(os.path.join(tree, "deep.py")), tree)
        tree_hits = T.audit_source_tree([tree])
        unread_reason = ""
        for name, offs in tree_hits.items():
            if "deep.py" in name and offs:
                unread_reason = offs[0].reason
        check("a path that exists but cannot be read is reported, not skipped",
              "could not be read" in unread_reason, unread_reason[:120])
        check("...and says nothing was audited rather than nothing was wrong",
              "nothing was audited" in unread_reason, unread_reason[:120])
        check("...while the readable file beside it is still audited clean",
              not any("ok.py" in name for name in tree_hits), list(tree_hits))
        check("...and the wire method surfaces the unreadable entry too",
              "could not be read" in svc.cpu_units_audit(path=tree),
              svc.cpu_units_audit(path=tree)[:160])

        cpu_missing = svc.cpu_units_audit(path=os.path.join(tmp, "typo.py"))
        gpu_missing = svc.gpu_units_audit(path=os.path.join(tmp, "typo.py"))
        check("cpu_units_audit says nothing was audited",
              "no such file or directory" in cpu_missing, cpu_missing[:120])
        check("gpu_units_audit gives the same answer for the same path",
              "no such file or directory" in gpu_missing, gpu_missing[:120])
        check("...and the two halves agree word for word",
              cpu_missing.split("typo.py", 1)[-1] == gpu_missing.split("typo.py", 1)[-1],
              (cpu_missing[-60:], gpu_missing[-60:]))
        cpu_empty = svc.cpu_units_audit()
        gpu_empty = svc.gpu_units_audit()
        check("an empty source is not a clean source",
              cpu_empty.startswith("Nothing to audit"), cpu_empty[:100])
        check("...and the GPU half says the same",
              gpu_empty.startswith("Nothing to audit"), gpu_empty[:100])
        real = os.path.join(tmp, "real_bug.py")
        with open(real, "w", encoding="utf-8") as fh:
            fh.write('print(f"{(t1 - t0) * 1e3:.3f} ms")\n')
        cpu_found = svc.cpu_units_audit(path=real)
        check("a file that really holds the bug is reported with its line",
              "line 1:" in cpu_found, cpu_found[:120])
        clean_file = os.path.join(tmp, "clean.py")
        with open(clean_file, "w", encoding="utf-8") as fh:
            fh.write("x = 1\n")
        genuine_clean = svc.cpu_units_audit(path=clean_file)
        check("a file that is clean answers differently from one never read",
              "No thousand-factor" in genuine_clean
              and "no such file" not in genuine_clean, genuine_clean[:120])

        # ---- preflight is a regex scanner, and says so ----------------------
        risky = svc.cpu_preflight("void f(){ char *p = malloc(4); strcpy(p, s); }",
                                  "risky")
        blocked = svc.cpu_preflight("void f(){ fork(); }", "forking")
        syntactically_wrong = svc.cpu_preflight("void f( {", "broken")
        clean = svc.cpu_preflight(
            "void saxpy(const float*x, float*y, float a, long n){"
            "for(long i=0;i<n;++i) y[i]=a*x[i]+y[i];}", "clean")
        check("a known-dangerous shape is found",
              "unsafe_str" in risky, risky[:160])
        check("a warning is counted as a warning, not as a block",
              "0 blocking, 1 warning" in risky, risky[:80])
        check("a blocking construct is marked as such",
              "1 blocking" in blocked and "[block] forks" in blocked, blocked[:160])
        # The range, written down. This scanner is nine regexes over
        # known-dangerous shapes; it never parses C. The proof is that source
        # which cannot possibly compile gets the *same* answer as source that
        # is fine -- writing that equality down is what keeps it a stated range
        # instead of a defect someone rediscovers.
        check("source that will not compile gets the same answer as clean source",
              syntactically_wrong == clean,
              (syntactically_wrong, clean))
        check("...and that answer is the scanner's no-findings line",
              syntactically_wrong.strip() == "preflight: nothing to flag",
              syntactically_wrong[:120])
        check("...so preflight never claims to have parsed the source",
              "does not parse" not in syntactically_wrong.lower()
              and "syntax" not in syntactically_wrong.lower(),
              syntactically_wrong[:120])

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
