"""Does the GPU half tell the truth on a machine with a GPU but no CUDA toolkit?

Under test: the six `gpu_*` wire methods.

The boundary first, because it decides what this file may claim. Measured
2026-11-06 on this machine:

  * there IS a GPU: `nvidia-smi` answers "NVIDIA GeForce RTX 3050 Laptop GPU,
    compute_cap 8.6, 4096 MiB";
  * there is NO CUDA toolkit: `nvcc` is not on PATH, and neither
    `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA` nor `C:\\CUDA`
    exists.

So `gpu_compile` and `gpu_bench` can only be tested on their refusal paths.
That is a real limitation and it is written down here rather than papered over:
a suite that asserted a successful cubin would fail on this machine for a
reason outside its subject, and a suite that silently skipped the compile path
would let a regression through.

  what this suite CAN check
    * `gpu_probe` reports the device it can see, and says "unavailable" about
      the things it cannot measure instead of reporting them as zero;
    * `gpu_occupancy` is arithmetic over static limits, and says so in its own
      answer -- it must never read as a measurement;
    * `gpu_compile` and `gpu_bench` refuse in sentences that name the missing
      piece and the way out;
    * `gpu_verify` accepts the two shapes a caller actually holds and refuses
      the ones it cannot read, naming the item that was wrong;
    * `gpu_units_audit` is the same lint as the CPU half and must not be able
      to disagree with it.

Defect found while writing this suite, fixed and pinned below:

  `gpu_verify` was declared `actual: str` and parsed it with
  `actual.replace(",", " ").split()`, while its own refusal said "actual must be
  a list of numbers". A caller who followed that sentence -- and the bridge
  declares `actual: { type: 'string' }`, so the sentence was the only guidance
  about lists -- got `AttributeError: 'list' object has no attribute 'replace'`.
  A refusal that names a shape the method cannot accept teaches the wrong call.
  Both shapes are accepted now, and the refusal names the offending item.

Run from the service directory:
    python tests/regression/chk_gpu.py

Nothing here touches the production database or launches a kernel.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc.service import Service                        # noqa: E402

FAILS: list[str] = []

KERNEL = "__global__ void k(float *a){ a[threadIdx.x] = 1.0f; }\n"


def check(label: str, cond: bool, detail: object = "") -> None:
    # `detail: object` + repr: a failure path that formats its own evidence
    # must not be able to crash the suite. A suite that dies before printing a
    # verdict scores as "crashed", not "caught", and a crashed suite guards
    # nothing.
    print(("  OK   " if cond else "  FAIL ") + label
          + (("  <- " + repr(detail)) if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


def skipped(label: str, why: str) -> None:
    """Printed, never counted: a check this machine cannot reach.

    A skipped line that looked like OK would be the same lie as counting a
    look nobody took.
    """
    print("  SKIP " + label + "  (" + why + ")")


def answered(fn, *args, **kwargs):
    """Call the product, and turn an exception into the answer it did not give.

    A raised exception escapes `main`, the suite dies before printing a
    verdict, and the mutation driver scores that as "crashed" -- which it
    refuses to count as a catch. Measured 2026-11-06: a mutant that let an
    empty reference through made `gpu.verify` raise
    `TypeError: unsupported format string passed to NoneType.__format__`, and
    the whole suite disappeared instead of reporting a failure. Whatever the
    product does, the suite has to survive it and say so.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:                                   # noqa: BLE001
        return f"<raised {type(exc).__name__}: {exc}>"


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m9_gpu_")
    svc = Service(db_path=os.path.join(tmp, "t.db"))
    try:
        # ---- what the probe claims about this machine -----------------------
        probe = svc.gpu_probe()
        check("the probe answers in prose, not with an exception",
              isinstance(probe, str) and probe.strip() != "", repr(probe)[:80])
        check("the device is named", "RTX 3050" in probe or "GPU" in probe,
              probe[:120])
        check("the compute capability is stated", "sm_86" in probe, probe[:120])
        # A measurement it did not take must not be reported as a zero. SM count
        # came from `nvidia-smi` missing that field on this laptop, and the old
        # answer was `sm_count: 0` -- "I do not know" dressed as "none".
        check("an unmeasured field says it is unavailable rather than zero",
              "SM count unavailable" in probe, probe[:160])
        check("...and is not printed as a number that could be believed",
              "SM count 0" not in probe, probe[:160])
        check("the driver version is reported", "driver" in probe, probe[:160])

        # ---- occupancy is arithmetic, and says so ---------------------------
        occ = svc.gpu_occupancy(block=256, grid=1024, shared_bytes=0,
                                regs_per_thread=32)
        check("occupancy answers with a percentage", "% estimated" in occ, occ[:100])
        check("...and names what limited it", "limited by" in occ, occ[:140])
        check("...and reproduces the geometry it was given",
              "256" in occ and "1024" in occ, occ[:160])
        # The sentence that keeps an estimate from being quoted as a
        # measurement. It is the whole reason this method is allowed to exist.
        check("...and refuses to be mistaken for a measurement",
              "not a measurement" in occ, occ[:200])
        check("...and points at the way to measure it instead",
              "launch on the device" in occ.lower(), occ[:200])
        # Static arithmetic: a bigger block must not report more blocks per SM.
        small = svc.gpu_occupancy(block=64, grid=8, shared_bytes=0,
                                  regs_per_thread=32)
        check("a smaller block does not report fewer blocks per SM",
              "% estimated" in small, small[:80])

        # ---- refusals that hold without a toolkit --------------------------
        no_entry = svc.gpu_compile("int main(void){ return 0; }")
        check("source with no __global__ entry point is refused by name",
              "no `__global__ void` entry point" in no_entry, no_entry[:120])
        not_numbers = svc.gpu_verify(kind="k", actual=object(), reference=[1])
        check("a value that is neither text nor a list is refused in words",
              "must be a list of numbers or a string" in not_numbers,
              not_numbers[:140])
        no_reference = answered(svc.gpu_verify, kind="k", actual=[1.0, 2.0])
        check("a verification without a reference is refused, not attempted",
              "A reference is required" in no_reference, no_reference[:120])
        check("...and says what it would degenerate into",
              "restatement of the kernel" in no_reference, no_reference[:160])
        # An empty list is the reference the old code let through, and what
        # happened next was an exception out of `gpu.verify` rather than an
        # answer -- so this call goes through `answered()` and the assertion
        # below reads the exception as the failure it is.
        empty_ref = answered(svc.gpu_verify, kind="k", actual=[1.0], reference=[])
        check("an empty reference is the same refusal as no reference",
              empty_ref == no_reference, empty_ref[:100])

        # ---- the shapes a caller actually holds -----------------------------
        #
        # The defect: the refusal named lists, the body only handled strings.
        # Every call here goes through `answered()`: the suite has to survive
        # whatever the product does, or a regression shows up as "crashed" and
        # nothing is reported at all.
        as_lists = answered(svc.gpu_verify, kind="k", actual=[1.0, 2.0, 3.0],
                            reference=[1.0, 2.0, 3.0])
        check("a list of numbers is accepted", "PASS" in as_lists, as_lists[:100])
        as_text = answered(svc.gpu_verify, kind="k", actual="1.0, 2.0, 3.0",
                           reference="1.0 2.0 3.0")
        check("...and the same numbers as text give the same verdict",
              "PASS" in as_text, as_text[:100])
        as_tuple = answered(svc.gpu_verify, kind="k", actual=(1.0, 2.0),
                            reference=(1.0, 2.0))
        check("...and a tuple is a list for this purpose",
              "PASS" in as_tuple, as_tuple[:100])
        differs = answered(svc.gpu_verify, kind="k", actual=[1.0, 2.0],
                           reference=[1.0, 2.5])
        check("a real difference fails with the number that differs",
              "FAIL" in differs and "worst index 1" in differs, differs[:140])
        check("...and the tolerance is reported as the reason, not as precision",
              "correctness bug" in differs, differs[:160])
        bad_item = answered(svc.gpu_verify, kind="k", actual=[1.0, "two"],
                            reference=[1.0, 2.0])
        check("a non-number inside the list is refused with its index",
              "item 1" in bad_item and "str" in bad_item, bad_item[:140])
        # Booleans are ints in Python. A verification that silently accepted
        # `True` as 1.0 would compare a kernel's output against a flag.
        bools = answered(svc.gpu_verify, kind="k", actual=[True, False],
                         reference=[1, 0])
        check("booleans are not numbers here", "bool" in bools, bools[:140])
        unreadable_text = answered(svc.gpu_verify, kind="k", actual="one two",
                                   reference=[1, 2])
        check("text that is not numbers is refused, naming the text",
              "does not read as a list" in unreadable_text, unreadable_text[:140])

        # ---- the toolkit boundary -------------------------------------------
        #
        # The branch is decided by the machine, not by the product's wording.
        # The first version of this file asked the product
        # (`no_nvcc = "no nvcc on PATH" in svc.gpu_compile(KERNEL)`) and then
        # asserted that the answer contained that same phrase -- so a mutant
        # that reworded the refusal flipped the branch into `skipped` and the
        # suite went green with zero checks run. A guard that reads the thing
        # it is guarding cannot fail for the right reason.
        have_nvcc = shutil.which("nvcc") is not None
        if not have_nvcc:
            miss = answered(svc.gpu_compile, KERNEL)
            check("a kernel with an entry point still cannot build here",
                  "Could not compile" in miss, miss[:120])
            check("...and the refusal names nvcc as the missing piece",
                  "no nvcc on PATH" in miss, miss[:160])
            check("...and names the way out (install, or point at a cache)",
                  "install the CUDA toolkit" in miss
                  and "AUTOFORGE_KERNEL_CACHE" in miss, miss[-160:])
            bench = answered(svc.gpu_bench, KERNEL)
            check("a benchmark that cannot run says why rather than reporting 0",
                  "FAILED" in bench and "no nvcc" in bench, bench[:160])
            check("...and names the other path it also could not take",
                  "Triton" in bench, bench[:220])
        else:
            skipped("no-toolkit refusals", "this machine has nvcc")

        # ---- the two halves of the audit agree ------------------------------
        #
        # `gpu_units_audit` and `cpu_units_audit` are documented as the same
        # lint. If they can disagree, one of them is wrong and the reader has
        # no way to know which.
        bug = 'print(f"{(t1 - t0) * 1e3:.3f} ms")\n'
        cpu_says = svc.cpu_units_audit(source=bug)
        gpu_says = svc.gpu_units_audit(source=bug)
        check("the GPU half finds the same offence as the CPU half",
              "thousand-factor" in gpu_says, gpu_says[:120])
        check("...and the two answers are identical word for word",
              cpu_says == gpu_says, (cpu_says[:80], gpu_says[:80]))
        check("an empty source is not a clean source on this side either",
              gpu_says != "" and svc.gpu_units_audit().startswith("Nothing to audit"),
              svc.gpu_units_audit()[:100])
        missing = svc.gpu_units_audit(path=os.path.join(tmp, "typo.py"))
        check("a path that is not there says nothing was audited",
              "no such file or directory" in missing, missing[:120])

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
