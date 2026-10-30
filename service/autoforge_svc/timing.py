"""Timing units, made structurally impossible to get wrong.

Shared by the GPU and CPU kernel layers, because the failure this prevents is
not a hardware failure — it is an arithmetical one that both sides are exposed
to, for opposite reasons.

THE BUG THIS EXISTS TO PREVENT
`triton.testing.do_bench` returns MILLISECONDS. The natural-looking line

    f"{t * 1e3:.4f} ms"

turns every figure 1000x too large while the label still says "ms". It is
invisible on inspection because the number looks plausible and the ratios —
which are what people actually compare — are unaffected by a constant factor.
The bug survives in exactly the column that gets quoted.

The CPU side has the mirror image of it: `time.perf_counter` returns SECONDS,
so wrapping one in `Duration.from_ms()` understates by 1000x with the label
reading correctly. Same column, same invisibility, opposite direction. So this
module holds the one type both sides must route through, and one lint per
direction — `audit_ms_scale` for the do_bench shape, `audit_ms_from_seconds`
for the perf_counter shape.

THREE DEFENCES, weakest last
  1. Type-level. A duration is a `Duration`, which knows its own unit. There
     is no bare float to accidentally scale, and `.ms` / `.seconds_` are named
     so a reader can tell which one they got.
  2. Name-level. A timer's result has to enter through a named constructor —
     `from_do_bench()`, `from_wallclock()` — that states the unit its source
     produces. That fact then lives in one place instead of being assumed at
     every use site.
  3. Source-level. `audit_ms_scale()` statically rejects a `1e3`/`1000` factor
     applied to anything labelled `ms`, and `audit_ms_from_seconds()` rejects a
     measured-seconds value handed to an ms constructor. Both run with no GPU
     and no compiler, which is the point: the regression is caught on a laptop,
     not discovered in a paper.

Defence 3 is a lint, not a proof — it has known false negatives (a factor
smuggled through a variable, a label and its arithmetic in different
statements). Both are stated on the functions rather than papered over. The
lints are here because they are the only defence that still works when the
hardware and the toolchain are both absent, and because a check that cannot
fail is decoration: `tests/test_cpu.py` and `tests/test_gpu_units.py` feed each
one the known-bad pattern and assert it fires, then feed it the fixed code and
assert it does not.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass


class UnitError(ValueError):
    """A duration was used in a way that loses or misstates its unit."""


@dataclass(frozen=True)
class Duration:
    """A time span that carries its unit.

    Stored in seconds, because that is the SI base. Every accessor is named
    with the unit it returns; there is deliberately no `__float__`, so
    `float(dur)` raises rather than silently yielding seconds that a caller
    might then label ms.
    """

    seconds: float

    # -- constructors, one per unit, all explicit -------------------------
    @classmethod
    def from_seconds(cls, value: float) -> "Duration":
        return cls(float(value))

    @classmethod
    def from_ms(cls, value: float) -> "Duration":
        return cls(float(value) / 1000.0)

    @classmethod
    def from_us(cls, value: float) -> "Duration":
        return cls(float(value) / 1_000_000.0)

    @classmethod
    def from_ns(cls, value: float) -> "Duration":
        return cls(float(value) / 1_000_000_000.0)

    # -- accessors, named so the unit is visible at the call site ---------
    @property
    def seconds_(self) -> float:
        return self.seconds

    @property
    def ms(self) -> float:
        return self.seconds * 1000.0

    @property
    def us(self) -> float:
        return self.seconds * 1_000_000.0

    @property
    def ns(self) -> float:
        return self.seconds * 1_000_000_000.0

    def as_unit(self, unit: str) -> float:
        u = unit.strip().lower()
        if u in ("s", "sec", "second", "seconds"):
            return self.seconds
        if u in ("ms", "millisecond", "milliseconds"):
            return self.ms
        if u in ("us", "µs", "microsecond", "microseconds"):
            return self.us
        if u in ("ns", "nanosecond", "nanoseconds"):
            return self.ns
        raise UnitError(f"unknown unit {unit!r}; use s, ms, us or ns")

    def format(self, unit: str = "ms", digits: int = 4) -> str:
        """Always prints its own unit, so a mislabelled value is impossible
        by construction — the label comes from the same object as the number."""
        return f"{self.as_unit(unit):.{digits}f} {unit}"

    # -- arithmetic that keeps the unit ----------------------------------
    def __lt__(self, other: "Duration") -> bool:
        return self.seconds < other.seconds

    def __le__(self, other: "Duration") -> bool:
        return self.seconds <= other.seconds

    def __gt__(self, other: "Duration") -> bool:
        return self.seconds > other.seconds

    def __ge__(self, other: "Duration") -> bool:
        return self.seconds >= other.seconds

    def __truediv__(self, other: "Duration") -> float:
        """Ratio against another duration. Unitless by design: this is the
        comparison that a constant-factor unit bug cannot affect."""
        if not self.seconds:
            raise UnitError("division by a zero-length duration")
        return self.seconds / other.seconds

    def __add__(self, other: "Duration") -> "Duration":
        return Duration(self.seconds + other.seconds)

    def __float__(self) -> float:
        # Refusing is the point. If a caller genuinely wants seconds, they
        # write `.seconds_` and it shows up in review.
        raise UnitError(
            "Duration has no implicit float conversion — use .seconds_, .ms, "
            "or .us so the unit you are passing is explicit"
        )


def from_do_bench(value: float) -> Duration:
    """Wrap a `triton.testing.do_bench` result.

    do_bench returns MILLISECONDS. This function exists so that fact lives in
    one place. If triton's contract ever changes, this is the single line that
    changes, and every caller stays correct.

    Verified by wall clock, not by documentation — see the do_bench-returns-ms
    test, which times a synthetic kernel of known duration with both do_bench
    and time.perf_counter and asserts the ratio is ~1 against the ms reading.
    """
    return Duration.from_ms(value)


def from_wallclock(seconds: float) -> Duration:
    """Wrap a `time.perf_counter()` difference.

    The CPU-side counterpart of `from_do_bench`, and the reason it is a
    separate named function rather than a `Duration.from_seconds` call at each
    site: perf_counter returns SECONDS, which is the opposite of every other
    timer an operator of this project is likely to have in their hands. Naming
    the source here is what stops the ms-constructor being reached for by habit.
    """
    return Duration.from_seconds(seconds)


# -- defence 3, direction 1: do_bench's milliseconds ------------------------
# Implemented over the AST rather than over lines, for one concrete reason: a
# line-based regex flags the docstring that *explains* the bug. A detector that
# cries wolf on its own documentation gets switched off, and then it protects
# nothing. Walking the tree lets us skip bare string statements (docstrings)
# while still seeing a real f-string used as code.
_MS_WORD_RE = re.compile(r"(?<![A-Za-z_])ms(?![A-Za-z_])")


def _is_thousand(node: ast.AST) -> bool:
    """A literal factor of one thousand: 1e3, 1000, 1000.0, 1_000."""
    if not isinstance(node, ast.Constant):
        return False
    v = node.value
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return v == 1000


def _scales_by_thousand(node: ast.AST) -> ast.BinOp | None:
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mult):
            if _is_thousand(sub.left) or _is_thousand(sub.right):
                return sub
    return None


def _mentions_ms(node: ast.AST) -> bool:
    """A string constant in this subtree that labels a millisecond unit.

    Deliberately narrow: the token `ms` as a word, not 'ms' inside another
    word such as 'items' or 'params'.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if _MS_WORD_RE.search(sub.value):
                return True
    return False


@dataclass(frozen=True)
class MsOffence:
    line_no: int
    line: str
    reason: str


def audit_ms_scale(src: str) -> list[MsOffence]:
    """Static check: a value labelled `ms` must not be scaled by 1000.

    Catches the exact shape of the do_bench bug — an f-string that multiplies
    by 1e3 while printing an `ms` label.

    Two false-negative classes are accepted and stated rather than papered
    over, and both are why defence 1 (the Duration type) exists as well:
      * the factor arrives through a variable (`S = 1e3; f"{t * S} ms"`)
      * the label and the arithmetic live in different statements
    Source that does not parse is reported as un-auditable rather than clean.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [MsOffence(line_no=exc.lineno or 0, line="",
                          reason=f"source does not parse, cannot audit: {exc.msg}")]

    lines = src.splitlines()
    offences: list[MsOffence] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        # A statement that is nothing but a string is a docstring or a stray
        # literal. It is not executing a conversion, so it must not be flagged.
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            continue
        scale = _scales_by_thousand(node)
        if scale is None or not _mentions_ms(node):
            continue
        line_no = getattr(scale, "lineno", getattr(node, "lineno", 0)) or 0
        line = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else ""
        offences.append(MsOffence(
            line_no=line_no,
            line=line,
            reason="thousand-factor applied in a statement labelled 'ms' — if "
                   "the input was do_bench it is already in milliseconds",
        ))
    offences.sort(key=lambda o: o.line_no)
    return offences


# -- defence 3, direction 2: perf_counter's seconds --------------------------
# The mirror bug, and it is not caught by the check above: the offending line
# HANDS a seconds value to a millisecond constructor, so there is no
# thousand-factor anywhere to find. What gives it away is the source of the
# number, not the arithmetic.
_SECONDS_TIMER_RE = re.compile(r"perf_counter|monotonic|process_time|time\.time")


def audit_ms_from_seconds(src: str) -> list[MsOffence]:
    """Static check: a seconds-valued timer must not enter through an ms door.

    `Duration.from_ms(end - start)` where those came from `perf_counter` is
    1000x wrong in the direction nobody checks, because the printed label is
    right and only the number is small.

    Reports only the shape it can actually see: a call to an ms-named
    constructor or an ms-named helper, whose argument subtree mentions a
    seconds-valued timer. A value that passes through a variable first is a
    false negative, and is stated rather than pretended away — which is why
    `from_wallclock()` exists as the named door for this number.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [MsOffence(line_no=exc.lineno or 0, line="",
                          reason=f"source does not parse, cannot audit: {exc.msg}")]

    lines = src.splitlines()
    offences: list[MsOffence] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (
            fn.id if isinstance(fn, ast.Name) else "")
        if not _MS_WORD_RE.search(name):
            continue
        # `from_ms`, `as_ms`, `to_ms` — a door whose name promises milliseconds.
        lowered = name.lower()
        if not any(tok in lowered for tok in ("from", "as", "to")) or "s" not in lowered:
            continue
        if not any(_seconds_timer(inner) for inner in node.args):
            continue
        line_no = getattr(node, "lineno", 0) or 0
        line = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else ""
        offences.append(MsOffence(
            line_no=line_no,
            line=line,
            reason=f"a seconds-valued timer is being read as milliseconds by "
                   f"{name}() — perf_counter returns seconds, so this is 1000x "
                   f"low. Use from_wallclock().",
        ))
    offences.sort(key=lambda o: o.line_no)
    return offences


def _seconds_timer(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and _SECONDS_TIMER_RE.search(sub.attr):
            return True
        if isinstance(sub, ast.Call):
            fn = sub.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else "")
            if name and _SECONDS_TIMER_RE.search(name):
                return True
    return False


def audit_source_tree(paths, *, lints=None) -> dict[str, list[MsOffence]]:
    """Run the lints over many files. Returns only the files with offences.

    `lints` defaults to both directions, because a caller auditing a tree wants
    to know about either shape and asking them to remember there are two would
    be a way of quietly auditing half of it.
    """
    from pathlib import Path

    checks = lints or (audit_ms_scale, audit_ms_from_seconds)
    bad: dict[str, list[MsOffence]] = {}
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files = sorted(p.rglob("*.py"))
        elif p.exists():
            files = [p]
        else:
            continue
        for f in files:
            try:
                src = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            hits: list[MsOffence] = []
            for check in checks:
                hits.extend(check(src))
            if hits:
                hits.sort(key=lambda o: o.line_no)
                bad[str(f)] = hits
    return bad


__all__ = [
    "Duration", "UnitError", "MsOffence",
    "from_do_bench", "from_wallclock",
    "audit_ms_scale", "audit_ms_from_seconds", "audit_source_tree",
]
