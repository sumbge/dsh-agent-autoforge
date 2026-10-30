"""Skills: procedural knowledge the agent keeps as files and finds by use.

Tools answer "what can be done here". Skills answer "how this is done here" —
the sequence, the pitfalls, the thing that went wrong last time. The agent had
plenty of the first and none of the second, which is why it re-derived the same
approach every session and could not tell a procedure it had run forty times
from one it had never tried.

Two decisions shape everything below.

*The files own the content.* A skill is markdown with a frontmatter header, on
disk, in a directory a human can open and edit. No part of a skill's text lives
only in the database, so a skill outlives the agent that wrote it.

*The database owns the history.* `loads` counts how often the agent actually
opened a skill. That counter is what makes retrieval behavioural rather than
lexical: a procedure that keeps getting reached for outranks one that merely
reads as relevant. The counter is written by the act of loading, so it cannot
drift away from what happened.

Progressive disclosure: the prompt carries a menu — name, when-to-use, load
count — and never the bodies. A library of forty skills costs forty lines, and
the agent pays for a body only when it decides to read one. That decision is
the retrieval event, which is why the menu is not a summary of the library but
the index of it.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .store import _default_home

SKILL_SUFFIX = ".md"
ARCHIVE_DIR = ".archive"
ENV_DIRS = "AUTOFORGE_SKILLS_DIRS"

#: A skill package's entry point. The library is a tree of categories, each
#: holding skill directories, each holding one of these — the layout every
#: skill on this machine actually uses.
SKILL_INDEX_NAME = "SKILL.md"

#: Directories a skill scan must never descend into. Ported from Hermes'
#: `skill_utils.EXCLUDED_SKILL_DIRS`, which is the implementation that already
#: reads this exact directory tree correctly: VCS and editor metadata, virtualenvs
#: and dependency trees, and every cache directory a Python project grows.
EXCLUDED_SKILL_DIRS = frozenset({
    ".git", ".github", ".hub", ".archive", ".venv", "venv",
    "node_modules", "site-packages", "__pycache__",
    ".tox", ".nox", ".pytest_cache", ".mypy_cache", ".ruff_cache",
})

#: Progressive-disclosure areas *inside a skill package*: loaded explicitly by
#: name, never discovered as skills of their own. Pruned only when the directory
#: holding them is itself a skill (see `iter_skill_index_files`) — otherwise a
#: legitimate category called `scripts/` or `templates/` would vanish.
SKILL_SUPPORT_DIRS = frozenset({"references", "templates", "assets", "scripts"})

MENU_LINE_CHARS = 160      # per-entry cap for a skill that has been used
MENU_COLD_LINE_CHARS = 80  # per-entry cap for one that never has
MENU_BUDGET_CHARS = 1400   # whole-menu cap, for the same reason memory has one
#: Never-loaded skills are collapsed into a single count line once there are
#: more than this many. Below it they are listed, because a handful of unused
#: skills is a library being built; above it they are sediment, and listing
#: them costs every turn to say nothing.
MENU_COLD_MAX_LISTED = 5


class SkillError(ValueError):
    """A skill file that cannot be used as written."""


# ---------------------------------------------------------------------------
# how proven a procedure is, without a counter that churns
# ---------------------------------------------------------------------------
#: Loads at which a skill counts as well proven. Not a tuning knob so much as
#: the point where the claim changes kind: run a few times is "this works here",
#: run five times is "this is how it is done here". Both render as fixed
#: strings, so a load that stays inside a tier does not move the menu — which
#: matters because the menu is prompt *prefix*, and the exact count was,
#: measurably, re-billing a whole conversation on every load.
PROVEN_LOADS = 5


def provenance(loads: int) -> str:
    """The one-line claim a menu may make about `loads` previous runs."""
    if not loads:
        return "never used"
    return "well proven" if loads >= PROVEN_LOADS else "used before"


# ---------------------------------------------------------------------------
@dataclass
class Skill:
    """One skill, as read from disk, with its usage history from the store."""

    name: str
    description: str
    when_to_use: str
    body: str
    path: str
    source: str = "user"
    tags: list[str] = field(default_factory=list)
    loads: int = 0
    last_loaded: float | None = None

    def menu_line(self, cold: bool = False) -> str:
        """One line for the prompt: what it is, when to reach for it, how proven.

        How proven it is, not how many times it ran. A procedure used twenty
        times is a different bet from one that has never run, and the menu would
        be lying if it presented them as equals — but `20x` versus `19x` is not a
        different bet, and the exact number costs far more than it says. This
        line rides in the system prompt, which is the *prefix* of every request,
        so a count that moves on every load re-bills the whole conversation
        uncached, every turn. Measured: one load moved 1,332 characters of a
        6,457-character prompt, and everything behind them.

        A tier moves the text at most twice in a skill's life, and each move is
        news the agent can act on: this has run here, or this is well proven.

        `cold` is the never-loaded case, and it gets a shorter line. The menu is
        an index, and an index should be sized by the chance it gets consulted:
        a skill that has never been opened is a weaker bet than one that has, so
        it earns less of the prompt. The full when_to_use is still on disk and
        still returned by skill_list — this shortens the index, not the skill.
        """
        when = self.when_to_use or self.description
        cap = MENU_COLD_LINE_CHARS if cold else MENU_LINE_CHARS
        if len(when) > cap:
            when = when[:cap].rstrip() + " ..."
        return f"    {self.name} ({provenance(self.loads)}) -- {when}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "when_to_use": self.when_to_use, "tags": list(self.tags),
            "source": self.source, "path": self.path, "loads": self.loads,
            "last_loaded": self.last_loaded, "body_chars": len(self.body),
        }


# ---------------------------------------------------------------------------
# frontmatter -- hand-parsed, no dependency, and strict enough to catch a file
# that would otherwise be silently misread as having no description.
# ---------------------------------------------------------------------------
def split_frontmatter(text: str, *, origin: str = "") -> tuple[dict[str, str], str]:
    """Split `---`-fenced frontmatter from the body.

    Raises SkillError rather than guessing when the fence is missing or
    unterminated: a file whose header was silently ignored would present as a
    skill with no description, which is a routing bug that looks like a content
    bug, and those take an afternoon to tell apart.
    """
    lines = text.splitlines()
    where = f" ({origin})" if origin else ""
    if not lines or lines[0].strip() != "---":
        raise SkillError(
            f"skill{where} does not start with a '---' frontmatter line")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return _parse_meta(lines[1:i], origin=origin), _join_body(lines[i + 1:])
    raise SkillError(f"skill{where} has frontmatter with no closing '---'")


def _join_body(lines: list[str]) -> str:
    return "\n".join(lines).strip("\n")


def _parse_meta(lines: list[str], *, origin: str = "") -> dict[str, str]:
    meta: dict[str, str] = {}
    where = f" ({origin})" if origin else ""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise SkillError(
                f"skill{where} frontmatter line is not 'key: value': {raw!r}")
        key, _, value = line.partition(":")
        k = key.strip().lower().replace("-", "_")
        v = value.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]                      # quoted value: strip the quotes
        meta[k] = v
    if not meta.get("description"):
        raise SkillError(f"skill{where} has no description, so it cannot be routed")
    return meta


def parse_tags(raw: str) -> list[str]:
    """Tags are comma-separated, because that is what a human types."""
    return [t.strip() for t in (raw or "").split(",") if t.strip()]


def render_skill_text(name: str, description: str, when_to_use: str,
                      tags: list[str], body: str) -> str:
    """Serialise a skill back to the on-disk format. Round-trips through parse."""
    lines = ["---", f"name: {name}", f"description: {description}"]
    if when_to_use:
        lines.append(f"when_to_use: {when_to_use}")
    if tags:
        lines.append("tags: " + ", ".join(tags))
    lines.append("---")
    lines.append("")
    lines.append(body.rstrip() or f"# {name}")
    return "\n".join(lines) + "\n"


def valid_name(name: str) -> bool:
    """Lower-case, hyphens/underscores/digits — a filename a human can type.

    Enforced rather than sanitised: silently rewriting `My Skill!` into
    `my-skill` would leave the agent unable to find the skill it just wrote,
    which reads as a broken write rather than as a rejected name.
    """
    if not name or len(name) > 64:
        return False
    if name[0] == "-" or name[0] == "_":
        return False
    return all(c.islower() or c.isdigit() or c in "-_" for c in name)


# ---------------------------------------------------------------------------
def default_skill_dirs(cwd: str | None = None, home: str | None = None
                       ) -> list[tuple[str, str]]:
    """Where skills are looked for, most specific first: (source, path).

    Project before user, because a project's conventions should beat a general
    one of the same name — and a shadowed skill is reported rather than hidden,
    so "why is my edit not taking effect" has an answer on the first look.
    """
    env = os.environ.get(ENV_DIRS)
    if env:
        out = []
        for i, part in enumerate(env.split(os.pathsep)):
            if part.strip():
                out.append((f"env{i}" if i else "env", os.path.abspath(part.strip())))
        return out
    home = home or _default_home()
    cwd = cwd or os.getcwd()
    dirs = [(("project"), os.path.join(os.path.abspath(cwd), "skills"))]
    dirs.append(("user", os.path.join(home, "skills")))
    return dirs


def iter_skill_index_files(directory: str | Path) -> Iterator[Path]:
    """Walk `directory` yielding sorted `SKILL.md` paths, pruning as it goes.

    Ported from Hermes' `skill_utils.iter_skill_index_files` because that is the
    scanner that already reads this machine's skill tree correctly, and this
    module's first version did not: it globbed `*.md` one level deep, which sees
    zero of the 243 skills on disk. Every skill here is a *package*
    (`<category>/<name>/SKILL.md`), and a flat glob cannot see a tree.

    Pruning happens on the directory list rather than on the results, so
    excluded subtrees are never even listed — the difference between reading a
    tree and filtering a mess.

    The support-dir rule carries the subtlety worth keeping: `references/`,
    `templates/`, `assets/` and `scripts/` are cut *only* when the directory
    holding them is itself a skill (`SKILL.md` present). A category that happens
    to be named `scripts` stays discoverable, and a skill's own support files —
    which may well contain an archived `SKILL.md` package — are not promoted to
    skills.
    """
    root = str(directory)
    matches: list[str] = []
    for here, dirs, files in os.walk(root, followlinks=True):
        is_a_skill = SKILL_INDEX_NAME in files
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDED_SKILL_DIRS
            and not (is_a_skill and d in SKILL_SUPPORT_DIRS)
        ]
        if SKILL_INDEX_NAME in files:
            matches.append(os.path.join(here, SKILL_INDEX_NAME))
    for path in sorted(matches):
        yield Path(path)


def iter_skill_files(directory: str | Path) -> Iterator[Path]:
    """Every skill file under `directory`, in both layouts this library supports.

    1. Flat `<name>.md` at the top level — what `SkillLibrary.write` produces,
       and therefore what the agent's own skills look like.
    2. `<category>/<name>/SKILL.md` anywhere below — the package layout, which is
       how skills are distributed and how the whole library on this machine is
       arranged.

    Flat first, so a skill the agent wrote itself outranks a packaged one of the
    same name — the agent's own edit is the more specific statement of intent.
    `SkillLibrary.scan` reports the collision either way, so this order decides
    which file wins, not whether the loser is noticed.
    """
    root = Path(directory)
    for path in sorted(root.glob("*" + SKILL_SUFFIX)):
        # A half-written file from an interrupted `write` (see the temp-then-move
        # dance there) must never be indexed.
        if path.name.endswith(SKILL_SUFFIX + ".tmp"):
            continue
        yield path
    yield from iter_skill_index_files(root)


# ---------------------------------------------------------------------------
class SkillLibrary:
    """Skills on disk, plus the usage history that makes them rankable."""

    def __init__(self, store: Any = None, dirs: list[tuple[str, str]] | None = None,
                 cwd: str | None = None) -> None:
        self.store = store
        self.dirs = dirs if dirs is not None else default_skill_dirs(cwd=cwd)
        self._skills: dict[str, Skill] = {}
        self.errors: list[str] = []       # unreadable files, with the reason
        self.shadowed: list[tuple[str, str]] = []   # (name, winning path)

    # -- reading ----------------------------------------------------------
    def dirs_present(self) -> list[tuple[str, str]]:
        return [(s, p) for s, p in self.dirs if os.path.isdir(p)]

    def scan(self) -> list[Skill]:
        """Read every skill directory. Content comes from disk, always.

        Errors are collected, never raised: one malformed file among forty must
        not cost the agent the other thirty-nine, and a silent skip would let
        the agent believe it has no skill for a task it has one for.
        """
        found: dict[str, Skill] = {}
        self.errors = []
        self.shadowed = []
        for source, directory in self.dirs:
            if not os.path.isdir(directory):
                continue
            # Both layouts, and a tree walk rather than a flat glob: every skill
            # is a package under a category, and the previous `glob("*.md")` saw
            # none of them. See `iter_skill_files`.
            for path in iter_skill_files(directory):
                try:
                    skill = self._read(path, source)
                except SkillError as exc:
                    self.errors.append(str(exc))
                    continue
                except OSError as exc:
                    self.errors.append(f"skill {path} is unreadable: {exc}")
                    continue
                prior = found.get(skill.name)
                if prior is not None:
                    self.shadowed.append((skill.name, prior.path))
                    continue
                found[skill.name] = skill

        # usage history is attached last, so a load count can never be
        # overwritten by whatever the file happens to say about itself.
        if self.store is not None:
            known = {r["name"]: r for r in self.store.skill_rows()}
            for name, skill in found.items():
                row = known.get(name)
                if row is not None:
                    skill.loads = int(row["loads"])
                    skill.last_loaded = row["last_loaded"]
                self.store.upsert_skill(
                    skill.name, skill.path, skill.source, skill.description,
                    skill.when_to_use, skill.tags)
            # A row is dropped only on evidence that the skill is gone: this
            # scan walked the directory it lives in, and the file is no longer
            # there. Absence is not evidence. `self.dirs` is per-instance
            # configuration, so a scan can legitimately see fewer skills than
            # the store knows — an `ENV`-overridden directory, a different
            # `home`, a cwd with no `skills/`, or a test that scans a temp
            # directory. The first version read every one of those as "the
            # human deleted the rest of the library" and cleared the table,
            # taking the `loads` counts with it: the only input the router has
            # for telling a proven procedure from an unused one, gone, with no
            # way to reconstruct it. Keeping a stale row costs a ranking nudge
            # that the next real scan corrects; deleting one is unrecoverable.
            walked = [os.path.abspath(d) for _, d in self.dirs if os.path.isdir(d)]
            for stale in set(known) - set(found):
                path = os.path.abspath(known[stale]["path"])
                if os.path.exists(path):
                    continue      # still on disk: unreadable, see self.errors
                if not any(path.startswith(d + os.sep) for d in walked):
                    continue      # this scan never looked where it lives
                self.store.forget_skill_row(stale)
        self._skills = found
        return self.all()

    def _read(self, path: Path, source: str) -> Skill:
        text = path.read_text(encoding="utf-8")
        meta, body = split_frontmatter(text, origin=str(path))
        name = (meta.get("name") or path.stem).strip()
        if not valid_name(name):
            raise SkillError(
                f"skill {path} has an unusable name {name!r}: use lower-case "
                "letters, digits, hyphens or underscores")
        return Skill(
            name=name, description=meta["description"],
            when_to_use=meta.get("when_to_use", ""), body=body, path=str(path),
            source=source, tags=parse_tags(meta.get("tags", "")),
        )

    def all(self) -> list[Skill]:
        """Every skill, most-used first — the order the router blends on."""
        return sorted(self._skills.values(),
                      key=lambda s: (-s.loads, s.name))

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def load(self, name: str) -> Skill | None:
        """Read a skill whole, and count that as the retrieval event it is.

        The counter is bumped here and not by `scan`, so "used" means the agent
        actually opened it. A menu that scores a skill it never reads would be
        measuring the library, not the agent.
        """
        skill = self._skills.get(name)
        if skill is None:
            return None
        if self.store is not None:
            n = self.store.skill_load(name)
            if n:
                skill.loads = n
        skill.last_loaded = time.time()
        return skill

    # -- writing ----------------------------------------------------------
    def _target_dir(self, name: str, scope: str) -> str:
        """Where a write for `name` goes: in place if it exists, else `scope`.

        Editing in place is deliberate. Writing a same-named skill into a
        different directory would shadow the file the agent just read, and the
        symptom — "I edited it and nothing changed" — points at everything
        except the cause.
        """
        existing = self._skills.get(name)
        if existing is not None:
            return os.path.dirname(existing.path)
        for source, directory in reversed(self.dirs):
            if source == scope:
                return directory
        return self.dirs[-1][1]

    def write(self, name: str, description: str, when_to_use: str = "",
              body: str = "", tags: list[str] | None = None,
              scope: str = "user") -> Skill:
        """Create or update a skill file, and index it straight away."""
        if not valid_name(name):
            raise SkillError(
                f"unusable skill name {name!r}: lower-case letters, digits, "
                "hyphens or underscores, starting with a letter or digit")
        if not (description or "").strip():
            raise SkillError("a skill needs a description or it cannot be routed")
        target_dir = self._target_dir(name, scope)
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, name + SKILL_SUFFIX)
        existing = self._skills.get(name)
        text = render_skill_text(name, description.strip(), when_to_use.strip(),
                                 list(tags or []), body)
        # write to a sibling then move, so a crash mid-write cannot leave a
        # half-file where a skill used to be.
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
        skill = self._read(Path(path), existing.source if existing else scope)
        if existing is not None:
            skill.loads = existing.loads
            skill.last_loaded = existing.last_loaded
        self._skills[name] = skill
        if self.store is not None:
            self.store.upsert_skill(skill.name, skill.path, skill.source,
                                    skill.description, skill.when_to_use,
                                    skill.tags)
        return skill

    def archive(self, name: str) -> str:
        """Move a skill out of the library without destroying it.

        Archive rather than delete: a skill is a written-down procedure whose
        value is precisely that it outlives the moment it was written. Its row
        goes, because a retired skill that still ranks is worse than one that
        never existed. A clashing archive name gets a timestamp, never an
        overwrite.
        """
        skill = self._skills.get(name)
        if skill is None:
            raise SkillError(f"no skill named {name!r}")
        src = Path(skill.path)
        dest_dir = src.parent / ARCHIVE_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists():
            dest = dest_dir / f"{src.stem}-{int(time.time())}{src.suffix}"
        os.replace(src, dest)
        self._skills.pop(name, None)
        if self.store is not None:
            self.store.forget_skill_row(name)
        return str(dest)

    # -- prompt surface ---------------------------------------------------
    def menu(self, budget: int = MENU_BUDGET_CHARS) -> list[str]:
        """The index lines that ride in every prompt. Bodies never do.

        Bounded, and it says when it truncated: an agent that cannot see its
        own menu was cut will conclude it has no procedure for a task it does
        have one for, which is the failure this whole module exists to fix.

        Three things the first version got wrong, all of them costing prompt
        space every turn:

        *The header was not counted.* `budget` was spent on entries while the
        header line rode free, so the menu was always over its own cap.

        *`if shown and ...` guaranteed at least one entry.* With many short
        lines the budget never bound at all — nineteen skills rendered in full
        under a 1400-character cap. The floor is now a floor on *reporting*,
        not on listing: if nothing fits, the menu says so instead of silently
        blowing the budget.

        *Never-loaded skills were listed at full width.* They are the majority
        of any library that has been written to more than it has been read
        from, and they are the weakest bets in it, so they get the shorter
        lines and a cap: past MENU_COLD_MAX_LISTED the remainder fold into the
        count line. Nothing is lost -- the files are on disk, `skill_list` reads
        them all, and `skill_view` still opens any of them by name -- and what
        is dropped is the per-turn cost of advertising procedures that have
        never once been reached for. That cap then over-reached: written as
        `if len(cold) <= MENU_COLD_MAX_LISTED`, it hid *every* cold skill in
        any library larger than the cap, so forty never-loaded skills rendered
        as no names at all. A cap on how much prompt the unread part of a
        library earns is not a licence to make it invisible -- the count line
        has to be there either way.
        """
        skills = self.all()
        if not skills:
            return ["- Skills: none yet (skill_write(name, description, body) "
                    "saves one)."]
        header = "- Skills (procedures; skill_view(name) reads one in full):"
        lines = [header]
        used = len(header)

        warm = [s for s in skills if s.loads]
        cold = [s for s in skills if not s.loads]

        shown = 0
        for skill in warm:
            line = skill.menu_line()
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line)
            shown += 1

        # Cold skills: the weakest bets, so they get the shortest lines, and
        # only the first MENU_COLD_MAX_LISTED of them are advertised at all.
        # Past that the remainder fold into the count line below.
        #
        # The cap used to be all-or-nothing -- `if len(cold) <= MAX: list all`
        # -- so a library of 40 never-loaded skills rendered *zero* names and
        # one count. An agent then looked at its own menu, saw nothing, and
        # concluded it had no procedure for a task it had written one for. That
        # is the precise failure this module exists to prevent, reproduced by
        # the code that prevents it. The cap belongs on how much prompt the
        # unread part of a library earns, not on whether it is visible at all;
        # the count line is what keeps a short menu from reading as a short
        # library. Measured 2026-09-15 with 40 cold skills and a 500-char
        # budget: 3 listed, "+37 more -- skill_list reads them all".
        cold_shown = 0
        for skill in cold[:MENU_COLD_MAX_LISTED]:
            line = skill.menu_line(cold=True)
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line)
            cold_shown += 1

        rest = (len(warm) - shown) + (len(cold) - cold_shown)
        if rest:
            lines.append(f"    (+{rest} more -- skill_list reads them all)")
        return lines

    def report(self) -> dict[str, Any]:
        """Facts for the agent's self-model, measured rather than asserted."""
        skills = self.all()
        return {
            "skills": len(skills),
            "loads_total": sum(s.loads for s in skills),
            "never_loaded": [s.name for s in skills if not s.loads],
            "most_used": [s.name for s in skills[:3] if s.loads],
            "dirs": [d for _, d in self.dirs_present()],
            "unreadable": list(self.errors),
            "shadowed": [name for name, _ in self.shadowed],
            "backend": "markdown on disk; usage in the store",
        }


__all__ = [
    "Skill", "SkillLibrary", "SkillError", "default_skill_dirs",
    "port_tars_skills", "TARS_SKILL_TAG",
    "render_skill_text", "split_frontmatter", "valid_name",
    "parse_tags", "SKILL_SUFFIX", "ARCHIVE_DIR", "MENU_BUDGET_CHARS",
    "port_openclaw_skills", "openclaw_skill_roots", "iter_openclaw_skills",
    "OPENCLAW_SKILL_ENV", "openclaw_runs_here",
]

# ---------------------------------------------------------------------------
# borrowed procedures: the tars port
# ---------------------------------------------------------------------------
#: Every skill that came in from outside carries this word in its tags, so
#: "which of these did I work out myself?" stays answerable after the fact. A
#: fact that arrived from a stranger is not the same kind of thing as one that
#: was learned here, and the only moment that distinction is cheap to record is
#: the moment of the copy -- a year later nobody can reconstruct it.
TARS_SKILL_TAG = "ported"
TARS_ORIGIN = "tars (intelligence-indeed), Apache-2.0, vendored"


#: Where an installed openclaw keeps its skills, in the order worth checking.
#: The environment variable wins, for the same reason AUTOFORGE_SKILLS_DIRS
#: exists: an installation in an unusual place should not require a code change
#: to be found. The rest are the packaging layouts npm actually uses.
OPENCLAW_SKILL_ENV = "OPENCLAW_SKILLS_DIR"
OPENCLAW_SKILL_SUBDIRS = (
    ("core", os.path.join("openclaw", "skills")),
    ("extensions", os.path.join("openclaw", "extensions")),
)

#: Skills whose whole body is a different operating system's tooling. Copied
#: only when `include_other_os` is asked for: a procedure for driving macOS
#: `osascript` is not made useful by being written down on Windows, and a menu
#: that is a third unusable entries is a menu that stops being read. The
#: detection is deliberately textual and dull -- the point is to *say* something
#: was skipped and why, not to be clever about it.
#: Narrow on purpose, and narrowed after measuring: the first draft matched
#: "macos" anywhere in the body and therefore threw away `xurl` (npm-installable),
#: `prose` (cross-platform), `obsidian` and `healthcheck` -- four usable
#: procedures discarded for mentioning the word. A wrongly skipped skill is a
#: loss nobody notices; a wrongly kept one costs a menu line. So the markers are
#: things that CANNOT run here: a declaration of darwin in the metadata, and the
#: handful of binaries that exist only on macOS.
OPENCLAW_OTHER_OS_MARKERS = (
    "osascript", "remindctl", "peekaboo", "application support/imessage",
    "aerospace", "apple silicon",
)

#: The authoritative signal, and the reason the marker list above is so short:
#: openclaw's skills DECLARE their platforms, e.g. `"os": ["darwin", "linux"]`
#: for `tmux` or `["darwin", "linux", "win32"]` for `sherpa-onnx-tts`. Reading
#: the declaration beats guessing from prose -- the prose said "macos" in
#: `xurl`, `prose`, `obsidian` and `healthcheck`, which all run here, and said
#: nothing about Windows at all in the three that do.
OPENCLAW_OS_RE = re.compile(r'"os"\s*:\s*\[(.*?)\]', re.S)


def openclaw_runs_here(text: str, *, here: str = "win32") -> tuple[bool, str]:
    """Whether a foreign SKILL.md declares a platform this machine has.

    Returns (usable, why-not). Absence of a declaration is treated as usable --
    most of the 52 declare nothing, and refusing them for silence would throw
    away the library to avoid four wrong menu lines.
    """
    m = OPENCLAW_OS_RE.search(text)
    if m:
        declared = {x.strip().strip('"\'').lower() for x in m.group(1).split(",")}
        if declared and here not in declared:
            return False, f"declares os={sorted(declared)}, not {here}"
    low = text.lower()
    marker = next((x for x in OPENCLAW_OTHER_OS_MARKERS if x in low), "")
    if marker:
        return False, f"another OS's tooling (matched {marker!r})"
    return True, ""


def _lenient_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """`name`/`description`/`when_to_use` out of a foreign SKILL.md, or empty.

    Deliberately NOT ``split_frontmatter``. That parser is this library's own
    contract: it raises on a line that is not ``key: value``, which is right for
    a file this agent wrote and wrong for one that arrived from openclaw --
    every one of openclaw's skills carries a nested ``metadata`` block, and a
    strict reader would have rejected all 52 of them at the door. The rule being
    followed is: be strict about what I write, be tolerant about what I read.

    Four shapes actually occur in the 52: plain `key: value`; a block scalar
    (`description: |`) whose lines are indented under it; no frontmatter at all
    (one skill); and nested YAML braces, which are simply skipped because they
    are not any of the three keys wanted.
    """
    if not text.lstrip().startswith("---"):
        return {}, text
    start = text.index("---") + 3
    end = text.find("\n---", start)
    if end == -1:
        return {}, text
    head, body = text[start:end], text[end + 4:]
    meta: dict[str, str] = {}
    lines = head.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, value = m.group(1).lower(), m.group(2).strip()
        if value in ("|", ">", "|-", ">-", "|+", ">+"):
            parts: list[str] = []
            j = i + 1
            while j < len(lines) and (not lines[j].strip()
                                      or lines[j].startswith((" ", "\t"))):
                parts.append(lines[j].strip())
                j += 1
            value = " ".join(x for x in parts if x)
            i = j
        else:
            i += 1
        if key in ("name", "description", "when_to_use") and value:
            meta.setdefault(key, value.strip().strip('"').strip("'"))
    return meta, body.lstrip("\n")


def _openclaw_skill_roots_doc() -> str:      # pragma: no cover - documentation
    """Kept next to the parser so the two cannot drift apart in a reader's mind."""
    return "see openclaw_skill_roots()"


def openclaw_skill_roots(explicit: str | None = None) -> list[str]:
    """Every place an installed openclaw may keep its skills, existing ones only."""
    roots: list[str] = []
    env = explicit or os.environ.get(OPENCLAW_SKILL_ENV)
    if env and os.path.isdir(env):
        roots.append(env)
    bases = [
        os.path.join(os.environ.get("APPDATA", ""), "npm", "node_modules"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "npm", "node_modules"),
        os.path.join(os.path.expanduser("~"), ".npm-global", "lib", "node_modules"),
        os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "npm",
                     "node_modules"),
        "/usr/local/lib/node_modules",
        "/usr/lib/node_modules",
        os.path.join(os.path.expanduser("~"), ".local", "lib", "node_modules"),
    ]
    for base in bases:
        if not base or not os.path.isdir(base):
            continue
        pkg = os.path.join(base, "openclaw")
        if not os.path.isdir(pkg):
            continue
        core = os.path.join(pkg, "skills")
        if os.path.isdir(core):
            roots.append(core)
        ext = os.path.join(pkg, "extensions")
        if os.path.isdir(ext):
            for name in sorted(os.listdir(ext)):
                sub = os.path.join(ext, name, "skills")
                if os.path.isdir(sub):
                    roots.append(sub)
    # Dedupe, keeping order.
    seen: set[str] = set()
    out = []
    for r in roots:
        key = os.path.normcase(os.path.abspath(r))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def iter_openclaw_skills(roots: list[str]) -> list[tuple[str, str, Path]]:
    """(group, name, SKILL.md path) for every skill package under `roots`."""
    found: list[tuple[str, str, Path]] = []
    for root in roots:
        # A root that does not exist is skipped, not an error: the discovery
        # function returns only directories it found, but an explicit root can
        # be anything, and "there is no package here" is an answer the caller
        # reports -- not a traceback.
        if not os.path.isdir(root):
            continue
        group = ("extensions:" + os.path.basename(os.path.dirname(root))
                 if os.path.basename(root) == "skills"
                 and os.path.basename(os.path.dirname(root)) != "openclaw"
                 else "core")
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d):
                continue
            md = os.path.join(d, SKILL_INDEX_NAME)
            if os.path.isfile(md):
                found.append((group, name, Path(md)))
                continue
            # One level deeper: some packages nest their entry point.
            for sub, _, files in os.walk(d):
                if SKILL_INDEX_NAME in files:
                    found.append((group, name, Path(sub) / SKILL_INDEX_NAME))
                    break
    return found


def port_openclaw_skills(home: str | None = None, *, overwrite_body: bool = True,
                         include_other_os: bool = False,
                         roots: list[str] | None = None,
                         library: "SkillLibrary | None" = None,
                         store: Any = None) -> dict[str, Any]:
    """Copy an installed openclaw's skills into this agent's own library.

    Why: openclaw ships 52 hand-written procedures plus extension packages, each
    one a `SKILL.md` with a `name`/`description`/`when_to_use` header -- the same
    shape this library already reads. They are somebody's accumulated knowledge
    about driving real tools (gh, tmux, whisper, Notion, Obsidian), and reading
    them costs nothing while re-deriving them costs a session each.

    Read this before treating the count as a win: copying the ACTIONS of another
    agent is not the same as having its reach. openclaw's skills drive CLIs that
    are installed on this machine -- `gh`, `tmux`, `ffmpeg` -- and a copied
    procedure is only useful where that binary exists. What is genuinely
    transferable is the *method*: which command sequence, in which order, with
    which pitfall. That is what is being kept, and it is worth keeping on those
    terms alone, not as a claim that this agent can now do what openclaw does.

    Three decisions, each with a reason, and all three inherited from the tars
    port because the same arguments apply:

    *Flat under an ``openclaw-`` prefix.* The prefix is how a reader of the menu
      can tell a procedure learned here from one that arrived from a stranger.

    *Body copied verbatim, behind one provenance line.* Editing it would be
      "improving" a procedure whose value is that somebody else ran it -- and
      this agent has not run it once. The line carries the source and the
      licence (MIT, Peter Steinberger) into the file itself, so a copy separated
      from the package still says where it came from.

    *Other-OS skills are skipped and reported, not copied.* Ten of the 52 are
      macOS-only. Copying them would be the appearance of borrowing without the
      possibility of use, and it would bury the usable entries. Pass
      `include_other_os=True` to keep them anyway.

    Idempotent, and it does not reset usage: re-running rewrites the text and
    leaves ``loads`` alone, so a re-port cannot look like a fresh, unproven
    procedure.
    """
    lib = library
    if lib is None:
        lib = SkillLibrary(store, dirs=[("user", os.path.join(
            home or _default_home(), "skills"))])
    lib.scan()

    resolved = list(roots) if roots else openclaw_skill_roots()
    if not resolved:
        return {"ok": False, "written": [], "skipped": [],
                "error": "no openclaw skills directory found; set "
                         + OPENCLAW_SKILL_ENV + " to the package's skills/ path",
                "roots": []}

    written: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for group, name, path in iter_openclaw_skills(resolved):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            skipped.append({"name": name, "why": f"unreadable: {exc}"})
            continue
        meta, body = _lenient_frontmatter(text)
        description = (meta.get("description") or "").strip()
        if not description:
            # One of the 52 has no frontmatter at all. Its first markdown
            # heading is the honest description; a skill with no description is
            # unroutable, and unroutable means never offered.
            first = next((ln.lstrip("# ").strip() for ln in body.splitlines()
                          if ln.strip().startswith("#")), "")
            description = first or ("Copied from openclaw: " + name)
        when = (meta.get("when_to_use") or description).strip()
        if not include_other_os:
            usable, why = openclaw_runs_here(text)
            if not usable:
                skipped.append({"name": name, "why": why + "; pass "
                                "include_other_os=True to keep"})
                continue
        target = "openclaw-" + name
        if not valid_name(target):
            skipped.append({"name": name, "why": f"unusable name {target!r}"})
            continue
        if not overwrite_body and lib.get(target) is not None:
            skipped.append({"name": name, "why": "already present"})
            continue
        header = (f"<!-- Copied verbatim from openclaw {group} skill "
                  f"{name!r} (MIT, Peter Steinberger / openclaw). Body unchanged. "
                  f"Re-port with skills.port_openclaw_skills(). -->")
        lib.write(name=target, description=description, when_to_use=when,
                  body=header + "\n\n" + body,
                  tags=["openclaw", TARS_SKILL_TAG, group], scope="user")
        written.append({"name": target, "group": group,
                        "description": description[:100]})

    return {"ok": True, "written": written, "skipped": skipped,
            "roots": resolved, "count": len(written),
            "skipped_count": len(skipped),
            "dir": os.path.join(home or _default_home(), "skills")}


def port_tars_skills(home: str | None = None, *, overwrite_body: bool = True,
                     library: "SkillLibrary | None" = None,
                     store: Any = None) -> dict[str, Any]:
    """Copy the vendored tars domain skills into this agent's own library.

    Why this exists at all: the borrowed part of Intelligence Indeed that is
    actually worth having is not its harness (an AWS VM, an OSWorld grader, an
    Anthropic key) -- it is nine hand-written procedures about driving desktop
    applications, which took somebody real effort to learn and which cost
    nothing to keep. They are in the tree under ``vendor/tars`` as a verbatim
    copy; this function is what makes them *mine* rather than a checkout
    sitting next to me.

    Three decisions, each with a reason:

    *Flat, in the user's own skills directory, under a ``tars-`` prefix.* The
      prefix is not cosmetic: it is how a reader of the menu can tell a
      procedure that arrived from a stranger from one this agent learned here.
      Flat rather than a subdirectory because ``SkillLibrary.scan`` finds flat
      files and ``<category>/<name>/SKILL.md`` packages, and inventing a third
      layout to hold borrowed files would be a scanner bug waiting to happen.
    *The body is copied unchanged, behind one provenance line.* The prefix line
      is the licence and the source, in the file itself, so a copy that got
      separated from ``ORIGIN.txt`` still says where it came from. Editing the
      body would be "improving" a procedure whose whole value is that somebody
      else ran it -- and this agent has not run it once.
    *Idempotent, and it does not reset usage.* Re-running rewrites the text and
      leaves ``loads`` alone (``SkillLibrary.write`` carries the counter over),
      so a re-port cannot be mistaken for a fresh, unproven procedure.

    Returns a small report so the caller can see what happened rather than
    trusting that it did.
    """
    from .tars_port import PortedSkillSystem, provenance

    lib = library
    if lib is None:
        # `store` is threaded through rather than left None on purpose: the
        # usage counters live in the database, and a write through a library
        # with no store indexes the file without the counts that make the menu
        # rank by behaviour. Nine skills silently reset to "never loaded" is
        # the kind of loss nothing reports.
        lib = SkillLibrary(store, dirs=[("user", os.path.join(
            home or _default_home(), "skills"))])
    lib.scan()
    prov = provenance()
    written: list[str] = []
    skipped: list[dict[str, str]] = []

    try:
        tars = PortedSkillSystem()
    except Exception as exc:                      # missing vendor tree, or broken
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "written": [], "skipped": [], "origin": prov["upstream"]}

    for meta in tars.registry.list_meta():
        name = "tars-" + meta.name
        body = tars.body(meta.name)
        if not overwrite_body and lib.get(name) is not None:
            skipped.append({"name": name, "why": "already present"})
            continue
        header = ("<!-- Copied verbatim from " + prov["upstream"] + " ("
                  + str(prov["licence"]) + "). Body unchanged. Re-port with "
                  "skills.port_tars_skills(). -->")
        lib.write(
            name=name,
            description=meta.description,
            when_to_use=meta.description,
            body=header + "\n\n" + body,
            tags=["tars", TARS_SKILL_TAG, "desktop-gui"],
            scope="user",
        )
        written.append(name)

    return {"ok": True, "written": written, "skipped": skipped,
            "origin": prov["upstream"], "licence": prov["licence"],
            "dir": os.path.join(home or _default_home(), "skills")}

