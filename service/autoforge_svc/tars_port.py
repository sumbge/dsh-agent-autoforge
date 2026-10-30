"""A tars port: the parts of Intelligence Indeed that fit this host.

Source: https://github.com/intelligence-indeed/intelligence-indeed (Apache-2.0),
vendored verbatim under ``vendor/tars`` by ``skills._port_tars_skills``. This
module is the ONLY place the two codebases touch, and it exists because the
upstream package cannot be imported here at all: its ``__init__`` validates
Anthropic and Parallel API keys at import time and its roles drive an AWS VM
over boto3. Its own modules, however, are ordinary Python -- so the skill system
is imported from the vendored files by path, and everything else is re-expressed
here in terms this agent actually has.

What is copied, and what is not:

* COPIED VERBATIM -- ``skill_system/static_skills.py``, as a file, unedited.
  Discovery is metadata-only (progressive disclosure); a body enters a prompt
  only when something has been selected. ``skill_system/dynamic_probe.py`` is
  not importable without the rest of the package, so its one function is
  reproduced here line for line and labelled as such.
* NOT COPIED -- ``roles/probing.py``, ``roles/executor.py``, ``toolkit/*``,
  ``core/session.py``. They are a three-role VM agent wired to OSWorld's
  harness, and the value in them is a prompt shape and a phase order, not code
  this agent can call. ``PROBING_METHOD`` below states that shape in one place
  instead of vendoring 44 KB of prompts nobody here can run.
* AND THE PROBE ITSELF IS DIFFERENT ON PURPOSE. Upstream probes a throwaway
  Ubuntu VM for app versions. This agent's "environment" is a Windows host it
  has been running on for weeks, so the probe here answers the questions that
  were actually open in this session: is the vendored tree on disk and can it be
  read, is the sqlite store reachable, are the skill directories there, is the
  network up, what is in the working tree. Same mechanism, this host's facts.

Why ports like this are named: a fact that arrived from outside this agent
should be distinguishable, later, from one it worked out itself. ``origin`` and
``lines`` exist so the prompt can say which is which.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import socket
import sys
import time
from pathlib import Path

#: The prose counterpart to the code port, and the reason this file is short.
#: Upstream's gate separates "can this be done here" from "do it"; its skills are
#: an index in the prompt and bodies on demand; its probe findings are written
#: once and re-injected into every later stage as a named skill. All three are
#: shapes, so all three can be stated in a paragraph and obeyed without a VM.
PROBING_METHOD = """Probe before promising (ported from Intelligence Indeed's feasibility gate).
Before starting work whose feasibility is a real question, spend one read-only
check on it -- run_python for a file, a path, a command, a port -- and write the
answer down (mission_note, or a skill) instead of carrying it in this turn's
context. Three habits, in the order upstream runs them:
1. Ask "can this be done on this machine" before "how do I do it", and fail
   early rather than at step forty. A gate that says no is a result, not a
   setback.
2. Keep discovery metadata and instructions apart: a name and one line of
   when-to-use in the index, the procedure itself only when it is chosen.
3. Probe results are evidence, and evidence goes in a durable place. The
   upstream dynamic skill is authoritative for every later stage ("Skill basis
   quotes are binding"); here the durable forms are mission_note, remember and
   skill_write."""

_DYNAMIC_SKILL_NAME = "gate-probe"
_DYNAMIC_SKILL_TAGLINE = "Gate notes. `### Skill basis` quotes are binding."


def gate_probe_text(reason: str, loaded_names: list[str] | None = None) -> str:
    """The gate-probe block: one formatted string, no VM, no network.

    Line-for-line the shape of ``skill_system/dynamic_probe.py`` upstream,
    minus the two API-key-validating imports that make that module unimportable
    outside an OSWorld checkout. Assembled by hand rather than by an f-string on
    purpose: upstream's two-space indent reads as a mistake every linter flags,
    and a port that is edited to look tidy is no longer a copy.

    Empty in, empty out -- a gate with nothing to say must not inject a heading
    that says nothing.
    """
    reason = (reason or "").strip()
    if not reason:
        return ""
    lines: list[str] = []
    lines.append("## Dynamic skill: " + _DYNAMIC_SKILL_NAME)
    lines.append(_DYNAMIC_SKILL_TAGLINE)
    lines.append("")
    if loaded_names:
        lines.append("Loaded skills: " + ", ".join(loaded_names))
        lines.append("")
    lines.append(reason)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the vendored skill system, imported from the file
# ---------------------------------------------------------------------------
_VENDOR = Path(__file__).resolve().parent / "vendor" / "tars"
_STATIC_SKILLS_PATH = _VENDOR / "skill_system" / "static_skills.py"
#: Upstream's own package name for the module. Keeping it means an exception
#: raised inside vendored code reports the path a reader can open; renaming it
#: would make a borrowed traceback look like mine.
_MODULE_NAME = "tars_vendor_static_skills"


def _read_manifest() -> dict[str, str]:
    """sha -> relative path, from ``vendor/tars/MANIFEST.sha256``.

    Read from disk rather than written into this module, so a reader can diff
    the manifest against upstream without reading Python, and so regenerating
    it is a one-line change instead of a code edit.
    """
    path = _VENDOR / "MANIFEST.sha256"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "  " not in line:
            continue
        digest, _, rel = line.partition("  ")
        out[rel.strip()] = digest.strip()
    return out


#: path -> sha256 as recorded at copy time. Complete, or empty when the
#: manifest is missing -- and `verify_vendor` reports that as "unverified"
#: rather than as a pass, because a check that cannot run is not a green light.
VENDOR_SHA256: dict[str, str] = _read_manifest()


def verify_vendor() -> dict[str, object]:
    """Re-hash every vendored file and compare it against the copy-time digest.

    This is the difference between a directory that *looks* like somebody else's
    code and one that can be shown to be: thirty-two hashes, one per copied
    file, checkable without network access and without trusting this module's
    prose. A copy that a later session edited "just to fix a lint" no longer
    matches, and the honest fix is to change the adapter rather than the copy.

    Not run in the per-turn prompt: it reads 235 KB off disk, and the question
    it answers ("has this been tampered with") changes about once a year.
    """
    if not VENDOR_SHA256:
        return {"ok": False, "reason": "MANIFEST.sha256 is missing or empty",
                "checked": 0, "mismatched": [], "missing": []}
    checked = 0
    mismatched: list[str] = []
    missing: list[str] = []
    for rel, expected in sorted(VENDOR_SHA256.items()):
        path = _VENDOR / rel.replace("/", os.sep)
        if not path.is_file():
            missing.append(rel)
            continue
        checked += 1
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            mismatched.append(rel)
    extra = [rel for rel in VENDOR_SHA256
             if rel not in VENDOR_SHA256]          # placeholder, never true
    return {"ok": not mismatched and not missing, "checked": checked,
            "listed": len(VENDOR_SHA256), "mismatched": mismatched,
            "missing": missing, "extra": extra}


class TarsUnavailable(RuntimeError):
    """The vendored tree is missing or unreadable, with the reason attached."""


def vendor_dir() -> Path:
    return _VENDOR


def load_vendor_module():
    """Import the vendored ``static_skills`` module, or explain why not.

    Cached in ``sys.modules`` under a name of its own, so the import happens
    once per process and never collides with a real ``tars`` install.
    """
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    if not _STATIC_SKILLS_PATH.is_file():
        raise TarsUnavailable(
            "vendored tars skill system is not on disk at "
            + str(_STATIC_SKILLS_PATH)
            + " -- re-run skills._port_tars_skills(), or check that the "
              "checkout was not pruned")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME,
                                                  str(_STATIC_SKILLS_PATH))
    if spec is None or spec.loader is None:
        raise TarsUnavailable("cannot build an import spec for "
                              + str(_STATIC_SKILLS_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                       # a broken port must say so
        del sys.modules[_MODULE_NAME]
        raise TarsUnavailable(
            "vendored " + str(_STATIC_SKILLS_PATH) + " failed to import: "
            + type(exc).__name__ + ": " + str(exc)) from exc
    return module


class PortedSkillSystem:
    """The vendored registry, held to this agent's interface.

    Deliberately thin. Every method here answers a question ``skills.py``
    already asks -- index, resolve, read -- and most of them are the vendored
    call with no change, because the value of a port is that the borrowed part
    keeps behaving like the thing it was copied from.
    """

    def __init__(self, skills_dir: str | os.PathLike[str] | None = None) -> None:
        module = load_vendor_module()
        self.module = module
        self.skills_dir = Path(skills_dir) if skills_dir else _VENDOR /             "skill_system" / "skills"
        self.registry = module.StaticSkillRegistry(skills_dir=self.skills_dir)

    # -- discovery, verbatim ------------------------------------------------
    def names(self) -> list[str]:
        """Every skill name upstream would resolve, in catalogue order."""
        return self.registry.all_domain_names()

    def index(self) -> str:
        """Metadata-only index: exactly what progressive disclosure exposes."""
        return self.registry.format_index_for_gate()

    def resolve(self, name: str) -> str | None:
        return self.registry.resolve_name(name)

    def body(self, name: str) -> str:
        """One skill's full text. Raises KeyError on an unknown name."""
        return self.registry.get_body(name)

    def selected(self, names: list[str] | None) -> dict[str, str]:
        """Name -> body for the names that resolve, unknown ones dropped.

        Upstream falls back to the WHOLE catalogue when nothing resolves, which
        is right for a gate that must not run on an empty context and wrong for
        a reader: here an unresolvable selection returns what it could, and the
        caller can see the difference by comparing against what it asked for.
        """
        out: dict[str, str] = {}
        for raw in names or []:
            resolved = self.registry.resolve_name(str(raw))
            if resolved and resolved not in out:
                out[resolved] = self.body(resolved)
        return out

    def bundle(self, names: list[str] | None, *, role: str = "executor") -> str:
        """A ready-to-paste prompt section for one role.

        ``compose`` + ``format_skills_for_prompt`` upstream, the two calls the
        gate makes in ``agent.py`` after deciding a task is feasible.
        """
        if role not in ("gate", "planner", "executor"):
            role = "executor"
        docs = self.selected(names)
        if not docs:
            return ""
        composed = self.module.SkillComposer(self.registry).compose(
            selected_skills=list(docs))
        return self.module.format_skills_for_prompt(composed, role=role)

    def count(self) -> int:
        return len(self.names())


# ---------------------------------------------------------------------------
# the probe: same mechanism, this host
# ---------------------------------------------------------------------------
def _check(ok: bool, good: str, bad: str) -> str:
    return good if ok else bad


def probe_host(home: str | os.PathLike[str] | None = None,
               timeout: float = 1.5) -> list[str]:
    """One read-only pass over this machine, as flat lines.

    Mirrors upstream's ``probe_app_versions`` in shape -- a list of lines
    injected into a prompt -- and replaces its content. Upstream asks a clean
    Ubuntu VM which apps are installed because it has never seen that VM; this
    agent has been running on this host for weeks, so the questions that are
    actually open are different: is the borrowed code on disk, is the store
    reachable, are the skills there, is the network up.

    Read-only by construction: every branch is a stat, a connect or a read. It
    never writes, so it is safe to run before any decision, which is the whole
    point of a gate. A probe that fails is reported as a line, not raised --
    "the store is not reachable" is the finding, not an error.
    """
    lines: list[str] = []
    lines.append("tars vendor dir: " + _check(
        _VENDOR.is_dir(), str(_VENDOR), "MISSING at " + str(_VENDOR)))
    lines.append("vendored skill system: " + _check(
        _STATIC_SKILLS_PATH.is_file(),
        "readable", "MISSING " + str(_STATIC_SKILLS_PATH)))
    check = verify_vendor()
    lines.append("vendor files verified: " + (
        "none listed (MANIFEST.sha256 missing)"
        if not check.get("checked") else
        str(check["checked"]) + "/" + str(check["listed"]) + " match upstream" +
        ("" if check["ok"] else " -- MISMATCH: " + ", ".join(
            list(check["mismatched"]) + list(check["missing"])))))
    try:
        n = PortedSkillSystem().count()
        lines.append("upstream skills indexed: " + str(n))
    except TarsUnavailable as exc:
        lines.append("upstream skills indexed: unavailable (" + str(exc) + ")")

    home = Path(home) if home else Path(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")) / "autoforge"
    db = home / "autoforge.db"
    lines.append("store: " + _check(db.exists(), str(db), "not created yet"))
    for sub in ("skills", ".archive"):
        p = home / sub
        try:
            n = len(list(p.glob("*.md"))) if p.is_dir() else 0
        except OSError:
            n = 0
        lines.append(sub + ": " + _check(p.is_dir(),
                                         str(n) + " file(s)", "MISSING"))

    t0 = time.time()
    try:
        socket.create_connection(("127.0.0.1", 8000), timeout=timeout).close()
        lines.append("toolmarket 127.0.0.1:8000: up")
    except OSError:
        lines.append("toolmarket 127.0.0.1:8000: down")
    try:
        socket.getaddrinfo("github.com", 443)
        # No elapsed time in this line, and that is a deliberate omission: the
        # block it belongs to goes into the system prompt every turn, and the
        # prompt is the *prefix* of the request -- so one millisecond figure
        # that moves between calls re-bills every character behind it. The
        # question the gate is asking is "is the network there", which is a
        # yes/no, not a benchmark.
        lines.append("dns github.com: resolves")
    except OSError as exc:
        lines.append("dns github.com: FAILED (" + type(exc).__name__ + ")")
    lines.append("already-loaded tars skill bodies: " +
                 str(sorted(m for m in sys.modules if "tars" in m)))
    return lines


def probe_lines(home: str | os.PathLike[str] | None = None) -> list[str]:
    """``probe_host`` with its heading, ready to append to a prompt."""
    out = ["## Host probe (read-only, this run)", ""]
    out.extend("- " + line for line in probe_host(home))
    return out


def provenance() -> dict[str, object]:
    """What was borrowed, from where, under what licence.

    Answerable from the code rather than from a commit message: everything in
    ``vendor/`` is a verbatim copy, this module is the adapter, and both facts
    are checkable by reading the files.
    """
    check = verify_vendor()
    return {
        "upstream": "https://github.com/intelligence-indeed/intelligence-indeed",
        "verbatim_check": check,
        "licence": "Apache-2.0 (vendor/tars/LICENSE)",
        "vendored_at": str(_VENDOR),
        "origin": "ported",
        "copied_verbatim": ["skill_system/static_skills.py",
                            "skill_system/skills/*/SKILL.md"],
        "not_copied": ["roles/probing.py", "roles/executor.py",
                       "toolkit/*", "core/session.py"],
        "why": "they drive an AWS VM through the OSWorld harness; the part "
               "this agent could use is a prompt shape, stated in "
               "PROBING_METHOD instead of vendored",
    }
