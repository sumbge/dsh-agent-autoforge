"""Intent declarations, so the limits are asked for rather than assumed.

The inversion this exists to fix: `Sandbox` sets caps -- 512 MB, 8 processes,
30 s CPU -- and then runs whatever lands in it. Those numbers bound blast
radius, which is worth having, but they are the *system's* guess about the
task rather than a statement about it. AgenticOS (arXiv 2606.21129) puts the
declaration first and synthesises the least-privilege environment from it; the
useful part of that for a host like this one is not the kernel, it is the
ordering.

So a run can now carry a manifest: what the tool is for, and the numbers it
claims to need. Three things follow, in increasing order of how much they are
worth:

  1. the declaration *sets* the sandbox fields, so a tool that needs 32 MB
     does not silently get 512;
  2. the measured run is reconciled against the declaration -- `query()`
     already returns peak bytes, CPU and processes launched, so "declared 64,
     used 71" is a fact and not a hope;
  3. a declaration above the operator's ceiling is refused *before* anything
     runs, which is the only place a limit can be refused cheaply.

What is deliberately NOT here: enforcing `network: False`. Windows has no
seccomp, and the WFP filter that would do it is not reachable from this
process. So the field is recorded, the run is measured, and the report says
`network: declared-none, not enforced` -- a declared-only field that lies
about being enforced is the single worst outcome available, worse than not
having the field.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

#: The operator's ceiling. A manifest asking past this is refused rather than
#: clamped: clamping silently gives a tool less than it declared and then
#: reports a failure that looks like the tool's fault.
CEILING = {
    "memory_mb": 2048,
    "max_processes": 32,
    "cpu_seconds": 120.0,
    "wall_s": 300.0,
}


class ManifestRefused(PermissionError):
    """The declaration is not admissible, so nothing was run."""


@dataclass
class CapabilityManifest:
    """What a run says it is for, and the resources it claims to need."""

    intent: str = ""
    memory_mb: int = 64
    max_processes: int = 2
    cpu_seconds: float = 10.0
    wall_s: float = 30.0
    network: bool = False
    #: Paths the run claims to write outside its own cwd. Recorded so the
    #: ledger can say which are habitual, not enforced -- see the module note.
    writes: tuple[str, ...] = ()
    #: True for the one thing that legitimately wants the whole machine:
    #: run_python, whose entire purpose is to be a general shell.
    general_purpose: bool = False

    def ceiling_breaches(self) -> list[str]:
        out = []
        for k, limit in CEILING.items():
            got = getattr(self, k)
            if got > limit:
                out.append(f"{k}={got} > ceiling {limit}")
        return out

    def admit(self) -> "CapabilityManifest":
        if self.general_purpose:
            return self
        breaches = self.ceiling_breaches()
        if breaches:
            raise ManifestRefused(
                "manifest refused before any code ran: " + "; ".join(breaches))
        if not self.intent.strip():
            raise ManifestRefused(
                "manifest has no intent line; a declaration without a stated "
                "purpose cannot be reconciled against anything afterwards")
        return self

    def sandbox_fields(self) -> dict[str, Any]:
        """The `Sandbox` fields this declaration sets."""
        return {
            "timeout": self.wall_s,
            "memory_mb": self.memory_mb,
            "max_processes": self.max_processes,
            "cpu_seconds": self.cpu_seconds,
        }

    def network_line(self) -> str:
        return ("declared-none, NOT enforced (no seccomp on this host); "
                "a run that opens a socket will not be stopped by this field"
                if not self.network else
                "declared-needed; outbound is the sandbox's normal reach")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["writes"] = list(self.writes)
        d["network_effect"] = self.network_line()
        return d


def reconcile(manifest: CapabilityManifest, result: Any,
              accounted: dict[str, Any] | None = None) -> dict[str, Any]:
    """Declared against measured. The verdict is a finding, not a gate.

    Nothing here kills a run for using more than it declared -- a tool that
    used 71 MB after declaring 64 is not misbehaving, it is imprecise, and
    killing it would punish the imprecision rather than the thing worth
    punishing. What this produces is the record that lets a *pattern* of
    under-declaration be seen later, which is the only form in which the
    numbers mean anything.
    """
    acc = accounted or {}
    used_mb = (acc.get("peak_job_bytes") or 0) / (1024 * 1024)
    used_cpu = (acc.get("cpu_ms") or 0) / 1000.0
    procs = acc.get("processes_launched") or 1
    findings = []
    if manifest.memory_mb and used_mb > manifest.memory_mb:
        findings.append(f"memory: declared {manifest.memory_mb} MB, peaked {used_mb:.0f} MB")
    if manifest.cpu_seconds and used_cpu > manifest.cpu_seconds:
        findings.append(f"cpu: declared {manifest.cpu_seconds}s, used {used_cpu:.1f}s")
    if manifest.max_processes and procs > manifest.max_processes:
        findings.append(f"processes: declared {manifest.max_processes}, launched {procs}")
    # Exhausting the declaration is the other half of the same signal, and it
    # reads differently: a run that used its whole 8 MB and died did not
    # misbehave, it asked for too little. Without this the report is silent in
    # exactly the case that costs the most time -- the tool failing for a
    # reason that looks like a bug in the tool.
    cap = manifest.memory_mb
    if cap and used_mb >= cap * 0.99:
        findings.append(
            f"memory: used its whole declared {cap} MB "
            f"({'failed' if not getattr(result, 'ok', True) else 'passed'}); "
            "the declaration was too small for this job")
    return {
        "intent": manifest.intent,
        "declared": manifest.to_dict(),
        "measured": {"memory_mb": round(used_mb, 1), "cpu_s": round(used_cpu, 2),
                     "processes": procs, "ok": getattr(result, "ok", None)},
        "under_declared": findings,
    }


def default_for(source: str) -> CapabilityManifest:
    """A manifest for a run that did not bring one.

    Named by where the run came from rather than defaulting all of them to one
    shape: a forged tool and a one-off `run_python` do not want the same
    environment, and pretending they do is how a general shell ends up capped
    by a tool's budget.
    """
    if source == "run_python":
        return CapabilityManifest(
            intent="operator-supplied snippet; general-purpose by definition",
            memory_mb=1024, max_processes=8, cpu_seconds=30.0, wall_s=30.0,
            general_purpose=True)
    return CapabilityManifest(
        intent=f"forged tool {source!r} from the forge verifier",
        memory_mb=128, max_processes=2, cpu_seconds=10.0, wall_s=10.0)


def intent_for(name: str | None = None, code: str = "", description: str = "") -> CapabilityManifest:
    """The declaration that fits this run, inferred from what it is.

    `run_python` is the general-purpose case: its whole purpose is to be an
    arbitrary shell, so it declares the machine. Everything else is a forged
    tool, and a forged tool is the shape the numbers are for -- it answers a
    question and returns, so 128 MB and 2 processes is generous. The heuristic
    below is deliberately crude and says so: it scans for the markers that
    mean a tool is *not* a pure function, and when it finds one it raises the
    process budget rather than guessing further. A tool that under-declares is
    caught afterwards by `reconcile` and reported, which is cheaper than a
    classifier that is wrong in the other direction.
    """
    if name == "run_python":
        return default_for("run_python")
    spawns = any(marker in (code or "") for marker in
                 ("subprocess", "os.system", "multiprocessing", "Popen", "threading"))
    return CapabilityManifest(
        intent=f"forged tool {name!r}: {description.strip()[:120]}".strip(),
        memory_mb=128,
        max_processes=4 if spawns else 2,
        cpu_seconds=15.0 if spawns else 10.0,
        wall_s=20.0 if spawns else 10.0,
    )


def apply_declaration(sandbox: "Any", manifest: CapabilityManifest):
    """Return a sandbox whose containment numbers come from the declaration.

    `dataclasses.replace` on the existing sandbox rather than a fresh one: the
    abort hook, the output cap and the environment allow-list are the agent's
    decisions and must survive the change. Only the four declared fields move.
    """
    from dataclasses import replace as _replace

    fields = manifest.sandbox_fields()
    try:
        return _replace(sandbox, **fields)
    except TypeError:
        # A test double that is not a dataclass: leave it alone rather than
        # failing a verification over the shape of a stub.
        return sandbox
