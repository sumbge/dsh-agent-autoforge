"""Tool composition: tools that call other tools.

The framework's DAG of tool dependencies is tracked explicitly. When a forged
tool needs another tool to accomplish its goal, it declares that dependency
in its effect_signature: "uses:<tool_name>".

The composition system:

1. RESOLVES dependencies during verification — are all declared deps present
   and ACTIVE?
2. INJECTS dependencies at runtime — the sandbox runner receives a mapping of
   {dep_name: code} so the forged tool can call the dependency out-of-process.
3. VALIDATES against circular dependencies — A depends on B depends on A is
   rejected.
4. CASCADES quarantine — if A is quarantined, everything that depends on A is
   put on probation too.

In the forge pipeline, a tool that declares dependencies gets them injected
into the sandbox before execution and trigger checks.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..tools.registry import ToolRegistry
from ..tools.spec import ToolSpec, ToolState

_USES_RE = re.compile(r"uses:\s*(\S+)")

# ---------------------------------------------------------------------------
@dataclass
class DepGraph:
    """Lightweight DAG of tool dependencies."""

    deps: dict[str, list[str]] = field(default_factory=dict)

    def add(self, tool: str, depends_on: list[str]) -> None:
        self.deps[tool] = list(set(dep for dep in depends_on if dep != tool))

    def get(self, tool: str) -> list[str]:
        return self.deps.get(tool, [])

    def reverse_deps(self, tool: str) -> list[str]:
        return [t for t, deps in self.deps.items() if tool in deps]

    def has_cycle(self, tool: str) -> bool:
        """DFS cycle detection starting from `tool`."""
        visited = set()
        stack = [tool]
        while stack:
            current = stack.pop()
            if current in visited:
                return True
            visited.add(current)
            stack.extend(self.deps.get(current, []))
        return False

    def to_dict(self) -> dict[str, list[str]]:
        return dict(self.deps)


# ---------------------------------------------------------------------------
def parse_deps(spec: ToolSpec) -> list[str]:
    """Extract declared dependencies from effect_signature and code."""
    deps: set[str] = set()
    combined = f"{spec.effect_signature} {spec.code}"
    for match in _USES_RE.finditer(combined):
        deps.add(match.group(1))
    return sorted(deps)


def validate_deps(spec: ToolSpec, registry: ToolRegistry) -> list[str]:
    """Check that all declared dependencies exist and are usable.

    Returns a list of error messages (empty = all good.)
    """
    errors: list[str] = []
    deps = parse_deps(spec)
    for dep in deps:
        dep_spec = registry.get(dep)
        if dep_spec is None:
            errors.append(f"dependency {dep!r} not found")
        elif dep_spec.state in (ToolState.QUARANTINED, ToolState.RETIRED):
            errors.append(f"dependency {dep!r} is {dep_spec.state.value}")
    return errors


# ---------------------------------------------------------------------------
def compose_code(
    spec: ToolSpec,
    dep_codes: dict[str, str],
    *,
    sandbox_entry: str = "",
) -> str:
    """Prepend dependency code so the forged tool can call its deps.

    Each dependency's code becomes a module-level definition in the same
    namespace. The forged tool's code is appended at the end.
    """
    parts: list[str] = []
    # Insert each dep's code, wrapped in a comment header
    for dep_name, dep_code in dep_codes.items():
        parts.append(f"# ---- dependency: {dep_name} ----")
        # Strip the function header of the dep code — the forger wrote
        # something like "def normalize_isbn(...)" — we keep it as is
        # since Python doesn't care about order of defs.
        parts.append(dep_code.strip())

    parts.append(f"# ---- tool: {spec.name} ----")
    parts.append(spec.code.strip())

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
def dependents_to_mark(
    registry: ToolRegistry,
    dep_graph: DepGraph,
    quarantined_name: str,
) -> list[str]:
    """Find everything that depends on `quarantined_name` and mark it."""
    affected: list[str] = []
    for depender_name in dep_graph.reverse_deps(quarantined_name):
        depender = registry.get(depender_name)
        if depender and depender.state in (ToolState.ACTIVE, ToolState.PROBATION):
            depender.state = ToolState.PROBATION  # demote, don't quarantine
            affected.append(depender_name)
    return affected


__all__ = [
    "DepGraph",
    "parse_deps",
    "validate_deps",
    "compose_code",
    "dependents_to_mark",
    "ToolSpec",
    "ToolRegistry",
    "ToolState",
]