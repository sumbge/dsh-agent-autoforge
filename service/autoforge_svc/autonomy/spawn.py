"""Spawning: the agent creates other agents, freely.

No pre-declared agent pool, no fixed orchestration graph. An agent decides at
runtime that a subtask wants its own worker, and spawns one. The child can
share the parent's tool registry (so tools forged by one are usable by all) or
get an isolated one (so experiments don't pollute the parent's library).

Two sharing modes:

  SHARED  — same registry object. A tool the child forges is immediately
            available to the parent. Fast convergence, but a bad tool
            propagates.
  ISOLATED — child gets a copy. It can forge freely; only tools that pass its
            verification AND the parent approves are merged back.

The parent chooses per-spawn. Default is SHARED, because the whole thesis is
that a growing shared library is the point — isolation is for when you want to
A/B a child's tool quality before trusting it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..tools.registry import ToolRegistry
from ..tools.spec import ToolSpec, ToolState
from .roles import ceiling, role_brief


class ShareMode(str, Enum):
    SHARED = "shared"
    ISOLATED = "isolated"


@dataclass
class SpawnRecord:
    child_id: str
    task: str
    mode: str
    parent_id: str
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    result: str = ""
    tools_forged: list[str] = field(default_factory=list)
    error: str | None = None
    # Which role the child ran as, and — if it tried to reach past that role —
    # what it was refused. A refusal is evidence the whitelist is live rather
    # than decorative, so it is recorded rather than swallowed.
    role: str = ""
    refusals: list[str] = field(default_factory=list)
    #: The directive the child was handed, kept so the trace shows the role
    #: reached a model instead of only reaching a dataclass.
    brief: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_id": self.child_id,
            "task": self.task[:200],
            "mode": self.mode,
            "role": self.role or None,
            "brief_chars": len(self.brief),
            "tools_forged": self.tools_forged,
            "refusals": self.refusals,
            "duration_s": round(self.finished - self.started, 2) if self.finished else None,
            "error": self.error,
        }


@dataclass
class Spawner:
    """Creates child agents on demand.

    `agent_factory(parent, registry) -> agent-like` must return an object with
    a `.run(task) -> result` method whose result exposes `.content` and the
    child's registry (if it forged anything).
    """

    registry: ToolRegistry
    agent_factory: Callable[[Any, ToolRegistry], Any]
    parent_id: str = "root"
    spawn_count: int = 0
    history: list[SpawnRecord] = field(default_factory=list)
    max_depth: int = 8              # a limit the agent can raise, not a cage
    policy: Any = None              # AutonomyPolicy | None — gates merge_back
    _depth: int = 0

    def spawn(
        self,
        task: str,
        *,
        mode: ShareMode = ShareMode.SHARED,
        child_registry: ToolRegistry | None = None,
        restrict_to: set[str] | list[str] | None = None,
        role: str = "",
        brief: str = "",
        deny: Callable[[Any], str] | None = None,
    ) -> SpawnRecord:
        self.spawn_count += 1
        child_id = f"{self.parent_id}.{self.spawn_count}"

        # A role is enforceable on its own. The caller may add a whitelist, a
        # richer brief (a node's hint on top of the role's), or a whole different
        # ceiling — but naming a role is enough to get the role's behaviour, so
        # there is no way to spawn a "critic" that is not held to a critic's
        # limits by simply forgetting to pass one.
        if role:
            brief = brief or role_brief(role)
            if deny is None:
                deny = ceiling(role)

        record = SpawnRecord(
            child_id=child_id, task=task, mode=mode.value, parent_id=self.parent_id,
            role=role, brief=brief,
        )

        if self._depth >= self.max_depth:
            record.error = f"max spawn depth {self.max_depth} reached (raise it if needed)"
            record.finished = time.time()
            self.history.append(record)
            return record

        base_reg = self.registry if mode == ShareMode.SHARED else (
            child_registry or ToolRegistry())
        # A role's whitelist becomes a lens over the real registry, so the child
        # can only see and call what its role allows. Registration still lands on
        # base_reg — which is why the forged-tool diff is taken there: a child
        # that forges something outside its whitelist really did create it, and
        # hiding that from the parent would be the wrong kind of tidiness.
        #
        # `deny` is the role's capability ceiling, which is not a whitelist: a
        # critic's tool set depends on what each tool does, and forged tools are
        # named at runtime, so the ceiling has to be judged per-spec.
        reg = base_reg.scoped(restrict_to or (), role=role, deny=deny)
        before = set(base_reg.names())

        try:
            child = self.agent_factory(self, reg)
            # Handed over after construction rather than through the factory
            # signature, so an existing two-argument `agent_factory` — including
            # the fakes in the tests — keeps working. A child that does not
            # implement the hook is simply one that cannot be briefed, and the
            # record says so by carrying an empty brief.
            if brief:
                hook = getattr(child, "apply_role", None)
                if callable(hook):
                    hook(brief, role)
            result = child.run(task)
            record.result = getattr(result, "content", str(result))
        except Exception as exc:  # noqa: BLE001
            record.error = f"{type(exc).__name__}: {exc}"

        after = set(base_reg.names())
        record.tools_forged = sorted(after - before)
        record.refusals = list(getattr(reg, "refusals", []) or [])
        record.finished = time.time()
        self.history.append(record)
        return record

    def child_spawner(self, child_registry: ToolRegistry) -> "Spawner":
        """A spawner for a child, one level deeper."""
        return Spawner(
            registry=child_registry,
            agent_factory=self.agent_factory,
            parent_id=f"{self.parent_id}.{self.spawn_count}",
            max_depth=self.max_depth,
            policy=self.policy,
            _depth=self._depth + 1,
        )

    # -- merging isolated children back --------------------------------
    def merge_back(
        self,
        child_registry: ToolRegistry,
        *,
        require_state: ToolState = ToolState.ACTIVE,
        approve: Callable[[ToolSpec], bool] | None = None,
    ) -> list[str]:
        """Pull an isolated child's trusted tools into the parent library.

        `approve=None` (the default) is not "no gate": it means the policy
        decides. When may_promote_tools is off the child's tools may not be
        sealed ACTIVE in the first place, so require_state already filters them
        and merge_back ends up pulling nothing. Pass an explicit `approve`
        callback to merge anyway, one spec at a time.
        """
        merged: list[str] = []
        collect = self._collect_approved
        for name, spec in child_registry._tools.items():
            if spec.state != require_state:
                continue
            if not collect(spec, approve):
                continue
            self.registry.register(spec, replace=True)
            merged.append(name)
        return merged

    @staticmethod
    def _collect_approved(
        spec: ToolSpec,
        approve: Callable[[ToolSpec], bool] | None,
    ) -> bool:
        """Explicit callback always wins. With none, accept: require_state
        already filtered, and the caller who owns the policy decides whether
        merge_back is reachable at all (see `denied_reason`)."""
        return approve(spec) if approve is not None else True

    def denied_reason(self) -> str | None:
        """Why merge_back would refuse to pull anything, or None if it won't."""
        if self.policy is not None and not self.policy.may_promote_tools:
            return (
                "may_promote_tools is off: children may not seal tools ACTIVE, "
                "so there is nothing at ACTIVE to merge. Pass approve=... to "
                "accept on a case-by-case basis."
            )
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "spawned": self.spawn_count,
            "max_depth": self.max_depth,
            "children": [r.to_dict() for r in self.history],
        }


__all__ = ["Spawner", "SpawnRecord", "ShareMode"]
