"""Agent topology: the agent can inspect and redesign its own multi-agent
architecture.

This is the dimension autoforge was missing relative to TPGO (ACL 2026).
A single agent loop, however many tools it can forge, is still a single
loop. The agent should be able to decide:

  - How many agents exist in its system
  - What role each agent plays
  - How they communicate (tool bridge, shared registry, pipe)
  - Which topology works better for which task type

The topology is modelled as a directed graph where nodes are agent roles
and edges are communication channels. The agent can mutate this graph,
run trials, and keep the winning configuration.

A topology is a *proposal*. It is not the runtime itself — the runtime
reads a topology and instantiates the described agents.
"""
from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import re


class RoleType(str, Enum):
    COORDINATOR = "coordinator"   # delegates, merges results
    WORKER = "worker"             # focused executor
    CRITIC = "critic"             # reviews outputs, provides feedback
    GATE = "gate"                 # filters/validates before passing on
    FORGE = "forge"               # specialised tool-forging agent


@dataclass
class AgentNode:
    """One agent in the topology."""
    id: str
    role: RoleType
    system_prompt_hint: str = ""
    tools_whitelist: list[str] = field(default_factory=list)
    max_turns: int | None = None


@dataclass
class TopologyEdge:
    """A communication channel between two agents."""
    source: str
    target: str
    channel: str = "tool_call"    # tool_call | shared_registry | pipe | broadcast


@dataclass
class Topology:
    """A complete multi-agent architecture proposal."""
    nodes: list[AgentNode] = field(default_factory=list)
    edges: list[TopologyEdge] = field(default_factory=list)
    rationale: str = ""
    created_at: float = field(default_factory=time.time)
    fitness: float = 0.0           # set after evaluation
    trials: int = 0

    def add(self, node: AgentNode) -> "Topology":
        self.nodes.append(node)
        return self

    def connect(self, source: str, target: str, channel: str = "tool_call") -> "Topology":
        self.edges.append(TopologyEdge(source, target, channel))
        return self

    def validate(self) -> list[str]:
        errors = []
        ids = {n.id for n in self.nodes}
        for e in self.edges:
            if e.source not in ids:
                errors.append(f"edge source {e.source!r} not in nodes")
            if e.target not in ids:
                errors.append(f"edge target {e.target!r} not in nodes")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [{"id": n.id, "role": n.role.value, "hint": n.system_prompt_hint[:80]}
                      for n in self.nodes],
            "edges": [{"from": e.source, "to": e.target, "channel": e.channel}
                      for e in self.edges],
            "rationale": self.rationale[:200],
            "fitness": round(self.fitness, 3) if self.fitness else None,
            "trials": self.trials,
        }

    @classmethod
    def single_agent(cls, prompt_hint: str = "") -> "Topology":
        return cls(
            nodes=[AgentNode(id="main", role=RoleType.COORDINATOR,
                             system_prompt_hint=prompt_hint)],
        )


_TOPOLOGY_MUTATE_SYSTEM = """You are an AI systems architect. You are given the current
multi-agent topology and a task or failure report. Design an IMPROVED topology.

The topology is a graph of agents. Each agent has:
  - id: unique name
  - role: coordinator | worker | critic | gate | forge
  - system_prompt_hint: what this agent should focus on
  - tools_whitelist: which tools it can access (empty = all)

Edges define who talks to whom. Communication channels:
  - tool_call: source calls target via a tool
  - shared_registry: both share a tool library
  - pipe: source streams results to target

Rules for good topologies:
- coordinators delegate and merge, they do NOT do fine work themselves
- workers focus on one type of task
- critics review worker outputs before they reach the coordinator
- a forge agent specialises in tool creation, not task execution
- keep it minimal — more agents means more latency
- 2-4 agents is usually enough

Output STRICT JSON only:
{
  "rationale": "why this topology fits the task",
  "nodes": [{"id": "...", "role": "...", "system_prompt_hint": "...", "tools_whitelist": [...]}],
  "edges": [{"source": "...", "target": "...", "channel": "tool_call"}]
}
"""

_TOPOLOGY_ANALYSE_SYSTEM = """You are a systems analyst. Given a topology and a trace of
execution (tool calls, errors, results), identify what went wrong architecturally.

Output STRICT JSON:
{
  "bottleneck": "which agent/edge caused the problem",
  "root_cause": "one sentence",
  "suggested_change": "what to add/remove/reassign"
}
"""


class TopologyDesigner:
    """An LLM-powered topology creator and mutator."""

    def __init__(self, llm: Any, generator_model: str | None = None) -> None:
        self.llm = llm
        self.generator_model = generator_model

    def design(self, task: str, current_topology: Topology | None = None,
               failure_report: str = "", max_agents: int = 4) -> Topology:
        """Design a topology from scratch or mutate an existing one.

        `max_agents` is a soft ceiling on team size, passed to the model as a
        budget hint. The result is not truncated if the model overshoots —
        clipping nodes would leave dangling edges and silently degrade the
        whole design, which is worse than a slightly oversized team.
        """
        from ..core.message import Message
        from ..forge.generator import extract_json

        prompt = f"Task: {task}\n"
        prompt += f"\nTeam-size budget: at most {max_agents} agents.\n"
        if current_topology:
            prompt += f"\nCurrent topology:\n{json.dumps(current_topology.to_dict(), indent=2)}\n"
            prompt += f"\nCurrent fitness: {current_topology.fitness}\n"
        if failure_report:
            prompt += f"\nFailure report:\n{failure_report}\n"
        prompt += "\nDesign the topology now."

        resp = self.llm.chat(
            [Message.system(_TOPOLOGY_MUTATE_SYSTEM), Message.user(prompt)],
            tools=None,
        )
        data = extract_json(resp.content)
        if data is None:
            return Topology.single_agent(task)

        return self._parse_topology(data, task)

    def analyse(self, topology: Topology, trace: list[dict[str, Any]]) -> dict[str, str]:
        """Analyse a topology's execution trace for bottlenecks."""
        from ..core.message import Message
        from ..forge.generator import extract_json

        prompt = (
            f"Topology:\n{json.dumps(topology.to_dict(), indent=2)}\n"
            f"\nExecution trace (last 10 events):\n"
            + "\n".join(
                json.dumps(t, ensure_ascii=False)[:200]
                for t in trace[-10:]
            )
            + "\nAnalyse and suggest improvements."
        )
        resp = self.llm.chat(
            [Message.system(_TOPOLOGY_ANALYSE_SYSTEM), Message.user(prompt)],
            tools=None,
        )
        data = extract_json(resp.content)
        if data is None:
            return {"bottleneck": "unknown", "root_cause": "analysis failed",
                    "suggested_change": "retry"}
        return {
            "bottleneck": str(data.get("bottleneck", "unknown")),
            "root_cause": str(data.get("root_cause", "")),
            "suggested_change": str(data.get("suggested_change", "")),
        }

    @staticmethod
    def _parse_topology(data: dict, fallback_task: str) -> Topology:
        t = Topology(rationale=str(data.get("rationale", "")))
        for n in data.get("nodes", []):
            try:
                role = RoleType(n.get("role", "worker"))
            except ValueError:
                role = RoleType.WORKER
            t.add(AgentNode(
                id=str(n.get("id", f"agent_{len(t.nodes)}")),
                role=role,
                system_prompt_hint=str(n.get("system_prompt_hint", "")),
                tools_whitelist=list(n.get("tools_whitelist") or []),
            ))
        for e in data.get("edges", []):
            t.connect(str(e.get("source", "")), str(e.get("target", "")),
                      str(e.get("channel", "tool_call")))
        errors = t.validate()
        if errors:
            # Degrade gracefully: return a simple topology rather than crashing
            return Topology.single_agent(fallback_task)
        return t


__all__ = ["Topology", "TopologyDesigner", "AgentNode", "TopologyEdge", "RoleType"]