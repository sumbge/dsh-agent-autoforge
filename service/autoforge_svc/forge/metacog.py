"""Meta-cognitive forge: the agent reflects on its tool library and proactively
fills capability gaps.

This is the mechanism that makes autoforge *anticipate* needs instead of
reacting to them. On a schedule (or triggered by task patterns), the
meta-cognitive loop:

1. ANALYSE — scans the tool library, looks for:
   - Tools with low trigger rates (being ignored)
   - Tasks that repeatedly failed with "no tool matches" (capability gaps)
   - Tools that could be generalised (e.g. "ISBN validator" → "identifier validator")

2. GENERATE — for each gap, proposes a new tool spec.

3. EVALUATE — ranks proposals by estimated value (trigger potential × gap
   frequency). Only the top-k get forged.

4. ACT — runs the forge pipeline for each accepted proposal.

The meta-cog is an optional plugin. Without it the framework works exactly the
same — tools are made on demand. With it, the framework *grows* proactively.

Implementation note: the meta-cog uses the task trace (history of user queries
and tool calls) to identify patterns. In standalone mode, it analyses the
tool library alone (what's missing relative to what's there).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.llm import LLMClient
from ..core.message import Message
from ..forge.generator import GeneratedTool, extract_json
from ..tools.registry import ToolRegistry
from ..tools.spec import ToolSpec

_ANALYSE_SYSTEM = (
    "You are a meta-cognitive architect reviewing an agent's tool library.\n"
    "Your job: identify capability gaps — specific recurring needs that the\n"
    "current tool library does NOT serve well.\n\n"
    "For each gap, propose a tool that would fill it.\n\n"
    "Output STRICT JSON:\n"
    "{\n"
    '  "gaps": [\n'
    "    {\n"
    '      "need": "one-line description of the capability gap",\n'
    '      "rationale": "why this is a gap and how often it comes up",\n'
    '      "proposed_tool": {\n'
    '        "name": "...",\n'
    '        "description": "...",\n'
    '        "parameters": {...},\n'
    '        "tags": ["..."]\n'
    "      }\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Focus on gaps that are:\n"
    "1. GENERAL — not one-off hacks, but reusable capabilities\n"
    "2. COMPOSABLE — could be combined with existing tools\n"
    "3. VERIFIABLE — a probe can test whether it works\n"
    "4. HIGH VALUE — would save many future turns\n\n"
    "Generate 2-4 gaps max. Quality over quantity."
)


@dataclass
class GapProposal:
    need: str
    rationale: str
    tool_spec: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"need": self.need, "rationale": self.rationale[:200], "tool": self.tool_spec.get("name", "")}


@dataclass
class MetaCogReport:
    gaps_found: int = 0
    proposals: list[GapProposal] = field(default_factory=list)
    forged: int = 0
    failed: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "gaps_found": self.gaps_found,
            "forged": self.forged,
            "failed": self.failed,
            "proposals": [p.to_dict() for p in self.proposals],
        }


class MetaCognition:
    """Proactive gap analysis and tool generation."""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        pipeline: Any,
        *,
        max_gaps: int = 4,
        min_interval_seconds: float = 300.0,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.pipeline = pipeline
        self.max_gaps = max_gaps
        self.min_interval_seconds = min_interval_seconds
        self._last_run: float = 0.0
        self._run_count: int = 0

    def analyse(self, task_history: list[str] | None = None) -> MetaCogReport:
        """Analyse the tool library and propose new tools for gaps."""
        now = time.time()
        if now - self._last_run < self.min_interval_seconds:
            return MetaCogReport(gaps_found=0)
        self._last_run = now
        self._run_count += 1
        started = time.perf_counter()

        # Build current tool summary
        tools_summary = "\n".join(
            f"- {s.name}: {s.description} [{s.state.value}] tags={s.tags}"
            for s in self.registry._tools.values()
        ) if self.registry._tools else "(no tools yet)"

        recent_tasks = ""
        if task_history:
            recent_tasks = "\n".join(
                t[:200] for t in task_history[-5:]
            )

        prompt = (
            f"Current tool library:\n{tools_summary}\n\n"
            + (f"Recent task history:\n{recent_tasks}\n\n" if recent_tasks else "")
            + "Analyse and identify capability gaps."
        )

        resp = self.llm.chat(
            [Message.system(_ANALYSE_SYSTEM), Message.user(prompt)],
            tools=None,
        )
        data = extract_json(resp.content)
        if data is None or not isinstance(data, dict):
            return MetaCogReport(gaps_found=0)

        gaps = data.get("gaps") or []
        proposals = []
        for g in gaps[:self.max_gaps]:
            if not isinstance(g, dict) or not g.get("need"):
                continue
            proposals.append(GapProposal(
                need=g["need"],
                rationale=g.get("rationale", ""),
                tool_spec=g.get("proposed_tool", {}),
            ))

        report = MetaCogReport(gaps_found=len(proposals), proposals=proposals)

        # Forge each accepted proposal
        for prop in proposals:
            need = prop.need
            result = self.pipeline.forge(need)
            if result.ok:
                report.forged += 1
            else:
                report.failed += 1

        report.duration_ms = (time.perf_counter() - started) * 1000
        return report


__all__ = ["MetaCognition", "MetaCogReport", "GapProposal"]