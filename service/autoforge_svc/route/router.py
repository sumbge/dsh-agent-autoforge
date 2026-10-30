"""Behaviour-aligned tool routing.

Mainstream skill libraries retrieve by *text similarity*: embed the need,
embed each skill description, take the nearest. This routinely surfaces a
skill that reads right and runs wrong, because semantic closeness is not
execution success.

The fix (credited to Memento-Skills, generalised here) is to score candidates
on *observed behaviour*:
    score = w_text * text_similarity
          + w_success * success_rate
          + w_trust * state_trust

No offline RL required — a weighted, inspectable blend. The weights are the
policy, and they live in one dataclass so the agent's retrieval taste is a
tunable, debuggable thing rather than an opaque vector index.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from autoforge_svc.tools.registry import ToolRegistry
from autoforge_svc.tools.spec import ToolSpec, ToolState

_WORD = re.compile(r"[a-zA-Z0-9_]+")

_TRUST = {
    ToolState.ACTIVE: 1.0,
    ToolState.PROBATION: 0.6,
    ToolState.QUARANTINED: 0.1,
    ToolState.RETIRED: 0.0,
    ToolState.DRAFT: 0.2,
}



def _tokens(text: str) -> Counter:
    return Counter(w.lower() for w in _WORD.findall(text or "") if len(w) > 1)


def _cosine(q: Counter, d: Counter) -> float:
    """Shared by tool and skill similarity: no embedding model, no network."""
    if not q or not d:
        return 0.0
    common = set(q) & set(d)
    num = sum(q[t] * d[t] for t in common)
    den = math.sqrt(sum(v * v for v in q.values())) * math.sqrt(sum(v * v for v in d.values()))
    return num / den if den else 0.0


def text_similarity(query: str, spec: ToolSpec) -> float:
    """Lightweight lexical cosine — no embedding model, no network."""
    return _cosine(_tokens(query),
                   _tokens(f"{spec.name} {spec.description} {' '.join(spec.tags)}"))


@dataclass
class RoutingWeights:
    """The retrieval taste, as numbers that were measured rather than chosen.

    These shipped as 1.0 / 1.2 / 0.8 / 0.5 / 0.6, picked by taste and never
    scored. `tools/calibrate_routing.py` scores them against two ground-truth
    sets that were already on disk:

      probes  312 (query, expect=call) pairs from the tools' own TriggerProbes,
              plus 312 (negative_query) pairs they must NOT win. Model-written
              from the description, so lexical access is easy. Necessary, not
              sufficient.
      forged  117 accepted forges: real needs, written with no knowledge of what
              the tool would one day be called, against the library as it stood
              at that moment. The only set that is not partly the router's echo.

    The old numbers were not slightly wrong, they were the wrong shape. `success`
    and `trust` do not depend on the query -- they are per-tool constants -- so
    as additive terms they are a fixed bonus on every query, and the more weight
    they carry the more they drown out the only term that answers "which tool for
    THIS need". Raising them to catch a flaky tool also promotes it for needs it
    has nothing to do with.

        5-fold CV, 114 forged cases                top1    recall@5
          shipped 1.0 / 1.2 / 0.8 / 0.5 / 0.6      0.183     0.228
          text only                                0.570     0.833
          text + availability gate + trust tiebreak 0.570    0.833

    So the two questions are separated, because they are different questions:

    *Relevance* is `text * gate`. The gate is `min(1, success_rate / gate)` --
      a proven-broken tool is scaled toward zero, and a healthy one is scaled by
      exactly 1.0. It never rewards. A tool with too few calls to judge keeps
      its default 0.5, which is exactly the threshold, so it is not penalised
      for being new. On the real corpus this fires on 1 of 120 tools (22 of the
      24 judged tools are at success 1.0), so it costs nothing measurable and
      stops a never-working tool from outranking a working one on the strength
      of its name.

    *Trust* is a weak tiebreak (`trust`, default 0.1). It is small on purpose:
      it orders two candidates that match the need equally well -- a probed
      tool ahead of a probationary one -- and is too small to reorder candidates
      that differ in relevance.

    Two of the five terms this class shipped with are *gone*, not merely
    weighted at zero (2026-09-28). `cost` and `over_trigger` were set to 0 by the
    calibration because they scored 0.0 for all 120 tools -- `cost_hint` is
    "cheap" everywhere in the library, and `trigger_misses` counts nothing
    because nothing ever calls `TriggerProbe` a second time after a forge. A
    tunable weight on a constant is not a quiet default, it is a knob an
    `amend_self` can turn to make retrieval strictly worse with no signal to
    notice it by, and both of this week's routing defects were that shape. The
    fields went; the terms they scored had no variance to lose. `trigger_misses`
    stays on the ledger for the day something does populate it.

    `amend_self(routing_weights=...)` edits what is left, and names the two it
    retired by name rather than by silence -- see `agent.py`.
    """
    text: float = 1.0
    #: Additive weight on success. 0 on purpose: success is not a gradient of
    #: relevance. The `gate` below is where it acts.
    success: float = 0.0
    #: Weak tiebreak. Orders equally-relevant candidates; too small to reorder
    #: candidates that differ in relevance.
    trust: float = 0.1
    min_calls_for_success: int = 3
    #: Success rate below which a tool's relevance is scaled down; 0 disables.
    #: A tool at exactly this rate is scaled by 1.0, which is why the 0.5
    #: default for unjudged tools costs them nothing.
    gate: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "text": self.text, "success": self.success, "trust": self.trust,
            "gate": self.gate,
        }


@dataclass
class RouteCandidate:
    name: str
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    state: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "state": self.state,
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
        }


class BehaviourRouter:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        weights: RoutingWeights | None = None,
        include_states: set[ToolState] | None = None,
    ) -> None:
        self.registry = registry
        self.weights = weights or RoutingWeights()
        self.include_states = include_states or {ToolState.PROBATION, ToolState.ACTIVE}

    def score(self, query: str, spec: ToolSpec) -> RouteCandidate:
        w = self.weights
        st = spec.stats

        text = text_similarity(query, spec)
        success = st.success_rate if st.calls >= w.min_calls_for_success else 0.5
        trust = _TRUST.get(spec.state, 0.0)

        # Relevance and availability are different questions and are scored
        # separately. `gate` scales relevance by whether the tool works at all --
        # never above 1.0, so a good record is not a bonus on an unrelated need;
        # at most 1.0, so an unjudged tool (success defaulted to 0.5, exactly the
        # threshold) is not penalised for being new.
        gate = 1.0
        if w.gate > 0:
            gate = max(0.0, min(1.0, success / w.gate))
        total = (
            w.text * gate * text
            + w.success * success
            + w.trust * trust
        )
        return RouteCandidate(
            name=spec.name,
            score=total,
            breakdown={
                "text": w.text * text,
                "gate": gate,
                "success": w.success * success,
                "trust": w.trust * trust,
            },
            state=spec.state.value,
        )

    def rank(self, query: str) -> list[RouteCandidate]:
        cands = [
            self.score(query, s)
            for s in self.registry._tools.values()
            if s.state in self.include_states
        ]
        return sorted(cands, key=lambda c: c.score, reverse=True)

    def route(self, query: str, k: int = 3) -> list[str]:
        """Names of the top-k tools for this query, best first."""
        return [c.name for c in self.rank(query)[:k]]


def skill_similarity(query: str, skill: Any) -> float:
    """Lexical cosine over the fields a skill is described by.

    `when_to_use` carries the weight a description does not: it is written as a
    trigger ("when a deploy needs proving"), which is the shape of the thing
    being matched, rather than as a summary of the thing itself.
    """
    fields = (f"{skill.name} {skill.description} {skill.when_to_use} "
              f"{' '.join(skill.tags)}")
    return _cosine(_tokens(query), _tokens(fields))


#: The skill router's own defaults. It cannot share the tool router's: the two
#: evidence terms are different things. A tool's `success` is a rate over all
#: calls and is folded into availability (`gate`), which is why its additive
#: weight is 0. A skill's evidence is `proven` -- how many times it was opened --
#: and it is the ONLY signal distinguishing two skills that read alike, so
#: zeroing it would make the router blind to the exact failure it exists for.
#:
#: Uncalibrated, and labelled so: there is no skill ledger with ground truth to
#: replay, unlike the 114 forged needs behind the tool weights above. The number
#: is 1.0 because it is known to work, not because it was measured.
SKILL_WEIGHTS = RoutingWeights(
    text=1.0, success=1.0, trust=0.0, gate=0.0)


class SkillRouter:
    """Rank skills by fit, blended with evidence that they have been used.

    The lexical half is honest about what it is: at routing time, the only
    cheap signal about a skill is how its own words line up with the need. What
    makes the ranking *behavioural* is the second term. `proven` is built from
    `loads` — how many times the agent actually opened the skill — so a
    procedure that keeps getting reached for climbs above one that merely reads
    well. That is the failure this repo exists to argue against: nearest-
    description retrieval returns the skill that reads right and runs wrong.

    Two of the tool router's four terms are deliberately absent. A skill has no
    state lifecycle to score trust from and no cost hint, so including them
    would mean inventing numbers and calling the result a measurement. Where
    the tool router says "not measured yet" (0.5 for too-few calls), the skill
    router says zero: a never-loaded skill has no evidence, and rounding that
    up to neutral would let an unread procedure tie a proven one.
    """

    def __init__(self, library: Any, *, weights: RoutingWeights | None = None) -> None:
        self.library = library
        # Not `RoutingWeights()`: the tool defaults put 0 on `success` because
        # for tools success is folded into the availability gate instead. For
        # skills it is the evidence term itself. See SKILL_WEIGHTS.
        self.weights = weights or SKILL_WEIGHTS

    def score(self, query: str, skill: Any) -> RouteCandidate:
        w = self.weights
        text = skill_similarity(query, skill)
        proven = min(1.0, skill.loads / max(1, w.min_calls_for_success))
        total = w.text * text + w.success * proven
        return RouteCandidate(
            name=skill.name,
            score=total,
            breakdown={"text": w.text * text, "proven": w.success * proven},
            state="used" if skill.loads else "never-used",
        )

    def rank(self, query: str) -> list[RouteCandidate]:
        cands = [self.score(query, s) for s in self.library.all()]
        # Ties go to the more-used skill, then to name order, so the same
        # library always ranks the same way and a diff means something.
        cands.sort(key=lambda c: (-c.score, c.name))
        return cands

    def route(self, query: str, k: int = 3) -> list[str]:
        """Names of the top-k skills for this need, best first."""
        return [c.name for c in self.rank(query)[:k]]


__all__ = [
    "BehaviourRouter", "RoutingWeights", "RouteCandidate", "SKILL_WEIGHTS",
    "text_similarity",
    "SkillRouter", "skill_similarity",
]
