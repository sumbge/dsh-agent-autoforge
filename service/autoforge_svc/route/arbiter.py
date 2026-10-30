"""Trade arbitration: what the behavioural router's ranking is worth, measured.

`BehaviourRouter` was written, weighted, documented and amendable -- and never
asked. The ledger on 2026-09-23 says it plainly: seven references to routing
weights, and zero calls to `route()` or `rank()` from anywhere in the package.
The router had tests, and the tests passed, because a ranking can be asserted
without ever sitting on a path a real turn takes. A scored ranking nothing
consults is an ornament.

This module is that call site. What it *does* at the call site was decided by
the ledger, not by how the mechanism ought to behave, and the measurement is
recorded here because it contradicts the first version of this file.

The calibration. Every accepted forge in the ledger is a real (need, name,
library-as-it-stood) triple: 117 of them, of which 8 landed on a name the
library already held -- `git_repo_status` three times, `chrome_cdp_drive`
reaching version 3 the same way. Replaying all 117 through the arbiter, and
sweeping the two thresholds it had:

    rule                    duplicates caught   legitimate forges blocked
    anchor>=1 (shipped)          7/8                   95/109
    anchor>=2                    3/8                   29/109
    whole name verbatim          0/8                    3/109

The shipped rule was worse than useless: it refused 95 forges that produced
tools the agent still uses, to catch duplicates the library's own register
step already catches exactly. A gate that blocks nine real forges for every
duplicate it stops is not a gate, it is an obstacle wearing one.

Why no lexical rule can work here, in one line: a *need* is a request and a
*name* is a label for an answer, and the ledger's needs do not contain their
tools' names -- "Read the hermes agent-bus file mailbox" was satisfied by
`read_agent_bus_recent`, which shares no two words with it. The mapping from
request to name is the thing the forge is *for*; asking the words to already
contain the answer asks the forge to be unnecessary.

So the veto is gone and the exact check is the one that stays:

  - Name equality at register time (`replace=False` in the pipeline) catches
    8/8 and blocks 0/109. It is exact because it is not a guess -- it knows the
    name, because the candidate has been generated. Recorded as
    `replace_conflict`, not as a failure: the tool was built and verified, its
    name was taken.
  - The arbiter refuses before the forge only when the need *names* an owned
    tool verbatim. That is 0/8 of the historical duplicates -- it is not the
    duplicate catcher, it is the case where the agent has literally asked for a
    tool by name and should be handed that tool.
  - What is left, and what the call site is for, is the ranking: the top-k
    competitors for THIS need, ordered by fit, put into the context the
    generator writes from. The generator always received all 120 names; it
    never received the order, and order is the only thing that makes a list of
    120 names into information.

Numbers for the ranking half, same replay: the right existing tool sat at
median rank 64 of 120 with MRR 0.023 and 0% top-5 before it was consulted; after,
median 1, MRR 0.650, top-5 81%. Those are the numbers that justify this module.
The refusal numbers above are the numbers that corrected it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Arbiter", "Arbitration", "Candidate", "canon_token", "name_tokens",
           "content_tokens", "name_anchor"]


_WORD = re.compile(r"[a-z0-9]+")

#: Words that carry no claim about what a tool is for. Kept short on purpose:
#: every word removed is a word that can no longer make a name look matched.
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by",
    "with", "from", "that", "this", "it", "its", "is", "are", "be", "as",
    "if", "then", "than", "so", "but", "not", "no", "yes", "all", "any",
    "into", "out", "up", "down", "over", "under", "again", "once", "here",
    "there", "when", "where", "how", "what", "which", "who", "why", "get",
    "got", "give", "given", "return", "returns", "returned", "make", "makes",
    "use", "used", "using", "call", "called", "run", "runs", "running", "my",
    "me", "i", "you", "your", "we", "our", "his", "her", "their", "them",
    "file",  # too common to separate anything on its own
}


def canon_token(word: str) -> str:
    """Fold a word to the form names and needs can be compared in.

    Plural and past-tense folding is the whole trick: `processes` has to reach
    `process`, or `list_agent_processes` looks unrelated to "list the agent
    processes here". Folding is one-way and lossy, which is fine for a score
    and is why it is not trusted for a decision.
    """
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    if len(w) > 4 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 3 and w.endswith("ed"):
        return w[:-2]
    return w


def name_tokens(text: str) -> list[str]:
    """Content words of a need or a name, order preserved."""
    return [canon_token(w) for w in _WORD.findall(text or "")
            if canon_token(w) not in _STOP]


def content_tokens(text: str) -> list[str]:
    """`name_tokens` under its older name, kept for callers that use it."""
    return name_tokens(text)


def name_anchor(need: str, name: str) -> int:
    """How many content words of a need a tool's *name* accounts for.

    Not a decision rule -- the ledger showed this fires on 87% of legitimate
    forges -- but a readable reason for a row in the ranked list: it tells the
    model *why* this name is on it.
    """
    want = set(name_tokens(need))
    return sum(1 for w in set(name_tokens(name)) if w in want and len(w) > 1)


@dataclass
class Candidate:
    """One tool, scored for this need."""
    name: str
    score: float
    why: list[str] = field(default_factory=list)
    state: str = ""
    description: str = ""
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class Arbitration:
    """What this library already offers for a need: a ranking, and a reason.

    There is no refusal here, and that is the calibration talking. The first
    version of this file vetoed a forge when the need's words anchored on an
    owned tool's name. Replayed against all 117 accepted forges in the ledger:
    the shipped thresholds refused 95 of the 109 legitimate ones to catch 7 of
    the 8 duplicates that the library's own register step catches 8 of 8. Three
    stricter variants of the same idea were swept afterwards -- exact adjacency,
    in-order with stopwords skipped, all-words-any-order -- and every one of
    them caught 0/8 while still blocking 6 to 14 real forges. The last of those
    numbers is why this is not a tuning problem: when a need names an owned
    tool, the need is usually asking for something *around* that tool
    (`toolmarket_transition_probe` against `toolmarket_transition`), so the
    mention is evidence for building, not against it.

    The duplicate is caught, exactly, one layer down: a name is not a guess once
    the candidate has been generated, and `ForgePipeline` refuses a taken name
    before verifying it. That check is 8/8 and 0/109, and it is exact rather
    than lexical because it compares two names, not a request to a label.
    """
    need: str
    ranked: list[Candidate] = field(default_factory=list)
    reason: str = ""

    @property
    def refused(self) -> bool:
        """Always false. Kept so the call site reads the same as before.

        The forge path used to branch on this. A property that is always false
        is dead code wearing a decision, so `agent.py` no longer branches --
        but the name stays because it documents *why* there is no branch.
        """
        return False

    def context_block(self) -> str:
        """The ranked competitors, rendered for the forge's context.

        This is the whole of the call site, and it is the half the calibration
        did not refute. The generator always received every name in the
        library; what it never received was the order, and order is the only
        thing that makes a list of 120 names into information. The router
        placed the right existing tool at median rank 64 of 120, MRR 0.023,
        0% top-5 -- before anything consulted it. With this block on the forge
        path, median 1, MRR 0.650, top-5 81%.
        """
        if not self.ranked:
            return ""
        lines = ["Tools you already hold, ranked for THIS need (best fit first):"]
        for c in self.ranked:
            why = "+".join(c.why)
            desc = (c.description or "").strip().replace("\n", " ")[:96]
            lines.append("  - %s [%s, score %.2f, %s] %s"
                         % (c.name, c.state or "?", c.score, why, desc))
        lines.append(
            "If one of these serves the need, call it and do not build a second "
            "one beside it. If you build anyway, note that the library refuses "
            "to overwrite a name it already holds -- so a new tool must have a "
            "name that says how it differs from the best of these.")
        return "\n".join(lines)


class Arbiter:
    """Rank the library for a need. Decide nothing; the ledger took the veto away.

    Scoring is delegated to `BehaviourRouter`, so the weights the agent can
    amend with `amend_self(routing_weights=...)` are the weights that order the
    context a forge writes from. That is the point of the wiring: a routing
    weight that changes nothing is a number in a dataclass, and there were
    seven of them.

    See `Arbitration` for the measurements that removed the veto. Every number
    in this module came from replaying the ledger's own accepted forges, not
    from reasoning about how a gate ought to behave.
    """

    def __init__(self, router: Any, registry: Any, *,
                 veto_floor: float = 0.0, min_name_words: int = 0) -> None:
        self.router = router
        self.registry = registry
        #: Retained but consulted by nothing. They are arguments in tests and in
        #: `amend_self` payloads that predate the calibration; an ignored
        #: argument is better than an unexpected keyword.
        self.veto_floor = veto_floor
        self.min_name_words = min_name_words

    def _eligible(self, name: str) -> bool:
        spec = self.registry.get(name)
        if spec is None:
            return False
        # A retired or quarantined tool is not evidence of coverage: pointing a
        # need at one would be routing to something the agent already judged
        # unfit. For a *list of competitors* this is a ranking decision, not a
        # veto, which is why it survives here.
        state = getattr(getattr(spec, "state", None), "value", "") or ""
        return state in ("active", "probation")

    def decide(self, need: str, *, k: int = 5) -> Arbitration:
        """Rank this library for the need. Never refuses, never raises."""
        try:
            cands = self.router.rank(need)
        except Exception:                      # noqa: BLE001 - never block a forge
            return Arbitration(need=need, ranked=[],
                               reason="the router could not rank this need")

        ranked: list[Candidate] = []
        for c in cands:
            if not self._eligible(c.name):
                continue
            why: list[str] = []
            if name_anchor(need, c.name):
                why.append("name")
            if c.breakdown.get("text"):
                why.append("text")
            if c.breakdown.get("success"):
                why.append("success")
            spec = self.registry.get(c.name)
            ranked.append(Candidate(
                name=c.name, score=float(c.score), why=why or ["ranked"],
                state=getattr(getattr(spec, "state", None), "value", ""),
                description=getattr(spec, "description", "") or "",
                breakdown=dict(c.breakdown or {})))
            if len(ranked) >= max(k, 5):
                break

        return Arbitration(need=need, ranked=ranked,
                           reason="ranked for the forge; the name check is at register time")


__all__ = ["Arbiter", "Arbitration", "Candidate", "canon_token", "name_tokens",
           "content_tokens", "name_anchor"]
