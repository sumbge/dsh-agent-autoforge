"""The agent loop.

Deliberately thin in mechanism, deliberately free in policy.

The three things that make this loop different from every other agent loop:

1. **No hidden turn cap.** `max_turns=None` (the default) means the agent runs
   until it produces a text answer or calls `terminate`. A cap exists only if
   the caller imposes one — and when one is imposed, it is logged, not silent.
   The agent decides when a task is done, not an arbitrary number baked into
   the framework.

2. **Self-termination is a first-class tool.** The agent can end its own turn
   loop, returning a summary and a reason. Combined with `unlimited_turns`,
   this means the stopping condition is *semantic* (the agent judges it's done)
   rather than *mechanical* (the counter hit N).

Everything else — tool dispatch, message threading — is ordinary, because that
part isn't what's broken in other frameworks.

3. **The context is compacted, never silently truncated.** An unbounded loop
   meets a bounded context window eventually, and the failure there is a
   provider error at the far end of the task. `compaction.py` replaces the
   middle of the history with a summary at a provably safe cut point. The loop
   itself only calls it; the policy — when, how much to keep, and what to do if
   summarising fails — lives there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .llm import LLMAborted, LLMClient
from .message import Message
from .compaction import OUTPUT_MAX_CHARS, bound_output

DEFAULT_SYSTEM = (
    "You are a capable agent. Call a tool when it genuinely helps; answer "
    "directly when none is needed. Do not call a tool you do not need.\n"
    "When you have fully completed the task, either answer in plain text or "
    "call terminate with a summary of what you did."
)



def image_parts(text: str) -> list[dict[str, Any]] | None:
    """The wire content for a line, pictures included, or None if there are none.

    A placeholder is what the terminal shows and what the transcript keeps; the
    picture itself lives in a file the placeholder names. This is the seam where
    the two get put back together, and it is here -- rather than in the client --
    because "which file does this line mean" is a question about the operator's
    line, not about HTTP.

    Returning None when there are no pictures is deliberate: a message that goes
    to the far end as a plain string is a message that a *text* model can still
    be given, and a request whose content is a list of one text part is not.
    """
    # Ported service: no terminal line editor here, so no placeholder syntax to
    # expand. The images a caller passes are handled by the caller's own layer.
    return None


@dataclass
class AgentResult:
    content: str
    messages: list[Message]
    turns: int
    tool_calls: list[str] = field(default_factory=list)
    self_terminated: bool = False
    termination_reason: str = ""
    # Kept apart from self_terminated on purpose: "the agent judged itself
    # done" and "a person stopped it" are different facts, and a report that
    # blurs them credits the agent with a decision it did not make.
    stopped_by_operator: bool = False

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)


class _TerminateSignal(BaseException):
    """Raised by the terminate tool to end the loop cleanly.

    Inherits BaseException (not Exception) so that the registry's
    `except Exception` tool-boundary handler does NOT swallow it — a
    self-termination must propagate all the way back to the loop.
    """

    def __init__(self, summary: str, reason: str = "") -> None:
        super().__init__(summary)
        self.summary = summary
        self.reason = reason


def _notify(hook: Callable[..., None] | None, *args: Any) -> None:
    """Call an observer hook, swallowing whatever it throws.

    Every hook here is a *reporter* — progress lines, trace records, UI. None of
    them decide anything, so a broken reporter must not take the run down with
    it. Before this, a CLI writing to a closed stream killed the task mid-loop.
    """
    if hook is None:
        return
    try:
        hook(*args)
    except Exception:  # noqa: BLE001 — reporting is never load-bearing
        pass


def _mark_run(steer: Any, live: bool) -> None:
    """Tell the channel whether a run is live, when it tracks that at all.

    The channel's reply to the operator promises that "the step in progress
    will yield to it". Only this loop can make that a fact rather than a hope,
    so the loop holds the flag -- and the guard is for the duck-typed channels
    (tests, embedders) that are usable without it: a missing marker must never
    be the thing that kills a run, exactly as a broken channel reads as "no"
    in `_operator_wants_the_floor`.
    """
    fn = getattr(steer, "begin_run" if live else "end_run", None)
    if fn is None:
        return
    try:
        fn()
    except Exception:  # noqa: BLE001 — see docstring
        pass


class Agent:
    """A tool-using loop over an LLMClient and a ToolRegistry.

    Parameters
    ----------
    max_turns:
        `None` (default) means unbounded — the agent stops when it answers or
        self-terminates. An integer imposes a cap, which is recorded on the
        result so it is visible, never silent.
    allow_self_terminate:
        Registers a `terminate` tool the model can call to end the loop with
        a summary. Requires a registry that accepts registration.
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: Any,
        *,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        allow_self_terminate: bool = True,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, Any], None] | None = None,
        on_turn: Callable[[int, Message], None] | None = None,
        on_request: Callable[[int], None] | None = None,
        on_steer: Callable[[str], None] | None = None,
        on_reply: Callable[[str], None] | None = None,
        steer: Any = None,
        compactor: Any = None,
        on_compact: Callable[[Any], None] | None = None,
        journal: Any = None,
        exposure: Any = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt or DEFAULT_SYSTEM
        self.max_turns = max_turns
        self.allow_self_terminate = allow_self_terminate
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_turn = on_turn
        self.on_request = on_request
        self.on_steer = on_steer
        #: `on_reply(text)` -- the run answering the person mid-flight. Separate
        #: from `on_turn`, which is the loop counting its own turns: this one is
        #: for the operator, and it is the difference between "it heard me" and
        #: "it said something back".
        self.on_reply = on_reply
        # Anything with `take_supplements()` / `stop_requested()`: the channel a
        # person uses to talk to this run while it is happening. Optional, so a
        # run with nobody watching is exactly what it was before.
        self.steer = steer
        # Anything with `maybe_compact(msgs, turn) -> event | None`. None means
        # the context is never compacted, which is correct for a short run and
        # is why the framework itself turns this off by default rather than
        # deciding a window size for the caller.
        self.compactor = compactor
        self.on_compact = on_compact
        #: Anything with `beat(msgs, turn=, last_action=)` -- where this loop is,
        #: written down so a wire that goes down mid-step costs a step rather
        #: than a task. Optional, and owned from outside: this loop knows the
        #: moment worth recording, but only the caller knows whether the run is
        #: one worth resuming. See `autoforge.runs`.
        self.journal = journal
        #: Anything with `exposure(protect=...) -> (hot, cold_lines, counts)`:
        #: the split between tool schemas that ride in every request and tools
        #: that are only named. Optional, and None means every visible tool is
        #: sent in full -- the behaviour before this existed, which is what a
        #: bare loop and every test that does not care about cost should get.
        #: See `autoforge/tools/exposure.py` for the measured reason it exists.
        self.exposure = exposure
        #: The last tool asked for, kept so the checkpoint can say what the run
        #: was in the middle of doing. A resume prompt that says "you were 4
        #: turns in" is far weaker than one that says "you had just called
        #: run_python", and the name is the difference.
        self._last_tool = ""
        self._terminated: _TerminateSignal | None = None
        #: How many operator lines the last `_absorb` folded in. Reset there, so
        #: a run with no steering channel never reads a stale count.
        self._folded = 0
        if allow_self_terminate:
            self._register_terminate_tool()

    def _beat(self, msgs: list[Message], turn: int, action: str = "") -> None:
        """Write down where this loop is, and never let that cost the run.

        Guarded, like every other reporter here: a checkpoint that cannot be
        written must not be the thing that kills the run it was recording. The
        loud half is on the reading side -- see `runs.scan` and the resume path
        in `ForgeAgent`, where a record that cannot be interpreted is reported
        rather than guessed at.
        """
        if self.journal is None:
            return
        try:
            self.journal.beat(msgs, turn=turn, last_action=action)
        except Exception:                             # noqa: BLE001 - see above
            pass

    def _request(self, msgs: list[Message]) -> tuple[list[Message], list[dict[str, Any]]]:
        """The messages and schemas for one request.

        Two things differ from `msgs` itself, and both are deliberate:

        * only the *hot* tool schemas are sent as tools;
        * the cold ones ride as one extra user message, placed immediately
          after the system prompt rather than at the end.

        The position is a cost decision. The provider bills a prompt prefix from
        cache only while it is byte-identical to the last request, and this
        block changes exactly when a tool is called -- so putting it early means
        a turn that calls nothing re-bills nothing, and the growing conversation
        behind it is the only part that moves. At the end it would sit after the
        whole conversation and be cached anyway; the reason for the front is
        that a model reads instructions before the transcript, and a list of
        what it has is an instruction, not a remark.

        A cold tool stays callable and stays discoverable. What is not sent is
        its parameter schema, which for 61 tools that were never called is the
        bulk of what this saves.
        """
        if self.exposure is None:
            return msgs, self.registry.schemas()
        try:
            hot, cold_lines, stats = self.exposure.exposure()
        except Exception:                                     # noqa: BLE001
            # A cost optimisation must never be the thing that loses the run.
            # Falling back to sending everything is the pre-existing behaviour:
            # expensive and correct.
            return msgs, self.registry.schemas()
        if not cold_lines:
            return msgs, hot
        from ..tools.exposure import manifest_header
        block = (manifest_header(stats, getattr(self.exposure, "window_days", 7.0))
                 + "\n" + "\n".join(cold_lines))
        # Copied, never appended to `msgs`: the transcript on disk is the
        # conversation, and a block injected per request that became part of the
        # record would be replayed by a resume as something the person said.
        req = list(msgs)
        insert_at = 1 if msgs and msgs[0].role == "system" else 0
        req.insert(insert_at, Message.user(block))
        return req, hot

    def _bounded(self, output: str, tool: str) -> str:
        """`output` cut down to the cap before it enters the context.

        Ported from Hermes' per-message truncation limits (`_CONTENT_MAX` with a
        head and a tail). Hermes applies them when it hands messages to the
        summarizer; this loop applies them at *ingest*, because the damage is
        already done by the time a summary runs: an unbounded `dir /s` or a whole
        file read is one message, and one 60k-token message can walk a transcript
        from under the threshold to three times over it in a single turn — past
        anything a cut of the *middle* can fix.

        The cap comes from the compactor's policy when a compactor is present, so
        whoever sized the window sized this too; with no compactor there is no
        threshold to protect and the module default stands.
        """
        cap = getattr(getattr(self.compactor, "policy", None),
                      "max_output_chars", OUTPUT_MAX_CHARS)
        return bound_output(output, max_chars=cap or 0, name=tool)

    def _register_terminate_tool(self) -> None:
        from ..tools.spec import ToolSpec, ToolState

        registry = self.registry
        if not hasattr(registry, "register") or "terminate" in getattr(registry, "_tools", {}):
            return

        def terminate(summary: str = "", reason: str = "") -> str:
            raise _TerminateSignal(summary or "task complete", reason)

        registry.register(
            ToolSpec(
                name="terminate",
                description=(
                    "End the task early and return a summary. Use this when the "
                    "task is fully complete or cannot proceed further."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "what was accomplished"},
                        "reason": {"type": "string", "description": "why stopping now"},
                    },
                },
                fn=terminate,
                source="builtin",
                tags=["meta"],
                state=ToolState.ACTIVE,
            )
        )

    def _absorb(self, msgs: list[Message]) -> bool:
        """Take whatever the operator said since the last check.

        Supplements land as user messages at the next safe point — before a
        model call, or right after a tool result — which is the latest moment
        they can still change the run they belong to. Returns True if the
        operator also asked to stop: the current step finishes, then the run
        ends. A stop is never applied mid-tool, because the alternative is
        killing a subprocess or a write halfway through.

        No `try/except` on purpose. The whole point is that a line the operator
        typed is never dropped, so a bug in this channel must be loud rather
        than swallowed into a run that quietly ignored them.

        Returns True when the operator asked the run to stop. How many lines were
        folded in is *accumulated* in `_folded` rather than reset here: a line
        can arrive while a tool is running, and it can only be answered at the
        top of the next turn -- the results of the tool calls already in flight
        have to be appended before any new assistant message, or the request that
        follows is malformed. `_answer_the_operator` is the consumer.
        """
        if self.steer is None:
            return False
        for text in self.steer.take_supplements():
            msgs.append(Message.user(text, images=image_parts(text)))
            _notify(self.on_steer, text)
            self._folded += 1
        return bool(self.steer.stop_requested())

    def _answer_the_operator(self, msgs: list[Message]) -> None:
        """Ask once with the tool list empty, so the answer goes to the person.

        A line folded into the message list is invisible to the person who
        typed it. The model is free to fold a question into its next tool call
        and say nothing at all, and that is not a hypothesis: the ledger on this
        host records the line "what are you doing?" absorbed at 00:00:27,
        followed by sixty more tool calls and not one word back. Saying "heard"
        is the harness talking; the reply has to be the *model* talking.

        So a line that arrives mid-run buys one question asked with no tools
        available, where answering is the only thing that can happen -- and the
        answer joins the conversation, so the work that resumes knows what the
        operator has already been told.

        Deliberately quiet on failure: their line is already in the context and
        the run is still making progress, so a broken reply turn must not cost
        them the run. The opposite trade -- a run that dies while explaining its
        own intercom -- is the one this module keeps refusing.
        """
        if not self._folded:
            return
        self._folded = 0
        try:
            resp = self.llm.chat(msgs, tools=[])
        except Exception:                     # noqa: BLE001 - see docstring
            return
        if not resp.content:
            return
        msgs.append(Message.assistant(resp.content))
        _notify(self.on_reply, resp.content)

    @staticmethod
    def _stopped(msgs: list[Message], turns: int, used: list[str]) -> AgentResult:
        return AgentResult(
            "(stopped by the operator)", msgs, turns, used,
            stopped_by_operator=True, termination_reason="stopped by the operator",
        )

    def _operator_said_something(self) -> bool:
        """Whether the operator has spoken and the loop has not consumed it.

        This is the *politeness* question, and it is deliberately not the same
        as `_operator_wants_the_floor`. A line typed while a long step runs --
        "how is it going?" -- used to arrive as an abort, because the only
        question asked of the channel was "is there anything pending?", and
        anything pending was treated as a reason to stop. So asking for a
        status update killed the download, the forge, the transcription: the
        one thing the person at the terminal did not ask for.

        Measured on this host on 2026-09-17: four separate long jobs were
        reported as "stopped at the operator's request" and every one of them
        was stopped by a question, not by a /stop. A question is not a stop
        command, and an agent that cannot tell them apart is worse than one
        that ignores them, because it destroys work while appearing responsive.

        What the answer should buy is the *reply*: the loop folds the line in
        at the next boundary and answers it (`_answer_the_operator`), without
        cutting short the step that is producing something.
        """
        if self.steer is None:
            return False
        try:
            return bool(self.steer.has_pending())
        except Exception:                     # noqa: BLE001 - reads as "no"
            return False

    def _operator_wants_the_floor(self) -> bool:
        """Whether the operator has said something the loop has not consumed.

        Asked by *long steps* about themselves -- an in-flight model call, a
        running sandbox child -- so their line lands inside the step rather than
        after it. `_absorb` at the loop boundaries is the same question asked at
        the cheap moments; this is the one asked at the expensive ones, and the
        difference is minutes.

        A stop counts as well as a note. A `/stop` that can only be honoured at
        the next boundary is not a stop, it is a note that the current step is
        still going -- and the step it is ignored by is the longest one.

        A failure here reads as "no". This is polled from inside a request; if
        the steering channel is somehow unusable, the right outcome is a run
        that finishes, not a run that dies investigating its own intercom.
        """
        if self.steer is None:
            return False
        try:
            return bool(self.steer.stop_requested())
        except Exception:                     # noqa: BLE001 - see docstring
            return False

    def _operator_should_yield(self) -> bool:
        """Whether a *cheap* step should give way so the operator gets an answer.

        The model call, not the tool run. Aborting a request costs nothing that
        cannot be asked again: no tool has run, no message has been appended,
        and the turn is given back. That is what lets a question land inside a
        long model call instead of after it.

        The expensive steps ask `_operator_wants_the_floor` instead, which is
        now strictly narrower: only a real stop. A question must never be the
        reason a download, a forge or a transcription dies -- see
        `_operator_said_something` for the measurement behind that.
        """
        if self.steer is None:
            return False
        try:
            return bool(self.steer.has_pending()) or bool(self.steer.stop_requested())
        except Exception:                     # noqa: BLE001 - reads as "no"
            return False

    def run(self, task: str, history: Sequence[Message] | None = None) -> AgentResult:
        """Run to a result, telling the channel that a run is live throughout.

        The channel's reply to the operator promises that the step in progress
        will yield to them. This is the loop that has a step in progress, so
        this is the place that makes the sentence true. The `finally` carries
        as much of it as the entry: a run that raises, or that self-terminates
        through the terminate tool, must not leave the channel promising a
        yield from a loop that is no longer there.
        """
        _mark_run(self.steer, True)
        try:
            return self._run(task, history)
        finally:
            _mark_run(self.steer, False)

    def _run(self, task: str, history: Sequence[Message] | None = None) -> AgentResult:
        msgs = list(history or [])
        if not msgs or msgs[0].role != "system":
            msgs.insert(0, Message.system(self.system_prompt))
        # Built once, from the words as given, and kept on the message: the
        # placeholder is expanded for the model's *text* either way, but this is
        # the multipart form the client is handed, so a line with a picture in it
        # is read rather than described.
        msgs.append(Message.user(task, images=image_parts(task)))

        used: list[str] = []
        turn = 0
        self._terminated = None

        while True:
            turn += 1
            if self.max_turns is not None and turn > self.max_turns:
                return AgentResult(
                    f"(turn cap of {self.max_turns} reached before completion)",
                    msgs, turn - 1, used,
                )
            # Before the model is asked: the cheapest place to find out the
            # task just changed.
            if self._absorb(msgs):
                return self._stopped(msgs, turn - 1, used)
            # And if it changed, they get an answer before the work resumes --
            # a line absorbed into the context is not a line the person has
            # heard back about.
            self._answer_the_operator(msgs)

            # After `_absorb`, so anything the operator just typed is part of
            # the list when the cut point is chosen — and therefore protected by
            # the rule that their last line is never summarized. Compacting
            # first would let a brand-new instruction land in the dropped range.
            #
            # It also has to be here rather than mid-tool: this is the only
            # point in the loop where the list is guaranteed to hold whole
            # groups (an assistant message and all the results answering it),
            # which is what a cut needs to be safe.
            if self.compactor is not None:
                event = self.compactor.maybe_compact(msgs, turn)
                # Only when something happened. A no-op is most turns, and a
                # reporter that fires with `None` trains its reader to ignore it.
                if event is not None:
                    _notify(self.on_compact, event)

            # Where the loop is, on disk, before the step that might not come
            # back. The turn boundary is the only safe place: the message list
            # here is guaranteed to hold whole groups -- an assistant message
            # and every result answering it -- which is what makes the recorded
            # transcript something that can be *replayed* rather than a request
            # that would be malformed. Recording it mid-tool would capture a
            # call whose result never arrived, and continuing from that is
            # asking the model to carry on from a step nobody knows the outcome
            # of.
            #
            self._beat(msgs, turn, self._last_tool)

            _notify(self.on_request, turn)
            try:
                _req_msgs, _req_tools = self._request(msgs)
                resp = self.llm.chat(
                    _req_msgs, tools=_req_tools,
                    # The cheap step: a question is reason enough to go round
                    # again and answer it, because nothing is lost by asking a
                    # request twice. Long steps ask the narrower question.
                    should_abort=self._operator_should_yield,
                )
            except LLMAborted:
                # They spoke while the answer was in flight. Nothing has been
                # decided yet -- no tool ran, no message was appended -- so the
                # only correct move is to go round again, where `_absorb` folds
                # their line into the list and the same question is asked, now
                # informed. The turn is given back: a turn that produced no
                # answer is not a turn spent, and counting it would let an
                # operator's own corrections race the turn cap.
                # Unless the abort was not this loop's to answer. A nested loop
                # -- a verification probe agent, an adversary's attacker -- runs
                # on the same client as the run that spawned it, so it inherits
                # that run's abort predicate and can be woken by a line
                # addressed to somebody else. Going round again here would spin:
                # this loop's own channel is empty, `_absorb` has nothing to
                # fold in, and the inherited predicate is still true, so the
                # next call aborts immediately and forever. Only a line this
                # loop can actually absorb is worth another turn; anything else
                # belongs to the caller, which is where the operator's message
                # is waiting.
                # Asked with the same predicate the call used. Asking the
                # narrower one here would re-raise our own question: the call
                # aborted because there was a line to answer, and the narrow
                # question ("did they send /stop?") is False, so the line would
                # be thrown away as somebody else's.
                if not self._operator_should_yield():
                    raise
                turn -= 1
                continue
            msgs.append(Message.assistant(resp.content, resp.tool_calls))
            _notify(self.on_turn, turn, msgs[-1])

            if not resp.tool_calls:
                return AgentResult(resp.content, msgs, turn, used)

            # The second moment worth recording, and the more important one: the
            # transcript now ends in an assistant message asking for tools whose
            # results do not exist yet, which is exactly the shape a resume has
            # to be able to recognise. Written *before* the tools run, so that a
            # process that dies inside one of them leaves a record naming the
            # step in flight -- and a reader can then trim it back to the last
            # whole group and know the step must be redone. Recorded after, this
            # beat would never happen at all in the case it exists for.
            self._last_tool = ", ".join(tc.name for tc in resp.tool_calls)
            self._beat(msgs, turn, self._last_tool)

            for tc in resp.tool_calls:
                used.append(tc.name)
                _notify(self.on_tool_call, tc.name, tc.arguments)
                try:
                    result = self.registry.call(tc.name, tc.arguments)
                except _TerminateSignal as sig:
                    self._terminated = sig
                    return AgentResult(
                        sig.summary, msgs, turn, used,
                        self_terminated=True, termination_reason=sig.reason,
                    )
                _notify(self.on_tool_result, tc.name, result)
                msgs.append(Message.tool(self._bounded(result.output, tc.name),
                                         tc.id, tc.name))
                # A note typed while that tool ran lands here, before the model
                # is asked again — a forge can take minutes, and waiting until
                # the task ended would make the correction useless.
                #
                # No reply turn here: this is inside the loop over the calls the
                # model just asked for, and an assistant message inserted before
                # those calls have their results makes the next request
                # malformed. `_absorb` accumulates the count instead, and the top
                # of the next turn -- a boundary where the list holds whole
                # groups -- is where the answer is asked for.
                if self._absorb(msgs):
                    return self._stopped(msgs, turn, used)


__all__ = ["Agent", "AgentResult", "DEFAULT_SYSTEM", "_TerminateSignal"]