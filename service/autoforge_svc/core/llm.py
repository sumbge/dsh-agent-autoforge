"""LLM clients.

`LLMClient` is a one-method protocol so swapping providers never touches the
agent loop. Two implementations ship in-tree:

* `OpenAICompatClient` — any `/chat/completions` endpoint (OpenAI, DeepSeek,
  Moonshot, DashScope, aiping, vLLM, Ollama, a local gateway...).
* `MockLLMClient` — scripted or callback-driven, used by tests and by the
  demo so the framework is provably runnable with zero API keys.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import requests

from .message import Message, ToolCall

#: How often a watched call hands control back to its caller. Small enough that
#: a typed correction is acted on well inside the operator's patience, large
#: enough that the polling itself is invisible.
_POLL_SECONDS = 0.25


def _sleep(seconds: float) -> None:
    """Indirection so tests can make retry backoff instant."""
    time.sleep(seconds)


def _as_int(value: Any) -> int:
    """`value` as an int, with anything unparseable reading as zero.

    Usage is a report from the provider about work already done, so a gateway
    that sends `"usage": {"prompt_tokens": "1234"}` or sends no usage at all
    must not raise here: a failed parse would lose the reply to report a number.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def new_usage_totals() -> dict[str, int]:
    """A fresh accumulator for one client's token accounting."""
    return {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0,
            "completion_tokens": 0}


def account_usage(totals: dict[str, int], resp: LLMResponse) -> dict[str, int]:
    """Fold one response's usage into `totals`, and return the totals.

    The provider only reports per call, so the cache-hit share of a whole
    session exists nowhere unless something adds it up. This is that something.
    """
    totals["calls"] += 1
    totals["prompt_tokens"] += resp.prompt_tokens
    totals["cached_tokens"] += resp.cached_tokens
    totals["completion_tokens"] += resp.completion_tokens
    return totals


class LLMError(RuntimeError):
    """One name for "this endpoint did not work", whatever the transport said.

    ``requests`` has six exception types that all mean the same thing to a
    caller: connection refused, timeout, proxy failure, TLS failure, truncated
    body. Catching them by name at every call site is how one gets forgotten,
    and a forgotten one turns a down endpoint into an unhandled traceback in
    the middle of a run. Failing over needs one thing to catch; so does the
    message "the gateway is down, and here is what it said".
    """


class LLMResponseError(LLMError):
    """The endpoint answered 2xx with a body that is not a completion.

    Raised instead of letting the malformation surface later as
    ``AttributeError: 'str' object has no attribute 'get'`` -- a message that
    names neither the endpoint nor the body, and lands three frames away from
    the reply that caused it. Retrying cannot help: the payload and the
    credentials were accepted, and what came back is not the promised shape.
    """


class LLMAborted(InterruptedError):
    """A call in flight was abandoned because the operator spoke.

    Not an error to report: the request was well-formed and would have
    answered. It means the run's instructions just changed, so the answer on
    its way is no longer the answer being asked for. The caller is expected to
    fold the operator's line into the conversation and ask again -- which is
    why this carries no partial content.
    """


def _any_predicate(
    own: Callable[[], bool] | None,
    attached: Callable[[], bool] | None,
) -> Callable[[], bool] | None:
    """Combine two abort predicates: either one saying yes is a yes.

    `own` is what the caller passed to this `chat`; `attached` is what the
    owner of the run hung on the client (`LLMClient.abort_check`).

    They are combined rather than one winning, because the two answer
    different questions and both are binding. An inner loop -- a verification
    probe agent, say -- asks "has *my* channel got something for me?", and the
    honest answer there is no. The run that spawned it asks "has the operator
    said anything to anyone?", and during a forge that answer is yes. If the
    inner predicate simply won, the probe would sit through the operator's
    message with the client-level predicate switched off, which is the exact
    two-minute block the attached predicate exists to close.

    `None` means "nobody is watching", not "do not watch": it drops out, it
    does not override.
    """
    live = [p for p in (own, attached) if p is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]
    return lambda: any(p() for p in live)


def _post_watchable(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    proxies: dict[str, str] | None,
    should_abort: Callable[[], bool] | None,
    on_wait: Callable[[float], None] | None,
) -> Any:
    """POST, but hand control back to the caller on every tick.

    A plain ``requests.post`` parks the calling thread for the whole read
    timeout -- two minutes here -- and nothing can happen inside that window:
    not a ``/stop``, not a correction the operator just typed. The agent's
    promise of "talk to me while I work" then only holds *between* calls, which
    is precisely the gap the person at the terminal experiences as being
    ignored. That gap is the bug this function exists to close.

    So when a watcher is supplied the request moves to a worker thread and the
    caller polls. ``should_abort`` is consulted every tick; the moment it goes
    true the request is abandoned and ``LLMAborted`` is raised. ``on_wait`` is
    fed the elapsed seconds so a progress line can be kept honest.

    Abandoning is safe here in a way it would not be for an arbitrary request:
    a completion has no side effect to undo. The POST does stay in flight until
    it finishes on its own, and the worker is a daemon so a process exit never
    blocks on it.
    """
    if should_abort is None and on_wait is None:
        return requests.post(endpoint, headers=headers, json=payload,
                             timeout=timeout, proxies=proxies)

    done = threading.Event()
    box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["resp"] = requests.post(endpoint, headers=headers, json=payload,
                                        timeout=timeout, proxies=proxies)
        except BaseException as exc:      # noqa: BLE001 - re-raised on the caller
            box["exc"] = exc
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()
    started = time.time()
    while not done.wait(_POLL_SECONDS):
        if on_wait is not None:
            on_wait(time.time() - started)
        if should_abort is not None and should_abort():
            raise LLMAborted(
                f"abandoned a model call in flight after {time.time() - started:.1f}s "
                "because the operator said something"
            )
    if "exc" in box:
        raise box["exc"]
    return box["resp"]


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] | None = None

    #: The model's private scratchpad, when the provider exposes one
    #: (`reasoning_content` on reasoning models). Not part of the answer, but it
    #: is not noise either: it is where the `max_tokens` budget goes when a
    #: reply comes back empty, so diagnostics that count it can say *why*.
    reasoning: str = ""

    #: The provider's token accounting for this call, verbatim.
    #:
    #: Kept because the interesting number is not the total but the split:
    #: `prompt_tokens_details.cached_tokens` (DeepSeek, OpenAI) or
    #: `prompt_cache_hit_tokens` (some gateways) is what says whether the
    #: conversation is being served from the prefix cache or re-billed from
    #: scratch. That share is invisible from the outside — a cached call and an
    #: uncached one look identical in the reply — so without reading it here
    #: there is no way to tell a cheap session from an expensive one.
    usage: dict[str, Any] = field(default_factory=dict)

    #: Prompt tokens the provider served from its cache.
    @property
    def cached_tokens(self) -> int:
        details = self.usage.get("prompt_tokens_details")
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            return _as_int(details.get("cached_tokens"))
        return _as_int(self.usage.get("prompt_cache_hit_tokens"))

    @property
    def prompt_tokens(self) -> int:
        return _as_int(self.usage.get("prompt_tokens"))

    @property
    def completion_tokens(self) -> int:
        return _as_int(self.usage.get("completion_tokens"))

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def finish_reason(self) -> str | None:
        """The provider's stop reason: 'stop', 'length', 'tool_calls', ...

        Worth a name rather than three copies of the same dict walk, because it
        is the only thing that separates "the model finished" from "the model was
        cut off" — a distinction the error paths live or die by.
        """
        raw = self.raw if isinstance(self.raw, dict) else {}
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        # A gateway that fails out-of-band sometimes answers HTTP 200 with
        # ``choices: ["<something went wrong>"]``. Reading the shape rather than
        # assuming it keeps that failure at the body, where it can be reported,
        # instead of at this property, where all it can be is an AttributeError.
        return first.get("finish_reason") if isinstance(first, dict) else None

    @property
    def ran_out_of_budget_thinking(self) -> bool:
        """True when the cap was spent reasoning and no answer ever started.

        Distinguishes a deterministic wall from a transient hiccup: retrying
        this is guaranteed to burn the same budget again for the same nothing.
        """
        if self.content or self.tool_calls:
            return False
        return bool(self.reasoning) and self.finish_reason == "length"

    def describe_shortfall(self) -> str:
        """One line explaining an unusable reply, for the caller's error path."""
        if self.ran_out_of_budget_thinking:
            return (f"the model spent the whole max_tokens budget on "
                    f"reasoning ({len(self.reasoning)} chars of "
                    f"reasoning_content) and never began the answer; raise "
                    f"max_tokens or pick a model that answers directly")
        return f"finish_reason={self.finish_reason!r}, {len(self.content)} chars of content"


class LLMClient:
    """Minimal protocol: turn messages (+tool schemas) into an LLMResponse."""

    name: str = "llm"

    #: A predicate attached to the *client* rather than to each call.
    #:
    #: Some model calls are not made by the loop the operator is talking to. A
    #: forge round fans out -- the generator, the adversary and its victim, and
    #: two probe agents that run the candidate tool -- and every one of them
    #: calls `chat` on a client that ultimately belongs to the run the operator
    #: is watching. Not one of them knows the operator exists.
    #:
    #: Threading `should_abort` through each of those call sites is a list that
    #: has to be kept complete by hand, and the stage somebody forgets is a
    #: two-minute block with the operator on the other side of it -- which is
    #: what happened: the sandbox's children yielded to them and the model calls
    #: did not. So the owner of the run attaches the predicate here, once, for
    #: the duration of the step (see `ForgePipeline.forge`), and every call
    #: beneath it inherits it. A call that also got one of its own is bound by
    #: both -- see `_any_predicate`.
    abort_check: Callable[[], bool] | None = None

    def chat(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAICompatClient(LLMClient):
    """Any OpenAI-compatible chat completions endpoint."""

    #: Sent when the caller does not pass `max_tokens` of its own.
    #:
    #: Not cosmetic: aiping.cn returns 503 for a completion body with no
    #: `max_tokens` while the identical body carrying a cap returns 200
    #: (12 requests, cleanly separated -- see probes/probe_gateway_max_tokens.py).
    #: Omitting the field is therefore a provider-specific landmine, and the
    #: safe place to defuse it is here, once, rather than at every call site.
    #: Pass `default_max_tokens=None` for providers that reject the field.
    #:
    #: The *size* matters as much as the presence. A reasoning model spends the
    #: budget on its trace before it writes anything, and a cap drawn for a
    #: non-reasoning model is spent entirely on thinking: against aiping.cn, this
    #: generator's prompt at a cap of 3000 came back `finish_reason='length'` with
    #: 0 chars of content and 10901 chars of `reasoning_content` -- an empty
    #: answer that looked like a provider fault. Sizing the cap above what a
    #: trace plausibly consumes fixes it, and 32768 is chosen against a measured
    #: bound rather than a guess: the agent that this repo's author actually runs
    #: sends 128000 to this same gateway, from the same profile, and gets answers.
    DEFAULT_MAX_TOKENS: int | None = 32768

    #: Statuses worth sending again: the request was well-formed and the answer
    #: is "later", not "no". Any other 4xx means the payload or the credentials
    #: are wrong, and retrying it only delays the real message by seconds.
    #:
    #: Not hypothetical: a forge round against aiping.cn died on a 503, and
    #: because a round is the unit of budget, one transient blip cost the whole
    #: tool. Retrying belongs here, one layer under the round, where a hiccup is
    #: invisible instead of expensive.
    RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

    #: The smallest cap worth sending after a provider rejects ours. Below this
    #: the reply cannot hold a tool envelope anyway, so the honest outcome is to
    #: let the provider's own error through instead of hunting for a number.
    MIN_MAX_TOKENS = 512

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        proxies: dict[str, str] | None = None,
        default_temperature: float = 0.0,
        default_max_tokens: int | None = DEFAULT_MAX_TOKENS,
        extra_headers: dict[str, str] | None = None,
        max_attempts: int = 5,
        retry_backoff: float = 2.0,
        retry_max_delay: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.proxies = proxies
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.extra_headers = extra_headers or {}
        # Five attempts at a base of two seconds is roughly 30 seconds of riding
        # out a 503 burst. The gateway's outages are bursty and outlast a short
        # ladder: a real forge died on `503 Service Unavailable` in round two
        # even with the three-attempt, 1.5x default, and a probe reproduced the
        # same 503 twice in a row while every cap returned 200 seconds later.
        self.max_attempts = max(1, max_attempts)
        self.retry_backoff = retry_backoff
        self.retry_max_delay = retry_max_delay
        #: Token accounting for this client's lifetime. `cached_tokens` against
        #: `prompt_tokens` is the one number that says whether the prompt prefix
        #: is stable enough to be served from cache -- which is the difference
        #: between a long conversation costing a tenth of what it looks like and
        #: costing exactly what it looks like.
        self.usage_total = new_usage_totals()
        self.name = f"openai-compat:{model}"

    @property
    def cache_hit_rate(self) -> float:
        """Share of prompt tokens the provider served from its cache, 0..1."""
        prompt = self.usage_total["prompt_tokens"]
        return self.usage_total["cached_tokens"] / prompt if prompt else 0.0

    def _wait_before_retry(self, attempt: int, resp: Any) -> None:
        """Exponential backoff, but never earlier than the server asked for."""
        delay = min(self.retry_backoff ** attempt, self.retry_max_delay)
        if resp is not None:
            asked = (getattr(resp, "headers", None) or {}).get("Retry-After")
            if asked:
                try:
                    delay = max(delay, float(asked))
                except ValueError:
                    pass
        _sleep(delay)

    def _clamp_cap_after_rejection(self, payload: dict, resp: Any) -> int | None:
        """The cap to resend with after a 400 that blames ``max_tokens``.

        Providers disagree about the ceiling, and the disagreement is fatal
        rather than cosmetic: aiping.cn accepts 128000, api.deepseek.com stops at
        8192, a local runtime stops at whatever its context allows. The cap has
        to be generous enough that a reasoning trace cannot eat the whole budget
        before the answer starts (that is why it is not small), and a number
        that generous is a 400 on the stricter providers. Since a whole tool dies
        otherwise, this answers that one 4xx with a smaller number instead of
        rethrowing it.

        Every other 400 returns None: a bad payload or a bad key is not going to
        be fixed by our improving the number.
        """
        if getattr(resp, "status_code", None) != 400:
            return None
        current = payload.get("max_tokens")
        if not isinstance(current, int):
            return None
        body = getattr(resp, "text", "") or ""
        if "max_tokens" not in body:
            return None
        # Providers state the ceiling in prose ("max_tokens is in [1, 8192]",
        # "must be <= 16384"), so the largest number in the message below ours is
        # a better guess than blind halving -- that is the ceiling they are
        # naming. Halving is the fallback for a message that names none.
        named = [int(n) for n in re.findall(r"\d{3,7}", body)]
        below = [n for n in named if n < current]
        if below:
            target = max(below)
            # They named their ceiling and it is too low to hold an envelope.
            # Sending it anyway would trade one clear error for a cryptic one.
            return target if target >= self.MIN_MAX_TOKENS else None
        if named:
            # A number was named, but none of them is below what we sent: the
            # complaint is not about the size, so halving would be guessing.
            return None
        return current // 2 if current // 2 >= self.MIN_MAX_TOKENS else None

    def _to_response(self, data: Any) -> LLMResponse:
        if not isinstance(data, dict):
            raise LLMResponseError(
                f"{self.name} answered with a JSON {type(data).__name__}, "
                f"not an object: {str(data)[:200]!r}"
            )
        choices = data.get("choices")
        if not isinstance(choices, list):
            choices = []
        if choices and not isinstance(choices[0], dict):
            # 200 with choices[0] as a bare string is how an OpenAI-compatible
            # gateway reports a downstream failure. Name it; do not crash on it.
            raise LLMResponseError(
                f"{self.name} returned choices[0] of type "
                f"{type(choices[0]).__name__}, not an object: "
                f"{str(choices[0])[:200]!r}"
            )
        choice = choices[0] if choices else {}
        msg = choice.get("message")
        if not isinstance(msg, dict):
            msg = {}
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=[ToolCall.from_api(tc) for tc in (msg.get("tool_calls") or [])],
            raw=data,
            reasoning=msg.get("reasoning_content") or "",
            usage=data.get("usage") if isinstance(data.get("usage"), dict) else {},
        )

    def chat(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # Popped before the payload is built, or they would be sent to the
        # endpoint as unknown body fields -- a 400 at best, silently ignored at
        # worst. They are addressed to *this* client, not to the provider.
        #
        # `should_abort` is merged with whatever the run attached to the client
        # rather than replacing it: a stage that asked its own channel and got
        # silence must still yield to the operator talking to the run above it.
        should_abort = _any_predicate(
            kwargs.pop("should_abort", None), self.abort_check,
        )
        on_wait = kwargs.pop("on_wait", None)

        # `m.to_api()` with no argument: a message that carries a picture answers
        # with parts by itself, and the client stays ignorant of the fact that
        # some messages are not only text. Passing the client in here as an
        # override is what turned it into a message's content once already, and
        # the failure that produced was a TypeError two layers away from the line.
        payload: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": [m.to_api() for m in messages],
            "temperature": kwargs.pop("temperature", self.default_temperature),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.pop("tool_choice", "auto")
        if self.default_max_tokens is not None and "max_tokens" not in kwargs:
            payload["max_tokens"] = self.default_max_tokens
        payload.update(kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }
        endpoint = f"{self.base_url}/chat/completions"

        # `allowed`, not `self.max_attempts`, is what the loop counts to: one
        # extra attempt is granted when the provider objects to the cap itself,
        # because that attempt carries a *different* payload rather than
        # repeating one already known to fail.
        allowed = self.max_attempts
        cap_clamped = False
        attempt = 0
        while attempt < allowed:
            attempt += 1
            final = attempt >= allowed
            try:
                resp = _post_watchable(
                    endpoint, headers, payload, self.timeout, self.proxies,
                    should_abort, on_wait,
                )
            except LLMAborted:
                # Not a transport failure, so it must not be retried: retrying
                # would re-ask the question the operator just changed. Let it
                # out to the loop, which folds their line in and asks again.
                raise
            except requests.RequestException as exc:
                if final:
                    # Transport failures are translated, not re-raised as
                    # themselves, and this is the reason: `requests` raises
                    # ConnectionError, Timeout, ProxyError, SSLError and
                    # ChunkedEncodingError, and a caller that wants to fail over
                    # or to report "the endpoint is down" would have to catch
                    # every one of them by name. Miss one and a dead endpoint
                    # looks like a crash in the loop.
                    if isinstance(exc, LLMError):
                        raise
                    raise LLMError(
                        f"{self.name}: {type(exc).__name__}: {exc}"
                    ) from exc
                self._wait_before_retry(attempt, None)
                continue

            if resp.status_code in self.RETRY_STATUSES and not final:
                self._wait_before_retry(attempt, resp)
                continue

            # A 400 that blames the cap is the one 4xx here that gets a second
            # look, because the request is fine and only its number is wrong.
            # The replacement attempt must not come out of the retry budget:
            # nothing has been learned about the endpoint yet, only about a
            # ceiling that is lower than the one we assumed.
            if not cap_clamped:
                clamped = self._clamp_cap_after_rejection(payload, resp)
                if clamped is not None:
                    payload["max_tokens"] = clamped
                    cap_clamped = True
                    allowed += 1
                    continue

            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                # The one that got away. A 503 that survives the whole retry
                # ladder leaves the loop through `raise_for_status`, NOT through
                # the transport handler above -- so translating only the transport
                # path meant an exhausted-retry 503 escaped as a raw HTTPError and
                # a failover chain above it never fired. Observed live: a forge
                # round died on "forge error: HTTPError: 503" while a healthy
                # fallback endpoint sat unused one line below in the chain.
                raise LLMError(
                    f"{self.name}: HTTP {resp.status_code} after {attempt} "
                    # getattr, not resp.text: this line runs *inside an error
                    # handler*, and a response-like object without `.text` would turn
                    # "HTTP 400 after 2 attempts" into "AttributeError: no attribute
                    # text" -- losing the status the message exists to carry. Real
                    # requests.Response always has it; the crash-on-the-error-path is
                    # the part worth closing. (Tests/test_llm_retry.py and
                    # test_setup.py were failing on exactly this.)
                    f"attempt(s): {(getattr(resp, 'text', '') or '')[:200]!r}"
                ) from exc
            try:
                body = resp.json()
            except ValueError as exc:
                # A proxy or gateway in front of the endpoint can answer 200
                # with an HTML error page. Decoding that is not the caller's
                # problem to guess at, and the decode failure names nothing.
                raise LLMResponseError(
                    f"{self.name} returned a non-JSON body "
                    f"(HTTP {resp.status_code}): {(resp.text or '')[:200]!r}"
                ) from exc
            parsed = self._to_response(body)
            account_usage(self.usage_total, parsed)

            # An empty completion is usually a provider hiccup, and hiccups are
            # worth a second send. One exception: if the whole budget went to a
            # reasoning trace, the wall is deterministic and resending just
            # spends the same tokens for the same empty reply.
            if parsed.content or parsed.tool_calls or final:
                return parsed
            if parsed.ran_out_of_budget_thinking:
                return parsed
            self._wait_before_retry(attempt, resp)

        raise AssertionError("unreachable: the loop returns or raises")


#: The same number, for callers that carry a cap of their own instead of
#: relying on the client default: the forge generator, the CLI, the wizard. One
#: value read from one place, because three independent copies is exactly how the
#: original 3000 outlived the failure that condemned it.
DEFAULT_MAX_TOKENS: int | None = OpenAICompatClient.DEFAULT_MAX_TOKENS

class DSHGatewayClient(LLMClient):
    """The model DSH itself is signed into, reached without a key of its own.

    DSH's desktop profile keeps the live sign-in at `.dsh/.xdpool/accounts.json`
    and routes chat through the xdpool gateway at `/v2/chat/completions`. That
    gateway refuses a non-streaming body outright (code 11101, "Non-stream chat
    request is currently not supported" -- measured), so this client always asks
    for SSE and joins the deltas itself.

    The point of reusing it rather than asking for an API key: a forge round is
    the agent talking to a model, and the operator already pays for one that is
    signed in three inches away. A second key would be a second bill for the
    same work.
    """

    name = "dsh-gateway"

    BASE = "https://copilot.tencent.com/v2/chat/completions"

    def __init__(self, model: str = "deepseek-v4.1-flash", *,
                 accounts_path: str | None = None, timeout: float = 180.0,
                 max_tokens: int = 32768) -> None:
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.accounts_path = accounts_path or os.path.join(
            os.path.expanduser("~"), ".dsh", ".xdpool", "accounts.json")

    def _credential(self) -> dict[str, Any]:
        if not os.path.exists(self.accounts_path):
            raise LLMError(f"no DSH sign-in at {self.accounts_path}")
        data = json.loads(open(self.accounts_path, encoding="utf-8").read())
        accounts = data.get("accounts") or []
        if not accounts:
            raise LLMError("DSH has no signed-in account in accounts.json")
        return accounts[0]["credential"]

    def chat(self, messages: Sequence[Message],
             tools: list[dict[str, Any]] | None = None,
             **kwargs: Any) -> LLMResponse:  # pragma: no cover - network
        cred = self._credential()
        body: dict[str, Any] = {
            "model": kwargs.get("model") or self.model,
            # `to_api()` is the wire form, and using anything else here is how
            # the gateway came back with 11148 "tool calls and tool results do
            # not match": an assistant turn carrying tool calls must be sent
            # with `tool_calls[].function.{name,arguments}`, and a tool result
            # must carry `tool_call_id`. A flattened dict drops both.
            "messages": [_as_dict(m) for m in messages],
            "stream": True,
            "max_tokens": kwargs.get("max_tokens") or self.max_tokens,
        }
        if tools:
            body["tools"] = tools
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {cred['accessToken']}",
            "X-Product": "SaaS",
            "X-IDE-Type": "dsh", "X-IDE-Name": "dsh", "X-IDE-Version": "1.0",
        }
        if not cred.get("uid"):
            headers["X-No-User-Id"] = "1"
        else:
            headers["X-User-Id"] = cred["uid"]
        req = urllib.request.Request(
            self.BASE, data=json.dumps(body).encode("utf-8"),
            headers=headers, method="POST")
        content, reasoning, calls, usage = "", "", [], {}
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except Exception:                          # noqa: BLE001
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in (chunk.get("choices") or []):
                        delta = choice.get("delta") or {}
                        content += delta.get("content") or ""
                        reasoning += delta.get("reasoning_content") or ""
                        # Tool calls arrive in *fragments*: one indexed slot
                        # per call, and the name and the JSON arguments are
                        # split across chunks. Appending each fragment as if it
                        # were a whole call produced a dozen nameless calls with
                        # `{"_raw": "{"}` for arguments, and the round trip then
                        # died on the gateway's parameter check. Accumulate by
                        # index, then emit the settled calls once the stream ends.
                        for frag in (delta.get("tool_calls") or []):
                            slot = frag.get("index", len(calls))
                            while len(calls) <= slot:
                                calls.append({"id": "", "type": "function",
                                              "function": {"name": "", "arguments": ""}})
                            acc = calls[slot]
                            if frag.get("id"):
                                acc["id"] = frag["id"]
                            fn = frag.get("function") or {}
                            if fn.get("name"):
                                acc["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                acc["function"]["arguments"] += fn["arguments"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise LLMError(f"gateway HTTP {exc.code}: {detail}") from exc
        except Exception as exc:                               # noqa: BLE001
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        # The loop downstream reads `tc.name` and `tc.arguments`, so deltas have
        # to become ToolCall objects here. Handing back the raw dicts made the
        # first forge round die in agent.py, three frames from the cause:
        # "AttributeError: 'dict' object has no attribute 'name'".
        parsed: list[Any] = []
        for raw_call in calls:
            try:
                parsed.append(ToolCall.from_api(raw_call))
            except Exception:                                  # noqa: BLE001
                continue
        return LLMResponse(content=content, tool_calls=parsed,
                           raw={"usage": usage}, reasoning=reasoning)


def _as_dict(message: Any) -> dict[str, Any]:
    """A Message, or anything that already looks like one, as a wire dict.

    Prefer the object's own `to_api()`: only it knows that an assistant turn
    carries `tool_calls` and a tool turn carries `tool_call_id`, and losing
    either makes the gateway reject the whole conversation.
    """
    if isinstance(message, dict):
        return message
    to_api = getattr(message, "to_api", None)
    if callable(to_api):
        try:
            return to_api()
        except Exception:                                      # noqa: BLE001
            pass
    role = getattr(message, "role", "user")
    content = getattr(message, "content", "")
    return {"role": str(role), "content": "" if content is None else str(content)}


class FailoverClient(LLMClient):
    """Several endpoints behind one client, tried in the order they still work.

    The problem this exists for is not hypothetical and not fixable at the
    provider: a gateway that answers 200 six times in a row can answer 503 for
    three minutes, and a forge round is the unit of budget -- one blip inside it
    costs the whole tool, whatever the retry ladder underneath does. Retrying
    rides out a short burst; it cannot ride out an outage. A second endpoint can.

    Three decisions worth their reasons:

    *Health is remembered, not recomputed.* An endpoint that failed is skipped
      for `cooldown` seconds instead of being tried first again, so a down
      primary costs one timeout per cooldown rather than one per call. The
      arithmetic is the argument: at three minutes per failed attempt, trying
      the dead endpoint first on every call is the outage.

    *The order is sticky, not round-robin.* Whoever answered last is tried
      first, so a working endpoint stays in use and the prompt prefix stays on
      one provider -- which is what prompt caching needs. Round-robin would
      spread the same conversation across providers and pay full price for the
      prefix every time.

    *`LLMAborted` is never a reason to fail over.* The operator asking to stop
      is a decision, not an outage; re-asking the same question on another
      endpoint is exactly what `should_abort` exists to prevent.

    The failure is reported as one error listing every endpoint and what it
    said, because "all of them are down" and "the one you configured is wrong"
    need different fixes and a bare exception cannot tell them apart.
    """

    def __init__(self, clients: "Sequence[LLMClient]", *,
                 cooldown: float = 120.0, clock: Callable[[], float] = time.monotonic
                 ) -> None:
        if not clients:
            raise ValueError("FailoverClient needs at least one client")
        self.clients = list(clients)
        self.cooldown = float(cooldown)
        self._clock = clock
        #: When each client may be tried again. Parallel to `clients`; a client
        #: that has never failed is open from the start.
        self._open_at = [0.0] * len(self.clients)
        self._abort_check: Callable[[], bool] | None = None
        self.name = "failover(" + " | ".join(c.name for c in self.clients) + ")"

    # -- the run attaches its predicate here; every endpoint must honour it --
    @property
    def abort_check(self) -> Callable[[], bool] | None:
        return self._abort_check

    @abort_check.setter
    def abort_check(self, predicate: Callable[[], bool] | None) -> None:
        self._abort_check = predicate
        for client in self.clients:
            client.abort_check = predicate

    @property
    def usage_total(self) -> dict[str, int]:
        """The chain's own accounting, summed over the endpoints that served it.

        The run reads `getattr(self.llm, "usage_total", {})` to write token
        spend into the ledger. That attribute existed on `OpenAICompatClient`
        and on the mock, but not here -- so from the day a `fallbacks` list put
        a FailoverClient in front of the clients (2026-09-17), every run
        recorded `usage: {}`, and the one number that says whether the prompt
        prefix is worth caching stopped being written down at all. A missing
        attribute that `getattr(..., default)` hides is worse than a missing
        one that raises: the counter read as a real, empty result.
        """
        total = new_usage_totals()
        for client in self.clients:
            u = getattr(client, "usage_total", None) or {}
            for k in total:
                total[k] += int(u.get(k, 0) or 0)
        return total

    @property
    def cache_hit_rate(self) -> float:
        """Aggregate, not the first client's: the number answers "is the prefix
        stable enough to cache", and with a failover chain that is a question
        about all the endpoints that served it."""
        prompt = sum(getattr(c, "usage_total", {}).get("prompt_tokens", 0)
                     for c in self.clients)
        cached = sum(getattr(c, "usage_total", {}).get("cached_tokens", 0)
                     for c in self.clients)
        return cached / prompt if prompt else 0.0

    def order(self) -> list[int]:
        """Which endpoints to try, in order, right now.

        Healthy ones first, in their CONFIGURED order -- the primary is the
        primary for a reason (quality, price, key quota), and a failover chain
        that quietly promotes the backup to primary forever is a different
        system from the one that was configured. Cooling ones after, oldest
        cooldown first.

        This is also what keeps the prompt prefix on one provider: while the
        primary is up it is tried first on every call, so the same endpoint
        serves the whole conversation and its cache stays warm. The alternative
        -- reordering by whoever answered last -- would look clever and would
        mean a recovered primary never comes back.

        If everything is cooling the call still happens, against the endpoint
        closed longest: refusing to call at all would turn a failover chain into
        an outage of its own.
        """
        now = self._clock()
        open_now = [i for i in range(len(self.clients)) if self._open_at[i] <= now]
        cooling = sorted((i for i in range(len(self.clients)) if i not in open_now),
                         key=lambda i: self._open_at[i])
        return open_now + cooling

    def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        errors: list[str] = []
        last: Exception | None = None
        order = self.order()
        for i in order:
            client = self.clients[i]
            try:
                parsed = client.chat(messages, tools=tools, **kwargs)
            except LLMAborted:
                raise
            except LLMError as exc:
                last = exc
                # Every endpoint failure -- refused connection, 503, truncated
                # body, non-JSON page from a gateway -- is a reason to try
                # elsewhere. `LLMError` and not `Exception` on purpose: a
                # TypeError in the payload is a bug in the caller, and hiding it
                # behind "every endpoint failed" would make this class the place
                # where real defects go to look like network weather.
                self._open_at[i] = self._clock() + self.cooldown
                errors.append(f"{client.name}: {type(exc).__name__}: {exc}")
                continue
            # Succeeded: it is healthy again, and the configured order stands.
            self._open_at[i] = 0.0
            return parsed
        # One error type, naming every endpoint and what it said, with the last
        # failure chained as __cause__. Re-raising the last exception as itself
        # would preserve the type and lose the summary -- and the summary is the
        # whole point when the answer to "why did the run stop" is "the primary
        # rejected the key and the backup timed out", which no single endpoint's
        # message says.
        raise LLMError(
            "every endpoint failed:\n  " + "\n  ".join(errors)) from last


class MockLLMClient(LLMClient):
    """Deterministic client for tests and offline demos.

    Two modes:
      * `script=[LLMResponse, ...]` — pop one response per call.
      * `handler=lambda messages, tools, **kw -> LLMResponse` — full control.

    Both record every call in `.calls` for assertions.
    """

    def __init__(
        self,
        script: list[LLMResponse] | None = None,
        handler: Callable[..., LLMResponse] | None = None,
        name: str = "mock",
    ) -> None:
        self.script = list(script or [])
        self.handler = handler
        self.name = name
        self.calls: list[tuple[list[Message], list[dict[str, Any]] | None]] = []
        # Same accounting as the real client, so a test can assert on cache
        # behaviour without a network.
        self.usage_total = new_usage_totals()

    @property
    def cache_hit_rate(self) -> float:
        prompt = self.usage_total["prompt_tokens"]
        return self.usage_total["cached_tokens"] / prompt if prompt else 0.0

    def chat(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append((list(messages), tools))
        if self.handler is not None:
            resp = self.handler(messages, tools, **kwargs)
        elif self.script:
            resp = self.script.pop(0)
        else:
            resp = LLMResponse(content="(mock: script exhausted)")
        account_usage(self.usage_total, resp)
        return resp


def tool_call(name: str, arguments: dict[str, Any] | None = None, call_id: str | None = None) -> ToolCall:
    """Convenience for building scripted responses."""
    return ToolCall(id=call_id or f"call_{name}", name=name, arguments=arguments or {})


__all__ = [
    "LLMClient",
    "LLMAborted",
    "LLMResponse",
    "LLMResponseError",
    "OpenAICompatClient",
    "MockLLMClient",
    "account_usage",
    "new_usage_totals",
    "tool_call",
    "json",
    "DEFAULT_MAX_TOKENS",
]
