"""Code generation for new tools.

Two generators, one interface:

* `LLMToolGenerator` — asks a model for a single self-contained Python
  function plus its schema and trigger probes. The model is told to emit a
  *strict JSON envelope*, and we parse defensively: models wrap JSON in
  prose, fences, or trailing commentary often enough that a strict
  `json.loads` is a bug, not a test. The envelope itself also arrives
  near-miss — unescaped quotes inside the `code` field, trailing commas, and
  (the defect that broke the first live runs on *both* aiping and a local
  qwen2.5:7b) a stray `}` that closes the object after `code` while the model
  then keeps writing `entry`/`parameters`/`probes`. See `_repair_variants`.

* `TemplateGenerator` — deterministic, offline, no API key. Used by the demo
  and by tests to prove the pipeline end-to-end without network.

Design note: the generator proposes, it does not decide. Nothing generated
here is trusted — the pipeline is what grants state.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.llm import DEFAULT_MAX_TOKENS, LLMClient
from ..core.message import Message
from .sandbox import Sandbox
from ..tools.spec import ToolSpec, ToolState, TriggerProbe, normalise_parameters

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)
_BRACKET_RE = re.compile(r"\[.*\]", re.DOTALL)


class UnrecoverableGeneration(RuntimeError):
    """The model cannot answer under this configuration, so a retry is futile.

    Deliberately *not* a ValueError. A malformed or truncated envelope is worth
    another round with sharper feedback -- that is what the round budget is for.
    A reasoning model that spends the whole `max_tokens` budget thinking and
    never begins its answer will do the identical thing next round, so the
    distinction has to be visible to whoever decides whether to loop.
    """


@dataclass
class GeneratedTool:
    name: str
    description: str
    code: str
    parameters: dict[str, Any] = field(default_factory=dict)
    entry: str = ""
    probes: list[TriggerProbe] = field(default_factory=list)
    # The valid call, and what a right answer looks like. Required by the
    # prompt: without it the verifier has no positive case and can only ask
    # "does it run", which a tool that refuses everything answers perfectly.
    sample_call: dict[str, Any] = field(default_factory=dict)
    sample_expect: str = ""
    effect_signature: str = ""
    tags: list[str] = field(default_factory=list)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.entry:
            self.entry = self.name
        self.parameters = normalise_parameters(self.parameters)


# JSON permits only these escapes after a backslash. Small models frequently
# emit ``\'`` (borrowed from Python/shell) which is a hard parse error.
_VALID_ESCAPE_RE = re.compile(r"\\(?![\\/\"bfnrtu])")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _next_significant(text: str, j: int) -> tuple[str, int]:
    """The next non-whitespace character at or after ``j``, and its index."""
    n = len(text)
    while j < n and text[j] in " \t\r\n":
        j += 1
    return (text[j] if j < n else ""), j


def _key_follows(text: str, j: int) -> bool:
    """At ``j`` (a ``"``), does a ``"key":`` pair start here?"""
    n = len(text)
    k = j + 1
    while k < n:
        if text[k] == "\\":
            k += 2
            continue
        if text[k] == '"':
            nxt, _ = _next_significant(text, k + 1)
            return nxt == ":"
        if text[k] == "\n":
            return False
        k += 1
    return False


def _ends_string_here(text: str, i: int, is_key: bool, stack: list[str]) -> bool:
    """Can the ``"`` at ``i`` close the string, given what follows it?

    The decision is structural, not textual. ``:`` may only close a *key*, so a
    ``"`` before a colon can never end a value -- which is exactly the case that
    broke: ``code`` holding ``{"isbn13": parts}``. A ``,`` may close a value,
    but only if what follows can still be JSON: another ``"key":`` in an object,
    or anything in an array.
    """
    nxt, j = _next_significant(text, i + 1)
    if nxt == "":
        return True                          # end of input closes the string
    if is_key:
        return nxt == ":"                    # only a key is followed by a colon
    if nxt == ",":
        if stack and stack[-1] == "[":
            return True                      # next array element
        after, k = _next_significant(text, j + 1)
        return after == '"' and _key_follows(text, k)
    if nxt == "}":
        return bool(stack) and stack[-1] == "{"
    if nxt == "]":
        return bool(stack) and stack[-1] == "["
    return False


def _fix_unescaped_quotes(text: str) -> str:
    """Escape double quotes that appear *inside* a JSON string value.

    Observed defect: the model embeds Python source in the ``code`` field and
    writes ``raise ValueError("Invalid check digit")`` -- or a dict literal like
    ``{"isbn13": parts}`` -- with the inner quotes unescaped. The JSON string
    then terminates early and the parser dies with "Expecting ',' delimiter".

    A purely textual rule cannot separate those inner quotes from the structural
    ones, because the same characters introduce them: in ``{"a": 1}`` the quote
    after the key is followed by ``:``, which looks exactly like a real key
    ending. What distinguishes them is *position* -- a value string can never be
    closed by ``:`` -- so the scanner tracks whether it is inside a key or a
    value, and which container it is in (see `_ends_string_here`).
    """
    out: list[str] = []
    stack: list[str] = []                    # '{' / '[' for each open container
    expect_key = False                       # a key may start here
    in_string = False
    is_key = False                           # the current string is a key
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if not in_string:
            if ch == '"':
                in_string = True
                is_key = expect_key
            elif ch in "{[":
                stack.append(ch)
                expect_key = ch == "{"
            elif ch in "}]":
                if stack:
                    stack.pop()
                expect_key = bool(stack) and stack[-1] == "{"
            elif ch == ",":
                expect_key = bool(stack) and stack[-1] == "{"
            elif ch == ":":
                expect_key = False
            out.append(ch)
            i += 1
            continue

        # -- inside a string -------------------------------------------
        if ch == "\\":                       # already-escaped pair: keep as-is
            out.append(ch)
            if i + 1 < n:
                out.append(text[i + 1])
            i += 2
            continue

        if ch == '"':
            if _ends_string_here(text, i, is_key, stack):
                out.append('"')              # genuine terminator
                in_string = False
                expect_key = bool(stack) and stack[-1] == "{"
            else:
                out.append('\\"')            # stray quote inside content
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def _drop_stray_closers(text: str) -> str:
    """Remove ``}``/``]`` the model wrote before the envelope was finished.

    The captured failure (Qwen3.5-Flash via aiping; qwen2.5:7b locally emits the
    same shape) closes the object right after ``code`` and then keeps going::

        { "name": ..., "description": ..., "code": "..." },   <- stray `}`
          "entry": ..., "parameters": {...}, "probes": [...] }

    ``json.loads`` reports ``Extra data: line 5 column 4``; ``raw_decode`` stops
    at the stray brace and hands back only three keys, silently losing
    ``entry``/``parameters``/``probes`` -- a worse outcome than an error, since
    the pipeline would accept a tool with no probes. Deleting the stray brace
    leaves ``"...code..."\\n,\\n"entry"``; whitespace before a comma is legal
    JSON, so the entire envelope survives intact.

    The rule is positional rather than textual. The envelope spans the first
    delimiter to the last, so everything between them must stay nested at depth
    >= 1; a closer that would drop below that is stray. Note a single stray
    brace shifts the depth of *every* closer after it, which is why "count
    matched pairs" and "keep the last depth-zero closer" both misread this text
    -- the untouched interior is what makes the detection unambiguous.
    """
    start, end = 0, len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end - start < 2 or text[start] not in "{[" or text[end - 1] not in "}]":
        return text

    kept: list[str] = []
    depth = 1                      # inside the outermost container
    in_string = False
    escaped = False
    for ch in text[start + 1:end - 1]:
        if escaped:
            escaped = False
        elif in_string:
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            if depth == 1:
                continue           # would close the envelope early -- drop it
            depth -= 1
        kept.append(ch)

    if depth != 1:                 # unclosed delimiters: a different defect
        return text
    return text[start] + "".join(kept) + text[end - 1]


def _repair_variants(text: str) -> list[str]:
    """Progressively repaired variants of near-miss JSON, cheapest first."""
    variants = [text]

    # 0. A stray closer that ended the envelope early. Cheap, structural, and
    #    the one that actually bit us, so it goes first.
    unwrapped = _drop_stray_closers(text)
    if unwrapped != text:
        variants.append(unwrapped)

    # 1. Escape stray double quotes inside string values (the ``code`` field
    #    blowing up on Python string literals).
    quoted = _fix_unescaped_quotes(text)
    if quoted != text:
        variants.append(quoted)

    # 2. Drop invalid escape sequences (``\'`` -> ``'``). JSON only permits
    #    " \ / b f n r t u after a backslash; models borrow Python's ``\'``.
    for variant in list(variants):
        unescaped = _VALID_ESCAPE_RE.sub("", variant)
        if unescaped != variant:
            variants.append(unescaped)

    # 3. Remove trailing commas before a closing brace/bracket.
    for variant in list(variants):
        squeezed = _TRAILING_COMMA_RE.sub(r"\1", variant)
        if squeezed != variant:
            variants.append(squeezed)

    return variants


def _loads_repairing(text: str) -> Any | None:
    """Parse JSON, tolerating the defects small models actually produce.

    Tries each repair variant twice: once strictly, once with ``strict=False``
    so literal control characters (raw newlines inside a ``code`` string) are
    accepted instead of aborting the parse.
    """
    for variant in _repair_variants(text):
        for strict in (True, False):
            try:
                data = json.loads(variant, strict=strict)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(data, (dict, list)):
                return data
    return None


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of model output that may be wrapped in prose."""
    data = extract_json_any(text)
    return data if isinstance(data, dict) else None


def extract_json_any(text: str) -> Any | None:
    """Like `extract_json`, but also returns JSON arrays.

    Strategy: try the raw text first (fast path when the LLM output is already
    clean JSON). Fall back to fence extraction, brace/bracket extraction, then
    *repair* of near-miss JSON. Small models routinely emit envelopes that are
    structurally right but fail strict parsing -- see `_loads_repairing`.
    """
    if not text:
        return None

    # Fast path: the output IS JSON already
    data = _loads_repairing(text)
    if data is not None:
        return data

    # Heuristic: extract content from ``` fences (most reliable)
    candidates: list[str] = []
    candidates.extend(m.group(1).strip() for m in _FENCE_RE.finditer(text) if m.group(1).strip())

    # Heuristic: find the outermost brace or bracket construct.
    # DOTALL greedy `.*` between delimiters is intentionally wrong for nested
    # structures (it over-matches), but it's good enough for our heuristic:
    # we only try it when the fast path failed (meaning the text is not clean
    # JSON), and we try each candidate separately.
    brace = _BRACE_RE.search(text)
    bracket = _BRACKET_RE.search(text)

    if brace and bracket:
        # Prefer the construct that starts first in the text.
        candidates.append((brace if brace.start() <= bracket.start() else bracket).group(0))
    elif brace:
        candidates.append(brace.group(0))
    elif bracket:
        candidates.append(bracket.group(0))

    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        data = _loads_repairing(cand)
        if data is not None:
            return data
    return None


GENERATOR_SYSTEM = """You are a tool smith inside an agent framework.

You write ONE self-contained Python function that solves a recurring subtask,
plus the metadata that lets the framework verify it.

Return STRICT JSON only, no prose, with this shape:
{
  "name": "snake_case_tool_name",
  "description": "one sentence, imperative: what it does and when to use it",
  "code": "def tool_name(...):\\n    ...",
  "entry": "tool_name",
  "parameters": {"type": "object", "properties": {...}, "required": [...]},
  "probes": [
    {"query": "a user request that SHOULD trigger this tool",
     "expect": "call",
     "negative_query": "a request that should NOT trigger it"}
  ],
  "sample_call": {"argument_name": "a REAL, valid value this tool is meant to handle"},
  "sample_expect": "a substring that a correct answer to sample_call must contain",
  "effect_signature": "pure | reads:<what> | writes:<what>",
  "tags": ["category"],
  "rationale": "why this tool is worth creating"
}

Rules:
- The function must be pure Python stdlib unless the description says otherwise.
- No imports outside the standard library. No file writes unless effect_signature says so.
- SAMPLE_CALL IS MANDATORY AND IT IS THE EXAM. Give one concrete, real,
  valid input -- the kind of value this tool exists to handle -- and the
  substring a correct answer must contain. The verifier runs it FIRST and
  rejects the tool if it answers "INVALID:", returns nothing, or does not
  contain sample_expect. Do not pick a value your tool would reject to look
  safe: a valid example that returns INVALID is an automatic failure, and an
  empty or absent sample_call means the tool is recorded as UNPROBED, which is
  worse than a failed probe. For a network tool the example is a real URL and
  a real query; for a parser, a real well-formed string; for arithmetic, real
  numbers and the expected result. If you cannot state a valid input and its
  answer, you do not yet understand the tool well enough to write it.
- DECLARED PARAMETERS ARE CLAIMS. Giving a parameter an `enum` ("enum": [...])
  or making it a boolean declares that the answer depends on it: the verifier
  flips it and requires the output to move. A flag that only affects formatting
  or is accepted and ignored must declare itself -- {"type": "boolean",
  "x-claim": "cosmetic"} -- because a declared rail the tool does not read fails
  the tool. Never declare a rail you will not read, and never leave the only
  input you vary unmarked.
- TOTALITY — the verifier enforces this and rejects the tool if you break it:
  never raise, and never return None. On input you cannot process, return a
  short string starting "INVALID:" with the reason, e.g.
  "INVALID: not a 10- or 13-digit ISBN". On success return the result itself.
  The verifier fuzzes every parameter with empty strings, whitespace, very long
  strings, emoji, digits-only, and None-ish tokens ("NULL", "None", "nan"). Any
  raise fails the tool, so guard every parse and index with a length or
  validity check first.
- Being total is necessary, NOT sufficient. "Returns INVALID for everything"
  satisfies totality and fails sample_call, which is the point: the tool is
  judged on the real input it was written for, not on its handling of junk.
- The invalid result must DIFFER from any valid result, so a wrong-but-total
  function that returns one constant everywhere is also rejected.
- Be whitespace-insensitive where whitespace is not content: strip surrounding
  whitespace before parsing, so "  x  " and "x" behave identically.
- LABELS ARE DECORATION. If the argument is a labelled identifier, a leading
  label is noise, not value: "x", "ISBN x", "ISBN: x", "ISBN-13: x" and the URL
  form must all normalise identically. Strip any such label before parsing. The
  verifier checks exactly this and rejects a tool that only handles the form you
  happened to think of.
- Raise ValueError belongs nowhere; return "INVALID: ..." instead.
- SELF-CONTAINED means every name your entry function calls must be defined in
  the same `code` string. Never call a helper you did not write out — inline it
  into the entry function instead. An undefined name is an execution failure.
- Be terse. No docstrings, no comments, no validation beyond what totality
  needs. Short code is less likely to be cut off mid-JSON.
- Include 2-4 probes. At least one probe must have a negative_query that is
  superficially related but must NOT trigger this tool (guards over-triggering).
- Keep `code` under 40 lines.
"""


@dataclass
class LLMToolGenerator:
    llm: LLMClient
    system_prompt: str = GENERATOR_SYSTEM
    model: str | None = None
    # A tool envelope is ~40 lines of code plus metadata, and the model that
    # writes it is a reasoning one: it produces a trace before it produces the
    # envelope, and that trace is paid for out of this cap first. So the cap has
    # to clear the trace, not the envelope. At 3000 the entire budget went to
    # `reasoning_content` and the reply carried nothing usable -- which is the
    # wall `UnrecoverableGeneration` reports instead of retrying. The default is
    # therefore the client's (`DEFAULT_MAX_TOKENS`), sized above a plausible trace
    # rather than around a non-reasoning answer; providers with a lower ceiling
    # answer 400 and `_clamp_cap_after_rejection` resends smaller, so a generous
    # number is not a trap on the stricter endpoints.
    max_tokens: int = DEFAULT_MAX_TOKENS

    def generate(self, need: str, context: str = "") -> GeneratedTool:
        prompt = f"Recurring need:\n{need}\n"
        if context:
            prompt += f"\nExisting tools (do not duplicate):\n{context}\n"
        prompt += "\nEmit the JSON envelope now."
        resp = self.llm.chat(
            [Message.system(self.system_prompt), Message.user(prompt)],
            tools=None,
            max_tokens=self.max_tokens,
        )
        data = extract_json(resp.content)
        if data is None:
            # A reasoning model that spent the entire budget thinking has not
            # written anything for "be more terse" to shorten. Another round
            # spends the same tokens to observe the same wall, so this is raised
            # as its own type and the pipeline stops rather than retrying.
            if resp.ran_out_of_budget_thinking:
                raise UnrecoverableGeneration(
                    f"generator produced no answer: {resp.describe_shortfall()} "
                    f"[model={self.llm.name}]"
                )
            finish = resp.finish_reason
            # This message is echoed back to the model as retry feedback, so the
            # advice is addressed to the model, not to the operator. A 'stop'
            # finish with unparseable content is a *different* defect from a
            # truncated one, and it used to get no advice at all: the retry then
            # repeated the same escaping mistake, so all three rounds failed the
            # same way. Name the actual defect instead of staying silent.
            hint = (
                " -- the answer was cut off mid-JSON; emit a shorter, denser"
                " function with every helper defined inline"
                if finish == "length" else
                " -- the answer was complete but not valid JSON; escape every"
                " inner double quote in `code` as \\\" and every newline as \\n,"
                " and emit exactly one object with nothing after it"
            )
            raise ValueError(
                f"generator returned no parseable JSON "
                f"(finish_reason={finish!r}, {len(resp.content)} chars){hint}: "
                f"{resp.content[:400]!r}"
            )
        if not isinstance(data, dict):
            raise ValueError(
                f"generator returned JSON of type {type(data).__name__}, "
                f"expected an object: {str(data)[:200]!r}"
            )
        # Models sometimes emit probes as bare strings ({"probes": ["..."]}) or
        # as objects with different key names. Coerce defensively instead of
        # crashing a whole round on a cosmetic schema slip.
        raw_probes = data.get("probes") or []
        if not isinstance(raw_probes, list):
            raw_probes = [raw_probes]
        probes: list[TriggerProbe] = []
        for p in raw_probes:
            if isinstance(p, str):
                probes.append(TriggerProbe(query=p.strip(), expect="call"))
                continue
            if not isinstance(p, dict):
                continue
            query = p.get("query") or p.get("q") or p.get("user_query") or ""
            if not isinstance(query, str) or not query.strip():
                continue
            neg = p.get("negative_query") or p.get("negative") or p.get("should_not")
            probes.append(TriggerProbe(
                query=query.strip(),
                expect=p.get("expect", "call"),
                negative_query=neg if isinstance(neg, str) else None,
            ))
        def _s(key: str, default: str = "") -> str:
            """Coerce a possibly-non-string field to a stripped string."""
            v = data.get(key, default)
            return v.strip() if isinstance(v, str) else (default if v is None else str(v))

        name = _s("name")
        if not name:
            raise ValueError(f"generator omitted tool name: {str(data)[:200]!r}")
        code = _s("code")
        if not code:
            raise ValueError(f"generator omitted code for {name!r}")
        tags = data.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        params = data.get("parameters")
        if not isinstance(params, dict):
            params = {}
        # Shape normalisation (bare type names, properties-as-string) is owned by
        # GeneratedTool.__post_init__ -- one choke point, every caller.

        # The positive case, which the prompt calls "THE EXAM".
        #
        # Both fields were DROPPED here until 2026-10-24: the prompt demanded
        # them, `GeneratedTool` declared them, `_to_spec` forwarded them and
        # `check_execution` read them -- and this constructor never passed them
        # through. Every tool on the ledger therefore verified with
        # "NO VALID SAMPLE CALL was declared, so correctness is unprobed", so
        # the whole battery could only ask "does it run", which a tool that
        # returns one constant answers perfectly.
        #
        # `sample_call` is coerced rather than trusted: models emit it as an
        # object, but also as a JSON string, and occasionally as a single
        # scalar when the tool takes one argument. A wrong shape must not cost
        # the round -- it degrades to "unprobed", which the verifier already
        # reports honestly.
        raw_sample = data.get("sample_call")
        sample_call: dict[str, Any] = {}
        if isinstance(raw_sample, dict):
            sample_call = {str(k): v for k, v in raw_sample.items()}
        else:
            # A string is ambiguous: it is either the object serialised as JSON
            # ("{\"text\": \"banana\"}") or a bare scalar for a one-argument
            # tool ("banana"). Try the object reading first and fall through to
            # the scalar one, rather than treating "not a dict" as "nothing" --
            # which is what silently dropped the exam for the scalar shape.
            parsed = None
            if isinstance(raw_sample, str) and raw_sample.strip():
                parsed = extract_json_any(raw_sample)
            if isinstance(parsed, dict):
                sample_call = {str(k): v for k, v in parsed.items()}
            elif raw_sample is not None and not isinstance(raw_sample, (list, bool)):
                scalar = parsed if parsed is not None else raw_sample
                # Name it after the tool's only required parameter, so a bare
                # value still becomes a callable example instead of a drop.
                required = params.get("required") if isinstance(params, dict) else None
                if isinstance(required, list) and len(required) == 1:
                    sample_call = {str(required[0]): scalar}

        return GeneratedTool(
            name=name,
            description=_s("description"),
            code=code,
            parameters=params,
            entry=_s("entry") or name,
            probes=probes,
            sample_call=sample_call,
            sample_expect=_s("sample_expect"),
            effect_signature=_s("effect_signature", "pure") or "pure",
            tags=[str(t) for t in tags] if isinstance(tags, list) else [],
            rationale=_s("rationale"),
        )


@dataclass
class TemplateGenerator:
    """Deterministic offline generator for demos and tests.

    `recipes` maps a substring of the need to a full GeneratedTool. If nothing
    matches, `fallback` (if given) is used, else a trivial echo tool is built.
    """

    recipes: dict[str, GeneratedTool] = field(default_factory=dict)
    fallback: Callable[[str], GeneratedTool] | None = None
    name: str = "template"

    def generate(self, need: str, context: str = "") -> GeneratedTool:
        low = need.lower()
        for key, tool in self.recipes.items():
            if key.lower() in low:
                return tool
        if self.fallback is not None:
            return self.fallback(need)
        return _echo_tool(need)


def _echo_tool(need: str) -> GeneratedTool:
    safe = re.sub(r"\W+", "_", need.strip().lower())[:30] or "echo"
    name = f"echo_{safe}".strip("_")
    code = (
        f"def {name}(text: str = '') -> str:\n"
        f'    """Echo back the given text (fallback tool)."""\n'
        f"    return text\n"
    )
    return GeneratedTool(
        name=name,
        description="Echo back the given text.",
        code=code,
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "text to echo"}},
            "required": ["text"],
        },
        entry=name,
        probes=[TriggerProbe(query=f"echo the word {safe}", expect="call")],
        tags=["fallback"],
    )


__all__ = [
    "GeneratedTool",
    "LLMToolGenerator",
    "TemplateGenerator",
    "extract_json",
    "extract_json_any",
    "GENERATOR_SYSTEM",
    "Sandbox",
]
