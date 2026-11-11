"""Which tool schemas ride in every request, and which are merely listed.

The library is not the prompt. Those became the same thing by accident: every
tool the registry exposes is serialised into every model call, and nothing ever
looked at whether the tool had been used. Measured on this machine 2026-09-28:
204 visible schemas, 106,503 characters, ~33k tokens, paid on every request of
every turn -- of which 61 tools had never been called once, and 147 of the 148
tools ever called sat behind them.

The fix is not deletion and not a smaller library. A tool that is never called
is still worth keeping -- it was forged for a need that may come back -- but it
does not need its full parameter schema in front of the model to be *findable*.
So exposure splits in two:

    hot   -- called within the window, plus a floor of tools that carry the
             interaction itself (run_python, forge_tool, describe_tool, ...).
             Full schema, as before.
    cold  -- everything else. Name, parameters and a one-line description:
             enough to recognise the tool and ask for it by name.

A cold tool is not hidden and not retired. It costs a line, it is callable as
soon as the model names it, and `describe_tool` hands back the full schema on
request. The saving is real only because the model does not need to know the
argument shapes of 61 tools it is not calling this turn -- and the price of
being wrong is one extra call, not a lost capability.

A third tier is deliberately absent: dropping the name entirely and making the
model guess from a search box. The market does that, and the reason a name-and-
line list beats it is that it costs 20 characters and removes the whole class of
failure where the tool exists and the model never learns it does.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Sequence

#: The tools that carry the interaction itself. A floor, not a preference: these
#: are the ones a model reaches for when it does not yet know what it needs, so
#: they must never be demoted to a name on a list. `run_python` is where a
#: one-off lookup goes; `forge_tool`/`describe_tool`/`list_tools` are how a cold
#: tool is found again; `remember`/`recall` carry facts across sessions;
#: `terminate` ends the run. Everything else earns its place by being called.
HOT_FLOOR: frozenset[str] = frozenset({
    "run_python", "describe_tool", "list_tools", "forge_tool", "terminate",
    "remember", "recall",
})

#: The floor expressed in the bridge's own namespace. The bridge registers
#: `autoforge_run_python`, not `run_python`, so a floor written in short names
#: protects nothing there -- the same naming gap that made `ProgramFiles` vanish
#: in the sandbox allow-list. Kept derived from HOT_FLOOR, not restated.
BRIDGE_FLOOR: frozenset[str] = frozenset(f"autoforge_{n}" for n in HOT_FLOOR)

#: Description characters kept in the cold list. Long enough to say what the
#: tool is for, short enough that 120 of them cost less than ten hot schemas.
COLD_DESC_CHARS = 100

#: Parameter names shown in the cold list. The point of showing any is that
#: "read_magic_value(path)" is recognisable where "read_magic_value" alone is
#: only a guess; the point of capping is that the list is one line per tool.
COLD_PARAM_NAMES = 4

#: What a cold tool's parameter map costs once it has been reduced to a stub:
#: `{"type":"object","properties":{}}`. Counted so `classify` can compare the
#: cold form against the full one instead of assuming cold is cheaper.
COLD_STUB_CHARS = len('{"type":"object","properties":{}}')


def _names(schema: dict[str, Any]) -> list[str]:
    """The argument names a tool takes, whichever of the two shapes it is in.

    Two shapes reach this function and only one of them used to be handled:

    * JSON Schema -- `{"type": "object", "properties": {"path": {...}}}`.
      This is what the registry stores and what `defineTool` compiles to.
    * A descriptor -- `{"path": {"type": "string", "required": true}}`.
      This is what the bridge DECLARES in `lib/index.js` and what it reports
      through `report_bridge_tools`, so `bridge_tools.parameters` is in this
      shape for all 66 names. Measured 2026-11-02: the cold line for
      `autoforge_skill_write` read "Takes no arguments." while its real
      parameters are `name` and `body`, i.e. the one line shown to the model
      was wrong about the one thing it exists to say.

    The shapes are told apart by their VALUES, not by their keys. The first
    attempt here kept a keyword blocklist (`type`, `required`, `description`,
    ...) and returned "no arguments" when any key matched -- which failed on
    `autoforge_tool_define`, whose arguments include one literally called
    `description`. A tool that takes a parameter named after a schema keyword is
    not exotic; it is the tool the whole manifest is built around. So: a
    descriptor is a mapping whose every value is a dict, which no JSON Schema's
    top level ever is (`type` is a string, `required` is a list).
    """
    fn = schema.get("function") if isinstance(schema.get("function"), dict) else schema
    params = fn.get("parameters") if isinstance(fn, dict) else None
    if not isinstance(params, dict) or not params:
        return []
    props = params.get("properties")
    if isinstance(props, dict):
        return list(props)
    if all(isinstance(value, dict) for value in params.values()):
        return list(params)
    return []


def _schema_name(schema: dict[str, Any]) -> str:
    fn = schema.get("function") if isinstance(schema.get("function"), dict) else schema
    return str(fn.get("name") or "")


def _schema_desc(schema: dict[str, Any]) -> str:
    fn = schema.get("function") if isinstance(schema.get("function"), dict) else schema
    return str(fn.get("description") or "").strip().split("\n")[0]


def cold_line(name: str, schema: dict[str, Any],
              desc_override: str = "") -> str:
    """One line standing in for a tool that is not being handed over in full."""
    params = _names(schema)
    shown = ", ".join(params[:COLD_PARAM_NAMES])
    if len(params) > COLD_PARAM_NAMES:
        shown += ", ..."
    desc = (desc_override or _schema_desc(schema)).strip().replace("\n", " ")
    if len(desc) > COLD_DESC_CHARS:
        desc = desc[:COLD_DESC_CHARS - 1].rstrip() + "\u2026"
    sig = f"{name}({shown})" if shown else f"{name}()"
    return f"  - {sig} \u2014 {desc}" if desc else f"  - {sig}"


def protect_from_task(task: str, names: Iterable[str]) -> set[str]:
    """The tools this task names, so they ride in full whatever the ledger says.

    The ledger answers "what has been used lately", which is the right question
    for a cost decision and a question about the *past*. A turn whose text says
    "compile the CUDA kernel" needs `gpu_compile` readable on its first step --
    and if the ledger has never seen that name, the one turn that needs the tool
    is the one turn that cannot read its arguments.

    Deliberately narrow: a single-word name matches a whole token, and a
    multi-word name matches a *contiguous* run of tokens. Nothing matches a
    fragment. That boundary is the whole mechanism -- a set that grows to
    everything protects nothing, and the failure is invisible, because a
    protected tool simply costs what it always did.

    Upstream records the bug that taught this (`agent.py::_hot_protect`): the
    first draft tested `name.replace("_", " ") in task` with a plain substring,
    so a tool called `see` was kept hot by the word "seen" and one called `run`
    by "running". Measured here 2026-11-03 before trusting the port:
    `protect_from_task("I have seen it running", ["see", "run"])` is empty --
    asserted in `chk_hot_protect.py`, because the difference between the two
    implementations is exactly what that test exists for.

    Returns the names as given, not lower-cased: the caller passes them straight
    back into `classify`, which compares against the registry's own spellings.
    """
    text = task or ""
    if not text.strip():
        return set()
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if not tokens:
        return set()
    words = set(tokens)
    out: set[str] = set()
    for name in names:
        low = str(name).lower()
        if not low:
            continue
        if low in words:
            out.add(name)
            continue
        parts = low.split("_")
        if len(parts) < 2:
            continue
        width = len(parts)
        if any(tokens[i:i + width] == parts for i in range(len(tokens) - width + 1)):
            out.add(name)
    return out


def classify(
    schemas: Sequence[dict[str, Any]],
    days_since: Callable[[str], float | None],
    *,
    window_days: float = 7.0,
    floor: Iterable[str] = HOT_FLOOR,
    protect: Iterable[str] = (),
    cold_text: Callable[[str, dict[str, Any]], str] | None = None,
    full_cost: Callable[[str, dict[str, Any]], int] | None = None,
) -> tuple[set[str], list[str], dict[str, int]]:
    """(hot names, cold lines) -- the decision, separated from the packaging.

    `split` hands back schemas; the DSH bridge needs names, because it registers
    its tools once and can only swap what each one *says* per request. Both
    answers come from the rule below and nowhere else: two implementations of
    "which tools are hot" would be one implementation and one opinion.

    A tool that is not hot is only made cold if being cold is CHEAPER. That is
    not an optimisation, it is the whole point: measured 2026-11-02, the first
    version of the DSH cold text (a 100-character description plus a pointer
    sentence) cost about 160 characters for 66 tools whose full parameters
    average 179 -- so "saving" 60 of them by name made the prompt 5,996
    characters LONGER, a 49% increase in the size of the tool table. A cold tool
    with no parameters at all was the worst case: it traded `{}` for a sentence.
    Pass `cold_text` and `full_cost` and the rule keeps such a tool hot, where
    it is at least honest about its size.
    """
    keep = set(floor) | set(protect)
    hot_names: set[str] = set()
    lines: list[str] = []
    for schema in schemas:
        name = _schema_name(schema)
        age = days_since(name)
        worth = True
        if cold_text is not None and full_cost is not None:
            text = cold_text(name, schema)
            worth = len(name) + len(text) + COLD_STUB_CHARS < full_cost(name, schema)
            if not worth:
                hot_names.add(name)
                continue
        if name in keep or (age is not None and age <= window_days):
            hot_names.add(name)
        else:
            lines.append(cold_line(name, schema))
    stats = {"hot": len(hot_names), "cold": len(lines), "total": len(schemas)}
    return hot_names, lines, stats


def cold_description(name: str, schema: dict[str, Any], window_days: float,
                     desc_override: str = "") -> str:
    """The `description` a cold tool carries in place of its real one.

    This is the DSH half of the same bargain `cold_line` makes in the CLI: the
    model keeps the name, the argument names and one line of purpose, and is told
    where the rest is. Written as prose in the description slot because that is
    the only place a registered tool can say anything -- the parameters slot has
    already been reduced to `{}` by the time this is rendered.
    """
    params = _names(schema)
    shown = ", ".join(params[:COLD_PARAM_NAMES])
    if len(params) > COLD_PARAM_NAMES:
        shown += ", ..."
    desc = (desc_override or _schema_desc(schema)).strip().replace("\n", " ")
    if len(desc) > COLD_DESC_CHARS:
        desc = desc[:COLD_DESC_CHARS - 1].rstrip() + "\u2026"
    where = f"Params: {shown}." if shown else "Takes no arguments."
    return (
        f"{desc} [{where} Not loaded in full: no call in {window_days:g} days. "
        f"autoforge_describe_tool({name!r}) returns the full schema.]"
    ).strip()


def split(
    schemas: Sequence[dict[str, Any]],
    days_since: Callable[[str], float | None],
    *,
    window_days: float = 7.0,
    floor: Iterable[str] = HOT_FLOOR,
    protect: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    """Split schemas into (hot, cold lines, counts).

    `days_since(name)` returns how long ago the tool was last called, or None if
    there is no record of it ever being called. A tool with no record is cold --
    that is the whole point: never called is the strongest evidence available
    that its argument list does not need to be in front of the model this turn.

    `window_days` bounds the number of schemas by *behaviour*, not by a count.
    A hard cap needs a tie-break, and every tie-break it could have (name order,
    cost, age) demotes a tool for a reason unrelated to whether it is wanted. A
    window demotes for exactly one reason -- nothing reached for it in a week --
    and self-corrects the moment something does.

    `protect` is the caller's escape hatch: names that must stay hot this run
    whatever the ledger says (a tool the task text names, a topology's role
    whitelist). Kept separate from `floor` so a caller's reason and the
    framework's reason are not folded into one set.
    """
    hot_names, lines, stats = classify(
        schemas, days_since,
        window_days=window_days, floor=floor, protect=protect,
    )
    hot = [s for s in schemas if _schema_name(s) in hot_names]
    return hot, lines, stats


def manifest_header(stats: dict[str, int], window_days: float) -> str:
    return (
        f"Tools you can call but are not carrying in full ({stats['cold']} of "
        f"{stats['total']}): not called in {window_days:g} days, so only their "
        "name, arguments and one line are shown. They are callable as they are "
        "\u2014 use one directly, or call describe_tool(name) for its full "
        "parameter schema first."
    )
