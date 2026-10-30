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

#: Description characters kept in the cold list. Long enough to say what the
#: tool is for, short enough that 120 of them cost less than ten hot schemas.
COLD_DESC_CHARS = 100

#: Parameter names shown in the cold list. The point of showing any is that
#: "read_magic_value(path)" is recognisable where "read_magic_value" alone is
#: only a guess; the point of capping is that the list is one line per tool.
COLD_PARAM_NAMES = 4


def _names(schema: dict[str, Any]) -> list[str]:
    fn = schema.get("function") if isinstance(schema.get("function"), dict) else schema
    params = fn.get("parameters") if isinstance(fn, dict) else None
    if not isinstance(params, dict):
        return []
    props = params.get("properties")
    return list(props) if isinstance(props, dict) else []


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
    keep = set(floor) | set(protect)
    hot: list[dict[str, Any]] = []
    lines: list[str] = []
    for schema in schemas:
        name = _schema_name(schema)
        age = days_since(name)
        if name in keep or (age is not None and age <= window_days):
            hot.append(schema)
        else:
            lines.append(cold_line(name, schema))
    stats = {
        "hot": len(hot),
        "cold": len(lines),
        "total": len(schemas),
    }
    return hot, lines, stats


def manifest_header(stats: dict[str, int], window_days: float) -> str:
    return (
        f"Tools you can call but are not carrying in full ({stats['cold']} of "
        f"{stats['total']}): not called in {window_days:g} days, so only their "
        "name, arguments and one line are shown. They are callable as they are "
        "\u2014 use one directly, or call describe_tool(name) for its full "
        "parameter schema first."
    )
