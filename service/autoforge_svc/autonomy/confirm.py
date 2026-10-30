"""The confirmation gate: turning a switched-off freedom into a question.

The four execution freedoms (`may_read_filesystem`, `may_write_filesystem`,
`may_access_network`, `may_install_packages`) used to be declared-only. The
policy classified them as such and printed the classification, which was
honest, but honesty about a dead switch is still a dead switch — turning one
off changed nothing at all.

This module gives them teeth of a specific kind: OFF now means *not without a
yes*. Before a tool whose declared scope needs a disabled freedom runs, the
gate stops and asks the operator. Three properties make that more useful than
a hard refusal:

  * it is a real gate — the switch you flip changes what happens next;
  * it is not a dead end — the operator can say yes, and the tool runs;
  * it fails closed when nobody is there to answer (a headless run has no one
    to ask, and "no answer" must not read as "yes").

The alternative reading of "off" — silently permit — is the one thing this
avoids: a policy that claims a restriction it does not apply is worse than no
restriction, because it is a safety net you would trust with your weight.
"""
from __future__ import annotations

from typing import Any, Callable

#: The capabilities a tool can declare needing.
#:
#: `read_only` is treated as reading even though validity's drift table lists
#: no capability for it. The two tables answer different questions — validity
#: asks "did the tool exceed what it declared?", the gate asks "may this run?"
#: — so the read capability is added here rather than by editing that table,
#: where dropping it into `read_only` would weaken the drift check.
CAPABILITY_FREEDOM: dict[str, str] = {
    "filesystem_read": "may_read_filesystem",
    "credential_access": "may_read_filesystem",
    "filesystem_write": "may_write_filesystem",
    "filesystem_delete": "may_write_filesystem",
    "network_egress": "may_access_network",
    "subprocess": "may_run_arbitrary_code",
    "process_control": "may_run_arbitrary_code",
    "dynamic_code_execution": "may_run_arbitrary_code",
    "deserialization": "may_run_arbitrary_code",
}

#: Capabilities implied by a scope beyond what validity's table lists.
_SCOPE_EXTRA: dict[str, frozenset[str]] = {
    "read_only": frozenset({"filesystem_read"}),
    "local_write": frozenset({"filesystem_read"}),
    "network": frozenset({"filesystem_read"}),
    "system": frozenset({"filesystem_read"}),
    "privileged": frozenset({"filesystem_read"}),
}

#: A confirmer is asked once per gated call: `(tool_name, arguments, freedoms)`.
#: It answers `True` to let the call through, `False` to refuse it, or `None`
#: when there is nobody to ask — a headless run, a worker thread with a browser
#: on the other end, a closed stdin. The gate refuses on `None` too, but says
#: "nobody to ask" rather than "the operator said no", because those are
#: different facts about who decided.
Confirmer = Callable[[str, dict[str, Any], list[str]], bool | None]


def declared_capabilities(scope: str) -> frozenset[str]:
    """What a tool's declared scope says it needs.

    An undeclared or unrecognised scope is treated as capable of everything:
    the gate's job is to ask when it cannot tell, not to assume innocence.
    """
    from ..forge.validity import SCOPE_ALLOWANCES          # local: avoid cycle

    if scope not in SCOPE_ALLOWANCES:
        scope = "undeclared"
    return frozenset(SCOPE_ALLOWANCES[scope]) | _SCOPE_EXTRA.get(scope, frozenset())


def _normalise_scope(effect_signature: str) -> str:
    """The scope name, or `undeclared` when it is not one we recognise.

    Mirrors `validity.declared_scope`, which reads the same field off a
    ToolSpec. Done here rather than by calling it with a stand-in object so the
    gate has no dependency on the tool being a real spec.
    """
    from ..forge.validity import SCOPE_ALLOWANCES          # local: avoid cycle

    sig = (effect_signature or "").strip().lower()
    return sig if sig in SCOPE_ALLOWANCES else "undeclared"


def required_freedoms(effect_signature: str) -> tuple[str, ...]:
    """The policy fields a tool with this effect signature depends on."""
    freedoms = {
        CAPABILITY_FREEDOM[cap]
        for cap in declared_capabilities(_normalise_scope(effect_signature))
        if cap in CAPABILITY_FREEDOM
    }
    return tuple(sorted(freedoms))


def disabled_freedoms(effect_signature: str, policy: Any) -> list[str]:
    """Which of a tool's required freedoms this policy has switched off.

    Empty means the gate has nothing to ask about and the tool may run.

    Only the four CONFIRM_REQUIRED freedoms are ever returned. The other
    freedoms a scope can need — `may_run_arbitrary_code` above all — already
    have their own gates inside the tools that honour them, and asking twice
    for one decision is how an operator learns to answer without reading.
    `required_freedoms` still reports the full dependency; this is the
    subset the gate is answerable for.
    """
    from .policy import CONFIRM_REQUIRED                 # local: avoid cycle

    return sorted(
        f for f in required_freedoms(effect_signature)
        if f in CONFIRM_REQUIRED and not getattr(policy, f, True)
    )


def describe(needed: list[str], tool_name: str, arguments: dict[str, Any]) -> str:
    """The question, in the shape an operator can answer.

    Names the switch, the tool, and the arguments — an approval prompt that
    does not say what is being approved is a prompt that trains its reader to
    say yes without looking.
    """
    what = ", ".join(needed)
    args = ", ".join(f"{k}={v!r}" for k, v in arguments.items()) or "(no arguments)"
    return (f"allow {tool_name!r} to run? it needs {what}, "
            f"which your policy has switched off. arguments: {args}")
