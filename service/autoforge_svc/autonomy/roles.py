"""What a role means, given that naming one has to change something.

`RoleType` had five members and one effect: the role name was used to look up a
node's `tools_whitelist`, so a node with an empty whitelist ran *unrestricted*.
CRITIC, GATE and FORGE were therefore labels — a "critic" that could rewrite the
artifact it was reviewing, a "gate" that could run the thing it was gating. The
designer could ask for a reviewer and get a second pair of hands.

Two things make a role real here:

**A brief.** Every role carries a directive that reaches the child's system
prompt. Before this, `AgentNode.system_prompt_hint` was written by the designer,
round-tripped through the store, printed in `design_team`'s output — and read by
nothing. A hint the child never sees is a note to the file.

**A ceiling.** Every role is decided on the capability axis the rest of the
framework already uses (`SCOPE_ALLOWANCES` / `declared_capabilities`), not a
second vocabulary invented here. CRITIC and GATE are *defined* by not changing
the world, so they cannot; COORDINATOR, WORKER and FORGE execute, and executing
means reaching it.

The ceiling is a hard cap, not a default: a whitelist can narrow a role further
but cannot widen it past its role. Otherwise a designer that whitelists a write
tool for a critic gets the old behaviour back and the role is decoration again —
and it would be *silent* decoration, which is the failure this module exists to
name. A refusal is recorded on the child's registry and reported by `spawn_agent`
("Reached past its role and was refused: ..."), so permission denied is loud.
"""
from __future__ import annotations

from typing import Any, Callable

from .topology import AgentNode, RoleType

#: What each role is for, in the words the child reads. Short on purpose: a
#: directive that has to be skimmed is a directive that gets skimmed.
ROLE_BRIEF: dict[RoleType, str] = {
    RoleType.COORDINATOR: (
        "You are the coordinator. Decompose the task, delegate the parts, and "
        "merge what comes back into one answer. Do not do the fine work "
        "yourself: if you are editing files, the split was wrong."
    ),
    RoleType.WORKER: (
        "You are a worker. Do the one task you were given, completely, and "
        "report what you actually changed rather than what you intended to. "
        "Do not redesign the wider plan."
    ),
    RoleType.CRITIC: (
        "You are a critic. Your job is to find what is wrong with the work you "
        "are shown — the false claim, the untested path, the case that was "
        "skipped. You cannot change anything; you report findings. A review "
        "that finds nothing is only useful if you say what you checked."
    ),
    RoleType.GATE: (
        "You are a gate. Decide whether the work you are shown may pass on, and "
        "answer yes or no with the reason. You cannot modify what you are "
        "judging or run it yourself. If you cannot tell whether it is correct, "
        "the answer is no."
    ),
    RoleType.FORGE: (
        "You are the forge. Build the tool the task needs and verify it works "
        "before handing it back — a tool that runs is not the same as a tool "
        "that is right. Prefer a small tool you have tested over a large one "
        "you have not."
    ),
}

#: The capabilities each role may reach, or None for "anything".
#:
#: A total function over `RoleType`, deliberately: adding a role without
#: deciding what it may touch is the bug this table is here to make impossible.
#: `test_roles.py` asserts totality, so a sixth member fails the suite rather
#: than defaulting to unrestricted.
ROLE_MAY: dict[RoleType, frozenset[str] | None] = {
    # Orchestration and execution both need the world. Restricting them would
    # make the role a description of the prompt rather than of the work.
    RoleType.COORDINATOR: None,
    RoleType.WORKER: None,
    RoleType.FORGE: None,
    # Defined by their limits: these two exist to judge, not to change. Read is
    # the whole allowance — anything that writes, runs, or reaches out would let
    # them alter or exercise the thing they are supposed to be assessing.
    RoleType.CRITIC: frozenset({"filesystem_read"}),
    RoleType.GATE: frozenset({"filesystem_read"}),
}

#: Roles whose ceiling is worth naming in a one-line refusal, for the message a
#: child gets when it reaches past its role.
_ROLE_DENIAL = {
    RoleType.CRITIC: "a critic reviews and reports; it does not change the work",
    RoleType.GATE: "a gate judges and passes on; it does not change the work",
}


def role_brief(role: RoleType | str | None) -> str:
    """The directive for a role, or "" for an unknown/absent one."""
    r = _as_role(role)
    return ROLE_BRIEF.get(r, "") if r is not None else ""


def role_ceiling(role: RoleType | str | None) -> frozenset[str] | None:
    """The capabilities a role may reach. None means unrestricted."""
    r = _as_role(role)
    return ROLE_MAY.get(r) if r is not None else None


def role_denial(role: RoleType | str | None) -> str:
    return _ROLE_DENIAL.get(_as_role(role), "")


def ceiling(role: RoleType | str | None) -> Callable[[Any], str] | None:
    """The predicate that holds a child to a role's ceiling, or None.

    None means "this role may reach anything", so the common case costs no extra
    check per tool call — and "unrestricted" is the *absence* of a predicate
    rather than a predicate that always says yes.

    Lives here rather than at the call site so that `Spawner.spawn(role="critic")`
    is safe on its own. A ceiling that only the caller can supply is a ceiling
    every caller but one forgets, and the forgotten one is the bug.
    """
    if not refuses(role, "privileged"):
        # Probed with the widest scope: a role that refuses nothing even there
        # has no ceiling to enforce.
        return None

    def predicate(spec: Any) -> str:
        missing = refuses(role, getattr(spec, "effect_signature", ""))
        if not missing:
            return ""
        why = role_denial(role)
        name = getattr(spec, "name", "?")
        tail = f" — {why}." if why else "."
        return (f"Denied: '{name}' needs {', '.join(missing)}, which "
                f"{str(getattr(role, 'value', role) or 'this role')} "
                f"may not use{tail}")

    return predicate


def refuses(role: RoleType | str | None, effect_signature: str) -> tuple[str, ...]:
    """Capabilities this role may not use, given a tool's declared scope.

    Returns the *capability* names rather than a bare bool so the refusal can
    say which switch was reached past. An unrecognised scope counts as capable
    of everything (see `confirm.declared_capabilities`): a role that cannot tell
    what a tool does does not get to assume it is harmless.
    """
    ceiling = role_ceiling(role)
    if ceiling is None:
        return ()
    from .confirm import declared_capabilities, _normalise_scope

    needed = declared_capabilities(_normalise_scope(effect_signature))
    return tuple(sorted(needed - ceiling))


def brief_for_node(node: AgentNode) -> str:
    """The full directive a child running as `node` should start with.

    The designer's own hint is kept, not replaced by the role's: the role says
    what kind of agent this is, the hint says what this particular one is doing.
    """
    parts = [role_brief(node.role)]
    hint = (node.system_prompt_hint or "").strip()
    if hint:
        parts.append(hint)
    return "\n\n".join(p for p in parts if p)


def _as_role(role: Any) -> RoleType | None:
    if isinstance(role, RoleType):
        return role
    if not role:
        return None
    try:
        return RoleType(str(role))
    except ValueError:
        return None


__all__ = [
    "ROLE_BRIEF",
    "ROLE_MAY",
    "brief_for_node",
    "ceiling",
    "refuses",
    "role_brief",
    "role_ceiling",
    "role_denial",
]
