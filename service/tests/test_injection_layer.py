"""What reaches the model every turn, and what deliberately does not.

The four blocks of `context_brief` are not four features, they are one claim:
that an agent which has forgotten to look somewhere will not look, so the things
it must not have to look for are paid for on every request. That claim fails in
two directions, and both are tested here.

  * A block that should be there is missing -- the model re-derives what it
    already wrote down, or reasons about its own reach from its tool list and
    concludes it has no hands.
  * A block that is there is paid for twice -- the brief and `my_capabilities`
    saying the same sentence, so the second copy buys nothing and the block
    teaches the model to skim.

The second direction was real and measured: the brief at 3209 chars and
`my_capabilities` at 1002 shared four lines verbatim (filesystem, network,
run_python, DSH). These tests pin the split so it cannot drift back.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoforge_svc.service import Service   # noqa: E402


@pytest.fixture()
def svc(tmp_path):
    """A service on its own ledger, with a skill directory holding one skill."""
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "sample-skill.md").write_text(
        "---\ndescription: does a sample thing\nwhen_to_use: when a sample is needed\n---\n"
        "body\n",
        encoding="utf-8",
    )
    old = os.environ.get("AUTOFORGE_SKILLS_DIRS")
    os.environ["AUTOFORGE_SKILLS_DIRS"] = str(skills)
    try:
        yield Service(str(tmp_path / "autoforge.db"))
    finally:
        if old is None:
            os.environ.pop("AUTOFORGE_SKILLS_DIRS", None)
        else:
            os.environ["AUTOFORGE_SKILLS_DIRS"] = old


# -- the four blocks are all present ---------------------------------------

def test_the_brief_carries_all_four_blocks(svc):
    """Four blocks, in the order the model should read them.

    Order is not decoration: it reads what it has kept (facts), what it owes
    (mission), what it has to work with (tools), what it has written down
    (skills), and finally what it is -- identity last, because it exists to stop
    the model from apologising for capabilities the four blocks above it just
    described.
    """
    text = svc.context_brief()["text"]
    headings = ["Kept facts", "MISSION", "TOOLS", "SKILLS", "WHO I AM"]
    for heading in headings:
        assert heading in text, f"{heading} missing from the brief"
    positions = [text.index(h) for h in headings]
    assert positions == sorted(positions), "blocks are out of order"


def test_the_permissions_block_states_reach_not_a_disclaimer(svc):
    """The block must say what is possible, in the imperative.

    "We advise caution" is the shape that produces an agent apologising for
    hands it has. The two sentences that change behaviour are named explicitly.
    """
    text = svc.context_brief()["text"]
    assert "run_python" in text
    assert "forge_tool" in text
    assert "not the same as being unable to do it" in text


# -- and paid for exactly once ---------------------------------------------

def test_the_brief_and_the_capability_table_do_not_repeat_each_other(svc):
    """The duplication that was measured: the same sentence in both blocks.

    `filesystem` and `network` are the ones that drifted. The brief states what
    they ARE; the table reports only the BOUNDS, which the brief does not carry.
    """
    brief = svc.context_brief()["text"]
    caps = svc.my_capabilities()["text"]
    # The values, not the words: `read-write, whole host` appears once.
    assert "read-write" in brief
    assert "read-write" not in caps
    assert "outbound" in brief
    assert "outbound" not in caps


def test_the_capability_table_still_reports_the_numbers(svc):
    """De-duplicating must not turn the table into a pointer at nothing."""
    caps = svc.my_capabilities()["text"]
    assert "timeout" in caps
    assert "cwd" in caps
    assert "env" in caps


# -- the budget is a budget, not a bug with a number in it ------------------

def test_the_identity_block_is_not_truncated_on_this_host(svc):
    """A cap that always fires is not a cap; it is a truncation.

    Measured at 480: the block is 523 chars once the host's reach strings are
    substituted, so every render ended "...(identity truncated)" mid-sentence.
    """
    text = svc.context_brief()["text"]
    assert "identity truncated" not in text
    assert "call my_capabilities when a boundary actually matters" in text


def test_every_block_degrades_to_a_line_instead_of_raising(svc, monkeypatch):
    """One unreadable block must not cost the other three.

    The caller is a prompt assembly path: an exception here drops the whole
    request, which is a strictly worse failure than a missing block.
    """
    def boom(*a, **k):
        raise RuntimeError("store is locked")

    monkeypatch.setattr(svc.store, "memory_for_injection", boom)
    monkeypatch.setattr(svc.sandbox, "reach", boom)
    text = svc.context_brief()["text"]
    assert "unreadable" in text.lower()
    # The blocks that do not depend on the broken pieces still render.
    assert "TOOLS" in text


# -- the call is visible from outside --------------------------------------

def test_a_served_brief_is_counted_on_the_ledger(svc):
    """Whether the block reaches the model is a claim about the host's wiring.

    The only way to check it from outside is to see the calls: an empty count in
    a session that has run turns means the plugin's refresh timer never fired,
    which is a different fault from the block rendering empty.
    """
    before = svc.store.report()["events"]
    svc.context_brief()
    after = svc.store.report()["events"]
    assert after > before


def test_the_brief_is_cheap_enough_to_pay_for_every_turn(svc):
    """3ms measured; the assertion is loose because CI hosts vary.

    The point is not the number, it is that nothing here does I/O the size of a
    database scan per request -- a block that costs 200ms on every prompt is a
    block that gets turned off.
    """
    import time

    svc.context_brief()          # warm any caches
    start = time.perf_counter()
    for _ in range(5):
        svc.context_brief()
    per_call_ms = (time.perf_counter() - start) * 200
    assert per_call_ms < 150, f"brief cost {per_call_ms:.0f}ms per turn"
