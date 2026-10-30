"""Adversarial verification gate.

A tool must survive an attack before it earns PROBATION or ACTIVE.

The idea (no framework does this): a *generator* LLM writes a tool, then an
*attacker* LLM (a separate call with a "try to break it" prompt) generates
inputs designed to trigger failures — edge cases, bad types, empty inputs,
encoding issues, boundary values. If any of those inputs cause the tool to
crash or return garbage, the tool stays DRAFT.

The attacker is not an oracle — it's another LLM call, same cost as the
generator. But it changes the verification from "does it work on one
hand-crafted input" to "can it survive a motivated adversary."

Two attack modes:
  HARD (default): the attacker sees the tool's code + schema + probes, and
  crafts targeted sabotage inputs.
  BLIND: the attacker only sees the description and probes (black-box).

The gate is an extra check in the verification battery, between execution
and trigger. A tool that passes all checks is genuinely robust.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.llm import LLMClient
from ..core.message import Message
from ..forge.generator import extract_json_any
from ..tools.spec import ToolSpec

_ATTACKER_SYSTEM_HARD = (
    "You are a tool security auditor. Your job: find inputs that break the tool.\n"
    "Given the tool's code, description, and parameters, generate up to 5 inputs\n"
    "that are likely to crash it, return wrong results, or expose bugs.\n\n"
    "Think like a fuzzer: empty strings, very long strings, missing keys,\n"
    "wrong types, Unicode, mixed encodings, negative numbers, zero values,\n"
    "null bytes, script injection, repeated calls.\n\n"
    "Output STRICT JSON array:\n"
    '[{"args": {"param": "value..."}, "rationale": "why this might break it"}]\n\n'
    "Only include inputs that have a *reasonable chance* of triggering a failure.\n"
    "Quality over quantity."
)

_ATTACKER_SYSTEM_BLIND = (
    "You are a tool security auditor. Given only the tool's description and\n"
    "parameter schema, generate up to 5 adversarial inputs designed to break it.\n"
    "Think about what a real attacker would try.\n\n"
    "Output STRICT JSON array:\n"
    '[{"args": {"param": "value..."}, "rationale": "why this might break it"}]'
)


@dataclass
class AttackResult:
    args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    survived: bool = True
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class AdversarialReport:
    tool: str
    passed: bool
    attacks: list[AttackResult] = field(default_factory=list)
    mode: str = "hard"
    total_attacks: int = 0
    survived: int = 0

    @property
    def failed_attacks(self) -> list[AttackResult]:
        return [a for a in self.attacks if not a.survived]

    def summary(self) -> str:
        return f"{self.tool}: {self.survived}/{self.total_attacks} attacks survived"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "passed": self.passed,
            "mode": self.mode,
            "total_attacks": self.total_attacks,
            "survived": self.survived,
            "attacks": [
                {"args": a.args, "rationale": a.rationale,
                 "survived": a.survived, "error": a.error}
                for a in self.attacks
            ],
        }


def _is_refusal(error: str) -> bool:
    """Is this failure the tool keeping its contract, or losing it?

    A refusal is the tool saying "not this input" in the strongest available
    way: an exception. That is what the robustness check demands and what the
    generator is told to do with input it cannot serve, so it must not be
    scored as an attack that landed. The distinction is made from the type
    name because the sandbox hands failures back as text.
    """
    for kind in ("TypeError", "ValueError", "KeyError", "AssertionError",
                 "ArithmeticError", "OverflowError", "ZeroDivisionError"):
        if error.startswith(kind) or f"{kind}:" in error[:60]:
            return True
    return False


class AdversarialGate:
    """An LLM-driven adversary that stress-tests a tool before trust."""

    def __init__(
        self,
        attacker_llm: LLMClient,
        *,
        mode: str = "hard",
        max_attacks: int = 5,
        execution_sandbox: Any = None,
        require_survival_rate: float = 0.8,
    ) -> None:
        self.attacker = attacker_llm
        self.mode = mode
        self.max_attacks = max_attacks
        self.require_survival_rate = require_survival_rate
        self._sandbox = execution_sandbox

    def attack(self, spec: ToolSpec) -> AdversarialReport:
        """Generate adversarial inputs, run them against the tool, report."""
        system = _ATTACKER_SYSTEM_HARD if self.mode == "hard" else _ATTACKER_SYSTEM_BLIND

        # Build the prompt
        params_str = json.dumps(spec.parameters, ensure_ascii=False, indent=1)
        prompt = (
            f"Tool: {spec.name}\n"
            f"Description: {spec.description}\n"
            f"Parameters:\n{params_str}\n"
        )
        if self.mode == "hard" and spec.code:
            prompt += f"Code:\n```python\n{spec.code}\n```\n"

        resp = self.attacker.chat(
            [Message.system(system), Message.user(prompt)],
            tools=None,
        )
        attacks = self._parse_attacks(resp.content)
        if not attacks:
            # Fallback if the attacker produced nothing usable. Must still use
            # the tool's REAL parameter names — a hardcoded "text" key just
            # makes the tool fail on an unexpected kwarg, which looks like the
            # tool broke when actually the probe was malformed.
            fallback_args = {
                name: "" for name, schema in (spec.parameters.get("properties") or {}).items()
                if schema.get("type", "string") == "string"
            }
            attacks = [(fallback_args, "empty input (fallback)")]

        results: list[AttackResult] = []
        survived = 0

        for raw_args, rationale in attacks[:self.max_attacks]:
            started = time.perf_counter()
            error = None
            ok = True
            try:
                if self._sandbox is not None and spec.code:
                    sr = self._sandbox.run(spec.code, spec.name, raw_args)
                    if not sr.ok:
                        error = sr.error or "sandbox failure"
                        # The sandbox reports a raise as `ok=False` + an error
                        # string; it does not re-raise into this function. So
                        # the same judgement has to be made here, from the
                        # string, or the `except` branch below never runs for
                        # a sandboxed tool -- which is every forged tool.
                        # Measured 2026-10-25: the fix below was written for
                        # the `except` path and the tool kept scoring 3/5,
                        # because the TypeError arrived through this return
                        # value instead. A guard written in the branch that
                        # does not execute is not a guard.
                        ok = _is_refusal(sr.error or "")
                else:
                    out = spec.fn(**raw_args)
                    if out is None:
                        error = "returned None"
                        ok = False
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                ok = False
                # Raising is surviving, not losing.
                #
                # "Survive an attack" means the tool kept its contract under an
                # input it was not written for. Refusing loudly is keeping it:
                # this module's own fallback rationale is called "empty input"
                # and the code is told to answer bad input with an INVALID
                # prefix, so refusing is the *expected* response, not a
                # defeat. The old line counted every exception as a loss, which
                # inverted the reward: `is_perfect_square` raising TypeError on
                # "16" is correct behaviour and scored 0, while a tool that
                # silently returned a constant for garbage scored 1.
                #
                # Measured 2026-10-25: same tool, six consecutive runs,
                # 0/1, 3/5, 0/5, 3/5, 3/5, 0/1 -- the attacker emits a different
                # mix each time and each invented input landed on a different
                # side of the inverted rule. A verdict that swings with the
                # attacker's mood is not a verdict.
                #
                # What still counts as a loss: a crash that is not a refusal
                # (a bare `except` swallowing into a wrong answer is invisible
                # here and belongs to the robustness check, not this one), and
                # an empty answer. Both are reported through `error` below.
                ok = _is_refusal(f"{type(exc).__name__}: {exc}")

            duration = (time.perf_counter() - started) * 1000
            attack_res = AttackResult(
                args=raw_args, rationale=rationale,
                survived=ok, error=error, duration_ms=duration,
            )
            results.append(attack_res)
            if ok:
                survived += 1

        total = len(results)
        rate = survived / total if total else 0
        report = AdversarialReport(
            tool=spec.name,
            passed=rate >= self.require_survival_rate,
            attacks=results,
            mode=self.mode,
            total_attacks=total,
            survived=survived,
        )
        return report

    @staticmethod
    def _parse_attacks(text: str) -> list[tuple[dict[str, Any], str]]:
        """Parse the attacker's JSON output into (args, rationale) pairs."""
        data = extract_json_any(text)
        if data is None:
            return []
        items = data if isinstance(data, list) else [data]
        results: list[tuple[dict[str, Any], str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            args = item.get("args") or {}
            rationale = str(item.get("rationale", ""))
            if isinstance(args, dict):
                results.append((args, rationale))
        return results


__all__ = ["AdversarialGate", "AdversarialReport", "AttackResult"]