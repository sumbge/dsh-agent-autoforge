"""Message primitives for the agent loop.

Deliberately provider-neutral: `to_api()` emits OpenAI-format dicts because
that is the lingua franca, but nothing else in the framework depends on it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None

    def to_api(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "ToolCall":
        fn = raw.get("function") or {}
        args_raw = fn.get("arguments")
        if isinstance(args_raw, dict):
            args = args_raw
        else:
            try:
                args = json.loads(args_raw or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": args_raw}
        if not isinstance(args, dict):
            args = {"_value": args}
        name = fn.get("name") or raw.get("name") or ""
        return cls(
            id=raw.get("id") or f"call_{name or 'unknown'}",
            name=name,
            arguments=args,
            raw=raw,
        )


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    #: The same message as wire parts, when it carries something the text
    #: cannot: a picture the operator pasted. `content` stays the text, so
    #: compacting, truncating and recording all keep working on a string --
    #: only the outgoing request sees the parts.
    images: list[dict[str, Any]] | None = None

    def to_api(self, content: Any = None) -> dict[str, Any]:
        """The wire form, with parts when this message carries a picture.

        The stored `content` stays the text: compaction, truncation, the ledger
        and the transcript all keep reading the string they always did, and only
        the outgoing request sees the parts. The `content` argument is an
        override for a caller with its own view of the message -- it is not how
        images get in, and handing it anything but wire content replaces what the
        model is sent.
        """
        if content is None and self.images:
            content = self.images
        msg: dict[str, Any] = {"role": self.role,
                               "content": self.content if content is None else content}
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_api() for tc in self.tool_calls]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg

    # -- persistence ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """This message as plain data, for a journal that outlives the process.

        The whole message, not a summary: a checkpoint that kept only the text
        would lose the tool calls, and a transcript whose tool calls are missing
        is not a transcript the model can be asked to continue -- it is a
        different conversation that happens to end with the same sentence.
        """
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        if self.images:
            data["images"] = self.images
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """The inverse of `to_dict`, tolerant of a row written by older code.

        Tolerant rather than strict because the reader is a resume path: it runs
        precisely when something has already gone wrong, and a journal written
        by the version before this one must not be the second failure. A missing
        field is a default; an unknown field is ignored.
        """
        if not isinstance(data, dict):
            return cls("user", str(data))
        calls = [
            ToolCall(
                id=str(tc.get("id") or f"call_{tc.get('name') or 'unknown'}"),
                name=str(tc.get("name") or ""),
                arguments=(tc.get("arguments") if isinstance(tc.get("arguments"), dict)
                           else {}),
            )
            for tc in (data.get("tool_calls") or [])
            if isinstance(tc, dict)
        ]
        return cls(
            role=data.get("role") or "user",
            content=data.get("content") or "",
            tool_calls=calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            images=(data.get("images") if isinstance(data.get("images"), list) else None),
        )

    # -- constructors -------------------------------------------------
    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)

    @classmethod
    def user(cls, content: str, images: list[dict[str, Any]] | None = None) -> "Message":
        return cls("user", content, images=images)

    @classmethod
    def assistant(
        cls, content: str = "", tool_calls: list[ToolCall] | None = None
    ) -> "Message":
        return cls("assistant", content, tool_calls or [])

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str | None = None) -> "Message":
        return cls("tool", content, tool_call_id=tool_call_id, name=name)
