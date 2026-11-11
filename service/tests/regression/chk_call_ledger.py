"""Does the `call` row say enough to reproduce the call?

`forge_events` is the only record that survives a restart. For a long time a
`call` row carried `{"tool", "ok"}` and nothing else -- measured 2026-11-02, all
15 rows in the live database had exactly those two keys. The consequences:

  * a failure could not be reproduced, because the arguments were not there;
  * a slow call could not be picked out, because the duration was not there;
  * the error string existed only in `ToolStats.history`, which is in memory and
    dies with the process.

What the row must NOT carry is the whole argument. `tool_define` carries a
tool's source, so writing values verbatim would put several thousand characters
of Python into an append-only table on every forge.

Run from the service directory:
    python tests/regression/chk_call_ledger.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc.service import Service                      # noqa: E402
from autoforge_svc.tools.registry import ToolResult            # noqa: E402
from autoforge_svc.tools.spec import ToolSpec, ToolState       # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    # `detail: object`, not `detail: str`: a failure path that formats its own
    # evidence must not be able to crash the suite. Measured 2026-11-06 -- a
    # mutant reached a check with a list and the suite died with
    # `TypeError: can only concatenate str (not "list") to str` before printing
    # any verdict, which the mutation driver correctly scores as "crashed",
    # not "caught". An assertion that cannot report its evidence is not there.
    print(("  OK   " if cond else "  FAIL ") + label
          + (("  <- " + repr(detail)) if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


def rows(svc, kind="call"):
    return [
        json.loads(r["payload"])
        for r in svc.store._conn.execute(
            "SELECT payload FROM forge_events WHERE kind=? ORDER BY id", (kind,)
        ).fetchall()
    ]


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m10_call_")
    svc = Service(db_path=os.path.join(tmp, "t.db"))
    try:
        def echo(text: str = "", n: int = 1) -> str:
            return text * n

        def boom(text: str = "") -> str:
            raise ValueError(f"cannot handle {text!r}")

        def slow(ms: int = 30) -> str:
            import time as _t
            _t.sleep(ms / 1000.0)
            return "done"

        for spec in (
            ToolSpec(name="echo_x", description="d", parameters={"type": "object",
                     "properties": {"text": {"type": "string"}, "n": {"type": "integer"}}},
                     fn=echo, state=ToolState.ACTIVE),
            ToolSpec(name="boom_x", description="d", parameters={"type": "object",
                     "properties": {"text": {"type": "string"}}},
                     fn=boom, state=ToolState.ACTIVE),
            ToolSpec(name="slow_x", description="d", parameters={"type": "object",
                     "properties": {"ms": {"type": "integer"}}},
                     fn=slow, state=ToolState.ACTIVE),
        ):
            svc.registry.register(spec)

        svc.tool_call("echo_x", {"text": "hello", "n": 2})
        svc.tool_call("boom_x", {"text": "a-very-specific-argument"})
        svc.tool_call("slow_x", {"ms": 40})
        svc.tool_call("echo_x", {})

        got = rows(svc)
        by_tool = {}
        for r in got:
            by_tool.setdefault(r["tool"], []).append(r)

        print("-- the row carries the arguments")
        first = by_tool["echo_x"][0]
        check("an argument name is recorded", "args" in first, str(first))
        check("a string argument keeps its value when it is short",
              (first.get("args", {}).get("text") or {}).get("value") == "hello", str(first.get("args")))
        check("an integer argument keeps its value and its type",
              (first.get("args", {}).get("n") or {}).get("value") == 2, str(first.get("args")))
        check("the type is written even when the value is short",
              (first.get("args", {}).get("n") or {}).get("type") == "int", str(first.get("args")))

        print("-- the row carries the failure")
        b = by_tool["boom_x"][0]
        check("ok is false", b["ok"] is False, str(b))
        check("the error text is on the row", "cannot handle" in str(b.get("error") or ""), str(b))
        check("the failing argument is on the row",
              "a-very-specific-argument" in json.dumps(b.get("args"), ensure_ascii=False), str(b.get("args")))

        print("-- the row carries the duration")
        s = by_tool["slow_x"][0]
        check("ms is present", "ms" in s, str(s))
        check("ms is in milliseconds, not seconds", float(s.get("ms") or 0) >= 25.0, str(s.get("ms")))
        check("a fast call still records a positive duration",
              float(first.get("ms") or 0) > 0, str(first.get("ms")))

        print("-- a call with no arguments says so by saying nothing")
        empty = by_tool["echo_x"][1]
        check("no `args` key when there were no arguments", "args" not in empty, str(empty))

        print("-- a big argument is summarised, not copied")
        svc.registry.register(ToolSpec(
            name="big_x", description="d",
            parameters={"type": "object", "properties": {"code": {"type": "string"}}},
            fn=lambda code="": str(len(code)), state=ToolState.ACTIVE))
        payload = "x" * 5000
        svc.tool_call("big_x", {"code": payload})
        big = by_tool.get("big_x") or [r for r in rows(svc) if r["tool"] == "big_x"]
        big = big[-1]
        arg = (big.get("args") or {}).get("code") or {}
        check("the full value is not on the row", len(json.dumps(big)) < 2000, f"{len(json.dumps(big))} chars")
        check("its true size is on the row", arg.get("chars") == 5000, str(arg))
        check("a prefix is kept so the call is recognisable", len(str(arg.get("head") or "")) > 0, str(arg))
        check("the prefix is bounded", len(str(arg.get("head") or "")) <= Service.CALL_ARGS_CHARS, str(len(str(arg.get("head") or ""))))

        print("-- many arguments do not fill the row")
        many = {f"k{i}": i for i in range(40)}
        svc.registry.register(ToolSpec(
            name="many_x", description="d",
            parameters={"type": "object", "properties": {k: {"type": "integer"} for k in many}},
            fn=lambda **kw: "ok", state=ToolState.ACTIVE))
        svc.tool_call("many_x", many)
        m = [r for r in rows(svc) if r["tool"] == "many_x"][-1]
        check("the argument count is capped",
              len([k for k in (m.get("args") or {}) if k != "…"]) <= Service.CALL_ARGS_MAX_KEYS,
              str(len(m.get("args") or {})))
        check("the overflow is counted rather than dropped",
              (m.get("args") or {}).get("…", {}).get("items") == 40 - Service.CALL_ARGS_MAX_KEYS,
              str((m.get("args") or {}).get("…")))

        print("-- the row is still readable by the old reader")
        imported = svc.store.last_called_by_tool()
        check("tool_usage still sees every tool", set(by_tool) <= set(imported) | {"big_x", "many_x"},
              f"{sorted(imported)}")

        print()
        print("TOTAL FAILURES:", len(FAILS), FAILS)
        return 0 if not FAILS else 1
    finally:
        try:
            svc.store.close()
            svc.missions.close()
        except Exception:                                      # noqa: BLE001
            pass
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
