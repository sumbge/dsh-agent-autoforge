"""Autoforge service -- line-delimited JSON-RPC over stdio.

One request per line, one response per line. No framing beyond newline, so a
Node parent can write JSON and read JSON without a length header.

    {"id": 1, "method": "remember", "params": {"key": "k", "value": "v"}}
    {"id": 1, "ok": true, "result": ...}
    {"id": 1, "ok": false, "error": "..."}

The point of this file is portability: everything it calls is the ported
autoforge code (store / mission / skills / runs), with no dependency on the
agent loop, the CLI, or the web console. The agent remains a separate process.
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from typing import Any, Callable

from .store import ToolStore
from .mission import MissionStore
from . import skills as skills_mod
from .notify import Notifier, channels_from_config
from .forge.sandbox import Sandbox
from .tools.registry import ToolRegistry
from .tools.spec import ToolSpec, ToolState, TriggerProbe


def _home() -> str:
    return os.environ.get("AUTOFORGE_HOME") or os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "autoforge")


def _dsh_home() -> str:
    """Where DSH keeps its own state, for the two things read out of it.

    `DSH_HOME` wins because that is what the harness itself honours; the
    fallback is the same default it uses. Reading a fixed `~/.dsh` would break
    every install that moved its home, and would do it silently -- the failure
    would arrive as "no gateway key", which reads like the gateway is missing.
    """
    return os.environ.get("DSH_HOME") or os.path.join(
        os.path.expanduser("~"), ".dsh")


def _gateway_key() -> str:
    """The bearer DSH's own OpenAI-compatible gateway expects.

    That gateway (`vendor/channel-pack/src/openai-gateway`) is the reason this
    service does not need a credential of its own: DSH is already signed in,
    already holds the account pool, and already routes `xdpool`,
    `xdpool-global` and the free lane. A second key would be a second bill for
    the same work.

    The key is persistent -- written once to
    `$DSH_HOME/openai-gateway/api-key` -- unlike the pool shim's secret, which
    is `randomBytes(32)` per process and therefore unreachable from here. That
    difference is the whole reason this path exists.

    Three sources, in the order the harness itself resolves them: the env var
    it documents, then the file, then empty. Empty is returned rather than
    raised so the caller can produce one error naming every place it looked.
    """
    configured = (os.environ.get("DSH_OPENAI_GATEWAY_API_KEY") or "").strip()
    if configured:
        return configured
    path = os.path.join(_dsh_home(), "openai-gateway", "api-key")
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _forge_route() -> tuple[str, str, str]:
    """The (model, base_url, api_key) the forge pipeline talks to.

    Defaults point at DSH's gateway on the port its own config documents, with
    a model id in the `provider/model` form that gateway serves. Each is
    overridable, because the gateway's port is configurable on the DSH side
    (`DSH_OPENAI_GATEWAY_PORT`) and an override here is cheaper than a code
    change every time someone moves it.
    """
    model = os.environ.get("AUTOFORGE_MODEL", "xdpool-global/deepseek-v4.1-flash")
    base = os.environ.get("AUTOFORGE_BASE_URL", "http://127.0.0.1:8326/v1")
    key = os.environ.get("AUTOFORGE_API_KEY") or _gateway_key()
    return model, base, key


def _notify_channels_from_config() -> tuple[list[Any], list[str]]:
    """Notification channels from the same config file the CLI reads.

    `~/.autoforge/config.json` is written by `auto setup` and is where a person
    puts `notify.channels`. Reading it here rather than taking channels from
    the plugin config keeps one source of truth: a channel added for the CLI
    works in a DSH session, and vice versa.

    A missing or unreadable file is not an error -- it means "no channels",
    which `notify_channels` reports honestly and `notify_send` refuses with a
    message that says where to add them.
    """
    try:
        from . import configfile
        return channels_from_config(configfile.load())
    except Exception as exc:                                # noqa: BLE001
        return [], [f"could not read the notification config: "
                    f"{type(exc).__name__}: {exc}"]


class Service:
    def _persist_tool_state(self, spec: Any) -> None:
        """Write a tool's state change through to the store.

        Retiring or quarantining a tool is a judgement about the tool, and a
        judgement that lasts only until the process exits is not one: the next
        session would load the tool back as `active`. Measured before this
        existed -- `retire_tool` said ok, the registry said retired, and the
        database still had the row as active.
        """
        try:
            self.store.save_tool(spec, allow_overwrite=True)
        except Exception:                                      # noqa: BLE001
            pass

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.path.join(_home(), "autoforge.db")
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.store = ToolStore(self.db_path)
        self.missions = MissionStore(self.db_path)
        # Matches the CLI's own choice (`autoforge/cli.py`: `Sandbox(timeout=30.0)`).
        # `Sandbox()` alone takes the class default of 10.0s, which is drawn for a
        # single probe rather than for `run_python` doing real work -- a directory
        # walk, a bench, an install -- and it is killed mid-write with the caller
        # told only that it timed out. The bound is still a bound; it is just the
        # one the CLI actually uses.
        self.sandbox = Sandbox(timeout=30.0)
        # Reaching a human. Built from the config at construction for the same
        # reason autoforge's agent builds it there: a config entry says how to
        # reach someone, not that a message is owed. Nothing is sent until
        # `notify_send` is called, and an unreadable channel is carried as a
        # problem so it shows up in `notify_channels` rather than raising at
        # the moment someone needs to be told something.
        self.notify_channels_cfg, self._notify_problems = (
            _notify_channels_from_config())
        self.notifier = Notifier(self.notify_channels_cfg)
        self.registry = ToolRegistry()
        self.registry.on_state_change = self._persist_tool_state
        for name, rec in self.store.load_all_tools().items():
            try:
                # `runner=`, not `fn=`. The sandbox bridge is an out-of-process
                # executor, and `ToolRegistry.call` prefers `runner` precisely
                # because it may answer with its own `ToolResult`. Handing it in
                # as `fn` made the registry call it as `fn(**arguments)` and
                # then json-encode whatever came back -- the very laundering the
                # `_runner_for` docstring describes. It happened to look right
                # only because both shapes end in a string.
                #
                # The entry symbol comes from the code, not from a constant:
                # see `_entry_of` for the restart-time failure that caused.
                entry = self._entry_of(rec.code, rec.name)
                self.registry.register(
                    rec.to_spec(runner=self._runner_for(rec.code, entry))
                )
            except Exception:
                # A row that cannot be rebuilt must not stop the service from
                # starting: the other tools and every mission are still owed.
                continue
        self.methods: dict[str, Callable[..., Any]] = {
            "ping": self.ping,
            "describe": self.describe,
            "context_brief": self.context_brief,
            "remember": self.remember,
            "recall": self.recall,
            "forget": self.forget,
            "mission_open": self.mission_open,
            "mission_note": self.mission_note,
            "mission_close": self.mission_close,
            "mission_list": self.mission_list,
            "mission_show": self.mission_show,
            "mission_focus": self.mission_focus,
            "mission_block": self.mission_block,
            "mission_sweep": self.mission_sweep,
            "mission_wake": self.mission_wake,
            "mission_history": self.mission_history,
            "skill_errors": self.skill_errors,
            "skill_forget": self.skill_forget,
            "install_system_task": self.install_system_task,
            "system_task_command": self.system_task_command,
            "log_event": self.log_event,
            "events": self.events,
            "report": self.report,
            "skills": self.skills,
            "skill_view": self.skill_view,
            "skill_write": self.skill_write,
            "my_history": self.my_history,
            "verify_ledger": self.verify_ledger,
            "tool_usage": self.tool_usage,
            "forge_tool": self.forge_tool,
            "evaluate_tool": self.evaluate_tool,
            "evolve_tool": self.evolve_tool,
            "find_gaps": self.find_gaps,
            "retire_tool": self.retire_tool,
            "list_tools": self.list_tools,
            "run_python": self.run_python,
            "schedule_add": self.schedule_add,
            "schedule_list": self.schedule_list,
            "runs_status": self.runs_status,
            "resume_run": self.resume_run,
            "runs_forget": self.runs_forget,
            "skill_port": self.skill_port,
            "schedule_tick": self.schedule_tick,
            "schedule_run_due": self.schedule_run_due,
            "schedule_done": self.schedule_done,
            "schedule_cancel": self.schedule_cancel,
            "tool_define": self.tool_define,
            "tool_list": self.tool_list,
            "tool_call": self.tool_call,
            # ── the parts autoforge has that DSH has no equivalent for ──
            "my_capabilities": self.my_capabilities,
            "gpu_probe": self.gpu_probe,
            "gpu_occupancy": self.gpu_occupancy,
            "gpu_units_audit": self.gpu_units_audit,
            "gpu_verify": self.gpu_verify,
            "gpu_compile": self.gpu_compile,
            "gpu_bench": self.gpu_bench,
            "cpu_probe": self.cpu_probe,
            "cpu_runs_here": self.cpu_runs_here,
            "cpu_preflight": self.cpu_preflight,
            "cpu_units_audit": self.cpu_units_audit,
            "cpu_cache_stats": self.cpu_cache_stats,
            "cpu_compile": self.cpu_compile,
            "cpu_run_isolated": self.cpu_run_isolated,
            "cpu_tune": self.cpu_tune,
            "notify_send": self.notify_send,
            "notify_channels": self.notify_channels,
        }

    # -- the small ones, verbatim from bytecode ---------------------------

    def ping(self) -> dict[str, Any]:
        return {"ok": True, "db": self.db_path}

    def describe(self) -> dict[str, Any]:
        return {"methods": sorted(self.methods), "db": self.db_path}

    def remember(self, key: str, value: str, tags: str = "") -> Any:
        self.store.remember(key, value, tags)
        return {"key": key}

    def recall(self, query: str = "", limit: int = 20) -> Any:
        return self.store.recall(query, limit)

    def forget(self, key: str) -> Any:
        return {"forgotten": self.store.forget(key)}

    # -- the per-turn brief -------------------------------------------------
    # agent.py carried these into every request: a kept fact the model has to
    # remember to ask for is identical, from its side, to a fact never kept.
    # Ported with agent.py's own budgets, because the numbers are what keep the
    # block from crowding out the conversation.

    MEMORY_SLOTS = 20
    MEMORY_ENTRY_CHARS = 240
    MEMORY_BUDGET_CHARS = 1200
    # The skills index is a menu, not a body of text: 90-char when_to_use
    # excerpts for every skill, capped so a large library cannot push the
    # missions out of the block.
    SKILL_BUDGET_CHARS = 700
    # The identity block is shorter than the others on purpose. What belongs
    # here is only the sentences that change what the model DOES: that it can
    # touch the filesystem, that its memory outlives the session, and that the
    # approval stack is DSH's. Numbers and paths live behind my_capabilities,
    # which the model calls when it needs them -- carrying them in every request
    # would cost 1000 chars to save one tool call.
    #
    # 600, not 480: the block is 523 chars once the host's reach strings are
    # substituted, and a cap below that truncated it mid-sentence ("when a
    # boundary m"). A budget that always fires is not a budget, it is a bug with
    # a number in it -- leave room for a longer host string on another machine.
    IDENTITY_BUDGET_CHARS = 600

    def _memory_lines(self) -> list[str]:
        """The facts kept, carried into every request."""
        try:
            rows = self.store.memory_for_injection(self.MEMORY_SLOTS)
        except Exception:                                      # noqa: BLE001
            return ["- Kept facts: unreadable this turn."]
        if not rows:
            return ["- Kept facts: none kept yet (remember() keeps one)."]
        head = "- Kept facts (deliberate memory — kept by me, survives restart):"
        body: list[str] = []
        used = 0
        for r in rows:
            value = " ".join((r.get("value") or "").split())
            if len(value) > self.MEMORY_ENTRY_CHARS:
                value = value[: self.MEMORY_ENTRY_CHARS] + " …"
            line = f"    {r.get('key')}: {value}"
            # `body and` guards the first line: a single oversized entry is
            # still shown, truncated, rather than producing an empty section
            # that reads as "I kept nothing".
            if body and used + len(line) > self.MEMORY_BUDGET_CHARS:
                body.append(f"    …and {len(rows) - len(body)} more (recall() lists them)")
                break
            body.append(line)
            used += len(line)
        return [head, *body]

    def _mission_lines(self) -> list[str]:
        """What I owe, next to what I know."""
        try:
            return self.missions.report().splitlines()
        except Exception as exc:                               # noqa: BLE001
            return [f"MISSION — unreadable: {exc}"]

    def context_brief(self, budget_chars: int = 4000) -> dict[str, Any]:
        """The block a host injects into every request, as plain text.

        Exists because the first ported version left the memory behind a tool:
        the model could `recall` it, which means it had to decide to look. The
        complaint that produced agent.py's per-turn block was that a session
        with many tasks forgets the one sentence they are all for -- and an
        agent that has forgotten to look somewhere will not look.

        Every call is counted on the ledger. That is not bookkeeping for its own
        sake: whether this block reaches the model is a claim about the host's
        wiring, and the only way to check it from outside is to see the calls.
        An empty `brief_served` count in a session that has run turns means the
        plugin's refresh timer never fired, which is a different fault from the
        block rendering empty -- and before this counter the two were
        indistinguishable.
        """
        self._record("brief_served", {"budget_chars": budget_chars})
        lines = self._memory_lines()
        lines.append("")
        lines.extend(self._mission_lines())
        try:
            # `registry.report()` is the registry's own accounting, and the same
            # call `list_tools` renders -- counting states here by hand would be
            # a second opinion that can disagree with the tool's own output.
            rep = self._registry().report()
            by_state = rep.get("by_state") or {}
            active = by_state.get("active", 0)
            total = rep.get("total", 0)
            detail = ", ".join(f"{k}={v}" for k, v in sorted(by_state.items())) or "none"
            lines.append("")
            lines.append(
                f"TOOLS — {active} active of {total} forged ({detail}). "
                "forge_tool creates one; list_tools lists them."
            )
        except Exception as exc:                               # noqa: BLE001
            lines.append(f"TOOLS — unreadable: {exc}")
        lines.extend(self._skill_lines())
        lines.extend(self._identity_lines())
        text = "\n".join(lines).strip()
        if len(text) > budget_chars:
            text = text[:budget_chars].rstrip() + "\n    …(brief truncated)"
        return {"text": text, "chars": len(text)}

    def _identity_lines(self) -> list[str]:
        """What this agent is, in the sentences that change its behaviour.

        The gap this closes: `my_capabilities` computes reach from the sandbox
        object, and until now that answer only existed inside a tool's return
        value -- so a model that never called the tool reasoned about its own
        reach from its tool list, and got it wrong in a specific direction. The
        persona in the preset argues against that at length ("'I have no file
        tools' is false"), but prose loses to the prior: an agent will explain
        at length that it has no hands while holding them.

        So the three facts that decide whether it reaches for `run_python` or
        apologises instead are carried here, measured from `Sandbox.reach()`
        rather than restated. Everything else -- paths, timeouts, counts -- stays
        behind the tool, because a block that repeats itself every turn is a
        block the model learns to skim.

        Degrades to one line, like the others: an identity that cannot be read
        must not cost the facts and missions above it.
        """
        try:
            reach = self.sandbox.reach(probe=False)
        except Exception as exc:                               # noqa: BLE001
            return ["", f"WHO I AM — unreadable this turn ({type(exc).__name__})."]
        fs = str(reach.get("filesystem") or "")
        net = str(reach.get("network") or "")
        host = str(reach.get("host") or "")
        lines = ["", "WHO I AM — measured from the sandbox, not declared:"]
        # Short, because these two are the ones that get denied in the wrong
        # direction. `read-write, whole host` is not a boast; it is the sentence
        # that stops "I cannot read that file" from being said out loud.
        lines.append(f"    forged code runs {host}")
        lines.append(f"    filesystem {fs}; network {net}")
        lines.append("    Read a file or run a command NOW with run_python; keep")
        lines.append("    the capability with forge_tool. Lacking a tool for")
        lines.append("    something is not the same as being unable to do it.")
        lines.append("    Approval and sandbox limits are DSH's, not autoforge's;")
        lines.append("    call my_capabilities when a boundary actually matters.")
        text = "\n".join(lines)
        if len(text) > self.IDENTITY_BUDGET_CHARS:
            text = text[:self.IDENTITY_BUDGET_CHARS].rstrip() + "\n    …(identity truncated)"
        return text.split("\n")

    def _skill_lines(self) -> list[str]:
        """The procedures written down, named in every request.

        A procedure the model must remember to ask for is, from its side,
        identical to one that was never written: it re-derives the same
        mistakes and the library never helps. `skill_view` still loads the
        body -- this is the menu, not the text, because a body per skill would
        crowd out the missions.

        Built from `self.skills()`, the same list `skill_list` renders. Note
        that this service has no SkillStore object: `skills` is a *method* on
        Service (it scans the directories per call), so `self.skills.menu()`
        is an AttributeError -- measured, out of curiosity, before this line
        was written the other way.
        """
        try:
            rows = self.skills()
        except Exception:                                      # noqa: BLE001
            # A skills directory that cannot be read must not cost the facts
            # and missions above it, so this degrades to one line.
            return ["", "SKILLS — unreadable this turn."]
        if not rows:
            return ["", "SKILLS — none written yet (skill_write saves one)."]
        head = ("SKILLS — procedures written down (skill_view loads one; "
                "bodies are not carried here):")
        body: list[str] = []
        used = 0
        for r in rows:
            name = r.get("name") or "?"
            when = " ".join((r.get("when_to_use") or r.get("description") or "").split())
            if len(when) > 90:
                when = when[:90].rstrip() + " …"
            line = f"    {name}" + (f" -- {when}" if when else "")
            if body and used + len(line) > self.SKILL_BUDGET_CHARS:
                body.append(f"    …and {len(rows) - len(body)} more (skill_list names them)")
                break
            body.append(line)
            used += len(line)
        return ["", head, *body]

    # -- missions ---------------------------------------------------------

    def mission_open(self, text: str, parent: int = 0, next_step: str = "",
                     blocked_on: str = "") -> Any:
        m = self.missions.open(text, parent, next_step, blocked_on)
        return _mission(m)

    def mission_note(self, mid: int, note: str = "", next_step: str = "",
                     blocked_on: str = "") -> Any:
        return _mission(self.missions.note(mid, note, next_step, blocked_on))

    def mission_close(self, mid: int, note: str = "", ok: bool = True) -> Any:
        if ok:
            return _mission(self.missions.finish(mid, note))
        return _mission(self.missions.drop(mid, note))

    def mission_list(self, status: str = "open") -> Any:
        return [_mission(m) for m in self.missions.all(status)]

    def mission_show(self, mid: int) -> Any:
        return _mission(self.missions.get(mid))

    def mission_focus(self, mid: int) -> Any:
        self.missions.set_focus(mid)
        return {"focus": mid, "focused": self.missions.focused()}

    def mission_block(self, mid: int, waiting_on: str) -> Any:
        return _mission(self.missions.note(
            mid, f"blocked on: {waiting_on}", None, blocked_on=waiting_on))

    def mission_sweep(self, idle_hours: float = 24) -> Any:
        idle_s = float(idle_hours) * 3600.0
        result = self.missions.sweep(idle_s)
        report = self.missions.assumption_report(idle_s)
        return {"result": result, "report": report}

    def mission_wake(self, mid: int, note: str = "") -> Any:
        from .mission import MissionError
        try:
            return _mission(self.missions.wake(mid, note))
        except MissionError as exc:
            return {"mid": mid, "woken": False, "error": str(exc)}

    def mission_history(self, mid: int, limit: int = 50) -> Any:
        return {"mid": mid, "history": self.missions.history(mid, limit)}

    # -- skills -----------------------------------------------------------

    def skill_errors(self) -> Any:
        out: list[Any] = []
        for kind, directory in skills_mod.default_skill_dirs():
            if not os.path.isdir(directory):
                continue
            for path in skills_mod.iter_skill_files(directory):
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except Exception as exc:                       # noqa: BLE001
                    out.append({"kind": kind, "path": str(path),
                                "error": f"{type(exc).__name__}: {exc}"})
                    continue
                fm, _body = skills_mod.split_frontmatter(text)
                if not (fm or {}).get("description"):
                    out.append({"kind": kind, "path": str(path),
                                "error": "missing frontmatter description"})
        return out

    def skill_forget(self, name: str) -> Any:
        import shutil
        for _kind, directory in skills_mod.default_skill_dirs():
            if not os.path.isdir(directory):
                continue
            for path in skills_mod.iter_skill_files(directory):
                label = (path.parent.name if path.name.lower() == "skill.md"
                         else path.stem)
                if label.lower() != name.lower():
                    continue
                archive = os.path.join(directory, ".archive")
                os.makedirs(archive, exist_ok=True)
                target = os.path.join(archive, path.name)
                shutil.move(str(path), target)
                return {"name": name, "archived": target}
        return {"name": name, "error": "not found"}

    def install_system_task(self, interval_minutes: int = 15, name: str = "") -> Any:
        from . import schedule as schedule_mod
        try:
            return {"ok": True, "result": schedule_mod.install_system_task(
                int(interval_minutes), name)}
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def system_task_command(self, interval_minutes: int = 15, name: str = "") -> Any:
        from . import schedule as schedule_mod
        return {"command": schedule_mod.system_task_command(int(interval_minutes), name)}

    # -- ledger -----------------------------------------------------------

    def log_event(self, kind: str, payload: dict[str, Any] | None = None) -> Any:
        self.store.log_event(kind, payload or {})
        return {"logged": kind}

    def events(self, limit: int = 20) -> Any:
        return self.store.get_events(limit)

    def report(self) -> Any:
        return self.store.report()

    def my_history(self, limit: int = 20, kind: str = "") -> Any:
        try:
            n = max(1, min(int(limit), 500))
        except Exception:                                      # noqa: BLE001
            n = 20
        events = self.store.get_events(limit=n)
        if kind:
            events = [e for e in events if e.get("kind") == kind]
        return {"count": len(events), "events": events}

    def verify_ledger(self) -> Any:
        try:
            return {"ok": True, "result": self.store.verify_chain()}
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def tool_usage(self) -> Any:
        try:
            return {"ok": True, "last_called": self.store.last_called_by_tool()}
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _skill_files(self) -> list[Any]:
        found: list[Any] = []
        for _kind, directory in skills_mod.default_skill_dirs():
            if not os.path.isdir(directory):
                continue
            for f in skills_mod.iter_skill_files(directory):
                found.append(f)
        return found

    def _skill_library(self) -> Any:
        """A `SkillLibrary` over the same directories, backed by this store.

        Why a library and not another directory walk: the previous `skills()`
        read the files itself and returned four fields, so `tags` and `loads`
        never reached the caller -- and `[3x]` in the menu, or a `query` that
        matches a tag, were both impossible no matter what the tool declared.
        `SkillLibrary.scan()` indexes the files into `skills` and merges the
        stored counters back on, which is exactly the join that was missing.

        Built per call rather than cached, matching `_skill_files()`: a skill
        written by another session between two calls must show up on the next
        one, and a cached index would hide it until a restart.
        """
        return skills_mod.SkillLibrary(
            self.store, dirs=skills_mod.default_skill_dirs())

    def skills(self, query: str = "") -> list[Any]:
        """Every skill, most-used first, optionally filtered.

        `query` matches name, description, when_to_use and tags, case-insensitive
        -- the same four fields the source agent's `skill_list` searched, copied
        rather than reinvented so the two cannot disagree about what "matches".

        Still returns a list of dicts, not the formatted string the source agent
        returned. `context_brief._skill_lines()` iterates this list every turn,
        and `rpcTool` hands the value straight to the model; a string would have
        to be re-parsed by both. The empty-case sentence ("No skills match
        'x'.") is therefore not produced here either -- an empty list plus the
        `query` echoed back says the same thing without a second shape.
        """
        lib = self._skill_library()
        lib.scan()
        found = lib.all()
        needle = (query or "").strip().lower()
        out: list[Any] = []
        for s in found:
            if needle:
                haystack = " ".join([
                    s.name, s.description or "", s.when_to_use or "",
                    " ".join(s.tags or []),
                ]).lower()
                if needle not in haystack:
                    continue
            out.append({"name": s.name,
                        "description": s.description or "",
                        "when_to_use": s.when_to_use or "",
                        "tags": list(s.tags or []),
                        "loads": int(getattr(s, "loads", 0) or 0),
                        "path": str(s.path)})
        return out

    def skill_view(self, name: str) -> Any:
        for f in self._skill_files():
            label = f.parent.name if f.name.lower() == "skill.md" else f.stem
            if label != name:
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                self.store.skill_load(name)
            except Exception:                                  # noqa: BLE001
                continue
            fm, body = skills_mod.split_frontmatter(text)
            return {"name": name, "frontmatter": fm or {}, "body": body}
        return {"name": name, "error": "not found"}

    def skill_write(self, name: str, description: str = "", body: str = "",
                    when_to_use: str = "", tags: str = "") -> Any:
        if not skills_mod.valid_name(name):
            return {"name": name,
                    "error": "invalid name: lower-case letters, digits, '-' or '_' (max 64)"}
        if not description.strip():
            return {"name": name, "error": "description is required"}
        if not body.strip():
            return {"name": name, "error": "body is required"}
        dirs = skills_mod.default_skill_dirs()
        if not dirs:
            return {"name": name, "error": "no skill directory configured"}
        target_dir = dirs[0][1]
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, f"{name}.md")
        text = skills_mod.render_skill_text(
            name, description.strip(), when_to_use.strip(),
            skills_mod.parse_tags(tags), body)
        existed = os.path.exists(path)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return {"name": name, "path": path,
                "created": not existed, "updated": existed}

    # -- forge / tools ----------------------------------------------------

    def list_tools(self) -> dict[str, Any]:
        rep = self._registry().report()
        if not rep["tools"]:
            return {"text": "No tools yet.", "total": 0, "by_state": {}}
        lines = []
        for s in rep["tools"]:
            lines.append(
                f"  {s['name']} [{s['state']}] sr={s['stats']['success_rate']} "
                f"calls={s['stats']['calls']} -- {(s.get('description') or '')[:60]}")
        return {"text": f"Tools ({rep['total']}, states {rep['by_state']}):\n" + "\n".join(lines),
                "total": rep["total"], "by_state": rep["by_state"]}

    # -- forge ------------------------------------------------------------

    def _forge_pipeline(self) -> Any:
        """The forge pipeline, wired exactly as agent.py wires it.

        Constructed lazily on first use, because it needs the model client and
        nothing else in this service does -- a service that starts a gateway
        connection on boot would fail to start whenever the network is down,
        and every mission and skill in the store is still owed then.
        """
        built = getattr(self, "_pipeline", None)
        if built is not None:
            return built
        from .core.llm import DEFAULT_MAX_TOKENS, OpenAICompatClient
        from .autonomy.policy import AutonomyPolicy
        from .forge.generator import LLMToolGenerator
        from .forge.pipeline import ForgeConfig, ForgePipeline
        from .forge.verifier import ToolVerifier

        model, base, key = _forge_route()
        if not key:
            raise RuntimeError(
                "no credential for the forge model: DSH's OpenAI gateway key is "
                "missing. Looked at $DSH_OPENAI_GATEWAY_API_KEY, $AUTOFORGE_API_KEY, "
                "and %s. Start DSH once so the gateway writes its key, or point "
                "AUTOFORGE_BASE_URL at another OpenAI-compatible endpoint and set "
                "AUTOFORGE_API_KEY." % os.path.join(_dsh_home(), "openai-gateway", "api-key"))
        # `timeout=600` is the CLI's own choice (`cli._client_for`): one forge is
        # six checks, several of which run a probe agent through a whole
        # conversation, and a timeout drawn for a single turn would kill the
        # round rather than the request.
        llm = OpenAICompatClient(model=model, base_url=base, api_key=key,
                                 timeout=600)
        policy = AutonomyPolicy()
        self.registry.policy = policy
        forge_config = ForgeConfig()
        verifier = ToolVerifier(
            llm, sandbox=self.sandbox,
            run_execution_check=forge_config.require_execution,
            run_trigger_check=forge_config.require_trigger,
            run_negative_check=forge_config.require_negative,
            run_claim_check=getattr(forge_config, "require_claim", True),
        )

        def event(kind: str, payload: dict[str, Any]) -> None:
            try:
                self.store.log_event(kind, payload)
            except Exception:                                  # noqa: BLE001
                pass

        built = ForgePipeline(
            # `LLMToolGenerator`, not `TemplateGenerator`. The template one is
            # an offline demo stub: it answers every need with `echo_<need>`
            # whose body is `return text`, and that stub then passes the whole
            # battery -- 19/19 robustness, 5/5 adversarial, trigger fires --
            # because every check feeds it input it echoes back. Sealing it as
            # `active` is the exact failure the framework exists to prevent,
            # and it is what this service did until the generator was swapped.
            # The CLI has always used the LLM generator (`cli._build`), so the
            # service was the odd one out.
            LLMToolGenerator(llm, max_tokens=DEFAULT_MAX_TOKENS),
            verifier, self.registry,
            config=forge_config, sandbox=self.sandbox,
            policy=policy, on_event=event,
        )
        self._pipeline = built
        return built

    def _registry(self) -> Any:
        self._forge_pipeline()
        return self.registry

    # -- the forge gate -----------------------------------------------------
    # agent.py's forge_tool does three things before it calls the pipeline, and
    # the service did none of them. Ported here verbatim in behaviour, because
    # each one exists for a measured reason and the reasons are in the docstrings:
    #
    #   _tool_budget_report  a forge is a permanent per-turn cost, so an
    #                        over-budget library makes the forge justify itself
    #   _arbitrate_forge     rank what the library already holds for this need
    #                        and put that ranking in front of the generator
    #   _prelookup_market    ask the shelf before building, not only after
    #
    # What is deliberately NOT ported: the arbiter's veto. It was removed in
    # agent.py after the ledger refuted it (7/8 duplicates caught, 95/109
    # legitimate forges blocked) -- the duplicate is refused exactly and cheaply
    # in the pipeline, on the generated name, before verification. `replace=True`
    # had quietly re-opened that hole from the other side: with it on, a name
    # clash was neither refused nor recorded, it was overwritten.

    TOOL_SCHEMA_BUDGET_CHARS = 6000

    _NAME_STOPWORDS = frozenset(
        "the and for with from into when then your have will does each some more "
        "this that need tool which what over under than make made used using".split())

    _LEXICAL_STOPWORDS = frozenset(
        "with this that from into when then your have will does each some more "
        "need tool which what over under than make made used using".split())

    _DISCOVER_WORDS_AT_MOST = 2

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        """One event on the ledger. Best-effort, like agent.py's.

        The service has no in-memory trace -- there is no run to have a
        timeline of -- so this is the store leg only. A store that is locked or
        full must not turn a finished forge into a failed one: the tool is
        already registered by the time this runs, and the cost of a lost event
        is a gap in the ledger, not a broken task.
        """
        try:
            self.store.log_event(kind, payload)
        except Exception:                                      # noqa: BLE001
            pass

    def _tool_schema_chars(self, *, own_only: bool = True) -> int:
        """Total characters of tool schema that ride in every prompt.

        `own_only` counts what this agent forged, not the frozen builtin
        baseline: shipping tools are not a library decision and no retire_tool
        can give that context back, so counting them put every machine with the
        full tool set permanently over the cap. Settled in agent.py 2026-09-15.
        """
        total = 0
        try:
            names = self.registry.names()
        except Exception:                                      # noqa: BLE001
            return 0
        for name in names:
            spec = self.registry.get(name)
            if spec is None:
                continue
            if own_only and (getattr(spec, "source", "") or "") == "builtin":
                continue
            try:
                total += len(json.dumps(spec.schema, ensure_ascii=False))
            except (TypeError, ValueError):
                total += len(getattr(spec, "name", "") or "") + len(getattr(spec, "description", "") or "")
        return total

    def _tool_budget_report(self) -> str:
        """Empty when there is room; a warning when the forge should justify itself.

        Returns a string rather than refusing, because the right response to an
        over-budget library is not to block the work -- the need may be real --
        but to make the agent name what it is trading away.
        """
        used = self._tool_schema_chars()
        if used <= self.TOOL_SCHEMA_BUDGET_CHARS:
            return ""
        return (
            f"Tool schema budget exceeded: {used} chars in the prompt every "
            f"turn, over the {self.TOOL_SCHEMA_BUDGET_CHARS}-char budget. Forging "
            "another adds to a cost that is paid on every request, not once. "
            "Retire a tool you no longer need (retire_tool) and forge again, or "
            "state which existing tool this one is worth replacing and why."
        )

    def _arbiter(self) -> Any:
        """Lazily built, because the registry is built lazily too."""
        got = getattr(self, "_arbiter_obj", None)
        if got is None:
            from autoforge_svc.route import Arbiter, BehaviourRouter

            got = Arbiter(BehaviourRouter(self.registry), self.registry)
            self._arbiter_obj = got
        return got

    def _arbitrate_forge(self, need: str) -> Any:
        """Ask the router which existing tool owns this need, and record it.

        Never raises. An arbiter that cannot answer returns no coverage, which
        licenses the forge: the failure mode of a broken gate must be the old
        behaviour, not a library that cannot be extended.
        """
        try:
            verdict = self._arbiter().decide(need)
        except Exception as exc:                               # noqa: BLE001
            self._record("arbitration", {"need": need, "ok": False,
                                         "error": f"{type(exc).__name__}: {exc}"})
            from autoforge_svc.route import Arbitration

            return Arbitration(reason="the arbiter could not be consulted")
        self._record("arbitration", {
            "need": need, "ok": True,
            # No "refused"/"veto": the arbiter ranks, it does not veto.
            "reason": getattr(verdict, "reason", ""),
            "ranked": [c.name for c in getattr(verdict, "ranked", ())[:5]],
        })
        return verdict

    # -- the pre-forge market lookup ----------------------------------------

    @staticmethod
    def _canon_token(token: str) -> str:
        """One word from a name, reduced to what it means, not how it ends."""
        for suffix in ("ing", "es", "s", "ed"):
            if len(token) > 4 and token.endswith(suffix):
                return token[: -len(suffix)]
        return token

    @classmethod
    def _name_tokens(cls, text: str) -> list[str]:
        """A tool name or a need, as comparable lower-case words."""
        return [cls._canon_token(t)
                for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t]

    def _lexical_name_hits(self, need: str, items: list[Any]) -> list[str]:
        """Entries whose *name* is the need's, judged from `/resources` alone.

        Deliberately name-only: matching on description words named unrelated
        tools as overlaps on two-word coincidences (agent.py, 2026-09-16).
        Equality after `_canon_token`, not a prefix and not a substring, so
        `repo` does not match `report`.
        """
        need_tokens = list(self._name_tokens(need))
        need_norm = " ".join(need_tokens)
        need_content = {t for t in need_tokens
                        if len(t) >= 3 and t not in self._NAME_STOPWORDS}
        hits: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if not name:
                continue
            if name.lower().replace("_", " ") in need_norm:
                hits.append(name)          # the whole name, verbatim
                continue
            tokens = {t for t in self._name_tokens(name)
                      if len(t) >= 3 and t not in self._NAME_STOPWORDS}
            if not tokens:
                continue
            if len(tokens) == 1:
                if tokens <= need_content:
                    hits.append(name)
                continue
            if len(tokens & need_content) >= 2:
                hits.append(name)
        return hits

    def _market_search(self, need: str, k: int = 5) -> list[dict[str, Any]]:
        """Semantic hits from the shelf's `/search`, best-effort.

        [] means *no answer*, never *nothing there*: the caller must not read a
        broken search as an empty shelf.
        """
        try:
            import urllib.parse as _parse
            import urllib.request as _url

            base = os.environ.get("TOOLMARKET_URL", "http://127.0.0.1:8000")
            qs = _parse.urlencode({"q": need, "k": int(k)})
            req = _url.Request(base.rstrip("/") + "/search?" + qs)
            budget = float(os.environ.get("TOOLMARKET_SEARCH_TIMEOUT") or 30.0)
            with _url.urlopen(req, timeout=budget) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as exc:                               # noqa: BLE001
            self._record("market_semantic_lookup", {
                "need": need, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return []
        if isinstance(data, dict):
            results = data.get("results") or []
            conf = data.get("confidence") if isinstance(data.get("confidence"), dict) else {}
        else:
            results, conf = (data if isinstance(data, list) else []), {}
        if not isinstance(results, list):
            results = []
        mode = data.get("mode") if isinstance(data, dict) else None
        if conf.get("confident") is False:
            self._record("market_semantic_lookup", {
                "need": need, "ok": True, "confident": False, "mode": mode, "hits": []})
            return []
        scored: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            scored.append({"name": str(item["name"]), "score": score,
                           "description": str(item.get("description") or "")[:110]})
        if scored:
            top = max(x["score"] for x in scored)
            floor = max(0.01, 0.5 * top)
            scored = [x for x in scored if x["score"] >= floor][:k]
        self._record("market_semantic_lookup", {
            "need": need, "ok": True, "confident": True, "mode": mode,
            "hits": [x["name"] for x in scored]})
        return scored

    @staticmethod
    def _peer_market_from_file() -> str | None:
        """``peers.json``, or None when there is no readable file."""
        home = (os.environ.get("AUTOFORGE_HOME")
                or os.path.join(os.path.expanduser("~"), ".autoforge"))
        try:
            with open(os.path.join(home, "peers.json"), encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:                                      # noqa: BLE001
            return None
        if isinstance(data, str):
            return data
        items = data.get("peers", data) if isinstance(data, dict) else data
        if isinstance(items, dict):
            return ",".join(f"{k}={v}" for k, v in items.items())
        if isinstance(items, list):
            return ",".join(str(x) for x in items)
        return None

    @staticmethod
    def _peer_market_from_registry() -> str:
        """Where ``setx`` writes it: HKCU\\Environment."""
        if os.name != "nt":
            return ""
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, "AUTOFORGE_PEER_MARKETS")
            return str(value).strip()
        except Exception:                                      # noqa: BLE001
            return ""

    @classmethod
    def _peer_market_spec_or_none(cls) -> str | None:
        """The precedence chain, with file-vs-environment made visible."""
        env_raw = os.environ.get("AUTOFORGE_PEER_MARKETS")
        file_raw = cls._peer_market_from_file()
        if env_raw is not None:
            if not file_raw:
                return env_raw
            if not env_raw.strip():
                return env_raw
            have = {e.split("=", 1)[0].strip() for e in env_raw.split(",") if e.strip()}
            extra = [e.strip() for e in file_raw.split(",")
                     if e.strip() and e.split("=", 1)[0].strip() not in have]
            return env_raw + (("," + ",".join(extra)) if extra else "")
        if file_raw is not None:
            return file_raw
        return cls._peer_market_from_registry()

    @classmethod
    def _peer_market_urls(cls) -> list[tuple[str, str]]:
        """Other machines' shelves, as ``(label, base_url)``.

        Configured, never discovered: a LAN sweep on every forge is a scan, not
        a lookup. The default market is not a peer of itself -- asking one shelf
        twice under two names is a double count, not redundancy.
        """
        raw = cls._peer_market_spec_or_none() or ""
        peers: list[tuple[str, str]] = []
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" in entry:
                label, url = entry.split("=", 1)
            else:
                label, url = entry, entry
            url = url.strip().rstrip("/")
            if url:
                peers.append((label.strip() or url, url))
        default_base = os.environ.get("TOOLMARKET_URL", "http://127.0.0.1:8000").rstrip("/")
        if default_base:
            peers = [(lab, u) for lab, u in peers
                     if u.split("|FILE:", 1)[0].rstrip("/") != default_base]
        return peers

    @staticmethod
    def _peer_headers(url: str) -> dict[str, str]:
        """Headers for a peer, including its token when one is named."""
        headers = {"User-Agent": "autoforge-peer-lookup/1.0"}
        if "|FILE:" not in url:
            return headers
        _, path = url.split("|FILE:", 1)
        try:
            with open(path.strip(), encoding="utf-8") as fh:
                token = fh.read().strip()
            if token:
                headers["Authorization"] = "Bearer " + token
        except OSError:
            pass
        return headers

    def _peer_lookup(self, need: str) -> tuple[list[dict[str, Any]], bool]:
        """Ask the sibling shelves the same two questions the local one answers.

        A peer that was *reachable* and carried nothing is an answer and
        licenses the forge; a peer that was switched off is no answer and must
        not. A machine going down must not look like a shelf that was searched.
        """
        import time as _time
        import urllib.parse as _parse
        import urllib.request as _url

        hits: list[dict[str, Any]] = []
        answered = False
        now = _time.time()
        dead: dict[str, float] = getattr(self, "_peer_dead_until", None) or {}
        self._peer_dead_until = dead
        for label, url in self._peer_market_urls():
            base = url.split("|FILE:", 1)[0].rstrip("/")
            if dead.get(base, 0.0) > now:
                self._record("market_peer_lookup", {
                    "need": need, "peer": label, "url": base, "ok": False,
                    "error": "skipped: failed recently"})
                continue
            headers = self._peer_headers(url)
            try:
                qs = _parse.urlencode({"q": need, "k": 5})
                req = _url.Request(base + "/search?" + qs, headers=headers)
                with _url.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8", "replace"))
            except Exception as exc:                           # noqa: BLE001
                try:
                    req = _url.Request(base + "/resources", headers=headers)
                    with _url.urlopen(req, timeout=5) as resp:
                        data = json.loads(resp.read().decode("utf-8", "replace"))
                    names = [(x.get("name"), x.get("description") or "")
                             for x in (data.get("resources") or data.get("items") or [])
                             if isinstance(x, dict)]
                    answered = True
                    want = {w for w in self._name_tokens(need) if len(w) > 2}
                    for n, d in names:
                        toks = set(self._name_tokens(n))
                        if toks & want:
                            hits.append({"name": str(n), "peer": label,
                                         "score": 0.0, "description": str(d)[:110]})
                    continue
                except Exception as exc2:                      # noqa: BLE001
                    refused = ("refused" in str(exc).lower()
                               or "refused" in str(exc2).lower()
                               or "getaddrinfo" in str(exc))
                    dead[base] = _time.time() + (120.0 if refused else 300.0)
                    self._record("market_peer_lookup", {
                        "need": need, "peer": label, "url": base, "ok": False,
                        "error": f"{type(exc).__name__}: {exc} / {type(exc2).__name__}: {exc2}"})
                    continue
            results = data.get("results") if isinstance(data, dict) else data
            results = results if isinstance(results, list) else []
            conf = data.get("confidence") if isinstance(data, dict) else {}
            confident = conf.get("confident") if isinstance(conf, dict) else None
            got = []
            for item in results:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                try:
                    score = float(item.get("score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                got.append({"name": str(item["name"]), "peer": label, "score": score,
                            "description": str(item.get("description") or "")[:110]})
            if got and confident is not False:
                top = max(x["score"] for x in got)
                floor = max(0.01, 0.5 * top)
                got = [x for x in got if x["score"] >= floor][:5]
            else:
                got = []
            answered = True
            self._record("market_peer_lookup", {
                "need": need, "peer": label, "url": base, "ok": True,
                "confident": confident, "hits": [x["name"] for x in got]})
            hits.extend(got)
        return hits, answered

    def _discover_upstream(self, need: str) -> list[dict[str, Any]]:
        """Ask the market to look in the public MCP registry. Third leg.

        Best-effort: `[]` means *no answer*, never *nothing there*.
        """
        import urllib.request as _url

        words = [w for w in re.split(r"[^a-z0-9]+", need.lower())
                 if len(w) > 2 and w not in self._LEXICAL_STOPWORDS]
        seen: list[str] = []
        for w in words:
            if w not in seen:
                seen.append(w)
        ranked = sorted(seen, key=lambda w: -len(w))[: self._DISCOVER_WORDS_AT_MOST]
        if not ranked:
            return []
        url = os.environ.get("TOOLMARKET_URL", "http://127.0.0.1:8000")
        found: list[dict[str, Any]] = []
        for word in ranked:
            try:
                body = json.dumps({"q": word, "limit": 8}).encode("utf-8")
                req = _url.Request(url.rstrip("/") + "/discover", data=body,
                                   headers={"Content-Type": "application/json"})
                with _url.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8", "replace"))
            except Exception as exc:                           # noqa: BLE001
                self._record("market_discover", {
                    "need": need, "q": word, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"})
                continue
            items = data.get("results") if isinstance(data, dict) else data
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                found.append({"name": str(item["name"]),
                              "description": str(item.get("description") or ""),
                              "state": str(item.get("state") or "draft"),
                              "callable": bool(item.get("callable"))})
        self._record("market_discover", {"need": need, "ok": True,
                                         "hits": [x["name"] for x in found[:8]]})
        return found[:8]

    def _prelookup_market(self, need: str) -> str:
        """Ask the shelf *before* the forge instead of only after it.

        An unreachable market returns "", and an empty string means *no
        answer*, never *nothing there*: only a lookup that came back and
        matched nothing licenses a forge.
        """
        import urllib.request as _url

        shelf_size = None
        lexical_ok = False
        lexical: list[str] = []
        payload = None
        try:
            url = os.environ.get("TOOLMARKET_URL", "http://127.0.0.1:8000")
            req = _url.Request(url.rstrip("/") + "/resources")
            with _url.urlopen(req, timeout=6) as resp:
                body = resp.read().decode("utf-8", "replace")
            payload = json.loads(body)
            lexical_ok = True
        except Exception as exc:                               # noqa: BLE001
            self._record("market_prelookup", {"need": need, "ok": False,
                                              "error": f"{type(exc).__name__}: {exc}"})
        if lexical_ok:
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict):
                items = payload.get("items") or payload.get("resources") or []
            else:
                items = []
            if not isinstance(items, list):
                items = []
            shelf_size = len(items)
            lexical = self._lexical_name_hits(need, items)[:5]
            self._record("market_prelookup", {"need": need, "ok": True,
                                              "shelf_size": shelf_size, "hits": lexical})

        semantic = self._market_search(need)
        # Peers are asked even -- especially -- when the local shelf is dead.
        peers, peer_answered = self._peer_lookup(need)
        if not lexical_ok and not semantic and not peers and not peer_answered:
            return ""

        upstream = ([] if (lexical or semantic or peers)
                    else self._discover_upstream(need))

        labelled: list[tuple[str, list[str]]] = []
        index: dict[str, list[str]] = {}
        for name, why in ([(h["name"], "synonym") for h in semantic]
                          + [(n, "name") for n in lexical]
                          + [(f"{p['name']} ({p['peer']})", "peer") for p in peers]
                          + [(u["name"], "upstream") for u in upstream]):
            if name not in index:
                index[name] = []
                labelled.append((name, index[name]))
            if why not in index[name]:
                index[name].append(why)
        rendered = ", ".join(f"{n} ({'+'.join(w)})" for n, w in labelled[:5])
        where = "/resources" if lexical_ok else "/resources (unreachable)"
        where += (" + /search" if semantic else
                  " + /search (no answer or no confident hit)")
        if peer_answered and not peers and not lexical and not semantic and not lexical_ok:
            return ("Tool-market pre-lookup: this host's shelf did not answer; a "
                    "peer shelf was searched and carries no entry matching this "
                    "need, so this forge is not a duplicate *there*. The local "
                    "half of the lookup is missing, not empty.")

        if peers:
            lines = ["Tool-market pre-lookup: nothing on this host's shelf, but a "
                     "peer machine's shelf has entries for this need:"]
            for p in peers[:5]:
                lines.append(f"  - {p['name']} [peer {p['peer']}] {p.get('description') or ''}")
            lines.append("Do not re-forge one of these. If one serves the need, "
                         "say which peer it is on and call it there; if none "
                         "does, say why not in the forge context, then forge.")
            return "\n".join(lines)

        if upstream:
            lines = ["Tool-market pre-lookup: nothing on the local shelf, but the "
                     "public MCP registry has entries for this need, now "
                     "registered here as DRAFT resources:"]
            for u in upstream[:5]:
                how = ("callable over the network" if u.get("callable")
                       else "needs its package launched first -- it is a pointer, "
                            "not a callable tool")
                lines.append(f"  - {u['name']} [{u.get('state', 'draft')}, {how}] "
                             f"{(u.get('description') or '')[:110]}")
            lines.append("Do not re-forge one of these blindly. If one serves the "
                         "need, say which and use it; if none does, say why not in "
                         "the forge context, then forge. A remote MCP entry is "
                         "called through the tool-market and does NOT run local code.")
            return "\n".join(lines)

        if not labelled:
            if not lexical_ok:
                return ""                 # unreachable shelf is not an answer
            tail = (", and the semantic search agreed" if semantic else
                    "; /search did not answer, so an entry that does this job "
                    "under different words cannot be ruled out")
            return ("Tool-market pre-lookup: the shelf answered and carries no "
                    "entry matching this need (%s, %s entries%s) -- this forge "
                    "is not a duplicate."
                    % (where, "?" if shelf_size is None else shelf_size, tail))
        return ("Tool-market pre-lookup: the shelf already carries entries that "
                "overlap this need: %s. Do not re-forge one of these -- if one "
                "serves the need, say which and use it instead. (Looked up via "
                "%s.)" % (rendered, where))

    def forge_tool(self, need: str, context: str = "") -> Any:
        """Forge a tool for a recurring need, then report it the way agent.py does.

        The wording is agent.py's, not mine, because the sentence *is* the
        result: a model reading "could not forge a working tool" concludes the
        need is unservable and forges something worse to compensate, while
        "the name was taken" sends it to the tool that already exists. Those
        are opposite pieces of advice and only one of them is true.
        """
        # The three gates agent.py runs before the pipeline, in its order.
        # `over` carries all of them into the forge as *context*, never as a
        # refusal: a warning the model reads changes what it builds, and a wall
        # it hits changes what it can do -- the second was the bug.
        over = self._tool_budget_report()
        try:
            verdict = self._arbitrate_forge(need)
            block = verdict.context_block() if hasattr(verdict, "context_block") else ""
            if block:
                over = (over + "\n\n" + block) if over else block
        except Exception:                                      # noqa: BLE001
            pass                     # a broken arbiter licenses the forge
        try:
            shelf = self._prelookup_market(need)
        except Exception:                                      # noqa: BLE001
            shelf = ""
        if shelf:
            over = (over + "\n\n" + shelf) if over else shelf
        if context:
            over = (over + "\n\n" + context) if over else context

        # `replace` is left at the pipeline default (False): a forge is not a
        # deliberate replacement, and a candidate whose name is already taken is
        # refused rather than silently overwriting it. agent.py states this in a
        # comment at the same call site, for the same reason -- so a test double
        # that models only the forge path still satisfies the protocol.
        try:
            pipeline = self._forge_pipeline()
            res = pipeline.forge(need, context=over)
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "text": f"Could not forge a working tool for: {need} (0 rounds).",
                    "error": f"{type(exc).__name__}: {exc}"}
        if getattr(res, "aborted", False):
            return {"ok": False, "aborted": True,
                    "text": ("Stopped at the operator's request before this tool was "
                             "verified. Nothing was registered and nothing was ruled "
                             "out -- read their message and continue from there: "
                             "answer them, or change the need to match what they asked "
                             "for. Do not call forge_tool again with this same need "
                             "unchanged.")}
        if not res.ok:
            if getattr(res, "replace_conflict", ""):
                return {"ok": False, "replace_conflict": res.replace_conflict,
                        "text": ("Not forged: %s The candidate passed verification, so the "
                                 "capability exists -- it just may not take a name this "
                                 "library already uses. Either evolve that tool, or forge "
                                 "again with a need that names what is different about this "
                                 "one. This refusal is on the ledger." % res.replace_conflict)}
            return {"ok": False, "rounds": res.rounds,
                    "text": f"Could not forge a working tool for: {need} ({res.rounds} rounds)."}
        spec = res.spec
        note = ""
        try:
            self.store.save_tool(spec)
        except Exception as exc:                               # noqa: BLE001
            note = f" Forged {spec.name!r} but did not save it: {exc}"
        self.registry.register(spec, replace=True)
        state = str(getattr(getattr(spec, "state", None), "value", ""))
        return {"ok": True, "name": spec.name, "state": state, "rounds": res.rounds,
                "text": f"Forged {spec.name!r} [{state}]: {spec.description}" + note}

    def evaluate_tool(self, name: str) -> Any:
        """Run the full battery on an existing tool and print it as agent.py does.

        The tool's own declared sample travels with it when it was forged with
        one; without it the battery invents arguments, and a verdict on invented
        arguments is one nobody earned.
        """
        spec = self._registry().get(name)
        if spec is None:
            return {"ok": False, "text": f"No tool named {name!r}."}
        try:
            pipeline = self._forge_pipeline()
            sample = dict(getattr(spec, "sample_call", None) or {}) or None
            report = pipeline.verifier.verify(spec, sample)
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "text": f"Could not evaluate {name!r}.",
                    "error": f"{type(exc).__name__}: {exc}"}
        # Three tags, not two. A waived check printed as `[FAIL]` reads as a
        # tool defect to the only reader that matters -- measured: a model that
        # saw `[FAIL] negative: WAIVED: the probe shares 5 content words ...`
        # had to parse the prose to learn the probe was at fault, not the tool.
        # The tag should say it without the sentence.
        lines = [
            f"  [{('WAIVED' if c.waived else 'UNPROBED' if c.unprobed else 'PASS' if c.passed else 'FAIL')}]"
            f" {c.name}: {c.detail}"
            for c in report.checks
        ]
        claim = getattr(pipeline.verifier, "claim_report", None)
        if claim is not None:
            for finding in claim.findings:
                flag = "  " if finding.ok else "! "
                lines.append(f"  {flag}claim/{finding.name}: {finding.detail}")
        return {"ok": bool(report.passed), "passed": bool(report.passed),
                "text": f"{report.summary()}\n" + "\n".join(lines)}

    def _baseline_for(self, spec: Any) -> Any:
        """The tool's pinned obligation set, created on first evolution.

        Loaded from the store so the ratchet survives restart; a baseline that
        only lives in RAM resets to "whatever the current version is", which is
        the erosion it exists to prevent.
        """
        from .forge.validity import FrozenBaseline
        stored = self.store.load_baseline(spec.name)
        return FrozenBaseline.from_dict(stored) if stored else FrozenBaseline.capture(spec)

    def evolve_tool(self, name: str, failure_report: str = "") -> Any:
        """Breed a better version of a tool, reported the way agent.py reports it."""
        spec = self.registry.get(name)
        if spec is None:
            return {"ok": False, "text": f"No tool named {name!r}."}
        try:
            from .forge.evolution import EvolutionEngine
            pipeline = self._forge_pipeline()
            engine = EvolutionEngine(pipeline.verifier.llm, pipeline.verifier,
                                     on_event=lambda k, pl: self.store.log_event(k, pl))
            report = failure_report or spec.verification.get("failed_detail", "underperforming")
            baseline = self._baseline_for(spec)
            result = engine.evolve(spec, report, baseline=baseline, context="internal")
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "text": f"Could not evolve {name!r}.",
                    "error": f"{type(exc).__name__}: {exc}"}
        self.store.log_event("evolve", {
            "tool": name, "improved": result.improved,
            "vetoed": [m.generated.name for m in result.mutants if not m.admissible]})
        if not result.improved:
            vetoed = sum(1 for m in result.mutants if not m.admissible)
            tail = f" {vetoed} vetoed by the validity gate." if vetoed else ""
            return {"ok": False, "improved": False, "tried": len(result.mutants),
                    "text": (f"Evolved {name!r}: no admissible mutant beat the original "
                             f"({len(result.mutants)} tried)." + tail)}
        best = result.best_mutant.generated
        new_spec = ToolSpec(
            name=best.name, description=best.description,
            parameters=best.parameters, fn=spec.fn, runner=spec.runner,
            code=best.code, source="evolved", probes=best.probes,
            effect_signature=best.effect_signature, tags=best.tags,
            state=ToolState.ACTIVE,
        )
        self.registry.register(new_spec, replace=True)
        self.store.save_baseline(baseline.extended_with(new_spec))
        note = ""
        try:
            self.store.archive_version(name, spec.code, spec.verification)
            self.store.save_tool(new_spec, allow_overwrite=True)
        except Exception as exc:                               # noqa: BLE001
            note = f" Evolved {name!r} in memory but did not save it: {exc}"
        return {"ok": True, "improved": True, "name": name,
                "fitness": result.best_mutant.fitness,
                "text": f"Evolved {name!r}: new version active (fitness {result.best_mutant.fitness:.2f})." + note}

    def find_gaps(self) -> Any:
        """Analyse the library and propose the capabilities it is missing."""
        try:
            from .forge.metacog import MetaCognition
            pipeline = self._forge_pipeline()
            metacog = getattr(self, "_metacog", None)
            if metacog is None:
                metacog = MetaCognition(pipeline.verifier.llm, self.registry, pipeline)
                self._metacog = metacog
            rep = metacog.analyse()
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "text": "Meta-cognition is disabled.",
                    "error": f"{type(exc).__name__}: {exc}"}
        if not rep.proposals:
            return {"ok": True, "gaps_found": 0,
                    "text": "No capability gaps identified right now."}
        lines = [f"  - {p.need} (forged={rep.forged})" for p in rep.proposals]
        return {"ok": True, "gaps_found": rep.gaps_found, "forged": rep.forged,
                "text": f"Found {rep.gaps_found} gaps; forged {rep.forged}:\n" + "\n".join(lines)}

    def retire_tool(self, name: str, rationale: str = "") -> Any:
        """Take a tool out of service on purpose, with a reason recorded.

        Distinct from quarantine, which is the automatic safety valve and stays
        on regardless of policy. Retiring is a judgement call, so it is the one
        that demands a rationale -- and refuses builtins, which are disabled
        through set_autonomy rather than retired.
        """
        if not rationale or not rationale.strip():
            return {"ok": False, "text": "A rationale is required to retire a tool."}
        spec = self.registry.get(name)
        if spec is None:
            return {"ok": False, "text": f"No tool named {name!r}."}
        if getattr(spec, "source", "") == "builtin":
            return {"ok": False, "text": (
                f"Refusing to retire {name!r}: it is a builtin meta-tool. "
                "Disable it with set_autonomy instead.")}
        self.registry.retire(name)
        self.store.log_event("retire", {"tool": name, "rationale": rationale})
        return {"ok": True, "retired": name,
                "text": f"Retired {name!r}. Rationale recorded: {rationale}"}

    def run_python(self, code: str, entry: str = "main",
                   args: dict[str, Any] | str | None = None,
                   timeout_s: float = 30) -> Any:
        if not str(code or "").strip():
            return {"ok": False, "error": "code was empty"}
        parsed: dict[str, Any] = {}
        if isinstance(args, str):
            raw = args.strip()
            if raw:
                try:
                    parsed = json.loads(raw)
                except Exception as exc:                       # noqa: BLE001
                    return {"ok": False, "error": f"args must be JSON: {exc}"}
                if not isinstance(parsed, dict):
                    return {"ok": False, "error": "args must be a JSON object"}
        elif isinstance(args, dict):
            parsed = args
        if not skills_mod.valid_name(entry.replace("-", "_")):
            return {"ok": False, "error": f"entry must be a plain identifier, got {entry!r}"}
        try:
            result = self.sandbox.run(code, entry, parsed)
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            return {"ok": bool(result.ok), "output": result.output,
                    "stdout": (result.stdout or "")[-8000:],
                    "stderr": (result.stderr or "")[-4000:],
                    "error": result.error, "timed_out": result.timed_out,
                    "duration_ms": result.duration_ms}
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # -- schedule ---------------------------------------------------------

    def _scheduler(self) -> Any:
        from .schedule import Schedule
        return Schedule()

    def schedule_add(self, text: str, when: str, repeat: int = 0) -> Any:
        from .schedule import ScheduleError
        try:
            task = self._scheduler().add(text, when, repeat or 0)
            return {"id": task.id, "text": task.text, "due_at": task.due_at,
                    "repeat_seconds": task.repeat, "line": task.line()}
        except ScheduleError as exc:
            return {"error": str(exc)}

    # ── interrupted runs ───────────────────────────────────────────────────
    # `runs.py` was fully implemented beside this service while none of it was
    # reachable from a DSH session, so the one thing a service is uniquely good
    # at -- outliving the process that started the work -- was the one thing
    # missing. Measured 2026-10-30: 8 journals on disk under
    # `%LOCALAPPDATA%\autoforge\runs`, 0 visible from here.
    #
    # The journal path is NOT re-derived here. `runs.journal_dir()` already
    # resolves `$AUTOFORGE_RUNS` before `$AUTOFORGE_HOME/runs`, so pointing this
    # service at the real journal is a config change on the mount (one
    # environment variable), not an edit to the reader. Re-deriving it would
    # give the same question two answers, and the two would drift.

    def runs_status(self) -> Any:
        """Every journalled run with the decision `runs.scan` makes about it.

        Deliberately the same function the CLI's own `runs_status` prints: a
        second implementation would be a second answer.
        """
        from . import runs as _runs

        rows = _runs.scan()
        return {"journal": str(_runs.journal_dir()),
                "count": len(rows),
                "runs": rows,
                "resumable": [r["run_id"] for r in rows
                              if r["status"] == "resumable"]}

    def resume_run(self, run_id: str = "") -> Any:
        """Read an interrupted run's transcript and the prompt that continues it.

        This does NOT resume anything, and the return value says so in as many
        words (`resumed: false`, `next_step`). The reason is structural rather
        than cautious: this service keeps no conversation history -- `_run_task`
        calls the agent with a task string and nothing else -- so "resuming"
        here would mean starting the task over while calling it a continuation.
        That is worse than not offering it, because the caller cannot tell.

        What this DOES provide is the evidence: the recorded transcript, and the
        prompt built by `runs.resume_prompt`, which names the interruption and
        asks for the next step rather than the first. Handing that to something
        that really does hold the history -- the DSH agent loop -- is a resume;
        this is the lookup that makes one possible.

        `run_id` may be a unique prefix. Empty means the run stranded longest,
        the same order `runs.resumable` keeps so the oldest waiting work is not
        starved behind a trickle of fresh failures.
        """
        from . import runs as _runs

        marks = _runs.unfinished()
        if not marks:
            return {"resumed": False,
                    "error": f"no runs journalled in {_runs.journal_dir()}"}
        if run_id:
            hit = [m for m in marks if m.run_id.startswith(run_id)]
            if not hit:
                return {"resumed": False,
                        "error": f"no run matches {run_id!r}"}
            if len(hit) > 1:
                return {"resumed": False,
                        "error": f"{run_id!r} is ambiguous: "
                                 f"{[m.run_id for m in hit]}"}
            mark = hit[0]
        else:
            ready = _runs.resumable()
            if not ready:
                return {"resumed": False,
                        "error": "no run is resumable right now",
                        "runs": [m.run_id for m in marks]}
            mark = ready[0]
        status, reason = mark.claim()
        return {"resumed": False,
                "run_id": mark.run_id,
                "task": mark.task,
                "turns": mark.turns,
                "messages": len(mark.messages),
                "status": status,
                "reason": reason,
                "prompt": _runs.resume_prompt(mark),
                "next_step": ("hand this prompt to the agent that holds the "
                              "transcript; nothing has been re-executed here")}

    def runs_forget(self, run_id: str, why: str = "") -> Any:
        """Archive one run record so it is never resumed again.

        Archived, not deleted. Reaching a result finishes a run; this is the
        other thing, and the difference is the whole value of the record -- an
        operator asking "did that get done, or did it get dropped" can only be
        answered if the dropped one still exists somewhere. So the file moves
        to `<journal>/archived/` and stays readable.

        The subdirectory is not a convention that has to be honoured by a
        filter: `runs.unfinished` globs `run-*.json` directly under the journal
        directory, so an archived record stops counting the moment it moves,
        and moving it back is what undoes the decision.

        `why` is required for the same reason. A record that vanishes with no
        note is indistinguishable from one that was never written.
        """
        from . import runs as _runs

        if not str(run_id or "").strip():
            return {"error": "run_id is required"}
        if not str(why or "").strip():
            return {"error": "a reason is required: this is abandonment, not "
                             "completion, and the note is the difference"}
        hits = [m for m in _runs.unfinished() if m.run_id.startswith(run_id)]
        if len(hits) != 1:
            return {"error": f"{run_id!r} matched {len(hits)} run(s)"}
        mark = hits[0]
        journal = _runs.journal_dir()
        src_path = journal / f"run-{mark.run_id}.json"
        if not src_path.exists():
            return {"error": f"{src_path} is already gone"}
        archive = journal / "archived"
        try:
            archive.mkdir(parents=True, exist_ok=True)
            dst_path = archive / src_path.name
            if dst_path.exists():
                import time as _time
                dst_path = archive / (src_path.stem
                                      + f".{int(_time.time())}.json")
            os.replace(src_path, dst_path)
        except OSError as exc:
            return {"error": f"could not archive {src_path}: {exc}"}
        return {"archived": mark.run_id,
                "task": mark.task,
                "why": why,
                "moved_to": str(dst_path),
                "restore": f"move it back to {journal} to make it resumable again"}

    def skill_port(self, source: str = "tars", overwrite_body: bool = False,
                   include_other_os: bool = False) -> Any:
        """Copy another agent's procedures into this library, verbatim.

        The functions this wraps (`port_tars_skills`, `port_openclaw_skills`)
        already live in `skills.py`; what was missing was a way to reach them
        from a DSH session.

        `source="openclaw"` is honest about a finding measured on 2026-10-30:
        `openclaw_skill_roots()` searches npm global node_modules for an
        `openclaw` package, and there is none installed on this host -- the
        only package under `%APPDATA%\npm\node_modules` is `claude-mem`. So
        that branch fails with a message naming the environment variable to
        set, rather than reporting "0 skills ported" as though the copy had
        been attempted and found empty. Those are different facts.

        `overwrite_body` defaults to False here, unlike the underlying
        function: a re-port that silently replaces a procedure this agent has
        since edited is a data loss dressed as an upgrade.
        """
        from . import skills as _skills

        which = str(source or "").strip().lower()
        if which == "tars":
            return _skills.port_tars_skills(overwrite_body=overwrite_body)
        if which == "openclaw":
            return _skills.port_openclaw_skills(
                overwrite_body=overwrite_body,
                include_other_os=include_other_os)
        return {"error": f"unknown source {source!r}; expected 'tars' or "
                         f"'openclaw'"}

    def schedule_list(self) -> Any:
        return {"report": self._scheduler().report()}

    def schedule_tick(self) -> Any:
        """What is due, without running it.

        `overdue_by` takes a timestamp, not an optional one. Passing `None` --
        as this did -- raised `TypeError: unsupported operand type(s) for -:
        'NoneType' and 'float'` from inside a comprehension, so the whole call
        failed rather than a single row. That made `schedule_tick` unusable
        whenever anything was actually due, which is the only time it matters;
        measured 2026-10-26 with one task two seconds past its time.
        """
        import time as _time

        sched = self._scheduler()
        now = _time.time()
        due = sched.due()
        return {"due": [{"id": t.id, "text": t.text, "due_at": t.due_at,
                         "repeat_seconds": t.repeat,
                         "overdue_by": t.overdue_by(now)} for t in due],
                "count": len(due)}

    def schedule_run_due(self, budget_s: float = 600.0) -> Any:
        """Attend to whatever the schedule says is due, then return.

        The difference from `schedule_tick` is the whole point of this method.
        `schedule_tick` REPORTS what is due and leaves the running to whoever
        asked -- which is right when a person is present and wrong when the
        operating system is the caller. A registered task that only lists what
        it found looks successful forever while nothing ever runs.

        This is the executor: it takes each due task, hands its text to the same
        agent the rest of the service uses, and records the outcome on the task
        so that a repeated failure is distinguishable from a long silence. One
        task raising does not stop the ones behind it -- one broken task must
        not cost every later task its turn.

        Nothing due means no model call and no output. That is deliberate rather
        than an optimisation: a scheduler that reports "nothing happened" every
        fifteen minutes trains whoever reads the log to stop reading it.

        `budget_s` is a wall-clock ceiling for the whole run, not per task. The
        caller is a scheduled task with no terminal, so an unbounded tick would
        hold a process open indefinitely on a task that never finishes.
        """
        import time as _time

        sched = self._scheduler()
        due = sched.due()
        if not due:
            return {"due": 0, "ran": 0, "failed": 0, "results": [],
                    "note": "nothing was due"}

        started = _time.monotonic()
        results: list[dict[str, Any]] = []
        failed = 0
        for task in due:
            remaining = budget_s - (_time.monotonic() - started)
            if remaining <= 0:
                results.append({"id": task.id, "ok": False,
                                "note": "skipped: the tick budget was spent "
                                        "before this task's turn",
                                "skipped": True})
                continue
            ok = False
            note = ""
            try:
                res = self._run_task(task.text, timeout_s=min(remaining, budget_s))
                ok = True
                note = str(res)[:500]
            except Exception as exc:                       # noqa: BLE001
                note = f"{type(exc).__name__}: {exc}"
            try:
                sched.complete(task.id, ok=ok, note=note)
            except Exception as exc:                       # noqa: BLE001
                note = f"{note} (could not record: {type(exc).__name__})"
            if not ok:
                failed += 1
            results.append({"id": task.id, "ok": ok, "note": note[:300]})

        self.store.log_event("schedule_run_due",
                             {"due": len(due), "failed": failed})
        return {"due": len(due), "ran": len(results) - failed,
                "failed": failed, "results": results,
                "elapsed_s": round(_time.monotonic() - started, 1)}

    def _run_task(self, text: str, timeout_s: float = 600.0) -> str:
        """One scheduled task through the agent, with the service's own model.

        The agent is built here rather than reused: a tick has no conversation
        to continue, and a half-used history from someone's chat is the wrong
        context for work the machine decided to do on its own.

        Tools come from the registry, so a tick can use everything this service
        has forged -- which is the point of a schedule outliving the session
        that created it.

        The client is built per call, not cached: a tick runs a handful of times
        an hour at most, and holding a gateway connection open in a process the
        OS starts for one task would defeat the reason it exits afterwards.
        """
        from .core.agent import Agent as _Agent
        from .core.llm import OpenAICompatClient

        model, base, key = _forge_route()
        if not key:
            raise RuntimeError(
                "no credential for the scheduled task's model: DSH's OpenAI "
                "gateway key is missing. Looked at "
                "$DSH_OPENAI_GATEWAY_API_KEY, $AUTOFORGE_API_KEY, and %s. "
                % os.path.join(_dsh_home(), "openai-gateway", "api-key"))
        llm = OpenAICompatClient(model=model, base_url=base, api_key=key,
                                 timeout=timeout_s)
        agent = _Agent(llm, self.registry, max_turns=12,
                       allow_self_terminate=False)
        result = agent.run(text)
        return getattr(result, "content", "") or ""

    def schedule_done(self, task_id: str, note: str = "", ok: bool = True) -> Any:
        from .schedule import ScheduleError
        try:
            task = self._scheduler().complete(task_id, ok=ok, note=note)
            return {"id": task.id, "text": task.text, "runs": task.runs,
                    "failures": task.failures, "cancelled": task.cancelled}
        except ScheduleError as exc:
            return {"error": str(exc)}

    def schedule_cancel(self, task_id: str) -> Any:
        from .schedule import ScheduleError
        try:
            task = self._scheduler().cancel(task_id)
            return {"id": task.id, "cancelled": task.cancelled}
        except ScheduleError as exc:
            return {"error": str(exc)}

    # -- user-defined tools ----------------------------------------------

    @staticmethod
    def _entry_of(code: str, name: str) -> str:
        """Which symbol in `code` the sandbox should call.

        This used to be the literal `"main"`, and it is why every forged tool
        failed after a restart. A generated tool defines a function named after
        itself (`def count_vowels(text):`), not `main`; the sandbox answered
        `entry 'main' not found or not callable` and the model saw a tool that
        had verified perfectly and then did nothing. The one tool that kept
        working was `real_env_probe`, whose body happens to define `main` --
        which is exactly why the defect survived: it looked like tools worked.

        Preference order, because a generated body may carry helpers:
          1. a top-level `def` whose name is the tool's own name;
          2. `main`, if the body really does define it;
          3. the only `def` in the body, when there is exactly one;
          4. the first `def`, as the last resort.
        """
        defs = re.findall(r"^def\s+(\w+)\s*\(", code, re.M)
        if not defs:
            return name
        if name in defs:
            return name
        if "main" in defs:
            return "main"
        if len(defs) == 1:
            return defs[0]
        return defs[0]

    def _runner_for(self, code: str, entry: str) -> Any:
        """The callable a durable tool runs through, out of process.

        It must answer with a `ToolResult`, not with the sandbox's own object.
        The registry logs the reason itself: anything that is not a
        `ToolResult` falls through to `json.dumps`, so a sandbox failure would
        be recorded as `ok=True` with the dataclass repr as output -- a failure
        laundered into a success. Measured before this fix: `tool_call`
        returned the string `"SandboxResult(ok=True, output=[7, 14, 21], ...)"`
        to the model.
        """
        from .tools.registry import ToolResult

        # `runner` is called as `runner(name, arguments)` -- see
        # `ToolRegistry.call`. A `**kwargs` signature here raised
        # "takes 0 positional arguments but 2 were given" on every call, which
        # is the second half of why persisted tools were dead: the first fix
        # (passing it as `runner` rather than `fn`) put it on a path whose
        # calling convention did not match its definition.
        def fn(name: str, arguments: dict[str, Any]) -> Any:
            res = self.sandbox.run(code, entry, arguments or {})
            ok = bool(getattr(res, "ok", False))
            payload = getattr(res, "output", None)
            if payload is None:
                payload = getattr(res, "stdout", "") or ""
            if not isinstance(payload, str):
                try:
                    payload = json.dumps(payload, ensure_ascii=False, default=str)
                except Exception:                              # noqa: BLE001
                    payload = str(payload)
            err = getattr(res, "error", None)
            if not ok and not err:
                err = "timed out" if getattr(res, "timed_out", False) else "tool failed"
                tail = (getattr(res, "stderr", "") or "").strip()
                if tail:
                    err = f"{err}: {tail[-800:]}"
            return ToolResult(
                "", ok, payload, error=err,
                duration_ms=float(getattr(res, "duration_ms", 0.0) or 0.0),
                aborted=bool(getattr(res, "aborted", False)),
            )

        return fn

    @staticmethod
    def _sample_args(params: dict[str, Any] | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, sub in ((params or {}).get("properties") or {}).items():
            kind = sub.get("type")
            out[key] = {"integer": 1, "number": 1.0, "boolean": True,
                        "string": "x", "array": [], "object": {}}.get(kind)
        return out

    def tool_define(self, name: str, code: str, description: str = "",
                    parameters: dict[str, Any] | None = None,
                    require: list[str] | None = None,
                    sample_call: dict[str, Any] | None = None,
                    sample_expect: str = "",
                    probes: list[dict[str, Any]] | None = None) -> Any:
        """Persist a tool, then run it once so a broken body fails here.

        The function is never bound into this process: `fn` runs the code in a
        child (see forge/sandbox.py), which is what keeps a forging tool from
        taking the service down with it.

        `sample_call` / `sample_expect` / `probes` are what a *forged* tool gets
        from the pipeline and a *hand-written* one did not. Without them the
        battery reports two honest UNPROBEDs:

            execution  "no valid sample call was declared, so correctness was
                       never examined; all this shows is that the code runs"
            trigger    "no positive probe was declared, so nothing asked this
                       tool to fire"

        Neither is a bug in the battery -- an un-argued verdict is the bug it
        exists to prevent. The bug was here: `tool_define` had no way to declare
        either, so every hand-written tool was sealed ACTIVE on the strength of
        "it runs", which is the exact standard the verifier's own comment says
        is not enough. Measured 2026-10-25 on is_perfect_square: 3/5 checks,
        with 16/15/0/-4 all correct and still nothing proving it.

        `probes` takes dicts: {"query": str, "expect": "call"|"ignore",
        "negative_query": str|None}.
        """
        params = parameters or {"type": "object", "properties": {}}
        # `_runner_for` needs the symbol to call, and a defined body normally
        # names its function after the tool (`def is_perfect_square(n)`), not
        # `main`. Resolving it here is the same fix the load path needed; the
        # probe below must use the same entry or it tests a symbol that will
        # never be called. Before this, every `tool_define` call raised
        # "Service._runner_for() missing 1 required positional argument: 'entry'".
        entry = self._entry_of(code, name)
        fn = self._runner_for(code, entry)
        spec = ToolSpec(name=name, description=description or name,
                        parameters=params, code=code, fn=fn, runner=fn)
        # The declaration travels with the tool so `evaluate_tool` can judge it
        # later, in another session, from the row and not from this call.
        spec.sample_call = dict(sample_call) if sample_call else None
        spec.sample_expect = sample_expect or ""
        for p in (probes or []):
            if not isinstance(p, dict) or not str(p.get("query") or "").strip():
                continue
            spec.probes.append(TriggerProbe(
                query=str(p["query"]),
                expect=str(p.get("expect") or "call"),
                negative_query=(str(p["negative_query"])
                                if p.get("negative_query") else None)))
        spec.state = ToolState.ACTIVE
        # The *declared* sample first: a synthetic {"n": 1} says only that the
        # code runs, and it is what the caller sees back. Measured 2026-10-25:
        # `square2` declared sample_call={"n": 4}, sample_expect="16", and
        # tool_define answered "output": "1" -- the probe's answer to n=1,
        # passed back as if it were the tool's real result. The declaration
        # was stored correctly; the reply just described a different call.
        probe_args = dict(sample_call) if sample_call else self._sample_args(params)
        probe = self.sandbox.run(code, entry, probe_args)
        out = getattr(probe, "output", None)
        if out is None:
            out = getattr(probe, "stdout", "") or ""
        if not isinstance(out, str):
            try:
                out = json.dumps(out, ensure_ascii=False, default=str)
            except Exception:                                  # noqa: BLE001
                out = str(out)
        err = getattr(probe, "error", None)
        if not getattr(probe, "ok", False) and not err:
            tail = (getattr(probe, "stderr", "") or "").strip()
            err = ("timed out" if getattr(probe, "timed_out", False) else "probe failed")
            if tail:
                err = f"{err}: {tail[-800:]}"
        payload: dict[str, Any] = {"name": name, "registered": True,
                                   "ok": getattr(probe, "ok", False),
                                   "output": out[:4000],
                                   "error": err}
        self.registry.register(spec, replace=True)
        try:
            self.store.save_tool(spec, allow_overwrite=True)
        except Exception as exc:                               # noqa: BLE001
            payload["save_error"] = f"{type(exc).__name__}: {exc}"
        self.store.log_event("tool_defined", {"tool": name})
        return payload

    def tool_list(self) -> list[Any]:
        out: list[Any] = []
        for n in self.registry.names():
            spec = self.registry.get(n)
            out.append({"name": n,
                        "description": getattr(spec, "description", ""),
                        "state": str(getattr(getattr(spec, "state", ""), "value", ""))})
        return out

    def tool_call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        res = self.registry.call(name, arguments or {})
        payload = {"ok": getattr(res, "ok", True),
                   "output": str(getattr(res, "output", res))[:4000]}
        # `error` is the only half of a ToolResult that says *why* a call did
        # not work. Dropping it here left the model with `ok: false` and an
        # empty string -- enough to know something went wrong and nothing
        # about what, which is worse than no answer at all. Kept as its own
        # key, and also folded into `output` when output is empty, so a
        # client that only reads `output` still sees a sentence.
        err = getattr(res, "error", None)
        if err:
            payload["error"] = str(err)[:2000]
            if not payload["output"]:
                payload["output"] = f"{name}: {err}"
        if getattr(res, "quarantined", False):
            payload["quarantined"] = True
        # `call`, not `tool_called`. The writer and the reader had drifted:
        # this line wrote `tool_called`, while `store.last_called_by_tool`
        # queries `kind='call'` -- so every call was recorded and none was
        # ever read back. Measured 2026-10-25: 42 `tool_called` rows in the
        # ledger, `tool_usage` returning {}. The docstring on the store side
        # says the ledger is "the only complete record of what was used";
        # the record was complete and the query was looking elsewhere.
        self.store.log_event("call", {"tool": name, "ok": payload["ok"]})
        # Write the counter back too, not just the ledger. `list_tools` renders
        # `stats.calls`, which lives in memory and is serialised only when the
        # tool row is saved -- and a call is not a save. So the two records
        # disagreed in the other direction as well: the ledger knew and the
        # listing did not. Measured 2026-10-25: count_words at calls=2 in the
        # listing with a ledger that had never seen the tool called once.
        # Best-effort: a store that cannot write must not fail a call that
        # already ran.
        try:
            spec = self.registry.get(name)
            if spec is not None:
                self.store.save_stats(spec)
        except Exception:                                      # noqa: BLE001
            pass
        return payload

    # -- self-report ------------------------------------------------------
    #
    # The one thing autoforge has that DSH has no equivalent for, and the one
    # its own design document argues hardest for: a description of the agent
    # that is *measured* rather than written down. `DESIGN.md` §5 puts it as
    # "prose loses to the prior: an agent will explain at length that it has no
    # hands while holding them" -- so every line here is read from the object
    # that knows, not from a paragraph that can drift away from it.
    #
    # Two of the three halves are here. The reach half is `Sandbox.reach()`,
    # the memory half is `ToolStore.report()`. The third -- the autonomy
    # policy's enforcement table -- belongs to `AutonomyPolicy`, which this
    # service does not own (DSH's sandbox and approval stack is the authority
    # in a DSH session). Claiming a policy table here would be inventing a
    # second one, which is exactly what `enforcement_table()` exists to make
    # impossible.

    def my_capabilities(self, probe: bool = False) -> dict[str, Any]:
        """The full table: numbers, paths, limits, and what the ledger holds.

        Split with `_identity_lines()` on a rule rather than a feeling. The brief
        carries the sentences that change what the model DOES and is paid for
        every turn; this carries the specifics and is paid once, when asked. The
        two must not say the same thing twice -- measured before that rule: the
        brief at 3209 chars and this at 1002 shared four lines verbatim
        (filesystem, network, run_python, DSH), so a model reading both read the
        same reassurance twice and the second copy bought nothing.

        What stays here and is deliberately NOT in the brief: the cwd, the
        timeout, the env allow-list, the database path, the counts. Those are the
        answers to precise questions ("how long do I have", "where does this
        live") and repeating them every request is how a block becomes skimmed.
        """
        reach = self.sandbox.reach(probe=probe)
        report = self.store.report()
        lines = [
            "What I can reach, in numbers (measured, not declared):",
            "",
            f"  Host: {reach['host']}",
            # The two sentences that matter are in the per-turn brief. Repeating
            # them here was the duplication this split exists to remove, so this
            # block points at them instead of copying them.
            "  Filesystem and network: as the WHO I AM block states -- this is",
            "  where their BOUNDS are, which the brief does not carry:",
            f"    cwd     - {reach['cwd']}",
            f"    env     - {reach['env']}",
            f"    timeout - {reach['timeout_s']}s per call",
            "  The bounds limit blast radius, not capability.",
        ]
        live = reach.get("probe")
        if isinstance(live, dict):
            if live.get("error"):
                lines.append(f"  Live self-test: did not run ({live['error']})")
            else:
                fs = live.get("host_filesystem") or {}
                net = live.get("network") or {}
                lines += [
                    "  Live self-test (measured just now):",
                    f"    - host filesystem: {'OK' if fs.get('ok') else 'FAILED'}"
                    f" - {fs.get('detail', '')}",
                    f"    - network:         {'OK' if net.get('ok') else 'FAILED'}"
                    f" - {net.get('detail', '')}",
                ]
        lines += [
            "",
            f"  Memory: sqlite at {report['db_path']}, survives restart. "
            f"{report['tools']} tool(s), {report['events']} ledger event(s).",
            "          Every forge, amendment and call is on that ledger;",
            "          my_history reads it back.",
        ]
        return {"text": "\n".join(lines), "reach": reach, "store": report}

    # -- GPU --------------------------------------------------------------
    #
    # Ported from `ForgeAgent._tool_gpu`. The read-only half is ungated on
    # purpose: probing a device, estimating occupancy from static limits and
    # linting timing units cannot wedge a machine. Compiling agent-authored
    # CUDA and launching it can, which is why autoforge puts those behind
    # `may_run_cuda_kernels`. This service has no autonomy policy, so the gate
    # is reported honestly instead of being pretended: the tool says what it
    # would need rather than silently refusing or silently proceeding.

    @staticmethod
    def _gpu():
        from . import gpu as G
        return G

    def gpu_probe(self) -> str:
        G = self._gpu()
        info = G.probe()
        if not info.available:
            return "No CUDA device reachable here.\n" + "\n".join(
                f"  - {n}" for n in info.notes)
        return info.summary()

    def gpu_occupancy(self, block: int = 256, grid: int = 1,
                      shared_bytes: int = 0, regs_per_thread: int = 0,
                      arch_major: int = 8, arch_minor: int = 9) -> str:
        G = self._gpu()
        cfg = G.LaunchConfig(grid=(grid, 1, 1), block=(block, 1, 1),
                             shared_bytes=shared_bytes)
        est = G.estimate_occupancy(cfg, arch=(arch_major, arch_minor),
                                   regs_per_thread=regs_per_thread)
        if est is None:
            return (f"No static limits on file for sm_{arch_major}{arch_minor}, "
                    f"so no estimate is offered. Add them to "
                    f"gpu.probe.ARCH_LIMITS or measure on the device.")
        head = (f"{est.summary()}\n"
                f"  threads/block {cfg.threads_per_block}, "
                f"blocks {cfg.total_blocks}, shared {cfg.shared_bytes} B")
        if not est.measured:
            head += ("\n  This is arithmetic from the architecture's static "
                     "limits. It is not a measurement - launch on the device "
                     "and use measured occupancy for anything you intend to "
                     "quote.")
        return head

    def gpu_units_audit(self, source: str = "", path: str = "") -> str:
        G = self._gpu()
        if path:
            hits = G.audit_source_tree([path])
            if not hits:
                return (f"No thousand-factor-against-ms offences in {path}. "
                        f"(Known false negative: the factor arriving through a "
                        f"variable and the ms label in a different statement.)")
            out = []
            for f, offs in hits.items():
                out.append(f"{f}:")
                out += [f"  line {o.line_no}: {o.line}" for o in offs[:10]]
            return "\n".join(out)
        offs = G.audit_ms_scale(source)
        if not offs:
            return ("No offence found. Note the two accepted false negatives: "
                    "a factor smuggled through a variable, and the label and "
                    "the multiplication in different statements.")
        return "\n".join(f"line {o.line_no}: {o.line}\n  {o.reason}" for o in offs)

    def gpu_verify(self, kind: str, actual: str, reference: str = "",
                   tolerance: float = 1e-4) -> str:
        G = self._gpu()
        try:
            act = [float(x) for x in actual.replace(",", " ").split()]
        except ValueError:
            return "actual must be a list of numbers."
        if not reference:
            return ("A reference is required. Without one this is not a "
                    "verification - it is a restatement of the kernel's own "
                    "output, which is what a fast-but-wrong kernel produces.")
        try:
            ref = [float(x) for x in reference.replace(",", " ").split()]
        except ValueError:
            return "reference must be a list of numbers."
        r = G.verify(act, ref, tolerance=tolerance, label=kind)
        return (f"{'PASS' if r['passed'] else 'FAIL'} max|diff|="
                f"{r['max_abs_diff']:.3e} worst index {r['worst_index']}\n"
                f"  {r['reason']}")

    def gpu_compile(self, code: str, name: str = "kernel",
                    arch_major: int = 8, arch_minor: int = 9) -> str:
        G = self._gpu()
        src = G.KernelSource(name=name, code=code, arch=(arch_major, arch_minor))
        if not src.entry_points():
            return ("Refusing to compile: no `__global__ void` entry point in "
                    "the source. Entry points found: none.")
        try:
            k = G.compile_kernel(src)
        except G.KernelUnavailable as exc:
            self.store.log_event("gpu_compile", {"name": name, "ok": False})
            return f"Could not compile: {exc}"
        self.store.log_event("gpu_compile",
                             {"name": name, "ok": True, "cached": k.from_cache})
        d = k.to_dict()
        return (f"Compiled {d['name']} for {d['arch']} via {d['compiler']}"
                f"{' (cache hit)' if k.from_cache else ''}; "
                f"entries: {', '.join(d['entry_points'])}"
                f"\ncache key {d['cache_key']}\nartefact {d['artefact']}")

    def gpu_bench(self, code: str, name: str = "kernel", block: int = 256,
                  grid: int = 1, shared_bytes: int = 0, reps: int = 5,
                  time_budget_s: float = 30.0, arch_major: int = 8,
                  arch_minor: int = 9) -> str:
        G = self._gpu()
        src = G.KernelSource(name=name, code=code, arch=(arch_major, arch_minor))
        cfg = G.LaunchConfig(grid=(grid, 1, 1), block=(block, 1, 1),
                             shared_bytes=shared_bytes)
        res = G.bench_kernel(src, cfg, label=name, reps=reps,
                             time_budget_s=time_budget_s)
        self.store.log_event("gpu_bench", {"name": name, "ok": res.ok})
        return res.summary()

    # -- CPU --------------------------------------------------------------

    @staticmethod
    def _cpu():
        from . import cpu as C
        return C

    def cpu_probe(self) -> dict[str, Any]:
        C = self._cpu()
        info = C.probe()
        return {
            "available": info.available,
            "name": info.name,
            "logical_cores": info.logical_cores,
            "physical_cores": info.physical_cores,
            "arch": getattr(info, "arch", ""),
            "compiler": getattr(info, "compiler", ""),
            "compiler_version": getattr(info, "compiler_version", ""),
            "openmp": getattr(info, "openmp", False),
            "cache_bytes": getattr(info, "cache_bytes", {}),
            "notes": list(info.notes),
            "text": info.summary(),
        }

    def cpu_runs_here(self, target: str = "native") -> str:
        """Whether an artefact built for `target` can execute on this CPU.

        The probe is passed in rather than taken from the module cache: a
        verdict about THIS machine must be about this machine, and `runs_here`
        reads the arch off the info it is handed.
        """
        C = self._cpu()
        info = C.probe()
        if not info.available:
            why = "; ".join(info.notes) if info.notes else "no C compiler found"
            return f"No verdict without a compiler to build for. {why}"
        try:
            ok, why = C.runs_here(target, info)
        except Exception as exc:                            # noqa: BLE001
            return f"No verdict: {type(exc).__name__}: {exc}"
        verdict = "RUNS HERE" if ok else "WILL NOT RUN HERE"
        return f"{verdict}: {target}\n  {why}"

    def cpu_preflight(self, code: str, name: str = "kernel") -> str:
        """Lint C source before it is compiled. Touches nothing.

        `preflight` takes a `KernelSource`, not raw text: the flags and target
        are part of what is being linted (a `-march` this CPU cannot run is a
        finding, and it only exists once the source has been parsed).
        """
        C = self._cpu()
        try:
            src = C.KernelSource(name=name, code=code)
        except ValueError as exc:
            return f"Nothing to lint: {exc}"
        return C.preflight(src).summary()

    def cpu_units_audit(self, source: str = "", path: str = "") -> str:
        """The do_bench-returns-ms lint, shared with the GPU half.

        `audit_ms_scale` lives in `timing`, not in `cpu` -- the defect it finds
        is about a benchmark harness, and both backends use the same one. This
        method exists on the CPU side because the tool catalog splits the two,
        and pointing at the same implementation keeps them from disagreeing.
        """
        from .timing import audit_ms_scale, audit_source_tree
        if path:
            hits = audit_source_tree([path])
            if not hits:
                return (f"No thousand-factor-against-ms offences in {path}. "
                        f"(Known false negative: the factor arriving through a "
                        f"variable and the ms label in a different statement.)")
            out = []
            for f, offs in hits.items():
                out.append(f"{f}:")
                out += [f"  line {o.line_no}: {o.line}" for o in offs[:10]]
            return "\n".join(out)
        offs = audit_ms_scale(source)
        if not offs:
            return ("No offence found. Note the two accepted false negatives: "
                    "a factor smuggled through a variable, and the label and "
                    "the multiplication in different statements.")
        return "\n".join(f"line {o.line_no}: {o.line}\n  {o.reason}" for o in offs)

    def cpu_cache_stats(self) -> str:
        C = self._cpu()
        st = C.cache_stats()
        total = st["hits"] + st["misses"]
        if not total:
            return (f"Nothing compiled on this machine yet. Cache root "
                    f"{st['root']} - the first compile will populate it, and a "
                    f"repeat of the same source, target, compiler and flags is "
                    f"a cache hit rather than a rebuild.")
        return (f"{st['hits']} hit(s), {st['misses']} miss(es), "
                f"{st['hit_rate']:.0%} hit rate under {st['root']}")

    def cpu_compile(self, code: str, name: str = "kernel",
                    target: str = "native") -> str:
        C = self._cpu()
        try:
            src = C.KernelSource(name=name, code=code, target=target)
        except ValueError as exc:
            return f"Refusing to compile: {exc}"
        if not src.entry_points():
            return ("Refusing to compile: no callable entry point in the "
                    "source. Static functions and `main` cannot be called "
                    "through ctypes, so a kernel made only of those has "
                    "nothing to invoke. Entry points found: none.")
        # A build for a target this machine cannot run is worse than no build:
        # the failure is SIGILL in the caller's own process rather than an
        # error anyone can read. Refused before the compiler runs.
        resolved = src.resolved_target()
        ok, why = C.runs_here(resolved, C.probe())
        if not ok:
            return (f"Refusing to compile for {target}: {why}\n"
                    f"  Build for a target this machine can run, or drop the "
                    f"artefact in a child process and let it die there.")
        try:
            k = C.compile_kernel(src)
        except C.KernelUnavailable as exc:
            self.store.log_event("cpu_compile", {"name": name, "ok": False})
            return f"Could not compile: {exc}"
        self.store.log_event("cpu_compile",
                             {"name": name, "ok": True, "cached": k.from_cache})
        d = k.to_dict()
        return (f"Compiled {d.get('name', name)} for {d.get('target', target)}"
                f"{' (cache hit)' if k.from_cache else ''}"
                f"\ncache key {d.get('cache_key', '?')}"
                f"\nartefact {d.get('artefact', '?')}")

    def cpu_run_isolated(
        self,
        code: str,
        name: str = "kernel",
        kind: str = "saxpy",
        size: int = 1 << 20,
        n: int = 256,
        dtype: str = "f32",
        seed: int = 0,
        reps: int = 5,
        warmup: int = 3,
        timeout: float = 0.0,
        target: str = "native",
    ) -> str:
        """Compile, then verify and time in a child process.

        Isolation is not a convenience. An unproven kernel's first call must
        not be in the caller's own process: a wrong index segfaults, an
        unbounded loop hangs, and both would otherwise end the session instead
        of producing a report.
        """
        C = self._cpu()
        try:
            prob = C.problem(kind, size=size, n=n, dtype=dtype, seed=seed)
        except C.CallError as exc:
            return f"No such problem: {exc}"
        try:
            src = C.KernelSource(name=name, code=code, target=target)
        except ValueError as exc:
            return f"Refusing to run: {exc}"
        try:
            k = C.compile_kernel(src)
        except C.KernelUnavailable as exc:
            return f"Could not compile: {exc}"
        guard = C.run_isolated(k, prob, reps=reps, warmup=warmup,
                               timeout=timeout or None)
        self.store.log_event("cpu_run_isolated",
                             {"name": name, "kind": kind, "ok": guard.ok})
        rows = [f"{'OK' if guard.ok else 'FAILED'}: {name} ({kind})"]
        for field in ("timed_out", "crashed", "returncode", "fault",
                      "verdict", "wall", "error"):
            value = getattr(guard, field, None)
            if value not in (None, "", False):
                rows.append(f"  {field}: {value}")
        return "\n".join(rows)

    def cpu_tune(
        self,
        kind: str = "saxpy",
        size: int = 1 << 20,
        n: int = 256,
        dtype: str = "f32",
        seed: int = 0,
        max_candidates: int = 0,
        budget_s: float = 0.0,
        reps: int = 5,
        warmup: int = 3,
    ) -> str:
        """Search candidate kernels for one that beats the baseline.

        `tune_kind` is the entry point that builds the problem AND the naive
        baseline from a kind name. Calling `tune` directly needs a `Problem`
        plus the baseline *source text*, and `NAIVE` is a dict of sources keyed
        by kind -- passing the dict where a string was expected is the
        AttributeError this comment replaces.

        Budgets default to the module's own, not to a number invented here: a
        tuning search is exactly the place where a made-up budget either
        starves the search or holds a session for minutes.
        """
        C = self._cpu()
        try:
            res = C.tune_kind(
                kind, size=size, n=n, dtype=dtype, seed=seed,
                max_candidates=max_candidates or C.DEFAULT_MAX_CANDIDATES,
                budget_s=budget_s or C.DEFAULT_BUDGET_S,
                reps=reps, warmup=warmup,
            )
        except C.CallError as exc:
            return f"No such problem: {exc}"
        except KeyError as exc:
            return f"Nothing to tune: {exc}"
        self.store.log_event("cpu_tune",
                             {"kind": kind, "candidates": res.n_candidates,
                              "ok": res.winner is not None})
        return res.report()

    # -- notifications ----------------------------------------------------

    def notify_channels(self) -> dict[str, Any]:
        """Which channels are configured, and which are broken.

        Read-only: it says how to reach a person, not that a message is owed.
        The problems are reported rather than raised, for the same reason the
        agent builds them at construction -- an unreadable channel must show up
        in a menu, not at the moment someone needs to be told something.
        """
        return {
            "channels": [getattr(c, "name", str(c)) for c in self.notifier.channels],
            "problems": list(self._notify_problems),
        }

    def notify_send(self, text: str, channel: str = "") -> dict[str, Any]:
        """Deliver to every configured channel, or the one named.

        `Notifier.send` takes `only` as a LIST of channel names, not a single
        name -- passing a bare string would be iterated character by character
        and match nothing, which reads as "no channel matched" rather than as
        the API misuse it is.
        """
        only = [channel] if channel else None
        try:
            deliveries = self.notifier.send(text, only=only)
        except Exception as exc:                            # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        rows = []
        for d in deliveries or []:
            rows.append({
                "channel": getattr(d, "channel", ""),
                "ok": bool(getattr(d, "ok", False)),
                "detail": str(getattr(d, "detail", ""))[:300],
            })
        return {"ok": all(r["ok"] for r in rows) if rows else False,
                "deliveries": rows}



def _mission(m: Any) -> dict[str, Any]:
    if hasattr(m, "__dict__"):
        d = {k: v for k, v in m.__dict__.items() if not k.startswith("_")}
    else:
        d = dict(m)
    for k, v in list(d.items()):
        if isinstance(v, (bytes,)) or not isinstance(v, (str, int, float, bool, type(None), list, dict)):
            d[k] = str(v)
    return d


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    svc = Service(argv[0] if argv else None)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            method = req.get("method")
            params = req.get("params") or {}
            fn = svc.methods.get(method)
            if fn is None:
                out = {"id": rid, "ok": False, "error": "unknown method: %s" % method}
            else:
                out = {"id": rid, "ok": True, "result": fn(**params)}
        except Exception as exc:  # noqa: BLE001
            out = {"id": rid, "ok": False,
                   "error": "%s: %s" % (type(exc).__name__, exc),
                   "trace": traceback.format_exc()[-1200:]}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
