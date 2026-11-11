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
import threading
import time
import traceback
from typing import Any, Callable

from .store import ToolStore, outside_home, state_map
from .mission import MissionStore
from . import skills as skills_mod
from .notify import Notifier, channels_from_config
from .forge.sandbox import Sandbox
from .forge.validity import review_tool
from .tools.registry import ToolRegistry
from .tools.spec import ToolSpec, ToolState, TriggerProbe
from .autonomy.scopes import derived_scope, scope_for
from .autonomy.controls import ControlPlane
from .egress import admit as _admit_egress, bind as egress_bind, coverage as _egress_coverage

#: How long a hot/cold decision stands before it is recomputed. The answer only
#: moves when a call happens or when a tool's week runs out, and both are slower
#: than a turn -- but the cache also has to be cheap enough to keep the prompt
#: byte-identical between turns that made no call, because that is what the
#: provider's prefix cache bills on.
_EXPOSURE_TTL_S = 60.0


def _adopt_sibling_ledger(home: str) -> tuple[str, str]:
    """Which ledger this session should open, and whether it was inherited.

    Kept as a name because callers and tests refer to it; the rule lives in
    `store.db_file`, which is the module that owns the database and the one
    `tars_port` also asks. Two readers of the same question were the defect
    twice today -- once when this function derived its own home, and once when
    the probe did -- so there is one implementation and this is a forwarder.
    """
    from .store import db_file

    return db_file(home)


def _home() -> str:
    """Where this service keeps its state -- asked of the store, not re-derived.

    This used to be a second implementation. It agreed with `store._default_home`
    whenever `LOCALAPPDATA` was set and diverged whenever it was not: this one
    answered `<home>/autoforge`, which is the path `store._default_home`'s
    docstring records as shadowing the package on 2026-09-16, while the store
    answered `<home>/.autoforge`. One question, two answers, and the wrong one
    on the side of the module that owns the database.
    """
    from .store import home_dir

    return home_dir()


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


def _dsh_profiles() -> list[str]:
    """Profile directories DSH could be running from, most likely first.

    Ordered rather than discovered-and-hoped: `desktop` and `web` are the two
    that carry an `agent-default-model` entry on this host, and `acp` shares
    the install. Anything else on disk is appended instead of ignored -- a
    profile somebody added is a profile that may be the one running.
    """
    root = os.path.join(_dsh_home(), "profiles")
    try:
        found = [n for n in os.listdir(root)
                 if os.path.isdir(os.path.join(root, n))]
    except OSError:
        return []
    rank = {"desktop": 0, "web": 1, "acp": 2}
    return sorted(found, key=lambda name: (rank.get(name, 9), name))


def _yaml_lenient(path: str) -> Any:
    """Load a DSH profile patch file, tolerating the tags it legitimately uses.

    `cordis.patch.yml` allows `!!js` expressions, and a plain `yaml.safe_load`
    refuses the whole document over one of them -- the error names the tag, not
    the setting the caller wanted, so a caller that gives up here reports "no
    model configured" for a file that configures one. The same lenient loader
    is already used by the presets check for exactly this reason.

    Returns None rather than raising: every caller of this is a fallback, and a
    fallback that throws is not a fallback.
    """
    try:
        import yaml
    except Exception:                                      # noqa: BLE001
        return None

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor(
        "tag:yaml.org,2002:js", lambda loader, suffix, node: None)
    try:
        with open(path, encoding="utf-8") as handle:
            return yaml.load(handle, Loader=_Loader)
    except Exception:                                      # noqa: BLE001
        return None


def _entry_config(entry: Any) -> dict[str, Any]:
    """The `config` mapping of one profile patch entry, or an empty dict."""
    if isinstance(entry, dict):
        config = entry.get("config")
        if isinstance(config, dict):
            return config
    return {}


def _yaml_cached(path: str) -> Any:
    """`_yaml_lenient`, re-read only when the file it read has changed.

    Keyed on (path, mtime_ns, size) rather than on elapsed time. A TTL here would
    mean a configuration edit stays invisible for up to the length of the window,
    which is the same drift-by-delay this whole function exists to avoid.
    """
    try:
        info = os.stat(path)
        stamp = (int(info.st_mtime_ns), int(info.st_size))
    except OSError:
        return None
    global _YAML_CACHE
    key = os.path.abspath(path)
    hit = _YAML_CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    doc = _yaml_lenient(path)
    # Bounded: profiles are a handful of files, and an unbounded cache keyed on a
    # path is a slow leak on anything that generates names.
    if len(_YAML_CACHE) > 16:
        _YAML_CACHE.clear()
    _YAML_CACHE[key] = (stamp, doc)
    return doc


def _dsh_model() -> tuple[str, str]:
    """The model id DSH itself routes to, and which file said so.

    Read from the harness rather than remembered here. There was a constant in
    this file -- `xdpool-global/deepseek-v4.1-flash` -- and it was wrong on the
    machine it shipped to: the gateway serves `xdpool/deepseek-v4.1-flash`
    (35 models listed, the `xdpool-global` lane only under a `-sg` suffix), so
    both the forge pipeline and every scheduled task failed at the first model
    call with `HTTP 404 ... could not be resolved`. The value had never been
    tested against the gateway it named, and the tests that touched those paths
    supplied their own route, so nothing noticed.

    A default that has to match somebody else's configuration is not a default;
    it is a copy that will drift. `cordis.patch.yml` already records the pair
    DSH uses (`agent-default-model: {provider, model}`), so this reads that.
    Second choice is the enabled model list from the same profile, which names
    what the lane is actually allowed to serve.

    Returns ("", reason) when nothing could be read, so the caller can raise an
    error that says where it looked instead of quoting a guess.
    """
    tried: list[str] = []
    for profile in _dsh_profiles():
        path = os.path.join(_dsh_home(), "profiles", profile, "cordis.patch.yml")
        if not os.path.isfile(path):
            continue
        tried.append(path)
        doc = _yaml_cached(path)
        if not isinstance(doc, list):
            continue
        provider = ""
        secondary: list[str] = []
        for entry in doc:
            if not isinstance(entry, dict):
                continue
            ident = str(entry.get("id") or entry.get("name") or "")
            config = _entry_config(entry)
            if ident.endswith("agent-default-model"):
                provider = str(config.get("provider") or "").strip()
                model = str(config.get("model") or "").strip()
                if model:
                    # `provider/model` is the form this gateway resolves; a model
                    # that already carries a slash is taken as written.
                    full = model if "/" in model else (
                        f"{provider}/{model}" if provider else model)
                    return full, path
            for key in ("modelSelectionCn", "modelSelectionGlobal"):
                block = config.get(key)
                if isinstance(block, dict):
                    for name in block.get("enabledModelIds") or ():
                        name = str(name).strip()
                        if name:
                            secondary.append(name)
        if secondary:
            chosen = secondary[0]
            full = chosen if "/" in chosen else (
                f"{provider}/{chosen}" if provider else chosen)
            return full, path
    return "", ("no profile configured a model; looked at: "
                + (", ".join(tried) if tried else "no DSH profile on disk"))


def _gateway_models(base: str, key: str, *, timeout: float = 5.0) -> list[str]:
    """What the gateway says it serves, or [] if it cannot be asked.

    Cached for a minute because both callers of `_forge_route` are on paths that
    run minutes at a time, and an answer that costs 470 ms (measured, three
    runs: 472.4 / 533.8 / 347.3) should not be paid per task in a tick.

    An empty list is a real answer here -- "could not ask" -- and callers must
    not read it as "nothing is served". The two are different sentences and only
    one of them is a reason to refuse a model id.
    """
    global _MODELS_CACHE
    now = time.time()
    stamp = (base, len(key or ""))
    if _MODELS_CACHE.get("stamp") == stamp and now - _MODELS_CACHE.get("at", 0.0) < 60.0:
        return list(_MODELS_CACHE.get("models") or [])
    models: list[str] = []
    try:
        import urllib.request as _url
        req = _url.Request(base.rstrip("/") + "/models",
                           headers={"Authorization": "Bearer " + key})
        _admit_egress(req.full_url, tool="service._gateway_models")
        with _url.urlopen(req, timeout=timeout) as answer:
            payload = json.loads(answer.read().decode("utf-8", "replace"))
        for row in payload.get("data") or ():
            name = row.get("id") if isinstance(row, dict) else None
            if name:
                models.append(str(name))
    except Exception:                                      # noqa: BLE001
        models = []
    _MODELS_CACHE = {"stamp": stamp, "at": now, "models": list(models)}
    return models


#: (model, base) -> whether it can see what a tool returned. `True`/`False`
#: are measured; `None` means the gateway could not be asked. Cached because the
#: probe is a real request (~1-3 s) whose answer is a property of the provider's
#: pairing, not of this run.
_TOOL_READING: dict[tuple[str, str], bool | None] = {}

#: What the last capability resolution did, so `report()` can say it without
#: paying for a probe: {"configured", "used", "why", "alternates", "at"}.
_ROUTE_NOTE: dict[str, Any] = {}

_PROBE_VALUE = "7391"
_PROBE_ALT = "4826"

#: Why a probe answered "could not ask", keyed like `_TOOL_READING`. A bare
#: None says only that asking failed; the operator needs to know whether the
#: gateway was unreachable, the key was refused, or the call ran past its own
#: timeout -- those are three different jobs.
_TOOL_READING_WHY: dict[tuple[str, str], str] = {}

#: How long "could not ask" is remembered before the probe is willing to try
#: again. True and False are measurements and are kept for the life of the
#: process; a failure to ask is an accident, and remembering an accident
#: forever turns one slow call into a permanent verdict. Measured 2026-11-11:
#: `xdpool/hy4free` answered a hand-written probe in 6.7 s and reports "No
#: result provided" for every tool turn, but an earlier probe of the same id
#: had been cached as None -- so `_tool_capable_model` answered "unverified"
#: and handed it real work, which then died as `LLMError: HTTP 502` far from
#: the cause. A 60 s memory of a failure is enough to stop a storm of retries
#: and short enough that a gateway restart does not need a service restart.
_UNASKED_TTL_S = 60.0
_UNASKED_AT: dict[tuple[str, str], float] = {}


def _reads_tool_results(model: str, base: str, key: str,
                        *, timeout: float = 90.0,
                        again: bool = False) -> bool | None:
    """Can this model see the contents of a `role: "tool"` turn?

    Not "does it answer" -- `xdpool/deepseek-v4.1-flash` answers perfectly well.
    It emits a structured call, the call runs, the result is delivered under the
    right id, and the model then reports that it received nothing. A 404 is
    loud; this is silent, and a loop over such a model looks like it is calling
    tools that have no effect.

    The payload is hand-written OpenAI wire format with none of ours in the path:
    two tool turns carrying two distinctive numbers, and a question answerable
    only by reading them. Everything this service could break is bypassed on
    purpose -- a probe routed through the code it is testing reports on that
    code, not on the model.

    Returns None when the gateway could not be asked at all. That is not a
    verdict: a model is not guilty of being unreachable.
    """
    stamp = (model, base)
    if stamp in _TOOL_READING and not again:
        cached = _TOOL_READING[stamp]
        if cached is not None:
            return cached
        if (time.time() - _UNASKED_AT.get(stamp, 0.0)) < _UNASKED_TTL_S:
            return None
    if not model or not key:
        _TOOL_READING[stamp] = None
        _UNASKED_AT[stamp] = time.time()
        _TOOL_READING_WHY[stamp] = "no model or no key to ask with"
        return None
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 60,
        "messages": [
            {"role": "system", "content": "Answer with the number alone."},
            {"role": "user", "content": "Report the sum of the two tool results."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "p1", "type": "function",
                 "function": {"name": "read", "arguments": "{\"slot\": 1}"}},
                {"id": "p2", "type": "function",
                 "function": {"name": "read", "arguments": "{\"slot\": 2}"}},
            ]},
            {"role": "tool", "tool_call_id": "p1", "name": "read",
             "content": "{\"slot\": 1, \"n\": " + _PROBE_VALUE + "}"},
            {"role": "tool", "tool_call_id": "p2", "name": "read",
             "content": "{\"slot\": 2, \"n\": " + _PROBE_ALT + "}"},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read one recorded slot.",
                "parameters": {"type": "object",
                               "properties": {"slot": {"type": "integer"}},
                               "required": ["slot"]},
            },
        }],
    }
    answer: bool | None = None
    reason = ""
    try:
        import urllib.request as _url
        req = _url.Request(
            base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + key},
            method="POST",
        )
        _admit_egress(req.full_url, tool="service._reads_tool_results")
        with _url.urlopen(req, timeout=timeout) as reply:
            body = json.loads(reply.read().decode("utf-8", "replace"))
        msg = ((body.get("choices") or [{}])[0].get("message") or {})
        said = (msg.get("content") or "").replace(",", "").replace(" ", "")
        wanted = int(_PROBE_VALUE) + int(_PROBE_ALT)
        # The sum has to appear as a standalone number. A substring test would
        # pass on "73914826" or on a model that merely echoes the inputs.
        answer = re.search(r"(?<!\d)" + str(wanted) + r"(?!\d)", said) is not None
    except Exception as exc:                               # noqa: BLE001
        answer = None
        reason = "%s: %s" % (type(exc).__name__, exc)
    _TOOL_READING[stamp] = answer
    if answer is None:
        _UNASKED_AT[stamp] = time.time()
        _TOOL_READING_WHY[stamp] = reason or "the gateway returned no answer"
    else:
        _UNASKED_AT.pop(stamp, None)
        _TOOL_READING_WHY.pop(stamp, None)
    return answer


def _tool_capable_model(model: str, base: str, key: str) -> tuple[str, str]:
    """The model to use for work that needs tools, and why it is that one.

    Returns (model, why). `why` is one of:
      "configured"          -- the configured model reads tool results
      "substituted: <name>" -- it cannot be used, and this one can
      "unverified"          -- the gateway could not be asked; configured kept
      "no alternative"      -- it cannot be used, and nothing else could be shown to

    Two questions get two answers here, because they have different costs and
    different consequences. Whether the gateway serves an id at all is one
    `/models` call; whether it can see what a tool returned is a real tool-turn
    probe. A model the gateway does not serve cannot do anything, so it is
    replaced on that fact alone -- measured 2026-11-07: the DSH profile named
    `qodercn/qfmodel`, the gateway served seven other ids, that one answered
    HTTP 502, and this function used to call that `unverified` and hand it back.
    Every caller then failed later with an `LLMError: ... HTTP 502` from inside
    a probe it could not interpret, which reads like a broken forge rather than
    a stale profile.

    The converse is deliberately not done: a model that answered a tool turn
    correctly is kept even if the published list omits it, because the list is
    the gateway's opinion and the answer is the measurement. `_ROUTE_NOTE`
    records that disagreement as `listed: False` rather than "correcting" a
    working model.

    The substitution is not silent: the caller records it and `report()` shows
    it, so "the forge works" and "the forge works because the model was
    replaced" stay distinguishable sentences.
    """
    global _ROUTE_NOTE
    served = _gateway_models(base, key)
    # An empty list means "could not ask", which is not "nothing is served" --
    # `_gateway_models` says so in its own docstring, and a refusal built on an
    # empty list would claim the gateway has no models at all.
    listed = (not served) or (model in served)
    got = _reads_tool_results(model, base, key)
    if got is True:
        _ROUTE_NOTE = {"configured": model, "used": model, "why": "configured",
                       "alternates": [], "listed": listed, "at": time.time()}
        return model, "configured"
    if got is None and listed:
        # One retry with a longer deadline before calling the gateway
        # unreachable. "Could not ask" is not a verdict, but it is also not
        # permission: measured 2026-11-11, a single probe of `xdpool/hy4free`
        # ran to 6.7 s, was remembered as None, and every caller afterwards
        # got "unverified" -- which this function then treats as a licence to
        # run real work on a model that reports "No result provided" for every
        # tool turn. The retry is what separates a slow gateway from a blind
        # model; it costs one call and only happens on the failure path.
        got = _reads_tool_results(model, base, key, timeout=180.0, again=True)
        # A successful retry is an answer, and it has to be read as one. It was
        # not, for one revision: the retry was added below the `got is True`
        # branch, so a model that answered on the second try fell through to the
        # substitution loop, found only itself among the candidates, and was
        # refused with "no alternative" -- a working model rejected because the
        # code that was supposed to accept it had already run. Caught by the
        # assertion "the second answer is the one that decides".
        if got is True:
            _ROUTE_NOTE = {"configured": model, "used": model, "why": "configured",
                           "alternates": [], "listed": listed, "at": time.time()}
            return model, "configured"
    if got is None and listed:
        _ROUTE_NOTE = {"configured": model, "used": model, "why": "unverified",
                       "probe": _TOOL_READING_WHY.get((model, base), ""),
                       "alternates": [], "listed": listed, "at": time.time()}
        return model, "unverified"

    # Two reasons to look elsewhere, and the note has to say which one:
    #   * the gateway does not serve this id at all (`missing`)
    #   * it serves it, and it cannot see tool output (`got is False`)
    # The wording of the eventual refusal differs, because one is a stale
    # configuration and the other is a provider defect.
    reason = ("not served by the gateway" if not listed
              else "cannot read tool results")

    # At most five candidates, stopping at the first that works -- this runs on
    # the way to real work, not as a survey. When the configured id was not
    # served, the candidates are the served list itself; when it was served but
    # blind, the same list minus the configured id.
    tried: list[str] = []
    for candidate in served:
        if candidate == model or candidate in tried:
            continue
        if len(tried) >= 5:
            break
        tried.append(candidate)
        if _reads_tool_results(candidate, base, key) is True:
            _ROUTE_NOTE = {"configured": model, "used": candidate,
                           "why": "substituted: " + candidate,
                           "reason": reason, "alternates": tried,
                           "listed": listed, "at": time.time()}
            return candidate, "substituted: " + candidate
    _ROUTE_NOTE = {"configured": model, "used": model, "why": "no alternative",
                   "reason": reason, "alternates": tried, "listed": listed,
                   "at": time.time()}
    return model, "no alternative"


def _as_numbers(value: Any, what: str) -> list[float]:
    """Coerce a caller's numbers, and say what was actually received.

    Two shapes, because callers hold two shapes: a string of numbers separated
    by commas or spaces, or a list/tuple of numbers. Returns floats. Raises
    `ValueError` whose text names the offending shape, so the reader learns
    what to send instead of what not to.
    """
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(
                    f"{what} must be numbers: item {index} is "
                    f"{type(item).__name__} ({item!r}). Send a list of "
                    f"numbers, or a string of them separated by commas or "
                    f"spaces.")
            out.append(float(item))
        return out
    if isinstance(value, str):
        try:
            return [float(part) for part in value.replace(",", " ").split()]
        except ValueError:
            raise ValueError(
                f"{what} must be numbers: {value!r} does not read as a list "
                f"of them. Send numbers separated by commas or spaces, or a "
                f"list of numbers.") from None
    raise ValueError(
        f"{what} must be a list of numbers or a string of them: got "
        f"{type(value).__name__} ({value!r}).")


def _has_numbers(value: Any) -> bool:
    """Whether `value` holds anything at all to compare against.

    An empty string, an empty list, and a missing argument all mean the same
    thing to a verification -- there is no reference -- and all three have to
    be refused for the same stated reason. A list is not falsy by accident
    here: `[]` would make `if not reference` true, which is the answer this
    check wants.
    """
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return bool(str(value).strip())


def _tool_read_note(where: str, base: str) -> str:
    """Shared wording for the refusal, so both call sites say the same thing.

    `base` is passed rather than read from `_ROUTE_NOTE`: the note records what
    happened, and a refusal that reads its own address out of the thing it is
    refusing about would print an empty URL if the note were ever reset.
    """
    configured = _ROUTE_NOTE.get("configured", "")
    probed = ", ".join(_ROUTE_NOTE.get("alternates") or []) or "none probed"
    if _ROUTE_NOTE.get("reason") == "not served by the gateway":
        # A different sentence for a different fault. The old wording said the
        # model "cannot see tool results", which would be a claim about a model
        # the gateway never ran -- measured 2026-11-07: `qodercn/qfmodel`
        # answered HTTP 502 while all seven served ids answered 200.
        return (
            "the configured model (%s) is not one this gateway serves, and no "
            "served model could be shown to read tool results either: %s. %s "
            "assembles its answer through tools, so it cannot run here. Set "
            "AUTOFORGE_MODEL to a served model -- %s lists them."
            % (configured, probed, where, base.rstrip("/") + "/models"))
    return (
        "the configured model (%s) cannot see tool results, and no other served "
        "model could be shown to: %s. %s assembles its answer through tools, so "
        "it cannot run here. Set AUTOFORGE_MODEL to a model that can -- %s lists "
        "them." % (configured, probed, where, base.rstrip("/") + "/models"))


def _forge_route() -> tuple[str, str, str]:
    """The (model, base_url, api_key) the forge pipeline talks to.

    Model resolution, in order: `$AUTOFORGE_MODEL` (an explicit choice wins),
    then the pair DSH's own profile configures via `_dsh_model()`. There is no
    third constant: the previous one was a guess about somebody else's gateway
    and it was wrong for months.

    When no model can be resolved this returns an empty string rather than a
    plausible one. Both callers then raise an error naming every place they
    looked, which is recoverable; a wrong id fails later, inside a probe the
    caller cannot interpret, and reads like a broken forge.
    """
    model = (os.environ.get("AUTOFORGE_MODEL") or "").strip()
    if not model:
        model, _why = _dsh_model()
    base = os.environ.get("AUTOFORGE_BASE_URL", "http://127.0.0.1:8326/v1")
    key = os.environ.get("AUTOFORGE_API_KEY") or _gateway_key()
    return model, base, key


_MODELS_CACHE: dict[str, Any] = {}
_YAML_CACHE: dict[str, tuple[tuple[int, int], Any]] = {}


def _notify_channels_from_config() -> tuple[list[Any], list[str]]:
    """Notification channels from the same config file the CLI reads.

    `configfile.config_path()` -- `~/.autoforge/config.json` unless
    `$AUTOFORGE_CONFIG` says otherwise -- is where a person puts
    `notify.channels`, and it is where the CLI would read them too, so both
    share one source of truth. Corrected 2026-11-10: this docstring used to say
    that file "is written by `auto setup`", and no such command ships here --
    `auto setup` appeared only in three docstrings and `configfile.save` has no
    caller. Nothing in this tree creates the file, so `notify`'s refusal names
    the path and says so rather than sending a reader after a missing wizard.

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


class TaskOutcome:
    """What one scheduled run actually came to, in fields rather than in prose.

    This exists because the prose was being read as if it were a verdict.
    Measured 2026-11-04, before this class: `_run_task` returned
    `getattr(result, "content", "") or ""`, `schedule_run_due` set `ok = True`
    whenever nothing raised, and a run that stopped on the turn cap was written
    down as a success --

        result row  -> {'ok': True, 'note': '(turn cap of 12 reached before completion)'}
        task on disk -> runs 1 / failures 0

    so the apology was stored in the one place the caller did not look. The
    consequence is not cosmetic: `failures` is how a task that never finishes
    becomes distinguishable from a task that is quietly working, and it stayed
    at zero for both.

    `ok` answers one question -- "did this run finish what it was asked to
    finish?" -- and only `hit_turn_cap` and `stopped_by_operator` can say no.
    A run the agent ended itself (`self_terminated`) counts as finished: the
    agent decided it was done, and calling that a failure would teach the
    scheduler to distrust the one ending the agent chose on purpose.

    Kept as a plain class rather than a dataclass so `service.py` gains no
    import for four fields and one factory.
    """

    __slots__ = ("content", "ok", "why", "turns", "tools",
                 "hit_turn_cap", "self_terminated", "stopped_by_operator")

    def __init__(
        self,
        content: str = "",
        *,
        ok: bool = True,
        why: str = "",
        turns: int = 0,
        tools: list[str] | None = None,
        hit_turn_cap: bool = False,
        self_terminated: bool = False,
        stopped_by_operator: bool = False,
    ) -> None:
        self.content = content
        self.ok = ok
        self.why = why
        self.turns = turns
        self.tools = list(tools or [])
        self.hit_turn_cap = hit_turn_cap
        self.self_terminated = self_terminated
        self.stopped_by_operator = stopped_by_operator

    @classmethod
    def from_result(cls, result: Any) -> "TaskOutcome":
        """Fold an `AgentResult` into a verdict the scheduler can act on.

        Read with `getattr` and defaults rather than attribute access: this is
        also the path a mock client in a test drives, and a scheduler that only
        works with a fully-populated result object is a scheduler whose
        failure mode is an `AttributeError` in the tick loop.

        `content` is kept whole. It is the closest thing to a transcript a
        later reader has, and truncating it here would make the note in
        `schedule.jsonl` the only surviving copy of what the run said.
        """
        content = getattr(result, "content", "") or ""
        turns = int(getattr(result, "turns", 0) or 0)
        tools = list(getattr(result, "tool_calls", []) or [])
        hit_cap = bool(getattr(result, "hit_turn_cap", False))
        self_terminated = bool(getattr(result, "self_terminated", False))
        stopped = bool(getattr(result, "stopped_by_operator", False))
        reason = str(getattr(result, "termination_reason", "") or "")

        why = ""
        if hit_cap:
            why = reason or f"ran out of turns after {turns}"
        elif stopped:
            why = reason or "stopped before it finished"
        # `self_terminated` is deliberately not a failure branch. See the class
        # docstring: the agent deciding it is done is an ending, not a fault.
        return cls(
            content,
            ok=not (hit_cap or stopped),
            why=why,
            turns=turns,
            tools=tools,
            hit_turn_cap=hit_cap,
            self_terminated=self_terminated,
            stopped_by_operator=stopped,
        )

    def to_dict(self) -> dict[str, Any]:
        """The shape `schedule_run_due` reports per task.

        `content` is left out on purpose: the per-task rows ride inside a
        single JSON-RPC answer, and a tick that ran four tasks would otherwise
        carry four transcripts. The caller already has it if it wants it --
        this object is what it is holding.
        """
        return {
            "ok": self.ok,
            "why": self.why,
            "turns": self.turns,
            "tools": list(self.tools),
            "hit_turn_cap": self.hit_turn_cap,
            "self_terminated": self.self_terminated,
            "stopped_by_operator": self.stopped_by_operator,
            "chars": len(self.content),
        }

    def __repr__(self) -> str:
        return (f"TaskOutcome(ok={self.ok!r}, turns={self.turns!r}, "
                f"hit_turn_cap={self.hit_turn_cap!r}, why={self.why!r})")


class Service:
    def _persist_tool_state(self, spec: Any) -> None:
        """Write a tool's state change through to the store.

        Retiring or quarantining a tool is a judgement about the tool, and a
        judgement that lasts only until the process exits is not one: the next
        session would load the tool back as `active`. Measured before this
        existed -- `retire_tool` said ok, the registry said retired, and the
        database still had the row as active.

        A failure here used to be swallowed in silence, which made the count of
        durable judgements unanswerable: the caller saw `state: active`, the
        registry held `active`, and the row on disk still said `probation`, with
        nothing anywhere recording that the write had been refused. Counting it
        instead -- the same treatment `_ledger_locked_drops` gets -- keeps the
        difference between "written" and "intended" observable.
        """
        try:
            self.store.save_tool(spec, allow_overwrite=True)
        except Exception as exc:                               # noqa: BLE001
            self._state_persist_failures += 1
            n = self._state_persist_failures
            if n == 1 or n % 10 == 0:
                print(
                    f"autoforge: could not persist state "
                    f"{getattr(spec.state, 'value', spec.state)!r} for "
                    f"{getattr(spec, 'name', '?')!r} to the store "
                    f"({n} such failures this process): "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr, flush=True)

    def __init__(self, db_path: str | None = None) -> None:
        adopted_from = ""
        if db_path is None:
            db_path, adopted_from = _adopt_sibling_ledger(_home())
        self.db_path = db_path or os.path.join(_home(), "autoforge.db")
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.store = ToolStore(self.db_path)
        self.missions = MissionStore(self.db_path)
        self._adopted_from = adopted_from
        if adopted_from:
            # Recorded before anything else runs: the next reader has to be able
            # to tell "this service lives in data/" from "this service was told
            # to live in data/". Same sentence, different facts.
            self._record("home_adopted", {"home": _home(), "db_path": self.db_path,
                                          "sibling": adopted_from})
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
        # A per-service control plane, bound to this service's own connection.
        #
        # `autonomy/controls.py` has had `bind(conn)` since it was written and
        # nothing ever called it, so every plane came up `writing: False` and a
        # refused call was counted in memory and lost with the process. Measured
        # 2026-11-03: zero call sites for `controls.bind` in this package, and
        # `plane.summary()` reported `writing: False` on a live service whose
        # ledger held eight thousand rows.
        #
        # A plane per service rather than `controls.plane()` (the process-wide
        # one): two services in one process would otherwise share a plane, and
        # the second one's refusals would land in the first one's database. That
        # is worse than an empty ledger -- it is a receipt filed under somebody
        # else's name.
        self.controls = ControlPlane(conn=self.store.connection(), agent="autoforge")
        self.registry.controls = self.controls
        # One door out, pointed at this service's ledger. Without it the
        # gateway still decides -- refusing is not conditional on having
        # somewhere to write -- but the decision leaves no trace, and a
        # boundary with no record of what it admitted is one nobody can
        # audit. Per service, like the control plane, and for the same
        # reason: a process-wide gateway would file one service's
        # refusals under another one's name.
        egress_bind(self.store.connection(), name="autoforge")
        # The ledger is where every call ever made on this machine is recorded,
        # so it is the only honest source for "how long ago was this called".
        # `days_since_called` was written to consult this hook first and fall
        # back to `stats.last_called` -- but nothing ever assigned it, so the
        # hook was dead code and the fallback was the only path. That fallback
        # is nearly always empty: `stats` is only written when a tool is saved,
        # which a call does not do. Measured 2026-11-02: `exposure()` therefore
        # reported 0 hot and 66 cold out of 66 bridge tools, i.e. the split
        # would have demoted everything forever.
        #
        # `last_called_by_tool()` reads the whole ledger at once, while the
        # registry asks one name at a time. Snapshot it once per call so the
        # first question pays for the query and the next 65 are a dict lookup.
        self._last_called_cache: tuple[float, dict[str, float]] = (0.0, {})
        self.registry.last_called_source = self._last_called_ts
        # Bridge tools are not in the registry at all -- they live in the
        # `bridge_tools` table, reported by the JS side at mount. `exposure()`
        # only ever saw the forged tools, so wiring the split without this would
        # have produced a "hot/cold" answer about 6 of the 66 tools the model
        # actually carries.
        self._exposure_lock = threading.Lock()
        self._exposure_view: dict[str, Any] = {
            "hot": None, "cold": {}, "stats": {}, "window_days": 7.0,
            "built_at": 0.0, "source": "none",
        }
        # Brief accounting. `_brief_window_s` is how often a `brief_served`
        # row may be written; the calls inside the window are folded into the
        # next row as `calls_in_window`. Measured 2026-10-31: unthrottled, this
        # one counter produced 7090 of 7110 ledger rows.
        self._brief_lock = threading.Lock()
        self._brief_last_at = 0.0
        self._brief_count = 0
        self._brief_window_s = 600.0
        # Host probe for the per-turn block. `probe_host` is a set of stats, one
        # connect and one DNS lookup; the answers do not change between one turn
        # and the next, so paying for it on every assembly would buy the same
        # sentences at the price of a fresh look. The window is short on
        # purpose: this block exists to say what the machine looks like *now*,
        # and a stale "down" is worse than a fresh "down" -- it is what sends
        # the model looking for a cause that has already gone.
        self._probe_lock = threading.Lock()
        self._probe_cache: tuple[float, list[str]] = (0.0, [])
        self._probe_window_s = 60.0
        #: How many ledger writes this process could not make. Reported rather
        #: than swallowed -- see `_record`.
        self._ledger_locked_drops = 0
        #: How many state transitions reached the registry but not the store.
        #: A judgement that never landed is not a judgement -- see
        #: `_persist_tool_state`.
        self._state_persist_failures = 0
        #: How many wire-call meter writes this process could not make. Same
        #: argument as `_ledger_locked_drops`: a meter that quietly stops
        #: counting looks exactly like a surface nobody uses.
        self._wire_call_failures = 0
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
                spec = rec.to_spec(runner=self._runner_for(rec.code, entry))
                # Rows written before `tool_define` derived a scope carry
                # `''`, and `''` is read as `undeclared` -- every capability.
                # Backfilling here rather than rewriting the database keeps
                # the ledger's account of what was written intact: the row
                # still says what it said, and the scope in memory is a
                # reading of that row's code.
                if not spec.effect_signature:
                    spec.effect_signature = derived_scope(rec.code)
                self.registry.register(spec)
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
            "anchor_ledger": self.anchor_ledger,
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
            "runs_prune": self.runs_prune,
            "skill_port": self.skill_port,
            "tars_probe": self.tars_probe,
            "tars_skills": self.tars_skills,
            "tars_port_skills": self.tars_port_skills,
            "schedule_tick": self.schedule_tick,
            "schedule_run_due": self.schedule_run_due,
            "schedule_done": self.schedule_done,
            "schedule_cancel": self.schedule_cancel,
            "tool_define": self.tool_define,
            "tool_list": self.tool_list,
            "describe_tool": self.describe_tool,
            "report_bridge_tools": self.report_bridge_tools,
            "tool_exposure": self.tool_exposure,
            "review_tool": self.review_tool,
            "rehab_tool": self.rehab_tool,
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
        """Keep a fact; `tags` is the comma-separated list a human types.

        The store normalises it to the list its column declares, so a fact kept
        over the wire and a fact kept by a caller holding a list come back the
        same shape. Measured 2026-11-06: they did not -- `recall` answered `str`
        for one and `list` for the other.
        """
        if not str(key or "").strip():
            return {"ok": False, "key": "",
                    "error": ("key was empty: a fact nobody can name is "
                              "one nobody can look up, and it still takes "
                              "a line of the per-turn brief. Nothing was kept.")}
        self.store.remember(key, value, tags)
        return {"key": key}

    def recall(self, query: str = "", limit: int = 20) -> Any:
        return self.store.recall(query, limit)

    def forget(self, key: str) -> Any:
        """Forget one fact. Empty is a refusal, not a miss.

        `remember` will not keep a fact with no key (measured 2026-11-07:
        it used to, and the brief rendered it as `    : a value with no
        key`), so nothing can hold that key. Answering `forgotten: true`
        for it -- which this did -- claimed to have removed something that
        by construction cannot exist.
        """
        if not str(key or "").strip():
            return {"ok": False, "forgotten": False,
                    "error": ("key was empty: there is no such fact to forget, "
                              "and saying it was removed would be a claim "
                              "about a line that cannot be in the store")}
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
        # The head is part of the block, so it counts against the block's budget.
        # Measured 2026-11-02: counting only the body let a 1,246-char block sit
        # under a cap named 1,200. A budget that excludes the heading it pays for
        # is a budget with a hole in it.
        used = len(head)
        for r in rows:
            key = str(r.get("key") or "")
            value = " ".join((r.get("value") or "").split())
            if len(value) > self.MEMORY_ENTRY_CHARS:
                full_len = len(value)
                cut = value[: self.MEMORY_ENTRY_CHARS]
                # Prefer not to stop in the middle of a latin word. Chinese has
                # no spaces, so this often does nothing -- it is a courtesy, not
                # a rule, and it never shortens the entry by more than 30 chars.
                space = cut.rfind(" ")
                if space > self.MEMORY_ENTRY_CHARS - 30:
                    cut = cut[:space]
                # A bare ellipsis says "there was more" and nothing else: not how
                # much, not where to get it. Measured 2026-11-02, the head fact
                # was 746 chars and the prompt carried 242, and the 505 chars cut
                # away were exactly the half that says what to do. The pointer is
                # the whole point of the ellipsis.
                #
                # It does not repeat the key: the key is the 20-odd characters at
                # the start of this very line, and spelling it out again cost ~35
                # chars on every truncated entry, every turn.
                value = f"{cut} … (+{full_len - len(cut)} chars; recall this key)"
            line = f"    {key}: {value}"
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
        """What I owe, next to what I know -- and the clock gets looked at.

        The clock is checked here, on every turn, because this is the one place
        that runs on every turn. Upstream calls `self._sweep_missions()` at the
        top of `_effective_prompt` unconditionally; `MissionStore.report()`'s
        own comment explains why this store cannot do that -- a report that
        writes takes a write lock on the prompt's own path.

        Both halves are kept: the report stays a pure read, and the rule still
        gets looked at. `pending_transitions()` is one statement (0.056 ms, flat
        at 0 or at 50 open missions); `sweep()` costs 11.49 ms at 50 because it
        re-reads `last_activity` per mission. So the price is paid on the turns
        where the answer is not zero -- a handful of turns a week -- and never
        on the others.

        A clock check that cannot run must not cost the turn its mission block:
        the report is the orientation, and losing it to protect the rule would
        trade the thing that is read for the thing that is merely timely.
        """
        try:
            if self.missions.pending_transitions().get("total"):
                self.missions.sweep()
        except Exception:                                      # noqa: BLE001
            pass
        try:
            return self.missions.report().splitlines()
        except Exception as exc:                               # noqa: BLE001
            return [f"MISSION — unreadable: {exc}"]

    def _note_brief_served(self, budget_chars: int) -> None:
        """Record that the brief reached the model, once per throttling window.

        `_brief_window_s` exists so a host that polls the brief on a short timer
        does not turn the audit log into a heartbeat. Zero disables throttling
        and writes every call, which is what a debugging session wants and what
        a permanently running host does not.
        """
        now = time.time()
        with self._brief_lock:
            self._brief_count += 1
            elapsed = now - self._brief_last_at
            if self._brief_last_at and elapsed < self._brief_window_s:
                return
            repeats = self._brief_count
            self._brief_last_at = now
            self._brief_count = 0
        try:
            self._record("brief_served", {"budget_chars": budget_chars,
                                          "calls_in_window": repeats,
                                          "window_s": self._brief_window_s})
        except Exception:                                  # noqa: BLE001
            # A brief that cannot be counted is still a brief that must be
            # delivered: this block is the agent's own orientation, and
            # withholding it to protect a counter would be the wrong trade.
            pass

    def _probe_lines(self, refresh: bool = False) -> list[str]:
        """The host-probe block, cached for a window, degrading rather than raising.

        Upstream calls the same thing `_host_probe_lines` and puts it third in
        the per-turn prompt, between the skill menu and the memory block. The
        reason it belongs there is not completeness: this agent runs on one
        machine for weeks, so "is the store reachable", "are the skills on
        disk", "is the network up" are facts it would otherwise spend a *turn*
        re-deriving with `run_python`. A fact that costs a turn to check and
        never changes between turns is exactly what a prompt prefix is for.

        The cache is the second half of making it affordable. Measured
        2026-11-03 after the timeout fix: 61-214 ms per probe, mostly the dead
        market socket and a cold skill index. Re-probing on every assembly
        would spend that on sentences the previous assembly already carried.

        `refresh=True` bypasses the window. It exists because the one moment
        the cached answer is wrong is the moment someone just fixed the thing
        it reports -- and that is also the moment they are least willing to
        wait sixty seconds for the block to agree with them.

        A failure is a line, never an exception: a probe that cannot run must
        not cost the memory, missions and identity below it. The line says the
        probe failed, because going silent would read as "nothing to report",
        which is a different claim.
        """
        now = time.time()
        with self._probe_lock:
            at, cached = self._probe_cache
            if not refresh and at and (now - at) < self._probe_window_s:
                return list(cached)
        try:
            from . import tars_port
            lines = tars_port.probe_lines(brief=True)
        except Exception as exc:                               # noqa: BLE001
            # Deliberately not cached: a failure here is usually a transient
            # (a directory being replaced, a module still importing), and
            # caching it would turn one bad second into a bad minute.
            return ["## Host probe (read-only, this turn)", "",
                    f"- unavailable: {type(exc).__name__}: {exc}"]
        with self._probe_lock:
            self._probe_cache = (now, lines)
        return list(lines)

    def context_brief(self, budget_chars: int = 4000) -> dict[str, Any]:
        """The block a host injects into every request, as plain text.

        Exists because the first ported version left the memory behind a tool:
        the model could `recall` it, which means it had to decide to look. The
        complaint that produced agent.py's per-turn block was that a session
        with many tasks forgets the one sentence they are all for -- and an
        agent that has forgotten to look somewhere will not look.

        Every call is counted, but only one row is written per window. That is
        not bookkeeping for its own sake: whether this block reaches the model
        is a claim about the host's wiring, and the only way to check it from
        outside is to see the calls. An empty `brief_served` count in a session
        that has run turns means the plugin's refresh timer never fired, which
        is a different fault from the block rendering empty -- and before this
        counter the two were indistinguishable.

        Throttled, because the block is refreshed on a timer and the ledger was
        paying for every tick. Measured 2026-10-31 on this machine: 7090 of 7110
        event rows were `brief_served`, at roughly one per 5 seconds, while the
        remaining 20 rows were the entire real history of forges, tool calls
        and mission changes. The consequence was not just size: `my_history`
        filtered by a forge-ish kind returned nothing at all, because the
        signal it was asked for had been buried, and any "what did this agent
        do" question had to skip past thousands of identical heartbeats to
        reach an answer. A log whose dominant content is the fact that it is
        being read is not an audit trail.

        So: the first call of a window writes a row, and repeated calls inside
        that window update a count held in memory and folded into the next
        row. The ledger still proves the timer fired -- once per window rather
        than once per tick -- and stops being mostly about itself.
        """
        self._note_brief_served(budget_chars)
        # Host facts first, as upstream does (`agent.py:6248` puts `facts` ahead
        # of the skill menu and the memory): the prompt is a request *prefix*,
        # so the most stable content goes where a change below it cannot
        # re-bill it. The probe changes once a minute at most; a mission note
        # changes whenever one is written.
        lines = self._probe_lines()
        lines.append("")
        lines.extend(self._memory_lines())
        lines.append("")
        lines.extend(self._mission_lines())
        try:
            # `registry.report()` is the registry's own accounting, and the same
            # call `list_tools` renders -- counting states here by hand would be
            # a second opinion that can disagree with the tool's own output.
            #
            # Read `self.registry` DIRECTLY, never `self._registry()`: the
            # latter builds the forge pipeline as a side effect, and pipeline
            # construction now probes the gateway for a tool-capable model
            # (real HTTP, up to 90s). Measured 2026-11-04: that turned this
            # per-turn brief into a 10-65s call on a fresh process, which on
            # the single-threaded stdio loop wedged `tool_exposure` behind it,
            # blew the bridge's 1500ms budget, and silently disabled hot/cold
            # projection (chk_exposure.mjs: 27413 -> 27413, 0.0% saved). The
            # registry is fully populated in __init__ from load_all_tools();
            # reading it needs no model client.
            rep = self.registry.report()
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
        return ["", head, *body, *self._skill_state_lines()]

    def _skill_state_lines(self) -> list[str]:
        """Borrowed and skipped, the two facts the menu cannot carry.

        Upstream puts both in `_self_report` (`agent.py:4178`); this branch put
        neither anywhere, so the brief described a library that is entirely the
        agent's own and entirely readable. Both claims are false here: nine of
        the procedures are ported from Intelligence Indeed, and `scan()` has two
        ways to drop a file without raising.

        Why these two and not `loads_total` / `never_loaded`, which upstream
        also renders: those move whenever any skill is read, and this text is a
        prompt *prefix*. A count that changes per turn re-bills every character
        behind it -- the same argument that keeps the ledger's event count out
        of the block. The numbers are still readable through `skill_list` and
        `skill_errors`; the prefix keeps what does not churn.

        Both are measurements off the library that was just scanned, not
        restatements: the borrowed count is `tags` membership, the skipped count
        is `scan()`'s own two casualty lists. A line appears only when there is
        something to say -- an always-present "Borrowed: 0" would be one more
        sentence the model learns to skim.
        """
        out: list[str] = []
        try:
            lib = self._skill_library()
            lib.scan()
            report = lib.report()
            borrowed = [s for s in lib.all()
                        if skills_mod.TARS_SKILL_TAG in (s.tags or [])]
        except Exception:                                      # noqa: BLE001
            # The menu already rendered; a library that cannot be re-read must
            # not turn a partial answer into no answer.
            return out
        if borrowed:
            out.append(
                f"    Borrowed: {len(borrowed)} procedure(s) ported from "
                "Intelligence Indeed (tars, Apache-2.0), bodies unedited and "
                f"tagged '{skills_mod.TARS_SKILL_TAG}'. tars_skills lists them; "
                "tars_probe says whether the ported code still runs here.")
        unreadable = list(report.get("unreadable") or [])
        shadowed = list(report.get("shadowed") or [])
        if unreadable or shadowed:
            out.append(
                f"    Skills skipped: {len(unreadable)} unusable file(s), "
                f"{len(shadowed)} shadowed by another file with the same name -- "
                "skill_errors names both copies.")
        return out

    # -- missions ---------------------------------------------------------

    def mission_open(self, text: str, parent: int = 0, next_step: str = "",
                     blocked_on: str = "") -> Any:
        m = self.missions.open(text, parent, next_step, blocked_on)
        return _mission(m)

    def mission_note(self, mid: int, note: str = "", next_step: str | None = None,
                     blocked_on: str | None = None) -> Any:
        """Add a note, and/or change the next step, and/or clear the block.

        The two `None` defaults are load-bearing. `MissionStore.note` uses them
        to mean "do not touch this field", and `""` to mean "clear it"; passing
        `""` for both -- which this method used to do, always, positionally --
        made "leave it alone" unsayable from the wire. Measured 2026-11-06: a
        call carrying only a note cleared `next_step` and `blocked_on` on a
        mission that was waiting on an answer, so the note looked like progress
        and released the very exemption the block existed for.

        JSON already speaks this language: an omitted key never reaches
        `fn(**params)`, so the default applies. A caller who wants a field empty
        passes `""` and gets exactly that.
        """
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
        """Overturn a presumption of completion. One answer shape either way.

        Measured 2026-11-06: this used to answer with the mission dict on
        success and with `{"mid", "woken", "error"}` on refusal. So
        `result.get("woken")` said None when it worked and False when it
        declined -- a success that reads as a failure, which is how I came
        to report that waking left no ledger row. The row was written; the
        answer had hidden the event.

        Every other mission method lets `MissionError` reach the dispatcher,
        which turns it into `{"ok": false, "error": ...}`. This one keeps
        catching it, because "you cannot wake this" is a fact about the
        mission rather than a broken request -- but it now says so in the
        same keys as a success:

            woken    true or false, always present
            status   the mission status after the call, or "" if unknown
            mission  the mission dict, or null when nothing changed
            error    "" or the reason it refused
        """
        from .mission import MissionError
        try:
            m = _mission(self.missions.wake(mid, note))
        except MissionError as exc:
            status = ""
            try:
                status = str(self.missions.get(mid).status)
            except MissionError:
                pass                 # gone entirely; the reason says so
            return {"mid": mid, "woken": False, "status": status,
                    "mission": None, "error": str(exc)}
        return {"mid": mid, "woken": True, "status": m.get("status", ""),
                "mission": m, "error": ""}

    def mission_history(self, mid: int, limit: int = 50) -> Any:
        return {"mid": mid, "history": self.missions.history(mid, limit)}

    # -- skills -----------------------------------------------------------

    def skill_errors(self) -> Any:
        """Which skill files were dropped, and why -- both ways that happens.

        Two things were wrong here, and they hid each other.

        First, this crashed on exactly the files it exists to describe:
        `split_frontmatter` raises `SkillError` for a missing or unterminated
        `---` fence -- that raise is the module's stated design ("Raises
        SkillError rather than guessing when the fence is missing") -- and this
        called it outside any `try`. So one malformed file turned the whole
        answer into a traceback, and the tool whose job is "say which file is
        unusable" became the tool that cannot say anything.

        Second, it reported only unreadable files. `scan()` drops files a second
        way: a skill whose name is already taken goes to `self.shadowed` and its
        body is never used. The prompt line said "`skill_errors` says which and
        why" about those too, and this method did not mention them at all --
        a pointer to an explanation that was not there.

        Shadowing is reported as what it is: the file that lost, and the file
        that won. Not "less specific" -- `scan()` keeps whichever it meets
        first, so between two packages of the same name under one root the
        winner is decided by walk order, and calling that "more specific" would
        be a reason invented after the fact.
        """
        out: list[Any] = []
        try:
            lib = self._skill_library()
            lib.scan()
            shadowed = list(lib.shadowed)
        except Exception:                                      # noqa: BLE001
            shadowed = []
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
                try:
                    fm, _body = skills_mod.split_frontmatter(text, origin=str(path))
                except skills_mod.SkillError as exc:
                    # The parsing error IS the reason. Re-raising it here is how
                    # this tool failed at its own job.
                    out.append({"kind": kind, "path": str(path),
                                "error": str(exc)})
                    continue
                if not (fm or {}).get("description"):
                    out.append({"kind": kind, "path": str(path),
                                "error": "missing frontmatter description"})
        for name, winner, ignored in shadowed:
            out.append({
                "kind": "shadowed", "name": name,
                "path": ignored, "winner": winner,
                "error": (f"skill {name!r}: {ignored} was read first and is the copy "
                          f"in use; {winner} is ignored. Which one wins is the "
                          f"directory walk order, not specificity. Rename or remove "
                          f"one of them."),
            })
        return out

    def skill_forget(self, name: str) -> Any:
        # The walk this replaces kept its own naming rule, so it could archive
        # a file that the listing called something else -- and it answered with
        # the name it was *asked* for rather than the name it moved, which made
        # a case-insensitive lookup look like a rename.  `_skill_named` resolves,
        # `SkillLibrary.archive` moves the file and drops the row, and the reply
        # names what actually moved.
        # One library instance for both halves.  Resolving with one and
        # archiving with another looked correct and was not: `_skill_library()`
        # is built per call, `archive` reads `_skills`, and a fresh library has
        # scanned nothing -- so `skill_forget` refused with "no skill named
        # 'flat_one'" for a skill the listing had just shown.  Measured by
        # probe, first run after the resolver landed: the file stayed on disk.
        # Two instances of one answer is the same defect as two rules for one
        # name, one layer down.
        lib = self._skill_library()
        lib.scan()
        found = self._skill_named(str(name).strip().lower(), lib=lib)
        target = lib.archive(found.name)
        return {"name": found.name, "archived": target}

    def install_system_task(self, interval_minutes: int = 15, name: str = "") -> Any:
        """Register the recurring run with the OS, or raise saying why not.

        No envelope of its own.  This used to return
        ``{"ok": True, "result": ...}`` on success and ``{"ok": False, ...}``
        on failure, which the dispatcher then wrapped again -- so a refusal
        arrived as ``{"ok": true, "result": {"ok": false}}`` and a caller
        reading the outer key learned the call succeeded.  Measured
        2026-11-06: ``schedule.install_system_task`` raises ``ScheduleError``
        on every failure path and never returns a false ``ok``, so the nested
        boolean carried no information the exception did not already carry.

        Raising also makes this method agree with the eight ``mission_*``
        methods and with ``schedule_add``/``schedule_done``/``schedule_cancel``:
        one refusal shape on the wire, not four.
        """
        from . import schedule as schedule_mod
        return schedule_mod.install_system_task(int(interval_minutes), name)

    def system_task_command(self, interval_minutes: int = 15, name: str = "") -> Any:
        from . import schedule as schedule_mod
        return {"command": schedule_mod.system_task_command(int(interval_minutes), name)}

    # -- the gate, on the wire path -------------------------------------

    def _gate_method(self, method: str) -> dict[str, Any] | None:
        """Whether this wire method may run under the current policy.

        Returns None when it may, or a refusal payload when it may not.

        This is the half of P0-1 that the upstream table could not supply.
        `autonomy/scopes.py` says what each method touches; `autonomy/confirm.py`
        turns that into the freedoms it needs; this method is where the answer
        is consulted. Without it the table would be a dict nothing reads --
        which is exactly what `dependencies`, `topologies` and `anchor_chain`
        were before this session, and the reason each of them looked like a
        feature while doing nothing.

        Two differences from `registry._gate`, both deliberate:

        * No confirmer. There is nobody on the other end of a stdio line to
          answer a question, and `registry._gate` already treats "nobody to
          ask" as no. Asking would only lengthen the path to the same answer.
        * It never touches `spec.stats`, because a wire method is not a tool:
          it has no stats, no quarantine threshold, and no history window.

        An unlabelled method (`scope_for` returns `""`) is gated as
        `undeclared`, which needs every capability. That is the fail-closed
        direction on purpose: a method added without a scope row should be
        refused under a restricted policy, not silently permitted.
        """
        policy = self.registry.policy
        if policy is None or not policy.denied:
            return None
        from .autonomy.confirm import describe, disabled_freedoms

        needed = disabled_freedoms(scope_for(method), policy)
        if not needed:
            return None
        question = describe(needed, method, {})
        # Same receipt the tool gate writes. `registry._gate` has always called
        # `plane.gate(...)`; this half did not exist, so a refused *method* left
        # nothing at all. One vocabulary for both gates: a reader asking "what
        # was refused, and by which rule" should not have to know which of the
        # two doors the refusal came through.
        try:
            self.controls.gate(tool=method, needed=needed, outcome="no_operator",
                               question=question)
        except Exception:                                     # noqa: BLE001
            pass   # a refusal must not become a crash because it could not be logged
        return {"ok": False, "error": f"Denied: {question}",
                "awaiting_confirmation": True, "needed": needed,
                "scope": scope_for(method)}

    def _call_method(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Run one wire method by name, through the gate.

        `main()` dispatches through here rather than calling `self.methods[name]`
        directly, so that the gate is on the only path a request can take. A
        gate that a caller can route around is a gate that documents an
        intention instead of enforcing one.
        """
        fn = self.methods.get(method)
        if fn is None:
            return None
        refusal = self._gate_method(method)
        if refusal is not None:
            return refusal
        return fn(**(params or {}))

    # -- ledger -----------------------------------------------------------

    def log_event(self, kind: str, payload: dict[str, Any] | None = None) -> Any:
        # An empty kind is refused rather than written down. A row whose kind is
        # `""` is not an event anybody can group, filter or count by --
        # `event_kinds()` answers `[['', 1]]` and every reader has to
        # special-case that bucket forever. Measured 2026-11-07.
        if not str(kind or "").strip():
            return {"ok": False, "logged": None,
                    "error": "kind was empty: an event with no kind cannot be "
                             "counted, grouped or read back as anything."}
        self.store.log_event(kind, payload or {})
        return {"logged": kind}

    def events(self, limit: int = 20) -> Any:
        return self.store.get_events(limit)

    def report(self) -> Any:
        """What this service holds, plus whether it could write it down.

        `ledger_locked_drops` is this process's count of ledger writes it could
        not make. It is a field rather than a log line because a ledger that
        quietly loses rows is exactly the failure mode this project treats as
        fatal, and a caller asking "what is in there" should be able to see
        that the answer may be incomplete.

        `state_persist_failures` is the same question about a different write:
        a state transition that reached the registry but not the store. Left
        out, the two are indistinguishable in a report -- the tool reads
        `active` in memory and `probation` on disk with nothing saying which
        one a later session will believe. It will believe the disk.
        """
        rep = dict(self.store.report())
        rep["ledger_locked_drops"] = self._ledger_locked_drops
        rep["state_persist_failures"] = self._state_persist_failures
        # Which wire methods have actually been pressed, and -- the half a
        # counter cannot answer on its own -- which of the ones this build
        # declares have never been pressed. Measured 2026-11-06: 40 of 76 had no
        # suite and no other evidence, and nothing in this report could have
        # said so. `never` is computed here because only this object knows the
        # whole dispatch table.
        try:
            _wc = self.store.wire_call_view()
            rep["wire_calls"] = {
                "meter_failures": self._wire_call_failures,
                "pressed": len(_wc["pressed"]),
                "declared": len(self.methods),
                "never": sorted(set(self.methods) - set(_wc["pressed"])),
                "unknown": sorted(_wc["unknown"]),
                "failing": sorted(m for m, e in _wc["pressed"].items()
                                  if e["failures"] and not (e["calls"] - e["failures"])),
            }
        except Exception as exc:                               # noqa: BLE001
            rep["wire_calls"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
        # The control plane counts every judgement and keeps what it could not
        # write. Reporting `writing: False` here is the difference between "no
        # refusals happened" and "refusals happened and nothing recorded them",
        # and a reader has no other way to tell those apart.
        rep["controls"] = self.controls.summary()
        rep["egress"] = _egress_coverage()
        # Where this service thinks its state is, and whether it was told or
        # worked it out. Left implicit, "the ledger has three rows" and "the
        # ledger has three rows because this process adopted the only populated
        # one it could find" read the same -- and the second is a fact the next
        # reader needs.
        _map = state_map(_home())
        # The ledger is the one entry that can sit outside `home` -- `db_file`
        # adopts a populated sibling one level down -- and a reader who learns
        # that only by noticing a longer path will not learn it. Measured
        # 2026-11-06: with an adopted sibling, `home.dir` and every other
        # entry were under one directory while the ledger was not, so "back up
        # the home directory" missed it and "delete the home directory" left
        # it behind.
        #
        # Computed once and reused: two calls would be two chances to answer
        # differently inside one report, and a boolean that disagrees with the
        # list beside it is worse than no boolean.
        _outside = outside_home(_map)
        rep["home"] = {
            "dir": _home(),
            "db_path": self.db_path,
            "adopted_from": getattr(self, "_adopted_from", ""),
            # A list, not a boolean per entry. The reader's question is "what
            # do I have to copy to have copied everything", and with the
            # shipped preset the answer is not "home": `runs` and `skills`
            # are siblings of it. Measured online 2026-11-06 -- the previous
            # version of this block named the ledger and left those two silent.
            "outside_home": _outside,
            "state": _map,
        }
        # Which model the two model-using paths will talk to, and where that
        # came from. Left out, "the forge works" and "the forge works because
        # somebody happened to export AUTOFORGE_MODEL" are indistinguishable --
        # and the constant this replaced was exactly that confusion: a value
        # nobody could see, wrong on this host, in a report that said nothing
        # about models at all.
        try:
            route_model, route_base, route_key = _forge_route()
            named_at = "AUTOFORGE_MODEL" if os.environ.get("AUTOFORGE_MODEL") else _dsh_model()[1]
            rep["model"] = {
                "id": route_model,
                "from": named_at or "",
                "base": route_base,
                "key": bool(route_key),
                # Whether the model can see what a tool returned, once that has
                # been measured. None means "not asked yet": `report()` must stay
                # cheap, so it never pays for the probe -- it reports it.
                "reads_tool_results": _TOOL_READING.get(
                    (route_model, route_base), None),
                # What the last resolution did: "configured", "substituted: X",
                # "unverified" or "no alternative". Without it, a report naming a
                # model id cannot tell you whether that is the model this service
                # chose or the one somebody configured for it.
                "resolution": dict(_ROUTE_NOTE) if _ROUTE_NOTE else None,
                # Whether the gateway's own list mentions the configured id,
                # answered from the cache when something already asked, null
                # otherwise. Measured 2026-11-07: the DSH profile named
                # `qodercn/qfmodel`, the gateway served seven other ids, and a
                # report naming the id without this field could not tell its
                # reader that nothing would ever answer for it. Null means "not
                # asked", a different sentence from false -- `report()` stays
                # cheap and never pays for the /models call itself.
                "served_by_gateway": (
                    None if not _MODELS_CACHE.get("models")
                    else route_model in (_MODELS_CACHE.get("models") or [])),
            }
        except Exception as exc:                           # noqa: BLE001
            rep["model"] = {"id": "", "from": "", "error": f"{type(exc).__name__}: {exc}"}
        return rep

    def my_history(self, limit: int = 20, kind: str = "",
                   include_bookkeeping: bool = False) -> Any:
        """Read the ledger back: what was forged, called, amended, run.

        The result carries `kinds_available` on every call. `kind` used to be a
        free-text filter applied to an already-truncated page, so filtering on
        a name nobody had written returned an empty list that read like "this
        agent has done nothing" -- and there was no way to ask which names
        existed. Measured 2026-10-31: `my_history(kind="forge")` answered
        `{"count": 0}` immediately after a forge had been recorded as
        `tool_defined`.

        Bookkeeping kinds (`brief_served`) are excluded by default, because a
        ledger that is mostly its own heartbeat cannot be read as a history of
        the work. Pass `include_bookkeeping=True` to see them anyway.
        """
        try:
            n = max(1, min(int(limit), 500))
        except Exception:                                      # noqa: BLE001
            n = 20
        try:
            events = self.store.get_events(limit=n, kind=kind,
                                           exclude_bookkeeping=not include_bookkeeping)
        except TypeError:
            # An older store without the filter arguments: fall back rather than
            # fail, so a service running against a not-yet-migrated database
            # still answers.
            events = self.store.get_events(limit=n)
            if kind:
                events = [e for e in events if e.get("kind") == kind]
        available = []
        try:
            available = [{"kind": k, "count": c} for k, c in self.store.event_kinds()]
        except Exception:                                      # noqa: BLE001
            available = []
        return {"count": len(events), "events": events,
                "filter": kind or "(none)", "kinds_available": available}

    def verify_ledger(self) -> Any:
        """Walk the whole log and answer the one question this tool exists for.

        The top-level `ok` is the LEDGER's verdict, not the call's. It used to be
        the other way round -- `{"ok": True, "result": {"ok": False, ...}}` --
        and measured 2026-11-03 on a fixture with one tampered row, that is
        precisely the shape a reader skims: the first field says fine, the
        answer is inside the second. Two meanings of one word, and the one a
        machine reads first was the reassuring one.

        The two failure modes stay distinguishable, by which key is present:
        `error` means the check could not be run; `result` means it ran and this
        is what it found. A broken chain is not a failed call -- it is a
        successful call with bad news, and the news is the point.
        """
        try:
            res = self.store.verify_chain()
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": bool(res.get("ok")), "result": res}

    def anchor_ledger(self, note: str = "") -> Any:
        """Freeze the ledger's current head as a baseline a later reader can trust.

        `chaining.anchor()` has always written `chain_anchors`, and
        `ToolStore.anchor_chain()` has always been the one-line wrapper around
        it -- but nothing ever called the wrapper. The table sat at zero rows
        and the report said `anchors: 0` with no way to tell whether anchoring
        was broken or simply never asked for. Measured 2026-11-03: zero call
        sites in the whole package, and no `anchor*` method on this service.
        So this is not a new capability; it is the button that was missing.

        Anchoring does not repair anything and does not verify anything. It
        records "as of row N, the head was H, and the legacy rows digest to D",
        which is what makes a later `verify_ledger` able to distinguish a row
        that was rewritten from a row that was merely written before chaining
        began. Call it after a bulk import, or before handing the file to
        someone else -- a point you are willing to call a boundary.
        """
        try:
            res = self.store.anchor_chain(note=note)
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        # Recorded on the ledger like any other act, and after the anchor itself
        # so the note cannot be the only evidence that it happened: the row in
        # `chain_anchors` is primary, this is the index into it.
        self._record("ledger_anchored", dict(res, note=note))
        return {"ok": True, "result": res}

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
        """Read one procedure, or raise saying there is no such procedure.

        Two defects lived in the four lines this replaces, both measured on the
        wire 2026-11-06:

        1. The name comparison was exact while `skill_forget` was
           case-insensitive, so `skill_view("GOOD_ONE")` answered "not found"
           for a skill that `skill_forget("GOOD_ONE")` would happily archive.
           A name that cannot be read but can be destroyed is the worse half of
           being case-insensitive.  `valid_name` is the rule -- lower-case,
           digits, hyphens, underscores -- so matching it case-insensitively
           costs nothing and removes the asymmetry.

        2. "not found" was returned as `{"error": ...}`, which the dispatcher
           wrapped into `{"ok": true, "result": {"error": "not found"}}`: one
           reply, two `ok`s, the outer one claiming the call succeeded.  The
           same file already answered a *third* way when the frontmatter was
           unreadable -- `split_frontmatter` raises `SkillError`, which reaches
           the dispatcher and produces `ok: false`.  So a missing skill was a
           success and a broken skill was a failure, decided by which of two
           branches ran rather than by what happened.  Raising makes all three
           paths agree with the eight `mission_*` methods and the four
           `schedule_*` ones.
        """
        found = self._skill_named(str(name).strip().lower())
        path = found.path
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError as exc:
            # Unreadable is a refusal with a reason, not "no such skill": the
            # name matched, so the file is there and could not be read.
            raise skills_mod.SkillError(
                f"skill {found.name!r} exists at {path} but could not be read: {exc}")
        # Counted under the name the library gave it, not under what was asked
        # for: a counter row keyed by a differently-cased name is a row the menu
        # will never match, so the usage would silently vanish.
        self.store.skill_load(found.name)
        fm, body = skills_mod.split_frontmatter(text)
        return {"name": found.name, "frontmatter": fm or {}, "body": body}

    def _skill_named(self, wanted: str, lib: Any = None) -> Any:
        """The skill whose name is `wanted`, case-insensitively, or raise.

        One resolver, because there were three and they disagreed.  Measured on
        the wire 2026-11-06 with three files on disk:

          * `alpha/dir_named_pkg/SKILL.md` declares `name: declared_name`.  The
            listing called it `declared_name`; `skill_view` called it
            `dir_named_pkg`, and refused `declared_name`.
          * `alpha/undeclared_pkg/SKILL.md` declares no name.  The listing did
            not show it at all (`path.stem` is "SKILL", which `valid_name`
            rejects, so the file fell into `errors`); `skill_view` served it
            happily as `undeclared_pkg`.
          * `flat_one.md` agreed, which is why this went unnoticed: the flat
            layout is the only one the agent writes itself, and it is the only
            layout where all three rules coincide.

        A name that lists, a name that reads and a name that can be archived
        have to be the same name, or "no such skill" says which code path was
        asked rather than what the library holds.

        `scan()` rather than a directory walk, for the second half of the same
        problem: `skill_load` ignores a name that has no row, so a skill that
        had never been listed could be read any number of times and still
        report "never used".  Measured: `skill_write` then `skill_view` left
        no row at all and `loads = 0`.

        `lib` is for callers that need to act on the same instance they
        resolved with -- see `skill_forget`.  Omitted, this builds and scans
        its own, which is what a read-only caller wants.
        """
        if lib is None:
            lib = self._skill_library()
            lib.scan()
        for skill in lib.all():
            if skill.name.lower() == wanted:
                return skill
        raise skills_mod.SkillError(f"no skill named {wanted!r}")

    def skill_write(self, name: str, description: str = "", body: str = "",
                    when_to_use: str = "", tags: str = "") -> Any:
        """Save a procedure, or raise saying why it was not saved.

        Each of the four refusals below used to be returned as
        `{"name": ..., "error": ...}`, so the wire answered
        `{"ok": true, "result": {"error": "description is required"}}` --
        measured 2026-11-06, all four.  A write that did not happen is not a
        successful call whose result mentions a problem; it is a call that did
        not happen, and the caller has to be able to tell the two apart without
        reading prose.  Same shape as `schedule_add`'s refusal, fixed the same
        way: raise, let the dispatcher wrap it once.
        """
        if not skills_mod.valid_name(name):
            raise skills_mod.SkillError(
                f"invalid name {name!r}: lower-case letters, digits, '-' or '_' (max 64)")
        if not description.strip():
            raise skills_mod.SkillError(
                f"no description for {name!r}: a procedure nothing can decide "
                "to load is a file, not a skill")
        if not body.strip():
            raise skills_mod.SkillError(f"no body for {name!r}: an empty procedure")
        dirs = skills_mod.default_skill_dirs()
        if not dirs:
            raise skills_mod.SkillError(
                "no skill directory configured: set AUTOFORGE_SKILLS_DIRS or rely "
                "on the home directory this service reports")
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
        # `self.registry`, not `self._registry()`: the latter forces pipeline
        # construction (a gateway probe) just to count rows. See context_brief.
        rep = self.registry.report()
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
        if model:
            # The forge needs tools, so it needs a model that can read what a
            # tool returned -- not merely one that answers. Measured 2026-11-04:
            # the model DSH's own profile names emits tool calls and then
            # reports that every one of them returned nothing.
            model, _route_why = _tool_capable_model(model, base, key)
            if _route_why.startswith("no alternative"):
                raise RuntimeError(_tool_read_note("A forge", base))
            if _route_why.startswith("substituted"):
                # One row per resolution. A reader of the ledger needs to know a
                # forge ran on a model nobody chose, and when.
                self._record("model_substituted",
                             {"configured": _ROUTE_NOTE.get("configured", ""),
                              "used": model, "why": _route_why})
        if not model:
            # Refused before the client is built, and refused in words a person
            # can act on. The alternative -- shipping a plausible-looking id --
            # is what this file did before, and the failure arrived as an
            # unrelated-looking 404 from inside a probe.
            raise RuntimeError(
                "no model for the forge pipeline: $AUTOFORGE_MODEL is unset and "
                "no DSH profile names one (%s). Set AUTOFORGE_MODEL to a model "
                "id the gateway serves -- %s lists them."
                % (_dsh_model()[1], base.rstrip("/") + "/models"))
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
                self._record(kind, payload)
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
        # NOTE: builds the forge pipeline (and therefore probes the gateway
        # for a tool-capable model) as a side effect. Pure reads --
        # context_brief, list_tools -- must use `self.registry` directly.
        # Measured 2026-11-04: routing a per-turn read through here wedged
        # tool_exposure behind a 10-65s brief and disabled hot/cold projection.
        # The heavy forge/evaluate paths build the pipeline explicitly anyway.
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

    #: How much of a tool's arguments a `call` row carries.
    #:
    #: The ledger answers "what did I actually run, with what, and how did it
    #: go". For eighteen months of this project the answer was `{"tool": ...,
    #: "ok": ...}` and nothing else: measured 2026-11-02, all 15 `call` rows in
    #: the live database had exactly those two keys. So a failure could not be
    #: reproduced, a slow call could not be identified, and `record_call`'s
    #: `history` -- the in-memory copy -- was the only place an error string
    #: existed at all, which is not a place that survives a restart.
    #:
    #: Not the whole argument value, though: `tool_define` carries the tool's
    #: source, so one forge would put several thousand characters of Python into
    #: an append-only table. The row keeps the shape (which arguments, of what
    #: type, how big) and at most `CALL_ARGS_CHARS` of the value.
    CALL_ARGS_CHARS = 160
    CALL_ARGS_MAX_KEYS = 12

    def _arg_digest(self, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """What a `call` row keeps about its arguments.

        Three facts per argument, in this order of importance: its name (to tell
        two calls apart), its size (to explain a slow or an oversized call), and
        a bounded prefix of its value (enough to recognise the call, not enough
        to clone the table). Type is written even when the value is short,
        because `"3"` and `3` are the bug this project has chased most often.
        """
        if not arguments:
            return {}
        out: dict[str, Any] = {}
        for k in list(arguments)[: self.CALL_ARGS_MAX_KEYS]:
            v = arguments[k]
            name = str(k)
            if isinstance(v, str):
                entry = {"type": "str", "chars": len(v)}
                if len(v) <= self.CALL_ARGS_CHARS:
                    entry["value"] = v
                else:
                    entry["head"] = v[: self.CALL_ARGS_CHARS]
            elif isinstance(v, (int, float, bool)) or v is None:
                entry = {"type": type(v).__name__, "value": v}
            elif isinstance(v, (list, tuple, dict)):
                entry = {"type": type(v).__name__, "items": len(v)}
            else:
                entry = {"type": type(v).__name__}
            out[name] = entry
        if len(arguments) > self.CALL_ARGS_MAX_KEYS:
            out["…"] = {"type": "more", "items": len(arguments) - self.CALL_ARGS_MAX_KEYS}
        return out

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        """One event on the ledger. Best-effort in outcome, NOT silent in fact.

        The service has no in-memory trace -- there is no run to have a
        timeline of -- so this is the store leg only. A store that is locked or
        full must not turn a finished forge into a failed one: the tool is
        already registered by the time this runs, and the cost of a lost event
        is a gap in the ledger, not a broken task.

        But the loss is counted. It used to be swallowed whole, and that made
        `success_rate` monotonically optimistic: measured 2026-10-31, a tool
        showed `sr=1.0 calls=18` after two real calls had both failed at the
        point of being recorded. The failures were real; the record of them
        was not, because the recording was the thing that failed. An audit log
        that silently drops the rows where it could not write is an audit log
        whose silence reads as success -- which is the exact failure this
        project was built around.
        """
        try:
            self.store.log_event(kind, payload)
        except Exception as exc:                               # noqa: BLE001
            self._ledger_locked_drops += 1
            if self._ledger_locked_drops in (1, 10, 100) or self._ledger_locked_drops % 1000 == 0:
                # No print spam: a stride, plus the shape of the failure, so a
                # log that stops growing has a visible reason.
                try:
                    print("[autoforge] ledger write failed (%d dropped): %s: %s"
                          % (self._ledger_locked_drops, type(exc).__name__, exc),
                          file=sys.stderr, flush=True)
                except Exception:                              # noqa: BLE001
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
            # One door out. The refusal raises GatewayDenied and falls into the
            # except below, which records the failure -- so a denied host is not
            # quietly retried as though the network were down.
            _admit_egress(req.full_url, tool="market_search")
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
        from .store import home_dir

        home = home_dir()
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
                _admit_egress(req.full_url, tool="peer_lookup")
                with _url.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8", "replace"))
            except Exception as exc:                           # noqa: BLE001
                try:
                    req = _url.Request(base + "/resources", headers=headers)
                    _admit_egress(req.full_url, tool="peer_lookup")
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
                _admit_egress(req.full_url, tool="discover_upstream")
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
            _admit_egress(req.full_url, tool="prelookup_market")
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
        # An empty need is refused here, before anything is spent on it.
        # Measured 2026-11-07: `forge_tool('')` ran 4,124.6 ms -- 4,132.8 of it
        # inside `_prelookup_market`, dialling a toolmarket that is down -- and
        # only then reported a stub forge reaching a session with no model. The
        # sentence was true and it was about the wrong thing. `run_python('')`
        # refuses in 0.0 ms with "code was empty", and two entry points that
        # disagree about what an empty required string means will disagree
        # about the next one too. A guard that runs after the side effects is
        # not a guard, it is a comment.
        if not str(need or "").strip():
            return {
                "ok": False,
                "text": "A forge needs a need: one sentence describing the capability "
                        "you want. Nothing was forged.",
                "error": "need was empty",
            }
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
        self.registry.register(spec, replace=True)
        try:
            self.store.save_tool(spec)
            persisted = True
        except Exception as exc:                               # noqa: BLE001
            persisted = False
            # Not `ok: True`. A forge that did not reach the library is a forge
            # that dies with this process, and a caller told "ok" would go on
            # to rely on a tool the next session cannot see. Measured
            # 2026-10-31 on this machine: the reply was
            #   {"ok": true, "state": "active", "text": "... but did not save it:
            #    database is locked"}
            # while `tools` held zero rows and `list_tools` showed nothing.
            # The text stays, because it is the useful part; `ok` changes, and
            # `persisted` is a field a caller can branch on.
            state = str(getattr(getattr(spec, "state", None), "value", ""))
            self._record("forge_tool", {"need": need, "tool": spec.name,
                                        "persisted": False,
                                        "error": f"{type(exc).__name__}: {exc}"})
            return {"ok": False, "name": spec.name, "state": state,
                    "rounds": res.rounds, "persisted": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "text": f"Forged {spec.name!r} [{state}] in memory only, NOT saved: "
                            f"{type(exc).__name__}: {exc}. It is callable now and will "
                            f"be gone when this process exits. Fix the store "
                            f"(see store.py busy_timeout / WAL) and forge again."}
        state = str(getattr(getattr(spec, "state", None), "value", ""))
        return {"ok": True, "name": spec.name, "state": state, "rounds": res.rounds,
                "persisted": persisted,
                "text": f"Forged {spec.name!r} [{state}]: {spec.description}"}

    def evaluate_tool(self, name: str, promote: bool = True) -> Any:
        """Run the full battery on an existing tool and print it as agent.py does.

        The tool's own declared sample travels with it when it was forged with
        one; without it the battery invents arguments, and a verdict on invented
        arguments is one nobody earned.

        This is also the promotion path, and that sentence used to be a comment
        describing behaviour that did not exist: `tool_define` placed every tool
        on PROBATION telling the reader `evaluate_tool` would lift it, while this
        method only *printed* a verdict and never touched `spec.state`. The
        result was a one-way door -- measured 2026-11-01, `file_facts` passed 6/6
        on three separate evaluations and sat on PROBATION throughout, with zero
        `promote` rows on the ledger. A state diagram whose `earn` edge has no
        implementation is the same defect this framework exists to catch, one
        level up.

        Two transitions, both recorded, and neither is a punishment:
          PROBATION -> ACTIVE   the battery passed, so the seal is earned
          ACTIVE -> PROBATION   the battery failed, so the seal is withdrawn
        A failure here is not evidence of harm -- that is `quarantine`'s job --
        so nothing is hidden or disabled, only demoted to where it stays
        callable and visible. `promote=False` runs the battery and reports it
        without moving anything, for a caller who wants a reading, not a verdict.
        """
        # `self.registry`, not `self._registry()`: the latter builds the forge
        # pipeline as a side effect, which probes the gateway for a model that
        # can read tool results. A name that does not exist cannot be helped by
        # any model, so the lookup has to answer before anything is built.
        # Measured 2026-11-12: `evaluate_tool("no_such_tool")` raised
        # RuntimeError about the gateway instead of answering about the missing
        # tool, because the lookup went through the heavy path first.
        spec = self.registry.get(name)
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

        moved = self._settle_state_after_evaluation(spec, report, promote)
        payload: dict[str, Any] = {
            "ok": bool(report.passed), "passed": bool(report.passed),
            "state": getattr(spec.state, "value", str(spec.state)),
            "text": f"{report.summary()}\n" + "\n".join(lines),
        }
        if moved:
            payload["state_changed"] = moved
            payload["text"] += (
                f"\n  -> state {moved['from']} -> {moved['to']}: {moved['why']}")
        return payload

    def _settle_state_after_evaluation(
        self, spec: Any, report: Any, promote: bool,
    ) -> dict[str, Any] | None:
        """Move the state to match the verdict, or report why it stayed put.

        Kept apart from the printing code above so the transition has exactly
        one implementation, reachable without reading the formatting.
        """
        current = getattr(spec, "state", None)
        if not promote:
            return None
        passed = bool(getattr(report, "passed", False))
        if passed and current is ToolState.PROBATION:
            move, why = "promote", "battery passed; the seal is earned"
        elif passed and current is ToolState.QUARANTINED:
            # `rehab`, not `promote`. A quarantine was earned by a run of
            # failures in real calls; one passing battery does not outweigh
            # that, and jumping straight to ACTIVE would launder a bad record
            # into a seal. Back on trial -- callable, visible, unsealed -- is
            # what the evidence supports.
            #
            # This branch is the missing half of the escape hatch the service
            # already advertised. Measured 2026-11-01: `tool_call` refuses a
            # QUARANTINED tool, `force` was not reachable from the bridge, and
            # `rehab` had no caller anywhere in the package -- so a tool the
            # automatic quarantine caught was dead from the model's side
            # forever, while the persona promised a way back.
            move, why = "rehab", (
                "battery passed; back on trial, not sealed -- a quarantine is "
                "cleared by evidence, not by a verdict")
        elif not passed and current is ToolState.ACTIVE:
            move, why = "demote", "battery failed; the seal is withdrawn"
        else:
            return None
        before = getattr(current, "value", str(current))
        try:
            # Through the registry's funnel, never by assigning the attribute:
            # `_set_state` is the single place a transition is announced and
            # persisted, and a second path is how a judgement comes to expire
            # with the process that made it.
            if move == "promote":
                self.registry.promote(spec.name)
            elif move == "rehab":
                self.registry.rehab(spec.name)
            else:
                self.registry.demote(spec.name)
        except Exception as exc:                               # noqa: BLE001
            return {"from": before, "to": before,
                    "why": f"could not move state: {type(exc).__name__}: {exc}"}
        after = getattr(spec.state, "value", str(spec.state))
        try:
            self._record("state_change", {
                "tool": spec.name, "from": before, "to": after,
                "move": move, "why": why, "passed": passed,
            })
        except Exception:                                      # noqa: BLE001
            pass
        return {"from": before, "to": after, "why": why}


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
                                     on_event=lambda k, pl: self._record(k, pl))
            report = failure_report or spec.verification.get("failed_detail", "underperforming")
            baseline = self._baseline_for(spec)
            result = engine.evolve(spec, report, baseline=baseline, context="internal")
        except Exception as exc:                               # noqa: BLE001
            # `tried` is answered even here, and for a reason: measured
            # 2026-11-07, `chk_five.py` run alone ten times failed twice with
            # exactly this answer and no `tried` at all, and `chk_report_cost`
            # / `chk_five` callers that ask "how many candidates were there"
            # then have nothing to read. One run of the same probe also hung
            # past 300 s inside the breed step. A caller cannot tell a breed
            # failure from a breeding step that produced nothing unless both
            # answers carry the same keys, so this one says `tried: None` --
            # not `0`, which would be a claim that nothing was generated.
            return {"ok": False, "text": f"Could not evolve {name!r}.",
                    "improved": False, "tried": None,
                    "error": f"{type(exc).__name__}: {exc}"}
        self._record("evolve", {
            "tool": name, "improved": result.improved,
            "vetoed": [m.generated.name for m in result.mutants if not m.admissible]})
        if not result.improved:
            vetoed = sum(1 for m in result.mutants if not m.admissible)
            tail = f" {vetoed} vetoed by the validity gate." if vetoed else ""
            tried = len(result.mutants)
            if not tried:
                # Zero candidates is not a verdict on the candidates. Measured
                # 2026-11-07: `evolve_tool` on a real tool with a healthy
                # pipeline answered "no admissible mutant beat the original
                # (0 tried)" after 40.3 s -- a sentence about a comparison in
                # which nothing was ever compared. Same shape as `find_gaps`
                # calling a broken pipeline "disabled": the reader is told the
                # attempt failed on the merits when in fact it never produced
                # anything to judge. Say which one happened.
                return {"ok": False, "improved": False, "tried": 0,
                        "text": (f"Evolved {name!r}: the engine produced no candidate "
                                 f"at all, so nothing was compared. That is a fault in "
                                 f"the breeding step, not a judgement that the tool is "
                                 f"already good." + tail)}
            return {"ok": False, "improved": False, "tried": tried,
                    "text": (f"Evolved {name!r}: no admissible mutant beat the original "
                             f"({tried} tried)." + tail)}
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
        """Analyse the library and propose the capabilities it is missing.

        Every failure used to be reported as "Meta-cognition is disabled.", and
        there is no disabled state in this method at all -- no flag, no config
        read, nothing that can be switched off. Measured 2026-11-07 with a
        pipeline that raises on any attribute access: the reply was that
        sentence plus `RuntimeError: pipeline is broken on purpose` in `error`.
        `text` is the half a model reads, and it said a feature was off when the
        feature had tried and broken. "Off" tells a reader to stop asking;
        "broke" tells them the reason is worth reporting. Phases are named here
        for the same reason: a pipeline that will not build, an analysis that
        raised, and a module that would not import are three different problems
        and a caller can act on exactly one of them at a time.
        """
        phase = "import"
        try:
            from .forge.metacog import MetaCognition
            phase = "pipeline"
            pipeline = self._forge_pipeline()
            phase = "analyser"
            metacog = getattr(self, "_metacog", None)
            if metacog is None:
                metacog = MetaCognition(pipeline.verifier.llm, self.registry, pipeline)
                self._metacog = metacog
            phase = "analysis"
            rep = metacog.analyse()
        except Exception as exc:                               # noqa: BLE001
            # `_record`, not a bare store write: the reply below is the work,
            # and a store that refuses must not turn a diagnosis into a crash.
            self._record("find_gaps_failed", {"phase": phase,
                                              "error": f"{type(exc).__name__}: {exc}"[:400]})
            return {"ok": False, "phase": phase,
                    "text": (f"Meta-cognition could not run: it failed while "
                             f"{_GAP_PHASES.get(phase, phase)}. This is a fault, "
                             f"not a setting -- nothing switches meta-cognition off, "
                             f"so there is no flag to turn back on."),
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
        that demands a rationale. It also refuses builtins: nothing in this
        package can register one (every Agent is built with
        `allow_self_terminate=False`), but a builtin that did exist must
        not be retirable by an operator who mistook it for a forged tool.
        """
        if not rationale or not rationale.strip():
            return {"ok": False, "text": "A rationale is required to retire a tool."}
        spec = self.registry.get(name)
        if spec is None:
            return {"ok": False, "text": f"No tool named {name!r}."}
        if getattr(spec, "source", "") == "builtin":
            # Kept as a guard even though nothing can reach it today: `source`
            # is set to "builtin" in exactly one place (`core/agent.py`, the
            # terminate tool), and that registration only runs when an Agent is
            # built with `allow_self_terminate` -- which every construction in
            # this package sets to False. Measured 2026-11-07: production `tools`
            # holds `source` distribution `{"human": 6}`, no builtins.
            #
            # The sentence used to end "Disable it with set_autonomy instead",
            # and there is no such method -- not among the 76 wire methods, not
            # an attribute on Service, not exported by the autonomy module. A
            # refusal whose advice is a call that does not exist leaves the
            # caller worse off than a plain no. So it says the plain no, plus
            # the fact that no switch exists here rather than inventing one.
            return {"ok": False, "text": (
                f"Refusing to retire {name!r}: it is a builtin meta-tool, and "
                "builtins are not retired. This service exposes no method to "
                "disable one either, so there is no alternative call to offer.")}
        self.registry.retire(name)
        # `_record`, not `self.store.log_event`. Measured 2026-11-01 22:35 with a
        # store that raises on every write: the retirement itself had completed
        # (the registry is a dict, and the state change went through the funnel),
        # but this line re-raised and the caller saw `OperationalError` instead
        # of the result. The same defect `tool_call` had, one line over: the
        # work was done and the note about it was allowed to cancel the answer.
        # A retirement that cannot be written down is still a retirement.
        self._record("retire", {"tool": name, "rationale": rationale})
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
        """Register a task. A bad `when` or `repeat` raises.

        The refusal is an exception, not `{"error": ...}` inside a successful
        reply. Measured 2026-11-06: this method (with `schedule_done` and
        `schedule_cancel`) answered `{"error": ...}` while the transport said
        `{"ok": true}` -- two `ok`s in one reply, the outer one reporting
        success for a refusal. `verify_ledger` carried the identical defect
        until it was made to answer `{"ok": False}`, and the eight mission
        methods never had it: their `MissionError` reaches the dispatcher,
        which answers `{"ok": false, "error": "MissionError: ..."}`.

        So all three now behave like everything else here: a refusal and a
        success cannot share a transport status.
        """
        task = self._scheduler().add(text, when, repeat or 0)
        return {"id": task.id, "text": task.text, "due_at": task.due_at,
                "repeat_seconds": task.repeat, "line": task.line()}

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

        It also counts what is past the TTL, and that is a notice rather than a
        cleanup on purpose. `runs.prune` exists and is reachable; nothing calls
        it on a schedule, and this is the decision, not an oversight. Upstream's
        CLI offers `runs list|resume|drop|prune` with no timer either, deletion
        cannot be undone, and a record removed by a machine nobody was watching
        cannot be argued about afterwards -- which is the entire reason
        `prune_detail` reports what it refused instead of returning a bare list.
        So the reader is told "these three are past seven days, `runs_prune`
        removes them", and the reader decides. The count comes from the rows
        already scanned instead of a second walk: one scan, one answer.
        """
        from . import runs as _runs

        rows = _runs.scan()
        ttl_s = _runs.DEFAULT_TTL_S
        expired = [r["run_id"] for r in rows if r["age_s"] > ttl_s]
        hint = ""
        if expired:
            hint = (f"{len(expired)} run record(s) are past "
                    f"{ttl_s / 86400.0:.0f} day(s) and can be removed with "
                    f"runs_prune. Nothing removes them on its own.")
        return {"journal": str(_runs.journal_dir()),
                "count": len(rows),
                "runs": rows,
                "resumable": [r["run_id"] for r in rows
                              if r["status"] == "resumable"],
                "ttl_days": round(ttl_s / 86400.0, 3),
                "past_ttl": expired,
                "hint": hint}

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
            return {"ok": False, "archived": None,
                    "error": "run_id is required"}
        if not str(why or "").strip():
            return {"ok": False, "archived": None,
                    "error": "a reason is required: this is abandonment, not "
                             "completion, and the note is the difference"}
        hits = [m for m in _runs.unfinished() if m.run_id.startswith(run_id)]
        if len(hits) != 1:
            return {"ok": False, "archived": None,
                    "error": f"{run_id!r} matched {len(hits)} run(s)"}
        mark = hits[0]
        journal = _runs.journal_dir()
        src_path = journal / f"run-{mark.run_id}.json"
        if not src_path.exists():
            return {"ok": False, "archived": None,
                    "error": f"{src_path} is already gone"}
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
            return {"ok": False, "archived": None,
                    "error": f"could not archive {src_path}: {exc}"}
        return {"archived": mark.run_id,
                "task": mark.task,
                "why": why,
                "moved_to": str(dst_path),
                "restore": f"move it back to {journal} to make it resumable again"}

    def runs_prune(self, ttl_days: float = 7.0) -> Any:
        """Delete run records past their TTL, and say what was kept and why.

        `runs.prune` has existed since the port and nothing ever called it --
        measured 2026-11-04: `prune(` appears once in the whole package, at its
        own definition. A seven-day retention that no code enforces is not a
        policy, it is a comment that happens to be executable.

        Wiring it to a tool rather than to the tick is the deliberate half. The
        tick runs unattended, and an unattended deleter over a directory whose
        whole purpose is to outlive the process that wrote it is a bad trade: a
        record removed by a machine nobody was watching cannot be argued about
        afterwards. So the deletion is a decision somebody makes with the counts
        in front of them -- and `runs_forget` remains the way to act on one
        record without touching the rest.

        Returns `runs.prune_detail`, not the id list, because "0 removed" alone
        reads as good news in both of the cases it can mean: nothing was due,
        or everything was refused. `kept` says what survives and by which
        verdict; `failed` names each record the filesystem would not release.
        That last part is not a corner case -- measured on forty expired
        records, 132.5 ms of the 157.3 ms was the unlink calls, so refusal is
        the bulk of the work and it used to be the part with no voice.
        """
        from . import runs as _runs

        try:
            ttl_s = float(ttl_days) * 86400.0
        except (TypeError, ValueError):
            return {"error": f"ttl_days wants a number of days, not {ttl_days!r}"}
        if ttl_s <= 0:
            return {"error": "a TTL of zero or less would delete records that "
                             "are still current; give a positive number of days"}
        report = _runs.prune_detail(ttl_s=ttl_s)
        removed = len(report["removed"])
        refused = len(report["failed"])
        kept = sum(report["kept"].values())
        self._record("runs_pruned",
                     {"ttl_days": ttl_days,
                      "removed": removed,
                      "failed": refused,
                      "kept": kept})
        if refused:
            tail = (f"; {refused} refused -- the failures list names them, "
                    f"and they are still on disk")
        else:
            tail = ""
        return dict(report,
                    journal=str(_runs.journal_dir()),
                    text=(f"removed {removed} record(s) past {float(ttl_days):g} "
                          f"day(s); {kept} kept{tail}"))

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
        # An unknown source is a refusal.  It used to be returned as
        # `{"error": ...}`, which the dispatcher wrapped into `ok: true` -- so
        # `skill_port(source="nonsense")` looked like a successful port that
        # reported zero, when in fact nothing was attempted (measured
        # 2026-11-06).
        raise skills_mod.SkillError(
            f"unknown source {source!r}; expected 'tars' or 'openclaw'")

    def tars_probe(self) -> Any:
        """What was borrowed from Intelligence Indeed, and does it still work HERE.

        `vendor/tars` has been in the tree since the port, `tars_port.py` can
        verify it against its copy-time digests, probe the host and index the
        nine borrowed procedures -- and measured 2026-11-03 nothing on the wire
        could reach any of it. The only tars-shaped method was
        `skill_port(source="tars")`, which writes nine files; the read-only half
        had no entry point at all. A capability nothing can call is a file in a
        tree.

        Read-only by construction, so it is safe to run before deciding
        anything, which is the whole point of a probe. `scope: read_only` is
        derived from its code (`autonomy.scopes.derived_scope`), not asserted
        here.

        Split from `tars_port_skills` on purpose: turning borrowed procedures
        into entries in my own library writes files under the skills directory,
        and a tool named like a read that quietly minted nine of them would make
        its `effect_signature` a lie -- the very signature the autonomy gate
        reads.
        """
        from .tars_port import PortedSkillSystem, probe_host, provenance, verify_vendor

        prov = provenance()
        check = prov.get("verbatim_check") or {}
        lines = ["tars port: " + str(prov.get("upstream"))]
        lines.append("  licence: " + str(prov.get("licence")))
        lines.append("  vendored at: " + str(prov.get("vendored_at")))
        lines.append("  copied verbatim: " + ", ".join(prov.get("copied_verbatim") or []))
        lines.append("  not copied: " + ", ".join(prov.get("not_copied") or []))
        lines.append("  why: " + str(prov.get("why")))
        if not check.get("checked"):
            lines.append("  byte-for-byte check: MANIFEST.sha256 missing -- unverified")
        else:
            matched = int(check["checked"]) - len(check.get("mismatched") or [])
            tail = ""
            if not check.get("ok"):
                bad = list(check.get("mismatched") or []) + list(check.get("missing") or [])
                tail = " -- MISMATCH: " + ", ".join(bad)
            lines.append("  byte-for-byte check: %d/%d files match the copy-time digest%s"
                         % (matched, int(check["listed"]), tail))
        lines.append("")
        lines.append("Host probe (read-only, this run):")
        probed = probe_host()
        lines += ["  - " + str(line) for line in probed]
        try:
            indexed = len(PortedSkillSystem().index())
        except Exception as exc:                      # noqa: BLE001
            indexed = 0
            lines.append("  (the vendored skill index is unreadable: %s: %s)"
                         % (type(exc).__name__, exc))
        if indexed:
            lines.append("")
            lines.append("Upstream skill index is %d chars of metadata; bodies are read "
                         "only when chosen." % indexed)
        return {"ok": True, "text": "\n".join(lines), "provenance": prov,
                "vendor": verify_vendor(), "probe": probed, "index_chars": indexed}

    def tars_skills(self, name: str = "") -> Any:
        """The borrowed procedures: the index, or one body in full.

        Two behaviours in one method because they are one question at two
        depths, and progressive disclosure is exactly that split: names and one
        line each by default, the body only when a name is given. Reading the
        whole index every time would put nine procedure descriptions in the
        prompt on every turn that mentions a skill at all.

        Accepts either spelling of a name -- `skill-chrome` or `tars-skill-chrome`
        -- because the index prints the second and the vendored tree uses the
        first, and making the caller remember which is which is a tax paid for
        nothing.
        """
        from .tars_port import PortedSkillSystem

        try:
            system = PortedSkillSystem()
        except Exception as exc:                      # noqa: BLE001
            return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                    "names": []}
        metas = sorted(system.registry.list_meta(), key=lambda m: m.name)
        wanted = str(name or "").strip()
        if not wanted:
            lines = ["%d borrowed procedure(s) from Intelligence Indeed "
                     "(Apache-2.0):" % len(metas)]
            for meta in metas:
                lines.append("  tars-%s  ->  mine as 'tars-%s'" % (meta.name, meta.name))
                lines.append("      " + str(meta.description))
            lines.append("")
            lines.append("tars_skills(name) reads one whole; "
                         "tars_port_skills() copies them into my own library.")
            return {"ok": True, "text": "\n".join(lines),
                    "names": ["tars-" + m.name for m in metas], "count": len(metas)}
        resolved = system.resolve(wanted) or system.resolve(
            wanted[len("tars-"):] if wanted.startswith("tars-") else wanted)
        if resolved is None:
            near = [m.name for m in metas if wanted.lower() in m.name.lower()][:5]
            return {"ok": False, "names": near,
                    "error": "no borrowed procedure named %r%s"
                             % (wanted, ("; closest: " + ", ".join(near)) if near else "")}
        body = system.body(resolved)
        meta = next(m for m in metas if m.name == resolved)
        return {"ok": True, "name": resolved, "chars": len(body),
                "text": "# %s  (borrowed, verbatim; %d chars)\n%s\n\n%s"
                        % (resolved, len(body), meta.description, body)}

    def tars_port_skills(self, overwrite_body: bool = False) -> Any:
        """Copy the borrowed procedures into my own skill library.

        The one tars operation with a side effect, which is why it is a separate
        method and a separate tool: nine files appear under the skills
        directory, each carrying its licence and origin in the file itself, so a
        copy separated from `ORIGIN.txt` still says where it came from.

        `overwrite_body` defaults to False. A re-port that silently replaces a
        procedure this agent has since edited is a data loss dressed as an
        upgrade -- and the underlying function's own default is True, so the
        choice has to be made here.
        """
        from . import skills as _skills

        return _skills.port_tars_skills(overwrite_body=overwrite_body,
                                        store=self.store)

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
            outcome: TaskOutcome | None = None
            try:
                outcome = self._run_task(task.text,
                                         timeout_s=min(remaining, budget_s))
                ok = outcome.ok
                # `why` first: a run that hit the cap has a reason, and that
                # reason is the sentence a person needs. A finished run has no
                # `why`, and its content is what is left to say.
                note = (outcome.why or outcome.content)[:500]
            except Exception as exc:                       # noqa: BLE001
                note = f"{type(exc).__name__}: {exc}"
            try:
                sched.complete(task.id, ok=ok, note=note)
            except Exception as exc:                       # noqa: BLE001
                note = f"{note} (could not record: {type(exc).__name__})"
            if not ok:
                failed += 1
            row: dict[str, Any] = {"id": task.id, "ok": ok, "note": note[:300]}
            if outcome is not None:
                # Kept beside the note rather than folded into it: a reader who
                # wants to know whether the run finished should not have to
                # parse a sentence, and this is the same argument that made
                # `AgentResult.hit_turn_cap` exist one layer down.
                row.update(outcome.to_dict())
                row["ok"] = ok
            results.append(row)

        self._record("schedule_run_due",
                          {"due": len(due), "failed": failed})
        return {"due": len(due), "ran": len(results) - failed,
                "failed": failed, "results": results,
                "elapsed_s": round(_time.monotonic() - started, 1)}

    def _run_task(self, text: str, timeout_s: float = 600.0) -> "TaskOutcome":
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

        Returns a `TaskOutcome`, not the content string, and the change is the
        whole point of this method's current shape. Measured 2026-11-04: this
        returned `getattr(result, "content", "") or ""`, and the caller set
        `ok = True` whenever nothing raised -- so a run that stopped on the turn
        cap was filed as a success with the apology sitting in free text where
        nothing reads it:

            schedule_run_due -> {'due': 1, 'ran': 1, 'failed': 0}
            result row       -> {'ok': True, 'note': '(turn cap of 12 reached
                                 before completion)'}
            task on disk     -> runs 1 / failures 0 / history[0].ok True

        `AgentResult` carries `turns`, `tool_calls`, `self_terminated` and
        `termination_reason` precisely so a caller can tell a finished run from
        a truncated one. Dropping them made four fields decorative. A scheduled
        task that keeps hitting the cap must accumulate `failures` -- that
        counter is the only thing that makes a task too big for one run
        distinguishable from a task that is quietly working.

        The journal is the other half, and there was none. `runs.py` (708 lines,
        a full resume mechanism) had zero writers in this package: all three
        `Agent(...)` constructions passed no `journal`, `Agent.journal` defaults
        to None, and `_beat()` returns at the first line. Measured the same day:
        `%LOCALAPPDATA%\\autoforge\\runs` exists and holds nothing, and the
        ledger has 0 rows for `resume`, `runs_status` and `runs_forget`.
        """
        from . import runs as _runs
        from .core.agent import Agent as _Agent
        from .core.llm import OpenAICompatClient

        model, base, key = _forge_route()
        if not key:
            raise RuntimeError(
                "no credential for the scheduled task's model: DSH's OpenAI "
                "gateway key is missing. Looked at "
                "$DSH_OPENAI_GATEWAY_API_KEY, $AUTOFORGE_API_KEY, and %s. "
                % os.path.join(_dsh_home(), "openai-gateway", "api-key"))
        if model:
            # Same requirement as the forge, and for the same reason: a task
            # that calls tools, cannot read the answers, and still reports
            # success produces nothing while looking finished.
            model, _route_why = _tool_capable_model(model, base, key)
            if _route_why.startswith("no alternative"):
                raise RuntimeError(_tool_read_note("A scheduled task", base))
        if not model:
            raise RuntimeError(
                "no model for the scheduled task: $AUTOFORGE_MODEL is unset and "
                "no DSH profile names one (%s). Set AUTOFORGE_MODEL to a model "
                "id the gateway serves -- %s lists them."
                % (_dsh_model()[1], base.rstrip("/") + "/models"))
        llm = OpenAICompatClient(model=model, base_url=base, api_key=key,
                                 timeout=timeout_s)
        # A journal, but only for work the machine started on its own. That is
        # what this method is: nobody is watching, so a run that dies mid-step
        # otherwise leaves nothing behind but a failure count. The record is
        # removed on the way out -- `finish()` below -- so it is a note to a
        # future process, not a growing pile of files.
        journal = None
        try:
            journal = _runs.Journal(text, cwd=os.getcwd())
        except Exception:                                  # noqa: BLE001
            # A checkpoint that cannot be opened must not be the thing that
            # stops the task it was going to record. `_beat` already swallows
            # write failures for the same reason.
            journal = None
        agent = _Agent(llm, self.registry, max_turns=12,
                       allow_self_terminate=False, journal=journal)
        try:
            result = agent.run(text)
        except BaseException:
            # Left standing on purpose: a run that raised is exactly the one a
            # later process wants to find. Removing it here would throw away the
            # only record that the task was ever entered.
            raise
        else:
            if journal is not None:
                try:
                    journal.finish()
                except Exception:                          # noqa: BLE001
                    pass
        return TaskOutcome.from_result(result)

    def schedule_done(self, task_id: str, note: str = "", ok: bool = True) -> Any:
        """Record a run against a task. An unknown id raises.

        See `schedule_add` for why the refusal is not a field in a successful
        reply. A run that could not be recorded must not look recorded.
        """
        task = self._scheduler().complete(task_id, ok=ok, note=note)
        return {"id": task.id, "text": task.text, "runs": task.runs,
                "failures": task.failures, "cancelled": task.cancelled}

    def schedule_cancel(self, task_id: str) -> Any:
        """Cancel a task. An unknown id raises, for the same reason."""
        task = self._scheduler().cancel(task_id)
        return {"id": task.id, "cancelled": task.cancelled}

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
                    probes: list[dict[str, Any]] | None = None,
                    replace: bool | None = None) -> Any:
        """Persist a tool, then run it once so a broken body fails here.

        A taken name is refused, always. There is no flag here that overwrites:
        the persona's third rule says "要替换就 `evolve_tool`，那是唯一会覆盖
        的路径", and this method used to make that sentence false. It called
        `register(spec, replace=True)` and `save_tool(..., allow_overwrite=
        True)` unconditionally, so redefining a name silently destroyed the
        previous body while the sentence promised that could not happen.
        Measured 2026-11-01: `registry.register(..., replace=True)` had three
        call sites, two of them here, and only `forge_tool` refused a taken
        name.

        The first repair was the wrong shape and is worth recording. It added
        `replace: bool = False` -- a *second* documented overwrite path, which
        made the persona's sentence false in a quieter way, and it meant the
        standard had been rewritten to fit the code instead of the other way
        round. The user caught that: "不是让你改人设，是改代码使得符合人设".
        So `replace` survives only as a tripwire. Passing `True` is refused
        with the route that does work, because a caller who read an old
        docstring deserves a sentence rather than a silent no-op -- and passing
        nothing is the same refusal, because the answer does not depend on it.

        Why `evolve_tool` and not this: a replacement is a *claim about
        behaviour* ("this body is better"), and `evolve_tool` makes that claim
        beat the incumbent on measurements before the old body is retired. A
        bare redefine makes the same claim with no evidence at all, and the
        thing being destroyed is the version history's meaning.

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
        "negative_query": str|None}. `negativeQuery` is accepted too: the DSH
        bridge sends the snake_case spelling, but a caller reading this source
        should not be punished for writing what the schema documents.

        `sample_call` accepts a mapping *or* the JSON text of one. It is
        declared `type: 'json'` at the bridge, and JSON text reaching `dict()`
        is the bug that produced "dictionary update sequence element #0 has
        length 1; 2 is required" -- a message that names neither the field nor
        JSON, on a call whose JSON was valid. Measured 2026-11-01.
        """
        # The body is checked before the name is taken, because the order was
        # the defect: `tool_define(name, '')` registered first, probed second,
        # and answered `registered: true` with `entry 'name' not found or not
        # callable` -- a tool nobody can call sitting in the library on
        # probation, listed by `list_tools`, waiting for the next reader to
        # believe it works. Measured 2026-11-07. `run_python('')` already said
        # "code was empty" in 0.0 ms; this is the same sentence at the other
        # door, in the same key, so a caller does not have to learn two shapes.
        if not str(code or "").strip():
            return {
                "ok": False, "name": name, "registered": False,
                "state": "", "output": "",
                "error": "code was empty: nothing was defined, and no name was taken.",
            }
        params = parameters or {"type": "object", "properties": {}}
        # The refusal comes first, before the probe runs: there is no point
        # spending a sandbox call on a body that will not be kept. It names the
        # route that does work rather than only saying no, because a refusal
        # that hides its own exit is the defect this whole framework exists to
        # catch.
        existing = self.registry.get(name)
        if existing is not None:
            # `ToolSpec` carries no version number of its own -- the version
            # that means anything is the one in `tool_versions`, so it is read
            # from there rather than invented. A refusal that cannot name what
            # would be lost is barely better than a silent overwrite.
            try:
                versions = self.store.versions_of(name)
                latest = versions[-1]["version"] if versions else "?"
                bodies = len(versions)
            except Exception:                                  # noqa: BLE001
                latest, bodies = "?", "?"
            # The asked-for-but-impossible override gets its own sentence. A
            # caller who passed `replace=True` is not confused about what they
            # want; they are working from a promise this method no longer keeps,
            # and telling them "already exists" would leave them guessing
            # whether the flag was ignored or misspelled.
            if replace:
                return {
                    "ok": False, "name": name, "registered": False,
                    "replace_refused": True,
                    "state": getattr(getattr(existing, "state", ""), "value", ""),
                    "error": (
                        f"{name!r} is already in this library, and `replace` is "
                        f"not a path this method has: no value of it changes "
                        f"the answer. The one overwrite path is "
                        f"autoforge_evolve_tool -- it breeds a new body, verifies "
                        f"it against the incumbent's measurements, and only then "
                        f"retires the old one, which is why it may overwrite "
                        f"where a bare redefine may not. To keep the existing "
                        f"body, use autoforge_describe_tool to see it, or define "
                        f"under a different name."),
                }
            return {
                "ok": False, "name": name, "registered": False,
                "state": getattr(getattr(existing, "state", ""), "value", ""),
                "error": (
                    f"{name!r} is already in this library (state "
                    f"{getattr(getattr(existing, 'state', ''), 'value', '?')}, "
                    f"version {latest} of {bodies} recorded). Nothing was "
                    f"changed. The one overwrite path is autoforge_evolve_tool "
                    f"-- it breeds a new body and verifies it against the "
                    f"incumbent before retiring the old one. Use "
                    f"autoforge_describe_tool to see what is there now, or "
                    f"define under a different name."),
            }
        # `_runner_for` needs the symbol to call, and a defined body normally
        # names its function after the tool (`def is_perfect_square(n)`), not
        # `main`. Resolving it here is the same fix the load path needed; the
        # probe below must use the same entry or it tests a symbol that will
        # never be called. Before this, every `tool_define` call raised
        # "Service._runner_for() missing 1 required positional argument: 'entry'".
        entry = self._entry_of(code, name)
        fn = self._runner_for(code, entry)
        # The scope is derived, never left empty. An empty signature reads as
        # `undeclared`, which `autonomy/confirm.py` turns into all eight
        # capabilities -- so an unlabelled tool is gated under any policy that
        # switches off a single freedom, and the question it gets asked is
        # about powers it may not have. Measured 2026-11-03: all six tools in
        # the live library carried `effect_signature=''`.
        #
        # Derived rather than asked: the model's own answer would be a claim
        # about the code, and this package's whole argument is that a claim
        # about code is worth less than a reading of it. `derived_scope` reads
        # `forge/validity.audit_effects` -- the same tripwire `_scope_gate`
        # uses on the other side -- so declaration and audit cannot disagree
        # by construction.
        spec = ToolSpec(name=name, description=description or name,
                        parameters=params, code=code, fn=fn, runner=fn,
                        effect_signature=derived_scope(code))
        # The record-inheritance block that used to sit here is gone, and its
        # absence is the point: it existed so a `replace=True` overwrite would
        # not wipe the call history that automatic quarantine is decided on.
        # With no overwrite path here, a name reaching this line is a name that
        # has no record to inherit -- `existing is None` by construction. The
        # code that remains is the tripwire two lines below.
        #
        # The declaration travels with the tool so `evaluate_tool` can judge it
        # later, in another session, from the row and not from this call.
        #
        # `sample_call` arrives from more than one caller and not all of them
        # coerce it: the DSH bridge declares it `type: 'json'` and accepted the
        # JSON *text* of an object. `dict()` on that string is not a parse -- it
        # is a conversion that walks the string's characters, so the first one
        # `{` yields
        #
        #   ValueError: dictionary update sequence element #0 has length 1;
        #               2 is required
        #
        # which mentions neither the field nor JSON. Measured 2026-11-01: a
        # correctly written sample call was rejected three times on that
        # message, and the only way through was to drop sample_call -- turning
        # a declaration bug into an UNPROBED execution check, which is a worse
        # state for the tool. Parse here too, so the service is correct whatever
        # the caller does: the schema says JSON, so JSON is accepted.
        if isinstance(sample_call, str):
            text = sample_call.strip()
            if not text:
                sample_call = None
            else:
                try:
                    sample_call = json.loads(text)
                except ValueError as exc:
                    raise ValueError(
                        f"sample_call must be a JSON object or a mapping, got "
                        f"{type(sample_call).__name__} that is not valid JSON "
                        f"({exc}); value={text[:200]!r}") from exc
        if sample_call is not None and not isinstance(sample_call, dict):
            raise ValueError(
                f"sample_call must be a JSON object or a mapping, got "
                f"{type(sample_call).__name__}: {sample_call!r}")
        spec.sample_call = dict(sample_call) if sample_call else None
        spec.sample_expect = sample_expect or ""
        for p in (probes or []):
            if not isinstance(p, dict) or not str(p.get("query") or "").strip():
                continue
            neg = p.get("negative_query") or p.get("negativeQuery")
            spec.probes.append(TriggerProbe(
                query=str(p["query"]),
                expect=str(p.get("expect") or "call"),
                negative_query=str(neg) if neg else None))
        # `probation`, not `active`. A hand-written body has been run exactly
        # once, on the caller's sample call, and that is the standard the module
        # docstring above calls "not enough". `forge_tool` earns ACTIVE by
        # surviving six checks; allowing `tool_define` to start there made the
        # two entry points disagree about what the same state word means.
        #
        # Measured 2026-10-31: a body that never opened a file --
        # `return {"bytes": 123456}` -- was registered by this call and
        # `list_tools` reported `[active]`. Probation still lets it be called
        # (that is deliberate: a tool must be usable to be evaluated); it says
        # it has not been examined yet, and the promotion path is
        # `evaluate_tool`, which is the same battery `forge_tool` runs.
        #
        # That last sentence was false until 2026-11-01: `evaluate_tool` ran the
        # battery and printed the verdict without ever moving the state, so
        # every `tool_define` tool was sealed in PROBATION permanently -- see
        # `evaluate_tool` for the fix and the measurement that found it.
        spec.state = ToolState.PROBATION
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
                                   "state": spec.state.value,
                                   "effect_signature": spec.effect_signature,
                                   "ok": getattr(probe, "ok", False),
                                   "output": out[:4000],
                                   "error": err}
        # `replace=False` and `allow_overwrite=False`, explicitly, and this is
        # the second half of keeping the persona's sentence true. The name came
        # back free a few lines up, which closes the ordinary case; these two
        # close the race. Between that check and this write, another session can
        # register the same name, and the previous values here -- `replace=True`,
        # `allow_overwrite=True` -- meant the loser's body vanished with no
        # error at all. A test that defines one tool cannot see that window; two
        # sessions can, and did: see `docs/DUPLICATE_NEEDS.md`.
        #
        # A registry refusal is reported as a refusal instead of escaping as an
        # exception, because the payload above already claims `registered: True`
        # and a caller deserves to be told which of the two is the truth.
        try:
            self.registry.register(spec, replace=False)
        except ValueError as exc:
            payload["registered"] = False
            payload["error"] = (
                f"refused at write time: {exc}. Another session took this name "
                f"between the check and the write; nothing of theirs was "
                f"changed. Use autoforge_evolve_tool, or define under a "
                f"different name.")
            self._record("tool_defined", {"tool": name, "persisted": False,
                                          "state": str(spec.state.value),
                                          "refused": "race"})
            return payload
        try:
            self.store.save_tool(spec)
            stored = True
        except Exception as exc:                               # noqa: BLE001
            # A tool that lives only in this process is a tool that disappears
            # with it, and the caller has to be told. This used to be reported
            # as `ok: True` with the failure buried in a free-text note at the
            # end of the sentence, which is the same defect the whole framework
            # exists to catch, occurring one level up: the reply described
            # this call rather than the library.
            payload["save_error"] = f"{type(exc).__name__}: {exc}"
            stored = False
        self._record("tool_defined", {"tool": name, "persisted": stored,
                                      "state": str(spec.state.value)})
        return payload

    def tool_list(self) -> list[Any]:
        out: list[Any] = []
        for n in self.registry.names():
            spec = self.registry.get(n)
            out.append({"name": n,
                        "description": getattr(spec, "description", ""),
                        "state": str(getattr(getattr(spec, "state", ""), "value", ""))})
        return out

    def describe_tool(self, name: str) -> dict[str, Any]:
        """Everything about one tool: its schema, its record, and its lineage.

        This method did not exist, and the persona told the model to call it --
        "`describe_tool(name)` 看一个工具的 schema 和战绩". The named neighbour,
        `describe`, answers a different question (which service methods exist,
        and which database this process is bound to), so an agent that followed
        the persona either got the wrong object or nothing at all. Measured
        2026-11-01: `describe_tool` appeared zero times in the bridge, zero
        times as a service method, and once inside a frozenset of hot tool
        names in `tools/exposure.py` -- a name kept alive by a list that nothing
        on the DSH path reads.

        Written as an answer to the sentence that promised it, so it returns
        the three things that sentence names and no prose: what the tool
        declares (`schema`), how it has behaved (`record`), and where it came
        from (`versions`, oldest first, because "v4" means nothing without the
        v1 it replaced).

        **Two namespaces, not one.** There are 66 tools the model can call and
        only a handful of them are *forged* -- the rest are the bridge's own
        `autoforge_*` wrappers, which never enter `self.registry`. Measured
        2026-11-02: `describe_tool('autoforge_run_python')` answered "no tool
        named ... the library holds: fake_file_size, file_facts, ...". That was
        true and useless, because `manifest_header` tells the model to call
        this method "for its full parameter schema first" and the persona says
        the same -- so the one path the cold-tool design depends on did not
        exist for 60 of the 66 names. A bridge tool is answered from the
        catalog the bridge reports (`store.bridge_tools`), and the answer says
        which namespace it came from, because "this is a wrapper around a
        service method" and "this is a tool somebody forged" are different
        facts about a tool.
        """
        spec = self.registry.get(name)
        if spec is None:
            return self._describe_bridge_tool(name)
        record = spec.to_dict()
        # The schema the model would see, wrapped the way a function schema is
        # wrapped, so the caller can compare it with what it thinks it has.
        record["schema"] = spec.schema
        # `stats` lives in memory and is only as fresh as the last save; the
        # ledger is the authority on whether a tool was ever used. Both are
        # returned rather than reconciled here -- a disagreement between them
        # is information, and picking a winner would hide it.
        try:
            record["ledger_calls"] = self.store.calls_of(name)
        except Exception as exc:                               # noqa: BLE001
            record["ledger_calls_error"] = f"{type(exc).__name__}: {exc}"
        # When `stats.calls` is ahead of `ledger_calls`, one of two things is
        # true: the tool was invoked through a path that never logged a `call`
        # row, or the row was refused by the store. The second is only visible
        # from here, and only if this process is the one that lost the rows --
        # so it is reported alongside the pair rather than left to be guessed.
        # Measured 2026-11-01: `path_facts` showed `calls: 1` against
        # `ledger_calls: {calls: 0}` and there was no way to tell which of the
        # two it was without reasoning about which code path increments stats.
        record["ledger_locked_drops"] = self._ledger_locked_drops
        # Lifetime calls, not trial calls: `rehab` zeroes the counters and
        # parks the old total in `calls_not_in_counters`, so comparing the
        # ledger against `stats.calls` alone would make every rehabilitated
        # tool report a clean record it never had.
        lifetime_calls = (getattr(spec.stats, "calls", 0)
                          + getattr(spec.stats, "calls_not_in_counters", 0))
        if record["ledger_calls"].get("calls", 0) < lifetime_calls:
            record["records_disagree"] = (
                "stats.calls is ahead of the ledger. Either this tool ran on a "
                "path that does not log a `call` row, or rows were lost: "
                f"ledger_locked_drops={self._ledger_locked_drops} for this "
                "process. A non-zero count means the ledger is missing rows it "
                "tried to write; zero means nothing was lost here, and the "
                "difference is a path that does not log.")
        # The tally and the list must account for each other exactly:
        #
        #     calls + calls_not_in_counters - history_dropped == len(history)
        #
        # Every call either added an entry or was dropped from the window by a
        # trim, and `rehab` moves its total from the first term to the second.
        # Written as an identity rather than a comment because it is the one
        # check that catches a counter zeroed without a record of it. Measured
        # 2026-11-03: `fake_file_size` held `calls: 0` beside six entries and
        # `records_disagree` above said nothing -- the ledger was *ahead* of the
        # stats, and that comparison only ever looked one way.
        hist_len = len(getattr(spec.stats, "history", []) or [])
        unexplained = lifetime_calls - getattr(spec.stats, "history_dropped", 0)
        if unexplained != hist_len:
            record["counters_unreconciled"] = (
                f"calls({getattr(spec.stats, 'calls', 0)}) + "
                f"calls_not_in_counters({getattr(spec.stats, 'calls_not_in_counters', 0)})"
                f" - history_dropped({getattr(spec.stats, 'history_dropped', 0)})"
                f" = {unexplained}, but history holds {hist_len}. Either the "
                "counters were zeroed without recording the old total, or "
                "entries left the list without being counted. The history is "
                "the harder thing to fake: treat it as the record and the "
                "tally as the claim.")
        try:
            record["versions"] = self.store.versions_of(name)
        except Exception as exc:                               # noqa: BLE001
            record["versions_error"] = f"{type(exc).__name__}: {exc}"
        record["kind"] = "forged"
        record["ok"] = True
        return record

    def _describe_bridge_tool(self, name: str) -> dict[str, Any]:
        """The bridge's own tool, answered from the catalog it reported.

        Kept separate from the forged branch because almost nothing carries
        over: a bridge tool has no version history, no probe battery and no
        sandbox -- its whole behaviour is one `client.call(method, params)`
        into this process. `describe_tool` used to answer for forged tools only
        and return an error for everything else, which made the manifest's
        "call describe_tool for the full schema" a dead end for 60 of 66 names.
        """
        entry = None
        try:
            entry = self.store.bridge_tool(name)
        except Exception:                                      # noqa: BLE001
            entry = None
        if entry is None:
            catalog = {"tools": [], "reported_at": 0.0}
            try:
                catalog = self.store.bridge_catalog()
            except Exception:                                  # noqa: BLE001
                pass
            bridged = sorted(t["name"] for t in catalog.get("tools", []))
            forged = sorted(self.registry.names())
            if not bridged and not forged:
                return {"ok": False, "name": name,
                        "error": f"no tool named {name!r}, and no bridge catalog "
                                 "has been reported to this service yet"}
            parts = []
            if bridged:
                parts.append(f"bridge ({len(bridged)}): {', '.join(bridged)}")
            else:
                parts.append("bridge: no catalog has been reported yet")
            if forged:
                parts.append(f"forged ({len(forged)}): {', '.join(forged)}")
            else:
                parts.append("forged: none")
            return {"ok": False, "name": name,
                    "error": f"no tool named {name!r}; " + "; ".join(parts)}
        # The parameters arrive as the bridge's own descriptor map (the shape
        # `defineTool` takes), which is what the model's view is built from.
        # Compiling it here would produce a JSON-Schema projection that is not
        # what DSH actually sends, so the descriptor is returned as-is and
        # labelled: a reader that wants the wire schema must compile it the way
        # the bridge does, and pretending otherwise would be a third copy.
        try:
            calls = self.store.calls_of(name)
        except Exception as exc:                               # noqa: BLE001
            calls = {"error": f"{type(exc).__name__}: {exc}"}
        age = None
        try:
            age = max(0.0, (time.time() - float(entry["reported_at"])) / 86400.0)
        except Exception:                                      # noqa: BLE001
            age = None
        return {
            "ok": True,
            "kind": "bridge",
            "name": entry["name"],
            "description": entry["description"],
            "parameters": entry["parameters"],
            "parameters_shape": "descriptor: {name: {type, required?, description?}} "
                                "-- the shape defineTool takes, not JSON Schema",
            "schema": {"type": "function",
                       "function": {"name": entry["name"],
                                    "description": entry["description"],
                                    "parameters": entry["parameters"]}},
            "ledger_calls": calls,
            "catalog_reported_at": entry["reported_at"],
            "catalog_age_days": age,
            "versions": [],
            "note": "This is one of the bridge's own autoforge_* tools: it is a "
                    "wrapper around a service method, not a tool anybody forged, "
                    "so it has no version history and no probe battery. Its "
                    "record is whatever the ledger saw.",
        }

    def _last_called_ts(self, name: str) -> float | None:
        """When this tool was last called, from the ledger, or None.

        One ledger scan per exposure refresh, not one per tool: the registry asks
        this question once per tool per turn (66 times for the bridge alone) and
        `last_called_by_tool()` reads every `call` row in the file to answer it.

        The naming gap is real and is handled by asking twice: the ledger stores
        the name the SERVICE was called with (`tool_call` receives the short
        forged name; the bridge forwards `run_python`), while the exposure split
        sees the name the MODEL sees (`autoforge_run_python`). Both spellings are
        checked, bare name first, because a forged tool called as
        `autoforge_foo` by mistake should not shadow the honest `foo` row.
        """
        now = time.time()
        at, table = self._last_called_cache
        if now - at > 30.0 or not table:
            try:
                table = self.store.last_called_by_tool()
            except Exception:                                  # noqa: BLE001
                table = {}
            self._last_called_cache = (now, table)
        ts = table.get(name)
        if ts is None and name.startswith("autoforge_"):
            ts = table.get(name[len("autoforge_"):])
        return ts

    def _days_since_called(self, name: str) -> float | None:
        """How long ago this tool was last called, in DAYS -- what classify wants.

        `_last_called_ts` answers in epoch seconds, and passing it where a
        duration is expected is not a type error anywhere: `classify` compares
        `age <= window_days`, so 1793631302 compared against 7.0 is simply
        False. Measured 2026-11-02: a tool called one second ago classified as
        cold, because "1,793,631,302 days ago" is not less than seven days.
        The two units get one named conversion and it lives here.
        """
        ts = self._last_called_ts(name)
        if ts is None:
            return None
        return max(0.0, (time.time() - float(ts)) / 86400.0)

    def _build_exposure_view(self, *, window_days: float = 7.0,
                             protect: list[str] | None = None,
                             force: bool = False) -> dict[str, Any]:
        """Decide which tools ride in full and which ride as a line.

        The bridge's 66 tools and the registry's forged tools are two different
        populations with two different owners: the schemas of the first live in
        `lib/index.js` and arrive through `report_bridge_tools`, the second are
        rows in `tools`. They are classified by the same rule (`exposure.classify`)
        because two implementations of "which tools are hot" would be one
        implementation and one opinion, and the one that drifted would be the
        invisible one.

        Cached for `_EXPOSURE_TTL_S`: the answer changes when a call happens or
        when an hour turns over, and both are slower than a turn. Recomputing per
        request would be a query per tool per turn to save characters, and it
        would also break the provider's prefix cache -- an identical prompt
        between two turns with no calls in between is worth real money.
        """
        from .tools import exposure as exposure_mod

        now = time.time()
        wanted = [str(n) for n in (protect or [])]
        #: The cache key includes what was asked, not just when.
        #:
        #: It did not at first, and the miss was invisible: `tool_exposure` was
        #: cached for 60s on a view built with an empty `protect`, so the first
        #: turn after a task began kept the previous turn's answer. A projection
        #: that ignores the argument it was given is not a cache of this call,
        #: it is a cache of *a* call, and the difference only shows up on the
        #: turn where the argument changed -- which is every turn that starts a
        #: new task. Order-insensitive: `protect` is a set of names, and two
        #: callers listing the same names differently ask the same question.
        key = tuple(sorted(set(wanted)))
        with self._exposure_lock:
            if (not force
                    and self._exposure_view.get("hot") is not None
                    and self._exposure_view.get("protect_key") == key
                    and now - float(self._exposure_view.get("built_at") or 0.0)
                    < _EXPOSURE_TTL_S):
                return dict(self._exposure_view)
            if force:
                # Two caches sit between a fresh call row and this answer, and
                # `force` used to clear only one of them. Measured 2026-11-02:
                # after writing a `call` row for `autoforge_gpu_bench`,
                # `tool_exposure(refresh=True)` still returned
                # `hot == ['autoforge_run_python']`, because
                # `_last_called_ts` holds `last_called_by_tool()` for 30s.
                # A refresh that answers from a cache is not a refresh; it is
                # the same answer with a different word on it.
                self._last_called_cache = (0.0, {})

        wanted = [str(n) for n in (protect or [])]
        floor = set(exposure_mod.BRIDGE_FLOOR) | set(wanted)
        hot: set[str] = set()
        cold: dict[str, str] = {}
        stats: dict[str, int] = {"hot": 0, "cold": 0, "total": 0}
        problems: list[str] = []

        def _cold_text(name: str, schema: dict[str, Any]) -> str:
            return exposure_mod.cold_description(
                name, schema, window_days,
                desc_override=exposure_mod._schema_desc(schema),
            )

        def _full_cost(name: str, schema: dict[str, Any]) -> int:
            """What this tool costs now: its name, its description, its schema.

            The schema is the COMPILED JSON Schema the provider is sent, when the
            catalog carries one, and only otherwise falls back to the descriptor.
            This is the number the bill is computed from, so it is the number the
            comparison has to use.
            """
            fn = schema.get("function") if isinstance(schema.get("function"), dict) else schema
            cost = len(str(fn.get("name") or "")) + len(str(fn.get("description") or ""))
            compiled = fn.get("schema") if isinstance(fn, dict) else None
            if isinstance(compiled, dict) and compiled:
                try:
                    return cost + len(json.dumps(compiled, separators=(",", ":"), sort_keys=True))
                except Exception:                              # noqa: BLE001
                    pass
            params = fn.get("parameters") if isinstance(fn, dict) else None
            try:
                body = json.dumps(params or {}, separators=(",", ":"), sort_keys=True)
            except Exception:                                  # noqa: BLE001
                body = ""
            return cost + len(body)

        try:
            catalog = self.store.bridge_catalog()
        except Exception as exc:                               # noqa: BLE001
            catalog = {"tools": []}
            problems.append(f"bridge catalog unreadable: {type(exc).__name__}: {exc}")

        bridge_schemas = []
        for row in catalog.get("tools") or []:
            # Two shapes arrive here and both now matter: `parameters` is the
            # descriptor the bridge declared (which is what `defineTool` takes),
            # and `schema` is the JSON Schema it compiled to (which is what the
            # provider actually receives). The cost comparison below must use
            # the second, or it measures 132 characters of descriptors and calls
            # it the price of 11,823 characters of schemas.
            raw_params = row.get("parameters")
            if isinstance(raw_params, str):
                try:
                    params = json.loads(raw_params or "{}")
                except Exception:                              # noqa: BLE001
                    params = {}
            elif isinstance(raw_params, dict):
                params = raw_params
            else:
                params = {}
            raw_schema = row.get("schema")
            if isinstance(raw_schema, str):
                try:
                    raw_schema = json.loads(raw_schema or "{}")
                except Exception:                              # noqa: BLE001
                    raw_schema = {}
            if not isinstance(raw_schema, dict) or not raw_schema:
                raw_schema = {}
            bridge_schemas.append({
                "type": "function",
                "function": {
                    "name": str(row.get("name") or ""),
                    "description": str(row.get("description") or ""),
                    "parameters": params if isinstance(params, dict) else {},
                    # The compiled form, for costing. Not sent anywhere as-is:
                    # `parameters` is what names the arguments in the cold line.
                    "schema": raw_schema,
                },
            })
        if bridge_schemas:
            b_hot, _b_lines, b_stats = exposure_mod.classify(
                bridge_schemas, self._days_since_called,
                window_days=window_days, floor=floor, protect=(),
                cold_text=_cold_text, full_cost=_full_cost,
            )
            hot |= b_hot
            for schema in bridge_schemas:
                name = exposure_mod._schema_name(schema)
                if name in b_hot:
                    continue
                cold[name] = exposure_mod.cold_description(
                    name, schema, window_days,
                    desc_override=exposure_mod._schema_desc(schema),
                )

        # Forged tools go through the registry's own exposure, which knows about
        # states: a retired tool must not be advertised at all, and the registry
        # already filters by `visible_states`.
        try:
            forged_schemas = self.registry.schemas()
        except Exception:                                      # noqa: BLE001
            forged_schemas = []
        forged_hot, _forged_lines, _f_stats = exposure_mod.classify(
            forged_schemas, self._days_since_called,
            window_days=window_days,
            # Forged tools are named without the prefix, so the floor that applies
            # to them is the short-name one. Using BRIDGE_FLOOR here would protect
            # nothing and demote every forged tool on its first quiet week.
            floor=set(exposure_mod.HOT_FLOOR) | set(wanted),
            protect=set(),
            cold_text=_cold_text, full_cost=_full_cost,
        )
        hot |= forged_hot
        for schema in forged_schemas:
            name = exposure_mod._schema_name(schema)
            if name in forged_hot:
                continue
            cold[name] = exposure_mod.cold_description(
                name, schema, window_days,
                desc_override=exposure_mod._schema_desc(schema),
            )

        # One tally over both populations, because that is the only tally that
        # can be checked against `hot` and `cold` in the same answer. Classifying
        # the bridge alone and then reporting the union of the two produced
        # "3 hot, 0 cold of 0 bridge tools" on 2026-11-02: three hot tools, a
        # bridge count of zero, and nothing in the sentence a reader could
        # reconcile with the other field in the same reply. A count that
        # describes a different set than the list beneath it is worse than no
        # count, because it is the one a reader trusts first.
        bridge_total = int((b_stats or {}).get("total") or 0) if bridge_schemas else 0
        forged_total = int((_f_stats or {}).get("total") or 0) if forged_schemas else 0
        hot_names = sorted(hot)
        stats = {
            "hot": len(hot_names),
            "cold": len(cold),
            "total": len(hot_names) + len(cold),
            "bridge_total": bridge_total,
            "forged_total": forged_total,
        }

        view = {
            "hot": hot_names,
            "cold": cold,
            "stats": stats,
            "window_days": window_days,
            "built_at": now,
            # What was asked, so the next caller asking something else does not
            # get this answer. Without it the TTL turned every `protect` list
            # after the first into a no-op for a minute.
            "protect_key": key,
            "source": "ledger" if catalog.get("tools") else "registry-only",
            "problems": problems,
        }
        with self._exposure_lock:
            self._exposure_view = dict(view)
        return view

    def tool_exposure(self, window_days: float = 7.0,
                      protect: list[str] | None = None,
                      task: str = "",
                      refresh: bool = False) -> dict[str, Any]:
        """Which tool schemas ride in full, and the one-liners standing in.

        Called by the bridge, which then decides per request what each of its 66
        tools says: the full schema when hot, a name-plus-arguments description
        and an empty parameter map when cold. `refresh` bypasses the TTL -- used
        by the tests, and by nothing on the model's path.

        `task` is the text of the turn being assembled. Anything it names stays
        hot regardless of the ledger, because the ledger only knows what was
        called -- and a tool a task names for the first time is exactly the tool
        whose arguments must be readable on that first step. The names come from
        `exposure.protect_from_task`, whose boundary rule was lifted from
        upstream `agent.py::_hot_protect` (whose docstring records why a
        substring match is not good enough). The text is passed rather than a
        name list on purpose: the caller is a JavaScript bridge with no opinion
        about tool names, and deriving them there would put a second
        implementation of the rule in the language that has no test for it.

        Protection is an argument to one decision, not a setting on the
        registry. An earlier design assigned `registry.protect` and nothing ever
        wrote it -- a feature that was accepted, documented, tested in isolation,
        and empty for as long as it existed.
        """
        names = list(protect or [])
        if task:
            from .tools import exposure as exposure_mod

            try:
                known = list(self.registry.names())
                known += [str(t.get("name") or "")
                          for t in (self.store.bridge_catalog().get("tools") or [])]
                names.extend(sorted(exposure_mod.protect_from_task(
                    str(task), [n for n in known if n])))
            except Exception as exc:                           # noqa: BLE001
                # A prompt block that cannot be computed is not a reason to
                # withhold the whole tool table: the fallback is the ledger's
                # answer, which is what this projection had before `task`
                # existed.
                self._record("exposure_task_failed",
                             {"error": f"{type(exc).__name__}: {exc}"})
        view = self._build_exposure_view(
            window_days=float(window_days or 7.0),
            protect=names,
            force=bool(refresh),
        )
        cold = dict(view.get("cold") or {})
        return {
            "ok": True,
            "window_days": view.get("window_days", 7.0),
            "hot": list(view.get("hot") or []),
            "cold": cold,
            "stats": dict(view.get("stats") or {}),
            "source": view.get("source", "none"),
            "problems": list(view.get("problems") or []),
            "text": (
                f"{len(view.get('hot') or [])} hot, {len(cold)} cold "
                f"of {int((view.get('stats') or {}).get('total') or 0)} tools "
                f"({int((view.get('stats') or {}).get('bridge_total') or 0)} bridge, "
                f"{int((view.get('stats') or {}).get('forged_total') or 0)} forged); "
                f"source {view.get('source', 'none')}"
            ),
        }

    def report_bridge_tools(self, catalog: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Accept the bridge's tool catalog, so `describe_tool` can answer for it.

        The bridge owns these schemas -- they are JavaScript object literals in
        `lib/index.js` -- and this process is the only one that can be asked a
        question by name. Without this handover, `describe_tool` could only
        answer for forged tools, and the manifest's advice to call it "for its
        full parameter schema first" pointed at an error for 60 of the 66 names
        the model can call.

        Reported on every bridge mount and on every reconnect, wholesale rather
        than incrementally, because "what the bridge registers right now" is the
        question -- and a name that was deleted from `lib/index.js` must stop
        being answered here. `store.save_bridge_tools` refuses an empty list for
        that reason: an empty catalog is a failed read, not a library that
        became empty.
        """
        written = self.store.save_bridge_tools(catalog or [])
        if not written.get("ok"):
            return written
        self._record("bridge_catalog", {"tools": written.get("written", 0)})
        return written

    def review_tool(self, name: str) -> dict[str, Any]:
        """Ask a reviewer what a tool's own claim says about its own code.

        This is not `evaluate_tool` with fewer checks. The battery asks whether
        the tool works when called, and it can only answer that after running
        the thing; this asks whether the sentence the tool writes about itself
        matches the code sitting beside it, and it answers from the spec alone.
        Two questions that fail independently -- a tool that returns the right
        number while claiming to touch nothing is a verified liar, and the
        battery has no vocabulary for that.

        The reviewer is deliberately not allowed to promote, demote, quarantine
        or retire. A reviewer that also seals its own findings has become the
        thing being reviewed; `evaluate_tool` owns the state transitions and
        `rehab_tool` owns the way back.

        Measured before this existed (2026-11-04): a tool declaring `pure` that
        opens a file for writing passed `ValidityGate` on a first approval,
        because the gate's scope check is guarded by
        `require_scope_declaration`, which defaults to False, and its strong
        branch needs a `FrozenBaseline` that a brand-new tool does not have.

        What it cannot see, stated so nobody reads `ok` as more than it is:
        the reviewer never runs the tool, so a tool that returns a wrong answer
        or a fabricated number is outside its reach -- `path_liar` reviews clean
        here and is still a liar, because its lie is in the value it returns,
        not in the sentence it writes about itself. That question belongs to
        `evaluate_tool`, which calls the thing; the two answers are meant to be
        read together and neither one alone decides a tool.
        """
        spec = self.registry.get(name)
        if spec is None:
            known = ", ".join(sorted(self.registry.names())) or "none"
            return {"ok": False, "tool": name,
                    "error": f"no tool named {name!r}; the library holds: {known}"}
        verdict = review_tool(spec)
        out = verdict.to_dict()
        out["text"] = verdict.summary()
        try:
            self._record("review", {
                "tool": name, "ok": bool(verdict.ok),
                "scope": verdict.scope,
                "objected": [f.name for f in verdict.objectors],
                "reviewed": bool(verdict.reviewed),
            })
        except Exception:                                      # noqa: BLE001
            # The review already happened; failing to file the receipt must not
            # turn a completed answer into an error the caller has to interpret.
            pass
        return out

    def rehab_tool(self, name: str, require_clean: int = 2) -> dict[str, Any]:
        """Put a quarantined (or retired) tool back on trial.

        The registry has had `rehab` since it had a state machine, and the
        persona promised it by name -- "`rehab` 能把它放回 `probation`" -- while
        nothing in the package called it and the bridge did not declare it. A
        promise with no caller is a promise the model cannot keep.
        (The persona now spells it `rehab_tool`, the name this tool is
        registered under; `chk_seams.py` checks every backticked name against
        the bridge, the wire table and the state enum, so prose can no longer
        name something in between.)

        Rehab is deliberately weaker than `promote`: it clears the counters and
        returns the tool to PROBATION, where it is callable and visible but
        unsealed. Clearing the record is the point and also the cost -- the
        history of why it was quarantined survives in the ledger, which is the
        only place it should survive.
        """
        spec = self.registry.get(name)
        if spec is None:
            known = ", ".join(sorted(self.registry.names())) or "none"
            return {"ok": False, "name": name,
                    "error": f"no tool named {name!r}; the library holds: {known}"}
        before = getattr(getattr(spec, "state", ""), "value", str(getattr(spec, "state", "")))
        try:
            n = max(0, int(require_clean))
        except (TypeError, ValueError):
            n = 2
        try:
            self.registry.rehab(name, require_clean=n)
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "name": name, "state": before,
                    "error": f"could not rehabilitate: {type(exc).__name__}: {exc}"}
        after = getattr(getattr(spec, "state", ""), "value", str(getattr(spec, "state", "")))
        try:
            self._record("state_change", {
                "tool": name, "from": before, "to": after, "move": "rehab",
                "why": f"manual rehabilitation; counters cleared, "
                       f"{n} clean calls wanted before it can be sealed again",
                "passed": None,
            })
        except Exception:                                      # noqa: BLE001
            pass
        return {"ok": True, "name": name, "from": before, "to": after,
                "require_clean": n,
                "text": f"{name}: {before} -> {after} (on trial; {n} clean calls "
                        f"before it can earn the seal again)"}

    def tool_call(self, name: str, arguments: dict[str, Any] | None = None,
                  force: bool = False) -> dict[str, Any]:
        """Run a forged tool. `force` overrides a quarantine or a retirement.

        The registry has honoured `force` since it had a state machine, and the
        refusal it returns says "pass force=True to run anyway" -- while this
        method had no such parameter and the DSH bridge declared no such field.
        So the message instructed the model to use something no caller could
        reach: measured 2026-11-01, three calls in a row to a retired tool, each
        returning that sentence, each with every spelling of `force` in
        `arguments` and at the top level ignored.

        A refusal that names a way out has to be a way out. Either drop the
        sentence or honour the parameter; honouring it is the smaller change
        and the more useful one -- a tool quarantined by a bad afternoon is
        exactly the case where an operator wants one deliberate run.
        """
        res = self.registry.call(name, arguments or {}, force=bool(force))
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
        #
        # `self._record`, not `self.store.log_event`. This line used to call the
        # store directly, and that one word made an audit write load-bearing for
        # the answer: `chaining.append_event` raises after its retries are spent,
        # so a locked database turned a call that HAD ALREADY RUN into an
        # exception at the caller -- no reply, and the tool's counter still
        # moved (the registry records before returning). Measured 2026-11-01:
        # `path_facts` held `stats.calls=1` at 15:12:58 with zero `call` rows in
        # the ledger, and `report()["ledger_locked_drops"]` stayed at 0 the whole
        # time -- because the direct call bypassed the counter that exists to
        # notice exactly this. Two records disagreed and neither one knew why.
        #
        # `_record` counts the loss and never raises, which is the right trade
        # for an event describing something that already happened: the call is
        # the work, the row is the note about the work. A note that cannot be
        # written must not retroactively un-run the work -- but it must be
        # counted, so a ledger that stops growing has a number pointing at why.
        #
        # The row used to carry `{"tool", "ok"}` and nothing else. Measured
        # 2026-11-02: all 15 `call` rows in the live database had exactly those
        # two keys, so the ledger could say a tool failed and never which
        # arguments failed, how long it took, or what it said -- the three
        # things needed to reproduce it. `duration_ms` and `error` are already
        # on the ToolResult the registry returns, so the note was dropping
        # facts that were sitting in its hand.
        event: dict[str, Any] = {"tool": name, "ok": payload["ok"]}
        digest = self._arg_digest(arguments)
        if digest:
            event["args"] = digest
        ms = float(getattr(res, "duration_ms", 0.0) or 0.0)
        if ms > 0:
            event["ms"] = round(ms, 3)
        if err:
            event["error"] = str(err)[:600]
        if getattr(res, "quarantined", False):
            event["quarantined"] = True
        if getattr(res, "aborted", False):
            event["aborted"] = True
        self._record("call", event)
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
            f"  Own source: {reach['source']}",
            # The persona says this block reports "射程/记忆/自身源码位置". It
            # reported the first two for two rounds; the third was a promise with
            # nothing behind it, and a promise is exactly what this whole file is
            # about. The path plus its mtime answers the question that matters in
            # practice -- "is the code I am reading the code that is running" --
            # against the process start time.
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
                # An offence with no line is a file that was never read, and
                # "line 0: " would claim the opposite -- that line zero has
                # something on it. Print the reason instead.
                out += [(f"  line {o.line_no}: {o.line}" if o.line
                         else f"  {o.reason}") for o in offs[:10]]
            return "\n".join(out)
        if not source.strip():
            return ("Nothing to audit: pass `source` with the text, or `path` "
                    "with a file. An empty source has no statements to check, "
                    "so no offence found would describe a look nobody took.")
        offs = G.audit_ms_scale(source)
        if not offs:
            return ("No offence found. Note the two accepted false negatives: "
                    "a factor smuggled through a variable, and the label and "
                    "the multiplication in different statements.")
        return "\n".join(f"line {o.line_no}: {o.line}\n  {o.reason}" for o in offs)

    def gpu_verify(self, kind: str, actual: Any, reference: Any = "",
                   tolerance: float = 1e-4) -> str:
        """Compare two sequences of numbers.

        `actual` and `reference` accept either shape a caller actually holds:
        a string of numbers separated by commas or spaces, or a list/tuple of
        numbers. Both, because the message this method used to return on a
        bad value said "must be a list of numbers" while the body only ever
        handled a string -- a caller who followed the message got
        `AttributeError: 'list' object has no attribute 'replace'`, measured
        2026-11-06. A refusal that names a type the method cannot accept is a
        refusal that teaches the wrong call.

        Text results, not raised errors, because every `gpu_*` method answers
        in prose a model can read; the two-`ok` problem the wire methods had
        does not apply to a string return.
        """
        G = self._gpu()
        try:
            act = _as_numbers(actual, "actual")
        except ValueError as exc:
            return str(exc)
        if not _has_numbers(reference):
            return ("A reference is required. Without one this is not a "
                    "verification - it is a restatement of the kernel's own "
                    "output, which is what a fast-but-wrong kernel produces.")
        try:
            ref = _as_numbers(reference, "reference")
        except ValueError as exc:
            return str(exc)
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
            self._record("gpu_compile", {"name": name, "ok": False})
            return f"Could not compile: {exc}"
        self._record("gpu_compile",
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
        self._record("gpu_bench", {"name": name, "ok": res.ok})
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
                # An offence with no line is a file that was never read, and
                # "line 0: " would claim the opposite -- that line zero has
                # something on it. Print the reason instead.
                out += [(f"  line {o.line_no}: {o.line}" if o.line
                         else f"  {o.reason}") for o in offs[:10]]
            return "\n".join(out)
        if not source.strip():
            return ("Nothing to audit: pass `source` with the text, or `path` "
                    "with a file. An empty source has no statements to check, "
                    "so no offence found would describe a look nobody took.")
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
            self._record("cpu_compile", {"name": name, "ok": False})
            return f"Could not compile: {exc}"
        self._record("cpu_compile",
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
        self._record("cpu_run_isolated",
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
        self._record("cpu_tune",
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



# What each phase was doing, in words a caller can act on. Kept beside
# `find_gaps` rather than inlined so the sentence and the phase list cannot
# drift: a phase present in one and not the other would print a blank.
_GAP_PHASES = {
    "import": "loading the meta-cognition module",
    "pipeline": "building the forge pipeline",
    "analyser": "constructing the analyser",
    "analysis": "analysing the library",
}


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
        method = None
        known = True
        started = time.time()
        try:
            req = json.loads(line)
            rid = req.get("id")
            method = req.get("method")
            params = req.get("params") or {}
            fn = svc.methods.get(method)
            if fn is None:
                known = False
                out = {"id": rid, "ok": False, "error": "unknown method: %s" % method}
            else:
                # Through `call_method`, not `fn(**params)`: the gate has to be
                # on the only path a request can take. Written this way after a
                # defect of exactly this shape -- `anchor_ledger` had a method,
                # a bridge registration and eight passing assertions, and the
                # wire answered `unknown method` because the dispatch dict had
                # never been told about it. A gate consulted anywhere but here
                # would be the same mistake with higher stakes.
                res = svc._call_method(method, params)
                if isinstance(res, dict) and res.get("awaiting_confirmation"):
                    out = {"id": rid, "ok": False, **res}
                else:
                    out = {"id": rid, "ok": True, "result": res}
        except Exception as exc:  # noqa: BLE001
            out = {"id": rid, "ok": False,
                   "error": "%s: %s" % (type(exc).__name__, exc),
                   "trace": traceback.format_exc()[-1200:]}
        finally:
            # The meter is written here, before the answer goes out, so a caller
            # that reads the reply and then asks `report` cannot see the call
            # missing from the tally. Counted-not-thrown: a meter that can fail
            # a request is worse than no meter.
            if method:
                try:
                    svc.store.record_wire_call(
                        method, ok=bool(out.get("ok")),
                        ms=(time.time() - started) * 1000.0, known=known)
                except Exception:                              # noqa: BLE001
                    svc._wire_call_failures += 1
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
