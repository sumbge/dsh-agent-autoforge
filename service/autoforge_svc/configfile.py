"""Persistent user config for `auto` — written by `auto setup`, read by everything.

Resolution order at use time (see `cli._config`):

    explicit flag  >  environment variable  >  this file  >  built-in default

The file exists so that a one-time `auto setup` is genuinely enough. Relying on
environment variables alone breaks in a way that is invisible and infuriating:
a shell or terminal that was already open when the variable was set never sees
it, so the same command works in one window and fails in the next. A file has
no such lag.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

# Keys the wizard writes. A hand-edited file may carry others; `save` merges
# rather than replaces, so anything unrecognised survives a round-trip.
KNOWN = ("base_url", "model", "api_key", "max_tokens", "proxy", "fast")

DEFAULT_PATH = Path.home() / ".autoforge" / "config.json"

#: The local relay used when `proxy` is on. Defined once here because three
#: callers need the same answer -- the text client, the setup wizard and the
#: vision client -- and a second copy is how one of them ends up silently
#: unproxied while the others work.
PROXIES = {"http": "socks5://127.0.0.1:9674",
           "https": "socks5://127.0.0.1:9674"}


def config_path() -> Path:
    """`$AUTOFORGE_CONFIG` wins; otherwise ~/.autoforge/config.json."""
    override = os.environ.get("AUTOFORGE_CONFIG")
    return Path(override) if override else DEFAULT_PATH


def load(path: Path | None = None) -> dict[str, Any]:
    """Read the file. A missing or corrupt file is an empty config, not an error.

    Nobody should be unable to run the tool because a config got truncated by a
    crash mid-write.
    """
    p = path or config_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(values: dict[str, Any], path: Path | None = None) -> Path:
    """Merge `values` into the file and return its path."""
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    merged = load(p)
    merged.update({k: v for k, v in values.items() if v is not None})
    text = json.dumps(merged, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    p.write_text(text, encoding="utf-8")
    _restrict(p)
    return p


def _restrict(p: Path) -> None:
    """Best-effort 0600 — the file holds an API key.

    Windows has no POSIX mode bits, so a failure here is expected and not worth
    surfacing; the ACL on the user's own profile directory is the real guard.
    """
    try:
        p.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def mask(secret: Any) -> str:
    """Render a key for display without leaking it."""
    if not secret:
        return "(unset)"
    s = str(secret)
    if len(s) <= 8:
        return "*" * len(s)
    return f"{s[:4]}...{s[-4:]} (len {len(s)})"
