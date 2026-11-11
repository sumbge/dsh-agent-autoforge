"""What the bridge assumes when the preset says nothing.

`DEFAULTS` in `lib/index.js` is the fallback for a mount that sets no options.
Three of its four paths named another machine:

    python: 'python'
    cwd:    'C:\\somebody\\dsh\\autoforge-svc'
    home:   'C:\\somebody\\dsh\\autoforge-svc\\.data'
    skills: 'C:\\somebody\\AppData\\Local\\autoforge\\skills'

None of those exist here. The shipped preset overrides all four, which is why
nothing broke -- a default that is never reached is a default nobody has had to
read. The next person to mount this bundle against a checkout without a preset
gets `spawn python ENOENT` in a service that reports itself as mounted, and the
error will name the wrong directory.

What "correct" means here is narrow and worth stating, because a default cannot
be universal:
  * `cwd` must be the `service/` directory beside this very file -- the one
    thing a bundle can know about its own layout;
  * `home` must be absolute and per-user, and must not be *inside* the checkout,
    because a service that writes its database into `node_modules` loses it to
    the next `pnpm install`;
  * `python` must not claim a bare `python` is fine: on Windows that resolves to
    the Microsoft Store placeholder, which prints a banner and exits 9009. The
    candidate list is tried in order and the first one that runs wins;
  * `skills` must be the user-level skills directory, not the checkout's.

Run:
  $env:PYTHONPATH = "...\\dsh-agent-autoforge\\service"
  & $py -P ...\\chk_defaults.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICE = os.path.dirname(os.path.dirname(HERE))     # regression -> tests -> service
PLUGIN = os.path.dirname(SERVICE)
LIB = os.path.join(PLUGIN, "lib", "index.js")

FAILS: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print("  OK   " + label)
    else:
        print("  FAIL " + label + (" <- " + str(detail)[:400] if detail else ""))
        FAILS.append(label)


def read_defaults() -> dict:
    """The DEFAULTS object as JavaScript evaluates it.

    Evaluated rather than parsed: the fix replaces a string table with a small
    amount of logic (resolve the service directory from `import.meta.url`,
    probe python candidates), so a regex over the source would assert the shape
    of the code instead of the value it produces.

    That cutting has to start at PYTHON_CANDIDATES, not at `const DEFAULTS`:
    DEFAULTS names that constant, so slicing only the object gives a
    ReferenceError, which is how this draft failed the first time.
    """
    src = open(LIB, encoding="utf-8").read()
    start = src.index("const PYTHON_CANDIDATES")
    i = src.index("const DEFAULTS = {")
    # walk to the brace that closes DEFAULTS
    j0 = src.index("{", i)
    depth = 0
    for j in range(j0, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    body = src[start:end]
    prelude = """
import { fileURLToPath } from 'node:url'
import { dirname, join, resolve } from 'node:path'
import { existsSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { homedir } from 'node:os'
"""
    script = prelude + body + "\nconsole.log(JSON.stringify(DEFAULTS))\n"
    tmp = os.path.join(PLUGIN, "lib", "_chk_defaults_tmp.mjs")
    open(tmp, "w", encoding="utf-8", newline="\n").write(script)
    try:
        r = subprocess.run(["node", tmp], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    finally:
        os.unlink(tmp)
    if r.returncode != 0:
        raise RuntimeError("could not evaluate DEFAULTS: " + (r.stderr or "")[-800:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def read_preset_home() -> str:
    """The `home` the shipped preset mounts this bundle with, or "".

    Reads the preset rather than hard-coding it: the assertion is "the default
    agrees with what actually ships", and a copy of the value here would drift
    the moment either side moves -- which is how the original DEFAULTS got a
    path off a different machine in the first place.

    The file uses a `!js` tag, so `yaml.safe_load` refuses it. A constructor
    for that tag is registered rather than the line being regexed, because the
    `prefix:` block below it is a multi-line string full of paths.
    """
    path = os.path.join(PLUGIN, "presets", "autoforge.patch.yml")
    if not os.path.isfile(path):
        return ""
    try:
        import yaml
    except ImportError:
        return ""
    loader = getattr(yaml, "SafeLoader", None)
    if loader is None:
        return ""

    class _Tolerant(loader):                             # type: ignore[misc,valid-type]
        pass

    _Tolerant.add_multi_constructor(
        "tag:yaml.org,2002:js", lambda l, suffix, node: None)
    try:
        doc = yaml.load(open(path, encoding="utf-8").read(), Loader=_Tolerant)
    except Exception:                                   # noqa: BLE001
        return ""
    for entry in doc or []:
        for ins in (entry or {}).get("insert") or []:
            for plug in (ins or {}).get("config", {}).get("plugins") or []:
                cfg = (plug or {}).get("config") or {}
                if cfg.get("home"):
                    return str(cfg["home"])
    return ""


def main() -> int:
    d = read_defaults()
    print("-- what the bridge assumes when nothing is configured")
    print("   " + json.dumps(d, ensure_ascii=False))

    cwd = str(d.get("cwd") or "")
    check("cwd points at the service directory beside lib/",
          os.path.normcase(os.path.abspath(cwd))
          == os.path.normcase(os.path.abspath(SERVICE)), cwd)
    check("and that directory really contains the package",
          os.path.isfile(os.path.join(cwd, "autoforge_svc", "__main__.py")), cwd)

    home = str(d.get("home") or "")
    check("home is an absolute path", os.path.isabs(home), home)
    check("home is not inside the checkout",
          not os.path.normcase(home).startswith(os.path.normcase(PLUGIN) + os.sep),
          home)
    check("home is not a directory that does not exist yet under another user's name",
          "32535" not in home, home)
    # The load-bearing one, and the first two drafts of it were both wrong.
    #
    # Draft 1 asserted only the properties (absolute, outside the checkout, no
    # other user's name) and a mutant that moved home to
    # `%LOCALAPPDATA%\autoforge\autoforge` passed all of them.
    #
    # Draft 2 asserted it matched the service's own `_home()` -- which is
    # `%LOCALAPPDATA%\autoforge`, NOT the `...\autoforge\data` the shipped
    # preset uses. So that assertion demanded the wrong answer: a mount with no
    # preset would have opened a second, empty library beside the real one.
    # I have created that exact stray file five times by accident; a default
    # that does it on purpose is worse.
    #
    # What actually matters: preset or no preset, one library. Read the preset
    # and compare.
    preset = read_preset_home()
    check("home is the directory the shipped preset points at",
          bool(preset) and os.path.normcase(os.path.abspath(home))
          == os.path.normcase(os.path.abspath(preset)), f"{home} vs preset {preset}")

    skills = str(d.get("skills") or "")
    check("skills is absolute", os.path.isabs(skills), skills)
    check("skills is not inside the checkout",
          not os.path.normcase(skills).startswith(os.path.normcase(PLUGIN) + os.sep),
          skills)
    check("and skills is a sibling of home, not buried inside it",
          os.path.normcase(os.path.dirname(os.path.abspath(skills)))
          == os.path.normcase(os.path.dirname(os.path.abspath(home))),
          f"{skills} / {home}")

    py = d.get("python")
    resolved = py if isinstance(py, str) else (py or [])
    check("python is not a bare 'python', which on Windows is a store stub",
          resolved != "python", resolved)
    check("python is a string or a list of candidates",
          isinstance(resolved, (str, list)), type(resolved).__name__)
    if isinstance(resolved, list):
        check("there is at least one candidate", len(resolved) >= 1, resolved)
        check("and at least one of them can run",
              any(_runs(p) for p in resolved), resolved)

    check("no path in DEFAULTS names another machine's user",
          all("32535" not in str(v) for v in d.values()),
          [k for k, v in d.items() if "32535" in str(v)])

    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1


def _runs(python: str) -> bool:
    try:
        r = subprocess.run([python, "-c", "print(1)"], capture_output=True,
                           timeout=10)
        return r.returncode == 0 and r.stdout.strip() == b"1"
    except Exception:                                   # noqa: BLE001
        return False


if __name__ == "__main__":
    sys.exit(main())
