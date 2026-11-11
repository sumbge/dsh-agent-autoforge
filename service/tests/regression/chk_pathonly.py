"""Does `x-claim: path-only` judge path measurers without freeing liars?

Measured 2026-11-01 in `chk_pathaxis.py`: an honest tool answering
`{"len": len(p)}` and a bare liar answering `{"bytes": 123456}` land on the SAME
verdict under the file probe, because that probe holds the path still on
purpose. The gate then called the honest tool "a constant about itself", which
is false. The fix is a declaration -- `x-claim: path-only` -- that swaps the
instrument, and this file states what that swap has to achieve:

  * a declaration that is TRUE and paid for passes (its answer moves with the
    path and not with the bytes at a fixed path),
  * a declaration that is FALSE fails (constant: answer does not move with the
    path; file reader: answer moves when only the bytes move),
  * a tool that declares nothing is judged exactly as before -- the fix must not
    become a hole in the file probe,
  * a path echo tool that declares path-only is honest: it really is reporting
    the path, and it is not claiming to measure the file.

Run with the real interpreter, not through the agent sandbox (30 s CPU cap):

    $env:PYTHONPATH = '<repo>/service'
    python chk_pathonly.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autoforge_svc.forge.claimgate import check_claim  # noqa: E402
from autoforge_svc.forge.sandbox import Sandbox  # noqa: E402
from autoforge_svc.tools.spec import ToolSpec  # noqa: E402


def params(claim: str | None) -> dict:
    schema: dict = {"type": "string"}
    if claim is not None:
        schema["x-claim"] = claim
    return {"type": "object", "properties": {"path": schema}}


SAMPLE = {"path": r"C:\Windows\win.ini"}

# name, code, declaration, expected pass
CASES: list[tuple[str, str, str | None, bool]] = [
    # -- declared path-only, declaration is true -------------------------
    ("p_pathlen", """
import os
def p_pathlen(**kw):
    p = kw.get('path')
    return {'len': len(p), 'stem': os.path.basename(p)}
""", "path-only", True),
    ("p_dirparts", """
import os
import re
def p_dirparts(**kw):
    p = kw.get('path')
    return {'parts': len([x for x in re.split(r'[\\\\/]+', p) if x]),
            'name_len': len(p)}
""", "path-only", True),
    ("p_dirname", """
import os
def p_dirname(**kw):
    p = kw.get('path')
    return {'dir': os.path.dirname(p), 'n': len(os.path.dirname(p))}
""", "path-only", True),
    # -- declared path-only, declaration is FALSE ------------------------
    ("p_const", """
def p_const(**kw):
    return {'bytes': 123456, 'lines': 789}
""", "path-only", False),
    ("p_filereader", """
import os
def p_filereader(**kw):
    p = kw.get('path')
    try:
        n = os.path.getsize(p)
    except OSError:
        n = -1
    return {'bytes': n}
""", "path-only", False),
    # The exact hole the first revision of `content_blind` had: a size answer
    # does not move with a path, so only `content_blind` can catch this one --
    # and it could not, while the same-path rewrite kept the length. Both
    # halves now move, so a size reader moves and is caught.
    ("p_filereader_len_same_dir", """
import os
def p_filereader_len_same_dir(**kw):
    p = kw.get('path')
    try:
        n = os.path.getsize(p)
    except OSError:
        n = -1
    return {'n': n, 'nd': len(os.path.dirname(p))}
""", "path-only", False),
    ("p_sha_reader", """
import hashlib
import os
def p_sha_reader(**kw):
    p = kw.get('path')
    try:
        with open(p, 'rb') as fh:
            return {'sha': hashlib.sha256(fh.read()).hexdigest()[:16], 'nd': len(os.path.dirname(p))}
    except OSError:
        return {'sha': 'n/a', 'nd': len(os.path.dirname(p))}
""", "path-only", False),
    # A tool whose declaration is TRUE but which also happens to read the file:
    # it reads bytes AND reports the path. `content_blind` is meant to fail it,
    # so the case documents that the declaration is not a licence to read.
    ("p_pathlen_plus_bytes", """
import os
def p_pathlen_plus_bytes(**kw):
    p = kw.get('path')
    try:
        n = os.path.getsize(p)
    except OSError:
        n = -1
    return {'len': len(p), 'bytes': n}
""", "path-only", False),
    # -- no declaration: the file probe must still judge these ----------
    ("n_pathlen_undeclared", """
import os
def n_pathlen_undeclared(**kw):
    p = kw.get('path')
    return {'len': len(p), 'stem': os.path.basename(p)}
""", None, False),
    ("n_filereader_undeclared", """
import os
def n_filereader_undeclared(**kw):
    p = kw.get('path')
    try:
        n = os.path.getsize(p)
    except OSError:
        n = -1
    return {'bytes': n}
""", None, True),
    ("n_const_undeclared", """
def n_const_undeclared(**kw):
    return {'bytes': 123456, 'lines': 789}
""", None, False),
]


def main() -> int:
    sandbox = Sandbox()
    rows: list[str] = []
    wrong = 0
    for name, code, claim, want in CASES:
        spec = ToolSpec(name=name, description="p", parameters=params(claim),
                        fn=lambda **kw: None, code=code)
        rep = check_claim(spec, sandbox, sample_args=dict(SAMPLE))
        got = bool(rep.passed)
        mark = "ok  " if got == want else "WRONG"
        if got != want:
            wrong += 1
        rows.append(f"{mark} {name:<24} pass={got!s:<5} want={want!s:<5} "
                    f"declared={claim!s:<10} {rep.summary()}")
        for f in rep.findings:
            flag = "+" if f.ok else "-"
            rows.append(f"       {flag} {f.name}: {f.detail[:150]}")
    print("\n".join(rows))
    print(f"\nTOTAL WRONG {wrong} of {len(CASES)}")
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
