"""Truth table: does an answer move with the file, or only with the path?

The claim gate's `difference` check varies two things at once -- file size and
file content -- while deliberately holding the three probe *paths* the same
length, so that a tool which merely echoes its argument cannot pass. That
choice is right, but it means a tool answering only about the path gets the
same verdict as a tool answering a constant, and those two are different
things. Before adding a path-sensitive probe (or declaring the case
un-probeable) this script measures which tools fall on which side.

Method: run each tool body directly through the sandbox -- not through
`check_claim` -- on four controlled inputs, so the dimensions are separated:

    base    path P1, content C1
    path    path P2 (same length, different name), content C1   -> path-only change
    content path P1, content C2 (same length, different bytes)   -> content-only change
    length  path P3 (different length), content C1              -> another path-only change

A tool is file-sensitive if `content` moved its answer, path-sensitive if
either `path` or `length` did.

Run with the real interpreter (the sandbox has a 30s CPU budget and this
spawns a subprocess per cell):

    $env:PYTHONPATH = '<plugin>\\service'
    & 'C:\\Users\\sumbger\\AppData\\Local\\Programs\\Python\\Python313\\python.exe' chk_pathaxis.py
"""

import os
import random
import shutil
import string
import sys
import tempfile

# Three levels: regression -> tests -> service, which is the directory that
# holds `autoforge_svc`. It said two, so it resolved to `tests/` and the import
# below only worked when the caller had exported PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from autoforge_svc.forge.sandbox import Sandbox  # noqa: E402

PARAMS = {"type": "object", "properties": {"path": {"type": "string"}}}

LOWER = string.ascii_lowercase
UPPER = string.ascii_uppercase
DIGITS = string.digits
PUNCT = ".,;:!?"


def payload(size, newlines, pools):
    """Deterministic bytes drawn only from the given character pools."""
    rnd = random.Random(hash((size, newlines, tuple(pools))) & 0xFFFF)
    pool = "".join(pools)
    body = [rnd.choice(pool) for _ in range(max(size, newlines + 1))]
    step = max(1, len(body) // (newlines + 1))
    for i in range(newlines):
        pos = min(len(body) - 1, i * step)
        body[pos] = "\n"
    return ("".join(body)[:size]).encode("ascii", "replace")


TOOLS = {
    # named "h_" when this suite was written -- that prefix was a guess, not a
    # finding. The truth table below is what decides.
    "h_size": "import os\ndef {n}(**kw):\n    p = kw['path']\n    try:\n        s = os.path.getsize(p)\n    except OSError as e:\n        return {{'ok': False, 'err': str(e)}}\n    return {{'ok': True, 'bytes': s}}",
    "h_lines": "def {n}(**kw):\n    p = kw['path']\n    try:\n        with open(p, 'rb') as fh:\n            data = fh.read()\n    except OSError as e:\n        return {{'ok': False, 'err': str(e)}}\n    return {{'ok': True, 'lines': data.count(b'\\n')}}",
    "h_sha": "import hashlib\ndef {n}(**kw):\n    p = kw['path']\n    try:\n        with open(p, 'rb') as fh:\n            d = hashlib.sha256(fh.read()).hexdigest()\n    except OSError as e:\n        return {{'ok': False, 'err': str(e)}}\n    return {{'ok': True, 'sha256': d}}",
    "h_digits": "def {n}(**kw):\n    p = kw['path']\n    try:\n        with open(p, 'rb') as fh:\n            data = fh.read()\n    except OSError as e:\n        return {{'ok': False, 'err': str(e)}}\n    return {{'ok': True, 'digits': sum(chr(b).isdigit() for b in data)}}",
    "h_words": "def {n}(**kw):\n    p = kw['path']\n    try:\n        with open(p, 'r', encoding='utf-8', errors='replace') as fh:\n            data = fh.read()\n    except OSError as e:\n        return {{'ok': False, 'err': str(e)}}\n    return {{'ok': True, 'words': len(data.split())}}",
    "h_mtime": "import os\ndef {n}(**kw):\n    p = kw['path']\n    try:\n        s = os.stat(p)\n    except OSError as e:\n        return {{'ok': False, 'err': str(e)}}\n    return {{'ok': True, 'mtime': round(s.st_mtime, 3)}}",
    "h_pathlen": "import os\ndef {n}(**kw):\n    p = kw['path']\n    return {{'ok': True, 'len': len(p), 'stem': os.path.splitext(os.path.basename(p))[0]}}",
    "h_dirparts": "import os, re\ndef {n}(**kw):\n    p = kw['path']\n    return {{'ok': True, 'parts': len([x for x in re.split(r'[\\\\/]+', p) if x]), 'name_len': len(p)}}",
    "l_echo": "def {n}(**kw):\n    return {{'ok': True, 'path': kw['path']}}",
    "l_echo_stem": "import os\ndef {n}(**kw):\n    p = kw['path']\n    return {{'ok': True, 'stem': os.path.splitext(os.path.basename(p))[0], 'len': len(p)}}",
    "l_abspath": "import os\ndef {n}(**kw):\n    p = kw['path']\n    return {{'ok': True, 'abs': os.path.abspath(p), 'n': len(os.path.abspath(p))}}",
    "l_const": "def {n}(**kw):\n    return {{'ok': True, 'bytes': 123456, 'lines': 789}}",
}


def run(sandbox, name, body, path):
    code = body.format(n=name)
    return sandbox.run(code, name, {"path": path})


def answer(result):
    """What the tool said, without the transport's ok/stdout/err envelope."""
    return repr(getattr(result, "output", result))


def main():
    sandbox = Sandbox()
    root = tempfile.mkdtemp(prefix="pathaxis_")
    # All three directories are the same length, and all three basenames are the
    # same length, so a same-length triple differs only in the characters used.
    dirs = []
    rnd = random.Random(5)
    for _ in range(3):
        while True:
            name = "".join(rnd.choice(LOWER + DIGITS) for _ in range(12))
            d = os.path.join(root, name)
            if os.path.isdir(d):
                continue
            os.mkdir(d)
            dirs.append(d)
            break
    # A fourth directory, deliberately a different length from the others.
    long_dir = os.path.join(root, "d" * 26)
    os.mkdir(long_dir)

    c1 = payload(24, 3, [LOWER, DIGITS])
    c2 = payload(24, 3, [UPPER, " ", PUNCT])

    f1 = os.path.join(dirs[0], "probeaaaaa.txt")
    f2 = os.path.join(dirs[1], "probeaaaaa.txt")
    f3 = os.path.join(long_dir, "probeaaaaa.txt")
    for p in (f1, f2, f3):
        with open(p, "wb") as fh:
            fh.write(c1)
    # Identical mtimes across the triple, so that a stat-based tool does not
    # look path-sensitive for the trivial reason that the files were written a
    # few milliseconds apart. Only the path dimension is being moved here.
    for p in (f1, f2, f3):
        os.utime(p, (1_700_000_000, 1_700_000_000))
    # Same path as f1, different content.
    f4 = os.path.join(dirs[0], "probeaaaaa.txt")

    cells = {
        "base": f1,
        "path": f2,      # same content, different (same-length) directory name
        "content": f4,   # same path, different content -- written below
        "length": f3,    # same content, longer path
    }

    out = []
    for name, body in TOOLS.items():
        got = {}
        for cell, path in cells.items():
            if cell == "content":
                with open(f1, "wb") as fh:
                    fh.write(c2)
                # The file really did change, so its mtime really did change.
                os.utime(f1, (1_700_000_100, 1_700_000_100))
            value = run(sandbox, name, body, path)
            got[cell] = value
            if cell == "content":
                with open(f1, "wb") as fh:
                    fh.write(c1)
                os.utime(f1, (1_700_000_000, 1_700_000_000))
        base = answer(got["base"])
        moved_path = base != answer(got["path"]) or base != answer(got["length"])
        moved_content = base != answer(got["content"])
        out.append((name, moved_content, moved_path, got))

    print(f"{'tool':<14} {'file-sensitive':<15} {'path-sensitive':<15} verdict")
    print("-" * 66)
    for name, fc, fp, got in out:
        if fc and not fp:
            verdict = "measures the file"
        elif fc and fp:
            verdict = "measures both"
        elif fp and not fc:
            verdict = "measures the path only"
        else:
            verdict = "constant"
        print(f"{name:<14} {str(fc):<15} {str(fp):<15} {verdict}")

    print()
    for name, fc, fp, got in out:
        pad = " " * len(name)
        print(f"{name}: base={answer(got['base'])[:90]}")
        print(f"{pad}  path={answer(got['path'])[:90]}")
        print(f"{pad}  cont={answer(got['content'])[:90]}")
        print(f"{pad}  len ={answer(got['length'])[:90]}")

    shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
