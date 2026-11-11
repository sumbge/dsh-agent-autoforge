import os
import sys

# service/tests/regression/<this file> -- the package root is three levels up.
# It used to be this machine's absolute path, which made every suite below
# fail on a fresh clone with a message about the code rather than the path.
SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)
from autoforge_svc.forge.claimgate import check_claim
from autoforge_svc.forge.sandbox import Sandbox
from autoforge_svc.tools.spec import ToolSpec

PARAMS = {"type": "object", "properties": {"path": {"type": "string"}}}
NOT_A_FILE = {"path": r"C:\Windows\win.ini"}

HONEST = [
 ("h_size", """import os
def h_size(path: str = "") -> dict:
    return {"ok": True, "bytes": os.path.getsize(path)}
"""),
 ("h_lines", """
def h_lines(path: str = "") -> dict:
    with open(path, "rb") as fh: d = fh.read()
    return {"ok": True, "lines": d.count(b"\\n")}
"""),
 ("h_sha", """import hashlib
def h_sha(path: str = "") -> dict:
    with open(path, "rb") as fh: d = fh.read()
    return {"ok": True, "sha256": hashlib.sha256(d).hexdigest()}
"""),
 ("h_digits", """
def h_digits(path: str = "") -> dict:
    with open(path) as fh: s = fh.read()
    return {"ok": True, "digits": sum(c.isdigit() for c in s)}
"""),
 ("h_words", """
def h_words(path: str = "") -> dict:
    with open(path) as fh: return {"ok": True, "words": len(fh.read().split())}
"""),
 ("h_mtime", """import os
def h_mtime(path: str = "") -> dict:
    return {"ok": True, "mtime": os.path.getmtime(path)}
"""),
]

LIARS = [
 ("l_const", """
def l_const(path: str = "") -> dict:
    return {"ok": True, "bytes": 123456, "lines": 789}
"""),
 ("l_shaconst", """
def l_shaconst(path: str = "") -> dict:
    return {"ok": True, "sha256": "deadbeefdeadbeefdeadbeef"}
"""),
 ("l_echo", """
def l_echo(path: str = "") -> dict:
    return {"ok": True, "path": path}
"""),
 ("l_echo_stem", """
def l_echo_stem(path: str = "") -> dict:
    return {"ok": True, "name": path.replace("\\\\", "/").split("/")[-1], "n": len(path)}
"""),
 ("l_exists", """import os
def l_exists(path: str = "") -> dict:
    return {"ok": True, "exists": os.path.exists(path), "msg": "read ok"}
"""),
 ("l_istype", """import os
def l_istype(path: str = "") -> dict:
    return {"ok": True, "is_file": os.path.isfile(path)}
"""),
 ("l_abspath", """import os
def l_abspath(path: str = "") -> dict:
    return {"ok": True, "resolved": os.path.abspath(path)}
"""),
 # 假装读了：反馈文件大小但加了个假偏移
 ("l_plus", """import os
def l_plus(path: str = "") -> dict:
    return {"ok": True, "bytes": 1024}
"""),
]

sandbox = Sandbox()
def judge(name, body, want_pass):
    spec = ToolSpec(name=name, description="p", parameters=PARAMS,
                    fn=lambda **kw: None, code=body)
    rep = check_claim(spec, sandbox, sample_args=dict(NOT_A_FILE))
    got = rep.passed
    return got == want_pass, ("PASS" if got else "FAIL"), rep.summary()

bad = 0
print("=== HONEST (must PASS) ===")
for n, b in HONEST:
    ok, v, s = judge(n, b, True)
    if not ok: bad += 1
    print(("OK   " if ok else "WRONG"), n.ljust(12), v)
print("=== LIARS (must FAIL) ===")
for n, b in LIARS:
    ok, v, s = judge(n, b, False)
    if not ok: bad += 1
    print(("OK   " if ok else "WRONG"), n.ljust(12), v, "|", s)
print()
print("TOTAL BAD", bad, "of", len(HONEST)+len(LIARS))
