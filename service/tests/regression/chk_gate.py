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

# 函数名必须与工具名一致，否则沙箱找不到 entry，测的就不是门禁
CASES = [
    ("liar_constant", False, """
def liar_constant(path: str = "") -> dict:
    return {"ok": True, "bytes": 123456, "lines": 789}
"""),
    ("honest_size", True, """
import os
def honest_size(path: str = "") -> dict:
    if not os.path.isfile(path):
        return {"ok": False, "error": "not a file"}
    return {"ok": True, "bytes": os.path.getsize(path)}
"""),
    ("honest_lines", True, """
def honest_lines(path: str = "") -> dict:
    with open(path, "rb") as fh:
        data = fh.read()
    return {"ok": True, "lines": data.count(b"\\n")}
"""),
    ("honest_sha", True, """
import hashlib
def honest_sha(path: str = "") -> dict:
    with open(path, "rb") as fh:
        data = fh.read()
    return {"ok": True, "sha256": hashlib.sha256(data).hexdigest()}
"""),
    ("liar_sha", False, """
def liar_sha(path: str = "") -> dict:
    return {"ok": True, "sha256": "deadbeef" * 8}
"""),
    ("echo_only", False, """
def echo_only(path: str = "") -> dict:
    return {"ok": True, "path": path}
"""),
    ("echo_stem", False, """
def echo_stem(path: str = "") -> dict:
    return {"ok": True, "name": path.split("\\\\")[-1], "len": len(path)}
"""),
]

sandbox = Sandbox()
bad = 0
for name, expect_pass, body in CASES:
    spec = ToolSpec(name=name, description="probe", parameters=PARAMS,
                    fn=lambda **kw: None, code=body)
    rep = check_claim(spec, sandbox, sample_args={"path": r"C:\Windows\win.ini"})
    verdict = "PASS" if rep.passed else "FAIL"
    want = "PASS" if expect_pass else "FAIL"
    ok = verdict == want
    if not ok:
        bad += 1
    print(("OK   " if ok else "WRONG"), name.ljust(15), "->", verdict,
          "(want", want + ")", "|", rep.summary())
    for f in rep.findings:
        print("        ", "ok" if f.ok else "NO", f.name, ":", f.detail[:115])
print()
print("BAD", bad)
