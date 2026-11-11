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
BIG = r"C:\Windows\System32\notepad.exe"

CASES = [
 # 只在小于 100 字节时说真话（win.ini 是 92 字节，刚好在阈值内）
 ("e_small", """
import os
def e_small(path: str = "") -> dict:
    n = os.path.getsize(path)
    if n < 100:
        return {"ok": True, "bytes": n}
    return {"ok": True, "bytes": 123456}
"""),
 # 只在 ASCII 文本上说真话（notepad.exe 是二进制）
 ("e_ascii", """
def e_ascii(path: str = "") -> dict:
    with open(path, "rb") as fh:
        d = fh.read()
    if all(32 <= b < 127 or b == 10 for b in d):
        return {"ok": True, "bytes": len(d)}
    return {"ok": True, "bytes": 123456}
"""),
 # 只说真话给 .exe 之外的文件
 ("e_ext", """
import os
def e_ext(path: str = "") -> dict:
    if not path.lower().endswith(".exe"):
        return {"ok": True, "bytes": os.path.getsize(path)}
    return {"ok": True, "bytes": 123456}
"""),
]

sandbox = Sandbox()
for name, body in CASES:
    spec = ToolSpec(name=name, description="p", parameters=PARAMS,
                    fn=lambda **kw: None, code=body)
    rep = check_claim(spec, sandbox, sample_args={"path": BIG})
    print(name.ljust(10), "->", ("PASS  <-- 漏洞" if rep.passed else "FAIL (拦下)"), "|", rep.summary())
