import os
import sys

# service/tests/regression/<this file> -- the package root is three levels up.
# It used to be this machine's absolute path, which made every suite below
# fail on a fresh clone with a message about the code rather than the path.
SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)
from autoforge_svc.forge.claimgate import ClaimGate, check_claim
from autoforge_svc.forge.sandbox import Sandbox
from autoforge_svc.tools.spec import ToolSpec

PARAMS = {"type": "object", "properties": {"path": {"type": "string"}}}
BODY = """
import os
def g_bigfile(path: str = "") -> dict:
    n = os.path.getsize(path)
    return {"ok": True, "bytes": n if n < 100000 else 123456}
"""
spec = ToolSpec(name="g_bigfile", description="p", parameters=PARAMS,
                fn=lambda **kw: None, code=BODY)
rep = check_claim(spec, Sandbox(), sample_args={"path": r"C:\Windows\System32\notepad.exe"})
print("passed:", rep.passed, "|", rep.summary())
for f in rep.findings:
    print(" ", "ok" if f.ok else "NO", f.name, ":", f.detail[:200])
    if f.evidence: print("     ev:", str(f.evidence)[:300])
