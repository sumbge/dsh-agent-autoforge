import os
import sys

# service/tests/regression/<this file> -- the package root is three levels up.
# It used to be this machine's absolute path, which made every suite below
# fail on a fresh clone with a message about the code rather than the path.
SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)
from autoforge_svc.forge.sandbox import Sandbox

CODE = """
CNT = 0
def counter(path: str = "") -> dict:
    global CNT
    CNT += 1
    import os
    return {"ok": True, "call_index": CNT, "pid": os.getpid()}
"""
sb = Sandbox()
for i in range(4):
    r = sb.run(CODE, "counter", {})
    print(i, "->", str(getattr(r, "output", None))[:80], "| ok:", getattr(r, "ok", None))
