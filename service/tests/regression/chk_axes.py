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

CASES = [
    ("w_words", "count words", """
def w_words(path: str = "") -> dict:
    with open(path) as fh:
        return {"ok": True, "words": len(fh.read().split())}
"""),
    ("w_digits", "count digits", """
def w_digits(path: str = "") -> dict:
    with open(path) as fh:
        s = fh.read()
    return {"ok": True, "digits": sum(c.isdigit() for c in s)}
"""),
    ("w_upper", "count uppercase", """
def w_upper(path: str = "") -> dict:
    with open(path) as fh:
        s = fh.read()
    return {"ok": True, "upper": sum(c.isupper() for c in s)}
"""),
    ("w_spaces", "count spaces", """
def w_spaces(path: str = "") -> dict:
    with open(path) as fh:
        s = fh.read()
    return {"ok": True, "spaces": s.count(" ")}
"""),
    ("w_longest", "longest line", """
def w_longest(path: str = "") -> dict:
    with open(path) as fh:
        lines = fh.read().splitlines() or [""]
    return {"ok": True, "longest": max(len(l) for l in lines)}
"""),
    ("w_entropy", "shannon entropy", """
import math, collections
def w_entropy(path: str = "") -> dict:
    with open(path) as fh:
        s = fh.read()
    if not s: return {"ok": True, "entropy": 0.0}
    c = collections.Counter(s)
    return {"ok": True, "entropy": round(-sum(n/len(s)*math.log2(n/len(s)) for n in c.values()), 4)}
"""),
    ("w_mtime", "mtime", """
import os
def w_mtime(path: str = "") -> dict:
    return {"ok": True, "mtime": os.path.getmtime(path)}
"""),
    ("w_first", "first char only", """
def w_first(path: str = "") -> dict:
    with open(path) as fh:
        s = fh.read(1)
    return {"ok": True, "first": s}
"""),
]

sandbox = Sandbox()
fails = []
for name, what, body in CASES:
    spec = ToolSpec(name=name, description="probe", parameters=PARAMS,
                    fn=lambda **kw: None, code=body)
    rep = check_claim(spec, sandbox, sample_args={"path": r"C:\Windows\win.ini"})
    v = "PASS" if rep.passed else "FAIL"
    if not rep.passed:
        fails.append((what, [f.detail[:150] for f in rep.findings if not f.ok]))
    det = ""
    for f in rep.findings:
        if f.name == "difference" and f.evidence:
            det = "A!=B:%s A!=C:%s" % (f.evidence.get("same_size_different_bytes"),
                                       f.evidence.get("different_size"))
    print(v.ljust(5), name.ljust(11), what.ljust(18), det)
print()
print("FALSE-FAILED", len(fails))
for what, ds in fails:
    print(" -", what, "|", ds[0] if ds else "")
