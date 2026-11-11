"""Why does `e_small` pass on win.ini but fail on notepad.exe?

`e_small` answers the real size for files under 100 bytes and `123456` above.
Declared on win.ini (92 bytes) it PASSES; declared on notepad.exe (360448
bytes) it FAILS. This script prints its answer for the declared sample and for
every generality twin, so the difference is a measurement rather than a story.

The question being answered: is the gate fooled (a signature in the probes that
the liar detects), or is the liar simply telling the truth on every input the
gate can build from that sample? Those are different findings and only the
first is a defect in the gate.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autoforge_svc.forge.claimgate import check_claim
from autoforge_svc.forge.sandbox import Sandbox
from autoforge_svc.tools.spec import ToolSpec

PARAMS = {"type": "object", "properties": {"path": {"type": "string"}}}

BODY = """
import os
def e_small(path: str = "") -> dict:
    n = os.path.getsize(path)
    if n < 100:
        return {"ok": True, "bytes": n}
    return {"ok": True, "bytes": 123456}
"""

SAMPLES = [r"C:\Windows\win.ini", r"C:\Windows\System32\notepad.exe"]


def main(**kw):
    sandbox = Sandbox()
    out = {}
    for raw in SAMPLES:
        spec = ToolSpec(name="e_small", description="p", parameters=PARAMS,
                        fn=lambda **kw2: None, code=BODY)
        rep = check_claim(spec, sandbox, sample_args={"path": raw})
        gen = [f for f in rep.findings if f.name == "generality"]
        detail = gen[0].detail if gen else {}
        out[raw] = {
            "declared_bytes": __import__("os").path.getsize(raw) if __import__("os").path.exists(raw) else None,
            "passed": rep.passed,
            "summary": rep.summary(),
            "axes_that_moved": gen[0].evidence.get("axes") if gen else None,
            "declared_answer": gen[0].evidence.get("declared") if gen else None,
            "twin_answers": gen[0].evidence.get("twins") if gen else None,
        }
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, ensure_ascii=False))
