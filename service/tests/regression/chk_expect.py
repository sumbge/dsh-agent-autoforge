import os
import sys

# service/tests/regression/<this file> -- the package root is three levels up.
# It used to be this machine's absolute path, which made every suite below
# fail on a fresh clone with a message about the code rather than the path.
SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)
from autoforge_svc.forge.verifier import _expect_matches, _expect_renderings

out_repr = "{'ok': True, 'path': 'x', 'bytes': 92, 'lines': 8, 'sha256': 'abc'}"
out_json = '{"ok": true, "path": "x", "bytes": 92, "lines": 8}'

cases = [
    ('"bytes": 92',       out_repr, True,  "JSON-style expect vs repr output"),
    ("'bytes': 92",       out_repr, True,  "Python-style expect vs repr output"),
    ('"bytes": 92',       out_json, True,  "JSON vs JSON"),
    ("'bytes': 92",       out_json, True,  "Python-style vs JSON output"),
    ('"bytes": 93',       out_repr, False, "wrong number must FAIL"),
    ('"lines": 8',        out_repr, True,  "second field"),
    ('"bytes":   92',     out_repr, True,  "extra spaces"),
    ('true',              out_json, True,  "bare json true"),
    ('"missing": 1',      out_repr, False, "absent key must FAIL"),
]
bad = 0
for expect, out, want, why in cases:
    got, tried = _expect_matches(expect, out)
    ok = (got == want)
    if not ok: bad += 1
    print(('OK  ' if ok else 'FAIL'), repr(expect), '->', got, '| want', want, '|', why, '| tried', tried)
print("BAD", bad, "of", len(cases))
