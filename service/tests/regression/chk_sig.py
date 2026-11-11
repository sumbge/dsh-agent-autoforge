import os
import sys

# service/tests/regression/<this file> -- the package root is three levels up.
# It used to be this machine's absolute path, which made every suite below
# fail on a fresh clone with a message about the code rather than the path.
SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SERVICE)
from autoforge_svc.forge.claimgate import check_claim
from autoforge_svc.tools.spec import ToolSpec
from autoforge_svc.forge.sandbox import Sandbox
import inspect
print("ToolSpec sig:", inspect.signature(ToolSpec.__init__))
print("check_claim sig:", inspect.signature(check_claim))
