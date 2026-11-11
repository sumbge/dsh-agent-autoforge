"""Tool specs and registry (ported).

`__all__` below used to name `ToolRegistry` and `ToolResult` while the module
imported only from `.spec` -- so `from autoforge_svc.tools import ToolRegistry`
raised `ImportError: cannot import name 'ToolRegistry'`, and the `__all__` was
a claim about names that did not exist. Nobody was harmed (every real caller
imports either `from .tools import exposure` or `from .tools.registry import
ToolRegistry`), which is exactly how a false export survives: it is only wrong
for the reader who trusts it.

Fixed 2026-11-03 by importing what is declared. `registry` imports `.spec` and
`..autonomy` lazily inside functions, so there is no cycle to trip over.
"""
from .spec import ToolSpec, ToolState, ToolStats, TriggerProbe
from .registry import ToolRegistry, ToolResult

__all__ = ["ToolSpec", "ToolState", "ToolStats", "TriggerProbe", "ToolRegistry", "ToolResult"]
