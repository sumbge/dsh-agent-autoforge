"""Can the gate read the sentence the generator teaches a tool to write?

Under test: `forge/validity.py` (`scope_of_signature`, `declared_scope`) and
`autonomy/confirm.py` (`_normalise_scope`), and the critic it feeds.

The two halves were written months apart and never met. `forge/generator.py:375`
tells the model, in the prompt it generates tools from:

    "effect_signature": "pure | reads:<what> | writes:<what>"

Both readers of that field accepted only the seven fixed scope names, so every
honestly written `reads:file` fell through to `undeclared`, and `SCOPE_ALLOWANCES`
defines `undeclared` as all eight capabilities. A tool that reads one file was
therefore treated as able to write, delete, execute and reach the network.

The visible symptom was in `autonomy/roles.py`: the critic's whole allowance is
`filesystem_read`, so it refused every tool that described itself truthfully.
Measured 2026-11-04 before the fix: a real `ScopedRegistry` under `role="critic"`
exposed one tool out of four, and the three refusals were all tools whose
signature was `''` or `reads:...`.

A second thing is under test here, and it is the one that makes the first
worth fixing: the translation must not be generous in the direction that
matters. An unreadable signature still means `undeclared`. A signature that
names a write is still refused to the critic. Getting tools *through* the gate
is easy -- answer `pure` to everything -- and it would break exactly the thing
the gate exists for.

Run from the service directory:

    python tests/regression/chk_signature_scope.py

The database is a throwaway under %TEMP%; the production one is never opened.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc.autonomy.confirm import (_normalise_scope,        # noqa: E402
                                            declared_capabilities)
from autoforge_svc.autonomy.roles import ceiling, refuses, role_brief  # noqa: E402
from autoforge_svc.forge.validity import (SCOPE_ALLOWANCES,          # noqa: E402
                                          declared_scope, scope_of_signature)
from autoforge_svc.tools.registry import ToolRegistry                # noqa: E402
from autoforge_svc.tools.spec import ToolSpec, ToolState             # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    # `detail: object`, not `detail: str`: a failure path that formats its own
    # evidence must not be able to crash the suite. Measured 2026-11-06 -- a
    # mutant reached a check with a list and the suite died with
    # `TypeError: can only concatenate str (not "list") to str` before printing
    # any verdict, which the mutation driver correctly scores as "crashed",
    # not "caught". An assertion that cannot report its evidence is not there.
    print(("  OK   " if cond else "  FAIL ") + label
          + (("  <- " + repr(detail)) if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


class _Sig:
    """The one attribute `declared_scope` reads off a spec."""

    def __init__(self, signature: str) -> None:
        self.effect_signature = signature


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="m18_sig_")
    try:
        # ---------------------------------------------------------------- table
        print("-- the vocabulary the generator teaches")
        # Before the fix every one of these answered "" (unreadable) or, worse,
        # `pure`; both are wrong and both are cheap to get wrong again.
        translations = [
            ("reads:filesystem", "read_only"),
            ("reads:file", "read_only"),
            ("reads:a file the operator names", "read_only"),
            ("read", "read_only"),
            ("reads:", "read_only"),
            ("reads : spaced", "read_only"),
            ("filesystem_read", "read_only"),
            ("writes:filesystem", "local_write"),
            ("writes:file", "local_write"),
            ("write", "local_write"),
            ("reads:file, writes:file", "local_write"),
            ("network", "network"),
            ("net", "network"),
            ("http", "network"),
            ("network_egress", "network"),
            ("reads:file, network", "network"),
            ("reads:the operator's config  ;  network", "network"),
            ("subprocess", "system"),
            ("exec", "system"),
            ("code", "system"),
            ("pure", "pure"),
            ("PURE", "pure"),
            ("privileged", "privileged"),
            # The delete vocabulary. `deletes:` is not taught by the generator
            # prompt, so this half is the translator accepting a spelling the
            # table was extended to answer -- and answering it narrowly.
            ("deletes:file", "destructive"),
            ("delete:the old log", "destructive"),
            ("filesystem_delete", "destructive"),
            ("reads:file, deletes:file", "destructive"),
            ("deletes:file, writes:file", "privileged"),
            ("deletes:file, network", "privileged"),
        ]
        bad = []
        for sig, want in translations:
            got = scope_of_signature(sig)
            if got != want:
                bad.append(f"{sig!r} -> {got!r}, wanted {want!r}")
        check("the taught syntax translates to the scope it describes", bad == [], "; ".join(bad))

        # A signature that names a read is `read_only`, never `pure`.
        #
        # `pure` claims the tool touches nothing. `_scope_gate` and
        # `scope_creep` treat the two differently, so answering `pure` for a
        # file-reading tool hands it a contract it cannot keep -- and it did,
        # until the first version of this translator was measured.
        check("reading is not the same claim as touching nothing",
              scope_of_signature("reads:file") != "pure",
              repr(scope_of_signature("reads:file")))
        check("and the pure claim still exists for a tool that deserves it",
              scope_of_signature("pure") == "pure" and scope_of_signature("") == "")

        # The object of `reads:` is prose. Rejecting a signature because it
        # contains a space is how this field ended up empty everywhere.
        check("a prose object after the colon does not make it unreadable",
              scope_of_signature("reads:a file the operator names") == "read_only",
              repr(scope_of_signature("reads:a file the operator names")))

        # ----------------------------------------------------------- fail closed
        print()
        print("-- what it must refuse to guess")
        for sig in ["something-else", "foo:bar", "read_files", "I touch files", "purely"]:
            check(f"an unreadable signature ({sig!r}) stays unreadable",
                  scope_of_signature(sig) == "",
                  repr(scope_of_signature(sig)))
        check("and `declared_scope` turns unreadable into all-capabilities",
              declared_scope(_Sig("something-else")) == "undeclared",
              repr(declared_scope(_Sig("something-else"))))
        check("while a readable signature survives it",
              declared_scope(_Sig("reads:file")) == "read_only",
              repr(declared_scope(_Sig("reads:file"))))

        # ------------------------------------------------------- both halves
        print()
        print("-- the gate and the contract read it the same way")
        # `confirm._normalise_scope` used to be a second, separate copy of the
        # seven-word rule. Two copies of a rule are two rules; they are now one
        # function, and this asserts the join rather than assuming it.
        for sig in ["", "pure", "reads:file", "writes:file", "network",
                    "subprocess", "privileged", "something-else", "reads:a file"]:
            a, b = scope_of_signature(sig), _normalise_scope(sig)
            want_b = a or "undeclared"
            if b != want_b:
                check(f"the two readers agree on {sig!r}", False, f"{a!r} vs {b!r}")
                break
        else:
            check("the two readers agree on every spelling tried", True)

        # --------------------------------------------------- what it is for
        print()
        print("-- the critic can finally see a tool that tells the truth")
        for sig in ["pure", "reads:file", "reads:filesystem",
                    "reads:a file the operator names", "filesystem_read"]:
            missing = refuses("critic", sig)
            check(f"critic allows {sig!r}", missing == (), str(list(missing)))
        # ...and still cannot see one that does not.
        for sig, why in [("writes:file", "filesystem_write"),
                         ("network", "network_egress"),
                         ("subprocess", "subprocess"),
                         ("privileged", "filesystem_delete"),
                         ("", "filesystem_write")]:
            missing = refuses("critic", sig)
            check(f"critic still refuses {sig!r}", why in missing, str(list(missing)))

        # A deleter is refused by capability, and the capability named is the
        # one it declared -- not `filesystem_write`, `network_egress` and five
        # others it never asked for.
        deleting = refuses("critic", "deletes:file")
        check("a deleter is refused for deleting, and only for deleting",
              deleting == ("filesystem_delete",), str(list(deleting)))
        check("which is narrower than what a privileged claim earns",
              len(refuses("critic", "privileged")) > len(deleting),
              str(list(refuses("critic", "privileged"))))

        check("the critic really has a ceiling (so those refusals mean something)",
              ceiling("critic") is not None)
        check("and a role with no ceiling is still unrestricted",
              ceiling("worker") is None)
        check("the brief a critic would be given is not empty",
              len(role_brief("critic")) > 100)

        # ------------------------------------------------- end to end, one tool
        print()
        print("-- end to end: a read-only tool under a critic's registry")
        reg = ToolRegistry()
        spec = ToolSpec(
            name="reader_probe",
            description="reads one file and reports its size",
            fn=lambda **kw: 1,
            code="def main(**kw):\n    return 1\n",
            parameters={"path": {"type": "string"}},
            state=ToolState.ACTIVE,
            effect_signature="reads:the file the operator names",
        )
        reg.register(spec)
        check("the signature translated onto a real spec",
              declared_scope(spec) == "read_only", repr(declared_scope(spec)))

        scoped = reg.scoped((), role="critic", deny=ceiling("critic"))
        check("the scoped registry exists (it is not the base registry)",
              scoped is not reg)
        # `schemas()` returns the OpenAI wire shape, not a flat record: the name
        # lives under `function`. Reading `s["name"]` off it gets `None` from
        # every entry -- which is the kind of "empty answer passes the check"
        # this whole file exists to avoid. Measured 2026-11-04: that is exactly
        # what the first version of this assertion did.
        def _shown(registry) -> list:
            return [(s.get("function") or {}).get("name") for s in registry.schemas()]

        check("its schema list shows the reader",
              _shown(scoped) == ["reader_probe"], str(_shown(scoped)))
        check("and its call gate lets it through",
              scoped._refusal("reader_probe") == "", scoped._refusal("reader_probe"))

        # The same registry, same role, a tool that writes.
        reg2 = ToolRegistry()
        writer = ToolSpec(
            name="writer_probe",
            description="writes one file",
            fn=lambda **kw: 1,
            code="def main(**kw):\n    return 1\n",
            parameters={"path": {"type": "string"}},
            state=ToolState.ACTIVE,
            effect_signature="writes:file",
        )
        reg2.register(writer)
        scoped2 = reg2.scoped((), role="critic", deny=ceiling("critic"))
        refusal = scoped2._refusal("writer_probe")
        check("a writer is still refused, by capability and not by name",
              "filesystem_write" in refusal, refusal)
        check("and it is not visible either",
              _shown(scoped2) == [], str(_shown(scoped2)))

        print()
        print("TOTAL FAILURES:", len(FAILS), FAILS)
        return 0 if not FAILS else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# `SCOPE_ALLOWANCES` is imported for the assertion below rather than above it, so
# that a reader sees the table next to the claim about it.
def _table_covers_every_scope_the_translator_can_answer() -> list[str]:
    """Every scope `scope_of_signature` may return must exist in the table.

    A translator that answers a name the gate has never heard of would move the
    failure from "the gate refuses everything" to "the gate lets everything
    through", which is the worse of the two.
    """
    answers = {scope_of_signature(s) for s in
               ["pure", "reads:file", "writes:file", "network", "subprocess",
                "privileged", "deletes:file", ""]} - {""}
    return sorted(a for a in answers if a not in SCOPE_ALLOWANCES)


if __name__ == "__main__":
    missing = _table_covers_every_scope_the_translator_can_answer()
    print(("  OK   " if not missing else "  FAIL ")
          + "every scope it can answer is a scope the gate knows"
          + (("  <- " + str(missing)) if missing else ""))
    if missing:
        FAILS.append("scope table coverage")
    raise SystemExit(main())
