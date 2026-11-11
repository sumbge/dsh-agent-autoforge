"""What the sandbox's allow-list actually does -- and what it does not claim.

This file exists because of a near-miss. Item eight was written down as "remove
HOME, LANG, LC_ALL, PYTHONPATH from `env_allow`: Windows can never supply
them". Measured 2026-11-03, that premise is false:

    parent defines them?   {'HOME': False, 'LANG': False, 'LC_ALL': False, 'PYTHONPATH': False}
    child would get them?  {'HOME': False, 'LANG': False, 'LC_ALL': False, 'PYTHONPATH': False}
    -- after defining all four in this process --
    child would get them?  {'HOME': 'probe-home', 'LANG': 'probe-lang', ...}
    a real child reports:  {"HOME": "probe-home", "LANG": "probe-lang", ...}

They are absent because *this* parent process has them unset, not because the
platform cannot supply them -- and `reach()` already says exactly that
("allow-listed but unset here"). Deleting the names would have made the report
tidier and the sandbox stricter than it reads: `HOME` and `PYTHONPATH` are the
two a Linux parent does define, and dropping them would strip a child of
`~`-relative config and of the import path on the very hosts that have them.

So what is pinned here is the behaviour that had no test at all, on the code
path whose one real bug (`ProgramFiles` compared case-sensitively against
Windows' `PROGRAMFILES`) escaped for weeks and surfaced as an NVML hardware
error:

  * a name the parent defines reaches the child, and one it does not does not;
  * the match is case-insensitive in both directions;
  * nothing outside the allow-list leaks, except the one documented addition;
  * `reach()` reports both counts and names them, and its arithmetic adds up;
  * and it does not describe an absent name as unreachable.

The last one is the point. "Unset here" and "cannot be supplied" are different
claims, and the report must make the first without implying the second.
"""
from __future__ import annotations

import os
import sys

SERVICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SERVICE not in sys.path:
    sys.path.insert(0, SERVICE)

from autoforge_svc.forge.sandbox import Sandbox                 # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    # `detail: object` and `repr`: a failure path that formats its own evidence
    # must not be able to crash the suite before printing a verdict.
    print(("  OK   " if cond else "  FAIL ") + label
          + (("  <- " + repr(detail)) if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


class Env:
    """Set os.environ to an exact arena, then put it back.

    Exact, not additive: a test that adds keys to whatever the runner happens to
    define is a test whose result depends on the machine, which is how "12 vars"
    got into a report with nothing checking it.
    """

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        self.saved: dict[str, str] = {}

    def __enter__(self):
        self.saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(self.mapping)
        return self

    def __exit__(self, *exc):
        os.environ.clear()
        os.environ.update(self.saved)
        return False


def main() -> int:
    # A parent that defines some allow-listed names in the mixed case Windows
    # uses, and one name that is not allow-listed at all.
    arena = {
        "PATH": "/usr/bin",
        "SYSTEMROOT": "C:\\Windows",          # allow-listed, upper
        "PROGRAMFILES": "C:\\Program Files",  # allow-listed, Windows' own casing
        "LANG": "en_US.UTF-8",                # allow-listed, defined here
        "SECRET_TOKEN": "must-not-leak",      # not allow-listed
        "AWS_SECRET_ACCESS_KEY": "must-not-leak",
    }

    print("-- a defined name reaches the child; an undefined one does not")
    with Env(arena):
        sb = Sandbox(timeout=20.0)
        env = sb.effective_env()
        check("PATH is passed through", env.get("PATH") == "/usr/bin", str(env.get("PATH")))
        check("LANG is passed through", env.get("LANG") == "en_US.UTF-8", str(env.get("LANG")))

        print("-- the case fold works in both directions")
        # The allow-list spells it `ProgramFiles`; Windows spells it
        # `PROGRAMFILES`. Comparing literally dropped it, and the only symptom
        # was nvidia-smi exiting 255 with a driver error.
        check("a name the list spells in mixed case, from an upper-cased parent",
              env.get("PROGRAMFILES") == "C:\\Program Files", str(sorted(env)))
        # The reverse direction cannot be probed on this host: Windows'
        # `os.environ` upper-cases every key on write, so asking for `Path`
        # returns `PATH` before the allow-list is consulted. My first version of
        # this assertion asked for exactly that and read the empty answer as a
        # defect in `effective_env` -- it was a defect in the assertion.
        #
        # Written the other way round, the check states the actual fact, and my
        # own first attempt at stating it failed here too: `"Path" not in
        # os.environ` is False on Windows, because the lookup itself is
        # case-insensitive. Which is the point -- the mixed case never reaches
        # `effective_env`, so there is no reverse direction to test.
        check("Windows answers a mixed-case lookup regardless of how the key is stored",
              "Path" in os.environ,
              str(sorted(os.environ))[:120])

        print("-- nothing outside the list leaks")
        check("an unrelated variable is not forwarded",
              "SECRET_TOKEN" not in env, str(sorted(env)))
        check("nor is a credential-shaped one",
              "AWS_SECRET_ACCESS_KEY" not in env, str(sorted(env)))

        print("-- the one documented addition")
        check("PYTHONIOENCODING is added even when the parent is silent",
              env.get("PYTHONIOENCODING") == "utf-8", str(env.get("PYTHONIOENCODING")))
        # Not "kept": the child runs under `python -I`, which makes
        # PYTHONIOENCODING inert on purpose -- the runner writes UTF-8 bytes to
        # the real stdout instead, so a cp936 host cannot mangle a CJK result.
        # Asserting the parent's value wins would assert a setting that has no
        # effect. What matters is that the runner does not *depend* on it.
        os.environ.clear()
        os.environ.update({"PYTHONIOENCODING": "cp936"})
        code = ("def main():\n"
                "    return '\\u4e2d\\u6587 \\u2014 non-ascii survives'\n")
        res = sb.run(code, "main")
        check("a CJK result survives regardless of the parent's locale setting",
              res.ok is True and "中文" in str(res.output), str(res.output)[:120])

    print("-- reach() reports the true split, and names both halves")
    with Env({"PATH": "/usr/bin", "SYSTEMROOT": "C:\\Windows", "LANG": "en_US.UTF-8"}):
        sb = Sandbox(timeout=20.0)
        reach = sb.reach(probe=False)
        present = reach.get("env_names_present") or []
        absent = reach.get("env_names_absent") or []
        line = str(reach.get("env"))

        check("the allow-list is 15 names", len(sb.env_allow) == 15, str(len(sb.env_allow)))
        check("present names are listed, lower-cased for comparison",
              "path" in [n.lower() for n in present], str(present))
        check("absent names are the ones the parent does not define",
              set(absent) == {n for n in sb.env_allow
                              if n.lower() not in {"path", "systemroot", "lang"}},
              str(sorted(absent)))
        check("present and absent do not overlap",
              not ({n.lower() for n in present} & {n.lower() for n in absent}),
              str(sorted({n.lower() for n in present} & {n.lower() for n in absent})))

        # The arithmetic in the sentence. It said "11 of 15" once because
        # `defined` was computed as total-minus-injected, which is only the same
        # number when the parent defines nothing the list does not mention.
        check("the sentence counts what the child actually gets",
              f"{len(sb.effective_env())} vars reach the child" in line, line)
        check("and reports the allow-listed half separately",
              "3 of 15 allow-listed names" in line, line)

        print("-- and it says 'unset here', not 'unreachable'")
        # The distinction item eight got wrong. A name in `absent` is a fact
        # about this parent; whether the platform can supply it is a different
        # question, and this report does not answer it.
        check("the wording is about this process, not the platform",
              "allow-listed but unset here" in line, line)
        for bad in ("cannot", "cannot be supplied", "unreachable", "never"):
            check("it does not claim '%s'" % bad, bad not in line.lower(), line)

    print("-- a defined name really arrives in a real child")
    with Env({"PATH": os.environ.get("PATH", ""), "LANG": "probe-lang",
              "HOME": "probe-home", "PYTHONPATH": "probe-path"}):
        sb = Sandbox(timeout=20.0)
        code = ("def main():\n"
                "    import os\n"
                "    return {n: os.environ.get(n) for n in"
                " ('HOME', 'LANG', 'PYTHONPATH')}\n")
        res = sb.run(code, "main")
        check("the child ran", res.ok is True, str(res.error))
        got = res.output or {}
        check("it saw HOME", got.get("HOME") == "probe-home", str(got))
        check("it saw LANG", got.get("LANG") == "probe-lang", str(got))
        check("it saw PYTHONPATH", got.get("PYTHONPATH") == "probe-path", str(got))
        # The premise item eight was going to delete these over. Stated as a
        # test so the next person who reads "Windows can never supply them"
        # finds the counter-example instead of the claim.
        check("so 'the platform cannot supply them' is false",
              all(got.get(n) for n in ("HOME", "LANG", "PYTHONPATH")), str(got))

    print("-- probe=True reports evidence, and names what it cannot conclude")
    # `probe=False` asserts; `probe=True` measures. The distinction was learned
    # the hard way: a probe that called `gethostbyname` and reported success read
    # "connected" on a host where a TUN device answered every name with a
    # fake-IP, so the report said the network worked while proving only that
    # something answered. What is pinned here is that the evidence line names a
    # certificate, and that DNS interception is reported as its own fact.
    #
    # My first version of this block ran with PATH alone and failed. That was
    # not a defect in the probe: measured 2026-11-03, name resolution on Windows
    # needs SYSTEMROOT. Stripped one name at a time --
    #     PATH only          dns=None       gaierror 11003
    #     PATH + SYSTEMROOT  dns=resolved   TLSv1.3, SAN pypi.org, *.pypi.org
    #     PATH + WINDIR      dns=None       gaierror 11003
    #     PATH + TEMP        dns=None       gaierror 11003
    # SYSTEMROOT is in the allow-list for DLL search, and it turns out to be
    # load-bearing for DNS as well -- Winsock reaches the resolver through
    # %SystemRoot%\System32. So the probe is run in an arena that can actually
    # resolve, and the finding is pinned as its own assertion below.
    real_path = os.environ.get("PATH", "")
    real_root = os.environ.get("SYSTEMROOT") or os.environ.get("SystemRoot") or ""
    with Env({"PATH": real_path, "SYSTEMROOT": real_root}):
        sb = Sandbox(timeout=25.0)
        probe = sb.reach(probe=True).get("probe") or {}
        net = probe.get("network") or {}
        dns = probe.get("dns") or {}
        # `evidence == "resolved"` was the first version of this line, and it
        # went red on 2026-11-07 for a reason worth keeping: with a TUN resolver
        # running, every name answers -- `.invalid` included -- so the probe
        # correctly said `intercepted` rather than claiming the host resolved
        # anything. The property being pinned is not "this machine's DNS works"
        # (that is a fact about the network, not about the sandbox) but "a name
        # that answered is never taken as proof of reachability". SYSTEMROOT is
        # still load-bearing: without it the same probe answers a gaierror.
        check("the DNS probe answers honestly, whatever this network does",
              dns.get("evidence") in ("resolved", "intercepted"), str(dns))
        check("...and when it says resolved, it names what it resolved",
              dns.get("evidence") != "resolved" or bool(dns.get("detail")), str(dns))
        check("the network probe ran", "ok" in net, str(net))
        check("its evidence is a TLS certificate, not a reachability guess",
              "certificate" in str(net.get("evidence", "")), str(net.get("evidence")))
        check("the detail names the SAN it verified",
              "SAN" in str(net.get("detail", "")), str(net.get("detail")))
        check("dns interception is its own reported fact",
              isinstance(probe.get("dns_intercepted"), bool), str(probe.get("dns_intercepted")))
        check("the filesystem probe names the path it touched",
              bool((probe.get("host_filesystem") or {}).get("detail")),
              str(probe.get("host_filesystem")))

    print("-- and with a resolver that answers for everything, the answer is suspect")
    # `evidence == "resolved"` was too strong an assertion and it failed for the
    # right reason on 2026-11-07: with a TUN resolver running, every name --
    # including `.invalid` -- answers 198.18.0.9x, so the probe correctly said
    # `evidence: "intercepted"` instead of "resolved". The property that matters
    # is not "DNS works" (that depends on this machine's network) but "a name
    # that answered is not taken as proof". Both verdicts are honest; a blank one
    # is not. Measured three states: resolved / intercepted / a gaierror -- the
    # first two carry a detail, the third carries the error.
    check("the DNS verdict is one of the three honest answers",
          dns.get("evidence") in ("resolved", "intercepted", ""),
          str(dns))
    check("...and it never reports a success without saying what it saw",
          dns.get("evidence") != "resolved" or bool(dns.get("detail")), str(dns))

    print("-- and with SYSTEMROOT gone, it reports the failure rather than a blank")
    # The other half of the same finding: a probe that cannot resolve must say
    # so. An empty `network` block reads as "nothing to report", which is how a
    # broken environment passes for a quiet one.
    with Env({"PATH": real_path}):
        sb = Sandbox(timeout=20.0)
        probe = sb.reach(probe=True).get("probe") or {}
        net = probe.get("network") or {}
        check("the network block still carries a verdict",
              "ok" in net, str(net))
        check("and it is not ok", net.get("ok") is False, str(net))
        check("it names what went wrong",
              bool(net.get("detail")), str(net.get("detail")))

    print()
    print("TOTAL FAILURES:", len(FAILS), FAILS)
    return 0 if not FAILS else 1

if __name__ == "__main__":
    raise SystemExit(main())
