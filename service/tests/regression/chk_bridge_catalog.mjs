// Does the bridge really report its catalog? The one link python cannot reach.
//
// Under test: mountTools() in lib/index.js, executed as shipped.
// A python test can prove the service stores a catalog; only this can prove the
// bridge builds one and sends it. A mock client would be my own copy of the
// thing under test -- the failure mode this project keeps finding.
//
// Run from this directory:
//   node chk_bridge_catalog.mjs
//
// What it deliberately does NOT do: touch the production database. It points
// the service at a throwaway home, so a broken report cannot leave a wrong
// catalog behind for the next session to trust.
import { register } from 'node:module'
import { mkdtempSync, rmSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { spawnSync } from 'node:child_process'

const HERE = dirname(fileURLToPath(import.meta.url))
const PLUGIN_LIB = join(HERE, '..', '..', '..', 'lib', 'index.js')
const SERVICE = join(HERE, '..', '..')
// Which interpreter? Probed, not assumed: a bare `python` on Windows is a
// Store stub that prints a banner and exits 9009, so it is tried only last.
// The answer is the absolute path the interpreter reports for itself, because
// the assertions below ask existsSync(PY) -- a command name is not a file.
// Candidates are (command, args) pairs, not bare names. On Windows `python` and
// `python3` are Store stubs that print a banner and exit 9009, while the only
// working entry point is the `py` launcher -- which needs `-3` because a bare
// `py` may select python 2. Tried in this order: the explicit override, then the
// launcher, then the names that work on other platforms.
//
// 2026-11-11: the first version listed only env/python3/python, and on this
// machine all three failed -- so both suites exited 2 with `UNPROVABLE` before
// printing one assertion. That is a crash, and the runner records it as
// `crashed=True`, which is neither a pass nor a fail. A probe that can only
// report "unprovable" on the very platform the package is developed on is not a
// probe; it is a machine name written as a search.
const PY = (() => {
  const candidates = [
    [process.env.AUTOFORGE_PYTHON, ['-c', 'import sys; print(sys.executable)']],
    ['py', ['-3', '-c', 'import sys; print(sys.executable)']],
    ['python3', ['-c', 'import sys; print(sys.executable)']],
    ['python', ['-c', 'import sys; print(sys.executable)']],
  ]
  for (const [candidate, args] of candidates) {
    if (!candidate) continue
    const probe = spawnSync(candidate, args, { encoding: 'utf8' })
    if (probe.status === 0) {
      const found = String(probe.stdout).trim()
      if (found && existsSync(found)) return found
    }
  }
  return null
})()
if (PY === null) {
  console.log('UNPROVABLE: no python 3 interpreter found -- set AUTOFORGE_PYTHON')
  process.exit(2)
}

const fails = []
const check = (label, cond, detail = '') => {
  console.log((cond ? 'ok   ' : 'FAIL ') + label + (detail ? '  -- ' + detail : ''))
  if (!cond) fails.push(label)
}

// The plugin's peer packages come from the host at runtime, so the test must
// supply that resolution itself. Registering the hook is the difference between
// running the shipped code and re-implementing it.
register(new URL('./chk_bridge_catalog.hooks.mjs', import.meta.url))

if (!existsSync(PLUGIN_LIB)) {
  console.log('UNPROVABLE: lib/index.js not found at', PLUGIN_LIB)
  process.exit(2)
}
if (!existsSync(PY)) {
  console.log('UNPROVABLE: no python at', PY, '-- set AUTOFORGE_PYTHON')
  process.exit(2)
}

const home = mkdtempSync(join(tmpdir(), 'chkcat_home_'))
const skills = mkdtempSync(join(tmpdir(), 'chkcat_skills_'))
const runs = mkdtempSync(join(tmpdir(), 'chkcat_runs_'))

const registered = []
const warns = []
const noop = () => {}
const ctx = {
  effect(fn) { const d = fn(); return typeof d === 'function' ? d : noop },
  tools: { register(def) { registered.push(def); return noop } },
  get(name) { return name === 'systemPrompt' ? { context() { return noop } } : undefined },
  on() { return noop },
  logger: { warn: (m) => warns.push(String(m).slice(0, 240)), info: noop, debug: noop },
}

let verdict = 0
try {
  const mod = await import(pathToFileURL(PLUGIN_LIB).href)
  check('the shipped plugin imports', mod.name === 'agent-autoforge', String(mod.name))

  mod.apply(ctx, { python: PY, cwd: SERVICE, home, skills, runs, timeoutMs: 30000 })
  check('the bridge registers a non-empty tool set', registered.length > 0, String(registered.length))
  check('every registered name is an autoforge_ tool',
        registered.every((t) => String(t.name).startsWith('autoforge_')), '')
  check('no tool is registered with an empty parameter map',
        registered.every((t) => Object.keys(t.parameters || {}).length > 0),
        registered.filter((t) => Object.keys(t.parameters || {}).length === 0).map((t) => t.name).join(','))

  const dbFile = join(home, 'autoforge.db')
  let rows = -1
  let sample = ''
  let lastError = ''
  const probe = [
    'import sqlite3, sys',
    'try:',
    '    c = sqlite3.connect(sys.argv[1])',
    '    n = c.execute("SELECT COUNT(*) FROM bridge_tools").fetchone()[0]',
    '    s = c.execute("SELECT name FROM bridge_tools ORDER BY name LIMIT 2").fetchall()',
    '    print(n, "|", s)',
    'except Exception as exc:',
    '    print("-1 |", exc)',
  ].join('\n')
  // Wait for the count the bridge says it registered -- not merely for a
  // non-zero count. The old loop broke on `rows > 0` and then asserted
  // `rows === 73`, so a read that landed between the service's insert and its
  // commit was reported as "the service recorded a wrong number" when it meant
  // "the service had not finished writing". Measured 2026-11-06: one run in
  // ~25 failed that way, and 20 consecutive direct runs failed zero times,
  // which is exactly the signature of a poll that stops too early.
  //
  // `save_bridge_tools` writes wholesale inside one transaction (DELETE,
  // executemany, commit), so a reader sees either the old set or the new one.
  // The only other legal answer is "no table yet", which the probe reports as
  // -1 with the sqlite error in `sample`.
  const want = registered.length
  let sawWrongCount = ''
  for (let i = 0; i < 60; i += 1) {
    await new Promise((r) => setTimeout(r, 250))
    const out = spawnSync(PY, ['-c', probe, dbFile], { encoding: 'utf-8' })
    const line = (out.stdout || '').trim()
    if (out.stderr) lastError = String(out.stderr).trim().slice(-200)
    if (line !== '') {
      rows = Number.parseInt(line.split('|')[0].trim(), 10)
      sample = (line.split('|')[1] || '').trim()
      if (rows === want) break
      if (rows > 0) sawWrongCount = `${rows} vs ${want} ${sample}`
    }
  }
  // Two different questions, so two different assertions: did it ever settle on
  // a wrong number, and did it arrive at all.
  check('the service never settled on a count of its own',
        sawWrongCount === '',
        sawWrongCount || `polled ${want} consistently`)
  check('the service recorded exactly what the bridge registered',
        rows === want,
        rows === -1
          ? `no bridge_tools table within 15s; last=${sample}${lastError ? ' err=' + lastError : ''}`
          : `${rows} vs ${want} ${sample}${lastError ? ' err=' + lastError : ''}`)
  // `warns.length === 0` asked the wrong question. It was here to catch "the
  // catalog never reached the service", but what it actually measured was how
  // fast this machine was: `fetchBlock` emits
  // "autoforge: memory brief did not answer within 1500ms; the previous block is
  // used and the turn continues" whenever the per-assembly budget runs out, and
  // `mountBrief` says in its own words that this is the design -- "Failure is
  // not fatal, on purpose: a service that died, a database that is locked, a
  // slow python must not brick a session. The block is dropped and a warning is
  // logged once." Measured 2026-11-06: 20 consecutive direct runs, zero
  // warnings; the mutation driver, which copies the whole tree and runs suites
  // back to back, reproduced it once -- and in that run the catalog assertions
  // all passed. A blanket count turns a documented degradation into a flaky
  // failure, which is worse than no assertion: a suite that fails for reasons
  // outside its subject teaches its reader to ignore it.
  //
  // So ask the two questions separately. The first is this suite's subject and
  // must never happen. The second keeps the old check's real value -- an
  // *unexpected* warning (the service unreachable, a refused call) still fails
  // here, because nothing else is allowed to appear.
  const CATALOG_WARN = 'could not report the bridge tool catalog'
  const BUDGET_WARN = 'did not answer within'
  const catalogWarns = warns.filter((w) => w.includes(CATALOG_WARN))
  const otherWarns = warns.filter((w) => !w.includes(CATALOG_WARN) && !w.includes(BUDGET_WARN))
  check('the bridge never failed to report its catalog',
        catalogWarns.length === 0, JSON.stringify(catalogWarns))
  check('nothing was logged that the budget does not explain',
        otherWarns.length === 0, JSON.stringify(otherWarns))

  // The catalog is only useful if describe_tool can answer from it. Ask the
  // service the same question a model would, over the same wire.
  const firstName = registered.length > 0 ? registered[0].name : ''
  if (firstName !== '') {
    // `-m autoforge_svc` needs the package on the path: neither cwd nor
    // PYTHONPATH resolves it from this directory, and a silent empty reply
    // looks exactly like a refusal, which is why stderr is carried into the
    // failure text below.
    const ask = spawnSync(PY, ['-m', 'autoforge_svc'], {
      input: JSON.stringify({ id: 1, method: 'describe_tool', params: { name: firstName } }) + '\n',
      encoding: 'utf-8',
      cwd: SERVICE,
      env: { ...process.env, PYTHONPATH: SERVICE, AUTOFORGE_HOME: home, PYTHONIOENCODING: 'utf-8' },
      timeout: 30000,
    })
    const reply = (ask.stdout || '').split('\n').find((l) => l.trim() !== '') || ''
    let parsed = null
    try { parsed = JSON.parse(reply) } catch (err) { void err }
    const res = parsed && parsed.result ? parsed.result : {}
    const why = reply === ''
      ? 'no reply at all; stderr=' + String(ask.stderr || '').trim().slice(-200)
      : JSON.stringify(res).slice(0, 140)
    check('describe_tool answers for a bridge tool by name', res.ok === true && res.kind === 'bridge',
          `${firstName} -> ${why}`)
  }
} catch (err) {
  check('the test ran to completion', false, String(err && err.message ? err.message : err).slice(0, 300))
} finally {
  verdict = fails.length === 0 ? 0 : 1
  // Cleanup never decides the verdict: the service child holds the db file for
  // a moment, so a failed removal here says nothing about what was measured.
  for (const d of [home, skills, runs]) {
    try { rmSync(d, { recursive: true, force: true }) } catch (err) { void err }
  }
}

console.log()
console.log('TOTAL FAILURES:', fails.length)
process.exit(verdict)
