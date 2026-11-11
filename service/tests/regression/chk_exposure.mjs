// Does a cold tool actually cost less, and still work?
//
// Under test: the accessors `mountTools` installs on every tool definition, and
// the `tool_exposure` answer that drives them. A python test can prove the
// service classifies a catalog; only this can prove the model-facing projection
// shrinks and that the shrunk tool still validates and still calls through.
//
// Run from this directory:
//   node chk_exposure.mjs
//
// What it deliberately does NOT do: touch the production database. It points the
// service at a throwaway home.
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

register(new URL('./chk_bridge_catalog.hooks.mjs', import.meta.url))

if (!existsSync(PLUGIN_LIB)) {
  console.log('UNPROVABLE: lib/index.js not found at', PLUGIN_LIB)
  process.exit(2)
}
if (!existsSync(PY)) {
  console.log('UNPROVABLE: no python at', PY, '-- set AUTOFORGE_PYTHON')
  process.exit(2)
}

const home = mkdtempSync(join(tmpdir(), 'chkexp_home_'))
const skills = mkdtempSync(join(tmpdir(), 'chkexp_skills_'))
const runs = mkdtempSync(join(tmpdir(), 'chkexp_runs_'))

const registered = []
const warns = []
const noop = () => {}
// The assembly hook is captured rather than run: this test drives the
// projection directly, so it can hold the view still and compare two reads.
const hooks = {}
const ctx = {
  effect(fn) { const d = fn(); return typeof d === 'function' ? d : noop },
  tools: { register(def) { registered.push(def); return noop } },
  get(name) { return name === 'systemPrompt' ? { context() { return noop } } : undefined },
  on(event, fn) { hooks[event] = fn; return noop },
  logger: { warn: (m) => warns.push(String(m).slice(0, 240)), info: noop, debug: noop },
}

//: What a provider would be sent for one registered tool, in characters. This
//: is the number the bill is computed from, so it is the number the test uses.
const cost = (def) => def.name.length
  + String(def.description || '').length
  + JSON.stringify(def.parameters || {}).length

let verdict = 0
try {
  const mod = await import(pathToFileURL(PLUGIN_LIB).href)
  check('the shipped plugin imports', mod.name === 'agent-autoforge', String(mod.name))

  mod.apply(ctx, { python: PY, cwd: SERVICE, home, skills, runs, timeoutMs: 30000 })
  // Not `=== 66`. The count is not the contract; a catalog that registers
  // something is. Pinning the number here made adding one tool
  // (`autoforge_anchor_ledger`, measured 2026-11-03) fail a test about schema
  // projection -- an accusation in the wrong file entirely. The catalog's
  // identity is checked where it belongs, against what the service recorded
  // (chk_bridge_catalog.mjs compares registered vs stored).
  check('the shipped plugin registers a non-empty tool set', registered.length > 0,
    String(registered.length))
  check('the assembly hook was registered', typeof hooks['system-prompt/assemble'] === 'function')

  // Before any exposure answer: every tool carries its full schema. This is the
  // expensive-and-safe default, and it must be the default -- a wrongly cold
  // tool is invisible in exactly the way this whole exercise exists to prevent.
  //
  // "Cold" is measured by what the description says, not by whether the tool has
  // parameters: 16 of the 66 bridge tools genuinely take none, so an empty
  // parameter map is their full form. The first version of this check used
  // `properties.length > 0` and accused all 16 of being cold.
  const looksCold = (d) => String(d.description || '').includes('Not loaded in full:')
  const coldFirst = registered.filter(looksCold).map((d) => d.name)
  check('before any answer, nothing is described as cold', coldFirst.length === 0, coldFirst.join(','))

  const name = 'autoforge_tool_define'
  const target = registered.find((d) => d.name === name)
  check('the sampled tool is registered', Boolean(target), name)
  // Snapshot the whole table BEFORE the hook runs. Reconstructing the old cost
  // afterwards is not possible: the accessors answer from the live view, so
  // once the projection lands the original description and schema are gone.
  const before = new Map(registered.map((d) => [d.name, cost(d)]))
  const fullCost = before.get(name)
  const fullDesc = target.description
  const fullProps = Object.keys(target.parameters.properties || {}).length
  check('the sampled tool really has parameters', fullProps > 0, String(fullProps))

  // Drive the real hook with a real service answer. `tool_exposure` is asked
  // over the same wire the plugin uses; the hook applies it to the accessors.
  const assembly = {
    contexts: [
      { name: 'autoforge:brief', text: 'x' },
      { name: 'autoforge:identity', text: 'y' },
    ],
  }
  const next = async () => assembly
  await hooks['system-prompt/assemble'](assembly, {}, next)

  const coldCost = cost(target)
  check('the sampled tool got cheaper', coldCost < fullCost, `${fullCost} -> ${coldCost}`)
  check('its parameters were reduced to the empty stub',
        Object.keys(target.parameters.properties || {}).length === 0,
        JSON.stringify(target.parameters))
  check('its description now says where the schema is',
        String(target.description).includes(`autoforge_describe_tool('${name}')`),
        String(target.description).slice(0, 160))
  check('its description still names its arguments',
        String(target.description).includes('Params:'), String(target.description).slice(0, 120))

  // The whole table, measured the way a provider measures it.
  let beforeTotal = 0
  let afterTotal = 0
  for (const d of registered) {
    beforeTotal += before.get(d.name) || 0
    afterTotal += cost(d)
  }
  check('the table as a whole shrank', afterTotal < beforeTotal, `${beforeTotal} -> ${afterTotal}`)
  console.log(`     table: ${beforeTotal} chars -> ${afterTotal} chars (${(100 * (beforeTotal - afterTotal) / beforeTotal).toFixed(1)}% saved)`)

  // A cold tool must still be callable, and must still validate against its FULL
  // argument list. That is the whole bargain: the projection changes what is
  // advertised, never what is accepted.
  let threw = null
  try {
    await target.execute({})
  } catch (err) {
    threw = err
  }
  check('a cold tool still rejects missing required arguments', threw !== null,
        threw === null ? 'execute({}) returned instead of throwing' : '')
  check('and the rejection names a real missing argument',
        threw !== null && /name/.test(String(threw.message || threw)),
        threw === null ? 'no error was raised' : String(threw.message).slice(0, 160))

  const ask = spawnSync(PY, ['-m', 'autoforge_svc'], {
    input: JSON.stringify({ id: 1, method: 'tool_list', params: {} }) + '\n',
    encoding: 'utf-8',
    cwd: SERVICE,
    env: { ...process.env, PYTHONPATH: SERVICE, AUTOFORGE_HOME: home, PYTHONIOENCODING: 'utf-8' },
    timeout: 30000,
  })
  const line = (ask.stdout || '').split('\n').find((l) => l.trim() !== '') || ''
  check('the service is still answering over the wire', line.includes('"ok":true') || line.includes('"ok": true'),
        line.slice(0, 120) + (ask.stderr ? ' err=' + String(ask.stderr).slice(-160) : ''))
  check('the refresh produced no warning', warns.length === 0, JSON.stringify(warns))
} catch (err) {
  check('the test ran to completion', false, String(err && err.message ? err.message : err).slice(0, 300))
} finally {
  verdict = fails.length === 0 ? 0 : 1
  for (const d of [home, skills, runs]) {
    try { rmSync(d, { recursive: true, force: true }) } catch (err) { void err }
  }
}

console.log()
console.log('TOTAL FAILURES:', fails.length)
process.exit(verdict)
