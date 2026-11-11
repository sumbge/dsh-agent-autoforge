// Harness that runs the REAL mountBrief source text from lib/index.js against stubs.
// Extraction is by brace matching, so what runs is the shipped characters, not a copy.
//
// 2026-11-03: this file was dead for a day and nobody noticed. Item 2 changed the
// signature to `mountBrief(ctx, client, view)` and this harness still searched for
// the literal string `function mountBrief(ctx, client) {` -- so it threw on the very
// first line and exited 1 before printing a single assertion. A regression suite that
// cannot start is indistinguishable from one that passes when the runner only reads
// the tail. Two fixes, and the second one is the point:
//   1. the signature is now parsed, and the parameters are matched by NAME against
//      the stubs, so adding a parameter fails loudly here instead of silently
//      extracting nothing;
//   2. `run` prints a line before it does anything, so "extracted 0 chars" can never
//      read as "no failures".
import fs from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

// `AUTOFORGE_LIB` exists so the assertions below can be pointed at a deliberately
// broken copy -- an assertion that has never been shown to fail is a claim about
// my confidence, not about the code.
const HERE = dirname(fileURLToPath(import.meta.url))
// tests/regression/ -> the plugin root is three levels up.
const target = process.env.AUTOFORGE_LIB
  || join(HERE, '..', '..', '..', 'lib', 'index.js')
const src = fs.readFileSync(target, 'utf8')

const sig = src.match(/function mountBrief\(([^)]*)\)\s*\{/)
if (!sig) throw new Error('mountBrief not found')
const params = sig[1].split(',').map((s) => s.trim()).filter(Boolean)
const startIdx = sig.index
let depth = 0
let endIdx = -1
for (let i = src.indexOf('{', startIdx); i < src.length; i += 1) {
  if (src[i] === '{') depth += 1
  else if (src[i] === '}') {
    depth -= 1
    if (depth === 0) { endIdx = i + 1; break }
  }
}
const body = src.slice(startIdx, endIdx)
console.log('mountBrief params:', params.join(', '))
console.log('extracted chars:', body.length, 'lines:', body.split('\n').length)

const checks = []
const check = (name, ok, detail) => {
  checks.push({ name, ok, detail })
  console.log((ok ? 'ok  ' : 'FAIL') + ' ' + name + (detail ? '  -- ' + detail : ''))
}

check('mountBrief parsed to a non-empty body', body.length > 2000, String(body.length))
check('mountBrief takes the shared exposure view', params.includes('view'), params.join(','))

// ---- stubs ------------------------------------------------------------
let briefCalls = 0
let capsCalls = 0
let exposureCalls = 0
const seenMethods = []
const exposureArgs = []
let exposureAnswer = { ok: true, hot: ['autoforge_ping', 'autoforge_report'], cold: { autoforge_cpu_tune: 'cold one-liner' }, text: '2 hot, 1 cold' }
const service = {
  async call(method, args) {
    seenMethods.push(method)
    if (method === 'context_brief') {
      briefCalls += 1
      return { text: `BRIEF#${briefCalls} memory + mission here` }
    }
    if (method === 'my_capabilities') {
      capsCalls += 1
      return { text: `CAPS#${capsCalls} reach measured`, reach: {}, store: {} }
    }
    if (method === 'tool_exposure') {
      exposureCalls += 1
      exposureArgs.push(args)
      return exposureAnswer
    }
    throw new Error('unexpected method ' + method)
  },
}

const registered = []
const hooks = []
const effects = []
const warned = []
const systemPromptStub = {
  context(spec) {
    registered.push(spec)
    return () => {}
  },
}
const ctx = {
  logger: { warn: (m) => warned.push(m), info: () => {} },
  effect(fn) { const d = fn(); effects.push(d); return () => { if (typeof d === 'function') d() } },
  on(name, handler) { hooks.push({ name, handler }); return () => {} },
  get(name) { return name === 'systemPrompt' ? systemPromptStub : undefined },
}

// The view object is created by mountTools and handed in; here it is handed in
// directly, which is the same contract.
const view = { ready: false, hot: new Set(), cold: {}, at: 0 }

// ---- mount ------------------------------------------------------------
const mountBrief = new Function('return ' + body)()
const values = { ctx, client: service, view }
const callArgs = params.map((p) => {
  if (!(p in values)) throw new Error('mountBrief gained an unhandled parameter: ' + p)
  return values[p]
})
mountBrief(...callArgs)

// ---- assertions -------------------------------------------------------
check('two contexts registered', registered.length === 2, String(registered.length))
check('brief order 126', registered[0]?.order === 126, String(registered[0]?.order))
check('identity order 127', registered[1]?.order === 127, String(registered[1]?.order))
check('both interpolate:false', registered.every((r) => r.interpolate === false))
check('brief name', registered[0]?.name === 'autoforge:brief', String(registered[0]?.name))
check('identity name', registered[1]?.name === 'autoforge:identity', String(registered[1]?.name))
check('text is a sync function', typeof registered[0]?.text === 'function')

// priming runs inside ctx.effect synchronously -> fetchBlock starts async
await new Promise((r) => setTimeout(r, 30))
check('primed brief', registered[0].text().startsWith('AUTOFORGE — carried from the local service on every request.'), JSON.stringify(registered[0].text().slice(0, 60)))
check('primed caps', registered[1].text().startsWith('CAPS#'), JSON.stringify(registered[1].text().slice(0, 30)))
check('priming called the service twice, and not for exposure', briefCalls === 1 && capsCalls === 1 && exposureCalls === 0, `brief=${briefCalls} caps=${capsCalls} exposure=${exposureCalls}`)

// Two hooks now, and the count was never the contract anyway -- the names are.
// Anything that asserts "exactly one hook" turns every future listener into a
// failure of a test that has nothing to say about it.
const assembleHook = hooks.find((h) => h.name === 'system-prompt/assemble')
const claimHook = hooks.find((h) => h.name === 'agent/inbox/claimed')
check('the assemble hook is registered', Boolean(assembleHook), hooks.map((h) => h.name).join(','))
check('the claim hook is registered', Boolean(claimHook), hooks.map((h) => h.name).join(','))
if (!assembleHook) {
  console.log('TOTAL FAILURES: 1 (no assemble hook -- every later assertion would be vacuous)')
  process.exit(1)
}
const hook = assembleHook.handler

// a real assembly shape, as resolveContext would have produced it
function makeAssembly() {
  return {
    contexts: [
      { name: 'sandbox-policy', order: 110, text: 'sandbox: full access' },
      { name: 'autoforge:brief', order: 126, text: registered[0].text(), interpolate: false },
      { name: 'autoforge:identity', order: 127, text: registered[1].text(), interpolate: false },
      { name: 'harness-source', order: 200, text: 'checkout: C:/cc/DSH' },
    ],
  }
}

// ---- the promise under test: every assembly recomputes ----------------
const a1 = await hook({}, {}, async () => makeAssembly())
const b1 = a1.contexts.find((c) => c.name === 'autoforge:brief').text
const c1 = a1.contexts.find((c) => c.name === 'autoforge:identity').text
check('assemble #1 refetched the brief', briefCalls === 2, `brief calls now ${briefCalls}`)
check('assemble #1 brief text is fresh', b1.includes(`BRIEF#${briefCalls}`), b1.slice(0, 40))
check('assemble #1 caps text is fresh', c1.includes(`CAPS#${capsCalls}`), c1.slice(0, 30))
check('assemble #1 asked for the exposure', exposureCalls === 1, String(exposureCalls))

// The view is written by the exposure answer, and it is the SAME object the
// tool accessors read. `ready` is what separates "the service said nothing is
// hot" from "the service has not spoken yet" -- the difference between a
// 27% saving and stripping every schema on turn one.
check('the exposure answer reached the shared view', view.ready === true, String(view.ready))
check('and it put the hot names in the set', view.hot.has('autoforge_ping') && view.hot.has('autoforge_report'), [...view.hot].join(','))
check('and it kept the cold one-liners', view.cold.autoforge_cpu_tune === 'cold one-liner', JSON.stringify(view.cold))

const a2 = await hook({}, {}, async () => makeAssembly())
const b2 = a2.contexts.find((c) => c.name === 'autoforge:brief').text
const c2 = a2.contexts.find((c) => c.name === 'autoforge:identity').text
check('assemble #2 refetched again (not cached)', briefCalls === 3, `brief calls now ${briefCalls}`)
check('assemble #2 text differs from #1', b2 !== b1, `${b1.slice(0, 12)} vs ${b2.slice(0, 12)}`)
check('assemble #2 caps differs from #1', c2 !== c1, `${c1.slice(0, 10)} vs ${c2.slice(0, 10)}`)
check('assemble #2 asked for the exposure again', exposureCalls === 2, String(exposureCalls))

// an answer that switches everything to hot must be applied too -- the flag is
// "the service has spoken", not "the service likes us"
exposureAnswer = { ok: true, hot: ['autoforge_ping'], cold: {}, text: '1 hot' }
await hook({}, {}, async () => makeAssembly())
check('a later answer replaces the view rather than merging', view.hot.size === 1 && !view.hot.has('autoforge_report'), [...view.hot].join(','))
check('a later answer clears stale cold entries', Object.keys(view.cold).length === 0, JSON.stringify(view.cold))
exposureAnswer = { ok: true, hot: ['autoforge_ping', 'autoforge_report'], cold: { autoforge_cpu_tune: 'cold one-liner' }, text: '2 hot, 1 cold' }

// a failed exposure answer must not move the view at all
const beforeFail = { ready: view.ready, hot: [...view.hot].join(','), at: view.at }
exposureAnswer = { ok: false, error: 'boom' }
await hook({}, {}, async () => makeAssembly())
check('a failed exposure answer leaves the previous view intact', view.ready === beforeFail.ready && [...view.hot].join(',') === beforeFail.hot && view.at === beforeFail.at, JSON.stringify({ ready: view.ready, hot: [...view.hot].join(','), at: view.at }))
exposureAnswer = { ok: true, hot: ['autoforge_ping', 'autoforge_report'], cold: { autoforge_cpu_tune: 'cold one-liner' }, text: '2 hot, 1 cold' }

// untouched entries must survive
check('other entries untouched', a2.contexts.find((c) => c.name === 'sandbox-policy').text === 'sandbox: full access')
check('order preserved', a2.contexts.map((c) => c.order).join(',') === '110,126,127,200')

// ---- the turn's words ride along to the exposure call -----------------
// `tool_exposure` needs the task text, and the assembly hook cannot get it:
// `system-prompt/assemble` is handed `{ agent, scope, signal }` and nothing
// else. The words arrive earlier, on `agent/inbox/claimed` -- emitted
// synchronously by `Inbox.claim()` inside `preStep`, before `assemble()` is
// awaited. So the claim listener stores them and the assembly reads them back.
//
// Measured 2026-11-03: without this, the exposure call was `{}` on every
// assembly for a day, so the whole `protect` mechanism -- matcher, service
// support, tests -- was wired to a socket nothing plugged into.
const sessionA = { id: 'sess-A' }
const noteClaim = (message, turn, session) => claimHook.handler({ agent: { session }, message, turn })
const userMessage = (text) => ({ role: 'user', content: [{ type: 'text', text }], source: { kind: 'user' } })

check('the claim hook accepts a message without throwing', (() => {
  try { noteClaim(userMessage('compile a CUDA kernel please'), 1, sessionA); return true } catch { return false }
})())

const ctxWith = (session) => ({ scope: { session } })
const beforeTask = exposureArgs.length
await hook({}, ctxWith(sessionA), async () => makeAssembly())
const taskArgs = exposureArgs[exposureArgs.length - 1]
check('assemble asked with the turn text', exposureArgs.length === beforeTask + 1 && typeof taskArgs?.task === 'string', JSON.stringify(taskArgs))
check('and the text is what the user said', String(taskArgs?.task || '').includes('compile a CUDA kernel'), String(taskArgs?.task || '').slice(0, 60))

// A block this preset injects itself must not be mistaken for the task. It is
// the same text on every turn, so honouring it would protect the same names
// forever -- a permanent tax dressed as a hint.
noteClaim({ role: 'user', content: [{ type: 'text', text: 'autoforge_brief schema dump text' }], source: { kind: 'autoforge:brief' } }, 1, sessionA)
await hook({}, ctxWith(sessionA), async () => makeAssembly())
const afterInjected = String(exposureArgs[exposureArgs.length - 1]?.task || '')
// Both halves matter. Without the first, the assertion passes on an empty task
// -- which is exactly what the broken build sends, so it would agree with the
// bug it is meant to catch (measured: the assertion passed against the
// `{}`-sending copy until this line was added).
check('the injected block left the real task in place', afterInjected.includes('compile a CUDA kernel'), afterInjected.slice(0, 60) || '(empty)')
check('an injected autoforge block is not the task', !afterInjected.includes('autoforge_brief'), afterInjected.slice(0, 80))

// A new turn starts empty rather than inheriting the last one's words: the old
// task is finished, and carrying it would protect its tools on every later turn.
const sessionB = { id: 'sess-B' }
noteClaim(userMessage('something else entirely'), 1, sessionB)
noteClaim(userMessage('about the second turn'), 2, sessionB)
await hook({}, ctxWith(sessionB), async () => makeAssembly())
const secondTurn = String(exposureArgs[exposureArgs.length - 1]?.task || '')
check('a new turn does not inherit the previous turn text', !secondTurn.includes('something else entirely'), secondTurn.slice(0, 80))
check('and it does carry its own', secondTurn.includes('about the second turn'), secondTurn.slice(0, 80))

// Within a turn, steps accumulate: the user may speak again mid-turn, and both
// halves are about the same task.
noteClaim(userMessage('and also read the file'), 2, sessionB)
await hook({}, ctxWith(sessionB), async () => makeAssembly())
const sameTurn = String(exposureArgs[exposureArgs.length - 1]?.task || '')
check('within a turn the text accumulates', sameTurn.includes('about the second turn') && sameTurn.includes('and also read the file'), sameTurn.slice(0, 120))

// No session, or a session that never claimed anything: ask with `{}` rather
// than with a stale key from another conversation.
const beforeNoSession = exposureArgs.length
await hook({}, {}, async () => makeAssembly())
check('an assembly with no scope asks with no task', JSON.stringify(exposureArgs[exposureArgs.length - 1]) === '{}', JSON.stringify(exposureArgs[exposureArgs.length - 1]))
await hook({}, ctxWith({ id: 'never-claimed' }), async () => makeAssembly())
check('a session that said nothing asks with no task', JSON.stringify(exposureArgs[exposureArgs.length - 1]) === '{}', JSON.stringify(exposureArgs[exposureArgs.length - 1]))
check('both no-task assemblies did ask', exposureArgs.length === beforeNoSession + 2, String(exposureArgs.length - beforeNoSession))

// ---- no ours in the assembly -----------------------------------------
// The tool table is refreshed whether or not our two context blocks are in the
// assembly: a scope that suppresses `autoforge:brief` still sends every schema,
// and that is the bill this refresh exists to reduce. The other two are NOT
// fetched -- there is nothing of ours to render.
const briefBefore2 = briefCalls
const capsBefore2 = capsCalls
const exposureBefore2 = exposureCalls
const bare = await hook({}, {}, async () => ({ contexts: [{ name: 'sandbox-policy', order: 110, text: 'x' }] }))
check('no autoforge entries -> the brief is not refetched', briefCalls === briefBefore2 && capsCalls === capsBefore2, `brief ${briefCalls} caps ${capsCalls}`)
check('no autoforge entries -> the exposure is still refreshed', exposureCalls === exposureBefore2 + 1, String(exposureCalls))
check('bare assembly passed through', bare.contexts.length === 1)

// ---- degradation: one budget for the whole assembly -------------------
// The budget is "how long one assembly may wait", not "how long each call may
// wait". The exposure fetch and the brief/caps fetches are sequential, so two
// independent 1500ms timers let one wedged service hold a turn for 3000ms --
// measured 3007ms on the first run of this file, which is exactly the stall the
// bound exists to prevent. Anything under 2000ms is one budget; over 2500ms is two.
//
// The baseline is taken HERE, not from an earlier snapshot: three assemblies ran
// after `b2` (the view-replacement checks) and each legally refreshed the brief.
// Comparing against a two-assemblies-old string tests my bookkeeping, not the
// degradation path.
const briefBeforeSlow = registered[0].text()
const t0 = Date.now()
service.call = async (method) => {
  if (method === 'my_capabilities') return { text: 'CAPS-slow' }
  await new Promise((r) => setTimeout(r, 5000))
  return { text: 'too late' }
}
const a3 = await hook({}, {}, async () => makeAssembly())
const waited = Date.now() - t0
const b3 = a3.contexts.find((c) => c.name === 'autoforge:brief').text
check('a wedged service is bounded by one budget, not two', waited < 2000, `${waited}ms`)
check('turn survived with the previous block', b3 === briefBeforeSlow, `${b3.slice(0, 30)} vs ${briefBeforeSlow.slice(0, 30)}`)
check('a warning was logged once', warned.length === 1, JSON.stringify(warned[0] || '').slice(0, 90))

// ---- a dead service does not throw ------------------------------------
service.call = async () => { throw new Error('ECONNREFUSED') }
let threw = null
let a4 = null
try { a4 = await hook({}, {}, async () => makeAssembly()) } catch (e) { threw = e }
check('dead service does not throw', threw === null, threw && threw.message)
check('dead service still returns an assembly', a4 && Array.isArray(a4.contexts) && a4.contexts.length === 4)

// ---- after dispose, the hook stops working ---------------------------
effects.forEach((d) => { if (typeof d === 'function') d() })
const after = await hook({}, {}, async () => makeAssembly())
check('after dispose the hook returns the assembly untouched', after.contexts.find((c) => c.name === 'autoforge:brief').text === b3)

const failed = checks.filter((c) => !c.ok)
console.log('')
console.log('methods called:', [...new Set(seenMethods)].join(', '))
console.log(`TOTAL FAILURES: ${failed.length} of ${checks.length}`)
process.exit(failed.length ? 1 : 0)