
/**
 * Agent-preset bundle that mounts the autoforge service as model-facing tools.
 *
 * The service (autoforge-svc) is a long-lived Python child process speaking
 * line-delimited JSON-RPC over stdio. This plugin owns that process, exposes
 * each RPC method as a DSH tool, and recycles the process when it dies.
 */
import { spawn, spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'
import { join } from 'node:path'
import {
  ToolArgsError,
  defineTool,
  parameterSchemaSpecToJsonSchema,
  validateArgs,
} from '@deepseek-ai/dsh-tools'

export const name = 'agent-autoforge'
export const inject = ['tools']

//: What a cold tool's parameter map is reduced to: valid, supported, and 33
//: characters where the real one averages 179. Shared by every cold tool rather
//: than rebuilt per tool per read.
const EMPTY_SCHEMA = Object.freeze({ type: 'object', properties: {} })

/** The compiled JSON Schema for one descriptor, or the empty one. */
function paramSchema(descriptor) {
  try {
    return parameterSchemaSpecToJsonSchema(descriptor || {})
  } catch (err) {
    // A descriptor the compiler refuses is a bug in this file, not a reason to
    // take the preset down: report it once through the caller's judgement and
    // fall back to the honest empty map, which accepts anything.
    return { type: 'object', properties: {} }
  }
}

// Where the service lives, when the mount says nothing.
//
// The previous table named one machine's user directory on three of its four
// paths and `'python'` on the fourth. Nothing broke because the shipped preset
// sets all of them -- which is exactly why it survived: a default that is
// always overridden is a default nobody has had to read. Mounted against a
// checkout with no preset it would spawn a missing python in a missing
// directory, and the failure would name a path belonging to somebody else.
//
// So the fallback is derived instead of remembered:
//   cwd    the `service/` directory beside this file -- the one thing a bundle
//          can know about its own layout
//   home   per-user and OUTSIDE the checkout, because a database written into
//          `node_modules` is deleted by the next install
//   skills the user-level skills directory, for the same reason
//   python a candidate list, because bare `python` on Windows is the Microsoft
//          Store stub: it prints a banner, exits 9009, and looks like a service
//          that will not start
// No absolute interpreter from this machine is in the list. A path that works
// only where it was written is the thing this block exists to stop shipping;
// `py` is the Windows launcher, which is the only entry point that exists on a
// machine where `python` and `python3` are the Store stubs (they print a banner
// and exit 9009). Measured 2026-11-11: with only env/python3/python the list had
// zero working candidates here.
const PYTHON_CANDIDATES = [
  process.env.AUTOFORGE_PYTHON,
  'py',
  'python3',
  'python',
].filter((p, i, all) => typeof p === 'string' && p.trim() !== '' && all.indexOf(p) === i)

// `$LOCALAPPDATA/autoforge/data` on Windows, `~/.autoforge/data` elsewhere --
// the same two places the service's own `_home()` would pick, so a mount that
// says nothing lands where the CLI would have put it and the two see one
// library instead of two.
const USER_BASE = process.platform === 'win32'
  ? join(process.env.LOCALAPPDATA || join(homedir(), 'AppData', 'Local'), 'autoforge')
  : join(homedir(), '.autoforge')

const DEFAULTS = {
  // Kept as an array: `start()` resolves it through `pickPython()` the first
  // time it spawns, so a candidate that cannot run is skipped rather than
  // reported as a broken service.
  python: PYTHON_CANDIDATES,
  cwd: fileURLToPath(new URL('../service/', import.meta.url)).replace(/[\\/]+$/, ''),
  home: join(USER_BASE, 'data'),
  skills: join(USER_BASE, 'skills'),
  // Where the interrupted-run journal lives. Empty means "let the service
  // decide", which is `$AUTOFORGE_HOME/runs` -- correct for a mount that keeps
  // its own data, wrong for one that should see the CLI's runs. Passing it
  // here is the whole fix: `runs.journal_dir()` already prefers
  // `$AUTOFORGE_RUNS`, so no service code changes when this moves.
  runs: '',
  timeoutMs: 120000,
}

/** The first python that actually runs, or the first candidate as a last word. */
function pickPython(options) {
  const list = Array.isArray(options.python)
    ? options.python
    : [options.python]
  const tried = []
  for (const candidate of list) {
    if (typeof candidate !== 'string' || candidate.trim() === '') continue
    tried.push(candidate)
    try {
      const probe = spawnSync(candidate, ['-c', 'print(1)'], {
        timeout: 15000,
        windowsHide: true,
        encoding: 'utf8',
      })
      // 9009 is what the Store stub returns; a 0 with no output is also not a
      // python. Both are "this candidate did not answer".
      if (probe.status === 0 && String(probe.stdout || '').trim() === '1') return candidate
    } catch (err) {
      // ENOENT and friends: try the next one.
    }
  }
  // Nothing answered. Say so with the list, rather than handing `spawn` a
  // value that will fail with a message about a file nobody chose.
  throw new Error(
    `autoforge: no usable python interpreter. Tried: ${tried.join(', ') || '(none)'}. `
    + 'Set `python` in the mount, or set $AUTOFORGE_PYTHON.',
  )
}


/** One JSON-RPC-over-stdio client, resilient to a dead child. */
class AutoforgeClient {
  constructor(options) {
    this.options = options
    this.child = null
    this.pending = new Map()
    this.nextId = 1
    this.buffer = ''
    this.lastError = null
  }

  start() {
    if (this.child && this.child.exitCode === null) return
    // Resolved once and remembered: probing candidates costs a process per
    // attempt, and a service that respawns every few minutes should not pay it
    // again. A later manual `python` in the mount still wins -- it is a string,
    // and `pickPython` accepts one.
    if (!this.python) this.python = pickPython(this.options)
    const child = spawn(this.python, ['-m', 'autoforge_svc'], {
      cwd: this.options.cwd,
      // AUTOFORGE_SKILLS_DIRS 必须显式给：服务端默认在 $AUTOFORGE_HOME/skills 下
      // 找技能，而 home 指向的是数据目录，那里没有技能库。不给这个变量，
      // `skills` 会返回空数组 —— 一个看起来像"你没有技能"的空答案。
      env: {
        ...process.env,
        AUTOFORGE_HOME: this.options.home,
        AUTOFORGE_SKILLS_DIRS: this.options.skills,
        // Only when configured: an empty value would make `journal_dir()`
        // take an empty string for a path, which is worse than unset.
        ...(this.options.runs ? { AUTOFORGE_RUNS: this.options.runs } : {}),
        PYTHONIOENCODING: 'utf-8',
        PYTHONUNBUFFERED: '1',
      },
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    })
    this.child = child
    this.buffer = ''
    child.stdout.setEncoding('utf8')
    child.stdout.on('data', (chunk) => this._onData(chunk))
    child.stderr.setEncoding('utf8')
    child.stderr.on('data', (chunk) => { this.lastError = String(chunk).slice(-2000) })
    const fail = (why) => {
      for (const [, entry] of this.pending) entry.reject(new Error(`autoforge service died: ${why}`))
      this.pending.clear()
      if (this.child === child) this.child = null
    }
    child.on('exit', (code) => fail(`exit ${code}`))
    child.on('error', (err) => fail(err && err.message ? err.message : String(err)))
  }

  /** Whether the child is still running (a dead child is restarted lazily). */
  alive() {
    return Boolean(this.child && this.child.exitCode === null)
  }

  _onData(chunk) {
    this.buffer += chunk
    let index
    while ((index = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, index).trim()
      this.buffer = this.buffer.slice(index + 1)
      if (!line) continue
      let message
      try { message = JSON.parse(line) } catch { continue }
      const entry = this.pending.get(message.id)
      if (!entry) continue
      this.pending.delete(message.id)
      clearTimeout(entry.timer)
      if (message.ok) entry.resolve(message.result)
      else entry.reject(new Error(message.error || 'autoforge service returned an error'))
    }
  }

  call(method, params) {
    this.start()
    const id = this.nextId++
    const child = this.child
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`autoforge ${method} timed out after ${this.options.timeoutMs}ms`))
      }, this.options.timeoutMs)
      this.pending.set(id, { resolve, reject, timer })
      try {
        child.stdin.write(JSON.stringify({ id, method, params: params || {} }) + '\n')
      } catch (err) {
        clearTimeout(timer)
        this.pending.delete(id)
        reject(err)
      }
    })
  }

  stop() {
    const child = this.child
    this.child = null
    if (child && child.exitCode === null) { try { child.kill() } catch {} }
  }
}

/** Text renderer for any JSON-serializable result. */
function asText(value) {
  if (value === undefined || value === null) return '(no result)'
  if (typeof value === 'string') return value
  return JSON.stringify(value, null, 2)
}

/**
 * A tool whose description and parameter schema are re-read on every request.
 *
 * `defineTool()` cannot be used for this, and that is the whole reason this
 * function exists: it compiles `parameterSchemaSpecToJsonSchema(parameters)` once
 * at definition time and validates against that frozen copy forever, so a tool
 * built that way can never shrink. DSH's own projection does the opposite -- it
 * destructures the definition on every request (`const { name, description,
 * parameters } = definition`), so accessor properties on that object are read
 * fresh each turn. Measured 2026-11-02: flipping a flag on a live definition
 * changed one tool's contribution from its full description and schema to a
 * one-line summary and `{"type":"object","properties":{}}` on the very next read.
 *
 * Validation stays on the FULL descriptor. A tool that has been reduced to a
 * name and a line must still reject a missing required argument, and a model
 * that guessed the arguments correctly must not be punished for the description
 * having been short: the projection changes what is advertised, never what is
 * accepted.
 *
 * `coerce` may either rewrite `params` in place (the JSON-field helpers) or
 * return a new object (the tool_define renamer). Ignoring the return value
 * silently dropped every rename: measured 2026-10-25, `sampleCall` reached
 * Python unchanged and raised "unexpected keyword argument", with the coerce
 * function correct and never used. So: take the return value when there is one,
 * otherwise keep what the in-place writer put in `params`.
 *
 * A `{ok:false}` answer is a failure, not data. It used to be returned as text,
 * which DSH recorded as completed and the model read as "empty output" rather
 * than "here is why it failed"; the reason is thrown instead so DSH records
 * status=error and the model reads the original words.
 */
function dynamicTool(client, spec, view) {
  const full = spec.parameters || {}
  const fullSchema = paramSchema(full)
  return {
    name: spec.toolName,
    get description() {
      if (!view.ready || view.hot.has(spec.toolName)) return spec.description || ''
      const line = view.cold[spec.toolName]
      return typeof line === 'string' && line !== '' ? line : (spec.description || '')
    },
    get parameters() {
      return (!view.ready || view.hot.has(spec.toolName)) ? fullSchema : EMPTY_SCHEMA
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args) {
      const violations = validateArgs(full, args || {})
      if (violations.length > 0) throw new ToolArgsError(violations)
      let params = { ...(args || {}) }
      if (spec.coerce) {
        const out = spec.coerce(params)
        if (out && typeof out === 'object') params = out
      }
      const result = await client.call(spec.method, params)
      const reason = failureReason(result)
      if (reason) throw new Error(`${spec.toolName}: ${reason}`)
      return asText(result)
    },
  }
}

/**
 * Why a service answer counts as a failure, or '' when it does not.
 *
 * The service is not uniform: some methods answer `{ok:false, text}`, some
 * `{ok:false, output, error}`, some `{error:'not found'}` with no `ok` at all.
 * All three mean the same thing to the model, so all three are read here --
 * three different spellings of one fact is exactly the inconsistency this
 * function exists to erase. A bare `ok:false` with no message still fails,
 * and says so, because silence about a failure is the bug being fixed.
 */
function failureReason(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return ''
  if (value.ok !== false && value.error == null) return ''
  const detail = value.error ?? value.text ?? value.output ?? value.message
  const text = typeof detail === 'string' ? detail : detail === undefined ? '' : JSON.stringify(detail)
  if (value.ok === false && !text) return 'failed with no reason given'
  return text
}

/** Parse a JSON-text argument into the object the service expects. */
function jsonField(field) {
  return (params) => {
    // A parameter the schema does not declare is not necessarily absent from
    // the call: the model may still send it. Coercing `undefined` used to do
    // nothing, so a JSON string in an undeclared field reached the service as
    // a string and the RPC failed on a type the schema never mentioned.
    const raw = params[field]
    if (typeof raw !== 'string') return
    const trimmed = raw.trim()
    if (trimmed === '') { delete params[field]; return }
    try { params[field] = JSON.parse(trimmed) }
    catch (err) { throw new Error(`${field} must be JSON: ${err.message}`) }
  }
}

export function apply(ctx, config = {}) {
  const options = { ...DEFAULTS, ...config }
  // 同一个服务库只允许一个 stdio 子进程。插件被挂两次（比如 home 层一份、
  // 预设里又一份）时，两个实例各起一个 python 子进程、绑同一个 sqlite，
  // 并发写会把库弄坏。这里按库路径去重：后来的实例复用在跑的那一个。
  const shared = (globalThis.__autoforgeClients ??= new Map())
  const clientKey = `${options.cwd}\u0000${options.home}`
  if (shared.has(clientKey) && shared.get(clientKey).alive()) {
    (ctx.logger?.warn ?? ((m) => console.warn(m)))(`autoforge: reusing the running service for ${options.home}; this plugin is mounted more than once`)
    const existing = shared.get(clientKey)
    ctx.effect(() => () => {})  // 后来的实例不负责关它
    return mountTools(ctx, existing)
  }
  const client = new AutoforgeClient(options)
  shared.set(clientKey, client)
  ctx.effect(() => () => {
    client.stop()
    if (shared.get(clientKey) === client) shared.delete(clientKey)
  })
  mountTools(ctx, client)
}

function mountTools(ctx, client) {
  // Every tool this file registers, collected as it is registered, so the
  // service can be told what the model can actually call. The service's own
  // `describe_tool` used to answer only for *forged* tools, and would reply
  // "no tool named ..." for the 60 bridge tools that are not forged -- while
  // the manifest text told the model to call it "for its full parameter schema
  // first". Measured 2026-11-02: `autoforge_describe_tool('autoforge_run_python')`
  // answered with the six forged names and nothing else.
  const catalog = []
  // Which tools ride in full this turn. `ready` is the whole of the default
  // policy and it exists because the obvious default is wrong: an empty `hot`
  // set means "every tool is cold", so the accessors below would strip all 66
  // schemas on the very first request, before any service answer had been
  // heard. Measured 2026-11-02: without this flag the table was 14,237 chars
  // before the first assembly and 19,802 after -- 39% LARGER, because 66 tools
  // each traded a compiled schema for a sentence. Until an answer arrives,
  // nothing is projected.
  const view = { ready: false, hot: new Set(), cold: {}, at: 0 }
  const register = (spec) => ctx.effect(() => {
    const definition = dynamicTool(client, spec, view)
    // The compiled schema is what the model is actually sent each turn; the raw
    // descriptor is not. Measured 2026-11-02: all 66 descriptors together are
    // 132 characters while their compiled schemas are 11,823. Reporting only
    // the descriptor made a cost comparison that should have said "keep this
    // tool hot" say "cold is cheaper" -- a 90x underestimate of the thing being
    // saved.
    catalog.push({
      name: spec.toolName,
      description: spec.description || '',
      parameters: spec.parameters || {},
      schema: definition.parameters || {},
    })
    const dispose = ctx.tools.register(definition)
    return () => { if (typeof dispose === 'function') dispose() }
  })

  register({
    toolName: 'autoforge_ping',
    method: 'ping',
    description: 'Check that the autoforge service is reachable and report which database it uses.',
  })

  register({
    toolName: 'autoforge_remember',
    method: 'remember',
    description: 'Keep a fact across sessions in the operator\'s on-disk autoforge memory. Use it for anything you would otherwise re-derive next time: a path, a preference, a lesson from a failure.',
    parameters: {
      key: { type: 'string', required: true, description: 'short handle to recall it by' },
      value: { type: 'string', required: true, description: 'the fact itself' },
      tags: { type: 'string', description: 'comma-separated labels' },
    },
  })

  register({
    toolName: 'autoforge_recall',
    method: 'recall',
    description: 'Read back facts kept with autoforge_remember, including from previous sessions. An empty query returns everything.',
    parameters: {
      query: { type: 'string', description: 'substring to match' },
      limit: { type: 'integer', description: 'max results' },
    },
  })

  register({
    toolName: 'autoforge_forget',
    method: 'forget',
    description: 'Drop one kept fact by key.',
    parameters: { key: { type: 'string', required: true, description: 'the key to drop' } },
  })

  register({
    toolName: 'autoforge_mission_list',
    method: 'mission_list',
    description: 'Read what is owed: open missions and their sub-missions, or the closed ones as history. Call it instead of trusting that you remember the whole list.',
    parameters: { status: { type: 'string', enum: ['open', 'done', 'dropped', 'all'], description: 'which missions to list' } },
  })

  register({
    toolName: 'autoforge_mission_open',
    method: 'mission_open',
    description: 'Record something you owe, so it stops being lost when the next request arrives. A request from the operator is a mission before it is a task -- open it first, then work.',
    parameters: {
      text: { type: 'string', required: true, description: 'what is owed, one sentence' },
      parent: { type: 'integer', description: 'mission id this belongs under, 0 for a new root' },
      next_step: { type: 'string', description: 'the very next action' },
      blocked_on: { type: 'string', description: 'what it is waiting for, if anything' },
    },
  })

  register({
    toolName: 'autoforge_mission_note',
    method: 'mission_note',
    description: 'Update one mission: a note on where it got to, and/or its next step. Use it when the work moves, so the mission describes now rather than when it was opened.',
    parameters: {
      mid: { type: 'integer', required: true, description: 'mission id, e.g. 3 for M3' },
      note: { type: 'string', description: 'what happened' },
      next_step: { type: 'string', description: 'the next action; omit to leave it as it is, or pass an empty string to clear it' },
      blocked_on: { type: 'string', description: 'what it waits on; omit to leave it as it is, or pass an empty string to clear it' },
    },
  })

  register({
    toolName: 'autoforge_mission_close',
    method: 'mission_close',
    description: 'End a mission, with a note saying what happened. ok=false drops it instead. The note is what lets a later run tell a finished mission from an abandoned one.',
    parameters: {
      mid: { type: 'integer', required: true, description: 'mission id' },
      note: { type: 'string', description: 'what happened' },
      ok: { type: 'boolean', description: 'true=done, false=dropped' },
    },
  })

  register({
    toolName: 'autoforge_mission_show',
    method: 'mission_show',
    description: 'One mission in full: its line, its sub-missions, and its history of openings, notes and closings.',
    parameters: { mid: { type: 'integer', required: true, description: 'mission id' } },
  })

  register({
    toolName: 'autoforge_mission_focus',
    method: 'mission_focus',
    description: 'Mark which open mission this session is on. Pass 0 to clear it.',
    parameters: { mid: { type: 'integer', required: true, description: 'mission id, 0 to clear' } },
  })

  register({
    toolName: 'autoforge_mission_block',
    method: 'mission_block',
    description: 'Record what a mission is waiting for -- a download, an authorisation, an answer. A blocked mission is exempt from the 24h presumed-complete rule, because waiting on someone else is not being forgotten.',
    parameters: {
      mid: { type: 'integer', required: true, description: 'mission id' },
      waiting_on: { type: 'string', required: true, description: 'what it waits for' },
    },
  })

  register({
    toolName: 'autoforge_mission_sweep',
    method: 'mission_sweep',
    description: 'Apply the house rule now: a mission with no activity for 24 hours is presumed complete. Presumed is not closed -- the report says which are assumptions.',
    parameters: { idle_hours: { type: 'number', description: 'idle threshold, default 24' } },
  })

  register({
    toolName: 'autoforge_mission_wake',
    method: 'mission_wake',
    description: 'Bring back a mission that was presumed complete from silence. Only an assumption can be overturned; a real close stays closed and you get the reason why.',
    parameters: {
      mid: { type: 'integer', required: true, description: 'mission id' },
      note: { type: 'string', description: 'why it is still owed' },
    },
  })


  register({
    toolName: 'autoforge_my_history',
    method: 'my_history',
    description: 'Read the on-disk ledger: every forge, tool call, mission change and run, newest last. This is a record, not a recollection.',
    parameters: {
      limit: { type: 'integer', description: 'max events, default 20' },
      kind: { type: 'string', description: 'filter by event kind' },
    },
  })

  register({
    toolName: 'autoforge_verify_ledger',
    method: 'verify_ledger',
    description: 'Check the event log for rewriting: every row carries a hash of itself and a pointer to the row before it.',
  })

  register({
    toolName: 'autoforge_anchor_ledger',
    method: 'anchor_ledger',
    description: 'Freeze the ledger as of now: record the current head hash, the row count and a digest of the rows written before chaining began. An anchor is a declared boundary, not a repair — it is what lets a later verify_ledger tell a rewritten row from one that was merely written earlier.',
    parameters: {
      note: { type: 'string', description: 'why this boundary is being declared' },
    },
  })





  register({
    toolName: 'autoforge_run_python',
    method: 'run_python',
    description: 'Run a Python snippet in a child process on this host and get back what it printed and what it returned. This is the ordinary way to read a file, run a command, or work out a number — no shell tool needed. Define `def main(**kwargs)`; its return value arrives as output, and anything it prints arrives as stdout. cwd is a fresh empty temp dir per call; a file an earlier call wrote must be named by absolute path.',
    parameters: {
      code: { type: 'string', required: true, description: 'the module body; define def main(**kwargs)' },
      entry: { type: 'string', description: 'function to call, default main' },
      args: { type: 'string', description: 'JSON object of keyword arguments passed to entry' },
    },
  })


  register({
    toolName: 'autoforge_skill_write',
    method: 'skill_write',
    description: 'Write a procedure down as a skill so the next session can load it. Use it for how-to knowledge: steps, order, pitfalls. Use autoforge_remember for facts and this for procedures.',
    parameters: {
      name: { type: 'string', required: true, description: 'short lower-case handle, e.g. deploy-verify' },
      description: { type: 'string', required: true, description: 'what the procedure does (one line)' },
      body: { type: 'string', required: true, description: 'the procedure itself, markdown' },
      when_to_use: { type: 'string', description: 'the trigger — when should future-you reach for this?' },
      tags: { type: 'string', description: 'comma-separated labels' },
    },
  })

  register({
    toolName: 'autoforge_schedule_add',
    method: 'schedule_add',
    description: 'Put something on the agenda: a thing to do later, or to do repeatedly. `when` takes +90s/+10m/+2h/+1d, a clock time, or an ISO date; `repeat` takes an interval like 10m.',
    parameters: {
      text: { type: 'string', required: true, description: 'what to do' },
      when: { type: 'string', required: true, description: 'due time, e.g. +2m or 2026-10-25T09:00:00' },
      repeat: { type: 'string', description: 'repeat interval, e.g. 10m; omit for one-shot' },
    },
  })

  // Interrupted runs. autoforge's edge over an ordinary chat loop is that a
  // killed process does not kill the work: every run keeps a journal file on
  // disk and a later session reads it back. Without these the mode could not
  // see a single stranded run -- measured 2026-10-30: 8 resumable journals on
  // disk under %LOCALAPPDATA%\autoforge\runs, 0 visible from here.
  //
  // The journal path is NOT derived in this file. `runs.journal_dir()` already
  // resolves $AUTOFORGE_RUNS before $AUTOFORGE_HOME/runs, so the mount passes
  // `runs` (one config field) and the service's reader needs no edit.
  register({
    toolName: 'autoforge_runs_status',
    method: 'runs_status',
    description: 'List every interrupted run recorded on disk: which are resumable, how many turns in, how long stranded, and what the next step would be. Read-only. Call it when work you remember doing is missing, or before claiming a task was never started.',
  })

  register({
    toolName: 'autoforge_resume_run',
    method: 'resume_run',
    description: 'Read an interrupted run and get the prompt that continues it. This does NOT re-execute anything: it returns resumed=false, the recorded transcript size, and the resume prompt, which you then act on yourself -- the process that resumed it is you, not the service. With no run_id it takes the run stranded longest.',
    parameters: {
      run_id: { type: 'string', description: 'a run id or its unique prefix; omit for the oldest resumable one' },
    },
  })

  register({
    toolName: 'autoforge_runs_prune',
    method: 'runs_prune',
    description: 'Delete run records past their TTL, and report what was kept and what the filesystem refused to release. Nothing else is deleted: a record that is merely stale -- resume budget spent, transcript unrunnable -- is left for runs_status to report, because deleting it would erase the evidence that a task was attempted and failed. Returns counts plus the journal directory; runs_forget is the way to act on one record without touching the rest.',
    parameters: {
      ttl_days: { type: 'number', description: 'age in days past which a record is deleted (default 7)' },
    },
  })

  register({
    toolName: 'autoforge_runs_forget',
    method: 'runs_forget',
    description: 'Archive one run record so it is never resumed again. For work abandoned rather than finished: reaching a result is what finishes a run, and this is the other thing. The file moves to <journal>/archived/ and stays readable, so the difference between "done" and "dropped" survives. The reason is required because it is the only thing that tells a later reader the two apart.',
    parameters: {
      run_id: { type: 'string', required: true, description: 'the run id, or a unique prefix' },
      why: { type: 'string', required: true, description: 'why this run is not wanted' },
    },
  })

  register({
    toolName: 'autoforge_skill_port',
    method: 'skill_port',
    description: 'Copy another agent\'s procedures into this library, verbatim, with a provenance line. source=tars copies the 9 vendored desktop-GUI skills; source=openclaw copies an installed openclaw package\'s skills. Measured 2026-10-30: no openclaw package is installed on this host, so that source returns an error naming OPENCLAW_SKILLS_DIR rather than "0 skills ported" -- the two are different facts.',
    parameters: {
      source: { type: 'string', description: "tars (default) or openclaw" },
      overwrite_body: { type: 'boolean', description: 'replace an existing body (default false: a re-port must not silently overwrite an edited procedure)' },
      include_other_os: { type: 'boolean', description: 'openclaw only: keep macOS/Linux-only skills instead of skipping them' },
    },
  })

  register({
    toolName: 'autoforge_tars_probe',
    method: 'tars_probe',
    description: 'The code ported from Intelligence Indeed (tars), and whether it still works on THIS host: what was copied verbatim, what was deliberately not, the byte-for-byte check against the copy-time digests, and one read-only pass over the vendored tree, the store, the skill directories, the market and DNS. Read-only, so it is safe before any decision.',
  })

  register({
    toolName: 'autoforge_tars_skills',
    method: 'tars_skills',
    description: 'The procedures borrowed from Intelligence Indeed: with no argument, the index (name, what it is for, what it is called in my own library); with a name, that procedure in full. Progressively disclosed on purpose -- nine bodies in the prompt on every turn is the tax this split avoids.',
    parameters: {
      name: { type: 'string', description: "a borrowed procedure, e.g. skill-chrome (the tars- prefix is accepted too); omit for the index" },
    },
  })

  register({
    toolName: 'autoforge_tars_port_skills',
    method: 'tars_port_skills',
    description: 'Copy the borrowed tars procedures into my own skill library as tars-<name>, bodies unedited and carrying their licence and origin in the file itself. This one writes nine files -- which is why it is a separate tool from tars_skills, not an argument to it.',
    parameters: {
      overwrite_body: { type: 'boolean', description: 'replace an existing body (default false: a re-port must not silently overwrite an edited procedure)' },
    },
  })

  register({
    toolName: 'autoforge_schedule_list',
    method: 'schedule_list',
    description: 'Show the agenda: what is overdue, what is coming, and whether anything can wake the agent at all.',
  })

  register({
    toolName: 'autoforge_schedule_tick',
    method: 'schedule_tick',
    description: 'Read the agenda for anything due right now. This is what a wake-up looks like from the agent side; call it at the start of a run.',
  })

  register({
    toolName: 'autoforge_schedule_run_due',
    method: 'schedule_run_due',
    description: 'Attend to whatever the schedule says is due, then return. The difference from schedule_tick is the whole point: tick reports what is due and leaves the running to you, run_due actually runs it. Nothing due means no model call and no output -- deliberately, because a scheduler that announces "nothing happened" every time teaches whoever reads the log to stop reading it. One task raising does not stop the ones behind it.',
    parameters: {
      budget_s: { type: 'number', description: 'wall-clock ceiling for the whole run, not per task (default 600)' },
    },
  })

  register({
    toolName: 'autoforge_schedule_done',
    method: 'schedule_done',
    description: 'Close a due task with a note about what happened. The note is the point: it is the only thing that distinguishes a repeated failure from a long silence.',
    parameters: {
      task_id: { type: 'string', required: true, description: 'the id schedule_list showed' },
      note: { type: 'string', description: 'what happened' },
      ok: { type: 'boolean', description: 'false if it failed' },
    },
  })

  register({
    toolName: 'autoforge_schedule_cancel',
    method: 'schedule_cancel',
    description: 'Remove a scheduled task without running it.',
    parameters: {
      task_id: { type: 'string', required: true, description: 'the id schedule_list showed' },
    },
  })

  register({
    toolName: 'autoforge_skill_view',
    method: 'skill_view',
    description: 'Read one stored procedure in full. Facts go in memory, how-to goes in a skill.',
    parameters: { name: { type: 'string', required: true, description: 'skill name' } },
  })

  register({
    toolName: 'autoforge_tool_list',
    method: 'tool_list',
    description: 'List the tools defined through this service, with their signatures.',
  })

  register({
    toolName: 'autoforge_describe_tool',
    method: 'describe_tool',
    // The persona told the model to call `describe_tool(name)` for a tool's
    // schema and record. Nothing by that name existed anywhere on the DSH path,
    // so the instruction pointed at `autoforge_describe`, which answers a
    // different question. This is that name, answered.
    description: 'One tool in full: the schema it declares, how it has actually behaved (its own counters next to what the ledger saw), and its version history oldest-first. Use it before redefining or evolving a tool, and when a call fails and you need to see what the tool promised.',
    parameters: {
      name: { type: 'string', required: true, description: 'Name of a forged tool, as autoforge_list_tools reports it.' },
    },
  })

  register({
    toolName: 'autoforge_tool_define',
    method: 'tool_define',
    // The last three arguments are what a *forged* tool gets from the pipeline
    // and a *hand-written* one used to lack entirely. Without them the battery
    // reports two honest UNPROBEDs and a tool is sealed ACTIVE on "it runs":
    //
    //   execution  UNPROBED: no valid sample call was declared, so correctness
    //              was never examined
    //   trigger    UNPROBED: no positive probe was declared, so nothing asked
    //              this tool to fire
    //
    // Measured 2026-10-25: the same tool scored 2/5 bare and 6/6 with these
    // three declared. Undeclared they still *worked*, because `execute`
    // spreads every argument through to the service -- which is the worse
    // state, not the better one: a caller that reads this schema can never
    // discover that the parameters exist. A tool that only works when you
    // already know the trick is not a tool.
    description: 'Define a durable tool from Python source. The service runs one sample call in a sandbox before keeping it, and a tool whose sample call fails is NOT registered. Pass sampleCall/sampleExpect/probes as well, or the verification battery can never judge the tool: execution and trigger report UNPROBED and the tool is kept on the strength of "it runs".',
    parameters: {
      name: { type: 'string', required: true, description: 'tool name' },
      code: { type: 'string', required: true, description: 'Python source defining the function named after the tool (or main), returning a JSON-serializable value.' },
      description: { type: 'string', description: 'what the tool does' },
      parameters: { type: 'json', description: 'Optional JSON object: the argument schema.' },
      sampleCall: { type: 'json', description: 'A real, valid call: the arguments a genuine use would pass, e.g. {"n": 16}. Required for the execution check to examine correctness at all.' },
      sampleExpect: { type: 'string', description: 'What that call must answer, as a substring of the output, e.g. "True". Compared as text, so "true" does not match a Python True.' },
      probes: { type: 'json', description: 'Optional JSON array of trigger probes: [{"query": "how many words are in this", "expect": "call", "negativeQuery": "what time is it"}]. Without a positive probe the trigger check reports UNPROBED.' },
      // This parameter exists to be refused. There was a short-lived version
      // that acted on it, and that was the wrong shape: it created a second
      // documented overwrite path, which made the library's own rule -- one
      // overwrite path, `evolve_tool` -- false, and the standard would have had
      // to be rewritten to fit the code. Keeping the parameter while making it
      // inert is deliberate: a caller who believes overwriting is available
      // gets told so in a sentence about `evolve_tool`, instead of silently
      // getting the old behaviour or an unexplained "already exists".
      replace: { type: 'boolean', description: 'Accepted for compatibility and always refused: no value of it overwrites anything. A taken name is a refusal here, whatever you pass. The one overwrite path is autoforge_evolve_tool, which breeds a new body and verifies it against the incumbent before retiring the old one.' },
    },
    coerce: (params) => {
      const out = { ...params }
      // `probes` keeps its name, so it is handled below and NOT in this rename
      // loop. It sat in the loop at first, where `out[to] = out[from]` was
      // immediately followed by `delete out[from]` on the same key: the probe
      // list was assigned and then deleted, and every declared probe vanished
      // in silence. Measured 2026-10-25 -- trigger went back to UNPROBED with
      // the parameter present and correct-looking at the call site.
      //
      // `sampleCall` is declared `type: 'json'`, so a caller may hand it either
      // an object or the JSON text of one. Only `parameters` and `probes` were
      // ever parsed here, so the string form of `sampleCall` was renamed and
      // forwarded untouched; the service then ran `dict(sample_call)` on a
      // str and raised
      //
      //   ValueError: dictionary update sequence element #0 has length 1;
      //               2 is required
      //
      // which names neither the field nor the real problem -- a `dict()` of a
      // one-character first element, from a call whose JSON was perfectly
      // valid. Measured 2026-11-01: three attempts to declare a sample call as
      // JSON text all failed on that message, and the workaround was to omit
      // sample_call entirely, which leaves the execution check UNPROBED.
      // Whatever is declared `json` gets parsed; the schema is the contract.
      for (const field of ['sampleCall', 'parameters', 'probes']) {
        if (typeof out[field] !== 'string') continue
        const text = out[field].trim()
        if (text === '') { delete out[field]; continue }
        try { out[field] = JSON.parse(text) }
        catch (err) { throw new Error(`${field} must be JSON: ${err.message}`) }
      }
      for (const [from, to] of [['sampleCall', 'sample_call'],
                                ['sampleExpect', 'sample_expect']]) {
        if (out[from] === undefined) continue
        out[to] = out[from]
        delete out[from]
      }
      // The service's probe dicts use snake_case keys; accept both spellings
      // so a caller who read the service's source is not punished for it.
      if (Array.isArray(out.probes)) {
        out.probes = out.probes.map((p) => {
          if (!p || typeof p !== 'object') return p
          const neg = p.negativeQuery ?? p.negative_query
          return { query: p.query, expect: p.expect ?? 'call',
                   ...(neg ? { negative_query: neg } : {}) }
        })
      }
      return out
    },
  })

  register({
    toolName: 'autoforge_review_tool',
    method: 'review_tool',
    // The reviewer the operator asked for, and the check that was missing:
    // `evaluate_tool` answers "does it work when called", which it can only
    // answer by running the tool. A tool that returns the right number while
    // claiming to touch nothing is a verified liar, and the battery has no
    // vocabulary for that. This reads the spec alone -- signature readable,
    // code inside the declared scope, code present to review -- and it cannot
    // promote, demote or quarantine, because a reviewer that also seals its
    // own findings has become the thing being reviewed.
    description: 'Ask a reviewer what a tool\'s own claim says about its own code, without running it. Answers three questions from the spec alone: can the declared effect_signature be read at all, does the code exercise an effect the declared scope does not permit, and is there any code to review. Reports objections by name (signature, contract, readable) and never changes the tool\'s state -- evaluate_tool owns promotion, rehab_tool owns the way back.',
    parameters: {
      name: { type: 'string', required: true,
              description: 'name of a tool in the library, as tool_list reports it' },
    },
  })

  register({
    toolName: 'autoforge_rehab_tool',
    method: 'rehab_tool',
    // Promised by the persona in a form nothing could call: it once said
    // "`rehab` 能把它放回 `probation`", and the registry method existed while no
    // service method wrapped it and this bridge declared nothing. A quarantine
    // that nothing can lift is a delete with extra steps. The persona now names
    // `rehab_tool`; `chk_seams.py` sweeps every backticked name it uses, so the
    // bare form would fail there rather than here.
    description: 'Put a quarantined or retired tool back on trial: its counters are cleared and it returns to probation, where it is callable again but not yet sealed. The record of why it was quarantined stays in the ledger. Pass require_clean to say how many clean calls it must make before it can earn ACTIVE again.',
    parameters: {
      name: { type: 'string', required: true, description: 'tool name' },
      require_clean: { type: 'number', description: 'clean calls wanted before it can be sealed again (default 2)' },
    },
  })

  register({
    toolName: 'autoforge_tool_call',
    method: 'tool_call',
    description: "Run one of YOUR forged tools by name: the tools `autoforge_forge_tool` created and `autoforge_list_tools` lists. This is not a way to reach the service's own methods -- `describe`, `list_tools`, `skills` and the rest each already have a tool of their own, and calling one here returns `unknown tool`. Check `autoforge_list_tools` first; a retired or quarantined tool is refused unless you pass force=true, which runs it anyway for this one call.",
    parameters: {
      name: { type: 'string', required: true, description: 'Name of a forged tool, as autoforge_list_tools reports it.' },
      arguments: { type: 'json', description: 'Arguments for the tool, as a JSON object matching its schema.' },
      force: { type: 'boolean', description: 'Run it even though it is quarantined or retired (default false). The call still counts against its record.' },
    },
    coerce: jsonField('arguments'),
  })

  register({
    toolName: 'autoforge_report',
    method: 'report',
    description: 'What this service holds right now: counts of facts, missions, skills and tools.',
  })


  // -- forge and skill plumbing, ported from agent.py's own tool specs -----
  // Each of these was a method on the service and had no way in from DSH.
  // The descriptions are agent.py's, because the wording is what the model
  // reads when it decides whether to call: a rewritten description is a
  // different tool wearing the same name.

  register({
    toolName: 'autoforge_describe',
    method: 'describe',
    description: 'List the service methods this process answers and the database it is bound to.',
  })

  register({
    toolName: 'autoforge_list_tools',
    method: 'list_tools',
    description: 'List the tools on your library with their lifecycle state and health.',
  })

  register({
    toolName: 'autoforge_retire_tool',
    method: 'retire_tool',
    description: 'Retire a tool you judge harmful or superseded. Requires a rationale. Builtin meta-tools cannot be retired.',
    parameters: {
      name: { type: 'string', required: true, description: 'tool name' },
      rationale: { type: 'string', required: true, description: 'why it is being taken out of service' },
    },
  })

  register({
    toolName: 'autoforge_forge_tool',
    method: 'forge_tool',
    description: 'Forge a new tool for a recurring need you cannot currently serve. The Python runs as a real subprocess on this host, is verified, and is kept as a named tool. For a one-off, call autoforge_run_python instead.',
    parameters: {
      need: { type: 'string', required: true, description: 'one-line description of the need' },
      context: { type: 'string', description: 'why the need arose; helps the generator' },
    },
  })

  register({
    toolName: 'autoforge_evaluate_tool',
    method: 'evaluate_tool',
    description: 'Run the full verification battery (execution, robustness, adversarial, trigger, negative, and the claim gate: is the returned number a fact about the world or about the data it was handed) on a tool. This is also the promotion path: a tool on probation that passes earns ACTIVE, and an active tool that fails drops back to PROBATION (still callable, still visible). Pass promote=false to read the verdict without moving the state.',
    parameters: {
      name: { type: 'string', required: true, description: 'tool name' },
      promote: { type: 'boolean', description: 'move the tool to ACTIVE on a pass and to PROBATION on a failure (default true); false only reports the verdict' },
    },
  })

  register({
    toolName: 'autoforge_evolve_tool',
    method: 'evolve_tool',
    description: 'Breed a better version of a tool that underperforms or fails.',
    parameters: {
      name: { type: 'string', required: true, description: 'tool name' },
      failure_report: { type: 'string', description: 'what went wrong; defaults to the recorded failure' },
    },
  })

  register({
    toolName: 'autoforge_find_gaps',
    method: 'find_gaps',
    description: 'Proactively analyse your library and fill missing capabilities.',
  })

  register({
    toolName: 'autoforge_tool_usage',
    method: 'tool_usage',
    description: 'When each tool was last called, read back from the event ledger.',
  })

  register({
    toolName: 'autoforge_mission_history',
    method: 'mission_history',
    description: 'The full history of one mission: when it was opened, every note, and how it closed.',
    parameters: {
      mid: { type: 'integer', required: true, description: 'mission id, e.g. 3 for M3' },
      limit: { type: 'integer', description: 'max notes (default 50)' },
    },
  })

  register({
    toolName: 'autoforge_skill_errors',
    method: 'skill_errors',
    description: 'Why a skill file is being skipped rather than offered.',
  })

  register({
    toolName: 'autoforge_skill_forget',
    method: 'skill_forget',
    description: 'Retire a skill: archived on disk, dropped from the menu.',
    parameters: {
      name: { type: 'string', required: true, description: 'skill handle' },
    },
  })

  register({
    toolName: 'autoforge_events',
    method: 'events',
    description: 'The raw event ledger, newest first.',
    parameters: {
      limit: { type: 'integer', description: 'how many rows (default 20)' },
    },
  })

  register({
    toolName: 'autoforge_log_event',
    method: 'log_event',
    description: 'Append one event to the ledger yourself, so a decision of yours is recorded with the tool runs.',
    parameters: {
      kind: { type: 'string', required: true, description: 'event kind' },
      payload: { type: 'json', description: 'JSON object of what happened' },
    },
    coerce: jsonField('payload'),
  })

  register({
    toolName: 'autoforge_system_task_command',
    method: 'system_task_command',
    description: 'Print the command that would register a recurring wake-up with the operating system, without running it.',
    parameters: {
      interval_minutes: { type: 'integer', description: 'minutes between runs (default 15)' },
      name: { type: 'string', description: 'task name' },
    },
  })

  register({
    toolName: 'autoforge_install_system_task',
    method: 'install_system_task',
    description: 'Register a recurring wake-up with the operating system, so you run even when nobody starts you.',
    parameters: {
      interval_minutes: { type: 'integer', description: 'minutes between runs (default 15)' },
      name: { type: 'string', description: 'task name' },
    },
  })

  // ── the parts DSH has no equivalent for ──────────────────────────────────
  //
  // These are the autoforge tools whose job is not covered by anything on the
  // DSH side: a measured self-report, CUDA kernel work, native C kernel work,
  // and reaching a human. Everything else in autoforge either has a DSH
  // counterpart (browser, vision, MCP, subagents, session resume) or belongs to
  // an autonomy model this harness does not share.

  register({
    toolName: 'autoforge_my_capabilities',
    method: 'my_capabilities',
    description: 'Report what you can actually reach, measured rather than declared: whether forged code touches the whole host filesystem and the network, what the sandbox bounds, and where your memory lives. Pass probe=true for a live write/read + DNS round-trip instead of taking it on faith. Trust this over your priors — an agent that reasons about its reach from its tool list gets it wrong.',
    parameters: {
      probe: { type: 'boolean', description: 'run a live write/read + DNS round-trip to measure reach' },
    },
  })

  register({
    toolName: 'autoforge_gpu_probe',
    method: 'gpu_probe',
    description: 'Report what CUDA silicon is reachable: device name, compute capability, memory, driver. Never fails; "none" is an answer.',
    parameters: {},
  })

  register({
    toolName: 'autoforge_gpu_occupancy',
    method: 'gpu_occupancy',
    description: 'Estimate occupancy for a launch geometry from the architecture\'s static limits. Labelled as an estimate — measured occupancy needs a device.',
    parameters: {
      block: { type: 'integer', description: 'threads per block' },
      grid: { type: 'integer', description: 'blocks per grid (x)' },
      shared_bytes: { type: 'integer' },
      regs_per_thread: { type: 'integer' },
      arch_major: { type: 'integer' },
      arch_minor: { type: 'integer' },
    },
  })

  register({
    toolName: 'autoforge_gpu_units_audit',
    method: 'gpu_units_audit',
    description: 'Lint source for the do_bench-returns-ms bug: a value scaled by 1000 in a statement labelled ms. Give source text or a path.',
    parameters: {
      source: { type: 'string' },
      path: { type: 'string', description: 'file or directory' },
    },
  })

  register({
    toolName: 'autoforge_gpu_verify',
    method: 'gpu_verify',
    description: 'Compare kernel output against a reference and report max absolute difference. Refuses without a reference — otherwise it would only restate the kernel\'s own output.',
    parameters: {
      kind: { type: 'string', required: true, description: 'label for the report' },
      actual: { type: 'string', required: true, description: 'numbers, space or comma separated' },
      reference: { type: 'string', description: 'numbers, space or comma separated' },
      tolerance: { type: 'number' },
    },
  })

  register({
    toolName: 'autoforge_gpu_compile',
    method: 'gpu_compile',
    description: 'Compile CUDA C you wrote to a cubin, content-addressed so a repeat is a cache hit. Needs the CUDA toolkit on this host; it says so rather than pretending.',
    parameters: {
      code: { type: 'string', required: true },
      name: { type: 'string' },
      arch_major: { type: 'integer' },
      arch_minor: { type: 'integer' },
    },
  })

  register({
    toolName: 'autoforge_gpu_bench',
    method: 'gpu_bench',
    description: 'Compile and time a kernel under a wall-clock budget, reporting the spread and whether the difference is signal or noise.',
    parameters: {
      code: { type: 'string', required: true },
      name: { type: 'string' },
      block: { type: 'integer' },
      grid: { type: 'integer' },
      shared_bytes: { type: 'integer' },
      reps: { type: 'integer' },
      time_budget_s: { type: 'number' },
      arch_major: { type: 'integer' },
      arch_minor: { type: 'integer' },
    },
  })

  register({
    toolName: 'autoforge_cpu_probe',
    method: 'cpu_probe',
    description: 'Report what native silicon and toolchain is reachable: CPU name, cores, SIMD features, cache, compiler and version, and what -march=native resolves to. Never fails; "none" is an answer.',
    parameters: {},
  })

  register({
    toolName: 'autoforge_cpu_runs_here',
    method: 'cpu_runs_here',
    description: 'Whether an artefact built for a given -march can actually execute on this CPU. A build this machine cannot run fails as SIGILL in your own process, so ask before compiling.',
    parameters: {
      target: { type: 'string', description: 'e.g. native, x86-64-v2, skylake' },
    },
  })

  register({
    toolName: 'autoforge_cpu_preflight',
    method: 'cpu_preflight',
    description: 'Lint C source before it is compiled: flags this CPU cannot run, missing symbols, obvious defects. Touches nothing.',
    parameters: {
      code: { type: 'string', required: true },
      name: { type: 'string' },
    },
  })

  register({
    toolName: 'autoforge_cpu_units_audit',
    method: 'cpu_units_audit',
    description: 'Lint source for the do_bench-returns-ms bug: a value scaled by 1000 in a statement labelled ms. Give source text or a path.',
    parameters: {
      source: { type: 'string' },
      path: { type: 'string' },
    },
  })

  register({
    toolName: 'autoforge_cpu_cache_stats',
    method: 'cpu_cache_stats',
    description: 'What the CPU kernel cache holds and how often it hits. A repeat of the same source, target, compiler and flags is a hit rather than a rebuild.',
    parameters: {},
  })

  register({
    toolName: 'autoforge_cpu_compile',
    method: 'cpu_compile',
    description: 'Compile C you wrote to a shared library, content-addressed. Refuses a target this machine cannot run, because that failure is SIGILL rather than a message you can read.',
    parameters: {
      code: { type: 'string', required: true },
      name: { type: 'string' },
      target: { type: 'string', description: 'e.g. native, x86-64-v2' },
    },
  })

  register({
    toolName: 'autoforge_cpu_run_isolated',
    method: 'cpu_run_isolated',
    description: 'Compile, then verify and time a C kernel in a child process. Isolation is the point: an unproven kernel\'s first call can segfault or hang, and that must cost a child, not the session.',
    parameters: {
      code: { type: 'string', required: true },
      name: { type: 'string' },
      kind: { type: 'string', description: 'problem kind, e.g. saxpy, matmul, softmax' },
      size: { type: 'integer' },
      n: { type: 'integer' },
      dtype: { type: 'string' },
      seed: { type: 'integer' },
      reps: { type: 'integer' },
      warmup: { type: 'integer' },
      timeout: { type: 'number' },
      target: { type: 'string' },
    },
  })

  register({
    toolName: 'autoforge_cpu_tune',
    method: 'cpu_tune',
    description: 'Search candidate kernels for one that beats the naive baseline and verifies. Reports the winner, the baseline and why the search stopped.',
    parameters: {
      kind: { type: 'string', description: 'problem kind, e.g. saxpy, matmul, softmax' },
      size: { type: 'integer' },
      n: { type: 'integer' },
      dtype: { type: 'string' },
      seed: { type: 'integer' },
      max_candidates: { type: 'integer' },
      budget_s: { type: 'number' },
      reps: { type: 'integer' },
      warmup: { type: 'integer' },
    },
  })

  register({
    toolName: 'autoforge_notify_channels',
    method: 'notify_channels',
    description: 'Which notification channels are configured, and which are broken. Read-only: it says how to reach a person, not that a message is owed.',
    parameters: {},
  })

  register({
    toolName: 'autoforge_notify_send',
    method: 'notify_send',
    description: 'Send a message to the operator through the configured channels. Use it when something needs a person before the turn ends.',
    parameters: {
      text: { type: 'string', required: true },
      channel: { type: 'string', description: 'one channel by name; omit for all' },
    },
  })

  // ── the read-only remainder ─────────────────────────────────────────────
  //
  // agent.py exposes 79 builtin tools; this bundle registered 49 of them. The
  // gap was not a decision about what belongs in DSH -- it was the read-only
  // meta tools and the self-report, which cost nothing to mount and without
  // which the preset is a smaller agent wearing autoforge's name.
  //
  // The methods below exist on the service already (`describe` lists them);
  // that is the test each row here had to pass before being added. A row for a
  // method the service does not answer would register a tool that can only
  // fail, and a tool that can only fail is worse than an absent one because
  // the model will call it and then believe the failure.



  register({
    toolName: 'autoforge_skill_list',
    method: 'skills',
    description: 'List my skills -- written-down procedures -- with when to use each and how often I have loaded it. An empty query lists all. `query` matches name, description, when_to_use and tags, case-insensitive; no match returns an empty list. Ordered most-used first. The same files are visible to the native `skill` tool through the skill-filesystem row this preset configures.',
    parameters: { query: { type: 'string', description: 'substring to filter by, matched against name/description/when_to_use/tags (optional)' } },
  })





  // ── the tools that reach a human or a peer, gated on the same freedom ──
  //
  // agent.py labels both `may_access_network`: sending a message is egress and
  // nothing else. They were already on the service; only the mount was missing.



  // ── deliberately absent ─────────────────────────────────────────────────
  //
  // Four agent.py tools have no row here, and this is the record of why, so a
  // later pass does not read their absence as an oversight:
  //
  //   resume_run / runs_forget   DSH owns the session and its own resume; a
  //                              second run record written by a python child
  //                              would be a second source of truth about which
  //                              task is in flight.
  //   browser_*                  DSH ships a browser tool. Mounting autoforge's
  //                              CDP client too would give the model two ways
  //                              to drive one browser.
  //   see / draw / animate       DSH has all three; autoforge's would be a
  //                              second endpoint billing the same account.
  //   spawn_agent / design_team  DSH has subagent + subagent_fork + workflow-ptc,
  //                              already configured below.
  //   mcp_servers / mcp_connect / mcp_call
  //                              DSH has an MCP client layer and a manager GUI;
  //                              autoforge-svc has no MCP config of its own, so
  //                              these methods do not exist on the service at
  //                              all (verified against `describe`).
  //   skill_port / tars_*        registered below. Both exist on this service
  //                              (`skill_port` since the port, the three `tars_*`
  //                              since 2026-11-03) -- the earlier note here said
  //                              they were "absent from autoforge-svc-gw", which
  //                              was true of a different build and had stopped
  //                              being true. `skill_port` was registered and the
  //                              three read-only ones were not, so the only tars
  //                              operation the model could reach was the one that
  //                              writes files.
  //   amend_self / set_autonomy  DSH's autonomy model is its approval policy,
  //   / import_package           not autoforge's freedom flags; self-editing the
  //                              service from inside a DSH turn would edit a
  //                              policy nothing here reads.

  ctx.effect(() => {
    // Start eagerly so a broken Python path surfaces at activation, not mid-turn.
    client.start()
    return () => client.stop()
  })

  // Tell the service what this file declares, so `describe_tool` can answer for
  // the 60 tools that are bridge wrappers rather than forged tools. Sent after
  // the `register()` calls above, which have already filled `catalog` (each one
  // runs its `ctx.effect` body immediately).
  //
  // Fire-and-forget on purpose: a service that cannot hear this is still a
  // service whose tools work, and the only thing lost is the manifest's advice
  // pointing at a name the service does not know. Losing that is degradation;
  // throwing here would take the whole preset down over a catalog.
  ctx.effect(() => {
    if (catalog.length === 0) return () => {}
    let cancelled = false
    client.call('report_bridge_tools', { catalog })
      .catch((err) => {
        if (cancelled) return
        ;(ctx.logger?.warn ?? ((m) => console.warn(m)))(
          `autoforge: could not report the bridge tool catalog (${err && err.message ? err.message : err}); describe_tool will only answer for forged tools`,
        )
      })
    return () => { cancelled = true }
  })

  mountBrief(ctx, client, view)
}

/**
 * Carry the service's memory and mission block into every request.
 *
 * This is the part of autoforge that lived in `agent.py`'s own loop and died in
 * the port: the loop went away, and with it the per-turn block. The tools
 * survived, so the facts were still reachable -- through `recall`, which the
 * model has to decide to call. From the model's side a kept fact it must
 * remember to ask for is identical to a fact that was never kept, and the
 * session that made agent.py write this block was one that had many tasks and
 * forgot the one sentence they were all for.
 *
 * The hook is `system-prompt/assemble`, not `agent/pre-step`. Both can carry
 * text, but pre-step feeds messages into a single step, while the assemble
 * waterfall feeds the request itself -- and DSH's own runtime context (sandbox
 * policy, approval policy, working directory) travels on that path, so this
 * block sits next to the facts the harness already treats as per-request truth
 * rather than competing with them.
 *
 * Failure is not fatal, on purpose: a service that died, a database that is
 * locked, a slow python must not brick a session. The block is dropped and a
 * warning is logged once -- an agent missing its memory is degraded, an agent
 * that cannot answer is useless.
 */
function mountBrief(ctx, client, view) {
  const warn = ctx.logger?.warn ?? ((m) => console.warn(m))
  let warned = false
  const warnOnce = (message) => {
    if (warned) return
    warned = true
    try { warn(message) } catch { /* logging must never be the failure */ }
  }

  // Two values ride into every request: the memory/mission brief, and the
  // measured self-description. Both come from the service, and the persona
  // makes a specific promise about the second one -- "`my_capabilities` 是测量
  // 出来的自我描述（射程/记忆/自身源码位置），每轮重算".
  //
  // It used to be false here. `SystemPrompt.context()` resolves `text` with a
  // plain call and does NOT await, so an async function renders as the literal
  // string "[object Promise]" (measured: a 16-character block). The first
  // version therefore refreshed out of band with `setInterval` and handed the
  // renderer a cache -- 60s for capabilities, 5s for the brief. Calling that
  // "每轮重算" was exactly the class of defect this whole exercise is about: a
  // promise in one place, an implementation in another, and nothing looking at
  // both.
  //
  // The shape that keeps the promise is DSH's own, from `dsh-working-directory`:
  // register a synchronous read for whatever you already have, then hook
  // `system-prompt/assemble` and await the real value there, replacing the
  // contribution before anything renders it. `resolveContext` runs on every
  // assemble, the waterfall runs after it, and the waterfall's return value is
  // authoritative -- so every request carries a value computed for that request.
  //
  // Synchronous reads stay as the cold-start and degradation path: an empty
  // string renders as nothing at all (`renderContextSections` filters empty
  // text), which is the honest answer while the service is still starting. A
  // heading with a placeholder in it would be read as a fact.
  const describe = { text: '', inflight: null, at: 0 }
  const brief = { text: '', inflight: null, at: 0 }
  // The exposure slot never renders; it exists so `fetchBlock` can supply its
  // single-flight and its bounded wait. `applyExposure` is the shape function,
  // which is why the "text" it returns is only used for the warning path.
  const exposure = { text: '', inflight: null, at: 0 }
  //: The text of the turn being assembled, per session.
  //:
  //: `tool_exposure` keeps every tool a task names in full, because the ledger
  //: only knows what has been *called* -- and the one turn that first needs a
  //: tool is exactly the turn whose ledger has never seen it. Upstream does this
  //: in `agent.py::_hot_protect`, reading `self._current_task`.
  //:
  //: The port has no `_current_task`, so the text comes from
  //: `agent/inbox/claimed`. That is a timing choice, not a convenience: in
  //: `dsh-agent-loop`, `preStep` runs `this.inbox.claim(target, position.turn)`
  //: -- which emits `agent/inbox/claimed` synchronously -- before
  //: `await this.systemPrompt.assemble(...)`. Claim first, assemble second, so a
  //: listener on the claim already knows this turn's words when the tool table
  //: is built. Reading the session instead would be one turn late: the user's
  //: message is only appended to the log in `step()`, which runs after the
  //: assembly. (`dsh-user-questions` registers this same event with
  //: `{ global: true }`, which is how the signature below is known.)
  //:
  //: Keyed by session and cumulative within a turn: a later step of the same
  //: turn adds what the user sent meanwhile and keeps what they already said,
  //: because the task did not change just because the step number did.
  const turnText = new Map()
  const TASK_CHARS = 4000
  let stopped = false

  //: How long one assembly may wait for the service before it uses what it has.
  //: The service is a local sqlite read behind a socket, so the honest case is
  //: single-digit milliseconds; this bound exists so a wedged python degrades
  //: one request instead of stalling the session. `my_capabilities` is asked
  //: with `probe: false` -- `probe: true` opens a real socket to measure reach,
  //: which is not something to do on every model step.
  const BUDGET_MS = 1500

  /**
   * Remember the words of the turn being assembled, from the claim that starts it.
   *
   * `message.source.kind` is not filtered here on purpose. A user message, a
   * steering message and a reply to a question all say something about what the
   * turn is for; the only ones worth skipping are the blocks this preset and the
   * harness inject themselves, and those are recognised by their `source.kind`
   * (`autoforge:brief`, `runtime-context`, the time-context plugin's own name).
   * Skipping the wrong one costs a protected tool; keeping the right ones costs
   * nothing that `protect_from_task` will honour anyway, because a name only
   * matches a whole token.
   */
  const noteTurnText = (agent, message, turn) => {
    const source = message && message.source
    const kind = source && typeof source.kind === 'string' ? source.kind : ''
    if (kind === 'runtime-context' || kind === 'system-prompt' || kind.startsWith('autoforge:')) return
    const blocks = message && Array.isArray(message.content) ? message.content : []
    let text = ''
    for (const block of blocks) {
      if (block && block.type === 'text' && typeof block.text === 'string') text += block.text + '\n'
    }
    if (text.trim() === '') return
    const session = agent && agent.session
    if (!session) return
    const held = turnText.get(session)
    // A new turn starts empty. Keeping the previous turn's words would protect
    // the tools of a task that has already finished, and the cost of that is
    // paid on every later turn as schema characters.
    const sameTurn = held !== undefined && held.turn === turn
    const together = sameTurn ? held.text + text : text
    turnText.set(session, {
      turn,
      text: together.length > TASK_CHARS ? together.slice(-TASK_CHARS) : together,
    })
  }

  /**
   * Ask the service for one block, bounded, and never throw.
   *
   * One attempt in flight at a time on purpose: the waterfall runs per request,
   * and a slow service must not turn into a queue of identical calls competing
   * with the request that is waiting on them. A caller that arrives while an
   * attempt is running waits on it rather than adding another.
   *
   * `budgetMs` is the time this attempt may spend. It is a parameter rather than
   * a constant because the constant was wrong: the hook fetches the exposure
   * first and the brief/caps after it, and with a fixed 1500ms each, one wedged
   * service held a turn for **3007ms** (measured 2026-11-03 by the harness in
   * `chk_bridge.mjs`). The bound is documented as "how long one assembly may
   * wait", so the assembly has to be the unit that owns it, not each call.
   *
   * ~0 or negative means the assembly is already out of time: return what the
   * slot holds and make no call at all. A call started with no budget left is a
   * call whose answer cannot be used.
   *
   * @returns the freshest text this attempt produced, or '' if it produced none.
   */
  const fetchBlock = (slot, method, args, shape, label, budgetMs = BUDGET_MS) => {
    if (slot.inflight) return slot.inflight
    if (!(budgetMs > 0)) return Promise.resolve(slot.text)
    let timer = null
    const giveUp = new Promise((resolve) => {
      timer = setTimeout(() => resolve(null), budgetMs)
      if (timer && typeof timer.unref === 'function') timer.unref()
    })
    slot.inflight = Promise.race([
      Promise.resolve().then(() => client.call(method, args)),
      giveUp,
    ])
      .then((res) => {
        if (res === null) {
          warnOnce(`autoforge: ${label} did not answer within ${budgetMs}ms; the previous block is used and the turn continues`)
          return ''
        }
        const text = shape(res)
        if (typeof text === 'string' && text.trim() !== '') {
          slot.text = text
          slot.at = Date.now()
        }
        return slot.text
      })
      .catch((err) => {
        warnOnce(`autoforge: ${label} unavailable (${String(err && err.message || err)}); tools still work`)
        return ''
      })
      .finally(() => {
        if (timer) clearTimeout(timer)
        slot.inflight = null
      })
    return slot.inflight
  }

  const briefText = (res) => {
    const text = res && res.text
    return typeof text === 'string' && text.trim() !== ''
      ? `AUTOFORGE — carried from the local service on every request.\n\n${text}`
      : ''
  }

  //: Which schemas ride in full this request. Refreshed on the same hook as the
  //: brief, and applied to the accessors that `mountTools` installed.
  //:
  //: `hot` starts empty and `ready` false: until the first answer arrives, every
  //: tool advertises its full schema. That is the expensive-but-safe default --
  //: a cold tool costs one extra `describe_tool` call at worst, while a wrongly
  //: cold tool is invisible in exactly the way this whole exercise exists to
  //: prevent.
  //:
  //: `ready` is not redundant with `hot.length`. An answer of `hot: []` is a
  //: legitimate answer (a registry with no calls in the window and no catalog),
  //: and treating it as "nothing is ready yet" would mean never projecting at
  //: all; treating a MISSING answer as `hot: []` would mean stripping every
  //: schema on turn one. The flag separates those two cases, and it is the only
  //: thing that separates them.
  const applyExposure = (res) => {
    if (!res || res.ok !== true) return ''
    const hot = Array.isArray(res.hot) ? res.hot : []
    const cold = res.cold && typeof res.cold === 'object' ? res.cold : {}
    view.hot = new Set(hot.map(String))
    view.cold = cold
    view.at = Date.now()
    // Only now may anything be shown as cold. Setting this on any successful
    // answer -- including an empty catalog, which answers `hot: []` -- is what
    // makes `ready` mean "the service has spoken", not "the service likes us".
    view.ready = true
    return typeof res.text === 'string' ? res.text : ''
  }

  // Prime once so a session whose first assembly loses the race is not starting
  // from nothing. This is a warm start, not the refresh mechanism: every
  // assembly recomputes below, and there is no interval left to fall back on.
  ctx.effect(() => {
    stopped = false
    fetchBlock(brief, 'context_brief', {}, briefText, 'memory brief')
    fetchBlock(describe, 'my_capabilities', { probe: false }, (res) => (res && res.text) || '', 'capability block')
    return () => { stopped = true }
  })
  ctx.effect(() => {
    // `systemPrompt.context()` is the registered way to contribute a named
    // runtime fact; it is ordered against the harness's own contributions and
    // disappears with this plugin. Writing into the assembled object directly
    // would work until two plugins did it.
    //
    // Resolved through `ctx.get`, NOT `ctx.systemPrompt`. Cordis's context
    // proxy throws on any property that is not in this plugin's `inject` list
    // (`cannot get property "systemPrompt" without inject`), and it throws at
    // the property READ -- so the `?.` below never runs and the graceful
    // degradation this block is written for could never happen. Measured: the
    // preset was rejected at mount with exactly that message, which is how the
    // whole autoforge preset went `broken`.
    //
    // `inject` is the wrong fix: adding `systemPrompt` there makes it a
    // REQUIRED dependency, so a mount without it fails the plugin instead of
    // skipping one block. This service is deliberately optional -- the tools
    // work without it, they just lose the memory brief.
    const systemPrompt = ctx.get('systemPrompt')
    if (typeof systemPrompt?.context !== 'function') {
      warnOnce('autoforge: no systemPrompt service on this mount, so the memory brief is not injected; tools still work')
      return () => {}
    }

    // Order 126 and 127 sit in the two empty slots between SUBAGENT_DELEGATION
    // (120) and HARNESS_SOURCE (200): after the harness's policy facts, before
    // its source pointers. 127 immediately follows 126 on purpose -- the model
    // reads what it owes (mission) and then what it is, in that order.
    //
    // `interpolate: false` because both blocks are service prose: a literal
    // `{{name}}` inside a remembered fact or a stored source path must survive
    // to the model unchanged. With interpolation on, `interpolate()` throws on
    // an unknown group and takes the request down over a brace in a note.
    const disposers = [
      systemPrompt.context({
        name: 'autoforge:brief',
        order: 126,
        interpolate: false,
        text: () => brief.text,
      }),
      systemPrompt.context({
        name: 'autoforge:identity',
        order: 127,
        interpolate: false,
        text: () => describe.text,
      }),
    ]

    return () => { for (const d of disposers) { if (typeof d === 'function') d() } }
  })

  // The per-request half, registered outside the effect above on purpose:
  // `ctx.on` is already scoped to this plugin's lifetime, and calling a Cordis
  // disposer from inside an effect that also disposes it is how double-dispose
  // bugs are made. `dsh-working-directory` registers its hook the same way.
  //
  // `resolveContext` has already run by the time the waterfall is invoked, so
  // the contributions above carry cached text; this hook recomputes both blocks
  // and overwrites them before rendering. The returned assembly is what the
  // request is built from.
  //
  // Failure here is degradation, not an error: each fetch is bounded and
  // swallows its own exceptions, so a dead service costs one stale block and
  // the turn still goes out.
  // Learn what the turn is about before its tool table is decided.
  //
  // `{ global: true }` because the event carries the agent explicitly and a
  // scoped listener would only hear about this plugin's own scope; the claim is
  // emitted by the loop for whichever agent started the step.
  // `dsh-user-questions` subscribes to the same event the same way.
  ctx.on('agent/inbox/claimed', ({ agent, message, turn }) => {
    try { noteTurnText(agent, message, turn) } catch { /* never fail a turn for a hint */ }
  }, { global: true })

  ctx.on('system-prompt/assemble', async (_assembly, context, next) => {
    const assembly = await next()
    if (stopped || !assembly) return assembly
    // One deadline for the whole assembly, not one per call. `BUDGET_MS` is the
    // documented promise ("how long one assembly may wait"); three sequential
    // fetches each carrying their own 1500ms would let a wedged service hold a
    // turn for 4.5s while every individual call stayed inside its bound. Measured
    // 2026-11-03: the two-call version took 3007ms on a service that never answered.
    const deadline = Date.now() + BUDGET_MS
    const left = () => deadline - Date.now()
    // The tool table is refreshed whether or not this assembly carries our two
    // context blocks: a scoped layer that suppresses `autoforge:brief` still
    // sends every tool schema, and that is the bill this refresh exists to
    // reduce. It runs before the assembly is handed back, because the schemas
    // for this request are projected from the accessors at send time.
    // The task text, when a claim has already told us what this turn is for.
    // `context.scope` is the agent for a per-agent assembly; an assembly with
    // no scope has no task and gets the ledger's answer, which is what this
    // projection was before `task` existed.
    const held = context && context.scope ? turnText.get(context.scope.session) : undefined
    const task = held && typeof held.text === 'string' ? held.text : ''
    await fetchBlock(exposure, 'tool_exposure', task ? { task } : {}, applyExposure, 'tool exposure', left())
    if (!Array.isArray(assembly.contexts)) return assembly
    const mine = assembly.contexts.filter(
      (entry) => entry && (entry.name === 'autoforge:brief' || entry.name === 'autoforge:identity'),
    )
    // Nothing of ours in this assembly (a scoped or suppressed layer): the
    // per-request work would be paid for a block that nobody renders.
    if (mine.length === 0) return assembly
    const [briefNow, capsNow] = await Promise.all([
      fetchBlock(brief, 'context_brief', {}, briefText, 'memory brief', left()),
      fetchBlock(describe, 'my_capabilities', { probe: false }, (res) => (res && res.text) || '', 'capability block', left()),
    ])
    for (const entry of mine) {
      if (entry.name === 'autoforge:brief') {
        if (briefNow) entry.text = briefNow
      } else if (capsNow) {
        entry.text = capsNow
      }
      entry.interpolate = false
    }
    return assembly
  })
}
