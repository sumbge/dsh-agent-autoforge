
/**
 * Agent-preset bundle that mounts the autoforge service as model-facing tools.
 *
 * The service (autoforge-svc) is a long-lived Python child process speaking
 * line-delimited JSON-RPC over stdio. This plugin owns that process, exposes
 * each RPC method as a DSH tool, and recycles the process when it dies.
 */
import { spawn } from 'node:child_process'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'agent-autoforge'
export const inject = ['tools']

const DEFAULTS = {
  python: 'python',
  cwd: 'C:\\Users\\32535\\Desktop\\dsh\\autoforge-svc',
  home: 'C:\\Users\\32535\\Desktop\\dsh\\autoforge-svc\\.data',
  skills: 'C:\\Users\\32535\\AppData\\Local\\autoforge\\skills',
  // Where the interrupted-run journal lives. Empty means "let the service
  // decide", which is `$AUTOFORGE_HOME/runs` -- correct for a mount that keeps
  // its own data, wrong for one that should see the CLI's runs. Passing it
  // here is the whole fix: `runs.journal_dir()` already prefers
  // `$AUTOFORGE_RUNS`, so no service code changes when this moves.
  runs: '',
  timeoutMs: 120000,
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
    const child = spawn(this.options.python, ['-m', 'autoforge_svc'], {
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

/** Build one model-facing tool bound to one RPC method. */
function rpcTool(client, { toolName, method, description, parameters, coerce }) {
  return defineTool({
    name: toolName,
    description,
    parameters: parameters || {},
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args) {
      // `coerce` may either rewrite `params` in place (the original style, used
      // by the JSON-field helpers) or return a new object (the tool_define
      // renamer). Ignoring the return value silently dropped every rename:
      // measured 2026-10-25, `sampleCall` reached Python unchanged and raised
      // "unexpected keyword argument", with the coerce function correct and
      // never used. So: take the return value when there is one, otherwise
      // keep what the in-place writers put in `params`.
      let params = { ...args }
      if (coerce) {
        const out = coerce(params)
        if (out && typeof out === 'object') params = out
      }
      const result = await client.call(method, params)
      // 一个方法回答 `{ok:false}` 时，那是失败，不是数据。以前一律当文本
      // 返回，DSH 记为 completed，模型看到的是「空输出」而不是「失败原因」。
      // 现在把失败原因抛出去：DSH 会把它记成 status=error，模型读得到原文。
      const reason = failureReason(result)
      if (reason) throw new Error(`${toolName}: ${reason}`)
      return asText(result)
    },
  })
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
  const register = (spec) => ctx.effect(() => {
    const dispose = ctx.tools.register(rpcTool(client, spec))
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
      next_step: { type: 'string', description: 'the next action' },
      blocked_on: { type: 'string', description: 'what it waits on; empty string clears it' },
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
    },
    coerce: (params) => {
      const out = { ...params }
      // `probes` keeps its name, so it is handled below and NOT in this rename
      // loop. It sat in the loop at first, where `out[to] = out[from]` was
      // immediately followed by `delete out[from]` on the same key: the probe
      // list was assigned and then deleted, and every declared probe vanished
      // in silence. Measured 2026-10-25 -- trigger went back to UNPROBED with
      // the parameter present and correct-looking at the call site.
      for (const [from, to] of [['sampleCall', 'sample_call'],
                                ['sampleExpect', 'sample_expect']]) {
        if (out[from] === undefined) continue
        out[to] = out[from]
        delete out[from]
      }
      if (typeof out.probes === 'string') {
        try { out.probes = JSON.parse(out.probes) } catch { /* left as-is */ }
      }
      if (typeof out.parameters === 'string') {
        try { out.parameters = JSON.parse(out.parameters) } catch { /* left as-is */ }
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
    toolName: 'autoforge_tool_call',
    method: 'tool_call',
    description: "Run one of YOUR forged tools by name: the tools `autoforge_forge_tool` created and `autoforge_list_tools` lists. This is not a way to reach the service's own methods -- `describe`, `list_tools`, `skills` and the rest each already have a tool of their own, and calling one here returns `unknown tool`. Check `autoforge_list_tools` first; a retired or quarantined tool is refused unless it is rehabilitated.",
    parameters: {
      name: { type: 'string', required: true, description: 'Name of a forged tool, as autoforge_list_tools reports it.' },
      arguments: { type: 'json', description: 'Arguments for the tool, as a JSON object matching its schema.' },
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
    description: 'Run the full verification battery (execution, robustness, adversarial, trigger, negative, and the claim gate: is the returned number a fact about the world or about the data it was handed) on a tool.',
    parameters: {
      name: { type: 'string', required: true, description: 'tool name' },
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
  //   skill_port / tars_*        autoforge-clean only -- absent from
  //                              autoforge-svc-gw, which is what this preset
  //                              actually spawns.
  //   amend_self / set_autonomy  DSH's autonomy model is its approval policy,
  //   / import_package           not autoforge's freedom flags; self-editing the
  //                              service from inside a DSH turn would edit a
  //                              policy nothing here reads.

  ctx.effect(() => {
    // Start eagerly so a broken Python path surfaces at activation, not mid-turn.
    client.start()
    return () => client.stop()
  })

  mountBrief(ctx, client)
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
function mountBrief(ctx, client) {
  const warn = ctx.logger?.warn ?? ((m) => console.warn(m))
  let warned = false
  const warnOnce = (message) => {
    if (warned) return
    warned = true
    try { warn(message) } catch { /* logging must never be the failure */ }
  }

  // The block is refreshed in the background and read synchronously.
  //
  // `SystemPrompt.context()` resolves `text` with a plain call -- it does not
  // await, so an async function renders as `[object Promise]` (measured: the
  // block came out 16 characters long, the string "[object Promise]"). The
  // service call behind it is a Promise, so the only shape that works is a
  // synchronous read of a value refreshed out of band. Nothing is lost: the
  // block changes when a fact is kept, which is a second-scale event, while
  // assembly happens on every model request.
  let cached = ''
  let inflight = null
  let stopped = false

  const refresh = () => {
    if (stopped || inflight) return inflight
    inflight = Promise.resolve()
      .then(() => client.call('context_brief', {}))
      .then((res) => {
        const text = res && res.text
        if (typeof text === 'string' && text.trim() !== '') {
          cached = `AUTOFORGE — carried from the local service on every request.\n\n${text}`
        }
      })
      .catch((err) => {
        warnOnce(`autoforge: memory brief unavailable (${String(err && err.message || err)}); tools still work`)
      })
      .finally(() => { inflight = null })
    return inflight
  }

  ctx.effect(() => {
    stopped = false
    // Prime it now so the first request already carries the block, then keep it
    // fresh. 5s is well under a person's attention and well over a model step.
    refresh()
    const timer = setInterval(refresh, 5000)
    if (typeof timer.unref === 'function') timer.unref()
    return () => { stopped = true; clearInterval(timer) }
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
    // The identity half. `my_capabilities` is the one description of this
    // agent that is *measured* rather than written down -- reach, memory and
    // what the ledger holds, all read from the objects that know. Left behind
    // a tool call, the model only learns it by deciding to ask, and from its
    // side that is the same as never knowing.
    //
    // Refreshed on a much longer clock than the brief, and with probe=false:
    // `probe=true` opens a real socket to measure reach, which is not something
    // to do every minute in the background. 60s covers the thing this block
    // actually tracks (tools appearing and retiring), and it is cached text, so
    // it costs nothing per request.
    const caps = { text: '', warm: false }
    let capsInflight = null
    const refreshCaps = () => {
      if (stopped || capsInflight) return capsInflight
      capsInflight = Promise.resolve()
        .then(() => client.call('my_capabilities', { probe: false }))
        .then((res) => {
          const text = res && res.text
          if (typeof text === 'string' && text.trim() !== '') {
            caps.text = text
            caps.warm = true
          }
        })
        .catch((err) => {
          warnOnce(`autoforge: capability block unavailable (${String(err && err.message || err)}); tools still work`)
        })
        .finally(() => { capsInflight = null })
      return capsInflight
    }

    ctx.effect(() => {
      refreshCaps()
      const timer = setInterval(refreshCaps, 60000)
      if (typeof timer.unref === 'function') timer.unref()
      return () => { clearInterval(timer) }
    })

    const disposers = [
      systemPrompt.context({
        name: 'autoforge:brief',
        // 126 sits between SUBAGENT_DELEGATION (120) and HARNESS_SOURCE (200):
        // after the harness's policy facts, before its source pointers.
        order: 126,
        text: () => cached,
      }),
      systemPrompt.context({
        name: 'autoforge:identity',
        // 127, immediately after the brief: the model reads what it owes
        // (mission) and then what it is, in that order.
        //
        // Empty until the first refresh lands -- 60s is a long time to render
        // an empty heading, but a heading with a placeholder in it would be
        // read as a fact about this agent. `caps.warm` is what keeps an empty
        // string out of the prompt entirely.
        order: 127,
        text: () => caps.text,
      }),
    ]
    return () => { for (const d of disposers) { if (typeof d === 'function') d() } }
  })
}
