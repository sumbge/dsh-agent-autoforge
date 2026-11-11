# dsh-agent-autoforge

让 DSH 的 agent preset 调用 [autoforge](https://github.com/ZhangYangyi03/autoforge) 的工具治理能力：
forge → 六道验证 → 封存，加上跨会话的结构化记忆、mission 账、技能库和留痕账本。

两部分：

- **`lib/` · `presets/`** —— Cordis 插件，把服务的方法声明成 DSH 工具（73 个）
- **`service/`** —— Python stdio JSON-RPC 服务，执行工具治理

---

## 装之前：先把六处路径换成本机的

`presets/autoforge.patch.yml` 里 `agent-autoforge-tools` 那一节的路径**是某一台机器的绝对路径**，
不是通用值。它们能跑，是因为那台机器的目录就叫这些名字。

| 行 | 键 | 那台机器上的值 | 你该填什么 |
| --- | --- | --- | --- |
| 98 | `python` | `C:\Users\sumbger\...\Python313\python.exe` | 任何能跑 `python -m autoforge_svc` 的 Python 3.10+ 解释器；Windows 上别写裸 `python`（那是 Microsoft Store 的存根，打印一段横幅后退出 9009） |
| 99 | `cwd` | `...\.dsh\profiles\desktop\node_modules\@local\dsh-agent-autoforge\service` | 本仓库 `service/` 目录的绝对路径。**这一条最容易错**：它的值只在插件已经装进 `desktop` profile 之后才存在，所以你复制别人的 yml 时它一定指向一个不存在的目录 |
| 100 | `home` | `%LOCALAPPDATA%\autoforge\data` | 账本、mission、日程的落点。**要跟 CLI 那份共用**，否则你会得到两个库，各自的 `report()` 都只看得见自己 |
| 101 | `skills` | `%LOCALAPPDATA%\autoforge\skills` | 技能目录，与 `skill-filesystem` 共用 |
| 105 | `runs` | `%LOCALAPPDATA%\autoforge\runs` | 中断续跑的 journal。**不填就看不到 CLI 那边的 run** |
| 143 | `customSkillDirs[0]` | `%LOCALAPPDATA%\autoforge\skills` | 与第 101 行保持一致，否则模型得用两种方式读技能 |

### 更省事的办法：把那几行删掉

`lib/index.js` 的 `DEFAULTS` 会为每个键推导一个安全的回落值，全部从**本文件的位置**和
**当前用户**算出来，没有任何一个键依赖这台开发机：

    cwd       fileURLToPath(new URL('../service/', import.meta.url))   本文件旁边的 service/
    home      %LOCALAPPDATA%\autoforge\data     (非 Windows: ~/.autoforge/data)
    skills    %LOCALAPPDATA%\autoforge\skills
    runs      ''                                空 = 由服务决定（$AUTOFORGE_HOME/runs）
    timeoutMs 120000

`python` 是一个**候选表**，`start()` 时逐个跑一遍、第一个能跑的胜出
（每个都真的执行 `-c "print(1)"`，要求退出码 0 且输出正好是 `1`）：

    $AUTOFORGE_PYTHON  →  py  →  python3  →  python

`py` 排在 `python` 前面是有原因的：Windows 上裸 `python`/`python3` 很可能指到 Microsoft Store
的存根，它打印一段横幅然后以 9009 退出——看起来像"服务起不来"。`py` 是 Windows 启动器，是那种
机器上唯一还在的入口。四个都不通时会抛一条带候选表的错误，而不是把一个谁都看不懂的 ENOENT
留给 `spawn`。

**表里没有本机的解释器绝对路径**：一条只在写作地能用的路径，正是这段代码存在要阻止的东西。
所以这一节不存在"要替换的本机用户名"，你只需要确保上面四个里有一个能跑——装了正式 Python
但不在 PATH 上时，设 `AUTOFORGE_PYTHON` 就够了。（`chk_defaults.py` 断言的正是这件事：
候选表非空、`python` 不在首位、任何候选都不含 `32535` 这种别人的用户名。）

所以「照抄一段路径」比「删掉这个键」更容易装坏。这两个行为都验证过（`service/tests/regression/chk_defaults.py`
读的就是这段 `DEFAULTS` 的真值，不是它上面的注释）。

给代装的 AI 的一句话：**上面那张表里每一行要填的是「本机上那个目录在哪」，不是「照抄左边那一列」。**

---

## 三个数量口径，别混着说

同一个仓库里「有多少个工具」有三个都正确、但互不相同的答案：

| 说法 | 数字 | 是什么 |
| --- | --- | --- |
| 注册的工具 | **73** | `lib/index.js` 里 `toolName: 'autoforge_*'` 的唯一取值数，也是 `method:` 的唯一取值数 —— 模型真能看见的那批 |
| 服务的线协议方法 | **76** | `len(Service().methods)`。多出的 3 个不挂成工具：`context_brief`（每轮注入的简报）、`tool_exposure`（按当轮任务决定哪些工具保留完整 schema）、`report_bridge_tools`（插件启动时上报自己的工具目录） |
| 库里的自造工具 | 运行时状态 | `registry.names()` 的长度。它是**这台机器上攒下来的**，不是源码里的常数；新装的库是 0 |

上游 79 个名字里 22 个没挂，`lib/index.js` 又新增了 16 个上游没有的名字：79 − 22 + 16 = 73
（两组清单见下面「上游没有的部分」）。

顺带：回归套件 **67 个**（63 个 `.py` + 4 个 `.mjs`），`service/` 树 **90 个 `.py` / 41,716 行**。

---

## 来源

`service/autoforge_svc/` 从 [ZhangYangyi03/autoforge](https://github.com/ZhangYangyi03/autoforge)
复制并改造，**不是 fork 关系，是复制品**。

### 原样复制的模块

| 服务里的 | 上游对应 |
| --- | --- |
| `mission.py`、`schedule.py`、`notify.py`、`runs.py` | 同名文件 |
| `forge/`（16 个） | `autoforge/forge/` |
| `cpu/`（6 个）、`gpu/`（8 个） | 同名目录 |
| `autonomy/`（8 个）、`tools/`（5 个）、`core/`（5 个） | 同名目录 |
| `vendor/tars/`（37 个，291 KB） | `autoforge/vendor/tars/` |

### 有改动的模块

| 模块 | 改动 |
| --- | --- |
| `store.py` | 增加 `save_stats`（一次调用只更新计数器，不重写整行）；`upsert_skill` 内容未变时跳过写入 |
| `skills.py` | `port_openclaw_skills` / `port_tars_skills` 增加 `roots` / `library` / `store` 注入参数 |
| `forge/verifier.py` | 增加 `unprobed` / `waived` 三态判定；`negative` 与 `trigger` 三次采样；探针距离评分 |
| `forge/sandbox.py` | `_probe` 改为父子进程各自解析 temp 目录，由父进程复核写入 |
| `cpu/safety.py` | `run_isolated` 的 `-m` 模块名从硬编码改为 `__name__` |

其中 `verifier.py` 的三态判定、`negative` 多数投票、`cpu/safety.py` 和 `sandbox.py` 的修复
**已推回上游 `feat/wbproxy-contrib` 分支**（commits `9daccd9`、`ea85c19`、`5c96d69`）。

### 本仓库独有

| 文件 | 作用 |
| --- | --- |
| `service.py`（289 KB / 5,547 行） | 76 个 RPC 方法 + stdio 循环 |
| `__main__.py` | 两个入口：stdio 服务、`tick` |
| `tests/` | 17 个单元测试（注入层、无人值守路径） |
| `tests/regression/` | 67 个断言套件（见下） |

---

## 上游没有的部分

**上游 79 个工具，这里挂 73 个。** 这个数曾经写成 64：那时 `lib/index.js` 里确实只有 64 条，
之后陆续加进来 `review_tool`、`anchor_ledger`、`runs_prune`、`schedule_run_due`、
`system_task_command` 等，表格没有跟着改。现在 73 是从源码数出来的，不是抄的。
缺的 22 个是刻意的，分成六类：

| 缺失 | 数量 | 理由 |
| --- | --- | --- |
| `browser_*`（click/close/eval/goto/open/press/screenshot/state/type） | 9 | DSH 自带 browser 工具 |
| `see` / `draw` / `animate` | 3 | DSH 自带同名工具 |
| `mcp_servers` / `mcp_connect` / `mcp_call` | 3 | DSH 有 MCP 客户端和 manager，服务侧没有 MCP 配置 |
| `spawn_agent` / `design_team` | 2 | DSH 有 `subagent` / `subagent_fork` / `workflow-ptc` |
| `set_autonomy` / `amend_self` / `import_package` | 3 | **机制冲突**：上游是「四类自由度 + confirm 闸」，DSH 是 sandbox + approval，两套权限模型不能共存 |
| `vision_status` / `media_status` | 2 | 服务侧没有对应方法，挂了只会返回 error |

补进来的 16 个（上游 `agent.py` 里没有这个名字）：
`anchor_ledger`、`describe`、`events`、`log_event`、`mission_history`、`ping`、`rehab_tool`、
`report`、`review_tool`、`runs_prune`、`schedule_run_due`、`system_task_command`、
`tool_call`、`tool_define`、`tool_list`、`tool_usage`。

**上游的 agent 循环没有搬。** 这里换成了 DSH 的 turn。上游循环的三个刻意约束
（`max_turns=None`、`terminate` 作为一等工具、compaction 只调策略）在这个模式下都不存在。

---

## 依赖

**Python 3.10+**，核心路径无第三方依赖。

| 功能 | 需要 | 缺失时的行为 |
| --- | --- | --- |
| CPU 内核 | C 编译器（gcc/cc/clang/cl/tcc 在 PATH） | `cpu_compile` 明确报错，`cpu_probe` 仍工作 |
| GPU 内核 | CUDA toolkit（`nvcc`） | `gpu_compile` 明确报错；`gpu_probe` 用 `nvidia-smi` 仍工作 |
| forge 模型 | OpenAI 兼容网关 | 报错并列出所有已检查的密钥来源 |
| 通知 | webhook 或 SMTP，配在 `~/.autoforge/config.json` | `notify_send` 说明该配什么（并给出该文件的完整路径；这个包不会替你创建它） |

---

## 配置

```yaml
- id: agent-autoforge-tools
  name: '@local/dsh-agent-autoforge'
  config:
    python: python
    cwd: <服务目录>              # 不填 = 本插件旁边的 service/
    home: <数据目录>             # sqlite 账本、mission、schedule
    skills: <技能目录>           # 与 DSH 的 skill-filesystem 共用
    runs: <journal 目录>         # 中断续跑的 run-*.json
    timeoutMs: 600000
```

`runs` 要单独给，是因为 journal 通常在 CLI 那边（`%LOCALAPPDATA%\autoforge\runs`），
不在 `home` 下面。不指这里，`runs_status` 会读一个空目录并报告 0 条 ——
那比缺功能更难查，因为它看起来像答案。

模型网关默认读 `$DSH_HOME/openai-gateway/api-key`，端点 `127.0.0.1:8326`。
可用环境变量覆盖：`AUTOFORGE_MODEL`、`AUTOFORGE_BASE_URL`、`AUTOFORGE_API_KEY`、
`AUTOFORGE_HOME`、`AUTOFORGE_RUNS`、`AUTOFORGE_SKILLS_DIRS`、`AUTOFORGE_PYTHON`。

不设 `AUTOFORGE_MODEL` 时，模型 id 从当前 DSH profile 的
`profiles/<name>/cordis.patch.yml` 推导（`agent-default-model` 的
`provider`/`model`，其次 `enabledModelIds`），不是常量。这里曾写死过
`xdpool-global/deepseek-v4.1-flash`，而本机网关不提供它，forge 与定时任务
两条路径都在第一次模型调用上 404；`report()` 的 `model` 块现在会说明当前
id 以及它是从哪个文件读到的。

**配的那个模型必须读得到工具结果。** 有一类模型能发 `tool_calls`、但把
`role: "tool"` 的消息当空——它会说"每次调用都返回 No result provided"，然后
拿一个凭空算出来的数当答案，整个 Agent 看起来"调了工具但没效果"。服务的
`_reads_tool_results()` 用一对裸 payload 实测这件事，`report()` 的
`model.reads_tool_results` 与 `model.resolution` 会给出判决（`configured` /
`substituted: X` / `unverified` / `no alternative`）以及替换的理由。

---

## 版本范围的依据

声明写的是 `engines.dsh: >=0.2.1-rc.2 <0.3.0`，**这个范围验证过，不是猜的。**

写死 `0.2.1-rc.2` 时，每次 DSH 升级都会被插件管理器判为不兼容并拒绝加载。改成范围前核对过
本插件依赖的三个 API 在 `0.2.1-rc.5` 下仍然成立：

```
defineTool(options)              签名与 output.render 调用约定未变
systemPrompt.context({...})       name / order / text 参数未变
ctx.tools.register(definition)    output.render 仍是必需项
```

上界 `0.3.0` 是有意的：0.2.x 是同一代 API，0.3 是下一代，跨过去需要重新验证。

`0.2.2-rc.1` 也逐项对撞过（六个 API 点全部兼容：`register()` 只多了可选 `options.when`，
`schemaOf` 改名 `projectToolSchema` 但函数体逐字一致，`agent/inbox/claimed` 的时机与 payload 不变，
`system-prompt/assemble` 的 waterfall 形状不变，`definition.execute(args)` 仍被调用）。
预发布版本本来不落在 semver 的 `<0.3.0` 里，但 DSH 自己的兼容性检查用的是
`semver.satisfies(v, range, { includePrerelease: true })`（`dsh-app-boot/lib/index.js`），
所以 `0.2.2-rc.1` 这一关是放行的 —— 不必为它改 `package.json`。

---

## 回归

67 个套件分成两类，**判决行不同，一套跑法**：

    有判决行的（56 个）  末行形如 TOTAL FAILURES: 0 / TOTAL BAD 0 of 14 /
                        TOTAL WRONG 0 of 11 / FALSE-FAILED 0 / BAD 0，退出码 0
    诊断脚本（11 个）    只打印数值，不看退出码（chk_counter / chk_dbg / chk_diag_small /
                        chk_pathaxis / chk_sig / chk_state / chk_evade* /
                        chk_bridge_catalog.hooks.mjs）

跑法：`PYTHONPATH=<repo>/service python -P service/tests/regression/chk_xxx.py`，
`pytest` 的 cwd 必须是 `service/`（不是仓库根）。

其中 `chk_seams.py` 会顺带核对「报告里引用的每个 harness 都存在」—— 它引用的是
开发过程那份审计报告，不在本仓库里。没有那份报告时它应当跳过并说明原因，而不是 FAIL。

---

## 许可

服务层代码源自 [autoforge](https://github.com/ZhangYangyi03/autoforge)（MIT，Copyright © 2026 Yangyi Zhang），
本仓库的改动同样以 MIT 发布，见 [LICENSE](LICENSE)。
哪些文件是复制的、哪些是新写的、`vendor/tars/` 的来源，见 [NOTICE](NOTICE)。
LICENSE 里只有 MIT 原文，没有附加说明——GitHub 的许可证识别器读到额外段落就会
把仓库标成 `NOASSERTION`，所以说明单独放一个文件。
插件侧说明见 [PLUGIN.md](PLUGIN.md)。
