# @local/dsh-agent-autoforge

把 autoforge 服务挂成 DSH 的 agent 预设。预设 id 为 `autoforge`，选中后模型多出
**73 个** `autoforge_*` 工具，全部经由本机 `autoforge-svc` 的 stdio JSON-RPC 往返。

> 这个数字曾经写成 61，也写成过 64；两个都是某个中间版本上的真值。现在的依据是
> `lib/index.js` 里 `toolName: 'autoforge_*'` 的**唯一取值数 = 73**，与 `method:` 的唯一取值数
> 一致（`chk_wire.py` 会核对这两份清单与服务的方法表三向一致）。
> 服务那边有 76 个线协议方法，多出的 3 个不挂工具：`context_brief`、`tool_exposure`、
> `report_bridge_tools` —— 它们是**每轮由插件自己按钩子调用**的，给模型等于给它一条
> 通往自己提示词内部的路。

工具按用途分十组（共 73）：

    记忆 3     remember / recall / forget
    使命 10    mission_open / close / note / show / list / focus / block / sweep /
               wake / history
    日程 8     schedule_add / cancel / done / list / tick / run_due /
               install_system_task / system_task_command
    技能 6     skill_write / view / list / errors / forget / port
    锻造 9     forge_tool / evolve_tool / evaluate_tool / review_tool / find_gaps /
               retire_tool / rehab_tool / list_tools / describe_tool
    工具库 4   tool_define / tool_call / tool_list / tool_usage
    账本 10    run_python / runs_status / runs_prune / runs_forget / resume_run /
               verify_ledger / anchor_ledger / my_history / log_event / events
    自身 6     describe / ping / report / my_capabilities / notify_channels / notify_send
    算力 14    cpu_compile / cpu_preflight / cpu_probe / cpu_run_isolated /
               cpu_runs_here / cpu_tune / cpu_units_audit / cpu_cache_stats /
               gpu_bench / gpu_compile / gpu_occupancy / gpu_probe /
               gpu_units_audit / gpu_verify
    tars 3     tars_probe / tars_skills / tars_port_skills

`describe_tool(name)` 有两层含义，别混：模型看见的 `autoforge_describe_tool` 查的是
**服务侧**的 schema 与战绩；插件启动时还调用服务端的 `report_bridge_tools` 把
**桥接层**那 73 条 descriptor 上报进库，所以同一个命令也能回答"桥接层向我广告的是什么"。

## 故意不挂的 22 个

上游 `agent.py` 有 79 个工具名，这里挂 73 个。差的那批不是漏了，多数是 DSH 自己已经有一份，
同时挂会给模型两条通往同一件事的路：

    browser_* (9)                click/close/eval/goto/open/press/screenshot/state/type
                                 DSH 自带 browser 工具
    see / draw / animate (3)     DSH 自带同名工具
    mcp_servers / connect / call (3)
                                 DSH 有 MCP 客户端和 manager，服务侧没有 MCP 配置
    spawn_agent / design_team (2)
                                 DSH 有 subagent / subagent_fork / workflow-ptc
    amend_self / set_autonomy / import_package (3)
                                 DSH 的自治模型是 approval policy，不是这套 freedom
    vision_status / media_status (2)
                                 服务侧没有对应方法，挂了只会返回 error

补进来的 16 个上游没有的名字：`anchor_ledger`、`describe`、`events`、`log_event`、
`mission_history`、`ping`、`rehab_tool`、`report`、`review_tool`、`runs_prune`、
`schedule_run_due`、`system_task_command`、`tool_call`、`tool_define`、`tool_list`、
`tool_usage`。

## 每轮注入

两个 context 块：`autoforge:brief`（事实、使命、工具计数）与 `autoforge:identity`（测量出来的
自我描述），另外还有一个**不渲染**的 `tool_exposure` 调用。

它们都挂在 `system-prompt/assemble` 上**每次装配都重算**，不是定时刷新。第一次实现是
`setInterval` + 一份缓存（brief 5s、capabilities 60s），而人设里写着"每轮重算" ——
一处承诺、另一处实现、没人同时看两处，这正是这套工具存在的理由。现在的形状抄的是
`dsh-working-directory`：先注册一个同步读（冷启动返回空串，空串渲染成 nothing），
再在 waterfall 里 await 真值并覆盖。

`BUDGET_MS = 1500` 是**整次装配**的预算，不是每次调用的预算：三次串行取数各带 1500ms
会让一个卡住的服务把一轮拖到 4.5 秒（实测过 3007ms）。服务挂了就丢掉那一段并 warn 一次，
工具照常可用。

`tool_exposure` 按当轮任务文本决定哪些工具保留完整 schema：任务里点名的工具全量发，
其余发一句"Not loaded in full"的存根（省 27.4% 字符，实测 23,887 → 17,347）。它只改
**广告**、不改**接受** —— 冷形态的工具被调用时照样按完整 schema 校验参数。

## 服务

插件启动时 spawn：

    <python> -m autoforge_svc      cwd: <config.cwd>
    AUTOFORGE_HOME=<config.home>
    AUTOFORGE_RUNS=<config.runs>   （给了才设）

进程死了会自动重启一次；请求超时由 `timeoutMs` 控制（默认 120000，预设里给 600000，
因为一次 forge 是六道检查、其中两道各跑一个完整探针会话）。

## 配置（preset 行里的 config）

字段      默认值                                   含义
python    [AUTOFORGE_PYTHON, python3, python]      逐个试，`print(1)` 能跑的那个胜出
cwd       <本文件>../service/                        服务包所在目录
home      %LOCALAPPDATA%/autoforge/data              AUTOFORGE_HOME，状态库落在这里
skills    %LOCALAPPDATA%/autoforge/skills            技能目录
runs      ''（由服务决定 → $AUTOFORGE_HOME/runs）     中断续跑的 journal
timeoutMs 120000                                     单次 RPC 超时

**默认值全部是推导出来的，不含本机的用户名。** 上一版的表在一个机器的路径上写了三次
`C:\Users\<another-user>\...`、第四处写 `python`；因为随包的预设把这四项全填了，所以一直没坏 ——
一个永远被覆盖的默认值，就是一个没人读过、也没人测过的默认值。`chk_defaults.py` 现在
读的是这段 `DEFAULTS` 的真值（求值而不是解析），并断言它与随包预设的 `home` 一致、
`skills` 与 `home` 同级。

## 已验证

    2026-10-24   ping / remember / recall / mission_list / report   经真 dsh-tools 注册表往返成功
                 tool_define 定义 def main(n=0)    registered:true，随 tool_call 返回 [7, 14, 21]
                 tool_define 定义 def main():1/0   registered:false，错误 ZeroDivisionError
                 package.json 过 dsh 自己的 evaluatePluginCompatibility：无未满足项
    2026-11-06   73 个工具全部注册（chk_bridge_catalog.mjs），schema 投影 27.4% 字符节省
    2026-11-07   模型路由会实测"这个模型读不读得到 tool 结果"并拒绝把废模型交出去
    2026-11-11   mission / schedule / system-task / skill / memory / wire 六个板块各有断言套件

## 约束

服务的工具入口是 `main(**kwargs)`，不是 `solve(args)`；定义时 parameters 要传
JSON schema，否则服务只按空参数做样调用。

工具的参数 schema 用 getter 而不是快照：`defineTool()` 在定义时编译一次参数 schema
并永久对那份冻结副本校验，那样造的工具永远不会缩小；DSH 自己的投影相反 ——
每次请求都解构 `definition` 的 `name`/`description`/`parameters`。所以
`dynamicTool()` 返回的对象用 `get description()` / `get parameters()`。
