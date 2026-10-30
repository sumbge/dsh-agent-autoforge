# @local/dsh-agent-autoforge

把 autoforge 服务挂成 DSH 的 agent 预设。预设 id 为 `autoforge`，选中后模型多出
61 个 `autoforge_*` 工具，全部经由本机 `autoforge-svc` 的 stdio JSON-RPC 往返。

工具按用途分九组：

    记忆      remember / recall / forget
    使命      mission_open / note / close / show / list / focus / block / sweep /
              wake / history
    日程      schedule_add / list / tick / done / cancel / install_system_task /
              system_task_command
    技能      skill_write / skill_view / skills / skill_list / skill_errors /
              skill_forget
    锻造      forge_tool / evolve_tool / evaluate_tool / find_gaps / retire_tool /
              list_tools / describe_tool(*) / tool_define / tool_call / tool_list /
              tool_usage
    运行      run_python / runs_status(*) / verify_ledger / my_history / events /
              log_event / describe / ping / report
    自我      my_capabilities / vision_status(*) / media_status(*) / skill_port(*)
    算力      gpu_* (6) / cpu_* (7)
    通知      notify_channels / notify_send

(*) 标出的在 autoforge-svc-gw 里没有对应方法，注册了只会返回 error，所以
**没有**挂：`describe_tool`、`runs_status`、`vision_status`、`media_status`、
`skill_port`、`tars_probe`、`tars_skills`、`tars_port_skills`。

## 故意不挂的

agent.py 有 79 个工具，这里挂 61 个。差的那批不是漏了，是 DSH 自己已经有一份，
同时挂会给模型两条通往同一件事的路：

    browser_* (8)                DSH 自带 browser 工具
    see / draw / animate         DSH 自带同名工具
    spawn_agent / design_team    DSH 有 subagent / subagent_fork / workflow-ptc
    mcp_servers / mcp_connect / mcp_call
                                 DSH 有 MCP 客户端和 manager，服务侧没有 MCP 配置
    resume_run / runs_forget     DSH 自己管 session 与 resume
    amend_self / set_autonomy / import_package
                                 DSH 的自治模型是 approval policy，不是这套 freedom

## 每轮注入

`context_brief` 经 `system-prompt/assemble` 每轮带进请求（事实、使命、工具计数），
刷新 5s 一次。服务挂了就丢掉这一段并 warn 一次，工具照常可用。

## 服务

插件启动时 spawn：

    python -m autoforge_svc      cwd: <config.cwd>
    AUTOFORGE_HOME=<config.home>

进程死了会自动重启一次；请求超时由 `timeoutMs` 控制。

## 配置（preset 行里的 config）

字段      默认值                                             含义
python    python                                              解释器
cwd       C:\\Users\\32535\\Desktop\\dsh\\autoforge-svc        服务包所在目录
home      <cwd>\\.data                                          AUTOFORGE_HOME，状态库落在这里
timeoutMs 120000                                              单次 RPC 超时

## 已验证（2026-10-24，本机）

    ping / remember / recall / mission_list / report   经真 dsh-tools 注册表往返成功
    tool_define 定义 def main(n=0)    registered:true，随 tool_call 返回 [7, 14, 21]
    tool_define 定义 def main():1/0   registered:false，错误 ZeroDivisionError
    package.json 过 dsh 自己的 evaluatePluginCompatibility：无未满足项

## 约束

服务的工具入口是 `main(**kwargs)`，不是 `solve(args)`；定义时 parameters 要传
JSON schema，否则服务只按空参数做样调用。
