# autoforge-svc（DSH 网关版）

这是 `../autoforge-svc` 的**副本**，唯一区别是 forge 的模型通道换成了 DSH 自带的
OpenAI 兼容网关，不再走 `DSHGatewayClient` 直连腾讯。原件保持不动，两边的数据库
也各自独立。

改了两处，都在 `autoforge_svc/service.py`：

| 位置 | 原件 | 这份 |
| --- | --- | --- |
| `_forge_pipeline()` 的客户端 | `DSHGatewayClient` 直连 `copilot.tencent.com`，盲取 `accounts[0]` | `OpenAICompatClient` 指向 `127.0.0.1:8326/v1`，走 DSH 自己的账号池与轮换 |
| `_forge_pipeline()` 的生成器 | `TemplateGenerator`（离线 demo 桩） | `LLMToolGenerator`（真让模型写代码） |

## 为什么要换客户端

`DSHGatewayClient` 只认 CN 网关、只读 `accounts.json` 的第一个账号、并且必须自己
拼 SSE（网关只接受流式）。账本里 5 条 `forge_error` 全挂在 SSE 拼包上。

DSH 自带网关（`vendor/channel-pack/src/openai-gateway`，默认 `127.0.0.1:8326`）
不一样：

- key 持久化在 `$DSH_HOME/openai-gateway/api-key`，外部进程能读
  （xdpool 的 shim 是 `randomBytes(32)` 每进程现生成，读不到，这是两者的根本区别）
- 接受非流式请求，`OpenAICompatClient` 不用改
- 账号池、401 判死（`penalizeCredentialDead`）、按模型冷却都是 xdpool 现成的

## 为什么要换生成器

`TemplateGenerator` 是个离线桩：任何 need 都返回 `echo_<need>`，函数体是
`return text`。它**能通过全套六道检查**——19/19 鲁棒性、5/5 对抗、trigger 正常触发——
因为每道检查喂给它的输入它都原样吐回来。于是它被 `sealed` 成 `active`。

这正是这套框架存在的理由要防的事。CLI 一直用的是 `LLMToolGenerator`
（`cli._build`），服务这份是唯一用了桩的地方。

## 环境

    AUTOFORGE_HOME    状态目录（默认 %LOCALAPPDATA%\autoforge）
    AUTOFORGE_MODEL   模型 id，默认 xdpool-global/deepseek-v4.1-flash
    AUTOFORGE_BASE_URL 网关地址，默认 http://127.0.0.1:8326/v1
    AUTOFORGE_API_KEY 网关 key；不设则读 $DSH_HOME/openai-gateway/api-key
    DSH_HOME          DSH 家目录，默认 ~/.dsh

模型 id 用 DSH 的 `provider/model` 形式，可选值见
`curl -H "Authorization: Bearer <key>" http://127.0.0.1:8326/v1/models`。

## 跑

    python -m autoforge_svc [db_path]

每行一个 JSON 请求，每行一个 JSON 回答：

    {"id":1,"method":"remember","params":{"key":"k","value":"v"}}
    {"id":1,"ok":true,"result":{"key":"k"}}

## 方法

    ping / describe
    remember(key,value,tags?) / recall(query?,limit?) / forget(key)
    mission_open(text,parent?,next_step?,blocked_on?)
    mission_note(mid,note?,next_step?,blocked_on?)
    mission_close(mid,note?,ok?)
    mission_list(status?) / mission_show(mid)
    log_event(kind,payload?) / events(limit?) / report()
    skills() / skill_view(name)
    tool_define(name,code,description?,parameters?) / tool_list() / tool_call(name,arguments?)
    forge_tool(need,context?) / evolve_tool(...) / evaluate_tool(...)

## 已验证（2026-10-24）

    _forge_route() -> model=xdpool-global/deepseek-v4.1-flash, base=...:8326/v1, key 43 字符
    forge_tool("Convert a temperature from Celsius to Fahrenheit...")
      -> 22.6s, celsius_to_fahrenheit [active], 六道全 PASS
      -> 生成的代码是真正的转换函数（含 None/bool/空串/单位后缀/NaN/Inf 处理）
    forge_tool 用桩生成器时 -> echo_<need>，body 是 return text，也 PASS（这就是问题）

## 与原件的关系

原件 `../autoforge-svc` 仍在跑，`DSHGatewayClient` + `TemplateGenerator` 原样保留。
两个 `.data/autoforge.db` 是独立的：这份从原件复制过来（6 工具 / 79 事件），
之后各走各的。

## 没搬进来

    agent.py（ForgeAgent 本体的循环与元工具注册）
    browser.py（CDP 驱动，直接连浏览器，没必要跨进程搬）
    cli.py / web/server.py / embedded/（终端与 web 控制台，宿主自己有界面）
