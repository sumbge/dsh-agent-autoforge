# dsh-agent-autoforge

让 DSH 的 agent preset 调用 [autoforge](https://github.com/ZhangYangyi03/autoforge) 的工具治理能力：
forge → 六道验证 → 封存，加上跨会话的结构化记忆、mission 账、技能库和留痕账本。

两部分：

- **`lib/` · `presets/`** —— Cordis 插件，把服务的方法声明成 DSH 工具（64 个）
- **`service/`** —— Python stdio JSON-RPC 服务，执行工具治理

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
| `service.py`（约 2370 行） | 66 个 RPC 方法 + stdio 循环 |
| `__main__.py` | 两个入口：stdio 服务、`tick` |
| `tests/` | 17 个测试（注入层、无人值守路径） |

---

## 上游没有的部分

**上游 79 个工具，这里声明 64 个。** 缺的 15 个是刻意的：

| 缺失 | 数量 | 理由 |
| --- | --- | --- |
| `browser_*`、`see`/`draw`/`animate`、`mcp_*` | 9 | DSH 自带等价物 |
| `spawn_agent`、`design_team` | 2 | DSH 有 subagent / workflow |
| `set_autonomy`、`amend_self`、`import_package` | 3 | **机制冲突**：上游是「四类自由度 + confirm 闸」，DSH 是 sandbox + approval，两套权限模型不能共存 |
| `tars_probe`、`tars_skills`、`tars_port_skills` | 3 | TARS 元工具；本机的 `skill_port` 覆盖了实际用途 |

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
| 通知 | webhook 或 SMTP，配在 `~/.autoforge/config.json` | `notify_send` 说明该配什么 |

---

## 配置

```yaml
- id: agent-autoforge-tools
  name: '@local/dsh-agent-autoforge'
  config:
    python: python
    cwd: <服务目录>
    home: <数据目录>            # sqlite 账本、mission、schedule
    skills: <技能目录>          # 与 DSH 的 skill-filesystem 共用
    runs: <journal 目录>        # 中断续跑的 run-*.json
    timeoutMs: 600000
```

`runs` 要单独给，是因为 journal 通常在 CLI 那边（`%LOCALAPPDATA%\autoforge\runs`），
不在 `home` 下面。不指这里，`runs_status` 会读一个空目录并报告 0 条 ——
那比缺功能更难查，因为它看起来像答案。

模型网关默认读 `$DSH_HOME/openai-gateway/api-key`，端点 `127.0.0.1:8326`。
可用环境变量覆盖：`AUTOFORGE_MODEL`、`AUTOFORGE_BASE_URL`、`AUTOFORGE_API_KEY`。

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

---

## 许可

服务层代码源自 [autoforge](https://github.com/ZhangYangyi03/autoforge)，遵循其许可。
`vendor/tars/` 的许可见 `service/autoforge_svc/vendor/PROVENANCE.txt`。
插件侧说明见 [PLUGIN.md](PLUGIN.md)。
