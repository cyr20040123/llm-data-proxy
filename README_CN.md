# llm-data-proxy

OpenAI 兼容的 LLM 代理，支持 ChatML 会话日志记录。将客户端请求转发到上游 LLM 服务，记录带时间戳的对话历史，同时支持流式和非流式请求。

## 安装

```bash
git clone <repo-url>
cd llm-data-proxy
pip install -e .
```

可选 brotli 压缩支持：

```bash
pip install -e ".[brotli]"
```

## 快速开始

```bash
llm-data-proxy --base-url http://localhost:8000 --api-key sk-your-key --log-chatml multi
```

该命令在 `0.0.0.0:8030` 启动代理，转发请求到 `http://localhost:8000`，并启用 ChatML 日志记录。

然后将 OpenAI 客户端指向 `http://localhost:8030/v1`。

也可以使用：

```bash
python -m llmdataproxy --base-url http://localhost:8000 --api-key sk-your-key
```

## 功能特性

- **透明代理** — `/v1/chat/completions`、`/v1/completions`，以及任意其他端点的通用转发（如 `/v1/models`）
- **流式 & 非流式** — SSE 流式重建与透传
- **ChatML 会话日志** — 基于前缀匹配的多轮对话追踪，输出为带 ISO 时间戳的 JSON 格式
- **持久化配置** — 参数自动保存到 `llm_proxy.yaml`，支持预设分组和基于差异的 RECENT 追踪
- **优雅关闭** — 在 SIGINT/SIGTERM 信号时自动保存未完成的 ChatML 会话
- **可编辑安装** — 通过 `pip install -e .` 安装，使用 `llm-data-proxy` 命令启动

> **重要提示：** 一个 llm-data-proxy 实例仅支持采集**单个 agent** 的对话轨迹，**不支持并发采集**。如果需要同时采集多个 agent 的轨迹，请在不同端口上启动多个 llm-data-proxy 实例。

## 配置

所有参数均可通过命令行、`llm_proxy.yaml` 或两者结合设置。配置文件使用 YAML 格式，支持命名分组。

配置文件查找顺序：
1. 当前工作目录 (`./llm_proxy.yaml`)
2. XDG 配置目录 (`$XDG_CONFIG_HOME/llmdataproxy/llm_proxy.yaml`)
3. 纯默认值（不写入文件）

**解析优先级（从高到低）：**

```
1. CLI 参数
2. --preset 分组        （如果指定了 --preset NAME）
3. DEFAULT 分组         （基线配置）
4. 硬编码默认值
```

> RECENT 分组由程序自动管理（每次运行写入与 DEFAULT 的差异），但**不用于**配置解析。

启动时，代理会记录所有已解析的参数；与 `DEFAULT` 不同的值会标记为 `*`。

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `--host` | `0.0.0.0` | 代理监听地址 |
| `--port` | `8030` | 代理监听端口 |
| `--base-url` | *(必填)* | 上游 LLM 服务地址（如 `http://localhost:8000`）。若缺少 `/v1` 前缀，启动时会自动检测并补全。 |
| `--api-key` | `""` | 上游 API 密钥 — 替换客户端发送的任意密钥 |
| `--log-folder` | `./logs/` | 日志和 ChatML 输出目录 |
| `--log-chatml` | `none` | ChatML 记录模式：`none`（禁用）、`multi`（前缀匹配多轮对话）、`single`（每个请求一条记录） |
| `--session-name` | `sess_MMdd_HHmmss` | 初始 ChatML 会话名称 |
| `--session-path` | *(--log-folder)* | ChatML 输出目录（默认使用 `--log-folder`） |
| `--temperature` | `-1.0` | 当客户端请求中缺少 temperature 时，注入到上游请求的默认 temperature。负值时禁用 |
| `--rl` | `false` | 启用 RL 专用的 ChatML 日志 — 从上游请求 logprobs 和 token_ids，并将其记录在每个 assistant 回复旁 |
| `--default-model` | `None` | 当请求中 `model` 字段为空或为 `"none"` 时的默认模型名称。若未指定且上游可达，则自动从第一个可用模型填充。 |
| `--preset` | `None` | 要加载的 YAML 配置分组名称（如 `DEEPSEEK`）。该分组的值覆盖 RECENT 和 DEFAULT。 |
| `--config-file` | *(自动)* | YAML 配置文件路径。覆盖默认查找逻辑。 |

## 接口

| 方法 | 路径 | 说明 |
|--------|------|-------------|
| GET | `/proxyhealth` | 健康检查 — 返回 `{"status": "ok"}` |
| POST | `/newsession` | 将当前 ChatML 会话保存到文件，切换到新会话。请求体：`{"session_name": "...", "session_path": "..."}` |
| GET | `/session_chats` | 以 ChatML JSON 格式返回所有当前会话（只读，不保存到文件）。需要 `--log-chatml` ≠ `none` |
| POST | `/change_override_model` | 覆盖转发请求中的 `model` 字段，或清除覆盖。请求体：`{"model": "model-id"}` |
| POST | `/hint` | 设置或清除全局提示词。当 assistant 消息数为偶数时，注入到聊天请求中。请求体：`{"hint": "be concise"}` |
| POST | `/v1/chat/completions` | 转发到上游，记录到 ChatML 会话 |
| POST | `/v1/completions` | 旧版 completions 接口 |
| * | `/{path}` | 通用转发 — 将任意其他请求转发到上游 |

## 模型处理

模型解析遵循三级优先级：

```
1. model_override   （POST /change_override_model — 强制覆盖所有请求）
   ↓  未设置
2. default_model    （--default-model 或自动检测 — 填充空 / "none"）
   ↓  未设置或模型已有效
3. 透传              （客户端模型原样发送）
```

### `/change_override_model` — 强制覆盖

设置全局模型覆盖，应用于**每个**请求，无论客户端发送何种模型。

```
POST /change_override_model  {"model": "Qwen/Qwen3.5-4B"}
                              → 覆盖 = "Qwen/Qwen3.5-4B"
客户端发送  "model": "gpt-4"  →  "model": "Qwen/Qwen3.5-4B"
客户端发送  无 model 字段     →  "model": "Qwen/Qwen3.5-4B"
```

```
POST /change_override_model  {"model": ""}  或  {"model": "none"}
                              → 覆盖 = None（已清除，恢复透传）
客户端发送  "model": "gpt-4"  →  "model": "gpt-4"
```

### `--default-model` / 自动检测 — 回退

**仅在**客户端请求的 model 为空 (`""`) 或 `"none"`、或完全省略该字段时填充。**不会**覆盖有效的模型名称。

```
--default-model 未设置 + 上游可达
                         → default_model = GET /v1/models 返回的第一个模型

--default-model Qwen/Qwen3.5-4B
                         → default_model = "Qwen/Qwen3.5-4B"

客户端发送  "model": ""       →  "model": default_model
客户端发送  "model": "none"   →  "model": default_model
客户端发送  无 model 字段     →  "model": default_model
客户端发送  "model": "gpt-4"  →  "model": "gpt-4"  （保留不变）
```

- 两种机制均适用于 `/v1/chat/completions` 和 `/v1/completions`。
- 启动时，代理从上游 `GET /v1/models` 获取可用模型列表并存储在 `app.state.available_models` 中。

## 全局提示词（Hint）

`/hint` 接口设置一个持久的全局提示词，自动注入到转发给上游的 `/v1/chat/completions` 请求中。适用于在多轮对话中强化指令，而无需修改客户端代码。

### 工作原理

1. 代理统计当前请求 `messages` 数组中 `role: "assistant"` 的消息数量。
2. 如果数量为**偶数**（0, 2, 4, …），则在数组末尾追加一条 `{"role": "user", "content": "(Global hint: <hint>)"}` 消息。
3. 如果数量为**奇数**，则不添加任何内容。

已有消息内容不会被修改 — 提示词始终以独立消息的形式追加。

### 使用方式

```bash
# 设置提示词
curl -X POST http://localhost:8030/hint \
  -H "Content-Type: application/json" \
  -d '{"hint": "回答简洁，使用要点列表"}'
# → {"status": "ok", "hint": "回答简洁，使用要点列表"}

# 清除提示词（空字符串或仅空白字符）
curl -X POST http://localhost:8030/hint \
  -H "Content-Type: application/json" \
  -d '{"hint": ""}'
# → {"status": "ok", "hint": null}
```

### 示例

**偶数 assistant 数量 (0) — 追加 hint 消息：**
```json
// 注入前的请求消息：
[{"role": "user", "content": "什么是 Python？"}]

// 注入后（hint = "回答简洁"）：
[
  {"role": "user", "content": "什么是 Python？"},
  {"role": "user", "content": "(Global hint: 回答简洁)"}
]
```

**偶数 assistant 数量 (2) — 追加 hint 消息：**
```json
// 注入前的请求消息：
[
  {"role": "user", "content": "Q1"},
  {"role": "assistant", "content": "A1"},
  {"role": "user", "content": "Q2"},
  {"role": "assistant", "content": "A2"}
]

// 注入后（hint = "回答简洁"）：
[
  {"role": "user", "content": "Q1"},
  {"role": "assistant", "content": "A1"},
  {"role": "user", "content": "Q2"},
  {"role": "assistant", "content": "A2"},
  {"role": "user", "content": "(Global hint: 回答简洁)"}
]
```

**奇数 assistant 数量 — 不注入：**
```json
// 请求消息：
[
  {"role": "user", "content": "你好"},
  {"role": "assistant", "content": "你好！"}
]
// Assistant 数量 = 1（奇数）→ 无变化
```

> **注意：** 提示词会被注入到转发的请求体中，**并且**记录在 ChatML 会话日志中。在前缀匹配过程中，独立的 hint 消息会被跳过 — 因此提示词的存在与否不会影响多轮对话匹配。

## ChatML 输出

当 `--log-chatml` 设置为 `multi` 或 `single` 时，对话历史将保存为 JSON 文件，输出到 `--session-path`（默认使用 `--log-folder`）。

### `multi` 模式（前缀匹配）

对话按前缀匹配分组 — 每个传入请求的消息历史会与已跟踪的会话进行匹配。输出文件名：`{session_name}.chatml.json`（或 `{session_name}_{i}.chatml.json`（当存在多个会话时））。

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有用的助手。", "timestamp": ""},
    {"role": "user", "content": "你好", "timestamp": "2026-05-19T06:04:22.669170+00:00"},
    {"role": "assistant", "content": "你好！", "timestamp": "2026-05-19T06:04:22.685971+00:00"}
  ],
  "remarks": {"incomplete": false}
}
```

### `single` 模式

每个请求/响应对作为独立条目存储。所有条目写入一个文件：`{session_name}.json`。不包含时间戳。

```json
[
  {"messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
  ]}
]
```

### RL 字段（启用 `--rl` 时）

启用 `--rl` 后，每条 assistant 回复消息会新增三个字段，与 `timestamp` 同级：

| 字段 | 来源 | 说明 |
|-------|--------|-------------|
| `prompt_ids` | `prompt_token_ids` | 整个 prompt 的 token ID |
| `completion_ids` | `choices[*].token_ids` | 该回复生成的 token ID |
| `logprobs` | `choices[*].logprobs.content[*].logprob` | 每个生成 token 的对数概率 |

### 通用说明

- 消息使用标准 OpenAI 角色（`system`、`user`、`assistant`、`tool`）
- 每条消息带有 ISO 8601 格式的 `timestamp` — 代理未直接见证的消息该字段为空字符串（`multi` 模式），或始终为空（`single` 模式）
- 当对话在中途被截断时，`remarks.incomplete` 为 `true`
- 工具调用存储在 assistant 消息的 `tool_calls` 字段中
- 工具定义（如有）包含在顶层的 `tools` 数组中
- 会话在调用 `/newsession` 或关闭代理时保存

## 工具

随包安装的额外 CLI 工具：

### llm-data-proxy-new-session

向运行中的代理发送 `/newsession` 请求：

```bash
llm-data-proxy-new-session my_session_name /path/to/output
llm-data-proxy-new-session --host 127.0.0.1 --port 8030 my_session
```

### llm-data-proxy-strip-chatml

从 ChatML JSON 中移除系统消息和非必要字段：

```bash
llm-data-proxy-strip-chatml session.chatml.json --output clean.json
llm-data-proxy-strip-chatml session.chatml.json --keep-system
```

## 测试

```bash
# 安装开发依赖
pip install -e ".[test]"

# 运行所有测试
pytest tests/ -v

# 针对真实上游运行测试
python tests/test_integration.py --real-upstream http://localhost:8030

# 测试已运行的代理
python tests/test_integration.py --proxy-url http://localhost:8031
```

## 项目结构

```
src/llmdataproxy/
├── __init__.py          # 包元数据，公开 API 导出
├── __main__.py          # python -m llmdataproxy 支持
├── main.py              # 入口点，服务启动
├── config.py            # CLI + YAML 配置解析
├── server.py            # FastAPI 应用工厂，路由处理
├── session.py           # SessionManager — 前缀匹配，ChatML 输出
├── sse.py               # SSE 流式重建
├── chatml_schema.json   # ChatML 输出的 JSON Schema
└── tools/
    ├── new_session.py   # CLI：在运行中的代理上触发 /newsession
    └── strip_chatml.py  # CLI：从 ChatML JSON 中移除系统消息
```
