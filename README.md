# llm-data-proxy

OpenAI-compatible LLM proxy with ChatML conversation logging. Forwards requests from clients to an upstream LLM service, records conversation history with per-message timestamps, and supports both streaming and non-streaming requests.

## Installation

```bash
pip install llm-data-proxy
```

For brotli compression support (optional):

```bash
pip install llm-data-proxy[brotli]
```

For development:

```bash
git clone <repo-url>
cd llm-data-proxy
pip install -e .
```

## Quick Start

```bash
llm-data-proxy --base-url http://localhost:8000 --api-key sk-your-key --log-chatml multi
```

This starts the proxy on `0.0.0.0:8030`, forwarding to `http://localhost:8000` with ChatML logging enabled.

Then point your OpenAI client at `http://localhost:8030/v1`.

You can also use:

```bash
python -m llmdataproxy --base-url http://localhost:8000 --api-key sk-your-key
```

## Features

- **Transparent proxying** — `/v1/chat/completions`, `/v1/completions`, and catch-all forwarding for any other endpoint (e.g. `/v1/models`)
- **Streaming & non-streaming** — SSE stream reconstruction and passthrough
- **ChatML session logging** — Multi-turn conversation tracking with prefix matching, output as JSON with per-message ISO timestamps
- **Persistent config** — Parameters auto-saved to `llm_proxy.yaml`, with preset groups and diff-based RECENT tracking
- **Graceful shutdown** — Dumps pending ChatML sessions on SIGINT/SIGTERM
- **Pip-installable** — Install with `pip install`, use via `llm-data-proxy` command

## Configuration

All parameters can be set via command line, `llm_proxy.yaml`, or both. The config file uses a YAML structure with named groups.

Config file lookup order:
1. Current working directory (`./llm_proxy.yaml`)
2. XDG config directory (`$XDG_CONFIG_HOME/llmdataproxy/llm_proxy.yaml`)
3. Pure defaults (no file written)

**Resolution priority (highest to lowest):**

```
1. CLI arguments
2. --preset group        (if --preset NAME is specified)
3. DEFAULT group         (baseline configuration)
4. Hardcoded defaults
```

> RECENT group is auto-managed (diffs from DEFAULT are written on each run) but is **not** used for config resolution.

On startup, the proxy logs all resolved parameters; values differing from `DEFAULT` are marked with `*`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--host` | `0.0.0.0` | Proxy listen address |
| `--port` | `8030` | Proxy listen port |
| `--base-url` | *(required)* | Upstream LLM service URL (e.g. `http://localhost:8000`). The `/v1` prefix is auto-detected on startup if missing. |
| `--api-key` | `""` | Upstream API key — replaces any key sent by the client |
| `--log-folder` | `./logs/` | Directory for logs and ChatML output |
| `--log-chatml` | `none` | ChatML recording mode: `none` (disabled), `multi` (prefix-matched multi-turn), `single` (one entry per request) |
| `--session-name` | `sess_MMdd_HHmmss` | Name for the initial ChatML session |
| `--session-path` | *(--log-folder)* | ChatML output directory (defaults to `--log-folder`) |
| `--temperature` | `-1.0` | Default temperature injected into upstream requests when absent from the client request. Disabled when negative |
| `--rl` | `false` | Enable RL-specific ChatML logging — requests logprobs & token_ids from upstream, records them alongside each assistant response |
| `--default-model` | `None` | Default model name when the request's `model` field is empty or `"none"`. If not specified and upstream is reachable, auto-populated from the first available model. |
| `--preset` | `None` | Name of a YAML config group to load (e.g. `DEEPSEEK`). Values from this group override RECENT and DEFAULT. |
| `--config-file` | *(auto)* | Path to YAML config file. Overrides the default lookup. |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/proxyhealth` | Health check — returns `{"status": "ok"}` |
| POST | `/newsession` | Dump current ChatML sessions to file, switch to a new session. Body: `{"session_name": "...", "session_path": "..."}` |
| GET | `/session_chats` | Return all current sessions in ChatML JSON format (read-only, does not dump to file). Requires `--log-chatml` ≠ `none` |
| POST | `/change_override_model` | Override the `model` field in forwarded requests, or clear the override. Body: `{"model": "model-id"}` |
| POST | `/hint` | Set or clear a global hint injected into chat requests when the assistant message count is even. Body: `{"hint": "be concise"}` |
| POST | `/v1/chat/completions` | Forward to upstream, record in ChatML session |
| POST | `/v1/completions` | Legacy completions endpoint |
| * | `/{path}` | Catch-all — forwards any other request to upstream |

## Model Handling

Model resolution follows a three-tier priority:

```
1. model_override   (POST /change_override_model — forces all requests)
   ↓  not set
2. default_model    (--default-model or auto-detected — fills empty / "none")
   ↓  not set or model already valid
3. pass-through      (client's model sent as-is)
```

### `/change_override_model` — Force Override

Sets a global model override applied to **every** request, regardless of what the client sends.

```
POST /change_override_model  {"model": "Qwen/Qwen3.5-4B"}
                              → override = "Qwen/Qwen3.5-4B"
client sends  "model": "gpt-4"  →  "model": "Qwen/Qwen3.5-4B"
client sends  no model field    →  "model": "Qwen/Qwen3.5-4B"
```

```
POST /change_override_model  {"model": ""}  or  {"model": "none"}
                              → override = None (cleared, back to pass-through)
client sends  "model": "gpt-4"  →  "model": "gpt-4"
```

### `--default-model` / Auto-detect — Fallback

Fills in the model name **only when** the client request has an empty (`""`) or `"none"` model, or omits the field entirely. Does **not** override a valid model name.

```
--default-model not set + upstream reachable
                         → default_model = first model from GET /v1/models

--default-model Qwen/Qwen3.5-4B
                         → default_model = "Qwen/Qwen3.5-4B"

client sends  "model": ""       →  "model": default_model
client sends  "model": "none"   →  "model": default_model
client sends  no model field    →  "model": default_model
client sends  "model": "gpt-4"  →  "model": "gpt-4"  (preserved)
```

- Both mechanisms apply to `/v1/chat/completions` and `/v1/completions`.
- On startup, the proxy fetches the available model list from upstream `GET /v1/models` and stores it in `app.state.available_models`.

## Global Hint

The `/hint` endpoint sets a persistent global hint that is automatically injected into `/v1/chat/completions` requests forwarded to the upstream. It is useful for reinforcing instructions across multi-turn conversations without modifying client code.

### How it works

1. The proxy counts how many `role: "assistant"` messages are in the current request's `messages` array.
2. If the count is **even** (0, 2, 4, …), a new `{"role": "user", "content": "(Global hint: <hint>)"}` message is appended to the end of the array.
3. If the count is **odd**, nothing is added.

Existing message content is never modified — the hint is always a standalone message.

### Usage

```bash
# Set a hint
curl -X POST http://localhost:8030/hint \
  -H "Content-Type: application/json" \
  -d '{"hint": "be concise and use bullet points"}'
# → {"status": "ok", "hint": "be concise and use bullet points"}

# Clear the hint (empty string or whitespace-only)
curl -X POST http://localhost:8030/hint \
  -H "Content-Type: application/json" \
  -d '{"hint": ""}'
# → {"status": "ok", "hint": null}
```

### Examples

**Even assistant count (0) — new hint message appended:**
```json
// Request messages before injection:
[{"role": "user", "content": "What is Python?"}]

// After injection (hint = "be concise"):
[
  {"role": "user", "content": "What is Python?"},
  {"role": "user", "content": "(Global hint: be concise)"}
]
```

**Even assistant count (2) — new hint message appended:**
```json
// Request messages before injection:
[
  {"role": "user", "content": "Q1"},
  {"role": "assistant", "content": "A1"},
  {"role": "user", "content": "Q2"},
  {"role": "assistant", "content": "A2"}
]

// After injection (hint = "be concise"):
[
  {"role": "user", "content": "Q1"},
  {"role": "assistant", "content": "A1"},
  {"role": "user", "content": "Q2"},
  {"role": "assistant", "content": "A2"},
  {"role": "user", "content": "(Global hint: be concise)"}
]
```

**Odd assistant count — no injection:**
```json
// Request messages:
[
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi!"}
]
// Assistant count = 1 (odd) → no change
```

> **Note:** The hint is injected into the forwarded request body *and* recorded in ChatML session logs. During prefix matching, standalone hint messages are skipped — so the presence or absence of hints does not affect multi-turn conversation matching.

## ChatML Output

When `--log-chatml` is set to `multi` or `single`, conversation history is saved as JSON files in `--session-path` (defaults to `--log-folder`).

### `multi` mode (prefix-matched)

Conversations are grouped by prefix matching — each incoming request's message history is matched against tracked sessions. Output filename: `{session_name}.chatml.json` (or `{session_name}_{i}.chatml.json` when multiple sessions exist).

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful.", "timestamp": ""},
    {"role": "user", "content": "Hello", "timestamp": "2026-05-19T06:04:22.669170+00:00"},
    {"role": "assistant", "content": "Hi there!", "timestamp": "2026-05-19T06:04:22.685971+00:00"}
  ],
  "remarks": {"incomplete": false}
}
```

### `single` mode

Each request/response pair is stored as a separate entry. All entries are written to one file: `{session_name}.json`. Timestamps are omitted.

```json
[
  {"messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]}
]
```

### RL fields (when `--rl` is enabled)

With `--rl`, each assistant response message gains three additional fields at the same level as `timestamp`:

| Field | Source | Description |
|-------|--------|-------------|
| `prompt_ids` | `prompt_token_ids` | Token IDs of the entire prompt |
| `completion_ids` | `choices[*].token_ids` | Token IDs generated for this response |
| `logprobs` | `choices[*].logprobs.content[*].logprob` | Log-probability for each generated token |

### General notes

- Messages use standard OpenAI roles (`system`, `user`, `assistant`, `tool`)
- Each message has an ISO 8601 `timestamp` — empty string for messages the proxy didn't directly witness (`multi` mode) or always empty (`single` mode)
- `remarks.incomplete` is `true` when a conversation was cut off mid-turn
- Tool calls are stored in the `tool_calls` field of assistant messages
- Tool definitions are included in the top-level `tools` array when present
- Sessions are dumped when `/newsession` is called or on shutdown

## Tools

Additional CLI tools installed with the package:

### llm-data-proxy-new-session

Send a `/newsession` request to a running proxy:

```bash
llm-data-proxy-new-session my_session_name /path/to/output
llm-data-proxy-new-session --host 127.0.0.1 --port 8030 my_session
```

### llm-data-proxy-strip-chatml

Strip system messages and non-essential fields from ChatML JSON:

```bash
llm-data-proxy-strip-chatml session.chatml.json --output clean.json
llm-data-proxy-strip-chatml session.chatml.json --keep-system
```

## Testing

```bash
# Install dev dependencies
pip install -e ".[test]"

# Run all tests
pytest tests/ -v

# Run against a real upstream
python tests/test_integration.py --real-upstream http://localhost:8030

# Test an already-running proxy
python tests/test_integration.py --proxy-url http://localhost:8031
```

## Package Structure

```
src/llmdataproxy/
├── __init__.py          # Package metadata, public API exports
├── __main__.py          # python -m llmdataproxy support
├── main.py              # Entry point, server startup
├── config.py            # CLI + YAML config resolution
├── server.py            # FastAPI app factory, route handlers
├── session.py           # SessionManager — prefix matching, ChatML output
├── sse.py               # SSE stream reconstruction
├── chatml_schema.json   # JSON Schema for ChatML output
└── tools/
    ├── new_session.py   # CLI: trigger /newsession on a running proxy
    └── strip_chatml.py  # CLI: strip system messages from ChatML JSON
```
