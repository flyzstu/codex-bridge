# codex-bridge

OpenAI-compatible local gateway for OpenAI Codex OAuth.

It exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

The gateway is intentionally small. It converts Chat Completions messages to Codex Responses API input items, streams Codex SSE back as OpenAI Chat Completions chunks, and does not run tools locally.

## Install

From source:
```bash
pip install -e .
```

Or directly from GitHub:
```bash
pip install git+https://github.com/flyzstu/codex-bridge.git
```

## Login

```bash
codex-bridge login
```

## Serve

```bash
codex-bridge serve --host 127.0.0.1 --port 8000
```

### Specifying supported models

Pass `--models` with a comma-separated list to restrict the gateway to a fixed set of models.
When configured, `/v1/models` returns this list directly (no upstream API call) and requests
for unlisted models are rejected with `400 Bad Request`.

```bash
codex-bridge serve --models "gpt-5.5,gpt-5.4,gpt-5.4-mini"
```

If `--models` is omitted, the gateway defaults to supporting `gpt-5.5`, `gpt-5.4`, and `gpt-5.4-mini`.

## Configuration

All settings can be provided via environment variables or CLI flags. CLI flags take precedence.

| Environment Variable | CLI Flag | Default | Description |
|---|---|---|---|
| `CODEX_API_HOST` | `--host` | `127.0.0.1` | Bind host |
| `CODEX_API_PORT` | `--port` | `8000` | Bind port |
| `CODEX_API_MODEL` | `--model` | `gpt-5.5` | Default / fallback model |
| `CODEX_API_MODELS` | `--models` | `gpt-5.5,gpt-5.4,gpt-5.4-mini` | Comma-separated supported model list |
| `CODEX_STREAM_IDLE_TIMEOUT_S` | — | `90` | Idle timeout for streaming (seconds) |
| `CODEX_MODELS_TIMEOUT_S` | — | `10` | Timeout for model discovery (seconds) |

## Usage

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

Streaming:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":"hello"}]}'
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```
