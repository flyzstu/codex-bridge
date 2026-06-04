# codex-openai-api

OpenAI-compatible local gateway for OpenAI Codex OAuth.

It exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

The gateway is intentionally small. It converts Chat Completions messages to Codex Responses API input items, streams Codex SSE back as OpenAI Chat Completions chunks, and does not run tools locally.

## Install

```bash
pip install -e .
```

## Login

```bash
codex-openai-api login
```

## Serve

```bash
codex-openai-api serve --host 127.0.0.1 --port 8000
```

Environment defaults:

- `CODEX_API_HOST`
- `CODEX_API_PORT`
- `CODEX_API_MODEL` fallback model when model discovery is unavailable
- `CODEX_STREAM_IDLE_TIMEOUT_S`
- `CODEX_MODELS_TIMEOUT_S`

`GET /v1/models` fetches the model list from OpenAI using the Codex OAuth token. If model
discovery fails, the server still returns the fallback model so OpenAI-compatible clients can
continue to start.

## Curl

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
