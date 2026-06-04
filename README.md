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
codex-openai-api serve --host 127.0.0.1 --port 8000 --model openai-codex/gpt-5.1-codex
```

Environment defaults:

- `CODEX_API_HOST`
- `CODEX_API_PORT`
- `CODEX_API_MODEL`
- `CODEX_STREAM_IDLE_TIMEOUT_S`

## Curl

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"openai-codex/gpt-5.1-codex","messages":[{"role":"user","content":"hello"}]}'
```

Streaming:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":"hello"}]}'
```
