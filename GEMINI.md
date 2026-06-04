# codex-bridge

OpenAI-compatible local gateway for OpenAI Codex OAuth. This project acts as a bridge, converting OpenAI Chat Completions requests into Codex Responses API calls and streaming the results back in an OpenAI-compatible format.

## Project Overview

- **Core Technologies:** Python 3.11+, aiohttp (server), httpx (client), Typer (CLI), Loguru (logging).
- **Architecture:**
  - `src/codex_bridge/config.py`: Centralised `Settings` dataclass for all configuration (env vars, CLI flags).
  - `src/codex_bridge/server.py`: aiohttp-based HTTP server exposing OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`).
  - `src/codex_bridge/codex.py`: Client for the upstream Codex Responses API, including SSE parsing and state management. Uses a shared `httpx.AsyncClient` connection pool.
  - `src/codex_bridge/conversion.py`: Logic for mapping between OpenAI and Codex message formats and tool calls.
  - `src/codex_bridge/cli.py`: Command-line interface for serving the gateway and managing authentication.
  - `src/codex_bridge/auth.py`: OAuth flow and token management (uses `oauth-cli-kit`).

## Building and Running

### Installation
```bash
pip install -e .
# Or with dev dependencies
pip install -e ".[dev]"
```

### Authentication
```bash
codex-bridge login
```

### Running the Gateway
```bash
codex-bridge serve --host 127.0.0.1 --port 8000
```

### Testing
```bash
pytest
```

## Development Conventions

- **Linting & Formatting:** The project uses `ruff` for linting and formatting. Line length is set to 100.
- **Async First:** The core logic is asynchronous, utilizing `asyncio`, `aiohttp`, and `httpx`.
- **Security:**
  - Never log prompts, content, or tokens.
  - Upstream error messages are scrubbed to prevent leaking sensitive information from the request body.
- **Testing:**
  - Tests use `pytest` and `pytest-asyncio`.
  - Integration tests use `aiohttp.test_utils.TestClient` and `httpx.MockTransport` to simulate upstream responses.
- **Environment Variables:**
  - `CODEX_API_HOST`: Bind host (default: `127.0.0.1`).
  - `CODEX_API_PORT`: Bind port (default: `8000`).
  - `CODEX_API_MODEL`: Fallback model (default: `openai-codex/gpt-5.1-codex`).
  - `CODEX_API_MODELS`: Comma-separated list of supported models. When set, `/v1/models` returns this list and unsupported models are rejected.
  - `CODEX_STREAM_IDLE_TIMEOUT_S`: Idle timeout for streaming (default: `90`).
  - `CODEX_MODELS_TIMEOUT_S`: Timeout for model discovery (default: `10`).
