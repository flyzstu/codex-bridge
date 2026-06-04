"""aiohttp OpenAI-compatible server."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from aiohttp import web
from loguru import logger

from .auth import load_token
from .codex import (
    CodexAPIError,
    CodexClient,
    StreamState,
    ToolCall,
    build_codex_body,
    make_completion_id,
)
from .conversion import validate_messages

DEFAULT_MODEL_KEY = web.AppKey("default_model", str)
CODEX_CLIENT_KEY = web.AppKey("codex_client", CodexClient)


def create_app(
    *,
    default_model: str,
    codex_client: CodexClient | None = None,
) -> web.Application:
    app = web.Application()
    app[DEFAULT_MODEL_KEY] = default_model
    app[CODEX_CLIENT_KEY] = codex_client or CodexClient(token_provider=load_token)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat_completions)
    return app


async def health(request: web.Request) -> web.Response:
    _ = request
    return web.json_response({"status": "ok"})


async def models(request: web.Request) -> web.Response:
    model = request.app[DEFAULT_MODEL_KEY]
    now = int(time.time())
    return web.json_response({
        "object": "list",
        "data": [{"id": model, "object": "model", "created": now, "owned_by": "openai-codex"}],
    })


async def chat_completions(request: web.Request) -> web.StreamResponse:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        validate_messages(payload.get("messages"))
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception:
        return error_response("invalid JSON request body", 400)

    model = str(payload.get("model") or request.app[DEFAULT_MODEL_KEY])
    try:
        body = build_codex_body(payload, request.app[DEFAULT_MODEL_KEY])
        if bool(payload.get("stream")):
            return await stream_chat_completion(request, body, model)
        return await complete_chat_completion(request, body, model)
    except CodexAPIError as exc:
        return error_response(str(exc), map_status(exc.status_code), retry_after=exc.retry_after)
    except Exception as exc:
        logger.warning("Chat completion request failed: type={}", type(exc).__name__)
        return error_response("upstream request failed", 502)


async def complete_chat_completion(
    request: web.Request,
    body: dict[str, Any],
    model: str,
) -> web.Response:
    client = request.app[CODEX_CLIENT_KEY]
    result = await client.complete(body)
    message: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.tool_calls:
        message["tool_calls"] = [openai_tool_call(tool) for tool in result.tool_calls]
    if result.reasoning_content:
        message["reasoning_content"] = result.reasoning_content

    return web.json_response({
        "id": make_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": result.finish_reason,
        }],
        "usage": result.usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


async def stream_chat_completion(
    request: web.Request,
    body: dict[str, Any],
    model: str,
) -> web.StreamResponse:
    client = request.app[CODEX_CLIENT_KEY]
    token = await asyncio.to_thread(client.token_provider)
    if token is None:
        raise CodexAPIError("Codex OAuth token unavailable. Run `codex-openai-api login`.", 401)

    completion_id = make_completion_id()
    created = int(time.time())
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)

    state = StreamState()
    await write_sse(response, {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })

    tool_indices: dict[str, int] = {}
    finish_reason = "stop"
    async for event in client.stream_events(body):
        for output in state.apply(event):
            if output["type"] == "content_delta":
                await write_sse(response, chunk(completion_id, created, model, {"content": output["delta"]}))
            elif output["type"] == "reasoning_delta":
                await write_sse(response, chunk(
                    completion_id,
                    created,
                    model,
                    {"reasoning_content": output["delta"]},
                ))
            elif output["type"] == "tool_call_added":
                idx = tool_indices.setdefault(output["call_id"], len(tool_indices))
                await write_sse(response, chunk(completion_id, created, model, {
                    "tool_calls": [{
                        "index": idx,
                        "id": output["call_id"],
                        "type": "function",
                        "function": {"name": output["name"], "arguments": ""},
                    }],
                }))
            elif output["type"] == "tool_call_delta":
                idx = tool_indices.setdefault(output["call_id"], len(tool_indices))
                await write_sse(response, chunk(completion_id, created, model, {
                    "tool_calls": [{
                        "index": idx,
                        "function": {"arguments": output["arguments_delta"]},
                    }],
                }))
            elif output["type"] == "completed":
                finish_reason = output["finish_reason"]

    await write_sse(response, {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    })
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


def openai_tool_call(tool: ToolCall) -> dict[str, Any]:
    call_id = tool.id.split("|", 1)[0]
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool.name,
            "arguments": json.dumps(tool.arguments, ensure_ascii=False),
        },
    }


def chunk(completion_id: str, created: int, model: str, delta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


async def write_sse(response: web.StreamResponse, payload: dict[str, Any]) -> None:
    await response.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"))


def error_response(message: str, status: int, retry_after: str | None = None) -> web.Response:
    headers = {"Retry-After": retry_after} if retry_after else None
    return web.json_response(
        {"error": {"message": message, "type": "codex_gateway_error", "code": None}},
        status=status,
        headers=headers,
    )


def map_status(status: int | None) -> int:
    if status in {401, 403}:
        return 401
    if status == 429:
        return 429
    if status == 504:
        return 504
    if status and status >= 500:
        return 502
    return status or 502
