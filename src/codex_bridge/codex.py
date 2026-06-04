"""Codex Responses API client and SSE parser."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import httpx
import json_repair
from loguru import logger

from .auth import CodexToken
from .config import DEFAULT_ORIGINATOR, Settings
from .conversion import build_reasoning_options, convert_messages, convert_tools, strip_model_prefix


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class CodexResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None


class CodexAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, retry_after: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def build_headers(token: CodexToken, originator: str = DEFAULT_ORIGINATOR) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token.access}",
        "chatgpt-account-id": token.account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": originator,
        "User-Agent": "codex-bridge (python)",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


def build_models_headers(token: CodexToken, originator: str = DEFAULT_ORIGINATOR) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token.access}",
        "OpenAI-Beta": "responses=experimental",
        "originator": originator,
        "User-Agent": "codex-bridge (python)",
        "accept": "application/json",
    }


def build_codex_body(payload: dict[str, Any], default_model: str) -> dict[str, Any]:
    model = str(payload.get("model") or default_model)
    instructions, input_items = convert_messages(payload["messages"])
    reasoning_effort = payload.get("reasoning_effort", payload.get("reasoningEffort"))

    body: dict[str, Any] = {
        "model": strip_model_prefix(model),
        "store": False,
        "stream": True,
        "instructions": instructions,
        "input": input_items,
        "text": {"verbosity": "medium"},
        "include": ["reasoning.encrypted_content"],
        "tool_choice": payload.get("tool_choice") or "auto",
        "parallel_tool_calls": True,
    }
    reasoning = build_reasoning_options(reasoning_effort)
    if reasoning:
        body["reasoning"] = reasoning
    tools = convert_tools(payload.get("tools"))
    if tools:
        body["tools"] = tools
    return body


class CodexClient:
    def __init__(
        self,
        *,
        settings: Settings,
        token_provider: Callable[[], CodexToken | None],
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self.token_provider = token_provider
        self._client = httpx.AsyncClient(transport=transport)

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        token = await asyncio.to_thread(self.token_provider)
        if token is None:
            raise CodexAPIError("Codex OAuth token unavailable. Run `codex-bridge login`.", 401)

        try:
            response = await self._client.get(
                self._settings.models_url,
                headers=build_models_headers(token, self._settings.originator),
                timeout=self._settings.models_timeout,
            )
        except httpx.TimeoutException as exc:
            raise CodexAPIError("OpenAI models request timed out", 504) from exc
        except httpx.TransportError as exc:
            raise CodexAPIError("OpenAI models request failed", 503) from exc

        if response.status_code != 200:
            await response.aread()
            raise CodexAPIError(
                f"OpenAI models request failed with HTTP {response.status_code}",
                response.status_code,
                response.headers.get("retry-after"),
            )

        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return []
        models = [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")]
        result = sorted(dict.fromkeys(models))
        logger.debug("Model discovery result: {}", result)
        return result

    async def stream_events(self, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        token = await asyncio.to_thread(self.token_provider)
        if token is None:
            raise CodexAPIError("Codex OAuth token unavailable. Run `codex-bridge login`.", 401)

        logger.debug("Codex request body: {}", json.dumps(body, ensure_ascii=False))

        try:
            async with self._client.stream(
                "POST",
                self._settings.codex_url,
                headers=build_headers(token, self._settings.originator),
                json=body,
                timeout=self._settings.stream_idle_timeout,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise CodexAPIError(
                        f"Codex API request failed with HTTP {response.status_code}",
                        response.status_code,
                        response.headers.get("retry-after"),
                    )
                async for event in iter_sse(response):
                    yield event
        except httpx.TimeoutException as exc:
            raise CodexAPIError("Codex API request timed out", 504) from exc
        except httpx.TransportError as exc:
            raise CodexAPIError("Codex API transport failed", 503) from exc

    async def complete(self, body: dict[str, Any]) -> CodexResult:
        result = CodexResult()
        state = StreamState()
        async for event in self.stream_events(body):
            for output in state.apply(event):
                if output["type"] == "content_delta":
                    result.content += output["delta"]
                elif output["type"] == "reasoning_delta":
                    result.reasoning_content = (result.reasoning_content or "") + output["delta"]
                elif output["type"] == "tool_call_done":
                    result.tool_calls.append(output["tool_call"])
                elif output["type"] == "completed":
                    result.finish_reason = output["finish_reason"]
                    result.usage = output["usage"]
        return result


async def iter_sse(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    buffer: list[str] = []

    def flush() -> dict[str, Any] | None:
        data_lines = [line[5:].strip() for line in buffer if line.startswith("data:")]
        buffer.clear()
        if not data_lines:
            return None
        data = "\n".join(data_lines).strip()
        if not data or data == "[DONE]":
            return None
        try:
            event = json.loads(data)
        except Exception:
            logger.warning("Failed to parse Codex SSE event JSON")
            return None
        return event if isinstance(event, dict) else None

    async for line in response.aiter_lines():
        if line == "":
            if buffer:
                event = flush()
                if event is not None:
                    yield event
            continue
        buffer.append(line)

    if buffer:
        event = flush()
        if event is not None:
            yield event


def map_finish_reason(status: str | None) -> str:
    return {
        "completed": "stop",
        "incomplete": "length",
        "failed": "error",
        "cancelled": "error",
    }.get(status or "completed", "stop")


def parse_usage(response_obj: dict[str, Any]) -> dict[str, int]:
    usage = response_obj.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    return {
        "prompt_tokens": int(usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def make_completion_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{int(time.time() * 1000)}"


class StreamState:
    """Stateful mapper from Codex SSE events to normalized deltas."""

    def __init__(self) -> None:
        self.tool_call_buffers: dict[str, dict[str, Any]] = {}
        self.reasoning_streamed = False

    def apply(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        event_type = event.get("type")
        outputs: list[dict[str, Any]] = []

        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call" and item.get("call_id"):
                call_id = str(item["call_id"])
                self.tool_call_buffers[call_id] = {
                    "item_id": item.get("id") or "fc_0",
                    "name": item.get("name") or "",
                    "arguments": item.get("arguments") or "",
                }
                outputs.append({"type": "tool_call_added", "call_id": call_id, "name": item.get("name") or ""})

        elif event_type == "response.output_text.delta":
            delta = event.get("delta") or ""
            if delta:
                outputs.append({"type": "content_delta", "delta": str(delta)})

        elif event_type == "response.reasoning_summary_text.delta":
            delta = event.get("delta") or ""
            if delta:
                self.reasoning_streamed = True
                outputs.append({"type": "reasoning_delta", "delta": str(delta)})

        elif event_type == "response.function_call_arguments.delta":
            call_id = str(event.get("call_id") or "")
            if call_id in self.tool_call_buffers:
                delta = event.get("delta") or ""
                self.tool_call_buffers[call_id]["arguments"] += delta
                if delta:
                    outputs.append({
                        "type": "tool_call_delta",
                        "call_id": call_id,
                        "name": self.tool_call_buffers[call_id].get("name") or "",
                        "arguments_delta": str(delta),
                    })

        elif event_type == "response.function_call_arguments.done":
            call_id = str(event.get("call_id") or "")
            if call_id in self.tool_call_buffers:
                self.tool_call_buffers[call_id]["arguments"] = event.get("arguments") or ""

        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call" and item.get("call_id"):
                outputs.append({"type": "tool_call_done", "tool_call": self._tool_call_from_item(item)})

        elif event_type == "response.completed":
            response_obj = event.get("response") or {}
            if isinstance(response_obj, dict):
                outputs.append({
                    "type": "completed",
                    "finish_reason": map_finish_reason(response_obj.get("status")),
                    "usage": parse_usage(response_obj),
                })

        elif event_type in {"error", "response.failed"}:
            raise CodexAPIError("Codex API response failed", 502)

        return outputs

    def _tool_call_from_item(self, item: dict[str, Any]) -> ToolCall:
        call_id = str(item.get("call_id") or "")
        buf = self.tool_call_buffers.get(call_id) or {}
        raw_args = buf.get("arguments") or item.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args = json_repair.loads(raw_args) if isinstance(raw_args, str) else raw_args
        if not isinstance(args, dict):
            args = {"raw": raw_args}
        return ToolCall(
            id=f"{call_id}|{buf.get('item_id') or item.get('id') or 'fc_0'}",
            name=str(buf.get("name") or item.get("name") or ""),
            arguments=args,
        )
