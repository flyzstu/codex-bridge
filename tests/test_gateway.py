from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from aiohttp.test_utils import TestClient, TestServer
from loguru import logger

from codex_bridge.auth import CodexToken
from codex_bridge.codex import CodexClient, build_codex_body, iter_sse
from codex_bridge.config import Settings
from codex_bridge.conversion import convert_messages, strip_model_prefix
from codex_bridge.server import create_app

DEFAULT_MODEL = "openai-codex/gpt-5.1-codex"


def _settings(**overrides: object) -> Settings:
    """Create a Settings instance with sensible test defaults."""
    defaults: dict[str, object] = {"default_model": DEFAULT_MODEL}
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def sse(*events: dict[str, object]) -> bytes:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


def token_provider() -> CodexToken:
    return CodexToken(account_id="acct-test", access="token-test")


def client_for(handler, **settings_kw: object) -> CodexClient:
    return CodexClient(
        settings=_settings(**settings_kw),
        token_provider=token_provider,
        transport=httpx.MockTransport(handler),
    )


@pytest.fixture
async def test_client():
    clients: list[TestClient] = []

    async def factory(codex_client: CodexClient, **settings_kw: object) -> TestClient:
        settings = _settings(**settings_kw)
        app = create_app(settings=settings, codex_client=codex_client)
        client = TestClient(TestServer(app))
        await client.start_server()
        clients.append(client)
        return client

    yield factory

    for client in clients:
        await client.close()


def test_multi_message_conversion_preserves_order() -> None:
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second", "tool_calls": [{
            "id": "call_a|fc_a",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_a|fc_a", "content": {"ok": True}},
        {"role": "user", "content": "third"},
    ]

    instructions, items = convert_messages(messages)

    assert instructions == "be brief"
    assert [item.get("role") or item.get("type") for item in items] == [
        "user",
        "assistant",
        "function_call",
        "function_call_output",
        "user",
    ]
    assert items[3]["call_id"] == "call_a"


def test_model_prefix_is_stripped() -> None:
    assert strip_model_prefix(DEFAULT_MODEL) == "gpt-5.1-codex"
    body = build_codex_body({"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": "hi"}]}, DEFAULT_MODEL)
    assert body["model"] == "gpt-5.1-codex"


async def test_missing_token_returns_401(test_client) -> None:
    codex_client = CodexClient(
        settings=_settings(),
        token_provider=lambda: None,
        transport=httpx.MockTransport(lambda request: None),
    )
    client = await test_client(codex_client)

    response = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "secret prompt"}],
    })

    assert response.status == 401
    payload = await response.json()
    assert "login" in payload["error"]["message"]


async def test_models_endpoint(test_client) -> None:
    client = await test_client(client_for(lambda request: httpx.Response(500)))

    response = await client.get("/v1/models")

    assert response.status == 200
    payload = await response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == DEFAULT_MODEL


async def test_models_endpoint_uses_openai_models(test_client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(200, json={
            "object": "list",
            "data": [
                {"id": "gpt-5.1-codex"},
                {"id": "gpt-5.1"},
                {"id": "gpt-5.1-codex"},
            ],
        })

    client = await test_client(client_for(handler))

    response = await client.get("/v1/models")

    assert response.status == 200
    payload = await response.json()
    assert [model["id"] for model in payload["data"]] == ["gpt-5.1", "gpt-5.1-codex"]


async def test_models_endpoint_returns_configured_models(test_client) -> None:
    """When settings.models is configured, /v1/models returns that list without calling upstream."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"object": "list", "data": []})

    configured_models = ["model-a", "model-b", "model-c"]
    client = await test_client(
        client_for(handler, models=configured_models),
        models=configured_models,
    )

    response = await client.get("/v1/models")

    assert response.status == 200
    payload = await response.json()
    assert [m["id"] for m in payload["data"]] == configured_models
    assert call_count == 0  # upstream was never called


async def test_non_streaming_codex_sse_aggregates_to_chat_completion(test_client) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(
            {"type": "response.output_text.delta", "delta": "hel"},
            {"type": "response.output_text.delta", "delta": "lo"},
            {"type": "response.completed", "response": {
                "status": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            }},
        ))

    client = await test_client(client_for(handler))
    response = await client.post("/v1/chat/completions", json={
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "there"},
            {"role": "user", "content": "again"},
        ],
    })

    assert response.status == 200
    payload = await response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "hello"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["total_tokens"] == 5
    assert captured["body"]["model"] == "gpt-5.1-codex"
    assert [item.get("role") or item.get("type") for item in captured["body"]["input"]] == [
        "user",
        "assistant",
        "user",
    ]


async def test_streaming_codex_sse_converts_to_openai_chunks(test_client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse(
            {"type": "response.output_text.delta", "delta": "A"},
            {"type": "response.output_text.delta", "delta": "B"},
            {"type": "response.completed", "response": {"status": "completed"}},
        ))

    client = await test_client(client_for(handler))
    response = await client.post("/v1/chat/completions", json={
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    })

    assert response.status == 200
    text = await response.text()
    lines = [line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")]
    chunks = [json.loads(line) for line in lines if line != "[DONE]"]

    assert lines[-1] == "[DONE]"
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["choices"][0]["delta"] == {"content": "A"}
    assert chunks[2]["choices"][0]["delta"] == {"content": "B"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_tool_calls_are_returned_without_execution(test_client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse(
            {"type": "response.output_item.added", "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "",
            }},
            {"type": "response.function_call_arguments.delta", "call_id": "call_1", "delta": '{"q"'},
            {"type": "response.function_call_arguments.done", "call_id": "call_1", "arguments": '{"q":"x"}'},
            {"type": "response.output_item.done", "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"q":"x"}',
            }},
            {"type": "response.completed", "response": {"status": "completed"}},
        ))

    client = await test_client(client_for(handler))
    response = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
    })

    payload = await response.json()
    assert response.status == 200
    tool_call = payload["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["id"] == "call_1"
    assert tool_call["function"]["name"] == "lookup"
    assert json.loads(tool_call["function"]["arguments"]) == {"q": "x"}


@pytest.mark.parametrize(("upstream_status", "expected_status"), [
    (429, 429),
    (401, 401),
    (403, 401),
    (500, 502),
])
async def test_upstream_errors_are_mapped(test_client, upstream_status: int, expected_status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(upstream_status, content=b"do not leak body", headers={"retry-after": "7"})

    client = await test_client(client_for(handler))
    response = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "secret prompt"}],
    })

    assert response.status == expected_status
    assert response.headers.get("Retry-After") == "7"
    payload = await response.json()
    assert "secret prompt" not in payload["error"]["message"]


async def test_timeout_maps_to_504(test_client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    client = await test_client(client_for(handler))
    response = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "secret prompt"}],
    })

    assert response.status == 504


async def test_dynamic_model_switching(test_client) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": {"status": "completed"}},
        ))

    client = await test_client(client_for(handler))

    # Request with a custom model name
    response = await client.post("/v1/chat/completions", json={
        "model": "my-custom-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status == 200
    payload = await response.json()
    assert payload["model"] == "my-custom-model"
    assert captured["body"]["model"] == "my-custom-model"

    # Request with a prefixed model name
    response = await client.post("/v1/chat/completions", json={
        "model": "openai-codex/gpt-5.1-codex",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status == 200
    payload = await response.json()
    assert payload["model"] == "openai-codex/gpt-5.1-codex"
    assert captured["body"]["model"] == "gpt-5.1-codex"


async def test_unsupported_model_returns_400(test_client) -> None:
    """When models are configured, requesting an unlisted model returns 400."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse(
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": {"status": "completed"}},
        ))

    configured_models = ["model-a", "model-b"]
    client = await test_client(
        client_for(handler, models=configured_models),
        models=configured_models,
    )

    # Unsupported model → 400
    response = await client.post("/v1/chat/completions", json={
        "model": "model-unknown",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status == 400
    payload = await response.json()
    assert "not supported" in payload["error"]["message"]

    # Supported model → 200
    response = await client.post("/v1/chat/completions", json={
        "model": "model-a",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status == 200


async def test_logs_do_not_include_prompt_or_token() -> None:
    logs: list[str] = []
    sink_id = logger.add(logs.append)
    try:
        response = httpx.Response(200, content=b'data: {"prompt":"secret prompt","token":"token-test"\n\n')
        events = [event async for event in iter_sse(response)]
    finally:
        logger.remove(sink_id)

    assert events == []
    rendered = "\n".join(logs)
    assert "secret prompt" not in rendered
    assert "token-test" not in rendered


async def test_request_logs_include_metadata_without_prompt(test_client) -> None:
    logs: list[str] = []
    sink_id = logger.add(logs.append)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse(
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": {"status": "completed"}},
        ))

    try:
        client = await test_client(client_for(handler))
        response = await client.post("/v1/chat/completions", json={
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "secret prompt"}],
        })
    finally:
        logger.remove(sink_id)

    assert response.status == 200
    rendered = "\n".join(logs)
    assert "HTTP request:" in rendered
    assert "Chat completion request:" in rendered
    assert "model=openai-codex/gpt-5.1-codex" in rendered
    assert "messages=1" in rendered
    assert "secret prompt" not in rendered
