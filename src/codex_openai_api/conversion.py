"""Convert OpenAI Chat Completions payloads to Codex Responses payloads."""

from __future__ import annotations

import json
from typing import Any

MODEL_PREFIXES = ("openai-codex/", "openai_codex/")


def strip_model_prefix(model: str) -> str:
    for prefix in MODEL_PREFIXES:
        if model.startswith(prefix):
            return model.split("/", 1)[1]
    return model


def build_reasoning_options(reasoning_effort: str | None) -> dict[str, str] | None:
    if reasoning_effort and reasoning_effort.lower() == "none":
        return {"effort": "none"}
    options = {"summary": "auto"}
    if reasoning_effort:
        options["effort"] = reasoning_effort
    return options


def validate_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("messages must contain only objects")
    if not any(message.get("role") == "user" for message in messages):
        raise ValueError("messages must include at least one user message")
    return messages


def convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    input_items: list[dict[str, Any]] = []
    used_item_ids: set[str] = set()

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue

        if role == "user":
            input_items.append(convert_user_message(content))
            continue

        if role == "assistant":
            if isinstance(content, str) and content:
                message_id = unique_item_id(f"msg_{idx}", used_item_ids)
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                    "status": "completed",
                    "id": message_id,
                })
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                call_id, item_id = split_tool_call_id(tool_call.get("id"))
                response_item_id = unique_item_id(item_id or f"fc_{idx}", used_item_ids)
                input_items.append({
                    "type": "function_call",
                    "id": response_item_id,
                    "call_id": call_id or f"call_{idx}",
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments") or "{}",
                })
            continue

        if role == "tool":
            call_id, _ = split_tool_call_id(msg.get("tool_call_id"))
            output_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            input_items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_text,
            })

    return "\n\n".join(system_parts), input_items


def convert_user_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}
    if isinstance(content, list):
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append({"type": "input_image", "image_url": url, "detail": "auto"})
        if converted:
            return {"role": "user", "content": converted}
    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def convert_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description") or "",
            "parameters": params if isinstance(params, dict) else {},
        })
    return converted


def unique_item_id(item_id: str, used: set[str]) -> str:
    if item_id not in used:
        used.add(item_id)
        return item_id
    suffix = 2
    while f"{item_id}_{suffix}" in used:
        suffix += 1
    unique = f"{item_id}_{suffix}"
    used.add(unique)
    return unique


def split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None
