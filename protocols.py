"""协议适配层：OpenAI 兼容 / Anthropic Messages / Ollama 原生 /api/chat。

内部统一使用 OpenAI 风格的中间表示（messages 数组 + tool_calls 结构），
由本模块负责请求转换与响应归一化，上层 conversation/ai_task 无需感知协议差异。
"""

# 部分实现参考 Local OpenAI LLM（Apache-2.0）：
# https://github.com/skye-harris/hass_local_openai_llm

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import aiohttp

_LOGGER = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_DEFAULT_MAX_TOKENS = 1024


# ----------------------------------------------------------------------
# URL
# ----------------------------------------------------------------------
def _trim(base_url: str) -> str:
    return (base_url or "").rstrip("/")


def chat_url(kind: str, base_url: str) -> str:
    """Return the chat endpoint URL for a provider kind."""
    base = _trim(base_url)
    if kind == "anthropic":
        if base.endswith("/messages"):
            return base
        if base.endswith("/v1"):
            return f"{base}/messages"
        return f"{base}/v1/messages"
    if kind == "ollama":
        if base.endswith("/api/chat"):
            return base
        if base.endswith("/api"):
            return f"{base}/chat"
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return f"{base}/api/chat"
    return f"{base}/chat/completions"


def models_url(kind: str, base_url: str) -> str:
    """Return the model-list endpoint URL for a provider kind."""
    base = _trim(base_url)
    if kind == "anthropic":
        return f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
    if kind == "ollama":
        if base.endswith("/api/tags"):
            return base
        if base.endswith("/api"):
            return f"{base}/tags"
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return f"{base}/api/tags"
    return f"{base}/models"


def auth_headers(kind: str, api_key: str) -> dict[str, str]:
    """Return authentication headers for a provider kind."""
    headers: dict[str, str] = {}
    if kind == "anthropic":
        if api_key:
            headers["x-api-key"] = api_key
        headers["anthropic-version"] = ANTHROPIC_VERSION
    elif not api_key:
        return headers
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# ----------------------------------------------------------------------
# 请求体转换
# ----------------------------------------------------------------------
def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return ""


def _gen_tool_id(index: int) -> str:
    return f"call_{index}"


def _image_block(part: dict[str, Any]) -> dict[str, Any] | None:
    image = part.get("image_url") or {}
    url = image.get("url") or ""
    if not url:
        return None
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header[5:].split(";")[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _append_user(messages: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> None:
    """Append user blocks, merging into the previous user message if needed."""
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"].extend(blocks)
        return
    messages.append({"role": "user", "content": blocks})


def to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Convert OpenAI-style messages to (system_parts, anthropic messages)."""
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            system_parts.append(_as_text(message.get("content")))
        elif role == "user":
            content = message.get("content")
            if isinstance(content, list):
                blocks: list[dict[str, Any]] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        blocks.append({"type": "text", "text": part.get("text") or ""})
                    elif part.get("type") == "image_url":
                        block = _image_block(part)
                        if block:
                            blocks.append(block)
                _append_user(converted, blocks or [{"type": "text", "text": ""}])
            else:
                _append_user(converted, [{"type": "text", "text": content or ""}])
        elif role == "assistant":
            blocks = []
            if message.get("content"):
                blocks.append({"type": "text", "text": message["content"]})
            for index, tool_call in enumerate(message.get("tool_calls") or []):
                function = tool_call.get("function") or {}
                raw_args = function.get("arguments")
                try:
                    args = (
                        json.loads(raw_args)
                        if isinstance(raw_args, str)
                        else (raw_args or {})
                    )
                except ValueError:
                    args = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id") or _gen_tool_id(index),
                        "name": function.get("name") or "",
                        "input": args,
                    }
                )
            converted.append(
                {"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]}
            )
        elif role == "tool":
            _append_user(
                converted,
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id") or "",
                        "content": message.get("content") or "",
                    }
                ],
            )

    # Anthropic 要求首条为 user；若历史以 assistant 开头则补一条空 user
    if converted and converted[0]["role"] == "assistant":
        converted.insert(0, {"role": "user", "content": [{"type": "text", "text": ""}]})
    return system_parts, converted


def to_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert OpenAI tool definitions to Anthropic tool definitions."""
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function") or {}
        converted.append(
            {
                "name": function.get("name") or "",
                "description": (function.get("description") or "")[:1024],
                "input_schema": function.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return converted


def to_ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style messages to Ollama /api/chat messages."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role in ("system", "user"):
            content = message.get("content")
            if isinstance(content, list):
                text_parts: list[str] = []
                images: list[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        text_parts.append(part.get("text") or "")
                    elif part.get("type") == "image_url":
                        block = _image_block(part)
                        if block:
                            images.append(block["source"].get("data") or block["source"].get("url") or "")
                payload: dict[str, Any] = {
                    "role": role,
                    "content": " ".join(text_parts),
                }
                if images:
                    payload["images"] = images
                converted.append(payload)
            else:
                converted.append({"role": role, "content": content or ""})
        elif role == "assistant":
            payload = {"role": "assistant", "content": message.get("content") or ""}
            if message.get("tool_calls"):
                tool_calls = []
                for tool_call in message["tool_calls"]:
                    function = tool_call.get("function") or {}
                    raw_args = function.get("arguments")
                    try:
                        args = (
                            json.loads(raw_args)
                            if isinstance(raw_args, str)
                            else (raw_args or {})
                        )
                    except ValueError:
                        args = {}
                    tool_calls.append(
                        {"function": {"name": function.get("name") or "", "arguments": args}}
                    )
                payload["tool_calls"] = tool_calls
            converted.append(payload)
        elif role == "tool":
            converted.append(
                {"role": "tool", "content": message.get("content") or ""}
            )
    return converted


def prepare_body(
    kind: str,
    body: dict[str, Any],
    *,
    thinking_mode: str = "",
    thinking_budget: int | None = None,
    iteration: int = 0,
) -> dict[str, Any]:
    """Convert an OpenAI-style request body for the target protocol."""
    if kind == "anthropic":
        return _prepare_anthropic(
            body,
            thinking_mode=thinking_mode,
            thinking_budget=thinking_budget,
            iteration=iteration,
        )
    if kind == "ollama":
        return _prepare_ollama(
            body,
            thinking_mode=thinking_mode,
            iteration=iteration,
        )
    return body


def _thinking_enabled(mode: str, iteration: int) -> bool | None:
    if mode in ("", "none"):
        return None
    if mode == "first_iteration" and iteration > 0:
        return False
    return True


def _prepare_anthropic(
    body: dict[str, Any],
    *,
    thinking_mode: str,
    thinking_budget: int | None,
    iteration: int,
) -> dict[str, Any]:
    system_parts, messages = to_anthropic_messages(body.get("messages") or [])
    max_tokens = int(body.get("max_tokens") or ANTHROPIC_DEFAULT_MAX_TOKENS)
    request: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if system_parts:
        request["system"] = "\n".join(part for part in system_parts if part)
    if body.get("temperature") is not None:
        # Anthropic 的 temperature 上限为 1
        request["temperature"] = min(float(body["temperature"]), 1.0)
    if body.get("stream"):
        request["stream"] = True

    tools = body.get("tools")
    if tools:
        request["tools"] = to_anthropic_tools(tools)
        if body.get("parallel_tool_calls"):
            request["tool_choice"] = {"type": "auto"}

    enabled = _thinking_enabled(thinking_mode, iteration)
    if enabled:
        budget = thinking_budget or min(max(max_tokens - 1, 256), 2048)
        # budget 必须小于 max_tokens
        budget = min(budget, max_tokens - 1)
        if budget > 0:
            request["thinking"] = {"type": "enabled", "budget_tokens": budget}
    return request


def _prepare_ollama(
    body: dict[str, Any], *, thinking_mode: str, iteration: int
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if body.get("temperature") is not None:
        options["temperature"] = float(body["temperature"])
    if body.get("max_tokens"):
        options["num_predict"] = int(body["max_tokens"])
    for key in ("top_p", "top_k", "repeat_penalty", "presence_penalty"):
        if body.get(key) is not None:
            options[key] = body[key]
    request: dict[str, Any] = {
        "model": body.get("model"),
        "messages": to_ollama_messages(body.get("messages") or []),
    }
    if options:
        request["options"] = options
    if body.get("tools"):
        request["tools"] = body["tools"]
    if body.get("stream"):
        request["stream"] = True
    enabled = _thinking_enabled(thinking_mode, iteration)
    if enabled is not None:
        request["think"] = enabled
    return request


# ----------------------------------------------------------------------
# 响应解析（归一化为 OpenAI 风格 message + usage）
# ----------------------------------------------------------------------
def parse_response(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a non-streaming response to {message, usage, timings}."""
    if kind == "anthropic":
        return _parse_anthropic(data)
    if kind == "ollama":
        return _parse_ollama(data)
    return _normalize_openai(data)


def _normalize_openai(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("chat_api_no_choices")
    return data


def _parse_anthropic(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("type") == "error":
        error = data.get("error") or {}
        raise RuntimeError(f"anthropic_error_{error.get('type')}")
    blocks = data.get("content") or []
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text") or "")
        elif block_type == "thinking":
            thinking_parts.append(block.get("thinking") or "")
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(
                            block.get("input") or {}, ensure_ascii=False
                        ),
                    },
                }
            )
    usage_raw = data.get("usage") or {}
    prompt_tokens = int(usage_raw.get("input_tokens") or 0)
    completion_tokens = int(usage_raw.get("output_tokens") or 0)
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "".join(text_parts) or None,
                    "reasoning_content": "".join(thinking_parts) or None,
                    "tool_calls": tool_calls or None,
                },
                "finish_reason": data.get("stop_reason"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "timings": {},
        "raw": data,
    }


def _parse_ollama(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("error"):
        raise RuntimeError(f"ollama_error_{str(data['error'])[:60]}")
    message = data.get("message") or {}
    tool_calls = []
    for index, tool_call in enumerate(message.get("tool_calls") or []):
        function = tool_call.get("function") or {}
        tool_calls.append(
            {
                "id": tool_call.get("id") or _gen_tool_id(index),
                "type": "function",
                "function": {
                    "name": function.get("name") or "",
                    "arguments": json.dumps(
                        function.get("arguments") or {}, ensure_ascii=False
                    )
                    if not isinstance(function.get("arguments"), str)
                    else function["arguments"],
                },
            }
        )
    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    completion_tokens = int(data.get("eval_count") or 0)
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": message.get("content") or None,
                    "reasoning_content": message.get("thinking") or None,
                    "tool_calls": tool_calls or None,
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "timings": {
            key: data.get(key)
            for key in ("total_duration", "load_duration", "eval_duration")
            if data.get(key) is not None
        },
        "raw": data,
    }


# ----------------------------------------------------------------------
# 流式解析
# ----------------------------------------------------------------------
async def iter_stream_events(
    kind: str, response: aiohttp.ClientResponse
) -> AsyncIterator[dict[str, Any]]:
    """Yield normalized stream events for the given protocol."""
    if kind == "anthropic":
        async for event in _iter_anthropic(response):
            yield event
    elif kind == "ollama":
        async for event in _iter_ollama(response):
            yield event
    else:
        async for event in _iter_openai(response):
            yield event


async def _iter_openai(response: aiohttp.ClientResponse) -> AsyncIterator[dict[str, Any]]:
    tool_calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None

    while True:
        raw_line = await response.content.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", "ignore").strip()
        if not line or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except ValueError:
            continue

        if chunk.get("usage"):
            usage = chunk["usage"]

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                yield {"content": delta["content"]}
            if delta.get("reasoning_content"):
                yield {"thinking": delta["reasoning_content"]}
            for index, part in enumerate(delta.get("tool_calls") or []):
                slot = tool_calls.setdefault(
                    int(part.get("index", index)),
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if part.get("id"):
                    slot["id"] = part["id"]
                function = part.get("function") or {}
                if function.get("name"):
                    slot["function"]["name"] = function["name"]
                if function.get("arguments"):
                    slot["function"]["arguments"] += function["arguments"]

    if tool_calls:
        yield {"tool_calls": [tool_calls[key] for key in sorted(tool_calls)]}
    if usage:
        yield {"usage": usage}


async def _iter_anthropic(
    response: aiohttp.ClientResponse,
) -> AsyncIterator[dict[str, Any]]:
    tool_blocks: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    current_event: str | None = None

    while True:
        raw_line = await response.content.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", "ignore").strip()
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            payload = json.loads(data_str)
        except ValueError:
            continue
        event_type = payload.get("type") or current_event

        if event_type == "message_start":
            message_usage = (payload.get("message") or {}).get("usage") or {}
            usage["prompt_tokens"] = int(message_usage.get("input_tokens") or 0)
            continue
        if event_type == "content_block_start":
            block = payload.get("content_block") or {}
            index = int(payload.get("index", 0))
            if block.get("type") == "tool_use":
                tool_blocks[index] = {
                    "id": block.get("id") or _gen_tool_id(index),
                    "type": "function",
                    "function": {"name": block.get("name") or "", "arguments": ""},
                }
            elif block.get("type") == "thinking" and block.get("thinking"):
                yield {"thinking": block["thinking"]}
            elif block.get("type") == "text" and block.get("text"):
                yield {"content": block["text"]}
            continue
        if event_type == "content_block_delta":
            delta = payload.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                yield {"content": delta.get("text") or ""}
            elif delta_type == "thinking_delta":
                yield {"thinking": delta.get("thinking") or ""}
            elif delta_type == "input_json_delta":
                slot = tool_blocks.get(int(payload.get("index", 0)))
                if slot is not None:
                    slot["function"]["arguments"] += delta.get("partial_json") or ""
            continue
        if event_type == "message_delta":
            message_usage = payload.get("usage") or {}
            if message_usage.get("output_tokens") is not None:
                usage["completion_tokens"] = int(message_usage["output_tokens"])
            continue
        if event_type == "message_stop":
            break
        if event_type == "error":
            error = payload.get("error") or {}
            raise RuntimeError(f"anthropic_error_{error.get('type')}")

    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    if tool_blocks:
        calls = []
        for index in sorted(tool_blocks):
            slot = tool_blocks[index]
            raw_args = slot["function"]["arguments"] or "{}"
            try:
                json.loads(raw_args)
            except ValueError:
                raw_args = "{}"
            calls.append({**slot, "function": {**slot["function"], "arguments": raw_args}})
        yield {"tool_calls": calls}
    yield {"usage": usage}


async def _iter_ollama(response: aiohttp.ClientResponse) -> AsyncIterator[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    timings: dict[str, Any] = {}

    while True:
        raw_line = await response.content.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", "ignore").strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        try:
            chunk = json.loads(line)
        except ValueError:
            continue
        if chunk.get("error"):
            raise RuntimeError(f"ollama_error_{str(chunk['error'])[:60]}")

        message = chunk.get("message") or {}
        if message.get("thinking"):
            yield {"thinking": message["thinking"]}
        if message.get("content"):
            yield {"content": message["content"]}
        for index, tool_call in enumerate(message.get("tool_calls") or []):
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            tool_calls.append(
                {
                    "id": tool_call.get("id") or _gen_tool_id(index),
                    "type": "function",
                    "function": {
                        "name": function.get("name") or "",
                        "arguments": json.dumps(arguments, ensure_ascii=False)
                        if not isinstance(arguments, str)
                        else arguments,
                    },
                }
            )
        prompt_tokens = int(chunk.get("prompt_eval_count") or prompt_tokens)
        completion_tokens = int(chunk.get("eval_count") or completion_tokens)
        for key in ("total_duration", "load_duration", "eval_duration"):
            if chunk.get(key) is not None:
                timings[key] = chunk[key]
        if chunk.get("done"):
            break

    if tool_calls:
        yield {"tool_calls": tool_calls}
    yield {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    }
    if timings:
        yield {"timings": timings}
