"""共享请求层：Jinja2 渲染、请求头/体构造、流式 SSE、输出清理、事件。

v1.2.0 引入，供 conversation / ai_task 复用。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import re
import unicodedata
from pathlib import Path
from typing import Any, AsyncIterator

import aiohttp
import probatio

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm, template
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util

from . import protocols
from .const import (
    CONF_STRIP_EMOJI,
    CONF_STRIP_THINK,
    CONF_CHAT_TEMPLATE_ARGS,
    CONF_CUSTOM_HEADERS,
    CONF_INCLUDE_PRIOR_THINKING,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_MIN_P,
    CONF_PRESENCE_PENALTY,
    CONF_REASONING_EFFORT,
    CONF_REPEAT_PENALTY,
    CONF_REQUEST_BODY_PARAMS,
    CONF_SERVER_TYPE,
    CONF_SLOT_ID,
    CONF_THINKING_MODE,
    CONF_THINKING_TOKEN_BUDGET,
    CONF_TOP_K,
    CONF_TOP_P,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_MIN_P,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    EVENT_GENERATION_COMPLETE,
    short_model_name,
)

_LOGGER = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=120)
STREAM_TIMEOUT = aiohttp.ClientTimeout(total=180, sock_read=120)

_UNSUPPORTED_TOOL_SCHEMA_KEYS = {"oneOf", "anyOf", "allOf", "enum", "not"}

_THINK_RE = re.compile(r"<think(?:ing)?>.*?(?:</think(?:ing)?>|\Z)", re.I | re.S)
_THINK_START_RE = re.compile(r"<think(?:ing)?>", re.I)
_THINK_END_RE = re.compile(r"</think(?:ing)?>", re.I)
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U00002B00-\U00002BFF"
    "\U0001F1E6-\U0001F1FF"
    "\U0000FE0F\U0000200D\U0000FE00"
    "]",
    re.UNICODE,
)


# ----------------------------------------------------------------------
# 配置读取
# ----------------------------------------------------------------------
def get_option(entry, key: str, default: Any = None) -> Any:
    """Read an option from options first, then entry data, then default."""
    if entry.options and key in entry.options:
        return entry.options[key]
    value = entry.data.get(key)
    return default if value is None or value == "" else value


def get_bool(entry, key: str, default: bool) -> bool:
    """Read a boolean option."""
    value = get_option(entry, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def get_int(entry, key: str, default: int) -> int:
    """Read an integer option."""
    try:
        return int(get_option(entry, key, default))
    except (TypeError, ValueError):
        return default


def get_float(entry, key: str, default: float) -> float:
    """Read a float option."""
    try:
        return float(get_option(entry, key, default))
    except (TypeError, ValueError):
        return default


def get_json(entry, key: str) -> dict[str, Any]:
    """Read a JSON-object option; also accepts ``k=v`` lines."""
    raw = entry.data.get(key) or (entry.options or {}).get(key) or ""
    return parse_json_object(raw)


def parse_json_object(raw: Any) -> dict[str, Any]:
    """Parse a JSON object from a string, tolerating ``k=v`` lines."""
    if isinstance(raw, dict):
        return dict(raw)
    if not raw or not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if parsed is None:
                parsed = {}
            parsed[key.strip()] = value.strip()
        if parsed is None:
            _LOGGER.warning("无法解析 JSON 参数：%s", text[:200])
            return {}
    return parsed if isinstance(parsed, dict) else {}


# ----------------------------------------------------------------------
# Jinja2 渲染
# ----------------------------------------------------------------------
def render_value(
    hass: HomeAssistant, value: Any, variables: dict[str, Any] | None = None
) -> Any:
    """Render a Jinja2 string; returns the raw value when rendering fails."""
    if not isinstance(value, str) or "{%" not in value and "{{" not in value:
        return value
    try:
        return template.Template(value, hass).async_render(
            variables or {}, parse_result=False
        )
    except template.TemplateError as err:
        _LOGGER.warning("模板渲染失败：%s", err)
        return value


def render_mapping(
    hass: HomeAssistant, raw: Any, variables: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Render and parse a JSON-object option."""
    parsed = parse_json_object(raw)
    rendered: dict[str, Any] = {}
    for key, value in parsed.items():
        if isinstance(value, str):
            value = render_value(hass, value, variables)
            # 渲染结果若为 JSON 片段，尝试解析为结构化数据
            if isinstance(value, str):
                stripped = value.strip()
                if stripped[:1] in ("{", "[", '"') or stripped in ("true", "false", "null"):
                    try:
                        value = json.loads(stripped)
                    except ValueError:
                        pass
        rendered[key] = value
    return rendered


# ----------------------------------------------------------------------
# 请求构造
# ----------------------------------------------------------------------
def build_headers(
    entry,
    target: dict,
    hass: HomeAssistant,
    variables: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build request headers including auth and custom headers."""
    kind = target.get("kind") or "openai"
    api_key = target.get("api_key") or ""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if kind == "mimo":
        headers["api-key"] = api_key
    elif kind == "doubao":
        headers["X-Api-Key"] = api_key
    else:
        headers.update(protocols.auth_headers(kind, api_key))

    for key, value in render_mapping(
        hass, get_json(entry, CONF_CUSTOM_HEADERS), variables
    ).items():
        if value is None:
            headers.pop(str(key), None)
            continue
        headers[str(key)] = str(value)
    return headers


def _apply_thinking(
    entry, body: dict[str, Any], kwargs: dict[str, Any], *, iteration: int
) -> None:
    """Apply thinking-mode related options for the configured server type."""
    server_type = entry.data.get(CONF_SERVER_TYPE) or "generic"
    mode = get_option(entry, CONF_THINKING_MODE, "")
    if mode in ("", "none"):
        mode = ""
    if mode and iteration > 0 and mode == "first_iteration":
        enabled: bool | None = False
    elif mode:
        enabled = mode == "always"
    else:
        enabled = None

    if enabled is not None:
        if server_type == "deepseek":
            body["thinking"] = {"type": "enabled" if enabled else "disabled"}
        else:
            kwargs["enable_thinking"] = enabled


def build_body(
    entry,
    target: dict,
    hass: HomeAssistant,
    base_body: dict[str, Any],
    *,
    variables: dict[str, Any] | None = None,
    iteration: int = 0,
) -> dict[str, Any]:
    """Build the chat-completion body with server-specific options."""
    body = dict(base_body)
    server_type = entry.data.get(CONF_SERVER_TYPE) or "generic"
    variables = variables or {}

    kwargs = render_mapping(hass, get_json(entry, CONF_CHAT_TEMPLATE_ARGS), variables)
    _apply_thinking(entry, body, kwargs, iteration=iteration)
    if target.get("kind") == "mimo" and "thinking" not in body:
        # 语音场景默认关闭 MiMo 深度思考，显著降低响应延迟
        body["thinking"] = {"type": "disabled"}

    if server_type == "deepseek":
        effort = get_option(entry, CONF_REASONING_EFFORT, "")
        if effort in ("high", "max"):
            body["reasoning_effort"] = effort

    elif server_type == "llama_cpp":
        if get_option(entry, CONF_SLOT_ID, ""):
            try:
                body["id_slot"] = int(get_option(entry, CONF_SLOT_ID))
            except (TypeError, ValueError):
                pass
        if get_option(entry, CONF_TOP_P, None) is not None:
            body["top_p"] = get_float(entry, CONF_TOP_P, DEFAULT_TOP_P)
        min_p = get_float(entry, CONF_MIN_P, DEFAULT_MIN_P)
        if min_p:
            body["min_p"] = min_p
        top_k = get_int(entry, CONF_TOP_K, DEFAULT_TOP_K)
        if top_k:
            body["top_k"] = top_k
        repeat_penalty = get_float(entry, CONF_REPEAT_PENALTY, DEFAULT_REPEAT_PENALTY)
        if repeat_penalty != DEFAULT_REPEAT_PENALTY:
            body["repeat_penalty"] = repeat_penalty
        presence_penalty = get_float(
            entry, CONF_PRESENCE_PENALTY, DEFAULT_PRESENCE_PENALTY
        )
        if presence_penalty:
            body["presence_penalty"] = presence_penalty

    elif server_type == "vllm":
        budget = get_option(entry, CONF_THINKING_TOKEN_BUDGET, "")
        if budget not in ("", None):
            try:
                body["thinking_token_budget"] = int(budget)
            except (TypeError, ValueError):
                pass

    elif server_type == "localai" and kwargs:
        metadata = dict(body.get("metadata") or {})
        metadata.update({str(k): str(v) for k, v in kwargs.items()})
        body["metadata"] = metadata
        kwargs = {}

    if kwargs:
        chat_kwargs = dict(body.get("chat_template_kwargs") or {})
        chat_kwargs.update(kwargs)
        body["chat_template_kwargs"] = chat_kwargs

    # 用户自定义请求体参数最后合并（model/messages/stream 受保护）
    for key, value in render_mapping(
        hass, get_json(entry, CONF_REQUEST_BODY_PARAMS), variables
    ).items():
        if key in ("model", "messages", "stream"):
            _LOGGER.warning("忽略受保护的请求体参数：%s", key)
            continue
        body[key] = value

    kind = target.get("kind") or "openai"
    if kind in ("anthropic", "ollama"):
        mode = get_option(entry, CONF_THINKING_MODE, "")
        if mode in ("", "none"):
            mode = ""
        budget_raw = get_option(entry, CONF_THINKING_TOKEN_BUDGET, "")
        budget: int | None = None
        if budget_raw not in ("", None):
            try:
                budget = int(budget_raw)
            except (TypeError, ValueError):
                budget = None
        body = protocols.prepare_body(
            kind,
            body,
            thinking_mode=mode,
            thinking_budget=budget,
            iteration=iteration,
        )

    return body


def conversation_variables(
    hass: HomeAssistant, conversation_id: str | None = None
) -> dict[str, Any]:
    """Variables exposed to Jinja2 templates."""
    return {"conversation_id": conversation_id or ""}


def include_prior_thinking(entry) -> bool:
    """Whether prior-turn thinking should be sent back to the model."""
    default = True
    value = get_option(entry, CONF_INCLUDE_PRIOR_THINKING, default)
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


# ----------------------------------------------------------------------
# 工具定义
# ----------------------------------------------------------------------
def convert_tools(llm_api: llm.APIInstance | None) -> list[dict[str, Any]]:
    """Convert HA LLM tools to OpenAI function definitions."""
    if llm_api is None:
        return []
    tools: list[dict[str, Any]] = []
    for tool in llm_api.tools:
        schema = probatio.to_openapi(
            tool.parameters,
            custom_serializer=llm_api.custom_serializer,
            openapi_version="3.1.0",
        )
        if _UNSUPPORTED_TOOL_SCHEMA_KEYS.intersection(schema):
            schema = {
                k: v for k, v in schema.items() if k not in _UNSUPPORTED_TOOL_SCHEMA_KEYS
            }
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": (tool.description or "")[:1024],
                    "parameters": schema,
                },
            }
        )
    return tools


def parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[llm.ToolInput]:
    """Convert OpenAI tool_calls payloads to llm.ToolInput list."""
    tool_calls: list[llm.ToolInput] = []
    for item in raw_calls or []:
        fn = item.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except ValueError:
            args = {}
        tool_calls.append(
            llm.ToolInput(
                id=item.get("id") or "",
                tool_name=fn.get("name") or "",
                tool_args=args or {},
            )
        )
    return tool_calls


class StreamCleaner:
    """Incremental cleaner for streamed content (thinking tags / emoji)."""

    _KEEP = 12

    def __init__(self, *, strip_think: bool, strip_emoji: bool) -> None:
        self._strip_think = strip_think
        self._strip_emoji = strip_emoji
        self._buffer = ""
        self._in_think = False

    def feed(self, chunk: str) -> str:
        """Consume a chunk and return the text safe to emit now."""
        if not chunk:
            return ""
        if not self._strip_think:
            return strip_emoji(chunk) if self._strip_emoji else chunk

        self._buffer += chunk
        output = ""
        while True:
            if self._in_think:
                end = _THINK_END_RE.search(self._buffer)
                if not end:
                    # 丢弃已确认属于思考内容的部分，保留尾部以防标签跨块
                    if len(self._buffer) > self._KEEP:
                        self._buffer = self._buffer[-self._KEEP :]
                    break
                self._buffer = self._buffer[end.end() :]
                self._in_think = False
                continue

            start = _THINK_START_RE.search(self._buffer)
            if start:
                output += self._buffer[: start.start()]
                self._buffer = self._buffer[start.end() :]
                self._in_think = True
                continue

            # 尾部可能与思考标签开头重叠，先保留等待下一块
            keep = 0
            for size in range(min(len(self._buffer), self._KEEP), 0, -1):
                tail = self._buffer[-size:]
                for prefix in ("<think", "<thinking"):
                    if prefix.startswith(tail.lower()):
                        keep = size
                        break
                if keep:
                    break
            if keep:
                output += self._buffer[:-keep]
                self._buffer = self._buffer[-keep:]
            else:
                output += self._buffer
                self._buffer = ""
            break

        return strip_emoji(output) if self._strip_emoji and output else output

    def flush(self) -> str:
        """Return any remaining text at end of stream."""
        if self._in_think:
            self._buffer = ""
            return ""
        output, self._buffer = self._buffer, ""
        if self._strip_think and re.match(r"<think(?:ing)?", output.strip(), re.I):
            return ""
        return strip_emoji(output) if self._strip_emoji and output else output


# ----------------------------------------------------------------------
# 输出清理
# ----------------------------------------------------------------------
def strip_think(text: str | None) -> str | None:
    """Remove  thinking/<thinking> blocks from a response."""
    if not text:
        return text
    cleaned = _THINK_RE.sub("", text)
    return cleaned.strip()


def strip_emoji(text: str | None) -> str | None:
    """Remove emoji characters from a response."""
    if not text:
        return text
    cleaned = _EMOJI_RE.sub("", text)
    # 去掉 emoji 留下的多余空格
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def clean_output(entry, text: str | None) -> str | None:
    """Apply configured output cleanup (thinking tags / emoji)."""
    if text is None:
        return None
    if get_bool(entry, CONF_STRIP_THINK, True):
        text = strip_think(text)
    if get_bool(entry, CONF_STRIP_EMOJI, False):
        text = strip_emoji(text)
    return text


# ----------------------------------------------------------------------
# HTTP 调用
# ----------------------------------------------------------------------
def completion_url(target: dict) -> str:
    """Return the chat endpoint for a provider target."""
    return protocols.chat_url(
        target.get("kind") or "openai", target.get("base_url") or ""
    )


async def async_chat_completion(
    hass: HomeAssistant,
    entry,
    target: dict,
    body: dict[str, Any],
    *,
    variables: dict[str, Any] | None = None,
    iteration: int = 0,
    timeout: aiohttp.ClientTimeout | None = None,
) -> dict[str, Any]:
    """Call /chat/completions (non-streaming) and return the payload."""
    session = async_get_clientsession(hass)
    variables = dict(variables or {})
    headers = build_headers(entry, target, hass, variables)
    request_body = build_body(
        entry, target, hass, body, variables=variables, iteration=iteration
    )
    started = dt_util.utcnow()
    async with session.post(
        completion_url(target),
        headers=headers,
        json=request_body,
        timeout=timeout or REQUEST_TIMEOUT,
    ) as resp:
        payload = await resp.text()
        if resp.status != 200:
            _LOGGER.error("Chat API error %s: %s", resp.status, payload[:300])
            raise RuntimeError(f"chat_api_http_{resp.status}")
    data = json.loads(payload)
    normalized = protocols.parse_response(target.get("kind") or "openai", data)
    fire_generation_event(hass, entry, target, normalized, started)
    return normalized


def first_choice(data: dict[str, Any]) -> dict[str, Any]:
    """Return the first choice of a completion payload."""
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("chat_api_no_choices")
    return choices[0]


async def async_stream_chat_completion(
    hass: HomeAssistant,
    entry,
    target: dict,
    body: dict[str, Any],
    *,
    variables: dict[str, Any] | None = None,
    timeout: aiohttp.ClientTimeout | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream a chat completion as deltas.

    Yields dicts with keys ``content`` / ``thinking`` / ``tool_calls`` /
    ``usage``. Tool calls are accumulated and yielded once, complete.
    """
    session = async_get_clientsession(hass)
    variables = dict(variables or {})
    variables.setdefault("iteration", 0)
    headers = build_headers(entry, target, hass, variables)
    request_body = build_body(
        entry,
        target,
        hass,
        {**body, "stream": True},
        variables=variables,
        iteration=variables.get("iteration", 0),
    )
    started = dt_util.utcnow()
    usage: dict[str, Any] | None = None
    kind = target.get("kind") or "openai"

    async with session.post(
        completion_url(target),
        headers=headers,
        json=request_body,
        timeout=timeout or STREAM_TIMEOUT,
    ) as resp:
        if resp.status != 200:
            payload = await resp.text()
            _LOGGER.error("Chat stream error %s: %s", resp.status, payload[:300])
            raise RuntimeError(f"chat_api_http_{resp.status}")

        async for event in protocols.iter_stream_events(kind, resp):
            if "usage" in event:
                usage = event["usage"]
                yield {"usage": usage}
            elif "timings" in event:
                continue
            else:
                yield event

    fire_generation_event(
        hass,
        entry,
        target,
        {"usage": usage or {}},
        started,
        extra={"streaming": True},
    )


# ----------------------------------------------------------------------
# 事件
# ----------------------------------------------------------------------
def fire_generation_event(
    hass: HomeAssistant,
    entry,
    target: dict,
    data: dict[str, Any],
    started,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit ai_voice_generation_complete with usage/timings when available."""
    try:
        usage = data.get("usage") or (data.get("data") or {}).get("usage") or {}
        timings = data.get("timings") or (data.get("data") or {}).get("timings") or {}
        finished = dt_util.utcnow()
        payload: dict[str, Any] = {
            "entry_id": entry.entry_id,
            "provider": target.get("name"),
            "model": target.get("model"),
            "usage": usage,
            "timings": timings,
            "duration_ms": int((finished - started).total_seconds() * 1000),
        }
        if extra:
            payload.update(extra)
        hass.bus.async_fire(EVENT_GENERATION_COMPLETE, payload)
    except Exception:  # noqa: BLE001
        _LOGGER.debug("Failed to fire generation-complete event", exc_info=True)


# ----------------------------------------------------------------------
# 多模态附件
# ----------------------------------------------------------------------
def _read_file(path: str) -> bytes:
    return Path(path).read_bytes()


async def attachment_to_image_part(
    hass: HomeAssistant, attachment
) -> dict[str, Any] | None:
    """Convert a HA conversation attachment to an OpenAI image_url part."""
    path = getattr(attachment, "path", None)
    mime = getattr(attachment, "mime_type", None) or ""
    if not path:
        return None
    if not mime:
        mime = mimetypes.guess_type(path)[0] or "image/png"
    if not mime.startswith("image/"):
        return None
    try:
        content = await hass.async_add_executor_job(_read_file, str(path))
    except OSError:
        _LOGGER.warning("无法读取附件：%s", path)
        return None
    encoded = base64.b64encode(content).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }


def text_of_content(content: Any) -> str:
    """Flatten HA conversation content (str or list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return ""


def tool_inputs_to_openai(tool_calls: list) -> list[dict[str, Any]]:
    """Convert llm.ToolInput objects back to OpenAI tool_calls payloads."""
    return [
        {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.tool_name,
                "arguments": json.dumps(
                    tool_call.tool_args or {}, ensure_ascii=False
                ),
            },
        }
        for tool_call in tool_calls
    ]


async def convert_content_with_attachments(
    hass: HomeAssistant, content: list, include_thinking: bool = False
) -> list[dict[str, Any]]:
    """Convert HA chat-log content (with image attachments) to chat messages."""
    messages: list[dict[str, Any]] = []
    for item in content:
        role = getattr(item, "role", None)
        if role == "system":
            messages.append({"role": "system", "content": item.content})
        elif role == "user":
            text = text_of_content(item.content)
            parts: list[dict[str, Any]] = []
            if text:
                parts.append({"type": "text", "text": text})
            for attachment in getattr(item, "attachments", None) or []:
                part = await attachment_to_image_part(hass, attachment)
                if part:
                    parts.append(part)
            if any(part.get("type") == "image_url" for part in parts):
                messages.append({"role": "user", "content": parts})
            else:
                messages.append({"role": "user", "content": text})
        elif role == "assistant":
            message: dict[str, Any] = {"role": "assistant"}
            if item.content:
                message["content"] = item.content
            if include_thinking and getattr(item, "thinking_content", None):
                message["reasoning_content"] = item.thinking_content
            if item.tool_calls:
                message["content"] = item.content or None
                message["tool_calls"] = tool_inputs_to_openai(item.tool_calls)
            if message.get("content") is None and not message.get("tool_calls"):
                continue
            messages.append(message)
        elif role == "tool_result":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.tool_call_id,
                    "content": json.dumps(
                        item.result.data, ensure_ascii=False, default=str
                    ),
                }
            )
    return messages


def max_tool_iterations(entry) -> int:
    """Configured maximum tool-call iterations."""
    return max(1, get_int(entry, CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS))