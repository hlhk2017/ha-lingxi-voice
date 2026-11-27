"""Conversation platform powered by an OpenAI-compatible chat API."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

import aiohttp

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AssistantContent,
    ConversationEntity,
    ConversationEntityFeature,
    ToolResultContent,
    UserContent,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    floor_registry as fr,
)
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import api
from .const import (
    CONF_STRIP_EMOJI,
    CONF_STRIP_THINK,
    CONF_CONTEXT_ROLE,
    CONF_CONTEXT_TEMPLATE,
    CONF_HISTORY_MAX,
    CONF_LLM_MODEL,
    CONF_MAX_TOKENS,
    CONF_PARALLEL_TOOL_CALLS,
    CONF_PROMPT,
    CONF_STREAM,
    CONF_TEMPERATURE,
    DEFAULT_CONTEXT_ROLE,
    DEFAULT_CONTEXT_TEMPLATE,
    DEFAULT_HISTORY_MAX,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    short_model_name,
)
from .helpers import build_device_info, resolve_target

_LOGGER = logging.getLogger(__name__)

LLM_HASS_API = "assist"

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=90)

VOICE_RULES = (
    "\n\n语音助手规则（必须遵守）："
    "1) 回答极简，1-2 句口语，不要列表/emoji/Markdown；"
    "2) 控制设备直接调用工具，最多调用一次 search_entities，"
    "找不到对应实体就立即用一句话说明，不要反复搜索；"
    "3) 简单确认用一句话完成。"
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation agent."""
    async_add_entities([AIVoiceConversationEntity(entry)])


def _convert_content(content: list, include_prior_thinking: bool) -> list[dict[str, Any]]:
    """Convert HA chat-log content to OpenAI chat messages."""
    messages: list[dict[str, Any]] = []
    for item in content:
        role = getattr(item, "role", None)
        if role == "system":
            messages.append({"role": "system", "content": item.content})
        elif role == "user":
            messages.append({"role": "user", "content": item.content})
        elif role == "assistant":
            msg: dict[str, Any] = {"role": "assistant"}
            if item.content:
                msg["content"] = item.content
            if include_prior_thinking and getattr(item, "thinking_content", None):
                msg["reasoning_content"] = item.thinking_content
            if item.tool_calls:
                msg["content"] = item.content or None
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(
                                tc.tool_args or {}, ensure_ascii=False
                            ),
                        },
                    }
                    for tc in item.tool_calls
                ]
            if msg.get("content") is None and not msg.get("tool_calls"):
                continue
            messages.append(msg)
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


def _trim_messages(messages: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Trim history to the latest N messages, keeping system prompts."""
    if limit <= 0 or len(messages) <= limit:
        return messages
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    trimmed = system + rest[-limit:]
    # 裁掉开头孤立的 tool 结果，避免 API 因缺少对应 tool_call 报错
    while (
        len(trimmed) > len(system)
        and trimmed[len(system)].get("role") == "tool"
    ):
        trimmed.pop(len(system))
    return trimmed


class AIVoiceConversationEntity(ConversationEntity):
    """Conversation agent backed by an OpenAI-compatible endpoint."""

    _attr_has_entity_name = True
    _attr_supports_streaming = False

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self._entry = entry
        self._attr_name = f"对话（{short_model_name(entry.data[CONF_LLM_MODEL])}）"
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}"
        self._attr_supported_features = ConversationEntityFeature.CONTROL
        self._attr_supports_streaming = api.get_bool(entry, CONF_STREAM, False)
        self._attr_device_info = build_device_info(entry, entry.data[CONF_LLM_MODEL])
        self._stream_partial = False

    @property
    def _target(self) -> dict:
        """Current provider+model for the conversation capability."""
        return resolve_target(self._entry, "llm")

    @property
    def _current_model(self) -> str:
        """Always read the chat model from the entry data."""
        return self._target["model"]

    @property
    def _current_prompt(self) -> str | None:
        """Always read the prompt from the entry data."""
        return self._entry.data.get(CONF_PROMPT)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return MATCH_ALL

    @callback
    def _async_set_agent(self) -> None:
        conversation.async_set_agent(self.hass, self._entry, self)

    async def async_added_to_hass(self) -> None:
        """Register the agent when the entity is added."""
        await super().async_added_to_hass()
        self._async_set_agent()

    async def async_will_remove_from_hass(self) -> None:
        """Unregister the agent when removed."""
        conversation.async_unset_agent(self.hass, self._entry)
        await super().async_will_remove_from_hass()

    # ------------------------------------------------------------------
    # 上下文注入
    # ------------------------------------------------------------------
    def _context_variables(
        self, user_input: conversation.ConversationInput, conversation_id: str
    ) -> dict[str, Any]:
        """Build Jinja2 variables including device/area/floor."""
        variables: dict[str, Any] = {"conversation_id": conversation_id}
        device_id = getattr(user_input, "device_id", None)
        if not device_id:
            return variables
        device = dr.async_get(self.hass).async_get(device_id)
        variables["device"] = device
        if device and device.area_id:
            area = ar.async_get(self.hass).async_get(device.area_id)
            variables["area"] = area
            if area and area.floor_id:
                variables["floor"] = fr.async_get(self.hass).async_get(area.floor_id)
        return variables

    def _async_inject_context(
        self,
        chat_log: conversation.ChatLog,
        rendered: str,
        role: str,
    ) -> None:
        """Insert rendered context before the current user message."""
        if role == "assistant":
            content: Any = AssistantContent(
                agent_id=self.entity_id, content=rendered
            )
        elif role == "user":
            content = UserContent(rendered)
        elif role == "tool":
            content = ToolResultContent(
                agent_id=self.entity_id,
                tool_call_id="ai_voice_context",
                tool_name="context",
                result=llm.ToolResult(data={"content": rendered}),
            )
        else:
            return
        # 当前用户消息位于最后一条，插到它前面
        index = len(chat_log.content) - 1
        chat_log.content.insert(max(index, 0), content)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process one user utterance."""
        context_role = api.get_option(
            self._entry, CONF_CONTEXT_ROLE, DEFAULT_CONTEXT_ROLE
        )
        rendered_context: str | None = None
        template_text = api.get_option(
            self._entry, CONF_CONTEXT_TEMPLATE, DEFAULT_CONTEXT_TEMPLATE
        )
        if context_role in ("", "none"):
            context_role = ""
        if template_text and context_role:
            rendered = api.render_value(
                self.hass,
                template_text,
                self._context_variables(user_input, chat_log.conversation_id),
            )
            rendered_context = str(rendered).strip() if rendered else None

        extra_system = (user_input.extra_system_prompt or "") + VOICE_RULES
        if context_role == "system" and rendered_context:
            extra_system = f"{extra_system}\n\n{rendered_context}"

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                LLM_HASS_API,
                self._current_prompt,
                extra_system,
            )
        except asyncio.CancelledError:
            # 被新一次唤醒打断：返回空响应，避免 TTS 朗读 "Error: CancelledError"
            chat_log.async_add_assistant_content_without_tools(
                AssistantContent(agent_id=self.entity_id, content="")
            )
            _LOGGER.debug("Conversation cancelled (interrupted by new wake)")
            return conversation.async_get_result_from_chat_log(user_input, chat_log)
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        if rendered_context and context_role and context_role != "system":
            self._async_inject_context(chat_log, rendered_context, context_role)

        try:
            await self._async_handle_chat_log(chat_log)
        except Exception:  # noqa: BLE001
            _LOGGER.exception("Error talking to %s", self._current_model)
            chat_log.async_add_assistant_content_without_tools(
                AssistantContent(
                    agent_id=self.entity_id,
                    content="抱歉，AI 服务暂时不可用，请稍后再试。",
                )
            )

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _async_handle_chat_log(self, chat_log: conversation.ChatLog) -> None:
        """Run the tool-calling loop until the model produces a final answer."""
        variables = api.conversation_variables(self.hass, chat_log.conversation_id)
        tools = api.convert_tools(chat_log.llm_api)

        for iteration in range(api.max_tool_iterations(self._entry)):
            body = self._build_request_body(chat_log, tools)
            variables["iteration"] = iteration

            streamed = await self._async_try_stream(chat_log, body, variables)
            if not streamed:
                await self._async_non_stream(chat_log, body, variables)

            if not chat_log.unresponded_tool_results:
                return

        _LOGGER.warning(
            "Reached max tool iterations (%s)", api.max_tool_iterations(self._entry)
        )

    def _build_request_body(
        self, chat_log: conversation.ChatLog, tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build the chat-completion body for one iteration."""
        messages = _convert_content(
            chat_log.content, api.include_prior_thinking(self._entry)
        )
        messages = _trim_messages(
            messages, api.get_int(self._entry, CONF_HISTORY_MAX, DEFAULT_HISTORY_MAX)
        )
        body: dict[str, Any] = {
            "model": self._current_model,
            "messages": messages,
            "temperature": api.get_float(
                self._entry, CONF_TEMPERATURE, DEFAULT_TEMPERATURE
            ),
            "max_tokens": api.get_int(
                self._entry, CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS
            ),
            "stream": False,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
            if api.get_bool(self._entry, CONF_PARALLEL_TOOL_CALLS, False):
                body["parallel_tool_calls"] = True
        return body

    async def _async_try_stream(
        self,
        chat_log: conversation.ChatLog,
        body: dict[str, Any],
        variables: dict[str, Any],
    ) -> bool:
        """Try a streaming completion; return False to fall back to sync."""
        if not self._attr_supports_streaming:
            return False
        self._stream_partial = False

        cleaner = api.StreamCleaner(
            strip_think=api.get_bool(self._entry, CONF_STRIP_THINK, True),
            strip_emoji=api.get_bool(self._entry, CONF_STRIP_EMOJI, False),
        )

        async def deltas():
            self._stream_partial = True
            emitted = False
            has_tool_calls = False
            yield {"role": "assistant"}
            async for event in api.async_stream_chat_completion(
                self.hass, self._entry, self._target, body, variables=variables
            ):
                if "content" in event:
                    text = cleaner.feed(event["content"])
                    if text:
                        emitted = True
                        yield {"content": text}
                elif "thinking" in event:
                    yield {"thinking_content": event["thinking"]}
                elif "tool_calls" in event:
                    parsed_calls = api.parse_tool_calls(event["tool_calls"])
                    if parsed_calls:
                        has_tool_calls = True
                        yield {"tool_calls": parsed_calls}
            tail = cleaner.flush()
            if tail:
                emitted = True
                yield {"content": tail}
            # 内容被完全剥离（只有 emoji/思考）时给一句兜底，避免静音
            if not emitted and not has_tool_calls:
                yield {"content": "好的。"}

        try:
            async for _ in chat_log.async_add_delta_content_stream(
                self.entity_id, deltas()
            ):
                pass
            return True
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            _LOGGER.warning("Streaming failed, falling back to sync", exc_info=True)
            # 已产出部分内容时不再重发，避免重复回答
            return self._stream_partial and chat_log.content[-1].role == "assistant"

    async def _async_non_stream(
        self,
        chat_log: conversation.ChatLog,
        body: dict[str, Any],
        variables: dict[str, Any],
    ) -> None:
        """Run one non-streaming completion iteration."""
        data = await api.async_chat_completion(
            self.hass,
            self._entry,
            self._target,
            body,
            variables=variables,
            iteration=variables.get("iteration", 0),
            timeout=REQUEST_TIMEOUT,
        )
        message = api.first_choice(data).get("message") or {}
        content_text = api.clean_output(self._entry, message.get("content"))
        thinking = message.get("reasoning_content") or message.get("thinking_content")
        if thinking and api.get_bool(self._entry, CONF_STRIP_THINK, True):
            thinking = api.strip_think(thinking)
        tool_calls = api.parse_tool_calls(message.get("tool_calls") or [])

        if content_text == "" and not tool_calls:
            content_text = "好的。"
        content = AssistantContent(
            agent_id=self.entity_id,
            content=content_text,
            thinking_content=thinking or None,
            tool_calls=tool_calls or None,
        )
        if content.content is None and not tool_calls:
            chat_log.async_add_assistant_content_without_tools(content)
            return

        async for _result in chat_log.async_add_assistant_content(content):
            pass