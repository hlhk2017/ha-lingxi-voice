"""Conversation platform powered by an OpenAI-compatible chat API."""

from __future__ import annotations

import asyncio

import json
import logging
from typing import Any, Literal

import aiohttp
import probatio

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationEntity, ConversationEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import ConfigType

from .helpers import build_device_info, resolve_target
from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_LLM_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    MAX_TOOL_ITERATIONS,
    short_model_name,
)

_LOGGER = logging.getLogger(__name__)

LLM_HASS_API = "assist"

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=90)
MAX_HISTORY_MESSAGES = 24


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation agent."""
    async_add_entities([AIVoiceConversationEntity(entry)])


def _convert_content(content: list) -> list[dict[str, Any]]:
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
            if item.tool_calls:
                msg["content"] = item.content or None
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.tool_args or {}, ensure_ascii=False),
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
                    "content": json.dumps(item.result.data, ensure_ascii=False, default=str),
                }
            )
    # 只保留最近若干条，避免 prompt 无限增长
    if len(messages) > MAX_HISTORY_MESSAGES:
        system = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        messages = system + rest[-MAX_HISTORY_MESSAGES:]
    return messages


def _convert_tools(llm_api: llm.APIInstance) -> list[dict[str, Any]]:
    """Convert HA LLM tools to OpenAI function definitions."""
    unsupported_keys = {"oneOf", "anyOf", "allOf", "enum", "not"}
    tools: list[dict[str, Any]] = []
    for tool in llm_api.tools:
        schema = probatio.to_openapi(
            tool.parameters,
            custom_serializer=llm_api.custom_serializer,
            openapi_version="3.1.0",
        )
        if unsupported_keys.intersection(schema):
            schema = {k: v for k, v in schema.items() if k not in unsupported_keys}
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


class AIVoiceConversationEntity(ConversationEntity):
    """Conversation agent backed by an OpenAI-compatible endpoint."""

    _attr_has_entity_name = True
    _attr_supports_streaming = False

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self._entry = entry
        self._attr_name = f"对话（{short_model_name(entry.data[CONF_LLM_MODEL])}）"
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}"
        self._base_url = (resolve_target(entry, "llm")["base_url"] or "").rstrip("/")
        self._api_key = resolve_target(entry, "llm")["api_key"]
        self._model = entry.data[CONF_LLM_MODEL]
        self._prompt = entry.data.get(CONF_PROMPT)
        self._attr_supported_features = ConversationEntityFeature.CONTROL
        self._attr_device_info = build_device_info(entry, entry.data[CONF_LLM_MODEL])

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

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process one user utterance."""
        try:
            voice_rules = (
                "\n\n语音助手规则（必须遵守）："
                "1) 回答极简，1-2 句口语，不要列表/emoji/Markdown；"
                "2) 控制设备直接调用工具，最多调用一次 search_entities，"
                "找不到对应实体就立即用一句话说明，不要反复搜索；"
                "3) 简单确认用一句话完成。"
            )
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                LLM_HASS_API,
                self._current_prompt,
                (user_input.extra_system_prompt or "") + voice_rules,
            )
        except asyncio.CancelledError:
            # 被新一次唤醒打断：返回空响应，避免 TTS 朗读 "Error: CancelledError"
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(agent_id=self.entity_id, content="")
            )
            _LOGGER.debug("Conversation cancelled (interrupted by new wake)")
            return conversation.async_get_result_from_chat_log(user_input, chat_log)
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        try:
            await self._async_handle_chat_log(chat_log)
        except Exception:  # noqa: BLE001
            _LOGGER.exception("Error talking to %s", self._model)
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=self.entity_id,
                    content="抱歉，AI 服务暂时不可用，请稍后再试。",
                )
            )

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _async_handle_chat_log(self, chat_log: conversation.ChatLog) -> None:
        """Run the tool-calling loop until the model produces a final answer."""
        for _iteration in range(MAX_TOOL_ITERATIONS):
            messages = _convert_content(chat_log.content)
            body: dict[str, Any] = {
                "model": self._current_model,
                "messages": messages,
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "stream": False,
            }
            if self._target["kind"] == "mimo":
                # 语音场景关闭深度思考，显著降低响应延迟
                body["thinking"] = {"type": "disabled"}
            tools = _convert_tools(chat_log.llm_api) if chat_log.llm_api else []
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"

            choice = await self._async_call_completion(body)
            message = choice.get("message") or {}

            tool_calls: list[llm.ToolInput] = []
            for tc in message.get("tool_calls") or []:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except ValueError:
                    args = {}
                tool_calls.append(
                    llm.ToolInput(
                        id=tc.get("id") or "",
                        tool_name=fn.get("name") or "",
                        tool_args=args or {},
                    )
                )

            content = conversation.AssistantContent(
                agent_id=self.entity_id,
                content=(message.get("content") or None),
                tool_calls=tool_calls or None,
            )
            if content.content is None and not tool_calls:
                chat_log.async_add_assistant_content_without_tools(content)
                return

            async for _result in chat_log.async_add_assistant_content(content):
                pass

            if not chat_log.unresponded_tool_results:
                return
        _LOGGER.warning("Reached max tool iterations (%s)", MAX_TOOL_ITERATIONS)

    async def _async_call_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        """Call /chat/completions and return the first choice."""
        session = async_get_clientsession(self.hass)
        target = self._target
        if target["kind"] == "mimo":
            headers = {
                "api-key": target["api_key"],
                "Content-Type": "application/json",
            }
        else:
            headers = {
                "Authorization": f"Bearer {target['api_key']}",
                "Content-Type": "application/json",
            }
        async with session.post(
            f"{target['base_url'].rstrip('/')}/chat/completions",
            headers=headers,
            json=body,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            payload = await resp.text()
            if resp.status != 200:
                _LOGGER.error("Chat API error %s: %s", resp.status, payload[:300])
                raise RuntimeError(f"chat_api_http_{resp.status}")
            data = json.loads(payload)
        choices = data.get("choices") or []
        if not choices:
            _LOGGER.error("Chat API returned no choices: %s", payload[:300])
            raise RuntimeError("chat_api_no_choices")
        return choices[0]