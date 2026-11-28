"""Config flow for AI Voice — per-capability providers with a missing-model loop."""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithReload,
)
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from . import protocols
from .helpers import get_providers, resolve_target

from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_DOUBAO_RESOURCE,
    CONF_IMAGE_BASE_URL,
    CONF_IMAGE_MODEL,
    CONF_IMAGE_SRC,
    CONF_LLM_MODEL,
    CONF_LLM_SRC,
    CONF_NAME,
    CONF_PROMPT,
    CONF_PROTOCOL,
    CONF_PROVIDER,
    CONF_PROVIDERS_EXTRA,
    CONF_CHAT_TEMPLATE_ARGS,
    CONF_CONTEXT_ROLE,
    CONF_CONTEXT_TEMPLATE,
    CONF_CUSTOM_HEADERS,
    CONF_HISTORY_MAX,
    CONF_INCLUDE_PRIOR_THINKING,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_MIN_P,
    CONF_PARALLEL_TOOL_CALLS,
    CONF_PRESENCE_PENALTY,
    CONF_REASONING_EFFORT,
    CONF_REPEAT_PENALTY,
    CONF_REQUEST_BODY_PARAMS,
    CONF_SERVER_TYPE,
    CONF_SLOT_ID,
    CONF_STREAM,
    CONF_STRIP_EMOJI,
    CONF_STRIP_THINK,
    CONF_TASK_MAX_TOKENS,
    CONF_TASK_TEMPERATURE,
    CONF_TEMPERATURE,
    CONF_THINKING_MODE,
    CONF_THINKING_TOKEN_BUDGET,
    CONF_TOP_K,
    CONF_TOP_P,
    CONF_STT_MODEL,
    CONF_STT_SRC,
    CONF_TASK_MODEL,
    CONF_TASK_SRC,
    CONF_TTS_MODEL,
    CONF_TTS_SRC,
    CONF_TTS_VOICE,
    CUSTOM_PROVIDER,
    CONTEXT_ROLES,
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_ROLE,
    DEFAULT_CONTEXT_TEMPLATE,
    DEFAULT_HISTORY_MAX,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_MIN_P,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_STREAM,
    DEFAULT_STRIP_EMOJI,
    DEFAULT_STRIP_THINK,
    DEFAULT_TASK_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_DOUBAO_RESOURCE,
    DEFAULT_PROMPT,
    DOUBAO_BASE_URL,
    DOMAIN,
    MIMO_DEFAULT_TTS_MODEL,
    MIMO_DEFAULT_VOICE,
    MIMO_TTS_MODELS,
    MIMO_DEFAULT_VOICE,
    MIMO_TTS_VOICES,
    PROVIDER_IMAGE_BASE,
    COSYVOICE_VOICES,
    DEFAULT_OPENAI_VOICE,
    OPENAI_TTS_VOICES,
    PROTOCOL_OPTIONS,
    PROVIDER_LABELS,
    REASONING_EFFORTS,
    SERVER_TYPE_OPTIONS,
    THINKING_MODES,
    CONF_MAX_TOKENS,
    PROVIDER_OPTIONS,
    PROVIDERS,
    SRC_MAIN,
    is_chat_model,
    is_image_model,
    is_stt_model,
    is_tts_model,
    provider_label,
    short_model_name,
)

_LOGGER = logging.getLogger(__name__)

CAPABILITIES: dict[str, dict[str, str]] = {
    "llm": {"label": "对话模型", "key": CONF_LLM_MODEL, "src": CONF_LLM_SRC, "bucket": "chat"},
    "task": {"label": "AI 任务模型", "key": CONF_TASK_MODEL, "src": CONF_TASK_SRC, "bucket": "chat"},
    "image": {"label": "图片生成模型", "key": CONF_IMAGE_MODEL, "src": CONF_IMAGE_SRC, "bucket": "image"},
    "stt": {"label": "语音转文本模型", "key": CONF_STT_MODEL, "src": CONF_STT_SRC, "bucket": "stt"},
    "tts": {"label": "文本转语音模型", "key": CONF_TTS_MODEL, "src": CONF_TTS_SRC, "bucket": "tts"},
}

NONE_OPTION = "（不使用）"
OTHER_OPTION = "（使用其它提供商配置）"


def _kind_of(
    provider_or_url: str, base_url: str, protocol: str | None = None
) -> str:
    """Return provider kind: openai / doubao / mimo / anthropic / ollama."""
    if protocol in ("openai", "anthropic", "ollama"):
        return protocol
    name = PROVIDER_OPTIONS.get(provider_or_url, provider_or_url)
    if name == "豆包语音（火山）":
        return "doubao"
    if name == "小米 MiMo":
        return "mimo"
    if name == "Anthropic Claude":
        return "anthropic"
    if name.startswith("Ollama"):
        return "ollama"
    base = (base_url or "").lower()
    if "openspeech.bytedance" in base:
        return "doubao"
    if "xiaomimimo" in base:
        return "mimo"
    if "anthropic" in base:
        return "anthropic"
    if "11434" in base or "ollama" in base:
        return "ollama"
    return "openai"


async def _fetch_models(hass, base_url: str, api_key: str, kind: str) -> list[str]:
    """Fetch model ids; Doubao has a fixed STT model."""
    if kind == "doubao":
        return [DEFAULT_DOUBAO_RESOURCE]
    session = async_get_clientsession(hass)
    url = protocols.models_url(kind, base_url)
    headers = protocols.auth_headers(kind, api_key)
    try:
        async with session.get(
            url, headers=headers, timeout=aiohttp.ClientTimeout(total=25)
        ) as resp:
            if resp.status in (401, 403):
                raise ValueError("invalid_auth")
            if resp.status != 200:
                if kind in ("anthropic", "ollama"):
                    _LOGGER.warning(
                        "模型列表不可用（%s %s），请手动填写模型名", resp.status, url
                    )
                    return []
                raise ValueError("cannot_connect")
            data = await resp.json()
    except ValueError:
        raise
    except Exception:  # noqa: BLE001
        if kind in ("anthropic", "ollama"):
            _LOGGER.warning("模型列表获取失败（%s），请手动填写模型名", url)
            return []
        raise
    if kind == "ollama":
        return [
            model.get("name")
            for model in data.get("models", [])
            if model.get("name")
        ]
    return [model.get("id") for model in data.get("data", []) if model.get("id")]


def _voice_presets(kind: str, model: str) -> dict[str, str]:
    """Built-in voice list based on provider kind / model (from vendor docs)."""
    low = (model or "").lower()
    if kind == "mimo":
        return dict(MIMO_TTS_VOICES)
    if "cosyvoice" in low:
        return dict(COSYVOICE_VOICES)
    return dict(OPENAI_TTS_VOICES)


def _default_voice(kind: str, model: str) -> str:
    """Default voice for a provider kind / model."""
    low = (model or "").lower()
    if kind == "mimo":
        return MIMO_DEFAULT_VOICE
    if "cosyvoice" in low:
        return "中文女"
    return DEFAULT_OPENAI_VOICE


def _extract_voice_names(data) -> list[str]:
    """Extract voice names from a few common response shapes."""
    if isinstance(data, list):
        return [item for item in data if isinstance(item, str)]
    if not isinstance(data, dict):
        return []
    for key in ("voices", "data"):
        value = data.get(key)
        if isinstance(value, list):
            names = []
            for item in value:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    name = item.get("id") or item.get("name")
                    if name:
                        names.append(name)
            if names:
                return names
    return []


async def _fetch_remote_voices(hass, target: dict) -> dict[str, str]:
    """Try to fetch a voice list from the provider; empty on failure."""
    base = (target.get("base_url") or "").rstrip("/")
    if not base:
        return {}
    kind = target.get("kind") or "openai"
    if kind == "mimo":
        headers = {"api-key": target.get("api_key") or ""}
    elif kind == "anthropic":
        return {}
    else:
        headers = protocols.auth_headers(kind, target.get("api_key") or "")

    session = async_get_clientsession(hass)
    for path in ("/audio/voices", "/voices", "/tts/voices"):
        try:
            async with session.get(
                base + path,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    continue
                data = await resp.json()
        except Exception:  # noqa: BLE001
            continue
        names = _extract_voice_names(data)
        if names:
            return {name: name for name in names}
    return {}


async def _async_voice_options(hass, target: dict) -> dict[str, str]:
    """Merge built-in voices with any voice list the provider exposes."""
    voices = _voice_presets(target.get("kind") or "openai", target.get("model") or "")
    remote = await _fetch_remote_voices(hass, target)
    voices.update(remote)
    return voices


def _buckets(kind: str, models: list[str]) -> dict[str, list[str]]:
    """Group model ids into chat / image / stt / tts buckets."""
    if kind == "doubao":
        return {"chat": [], "image": [], "stt": [DEFAULT_DOUBAO_RESOURCE], "tts": []}
    out: dict[str, list[str]] = {"chat": [], "image": [], "stt": [], "tts": []}
    for model in models:
        if is_tts_model(model):
            out["tts"].append(model)
        elif is_stt_model(model):
            out["stt"].append(model)
        elif is_image_model(model):
            out["image"].append(model)
        elif is_chat_model(model):
            out["chat"].append(model)
    return out


class AIVoiceConfigFlow(ConfigFlow, domain=DOMAIN):
    """Multi-step flow: provider → models → (missing) extra provider loop."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the flow."""
        self._providers: list[dict[str, Any]] = []
        self._buckets: list[dict[str, list[str]]] = []
        self._selections: dict[str, tuple[int, str]] = {}
        self._prompt: str = DEFAULT_PROMPT
        self._custom_name: str = ""
        self._need_extra: list[str] = []
        self._pending_base_url: str = ""
        self._pending_provider: str = ""
        self._pending_models: list[str] = []
        self._pending_kind: str = "openai"
        self._pending_buckets: dict[str, list[str]] = {}
        self._apikey: str = ""

    # ------------------------------------------------------------------ step 1
    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step 1: initial provider (provider + api-key + optional name)."""
        errors: dict[str, str] = {}
        if user_input is not None:
            provider_url = user_input.get(CONF_PROVIDER) or DEFAULT_BASE_URL
            if provider_url == CUSTOM_PROVIDER:
                base_url = (user_input.get(CONF_BASE_URL) or "").strip()
                label = (user_input.get(CONF_NAME) or "").strip() or "自定义服务"
            else:
                base_url = provider_url
                label = PROVIDER_OPTIONS.get(provider_url, provider_url)
            if not base_url:
                errors["base"] = "need_base_url"
            else:
                kind = _kind_of(provider_url, base_url, user_input.get(CONF_PROTOCOL, "auto"))
                try:
                    models = await _fetch_models(self.hass, base_url, user_input[CONF_API_KEY], kind)
                except ValueError as err:
                    errors["base"] = str(err)
                except Exception:  # noqa: BLE001
                    _LOGGER.exception("Error fetching model list")
                    errors["base"] = "cannot_connect"
                else:
                    self._providers = [
                        {
                            "name": label,
                            "base_url": base_url,
                            "api_key": user_input[CONF_API_KEY],
                            "kind": kind,
                            "protocol": user_input.get(CONF_PROTOCOL, "auto"),
                        }
                    ]
                    self._buckets = [_buckets(kind, models)]
                    self._custom_name = (user_input.get(CONF_NAME) or "").strip()
                    self._prompt = (user_input.get(CONF_PROMPT) or DEFAULT_PROMPT).strip()
                    return await self.async_step_models()

        schema = vol.Schema(
            {
                vol.Required(CONF_PROVIDER, default=DEFAULT_BASE_URL): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": url, "label": label}
                            for url, label in PROVIDER_OPTIONS.items()
                        ]
                    )
                ),
                vol.Required(CONF_API_KEY): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.PASSWORD)
                ),
                vol.Optional(CONF_NAME): TextSelector(),
                vol.Optional(CONF_BASE_URL): TextSelector(),
                vol.Optional(CONF_PROTOCOL, default="auto"): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": value, "label": label}
                            for value, label in PROTOCOL_OPTIONS.items()
                        ],
                        sort=False,
                    )
                ),
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    async def async_step_add_provider(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Add one more provider (used to enrich the model dropdowns)."""
        errors: dict[str, str] = {}
        if user_input is not None:
            provider_url = user_input.get(CONF_PROVIDER) or DEFAULT_BASE_URL
            if provider_url == CUSTOM_PROVIDER:
                base_url = (user_input.get(CONF_BASE_URL) or "").strip()
                label = (user_input.get(CONF_NAME) or "").strip() or "自定义服务"
            else:
                base_url = provider_url
                label = PROVIDER_OPTIONS.get(provider_url, provider_url)
            if not base_url:
                errors["base"] = "need_base_url"
            elif any(p.get("base_url") == base_url for p in self._providers):
                errors["base"] = "already_added"
            else:
                kind = _kind_of(provider_url, base_url, user_input.get(CONF_PROTOCOL, "auto"))
                try:
                    models = await _fetch_models(
                        self.hass, base_url, user_input[CONF_API_KEY], kind
                    )
                except ValueError as err:
                    errors["base"] = str(err)
                except Exception:  # noqa: BLE001
                    _LOGGER.exception("Error fetching extra provider models")
                    errors["base"] = "cannot_connect"
                else:
                    self._providers.append(
                        {
                            "name": label,
                            "base_url": base_url,
                            "api_key": user_input[CONF_API_KEY],
                            "kind": kind,
                            "protocol": user_input.get(CONF_PROTOCOL, "auto"),
                        }
                    )
                    self._buckets.append(_buckets(kind, models))
                    self._pending_provider = label
                    self._pending_base_url = base_url
                    self._pending_kind = kind
                    self._pending_buckets = _buckets(kind, models)
                    self._apikey = user_input[CONF_API_KEY]
                    if self._need_extra:
                        return await self.async_step_extra_models()
                    return await self.async_step_models()
        schema = vol.Schema(
            {
                vol.Required(CONF_PROVIDER, default=DEFAULT_BASE_URL): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": url, "label": label}
                            for url, label in PROVIDER_OPTIONS.items()
                        ]
                    )
                ),
                vol.Required(CONF_API_KEY): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.PASSWORD)
                ),
                vol.Optional(CONF_NAME): TextSelector(),
                vol.Optional(CONF_BASE_URL): TextSelector(),
                vol.Optional(CONF_PROTOCOL, default="auto"): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": value, "label": label}
                            for value, label in PROTOCOL_OPTIONS.items()
                        ],
                        sort=False,
                    )
                ),
            }
        )
        return self.async_show_form(step_id="add_provider", data_schema=schema, errors=errors)

    # ------------------------------------------------------------------ step 2
    async def async_step_models(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step 2: pick the five models from the initial provider."""
        errors: dict[str, str] = {}
        if user_input is not None:
            self._prompt = user_input.get(CONF_PROMPT) or self._prompt or DEFAULT_PROMPT
            self._need_extra = []
            for cap, info in CAPABILITIES.items():
                raw = (user_input.get(info["key"]) or "").strip()
                src_index, model = 0, ""
                if raw == NONE_OPTION:
                    model = ""
                elif raw == OTHER_OPTION:
                    model = ""
                    self._need_extra.append(cap)
                else:
                    model = raw
                    src_index = self._provider_index_for(info["bucket"], model)
                self._selections[cap] = (src_index, model)
            if self._need_extra:
                # 只有主动选择「使用其它提供商配置」才进入添加提供商
                return await self.async_step_add_provider()
            return self._async_create()

        schema: dict[Any, Any] = {}
        for cap, info in CAPABILITIES.items():
            options: list[dict[str, str]] = []
            seen: set[str] = set()
            for index, provider in enumerate(self._providers):
                for model in self._buckets[index].get(info["bucket"], []):
                    if model in seen:
                        continue
                    seen.add(model)
                    options.append(
                        {"value": model, "label": f"{provider['name']} · {model}"}
                    )
            if options:
                schema[
                    vol.Optional(info["key"], default=options[0]["value"])
                ] = SelectSelector(
                    SelectSelectorConfig(
                        options=options
                        + [
                            {"value": NONE_OPTION, "label": NONE_OPTION},
                            {"value": OTHER_OPTION, "label": OTHER_OPTION},
                        ],
                        custom_value=True,
                    )
                )
            else:
                # 所有已添加提供商都没有这类模型：给出下拉（不使用/其它提供商），也允许手输
                schema[vol.Optional(info["key"])] = SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": NONE_OPTION, "label": NONE_OPTION},
                            {"value": OTHER_OPTION, "label": OTHER_OPTION},
                        ],
                        custom_value=True,
                    )
                )
        schema[vol.Optional(CONF_PROMPT, default=self._prompt)] = TextSelector(
            TextSelectorConfig(multiline=True)
        )
        return self.async_show_form(
            step_id="models",
            data_schema=vol.Schema(schema),
            description_placeholders={"provider": self._providers[0]["name"]},
        )

    # ------------------------------------------------------------------ step 3
    def _provider_index_for(self, bucket: str, model: str) -> int:
        """Find which provider offers this model."""
        for index, buckets in enumerate(self._buckets):
            if model in buckets.get(bucket, []):
                return index
        return 0

    def _missing(self) -> list[str]:
        """Capabilities that currently have no model configured."""
        return [cap for cap in CAPABILITIES if not (self._selections.get(cap) or (0, ""))[1]]

    # ------------------------------------------------------------------ step 4
    async def async_step_extra(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Add another provider that supplies the missing models."""
        errors: dict[str, str] = {}
        if user_input is not None:
            provider_url = user_input.get(CONF_PROVIDER) or DEFAULT_BASE_URL
            if provider_url == CUSTOM_PROVIDER:
                base_url = (user_input.get(CONF_BASE_URL) or "").strip()
                label = (user_input.get(CONF_NAME) or "").strip() or "自定义服务"
            else:
                base_url = provider_url
                label = PROVIDER_OPTIONS.get(provider_url, provider_url)
            if not base_url:
                errors["base"] = "need_base_url"
            else:
                kind = _kind_of(provider_url, base_url, user_input.get(CONF_PROTOCOL, "auto"))
                try:
                    models = await _fetch_models(self.hass, base_url, user_input[CONF_API_KEY], kind)
                except ValueError as err:
                    errors["base"] = str(err)
                except Exception:  # noqa: BLE001
                    _LOGGER.exception("Error fetching extra provider models")
                    errors["base"] = "cannot_connect"
                else:
                    self._pending_provider = label
                    self._pending_base_url = base_url
                    self._pending_kind = kind
                    self._pending_models = models
                    self._pending_buckets = _buckets(kind, models)
                    self._apikey = user_input[CONF_API_KEY]
                    return await self.async_step_extra_models()

        schema = vol.Schema(
            {
                vol.Required(CONF_PROVIDER, default=DEFAULT_BASE_URL): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": url, "label": label}
                            for url, label in PROVIDER_OPTIONS.items()
                        ]
                    )
                ),
                vol.Required(CONF_API_KEY): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.PASSWORD)
                ),
                vol.Optional(CONF_NAME): TextSelector(),
                vol.Optional(CONF_BASE_URL): TextSelector(),
                vol.Optional(CONF_PROTOCOL, default="auto"): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": value, "label": label}
                            for value, label in PROTOCOL_OPTIONS.items()
                        ],
                        sort=False,
                    )
                ),
            }
        )
        return self.async_show_form(step_id="extra", data_schema=schema, errors=errors)

    # ------------------------------------------------------------------ step 5
    async def async_step_extra_models(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Pick models for the missing capabilities from the extra provider."""
        target_caps = self._need_extra or self._missing()
        if user_input is not None:
            index = len(self._providers)
            added = {
                "name": self._pending_provider,
                "base_url": self._pending_base_url,
                "api_key": self._apikey,
                "kind": self._pending_kind,
            }
            if not any(p.get("base_url") == added["base_url"] for p in self._providers):
                self._providers.append(added)
                self._buckets.append(self._pending_buckets)
            else:
                index = next(
                    i for i, p in enumerate(self._providers) if p.get("base_url") == added["base_url"]
                )
            for cap in target_caps:
                model = (user_input.get(cap) or "").strip()
                if model == NONE_OPTION:
                    model = ""
                if model:
                    self._selections[cap] = (index, model)
            self._need_extra = []
            return self._async_create()

        buckets = self._pending_buckets
        schema: dict[Any, Any] = {}
        for cap in target_caps:
            info = CAPABILITIES[cap]
            options = buckets.get(info["bucket"], [])
            if options:
                schema[vol.Optional(cap, default=options[0])] = SelectSelector(
                    SelectSelectorConfig(
                        options=[{"value": m, "label": m} for m in sorted(options)]
                    )
                )
            else:
                schema[vol.Optional(cap)] = SelectSelector(
                    SelectSelectorConfig(
                        options=[{"value": NONE_OPTION, "label": NONE_OPTION}],
                        custom_value=True,
                    )
                )
        return self.async_show_form(
            step_id="extra_models",
            data_schema=vol.Schema(schema),
            description_placeholders={
                "provider": self._pending_provider,
                "missing": "、".join(CAPABILITIES[c]["label"] for c in target_caps),
            },
        )

    # ------------------------------------------------------------------ create
    def _async_create(self) -> ConfigFlowResult:
        main = self._providers[0]
        title = self._custom_name or main["name"]
        data: dict[str, Any] = {
            CONF_NAME: title,
            CONF_PROVIDER: main["name"],
            CONF_BASE_URL: main["base_url"],
            CONF_API_KEY: main["api_key"],
            CONF_PROMPT: self._prompt or DEFAULT_PROMPT,
            CONF_PROTOCOL: self._providers[0].get("protocol", "auto"),
            CONF_PROVIDERS_EXTRA: json.dumps(self._providers[1:], ensure_ascii=False)
            if len(self._providers) > 1
            else "[]",
        }
        for cap, info in CAPABILITIES.items():
            index, model = self._selections.get(cap, (0, ""))
            data[info["key"]] = model
            data[info["src"]] = SRC_MAIN if index == 0 else f"p{index - 1}"
        data[CONF_IMAGE_BASE_URL] = PROVIDER_IMAGE_BASE.get(
            main["base_url"], main["base_url"]
        )
        def provider_for(src: str) -> dict:
            return self._providers[0] if src == SRC_MAIN else self._providers[int(src[1:])]

        tts_provider = provider_for(data.get(CONF_TTS_SRC) or SRC_MAIN)
        if data.get(CONF_TTS_MODEL) and tts_provider["kind"] == "mimo":
            data[CONF_TTS_VOICE] = MIMO_DEFAULT_VOICE
        else:
            data[CONF_TTS_VOICE] = ""
        stt_provider = provider_for(data.get(CONF_STT_SRC) or SRC_MAIN)
        if data.get(CONF_STT_MODEL) and stt_provider["kind"] == "doubao":
            data[CONF_DOUBAO_RESOURCE] = DEFAULT_DOUBAO_RESOURCE
        return self.async_create_entry(title=title, data=data)

    @classmethod
    def async_get_options_flow(cls, config_entry):
        """Return the options flow."""
        return AIVoiceOptionsFlow()


class AIVoiceOptionsFlow(OptionsFlowWithReload):
    """二级菜单：模型设置 / 提示词设置 / 音色设置 / 高级设置。"""

    def __init__(self) -> None:
        self._data: dict[str, Any] | None = None
        self._collected: dict[str, Any] = {}
        self._need: list[str] = []
        self._new_provider: dict[str, Any] | None = None
        self._new_buckets: dict[str, list[str]] = {}
        self._api_key: str = ""
        self._buckets_list: list[dict[str, list[str]]] = []
        self._advanced_values: dict[str, Any] = {}

    @property
    def _entry_data(self) -> dict[str, Any]:
        if self._data is None:
            self._data = dict(self.config_entry.data)
        return self._data

    def _update_entry(self, new_data: dict[str, Any]) -> None:
        self._data = dict(new_data)
        entry = self.config_entry
        self.hass.config_entries.async_update_entry(
            entry,
            data=new_data,
            title=new_data.get(CONF_NAME)
            or new_data.get(CONF_PROVIDER)
            or "灵犀语音",
        )
        self.hass.config_entries.async_schedule_reload(entry.entry_id)

    # ------------------------------------------------------------ 菜单
    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show the settings menu."""
        data = self._entry_data
        menu = ["models", "prompt"]
        if data.get(CONF_TTS_MODEL):
            menu.append("voice")
        menu.extend(["advanced", "finish"])
        return self.async_show_menu(step_id="init", menu_options=menu)

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Finish the options flow (changes are already applied)."""
        return self.async_create_entry(title="", data={})

    # ------------------------------------------------------------ 模型
    async def async_step_models(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Pick models from every configured provider."""
        entry = self.config_entry
        data = self._entry_data
        errors: dict[str, str] = {}

        providers = list(get_providers(entry).values())
        buckets_list: list[dict[str, list[str]]] = []
        for provider in providers:
            try:
                models = await _fetch_models(
                    self.hass,
                    provider["base_url"],
                    provider["api_key"],
                    provider["kind"],
                )
            except Exception:  # noqa: BLE001
                models = []
            buckets_list.append(_buckets(provider["kind"], models))

        if user_input is not None:
            self._collected = dict(user_input)
            self._need = [
                cap
                for cap, info in CAPABILITIES.items()
                if (user_input.get(info["key"]) or "").strip() == OTHER_OPTION
            ]
            self._buckets_list = buckets_list
            if self._need:
                return await self.async_step_add_provider()
            return await self._apply_models(buckets_list)

        schema: dict[Any, Any] = {}
        for cap, info in CAPABILITIES.items():
            current_src = data.get(info["src"]) or SRC_MAIN
            src_index = 0 if current_src == SRC_MAIN else int(current_src[1:]) + 1
            current_model = data.get(info["key"]) or ""
            options: list[dict[str, str]] = []
            seen: set[str] = set()
            for index, provider in enumerate(providers):
                for model in buckets_list[index].get(info["bucket"], []):
                    if model in seen:
                        continue
                    seen.add(model)
                    options.append(
                        {"value": model, "label": f"{provider['name']} · {model}"}
                    )
            default_value = current_model
            if current_model and current_model not in seen:
                name = (
                    providers[src_index]["name"]
                    if 0 <= src_index < len(providers)
                    else providers[0]["name"]
                )
                options.append(
                    {"value": current_model, "label": f"{name} · {current_model}"}
                )
            options.append({"value": NONE_OPTION, "label": NONE_OPTION})
            options.append({"value": OTHER_OPTION, "label": OTHER_OPTION})
            schema[vol.Optional(info["key"], default=default_value)] = SelectSelector(
                SelectSelectorConfig(options=options, custom_value=True, sort=False)
            )
        return self.async_show_form(
            step_id="models", data_schema=vol.Schema(schema), errors=errors
        )

    async def _apply_models(
        self, buckets_list: list[dict[str, list[str]]]
    ) -> ConfigFlowResult:
        """Save the selected models against their providers."""
        new_data = dict(self._entry_data)
        for cap, info in CAPABILITIES.items():
            raw = (self._collected.get(info["key"]) or "").strip()
            model = ""
            src_index = 0
            if raw and raw not in (NONE_OPTION, OTHER_OPTION):
                model = raw
                for index, buckets in enumerate(buckets_list or []):
                    if model in buckets.get(info["bucket"], []):
                        src_index = index
                        break
            new_data[info["key"]] = model
            new_data[info["src"]] = (
                SRC_MAIN if src_index == 0 else f"p{src_index - 1}"
            )
        self._update_entry(new_data)
        return await self.async_step_init()

    # ------------------------------------------------------------ 提示词
    async def async_step_prompt(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit the conversation prompt."""
        data = self._entry_data
        if user_input is not None:
            new_data = dict(data)
            new_data[CONF_PROMPT] = (
                user_input.get(CONF_PROMPT) or ""
            ).strip() or DEFAULT_PROMPT
            self._update_entry(new_data)
            return await self.async_step_init()
        return self.async_show_form(
            step_id="prompt",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_PROMPT, default=data.get(CONF_PROMPT) or DEFAULT_PROMPT
                    ): TextSelector(TextSelectorConfig(multiline=True))
                }
            ),
        )

    # ------------------------------------------------------------ 音色
    async def async_step_voice(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit the TTS voice of the configured TTS entity."""
        data = self._entry_data
        target = resolve_target(self.config_entry, "tts")
        if user_input is not None:
            new_data = dict(data)
            new_data[CONF_TTS_VOICE] = (
                user_input.get(CONF_TTS_VOICE) or ""
            ).strip()
            self._update_entry(new_data)
            return await self.async_step_init()

        voices = await _async_voice_options(self.hass, target)
        return self.async_show_form(
            step_id="voice",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_TTS_VOICE,
                        default=data.get(CONF_TTS_VOICE)
                        or _default_voice(target["kind"], target["model"]),
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": value, "label": label}
                                for value, label in voices.items()
                            ],
                            custom_value=True,
                            sort=False,
                        )
                    )
                }
            ),
            description_placeholders={
                "provider": target.get("name") or "",
                "model": target.get("model") or "",
            },
        )

    # ------------------------------------------------------------ 高级
    def _advanced_schema(self, data: dict[str, Any]) -> vol.Schema:
        """Build the advanced-parameters form."""
        def number(key: str, default: float, low: float, high: float, step: float):
            return NumberSelector(
                NumberSelectorConfig(
                    min=low, max=high, step=step, mode=NumberSelectorMode.BOX
                )
            )

        def select(options: dict[str, str]):
            return SelectSelector(
                SelectSelectorConfig(
                    options=[
                        {"value": value, "label": label}
                        for value, label in options.items()
                    ],
                    sort=False,
                )
            )

        schema: dict[Any, Any] = {
            vol.Optional(
                CONF_SERVER_TYPE, default=data.get(CONF_SERVER_TYPE) or "generic"
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        {"value": value, "label": label}
                        for value, label in SERVER_TYPE_OPTIONS.items()
                    ],
                    sort=False,
                )
            ),
            vol.Optional(
                CONF_TEMPERATURE,
                default=float(data.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)),
            ): number(CONF_TEMPERATURE, DEFAULT_TEMPERATURE, 0, 2, 0.05),
            vol.Optional(
                CONF_MAX_TOKENS,
                default=int(data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)),
            ): number(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS, 32, 32768, 1),
            vol.Optional(
                CONF_TASK_TEMPERATURE,
                default=float(
                    data.get(
                        CONF_TASK_TEMPERATURE,
                        data.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                    )
                ),
            ): number(CONF_TASK_TEMPERATURE, DEFAULT_TEMPERATURE, 0, 2, 0.05),
            vol.Optional(
                CONF_TASK_MAX_TOKENS,
                default=int(data.get(CONF_TASK_MAX_TOKENS, DEFAULT_TASK_MAX_TOKENS)),
            ): number(CONF_TASK_MAX_TOKENS, DEFAULT_TASK_MAX_TOKENS, 32, 32768, 1),
            vol.Optional(
                CONF_HISTORY_MAX,
                default=int(data.get(CONF_HISTORY_MAX, DEFAULT_HISTORY_MAX)),
            ): number(CONF_HISTORY_MAX, DEFAULT_HISTORY_MAX, 2, 100, 1),
            vol.Optional(
                CONF_STRIP_THINK,
                default=bool(data.get(CONF_STRIP_THINK, DEFAULT_STRIP_THINK)),
            ): BooleanSelector(),
            vol.Optional(
                CONF_STRIP_EMOJI,
                default=bool(data.get(CONF_STRIP_EMOJI, DEFAULT_STRIP_EMOJI)),
            ): BooleanSelector(),
            vol.Optional(
                CONF_STREAM, default=bool(data.get(CONF_STREAM, DEFAULT_STREAM))
            ): BooleanSelector(),
            vol.Optional(
                CONF_PARALLEL_TOOL_CALLS,
                default=bool(data.get(CONF_PARALLEL_TOOL_CALLS, False)),
            ): BooleanSelector(),
            vol.Optional(
                CONF_MAX_TOOL_ITERATIONS,
                default=int(
                    data.get(CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS)
                ),
            ): number(
                CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS, 1, 12, 1
            ),
            vol.Optional(
                CONF_CONTEXT_TEMPLATE,
                default=data.get(CONF_CONTEXT_TEMPLATE, DEFAULT_CONTEXT_TEMPLATE),
            ): TextSelector(TextSelectorConfig(multiline=True)),
            vol.Optional(
                CONF_CONTEXT_ROLE,
                default=data.get(CONF_CONTEXT_ROLE, DEFAULT_CONTEXT_ROLE),
            ): select(CONTEXT_ROLES),
            vol.Optional(
                CONF_THINKING_MODE, default=data.get(CONF_THINKING_MODE, "none")
            ): select(THINKING_MODES),
            vol.Optional(
                CONF_REASONING_EFFORT, default=data.get(CONF_REASONING_EFFORT, "high")
            ): select(REASONING_EFFORTS),
            vol.Optional(
                CONF_THINKING_TOKEN_BUDGET,
                default=str(data.get(CONF_THINKING_TOKEN_BUDGET, "")),
            ): TextSelector(),
            vol.Optional(
                CONF_INCLUDE_PRIOR_THINKING,
                default=bool(data.get(CONF_INCLUDE_PRIOR_THINKING, True)),
            ): BooleanSelector(),
            vol.Optional(
                CONF_SLOT_ID, default=str(data.get(CONF_SLOT_ID, ""))
            ): TextSelector(),
            vol.Optional(
                CONF_TOP_P, default=float(data.get(CONF_TOP_P, DEFAULT_TOP_P))
            ): number(CONF_TOP_P, DEFAULT_TOP_P, 0, 1, 0.05),
            vol.Optional(
                CONF_MIN_P, default=float(data.get(CONF_MIN_P, DEFAULT_MIN_P))
            ): number(CONF_MIN_P, DEFAULT_MIN_P, 0, 1, 0.01),
            vol.Optional(
                CONF_TOP_K, default=int(data.get(CONF_TOP_K, DEFAULT_TOP_K))
            ): number(CONF_TOP_K, DEFAULT_TOP_K, 0, 1000, 1),
            vol.Optional(
                CONF_REPEAT_PENALTY,
                default=float(data.get(CONF_REPEAT_PENALTY, DEFAULT_REPEAT_PENALTY)),
            ): number(CONF_REPEAT_PENALTY, DEFAULT_REPEAT_PENALTY, 0, 2, 0.01),
            vol.Optional(
                CONF_PRESENCE_PENALTY,
                default=float(
                    data.get(CONF_PRESENCE_PENALTY, DEFAULT_PRESENCE_PENALTY)
                ),
            ): number(
                CONF_PRESENCE_PENALTY, DEFAULT_PRESENCE_PENALTY, -2, 2, 0.01
            ),
            vol.Optional(
                CONF_CHAT_TEMPLATE_ARGS,
                default=str(data.get(CONF_CHAT_TEMPLATE_ARGS, "")),
            ): TextSelector(TextSelectorConfig(multiline=True)),
            vol.Optional(
                CONF_REQUEST_BODY_PARAMS,
                default=str(data.get(CONF_REQUEST_BODY_PARAMS, "")),
            ): TextSelector(TextSelectorConfig(multiline=True)),
            vol.Optional(
                CONF_CUSTOM_HEADERS,
                default=str(data.get(CONF_CUSTOM_HEADERS, "")),
            ): TextSelector(TextSelectorConfig(multiline=True)),
        }
        return vol.Schema(schema)

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit advanced generation parameters."""
        data = self._entry_data
        errors: dict[str, str] = {}
        if user_input is not None:
            for key in (
                CONF_CHAT_TEMPLATE_ARGS,
                CONF_REQUEST_BODY_PARAMS,
                CONF_CUSTOM_HEADERS,
            ):
                raw = (user_input.get(key) or "").strip()
                if not raw:
                    continue
                try:
                    json.loads(raw)
                except ValueError:
                    if all(
                        "=" in line
                        for line in raw.splitlines()
                        if line.strip() and not line.strip().startswith("#")
                    ):
                        continue
                    errors["base"] = "invalid_json"
                    break
            if not errors:
                new_data = dict(data)
                new_data.update(user_input)
                self._update_entry(new_data)
                return await self.async_step_init()
        return self.async_show_form(
            step_id="advanced",
            data_schema=self._advanced_schema(data),
            errors=errors,
        )

    # ------------------------------------------------------------ 追加提供商
    async def async_step_add_provider(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Add another provider for the selected capabilities."""
        errors: dict[str, str] = {}
        if user_input is not None:
            provider_url = user_input.get(CONF_PROVIDER) or DEFAULT_BASE_URL
            if provider_url == CUSTOM_PROVIDER:
                base_url = (user_input.get(CONF_BASE_URL) or "").strip()
                label = (user_input.get(CONF_NAME) or "").strip() or "自定义服务"
            else:
                base_url = provider_url
                label = PROVIDER_OPTIONS.get(provider_url, provider_url)
            if not base_url:
                errors["base"] = "need_base_url"
            else:
                kind = _kind_of(
                    provider_url, base_url, user_input.get(CONF_PROTOCOL, "auto")
                )
                try:
                    models = await _fetch_models(
                        self.hass, base_url, user_input[CONF_API_KEY], kind
                    )
                except ValueError as err:
                    errors["base"] = str(err)
                except Exception:  # noqa: BLE001
                    _LOGGER.exception("Error fetching extra provider models")
                    errors["base"] = "cannot_connect"
                else:
                    self._new_provider = {
                        "name": label,
                        "base_url": base_url,
                        "api_key": user_input[CONF_API_KEY],
                        "kind": kind,
                        "protocol": user_input.get(CONF_PROTOCOL, "auto"),
                    }
                    self._new_buckets = _buckets(kind, models)
                    self._api_key = user_input[CONF_API_KEY]
                    return await self.async_step_extra_models()
        schema = vol.Schema(
            {
                vol.Required(CONF_PROVIDER, default=DEFAULT_BASE_URL): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": url, "label": label}
                            for url, label in PROVIDER_OPTIONS.items()
                        ]
                    )
                ),
                vol.Required(CONF_API_KEY): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.PASSWORD)
                ),
                vol.Optional(CONF_NAME): TextSelector(),
                vol.Optional(CONF_BASE_URL): TextSelector(),
                vol.Optional(CONF_PROTOCOL, default="auto"): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": value, "label": label}
                            for value, label in PROTOCOL_OPTIONS.items()
                        ],
                        sort=False,
                    )
                ),
            }
        )
        return self.async_show_form(
            step_id="add_provider", data_schema=schema, errors=errors
        )

    async def async_step_extra_models(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Pick models for the selected capabilities from the new provider."""
        if user_input is not None:
            for cap in self._need:
                model = (user_input.get(cap) or "").strip()
                if model == NONE_OPTION:
                    model = ""
                if model:
                    self._collected[CAPABILITIES[cap]["key"]] = model
                    self._collected[f"{cap}__new"] = "1"
            self._save_with_new_provider()
            return await self.async_step_init()

        schema: dict[Any, Any] = {}
        for cap in self._need:
            info = CAPABILITIES[cap]
            options = self._new_buckets.get(info["bucket"], [])
            if options:
                schema[vol.Optional(cap, default=options[0])] = SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": model, "label": f"{self._new_provider['name']} · {model}"}
                            for model in sorted(options)
                        ]
                    )
                )
            else:
                schema[vol.Optional(cap)] = SelectSelector(
                    SelectSelectorConfig(
                        options=[{"value": NONE_OPTION, "label": NONE_OPTION}],
                        custom_value=True,
                    )
                )
        return self.async_show_form(
            step_id="extra_models",
            data_schema=vol.Schema(schema),
            description_placeholders={
                "provider": (self._new_provider or {}).get("name", ""),
                "missing": "、".join(CAPABILITIES[c]["label"] for c in self._need),
            },
        )

    def _save_with_new_provider(self) -> None:
        """Append the new provider and save the selected models."""
        data = self._entry_data
        new_data = dict(data)
        try:
            extras = json.loads(data.get(CONF_PROVIDERS_EXTRA) or "[]")
        except ValueError:
            extras = []
        extras = extras if isinstance(extras, list) else []
        if self._new_provider and not any(
            (p or {}).get("base_url") == self._new_provider["base_url"]
            for p in extras
        ):
            extras.append(self._new_provider)
        new_data[CONF_PROVIDERS_EXTRA] = json.dumps(extras, ensure_ascii=False)
        new_index = 1 + next(
            (
                i
                for i, p in enumerate(extras)
                if (p or {}).get("base_url")
                == (self._new_provider or {}).get("base_url")
            ),
            len(extras) - 1,
        )
        for cap, info in CAPABILITIES.items():
            raw = (self._collected.get(info["key"]) or "").strip()
            model = ""
            src_index = 0
            if raw and raw not in (NONE_OPTION, OTHER_OPTION):
                model = raw
                if self._collected.get(f"{cap}__new"):
                    src_index = new_index
                elif raw not in self._new_buckets.get(info["bucket"], []):
                    src_index = 0
            new_data[info["key"]] = model
            new_data[info["src"]] = (
                SRC_MAIN if src_index == 0 else f"p{src_index - 1}"
            )
        self._update_entry(new_data)
