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
    SelectSelector,
    SelectSelectorConfig,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .helpers import get_providers

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
    CONF_PROVIDER,
    CONF_PROVIDERS_EXTRA,
    CONF_STT_MODEL,
    CONF_STT_SRC,
    CONF_TASK_MODEL,
    CONF_TASK_SRC,
    CONF_TTS_MODEL,
    CONF_TTS_SRC,
    CONF_TTS_VOICE,
    CUSTOM_PROVIDER,
    DEFAULT_BASE_URL,
    DEFAULT_DOUBAO_RESOURCE,
    DEFAULT_PROMPT,
    DOUBAO_BASE_URL,
    DOMAIN,
    MIMO_DEFAULT_TTS_MODEL,
    MIMO_DEFAULT_VOICE,
    MIMO_TTS_MODELS,
    MIMO_TTS_VOICES,
    PROVIDER_IMAGE_BASE,
    PROVIDER_LABELS,
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


def _auth_headers(base_url: str, api_key: str) -> dict[str, str]:
    if "xiaomimimo.com" in (base_url or ""):
        return {"api-key": api_key}
    return {"Authorization": f"Bearer {api_key}"}


def _kind_of(provider_or_url: str, base_url: str) -> str:
    """Return provider kind: openai / doubao / mimo."""
    name = PROVIDER_OPTIONS.get(provider_or_url, provider_or_url)
    if name == "豆包语音（火山）":
        return "doubao"
    if name == "小米 MiMo":
        return "mimo"
    if "openspeech.bytedance" in (base_url or ""):
        return "doubao"
    if "xiaomimimo" in (base_url or ""):
        return "mimo"
    return "openai"


async def _fetch_models(hass, base_url: str, api_key: str, kind: str) -> list[str]:
    """Fetch model ids; Doubao has a fixed STT model."""
    if kind == "doubao":
        return [DEFAULT_DOUBAO_RESOURCE]
    session = async_get_clientsession(hass)
    async with session.get(
        f"{base_url.rstrip('/')}/models",
        headers=_auth_headers(base_url, api_key),
        timeout=aiohttp.ClientTimeout(total=25),
    ) as resp:
        if resp.status != 200:
            raise ValueError("invalid_auth" if resp.status in (401, 403) else "cannot_connect")
        data = await resp.json()
    return [m.get("id") for m in data.get("data", []) if m.get("id")]


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
                kind = _kind_of(provider_url, base_url)
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
                kind = _kind_of(provider_url, base_url)
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
                kind = _kind_of(provider_url, base_url)
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
    """Edit models of an existing entry; also supports adding another provider."""

    def __init__(self) -> None:
        self._collected: dict[str, str] = {}
        self._need: list[str] = []
        self._new_provider: dict[str, Any] | None = None
        self._new_buckets: dict[str, list[str]] = {}
        self._api_key: str = ""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show each model as a provider-prefixed dropdown."""
        entry = self.config_entry
        data = dict(entry.data)
        errors: dict[str, str] = {}

        providers = list(get_providers(entry).values())
        buckets_list: list[dict[str, list[str]]] = []
        for provider in providers:
            try:
                models = await _fetch_models(
                    self.hass, provider["base_url"], provider["api_key"], provider["kind"]
                )
            except Exception:  # noqa: BLE001
                models = []
            buckets_list.append(_buckets(provider["kind"], models))

        if user_input is not None:
            self._collected = {
                k: v for k, v in user_input.items() if k != CONF_PROMPT
            }
            self._collected[CONF_PROMPT] = user_input.get(CONF_PROMPT) or DEFAULT_PROMPT
            self._need = []
            for cap, info in CAPABILITIES.items():
                if (user_input.get(info["key"]) or "").strip() == OTHER_OPTION:
                    self._need.append(cap)
            if self._need:
                return await self.async_step_add_provider()
            return self._async_save(buckets_list)

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
        schema[
            vol.Optional(CONF_PROMPT, default=data.get(CONF_PROMPT) or DEFAULT_PROMPT)
        ] = TextSelector(TextSelectorConfig(multiline=True))
        return self.async_show_form(
            step_id="init", data_schema=vol.Schema(schema), errors=errors
        )

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
                kind = _kind_of(provider_url, base_url)
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
            }
        )
        return self.async_show_form(
            step_id="add_provider", data_schema=schema, errors=errors
        )

    async def async_step_extra_models(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Pick models for the selected capabilities from the new provider."""
        entry = self.config_entry
        if user_input is not None:
            for cap in self._need:
                model = (user_input.get(cap) or "").strip()
                if model == NONE_OPTION:
                    model = ""
                if model:
                    self._collected[CAPABILITIES[cap]["key"]] = model
                    self._collected[f"{cap}__new"] = "1"
            return self._async_save_with_new_provider(buckets_list=None)

        schema: dict[Any, Any] = {}
        for cap in self._need:
            info = CAPABILITIES[cap]
            options = self._new_buckets.get(info["bucket"], [])
            if options:
                schema[vol.Optional(cap, default=options[0])] = SelectSelector(
                    SelectSelectorConfig(
                        options=[{"value": m, "label": f"{self._new_provider['name']} · {m}"} for m in sorted(options)]
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

    def _async_save(self, buckets_list: list[dict[str, list[str]]]) -> ConfigFlowResult:
        """Save collected selections against the existing providers."""
        entry = self.config_entry
        data = dict(entry.data)
        providers = list(get_providers(entry).values())
        new_data = dict(data)
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
            new_data[info["src"]] = SRC_MAIN if src_index == 0 else f"p{src_index - 1}"
        new_data[CONF_PROMPT] = self._collected.get(CONF_PROMPT) or DEFAULT_PROMPT
        self.hass.config_entries.async_update_entry(
            entry,
            data=new_data,
            title=data.get(CONF_NAME) or data.get(CONF_PROVIDER) or "灵犀语音",
        )
        self.hass.config_entries.async_schedule_reload(entry.entry_id)
        return self.async_create_entry(title="", data={})

    def _async_save_with_new_provider(self, buckets_list=None) -> ConfigFlowResult:
        """Save selections, appending the newly added provider."""
        entry = self.config_entry
        data = dict(entry.data)
        new_data = dict(data)
        try:
            extras = json.loads(data.get(CONF_PROVIDERS_EXTRA) or "[]")
        except ValueError:
            extras = []
        extras = extras if isinstance(extras, list) else []
        if self._new_provider and not any(
            (p or {}).get("base_url") == self._new_provider["base_url"] for p in extras
        ):
            extras.append(self._new_provider)
        new_data[CONF_PROVIDERS_EXTRA] = json.dumps(extras, ensure_ascii=False)
        new_index = 1 + next(
            (
                i
                for i, p in enumerate(extras)
                if (p or {}).get("base_url") == (self._new_provider or {}).get("base_url")
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
                # 其它类别：沿用原有提供商（这里简化取 main 或已有模型匹配）
                elif raw not in self._new_buckets.get(info["bucket"], []):
                    src_index = 0
            new_data[info["key"]] = model
            new_data[info["src"]] = SRC_MAIN if src_index == 0 else f"p{src_index - 1}"
        new_data[CONF_PROMPT] = self._collected.get(CONF_PROMPT) or DEFAULT_PROMPT
        self.hass.config_entries.async_update_entry(
            entry,
            data=new_data,
            title=data.get(CONF_NAME) or data.get(CONF_PROVIDER) or "灵犀语音",
        )
        self.hass.config_entries.async_schedule_reload(entry.entry_id)
        return self.async_create_entry(title="", data={})
