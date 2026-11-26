"""Shared helpers for the AI Voice integration."""

from __future__ import annotations

import json

from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo

from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_DOUBAO_API_KEY,
    CONF_IMAGE_MODEL,
    CONF_IMAGE_SRC,
    CONF_LLM_MODEL,
    CONF_LLM_SRC,
    CONF_NAME,
    CONF_PROVIDER,
    CONF_PROVIDERS_EXTRA,
    CONF_STT_MODEL,
    CONF_STT_PROVIDER,
    CONF_STT_SRC,
    CONF_TASK_MODEL,
    CONF_TASK_SRC,
    CONF_TTS_MODEL,
    CONF_TTS_PROVIDER,
    CONF_TTS_SRC,
    CONF_TTS_VOICE,
    DEFAULT_DOUBAO_UID,
    DOUBAO_BASE_URL,
    DOMAIN,
    PROVIDER_KINDS,
    SRC_MAIN,
    STT_PROVIDER_DOUBAO,
    STT_PROVIDER_MIMO,
    TTS_PROVIDER_MIMO,
    provider_label,
    short_model_name,
)

CAPABILITY_MODEL_KEY = {
    "llm": CONF_LLM_MODEL,
    "task": CONF_TASK_MODEL,
    "image": CONF_IMAGE_MODEL,
    "stt": CONF_STT_MODEL,
    "tts": CONF_TTS_MODEL,
}
CAPABILITY_SRC_KEY = {
    "llm": CONF_LLM_SRC,
    "task": CONF_TASK_SRC,
    "image": CONF_IMAGE_SRC,
    "stt": CONF_STT_SRC,
    "tts": CONF_TTS_SRC,
}


def provider_kind(name: str | None, base_url: str | None) -> str:
    """Classify a provider as openai / doubao / mimo."""
    if name in PROVIDER_KINDS:
        return PROVIDER_KINDS[name]
    base = (base_url or "").lower()
    if "xiaomimimo" in base:
        return "mimo"
    if "openspeech.bytedance" in base:
        return "doubao"
    return "openai"


def get_providers(entry: ConfigEntry) -> dict[str, dict]:
    """Return all configured providers: main + extra ones."""
    data = entry.data
    main_base = data.get(CONF_BASE_URL) or ""
    providers: dict[str, dict] = {
        SRC_MAIN: {
            "name": data.get(CONF_PROVIDER) or "main",
            "base_url": main_base,
            "api_key": data.get(CONF_API_KEY) or "",
            "kind": provider_kind(data.get(CONF_PROVIDER), main_base),
        }
    }
    try:
        extras = json.loads(data.get(CONF_PROVIDERS_EXTRA) or "[]")
    except ValueError:
        extras = []
    for index, item in enumerate(extras if isinstance(extras, list) else []):
        if not isinstance(item, dict):
            continue
        base = item.get("base_url") or ""
        providers[f"p{index}"] = {
            "name": item.get("name") or f"提供商 {index + 1}",
            "base_url": base,
            "api_key": item.get("api_key") or "",
            "kind": item.get("kind") or provider_kind(item.get("name"), base),
        }
    return providers


def resolve_target(entry: ConfigEntry, capability: str) -> dict:
    """Resolve the provider + model used by one capability.

    Returns: {name, base_url, api_key, kind, model, voice}
    """
    data = entry.data
    providers = get_providers(entry)
    src = data.get(CAPABILITY_SRC_KEY[capability]) or SRC_MAIN
    provider = providers.get(src) or providers[SRC_MAIN]
    result = {
        "name": provider["name"],
        "base_url": provider["base_url"],
        "api_key": provider["api_key"],
        "kind": provider["kind"],
        "model": data.get(CAPABILITY_MODEL_KEY[capability]) or "",
        "voice": data.get(CONF_TTS_VOICE) or "",
    }

    # ---- 旧配置兼容（没有 *_src 字段的条目） ----
    if capability == "stt":
        legacy = data.get(CONF_STT_PROVIDER) or ""
        if not data.get(CONF_STT_SRC):
            if legacy == STT_PROVIDER_DOUBAO:
                result.update(
                    base_url=DOUBAO_BASE_URL,
                    api_key=data.get(CONF_DOUBAO_API_KEY) or provider["api_key"],
                    kind="doubao",
                )
            elif legacy == STT_PROVIDER_MIMO:
                result["kind"] = "mimo"
    if capability == "tts" and not data.get(CONF_TTS_SRC):
        if (data.get(CONF_TTS_PROVIDER) or "") == TTS_PROVIDER_MIMO:
            result["kind"] = "mimo"
    return result


def resolve_uid(entry: ConfigEntry) -> str:
    """Stable uid used by Doubao SeedASR requests."""
    return DEFAULT_DOUBAO_UID


def build_device_info(entry: ConfigEntry, model: str | None = None, version: str = "1.1.0") -> DeviceInfo:
    """Return a shared DeviceInfo so all platforms show one device per provider."""
    data = entry.data
    # 设备型号统一显示对话主模型，避免被 AI 任务/STT 模型覆盖
    model = data.get(CONF_LLM_MODEL) or model
    display_name = data.get(CONF_NAME) or data.get(CONF_PROVIDER) or "灵犀语音"
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=display_name,
        manufacturer=provider_label(data.get(CONF_PROVIDER)),
        model=short_model_name(model),
        entry_type=DeviceEntryType.SERVICE,
        configuration_url=data.get(CONF_BASE_URL),
        sw_version=version,
    )