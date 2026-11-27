"""TTS platform for 灵犀语音 — currently supports Xiaomi MiMo TTS."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import aiohttp

from homeassistant.components.tts import TextToSpeechEntity, TtsAudioType
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .helpers import resolve_target

from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_TTS_MODEL,
    CONF_TTS_PROVIDER,
    CONF_TTS_VOICE,
    DOMAIN,
    MIMO_DEFAULT_TTS_MODEL,
    MIMO_DEFAULT_VOICE,
    MIMO_TTS_VOICES,
    TTS_PROVIDER_MIMO,
    short_model_name,
)

_LOGGER = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=120)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the TTS entity when a TTS engine is configured."""
    target = resolve_target(entry, "tts")
    if target["model"] and target["kind"] in ("mimo", "openai"):
        async_add_entities([AIVoiceTTSEntity(entry)])


class AIVoiceTTSEntity(TextToSpeechEntity):
    """TTS entity: OpenAI-compatible (/audio/speech) or Xiaomi MiMo."""

    _attr_should_poll = False
    _attr_has_entity_name = True


    def __init__(self, entry: ConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = f"{DOMAIN}_tts_{entry.entry_id}"
        model = resolve_target(entry, "tts")["model"] or MIMO_DEFAULT_TTS_MODEL
        self._attr_name = f"语音合成（{short_model_name(model)}）"

    @property
    def _target(self) -> dict:
        return resolve_target(self._entry, "tts")

    @property
    def _current_model(self) -> str:
        return self._target["model"] or MIMO_DEFAULT_TTS_MODEL

    @property
    def _current_voice(self) -> str:
        return self._target["voice"] or MIMO_DEFAULT_VOICE

    @property
    def default_language(self) -> str:
        return "zh"

    @property
    def supported_languages(self) -> list[str]:
        return ["zh", "en"]

    @property
    def supported_options(self) -> list[str]:
        return [CONF_TTS_VOICE]

    @property
    def voice(self) -> str:
        return MIMO_TTS_VOICES.get(self._current_voice, self._current_voice)

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            name=self._entry.title,
            manufacturer="Xiaomi MiMo",
            model=self._current_model,
        )

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Synthesize speech with the configured TTS provider."""
        target = self._target
        if target["kind"] == "openai":
            return await self._async_openai_speech(target, message, options)
        opts = options or {}
        voice = opts.get(CONF_TTS_VOICE) or self._current_voice
        body = {
            "model": self._current_model,
            "messages": [{"role": "assistant", "content": message}],
            "audio": {"format": "wav", "voice": voice},
            "stream": False,
        }
        session = async_get_clientsession(self.hass)
        base = (self._target["base_url"] or "").rstrip("/")
        url = f"{base}/chat/completions"
        async with session.post(
            url,
            headers={
                "api-key": self._target["api_key"],
                "Content-Type": "application/json",
            },
            json=body,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            payload = await resp.text()
            if resp.status != 200:
                _LOGGER.error(
                    "MiMo TTS failed (%s %s): %s", resp.status, url, payload[:300]
                )
                raise RuntimeError(f"tts_api_http_{resp.status}")
        data = json.loads(payload)
        try:
            message_obj = (data.get("choices") or [{}])[0].get("message") or {}
            audio = message_obj.get("audio") or {}
            b64 = audio.get("data")
        except (AttributeError, IndexError, TypeError):
            b64 = None
        if not b64:
            _LOGGER.error("MiMo TTS returned no audio: %s", payload[:300])
            raise RuntimeError("tts_api_no_audio")
        return "wav", base64.b64decode(b64)

    async def _async_openai_speech(
        self, target: dict, message: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """OpenAI-compatible /audio/speech."""
        opts = options or {}
        body = {
            "model": target["model"],
            "input": message,
            "voice": opts.get(CONF_TTS_VOICE) or self._current_voice or "alloy",
            "response_format": "mp3",
        }
        session = async_get_clientsession(self.hass)
        url = f"{target['base_url'].rstrip('/')}/audio/speech"
        async with session.post(
            url,
            headers={
                "Authorization": f"Bearer {target['api_key']}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            if resp.status != 200:
                payload = await resp.text()
                _LOGGER.error(
                    "TTS /audio/speech failed (%s %s): %s", resp.status, url, payload[:300]
                )
                raise RuntimeError(f"tts_api_http_{resp.status}")
            audio = await resp.read()
        if not audio:
            raise RuntimeError("tts_api_empty_audio")
        return "mp3", audio