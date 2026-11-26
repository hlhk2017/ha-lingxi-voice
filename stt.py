"""Speech-to-text for 灵犀语音: OpenAI-compatible or Doubao SeedASR."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import uuid
import wave

import aiohttp

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .helpers import build_device_info, resolve_target
from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_DOUBAO_API_KEY,
    CONF_DOUBAO_RESOURCE,
    CONF_STT_MODEL,
    CONF_STT_PROVIDER,
    DEFAULT_DOUBAO_RESOURCE,
    DEFAULT_DOUBAO_UID,
    DOUBAO_BASE_URL,
    DOUBAO_QUERY_PATH,
    DOUBAO_SUBMIT_PATH,
    DOMAIN,
    MIMO_ASR_MODEL,
    STT_PROVIDER_DOUBAO,
    STT_PROVIDER_MIMO,
    STT_PROVIDER_NONE,
    STT_PROVIDER_OPENAI,
    short_model_name,
)

_LOGGER = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)
POLL_INTERVAL = 1.5
POLL_TIMEOUT = 40

DOUBAO_OK = {"20000000", "20000003"}
DOUBAO_PROCESSING = {"20000001", "20000002"}

LANGUAGES = ["zh", "en", "ja", "ko", "yue", "fr", "de", "es", "ru"]


def _resolve_provider(entry: ConfigEntry) -> str:
    """Resolve the STT engine from the per-capability provider config."""
    target = resolve_target(entry, "stt")
    kind = target["kind"]
    if not target["model"] and kind != "doubao":
        return STT_PROVIDER_NONE
    return kind


def _stt_enabled(entry: ConfigEntry) -> bool:
    return _resolve_provider(entry) != STT_PROVIDER_NONE


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the STT entity."""
    if _stt_enabled(entry):
        async_add_entities([AIVoiceSTTEntity(entry)])


def _pcm_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw 16-bit mono PCM into a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


class AIVoiceSTTEntity(SpeechToTextEntity):
    """Speech recognition using either OpenAI-compatible or Doubao SeedASR."""

    _attr_should_poll = False
    _attr_has_entity_name = True

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the entity."""
        self._entry = entry
        self._attr_unique_id = f"{DOMAIN}_stt_{entry.entry_id}"
        self._target = resolve_target(entry, "stt")
        self._provider = _resolve_provider(entry) or STT_PROVIDER_OPENAI
        self._model: str = (
            (self._target["model"] or "").strip()
            or (MIMO_ASR_MODEL if self._provider == STT_PROVIDER_MIMO else "")
        )
        self._api_key: str = self._target["api_key"]
        self._base_url: str = (self._target["base_url"] or "").rstrip("/")
        self._doubao_api_key: str = (
            self._target["api_key"] or entry.data.get(CONF_DOUBAO_API_KEY, "")
        )
        self._doubao_resource: str = entry.data.get(
            CONF_DOUBAO_RESOURCE, DEFAULT_DOUBAO_RESOURCE
        )
        if self._provider == STT_PROVIDER_DOUBAO:
            self._attr_name = "语音识别（豆包 SeedASR）"
        elif self._provider == STT_PROVIDER_MIMO:
            self._attr_name = "语音识别（MiMo ASR）"
        else:
            self._attr_name = f"语音识别（{short_model_name(self._model)}）"
        self._attr_device_info = build_device_info(entry, self._model)

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return supported formats."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return supported codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return supported bit rates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return supported sample rates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return supported channels."""
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream
    ) -> SpeechResult:
        """Transcribe one utterance."""
        try:
            pcm = b"".join([chunk async for chunk in stream])
            if not pcm:
                _LOGGER.warning("Empty audio stream received")
                return SpeechResult(None, SpeechResultState.ERROR)
            wav_bytes = _pcm_to_wav(pcm, int(metadata.sample_rate))
            if self._provider == STT_PROVIDER_DOUBAO:
                return await self._transcribe_doubao(wav_bytes, metadata)
            if self._provider == STT_PROVIDER_MIMO:
                return await self._transcribe_mimo(wav_bytes, metadata)
            return await self._transcribe_openai(wav_bytes)
        except TimeoutError:
            _LOGGER.error("ASR request timed out")
            return SpeechResult(None, SpeechResultState.ERROR)
        except Exception:  # noqa: BLE001
            _LOGGER.exception("Error processing audio stream")
            return SpeechResult(None, SpeechResultState.ERROR)

    # ------------------------------------------------------------------ OpenAI
    async def _transcribe_openai(self, wav_bytes: bytes) -> SpeechResult:
        """OpenAI-compatible /audio/transcriptions."""
        form = aiohttp.FormData()
        form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", self._model)
        session = async_get_clientsession(self.hass)
        async with session.post(
            f"{self._base_url}/audio/transcriptions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            data=form,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            body = await resp.text()
            if resp.status != 200:
                _LOGGER.error("ASR failed (%s): %s", resp.status, body[:300])
                return SpeechResult(None, SpeechResultState.ERROR)
        data = json.loads(body)
        text = (data.get("text") or "").strip()
        if not text:
            _LOGGER.warning("ASR returned an empty transcript")
            return SpeechResult(None, SpeechResultState.ERROR)
        _LOGGER.debug("Transcript: %s", text)
        return SpeechResult(text, SpeechResultState.SUCCESS)

    # -------------------------------------------------------------------- MiMo
    async def _transcribe_mimo(self, wav_bytes: bytes, metadata: SpeechMetadata) -> SpeechResult:
        """Xiaomi MiMo ASR: chat/completions with input_audio (base64 data URL)."""
        session = async_get_clientsession(self.hass)
        language = metadata.language if metadata.language in ("zh", "en") else "zh"
        body = {
            "model": self._model or MIMO_ASR_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": "data:audio/wav;base64,"
                                + base64.b64encode(wav_bytes).decode()
                            },
                        }
                    ],
                }
            ],
            "asr_options": {"language": language},
        }
        async with session.post(
            f"{self._base_url}/chat/completions",
            headers={"api-key": self._api_key, "Content-Type": "application/json"},
            json=body,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            payload = await resp.text()
            if resp.status != 200:
                _LOGGER.error("MiMo ASR failed (%s): %s", resp.status, payload[:300])
                return SpeechResult(None, SpeechResultState.ERROR)
        data = json.loads(payload)
        try:
            message = (data.get("choices") or [{}])[0].get("message") or {}
            text = message.get("content")
            if isinstance(text, list):
                text = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in text
                )
            if text is None:
                text = (
                    message.get("asr_result")
                    or data.get("text")
                    or ""
                )
        except (AttributeError, IndexError, TypeError):
            text = ""
        text = (text or "").strip()
        if not text:
            # 静音/唤醒后没接语音：返回空结果（SUCCESS），
            # 让 pipeline 用“没听清”这类友好提示，而不是播报错误音
            _LOGGER.warning("MiMo ASR returned an empty transcript (silence?)")
            return SpeechResult("", SpeechResultState.SUCCESS)
        _LOGGER.debug("MiMo transcript: %s", text)
        return SpeechResult(text, SpeechResultState.SUCCESS)

    # ------------------------------------------------------------------ Doubao
    def _doubao_headers(self, req_id: str) -> dict[str, str]:
        return {
            "X-Api-Key": self._doubao_api_key,
            "X-Api-Request-Id": req_id,
            "X-Api-Resource-Id": self._doubao_resource,
            "Content-Type": "application/json",
        }

    async def _transcribe_doubao(
        self, wav_bytes: bytes, metadata: SpeechMetadata
    ) -> SpeechResult:
        """Doubao SeedASR: submit then poll."""
        if not self._doubao_api_key:
            _LOGGER.error("Doubao SeedASR API key is not configured")
            return SpeechResult(None, SpeechResultState.ERROR)

        req_id = str(uuid.uuid4())
        session = async_get_clientsession(self.hass)
        submit_body = {
            "user": {"uid": DEFAULT_DOUBAO_UID},
            "audio": {
                "data": base64.b64encode(wav_bytes).decode(),
                "format": "wav",
                "rate": int(metadata.sample_rate),
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
            },
        }
        async with session.post(
            f"{DOUBAO_BASE_URL}{DOUBAO_SUBMIT_PATH}",
            headers=self._doubao_headers(req_id),
            json=submit_body,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            code = resp.headers.get("X-Api-Status-Code", "")
            body = await resp.text()
            if resp.status != 200 or code not in DOUBAO_OK:
                _LOGGER.error(
                    "SeedASR submit failed (%s / %s): %s", resp.status, code, body[:300]
                )
                return SpeechResult(None, SpeechResultState.ERROR)

        deadline = asyncio.get_event_loop().time() + POLL_TIMEOUT
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(POLL_INTERVAL)
            async with session.post(
                f"{DOUBAO_BASE_URL}{DOUBAO_QUERY_PATH}",
                headers=self._doubao_headers(req_id),
                json={"user": DEFAULT_DOUBAO_UID},
                timeout=REQUEST_TIMEOUT,
            ) as resp:
                code = resp.headers.get("X-Api-Status-Code", "")
                body = await resp.text()
            if code in DOUBAO_PROCESSING:
                continue
            if code not in DOUBAO_OK:
                _LOGGER.error(
                    "SeedASR query failed (%s / %s): %s", resp.status, code, body[:300]
                )
                return SpeechResult(None, SpeechResultState.ERROR)
            try:
                data = json.loads(body) if body else {}
            except ValueError:
                data = {}
            text = ((data.get("result") or {}).get("text") or "").strip()
            if text:
                _LOGGER.debug("SeedASR transcript: %s", text)
                return SpeechResult(text, SpeechResultState.SUCCESS)
            if code == "20000003":
                return SpeechResult(None, SpeechResultState.ERROR)

        _LOGGER.error("SeedASR timed out waiting for the transcript")
        return SpeechResult(None, SpeechResultState.ERROR)