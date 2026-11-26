"""Nuoan AI Voice - OpenAI-compatible conversation + STT + AI Task integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .helpers import resolve_target

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.CONVERSATION,
    Platform.STT,
    Platform.AI_TASK,
    Platform.TTS,
]


def _platforms_for(entry: ConfigEntry) -> list[Platform]:
    """Return the platforms enabled by this entry."""
    platforms = list(PLATFORMS)
    stt = resolve_target(entry, "stt")
    if not stt["model"] and stt["kind"] != "doubao":
        platforms.remove(Platform.STT)
    tts = resolve_target(entry, "tts")
    if not (tts["kind"] in ("mimo", "openai") and tts["model"]):
        platforms.remove(Platform.TTS)
    return platforms


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up 灵犀语音 from a config entry."""
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entry
    await hass.config_entries.async_forward_entry_setups(entry, _platforms_for(entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unloaded = await hass.config_entries.async_unload_platforms(
        entry, _platforms_for(entry)
    )
    if unloaded:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    return unloaded