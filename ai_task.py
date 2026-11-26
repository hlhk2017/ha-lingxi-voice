"""AI Task platform (Home Assistant "AI 建议" / AI Task entities)."""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

import aiohttp
import probatio

from homeassistant.components.ai_task import (
    AITaskEntity,
    AITaskEntityFeature,
    ChoiceAnswer,
    EvaluationTask,
    EvaluationTaskResult,
    GenDataTask,
    GenDataTaskResult,
    GenImageTask,
    GenImageTaskResult,
    NoulAnswer,
    ScoreAnswer,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .helpers import build_device_info, resolve_target
from .const import (
    PROVIDER_IMAGE_BASE,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_IMAGE_BASE_URL,
    CONF_IMAGE_MODEL,
    CONF_LLM_MODEL,
    CONF_TASK_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    short_model_name,
)

_LOGGER = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=120)

IMAGE_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for text-to-image models. "
    "Translate the user's request into a single vivid English image prompt. "
    "Reply with the prompt only, no explanations."
)

EVAL_SYSTEM_PROMPT = (
    "You evaluate questions about Home Assistant state. "
    "Answer strictly as JSON. For each question id return an object: "
    '{"type":"noul","noul":<probability 0..1>} for yes/no, '
    '{"type":"choice","choice":"<one of the criteria keys>","probabilities":{...}} for choice, '
    '{"type":"score","score":<expected level index>,"probabilities":[...]} for score. '
    'Wrap all answers in {"answers": {...}}.'
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the AI Task entity."""
    async_add_entities([AIVoiceAITaskEntity(entry)])


class AIVoiceAITaskEntity(AITaskEntity):
    """AI Task entity that reuses the configured chat (and optional image) model."""

    _attr_should_poll = False
    _attr_has_entity_name = True

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the AI task entity."""
        self._entry = entry
        self._attr_unique_id = f"{DOMAIN}_ai_task_{entry.entry_id}"
        self._attr_name = f"AI 任务（{short_model_name(entry.data.get(CONF_TASK_MODEL) or entry.data[CONF_LLM_MODEL])}）"
        self._attr_supported_features = AITaskEntityFeature.GENERATE_DATA | AITaskEntityFeature.EVALUATE
        if entry.data.get(CONF_IMAGE_MODEL):
            self._attr_supported_features |= AITaskEntityFeature.GENERATE_IMAGE
        self._model: str = entry.data.get(CONF_TASK_MODEL) or entry.data[CONF_LLM_MODEL]
        self._attr_device_info = build_device_info(entry, self._model)

    @property
    def _chat_target(self) -> dict:
        """Provider + model used for AI tasks (chat)."""
        return resolve_target(self._entry, "task")

    @property
    def _image_target(self) -> dict:
        """Provider + model used for image generation."""
        target = resolve_target(self._entry, "image")
        base = (target["base_url"] or "").rstrip("/")
        base = PROVIDER_IMAGE_BASE.get(target["base_url"], base)
        return {**target, "base_url": base}

    @property
    def _chat_model(self) -> str:
        """Always read the current task/chat model from the entry data."""
        return self._chat_target["model"]

    @property
    def _image_model_current(self) -> str:
        """Always read the current image model from the entry data."""
        return self._image_target["model"]

    @property
    def supported_features(self) -> AITaskEntityFeature:
        """Expose image generation only when an image model is configured now."""
        features = AITaskEntityFeature.GENERATE_DATA | AITaskEntityFeature.EVALUATE
        if self._image_model_current:
            features |= AITaskEntityFeature.GENERATE_IMAGE
        return features

    async def _async_chat(self, messages: list[dict[str, Any]], *, json_mode: bool = False) -> str:
        """Call /chat/completions and return the assistant text."""
        body: dict[str, Any] = {
            "model": self._chat_model,
            "messages": messages,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": max(DEFAULT_MAX_TOKENS, 800),
            "stream": False,
        }
        target = self._chat_target
        if target["kind"] == "mimo":
            body["thinking"] = {"type": "disabled"}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        session = async_get_clientsession(self.hass)
        if target["kind"] == "mimo":
            headers = {"api-key": target["api_key"], "Content-Type": "application/json"}
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
        return ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""

    async def _async_generate_data(
        self, task: GenDataTask, chat_log
    ) -> GenDataTaskResult:
        """Generate structured/text data for an AI Task."""
        messages = [
            {"role": "system", "content": "You are a helpful Home Assistant expert. Follow the instructions exactly."},
            {"role": "user", "content": task.instructions},
        ]
        if task.structure is not None:
            schema = probatio.to_openapi(task.structure, openapi_version="3.1.0")
            messages[0]["content"] += (
                " Respond with a single JSON object that validates against this JSON Schema: "
                + json.dumps(schema, ensure_ascii=False)
            )
            text = await self._async_chat(messages, json_mode=True)
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = text
            return GenDataTaskResult(
                conversation_id=chat_log.conversation_id, data=parsed
            )

        text = await self._async_chat(messages)
        return GenDataTaskResult(conversation_id=chat_log.conversation_id, data=text)

    async def _async_evaluate(self, task: EvaluationTask) -> EvaluationTaskResult:
        """Answer typed evaluation questions."""
        questions = {qid: dict(q) for qid, q in task.questions.items()}
        payload = {
            "questions": questions,
            "state": task.state,
        }
        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ]
        text = await self._async_chat(messages, json_mode=True)
        try:
            raw = json.loads(text).get("answers") or {}
        except ValueError:
            _LOGGER.warning("Evaluation returned non-JSON: %s", text[:200])
            raw = {}

        answers = {}
        for qid, question in questions.items():
            item = raw.get(qid) or {}
            qtype = question.get("type")
            if qtype == "noul":
                try:
                    answers[qid] = NoulAnswer(noul=float(item.get("noul", 0)))
                except (TypeError, ValueError):
                    answers[qid] = NoulAnswer(noul=0.0)
            elif qtype == "choice":
                choice = str(item.get("choice") or "")
                if choice not in (question.get("criteria") or {}):
                    choice = next(iter((question.get("criteria") or {}).keys()), "")
                answers[qid] = ChoiceAnswer(choice=choice, probabilities={})
            else:  # score
                try:
                    answers[qid] = ScoreAnswer(score=float(item.get("score", 0)), probabilities=[])
                except (TypeError, ValueError):
                    answers[qid] = ScoreAnswer(score=0.0, probabilities=[])
        return EvaluationTaskResult(answers=answers)

    async def _async_generate_image(
        self, task: GenImageTask, chat_log
    ) -> GenImageTaskResult:
        """Generate an image with the configured image model."""
        image_model = self._image_model_current
        if not image_model:
            raise RuntimeError("No image model configured")

        session = async_get_clientsession(self.hass)
        image_target = self._image_target
        headers = {
            "Authorization": f"Bearer {image_target['api_key']}",
            "Content-Type": "application/json",
        }

        # 1) 用对话模型把中文需求转成英文提示词
        prompt = task.instructions
        try:
            prompt = await self._async_chat(
                [
                    {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": task.instructions},
                ]
            ) or task.instructions
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Prompt rewrite failed, using raw instructions")

        # 2) 请求图像生成（兼容 OpenAI 风格与 SiliconFlow 风格）
        payload = {
            "model": image_model,
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
        }
        async with session.post(
            f"{image_target['base_url']}/images/generations",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            body = await resp.text()
            if resp.status != 200:
                _LOGGER.error("Image API error %s: %s", resp.status, body[:300])
                raise RuntimeError(f"image_api_http_{resp.status}")
        data = json.loads(body)

        image_bytes: bytes | None = None
        items = data.get("images") or data.get("data") or []
        if items:
            first = items[0]
            if first.get("b64_json"):
                b64 = first["b64_json"]
                if b64.startswith("data:"):
                    # 华为云 MaaS 返回 data:image/png;base64,xxx
                    b64 = b64.split(",", 1)[1]
                image_bytes = base64.b64decode(b64)
            elif first.get("url"):
                async with session.get(first["url"], timeout=REQUEST_TIMEOUT) as img_resp:
                    image_bytes = await img_resp.read()
            elif first.get("image"):
                image_bytes = base64.b64decode(first["image"])

        if not image_bytes:
            raise RuntimeError("image_api_no_image")

        return GenImageTaskResult(
            image_data=image_bytes,
            conversation_id=chat_log.conversation_id,
            mime_type="image/png",
            width=1024,
            height=1024,
            model=image_model,
            revised_prompt=prompt,
        )