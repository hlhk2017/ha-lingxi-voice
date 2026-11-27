"""AI Task platform (Home Assistant "AI 建议" / AI Task entities)."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

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
from homeassistant.components.conversation import AssistantContent
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.llm import selector_serializer
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import api
from .const import (
    CONF_IMAGE_MODEL,
    CONF_LLM_MODEL,
    CONF_TASK_MAX_TOKENS,
    CONF_TASK_MODEL,
    CONF_TASK_TEMPERATURE,
    CONF_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TASK_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    PROVIDER_IMAGE_BASE,
    short_model_name,
)
from .helpers import build_device_info, resolve_target

_LOGGER = logging.getLogger(__name__)

IMAGE_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for text-to-image models. "
    "Translate the user's request into a single vivid English image prompt. "
    "Reply with the prompt only, no explanations."
)

EVAL_SYSTEM_PROMPT = (
    "You evaluate questions about Home Assistant state. "
    "Answer strictly as JSON. For each question id return an object: "
    '{"type":"noul","noul":<probability 0..1>} for yes/no, '
    '{"type":"choice","choice":"<one of the criteria keys>","probabilities":{<key>:<probability>}} '
    "for choice, and "
    '{"type":"score","score":<expected level index>,"probabilities":[<prob per level>]} '
    "for score. Probabilities must sum to 1. "
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
        self._attr_device_info = build_device_info(
            entry, entry.data.get(CONF_TASK_MODEL) or entry.data[CONF_LLM_MODEL]
        )

    # ------------------------------------------------------------------
    # 目标解析
    # ------------------------------------------------------------------
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
    def _temperature(self) -> float:
        """Temperature used for AI tasks."""
        return api.get_float(
            self._entry,
            CONF_TASK_TEMPERATURE,
            api.get_float(self._entry, CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
        )

    @property
    def _max_tokens(self) -> int:
        """Max output tokens used for AI tasks."""
        return api.get_int(self._entry, CONF_TASK_MAX_TOKENS, DEFAULT_TASK_MAX_TOKENS)

    @property
    def supported_features(self) -> AITaskEntityFeature:
        """Expose image generation only when an image model is configured now."""
        features = (
            AITaskEntityFeature.GENERATE_DATA
            | AITaskEntityFeature.EVALUATE
            | AITaskEntityFeature.SUPPORT_ATTACHMENTS
        )
        if self._image_model_current:
            features |= AITaskEntityFeature.GENERATE_IMAGE
        return features

    # ------------------------------------------------------------------
    # 通用请求
    # ------------------------------------------------------------------
    async def _async_chat(
        self,
        chat_log,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        iteration: int = 0,
    ) -> dict[str, Any]:
        """Call /chat/completions and return the first choice's message."""
        body: dict[str, Any] = {
            "model": self._chat_model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": max(self._max_tokens, DEFAULT_MAX_TOKENS),
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        variables = api.conversation_variables(
            self.hass, getattr(chat_log, "conversation_id", None)
        )
        data = await api.async_chat_completion(
            self.hass,
            self._entry,
            self._chat_target,
            body,
            variables=variables,
            iteration=iteration,
        )
        return api.first_choice(data).get("message") or {}

    @staticmethod
    def _schema_hint(messages: list[dict[str, Any]], structure: Any) -> None:
        """Append a JSON-Schema instruction to the system message."""
        schema = probatio.to_openapi(
            structure,
            custom_serializer=selector_serializer,
            openapi_version="3.1.0",
        )
        hint = (
            " Respond with a single JSON object that validates against this JSON Schema: "
            + json.dumps(schema, ensure_ascii=False)
        )
        for message in messages:
            if message.get("role") == "system":
                message["content"] = (message.get("content") or "") + hint
                return
        messages.insert(0, {"role": "system", "content": hint.strip()})

    # ------------------------------------------------------------------
    # 数据生成（支持附件 + 工具）
    # ------------------------------------------------------------------
    async def _async_generate_data(
        self, task: GenDataTask, chat_log
    ) -> GenDataTaskResult:
        """Generate structured/text data for an AI Task."""
        tools = api.convert_tools(chat_log.llm_api)
        max_iterations = 3 if tools else 1
        answer_text = ""

        for iteration in range(max_iterations):
            messages = await api.convert_content_with_attachments(
                self.hass, chat_log.content, include_thinking=False
            )
            if task.structure is not None:
                self._schema_hint(messages, task.structure)

            message = await self._async_chat(
                chat_log,
                messages,
                json_mode=task.structure is not None,
                tools=tools or None,
                iteration=iteration,
            )
            tool_calls = api.parse_tool_calls(message.get("tool_calls") or [])
            content_text = api.clean_output(self._entry, message.get("content")) or ""

            if tool_calls:
                content = AssistantContent(
                    agent_id=self.entity_id,
                    content=content_text or None,
                    tool_calls=tool_calls,
                )
                async for _result in chat_log.async_add_assistant_content(content):
                    pass
                continue

            answer_text = content_text
            break

        if task.structure is not None and answer_text:
            try:
                parsed = json.loads(answer_text)
            except ValueError:
                parsed = answer_text
            return GenDataTaskResult(
                conversation_id=chat_log.conversation_id, data=parsed
            )
        return GenDataTaskResult(
            conversation_id=chat_log.conversation_id, data=answer_text
        )

    # ------------------------------------------------------------------
    # 决策评估
    # ------------------------------------------------------------------
    async def _async_evaluate(self, task: EvaluationTask) -> EvaluationTaskResult:
        """Answer typed evaluation questions."""
        questions = {qid: dict(q) for qid, q in task.questions.items()}
        payload = {
            "questions": questions,
            "state": task.state,
        }
        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            },
        ]
        body: dict[str, Any] = {
            "model": self._chat_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max(self._max_tokens, 800),
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        try:
            data = await api.async_chat_completion(
                self.hass,
                self._entry,
                self._chat_target,
                body,
                variables=api.conversation_variables(self.hass, None),
            )
            text = (api.first_choice(data).get("message") or {}).get("content") or ""
            raw = json.loads(text).get("answers") or {}
        except (RuntimeError, ValueError):
            _LOGGER.warning("Evaluation request failed", exc_info=True)
            raw = {}

        answers = {}
        for qid, question in questions.items():
            item = raw.get(qid) or {}
            qtype = question.get("type")
            if qtype == "noul":
                answers[qid] = NoulAnswer(
                    noul=_clamp(_as_float(item.get("noul"), 0.0), 0.0, 1.0)
                )
            elif qtype == "choice":
                criteria = question.get("criteria") or {}
                probabilities = _normalize(
                    {
                        str(k): _as_float(v, 0.0)
                        for k, v in (item.get("probabilities") or {}).items()
                        if str(k) in criteria
                    }
                )
                choice = str(item.get("choice") or "")
                if choice not in criteria:
                    choice = max(
                        probabilities, key=probabilities.get, default=""
                    ) or next(iter(criteria), "")
                answers[qid] = ChoiceAnswer(choice=choice, probabilities=probabilities)
            else:  # score
                levels = question.get("criteria") or []
                raw_probs = [
                    _as_float(value, 0.0) for value in (item.get("probabilities") or [])
                ][: len(levels)]
                if levels and len(raw_probs) != len(levels):
                    raw_probs = [1.0 / len(levels)] * len(levels)
                probabilities = _normalize_list(raw_probs)
                score = _as_float(item.get("score"), 0.0)
                if levels and not 0 <= score <= len(levels) - 1:
                    score = (
                        max(range(len(probabilities)), key=probabilities.__getitem__)
                        if probabilities
                        else 0.0
                    )
                answers[qid] = ScoreAnswer(score=score, probabilities=probabilities)
        return EvaluationTaskResult(answers=answers)

    # ------------------------------------------------------------------
    # 图片生成
    # ------------------------------------------------------------------
    async def _async_generate_image(
        self, task: GenImageTask, chat_log
    ) -> GenImageTaskResult:
        """Generate an image with the configured image model."""
        image_model = self._image_model_current
        if not image_model:
            raise RuntimeError("No image model configured")

        session = async_get_clientsession(self.hass)
        image_target = self._image_target
        headers = api.build_headers(
            self._entry,
            {**image_target, "kind": "openai"},
            self.hass,
            api.conversation_variables(self.hass, getattr(chat_log, "conversation_id", None)),
        )

        # 1) 用对话模型把中文需求转成英文提示词
        prompt = task.instructions
        try:
            message = await self._async_chat(
                chat_log,
                [
                    {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": task.instructions},
                ],
            )
            rewritten = (message.get("content") or "").strip()
            prompt = api.strip_think(rewritten) or task.instructions
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
            timeout=api.REQUEST_TIMEOUT,
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
                async with session.get(
                    first["url"], timeout=api.REQUEST_TIMEOUT
                ) as img_resp:
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


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_list(values: list[float]) -> list[float]:
    """Normalize a probability list."""
    total = sum(values)
    if total <= 0:
        return []
    return [value / total for value in values]


def _normalize(values: dict[str, float]) -> dict[str, float]:
    """Normalize a probability map, falling back to an empty map."""
    total = sum(values.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in values.items()}