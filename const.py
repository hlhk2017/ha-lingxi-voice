"""Constants for the AI Voice integration."""

DOMAIN = "ai_voice"

CONF_PROVIDER = "provider"
CONF_NAME = "name"
CONF_API_KEY = "api_key"
CONF_BASE_URL = "base_url"
CONF_LLM_MODEL = "llm_model"
CONF_STT_MODEL = "stt_model"
CONF_IMAGE_MODEL = "image_model"
CONF_TASK_MODEL = "task_model"
CONF_IMAGE_BASE_URL = "image_base_url"
CONF_STT_PROVIDER = "stt_provider"
CONF_TTS_PROVIDER = "tts_provider"
CONF_TTS_MODEL = "tts_model"
CONF_TTS_VOICE = "tts_voice"
CONF_USE_DOUBAO = "use_doubao"
CONF_PROVIDERS_EXTRA = "extra_providers"
CONF_LLM_SRC = "llm_src"
CONF_TASK_SRC = "task_src"
CONF_IMAGE_SRC = "image_src"
CONF_STT_SRC = "stt_src"
CONF_TTS_SRC = "tts_src"
SRC_MAIN = "main"
CONF_DOUBAO_API_KEY = "doubao_api_key"
CONF_DOUBAO_RESOURCE = "doubao_resource"

# STT 引擎类型
STT_PROVIDER_NONE = ""
STT_PROVIDER_OPENAI = "openai"
STT_PROVIDER_DOUBAO = "doubao"
STT_PROVIDER_MIMO = "mimo"

# TTS 引擎（目前支持小米 MiMo）
TTS_PROVIDER_NONE = ""
TTS_PROVIDER_MIMO = "mimo"

TTS_PROVIDER_OPTIONS = {
    TTS_PROVIDER_NONE: "（不使用语音合成）",
    TTS_PROVIDER_MIMO: "MiMo TTS（小米）",
}

MIMO_ASR_MODEL = "mimo-v2.5-asr"
MIMO_TTS_MODELS = [
    "mimo-v2.5-tts",
    "mimo-v2.5-tts-voicedesign",
    "mimo-v2.5-tts-voiceclone",
]
MIMO_TTS_VOICES = {
    "mimo_default": "MiMo-默认",
    "冰糖": "冰糖（女·中文）",
    "茉莉": "茉莉（女·中文）",
    "苏打": "苏打（男·中文）",
    "白桦": "白桦（男·中文）",
    "Mia": "Mia（女·英文）",
    "Chloe": "Chloe（女·英文）",
    "Milo": "Milo（男·英文）",
    "Dean": "Dean（男·英文）",
}
MIMO_DEFAULT_TTS_MODEL = "mimo-v2.5-tts"
MIMO_DEFAULT_VOICE = "mimo_default"

STT_PROVIDER_OPTIONS = {
    STT_PROVIDER_NONE: "（不使用语音识别）",
    STT_PROVIDER_OPENAI: "OpenAI 兼容（/audio/transcriptions）",
    STT_PROVIDER_DOUBAO: "豆包 SeedASR（火山引擎）",
    STT_PROVIDER_MIMO: "MiMo ASR（小米）",
}

# 豆包（火山引擎）语音识别
DOUBAO_BASE_URL = "https://openspeech.bytedance.com/api/v3"
DOUBAO_SUBMIT_PATH = "/auc/bigmodel/submit"
DOUBAO_QUERY_PATH = "/auc/bigmodel/query"
DEFAULT_DOUBAO_RESOURCE = "volc.seedasr.auc"
DOUBAO_RESOURCE_OPTIONS = [
    "volc.seedasr.auc",
    "volc.seedasr.sauc.duration",
]
DEFAULT_DOUBAO_UID = "ha-voice"
CONF_PROMPT = "prompt"
CONF_TEMPERATURE = "temperature"
CONF_MAX_TOKENS = "max_tokens"

# 常见 OpenAI 兼容提供商（选择后自动填充 base_url，也可选“自定义”手动填）
PROVIDERS: dict[str, str] = {
    "Anthropic Claude": "https://api.anthropic.com",
    "Ollama（本地，需填宿主机地址）": "http://localhost:11434",
    "Ollama Cloud": "https://ollama.com",
    "硅基流动 SiliconFlow": "https://api.siliconflow.cn/v1",
    "DeepSeek 深度求索": "https://api.deepseek.com/v1",
    "阿里云百炼 DashScope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "月之暗面 Moonshot": "https://api.moonshot.cn/v1",
    "智谱 BigModel": "https://open.bigmodel.cn/api/paas/v4",
    "Groq": "https://api.groq.com/openai/v1",
    "OpenAI": "https://api.openai.com/v1",
    "华为云 MaaS（按量付费）": "https://api.modelarts-maas.com/openai/v1",
    "华为云 MaaS（Token Plan）": "https://api.modelarts-maas.com/plan/v2",
    "火山引擎 方舟（Ark）": "https://ark.cn-beijing.volces.com/api/v3",
    "小米 MiMo": "https://api.xiaomimimo.com/v1",
    "豆包语音（火山）": "https://openspeech.bytedance.com/api/v3",
    "自定义 Custom": "",
}

# 自定义服务商的哨兵值（不能用空字符串，否则必填校验会失败）
CUSTOM_PROVIDER = "__custom__"

# 特殊服务商（非标准 OpenAI 兼容，或需要特殊协议）
DOUBAO_PROVIDER_LABEL = "豆包语音（火山）"
PROVIDER_KINDS: dict[str, str] = {
    "豆包语音（火山）": "doubao",
    "小米 MiMo": "mimo",
    "Anthropic Claude": "anthropic",
    "Ollama（本地，需填宿主机地址）": "ollama",
    "Ollama Cloud": "ollama",
}
PROVIDER_KIND_LABELS: dict[str, str] = {
    "openai": "OpenAI 兼容",
    "doubao": "豆包 SeedASR",
    "mimo": "小米 MiMo",
    "anthropic": "Anthropic 兼容",
    "ollama": "Ollama 原生",
}

# UI 下拉用 {url: 中文名}，这样界面显示中文名、提交的是 URL
PROVIDER_OPTIONS: dict[str, str] = {
    url: label for label, url in PROVIDERS.items() if url
}
PROVIDER_OPTIONS[CUSTOM_PROVIDER] = "自定义 Custom"
PROVIDER_LABELS: dict[str, str] = dict(PROVIDERS)

# 部分服务商的“图像生成”端点与对话端点不在同一路径
PROVIDER_IMAGE_BASE: dict[str, str] = {
    # 华为云 MaaS：对话 /openai/v1，生图 /v1/images/generations
    "https://api.modelarts-maas.com/openai/v1": "https://api.modelarts-maas.com/v1",
    "https://api.modelarts-maas.com/plan/v2": "https://api.modelarts-maas.com/v1",
}

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

DEFAULT_PROMPT = (
    "你是 Home Assistant 的家庭语音助手。"
    "用简短、口语化的中文回答，一到两句话，适合直接朗读出来；"
    "不要使用列表、Markdown、换行或表情；不要读实体 ID、技术参数或代码。"
    "需要控制设备时，直接调用工具完成，不要反问用户。"
)

DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 240
MAX_TOOL_ITERATIONS = 4

# ---- 模型分类（尽量覆盖已知的国内外模型命名）----

# 语音识别（STT）
STT_KEYWORDS = (
    "asr",
    "whisper",
    "sensevoice",
    "paraformer",
    "transcribe",
    "transcription",
    "speech-to-text",
    "speech_to_text",
    "stt",
    "gsr",  # 生成式语音识别（如 XingChenGSR）
    "wav2vec",
    "moonshine",
    "vosk",
    "sherpa",
    "funasr",
    "qwen3-asr",
    "qwen-audio-asr",
    "sense-voice",
    "doubao-asr",
    "doubao-streaming-asr",
    "doubao-realtime-asr",
    "doubao-speech",
    "volc-asr",
)

# 图像生成 / 图像编辑
IMAGE_KEYWORDS = (
    "image",  # Qwen-Image / Z-Image / MOSS-Image / ERNIE-Image / gpt-image / Imagen
    "flux",
    "kolors",
    "stable-diffusion",
    "stable_diffusion",
    "sd3",
    "sd-3",
    "sd-xl",
    "sdxl",
    "dall-e",
    "dalle",
    "imagen",
    "seedream",
    "doubao-seedream",
    "kling",
    "janus",
    "cogview",
    "wanx",
    "wan2",
    "hidream",
    "lumina",
    "pixart",
    "playground",
    "text-to-image",
    "t2i",
    "img2img",
    "image-edit",
    "image_edit",
    "image-editing",
)

# 明确不属于“对话模型”的其他类别（TTS / 向量 / 重排 / OCR / 代码补全等）
NON_CHAT_KEYWORDS = (
    "tts",
    "cosyvoice",
    "moss-ttsd",
    "f5-tts",
    "spark-tts",
    "bark",
    "vits",
    "melo",
    "embedding",
    "embed",
    "bge",  # BAAI/bge-* 向量与重排模型
    "rerank",
    "vision-encoder",
    "code_generation",
    "ocr",
    "video",
    "2v",
    "doubao-tts",
    "doubao-voice",
    "音色",
    "pixverse",
    "vidu",
    "sora",
    "veo",
    "kling-video",
) + STT_KEYWORDS + IMAGE_KEYWORDS


def is_stt_model(model_id: str) -> bool:
    """Return True if the model id looks like a speech-to-text model."""
    low = model_id.lower()
    return any(k in low for k in STT_KEYWORDS)


TTS_KEYWORDS = ("tts", "cosyvoice", "sovits", "chattts", "fish-speech", "fishspeech")


def is_tts_model(model_id: str) -> bool:
    """Return True if the model id looks like a text-to-speech model."""
    low = model_id.lower()
    return any(k in low for k in TTS_KEYWORDS)


def is_image_model(model_id: str) -> bool:
    """Return True if the model id looks like an image generation/editing model."""
    low = model_id.lower()
    return any(k in low for k in IMAGE_KEYWORDS)


def is_chat_model(model_id: str) -> bool:
    """Return True if the model id looks usable for chat."""
    low = model_id.lower()
    if is_stt_model(model_id) or is_image_model(model_id):
        return False
    return not any(k in low for k in NON_CHAT_KEYWORDS)


def provider_label(provider_or_url: str | None) -> str:
    """Return a human friendly provider label from a label or an API URL."""
    if not provider_or_url:
        return "AI 服务"
    if provider_or_url == CUSTOM_PROVIDER:
        return "自定义服务"
    # 已经是中文名
    if provider_or_url in PROVIDER_LABELS:
        return provider_or_url
    # 传了 URL，转成中文名
    return PROVIDER_OPTIONS.get(provider_or_url, provider_or_url)


def short_model_name(model_id: str | None) -> str:
    """Return the part after the last slash, for compact entity names."""
    return (model_id or "").split("/")[-1] or "未设置"


# ======================================================================
# v1.2.0 高级设置（对标 Local OpenAI LLM 的能力）
# ======================================================================

CONF_PROTOCOL = "protocol"
CONF_SERVER_TYPE = "server_type"
CONF_HISTORY_MAX = "history_max_messages"
CONF_STRIP_THINK = "strip_think"
CONF_STRIP_EMOJI = "strip_emoji"
CONF_STREAM = "stream"
CONF_PARALLEL_TOOL_CALLS = "parallel_tool_calls"
CONF_MAX_TOOL_ITERATIONS = "max_tool_iterations"
CONF_CHAT_TEMPLATE_ARGS = "chat_template_args"
CONF_REQUEST_BODY_PARAMS = "request_body_params"
CONF_CUSTOM_HEADERS = "custom_headers"
CONF_CONTEXT_TEMPLATE = "context_injection_template"
CONF_CONTEXT_ROLE = "context_injection_role"

# 推理服务端专项
CONF_THINKING_MODE = "thinking_mode"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_THINKING_TOKEN_BUDGET = "thinking_token_budget"
CONF_INCLUDE_PRIOR_THINKING = "include_prior_thinking"
CONF_SLOT_ID = "slot_id"
CONF_USE_LOADED_MODEL = "use_loaded_model"
CONF_TOP_P = "top_p"
CONF_MIN_P = "min_p"
CONF_TOP_K = "top_k"
CONF_REPEAT_PENALTY = "repeat_penalty"
CONF_PRESENCE_PENALTY = "presence_penalty"

# AI Task 专用采样
CONF_TASK_TEMPERATURE = "task_temperature"
CONF_TASK_MAX_TOKENS = "task_max_tokens"

DEFAULT_HISTORY_MAX = 24
DEFAULT_STRIP_THINK = True
DEFAULT_STRIP_EMOJI = False
DEFAULT_STREAM = False
DEFAULT_PARALLEL_TOOL_CALLS = False
DEFAULT_MAX_TOOL_ITERATIONS = 4
DEFAULT_TASK_MAX_TOKENS = 800
DEFAULT_CONTEXT_ROLE = "system"
DEFAULT_CONTEXT_TEMPLATE = (
    "{% set t = now() %}当前时间是 {{ t.strftime('%Y年%m月%d日 %H:%M') }}（{{ "
    "'一二三四五六日'[t.weekday()] }}）。"
)
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.0
DEFAULT_TOP_K = 0
DEFAULT_REPEAT_PENALTY = 1.0
DEFAULT_PRESENCE_PENALTY = 0.0

PROTOCOL_OPTIONS: dict[str, str] = {
    "auto": "自动识别（按服务商/地址）",
    "openai": "OpenAI 兼容（/chat/completions）",
    "anthropic": "Anthropic 兼容（/v1/messages）",
    "ollama": "Ollama 原生（/api/chat）",
}

# server_type → 中文名
SERVER_TYPES: dict[str, str] = {
    "generic": "通用 OpenAI 兼容",
    "deepseek": "DeepSeek Cloud",
    "llama_cpp": "llama.cpp",
    "vllm": "vLLM",
    "localai": "LocalAI",
    "ollama": "Ollama",
}
SERVER_TYPE_OPTIONS: dict[str, str] = {k: v for k, v in SERVER_TYPES.items()}

THINKING_MODES: dict[str, str] = {
    "none": "（不干预）",
    "always": "始终思考",
    "first_iteration": "仅首轮思考",
}
REASONING_EFFORTS: dict[str, str] = {
    "high": "High（标准）",
    "max": "Max（最大）",
}
CONTEXT_ROLES: dict[str, str] = {
    "system": "System（推荐）",
    "tool": "Tool Result",
    "assistant": "Assistant",
    "user": "User",
    "none": "禁用",
}

EVENT_GENERATION_COMPLETE = "ai_voice_generation_complete"
