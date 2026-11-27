# 灵犀语音（ai_voice）更新记录

## 1.3.0 — 新增 Anthropic 与 Ollama 原生协议

### 新增

- **LLM 协议显式选择**：配置流程新增「协议」选项（`protocol`）：
  - 自动识别（按服务商/地址推断，兼容旧条目）
  - OpenAI 兼容（`/chat/completions`，原有行为）
  - Anthropic 兼容（`/v1/messages`）
  - Ollama 原生（`/api/chat`）
- **协议适配层**（新文件 `protocols.py`）：
  - Anthropic：`x-api-key` + `anthropic-version` 认证；`system` 字段分离；
    `tool_use` / `tool_result` 双向转换（工具结果合并进 user 消息）；
    `thinking` block 与 `thinking_delta` 映射为思考内容；
    多模态 `image` block；SSE 事件流（message_start / content_block_* / message_delta）。
  - Ollama：NDJSON 流式；`options`（temperature / num_predict / top_p / top_k /
    repeat_penalty / presence_penalty）；tools 透传；`images` 多模态；
    `prompt_eval_count` / `eval_count` / `total_duration` 归一化为 usage/timings。
  - 模型列表：Anthropic 走 `/v1/models`、Ollama 走 `/api/tags`；拉取失败不阻塞配置，
    可手动填写模型名。
- **服务商预设**：Anthropic Claude、Ollama（本地）、Ollama Cloud。
- **平台裁剪**：STT / TTS 仅在 OpenAI / 豆包 / MiMo 协议下加载，Anthropic、Ollama 条目
  只生成对话与 AI Task 实体。

### 修复

- 流式工具调用此前把 OpenAI 格式的 dict 直接交给 `ChatLog`，触发
  `'dict' object has no attribute 'external'` 并回退非流式（导致工具重复执行）；
  现在统一转换为 `llm.ToolInput`。该问题在 1.2.0 已存在，仅流式 + 工具场景可见。

### 验证（HA 2026.11.1，本地 mock 服务端到端）

| 用例 | Anthropic | Ollama |
| --- | --- | --- |
| 非流式对话 | 通过 | 通过 |
| 流式对话（SSE / NDJSON） | 通过 | 通过 |
| 流式工具调用（开灯） | 通过 | 通过 |
| 工具结果回传模型 | 通过 | 通过 |
| 结构化 AI Task 生成 | 通过 | 通过 |
| `ai_task.evaluate` 不崩溃 | 通过 | 通过 |
| 模型列表拉取 | 通过 | 通过 |
| STT/TTS 不加载 | 通过 | 通过 |
| OpenAI 兼容回归（MiMo） | 通过 | — |

请求转换已核对：Anthropic 请求的 `system` 独立字段、`tool_use` / `tool_result` 结构、
`input_schema` 工具定义；Ollama 请求的 `options.num_predict`、assistant `tool_calls`
（arguments 为对象）、`tool` 消息。

### 已知限制

- Anthropic / Ollama 的 `ai_task.evaluate` 依赖 JSON 提示词，不用 `response_format`，
  概率来自模型输出解析；未接 TypeSafe `/v1/systemone`。
- Anthropic 不支持回传历史 thinking（缺少 signature），`include_prior_thinking` 对其无效。
- 图片输入：代码已支持（Anthropic `image` block / Ollama `images`），本轮仅验证 OpenAI 路径。

### 致谢

本版本及 1.2.0 中的生成控制能力（流式、思考标签处理、上下文注入、服务端专项参数、
自定义 HTTP 头与请求体参数等）参考并改写了
[Local OpenAI LLM](https://github.com/skye-harris/hass_local_openai_llm)（Apache-2.0）的实现，
版权归其作者及贡献者所有；该 upstream 本身 fork 自 Home Assistant 官方 OpenRouter 集成。

## 1.2.0 — 整合 Local OpenAI LLM 的生成控制能力

本次更新把 [Local OpenAI LLM](https://github.com/skye-harris/hass_local_openai_llm) 中
对话/AI Task 的生成控制能力移植进本集成，STT/TTS 与既有多服务商架构保持不变。

新增文件 `api.py`：共享请求层（Jinja2 渲染、请求头/体构造、SSE 流式、输出清理、事件、工具定义）。

### 新增能力

- **流式响应**（可选，默认关闭）：SSE 增量解析，支持 `content` / `reasoning_content` /
  `tool_calls` 增量；流式失败自动回退非流式，已产出的内容不会重复。
- **可配置生成参数**：对话与 AI Task 的温度、最大输出 token，历史消息裁剪条数（默认 24）。
- **思考内容处理**：自动剥离 `think` / `thinking` 标签（流式用增量状态机，支持标签跨分块），
  可选回传历史 `reasoning_content`（`include_prior_thinking`）。
- **emoji 剥离**（可选）：流式与非流式均生效；内容被完全剥离时回退为「好的。」，避免静音。
- **上下文注入**：Jinja2 模板，可选 System / Tool Result / Assistant / User 角色，
  模板变量含 `device` / `area` / `floor` / `conversation_id`；默认注入当前时间。
- **自定义 HTTP 头**：JSON 或 `k=v` 行格式，支持 Jinja2，可引用 `conversation_id`。
- **附加请求体参数 / Chat Template 参数**：JSON 或 `k=v`，支持 Jinja2，
  合并进请求体；`model` / `messages` / `stream` 受保护。
- **服务端类型专项参数**：
  - DeepSeek Cloud：`reasoning_effort`（high/max）+ thinking 开关；
  - llama.cpp：`id_slot`、`top_p` / `min_p` / `top_k` / `repeat_penalty` / `presence_penalty`、
    `enable_thinking`；
  - vLLM：`thinking_token_budget`；
  - LocalAI：Chat Template 参数经 `metadata` 传递；
  - 通用：`chat_template_kwargs`。
- **并行工具调用**（可选）与可配置的工具循环轮数（默认 4）。
- **AI Task 增强**：
  - 支持图片附件（`SUPPORT_ATTACHMENTS`，转 base64 `image_url`）；
  - Generate Data 支持工具调用循环；
  - 结构化输出使用 `probatio.to_openapi(..., custom_serializer=selector_serializer)`；
  - evaluate 概率归一化，`noul` 截断到 0–1。
- **用量/耗时事件**：每次补全请求发出 `ai_voice_generation_complete`，
  含 `usage`、`timings`、`duration_ms`、`streaming`。
- **配置入口**：选项流程 → 「配置高级参数」→ 高级参数页。

### 兼容性说明

- 配置仍存于 `entry.data`，与 1.1.0 条目完全兼容；未设置的项按默认值生效。
- MiMo 仍默认关闭深度思考（`thinking: {"type": "disabled"}`），可在服务端类型为
  generic 时用 `chat_template_args` 覆盖。
- 实体 unique_id、设备信息与 1.1.0 一致，无迁移步骤。

### 验证记录（HA 2026.11.1，192.168.2.7:8133）

| 用例 | 结果 |
| --- | --- |
| 上下文注入（模板标记回读） | 通过 |
| 流式响应 + 用量事件（`streaming: true`） | 通过 |
| emoji 剥离 + 空内容兜底 | 通过 |
| 工具调用（开/关 `light.re_shui`） | 通过 |
| 并行工具调用（开灯 + 报时一次完成） | 通过 |
| 结构化 AI Task 生成 | 通过 |
| `ai_task.evaluate`（noul + choice 概率） | 通过 |
| AI Task 图片生成（本地号池 gpt-image-2） | 通过（52 s） |
| STT/TTS 实体加载 | 通过（未改动） |
| 华为云 MaaS 对话 | 环境问题：直连亦返回 `403 ModelArts.81004`，与集成无关 |
| 本地号池对话 | 环境问题：号池返回 `502 bootstrap failed: 429`，与集成无关 |

### 未端到端验证

- AI Task 图片附件：已由用户实测通过（OpenAI 兼容路径）。
- AI Task Generate Data 工具循环：依赖服务内部传入 `llm_api`，普通服务调用不触发。
- 决策模型 `/v1/systemone` 真实概率：当前仍用 chat 模型 JSON 兜底（未接 TypeSafe API）。

### 回滚

```bash
cd /vol1/1000/docker/ha-test2
tar xzf ai_voice_backup_20261126.tar.gz -C custom_components/ai_voice --strip-components=1
# 然后重启 HA
```