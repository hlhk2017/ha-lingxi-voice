# 灵犀语音（Lingxi Voice）· Home Assistant 集成

> **一个集成，搞定 Home Assistant 里全部 AI 实体**：对话助手、语音识别（STT）、语音合成（TTS）、AI 任务（含图像生成）一次配置完成。

灵犀语音是一个面向中国用户的 Home Assistant 自定义集成。它把「大模型对话」「语音识别」「语音合成」「AI 任务」四类平台打包在一起，并允许**每个模型来自不同的服务商**——对话用一家、语音识别用另一家、语音合成再用第三家也没问题。

<sub>**1.2** 起加入生成控制（流式、采样参数、上下文注入等），**1.3** 起支持 Anthropic 与 Ollama 原生协议。版本差异见 [CHANGELOG.md](./CHANGELOG.md)。</sub>

---

## ✨ 核心特点

- **一个集成 = 全部 AI 实体**
  - `conversation`：接入 Assist 的对话代理（支持工具调用、实体控制）
  - `stt`：语音识别，供语音助手 / Assist pipeline 使用
  - `tts`：语音合成，供语音助手 / TTS 服务使用
  - `ai_task`：AI 任务（文本生成、图像生成、评估），供「AI 任务」类自动化/服务调用

- **多提供商混搭（每个模型独立来源）**
  - 对话模型、AI 任务模型、图片生成模型、语音识别模型、语音合成模型可以分别选择不同服务商
  - 配置时自动读取各服务商的模型列表分类展示；支持「使用其它提供商配置」为某个模型单独加一家服务商

- **三种 LLM 协议，开箱即用**
  - **OpenAI 兼容**：`/chat/completions` —— 硅基流动、DeepSeek、DashScope、Moonshot、智谱、Groq、OpenAI、华为云 MaaS、火山方舟、小米 MiMo 等
  - **Anthropic 兼容**：`/v1/messages` —— Claude 及兼容端点（含 `x-api-key` 认证、`thinking` 块、工具调用、多模态）
  - **Ollama 原生**：`/api/chat` —— 本地 Ollama 或 Ollama Cloud（NDJSON 流式、`options` 采样、`images` 多模态）
  - 配置流程里的「协议」字段可显式指定，或选「自动识别」按服务商/地址推断

- **常见服务商开箱即用**（内置中文预设）
  - 硅基流动 SiliconFlow、DeepSeek、阿里云百炼 DashScope、月之暗面 Moonshot、智谱 BigModel、Groq、OpenAI
  - **Anthropic Claude**、**Ollama（本地 / Cloud）**
  - 华为云 MaaS（按量付费 / Token Plan）
  - 火山引擎 方舟（Ark）、**豆包语音（SeedASR）**、**小米 MiMo**
  - 自定义服务（填 base_url 即可）

- **生成控制（1.2+）**
  - 流式响应（OpenAI SSE / Anthropic SSE / Ollama NDJSON），失败自动回退非流式
  - 温度、最大输出 token、历史消息条数可配（对话与 AI Task 分开）
  - 自动剥离 `think` 标签（流式跨分块也能处理）、可选剥离 emoji
  - 上下文注入：Jinja2 模板 + System / Tool Result / Assistant / User 角色，可使用 `device` / `area` / `floor` 变量
  - 自定义 HTTP 头、附加请求体参数、Chat Template 参数（均支持 Jinja2）
  - 服务端专项：DeepSeek `reasoning_effort`、llama.cpp slot 与采样参数、vLLM 思考 token 预算、LocalAI `metadata`
  - 并行工具调用、工具调用轮数上限
  - 每次补全发出 `ai_voice_generation_complete` 事件（usage / timings / 是否流式）

- **AI Task 增强**
  - 支持图片附件（多模态输入）
  - 结构化数据生成支持工具调用
  - `ai_task.evaluate` 概率归一化
  - 图片生成走 OpenAI Images API，自动适配华为云 MaaS

- **配置流程友好**
  - 中文界面（含字段汉化）
  - 分步向导：服务商 → 五类模型；缺什么补什么，不强迫一次配全
  - 模型下拉直接显示「提供商 · 模型」，一目了然

- **改配置不用重启**
  - 模型/提示词在设置页改完即热更新，实体自动重建
  - 支持同一集成内随时切换模型来源


---

## 📦 安装

> **版本差异**：从 **Home Assistant 2026.11** 开始，社区商店 **HACS 已内置为 Marketplace（市场）**，
> 不需要再单独安装 HACS；**2026.10 及更早版本**仍然使用 HACS。
> 本集成同时兼容两者（仓库里带 `hacs.json`，Marketplace 与 HACS 都能识别）。

### 方式一：Home Assistant 2026.11 及更新版本（内置 Marketplace）

1. 打开 **设置 → 市场（Marketplace）**
2. 点击**从链接添加**，填入：
   `https://github.com/hlhk2017/ha-lingxi-voice`
   需登录 GitHub，用你自己的账号授权即可
3. 在 Marketplace 中搜索 **灵犀语音** 并安装
4. 安装后自动出配置界面（没有就重启后添加集成搜索）

### 方式二：Home Assistant 2026.10 及更早版本（HACS）

1. 确保已安装 HACS（旧版本需要单独安装）
2. **HACS → 集成 → 右上角 ⋮ → 自定义存储库**
3. 添加 `https://github.com/hlhk2017/ha-lingxi-voice`，类别选「集成」
4. 在 HACS 中搜索 **灵犀语音** → 下载 → 重启 Home Assistant

### 方式三：手动安装（任意版本）

把 `custom_components/ai_voice` 整个目录复制到 Home Assistant 配置目录：

```
/config/custom_components/ai_voice/
```

重启 Home Assistant。

---

## 🚀 配置

1. 设置 → 设备与服务 → 添加集成 → 搜索 **灵犀语音**
2. **步骤 1**：选择服务商 + 填 API Key +（可选）自定义名称与**协议**
3. **步骤 2**：依次选择
   - 对话模型
   - AI 任务模型
   - 图片生成模型
   - 语音转文本模型
   - 文本转语音模型
   - 对话提示词
4. 某类模型想要别的服务商？在下拉里选「（使用其它提供商配置）」→ 添加服务商 → 只选该类模型
5. 完成。集成会自动创建对应实体（对话 / STT / TTS / AI 任务）
6. （可选）要调整生成参数：设置 → 设备与服务 → 灵犀语音 → **配置** → 勾选「配置高级参数」

### 高级参数（1.2+）

在集成的**配置**页面勾选「配置高级参数」后进入，按需调整：

| 参数 | 默认 | 说明 |
|---|---|---|
| 服务端类型 | 通用 | 通用 / DeepSeek Cloud / llama.cpp / vLLM / LocalAI / Ollama，决定显示哪些专项参数 |
| 温度 / 最大输出 token | 0.4 / 240 | 对话采样 |
| AI 任务温度 / 最大 token | 跟随对话 / 800 | 结构化输出可加大 |
| 保留历史消息条数 | 24 | 防止上下文溢出 |
| 剥离思考标签 / emoji | 开 / 关 | `think` 标签在流式下同样处理 |
| 流式响应 | 关 | 开启后对话增量输出；对语音 pipeline 无影响 |
| 并行工具调用 / 工具轮数 | 关 / 4 | 需模型与服务端支持 |
| 上下文注入模板 / 角色 | 当前时间 / System | Jinja2，模板可用 `device` / `area` / `floor` / `conversation_id` |
| 思考模式 | 不干预 | 始终思考 / 仅首轮思考 |
| 推理强度（DeepSeek） | High | High / Max |
| 思考 token 预算（vLLM） | 空 | 留空用服务端默认 |
| 回传历史思考 | 开 | 部分模型需关闭 |
| Slot ID / 采样参数 | 空 / 默认 | llama.cpp 专用 |
| Chat Template 参数 / 附加请求体参数 / 自定义 HTTP 头 | 空 | JSON 或 `k=v` 行格式，支持 Jinja2 |

### 语音助手（Assist）接入

- 设置 → 语音助手 → 选择本集成的**语音识别**与**语音合成**实体
- 对话代理选择本集成的**对话**实体
- 建议开启「优先使用本地意图」，简单指令（开灯/关灯/问时间）毫秒级响应，复杂对话才走大模型

---

## 🔌 支持的提供商一览

| 类型 | 提供商 | 说明 |
|---|---|---|
| 对话 / AI 任务 / 图片 | 硅基流动、DeepSeek、DashScope、Moonshot、智谱、Groq、OpenAI | OpenAI 兼容 |
| 对话 / AI 任务 | Anthropic Claude | Anthropic 兼容 `/v1/messages` |
| 对话 / AI 任务 | Ollama（本地 / Cloud） | 原生 `/api/chat` |
| 对话 / 图片（部分） | 华为云 MaaS | 对话 `/openai/v1`，生图 `/v1`（自动适配） |
| 对话 / 图片（需开通） | 火山引擎 方舟 Ark | 注意：需先在方舟控制台「开通模型」 |
| 语音识别 | 硅基流动（Qwen3-ASR / SenseVoice 等）、豆包 SeedASR、小米 MiMo | 三种协议 |
| 语音合成 | 硅基流动（CosyVoice / MOSS-TTSD 等）、小米 MiMo TTS | OpenAI 兼容 / MiMo |
| 图片生成 | 硅基流动、华为云 MaaS、OpenAI 等 | OpenAI 兼容 |

> 各服务商对模型/权限有差异，以实际账号权限为准。
> Anthropic / Ollama 只提供对话与 AI Task 实体（STT / TTS 不会加载）。

---

## 🖼️ 截图 
<img width="697" height="865" alt="image" src="https://github.com/user-attachments/assets/28d8cd6d-7fdc-4def-9b63-f9a512ed1069" />
<img width="584" height="819" alt="image" src="https://github.com/user-attachments/assets/8cc00ff6-fb19-4546-95c8-90e7856326cc" />
<img width="580" height="422" alt="image" src="https://github.com/user-attachments/assets/890be70e-1887-4867-8a37-f1047e27e28c" />
<img width="773" height="450" alt="image" src="https://github.com/user-attachments/assets/d95895e9-1460-43b6-9c18-e44529e80429" />

---

## ⚠️ 注意事项

- 本集成不包含任何 API Key，请使用自己的服务商账号
- 部分服务商（如火山方舟、豆包 SeedASR、小米 MiMo）需要单独开通对应服务
- 语音识别对网络延迟敏感，建议使用国内服务商
- 修改模型配置会自动重载，无需重启 HA
- **Ollama 在 Docker 中部署 HA 时**：预设的 `localhost:11434` 指向容器自身，请改成宿主机 IP（如 `http://192.168.2.7:11434`）
- 若要用 Ollama 的 OpenAI 兼容端点（`/v1`），协议请显式选「OpenAI 兼容」，否则自动识别会走原生 `/api/chat`

---

## 🙏 致谢与参考

本项目的部分实现**参考并改写了**以下开源项目的代码与设计，特此注明：

- **[Local OpenAI LLM](https://github.com/skye-harris/hass_local_openai_llm)**（Apache-2.0）
  —— 流式响应、思考标签处理、上下文注入、服务端专项参数（DeepSeek / llama.cpp / vLLM / LocalAI）、
  自定义 HTTP 头与请求体参数等能力参考了该项目的实现。
  相关代码版权归其作者 [@skye-harris](https://github.com/skye-harris) 及贡献者所有，
  如再分发请遵守其 [Apache-2.0 许可](https://github.com/skye-harris/hass_local_openai_llm/blob/main/LICENSE)。
- Local OpenAI LLM 本身 fork 自 Home Assistant 官方
  [OpenRouter 集成](https://github.com/home-assistant/core/tree/dev/homeassistant/components/open_router)
  （作者 [@joostlek](https://github.com/joostlek)），特此一并致谢。
- 语音侧协议（豆包 SeedASR、小米 MiMo ASR/TTS）参考了各服务商公开文档与社区示例。

本仓库自身代码采用 MIT 许可；上述参考部分若涉及再分发，请同时遵守其原始许可。

---

## 📄 许可

MIT License
