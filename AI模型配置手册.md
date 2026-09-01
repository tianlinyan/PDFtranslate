# AI模型配置手册

> 本手册面向普通用户。教你如何为 **PDF Translate** 添加、修改一个可用的 AI 翻译模型，
> 以及遇到问题时如何排查。看懂这一篇，就能自己接上本地模型或任意云模型。

PDF Translate 本身不含任何 AI 模型，它只是“打电话”给一个**开源的模型接口**，
这个接口是谁、地址在哪、怎么连，全部集中在项目根目录的一个配置文件里：

```
<项目目录>\models.json
```

只要按要求改好这个文件，再点几次界面，就能用上你自己的 AI 模型来翻译。

---

## 1. 先看懂“一个模型”是什么

在 PDF Translate 里，**一个模型 = 一个能对话的 AI 接口**。它通常由三部分组成：

| 部分 | 举例 | 说明 |
| --- | --- | --- |
| 接口地址（端） | `http://127.0.0.1:1234/v1/chat/completions` | 模型“住”在哪；本地模型或云端都行 |
| 模型名字 | `qwen3.8`、`deepseek-v4-flash` | 告诉接口你想用哪个模型 |
| 密钥（可选） | `sk-xxxx` | 云端服务通常需要；本地模型一般不用 |

`models.json` 里可以同时写很多个模型，界面上的 **AI 模型下拉框** 会列出它们，
你一次选一个来翻译。

---

## 2. 配置文件长什么样

用记事本或任意文本编辑器打开 `models.json`，它看起来像这样（即开即用的示例）：

```json
{
  "models": [
    {
      "id": "my-local",
      "name": "我的本地模型",
      "type": "llama-server",
      "endpoint": "http://127.0.0.1:1234/v1/chat/completions",
      "model": "qwen3.8",
      "reasoning_effort": "low",
      "concurrency": 1,
      "batch_size": 12000
    },
    {
      "id": "ds4-pro",
      "name": "DS4（云端）",
      "type": "deepseek",
      "endpoint": "https://api.deepseek.com/v1/chat/completions",
      "model": "deepseek-v4-flash",
      "api_key": "${DEEPSEEK_API_KEY}"
    }
  ]
}
```

要点：
- `models` 是**一个 JSON 数组**，数组里的每对 `{ ... }` 就是一个模型。
- **逗号很关键**：模型之间用 `,` 分隔；最后一个模型后面**不要**加逗号。
- 所有双引号、花括号都要成对；写错任何一处，界面上的模型下拉框就会变成空的。
- 改完保存，**重启程序**（或重新打开翻译界面）后生效。

---

## 3. 字段详解（照着填就不会错）

| 字段 | 必填 | 作用 | 建议 |
| --- | --- | --- | --- |
| `id` | ✔ | 模型的唯一编号，程序内部用它区分模型 | 取个小写英文，如 `deepseek`、`llama` |
| `name` | 建议 | 界面下拉框里**显示的名字** | 写中文最直观，如 `DeepSeek`、`本地Qwen` |
| `type` | 建议 | 只是一个分类标签（如 `llama-server`、`deepseek`、`openai`） | 写清楚方便自己辨认，不影响功能 |
| `endpoint` | ✔ | 模型的完整 Chat Completions 地址 | 必须以 `/chat/completions` 结尾 |
| `model` | ✔ | 请求时发送的模型名 | 要和服务端加载的模型名一致 |
| `api_key` | 云端必填 | 接口密钥 | 建议用 `${环境变量}` 方式，见第 5 节 |
| `temperature` | 否 | 采样温度，越低越稳定严谨 | 默认 0.2，翻译一般不用改 |
| `max_tokens` | 否 | 单次请求最大输出 token 数 | 不填则用服务端默认 |
| `reasoning_effort` | 建议 | 对本地 `llama.cpp` 类服务**必须**设，否则会 500 | 建议填 `low` |
| `concurrency` | 否 | 同时发送的翻译批次数量 | 默认 1（串行）；云端想快可填 2–4 |
| `batch_size` | 否 | 每批请求的原文字符预算 | 默认 4000；本地慢模型可加大到 12000 |
| `tools_choice` | 否 | 某些模型需要的工具调用选项 | 一般不用填 |

> 除以上字段外，任何**未识别**的字段都会原样传给 OpenAI 客户端构造参数，
> 例如 `"timeout": 300`（请求超时秒数）。

---

## 4. 三套可直接复制的示例

### 示例 A：本地模型（LM Studio / llama.cpp / Ollama 兼容）

本地模型通常**不需要密钥**，删掉 `api_key` 即可；但要在本地先启动一个
OpenAI 兼容的服务（LM Studio 默认端口 `1234`，llama.cpp 默认 `8080`）。

```json
{
  "id": "llama-local",
  "name": "本地大模型",
  "type": "llama-server",
  "endpoint": "http://127.0.0.1:1234/v1/chat/completions",
  "model": "qwen3.8",
  "reasoning_effort": "low",
  "concurrency": 1,
  "batch_size": 12000
}
```

> 如果你的本地服务不在本机，把 `127.0.0.1` 换成该机器的 IP，如
> `http://192.168.0.19:1234/...`。

### 示例 B：云端 DeepSeek

云端需要密钥；密钥推荐用环境变量方式（见第 5 节），不要把真实密钥写死在文件里。

```json
{
  "id": "ds4-pro",
  "name": "DS4",
  "type": "deepseek",
  "endpoint": "https://api.deepseek.com/v1/chat/completions",
  "model": "deepseek-v4-flash",
  "api_key": "${DEEPSEEK_API_KEY}"
}
```

### 示例 C：任意 OpenAI 兼容服务

几乎所有云厂商都提供 OpenAI 兼容接口，改一下 `endpoint`、`model`、`api_key` 即可。

```json
{
  "id": "openai",
  "name": "OpenAI",
  "type": "openai",
  "endpoint": "https://api.openai.com/v1/chat/completions",
  "model": "gpt-4o-mini",
  "api_key": "${OPENAI_API_KEY}"
}
```

---

## 5. 密钥不写进文件（用环境变量）

`api_key` 写成 `${环境变量名}`，程序会从**系统环境变量**读取真实密钥，
这样密钥不会明文存放在 `models.json` 里。

Windows 设置环境变量（PowerShell）：

```powershell
# 当前会话生效（关掉窗口失效）
$env:DEEPSEEK_API_KEY = "sk-你的真实密钥"

# 永久生效（推荐；设置后重启程序）
setx DEEPSEEK_API_KEY "sk-你的真实密钥"
```

> 如果环境变量没设置，程序会**拒绝**把 `${...}` 当作密钥使用，翻译会失败，
> 请确认变量名与 `{}` 里的名字完全一致。

---

## 6. 怎么把某个模型设为默认

界面弹出时默认选中**数组里第一个**模型。想让哪个成为默认，就把它移到 `models` 数组的最前面。

---

## 7. 常见问题排查

| 现象 | 可能原因 | 解决办法 |
| --- | --- | --- |
| 模型下拉框是空的 | `models.json` JSON 语法错误 | 检查逗号、引号、花括号是否配对；可用 JSON 校验工具检查 |
| 一直转圈 / “连接失败” | `endpoint` 不对或服务没启动 | 浏览器打开该地址；本地模型先启动服务并确认端口正确 |
| 报 401 / 403 | 密钥错误或未设置 | 检查 `api_key` 及环境变量 |
| 报 500，含 “reasoning effort none” | 本地 `llama.cpp` 类服务没设 `reasoning_effort` | 添加 `"reasoning_effort": "low"` |
| 翻译结果为空 / 一直重试失败 | `model` 名不对，或服务不支持中文、上下文放不下 | 确认模型名、调小 `batch_size`、换支持中文的模型 |
| 速度太慢 | 串行 + 每批太小 | 适当增大 `concurrency`（云端 2–4）和 `batch_size`（如 12000） |

---

## 8. 小贴士

- **加批次能显著提速**：`batch_size` 越大，每次请求翻译的原文越多、请求次数越少；
  对本地慢模型收益最明显，但别超过服务端的上下文长度。
- **云端提速（可以并行）**：把 `concurrency` 设为 2–4，同时发送多个批次。
- **改字段后记得重启**程序让配置生效。
- 界面底部 **「清除缓存」** 只清空已翻译的缓存，不影响 `models.json`。

把 `models.json` 配置好，你的 PDF Translate 就能用上心仪的 AI 模型了。
