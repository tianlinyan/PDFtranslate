# PDFtranslate

> 当前版本：**v0.2.3**（版本号定义于 `translate_app/__init__.py` 的 `__version__`）

一个 Windows 桌面 **PDF AI 翻译**工具。打开一个 PDF，选择 AI 模型与目标语言，
即可把文档翻译成指定语言并保存为双语 PDF、原位翻译 PDF、Markdown 或纯文本。

* **界面**：PyQt6（简洁的翻译主界面）
* **PDF 处理/导出**：PyMuPDF（原位翻译时保留原版式与图片）
* **AI 翻译**：OpenAI 兼容的 `chat/completions` 接口，模型由 `models.json` 配置

## 运行

环境要求：Python 3.10+。

```powershell
pip install -r requirements.txt
python main.py
```

或直接双击（Windows）：

```bat
run.bat
```

也可以在启动时直接指定要翻译的 PDF：

```powershell
python main.py "C:\path\to\doc.pdf"
```

## 使用

程序启动后就是翻译界面：

1. **选择源文件**：点击「打开 PDF…」选择一个 PDF，或直接把 PDF 拖放到窗口中。
2. 选择 **AI 模型**（来自 `models.json`）、**目标语言**、**输出格式**：
   - **双语 PDF**：每一页原文之后紧跟一页译文，译文页按原版式对齐排版
     （每个译文块位于对应原文块的位置，自动换行并缩字号适配）。
   - **仅译文 PDF**：原位翻译，保留原页面图片/图形在原本位置，把译文写入原文位置，
     字号/对齐（左/中/右）/粗体尽量跟随原文。
   - **Markdown 文档** / **纯文本**：仅文字。
3. **保存路径**：可点击「浏览…」指定，留空则自动生成到源文件目录。
4. 点击 **开始翻译**，等待进度完成，然后点 **打开输出** 查看结果。

底部还有两个辅助按钮：**清除缓存**（清空已缓存的译文）与 **关于**（查看版本、开发者与项目主页）。

> **支持的目标语种（按设计限定）**：简体中文、英语、西班牙语、法语、德语、意大利语。
> 其余语种不在此范围内。

## 模型配置 (`models.json`)

每个条目描述一个 OpenAI 兼容的 `/chat/completions` 端点：

```json
{
  "models": [
    {
      "id": "qwen3.8",
      "name": "qwen3.8",
      "type": "llama-server",
      "endpoint": "http://192.168.0.19:8888/v1/chat/completions",
      "model": "qwen3.8",
      "reasoning_effort": "low",
      "concurrency": 2,
      "batch_size": 12000
    },
    {
      "id": "ds4-pro",
      "name": "DS4",
      "type": "deepseek",
      "endpoint": "https://api.deepseek.com/v1/chat/completions",
      "model": "deepseek-v4-flash",
      "api_key": "${DEEPSEEK_API_KEY}"
    }
  ]
}
```

* `endpoint`：Chat Completions 地址（`base_url` 为该地址去掉 `/chat/completions`）。
* `api_key`：JSON 中的 `${ENV_VAR}` 会从环境变量读取，避免把密钥写进文件。
* 本地 `llama-server` 端点通常无需 key，可省略 `api_key`。
* `reasoning_effort`（可选）：发送给该模型的请求参数。对 llama.cpp/某些推理模型，
  若不设置会导致服务器 500（如 “Unexpected reasoning effort none”）。可设为
  `xhigh`（默认）/`medium`/`low`，翻译建议 `low`。
* `concurrency`（可选，默认 1，即串行）：同时发送的翻译批次请求数。云 API 想
  提速可设为 2–4；本地 `llama-server` 视并行槽位可试 1–2（GPU 算力是瓶颈时
  并发收益有限，不会线性加速）。
* `batch_size`（可选，默认 4000）：每批请求的原文字符预算。**加大批次减少
  请求次数**，能显著摊薄每个请求的固定开销（提示词处理 + 推理模型的思考），
  对本地慢模型提速最明显——只要服务端上下文放得下（如 12000）。
* `temperature`（可选，默认 0.2）：采样温度。
* `max_tokens`（可选）：单次请求的最大输出 token 数（缺省使用服务端默认）。
* 其余未识别键透传给 OpenAI client 构造参数（如 `timeout`，默认 300 秒）。

## 说明

* **翻译缓存**：翻译结果按「文档 + 目标语言 + 模型」缓存到
  `~/.pdftranslate/cache`，再次翻译相同内容时会复用，节省调用。缓存带版本号，
  提示词/协议升级后旧缓存自动失效；每完成一批即落盘，取消/中断不丢已完成部分。
  **失败批次的原文不会被写入缓存**（再次运行会重试而不是复用一个错误的“译文”）。
  点击主界面 **「清除缓存」** 按钮可清空所有译文缓存。
* **并发翻译**：默认串行（`concurrency` 为 1）；如需提速可在 `models.json` 中
  按模型提高并发数和/或加大 `batch_size`（减少请求次数），进度条仍按已完成
  块数实时更新。
* **跳过无文字块**：页码、分隔线等纯数字/符号块不发送给模型，原样保留。
* **重试**：瞬时故障（网络错误、429/5xx、回复格式不完整）最多重试 3 次并带退避；
  认证/参数类错误（4xx）直接失败不重试。重试耗尽后该批次保留原文，绝不丢内容。
* **进度显示**：翻译过程中进度条实时显示完成百分比；提取文本阶段为不确定进度，
  翻译阶段按已翻译块数实时更新。
* **保留原版式（图片在正确位置）**：**仅译文 PDF** 采用“原位翻译”——每一页保留
  原有的图片/图形（照片、CAD 图、Logo、二维码、矢量图等）在原来位置，只把原文
  文字移除并在原位置写入译文；字号跟随原文（标题可到 24pt），单行文本垂直居中，
  支持左/中/右对齐，多栏页面按栏阅读顺序提取。全部译文统一用同一款 CJK 字体
  渲染（内置字体无粗体字重，粗体块不单独换字体，保证版面字体一致）。
* **可保存**：`models.json` 中密钥通过环境变量注入，不在文件中明文存储。

## 测试

用 `unittest` 运行回归测试（无需联网；用本地 mock 服务模拟 chat/completions）：

```powershell
python -m unittest discover -s tests -v
```

覆盖：模型配置解析/校验、环境变量密钥解析、翻译分块与编号对齐、多行回复解析、
纯符号块跳过、并发批次顺序、进度从已缓存块数起始、失败批次不写缓存、请求失败时
的原文兜底、文本提取与各类导出（Markdown / 双语 PDF 按原版式对齐 / 原位翻译 PDF
移除原文、缩字号适配、垂直居中、右对齐、分栏阅读顺序）。
