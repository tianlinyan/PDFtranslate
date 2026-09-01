# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

Windows 桌面 PDF AI 翻译工具（Python 3.10+，PyQt6）。从 PDF 提取文本，通过 `models.json` 中配置的任意 OpenAI 兼容 `/chat/completions` 端点翻译，导出为双语 PDF、原位翻译 PDF、Markdown 或纯文本。界面、日志与注释大量使用简体中文——新增用户可见文本时保持一致。

## 常用命令

```powershell
pip install -r requirements.txt          # PyQt6、PyMuPDF、openai
python main.py ["C:\path\to\doc.pdf"]    # 运行程序（run.bat 为 Windows 启动器）
python -m unittest discover -s tests -v  # 运行全部测试
python -m unittest tests.test_pdfio.PdfioTest.test_bilingual_pdf -v  # 运行单个测试
```

测试离线运行：`tests/_helpers.py` 提供本地 mock chat-completions HTTP 服务（按块回显 `[n] MOCK:<原文>`），`build_sample_pdf()` 用于生成小型测试 PDF（拉丁 + CJK 文本）。未配置 linter 或格式化工具。

## 架构

单向流水线，全程以扁平块列表对齐：

```
main.py → MainWindow (PyQt6)
  → TranslateWorker（QObject，moveToThread 到 QThread）— 保持界面响应
    → pdfio.extract_document_text → 扁平 blocks + 平行的 block_pages 列表
    → translator.TranslationEngine.translate_blocks
    → pdfio.group_by_page（扁平列表 → 每页列表）
    → pdfio.save_* 导出（按 worker.py 中 OUTPUT_TYPES 的键分发）
```

### translate_app/settings.py
- `models.json`（项目根目录）声明模型；`ModelConfig.from_dict` 解析。形如 `${ENV_VAR}` 的值从环境变量替换；未解析的占位符绝不会作为 API key 发送（改用 `"not-needed"`）。
- `endpoint` 是**完整**的 chat-completions URL；`client_kwargs()` 去掉 `/chat/completions` 后缀得到 `base_url`，并默认 `timeout=300`。models.json 条目中未识别的键进入 `ModelConfig.extra`，透传给 OpenAI client kwargs（可覆盖 timeout）。
- 按模型可配：`temperature`（默认 0.2）、`max_tokens`、`concurrency`（默认 1，串行；云 API 提速可设 2–4）、`batch_size`（每批原文字符预算，默认 4000；本地慢模型加大可显著减少请求次数）。
- `reasoning_effort`（以及未来的按请求参数）通过 `extra_body` 发送，使 llama.cpp 类服务端能收到；缺少 `reasoning_effort` 时 llama.cpp 会 500（见 README）。
- 用户偏好保存在 `~/.pdftranslate/prefs.json`（模型 id、语言、输出格式、上次目录）。

### translate_app/translator.py
- 翻译协议：每个批次以编号块发送 `[1] 文本 … [k] 文本`；模型必须回显 `[n] 译文`（一个块可跨多行，`_MULTI_BLOCK_RE` 折叠内部换行）。编号**仅在批次内局部使用**（1..k，而非全局索引），因为已缓存的块会被跳过。若模型未返回 `[n]` 标记，`_parse_response` 退化为按行位置匹配。**改协议前先看 `tests/_helpers.py` 的 mock**（按 `\n\n` 分块解析 user 消息）。
- 按字符预算分批（`_CHAR_BUDGET` 默认 4000，可按模型 `batch_size` 覆盖，`_make_chunks` 读取）；纯数字/符号块（`_needs_translation` 判空）不发送、原样保留。
- 批次在 `ThreadPoolExecutor` 中并发请求（`model.concurrency` 个 worker）；结果按完成序回收，输出仍按输入块序对齐。缓存字典更新加 `_cache_lock`。
- 瞬时故障（网络错误、429/5xx、解析不匹配）重试 3 次并带退避（`retry_delays`，测试注入 `(0,0)`）；4xx 认证/参数错误直接失败。重试耗尽后该批次**保留原文**（绝不丢内容）且**不写入缓存**（防污染 resume），失败原因记入 `result.errors`。
- 磁盘缓存位于 `~/.pdftranslate/cache`（该目录不可写时回退到系统临时目录下的 `pdftranslate_cache`；`_cache_dir` 会实写探针文件验证可写性，仅 `mkdir` 成功不算数）：文件名 = `trans_v{_CACHE_VERSION}_` + sha1(文档路径 | 目标语言 | 模型 id) 前 16 位，内容 = JSON 映射 md5(块文本) → 译文。改动缓存语义时**递增 `_CACHE_VERSION`**（当前为 3）令旧缓存自动失效。每批完成即落盘。进度从已缓存块数+跳过块数起始。测试用唯一模型 id + 唯一文档路径避开热缓存。
- 缓存写入是尽力而为：失败**不中断翻译**，但会经 `log` 回调告警（每次运行只警告一次）。若不告警，只读 home / 沙箱等场景下会表现为「每次重跑都全量重译」而无任何线索。
- 取消抛出 `TranslationCancelled`（重试退避的 sleep 分片检查），worker 捕获后仅输出“已取消”日志（不算错误）。

### translate_app/pdfio.py
- `extract_document_text` 使用 `page.get_text("blocks")`，仅保留文本块（type 0）；`_order_blocks` 按 x 重叠聚类分栏，多栏页按列（左列自上而下，再右列）阅读，单栏保持 y 取整后按 x。另用 `page.get_text("dict")` 一次提取 span，为每个 `Block` 填充 `size`（span 字号中位数）、`bold`、`align`（left/center/right）与 `single_line`。`DocumentText` 同时携带 `pages`（带布局元数据的 `Block`）和扁平的 `blocks`/`block_pages`。
- CJK 文本用 PyMuPDF 内置的 `fitz.Font("cjk")` 渲染（回退 helv）。所有块（含粗体）统一用同一款字体——曾尝试用 `insert_text("china-s", render_mode=2)` 描边模拟粗体，但会混入第二种字体（Heiti），造成版面字体不一致，已移除；`Block.bold` 字段保留但暂不改变渲染。
- `_draw_translated_block`（原位与双语共用的绘制助手）：字形框（ascender+行+descender）锚定块 bbox 顶部，字号从原块字号起（上限 `_MAX_FONT` 24pt）以 0.9 因子缩减直到装进框内（下限 3pt 保证不溢出）；单行块在框内垂直居中，多行块贴顶；支持左/中/右对齐。
- `save_translated_pdf`（仅译文/原位）：原样插入每页原文，涂盖原文文本矩形但保留图片与线条（`PDF_REDACT_IMAGE_NONE` / `PDF_REDACT_LINE_ART_NONE`），再按块调用 `_draw_translated_block`。
- `save_interleaved_pdf`（双语）：原文页之后新建空白译文页，译文块按原文块的 bbox 位置绘制（镜像版式）；无文本页写提示文案。worker 传 `doc.pages` 供其使用。
- `_wrap` 必须按字符断开无空格的 CJK「词」（`_break_word` 中二分查找）且不丢字——由 `WrapTest` 覆盖。

### translate_app/main_window.py
- `LANGUAGES` 将界面显示名映射为发送给模型的语言名。
- 线程生命周期：每次运行创建 `QThread` + worker；翻译进行中关闭窗口会弹确认框，置 `_closing`，取消 worker，等线程结束后才真正关闭。信号连接顺序有讲究：`finished → _cleanup` 必须先于 `finished → _finish_close` 连接。
- PDF 可拖放到窗口设置源文件；命令行传入的 PDF 路径效果相同。
