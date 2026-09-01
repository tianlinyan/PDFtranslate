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

测试离线运行：`tests/_helpers.py` 提供本地 mock chat-completions HTTP 服务（按块回显 `[n] MOCK:<原文>`），`build_sample_pdf()` 用于生成小型测试 PDF（拉丁 + CJK 文本）。测试模块：`test_settings`（配置解析）、`test_pdfio`（提取与导出）、`test_translator`（分批/对齐/缓存/致命错误）、`test_ocr`（OCR 管道与缓存）、`test_worker`（worker 信号契约，直接同步调用 `run()`，无需 `QApplication`）、`test_ui`（`main_window` 中与 Qt 无关的纯函数）。所有测试通过 `PDFTRANSLATE_CACHE_DIR` / `PDFTRANSLATE_OCR_CACHE_DIR` 把缓存重定向到临时目录——**新增涉及缓存的测试务必照做**，否则会污染开发者 home 并因热缓存而假绿。未配置 linter 或格式化工具。

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
- 翻译协议：每个批次以编号块发送 `[1] 文本 … [k] 文本`；模型必须回显 `[n] 译文`（一个块可跨多行，`_MULTI_BLOCK_RE` 折叠内部换行）。编号**仅在批次内局部使用**（1..k，而非全局索引），因为已缓存的块会被跳过。`_parse_response` **要么返回恰好 k 条译文，要么抛 `ValueError`**：有 `[n]` 标记时编号必须完整无缺、不越界；完全没有 `[n]` 标记时只接受「行数 == 块数」的回复（单块批次例外：整段回复折叠为该块译文）。**绝不用原文填补缺口**——那会被判为成功并写入缓存，模型的一句拒答就能永久污染该块。**改协议前先看 `tests/_helpers.py` 的 mock**（按 `\n\n` 分块解析 user 消息）。
- 按字符预算分批（`_CHAR_BUDGET` 默认 4000，可按模型 `batch_size` 覆盖，`_make_chunks` 读取）；纯数字/符号块（`_needs_translation` 判空）不发送、原样保留。
- 批次在 `ThreadPoolExecutor` 中并发请求（`model.concurrency` 个 worker）；结果按完成序回收，输出仍按输入块序对齐。缓存字典更新加 `_cache_lock`。
- 瞬时故障（网络错误、429/5xx、解析不匹配）重试 3 次并带退避（`retry_delays`，测试注入 `(0,0)`）。重试耗尽后该批次**保留原文**（绝不丢内容）且**不写入缓存**（防污染 resume），失败原因记入 `result.errors`，由 worker 汇总成一条中文告警日志。
- **致命配置错误**（401/403/404：`_is_fatal`）不重试也不降级：设置 `abort` 事件让排队中的批次立即让路，并抛 `TranslationAborted`，worker 转成 `error` 信号。否则密钥填错会「成功」导出一份与原文逐字相同的文档。400（`BadRequestError`）视为单批次问题，仍只保留原文。
- 磁盘缓存位于 `~/.pdftranslate/cache`（`PDFTRANSLATE_CACHE_DIR` 可覆盖；该目录不可写时回退到系统临时目录下的 `pdftranslate_cache`；`_cache_dir` 会实写探针文件验证可写性，仅 `mkdir` 成功不算数）：文件名 = `trans_v{_CACHE_VERSION}_` + sha1(文档路径 | 目标语言 | 模型 id) 前 16 位，内容 = JSON 映射 md5(块文本) → 译文。改动缓存语义时**递增 `_CACHE_VERSION`**（当前为 4）令旧缓存自动失效。每批完成即落盘，且**写入是原子的**（临时文件 + `os.replace`）——关闭窗口会 `os._exit(0)` 硬退，普通覆盖写可能留下半截 JSON，读回时等同「无缓存」而全量重译。非 dict 的缓存文件按损坏处理并忽略。进度从已缓存块数+跳过块数起始。测试用 `PDFTRANSLATE_CACHE_DIR` 指向临时目录，避免污染开发者 home 并保证冷缓存。
- 缓存写入是尽力而为：失败**不中断翻译**，但会经 `log` 回调告警（每次运行只警告一次）。若不告警，只读 home / 沙箱等场景下会表现为「每次重跑都全量重译」而无任何线索。
- 取消抛出 `TranslationCancelled`（重试退避的 sleep 分片检查），worker 捕获后仅输出“已取消”日志（不算错误）；无论成功、失败还是取消，worker 都在 `finally` 里发 `stopped` 信号。

### translate_app/pdfio.py
- `extract_document_text` 使用 `page.get_text("blocks")`，仅保留文本块（type 0）；`_order_blocks` 按 x 重叠聚类分栏，多栏页按列（左列自上而下，再右列）阅读，单栏保持 y 取整后按 x。另用 `page.get_text("dict")` 一次提取 span，为每个 `Block` 填充 `size`（span 字号中位数）、`bold`、`align`（left/center/right）与 `single_line`。`DocumentText` 同时携带 `pages`（带布局元数据的 `Block`）和扁平的 `blocks`/`block_pages`。
- CJK 文本用 PyMuPDF 内置的 `fitz.Font("cjk")` 渲染（回退 helv）。所有块（含粗体）统一用同一款字体——曾尝试用 `insert_text("china-s", render_mode=2)` 描边模拟粗体，但会混入第二种字体（Heiti），造成版面字体不一致，已移除；`Block.bold` 字段保留但暂不改变渲染。
- `_draw_translated_block`（原位与双语共用的绘制助手）：字形框（ascender+行+descender）锚定块 bbox 顶部，字号从原块字号起（上限 `_MAX_FONT` 24pt）以 0.9 因子缩减直到装进框内（下限 3pt 保证不溢出）；单行块在框内垂直居中，多行块贴顶；支持左/中/右对齐。
- `save_translated_pdf`（仅译文/原位）：原样插入每页原文，涂盖原文文本矩形但保留图片与线条（`PDF_REDACT_IMAGE_NONE` / `PDF_REDACT_LINE_ART_NONE`），再按块调用 `_draw_translated_block`。对 `Block.ocr` 为真的块（其文字是位图、无文本层可红action）**不红action**，而是先在该块 bbox 画一个不透明白色矩形盖住扫描字，再画译文，避免叠字。
- `save_interleaved_pdf`（双语）：原文页之后新建空白译文页，译文块按原文块的 bbox 位置绘制（镜像版式）；无文本页写提示文案。worker 传 `doc.pages` 供其使用。
- OCR（扫描页）：`extract_document_text(...)` 接受 `ocr` / `ocr_fn` / `cancel` / `log`。当页无文本层且 `_needs_ocr(page)`（有图片或图形）为真时，用 RapidOCR（`rapidocr_onnxruntime`，内置中英文 PP-OCRv3 模型、离线）识别。生产路径：`_page_to_array` 按 `_OCR_DPI` 渲染并做 **RGB→BGR**（RapidOCR/OpenCV 约定），`engine(img)` 输出经防御式解析（兼容 `(list, timings)` 或裸 list），像素框 `/zoom` 转成 PDF 点，再经 `_synthesize_ocr_blocks`（`_clean_text` 清控制字符 → `_order_blocks` 按列阅读序 → 字号 `(y1-y0)/1.2` 截断 `[5, _MAX_FONT]`）还原成 `Block`（`ocr=True`、`single_line=True`）。**OCR 结果按文档缓存**（`~/.pdftranslate/ocr_cache/ocr_<sha1>.json`，key 含文件 mtime+size；`PDFTRANSLATE_OCR_CACHE_DIR` 可覆盖目录），且**每识别完一页就原子落盘**（临时文件 + `os.replace`）——OCR 是全流程最慢的一环，取消或崩溃绝不能让已识别的页面白做；缓存对注入的 `ocr_fn` 同样生效，因此缓存逻辑本身可被测试。`DocumentText.ocr_count` 记录 OCR 页数供 worker 日志。`_get_ocr_engine` 惰性单例（`_OCR_LOCK` + `_OCR_FAILED` 缓存，失败返回 `None` 并**降级跳过**而非抛异常，由 `_warn_ocr_unavailable` 全局告警一次——该函数是唯一改写 `_OCR_WARNED` 的地方，模块级可变状态必须在带 `global` 声明的函数内改写）；`ocr_fn(page_index, page)->[(box,text)]`（box 为 PDF 点）是测试注入缝；`cancel()` 每页触发抛 `TranslationCancelled`。`_ocr_page_blocks` 里单页识别失败**降级为空并经 `log` 报出原因**（静默 `except` 与「这页本来就没字」无法区分），但 `TranslationCancelled` 属控制信号，必须原样上抛。**OCR 语言自动跟随原文**（RapidOCR 默认中英文自动识别模型，无独立选择项）。
- `_wrap` 必须按字符断开无空格的 CJK「词」（`_break_word` 中二分查找）且不丢字——由 `WrapTest` 覆盖。

### translate_app/main_window.py
- `LANGUAGES` 将界面显示名映射为发送给模型的语言名。语言下拉框**可编辑**，因此必须用 `resolve_language(currentText())` 取值，**不可用 `currentData()`**：输入的文本不匹配任何条目时 Qt 会保留原 `currentIndex`，`currentData()` 仍指向上一次选中项，用户输入会被静默丢弃（请求与输出文件名一起用错语言）。表中没有的语言原样透传给模型。
- 线程生命周期：每次运行创建 `QThread` + worker；`worker.stopped`（在 `run()` 的 `finally` 中发出）连到 `thread.quit`，`thread.finished` 连到 `_cleanup`。**不要改回从 `finished`/`error` 退出线程**——取消时二者都不发，线程事件循环会永远不退出，`_cleanup` 不执行，「开始翻译」按钮永久禁用。关闭窗口即「六亲不认」地强行退出——不弹任何确认框，直接 `thread.terminate()` + `thread.wait()` 强制结束 worker 线程，再用 `os._exit(0)` 硬退整个进程，以绕过 `concurrent.futures` 在解释器退出时对非 daemon HTTP 线程的 join 阻塞（因此绝不等待当前请求完成；缓存的原子写正是为此保底）。
- OCR 开关：表单中以「识别扫描页」行提供 `QCheckBox`（默认勾选；OCR 语言自动跟随原文，无独立下拉），随 `_save_prefs` 持久化到 `prefs.json`（`ocr`），`_start` 时传给 `TranslateWorker(ocr=...)`。
- PDF 可拖放到窗口设置源文件；命令行传入的 PDF 路径效果相同。
