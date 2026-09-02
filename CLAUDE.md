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
python check_translation.py 原文.pdf 译文.pdf [--lang English] [--strict] [--skip 24-27]
                                         # 导出后体检：数字一致性（exit 1）/残留中文/章节编号/页数
```

测试离线运行：`tests/_helpers.py` 提供本地 mock chat-completions HTTP 服务（按块回显 `[n] MOCK:<原文>`），`build_sample_pdf()` 用于生成小型测试 PDF（拉丁 + CJK 文本）。测试模块：`test_settings`（配置解析）、`test_pdfio`（提取与导出/数字原子性/竖排标签）、`test_translator`（分批/对齐/缓存/致命错误/术语表）、`test_ocr`（OCR 管道与缓存/数字归一化）、`test_worker`（worker 信号契约与姓名列方向门控，直接同步调用 `run()`，无需 `QApplication`）、`test_ui`（`main_window` 中与 Qt 无关的纯函数）、`test_check_translation`（校对脚本）。所有测试通过 `PDFTRANSLATE_CACHE_DIR` / `PDFTRANSLATE_OCR_CACHE_DIR` 把缓存重定向到临时目录——**新增涉及缓存的测试务必照做**，否则会污染开发者 home 并因热缓存而假绿。未配置 linter 或格式化工具。

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
- `models.json`（项目根目录）声明模型；`ModelConfig.from_dict` 解析。`api_key` 字段存**原始值**且 `repr=False`（repr/断言不会打印密钥）；形如 `${ENV_VAR}` 的占位符在 `_resolved_api_key` / `validate` 里**惰性**替换（不在 `from_dict` 时替换——加载时替换 + 使用时再替换的双重替换会造成不一致，如 env 变量被设为空串时会静默变 `"not-needed"` 且 `validate` 不告警）；未解析的占位符绝不会作为 API key 发送（改用 `"not-needed"`，且 `validate` 会告警）。
- `endpoint` 是**完整**的 chat-completions URL；`client_kwargs()` 去掉 `/chat/completions` 后缀得到 `base_url`，并默认 `timeout=300`。models.json 条目中未识别的键进入 `ModelConfig.extra`，透传给 OpenAI client kwargs（可覆盖 timeout）。
- 按模型可配：`temperature`（默认 0.2）、`max_tokens`、`concurrency`（默认 1，串行；云 API 提速可设 2–4）、`batch_size`（每批原文字符预算，默认 4000；本地慢模型加大可显著减少请求次数）。
- `reasoning_effort`（以及未来的按请求参数）通过 `extra_body` 发送，使 llama.cpp 类服务端能收到；缺少 `reasoning_effort` 时 llama.cpp 会 500（见 README）。
- 用户偏好保存在 `~/.pdftranslate/prefs.json`（模型 id、语言、输出格式、上次目录）；`save_prefs` 原子写（临时文件 + `os.replace`，程序硬退不会留下半截 JSON）并返回失败原因（`None` = 成功），UI 侧保存失败会写入日志而非静默丢失（静默丢失的偏好要等下次启动才暴露、无法诊断）。

### translate_app/translator.py
- 翻译协议：每个批次以编号块发送 `[1] 文本 … [k] 文本`；模型必须回显 `[n] 译文`（一个块可跨多行，`_MULTI_BLOCK_RE` 折叠内部换行）。编号**仅在批次内局部使用**（1..k，而非全局索引），因为已缓存的块会被跳过。`_parse_response` **要么返回恰好 k 条译文，要么抛 `ValueError`**：有 `[n]` 标记时编号必须完整无缺、不越界；完全没有 `[n]` 标记时只接受「行数 == 块数」的回复（单块批次例外：整段回复折叠为该块译文）。**空译文同样被拒**（编号块无内容、或整段空白回复折叠成 `""`）——空串会被判成功、写进缓存并让该块在每次导出里变空白，且被永久复用。**绝不用原文填补缺口**——那会被判为成功并写入缓存，模型的一句拒答就能永久污染该块。**改协议前先看 `tests/_helpers.py` 的 mock**（按 `\n\n` 分块解析 user 消息）。
- 按字符预算分批（`_CHAR_BUDGET` 默认 4000，可按模型 `batch_size` 覆盖，`_make_chunks` 读取）；纯数字/符号块（`_needs_translation` 判空）不发送、原样保留。
- 系统提示词（`_system_prompt(language, glossary)`）面向中文年报翻译强化规则：数字/单位不得改格式（万元按「ten thousand yuan」保留数值）、章节编号沿用原文风格、报表/文件编号不得拼音化；目标语言为**西文**时额外要求中文人名统一拼音罗马化、**名在前姓在后**（如 王晓东 → `Xiaodong Wang`）。`glossary.json`（**文档所在目录**的扁平 JSON 对象 `源词→目标词`）随批次注入提示词；加载失败经 `log` 告警并忽略（静默跳过会让用户误以为术语已生效）。`translate_blocks` 的 `keep_original`（姓名列）由 **worker 按目标语言方向门控**：目标含 CJK 时保留原文，否则释放给模型按规则罗马化（并记日志说明）。
- 批次在 `ThreadPoolExecutor` 中并发请求（`model.concurrency` 个 worker）；结果按完成序回收，输出仍按输入块序对齐。缓存字典更新加 `_cache_lock`；**磁盘落盘在 `_cache_lock` 之外**进行（持锁取快照 + 单调序号，独立 `_persist_lock` 串行化 I/O，`seq < _persisted_seq` 的过期快照直接丢弃）——持锁做文件 I/O 会让并发批次互相阻塞。
- 瞬时故障（网络错误、408/429/5xx、解析不匹配）重试 3 次并带退避（`retry_delays`，测试注入 `(0,0)`）。重试耗尽后该批次**保留原文**（绝不丢内容）且**不写入缓存**（防污染 resume），失败原因记入 `result.errors`，由 worker 汇总成一条中文告警日志。
- **致命配置错误**（401/403/404：`_is_fatal`）不重试也不降级：设置 `abort` 事件让排队中的批次立即让路，并抛 `TranslationAborted`，worker 转成 `error` 信号。否则密钥填错会「成功」导出一份与原文逐字相同的文档。400（`BadRequestError`）视为单批次问题，仍只保留原文。
- 磁盘缓存位于 `~/.pdftranslate/cache`（`PDFTRANSLATE_CACHE_DIR` 可覆盖；该目录不可写时回退到系统临时目录下的 `pdftranslate_cache`；`_cache_dir` 会实写探针文件验证可写性，仅 `mkdir` 成功不算数）：文件名 = `trans_v{_CACHE_VERSION}_` + sha1(文档路径 | 目标语言 | 模型 id | **术语表内容哈希**) 前 16 位，内容 = JSON 映射 md5(块文本) → 译文。改动缓存语义时**递增 `_CACHE_VERSION`**（当前为 6：5 时引入术语表入键 + 名称西文罗马化，6 时人名顺序改为名在前姓在后——规则变动不递增版本时，旧缓存里姓在前的老译名会被原样复用）令旧缓存自动失效——**术语表改变也会换缓存名**，否则加了术语表后已缓存块永远不按新术语重译。每批完成即落盘，且**写入是原子的**（临时文件 + `os.replace`）——关闭窗口会 `os._exit(0)` 硬退，普通覆盖写可能留下半截 JSON，读回时等同「无缓存」而全量重译。非 dict 的缓存文件按损坏处理并忽略。进度从已缓存块数+跳过块数起始。测试用 `PDFTRANSLATE_CACHE_DIR` 指向临时目录，避免污染开发者 home 并保证冷缓存。
- 缓存写入是尽力而为：失败**不中断翻译**，但会经 `log` 回调告警（每次运行只警告一次）。若不告警，只读 home / 沙箱等场景下会表现为「每次重跑都全量重译」而无任何线索。
- 取消抛出 `TranslationCancelled`（重试退避的 sleep 分片检查），worker 捕获后仅输出“已取消”日志（不算错误）；无论成功、失败还是取消，worker 都在 `finally` 里发 `stopped` 信号。

### translate_app/pdfio.py
- `extract_document_text` 使用 `page.get_text("blocks")`，仅保留文本块（type 0）；`_order_lines` / `_order_blocks` 共用 `_order_generic` 按 x 重叠聚类分栏，多栏页按列（左列自上而下，再右列）阅读，单栏保持 y 取整后按 x。**整宽元素（横跨两栏的标题/分隔线）在聚类前先抽出**：贪心聚类里宽元素会成为右边界=页宽的「列」，两栏所有行都与之「重叠」而被并成一栏、左右行按 y 交错（`test_full_width_heading_does_not_merge_columns` 回归）；判据「宽度 > 1.5× 中位宽度」在单栏页永不成立、自动退化为纯 y 序，整宽元素按 y 插到列前（页眉）或列后（页脚）。每页 `page.get_text("dict")` 只提取**一次**，行收集器（`_collect_lines`）与 span 收集器（`_collect_spans`）共享同一 dict（曾各自重取、成本翻倍），为每个 `Block` 填充 `size`（span 字号中位数）、`bold`、`align`（left/center/right）与 `single_line`。`DocumentText` 同时携带 `pages`（带布局元数据的 `Block`）和扁平的 `blocks`/`block_pages`。
- CJK 文本用 PyMuPDF 内置的 `fitz.Font("cjk")` 渲染（回退 helv）。所有块（含粗体）统一用同一款字体——曾尝试用 `insert_text("china-s", render_mode=2)` 描边模拟粗体，但会混入第二种字体（Heiti），造成版面字体不一致，已移除；`Block.bold` 字段保留但暂不改变渲染。
- `_draw_translated_block`（原位与双语共用的绘制助手）：字形框（ascender+行+descender）锚定块 bbox 顶部，字号从原块字号起（上限 `_MAX_FONT` 24pt）以 0.9 因子缩减直到装进框内；**3pt 是可读性下限，不是不溢出保证**——病态长译文（远超原文、框又极小）在 3pt 仍超框时宁可溢出绘制也不丢内容；缩减步用 `max(3.0, round(fs * 0.9, 2))` 钳制使下限恰好为 3.0（裸 `round` 会跌破，如 3.28→2.95）；单行块在框内垂直居中，多行块贴顶（若多行块比框还高，居中偏移钳为 0，即贴顶）；支持左/中/右对齐。**表格单元（`in_table`）的拟合规则**：先按列宽（`fit_width` 或缺省框宽）单行拟合、缩至 `_MIN_TABLE_READABLE`(6pt)——单行 ≥6pt 时保持一行（数字/短译文永不换行）；**可读单行放不下时改为在多行下限 6pt 处换行，绝不再缩到 3pt 一条**（旧规则在扫描报表上产出 44 格 <5pt、5 格触底 3.0 的实锤软肋）；**网格单元格（`fit_height` > 0，见下）的换行以行带为界**（`_fit_band` 在 [[0.01pt 网格]] 上二分取得 ≤ `fit_height` 的最大字号），换行绝不越过下一条表格线；文本层单元格（`fit_height`=0）的换行由 `_measure_block_height` → 行高扩展吸收（矢量线可整体重排）；非表格块仍走 wrap 路径。
- `save_translated_pdf`（仅译文/原位）：原样插入每页原文，涂盖原文文本矩形但保留图片与线条（`PDF_REDACT_IMAGE_NONE` / `PDF_REDACT_LINE_ART_NONE`），再按块调用 `_draw_translated_block`。对 `Block.ocr` 为真的块（其文字是位图、无文本层可红action）**不红action**，而是先在该块 bbox 画一个不透明白色矩形盖住扫描字，再画译文，避免叠字。
- `save_interleaved_pdf`（双语）：原文页之后新建空白译文页，译文块按原文块的 bbox 位置绘制（镜像版式）；无文本页写提示文案。worker 传 `doc.pages` 供其使用。
- OCR（扫描页）：`extract_document_text(...)` 接受 `ocr` / `ocr_fn` / `cancel` / `log`。当页无文本层且 `_needs_ocr(page)`（有图片或图形）为真时，用 RapidOCR（`rapidocr_onnxruntime`，内置中英文 PP-OCRv3 模型、离线）识别。生产路径：`_page_to_array` 按 `_OCR_DPI` 渲染并做 **RGB→BGR**（RapidOCR/OpenCV 约定；`[::-1]` 翻转得到的是负步长视图，必须 `np.ascontiguousarray` 转连续数组再交给引擎），`engine(img)` 输出经防御式解析（兼容 `(list, timings)` 或裸 list），像素框 `/zoom` 转成 PDF 点，再经 `_synthesize_ocr_blocks`（`_clean_text` 清控制字符 → `_order_blocks` 按列阅读序 → 字号 `(y1-y0)/1.2` 截断 `[5, _MAX_FONT]`）还原成 `Block`（`ocr=True`、`single_line=True`）。**OCR 结果按文档缓存**（`~/.pdftranslate/ocr_cache/ocr_v{_OCR_CACHE_VERSION}_<sha1>.json`，key 含文件 mtime+size；`PDFTRANSLATE_OCR_CACHE_DIR` 可覆盖目录）。**改动缓存块字段语义时递增 `_OCR_CACHE_VERSION`**——OCR 缓存文件名无版本号时，旧版本写入的块（如无 `in_table` 的预网格重建格式）会被逐字复用，单行布局/列加宽逻辑对已缓存的文档全部失效（表现为扫描版资产负债表整表换行）；递增版本令其全部重 OCR。且**每识别完一页就原子落盘**（临时文件 + `os.replace`）——OCR 是全流程最慢的一环，取消或崩溃绝不能让已识别的页面白做；缓存对注入的 `ocr_fn` 同样生效，因此缓存逻辑本身可被测试。`DocumentText.ocr_count` 记录 OCR 页数供 worker 日志。`_get_ocr_engine` 惰性单例（`_OCR_LOCK` + `_OCR_FAILED` 缓存，失败返回 `None` 并**降级跳过**而非抛异常，由 `_warn_ocr_unavailable` 全局告警一次——该函数是唯一改写 `_OCR_WARNED` 的地方，模块级可变状态必须在带 `global` 声明的函数内改写）；`ocr_fn(page_index, page)->[(box,text)]`（box 为 PDF 点）是测试注入缝；`cancel()` 每页触发抛 `TranslationCancelled`。`_ocr_page_blocks` 里单页识别失败**降级为空并经 `log` 报出原因**（静默 `except` 与「这页本来就没字」无法区分），但 `TranslationCancelled` 属控制信号，必须原样上抛。**OCR 语言自动跟随原文**（RapidOCR 默认中英文自动识别模型，无独立选择项）。
- **手写体签字不翻译**（`_drop_signature_items`）：扫描报表底部的签字区（法定代表人：/主管会计工作负责人：/会计机构负责人：行）会被 OCR 成一条「高出一大截、纯字母无数字」的框（手迹墨水范围≈40pt vs 印刷行 ≈9pt，且位于页底约 30% 区域）；若当作文本块，模型会把签字拼音化成 `Xiaobo` 并盖住真实手迹（白矩形遮盖扫描像素——签字是身份不是内容）。检测到即从 `items` 剔除：不翻译、不覆盖、经 `log` 确认；页底高框含数字的（印章外沿/图注）不剔除。
**OCR 数字归一化**（T0）：RapidOCR 会把千分位/小数点看错——`_normalize_number` 只改动**数字形状**的块（`_NUM_SHAPE_RE`），修复逗号小数点互换（`3,702.726,474.45`→`3,702,726,474.45`）、斜杠断裂（`231. / 81`→`231.81`）、逗号写小数（`11,530,351,55`→`11,530,351.55`）；日期（`1960.08`）、纯值（`92.5%`）与无法验证的字符串原样保留。**旧 OCR 缓存在加载时重归一化**（`extract_document_text` 命中缓存的块经 `replace(b, text=norm)` 修订并记档），因此历史缓存同样修复；未生成读数的逐位校验可交给 `check_translation.py` 的导出后比对。
- 绘制期**数字原子性**（T1）：`_num_atom` / `_is_number_atom` 使 `_break_word` 绝不拆断整数字形（宁可交给单行/溢出处理）；`_break_latin_word` 每片 ≥2 字符且尾部 ≥2（杜绝 `P- ar- ty` 与 `tiv-`+`e` 碎块）；`_wrap` 合并换行断开的数字（`(?<=\d)([.,])\s+(?=\d)`）。
- **竖排标签**（T1）：`_is_vertical_label`（h ≥ 2w、w ≤ 30、h ≥ 22、单行、非表格）判定中文架构图里的窄高框，`_draw_vertical_label` 经 `TextWriter` 的 `morph` 旋转 90° 绘制（几何与 `insert_text(rotate=90)` 完全相同，见 `_draw_vertical_label` 注释）——**绝不回退到横排逐字碎块**；过长标签按列宽字号居中、越线时**钳制在页内**（页外字形会被提取/渲染裁掉）。
- `_wrap` 必须按字符断开无空格的 CJK「词」且不丢字——由 `WrapTest` 覆盖。`_break_word` 对剩余字符**单遍累加**字宽、增量跟踪剩余宽度（`text_length` 已验证为纯字宽求和、与逐字求和精确相等、无 kerning，故累加精确），替代了原来「二分查找 + 反复重测整段前缀」的 O(n log n) 做法；长无空格段落提速约 10×（2000 字 `_wrap` ≈ 0.6s → 0.05s）。
- **OCR 表格不再行扩展**（T1）：`save_translated_pdf` 对 `_reconstruct_ocr_tables`（无文本层页重建的网格）**跳过** `_compute_table_layout` 的行高扩展——重建行高与译文测量差太多，曾把 `292,712,933,925.17` 下推约 355pt（回归 `test_ocr_grid_in_place_keeps_rows_below_the_wrapping_cell`）；扫描页保持 OCR 几何，靠单元格行带换行（`fit_height`，见下）+ 缩字兜底——位图表格线与签字墨迹钉死在像素上，无论怎么推挤行都会被察觉。
- **表格格内行数**（T1）：`_fit_block` 的 `in_table` 分支以**源文本**的行数（`block.text.split("\n")`，网格单元格恒为 1 行、文本层表格单元格可为多行）为目标，但**可读性优先于行数**：1 行源 → `_fit_one_line`（缩字不换行，仅当单行 ≥ `_MIN_TABLE_READABLE`）；单行会跌破 6pt 时**改为换行**（旧规则缩到 3pt 保持一行，即扫描报表上 44 格 <5pt 的软肋；本次按需求反转——多行 + 行带限制，见上）；≥2 行源 → `_fit_exact_n`（二分查最大字号使 `_wrap` 行数 ≤ n，再 `_rebalance_to_n` 拆宽行 / `_merge_to_n` 合并短行凑到恰好 n——拆行用 `_split_line_half` 尽量取空格切、决不断整数字形）。曾出现译文 2 行对原文 1 行（行长推挤、数字错位），**改协议前先看 `_fit_exact_n` 相关回归**（`TableCellFitTest.test_fit_block_preserves_two_line_source_count` 等）。
- **OCR 网格的绘制空间（`fit_width`）**：扫描报表的 OCR 框只围住印刷字形——「合并」表头框约 18pt 而真列约 70pt，按框拟合会把 "Consolidated" 压到 3pt（回归 `test_grid_subcolumn_header_gets_row_gap_as_fit_width`）。`_reconstruct_ocr_grid` 由此规则：**bibbox 恒保持自家 OCR 框**（cover/红action 只擦印刷字，绝不擦邻居单元格）；**图格子列里的文本**（合并/母公司表头、签字行标签）`fit_width = 行内下一格 x0 − 2 − x0`（无下一格用页右缘）——译文可用到行内空白；**标签列最左文本格**同理，但以行内下一个单元格（含附注 `(二)` 带、行次数字）为界，长译文在下一格前停下而不被其白色 cover 切成断层（回归 `test_grid_label_fit_width_stops_before_the_note_marker`）；**附注 `(二)`/行次数字**（标签列内嵌套、非最左）保持自家框 + `align="center"`（曾被误扩到整列并压在标签译文上）；**OCR 拆碎的标签碎片**（`营业利润（亏损以` + `号填列）`，间隔 ≤ 20pt、非 `(二)` 式注记）由 `_merge_label_fragments` 合并为一个单元格（回归 `test_grid_merges_split_label_fragments`）。**`fit_height`（行带）**：网格行距不固定——报表各节实测 ≈10pt（利润表「其他综合收益」密集小行）到 ≈23pt（标签行）——行带本身装得下 1–2 行文本，扫描件根本不需要推挤行高（位图表格线、签字墨迹钉死在原位，行扩展必然错位），所以给每个单元格加 `fit_height = 下一行行顶 − 单元格 y0 − 1.5`（末行 = 0），换行被限制在行带内、停在下一条表格线之前（回归 `test_grid_cells_get_row_pitch_as_fit_height`、`test_fit_block_band_wraps_at_readable_floor_when_band_is_plenty`、`test_fit_block_band_descends_when_the_band_is_tight`；真机实测见下）。改动这些语义时递增 `_OCR_CACHE_VERSION`（当前 5：2=加 fit_width、3=标签不再扩 bbox、4=碎片合并、5=加 fit_height——v4 缓存反序列化得 fit_height=0，换行无界、译文会挂进下一行）。
**真机实测**（`verify_real_run.py`：配置的 qwen3.8 翻译 p24-27 的 824 表内格，`resume=False` 不写缓存）：42 格多行、**零行带越界**、0 格触底 3.0（旧规则 5 格）、<5pt 从 44 格降到 29 格。余下 15 格 3.1–4.0pt 全是长英文标签 × 密集行距的**物理极限**（p26 OCI 1–9 子行 ≈9.8pt 行距、p27 现金流量补充行）：行带在 3–4pt 两行与 4pt 单行之间自动选更大字号（`_fit_band` 的二分已涵盖「行内放得下时退回单行」），再大的字号必然越线或丢字——这对扫描件不可回避，而非实现缺陷。附注/行次/数字列逐格核对无误（源扫描本身即有重复 `（四十九）` 三处，OCR 与译文忠实保留）。

### translate_app/main_window.py
- `LANGUAGES` 将界面显示名映射为发送给模型的语言名。语言下拉框**可编辑**，因此必须用 `resolve_language(currentText())` 取值，**不可用 `currentData()`**：输入的文本不匹配任何条目时 Qt 会保留原 `currentIndex`，`currentData()` 仍指向上一次选中项，用户输入会被静默丢弃（请求与输出文件名一起用错语言）。表中没有的语言原样透传给模型。
- 线程生命周期：每次运行创建 `QThread` + worker；`worker.stopped`（在 `run()` 的 `finally` 中发出）连到 `thread.quit`，**同时连到 `worker.deleteLater`**——此刻 worker 线程的事件循环仍在运行，`DeferredDelete` 会被真正处理、C++ 对象被释放；若在 `_cleanup`（线程已结束后）里对 worker 调 `deleteLater`，事件会投给已死的事件循环、永不处理，每次运行泄漏一个 QObject，所以 `_cleanup` 只丢弃 Python 引用（**不要**把 worker 的 `deleteLater` 加回 `_cleanup`）。`thread.finished` 连到 `_cleanup`。**不要改回从 `finished`/`error` 退出线程**——取消时二者都不发，线程事件循环会永远不退出，`_cleanup` 不执行，「开始翻译」按钮永久禁用。关闭窗口即「六亲不认」地强行退出——不弹任何确认框，直接 `thread.terminate()` + `thread.wait()` 强制结束 worker 线程，再用 `os._exit(0)` 硬退整个进程，以绕过 `concurrent.futures` 在解释器退出时对非 daemon HTTP 线程的 join 阻塞（因此绝不等待当前请求完成；缓存的原子写正是为此保底）。
- 运行状态闭环：`_start` 重置 `_run_ok` / `_cancelled_by_user` 标志；`_on_finished` 置 `_run_ok=True`；`_on_error` 调 `_settle_progress(0, "发生错误")` 把进度条从忙状态（`setRange(0, 0)` 旋转条）落定；`_cleanup` 只在**未成功且进度条仍是忙状态**时落定为「已取消」/「就绪」——取消不发 `finished`/`error`，没有这个兜底，进度条会永远旋转、阶段标签停在最后阶段。
- OCR 开关：表单中以「识别扫描页」行提供 `QCheckBox`（默认勾选；OCR 语言自动跟随原文，无独立下拉），随 `_save_prefs` 持久化到 `prefs.json`（`ocr`），`_start` 时传给 `TranslateWorker(ocr=...)`。
- PDF 可拖放到窗口设置源文件；命令行传入的 PDF 路径效果相同。拖放只接受**本地 PDF 文件**（`dragEnterEvent` 校验 `isLocalFile()` + `.pdf` 后缀；旧实现接受任意 URL，目录/网页链接悬停显示可接受、落下却静默忽略）。
