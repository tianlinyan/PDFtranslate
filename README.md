# PDFtranslate

> 当前版本：**v0.6.4**（版本号定义于 `translate_app/__init__.py` 的 `__version__`）
> 更新日志：[`CHANGELOG.md`](CHANGELOG.md) ｜ 设计文档：`docs/` ｜ 架构与实现约束：`CLAUDE.md`

一个 Windows 桌面 **PDF AI 翻译**工具。打开一个 PDF，选择 AI 模型与目标语言，
即可把文档翻译成指定语言，并保存为**双语 PDF / 仅译文 PDF / Markdown / 纯文本**。

* 扫描件自动 OCR（`rapidocr_onnxruntime`，内置中英文模型、离线）
* 原位翻译：保留原页面图片/矢量图与版式，译文写回原文位置
* 侧栏常驻 AI 对话：用一句中文下达「翻译第 3–8 页」「只查数字」这类要求
* 导出后有两套**离线体检脚本**（内容 / 排版），不需要模型即可验收产物

---

## 目录

1. [功能概览](#功能概览)
2. [安装与运行](#安装与运行)
3. [界面与使用](#界面与使用)
4. [三条翻译路径](#三条翻译路径)
5. [模型配置 `models.json`](#模型配置-modelsjson)
6. [环境变量](#环境变量)
7. [输出格式与版式](#输出格式与版式)
8. [术语表 `glossary.json`](#术语表-glossaryjson)
9. [导出后体检（CLI）](#导出后体检cli)
10. [项目结构](#项目结构)
11. [测试](#测试)
12. [故障排查](#故障排查)

---

## 功能概览

| 能力 | 说明 |
|---|---|
| 输入 | 单个 PDF（文本层 / 扫描件 / 混合页），支持拖放 |
| 输出 | 双语 PDF、仅译文 PDF、Markdown、纯文本 |
| 模型 | `models.json` 中任意 OpenAI 兼容 `/chat/completions` 端点（本地 llama.cpp / 云端） |
| OCR | 无文本层页自动识别；混合页里**烧在位图上的文字**可选识别（默认开） |
| 表格 | 文本层表格原位重排、无框线表格识别、扫描表格可选重建为矢量表格 |
| 交互 | 侧栏 AI 对话（控制台）＋特殊页协商＋预览框选 |
| 质检 | `check_translation.py`（内容）、`check_layout.py`（排版）离线验收 |

## 安装与运行

环境要求：**Python 3.10+**（Windows）。

```powershell
pip install -r requirements.txt
python main.py                     # 启动界面
python main.py "C:\path\to\doc.pdf"  # 启动并预选文件
```

Windows 下也可以直接双击 `run.bat`。注意 `run.bat` 与 `python main.py` 的**默认行为不同**：

| 启动方式 | 默认管线 | 说明 |
|---|---|---|
| `python main.py` | agent / 确定性批次 | 交互最完整：特殊页协商、自检、预览 |
| `run.bat` | **IR 文档级管线** | 预设 `PDFTRANSLATE_IR_MODE=1`、`PDFTRANSLATE_STRUCTURE_PARSER=doclayout`、`PDFTRANSLATE_AGENT_TERMS=1`、`PDFTRANSLATE_DOCLAYOUT_DEVICE=cpu`；装了 `doclayout_yolo` 走语义结构，缺则自动降级几何后端 |
| 主界面勾选「IR 文档级管线」 | IR | 等价于 `PDFTRANSLATE_IR_MODE=1`，状态写入 `prefs.json` |

依赖（`requirements.txt`）：`PyQt6`（界面）、`PyMuPDF`（PDF 处理与导出）、`openai`（模型调用）、
`Pillow`（对话截图压缩）、`rapidocr_onnxruntime`（OCR）、`numpy<2`（onnxruntime/opencv 兼容）。

## 界面与使用

### 主窗体

1. **源文件**：点「打开 PDF…」，或把 PDF **拖放**到窗口上。
2. **AI 模型 / 目标语言 / 输出格式 / 保存到**：保存路径留空则自动生成到源文件目录
   （目标文件已存在时**直接覆盖**；仅当输出路径等于源 PDF 时才另存 `(1)`，避免毁掉输入）。
3. **选项区**（三行，每行两格）：

   | 标签 | 复选框 | 默认 | 作用 |
   |---|---|---|---|
   | 翻译管线 | IR 文档级管线 | 关 | 走 `build_ir → translate_ir → 导出`；公式/数字保真、术语跨页一致，但无特殊页协商/自检/预览 |
   | 术语注入 | 文档级术语 | **开** | AI 编排翻译前先抽取全文术语统一翻译一次；只对 agent 路径生效（IR 自带术语抽取） |
   | 扫描重建 | OCR表格重建为矢量表格 | 关 | 扫描表格页重绘为干净矢量表格；**扫描底图、印章、手写签字不再保留** |
   | 图内文字 | 翻译图内文字 | **开** | 混合页里烧在位图上的文字（柱状图标题/轴标签/截图/流程图）OCR 后覆盖并重画译文 |
   | 译文扩页 | 译文扩页 | 关 | 源页放不下的表格行/正文改排到**新增的后续页**（续页重复表头），不再压缩行高或缩到可读下限以下；**只对「仅译文/原位」PDF 的文本层内容生效**，扫描件与图内文字不可重排 |

4. **按钮**：`预览`（源页/译文页对照、可框选区域提问）、`重新导出`（用对话/标注里的修改重出文件，
   **不重新翻译**，秒级）、`开始翻译`、`取消`、`打开输出`、`关于`。

> **「重新导出」复用上一次运行对齐用的提取结果**（`DocumentText`）——所以它不会二次 OCR。
> 若源文件或提取设置变了（例如中途切换「翻译图内文字」、块数不一致），会提示重新点「开始翻译」。

### 侧栏 AI 对话（控制台）

按钮「开始翻译」等价于在侧栏输入「开始翻译」并发送：AI 读取当前界面设置后调用 `run_translate`。
你也可以直接说自然语言，例如：

```
翻译第 3 到 8 页
只检查数字，不要改任何东西
第 5 页的图保留原文
把这句话改得更正式
重新导出
```

* 页范围、检查项、是否自动修改等由**模型解析**；模型不可用时退回离线规则解析。
* 特殊页（扫描/图表/待确认）默认**自动翻译**，不再逐页询问；你的具体要求会作为上下文注入。
* 对话的修改保存在受保护覆盖层里，导出时优先于整篇翻译结果。

## 三条翻译路径

`TranslateWorker` 按**模型与开关**自动选择（`translate_app/worker.py`）：

| 路径 | 触发条件 | 特点 |
|---|---|---|
| **IR 文档级管线** | 勾选 IR 复选框，或 `PDFTRANSLATE_IR_MODE=1` | 无交互批处理；公式/数字/图表角色保真；段落成组送译（`PDFTRANSLATE_IR_GROUP=0` 关闭）；术语跨页一致 |
| **AI 编排（agent 视觉闭环）** | 未开 IR，且模型 `vision: true` | 逐页「观察→翻译→校验→复检」；可 `render_page` 看图自检、可 `ask_user`；支持特殊页协商与预览框选 |
| **确定性批次流水线** | 未开 IR，且模型无视觉能力 | 按字符预算分批直译；不逐页视觉判断，印章/手迹等由**内容策略**（模型，文本侧）判定 |

三条路径共用同一翻译引擎（分批、编号协议、重试、取消）与同一导出器。

## 模型配置 `models.json`

每个条目描述一个 OpenAI 兼容的 `/chat/completions` 端点。项目自带的 `models.json` 可直接改，
也可复制 `models.example.json`（带全字段样板）。**面向普通用户的图文教程见 [`AI模型配置手册.md`](AI模型配置手册.md)。**

```json
{
  "models": [
    {
      "id": "qwen3.8-local",
      "name": "qwen3.8（本地）",
      "type": "llama-server",
      "endpoint": "http://192.168.0.48:8888/v1/chat/completions",
      "model": "qwen3.8-27b",
      "vision": true,
      "reasoning_effort": "low",
      "temperature": 0.1,
      "concurrency": 1,
      "page_concurrency": 1,
      "batch_size": 8192,
      "max_blocks_per_batch": 40
    },
    {
      "id": "ds4-pro",
      "name": "DS4",
      "type": "deepseek",
      "endpoint": "https://api.deepseek.com/v1/chat/completions",
      "model": "deepseek-flash",
      "api_key": "${DEEPSEEK_API_KEY}",
      "tools_choice": "auto",
      "concurrency": 4,
      "page_concurrency": 4,
      "batch_size": 8000
    }
  ]
}
```

### 字段全表

| 字段 | 默认 | 说明 |
|---|---|---|
| `id` / `name` | — | 内部标识（偏好记忆用）与界面显示名 |
| `type` | `openai` | 仅作标签，不改变行为 |
| `endpoint` | — | **完整** chat-completions URL（`client_kwargs()` 去掉 `/chat/completions` 得 `base_url`） |
| `model` | — | 传给服务端的模型名 |
| `api_key` | 无 | 支持 `${ENV_VAR}` 占位符，运行时读取环境变量；本地端点可省略（发送 `not-needed`，未解析的占位符**绝不**当密钥发出） |
| `tools_choice` | 无 | 映射为请求体 `tool_choice`（如 `"auto"`） |
| `reasoning_effort` | 无（不发送） | **翻译侧**请求参数，经 `extra_body` 发出；llama.cpp 类服务端缺它可能 500，一般设 `low` |
| `enable_thinking` | 无（不发送） | Qwen3 类思考开关，经 `extra_body` 发出；部分服务端忽略 |
| `temperature` | 引擎默认 0.2 | **翻译侧**采样温度 |
| `max_tokens` | 服务端默认 | 单次请求最大输出 token 数 |
| `concurrency` | 1 | 同时发送的**批次**请求数（云 API 可 2–4） |
| `page_concurrency` | 1 | agent 路径的**并行页**翻译数（云 API 2–4 近线性提速，本地单卡仍排队） |
| `batch_size` | 4000 | 每批请求的**原文字符预算**；加大可显著减少请求次数 |
| `max_blocks_per_batch` | 25 | 每批**块数**上限（与字符预算同时生效）；块太多时模型容易回不对 `[[n]]` 标记 |
| `vision` | `false` | 该模型能否看图；为真才走 agent 视觉闭环，否则回退确定性批次 |
| `interaction_temperature` | 0.6 | **交互侧**（侧栏对话）温度 |
| `interaction_reasoning_effort` | `"medium"` | **交互侧** `reasoning_effort`，同样经 `extra_body` |
| 其它未识别键 | — | 透传给 OpenAI client 构造参数（如 `timeout`，默认 300 秒） |

> **两套请求参数不要混**：`temperature` / `reasoning_effort` = **翻译侧**（翻译引擎、agent 决策、
> 自检、内容策略、导出）；`interaction_temperature` / `interaction_reasoning_effort` = **交互侧**
> （仅侧栏常驻对话）。二者独立配置。

### 调优建议

| 参数 | 本地 `llama-server` | 云端 API |
|---|---|---|
| `concurrency`（批次并发） | 1–2（GPU 是瓶颈） | 4 |
| `page_concurrency`（并行页） | 1（本地串行排队） | 4 |
| `batch_size`（每批字符预算） | 12000（少请求、摊薄固定开销） | 4000 |
| `max_blocks_per_batch` | 40 | 25 |
| `temperature` | 0.1–0.2 | 0.2 |
| `reasoning_effort` | `low`（缺它 llama.cpp 可能 500） | `low` |
| `vision` | `true`（走 agent 视觉闭环） | 视模型 |

## 环境变量

界面开关已覆盖绝大多数场景；下列变量用于**强制覆盖**或排查（多数与开关同级或更高优先级）。
设置后需**新开终端**才生效。

| 变量 | 默认 | 作用 |
|---|---|---|
| `PDFTRANSLATE_IR_MODE=1` | 关 | 强制走 IR 文档级管线（**优先级高于界面复选框**，勾选状态无法覆盖它） |
| `PDFTRANSLATE_STRUCTURE_MODE=1` | 关 | 提取时带语义结构（公式/图/标题/图注/表格） |
| `PDFTRANSLATE_STRUCTURE_PARSER` | `geo` | 结构后端：`geo`（几何、离线）或 `doclayout`（DocLayout-YOLO，**设为它即隐含开启结构模式**） |
| `PDFTRANSLATE_DOCLAYOUT_DEVICE` | `cpu` | DocLayout 推理设备（如 `cuda:0`） |
| `PDFTRANSLATE_IR_GROUP=0` | 开 | 关闭 IR 的段落成组送译（退回逐块） |
| `PDFTRANSLATE_AGENT_TERMS=0` | 开 | 关闭 agent 的文档级术语抽取 |
| `PDFTRANSLATE_REFLOW=0` | 开 | 关闭文本层表格**列宽重排**（v0.5.47 起该能力恒开、无界面开关，此变量供对照旧行为） |
| `PDFTRANSLATE_REBUILD_TABLE=1` | 关 | 强制开启「OCR表格重建为矢量表格」 |
| `PDFTRANSLATE_EXPAND_PAGES=1` | 关 | 强制开启「译文扩页」（源页放不下的内容排到新增的后续页） |
| `PDFTRANSLATE_FONT_SCALE` | `0.9` | 译文字号缩放（同号下西文比汉字显大）；`1.0` = 不缩放，越界/非法值回落到 0.9 |
| `PDFTRANSLATE_CONTENT_POLICY=0` | 开 | 关闭 **AI 内容策略**（哪些内容保留原文交给模型判定） |
| `PDFTRANSLATE_OCR_BACKEND` | `rapidocr` | OCR 后端：`rapidocr` 或 `vlm`（需模型支持视觉；不可用时降级） |
| `PDFTRANSLATE_CACHE_DIR` | **未设＝内存缓存** | 设了才把译文缓存**落盘**到该目录（测试用；生产不写盘，见下） |
| `PDFTRANSLATE_OCR_CACHE_DIR` | **未设＝内存缓存** | 同上，针对 OCR 结果缓存 |
| `PDFTRANSLATE_FLOWS_DIR` | 未设＝仅内存 | 用户自定义流程（`FlowSpec`）的落盘目录 |

### 缓存与复跑语义（容易误解）

* **生产运行不写任何中间缓存文件**：译文与 OCR 结果只在本次进程的内存里累积，只保存最终输出。
  每次运行都是全新的——取消或崩溃后重跑会从头提取/OCR/翻译。
  如需磁盘缓存（测试、或想跨运行复用），显式设置 `PDFTRANSLATE_CACHE_DIR` / `PDFTRANSLATE_OCR_CACHE_DIR`。
* 缓存键含**文档路径 + 目标语言 + 模型 id + 术语表内容哈希**；改动缓存语义时会递增内部版本号，
  旧缓存自动失效。
* 失败批次**绝不写入缓存**，也绝不用原文填补缺口（否则一次模型拒答会永久污染该块）。
* 缓存目录默认 `~/.pdftranslate/cache`（OCR 为 `~/.pdftranslate/ocr_cache`），不可写时回退系统临时目录。

## 输出格式与版式

| 格式 | 行为 |
|---|---|
| **双语 PDF** | 原文页之后紧跟译文页（源页 i ↔ 译文页 2i+1）；译文块按原文块 bbox 对齐排版；带图片/矢量图的页会保留图形；无文本页写提示文案 |
| **仅译文 PDF** | 原位翻译：保留图片与线条，抹掉原文文字后在原位置画译文；字号从「原字号 × 0.9」（`PDFTRANSLATE_FONT_SCALE`）起缩到装进原框，正文可读下限 6.3pt（表格 5.4pt） |
| **Markdown / 纯文本** | 仅文字，按页顺序输出 |

版式细节（用户可见的行为承诺）：

* 多栏页按「左列自上而下、再右列」的阅读顺序提取；横跨两栏的整宽标题不会被并成一栏。
* 表格单元优先保持**单行**（数字/短译文永不换行）；装不下时在可读下限换行，且**绝不越过表格线**。
* 全部翻译文字按 `PDFTRANSLATE_FONT_SCALE`（默认 0.9）统一缩放：同号下西文比汉字「重」，不缩放会整体显大；文本层页与 OCR 页走同一条拟合路径，一处生效、全局生效。
* 扫描页 OCR 块先按实测字形带画白底再画译文（白底避开表格线；照片/彩色底上的文字保留原像素，只画译文）。
* 手写体签字区（页底、超高、纯字母无数字且 ≤6 字符）**不翻译、不覆盖**，保留扫描墨迹。
* 旋转页（`/Rotate 90/180/270`）与 CropBox 裁剪页按未旋转帧判界，白底/采样映射到渲染帧。
* 组织架构图 / 架构图（`≥3` 个窄高节点框的页）默认**保留原图内标签**（只译页眉/大标题）；
  这是默认策略而非硬锁——你明确要求时 AI 可以逐块覆盖。
* 数字、金额、公式属**红线**：保留原文或按值严格对应，AI 内容策略也动不了它们。

## 术语表 `glossary.json`

把 `glossary.json` 放在**文档所在目录**（扁平 JSON 对象，键=源词、值=目标词）即可生效：

```json
{ "合并资产负债表": "Consolidated Balance Sheet", "万元": "ten thousand yuan" }
```

* 内容随提示词注入每个批次，并计入缓存键（改术语表 = 换缓存），不会沿用旧译法。
* 加载失败会写日志并忽略（不会静默当作已生效）。
* 侧栏对话也可以用「术语」类要求补充本次运行的临时术语（`apply_terminology` 工具，优先级高于文件）。

## 导出后体检（CLI）

两个脚本都**只看导出产物、不需要模型**，供人工或 CI 验收。

```powershell
python check_translation.py 原文.pdf 译文.pdf [--lang English] [--strict] [--skip 24-27]
```

检查**内容**：数字一致性（按值比较，含全角/千分位/单位倍率换算）、残留未译文本、章节编号、页数。
退出码：`0` 通过 ｜ `1` 数字不一致（或 `--strict` 下任意告警）｜ `2` 用法错误 / 文件打不开。
双语产物按「源页 i ↔ 译文页 2i+1」配对；扫描页跳过数字比对但仍查残留。

```powershell
python check_layout.py 原文.pdf 译文.pdf [--page 3,5-7] [--dpi 150]
```

检查**排版**：译文重叠（字形带相交）、越出页面、压在源页表格线上（该线必须**延伸到字形之外**
才算压线——12pt 汉字的横笔也有 ≈11pt 长）、字号过小、漏画、页数/页尺寸。
字号按**该块自己的**下限判（与导出器 / AI 自检同口径：表格单元 3pt，其它 `min(原字号×0.9, 6.3pt)`，
所以 5pt 的脚注按 5pt 画不算问题）；源页没有文本层的页（扫描件）或对应不到原文块的译文
不与原文比对，只报 < 3pt 的，其余偏小文字**汇总成一行提示**。
退出码：`0` 无结构性问题（字号偏小只告警）｜ `1` 重叠/出页/压线/漏画 ｜ `2` 用法错误 / 页码越界。

### 其它开发/评估脚本

| 脚本 | 用途 | 需要模型 |
|---|---|---|
| `make_diverse_test_pdf.py [out.pdf]` | 生成 5 页多样化测试 PDF（双栏 / 图文表 / OCR 文本 / OCR 表格 / OCR 混合） | 否 |
| `eval_harness.py --pdf-only 原文 译文` | 文本层后验评估（复用 `check_translation`，可 `--json` 出报告） | 否 |
| `eval_harness.py --outdoc 原文 out_doc.json` | 逐块版面度量与 A/B 对比（`--baseline`/`--compare`/`--judge`） | 否（`--judge` 需在线） |
| `eval_ir.py [--pdf X] [--run] [--model ID] [--pages 0-4]` | IR 文档级翻译 vs 逐块翻译 A/B | `--run` 时是 |
| `eval_doclayout.py [--pdf X]` / `verify_doclayout.py [--probe]` | DocLayout 语义结构 vs 几何后端的效果核对 | 否 |
| `verify_real_run.py` | 真机对照：翻译年报 p24–27，测量表内格拟合（字号桶/行带越界） | 是 |
| `verify_selfcheck.py` | 真模型走查：对样例页跑确定性审计 + AI 复核循环 | 是 |
| `check_band_fit.py` / `export_band_check.py` | 单页行带拟合与再导出的定向排查（硬编码样例文件名） | 否 |

## 项目结构

```
main.py                   入口（PyQt6）        run.bat   Windows 启动器（预设 IR/DocLayout）
models.json / models.example.json              模型配置（配错时看 AI模型配置手册.md）
glossary.json（可选，放在文档目录）              术语表
check_translation.py / check_layout.py         导出后内容 / 排版体检
eval_*.py / verify_*.py / make_diverse_test_pdf.py   开发、评估与真机验证脚本
translate_app/
  main_window.py          主窗体、worker 生命周期、导出触发
  worker.py               TranslateWorker：提取 → 翻译 → 导出；三条路径的分派
  pdfio.py                提取、OCR、表格/图表重建、全部导出器（最大的模块）
  translator.py           批次翻译引擎：分批、编号协议、重试、缓存、术语
  prompts.py              全部提示词（系统/任务/工具策略/对话），单一来源
  policy.py               AI 内容策略：候选筛选 + 模型判定 + 落到 keep_original
  ir.py / ir_pipeline.py  IR 文档级管线（build_ir → translate_ir）
  chat.py / chat_tools.py 侧栏常驻对话与对话工具
  doc_context.py          对话可见的文档上下文（当前译文/结构）
  preview.py / sidebar.py / about_dialog.py   预览窗口 / 侧栏 / 关于
  settings.py             models.json 与 prefs.json 的解析与原子保存
  eval.py / mcp.py        评估原语 / MCP 相关
  control.py              控制信号基类（取消/中止统一上抛）
  agent/
    flow.py               DocumentSession：逐页/特殊页/自检流程与确定性审计引擎
    flow_steps.py         Flow/ToolStep/AgentStep/UserStep/LoopStep/IfStep + run_flow
    tool_catalog.py       工具目录（schema + description + audience 单一来源）
    tools.py / state.py   工具装配 / WorkflowState
    user_flows.py         用户自定义流程：规则或 AI 把一句话编译成 FlowSpec / Plan
tests/                    23 个模块、877 条用例（离线，本地 mock chat-completions）
docs/                     阶段性设计、路线图与代码审查报告（20 篇）
```

用户偏好保存在 `~/.pdftranslate/prefs.json`（模型、语言、输出格式、上次目录、四个复选框状态），
原子写入，跨启动记忆。

## 测试

用 `unittest` 运行回归测试，**无需联网**：`tests/_helpers.py` 提供本地 mock chat-completions 服务
（按块回显 `[N] MOCK:<原文>`），`build_sample_pdf()` 生成小型测试 PDF。

```powershell
python -m unittest discover -s tests -v                       # 全部（877 条）
python -m unittest tests.test_pdfio.PdfioTest.test_bilingual_pdf -v   # 单条
```

> ⚠️ **涉及缓存的测试必须把缓存重定向到临时目录**（`PDFTRANSLATE_CACHE_DIR` /
> `PDFTRANSLATE_OCR_CACHE_DIR`）——否则会污染开发者 home，并因热缓存而**假绿**。

覆盖范围：配置解析与校验、环境变量密钥、翻译分批/编号对齐/多行回复/并发顺序/失败不写缓存、
文本提取与各类导出（双语对齐、原位翻译、缩字号、对齐、分栏顺序）、OCR 管道与数字归一化、
worker 信号契约、工具注册表与流程引擎、会话控制器、对话与对话工具、内容策略、
质检脚本（`test_check_translation` / `test_check_layout` 会真正跑导出产物）、Qt 接线。

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| 日志提示「翻译已中止：…」 | 认证/模型名/地址类错误（HTTP 401/403/404）不重试、不降级——否则会「成功」导出一份与原文逐字相同的文档。检查 `api_key`/`model`/`endpoint` |
| 部分块保留原文并告警 | 瞬时故障（网络、408/429/5xx、回复格式不完整）重试 3 次仍失败。失败块**不写缓存**，重跑会重试 |
| llama.cpp 返回 500 | 多为缺 `reasoning_effort`——在 `models.json` 里设 `"reasoning_effort": "low"` |
| 界面勾了选项却看不出效果 | 日志首行有 `导出选项：OCR表格重建=开/关，表格列宽重排=开/关`，先核对该行是否为当前勾选值 |
| 取消勾选「翻译图内文字」后重跑，图内文字仍是译文 | 「重新导出」复用上次提取结果；块数/设置不一致时会提示，请点「开始翻译」重新提取 |
| OCR 相关告警一次后扫描页无译文 | OCR 引擎不可用（未装 `rapidocr_onnxruntime` 等）会**降级跳过**该页并告警一次，不中断整篇 |
| 扫描表格译文很小 | 密集报表窄列的物理极限（可低到 3–5.4pt）。勾选「OCR表格重建为矢量表格」让行高按译文重排 |
| 想让 AI 不翻译某些内容 | 直接说（如「第 5 页图保留原文」「这个印章别翻」）；默认策略之外的决定由 AI 内容策略逐块给出 |
| 偏好没有跨启动记住 | 检查 `~/.pdftranslate/prefs.json` 是否可写；保存失败会写入日志而非静默丢弃 |

---

* 版本变更历史：[`CHANGELOG.md`](CHANGELOG.md)
* 模型配置图文教程：[`AI模型配置手册.md`](AI模型配置手册.md)
* 阶段性设计 / 代码审查报告：`docs/`
* 架构约束与实现要点（面向维护者）：`CLAUDE.md`
* 开发者：Tian Linyan ｜ 项目主页：<https://github.com/tianlinyan/PDFtranslate>
