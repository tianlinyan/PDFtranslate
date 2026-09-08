"""The single source of truth for every tool the AI may expose.

Consolidates what used to live in ``agent/tools.py`` (``AGENT_TOOLS``) and
``chat_tools.py`` (``CHAT_TOOL_SPECS``) plus their two prompt-description dicts
(``AGENT_TOOL_DESCRIPTIONS`` / ``CHAT_TOOL_DESCRIPTIONS``) into **one catalog**:
each tool is a :class:`ToolDef` with its schema, its model-facing description and
its ``audience`` (``{"agent"}`` / ``{"chat"}`` / both).  The two exposure lists are
then just audience-filtered views (``agent.agent_openai_tools`` / ``catalog_for``),
so a tool's schema + description live in exactly one place.

Semantics differ by side only where genuinely needed:

* ``read_page`` has **two** entries (agent: source blocks + ``offset/limit`` paging;
  chat: source + current translation, no pager) — same model-facing name but disjoint
  audiences, so the two never collide in one ``tools`` array.
* The other shared tools (``get_doc_info`` / ``classify_page`` / ``get_structure`` /
  ``get_table`` / ``apply_annotation``) have identical schemas and near-identical
  descriptions, so one entry serves both audiences; their *implementations* are still
  bound separately per side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Tool categories (grouping for the controller's prompt / the UI).
CAT_READ = "read"
CAT_CONTENT = "content"
CAT_VERIFY = "verify"
CAT_UI = "ui"   # interaction / presentation (e.g. pop a preview window)


@dataclass(frozen=True)
class ToolDef:
    """One tool's schema + metadata.

    ``parameters`` is the JSON Schema for the arguments the model may pass.
    ``target`` says which side the tool acts on: ``"source"`` (the immutable
    original — read/observe only, never written) or ``"output"`` (the mutable
    translation — freely editable).  **The source is never written**.
    ``audience`` is the set of exposing sides (``"agent"`` = translation
    orchestrator, ``"chat"`` = the persistent conversation console).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    category: str
    target: str = "output"
    destructive: bool = False
    returns: str = ""
    audience: frozenset[str] = field(default_factory=lambda: frozenset({"agent"}))
    #: The tool-tier taxonomy (see docs / CLAUDE.md): every catalog entry is a
    #: **原子工具** (``"atomic"``) — a single-purpose, irreducible operation.  Higher
    #: tiers live on ``Flow`` (``"process"`` / ``"composite"``).
    tier: str = "atomic"


def _tool(name: str, description: str, properties: dict[str, dict], required: list[str],
          category: str, *, target: str = "output", destructive: bool = False,
          returns: str = "", audience: tuple[str, ...] = ("agent",),
          tier: str = "atomic") -> ToolDef:
    return ToolDef(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
        },
        category=category,
        target=target,
        destructive=destructive,
        returns=returns,
        audience=frozenset(audience),
        tier=tier,
    )


def _compose_description(t: ToolDef) -> str:
    """The model-facing description: WHAT the tool does + what it returns."""
    desc = t.description
    if t.returns:
        desc = f"{desc}。返回：{t.returns}"
    return desc


def to_openai_schema(t: ToolDef) -> dict[str, Any]:
    """The OpenAI ``tools``-array entry for a :class:`ToolDef`."""
    return {
        "type": "function",
        "function": {
            "name": t.name,
            # ``description`` (the WHAT) plus ``returns`` (the RESULT shape and any
            # index semantics) — both are prompt text the model reads to understand a
            # tool's capability and pick the right one.
            "description": _compose_description(t),
            "parameters": t.parameters,
        },
    }


def catalog_for(audience: str) -> list[ToolDef]:
    """The tool definitions visible to a side (``"agent"`` or ``"chat"``)."""
    return [t for t in TOOL_CATALOG if audience in t.audience]


#: The deterministic audit check names ``audit_page`` accepts — the **single
#: source** for both the check registry (``agent.flow._AUDIT_CHECKS``) and the
#: validation of model-supplied names (``agent.flow.audit_page`` /
#: ``agent.user_flows``).  A name outside this tuple must never be silently
#: dropped: ``audit_page`` reports it as an ``unknown_checks`` issue and forces
#: ``clean=False``, otherwise a misspelled / Chinese check name would return a
#: "clean" page with nothing having run.
AUDIT_CHECK_NAMES: tuple[str, ...] = ("layout", "residual", "missing", "numbers", "table")

#: Tool tiers (the app single-request taxonomy).  Catalog entries are always atomic;
#: ``Flow`` adds the two higher tiers.
TIER_ATOMIC = "atomic"
TIER_PROCESS = "process"
TIER_COMPOSITE = "composite"
TASK_TIERS = (TIER_ATOMIC, TIER_PROCESS, TIER_COMPOSITE)


def atomic_tool_names() -> set[str]:
    """Every atomic-tool name (all audiences), for a Path-B plan validator."""
    return {t.name for t in TOOL_CATALOG if t.tier == TIER_ATOMIC}


def tool_tier(name: str) -> str:
    """The tier of a named atomic tool (``"atomic"``, or ``""`` when unknown)."""
    for t in TOOL_CATALOG:
        if t.name == name:
            return t.tier
    return ""


#: The one catalog — EVERY tool the AI may expose, keyed by side via ``audience``.
TOOL_CATALOG: list[ToolDef] = [
    # ------------------------------------------------------------------ 共享（两边都用）
    _tool("get_doc_info",
          "返回当前 PDF 的信息：页数/标题/语言/文本页/扫描页/图表页/块数/每页类型（表格页恒为 0）。",
          {}, [], CAT_READ, target="source",
          returns="{pages, title, language, text_pages, scan_pages, chart_pages, table_pages, block_count, kinds}",
          audience=("agent", "chat")),
    _tool("classify_page",
          "判定某页类型：normal/scan/chart/uncertain；有语义结构时可为 formula/figure。",
          {"page": {"type": "integer"}}, ["page"], CAT_READ, target="source",
          returns="{kind}",
          audience=("agent", "chat")),
    _tool("get_structure",
          "返回某页的语义结构：{page, parser, elements:[{kind,bbox,level,block_indices}], tables}，"
          "kind 为 text/heading/table/figure/formula/caption/note。用于识别公式/图表/标题层级/表格语义；"
          "无结构后端时 elements 为空（按普通文本处理）。",
          {"page": {"type": "integer"}}, ["page"], CAT_READ, target="source",
          returns="该页语义结构：{page, parser, elements:[{kind,bbox,level,block_indices}], tables}；"
                  "无结构后端时 elements 为空",
          audience=("agent", "chat")),
    _tool("get_table",
          "返回某页第 index 个语义表格 {rows, cols, bbox, cells(行×列的扁平块索引), block_ref}；"
          "无表或无结构时返回 None。",
          {"page": {"type": "integer"},
           "index": {"type": "integer", "description": "该页语义表格序号（默认 0）"}},
          ["page"], CAT_READ, target="source",
          returns="该页第 index 个语义表格 {rows,cols,bbox,cells,block_ref}；无则 None",
          audience=("agent", "chat")),
    _tool("apply_annotation",
          "按预览中用户框选的区域改写/删除对应块（用 read_page 拿到的 bbox）。",
          {"page": {"type": "integer"},
           "bbox": {"type": "array", "items": {"type": "number"},
                    "description": "标注框 [x0,y0,x1,y1]，PDF 点"},
           "text": {"type": "string", "description": "替换译文（action=set 时必填）"},
           "action": {"type": "string", "enum": ["set", "delete"], "default": "set"}},
          ["page", "bbox"], CAT_CONTENT,
          returns="是否成功及改写的块/文本",
          audience=("agent", "chat")),

    # ------------------------------------------------------------------ agent 专用
    _tool("read_page",
          "读取指定页的文本块与布局元数据",
          {"page": {"type": "integer"},
           "offset": {"type": "integer", "description": "起始块偏移，用于分页读取大页（默认 0）"},
           "limit": {"type": "integer", "description": "最多返回几块（默认返回该页全部；大页可借此分页）"}},
          ["page"], CAT_READ, target="source",
          returns="该页的文本块列表（每块的 index 为全文档扁平索引，可直接用于 set_text/translate_block；"
                  "含 bbox/字号/对齐/是否表格/是否图表节点；超大页会带 total/truncated 供分页）"),
    _tool("get_layout", "提取某页的行/列聚类、二维网格与单元格跨度",
          {"page": {"type": "integer"}}, ["page"], CAT_READ, target="source",
          returns="{rows, cols, grid}"),
    _tool("render_page",
          "把当前处理到该页的译文渲染成 PNG 供视觉自检（检查溢出/越线/密度）",
          {"page": {"type": "integer", "description": "页号（0 起）"},
           "what": {"type": "string", "enum": ["source", "translation"],
                    "description": "是否渲染原文页，还是当前处理到该页的译文（默认 translation）"}},
          ["page"], CAT_READ, target="source",
          returns="渲染出的 PNG（模型的视觉观察对象；无译文时可渲染源页）"),
    _tool("translate_block",
          "翻译某块并写入其译文（走缓存+编号协议；单块请求，仅翻少数块时用）",
          {"index": {"type": "integer"}, "text": {"type": "string"},
           "target_lang": {"type": "string"}},
          ["index", "text", "target_lang"], CAT_CONTENT,
          returns="翻译后的文本"),
    _tool("translate_blocks",
          "一次批量翻译多块并写入（单次请求、引擎自动按字符预算分批+并发）：传扁平块索引列表，或只传 page "
          "翻译整页所有可翻译块；返回 {count, indices, translated:{index:text}, failed}。整行/整表/整页应优先用它，"
          "比逐块 translate_block 快得多；数字/代码/空块被自动跳过",
          {"page": {"type": "integer", "description": "页号（0 起）"},
           "indices": {"type": "array", "items": {"type": "integer"},
                       "description": "要批量翻译的扁平块索引列表（来自 read_page）；不传则翻译整页所有可翻译块"},
           "target_lang": {"type": "string", "description": "目标语言（默认当前页语言）"}},
          ["page"], CAT_CONTENT,
          returns="一次批量翻译并写入多块：{count, indices, translated:{index:text}, failed:[...]}——"
                  "引擎自动按字符预算分批+并发，远快于逐块 translate_block；数字/代码/空块被自动跳过"),
    _tool("retranslate_block",
          "避开缓存强制重译一段文本，**只返回译文**（不会写入；需再用 set_text 把它写到目标块）。常用于残中/空缺修正",
          {"text": {"type": "string"}, "target_lang": {"type": "string"}},
          ["text", "target_lang"], CAT_CONTENT,
          returns="重译后的文本"),
    _tool("retranslate_blocks",
          "一次批量重译多个块并**直接写入**（单次请求、绕过缓存，避免复用旧译文）：传扁平块索引列表，或只传 page "
          "重译整页所有可翻译块；返回 {count, indices, translated:{index:text}, failed:[...]}。**修正多条 finding 时应优先用它**"
          "（远快于逐条 retranslate_block）；数字/代码/空块被自动跳过",
          {"page": {"type": "integer", "description": "页号（0 起）"},
           "indices": {"type": "array", "items": {"type": "integer"},
                       "description": "要批量重译的扁平块索引列表（来自 read_page）；不传则重译整页所有可翻译块"},
           "target_lang": {"type": "string", "description": "目标语言（默认当前页语言）"}},
          ["page"], CAT_CONTENT,
          returns="一次批量重译并写入多块：{count, indices, translated:{index:text}, failed:[...]}——"
                  "绕过缓存、单次请求，远快于逐条 retranslate_block（修正多条 finding 时首选）；"
                  "数字/代码/空块被自动跳过"),
    _tool("set_text", "把某块文本直接置为指定值（数字/代码块会被拒绝）",
          {"page": {"type": "integer"}, "index": {"type": "integer"}, "text": {"type": "string"}},
          ["page", "index", "text"], CAT_CONTENT,
          returns="是否成功（布尔）"),
    _tool("apply_terminology", "为某源词设定统一的术语译文（会并入本页翻译所用的术语表）",
          {"source": {"type": "string"}, "target": {"type": "string"}},
          ["source", "target"], CAT_CONTENT,
          returns="是否成功（术语并入本页翻译所用术语表）"),
    _tool("delete_block", "移除某块的译文覆盖，恢复为保留原文（不再翻译该块）",
          {"page": {"type": "integer"}, "index": {"type": "integer"}},
          ["page", "index"], CAT_CONTENT,
          returns="是否成功（恢复为该块保留原文）"),
    _tool("check_residual",
          "检查某页是否有未翻译残留（目标为西文时看残留中文；目标为中文时看未译的英文成句；纯代码/缩写/单位不算）与空块",
          {"page": {"type": "integer"}}, ["page"], CAT_VERIFY,
          returns="残留块列表 [{index, text}]"),
    _tool("check_missing", "检查某页是否有源有译文空的块（内容缺失；纯数字/代码块不算）",
          {"page": {"type": "integer"}}, ["page"], CAT_VERIFY,
          returns="缺失块索引列表"),
    _tool("check_numbers",
          "核对某页译文的数字/金额是否与原文**按值**一致：源里的数值被删或改错才报告（千分位/小数/单位拆分等格式差异不算；"
          "中文【序数→数字】如 二→2 不算），返回不一致的块",
          {"page": {"type": "integer"}}, ["page"], CAT_VERIFY,
          returns="数字/金额不一致的块 [{index, source, translation, missing, extra}]"),
    _tool("check_table", "检查某页表格单元格与普通文本块的完整性：源可译单元格数 vs 已译数、空/缺失单元格、遗漏文本块",
          {"page": {"type": "integer"}}, ["page"], CAT_VERIFY,
          returns="{source_cells, translated_cells, empty_cells, empty_text, complete}"),
    _tool("check_layout",
          "仿照导出器重新测量某页译文的版面：看是否低于可读下限、溢出自身框、压入同列下一块（两栏页只与同栏比较）",
          {"page": {"type": "integer"}}, ["page"], CAT_VERIFY,
          returns="版面问题块 [{index, kind, detail}]"),
    _tool("audit_page",
          "对某页一次性跑指定的确定性审计并合并结果：返回 {checks_requested, checks, issues, clean}（issues 是带 check 标签的列单项，"
          "供你逐条修正）；checks 可传子集（如 ['numbers','table']），默认全五类。只读复核（不修任何东西）时把 clean 当作本轮是否达标",
          {"page": {"type": "integer"},
           "checks": {"type": "array", "items": {"type": "string"},
                      "description": "要跑的检查子集：layout/residual/missing/numbers/table（默认全部）；"
                                     "只能传这五个英文名——传了别的名字会作为 unknown_checks 问题返回且 clean=false"}},
          ["page"], CAT_VERIFY,
          returns="{checks_requested, checks, issues, clean}——issues 为带 check 标签的列单项，clean 为本轮是否无问题；"
                  "未知检查名会以 unknown_checks 出现在 issues 里（clean=false，不会静默通过）"),
    _tool("preview_page",
          "在预览窗口显示指定页面（供用户查看），可聚焦某区域/块",
          {"page": {"type": "integer", "description": "页号（0 起）"},
           "what": {"type": "string", "enum": ["source", "translation"],
                    "description": "显示原文页还是译文页，默认 translation"},
           "region": {"type": "array", "items": {"type": "number"},
                      "description": "可选：聚焦的区域 [x0,y0,x1,y1]"}},
          ["page"], CAT_UI,
          returns="是否成功（布尔；弹出非阻塞预览窗口）"),
    _tool("detect_page_skew",
          "检测某扫描页文本的整体倾斜角（度）：返回 {page, skew_degrees, recommended, decision, reason}；若 recommended 且已接入问答通道，"
          "会向用户询问是否做几何校正并把决定记入状态。扫描件翻译前可先调用它判断是否需要（低风险定向）几何校正——它只检测/询问、不修改原 PDF",
          {"page": {"type": "integer", "description": "页号（0 起）"}},
          ["page"], CAT_READ, target="source",
          returns="{page, skew_degrees, recommended, decision, reason}；若 recommended 且已接入问答通道，会向用户询问是否做几何校正并把决定记入状态"),
    _tool("ask_user",
          "向用户提一个清晰的自然语言问题并等待回答（用户用一句话自由回答；options 只是可选的提示文字，不作为按钮）——关键决策/歧义/术语确认时用",
          {"question": {"type": "string"},
           "options": {"type": "array", "items": {"type": "string"},
                       "description": "可选候选答案"},
           "target": {"type": "string", "description": "回答存储键"}},
          ["question"], CAT_UI,
          returns="用户的回答"),

    # ------------------------------------------------------------------ chat 专用
    _tool("read_page",
          "读取某页的原始文本块与当前译文，含扁平块索引（供 set_block_text 使用）与布局元数据。"
          "返回 `page`（0 起）与 `page_number`（1 起）——向用户说明页码时用 `page_number`。",
          {"page": {"type": "integer", "description": "页号（0 起；第 1 页 = 0）"}}, ["page"],
          CAT_READ, target="source", audience=("chat",)),
    _tool("get_settings",
          "返回当前应用设置快照：源文件名、目标语言、输出格式键与显示名、输出路径、模型名称/id、是否 OCR/智能编排。"
          "开始翻译前先看它确认设置。",
          {}, [], CAT_READ, target="source", audience=("chat",)),
    _tool("goto_page",
          "在预览窗口显示指定页（原文/译文侧）。",
          {"page": {"type": "integer", "description": "页号（0 起）"},
           "what": {"type": "string", "enum": ["source", "translation"],
                    "description": "显示原文页还是译文页，默认 source"}},
          ["page"], CAT_UI, audience=("chat",)),
    _tool("render_page",
          "把当前 PDF 的某一页渲染成图片返回（视觉观察）：what=translation 渲染当前译文页"
          "（盖掉原文、画上译文；无译文则回落原文页），what=source 渲染原文页。用于让模型直接「看」"
          "某页的版面/译文效果（如自检溢出/越线、看图注与图表）。",
          {"page": {"type": "integer", "description": "页号（0 起）"},
           "what": {"type": "string", "enum": ["source", "translation"],
                    "description": "渲染原文页还是当前译文页（默认 translation）"}},
          ["page"], CAT_READ, target="source", audience=("chat",)),
    _tool("set_block_text",
          "把某块的译文直接置为指定文本（数字/代码块会被拒绝；写的是受保护的译文层）。",
          {"index": {"type": "integer", "description": "扁平块索引（来自 read_page）"},
           "text": {"type": "string"}},
          ["index", "text"], CAT_CONTENT, audience=("chat",)),
    _tool("delete_block_text",
          "移除某块的 AI 编辑，恢复为未被覆盖的译文。",
          {"index": {"type": "integer", "description": "扁平块索引（来自 read_page）"}},
          ["index"], CAT_CONTENT, audience=("chat",)),
    _tool("self_check",
          "对当前已翻译的 PDF 跑**确定性质检**（只读、不重译、不改动）：残留/漏译/数字保真/表格完整性/版面五类。"
          "用户说“检查第N页的数字/有没有漏译/数字对不对/翻译得怎么样”时调用；page 不传则查全文，checks 可传子集（如只查数字 ['numbers']）。"
          "返回 {checks_requested, checks, issues, clean}，issues 是带 check 标签的问题清单，clean 为是否无问题。"
          "checks 只能传 layout/residual/missing/numbers/table 这五个英文名；传别的名字会作为 unknown_checks 问题返回（clean=false），"
          "不要凭猜测编检查名。",
          {"page": {"type": "integer", "description": "页号（0 起）；不传则审计全文"},
           "checks": {"type": "array", "items": {"type": "string"},
                      "description": "检查子集：layout/residual/missing/numbers/table（默认全部）"}},
          [], CAT_VERIFY, audience=("chat",)),
    _tool("retranslate",
          "**局部/定点重译**指定的块并写入受保护覆盖层（不用整篇重跑）：传 page 与（可选的扁平块）indices；"
          "indices 不传则重译整页所有可翻译块；返回 {count, indices, translated:{index:text}, failed:[...]}。"
          "failed 是重译失败而**保留原文**的块（数字/代码块被自动跳过，不属于失败），应如实转告用户。改完提醒用户用 re_export 生成最新译文。",
          {"page": {"type": "integer", "description": "页号（0 起）"},
           "indices": {"type": "array", "items": {"type": "integer"},
                       "description": "要重译的扁平块索引（来自 read_page）；不传则重译整页所有可翻译块"},
           "target_lang": {"type": "string", "description": "目标语言（默认当前设置的目标语言）"}},
          ["page"], CAT_CONTENT, audience=("chat",)),
    _tool("run_flow",
          "把用户的一句话要求**编译成一个自定义流程**并执行（路径 A 参数化，需模型在线）：如“自检第3到第8页只查数字和表格，不修改”→ 解析页范围/检查子集/"
          "是否只读；如“第5页数字错了自动改”→ 会**就地修正**审计发现的问题块并写回覆盖层。可传 name 把该流程**登记为命名流程**（本次会话内可复用）。"
          "默认只读审计；auto_fix=True 且重译通道可用时才写回覆盖层。"
          "注意：无可用模型时**不会降级**为规则解析，而是返回“需要模型在线”。",
          {"requirement": {"type": "string", "description": "用户的一句话要求（如“自检第3到第8页只查数字和表格，不修改”）"},
           "name": {"type": "string", "description": "可选：把该流程登记为命名流程（本次会话内可复用）"}},
          ["requirement"], CAT_CONTENT, audience=("chat",)),
    _tool("re_export",
          "用当前已加载 PDF 上一次的成功译文，重新导出（应用本次对话/标注里已有的修改；不重新翻译、秒级）。",
          {}, [], CAT_CONTENT, audience=("chat",)),
    _tool("run_plan",
          "把用户的一句话要求**分解成按顺序执行的若干任务**并依次实施（AI 自由组合，需模型在线）：可混合调用单个工具"
          "（读/改/核/设置）与标准流程（翻译一页/整篇/特殊页/自检/导出）。如“先检查第3-8页数字，再导出”。"
          "返回每步结果 {tier, name, params, ok}；某步失败会停止并说明是哪个任务。"
          "注意：无可用模型时**不会降级**为确定性单任务，而是返回“需要模型在线”。",
          {"requirement": {"type": "string", "description": "用户的一句话要求（会分解成任务序列）"}},
          ["requirement"], CAT_CONTENT, audience=("chat",)),
    _tool("run_translate",
          "用**当前设置**开始翻译（把用户的具体要求作为 requirement 传入，会随运行注入到 AI 编排层）。控制权交给翻译流水线，完成在主窗口日志/进度提示。",
          {"requirement": {"type": "string", "description": "用户的具体要求（可选，如\"第3页公司名翻成Bank\"），会随运行注入 AI 编排层"}},
          [], CAT_CONTENT, audience=("chat",)),
    _tool("set_setting",
          "修改翻译设置项（key 为 target_language 或 output_type，value 为语言名/输出格式键 translated_pdf|bilingual_pdf|markdown|plain_text），下次运行生效。",
          {"key": {"type": "string", "enum": ["target_language", "output_type"],
                   "description": "要改的设置项：target_language（目标语言名）或 output_type（输出格式键）"},
           "value": {"type": "string", "description": "语言名（如 French）；output_type 取 translated_pdf / bilingual_pdf / markdown / plain_text"}},
          ["key", "value"], CAT_CONTENT, audience=("chat",)),
]
