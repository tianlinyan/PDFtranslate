"""Deterministic tool layer for the v0.3.0 AI-orchestration flow (P1).

Each tool is the AI's "hand" — a named, typed primitive with an OpenAI-compatible
``parameters`` schema.  The ``FlowAgent`` controller (P2+) decides *what/where/
when*; these tools do *how*, deterministically.  This module defines the registry
and the schemas; implementations that already exist (the judgment tools in
``translator``) are bound by :func:`make_agent_tools`.  Pure ``pdfio`` ops are
declared here as schemas and wired to their deterministic implementations in P2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .. import translator as _tr

#: Tool categories (grouping for the controller's prompt / the UI).
CAT_READ = "read"
CAT_CONTENT = "content"
CAT_DRAW = "draw"
CAT_ERASE = "erase"
CAT_VERIFY = "verify"
CAT_STRUCTURE = "structure"
CAT_UI = "ui"   # interaction / presentation (e.g. pop a preview window)


@dataclass(frozen=True)
class ToolDef:
    """One tool's schema + metadata.

    ``parameters`` is the JSON Schema for the arguments the model may pass.
    ``target`` says which side the tool acts on: ``"source"`` (the immutable
    original — these are read/observe only and never write) or ``"output"`` (the
    mutable translation — freely editable).  **The source is never written**: a
    tool's implementation must only ever mutate the output model, never the
    original (hard invariant — see ``docs/0.3.0-设计.md``).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    category: str
    target: str = "output"
    destructive: bool = False
    returns: str = ""


def _tool(name: str, description: str, properties: dict[str, dict], required: list[str],
          category: str, *, target: str = "output", destructive: bool = False,
          returns: str = "", extra: dict[str, Any] | None = None) -> ToolDef:
    return ToolDef(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
            **(extra or {}),
        },
        category=category,
        target=target,
        destructive=destructive,
        returns=returns,
    )


def _from_schema(name: str, category: str, *, target: str = "output",
                 destructive: bool = False, returns: str = "") -> ToolDef:
    """Build a ``ToolDef`` from an existing ``translator`` module-level tool schema."""
    # The existing schemas live as ``{"type":"function","function":{...}}``.
    fn = getattr(_tr, name).get("function", {})
    return ToolDef(
        name=str(fn.get("name", name)),
        description=str(fn.get("description", "")),
        parameters=dict(fn.get("parameters", {})),
        category=category,
        target=target,
        destructive=destructive,
        returns=returns,
    )


#: The registry.  Ordered so ``classify_block`` / ``detect_table_merge`` /
#: ``verify_number`` carry the schemas already validated in ``translator``.
AGENT_TOOLS: list[ToolDef] = [
    # --- A. 读取 / 观察（只读原文，绝不改写）-----------------------------------
    _tool("read_page", "读取指定页的文本块与布局元数据", {"page": {"type": "integer"}},
          ["page"], CAT_READ, target="source",
          returns="该页的文本块列表（含 bbox/字号/对齐/是否表格/是否图表节点）"),
    _tool("render_region", "把某页的指定区域渲染成 PNG 供视觉观察",
          {"page": {"type": "integer"}, "bbox": {"type": "array", "items": {"type": "number"}},
           "dpi": {"type": "integer", "description": "渲染分辨率，默认 150"}},
          ["page"], CAT_READ, target="source",
          returns="PNG 图像（视觉模型的观察对象）"),
    _tool("get_layout", "提取某页的行/列聚类、二维网格与单元格跨度",
          {"page": {"type": "integer"}}, ["page"], CAT_READ, target="source",
          returns="{rows, cols, grid}"),
    _from_schema("_MERGE_TOOL", CAT_STRUCTURE, target="source", returns="需要跨列/跨行合并的单元格列表"),
    # --- B. 内容（编辑/修改/生成）--------------------------------------------
    _tool("translate_block", "把某一块文本翻译成目标语言（走缓存+编号协议）",
          {"index": {"type": "integer"}, "text": {"type": "string"},
           "target_lang": {"type": "string"}},
          ["index", "text", "target_lang"], CAT_CONTENT,
          returns="翻译后的文本"),
    _tool("retranslate_block", "强制重译某块（绕过缓存，常用于残中/空缺修正）",
          {"text": {"type": "string"}, "target_lang": {"type": "string"}},
          ["text", "target_lang"], CAT_CONTENT,
          returns="重译后的文本"),
    _tool("rewrite_block", "改写某块译文的措辞（面向用户预览）",
          {"index": {"type": "integer"}, "text": {"type": "string"},
           "instruction": {"type": "string", "description": "改写要求，如'更正式/更简洁'"}},
          ["index", "text"], CAT_CONTENT,
          returns="改写后的文本"),
    _tool("set_text", "把某块文本直接置为指定值（数字/代码块会被拒绝）",
          {"page": {"type": "integer"}, "index": {"type": "integer"}, "text": {"type": "string"}},
          ["page", "index", "text"], CAT_CONTENT,
          returns="是否成功（布尔）"),
    _tool("apply_terminology", "为某源词设定统一的术语译文（后续按术语翻译）",
          {"source": {"type": "string"}, "target": {"type": "string"}},
          ["source", "target"], CAT_CONTENT,
          returns="是否成功（布尔）"),
    _tool("delete_block", "删除译文中某一块（输出侧，自由；原文不受影响）",
          {"page": {"type": "integer"}, "index": {"type": "integer"}},
          ["page", "index"], CAT_CONTENT,
          returns="是否成功（布尔）"),
    _tool("create_block", "在译文某位置创建一块文本（自由；不修改原文）",
          {"page": {"type": "integer"}, "index": {"type": "integer"},
           "text": {"type": "string"},
           "bbox": {"type": "array", "items": {"type": "number"},
                    "description": "可选：新块位置 [x0,y0,x1,y1]"}},
          ["page", "index", "text"], CAT_CONTENT,
          returns="是否成功（布尔）"),
    _tool("move_block", "把译文块移到新位置/重排（自由）",
          {"page": {"type": "integer"}, "index": {"type": "integer"},
           "to_index": {"type": "integer"}},
          ["page", "index", "to_index"], CAT_CONTENT,
          returns="是否成功（布尔）"),
    # --- C. 绘制 / 布局 -------------------------------------------------------
    _tool("draw_table", "绘制一张干净 N×M 表格（含表头合并）",
          {"page": {"type": "integer"}, "rows": {"type": "array", "items": {"type": "array"}},
           "bbox": {"type": "array", "items": {"type": "number"}},
           "merges": {"type": "array"}},
          ["page", "rows"], CAT_DRAW,
          returns="是否成功（布尔）"),
    _tool("draw_block", "在指定块的位置绘制其译文",
          {"page": {"type": "integer"}, "index": {"type": "integer"}}, ["page", "index"],
          CAT_DRAW, returns="是否成功（布尔）"),
    _tool("set_font", "设置某块的渲染字号（夹逼到可读下限/上限）",
          {"page": {"type": "integer"}, "index": {"type": "integer"}, "size": {"type": "number"}},
          ["page", "index", "size"], CAT_DRAW,
          returns="实际生效的字号（可能被夹逼）"),
    _tool("set_align", "设置某块的水平对齐方式",
          {"page": {"type": "integer"}, "index": {"type": "integer"},
           "align": {"type": "string", "enum": ["left", "center", "right"]}},
          ["page", "index", "align"], CAT_DRAW,
          returns="实际对齐方式"),
    _tool("merge_cells", "把表格某格跨列/跨行合并",
          {"page": {"type": "integer"}, "row": {"type": "integer"},
           "col": {"type": "integer"}, "rowspan": {"type": "integer"},
           "colspan": {"type": "integer"}},
          ["page", "row", "col", "rowspan", "colspan"], CAT_STRUCTURE,
          returns="是否成功（布尔）"),
    _tool("grid_rule", "画一条表格线/分隔线",
          {"page": {"type": "integer"}, "x0": {"type": "number"}, "y0": {"type": "number"},
           "x1": {"type": "number"}, "y1": {"type": "number"}},
          ["page", "x0", "y0", "x1", "y1"], CAT_DRAW,
          returns="是否成功（布尔）"),
    # --- D. 擦除 / 覆盖（作用于**译文/输出**侧；原文受保护，绝不影响原文）----
    _tool("cover_region", "用不透明白块覆盖输出页某区域（盖扫描字/杂讯）",
          {"page": {"type": "integer"}, "bbox": {"type": "array", "items": {"type": "number"}}},
          ["page", "bbox"], CAT_ERASE,
          returns="是否成功（布尔）"),
    _tool("erase_text_layer", "删除**输出页**某区域的文本层（保留图片/线条）",
          {"page": {"type": "integer"}, "bbox": {"type": "array", "items": {"type": "number"}}},
          ["page", "bbox"], CAT_ERASE,
          returns="是否成功（布尔）"),
    _tool("drop_element", "移除**输出页**上的某元素（签字区/水印/照片）",
          {"page": {"type": "integer"}, "index": {"type": "integer"}},
          ["page", "index"], CAT_ERASE,
          returns="是否成功（布尔）"),
    # --- E. 校验 / 反馈 -------------------------------------------------------
    _from_schema("_CLASSIFY_TOOL", CAT_STRUCTURE, target="source",
                 returns="块的保留/翻译/签字判定"),
    _from_schema("_VERIFY_NUMBER_TOOL", CAT_VERIFY, target="source",
                 returns="数字与原件是否一致及修正值"),
    _tool("check_residual", "检查某页是否有残留中文/漏译的块",
          {"page": {"type": "integer"}}, ["page"], CAT_VERIFY,
          returns="残留块列表 [{index, text}]"),
    _tool("check_missing", "检查某页是否有源有译文空的块（内容缺失）",
          {"page": {"type": "integer"}}, ["page"], CAT_VERIFY,
          returns="缺失块索引列表"),
    _tool("qa_render", "读回成品页找渲染问题（溢出/越线/压叠/字过小）",
          {"page": {"type": "integer"}}, ["page"], CAT_VERIFY, target="source",
          returns="问题列表 [{kind, message, confidence}]"),
    _tool("audit", "汇总全文档差异报告（数字/术语/编号/残留）",
          {}, [], CAT_VERIFY,
          returns="审计报告"),
    # --- F. 交互 / 预览（供用户查看，不修改文档）-------------------------------
    _tool("preview_page", "在预览窗口显示指定页面（供用户查看），可聚焦某区域/块",
          {"page": {"type": "integer", "description": "页号（0 起）"},
           "what": {"type": "string", "enum": ["source", "translation"],
                    "description": "显示原文页还是译文页，默认 translation"},
           "region": {"type": "array", "items": {"type": "number"},
                      "description": "可选：聚焦的区域 [x0,y0,x1,y1]"}},
          ["page"], CAT_UI,
          returns="是否成功（布尔；弹出非阻塞预览窗口）"),
    _tool("ask_user", "向用户提问并等待回答（关键决策/歧义/术语确认）",
          {"question": {"type": "string"},
           "options": {"type": "array", "items": {"type": "string"},
                       "description": "可选候选答案"},
           "target": {"type": "string", "description": "回答存储键"}},
          ["question"], CAT_UI,
          returns="用户的回答"),
    # --- G. 译文页操作（输出侧，自由；原文页永不删除/改写）--------------------
    _tool("delete_page", "删除译文中某一页（输出侧，自由；源页不受影响）",
          {"page": {"type": "integer"}}, ["page"], CAT_STRUCTURE,
          returns="是否成功（布尔）"),
    _tool("create_page", "在译文某处创建/插入一页（自由）",
          {"at": {"type": "integer", "description": "插入位置（页号）"},
           "template": {"type": "string", "enum": ["blank", "copy_prev"],
                        "description": "新建页为空白还是复制上一页"}},
          ["at"], CAT_STRUCTURE,
          returns="新页的页号"),
    _tool("move_page", "把译文页移动到新位置（重排页序，自由）",
          {"from": {"type": "integer"}, "to": {"type": "integer"}},
          ["from", "to"], CAT_STRUCTURE,
          returns="是否成功（布尔）"),
]


#: Helper to publish a clean OpenAI ``tools=[...]`` list for a chat-completions call.
def agent_openai_tools(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Return the OpenAI ``tools`` array for the registry (optionally filtered)."""
    out: list[dict[str, Any]] = []
    for t in AGENT_TOOLS:
        if names is not None and t.name not in names:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        })
    return out


def by_name(name: str) -> ToolDef | None:
    for t in AGENT_TOOLS:
        if t.name == name:
            return t
    return None


def make_agent_tools(model, log: Callable[[str], None] | None = None) -> dict[str, Callable]:
    """Bind the existing judgment-tool factories into runnable callables (P1).

    Returns ``{name: callable}`` for the AI tools that already have a
    ``translator`` factory (``classify_block`` / ``detect_table_merge`` /
    ``verify_number`` / ``retranslate_block``).  A non-vision model yields an
    empty dict (the controller then falls back to the deterministic pipeline).
    Pure ``pdfio`` ops are wired in P2.
    """
    if model is None or not getattr(model, "vision", False):
        return {}
    tools: dict[str, Callable] = {}
    classify = _tr.make_classify_tool_fn(model, log)
    if classify is not None:
        tools.setdefault("classify_block", classify)
    merge = _tr.make_merge_tool_fn(model, log)
    if merge is not None:
        tools.setdefault("detect_table_merge", merge)
    verify = _tr.make_verify_number_tool_fn(model, log)
    if verify is not None:
        tools.setdefault("verify_number", verify)
    retranslate = _tr.make_retranslate_fn(model, log)
    if retranslate is not None:
        tools.setdefault("retranslate_block", retranslate)
    return tools
