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

from .. import prompts

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


def _tool(name: str, properties: dict[str, dict], required: list[str],
          category: str, *, target: str = "output", destructive: bool = False,
          returns: str = "", extra: dict[str, Any] | None = None) -> ToolDef:
    # The tool's ``description`` is prompt text and is authored centrally in
    # ``translate_app.prompts`` (``AGENT_TOOL_DESCRIPTIONS``), not here.
    return ToolDef(
        name=name,
        description=prompts.AGENT_TOOL_DESCRIPTIONS[name],
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


#: The registry — ONLY the tools actually bound in ``run_page_visual`` and that have
#: a real effect on the output.  Dead declarations and tools the renderer can't
#: honour (font/align/move/create-block/page ops) are intentionally absent so the
#: model never sees a tool that silently does nothing.
AGENT_TOOLS: list[ToolDef] = [
    # --- A. 读取 / 观察（只读原文，绝不改写）-----------------------------------
    _tool("read_page",
          {"page": {"type": "integer"},
           "offset": {"type": "integer", "description": "起始块偏移，用于分页读取大页（默认 0）"},
           "limit": {"type": "integer", "description": "最多返回几块（默认返回该页全部；大页可借此分页）"}},
          ["page"], CAT_READ, target="source",
          returns="该页的文本块列表（每块的 index 为全文档扁平索引，可直接用于 set_text/translate_block；"
                  "含 bbox/字号/对齐/是否表格/是否图表节点；超大页会带 total/truncated 供分页）"),
    _tool("get_layout", {"page": {"type": "integer"}}, ["page"], CAT_READ, target="source",
          returns="{rows, cols, grid}"),
    _tool("get_doc_info", {}, [], CAT_READ, target="source",
          returns="{pages, title, language, text_pages, scan_pages, chart_pages, table_pages, block_count, kinds}"),
    _tool("classify_page", {"page": {"type": "integer"}}, ["page"], CAT_READ, target="source",
          returns="{kind}"),
    _tool("render_page",
          {"page": {"type": "integer", "description": "页号（0 起）"},
           "what": {"type": "string", "enum": ["source", "translation"],
                    "description": "是否渲染原文页，还是当前处理到该页的译文（默认 translation）"}},
          ["page"], CAT_READ, target="source",
          returns="渲染出的 PNG（模型的视觉观察对象；无译文时可渲染源页）"),
    # --- B. 内容（翻译/编辑/覆盖，写的是可编辑译文，绝不改原文）---------------
    _tool("translate_block",
          {"index": {"type": "integer"}, "text": {"type": "string"},
           "target_lang": {"type": "string"}},
          ["index", "text", "target_lang"], CAT_CONTENT,
          returns="翻译后的文本"),
    _tool("retranslate_block",
          {"text": {"type": "string"}, "target_lang": {"type": "string"}},
          ["text", "target_lang"], CAT_CONTENT,
          returns="重译后的文本"),
    _tool("set_text",
          {"page": {"type": "integer"}, "index": {"type": "integer"}, "text": {"type": "string"}},
          ["page", "index", "text"], CAT_CONTENT,
          returns="是否成功（布尔）"),
    _tool("apply_annotation",
          {"page": {"type": "integer"},
           "bbox": {"type": "array", "items": {"type": "number"},
                    "description": "标注框 [x0,y0,x1,y1]，PDF 点"},
           "text": {"type": "string", "description": "替换译文（action=set 时必填）"},
           "action": {"type": "string", "enum": ["set", "delete"], "default": "set"}},
          ["page", "bbox"], CAT_CONTENT,
          returns="是否成功及改写的块/文本"),
    _tool("apply_terminology",
          {"source": {"type": "string"}, "target": {"type": "string"}},
          ["source", "target"], CAT_CONTENT,
          returns="是否成功（术语并入本页翻译所用术语表）"),
    _tool("delete_block",
          {"page": {"type": "integer"}, "index": {"type": "integer"}},
          ["page", "index"], CAT_CONTENT,
          returns="是否成功（恢复为该块保留原文）"),
    # --- C. 校验 / 反馈 -------------------------------------------------------
    _tool("check_residual", {"page": {"type": "integer"}}, ["page"], CAT_VERIFY,
          returns="残留块列表 [{index, text}]"),
    _tool("check_missing", {"page": {"type": "integer"}}, ["page"], CAT_VERIFY,
          returns="缺失块索引列表"),
    # --- D. 交互 / 预览（供用户查看，不修改文档）-------------------------------
    _tool("preview_page",
          {"page": {"type": "integer", "description": "页号（0 起）"},
           "what": {"type": "string", "enum": ["source", "translation"],
                    "description": "显示原文页还是译文页，默认 translation"},
           "region": {"type": "array", "items": {"type": "number"},
                      "description": "可选：聚焦的区域 [x0,y0,x1,y1]"}},
          ["page"], CAT_UI,
          returns="是否成功（布尔；弹出非阻塞预览窗口）"),
    _tool("ask_user",
          {"question": {"type": "string"},
           "options": {"type": "array", "items": {"type": "string"},
                       "description": "可选候选答案"},
           "target": {"type": "string", "description": "回答存储键"}},
          ["question"], CAT_UI,
          returns="用户的回答"),
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
