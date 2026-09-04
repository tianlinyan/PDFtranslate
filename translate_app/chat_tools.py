"""Deterministic tool set for the interaction-chat AI (read + navigate + edit).

The free-text chat model (``chat.ChatSession``) uses these tools to act on the
persistent document context (:class:`translate_app.doc_context.DocContext`) when a
source PDF is loaded.  They are the chat side's "hands": read the document, drive
the preview, and edit the protected translation overlay.

Semantics mirror the translation-agent tools but operate on the **persistent**
context:

* source is read-only (``read_page`` / ``classify_page`` / ``get_doc_info``);
* edits go to the context's ``overlay`` (flat block index → ``{"text": ...}``), the
  protected layer that the pipeline applies on top of each run's output; and
* numeric / code blocks are never rewritten (financial figures stay faithful).

``make_chat_tools(ctx, show_preview=None)`` returns ``{name: callable(**args)}``;
``chat_openai_tools(names)`` returns the OpenAI ``tools`` schema for the same set.
"""

from __future__ import annotations

from typing import Any, Callable

from . import prompts

#: Valid ``output_type`` values for ``set_setting`` (mirrors ``worker.OUTPUT_TYPES``).
_OUTPUT_TYPE_KEYS = frozenset({"translated_pdf", "bilingual_pdf", "markdown", "plain_text"})


def _tool(name: str, properties: dict[str, dict], required: list[str]) -> dict[str, Any]:
    # The tool's ``description`` is prompt text, authored centrally in
    # ``translate_app.prompts`` (``CHAT_TOOL_DESCRIPTIONS``), not here.
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": prompts.CHAT_TOOL_DESCRIPTIONS[name],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


#: The chat tool schemas (OpenAI ``tools`` array).  The chat model only ever sees
#: these — a subset of the translation agent's registry, chosen for the persistent
#: context (no draw / page / verify tools yet).
CHAT_TOOL_SPECS: list[dict[str, Any]] = [
    _tool("get_doc_info", {}, []),
    _tool("get_settings", {}, []),
    _tool("classify_page",
          {"page": {"type": "integer", "description": "页号（0 起）"}}, ["page"]),
    _tool("read_page",
          {"page": {"type": "integer", "description": "页号（0 起）"}}, ["page"]),
    _tool("goto_page",
          {"page": {"type": "integer", "description": "页号（0 起）"},
           "what": {"type": "string", "enum": ["source", "translation"],
                    "description": "显示原文页还是译文页，默认 source"}},
          ["page"]),
    _tool("set_block_text",
          {"index": {"type": "integer", "description": "扁平块索引（来自 read_page）"},
           "text": {"type": "string"}},
          ["index", "text"]),
    _tool("delete_block_text",
          {"index": {"type": "integer", "description": "扁平块索引（来自 read_page）"}},
          ["index"]),
    _tool("apply_annotation",
          {"page": {"type": "integer"},
           "bbox": {"type": "array", "items": {"type": "number"},
                    "description": "标注框 [x0,y0,x1,y1]，PDF 点"},
           "text": {"type": "string", "description": "替换译文（action=set 时必填）"},
           "action": {"type": "string", "enum": ["set", "delete"], "default": "set"}},
          ["page", "bbox"]),
    _tool("re_export", {}, []),
    _tool("run_translate",
          {"requirement": {"type": "string",
                           "description": "用户的具体要求（可选，如\"第3页公司名翻成Bank\"），会随运行注入 AI 编排层"}},
          []),
    _tool("set_setting",
          {"key": {"type": "string", "enum": ["target_language", "output_type"],
                   "description": "要改的设置项：target_language（目标语言名）或 output_type（输出格式键）"},
           "value": {"type": "string",
                     "description": "语言名（如 French）；output_type 取 translated_pdf / bilingual_pdf / markdown / plain_text"}},
          ["key", "value"]),
]


def chat_openai_tools(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Return the OpenAI ``tools`` array (optionally filtered by ``names``)."""
    if names is None:
        return list(CHAT_TOOL_SPECS)
    return [t for t in CHAT_TOOL_SPECS if t["function"]["name"] in names]


def make_chat_tools(ctx, *, show_preview: Callable[[int, str], None] | None = None,
                    re_export: Callable[[], None] | None = None,
                    start_translate: Callable[[str], None] | None = None,
                    set_setting: Callable[[str, str], None] | None = None,
                    log: Callable[[str], None] | None = None) -> dict[str, Callable]:
    """Bind the chat tools to a live :class:`DocContext` (``ctx``).

    ``show_preview`` (optional, thread-safe) opens a preview page — the GUI wires it
    to the preview bridge so ``goto_page`` runs on the GUI thread.  ``re_export``
    (optional, thread-safe) re-exports the last translation with the current overlay
    edits — the GUI wires it to a signal so ``re_export`` runs on the GUI thread.
    When ``None``, these report "通道未接线" instead of crashing.
    """
    from . import pdfio

    def _flat_blocks() -> list:
        doc = ctx.ensure_doc()
        if doc is None:
            return []
        return [b for page in doc.pages for b in page]

    def _flat_of(block) -> int | None:
        for i, x in enumerate(_flat_blocks()):
            if x is block:
                return i
        return None

    def get_doc_info() -> dict[str, Any]:
        doc = ctx.ensure_doc()
        if doc is None:
            return {"error": "没有已加载的 PDF。"}
        return pdfio.get_doc_info(doc)

    def get_settings() -> dict[str, Any]:
        """The current run settings snapshot (source / language / output / model)."""
        s = ctx.get_settings()
        if not s:
            return {"ok": False, "error": "尚未初始化翻译设置。", "settings": {}}
        return {"ok": True, **s}

    def classify_page(page: int) -> dict[str, Any]:
        doc = ctx.ensure_doc()
        if doc is None or not (0 <= page < len(doc.pages)):
            return {"error": "页号越界或无文档。"}
        return {"kind": pdfio.classify_page(doc.pages[page])}

    def read_page(page: int) -> dict[str, Any]:
        doc = ctx.ensure_doc()
        if doc is None or not (0 <= page < len(doc.pages)):
            return {"page": page, "error": "页号越界或无文档。", "blocks": []}
        offset = sum(len(p) for p in doc.pages[:page])
        overlay = ctx.overlay()
        last = ctx.get_last_translated()
        blocks = []
        for i, b in enumerate(doc.pages[page]):
            idx = offset + i
            entry = overlay.get(idx)
            translated = str(entry.get("text", "")) if isinstance(entry, dict) else ""
            if not translated and last is not None and idx < len(last):
                translated = str(last[idx])
            blocks.append({
                "index": idx,
                "source": str(b.text),
                "translated": translated,
                "bbox": [b.x0, b.y0, b.x1, b.y1],
                "in_table": bool(getattr(b, "in_table", False)),
                "is_chart": bool(getattr(b, "is_chart", False)),
            })
        return {"page": page, "blocks": blocks}

    def goto_page(page: int, what: str = "source") -> dict[str, Any]:
        if show_preview is None:
            return {"ok": False, "error": "预览通道未接线"}
        try:
            show_preview(int(page), str(what or "source"))
        except Exception as exc:  # noqa: BLE001 — fail-closed
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "page": int(page), "what": what or "source"}

    def set_block_text(index: int, text: str) -> dict[str, Any]:
        blocks = _flat_blocks()
        if not (0 <= int(index) < len(blocks)):
            return {"ok": False, "error": f"越界块索引 {index}"}
        src = str(blocks[int(index)].text)
        if pdfio._is_numeric_cell(src):
            return {"ok": False, "error": "数字/代码块不可被 AI 改写（保真）"}
        ctx.set_overlay(int(index), str(text), action="set")
        return {"ok": True, "index": int(index), "text": str(text)}

    def delete_block_text(index: int) -> dict[str, Any]:
        if ctx.get_overlay(int(index)) is None:
            return {"ok": False, "error": f"index {index} 没有 AI 编辑"}
        ctx.set_overlay(int(index), None, action="delete")
        return {"ok": True, "index": int(index)}

    def apply_annotation(page: int, bbox, text: str | None = None, action: str = "set") -> dict[str, Any]:
        doc = ctx.ensure_doc()
        if doc is None or not (0 <= page < len(doc.pages)):
            return {"ok": False, "error": f"bad page {page}"}
        block = pdfio.nearest_block(doc.pages[page], bbox)
        if block is None:
            return {"ok": False, "error": "标注区域未命中任何块"}
        flat = _flat_of(block)
        if flat is None:
            return {"ok": False, "error": "块定位失败"}
        action = (action or "set").strip().lower()
        if action in ("delete", "void"):
            ctx.set_overlay(flat, None, action="delete")
            return {"ok": True, "page": page, "index": flat, "action": action}
        if text is None:
            return {"ok": False, "error": "action=set 需要 text"}
        if pdfio._is_numeric_cell(str(block.text)):
            return {"ok": False, "error": "数字/代码块不可被 AI 改写（保真）"}
        ctx.set_overlay(flat, str(text), action="set")
        return {"ok": True, "page": page, "index": flat, "action": action,
                "text": str(text)}

    def _re_export():
        """Re-write the exported output with the current overlay edits, no re-translate."""
        if re_export is None:
            return {"ok": False, "error": "重新导出通道未接线"}
        try:
            re_export()
        except Exception as exc:  # noqa: BLE001 — fail-closed
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "message": "已触发重新导出（后台执行，完成会在主窗口日志/进度提示）。"}

    def run_translate(requirement: str = ""):
        """Start the translation pipeline with the current settings (the AI entry).

        A missing source is a hard precondition (the chat worker is only handed tools
        when a PDF is loaded, but the model may still call this); fail-closed clearly
        instead of reporting success while nothing starts.
        """
        if not ctx.has_source():
            return {"ok": False, "error": "请先选择一个 PDF 源文件（点「打开 PDF…」或拖入窗口）。"}
        if start_translate is None:
            return {"ok": False, "error": "开始翻译通道未接线"}
        try:
            start_translate(str(requirement or ""))
        except Exception as exc:  # noqa: BLE001 — fail-closed
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "message": "已触发开始翻译（按当前设置后台执行；完成会在主窗口日志/进度提示）。"}

    def set_setting_tool(key: str, value: str):
        """Change a run setting (target_language / output_type) before starting."""
        if set_setting is None:
            return {"ok": False, "error": "设置通道未接线"}
        k = str(key or "").strip()
        if k not in ("target_language", "output_type"):
            return {"ok": False, "error": f"未知设置项：{k!r}"}
        v = str(value or "").strip()
        if k == "output_type" and v not in _OUTPUT_TYPE_KEYS:
            return {"ok": False, "error": f"未知输出格式：{v!r}（可选 " + " / ".join(sorted(_OUTPUT_TYPE_KEYS)) + "）"}
        try:
            set_setting(k, v)
        except Exception as exc:  # noqa: BLE001 — fail-closed
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "key": k, "value": str(value or "")}

    return {
        "get_doc_info": get_doc_info,
        "get_settings": get_settings,
        "classify_page": classify_page,
        "read_page": read_page,
        "goto_page": goto_page,
        "set_block_text": set_block_text,
        "delete_block_text": delete_block_text,
        "apply_annotation": apply_annotation,
        "re_export": _re_export,
        "run_translate": run_translate,
        "set_setting": set_setting_tool,
    }
