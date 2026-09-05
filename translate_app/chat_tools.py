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
    _tool("get_structure",
          {"page": {"type": "integer", "description": "页号（0 起）"}}, ["page"]),
    _tool("get_table",
          {"page": {"type": "integer", "description": "页号（0 起）"},
           "index": {"type": "integer", "description": "该页语义表格序号（默认 0）"}},
          ["page"]),
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
    _tool("self_check",
          {"page": {"type": "integer", "description": "页号（0 起）；不传则审计全文"},
           "checks": {"type": "array", "items": {"type": "string"},
                      "description": "检查子集：layout/residual/missing/numbers/table（默认全部）"}},
          []),
    _tool("retranslate",
          {"page": {"type": "integer", "description": "页号（0 起）"},
           "indices": {"type": "array", "items": {"type": "integer"},
                       "description": "要重译的扁平块索引（来自 read_page）；不传则重译整页所有可翻译块"},
           "target_lang": {"type": "string",
                           "description": "目标语言（默认当前设置的目标语言）"}},
          ["page"]),
    _tool("run_flow",
          {"requirement": {"type": "string",
                           "description": "用户的一句话要求（如“自检第3到第8页只查数字和表格，不修改”）"},
           "name": {"type": "string",
                    "description": "可选：把该流程登记为命名流程（本次会话内可复用）"}},
          ["requirement"]),
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


def _audit_state(ctx) -> Any:
    """Build a minimal ``WorkflowState`` over the persistent context for a deterministic audit.

    ``out_doc`` merges the last run's committed translation with the protected
    overlay (overlay wins), mirroring ``read_page`` — so the audit sees exactly the
    current translation the user has, not a stale one.  Returns ``None`` when no
    source document is available.
    """
    from . import agent

    doc = ctx.ensure_doc()
    if doc is None:
        return None
    lang = ctx.get_settings().get("target_language") or ctx.lang or "English"
    state = agent.WorkflowState(src_path=ctx.src_path or "", lang=lang)
    state.src_doc = doc
    out: dict[int, dict[str, Any]] = {}
    last = ctx.get_last_translated()
    if last is not None:
        for i, t in enumerate(last):
            if str(t).strip():
                out[i] = {"text": str(t)}
    for idx, entry in ctx.overlay().items():
        if isinstance(entry, dict) and str(entry.get("text", "")).strip():
            out[int(idx)] = {"text": str(entry.get("text", ""))}
    state.out_doc = out
    return state


def _run_audit(ctx, page=None, checks=None) -> dict[str, Any]:
    """Deterministic audit of ``ctx``'s current translation; returns a chat-ready report."""
    from . import agent

    state = _audit_state(ctx)
    if state is None:
        return {"ok": False, "error": "没有已加载的 PDF。"}
    report = agent.audit_page(state, page, checks)
    issues = report.get("issues", [])
    truncated = len(issues) > 120
    if truncated:
        report["issues"] = issues[:120]
    report["truncated"] = truncated
    return {"ok": True, **report}


def _finding_block_indices(issues: list[dict] | None) -> list[int]:
    """The flat block indices an audit report flags, for an in-place fix.

    ``residual``/``missing``/``numbers``/``layout`` issues carry a single ``index``;
    a ``table`` issue carries ``empty_cells`` and ``empty_text`` (each a list of
    indices).  Deduped and returned in report order.
    """
    out: list[int] = []
    seen: set[int] = set()
    for iss in issues or []:
        idx = iss.get("index")
        if idx is not None:
            i = int(idx)
            if i not in seen:
                seen.add(i)
                out.append(i)
        for key in ("empty_cells", "empty_text"):
            for v in (iss.get(key) or []):
                i = int(v)
                if i not in seen:
                    seen.add(i)
                    out.append(i)
    return out


def make_chat_tools(ctx, *, show_preview: Callable[[int, str], None] | None = None,
                    re_export: Callable[[], None] | None = None,
                    start_translate: Callable[[str], None] | None = None,
                    set_setting: Callable[[str, str], None] | None = None,
                    llm: Callable[[str], dict] | None = None,
                    log: Callable[[str], None] | None = None,
                    translate_texts: Callable[[list[str], str], list[str]] | None = None,
                    ) -> dict[str, Callable]:
    """Bind the chat tools to a live :class:`DocContext` (``ctx``).

    ``show_preview`` (optional, thread-safe) opens a preview page — the GUI wires it
    to the preview bridge so ``goto_page`` runs on the GUI thread.  ``re_export``
    (optional, thread-safe) re-exports the last translation with the current overlay
    edits — the GUI wires it to a signal so ``re_export`` runs on the GUI thread.
    ``translate_texts`` (optional) is the *translation-side* batch re-translation
    callback ``(texts, target_language) -> list[str]`` (aligned with ``texts``); it is
    what lets ``retranslate`` and a ``run_flow(...)`` with ``auto_fix`` actually
    re-translate problem blocks and write the result to the protected overlay.  When
    ``None``, those tools report "重译通道未接线" instead of crashing.  When ``None``
    reporters are wired, these report "通道未接线" instead of crashing.
    """
    from . import pdfio
    from . import translator as _tr

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
        structure = (doc.page_structure[page]
                     if doc.page_structure and page < len(doc.page_structure) else None)
        return {"kind": pdfio.classify_page(doc.pages[page], structure=structure)}

    def get_structure(page: int) -> dict[str, Any]:
        doc = ctx.ensure_doc()
        if doc is None or not (0 <= page < len(doc.page_structure)):
            return {"page": page, "parser": "", "elements": [], "tables": []}
        ps = doc.page_structure[page]
        tables = []
        for i in range(len(ps.tables)):
            t = pdfio.get_table(doc, page, i)
            if t is not None:
                tables.append(t)
        return {"page": page, "parser": ps.parser, "elements": ps.elements, "tables": tables}

    def get_table(page: int, index: int = 0) -> dict[str, Any] | None:
        doc = ctx.ensure_doc()
        if doc is None:
            return None
        return pdfio.get_table(doc, page, index)

    def read_page(page: int) -> dict[str, Any]:
        doc = ctx.ensure_doc()
        if doc is None or not (0 <= page < len(doc.pages)):
            return {"page": page, "error": "页号越界或无文档。", "blocks": []}
        offset = sum(len(p) for p in doc.pages[:page])
        overlay = ctx.overlay()
        last = ctx.get_last_translated()
        # B-④: per-block semantic kind/level when the doc carries a structure layer.
        kind_by_idx: dict[int, tuple[str, int]] = {}
        if doc.page_structure and page < len(doc.page_structure):
            for el in doc.page_structure[page].elements:
                for idx in el.get("block_indices", []):
                    kind_by_idx[int(idx)] = (str(el.get("kind", "text")),
                                             int(el.get("level", 0) or 0))
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
                "kind": kind_by_idx.get(idx, ("text", 0))[0],
                "level": kind_by_idx.get(idx, ("text", 0))[1],
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

    def _flat_block_list() -> list:
        doc = ctx.ensure_doc()
        if doc is None:
            return []
        return [b for page in doc.pages for b in page]

    def _target_lang(target_lang: str | None) -> str:
        if target_lang and str(target_lang).strip():
            return str(target_lang).strip()
        s = ctx.get_settings()
        return str(s.get("target_language") or ctx.lang or "English")

    def _pick_translatable(cands: list[int], blocks: list) -> list[int]:
        """Dedupe + keep in-range blocks that contain letters and are not a
        numeric/code cell — re-translation must never rewrite a financial figure."""
        picked: list[int] = []
        seen: set[int] = set()
        for raw in cands:
            idx = int(raw)
            if idx in seen or not (0 <= idx < len(blocks)):
                continue
            seen.add(idx)
            b = blocks[idx]
            if not _tr._needs_translation(str(b.text)) or pdfio._is_numeric_cell(str(b.text)):
                continue
            picked.append(idx)
        return picked

    def retranslate(page: int, indices: list[int] | None = None,
                    target_lang: str | None = None) -> dict[str, Any]:
        """Re-translate selected blocks on ``page`` (or every translatable one) and
        write the result to the protected overlay — the chat's "改一处就重译那一处",
        without a full agent run.  ``failed`` lists the blocks that kept their source
        because the re-translation call failed or returned no usable text, so the AI
        can tell the user exactly which blocks were not fixed.
        """
        if translate_texts is None:
            return {"ok": False, "error": "重译通道未接线"}
        doc = ctx.ensure_doc()
        if doc is None or not (0 <= int(page) < len(doc.pages)):
            return {"ok": False, "error": f"bad page {page}"}
        blocks = _flat_block_list()
        offset = sum(len(p) for p in doc.pages[: int(page)])
        if indices is None:
            cands = list(range(offset, offset + len(doc.pages[int(page)])))
        else:
            cands = [int(i) for i in indices]
        picked = _pick_translatable(cands, blocks)
        if not picked:
            return {"ok": False, "error": "没有可重译的块（全部为数字/代码/空块）"}
        lang = _target_lang(target_lang)
        sources = [str(blocks[i].text) for i in picked]
        try:
            out = translate_texts(sources, lang)
        except Exception as exc:  # noqa: BLE001 — fail-closed, keep every source
            if log:
                log(f"  对话重译失败：{type(exc).__name__}: {exc}")
            return {"ok": False, "error": f"重译调用失败：{type(exc).__name__}: {exc}",
                    "failed": picked}
        if not isinstance(out, (list, tuple)) or len(out) != len(sources):
            return {"ok": False, "error": "重译返回的块数与请求不符", "failed": picked}
        written = 0
        failed: list[int] = []
        for i, txt in zip(picked, out):
            new = str(txt or "").strip()
            src = str(blocks[i].text)
            if new and new != src:
                ctx.set_overlay(i, new, action="set")
                written += 1
            else:
                failed.append(i)
        translated = {
            str(i): str((ctx.get_overlay(i) or {}).get("text", "")) for i in picked
        }
        note = (f"重译完成：{written} 块已写入，{len(failed)} 块失败保留原文（failed={failed}）。"
                "改完想要最新译文请用 re_export。" if failed
                else "重译完成。改完想要最新译文请用 re_export。")
        return {"ok": True, "page": int(page), "count": written,
                "indices": picked, "translated": translated, "failed": failed,
                "note": note}

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

    def self_check(page=None, checks=None):
        """Deterministic QA over the current translation (read-only, no re-translate)."""
        return _run_audit(ctx, page, checks)

    def run_flow(requirement="", name=None):
        """Compile a natural-language requirement into a custom flow and run it over
        the current document; optionally promote it.  ``auto_fix`` (from the compiled
        spec) decides whether it is read-only audit (default when the channel is not
        wired) or an in-place fix plus re-audit.

        Path A (rule-based ``compile_from_user``): the requirement fills scope/checks/
        auto_fix.  Only audit-type flows run here; translate/export bases are redirected
        to the tools that can actually do them (``run_translate`` / ``re_export``).
        """
        from . import agent

        state = _audit_state(ctx)
        if state is None:
            return {"ok": False, "error": "没有已加载的 PDF。"}
        spec = agent.compile_from_user(str(requirement or ""), default_base="self_check_page",
                                       llm=llm)
        if spec.base == "export":
            return {"ok": False, "base": "export",
                    "error": "导出/重新导出请用 re_export 工具（或点主界面「重新导出」按钮）。"}
        if spec.base == "translate_page":
            return {"ok": False, "base": "translate_page",
                    "error": "重译需完整翻译运行：请用 run_translate 并在要求里写清要重译的页。"}
        if spec.base not in ("self_check_page", "ai_self_check"):
            return {"ok": False, "base": spec.base,
                    "error": f"该流程类型（{spec.base}）暂不能在对话中直接执行。"}

        total = len(state.src_doc.pages)
        scope = spec.scope if spec.scope is not None else list(range(total))
        scope = [p for p in scope if 0 <= p < total]
        checks = spec.checks
        # ``auto_fix`` is True unless the user said 只查/只读/不修改.  It can only
        # actually fix when a translation channel is wired; otherwise it stays read-only.
        auto_fix = bool(spec.auto_fix) if spec.auto_fix is not None else True
        can_fix = auto_fix and translate_texts is not None

        blocks = _flat_block_list()
        lang = _target_lang(None)

        per_page: dict[str, dict[str, Any]] = {}
        all_clean = True
        n_issues = 0          # issues found before any fix
        remaining = 0         # issues left after any fix
        fixed_total = 0
        failed_blocks: list[int] = []

        for p in scope:
            before = _run_audit(ctx, p, checks)
            pre = len(before.get("issues", []))
            n_issues += pre
            fixed_on_page = 0
            if can_fix and pre:
                idxs = _finding_block_indices(before.get("issues", []))
                picked = _pick_translatable(idxs, blocks)
                if picked:
                    sources = [str(blocks[i].text) for i in picked]
                    try:
                        out = translate_texts(sources, lang)
                    except Exception as exc:  # noqa: BLE001 — fail-closed
                        if log:
                            log(f"  流程修正确译失败：{type(exc).__name__}: {exc}")
                        out = None
                    if isinstance(out, (list, tuple)) and len(out) == len(sources):
                        for i, txt in zip(picked, out):
                            new = str(txt or "").strip()
                            prev = str((ctx.get_overlay(i) or {}).get("text", "")).strip()
                            if new and new != str(blocks[i].text):
                                ctx.set_overlay(i, new, action="set")
                                if new != prev:
                                    fixed_on_page += 1
                            else:
                                failed_blocks.append(i)
                    else:
                        failed_blocks.extend(picked)
            after = _run_audit(ctx, p, checks)
            after_issues = len(after.get("issues", []))
            remaining += after_issues
            per_page[str(p)] = {
                "clean": after["clean"],
                "issue_count": after_issues,
                "pre_fix_issues": pre,
                "fixed_block_count": fixed_on_page,
            }
            all_clean = all_clean and after["clean"]
            fixed_total += fixed_on_page

        promoted = False
        if name and str(name).strip():
            # Persistence is env-gated (``PDFTRANSLATE_FLOWS_DIR``): without it the
            # spec stays in memory this session; with it, it is also written to disk.
            agent.save_flow_spec(str(name).strip(), spec)
            promoted = True

        if can_fix:
            mode = "fixed"
            note = (f"就地修正完成：对 {len(scope)} 页审计，重译写入 {fixed_total} 块；"
                    f"{len(failed_blocks)} 块重译失败保留原文（failed={failed_blocks}）。"
                    f"修正后仍有 {remaining} 个问题。改完想要最新译文请用 re_export。")
        else:
            mode = "read_only"
            note = ("只读审计完成（未改动译文）。" if auto_fix else
                    "只读复核（按用户要求不修改译文）。")

        return {
            "ok": True,
            "base": spec.base,
            "checks": checks,
            "scope": scope,
            "pages_audited": len(scope),
            "clean": all_clean,
            "issue_count": n_issues,
            "remaining_issue_count": remaining,
            "fixed_blocks": fixed_total,
            "failed_blocks": failed_blocks,
            "mode": mode,
            "per_page": per_page,
            "auto_fix": auto_fix,
            "promoted": promoted,
            "note": note,
        }

    return {
        "get_doc_info": get_doc_info,
        "get_settings": get_settings,
        "classify_page": classify_page,
        "get_structure": get_structure,
        "get_table": get_table,
        "read_page": read_page,
        "goto_page": goto_page,
        "set_block_text": set_block_text,
        "delete_block_text": delete_block_text,
        "apply_annotation": apply_annotation,
        "self_check": self_check,
        "retranslate": retranslate,
        "run_flow": run_flow,
        "re_export": _re_export,
        "run_translate": run_translate,
        "set_setting": set_setting_tool,
    }
