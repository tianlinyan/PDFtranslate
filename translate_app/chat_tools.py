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

from .agent.tool_catalog import catalog_for, to_openai_schema

#: Valid ``output_type`` values for ``set_setting`` (mirrors ``worker.OUTPUT_TYPES``).
_OUTPUT_TYPE_KEYS = frozenset({"translated_pdf", "bilingual_pdf", "markdown", "plain_text"})

#: The chat tool schemas (OpenAI ``tools`` array).  The chat model only ever sees
#: the ``"chat"``-audience subset of the single tool catalog (see
#: ``agent.tool_catalog``) -- chosen for the persistent context, no draw / page /
#: verify tools.
CHAT_TOOL_SPECS: list[dict[str, Any]] = [to_openai_schema(t) for t in catalog_for("chat")]
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
                    start_translate: Callable[[str, list | None], None] | None = None,
                    set_setting: Callable[[str, str], None] | None = None,
                    llm: Callable[[str], dict] | None = None,
                    plan_llm: Callable[[str], dict] | None = None,
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
        return {"page": page, "page_number": page + 1, "blocks": blocks}

    def goto_page(page: int, what: str = "source") -> dict[str, Any]:
        if show_preview is None:
            return {"ok": False, "error": "预览通道未接线"}
        try:
            show_preview(int(page), str(what or "source"))
        except Exception as exc:  # noqa: BLE001 — fail-closed
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "page": int(page), "what": what or "source"}

    def _chat_translation(offset: int, i: int) -> str:
        """The current translation of a flat block (overlay wins, else last run)."""
        entry = ctx.overlay().get(offset + i)
        if isinstance(entry, dict) and str(entry.get("text", "")).strip():
            return str(entry.get("text", ""))
        last = ctx.get_last_translated()
        if last is not None and (offset + i) < len(last):
            return str(last[offset + i])
        return ""

    def render_page(page: int, what: str = "translation") -> dict[str, Any]:
        """Render one page to a PNG (the chat's visual observation, no model image).

        ``what="translation"`` builds the in-place translated page (redact the source
        text the model has translated, draw the current ``overlay``/last-run text);
        ``what="source"`` renders the raw source.  Returns ``{ok, page, what, image}``
        where ``image`` is the PNG bytes (the chat loop re-injects it as ``image_url``).
        """
        import pymupdf as fitz

        from . import pdfio

        doc = ctx.ensure_doc()
        what = str(what or "translation").strip()
        if doc is None or not (0 <= int(page) < len(doc.pages)):
            return {"ok": False, "error": f"bad page {page}"}
        if not ctx.src_path:
            return {"ok": False, "error": "没有源文件。"}
        try:
            fdoc = fitz.open(str(ctx.src_path))
        except Exception as exc:  # noqa: BLE001 — fail-closed
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            if not (0 <= int(page) < fdoc.page_count):
                return {"ok": False, "error": "页号越界"}
            page_obj = fdoc[int(page)]
            p = doc.pages[int(page)]
            offset = sum(len(x) for x in doc.pages[:int(page)])
            if what != "source":
                to_draw = [
                    (b, _chat_translation(offset + i))
                    for i, b in enumerate(p)
                ]
                to_draw = [(b, t) for b, t in to_draw
                           if t and t != str(b.text)]  # only redraw genuinely translated
                if to_draw:
                    font = pdfio._CJK_FONT
                    for b, _t in to_draw:
                        if not getattr(b, "ocr", False) and not getattr(b, "is_chart", False):
                            page_obj.add_redact_annot(fitz.Rect(b.x0, b.y0, b.x1, b.y1))
                    page_obj.apply_redactions(
                        images=fitz.PDF_REDACT_IMAGE_NONE,
                        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                    )
                    for b, t in to_draw:
                        if getattr(b, "ocr", False):
                            page_obj.draw_rect(fitz.Rect(b.x0 - 0.5, b.y0 - 0.5,
                                                          b.x1 + 0.5, b.y1 + 0.5),
                                               color=None, fill=(1, 1, 1))
                        pdfio._draw_translated_block(page_obj, font, b, t)
            png = pdfio._render_page_png(page_obj, dpi=200)
        except Exception as exc:  # noqa: BLE001 — fail-closed
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            fdoc.close()
        return {"ok": True, "page": int(page), "what": what, "image": png}

    def set_block_text(index: int, text: str) -> dict[str, Any]:
        blocks = _flat_blocks()
        if not (0 <= int(index) < len(blocks)):
            return {"ok": False, "error": f"越界块索引 {index}"}
        src = str(blocks[int(index)].text)
        if pdfio._is_numeric_cell(src):
            return {"ok": False, "error": "数字格不可被 AI 改写（保真）"}
        # ``str(text)`` would turn a missing/None argument into the literal "None"
        # and write it into the protected overlay — which wins at export, so the
        # reader saw "None" instead of the translation.
        if not isinstance(text, str) or not text.strip():
            return {"ok": False, "error": "set_text 需要非空字符串 text"}
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
        if not isinstance(text, str) or not text.strip():
            return {"ok": False, "error": "action=set 需要非空字符串 text"}
        if pdfio._is_numeric_cell(str(block.text)):
            return {"ok": False, "error": "数字格不可被 AI 改写（保真）"}
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
        page_end = offset + len(doc.pages[int(page)])
        if indices is None:
            cands = list(range(offset, page_end))
        else:
            # A "重译第 N 页" request must never touch another page: the model
            # sometimes passes page-local indices, which used to be taken as flat
            # indices and silently rewrote blocks of a *different* page (while
            # still reporting ok=True).  Keep only the indices that belong to
            # ``page`` and say so when none do.
            cands = [int(i) for i in indices]
            in_page = [i for i in cands if offset <= i < page_end]
            if not in_page:
                return {
                    "ok": False,
                    "error": (f"indices {cands} 不属于第 {int(page) + 1} 页"
                              f"（该页的扁平索引范围 {offset}..{page_end - 1}）；"
                              "请用 read_page 返回的扁平索引。"),
                }
            cands = in_page
        picked = _pick_translatable(cands, blocks)
        if not picked:
            return {"ok": False, "error": "没有可重译的块（全部为数字格/公式/空块）"}
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

    def _audit_scope(scope, checks, auto_fix, lang):
        """Deterministic audit (+ optional in-place fix) over ``scope`` pages.

        Shared by ``run_flow`` (a self_check/ai_self_check base) and ``run_plan`` (an
        audit process task).  Returns the core result dict (pages_audited / clean /
        issue_count / remaining / fixed / mode / per_page / note).
        """
        blocks = _flat_block_list()
        can_fix = bool(auto_fix) and translate_texts is not None
        per_page: dict[str, dict[str, Any]] = {}
        all_clean = True
        n_issues = 0
        remaining = 0
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
            "pages_audited": len(scope),
            "clean": all_clean,
            "issue_count": n_issues,
            "remaining_issue_count": remaining,
            "fixed_blocks": fixed_total,
            "failed_blocks": failed_blocks,
            "mode": mode,
            "per_page": per_page,
            "note": note,
        }

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
        # U1: an **explicitly restricted** page range in the requirement (e.g.
        # "只翻第2-5页") becomes the run's page scope, so the console defines WHAT
        # to translate and the pipeline limits itself to those pages (None = the
        # whole document).  A requirement that merely *mentions* a page ("翻译整篇，
        # 第5页图表保留原文") must NOT narrow the run — that used to translate only
        # page 5 and report success while every other page stayed untranslated;
        # such a mention stays a per-page instruction the agent reads from
        # ``state.requirements``.
        from . import agent as _agent
        try:
            page_scope = _agent.parse_explicit_scope(str(requirement or ""))
        except Exception:  # noqa: BLE001 — a bad parse degrades to the whole document
            page_scope = None
        try:
            start_translate(str(requirement or ""), page_scope)
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

        Path A (rule-based ``compile_from_user``): the requirement fills base/scope/checks/
        auto_fix.  The compiled ``base`` dispatches the execution: translate / special-page
        flows start the pipeline (with the spec's page scope), export re-exports (no
        re-translate), and audit flows run the deterministic checks in place.
        """
        from . import agent

        state = _audit_state(ctx)
        if state is None:
            return {"ok": False, "error": "没有已加载的 PDF。"}
        # run_flow compiles a user's *free-text* requirement into a custom flow — that
        # interpretation is AI-driven.  It must NOT silently fall back to deterministic
        # keyword rules when no model is wired; refuse loudly instead (mirrors
        # ``run_plan``).  A user who just wants 重新导出 uses the ``re_export`` tool.
        if llm is None:
            return {"ok": False,
                    "error": "run_flow（自定义流程）需要模型在线；当前没有可用模型，无法解读要求。"
                             "请先在主窗口选择并配置模型（models.json）。"}
        spec = agent.compile_from_user(str(requirement or ""), default_base="self_check_page",
                                       llm=llm)
        # U1: 编译出的 base 决定"翻译 / 导出 / 审计"三种执行 —— 用同一套"定义流程"机制。
        if spec.base in ("translate_page", "translate_normal", "special_pages", "special_page"):
            # 翻译/特殊页：按 scope 启动翻译流水线（scope 缺省 = 整篇）。
            if start_translate is None:
                return {"ok": False, "base": spec.base, "error": "开始翻译通道未接线"}
            try:
                start_translate(str(requirement or ""), spec.scope)
            except Exception as exc:  # noqa: BLE001 — fail-closed
                return {"ok": False, "base": spec.base, "error": f"{type(exc).__name__}: {exc}"}
            return {"ok": True, "base": spec.base, "scope": spec.scope,
                    "message": "已触发翻译（按当前要求后台执行；完成会在主窗口日志/进度提示）。"}
        if spec.base == "export":
            # 导出/重新导出：复用 re_export（用上次译文 + 当前 overlay 重写，不重译）。
            if re_export is None:
                return {"ok": False, "base": "export", "error": "重新导出通道未接线"}
            try:
                re_export()
            except Exception as exc:  # noqa: BLE001 — fail-closed
                return {"ok": False, "base": "export", "error": f"{type(exc).__name__}: {exc}"}
            return {"ok": True, "base": "export", "message": "已触发重新导出（后台执行）。"}
        if spec.base not in ("self_check_page", "ai_self_check"):
            return {"ok": False, "base": spec.base,
                    "error": f"该流程类型（{spec.base}）暂不能在对话中直接执行。"}

        total = len(state.src_doc.pages)
        scope = spec.scope if spec.scope is not None else list(range(total))
        scope = [p for p in scope if 0 <= p < total]
        checks = spec.checks
        # ``auto_fix`` is OPT-IN (v0.5.24): a plain "自检第3到第8页" is a READ-ONLY
        # audit — the tool description and CLAUDE.md both say so, and silently
        # rewriting the translation (and the protected overlay, which then wins at
        # export) is exactly the kind of surprise the human-in-the-loop design
        # avoids.  A fix only happens when the user asked for one ("自动改/修一下")
        # and a translation channel is wired; otherwise it stays read-only.
        auto_fix = bool(spec.auto_fix) if spec.auto_fix is not None else False
        body = _audit_scope(scope, checks, auto_fix, _target_lang(None))

        promoted = False
        if name and str(name).strip():
            # Persistence is env-gated (``PDFTRANSLATE_FLOWS_DIR``): without it the
            # spec stays in memory this session; with it, it is also written to disk.
            agent.save_flow_spec(str(name).strip(), spec)
            promoted = True

        return {"ok": True, "base": spec.base, "checks": checks, "scope": scope,
                "auto_fix": auto_fix, "promoted": promoted, **body}

    #: Atomic chat tools a Path-B plan may call directly (read / write / verify /
    #: settings / re-export).  Deliberately excludes the orchestration tools
    #: (``run_translate`` / ``run_flow`` / ``run_plan``) so a plan never recurses.
    _ATOMIC: dict[str, Callable] = {
        "get_doc_info": get_doc_info,
        "get_settings": get_settings,
        "classify_page": classify_page,
        "get_structure": get_structure,
        "get_table": get_table,
        "read_page": read_page,
        "render_page": render_page,
        "goto_page": goto_page,
        "set_block_text": set_block_text,
        "delete_block_text": delete_block_text,
        "apply_annotation": apply_annotation,
        "retranslate": retranslate,
        "self_check": self_check,
        "set_setting": set_setting_tool,
        "re_export": _re_export,
    }

    def _validated_scope(scope):
        """Normalise a plan/flow ``scope`` coming from the model's JSON.

        Returns ``(scope, error)``: a list of ints clamped to the document, or an
        error string.  A string (``"3-8"``) used to reach the pipeline and make
        ``i in scope`` raise mid-run, while a list of strings silently matched no
        page — the run then "succeeded" without translating anything.
        """
        if scope is None:
            return None, None
        if not isinstance(scope, list):
            return None, "scope 必须是整数页码列表（0 起）"
        try:
            scope = [int(p) for p in scope]
        except (TypeError, ValueError):
            return None, "scope 必须是整数页码列表（0 起）"
        state = _audit_state(ctx)
        if state is not None:
            total = len(state.src_doc.pages)
            scope = [p for p in scope if 0 <= p < total]
            if not scope:
                return None, f"scope 中没有有效页码（共 {total} 页）"
        return scope, None

    def _dispatch_plan_task(task) -> dict:
        """Execute one path-B :class:`Task` (atomic tool or a process/composite flow)."""
        from . import agent as _agent

        # ``params`` is read by the ``run_translate`` atomic branch below, so it must be
        # bound before the first branch (a ``NameError``/``UnboundLocalError`` otherwise).
        params = dict(task.params or {})

        if task.tier == "atomic":
            if task.name == "run_translate":
                # A plan may legitimately start a translation mid-sequence.
                if start_translate is None:
                    return {"ok": False, "error": "开始翻译通道未接线"}
                scope, err = _validated_scope(params.get("scope"))
                if err:
                    return {"ok": False, "base": "run_translate", "error": err}
                try:
                    start_translate(params.get("requirement", ""), scope)
                except Exception as exc:  # noqa: BLE001 — fail-closed
                    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                return {"ok": True, "base": "run_translate", "scope": scope,
                        "message": "已触发翻译（后台执行）。"}
            if task.name in ("run_flow", "run_plan"):
                return {"ok": False,
                        "error": f"{task.name} 不能在计划内嵌套调用（计划本身就是这一层）。"}
            fn = _ATOMIC.get(task.name)
            if fn is None:
                return {"ok": False, "error": f"未知原子工具：{task.name}"}
            try:
                out = fn(**dict(task.params))
            except Exception as exc:  # noqa: BLE001 — fail-closed per task
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            return out if isinstance(out, dict) else {"ok": True, "result": out}

        name = task.name
        if name in ("translate_page", "translate_normal", "special_pages", "special_page",
                    "translate_doc", "preprocess"):
            if start_translate is None:
                return {"ok": False, "base": name, "error": "开始翻译通道未接线"}
            scope, err = _validated_scope(params.get("scope"))
            if err:
                return {"ok": False, "base": name, "error": err}
            try:
                start_translate(params.get("requirement", "") or "", scope)
            except Exception as exc:  # noqa: BLE001 — fail-closed
                return {"ok": False, "base": name, "error": f"{type(exc).__name__}: {exc}"}
            return {"ok": True, "base": name, "scope": scope,
                    "message": "已触发翻译（按当前要求后台执行，完成会在主窗口日志/进度提示）。"}
        if name == "export":
            if re_export is None:
                return {"ok": False, "base": "export", "error": "重新导出通道未接线"}
            try:
                re_export()
            except Exception as exc:  # noqa: BLE001 — fail-closed
                return {"ok": False, "base": "export", "error": f"{type(exc).__name__}: {exc}"}
            return {"ok": True, "base": "export", "message": "已触发重新导出（后台执行）。"}
        if name in ("self_check_page", "ai_self_check"):
            state = _audit_state(ctx)
            if state is None:
                return {"ok": False, "base": name, "error": "没有已加载的 PDF。"}
            total = len(state.src_doc.pages)
            scope = params.get("scope")
            scope = ([p for p in scope if 0 <= p < total] if isinstance(scope, list)
                     else list(range(total)))
            checks = params.get("checks")
            # OPT-IN, mirroring ``run_flow``: a plain "自检" is a READ-ONLY audit —
            # only an explicit ``auto_fix: true`` in the plan may rewrite the
            # protected overlay (the tool description promises exactly that).
            auto_fix = bool(params.get("auto_fix", False))
            body = _audit_scope(scope, checks, auto_fix, _target_lang(params.get("lang")))
            return {"ok": True, "base": name, **body}
        return {"ok": False, "base": name, "error": f"该流程（{name}）暂不能在对话中直接执行。"}

    def run_plan(requirement: str = ""):
        """Path B: decompose ``requirement`` into an ordered task plan and execute it.

        The AI compiles the requirement into a sequence of mixed-tier tasks (atomic
        tool / process flow / composite flow) and runs them in order, returning each
        step's ``{tier, name, params, ok}``.  A failing step stops the plan and names
        the failure (fail-closed).
        """
        from . import agent as _agent

        if not ctx.has_source():
            return {"ok": False, "error": "请先选择一个 PDF 源文件（点「打开 PDF…」或拖入窗口）。"}
        # Path B is the AI *free-composition* entry — it must not silently degrade to a
        # deterministic single task when no model is wired.  Refuse loudly instead.
        if plan_llm is None:
            return {"ok": False,
                    "error": "run_plan（AI 自由组合）需要模型在线；当前没有可用模型，无法自由分解。"
                             "请先在主窗口选择并配置模型（models.json）。"}
        plan = _agent.compile_plan(str(requirement or ""), llm=plan_llm)
        if not plan.tasks:
            return {"ok": False,
                    "error": "无法把该要求分解成任何可执行任务（可换个说法，或改用 run_flow/run_translate）。"}
        if log:
            log(f"  [计划] 分解为 {len(plan.tasks)} 个任务："
                + "; ".join(f"{t.tier}:{t.name}" for t in plan.tasks))
        res = _agent.run_plan(plan, dispatch=_dispatch_plan_task, log=log)
        # Lift a task's image (render_page / an annotated region) to the top level so
        # the chat loop re-injects it as a fresh visual observation for the model.
        for r in reversed(res.get("results", [])):
            img = r.get("image") if isinstance(r, dict) else None
            if isinstance(img, (bytes, bytearray)) and img:
                res["image"] = bytes(img)
                break
        res["note"] = plan.note
        return res

    return {
        "get_doc_info": get_doc_info,
        "get_settings": get_settings,
        "classify_page": classify_page,
        "get_structure": get_structure,
        "get_table": get_table,
        "read_page": read_page,
        "render_page": render_page,
        "goto_page": goto_page,
        "set_block_text": set_block_text,
        "delete_block_text": delete_block_text,
        "apply_annotation": apply_annotation,
        "self_check": self_check,
        "retranslate": retranslate,
        "run_flow": run_flow,
        "run_plan": run_plan,
        "re_export": _re_export,
        "run_translate": run_translate,
        "set_setting": set_setting_tool,
    }
