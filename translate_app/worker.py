"""Background translation worker.

Runs the extract → translate → export pipeline on a worker thread so the GUI
stays responsive.  Progress, log lines and the final outcome are reported back
to the main thread through Qt signals.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from . import pdfio
from .settings import ModelConfig
from .translator import (
    TranslationAborted,
    TranslationCancelled,
    TranslationEngine,
    TranslationResult,
)

#: Output formats offered by the translation dialog.
OUTPUT_TYPES = {
    "bilingual_pdf": ("双语 PDF", ".pdf"),
    "translated_pdf": ("仅译文 PDF", ".pdf"),
    "markdown": ("Markdown 文档", ".md"),
    "plain_text": ("纯文本", ".txt"),
}


def format_duration(seconds: float) -> str:
    """把秒数格式化成中文可读时长，例如 ``12.3 秒`` / ``1 分 05 秒``。"""
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours} 小时 {minutes:02d} 分 {secs:02d} 秒"
    return f"{minutes} 分 {secs:02d} 秒"


class TranslateWorker(QObject):
    """Translates the text of a PDF and writes it to a file."""

    progress = pyqtSignal(int, int, str)   # done, total, stage
    log = pyqtSignal(str)
    finished = pyqtSignal(str)             # output path
    error = pyqtSignal(str)
    #: Emitted on *every* exit path (success, error, cancellation).  The GUI
    #: connects it to ``QThread.quit``: a cancelled run emits neither
    #: ``finished`` nor ``error``, so without this signal the worker thread's
    #: event loop would keep running forever and the window could never start
    #: another translation.
    stopped = pyqtSignal()

    def __init__(
        self,
        source_path: str,
        model: ModelConfig,
        target_language: str,
        output_type: str,
        output_path: str,
        ocr: bool = False,
        preview_handler=None,
        answer_handler=None,
        show_preview=None,
        agent_mode: bool = True,
        overlay: dict[int, dict] | None = None,
    ):
        super().__init__()
        self._source = source_path
        self._model = model
        self._lang = target_language
        self._output_type = output_type
        self._output_path = output_path
        self._ocr = ocr
        #: The persistent, protected translation overlay (from the interaction
        #: chat's ``DocContext``): flat block index → ``{"text": ...}``.  Applied on
        #: top of whatever the run produced, so a chat/AI edit always wins at export.
        self._overlay: dict[int, dict] = dict(overlay or {})
        #: The run's final aligned translation (flat index → text), after the overlay
        #: is applied.  ``MainWindow`` reads this after a successful run to update the
        #: chat's ``DocContext`` so the chat sees the real current translation.
        self._last_translated: list[str] | None = None
        # ``preview_handler`` (e.g. ``preview.PreviewBridge.get_region``) is the
        # worker↔GUI channel the v0.3.0 agent uses to show a page and receive a
        # user-framed region back (see ``agent.run_page_visual``).
        self.preview_handler = preview_handler
        # worker↔GUI channel for the agent's questions (see ``agent.run_page_visual``).
        self.answer_handler = answer_handler
        # worker↔GUI channel for a *non-blocking* preview show (M3 special-page
        # negotiation: show the page without waiting for a region).
        self._show_preview = show_preview
        # v0.3.0: when on (default) the worker drives translation through the AI
        # orchestration loop (``agent.run_page_visual``) instead of the batch
        # translation engine; a non-vision model falls back to the deterministic
        # baseline (fail-closed).
        self._agent_mode = agent_mode
        # Cancellation flag.  An ``Event`` (not a bare bool) because it is
        # written from the GUI thread (``cancel``) and read from the worker
        # thread: the Event gives explicit, memory-model-safe signalling
        # instead of relying on the CPython GIL to make a bool atomic.
        self._cancelled = threading.Event()
        #: The agent's ``WorkflowState`` while an AI-orchestrated run is active.
        #: Set by ``_run_agent`` and read (while the worker is blocked in a
        #: preview) to render an in-progress "translation" preview page.
        self._agent_state: Any = None

    @pyqtSlot()
    def run(self) -> None:
        started = time.monotonic()
        try:
            if self._cancelled.is_set():
                raise TranslationCancelled()

            self.log.emit(f"开始时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log.emit(f"正在提取文本：{self._source}")
            if self._ocr:
                self.log.emit("已启用 OCR（自动识别原文语言），将识别无文本层的扫描页。")
            self.progress.emit(0, 0, "提取文本…")
            doc = pdfio.extract_document_text(
                self._source,
                ocr=self._ocr,
                cancel=lambda: self._cancelled.is_set(),
                log=lambda m: self.log.emit(m),
            )
            if doc.ocr_count:
                self.log.emit(f"有 {doc.ocr_count} 个页面无文本层，已通过 OCR 提取。")
            extract_elapsed = time.monotonic() - started

            if not doc.blocks:
                self.error.emit("未从该 PDF 中提取到任何文本，无法翻译。")
                return

            self.log.emit(f"共提取 {len(doc.blocks)} 个文本块，{doc.page_count} 页。")

            engine = TranslationEngine(self._model)
            self.log.emit(f"模型：{self._model.name} ({self._model.model})")

            # v0.3.0: the hardcoded special-casing (org-chart / signature / name
            # cells) is removed — those are decided by the AI orchestrator at
            # runtime.  ``keep_original`` starts empty; the agent determines what
            # to keep.  In the deterministic fallback every block is translated.
            target_is_cjk = any("一" <= c <= "鿿" for c in self._lang)
            keep_original: set[int] = set()

            translate_started = time.monotonic()
            if self._agent_mode and getattr(self._model, "vision", False):
                self.log.emit("已启用 AI 编排：采用 agent 驱动的单页视觉闭环。")
                result = self._run_agent(doc, keep_original)
            else:
                # Deterministic fallback: the model cannot see the page (no vision),
                # so AI orchestration can't drive it.  v0.3.0 removed the old
                # hardcoded chart-node / signature / name-cell protections and hands
                # those decisions to the AI at runtime, so the fallback translates
                # everything — say so instead of silently matching the old behaviour.
                self.log.emit(
                    "已回退到确定性批次流水线（当前模型不支持视觉）。"
                    "注意：组织结构图节点/姓名列/扫描件手写签字将不再自动保留原文，"
                    "统一按文本翻译；如需保留请改用支持视觉的模型。"
                )
                result = engine.translate_blocks(
                    doc.blocks,
                    self._lang,
                    on_progress=lambda d, t: self.progress.emit(d, t, "翻译中…"),
                    log=lambda m: self.log.emit(m),
                    cancel=lambda: self._cancelled.is_set(),
                    doc_path=Path(self._source),
                    keep_original=keep_original,
                )
            translate_elapsed = time.monotonic() - translate_started

            if self._cancelled.is_set():
                raise TranslationCancelled()

            # Batches that failed every retry kept their source text.  Say so:
            # otherwise the run reports "完成" and the user has to notice on
            # their own that parts of the document were never translated.
            if result.errors:
                self.log.emit(
                    f"警告：{len(result.errors)} 个文本块翻译失败，已保留原文。"
                    "可稍后重新运行，已成功的部分会直接从缓存恢复。"
                )

            # Apply the protected chat/manual edits (the interaction chat's overlay)
            # on top of the run's output: a user-AI edit always wins at export.
            self._apply_overlay(result)
            # Remember the final aligned translation so the chat can read the real
            # current output (stops the AI re-editing already-translated blocks).
            self._last_translated = list(result.translated)

            per_page = pdfio.group_by_page(
                doc.block_pages, result.translated, doc.page_count
            )
            self.log.emit("正在生成输出文件…")
            self.progress.emit(len(doc.blocks), len(doc.blocks), "导出…")

            export_started = time.monotonic()
            out_path = self._export(doc, per_page)
            export_elapsed = time.monotonic() - export_started

            total_elapsed = time.monotonic() - started
            self.log.emit(f"完成：{out_path}")
            self.log.emit(
                f"总用时：{format_duration(total_elapsed)}"
                f"（提取 {format_duration(extract_elapsed)}，"
                f"翻译 {format_duration(translate_elapsed)}，"
                f"导出 {format_duration(export_elapsed)}）"
            )
            self.finished.emit(out_path)
        except TranslationCancelled:
            self.log.emit(f"已取消。用时 {format_duration(time.monotonic() - started)}")
        except TranslationAborted as exc:
            # A configuration error (bad key / unknown model): report it as a
            # failure instead of exporting a document that is just the source.
            self.error.emit(
                f"翻译已中止：{exc}"
                f"（用时 {format_duration(time.monotonic() - started)}）"
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            self.error.emit(
                f"翻译失败：{exc}"
                f"（用时 {format_duration(time.monotonic() - started)}）"
                f"\n\n{traceback.format_exc()}"
            )
        finally:
            # Always release the thread, whatever happened above.
            self.stopped.emit()

    def cancel(self) -> None:
        """Request cancellation (safe to call from the GUI thread)."""
        self._cancelled.set()

    def add_user_requirement(self, text: str) -> None:
        """Inject a sidebar free-text message into the running AI agent.

        The agent's ``observe()`` serialises ``WorkflowState.requirements`` into the
        observation handed to the LLM each step, so appending here makes the model
        see the user's requirement on its next tool decision.  Best-effort: if no
        agent run is active, the message is a no-op.  The chat text stays only in the
        sidebar — the main log gets a short operation status without echoing it.
        """
        state = self._agent_state
        if state is not None and text and text.strip():
            state.requirements.append(text.strip())
            self.log.emit(f"  [编排] 已注入一条用户要求（当前共 {len(state.requirements)} 条）。")

    def _apply_overlay(self, result: "TranslationResult") -> None:
        """Overwrite the run's output with the protected chat/manual edits.

        ``self._overlay`` maps a flat block index to ``{"text": ...}``; for each in
        range, the committed translation is replaced with the overlay text (a blank
        entry is skipped so it never wipes content to nothing).  Besides faithfully
        applying a user-AI edit, this is the mechanism that keeps a chat edit across
        runs.  On a cancelled or errored run the overlay is simply not applied (the
        run reports as usual).
        """
        if not self._overlay:
            return
        translated = list(result.translated)
        changed = 0
        for idx, entry in self._overlay.items():
            if isinstance(entry, dict) and 0 <= idx < len(translated):
                new = str(entry.get("text", "")).strip()
                if new and new != str(translated[idx]):
                    translated[idx] = new
                    changed += 1
        result.translated = translated
        if changed:
            self.log.emit(f"  已应用 {changed} 处 AI 对话/标注编辑（受保护覆盖）。")

    def set_current_page(self, page: int) -> None:
        """Update the running session's ``current_page`` (preview navigation)."""
        state = self._agent_state
        if state is not None:
            state.current_page = int(page)

    def render_translation(self, page: int) -> bytes | None:
        """Render an in-progress "translation" preview page for ``page``.

        During an AI-orchestrated run the worker is blocked in the preview round
        trip, so reading ``_agent_state`` here is safe.  Only blocks that already
        have a translation (in ``out_doc``) are redacted and redrawn with the
        translated text; the rest keep their source, so the preview shows exactly
        the work done so far.  Returns ``None`` when there is no agent run or the
        page has not translated anything (the caller falls back to the source page).
        """
        state = self._agent_state
        if state is None or state.src_doc is None:
            return None
        pages = state.src_doc.pages
        if not (0 <= page < len(pages)) or not pages[page]:
            return None
        offset = sum(len(p) for p in pages[:page])
        out = state.out_doc or {}
        to_draw = [
            (b, str(entry.get("text", "")).strip())
            for i, b in enumerate(pages[page])
            if isinstance(entry := out.get(offset + i), dict) and str(entry.get("text", "")).strip()
        ]
        if not to_draw:
            return None
        import pymupdf as fitz

        doc = fitz.open(str(self._source))
        try:
            if not (0 <= page < doc.page_count):
                return None
            page_obj = doc[page]
            font = pdfio._CJK_FONT
            # Redact the original text of the blocks we are about to replace (keep
            # images/line-art), then draw the translation on top — the same in-place
            # pattern the exporter uses (minus the table-layout expansion, which is a
            # refinement not needed for a preview).
            for b, _t in to_draw:
                if not getattr(b, "ocr", False) and not getattr(b, "is_chart", False):
                    page_obj.add_redact_annot(fitz.Rect(b.x0, b.y0, b.x1, b.y1))
            page_obj.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            )
            for b, t in to_draw:
                if getattr(b, "ocr", False):
                    page_obj.draw_rect(
                        fitz.Rect(b.x0 - 0.5, b.y0 - 0.5, b.x1 + 0.5, b.y1 + 0.5),
                        color=None, fill=(1, 1, 1),
                    )
                pdfio._draw_translated_block(page_obj, font, b, t)
            return pdfio._render_page_png(page_obj, dpi=200)
        finally:
            doc.close()

    def _run_agent(self, doc: pdfio.DocumentText, keep_original: set[int]):
        """v0.3.0: drive translation through the AI-orchestration loop per page.

        Each page is handed to :func:`agent.run_page_visual` (the model sees it,
        calls the deterministic tools and writes its choices to ``out_doc``); the
        resulting translations are collected back into a ``TranslationResult`` so
        the rest of the pipeline (group_by_page → export) is unchanged.  A page
        that fails is fail-closed to its source text.
        """
        from . import agent as agent_mod

        state = agent_mod.WorkflowState(src_path=self._source, lang=self._lang)
        state.src_doc = doc
        self._agent_state = state

        def _translate_page(st, page, model, *, task, **kw):
            # Resolve through ``agent_mod`` at call time so a test's mock of
            # ``agent_module.run_page_visual`` is honoured.
            return agent_mod.run_page_visual(st, page, model, task=task, **kw)

        try:
            agent_mod.DocumentSession(
                state, doc, self._model, log=self.log.emit,
                translate_page=_translate_page,
                progress=lambda d, t, s: self.progress.emit(d, t, s),
                cancel=self._cancelled.is_set,
                preview_handler=self.preview_handler,
                answer_handler=self.answer_handler,
                show_preview=self._show_preview,
                max_steps_per_page=24,
            ).run()
        finally:
            self._agent_state = None
        translated = list(doc.blocks)
        n_changed = 0
        for idx, entry in (state.out_doc or {}).items():
            if isinstance(entry, dict) and 0 <= idx < len(translated):
                new = str(entry.get("text", "")).strip()
                if new and new != str(translated[idx]):
                    translated[idx] = new
                    n_changed += 1
        kept = 0
        for i in keep_original:
            if 0 <= i < len(translated):
                translated[i] = str(doc.blocks[i])
                kept += 1
        if n_changed:
            self.log.emit(f"  AI 编排完成：翻译 {n_changed} 块，强制保留 {kept} 块。")
        else:
            self.log.emit("  AI 编排未产生翻译（模型可能无法视觉编排），输出将保留原文。")
        return TranslationResult(blocks=doc.blocks, translated=translated)

    def _export(self, doc: pdfio.DocumentText, per_page: list[list[str]]) -> str:
        out = Path(self._output_path)
        kind = self._output_type
        if kind == "bilingual_pdf":
            pdfio.save_interleaved_pdf(
                self._source, per_page, out, self._lang, doc.pages
            )
        elif kind == "translated_pdf":
            pdfio.save_translated_pdf(
                self._source, doc.pages, per_page, out, self._lang,
                log=self.log.emit,
            )
        elif kind == "markdown":
            pdfio.save_markdown(
                per_page, doc.blocks, doc.block_pages, out, self._lang, doc.title
            )
        elif kind == "plain_text":
            pdfio.save_plain_text(per_page, out)
        else:
            raise ValueError(f"Unknown output type: {kind}")
        return str(out)
