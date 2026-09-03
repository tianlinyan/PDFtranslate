"""Background translation worker.

Runs the extract → translate → export pipeline on a worker thread so the GUI
stays responsive.  Progress, log lines and the final outcome are reported back
to the main thread through Qt signals.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from . import pdfio
from .settings import ModelConfig
from .translator import (
    TranslationAborted,
    TranslationCancelled,
    TranslationEngine,
    make_classify_review_fn,
    make_classify_tool_fn,
    make_merge_tool_fn,
    make_rendered_review_fn,
    make_retranslate_fn,
    make_review_fn,
    make_table_rebuild_fn,
    make_verify_number_tool_fn,
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


def _contains_cjk(text: str) -> bool:
    """True when ``text`` still carries CJK ideographs (a residual Chinese leak)."""
    return any("一" <= c <= "鿿" for c in text)


#: Minimum confidence for the ``classify_block`` tool to release a rule-kept
#: block for translation (only a confident judgement overrides the rule).
_KEEP_CONF_MIN = 0.7


def _release_kept_blocks_via_tool(
    doc, keep_original: set[int], classify_fn, conf_min: float, log
) -> set[int]:
    """Release rule-kept chart-node blocks the tool confidently says to translate.

    The tool-use ``classify_block`` is given the *text* of each kept chart-node
    block (no page image needed); a block it judges ``translate`` at or above
    ``conf_min`` is removed from ``keep_original``.  Only releases (never keeps
    more), and a block not judged is left to the rule.
    """
    if classify_fn is None or not keep_original:
        return set()
    candidates: list[tuple[int, str]] = []
    flat = 0
    for page_blocks in doc.pages:
        for b in page_blocks:
            if flat in keep_original and getattr(b, "is_chart", False):
                candidates.append((flat, b.text))
            flat += 1
    if not candidates:
        return set()
    decisions = classify_fn([t for _i, t in candidates])
    released: set[int] = set()
    for d in decisions:
        try:
            idx = int(d.get("index", -1))
            conf = float(d.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(candidates)):
            continue
        if d.get("action") == "translate" and conf >= conf_min:
            flat_idx = candidates[idx][0]
            released.add(flat_idx)
            if log:
                log(
                    f"  [保留复核] 块「{candidates[idx][1][:16]}」判定为应翻译"
                    f"（@{conf:.2f}），已从保留集合释放。"
                )
    return released


def _correct_residual_blocks(
    doc,
    result,
    keep: set[int],
    target_is_cjk: bool,
    flagged: set[int],
    lang: str,
    retranslate,
    log,
) -> int:
    """Re-translate residual / empty blocks on QC-flagged pages.

    A block is "residual" when, for a non-CJK target, its translation still
    contains Chinese, or when it came back empty (missing content).  Only those
    are re-translated; kept blocks (names / chart nodes) are never touched, and
    a block that already translated cleanly is left as-is.  Returns the count
    corrected.  Best-effort — a re-translation failure keeps the original text.
    """
    corrected = 0
    for bi, page in enumerate(doc.block_pages):
        if page not in flagged:
            continue
        if bi in keep:
            continue  # intentionally kept (name column / chart node)
        txt = str(result.translated[bi])
        residual = (not txt.strip()) or (not target_is_cjk and _contains_cjk(txt))
        if not residual:
            continue
        new = retranslate(doc.blocks[bi], lang)
        new = str(new or "").strip()
        if new and new != txt:
            result.translated[bi] = new
            corrected += 1
            if log:
                log(f"    [修正] 块 {bi + 1} 已重译：{txt[:18]!r} -> {new[:18]!r}")
    return corrected


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
        render_qa: bool = True,
        ai_table_rebuild: bool = False,
    ):
        super().__init__()
        self._source = source_path
        self._model = model
        self._lang = target_language
        self._output_type = output_type
        self._output_path = output_path
        self._ocr = ocr
        # Rendered-output QA (report-only, read back the exported PDF).  Turned
        # off by the user via the "译文质检" checkbox; only a PDF output with a
        # vision-capable model actually runs it.
        self._render_qa = render_qa
        # AI table rebuild of scanned (OCR) statement pages: on, the vision model
        # counts rows/columns and redraws a clean, regular table (ignoring the
        # raster background / stamps / handwriting).  Falls back to the geometric
        # OCR-grid redraw if the model is unavailable or misreads the table.
        self._ai_table_rebuild = ai_table_rebuild
        # Cancellation flag.  An ``Event`` (not a bare bool) because it is
        # written from the GUI thread (``cancel``) and read from the worker
        # thread: the Event gives explicit, memory-model-safe signalling
        # instead of relying on the CPython GIL to make a bool atomic.
        self._cancelled = threading.Event()

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
            # Whole-page AI review: after an OCR page is rebuilt, the original
            # scan + the reconstruction are sent to the model, whose text fixes
            # are applied conservatively and whose layout flags are logged.  Only
            # for OCR'd (scanned) pages and only when the model opts in.
            review = make_review_fn(self._model, self.log.emit) if self._ocr else None
            if review is not None:
                self.log.emit(
                    "该模型支持视觉：将对重建的扫描页做整页审查（文字纠错 + 布局提示，不改几何）。"
                )
            # OCR-number verification (vision tool-use): re-read the plausible
            # misreads on a freshly OCR'd page and correct them in place, so the
            # figures feeding the translation are right.  Only for OCR'd pages and
            # only when the model opts in (make_* returns None otherwise).
            verify_fn = make_verify_number_tool_fn(self._model, self.log.emit) if self._ocr else None
            doc = pdfio.extract_document_text(
                self._source,
                ocr=self._ocr,
                cancel=lambda: self._cancelled.is_set(),
                log=lambda m: self.log.emit(m),
                review_fn=review,
                verify_fn=verify_fn,
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

            # Personal-name cells (a "姓名 / Name" table column) keep the
            # original text — but only for a CJK target: when the output is
            # Latin-script, leaving the Chinese names in place would produce a
            # mixed-language document, so they are translated instead (the
            # prompt rule romanizes them with consistent pinyin).
            #
            # Org-chart / architecture-diagram node labels are the opposite:
            # they are structural, not prose, so they are never translated
            # regardless of the target language (a diagram's box labels survive
            # a language switch; transcribing them would mangle the diagram).
            # The flat block order in ``doc.blocks`` matches ``doc.pages``, so
            # walk the pages to collect those indices.
            target_is_cjk = any("一" <= c <= "鿿" for c in self._lang)
            keep_original: set[int] = set()
            n_chart = 0
            n_names = 0
            flat = 0
            for page_blocks in doc.pages:
                for b in page_blocks:
                    if b.is_chart:
                        keep_original.add(flat)
                        n_chart += 1
                    elif b.keep_original:
                        n_names += 1
                        if target_is_cjk:
                            keep_original.add(flat)
                    flat += 1
            if n_chart:
                self.log.emit(
                    f"已识别 {n_chart} 个组织结构图/架构图节点，将保留原文不做翻译。"
                )
            if n_names and target_is_cjk:
                self.log.emit(
                    f"已识别 {n_names} 个姓名块，将保留原文不做翻译。"
                )
            elif n_names:
                self.log.emit(
                    f"目标语言为西文，{n_names} 个姓名块将按规则罗马化（不保留中文原文）。"
                )

            # Vision second opinion (P1c): the rule-based chart-node detection can
            # misjudge a compact heading (e.g. 二、公司组织架构图) as a diagram node
            # and keep it untranslated.  A vision model that sees the source page
            # may flag such a block as translatable content — this pass only
            # *releases* (never keeps more), and only on high confidence; any
            # classifier error is a no-op so the rule's decision stands.  The
            # tool-use ``classify_block`` (text-based) is preferred; the image-based
            # whole-page review is the fallback.
            if n_chart:
                released: set[int] = set()
                tool_classify = make_classify_tool_fn(self._model, self.log.emit)
                if tool_classify is not None:
                    released = _release_kept_blocks_via_tool(
                        doc, keep_original, tool_classify, _KEEP_CONF_MIN, self.log.emit
                    )
                else:
                    review_classify = make_classify_review_fn(self._model, self.log.emit)
                    released = pdfio.classify_keep_blocks(
                        self._source, doc.pages, review_classify,
                        keep_original, self.log.emit,
                    )
                if released:
                    keep_original -= released
                    n_chart -= len(released)
                    self.log.emit(
                        f"保留复核后，实际保留 {n_chart} 个图表节点（其余将翻译）。"
                    )

            translate_started = time.monotonic()
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

            per_page = pdfio.group_by_page(
                doc.block_pages, result.translated, doc.page_count
            )
            self.log.emit("正在生成输出文件…")
            self.progress.emit(len(doc.blocks), len(doc.blocks), "导出…")

            export_started = time.monotonic()
            out_path = self._export(doc, per_page)
            export_elapsed = time.monotonic() - export_started

            # Rendered-output QA: read back the exported PDF and surface anything
            # a reader would notice (untranslated text, overlaps, overflowing
            # glyphs, broken labels, too-small text).  A *correctable* report
            # (residual Chinese / missing content) triggers a bounded re-translate
            # of those cells; the QA may also return structured *adjustments* (a
            # replacement text / a font-size target), applied deterministically by
            # ``apply_render_adjustments``.  Either way we re-export once, bounded
            # (single pass).  Fail-closed: any reviewer error is a no-op.
            qa_elapsed = 0.0
            correct_elapsed = 0.0
            if self._render_qa and self._output_type in ("translated_pdf", "bilingual_pdf"):
                rendered_review = make_rendered_review_fn(self._model, self.log.emit)
                if rendered_review is not None:
                    qa_started = time.monotonic()
                    adjustments: list = []
                    flagged = pdfio.review_rendered_pages(
                        self._source, out_path, rendered_review,
                        self.log.emit, adjustments=adjustments,
                    )
                    qa_elapsed = time.monotonic() - qa_started
                    if (flagged or adjustments) and not self._cancelled.is_set():
                        changed = 0
                        retranslate = (
                            make_retranslate_fn(self._model, self.log.emit) if flagged else None
                        )
                        if retranslate is not None:
                            changed = _correct_residual_blocks(
                                doc, result, keep_original, target_is_cjk,
                                flagged, self._lang, retranslate, self.log.emit,
                            )
                        if adjustments:
                            changed += pdfio.apply_render_adjustments(
                                doc, result.translated, adjustments,
                                keep_original, self.log.emit,
                            )
                        if changed:
                            correct_started = time.monotonic()
                            per_page = pdfio.group_by_page(
                                doc.block_pages, result.translated, doc.page_count
                            )
                            out_path = self._export(doc, per_page)
                            correct_elapsed = time.monotonic() - correct_started
                            self.log.emit(
                                f"  质检修正：应用 {changed} 处修改并重新导出。"
                            )

            total_elapsed = time.monotonic() - started
            self.log.emit(f"完成：{out_path}")
            self.log.emit(
                f"总用时：{format_duration(total_elapsed)}"
                f"（提取 {format_duration(extract_elapsed)}，"
                f"翻译 {format_duration(translate_elapsed)}，"
                f"导出 {format_duration(export_elapsed)}"
                + (f"，渲染校验 {format_duration(qa_elapsed)}" if qa_elapsed else "")
                + (f"，质检修正 {format_duration(correct_elapsed)}" if correct_elapsed else "")
                + "）"
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

    def _export(self, doc: pdfio.DocumentText, per_page: list[list[str]]) -> str:
        out = Path(self._output_path)
        kind = self._output_type
        if kind == "bilingual_pdf":
            pdfio.save_interleaved_pdf(
                self._source, per_page, out, self._lang, doc.pages
            )
        elif kind == "translated_pdf":
            table_rebuild = (
                make_table_rebuild_fn(self._model, self._lang, self.log.emit)
                if self._ai_table_rebuild else None
            )
            merge_tool = (
                make_merge_tool_fn(self._model, self.log.emit) if self._ai_table_rebuild else None
            )
            pdfio.save_translated_pdf(
                self._source, doc.pages, per_page, out, self._lang,
                redraw_ocr=self._ai_table_rebuild,
                table_rebuild_fn=table_rebuild,
                merge_tool_fn=merge_tool,
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
