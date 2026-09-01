"""Background translation worker.

Runs the extract → translate → export pipeline on a worker thread so the GUI
stays responsive.  Progress, log lines and the final outcome are reported back
to the main thread through Qt signals.
"""

from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from . import pdfio
from .settings import ModelConfig
from .translator import TranslationCancelled, TranslationEngine

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

    def __init__(
        self,
        source_path: str,
        model: ModelConfig,
        target_language: str,
        output_type: str,
        output_path: str,
        ocr: bool = False,
    ):
        super().__init__()
        self._source = source_path
        self._model = model
        self._lang = target_language
        self._output_type = output_type
        self._output_path = output_path
        self._ocr = ocr
        self._cancelled = False

    @pyqtSlot()
    def run(self) -> None:
        started = time.monotonic()
        try:
            if self._cancelled:
                raise TranslationCancelled()

            self.log.emit(f"开始时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log.emit(f"正在提取文本：{self._source}")
            if self._ocr:
                self.log.emit("已启用 OCR（自动识别原文语言），将识别无文本层的扫描页。")
            self.progress.emit(0, 0, "提取文本…")
            doc = pdfio.extract_document_text(
                self._source,
                ocr=self._ocr,
                cancel=lambda: self._cancelled,
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

            translate_started = time.monotonic()
            result = engine.translate_blocks(
                doc.blocks,
                self._lang,
                on_progress=lambda d, t: self.progress.emit(d, t, "翻译中…"),
                log=lambda m: self.log.emit(m),
                cancel=lambda: self._cancelled,
                doc_path=Path(self._source),
            )
            translate_elapsed = time.monotonic() - translate_started

            if self._cancelled:
                raise TranslationCancelled()

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
        except Exception as exc:  # noqa: BLE001
            import traceback

            self.error.emit(
                f"翻译失败：{exc}"
                f"（用时 {format_duration(time.monotonic() - started)}）"
                f"\n\n{traceback.format_exc()}"
            )

    def cancel(self) -> None:
        """Request cancellation (safe to call from the GUI thread)."""
        self._cancelled = True

    def _export(self, doc: pdfio.DocumentText, per_page: list[list[str]]) -> str:
        out = Path(self._output_path)
        kind = self._output_type
        if kind == "bilingual_pdf":
            pdfio.save_interleaved_pdf(
                self._source, per_page, out, self._lang, doc.pages
            )
        elif kind == "translated_pdf":
            pdfio.save_translated_pdf(self._source, doc.pages, per_page, out, self._lang)
        elif kind == "markdown":
            pdfio.save_markdown(
                per_page, doc.blocks, doc.block_pages, out, self._lang, doc.title
            )
        elif kind == "plain_text":
            pdfio.save_plain_text(per_page, out)
        else:
            raise ValueError(f"Unknown output type: {kind}")
        return str(out)
