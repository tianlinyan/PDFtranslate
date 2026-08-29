"""Background translation worker.

Runs the extract → translate → export pipeline on a worker thread so the GUI
stays responsive.  Progress, log lines and the final outcome are reported back
to the main thread through Qt signals.
"""

from __future__ import annotations

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
    ):
        super().__init__()
        self._source = source_path
        self._model = model
        self._lang = target_language
        self._output_type = output_type
        self._output_path = output_path
        self._cancelled = False

    @pyqtSlot()
    def run(self) -> None:
        try:
            if self._cancelled:
                raise TranslationCancelled()

            self.log.emit(f"正在提取文本：{self._source}")
            self.progress.emit(0, 0, "提取文本…")
            doc = pdfio.extract_document_text(self._source)

            if not doc.blocks:
                self.error.emit("未从该 PDF 中提取到任何文本，无法翻译。")
                return

            self.log.emit(f"共提取 {len(doc.blocks)} 个文本块，{doc.page_count} 页。")

            engine = TranslationEngine(self._model)
            self.log.emit(f"模型：{self._model.name} ({self._model.model})")

            result = engine.translate_blocks(
                doc.blocks,
                self._lang,
                on_progress=lambda d, t: self.progress.emit(d, t, "翻译中…"),
                log=lambda m: self.log.emit(m),
                cancel=lambda: self._cancelled,
                doc_path=Path(self._source),
            )

            if self._cancelled:
                raise TranslationCancelled()

            per_page = pdfio.group_by_page(
                doc.block_pages, result.translated, doc.page_count
            )
            self.log.emit("正在生成输出文件…")
            self.progress.emit(len(doc.blocks), len(doc.blocks), "导出…")

            out_path = self._export(doc, per_page)

            self.log.emit(f"完成：{out_path}")
            self.finished.emit(out_path)
        except TranslationCancelled:
            self.log.emit("已取消。")
        except Exception as exc:  # noqa: BLE001
            import traceback

            self.error.emit(f"翻译失败：{exc}\n\n{traceback.format_exc()}")

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
