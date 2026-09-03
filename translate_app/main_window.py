"""Main window: a focused PDF AI-translation tool (no reader).

The application opens straight into the translation interface: pick a PDF, an
AI model (from ``models.json``), a target language and an output format, then run
the translation.  Result is saved to file (bilingual PDF / in-place translated
PDF / Markdown / plain text) and can be opened from the window.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import __app_name__, __version__
from .about_dialog import AboutDialog
from .settings import ModelConfig, load_models, load_prefs, save_prefs
from .translator import clear_translation_cache
from .worker import OUTPUT_TYPES, TranslateWorker

#: (display label, language name sent to the model)
#: Supported target languages — limited by design to Chinese + the five core
#: Latin-script European languages (character sets all handled by the extractor
#: and the CJK renderer).  The parentheses show the Chinese name of each language.
LANGUAGES = [
    ("简体中文（中文）", "Simplified Chinese"),
    ("English（英语）", "English"),
    ("Español（西班牙语）", "Spanish"),
    ("Français（法语）", "French"),
    ("Deutsch（德语）", "German"),
    ("Italiano（意大利语）", "Italian"),
]

_DEFAULT_LANG = "简体中文（中文）"
_FILE_FILTER = "PDF 文件 (*.pdf);;所有文件 (*.*)"


def resolve_language(text: str) -> str:
    """Map the language combo's *text* to the language name sent to the model.

    The combo is editable, so the user may type a language that is not in
    :data:`LANGUAGES`.  ``QComboBox.currentData()`` must not be used to read it:
    when the typed text matches no item, Qt leaves ``currentIndex`` — and thus
    the item data — pointing at the previously selected entry.  Typing
    "Português" therefore translated the document into whatever was selected
    before, and even named the output file after it, with nothing on screen to
    suggest the input had been ignored.
    """
    typed = (text or "").strip()
    for label, code in LANGUAGES:
        if typed == label or typed.casefold() == code.casefold():
            return code
    # An unknown value is passed through: the model can translate into far more
    # languages than the six listed here.
    return typed or LANGUAGES[0][1]



class MainWindow(QWidget):
    """The translation tool's main window."""

    def __init__(self):
        super().__init__()
        self._source: str | None = None
        self._thread: QThread | None = None
        self._worker: TranslateWorker | None = None
        self._last_output: str | None = None
        self._models_error = ""
        # Run outcome, used by ``_cleanup`` to decide whether to reset the
        # progress bar / stage (a cancelled run leaves the bar spinning).
        self._run_ok = False
        self._cancelled_by_user = False

        self.setWindowTitle(f"{__app_name__} — AI 翻译 v{__version__}")
        self.resize(620, 560)
        self.setAcceptDrops(True)

        prefs = self._load_prefs()

        # --- Source file ---
        self._src_edit = QLineEdit()
        self._src_edit.setReadOnly(True)
        self._src_edit.setPlaceholderText("选择要翻译的 PDF 文件，或拖放到此窗口")
        open_btn = QPushButton("打开 PDF…")
        open_btn.clicked.connect(self._browse_source)
        src_row = QHBoxLayout()
        src_row.addWidget(self._src_edit, 1)
        src_row.addWidget(open_btn)

        # --- Model ---
        self._model_combo = QComboBox()
        self.models = self._load_models()
        if not self.models:
            self._model_combo.addItem("无可用模型（请检查 models.json）", None)
        else:
            for m in self.models:
                self._model_combo.addItem(f"{m.name}  ({m.model})", m.id)
            saved_model = prefs.get("model_id")
            if saved_model:
                idx = self._model_combo.findData(saved_model)
                if idx >= 0:
                    self._model_combo.setCurrentIndex(idx)

        # --- Language ---
        self._lang_combo = QComboBox()
        self._lang_combo.setEditable(True)
        for label, code in LANGUAGES:
            self._lang_combo.addItem(label, code)
        self._lang_combo.setCurrentText(prefs.get("language", _DEFAULT_LANG))

        # --- Output type ---
        self._type_combo = QComboBox()
        type_keys = list(OUTPUT_TYPES)
        for key, (label, _ext) in OUTPUT_TYPES.items():
            self._type_combo.addItem(label, key)
        # 默认输出格式为「仅译文 PDF」（原位翻译）。
        default_type_idx = type_keys.index("translated_pdf")
        idx = default_type_idx
        saved_type = prefs.get("output_type")
        if isinstance(saved_type, int):
            # legacy: saved as combo index
            if 0 <= saved_type < self._type_combo.count():
                idx = saved_type
        elif isinstance(saved_type, str):
            # saved as the output key (e.g. "translated_pdf")
            if saved_type in type_keys:
                idx = type_keys.index(saved_type)
        self._type_combo.setCurrentIndex(idx)

        # --- Output path ---
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("留空则自动生成到源文件目录")
        path_browse = QPushButton("浏览…")
        path_browse.clicked.connect(self._browse_output)
        path_row = QHBoxLayout()
        path_row.addWidget(self._path_edit, 1)
        path_row.addWidget(path_browse)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow("源文件", src_row)
        form.addRow("AI 模型", self._model_combo)
        form.addRow("目标语言", self._lang_combo)
        form.addRow("输出格式", self._type_combo)
        form.addRow("保存到", path_row)

        # --- OCR (scanned pages) ---
        # OCR 语言自动跟随原文（RapidOCR 内置中英文自动识别模型），无需手动选择。
        self._ocr_check = QCheckBox("启用 OCR（识别扫描 PDF 中的文本）")
        self._ocr_check.setChecked(bool(prefs.get("ocr", True)))
        ocr_row = QHBoxLayout()
        ocr_row.addWidget(self._ocr_check)
        ocr_row.addStretch()
        form.addRow("识别扫描页", ocr_row)

        # --- Rendered-output QA ---
        # 渲染校验：生成译文 PDF 后用视觉模型「读回」译文页，报告残留中文/重叠/越线等
        # 问题（只报告不改文件）。默认开启；PDF 输出且模型支持视觉时才生效。
        self._render_qa_check = QCheckBox("渲染校验（读回译文 PDF，报告残留/重叠/越线等问题）")
        self._render_qa_check.setChecked(bool(prefs.get("render_qa", True)))
        qa_row = QHBoxLayout()
        qa_row.addWidget(self._render_qa_check)
        qa_row.addStretch()
        form.addRow("译文质检", qa_row)

        # --- AI table rebuild ---
        # 扫描(OCR)表格页默认沿用原扫描底图、把译文叠上去。勾选后由视觉模型数行列并重建一张
        # 干净的规整表格（忽略扫描底图、印章、手写签字等非文本内容）；模型不可用/读不好时
        # 回退到几何重绘。
        self._ai_table_check = QCheckBox("AI 表格重建（由视觉模型数行列，重建干净表格，忽略扫描底图/印章/签字）")
        self._ai_table_check.setChecked(bool(prefs.get("ai_table_rebuild", False)))
        ai_row = QHBoxLayout()
        ai_row.addWidget(self._ai_table_check)
        ai_row.addStretch()
        form.addRow("扫描表格", ai_row)

        # --- Progress + log ---
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("完成 0%")
        self._progress.setFixedHeight(22)
        self._stage = QLabel("就绪")
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(3000)

        # --- Buttons ---
        self._start_btn = QPushButton("开始翻译")
        self._start_btn.clicked.connect(self._start)
        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        self._open_btn = QPushButton("打开输出")
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_output)
        self._clear_cache_btn = QPushButton("清除缓存")
        self._clear_cache_btn.clicked.connect(self._clear_cache)
        self._about_btn = QPushButton("关于")
        self._about_btn.clicked.connect(self._show_about)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._clear_cache_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._open_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._about_btn)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(self._stage)
        root.addWidget(self._progress)
        root.addWidget(self._log, 1)
        root.addLayout(btn_row)

        if self._models_error:
            QMessageBox.warning(
                self, "模型配置", f"models.json 加载失败：\n{self._models_error}"
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_source_path(self, path: str) -> None:
        """Set the source PDF (e.g. from the command line)."""
        self._source = path
        self._src_edit.setText(path)

    def current_source(self) -> str | None:
        return self._source

    # ------------------------------------------------------------------
    @staticmethod
    def _load_prefs() -> dict:
        try:
            return load_prefs()
        except Exception:
            return {}

    def _load_models(self) -> list[ModelConfig]:
        try:
            return load_models()
        except Exception as exc:
            self._models_error = str(exc)
            return []

    def _browse_source(self) -> None:
        start = str(Path(self._source).parent) if self._source else ""
        path, _f = QFileDialog.getOpenFileName(
            self, "打开 PDF 文件", start, _FILE_FILTER
        )
        if path:
            self.set_source_path(path)

    def _browse_output(self) -> None:
        key = self._type_combo.currentData()
        _label, ext = OUTPUT_TYPES[key]
        start = self._path_edit.text().strip() or self._default_output_path()
        path, _f = QFileDialog.getSaveFileName(
            self, "保存翻译结果", start, f"文件 (*{ext})"
        )
        if path:
            self._path_edit.setText(path)

    def _target_language(self) -> str:
        """The language name sent to the model (typed values win, see above)."""
        return resolve_language(self._lang_combo.currentText())

    def _default_output_path(self) -> str:
        source = Path(self._source or "document.pdf")
        lang = self._target_language()
        lang_slug = "".join(ch for ch in str(lang) if ch.isalnum())[:24] or "translated"
        key = self._type_combo.currentData()
        _label, ext = OUTPUT_TYPES[key]
        return str(source.with_name(f"{source.stem}_{lang_slug}{ext}"))

    def _selected_model(self) -> ModelConfig | None:
        mid = self._model_combo.currentData()
        for m in self.models:
            if m.id == mid:
                return m
        return self.models[0] if self.models else None

    def _start(self) -> None:
        if self._thread is not None:
            return
        if not self._source or not Path(self._source).exists():
            QMessageBox.information(self, "翻译", "请先选择一个 PDF 源文件。")
            self._browse_source()
            return

        model = self._selected_model()
        if model is None:
            QMessageBox.warning(self, "翻译", "没有可用模型，请检查 models.json 配置。")
            return
        problems = model.validate()
        if problems:
            QMessageBox.warning(self, "模型配置", "\n".join(problems))
            return

        lang = self._target_language()
        key = self._type_combo.currentData()
        out_path = self._path_edit.text().strip() or self._default_output_path()

        self._run_ok = False
        self._cancelled_by_user = False
        # Clear the log *before* saving prefs so a prefs-save warning is not
        # wiped out.
        self._log.clear()
        # A malformed endpoint string must not block the run (it can still be a
        # valid bare base_url, and the vision review shares the same client
        # config); surface it as a log hint instead.
        for w in model.endpoint_warnings():
            self._append_log(w)
        self._save_prefs(model.id, self._lang_combo.currentText(), key)
        self._stage.setText("准备…")
        self._progress.setValue(0)
        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._open_btn.setEnabled(False)

        self._thread = QThread(self)
        self._worker = TranslateWorker(
            self._source,
            model,
            str(lang),
            key,
            out_path,
            ocr=self._ocr_check.isChecked(),
            render_qa=self._render_qa_check.isChecked(),
            ai_table_rebuild=self._ai_table_check.isChecked(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        # ``stopped`` fires on every exit path — including a cancellation, which
        # emits neither ``finished`` nor ``error``.  Quitting the thread from it
        # (instead of from those two) guarantees ``_cleanup`` always runs and the
        # "开始翻译" button comes back.
        #
        # ``stopped`` is also where the worker C++ object is scheduled for
        # deletion: at this moment the worker thread's event loop is still
        # running, so the ``DeferredDelete`` is actually processed and the
        # object is freed.  Deleting it later, from ``_cleanup`` (which runs
        # after the thread has finished), would post the ``DeferredDelete`` to
        # a dead event loop that never processes it, leaking one QObject per
        # run.
        self._worker.stopped.connect(self._worker.deleteLater)
        self._worker.stopped.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()

    def _save_prefs(self, model_id: str, language: str, output_key: str) -> None:
        try:
            prefs = load_prefs()
            prefs.update(
                {
                    "model_id": model_id,
                    "language": language,
                    "output_type": output_key,
                    "ocr": self._ocr_check.isChecked(),
                    "render_qa": self._render_qa_check.isChecked(),
                    "ai_table_rebuild": self._ai_table_check.isChecked(),
                    "last_dir": str(Path(self._source or "").parent),
                }
            )
            reason = save_prefs(prefs)
            if reason:
                # A silently lost preference is invisible until the next launch;
                # say so (this runs after the log is cleared for the new run).
                self._append_log(f"  警告：用户偏好保存失败（{reason}），本次设置不会被记住。")
        except Exception:
            pass

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancelled_by_user = True
            self._append_log("正在取消…")
        self._cancel_btn.setEnabled(False)

    def _on_progress(self, done: int, total: int, stage: str) -> None:
        if total:
            # determinate: show a live percentage AND the block ratio
            pct = int(done / total * 100)
            self._progress.setRange(0, 100)
            self._progress.setFormat(f"完成 {done}/{total} 块（%p%）")
            self._progress.setValue(pct)
            self._stage.setText(f"{stage} 已完成 {done}/{total} 块")
        else:
            # total unknown yet (e.g. extracting text): show a busy bar
            self._progress.setRange(0, 0)
            self._stage.setText(stage)

    def _settle_progress(self, value: int, stage: str) -> None:
        """Restore the progress bar from the busy (indeterminate) state.

        A cancelled or failed run can leave the bar spinning
        (``setRange(0, 0)``) with the stage label stuck on the last phase;
        settle both so the UI reflects a finished run, not one still working.
        """
        self._progress.setRange(0, 100)
        self._progress.setFormat(f"完成 {value}%")
        self._progress.setValue(value)
        self._stage.setText(stage)

    def _clear_cache(self) -> None:
        reply = QMessageBox.question(
            self,
            "清除缓存",
            "确定要清除所有翻译缓存吗？\n已缓存的译文将丢失，下次需重新翻译。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            removed = clear_translation_cache()
            self._append_log(f"已清除 {removed} 个缓存文件。")
        except Exception as exc:
            self._append_log(f"清除缓存失败：{exc}")

    def _show_about(self) -> None:
        AboutDialog(self).exec()

    def _append_log(self, msg: str) -> None:
        self._log.appendPlainText(msg)

    def _on_finished(self, out_path: str) -> None:
        self._run_ok = True
        self._last_output = out_path
        self._progress.setRange(0, 100)
        self._progress.setFormat("完成 100%")
        self._progress.setValue(100)
        self._stage.setText("完成")
        self._open_btn.setEnabled(True)
        self._append_log(f"翻译完成，已保存：{out_path}")

    def _on_error(self, msg: str) -> None:
        # Settle the progress bar (it may be stuck in the busy state) and mark
        # the run failed so ``_cleanup`` keeps this state instead of resetting
        # it.
        self._settle_progress(0, "发生错误")
        self._append_log(msg)
        # Show only the headline in the dialog (the log keeps the full detail).
        headline = next((ln for ln in msg.splitlines() if ln.strip()), "翻译失败")
        QMessageBox.critical(self, "翻译失败", headline)

    def _open_output(self) -> None:
        if not self._last_output:
            return
        import os

        try:
            os.startfile(self._last_output)  # type: ignore[attr-defined]
        except Exception:
            QMessageBox.information(
                self, "输出文件", f"文件已保存到：\n{self._last_output}"
            )

    def _cleanup(self) -> None:
        # The worker C++ object was already deleted via ``stopped ->
        # deleteLater`` (processed in the worker thread before its loop
        # exited); just drop the Python reference here.  Calling
        # ``deleteLater`` again would target an object whose owning thread is
        # finished, so the event would never be processed.
        self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
        # A cancelled run emits neither ``finished`` nor ``error``, so nothing
        # else settles the progress bar — stop the busy bar and restore a ready
        # stage.  Successful / failed runs keep the state their slot set.
        if not self._run_ok and self._progress.maximum() == 0:
            stage = "已取消" if self._cancelled_by_user else "就绪"
            self._settle_progress(0, stage)
        self._run_ok = False
        self._cancelled_by_user = False
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Drag & drop a PDF to set the source
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event):  # noqa: N802
        # Only accept a drop we will actually act on: a local PDF file.  (The
        # old code accepted *any* URL — a directory or a web link was accepted
        # on hover but silently ignored on drop.)
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile() and urls[0].toLocalFile().lower().endswith(".pdf"):
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(".pdf"):
                self.set_source_path(path)
                event.acceptProposedAction()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        # 六亲不认的强行退出：关闭窗口即刻强制结束工作线程，并直接硬退出整个
        # 进程——不弹任何确认框、不等待当前请求完成、不理会任何后台线程。
        if self._thread is not None and self._thread.isRunning():
            if self._worker is not None:
                self._worker.cancel()
            # Force-kill the background worker thread, then reap it from the OS.
            self._thread.terminate()
            self._thread.wait(5000)
        # ``terminate`` only kills the Qt worker thread.  The translation
        # pipeline may have left non-daemon ``ThreadPoolExecutor`` / HTTP
        # threads running; a normal close would hang the interpreter joining
        # them at exit.  Hard-exit the process so the program really quits now.
        import os

        os._exit(0)
