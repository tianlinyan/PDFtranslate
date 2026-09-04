"""Main window: a focused PDF AI-translation tool (no reader).

The application opens straight into the translation interface: pick a PDF, an
AI model (from ``models.json``), a target language and an output format, then run
the translation.  Result is saved to file (bilingual PDF / in-place translated
PDF / Markdown / plain text) and can be opened from the window.
"""

from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QTextCursor
from PyQt6.QtWidgets import (
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
from . import pdfio
from . import preview
from . import sidebar
from .about_dialog import AboutDialog
from .chat import ChatWorker
from .doc_context import DocContext
from . import prompts
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


#: Chinese digit chars for parsing page numbers like 第三页.
_CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
              "七": 7, "八": 8, "九": 9, "十": 10, "百": 100}


def _cn_to_int(s: str) -> int | None:
    """Convert a page-number token (Arabic or Chinese) to an int, or None."""
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    # 十 / 十一 / 二十 / 三十 … 九十九 (enough for document page numbers).
    if s == "十":
        return 10
    if "十" in s:
        tens_s, _, ones_s = s.partition("十")
        tens = 1 if tens_s == "" else sum(_CN_DIGITS.get(c, 0) for c in tens_s)
        ones = sum(_CN_DIGITS.get(c, 0) for c in ones_s) if ones_s else 0
        return tens * 10 + ones
    vals = [_CN_DIGITS.get(c) for c in s]
    if all(v is not None for v in vals):
        return sum(vals)
    return None


def parse_preview_command(text: str):
    """M5: parse a preview-navigation command from the sidebar.

    Returns ``("goto", page_0based, what)`` / ``("prev", None, what)`` /
    ``("next", None, what)``, or ``None`` (not a navigation command).  ``what`` is
    ``"source"`` / ``"translation"`` when the command says 原文/译文, else ``None``
    (keep the current side).  Recognises WHOLE commands only — a content request
    that merely mentions 第 N 页 (e.g. "把第 3 页公司名换成 Bank") is left to the
    agent:
      上一页 / 下一页 / 第 N 页 / 显示第 N 页 / 打开[译文|原文]第 N 页 /
      预览[译文|原文]第 N 页
    """
    t = (text or "").strip()
    if not t:
        return None
    what = None
    if "译文" in t or "翻译" in t or "translation" in t.lower():
        what = "translation"
    elif "原文" in t or "源文" in t or "source" in t.lower():
        what = "source"
    m = re.fullmatch(
        r"(?:打开|预览|显示|去|看)?(?:译文|原文|翻译|源文)?"
        r"第\s*([0-9一二三四五六七八九十百]+)\s*页", t,
    )
    if m:
        page = _cn_to_int(m.group(1))
        if page is not None:
            return ("goto", page - 1, what)
    if re.fullmatch(r"(?:打开|预览|显示|看)?(?:译文|原文)?(上一页|下一页)", t):
        kind = "next" if "下一页" in t else "prev"
        return (kind, None, what)
    return None


class _LogBridge(QObject):
    """Thread-safe log channel: emitting from any thread is marshalled to the GUI.

    The chat worker thread may log (e.g. during lazy document extraction); emitting
    this signal from there dispatches to the connected slot on the GUI thread via a
    queued connection, so Qt widgets are never touched off the main thread.
    """

    log = pyqtSignal(str)


class MainWindow(QWidget):
    """The translation tool's main window."""

    def __init__(self):
        super().__init__()
        self._source: str | None = None
        self._thread: QThread | None = None
        self._worker: TranslateWorker | None = None
        self._last_output: str | None = None
        self._models_error = ""
        #: The last page + side (source/translation) shown in the preview window; the
        #: "预览" button reopens at this page/side instead of always page 0 / source.
        self._preview_current_page = 0
        self._preview_current_what = "source"
        # Run outcome, used by ``_cleanup`` to decide whether to reset the
        # progress bar / stage (a cancelled run leaves the bar spinning).
        self._run_ok = False
        self._cancelled_by_user = False
        self._errored = False

        #: Persistent document context shared with the interaction chat.  Its log is
        #: routed through a thread-safe signal bridge (extraction may run on the chat
        #: worker thread, so it must never touch Qt widgets directly).
        self._chat_log = _LogBridge()
        self._chat_log.log.connect(self._append_log)
        self.doc_ctx = DocContext(log=self._chat_log.log.emit)

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
        # Keep the chat's document context in sync with the language the user typed.
        self._lang_combo.editTextChanged.connect(lambda _t: self._refresh_doc_ctx())

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

        # --- 智能编排 + 扫描页识别：已交由 AI 自动处理，不再提供开关 ---
        # v0.3 起默认由 agent 视觉闭环驱动全流程翻译；扫描页自动触发 OCR（按页无文本层才识别），
        # 具体翻译/保留由 agent 与特殊页协商决定，因此主界面不再暴露这两个选项。

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
        self._preview_btn = QPushButton("预览")
        self._preview_btn.clicked.connect(
            lambda: self._show_preview(self._preview_current_page, self._preview_current_what)
        )

        # Worker ↔ GUI channel for the AI preview round-trip: the agent may open a
        # preview and wait for the user to frame a region and send it back.
        self.preview_bridge = preview.PreviewBridge(self)
        self.preview_bridge.showPreview.connect(self._show_preview)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._clear_cache_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._preview_btn)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._open_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._about_btn)

        # --- AI interaction sidebar (chat + agent questions) ---
        # worker↔GUI channels: agent asks (AnswerBridge) and preview (PreviewBridge).
        self.answer_bridge = sidebar.AnswerBridge(self)
        self.agent_sidebar = sidebar.SidebarChat()
        # Special-page / agent questions surface in the SIDEBAR (buttons), never as a
        # modal dialog — a modal would cover the preview the user is looking at.
        self.answer_bridge.showQuestion.connect(self.agent_sidebar.show_question)
        self.agent_sidebar.answerChosen.connect(self.answer_bridge.answer)
        self.agent_sidebar.userMessage.connect(self._on_user_message)

        # --- Persistent AI chat (free-text conversation with the interaction model) ---
        # Runs on its own thread so a reply never blocks the GUI.  Model is captured
        # on the GUI thread (``_selected_model``) and passed to the worker via the queued
        # signal, so the worker never touches Qt widgets.
        self._chat_thread = QThread(self)
        # ``ctx`` lets the interaction model call the chat tools (read/navigate/edit
        # the document); ``show_preview`` is the thread-safe bridge to open a page.
        self._chat_worker = ChatWorker(
            ctx=self.doc_ctx,
            log=self._chat_log.log.emit,
            show_preview=self.preview_bridge.showPreview.emit,
        )
        self._chat_worker.moveToThread(self._chat_thread)
        self._chat_worker.ask_requested.connect(self._chat_worker.ask)
        self._chat_worker.reply_ready.connect(self._on_chat_reply)
        self._chat_worker.error.connect(self._on_chat_error)
        self._chat_thread.start()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(form)
        left_layout.addWidget(self._stage)
        left_layout.addWidget(self._progress)
        left_layout.addWidget(self._log, 1)
        left_layout.addLayout(btn_row)

        root = QHBoxLayout(self)
        root.addWidget(left, 1)
        root.addWidget(self.agent_sidebar)

        if self._models_error:
            QMessageBox.warning(
                self, "模型配置", f"models.json 加载失败：\n{self._models_error}"
            )

        # On startup, proactively say hi to the AI so the conversation is open and the
        # user sees the assistant respond using the configured interaction model.
        self.agent_sidebar.send_message(prompts.CHAT_GREETING)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_source_path(self, path: str) -> None:
        """Set the source PDF (e.g. from the command line)."""
        self._source = path
        self._src_edit.setText(path)
        # Point the chat's document context at the new file (keeps any prefs lang/ocr).
        self._refresh_doc_ctx(path)

    def current_source(self) -> str | None:
        return self._source

    def _refresh_doc_ctx(self, src_path: str | None = None) -> None:
        """Point the chat's ``DocContext`` at the current source + latest lang/ocr.

        Called whenever the source or target language changes so the chat tools read
        the right document.  Clearing the source keeps the context but drops the
        document/overlay (a fresh file starts clean).  OCR is always on (scanned pages
        are identified by the AI), so the chat can read scanned pages too.  Guarded
        because the language combo's ``editTextChanged`` may fire before the full
        window is built during ``__init__``.
        """
        target = src_path if src_path is not None else self._source
        self.doc_ctx.set_source(
            target or None,
            lang=self._target_language(),
            ocr=bool(target),
        )

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

    def _on_user_message(self, text: str) -> None:
        """A sidebar message; route it to the agent WITHOUT echoing it in the main log.

        The conversation lives only in the sidebar; the main window shows the flow
        status + AI tool/operation logs, not the chat text.
        """
        # M5 preview-navigation commands drive the preview window directly.
        if self._maybe_preview_command(text):
            return
        # Best-effort: inject the free-text requirement into the running agent so
        # its next decision sees it (a no-op if no agent run is active).
        if self._worker is not None and text.strip():
            self._worker.add_user_requirement(text)
        # Persistent chat: ask the interaction model on the background thread.  The
        # model is captured here (GUI thread) and passed over, so the worker never
        # touches Qt widgets.
        model = self._selected_model()
        if model is not None:
            self._chat_worker.ask_requested.emit(text, model)

    def _on_chat_reply(self, reply: str) -> None:
        """The interaction model answered; show it only in the sidebar."""
        self.agent_sidebar.add_message("ai", reply)

    def _on_chat_error(self, err: str) -> None:
        self.agent_sidebar.add_message("ai", f"（对话失败：{err}）")

    def _show_preview(self, page: int, what: str = "source") -> None:
        """Open the preview window for ``page``; framed sends go to the bridge."""
        if not self._source or not Path(self._source).exists():
            QMessageBox.information(self, "预览", "请先选择一个要预览的 PDF。")
            return
        import pymupdf as fitz

        if what == "translation" and self._worker is not None:
            # During an AI-orchestrated run, render the in-progress translation for
            # this page (redact + redraw the blocks already translated).  The worker
            # is blocked in the preview round trip, so this reads its state safely.
            trans = self._worker.render_translation(page)
            if trans:
                png = trans
            else:
                png = self._render_source_preview(page)
        else:
            png = self._render_source_preview(page)
        if not png:
            QMessageBox.warning(self, "预览", "无法渲染该页。")
            return
        win = getattr(self, "_preview_win", None)
        if win is None:
            win = preview.PreviewWindow()
            win.sendRequested.connect(self.preview_bridge.on_region)
            win.pageChanged.connect(self._on_preview_page_changed)
            self._preview_win = win
        # Remember the page + side so a later re-open (the "预览" button) shows the
        # same page and the same source/translation as the AI's command, instead of
        # always page 0 / source.
        self._preview_current_page = page
        self._preview_current_what = what
        win.set_page_info(page, self._page_count())
        prefix = "预览（原文）" if what == "source" else "预览（译文）"
        win.setWindowTitle(f"{prefix} · 第 {page + 1} 页")
        win.show_png(png)

    def _page_count(self) -> int:
        """The source PDF page count (1 if unavailable)."""
        import pymupdf as fitz

        try:
            doc = fitz.open(str(self._source))
            try:
                return doc.page_count
            finally:
                doc.close()
        except Exception:  # noqa: BLE001
            return 1

    def _on_preview_page_changed(self, page: int) -> None:
        """A navigation request from the preview window (prev/next/jump)."""
        if self._worker is not None:
            self._worker.set_current_page(page)
        self._show_preview(page, self._preview_current_what)

    def _maybe_preview_command(self, text: str) -> bool:
        """Route M5 preview-navigation commands; returns True if handled.

        The command is NOT also fed to the agent as a requirement.
        """
        cmd = parse_preview_command(text)
        if cmd is None:
            return False
        kind, page, what = cmd
        target_what = what or self._preview_current_what
        total = self._page_count()
        if kind == "prev":
            page = max(0, self._preview_current_page - 1)
        elif kind == "next":
            page = min(total - 1, self._preview_current_page + 1)
        else:  # goto
            page = max(0, min(total - 1, page))
        if self._worker is not None:
            self._worker.set_current_page(page)
        self._show_preview(page, target_what)
        return True

    def _render_source_preview(self, page: int) -> bytes:
        """Render the source page ``page`` to a PNG (``b""`` if unusable)."""
        import pymupdf as fitz

        try:
            doc = fitz.open(str(self._source))
            try:
                if not (0 <= page < doc.page_count):
                    return b""
                return pdfio._render_page_png(doc[page], dpi=200)
            finally:
                doc.close()
        except Exception:  # noqa: BLE001
            return b""

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
        # Keep the chat's document context in sync before handing the worker its
        # protected overlay (the overlay is the source of the chat's edits).
        self._refresh_doc_ctx()

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
        self._errored = False
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
            ocr=True,
            preview_handler=self.preview_bridge.get_region,
            answer_handler=self.answer_bridge.ask,
            show_preview=self.preview_bridge.show_page,
            agent_mode=True,
            overlay=self.doc_ctx.overlay(),
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
        # Keep the newest line visible: move the cursor to the end and scroll.
        self._log.moveCursor(QTextCursor.MoveOperation.End)
        self._log.ensureCursorVisible()

    def _on_finished(self, out_path: str) -> None:
        self._run_ok = True
        self._last_output = out_path
        self._progress.setRange(0, 100)
        self._progress.setFormat("完成 100%")
        self._progress.setValue(100)
        self._stage.setText("完成")
        self._open_btn.setEnabled(True)
        self._append_log(f"翻译完成，已保存：{out_path}")
        # Feed the chat's document context the final translation so the AI reads the
        # real current output (not just the protected chat edits).
        worker = self._worker
        if worker is not None and getattr(worker, "_last_translated", None):
            self.doc_ctx.set_last_translated(worker._last_translated)

    def _on_error(self, msg: str) -> None:
        # Settle the progress bar (it may be stuck in the busy state) and mark
        # the run failed so ``_cleanup`` keeps this state instead of resetting
        # it.
        self._errored = True
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
        # stage.  Successful runs keep ``完成``; failed runs were already settled
        # to ``发生错误`` by ``_on_error`` (so we must not override that).
        if not self._run_ok and not self._errored:
            stage = "已取消" if self._cancelled_by_user else "就绪"
            self._settle_progress(0, stage)
        self._run_ok = False
        self._cancelled_by_user = False
        self._errored = False
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
        # Stop the chat worker thread so there is no "QThread destroyed while
        # running" noise just before the hard exit below.
        if self._chat_thread is not None and self._chat_thread.isRunning():
            self._chat_thread.quit()
            self._chat_thread.wait(2000)
        # ``terminate`` only kills the Qt worker thread.  The translation
        # pipeline may have left non-daemon ``ThreadPoolExecutor`` / HTTP
        # threads running; a normal close would hang the interpreter joining
        # them at exit.  Hard-exit the process so the program really quits now.
        import os

        os._exit(0)
