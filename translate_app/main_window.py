"""Main window: a focused PDF AI-translation tool (no reader).

The application opens straight into the translation interface: pick a PDF, an
AI model (from ``models.json``), a target language and an output format, then run
the translation.  Result is saved to file (bilingual PDF / in-place translated
PDF / Markdown / plain text) and can be opened from the window.
"""

from __future__ import annotations

import re
from pathlib import Path

from PyQt6 import sip
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QTextCursor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
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
from .chat import ChatWorker, connect_sidebar_cancel
from .doc_context import DocContext
from . import prompts
from .settings import ModelConfig, load_models, load_prefs, save_prefs
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
    ("Русский（俄语）", "Russian"),
    ("日本語（日语）", "Japanese"),
    ("한국어（韩语）", "Korean"),
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


def worker_is_alive(worker: object) -> bool:
    """True while ``worker``'s C++ object still exists.

    ``TranslateWorker.stopped`` is connected to ``deleteLater`` *inside the
    worker thread*, so the C++ object can already be freed when the GUI thread
    processes the queued ``finished``/``error`` signal.  Any Qt-level call on the
    dead wrapper raises ``RuntimeError: wrapped C/C++ object ... has been
    deleted``, and PyQt aborts the whole process for an unhandled exception in a
    slot — v0.5.21/0.5.22 crashed exactly that way ("异常退出").
    """
    return worker is not None and not sip.isdeleted(worker)


def worker_field(worker: object, name: str, default=None):
    """Read a worker field, tolerating a worker whose C++ object was deleted.

    Python attributes live in the wrapper's ``__dict__`` and survive the C++
    deletion, so the values ``_on_finished`` needs are still readable; only a
    *missing* attribute falls through to sip and raises.  Reading defensively
    keeps the completion summary/``_last_translated``/``_last_pdf`` working even
    when the deletion won the race, instead of dropping them.
    """
    if worker is None:
        return default
    try:
        return getattr(worker, name, default)
    except RuntimeError:
        return default


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
        #: An image taken from the preview ("发送" in the preview window) and buffered
        #: until the next chat send, so it is sent to the AI together with the user's
        #: typed text.  Cleared after it is consumed.
        self._pending_image: bytes | None = None
        #: Whether the sidebar is currently streaming an AI reply (a live bubble is open).
        self._chat_streaming = False
        #: Set when the 「开始翻译」 button asks the chat AI to start a run.  If that
        #: chat turn ends without the AI actually starting one (it asked a question or
        #: only narrated), the button starts the pipeline itself — the button is an
        #: explicit user command, so "nothing happened" is never acceptable.
        self._pending_start = False
        #: True once the window's controls are fully built; guards ``_refresh_chat_settings``
        #: from running during ``__init__`` (before the model/type combos exist).
        self._settings_ready = False
        #: Right-shifts the window once on first show so the preview (docked to the
        #: left) has room.  Set in ``showEvent`` the first time.
        self._geometry_set = False
        #: The last page + side (source/translation) shown in the preview window; the
        #: "预览" button reopens at this page/side instead of always page 0 / source.
        self._preview_current_page = 0
        self._preview_current_what = "source"
        #: The exported PDF of the last successful run + its output type.  Kept here
        #: (not just on the worker) because ``_cleanup`` drops the worker reference
        #: after the run — the preview's "译文" side needs this to render the real
        #: translated output, and ``self._last_output`` may be a .md/.txt.
        self._last_pdf: str | None = None
        self._last_output_type: str = ""
        #: Source page → output page map for ``_last_pdf``.  ``translated_pdf``
        #: normally maps page i to output page i, but ``expand_pages`` inserts
        #: continuation pages, so the preview must use the map the worker reported.
        self._last_page_map: list[int] | None = None
        #: The last run's final aligned translation (flat index → text) and the source
        #: it came from.  Lets "重新导出" re-write the output with the current chat /
        #: annotation edits without re-translating.
        self._last_translated: list[str] | None = None
        self._last_translated_source: str | None = None
        #: The ``DocumentText`` that translation was aligned against.  Passed to the
        #: "重新导出" worker so it re-exports the *same* block list (no second OCR
        #: pass, no drift between the translation and a fresh extraction).
        self._last_doc = None
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
        # Keep the AI settings snapshot in sync when the model / output type change.
        self._model_combo.currentIndexChanged.connect(lambda _i: self._refresh_chat_settings())
        self._type_combo.currentIndexChanged.connect(lambda _i: self._refresh_chat_settings())

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

        # --- 翻译管线：IR 文档级管线（可选，默认关） ---
        # OCR / 智能编排已交由 AI 自动处理、不再暴露开关；但 IR 是一条独立的
        # 确定性翻译管线（无交互、公式/数字保真、术语跨页一致），可作为用户选项。
        self._ir_check = QCheckBox("IR 文档级管线")
        self._ir_check.setToolTip(
            "勾选后翻译走 IR 文档级管线（build_ir → translate_ir → 自适应导出），"
            "跳过 AI 单页视觉编排；公式/图/数字原样保留、术语跨页一致，"
            "并按段落成组送译（同一段落从几行合并为一条请求，译后按比例回填各行），"
            "但无特殊页协商 / 自检 / 预览。\n"
            "也可用环境变量 PDFTRANSLATE_IR_MODE=1 在启动时强制开启（优先级更高、无法在此关闭）；"
            "段落成组可用 PDFTRANSLATE_IR_GROUP=0 关闭。"
        )
        self._ir_check.setChecked(bool(prefs.get("ir_mode", False)))
        self._ir_check.toggled.connect(self._persist_ir_mode)

        # --- 文档级术语（agent 路径，默认开启） ---
        self._agent_terms_check = QCheckBox("文档级术语")
        self._agent_terms_check.setToolTip(
            "勾选后，AI 编排翻译前先抽取全文高频/专有术语并统一翻译一次，"
            "再注入每页翻译，保证人名/表头/财务术语跨页一致。\n"
            "仅对 AI 编排（agent）路径生效；IR 文档级管线自带术语抽取，不受此项影响。"
            "环境变量 PDFTRANSLATE_AGENT_TERMS=0 可强制关闭（优先级更高）。"
        )
        self._agent_terms_check.setChecked(bool(prefs.get("agent_terms", True)))
        self._agent_terms_check.toggled.connect(self._persist_agent_terms)

        # --- 表格列宽重排（reflow 保守层）自 v0.5.47 起默认生效、不再提供界面开关 ---
        # 文本层表格的列宽按译文需求重分配：长译文列借用相邻列的空白，数字列保持
        # 不缩，表格总宽不变。只影响「仅译文/原位」PDF 的文本层（矢量线）表格；
        # 扫描件位图表格线与双语 PDF 不受影响。需要排查时可设 PDFTRANSLATE_REFLOW=0。

        # --- 扫描表格重绘为矢量表格（默认关闭） ---
        self._rebuild_table_check = QCheckBox("OCR表格重建为矢量表格")
        self._rebuild_table_check.setToolTip(
            "勾选后，扫描（OCR）表格页会被**重绘为一张干净的矢量表格**：新建空白页，"
            "按 OCR 块聚类出的行列画网格线，行高按译文扩展，再填入译文——扫描底图、"
            "印章、手写签字不再保留（签字本就不翻译）。\n"
            "若模型支持视觉，先由模型把整张表重建为「译文 2D 网格」（并可标注合并单元格），"
            "失败或不可用时回退到上面的几何重绘。\n"
            "环境变量 PDFTRANSLATE_REBUILD_TABLE=1 可强制开启。"
        )
        self._rebuild_table_check.setChecked(bool(prefs.get("rebuild_table", False)))
        self._rebuild_table_check.toggled.connect(self._persist_rebuild_table)

        # --- 混合页（文本+图片）图内文字翻译（默认开启） ---
        self._image_text_check = QCheckBox("翻译图内文字")
        self._image_text_check.setToolTip(
            "勾选后，**混合页**（本身有文本层、又嵌了位图图片的页）里**烧在图片上的文字**"
            "会被 OCR 识别、按图片背景色覆盖，再在原位画出译文——论文的柱状图标题/轴标签、"
            "截图、流程图框内文字都属于这一类。\n"
            "安全边界（无论如何都不会做）：图片整体背景不是纸面（照片、彩色底）时**整张图一律"
            "不动**；图内纯数字/坐标刻度不翻译（覆盖后重画同一串数字只有风险）；图内文字块不会"
            "被当成表格/图表参与网格重建；换行由图片自身的行距约束，不会压到图里下一行。\n"
            "取消勾选则图内文字保持原样（与旧版一致）。扫描页（整页位图）的 OCR 不受此项影响。"
        )
        self._image_text_check.setChecked(bool(prefs.get("image_text", True)))
        self._image_text_check.toggled.connect(self._persist_image_text)

        # --- 译文扩页（默认关闭） ---
        self._expand_pages_check = QCheckBox("译文扩页")
        self._expand_pages_check.setToolTip(
            "勾选后，源页放不下的内容不再被压缩：表格行高按译文自然增长，超出页底的"
            "行与正文改排到**新增的后续页**（表格续页会重复表头行），译文页数可以多于原文。\n"
            "代价是：源页底部留白（内容已移走），且译文与原文不再逐页对应。\n"
            "适用范围：只对「仅译文/原位」PDF 的**文本层**内容生效。扫描件（位图表格线、"
            "印章、手写签字）与图内文字钉死在像素上、不可重排，仍走原逻辑；"
            "「双语 PDF」保持逐页对照，不受此项影响。\n"
            "环境变量 PDFTRANSLATE_EXPAND_PAGES=1 可强制开启。"
        )
        self._expand_pages_check.setChecked(bool(prefs.get("expand_pages", False)))
        self._expand_pages_check.toggled.connect(self._persist_expand_pages)

        # 选项排成两行、每行两个，用 QGridLayout 对齐两列：左列标签右对齐、勾选框
        # 左对齐，各行的标签/勾选框在同一竖直线上（HBox 拼装会因标签字数不同而左右
        # 错位）。
        opt_grid = QGridLayout()
        opt_grid.setContentsMargins(0, 0, 0, 0)
        opt_grid.setHorizontalSpacing(16)
        opt_grid.setVerticalSpacing(4)
        for row, (lab1, cb1, lab2, cb2) in enumerate((
            ("翻译管线", self._ir_check, "术语注入", self._agent_terms_check),
            ("扫描重建", self._rebuild_table_check, "图内文字", self._image_text_check),
            ("译文扩页", self._expand_pages_check, None, None),
        )):
            left = QLabel(lab1)
            left.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            opt_grid.addWidget(left, row, 0)
            opt_grid.addWidget(cb1, row, 1)
            if cb2 is not None:
                right = QLabel(lab2)
                right.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                opt_grid.addWidget(right, row, 2)
                opt_grid.addWidget(cb2, row, 3)
        # 第 1、3 列（勾选框列）不参与拉伸，保证两列选项各自成一条竖线。
        opt_grid.setColumnStretch(1, 0)
        opt_grid.setColumnStretch(3, 0)
        opt_grid.setColumnStretch(4, 1)
        form.addRow(opt_grid)

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
        self._start_btn.setToolTip(
            "直接开始全新翻译：重新提取 + 重新翻译 + 导出（按当前界面设置，"
            "包括「OCR表格重建」等导出选项）。等价于在侧栏输入「开始翻译」并发送。"
        )
        # The button drives the AI entry (the chat AI reads settings and calls
        # ``run_translate``), so the action shows as a user message in the sidebar.
        self._start_btn.clicked.connect(self._start_via_chat)
        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        self._open_btn = QPushButton("打开输出")
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_output)
        self._about_btn = QPushButton("关于")
        self._about_btn.clicked.connect(self._show_about)
        self._preview_btn = QPushButton("预览")
        self._preview_btn.clicked.connect(
            lambda: self._show_preview(self._preview_current_page, self._preview_current_what)
        )
        self._re_export_btn = QPushButton("重新导出")
        self._re_export_btn.setEnabled(False)
        self._re_export_btn.setToolTip(
            "用对话/标注里已有的修改，重新导出上一次的译文（不重新翻译）。"
        )
        self._re_export_btn.clicked.connect(self._re_export)

        # Worker ↔ GUI channel for the AI preview round-trip: the agent may open a
        # preview and wait for the user to frame a region and send it back.
        self.preview_bridge = preview.PreviewBridge(self)
        self.preview_bridge.showPreview.connect(self._show_preview)
        # The chat AI's ``re_export`` tool triggers the same re-export on the GUI
        # thread (the tool runs on the chat worker thread).
        self.preview_bridge.reExportRequested.connect(self._re_export)
        # The chat AI's translate-entry tools (``run_translate`` / ``set_setting``)
        # also run on the chat worker thread; the bridge queues them to the GUI where
        # they start the pipeline / change a setting.
        self.preview_bridge.translateRequested.connect(self._start)
        self.preview_bridge.setSettingRequested.connect(self._on_chat_set_setting)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self._preview_btn)
        btn_row.addWidget(self._re_export_btn)
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
        # (b) Share a flow-time Q&A with the console conversation (recorded into its
        # history) so the console and the flow keep one coherent thread.
        self.answer_bridge.exchangeMade.connect(self._record_chat_exchange)
        self.agent_sidebar.userMessage.connect(self._on_user_message)

        # --- Persistent AI chat (free-text conversation with the interaction model) ---
        # Runs on its own thread so a reply never blocks the GUI.  Model is captured
        # on the GUI thread (``_selected_model``) and passed to the worker via the queued
        # signal, so the worker never touches Qt widgets.
        self._chat_thread = QThread(self)
        # ``ctx`` lets the interaction model call the chat tools (read/navigate/edit
        # the document); ``show_preview`` is the thread-safe bridge to open a page;
        # ``re_export`` triggers the same re-export (thread-safe via the bridge).
        self._chat_worker = ChatWorker(
            ctx=self.doc_ctx,
            log=self._chat_log.log.emit,
            show_preview=self.preview_bridge.showPreview.emit,
            re_export=self.preview_bridge.reExportRequested.emit,
            start_translate=self._chat_start_translate,
            set_setting=self.preview_bridge.setSettingRequested.emit,
        )
        self._chat_worker.moveToThread(self._chat_thread)
        self._chat_worker.ask_requested.connect(self._chat_worker.ask)
        self._chat_worker.record_exchange_requested.connect(self._chat_worker._record_exchange)
        self._chat_worker.reply_ready.connect(self._on_chat_reply)
        self._chat_worker.reply_chunk.connect(self._on_chat_reply_chunk)
        self._chat_worker.error.connect(self._on_chat_error)
        self._chat_worker.cancelled.connect(self._on_chat_cancelled)
        # Sidebar "取消" (or Enter while the AI is replying) aborts the in-flight reply.
        # MUST be a DirectConnection: the worker thread is blocked inside ``ask`` for
        # the whole reply, so a queued slot would only run after it finished.
        connect_sidebar_cancel(self.agent_sidebar.cancelRequested, self._chat_worker)
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

        # Controls are built; start keeping the AI ``get_settings`` snapshot current.
        self._settings_ready = True
        self._refresh_chat_settings()

    def showEvent(self, event) -> None:  # noqa: N802
        """On first show, shift the window 200px right of its default position.

        Qt/WM places the window (usually centred) on ``show()``; nudging it 200px
        to the right once gives the preview window (docked to its LEFT, right edge
        abutting this window's left edge) room to sit beside it on screen.
        """
        super().showEvent(event)
        if not self._geometry_set:
            self._geometry_set = True
            self.move(self.x() + 200, self.y())

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_source_path(self, path: str) -> None:
        """Set the source PDF (e.g. from the command line or the file browser)."""
        self._source = path
        self._src_edit.setText(path)
        # The previous run's export belongs to the *old* file: keeping it would make
        # the preview's "译文" side render the old document's pages (measured: the
        # byte-identical PNG of the old product, labelled with the new page number),
        # and "重新导出" would rewrite the old translation next to the new source.
        self._invalidate_previous_run()
        # Point the chat's document context at the new file (keeps any prefs lang/ocr).
        self._refresh_doc_ctx(path)

    def _invalidate_previous_run(self) -> None:
        """Drop everything that belonged to the previous source / run.

        Called when the source changes (and at the start of a new run): the exported
        PDF, its page map, the committed translation and the aligned document are all
        keyed to one source file, so a stale copy is not "the last run" any more.
        """
        self._last_pdf = None
        self._last_output_type = ""
        self._last_page_map = None
        self._last_translated = None
        self._last_translated_source = None
        self._last_doc = None
        self.doc_ctx.set_last_translated(None)
        if hasattr(self, "_re_export_btn"):
            self._re_export_btn.setEnabled(False)

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
        self._refresh_chat_settings()

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
        # (a) A flow question is pending (special-page / skew / review-mode): the user's
        # message IS the answer — route it back to the flow (unblocking the worker)
        # instead of sending it to the chat model.  Checked FIRST so an answer like
        # "第3页" is not swallowed by the preview-command reader.  The "我" bubble was
        # already rendered by ``SidebarChat.send_message(show=True)`` (the only emitter
        # of ``userMessage``), so do NOT add it again here — that made the same user
        # message appear twice.
        # Any message other than the 「开始翻译」 button's own cancels a pending
        # fallback start, so a later unrelated reply never kicks off a translation.
        if self._pending_start and text.strip() != "开始翻译":
            self._pending_start = False
        if self.answer_bridge.is_pending() and text.strip():
            self.answer_bridge.answer(str(text).strip(), self.answer_bridge.pending_target)
            self.agent_sidebar.set_busy(False)   # routed as an answer, no chat turn
            return
        # M5 preview-navigation commands drive the preview window directly.
        if self._maybe_preview_command(text):
            self.agent_sidebar.set_busy(False)   # preview command, not a chat turn
            return
        # Best-effort: inject the free-text requirement into the running agent so
        # its next decision sees it (a no-op if no agent run is active).
        worker = self._live_worker()
        if worker is not None and text.strip():
            worker.add_user_requirement(text)
        # Persistent chat: ask the interaction model on the background thread.  A
        # buffered preview image (from the preview's "发送") is sent together with the
        # user's text when the model is vision-capable; otherwise the image is dropped
        # (a non-vision model cannot see it) and only the text is sent.
        model = self._selected_model()
        if model is None:
            # No usable model (models.json missing / unparsable): say so instead of
            # dropping the message silently.  ``send_message`` already set the sidebar
            # busy and rendered the "我" bubble, and nothing would ever reset either.
            self.agent_sidebar.set_busy(False)
            self.agent_sidebar.add_message(
                "ai", "没有可用的 AI 模型，请检查 models.json 配置。")
            return
        img = self._pending_image
        self._pending_image = None
        if img is not None and getattr(model, "vision", False):
            self.agent_sidebar.add_notice("（已附上截图与文本发送）")
            self._chat_worker.ask_requested.emit(text, model, img)
        else:
            self._chat_worker.ask_requested.emit(text, model, None)

    def _on_chat_reply_chunk(self, chunk: str) -> None:
        """A streamed chunk of the in-progress reply: open a live bubble, then append."""
        if not self._chat_streaming:
            self._chat_streaming = True
            self.agent_sidebar.begin_ai_message()
        self.agent_sidebar.append_ai_text(chunk)

    def _on_chat_reply(self, reply: str) -> None:
        """The interaction model answered; show it only in the sidebar."""
        if self._chat_streaming:
            # The reply was already streamed into a live bubble; finalize it with the
            # full text (idempotent — the streamed text equals ``reply``).
            self.agent_sidebar.finish_ai_message(str(reply))
            self._chat_streaming = False
        else:
            self.agent_sidebar.add_message("ai", reply)
        self.agent_sidebar.set_busy(False)
        self._maybe_fallback_start()

    def _on_chat_error(self, err: str) -> None:
        if self._chat_streaming:
            self.agent_sidebar.end_ai_message()
            self._chat_streaming = False
        self.agent_sidebar.add_message("ai", f"（对话失败：{err}）")
        self.agent_sidebar.set_busy(False)
        # Do NOT fall back to starting a run here: the model itself is unreachable, so
        # the pipeline would fail too — leave the error as the single, accurate message.
        self._pending_start = False

    def _on_chat_cancelled(self, msg: str) -> None:
        """The user aborted the in-flight reply; show it and clear the busy state."""
        if self._chat_streaming:
            self.agent_sidebar.end_ai_message()
            self._chat_streaming = False
        self.agent_sidebar.set_busy(False)
        self.agent_sidebar.add_notice(str(msg or "已取消"))
        # An aborted reply must not start a translation later on.
        self._pending_start = False

    def _show_preview(self, page: int, what: str = "source") -> None:
        """Open the preview window for ``page``; framed sends go to the bridge."""
        if not self._source or not Path(self._source).exists():
            QMessageBox.information(self, "预览", "请先选择一个要预览的 PDF。")
            return
        import pymupdf as fitz

        shown_what = "source"
        if what == "translation":
            # Show the translated output: after a run this renders from the exported
            # PDF (the real translation); during a live agent run it shows the
            # in-progress translation; with a cached last run it draws the in-place
            # translation.  Only when none is available does it fall back to the
            # source page — and then it must be labelled 原文, not 译文.
            png = self._render_translation_preview(page)
            if not png:
                png = self._render_source_preview(self._source_page_for_preview(page))
                self._append_log("  [预览] 该页暂无译文可显示，已回退到原文页。")
            else:
                shown_what = "translation"
        else:
            png = self._render_source_preview(page)
        if not png:
            QMessageBox.warning(self, "预览", "无法渲染该页。")
            return
        win = getattr(self, "_preview_win", None)
        if win is None:
            win = preview.PreviewWindow()
            win.sendRequested.connect(self._on_preview_send)
            win.pageChanged.connect(self._on_preview_page_changed)
            self._preview_win = win
        # Remember the page + side so a later re-open (the "预览" button) shows the
        # same page and the same source/translation as the AI's command, instead of
        # always page 0 / source.  Remember the ACTUAL side shown so a later
        # "下一页" keeps showing what is really on screen (not a mislabeled side).
        self._preview_current_page = page
        self._preview_current_what = shown_what
        win.set_page_info(page, self._preview_page_count(shown_what))
        prefix = "预览（译文）" if shown_what == "translation" else "预览（原文）"
        win.setWindowTitle(f"{prefix} · 第 {page + 1} 页"
                           f"{self._preview_page_suffix(page, shown_what)}")
        # A fresh popup resets size/zoom and re-docks the window; an in-place refresh
        # (prev/next/jump inside the window) keeps the user's resized window, zoomed
        # view and where they parked it.
        popup = not win.isVisible()
        win.show_png(png, reset_geometry=popup)
        if popup:
            # Sit the preview to the LEFT of the main window, its right edge abutting
            # the main window's left edge (restored every time it pops up).
            win.place_left_of(self.frameGeometry())

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

    # -- preview page numbering (source pages vs the exported PDF's pages) ----

    @staticmethod
    def _uses_output_paging(kind: str, page_map: list[int] | None) -> bool:
        """True when the 「译文」side must navigate the *exported PDF's* pages.

        An expanded in-place product no longer has one output page per source page:
        the overflow lives on continuation pages that a source-page-numbered preview
        can never reach (measured: a 1-page source exported to 2 pages, the window
        offered 「第 1/1 页」 only — the very page the option produced was invisible).
        Every other product keeps source-page navigation.
        """
        if kind != "translated_pdf" or not page_map:
            return False
        return list(page_map) != list(range(len(page_map)))

    def _output_page_count(self) -> int:
        """Page count of the last exported PDF (0 when there is none)."""
        import pymupdf as fitz

        if not self._last_pdf:
            return 0
        try:
            doc = fitz.open(str(self._last_pdf))
            try:
                return doc.page_count
            finally:
                doc.close()
        except Exception:  # noqa: BLE001
            return 0

    @staticmethod
    def _source_page_for_output(page: int, page_map: list[int] | None,
                                output_pages: int) -> int:
        """Inverse of ``_last_page_map``: which source page an output page came from."""
        if not page_map:
            return page
        out = int(page)
        for i, start in enumerate(page_map):
            end = page_map[i + 1] if i + 1 < len(page_map) else output_pages
            if int(start) <= out < int(end):
                return i
        return max(0, min(out, len(page_map) - 1))

    def _source_page_for_preview(self, page: int) -> int:
        """The source page a preview page number refers to (fallback rendering)."""
        if self._uses_output_paging(self._last_output_type, self._last_page_map):
            return self._source_page_for_output(page, self._last_page_map,
                                                self._output_page_count())
        return page

    def _preview_page_count(self, what: str) -> int:
        """How many pages the preview's navigation may visit on this side."""
        if what == "translation" and self._uses_output_paging(
                self._last_output_type, self._last_page_map):
            return self._output_page_count() or self._page_count()
        return self._page_count()

    def _preview_page_suffix(self, page: int, what: str) -> str:
        """Title suffix tying an output page back to its source page."""
        if what == "translation" and self._uses_output_paging(
                self._last_output_type, self._last_page_map):
            src = self._source_page_for_output(page, self._last_page_map,
                                               self._output_page_count())
            return f"（续页归属原文第 {src + 1} 页）"
        return ""

    def _on_preview_page_changed(self, page: int) -> None:
        """A navigation request from the preview window (prev/next/jump)."""
        worker = self._live_worker()
        if worker is not None:
            worker.set_current_page(page)
        self._show_preview(page, self._preview_current_what)

    def _on_preview_send(self, png: bytes, pdf_rect: list[float]) -> None:
        """The preview's "发送" pressed: serve the agent annotation flow AND buffer a copy.

        ``preview_bridge.on_region`` unblocks a waiting agent ``get_region`` (M6
        annotation); in parallel the cropped image is buffered so the next chat send
        carries it together with the user's text (the sidebar shows "已复制图片").
        """
        self.preview_bridge.on_region(png, pdf_rect)
        if png:
            self._pending_image = png
            self.agent_sidebar.add_notice("已复制图片，发送消息时将连同文本一起发给 AI。")

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
        worker = self._live_worker()
        if worker is not None:
            worker.set_current_page(page)
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

    def _render_translation_preview(self, page: int) -> bytes | None:
        """Render the *translated* output for ``page`` (``None`` if unavailable).

        Priority: (1) the exported PDF (the real translated output — this is what the
        user wants to see after a run, and what fixes the "preview shows only the
        source" bug); (2) the worker's in-progress translation during a live agent
        run; (3) an in-place translation drawn from the cached doc + last run (so the
        "译文" side still works for Markdown/text output or a re-open).  The caller
        falls back to the source page when this returns ``None``.
        """
        pdf_path = self._last_pdf
        if pdf_path and Path(pdf_path).exists():
            if self._uses_output_paging(self._last_output_type, self._last_page_map):
                # The preview navigates the exported PDF's own pages here, so the
                # continuation pages carrying the overflow are reachable and the
                # numbering matches the file the user opens.
                out_page = page
            else:
                out_page = self._translation_output_page(
                    page, self._last_output_type, self._last_page_map
                )
            return self._render_pdf_page_png(pdf_path, out_page)
        worker = self._live_worker()
        if worker is not None:
            return worker.render_translation(page)
        return self._render_cached_translation_preview(page)

    def _render_cached_translation_preview(self, page: int) -> bytes | None:
        """Render an in-place translation from the cached doc + last run's text.

        ``None`` when there is no cached translation for this page (so the caller
        falls back to the source, and the caller should label it honestly).  Only runs
        when the last run's translation is still in the document context — it never
        triggers a lazy document extraction (which could OCR on the GUI thread).
        """
        if self.doc_ctx.get_last_translated() is None and not self.doc_ctx.overlay():
            return None
        # ``peek_doc`` — never ``ensure_doc``: a lazy extraction here would run a
        # whole-document OCR on the GUI thread and freeze the window for minutes.
        doc = self.doc_ctx.peek_doc()
        if doc is None:
            return None
        if not (0 <= int(page) < len(doc.pages)):
            return None
        last = self.doc_ctx.get_last_translated()
        overlay = self.doc_ctx.overlay()

        def _text(idx: int) -> str:
            entry = overlay.get(idx)
            if isinstance(entry, dict) and str(entry.get("text", "")).strip():
                return str(entry.get("text", ""))
            if last is not None and idx < len(last) and str(last[idx]).strip():
                return str(last[idx])
            return ""

        offset = sum(len(p) for p in doc.pages[: int(page)])
        to_draw = [(b, _text(offset + i)) for i, b in enumerate(doc.pages[int(page)])]
        to_draw = [(b, t) for b, t in to_draw if t and t != str(b.text)]
        if not to_draw:
            return None
        import pymupdf as fitz

        try:
            fdoc = fitz.open(str(self._source))
            try:
                page_obj = fdoc[int(page)]
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
                return pdfio._render_page_png(page_obj, dpi=200)
            finally:
                fdoc.close()
        except Exception:  # noqa: BLE001 — a bad page must not crash the preview
            return None

    def _translation_output_page(self, page: int, kind: str,
                                 page_map: list[int] | None = None) -> int:
        """Map a source ``page`` to its page index in the exported PDF.

        ``translated_pdf`` overlays the translation in place, so the output page
        index normally equals the source index — except with ``expand_pages``,
        where ``page_map`` (from the export) gives the real index.  ``bilingual_pdf``
        inserts a mirror translation page after every source page, so source ``i``
        lives at output ``2*i + 1``.  Any other kind keeps the index unchanged.
        """
        if kind == "bilingual_pdf":
            return page * 2 + 1
        if page_map and 0 <= page < len(page_map):
            return int(page_map[page])
        return page

    def _render_pdf_page_png(self, path: str | Path, page: int) -> bytes | None:
        """Render ``page`` of the PDF at ``path`` to a PNG (``None`` if unusable)."""
        import pymupdf as fitz

        try:
            doc = fitz.open(str(path))
            try:
                if not (0 <= page < doc.page_count):
                    return None
                return pdfio._render_page_png(doc[page], dpi=200)
            finally:
                doc.close()
        except Exception:  # noqa: BLE001
            return None

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

    def _chat_start_translate(self, requirement: str = "",
                              page_scope: list[int] | None = None) -> dict:
        """The chat's "开始翻译" entry: emit the request, but answer honestly.

        ``translateRequested`` is a signal, so the tool could not learn whether the run
        actually started and reported "已触发开始翻译" even while a run was already going
        (the GUI just returned).  Check the one condition that makes ``_start`` a no-op
        and say so; the emit itself is thread-safe (queued to this (GUI) thread).
        """
        if self._thread is not None:
            return {"ok": False, "error": "已有翻译在运行中（可先点「取消」再重新开始）。"}
        self.preview_bridge.translateRequested.emit(requirement, page_scope)
        return {"ok": True, "message": "已触发开始翻译（按当前设置后台执行）。"}

    def _refresh_chat_settings(self) -> None:
        """Keep the AI ``get_settings`` snapshot current (source/lang/format/model)."""
        if not self._settings_ready:
            return
        model = self._selected_model()
        key = self._type_combo.currentData()
        if key not in OUTPUT_TYPES:
            key = list(OUTPUT_TYPES)[0]   # a defensive fallback (combo is always populated)
        self.doc_ctx.set_settings(
            source=self._source,
            source_name=Path(self._source).name if self._source else None,
            target_language=self._target_language(),
            output_type=key,
            output_label=OUTPUT_TYPES[key][0],
            output_path=self._path_edit.text().strip() or self._default_output_path(),
            model=model.name if model else None,
            model_id=model.id if model else None,
            ocr=True,
            agent_mode=True,
        )

    def _on_chat_set_setting(self, key: str, value: str) -> None:
        """Apply a chat AI ``set_setting`` (target_language / output_type) on the GUI."""
        key, value = str(key or ""), str(value or "")
        if key == "target_language":
            self._lang_combo.setCurrentText(value)   # typed value wins (see resolve_language)
        elif key == "output_type":
            idx = self._type_combo.findData(value)
            if idx >= 0:
                self._type_combo.setCurrentIndex(idx)
        # Reflect any change into the doc context + settings snapshot immediately.
        self._refresh_doc_ctx()

    def _start_via_chat(self) -> None:
        """Behave exactly like the user typing & sending "开始翻译" in the sidebar.

        The button no longer starts the pipeline directly — it drives the AI entry
        (``run_translate``), so the whole flow is AI-centric and the action is recorded
        in the sidebar conversation.  「开始翻译」 always means a **fresh** translation
        (re-extract + re-translate + export): the AI must not ask for confirmation
        first, and if this chat turn ends without it actually starting a run, the
        pipeline is started directly (see ``_maybe_fallback_start``).
        """
        self._pending_start = True
        self.agent_sidebar.send_message("开始翻译")

    def _maybe_fallback_start(self) -> None:
        """Start the run ourselves if the chat turn for 「开始翻译」 didn't.

        The AI is the console, but the button is an explicit user command: a reply that
        only asks a question ("要重新翻译还是重新导出？" / "确认开始吗？") or narrates
        ("即将开始…") must not leave the user with nothing running.  Called when a chat
        reply settles; a run started by ``run_translate`` sets ``self._thread`` first
        (its queued signal is delivered before the reply), so this is a no-op then.
        """
        if not self._pending_start:
            return
        self._pending_start = False
        if self._thread is not None:
            return
        self._append_log("AI 未启动翻译，已直接开始全新翻译（重新提取 + 重新翻译 + 导出）。")
        self._start()

    def _start(self, requirement: str = "", page_scope: list[int] | None = None) -> None:
        """Start the translation pipeline (the "开始翻译" entry, button or AI tool).

        ``requirement`` is an optional user requirement supplied by the AI's
        ``run_translate`` / ``run_flow`` tools; it is seeded into the run's workflow
        state so the translation agent sees it from the first decision.
        ``page_scope`` (optional) limits the run to these 0-based pages (None = all),
        so the console can define a scoped translation flow.
        """
        requirement = str(requirement or "").strip()
        if self._thread is not None:
            # A run is already going: say so instead of a silent no-op.  The chat tool
            # used to answer "已触发开始翻译" while nothing happened.
            self._append_log("  已有翻译在运行中，本次「开始翻译」请求已忽略"
                             "（可先点「取消」再重新开始）。")
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
        self._last_pdf = None   # the previous run's exported PDF no longer applies
        self._last_page_map = None
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

        self._launch_worker(TranslateWorker(
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
            ir_mode=self._ir_check.isChecked(),
            agent_terms=self._agent_terms_check.isChecked(),
            rebuild_table=self._rebuild_table_check.isChecked(),
            image_text=self._image_text_check.isChecked(),
            expand_pages=self._expand_pages_check.isChecked(),
            overlay=self.doc_ctx.overlay(),
            requirements=[requirement] if requirement else None,
            page_scope=page_scope,
        ))

    def _live_worker(self) -> TranslateWorker | None:
        """``self._worker`` while its C++ object is alive, else ``None``.

        GUI callbacks (sidebar requirement, preview navigation, cancel) must go
        through here: the worker's C++ object is deleted from the worker thread,
        so between a run finishing and ``_cleanup`` it can be gone while
        ``self._worker`` still points at the wrapper — and a Qt-level call on it
        (e.g. ``add_user_requirement``'s ``log.emit``) raises ``RuntimeError``,
        which PyQt turns into a process abort.
        """
        worker = self._worker
        return worker if worker_is_alive(worker) else None

    def _launch_worker(self, worker: TranslateWorker) -> None:
        """Move ``worker`` onto a fresh thread and wire its signals.

        ``stopped`` fires on every exit path — including a cancellation, which
        emits neither ``finished`` nor ``error``.  Quitting the thread from it
        (instead of from those two) guarantees ``_cleanup`` always runs and the
        "开始翻译" button comes back.
        """
        self._worker = worker
        self._thread = QThread(self)
        # A pending sidebar question must be interruptible by "取消": wire the worker's
        # cancellation flag so a blocked ``AnswerBridge.ask`` returns promptly.
        if getattr(self, "answer_bridge", None) is not None:
            self.answer_bridge.set_cancel(worker._cancelled.is_set)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.log.connect(self._append_log)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        # ``stopped`` is also where the worker C++ object is scheduled for
        # deletion: at this moment the worker thread's event loop is still
        # running, so the ``DeferredDelete`` is actually processed and the
        # object is freed.  Deleting it later, from ``_cleanup`` (which runs
        # after the thread has finished), would post the ``DeferredDelete`` to
        # a dead event loop that never processes it, leaking one QObject per
        # run.
        worker.stopped.connect(worker.deleteLater)
        worker.stopped.connect(self._thread.quit)
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
                    "ir_mode": bool(self._ir_check.isChecked()),
                    "agent_terms": bool(self._agent_terms_check.isChecked()),
                    "rebuild_table": bool(self._rebuild_table_check.isChecked()),
                    "image_text": bool(self._image_text_check.isChecked()),
                    "expand_pages": bool(self._expand_pages_check.isChecked()),
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

    def _persist_ir_mode(self, checked: bool) -> None:
        """Save the IR-mode checkbox the moment it is toggled.

        Unlike the combo-based prefs (saved when a run starts), a checkbox is
        toggled well before any run, so persist it here too — otherwise the toggle
        is only remembered after the next successful run.
        """
        try:
            prefs = load_prefs()
            prefs["ir_mode"] = bool(checked)
            reason = save_prefs(prefs)
            if reason and hasattr(self, "_log"):
                self._append_log(f"  警告：用户偏好保存失败（{reason}），IR 开关不会被记住。")
        except Exception:
            pass

    def _persist_agent_terms(self, checked: bool) -> None:
        """Save the agent-terminology checkbox the moment it is toggled."""
        try:
            prefs = load_prefs()
            prefs["agent_terms"] = bool(checked)
            reason = save_prefs(prefs)
            if reason and hasattr(self, "_log"):
                self._append_log(f"  警告：用户偏好保存失败（{reason}），术语开关不会被记住。")
        except Exception:
            pass

    def _persist_rebuild_table(self, checked: bool) -> None:
        """Save the rebuild-table checkbox the moment it is toggled."""
        try:
            prefs = load_prefs()
            prefs["rebuild_table"] = bool(checked)
            reason = save_prefs(prefs)
            if reason and hasattr(self, "_log"):
                self._append_log(f"  警告：用户偏好保存失败（{reason}），扫描重建开关不会被记住。")
        except Exception:
            pass

    def _persist_image_text(self, checked: bool) -> None:
        """Save the figure-text checkbox the moment it is toggled."""
        try:
            prefs = load_prefs()
            prefs["image_text"] = bool(checked)
            reason = save_prefs(prefs)
            if reason and hasattr(self, "_log"):
                self._append_log(f"  警告：用户偏好保存失败（{reason}），图内文字开关不会被记住。")
        except Exception:
            pass

    def _persist_expand_pages(self, checked: bool) -> None:
        """Save the expand-pages checkbox the moment it is toggled.

        Export-only option: it is read when the export runs, so unlike the
        figure-text switch no re-extraction is needed for a re-export to honour it.
        """
        try:
            prefs = load_prefs()
            prefs["expand_pages"] = bool(checked)
            reason = save_prefs(prefs)
            if reason and hasattr(self, "_log"):
                self._append_log(f"  警告：用户偏好保存失败（{reason}），译文扩页开关不会被记住。")
        except Exception:
            pass

    def _cancel(self) -> None:
        worker = self._live_worker()
        if worker is not None:
            worker.cancel()
            self._cancelled_by_user = True
            self._append_log("正在取消…")
        self._cancel_btn.setEnabled(False)

    def _re_export(self) -> None:
        """Re-write the output with the current chat / annotation edits, no re-translation.

        The AI interaction chat edits the protected overlay (``doc_ctx``).  Re-running
        "开始翻译" would apply it but re-translate everything (expensive).  This reuses
        the last run's committed translation and re-runs only the export step with the
        current overlay applied — so the user sees their edits in a fresh PDF quickly.
        """
        if self._thread is not None:
            return
        if not self._last_translated:
            QMessageBox.information(self, "重新导出", "还没有可重新导出的译文。")
            return
        if not self._source or not Path(self._source).exists():
            QMessageBox.information(self, "重新导出", "请先选择源 PDF。")
            return
        if self._last_translated_source and self._last_translated_source != self._source:
            QMessageBox.warning(
                self, "重新导出",
                "源文件已变更，上一次的译文不再适用，请直接点「开始翻译」重新翻译。",
            )
            return
        model = self._selected_model()
        if model is None:
            QMessageBox.warning(self, "重新导出", "没有可用模型，请检查 models.json 配置。")
            return
        key = self._type_combo.currentData()
        # 导出直接覆盖目标文件（v0.5.24 起不再生成 ``(n)`` 副本），所以「重新导出」
        # 就是重新写同一个路径——无需再从上次输出反推文件名。
        out_path = self._path_edit.text().strip() or self._default_output_path()
        self._run_ok = False
        self._cancelled_by_user = False
        self._errored = False
        self._log.clear()
        self._stage.setText("重新导出…")
        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._re_export_btn.setEnabled(False)
        self._open_btn.setEnabled(False)
        self._launch_worker(TranslateWorker(
            self._source,
            model,
            self._target_language(),
            key,
            out_path,
            ocr=True,
            agent_mode=False,
            overlay=self.doc_ctx.overlay(),
            re_export=True,
            last_translated=self._last_translated,
            last_doc=self._last_doc,
            # 导出选项必须与「开始翻译」一致：旧实现只传 ocr/agent_mode，导致勾选了
            # 「OCR表格重建为矢量表格」后点「重新导出」仍旧输出扫描表格（用户看到的
            # 就是「选项失效」）。表格列宽重排自 v0.5.47 起恒为默认值，无需传参。
            rebuild_table=self._rebuild_table_check.isChecked(),
            image_text=self._image_text_check.isChecked(),
            expand_pages=self._expand_pages_check.isChecked(),
        ))

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
        #
        # ``worker_field`` tolerates a worker whose C++ object was already freed
        # (``stopped -> deleteLater`` runs in the worker thread, so the deletion can
        # win the race against this queued slot): the fields live in the Python
        # wrapper and are still valid, so a lost race must neither drop the run's
        # results nor raise — PyQt aborts the process for an unhandled slot error.
        worker = self._worker
        last_translated = worker_field(worker, "_last_translated")
        if last_translated:
            self.doc_ctx.set_last_translated(last_translated)
            self._last_translated = list(last_translated)
            self._last_translated_source = self._source
            # Keep the document that translation was aligned against, so a later
            # "重新导出" reuses it instead of re-OCRing the source (and possibly
            # producing a different block list).
            self._last_doc = worker_field(worker, "_doc")
            self._re_export_btn.setEnabled(True)
        # Remember the exported PDF + its type so the preview's "译文" side can
        # render the REAL translated output after the run.  ``_cleanup`` drops the
        # worker reference, so this must live here.
        if worker is not None:
            self._last_pdf = worker_field(worker, "_last_pdf")
            self._last_output_type = worker_field(worker, "_output_type", "")
            self._last_page_map = worker_field(worker, "_page_map")
        else:
            self._last_pdf = None
            self._last_page_map = None
        # Gap1: feed the run's result back to the console surface so the user sees a
        # clear completion summary in the sidebar (the console can then follow up).
        report = worker_field(worker, "_report", "")
        if report:
            self.agent_sidebar.add_notice(report)

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

    def _record_chat_exchange(self, question: str, answer: str, target: str = "") -> None:
        """(b) Forward a flow-time Q&A to the console's conversation history."""
        self._chat_worker.record_exchange_requested.emit(question, answer, target)

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
        self._re_export_btn.setEnabled(self._last_translated is not None)

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
            worker = self._live_worker()
            if worker is not None:
                worker.cancel()
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
