"""Qt *wiring* regressions that need a live ``QApplication`` (offscreen).

The rest of the suite is deliberately Qt-widget-free (see ``test_ui.py``) so it runs
headless; the three defects below are wiring defects that only appear with a real
event loop, and each one is a regression guard for a fix:

* **F15** — ``SidebarChat.send_message`` set the busy state *after* emitting
  ``userMessage``, so a handler that routed the text elsewhere (answering a pending
  flow question, a preview-navigation command) had its ``set_busy(False)``
  immediately overwritten and the button stuck on "取消".
* **F12** — the sidebar's "取消" was connected with the default (queued) connection
  to a worker living on another thread that is blocked inside ``ask`` for the whole
  reply, so the cancel slot only ran after the reply had already finished.
* **F14** — with no usable model the user's message was dropped silently and the
  sidebar stayed busy forever.
"""
from __future__ import annotations

import os
import threading
import time
import unittest

# Must be set before any QApplication is created; the suite runs headless.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication

from translate_app.chat import connect_sidebar_cancel
from translate_app.main_window import MainWindow
from translate_app.sidebar import SidebarChat


#: Module-level reference: a locally created ``QApplication`` would be garbage
#: collected (taking the event loop with it) between tests.
_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class SendMessageBusyTest(unittest.TestCase):
    """F15: the busy state must survive a handler that resets it."""

    def test_routed_answer_leaves_the_button_ready(self):
        _app()
        sidebar = SidebarChat()
        # ``_on_user_message`` routes a pending flow question's answer here.
        sidebar.userMessage.connect(lambda _t: sidebar.set_busy(False))
        sidebar.send_message("保留原文")
        self.assertEqual("发送", sidebar.send_btn.text())

    def test_plain_message_stays_busy_until_the_reply_arrives(self):
        _app()
        sidebar = SidebarChat()
        sidebar.userMessage.connect(lambda _t: None)   # a normal chat turn
        sidebar.send_message("你好")
        self.assertEqual("取消", sidebar.send_btn.text())
        sidebar.set_busy(False)                        # what _on_chat_reply does
        self.assertEqual("发送", sidebar.send_btn.text())


class _BlockingWorker(QObject):
    """Stands in for ``ChatWorker``: on another thread, blocked inside ``ask``."""

    go = pyqtSignal()
    cancel_sig = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = False

    @pyqtSlot()
    def ask(self):
        self.started.set()
        self.release.wait(5)

    @pyqtSlot()
    def cancel_current(self):
        self.cancelled = True


class CancelWiringTest(unittest.TestCase):
    """F12: "取消" must run while the worker thread is blocked."""

    def test_cancel_runs_immediately_while_the_worker_is_blocked(self):
        _app()
        worker = _BlockingWorker()
        thread = QThread()
        worker.moveToThread(thread)
        worker.go.connect(worker.ask)
        connect_sidebar_cancel(worker.cancel_sig, worker)   # the app's wiring
        thread.start()
        try:
            worker.go.emit()
            self.assertTrue(worker.started.wait(3), "ask() did not start")
            worker.cancel_sig.emit()          # GUI thread presses 取消
            time.sleep(0.3)
            self.assertTrue(
                worker.cancelled,
                "cancel must be delivered directly; with the default (queued) "
                "connection it only arrives after ask() returns")
        finally:
            worker.release.set()
            thread.quit()
            thread.wait(2000)


class SidebarEscapingTest(unittest.TestCase):
    """侧栏是富文本控件：用户/模型的文本必须原样显示。

    非流式气泡直接拼 HTML，含 ``<`` 的文本被当成标签吃掉，``<table>`` 之后的整段
    直接消失（流式气泡一直是转义的）。
    """

    def test_markup_in_a_message_is_shown_verbatim(self):
        _app()
        sidebar = SidebarChat()
        sidebar.add_message("ai", "把 <b>公司名</b> 改成 Bank，注意 x < y")
        text = sidebar._log.toPlainText()
        self.assertIn("<b>公司名</b>", text)
        self.assertIn("x < y", text)
        sidebar.add_notice("<table> 之后的内容也必须保留")
        self.assertIn("<table> 之后的内容也必须保留", sidebar._log.toPlainText())


class SourceSwitchInvalidatesExportTest(unittest.TestCase):
    """换源文件必须丢掉上一份文档的导出状态。

    否则预览的「译文」侧会拿旧文档的导出 PDF 渲染（实测与新源页号错位、逐字节等于
    旧产物），「重新导出」也会把旧译文写到新源文件旁边。
    """

    def test_switching_the_source_drops_the_previous_export(self):
        _app()
        win = MainWindow()
        try:
            win._last_pdf = "old.pdf"
            win._last_output_type = "translated_pdf"
            win._last_page_map = [0]
            win._last_translated = ["previous"]
            win._last_translated_source = "a.pdf"
            win._last_doc = object()
            win.set_source_path("b.pdf")

            self.assertIsNone(win._last_pdf)
            self.assertIsNone(win._last_page_map)
            self.assertIsNone(win._last_translated)
            self.assertIsNone(win._last_doc)
            self.assertEqual("", win._last_output_type)
            self.assertFalse(win._re_export_btn.isEnabled(),
                             "换源后不能再用旧译文重新导出")
        finally:
            win._chat_thread.quit()
            win._chat_thread.wait(2000)


class NoModelMessageTest(unittest.TestCase):
    """F14: no usable model must answer the user, not swallow the message."""

    def test_message_without_a_model_is_answered(self):
        _app()
        win = MainWindow()
        try:
            win.models = []                      # models.json failed to load
            asked: list = []
            win._chat_worker.ask_requested.connect(lambda *a: asked.append(a))
            win.agent_sidebar.send_message("你好")
            self.assertEqual([], asked)
            self.assertEqual("发送", win.agent_sidebar.send_btn.text())
            self.assertIn("models.json", win.agent_sidebar._log.toPlainText())
        finally:
            win._chat_thread.quit()
            win._chat_thread.wait(2000)


class PreviewSendRegionTest(unittest.TestCase):
    """F17: "发送" must hand the AI the marked region, not the whole page."""

    def _window(self):
        from translate_app.preview import PreviewWindow
        import pymupdf as fitz

        _app()
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "Hello PDF", fontsize=12)
        png = page.get_pixmap(dpi=200).tobytes("png")
        doc.close()
        win = PreviewWindow()
        win.show_png(png, reset_geometry=False)
        return win

    def test_marker_strokes_become_the_sent_region(self):
        from PyQt6.QtCore import QPointF

        win = self._window()
        win.canvas._strokes.append([QPointF(400.0, 500.0), QPointF(500.0, 560.0)])
        got: list = []
        win.sendRequested.connect(lambda png, rect: got.append(rect))
        win._send()
        self.assertEqual(1, len(got))
        rect = got[0]
        # Image pixels (394, 494)-(506, 566) at 200 dpi → PDF points (×0.36).
        self.assertAlmostEqual(141.84, rect[0], places=1)
        self.assertAlmostEqual(177.84, rect[1], places=1)
        self.assertAlmostEqual(182.16, rect[2], places=1)
        self.assertAlmostEqual(203.76, rect[3], places=1)

    def test_no_marker_sends_no_region(self):
        win = self._window()
        got: list = []
        win.sendRequested.connect(lambda png, rect: got.append(rect))
        win._send()
        self.assertEqual([None], got)


class ReExportForwardsExportFlagsTest(unittest.TestCase):
    """v0.5.25: 「重新导出」 must honor the export knobs, like 「开始翻译」.

    It used to pass only ``ocr``/``agent_mode``, so with 「OCR表格重建为矢量表格」 ticked
    the re-export silently wrote the same scanned tables again — the option looked
    broken.  (「表格列宽重排」自 v0.5.47 起恒为默认值，不再是界面旋钮。)
    """

    def _window(self, tmp: str):
        import pymupdf as fitz

        from translate_app.settings import ModelConfig

        win = MainWindow()
        src = os.path.join(tmp, "src.pdf")
        doc = fitz.open()
        doc.new_page(width=595, height=842)
        doc.save(src)
        doc.close()
        win.models = [ModelConfig(
            id="t", name="t", type="chat",
            endpoint="http://127.0.0.1:1/v1/chat/completions", model="m")]
        win._source = src
        win._last_translated_source = src
        win._last_translated = [["x"]]         # per-page translations, non-empty
        win._path_edit.setText(os.path.join(tmp, "out.pdf"))
        return win

    def test_re_export_forwards_rebuild_table_and_default_reflow(self):
        import tempfile

        _app()
        with tempfile.TemporaryDirectory() as tmp:
            win = self._window(tmp)
            try:
                # blockSignals: toggling would persist to the developer's real prefs.json
                win._rebuild_table_check.blockSignals(True)
                win._rebuild_table_check.setChecked(True)
                win._rebuild_table_check.blockSignals(False)
                captured: list = []
                win._launch_worker = captured.append
                win._re_export()
                self.assertEqual(1, len(captured), "重新导出 did not start a worker")
                worker = captured[0]
                # v0.5.47：reflow 不再是界面旋钮，重导出的 worker 取默认值（开）。
                self.assertTrue(worker._reflow)
                self.assertTrue(worker._rebuild_table)
            finally:
                win._chat_thread.quit()
                win._chat_thread.wait(2000)

    def test_the_figure_text_checkbox_is_wired_and_on_by_default(self):
        import tempfile

        _app()
        with tempfile.TemporaryDirectory() as tmp:
            win = self._window(tmp)
            try:
                self.assertTrue(win._image_text_check.isChecked(),
                                "图内文字默认开启（v0.5.44/patch 起的行为）")
                self.assertIn("图内文字", win._image_text_check.toolTip())
                captured: list = []
                win._launch_worker = captured.append
                win._image_text_check.blockSignals(True)
                win._image_text_check.setChecked(False)
                win._image_text_check.blockSignals(False)
                win._re_export()
                self.assertEqual(1, len(captured))
                self.assertIs(False, captured[0]._image_text,
                              "勾选框状态必须传给重新导出的 worker")
            finally:
                win._chat_thread.quit()
                win._chat_thread.wait(2000)

    def test_re_export_reextracts_when_the_figure_text_setting_changed(self):
        # The flag is applied at extraction time: reusing a document extracted with
        # the other setting would silently ignore the checkbox.
        import tempfile

        import pymupdf as fitz

        from translate_app import pdfio
        from translate_app.worker import TranslateWorker
        from translate_app.settings import ModelConfig

        _app()
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.pdf")
            doc = fitz.open()
            doc.new_page(width=300, height=300)
            doc.save(src)
            doc.close()
            model = ModelConfig(id="t", name="t", type="chat",
                                endpoint="http://127.0.0.1:1/v1/chat/completions",
                                model="m")
            stale = pdfio.DocumentText(title="stale", image_text=False)
            worker = TranslateWorker(src, model, "English", "translated_pdf",
                                     os.path.join(tmp, "out.pdf"),
                                     ocr=True, re_export=True,
                                     last_translated=["x"], last_doc=stale,
                                     image_text=True)
            seen: list[int] = []
            worker._extract_doc = lambda: (seen.append(1),
                                           pdfio.DocumentText(title="fresh"))[1]
            worker._run_re_export()
            self.assertTrue(seen, "设置不一致时必须重新提取，而不是复用旧文档")


class OptionsGridLayoutTest(unittest.TestCase):
    """v0.6.14：选项区删掉每项前面的分类标签，5 个勾选框排成两行三列。

    标签（「翻译管线」「术语注入」…）只是把勾选框文字换个说法重复一遍，还占掉半行
    宽度、把 5 个选项挤成 3 行；删掉后按 2×3 排。勾选框自带说明文字，悬停提示照旧。
    """

    def test_five_checkboxes_in_two_rows_of_three_without_labels(self):
        from PyQt6.QtWidgets import QCheckBox

        _app()
        win = MainWindow()
        try:
            grid = win._option_grid
            boxes = (win._ir_check, win._agent_terms_check,
                     win._rebuild_table_check, win._image_text_check,
                     win._expand_pages_check)
            self.assertEqual(len(boxes), grid.count(),
                             "网格里只能有勾选框：分类标签必须已删除")
            spots = []
            for cb in boxes:
                index = grid.indexOf(cb)
                self.assertGreaterEqual(index, 0, f"{cb.text()} 不在选项网格里")
                self.assertIsInstance(grid.itemAt(index).widget(), QCheckBox)
                row, col, row_span, col_span = grid.getItemPosition(index)
                self.assertEqual((1, 1), (row_span, col_span))
                spots.append((row, col))
            # 顺序不变：管线 → 术语 → 扫描重建（第一行）→ 图内文字 → 扩页（第二行）。
            self.assertEqual([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)], spots)
        finally:
            win._chat_thread.quit()
            win._chat_thread.wait(2000)


class StartButtonAlwaysStartsTest(unittest.TestCase):
    """v0.5.31: 「开始翻译」＝全新翻译，且 AI 没启动时按钮自己启动。

    The button drives the AI entry, but the AI sometimes only asked a question
    (「要重新翻译还是重新导出？」「确认开始吗？」) or narrated 「即将开始…」 and nothing
    ran.  A button the user pressed must never end with nothing happening — and a
    later, unrelated chat reply must not start a translation either.
    """

    class _StubChatWorker(QObject):
        """Swallows ``ask_requested`` so no real chat turn (or network) happens."""

        class _Sig:
            @staticmethod
            def emit(*_a) -> None:
                pass

        ask_requested = _Sig()
        record_exchange_requested = _Sig()

    def _window(self, tmp: str):
        import pymupdf as fitz

        from translate_app.settings import ModelConfig

        win = MainWindow()
        src = os.path.join(tmp, "src.pdf")
        doc = fitz.open()
        doc.new_page(width=595, height=842)
        doc.save(src)
        doc.close()
        win.models = [ModelConfig(
            id="t", name="t", type="chat",
            endpoint="http://127.0.0.1:1/v1/chat/completions", model="m")]
        win._source = src
        win._path_edit.setText(os.path.join(tmp, "out.pdf"))
        win._save_prefs = lambda *_a, **_k: None   # never touch the real prefs.json
        win._chat_worker = self._StubChatWorker()
        return win

    def test_button_starts_the_run_itself_when_the_ai_did_not(self):
        import tempfile

        _app()
        with tempfile.TemporaryDirectory() as tmp:
            win = self._window(tmp)
            try:
                started: list = []
                win._launch_worker = started.append
                win._start_via_chat()
                self.assertTrue(win._pending_start)   # waiting for the chat turn
                self.assertEqual([], started)
                win._maybe_fallback_start()           # the AI's reply arrived, no run
                self.assertEqual(1, len(started), "按钮没有启动翻译")
                self.assertFalse(win._pending_start)
                worker = started[0]
                # It is a genuine fresh run: no re-export, no reused translation.
                self.assertFalse(getattr(worker, "_re_export", False))
                self.assertIsNone(win._last_translated)
            finally:
                win._chat_thread.quit()
                win._chat_thread.wait(2000)

    def test_no_second_start_when_the_ai_already_started_one(self):
        import tempfile

        _app()
        with tempfile.TemporaryDirectory() as tmp:
            win = self._window(tmp)
            try:
                started: list = []
                win._launch_worker = started.append
                win._start_via_chat()
                win._thread = object()                # run_translate already launched
                win._maybe_fallback_start()
                self.assertEqual([], started, "AI 已启动却重复启动")
                self.assertFalse(win._pending_start)
            finally:
                win._thread = None
                win._chat_thread.quit()
                win._chat_thread.wait(2000)

    def test_another_message_cancels_the_pending_start(self):
        import tempfile

        _app()
        with tempfile.TemporaryDirectory() as tmp:
            win = self._window(tmp)
            try:
                started: list = []
                win._launch_worker = started.append
                win._start_via_chat()
                win.agent_sidebar.send_message("第3页翻成什么了？")   # unrelated turn
                self.assertFalse(win._pending_start)
                win._maybe_fallback_start()
                self.assertEqual([], started, "无关消息之后仍然启动了翻译")
            finally:
                win._chat_thread.quit()
                win._chat_thread.wait(2000)


if __name__ == "__main__":
    unittest.main()
