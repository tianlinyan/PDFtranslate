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

    It used to pass only ``ocr``/``agent_mode``, so with 「OCR表格重建为矢量表格」 (or
    「表格列宽重排」) ticked the re-export silently wrote the same scanned tables again
    — the option looked broken.
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

    def test_re_export_forwards_reflow_and_rebuild_table(self):
        import tempfile

        _app()
        with tempfile.TemporaryDirectory() as tmp:
            win = self._window(tmp)
            try:
                # blockSignals: toggling would persist to the developer's real prefs.json
                for check in (win._reflow_check, win._rebuild_table_check):
                    check.blockSignals(True)
                    check.setChecked(True)
                    check.blockSignals(False)
                captured: list = []
                win._launch_worker = captured.append
                win._re_export()
                self.assertEqual(1, len(captured), "重新导出 did not start a worker")
                worker = captured[0]
                self.assertTrue(worker._reflow)
                self.assertTrue(worker._rebuild_table)
            finally:
                win._chat_thread.quit()
                win._chat_thread.wait(2000)


if __name__ == "__main__":
    unittest.main()
