"""Non-blocking AI interaction sidebar + the worker↔GUI answer bridge.

* :class:`AnswerBridge` — the worker's agent asks a question; the sidebar shows it
  and returns the answer (threading + Qt signal, like ``preview.PreviewBridge``).
* :class:`SidebarChat` — the non-blocking chat panel: a log, a free-text input,
  and AI questions with answer buttons.  It is the "侧边栏聊天 + 代理决策" entry.
"""

from __future__ import annotations

import threading
from typing import Callable

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class AnswerBridge(QObject):
    """Worker ↔ GUI channel: the agent asks, the sidebar answers.

    The worker (agent) calls :meth:`ask` to surface a question and block until the
    user answers; the GUI connects :attr:`showQuestion` to a slot that displays it
    in ``SidebarChat`` and wires the answer to :meth:`answer`.
    """

    showQuestion = pyqtSignal(str, list, str)   # question, options, target
    #: (b) A completed agent question/answer, for the console conversation to share
    #: memory with the flow: ``(question, answer, target)``.
    exchangeMade = pyqtSignal(str, str, str)

    def __init__(self, parent: QObject | None = None, timeout: float | None = None,
                 cancel: Callable[[], bool] | None = None) -> None:
        super().__init__(parent)
        self._ev = threading.Event()
        self._value: dict | None = None
        self._last_q: str = ""
        #: (a) Whether an agent question is currently awaiting the user's answer.  The
        #: answer is typed in the MAIN chat input and routed here (not a separate row).
        self._pending: bool = False
        self.pending_target: str = ""
        # A user decision must NOT silently skip: ``timeout=None`` (default) waits until
        # the user answers.  An old 600s timeout made the flow proceed as "未选择" and
        # stacked a second question row.  ``cancel`` (optional) is polled so a worker
        # cancellation interrupts the wait instead of hanging until answered.
        self._timeout = timeout
        self._cancel = cancel

    def answer(self, value, target: str = "") -> None:
        """GUI side: the user answered (value is the chosen option or free text)."""
        self._value = {"value": value, "target": target}
        self._pending = False
        if self._last_q:
            self.exchangeMade.emit(self._last_q, str(value or ""), target)
        self._ev.set()

    def ask(self, question: str, options: list[str] | None = None, target: str = "") -> dict | None:
        """Worker side: surface ``question`` and block until the user answers.

        Waits indefinitely (a user decision is honored, never skipped).  If ``cancel``
        is wired, the wait polls it every 100 ms so a user cancellation returns ``None``
        (the caller treats that as a control signal) instead of hanging.
        """
        self._value = None
        self._ev.clear()
        self._last_q = str(question or "")
        self._pending = True
        self.pending_target = target
        self.showQuestion.emit(question, list(options or []), target)
        if self._cancel is None:
            self._ev.wait(self._timeout)          # None → block until answered
        else:
            while not self._ev.wait(0.1):
                if self._cancel():
                    self._pending = False
                    return None
        return self._value

    def set_cancel(self, cancel: Callable[[], bool] | None) -> None:
        """Wire a cancellation probe (e.g. ``worker._cancelled.is_set``) so a pending
        ask returns promptly (with ``None``) when the user cancels the run."""
        self._cancel = cancel

    def clear(self) -> None:
        self._value = None
        self._ev.clear()
        self._pending = False

    def is_pending(self) -> bool:
        """Whether an agent question is currently awaiting the user's answer."""
        return self._pending


class _AskRow(QWidget):
    """A natural-language answer field for one agent question (no buttons).

    The question is already shown as an AI chat message above; this row is only a
    free-text field where the user types their answer in plain language (press Enter).
    The ``options`` list is deliberately NOT rendered as buttons — choices are made
    through a natural-language reply that the AI (or a keyword matcher) interprets.
    """

    chosen = pyqtSignal(object, str)   # value, target

    def __init__(self, target: str) -> None:
        super().__init__()
        self._target = target
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        field = QLineEdit()
        field.setPlaceholderText("请用自然语言回答（按回车确认）…")
        field.returnPressed.connect(lambda: self.chosen.emit(field.text().strip(), self._target)
                                    if field.text().strip() else None)
        box.addWidget(field)


class _ChatInput(QPlainTextEdit):
    """A plain-text chat input that wraps and auto-grows to ~``max_lines`` lines.

    Enter submits (without Shift); Shift+Enter inserts a newline.  The box grows with
    its content up to ``max_lines``, then keeps that height and scrolls internally —
    the classic multi-line chat input.  Emits :attr:`submitted` on Enter.
    """

    submitted = pyqtSignal()

    def __init__(self, max_lines: int = 3, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._max_lines = max(1, int(max_lines))
        self.setPlaceholderText("随时提问或给要求…（Enter 发送，Shift+Enter 换行）")
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # A FIXED ~3-row tall input: it visibly holds 3 lines even when empty, wraps
        # long text, and scrolls internally for more.
        self.setFixedHeight(self._height_for(self._max_lines))

    def keyPressEvent(self, event) -> None:
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)):
            self.submitted.emit()
            return
        super().keyPressEvent(event)

    def _height_for(self, lines: int) -> int:
        return int(lines * self.fontMetrics().height()
                   + 2 * self.document().documentMargin() + 4)


class SidebarChat(QWidget):
    """Non-blocking AI chat sidebar: log + free-text input + agent questions."""

    userMessage = pyqtSignal(str)               # user typed a message
    answerChosen = pyqtSignal(object, str)      # value, target
    #: User pressed "取消" (or Enter) while the AI was replying → abort the in-flight reply.
    cancelRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(300)
        #: Whether the AI is replying (so the send button becomes "取消").
        self._busy = False
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("AI 对话记录…")

        self._input = _ChatInput()
        self._input.submitted.connect(self._send)
        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self._send)
        # Match the send button to the taller multi-line input (it grows with it).
        self.send_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self._asks_box = QWidget()
        self._asks_box.setMinimumHeight(44)   # keep agent-question buttons visible
        self._asks_layout = QVBoxLayout(self._asks_box)
        self._asks_layout.setContentsMargins(0, 4, 0, 4)

        input_row = QHBoxLayout()
        input_row.addWidget(self._input, 1)
        input_row.addWidget(self.send_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self._log, 1)
        layout.addWidget(self._asks_box)
        layout.addLayout(input_row)

    # -- messages ------------------------------------------------------------
    def add_message(self, role: str, text: str) -> None:
        tag = "AI" if role == "ai" else "我"
        html = f'<p><b style="color:{"#2b6cb0" if role == "ai" else "#805ad5"}">{tag}:</b> ' \
               f'{str(text)}</p>'
        self._log.append(html)
        # Keep the newest chat line visible.
        self._log.moveCursor(QTextCursor.MoveOperation.End)
        self._log.ensureCursorVisible()

    def add_notice(self, text: str) -> None:
        """A muted system line (not a user/AI bubble), e.g. "已复制图片"."""
        self._log.append(f'<p style="color:#718096"><i>{str(text)}</i></p>')
        self._log.moveCursor(QTextCursor.MoveOperation.End)
        self._log.ensureCursorVisible()

    def show_question(self, question: str, options: list[str], target: str) -> None:
        """(a) Surface an agent question as an AI message in the conversation.

        The answer is typed in the MAIN chat input (``_on_user_message`` routes it back
        to the flow as the answer) — no separate button row.  ``options`` is carried for
        the interpreter but is not rendered; the user answers in plain language.
        """
        self.add_message("ai", question)

    def _on_chosen(self, value, target: str) -> None:
        self.add_message("我", str(value))
        self.answerChosen.emit(value, target)
        # Remove the answered question's buttons so old options don't pile up in
        # the sidebar (only the Q&A text stays in the log).
        row = self.sender()
        if isinstance(row, _AskRow):
            self._asks_layout.removeWidget(row)
            row.deleteLater()

    def send_message(self, text: str, show: bool = True) -> None:
        """Send ``text`` as a user message (emitted as ``userMessage``).

        ``show`` controls whether it is also rendered as a "我" bubble in the log
        (True for the input box).  The startup greeting passes ``show=False`` so the
        conversation opens and the AI replies, but the hidden "你好" is not echoed in
        the sidebar.
        """
        text = (text or "").strip()
        if not text:
            return
        if show:
            self.add_message("我", text)
        self.userMessage.emit(text)
        self.set_busy(True)

    def set_busy(self, busy: bool) -> None:
        """Toggle the send button: "发送" when idle, "取消" while the AI is replying."""
        self._busy = bool(busy)
        self.send_btn.setText("取消" if self._busy else "发送")

    def _send(self) -> None:
        # While the AI is replying, Send becomes Cancel (classic chat "stop generating"):
        # pressing it (or Enter) aborts the in-flight reply instead of sending more text.
        if self._busy:
            self.cancelRequested.emit()
            self.set_busy(False)
            return
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        self.send_message(text)
