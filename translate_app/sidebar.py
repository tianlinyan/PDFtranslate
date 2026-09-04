"""Non-blocking AI interaction sidebar + the worker↔GUI answer bridge.

* :class:`AnswerBridge` — the worker's agent asks a question; the sidebar shows it
  and returns the answer (threading + Qt signal, like ``preview.PreviewBridge``).
* :class:`SidebarChat` — the non-blocking chat panel: a log, a free-text input,
  and AI questions with answer buttons.  It is the "侧边栏聊天 + 代理决策" entry.
"""

from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
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

    def __init__(self, parent: QObject | None = None, timeout: float = 600.0) -> None:
        super().__init__(parent)
        self._ev = threading.Event()
        self._value: dict | None = None
        self._timeout = timeout

    def answer(self, value, target: str = "") -> None:
        """GUI side: the user answered (value is the chosen option or free text)."""
        self._value = {"value": value, "target": target}
        self._ev.set()

    def ask(self, question: str, options: list[str] | None = None, target: str = "") -> dict | None:
        """Worker side: surface ``question`` and block until the user answers."""
        self._value = None
        self._ev.clear()
        self.showQuestion.emit(question, list(options or []), target)
        self._ev.wait(self._timeout)
        return self._value

    def clear(self) -> None:
        self._value = None
        self._ev.clear()


class _AskRow(QWidget):
    """A stack of answer buttons for one agent question, plus a free-text field."""

    chosen = pyqtSignal(object, str)   # value, target

    def __init__(self, question: str, options: list[str], target: str) -> None:
        super().__init__()
        self._target = target
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        buttons = QHBoxLayout()
        for opt in options:
            b = QPushButton(str(opt))
            b.clicked.connect(lambda _c, o=opt: self.chosen.emit(o, self._target))
            buttons.addWidget(b)
        buttons.addStretch()
        box.addLayout(buttons)
        field = QLineEdit()
        field.setPlaceholderText("或输入其他回答…")
        field.returnPressed.connect(lambda: self.chosen.emit(field.text().strip(), self._target)
                                    if field.text().strip() else None)
        box.addWidget(field)


class SidebarChat(QWidget):
    """Non-blocking AI chat sidebar: log + free-text input + agent questions."""

    userMessage = pyqtSignal(str)               # user typed a message
    answerChosen = pyqtSignal(object, str)      # value, target

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(300)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("AI 对话记录…")

        self._input = QLineEdit()
        self._input.setPlaceholderText("随时提问或给要求…")
        self._input.returnPressed.connect(self._send)
        send_btn = QPushButton("发送")
        send_btn.clicked.connect(self._send)

        self._asks_box = QWidget()
        self._asks_box.setMinimumHeight(44)   # keep agent-question buttons visible
        self._asks_layout = QVBoxLayout(self._asks_box)
        self._asks_layout.setContentsMargins(0, 4, 0, 4)

        input_row = QHBoxLayout()
        input_row.addWidget(self._input, 1)
        input_row.addWidget(send_btn)

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
        """Display an agent question with answer buttons; forward the answer."""
        self.add_message("ai", question)
        row = _AskRow(question, options, target)
        row.chosen.connect(self._on_chosen)
        self._asks_layout.addWidget(row)

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

    def _send(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self.send_message(text)
