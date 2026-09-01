"""关于对话框：展示程序名称、版本、开发者、项目主页与功能简介。

界面文本保持简体中文，版本号统一取自 ``translate_app.__version__``，
避免在多处硬编码而不同步。
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from . import __app_name__, __version__

#: 开发者署名与项目主页（与 git remote 一致）
AUTHOR = "Tian Linyan"
HOMEPAGE = "https://github.com/tianlinyan/PDFtranslate"

#: 功能简介（每行一句，便于在窗口内居中排版）
DESCRIPTION = (
    "Windows 桌面 PDF AI 翻译工具。\n"
    "从 PDF 提取文本，通过本地模型或云端模型翻译，\n"
    "导出为双语 PDF、仅译文 PDF、Markdown 或纯文本。\n"
    "AI模型配置详见《AI模型配置手册》。"
)


def about_lines() -> list[str]:
    """返回纯文本形式的关于信息（不依赖界面，便于测试或写日志）。"""
    return [
        __app_name__,
        f"版本：v{__version__}",
        f"开发者：{AUTHOR}",
        HOMEPAGE,
        DESCRIPTION,
    ]


class AboutDialog(QDialog):
    """模态「关于」窗口。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("关于")
        self.setMinimumWidth(420)

        # --- 标题：程序名 ---
        title = QLabel(__app_name__)
        font = title.font()
        font.setPointSize(max(font.pointSize(), 9) + 10)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        version = self._text_label(f"版本：v{__version__}")
        author = self._text_label(f"开发者：{AUTHOR}")

        # --- 项目主页：可点击，用系统浏览器打开 ---
        link = QLabel(f'<a href="{HOMEPAGE}">{HOMEPAGE}</a>')
        link.setTextFormat(Qt.TextFormat.RichText)
        link.setOpenExternalLinks(True)
        link.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        link.setAlignment(Qt.AlignmentFlag.AlignCenter)

        desc = self._text_label(DESCRIPTION)
        desc.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.setText("关闭")
            close_btn.setDefault(True)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(8)
        root.addWidget(title)
        root.addSpacing(4)
        root.addWidget(version)
        root.addWidget(author)
        root.addWidget(link)
        root.addSpacing(12)
        root.addWidget(desc)
        root.addStretch(1)
        root.addWidget(buttons)

    @staticmethod
    def _text_label(text: str) -> QLabel:
        """居中、可用鼠标选中复制的普通文本标签。"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        return label
