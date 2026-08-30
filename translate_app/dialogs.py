"""Reusable dialogs: settings/glossary editor and an About dialog.

Kept separate from :mod:`.main_window` so the pure logic (glossary load/save via
:mod:`.settings`) stays simple and testable; these are thin Qt wrappers.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from . import __app_name__, __developer__, __version__
from .settings import DEFAULT_GLOSSARY_PATH, load_glossary, save_glossary


class GlossaryEditorDialog(QDialog):
    """Edit the project glossary (source -> target) and save it to disk."""

    def __init__(self, parent=None, glossary_path: str | Path | None = None):
        super().__init__(parent)
        self._path = glossary_path or DEFAULT_GLOSSARY_PATH
        self.setWindowTitle("术语表（跨分块保持一致）")
        self.setMinimumSize(520, 420)
        self._terms: dict[str, str] = load_glossary(self._path)

        path_label = QLabel(f"文件：{self._path}")
        path_label.setWordWrap(True)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["源词", "目标词"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        for src, tgt in self._terms.items():
            self._add_row(src, tgt)

        add_btn = QPushButton("添加行")
        add_btn.clicked.connect(lambda: self._add_row("", ""))
        del_btn = QPushButton("删除选中行")
        del_btn.clicked.connect(self._remove_selected)

        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self._save)
        tip = QLabel("提示：只有当前分块实际出现的词才会被注入该块的提示词。")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        box = QGroupBox("术语表")
        inner = QVBoxLayout(box)
        inner.addWidget(path_label)
        inner.addWidget(self._table)
        row = QHBoxLayout()
        row.addWidget(add_btn)
        row.addWidget(del_btn)
        row.addStretch()
        row.addWidget(save_btn)
        inner.addLayout(row)
        inner.addWidget(tip)

        root = QVBoxLayout(self)
        root.addWidget(box)
        root.addWidget(buttons)

    def _add_row(self, src: str, tgt: str) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(src))
        self._table.setItem(row, 1, QTableWidgetItem(tgt))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        for row in rows:
            self._table.removeRow(row)

    def _rows(self) -> dict[str, str]:
        terms: dict[str, str] = {}
        for row in range(self._table.rowCount()):
            src = (self._table.item(row, 0).text() if self._table.item(row, 0) else "").strip()
            tgt = (self._table.item(row, 1).text() if self._table.item(row, 1) else "").strip()
            if src and tgt:
                terms[src] = tgt
        return terms

    def _save(self) -> None:
        save_glossary(self._path, self._rows())
        self._terms = self._rows()

    def terms(self) -> dict[str, str]:
        return self._rows()


class SettingsDialog(QDialog):
    """App settings: the OCR toggle (plus a shortcut to the glossary editor)."""

    def __init__(self, parent=None, ocr_enabled: bool = True):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumSize(380, 200)

        self._ocr = QCheckBox("对无文本层的扫描页启用 OCR（RapidOCR）")
        self._ocr.setChecked(ocr_enabled)

        glossary_btn = QPushButton("编辑术语表…")
        glossary_btn.clicked.connect(self._edit_glossary)

        tip = QLabel("其余参数（并发、batch_size、temperature）在 models.json 中按模型配置。")
        tip.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addWidget(self._ocr)
        row = QHBoxLayout()
        row.addWidget(glossary_btn)
        row.addStretch()
        root.addLayout(row)
        root.addWidget(tip)
        root.addWidget(buttons)

    def _edit_glossary(self) -> None:
        GlossaryEditorDialog(self).exec()

    def use_ocr(self) -> bool:
        return self._ocr.isChecked()


class AboutDialog(QDialog):
    """Shows application information: name, version, developer and purpose."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("关于")
        self.setMinimumSize(400, 300)

        name = QLabel(__app_name__)
        name.setStyleSheet("font-size: 18pt; font-weight: bold;")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)

        version = QLabel(f"版本：v{__version__}")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)

        developer = QLabel(f"开发者：{__developer__}")
        developer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        email = QLabel("Email：tly001@vip.sina.com")
        email.setAlignment(Qt.AlignmentFlag.AlignCenter)

        summary = QLabel(
            "Windows 桌面 PDF AI 翻译工具。\n"
            "从 PDF 提取文本，通过本地模型或云端模型翻译，\n"
            "导出为双语 PDF、仅译文 PDF、Markdown 或纯文本。\n"
            "详见《AI配置手册》。"
        )
        summary.setWordWrap(True)
        summary.setAlignment(Qt.AlignmentFlag.AlignCenter)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addStretch()
        root.addWidget(name)
        root.addWidget(version)
        root.addWidget(developer)
        root.addWidget(email)
        root.addSpacing(16)
        root.addWidget(summary)
        root.addStretch()
        root.addWidget(buttons)
