"""Reusable dialogs: settings/glossary editor and an export preview.

Kept separate from :mod:`.main_window` so the pure logic (glossary load/save via
:mod:`.settings`) stays simple and testable; these are thin Qt wrappers.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QFont
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
    QTextBrowser,
    QVBoxLayout,
)

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


class PreviewDialog(QDialog):
    """Bilingual preview of the translation, page by page, before export."""

    def __init__(self, parent=None, per_page_translated=None, per_page_source=None):
        super().__init__(parent)
        self.setWindowTitle("译文预览")
        self.resize(680, 640)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setFont(QFont("Microsoft YaHei", 10))
        browser.setHtml(self._build_html(per_page_translated or [], per_page_source or []))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addWidget(browser)
        root.addWidget(buttons)

    @staticmethod
    def _build_html(per_page_translated, per_page_source) -> str:
        parts = ["<style>body{font-family:'Segoe UI',sans-serif}"
                 "details{margin:4px 0}summary{color:#666;cursor:pointer}</style>"]
        n = max(len(per_page_translated), len(per_page_source))
        for i in range(n):
            trans = per_page_translated[i] if i < len(per_page_translated) else []
            src = per_page_source[i] if i < len(per_page_source) else []
            parts.append(f"<h3>第 {i + 1} 页</h3>")
            for block in trans:
                if block:
                    parts.append(f"<p>{block}</p>")
            src_lines = "".join(f"<p>{b}</p>" for b in src if b)
            if src_lines:
                parts.append(f"<details><summary>原文</summary>{src_lines}</details>")
        return "<html><body>" + "".join(parts) + "</body></html>"
