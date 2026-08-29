"""Application entry point for PDF Translate.

Usage:
    python main.py [file.pdf]
"""

from __future__ import annotations

import sys

from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication

from translate_app import __app_name__, __version__
from translate_app.main_window import MainWindow


def _apply_style(app: QApplication) -> None:
    """Use the Fusion style for a clean desktop look."""
    app.setStyle("Fusion")
    palette = app.style().standardPalette()
    app.setPalette(palette)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv
    QGuiApplication.setApplicationName(__app_name__)
    QGuiApplication.setApplicationVersion(__version__)
    QGuiApplication.setOrganizationName("PDFtranslate")

    app = QApplication(argv)
    _apply_style(app)

    window = MainWindow()
    # Optionally pre-select a PDF given on the command line.
    if len(argv) > 1:
        window.set_source_path(argv[1])
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
