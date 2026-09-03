"""Preview window for the v0.3.0 agent (drawable region + send-to-AI).

A preview page is shown as an image; the user holds the left mouse button and
draws a frame on it, then presses "发送" to crop the framed region and hand it to
the AI (the cropped image is emitted on :attr:`PreviewWindow.sendRequested`).
Cropping / coordinate mapping are pure helpers (:func:`crop_region`,
:func:`scale_rect`) so they are unit-testable without a :class:`QApplication`.
"""

from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


def crop_region(png: bytes, rect, dpi: int = 72) -> bytes:
    """Crop ``rect`` (image-pixel coords ``(x0, y0, x1, y1)``) out of ``png``.

    Returns the cropped PNG.  An empty/out-of-bounds selection is clamped to the
    image; a degenerate (zero-area) region returns ``b""``.
    """
    import pymupdf as fitz

    doc = fitz.open(stream=png, filetype="png")
    try:
        page = doc[0]
        r = fitz.Rect(float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
        r = r & page.rect
        if r.is_empty or r.width <= 0 or r.height <= 0:
            return b""
        pix = page.get_pixmap(clip=r, dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def scale_rect(rect, disp_x: float, disp_y: float, disp_w: float, disp_h: float,
               img_w: float, img_h: float) -> list[float]:
    """Map a rect from display space to image space (``(x0,y0,x1,y1)``)."""
    sx = img_w / disp_w if disp_w else 1.0
    sy = img_h / disp_h if disp_h else 1.0
    return [(rect[0] - disp_x) * sx, (rect[1] - disp_y) * sy,
            (rect[2] - disp_x) * sx, (rect[3] - disp_y) * sy]


class SelectionCanvas(QWidget):
    """Displays a preview image and lets the user rubber-band a region."""

    regionChanged = pyqtSignal(object)   # image-space rect (list) or None

    def __init__(self) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._start = None
        self._rect: QRect | None = None
        self.setMinimumSize(360, 240)

    # -- pixmap / geometry ---------------------------------------------------
    def set_image(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._start = None
        self._rect = None
        self.update()

    def _fit_rect(self) -> QRectF:
        if self._pixmap is None:
            return QRectF(self.rect())
        avail = self.rect()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw <= 0 or ph <= 0:
            return QRectF(avail)
        scale = min(avail.width() / pw, avail.height() / ph)
        w, h = pw * scale, ph * scale
        x = avail.x() + (avail.width() - w) / 2
        y = avail.y() + (avail.height() - h) / 2
        return QRectF(x, y, w, h)

    def _image_rect(self, display_rect: QRect) -> list[float] | None:
        if self._pixmap is None or display_rect.width() <= 0 or display_rect.height() <= 0:
            return None
        fit = self._fit_rect()
        rect = [float(display_rect.x()), float(display_rect.y()),
                float(display_rect.x() + display_rect.width()),
                float(display_rect.y() + display_rect.height())]
        return scale_rect(rect, fit.x(), fit.y(), fit.width(), fit.height(),
                          self._pixmap.width(), self._pixmap.height())

    # -- painting ------------------------------------------------------------
    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        if self._pixmap is not None:
            p.drawPixmap(self._fit_rect().toRect(), self._pixmap)
            if self._rect is not None:
                p.setPen(QPen(QColor(255, 0, 0), 2))
                p.drawRect(self._rect)
        p.end()

    # -- mouse ---------------------------------------------------------------
    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            self._start = ev.position().toPoint()
            self._rect = QRect(self._start, self._start)
            self.regionChanged.emit(None)
            self.update()

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._start is not None:
            self._rect = QRect(self._start, ev.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if self._start is not None and self._pixmap is not None:
            self._start = None
            if self._rect is not None:
                self.regionChanged.emit(self._image_rect(self._rect.normalized()))
            self.update()

    def clear_selection(self) -> None:
        self._start = None
        self._rect = None
        self.regionChanged.emit(None)
        self.update()


class PreviewWindow(QWidget):
    """A non-modal preview: draw a region on the page, then send it to the AI."""

    sendRequested = pyqtSignal(bytes)   # cropped PNG

    def __init__(self, title: str = "预览") -> None:
        super().__init__()
        self.setWindowTitle(title)
        self._png: bytes | None = None
        self._selected: list[float] | None = None

        self.canvas = SelectionCanvas()
        self.canvas.regionChanged.connect(self._on_region)
        hint = QLabel("按住左键在图上框选区域，点「发送」把框内内容发给 AI。")
        hint.setWordWrap(True)
        self._send_btn = QPushButton("发送")
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._send)
        clear_btn = QPushButton("清除")
        clear_btn.clicked.connect(self.canvas.clear_selection)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(clear_btn)
        row.addWidget(self._send_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(row)
        self.resize(560, 760)

    def show_png(self, png: bytes) -> None:
        """Display a page PNG and reset the selection."""
        self._png = png
        qimg = QImage.fromData(png)
        self.canvas.set_image(QPixmap.fromImage(qimg))
        self._selected = None
        self._send_btn.setEnabled(False)
        self.show()
        self.raise_()

    def _on_region(self, rect) -> None:
        self._selected = rect
        self._send_btn.setEnabled(rect is not None)

    def _send(self) -> None:
        if self._png is None or not self._selected:
            return
        cropped = crop_region(self._png, self._selected)
        if cropped:
            self.sendRequested.emit(cropped)


def make_preview_window(send_handler=None, title: str = "预览") -> PreviewWindow:
    """Build a ``PreviewWindow``; ``send_handler(cropped_png)`` routes the send."""
    win = PreviewWindow(title=title)
    if send_handler is not None:
        win.sendRequested.connect(send_handler)
    return win


class PreviewBridge(QObject):
    """Worker ↔ GUI channel for the preview round-trip.

    The worker (agent) calls :meth:`get_region` to show a page and block until the
    user draws a region on the preview and presses send.  The GUI connects
    :attr:`showPreview` to a slot that opens a :class:`PreviewWindow`, and wires the
    window's ``sendRequested`` to :meth:`on_region`, which unblocks the worker with
    the cropped region image.

    Only the payload / timing logic is unit-testable without an event loop; the
    actual GUI round-trip happens through the wired slot.
    """

    showPreview = pyqtSignal(int, str)   # page, what

    def __init__(self, parent: QObject | None = None, timeout: float = 120.0) -> None:
        super().__init__(parent)
        self._ev = threading.Event()
        self._payload: dict | None = None
        self._timeout = timeout

    def on_region(self, png: bytes, rect=None) -> None:
        """GUI side: the user sent a framed region (cropped ``png``)."""
        self._payload = {"png": png, "rect": rect}
        self._ev.set()

    def get_region(self, page: int, what: str = "source", region=None) -> dict | None:
        """Worker side: show ``page`` and block until the user sends a region."""
        self._payload = None
        self._ev.clear()
        self.showPreview.emit(page, what)
        self._ev.wait(self._timeout)
        return self._payload

    def clear(self) -> None:
        self._payload = None
        self._ev.clear()
