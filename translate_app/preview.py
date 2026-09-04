"""Preview window for the v0.3.0 agent (drawable region + send-to-AI).

A preview page is shown as an image; the user holds the left mouse button and
draws a frame on it, then presses "发送" to crop the framed region and hand it to
the AI (the cropped image is emitted on :attr:`PreviewWindow.sendRequested`).
Cropping / coordinate mapping are pure helpers (:func:`crop_region`,
:func:`scale_rect`) so they are unit-testable without a :class:`QApplication`.
"""

from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)


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
    """Displays a preview image; zoomable/pannable; rubber-band a region at fit zoom."""

    regionChanged = pyqtSignal(object)   # image-space rect (list) or None

    def __init__(self) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._start = None
        self._rect: QRect | None = None
        self._zoom = 1.0            # multiplier over the fit-to-window scale
        self._pan = QPointF(0, 0)   # pan offset (display px), 0 = centred
        self._dragging = None       # "pan" | "select" (or None)
        self._pan_start = QPointF(0, 0)
        self._pan_base = QPointF(0, 0)
        self.setMinimumSize(500, 400)

    # -- pixmap / geometry ---------------------------------------------------
    def set_image(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._start = None
        self._rect = None
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._dragging = None
        self.update()

    def set_zoom(self, factor: float | None) -> None:
        """Set the zoom multiplier (``None`` resets to fit-to-window)."""
        if factor is None:
            self._zoom = 1.0
            self._pan = QPointF(0, 0)
        else:
            self._zoom = max(1.0, min(8.0, float(factor)))
        self.update()

    def zoom_fit(self) -> None:
        self.set_zoom(None)

    def wheelEvent(self, ev) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        delta = ev.angleDelta().y()
        factor = 1.2 if delta > 0 else 1.0 / 1.2
        self._zoom = max(1.0, min(8.0, self._zoom * factor))
        if self._zoom <= 1.0:
            self._pan = QPointF(0, 0)
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

    def _view_rect(self) -> QRectF:
        """The on-screen rect of the image at the current zoom + pan."""
        base = self._fit_rect()
        w, h = base.width() * self._zoom, base.height() * self._zoom
        x = base.center().x() - w / 2 + self._pan.x()
        y = base.center().y() - h / 2 + self._pan.y()
        return QRectF(x, y, w, h)

    def _image_rect(self, display_rect: QRect) -> list[float] | None:
        if self._pixmap is None or display_rect.width() <= 0 or display_rect.height() <= 0:
            return None
        v = self._view_rect()
        rect = [float(display_rect.x()), float(display_rect.y()),
                float(display_rect.x() + display_rect.width()),
                float(display_rect.y() + display_rect.height())]
        return scale_rect(rect, v.x(), v.y(), v.width(), v.height(),
                          self._pixmap.width(), self._pixmap.height())

    def resizeEvent(self, _ev) -> None:  # noqa: N802
        self.update()

    # -- painting ------------------------------------------------------------
    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        if self._pixmap is not None:
            p.drawPixmap(self._view_rect().toRect(), self._pixmap)
            if self._rect is not None and self._zoom <= 1.0:
                p.setPen(QPen(QColor(255, 0, 0), 2))
                p.drawRect(self._rect)
        p.end()

    # -- mouse ---------------------------------------------------------------
    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() != Qt.MouseButton.LeftButton or self._pixmap is None:
            return
        if self._zoom > 1.0:
            # Zoomed in: drag pans; region-selection only at fit zoom.
            self._dragging = "pan"
            self._pan_start = ev.position().toPoint()
            self._pan_base = QPointF(self._pan)
        else:
            self._dragging = None
            self._start = ev.position().toPoint()
            self._rect = QRect(self._start, self._start)
            self.regionChanged.emit(None)
        self.update()

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._dragging == "pan":
            pos = ev.position().toPoint()
            self._pan = QPointF(
                self._pan_base.x() + (pos.x() - self._pan_start.x()),
                self._pan_base.y() + (pos.y() - self._pan_start.y()),
            )
            self.update()
        elif self._start is not None:
            self._rect = QRect(self._start, ev.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if self._dragging == "pan":
            self._dragging = None
            self.update()
        elif self._start is not None and self._pixmap is not None:
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
    """A non-modal preview: draw a region on the page, then send it to the AI.

    Supports zoom (mouse wheel or ＋/－ buttons) so dense content is readable, and
    page navigation (上一页 / 下一页 / 跳转).  Navigation emits :attr:`pageChanged`
    with the requested 0-based page; the owner decides what to show.
    """

    sendRequested = pyqtSignal(bytes, object)   # cropped PNG, bbox in PDF points
    pageChanged = pyqtSignal(int)       # 0-based absolute page requested

    def __init__(self, title: str = "预览") -> None:
        super().__init__()
        self.setWindowTitle(title)
        self._png: bytes | None = None
        self._selected: list[float] | None = None
        self._current = 0
        self._total = 1
        #: DPI the preview PNG is rendered at; used to convert the framed region
        #: from image pixels to PDF points (the block coordinate space).
        self._dpi = 200.0

        self.canvas = SelectionCanvas()
        self.canvas.regionChanged.connect(self._on_region)

        hint = QLabel("按住左键在图上框选区域，点「发送」把框内内容发给 AI（滚轮缩放）。")
        hint.setWordWrap(True)

        # --- zoom controls ---
        zoom_in = QPushButton("＋")
        zoom_in.clicked.connect(lambda: self.canvas.set_zoom(self.canvas._zoom * 1.2))
        zoom_out = QPushButton("－")
        zoom_out.clicked.connect(lambda: self.canvas.set_zoom(self.canvas._zoom / 1.2))
        zoom_fit = QPushButton("适应")
        zoom_fit.clicked.connect(self.canvas.zoom_fit)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("缩放:"))
        zoom_row.addWidget(zoom_out)
        zoom_row.addWidget(QLabel("100%"))
        zoom_row.addWidget(zoom_in)
        zoom_row.addWidget(zoom_fit)
        zoom_row.addStretch()

        # --- page navigation ---
        self._page_label = QLabel("第 1 / 1 页")
        prev_btn = QPushButton("上一页")
        prev_btn.clicked.connect(lambda: self._request_page(self._current - 1))
        next_btn = QPushButton("下一页")
        next_btn.clicked.connect(lambda: self._request_page(self._current + 1))
        self._jump = QLineEdit()
        self._jump.setPlaceholderText("页号")
        self._jump.setFixedWidth(64)
        self._jump.returnPressed.connect(self._jump_to)
        jump_btn = QPushButton("跳转")
        jump_btn.clicked.connect(self._jump_to)
        nav_row = QHBoxLayout()
        nav_row.addWidget(prev_btn)
        nav_row.addWidget(next_btn)
        nav_row.addWidget(self._page_label)
        nav_row.addWidget(QLabel("到第"))
        nav_row.addWidget(self._jump)
        nav_row.addWidget(QLabel("页"))
        nav_row.addWidget(jump_btn)
        nav_row.addStretch()

        # --- send row ---
        self._send_btn = QPushButton("发送")
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._send)
        clear_btn = QPushButton("清除")
        clear_btn.clicked.connect(self.canvas.clear_selection)
        send_row = QHBoxLayout()
        send_row.addStretch()
        send_row.addWidget(clear_btn)
        send_row.addWidget(self._send_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(zoom_row)
        layout.addLayout(nav_row)
        layout.addLayout(send_row)
        self.resize(900, 1150)

    def show_png(self, png: bytes) -> None:
        """Display a page PNG and reset the selection."""
        self._png = png
        qimg = QImage.fromData(png)
        self.canvas.set_image(QPixmap.fromImage(qimg))
        self._selected = None
        self._send_btn.setEnabled(False)
        self.show()
        self.raise_()

    def set_page_info(self, current: int, total: int) -> None:
        """Show the current page / total for the navigation controls."""
        self._current = max(0, current)
        self._total = max(1, total)
        self._page_label.setText(f"第 {self._current + 1} / {self._total} 页")

    def _request_page(self, page: int) -> None:
        new = max(0, min(self._total - 1, page))
        if new != self._current:
            self.pageChanged.emit(new)

    def _jump_to(self) -> None:
        try:
            n = int(self._jump.text().strip())
        except (TypeError, ValueError):
            return
        self._request_page(n - 1)

    def _on_region(self, rect) -> None:
        self._selected = rect
        self._send_btn.setEnabled(rect is not None)

    def _send(self) -> None:
        if self._png is None or not self._selected:
            return
        cropped = crop_region(self._png, self._selected)
        if not cropped:
            return
        # Convert the framed region from image pixels to PDF points (the block
        # coordinate space) so ``apply_annotation`` can match it to a block.
        s = 72.0 / self._dpi
        pdf_rect = [float(v) * s for v in self._selected]
        self.sendRequested.emit(cropped, pdf_rect)


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
        if self._payload is not None:
            self._payload["page"] = page
        return self._payload

    def show_page(self, page: int, what: str = "source") -> None:
        """Worker side: show ``page`` in the preview window WITHOUT blocking.

        Unlike :meth:`get_region`, this does not wait for the user to frame a
        region — it just opens the preview so the user can *look* while the worker
        blocks on a question (the M3 special-page negotiation).
        """
        self.showPreview.emit(page, what)

    def clear(self) -> None:
        self._payload = None
        self._ev.clear()
