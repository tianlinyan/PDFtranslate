"""Preview window for the v0.3.0 agent (drawable region + send-to-AI).

A preview page is shown as an image; the user holds the left mouse button and
draws a frame on it, then presses "发送" to crop the framed region and hand it to
the AI (the cropped image is emitted on :attr:`PreviewWindow.sendRequested`).
Cropping / coordinate mapping are pure helpers (:func:`crop_region`,
:func:`scale_rect`) so they are unit-testable without a :class:`QApplication`.
"""

from __future__ import annotations

import threading

from PyQt6.QtCore import QBuffer, QIODevice, QObject, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

#: Default size the preview window opens at.  Every popup resets to this size, and
#: the window is placed immediately left of the main window (its right edge abuts
#: the main window's left edge) — see :meth:`PreviewWindow.place_left_of`.
_PREVIEW_DEFAULT_W = 680
_PREVIEW_DEFAULT_H = 900


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
    """Displays a preview image; zoomable (wheel/buttons); right-click draws a stroke."""

    def __init__(self) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._zoom = 1.0            # multiplier over the fit-to-window scale
        self._pan = QPointF(0, 0)   # pan offset (display px), 0 = centred
        # Right-button freehand annotation strokes, in IMAGE space so they stay glued
        # to the page when the user zooms.  ``_active_ink`` is the stroke being drawn
        # right now; ``_strokes`` are the finished ones (accumulated markup, cleared
        # when a new page/side is shown).  Left-click region selection is removed.
        self._strokes: list[list[QPointF]] = []
        self._active_ink: list[QPointF] | None = None
        self.setMinimumSize(500, 400)

    # -- pixmap / geometry ---------------------------------------------------
    def set_image(self, pixmap: QPixmap, keep_view: bool = False) -> None:
        # ``keep_view`` preserves the zoom/pan when the new image has the same
        # dimensions (prev/next page of the same render — the user stays zoomed in
        # where they were) and resets otherwise (a fresh popup, the other side).
        same = (keep_view and self._pixmap is not None
                and pixmap.width() == self._pixmap.width()
                and pixmap.height() == self._pixmap.height())
        self._pixmap = pixmap
        self._strokes = []
        self._active_ink = None
        if not same:
            self._zoom = 1.0
            self._pan = QPointF(0, 0)
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

    def _to_image_pos(self, pos: QPointF) -> QPointF:
        """Map a display point to image space (so a freehand stroke sticks to the page)."""
        v = self._view_rect()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if v.width() <= 0 or v.height() <= 0 or pw <= 0 or ph <= 0:
            return QPointF(0, 0)
        return QPointF((pos.x() - v.x()) / v.width() * pw,
                       (pos.y() - v.y()) / v.height() * ph)

    def _from_image_pos(self, ip: QPointF) -> QPointF:
        v = self._view_rect()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw <= 0 or ph <= 0:
            return QPointF(0, 0)
        return QPointF(ip.x() / pw * v.width() + v.x(),
                       ip.y() / ph * v.height() + v.y())

    def _ink_contains(self, pos: QPointF) -> bool:
        """True when ``pos`` (display) is over the displayed image area."""
        return self._pixmap is not None and self._view_rect().contains(pos)

    def resizeEvent(self, _ev) -> None:  # noqa: N802
        self.update()

    # -- painting ------------------------------------------------------------
    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        if self._pixmap is not None:
            p.drawPixmap(self._view_rect().toRect(), self._pixmap)
            # Freehand right-button annotations (image space → display space, so they
            # stay glued to the page through zoom).  Semi-transparent red.
            if self._strokes or self._active_ink:
                pen = QPen(QColor(255, 0, 0, 200), 2)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.setPen(pen)
                for stroke in list(self._strokes) + ([self._active_ink]
                                                     if self._active_ink else []):
                    if not stroke:
                        continue
                    pts = QPolygonF(self._from_image_pos(ip) for ip in stroke)
                    if pts.count() > 1:
                        p.drawPolyline(pts)
        p.end()

    # -- mouse ---------------------------------------------------------------
    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        if ev.button() == Qt.MouseButton.RightButton:
            # Freehand annotation: hold right + drag to draw a stroke over the image.
            if self._ink_contains(ev.position()):
                self._active_ink = [self._to_image_pos(ev.position())]
                self.update()

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._active_ink is not None:
            # Freehand stroke being drawn (right button held): append the point.
            self._active_ink.append(self._to_image_pos(ev.position()))
            self.update()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if self._active_ink is not None:
            # Finish the freehand stroke; keep it as markup on this page.
            self._strokes.append(self._active_ink)
            self._active_ink = None
            self.update()

    def clear_ink(self) -> None:
        """Erase all freehand right-button strokes on the current page."""
        self._strokes = []
        self._active_ink = None
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

        hint = QLabel("右键拖动画任意线；点「发送」把整页图片（含标注）发给 AI（滚轮缩放）。")
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
        clear_ink_btn = QPushButton("清除标注")
        clear_ink_btn.clicked.connect(self.canvas.clear_ink)
        send_row = QHBoxLayout()
        send_row.addStretch()
        send_row.addWidget(clear_ink_btn)
        send_row.addWidget(self._send_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(zoom_row)
        layout.addLayout(nav_row)
        layout.addLayout(send_row)
        self.resize(_PREVIEW_DEFAULT_W, _PREVIEW_DEFAULT_H)

    def show_png(self, png: bytes, reset_geometry: bool = True) -> None:
        """Display a page PNG; reset the selection and (by default) the size/zoom.

        A fresh popup resets to the default size and fit view; an in-place refresh
        (prev/next/jump inside the window, ``reset_geometry=False``) keeps the user's
        resized window and zoom so flipping pages does not throw them back to the
        default view.
        """
        self._png = png
        qimg = QImage.fromData(png)
        self.canvas.set_image(QPixmap.fromImage(qimg), keep_view=not reset_geometry)
        self._selected = None
        # No region selection anymore — the page itself is what gets sent.
        self._send_btn.setEnabled(True)
        if reset_geometry:
            # Every popup resets to the default size (the user may have resized it).
            self.resize(_PREVIEW_DEFAULT_W, _PREVIEW_DEFAULT_H)
        self.show()
        self.raise_()

    @staticmethod
    def _virtual_desktop() -> QRect:
        """The union of all screens' geometries (a spanning multi-monitor desktop)."""
        geo = QRect()
        for scr in QGuiApplication.screens():
            geo = geo.united(scr.geometry())
        return geo

    def place_left_of(self, anchor: QRect) -> None:
        """Place this window so its right edge abuts the left edge of ``anchor``.

        ``anchor`` is the geometry of the window to sit next to (e.g. the main
        window's ``frameGeometry()``).  The preview is **vertically centred** on the
        anchor (its 1/2-height aligns with the anchor's 1/2-height), so it sits
        raised/centred on screen rather than pinned to the asset's top.  The window
        is clamped inside the virtual desktop on BOTH axes — the old placement only
        clamped x ≥ 0 and never touched y — and when there is no room on the left
        (the anchor already hugs the left screen edge) the preview is placed to the
        RIGHT of the anchor instead of sailing off-screen.
        """
        win = self.frameGeometry()
        w = win.width() or self.width()
        h = win.height() or self.height()
        desktop = self._virtual_desktop()
        x = anchor.left() - w
        y = anchor.top() + (anchor.height() - h) // 2   # vertical-centre on the anchor
        if x < desktop.left():          # no room on the left → sit on the right
            x = anchor.right() + 1
        # Clamp both axes inside the virtual desktop (never partially off-screen).
        max_x = desktop.left() + max(0, desktop.width() - w)
        max_y = desktop.top() + max(0, desktop.height() - h)
        x = min(max(x, desktop.left()), max_x)
        y = min(max(y, desktop.top()), max_y)
        self.move(x, y)

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

    def _send(self) -> None:
        if self._png is None:
            return
        # No region selection anymore — send the whole page (with any annotations).
        img = QImage.fromData(self._png, "PNG")
        if img.isNull():
            return
        rect = self._selected or [0.0, 0.0, float(img.width()), float(img.height())]
        # Include the freehand right-button marker strokes drawn on the page, so the
        # image handed to the AI carries the user's annotations too.
        strokes = self._ink_strokes()
        if strokes:
            cropped = self._crop_with_ink(self._png, rect, strokes)
        else:
            cropped = crop_region(self._png, rect)
        if not cropped:
            return
        # Convert the region from image pixels to PDF points (the block coordinate
        # space) so ``apply_annotation`` can match it to a block.
        s = 72.0 / self._dpi
        pdf_rect = [float(v) * s for v in rect]
        self.sendRequested.emit(cropped, pdf_rect)

    def _ink_strokes(self) -> list[list[QPointF]]:
        """All freehand strokes (finished + any in-progress) in image-pixel coords."""
        strokes = list(self.canvas._strokes)
        if self.canvas._active_ink:
            strokes.append(self.canvas._active_ink)
        return strokes

    def _crop_with_ink(self, png: bytes, rect, strokes: list[list[QPointF]]) -> bytes:
        """Crop ``rect`` (image px) from ``png`` and draw the marker strokes onto it.

        Strokes are (image-pixel) point lists; they are shifted by the crop origin and
        clipped to the crop, so the sent image carries the user's hand-drawn lines.
        """
        img = QImage.fromData(png, "PNG")
        if img.isNull():
            return crop_region(png, rect)
        x0 = max(0, min(img.width(), float(rect[0])))
        y0 = max(0, min(img.height(), float(rect[1])))
        x1 = max(x0, min(img.width(), float(rect[2])))
        y1 = max(y0, min(img.height(), float(rect[3])))
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            return crop_region(png, rect)
        cropped = img.copy(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        p = QPainter(cropped)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(255, 0, 0, 200), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        for stroke in strokes:
            pts = [QPointF(ip.x() - x0, ip.y() - y0) for ip in stroke if ip is not None]
            poly = QPolygonF(pts)
            if poly.count() > 1:
                p.drawPolyline(poly)
        p.end()
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        cropped.save(buf, "PNG")
        data = bytes(buf.data())
        buf.close()
        return data


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
    #: Thread-safe channel for the chat AI's ``re_export`` tool: emitted from the
    #: chat worker thread, queued to the GUI where it triggers the re-export.
    reExportRequested = pyqtSignal()
    #: Thread-safe channels for the chat AI's translate-entry tools: emitted from
    #: the chat worker thread, queued to the GUI where they trigger the pipeline
    #: start (with an optional user requirement) / a setting change.
    translateRequested = pyqtSignal(str)        # requirement
    setSettingRequested = pyqtSignal(str, object)  # key, value

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
