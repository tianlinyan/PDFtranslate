"""Shared helpers for the tests: a mock chat-completions server and a sample PDF."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pymupdf as fitz


class _MockHandler(BaseHTTPRequestHandler):
    """Echoes back ``[n] MOCK:<source-text>`` for every numbered block."""

    last_body: dict | None = None

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).last_body = body
        user = ""
        for m in body.get("messages", []):
            if m.get("role") == "user":
                user = m.get("content", "")
                break
        lines = []
        for chunk in str(user).split("\n\n"):
            chunk = chunk.strip()
            if not chunk or "]" not in chunk.splitlines()[0]:
                continue
            first = chunk.splitlines()[0].strip("[]")
            text = chunk.split("\n", 1)[1].strip() if "\n" in chunk else ""
            lines.append(f"[[{first}]] MOCK:{text}")
        content = "\n".join(lines) or "[[1]] MOCK:"
        data = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # noqa: D102
        pass


class MockServer:
    """Runs a local chat-completions mock server in a background thread."""

    def __init__(self):
        self._server = HTTPServer(("127.0.0.1", 0), _MockHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1/chat/completions"

    @property
    def last_body(self) -> dict | None:
        return _MockHandler.last_body

    def __enter__(self):
        _MockHandler.last_body = None
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=3)


def build_sample_pdf(path: str | Path, pages: int = 2) -> Path:
    """Create a small PDF with English and CJK text."""
    path = Path(path)
    doc = fitz.open()
    font = fitz.Font("cjk")
    for p in range(pages):
        page = doc.new_page()
        page.insert_text((72, 80), f"Page {p + 1} heading text.", fontsize=12)
        page.insert_text((72, 120), "This is a sample body paragraph to translate.", fontsize=12)
        tw = fitz.TextWriter(page.rect)
        tw.append(fitz.Point(72, 170), "这是一个测试文档。", font=font, fontsize=12)
        tw.write_text(page)
    doc.save(str(path))
    doc.close()
    return path


def build_two_column_pdf(path: str | Path) -> Path:
    """Create a PDF with two text columns and a right-aligned line."""
    path = Path(path)
    doc = fitz.open()
    page = doc.new_page()  # A4: 595 x 842
    # Left column (x ~60..215) and right column (x ~315..480).
    page.insert_text((60, 80), "Left column first line.", fontsize=12)
    page.insert_text((60, 110), "Left column second line.", fontsize=12)
    page.insert_text((60, 140), "Left column third line.", fontsize=12)
    page.insert_text((315, 80), "Right column first line.", fontsize=12)
    page.insert_text((315, 110), "Right column second line.", fontsize=12)
    # Right-aligned footer line hugging the page's right margin.
    right_text = "Page 1 of 9"
    helv = fitz.Font("helv")
    page.insert_text(
        (page.rect.x1 - 60 - helv.text_length(right_text, fontsize=12), 200),
        right_text, fontsize=12,
    )
    doc.save(str(path))
    doc.close()
    return path


def build_two_column_pdf_with_heading(path: str | Path) -> Path:
    """Create a PDF with a full-width heading above two text columns.

    Regression fixture for the column clustering: a heading spanning both
    columns used to become a "column" whose right edge was the page width, so
    every line of both real columns "overlapped" it and the two columns were
    merged into one (left and right lines interleaved by y).
    """
    path = Path(path)
    doc = fitz.open()
    page = doc.new_page()  # A4: 595 x 842
    # Full-width heading spanning both columns.
    page.insert_text(
        (60, 50), "FULL WIDTH HEADING ACROSS BOTH COLUMNS OF THE PAGE", fontsize=14
    )
    # Left column (x ~60..200) and right column (x ~315..460).
    page.insert_text((60, 100), "Left column first line.", fontsize=12)
    page.insert_text((60, 130), "Left column second line.", fontsize=12)
    page.insert_text((60, 160), "Left column third line.", fontsize=12)
    page.insert_text((315, 100), "Right column first line.", fontsize=12)
    page.insert_text((315, 130), "Right column second line.", fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def _build_two_column_pdf_with_wide_line(
    path: str | Path,
    wide_line: str,
    wide_size: float,
    wide_x: float,
) -> Path:
    """Two tightly-leaded column paragraphs plus one wide line above them.

    `wide_line` is deliberately wider than a column yet **narrower than the
    full-width threshold** (`max(1.5 x median width, 0.75 x span)`) — exactly
    the shape that used to collapse the page: it joined the left column and
    dragged that column's right edge past the right column's left edge, so every
    right-column line "overlapped" the left column and the two were interleaved
    by y into one block per line.
    """
    path = Path(path)
    doc = fitz.open()
    page = doc.new_page()  # A4: 595 x 842
    page.insert_text((wide_x, 50), wide_line, fontsize=wide_size)
    # Tight 12 pt leading on 10 pt type keeps each column ONE paragraph (the
    # grouping rule only breaks above 1.1 x size), so a collapsed page shows up
    # as a run of single-line blocks instead of two paragraph blocks.
    for i, text in enumerate(
        (
            "Left column first line of a single paragraph,",
            "left column second line of the same paragraph,",
            "left column third line of the same paragraph.",
        )
    ):
        page.insert_text((60, 100 + 12 * i), text, fontsize=10)
    for i, text in enumerate(
        (
            "Right column first line of a single paragraph,",
            "right column second line of the same paragraph,",
            "right column third line of the same paragraph.",
        )
    ):
        page.insert_text((315, 100 + 12 * i), text, fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def build_two_column_pdf_with_running_head(path: str | Path) -> Path:
    """Two columns under a *centred running head* that is not full width.

    Regression fixture (ICML-style paper): the running head covers ~0.6 of the
    page's text span, so it is neither a full-width heading nor a column line.
    It must not merge the two columns.
    """
    return _build_two_column_pdf_with_wide_line(
        path, "ANNUAL REPORT OF THE BANK RUNNING HEAD", 12, 150
    )


def build_two_column_pdf_with_author_line(path: str | Path) -> Path:
    """Two columns under a wide *author line* that is not full width.

    Regression fixture (ICML-style paper): the author list spans both columns
    ("Meimingwei Li * 1 Yuanhao Ding * 2 ..." on the real page) yet stays below
    the full-width threshold.  It must not merge the two columns.
    """
    return _build_two_column_pdf_with_wide_line(
        path, "A. Author * 1 B. Author * 2 C. Author * 3", 12, 118
    )


def build_list_table_pdf(path: str | Path) -> Path:
    """Create a PDF with a close-spaced numbered list and ``Label:`` table rows.

    The items are placed so close together that a block-level extraction would
    merge them all into one run-on block; the line-aware extraction must keep
    every item / table row as its own single-line block.
    """
    path = Path(path)
    doc = fitz.open()
    page = doc.new_page()
    # Numbered list, 15 pt apart (well inside a single block's bbox span).
    for i, text in enumerate(
        ["1. First item", "2. Second item", "3. Third item"], start=0
    ):
        page.insert_text((60, 80 + i * 15), text, fontsize=12)
    # Label: table rows.
    labels = [("Powerplant:", 140), ("Brand: Pratt & Whitney", 155),
              ("Model: PT6A-140", 170)]
    for text, y in labels:
        page.insert_text((60, y), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path
