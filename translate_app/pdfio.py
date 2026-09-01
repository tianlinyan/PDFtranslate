"""PDF reading and export utilities built on PyMuPDF.

* :func:`extract_document_text` pulls the text out of a PDF in reading order
  (column by column) and captures per-block layout hints (font size, bold,
  alignment, line count).
* :func:`save_markdown` / :func:`save_plain_text` write the translation to file.
* :func:`save_interleaved_pdf` builds a bilingual PDF in which every original
  page is followed by a translation page mirroring the original layout.
* :func:`save_translated_pdf` redacts the original text and redraws the
  translation at the same positions, sizes and alignments.

CJK-capable text is rendered with PyMuPDF's bundled ``cjk`` font so translations
into Chinese / Japanese / Korean display correctly in the exported PDF.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import hashlib
import os
import re
import tempfile
import threading
from typing import Callable, Sequence

import pymupdf as fitz

#: Cancellation signal raised by the extractor (OCR) so the worker can treat a
#: cancelled OCR pass exactly like a cancelled translation.
from .translator import TranslationCancelled

#: Font used for rendered text (covers CJK plus Latin).
try:
    _CJK_FONT = fitz.Font("cjk")
except Exception:  # pragma: no cover - fall back to built-in helv
    _CJK_FONT = fitz.Font("helv")


@dataclass
class Block:
    """A single text block extracted from a page.

    Besides the text and its bbox, a block carries the layout hints needed to
    redraw its translation in place: the original font size (median over the
    block's spans), horizontal alignment, whether the original text was bold,
    and whether it consisted of a single line.
    """

    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    size: float = 10.0
    align: str = "left"          # left / center / right
    bold: bool = False
    single_line: bool = True
    #: True when this block was recovered by OCR from a scan.  Such blocks sit
    #: on top of a raster image rather than a text layer, so the in-place
    #: exporter must first cover the original pixels instead of redacting text.
    ocr: bool = False


@dataclass
class DocumentText:
    """All extractable text from a PDF, in reading order."""

    pages: list[list[Block]] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)     # flat, reading order
    block_pages: list[int] = field(default_factory=list)  # page index per block
    title: str = ""
    ocr_count: int = 0          # pages whose text came from OCR (was scanned)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def extract_document_text(
    path: str | Path,
    title: str | None = None,
    ocr: bool = False,
    ocr_fn: Callable[[int, "fitz.Page"], list[tuple[list, str]]] | None = None,
    cancel: Callable[[], bool] | None = None,
    log: Callable[[str], None] | None = None,
) -> DocumentText:
    """Extract text blocks from ``path`` in reading order.

    Column layouts are read column by column (left column top to bottom, then
    the next).  Span-level layout hints (font size, bold, alignment, line
    count) are captured per block so the exporters can redraw translations at
    the original positions without re-parsing the page.

    Blocks are rebuilt from individual visual lines rather than from PyMuPDF's
    own ``blocks`` output, because PyMuPDF merges several sibling lines (list
    items, table rows, section titles) into a single block; a translation drawn
    over such a merged block collapses the original line structure into one
    run-on paragraph.  Lines that begin a list / table entry (bullet, number or
    ``Label:`` style) or that change style (size / bold / colour) start a new
    block, so tables, bullet and numbered lists keep one entry per line, while
    lines of a flowing paragraph stay merged and translate as a whole.

    When ``ocr`` is enabled, pages that carry no embedded text layer (scans) are
    recognised with RapidOCR (its default model auto-detects Chinese + English,
    i.e. it follows the original text's language) and injected into the same
    block pipeline, so the extracted ``blocks`` / ``block_pages`` / ``pages``
    keep the exact layout the exporters expect.  ``ocr_fn`` is an injectable
    OCR callback ``(page_index, page) -> [(box, text)]`` (box already in PDF
    points) used by tests to avoid running real OCR; when it is ``None`` the
    shared RapidOCR engine is used and per-page results are cached per document
    (keyed by file mtime + size, so an edited PDF is re-OCR'd).  ``cancel`` is
    polled per page (raising :class:`TranslateCancelled`), and ``log`` receives
    per-page OCR progress.
    """
    doc = fitz.open(str(path))
    result = DocumentText(title=title or Path(path).stem)
    ocr_cache: dict[int, list[dict]] = {}
    ocr_cache_path: Path | None = None
    if ocr and ocr_fn is None:
        ocr_cache_path = _ocr_cache_path(path)
        ocr_cache = _load_ocr_cache(ocr_cache_path)
    try:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            lines = _collect_lines(page)
            if not lines:
                page_blocks: list[Block] = []
                if ocr and (ocr_fn is not None or _needs_ocr(page)):
                    if page_index in ocr_cache:
                        page_blocks = [_block_from_dict(d) for d in ocr_cache[page_index]]
                    else:
                        if log:
                            log(f"  OCR 第 {page_index + 1}/{doc.page_count} 页…")
                        if ocr_fn is None and _get_ocr_engine() is None:
                            if not _OCR_WARNED:
                                _OCR_WARNED = True
                                if log:
                                    log(
                                        "  OCR 引擎不可用：未安装 rapidocr_onnxruntime，"
                                        "扫描页将不识别。"
                                    )
                            page_blocks = []
                        else:
                            page_blocks = _ocr_page_blocks(page_index, page, ocr_fn, cancel)
                            if page_blocks:
                                ocr_cache[page_index] = [
                                    _block_to_dict(b) for b in page_blocks
                                ]
                if page_blocks:
                    for b in page_blocks:
                        result.blocks.append(b.text)
                        result.block_pages.append(page_index)
                    result.pages.append(page_blocks)
                    result.ocr_count += 1
                    continue
                result.pages.append([])
                continue

            spans = _collect_spans(page)
            page_x0 = min(ln["x0"] for ln in lines)
            page_x1 = max(ln["x1"] for ln in lines)
            page_blocks: list[Block] = []
            for group in _group_lines(_order_lines(lines)):
                if not group:
                    continue
                x0 = min(ln["x0"] for ln in group)
                y0 = min(ln["y0"] for ln in group)
                x1 = max(ln["x1"] for ln in group)
                y1 = max(ln["y1"] for ln in group)
                text = " ".join(ln["text"] for ln in group)
                if not text.strip():
                    continue
                meta = _block_meta(fitz.Rect(x0, y0, x1, y1), spans, page_x0, page_x1)
                block = Block(
                    text=text, page=page_index, x0=x0, y0=y0, x1=x1, y1=y1,
                    **meta,
                )
                page_blocks.append(block)
                result.blocks.append(block.text)
                result.block_pages.append(page_index)
            result.pages.append(page_blocks)

        if ocr_cache_path is not None and ocr_cache:
            _save_ocr_cache(ocr_cache_path, ocr_cache)
    finally:
        doc.close()
    return result


# ---------------------------------------------------------------------------
# OCR for scanned / image-only pages (RapidOCR, lazily loaded)
# ---------------------------------------------------------------------------

#: Lazily-created RapidOCR engine (``None`` = not yet loaded, ``False`` = failed).
_OCR_ENGINE: object | None = None
_OCR_FAILED = False
_OCR_WARNED = False
_OCR_LOCK = threading.Lock()

#: Render DPI for OCR — high enough for readable text, not so high the model
#: works on huge images (RapidOCR upsamples internally anyway).
_OCR_DPI = 300.0

#: Control characters (C0/C1 + DEL) leaked by bad font encodings.
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _clean_text(text: str) -> str:
    """Map control characters to spaces and collapse whitespace."""
    return " ".join(_CTRL_RE.sub(" ", str(text)).split())


def _get_ocr_engine():
    """Return a shared RapidOCR engine, or ``None`` if unavailable (degrade).

    Loading is guarded by a lock; a failure is cached (``_OCR_FAILED``) so the
    import is not retried on every page.  The caller may warn once (via its
    ``log`` callback) when this returns ``None``.
    """
    global _OCR_ENGINE, _OCR_FAILED
    if _OCR_FAILED:
        return None
    if _OCR_ENGINE is None:
        with _OCR_LOCK:
            if _OCR_ENGINE is None and not _OCR_FAILED:
                try:
                    from rapidocr_onnxruntime import RapidOCR

                    _OCR_ENGINE = RapidOCR()
                except Exception:
                    _OCR_FAILED = True
                    return None
    return _OCR_ENGINE


def _ocr_cache_dir() -> Path:
    """A writable dir for OCR results (OCR output is slow — reuse on re-run).

    ``PDFTRANSLATE_OCR_CACHE_DIR`` overrides the location; the default is the
    user's ``~/.pdftranslate/ocr_cache``.
    """
    override = os.environ.get("PDFTRANSLATE_OCR_CACHE_DIR")
    if override:
        try:
            p = Path(override)
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass
    for base in (
        Path.home() / ".pdftranslate" / "ocr_cache",
        Path(tempfile.gettempdir()) / "pdftranslate_ocr_cache",
    ):
        try:
            base.mkdir(parents=True, exist_ok=True)
            return base
        except Exception:
            continue
    return Path.home() / ".pdftranslate" / "ocr_cache"


def _ocr_cache_path(doc_path: str | Path) -> Path:
    """Cache file for a document's OCR results.

    The key includes the file's mtime and size, not just its path: reusing OCR
    results from a *replaced or edited* PDF would silently translate stale
    content.  (The translation cache needs no such stamp — it looks blocks up
    by content hash.)
    """
    p = Path(doc_path)
    try:
        st = p.stat()
        stamp = f"|{int(st.st_mtime)}|{st.st_size}"
    except OSError:
        stamp = ""
    h = hashlib.sha1(f"{p.resolve()}{stamp}".encode("utf-8")).hexdigest()[:16]
    return _ocr_cache_dir() / f"ocr_{h}.json"


def _load_ocr_cache(cache_path: Path) -> dict[int, list[dict]]:
    """Load per-page OCR block dicts (``{page_index: [block_dict]}``)."""
    if not cache_path.exists():
        return {}
    try:
        import json

        raw = json.loads(cache_path.read_text("utf-8"))
        return {int(k): v for k, v in raw.items()}
    except Exception:
        return {}


def _save_ocr_cache(cache_path: Path, data: dict[int, list[dict]]) -> None:
    """Best-effort persist of OCR results (OCR is slow — reuse on re-run)."""
    try:
        import json

        cache_path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    except Exception:
        pass


def _page_to_array(page) -> tuple[object, float]:
    """Render a page to a BGR numpy array plus the pixel-per-point zoom."""
    import numpy as np

    zoom = _OCR_DPI / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom), alpha=False, colorspace=fitz.csRGB
    )
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        img = img[:, :, :3][:, :, ::-1]  # RGB -> BGR (RapidOCR / OpenCV convention)
    return img, zoom


def _block_to_dict(b: Block) -> dict:
    return asdict(b)


def _block_from_dict(data: dict) -> Block:
    allowed = {f.name for f in fields(Block)}
    return Block(**{k: v for k, v in data.items() if k in allowed})


def _synthesize_ocr_blocks(
    results: Sequence[tuple[list, str]], page_index: int
) -> list[Block]:
    """Turn ``[(box, text), ...]`` (box already in PDF points) into blocks.

    Text is cleaned, ordered with the same column-aware reading order as native
    text, and given a font size estimated from the box height.
    """
    items: list[tuple] = []
    for box, text in results:
        cleaned = _clean_text(text)
        if not cleaned:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        items.append((min(ys), min(xs), max(xs), max(ys), cleaned))
    blocks: list[Block] = []
    for y0, x0, x1, y1, text in _order_blocks(items):
        size = min(_MAX_FONT, max(5.0, (y1 - y0) / 1.2))
        blocks.append(
            Block(
                text=text, page=page_index, x0=x0, y0=y0, x1=x1, y1=y1,
                size=round(size, 2), align="left", bold=False, single_line=True,
                ocr=True,
            )
        )
    return blocks


def _needs_ocr(page) -> bool:
    """True when a page has no embedded text but is clearly not blank.

    A page that already exposes a text layer keeps using it.  A truly blank page
    (no images, no drawings) is skipped so the renderer is not wasted.  Scanned
    pages reach us as a full-page image, which ``get_images`` detects.
    """
    try:
        if page.get_images(full=True):
            return True
    except Exception:
        pass
    try:
        return bool(page.get_drawings())
    except Exception:
        return False


def _ocr_page_blocks(
    page_index: int,
    page,
    ocr_fn: Callable[[int, "fitz.Page"], list[tuple[list, str]]] | None,
    cancel: Callable[[], bool] | None,
) -> list[Block]:
    """OCR one page and return its blocks (empty if OCR is unavailable/failed).

    ``ocr_fn`` is injected by tests (returns ``[(box, text)]`` in PDF points);
    production falls back to the shared RapidOCR engine.  ``cancel`` is polled
    before the (potentially slow) render so a cancelled run does not start a
    page it will never use.
    """
    if cancel is not None and cancel():
        raise TranslationCancelled()
    if ocr_fn is not None:
        try:
            return _synthesize_ocr_blocks(list(ocr_fn(page_index, page)), page_index)
        except Exception:
            return []
    engine = _get_ocr_engine()
    if engine is None:
        return []
    try:
        img, zoom = _page_to_array(page)
        out = engine(img)
        # RapidOCR returns (list of [box, text, score] or None, timings).
        items = out[0] if isinstance(out, tuple) else out
        if not items:
            return []
        results: list[tuple[list, str]] = []
        for item in items:
            if not item or len(item) < 2:
                continue
            box = item[0]
            text = item[1]
            if not text:
                continue
            pdf_box = [[float(px) / zoom, float(py) / zoom] for px, py in box]
            results.append((pdf_box, text))
        return _synthesize_ocr_blocks(results, page_index)
    except Exception:
        return []


#: Characters treated as a leading bullet (e.g. ``❖``, ``•``, ``▪``).
_BULLET_CHARS = (
    "\u2022\u25aa\u25cf\u25a0\u25c6\u25c7\u2726\u2727\u2756\u2764\u27a1"
    "\u2192\u2190\u25b8\u25b6\u00bb\u2023\u2043\u2219\u00b7\u275a\u25d8\u25cb"
)
_BULLET_RE = re.compile(r"^\s*[" + re.escape(_BULLET_CHARS) + r"]\s*")
#: A dash/asterisk used as a bullet only when followed by whitespace (so a
#: leading hyphenated word is not mistaken for a list marker).
_DASH_BULLET_RE = re.compile(r"^\s*(?:\*\s+|-\s+|\u2013\s+|\u2014\s+)")
#: Leading number of a numbered-list item, e.g. ``1.``, ``27)``.
_NUM_RE = re.compile(r"^\s*\d{1,3}[.)]\s*")
#: A short ``Label:`` row / heading that begins a table entry.
_LABEL_RE = re.compile(r"^\s*[A-Za-z\u4e00-\u9fff\u4e00-\u9fa5][^:\n]{0,40}:\s")


def _is_entry(text: str) -> bool:
    """True when ``text`` begins a new list / table entry (bullet, number, ``Label:``)."""
    t = text.strip()
    if not t:
        return False
    return bool(
        _BULLET_RE.match(t) or _DASH_BULLET_RE.match(t) or _NUM_RE.match(t) or _LABEL_RE.match(t)
    )


def _collect_lines(page) -> list[dict]:
    """Return one record per visual line (a PyMuPDF ``line``), with bbox and
    style hints used to decide paragraph vs. entry grouping."""
    out: list[dict] = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = " ".join("".join(s["text"] for s in spans).split())
            if not text:
                continue
            sizes = sorted(float(s.get("size", 10.0)) for s in spans)
            size = sizes[len(sizes) // 2]
            bold = any(
                bool(int(s.get("flags", 0)) & 16) or "bold" in str(s.get("font", "")).lower()
                for s in spans
            )
            colors = [int(s.get("color", 0)) for s in spans]
            color = Counter(colors).most_common(1)[0][0]
            out.append(
                {
                    "x0": min(s["bbox"][0] for s in spans),
                    "y0": min(s["bbox"][1] for s in spans),
                    "x1": max(s["bbox"][2] for s in spans),
                    "y1": max(s["bbox"][3] for s in spans),
                    "size": size,
                    "bold": bold,
                    "color": color,
                    "text": text,
                }
            )
    return out


def _order_lines(lines: Sequence[dict]) -> list[dict]:
    """Order visual lines into reading order (column by column)."""
    xsorted = sorted(lines, key=lambda ln: ln["x0"])
    columns: list[list[dict]] = []
    col_max_x1: list[float] = []
    for ln in xsorted:
        for c in range(len(columns)):
            if ln["x0"] < col_max_x1[c] - 2.0:
                columns[c].append(ln)
                col_max_x1[c] = max(col_max_x1[c], ln["x1"])
                break
        else:
            columns.append([ln])
            col_max_x1.append(ln["x1"])

    def rows(col: Sequence[dict]) -> list[dict]:
        return sorted(col, key=lambda ln: (round(ln["y0"], 1), ln["x0"]))

    if len(columns) == 1:
        return rows(lines)
    ordered: list[dict] = []
    for col in columns:
        ordered.extend(rows(col))
    return ordered


def _group_lines(ordered: Sequence[dict]) -> list[list[dict]]:
    """Group ordered lines into blocks: split at entries / style changes /
    paragraph gaps, but keep the lines of a flowing paragraph together."""
    groups: list[list[dict]] = []
    i = 0
    n = len(ordered)
    while i < n:
        group = [ordered[i]]
        base = ordered[i]
        i += 1
        while i < n and not _break_between(base, group[-1], ordered[i]):
            group.append(ordered[i])
            i += 1
        groups.append(group)
    return groups


def _break_between(base: dict, prev: dict, cur: dict) -> bool:
    """True when ``cur`` starts a new block rather than joining the running one."""
    # Moving back up (e.g. the last line of one column followed by the first
    # line of the next column) is always a hard break.
    if cur["y0"] < prev["y0"] - 1.0:
        return True
    # Style jump (different size / bold / colour) starts a new entry: catches
    # section headings (e.g. gold sub-titles) and emphasis changes.
    if (
        abs(cur["size"] - base["size"]) > 0.6
        or cur["bold"] != base["bold"]
        or cur["color"] != base["color"]
    ):
        return True
    # A line that begins a list / table entry stands alone.
    if _is_entry(cur["text"]):
        return True
    # A vertical gap larger than half a line marks a paragraph / entry break.
    if cur["y0"] - prev["y1"] > 0.5 * base["size"]:
        return True
    return False


def _collect_spans(page: fitz.Page) -> list[tuple[fitz.Rect, float, bool]]:
    """Return ``(bbox, size, bold)`` for every text span on the page."""
    spans: list[tuple[fitz.Rect, float, bool]] = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for s in line.get("spans", []):
                rect = fitz.Rect(s["bbox"])
                size = float(s.get("size", 10.0))
                flags = int(s.get("flags", 0))
                font = str(s.get("font", ""))
                bold = bool(flags & 16) or "bold" in font.lower()
                spans.append((rect, size, bold))
    return spans


def _order_blocks(text_blocks: Sequence[tuple]) -> list[tuple]:
    """Sort blocks into reading order.

    Blocks whose horizontal ranges overlap are clustered into columns.  A
    multi-column page is read column by column (left column top to bottom,
    then the next column) instead of interleaving the columns row by row;
    single-column pages keep the plain top-to-bottom, left-to-right order.
    """
    xsorted = sorted(text_blocks, key=lambda b: b[1])
    columns: list[list[tuple]] = []
    col_max_x1: list[float] = []
    for b in xsorted:
        for c in range(len(columns)):
            if b[1] < col_max_x1[c] - 2.0:
                columns[c].append(b)
                col_max_x1[c] = max(col_max_x1[c], b[2])
                break
        else:
            columns.append([b])
            col_max_x1.append(b[2])

    def rows(col: Sequence[tuple]) -> list[tuple]:
        return sorted(col, key=lambda b: (round(b[0], 1), b[1]))

    if len(columns) == 1:
        return rows(text_blocks)
    out: list[tuple] = []
    for col in columns:
        out.extend(rows(col))
    return out


def _block_meta(
    rect: fitz.Rect,
    spans: Sequence[tuple[fitz.Rect, float, bool]],
    page_x0: float,
    page_x1: float,
) -> dict:
    """Layout hints for one block: size, alignment, bold, single-line."""
    sizes: list[float] = []
    bold = False
    minx = maxx = None
    per_line: dict[float, list[float]] = {}  # line y -> [min_x0, max_x1]
    for srect, size, is_bold in spans:
        if not srect.intersects(rect):
            continue
        sizes.append(size)
        bold = bold or is_bold
        minx = srect.x0 if minx is None else min(minx, srect.x0)
        maxx = srect.x1 if maxx is None else max(maxx, srect.x1)
        key = round(srect.y0, 1)
        if key not in per_line:
            per_line[key] = [srect.x0, srect.x1]
        else:
            per_line[key][0] = min(per_line[key][0], srect.x0)
            per_line[key][1] = max(per_line[key][1], srect.x1)
    if sizes:
        sizes.sort()
        size = sizes[len(sizes) // 2]  # median: robust against stray glyphs
    else:
        size = 10.0
    align = "left"
    if minx is not None:
        left_gap = minx - rect.x0
        right_gap = rect.x1 - maxx
        centered = (
            left_gap > 2
            and right_gap > 2
            and abs((minx + maxx) / 2 - (rect.x0 + rect.x1) / 2)
            < max(2.0, rect.width * 0.1)
        )
        if centered:
            align = "center"
        elif len(per_line) >= 2:
            # Multi-line: right-aligned lines share a flush right edge while
            # their left edges vary.
            line_x0 = [v[0] for v in per_line.values()]
            line_x1 = [v[1] for v in per_line.values()]
            if max(line_x1) - min(line_x1) <= 2.0 and max(line_x0) - min(line_x0) > 2.0:
                align = "right"
        else:
            # Single line: right-aligned page numbers etc. hug the page's
            # rightmost text edge and start in the right half of the text area.
            if page_x1 - maxx <= 2.0 and minx - page_x0 > (page_x1 - page_x0) / 2:
                align = "right"
    single_line = rect.height <= 1.5 * size
    return {"size": size, "align": align, "bold": bold, "single_line": single_line}


def group_by_page(block_pages: Sequence[int], values: Sequence[str], page_count: int) -> list[list[str]]:
    """Regroup a flat ``values`` list back into per-page lists."""
    per_page: list[list[str]] = [[] for _ in range(page_count)]
    for page, value in zip(block_pages, values):
        per_page[page].append(value)
    return per_page


# ---------------------------------------------------------------------------
# Plain text / Markdown export
# ---------------------------------------------------------------------------

def save_plain_text(per_page: Sequence[Sequence[str]], out_path: str | Path) -> None:
    """Write translated page text (only) to a .txt file."""
    lines: list[str] = []
    for i, blocks in enumerate(per_page):
        lines.append(f"===== Page {i + 1} =====")
        lines.extend(b for b in blocks if b)
        lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def save_markdown(
    per_page: Sequence[Sequence[str]],
    source_blocks: Sequence[str],
    block_pages: Sequence[int],
    out_path: str | Path,
    lang: str,
    title: str = "",
) -> None:
    """Write a bilingual Markdown file: translation first, original in a quote."""
    out: list[str] = []
    out.append(f"# {title or 'Translated Document'}")
    out.append("")
    out.append(f"> 译文语言 / Target language: **{lang}**")
    out.append("")

    per_page_src = group_by_page(block_pages, source_blocks, len(per_page))
    for i, blocks in enumerate(per_page):
        out.append(f"## Page {i + 1}")
        out.append("")
        for text in blocks:
            if text:
                out.append(text)
                out.append("")
        # Original text, quoted.
        src = [b for b in per_page_src[i] if b]
        if src:
            out.append("")
            out.append("<details>")
            out.append("<summary>原文 / Original</summary>")
            out.append("")
            for text in src:
                out.append(f"> {text}")
                out.append("")
            out.append("</details>")
            out.append("")

    Path(out_path).write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------------------
# Bilingual PDF export
# ---------------------------------------------------------------------------

_MARGIN = 42.0
_FONT_SIZE = 11.0

#: Largest font size a translation may start at (original headings are capped
#: here; the shrink loop below keeps every block inside its box).
_MAX_FONT = 24.0


def _render_note(page: fitz.Page, font, lang: str) -> None:
    """Write the 'nothing to translate on this page' note."""
    note = "（本页无可翻译文本 / No translatable text on this page）"
    tw = fitz.TextWriter(page.rect)
    tw.append(fitz.Point(_MARGIN, _MARGIN + 14), note, font=font, fontsize=_FONT_SIZE)
    tw.write_text(page)


def _draw_translated_block(page: fitz.Page, font, block: Block, text: str) -> None:
    """Draw ``text`` into ``block``'s box, mirroring the original layout.

    The glyph box (ascent + lines + descent) is anchored to the block's bbox
    top; the font size starts at the block's original size and is trimmed
    until the wrapped text fits the box height.  Single-line boxes centre the
    translation vertically while paragraph blocks stay top-anchored like the
    source.
    """
    r = fitz.Rect(block.x0, block.y0, block.x1, block.y1)
    max_width = max(1.0, r.width)
    fs = max(5.0, min(block.size, _MAX_FONT))
    lines = _wrap(font, text, max_width, fs)
    ascent = fs * font.ascender
    descent = -fs * font.descender

    def height() -> float:
        return ascent + (len(lines) - 1) * fs * 1.35 + descent

    # Trim the font so the translation's height no longer exceeds the box it
    # replaces (a smaller font also wraps to fewer lines).  This keeps dense
    # tables / closely-spaced blocks from overlapping their neighbours.  The
    # 3pt floor guarantees the no-overflow invariant even for pathological
    # translations; such text is unreadable either way.
    while fs > 3.0 and height() > r.height + 1.0:
        fs = round(fs * 0.9, 2)
        lines = _wrap(font, text, max_width, fs)
        ascent = fs * font.ascender
        descent = -fs * font.descender

    y = r.y0 + ascent
    if block.single_line:
        y = r.y0 + max(0.0, (r.height - height()) / 2) + ascent
    # Every block renders with the same CJK font.  Bold is intentionally NOT
    # simulated: the bundled font has no bold face, and mixing a second font
    # (e.g. the stroke-rendered "china-s") made pages visibly inconsistent.
    tw = fitz.TextWriter(page.rect)
    for line in lines:
        x = r.x0
        if block.align == "center":
            lw = font.text_length(line, fontsize=fs)
            x = max(r.x0, r.x0 + (r.width - lw) / 2)
        elif block.align == "right":
            lw = font.text_length(line, fontsize=fs)
            x = max(r.x0, r.x1 - lw)
        tw.append(fitz.Point(x, y), line, font=font, fontsize=fs)
        y += fs * 1.35
    tw.write_text(page)


def _wrap(font, text: str, width: float, fontsize: float) -> list[str]:
    """Greedy word-wrap that also breaks long words (e.g. CJK) by character.

    ``width`` is the maximum allowed line width.  Words are kept intact where
    possible; a single word that is wider than ``width`` (typical for CJK text,
    which has no spaces, so a whole paragraph is one giant "word") is broken into
    character pieces so that long translations never overflow the page.
    """
    lines: list[str] = []
    for paragraph_line in str(text).split("\n"):
        current = ""
        for word in paragraph_line.split(" "):
            if not word:
                continue  # consecutive/leading spaces: nothing to add
            if not current:
                # First token on a line.  If it fits, keep it whole; otherwise
                # it must be broken into character pieces (CJK / long words).
                if font.text_length(word, fontsize=fontsize) <= width:
                    current = word
                else:
                    current = _break_word(font, word, width, fontsize, lines)
                continue
            probe = f"{current} {word}".strip()
            if font.text_length(probe, fontsize=fontsize) <= width:
                current = probe
            else:
                lines.append(current)
                if font.text_length(word, fontsize=fontsize) <= width:
                    current = word
                else:
                    current = _break_word(font, word, width, fontsize, lines)
        if current:
            lines.append(current)
    return lines if lines else [str(text)]


def _break_word(font, word: str, width: float, fontsize: float, lines: list[str]) -> str:
    """Split ``word`` into pieces that each fit within ``width``.

    Every piece except the trailing one (which becomes the new current line) is
    appended to ``lines``.  Returns the trailing piece, or ``""`` if ``word``
    was consumed exactly.
    """
    while word and font.text_length(word, fontsize=fontsize) > width:
        lo, hi = 1, len(word)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if font.text_length(word[:mid], fontsize=fontsize) <= width:
                lo = mid
            else:
                hi = mid - 1
        lines.append(word[:lo])
        word = word[lo:]
    return word


def save_interleaved_pdf(
    src_path: str | Path,
    per_page: Sequence[Sequence[str]],
    out_path: str | Path,
    lang: str,
    pages: Sequence[Sequence[Block]] | None = None,
) -> None:
    """Create a bilingual PDF: each original page followed by a translation
    page that mirrors the original layout (every translated block sits at its
    source block's position)."""
    src = fitz.open(str(src_path))
    new_doc = fitz.open()
    try:
        font = _CJK_FONT
        for i in range(src.page_count):
            new_doc.insert_pdf(src, from_page=i, to_page=i)
            page_rect = src[i].rect
            tpage = new_doc.new_page(width=page_rect.width, height=page_rect.height)
            blocks = pages[i] if pages is not None and i < len(pages) else []
            trans = per_page[i] if i < len(per_page) else []
            m = min(len(blocks), len(trans))
            if m == 0:
                _render_note(tpage, font, lang)
                continue
            for j in range(m):
                _draw_translated_block(tpage, font, blocks[j], trans[j])
        new_doc.set_metadata({"title": "Bilingual translation", "creator": "PDFtranslate"})
        new_doc.save(str(out_path), garbage=4, deflate=True)
    finally:
        new_doc.close()
        src.close()


def save_translated_pdf(
    src_path: str | Path,
    pages: Sequence[Sequence[Block]],
    per_page: Sequence[Sequence[str]],
    out_path: str | Path,
    lang: str,
) -> None:
    """Create a layout-preserving translation PDF.

    Every output page keeps the original page (all images, photos, drawings and
    vector graphics in their exact places), while the original text is redacted
    and replaced by the translated text at the same positions.  This is the
    ``仅译文 / translation in place`` output.
    """
    src = fitz.open(str(src_path))
    out_doc = fitz.open()
    try:
        font = _CJK_FONT
        n = min(src.page_count, len(per_page))
        for i in range(n):
            out_doc.insert_pdf(src, from_page=i, to_page=i)
            page = out_doc[-1]
            blocks = pages[i] if i < len(pages) else []
            trans = per_page[i]
            m = min(len(blocks), len(trans))
            if m == 0:
                continue

            # Remove the original text (keep images and line art/graphics).
            # OCR blocks sit on a raster image rather than a text layer, so
            # nothing is redacted for them — they are covered below instead.
            for j in range(m):
                b = blocks[j]
                if not b.ocr:
                    page.add_redact_annot(fitz.Rect(b.x0, b.y0, b.x1, b.y1))
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            )

            # Draw the translation at the original positions / alignment /
            # font size (see ``_draw_translated_block`` for the fitting rules).
            for j in range(m):
                b = blocks[j]
                if b.ocr:
                    # Cover the underlying scan pixels so the translation does
                    # not overprint the original (raster) text.
                    page.draw_rect(
                        fitz.Rect(b.x0 - 0.5, b.y0 - 0.5, b.x1 + 0.5, b.y1 + 0.5),
                        color=None,
                        fill=(1, 1, 1),
                    )
                _draw_translated_block(page, font, b, trans[j])

        out_doc.set_metadata({"title": "Translated text", "creator": "PDFtranslate"})
        out_doc.save(str(out_path), garbage=4, deflate=True)
    finally:
        out_doc.close()
        src.close()
