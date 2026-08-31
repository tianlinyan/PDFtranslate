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

import hashlib
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Sequence

import pymupdf as fitz

_logger = logging.getLogger(__name__)

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
    ocr: bool = True,
    ocr_fn: Callable[[int, "fitz.Page"], list[tuple[list, str]]] | None = None,
) -> DocumentText:
    """Extract text blocks from ``path`` in reading order.

    Column layouts are read column by column (left column top to bottom, then
    the next).  Span-level layout hints (font size, bold, alignment, line
    count) are captured per block so the exporters can redraw translations at
    the original positions without re-parsing the page.

    Pages with **no text layer** (scanned / image-only) are OCR'd when ``ocr``
    is true (and RapidOCR is available), producing ``Block`` objects with real
    bounding boxes so the translation still honours the original layout.
    ``ocr_fn`` is an injectable OCR callback ``(page_index, page) -> [(box, text)]``
    used by tests to avoid running real OCR.
    """
    doc = fitz.open(str(path))
    result = DocumentText(title=title or Path(path).stem)
    # OCR results are cached per document so a re-run does not redo slow OCR.
    ocr_cache: dict[int, list[dict]] = {}
    ocr_cache_path: Path | None = None
    if ocr and ocr_fn is None:
        ocr_cache_path = _ocr_cache_path(path)
        ocr_cache = _load_ocr_cache(ocr_cache_path)
    try:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            raw = page.get_text("blocks")
            # Keep text blocks (type 0).
            text_blocks = [
                (b[1], b[0], b[2], b[3], b[4])  # y0, x0, x1, y1, text
                for b in raw
                if b[6] == 0 and str(b[4]).strip()
            ]
            if not text_blocks:
                if ocr and (ocr_fn is not None or _looks_scanned(page)):
                    if page_index in ocr_cache:
                        ocr_blocks = [_block_from_dict(d) for d in ocr_cache[page_index]]
                    else:
                        ocr_blocks = _ocr_page_blocks(page_index, page, ocr_fn)
                        if ocr_blocks:
                            ocr_cache[page_index] = [_block_to_dict(b) for b in ocr_blocks]
                    if ocr_blocks:
                        result.pages.append(ocr_blocks)
                        for block in ocr_blocks:
                            result.blocks.append(block.text)
                            result.block_pages.append(page_index)
                        result.ocr_count += 1
                        continue
                result.pages.append([])
                continue

            spans, page_lines = _collect_page_struct(page)
            # Horizontal extents of the page's text, used for right-align
            # detection of single-line blocks (page numbers, signatures).
            page_x0 = min(b[1] for b in text_blocks)
            page_x1 = max(b[2] for b in text_blocks)
            page_blocks: list[Block] = []
            for y0, x0, x1, y1, text in _order_blocks(text_blocks):
                cleaned = " ".join(str(text).split())
                if not cleaned:
                    continue
                rect = fitz.Rect(x0, y0, x1, y1)
                pieces = _split_list_lines(rect, cleaned, page_lines)
                if len(pieces) == 1:
                    # A normal paragraph / single line: keep it as one block.
                    meta = _block_meta(rect, spans, page_x0, page_x1)
                    block = Block(
                        text=cleaned, page=page_index,
                        x0=x0, y0=y0, x1=x1, y1=y1,
                        **meta,
                    )
                    page_blocks.append(block)
                    result.blocks.append(cleaned)
                    result.block_pages.append(page_index)
                else:
                    # A block PyMuPDF merged from several visually separate
                    # lines (e.g. a bullet list).  Emit one block per line so
                    # each line translates and redraws on its own — otherwise a
                    # list becomes a single flowing paragraph that no longer
                    # matches the original layout.
                    for line_rect, line_text in pieces:
                        meta = _block_meta(line_rect, spans, page_x0, page_x1)
                        block = Block(
                            text=line_text, page=page_index,
                            x0=line_rect.x0, y0=line_rect.y0,
                            x1=line_rect.x1, y1=line_rect.y1,
                            **meta,
                        )
                        page_blocks.append(block)
                        result.blocks.append(line_text)
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
_OCR_LOCK = threading.Lock()

#: Render pages at this DPI for OCR — high enough for readable text, not so high
#: the model it cranks on huge images (RapidOCR upsamples internally anyway).
_OCR_DPI = 300.0


def _ocr_cache_dir() -> Path:
    """A writable dir for OCR results (model output is usually slow to redo).

    ``PDFTRANSLATE_OCR_CACHE_DIR`` overrides the location; the default is the
    user's ``~/.pdftranslate/ocr_cache``.
    """
    override = os.environ.get("PDFTRANSLATE_OCR_CACHE_DIR")
    if override:
        try:
            p = Path(override)
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception as exc:  # noqa: BLE001
            _logger.warning("无法使用自定义 OCR 缓存目录 %s: %s", override, exc)
    for base in (
        Path.home() / ".pdftranslate" / "ocr_cache",
        Path(tempfile.gettempdir()) / "pdftranslate_ocr_cache",
    ):
        try:
            base.mkdir(parents=True, exist_ok=True)
            return base
        except Exception as exc:  # noqa: BLE001
            _logger.warning("OCR 缓存目录不可用 %s: %s", base, exc)
            continue
    return Path.home() / ".pdftranslate" / "ocr_cache"


def _ocr_cache_path(doc_path: str | Path) -> Path:
    """Cache file for a document's OCR results.

    The key includes the file's mtime and size, not just its path: reusing
    OCR results from a *replaced or edited* PDF would silently translate the
    old content.  (The translation cache needs no such stamp because it looks
    blocks up by content hash.)
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
    except Exception as exc:  # noqa: BLE001
        _logger.debug("读取 OCR 缓存失败 %s: %s", cache_path, exc)
        return {}


def _save_ocr_cache(cache_path: Path, data: dict[int, list[dict]]) -> None:
    """Best-effort persist of OCR results (OCR is slow — reuse on re-run)."""
    try:
        import json

        cache_path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    except Exception as exc:  # noqa: BLE001
        _logger.warning("保存 OCR 缓存失败: %s", exc)


def _ocr_engine():
    """Return a shared RapidOCR engine, or ``None`` if unavailable."""
    global _OCR_ENGINE, _OCR_FAILED
    if _OCR_FAILED:
        return None
    if _OCR_ENGINE is None:
        with _OCR_LOCK:
            if _OCR_ENGINE is None and not _OCR_FAILED:
                try:
                    from rapidocr_onnxruntime import RapidOCR

                    _OCR_ENGINE = RapidOCR()
                except Exception:  # noqa: BLE001
                    _OCR_FAILED = True
                    _logger.warning("无法加载 RapidOCR（扫描页将无法识别）")
                    return None
    return _OCR_ENGINE if _OCR_ENGINE is not False else None


def _page_to_array(page: "fitz.Page") -> tuple[object, float]:
    """Render a page to a BGR numpy array plus the pixel-per-point zoom."""
    import numpy as np

    zoom = _OCR_DPI / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom), alpha=False, colorspace=fitz.csRGB
    )
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n >= 3:
        img = img[:, :, :3][:, :, ::-1]  # RGB -> BGR
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

    Text is deduped/cleaned, ordered with the same column-aware reading order
    as native text, and given a font size estimated from the box height.
    """
    items: list[tuple] = []
    for box, text in results:
        cleaned = " ".join(str(text).split())
        if not cleaned:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        items.append((min(ys), min(xs), max(xs), max(ys), cleaned))
    blocks: list[Block] = []
    for y0, x0, x1, y1, text in _order_blocks(items):
        size = min(24.0, max(5.0, (y1 - y0) / 1.2))
        blocks.append(
            Block(
                text=text, page=page_index, x0=x0, y0=y0, x1=x1, y1=y1,
                size=round(size, 2), align="left", bold=False, single_line=True,
            )
        )
    return blocks


def _looks_scanned(page: "fitz.Page") -> bool:
    """True for a page with no text layer that does carry an image (a scan)."""
    try:
        return bool(page.get_images(full=True))
    except Exception:
        return False


def _ocr_page_blocks(
    page_index: int,
    page: "fitz.Page",
    ocr_fn: Callable[[int, "fitz.Page"], list[tuple[list, str]]] | None,
) -> list[Block]:
    """OCR one page and return its blocks (empty if OCR is unavailable/failed).

    ``ocr_fn`` is injected by tests (returns ``[(box, text)]`` in PDF points);
    production falls back to the shared RapidOCR engine.
    """
    if ocr_fn is not None:
        try:
            return _synthesize_ocr_blocks(list(ocr_fn(page_index, page)), page_index)
        except Exception:
            return []
    engine = _ocr_engine()
    if engine is None:
        return []
    try:
        img, zoom = _page_to_array(page)
        out = engine(img)
        results: list[tuple[list, str]] = []
        # RapidOCR returns (list of [box, text, score] or None, timings).
        items = out[0] if isinstance(out, tuple) else out
        if not items:
            return []
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


def _collect_page_struct(
    page: fitz.Page,
) -> tuple[list[tuple[fitz.Rect, float, bool]], list[tuple[fitz.Rect, str]]]:
    """Return ``(spans, lines)`` for a page.

    ``spans`` is ``(bbox, size, bold)`` for every span; ``lines`` is
    ``(bbox, text)`` for every non-empty visual line (the smallest unit the
    original PDF renders).  Line data lets the extractor split a block that
    PyMuPDF merged from several visually separate lines (e.g. a bullet list)
    back into per-line blocks so a translation can honour the original layout.
    """
    spans: list[tuple[fitz.Rect, float, bool]] = []
    lines: list[tuple[fitz.Rect, str]] = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            line_rect = fitz.Rect(line["bbox"])
            parts: list[str] = []
            for s in line.get("spans", []):
                rect = fitz.Rect(s["bbox"])
                size = float(s.get("size", 10.0))
                flags = int(s.get("flags", 0))
                font = str(s.get("font", ""))
                bold = bool(flags & 16) or "bold" in font.lower()
                spans.append((rect, size, bold))
                parts.append(s.get("text", ""))
            text = " ".join("".join(parts).split())
            if text:
                lines.append((line_rect, text))
    return spans, lines


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


def _split_list_lines(
    block_rect: fitz.Rect,
    block_text: str,
    page_lines: Sequence[tuple[fitz.Rect, str]],
) -> list[tuple[fitz.Rect, str]]:
    """Return the per-line pieces of a block.

    A block PyMuPDF merged from several short, left-aligned lines (a bullet or
    numbered list) is split back into one ``(rect, text)`` per line so each
    line is translated and drawn separately.  Wrapped-paragraph blocks (whose
    lines fill the block width) are returned unchanged as a single ``(rect,
    text)``.
    """
    matching = [
        (lr, lt) for lr, lt in page_lines if lr.intersects(block_rect)
    ]
    matching.sort(key=lambda rt: (round(rt[0].y0, 1), rt[0].x0))
    if len(matching) >= 2 and _looks_like_list(block_rect, matching):
        return matching
    return [(block_rect, block_text)]


def _looks_like_list(
    block_rect: fitz.Rect, lines: Sequence[tuple[fitz.Rect, str]]
) -> bool:
    """True when most of a block's lines are short (list items).

    A wrapped paragraph's lines each reach close to the block's right edge,
    whereas a list's items end well before it.  If most lines leave a clear
    right margin the block is a list, not a paragraph.
    """
    if len(lines) < 2:
        return False
    min_height = min(lr.y1 - lr.y0 for lr, _t in lines)
    short = sum(1 for lr, _t in lines if block_rect.x1 - lr.x1 > 4.0)
    # Require a clear majority of short lines; a wrapped paragraph has few.
    return short >= max(2, int(len(lines) * 0.6)) and min_height > 0


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
    """Regroup a flat ``values`` list back into per-page lists.

    ``block_pages`` and ``values`` must be the same length (they both derive
    from one document's blocks).  A mismatch is a bug in the caller and would
    otherwise be silently truncated by ``zip`` — refuse loudly instead, so a
    translation is never dropped without a trace.
    """
    if len(block_pages) != len(values):
        raise ValueError(
            "分页对齐失败：块页码数 "
            f"{len(block_pages)} 与译文数 {len(values)} 不一致"
        )
    per_page: list[list[str]] = [[] for _ in range(page_count)]
    for page, value in zip(block_pages, values):
        if page < 0 or page >= page_count:
            raise ValueError(f"块页码 {page} 超出页面范围 [0, {page_count})")
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
    pages: Sequence[Sequence[Block]],
) -> None:
    """Create a bilingual PDF: each original page followed by a translation
    page that mirrors the original layout (every translated block sits at its
    source block's position).

    ``pages`` (the layout blocks from :func:`extract_document_text`) is
    required: translation text is drawn at its source block's position, so
    without it there is nothing to mirror — silently dropping every
    translation would be worse than refusing the call.
    """
    src = fitz.open(str(src_path))
    new_doc = fitz.open()
    try:
        font = _CJK_FONT
        for i in range(src.page_count):
            new_doc.insert_pdf(src, from_page=i, to_page=i)
            page_rect = src[i].rect
            tpage = new_doc.new_page(width=page_rect.width, height=page_rect.height)
            blocks = pages[i] if i < len(pages) else []
            trans = per_page[i] if i < len(per_page) else []
            if len(blocks) != len(trans):
                raise ValueError(
                    f"第 {i + 1} 页布局块数 {len(blocks)} 与译文块数 "
                    f"{len(trans)} 不一致，已放弃导出（避免静默丢弃译文）"
                )
            if not blocks:
                _render_note(tpage, font, lang)
                continue
            for j in range(len(blocks)):
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
            if len(blocks) != len(trans):
                raise ValueError(
                    f"第 {i + 1} 页布局块数 {len(blocks)} 与译文块数 "
                    f"{len(trans)} 不一致，已放弃导出（避免静默丢弃译文）"
                )
            if not blocks:
                continue

            # Remove the original text (keep images and line art/graphics).
            for j in range(len(blocks)):
                b = blocks[j]
                page.add_redact_annot(fitz.Rect(b.x0, b.y0, b.x1, b.y1))
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            )

            # Draw the translation at the original positions / alignment /
            # font size (see ``_draw_translated_block`` for the fitting rules).
            for j in range(len(blocks)):
                _draw_translated_block(page, font, blocks[j], trans[j])

        out_doc.set_metadata({"title": "Translated text", "creator": "PDFtranslate"})
        out_doc.save(str(out_path), garbage=4, deflate=True)
    finally:
        out_doc.close()
        src.close()
