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
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Sequence

import pymupdf as fitz

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

    @property
    def page_count(self) -> int:
        return len(self.pages)


def extract_document_text(path: str | Path, title: str | None = None) -> DocumentText:
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
    """
    doc = fitz.open(str(path))
    result = DocumentText(title=title or Path(path).stem)
    try:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            lines = _collect_lines(page)
            if not lines:
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
    finally:
        doc.close()
    return result


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
            for j in range(m):
                b = blocks[j]
                page.add_redact_annot(fitz.Rect(b.x0, b.y0, b.x1, b.y1))
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            )

            # Draw the translation at the original positions / alignment /
            # font size (see ``_draw_translated_block`` for the fitting rules).
            for j in range(m):
                _draw_translated_block(page, font, blocks[j], trans[j])

        out_doc.set_metadata({"title": "Translated text", "creator": "PDFtranslate"})
        out_doc.save(str(out_path), garbage=4, deflate=True)
    finally:
        out_doc.close()
        src.close()
