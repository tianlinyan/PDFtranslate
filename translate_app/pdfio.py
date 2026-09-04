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
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
import hashlib
import json
import os
import re
import statistics
import tempfile
import threading
from typing import Any, Callable, Sequence

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
    #: 24-bit RGB color of the source text (as an int, e.g. ``0xCC0000``); kept so
    #: the in-place exporter can restore coloured headings instead of rendering
    #: everything black.  ``0`` (black) is the default.
    color: int = 0
    #: True when this block's text must be kept verbatim (not translated) — e.g.
    #: a personal-name cell in a "姓名 / Name" table column.  Such blocks are
    #: never sent to the model and always export as the original text.
    keep_original: bool = False
    #: True when this block is a node label of an org chart / architecture
    #: diagram (usually a narrow-tall box whose text reads vertically).  Such
    #: labels are structural, not prose — translating them mangles the diagram —
    #: so they are kept verbatim regardless of target language (unlike
    #: ``keep_original``, which for a Western target re-romanizes the name).
    is_chart: bool = False
    #: True when this block was recovered by OCR from a scan.  Such blocks sit
    #: on top of a raster image rather than a text layer, so the in-place
    #: exporter must first cover the original pixels instead of redacting text.
    ocr: bool = False
    #: True when this block is a cell of a ruled / detected table (as opposed to
    #: a flowing paragraph).  Its ``x0``/``x1`` are widened to the full column
    #: cell, so a (usually wider) translation has the whole column width to fit
    #: on one line instead of wrapping inside the narrow source-text extent —
    #: wrapped table rows are what push the following rows down and misalign the
    #: table.  The fit helper shrinks the font to a single line when possible.
    in_table: bool = False
    #: When > 0, the width the translation fitter may use, overriding the bbox
    #: width.  A scanned statement's OCR box only encloses the printed glyphs —
    #: a 2-char "合并" header box is ~18pt while the figure sub-column it heads is
    #: ~70pt wide, and fitting the 12-char English header against the glyph box
    #: crushes it to the 3pt floor (the reported "Consolidated" illegibility).
    #: The bbox still marks the source pixels for the cover/redact step, so this
    #: only widens where the translation may draw, never what gets covered.
    #: 0 = use the bbox width.
    fit_width: float = 0.0
    #: When > 0, the extra draw height beyond the bbox that an ``in_table`` cell
    #: may use: a scanned statement's own OCR box is only as tall as the printed
    #: glyphs, but the row gap up to the next grid row is empty raster whitespace.
    #: A cell whose translation cannot hold a *readable* single line wraps
    #: instead of shrinking below the readability floor (the old fixed one-line
    #: rule produced 3pt glyphs).  The wrap is bounded by this height so the
    #: translation never crosses the table line below the row.  0 = no band
    #: (last row of a grid, or a text-layer cell), each with its own fallback.
    fit_height: float = 0.0


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


#: Page kinds the AI-interaction session triages a page into.
PAGE_NORMAL = "normal"
PAGE_SCAN = "scan"
PAGE_CHART = "chart"
PAGE_TABLE = "table"
PAGE_UNCERTAIN = "uncertain"

#: A page is treated as a *chart* (org / architecture diagram) when at least this
#: many narrow-tall node boxes are present (the strong diagram signature).
_CHART_NODE_MIN = 3


def detect_language(texts: Sequence[str]) -> str:
    """Heuristically guess the document language: ``"zh"`` / ``"en"`` / ``"mixed"``.

    Counts CJK ideographs vs ASCII letters across the blocks.  A document with
    only CJK is ``zh``; only Latin is ``en``; a substantial mixture (a report with
    Chinese prose and English table headers) is ``mixed``.  No letters → ``unknown``.
    """
    cjk = latin = 0
    for t in texts:
        for ch in str(t):
            if "\u4e00" <= ch <= "\u9fff":
                cjk += 1
            elif ch.isascii() and ch.isalpha():
                latin += 1
    if cjk == 0 and latin == 0:
        return "unknown"
    if cjk >= latin:
        return "zh"
    if latin >= cjk * 5:
        return "en"
    return "mixed"


def classify_page(blocks: Sequence[Block]) -> str:
    """Classify one page as ``normal`` / ``scan`` / ``chart`` / ``uncertain``.

    Deterministic, block-flag based (no model call):
    * ``scan`` — every block came from OCR (``Block.ocr``) i.e. a scanned raster page
      (this includes scanned statements / tables — those stay special for negotiation).
    * ``chart`` — at least ``_CHART_NODE_MIN`` narrow-tall node boxes
      (``_is_vertical_label``), the org-chart / architecture-diagram signature.
    * ``uncertain`` — mixed OCR + text, or ambiguous (very few blocks).
    * ``normal`` — everything else, **including a non-scanned (text-layer) table page**:
      a normal table is just ordinary text to translate, not a special page.
    """
    if not blocks:
        return PAGE_UNCERTAIN
    n = len(blocks)
    n_ocr = sum(1 for b in blocks if getattr(b, "ocr", False))
    n_vertical = sum(1 for b in blocks if _is_vertical_label(b))
    if n_ocr == n:
        return PAGE_SCAN
    if n_vertical >= _CHART_NODE_MIN:
        return PAGE_CHART
    # A mixed OCR + text page (or a genuinely ambiguous one) is flagged for the
    # user; a pure-text page — even with a single block — is normal.
    if 0 < n_ocr < n:
        return PAGE_UNCERTAIN
    return PAGE_NORMAL


def get_doc_info(doc: "DocumentText") -> dict[str, Any]:
    """Summarise a ``DocumentText`` for the AI-interaction session.

    Returns page count, language guess, per-page kinds, and counts of
    text / scan / chart / table / special pages (used by ``get_doc_info`` tool and
    the preprocess phase).  Never touches the model — purely deterministic.
    """
    kinds = [classify_page(p) for p in doc.pages]
    text_pages = sum(1 for k in kinds if k == PAGE_NORMAL)
    chart_pages = sum(1 for k in kinds if k == PAGE_CHART)
    table_pages = sum(1 for k in kinds if k == PAGE_TABLE)
    uncertain_pages = sum(1 for k in kinds if k == PAGE_UNCERTAIN)
    scan_pages = doc.ocr_count
    return {
        "pages": doc.page_count,
        "title": doc.title,
        "language": detect_language(doc.blocks),
        "text_pages": text_pages,
        "scan_pages": scan_pages,
        "chart_pages": chart_pages,
        "table_pages": table_pages,
        "uncertain_pages": uncertain_pages,
        "special_pages": chart_pages + table_pages + uncertain_pages + scan_pages,
        "block_count": len(doc.blocks),
        "kinds": kinds,
    }


def nearest_block(blocks: Sequence["Block"], bbox) -> "Block | None":
    """The block whose centre falls in ``bbox`` (PDF points); nearest if several overlap.

    Used by the M6 ``apply_annotation`` tool: a user-drawn region on the preview is
    matched back to the source block it points at, so the AI can edit that block.
    Returns ``None`` when the region matches no block (e.g. it is empty scan space).
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        r = fitz.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    cx = (r.x0 + r.x1) / 2.0
    cy = (r.y0 + r.y1) / 2.0
    best: Block | None = None
    best_d = 1e18
    for b in blocks:
        bcx = (b.x0 + b.x1) / 2.0
        bcy = (b.y0 + b.y1) / 2.0
        if r.x0 - 6.0 <= bcx <= r.x1 + 6.0 and r.y0 - 6.0 <= bcy <= r.y1 + 6.0:
            d = abs(bcx - cx) + abs(bcy - cy)
            if d < best_d:
                best_d, best = d, b
    return best


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
    shared RapidOCR engine is used.  Recognised pages are cached per document
    (keyed by file mtime + size, so an edited PDF is re-OCR'd) and the cache is
    written after every page, so a cancelled run keeps what it already did.
    ``cancel`` is polled per page (raising :class:`TranslationCancelled`), and
    ``log`` receives per-page OCR progress.
    """
    doc = fitz.open(str(path))
    result = DocumentText(title=title or Path(path).stem)
    ocr_cache: dict[int, list[dict]] = {}
    ocr_cache_path: Path | None = None
    ocr_cache_warned = False
    if ocr:
        try:
            # In-memory-only runs (production) keep ``ocr_cache_path`` None so OCR
            # results are never read from / written to disk — just the final output.
            if _ocr_cache_persist_enabled():
                ocr_cache_path = _ocr_cache_path(path)
                ocr_cache = _load_ocr_cache(ocr_cache_path)
        except Exception:
            # The main ``finally`` only starts below, so close the document here
            # rather than leaking the open file handle on a cache failure.
            doc.close()
            raise
    try:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            # ``get_text("dict")`` is the expensive extraction; fetch it once
            # per page and share it between the line and span collectors
            # (they used to each re-extract it, doubling the cost).
            page_dict = page.get_text("dict")
            lines = _collect_lines(page_dict)
            # A page whose text layer is only a page number / short title (a
            # scan, or a vector chart such as the balance sheets and the org
            # chart) has no extractable content; when it also carries images or
            # drawings the real content must come from OCR instead of being
            # silently dropped.  ``len(lines) <= 3`` keeps this conservative so
            # a normal text page is never OCR'd just because it also embeds an
            # image or two.
            sparse_text = len(lines) <= 3
            if ocr and (ocr_fn is not None or _needs_ocr(page)) and (not lines or sparse_text):
                page_blocks: list[Block] = []
                if page_index in ocr_cache:
                    # Old caches predate number normalization; renormalize on
                    # load so a stale cache cannot keep the garbled figures.
                    page_blocks = [_block_from_dict(d) for d in ocr_cache[page_index]]
                    fixed_count, fixed_examples = 0, []
                    renorm: list[Block] = []
                    for b in page_blocks:
                        norm = _normalize_number(b.text)
                        if norm != b.text:
                            fixed_count += 1
                            if len(fixed_examples) < 2:
                                fixed_examples.append(f"{b.text} → {norm}")
                            b = replace(b, text=norm)
                        renorm.append(b)
                    page_blocks = renorm
                    _log_number_fixes(log, page_index, fixed_count, fixed_examples)
                else:
                    if log:
                        log(f"  OCR 第 {page_index + 1}/{doc.page_count} 页…")
                    if ocr_fn is None and _get_ocr_engine() is None:
                        _warn_ocr_unavailable(log)
                        page_blocks = []
                    else:
                        page_blocks = _ocr_page_blocks(
                            page_index, page, ocr_fn, cancel, log
                        )
                        if page_blocks:
                            ocr_cache[page_index] = [
                                _block_to_dict(b) for b in page_blocks
                            ]
                            # Persist after *every* page: OCR is by far the
                            # slowest stage, so a cancel or a crash must not
                            # throw away the pages already recognised.
                            if ocr_cache_path is not None:
                                reason = _save_ocr_cache(ocr_cache_path, ocr_cache)
                                if reason and not ocr_cache_warned and log:
                                    ocr_cache_warned = True
                                    log(
                                        "  警告：OCR 缓存写入失败（"
                                        f"{reason}），已识别页面不会被缓存，"
                                        f"重跑将重新 OCR：{ocr_cache_path}"
                                    )
                if page_blocks:
                    for b in page_blocks:
                        result.blocks.append(b.text)
                        result.block_pages.append(page_index)
                    result.pages.append(page_blocks)
                    result.ocr_count += 1
                    continue
                if not lines:
                    result.pages.append([])
                    continue
                # OCR returned nothing but the page has a (sparse) text layer:
                # fall through and translate whatever text was extractable.

            if not lines:
                # No text layer and nothing came from OCR: an empty page.
                result.pages.append([])
                continue

            spans = _collect_spans(page_dict)
            page_x0 = min(ln["x0"] for ln in lines)
            page_x1 = max(ln["x1"] for ln in lines)
            page_blocks: list[Block] = []

            # Detect ruled tables so their cells stay as separate blocks AND so
            # the prose pipeline (which merges flowing paragraphs) never sees
            # them — otherwise its relaxed vertical-gap break would collapse the
            # table's rows into one another.
            cell_rects = _detect_table_cell_rects(page)
            if cell_rects:
                table_lines = [ln for ln in lines if _line_center_in_rects(ln, cell_rects)]
                text_lines = [ln for ln in lines if not _line_center_in_rects(ln, cell_rects)]
                prose = _group_lines(_order_lines(text_lines))
            else:
                table_lines, text_lines = [], []
                prose = _group_lines(_order_lines(lines))

            for group in prose:
                block = _group_to_block(group, spans, page_x0, page_x1, page_index)
                if block is not None:
                    page_blocks.append(block)
            page_blocks.extend(
                _build_table_blocks(
                    table_lines, spans, page_x0, page_x1, page_index, cell_rects
                )
            )

            for block in page_blocks:
                result.blocks.append(block.text)
                result.block_pages.append(page_index)
            result.pages.append(page_blocks)
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


#: Looks like an OCR'd number: digits plus separators (comma/dot/space), an
#: optional stray ``/`` (some engines emit ``192, / 003,164.72``), a leading
#: sign and a trailing ``%``.  Strings containing letters/CJK are never numbers
#: (and are never touched): they are prose or a label, not a figure cell.
_NUM_SHAPE_RE = re.compile(r"^[0-9,.\s/]+[%％]?$")
#: A correctly-grouped amount: 1-3 digits at the head, groups of 3, an optional
#: 1-3 digit decimal.  Anything matching this exactly is already well formed.
_WELL_FORMED_NUM_RE = re.compile(r"^[0-9]{1,3}(,[0-9]{3})*(\.[0-9]{1,3})?$")


def _repair_number_separators(s: str) -> str:
    """Turn a mangled grouping into the conventional ``x,xxx,xxx.xx`` shape.

    Three repairs are attempted (each a minimal edit, then the result is
    re-validated by the caller):

    1. A ``/`` joins the two digital runs: it is the residual of a wrap split
       (``231. / 81`` → ``231.81``, ``192, / 003,164.72`` → ``192,003,164.72``,
       ``150.0 / 8`` → ``150.08``).
    2. The last ``.`` is the decimal point; every other ``.`` stands for a
       thousands separator (``3,702.726,474.45`` → ``3,702,726,474.45``).
    3. A *trailing* ``,`` followed by 1-2 digits is a comma-written decimal
       point (``11,530,351,55`` → ``11,530,351.55``).
    """
    s = s.replace("/", "")
    s = re.sub(r",\s*,", ",", s)
    tail_dot = s.rfind(".")
    if tail_dot >= 0:
        frac = s[tail_dot + 1:]
        if frac.isdigit() and 1 <= len(frac) <= 3:
            out = s[:tail_dot].replace(".", ",") + "." + frac
        else:
            out = s.replace(".", ",")
    else:
        m = re.search(r",(\d{1,2})$", s)
        if m:
            out = s[: m.start()] + "." + m.group(1)
        else:
            out = s.replace(".", ",")
    # Placeholder-separator artifacts (``192, / 003,164.72``, ``552,.394``)
    # leave doubled commas after the repairs; collapse them.
    return re.sub(r",{2,}", ",", out)


def _normalize_number(text: str) -> str:
    """Repair OCR-degraded amounts (``65, 334, 085.99`` → ``65,334,085.99``).

    RapidOCR re-segments dense figure columns: it inserts stray spaces around
    separators, swaps dots/commas (``3,702.726,474.45``) and drops a decimal
    point (``11,530,351,55``).  Pure digit cells bypass the translation model
    (``_needs_translation``), so whatever this returns is what the export shows
    — the corruption must be repaired here or it reaches the reader verbatim.

    Only strings that consist of digits and separators are touched; prose with
    letters/CJK, dates (``1960.08``), plain ungrouped numbers (``0.98``,
    ``92.5%``) and anything unverifiable are returned unchanged.  The change is
    formatting only — the digit sequence is never altered.
    """
    s = str(text)
    s = (
        s.replace("，", ",").replace("．", ".")
        .replace("　", " ").replace("\xa0", " ")
        .replace("％", "%").replace("−", "-").replace("－", "-")
    )
    s = s.strip()
    if not s:
        return str(text)
    tail = ""
    if s.endswith("%"):
        tail = "%"
        s = s[:-1].strip()
    sign = ""
    if s[:1] in ("+", "-"):
        sign, s = s[0], s[1:].strip()
    if not s or not _NUM_SHAPE_RE.match(s):
        return str(text)
    if "," not in s:
        # No thousands separators at all: a date (1960.08), a plain value
        # (0.98) or ungrouped digits.  Not enough signal to regroup; drop stray
        # spaces (the run must survive the wrap) but keep the digit sequence
        # intact.  One extra rule: a digit group after ``/`` where the string
        # already has a decimal point is a wrapped figure (``231. / 81`` →
        # ``231.81``), not a ratio — join it.
        core = re.sub(r"\s+", "", s)
        if "." in core and "/" in core:
            core = core.replace("/", "")
        return sign + core + tail if core else str(text)
    s2 = s.replace(" ", "")
    if _WELL_FORMED_NUM_RE.match(s2):
        return sign + s2 + tail
    repaired = _repair_number_separators(s2)
    if _WELL_FORMED_NUM_RE.match(repaired):
        return sign + repaired + tail
    # Digit count cannot form valid 3-digit groups: leave it for human review
    # (silently "fixing" it could emit a wrong amount).
    return str(text)


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


def _warn_ocr_unavailable(log: Callable[[str], None] | None) -> None:
    """Warn (once per process) that the OCR engine could not be loaded.

    The "warned already?" flag is module-level mutable state, so it is rebound
    *here* — inside a function that declares ``global`` — rather than inside the
    extraction loop.  Assigning it there without the declaration made Python
    treat the name as a local for the whole function, so merely reading it
    raised ``UnboundLocalError`` and the intended "degrade and skip" path
    crashed the entire run instead.
    """
    global _OCR_WARNED
    if _OCR_WARNED:
        return
    _OCR_WARNED = True
    if log:
        log("  OCR 引擎不可用：未安装 rapidocr_onnxruntime，扫描页将不识别。")


def _ocr_cache_persist_enabled() -> bool:
    """Whether OCR results persist to disk (OPT-IN via ``PDFTRANSLATE_OCR_CACHE_DIR``).

    Production is in-memory only (minimise disk writes / SSD wear): OCR results live
    in memory for the run and are not written, so a re-run re-OCRs scans.  Tests set
    the env dir, which they ALSO use to enable persistence so the cache stays tested.
    """
    return bool(os.environ.get("PDFTRANSLATE_OCR_CACHE_DIR"))


def _ocr_cache_dir() -> Path:
    """A writable dir for OCR results (OCR output is slow — reuse on re-run).

    ``PDFTRANSLATE_OCR_CACHE_DIR`` overrides the location; the default is the
    user's ``~/.pdftranslate/ocr_cache``.

    Unlike a bare ``mkdir`` (which is a no-op on an existing directory), each
    candidate is verified with a real probe write before it is accepted.  Under a
    read-only / sandboxed ``$HOME`` ``mkdir(exist_ok=True)`` succeeds while every
    later write is denied — which would silently disable OCR reuse and re-run the
    slowest stage of the pipeline on every re-run.
    """
    candidates: list[Path] = []
    override = os.environ.get("PDFTRANSLATE_OCR_CACHE_DIR")
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [
            Path.home() / ".pdftranslate" / "ocr_cache",
            Path(tempfile.gettempdir()) / "pdftranslate_ocr_cache",
        ]
    )
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / f".ocr_write_probe_{os.getpid()}"
            probe.write_text("", "utf-8")
            probe.unlink()
            return path
        except Exception:
            continue
    return Path.home() / ".pdftranslate" / "ocr_cache"


#: Bump when the cached block *meaning* changes so old files are ignored.  The
#: untagged legacy format predates the OCR grid reconstruction: its blocks carry
#: no ``in_table`` flag or column widening, so a loading cache hit uses them
#: verbatim and every scanned-table cell (the balance sheets) wraps inside its
#: own sliver instead of staying on one line — the grid exporter is silently
#: dead for every document OCR'd by an old build.
#:
#: v2: ``fit_width`` added (widens the draw space of text cells inside figure
#: sub-columns — the "合并"/"母公司" headers that were crushed to the 3pt floor)
#: and 附注 reference cells keep their glyph box instead of being widened onto
#: the label.  Blocks cached by v1 lack both, so they must be re-synthesized.
#:
#: v3: label cells no longer widen their own bbox to the whole column — v2 ran
#: the label cover/redact over the full column extent, erasing the (二)-band and
#: 行次 glyphs and letting long label translations be sliced by their white
#: covers.  v3 keeps the OCR glyph box as the bbox and puts the whole (and
#: row-next-cell-bounded) column width into ``fit_width`` only.
#:
#: v4: OCR-split label fragments (``资产处置收益（损失以`` + ``号填列）``) are
#: merged into one cell; v3 cached each fragment as a separate block so later
#: builds would bound each fragment by the next one and still squeeze the
#: translation into a 60pt sliver.
#:
#: v5: ``fit_height`` added — a grid cell's translation may wrap (readability
#: floor beats the 3pt slug), bounded by the row band down to the next grid
#: line.  Blocks cached by v4 deserialize to ``fit_height = 0`` (the field is
#: new), so their wraps would be *unbounded* — translations hang over the rows
#: below instead of stopping at the band.  They must be re-synthesized.
_OCR_CACHE_VERSION = 5


def _ocr_cache_path(doc_path: str | Path) -> Path:
    """Cache file for a document's OCR results.

    The key includes the file's mtime and size, not just its path: reusing OCR
    results from a *replaced or edited* PDF would silently translate stale
    content.  (The translation cache needs no such stamp — it looks blocks up by
    content hash.)
    """
    p = Path(doc_path)
    try:
        st = p.stat()
        stamp = f"|{int(st.st_mtime)}|{st.st_size}"
    except OSError:
        stamp = ""
    h = hashlib.sha1(f"{p.resolve()}{stamp}".encode("utf-8")).hexdigest()[:16]
    return _ocr_cache_dir() / f"ocr_v{_OCR_CACHE_VERSION}_{h}.json"


def _load_ocr_cache(cache_path: Path) -> dict[int, list[dict]]:
    """Load per-page OCR block dicts (``{page_index: [block_dict]}``)."""
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text("utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {int(k): v for k, v in raw.items()}
    except Exception:
        return {}


def _save_ocr_cache(cache_path: Path, data: dict[int, list[dict]]) -> str | None:
    """Best-effort *atomic* persist of OCR results (OCR is slow — reuse it).

    Written through a temp file + :func:`os.replace` because this is now called
    after every OCR'd page: a cancel (or the GUI's hard ``os._exit``) landing
    mid-write would otherwise leave truncated JSON, which reads back as "no
    cache" and re-runs the whole slow OCR pass next time.

    Returns ``None`` on success, or a short human-readable reason on failure.  A
    failed OCR cache write must never abort a run, but — mirroring the translation
    cache — the caller should surface it once, or a read-only ``~/.pdftranslate``
    silently re-OCRs the whole document on every run with no clue why.
    """
    tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
        os.replace(tmp, cache_path)
        return None
    except Exception as exc:  # noqa: BLE001 — the cache is optional by design
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return f"{type(exc).__name__}: {exc}"


def _page_to_array(page) -> tuple[object, float]:
    """Render a page to a BGR numpy array plus the pixel-per-point zoom."""
    import numpy as np

    zoom = _OCR_DPI / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom), alpha=False, colorspace=fitz.csRGB
    )
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        # RGB -> BGR (RapidOCR / OpenCV convention).  The ``[::-1]`` flip yields
        # a negative-stride, non-contiguous view; RapidOCR/OpenCV may reject or
        # silently copy such buffers, so hand over a contiguous array instead.
        img = np.ascontiguousarray(img[:, :, :3][:, :, ::-1])
    return img, zoom


def _block_to_dict(b: Block) -> dict:
    return asdict(b)


def _block_from_dict(data: dict) -> Block:
    allowed = {f.name for f in fields(Block)}
    return Block(**{k: v for k, v in data.items() if k in allowed})


#: An amount / figure cell: digits with optional thousands separators, a decimal
#: point (ASCII or full-width), leading minus, percent, or accounting parentheses
#: (``(1,234.56)``).  OCR can insert stray spaces inside a figure (``65, 334, 085.99``)
#: so separators and the decimal point tolerate surrounding whitespace.  Such cells
#: are *numbers by content*, so a whole table column of them is right-aligned
#: (matching the source) and none of them is ever sent to the model.  Letters/CJK
#: never match, so an ordinary label (``营业收入``) is not mistaken for a figure.
_NUMERIC_CELL_RE = re.compile(
    r"^\s*[（(－\-−]?\s*"
    r"\d{1,3}(?:\s*[，,]\s*\d{3})*(?:\s*[.．]\s*\d+)?"
    r"\s*[%％]?\s*[）)]?\s*$"
)


def _is_numeric_cell(text: str) -> bool:
    """True when ``text`` is a pure figure / amount cell (not a label)."""
    t = str(text).strip()
    if not t:
        return False
    return bool(_NUMERIC_CELL_RE.match(t))


def _cluster_ocr_rows(items: Sequence[tuple], ytol: float = 4.5) -> list[list[tuple]]:
    """Group OCR ``(y0, x0, x1, y1, text)`` items into table rows by y-centre.

    Cells of the same source row share a baseline, so their y-centres cluster
    within a couple of points; the tolerance is a fraction of the median row
    height so distinct rows (which are ~8-15pt apart) never merge.
    """
    # y-centre of item i is (y0 + y1) / 2.
    by_centre = sorted(items, key=lambda it: (it[0] + it[3]) / 2.0)
    rows: list[list[tuple]] = []
    for it in by_centre:
        cy = (it[0] + it[3]) / 2.0
        if rows and abs(cy - (rows[-1][0][0] + rows[-1][0][3]) / 2.0) <= ytol:
            rows[-1].append(it)
        else:
            rows.append([it])
    return rows


def _cluster_ocr_columns(items: Sequence[tuple], xtol: float = 4.0) -> list[list[tuple]]:
    """Group OCR items into columns by greedy x-range overlap (reading order).

    Using the *range* (``[x0, x1]``) rather than the centre is what keeps a
    right-aligned numeric column together: its cells all touch the same right
    edge but recede leftward as the magnitude shrinks, so their centres scatter
    and a centre-based cluster would split them into several columns (exactly
    the misalignment seen in the translated balance sheet).

    Full-width items (a spanning title like ``资产负债表``, or a long ``编制单位``
    line) are pulled out *before* clustering: in the greedy overlap a wide item
    becomes a column whose right edge is the page margin and every cell of the
    real columns then "overlaps" it — collapsing several columns into one and
    interleaving their cells.  The median-width guard (same as
    :func:`_order_generic`) never flags a single-column page.  Columns are ordered
    left to right by their leftmost edge.
    """
    if len(items) < 2:
        return [list(items)]
    med_w = statistics.median(it[2] - it[1] for it in items)
    if med_w <= 0:
        return [list(items)]
    threshold = 1.5 * med_w
    full = [it for it in items if (it[2] - it[1]) > threshold]
    rest = [it for it in items if (it[2] - it[1]) <= threshold]
    xsorted = sorted(rest, key=lambda it: it[1])
    cols: list[list[tuple]] = []
    col_max_x1: list[float] = []
    for it in xsorted:
        x0, x1 = it[1], it[2]
        for c in range(len(cols)):
            if x0 < col_max_x1[c] - xtol:
                cols[c].append(it)
                col_max_x1[c] = max(col_max_x1[c], x1)
                break
        else:
            cols.append([it])
            col_max_x1.append(x1)
    cols.sort(key=lambda c: min(it[1] for it in c))
    # Reattach full-width items to the nearest column by their centre.
    for it in full:
        cx = (it[1] + it[2]) / 2.0
        best, bd = 0, float("inf")
        for ci, col in enumerate(cols):
            left = min(b[1] for b in col)
            right = max(b[2] for b in col)
            dist = 0.0 if left <= cx <= right else min(abs(cx - left), abs(cx - right))
            if dist < bd:
                bd, best = dist, ci
        cols[best].append(it)
    return cols


def _is_numeric_column(items: Sequence[tuple]) -> bool:
    """True when a column's cells are mostly figures (so they should right-align)."""
    if not items:
        return False
    numeric = sum(1 for it in items if _is_numeric_cell(it[4]))
    return numeric > 0 and numeric >= max(1, int(len(items) * 0.5))


#: A pure 附注-by-reference cell: ``(二)``, ``八)``, ``(十一)`` — the note marker
#: column of a scanned statement, not a continuation of the left label above it.
_NOTE_MARK_RE = re.compile(r"^\s*[（(]?\s*[一二三四五六七八九十0-9]{1,3}\s*[)）]\s*$")


def _join_fragments(a: str, b: str) -> str:
    """Concatenate two label fragments: ASCII edges get a space, CJK do not."""
    if a and b and a[-1].isascii() and b[0].isascii():
        return a + " " + b
    return a + b


def _merge_label_fragments(cells: list[tuple]) -> list[tuple]:
    """Merge OCR-split fragments of one label into a single cell.

    RapidOCR frequently breaks one printed label across two items (``资产处置
    收益（损失以`` + ``号填列）``), each separated by a few points.  Left as
    separate cells, each fragment bounds its fit width by the *next fragment's*
    x0 — squeezing a 50-char translation into a 60pt sliver.  Merge consecutive
    same-column text cells (gap <= 15pt) whose text is not a standalone
    ``(二)``-style note marker; the merged run then forms the row's one label,
    fit against the whole run.
    """
    merged: list[tuple] = []
    for it, ci, numeric in cells:
        if merged:
            (pit, pci, pnum) = merged[-1]
            if _can_merge_label(pit, pnum, it, numeric) and ci == pci:
                y0, x0, x1, y1, text = pit
                merged[-1] = (
                    (
                        min(y0, it[0]), x0, max(it[2], x1), max(y1, it[3]),
                        _join_fragments(text, it[4]),
                    ),
                    pci, False,
                )
                continue
        merged.append((it, ci, numeric))
    return merged


def _can_merge_label(prev: tuple, prev_numeric: bool, nxt: tuple, nxt_numeric: bool) -> bool:
    """True when ``nxt`` continues the label of ``prev`` (same row, adjacent, prose)."""
    if prev_numeric or nxt_numeric:
        return False
    _y0, x0, x1, _y1, text = prev
    _ny0, nx0, _nx1, _ny1, ntext = nxt
    if not text or not ntext:
        return False
    # Split fragments sit 12-18pt apart; a 附注 "(二)" reference is 40pt+ away
    # (its own marker-exception guard double-covers that case).
    if nx0 - x1 > 20.0:
        return False
    return not _NOTE_MARK_RE.match(ntext)


def _reconstruct_ocr_grid(items: Sequence[tuple]) -> tuple[list[Block], list[dict]]:
    """Turn OCR ``items`` into a row-major grid of cells, when they form a table.

    Scanned financial statements reach us with *no* text layer and *no* vector
    rules, so ``page.find_tables()`` finds nothing and the whole table pipeline is
    bypassed — the cells get treated as unrelated prose blocks.  The result is the
    garbled balance sheet: column-major reading order the model must untangle,
    every numeric column left-aligned, labels wrapped to a few points wide, and
    each cell overlapping its neighbours once the (longer) English translation is
    drawn at the tiny OCR box.

    This reconstructs the grid from the OCR boxes themselves: rows by y-centre
    (cells of a row share a baseline), columns by x-range overlap with a
    full-width guard.  When at least two rows and two columns line up it returns:

    * ``blocks`` — one cell per item in **row-major** reading order, numeric
      columns right-aligned (to each cell's own right edge, which the OCR box
      already encodes for a right-aligned figure), and label cells widened to the
      column extent so a long label wraps inside its column instead of spilling
      into the next.
    * ``tables`` — the same structure ``_extract_tables`` returns, so the existing
      row-height expansion + grid redraw runs for scans too.

    Returns ``([], [])`` when the page does not form a detectable grid (a sparse
    scan or a single column), leaving the caller on the old prose path.
    """
    rows = _cluster_ocr_rows(items)
    if len(rows) < 2:
        return [], []
    cols = _cluster_ocr_columns(items)
    if len(cols) < 2:
        return [], []
    # Only treat the page as a table when it really looks like one.  A prose or
    # two-column scan also lines up into rows/columns; forcing it into grid mode
    # would widen every text box to a column and wrap paragraphs into slabs.  A
    # table (a financial statement) has at least one whole column of figures, so
    # require that — a page of prose never has a numeric column.
    if not any(_is_numeric_column(c) for c in cols):
        return [], []

    def col_for(x_centre: float) -> int:
        for ci, col in enumerate(cols):
            left = min(b[1] for b in col)
            right = max(b[2] for b in col)
            if left - 3.0 <= x_centre <= right + 3.0:
                return ci
        return -1

    def col_extent(ci: int) -> tuple[float, float]:
        col = cols[ci]
        return min(b[1] for b in col), max(b[2] for b in col)

    numeric_cols = [_is_numeric_column(c) for c in cols]
    blocks: list[Block] = []
    # Rightmost edge on the page: the fit for a row's last cell extends up to it,
    # so e.g. a signature-row label (the last thing in its row) is not crushed to
    # ~3.5pt against its own ~2-char box.
    page_right = max(it[2] for it in items)
    # Row-major placement: cells of a row stay together (column-major ordering
    # used to interleave cells of different rows, scrambling the translation).
    # Each row's draw band runs down to the *next* row's top: the fix for the
    # one-line-crush is not to shrink below the readability floor but to wrap,
    # and the wrap must stop before the raster table line of the row below.
    row_tops = [min(it[0] for it in row) for row in rows]
    for k, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda it: it[1])
        next_top = row_tops[k + 1] if k + 1 < len(row_tops) else None
        # Cart this row's cells once: column id and figure-ness drive every
        # placement rule below, and "the leftmost text cell of the column" must
        # be decided against all the row's cells, not one at a time.  OCR-split
        # label fragments are merged first so they become one row cell (and the
        # ``row_sorted[i+1]`` bound below sees the whole run, not a fragment).
        cells = _merge_label_fragments(
            [
                (it, col_for((it[1] + it[2]) / 2.0), _is_numeric_cell(it[4]))
                for it in row_sorted
            ]
        )
        row_sorted = [c[0] for c in cells]
        for i, (it, ci, numeric) in enumerate(cells):
            y0, x0, x1, y1, text = it
            if _is_pure_symbol(text):
                continue
            fit_width = 0.0
            if ci >= 0 and not numeric and not numeric_cols[ci]:
                left, right = col_extent(ci)
                # The label column's text does not start at the column's left
                # edge (page-header items widen it), so a left-edge test
                # ("x0 <= left+4") cannot distinguish label from the nested
                # 附注/行次 band.  The label is the *leftmost text cell* of
                # its column within the row: widen it to the whole column so
                # the (usually longer) translation wraps inside the real
                # column width instead of a 40pt sliver.
                leftmost = all(
                    other[1] >= x0 - 1.0
                    for (other, other_ci, other_num) in cells
                    if other_ci == ci and not other_num
                )
                if leftmost:
                    # Keep the OCR glyph box as the bbox — the cover/redact step
                    # then erases exactly the printed label pixels and never a
                    # neighbouring cell — and give the fitter the whole label
                    # column instead, bounded by the next cell of the row so a
                    # long translation stops before the (二)-band / 行次 numbers
                    # instead of being sliced by their white covers.
                    next_x0 = (
                        row_sorted[i + 1][1] if i + 1 < len(row_sorted) else right
                    )
                    box_x0, box_x1 = x0, x1
                    fit_width = max(
                        x1 - x0, min(next_x0, right) - 2.0 - x0
                    )
                    align = "left"
                else:
                    # The 附注 "(二)" reference column lives *inside* the label
                    # extent; a widened box would splatter it all over the label.
                    # Keep the glyph box and centre it where the source put it.
                    box_x0, box_x1 = x0, x1
                    align = "center"
            else:
                # Figure cell (or a cell in a numeric column): keep the OCR box —
                # it already encodes the source-aligned right edge — and right-align
                # so the two stacked numeric sub-columns (合并 / 母公司) stay apart.
                box_x0, box_x1 = x0, x1
                align = "right" if numeric else "left"
                if not numeric and ci >= 0:
                    # Text inside a figure sub-column: the "合并"/"母公司" header
                    # cells and the signature-row labels.  Their glyph box is far
                    # narrower than the real cell, so anything longer than ~2
                    # chars is crushed to the 3pt floor.  Let the fitter use the
                    # empty gap up to the row's next cell (this column is all
                    # blank space there) instead.
                    nxt = row_sorted[i + 1][1] if i + 1 < len(row_sorted) else page_right
                    fit_width = max(x1 - x0, (nxt - 2.0) - x0)
            # The wrap band down to the row below (minus a 1.5pt margin for the
            # raster line itself).  Only when it exceeds the glyph box; the last
            # row has no band and keeps the box.
            fit_height = 0.0
            if next_top is not None:
                band = next_top - y0 - 1.5
                if band > y1 - y0:
                    fit_height = band
            size = min(_MAX_FONT, max(5.0, (y1 - y0) / 1.2))
            blocks.append(
                Block(
                    text=text, page=0, x0=box_x0, y0=y0, x1=box_x1, y1=y1,
                    size=round(size, 2), align=align, bold=False, single_line=True,
                    ocr=True, in_table=True, fit_width=fit_width,
                    fit_height=fit_height,
                )
            )

    if len(blocks) < 2:
        return [], []
    # Cell rects per (row, column) for the row-expansion machinery.
    row_rects: list[list[fitz.Rect]] = []
    col_edges = sorted(
        {round(it[1], 1) for it in items} | {round(it[2], 1) for it in items}
    )
    for row in rows:
        top = min(it[0] for it in row)
        bottom = max(it[3] for it in row)
        row_rects.append(
            [
                fitz.Rect(col_extent(ci)[0], top, col_extent(ci)[1], bottom)
                for ci in range(len(cols))
            ]
        )
    bbox = fitz.Rect(
        min(it[1] for it in items), min(it[0] for it in items),
        max(it[2] for it in items), max(it[3] for it in items),
    )
    tables = [{"bbox": bbox, "rows": row_rects, "col_edges": col_edges}]
    return blocks, tables


def _synthesize_ocr_blocks(
    results: Sequence[tuple[list, str]], page_index: int,
    log: Callable[[str], None] | None = None,
    page_height: float | None = None,
) -> list[Block]:
    """Turn ``[(box, text), ...]`` (box already in PDF points) into blocks.

    Text is cleaned, ordered with the same column-aware reading order as native
    text, and given a font size estimated from the box height.  When the OCR
    boxes form a table grid (scanned financial statements), the cells are rebuilt
    row-major with column-wide boxes and numeric columns right-aligned so the
    whole table keeps its shape instead of collapsing into a jumble.

    Numbers are normalized *before* the grid is reconstructed (a mangled figure
    like ``65, 334, 085.99`` would not even be recognised as a numeric cell);
    each fix is reported through ``log`` so corrupted values never reach the
    reader silently.
    """
    items: list[tuple] = []
    fixed_count = 0
    fixed_examples: list[str] = []
    for box, text in results:
        cleaned = _clean_text(text)
        if not cleaned:
            continue
        normalized = _normalize_number(cleaned)
        if normalized != cleaned:
            fixed_count += 1
            if len(fixed_examples) < 2:
                fixed_examples.append(f"{cleaned} → {normalized}")
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        items.append((min(ys), min(xs), max(xs), max(ys), normalized))
    if not items:
        return []
    grid_blocks, _grid_tables = _reconstruct_ocr_grid(items)
    if grid_blocks:
        for b in grid_blocks:
            b.page = page_index
        _log_number_fixes(log, page_index, fixed_count, fixed_examples)
        return grid_blocks
    blocks: list[Block] = []
    for y0, x0, x1, y1, text in _order_blocks(items):
        if _is_pure_symbol(text):
            continue
        size = min(_MAX_FONT, max(5.0, (y1 - y0) / 1.2))
        blocks.append(
            Block(
                text=text, page=page_index, x0=x0, y0=y0, x1=x1, y1=y1,
                size=round(size, 2), align="left", bold=False, single_line=True,
                ocr=True,
            )
        )
    _log_number_fixes(log, page_index, fixed_count, fixed_examples)
    return blocks


def _log_number_fixes(log, page_index: int, count: int, examples: list[str]) -> None:
    """Report OCR number fixes so a wrong repair is at least visible in the log."""
    if not count or not log:
        return
    log(
        f"  第 {page_index + 1} 页：OCR 数字格式异常已修正 {count} 处"
        + (f"（示例：{'；'.join(examples)}）" if examples else "")
        + "。若与报表实际金额不符，请人工复核。"
    )


def _render_page_png(page, dpi: int = 200) -> bytes:
    """Render ``page`` to a PNG (used by the agent's ``render_page`` / preview)."""
    return page.get_pixmap(dpi=dpi).tobytes("png")




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
    log: Callable[[str], None] | None = None,
) -> list[Block]:
    """OCR one page and return its blocks (empty if OCR is unavailable/failed).

    ``ocr_fn`` is injected by tests (returns ``[(box, text)]`` in PDF points);
    production falls back to the shared RapidOCR engine.  ``cancel`` is polled
    before the (potentially slow) render so a cancelled run does not start a
    page it will never use.

    A recognition failure degrades to "no text on this page" — one bad page
    must not kill the run — but the reason is reported through ``log`` instead
    of being swallowed: a silent ``except`` here looks exactly like a scan that
    contains no text, which is impossible to diagnose from the outside.
    :class:`TranslationCancelled` is deliberately re-raised: it is a control
    signal, not a page-level failure.
    """
    if cancel is not None and cancel():
        raise TranslationCancelled()
    if ocr_fn is not None:
        try:
            return _synthesize_ocr_blocks(
                list(ocr_fn(page_index, page)), page_index, log,
                page_height=page.rect.height,
            )
        except TranslationCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 — one bad page must not abort
            if log:
                log(f"  第 {page_index + 1} 页 OCR 失败：{type(exc).__name__}: {exc}")
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
        return _synthesize_ocr_blocks(
            results, page_index, log, page_height=page.rect.height
        )
    except TranslationCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 — one bad page must not abort
        if log:
            log(f"  第 {page_index + 1} 页 OCR 失败：{type(exc).__name__}: {exc}")
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
#: A short ``Label:`` row / heading that begins a table entry.  The colon may be
#: followed by a space **or end the line** (a data label with no inline value),
#: so rows like ``Powerplant:`` stay on their own single-line block.
_LABEL_RE = re.compile(r"^\s*[A-Za-z\u4e00-\u9fff\u4e00-\u9fa5][^:\n]{0,40}:(?:\s|$)")
#: A Chinese ``标签：值`` field row: a short label then the *full-width* colon
#: (U+FF1A) immediately followed by a value.  The ASCII ``_LABEL_RE`` needs ``:``
#: followed by whitespace/EOF, so it misses ``联系地址：…`` / ``联系电话：…`` —
#: which left a whole "company basic info" subsection collapsed into one block.
_CN_LABEL_RE = re.compile(r"^\s*[A-Za-z\u4e00-\u9fff\u4e00-\u9fa5][^:\n：]{0,14}：")
#: Chinese ordinal / enumeration markers that begin a list or table item, which
#: the Arabic ``_NUM_RE`` (``1.`` / ``1)``) does not recognise: ``（一）（二）…``,
#: ``①…`` and ``一、二、…``.  Without these, a tightly-spaced Chinese subsection
#: used to collapse into one run-on paragraph (its entries did not start a new
#: block).
_CN_DIGITS = "一二三四五六七八九十百零〇0-9"
_CN_ORDINAL_RE = re.compile(
    r"^\s*(?:"
    r"[（(]\s*[" + _CN_DIGITS + r"]{1,4}\s*[）)]"      # （一）（十二）(5)
    r"|[\u2460-\u2473]"                                 # ①..⑳
    r"|[" + _CN_DIGITS + r"]{1,3}[、．]"                  # 一、 十二、
    r")"
)



#: Characters that make a block worth translating: a letter, a digit or a CJK
#: ideograph.  A block with none of these (e.g. a stray ``。``, ``%``, ``①`` or a
#: decorative bullet) carries no translatable content and is dropped instead of
#: being sent to the model — otherwise it is translated into a stray glyph that
#: shows up in the output (a lone ``.`` or a spurious ``%`` after the text).
_MEANINGFUL_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


def _is_pure_symbol(text: str) -> bool:
    """True when ``text`` has no letters, digits or CJK ideographs (only symbols)."""
    return not bool(_MEANINGFUL_RE.search(text))


def _is_entry(text: str) -> bool:
    """True when ``text`` begins a new list / table entry (bullet, number, ``Label:``)."""
    t = text.strip()
    if not t:
        return False
    return bool(
        _BULLET_RE.match(t)
        or _DASH_BULLET_RE.match(t)
        or _NUM_RE.match(t)
        or _CN_ORDINAL_RE.match(t)
        or _LABEL_RE.match(t)
        or _CN_LABEL_RE.match(t)
    )


def _collect_lines(page_dict: dict) -> list[dict]:
    """Return one record per visual line (a PyMuPDF ``line``), with bbox and
    style hints used to decide paragraph vs. entry grouping.

    ``page_dict`` is the result of ``page.get_text("dict")`` (fetched once per
    page by the caller and shared with :func:`_collect_spans`).
    """
    out: list[dict] = []
    for b in page_dict.get("blocks", []):
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
            # Per-span records (``x0, y0, x1, y1, text``) let the table block
            # builder split one visual line across the cells it actually spans
            # (a ``Capacity: | value`` row is a single line in the text layer but
            # two table cells) instead of pinning the whole line to one cell.
            span_records = [
                (
                    float(s["bbox"][0]),
                    float(s["bbox"][1]),
                    float(s["bbox"][2]),
                    float(s["bbox"][3]),
                    "".join(s["text"]).strip(),
                )
                for s in spans
            ]
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
                    "spans": span_records,
                }
            )
    return out


def _order_generic(
    items: Sequence,
    x0: Callable,
    x1: Callable,
    y0: Callable,
) -> list:
    """Order items into reading order (column by column).

    Shared by :func:`_order_lines` (native text lines) and
    :func:`_order_blocks` (OCR boxes); ``x0``/``x1``/``y0`` are accessors for
    the item's box.  Items are clustered into columns by their x range and
    read column by column (left column top to bottom, then the next).

    Full-width items (a heading or rule spanning both columns) are pulled out
    *before* clustering, but only when the page really is multi-column: in the
    greedy clustering a wide item becomes a column whose right edge is the
    page width, and every line of both real columns then "overlaps" it —
    merging the two columns into one and interleaving their text.  A
    full-width item is wider than 1.5× the median item width, so on a
    single-column page (all items about equally wide) nothing is ever
    misclassified and the plain top-to-bottom order is kept.
    """
    if len(items) < 2:
        return list(items)

    med_w = statistics.median(x1(it) - x0(it) for it in items)
    if med_w <= 0:
        return sorted(items, key=lambda it: (round(y0(it), 1), x0(it)))
    threshold = 1.5 * med_w
    full = [it for it in items if (x1(it) - x0(it)) > threshold]
    rest = [it for it in items if (x1(it) - x0(it)) <= threshold]

    def rows(seq: Sequence) -> list:
        return sorted(seq, key=lambda it: (round(y0(it), 1), x0(it)))

    xsorted = sorted(rest, key=lambda it: x0(it))
    columns: list = []
    col_max_x1: list[float] = []
    for it in xsorted:
        for c in range(len(columns)):
            if x0(it) < col_max_x1[c] - 2.0:
                columns[c].append(it)
                col_max_x1[c] = max(col_max_x1[c], x1(it))
                break
        else:
            columns.append([it])
            col_max_x1.append(x1(it))

    if len(columns) <= 1:
        # Single-column page (or no non-full-width items): plain reading order,
        # full-width items included in their natural y position.
        return rows(items)

    ordered: list = []
    for col in columns:
        ordered.extend(rows(col))

    if full:
        # A full-width item above the columns (a top heading) reads first; one
        # at or below them (a footer / mid-page banner) reads after.
        first_y = y0(ordered[0])
        front = rows([it for it in full if y0(it) < first_y])
        back = rows([it for it in full if y0(it) >= first_y])
        ordered = front + ordered + back
    return ordered


def _order_lines(lines: Sequence[dict]) -> list[dict]:
    """Order visual lines into reading order (column by column)."""
    return _order_generic(
        lines,
        lambda ln: ln["x0"],
        lambda ln: ln["x1"],
        lambda ln: ln["y0"],
    )


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
    # A vertical gap larger than ~1.1× the line height marks a paragraph / entry
    # break.  This is deliberately looser than the old 0.5× (which split every
    # line of a loose-leading CJK paragraph into its own single-line block,
    # forcing translations into tiny boxes): table lines are already routed to
    # the table block builder, so this threshold only decides where a flowing
    # prose paragraph ends.
    if cur["y0"] - prev["y1"] > 1.1 * base["size"]:
        return True
    # Side-by-side cells of a table row: two lines sharing a baseline but sitting
    # in different horizontal regions are distinct cells, not one wrapped line.
    # Keeping them as separate blocks lets each cell be drawn at its own column
    # position (PyMuPDF merges a whole row into one block, which a translation
    # then collapses into a single jumbled column).  The same-row tolerance is a
    # fraction of the line height so a genuinely multi-line paragraph is untouched.
    if (
        abs(cur["y0"] - prev["y0"]) <= max(1.0, 0.4 * base["size"])
        and (cur["x0"] > prev["x1"] + 1.0 or prev["x0"] > cur["x1"] + 1.0)
    ):
        return True
    return False


def _collect_spans(page_dict: dict) -> list[tuple[fitz.Rect, float, bool]]:
    """Return ``(bbox, size, bold)`` for every text span on the page.

    ``page_dict`` is the result of ``page.get_text("dict")`` (fetched once per
    page by the caller and shared with :func:`_collect_lines`).
    """
    spans: list[tuple[fitz.Rect, float, bool]] = []
    for b in page_dict.get("blocks", []):
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
    """Sort blocks into reading order (see :func:`_order_generic`).

    Items are ``(y0, x0, x1, y1, text)`` tuples.  A multi-column page is read
    column by column (left column top to bottom, then the next column) instead
    of interleaving the columns row by row; single-column pages keep the plain
    top-to-bottom, left-to-right order.
    """
    return _order_generic(
        text_blocks,
        lambda b: b[1],
        lambda b: b[2],
        lambda b: b[0],
    )


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


def _line_center_in_rects(ln: dict, rects: Sequence[fitz.Rect]) -> bool:
    """True when a line's bbox centre falls inside one of ``rects`` (a table cell)."""
    cx = (ln["x0"] + ln["x1"]) / 2.0
    cy = (ln["y0"] + ln["y1"]) / 2.0
    for r in rects:
        if r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1:
            return True
    return False


def _cell_for_line(ln: dict, rects: Sequence[fitz.Rect]) -> fitz.Rect | None:
    """Return the ruled-table cell rect containing ``ln``'s centre, or ``None``.

    Cells of a detected table never overlap in 2-D, so testing both the x and the
    y centre pins the line to exactly one cell (two tables stacked on a page each
    claim a different y band even when their columns share x edges).
    """
    cx = (ln["x0"] + ln["x1"]) / 2.0
    cy = (ln["y0"] + ln["y1"]) / 2.0
    for r in rects:
        if r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1:
            return fitz.Rect(r)
    return None


def _detect_table_cell_rects(page) -> list[fitz.Rect]:
    """Return every detected table cell's rect on ``page`` (empty when no table).

    PyMuPDF's ``find_tables`` reliably separates the cells of a ruled table.  We
    use it to (a) build each cell as its own block so a translation lands in the
    right column, and (b) exclude the table's lines from the prose pipeline, so
    that pipeline can safely merge flowing paragraphs (its vertical-gap break is
    too tight for the loose line leading this report uses).  A page with no
    detectable table returns ``[]`` and the caller falls back to the old path.
    """
    try:
        tabs = page.find_tables()
    except Exception:
        return []
    rects: list[fitz.Rect] = []
    for t in getattr(tabs, "tables", None) or []:
        for row in getattr(t, "rows", None) or []:
            # ``find_tables`` may report a ``None`` cell for a ragged / empty
            # slot (e.g. a merged or missing column).  Casting that to a Rect
            # raised ``AssertionError: arg=None ret=(None,)`` on some PDFs
            # (``demo.pdf``, the DracoX manual), aborting extraction.  Skip any
            # falsy cell instead, exactly as ``_extract_tables`` already does.
            for cell in getattr(row, "cells", None) or []:
                if cell:
                    rects.append(fitz.Rect(cell))
    return rects


def _table_cell_key(ln: dict) -> tuple:
    # Row-major order for a table's cells: group rows by rounded baseline, then
    # read left-to-right across the columns of each row.
    return (round(ln["y0"], 0), ln["x0"])


def _group_to_block(
    group: Sequence[dict], spans, page_x0: float, page_x1: float, page_index: int
) -> Block | None:
    """Merge a group of ordered lines into one :class:`Block`, or ``None`` to skip."""
    if not group:
        return None
    x0 = min(ln["x0"] for ln in group)
    y0 = min(ln["y0"] for ln in group)
    x1 = max(ln["x1"] for ln in group)
    y1 = max(ln["y1"] for ln in group)
    text = " ".join(ln["text"] for ln in group)
    if not text.strip() or _is_pure_symbol(text):
        return None
    meta = _block_meta(fitz.Rect(x0, y0, x1, y1), spans, page_x0, page_x1)
    color = Counter(ln["color"] for ln in group).most_common(1)[0][0]
    return Block(text=text, page=page_index, x0=x0, y0=y0, x1=x1, y1=y1, color=color, **meta)


#: Horizontal gutter (points) kept between a table cell's text and its ruled
#: borders.  The one-line fit shrinks the font to *fill* the cell width, so
#: without a gutter the text sits flush against the borders (two long header
#: cells were visually colliding across the shared rule).  A small inset keeps
#: the text clear of the lines while still letting it use almost the whole
#: column.
_TABLE_CELL_PAD = 2.0

#: A ruled-table cell narrower than this is padding between real columns (e.g.
#: the few-point gutter ``find_tables`` reports next to the page edge, or between
#: a label column and its value column), not a content cell.  Content must never
#: be fitted into one: a long row whose whole-line centre lands in such a gutter
#: (long labels like ``Max Gross Weight:``) was pinning the translation into a
#: ~1.4pt box and rendering it as an unreadable vertical stack.
_TABLE_CELL_MIN_WIDTH = 6.0


def _cell_for_span(rect: fitz.Rect, cell_rects: Sequence[fitz.Rect]) -> fitz.Rect | None:
    """Return the real content cell containing ``rect``'s centre (x and y), or
    ``None``.

    Only cells at least :data:`_TABLE_CELL_MIN_WIDTH` wide are considered, so a
    span is never assigned to a padding gutter.  Cells never overlap in 2-D, so
    checking both the x and the y centre pins the span to exactly one cell — a
    row's label span (its centre ends in the label column) and its value span
    (its centre is in the value column) split cleanly, and a full-width section
    header keeps its own spanning cell rather than being squeezed into a narrow
    label cell two rows down.
    """
    cx = (rect.x0 + rect.x1) / 2.0
    cy = (rect.y0 + rect.y1) / 2.0
    for r in cell_rects:
        if r.width < _TABLE_CELL_MIN_WIDTH:
            continue
        if r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1:
            return fitz.Rect(r)
    return None


def _line_cell_groups(
    ln: dict, cell_rects: Sequence[fitz.Rect]
) -> list[tuple[fitz.Rect | None, str, fitz.Rect]] | None:
    """Split one merged table line into per-cell ``(cell, text, extent)`` groups.

    Returns ``None`` when the line carries no per-span data (a caller-constructed
    line in a test), so the caller keeps the whole-line behaviour.  Groups are
    returned left to right by the first span that lands in each cell; a cell with
    no span produces no group.  A span whose centre is not in any real content
    cell yields a ``(None, ...)`` group drawn from its own extent (kept
    ``in_table=False`` so it falls back to paragraph fitting).
    """
    span_records = ln.get("spans")
    if not span_records:
        return None
    assigns: list[tuple[float, fitz.Rect | None, fitz.Rect, str]] = []
    for rec in span_records:
        x0, y0, x1, y1, t = rec
        if not (t and t.strip()):
            continue
        srect = fitz.Rect(float(x0), float(y0), float(x1), float(y1))
        assigns.append((srect.x0, _cell_for_span(srect, cell_rects), srect, t.strip()))
    if not assigns:
        return None
    assigns.sort(key=lambda a: a[0])
    groups: list[list] = []  # [key, texts, union]
    index: dict = {}
    for _, cell, srect, t in assigns:
        key = (cell.x0, cell.y0, cell.x1, cell.y1) if cell is not None else None
        if key in index:
            group = groups[index[key]]
            group[1].append(t)
            group[2] = group[2] | srect
        else:
            index[key] = len(groups)
            groups.append([key, [t], fitz.Rect(srect)])
    out: list[tuple[fitz.Rect | None, str, fitz.Rect]] = []
    for key, texts, union in groups:
        cell = None if key is None else fitz.Rect(*key)
        joined = " ".join(p for p in texts if p).strip()
        if joined:
            out.append((cell, joined, fitz.Rect(union)))
    return out or None


def _build_table_blocks(
    table_lines: Sequence[dict],
    spans,
    page_x0: float,
    page_x1: float,
    page_index: int,
    cell_rects: Sequence[fitz.Rect] = (),
) -> list[Block]:
    """Turn a table's per-cell lines into row-major per-cell blocks.

    A visual line can hold two cells on the same baseline (a label column and a
    value column, e.g. ``Capacity: | Pilot + Copilot + Two Passengers``): the
    text layer reports them as one line but ``find_tables`` sees two cells.  The
    line is split per cell so each cell becomes its own block (label stays in the
    label column, value in the value column) instead of the whole line being
    pinned to whichever cell its centre happens to land in — which dropped the
    label column entirely, and pinned long labels to the narrow gutter cell
    (a vertical stack).

    Each cell's block is widened to the full ruled-table cell (not the source
    text's own extent) and flagged ``in_table``, so the exporter can lay out the
    (usually longer) translation on one line within the whole column instead of
    wrapping it inside a too-narrow box.  Alignment is re-derived from the cell:
    figures hug the column's right edge (as in the ``股份类型`` share table),
    everything else is centred (as the source ``前十名股东`` table centres its
    cells).  The cell is inset by :data:`_TABLE_CELL_PAD` on each side so the
    one-line text does not touch the borders.  A line with no per-span data (a
    caller-constructed line) keeps the whole-line behaviour.
    """
    blocks: list[Block] = []
    cell_rects = list(cell_rects)
    for ln in sorted(table_lines, key=_table_cell_key):
        if not ln["text"].strip() or _is_pure_symbol(ln["text"]):
            continue
        x0, y0, x1, y1 = ln["x0"], ln["y0"], ln["x1"], ln["y1"]
        groups = _line_cell_groups(ln, cell_rects)
        if groups is None:
            # No per-span info (test-constructed line): keep the single-block,
            # whole-line behaviour so a whole row is one cell.
            meta = _block_meta(fitz.Rect(x0, y0, x1, y1), spans, page_x0, page_x1)
            cell = _cell_for_line(ln, cell_rects)
            if cell is not None:
                x0, x1 = cell.x0 + _TABLE_CELL_PAD, cell.x1 - _TABLE_CELL_PAD
                # Re-derive alignment for the widened cell: numeric columns
                # right-align (figures share a flush right edge), others centre.
                meta["align"] = "right" if _is_numeric_cell(ln["text"]) else "center"
            blocks.append(
                Block(
                    text=ln["text"], page=page_index, x0=x0, y0=y0, x1=x1, y1=y1,
                    color=ln["color"], in_table=cell is not None, **meta,
                )
            )
            continue
        for cell, text, extent in groups:
            if cell is not None:
                cx0, cx1 = cell.x0 + _TABLE_CELL_PAD, cell.x1 - _TABLE_CELL_PAD
                meta = _block_meta(cell, spans, page_x0, page_x1)
                meta["align"] = "right" if _is_numeric_cell(text) else "center"
                blocks.append(
                    Block(
                        text=text, page=page_index, x0=cx0, y0=y0, x1=cx1, y1=y1,
                        color=ln["color"], in_table=True, **meta,
                    )
                )
            else:
                # A span that sits outside any content cell: paragraph-style block.
                meta = _block_meta(extent, spans, page_x0, page_x1)
                blocks.append(
                    Block(
                        text=text, page=page_index, x0=extent.x0, y0=extent.y0,
                        x1=extent.x1, y1=extent.y1, color=ln["color"],
                        in_table=False, **meta,
                    )
                )
    return blocks


def group_by_page(block_pages: Sequence[int], values: Sequence[str], page_count: int) -> list[list[str]]:
    """Regroup a flat ``values`` list back into per-page lists.

    ``block_pages`` and ``values`` must be parallel (same length); a mismatch
    used to be silently truncated by ``zip``, dropping content with no error.
    """
    if len(block_pages) != len(values):
        raise ValueError(
            f"block_pages ({len(block_pages)}) and values ({len(values)}) "
            "must be parallel lists"
        )
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
#: here; the shrink loop below keeps every block inside its box).  Raised to 40
#: so large display headings (cover titles, big chapter heads) are not crushed
#: to a uniform 24pt.
_MAX_FONT = 40.0

#: Smallest size a translation is allowed to shrink to.  Going below this makes
#: the rendered text barely legible; when a translation still does not fit its
#: source box at this size it is drawn slightly past the box (the original text
#: behind it is redacted) rather than crushed into an unreadable sliver.
_MIN_READABLE = 7.0

#: Smallest size a *table cell* translation may shrink to so it fits on **one
#: line**.  A cell whose English translation is wider than its column wraps and
#: grows the row, which pushes every row beneath it down and misaligns the
#: table (the reported defect).  A cell is allowed to go a little smaller than
#: the general ``_MIN_READABLE`` floor because a one-line figure/name in a
#: reference table is worth the slightly reduced size.
_MIN_TABLE_READABLE = 6.0

#: Absolute floor a table cell shrinks to before its single line just overflows
#: the column instead.  The same 3pt figure the draw loop treats as the
#: readability limit: below it a line is unreadable, and overflowing by a hair
#: is cheaper than splitting a table row — the original scan is also one line
#: per cell, so a wrapped cell is a visible misalignment, not a readable fix.
_MIN_TABLE_FLOOR = 3.0

def _render_note(page: fitz.Page, font, lang: str) -> None:
    """Write the 'nothing to translate on this page' note."""
    note = "（本页无可翻译文本 / No translatable text on this page）"
    tw = fitz.TextWriter(page.rect)
    tw.append(fitz.Point(_MARGIN, _MARGIN + 14), note, font=font, fontsize=_FONT_SIZE)
    tw.write_text(page)


def _fit_one_line(
    font, text: str, max_width: float, start_fs: float,
    floor: float = _MIN_TABLE_READABLE,
) -> tuple[list[str], float] | None:
    """Largest font size that fits ``text`` on a single line within ``max_width``.

    Returns ``([text], fs)`` when a readable single line is possible, or ``None``
    when even the smallest allowed size still overflows (the caller then falls
    back to wrapping).  ``start_fs`` is the font size to try first (the source
    size); it never grows beyond that.  A binary search finds the largest fitting
    size, so a cell that only needs a small reduction keeps as much size as
    possible (instead of stepping down in coarse 0.9× jumps).

    ``floor`` is the search's readability limit (defaults to the table floor;
    :func:`_fit_block` passes ``_MIN_TABLE_FLOOR`` when a single line is
    preferred over wrapping).
    """
    floor = min(start_fs, floor)
    if font.text_length(text, fontsize=start_fs) <= max_width:
        return [text], start_fs
    if font.text_length(text, fontsize=floor) > max_width:
        # Even the readability floor is too wide — a single line is impossible.
        return None
    lo, hi = floor, start_fs
    for _ in range(32):
        mid = (lo + hi) / 2
        if font.text_length(text, fontsize=mid) <= max_width:
            lo = mid
        else:
            hi = mid
    # Round *down* to 2 decimals: rounding up (``round``) could push the line a
    # hair past ``max_width`` (e.g. 200.0086 > 200) which shows up as text just
    # touching a cell border.  Flooring guarantees it stays within the width.
    fs = int(lo * 100) / 100.0
    return [text], fs


def _fit_exact_n(
    font, text: str, max_width: float, start_fs: float, n: int,
) -> tuple[list[str], float]:
    """Produce **exactly** ``n`` lines, at the largest font size that allows it.

    ``n`` is the source cell's line count.  Greedy wrap gives the fewest lines
    for a size, so the size is searched down until the text needs at most ``n``
    lines, then the result is re-balanced to exactly ``n``: a too-short cell
    splits its longest line (never a whole figure — a number cut in half looks
    exactly like a digit grew or lost a decimal point), a too-long one merges
    back so the overflow spills horizontally instead of growing the row and
    misaligning the table.
    """
    text = " ".join(str(text).split())
    if n <= 1:
        return _fit_one_line(font, text, max_width, start_fs, floor=_MIN_TABLE_FLOOR) or (
            [text], _MIN_TABLE_FLOOR
        )
    floor = _MIN_TABLE_FLOOR

    def can(fs: float) -> bool:
        return len(_wrap(font, text, max_width, fs)) <= n

    if can(start_fs):
        fs = start_fs
    elif not can(floor):
        fs = floor
    else:
        lo, hi = floor, start_fs
        for _ in range(24):
            mid = (lo + hi) / 2
            if can(mid):
                lo = mid
            else:
                hi = mid
        fs = int(lo * 100) / 100.0
    lines = _wrap(font, text, max_width, fs)
    if len(lines) > n:
        lines = _merge_to_n(lines, n)
    elif len(lines) < n:
        lines = _rebalance_to_n(lines, font, fs, n)
    return lines, fs


def _merge_to_n(lines: Sequence[str], n: int) -> list[str]:
    """Merge lines until exactly ``n``, always joining the shortest pair."""
    out = list(lines)
    while len(out) > n:
        i = min(
            range(len(out) - 1),
            key=lambda j: len(out[j]) + len(out[j + 1]),
        )
        out[i:i + 2] = [out[i] + " " + out[i + 1].lstrip()]
    return out


def _rebalance_to_n(lines: Sequence[str], font, fs: float, n: int) -> list[str]:
    """Split lines until exactly ``n``, starting from the widest splittable."""
    out = list(lines)
    while len(out) < n:
        idxs = [i for i in range(len(out)) if _split_line_half(out[i]) is not None]
        if not idxs:
            break  # only atomic figures left: keep the shorter (and correct) count
        i = max(idxs, key=lambda j: font.text_length(out[j], fontsize=fs))
        a, b = _split_line_half(out[i])
        out[i:i + 1] = [a, b]
    return out


def _split_line_half(line: str) -> tuple[str, str] | None:
    """Split ``line`` near its middle, preferring a space boundary.

    Returns ``None`` when the line cannot usefully be cut: a single character,
    or a whole figure atom — a number broken in half looks exactly like a digit
    grew or lost a decimal point (the same reason a figure is never wrap-broken
    by :func:`_break_word`).
    """
    t = line.strip()
    if len(t) < 2 or _is_number_atom(t):
        return None
    half = len(t) // 2
    best = None
    best_d = 1e9
    for i in range(1, len(t)):
        if t[i] == " " or t[i - 1] == " ":
            d = abs(i - half)
            if d < best_d:
                best_d, best = d, i
    if best is None:
        # A space-less CJK run: CJK tolerates a cut anywhere, and rebalancing
        # only fires when the count must match the source — never for a figure.
        best = half
    a, b = t[:best].rstrip(), t[best:].lstrip()
    if not a or not b:
        return None
    return a, b


#: Loose leading (baseline-to-baseline) for paragraph blocks and single-line
#: cells: 1.35× the font size, keeping the source layout's airy look.
_LOOSE_LEADING = 1.35

#: Leading (baseline-to-baseline) for a *multi-line* table cell: 1.0× the font
#: size — the tightest spacing that keeps a run of lines readable while staying
#: compact, so a long translation in a narrow column fits the row band without
#: growing the row.  (Below this, characters start to overlap; 1.0 is the
#: requested minimum.)
_TABLE_CELL_LEADING = 1.0


def _line_leading(font, *, in_table: bool, n_lines: int) -> float:
    """Baseline-to-baseline multiplier for a block's wrapped lines.

    A *multi-line* table cell uses the tight ``_TABLE_CELL_LEADING`` (1.0× font
    size) instead of the loose 1.35× leading that paragraphs and single-line
    cells keep: a long translation in a narrow column must stay compact so the
    wrap fits the row band without growing the row or pushing the rows beneath
    it down.  A single line has no inter-line gap, so its leading never matters.
    """
    if in_table and n_lines > 1:
        return _TABLE_CELL_LEADING
    return _LOOSE_LEADING


def _wrapped_height(font, lines: Sequence[str], fs: float,
                    leading: float = _LOOSE_LEADING) -> float:
    """Vertical extent of ``lines`` at ``fs`` (same metric ``_fit_block`` uses).

    ``leading`` is the baseline-to-baseline multiplier; pass
    :func:`_line_leading`'s value for a table cell so this measurement agrees
    with the drawing pass (which decides the multiplier the same way).
    """
    return (
        fs * font.ascender + (len(lines) - 1) * fs * leading - fs * font.descender
    )


def _fit_band(
    font, text: str, max_width: float, band: float
) -> tuple[list[str], float]:
    """Largest readable font size whose wrapped translation fits in ``band`` pt.

    A scanned grid's raster lines cannot move (a grown row would overlap the
    pixels below and the signature ink stays pinned to the scan), so a cell
    that cannot hold a readable single line wraps *down* into the row gap, and
    the wrap must stop at ``band`` — the distance to the next grid line.  The
    wrap at ``_MIN_TABLE_READABLE`` is tried first; the search then descends
    (smaller sizes pack more lines) until the wrapped height fits the band.
    ``_MIN_TABLE_FLOOR`` only bounds the descent.
    """
    # A scanned cell's wrap always yields at least two lines, so its leading is
    # the tight ``_TABLE_CELL_LEADING`` (see ``_line_leading``).  Using it here
    # keeps the band check in agreement with the drawing pass.
    leading = _TABLE_CELL_LEADING
    lines = _wrap(font, text, max_width, _MIN_TABLE_READABLE)
    if _wrapped_height(font, lines, _MIN_TABLE_READABLE, leading) <= band:
        return lines, _MIN_TABLE_READABLE
    # Search the 0.01 grid explicitly (a plain float bisection then rounded to
    # 2dp could land 0.02pt over the band — the whole point is staying inside).
    lo, hi = int(_MIN_TABLE_FLOOR * 100), int(_MIN_TABLE_READABLE * 100)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        fsm = mid / 100
        if _wrapped_height(font, _wrap(font, text, max_width, fsm), fsm, leading) <= band:
            lo = mid
        else:
            hi = mid - 1
    fs = lo / 100
    return _wrap(font, text, max_width, fs), fs


def _fit_block(block: Block, font, text: str) -> tuple[list[str], float]:
    """Measure a block's translation without drawing it.

    Returns ``(lines, fontsize)``: the wrapped lines and the font size the
    translation would render at inside ``block``'s box (see
    :func:`_draw_translated_block` for the rules).  This is shared by the
    drawing path and the table row-height re-layout so the two never disagree
    on how much vertical space a block needs.

    A table cell (``block.in_table``) keeps the source line count where that is
    *compatible with readability*: a shrinking search never goes below
    ``_MIN_TABLE_READABLE``.  A one-line source that cannot hold a readable
    single line wraps at the readability floor instead of collapsing to the
    3pt floor — the old rule produced 44 slug cells (< 5pt) on the scan — and
    when ``block.fit_height`` (> 0) bounds the wrap by the scan's row gap, the
    translation stops before the grid line of the row below instead of
    crossing it (raster lines and signature ink cannot move).  A two-line
    source cell still comes back as exactly two lines (``_fit_exact_n``).
    ``block.fit_width`` (when > 0) widens the fit beyond the glyph box — a
    scanned header cell's own OCR box is much narrower than its real column,
    and fitting against it crushes the translation to the floor.  Only a
    non-table block takes the wrap-and-fit path below.
    """
    r = fitz.Rect(block.x0, block.y0, block.x1, block.y1)
    max_width = getattr(block, "fit_width", 0.0) or max(1.0, r.width)
    fs = max(5.0, min(block.size, _MAX_FONT))

    if getattr(block, "in_table", False):
        # The line count of the SOURCE cell (grid cells are one OCR line; a
        # text-layer cell may span several source lines joined by \n).
        n_lines = max(1, len(block.text.split("\n")))
        # Flatten: the translation's own newlines don't define the count — the
        # source does (and single-line cells must not hide stray breaks).
        flat = " ".join(str(text).split())
        if n_lines == 1:
            one_line = _fit_one_line(font, flat, max_width, fs)
            if one_line is not None:
                return one_line
            # A readable single line is impossible.  The old rule then shrank
            # to ``_MIN_TABLE_FLOOR`` and kept one line — the 3pt slug cells.
            # Wrap at the readability floor instead; a scan row band
            # (``fit_height``) bounds the wrap so it stops before the grid
            # line of the row below (see ``_fit_band``).
            lines = _wrap(font, flat, max_width, _MIN_TABLE_READABLE)
            band = getattr(block, "fit_height", 0.0)
            if band > 0.0:
                return _fit_band(font, flat, max_width, band)
            return lines, _MIN_TABLE_READABLE
        return _fit_exact_n(font, flat, max_width, fs, n_lines)

    lines = _wrap(font, text, max_width, fs)

    def height() -> float:
        return (fs * font.ascender + (len(lines) - 1) * fs * _LOOSE_LEADING
                - fs * font.descender)

    # Trim the font so the translation's height no longer exceeds the box it
    # replaces (a smaller font also wraps to fewer lines).
    #
    # The floor is a *readability* floor, not a no-overflow guarantee: a
    # pathological translation (far longer than the source, in a very tight
    # box) can still exceed the box at ``_MIN_READABLE``, and it is then drawn
    # overflowing rather than shrunk to an illegible size — losing content *or*
    # ending up unreadable would be worse.  ``max(floor, ...)`` clamps the step
    # so the loop never goes below the floor (a bare ``round(fs * 0.9, 2)``
    # could undershoot it, e.g. 7.2 -> 6.48).
    floor = min(fs, _MIN_READABLE)
    while fs > floor and height() > r.height + 1.0:
        fs = max(floor, round(fs * 0.9, 2))
        lines = _wrap(font, text, max_width, fs)
    return lines, fs


def _is_vertical_label(block: Block) -> bool:
    """True for a narrow-tall box that holds a vertically-typeset label.

    Org-chart boxes in Chinese layouts are drastically taller than they are
    wide (measured: ~7.4 x 29.5 pt) and their labels read vertically.  The
    x-y box shape, not the text, is the signal — it must be a single line and
    outside any table grid (a ruled cell can be narrow too, but it gets the
    cell's single-line fit instead of a rotation).
    """
    if block.in_table or not block.single_line or "\n" in block.text:
        return False
    w = block.x1 - block.x0
    h = block.y1 - block.y0
    return h >= 2.0 * w and w <= 30.0 and h >= 22.0


def _draw_vertical_label(page: fitz.Page, font, block: Block, text: str) -> None:
    """Draw ``text`` rotated 90° inside/around ``block``'s narrow-tall box.

    Rotation is the only readable option for such a box: a horizontal draw has
    to hyphenate the label *per character* (``P- ar- ty a- n- d M- as- s``),
    which is what the org chart looked like.  The run is fitted along the
    box's height first; when even the floor is too long, it is drawn at the
    *column* size anyway and extends past the box (centred on it) — a small
    whole label reads at zoom, where per-character shards never do.

    Exactly the same glyph layout as :func:`fitz.Page.insert_text` with
    ``rotate=90`` is used (verified: with ``morph=(P, Matrix(0, 1, -1, 0, 0,
    0))`` and the writer anchored at ``P``, glyphs land at the identical
    coordinates), just via ``TextWriter`` so the page/off-page font stays the
    shared ``_CJK_FONT``: the run flows upward from ``P.y`` (occupying
    ``y ∈ [P.y-L, P.y]``) and the glyph column spans
    ``x ∈ [P.x-ascent, P.x+descent]``.
    """
    r = fitz.Rect(block.x0, block.y0, block.x1, block.y1)
    fs = max(5.0, min(block.size, _MAX_FONT))
    # Fit the run along the box *height* first — a tall box usually needs no
    # reduction.
    fitted = _fit_one_line(font, text, max(1.0, r.height), fs)
    if fitted is None:
        # Too long even for the height at the floor: shrink to the column
        # width (below the readability floor — a 4pt one-line label in a 7 pt
        # column beats ``a- n- d`` shards) and let it overflow vertically.
        run_fs = _MIN_READABLE
    else:
        _, run_fs = fitted
    asc_desc = font.ascender - font.descender
    if asc_desc > 0:
        run_fs = min(run_fs, max(3.0, r.width / asc_desc))
    length = font.text_length(text, fontsize=run_fs)
    ascent = run_fs * font.ascender
    descent = -run_fs * font.descender
    px = r.x0 + (r.width + ascent - descent) / 2.0
    py = r.y0 + (r.height + length) / 2.0
    # An over-long run must stay on the page (glyphs past the crop get clipped
    # out of the extracted and rendered text): push the run's bottom so it
    # starts near the page top rather than sailing off the sheet.
    pmin = page.rect.y0 + 2.0 + length
    pmax = page.rect.y1 - 2.0
    if pmax >= pmin:
        py = min(max(py, pmin), pmax)
    else:  # the run itself is taller than the page
        py = pmin
    tw = fitz.TextWriter(page.rect)
    tw.append(fitz.Point(px, py), text, font=font, fontsize=run_fs)
    tw.write_text(
        page,
        color=_color_tuple(block.color) if block.color else None,
        morph=(fitz.Point(px, py), fitz.Matrix(0, 1, -1, 0, 0, 0)),
    )


def _draw_translated_block(page: fitz.Page, font, block: Block, text: str) -> None:
    """Draw ``text`` into ``block``'s box, mirroring the original layout.

    The glyph box (ascent + lines + descent) is anchored to the block's bbox
    top; the font size starts at the block's original size and is trimmed
    until the wrapped text fits the box height.  Single-line boxes centre the
    translation vertically while paragraph blocks stay top-anchored like the
    source.  The source text colour is restored so red/gold headings are not
    flattened to black.
    """
    if _is_vertical_label(block):
        _draw_vertical_label(page, font, block, text)
        return
    r = fitz.Rect(block.x0, block.y0, block.x1, block.y1)
    lines, fs = _fit_block(block, font, text)
    ascent = fs * font.ascender
    descent = -fs * font.descender
    leading = _line_leading(font, in_table=block.in_table, n_lines=len(lines))

    def height() -> float:
        return ascent + (len(lines) - 1) * fs * leading + descent

    y = r.y0 + ascent
    # Multi-line text in a table cell hugs the cell's top rule instead of being
    # vertically centred: a centred wrap can push the last line down toward the
    # cell's lower boundary (and, for a scanned *row band*, out of the band),
    # while top-anchoring keeps the block inside and leans on the empty space
    # below.  A single-line cell stays centred as before.
    if block.single_line and not (block.in_table and len(lines) > 1):
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
        y += fs * leading
    tw.write_text(page, color=_color_tuple(block.color) if block.color else None)


def _has_latin(text: str) -> bool:
    """True when ``text`` contains a Latin letter (so it can be hyphenated)."""
    for ch in text:
        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            return True
    return False


#: A figure cell: digits and value punctuation only (plus a leading sign /
#: trailing percent).  Everything else (letters, CJK) is prose and wraps.
_NUM_ATOM_RE = re.compile(r"^[+-]?[0-9][0-9,.\s/%]*[%％]?$")


def _is_number_atom(text: str) -> bool:
    """True when ``text`` is a *whole* single number (wrap it, never split it)."""
    return bool(_NUM_ATOM_RE.match(text))


def _color_tuple(color: int) -> tuple[float, float, float]:
    """Convert a 24-bit color int (PyMuPDF span ``color``) to a float RGB triple."""
    return (
        ((color >> 16) & 255) / 255.0,
        ((color >> 8) & 255) / 255.0,
        (color & 255) / 255.0,
    )


def _wrap(font, text: str, width: float, fontsize: float) -> list[str]:
    """Greedy word-wrap that also breaks long words (e.g. CJK) by character.

    ``width`` is the maximum allowed line width.  Words are kept intact where
    possible; a single word that is wider than ``width`` (typical for CJK text,
    which has no spaces, so a whole paragraph is one giant "word") is broken into
    character pieces so that long translations never overflow the page.
    """
    lines: list[str] = []
    for paragraph_line in str(text).split("\n"):
        # A separator followed by a space and more digits (``65, 334, 085.99``)
        # is one amount OCR saw with stray padding — merging it keeps the figure
        # from being broken into ``65,`` / ``334,`` / ``085.99``.  Only an
        # explicit ``,``/``.`` before the space qualifies, so ``10 000`` (a
        # space used as a thousands separator) is left alone.
        paragraph_line = re.sub(r"(?<=\d)([.,])\s+(?=\d)", r"\1", paragraph_line)
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

    A Latin word is broken **with a hyphen** (preferring an explicit hyphen
    already inside it) so an English table entry like ``Non-Performing`` is no
    longer split into a bare ``Non-Perform`` + ``ing``.  A space-less CJK /
    symbol run is broken straight by glyph, as before.
    """
    if _has_latin(word):
        return _break_latin_word(font, word, width, fontsize, lines)
    if _is_number_atom(word):
        # A figure is never broken in two: ``292,712,933,925.17`` split as
        # ``...925.1`` / ``7`` looks exactly like a digit grew or lost a decimal
        # point, and in a financial statement a broken amount is worse than an
        # amount that overflows its box slightly.  Keep it whole on its own
        # line even when that line is a hair wider than the box.
        lines.append(word)
        return ""
    remaining = font.text_length(word, fontsize=fontsize)
    while word and remaining > width:
        acc = 0.0
        k = 0
        for i, ch in enumerate(word):
            wch = font.text_length(ch, fontsize=fontsize)
            if k > 0 and acc + wch > width:
                break
            acc += wch
            k = i + 1
        # ``k`` is always >= 1 here: the first character is unconditionally
        # taken, so the loop makes progress even when a single character
        # already exceeds ``width`` (a very narrow box) — that character is
        # emitted alone, exactly as the old binary search did.
        lines.append(word[:k])
        word = word[k:]
        remaining -= acc
    return word


def _break_latin_word(font, word: str, width: float, fontsize: float, lines: list[str]) -> str:
    """Break a too-wide Latin word, hyphenating at every piece boundary.

    Prefers to split at an explicit hyphen already inside the word (so
    ``Non-Performing`` → ``Non-`` + ``Performing``); otherwise it breaks at a
    character boundary and appends a trailing ``-`` (so nobody reads the
    unhyphenated ``Indicat or`` split the character breaker used to emit).
    """
    while word and font.text_length(word, fontsize=fontsize) > width:
        # 1. Split at an explicit hyphen that fits — best typography.
        best = None
        for i, ch in enumerate(word):
            if ch in ("-", "\u2010"):
                head = word[: i + 1]
                if font.text_length(head, fontsize=fontsize) <= width:
                    best = i + 1
        if best is not None:
            lines.append(word[:best])
            word = word[best:]
            continue
        # 2. Otherwise break at a character boundary, hyphenating the cut.
        acc = 0.0
        k = 0
        for i, ch in enumerate(word):
            wch = font.text_length(ch, fontsize=fontsize)
            if k > 0 and acc + wch > width:
                break
            acc += wch
            k = i + 1
        # Single-character pieces are unreadable shards (``P- ar- ty a- n- d``
        # was what the org-chart boxes produced): take at least two characters,
        # keep the tail at least two, and never hyphenate a word that comes out
        # at three characters or fewer — the overflow by a hair is the cheaper
        # failure.
        if len(word) <= 3:
            return word
        take = max(2, k)
        if take >= len(word) - 1:
            # A cut like ``tiv-`` + ``e`` leaves a lone trailing letter; take
            # one char less so the tail is ``ve``.
            take = max(2, len(word) - 2)
        lines.append(word[:take] + "-")
        word = word[take:]
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


def _reconstruct_ocr_tables(blocks: Sequence[Block]) -> list[dict]:
    """Rebuild a grid from OCR blocks when ``find_tables`` could not (a scan).

    ``save_translated_pdf`` has already split a scanned table page into cell
    blocks via ``_reconstruct_ocr_grid``; this re-derives the same structure from
    those blocks so the row-height expansion and grid redraw that the vector-table
    path gets also run for scans.  Non-OCR (or non-grid) pages return ``[]``.
    """
    ocr_blocks = [b for b in blocks if getattr(b, "ocr", False)]
    if len(ocr_blocks) < 4:
        return []
    items = [(b.y0, b.x0, b.x1, b.y1, b.text) for b in ocr_blocks]
    rows = _cluster_ocr_rows(items)
    if len(rows) < 2:
        return []
    cols = _cluster_ocr_columns(items)
    if len(cols) < 2:
        return []
    # Map each OCR block to the column whose extent contains its centre, then
    # group into rows by y, emitting a cell rect per (row, column) intersection.
    col_edges = sorted(
        {round(b.x0, 1) for b in ocr_blocks} | {round(b.x1, 1) for b in ocr_blocks}
    )
    row_rects: list[list[fitz.Rect]] = []
    for row in rows:
        top = min(it[0] for it in row)
        bottom = max(it[3] for it in row)
        cells: list[fitz.Rect] = []
        for col in cols:
            left = min(it[1] for it in col)
            right = max(it[2] for it in col)
            cells.append(fitz.Rect(left, top, right, bottom))
        row_rects.append(cells)
    bbox = fitz.Rect(
        min(b.x0 for b in ocr_blocks), min(b.y0 for b in ocr_blocks),
        max(b.x1 for b in ocr_blocks), max(b.y1 for b in ocr_blocks),
    )
    return [{"bbox": bbox, "rows": row_rects, "col_edges": col_edges}]


def _extract_tables(page) -> list[dict]:
    """Return every ruled table on ``page`` as ``{"bbox", "rows", "col_edges"}``.

    ``rows`` is the table's rows as a list of cell rects (left-to-right); the
    rects come straight from ``find_tables`` so they are authoritative.  A page
    with no detectable table returns ``[]``.
    """
    try:
        tabs = page.find_tables()
    except Exception:
        return []
    out: list[dict] = []
    for t in getattr(tabs, "tables", None) or []:
        rows: list[list[fitz.Rect]] = []
        for row in getattr(t, "rows", None) or []:
            cells = [fitz.Rect(c) for c in (getattr(row, "cells", None) or []) if c]
            if cells:
                rows.append(cells)
        if not rows:
            continue
        all_cells = [c for row in rows for c in row]
        bbox = fitz.Rect(
            min(c.x0 for c in all_cells), min(c.y0 for c in all_cells),
            max(c.x1 for c in all_cells), max(c.y1 for c in all_cells),
        )
        col_edges = sorted(
            {round(c.x0, 1) for c in all_cells} | {round(c.x1, 1) for c in all_cells}
        )
        out.append({"bbox": bbox, "rows": rows, "col_edges": col_edges})
    return out


def _map_blocks_to_table_cells(blocks, tables) -> dict[int, tuple[int, int]]:
    """Map block index -> ``(table_index, row_index)`` for every table cell block.

    A block belongs to a table when its bbox centre falls inside one of that
    table's cell rects.  Non-table blocks (headings, prose, footers) are absent.
    """
    mapping: dict[int, tuple[int, int]] = {}
    for bi, b in enumerate(blocks):
        cx = (b.x0 + b.x1) / 2.0
        cy = (b.y0 + b.y1) / 2.0
        for ti, tb in enumerate(tables):
            for ri, row in enumerate(tb["rows"]):
                for cell in row:
                    if cell.x0 <= cx <= cell.x1 and cell.y0 <= cy <= cell.y1:
                        mapping[bi] = (ti, ri)
                        break
    return mapping


def _measure_block_height(block: Block, font, text: str) -> float:
    """Height the translated ``text`` would occupy in ``block`` (via ``_fit_block``)."""
    lines, fs = _fit_block(block, font, text)
    leading = _line_leading(font, in_table=block.in_table, n_lines=len(lines))
    return fs * font.ascender + (len(lines) - 1) * fs * leading - fs * font.descender


def _compute_table_layout(tables, mapping, blocks, trans, font):
    """Work out how far every table row must be pushed down so translations fit.

    Returns ``(shifts, new_bottoms, grid, bboxes)``:

    * ``shifts``: ``block_index -> dy`` (downward offset for a table cell block).
    * ``new_bottoms``: ``block_index -> new row bottom y`` so the drawing pass can
      hand the cell its expanded box (letting the font shrink loop use the extra
      room instead of crushing the text).
    * ``grid``: ``("h"|"v", ...)`` line specs to redraw at the new positions.
    * ``bboxes``: each detected table's original extent, to be redacted so the
      stale grid lines don't stay behind at the old row positions.
    """
    if not tables:
        return {}, {}, [], []
    # Per table: measure the rows and the within-table cumulative shift.
    tinfo: list[dict] = []
    for ti, tb in enumerate(tables):
        row_count = len(tb["rows"])
        orig_top = [min(c.y0 for c in row) for row in tb["rows"]]
        orig_h = [max(c.y1 for c in row) - min(c.y0 for c in row) for row in tb["rows"]]
        needed_h = list(orig_h)
        for bi, key in mapping.items():
            if key[0] == ti and bi < len(trans):
                r = key[1]
                needed_h[r] = max(needed_h[r], _measure_block_height(blocks[bi], font, trans[bi]))
        cum: list[float] = []
        run = 0.0
        for r in range(row_count):
            cum.append(run)
            run += max(0.0, needed_h[r] - orig_h[r])
        tinfo.append(
            {
                "orig_top": orig_top, "needed_h": needed_h, "cum": cum,
                "extra": run, "bbox": tb["bbox"], "col_edges": tb["col_edges"],
            }
        )
    # Rigid-body push-down over the whole page: every element is pushed by the
    # total extra of the elements above it, so a heading between two tables (and
    # the lower table itself) shifts down by the growth of the upper table while
    # content above the highest table stays put.
    prose_blocks = [bi for bi in range(len(blocks)) if bi not in mapping]
    elems = [(tinfo[ti]["bbox"].y0, tinfo[ti]["extra"], ("table", ti)) for ti in range(len(tinfo))]
    elems += [(blocks[bi].y0, 0.0, ("prose", bi)) for bi in prose_blocks]
    elems.sort(key=lambda e: (e[0], e[2][0] == "prose"))
    base: dict[tuple, float] = {}
    run = 0.0
    i = 0
    while i < len(elems):
        y0 = elems[i][0]
        j = i
        while j < len(elems) and elems[j][0] == y0:
            j += 1
        for k in range(i, j):
            base[elems[k][2]] = run  # extras of strictly-smaller y0 only
        for k in range(i, j):
            run += elems[k][1]
        i = j
    shifts: dict[int, float] = {}
    new_bottoms: dict[int, float] = {}
    for bi, (ti, r) in mapping.items():
        t = tinfo[ti]
        shifts[bi] = base[("table", ti)] + t["cum"][r]
        new_bottoms[bi] = t["orig_top"][r] + shifts[bi] + t["needed_h"][r]
    for bi in prose_blocks:
        shifts[bi] = base.get(("prose", bi), 0.0)
    grid: list = []
    bboxes: list[fitz.Rect] = []
    for ti, t in enumerate(tinfo):
        base_shift = base[("table", ti)]
        left, right = t["bbox"].x0, t["bbox"].x1
        for r in range(len(t["orig_top"])):
            top = t["orig_top"][r] + base_shift + t["cum"][r]
            grid.append(("h", left, right, top))
            grid.append(("h", left, right, top + t["needed_h"][r]))
        new_top0 = t["orig_top"][0] + base_shift + t["cum"][0]
        new_bot_last = t["orig_top"][-1] + base_shift + t["cum"][-1] + t["needed_h"][-1]
        for x in t["col_edges"]:
            grid.append(("v", x, new_top0, new_bot_last))
        bboxes.append(fitz.Rect(left, t["bbox"].y0, right, t["bbox"].y1))
    return shifts, new_bottoms, grid, bboxes


def save_translated_pdf(
    src_path: str | Path,
    pages: Sequence[Sequence[Block]],
    per_page: Sequence[Sequence[str]],
    out_path: str | Path,
    lang: str,
    log: Callable[[str], None] | None = None,
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
            blocks = pages[i] if i < len(pages) else []
            trans = per_page[i]
            m = min(len(blocks), len(trans))
            if m == 0:
                out_doc.insert_pdf(src, from_page=i, to_page=i)
                continue

            out_doc.insert_pdf(src, from_page=i, to_page=i)
            page = out_doc[-1]

            # Ruled tables are re-laid-out: an English translation is usually
            # longer than the Chinese it replaces, so a table row that no longer
            # fits its cell is pushed down (and the grid redrawn) instead of
            # overlapping the row beneath it.  This is scoped to ``find_tables``
            # — a vector chart (org chart etc.) is never mistaken for a table.
            # A *scanned* table (no text layer, no vector rules, found via OCR)
            # has its grid reconstructed from the OCR block boxes, but the row
            # expansion is NOT run for it: the reconstructed "row height" is
            # only as tall as the scan text itself, yet measuring the translated
            # cell against it grew the rows so far that e.g. 292,712,933,925.17
            # was pushed ~355 pt down — several table rows below its own cell.
            # Scan geometry is kept as-is and the cells' single-line fit (which
            # shrinks the font instead) keeps the rows from overlapping.
            tables = _extract_tables(page)
            ocr_table = False
            if not tables:
                tables = _reconstruct_ocr_tables(blocks)
                ocr_table = bool(tables)
            mapping = _map_blocks_to_table_cells(blocks, tables) if tables else {}
            if ocr_table:
                shifts, new_bottoms, grid, bboxes = {}, {}, [], []
            else:
                shifts, new_bottoms, grid, bboxes = _compute_table_layout(
                    tables, mapping, blocks, trans, font
                )

            # Remove the original text (keep images and other line art/graphics).
            # OCR blocks sit on a raster image rather than a text layer, so
            # nothing is redacted for them — they are covered below instead.
            # ``is_chart`` node labels are left entirely untouched: the source
            # diagram's label is the "translation", so neither the original nor
            # a redraw is needed (redrawing a vertical label would mangle it).
            for j in range(m):
                b = blocks[j]
                if not b.ocr and not b.is_chart:
                    page.add_redact_annot(fitz.Rect(b.x0, b.y0, b.x1, b.y1))
            # A detected table's grid lines must go too, or they stay at the old
            # row positions while the (now taller) translations are drawn lower.
            if tables:
                for bb in bboxes:
                    page.add_redact_annot(bb)
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=(
                    fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED
                    if tables else fitz.PDF_REDACT_LINE_ART_NONE
                ),
            )

            # Draw the translation at the original positions / alignment /
            # font size (see ``_draw_translated_block`` for the fitting rules).
            for j in range(m):
                b = blocks[j]
                if b.is_chart:
                    # A diagram node label keeps its source: the original is
                    # already on the page (nothing was redacted/covered above),
                    # so there is nothing to draw.
                    continue
                draw_b = b
                if j in shifts and shifts[j]:
                    dy = shifts[j]
                    if j in new_bottoms:
                        # A table cell moved down: give it the expanded row box
                        # so the fit loop uses the extra height rather than
                        # crushing the font.
                        draw_b = replace(b, y0=b.y0 + dy, y1=new_bottoms[j])
                    else:
                        # Prose below a grown table shifts down with it.
                        draw_b = replace(b, y0=b.y0 + dy, y1=b.y1 + dy)
                # A table cell whose translation wraps to >1 line is anchored at the
                # cell's OWN top border rather than the source glyph box top: the
                # glyph box starts a few points below the row's top (cell padding)
                # and a single-line source box is short, so a centred/wraps stretch
                # would push the last line past the row's bottom rule.  Anchoring at
                # the cell's full vertical span keeps the wrapped block inside.  This
                # covers both ``find_tables`` (text-layer) and reconstructed OCR
                # grids (``mapping`` is non-empty for either).
                if j in mapping and getattr(b, "in_table", False):
                    if len(_fit_block(b, font, trans[j])[0]) > 1:
                        t_i, r_i = mapping[j]
                        row = tables[t_i]["rows"][r_i]
                        dy = shifts.get(j, 0.0)
                        draw_b = replace(
                            draw_b,
                            y0=min(c.y0 for c in row) + dy,
                            y1=max(c.y1 for c in row) + dy,
                        )
                if b.ocr:
                    # Cover the underlying scan pixels so the translation does
                    # not overprint the original (raster) text.  Use the (possibly
                    # expanded / shifted) draw box, not the original bbox, or the
                    # translation drawn lower would sit on uncovered scan text.
                    page.draw_rect(
                        fitz.Rect(
                            draw_b.x0 - 0.5, draw_b.y0 - 0.5,
                            draw_b.x1 + 0.5, draw_b.y1 + 0.5,
                        ),
                        color=None,
                        fill=(1, 1, 1),
                    )
                _draw_translated_block(page, font, draw_b, trans[j])

            # Redraw the table grid over the expanded rows.
            if tables:
                for spec in grid:
                    if spec[0] == "h":
                        _, x0, x1, y = spec
                        page.draw_line(
                            fitz.Point(x0, y), fitz.Point(x1, y), color=(0, 0, 0), width=0.6
                        )
                    else:
                        _, x, y0, y1 = spec
                        page.draw_line(
                            fitz.Point(x, y0), fitz.Point(x, y1), color=(0, 0, 0), width=0.6
                        )

        out_doc.set_metadata({"title": "Translated text", "creator": "PDFtranslate"})
        out_doc.save(str(out_path), garbage=4, deflate=True)
    finally:
        out_doc.close()
        src.close()
