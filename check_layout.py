"""版面体检：导出后核对**译文排版**（重叠 / 出页 / 压线 / 过小 / 漏画）。

用法::

    python check_layout.py 原文.pdf 译文.pdf [--page 3,5-7] [--dpi 150]

与 ``check_translation.py`` 互补：那个查**内容**（数字/残留中文/编号），这个查
**排版**。两者都只看导出产物，不需要模型在线。

检查项
------
1. **译文重叠**：两个译文字形带（按字体上升/下降部算，不是 bbox）相交面积 ≥
   较小者的 25%。长译文压住相邻块是原位导出的典型故障（行高只有被识别成表格
   时才会自动扩展）。
2. **超出页面**：字形带越过页面左/右/下边界（页外字形会被裁掉，等于丢内容）。
   判界用**未旋转帧**（`get_text` 的坐标就是那一帧），旋转页不会再把页内的
   文字误判成出页。
3. **压线**：字形带落在**源页印刷表格线**上的比例 > 5%，**且该线要延伸到字形带
   之外**。表格线来自自适应掩膜（``pdfio._page_rule_mask``），所以扫描件/文本层件
   一视同仁；采样前把字形带映射到渲染帧，旋转页同样有效。延伸判据是必需的：12pt
   汉字的横笔就有 ≈11pt 长，正好够上掩膜的「游程」判据——真机把『二、公司组织架构图』
   的第二行译文判成 12% 压线，而那段「线」是 90.2–101.3pt、整段落在 90.0–105.5pt
   的译文里（「司」的横笔），源文墨迹豁免也救不了它（那条「线」本身就是豁免要测的
   墨）。印刷表格线横跨单元格，必然越过它穿过的文字。
4. **字号过小**：低于**该块自己的**可读下限——与导出器 / ``flow._check_layout`` 同口径：
   表格单元 ``pdfio._MIN_TABLE_FLOOR``(3pt)，其它 ``min(pdfio._font_start(块),
   pdfio._MIN_READABLE)``（源文本身就是 5pt 的脚注，按 5pt 画出来是合法的）。
   **源页没有文本层时**（扫描密集报表）无从知道原文用了多大字号——那些 3–5pt 的格内
   文字是 OCR 行高与表格格的物理极限，导出器本来就这么画——该页只报 < 3pt 的文字，
   其余偏小文字**汇总成一行提示**（此前拿 6.3pt 正文下限去量 5 页扫描件，刷出 331 条
   假「字号」，把唯一一条真问题埋了）。
5. **漏画**（仅源页有文本层时）：源页有文字的块，译文里没有对应位置的文字。
   判据是「有译文字形带落在这个块里（或落在它被表格下推后的位置附近）」——
   按**整块面积**要求 10% 会把长段落永远误报（短译文只盖住一行）。
   每条译文字形带只归**一个**源块（见 `_credit_key`）：否则「同列且在半块高以内」
   这条宽松判据会让*相邻*块的译文替漏译的块作答，整段漏译也能「体检通过」。
   扫描页需要 OCR 才能判断，这里跳过并提示。
6. **页数 / 页尺寸**：译文页数不应少于原文；逐页尺寸应与原文一致。
   双语（原文页 + 译文页交错）产物会自动按「源页 i ↔ 译文页 2i+1」配对。

退出码：0 = 没有结构性问题（重叠/出页/压线/漏画）；1 = 有；2 = 用法错误
（含 `--page` 越界 / 格式非法）。
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pymupdf as fitz

from translate_app import pdfio

#: Share of the smaller glyph band that must be overlapped to report a collision.
_OVERLAP_SHARE = 0.25
#: Share of a glyph band's *inner* part that must sit on printed rules to report a
#: crossing.  The band is shrunk by ``_RULE_BAND_INSET`` at top and bottom first:
#: a rule grazing the ascenders/descenders of a dense row is normal, a rule
#: running through the middle of the glyphs is not.
_RULE_SHARE = 0.05
_RULE_BAND_INSET = 0.20
#: If the *source* page already had printed text ink in that same inner band, the
#: rule runs through the source's own text too (dotted statement lines do) — the
#: translation being crossed is faithful, not a defect.
_SOURCE_INK_SHARE = 0.04
#: How far (pt) a printed rule must reach **outside** the glyph run to count as crossing
#: it.  A printed table rule spans its cell, so it continues past the text it crosses;
#: a CJK glyph stroke stops inside the run — and a 12 pt glyph's horizontal stroke
#: (≈11 pt) already satisfies ``pdfio._RULE_MIN_RUN_PT`` (10 pt), which is how a real
#: 「压线」 came out of 「二、公司组织架构图」's wrapped second line (see
#: ``_rule_reaches_beyond``).
_RULE_EXTEND_PT = 3.0
#: Share of a *span's own* glyph band that must fall inside the source block's box
#: for the span to count as that block's translation.  (The old rule asked for
#: ``_COVER_SHARE`` of the whole *block* area from a single span: a correctly
#: translated 12-line paragraph covers one line ≈ 8% of the box, so every long
#: paragraph was reported as "漏画".)
_COVER_SHARE = 0.30
#: A block pushed down by a grown table above it is still that block: a span may
#: count when it sits in the same column within this slack of the box.
_MISSING_SLACK_PT = 12.0
#: DPI used to sample the source page for its printed-rule mask.
_SAMPLE_DPI = 150
#: How many findings of one kind the CLI prints before summarising the rest.
_MAX_PRINT = 20


@dataclass
class Span:
    """One drawn translation run with its glyph band (not the loose bbox)."""

    text: str
    size: float
    rect: fitz.Rect          # glyph band: ascender .. descender
    page: int = 0


@dataclass
class LayoutReport:
    """Collected findings, grouped so the CLI can print and grade them."""

    pages: int = 0
    overlap: list[str] = field(default_factory=list)
    off_page: list[str] = field(default_factory=list)
    rule: list[str] = field(default_factory=list)
    small: list[str] = field(default_factory=list)
    #: ``(1-based page, count, smallest size)`` for pages whose source has **no text
    #: layer**: what size the source used there is unknowable without OCR, so their
    #: small spans are summarised (one note) instead of individually graded.  A note
    #: never changes the verdict — a limitation stated honestly is not a defect.
    small_no_source: list[tuple[int, int, float]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    pages_issue: list[str] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)
    #: Informational lines (what was checked / skipped and why).  They must not change
    #: the verdict: a limitation stated honestly is not a defect.
    notes: list[str] = field(default_factory=list)
    span_count: int = 0
    size_median: float = 0.0
    interleaved: bool = False
    #: The source→output page map read from the target (empty = none recorded).
    page_map: list[int] = field(default_factory=list)

    def structural(self) -> list[str]:
        """Findings that mean content is wrong (exit 1), not just cosmetic."""
        return [*self.overlap, *self.off_page, *self.rule, *self.missing,
                *self.pages_issue]

    def all_clear(self) -> bool:
        return not self.structural() and not self.small


def glyph_band(span: dict) -> fitz.Rect:
    """Glyph band of a text span: origin ± the font's ascender/descender.

    ``bbox`` includes the font's full line box, so two neighbouring lines whose
    *boxes* touch are not necessarily colliding text; the band is what is visible.
    """
    size = float(span.get("size", 0.0))
    x0, _y0, x1, _y1 = span["bbox"]
    ox, oy = span["origin"]
    asc = float(span.get("ascender", 0.75) or 0.75)
    desc = float(span.get("descender", -0.25) or -0.25)
    return fitz.Rect(x0, oy - size * max(0.0, asc), x1, oy - size * min(0.0, desc))


def collect_spans(page: fitz.Page, page_index: int = 0) -> list[Span]:
    """All non-empty text spans of a page, with their glyph bands."""
    out: list[Span] = []
    for block in page.get_text("dict").get("blocks", []):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = str(span.get("text", "")).strip()
                if text:
                    out.append(Span(text, float(span["size"]),
                                    glyph_band(span), page_index))
    return out


def find_overlaps(spans: Sequence[Span],
                  share: float = _OVERLAP_SHARE) -> list[str]:
    """Pairs of spans whose glyph bands collide by at least ``share``."""
    out: list[str] = []
    for i, a in enumerate(spans):
        for b in spans[i + 1:]:
            inter = a.rect & b.rect
            if inter.is_empty:
                continue
            small = min(a.rect.get_area(), b.rect.get_area())
            if small <= 0:
                continue
            got = inter.get_area() / small
            if got >= share:
                out.append(
                    f"第 {a.page + 1} 页：{got:.0%} 重叠 —— "
                    f"「{_clip(a.text)}」× 「{_clip(b.text)}」 {_fmt(inter)}"
                )
    return out


def find_off_page(spans: Sequence[Span], page_rect: fitz.Rect,
                  margin: float = 1.0) -> list[str]:
    """Spans whose glyph band leaves the page (glyphs would be clipped)."""
    out: list[str] = []
    for s in spans:
        if (s.rect.x0 < page_rect.x0 - margin
                or s.rect.x1 > page_rect.x1 + margin
                or s.rect.y1 > page_rect.y1 + margin):
            out.append(f"第 {s.page + 1} 页：「{_clip(s.text)}」 "
                       f"{_fmt(s.rect)} 越出页面 {_fmt(page_rect)}")
    return out


def _rule_reaches_beyond(rules, py0: int, py1: int, px0: int, px1: int,
                         extend_px: int) -> bool:
    """True when a rule in the band's *row* reaches **outside** the glyph run.

    A printed table rule spans its cell, so it continues left of / right of the text it
    crosses; a CJK glyph stroke stops inside the run.  Measured false positive:
    「二、公司组织架构图」's translation wrapped to two lines and the second line
    ("Chart", 90.0–105.5 pt) sat on a run of 90.2–101.3 pt — 司's horizontal stroke,
    ≈11 pt long, enough for ``pdfio._RULE_MIN_RUN_PT`` (10 pt).  The source-ink excuse
    cannot help there: the stroke *is* the ink that excuse measures.

    Only the band's own rows are searched (a rule elsewhere cannot condemn this span),
    and only ``extend_px`` beyond either edge — the same stroke that crosses the text is
    the one that continues past it.
    """
    height, width = rules.shape
    y0, y1 = max(0, py0), min(height, py1)
    if y1 <= y0 or width <= 0 or extend_px <= 0:
        return False
    lo = max(0, px0 - extend_px)
    hi = min(width, px1 + extend_px + 1)
    x0, x1 = px0 - lo, px1 - lo
    for r in range(y0, y1):
        row = rules[r, lo:hi]
        if not row[x0:x1].any():
            continue                   # this row does not cross the run at all
        left = row[max(0, x0 - extend_px):x0]
        right = row[x1 + 1:min(row.size, x1 + 1 + extend_px)]
        if (left.size and left.any()) or (right.size and right.any()):
            return True
    return False


def find_rule_crossings(spans: Sequence[Span], rules, scale: float,
                        share: float = _RULE_SHARE,
                        inset: float = _RULE_BAND_INSET,
                        ink=None) -> list[str]:
    """Spans whose glyphs have a printed rule running through them.

    ``ink`` is the source page's glyph-core mask (same frame as ``rules``).  A rule
    that already ran through the source's own text (dotted statement lines do) is
    *not* reported: the translation is faithful there.  Only a rule that crosses a
    translation where the source had no text is a defect.

    The rule must also reach beyond the glyph run (:func:`_rule_reaches_beyond`):
    a CJK stroke long enough to look like a rule sits *inside* the text, while a
    printed table line continues past it.
    """
    if rules is None:
        return []
    out: list[str] = []
    height, width = rules.shape
    for s in spans:
        pad = s.rect.height * inset
        y0 = s.rect.y0 + pad
        y1 = s.rect.y1 - pad
        if y1 - y0 < 1.0:              # too short to judge
            continue
        px0 = max(0, int(s.rect.x0 * scale))
        px1 = min(width, int(s.rect.x1 * scale) + 1)
        py0 = max(0, int(y0 * scale))
        py1 = min(height, int(y1 * scale) + 1)
        if px1 <= px0 or py1 <= py0:
            continue
        got = float(rules[py0:py1, px0:px1].mean())
        if got <= share:
            continue
        if not _rule_reaches_beyond(rules, py0, py1, px0, px1,
                                   max(1, int(round(_RULE_EXTEND_PT * scale)))):
            continue               # a glyph stroke, not a printed rule
        if ink is not None and float(ink[py0:py1, px0:px1].mean()) >= _SOURCE_INK_SHARE:
            continue                   # the source text was crossed there too
        out.append(f"第 {s.page + 1} 页：{got:.0%} 压在表格线上 —— "
                   f"「{_clip(s.text)}」 {_fmt(s.rect)}")
    return out


def _body_floor_for(blocks) -> float:
    """The smallest body font size the *source* page legitimately uses.

    A 5 pt footnote drawn at 5 pt is not a defect: the exporter scales its start size
    from ``pdfio._font_start`` and ``flow._check_layout`` judges each block against
    ``min(_font_start(block), _MIN_READABLE)``.  Comparing every span with the absolute
    6.3 pt body floor reported legal small text as too small (the agent's own audit
    reported 0 issues for the same document).
    """
    floors = [min(pdfio._font_start(b), pdfio._MIN_READABLE)
              for b in blocks or [] if not getattr(b, "in_table", False)]
    return min(floors) if floors else pdfio._MIN_READABLE


def _block_floor(block) -> float:
    """The readability floor the **exporter** uses for one source block.

    Identical to ``flow._check_layout``: a table cell may legitimately reach
    ``_MIN_TABLE_FLOOR`` (3 pt — a dense scanned statement's cells are at the physical
    limit of the medium), anything else ``min(_font_start(block), _MIN_READABLE)``, so a
    5 pt footnote drawn at 5 pt is faithful rather than too small.
    """
    if getattr(block, "in_table", False):
        return pdfio._MIN_TABLE_FLOOR
    return min(pdfio._font_start(block), pdfio._MIN_READABLE)


def find_too_small(spans: Sequence[Span], blocks=None) -> list[str]:
    """Spans below the readable floor **of the block they belong to**.

    ``blocks`` are the page's *source* blocks (its text layer).  Grading every span with
    the page's prose floor reported 331 legitimately small spans on a real 5-page scan:
    a table cell drawn at 5.2 pt is what the exporter deliberately does
    (``_MIN_TABLE_FLOOR``), and ``flow._check_layout`` grades each block with its *own*
    floor.  So a span is graded against the block it belongs to (see :func:`_covers`).

    A span that belongs to **no** source block is not graded against a prose floor at
    all — the CLI cannot know what the source used there (an OCR'd statement's rows are
    drawn at ~5 pt by design, and on a partly-OCR'd page the fragment in the text layer
    is unrelated to the dense block of cells).  Only the universal limit applies:
    ``blocks is None`` (no text layer) or an unattributable span is reported below
    ``_MIN_TABLE_FLOOR``, which nothing legitimate reaches — not even an OCR-synthesised
    block (never under 5 pt, i.e. ≥4.5 pt once scaled).  The unattributable mass is the
    caller's summary note (``LayoutReport.small_no_source``), and a source block whose
    translation really went astray is reported by :func:`find_missing` instead.
    """
def attribute_spans(spans: Sequence[Span], blocks) -> list[tuple[Span, object | None]]:
    """Pair every span with the source block it belongs to (``None`` = unattributable).

    One attribution rule for the module: the same :func:`_covers` / :func:`_credit_key`
    arbitration :func:`find_missing` uses, so a span cannot answer for a neighbour's
    block here either.  ``blocks`` may be ``None`` (the page has no text layer) — then
    nothing is attributable.
    """
    boxes: list[tuple[object, fitz.Rect]] = []
    for block in blocks or []:
        if not str(getattr(block, "text", "")).strip():
            continue
        box = fitz.Rect(block.x0, block.y0, block.x1, block.y1)
        if not box.is_empty:
            boxes.append((block, box))
    pairs: list[tuple[Span, object | None]] = []
    for s in spans:
        cands = [i for i, (_b, box) in enumerate(boxes) if _covers(box, s)]
        pairs.append((s, boxes[min(cands, key=lambda i:
                                  _credit_key(boxes[i][1], s, i))][0] if cands else None))
    return pairs


def find_too_small(spans: Sequence[Span], blocks=None) -> list[str]:
    """Spans below the readable floor **of the block they belong to**.

    ``blocks`` are the page's *source* blocks (its text layer).  Grading every span with
    the page's prose floor reported 331 legitimately small spans on a real 5-page scan:
    a table cell drawn at 5.2 pt is what the exporter deliberately does
    (``_MIN_TABLE_FLOOR``), and ``flow._check_layout`` grades each block with its *own*
    floor.  So a span is graded against the block it belongs to (see
    :func:`attribute_spans`).

    A span belonging to **no** source block is not graded against a prose floor at all —
    the CLI cannot know what size the source used there (an OCR'd statement's rows are
    drawn at ~5 pt by design, and on a partly-OCR'd page the fragment in the text layer
    is unrelated to the dense block of cells).  Only the universal limit applies:
    ``_MIN_TABLE_FLOOR``, which nothing legitimate reaches — not even an OCR-synthesised
    block (never under 5 pt, i.e. ≥4.5 pt once scaled).  The unattributable mass is the
    caller's summary note (``LayoutReport.small_no_source``); a source block whose
    translation really went astray is reported by :func:`find_missing` instead.
    """
    return grade_attributed(attribute_spans(spans, blocks))


def grade_attributed(pairs) -> list[str]:
    """The ``find_too_small`` findings for already-attributed ``(span, block)`` pairs.

    Split out because the caller needs the attribution itself (to summarise the
    spans that could not be attributed) — attributing twice per output page would
    double the ``_covers`` work on every scanned page.
    """
    out: list[str] = []
    for s, block in pairs:
        if block is None:
            floor, why = pdfio._MIN_TABLE_FLOOR, "无对应源块"
        else:
            floor = _block_floor(block)
            why = ("表格单元" if getattr(block, "in_table", False)
                   else f"源文字号 {float(getattr(block, 'size', 0.0)):.1f}pt")
        if s.size < floor - 0.01:
            out.append(f"第 {s.page + 1} 页：{s.size:.1f}pt（低于可读下限 "
                       f"{floor:.1f}pt，{why}）—— 「{_clip(s.text)}」")
    return out


def _covers(box: fitz.Rect, span: Span) -> bool:
    """True when ``span`` *could* be the drawn translation of a block at ``box``.

    Two ways to qualify: the span's glyph band lies mostly *inside* the box (the
    normal in-place draw), or it sits in the same column within half a box height
    of it — a table above grew and pushed the prose down, which is a faithful
    export, not a missing block.

    "Could" is the operative word: the slack branch is deliberately loose (the
    *next* block's translation falls inside it too), so :func:`find_missing` still
    has to decide which single block a span answers for.
    """
    band = span.rect
    if band.get_area() <= 0:
        return False
    inter = band & box
    if not inter.is_empty and inter.get_area() >= _COVER_SHARE * band.get_area():
        return True
    slack = box.height / 2.0 + _MISSING_SLACK_PT
    near = fitz.Rect(box.x0 - 2.0, box.y0 - slack, box.x1 + 2.0, box.y1 + slack)
    inter_near = band & near
    return (inter_near.width >= 0.5 * band.width
            and inter_near.height >= 0.5 * band.height)


def _band_distance(box: fitz.Rect, band: fitz.Rect) -> float:
    """Vertical gap between a span band and a block box (0.0 when they overlap)."""
    if band.y1 < box.y0:
        return box.y0 - band.y1
    if band.y0 > box.y1:
        return band.y0 - box.y1
    return 0.0


def _credit_key(box: fitz.Rect, span: Span, index: int) -> tuple:
    """Sort key deciding which block gets credited for ``span``.

    The block whose box *contains* the span's centre wins; otherwise the nearest
    box does, ties going to the earlier block.  Ordering matters: a miss must be
    reported (fail-closed) rather than excused by a neighbour's translation.
    """
    band = span.rect
    cx = (band.x0 + band.x1) / 2.0
    cy = (band.y0 + band.y1) / 2.0
    inside = box.x0 <= cx <= box.x1 and box.y0 <= cy <= box.y1
    return (0 if inside else 1, _band_distance(box, band), index)


def find_missing(source_blocks: Sequence[object],
                 spans: Sequence[Span]) -> list[str]:
    """Source blocks with text but no drawn translation anywhere near them.

    Only meaningful for pages that have a text layer (scanned pages need OCR to
    know their blocks); the caller passes those pages only.

    Every span is credited to **one** block — the best candidate by
    :func:`_credit_key` — and only credited blocks count as translated.  Without
    that arbitration the loose "same column, within half a block height" slack let
    a paragraph's neighbour answer for it: a 12-line paragraph whose translation
    was never drawn passed because the next block's translation sat inside the
    slack (measured: whole page reported clean, exit 0).
    """
    out: list[str] = []
    boxes: list[tuple[object, fitz.Rect]] = []
    for block in source_blocks:
        text = str(getattr(block, "text", "")).strip()
        if not text:
            continue
        box = fitz.Rect(block.x0, block.y0, block.x1, block.y1)
        if box.is_empty:
            continue
        boxes.append((block, box))
    credited: set[int] = set()
    for s in spans:
        cands = [i for i, (_b, box) in enumerate(boxes) if _covers(box, s)]
        if cands:
            credited.add(min(cands, key=lambda i: _credit_key(boxes[i][1], s, i)))
    for i, (block, box) in enumerate(boxes):
        if i in credited:
            continue
        out.append(f"第 {int(getattr(block, 'page', 0)) + 1} 页：源块 "
                   f"{_fmt(box)} 附近没有译文 —— "
                   f"「{_clip(str(getattr(block, 'text', '')).strip())}」")
    return out


def frame_rect(page: fitz.Page) -> fitz.Rect:
    """The **unrotated** frame ``get_text`` coordinates live in (cropbox, origin 0).

    ``page.rect`` is the *displayed* (rotated) page, so on a ``/Rotate 90`` sheet
    it swaps width and height and every in-page span looked off-page.
    """
    cb = page.cropbox
    return fitz.Rect(0.0, 0.0, float(cb.width), float(cb.height))


def to_render_frame(rect: fitz.Rect, page: fitz.Page) -> fitz.Rect:
    """Map a :func:`frame_rect` rect into the *rendered* (rotated) frame.

    The printed-rule mask is sampled from ``get_pixmap``, which renders the page as
    displayed; span coordinates are unrotated.  Without this mapping the rule
    check sampled the wrong place on every rotated page and reported 0 crossings
    (a silent false green).  The math is cropbox-based, matching
    ``pdfio._rot_map_rect`` (both frames have their origin at the cropbox's
    top-left — measured: ``get_text`` bbox == the rendered ink position).
    """
    rot = int(getattr(page, "rotation", 0) or 0) % 360
    if rot == 0:
        return fitz.Rect(rect)
    frame = frame_rect(page)
    w, h = frame.width, frame.height
    if rot == 90:                                  # (x', y') = (H - y, x)
        return fitz.Rect(h - rect.y1, rect.x0, h - rect.y0, rect.x1)
    if rot == 180:                                 # (x', y') = (W - x, H - y)
        return fitz.Rect(w - rect.x1, h - rect.y1, w - rect.x0, h - rect.y0)
    return fitz.Rect(rect.y0, w - rect.x1, rect.y1, w - rect.x0)   # 270: (y, W - x)


def check_document(source: Path, target: Path,
                   pages: set[int] | None = None,
                   dpi: int = _SAMPLE_DPI) -> LayoutReport:
    """Run every layout check over ``source`` / ``target``.

    ``pages`` are **1-based source pages**; a bilingual target (original page
    followed by its translation page) is paired automatically — comparing
    ``tgt[i]`` with ``src[i]`` there reported the *original* page of the next
    source page as "漏画" and never looked at the real translation page.

    Target pages that pair with no source page (a longer target, when the product
    is not a clean 2× bilingual file) are still checked for what needs no source:
    overlap, off-page spans and unreadable sizes — and counted in ``pages_issue``.
    """
    report = LayoutReport()
    src = fitz.open(str(source))
    tgt = fitz.open(str(target))
    try:
        report.pages = tgt.page_count
        report.interleaved = src.page_count > 0 and tgt.page_count == 2 * src.page_count
        # An expanded product ("译文扩页") lets one source page occupy several output
        # pages; the exporter records the source→output map in the PDF's XMP
        # (``pdfio.document_page_map``).  Without it, ``tgt[i]`` pairs a source page
        # with the *continuation* of an earlier one, so every page after the first
        # expansion is compared with the wrong translation.
        page_map = pdfio.document_page_map(tgt)
        if page_map is None and not report.interleaved \
                and tgt.page_count > src.page_count:
            report.pages_issue.append(
                f"译文 {tgt.page_count} 页多于原文 {src.page_count} 页，"
                "且文件内没有扩页映射（XMP pageMap）——按页序配对可能错位，"
                "「漏画」等结论仅供参考；请用「译文扩页」导出的原始产物重跑。")
        ranges = (pdfio.mapped_page_ranges(page_map, src.page_count, tgt.page_count)
                  if page_map is not None else None)
        if ranges is not None:
            report.page_map = list(page_map)
            report.notes.append(
                f"检测到扩页产物：按文件内记录的页映射配对（源页 → 首个输出页 "
                f"{','.join(str(t + 1) for t in page_map)}）。")
        if tgt.page_count < src.page_count:
            report.pages_issue.append(
                f"译文 {tgt.page_count} 页少于原文 {src.page_count} 页")
        text_layers = _text_layer_blocks(src)
        text_layer_known = text_layers is not None
        if not text_layer_known:
            # Fail-closed: a failed extraction is not "no text layer".  Report it
            # as a page issue (exit 1) instead of silently skipping every missing
            # check and printing 体检通过.
            report.pages_issue.append(
                "无法提取源文档文本层（pdfio.extract_document_text 失败），"
                "本次未做漏画检查，结论不可视为通过。")
            text_layers = {}
        sizes: list[float] = []
        visited: set[int] = set()
        for i in range(src.page_count):
            if pages and (i + 1) not in pages:
                continue
            if ranges is not None:
                tis = [t for t in ranges[i] if t < tgt.page_count]
            else:
                ti = 2 * i + 1 if report.interleaved else i
                tis = [ti] if ti < tgt.page_count else []
            if not tis:
                continue
            src_page = src[i]
            src_frame = frame_rect(src_page)
            try:
                luma = pdfio._pixmap_luma(src_page.get_pixmap(dpi=dpi))
            except Exception:              # noqa: BLE001 — sampling is best-effort
                luma = None
            # Every output page belonging to this source page is measured on its own
            # (collisions / off-page / sizes are page-local), while the source
            # comparison below sees the *union*: with expansion part of the page
            # legitimately lives on a continuation page, so comparing the source with
            # one output page would report the moved blocks as missing.
            page_spans: list = []
            # ``None`` = this page has no text layer (a scan): its blocks are OCR'd
            # at export time, so the CLI has nothing to grade the spans against.
            src_blocks = text_layers.get(i) or None
            body_floor = _body_floor_for(src_blocks)
            unattributed: list = []
            for ti in tis:
                visited.add(ti)
                page = tgt[ti]
                spans = collect_spans(page, i)
                page_spans.extend(spans)
                report.span_count += len(spans)
                sizes.extend(s.size for s in spans)
                report.overlap += find_overlaps(spans)
                report.off_page += find_off_page(spans, frame_rect(page))
                pairs = attribute_spans(spans, src_blocks)
                report.small += grade_attributed(pairs)
                unattributed.extend(s for s, b in pairs if b is None)
                if (abs(src_frame.width - frame_rect(page).width) > 1.0
                        or abs(src_frame.height - frame_rect(page).height) > 1.0):
                    report.pages_issue.append(
                        f"第 {i + 1} 页尺寸不一致：源 {_fmt(src_frame)} "
                        f"译文 {_fmt(frame_rect(page))}")
                if luma is not None:
                    scale = luma.shape[1] / max(1e-6, float(src_page.rect.width))
                    levels = pdfio._page_levels(luma, scale)
                    rules = pdfio._page_rule_mask(luma, levels, scale)
                    ink = None
                    if levels is not None and rules is not None:
                        bg, _span, offsets = levels
                        # Source *text* ink only: the rule pixels themselves are dark
                        # too and must not excuse a crossing.
                        ink = (luma < bg - offsets[1]) & ~rules
                    rendered = [Span(s.text, s.size,
                                     to_render_frame(s.rect, src_page), s.page)
                                for s in spans]
                    report.rule += find_rule_crossings(rendered, rules, scale,
                                                       ink=ink)
            if unattributed:
                # Spans the CLI cannot tie to a source block: say *what* was seen
                # instead of grading it.  Grading these against the 6.3 pt prose floor
                # produced 331 findings on a real 5-page scan (a dense statement's rows
                # are drawn at ~5 pt by design), which buried the one real problem.
                below = [s for s in unattributed if s.size < body_floor - 0.01]
                if below:
                    report.small_no_source.append(
                        (i + 1, len(below), min(s.size for s in below)))
            # The missing check needs no pixels (it compares text spans), so it
            # runs even when the page could not be sampled — an unsampled page
            # used to skip it silently.
            if len(tis) > 1:
                # An expanded source page: blocks that did not fit were moved to
                # continuation pages, so their *coordinates* no longer match the source
                # geometry the missing check compares against.  Reporting them as
                # "漏画" would be a false positive on every expanded page; say the
                # check was skipped instead.
                report.notes.append(
                    f"第 {i + 1} 页为扩页页（输出第 {tis[0] + 1}–{tis[-1] + 1} 页）："
                    "搬到续页的块坐标已改变，该页跳过「漏画」检查。")
            elif not text_layer_known:
                pass               # already reported as a pages_issue above
            elif text_layers.get(i):
                report.missing += find_missing(text_layers[i], page_spans)
            else:
                report.scanned_pages.append(i + 1)
        if not report.interleaved and not pages:
            # Target pages with no source counterpart: the loop above runs over
            # *source* pages, so a longer target used to be skipped entirely (only
            # the "fewer pages" direction was reported) — whatever was on those
            # pages never got looked at, yet the run still printed "体检通过".
            # They have no source page to compare or to sample rules from, but
            # collisions, off-page spans and unreadable sizes need no source.
            extra = sorted(set(range(tgt.page_count)) - visited)
            for t in extra:
                page = tgt[t]
                spans = collect_spans(page, t)
                report.span_count += len(spans)
                sizes.extend(s.size for s in spans)
                report.overlap += find_overlaps(spans)
                report.off_page += find_off_page(spans, frame_rect(page))
                report.small += find_too_small(spans, None)
                below = [s for s in spans if s.size < pdfio._MIN_READABLE - 0.01]
                if below:
                    report.small_no_source.append(
                        (t + 1, len(below), min(s.size for s in below)))
            if extra:
                report.pages_issue.append(
                    f"译文多出 {len(extra)} 页没有对应原文（第 "
                    f"{'、'.join(str(t + 1) for t in extra)} 页），"
                    "已按无源页检查重叠/出页/字号")
        if sizes:
            report.size_median = statistics.median(sizes)
    finally:
        src.close()
        tgt.close()
    return report


def _text_layer_blocks(src: fitz.Document) -> dict[int, list[object]] | None:
    """Per-page source blocks that have a text layer.

    ``{}``  = the document has no text layer at all (a pure scan: the missing
    check does not apply).
    ``None`` = the extraction **failed**, which must not be read as "no text
    layer": that turned every page into a "scan", skipped the missing check
    and still printed 体检通过 (a silent false green).
    """
    if not any(src[i].get_text("text").strip() for i in range(src.page_count)):
        return {}
    try:
        doc = pdfio.extract_document_text(Path(src.name), ocr=False)
    except Exception:                  # noqa: BLE001 — the caller reports it
        return None
    return {i: list(blocks) for i, blocks in enumerate(doc.pages)}


def _fmt(rect: fitz.Rect) -> str:
    return (f"({rect.x0:.1f},{rect.y0:.1f})-({rect.x1:.1f},{rect.y1:.1f})")


def _clip(text: str, width: int = 28) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _parse_pages(text: str) -> set[int] | None:
    """Parse ``3,5-7``; ``None`` when the spec is malformed (caller exits 2)."""
    pages: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo, hi = part.split("-", 1)
                lo_i, hi_i = int(lo), int(hi)
                if hi_i < lo_i:
                    return None
                pages.update(range(lo_i, hi_i + 1))
            else:
                pages.add(int(part))
        except ValueError:
            return None
    return pages or None


def main(argv: Sequence[str] | None = None) -> int:
    # A redirected stdout on Windows uses the locale codec; the report is Chinese
    # and contains ellipses/dashes, so pin UTF-8 (best-effort) rather than dying
    # with a UnicodeEncodeError halfway through a run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:              # noqa: BLE001 — older/exotic streams
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    pages: set[int] | None = None
    dpi = _SAMPLE_DPI
    files: list[str] = []
    it = iter(argv)
    for arg in it:
        if arg == "--page":
            pages = _parse_pages(next(it, ""))
            if pages is None:
                print("--page 需要一个形如 3,5-7 的页码列表", file=sys.stderr)
                return 2
        elif arg == "--dpi":
            try:
                dpi = max(72, min(300, int(next(it, ""))))
            except ValueError:
                print("--dpi 需要一个整数", file=sys.stderr)
                return 2
        elif arg in ("-h", "--help"):
            print(__doc__)
            return 0
        elif arg.startswith("-"):
            print(f"未知参数：{arg}", file=sys.stderr)
            return 2
        else:
            files.append(arg)
    if len(files) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    # Open both inputs up front: a missing / encrypted file is a *usage* error
    # (exit 2), never an uncaught traceback — which used to exit 1, the same code
    # as "there are structural problems".  The probe lived inside ``if pages:``,
    # so it only ran for --page invocations.
    try:
        with fitz.open(files[0]) as probe:
            total = probe.page_count
        with fitz.open(files[1]):
            pass
    except Exception as exc:               # noqa: BLE001 — missing/encrypted input
        print(f"无法打开文件：{exc}", file=sys.stderr)
        return 2

    if pages:
        # An out-of-range --page used to select nothing and still print
        # "体检通过" with exit 0 — a silent false green.
        bad = sorted(p for p in pages if p < 1 or p > total)
        if bad:
            print(f"页码越界：{bad}（原文共 {total} 页）", file=sys.stderr)
            return 2

    report = check_document(Path(files[0]), Path(files[1]), pages=pages, dpi=dpi)

    print(f"原文：{files[0]}")
    print(f"译文：{files[1]}")
    print(f"页数：{report.pages}，文字块 {report.span_count} 段")
    if report.interleaved and not report.page_map:
        print("（双语交错产物：按「源页 i ↔ 译文页 2i+1」配对检查）")
    print()

    for msg in report.overlap:
        print(f"[重叠] {msg}")
    for msg in report.off_page:
        print(f"[出页] {msg}")
    for msg in report.rule:
        print(f"[压线] {msg}")
    for msg in report.missing:
        print(f"[漏画] {msg}")
    for msg in report.pages_issue:
        print(f"[页数] {msg}")
    for msg in report.small[: _MAX_PRINT]:
        print(f"[字号] {msg}")
    if len(report.small) > _MAX_PRINT:
        print(f"[字号] …另有 {len(report.small) - _MAX_PRINT} 段偏小文字未逐条列出"
              f"（共 {len(report.small)} 段）。")
    if report.scanned_pages:
        print(f"[提示] 第 {report.scanned_pages} 页源页无文本层，"
              f"跳过「漏画」检查（需 OCR）。")
    if report.small_no_source:
        pages_txt = "、".join(str(p) for p, _c, _s in report.small_no_source)
        total = sum(c for _p, c, _s in report.small_no_source)
        smallest = min(s for _p, _c, s in report.small_no_source)
        print(f"[提示] 第 {pages_txt} 页：另有 {total} 段译文无法对应到原文块"
              f"（该页无文本层，或为扫描/图表页），其字号由 OCR 行高与表格格决定，"
              f"未与原文比对（最小 {smallest:.1f}pt）。")
    for msg in report.notes:
        print(f"[提示] {msg}")

    print()
    if report.structural():
        print("结论：存在结构性排版问题（重叠/出页/压线/漏画），请修正后重新导出。")
        return 1
    if report.small:
        print("结论：无结构性排版问题；存在字号偏小的文字，请人工确认可读性。")
        return 0
    print("体检通过：未发现排版问题。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
