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
3. **压线**：字形带落在**源页印刷表格线**上的比例 > 5%。表格线来自自适应掩膜
   （``pdfio._page_rule_mask``），所以扫描件/文本层件一视同仁；采样前把字形带
   映射到渲染帧，旋转页同样有效。
4. **字号过小**：低于 ``pdfio._MIN_READABLE``（正文可读下限）。表格单元允许到
   ``pdfio._MIN_TABLE_READABLE``——扫描密集报表里那 3–5pt 的格子属于物理极限，
   这里只报出来供人工判断，不算致命。
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
    missing: list[str] = field(default_factory=list)
    pages_issue: list[str] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)
    span_count: int = 0
    size_median: float = 0.0
    interleaved: bool = False

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


def find_rule_crossings(spans: Sequence[Span], rules, scale: float,
                        share: float = _RULE_SHARE,
                        inset: float = _RULE_BAND_INSET,
                        ink=None) -> list[str]:
    """Spans whose glyphs have a printed rule running through them.

    ``ink`` is the source page's glyph-core mask (same frame as ``rules``).  A rule
    that already ran through the source's own text (dotted statement lines do) is
    *not* reported: the translation is faithful there.  Only a rule that crosses a
    translation where the source had no text is a defect.
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
        if ink is not None and float(ink[py0:py1, px0:px1].mean()) >= _SOURCE_INK_SHARE:
            continue                   # the source text was crossed there too
        out.append(f"第 {s.page + 1} 页：{got:.0%} 压在表格线上 —— "
                   f"「{_clip(s.text)}」 {_fmt(s.rect)}")
    return out


def find_too_small(spans: Sequence[Span], in_table_floor: float,
                   body_floor: float) -> list[str]:
    """Spans below the readable floor (table cells may go down to their floor)."""
    out: list[str] = []
    for s in spans:
        if s.size < in_table_floor - 0.01:
            out.append(f"第 {s.page + 1} 页：{s.size:.1f}pt（低于表格下限 "
                       f"{in_table_floor}pt）—— 「{_clip(s.text)}」")
        elif s.size < body_floor - 0.01:
            out.append(f"第 {s.page + 1} 页：{s.size:.1f}pt（低于正文下限 "
                       f"{body_floor}pt）—— 「{_clip(s.text)}」")
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
        if tgt.page_count < src.page_count:
            report.pages_issue.append(
                f"译文 {tgt.page_count} 页少于原文 {src.page_count} 页")
        text_layers = _text_layer_blocks(src)
        sizes: list[float] = []
        visited: set[int] = set()
        for i in range(src.page_count):
            if pages and (i + 1) not in pages:
                continue
            ti = 2 * i + 1 if report.interleaved else i
            if ti >= tgt.page_count:
                continue
            visited.add(ti)
            page = tgt[ti]
            src_page = src[i]
            spans = collect_spans(page, i)
            report.span_count += len(spans)
            sizes.extend(s.size for s in spans)
            report.overlap += find_overlaps(spans)
            report.off_page += find_off_page(spans, frame_rect(page))
            report.small += find_too_small(
                spans, pdfio._MIN_TABLE_READABLE, pdfio._MIN_READABLE)
            src_frame = frame_rect(src_page)
            if (abs(src_frame.width - frame_rect(page).width) > 1.0
                    or abs(src_frame.height - frame_rect(page).height) > 1.0):
                report.pages_issue.append(
                    f"第 {i + 1} 页尺寸不一致：源 {_fmt(src_frame)} "
                    f"译文 {_fmt(frame_rect(page))}")
            try:
                luma = pdfio._pixmap_luma(src_page.get_pixmap(dpi=dpi))
            except Exception:              # noqa: BLE001 — sampling is best-effort
                luma = None
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
                rendered = [Span(s.text, s.size, to_render_frame(s.rect, src_page), s.page)
                            for s in spans]
                report.rule += find_rule_crossings(rendered, rules, scale, ink=ink)
            # The missing check needs no pixels (it compares text spans), so it
            # runs even when the page could not be sampled — an unsampled page
            # used to skip it silently.
            if text_layers.get(i):
                report.missing += find_missing(text_layers[i], spans)
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
                report.small += find_too_small(
                    spans, pdfio._MIN_TABLE_READABLE, pdfio._MIN_READABLE)
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


def _text_layer_blocks(src: fitz.Document) -> dict[int, list[object]]:
    """Per-page source blocks that have a text layer (empty dict when none)."""
    if not any(src[i].get_text("text").strip() for i in range(src.page_count)):
        return {}
    try:
        doc = pdfio.extract_document_text(Path(src.name), ocr=False)
    except Exception:                  # noqa: BLE001 — best-effort check
        return {}
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

    if pages:
        # An out-of-range --page used to select nothing and still print
        # "体检通过" with exit 0 — a silent false green.
        try:
            with fitz.open(files[0]) as probe:
                total = probe.page_count
        except Exception as exc:           # noqa: BLE001 — bad/encrypted input
            print(f"无法打开原文：{exc}", file=sys.stderr)
            return 2
        bad = sorted(p for p in pages if p < 1 or p > total)
        if bad:
            print(f"页码越界：{bad}（原文共 {total} 页）", file=sys.stderr)
            return 2

    report = check_document(Path(files[0]), Path(files[1]), pages=pages, dpi=dpi)

    print(f"原文：{files[0]}")
    print(f"译文：{files[1]}")
    print(f"页数：{report.pages}，文字块 {report.span_count} 段")
    if report.interleaved:
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
