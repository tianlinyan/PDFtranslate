"""Generate a diverse 5-page English PDF for exercising the translation pipeline.

Page 1  rich text layer, two columns, heading / italic / bullet / callout box
Page 2  text layer + raster chart image + ruled table (text + image + table mix)
Page 3  OCR text page  (rasterised letter, NO text layer → OCR path)
Page 4  OCR table page (rasterised financial statement, NO text layer → OCR + grid)
Page 5  OCR mixed page (rasterised text + table + diagram, NO text layer)

Run:  python make_diverse_test_pdf.py [out.pdf]
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw

W, H = 595.0, 842.0            # A4 in points
M = 50.0                       # page margin
BODY = 10.5
H1 = 20.0
H2 = 13.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _text(page, x, y, s, size=BODY, font="helv", color=(0, 0, 0)):
    page.insert_text((x, y), s, fontsize=size, fontname=font, color=color)


def _para(page, rect, s, size=BODY, font="helv", align=0, color=(0, 0, 0)):
    page.insert_textbox(fitz.Rect(*rect), s, fontsize=size, fontname=font,
                        align=align, color=color)


def _rule(page, y, x0=M, x1=W - M, width=0.8, color=(0.2, 0.2, 0.2)):
    page.draw_line(fitz.Point(x0, y), fitz.Point(x1, y), color=color, width=width)


def _table(page, x0, y0, col_w, row_h, rows, header=True, size=9.0):
    """Draw a ruled table; ``rows`` is a list of lists of cell strings."""
    ncols = len(col_w)
    total_w = sum(col_w)
    nrows = len(rows)
    # grid
    for r in range(nrows + 1):
        y = y0 + r * row_h
        page.draw_line(fitz.Point(x0, y), fitz.Point(x0 + total_w, y),
                       color=(0.25, 0.25, 0.25), width=0.6)
    xs = [x0]
    for w in col_w:
        xs.append(xs[-1] + w)
    for x in xs:
        page.draw_line(fitz.Point(x, y0), fitz.Point(x, y0 + nrows * row_h),
                       color=(0.25, 0.25, 0.25), width=0.6)
    # cells
    for r, row in enumerate(rows):
        cy = y0 + r * row_h + row_h * 0.68
        for c, cell in enumerate(row):
            font = "hebo" if (header and r == 0) else "helv"
            # numeric-looking cells right-align inside their column
            num = any(ch.isdigit() for ch in cell) and not any(ch.isalpha() for ch in cell)
            if num:
                tw = fitz.Font(font).text_length(cell, fontsize=size)
                _text(page, xs[c + 1] - 5 - tw, cy, cell, size, font)
            else:
                _text(page, xs[c] + 5, cy, cell, size, font)


def _chart_png() -> bytes:
    """A clean raster bar chart (a genuine embedded image, not vector art)."""
    img = Image.new("RGB", (900, 420), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 20, 860, 380], outline=(200, 200, 200))
    base, top = 360, 40
    values = [0.55, 0.72, 0.63, 0.88, 0.79, 0.94]
    labels = ["North", "South", "East", "West", "Central", "Online"]
    colors = [(52, 122, 183), (68, 170, 153), (240, 173, 78),
              (196, 78, 82), (120, 96, 178), (90, 150, 110)]
    n = len(values)
    slot = (820 - 80) / n
    for i, (v, lab, col) in enumerate(zip(values, labels, colors)):
        bx = 80 + i * slot + slot * 0.2
        bw = slot * 0.6
        bh = (base - top) * v
        d.rectangle([bx, base - bh, bx + bw, base], fill=col)
        d.text((bx + bw / 2 - 18, base + 8), lab, fill=(60, 60, 60))
        d.text((bx + bw / 2 - 16, base - bh - 18), f"{int(v*100)}%", fill=(40, 40, 40))
    d.line([80, base, 840, base], fill=(90, 90, 90), width=2)
    d.line([80, top, 80, base], fill=(90, 90, 90), width=2)
    d.text((40, 4), "Revenue contribution by region (%)", fill=(30, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _raster(w, h, draw_fn, zoom=3.0) -> bytes:
    """Render ``draw_fn`` on a throwaway page and return a PNG (no text layer)."""
    tmp = fitz.open()
    try:
        p = tmp.new_page(width=w, height=h)
        draw_fn(p)
        pix = p.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")
    finally:
        tmp.close()


# --------------------------------------------------------------------------- #
# page 1 — rich text, two columns
# --------------------------------------------------------------------------- #
def draw_page1(page):
    _text(page, M, 62, "NORTHWIND ANALYTICS", 9, "hebo", (0.35, 0.35, 0.35))
    _text(page, W - M - 92, 62, "Q3 2025 Review", 9, "heit", (0.35, 0.35, 0.35))
    _rule(page, 70)
    _text(page, M, 104, "Quarterly Business Review", H1, "hebo")
    _text(page, M, 126, "Prepared for the Executive Committee  •  September 2025",
          10, "heit", (0.4, 0.4, 0.4))

    left = (M, 150, 290, 520)
    right = (305, 150, W - M, 520)
    _para(page, left,
          "Revenue grew 14.2% year over year, driven by a resilient "
          "enterprise segment and improving retention in the mid-market. "
          "Operating margin expanded by 180 basis points as procurement "
          "savings and automation offset wage inflation.\n\n"
          "Customer acquisition cost declined for the third consecutive "
          "quarter. Net revenue retention held at 112%, and the pipeline "
          "for the fourth quarter is the strongest in three years.\n\n"
          "Management remains cautious about currency headwinds and longer "
          "enterprise sales cycles, and has updated the full-year outlook "
          "accordingly.")
    _para(page, right,
          "Highlights for the quarter include the launch of two new product "
          "lines, the completion of the data-platform migration, and the "
          "opening of a regional delivery hub.\n\n"
          "The board approved an expanded share-repurchase programme and a "
          "modest increase to the research and development budget.\n\n"
          "Key priorities for the next quarter:")
    # bullet list
    bullets = [
        "Scale the partner channel in the APAC region",
        "Reduce cloud unit cost by a further 8%",
        "Complete SOC 2 Type II certification",
        "Hire 40 engineers for the platform team",
    ]
    y = 372
    for b in bullets:
        page.draw_circle(fitz.Point(312, y - 3), 1.6, color=None, fill=(0.2, 0.4, 0.7))
        _para(page, (322, y - 11, W - M, y + 12), b, 10)
        y += 20

    # callout box
    page.draw_rect(fitz.Rect(M, 545, W - M, 625), color=(0.6, 0.7, 0.85),
                   fill=(0.94, 0.96, 0.99), width=0.8)
    _text(page, M + 12, 566, "KEY TAKEAWAY", 9, "hebo", (0.2, 0.35, 0.6))
    _para(page, (M + 12, 574, W - M - 12, 618),
          "The quarter confirms that the strategy is working: growth is "
          "broad-based, margins are improving, and the balance sheet remains "
          "strong enough to fund both organic investment and shareholder returns.",
          10)

    _rule(page, 790, color=(0.7, 0.7, 0.7))
    _text(page, M, 806, "Confidential — internal use only", 8, "helv", (0.5, 0.5, 0.5))
    _text(page, W - M - 46, 806, "Page 1 of 5", 8, "helv", (0.5, 0.5, 0.5))


# --------------------------------------------------------------------------- #
# page 2 — text + image + table (all in the text layer)
# --------------------------------------------------------------------------- #
def draw_page2(page):
    _text(page, M, 70, "Regional Performance Overview", H1, "hebo")
    _rule(page, 80)
    _para(page, (M, 96, W - M, 150),
          "The chart below shows each region's contribution to revenue in the "
          "quarter. Online and West led growth, while Central and South "
          "remained broadly stable. The table that follows breaks the figures "
          "down by quarter and compares them with the prior year.")
    page.insert_image(fitz.Rect(M, 160, W - M, 395), stream=_chart_png())

    rows = [
        ["Region", "Q1", "Q2", "Q3", "YoY"],
        ["North", "1,204,500", "1,318,900", "1,402,300", "+12.4%"],
        ["South", "980,120", "1,010,440", "1,038,220", "+5.9%"],
        ["East", "1,455,800", "1,512,330", "1,598,470", "+9.8%"],
        ["West", "1,102,640", "1,240,510", "1,366,900", "+18.1%"],
        ["Online", "2,010,330", "2,288,760", "2,510,140", "+24.9%"],
    ]
    _table(page, M, 415, [110, 95, 95, 95, 100], 22, rows, header=True, size=9.5)
    _para(page, (M, 580, W - M, 620),
          "Table 2.1 — Revenue by region (USD). Year-on-year figures compare "
          "the third quarter of 2025 with the same period in 2024.", 9,
          "heit", color=(0.4, 0.4, 0.4))

    _rule(page, 790, color=(0.7, 0.7, 0.7))
    _text(page, M, 806, "Confidential — internal use only", 8, "helv", (0.5, 0.5, 0.5))
    _text(page, W - M - 46, 806, "Page 2 of 5", 8, "helv", (0.5, 0.5, 0.5))


# --------------------------------------------------------------------------- #
# page 3 — OCR text page (raster only)
# --------------------------------------------------------------------------- #
def draw_page3(page):
    page.draw_rect(fitz.Rect(0, 0, W, H), color=None, fill=(0.985, 0.985, 0.985))
    _text(page, M, 70, "NORTHWIND ANALYTICS", 12, "hebo", (0.15, 0.25, 0.45))
    _rule(page, 80, color=(0.3, 0.4, 0.6), width=1.2)
    _text(page, M, 112, "INTERNAL MEMORANDUM", H2, "hebo")

    fields = [("To:", "Executive Committee"),
              ("From:", "Office of the Chief Financial Officer"),
              ("Date:", "18 September 2025"),
              ("Re:", "Third-quarter results and revised outlook")]
    y = 142
    for k, v in fields:
        _text(page, M, y, k, 10, "hebo", (0.25, 0.25, 0.25))
        _text(page, M + 52, y, v, 10)
        y += 18

    _rule(page, y + 6, color=(0.75, 0.75, 0.75))
    _para(page, (M, y + 26, W - M, y + 150),
          "The third quarter closed ahead of plan. Group revenue reached "
          "USD 7.92 million, an increase of 14.2 percent over the prior-year "
          "period, while operating expenses grew by only 6.4 percent. As a "
          "result, operating profit improved to USD 1.36 million.\n\n"
          "Cash generation remained healthy. Operating cash flow was USD 1.11 "
          "million and the group closed the quarter with USD 642 thousand in "
          "cash and cash equivalents and no drawn borrowings.\n\n"
          "Looking ahead, we expect fourth-quarter revenue in the range of "
          "USD 8.3 to 8.6 million. This assumes no material change in exchange "
          "rates and a normal seasonal pattern in enterprise renewals.")
    _text(page, M, 640, "Prepared by: Finance Operations", 10, "heit", (0.35, 0.35, 0.35))
    page.draw_line(fitz.Point(M, 690), fitz.Point(M + 200, 690),
                   color=(0.4, 0.4, 0.4), width=0.7)
    _text(page, M, 704, "Authorised signature", 9, "helv", (0.5, 0.5, 0.5))


# --------------------------------------------------------------------------- #
# page 4 — OCR table page (raster only)
# --------------------------------------------------------------------------- #
def draw_page4(page):
    page.draw_rect(fitz.Rect(0, 0, W, H), color=None, fill=(0.99, 0.99, 0.99))
    _text(page, M, 80, "Consolidated Statement of Financial Position", 15, "hebo")
    _text(page, M, 100, "(All amounts in thousands of USD)", 9, "heit", (0.4, 0.4, 0.4))
    _rule(page, 110, color=(0.4, 0.4, 0.4))

    rows = [
        ["Item", "2025", "2024"],
        ["Total assets", "4,821,530", "4,102,988"],
        ["Current assets", "1,905,220", "1,733,410"],
        ["Cash and cash equivalents", "642,118", "588,204"],
        ["Trade receivables", "701,455", "650,332"],
        ["Inventories", "388,640", "361,220"],
        ["Non-current assets", "2,916,310", "2,369,578"],
        ["Total liabilities", "2,244,905", "1,988,120"],
        ["Total equity", "2,576,625", "2,114,868"],
    ]
    _table(page, M, 130, [280, 105, 105], 26, rows, header=True, size=10.5)

    _text(page, M, 430, "Notes", 12, "hebo")
    _para(page, (M, 444, W - M, 560),
          "1. The financial information above has been prepared on a going-concern "
          "basis and is presented in accordance with the group's accounting "
          "policies.\n"
          "2. Cash and cash equivalents include short-term deposits with an "
          "original maturity of three months or less.\n"
          "3. Trade receivables are stated net of an allowance for expected "
          "credit losses.\n"
          "4. The comparative figures for 2024 have been restated to reflect a "
          "reclassification between current and non-current assets.")
    _text(page, M, 720, "Page 4 of 5", 9, "helv", (0.5, 0.5, 0.5))


# --------------------------------------------------------------------------- #
# page 5 — OCR mixed page: text + table + diagram (raster only)
# --------------------------------------------------------------------------- #
def draw_page5(page):
    page.draw_rect(fitz.Rect(0, 0, W, H), color=None, fill=(0.985, 0.985, 0.985))
    _text(page, M, 70, "Operating Review", H1, "hebo")
    _rule(page, 82, color=(0.4, 0.4, 0.4))
    _para(page, (M, 100, W - M, 180),
          "The operating review covers the three reporting segments. Revenue "
          "and growth are summarised in the table below, followed by the "
          "reporting structure of the segment management team.")

    seg = [
        ["Segment", "Revenue", "Growth"],
        ["Enterprise", "3,420,000", "+16.0%"],
        ["Mid-market", "2,610,500", "+11.2%"],
        ["Small business", "1,893,700", "+8.5%"],
    ]
    _table(page, M, 195, [180, 160, 110], 24, seg, header=True, size=10)

    _text(page, M, 330, "Segment reporting structure", 12, "hebo")
    # simple diagram: boxes + connectors
    def box(cx, cy, w, h, label):
        r = fitz.Rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        page.draw_rect(r, color=(0.25, 0.35, 0.55), fill=(0.9, 0.94, 0.99), width=1.0)
        _para(page, (r.x0, r.y0 + 4, r.x1, r.y1), label, 9, align=1)
        return r

    top = box(W / 2, 372, 190, 40, "Group CFO")
    l = box(150, 480, 150, 40, "Enterprise")
    c = box(W / 2, 480, 150, 40, "Mid-market")
    r = box(W - 150, 480, 150, 40, "Small business")
    for b in (l, c, r):
        page.draw_line(fitz.Point(top.x0 + top.width / 2, top.y1),
                       fitz.Point(b.x0 + b.width / 2, b.y0),
                       color=(0.4, 0.5, 0.7), width=1.0)
    page.draw_line(fitz.Point(top.x0 + top.width / 2, top.y1),
                   fitz.Point(top.x0 + top.width / 2, 452), color=(0.4, 0.5, 0.7), width=1.0)
    page.draw_line(fitz.Point(l.x0 + l.width / 2, 452),
                   fitz.Point(r.x0 + r.width / 2, 452), color=(0.4, 0.5, 0.7), width=1.0)

    _para(page, (M, 560, W - M, 660),
          "Each segment leader reports directly to the group chief financial "
          "officer and is accountable for revenue, gross margin and customer "
          "retention. Segment results are reviewed monthly and consolidated "
          "quarterly for external reporting.")
    _text(page, M, 720, "Page 5 of 5", 9, "helv", (0.5, 0.5, 0.5))


# --------------------------------------------------------------------------- #
def build(out_path: str | Path) -> Path:
    out = fitz.open()
    # 1) text layer, rich layout
    p1 = out.new_page(width=W, height=H)
    draw_page1(p1)
    # 2) text layer + embedded image + table
    p2 = out.new_page(width=W, height=H)
    draw_page2(p2)
    # 3) OCR text (raster only)
    p3 = out.new_page(width=W, height=H)
    p3.insert_image(fitz.Rect(0, 0, W, H), stream=_raster(W, H, draw_page3))
    # 4) OCR table (raster only)
    p4 = out.new_page(width=W, height=H)
    p4.insert_image(fitz.Rect(0, 0, W, H), stream=_raster(W, H, draw_page4))
    # 5) OCR mixed text + table + diagram (raster only)
    p5 = out.new_page(width=W, height=H)
    p5.insert_image(fitz.Rect(0, 0, W, H), stream=_raster(W, H, draw_page5))

    out_path = Path(out_path)
    # ``deflate``/``garbage``/``clean`` matter a lot here: ``insert_image`` stores
    # the raster pages as *uncompressed* RGB streams, so a naive save is ~40 MB;
    # this brings the same file down to a few hundred KB (lossless, OCR-safe).
    out.save(str(out_path), garbage=4, deflate=True, clean=True)
    out.close()
    return out_path


def _verify(path: Path) -> None:
    doc = fitz.open(str(path))
    try:
        print(f"file: {path}  ({path.stat().st_size/1024:.0f} KB, {doc.page_count} pages)")
        for i, page in enumerate(doc):
            txt = page.get_text("text").strip()
            imgs = len(page.get_images(full=True))
            kind = "text-layer" if len(txt) > 40 else "raster/no-text (OCR)"
            print(f"  p{i+1}: {len(txt):5d} chars, {imgs} image(s)  -> {kind}")
    finally:
        doc.close()


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_diverse_en_5p.pdf")
    build(out)
    _verify(out)
