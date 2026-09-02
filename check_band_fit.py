"""Verify the new band-fit semantics on the real scanned statements.

Pairs each grid cell of pages 24-27 with the translation from the previous
English export (drawn at the same positions) and reports the fitter's font
buckets and any band violation (wrapped height > fit_height — would cross the
grid line of the row below).  Pure analysis; writes nothing.
"""

import sys
import fitz

sys.path.insert(0, ".")
from translate_app import pdfio

SRC = "annual report - Mintai Commercial Bank 2025.pdf"
OLD_EN = "annual report - Mintai Commercial Bank 2025_English.pdf"
PAGES = [23, 24, 25, 26]  # 1-based 24-27 (statement pages)

font = fitz.Font("cjk")


def old_english_spans(pno):
    """All drawn text spans (bbox + text) from the old export's page.

    ``get_text("blocks")`` merges a whole row into ONE block/line, so pairing
    by block would give every cell of the row the concatenation of label +
    (n) + line + figures.  Spans keep per-draw runs; a cell's translation is
    the span(s) whose bbox fits inside that cell's x-range.
    """
    doc = fitz.open(OLD_EN)
    spans = []
    for blk in doc[pno].get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for l in blk.get("lines", []):
            for s in l.get("spans", []):
                if s["text"].strip():
                    spans.append((fitz.Rect(*s["bbox"]), s["text"]))
    doc.close()
    return spans


def pair_translation(block, spans):
    """Assemble the old-export text drawn at ``block``: the spans whose bbox
    sits inside the cell's x-range (and near its y), joined in order."""
    out = []
    for bb, text in sorted(spans, key=lambda s: (round(s[0].y0, 1), s[0].x0)):
        # A drawn span lies inside the cell's column (glyph box or its
        # right-aligned figure edge); tolerance covers the centre x drift.
        inside = bb.x0 >= block.x0 - 3.0 and bb.x1 <= block.x1 + 3.0
        neary = abs((bb.y0 + bb.y1) / 2 - (block.y0 + block.y1) / 2) < 6.0
        if inside and neary:
            out.append(text.strip())
    if out:
        return " ".join(out)
    return None


def main():
    dt = pdfio.extract_document_text(SRC, ocr=True, log=lambda s: None)
    totals = {"<3": 0, "3-4": 0, "4-5": 0, "5-6": 0, ">=6": 0, "multi": 0}
    viols = []
    for pno in PAGES:
        print(f"--- page {pno + 1} ---")
        page_blocks = dt.pages[pno]
        spans = old_english_spans(pno)
        buckets = {"<3": 0, "3-4": 0, "4-5": 0, "5-6": 0, ">=6": 0}
        multi = 0
        for b in page_blocks:
            if not getattr(b, "in_table", False):
                continue
            trans = pair_translation(b, spans)
            if trans is None:
                continue
            lines, fs = pdfio._fit_block(b, font, trans)
            h = pdfio._wrapped_height(font, lines, fs)
            if fs < 3:
                buckets["<3"] += 1
            elif fs < 4:
                buckets["3-4"] += 1
            elif fs < 5:
                buckets["4-5"] += 1
            elif fs < 6:
                buckets["5-6"] += 1
            else:
                buckets[">=6"] += 1
            if len(lines) > 1:
                multi += 1
            if b.fit_height > 0 and h > b.fit_height + 0.01:
                viols.append((pno + 1, b.text[:20], trans[:60], round(fs, 2),
                              round(h, 1), round(b.fit_height, 1)))
        print(" ", buckets, f"multi-line={multi}")
        for k in buckets:
            totals[k] += buckets[k]
        totals["multi"] += multi
    print("=== totals ===")
    print(totals)
    if viols:
        print("=== BAND VIOLATIONS ===")
        for v in viols:
            print(v)
    else:
        print("no band violations — every wrap stops before the line below")


if __name__ == "__main__":
    main()
