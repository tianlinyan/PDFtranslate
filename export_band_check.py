"""Regenerate pages 24-27 with the band-fit semantics for visual inspection.

Uses the full-file extraction (OCR cache hit) plus the translations paired
from the previous English export, and writes them back onto a 4-page copy of
the source.  Output: %TEMP%/pdfcmp/bandfit_p24_27.pdf + p24/p27 crops.
"""

import os
import sys
import fitz

sys.path.insert(0, ".")
from translate_app import pdfio

SRC = "annual report - Mintai Commercial Bank 2025.pdf"
OLD_EN = "annual report - Mintai Commercial Bank 2025_English.pdf"
PAGES = [23, 24, 25, 26]
OUT_DIR = os.path.join(os.environ.get("TEMP", r"C:\Users\tly00\AppData\Local\Temp"), "pdfcmp")
OUT_PDF = os.path.join(OUT_DIR, "bandfit_p24_27.pdf")

font = fitz.Font("cjk")


def old_english_spans(pno):
    """All drawn text spans (bbox + text) — ``get_text("blocks")`` merges a
    whole row into one block, which would pair every cell of the row with the
    concatenation of the row's texts.  Spans keep per-draw runs; a cell's
    translation is the span(s) fitting inside that cell's x-range."""
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
    out = []
    for bb, text in sorted(spans, key=lambda s: (round(s[0].y0, 1), s[0].x0)):
        inside = bb.x0 >= block.x0 - 3.0 and bb.x1 <= block.x1 + 3.0
        neary = abs((bb.y0 + bb.y1) / 2 - (block.y0 + block.y1) / 2) < 6.0
        if inside and neary:
            out.append(text.strip())
    if out:
        return " ".join(out)
    return None


def main():
    dt = pdfio.extract_document_text(SRC, ocr=True, log=lambda s: None)

    src = fitz.open(SRC)
    copy = fitz.open()
    for i in PAGES:
        copy.insert_pdf(src, from_page=i, to_page=i)
    copy.save(OUT_PDF)
    copy.close()
    src.close()

    pages = [dt.pages[i] for i in PAGES]
    per_page = []
    for i in PAGES:
        spans = old_english_spans(i)
        per_page.append(
            [pair_translation(b, spans) or "" for b in dt.pages[i]]
        )

    pdfio.save_translated_pdf(OUT_PDF, pages, per_page, os.path.join(OUT_DIR, "bandfit_out.pdf"), "English")

    # Crops: the signature band of p24 and the whole statement area of p27.
    src2 = fitz.open(os.path.join(OUT_DIR, "bandfit_out.pdf"))
    for pno, name, rect in [
        (0, "bandfit_p24_sig.png", fitz.Rect(40, 570, 560, 730)),
        (3, "bandfit_p27_all.png", fitz.Rect(40, 380, 560, 640)),
    ]:
        page = src2[pno]
        pix = page.get_pixmap(clip=rect, dpi=110)
        pix.save(os.path.join(OUT_DIR, name))
    src2.close()
    print("wrote", OUT_PDF, "and crops to", OUT_DIR)


if __name__ == "__main__":
    main()
