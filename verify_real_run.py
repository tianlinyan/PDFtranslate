"""Ground-truth verification: translate statement pages 24-27 with the user's
configured local model, export, and measure the fit result (font buckets,
band violations) against REAL translations.
"""

import json
import os
import sys
from pathlib import Path

import fitz

sys.path.insert(0, ".")
from translate_app import pdfio
from translate_app.settings import ModelConfig
from translate_app.translator import TranslationEngine

SRC = "annual report - Mintai Commercial Bank 2025.pdf"
PAGES = [23, 24, 25, 26]  # 1-based 24-27
OUT_DIR = os.path.join(os.environ.get("TEMP", r"C:\Users\tly00\AppData\Local\Temp"), "pdfcmp")

font = fitz.Font("cjk")


def main():
    raw = json.load(open("models.json", encoding="utf-8"))["models"]
    model = ModelConfig.from_dict(next(m for m in raw if m["id"] == "qwen3.8"))
    print("model:", model.name, model.endpoint)

    dt = pdfio.extract_document_text(SRC, ocr=True, log=lambda s: None)
    flat4, page_lens = [], []
    for i in PAGES:
        page_lens.append(len(dt.pages[i]))
        flat4.extend(dt.pages[i])
    print("blocks to translate:", len(flat4))

    saved = os.path.join(OUT_DIR, "real_translations.json")
    if os.path.exists(saved):
        print("reusing saved translations")
        blob = json.load(open(saved, encoding="utf-8"))
        page_lens = blob["page_lens"]
        translated = blob["translated"]
    else:
        engine = TranslationEngine(model)
        result = engine.translate_blocks(
            [b.text for b in flat4],
            "English",
            log=lambda m: print("  [log]", m),
            doc_path=Path(SRC),
            resume=False,
        )
        print("result errors:", result.errors)
        for i in (0, 1, 2, 100, 400, 800):
            if i < len(result.translated):
                print("  sample[%d]: %r -> %r" % (i, flat4[i].text if hasattr(flat4[i], "text") else flat4[i], result.translated[i]))
        translated = result.translated
        with open(saved, "w", encoding="utf-8") as fh:
            json.dump({"page_lens": page_lens, "translated": translated}, fh, ensure_ascii=False)

    per_page = []
    idx = 0
    for n in page_lens:
        per_page.append(translated[idx:idx + n])
        idx += n

    src = fitz.open(SRC)
    copy = fitz.open()
    for i in PAGES:
        copy.insert_pdf(src, from_page=i, to_page=i)
    copy.save(os.path.join(OUT_DIR, "real4.pdf"))
    copy.close()
    src.close()

    out = os.path.join(OUT_DIR, "real_out.pdf")
    pdfio.save_translated_pdf(os.path.join(OUT_DIR, "real4.pdf"), [dt.pages[i] for i in PAGES], per_page, out, "English")

    # ---- fit analysis on REAL translations ----
    totals = worst = 0
    multi_total = 0
    violations = []          # multi-line wraps crossing the band — real crossings
    single_over = 0          # 1-line cells whose *metric* extent > band (benign:
                             # digits/letters have no descent; ink stays in the box)
    tiny = []
    for k, pno in enumerate(PAGES):
        buckets = {"<4": 0, "4-5": 0, "5-6": 0, ">=6": 0}
        multi = 0
        n_in_table = 0
        for b, t in zip(dt.pages[pno], per_page[k]):
            if not getattr(b, "in_table", False):
                continue
            n_in_table += 1
            lines, fs = pdfio._fit_block(b, font, t)
            h = pdfio._wrapped_height(font, lines, fs)
            if fs < 4:
                buckets["<4"] += 1
                worst += 1
                tiny.append((pno + 1, round(fs, 2), len(lines),
                             round(b.x0, 1), round(b.x1, 1), round(b.y0, 1),
                             round(b.fit_width, 1) if b.fit_width else 0,
                             round(b.fit_height, 1), b.text[:16], t[:48]))
            elif fs < 5:
                buckets["4-5"] += 1
            elif fs < 6:
                buckets["5-6"] += 1
            else:
                buckets[">=6"] += 1
            if len(lines) > 1:
                multi += 1
                if b.fit_height > 0 and h > b.fit_height + 0.05:
                    violations.append((pno + 1, b.text[:20], t[:40], round(fs, 2),
                                       round(h, 1), round(b.fit_height, 1)))
            elif b.fit_height > 0 and h > b.fit_height + 0.05:
                single_over += 1
        totals += n_in_table
        multi_total += multi
        print(f"page {pno + 1}: {buckets} in_table={n_in_table} multi={multi}")
    print("totals:", totals, "multi:", multi_total,
          "multi-line band violations:", len(violations), "single-line over (benign):", single_over)
    for v in violations:
        print("  CROSSING", v)
    print("== tiny (<4pt) cells: page fs nlines x0 x1 y0 fitw fith src -> trans")
    for w in tiny:
        print(" ", w)

    d = fitz.open(out)
    crops = [
        (0, "real_p24_sig.png", fitz.Rect(40, 570, 560, 730)),
        (2, "real_p26_oci.png", fitz.Rect(40, 460, 560, 660)),
        (3, "real_p27_cf.png", fitz.Rect(40, 140, 560, 280)),
    ]
    for pno, name, rect in crops:
        d[pno].get_pixmap(clip=rect, dpi=110).save(os.path.join(OUT_DIR, name))
    d.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
