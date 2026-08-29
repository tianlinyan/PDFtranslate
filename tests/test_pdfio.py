"""Tests for PDF text extraction and the export writers."""
import os
import shutil
import unittest
from pathlib import Path

import pymupdf as fitz

from translate_app import pdfio

from tests._helpers import build_sample_pdf, build_two_column_pdf

_OUT = Path(__file__).resolve().parent / "_out"


def _text_lines(page):
    """Return ``(bbox, text)`` for every rendered text line on ``page``."""
    out = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if text:
                out.append((fitz.Rect(line["bbox"]), text))
    return out


def setUpModule():  # noqa: N802
    _OUT.mkdir(exist_ok=True)


def tearDownModule():  # noqa: N802
    shutil.rmtree(_OUT, ignore_errors=True)


class PdfioTest(unittest.TestCase):
    def test_extract_and_exports(self):
        src = _OUT / "sample.pdf"
        build_sample_pdf(src, pages=2)

        doc = pdfio.extract_document_text(src)
        self.assertEqual(doc.page_count, 2)
        self.assertTrue(doc.blocks)
        self.assertEqual(len(doc.blocks), len(doc.block_pages))

        per_page = pdfio.group_by_page(doc.block_pages, doc.blocks, doc.page_count)
        self.assertEqual(len(per_page), 2)

        # Plain text + markdown
        txt = _OUT / "out.txt"
        pdfio.save_plain_text(per_page, txt)
        self.assertTrue(txt.exists() and txt.stat().st_size > 0)

        md = _OUT / "out.md"
        pdfio.save_markdown(per_page, doc.blocks, doc.block_pages, md, "Chinese", "T")
        self.assertTrue(md.exists() and md.stat().st_size > 0)
        self.assertIn("# T", md.read_text("utf-8"))

    def test_bilingual_pdf(self):
        src = _OUT / "sample_b.pdf"
        build_sample_pdf(src, pages=2)
        doc = pdfio.extract_document_text(src)
        per_page = pdfio.group_by_page(doc.block_pages, doc.blocks, doc.page_count)
        out = _OUT / "bilingual.pdf"
        pdfio.save_interleaved_pdf(src, per_page, out, "Chinese")
        d = fitz.open(str(out))
        self.assertGreaterEqual(d.page_count, 4)  # 2 original + 2 translation pages
        d.close()

    def test_translated_pdf_removes_original_text(self):
        src = _OUT / "sample_t.pdf"
        build_sample_pdf(src, pages=2)
        doc = pdfio.extract_document_text(src)
        per_page = pdfio.group_by_page(doc.block_pages, doc.blocks, doc.page_count)
        fake = [
            ["TRANSLATED-" + str(idx) for idx, _t in enumerate(pg)]
            for pg in per_page
        ]
        out = _OUT / "translated.pdf"
        pdfio.save_translated_pdf(src, doc.pages, fake, out, "Chinese")
        d = fitz.open(str(out))
        # Original English body must be gone; placeholder must be present.
        all_text = "\n".join(d[p].get_text() for p in range(d.page_count))
        self.assertNotIn("sample body paragraph", all_text)
        self.assertIn("TRANSLATED-", all_text)
        d.close()

    def test_translated_pdf_trims_font_to_fit_box(self):
        # A translation that is much taller than its box must be font-trimmed so
        # it stays inside the box instead of overlapping what is below it.
        src = _OUT / "sample_over.pdf"
        build_sample_pdf(src, pages=1)
        doc = pdfio.extract_document_text(src)
        # Give the first block a long translation that needs many wrapped lines.
        long_cjk = (
            "本飞机专为单飞行员操作而开发，并经过相应调整以更好地模拟真实环境。"
            "其制作综合了多个真实世界的数据点和来自不同时期、不同来源的手册，"
            "并通过对各类组件的修改来让它们更容易在微软飞行模拟器中管理。"
        )
        per_page = [[""] * len(doc.pages[0])]
        per_page[0][0] = long_cjk
        out = _OUT / "translated_over.pdf"
        pdfio.save_translated_pdf(src, doc.pages, per_page, out, "Chinese")
        d = fitz.open(str(out))
        page = d[0]
        info = page.get_text("dict")
        rects = []
        for b in info.get("blocks", []):
            if b.get("type") != 0:
                continue
            for l in b.get("lines", []):
                t = "".join(s["text"] for s in l["spans"]).strip()
                if t:
                    rects.append((fitz.Rect(l["bbox"]), t))
        self.assertTrue(rects, "expected rendered translation text")
        box = fitz.Rect(doc.pages[0][0].x0, doc.pages[0][0].y0,
                        doc.pages[0][0].x1, doc.pages[0][0].y1)
        # Text must not spill below the original box (allow a small tolerance).
        bottom = max(r.y1 for r, _t in rects)
        self.assertLessEqual(bottom, box.y1 + 1.0)
        # And every line must fit within the box width (no page overflow).
        for r, _t in rects:
            self.assertLessEqual(r.x1, box.x1 + 1.0)
        d.close()


    def test_two_column_reading_order_and_metadata(self):
        src = _OUT / "two_col.pdf"
        build_two_column_pdf(src)
        doc = pdfio.extract_document_text(src)
        self.assertEqual(doc.page_count, 1)
        # Column-major reading order: the whole left column first, then the
        # right column (including the right-aligned footer line).
        texts = doc.blocks
        left_idx = [i for i, t in enumerate(texts) if t.startswith("Left column")]
        right_idx = [i for i, t in enumerate(texts) if t.startswith("Right column")]
        self.assertTrue(left_idx and right_idx)
        self.assertLess(max(left_idx), min(right_idx))
        self.assertEqual(texts[-1], "Page 1 of 9")

        # Layout hints are captured: every block carries a size, an alignment
        # and a single-line flag, and the footer is detected as right-aligned.
        for block in doc.pages[0]:
            self.assertGreater(block.size, 0)
            self.assertIn(block.align, ("left", "center", "right"))
            self.assertTrue(block.single_line)
        footer = doc.pages[0][-1]
        self.assertEqual(footer.text, "Page 1 of 9")
        self.assertEqual(footer.align, "right")

    def test_bilingual_pdf_mirrors_block_positions(self):
        src = _OUT / "sample_m.pdf"
        build_sample_pdf(src, pages=1)
        doc = pdfio.extract_document_text(src)
        per_page = pdfio.group_by_page(doc.block_pages, doc.blocks, doc.page_count)
        fake = [[f"TR-{i}" for i, _t in enumerate(pg)] for pg in per_page]
        out = _OUT / "bilingual_mirror.pdf"
        pdfio.save_interleaved_pdf(src, fake, out, "Chinese", doc.pages)
        d = fitz.open(str(out))
        self.assertEqual(d.page_count, 2)  # original page + translation page
        trans_lines = _text_lines(d[1])
        self.assertEqual(len(trans_lines), len(doc.pages[0]))
        # Each translated block must sit at its source block's y position.
        for block, (rect, _t) in zip(doc.pages[0], trans_lines):
            self.assertGreaterEqual(rect.y0, block.y0 - 2.0)
            self.assertLessEqual(rect.y0, block.y1 + 2.0)
        d.close()

    def test_single_line_block_vertically_centered(self):
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=50, y0=100, x1=350, y1=200,
            size=12.0, align="left", bold=False, single_line=True,
        )
        pdfio._draw_translated_block(page, font, block, "居中文本")
        lines = _text_lines(page)
        self.assertEqual(len(lines), 1)
        (rect, text), = lines
        self.assertEqual(text, "居中文本")
        # The rendered line's vertical centre sits in the box's centre.
        self.assertAlmostEqual(
            (rect.y0 + rect.y1) / 2, (100 + 200) / 2, delta=2.0,
        )

    def test_right_aligned_block_hugs_right_edge(self):
        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=50, y0=100, x1=350, y1=130,
            size=12.0, align="right", bold=False, single_line=True,
        )
        pdfio._draw_translated_block(page, font, block, "右对齐")
        (rect, _t), = _text_lines(page)
        self.assertAlmostEqual(rect.x1, 350.0, delta=1.5)

    def test_bold_block_renders_text(self):
        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=50, y0=100, x1=350, y1=130,
            size=12.0, align="left", bold=True, single_line=True,
        )
        pdfio._draw_translated_block(page, font, block, "粗体标题")
        (rect, text), = _text_lines(page)
        self.assertEqual(text, "粗体标题")
        # Bold must NOT pull in a second font: every rendered span uses the
        # standard cjk font (a mixed Heiti/Droid look was a visible defect).
        fonts = {
            s["font"]
            for b in page.get_text("dict")["blocks"] if b.get("type") == 0
            for l in b["lines"] for s in l["spans"]
        }
        self.assertEqual(fonts, {"Droid Sans Fallback Regular"})


class WrapTest(unittest.TestCase):
    """``_wrap`` must wrap space-less CJK text (and keep Latin words intact)."""

    def _font(self):
        return fitz.Font("cjk")

    def test_cjk_long_paragraph_wraps_into_multiple_lines(self):
        font = self._font()
        text = (
            "本飞机专为单飞行员操作而开发，并经过相应调整以更好地模拟真实环境。"
            "其制作综合了多个真实世界的数据点和来自不同时期、不同来源的手册，"
            "并通过对各类组件的修改来让它们更容易在微软飞行模拟器中管理。"
        )
        lines = pdfio._wrap(font, text, 585.1, 11.0)
        # Fix regression: a single space-less CJK "word" must be broken, not kept
        # as one overflowing line.
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(
                font.text_length(line, fontsize=11.0), 585.1,
                f"line too wide: {line!r}",
            )
        # Wrapping must not lose or duplicate characters.
        self.assertEqual("".join(lines), text)

    def test_latin_words_stay_intact(self):
        font = self._font()
        text = "This aircraft has been developed for single pilot operations and has been adapted."
        lines = pdfio._wrap(font, text, 300.0, 11.0)
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(font.text_length(line, fontsize=11.0), 300.0)
        # The content must be preserved exactly (words joined by single spaces).
        self.assertEqual(" ".join(lines).replace("  ", " "), text)

    def test_short_cjk_fits_on_one_line(self):
        font = self._font()
        lines = pdfio._wrap(font, "仅供模拟使用", 585.1, 11.0)
        self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()
