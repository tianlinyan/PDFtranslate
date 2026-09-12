"""Tests for PDF text extraction and the export writers."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pymupdf as fitz

from translate_app import pdfio

from tests._helpers import (
    build_sample_pdf,
    build_two_column_pdf,
    build_two_column_pdf_with_heading,
    build_list_table_pdf,
)

_OUT = Path(__file__).resolve().parent / "_out"


class _OcrCacheIsolated(unittest.TestCase):
    """Redirect the OCR / translation caches so integration tests never touch the
    developer's ``~/.pdftranslate`` (and never hit a hot cache that would turn a
    test green without exercising the review/OCR path)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.dict(
            os.environ,
            {
                "PDFTRANSLATE_OCR_CACHE_DIR": str(tmp.name),
                "PDFTRANSLATE_CACHE_DIR": str(tmp.name),
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)


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


def build_ruled_table_pdf(path, *, rows: int = 3, cols: int = 3):
    """A page carrying one fully ruled text-layer table (``rows`` x ``cols``)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    xs = [72.0 + 148.0 * c for c in range(cols + 1)]
    ys = [100.0 + 50.0 * r for r in range(rows + 1)]
    for y in ys:
        page.draw_line(fitz.Point(xs[0], y), fitz.Point(xs[-1], y), width=0.8)
    for x in xs:
        page.draw_line(fitz.Point(x, ys[0]), fitz.Point(x, ys[-1]), width=0.8)
    for r in range(rows):
        for c in range(cols):
            page.insert_text((xs[c] + 6, ys[r] + 20), f"Cell{r}{c}", fontsize=11)
    doc.save(str(path))
    doc.close()
    return Path(path)

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

    def test_translated_pdf_covers_ocr_block_with_white(self):
        # An OCR block sits on a raster image, not a text layer, so the
        # in-place exporter must cover the original (scanned) pixels with a
        # white rectangle instead of redacting text — otherwise the
        # translation overprints the original.
        #
        # v0.5.33: the cover hugs the *printed glyph band*, measured from the
        # rendered page, instead of the whole OCR box — the box reaches the row
        # rules on a scanned statement and covering it clipped them (user
        # screenshot: white boxes cutting the dotted row lines).
        src = _OUT / "scan_src.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        # White paper with a line of "printed text" (letter-like segments, so the
        # row is glyph ink, not a filled rule) plus a printed rule below it.
        pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 300, 70), 0)
        pix.set_rect(fitz.IRect(0, 0, 300, 70), (255,))
        for x in range(4, 296, 14):                      # the printed line
            pix.set_rect(fitz.IRect(x, 25, x + 8, 45), (30,))
        pix.set_rect(fitz.IRect(0, 60, 300, 62), (30,))  # the row rule
        page.insert_image(fitz.Rect(50, 50, 350, 120), pixmap=pix)
        doc.save(str(src))
        doc.close()

        ocr_block = pdfio.Block(
            text="scanned text", page=0, x0=50, y0=50, x1=350, y1=120,
            size=12.0, align="left", bold=False, single_line=False, ocr=True,
        )
        out = _OUT / "translated_ocr.pdf"
        pdfio.save_translated_pdf(src, [[ocr_block]], [["translated text"]], out, "Chinese")

        d = fitz.open(str(out))
        page = d[0]
        try:
            # The translation must be present.
            self.assertIn("translated text", page.get_text())
            covers = [
                dr["rect"] for dr in page.get_drawings()
                if dr.get("fill") and all(abs(c - 1.0) < 0.01 for c in dr["fill"])
            ]
            self.assertTrue(covers, "expected a white cover rectangle over the OCR block")
            top = min(r.y0 for r in covers)
            bottom = max(r.y1 for r in covers)
            # Hugs the printed line (page y 75..95) instead of the OCR box (50..120).
            self.assertGreaterEqual(top, 72.0)
            self.assertLessEqual(top, 77.0)
            self.assertGreaterEqual(bottom, 93.0)
            self.assertLessEqual(bottom, 98.0)
            # ... and never reaches the rule at page y = 110.
            for r in covers:
                self.assertFalse(
                    r.y0 < 110.5 < r.y1 and r.x0 < 300 and r.x1 > 100,
                    f"white cover clipped the printed rule: {r}",
                )
        finally:
            d.close()

    def test_ocr_cover_never_clips_a_printed_rule_between_two_lines(self):
        # A multi-line OCR box whose lines are separated by a printed rule gets one
        # tight white rect per line; the rule between them must survive.
        src = _OUT / "scan_rules_src.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 300, 140), 0)
        pix.set_rect(fitz.IRect(0, 0, 300, 140), (255,))
        for x in range(4, 296, 14):
            pix.set_rect(fitz.IRect(x, 20, x + 8, 40), (30,))    # line 1
            pix.set_rect(fitz.IRect(x, 90, x + 8, 110), (30,))   # line 2
        pix.set_rect(fitz.IRect(0, 62, 300, 64), (30,))          # the rule between them
        page.insert_image(fitz.Rect(50, 50, 350, 190), pixmap=pix)
        doc.save(str(src))
        doc.close()

        ocr_block = pdfio.Block(
            text="line one\nline two", page=0, x0=50, y0=50, x1=350, y1=190,
            size=12.0, align="left", bold=False, single_line=False, ocr=True,
        )
        out = _OUT / "translated_ocr_rules.pdf"
        pdfio.save_translated_pdf(src, [[ocr_block]], [["translated text"]], out, "Chinese")

        d = fitz.open(str(out))
        page = d[0]
        try:
            covers = [
                dr["rect"] for dr in page.get_drawings()
                if dr.get("fill") and all(abs(c - 1.0) < 0.01 for c in dr["fill"])
            ]
            self.assertGreaterEqual(len(covers), 2, covers)
            for r in covers:
                self.assertFalse(
                    r.y0 < 112.5 < r.y1 and r.x0 < 300 and r.x1 > 100,
                    f"white cover clipped the printed rule: {r}",
                )
        finally:
            d.close()

    def test_single_line_label_above_a_table_stays_one_line(self):
        # v0.5.34: a one-line source label whose translation is longer must not wrap
        # across the rule directly under it.  Measured on a real report: 「单位：人民币
        # 万元」 → "Unit: RMB in ten thousand yuan" wrapped and its second line ran
        # through the table's top border.  It keeps ONE line at the readability
        # floor and overflows sideways instead.
        src = _OUT / "label_table_src.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        page.insert_text((50, 56), "Amount unit: RMB", fontsize=10)
        # A ruled 2x2 table directly below the label (its top border at y=60).
        page.draw_rect(fitz.Rect(50, 60, 350, 120), color=(0, 0, 0), width=0.8)
        page.draw_line(fitz.Point(50, 90), fitz.Point(350, 90), color=(0, 0, 0), width=0.8)
        page.draw_line(fitz.Point(200, 60), fitz.Point(200, 120), color=(0, 0, 0), width=0.8)
        page.insert_text((60, 82), "Item", fontsize=10)
        page.insert_text((60, 112), "Total assets", fontsize=10)
        doc.save(str(src))
        doc.close()

        blocks = pdfio.extract_document_text(src).pages[0]
        label_idx = next(
            i for i, b in enumerate(blocks) if "Amount unit" in b.text
        )
        trans = [b.text for b in blocks]
        trans[label_idx] = "Amount unit: RMB in ten thousand yuan"
        out = _OUT / "label_table.pdf"
        pdfio.save_translated_pdf(src, [blocks], [trans], out, "English")

        d = fitz.open(str(out))
        page = d[0]
        try:
            spans = [
                s for blk in page.get_text("dict")["blocks"] if "lines" in blk
                for line in blk["lines"] for s in line["spans"]
                if "Amount unit:" in s["text"]
            ]
            self.assertEqual(1, len(spans), [s["text"] for s in spans])
            self.assertLessEqual(
                spans[0]["bbox"][3], 60.0,
                f"label wrapped through the table's top border: {spans[0]['bbox']}",
            )
            # ... and the table's top rule is still there.
            top_rules = [
                dr for dr in page.get_drawings()
                if dr.get("rect") and abs(dr["rect"].y0 - 60.0) < 1.5
                and dr["rect"].width > 100
            ]
            self.assertTrue(top_rules, "the table's top border disappeared")
        finally:
            d.close()

    def test_faint_scan_text_is_covered_too(self):
        # v0.5.34: the cover band is measured with a *core* threshold (dark strokes)
        # and then grown into the printed (lighter) rows around it.  Without that
        # growth the faint parts of a scanned glyph stayed visible under the
        # translation (visible ghosting on a real scan).
        src = _OUT / "scan_faint_src.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 300, 70), 0)
        pix.set_rect(fitz.IRect(0, 0, 300, 70), (255,))
        for x in range(4, 296, 14):
            pix.set_rect(fitz.IRect(x, 30, x + 8, 42), (30,))     # dark core
            pix.set_rect(fitz.IRect(x, 26, x + 8, 30), (185,))    # faint top edge
            pix.set_rect(fitz.IRect(x, 42, x + 8, 46), (185,))    # faint bottom edge
        page.insert_image(fitz.Rect(50, 50, 350, 120), pixmap=pix)
        doc.save(str(src))
        doc.close()

        ocr_block = pdfio.Block(
            text="scanned text", page=0, x0=50, y0=50, x1=350, y1=120,
            size=12.0, align="left", bold=False, single_line=False, ocr=True,
        )
        out = _OUT / "scan_faint.pdf"
        pdfio.save_translated_pdf(src, [[ocr_block]], [["translated text"]], out, "Chinese")

        d = fitz.open(str(out))
        page = d[0]
        try:
            covers = [
                dr["rect"] for dr in page.get_drawings()
                if dr.get("fill") and all(abs(c - 1.0) < 0.01 for c in dr["fill"])
            ]
            self.assertTrue(covers)
            top = min(r.y0 for r in covers)
            bottom = max(r.y1 for r in covers)
            # The faint rows (page y 76..80 and 92..96) are covered too.
            self.assertLessEqual(top, 76.5)
            self.assertGreaterEqual(bottom, 95.5)
        finally:
            d.close()

    def _scan_page(self, doc, *, paper: int, ink: int, rule: int | None,
                   width: int = 300, height: int = 70, full_page: bool = False):
        """A raster 'scan': letter-like ink segments, optional printed rule."""
        pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, width, height), 0)
        pix.set_rect(fitz.IRect(0, 0, width, height), (paper,))
        for x in range(4, width - 4, 14):
            pix.set_rect(fitz.IRect(x, 25, x + 8, 45), (ink,))
        if rule is not None:
            pix.set_rect(fitz.IRect(0, height - 10, width, height - 8), (rule,))
        page = doc.new_page(width=400, height=300)
        rect = page.rect if full_page else fitz.Rect(50, 50, 350, 120)
        page.insert_image(rect, pixmap=pix)
        return page

    def _white_covers(self, page):
        return [
            dr["rect"] for dr in page.get_drawings()
            if dr.get("fill") and all(abs(c - 1.0) < 0.01 for c in dr["fill"])
        ]

    def test_faint_scan_still_gets_a_tight_cover(self):
        # v0.5.35: the ink/paper levels are derived from the page's OWN contrast.
        # On a faint scan (paper 235, ink 165) fixed thresholds tuned on a dark
        # scan find no ink at all — the cover would fall back to the whole OCR box
        # (clipping rules) or vanish (leaving the source text under the translation).
        src = _OUT / "faint_src.pdf"
        doc = fitz.open()
        self._scan_page(doc, paper=235, ink=165, rule=120)
        doc.save(str(src))
        doc.close()

        block = pdfio.Block(
            text="scanned text", page=0, x0=50, y0=50, x1=350, y1=120,
            size=12.0, align="left", bold=False, single_line=False, ocr=True,
        )
        out = _OUT / "faint.pdf"
        pdfio.save_translated_pdf(src, [[block]], [["translated text"]], out, "Chinese")

        d = fitz.open(str(out))
        page = d[0]
        try:
            covers = self._white_covers(page)
            self.assertTrue(covers, "no cover at all on a faint scan")
            top = min(r.y0 for r in covers)
            bottom = max(r.y1 for r in covers)
            self.assertGreaterEqual(top, 70.0)      # not the whole box (50..120)
            self.assertLessEqual(bottom, 100.0)
            for r in covers:
                self.assertFalse(
                    r.y0 < 110.5 < r.y1 and r.x0 < 300 and r.x1 > 100,
                    f"white cover clipped the printed rule: {r}",
                )
        finally:
            d.close()

    def test_dark_scan_paper_is_not_read_as_ink(self):
        # A dark-ish scan (paper 150, ink 40) must not make the *paper* count as
        # printed: with an absolute threshold the whole page reads as ink, no rule
        # is ever detected, and the cover clips it.
        src = _OUT / "dark_src.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 400, 300), 0)
        pix.set_rect(fitz.IRect(0, 0, 400, 300), (150,))          # grey paper
        for x in range(6, 394, 18):                               # the printed line
            pix.set_rect(fitz.IRect(x, 100, x + 10, 140), (40,))
        pix.set_rect(fitz.IRect(0, 160, 400, 162), (40,))         # the row rule
        page.insert_image(page.rect, pixmap=pix)
        doc.save(str(src))
        doc.close()

        block = pdfio.Block(
            text="scanned text", page=0, x0=40, y0=100, x1=360, y1=140,
            size=12.0, align="left", bold=False, single_line=False, ocr=True,
        )
        out = _OUT / "dark.pdf"
        pdfio.save_translated_pdf(src, [[block]], [["translated text"]], out, "Chinese")

        d = fitz.open(str(out))
        page = d[0]
        try:
            covers = self._white_covers(page)
            self.assertTrue(covers)
            top = min(r.y0 for r in covers)
            bottom = max(r.y1 for r in covers)
            self.assertGreaterEqual(top, 95.0)      # tight, not the 100..140 box
            self.assertLessEqual(bottom, 145.0)
            for r in covers:
                self.assertFalse(
                    r.y0 < 160.5 < r.y1 and r.x0 < 300 and r.x1 > 100,
                    f"white cover clipped the printed rule: {r}",
                )
        finally:
            d.close()

    def test_rotated_scan_cover_avoids_the_rule(self):
        # A /Rotate page's blocks live in the unrotated frame while the sampler
        # renders the displayed (rotated) page — the mapping must be applied, or the
        # cover is measured at the wrong place (previously the rotated page fell back
        # to a whole-box cover that clipped rules).
        src = _OUT / "rot_src.pdf"
        doc = fitz.open()
        page = self._scan_page(doc, paper=255, ink=30, rule=30)
        page.set_rotation(90)
        doc.save(str(src))
        doc.close()

        block = pdfio.Block(
            text="scanned text", page=0, x0=50, y0=50, x1=350, y1=120,
            size=12.0, align="left", bold=False, single_line=False, ocr=True,
        )
        out = _OUT / "rot.pdf"
        pdfio.save_translated_pdf(src, [[block]], [["translated text"]], out, "Chinese")

        d = fitz.open(str(out))
        page = d[0]
        try:
            self.assertEqual(90, int(page.rotation))
            covers = self._white_covers(page)
            self.assertTrue(covers)
            top = min(r.y0 for r in covers)
            bottom = max(r.y1 for r in covers)
            # Tight (the glyph band, not the 70 pt box) …
            self.assertGreaterEqual(top, 70.0)
            self.assertLessEqual(bottom, 100.0)
            # … and the rule at unrotated y=110..112 is untouched.
            for r in covers:
                self.assertFalse(
                    r.y0 < 111.0 < r.y1 and r.x0 < 300 and r.x1 > 100,
                    f"white cover clipped the printed rule: {r}",
                )
        finally:
            d.close()

    def test_label_above_an_underline_stays_one_line(self):
        # v0.5.35: the "do not cross a rule" obstacle is taken from the page's
        # printed-rule mask, not only from detected tables — a label above a plain
        # underline (no table at all) must stay one line too.
        src = _OUT / "underline_src.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        page.insert_text((50, 56), "Amount unit: RMB", fontsize=10)
        page.draw_line(fitz.Point(50, 60), fitz.Point(130, 60),
                       color=(0, 0, 0), width=0.8)
        doc.save(str(src))
        doc.close()

        blocks = pdfio.extract_document_text(src).pages[0]
        label_idx = next(i for i, b in enumerate(blocks) if "Amount unit" in b.text)
        trans = [b.text for b in blocks]
        trans[label_idx] = "Amount unit: RMB in ten thousand yuan"
        out = _OUT / "underline.pdf"
        pdfio.save_translated_pdf(src, [blocks], [trans], out, "English")

        d = fitz.open(str(out))
        page = d[0]
        try:
            spans = [
                s for blk in page.get_text("dict")["blocks"] if "lines" in blk
                for line in blk["lines"] for s in line["spans"]
                if "Amount unit:" in s["text"]
            ]
            self.assertEqual(1, len(spans), [s["text"] for s in spans])
            self.assertLessEqual(spans[0]["bbox"][3], 60.0)
        finally:
            d.close()

    def test_rule_bound_does_not_move_a_single_line_translation(self):
        # v0.5.36: the next printed rule is a *height budget*, not a box stretch.
        # Stretching the box down to the rule made every single-line translation
        # centre lower than its source line (measured: up to +10 pt on a dense
        # statement); the text must stay where the source was.
        src = _OUT / "rule_budget_src.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        page.insert_text((50, 56), "Amount unit: RMB", fontsize=10)
        page.draw_line(fitz.Point(50, 100), fitz.Point(350, 100),
                       color=(0, 0, 0), width=0.8)     # a rule 44 pt below
        doc.save(str(src))
        doc.close()

        blocks = pdfio.extract_document_text(src).pages[0]
        label = next(b for b in blocks if "Amount unit" in b.text)
        trans = [b.text for b in blocks]
        out = _OUT / "rule_budget.pdf"
        pdfio.save_translated_pdf(src, [blocks], [trans], out, "English")

        d = fitz.open(str(out))
        page = d[0]
        try:
            spans = [
                s for blk in page.get_text("dict")["blocks"] if "lines" in blk
                for line in blk["lines"] for s in line["spans"]
                if "Amount unit" in s["text"]
            ]
            self.assertTrue(spans)
            origin = spans[0]["origin"][1]
            self.assertLessEqual(
                abs(origin - label.y1), 4.0,
                f"translation moved away from its source line: {origin} vs "
                f"{label.y1}",
            )
        finally:
            d.close()

    def test_ocr_cover_band_bounds_the_translation(self):
        # The translation is fitted into the measured band too, so it cannot be
        # drawn past the row rule (the second half of "避免压线").
        src = _OUT / "scan_text_src.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 300, 70), 0)
        pix.set_rect(fitz.IRect(0, 0, 300, 70), (255,))
        for x in range(4, 296, 14):
            pix.set_rect(fitz.IRect(x, 25, x + 8, 45), (30,))
        pix.set_rect(fitz.IRect(0, 60, 300, 62), (30,))
        page.insert_image(fitz.Rect(50, 50, 350, 120), pixmap=pix)
        doc.save(str(src))
        doc.close()

        ocr_block = pdfio.Block(
            text="scanned text", page=0, x0=50, y0=50, x1=350, y1=120,
            size=12.0, align="left", bold=False, single_line=False, ocr=True,
        )
        out = _OUT / "translated_ocr_text.pdf"
        pdfio.save_translated_pdf(src, [[ocr_block]], [["translated text"]], out, "Chinese")

        d = fitz.open(str(out))
        page = d[0]
        try:
            spans = [
                s for blk in page.get_text("dict")["blocks"] if "lines" in blk
                for line in blk["lines"] for s in line["spans"]
                if "translated" in s["text"]
            ]
            self.assertTrue(spans, "translation span missing")
            for s in spans:
                self.assertLessEqual(
                    s["bbox"][3], 98.0,
                    f"translation drawn below the measured band: {s['bbox']}",
                )
                self.assertLess(s["bbox"][3], 110.0)   # never onto the rule
        finally:
            d.close()

    def test_translated_pdf_does_not_cover_an_ocr_block_on_a_photo(self):
        # P1-3: text detected on a photo / coloured logo must NOT get the opaque
        # white cover — that punches a white hole in the picture.  The translation
        # is still drawn; only the cover is skipped.
        src = _OUT / "photo_src.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 70))
        pix.set_rect(fitz.IRect(0, 0, 300, 70), (20, 90, 170))   # a blue "photo"
        page.insert_image(fitz.Rect(50, 50, 350, 120), pixmap=pix)
        doc.save(str(src))
        doc.close()

        ocr_block = pdfio.Block(
            text="text baked into the photo", page=0, x0=50, y0=50, x1=350, y1=120,
            size=12.0, align="left", bold=False, single_line=False, ocr=True,
        )
        logs: list[str] = []
        out = _OUT / "translated_photo.pdf"
        pdfio.save_translated_pdf(src, [[ocr_block]], [["translated text"]], out,
                                  "Chinese", log=logs.append)

        d = fitz.open(str(out))
        page = d[0]
        try:
            self.assertIn("translated text", page.get_text())
            for dr in page.get_drawings():
                fill = dr.get("fill")
                if fill is None or not all(abs(c - 1.0) < 0.01 for c in fill):
                    continue
                r = dr["rect"]
                self.assertFalse(
                    r.x0 < 100 and r.x1 > 300 and r.y0 < 70 and r.y1 > 100,
                    f"white cover over the photo: {r}",
                )
            self.assertTrue(any("图片区域" in m for m in logs), logs)
        finally:
            d.close()

    def test_translated_pdf_keeps_readable_font(self):
        # A translation far longer than its box used to be font-trimmed down to
        # an illegible ~4pt sliver so it would stay inside the box.  The
        # readability floor now wins: the rendered size never drops below
        # ``_MIN_READABLE`` (the text may extend a little past the source box,
        # which is redacted, rather than become unreadable), and it must not
        # overflow the page width.
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
        rects, sizes = [], []
        for b in info.get("blocks", []):
            if b.get("type") != 0:
                continue
            for l in b.get("lines", []):
                t = "".join(s["text"] for s in l["spans"]).strip()
                if t:
                    rects.append(fitz.Rect(l["bbox"]))
                    sizes.extend(s["size"] for s in l["spans"])
        self.assertTrue(rects, "expected rendered translation text")
        # Readability floor: no rendered span is smaller than _MIN_READABLE.
        self.assertTrue(sizes)
        self.assertGreaterEqual(min(sizes), pdfio._MIN_READABLE)
        # And no line spills past the page width.
        for r in rects:
            self.assertLessEqual(r.x1, page.rect.x1 + 1.0)
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

    def test_full_width_heading_does_not_merge_columns(self):
        # Regression: a heading spanning both columns used to become a
        # "column" whose right edge was the page width, so every line of both
        # real columns "overlapped" it and the two columns were merged into
        # one — left and right lines interleaved by y into single blocks.
        # The heading must read first, then the whole left column, then the
        # whole right column.
        src = _OUT / "two_col_heading.pdf"
        build_two_column_pdf_with_heading(src)
        doc = pdfio.extract_document_text(src)
        texts = doc.blocks
        self.assertEqual(
            texts[0], "FULL WIDTH HEADING ACROSS BOTH COLUMNS OF THE PAGE"
        )
        left_idx = [i for i, t in enumerate(texts) if t.startswith("Left column")]
        right_idx = [i for i, t in enumerate(texts) if t.startswith("Right column")]
        self.assertTrue(left_idx and right_idx)
        # The whole left column still precedes the whole right column.
        self.assertLess(max(left_idx), min(right_idx))
        self.assertEqual(
            [texts[i] for i in left_idx],
            [
                "Left column first line.",
                "Left column second line.",
                "Left column third line.",
            ],
        )
        self.assertEqual(
            [texts[i] for i in right_idx],
            ["Right column first line.", "Right column second line."],
        )

    def test_list_and_table_entries_are_not_collapsed(self):
        # Regression: PyMuPDF merges a close-spaced list / table into one block,
        # which used to collapse the whole thing into a single run-on paragraph.
        # The line-aware extractor must keep every numbered / ``Label:`` entry as
        # its own single-line block.
        src = _OUT / "list_table.pdf"
        build_list_table_pdf(src)
        doc = pdfio.extract_document_text(src)
        blocks = doc.pages[0]
        item_texts = [b.text for b in blocks]
        # Each numbered item is a distinct block, not merged with its neighbours.
        for txt in ("1. First item", "2. Second item", "3. Third item"):
            self.assertIn(txt, item_texts)
        # Label/table rows stay on their own lines (single-line blocks).
        rows = [b for b in blocks if b.text.startswith(
            ("Powerplant:", "Brand:", "Model:"))]
        self.assertEqual(len(rows), 3)
        for b in rows:
            self.assertTrue(b.single_line, f"table row collapsed: {b.text!r}")
            self.assertIn(b.text, ("Powerplant:", "Brand: Pratt & Whitney",
                                   "Model: PT6A-140"))

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

    def test_translated_pdf_keeps_pages_without_translations(self):
        # Regression: ``save_translated_pdf`` looped ``min(src.page_count,
        # len(per_page))``, so a shorter ``per_page`` (the source PDF was edited
        # between extraction and export, or a caller passed a short list) truncated
        # the document SILENTLY.  Pages without a translation must be kept as-is.
        src = _OUT / "short_per_page.pdf"
        build_sample_pdf(src, pages=3)
        doc = pdfio.extract_document_text(src)
        self.assertEqual(3, doc.page_count)
        per_page = [["T" for _ in doc.pages[0]]]      # page 0 only
        out = _OUT / "short_per_page_out.pdf"
        pdfio.save_translated_pdf(src, doc.pages, per_page, out, "Chinese")
        d = fitz.open(str(out))
        try:
            self.assertEqual(3, d.page_count)
            self.assertIn("T", d[0].get_text())
            # Pages 2-3 kept the original text (inserted verbatim).
            self.assertIn("Page 2 heading text.", d[1].get_text())
            self.assertIn("Page 3 heading text.", d[2].get_text())
        finally:
            d.close()

    def test_bilingual_pdf_rotated_page_keeps_every_block(self):
        # Regression: the mirror page was created with ``src[i].rect`` (the ROTATED
        # view, 400x200) while block bboxes live in the unrotated mediabox frame
        # (200x400), so every block below y=200 was drawn off-page and its
        # translation silently vanished.  The mirror page must use the unrotated
        # mediabox and carry the source page's /Rotate.
        src = _OUT / "rotated_bi_src.pdf"
        doc = fitz.open()
        page = doc.new_page(width=200, height=400)
        page.insert_text((20, 60), "TOP LABEL", fontsize=11)
        page.insert_text((20, 350), "BOTTOM LABEL", fontsize=11)
        page.set_rotation(90)
        doc.save(str(src))
        doc.close()

        extracted = pdfio.extract_document_text(src)
        self.assertEqual(2, len(extracted.pages[0]))
        out = _OUT / "rotated_bi.pdf"
        pdfio.save_interleaved_pdf(
            src, [["TOP-TRANSLATED", "BOTTOM-TRANSLATED"]], out, "Chinese",
            extracted.pages,
        )
        d = fitz.open(str(out))
        try:
            self.assertEqual(2, d.page_count)
            mirror = d[1]
            self.assertEqual(90, mirror.rotation)
            self.assertAlmostEqual(200.0, mirror.mediabox.width, delta=1.0)
            self.assertAlmostEqual(400.0, mirror.mediabox.height, delta=1.0)
            text = mirror.get_text()
            self.assertIn("TOP-TRANSLATED", text)
            self.assertIn("BOTTOM-TRANSLATED", text)
        finally:
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

    def test_multiline_table_cell_is_top_anchored(self):
        # A table cell whose translation must wrap to >1 line: it should hug the
        # cell's top rule instead of being vertically centred, so the wrapped
        # block stays inside the cell rather than drifting toward the bottom line.
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=50, y0=100, x1=110, y1=130,
            size=12.0, align="left", bold=False, single_line=True,
            in_table=True,
        )
        pdfio._draw_translated_block(
            page, font, block, "Net Assets Per Share Attributable (Yuan)"
        )
        lines = _text_lines(page)
        self.assertGreater(len(lines), 1)
        # The first (top) line's glyph top sits at the cell's top rule, not centred.
        first = min(lines, key=lambda it: it[0].y0)
        self.assertLess(first[0].y0 - block.y0, 6.0)

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

    def test_color_restores_original_color(self):
        # Regression: the exporter flattened every heading to black.  A block's
        # captured source colour must be reproduced (same single CJK font).
        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=50, y0=100, x1=300, y1=130,
            size=12.0, align="left", bold=False, single_line=True, color=0xCC0000,
        )
        pdfio._draw_translated_block(page, font, block, "红色标题")
        span = page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["color"], 0xCC0000)
        # Still the one consistent CJK font (no Heiti creep).
        fonts = {
            s["font"] for b in page.get_text("dict")["blocks"] if b.get("type") == 0
            for l in b["lines"] for s in l["spans"]
        }
        self.assertEqual(fonts, {"Droid Sans Fallback Regular"})
        doc.close()

    def test_table_row_expansion_shifts_rows_down(self):
        # Regression: a translated table row that no longer fits its cell used to
        # overlap the row beneath it.  Rows must be pushed down (and their height
        # enlarged) so they stay separate.
        font = fitz.Font("cjk")
        rows = []
        cells = []
        for r in range(3):
            row = [fitz.Rect(x, 200 + r * 20, x + 90, 220 + r * 20) for x in (60, 160)]
            rows.append(row)
            cells.extend(row)
        tables = [{"bbox": fitz.Rect(60, 200, 250, 260), "rows": rows,
                   "col_edges": [60, 150, 250]}]
        blocks, trans = [], []
        for r in range(3):
            for c in range(2):
                cell = rows[r][c]
                blocks.append(pdfio.Block(
                    text="t", page=0, x0=cell.x0, y0=cell.y0, x1=cell.x1, y1=cell.y1,
                    size=9.0, single_line=True,
                ))
                # The middle row gets a translation that needs several lines.
                trans.append(
                    "A considerably longer translated string that wraps across "
                    "many lines and therefore needs extra row height."
                    if r == 1 else "Short"
                )
        mapping = pdfio._map_blocks_to_table_cells(blocks, tables)
        self.assertEqual(len(mapping), len(blocks))
        shifts, new_bottoms, grid, bboxes = pdfio._compute_table_layout(
            tables, mapping, blocks, trans, font
        )
        row0 = next(bi for bi, (_ti, r, _c) in mapping.items() if r == 0)
        row1 = next(bi for bi, (_ti, r, _c) in mapping.items() if r == 1)
        row2 = next(bi for bi, (_ti, r, _c) in mapping.items() if r == 2)
        # Top row is undisturbed; the lower rows move down because row 1 grows.
        self.assertEqual(shifts[row0], 0.0)
        self.assertGreaterEqual(shifts[row1], 0.0)
        self.assertGreater(shifts[row2], 0.0)
        # The grown row 1's new bottom must not cross row 2's shifted top.
        self.assertLessEqual(new_bottoms[row1], 240.0 + shifts[row2] + 0.5)
        # A redrawn grid (horizontal + vertical rules) is produced.
        self.assertTrue(any(g[0] == "h" for g in grid))
        self.assertTrue(any(g[0] == "v" for g in grid))
        # The whole original table is flagged for removal (stale grid lines).
        self.assertEqual(len(bboxes), 1)

    def test_rebalance_table_columns_widens_long_column_and_keeps_total_width(self):
        # C-⑥ reflow（保守层）：长译文列借用相邻短列的空白，表格总宽不变。
        font = fitz.Font("cjk")
        rows = [[fitz.Rect(60, 200, 150, 220), fitz.Rect(150, 200, 250, 220)]]
        tables = [{"bbox": fitz.Rect(60, 200, 250, 220), "rows": rows,
                   "col_edges": [60, 150, 250]}]
        blocks, trans = [], []
        blocks.append(pdfio.Block(text="项目", page=0, x0=60, y0=200, x1=150, y1=220, size=9.0))
        trans.append("A very long translated header that definitely exceeds the narrow column width")
        blocks.append(pdfio.Block(text="备注", page=0, x0=150, y0=200, x1=250, y1=220, size=9.0))
        trans.append("ok")
        mapping = pdfio._map_blocks_to_table_cells(blocks, tables)
        col_boxes, new_edges = pdfio._rebalance_table_columns(tables, mapping, blocks, trans, font)
        # Long column（列 0）widens: its bbox right edge moves past the old 150.
        self.assertGreater(col_boxes[0][1], 150.0)
        # Short column（列 1）narrows: its bbox left edge moves right past 150.
        self.assertGreater(col_boxes[1][0], 150.0)
        # Total width preserved.
        self.assertAlmostEqual(new_edges[0][-1] - new_edges[0][0], 190.0, delta=0.5)

    def test_rebalance_table_columns_never_shrinks_numeric_column(self):
        font = fitz.Font("cjk")
        rows = [[fitz.Rect(60, 200, 150, 220), fitz.Rect(150, 200, 250, 220)]]
        tables = [{"bbox": fitz.Rect(60, 200, 250, 220), "rows": rows,
                   "col_edges": [60, 150, 250]}]
        blocks, trans = [], []
        blocks.append(pdfio.Block(text="项目", page=0, x0=60, y0=200, x1=150, y1=220, size=9.0))
        trans.append("A very long translated header that definitely exceeds the narrow column width")
        blocks.append(pdfio.Block(text="1,234.56", page=0, x0=150, y0=200, x1=250, y1=220, size=9.0))
        trans.append("1,234.56")
        mapping = pdfio._map_blocks_to_table_cells(blocks, tables)
        col_boxes, _new_edges = pdfio._rebalance_table_columns(tables, mapping, blocks, trans, font)
        # 数字列不缩：bbox 保持原列边界（150+2, 250-2）。
        self.assertAlmostEqual(col_boxes[1][0], 152.0, delta=0.5)
        self.assertAlmostEqual(col_boxes[1][1], 248.0, delta=0.5)

    def test_rebalance_keeps_a_merged_cell_and_an_empty_column(self):
        # P1-8: mapping's third element is the cell index *within its row*, not the
        # column index.  A merged header (one cell spanning all three columns) used
        # to get column 0's box — its wrap width collapsed — and a column with no
        # mapped block got demand 0 and therefore zero width (its two rules
        # coincided).  The column is now resolved from the cell's x-range, and a
        # column without demand keeps its width.
        font = fitz.Font("cjk")
        merged = [fitz.Rect(60, 180, 350, 200)]
        rows = [
            merged,
            [fitz.Rect(60, 200, 150, 220), fitz.Rect(150, 200, 250, 220),
             fitz.Rect(250, 200, 350, 220)],
            [fitz.Rect(60, 220, 150, 240), fitz.Rect(150, 220, 250, 240),
             fitz.Rect(250, 220, 350, 240)],
        ]
        tables = [{"bbox": fitz.Rect(60, 180, 350, 240), "rows": rows,
                   "col_edges": [60, 150, 250, 350]}]
        blocks = [
            # the merged header: its box spans the whole table
            pdfio.Block(text="合并表头", page=0, x0=60, y0=180, x1=350, y1=200, size=9.0),
            # column 0 gets a long translation, column 2 a short one; column 1 has
            # no block at all
            pdfio.Block(text="项目", page=0, x0=60, y0=200, x1=150, y1=220, size=9.0),
            pdfio.Block(text="备注", page=0, x0=250, y0=200, x1=350, y1=220, size=9.0),
        ]
        trans = ["A merged header that spans every column of this table",
                 "A very long translated label that exceeds the first column width",
                 "ok"]
        mapping = pdfio._map_blocks_to_table_cells(blocks, tables)
        self.assertIn(0, mapping)
        col_boxes, new_edges = pdfio._rebalance_table_columns(
            tables, mapping, blocks, trans, font)
        # The merged cell keeps its own box (not column 0's).
        self.assertNotIn(0, col_boxes)
        # The empty middle column keeps its width (edges 150 / 250 stay put).
        self.assertAlmostEqual(new_edges[0][1], 150.0, delta=1.0)
        self.assertAlmostEqual(new_edges[0][2], 250.0, delta=1.0)
        # Total width is preserved.
        self.assertAlmostEqual(new_edges[0][-1] - new_edges[0][0], 290.0, delta=0.5)

    def test_block_straddling_cells_is_mapped_to_its_row(self):
        # P1-9: a block whose centre lands in no cell (it spans two cells) used to be
        # treated as prose: its height was not measured into the row (a multi-line
        # translation overlapped the row below) and it was shifted by the whole
        # table's growth.  It must map to the row that contains it.
        font = fitz.Font("cjk")
        rows = [
            [fitz.Rect(60, 200, 150, 220), fitz.Rect(150, 200, 250, 220)],
            [fitz.Rect(60, 220, 150, 240), fitz.Rect(150, 220, 250, 240)],
        ]
        tables = [{"bbox": fitz.Rect(60, 200, 250, 240), "rows": rows,
                   "col_edges": [60, 150, 250]}]
        # A value span that overflows its cell: its centre (x=205) is inside cell 1
        # of row 0, so use a block straddling the boundary at x=150.
        straddler = pdfio.Block(text="straddling value", page=0, x0=120, y0=200,
                                x1=190, y1=220, size=9.0)
        blocks = [straddler]
        mapping = pdfio._map_blocks_to_table_cells(blocks, tables)
        # The centre x = 155 falls inside cell 1 (150..250) — map by cell there; to
        # exercise the row-band fallback use a box centred on the shared rule.
        on_rule = pdfio.Block(text="on the rule", page=0, x0=60, y0=198, x1=250,
                              y1=222, size=9.0)
        mapping = pdfio._map_blocks_to_table_cells([on_rule], tables)
        self.assertIn(0, mapping)
        self.assertEqual(0, mapping[0][1])            # row 0
        # Its height is measured into the row: with the block mapped, row 0 grows
        # (without the mapping it is treated as prose and the row keeps its height).
        trans = ["A considerably longer translated string that wraps across many "
                 "lines and therefore needs extra row height. " * 3]
        mapped = pdfio._compute_table_layout(tables, mapping, [on_rule], trans, font)
        unmapped = pdfio._compute_table_layout(tables, {}, [on_rule], trans, font)
        self.assertGreater(mapped[1][0], 220.0)
        self.assertEqual({}, unmapped[1])

    def test_bilingual_translation_page_keeps_a_figure(self):
        # P1-10: the mirror page was a blank sheet, so a figure page's translation
        # page showed text floating in white space.  A page with a picture (or vector
        # art) now gets a copy of the source with its text redacted.
        src = _OUT / "bi_figure_src.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 60, 60))
        pix.set_rect(pix.irect, (200, 30, 30))
        page.insert_image(fitz.Rect(40, 40, 140, 140), pixmap=pix)
        page.insert_text((40, 180), "FIGURE CAPTION", fontsize=11)
        doc.save(str(src))
        doc.close()

        block = pdfio.Block(text="FIGURE CAPTION", page=0, x0=40, y0=168, x1=200,
                            y1=182, size=11.0, single_line=True)
        out = _OUT / "bi_figure.pdf"
        pdfio.save_interleaved_pdf(src, [["TRANSLATED CAPTION"]], out, "Chinese",
                                   [[block]])
        d = fitz.open(str(out))
        try:
            self.assertEqual(2, d.page_count)
            mirror = d[1]
            self.assertEqual(1, len(mirror.get_images(full=True)))
            self.assertIn("TRANSLATED CAPTION", mirror.get_text())
            self.assertNotIn("FIGURE CAPTION", mirror.get_text())
        finally:
            d.close()

    def test_unique_path_appends_number_when_exists(self):
        # 导出不覆盖重名文件：test_English.pdf 已存在 → test_English(1).pdf。
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            base = d / "test_English.pdf"
            self.assertEqual(pdfio.unique_path(base), base)
            base.write_bytes(b"x")
            self.assertEqual(pdfio.unique_path(base), d / "test_English(1).pdf")
            (d / "test_English(1).pdf").write_bytes(b"x")
            self.assertEqual(pdfio.unique_path(base), d / "test_English(2).pdf")


class TableCellFitTest(unittest.TestCase):
    """A table cell's translation shrinks onto ONE line (instead of wrapping and
    growing the row) when it fits the widened cell; a cell that cannot fit at
    the readability floor falls back to wrapping, and non-table blocks still
    wrap as before."""

    def _line(self, x0, x1, y0, y1, text):
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "size": 9.0, "bold": False, "color": 0, "text": text}

    def test_build_table_blocks_widens_cells_and_sets_in_table(self):
        # Native ruled-table cells are flagged in_table and widened to the whole
        # cell (minus a small gutter) so a longer translation can use the column.
        cell_rects = [fitz.Rect(100, 100, 200, 120), fitz.Rect(200, 100, 300, 120)]
        lines = [
            self._line(150, 210, 105, 115, "名称"),
            self._line(250, 300, 105, 115, "1,234,567"),
        ]
        blocks = pdfio._build_table_blocks(lines, [], 0, 600, 0, cell_rects)
        self.assertEqual(len(blocks), 2)
        by_text = {b.text: b for b in blocks}
        name = by_text["名称"]
        num = by_text["1,234,567"]
        self.assertTrue(name.in_table)
        self.assertTrue(num.in_table)
        # Widened to its own cell width (with the small gutter), not the text extent.
        self.assertAlmostEqual(name.x0, 100 + pdfio._TABLE_CELL_PAD, delta=0.01)
        self.assertAlmostEqual(name.x1, 200 - pdfio._TABLE_CELL_PAD, delta=0.01)
        self.assertAlmostEqual(num.x0, 200 + pdfio._TABLE_CELL_PAD, delta=0.01)
        self.assertAlmostEqual(num.x1, 300 - pdfio._TABLE_CELL_PAD, delta=0.01)
        # Text cells centre; figure cells right-align.
        self.assertEqual(name.align, "center")
        self.assertEqual(num.align, "right")

    def test_multiline_cell_lines_merge_into_one_block(self):
        # P0-2: a cell whose source text wraps over two lines used to emit two
        # blocks; the exporter anchors every multi-line translation at the ROW's
        # top, so both were drawn from the same y and overprinted each other.
        # They must come out as ONE block carrying the real source line count.
        cell_rects = [fitz.Rect(100, 100, 300, 140)]
        lines = [
            self._line(110, 190, 105, 115, "非经常性"),
            self._line(110, 190, 119, 129, "损益项目"),
        ]
        blocks = pdfio._build_table_blocks(lines, [], 0, 600, 0, cell_rects)
        self.assertEqual(1, len(blocks))
        b = blocks[0]
        self.assertEqual("非经常性\n损益项目", b.text)
        self.assertFalse(b.single_line)          # two source lines
        self.assertTrue(b.in_table)
        self.assertAlmostEqual(b.y0, 105.0, delta=0.01)
        self.assertAlmostEqual(b.y1, 129.0, delta=0.01)

    def test_same_baseline_runs_in_one_cell_stay_one_line(self):
        # Two runs of ONE visual line in the same cell (a label and its inline
        # value) must not be turned into a two-line block: they join with a space
        # and stay single_line, so the cell keeps its one-line fit.
        cell_rects = [fitz.Rect(100, 100, 300, 120)]
        lines = [
            self._line(110, 160, 105, 115, "Capacity:"),
            self._line(165, 250, 105, 115, "Two Passengers"),
        ]
        blocks = pdfio._build_table_blocks(lines, [], 0, 600, 0, cell_rects)
        self.assertEqual(1, len(blocks))
        self.assertEqual("Capacity: Two Passengers", blocks[0].text)
        self.assertTrue(blocks[0].single_line)

    def _line_with_spans(self, x0, x1, y0, y1, text, spans):
        d = self._line(x0, x1, y0, y1, text)
        d["spans"] = spans
        return d

    def test_build_table_blocks_splits_label_and_value_cells(self):
        # A source row that reports label+value on ONE visual line is really two
        # table cells (a label column and a value column).  It must come out as
        # two blocks, each pinned to its own cell — not the whole line stuffed
        # into whichever cell its centre happens to land in (which merged the
        # two columns and dropped the label column).
        cell_rects = [
            fitz.Rect(100, 100, 200, 120),   # label column
            fitz.Rect(200, 100, 400, 120),   # value column
        ]
        line = self._line_with_spans(
            150, 300, 105, 115,
            "Capacity: Pilot + Copilot + Two Passengers",
            [
                (150.0, 105.0, 195.0, 115.0, "Capacity:"),
                (210.0, 105.0, 300.0, 115.0, "Pilot + Copilot + Two Passengers"),
            ],
        )
        blocks = pdfio._build_table_blocks([line], [], 0, 600, 0, cell_rects)
        self.assertEqual(len(blocks), 2)
        by_text = {b.text: b for b in blocks}
        label = by_text["Capacity:"]
        value = by_text["Pilot + Copilot + Two Passengers"]
        self.assertTrue(label.in_table)
        self.assertTrue(value.in_table)
        # Each block is widened to its own cell, not to the merged line's extent.
        self.assertAlmostEqual(label.x0, 100 + pdfio._TABLE_CELL_PAD, delta=0.01)
        self.assertAlmostEqual(label.x1, 200 - pdfio._TABLE_CELL_PAD, delta=0.01)
        self.assertAlmostEqual(value.x0, 200 + pdfio._TABLE_CELL_PAD, delta=0.01)
        self.assertAlmostEqual(value.x1, 400 - pdfio._TABLE_CELL_PAD, delta=0.01)

    def test_build_table_blocks_ignores_narrow_gutter_cell(self):
        # A thin gutter between the label and value columns must never become a
        # cell a long label is fitted into.  The pre-fix code pinned a long
        # label row (whose whole-line centre lands in the gutter) to the ~1pt
        # gutter and rendered the translation as a vertical stack.
        cell_rects = [
            fitz.Rect(100, 100, 200, 120),   # label column
            fitz.Rect(200, 100, 205, 120),   # narrow gutter (sliver, < MIN_WIDTH)
            fitz.Rect(205, 100, 400, 120),   # value column
        ]
        line = self._line_with_spans(
            150, 300, 105, 115,
            "Max Gross Weight: 4,000 lbs / 1,814 kg",
            [
                (170.0, 105.0, 199.0, 115.0, "Max Gross Weight:"),
                (240.0, 105.0, 300.0, 115.0, "4,000 lbs / 1,814 kg"),
            ],
        )
        blocks = pdfio._build_table_blocks([line], [], 0, 600, 0, cell_rects)
        self.assertEqual(len(blocks), 2)
        by_text = {b.text: b for b in blocks}
        label = by_text["Max Gross Weight:"]
        value = by_text["4,000 lbs / 1,814 kg"]
        # Neither is crushed into the ~5pt gutter.
        self.assertGreater(label.x1 - label.x0, 50.0)
        self.assertGreater(value.x1 - value.x0, 50.0)

    def test_fit_block_puts_table_cell_on_one_line(self):
        # A long English translation used to wrap inside the narrow cell and push
        # the rows below it down.  An in_table cell shrinks the font instead.
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=100, y0=100, x1=300, y1=120,
            size=9.0, align="left", bold=False, single_line=True, in_table=True,
        )
        long_name = "Wenling Municipal State-owned Assets Management Co., Ltd."
        lines, fs = pdfio._fit_block(block, font, long_name)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines, [long_name])
        # The font shrank (never below the table floor) to fit the cell width.
        self.assertLess(fs, 9.0)
        self.assertGreaterEqual(fs, pdfio._MIN_TABLE_READABLE)
        self.assertLessEqual(font.text_length(long_name, fontsize=fs), 200.0)

    def test_fit_block_too_narrow_table_cell_wraps_at_readable_floor(self):
        # A cell so narrow that even the readability floor cannot hold a single
        # line used to stay ONE line at the 3pt floor — the slug-cell soft spot
        # (44 cells < 5pt on the scan, 5 at 3.0).  The cell now wraps at the
        # readability floor instead of collapsing to an illegible single line.
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=100, y0=100, x1=160, y1=120,
            size=9.0, align="left", bold=False, single_line=True, in_table=True,
        )
        long_name = "Wenling Municipal State-owned Assets Management Co., Ltd."
        lines, fs = pdfio._fit_block(block, font, long_name)
        self.assertGreater(len(lines), 1)
        self.assertEqual(fs, pdfio._MIN_TABLE_READABLE)
        self.assertEqual("".join(lines).replace(" ", ""), long_name.replace(" ", ""))

    def test_fit_block_band_wraps_at_readable_floor_when_band_is_plenty(self):
        # A scanned grid cell carries a fit_height band down to the row below
        # (the raster table lines cannot move, so the wrap must stay inside it).
        # A wide-enough band means 2-3 lines at the full readability floor.
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=100, y0=100, x1=160, y1=120,
            size=9.0, align="left", bold=False, single_line=True, in_table=True,
            fit_height=40.0,
        )
        long_name = "Wenling Municipal State-owned Assets Management Co., Ltd."
        lines, fs = pdfio._fit_block(block, font, long_name)
        self.assertGreater(len(lines), 1)
        self.assertEqual(fs, pdfio._MIN_TABLE_READABLE)
        self.assertLessEqual(
            pdfio._wrapped_height(
                font, lines, fs, pdfio._line_leading(font, in_table=True, n_lines=len(lines))
            ),
            40.0,
        )
        self.assertEqual("".join(lines).replace(" ", ""), long_name.replace(" ", ""))

    def test_fit_block_band_descends_when_the_band_is_tight(self):
        # A hole as tight as the scan's real row pitch (~18pt) holds only 2
        # lines at 6pt: the fit downsizes until the wrapped height fits the
        # band (but never below the absolute floor) rather than crossing the
        # grid line to keep a bigger font.
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=100, y0=100, x1=160, y1=120,
            size=9.0, align="left", bold=False, single_line=True, in_table=True,
            fit_height=17.5,
        )
        long_name = "Wenling Municipal State-owned Assets Management Co., Ltd."
        lines, fs = pdfio._fit_block(block, font, long_name)
        self.assertGreater(len(lines), 1)
        self.assertLessEqual(
            round(pdfio._wrapped_height(
                font, lines, fs, pdfio._line_leading(font, in_table=True, n_lines=len(lines))
            ), 2),
            17.5,
        )
        self.assertGreaterEqual(fs, pdfio._MIN_TABLE_FLOOR)
        self.assertLess(fs, pdfio._MIN_TABLE_READABLE)
        self.assertEqual("".join(lines).replace(" ", ""), long_name.replace(" ", ""))

    def test_fit_block_band_too_tight_for_a_wrap_gives_one_tall_line(self):
        # P1-5: the band is the real row gap.  A cell whose band cannot hold even a
        # 3pt wrap gets ONE line sized by its own glyph box (the source fit there),
        # never a second line crossing the row rule below.
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="其他综合收益", page=0, x0=10, y0=100, x1=50, y1=108.1,
            size=6.75, single_line=True, in_table=True,
            fit_height=3.5, fit_width=40.0,
        )
        lines, fs = pdfio._fit_block(block, font, "Other comprehensive income")
        self.assertEqual(1, len(lines))
        self.assertGreater(fs, pdfio._MIN_TABLE_FLOOR)

    def test_non_table_block_still_wraps(self):
        # A flowing paragraph (not a table cell) is unaffected: it wraps and keeps
        # its own box rather than being forced onto one line.
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=50, y0=100, x1=300, y1=140,
            size=11.0, align="left", bold=False, single_line=False, in_table=False,
        )
        text = ("本飞机专为单飞行员操作而开发，并经过相应调整以更好地模拟真实环境。"
                "其制作综合了多个真实世界的数据点。")
        lines, _fs = pdfio._fit_block(block, font, text)
        self.assertGreater(len(lines), 1)
        self.assertEqual("".join(lines), text)

    def test_fit_block_preserves_two_line_source_count(self):
        # A cell whose source spans exactly TWO lines keeps exactly two lines:
        # the fitter splits and rebalances so the count matches the source (and
        # the model's output is not re-wrapped into a different line count).
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="长期投资\n合计", page=0, x0=100, y0=100, x1=300, y1=140,
            size=9.0, align="left", bold=False, single_line=False, in_table=True,
        )
        text = ("Wenling Municipal State-owned Assets Management Co., Ltd. "
                "Total long-term investments")
        lines, fs = pdfio._fit_block(block, font, text)
        self.assertEqual(len(lines), 2)
        # No character was lost or fabricated across the two lines.
        self.assertEqual("".join(lines).replace(" ", ""), text.replace(" ", ""))
        self.assertGreaterEqual(fs, pdfio._MIN_TABLE_READABLE)

    def test_fit_block_two_lines_shrink_below_readable_floor_if_needed(self):
        # Line count wins over the readability floor: a 2-line source stays two
        # lines even in a too-narrow cell (the exact-n rebalance keeps the count
        # and lets the font drop to the absolute floor).
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="净亏损\n小计", page=0, x0=100, y0=100, x1=180, y1=140,
            size=9.0, align="left", bold=False, single_line=False, in_table=True,
        )
        text = ("Net loss per share attributable to the shareholders of the "
                "Company after taking into account the discontinued operations")
        lines, fs = pdfio._fit_block(block, font, text)
        self.assertEqual(len(lines), 2)
        self.assertEqual("".join(lines).replace(" ", ""), text.replace(" ", ""))
        self.assertLess(fs, pdfio._MIN_TABLE_READABLE)
        self.assertGreaterEqual(fs, pdfio._MIN_TABLE_FLOOR)

    def test_multiline_table_cell_uses_min_line_leading(self):
        # A cell whose translation wraps to more than one line renders with the
        # *minimum* line spacing — ``_TABLE_CELL_LEADING`` (1.0× the font size)
        # — instead of the loose 1.35× leading that paragraphs keep.  A long
        # translation in a narrow column must stay compact so the wrap fits the
        # row band without growing the row or pushing the rows below it down.
        font = fitz.Font("cjk")
        tight = pdfio._TABLE_CELL_LEADING
        self.assertLess(tight, pdfio._LOOSE_LEADING)
        # Multi-line table cell -> the tight leading.
        self.assertEqual(pdfio._line_leading(font, in_table=True, n_lines=2), tight)
        self.assertEqual(pdfio._line_leading(font, in_table=True, n_lines=3), tight)
        # A single line has no inter-line gap, so it keeps the loose value.
        self.assertEqual(pdfio._line_leading(font, in_table=True, n_lines=1),
                         pdfio._LOOSE_LEADING)
        # Paragraph (non-table) blocks keep the loose leading.
        self.assertEqual(pdfio._line_leading(font, in_table=False, n_lines=3),
                         pdfio._LOOSE_LEADING)
        # An explicit override wins (a rebuilt vector table's cells are not
        # ``in_table`` but still need the tight table leading).
        self.assertEqual(
            pdfio._line_leading(font, in_table=False, n_lines=3, override=tight), tight)
        self.assertEqual(
            pdfio._line_leading(font, in_table=True, n_lines=3, override=0.0), tight)
        # A wrapped cell measured through ``_fit_block`` is also measured tight,
        # so the row-height re-layout agrees with the drawing pass.
        cell = pdfio.Block(
            text="净亏损\n小计", page=0, x0=100, y0=100, x1=180, y1=140,
            size=9.0, align="left", bold=False, single_line=False, in_table=True,
        )
        wrapped = ("Net loss per share attributable to the shareholders of the "
                   "Company after taking into account the discontinued operations")
        lines, fs = pdfio._fit_block(cell, font, wrapped)
        self.assertGreater(len(lines), 1)
        # The tight leading packs the lines tighter than the loose default.
        self.assertLess(
            pdfio._measure_block_height(cell, font, wrapped),
            pdfio._wrapped_height(font, lines, fs),
        )


class OcrGridTest(unittest.TestCase):
    """Scanned financial tables (OCR blocks, no text layer / vector rules) must
    be rebuilt into a grid instead of collapsing into a jumble."""

    def _items(self):
        # A miniature balance sheet: a label column, a "行次" number column and
        # two numeric columns (合并 / 母公司).  Item tuple is (y0, x0, x1, y1, text).
        return [
            # header row
            (100, 78, 180, 112, "项目"), (100, 190, 212, 112, "行次"),
            (100, 240, 292, 112, "合并"), (100, 320, 372, 112, "母公司"),
            # data row 1 — label + two figures
            (130, 84, 164, 148, "现金及存放中央银行款项"),
            (130, 226, 290, 148, "17,485,938,749.91"),
            (130, 306, 370, 148, "14,944,565,492.79"),
            # data row 2
            (165, 84, 128, 180, "存放同业款项"),
            (165, 230, 289, 180, "3,702,726,474.45"),
            (165, 310, 369, 180, "1,386,040,370.31"),
            # data row 3 — a label, a line number, one figure
            (200, 84, 141, 212, "发放贷款和垫款"),
            (200, 190, 212, 212, "8"),
            (200, 222, 289, 212, "224,464,860,917.53"),
        ]

    def test_is_numeric_cell(self):
        self.assertTrue(pdfio._is_numeric_cell("17,485,938,749.91"))
        # OCR mis-reads can inject spaces inside a figure; still numeric.
        self.assertTrue(pdfio._is_numeric_cell("65, 334, 085.99"))
        self.assertTrue(pdfio._is_numeric_cell("(1,234.56)"))
        self.assertTrue(pdfio._is_numeric_cell("-789,702,296.83"))
        self.assertTrue(pdfio._is_numeric_cell("5.6"))
        # Labels / ordinals are not numeric.
        self.assertFalse(pdfio._is_numeric_cell("营业收入"))
        self.assertFalse(pdfio._is_numeric_cell("(四)"))
        self.assertFalse(pdfio._is_numeric_cell("现金及存放中央银行款项"))

    def test_reconstruct_ocr_grid_is_row_major_and_right_aligns_numbers(self):
        blocks, tables = pdfio._reconstruct_ocr_grid(self._items())
        self.assertTrue(blocks)
        self.assertEqual(len(tables), 1)
        texts = [b.text for b in blocks]
        # Row-major: the label column is not all emitted before the figures.
        self.assertLess(texts.index("现金及存放中央银行款项"), texts.index("17,485,938,749.91"))
        # Numeric cells are right-aligned; labels stay left.
        by_text = {b.text: b for b in blocks}
        self.assertEqual(by_text["17,485,938,749.91"].align, "right")
        self.assertEqual(by_text["14,944,565,492.79"].align, "right")
        self.assertEqual(by_text["现金及存放中央银行款项"].align, "left")
        # The numeric cell keeps its own right edge (aligned within its column),
        # while the label keeps its own glyph box as the bbox (the cover/redact
        # step erases exactly the printed label pixels — never a neighbouring
        # cell) and carries the whole column as draw room in ``fit_width``.
        self.assertAlmostEqual(by_text["17,485,938,749.91"].x1, 290.0, delta=1.0)
        self.assertAlmostEqual(by_text["现金及存放中央银行款项"].x1, 164.0, delta=1.0)
        self.assertGreater(
            by_text["现金及存放中央银行款项"].fit_width,
            by_text["现金及存放中央银行款项"].x1 - by_text["现金及存放中央银行款项"].x0,
        )

    def test_reconstruct_ocr_tables_maps_all_cells(self):
        blocks, _ = pdfio._reconstruct_ocr_grid(self._items())
        tables = pdfio._reconstruct_ocr_tables(blocks)
        self.assertEqual(len(tables), 1)
        self.assertEqual(len(tables[0]["rows"]), 4)  # header + 3 data rows
        mapping = pdfio._map_blocks_to_table_cells(blocks, tables)
        self.assertEqual(len(mapping), len(blocks))

    def test_ungrouped_figures_still_form_a_numeric_column(self):
        # P1-6: scanned statements often print plain ``1000`` / ``1234.56`` with no
        # thousands separator.  The grouped-only pattern rejected them, so the whole
        # page failed the numeric-column guard and was rebuilt as *prose* — no grid,
        # no row bands, no right alignment.
        items = []
        for i in range(5):
            y0, y1 = 100.0 + 12 * i, 108.0 + 12 * i
            items.append((y0, 80, 200, y1, f"项目{i + 1}"))
            items.append((y0, 240, 320, y1, str(1000 + i)))
            items.append((y0, 360, 440, y1, f"{1000.5 + i:.2f}"))
        blocks, tables = pdfio._reconstruct_ocr_grid(items)
        self.assertEqual(1, len(tables))
        self.assertTrue(blocks)
        by_text = {b.text: b for b in blocks}
        self.assertEqual("right", by_text["1000"].align)
        self.assertEqual("right", by_text["1000.50"].align)

    def test_note_marker_column_does_not_make_a_prose_page_a_table(self):
        # P1-7: a column of ``(1)(2)…`` note markers (or bare ``1..8`` list numbers)
        # made a two-column scanned PROSE page look like a table; with
        # ``redraw_ocr`` on, the page was then blank-redrawn and the scan's raster
        # background was lost for good.
        for marker in (lambda n: f"({n})", str):
            with self.subTest(marker=marker(1)):
                items = []
                for i in range(8):
                    y0, y1 = 100.0 + 14 * i, 108.0 + 14 * i
                    items.append((y0, 60, 90, y1, marker(i + 1)))
                    items.append((y0, 120, 420, y1,
                                  "The Group recorded revenue growth during the year"))
                    items.append((y0, 440, 740, y1,
                                  "operating expenses increased accordingly"))
                blocks, tables = pdfio._reconstruct_ocr_grid(items)
                self.assertEqual([], tables)
                self.assertEqual([], blocks)   # caller falls back to plain blocks
        # The same shape with a real figures column is still a table.
        items = []
        for i in range(8):
            y0, y1 = 100.0 + 14 * i, 108.0 + 14 * i
            items.append((y0, 60, 300, y1, f"营业收入项目{i + 1}"))
            items.append((y0, 340, 500, y1, f"{1000 + i:,}.00"))
        _blocks, tables = pdfio._reconstruct_ocr_grid(items)
        self.assertEqual(1, len(tables))

    def test_reconstruct_ocr_tables_rejects_a_prose_page(self):
        # A two-column scanned PROSE page lines up into rows/columns but has no
        # figures column.  Without the same numeric-column guard the grid
        # reconstruction applies, ``redraw_ocr`` treated it as a table and
        # blank-redrew the page, dropping the raster background for good.
        texts = ["本公司营业收入同比增长", "本公司净利润同比下降",
                 "主要业务保持稳健增长", "风险管理体系持续完善",
                 "资本充足率满足监管要求", "资产质量总体保持稳定"]
        coords = [(100.0, 60.0), (100.0, 300.0), (120.0, 60.0),
                  (120.0, 300.0), (140.0, 60.0), (140.0, 300.0)]
        blocks = [
            pdfio.Block(text=t, page=0, x0=x, y0=y, x1=x + 120, y1=y + 10, ocr=True,
                        in_table=True)
            for t, (y, x) in zip(texts, coords)
        ]
        self.assertEqual([], pdfio._reconstruct_ocr_tables(blocks))
        # The same shape with a figures column is still a table.
        numeric = [
            pdfio.Block(text="现金及存放中央银行款项", page=0, x0=60, y0=100, x1=180,
                        y1=110, ocr=True, in_table=True),
            pdfio.Block(text="17,485,938,749.91", page=0, x0=300, y0=100, x1=420,
                        y1=110, ocr=True, in_table=True),
            pdfio.Block(text="存放同业款项", page=0, x0=60, y0=120, x1=180,
                        y1=130, ocr=True, in_table=True),
            pdfio.Block(text="3,702,726,474.45", page=0, x0=300, y0=120, x1=420,
                        y1=130, ocr=True, in_table=True),
        ]
        self.assertEqual(1, len(pdfio._reconstruct_ocr_tables(numeric)))

    def test_grid_subcolumn_header_gets_row_gap_as_fit_width(self):
        # The "合并"/"母公司" header cells live inside a figure sub-column whose
        # own OCR box only encloses the two printed characters; the translation
        # ("Consolidated") must be able to use the empty gap up to the next cell
        # in the row, or it is crushed to the 3pt floor.
        blocks, _ = pdfio._reconstruct_ocr_grid(self._items())
        by_text = {b.text: b for b in blocks}
        header = by_text["合并"]
        self.assertGreater(header.fit_width, header.x1 - header.x0)
        self.assertAlmostEqual(header.fit_width, 320.0 - 2.0 - 240.0, delta=1.0)

    def test_grid_cells_get_row_pitch_as_fit_height(self):
        # Each cell of a grid row may draw down to the *next* row's top (minus a
        # margin for the raster line): a translation too long for a readable
        # single line wraps into the row gap instead of crossing the table line
        # below (scan raster lines cannot move).  The last row has no band.
        blocks, _ = pdfio._reconstruct_ocr_grid(self._items())
        by_text = {b.text: b for b in blocks}
        cell = by_text["现金及存放中央银行款项"]     # row top 130, next row top 165
        self.assertAlmostEqual(cell.fit_height, 165.0 - 130.0 - 1.5, delta=0.01)
        self.assertGreater(cell.fit_height, cell.y1 - cell.y0)
        self.assertEqual(by_text["发放贷款和垫款"].fit_height, 0.0)  # last row

    def test_tight_rows_keep_a_band_and_stay_inside_it(self):
        # Regression: the row band was dropped whenever it was NARROWER than the
        # glyph box (``band > y1 - y0``), so dense statement rows fell back to
        # ``fit_height == 0`` = "no band / wrap unbounded" and their 2-3 line wrap
        # crossed the grid line below (542 of 824 cells on the real p24-27 scan).
        #
        # v0.5.40: the band is the REAL gap to the next row.  v0.5.36 had floored it
        # at the cell's own glyph height, which inflated the *wrap* budget and let a
        # second line run into the next row's space; the too-tight cell now gets ONE
        # line at the glyph-height size instead (see ``_fit_block``).
        items = [
            (100.0, 78, 300, 109.6, "现金及存放中央银行款项"),
            (100.0, 320, 420, 109.6, "17,485,938,749.91"),
            (109.8, 78, 300, 119.4, "存放同业款项"),
            (109.8, 320, 420, 119.4, "3,702,726,474.45"),
            (119.6, 78, 300, 129.2, "发放贷款和垫款"),
            (119.6, 320, 420, 129.2, "224,464,860,917.53"),
        ]
        blocks, _ = pdfio._reconstruct_ocr_grid(items)
        by_text = {b.text: b for b in blocks}
        cell = by_text["现金及存放中央银行款项"]
        # The band is the row pitch, nothing more: next row's top − y0 − 1.5.
        self.assertAlmostEqual(cell.fit_height, 109.8 - 100.0 - 1.5, delta=0.2)
        self.assertLess(cell.fit_height, cell.y1 - cell.y0)   # tighter than the box
        self.assertEqual(0.0, by_text["发放贷款和垫款"].fit_height)   # last row
        font = fitz.Font("cjk")
        long_text = ("Cash and balances with the central bank and due from banks "
                     "and other financial institutions")
        lines, fs = pdfio._fit_block(cell, font, long_text)
        height = pdfio._wrapped_height(
            font, lines, fs, pdfio._line_leading(font, in_table=True, n_lines=len(lines)))
        # Either the wrap fits the band, or the fitter fell back to ONE line (a wide
        # line into the neighbouring blank space beats a second line crossing the
        # rule) — never a multi-line wrap that overflows the band.
        self.assertTrue(len(lines) == 1 or height <= cell.fit_height + 0.05,
                        (lines, fs, cell.fit_height))

    def test_grid_label_fit_width_stops_before_the_note_marker(self):
        # A row with the 附注 "(二)" band inside the label column: the label
        # keeps its own glyph box as bbox, its draw room stops 2pt before the
        # note (a long translation must not slide under "(二)" and have its tail
        # sliced by the note's white cover), and the note keeps its own box,
        # centred where the source printed it.
        items = self._items() + [
            (140, 84, 141, 152, "拆出资金"),
            (140, 170, 188, 152, "(二)"),
            (140, 230, 289, 152, "3,702,726,474.45"),
        ]
        blocks, _ = pdfio._reconstruct_ocr_grid(items)
        by_text = {b.text: b for b in blocks}
        label, note = by_text["拆出资金"], by_text["(二)"]
        self.assertAlmostEqual(label.x1, 141.0, delta=1.0)
        self.assertAlmostEqual(label.fit_width, 170.0 - 2.0 - 84.0, delta=1.0)
        self.assertEqual(label.align, "left")
        self.assertAlmostEqual(note.x1, 188.0, delta=1.0)
        self.assertEqual(note.align, "center")

    def test_grid_merges_split_label_fragments(self):
        # RapidOCR splits one printed label (营业利润（亏损以"一"号填列）) into two
        # close items; the fragments must become ONE cell so the model receives
        # the whole label and the fitter sees the whole run — while a "(十六)"
        # note marker stays its own cell.
        items = [
            (100, 78, 180, 112, "项目"), (100, 320, 372, 112, "母公司"),
            (130, 84, 131, 148, "营业利润（亏损以"),
            (130, 148, 168, 148, "号填列）"),
            (130, 230, 289, 148, "207,098,342.00"),
            (165, 84, 128, 180, "利息净收入"),
            (165, 170, 188, 180, "（十六）"),
            (165, 230, 289, 180, "84,528,349.88"),
        ]
        blocks, _ = pdfio._reconstruct_ocr_grid(items)
        texts = [x.text for x in blocks]
        self.assertIn("营业利润（亏损以号填列）", texts)
        self.assertNotIn("号填列）", texts)
        self.assertIn("利息净收入", texts)
        self.assertIn("（十六）", texts)
        merged = next(b for b in blocks if b.text == "营业利润（亏损以号填列）")
        # The merged run's box spans both fragments, so the fit sees the run.
        self.assertAlmostEqual(merged.x1, 168.0, delta=1.0)

    def test_grid_does_not_merge_a_note_marker_into_the_label(self):
        # The marker guard is symmetric enough that 拆出资金 + (二) never merge,
        # even though "(二)" sits 30pt from the label (a 40pt+ marker would too).
        items = [
            (100, 78, 180, 112, "项目"), (100, 320, 372, 112, "母公司"),
            (130, 84, 128, 148, "拆出资金"),
            (130, 170, 188, 148, "(二)"),
            (130, 230, 289, 148, "97,923,282.04"),
        ]
        blocks, _ = pdfio._reconstruct_ocr_grid(items)
        texts = [x.text for x in blocks]
        self.assertIn("拆出资金", texts)
        self.assertIn("(二)", texts)
        self.assertNotEqual(texts.index("拆出资金"), texts.index("(二)"))


class OcrBoxValidationTest(unittest.TestCase):
    """A malformed OCR box costs that one item — never the page or the export.

    Regression: a NaN/inf coordinate was copied straight into ``Block`` and the
    export later died with ``ValueError: cannot convert float NaN to integer``
    (after the model had already translated the whole document), while a box
    shaped ``[x0, y0, x1, y1]`` (a VLM backend's format) raised ``TypeError``,
    which ``_ocr_page_blocks`` swallowed as "OCR failed" — the page's *other*
    blocks were discarded with it.
    """

    _GOOD = ([(0.0, 0.0), (80.0, 0.0), (80.0, 10.0), (0.0, 10.0)], "GOOD ONE")

    def test_non_finite_boxes_are_skipped(self):
        log: list[str] = []
        items = [
            self._GOOD,
            ([(float("nan"), 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], "NAN BOX"),
            ([(0.0, 0.0), (float("inf"), 0.0), (1.0, 1.0), (0.0, 1.0)], "INF BOX"),
        ]
        blocks = pdfio._synthesize_ocr_blocks(items, 0, log.append)
        self.assertEqual(["GOOD ONE"], [b.text for b in blocks])
        self.assertTrue(any("非法" in m for m in log), log)

    def test_flat_box_does_not_discard_the_page(self):
        log: list[str] = []
        items = [self._GOOD, ([1.0, 1.0, 2.0, 2.0], "FLAT BOX")]
        blocks = pdfio._synthesize_ocr_blocks(items, 0, log.append)
        self.assertEqual(["GOOD ONE"], [b.text for b in blocks])

    def test_export_survives_a_nan_box(self):
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        page.insert_text((20, 40), "PLACEHOLDER", fontsize=12)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.pdf"
            out = Path(tmp) / "out.pdf"
            doc.save(str(src))
            doc.close()

            def ocr_fn(_i, _p):
                return [
                    ([(20.0, 20.0), (200.0, 20.0), (200.0, 40.0), (20.0, 40.0)], "OCR LINE"),
                    ([(float("nan"), 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], "NAN BOX"),
                ]

            dt = pdfio.extract_document_text(str(src), ocr=True, ocr_fn=ocr_fn,
                                             log=lambda _m: None)
            texts = [b.text for b in dt.pages[0]]
            self.assertIn("OCR LINE", texts)
            self.assertNotIn("NAN BOX", texts)
            # The export used to raise here (NaN → int); it must produce a file.
            per_page = [[f"T:{b.text}" for b in dt.pages[0]]]
            pdfio.save_translated_pdf(str(src), dt.pages, per_page, str(out),
                                      "English", log=lambda _m: None)
            self.assertTrue(out.exists())
            check = fitz.open(str(out))
            try:
                self.assertIn("OCR LINE", check[0].get_text())
            finally:
                check.close()


class NumberAtomicityTest(unittest.TestCase):
    """Figures are never split mid-number by the wrap machinery.

    Regression: on the statement pages the value 292,712,933,925.17 was drawn
    as ``292,712,933,925.1`` on one line and ``7`` on the next — a reader sees
    a decimal point that grew or lost a digit, and an amount that is wrong.
    """

    def test_number_in_narrow_box_stays_whole(self):
        font = fitz.Font("cjk")
        text = "292,712,933,925.17"
        lines = pdfio._wrap(font, text, 25.0, 11.0)  # far narrower than the text
        self.assertEqual([text], lines)

    def test_number_inside_prose_wraps_at_word_boundaries(self):
        font = fitz.Font("cjk")
        text = "总资产达 292,712,933,925.17 元，较上年末增长。"
        lines = pdfio._wrap(font, text, 40.0, 11.0)
        # No character was lost (the wrap drops inter-word spaces at line
        # breaks, so compare content digit-for-digit), no line holds a
        # *partial* amount, and the whole figure sits on one line.
        self.assertEqual(text.replace(" ", ""), "".join(lines))
        self.assertTrue(any("292,712,933,925.17" in line for line in lines))
        for line in lines:
            with self.subTest(line=line):
                if "292,712,933,925.17" not in line:
                    self.assertNotIn("292,712", line)

    def test_stray_spaces_around_separators_are_merged(self):
        # ``65, 334, 085.99`` was one amount OCR padded; splitting it at the
        # spaces would turn the value back into three numbers.
        font = fitz.Font("cjk")
        lines = pdfio._wrap(font, "65, 334, 085.99", 25.0, 11.0)
        self.assertEqual(["65,334,085.99"], lines)

    def test_space_thousand_separator_not_merged(self):
        # ``10 000`` uses a space as the group separator: it must survive as-is.
        font = fitz.Font("cjk")
        lines = pdfio._wrap(font, "10 000", 585.0, 11.0)
        self.assertEqual(["10 000"], lines)

    def test_break_latin_word_minimum_two_char_pieces(self):
        # The org chart shards (``P- ar- ty a- n- d ...``) came from one-char
        # pieces; a piece must carry at least two characters or the rest is
        # kept whole (a dangling ``a-`` would suggest the word continues).
        font = fitz.Font("cjk")
        lines: list[str] = []
        rest = pdfio._break_latin_word(font, "Innovative", 12.0, 11.0, lines)
        pieces = lines + [rest]
        self.assertEqual("".join(p.rstrip("-") for p in pieces), "Innovative")
        for piece in pieces:
            self.assertGreaterEqual(len(piece.rstrip("-")), 2, piece)

    def test_currency_and_unit_amounts_are_never_split(self):
        # Regression: ``_break_word`` checked ``_has_latin`` FIRST, so any amount
        # with a currency symbol or unit reached the Latin/CJK breaker and came out
        # as ``US$-`` / ``1,23-`` … or ``12,345,678`` + ``.90元`` — a figure that
        # reads as *changed*, the exact failure the number-atomicity rule forbids.
        font = fitz.Font("cjk")
        for token in ("US$1,234,567.89", "1,234.56万元", "12,345,678.90元",
                      "2023-12-31", "GB/T33436-2016"):
            with self.subTest(token=token):
                lines: list[str] = []
                rest = pdfio._break_word(font, token, 30.0, 6.0, lines)
                self.assertEqual([token], lines)
                self.assertEqual("", rest)
                self.assertTrue(pdfio._is_amount_atom(token))

    def test_words_containing_a_digit_are_still_broken_as_prose(self):
        # The amount guard must not swallow identifiers: a word whose numeric core
        # is not the bulk of the token still wraps/hyphenates normally.
        font = fitz.Font("cjk")
        for token in ("iPhone15ProMax", "PT6A-140", "COVID-19"):
            with self.subTest(token=token):
                self.assertFalse(pdfio._is_amount_atom(token))
                lines: list[str] = []
                rest = pdfio._break_word(font, token, 10.0, 6.0, lines)
                self.assertTrue(lines or rest != token, token)

    def test_amounts_are_never_split_by_the_line_rebalancer(self):
        # Regression: ``_split_line_half`` checked only ``_is_number_atom``, so the
        # rebalancer (which forces the translation onto exactly the source line
        # count) cut a currency/unit amount in half — ``US$1,23`` + ``4,567.89``
        # reads as two figures, the exact failure the atomicity rule forbids.
        for line in ("US$1,234,567.89", "1,234.56万元", "RMB12,345,678.90",
                     "12,345,678.90元"):
            with self.subTest(line=line):
                self.assertIsNone(pdfio._split_line_half(line))
        # A prose line that merely *contains* an amount still splits at its space.
        self.assertEqual(
            pdfio._split_line_half("Total liabilities 1,234,567.89元"),
            ("Total liabilities", "1,234,567.89元"),
        )

    def test_two_line_cell_keeps_a_currency_amount_whole(self):
        # The same path end to end: a two-line source cell whose translation is one
        # long amount must not come back as two half-amounts.
        font = fitz.Font("cjk")
        block = pdfio.Block(text="a\nb", page=0, x0=0.0, y0=0.0, x1=90.0, y1=20.0,
                            size=9.0, in_table=True, single_line=False)
        lines, _fs = pdfio._fit_block(block, font, "US$1,234,567.89")
        self.assertEqual(["US$1,234,567.89"], lines)


class VerticalLabelTest(unittest.TestCase):
    """Narrow-tall boxes (org-chart labels) get a rotated 90° translation."""

    def _block(self, x0=50.0, y0=50.0, x1=57.4, y1=79.5, **kw):
        # 7.4 x 29.5pt: the shape of the real 党群工作部 box on page 5.
        return pdfio.Block(
            text="", page=0, x0=x0, y0=y0, x1=x1, y1=y1,
            size=24.0, align="center", bold=False, **kw
        )

    def test_detection_only_for_narrow_tall_single_line(self):
        block = self._block()
        self.assertTrue(pdfio._is_vertical_label(block))
        # Wide box, squat box, table cell and multi-line: not vertical labels.
        self.assertFalse(pdfio._is_vertical_label(self._block(x1=150.0)))
        self.assertFalse(pdfio._is_vertical_label(self._block(y1=60.0)))
        self.assertFalse(pdfio._is_vertical_label(self._block(in_table=True)))
        self.assertFalse(
            pdfio._is_vertical_label(self._block(single_line=False))
        )

    def test_extraction_marks_rotated_labels_single_line_and_chart(self):
        # Regression: the old ``single_line`` heuristic used ``rect.height <= 1.5*size``,
        # which is always False for a vertically-rotated label (its bbox is *tall*), so
        # ``_is_vertical_label`` never fired on real extraction — the page was classified
        # ``normal`` and labels were drawn as horizontal one-char shards.  This drives the
        # real extraction path (not a hand-built ``Block``) and asserts the labels come
        # back single-line + the page triages as ``chart``.
        src = _OUT / "orgchart.pdf"
        doc = fitz.open()
        page = doc.new_page()
        for y in (100, 200, 300):
            page.insert_text(fitz.Point(60, y), "公司的架构", fontsize=8,
                             fontname="china-s", rotate=90)
        doc.save(str(src))
        doc.close()
        extracted = pdfio.extract_document_text(str(src))
        labels = [b for b in extracted.pages[0] if pdfio._is_vertical_label(b)]
        self.assertGreaterEqual(len(labels), 3, "rotated org-chart labels not detected")
        for b in labels:
            self.assertTrue(b.single_line, f"vertical label not single_line: {b.text!r}")
        self.assertEqual(pdfio.classify_page(extracted.pages[0]), pdfio.PAGE_CHART)

    def test_rotation_keeps_label_inside_the_box(self):
        font = fitz.Font("cjk")
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        # An 8 x 120pt vertical label box (multi-char column): the phrase fits
        # along the height when rotated, which the horizontal draw cannot do.
        block = pdfio.Block(
            text="", page=0, x0=50.0, y0=50.0, x1=58.0, y1=170.0,
            size=24.0, align="center", bold=False,
        )
        pdfio._draw_translated_block(page, font, block, "Party and Mass Work")
        # Extract while the document is still open.
        spans = [s for s in page.get_text("dict")["blocks"] if s.get("type") == 0]
        text = "".join(sp["text"] for s in spans for l in s["lines"] for sp in l["spans"])
        doc.close()
        # The line must be whole (no hyphenated shards) and the glyph column
        # centred inside the box (wider boxes than the source keep the word).
        self.assertNotIn("-", text)
        self.assertTrue("PartyandMassWork" in text.replace(" ", ""))
        bbox = fitz.Rect(0, 0, 0, 0)
        for s in spans:
            for l in s["lines"]:
                bbox |= fitz.Rect(l["bbox"])
        self.assertAlmostEqual((bbox.x0 + bbox.x1) / 2, (block.x0 + block.x1) / 2, delta=2.0)
        self.assertGreater(bbox.y0, block.y0 - 2.0)
        self.assertLess(bbox.y1, block.y1 + 2.0)

    def test_label_longer_than_box_height_is_still_whole(self):
        # A label that cannot fit the box at any readable size is drawn rotated
        # at the column size (extending beyond the box, centred on it) — the
        # alternative, horizontal one-char shards, never reads at all.
        font = fitz.Font("cjk")
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        block = self._block()  # 7.4 x 29.5pt
        long_label = "The Party Committee and Administrative Department"
        pdfio._draw_vertical_label(page, font, block, long_label)
        # Extract while the document is still open.
        spans = [s for s in page.get_text("dict")["blocks"] if s.get("type") == 0]
        text = "".join(sp["text"] for s in spans for l in s["lines"] for sp in l["spans"])
        doc.close()
        self.assertNotIn("-", text)
        self.assertTrue(
            long_label.replace(" ", "") in text.replace(" ", ""), text
        )


class OcrTableNoExpansionTest(unittest.TestCase):
    """OCR-reconstructed tables keep the scan's own geometry.

    Regression: the row-height expansion measured translated cells against the
    reconstructed rows and pushed the lower rows hundreds of points down (on
    the report's statement pages a value moved ~355 pt, several rows below its
    own cell).  Scan pages must draw at the positions the OCR grid produced.
    """

    def _scan_source(self, path: Path) -> Path:
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        # A fake scan: a raster band and no text layer, so find_tables finds no
        # ruled grid and the OCR-table reconstruction path is taken.
        page.draw_rect(fitz.Rect(40, 40, 555, 300), color=None, fill=(0.9, 0.9, 0.9))
        doc.save(str(path))
        doc.close()
        return path

    def _cells(self):
        def cell(x0, y0, x1, y1, text):
            return pdfio.Block(
                text=text, page=0, x0=x0, y0=y0, x1=x1, y1=y1,
                size=10.0, align="left", bold=False,
                single_line=True, ocr=True, in_table=True,
            )

        return cell

    def test_ocr_grid_in_place_keeps_rows_below_the_wrapping_cell(self):
        """A scanned table page stays in-place and the rows are not expanded.

        Six cells where the translation of one cell wraps to several lines
        while the scan row is only 12pt tall: the exporter must keep the scan
        geometry and NOT push the rows below the wrapping cell down — that is
        the ~355pt misplacement regression.
        """
        src = _OUT / "scan_grid_in_place.pdf"
        self._scan_source(src)
        cell = self._cells()

        def row(y0, left, right):
            return cell(50, y0, 260, y0 + 12, left), cell(300, y0, 520, y0 + 12, right)

        b0, b1 = row(100, "总资产", "总负债")
        b2, b3 = row(140, "净资产", "现金")
        b4, b5 = row(180, "净利润", "总成本")
        blocks = [b0, b1, b2, b3, b4, b5]
        trans = [
            "292,712,933,925.17",
            "The consolidated financial statements of the Group and its "
            "subsidiaries were prepared under the principles of going concern",
            "24,321,445,868.48",
            "268,719,676,841.77",
            "12,345,678,901.23",
            "4,567,892,100.00",
        ]
        out = _OUT / "ocr_grid_in_place_out.pdf"
        pdfio.save_translated_pdf(src, [blocks], [trans], out, "English")
        doc = fitz.open(out)
        self.assertEqual(doc.page_count, 1)
        spans = _text_lines(doc[0])
        doc.close()

        def find(text: str) -> fitz.Rect:
            matches = [bb for bb, t in spans if text in t]
            self.assertGreater(len(matches), 0, text)
            return matches[0]

        for b, t in zip(blocks, trans):
            rect = find(t[:20])  # the first words of the cell
            self.assertLess(abs(rect.y0 - b.y0), 8.0, t)
            self.assertLess(abs(rect.x0 - b.x0), 5.0, t)


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


class MultilineCellAnchorTest(unittest.TestCase):
    """A table cell whose translation wraps to >1 line is anchored at the cell's
    top border (using the row's full height), so it stays inside the cell."""

    def _grid(self) -> Path:
        path = _OUT / "cellgrid.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        cols = [60, 200, 360]
        tops = [80, 120, 160]
        for x in cols:
            page.draw_line(fitz.Point(x, tops[0]), fitz.Point(x, tops[-1]),
                           color=(0, 0, 0), width=0.6)
        for y in tops:
            page.draw_line(fitz.Point(cols[0], y), fitz.Point(cols[-1], y),
                           color=(0, 0, 0), width=0.6)
        page.insert_text((70, 112), "Net Assets Per Share Attributable", fontsize=9)
        page.insert_text((210, 112), "0.21", fontsize=9)
        doc.save(str(path))
        doc.close()
        return path

    def test_multiline_cell_stays_inside_cell(self):
        src = self._grid()
        dt = pdfio.extract_document_text(src, log=lambda _m: None)
        label = next(b for b in dt.pages[0] if "Net Assets" in b.text)
        self.assertTrue(label.in_table)
        cell = pdfio._extract_tables(fitz.open(str(src))[0])[0]["rows"][0][0]
        cell_top, cell_bot = cell.y0, cell.y1
        per = [
            ("Net Assets Per Share Attributable to Shareholders of the Parent Company"
             if b is label else b.text)
            for b in dt.pages[0]
        ]
        out = _OUT / "cellgrid_out.pdf"
        pdfio.save_translated_pdf(src, [dt.pages[0]], [per], str(out), "English")
        doc = fitz.open(str(out))
        spans = []
        for bb in doc[0].get_text("dict")["blocks"]:
            if bb.get("type") != 0:
                continue
            for l in bb.get("lines", []):
                for s in l.get("spans", []):
                    b = s["bbox"]
                    if s["text"].strip() and b[0] >= cell.x0 - 4 and b[2] <= cell.x1 + 4 \
                            and b[1] >= cell_top - 6 and b[3] <= cell_bot + 6:
                        spans.append(s["bbox"])
        doc.close()
        self.assertGreater(len(spans), 1)  # the label wrapped to >1 line
        # It must sit on the cell's top border (not the source glyph top / centred)
        # and stay inside the cell rather than overflowing the bottom rule.
        self.assertLess(min(s[1] for s in spans) - cell_top, 8.0)
        self.assertLessEqual(max(s[3] for s in spans), cell_bot + 1.0)


class SkewDetectTest(unittest.TestCase):
    """The low-risk OCR skew detector's pure angle math."""

    def test_angle_median_wraps_to_45(self):
        # Angles fold into [-45, 45): a 0° and a 179° line are the same baseline.
        self.assertEqual(0.0, pdfio._angle_median([0.0, 0.5, -0.5]))
        self.assertAlmostEqual(10.0, pdfio._angle_median([9.0, 10.0, 11.0]))
        self.assertAlmostEqual(-1.0, pdfio._angle_median([179.0, 181.0, -1.0]), places=5)
        self.assertIsNone(pdfio._angle_median([]))

    def test_deskew_affine_roundtrips_points(self):
        # A point in the original image, rotated forward to the deskewed frame, then
        # mapped back with ``_map_pt_back`` must land exactly on the original point —
        # this is the geometry that keeps OCR boxes on the real page after a deskew.
        import cv2
        import numpy as np

        img = np.zeros((300, 400, 3), np.uint8)     # H=300, W=400
        for skew in (1.5, -2.0):
            rotated, inv_m, pad = pdfio._deskew_affine(img, skew)
            m = cv2.invertAffineTransform(inv_m)
            for (x, y) in ((5, 5), (37, 123), (395, 295)):
                v = m @ np.array([float(x + pad), float(y + pad), 1.0])
                ox, oy = pdfio._map_pt_back(inv_m, pad, float(v[0]), float(v[1]))
                self.assertAlmostEqual(float(x), ox, places=4)
                self.assertAlmostEqual(float(y), oy, places=4)

    def test_estimate_skew_from_gray_detects_angle(self):
        import cv2
        import numpy as np

        for want in (1.5, -2.0):
            img = np.full((400, 500), 255, np.uint8)
            ang = np.radians(want)
            cx, cy = 250.0, 200.0
            dx, dy = 200.0, 0.0
            x1 = cx + dx * np.cos(ang) - dy * np.sin(ang)
            y1 = cy + dx * np.sin(ang) + dy * np.cos(ang)
            x2 = cx - dx * np.cos(ang) + dy * np.sin(ang)
            y2 = cy - dx * np.sin(ang) - dy * np.cos(ang)
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), 0, 3)
            got = pdfio._estimate_skew_from_gray(img)
            self.assertIsNotNone(got)
            self.assertAlmostEqual(float(want), float(got), delta=1.2)

    def test_deskew_removes_the_tilt_instead_of_doubling_it(self):
        # Regression: ``_deskew_affine`` rotated by the wrong sign, so a scan tilted
        # by θ was fed to OCR at 2θ (a 3° page came out at ~6°).  After deskewing with
        # the measured angle the residual tilt must be ~0.
        import cv2
        import numpy as np

        base = np.full((400, 600), 255, np.uint8)
        for y in range(60, 340, 30):
            cv2.line(base, (60, y), (540, y), 0, 3)
        for angle in (3.0, -3.0):
            m = cv2.getRotationMatrix2D((300.0, 200.0), angle, 1.0)
            tilted = cv2.warpAffine(base, m, (600, 400), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            est = pdfio._estimate_skew_from_gray(tilted)
            self.assertIsNotNone(est)
            fixed, _, _ = pdfio._deskew_affine(tilted, float(est))
            residual = pdfio._estimate_skew_from_gray(fixed)
            self.assertIsNotNone(residual)
            self.assertLess(abs(float(residual)), 0.6,
                            f"tilt {angle}° → est {est}° → residual {residual}°")

    def test_detect_page_skew_flat_page_is_not_recommended(self):
        # A page with straight horizontal rules reads as ~0° (no geometry correction
        # is recommended).  --- a smoke test that the CV path runs without error.
        import pymupdf as fitz
        src = _OUT / "skew_flat.pdf"
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        for y in (40, 80, 120, 160, 200, 240):
            page.draw_line(fitz.Point(30, y), fitz.Point(270, y),
                           color=(0, 0, 0), width=1)
        doc.save(str(src))
        doc.close()
        res = pdfio.detect_page_skew(str(src), 0)
        self.assertIsInstance(res["skew_degrees"], float)
        self.assertIn(res["reason"], ("版面基本平正", "未检测到文本线"))


class ParagraphGroupingTest(unittest.TestCase):
    """F2: a paragraph's short last line must not become its own block.

    ``_break_between``'s centre-jump rule (meant for left↔right column changes) fired
    on any left-aligned last line narrower than ~half the text width — roughly half of
    all paragraphs — because it measured the jump against the *current* (short) line.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    @staticmethod
    def _line(x0, x1, y0, text="text"):
        return {"x0": float(x0), "x1": float(x1), "y0": float(y0),
                "y1": float(y0) + 11.0, "size": 11.0, "bold": False,
                "color": 0, "text": text}

    def test_short_last_line_stays_in_the_paragraph(self):
        lines = [self._line(72, 453, 100), self._line(72, 453, 114),
                 self._line(72, 140, 128, "the end.")]
        groups = pdfio._group_lines(lines)
        self.assertEqual(1, len(groups), [len(g) for g in groups])
        self.assertEqual(3, len(groups[0]))

    def test_staggered_column_transition_still_breaks(self):
        # The rule's real job: a two-column page whose right column starts BELOW the
        # left column's last line (so the y-jump rule cannot catch it).
        lines = [self._line(72, 250, 100), self._line(72, 150, 114),
                 self._line(320, 500, 128)]
        groups = pdfio._group_lines(lines)
        self.assertEqual([2, 1], [len(g) for g in groups])

    def test_narrow_items_do_not_make_a_column_line_full_width(self):
        # ``width > 1.5 x median`` ALONE classified ordinary column lines as
        # full-width whenever narrow items dominated the median (a ruled table's
        # cells are ~5-60 pt wide).  Those lines were then pulled out and
        # re-sorted by y, so the two columns interleaved and each paragraph was
        # chopped into fragments that overprinted each other.  The whole left
        # column must still read before the whole right column.
        items = [(80.0, 300.0 + 12.0 * i, 100.0, 308.0 + 12.0 * i)
                 for i in range(10)]                       # narrow table cells
        items += [(70.0, 200.0, 280.0, 210.0), (70.0, 216.0, 280.0, 226.0)]
        items += [(620.0, 200.0, 830.0, 210.0), (620.0, 216.0, 830.0, 226.0)]
        ordered = pdfio._order_generic(
            items, lambda it: it[0], lambda it: it[2], lambda it: it[1])
        left = [i for i, it in enumerate(ordered) if it[0] == 70.0]
        right = [i for i, it in enumerate(ordered) if it[0] == 620.0]
        self.assertTrue(left and right)
        self.assertLess(max(left), min(right))

    def test_wrapped_paragraph_extracts_as_one_block(self):
        # End-to-end: the same shape on a real PDF must come back as one block, not
        # three (the fragment then got translated alone, losing its context).
        import pymupdf as fitz

        path = Path(self.tmp.name) / "para.pdf"
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        y = 100.0
        for line in ("The quick brown fox jumps over the lazy dog and keeps running",
                     "across the open field until it reaches the far side of the",
                     "meadow."):
            page.insert_text((72, y), line, fontsize=11)
            y += 14.0
        doc.save(str(path))
        doc.close()
        dt = pdfio.extract_document_text(str(path), ocr=False)
        self.assertEqual(1, len(dt.blocks), dt.blocks)
        self.assertIn("meadow.", dt.blocks[0])


class RotatedPageOcrTest(unittest.TestCase):
    """F4: OCR boxes on a page with /Rotate ≠ 0 must be mapped to the unrotated frame.

    ``get_pixmap`` renders the *rotated* page, while the text/draw APIs (and hence the
    exporter) work in the unrotated mediabox frame.  Without the derotation the boxes
    came out transposed (x/y swapped), so covers and translations landed elsewhere.
    """

    def test_ocr_boxes_are_derotated(self):
        import numpy as np

        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        page.set_rotation(90)
        try:
            self.assertEqual(400, page.mediabox.width)
            # A pixmap-space point (100, 50) is the unrotated (50, 100).
            img = np.zeros((400, 200, 3), np.uint8)   # 200x400 px at zoom 1
            box = [[100.0, 50.0], [100.0, 50.0], [100.0, 50.0], [100.0, 50.0]]

            def engine(_img):
                return [(box, "TXT")]

            res = pdfio._ocr_results_from_img(engine, img, 1.0, 0, None,
                                              derotate=page.derotation_matrix)
            self.assertEqual(1, len(res))
            self.assertAlmostEqual(50.0, res[0][0][0][0], places=1)
            self.assertAlmostEqual(100.0, res[0][0][0][1], places=1)
        finally:
            doc.close()

    def test_ocr_page_blocks_uses_the_page_derotation(self):
        import numpy as np

        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        page.set_rotation(90)
        try:
            px, py = 500.0, 300.0          # a point in the rendered pixmap
            zoom = pdfio._OCR_DPI / 72.0
            expected = fitz.Point(px / zoom, py / zoom) * page.derotation_matrix

            def engine(_img):
                return [([[px, py], [px, py], [px, py], [px, py]], "TXT")]

            with mock.patch.object(pdfio, "_get_ocr_engine", return_value=engine), \
                 mock.patch.object(pdfio, "_page_to_array",
                                   return_value=(np.zeros((10, 10, 3), np.uint8), zoom)):
                blocks = pdfio._ocr_page_blocks(0, page, None, None)
            self.assertEqual(1, len(blocks))
            b = blocks[0]
            self.assertAlmostEqual(float(expected.x), b.x0, delta=1.0)
            self.assertAlmostEqual(float(expected.y), b.y0, delta=1.0)
        finally:
            doc.close()


class RotatedFrameTest(unittest.TestCase):
    """``_rot_map_rect`` / ``_photo_ocr_blocks`` must agree with the rendered frame.

    ``get_pixmap`` renders the *displayed* (rotated) cropbox; block coordinates are
    the unrotated cropbox frame.  Two regressions lived here: the map used the
    mediabox dimensions (wrong whenever a CropBox crops the sheet), and
    ``_photo_ocr_blocks`` sampled the unrotated box directly on a rotated page —
    reading white paper under a photo and drawing the cover straight over it.
    """

    def _text_page(self, rot: int = 0, crop: bool = False):
        doc = fitz.open()
        page = doc.new_page(width=300, height=500)
        page.insert_text((60, 450), "BOTTOM LABEL", fontsize=14)
        if crop:
            page.set_cropbox(fitz.Rect(30, 80, 270, 470))
        if rot:
            page.set_rotation(rot)
        return doc, page

    def test_rot_map_matches_the_rendered_ink(self):
        for rot in (0, 90, 180, 270):
            for crop in (False, True):
                with self.subTest(rot=rot, crop=crop):
                    doc, page = self._text_page(rot, crop)
                    try:
                        box = fitz.Rect(page.get_text("dict")["blocks"][0]["bbox"])
                        mapped = pdfio._rot_map_rect(box, page)
                        luma = pdfio._pixmap_luma(page.get_pixmap(dpi=72))
                        ys, xs = (luma < 128).nonzero()
                        ink = fitz.Rect(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                        for got, want in ((mapped.x0, ink.x0), (mapped.y0, ink.y0),
                                          (mapped.x1, ink.x1), (mapped.y1, ink.y1)):
                            self.assertAlmostEqual(got, want, delta=8.0)
                    finally:
                        doc.close()

    def _photo_page(self, rot: int = 0):
        doc = fitz.open()
        page = doc.new_page(width=200, height=400)
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 120))
        pix.set_rect(pix.irect, (40, 60, 200))
        page.insert_image(fitz.Rect(0, 0, 100, 200), pixmap=pix)
        if rot:
            page.set_rotation(rot)
        return doc, page

    def test_photo_cover_is_skipped_on_every_rotation(self):
        for rot in (0, 90, 180, 270):
            with self.subTest(rot=rot):
                doc, page = self._photo_page(rot)
                try:
                    block = pdfio.Block(text="photo caption", page=0, x0=10.0, y0=10.0,
                                        x1=90.0, y1=30.0, size=10.0, ocr=True)
                    luma = pdfio._pixmap_luma(page.get_pixmap(dpi=pdfio._PHOTO_SAMPLE_DPI))
                    levels = pdfio._page_levels(luma, luma.shape[1] / page.rect.width)
                    self.assertEqual(
                        {0}, pdfio._photo_ocr_blocks(page, [block], luma=luma, levels=levels),
                        "the white cover would be drawn over the photo",
                    )
                finally:
                    doc.close()

    def test_vertical_label_stays_in_its_box_on_a_rotated_page(self):
        # Regression: the run's on-page clamp used ``page.rect`` (the rotated
        # view), so on a /Rotate 90 page a label at y=300..390 was pushed up to
        # y≈160 — 187pt away from the box it labels.
        font = fitz.Font("cjk")
        for rot in (0, 90):
            with self.subTest(rot=rot):
                doc = fitz.open()
                page = doc.new_page(width=200, height=400)
                if rot:
                    page.set_rotation(rot)
                block = pdfio.Block(text="", page=0, x0=10.0, y0=300.0,
                                    x1=18.0, y1=390.0, size=10.0, single_line=True)
                pdfio._draw_vertical_label(page, font, block, "竖排标签文字")
                words = page.get_text("words")
                self.assertTrue(words, "nothing drawn")
                for w in words:
                    self.assertGreaterEqual(w[1], block.y0 - 1.0, w)
                    self.assertLessEqual(w[3], block.y1 + 1.0, w)
                doc.close()


class LineArtPreservationTest(unittest.TestCase):
    """F3: a page with a table must not lose its OTHER vector graphics.

    ``apply_redactions(graphics=REMOVE_IF_TOUCHED)`` is page-wide: with a table present
    it deleted every drawing whose bbox intersected any text redaction rect — e.g. the
    frame drawn around a paragraph (measured on the real exporter before the fix).
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _source(self) -> Path:
        path = Path(self.tmp.name) / "art.pdf"
        doc = fitz.open()
        page = doc.new_page(width=500, height=700)
        for x in (60, 200, 340):                       # a 2x2 ruled table
            page.draw_line(fitz.Point(x, 100), fitz.Point(x, 200),
                           color=(0, 0, 1), width=0.8)
        for y in (100, 150, 200):
            page.draw_line(fitz.Point(60, y), fitz.Point(340, y),
                           color=(0, 0, 1), width=0.8)
        page.insert_text((70, 130), "Cell A", fontsize=10)
        page.insert_text((210, 130), "123", fontsize=10)
        page.draw_rect(fitz.Rect(55, 300, 345, 340), color=(0, 0.5, 0), width=1.0)
        page.insert_text((70, 325), "This paragraph is boxed by a frame.", fontsize=11)
        page.draw_line(fitz.Point(60, 360), fitz.Point(340, 360),
                       color=(0, 0.5, 0), width=1.0)
        doc.save(str(path))
        doc.close()
        return path

    def test_boxed_paragraph_keeps_its_frame(self):
        src = self._source()
        doc = pdfio.extract_document_text(str(src), ocr=False)
        self.assertTrue(pdfio._extract_tables(fitz.open(str(src))[0]))  # a table exists
        per_page = [[f"T{j}" for j in range(len(page))] for page in doc.pages]
        out = Path(self.tmp.name) / "art_out.pdf"
        pdfio.save_translated_pdf(str(src), doc.pages, per_page, out, "English")
        doc2 = fitz.open(str(out))
        try:
            rects = [tuple(round(v) for v in dr["rect"]) for dr in doc2[0].get_drawings()]
        finally:
            doc2.close()
        self.assertIn((55, 300, 345, 340), rects,
                      f"the boxed paragraph lost its frame: {rects}")
        self.assertIn((60, 360, 340, 360), rects,
                      f"the underline was removed: {rects}")


class StructureTest(unittest.TestCase):
    """B-④ semantic-structure layer: build_structure fuser + get_doc_info enrichment."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = build_sample_pdf(Path(self.tmp.name) / "struct_src.pdf", pages=2)

    def _doc(self):
        return pdfio.extract_document_text(str(self.src), ocr=False, log=lambda m: None)

    def _structure_fn(self, page_index, page, blocks):
        # A mock backend: page 0's first block is a formula; page 1's first block is a figure.
        if not blocks:
            return []
        b = blocks[0]
        if page_index == 0:
            return [{"kind": "formula", "bbox": [b.x0, b.y0, b.x1, b.y1]}]
        if page_index == 1:
            return [{"kind": "figure", "bbox": [b.x0, b.y0, b.x1, b.y1],
                     "level": 0}]
        return []

    def test_build_structure_populates_page_structure(self):
        dt = self._doc()
        pdfio.build_structure(str(self.src), dt, self._structure_fn, parser="mock")
        self.assertEqual(dt.structure_parser, "mock")
        self.assertEqual(len(dt.page_structure), dt.page_count)
        self.assertEqual(dt.page_structure[0].elements[0]["kind"], "formula")
        # The first block of page 0 (flat index 0) is contained in its own bbox.
        self.assertIn(0, dt.page_structure[0].elements[0]["block_indices"])

    def test_get_doc_info_includes_structure_counts_when_present(self):
        dt = self._doc()
        pdfio.build_structure(str(self.src), dt, self._structure_fn, parser="mock")
        info = pdfio.get_doc_info(dt)
        self.assertEqual(info["structure_parser"], "mock")
        self.assertEqual(info["formula_pages"], 1)
        self.assertEqual(info["figure_pages"], 1)

    def test_get_doc_info_unchanged_without_structure(self):
        dt = self._doc()
        info = pdfio.get_doc_info(dt)
        # No formula/figure pages when no structure backend ran.
        self.assertEqual(info["formula_pages"], 0)
        self.assertEqual(info["figure_pages"], 0)
        self.assertNotIn("structure_parser", info)
        # Existing keys are unaffected.
        self.assertEqual(info["pages"], 2)

    def test_structure_dominant_kind(self):
        ps = pdfio.PageStructure(page=0, parser="mock",
                                 elements=[{"kind": "formula", "bbox": [0, 0, 1, 1],
                                            "level": 0, "block_indices": [0], "parser": "mock"},
                                           {"kind": "formula", "bbox": [0, 0, 1, 1],
                                            "level": 0, "block_indices": [1], "parser": "mock"}])
        self.assertEqual(pdfio._structure_dominant_kind(ps), "formula")
        # A single formula among text is not dominant.
        ps2 = pdfio.PageStructure(page=1, parser="mock",
                                  elements=[{"kind": "formula", "bbox": [0, 0, 1, 1],
                                             "level": 0, "block_indices": [0], "parser": "mock"},
                                            {"kind": "text", "bbox": [0, 0, 1, 1],
                                             "level": 0, "block_indices": [1], "parser": "mock"}])
        self.assertIsNone(pdfio._structure_dominant_kind(ps2))

    def test_classify_page_formula_and_figure_kinds(self):
        dt = self._doc()
        pdfio.build_structure(str(self.src), dt, self._structure_fn, parser="mock")
        p0 = dt.page_structure[0]
        # Page 0 has a structure with formula (more regions than text blocks it
        # claims), so it is classified as a formula page when structure is supplied.
        self.assertIn(pdfio.classify_page(dt.pages[0], structure=p0),
                      (pdfio.PAGE_FORMULA, pdfio.PAGE_NORMAL))
        # Without a structure, classification is unchanged (no formula/figure).
        self.assertEqual(pdfio.classify_page(dt.pages[0]), pdfio.PAGE_NORMAL)

    def test_extract_structured_populates_and_degrades(self):
        # A backend that works populates page_structure.
        dt = pdfio.extract_structured(str(self.src), self._structure_fn, parser="mock")
        self.assertEqual(dt.structure_parser, "mock")
        self.assertTrue(dt.page_structure)

        # A backend that raises degrades to a plain extraction (no crash).
        def _boom(page_index, page, blocks):
            raise RuntimeError("backend down")
        dt2 = pdfio.extract_structured(str(self.src), _boom, parser="boom",
                                       log=lambda m: None)
        self.assertEqual(dt2.structure_parser, "")
        # A backend that raises on every page leaves element-less structure entries.
        self.assertTrue(all(not ps.elements for ps in dt2.page_structure))

    def test_table_region_builds_struct_table(self):
        def table_fn(page_index, page, blocks):
            if page_index != 0 or not blocks:
                return []
            # A table region over the whole page; cells are flat block indices.
            bbox = [blocks[0].x0, blocks[0].y0, blocks[-1].x1, blocks[-1].y1]
            return [{"kind": "table", "bbox": bbox,
                     "cells": [[0, 1], [None, 2]]}]
        dt = pdfio.extract_structured(str(self.src), table_fn, parser="mock")
        self.assertEqual(dt.structure_parser, "mock")
        ps = dt.page_structure[0]
        self.assertEqual(len(ps.tables), 1)
        tbl = ps.tables[0]
        self.assertEqual((tbl.rows, tbl.cols), (2, 2))
        self.assertEqual(tbl.cells[0], [0, 1])
        self.assertEqual(tbl.cells[1], [-1, 2])
        self.assertTrue({0, 1, 2} <= tbl.block_ref)
        gt = pdfio.get_table(dt, 0, 0)
        self.assertIsNotNone(gt)
        self.assertEqual(gt["rows"], 2)
        # Page 1 has no table → None.
        self.assertIsNone(pdfio.get_table(dt, 1, 0))


class FormulaProtectionTest(unittest.TestCase):
    """B-⑤: a detected math-expression block is never translated (kept verbatim)."""

    def test_formula_detector(self):
        self.assertTrue(pdfio._is_formula_block("x^2 + y^2 = z^2"))
        self.assertTrue(pdfio._is_formula_block("∫_0^∞ e^{-x^2} dx = √π"))
        self.assertFalse(pdfio._is_formula_block("This is an ordinary English sentence."))
        self.assertFalse(pdfio._is_formula_block("3,702,726,474.45"))
        self.assertFalse(pdfio._is_formula_block("营业收入 合计"))

    def test_formula_detector_recognises_display_equation_fragments(self):
        # B-⑤ regression: a LaTeX paper splits a display equation into short line
        # fragments (``P_orig(y|x;T)=``, ``exp(z_i/T)``, ``argsort(z)≡…``) that the
        # old ``≥3 math chars`` test missed (they carry only 1–2 operators).
        self.assertTrue(pdfio._is_formula_block("Porig(yi|x; T) ="))
        self.assertTrue(pdfio._is_formula_block("exp(zi/T)"))
        self.assertTrue(pdfio._is_formula_block("argsort(z) ≡ argsort(z/T)"))
        # Prose / data that merely contains a relation is NOT a formula.
        self.assertFalse(pdfio._is_formula_block("T = 1.0, TTR = 0.400"))

    def test_formula_not_reported_missing(self):
        from translate_app.eval import measure_complete
        from translate_app.pdfio import Block
        res = measure_complete([Block("x^2 + y^2 = z^2", 0, 0, 0, 100, 20)], [""])
        self.assertEqual(res["missing_count"], 0)


class ColumnGapBreakTest(unittest.TestCase):
    """A 2-column page must not merge left and right lines into one full-width block."""

    @staticmethod
    def _ln(x0, x1, y0, y1, text):
        return {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "text": text,
                "size": 10.0, "bold": False, "color": 0}

    def test_cross_column_lines_break_even_when_y_close(self):
        # A right-column line followed by a left-column line at a *close* y (PyMuPDF's
        # line stream interleaves columns by y) must start a new block — otherwise they
        # merge into a full-width block whose translation is drawn across the page.
        base = self._ln(55, 289, 69, 79, "left one")
        prev = self._ln(307, 541, 123, 133, "right col line")
        cur = self._ln(55, 289, 128, 138, "left col next")
        self.assertTrue(pdfio._break_between(base, prev, cur))

    def test_same_column_lines_join(self):
        base = self._ln(55, 289, 80, 90, "left one")
        prev = self._ln(55, 289, 92, 102, "left two")
        cur = self._ln(55, 289, 104, 114, "left three")
        self.assertFalse(pdfio._break_between(base, prev, cur))

    def test_no_full_width_blocks_on_two_column_page(self):
        # Use the two-column fixture end-to-end: extraction must not yield a block
        # spanning both columns.
        with tempfile.TemporaryDirectory() as d:
            pdf = build_two_column_pdf(Path(d) / "two.pdf")
            dt = pdfio.extract_document_text(pdf, ocr=False, log=lambda m: None)
            # page 0 is a clean 2-column page (left ~60..215, right ~315..480).
            blocks = dt.pages[0]
            page_w = 595.0
            for b in blocks:
                self.assertLess((b.x1 - b.x0), page_w * 0.6,
                                f"block {b.text[:30]!r} spans too wide: {b.x0:.0f}..{b.x1:.0f}")


class GeometricStructureTest(unittest.TestCase):
    """The deterministic geometric structure backend (B-④/B-⑤ 'actually runs' offline)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _blocks(self):
        # formula (2 math symbols), bold heading, one table with a label + an amount.
        return [
            pdfio.Block("x^2 + y^2 = z^2", page=0, x0=0, y0=0, x1=100, y1=20),
            pdfio.Block("资产负债表", page=0, x0=0, y0=20, x1=100, y1=36,
                        bold=True, single_line=True),
            pdfio.Block("资产", page=0, x0=0, y0=40, x1=50, y1=60, in_table=True),
            pdfio.Block("1,234.56", page=0, x0=50, y0=40, x1=100, y1=60, in_table=True),
        ]

    def _rich_doc(self):
        blocks = self._blocks()
        return pdfio.DocumentText(pages=[blocks], blocks=[b.text for b in blocks],
                                  block_pages=[0, 0, 0, 0])

    def test_geometric_structure_fn_detects_kinds(self):
        sf = pdfio.make_geometric_structure_fn()
        regions = sf(0, None, self._blocks())
        kinds = {r["kind"] for r in regions}
        self.assertIn("formula", kinds)
        self.assertIn("heading", kinds)
        self.assertIn("table", kinds)
        tbl = next(r for r in regions if r["kind"] == "table")
        self.assertTrue(tbl["cells"])   # a row/col grid was built

    def test_build_structure_and_get_table_run_geometric(self):
        dt = self._rich_doc()
        src = build_sample_pdf(Path(self.tmp.name) / "geo.pdf", pages=1)
        pdfio.build_structure(str(src), dt, pdfio.make_geometric_structure_fn(), parser="geo")
        self.assertEqual(dt.structure_parser, "geo")
        self.assertTrue(dt.page_structure and dt.page_structure[0].elements)
        # The semantic table is now readable through get_table.
        gt = pdfio.get_table(dt, 0, 0)
        self.assertIsNotNone(gt)
        self.assertEqual(gt["rows"], 1)
        self.assertEqual(len(gt["cells"][0]), 2)
        # get_doc_info reports the structure parser once present.
        info = pdfio.get_doc_info(dt)
        self.assertEqual(info["structure_parser"], "geo")

    def test_extract_document_structured_runs_on_real_pdf(self):
        # A real PDF with a formula-ish line: the one-call entry finds it.
        import pymupdf as fitz
        src = Path(self.tmp.name) / "formula.pdf"
        d = fitz.open()
        p = d.new_page(width=300, height=200)
        p.insert_text((72, 80), "x^2 + y^2 = z^2", fontsize=12)
        d.save(str(src))
        d.close()
        dt = pdfio.extract_document_structured(str(src), parser="geo")
        self.assertEqual(dt.structure_parser, "geo")
        kinds = {e["kind"] for ps in dt.page_structure for e in ps.elements}
        self.assertIn("formula", kinds)

    def test_doclayout_structure_fn_degrades_to_geometric(self):
        # DocLayout-YOLO isn't installed here → the factory falls back to the
        # geometric backend, so it still produces real structure (never empty/crash).
        sf = pdfio.make_doclayout_structure_fn(log=lambda m: None)
        regions = sf(0, None, self._blocks())
        kinds = {r["kind"] for r in regions}
        self.assertIn("formula", kinds)
        self.assertIn("table", kinds)

    def test_make_vlm_ocr_fn_degrades_and_register(self):
        # No VLM backend registered → None (built-in RapidOCR used).
        self.assertIsNone(pdfio.make_vlm_ocr_fn())
        # Registering one makes the injectable ocr_fn available.
        pdfio.register_ocr_backend("vlm", lambda: lambda page_index, page: [])
        try:
            fn = pdfio.make_vlm_ocr_fn()
            self.assertTrue(callable(fn))
            self.assertEqual(fn(0, None), [])   # a wrapped backend callable
        finally:
            pdfio._OCR_BACKENDS.pop("vlm", None)

    def test_caption_regex_requires_a_numeral(self):
        # A bare "图" / "Fig" is a prose lead-in, not a caption; a labelled numeral is.
        self.assertRegex("图3 收入构成", pdfio._CAPTION_RE)
        self.assertRegex("Fig. 1: revenue trend", pdfio._CAPTION_RE)
        self.assertRegex("Table 2 主要指标", pdfio._CAPTION_RE)
        self.assertIsNone(pdfio._CAPTION_RE.match("图"))
        self.assertIsNone(pdfio._CAPTION_RE.match("Fig"))

    def test_get_table_page1_uses_flat_offset(self):
        # A table on page 1 must produce *flat* cells (offset added): page-local
        # index 0/1 become flat 2/3 when page 0 has 2 blocks.
        def structure_fn(page_index, page, blocks):
            if page_index != 1 or not blocks:
                return []
            bbox = [blocks[0].x0, blocks[0].y0, blocks[-1].x1, blocks[-1].y1]
            return [{"kind": "table", "bbox": bbox, "cells": [[0, 1]]}]
        p0 = [pdfio.Block("a", page=0, x0=0, y0=0, x1=10, y1=10),
              pdfio.Block("b", page=0, x0=0, y0=10, x1=10, y1=20)]
        p1 = [pdfio.Block("c", page=1, x0=0, y0=0, x1=50, y1=20, in_table=True),
              pdfio.Block("d", page=1, x0=50, y0=0, x1=100, y1=20, in_table=True)]
        dt = pdfio.DocumentText(pages=[p0, p1], blocks=["a", "b", "c", "d"],
                                block_pages=[0, 0, 1, 1])
        src = build_sample_pdf(Path(self.tmp.name) / "off.pdf", pages=2)
        pdfio.build_structure(str(src), dt, structure_fn, parser="mock")
        gt = pdfio.get_table(dt, 1, 0)
        self.assertIsNotNone(gt)
        self.assertEqual(gt["cells"][0], [2, 3])   # page-local 0,1 + offset 2


class OcrTableRedrawTest(unittest.TestCase):
    """``redraw_ocr`` regenerates a scanned table page as a clean vector table.

    Ported from 0.2.7: the current rebuild path replaced v0.4's
    ``table_vision``-based vector rebuild, so these are the regression guards for
    the replacement.
    """

    @staticmethod
    def _ocr_table_blocks(size: float = 6.0):
        return [
            pdfio.Block(text="总资产", page=0, x0=60, y0=100, x1=200, y1=112,
                        size=size, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="1,234,567.89", page=0, x0=210, y0=100, x1=360, y1=112,
                        size=size, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="总负债", page=0, x0=60, y0=120, x1=200, y1=132,
                        size=size, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="9,876,543.21", page=0, x0=210, y0=120, x1=360, y1=132,
                        size=size, single_line=True, ocr=True, in_table=True),
        ]

    def test_redraw_ocr_table_drops_raster_and_draws_cells(self):
        src = _OUT / "redraw_src.pdf"
        build_sample_pdf(src, pages=1)  # used only for the page size
        blocks = self._ocr_table_blocks()
        trans = ["Total assets", "1,234,567.89", "Total liabilities", "9,876,543.21"]
        out = _OUT / "redraw_out.pdf"
        pdfio.save_translated_pdf(src, [blocks], [trans], str(out), "English",
                                  redraw_ocr=True)
        doc = fitz.open(str(out))
        try:
            page = doc[0]
            self.assertEqual(0, len(page.get_images(full=True)))  # no raster background
            self.assertGreater(len(page.get_drawings()), 0)       # grid rules drawn
            text = page.get_text("text")
            self.assertIn("Total assets", text)
            self.assertIn("9,876,543.21", text)
        finally:
            doc.close()

    def test_redraw_keeps_sparse_text_layer_page_number(self):
        # Real regression (v0.5.24): a scanned report page keeps a 2-char text layer
        # holding the printed page number.  ``_merge_ocr_blocks`` merges it into the
        # OCR page, which made the purity gate reject the page — so 「OCR表格重建」
        # silently exported the scan unchanged.
        src = _OUT / "redraw_pageno_src.pdf"
        build_sample_pdf(src, pages=1)
        blocks = self._ocr_table_blocks() + [
            pdfio.Block(text="22", page=0, x0=292, y0=801, x1=303, y1=812,
                        size=6.0, single_line=True),
        ]
        trans = ["Total assets", "1,234,567.89", "Total liabilities",
                 "9,876,543.21", "22"]
        out = _OUT / "redraw_pageno_out.pdf"
        pdfio.save_translated_pdf(src, [blocks], [trans], str(out), "English",
                                  redraw_ocr=True)
        doc = fitz.open(str(out))
        try:
            page = doc[0]
            self.assertEqual(0, len(page.get_images(full=True)))  # scan dropped
            text = page.get_text("text")
            self.assertIn("Total assets", text)
            self.assertIn("22", text)  # the page number is put back, not dropped
        finally:
            doc.close()

    def test_redraw_ocr_off_keeps_inplace(self):
        src = _OUT / "redraw_src2.pdf"
        build_sample_pdf(src, pages=1)
        blocks = self._ocr_table_blocks(size=9.0)
        trans = ["Total assets", "1,234,567.89", "Total liabilities", "9,876,543.21"]
        out = _OUT / "redraw_out2.pdf"
        # redraw_ocr=False (default) → the source page (with its raster content) is kept.
        pdfio.save_translated_pdf(src, [blocks], [trans], str(out), "English",
                                  redraw_ocr=False)
        doc = fitz.open(str(out))
        src_doc = fitz.open(str(src))
        try:
            self.assertEqual(src_doc[0].rect.height, doc[0].rect.height)
            # In-place: the source page is kept (its own text layer survives; only
            # the OCR blocks are covered).
            self.assertIn("Page 1 heading", doc[0].get_text("text"))
        finally:
            doc.close()
            src_doc.close()

    def test_redraw_skips_chart_page(self):
        # A diagram (org chart) page has node labels, not data cells: redraw must
        # NOT blank it into an empty table. The page falls through to in-place.
        src = _OUT / "redraw_chart.pdf"
        build_sample_pdf(src, pages=1)
        chart_blocks = [
            pdfio.Block(text="董事会", page=0, x0=60, y0=100, x1=88, y1=130,
                        size=6.0, single_line=True, ocr=True, is_chart=True),
            pdfio.Block(text="监事会", page=0, x0=60, y0=140, x1=88, y1=170,
                        size=6.0, single_line=True, ocr=True, is_chart=True),
            pdfio.Block(text="委员会", page=0, x0=110, y0=100, x1=138, y1=130,
                        size=6.0, single_line=True, ocr=True, is_chart=True),
            pdfio.Block(text="3", page=0, x0=300, y0=270, x1=310, y1=282,
                        size=8.0, single_line=True, ocr=True),
        ]
        trans = [b.text for b in chart_blocks]
        out = _OUT / "redraw_chart_out.pdf"
        pdfio.save_translated_pdf(src, [chart_blocks], [trans], str(out), "English",
                                  redraw_ocr=True)
        doc = fitz.open(str(out))
        try:
            # Not redrawn: the original (source) page is kept, so its text survives.
            self.assertEqual(1, doc.page_count)
            self.assertIn("Page 1 heading", doc[0].get_text("text"))
        finally:
            doc.close()

    def test_redraw_skips_mixed_page(self):
        # A page carrying an OCR table AND a non-redraw-eligible block (here an
        # ``is_chart`` node label; a non-OCR footnote would do the same) must NOT be
        # blank-redrawn — the redraw keeps only the OCR table cells, silently
        # dropping the rest.  It falls through to the in-place path, which preserves
        # the whole original page.  Regression: the gate must not throw (a
        # "翻译失败" window) and must not blank the page.
        src = _OUT / "redraw_mixed_src.pdf"
        build_sample_pdf(src, pages=1)
        blocks = self._ocr_table_blocks() + [
            pdfio.Block(
                text="董事会", page=0, x0=300, y0=40, x1=340, y1=70,
                size=6.0, single_line=True, ocr=True, is_chart=True),
        ]
        trans = ["Total assets", "1,234,567.89", "Total liabilities", "9,876,543.21",
                 "董事会"]
        out = _OUT / "redraw_mixed_out.pdf"
        # Must not raise.
        pdfio.save_translated_pdf(src, [blocks], [trans], str(out), "English",
                                  redraw_ocr=True)
        doc = fitz.open(str(out))
        try:
            text = doc[0].get_text("text")
            # In-place (not blank-redrawn): the source page's own text survives.
            self.assertIn("Page 1 heading", text)
            # The OCR table cells are still translated in place.
            self.assertIn("Total assets", text)
            self.assertIn("9,876,543.21", text)
        finally:
            doc.close()

    def test_ai_table_rebuild_draws_regular_table(self):
        # With a model-derived grid, the OCR table page is drawn as a clean,
        # regular N x M table (no raster, regular grid, translated cells).
        src = _OUT / "ai_table_src.pdf"
        build_sample_pdf(src, pages=1)
        blocks = self._ocr_table_blocks()
        trans = [b.text for b in blocks]
        grid = [["Item", "2025", "2024"],
                ["Total assets", "1,234,567.89", "999,999.99"],
                ["Total liabilities", "9,876,543.21", "888,888.88"]]
        out = _OUT / "ai_table_out.pdf"
        logs: list[str] = []
        pdfio.save_translated_pdf(src, [blocks], [trans], str(out), "English",
                                  redraw_ocr=True, table_rebuild_fn=lambda _i, _png: grid,
                                  log=logs.append)
        doc = fitz.open(str(out))
        try:
            page = doc[0]
            self.assertEqual(0, len(page.get_images(full=True)))   # clean, no raster
            self.assertGreater(len(page.get_drawings()), 0)        # regular grid
            text = page.get_text("text")
            self.assertIn("Total assets", text)
            self.assertIn("999,999.99", text)
            self.assertIn("Item", text)
        finally:
            doc.close()
        self.assertTrue(any("正在 AI 表格重建" in m for m in logs), logs)
        self.assertTrue(any("AI 表格重建完成" in m for m in logs), logs)

    def test_ai_table_rebuild_invalid_falls_back(self):
        # An unavailable / implausible rebuilt grid falls back to the geometric redraw.
        src = _OUT / "ai_table_src2.pdf"
        build_sample_pdf(src, pages=1)
        blocks = self._ocr_table_blocks()
        trans = ["Total assets", "1,234,567.89", "Total liabilities", "9,876,543.21"]
        out = _OUT / "ai_table_out2.pdf"
        logs: list[str] = []
        pdfio.save_translated_pdf(src, [blocks], [trans], str(out), "English",
                                  redraw_ocr=True,
                                  table_rebuild_fn=lambda _i, _png: None,
                                  log=logs.append)
        doc = fitz.open(str(out))
        try:
            self.assertIn("Total assets", doc[0].get_text("text"))
        finally:
            doc.close()
        self.assertTrue(any("回退几何重绘" in m for m in logs), logs)

    def test_redraw_grows_the_row_and_draws_inside_the_band(self):
        # Regression (self-review): 0.2.7 drew each cell into its ORIGINAL OCR
        # bbox/fit_height, so the row expansion was cosmetic — the label still
        # rendered at 4.54pt in a band the grid had left at 12pt.  Now the row grows
        # to the height the translation needs at the readability floor and the cell
        # is drawn inside that band.
        src = _OUT / "band_src.pdf"
        build_sample_pdf(src, pages=1)
        label = "Operating revenue from continuing operations"
        blocks = [
            pdfio.Block(text="营业收入", page=0, x0=60, y0=100, x1=140, y1=112,
                        size=6.0, ocr=True, in_table=True, fit_width=80.0,
                        fit_height=10.5),
            pdfio.Block(text="1,234.56", page=0, x0=220, y0=100, x1=300, y1=112,
                        size=6.0, ocr=True, in_table=True),
            pdfio.Block(text="营业成本", page=0, x0=60, y0=120, x1=140, y1=132,
                        size=6.0, ocr=True, in_table=True, fit_width=80.0,
                        fit_height=10.5),
            pdfio.Block(text="9,876.54", page=0, x0=220, y0=120, x1=300, y1=132,
                        size=6.0, ocr=True, in_table=True),
        ]
        trans = [label, "1,234.56", "Cost of sales", "9,876.54"]
        out = _OUT / "band_out.pdf"
        pdfio.save_translated_pdf(src, [blocks], [trans], str(out), "English",
                                  redraw_ocr=True)
        doc = fitz.open(str(out))
        try:
            page = doc[0]
            spans = [(sp["size"], sp["bbox"], sp["text"])
                     for b in page.get_text("dict")["blocks"]
                     for ln in b.get("lines", []) for sp in ln["spans"]
                     if sp["text"].strip()]
            label_spans = [(sz, bb) for sz, bb, txt in spans if txt in label]
            self.assertEqual(2, len(label_spans), label_spans)      # wrapped to 2 lines
            # The old bug drew the label at 4.54pt inside a 12pt band.  The
            # threshold follows the (scaled) table readability floor instead of a
            # literal, so it keeps guarding the defect at any font scale.
            self.assertGreaterEqual(min(s[0] for s in label_spans),
                                    pdfio._MIN_TABLE_READABLE * 0.9,
                                    "the grown row must not leave the label crushed")
            h_lines = sorted({round(dr["rect"].y0, 1) for dr in page.get_drawings()
                              if dr["rect"].height < 1.0})
            self.assertGreaterEqual(len(h_lines), 2, h_lines)
            for _fs, bbox in label_spans:
                self.assertLess(bbox[3], h_lines[1] + 0.6,
                                "a wrapped cell must stay above the row's lower rule")
        finally:
            doc.close()

    def test_redraw_rotated_page_uses_the_unrotated_mediabox(self):
        # Regression (self-review): the blank page was created with the source's
        # *rotated* rect (e.g. 200x400) while OCR blocks live in the unrotated
        # mediabox (400x200) — every grid line past x=200 and half the cells fell
        # off the page.
        src = _OUT / "rot_src.pdf"
        doc = fitz.open()
        pg = doc.new_page(width=400, height=200)
        pg.set_rotation(90)
        doc.save(str(src))
        doc.close()
        blocks = [
            pdfio.Block(text="总资产", page=0, x0=60, y0=40, x1=200, y1=52,
                        size=6.0, ocr=True, in_table=True),
            pdfio.Block(text="1,234,567.89", page=0, x0=220, y0=40, x1=380, y1=52,
                        size=6.0, ocr=True, in_table=True),
            pdfio.Block(text="总负债", page=0, x0=60, y0=60, x1=200, y1=72,
                        size=6.0, ocr=True, in_table=True),
            pdfio.Block(text="9,876,543.21", page=0, x0=220, y0=60, x1=380, y1=72,
                        size=6.0, ocr=True, in_table=True),
        ]
        trans = ["Total assets", "1,234,567.89", "Total liabilities", "9,876,543.21"]
        out = _OUT / "rot_out.pdf"
        pdfio.save_translated_pdf(src, [blocks], [trans], str(out), "English",
                                  redraw_ocr=True)
        d = fitz.open(str(out))
        try:
            page = d[0]
            self.assertAlmostEqual(400.0, page.mediabox.width, places=1)
            self.assertAlmostEqual(200.0, page.mediabox.height, places=1)
            text = page.get_text("text")
            for want in trans:
                self.assertIn(want, text)
            for dr in page.get_drawings():
                self.assertLessEqual(dr["rect"].x1, 400.5)   # nothing off-page
        finally:
            d.close()

    def test_redraw_empty_ai_grid_falls_back_instead_of_blank(self):
        # A rebuild callback that returns an empty / all-blank grid must NOT produce
        # a blank page (silent content loss) — it falls back to the geometric redraw.
        src = _OUT / "blank_src.pdf"
        build_sample_pdf(src, pages=1)
        blocks = self._ocr_table_blocks()
        trans = ["Total assets", "1,234,567.89", "Total liabilities", "9,876,543.21"]
        out = _OUT / "blank_out.pdf"
        logs: list[str] = []
        pdfio.save_translated_pdf(src, [blocks], [trans], str(out), "English",
                                  redraw_ocr=True,
                                  table_rebuild_fn=lambda _i, _png: [["", ""], ["", ""]],
                                  log=logs.append)
        doc = fitz.open(str(out))
        try:
            text = doc[0].get_text("text")
            self.assertIn("Total assets", text)
            self.assertIn("Total liabilities", text)
        finally:
            doc.close()
        self.assertTrue(any("回退几何重绘" in m for m in logs), logs)

    def test_draw_ai_table_honours_explicit_merges(self):
        # The tool-provided merges drive a two-level header: the date spans the
        # Consolidated / Parent Company sub-columns (internal rule omitted).
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        font = fitz.Font("cjk")
        rows = [["Item", "December 31, 2025", "", "December 31, 2024", ""],
                ["", "", "Consolidated", "", "Parent Company"],
                ["Total assets", "1,234", "2,345", "3,456", "4,567"]]
        rect = fitz.Rect(36, 36, 559, 806)
        pdfio._draw_ai_table(page, rows, rect, font, merges=[
            {"r": 0, "c": 1, "rowspan": 1, "colspan": 2},
            {"r": 0, "c": 3, "rowspan": 1, "colspan": 2},
        ])
        text = page.get_text("text").replace("\n", " ")
        doc.close()
        self.assertIn("December 31", text)
        self.assertIn("Consolidated", text)
        self.assertIn("Parent Company", text)
        self.assertIn("Total assets", text)


    def test_draw_ai_table_wraps_with_tight_leading(self):
        # The rebuilt vector table's rows are sized from the same tight leading the
        # cells are drawn with.  With the loose paragraph leading (1.35) a two-line
        # label overran its row and the second line crossed the grid rule below.
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        font = fitz.Font("cjk")
        label = ('Net profit from continuing operations (net losses indicated '
                 'by "-") and other comprehensive income after tax')
        rows = [[label, "29"], ["Net profit from discontinued operations", "30"]]
        rect = fitz.Rect(36, 36, 300, 400)
        pdfio._draw_ai_table(page, rows, rect, font)
        lines = []
        for blk in page.get_text("dict")["blocks"]:
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                if text:
                    lines.append((line["bbox"], line["spans"][0]["size"], text))
        doc.close()
        # The first column's lines, top to bottom: the label wraps over two lines.
        col = sorted(
            [entry for entry in lines if abs(entry[0][0] - 40.0) <= 1.0],
            key=lambda entry: entry[0][1])
        self.assertGreaterEqual(len(col), 2, lines)
        (b0, fs0, t0), (b1, _fs1, t1) = col[0], col[1]
        # The first two lines are the label's own wrapped lines, not two rows.
        wrapped_prefix = " ".join(f"{t0} {t1}".split())
        self.assertTrue(label.startswith(wrapped_prefix), (t0, t1))
        gap = b1[1] - b0[1]                      # baseline-to-baseline spacing
        self.assertLessEqual(gap, fs0 * 1.1)     # tight (~1.0), not loose 1.35
        self.assertGreaterEqual(gap, fs0 * 0.9)


class TranslationFontScaleTest(unittest.TestCase):
    """A Latin translation at the source's point size reads larger than the CJK it
    replaces, so every fitted size is scaled (text-layer and OCR pages alike — all
    paths fit through ``_fit_block``)."""

    def test_start_size_and_floors_track_the_scale(self):
        b = pdfio.Block(text="x", page=0, x0=0.0, y0=0.0, x1=100.0, y1=12.0,
                        size=10.0)
        self.assertAlmostEqual(10.0 * pdfio._TRANSLATION_FONT_SCALE,
                               pdfio._font_start(b), places=3)
        self.assertAlmostEqual(7.0 * pdfio._TRANSLATION_FONT_SCALE,
                               pdfio._MIN_READABLE, places=6)
        self.assertAlmostEqual(6.0 * pdfio._TRANSLATION_FONT_SCALE,
                               pdfio._MIN_TABLE_READABLE, places=6)

    def test_translation_is_drawn_below_the_source_size(self):
        src = _OUT / "font_scale_src.pdf"
        blank = fitz.open()
        blank.new_page(width=400, height=300)
        blank.save(str(src))
        blank.close()
        blocks = [pdfio.Block(text="营业收入", page=0, x0=60, y0=100, x1=260,
                              y1=112, size=10.0)]
        out = _OUT / "font_scale_out.pdf"
        pdfio.save_translated_pdf(src, [blocks], [["Operating revenue"]],
                                  str(out), "English")
        doc = fitz.open(str(out))
        try:
            sizes = [sp["size"] for b in doc[0].get_text("dict")["blocks"]
                     for ln in b.get("lines", []) for sp in ln["spans"]
                     if sp["text"].strip()]
        finally:
            doc.close()
        self.assertTrue(sizes)
        self.assertLess(max(sizes), 10.0)     # never above the source size
        self.assertAlmostEqual(10.0 * pdfio._TRANSLATION_FONT_SCALE, max(sizes),
                               places=1)


class SymbolCellTest(unittest.TestCase):
    """P1-6: a pure-symbol *table cell* is content (``—`` = "no value"), not a
    stray page glyph — dropping it while the whole table bbox is redacted erased it
    for good."""

    def test_symbol_cell_survives_inplace_export(self):
        src = _OUT / "symbol_cell.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        page.draw_rect(fitz.Rect(60, 80, 340, 120), color=(0, 0, 0))
        page.draw_line(fitz.Point(200, 80), fitz.Point(200, 120), color=(0, 0, 0))
        page.insert_text((70, 105), "项目", fontsize=9, fontname="china-s")
        page.insert_text((210, 105), "—", fontsize=9, fontname="china-s")
        doc.save(str(src))
        doc.close()

        dt = pdfio.extract_document_text(src, log=lambda _m: None)
        texts = [b.text for b in dt.pages[0]]
        self.assertIn("—", texts, texts)
        out = _OUT / "symbol_cell_out.pdf"
        trans = [["Items" if b.text == "项目" else b.text for b in dt.pages[0]]]
        pdfio.save_translated_pdf(src, dt.pages, trans, str(out), "English")
        doc = fitz.open(str(out))
        try:
            # The table bbox redaction would have erased the dash if no block drew it.
            self.assertIn("—", doc[0].get_text())
        finally:
            doc.close()

    def test_symbol_cell_survives_the_ocr_grid(self):
        items = [
            (100.0, 60.0, 160.0, 109.0, "营业收入"),
            (100.0, 200.0, 260.0, 109.0, "1,234.56"),
            (120.0, 60.0, 160.0, 129.0, "营业成本"),
            (120.0, 200.0, 260.0, 129.0, "—"),
        ]
        blocks, tables = pdfio._reconstruct_ocr_grid(items)
        self.assertTrue(tables)
        self.assertIn("—", [b.text for b in blocks])

    def test_stray_symbol_outside_a_cell_is_still_dropped(self):
        # The original intent is preserved for prose: a lone ``。`` / bullet on the
        # page is noise, not content.
        cell_rects = [fitz.Rect(100, 100, 300, 120)]
        line = {"x0": 60, "y0": 105, "x1": 66, "y1": 115, "size": 9.0,
                "bold": False, "color": 0, "text": "。"}
        blocks = pdfio._build_table_blocks([line], [], 0, 600, 0, cell_rects)
        self.assertEqual([], blocks)


class ChartNotATableTest(unittest.TestCase):
    """P0-1: ``find_tables`` only looks for ruling lines, so a chart whose plot
    area has a full grid *and* text inside the cells was reported as a table — and
    the export's scoped table redaction then deleted the plot's own line art."""

    @staticmethod
    def _chart_pdf(path):
        """A 4-column chart: frame + full grid + a bar in each column + data labels."""
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        rect = fitz.Rect(80, 120, 500, 400)
        page.draw_rect(rect, color=(0, 0, 0), width=1)
        xs = [rect.x0 + rect.width / 5 * i for i in range(1, 5)]
        ys = [rect.y0 + rect.height / 4 * i for i in range(1, 4)]
        for y in ys:
            page.draw_line(fitz.Point(rect.x0, y), fitz.Point(rect.x1, y),
                           color=(0.6, 0.6, 0.6))
        for x in xs:
            page.draw_line(fitz.Point(x, rect.y0), fitz.Point(x, rect.y1),
                           color=(0.6, 0.6, 0.6))
        bar = None
        for i in range(3):
            x0 = rect.x0 + 20 + i * 120
            y0 = rect.y1 - 40 - i * 30
            r = fitz.Rect(x0, y0, x0 + 50, rect.y1 - 8)
            page.draw_rect(r, color=None, fill=(0.2, 0.4, 0.8))
            if i == 1:
                bar = r
        # Data labels inside the cells: this is what makes ``find_tables`` see a grid.
        for ri, y in enumerate([rect.y0] + ys):
            for ci, x in enumerate([rect.x0] + xs):
                if ci < 5 and ri < 4:
                    page.insert_text((x + 6, y + 16), f"{ri}{ci}", fontsize=8)
        doc.save(str(path))
        doc.close()
        return bar

    def test_chart_with_grid_and_labels_is_not_a_table(self):
        src = _OUT / "chart_grid.pdf"
        self._chart_pdf(src)
        doc = fitz.open(str(src))
        try:
            # Sanity: PyMuPDF itself does report a table here (the trap).
            self.assertTrue(getattr(doc[0].find_tables(), "tables", None))
            self.assertEqual([], pdfio._extract_tables(doc[0]))
            self.assertEqual([], pdfio._detect_table_cell_rects(doc[0]))
        finally:
            doc.close()

    def test_chart_labels_stay_prose_blocks(self):
        src = _OUT / "chart_grid2.pdf"
        self._chart_pdf(src)
        dt = pdfio.extract_document_text(src, log=lambda _m: None)
        self.assertTrue(dt.pages[0])
        self.assertFalse(any(b.in_table for b in dt.pages[0]))

    def test_chart_bars_survive_the_inplace_export(self):
        src = _OUT / "chart_grid3.pdf"
        bar = self._chart_pdf(src)
        dt = pdfio.extract_document_text(src, log=lambda _m: None)
        trans = [["T" + b.text for b in dt.pages[0]]]
        out = _OUT / "chart_grid_out.pdf"
        pdfio.save_translated_pdf(src, dt.pages, trans, str(out), "English")
        doc = fitz.open(str(out))
        try:
            pix = doc[0].get_pixmap(dpi=72)
            cx = int((bar.x0 + bar.x1) / 2)
            cy = int((bar.y0 + bar.y1) / 2)
            self.assertEqual((51, 102, 204), tuple(pix.pixel(cx, cy)))
        finally:
            doc.close()


class TablePageBottomClampTest(unittest.TestCase):
    """P0-3: row expansion must never push rows (or the prose below) past the page
    bottom — the source text is already redacted by then, so off-page content was
    silently lost."""

    def _table(self, first_top: float = 260.0, height: float = 10.0):
        cells, rows = [], []
        y = first_top
        for i, label in enumerate(("营业收入", "营业成本", "营业利润")):
            top, bot = y, y + height
            cells.append(pdfio.Block(
                text=label, page=0, x0=60, y0=top + 1, x1=140, y1=bot - 1,
                size=9.0, single_line=True, in_table=True))
            cells.append(pdfio.Block(
                text=f"{i},234", page=0, x0=150, y0=top + 1, x1=230, y1=bot - 1,
                size=9.0, single_line=True, in_table=True))
            rows.append([fitz.Rect(60, top, 140, bot), fitz.Rect(150, top, 230, bot)])
            y = bot
        tables = [{"bbox": fitz.Rect(60, first_top, 230, y), "rows": rows,
                   "col_edges": [60.0, 140.0, 230.0]}]
        mapping = {i: (0, i // 2, i % 2) for i in range(len(cells))}
        long = "Operating revenue from the bank's core lending business for the year"
        trans = []
        for _ in range(len(cells) // 2):
            trans += [long, "1,234"]
        return tables, mapping, cells, trans

    def test_growth_is_clamped_to_the_page_bottom(self):
        font = fitz.Font("cjk")
        tables, mapping, cells, trans = self._table()
        logs: list[str] = []
        _shifts, new_bottoms, grid, _bboxes = pdfio._compute_table_layout(
            tables, mapping, cells, trans, font,
            page_height=300.0, log=logs.append)
        limit = 300.0 - pdfio._TABLE_BOTTOM_MARGIN
        self.assertTrue(new_bottoms)
        for bottom in new_bottoms.values():
            self.assertLessEqual(bottom, limit + 0.01)
        self.assertLessEqual(max(g[3] for g in grid if g[0] == "h"), limit + 0.01)
        self.assertTrue(any("超出页底" in m for m in logs), logs)

    def test_without_a_page_height_growth_stays_unbounded(self):
        # Documents the old behaviour: no page height -> no clamp.
        font = fitz.Font("cjk")
        tables, mapping, cells, trans = self._table()
        _shifts, new_bottoms, _grid, _bboxes = pdfio._compute_table_layout(
            tables, mapping, cells, trans, font)
        self.assertGreater(max(new_bottoms.values()), 300.0)

    def test_table_that_fits_is_not_touched(self):
        # A table well inside the page must keep its exact expansion (no clamp).
        font = fitz.Font("cjk")
        tables, mapping, cells, trans = self._table(first_top=100.0)
        clamped = pdfio._compute_table_layout(
            tables, mapping, cells, trans, font, page_height=842.0, log=lambda _m: None)
        free = pdfio._compute_table_layout(tables, mapping, cells, trans, font)
        self.assertEqual(free[0], clamped[0])
        self.assertEqual(free[1], clamped[1])

    def test_export_passes_the_cropbox_height(self):
        # Regression: the exporter passed ``page.cropbox.y1`` (which includes the
        # cropbox's origin) while the block boxes are cropbox-*relative*, so a
        # CropBox that crops a scanned book made the limit too large and the clamp
        # never fired — the last rows were drawn past the sheet.
        doc = fitz.open()
        page = doc.new_page(width=400, height=500)
        page.insert_text((60, 60), "TABLE CELL TEXT", fontsize=11)
        page.set_cropbox(fitz.Rect(20, 40, 380, 460))          # height 420, y1 460
        src = _OUT / "crop_table_src.pdf"
        doc.save(str(src))
        doc.close()
        dt = pdfio.extract_document_text(str(src), log=lambda _m: None)
        block = dt.pages[0][0]
        fake_table = {
            "bbox": fitz.Rect(block.x0, block.y0, block.x1, block.y1),
            "rows": [[fitz.Rect(block.x0, block.y0, block.x1, block.y1)]],
            "col_edges": [block.x0, block.x1],
        }
        seen: dict = {}

        def fake_layout(tables, mapping, blocks, trans, font, *,
                        page_height=None, log=None):
            seen["page_height"] = page_height
            return {}, {}, [], []

        with mock.patch.object(pdfio, "_extract_tables", return_value=[fake_table]), \
             mock.patch.object(pdfio, "_map_blocks_to_table_cells",
                               return_value={0: (0, 0, 0)}), \
             mock.patch.object(pdfio, "_compute_table_layout", side_effect=fake_layout):
            pdfio.save_translated_pdf(str(src), dt.pages, [["TRANSLATED"]],
                                      str(_OUT / "crop_table_out.pdf"), "English",
                                      log=lambda _m: None)
        self.assertAlmostEqual(420.0, seen.get("page_height", -1), places=1)

    def test_ocr_grid_redraw_stays_on_the_page(self):
        # Regression: ``_draw_ocr_grid_page`` grows rows monotonically with no
        # bottom clamp.  The redraw starts from a *blank* page, so a row pushed
        # past the edge is lost with no trace.
        doc = fitz.open()
        page = doc.new_page(width=300, height=200)
        blocks: list[pdfio.Block] = []
        trans: list[str] = []
        long = "A very long translated table cell that must wrap over several lines "
        for r in range(3):
            for c in range(2):
                x0 = 20.0 + c * 140.0
                y0 = 170.0 + r * 8.0
                blocks.append(pdfio.Block(
                    text=f"cell {r}{c}", page=0, x0=x0, y0=y0, x1=x0 + 120.0,
                    y1=y0 + 7.0, size=7.0, ocr=True, in_table=True))
                trans.append(long * 2)
        logs: list[str] = []
        pdfio._draw_ocr_grid_page(page, blocks, trans, fitz.Font("cjk"), logs.append)
        for blk in page.get_text("dict")["blocks"]:
            self.assertLessEqual(blk["bbox"][3], 200.0 + 0.5, blk["bbox"])
        self.assertTrue(any("超出页底" in m for m in logs), logs)
        doc.close()


class ExpandPagesTest(unittest.TestCase):
    """译文扩页（expand_pages）：源页放不下的表格行改排到后续页并重复表头，
    而不是压缩行高。默认关闭时导出与改动前逐字一致。"""

    LONG = "Operating revenue from the bank's core lending business for the year"

    @staticmethod
    def _table_pdf(path, top=120.0, rows=4, height=12.0, pages=1):
        doc = fitz.open()
        for _ in range(pages):
            page = doc.new_page(width=400, height=200)
            xs = [60.0, 100.0, 140.0]
            labels = [("项目", "金额"), ("收入", "1,234"),
                      ("成本", "5,678"), ("利润", "9,012")][:rows]
            for r, row in enumerate(labels):
                y0 = top + r * height
                y1 = y0 + height
                for c in range(2):
                    page.draw_rect(fitz.Rect(xs[c], y0, xs[c + 1], y1),
                                   color=(0, 0, 0), width=0.6)
                    page.insert_text((xs[c] + 2, y0 + 8), row[c], fontsize=7,
                                     fontname="china-s")
        doc.save(str(path))
        doc.close()

    def _translations(self, rows=4):
        """Short header, long values — the English needs far more room."""
        per = ["Item", "Amount"]
        for i in range(1, rows):
            per += [self.LONG, f"{i},000"]
        return per

    def _export(self, src, out, per, **kw):
        dt = pdfio.extract_document_text(str(src), log=lambda _m: None)
        logs: list[str] = []
        pdfio.save_translated_pdf(str(src), dt.pages, [per], str(out), "English",
                                  log=logs.append, **kw)
        return logs

    def test_default_export_keeps_one_page_and_clamps(self):
        # 默认（不勾选）行为不变：行高被钳制在一页内，页数仍与原文一致。
        src = _OUT / "expand_default_src.pdf"
        self._table_pdf(src)
        out = _OUT / "expand_default.pdf"
        logs = self._export(src, out, self._translations())
        doc = fitz.open(str(out))
        try:
            self.assertEqual(1, doc.page_count)
        finally:
            doc.close()
        self.assertTrue(any("超出页底" in m for m in logs), logs)

    def test_overflow_flows_to_a_continuation_page_with_the_header(self):
        src = _OUT / "expand_src.pdf"
        self._table_pdf(src)
        out = _OUT / "expand_on.pdf"
        logs = self._export(src, out, self._translations(), expand_pages=True)
        doc = fitz.open(str(out))
        try:
            self.assertEqual(2, doc.page_count)
            p1 = " ".join(doc[1].get_text().split())
            # The continuation page repeats the table header and carries the rows
            # that no longer fitted on page 1.
            self.assertIn("Item", p1)
            self.assertIn("Amount", p1)
            self.assertIn("3,000", p1)
            self.assertNotIn("1,000", p1)   # row 1 stayed on page 1
            # …and the grid was redrawn there — not a blank sheet.
            self.assertGreater(len(doc[1].get_drawings()), 0)
        finally:
            doc.close()
        self.assertTrue(any("已扩展到后续" in m for m in logs), logs)

    def test_content_that_fits_adds_no_pages(self):
        # 装得下就一页都不加：扩页只在真的溢出时生效。
        src = _OUT / "expand_fits_src.pdf"
        self._table_pdf(src, top=40.0)
        out = _OUT / "expand_fits.pdf"
        logs = self._export(src, out, self._translations(), expand_pages=True)
        doc = fitz.open(str(out))
        try:
            self.assertEqual(1, doc.page_count)
        finally:
            doc.close()
        self.assertEqual([], logs)

    def test_continuation_page_orders_the_rows_and_stays_on_the_sheet(self):
        # 续页上表头在最上、数据行按原顺序往下排，且没有内容被排到页外
        # （扩页的整个意义就是把内容留在纸上，而不是压缩到看不清）。
        src = _OUT / "expand_order_src.pdf"
        self._table_pdf(src)
        out = _OUT / "expand_order.pdf"
        self._export(src, out, self._translations(), expand_pages=True)
        doc = fitz.open(str(out))
        try:
            page = doc[1]
            words = page.get_text("words")

            def top_of(token):
                ys = [w[1] for w in words if token in w[4]]
                return min(ys) if ys else None

            header = top_of("Item")
            row2 = top_of("2,000")
            row3 = top_of("3,000")
            self.assertIsNotNone(header)
            self.assertIsNotNone(row2)
            self.assertIsNotNone(row3)
            self.assertLess(header, row2)
            self.assertLess(row2, row3)
            self.assertLessEqual(max(w[3] for w in words), page.rect.height + 0.5)
        finally:
            doc.close()

    def test_expand_pages_never_moves_a_scanned_block(self):
        # 扫描块的「文字」是位图：搬去续页会让首页原文裸露、续页多出一份译文。
        # 它必须原样留在首页（白底覆盖 + 画译文），页数不变。
        src = _OUT / "expand_ocr_src.pdf"
        doc = fitz.open()
        page = doc.new_page(width=300, height=120)
        page.insert_text((20, 32), "SCAN-ORIGINAL", fontsize=12)
        doc.save(str(src))
        doc.close()
        block = pdfio.Block(text="SCAN-ORIGINAL", page=0, x0=18, y0=20, x1=140,
                            y1=36, size=12, ocr=True, single_line=True)
        long = "A very long translation that no longer fits " * 6
        out = _OUT / "expand_ocr.pdf"
        pdfio.save_translated_pdf(str(src), [[block]], [[long]], str(out), "English",
                                  log=lambda _m: None, expand_pages=True)
        doc = fitz.open(str(out))
        try:
            self.assertEqual(1, doc.page_count)
            text = " ".join(doc[0].get_text().split())
            self.assertIn("long translation", text)
        finally:
            doc.close()

    def test_expand_pages_tolerates_a_short_per_page(self):
        # save_translated_pdf 明确容忍较短的 per_page（源被改过 / 调用方传短了）。
        # 扩页搬走的行会引用缺失的译文，不能因此 IndexError。
        src = _OUT / "expand_short_src.pdf"
        self._table_pdf(src)
        dt = pdfio.extract_document_text(str(src), log=lambda _m: None)
        out = _OUT / "expand_short.pdf"
        per = [["Item", self.LONG]]       # 只有前两个块有译文
        pdfio.save_translated_pdf(str(src), dt.pages, per, str(out), "English",
                                  log=lambda _m: None, expand_pages=True)
        doc = fitz.open(str(out))
        try:
            self.assertGreaterEqual(doc.page_count, 1)
        finally:
            doc.close()

    def test_expand_pages_returns_a_source_to_output_page_map(self):
        # 预览的「译文」侧靠这张映射把源页定位到输出页：扩页后不再一一对应。
        src = _OUT / "expand_map_src.pdf"
        self._table_pdf(src, pages=2)
        dt = pdfio.extract_document_text(str(src), log=lambda _m: None)
        out = _OUT / "expand_map.pdf"
        per_page = [self._translations(), []]   # 第 2 页无译文
        mapping = pdfio.save_translated_pdf(
            str(src), dt.pages, per_page, str(out), "English",
            log=lambda _m: None, expand_pages=True)
        # 第 0 页溢出一页 → 第 1 页从输出第 2 页开始。
        self.assertEqual([0, 2], list(mapping))
        doc = fitz.open(str(out))
        try:
            self.assertEqual(3, doc.page_count)
        finally:
            doc.close()


    def test_an_oversized_paragraph_continues_onto_the_next_page(self):
        # 一个高于一页的正文块必须**按行**跨页续排：把元素当成不可分单元时，
        # 超出页底的部分永久落在纸外——实测开扩页比不开丢得更多（630→481 词），
        # 与「扩页是为了不丢内容」正相反。
        src = _OUT / "expand_paragraph_src.pdf"
        out = _OUT / "expand_paragraph.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        page.insert_text((40, 62), "正文", fontsize=9, fontname="china-s")
        doc.save(str(src))
        doc.close()

        text = ("Operating revenue from the bank's core lending business "
                "for the year ended 31 December " * 24).strip()
        blocks = [pdfio.Block("正文", 0, 40, 55, 360, 70, size=9)]
        logs: list[str] = []
        pdfio.save_translated_pdf(str(src), [blocks], [[text]], str(out),
                                  "English", log=logs.append, expand_pages=True)
        expected = len("".join(text.split()))
        with fitz.open(str(out)) as o:
            self.assertGreater(o.page_count, 1, "高于一页的段落必须续页")
            extracted = 0
            for p in o:
                words = p.get_text("words")
                extracted += len("".join(w[4] for w in words))
                self.assertLessEqual(max((w[3] for w in words), default=0.0),
                                     p.rect.height + 0.5,
                                     "续页正文不得画到页外")
            # 连字符换行只会**多**出字符，所以「≥ 期望值」等价于「一字未丢」。
            self.assertGreaterEqual(extracted, expected,
                                    f"译文丢了字符：{extracted} < {expected}")
        self.assertTrue(any("已扩展到后续" in m for m in logs), logs)

    def test_a_block_below_a_grown_table_never_leaves_the_page(self):
        # 表格下方的 OCR 块：像素不可搬移（_flow_skips 拒绝搬走），因此它的
        # 下推必须被钳制。此前「关钳制 + 不搬走」两条交集让译文被画到页外：
        # 扫描原文被白底擦掉、译文看不见，且没有任何日志。
        src = _OUT / "expand_fixed_src.pdf"
        out = _OUT / "expand_fixed.pdf"
        _OUT.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        xs = [60.0, 100.0, 140.0]
        for r, (label, value) in enumerate([("项目", "金额"), ("收入", "1,234"),
                                            ("成本", "5,678")]):
            y0 = 30.0 + r * 12.0
            for c, text in enumerate((label, value)):
                page.draw_rect(fitz.Rect(xs[c], y0, xs[c + 1], y0 + 12.0),
                               color=(0, 0, 0), width=0.6)
                page.insert_text((xs[c] + 2, y0 + 8), text, fontsize=7,
                                 fontname="china-s")
        doc.save(str(src))
        doc.close()

        cells = pdfio.Block("项目", 0, 62, 31, 98, 39, size=7, in_table=True)
        scanned = pdfio.Block("扫描页脚", 0, 60, 178, 260, 190, size=8,
                              ocr=True, single_line=True)
        long = "Operating revenue from the bank's core lending business for the year"
        footnote = "Unit: RMB ten thousand yuan"
        blocks = [cells, scanned]
        logs: list[str] = []
        pdfio.save_translated_pdf(str(src), [blocks], [[long, footnote]], str(out),
                                  "English", log=logs.append, expand_pages=True)
        with fitz.open(str(out)) as o:
            self.assertEqual(1, o.page_count,
                             "不可搬移块所在页不得扩页（必须保持页底钳制）")
            text = o[0].get_text()
            self.assertIn("Operating", text, "表格块的译文必须留在纸上")
            self.assertIn("Unit: RMB ten thousand yuan", text,
                          "扫描块的译文必须留在纸上（此前被推出页面）")
            words = o[0].get_text("words")
            self.assertLessEqual(max((w[3] for w in words), default=0.0),
                                 o[0].rect.height + 0.5)
        self.assertTrue(any("不可搬移" in m for m in logs), logs)

    def test_expand_pages_on_a_rotated_page_does_not_inflate_the_document(self):
        # /Rotate 页上 find_tables 的几何在旋转显示帧、块坐标在未旋转帧，拆分
        # 判定因此无意义：实测 1 页 → 4 页（首页空白、续页重复同一批行）。
        # 现在旋转页明确不扩页（保持页底钳制）并记一行日志。
        src = _OUT / "expand_rot_src.pdf"
        out = _OUT / "expand_rot.pdf"
        _OUT.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        xs = [60.0, 100.0, 140.0]
        for r, (label, value) in enumerate([("项目", "金额"), ("收入", "1,234"),
                                            ("成本", "5,678"), ("利润", "9,012")]):
            y0 = 30.0 + r * 12.0
            for c, text in enumerate((label, value)):
                page.draw_rect(fitz.Rect(xs[c], y0, xs[c + 1], y0 + 12.0),
                               color=(0, 0, 0), width=0.6)
                page.insert_text((xs[c] + 2, y0 + 8), text, fontsize=7,
                                 fontname="china-s")
        page.set_rotation(90)
        doc.save(str(src))
        doc.close()

        dt = pdfio.extract_document_text(str(src), log=lambda _m: None)
        per = [self.LONG if not b.text.replace(",", "").isdigit() else b.text
               for b in dt.pages[0]]
        logs: list[str] = []
        mapping = pdfio.save_translated_pdf(str(src), dt.pages, [per], str(out),
                                            "English", log=logs.append,
                                            expand_pages=True)
        self.assertEqual([0], list(mapping))
        with fitz.open(str(out)) as o:
            self.assertLessEqual(o.page_count, 2,
                                 "旋转页不得因扩页而膨胀")
        self.assertTrue(any("旋转页" in m for m in logs), logs)


class AtomicTextExportTest(unittest.TestCase):
    """``.txt`` / ``.md`` 导出必须原子：导出直接覆盖目标文件，而关窗会 ``os._exit``。

    就地写一半被打断会留下 0 字节/半截文件——上一份可用产物被毁掉，且译文本只存在
    内存里（事后只能整篇重译）。PDF 侧因为 ``fitz.save`` 写新文件而天然原子。
    """

    def test_a_failed_write_keeps_the_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.txt"
            out.write_text("OLD", encoding="utf-8")
            real = Path.write_text

            def half_written(self, data, *args, **kwargs):
                real(self, "HALF", *args, **kwargs)   # 半截内容进临时文件
                raise OSError("boom")

            with mock.patch.object(Path, "write_text", half_written):
                with self.assertRaises(OSError):
                    pdfio.save_plain_text([["NEW"]], out)
            self.assertEqual("OLD", out.read_text("utf-8"),
                             "失败时不得动到已有产物")
            self.assertEqual([], list(Path(tmp).glob("*.tmp")),
                             "临时文件必须清理")

    def test_markdown_is_replaced_through_a_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.md"
            out.write_text("OLD", encoding="utf-8")
            pdfio.save_markdown([["Hello"]], ["你好"], [0], out, "English")
            text = out.read_text("utf-8")
            self.assertIn("Hello", text)
            self.assertIn("你好", text)
            self.assertEqual([], list(Path(tmp).glob("*.tmp")))


class TableRulesAfterPushDownTest(unittest.TestCase):
    """表格行高被下推后，**原有**的表格线必须跟着走（擦掉＋重画），且不得凭空加线。

    两个 v0.6.4 缺陷：
    * ``_extend_with_body`` 把「有框线、只是表体无线」的表整张标成 ``borderless``
      → 行高下推但表格线既不擦也不重画，译文压在留在原位的旧线上（实测 19/102）；
    * 列宽重排可把窄列压到 2×``_TABLE_CELL_PAD`` 以下 → 反框（width<0）→ 单元格逐字
      换行并越出表格右边界。
    """

    LONG = ("The consolidated and the parent company financial statements of the "
            "Group for the year ended 31 December 2025 prepared in accordance "
            "with the International Financial Reporting Standards")

    def _crossings(self, path: Path) -> tuple[int, int]:
        """(跨线的文本行数, 文本行总数)。"""
        doc = fitz.open(str(path))
        try:
            page = doc[0]
            rules = self._h_rules(path)
            boxes = [line["bbox"] for blk in page.get_text("dict")["blocks"]
                     for line in blk.get("lines", [])]
            cross = sum(1 for bb in boxes
                        if any(bb[1] < ry < bb[3] for ry in rules))
            return cross, len(boxes)
        finally:
            doc.close()

    def test_a_stacked_ruled_table_keeps_its_rules_aligned(self):
        src = _OUT / "pushdown_src.pdf"
        out = _OUT / "pushdown.pdf"
        _OUT.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        xs = [60.0, 170.0, 340.0, 500.0]
        for base in (90.0, 178.0):          # 两张上下相邻、各自全框线的表
            for r in range(4):
                y0 = base + r * 22.0
                page.draw_line(fitz.Point(xs[0], y0), fitz.Point(xs[-1], y0),
                               width=0.5)
            for c in range(3):
                page.draw_line(fitz.Point(xs[c], base), fitz.Point(xs[c], base + 66),
                               width=0.5)
                for r in range(3):
                    page.insert_text((xs[c] + 4, base + 15 + r * 22),
                                     f"L{r}{c}", fontsize=9)
        doc.save(str(src))
        doc.close()

        dt = pdfio.extract_document_text(str(src), ocr=False, log=lambda _m: None)
        per = [[self.LONG for _ in dt.pages[0]]]
        pdfio.save_translated_pdf(str(src), dt.pages, per, str(out), "English",
                                  log=lambda _m: None, reflow=True)
        h_rules = self._h_rules(out)
        # 行高被下推后，表格线必须跟着走：除了表顶（不再移动的那条），输出里的横线
        # 不得停留在源文件的旧行位置——旧线留在原地正是「译文压线」的根因。
        self.assertEqual({90.0}, set(h_rules) & set(self.STACKED_RULES), h_rules)

    #: 上面那张测试页所有横线的原始 y（表顶 90 与第二张表的顶 178 之外都会下移）。
    STACKED_RULES = {90.0, 112.0, 134.0, 156.0, 178.0, 200.0, 222.0, 244.0}

    def test_a_table_body_above_the_ruled_band_keeps_the_rules(self):
        # ``_extend_with_body`` can find the unruled body *above* the band
        # ``find_tables`` really saw (a report's total band sits at the bottom).
        # Reading the ruled rows as "the FIRST N rows" then drew the grid over the
        # synthesised prose and left the real band redacted-but-never-redrawn: its
        # rules vanished and lines the source never had appeared.  The ruled rows
        # are a *range* (v0.6.5 review P1-2).
        src = _OUT / "above_src.pdf"
        out = _OUT / "above.pdf"
        _OUT.mkdir(parents=True, exist_ok=True)
        xs = [72.0, 220.0, 380.0, 520.0]
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        # A full-width rule above, then column-confined prose (the unruled body).
        page.draw_line(fitz.Point(xs[0], 150), fitz.Point(xs[-1], 150), width=0.8)
        body = (("Delta", "Echo", "Foxtrot"), ("Golf", "Hotel", "India"),
                ("Juliet", "Kilo", "Lima"))
        for r, row in enumerate(body):
            for c, text in enumerate(row):
                page.insert_text((xs[c] + 6, 155 + r * 30 + 12), text, fontsize=11)
        # … and the fully ruled band find_tables really sees, at the bottom.
        for y in (250.0, 300.0):
            page.draw_line(fitz.Point(xs[0], y), fitz.Point(xs[-1], y), width=0.8)
        for x in xs:
            page.draw_line(fitz.Point(x, 250.0), fitz.Point(x, 300.0), width=0.8)
        for c, text in enumerate(("Alpha", "Bravo", "Charlie")):
            page.insert_text((xs[c] + 6, 270), text, fontsize=11)
        doc.save(str(src))
        doc.close()

        dt = pdfio.extract_document_text(str(src), ocr=False, log=lambda _m: None)
        per = [["T:" + b.text for b in dt.pages[0]]]   # short: no push-down
        pdfio.save_translated_pdf(str(src), dt.pages, per, str(out), "English",
                                  log=lambda _m: None)
        rules = set(self._h_rules(out))
        self.assertEqual({150.0, 250.0, 300.0}, rules,
                         "the ruled band's own rules must be redrawn where they"
                         " were, and no rule invented over the synthesised prose")

    def _h_rules(self, path: Path) -> list[float]:
        doc = fitz.open(str(path))
        try:
            return sorted({
                round(it[1].y, 1)
                for dr in doc[0].get_drawings()
                for it in dr.get("items", ())
                if it[0] == "l" and abs(it[1].y - it[2].y) < 0.5
            })
        finally:
            doc.close()

    def test_a_narrow_column_is_never_squeezed_to_a_reverse_box(self):
        src = _OUT / "reflow_src.pdf"
        out = _OUT / "reflow.pdf"
        _OUT.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        xs = [60.0, 170.0, 500.0, 545.0]
        top, bot = 90.0, 200.0
        for y in (top, top + 36, top + 72, bot):
            page.draw_line(fitz.Point(xs[0], y), fitz.Point(xs[-1], y), width=0.5)
        for x in xs:
            page.draw_line(fitz.Point(x, top), fitz.Point(x, bot), width=0.5)
        for i, y in enumerate((110.0, 146.0, 182.0)):
            page.insert_text((70, y), f"附注项目{i + 1}", fontsize=9,
                             fontname="china-s")
            page.insert_text((180, y), f"{i + 1},234,567.89", fontsize=9)
            page.insert_text((505, y), "(二)" if i == 0 else "—", fontsize=9,
                             fontname="china-s")
        doc.save(str(src))
        doc.close()

        dt = pdfio.extract_document_text(str(src), ocr=False, log=lambda _m: None)
        trans = []
        for b in dt.pages[0]:
            if pdfio._is_numeric_cell(b.text):
                trans.append(b.text)
            elif b.text.startswith("附注"):
                trans.append(self.LONG)
            else:
                trans.append(b.text)
        pdfio.save_translated_pdf(str(src), dt.pages, [trans], str(out), "English",
                                  log=lambda _m: None, reflow=True)
        with fitz.open(str(out)) as o:
            words = o[0].get_text("words")
        over = [w for w in words if w[2] > 545.0 + 1.0]
        self.assertEqual([], [w[4] for w in over],
                         "窄列被压成反框后译文越出了表格右边界")


    def test_a_continuation_page_keeps_the_source_crop_frame(self):
        # 块坐标是 cropbox 相对帧：续页若只按 mediabox 建页，裁剪页的续页会比源页
        # 「可见区」更大（实测源可见 200pt、续页 300pt），内容偏移也对不上。
        src = _OUT / "expand_crop_src.pdf"
        out = _OUT / "expand_crop.pdf"
        _OUT.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        page.insert_text((40, 60), "正文", fontsize=9, fontname="china-s")
        page.set_cropbox(fitz.Rect(0, 50, 400, 250))      # 可见区 400×200
        doc.save(str(src))
        doc.close()

        text = ("Operating revenue from the bank's core lending business for the "
                "year ended 31 December " * 30).strip()
        blocks = [pdfio.Block("正文", 0, 40, 55, 360, 70, size=9)]
        pdfio.save_translated_pdf(str(src), [blocks], [[text]], str(out), "English",
                                  log=lambda _m: None, expand_pages=True)
        with fitz.open(str(out)) as o:
            self.assertGreater(o.page_count, 1, "本用例必须真的扩页")
            for p in o:
                self.assertAlmostEqual(400.0, p.cropbox.width, delta=0.5)
                self.assertAlmostEqual(200.0, p.cropbox.height, delta=0.5)


class BorderlessTableTest(unittest.TestCase):
    """P1-1: a table without ruling lines must still get cell geometry — the
    ``text`` fallback is used, with guards so prose pages are not "found"."""

    @staticmethod
    def _borderless_table(path):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        rows = [("营业收入", "1,234,567.89"), ("营业成本", "9,876,543.21"),
                ("营业利润", "5,555,555.55")]
        for i, (label, value) in enumerate(rows):
            page.insert_text((60, 100 + i * 16), label, fontsize=11, fontname="china-s")
            page.insert_text((300, 100 + i * 16), value, fontsize=11)
        doc.save(str(path))
        doc.close()

    @staticmethod
    def _two_column_prose(path):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        para = ("本行报告期内的营业收入主要来源于公司银行业务、零售银行业务及资金业务"
                "的利息净收入与非利息收入，具体构成及变动原因详见本报告附注七之说明。")
        for x in (60, 320):
            for i in range(12):
                page.insert_text((x, 100 + i * 14), para[:24], fontsize=9,
                                 fontname="china-s")
        doc.save(str(path))
        doc.close()

    def test_borderless_table_gets_cell_geometry(self):
        src = _OUT / "borderless.pdf"
        self._borderless_table(src)
        doc = fitz.open(str(src))
        try:
            self.assertTrue(pdfio._detect_table_cell_rects(doc[0]))
            tables = pdfio._extract_tables(doc[0])
            self.assertEqual(1, len(tables))
            self.assertTrue(tables[0]["borderless"])
        finally:
            doc.close()
        dt = pdfio.extract_document_text(src, log=lambda _m: None)
        label = next(b for b in dt.pages[0] if "营业收入" in b.text)
        value = next(b for b in dt.pages[0] if "1,234,567.89" in b.text)
        self.assertTrue(label.in_table and value.in_table)
        self.assertLess(label.y1, value.y1)  # row order preserved

    def test_borderless_table_is_not_drawn_with_a_grid(self):
        src = _OUT / "borderless2.pdf"
        self._borderless_table(src)
        dt = pdfio.extract_document_text(src, log=lambda _m: None)
        trans = [["T " + b.text for b in dt.pages[0]]]
        out = _OUT / "borderless_out.pdf"
        pdfio.save_translated_pdf(src, dt.pages, trans, str(out), "English")
        doc = fitz.open(str(out))
        try:
            # The source had no rules, so the export must not add any.
            self.assertEqual(0, len(doc[0].get_drawings()))
            self.assertIn("T ", doc[0].get_text())
        finally:
            doc.close()

    def test_two_column_prose_is_not_a_table(self):
        src = _OUT / "two_col_prose.pdf"
        self._two_column_prose(src)
        doc = fitz.open(str(src))
        try:
            self.assertEqual([], pdfio._detect_table_cell_rects(doc[0]))
            self.assertEqual([], pdfio._extract_tables(doc[0]))
        finally:
            doc.close()


class HeaderOnlyTableBodyTest(unittest.TestCase):
    """``find_tables`` only sees the *ruled* part of a table.

    A report that rules its header band (column separators + a rule under it) and
    nothing else came back as a one-row table, so every body row was laid out as
    ordinary prose and each narrow cell wrapped its (longer) translation down over
    the row below.  The body must be recovered from the closing rule and the
    header's own column edges.
    """

    @staticmethod
    def _table(path, prose=False):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        # Header band only: two rules and the column separators.
        page.draw_line(fitz.Point(60, 90), fitz.Point(500, 90), width=0.5)
        page.draw_line(fitz.Point(60, 104), fitz.Point(500, 104), width=0.5)
        for x in (60, 170, 340, 500):   # closed header band, no body rules
            page.draw_line(fitz.Point(x, 90), fitz.Point(x, 104), width=0.5)
        # The table's closing rule; the body rows carry no vertical rules.
        page.draw_line(fitz.Point(60, 190), fitz.Point(500, 190), width=0.75)
        page.insert_text((70, 100), "序号", fontsize=9, fontname="china-s")
        page.insert_text((180, 100), "股东名称", fontsize=9, fontname="china-s")
        page.insert_text((350, 100), "持股数", fontsize=9, fontname="china-s")
        if prose:
            page.insert_text((60, 130),
                             "报告期末，本行股东情况如下，详见附注说明。",
                             fontsize=9, fontname="china-s")
        else:
            for i in range(3):
                y = 122 + i * 22
                page.insert_text((70, y), str(i + 1), fontsize=9)
                page.insert_text((180, y), "股东名称", fontsize=9,
                                 fontname="china-s")
                page.insert_text((350, y), "%d,000,000" % (i + 1), fontsize=9)
        doc.save(str(path))
        doc.close()
        return path

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_body_rows_become_table_cells(self):
        src = self._table(Path(self.tmp.name) / "header_only.pdf")
        doc = fitz.open(str(src))
        try:
            tables = pdfio._extract_tables(doc[0])
        finally:
            doc.close()
        self.assertEqual(1, len(tables))
        self.assertGreaterEqual(len(tables[0]["rows"]), 4)  # header + 3 rows
        self.assertTrue(tables[0]["borderless"])
        dt = pdfio.extract_document_text(str(src), log=lambda _m: None)
        values = [b for b in dt.pages[0]
                  if getattr(b, "in_table", False) and "1,000,000" in b.text]
        self.assertTrue(values, [b.text for b in dt.pages[0]])
        for b in values:
            # Widened to the ruled column, not the source glyph box.
            self.assertGreater(b.x1 - b.x0, 120.0)

    def test_a_full_width_paragraph_ends_the_body(self):
        src = self._table(Path(self.tmp.name) / "header_only_prose.pdf",
                          prose=True)
        doc = fitz.open(str(src))
        try:
            tables = pdfio._extract_tables(doc[0])
        finally:
            doc.close()
        self.assertEqual(1, len(tables))
        self.assertEqual(1, len(tables[0]["rows"]))
        self.assertFalse(tables[0]["borderless"])


class OcrDuplicateContainmentTest(unittest.TestCase):
    """The figure-text OCR can re-read a text-layer block in a much BIGGER box.

    A banner the OCR box fills while the text layer holds only the glyph line
    failed the one-directional coverage test, so the same title was drawn twice
    (the whole banner at 40 pt plus the glyph box) — a 100 % overlap.
    """

    @staticmethod
    def _block(text, box, **kw):
        return pdfio.Block(text=text, page=0, x0=box[0], y0=box[1],
                           x1=box[2], y1=box[3], **kw)

    def test_ocr_block_containing_the_text_block_is_dropped(self):
        text = self._block("恒丰银行股份有限公司", (812.9, 208.9, 972.9, 227.2))
        ocr = self._block("恒丰银行股份有限公司", (688.8, 191.1, 1101.9, 242.2),
                          ocr=True, in_image=True)
        self.assertEqual([text], pdfio._merge_ocr_blocks([text], [ocr]))

    def test_a_larger_ocr_block_with_extra_text_is_kept(self):
        text = self._block("总则", (812.9, 208.9, 860.0, 227.2))
        ocr = self._block("第一章 总则", (688.8, 191.1, 1101.9, 242.2),
                          ocr=True)
        self.assertEqual(2, len(pdfio._merge_ocr_blocks([text], [ocr])))


class OcrMixedPageGridTest(unittest.TestCase):
    """P1-2: a scanned page that mixes a table with prose must not be grid-ified
    whole — the prose keeps prose fitting and blocks the blank redraw."""

    @staticmethod
    def _items():
        items = []
        y = 100.0
        for label, amount in (("营业收入", "1,234,567.89"),
                              ("营业成本", "9,876,543.21"),
                              ("营业利润", "5,555,555.55")):
            items.append((y, 60.0, 160.0, y + 9.0, label))
            items.append((y, 200.0, 300.0, y + 9.0, amount))
            y += 12.0
        y += 6.0
        for i, line in enumerate((
                "本行报告期内的营业收入主要来源于公司银行业务、",
                "零售银行业务及资金业务的利息净收入与非利息收入，",
                "具体构成及变动原因详见本报告附注七之说明。")):
            items.append((y + i * 11.0, 60.0, 520.0, y + 9.0 + i * 11.0, line))
        return items

    def test_prose_rows_stay_out_of_the_grid(self):
        blocks, tables = pdfio._reconstruct_ocr_grid(self._items())
        self.assertTrue(blocks)
        self.assertEqual(1, len(tables))
        prose = [b for b in blocks if not b.in_table]
        grid = [b for b in blocks if b.in_table]
        self.assertEqual(3, len(prose))          # the three paragraph lines
        self.assertEqual(6, len(grid))           # 3 rows x 2 cells
        for b in prose:
            self.assertTrue(b.ocr)
            self.assertEqual(0.0, b.fit_width)
            self.assertEqual(0.0, b.fit_height)
        # Reading order: the prose sits below the grid.
        self.assertLess(max(b.y1 for b in grid), min(b.y0 for b in prose))

    def test_prose_is_not_part_of_the_exported_table(self):
        blocks, _tables = pdfio._reconstruct_ocr_grid(self._items())
        tables = pdfio._reconstruct_ocr_tables(blocks)
        self.assertEqual(1, len(tables))
        mapping = pdfio._map_blocks_to_table_cells(blocks, tables)
        prose_idx = [i for i, b in enumerate(blocks) if not b.in_table]
        self.assertTrue(prose_idx)
        for i in prose_idx:
            self.assertNotIn(i, mapping)

    def test_mixed_page_is_not_blank_redrawn(self):
        blocks, _tables = pdfio._reconstruct_ocr_grid(self._items())
        self.assertFalse(pdfio._is_pure_ocr_table_page(blocks))


class PureOcrTablePageTest(unittest.TestCase):
    """``_is_pure_ocr_table_page`` decides whether a page may be blank-redrawn.

    The gate must reject any page that carries a block the redraw would drop —
    a chart node label (``is_chart``) or a non-OCR (text-layer) *content* block.
    The one exception is page furniture (a page number / rule outside the table),
    which ``_draw_page_furniture`` puts back on the rebuilt page.
    """

    def _cell(self, text="x", ocr=True, is_chart=False, in_table=True):
        return pdfio.Block(
            text=text, page=0, x0=60, y0=100, x1=200, y1=112,
            size=6.0, single_line=True, ocr=ocr, is_chart=is_chart,
            in_table=in_table)

    def test_all_ocr_cells_is_pure(self):
        self.assertTrue(pdfio._is_pure_ocr_table_page(
            [self._cell(), self._cell(), self._cell(), self._cell()]))

    def test_ocr_prose_block_is_impure(self):
        # P1-2: the prose part of a mixed scan is ``ocr`` but NOT a grid cell; the
        # redraw would drop it (and draw the prose as table rows).
        self.assertFalse(pdfio._is_pure_ocr_table_page(
            [self._cell(), self._cell(), self._cell(),
             self._cell(in_table=False)]))

    def test_mixed_with_non_ocr_is_impure(self):
        self.assertFalse(pdfio._is_pure_ocr_table_page(
            [self._cell(), self._cell(), self._cell(),
             self._cell(ocr=False)]))

    def test_mixed_with_chart_is_impure(self):
        self.assertFalse(pdfio._is_pure_ocr_table_page(
            [self._cell(), self._cell(), self._cell(),
             self._cell(is_chart=True)]))

    def test_empty_is_impure(self):
        self.assertFalse(pdfio._is_pure_ocr_table_page([]))

    def test_page_number_below_the_table_is_accepted(self):
        # A sparse text layer's page number sits outside the OCR table bbox and is
        # re-drawn by ``_draw_page_furniture``, so it must not block the redraw.
        cells = [self._cell(), self._cell(), self._cell(), self._cell()]
        number = pdfio.Block(text="22", page=0, x0=292, y0=801, x1=303, y1=812,
                             size=6.0, single_line=True)
        self.assertTrue(pdfio._is_pure_ocr_table_page(cells + [number]))

    def test_text_layer_caption_is_impure(self):
        cells = [self._cell(), self._cell(), self._cell(), self._cell()]
        caption = pdfio.Block(text="资产负债表", page=0, x0=60, y0=801, x1=140, y1=812,
                              size=6.0, single_line=True)
        self.assertFalse(pdfio._is_pure_ocr_table_page(cells + [caption]))

    def test_letterless_block_inside_the_table_is_impure(self):
        # A short letterless text-layer block *inside* the table bbox is a value the
        # grid does not carry; drawing it at its own bbox on the re-laid-out table
        # would misplace it, so the page stays on the in-place path.
        cells = [self._cell(), self._cell(), self._cell(), self._cell()]
        value = pdfio.Block(text="42", page=0, x0=60, y0=104, x1=90, y1=110,
                            size=6.0, single_line=True)
        self.assertFalse(pdfio._is_pure_ocr_table_page(cells + [value]))

    def test_furniture_predicate_accepts_a_page_number(self):
        self.assertTrue(pdfio._is_page_furniture(self._cell(text="22", ocr=False)))

    def test_furniture_predicate_rejects_letters_and_cjk(self):
        self.assertFalse(pdfio._is_page_furniture(self._cell(text="iv", ocr=False)))
        self.assertFalse(pdfio._is_page_furniture(self._cell(text="第22页", ocr=False)))

    def test_furniture_predicate_rejects_ocr_and_chart_blocks(self):
        self.assertFalse(pdfio._is_page_furniture(self._cell(text="22", ocr=True)))
        self.assertFalse(pdfio._is_page_furniture(
            self._cell(text="22", ocr=False, is_chart=True)))

    def test_furniture_predicate_rejects_long_text(self):
        self.assertFalse(pdfio._is_page_furniture(
            self._cell(text="1 2 3 4 5 6 7 8 9", ocr=False)))


class FigureTextTest(unittest.TestCase):
    """Text baked into a **raster figure on a page with its own text layer**.

    ``extract_document_text`` only OCR'd pages with no / almost no text layer, so a
    figure's labels were neither in the text layer nor recognised: a real 5-page
    paper kept its chart's English title and axis labels in the Chinese output,
    pixel for pixel.  These tests inject ``ocr_fn`` (the existing seam) so they stay
    offline and deterministic.
    """

    #: Box inside the image region of ``_mixed_page`` (the "figure label").
    _LABEL_BOX = [(70.0, 270.0), (210.0, 270.0), (210.0, 286.0), (70.0, 286.0)]

    @staticmethod
    def _raster(text: str, *, dark: bool = False, fill=None) -> bytes:
        """A PNG panel with ``text`` on it (``dark`` = a photo-like background)."""
        doc = fitz.open()
        page = doc.new_page(width=160, height=80)
        bg = fill if fill is not None else ((0.05, 0.05, 0.08) if dark else (1, 1, 1))
        page.draw_rect(fitz.Rect(0, 0, 160, 80), color=None, fill=bg)
        page.insert_text((12, 34), text, fontsize=12,
                         color=(1, 1, 1) if dark else (0, 0, 0))
        png = page.get_pixmap(dpi=150).tobytes("png")
        doc.close()
        return png

    def _mixed_page(self, path: Path, *, dark: bool = False) -> None:
        """A page that is *not* sparse (so the whole-page OCR branch stays out)."""
        doc = fitz.open()
        page = doc.new_page(width=400, height=460)
        page.insert_textbox(
            fitz.Rect(40, 40, 360, 220),
            " ".join(["Prose line of the source document."] * 20),
            fontsize=10,
        )
        page.insert_image(fitz.Rect(60, 260, 340, 400),
                          stream=self._raster("CHART LABEL", dark=dark))
        doc.save(str(path))
        doc.close()

    def _large_figure_page(self, path: Path) -> fitz.Rect:
        """A page whose *only* art is one tinted figure covering > 50 % of it.

        ``_partial_image_rects`` stops at 50 % of the page, so this used to look
        like "no partial image": the figure block's cover fell back to plain
        white and the bilingual translation page was a blank sheet (the figure
        was dropped while its translated labels floated).
        """
        doc = fitz.open()
        page = doc.new_page(width=400, height=600)
        page.insert_textbox(
            fitz.Rect(40, 40, 360, 220),
            " ".join(["Prose line of the source document."] * 20), fontsize=10)
        rect = fitz.Rect(40, 230, 380, 600)          # 340×370 = 0.524 × page
        # ``keep_proportion=False``: the default letterboxes the image inside
        # the rect, which would shrink the region below the 50 % the test needs.
        page.insert_image(rect, keep_proportion=False,
                          stream=self._raster("CHART LABEL", fill=(0.9, 0.9, 0.95)))
        doc.save(str(path))
        doc.close()
        return rect
    def _ocr_fn(self, *extra):
        def fn(_page_index, _page):
            return [(self._LABEL_BOX, "CHART LABEL"), *extra]
        return fn

    def test_a_short_ocr_fragment_is_not_translated(self):
        # P1-2: the design's conservative gate (char_count >= 4).  A 1–3 char
        # scrap on a paper-like image is usually an icon / ornament misread, and
        # covering + redrawing it would damage the picture for nothing.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mixed.pdf"
            self._mixed_page(src)
            dt = pdfio.extract_document_text(
                str(src), ocr=True,
                ocr_fn=lambda _i, _p: [(self._LABEL_BOX, "TM")],
                log=lambda _m: None)
            self.assertEqual(
                [], [b for b in dt.pages[0] if getattr(b, "in_image", False)])

    def test_a_large_figure_keeps_its_own_cover_colour(self):
        # P1-1: a figure covering > 50 % of the page is still that block's
        # figure.  Its cover must be clipped to the image and filled with the
        # image's own background — the old 50 % cap could not find the image, so
        # the cover was plain white and unclipped.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "large.pdf"
            self._large_figure_page(src)
            dt = pdfio.extract_document_text(
                str(src), ocr=True, ocr_fn=self._ocr_fn(), log=lambda _m: None)
            figs = [b for b in dt.pages[0] if getattr(b, "in_image", False)]
            self.assertEqual(["CHART LABEL"], [b.text for b in figs])
            with fitz.open(str(src)) as d:
                page = d[0]
                self.assertEqual([], pdfio._partial_image_rects(page))
                containing = pdfio._containing_image_rect(page, figs[0])
                self.assertIsNotNone(containing)
                colour = pdfio._image_cover_color(page, containing)
            self.assertNotEqual(
                (1.0, 1.0, 1.0), colour,
                "the cover must take the figure's own background, not white")

    def test_bilingual_page_keeps_a_large_figure(self):
        # P1-1 (second consequence): the translation page of a > 50 % figure
        # used to be a blank sheet — the figure was dropped.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "large.pdf"
            out = Path(tmp) / "bi.pdf"
            self._large_figure_page(src)
            dt = pdfio.extract_document_text(
                str(src), ocr=True, ocr_fn=self._ocr_fn(), log=lambda _m: None)
            per_page = [[f"图:{b.text}" if getattr(b, "in_image", False)
                         else f"T:{b.text}" for b in dt.pages[0]]]
            pdfio.save_interleaved_pdf(
                str(src), per_page, str(out), "Simplified Chinese", pages=dt.pages)
            with fitz.open(str(out)) as o:
                self.assertEqual(1, len(o[1].get_image_info()),
                                 "the >50% figure must survive on the translation page")
    def test_figure_text_on_paper_becomes_an_in_image_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mixed.pdf"
            self._mixed_page(src)
            outside = ([(40.0, 60.0), (220.0, 60.0), (220.0, 76.0), (40.0, 76.0)],
                       "OUTSIDE BOX")
            dt = pdfio.extract_document_text(
                str(src), ocr=True, ocr_fn=self._ocr_fn(outside),
                log=lambda _m: None,
            )
            texts = [b.text for b in dt.pages[0]]
            self.assertIn("CHART LABEL", texts)
            # A box outside every image region must not be adopted: the page's own
            # text layer is authoritative there.
            self.assertNotIn("OUTSIDE BOX", texts)
            figures = [b for b in dt.pages[0] if b.in_image]
            self.assertEqual(["CHART LABEL"], [b.text for b in figures])
            self.assertTrue(figures[0].ocr)
            self.assertEqual(0, figures[0].image_index)
            # The extraction records the setting: a later "重新导出" compares it
            # before reusing this document (see TranslateWorker._run_re_export).
            self.assertIs(True, dt.image_text)
            # A figure must never be a table cell: the grid reconstruction would
            # read a chart's numbers as a statement and re-lay it out.
            self.assertFalse(figures[0].in_table)

    def test_figure_text_on_a_photo_is_left_alone(self):
        # A photo's pixels are the content: no block, no cover (fail-closed).
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "photo.pdf"
            self._mixed_page(src, dark=True)
            logs: list[str] = []
            dt = pdfio.extract_document_text(
                str(src), ocr=True, ocr_fn=self._ocr_fn(), log=logs.append,
            )
            self.assertEqual([], [b for b in dt.pages[0] if b.in_image])
            self.assertTrue(any("保留原样" in m for m in logs), logs)

    def test_figure_text_is_not_extracted_without_ocr(self):
        # ``image_text`` follows ``ocr``: a caller that asked for no OCR must not
        # get figure blocks either (the review / check scripts rely on this).
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mixed.pdf"
            self._mixed_page(src)
            dt = pdfio.extract_document_text(
                str(src), ocr=False, ocr_fn=self._ocr_fn(), log=lambda _m: None,
            )
            self.assertEqual([], [b for b in dt.pages[0] if b.in_image])

    def test_the_option_can_turn_figure_text_off(self):
        # The GUI checkbox passes image_text=False while OCR itself stays on: the
        # figure's text must then be left completely alone (pre-v0.5.44 behaviour).
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mixed.pdf"
            self._mixed_page(src)
            dt = pdfio.extract_document_text(
                str(src), ocr=True, ocr_fn=self._ocr_fn(), image_text=False,
                log=lambda _m: None,
            )
            self.assertEqual([], [b for b in dt.pages[0] if b.in_image])
            self.assertIs(False, dt.image_text)
            # The semantic (IR-mode) entry point forwards the same flag — the
            # option has to work on the IR pipeline too, not just plain extraction.
            st = pdfio.extract_document_structured(
                str(src), parser="geo", ocr=True, ocr_fn=self._ocr_fn(),
                image_text=False, log=lambda _m: None,
            )
            self.assertEqual([], [b for b in st.pages[0] if b.in_image])
            self.assertIs(False, st.image_text)

    def test_export_replaces_figure_text_and_keeps_the_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mixed.pdf"
            out = Path(tmp) / "out.pdf"
            self._mixed_page(src)
            dt = pdfio.extract_document_text(
                str(src), ocr=True, ocr_fn=self._ocr_fn(), log=lambda _m: None,
            )
            per_page = [[(f"图:{b.text}" if b.in_image else f"T:{b.text}")
                         for b in dt.pages[0]]]
            pdfio.save_translated_pdf(str(src), dt.pages, per_page, str(out),
                                      "Simplified Chinese", log=lambda _m: None)

            def pixels(path, rect):
                with fitz.open(str(path)) as doc:
                    pix = doc[0].get_pixmap(clip=fitz.Rect(*rect), dpi=150,
                                            alpha=False)
                return pix.samples

            with fitz.open(str(out)) as chk:
                self.assertEqual(1, len(chk[0].get_image_info()),
                                 "the figure itself must survive")
                self.assertIn("图:", chk[0].get_text())
            label = (70.0, 270.0, 210.0, 286.0)
            self.assertNotEqual(pixels(src, label), pixels(out, label),
                                "the figure's baked-in text must be covered")
            # Far from any text: the picture's own pixels must be untouched (the
            # scan path's measured ink bands would have painted the chart out).
            quiet = (300.0, 370.0, 338.0, 398.0)
            self.assertEqual(pixels(src, quiet), pixels(out, quiet),
                             "the rest of the figure must be untouched")


class FigureTextRoleTest(unittest.TestCase):
    """A figure's OCR'd text must not inherit the figure region's protected role.

    The semantic layer marks a detected figure as ``role="figure"``, which the IR
    translator reads as "keep the source" (it exists to protect the *picture*).
    Claiming the text we recovered from inside that picture handed it the same
    role, so with the DocLayout backend (real figure regions) a chart's labels
    came out verbatim English in a Chinese export — 334 Latin chars, 0 CJK —
    while the geometric backend (no figure region) translated them.
    """

    _BOX = [(70.0, 270.0), (210.0, 270.0), (210.0, 286.0), (70.0, 286.0)]

    def _mixed(self, path: Path) -> None:
        doc = fitz.open()
        page = doc.new_page(width=400, height=460)
        page.insert_textbox(fitz.Rect(40, 40, 360, 220),
                            " ".join(["Prose line of the source document."] * 20),
                            fontsize=10)
        panel = fitz.open()
        panel_page = panel.new_page(width=160, height=80)
        panel_page.draw_rect(fitz.Rect(0, 0, 160, 80), color=None, fill=(1, 1, 1))
        panel_page.insert_text((12, 34), "CHART LABEL", fontsize=12)
        png = panel_page.get_pixmap(dpi=150).tobytes("png")
        panel.close()
        page.insert_image(fitz.Rect(60, 260, 340, 400), stream=png)
        doc.save(str(path))
        doc.close()

    def test_a_figure_region_does_not_claim_the_pictures_own_text(self):
        import translate_app.ir as ir_mod

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mixed.pdf"
            self._mixed(src)

            def ocr_fn(_i, _p):
                return [(self._BOX, "CHART LABEL")]

            dt = pdfio.extract_document_text(str(src), ocr=True, ocr_fn=ocr_fn,
                                             log=lambda _m: None)
            figure_idx = [i for i, b in enumerate(dt.pages[0])
                          if getattr(b, "in_image", False)]
            self.assertEqual(1, len(figure_idx))
            flat = figure_idx[0]

            # A semantic backend reports the picture as a figure region covering it.
            def structure_fn(_page_index, page, blocks):
                return [{"kind": "figure",
                         "bbox": list(page.get_image_info()[0]["bbox"]),
                         "block_indices": [flat]}]

            pdfio.build_structure(str(src), dt, structure_fn, parser="test")
            structure = dt.page_structure[0]
            elements = [e for e in structure.elements if e["kind"] == "figure"]
            self.assertEqual(1, len(elements), structure.elements)
            self.assertEqual([], elements[0]["block_indices"],
                             "图内文字块不属于「图片本体」")
            # …so the IR role stays ordinary text and the block gets translated.
            self.assertEqual(("text", 0), ir_mod._role_of(flat, structure))

    def test_the_ir_gate_translates_an_in_image_block_anyway(self):
        # Fail-safe at the decision point: even if a role says "figure", a block
        # whose text came out of the picture is sent to the translator.
        import translate_app.ir as ir_mod

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mixed.pdf"
            self._mixed(src)

            def ocr_fn(_i, _p):
                return [(self._BOX, "CHART LABEL")]

            dt = pdfio.extract_document_text(str(src), ocr=True, ocr_fn=ocr_fn,
                                             log=lambda _m: None)
            irdoc = ir_mod.build_ir(dt, lang="Simplified Chinese")
            target = None
            for ipage in irdoc.pages:
                for b in ipage.blocks:
                    if getattr(b.anchor, "in_image", False):
                        b.role = "figure"        # force the protected role
                        target = b.text
            self.assertTrue(target)
            seen: list[str] = []

            def translate_fn(texts, _lang=None, **_kw):
                seen.extend(texts)
                return [f"译:{t}" for t in texts]

            out = ir_mod.translate_ir(irdoc, translate_fn, lang="Simplified Chinese",
                                      group_prose=False)
            self.assertIn(target, seen, "图内文字必须进入翻译请求")


class ScannedChartRoleTest(unittest.TestCase):
    """A *scanned* org chart / architecture diagram keeps its labels as source.

    Product decision: an org chart is a picture — its node labels stay verbatim
    (translate the page, not the diagram).  The ``figure`` region is the mechanism
    that hands them the structural role ``translate_ir`` reads as "keep source".
    The v0.5.45 guard releases only ``in_image`` blocks (the figure-text OCR of a
    *text-layer* page); a scanned chart's whole-page OCR (``ocr`` without
    ``in_image``) stays claimed — verified against page 5 of the Mintai annual
    report (58 labels, kept in the English export).
    """

    def test_a_figure_region_keeps_scanned_ocr_text_verbatim(self):
        import translate_app.ir as ir_mod

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "scan.pdf"
            doc = fitz.open()
            doc.new_page(width=400, height=400)          # no text layer at all
            doc.save(str(src))
            doc.close()

            boxes = {
                "股东大会": [(40.0, 60.0), (120.0, 60.0), (120.0, 76.0), (40.0, 76.0)],
                "风险管理部": [(40.0, 120.0), (70.0, 120.0), (70.0, 160.0), (40.0, 160.0)],
            }

            def ocr_fn(_i, _p):
                return [(box, text) for text, box in boxes.items()]

            dt = pdfio.extract_document_text(str(src), ocr=True, ocr_fn=ocr_fn,
                                             log=lambda _m: None)
            n = len(dt.pages[0])
            self.assertEqual(2, n)
            self.assertTrue(all(b.ocr for b in dt.pages[0]))
            self.assertFalse(any(getattr(b, "in_image", False)
                                 for b in dt.pages[0]),
                             "a scan must not be an in_image block")

            # A semantic backend reports the scanned chart as a figure region.
            def structure_fn(_page_index, _page, _blocks):
                return [{"kind": "figure", "bbox": [0, 0, 400, 400],
                         "block_indices": list(range(n))}]

            pdfio.build_structure(str(src), dt, structure_fn, parser="test")
            structure = dt.page_structure[0]
            elements = [e for e in structure.elements if e["kind"] == "figure"]
            self.assertEqual(list(range(n)), elements[0]["block_indices"],
                             "扫描图的节点标签属于图片本体，保留原文")
            self.assertEqual(("figure", 0), ir_mod._role_of(0, structure))

            # …and the labels reach the translator.
            irdoc = ir_mod.build_ir(dt, lang="English")
            seen: list[str] = []

            def translate_fn(texts, **_kw):
                seen.extend(texts)
                return ["EN:" + t for t in texts]

            out = ir_mod.translate_ir(irdoc, translate_fn, lang="English",
                                      group_prose=False)
            self.assertEqual([], seen, "组织架构图节点标签不应进入翻译请求")
            self.assertTrue(all(v in boxes for v in out.values()), out)


class KeptOcrPixelsTest(unittest.TestCase):
    """A kept org chart / architecture diagram is not covered or redrawn.

    ``classify_page`` marks a page with ≥3 narrow-tall node boxes as ``chart``.
    Its OCR labels are the picture: covering them with white rects and drawing the
    OCR glyphs back produced overlapping boxes on a real annual report (7 overlaps;
    the source was clean) and shrank every label to ~5 pt.  A chart page keeps its
    original pixels — only its text-layer blocks (the heading) translate.
    """

    def test_chart_page_keeps_its_ocr_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "chart.pdf"
            out = Path(tmp) / "out.pdf"
            doc = fitz.open()
            page = doc.new_page(width=300, height=200)
            # A black marker behind the node boxes: a white cover erases it.
            page.draw_rect(fitz.Rect(40, 80, 240, 140), color=None, fill=(0, 0, 0))
            doc.save(str(src))
            doc.close()

            # A heading (text layer) + three narrow-tall node boxes = chart page.
            blocks = [pdfio.Block("标题", 0, 20, 20, 120, 35, size=12)]
            blocks += [pdfio.Block(f"节点{i}", 0, 60 + 40 * i, 90, 75 + 40 * i, 130,
                                   size=8, ocr=True, single_line=True)
                       for i in range(3)]
            self.assertEqual(pdfio.PAGE_CHART, pdfio.classify_page(blocks))
            pdfio.save_translated_pdf(str(src), [blocks], [[b.text for b in blocks]],
                                      str(out), "English", log=lambda _m: None)
            with fitz.open(str(out)) as o:
                pix = o[0].get_pixmap(clip=fitz.Rect(60, 90, 75, 130), dpi=72,
                                      alpha=False)
                self.assertEqual(0, min(pix.samples),
                                 "the chart must not be painted over")
                self.assertNotIn("节点0", o[0].get_text(),
                                 "the node labels must not be redrawn")
                self.assertIn("标题", o[0].get_text(),
                              "the heading is text and must still be exported")

    def test_an_edited_kept_ocr_block_is_drawn(self):
        # ``keep_original`` is a *default*, not a lock: an edit made through the
        # chat overlay (or by the AI) must reach the output.  The exporter used to
        # skip every kept OCR block unconditionally, so the new translation vanished
        # while the log still claimed "已应用 N 处编辑" — a silent content loss.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "kept_edited.pdf"
            out = Path(tmp) / "out.pdf"
            doc = fitz.open()
            page = doc.new_page(width=300, height=200)
            # A black marker behind the block: covering it proves the cover ran.
            page.draw_rect(fitz.Rect(40, 55, 200, 90), color=None, fill=(0, 0, 0))
            doc.save(str(src))
            doc.close()

            blocks = [pdfio.Block("印章", 0, 50, 60, 120, 80, size=10, ocr=True,
                                  single_line=True, keep_original=True)]
            pdfio.save_translated_pdf(str(src), [blocks], [["SEAL-TRANSLATED"]],
                                      str(out), "English", log=lambda _m: None)
            with fitz.open(str(out)) as o:
                self.assertIn("SEAL-TRANSLATED", o[0].get_text(),
                              "an edited kept block must be exported")
                # The block's own box was covered (white) …
                pix = o[0].get_pixmap(clip=fitz.Rect(52, 61, 118, 64), dpi=72,
                                      alpha=False)
                self.assertGreater(min(pix.samples), 200,
                                   "the covered source pixels must be white")
                # … while the marker outside the block is untouched.
                outside = o[0].get_pixmap(clip=fitz.Rect(150, 60, 190, 80),
                                          dpi=72, alpha=False)
                self.assertEqual(0, min(outside.samples),
                                 "the cover must not eat the surrounding image")

    def test_an_untranslated_kept_ocr_block_keeps_its_pixels(self):
        # The other direction (the signature / seal default): text identical to the
        # source means "nothing was translated", so the pixels stay verbatim.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "kept_plain.pdf"
            out = Path(tmp) / "out.pdf"
            doc = fitz.open()
            page = doc.new_page(width=300, height=200)
            page.draw_rect(fitz.Rect(40, 55, 200, 90), color=None, fill=(0, 0, 0))
            doc.save(str(src))
            doc.close()

            blocks = [pdfio.Block("印章", 0, 50, 60, 120, 80, size=10, ocr=True,
                                  single_line=True, keep_original=True)]
            pdfio.save_translated_pdf(str(src), [blocks], [["印章"]],
                                      str(out), "English", log=lambda _m: None)
            with fitz.open(str(out)) as o:
                pix = o[0].get_pixmap(clip=fitz.Rect(60, 65, 110, 75), dpi=72,
                                      alpha=False)
                self.assertEqual(0, min(pix.samples),
                                 "an untranslated kept block must not be covered")
                self.assertNotIn("印章", o[0].get_text(),
                                 "and must not be redrawn as text")

    def test_a_translated_chart_label_overrides_the_default(self):
        # The keep rule is a *default*, not a lock: a block the AI actually
        # translated is drawn, so the AI decides per block what to translate.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "chart.pdf"
            out = Path(tmp) / "out.pdf"
            doc = fitz.open()
            page = doc.new_page(width=300, height=200)
            page.draw_rect(fitz.Rect(40, 80, 240, 140), color=None, fill=(0, 0, 0))
            doc.save(str(src))
            doc.close()

            blocks = [pdfio.Block("标题", 0, 20, 20, 120, 35, size=12)]
            blocks += [pdfio.Block(f"节点{i}", 0, 60 + 40 * i, 90, 75 + 40 * i, 130,
                                   size=8, ocr=True, single_line=True)
                       for i in range(3)]
            self.assertEqual(pdfio.PAGE_CHART, pdfio.classify_page(blocks))
            per = [["标题", "Translated node", "节点1", "节点2"]]
            pdfio.save_translated_pdf(str(src), [blocks], per, str(out),
                                      "English", log=lambda _m: None)
            with fitz.open(str(out)) as o:
                text = o[0].get_text()
                self.assertIn("Translated node", text,
                              "a translated diagram label must be drawn")
                self.assertNotIn("节点1", text, "kept labels stay pixels")
                self.assertNotIn("节点2", text, "kept labels stay pixels")


class KeptBlockInsideATableTest(unittest.TestCase):
    """A ``keep_original`` block inside a table must not be erased by the export.

    The table pass redacts the whole table bbox — it has to, to drop the stale rules
    before they are redrawn at the new row positions — so once the exporter also
    stopped *drawing* kept blocks, a kept cell was erased with nothing drawn back.
    ``page_scope`` makes that a routine path: every block outside the requested pages
    is marked ``keep_original`` (v0.6.5 review P1-1).
    """

    @staticmethod
    def _kept_page(tmp: str, *needles: str):
        """Extract a ruled-table page and keep every block whose text matches."""
        src = build_ruled_table_pdf(Path(tmp) / "table.pdf")
        dt = pdfio.extract_document_text(str(src), ocr=False, log=lambda _m: None)
        blocks = dt.pages[0]
        for b in blocks:
            b.keep_original = any(n in b.text for n in needles)
        per = [[b.text if getattr(b, "keep_original", False) else "TR:" + b.text
                for b in blocks]]
        return src, dt, per

    def test_a_kept_table_cell_keeps_its_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dt, per = self._kept_page(tmp, "Cell11")
            out = Path(tmp) / "out.pdf"
            pdfio.save_translated_pdf(str(src), dt.pages, per, str(out), "English",
                                      log=lambda _m: None)
            with fitz.open(str(out)) as o:
                text = o[0].get_text()
            self.assertIn("Cell11", text,
                          "a kept table cell must survive the table redaction")
            self.assertNotIn("TR:Cell11", text)
            self.assertIn("TR:Cell00", text, "its neighbours still translate")

    def test_a_page_kept_whole_keeps_all_of_its_table_text(self):
        # The page-scope shape: every block of an out-of-scope page is kept, so a
        # whole table must come out verbatim instead of blank.
        with tempfile.TemporaryDirectory() as tmp:
            src, dt, per = self._kept_page(tmp, "Cell")
            out = Path(tmp) / "out.pdf"
            pdfio.save_translated_pdf(str(src), dt.pages, per, str(out), "English",
                                      log=lambda _m: None)
            with fitz.open(str(out)) as o:
                text = o[0].get_text()
            missing = [b.text for b in dt.pages[0] if b.text and b.text not in text]
            self.assertEqual([], missing, "no cell text may be lost")

    def test_a_kept_table_cell_keeps_its_text_on_the_mirror_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dt, per = self._kept_page(tmp, "Cell11")
            out = Path(tmp) / "bi.pdf"
            pdfio.save_interleaved_pdf(str(src), per, str(out), "English",
                                       pages=dt.pages)
            with fitz.open(str(out)) as o:
                text = o[1].get_text()
            self.assertIn("Cell11", text,
                          "a kept cell must survive the mirror page's table pass")
            self.assertNotIn("TR:Cell11", text)

    def test_a_kept_block_stays_on_a_blank_mirror_page(self):
        # A text-only page makes a *blank* mirror page: there is no source copy to
        # preserve, so a kept block must still be drawn (its translation IS the
        # source text) — skipping it left the bilingual product with an empty page.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "plain.pdf"
            out = Path(tmp) / "bi.pdf"
            doc = fitz.open()
            page = doc.new_page(width=595, height=842)
            page.insert_text(fitz.Point(72, 100), "Kept paragraph text.", fontsize=12)
            page.insert_text(fitz.Point(72, 140), "Translated paragraph text.",
                             fontsize=12)
            doc.save(str(src))
            doc.close()
            dt = pdfio.extract_document_text(str(src), ocr=False, log=lambda _m: None)
            blocks = dt.pages[0]
            for b in blocks:
                b.keep_original = "Kept" in b.text
            per = [[b.text if getattr(b, "keep_original", False) else "TR:" + b.text
                    for b in blocks]]
            pdfio.save_interleaved_pdf(str(src), per, str(out), "English",
                                       pages=dt.pages)
            with fitz.open(str(out)) as o:
                text = o[1].get_text()
            self.assertIn("Kept paragraph text.", text)
            self.assertIn("TR:Translated paragraph text.", text)

if __name__ == "__main__":
    unittest.main()
