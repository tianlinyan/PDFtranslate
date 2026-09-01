"""Tests for the OCR extraction plumbing (no real model inference)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import pymupdf as fitz

from translate_app import pdfio
from translate_app.translator import TranslationCancelled


class _FakePage:
    """Minimal stand-in for a fitz page used by :func:`pdfio._needs_ocr`."""

    def __init__(self, images=(), drawings=()):
        self._images = list(images)
        self._drawings = list(drawings)

    def get_images(self, full=False):  # noqa: ARG002
        return list(self._images)

    def get_drawings(self):
        return list(self._drawings)


def _ocr_fn(page_index, page):  # noqa: ARG001
    """Fake OCR: returns ``[(box, text)]`` already in PDF points."""
    return [
        ([[2.4, 4.8], [26.4, 4.8], [26.4, 21.6], [2.4, 21.6]], "Hello"),
        ([[2.4, 26.4], [48.0, 26.4], [48.0, 43.2], [2.4, 43.2]], "World"),
    ]


class OcrPlumbingTest(unittest.TestCase):
    """Tests independent of the real OCR engine."""

    def test_synthesize_orders_and_flags(self):
        # Supplied out of order to prove reading order is restored.
        blocks = pdfio._synthesize_ocr_blocks(
            [
                ([[2.4, 26.4], [48.0, 26.4], [48.0, 43.2], [2.4, 43.2]], "World"),
                ([[2.4, 4.8], [26.4, 4.8], [26.4, 21.6], [2.4, 21.6]], "Hello"),
            ],
            0,
        )
        self.assertEqual(2, len(blocks))
        self.assertEqual(["Hello", "World"], [b.text for b in blocks])
        self.assertTrue(all(b.ocr for b in blocks))
        self.assertTrue(all(b.page == 0 for b in blocks))
        # size = (y1 - y0) / 1.2 = 16.8 / 1.2 = 14.0
        self.assertAlmostEqual(14.0, blocks[0].size)

    def test_synthesize_cleans_control_chars(self):
        blocks = pdfio._synthesize_ocr_blocks(
            [([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]], "a\x00b\x7f c")], 1
        )
        self.assertEqual(["a b c"], [b.text for b in blocks])

    def test_needs_ocr_detects_image_and_drawing_pages(self):
        self.assertTrue(pdfio._needs_ocr(_FakePage(images=["x"])))
        self.assertTrue(pdfio._needs_ocr(_FakePage(drawings=["x"])))
        self.assertFalse(pdfio._needs_ocr(_FakePage()))  # truly blank

    def test_block_dict_round_trip(self):
        b = pdfio.Block(text="Hi", page=2, x0=1.0, y0=2.0, x1=30.0, y1=10.0, ocr=True)
        d = pdfio._block_to_dict(b)
        back = pdfio._block_from_dict(d)
        self.assertEqual(b, back)
        self.assertTrue(back.ocr)


class OcrExtractionTest(unittest.TestCase):
    """End-to-end wiring: OCR blocks are injected into the extraction result."""

    def _build_vector_page(self) -> Path:
        """A page with a drawing but no text layer -> a non-blank 'scan' page."""
        path = Path(__file__).resolve().parent / f"_ocr_scan_{os.getpid()}.pdf"
        self.addCleanup(path.unlink, missing_ok=True)
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.draw_rect(fitz.Rect(50, 50, 300, 150), color=(0, 0, 0), fill=(1, 1, 1))
        doc.save(str(path))
        doc.close()
        return path

    def test_extract_injects_ocr_blocks_via_ocr_fn(self):
        path = self._build_vector_page()
        on = pdfio.extract_document_text(path, ocr=True, ocr_fn=_ocr_fn)
        off = pdfio.extract_document_text(path, ocr=False)
        self.assertEqual(2, len(on.blocks))
        self.assertEqual(1, on.ocr_count)
        self.assertEqual([0, 0], on.block_pages)
        self.assertTrue(all(b.ocr for b in on.pages[0]))
        self.assertEqual(0, len(off.blocks))
        self.assertEqual(0, off.ocr_count)

    def test_extract_honours_cancel(self):
        path = self._build_vector_page()
        with self.assertRaises(TranslationCancelled):
            pdfio.extract_document_text(path, ocr=True, ocr_fn=_ocr_fn, cancel=lambda: True)
