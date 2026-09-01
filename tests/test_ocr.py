"""Tests for the OCR extraction plumbing (no real model inference)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class _TempOcrCacheMixin:
    """Redirect the OCR cache into a temp dir (never touch the user's home)."""

    def setUp(self):  # noqa: N802
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ocr_cache_dir = Path(tmp.name)
        patcher = mock.patch.dict(
            os.environ, {"PDFTRANSLATE_OCR_CACHE_DIR": str(self.ocr_cache_dir)}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.doc_dir = self.ocr_cache_dir / "docs"
        self.doc_dir.mkdir()
        self._page_seq = 0

    def _build_vector_page(self, pages: int = 1) -> Path:
        """A PDF whose pages carry a drawing but no text layer (a fake 'scan').

        Each call gets its own file name so the per-document OCR cache (keyed by
        path + mtime + size) never leaks between assertions.
        """
        self._page_seq += 1
        path = self.doc_dir / f"scan_{self._page_seq}.pdf"
        doc = fitz.open()
        for _ in range(pages):
            page = doc.new_page(width=595, height=842)
            page.draw_rect(fitz.Rect(50, 50, 300, 150), color=(0, 0, 0), fill=(1, 1, 1))
        doc.save(str(path))
        doc.close()
        return path



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


class OcrExtractionTest(_TempOcrCacheMixin, unittest.TestCase):
    """End-to-end wiring: OCR blocks are injected into the extraction result."""

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

    def test_page_failure_is_logged_and_skipped(self):
        """A failing page degrades to "no text", but never silently: a bare
        ``except`` here is indistinguishable from a scan with nothing on it."""
        path = self._build_vector_page()

        def boom(_page_index, _page):
            raise RuntimeError("engine exploded")

        logs: list[str] = []
        doc = pdfio.extract_document_text(path, ocr=True, ocr_fn=boom, log=logs.append)
        self.assertEqual([], doc.blocks)
        failures = [m for m in logs if "OCR 失败" in m and "engine exploded" in m]
        self.assertEqual(1, len(failures), logs)



class OcrEngineUnavailableTest(_TempOcrCacheMixin, unittest.TestCase):
    """A missing OCR engine must degrade to "skip the page", never crash.

    Regression: ``_OCR_WARNED`` used to be assigned inside
    ``extract_document_text`` without a ``global`` declaration, which made
    Python treat it as a local variable — so reading it raised
    ``UnboundLocalError`` and every run with OCR enabled on a machine without
    ``rapidocr_onnxruntime`` died with a traceback instead of skipping scans.
    """

    def setUp(self):  # noqa: N802
        super().setUp()
        # Simulate "rapidocr not installed" and restore the real state after.
        for name in ("_OCR_ENGINE", "_OCR_FAILED", "_OCR_WARNED"):
            self.addCleanup(setattr, pdfio, name, getattr(pdfio, name))
        pdfio._OCR_ENGINE = None
        pdfio._OCR_FAILED = True
        pdfio._OCR_WARNED = False

    def test_missing_engine_skips_pages_and_warns_once(self):
        path = self._build_vector_page()
        logs: list[str] = []

        # Must not raise: the page is simply left without text.
        doc = pdfio.extract_document_text(path, ocr=True, log=logs.append)
        self.assertEqual([], doc.blocks)
        self.assertEqual(0, doc.ocr_count)
        self.assertEqual([[]], doc.pages)

        warnings = [m for m in logs if "OCR 引擎不可用" in m]
        self.assertEqual(1, len(warnings), logs)

        # A second document must not repeat the warning (once per process).
        logs2: list[str] = []
        pdfio.extract_document_text(self._build_vector_page(), ocr=True, log=logs2.append)
        self.assertEqual([], [m for m in logs2 if "OCR 引擎不可用" in m])


class OcrCacheTest(_TempOcrCacheMixin, unittest.TestCase):
    """OCR results are cached per page, as they are produced."""

    def test_pages_are_cached_as_they_are_recognised(self):
        """Regression: the cache was written only after the *last* page, so a
        cancel (or a crash) threw away every page already recognised — and OCR
        is by far the slowest stage of the pipeline."""
        path = self._build_vector_page(pages=3)
        seen: list[int] = []

        def ocr_fn(page_index, page):
            seen.append(page_index)
            if page_index == 2:  # the user cancels while page 3 is being read
                raise TranslationCancelled()
            return _ocr_fn(page_index, page)

        with self.assertRaises(TranslationCancelled):
            pdfio.extract_document_text(path, ocr=True, ocr_fn=ocr_fn)
        self.assertEqual([0, 1, 2], seen)

        cached = pdfio._load_ocr_cache(pdfio._ocr_cache_path(path))
        # The two finished pages survived the cancellation.
        self.assertEqual([0, 1], sorted(cached))
        self.assertEqual(["Hello", "World"], [b["text"] for b in cached[0]])

    def test_second_run_reuses_the_cache(self):
        path = self._build_vector_page(pages=2)
        calls: list[int] = []

        def counting_ocr_fn(page_index, page):
            calls.append(page_index)
            return _ocr_fn(page_index, page)

        first = pdfio.extract_document_text(path, ocr=True, ocr_fn=counting_ocr_fn)
        self.assertEqual([0, 1], calls)
        second = pdfio.extract_document_text(path, ocr=True, ocr_fn=counting_ocr_fn)
        # No further OCR calls, and the reloaded blocks are identical.
        self.assertEqual([0, 1], calls)
        self.assertEqual(first.blocks, second.blocks)
        self.assertEqual(first.pages, second.pages)
        self.assertEqual(first.ocr_count, second.ocr_count)

    def test_cache_write_leaves_no_temp_files(self):
        path = self._build_vector_page(pages=1)
        pdfio.extract_document_text(path, ocr=True, ocr_fn=_ocr_fn)
        self.assertEqual([], list(self.ocr_cache_dir.glob("*.tmp")))

    def test_corrupted_cache_is_ignored(self):
        path = self._build_vector_page(pages=1)
        # A half-written file (what a non-atomic write used to leave behind).
        pdfio._ocr_cache_path(path).write_text('{"0": [truncated', "utf-8")
        doc = pdfio.extract_document_text(path, ocr=True, ocr_fn=_ocr_fn)
        self.assertEqual(["Hello", "World"], doc.blocks)


