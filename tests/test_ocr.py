"""Tests for the OCR extraction plumbing (no real model inference)."""

from __future__ import annotations

import re
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



class NumberNormalizationTest(unittest.TestCase):
    """OCR-degraded amounts must be repaired, and only when provably safe.

    Regression for the real report (Mintai 2025, pages 24-27): RapidOCR tore
    separator groups apart (``65, 334, 085.99``), swapped dots and commas
    (``3,702.726,474.45``) and dropped decimal points (``11,530,351,55``).
    These cells bypass the translation model (pure digits), so the repair here
    is the *last* line of defence before the amount reaches the reader.
    """

    #: (corrupted as OCR read it, intended amount).
    REAL_CASES = [
        ("3,702.726,474.45", "3,702,726,474.45"),
        ("65, 334, 085.99", "65,334,085.99"),
        ("8, 677,890, 288.88", "8,677,890,288.88"),
        ("228, 468, 960.74", "228,468,960.74"),
        ("766, 777,688.19", "766,777,688.19"),
        ("72, 848, 047. 80", "72,848,047.80"),
        ("268,719, 676,841.77", "268,719,676,841.77"),
        ("11,402.358,739.92", "11,402,358,739.92"),
        ("189.060,973.80", "189,060,973.80"),
        ("11.564,079,882.19", "11,564,079,882.19"),
        ("309,120.175.36", "309,120,175.36"),
        ("161.708,976.70", "161,708,976.70"),
        ("24,321.445,868.48", "24,321,445,868.48"),
        ("1,295,874,138,65", "1,295,874,138.65"),
        ("1,150,111,164,25", "1,150,111,164.25"),
        ("11,530,351,55", "11,530,351.55"),
        ("87,893, 014.86", "87,893,014.86"),
        ("68, 081,405.42", "68,081,405.42"),
        ("4, 641,610,929.75", "4,641,610,929.75"),
        ("33,024,552,.394.27", "33,024,552,394.27"),
        # A wrapped figure leaves a ``/`` where the line broke.
        ("231. / 81", "231.81"),
        ("150.0 / 8", "150.08"),
        ("1,304,083,150.0 / 8", "1,304,083,150.08"),
        ("192, / 003,164.72", "192,003,164.72"),
        ("292,712,933,925.1 / 7", "292,712,933,925.17"),
    ]

    #: Strings that are *not* regrouped: dates, plain values, prose.
    UNTOUCHED = [
        "1960.08",
        "0.98",
        "92.5%",
        "97.05%",
        "29",
        "-60,327,958.12",
        "0",
        "100.00",
        "2025年12月31日",
        "总资产",
        "1.5亿元",
        "2025",
        "10/20",  # no comma and no dot: ratio, not a wrapped figure
        "3/4",
        "12,34,56,789",  # cannot form valid 3-groups: leave for human review
    ]

    def test_real_corrupted_samples_are_repaired(self):
        for corrupted, expected in self.REAL_CASES:
            with self.subTest(corrupted=corrupted):
                self.assertEqual(expected, pdfio._normalize_number(corrupted))

    def test_plain_text_is_never_touched(self):
        for text in self.UNTOUCHED:
            with self.subTest(text=text):
                self.assertEqual(text, pdfio._normalize_number(text))

    def test_digit_sequence_is_never_altered(self):
        """Repairs are formatting only: the digits are the same, in the same order."""
        for corrupted, expected in self.REAL_CASES:
            with self.subTest(corrupted=corrupted):
                self.assertEqual(
                    re.sub(r"\D", "", corrupted), re.sub(r"\D", "", expected)
                )

    def test_synthesize_reports_fixes_through_log(self):
        logs: list[str] = []
        blocks = pdfio._synthesize_ocr_blocks(
            [
                ([[0.0, 0.0], [30.0, 0.0], [30.0, 10.0], [0.0, 10.0]],
                 "3,702.726,474.45"),
                ([[0.0, 12.0], [30.0, 12.0], [30.0, 22.0], [0.0, 22.0]],
                 "未交货"),  # prose: untouched
            ],
            0, log=logs.append,
        )
        self.assertEqual("3,702,726,474.45", blocks[0].text)
        note = [m for m in logs if "数字格式异常已修正 1 处" in m and "3,702.726,474.45" in m]
        self.assertEqual(1, len(note), logs)

    def test_cannot_repair_is_left_verbatim_and_never_invented(self):
        # A digit count that cannot form valid 3-groups must not be "fixed":
        # a made-up amount in a financial statement is worse than a visible error.
        self.assertEqual("12,34,56,789", pdfio._normalize_number("12,34,56,789"))
        # Digits only: fold spaces (the sequence is intact, the grouping kind
        # can't be known) — never invent a separator.
        self.assertEqual("2025", pdfio._normalize_number("2 0 2 5"))


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

    def test_signature_items_are_kept_as_blocks(self):
        # v0.3.0: the handwritten-signature *auto-drop* is removed — the AI agent
        # decides at runtime.  A tall name-only box now stays a block (the model
        # may classify it as signature/keep); no "手写体签字" drop log is emitted.
        logs: list[str] = []
        blocks = pdfio._synthesize_ocr_blocks(
            [
                ([[2.0, 100.0], [30.0, 100.0], [30.0, 109.0], [2.0, 109.0]], "项目"),
                ([[2.0, 120.0], [30.0, 120.0], [30.0, 129.0], [2.0, 129.0]], "资产"),
                ([[2.0, 140.0], [30.0, 140.0], [30.0, 149.0], [2.0, 149.0]], "利润"),
                # handwritten 小波: 40pt tall, bottom band of the 800pt page — kept
                ([[120.0, 700.0], [260.0, 700.0], [260.0, 740.0], [120.0, 740.0]], "小波"),
                # tall bottom-band item WITH digits: a figure, also kept
                ([[300.0, 690.0], [380.0, 690.0], [380.0, 715.0], [300.0, 715.0]],
                 "V001"),
            ],
            0, log=logs.append, page_height=800.0,
        )
        texts = [b.text for b in blocks]
        self.assertIn("项目", texts)
        self.assertIn("资产", texts)
        self.assertIn("小波", texts)
        self.assertIn("V001", texts)
        self.assertEqual([], [m for m in logs if "手写体签字" in m], logs)

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



class OcrProductionPathTest(unittest.TestCase):
    """The real (non-injected) OCR path: page rendering + engine output parsing.

    Exercises ``_page_to_array`` (RGB->BGR, zoom, contiguity) and the defensive
    parsing of the RapidOCR engine's output in ``_ocr_page_blocks`` — the part
    previously only covered indirectly through the ``ocr_fn`` injection seam.
    """

    def test_page_to_array_is_bgr_zoomed_and_contiguous(self):
        import numpy as np  # noqa: F401  (ensures numpy is importable)

        doc = fitz.open()
        page = doc.new_page(width=100, height=100)
        # Solid red rectangle (RGB 1,0,0).
        page.draw_rect(fitz.Rect(0, 0, 100, 100), color=None, fill=(1, 0, 0))
        img, zoom = pdfio._page_to_array(page)
        doc.close()
        self.assertAlmostEqual(zoom, 300.0 / 72.0)
        self.assertEqual(3, img.shape[2])
        # 100pt * (300/72) = 416.67px.
        self.assertIn(img.shape[0], (416, 417))
        # Must be a contiguous array: RapidOCR/OpenCV may reject or silently
        # copy the negative-stride view the BGR flip would otherwise produce.
        self.assertTrue(img.flags.c_contiguous)
        # A red pixel (R=255,G=0,B=0) is stored as [B,G,R]=[0,0,255] in BGR.
        px = img[img.shape[0] // 2, img.shape[1] // 2]
        self.assertEqual((int(px[0]), int(px[1]), int(px[2])), (0, 0, 255))

    def test_ocr_engine_output_parsed_and_zoom_converted(self):
        doc = fitz.open()
        page = doc.new_page(width=100, height=100)
        page.draw_rect(fitz.Rect(10, 10, 90, 90), color=(0, 0, 0), fill=(1, 1, 1))

        class FakeEngine:
            def __call__(self, img):
                # (list, timings) form, with degenerate items that must be
                # filtered out (no text / None / empty text).
                return (
                    [
                        [[[0, 0], [40, 0], [40, 40], [0, 40]], "Hello", 0.9],
                        [[[0, 0], [40, 0], [40, 40], [0, 40]]],  # no text -> skip
                        None,  # skip
                        [[[0, 0], [40, 0], [40, 40], [0, 40]], "", 0.5],  # empty -> skip
                    ],
                    {"det": 0.1},
                )

        with mock.patch.object(pdfio, "_get_ocr_engine", return_value=FakeEngine()):
            blocks = pdfio._ocr_page_blocks(0, page, None, None, lambda m: None)
        doc.close()
        self.assertEqual(["Hello"], [b.text for b in blocks])
        # The pixel box was converted to PDF points (divided by zoom).
        zoom = 300.0 / 72.0
        self.assertAlmostEqual(40.0 / zoom, blocks[0].x1, places=3)

    def test_ocr_engine_bare_list_output_is_accepted(self):
        # Some RapidOCR versions return a bare list (no timings tuple); the
        # parser must handle both shapes.
        doc = fitz.open()
        page = doc.new_page(width=100, height=100)

        class FakeEngine:
            def __call__(self, img):
                return [[[[0, 0], [20, 0], [20, 20], [0, 20]], "World", 0.8]]

        with mock.patch.object(pdfio, "_get_ocr_engine", return_value=FakeEngine()):
            blocks = pdfio._ocr_page_blocks(0, page, None, None, lambda m: None)
        doc.close()
        self.assertEqual(["World"], [b.text for b in blocks])


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

    def test_cache_filename_embeds_the_version(self):
        # Bumping _OCR_CACHE_VERSION must actually change the file name — a
        # cached build of blocks from an older synthesis (no fit_width, labels
        # widened onto the note band, fragments unmerged) is reused verbatim and
        # silently defeats the grid fixes otherwise.
        path = self._build_vector_page(pages=1)
        cache = pdfio._ocr_cache_path(path)
        self.assertIn(f"ocr_v{pdfio._OCR_CACHE_VERSION}_", cache.name)

    def test_ocr_cache_dir_probes_writability(self):
        # mkdir(exist_ok=True) succeeds on an existing directory even when
        # writing into it is denied (read-only home, sandbox).  Only a real probe
        # write catches that, so an unwritable override must fall back to temp.
        override = self.ocr_cache_dir
        home_cache = Path.home() / ".pdftranslate" / "ocr_cache"
        real_write_text = Path.write_text

        def fake_write_text(self, *args, **kwargs):
            if self.name.startswith(".ocr_write_probe_"):
                if override in self.parents or home_cache in self.parents:
                    raise PermissionError("denied by test")
            return real_write_text(self, *args, **kwargs)

        with mock.patch.object(Path, "write_text", fake_write_text):
            resolved = pdfio._ocr_cache_dir()
        self.assertEqual(resolved, Path(tempfile.gettempdir()) / "pdftranslate_ocr_cache")

    def test_ocr_cache_write_failure_is_logged_once(self):
        # A failed OCR cache write must not abort the run, but it must be
        # reported — once — otherwise a read-only cache silently re-OCRs the
        # whole document on every run.
        path = self._build_vector_page(pages=1)
        logs: list[str] = []
        with mock.patch.object(
            pdfio, "_save_ocr_cache", return_value="PermissionError: denied"
        ):
            doc = pdfio.extract_document_text(
                path, ocr=True, ocr_fn=_ocr_fn, log=logs.append
            )
        # OCR itself still succeeds ...
        self.assertEqual(["Hello", "World"], doc.blocks)
        # ... and the failure is reported exactly once, not per page or per block.
        warnings = [m for m in logs if "OCR 缓存写入失败" in m]
        self.assertEqual(1, len(warnings), logs)

    def test_corrupted_cache_is_ignored(self):
        path = self._build_vector_page(pages=1)
        # A half-written file (what a non-atomic write used to leave behind).
        pdfio._ocr_cache_path(path).write_text('{"0": [truncated', "utf-8")
        doc = pdfio.extract_document_text(path, ocr=True, ocr_fn=_ocr_fn)
        self.assertEqual(["Hello", "World"], doc.blocks)

    def test_stale_cache_is_renormalized_on_load(self):
        """A cache written before number normalization must not keep the
        garbled figures: renormalize on load instead of requiring the user to
        wipe ``~/.pdftranslate/ocr_cache`` (which means re-OCRing everything)."""
        path = self._build_vector_page(pages=1)
        stale = pdfio.Block(
            text="3,702.726,474.45", page=0, x0=50.0, y0=50.0, x1=200.0, y1=70.0,
            ocr=True,
        )
        pdfio._save_ocr_cache(
            pdfio._ocr_cache_path(path), {0: [pdfio._block_to_dict(stale)]}
        )

        def must_not_call(*_args):
            raise AssertionError("cache hit must not call the OCR function")

        logs: list[str] = []
        doc = pdfio.extract_document_text(
            path, ocr=True, ocr_fn=must_not_call, log=logs.append
        )
        self.assertEqual(["3,702,726,474.45"], doc.blocks)
        note = [m for m in logs if "数字格式异常已修正 1 处" in m and "→ 3,702,726,474.45" in m]
        self.assertEqual(1, len(note), logs)


