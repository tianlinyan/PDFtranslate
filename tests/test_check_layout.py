"""Tests for the layout checker (``check_layout.py``).

The checks are pure functions over glyph bands / rule masks, so most of them are
tested without any PDF; two end-to-end tests build a source page and a
deliberately broken target page and assert the report's grouping.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pymupdf as fitz

import check_layout
from translate_app import pdfio

_OUT = Path(tempfile.gettempdir()) / "pdftranslate_check_layout"
_OUT.mkdir(parents=True, exist_ok=True)


def _span(text: str, rect: tuple[float, float, float, float],
          size: float = 10.0, page: int = 0) -> check_layout.Span:
    return check_layout.Span(text, size, fitz.Rect(*rect), page)


class GlyphBandTest(unittest.TestCase):
    def test_band_uses_origin_and_font_metrics(self):
        span = {
            "text": "x", "size": 10.0, "bbox": (0.0, 0.0, 50.0, 20.0),
            "origin": (0.0, 10.0), "ascender": 0.8, "descender": -0.2,
        }
        band = check_layout.glyph_band(span)
        self.assertAlmostEqual(band.y0, 2.0)     # 10 - 10*0.8
        self.assertAlmostEqual(band.y1, 12.0)    # 10 + 10*0.2
        self.assertAlmostEqual(band.height, 10.0)

    def test_missing_metrics_fall_back_to_defaults(self):
        span = {"text": "x", "size": 10.0, "bbox": (0.0, 0.0, 5.0, 5.0),
                "origin": (0.0, 10.0)}
        band = check_layout.glyph_band(span)
        self.assertGreater(band.height, 0.0)


class OverlapTest(unittest.TestCase):
    def test_detects_a_collision(self):
        spans = [
            _span("Income Statement", (10, 10, 110, 24)),
            _span("Year 2025", (60, 18, 110, 32)),
        ]
        found = check_layout.find_overlaps(spans)
        self.assertEqual(1, len(found))
        self.assertIn("重叠", found[0])

    def test_touching_bands_are_not_a_collision(self):
        spans = [
            _span("a", (10, 10, 100, 20)),
            _span("b", (10, 20, 100, 30)),
        ]
        self.assertEqual([], check_layout.find_overlaps(spans))

    def test_tiny_label_clipped_by_a_paragraph_is_reported(self):
        # The *smaller* band is what counts: a label whose glyphs are half covered
        # by a neighbouring paragraph is a real collision, however tall the other
        # block is.
        spans = [
            _span("long paragraph", (10, 0, 200, 100)),
            _span("x", (10, 98, 20, 104)),
        ]
        self.assertEqual(1, len(check_layout.find_overlaps(spans)))


class OffPageTest(unittest.TestCase):
    def test_detects_right_and_bottom_overflow(self):
        page = fitz.Rect(0, 0, 100, 100)
        spans = [
            _span("ok", (0, 0, 90, 10)),
            _span("wide", (0, 0, 140, 10)),
            _span("low", (0, 95, 50, 120)),
        ]
        found = check_layout.find_off_page(spans, page)
        self.assertEqual(2, len(found))
        self.assertTrue(any("wide" in m for m in found))
        self.assertTrue(any("low" in m for m in found))


class RuleCrossingTest(unittest.TestCase):
    def test_reports_spans_sitting_on_a_rule(self):
        import numpy as np

        rules = np.zeros((100, 100), dtype=bool)
        rules[40:42, :] = True                    # a rule at y = 40..42
        spans = [
            _span("on the rule", (0, 32, 90, 52)),   # rule runs through it
            _span("clear", (0, 10, 90, 20)),
        ]
        found = check_layout.find_rule_crossings(spans, rules, scale=1.0)
        self.assertEqual(1, len(found))
        self.assertIn("压在表格线上", found[0])

    def test_a_rule_at_the_band_edge_is_not_a_crossing(self):
        import numpy as np

        rules = np.zeros((100, 100), dtype=bool)
        rules[40:42, :] = True
        # Glyphs sit just below the rule (ascenders graze it) — normal on a dense
        # statement, not a defect.
        spans = [_span("under it", (0, 41, 90, 61))]
        self.assertEqual([], check_layout.find_rule_crossings(
            spans, rules, scale=1.0))

    def test_no_mask_is_not_an_error(self):
        self.assertEqual([], check_layout.find_rule_crossings(
            [_span("x", (0, 0, 10, 10))], None, scale=1.0))


class SmallTextTest(unittest.TestCase):
    def test_grades_table_floor_and_body_floor(self):
        spans = [
            _span("tiny", (0, 0, 10, 10), size=2.5),
            _span("cell", (0, 20, 10, 30), size=6.2),
            _span("body", (0, 40, 10, 50), size=9.0),
        ]
        found = check_layout.find_too_small(spans, 6.0, 7.0)
        self.assertEqual(2, len(found))
        self.assertTrue(any("低于表格下限" in m for m in found))
        self.assertTrue(any("低于正文下限" in m for m in found))


class MissingTest(unittest.TestCase):
    def test_uncovered_source_block_is_reported(self):
        block = pdfio.Block(text="利润表", page=0, x0=10, y0=10, x1=60, y1=24,
                            size=12.0, align="left", bold=False,
                            single_line=True)
        covered = [_span("Income Statement", (10, 10, 60, 24))]
        self.assertEqual([], check_layout.find_missing([block], covered))
        self.assertEqual(1, len(check_layout.find_missing([block], [])))

    def test_empty_source_blocks_are_ignored(self):
        block = pdfio.Block(text="   ", page=0, x0=10, y0=10, x1=60, y1=24,
                            size=12.0, align="left", bold=False,
                            single_line=True)
        self.assertEqual([], check_layout.find_missing([block], []))


class EndToEndTest(unittest.TestCase):
    def _source(self, path: Path) -> None:
        doc = fitz.open()
        page = doc.new_page(width=300, height=200)
        page.insert_text((20, 40), "Amount unit", fontsize=10)
        page.draw_line(fitz.Point(20, 60), fitz.Point(280, 60), width=0.8)
        doc.save(str(path))
        doc.close()

    def _target(self, path: Path, *, broken: bool) -> None:
        doc = fitz.open()
        page = doc.new_page(width=300, height=200)
        if broken:
            # Baseline 62 ⇒ band 54..64, so the rule at y=60 runs through the
            # middle of the glyphs.
            page.insert_text((20, 62), "Amount unit: RMB in ten thousand yuan",
                             fontsize=10)
            page.insert_text((20, 40), "Amount unit", fontsize=10)
            page.insert_text((250, 40), "overflows the page", fontsize=10)
        else:
            page.insert_text((20, 40), "Amount unit", fontsize=10)
        doc.save(str(path))
        doc.close()

    def test_clean_target_reports_nothing_structural(self):
        src = _OUT / "clean_src.pdf"
        tgt = _OUT / "clean_tgt.pdf"
        self._source(src)
        self._target(tgt, broken=False)
        report = check_layout.check_document(src, tgt)
        self.assertEqual([], report.structural(), report.structural())

    def test_broken_target_reports_rule_and_page_and_overlap(self):
        src = _OUT / "broken_src.pdf"
        tgt = _OUT / "broken_tgt.pdf"
        self._source(src)
        self._target(tgt, broken=True)
        report = check_layout.check_document(src, tgt)
        self.assertTrue(report.rule, "rule crossing not detected")
        self.assertTrue(report.off_page, "off-page text not detected")
        self.assertTrue(report.structural())

    def test_missing_translation_is_reported_for_text_layer_sources(self):
        src = _OUT / "missing_src.pdf"
        tgt = _OUT / "missing_tgt.pdf"
        self._source(src)
        self._target(tgt, broken=False)
        report = check_layout.check_document(src, tgt)
        # The source's "Amount unit" line has a translation; the rule has none.
        self.assertEqual([], [m for m in report.missing if "Amount unit" in m])


if __name__ == "__main__":
    unittest.main()
