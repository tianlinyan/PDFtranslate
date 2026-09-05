"""Tests for the A-② evaluation harness (``translate_app/eval.py``).

These run offline (no PDF, no model): they construct ``Block`` objects directly and
exercise the layout / number / completeness metrics plus the aggregator and the
A/B comparator.  All caches are irrelevant here (no cache path is touched).
"""

from __future__ import annotations

import unittest

from translate_app import pdfio
from translate_app.eval import (
    aggregate,
    compare,
    eval_pages,
    measure_complete,
    measure_layout,
    measure_numbers,
)
from translate_app.pdfio import Block


def _block(text, *, w=200, h=50, size=10.0, in_table=False, fit_width=0.0,
           fit_height=0.0, y0=0.0):
    return Block(text, 0, 0, y0, w, y0 + h, size=size,
                 in_table=in_table, fit_width=fit_width, fit_height=fit_height)


class MeasureLayoutTest(unittest.TestCase):
    def test_clean_block_has_no_issues(self):
        report = measure_layout([_block("hello world")], ["bonjour tout le monde"])
        self.assertEqual(report.total, 1)
        self.assertEqual(report.counts["overflow"], 0)
        self.assertEqual(report.counts["too_small"], 0)
        self.assertEqual(report.counts["crowding"], 0)
        # A short translation in a large box renders at a readable size.
        self.assertEqual(report.buckets[">=6"], 1)

    def test_long_translation_in_small_box_overflows(self):
        # A tiny 60x12 box cannot hold a long line at a readable size -> overflow.
        long_text = ("this is a translation much longer than the tiny box can "
                     "possibly hold on a single line and it will certainly wrap")
        report = measure_layout([_block("src", w=60, h=12)], [long_text])
        self.assertEqual(report.total, 1)
        self.assertGreaterEqual(report.counts["overflow"], 1)

    def test_numeric_cell_is_not_measured(self):
        # A pure figure is a protected block: it never counts as a layout defect.
        report = measure_layout([_block("1,234.56")], ["1,234.56"])
        self.assertEqual(report.total, 0)

    def test_in_table_band_violation(self):
        # A table cell with a fixed row band (fit_height) and a long translation
        # that wraps to >1 line and exceeds the band is flagged.
        cell = _block("amount", w=60, h=10, in_table=True,
                      fit_width=60, fit_height=9.0)
        long_text = ("Consolidated Statement of Comprehensive Income and "
                     "Other Comprehensive Income For The Period")
        report = measure_layout([cell], [long_text])
        self.assertEqual(report.total, 1)
        self.assertGreaterEqual(report.counts["band_violation"], 1)


class MeasureNumbersTest(unittest.TestCase):
    def test_separator_swap_is_caught(self):
        res = measure_numbers([_block("3,702,726,474.45")], ["3,702.726,474.45"])
        self.assertEqual(res["count"], 1)
        item = res["numbers"][0]
        self.assertTrue(item["missing"] or item["extra"])

    def test_value_equivalent_units_pass(self):
        # "3.14 亿元" equals "314 million yuan" by value via the unit multiplier.
        res = measure_numbers([_block("3.14 亿元")], ["314 million yuan"])
        self.assertEqual(res["count"], 0)

    def test_dropped_digit_is_caught(self):
        res = measure_numbers([_block("3,702,726,474.45")], ["3,702,726,474.4"])
        self.assertEqual(res["count"], 1)


class MeasureCompleteTest(unittest.TestCase):
    def test_missing_translation(self):
        res = measure_complete([_block("hello world")], [""])
        self.assertEqual(res["missing_count"], 1)

    def test_residual_cjk_for_western_target(self):
        res = measure_complete([_block("hello")], ["你好"], lang="English")
        self.assertEqual(res["residual_count"], 1)

    def test_numeric_block_not_reported_as_missing(self):
        res = measure_complete([_block("1,234.56")], [""])
        self.assertEqual(res["missing_count"], 0)


class AggregateCompareTest(unittest.TestCase):
    def test_clean_document_scores_100(self):
        pages = [[_block("hello world")], [_block("good morning")]]
        trans = [[""], [""]]  # placeholder, replaced below
        # Use a translation that fits so measure_layout is clean and complete is clean.
        trans = [["bonjour tout le monde"], ["bonjour tout le monde"]]
        res = eval_pages(pages, trans, lang="English")
        self.assertEqual(res["layout"]["score"], 100.0)
        self.assertEqual(res["complete"]["missing"], 0)

    def test_aggregate_weights_overflow_over_tiny(self):
        rep = measure_layout([_block("x", w=60, h=12)], [
            "a translation that is far too long for this very small box"])
        agg = aggregate([rep])
        self.assertLess(agg["score"], 100.0)
        self.assertEqual(agg["total"], rep.total)

    def test_compare_reports_delta(self):
        clean = eval_pages([[_block("hello")]], [["bonjour"]], lang="English")
        bad = eval_pages([[_block("3,702,726,474.45")]],
                         [["3,702.726,474.45"]], lang="English")
        out = compare(clean, bad)
        # The candidate introduced a number defect, so the score must drop.
        self.assertIsNotNone(out["score_delta"])
        self.assertGreaterEqual(out["numbers"]["total_delta"], 1)


if __name__ == "__main__":
    unittest.main()
