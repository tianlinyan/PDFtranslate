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
    judge_pages,
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
        # A MULTI-LINE source cell keeps its line count (``_fit_exact_n``) and can
        # still exceed the row band — that is what the metric reports.
        cell = _block("amount\nnote", w=200, h=10, in_table=True,
                      fit_width=200, fit_height=9.0)
        long_text = ("Consolidated Statement of Comprehensive Income and "
                     "Other Comprehensive Income For The Period")
        report = measure_layout([cell], [long_text])
        self.assertEqual(report.total, 1)
        self.assertGreaterEqual(report.counts["band_violation"], 1)

    def test_single_line_cell_stays_inside_its_band(self):
        # A single-line source cell never reports a band violation any more: when
        # the row band cannot hold the wrapped translation, ``_fit_block`` falls
        # back to ONE line that fits the band (horizontal overflow beats crossing
        # the grid line below), so the fitter's own output is inside the band.
        cell = _block("amount", w=60, h=10, in_table=True,
                      fit_width=60, fit_height=9.0)
        long_text = ("Consolidated Statement of Comprehensive Income and "
                     "Other Comprehensive Income For The Period")
        report = measure_layout([cell], [long_text])
        self.assertEqual(report.total, 1)
        self.assertEqual(0, report.counts["band_violation"])
        lines, fs = pdfio._fit_block(cell, pdfio._CJK_FONT, long_text)
        self.assertEqual(1, len(lines))
        height = pdfio._wrapped_height(
            pdfio._CJK_FONT, lines, fs,
            pdfio._line_leading(pdfio._CJK_FONT, in_table=True, n_lines=1))
        self.assertLessEqual(height, cell.fit_height + 0.05)


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

    def test_source_echo_is_flagged_as_identity(self):
        text = "Revenue increased by ten percent in 2024."
        res = measure_complete([_block(text)], [text], lang="English")
        self.assertEqual(res["identity_count"], 1)
        self.assertEqual("identity", res["identity"][0]["reason"])

    def test_short_technical_token_is_not_identity(self):
        # "PDF" / "OK" legitimately translate to themselves — flagging those would
        # be noise, so identity needs a prose-like source.
        self.assertEqual(0, measure_complete([_block("PDF")], ["PDF"])["identity_count"])


class IdentityScoringTest(unittest.TestCase):
    """F9: an untranslated echo must never score a perfect 100.

    Regression: ``measure_complete`` only looked for *empty* blocks and for
    source-language residue, and residue was only checked for CJK targets / CJK
    content — so a Latin→Latin run (e.g. Spanish→English) that echoed the source
    verbatim scored 100 and ``eval_harness`` returned exit 0.
    """

    def test_latin_to_latin_echo_does_not_score_100(self):
        text = "Los ingresos aumentaron un diez por ciento en 2024."
        res = eval_pages([[_block(text)]], [[text]], lang="English")
        self.assertEqual(res["complete"]["identity"], 1)
        self.assertLess(res["score"], 100.0)

    def test_identity_outweighs_a_single_missing_block(self):
        # IDENTITY_WEIGHT (3.0) > the missing/residual weight (2.0): with enough
        # clean blocks that the ratio is not saturated, an echo must score lower
        # than a single untranslated block.
        text = "Revenue increased by ten percent in 2024."
        good = "Revenue grew by ten percent in 2024."
        blocks = [_block(text) for _ in range(5)]
        echo = eval_pages([blocks], [[good] * 4 + [text]], lang="English")
        missing = eval_pages([blocks], [[good] * 4 + [""]], lang="English")
        self.assertEqual(1, echo["complete"]["identity"])
        self.assertEqual(1, missing["complete"]["missing"])
        self.assertLess(echo["score"], missing["score"])

    def test_no_measurable_content_scores_zero_and_flags_no_data(self):
        # An empty measurement is not a perfect one: with nothing to judge the score
        # is 0 and ``no_data`` is set (it used to be a misleading 100).
        res = eval_pages([[_block("1,234.56")]], [["1,234.56"]], lang="English")
        self.assertEqual(res["score"], 0.0)
        self.assertTrue(res["no_data"])
        self.assertTrue(res["layout"]["no_data"])


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


class JudgePagesTest(unittest.TestCase):
    def test_judge_pages_aggregates_mock_scores(self):
        pages = [[_block("hello")], [_block("goodbye")]]
        trans = [["bonjour"], ["au revoir"]]
        res = judge_pages(pages, trans, lang="English",
                          judge_fn=lambda page, s, t: 50 + page * 10)
        self.assertEqual(res["score"], 55.0)     # (50 + 60) / 2
        self.assertEqual(res["per_page"], {0: 50.0, 1: 60.0})

    def test_judge_outage_degrades_to_zero_on_that_page(self):
        pages = [[_block("hello")]]
        res = judge_pages(pages, [["bonjour"]], lang="English",
                          judge_fn=lambda page, s, t: (_ for _ in ()).throw(RuntimeError()))
        self.assertEqual(res["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
