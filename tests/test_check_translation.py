"""Tests for the post-translation checker (``check_translation.py``).

The checker is advisory: it compares the raw text layers of two PDFs.  These
tests build small fixture PDFs with pymupdf and assert on the issues the
checker collects.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pymupdf as fitz

from check_translation import _cn_to_int, _section_numbers, main, run_checks


def _pdf(path: Path, page_lines: list[list[str]]) -> Path:
    doc = fitz.open()
    for lines in page_lines:
        page = doc.new_page()
        y = 70.0
        for ln in lines:
            kw = {}
            if any("一" <= c <= "鿿" for c in ln):
                kw = {"fontname": "china-ts"}
            page.insert_text((60, y), ln, fontsize=11, **kw)
            y += 20.0
    doc.save(str(path))
    doc.close()
    return path


class CheckerTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_clean_run_passes(self):
        src = _pdf(self.tmp / "src.pdf", [["总资产 1,234,567.89", "2025 年度报告"]])
        tgt = _pdf(
            self.tmp / "tgt.pdf",
            [["Total assets were 1,234,567.89", "Annual report 2025"]],
        )
        checker = run_checks(src, tgt, lang="English")
        self.assertTrue(checker.all_clear(), checker.numeric + checker.cjk)

    def test_separator_swap_is_caught(self):
        # The benchmark bug 3,702.726,474.45 has the *same* digit sequence as
        # 3,702,726,474.45 — only the separator roles differ, so the check must
        # compare separator patterns too, not just digits.
        src = _pdf(self.tmp / "src.pdf", [["总资产 3,702,726,474.45"]])
        tgt = _pdf(self.tmp / "tgt.pdf", [["Total assets 3,702.726,474.45"]])
        checker = run_checks(src, tgt, lang="English")
        self.assertFalse(checker.numeric_ok())
        self.assertTrue(checker.numeric[0].startswith("第 1 页"), checker.numeric)

    def test_dropped_digits_are_caught(self):
        src = _pdf(self.tmp / "src.pdf", [["金额 11,530,351.55"]])
        tgt = _pdf(self.tmp / "tgt.pdf", [["Amount 11,530,351,55"]])  # decimal lost
        self.assertFalse(run_checks(src, tgt).numeric_ok())

    def test_cjk_residual_flagged_for_latin_target_only(self):
        src = _pdf(self.tmp / "src.pdf", [["董事长 汪建法"]])
        tgt = _pdf(self.tmp / "tgt.pdf", [["Chairman Wang Jianfa 汪建法"]])
        latin = run_checks(src, tgt, lang="English")
        self.assertTrue(latin.cjk)
        self.assertIn("汪建法", latin.cjk[0])
        cjk = run_checks(src, tgt, lang="简体中文")
        self.assertEqual([], cjk.cjk)

    def test_scan_pages_are_skipped(self):
        # A source page with no text layer (a scan) has no reliable numbers:
        # its page must be skipped instead of tripping the digit comparison.
        src = self.tmp / "scan.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.draw_rect(fitz.Rect(40, 40, 300, 200), color=None, fill=(0.9, 0.9, 0.9))
        doc.save(str(src))
        doc.close()
        tgt = _pdf(self.tmp / "tgt.pdf", [["Random 999,999,999.99"]])
        checker = run_checks(src, tgt)
        self.assertTrue(checker.numeric_ok())
        self.assertTrue(checker.all_clear())

    def test_fewer_target_pages_is_flagged(self):
        src = _pdf(self.tmp / "src.pdf", [["Page one"], ["Page two"]])
        tgt = _pdf(self.tmp / "tgt.pdf", [["Page one"]])
        checker = run_checks(src, tgt)
        self.assertTrue(checker.pages)
        self.assertIn("少于", checker.pages[0])

    def test_section_sequence_mismatch_is_flagged(self):
        src = _pdf(self.tmp / "src.pdf", [["1. First", "2. Second", "3. Third"]])
        tgt = _pdf(self.tmp / "tgt.pdf", [["1. First", "2. Second"]])
        checker = run_checks(src, tgt)
        self.assertTrue(checker.numbering)
        self.assertIn("章节编号不一致", checker.numbering[0])

    def test_style_parity_between_languages_is_allowed(self):
        # 第4章 -> "Chapter 4" is the natural translation, not an inconsistency:
        # the values match and the per-doc style is uniform.
        src = _pdf(self.tmp / "src.pdf", [["第四章 结果", "第五章 讨论"]])
        tgt = _pdf(self.tmp / "tgt.pdf", [["Chapter 4 Results", "Chapter 5 Discussion"]])
        checker = run_checks(src, tgt, lang="English")
        self.assertEqual([], checker.numbering, checker.numbering)

    def test_mixed_styles_across_translation_pages_are_flagged(self):
        src = _pdf(
            self.tmp / "src.pdf",
            [["1. First", "2. Second"], ["3. Third"]],
        )
        tgt = _pdf(
            self.tmp / "tgt.pdf",
            [["1. First", "2. Second"], ["Chapter 3 Third"]],
        )
        checker = run_checks(src, tgt, lang="English")
        self.assertTrue(checker.numbering)
        self.assertIn("风格全文不一致", checker.numbering[0])

    def test_main_exit_codes(self):
        good_src = _pdf(self.tmp / "g_src.pdf", [["1,234.56"]])
        good_tgt = _pdf(self.tmp / "g_tgt.pdf", [["1,234.56"]])
        self.assertEqual(0, main([str(good_src), str(good_tgt)]))
        bad_tgt = _pdf(self.tmp / "b_tgt.pdf", [["9,999.99"]])
        self.assertEqual(1, main([str(good_src), str(bad_tgt)]))
        self.assertEqual(2, main([]))

    def test_cn_numeral_helper(self):
        self.assertEqual(4, _cn_to_int("四"))
        self.assertEqual(12, _cn_to_int("十二"))
        self.assertEqual(45, _cn_to_int("四十五"))
        self.assertIsNone(_cn_to_int("三百"))
        self.assertEqual(4, _section_numbers("第四章 结果")[0][1])


if __name__ == "__main__":
    unittest.main()
