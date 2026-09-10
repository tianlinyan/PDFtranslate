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

from check_translation import (
    _cjk_residual,
    _is_scan_like_text,
    _normalize_cjk_ordinals,
    _numeric_diff,
    _section_numbers,
    _cn_to_int,
    main,
    run_checks,
)


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

    def test_sparse_text_scan_page_is_skipped(self):
        # A scanned statement often leaves only a page number in the text layer
        # (e.g. "22").  That is still a scan whose figures live in the image, so
        # the page must be skipped rather than digit-compared against it.
        src = _pdf(self.tmp / "src.pdf", [["22"]])
        tgt = _pdf(self.tmp / "tgt.pdf", [["22", "97,923,282.04 NP 65,334,085.99"]])
        checker = run_checks(src, tgt)
        self.assertTrue(checker.numeric_ok(), checker.numeric)
        self.assertTrue(checker.all_clear())

    def test_chinese_ordinals_normalized_not_flagged_as_numbers(self):
        # 一、二、 → 1. 2. and （四） → (4) are the expected rendering of section
        # markers for a Latin target, not a new figure the source lacks.
        src = _pdf(self.tmp / "src.pdf", [["一、主要会计数据", "总资产 1,234,567.89"]])
        tgt = _pdf(self.tmp / "tgt.pdf", [["1. Key Accounting Data", "Total assets 1,234,567.89"]])
        checker = run_checks(src, tgt, lang="English")
        self.assertTrue(checker.numeric_ok(), checker.numeric)

    def test_cn_ordinal_enum_recognized_as_section(self):
        # A Chinese ordinal enumeration heading (一、 二、 （四）) must be picked up
        # as a section marker so it aligns with the translated "1." / "2." / "(4)".
        self.assertEqual(1, _section_numbers("一、主要会计数据")[0][1])
        self.assertEqual(2, _section_numbers("二、补充财务数据")[0][1])
        self.assertEqual(4, _section_numbers("（四）市场风险")[0][1])

    def test_paren_headings_align_between_languages(self):
        # （四）（五） → (4) (5): both the source's Chinese ordinal and the
        # translation's parenthesised Arabic form must be seen as the same
        # section markers, so no "numbering mismatch" / "style" warning fires.
        src = _pdf(self.tmp / "src.pdf", [["（四）市场风险", "（五）流动性风险"]])
        tgt = _pdf(self.tmp / "tgt.pdf", [["(4) Market risk", "(5) Liquidity risk"]])
        checker = run_checks(src, tgt, lang="English")
        self.assertEqual([], checker.numbering, checker.numbering)

    def test_normalize_cjk_ordinals_handles_common_forms(self):
        # The numeral becomes Arabic but the marker's own delimiter is KEPT, so the
        # substituted digit can never merge with an adjacent figure.
        self.assertEqual("1、主要", _normalize_cjk_ordinals("一、主要"))
        self.assertEqual("(4)市场", _normalize_cjk_ordinals("（四）市场"))
        self.assertEqual("2节 数据", _normalize_cjk_ordinals("第二节 数据"))

    def test_ordinal_next_to_an_amount_does_not_glue_digits(self):
        # Regression: 一、1,234.56 used to normalize to "11,234.56" (the marker was
        # dropped and its digit glued to the amount), so a *correct* translation was
        # reported as a fatal 数字不一致 and the script exited 1.
        self.assertEqual("1、1,234.56", _normalize_cjk_ordinals("一、1,234.56"))
        self.assertEqual("1,234(1)", _normalize_cjk_ordinals("1,234（一）"))

    def test_correct_translation_beside_an_ordinal_passes_end_to_end(self):
        src = _pdf(self.tmp / "src.pdf", [["一、1,234.56 营业收入同比增长。"]])
        tgt = _pdf(self.tmp / "tgt.pdf", [["1. 1,234.56 Revenue increased."]])
        checker = run_checks(src, tgt, lang="English")
        self.assertTrue(checker.numeric_ok(), checker.numeric)
        self.assertEqual(0, main([str(src), str(tgt)]))

    def test_unit_conversions_are_not_number_mismatches(self):
        # Regression: the checker compared surface digit strings, so a *correct*
        # translation that converted 亿元 → million / 万元 → plain yuan was
        # reported as a fatal 数字不一致 and the script exited 1.
        src = _pdf(self.tmp / "src.pdf", [["总资产 3.14 亿元", "净利润 1,234.56 万元"]])
        tgt = _pdf(
            self.tmp / "tgt.pdf",
            [["Total assets 314 million yuan",
              "Net profit 12,345,600 yuan"]],
        )
        checker = run_checks(src, tgt, lang="English")
        self.assertTrue(checker.numeric_ok(), checker.numeric)
        self.assertEqual(0, main([str(src), str(tgt)]))

    def test_unit_conversion_with_a_wrong_value_is_still_caught(self):
        src = _pdf(self.tmp / "src.pdf", [["总资产 3.14 亿元"]])
        tgt = _pdf(self.tmp / "tgt.pdf", [["Total assets 314 thousand yuan"]])
        self.assertFalse(run_checks(src, tgt, lang="English").numeric_ok())

    def test_full_width_digits_and_minus_compare_by_value(self):
        # A full-width source page produced NO tokens at all, so its ASCII
        # translation looked like "译文多出" numbers.
        self.assertEqual([], _numeric_diff(
            "总资产 １，２３４．５６ 万元", "Total assets 1,234.56 ten thousand yuan"))
        self.assertEqual([], _numeric_diff("金额 －1,234.56", "Amount -1,234.56"))
        self.assertTrue(_numeric_diff("金额 １，２３４．５６", "Amount 9,999.99"))

    def test_unit_multiplier_table_matches_the_agent_number_audit(self):
        # The checker mirrors ``translate_app.agent.flow._unit_multiplier`` (the
        # canonical implementation for the in-app audit); this pins the two tables
        # together so they cannot drift silently.
        import check_translation as ct
        from translate_app.agent import flow as agent_flow

        for window in (" 亿元", "万元", " million yuan", "ten thousand", "hundred million",
                       "billion", "trillion", "元", "yuan", "", " 行次"):
            self.assertEqual(agent_flow._unit_multiplier(window), ct._unit_multiplier(window),
                             window)

    def test_separator_style_change_is_not_reported(self):
        # Same value, different separator style: not a number defect (the value is
        # what matters; formatting is a separate, advisory concern).
        self.assertEqual([], _numeric_diff("金额 1,234.56", "Amount 1234.56"))

    def test_lost_unit_multiplier_is_reported(self):
        # Regression: ``1,234.56 万元`` and ``1,234.56 yuan`` have the SAME separator
        # roles (the unit is not part of the token), so the old "roles equal →
        # clean" short-circuit skipped the value check and a translation that
        # dropped the 万 multiplier was reported as consistent.
        self.assertTrue(_numeric_diff("总资产 1,234.56 万元", "Total assets 1,234.56 yuan"))
        # A correct conversion is still clean (negative control).
        self.assertEqual([], _numeric_diff("总资产 1,234.56 万元",
                                           "Total assets 12,345,600 yuan"))

    def test_changed_percentage_is_reported(self):
        # ``%`` was not stripped before ``Decimal`` (so the token got no value at
        # all) and a 10× percentage change compared "consistent".
        self.assertTrue(_numeric_diff("毛利率 92.5%", "Gross margin 9.25%"))
        self.assertEqual([], _numeric_diff("毛利率 92.5%", "Gross margin 92.5%"))

    def test_accounting_parenthesis_sign_is_compared(self):
        # The sign of ``（1,234.56）`` lives outside the numeric token; losing it
        # used to compare equal to the positive value.
        self.assertTrue(_numeric_diff("金额 （1,234.56）", "Amount 1,234.56"))
        self.assertEqual([], _numeric_diff("金额 （1,234.56）", "Amount -1,234.56"))
        self.assertTrue(_numeric_diff("金额 -1,234.56", "Amount 1,234.56"))

    def test_scan_page_still_checks_residual_cjk(self):
        # A scan-like source page (only a page number in the text layer) skips the
        # digit comparison — but the residual-CJK check needs no numbers, and the
        # whole-page early return used to skip it too (750 Han chars on a real
        # sample went unreported).
        src = _pdf(self.tmp / "scan_src.pdf", [["22"]])
        tgt = _pdf(self.tmp / "scan_tgt.pdf",
                   [["22", "Net interest income 汪建法 123,456.78"]])
        checker = run_checks(src, tgt, lang="English")
        self.assertTrue(checker.numeric_ok(), checker.numeric)
        self.assertTrue(checker.cjk, "residual Chinese on a scanned page not reported")
        self.assertIn("汪建法", checker.cjk[0])

    def test_statement_codes_exempt_from_residual_cjk(self):
        # Statement / subject codes (会商银02表, 会企01表-1) are deliberately kept
        # verbatim; their CJK must not be reported as residual Chinese.
        self.assertEqual([], _cjk_residual("Statement code 会商银01表-1 is kept."))
        # Prose around the code is still counted.
        residual = _cjk_residual("正文 中文 会商银02表 残留")
        self.assertEqual("正文中文残留", "".join(residual))

    def test_scan_like_text_detector(self):
        self.assertTrue(_is_scan_like_text(""))
        self.assertTrue(_is_scan_like_text("22"))
        self.assertFalse(_is_scan_like_text("二、公司组织架构图"))
        self.assertFalse(_is_scan_like_text("Total assets"))


if __name__ == "__main__":
    unittest.main()
