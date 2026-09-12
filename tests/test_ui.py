"""Tests for the pure (Qt-free) helpers of the main window.

Only functions that need no ``QApplication`` are exercised here, so the suite
stays runnable on a headless machine.
"""

from __future__ import annotations

import unittest

from translate_app.main_window import (
    LANGUAGES,
    MainWindow,
    parse_preview_command,
    resolve_language,
)


class ResolveLanguageTest(unittest.TestCase):
    def test_every_listed_label_maps_to_its_model_name(self):
        for label, code in LANGUAGES:
            self.assertEqual(code, resolve_language(label))

    def test_language_name_maps_to_itself(self):
        self.assertEqual("Simplified Chinese", resolve_language("Simplified Chinese"))
        self.assertEqual("German", resolve_language("  german  "))

    def test_typed_language_is_honoured(self):
        """Regression: the combo is editable, but the code read
        ``currentData()`` — which still points at the previously selected item
        when the typed text matches nothing.  A typed language was silently
        replaced by the last selected one, in both the request and the output
        file name."""
        self.assertEqual("Português", resolve_language("Português"))
        self.assertEqual("Japanese", resolve_language("Japanese"))

    def test_empty_input_falls_back_to_the_first_language(self):
        self.assertEqual(LANGUAGES[0][1], resolve_language(""))
        self.assertEqual(LANGUAGES[0][1], resolve_language("   "))


class ParsePreviewCommandTest(unittest.TestCase):
    """M5: sidebar preview-navigation commands are recognised."""

    def test_next_prev(self):
        self.assertEqual(("next", None, None), parse_preview_command("下一页"))
        self.assertEqual(("prev", None, None), parse_preview_command("上一页"))

    def test_goto_page(self):
        self.assertEqual(("goto", 2, None), parse_preview_command("第 3 页"))
        self.assertEqual(("goto", 2, None), parse_preview_command("显示第 3 页"))
        self.assertEqual(("goto", 0, None), parse_preview_command("第 1 页"))

    def test_goto_with_side_and_chinese_numeral(self):
        self.assertEqual(("goto", 1, "translation"), parse_preview_command("打开译文第二页"))
        self.assertEqual(("goto", 2, "translation"), parse_preview_command("打开译文第三页"))
        self.assertEqual(("goto", 2, "translation"), parse_preview_command("预览译文第三页"))
        self.assertEqual(("goto", 1, "source"), parse_preview_command("预览原文第二页"))
        self.assertEqual(("goto", 12, "translation"), parse_preview_command("预览译文第十三页"))
        # "翻译" is a shorthand for the translation side (regression).
        self.assertEqual(("goto", 2, "translation"), parse_preview_command("打开翻译第 3 页"))

    def test_not_a_navigation_command(self):
        self.assertIsNone(parse_preview_command("把第 3 页公司名换成 Bank"))
        self.assertIsNone(parse_preview_command(""))
        self.assertIsNone(parse_preview_command("   "))


class TranslationOutputPageTest(unittest.TestCase):
    """Preview bug: the "译文" side showed only the source after export.

    The source page → exported-PDF page mapping (in-place vs bilingual) must be
    right so the preview actually renders the translated output, not the source.
    """

    def test_inplace_pdf_keeps_same_page_index(self):
        # translated_pdf overlays the translation in place → same page index.
        self.assertEqual(3, MainWindow._translation_output_page(None, 3, "translated_pdf"))

    def test_inplace_pdf_uses_the_reported_page_map(self):
        # expand_pages 后源页 i 不再等于输出页 i：预览必须用导出返回的映射，
        # 越界的映射则退回原页号。
        self.assertEqual(2, MainWindow._translation_output_page(
            None, 1, "translated_pdf", [0, 2, 3]))
        self.assertEqual(5, MainWindow._translation_output_page(
            None, 5, "translated_pdf", [0, 2, 3]))

    def test_bilingual_pdf_mirrors_to_2i_plus_1(self):
        # bilingual inserts a translation page after every source page.
        self.assertEqual(1, MainWindow._translation_output_page(None, 0, "bilingual_pdf"))
        self.assertEqual(7, MainWindow._translation_output_page(None, 3, "bilingual_pdf"))

    def test_unknown_kind_keeps_index(self):
        self.assertEqual(2, MainWindow._translation_output_page(None, 2, "markdown"))


class PreviewOutputPagingTest(unittest.TestCase):
    """扩页产物的续页必须能在预览里翻到。

    以源页号为单位翻页时，源页 i 的续页 i+1… 在「译文」侧没有任何入口：实测源 1 页
    → 产物 2 页，窗口只给「第 1/1 页」，被搬走的表格行永远看不到。现在扩页产物改成
    按**输出页**翻页，并把输出页反查回源页写进标题。
    """

    def test_only_an_expanded_inplace_product_uses_output_pages(self):
        self.assertFalse(MainWindow._uses_output_paging("translated_pdf", None))
        self.assertFalse(MainWindow._uses_output_paging("translated_pdf", [0, 1]))
        self.assertFalse(MainWindow._uses_output_paging("bilingual_pdf", [0, 2]))
        self.assertFalse(MainWindow._uses_output_paging("markdown", [0, 2]))
        self.assertTrue(MainWindow._uses_output_paging("translated_pdf", [0, 2]))

    def test_an_output_page_maps_back_to_its_source_page(self):
        page_map = [0, 2, 4]
        self.assertEqual(0, MainWindow._source_page_for_output(0, page_map, 6))
        self.assertEqual(0, MainWindow._source_page_for_output(1, page_map, 6))
        self.assertEqual(1, MainWindow._source_page_for_output(2, page_map, 6))
        self.assertEqual(2, MainWindow._source_page_for_output(5, page_map, 6))
        # 映射不全（越界）时夹到最后一页，绝不抛错
        self.assertEqual(2, MainWindow._source_page_for_output(9, page_map, 6))

    def test_without_a_map_the_page_number_is_the_source_page(self):
        self.assertEqual(3, MainWindow._source_page_for_output(3, None, 10))


class ExportOverwriteTest(unittest.TestCase):
    """v0.5.24: exports overwrite the target file instead of making ``(n)`` copies.

    The rename-on-collision behaviour (``unique_path``) is gone from the export
    path, so 「重新导出」 and a normal run write the same path and there is no
    ``doc_English(1).pdf`` to reconcile.
    """

    def test_unique_path_helper_still_exists_for_the_source_guard(self):
        # The only remaining caller is the "output path == source PDF" guard in
        # ``TranslateWorker._export``; the helper itself is unchanged.
        from translate_app import pdfio

        self.assertTrue(callable(pdfio.unique_path))


if __name__ == "__main__":
    unittest.main()
