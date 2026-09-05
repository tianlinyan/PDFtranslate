"""Tests for the persistent document context and the interaction-chat tools."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pymupdf as fitz

from translate_app import chat_tools
from translate_app.doc_context import DocContext

from tests._helpers import build_sample_pdf


def _build_numeric_pdf(path: str | Path) -> Path:
    """A tiny PDF whose only text is a pure amount cell (must never be rewritten)."""
    path = Path(path)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), "1,234.56", fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


class _CtxTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        # Keep caches out of the user's home so cold, clean extraction is guaranteed.
        patcher = mock.patch.dict(os.environ, {
            "PDFTRANSLATE_CACHE_DIR": str(self.tmp / "cache"),
            "PDFTRANSLATE_OCR_CACHE_DIR": str(self.tmp / "ocr"),
        })
        patcher.start()
        self.addCleanup(patcher.stop)


class DocContextTest(_CtxTest):
    def test_set_source_resets_overlay_only_on_path_change(self):
        a = build_sample_pdf(self.tmp / "a.pdf", pages=1)
        b = build_sample_pdf(self.tmp / "b.pdf", pages=1)
        ctx = DocContext()
        ctx.set_source(str(a))
        ctx.set_overlay(0, "hello")
        self.assertTrue(ctx.has_source())
        # Re-pointing at the same path keeps the overlay (a re-run of one doc).
        ctx.set_source(str(a))
        self.assertIsNotNone(ctx.get_overlay(0))
        # A different file drops inherited edits.
        ctx.set_source(str(b))
        self.assertIsNone(ctx.get_overlay(0))

    def test_ensure_doc_extracts_lazily_and_caches(self):
        ctx = DocContext()
        self.assertIsNone(ctx.ensure_doc())
        path = build_sample_pdf(self.tmp / "s.pdf", pages=1)
        ctx.set_source(str(path))
        doc = ctx.ensure_doc()
        self.assertIsNotNone(doc)
        self.assertTrue(doc.blocks)
        self.assertIs(doc, ctx.ensure_doc())  # cached

    def test_overlay_set_and_delete(self):
        ctx = DocContext()
        ctx.set_source(str(build_sample_pdf(self.tmp / "o.pdf", pages=1)))
        self.assertTrue(ctx.set_overlay(3, "译文"))
        self.assertEqual("译文", ctx.get_overlay(3)["text"])
        self.assertEqual([3], list(ctx.overlay()))
        self.assertTrue(ctx.set_overlay(3, None, action="delete"))
        self.assertIsNone(ctx.get_overlay(3))


class ChatToolsTest(_CtxTest):
    def setUp(self) -> None:
        super().setUp()
        self.src = build_sample_pdf(self.tmp / "src.pdf", pages=1)
        self.ctx = DocContext()
        self.ctx.set_source(str(self.src))
        self.tools = chat_tools.make_chat_tools(self.ctx)

    def _tool_names(self):
        return set(self.tools)

    def test_tools_expose_expected_set(self):
        self.assertEqual({
            "get_doc_info", "get_settings", "classify_page", "read_page", "goto_page",
            "set_block_text", "delete_block_text", "apply_annotation", "re_export",
            "run_translate", "set_setting", "self_check", "run_flow", "retranslate",
        }, self._tool_names())

    # ---- retranslate: the chat can re-translate specific blocks/page in place ----

    def _retranslate_tools(self, translate_texts=None):
        return chat_tools.make_chat_tools(self.ctx, translate_texts=translate_texts)

    def test_retranslate_requires_engine(self):
        # No translation channel wired → fail-closed, not a crash.
        res = self.tools["retranslate"](0)
        self.assertFalse(res["ok"])
        self.assertIn("重译通道", res["error"])

    def test_retranslate_whole_page_writes_overlay(self):
        tools = self._retranslate_tools(lambda texts, lang: ["Translated text"] * len(texts))
        n = len(self.tools["read_page"](0)["blocks"])
        res = tools["retranslate"](0)
        self.assertTrue(res["ok"], res)
        self.assertEqual(n, res["count"])
        self.assertEqual(n, len(res["indices"]))
        self.assertEqual([], res["failed"])
        for i in res["indices"]:
            self.assertEqual("Translated text", self.ctx.get_overlay(i)["text"])

    def test_retranslate_specific_indices(self):
        tools = self._retranslate_tools(lambda texts, lang: ["Fixed " + t for t in texts])
        idx = self.tools["read_page"](0)["blocks"][0]["index"]
        res = tools["retranslate"](0, [idx])
        self.assertTrue(res["ok"], res)
        self.assertEqual([idx], res["indices"])
        self.assertTrue(self.ctx.get_overlay(idx)["text"].startswith("Fixed "))

    def test_retranslate_reports_failed_blocks(self):
        # The model returns the source verbatim → no write; those blocks are "failed"
        # and the tool must say so (failure transparency).
        tools = self._retranslate_tools(lambda texts, lang: list(texts))
        n = len(self.tools["read_page"](0)["blocks"])
        res = tools["retranslate"](0)
        self.assertTrue(res["ok"], res)
        self.assertEqual(0, res["count"])
        self.assertEqual(n, len(res["failed"]))
        self.assertTrue(res["note"])  # mentions the failed blocks

    def test_retranslate_skips_numeric_only_page(self):
        num_ctx = DocContext()
        num_ctx.set_source(str(_build_numeric_pdf(self.tmp / "num2.pdf")))
        tools = chat_tools.make_chat_tools(
            num_ctx, translate_texts=lambda texts, lang: ["x"] * len(texts))
        res = tools["retranslate"](0)
        self.assertFalse(res["ok"])
        self.assertIn("没有可重译的块", res["error"])

    def test_retranslate_bad_page(self):
        tools = self._retranslate_tools(lambda texts, lang: ["x"] * len(texts))
        res = tools["retranslate"](99)
        self.assertFalse(res["ok"])
        self.assertIn("bad page", res["error"])

    def test_retranslate_call_failure_is_transparent(self):
        # The translation call itself raises → fail-closed, all picked blocks reported.
        tools = self._retranslate_tools(lambda texts, lang: (_ for _ in ()).throw(RuntimeError("boom")))
        res = tools["retranslate"](0)
        self.assertFalse(res["ok"])
        self.assertIn("boom", res["error"])
        self.assertTrue(res["failed"])
        self.assertTrue(self.ctx.ensure_doc().blocks)

    # ---- run_flow: an ``auto_fix`` flow can re-translate problem blocks ----

    def test_run_flow_stays_read_only_when_channel_missing(self):
        # No translation channel → mode read_only (existing behaviour).
        res = self.tools["run_flow"]("自检只查数字，第1页")
        self.assertTrue(res["ok"], res)
        self.assertEqual("read_only", res["mode"])
        self.assertEqual(0, res["fixed_blocks"])

    def test_run_flow_fixes_in_place_when_channel_wired(self):
        tools = self._retranslate_tools(lambda texts, lang: ["Translated text"] * len(texts))
        res = tools["run_flow"]("自检第1页残留和漏译")
        self.assertTrue(res["ok"], res)
        self.assertEqual("self_check_page", res["base"])
        self.assertEqual("fixed", res["mode"])
        self.assertGreater(res["fixed_blocks"], 0)
        self.assertEqual(0, res["remaining_issue_count"])
        self.assertTrue(res["clean"], res)
        # The fix wrote the protected overlay, so the chat's read_page picks it up.
        overlay = self.ctx.overlay()
        self.assertTrue(overlay)

    def test_run_flow_read_only_when_spec_says_no_fix(self):
        tools = self._retranslate_tools(lambda texts, lang: ["x"] * len(texts))
        # "不修改" → auto_fix False → read-only even though the channel is wired.
        res = tools["run_flow"]("自检第1页残留，不修改")
        self.assertTrue(res["ok"], res)
        self.assertEqual("read_only", res["mode"])
        self.assertEqual(0, res["fixed_blocks"])

    def test_run_flow_reports_unfixable_blocks(self):
        # The model keeps the source → those blocks stay untranslated and are reported.
        tools = self._retranslate_tools(lambda texts, lang: list(texts))
        res = tools["run_flow"]("自检第1页残留和漏译")
        self.assertTrue(res["ok"], res)
        self.assertEqual("fixed", res["mode"])
        self.assertGreater(res["failed_blocks"], [])
        self.assertGreater(res["remaining_issue_count"], 0)
        self.assertFalse(res["clean"], res)

    def test_get_doc_info(self):
        info = self.tools["get_doc_info"]()
        self.assertGreaterEqual(info["pages"], 1)
        self.assertGreater(info["block_count"], 0)

    def test_read_page_returns_flat_blocks_with_status(self):
        out = self.tools["read_page"](0)
        self.assertEqual(0, out["page"])
        self.assertTrue(out["blocks"])
        first = out["blocks"][0]
        self.assertIn("index", first)
        self.assertIn("source", first)
        self.assertIn("translated", first)
        self.assertIn("bbox", first)

    def test_read_page_falls_back_to_last_translation(self):
        # A prior run's full translation is shown so the AI knows what is already
        # translated and does not re-edit it.
        n = len(self.ctx.ensure_doc().blocks)
        self.ctx.set_last_translated([f"T{i}" for i in range(n)])
        out = self.tools["read_page"](0)
        self.assertEqual(0, out["blocks"][0]["index"])
        self.assertEqual("T0", out["blocks"][0]["translated"])

    def test_set_block_text_writes_protected_overlay(self):
        block = self.tools["read_page"](0)["blocks"][0]
        idx = block["index"]
        res = self.tools["set_block_text"](idx, "改写的译文")
        self.assertTrue(res["ok"])
        self.assertEqual("改写的译文", self.ctx.get_overlay(idx)["text"])

    def test_set_block_text_refuses_numeric_cell(self):
        num_ctx = DocContext()
        num_ctx.set_source(str(_build_numeric_pdf(self.tmp / "num.pdf")))
        num_tools = chat_tools.make_chat_tools(num_ctx)
        idx = num_tools["read_page"](0)["blocks"][0]["index"]
        res = num_tools["set_block_text"](idx, "x")
        self.assertFalse(res["ok"])
        self.assertIn("数字", res["error"])
        self.assertIsNone(num_ctx.get_overlay(idx))

    def test_delete_block_text(self):
        idx = self.tools["read_page"](0)["blocks"][0]["index"]
        self.tools["set_block_text"](idx, "tmp")
        self.assertTrue(self.tools["delete_block_text"](idx)["ok"])
        self.assertIsNone(self.ctx.get_overlay(idx))
        # Second delete: no overlay entry → not ok (nothing to undo).
        self.assertFalse(self.tools["delete_block_text"](idx)["ok"])

    def test_apply_annotation_maps_bbox_to_block(self):
        block = self.tools["read_page"](0)["blocks"][0]
        idx = block["index"]
        res = self.tools["apply_annotation"](0, block["bbox"], text="标注改写")
        self.assertTrue(res["ok"], res)
        self.assertEqual(idx, res["index"])
        self.assertEqual("标注改写", self.ctx.get_overlay(idx)["text"])

    def test_apply_annotation_delete(self):
        block = self.tools["read_page"](0)["blocks"][0]
        idx = block["index"]
        self.tools["set_block_text"](idx, "tmp")
        res = self.tools["apply_annotation"](0, block["bbox"], action="delete")
        self.assertTrue(res["ok"])
        self.assertIsNone(self.ctx.get_overlay(idx))

    def test_apply_annotation_out_of_bounds(self):
        res = self.tools["apply_annotation"](99, [0, 0, 10, 10], text="x")
        self.assertFalse(res["ok"])

    def test_goto_page_requires_preview_channel(self):
        # No preview channel wired → fail-closed, not a crash.
        res = self.tools["goto_page"](0, "translation")
        self.assertFalse(res["ok"])
        self.assertIn("预览通道", res["error"])

    def test_goto_page_calls_preview_channel(self):
        calls: list = []
        tools = chat_tools.make_chat_tools(
            self.ctx, show_preview=lambda page, what: calls.append((page, what)))
        res = tools["goto_page"](2, "translation")
        self.assertTrue(res["ok"])
        self.assertEqual([(2, "translation")], calls)

    def test_re_export_requires_channel(self):
        # No re-export channel wired → fail-closed, not a crash.
        res = self.tools["re_export"]()
        self.assertFalse(res["ok"])
        self.assertIn("重新导出通道", res["error"])

    def test_re_export_calls_channel(self):
        calls: list = []
        tools = chat_tools.make_chat_tools(self.ctx, re_export=lambda: calls.append(1))
        res = tools["re_export"]()
        self.assertTrue(res["ok"])
        self.assertEqual([1], calls)

    def test_get_settings_returns_snapshot(self):
        self.ctx.set_settings(source="a.pdf", target_language="English",
                              output_type="bilingual_pdf", output_label="双语 PDF",
                              model="qwen")
        res = self.tools["get_settings"]()
        self.assertTrue(res["ok"])
        self.assertEqual("a.pdf", res["source"])
        self.assertEqual("English", res["target_language"])
        self.assertEqual("bilingual_pdf", res["output_type"])
        self.assertEqual("qwen", res["model"])

    def test_get_settings_empty_snapshot_fail_closed(self):
        tools = chat_tools.make_chat_tools(DocContext())
        res = tools["get_settings"]()
        self.assertFalse(res["ok"])
        self.assertIn("error", res)

    def test_run_translate_requires_channel(self):
        res = self.tools["run_translate"]()
        self.assertFalse(res["ok"])
        self.assertIn("开始翻译通道", res["error"])

    def test_run_translate_requires_source(self):
        # No source PDF loaded → the tool must fail-closed, not report success while
        # nothing actually starts.
        tools = chat_tools.make_chat_tools(DocContext(), start_translate=lambda _r: None)
        res = tools["run_translate"]()
        self.assertFalse(res["ok"])
        self.assertIn("源文件", res["error"])

    def test_run_translate_calls_channel_with_requirement(self):
        calls: list = []
        tools = chat_tools.make_chat_tools(self.ctx,
                                           start_translate=lambda req: calls.append(req))
        res = tools["run_translate"]("第3页公司名翻成Bank")
        self.assertTrue(res["ok"])
        self.assertEqual(["第3页公司名翻成Bank"], calls)

    def test_set_setting_validates_and_calls_channel(self):
        calls: list = []
        tools = chat_tools.make_chat_tools(self.ctx,
                                           set_setting=lambda k, v: calls.append((k, v)))
        self.assertTrue(tools["set_setting"]("target_language", "French")["ok"])
        self.assertEqual([("target_language", "French")], calls)
        bad = tools["set_setting"]("nope", "x")
        self.assertFalse(bad["ok"])
        self.assertIn("未知设置项", bad["error"])
        # An invalid output_type value is likewise rejected (not silently "success").
        bad_otype = tools["set_setting"]("output_type", "docx")
        self.assertFalse(bad_otype["ok"])
        self.assertIn("未知输出格式", bad_otype["error"])
        self.assertTrue(tools["set_setting"]("output_type", "bilingual_pdf")["ok"])

    def test_set_setting_requires_channel(self):
        res = self.tools["set_setting"]("target_language", "French")
        self.assertFalse(res["ok"])
        self.assertIn("设置通道", res["error"])

    def test_classify_page_ok(self):
        res = self.tools["classify_page"](0)
        self.assertIn("kind", res)

    def test_self_check_returns_structured_report(self):
        # Nothing translated yet → the deterministic audit reports residual/missing.
        res = self.tools["self_check"](0)
        self.assertTrue(res["ok"], res)
        self.assertIn("checks_requested", res)
        self.assertIn("issues", res)
        self.assertIn("clean", res)
        self.assertFalse(res["clean"])

    def test_self_check_filters_checks(self):
        res = self.tools["self_check"](0, checks=["numbers"])
        self.assertTrue(res["ok"], res)
        self.assertEqual(["numbers"], res["checks_requested"])

    def test_self_check_clean_when_translated(self):
        # With a committed translation (no CJK residual, nothing missing) the
        # residual+missing checks come back clean.
        n = len(self.ctx.ensure_doc().blocks)
        self.ctx.set_last_translated([f"Translated text {i}" for i in range(n)])
        res = self.tools["self_check"](0, checks=["residual", "missing"])
        self.assertTrue(res["ok"], res)
        self.assertTrue(res["clean"], res)

    def test_self_check_requires_source(self):
        tools = chat_tools.make_chat_tools(DocContext())
        res = tools["self_check"]()
        self.assertFalse(res["ok"])
        self.assertIn("没有已加载的 PDF", res["error"])

    def test_run_flow_compiles_and_audits_scope(self):
        res = self.tools["run_flow"]("自检只查数字，第1页")
        self.assertTrue(res["ok"], res)
        self.assertEqual("self_check_page", res["base"])
        self.assertEqual(["numbers"], res["checks"])
        self.assertEqual([0], res["scope"])
        self.assertEqual(1, res["pages_audited"])
        self.assertIn("issue_count", res)

    def test_run_flow_redirects_translate_base(self):
        res = self.tools["run_flow"]("重译第2页")
        self.assertFalse(res["ok"])
        self.assertEqual("translate_page", res["base"])
        self.assertIn("run_translate", res["error"])

    def test_run_flow_redirects_export_base(self):
        res = self.tools["run_flow"]("重新导出")
        self.assertFalse(res["ok"])
        self.assertEqual("export", res["base"])
        self.assertIn("re_export", res["error"])

    def test_run_flow_promotes_named_flow_in_memory(self):
        from translate_app.agent import user_flows as uf

        saved = dict(uf.USER_FLOW_SPECS)
        try:
            res = self.tools["run_flow"]("自检只查表格", name="my_table_check")
            self.assertTrue(res["ok"], res)
            self.assertTrue(res["promoted"])
            self.assertIn("my_table_check", uf.USER_FLOW_SPECS)
            self.assertEqual(["table"], uf.USER_FLOW_SPECS["my_table_check"].checks)
        finally:
            uf.USER_FLOW_SPECS.clear()
            uf.USER_FLOW_SPECS.update(saved)


if __name__ == "__main__":
    unittest.main()
