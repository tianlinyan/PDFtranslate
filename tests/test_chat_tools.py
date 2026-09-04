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
            "get_doc_info", "classify_page", "read_page", "goto_page",
            "set_block_text", "delete_block_text", "apply_annotation",
        }, self._tool_names())

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

    def test_classify_page_ok(self):
        res = self.tools["classify_page"](0)
        self.assertIn("kind", res)


if __name__ == "__main__":
    unittest.main()
