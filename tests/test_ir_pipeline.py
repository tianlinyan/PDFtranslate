"""Tests for the end-to-end IR pipeline (``translate_app/ir_pipeline.py``)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pymupdf as fitz

from translate_app import ir_pipeline, pdfio
from translate_app.ir import IRBlock, IRDoc, IRPage

from tests._helpers import build_sample_pdf


def _mock_translate(texts, *, lang, extra_glossary=None):
    """A translation mock: every text becomes ``T|<text>`` (no network)."""
    return ["T|" + t for t in texts]


class RunIrPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = build_sample_pdf(Path(self.tmp.name) / "pipe_src.pdf", pages=2)

    def test_end_to_end_translated_pdf(self):
        out, doc_ir, translated = ir_pipeline.run_ir_pipeline(
            self.src, lang="English", translate_fn=_mock_translate)
        self.assertTrue(out.exists())
        self.assertEqual(len(doc_ir.pages), 2)
        self.assertTrue(translated)
        # The output PDF carries the mock translations.
        d = fitz.open(str(out))
        try:
            text = "".join(d[i].get_text() for i in range(d.page_count))
            self.assertIn("T|", text)
        finally:
            d.close()

    def test_end_to_end_bilingual(self):
        out, _doc_ir, _tr = ir_pipeline.run_ir_pipeline(
            self.src, lang="English", translate_fn=_mock_translate, mode="bilingual_pdf")
        d = fitz.open(str(out))
        try:
            self.assertEqual(d.page_count, 2 * 2)  # 2 source + 2 mirror translation pages
        finally:
            d.close()

    def test_end_to_end_with_structure_skips_formula(self):
        # A structure backend marks page 0's first block as formula → kept verbatim
        # (not prefixed with "T|").  This exercises the semantic path end-to-end.
        def structure_fn(page_index, page, blocks):
            if page_index != 0 or not blocks:
                return []
            b = blocks[0]
            return [{"kind": "formula", "bbox": [b.x0, b.y0, b.x1, b.y1]}]
        out, doc_ir, translated = ir_pipeline.run_ir_pipeline(
            self.src, lang="English", translate_fn=_mock_translate,
            structure_fn=structure_fn, parser="mock")
        # There is at least one formula block that was kept verbatim (no "T|" prefix).
        self.assertTrue(any(blk.role == "formula" for pg in doc_ir.pages for blk in pg.blocks))

    def test_run_ir_pipeline_returns_modifiers(self):
        # The pipeline returns a Path, an IRDoc and a src_id→text map usable by out_doc.
        out, doc_ir, translated = ir_pipeline.run_ir_pipeline(
            self.src, lang="English", translate_fn=_mock_translate)
        self.assertIsInstance(out, Path)
        self.assertIsInstance(doc_ir, IRDoc)
        self.assertTrue(all(isinstance(k, int) for k in translated))


if __name__ == "__main__":
    unittest.main()
