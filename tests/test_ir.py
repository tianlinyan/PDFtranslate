"""Tests for the C-⑥ IR layer (``translate_app/ir.py``)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from translate_app import ir, pdfio
from translate_app.ir import IRBlock, IRDoc, IRPage
from translate_app.pdfio import Block

from tests._helpers import build_sample_pdf


class BuildIrTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = build_sample_pdf(Path(self.tmp.name) / "ir_src.pdf", pages=2)

    def _doc(self):
        return pdfio.extract_document_text(str(self.src), ocr=False, log=lambda m: None)

    def test_build_ir_matches_block_count(self):
        dt = self._doc()
        doc_ir = ir.build_ir(dt, lang="English")
        self.assertEqual(doc_ir.block_count, len(dt.blocks))
        self.assertEqual(len(doc_ir.pages), dt.page_count)
        self.assertEqual(doc_ir.lang, "English")
        # Each page's block src_ids are ascending and match the flat offsets.
        flat = sum(len(pg) for pg in dt.pages)
        self.assertEqual(sum(len(p.blocks) for p in doc_ir.pages), flat)

    def test_build_ir_keeps_anchor_and_style(self):
        dt = self._doc()
        doc_ir = ir.build_ir(dt)
        blk = doc_ir.pages[0].blocks[0]
        self.assertIs(blk.anchor, dt.pages[0][0])
        self.assertEqual(blk.text, dt.pages[0][0].text)
        self.assertEqual(blk.style["size"], dt.pages[0][0].size)
        # Groups are continuous and non-decreasing within a page.
        groups = [b.group_id for b in doc_ir.pages[0].blocks]
        self.assertEqual(groups, sorted(groups))

    def test_table_binds_ref_and_groups(self):
        dt = self._doc()
        # A mock structure with a table grid over page 0's blocks.
        def structure_fn(page_index, page, blocks):
            if page_index != 0 or not blocks:
                return []
            bbox = [blocks[0].x0, blocks[0].y0, blocks[-1].x1, blocks[-1].y1]
            return [{"kind": "table", "bbox": bbox,
                     "cells": [[0, 1], [None, 2]]}]
        pdfio.build_structure(str(self.src), dt, structure_fn, parser="mock")
        doc_ir = ir.build_ir(dt)
        ipage = doc_ir.pages[0]
        self.assertTrue(ipage.tables)  # semantic table attached
        # The table region claims page-0 blocks; at least one gets a table_ref.
        refs = [b.table_ref for b in ipage.blocks]
        self.assertTrue(any(r > 0 for r in refs))
        # All cells of the table share one group id.
        table_groups = {b.group_id for b in ipage.blocks if b.table_ref > 0}
        self.assertLessEqual(len(table_groups), 1)

    def test_structural_grouping(self):
        dt = self._doc()
        doc_ir = ir.build_ir(dt)
        groups = ir.structural_groups(doc_ir)
        # Every block lands in exactly one group, and grouping is stable.
        total = sum(len(g) for g in groups)
        self.assertEqual(total, doc_ir.block_count)
        self.assertTrue(ir.is_structural_role("formula"))
        self.assertFalse(ir.is_structural_role("text"))


class TranslateIrTest(unittest.TestCase):
    def _ir(self):
        ir0 = IRDoc(title="t", block_count=3)
        ir0.pages.append(IRPage(page=0, blocks=[
            IRBlock(anchor=Block("alpha", 0, 0, 0, 100, 20), text="alpha",
                    role="text", src_id=0),
            IRBlock(anchor=Block("x^2 + y^2 = z^2", 0, 0, 0, 100, 20),
                    text="x^2 + y^2 = z^2", role="formula", src_id=1),
            IRBlock(anchor=Block("1,234.56", 0, 0, 0, 100, 20),
                    text="1,234.56", role="text", src_id=2),
        ]))
        return ir0

    def test_structural_and_verbatim_kept(self):
        ir0 = self._ir()
        out = ir.translate_ir(ir0, lambda texts, *, lang, extra_glossary=None:
                              ["BETA"], lang="English")
        # Formula + numeric are kept verbatim; the one prose block is translated.
        self.assertEqual(out[1], "x^2 + y^2 = z^2")
        self.assertEqual(out[2], "1,234.56")
        self.assertEqual(out[0], "BETA")

    def test_glossary_passed_through(self):
        ir0 = self._ir()
        seen = {}
        def fn(texts, *, lang, extra_glossary=None):
            seen["g"] = extra_glossary
            return ["BETA"]
        ir.set_terms(ir0, {"report": "报告", "revenue": "收入"})
        ir.translate_ir(ir0, fn, lang="English")
        self.assertEqual(seen["g"], {"report": "报告", "revenue": "收入"})

    def test_translate_ir_length_mismatch_raises(self):
        ir0 = self._ir()
        with self.assertRaises(ValueError):
            ir.translate_ir(ir0, lambda *a, **k: [], lang="English")

    def test_make_ir_translate_fn_binds_engine(self):
        class FakeEngine:
            def __init__(self):
                self.captured = None
            def translate_blocks(self, blocks, lang, *, log=None, doc_path=None,
                                 resume=True, extra_glossary=None):
                self.captured = (lang, doc_path, extra_glossary)
                class R: pass
                r = R()
                r.translated = ["T|" + b for b in blocks]
                return r
        engine = FakeEngine()
        fn = ir.make_ir_translate_fn(engine, doc_path=Path("/x"), log=lambda m: None)
        out = fn(["a", "b"], lang="English", extra_glossary={"k": "v"})
        self.assertEqual(out, ["T|a", "T|b"])
        self.assertEqual(engine.captured[1], Path("/x"))
        self.assertEqual(engine.captured[2], {"k": "v"})


class InferTermsTest(unittest.TestCase):
    def _ir(self):
        blocks = [
            IRBlock(anchor=Block("Revenue in 2024", 0, 0, 0, 100, 20),
                    text="Revenue in 2024", role="text", src_id=0),
            IRBlock(anchor=Block("Revenue and Costs", 0, 0, 0, 100, 20),
                    text="Revenue and Costs", role="text", src_id=1),
            IRBlock(anchor=Block("hello world", 0, 0, 0, 100, 20),
                    text="hello world", role="text", src_id=2),
        ]
        return IRDoc(title="t", pages=[IRPage(page=0, blocks=blocks)], block_count=3)

    def test_infer_terms_detects_repeated_candidates(self):
        terms = ir.infer_terms(self._ir())
        self.assertIn("Revenue", terms)   # appears twice (TitleCase)
        self.assertNotIn("hello", terms)  # appears once

    def test_translate_ir_infer_injects_computed_glossary(self):
        ir0 = self._ir()
        calls = []
        def fn(texts, *, lang, extra_glossary=None):
            calls.append({"texts": list(texts), "g": dict(extra_glossary or {})})
            return ["T|" + t for t in texts]
        ir.translate_ir(ir0, fn, lang="English", infer=True)
        # The term batch ran first, then the main batch got the inferred glossary.
        self.assertGreater(len(calls), 1)
        self.assertIn("Revenue", calls[0]["texts"])          # term minibatch
        self.assertEqual(calls[-1]["g"].get("Revenue"), "T|Revenue")  # injected
        # ir.terms now carries the computed doc glossary.
        self.assertEqual(ir0.terms.get("Revenue"), "T|Revenue")

    def test_translate_ir_infer_skips_when_terms_present(self):
        ir0 = self._ir()
        user_glossary = {"Revenue": "营业收入"}
        calls = []
        def fn(texts, *, lang, extra_glossary=None):
            calls.append({"texts": list(texts), "g": dict(extra_glossary or {})})
            return ["T|" + t for t in texts]
        ir.translate_ir(ir0, fn, lang="English", extra_glossary=user_glossary, infer=True)
        # A caller-provided glossary is honoured and no term minibatch is run.
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[-1]["g"], user_glossary)


class SaveIrTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = build_sample_pdf(Path(self.tmp.name) / "save_src.pdf", pages=2)
        self.dt = pdfio.extract_document_text(str(self.src), ocr=False, log=lambda m: None)
        self.ir = ir.build_ir(self.dt)

    def _translated(self):
        # A mock translate_fn: every translatable block becomes "T|<text>".
        return ir.translate_ir(
            self.ir,
            lambda texts, *, lang, extra_glossary=None: ["T|" + t for t in texts],
            lang="English")

    def test_save_ir_in_place(self):
        out = Path(self.tmp.name) / "out.pdf"
        ir.save_ir(str(self.src), str(out), self.ir, self._translated(), lang="English")
        import pymupdf as fitz
        doc = fitz.open(str(out))
        try:
            self.assertEqual(doc.page_count, self.dt.page_count)
            text = "".join(doc[i].get_text() for i in range(doc.page_count))
            self.assertIn("T|", text)  # translations drawn back into the PDF
        finally:
            doc.close()

    def test_save_ir_bilingual(self):
        out = Path(self.tmp.name) / "bilingual.pdf"
        ir.save_ir(str(self.src), str(out), self.ir, self._translated(),
                   lang="English", mode="bilingual_pdf")
        import pymupdf as fitz
        doc = fitz.open(str(out))
        try:
            # Bilingual: each source page followed by a mirror translation page.
            self.assertEqual(doc.page_count, self.dt.page_count * 2)
        finally:
            doc.close()

    def test_per_page_from_ir_falls_back_to_source(self):
        # A block without an entry keeps its source text.
        pages, per_page = ir.per_page_from_ir(self.ir, {})
        self.assertEqual(len(pages), len(self.ir.pages))
        self.assertEqual(per_page[0], [b.text for b in pages[0]])


if __name__ == "__main__":
    unittest.main()
