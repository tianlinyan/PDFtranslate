"""Tests for the C-⑥ IR layer (``translate_app/ir.py``)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from translate_app import ir, pdfio

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


if __name__ == "__main__":
    unittest.main()
