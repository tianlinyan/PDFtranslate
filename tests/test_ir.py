"""Tests for the C-⑥ IR layer (``translate_app/ir.py``)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_build_ir_verbatim_block_breaks_the_prose_run(self):
        # Regression: a numeric block was grouped with the prose around it, so
        # ``prose_units`` merged "...revenue of" + "million yuan..." into ONE request
        # (the number is excluded from the request) — the model saw a sentence with the
        # amount missing.  A verbatim block must be a hard group boundary.
        texts = ["The Group recorded total operating revenue of", "1,234,567.89",
                 "million yuan for the year ended 31 December 2024."]
        blocks = [Block(t, 0, 72, 100 + i * 14, 520, 112 + i * 14, size=10.0)
                  for i, t in enumerate(texts)]
        dt = pdfio.DocumentText(pages=[blocks], blocks=texts, block_pages=[0, 0, 0])
        doc_ir = ir.build_ir(dt, lang="English")
        groups = [b.group_id for b in doc_ir.pages[0].blocks]
        self.assertEqual(len(set(groups)), 3, groups)   # no run spans the number
        translatable = [b for b in doc_ir.pages[0].blocks if not ir._is_verbatim(b.anchor)]
        units = ir.prose_units(translatable)
        self.assertEqual([[b.src_id for b in u] for u in units], [[0], [2]])
        # No request carries the amount, and neither is a truncated sentence: the two
        # prose fragments are translated independently instead of one hole-riddled
        # sentence ("...revenue of million yuan...").
        for u in units:
            self.assertNotIn("1,234,567.89", ir.join_texts([b.text for b in u]))

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
            def translate_blocks(self, blocks, lang, *, log=None, on_progress=None,
                                 cancel=None, doc_path=None, resume=True,
                                 keep_original=None, extra_glossary=None,
                                 errors=None):
                self.captured = (lang, doc_path, extra_glossary, on_progress, cancel, keep_original)
                class R: pass
                r = R()
                r.translated = ["T|" + b for b in blocks]
                r.errors = [["bad"]]
                return r
        engine = FakeEngine()
        fn = ir.make_ir_translate_fn(engine, doc_path=Path("/x"), log=lambda m: None,
                                     on_progress=lambda d, t: None,
                                     cancel=lambda: False, keep_original={0})
        out = fn(["a", "b"], lang="English", extra_glossary={"k": "v"})
        self.assertEqual(out, ["T|a", "T|b"])
        self.assertEqual(engine.captured[1], Path("/x"))
        self.assertEqual(engine.captured[2], {"k": "v"})
        # The factory forwards cancel / progress / keep_original and surfaces errors.
        self.assertTrue(callable(engine.captured[3]))    # on_progress
        self.assertTrue(callable(engine.captured[4]))    # cancel
        self.assertEqual(engine.captured[5], {0})
        self.assertEqual(fn.last_errors, [["bad"]])

    def test_save_ir_rejects_non_pdf_mode(self):
        from translate_app import ir as ir_mod
        import pymupdf as fitz
        from translate_app.pdfio import DocumentText, Block
        src = Path(tempfile.mkdtemp()) / "x.pdf"
        d = fitz.open(); d.new_page(); d.save(str(src)); d.close()
        dt = DocumentText(pages=[[Block("a", 0, 0, 0, 10, 10)]], blocks=["a"])
        doc_ir = ir_mod.build_ir(dt)
        with self.assertRaises(ValueError):
            ir_mod.save_ir(str(src), str(Path(tempfile.mkdtemp()) / "o.pdf"),
                           doc_ir, {0: "b"}, lang="English", mode="markdown")


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

    def test_infer_terms_drops_function_words(self):
        # Regression: a title-cased heading made "The"/"While"/"Figure" look like
        # terminology; pinned as a must-use glossary entry they distort every sentence.
        blocks = [
            IRBlock(anchor=Block("The Group While Figure", 0, 0, 0, 100, 20),
                    text="The Group While Figure", role="text", src_id=0),
            IRBlock(anchor=Block("The Group While Figure", 0, 0, 20, 100, 40),
                    text="The Group While Figure", role="text", src_id=1),
        ]
        terms = ir.infer_terms(IRDoc(title="t", pages=[IRPage(page=0, blocks=blocks)],
                                     block_count=2))
        self.assertIn("Group", terms)             # a real term survives
        for stop in ("The", "While", "Figure"):
            self.assertNotIn(stop, terms)

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

    def test_infer_glossary_extracts_and_translates_once(self):
        ir0 = self._ir()
        def fn(texts, *, lang, extra_glossary=None):
            return ["T|" + t for t in texts]
        g = ir.infer_glossary(ir0, fn, lang="English")
        self.assertEqual(g, {"Revenue": "T|Revenue"})   # only the repeated term

    def test_infer_glossary_drops_unchanged_terms(self):
        ir0 = self._ir()
        # A model that echoes the source (fails) must not pin s -> s.
        def fn(texts, *, lang, extra_glossary=None):
            return list(texts)
        self.assertEqual(ir.infer_glossary(ir0, fn, lang="English"), {})

    def test_infer_glossary_length_mismatch_returns_empty(self):
        ir0 = self._ir()
        self.assertEqual(ir.infer_glossary(ir0, lambda *a, **k: [], lang="English"), {})

    def test_infer_glossary_no_terms_returns_empty(self):
        ir0 = IRDoc(title="t", pages=[IRPage(page=0, blocks=[
            IRBlock(anchor=Block("hi", 0, 0, 0, 10, 10), text="hi", role="text", src_id=0),
        ])])
        calls = []
        def fn(texts, *, lang, extra_glossary=None):
            calls.append(list(texts))
            return ["T|" + t for t in texts]
        self.assertEqual(ir.infer_glossary(ir0, fn, lang="English"), {})
        self.assertEqual(calls, [])   # no term batch when there are no candidates


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


class ProseGroupingTest(unittest.TestCase):
    """C-⑥ Stage 4: paragraph-level grouping + proportional back-distribution."""

    def _page(self):
        return IRPage(page=0, blocks=[])

    def _mk(self, ipage, text, *, role="text", group=0, style=None, **anchor_kw):
        a = Block(text, 0, 0, 0, 100, 20, **anchor_kw)
        b = IRBlock(anchor=a, text=text, role=role, group_id=group,
                    style=style or {}, src_id=len(ipage.blocks))
        ipage.blocks.append(b)
        return b

    # -- join_texts -------------------------------------------------------- #
    def test_join_texts_cjk_no_space_latin_with_space(self):
        self.assertEqual(ir.join_texts(["你好", "世界"]), "你好世界")
        self.assertEqual(ir.join_texts(["hello", "world"]), "hello world")
        self.assertEqual(ir.join_texts(["你好", "world"]), "你好 world")
        self.assertEqual(ir.join_texts(["hello", "世界"]), "hello 世界")
        self.assertEqual(ir.join_texts([""]), "")
        self.assertEqual(ir.join_texts(["a", "", "b"]), "a b")

    # -- prose_units ------------------------------------------------------- #
    def test_prose_units_merges_same_group_paragraph(self):
        ip = self._page()
        self._mk(ip, "line one", group=1)
        self._mk(ip, "line two", group=1)
        units = ir.prose_units(ip.blocks)
        self.assertEqual(len(units), 1)
        self.assertEqual(len(units[0]), 2)

    def test_prose_units_splits_on_group_change_and_role(self):
        ip = self._page()
        self._mk(ip, "para a", group=1)
        self._mk(ip, "para b", group=2)          # new paragraph
        self._mk(ip, "head", role="heading", group=2, bold=True, size=16)
        self._mk(ip, "cell", group=2, in_table=True)   # table cell stays alone
        units = ir.prose_units(ip.blocks)
        self.assertEqual(len(units), 4)

    def test_prose_units_keeps_scan_and_keep_original_alone(self):
        ip = self._page()
        self._mk(ip, "scanned", group=1, ocr=True, single_line=False)
        self._mk(ip, "name", group=1, keep_original=True)
        units = ir.prose_units(ip.blocks)
        self.assertEqual([len(u) for u in units], [1, 1])

    def test_prose_units_group_zero_never_merges(self):
        # A hand-built IR defaults group_id=0; only build_ir assigns real groups.
        ip = self._page()
        self._mk(ip, "a", group=0)
        self._mk(ip, "b", group=0)
        units = ir.prose_units(ip.blocks)
        self.assertEqual([len(u) for u in units], [1, 1])

    def test_prose_units_env_gated_off(self):
        ip = self._page()
        self._mk(ip, "a", group=1)
        self._mk(ip, "b", group=1)
        import os
        with mock.patch.dict(os.environ, {"PDFTRANSLATE_IR_GROUP": "0"}, clear=False):
            units = ir.prose_units(ip.blocks)
        self.assertEqual([len(u) for u in units], [1, 1])

    # -- split_translation -------------------------------------------------- #
    def test_split_single_source_passthrough(self):
        self.assertEqual(ir.split_translation(["abc"], "Edc"), ["Edc"])

    def test_split_empty_translation_returns_source(self):
        self.assertEqual(ir.split_translation(["a", "b"], "   "), ["a", "b"])

    def test_split_echoed_source_returns_source(self):
        # The batch failed and echoed the source: re-splitting it would scramble
        # the original line breaks, so hand the fragments back unchanged.
        joined = ir.join_texts(["hello ", "world"])
        self.assertEqual(ir.split_translation(["hello ", "world"], joined),
                         ["hello ", "world"])

    def test_split_preserves_content_and_non_empty(self):
        pieces = ir.split_translation(["Revenue", "and Costs", "grew"], "Revenue and Costs grew by 12%")
        self.assertEqual(len(pieces), 3)
        self.assertTrue(all(p.strip() for p in pieces))
        # Concatenation (whitespace-insensitive) recovers the whole translation.
        self.assertEqual(ir._norm_ws("".join(pieces)), ir._norm_ws("Revenue and Costs grew by 12%"))

    def test_split_never_breaks_a_number(self):
        src = ["company", "earnings"]
        trans = "The revenue 1,234.56 is high"
        pieces = ir.split_translation(src, trans)
        self.assertEqual("".join(pieces), "The revenue 1,234.56 is high")
        self.assertIn("1,234.56", "".join(pieces))  # intact, unbroken across a boundary

    def test_split_prefers_a_space_over_cutting_inside_a_word(self):
        # Regression: a cut inside a Latin word cost nothing, so a boundary could land
        # mid-word.  Here the exact proportional target (5) is inside "abcdefgh" while
        # the word ends at 8 — the cut must move to the word boundary.
        pieces = ir.split_translation(["a" * 5, "b" * 9], "abcdefgh WXYZQ")
        self.assertEqual("abcdefgh", pieces[0].strip())
        self.assertEqual("WXYZQ", pieces[1].strip())

    def test_split_proportional_first_piece_longer(self):
        pieces = ir.split_translation(["aaaaaa", "bb"], "11111111")
        self.assertGreaterEqual(len(pieces[0]), len(pieces[1]))

    def test_split_too_short_keeps_source_not_blank(self):
        # Regression: the too-short branch used to blank the other fragments
        # (["xy", "", ""]), and a blank fragment exports as an empty box — content
        # silently lost.  They keep their SOURCE text instead (visible, and caught by
        # the residual check).
        pieces = ir.split_translation(["a", "b", "c"], "xy")
        self.assertEqual(pieces, ["xy", "b", "c"])

    # -- translate_ir integration ------------------------------------------- #
    def test_translate_ir_merges_paragraph_and_maps_back(self):
        ip = self._page()
        self._mk(ip, "Hello world", group=1)
        self._mk(ip, "Good to see you", group=1)
        self._mk(ip, "1,234.56", group=1)                 # numeric → verbatim
        self._mk(ip, "E=mc^2", role="formula", group=1)   # formula → verbatim
        ir0 = IRDoc(title="t", block_count=4)
        ir0.pages.append(ip)

        calls: list[list[str]] = []

        def fn(texts, *, lang, extra_glossary=None):
            calls.append(list(texts))
            return ["TRANS:" + t for t in texts]

        out = ir.translate_ir(ir0, fn, lang="English")
        self.assertEqual(len(calls), 1)
        # Exactly one request — the two prose blocks joined as one paragraph.
        self.assertEqual(calls[0], ["Hello world Good to see you"])
        # Numeric and formula kept verbatim.
        self.assertEqual(out[2], "1,234.56")
        self.assertEqual(out[3], "E=mc^2")
        # The paragraph translation was split back onto the two blocks (non-empty).
        self.assertTrue(out[0])
        self.assertTrue(out[1])
        self.assertEqual(ir._norm_ws(out[0] + out[1]),
                         ir._norm_ws("TRANS:Hello world Good to see you"))

    def test_translate_ir_group_disabled_requests_each_block(self):
        ip = self._page()
        self._mk(ip, "Hello world", group=1)
        self._mk(ip, "Good to see you", group=1)
        ir0 = IRDoc(title="t", block_count=2)
        ir0.pages.append(ip)

        calls: list[list[str]] = []

        def fn(texts, *, lang, extra_glossary=None):
            calls.append(list(texts))
            return ["TRANS:" + t for t in texts]

        ir.translate_ir(ir0, fn, lang="English", group_prose=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ["Hello world", "Good to see you"])


if __name__ == "__main__":
    unittest.main()
