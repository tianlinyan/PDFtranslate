"""M1+M2: source-document info/triage (pdfio) and the document session controller.

Everything is offline: ``DocumentSession`` receives an injected ``translate_page``
(a fake), so no model / network is used.
"""
from __future__ import annotations

import unittest

from translate_app import agent
from translate_app import pdfio
from translate_app.agent.flow import DocumentSession, PHASE_DONE
from translate_app.settings import ModelConfig
from translate_app.translator import TranslationCancelled


def _blk(text, page=0, x0=0, y0=0, x1=50, y1=10, *, ocr=False, in_table=False,
         single_line=True):
    return pdfio.Block(
        text=text, page=page, x0=x0, y0=y0, x1=x1, y1=y1,
        size=10.0, align="left", bold=False, single_line=single_line,
        ocr=ocr, in_table=in_table,
    )


def _vertical(text, page, y0):
    return _blk(text, page=page, x0=50, y0=y0, x1=57.4, y1=y0 + 29.5, single_line=True)


def _mixed_doc():
    """One page each: normal / scan / chart / table / uncertain."""
    normal = _blk("总资产", page=0, x0=0, y0=0, x1=60, y1=10)
    scan = _blk("扫描页文字", page=1, x0=0, y0=0, x1=60, y1=10, ocr=True)
    chart = [_vertical("董事会", 2, 50), _vertical("监事会", 2, 90), _vertical("经理层", 2, 130)]
    table = [
        _blk("项目", page=3, x0=0, y0=0, x1=30, y1=10, in_table=True),
        _blk("金额", page=3, x0=0, y0=20, x1=30, y1=30, in_table=True),
        _blk("备注", page=3, x0=0, y0=40, x1=30, y1=50, in_table=True),
    ]
    uncertain = [
        _blk("标题", page=4, x0=0, y0=0, x1=30, y1=10),
        _blk("扫", page=4, x0=0, y0=20, x1=30, y1=30, ocr=True),
    ]
    doc = pdfio.DocumentText(
        pages=[[normal], [scan], chart, table, uncertain],
        blocks=["总资产", "扫描页文字", "董事会", "监事会", "经理层", "项目", "金额", "备注", "标题", "扫"],
        block_pages=[0, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        title="样表",
    )
    doc.ocr_count = 1
    return doc


class DocInfoTriageTest(unittest.TestCase):
    def test_detect_language(self):
        self.assertEqual("zh", pdfio.detect_language(["中文内容", "你好"]))
        self.assertEqual("en", pdfio.detect_language(["Hello world", "abc"]))
        self.assertEqual("mixed", pdfio.detect_language(["中文", "English"]))
        self.assertEqual("unknown", pdfio.detect_language(["123", ""]))

    def test_classify_page_kinds(self):
        self.assertEqual(pdfio.PAGE_NORMAL, pdfio.classify_page([_blk("正文")]))
        self.assertEqual(pdfio.PAGE_SCAN, pdfio.classify_page([_blk("扫", ocr=True)]))
        self.assertEqual(
            pdfio.PAGE_CHART,
            pdfio.classify_page([_vertical("董事会", 0, 50), _vertical("监事会", 0, 90),
                                 _vertical("经理层", 0, 130)]),
        )
        self.assertEqual(
            pdfio.PAGE_TABLE,
            pdfio.classify_page([_blk("a", in_table=True), _blk("b", in_table=True),
                                 _blk("c", in_table=True)]),
        )
        self.assertEqual(pdfio.PAGE_UNCERTAIN, pdfio.classify_page([_blk("标题"), _blk("扫", ocr=True)]))

    def test_get_doc_info_counts(self):
        info = pdfio.get_doc_info(_mixed_doc())
        self.assertEqual(5, info["pages"])
        self.assertEqual("样表", info["title"])
        self.assertEqual("zh", info["language"])
        self.assertEqual(1, info["text_pages"])
        self.assertEqual(1, info["scan_pages"])
        self.assertEqual(1, info["chart_pages"])
        self.assertEqual(1, info["table_pages"])
        self.assertEqual(1, info["uncertain_pages"])
        self.assertEqual(4, info["special_pages"])   # scan+chart+table+uncertain
        self.assertEqual(["normal", "scan", "chart", "table", "uncertain"], info["kinds"])


def _small_special_doc():
    """One normal page (0) + one scanned special page (1)."""
    normal = _blk("正文", page=0, x0=0, y0=0, x1=60, y1=10)
    scan = _blk("扫描页文字", page=1, x0=0, y0=0, x1=60, y1=10, ocr=True)
    doc = pdfio.DocumentText(
        pages=[[normal], [scan]], blocks=["正文", "扫描页文字"], block_pages=[0, 1], title="s",
    )
    doc.ocr_count = 1
    return doc


class DocumentSessionTest(unittest.TestCase):
    def _session(self, doc, translate_page, progress=None, cancel=None):
        state = agent.WorkflowState(src_path="a.pdf", lang="English")
        state.src_doc = doc
        session = DocumentSession(state, doc, model=object(), log=lambda m: None,
                                  translate_page=translate_page, progress=progress,
                                  cancel=cancel)
        return state, session

    def test_translates_normal_pages_first_then_marks_special(self):
        doc = _mixed_doc()
        calls: list[int] = []

        def fake_translate(st, page, _model, *, task, **kw):
            calls.append(page)
            st.out_doc = st.out_doc or {}
            st.out_doc[page] = {"text": f"T{page}"}
            return st

        state, session = self._session(doc, fake_translate)
        session.run()

        # Only the normal page (index 0) was translated, and exactly once.
        self.assertEqual([0], calls)
        # Special pages are NOT translated here (deferred to SPECIAL_PAGES phase).
        self.assertNotIn("T1", [str(v.get("text")) for v in (state.out_doc or {}).values()])
        # Phase reached DONE.
        self.assertEqual(PHASE_DONE, state.phase)
        # Doc info + triage populated by preprocess.
        self.assertIsNotNone(state.doc_info)
        self.assertEqual(5, state.doc_info.pages)
        self.assertEqual("normal", state.triage[0].kind)
        self.assertEqual("scan", state.triage[1].kind)
        self.assertEqual("chart", state.triage[2].kind)
        self.assertEqual("table", state.triage[3].kind)
        self.assertEqual("uncertain", state.triage[4].kind)
        # Special pages flagged needs_user + a decision marker.
        for i in (1, 2, 3, 4):
            self.assertEqual(agent.STATUS_NEEDS_USER, state.page(i).status)
            self.assertTrue(state.triage[i].decided)

    def test_phases_advance_to_done(self):
        doc = _mixed_doc()
        state, session = self._session(doc, lambda st, p, m, **kw: st)
        session.run()
        self.assertEqual(PHASE_DONE, state.phase)
        self.assertEqual(agent.PHASE_COMPLETED, "completed")  # constant sanity

    def test_progress_reports_preprocess_and_translate(self):
        doc = _mixed_doc()
        progress: list[tuple] = []

        def fake_translate(st, page, _model, *, task, **kw):
            st.out_doc = st.out_doc or {}
            return st

        state, session = self._session(doc, fake_translate, progress=lambda d, t, s: progress.append((d, t, s)))
        session.run()
        stages = [p[2] for p in progress]
        self.assertIn("预处理", stages)
        self.assertIn("翻译正常页", stages)

    def test_cancel_during_normal_translation_raises(self):
        doc = _mixed_doc()
        state, session = self._session(doc, lambda st, p, m, **kw: st, cancel=lambda: True)
        with self.assertRaises(TranslationCancelled):
            session.run()

    def test_special_pages_ask_and_translate_on_request(self):
        doc = _small_special_doc()
        calls: list[int] = []
        answers: list[tuple] = []
        shows: list[tuple] = []

        def fake_translate(st, page, _model, *, task, **kw):
            calls.append(page)
            st.out_doc = st.out_doc or {}
            st.out_doc[page] = {"text": f"T{page}"}
            return st

        def answer_handler(question, options, target):
            answers.append((question, options, target))
            return {"value": "OCR并翻译", "target": target}

        state = agent.WorkflowState(src_path="a.pdf", lang="English")
        state.src_doc = doc
        session = DocumentSession(state, doc, model=object(), log=lambda m: None,
                                  translate_page=fake_translate, answer_handler=answer_handler,
                                  show_preview=lambda p, w: shows.append((p, w)))
        session.run()
        # Normal page 0 first, then the special scan page 1 (translated per user).
        self.assertEqual([0, 1], calls)
        # The scan page's preview was shown and the user was asked.
        self.assertEqual([(1, "source")], shows)
        q, opts, _t = answers[0]
        self.assertIn("扫描件", q)
        self.assertIn("OCR并翻译", opts)
        # The negotiation is recorded as user-confirmed and the page done.
        self.assertTrue(any(op.tool == "ask_user" and op.user_confirmed for op in state.ops))
        self.assertEqual("translate", state.triage[1].decision)
        self.assertEqual(agent.STATUS_DONE, state.page(1).status)

    def test_special_pages_keep_when_no_answer_handler(self):
        doc = _small_special_doc()
        calls: list[int] = []

        def fake_translate(st, page, _model, *, task, **kw):
            calls.append(page)
            st.out_doc = st.out_doc or {}
            st.out_doc[page] = {"text": f"T{page}"}
            return st

        state, session = self._session(doc, fake_translate)
        session.run()
        # Only the normal page translated; the scan page is conservatively kept.
        self.assertEqual([0], calls)
        self.assertEqual("keep", state.triage[1].decision)
        self.assertEqual(agent.STATUS_NEEDS_USER, state.page(1).status)
        # The ask op is recorded but NOT user-confirmed (no channel answered).
        ask_op = next(op for op in state.ops if op.tool == "ask_user")
        self.assertFalse(ask_op.user_confirmed)


class NearestBlockTest(unittest.TestCase):
    """M6: a user-drawn region maps back to the nearest source block."""

    def test_matches_block_under_bbox(self):
        blocks = [_blk("总资产", page=0, x0=0, y0=0, x1=50, y1=10),
                  _blk("总负债", page=0, x0=0, y0=20, x1=50, y1=30)]
        b = pdfio.nearest_block(blocks, [0, 20, 50, 30])
        self.assertIsNotNone(b)
        self.assertEqual("总负债", b.text)

    def test_bbox_that_hits_no_block_returns_none(self):
        blocks = [_blk("总资产", page=0, x0=0, y0=0, x1=50, y1=10)]
        self.assertIsNone(pdfio.nearest_block(blocks, [300, 300, 400, 400]))
        self.assertIsNone(pdfio.nearest_block(blocks, [0, 0]))


class ApplyAnnotationTest(unittest.TestCase):
    """M6: ``apply_annotation`` edits the block under the framed region."""

    def _state_and_tools(self):
        state = agent.WorkflowState(src_path="a.pdf", lang="English")
        state.src_doc = pdfio.DocumentText(
            pages=[[_blk("总资产", page=0, x0=0, y0=0, x1=50, y1=10),
                    _blk("总负债", page=0, x0=0, y0=20, x1=50, y1=30)]],
            blocks=["总资产", "总负债"], block_pages=[0, 0], title="s",
        )
        model = ModelConfig(id="m", name="m", type="openai",
                            endpoint="http://127.0.0.1:9/v1", model="mock")
        return state, agent.make_page_executors(state, model)

    def test_set_replaces_block_under_bbox(self):
        state, tools = self._state_and_tools()
        res = tools["apply_annotation"](0, [0, 20, 50, 30], text="Total liabilities")
        self.assertTrue(res["ok"])
        self.assertEqual("Total liabilities", state.out_doc[1]["text"])
        self.assertTrue(any(op.tool == "apply_annotation" and op.user_confirmed
                            for op in state.ops))

    def test_delete_removes_translation(self):
        state, tools = self._state_and_tools()
        state.out_doc = {0: {"text": "Total assets"}}
        res = tools["apply_annotation"](0, [0, 0, 50, 10], action="delete")
        self.assertTrue(res["ok"])
        self.assertNotIn(0, state.out_doc)

    def test_no_match_is_fail_closed(self):
        state, tools = self._state_and_tools()
        res = tools["apply_annotation"](0, [300, 300, 400, 400], text="x")
        self.assertFalse(res["ok"])


if __name__ == "__main__":
    unittest.main()
