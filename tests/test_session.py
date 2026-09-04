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
    """One page each: normal / scan / chart / text-layer-table→normal / uncertain."""
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
            pdfio.PAGE_NORMAL,   # a non-scanned (text-layer) table is a normal page now
            pdfio.classify_page([_blk("a", in_table=True), _blk("b", in_table=True),
                                 _blk("c", in_table=True)]),
        )
        self.assertEqual(pdfio.PAGE_UNCERTAIN, pdfio.classify_page([_blk("标题"), _blk("扫", ocr=True)]))

    def test_get_doc_info_counts(self):
        info = pdfio.get_doc_info(_mixed_doc())
        self.assertEqual(5, info["pages"])
        self.assertEqual("样表", info["title"])
        self.assertEqual("zh", info["language"])
        # The text-layer table page is now a *normal* page, not a special table page.
        self.assertEqual(2, info["text_pages"])
        self.assertEqual(1, info["scan_pages"])
        self.assertEqual(1, info["chart_pages"])
        self.assertEqual(0, info["table_pages"])
        self.assertEqual(1, info["uncertain_pages"])
        self.assertEqual(3, info["special_pages"])   # scan+chart+uncertain (no table)
        self.assertEqual(["normal", "scan", "chart", "normal", "uncertain"], info["kinds"])


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
            if "复核" in str(task):
                return st   # the M4 review pass is separate; this test checks the translate phase
            calls.append(page)
            st.out_doc = st.out_doc or {}
            st.out_doc[page] = {"text": f"T{page}"}
            return st

        state, session = self._session(doc, fake_translate)
        session.run()

        # Both normal pages (index 0 and the text-layer table page 3) were translated
        # in the normal phase; the special pages (scan/chart/uncertain) were not.
        self.assertEqual([0, 3], calls)
        self.assertNotIn("T1", [str(v.get("text")) for v in (state.out_doc or {}).values()])
        # Phase reached DONE.
        self.assertEqual(PHASE_DONE, state.phase)
        # Doc info + triage populated by preprocess.
        self.assertIsNotNone(state.doc_info)
        self.assertEqual(5, state.doc_info.pages)
        self.assertEqual("normal", state.triage[0].kind)
        self.assertEqual("scan", state.triage[1].kind)
        self.assertEqual("chart", state.triage[2].kind)
        self.assertEqual("normal", state.triage[3].kind)   # non-scanned table → normal
        self.assertEqual("uncertain", state.triage[4].kind)
        # Only the special pages are flagged needs_user + a decision marker.
        for i in (1, 2, 4):
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
            if "复核" in str(task):
                return st   # skip the separate M4 review pass in this phase test
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
            if "复核" in str(task):
                return st   # skip the separate M4 review pass in this phase test
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

    def test_review_phase_ai_self_check_then_export(self):
        doc = _mixed_doc()
        review_pages: list[int] = []
        answers: list[tuple] = []
        audit_calls: dict[int, int] = {}

        def fake_translate(st, page, _model, *, task, **kw):
            if "复核" in str(task):
                review_pages.append(page)
                return st
            st.out_doc = st.out_doc or {}
            st.out_doc[page] = {"text": f"T{page}"}
            return st

        def answer_handler(question, options, target):
            answers.append(target)
            return {"value": "AI 自检" if target == "review_mode" else "导出", "target": target}

        # M4 review gate: the deterministic audit runs first; a page that is clean is
        # done WITHOUT the AI fix pass.  The fake reports an issue on the first audit
        # (so the agent runs to fix it) and clean afterwards (so the loop ends).
        def fake_audit(page, checks=None):
            audit_calls[page] = audit_calls.get(page, 0) + 1
            clean = audit_calls[page] > 1
            return {"page": page, "checks_requested": list(checks or []), "checks": {},
                    "issues": [] if clean else [{"check": "residual", "index": page,
                                                 "text": "x", "reason": "empty"}],
                    "clean": clean}

        state = agent.WorkflowState(src_path="a.pdf", lang="English")
        state.src_doc = doc
        session = DocumentSession(state, doc, model=object(), log=lambda m: None,
                                  translate_page=fake_translate, answer_handler=answer_handler,
                                  audit=fake_audit)
        session.run()
        # Mode asked; AI self-check re-read only the TRANSLATED pages (0, 3) and
        # skipped the special pages the user chose to keep/skip (1, 2, 4).
        self.assertEqual("ai", state.review_mode)
        self.assertEqual([0, 3], review_pages)
        # Each translated page was audited (found an issue) then re-audited (clean).
        self.assertEqual({0: 2, 3: 2}, audit_calls)
        self.assertIn("review_mode", answers)
        self.assertIn("export", answers)
        self.assertEqual(PHASE_DONE, state.phase)

    def test_review_phase_clean_page_is_skipped(self):
        # M4: a page already clean by the deterministic audit must NOT go to the AI
        # fix pass — the review gate saves the model budget on already-clean pages.
        doc = _mixed_doc()
        review_pages: list[int] = []
        answers: list[tuple] = []

        def fake_translate(st, page, _model, *, task, **kw):
            if "复核" in str(task):
                review_pages.append(page)
                return st
            st.out_doc = st.out_doc or {}
            st.out_doc[page] = {"text": f"T{page}"}
            return st

        def answer_handler(question, options, target):
            answers.append(target)
            return {"value": "AI 自检" if target == "review_mode" else "导出", "target": target}

        def fake_audit(page, checks=None):   # every page is already clean
            return {"page": page, "checks_requested": list(checks or []), "checks": {},
                    "issues": [], "clean": True}

        state = agent.WorkflowState(src_path="a.pdf", lang="English")
        state.src_doc = doc
        session = DocumentSession(state, doc, model=object(), log=lambda m: None,
                                  translate_page=fake_translate, answer_handler=answer_handler,
                                  audit=fake_audit)
        session.run()
        # No page had findings → the AI fix pass never ran for any translated page.
        self.assertEqual("ai", state.review_mode)
        self.assertEqual([], review_pages)
        self.assertEqual(PHASE_DONE, state.phase)

    def test_review_phase_audit_failure_is_not_clean(self):
        # A page whose deterministic audit itself *errored* must NOT be silently marked
        # clean/已复核 — that would pass a page whose review never actually ran.
        doc = _mixed_doc()
        answers: list[tuple] = []

        def fake_translate(st, page, _model, *, task, **kw):
            if "复核" in str(task):
                return st
            st.out_doc = st.out_doc or {}
            st.out_doc[page] = {"text": f"T{page}"}
            return st

        def answer_handler(question, options, target):
            answers.append(target)
            return {"value": "AI 自检" if target == "review_mode" else "导出", "target": target}

        def fake_audit(page, checks=None):
            raise RuntimeError("audit boom")

        state = agent.WorkflowState(src_path="a.pdf", lang="English")
        state.src_doc = doc
        session = DocumentSession(state, doc, model=object(), log=lambda m: None,
                                  translate_page=fake_translate, answer_handler=answer_handler,
                                  audit=fake_audit)
        session.run()
        self.assertEqual("ai", state.review_mode)
        for i in (0, 3):   # the translated pages
            self.assertTrue(
                any("自检未完成" in x or "自检失败" in x for x in state.page(i).issues),
                i)
            self.assertFalse(any(x == "已复核" for x in state.page(i).issues), i)

    def test_review_phase_user_mode_skips_ai_and_can_continue(self):
        doc = _mixed_doc()
        review_pages: list[int] = []
        answers: list[tuple] = []

        def fake_translate(st, page, _model, *, task, **kw):
            if "复核" in str(task):
                review_pages.append(page)
                return st
            st.out_doc = st.out_doc or {}
            st.out_doc[page] = {"text": f"T{page}"}
            return st

        def answer_handler(question, options, target):
            answers.append(target)
            return {"value": "我手动检查" if target == "review_mode" else "继续检查", "target": target}

        state = agent.WorkflowState(src_path="a.pdf", lang="English")
        state.src_doc = doc
        session = DocumentSession(state, doc, model=object(), log=lambda m: None,
                                  translate_page=fake_translate, answer_handler=answer_handler)
        session.run()
        # User mode: no AI self-check ran; the export confirm answered "继续检查".
        self.assertEqual("user", state.review_mode)
        self.assertEqual([], review_pages)
        self.assertEqual(PHASE_DONE, state.phase)

    def test_special_page_translate_failure_marks_needs_user(self):
        doc = _small_special_doc()

        def fake_translate(st, page, _model, *, task, **kw):
            if "复核" in str(task):
                return st   # the separate M4 review pass
            if page == 1:
                raise RuntimeError("boom")   # the special scan page fails to translate
            st.out_doc = st.out_doc or {}
            st.out_doc[page] = {"text": f"T{page}"}
            return st

        def answer_handler(question, options, target):
            return {"value": "OCR并翻译", "target": target}

        state = agent.WorkflowState(src_path="a.pdf", lang="English")
        state.src_doc = doc
        session = DocumentSession(state, doc, model=object(), log=lambda m: None,
                                  translate_page=fake_translate, answer_handler=answer_handler)
        session.run()
        # The failed special page is NOT silently marked done; it is flagged needs_user.
        self.assertEqual(agent.STATUS_NEEDS_USER, state.page(1).status)
        self.assertTrue(any("翻译失败" in i for i in state.page(1).issues))
        # The normal page 0 was still translated.
        self.assertEqual("T0", state.out_doc[0]["text"])
        self.assertEqual(PHASE_DONE, state.phase)


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
