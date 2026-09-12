"""Tests for the document-level translation plan (v0.6.6 M1).

``agent/plan.py`` is pure data + validation, so most of it is tested without any
session; the channel tests reuse the ``DocumentSession`` harness from
``test_session`` and assert the plan only ever writes the three channels that
already exist.
"""
import unittest
from types import SimpleNamespace

from translate_app import agent, pdfio
from translate_app.agent import plan as plan_mod
from translate_app.agent.flow import DocumentSession


def _model():
    return SimpleNamespace(
        model="m",
        client_kwargs=lambda: {"api_key": "x",
                               "base_url": "http://127.0.0.1:9/v1"},
        request_params=lambda: {})


def _client(text=None, error=None):
    """A minimal chat-completions stub: replies with ``text`` or raises ``error``."""
    def create(**_kw):
        if error is not None:
            raise error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))


def _page(*texts):
    return [pdfio.Block(t, 0, 10, 10 + 20 * i, 60, 25 + 20 * i, size=12)
            for i, t in enumerate(texts)]


def _doc(*pages):
    blocks = [b.text for pg in pages for b in pg]
    block_pages = [i for i, pg in enumerate(pages) for _ in pg]
    return pdfio.DocumentText(pages=[list(p) for p in pages], blocks=blocks,
                              block_pages=block_pages, title="t")


class ValidatePlanTest(unittest.TestCase):
    """Everything the model returns is bound to the document before it is used."""

    def test_parse_plan_json_survives_fences_and_garbage(self):
        self.assertEqual({"keep": [1]},
                         plan_mod.parse_plan_json("```json\n{\"keep\": [1]}\n```"))
        self.assertEqual({}, plan_mod.parse_plan_json("no json here"))
        self.assertEqual({}, plan_mod.parse_plan_json("[1, 2]"))

    def test_validate_drops_a_term_the_document_does_not_contain(self):
        plan = plan_mod.validate_plan(
            {"glossary": {"总资产": "Total assets", "编造的词": "Invented"}},
            n_blocks=5, source_text="总资产 营业收入")
        self.assertEqual({"总资产": "Total assets"}, plan.glossary)
        self.assertTrue(any("原文中不存在" in d for d in plan.dropped))

    def test_validate_drops_out_of_range_and_malformed_keeps(self):
        plan = plan_mod.validate_plan({"keep": [0, 7, "x", -1, 2]}, n_blocks=3,
                                      source_text="")
        self.assertEqual({0, 2}, plan.keep)
        self.assertEqual(3, len(plan.dropped), plan.dropped)

    def test_validate_clamps_an_over_long_style(self):
        plan = plan_mod.validate_plan({"style": "公" * 400}, n_blocks=1, source_text="")
        self.assertEqual(plan_mod._PLAN_MAX_STYLE, len(plan.style))
        self.assertTrue(any("style" in d for d in plan.dropped))

    def test_an_empty_reply_is_no_plan(self):
        plan = plan_mod.validate_plan({}, n_blocks=3, source_text="x")
        self.assertFalse(plan.has_content())
        self.assertEqual("", plan.source)
        self.assertIn("无有效内容", plan_mod.plan_summary(plan))

    def test_document_summary_is_bounded_and_lists_special_pages_first(self):
        doc = _doc(_page("正文一"), _page("正文二"), _page("扫描内容"))
        info = SimpleNamespace(language="zh", text_pages=2, scan_pages=1,
                               chart_pages=0, uncertain_pages=0,
                               kinds=["normal", "normal", "scan"])
        text = plan_mod.document_summary(doc, info, {}, terms=["总资产"],
                                         requirements=["保留英文单位"])
        self.assertLessEqual(len(text), plan_mod._PLAN_INPUT_BUDGET)
        self.assertIn("候选术语=总资产", text)
        self.assertIn("用户要求=保留英文单位", text)
        self.assertLess(text.index("第3页[scan]"), text.index("第1页[normal]"),
                        "特殊页排在前面（它们的处理最可能逐页不同）")


class MakeLlmPlanTest(unittest.TestCase):
    """The request layer only owns the call and the fail-open; no validation here."""

    def test_a_failing_request_yields_no_plan(self):
        logs: list[str] = []
        fn = plan_mod.make_llm_plan(_model(),
                                    client=_client(error=RuntimeError("boom")),
                                    log=logs.append)
        self.assertEqual({}, fn("概况", lang="English"))
        self.assertTrue(any("请求失败" in m for m in logs), logs)

    def test_a_reply_is_parsed_into_json(self):
        fn = plan_mod.make_llm_plan(
            _model(),
            client=_client(text="说明\n{\"glossary\": {\"总资产\": \"Total assets\"}}\n完"),
            log=lambda _m: None)
        self.assertEqual({"glossary": {"总资产": "Total assets"}},
                         fn("概况", lang="English"))

    def test_no_client_means_no_plan_callback(self):
        class _NoKwargs:
            pass

        self.assertIsNone(plan_mod.make_llm_plan(_NoKwargs(), client=None, log=None))


class PlanChannelTest(unittest.TestCase):
    """The plan is consumed only through the three channels that already exist."""

    def _session(self, doc, *, plan=True, plan_llm=None, logs=None, scope=None):
        state = agent.WorkflowState(src_path="a.pdf", lang="English")
        state.src_doc = doc
        session = DocumentSession(
            state, doc, model=object(), log=(logs if logs is not None else (lambda _m: None)),
            translate_page=lambda st, page, model, **kw: st,
            plan=plan, plan_llm=plan_llm, scope=scope)
        return state, session

    def test_a_plan_fills_the_three_existing_channels(self):
        doc = _doc(_page("总资产", "其他内容"), _page("营业收入"), _page("注释"))
        seen: list[str] = []

        def fake(summary, *, terms=(), requirements=(), lang=""):
            seen.append(summary)
            return {"glossary": {"总资产": "Total assets"},
                    "style": "公文，金额用万元", "keep": [1], "notes": "说明"}

        logs: list[str] = []
        state, session = self._session(doc, plan_llm=fake, logs=logs.append)
        session._preprocess()
        self.assertTrue(seen and "文档概况" not in seen[0])   # digest, not the document
        self.assertEqual({"总资产": "Total assets"},
                         state.user_decisions["terminology"])
        self.assertTrue(any("公文" in r for r in state.requirements), state.requirements)
        self.assertTrue(doc.pages[0][1].keep_original)
        self.assertFalse(doc.pages[0][0].keep_original)
        self.assertIsNotNone(state.plan)
        self.assertEqual("llm", state.plan.source)
        self.assertTrue(any("文档级方案" in m for m in logs), logs)

    def test_a_plan_can_only_add_keeps(self):
        doc = _doc(_page("总资产", "其他内容"), _page("营业收入"), _page("注释"))
        doc.pages[0][0].keep_original = True          # e.g. the content policy kept it

        def fake(summary, *, terms=(), requirements=(), lang=""):
            return {"keep": [1], "release": [0]}      # "release" is not a channel we honour

        state, session = self._session(doc, plan_llm=fake)
        session._preprocess()
        self.assertTrue(doc.pages[0][0].keep_original, "保留只能增加，不能取消")
        self.assertTrue(doc.pages[0][1].keep_original)

    def test_no_plan_is_built_when_disabled(self):
        doc = _doc(_page("总资产"), _page("营业收入"), _page("注释"))
        calls: list[int] = []

        def fake(*_a, **_k):
            calls.append(1)
            return {}

        logs: list[str] = []
        state, session = self._session(doc, plan=False, plan_llm=fake, logs=logs.append)
        session._preprocess()
        self.assertEqual([], calls)
        self.assertIsNone(state.plan)
        self.assertFalse(any("文档级方案" in m for m in logs), logs)


if __name__ == "__main__":
    unittest.main()
