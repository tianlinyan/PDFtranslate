"""Tests for the AI content policy (``translate_app/policy.py``).

The policy is the IR / direct path's answer to "which block keeps its source?": the
red lines (amounts, formulas, table cells) must be unreachable from the model's
reply, and every failure must leave the deterministic default in place.
"""

from __future__ import annotations

import unittest
from unittest import mock

from translate_app import policy, prompts, translator
from translate_app.pdfio import Block, DocumentText, PageStructure
from translate_app.settings import ModelConfig


def _doc(pages, structure=None):
    """A ``DocumentText`` whose flat lists are aligned with ``pages``."""
    flat = [b for pg in pages for b in pg]
    dt = DocumentText(pages=pages, blocks=[b.text for b in flat],
                      block_pages=[i for i, pg in enumerate(pages) for _ in pg])
    if structure is not None:
        dt.page_structure = structure
    return dt


def _figure_structure(page, indices, kind="figure"):
    return PageStructure(page=page, parser="mock",
                         elements=[{"kind": kind, "bbox": (0, 0, 1, 1),
                                    "block_indices": list(indices)}])


class CandidateTest(unittest.TestCase):
    def test_only_figures_and_scanned_blocks_are_ambiguous(self):
        body = Block("普通正文", 0, 0, 0, 100, 20)
        figure = Block("公司治理结构", 0, 0, 30, 100, 50)
        seal = Block("印章", 0, 0, 60, 100, 80, ocr=True)
        amount = Block("1,234.56", 0, 0, 90, 100, 110, ocr=True)
        cell = Block("合计", 0, 0, 120, 100, 140, ocr=True, in_table=True)
        dt = _doc([[body, figure, seal, amount, cell]],
                  structure=[_figure_structure(0, [1])])
        cands = policy.candidates(dt)
        self.assertEqual([(1, "figure", True), (2, "ocr", False)],
                         [(c.index, c.kind, c.default_keep) for c in cands])

    def test_a_formula_is_never_offered(self):
        # A formula is a red line, not a decision: the model may not even see it.
        formula = Block("E = mc^2", 0, 0, 0, 100, 20)
        dt = _doc([[formula]], structure=[_figure_structure(0, [0], kind="formula")])
        self.assertEqual([], policy.candidates(dt))

    def test_figure_text_on_a_text_layer_page_is_not_ambiguous(self):
        # ``in_image`` blocks (text OCR'd out of a raster figure on a text-layer
        # page) always translate, so offering them would invite the model to keep
        # content the product decision says must be translated.
        b = Block("Revenue by year", 0, 0, 0, 100, 20, in_image=True)
        dt = _doc([[b]], structure=[_figure_structure(0, [0])])
        self.assertEqual([], policy.candidates(dt))

    def test_an_already_kept_block_is_not_re_offered(self):
        b = Block("印章", 0, 0, 0, 100, 20, ocr=True, keep_original=True)
        self.assertEqual([], policy.candidates(_doc([[b]])))

    def test_caps_keep_the_prompt_bounded(self):
        blocks = [Block(f"扫描行{i}", 0, 0, i * 10, 100, i * 10 + 8, ocr=True)
                  for i in range(10)]
        dt = _doc([blocks[0:5], blocks[5:10]])
        cands = policy.candidates(dt, max_per_page=2, max_blocks=3)
        self.assertEqual([0, 1, 5], [c.index for c in cands])

    def test_page_numbers_are_one_based_for_display(self):
        b = Block("印章", 0, 0, 0, 100, 20, ocr=True)
        dt = _doc([[Block("x", 0, 0, 0, 10, 10)], [b]])
        entries = policy.policy_entries(policy.candidates(dt))
        self.assertEqual([(1, 2, "ocr", False, "印章")], entries)

    def test_an_empty_document_has_no_candidates(self):
        self.assertEqual([], policy.candidates(_doc([])))


class ParsePolicyJsonTest(unittest.TestCase):
    def test_reads_a_fenced_reply(self):
        text = ('好的：\n```json\n{"release": [7], "keep": [3, 4], '
                '"reason": "印章保留"}\n```')
        decision, reason = policy.parse_policy_json(text, allowed={3, 4, 7})
        self.assertEqual({3: policy.KEEP, 4: policy.KEEP, 7: policy.TRANSLATE}, decision)
        self.assertEqual("印章保留", reason)

    def test_an_index_outside_the_offered_set_is_dropped(self):
        # A hallucinated index must never mark a block the model never saw.
        decision, _ = policy.parse_policy_json('{"keep": [3, 999]}', allowed={3})
        self.assertEqual({3: policy.KEEP}, decision)

    def test_unusable_values_are_dropped(self):
        decision, _ = policy.parse_policy_json(
            '{"keep": [true, 1.5, "x", null, "12"], "release": "3"}', allowed={1, 12})
        self.assertEqual({12: policy.KEEP}, decision)

    def test_malformed_replies_are_no_decision(self):
        for text in ("", "没有 JSON", "{不是 json}", "[1, 2]", "null"):
            with self.subTest(text=text):
                self.assertEqual(({}, ""), policy.parse_policy_json(text, allowed={1}))


class ApplyPolicyTest(unittest.TestCase):
    def test_keep_marks_the_block_and_release_does_not(self):
        a = Block("正文", 0, 0, 0, 100, 20)
        b = Block("印章", 0, 0, 30, 100, 50, ocr=True)
        dt = _doc([[a, b]])
        applied = policy.apply_policy(dt, {0: policy.KEEP, 1: policy.TRANSLATE})
        self.assertTrue(a.keep_original, "keep must reach the exporter/audit channel")
        self.assertFalse(b.keep_original, "a release is an IR decision, not a block mark")
        self.assertEqual({"kept": 1, "released": [1], "skipped": 0}, applied)

    def test_decisions_outside_the_document_are_skipped(self):
        dt = _doc([[Block("正文", 0, 0, 0, 100, 20)]])
        logs: list[str] = []
        applied = policy.apply_policy(dt, {5: policy.KEEP}, log=logs.append)
        self.assertEqual({"kept": 0, "released": [], "skipped": 1}, applied)
        self.assertTrue(any("索引不在文档内" in m for m in logs), logs)

    def test_keeping_an_already_kept_block_is_not_double_counted(self):
        a = Block("印章", 0, 0, 0, 100, 20, ocr=True, keep_original=True)
        applied = policy.apply_policy(_doc([[a]]), {0: policy.KEEP})
        self.assertEqual(0, applied["kept"])


class _Msg:
    def __init__(self, content):
        self.content = content


class _Resp:
    def __init__(self, content):
        self.choices = [type("_C", (), {"message": _Msg(content)})()]


class _Client:
    def __init__(self, content='{"keep": [0]}', error=None):
        self.calls: list[dict] = []
        self._content = content
        self._error = error

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kw):
        self.calls.append(kw)
        if self._error is not None:
            raise self._error
        return _Resp(self._content)


class LlmPolicyFnTest(unittest.TestCase):
    def _model(self):
        return ModelConfig(id="m", name="m", type="openai",
                           endpoint="http://127.0.0.1:9/v1", model="mock")

    def _cands(self):
        dt = _doc([[Block("印章", 0, 0, 0, 100, 20, ocr=True)]])
        return policy.candidates(dt), dt

    def test_the_prompt_carries_the_blocks_and_the_requirement(self):
        cands, _dt = self._cands()
        client = _Client()
        fn = policy.make_llm_policy_fn(self._model(), client=client)
        decision, reason = fn(cands, lang="English", requirement="第5页图表翻译一下")
        self.assertEqual({0: policy.KEEP}, decision)
        self.assertEqual("", reason)
        message = client.calls[0]["messages"][0]["content"]
        # A deterministic judgement call, not creative writing: temperature 0.
        self.assertEqual(0.0, client.calls[0]["temperature"])
        self.assertIn("印章", message)
        self.assertIn("第5页图表翻译一下", message)
        self.assertIn("English", message)

    def test_the_reason_comes_back_for_the_log(self):
        cands, _dt = self._cands()
        client = _Client('{"keep": [0], "reason": "扫描印章"}')
        fn = policy.make_llm_policy_fn(self._model(), client=client)
        decision, reason = fn(cands, lang="English")
        self.assertEqual({0: policy.KEEP}, decision)
        self.assertEqual("扫描印章", reason)

    def test_a_failing_model_is_no_decision(self):
        cands, _dt = self._cands()
        logs: list[str] = []
        client = _Client(error=RuntimeError("boom"))
        fn = policy.make_llm_policy_fn(self._model(), client=client, log=logs.append)
        self.assertEqual(({}, ""), fn(cands, lang="English"))
        self.assertTrue(any("内容策略判定失败" in m for m in logs), logs)

    def test_no_candidates_makes_no_request(self):
        client = _Client()
        fn = policy.make_llm_policy_fn(self._model(), client=client)
        self.assertEqual(({}, ""), fn([], lang="English"))
        self.assertEqual([], client.calls)

    def test_without_a_usable_client_there_is_no_policy(self):
        model = self._model()
        with mock.patch.object(translator, "OpenAI",
                               mock.Mock(side_effect=RuntimeError("no client"))):
            self.assertIsNone(policy.make_llm_policy_fn(model))


class ContentPolicyPromptTest(unittest.TestCase):
    def test_the_red_lines_are_stated_and_absent_from_the_list(self):
        text = prompts.content_policy_task([(4, 2, "ocr", False, "印章")], "English")
        self.assertIn("[4] 第2页 类型=ocr 默认=翻译 | 文本: 印章", text)
        self.assertIn("数字/金额/公式块是流水线红线、永不翻译", text)
        # No requirement = nothing to release, and the prompt says so.
        self.assertIn("本轮用户要求：（无）", prompts.content_policy_task([], "English"))

    def test_long_block_text_is_truncated(self):
        text = prompts.content_policy_task([(0, 1, "ocr", False, "字" * 200)], "English")
        self.assertIn("…", text)
        self.assertNotIn("字" * 61, text)


if __name__ == "__main__":
    unittest.main()
