"""Tests for the M1 free-text intent slot-fill (``agent/intent.py``).

The reader is a *decision* wired in by the caller; when no client / a bad reply exists
it is fail-closed to ``""`` so the deterministic matcher decides.  These tests use a
mock OpenAI client and never hit the network.  Caches are unaffected (no cache here).
"""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from translate_app.agent import intent
from translate_app.agent.flow import DocumentSession


def _model():
    return SimpleNamespace(model="m", client_kwargs=lambda: {}, request_params=lambda: {})


def _client(content: str):
    c = MagicMock()
    c.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return c


class IntentParseTest(unittest.TestCase):
    def test_parse_flow_json_wraps_prose(self):
        self.assertEqual({"choice": "user"},
                         intent._parse_json_object('说明文字 {"choice": "user"} 尾巴'))

    def test_parse_json_bad_returns_empty(self):
        self.assertEqual({}, intent._parse_json_object("没有 JSON"))
        self.assertEqual({}, intent._parse_json_object("[1, 2]"))
        self.assertEqual({}, intent._parse_json_object(None))


class MakeLlmIntentFillTest(unittest.TestCase):
    def test_read_returns_chosen(self):
        read = intent.make_llm_intent_fill(_model(), client=_client('{"choice": "user"}'),
                                           log=lambda m: None)
        self.assertEqual("user", read("我自己来检查", ["user", "ai"]))

    def test_read_falls_back_to_verbatim(self):
        read = intent.make_llm_intent_fill(_model(), client=_client('{"other": 1} 有 continue'),
                                           log=lambda m: None)
        self.assertEqual("continue", read("先别导出", ["continue", "export"]))

    def test_read_fail_closed_on_bad_reply(self):
        read = intent.make_llm_intent_fill(_model(), client=_client("完全不是 JSON"),
                                           log=lambda m: None)
        self.assertEqual("", read("先别导出", ["continue", "export"]))

    def test_read_disallows_out_of_set(self):
        read = intent.make_llm_intent_fill(_model(), client=_client('{"choice": "translate"}'),
                                           log=lambda m: None)
        self.assertEqual("", read("先别导出", ["continue", "export"]))

    def test_no_client_returns_none(self):
        m = SimpleNamespace(model="m", client_kwargs=lambda: (_ for _ in ()).throw(RuntimeError))
        self.assertIsNone(intent.make_llm_intent_fill(m))


class ClassifyChoiceIntentTest(unittest.TestCase):
    def test_injected_llm_wins(self):
        stub = SimpleNamespace(intent_llm=lambda text, choices: "user")
        self.assertEqual("user", DocumentSession._classify_choice(stub, "我自己来检查", "review_mode"))

    def test_injected_llm_out_of_set_degrades_to_keyword(self):
        stub = SimpleNamespace(intent_llm=lambda text, choices: "translate")  # not allowed
        self.assertEqual("user", DocumentSession._classify_choice(stub, "我自己来检查", "review_mode"))

    def test_injected_llm_exception_degrades_to_keyword(self):
        def boom(text, choices):
            raise RuntimeError("model down")
        stub = SimpleNamespace(intent_llm=boom)
        self.assertEqual("continue", DocumentSession._classify_choice(stub, "先别导出", "export"))

    def test_no_llm_keeps_deterministic(self):
        stub = SimpleNamespace(intent_llm=None)
        self.assertEqual("user", DocumentSession._classify_choice(stub, "我自己来检查", "review_mode"))
        self.assertEqual("ai", DocumentSession._classify_choice(stub, "AI 来检查", "review_mode"))
        self.assertEqual("export", DocumentSession._classify_choice(stub, "导出吧", "export"))


if __name__ == "__main__":
    unittest.main()
