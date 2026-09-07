"""Tests for the persistent AI chat (interaction-model conversation)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PyQt6.QtCore import Qt

from translate_app import chat
from translate_app.doc_context import DocContext
from translate_app.settings import ModelConfig

from tests._helpers import build_sample_pdf


def _model(temp=None, effort=None, max_tokens=None, vision=True) -> ModelConfig:
    d = {
        "id": "m", "name": "m", "type": "openai",
        "endpoint": "http://127.0.0.1:9/v1/chat/completions", "model": "mod",
        "vision": vision,
    }
    if temp is not None:
        d["interaction_temperature"] = temp
    if effort is not None:
        d["interaction_reasoning_effort"] = effort
    if max_tokens is not None:
        d["max_tokens"] = max_tokens
    return ModelConfig.from_dict(d)


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeResp:
    def __init__(self, content):
        # ``choices[0].message`` is a namespace-like object; reuse a real class.
        self.choices = [type("_C", (), {"message": _FakeMsg(content)})()]


class _FakeCompletions:
    def __init__(self, reply, seen=None):
        self._reply = reply
        self.seen = seen if seen is not None else []

    def create(self, **kwargs):
        self.seen.append(kwargs)
        return _FakeResp(self._reply)


class _FakeClient:
    def __init__(self, reply, seen=None):
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(reply, seen)})()


class _FakeToolCall:
    def __init__(self, name, args, call_id="c1"):
        self.id = call_id
        self.type = "function"
        self.function = type("_F", (), {"name": name, "arguments": json.dumps(args)})()


class _FakeToolMsg:
    def __init__(self, tool_calls, content=""):
        self.tool_calls = tool_calls
        self.content = content


class _FakeToolResp:
    def __init__(self, tool_calls, content=""):
        self.choices = [type("_C", (), {"message": _FakeToolMsg(tool_calls, content)})()]


class _FakeToolCompletions:
    """Yields a scripted sequence of responses (last one repeats)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.seen: list = []

    def create(self, **kwargs):
        self.seen.append(kwargs)
        i = min(len(self._responses) - 1, len(self.seen) - 1)
        return self._responses[i]


class _FakeToolClient:
    def __init__(self, responses):
        self.chat = type("_Chat", (), {"completions": _FakeToolCompletions(responses)})()


class ChatSessionTest(unittest.TestCase):
    def test_reply_records_history_and_uses_interaction_params(self):
        seen: list = []
        client = _FakeClient("你好！我是 PDF Translate 助手。", seen)
        with mock.patch.object(chat, "OpenAI", lambda **_k: client):
            session = chat.ChatSession(_model())
        reply = session.reply("你好")
        self.assertEqual("你好！我是 PDF Translate 助手。", reply)
        # The conversation history carries both turns, so the model has context.
        self.assertEqual(
            [{"role": "user", "content": "你好"}, {"role": "assistant", "content": reply}],
            session.history,
        )
        # Interaction params: temperature 0.6 + reasoning_effort=medium via extra_body.
        kw = seen[0]
        self.assertEqual(0.6, kw["temperature"])
        self.assertEqual({"reasoning_effort": "medium"}, kw["extra_body"])
        self.assertEqual("mod", kw["model"])

    def test_reply_attaches_image_for_vision_model(self):
        # A preview image buffered for the chat is sent as a vision input (image_url)
        # alongside the user's text, for a multimodal model.
        seen: list = []
        with mock.patch.object(chat, "OpenAI", lambda **_k: _FakeClient("ok", seen)):
            session = chat.ChatSession(_model())   # vision=True
        session.reply("看这张图", image=b"\x89PNG\x0d\x0a\x1a\x0a")
        user = next(m for m in seen[0]["messages"] if m["role"] == "user")
        content = user["content"]
        self.assertIsInstance(content, list)
        self.assertEqual("看这张图", content[0]["text"])
        self.assertIn("image_url", content[1])
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_reply_ignores_image_for_non_vision_model(self):
        # A non-vision model must keep the plain message string (no image) — sending
        # an image_url would make the request fail.
        seen: list = []
        with mock.patch.object(chat, "OpenAI", lambda **_k: _FakeClient("ok", seen)):
            session = chat.ChatSession(_model(vision=False))
        session.reply("hi", image=b"\x89PNG")
        user = next(m for m in seen[0]["messages"] if m["role"] == "user")
        self.assertEqual("hi", user["content"])

    def test_reply_honours_configured_interaction_overrides(self):
        seen: list = []
        with mock.patch.object(chat, "OpenAI", lambda **_k: _FakeClient("ok", seen)):
            session = chat.ChatSession(_model(temp=0.9, effort="high", max_tokens=2048))
        session.reply("hi")
        kw = seen[0]
        self.assertEqual(0.9, kw["temperature"])
        self.assertEqual({"reasoning_effort": "high"}, kw["extra_body"])
        self.assertEqual(2048, kw["max_tokens"])

    def test_reply_appends_history_across_turns(self):
        seen: list = []
        with mock.patch.object(chat, "OpenAI", lambda **_k: _FakeClient("re", seen)):
            session = chat.ChatSession(_model())
        session.reply("第一句")
        session.reply("第二句")
        roles = [h["role"] for h in session.history]
        self.assertEqual(["user", "assistant", "user", "assistant"], roles)

    def test_record_exchange_notes_flow_qa_into_history(self):
        # (b) A flow-time agent question + the user's answer is recorded into the
        # console's history, so the console and the flow share one coherent thread.
        with mock.patch.object(chat, "OpenAI", lambda **_k: _FakeClient("re", [])):
            session = chat.ChatSession(_model())
        session.record_exchange("特殊页 chart 怎么处理？", "保留原文", "page:1")
        self.assertEqual(["assistant", "user"], [h["role"] for h in session.history])
        self.assertIn("特殊页 chart", session.history[0]["content"])
        self.assertIn("page:1", session.history[0]["content"])
        self.assertEqual("保留原文", session.history[1]["content"])
        # An empty answer (e.g. a cancelled ask) is skipped, not recorded.
        session.record_exchange("q", "", "page:2")
        self.assertEqual(2, len(session.history))

    def test_reply_runs_tool_loop_and_feeds_results_back(self):
        # The model calls two tools, then returns a plain reply.  Each tool result
        # must be fed back as a ``tool`` role message and the loop must stop on the
        # final content.
        executed: list = []

        def executor(name, args):
            executed.append((name, args))
            return {"ok": True, "value": f"res-{name}"}

        responses = [
            _FakeToolResp([_FakeToolCall("get_doc_info", {})], ""),
            _FakeToolResp([_FakeToolCall("set_block_text", {"index": 5, "text": "hi"})], ""),
            _FakeToolResp(None, "完成"),
        ]
        client = _FakeToolClient(responses)
        with mock.patch.object(chat, "OpenAI", lambda **_k: client):
            session = chat.ChatSession(_model())
        reply = session.reply(
            "把第五块改成 hi",
            tools=[{"type": "function", "function": {"name": "get_doc_info", "parameters": {}}}],
            executor=executor,
        )
        self.assertEqual("完成", reply)
        self.assertEqual(
            [("get_doc_info", {}), ("set_block_text", {"index": 5, "text": "hi"})],
            executed,
        )
        roles = [h["role"] for h in session.history]
        self.assertIn("tool", roles)
        # The tool schemas are advertised on the first model call.
        first = client.chat.completions.seen[0]
        self.assertEqual("auto", first["tool_choice"])
        self.assertTrue(first["tools"])

    def test_reply_reinjects_tool_image_for_vision_model(self):
        # A tool result carrying ``image`` is stripped from the tool text message and
        # re-injected as an ``image_url`` user message (a fresh visual observation).
        def executor(name, args):
            return {"ok": True, "page": 1, "image": b"\x89PNG\x0d\x0a\x1a\x0a"}

        responses = [
            _FakeToolResp([_FakeToolCall("render_page", {"page": 0})], ""),
            _FakeToolResp(None, "我看完了"),
        ]
        client = _FakeToolClient(responses)
        with mock.patch.object(chat, "OpenAI", lambda **_k: client):
            session = chat.ChatSession(_model(vision=True))
        reply = session.reply(
            "第1页翻得怎么样",
            tools=[{"type": "function", "function": {"name": "render_page", "parameters": {}}}],
            executor=executor,
        )
        self.assertEqual("我看完了", reply)
        # The tool text message has the image stripped (no base64/blob noise).
        tool_msg = [h for h in session.history if h.get("role") == "tool"][0]
        self.assertNotIn("image", tool_msg["content"])
        # A fresh user observation carries the image as image_url.
        user_msgs = [h for h in session.history if h.get("role") == "user"]
        obs = user_msgs[-1]["content"] if user_msgs else None
        self.assertIsInstance(obs, list)
        self.assertIn("image_url", [c.get("type") for c in obs if isinstance(c, dict)])
        url = next(c["image_url"]["url"] for c in obs
                   if isinstance(c, dict) and c.get("type") == "image_url")
        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_reply_rescues_empty_model_response(self):
        # Model returns empty text with no tool call → nudge once; still empty → the
        # clear fallback placeholder (the sidebar must never show a blank bubble).
        responses = [
            _FakeToolResp(None, ""),   # empty, no tool call → rescue nudge
            _FakeToolResp(None, ""),   # still empty → fallback
        ]
        client = _FakeToolClient(responses)
        with mock.patch.object(chat, "OpenAI", lambda **_k: client):
            session = chat.ChatSession(_model())
        reply = session.reply("第二页有问题，修改一下")
        self.assertEqual(chat._EMPTY_REPLY, reply)
        nudges = [h for h in session.history if h.get("role") == "user"
                  and isinstance(h.get("content"), str)
                  and h["content"].startswith("（你刚才没有返回任何文字")]
        self.assertTrue(nudges)

    def test_reply_recovers_after_empty_then_answers(self):
        # A single empty reply is rescued; the next non-empty reply is returned.
        responses = [
            _FakeToolResp(None, ""),
            _FakeToolResp(None, "已检查完第2页，第3块数字有误，已重译并通过。"),
        ]
        client = _FakeToolClient(responses)
        with mock.patch.object(chat, "OpenAI", lambda **_k: client):
            session = chat.ChatSession(_model())
        reply = session.reply("第二页有问题，修改一下")
        self.assertEqual("已检查完第2页，第3块数字有误，已重译并通过。", reply)

    def test_window_history_caps_and_starts_at_user(self):
        # The window keeps at most ``_CHAT_HISTORY_CAP`` messages and always starts at
        # a ``user`` turn, so the role sequence a small-context model sees stays valid.
        s = chat.ChatSession(_model())
        s.history = [
            {"role": ("user" if i % 3 == 0 else ("assistant" if i % 3 == 1 else "tool")),
             "content": f"m{i}"}
            for i in range(40)
        ]
        w = s._window_history()
        self.assertLessEqual(len(w), chat._CHAT_HISTORY_CAP)
        self.assertEqual("user", w[0]["role"])
        # A short history passes through untouched.
        self.assertEqual(3, len(s._window_history()[:3]))

    def test_call_sends_bounded_window(self):
        seen: list = []
        client = _FakeClient("ok", seen)
        with mock.patch.object(chat, "OpenAI", lambda **_k: client):
            session = chat.ChatSession(_model())
        # Pre-fill history far beyond the cap, then send a fresh message.
        session.history = [{"role": "user", "content": f"old{i}"} for i in range(60)]
        session.reply("新消息")
        msgs = seen[0]["messages"]           # [system, ...windowed...]
        self.assertLessEqual(len(msgs), 1 + chat._CHAT_HISTORY_CAP)

    def test_downscale_png_shrinks_large_image(self):
        from io import BytesIO

        from PIL import Image

        def _png(w, h):
            im = Image.new("RGB", (w, h))
            buf = BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()

        small = chat._downscale_png(_png(2000, 1000), 1024)
        self.assertLessEqual(max(Image.open(BytesIO(small)).size), 1024)
        # A small image is returned unchanged.
        tiny = _png(100, 50)
        self.assertEqual(chat._downscale_png(tiny, 1024), tiny)
        # A corrupt payload degrades to the original bytes (never crashes).
        self.assertEqual(chat._downscale_png(b"not-a-png", 1024), b"not-a-png")

    def test_reply_with_tools_but_no_executor_returns_error_result(self):
        # A tool call with no executor never crashes; the result says so and the
        # loop passes it back for the model to see.
        responses = [
            _FakeToolResp([_FakeToolCall("get_doc_info", {})], ""),
            _FakeToolResp(None, "ok"),
        ]
        client = _FakeToolClient(responses)
        with mock.patch.object(chat, "OpenAI", lambda **_k: client):
            session = chat.ChatSession(_model())
        reply = session.reply("hi", tools=[{"type": "function", "function": {"name": "x"}}])
        self.assertEqual("ok", reply)
        tool_msgs = [h for h in session.history if h.get("role") == "tool"]
        self.assertIn("未执行工具", tool_msgs[0]["content"])


class ChatWorkerTest(unittest.TestCase):
    def test_ask_behind_single_session(self):
        # Two asks with the same model reuse one session (history preserved); the
        # worker emits ``reply_ready`` for each.
        seen: list = []
        with mock.patch.object(chat, "OpenAI", lambda **_k: _FakeClient("re", seen)):
            worker = chat.ChatWorker()
            replies: list[str] = []
            worker.reply_ready.connect(replies.append)
            worker.ask("你好", _model())
            worker.ask("继续", _model())
        self.assertEqual(["re", "re"], replies)
        self.assertIsNotNone(worker._session)

    def test_ask_with_no_model_emits_error(self):
        worker = chat.ChatWorker()
        errors: list[str] = []
        worker.error.connect(errors.append)
        worker.ask("你好", None)
        self.assertEqual(1, len(errors))
        self.assertIn("没有可用的 AI 模型", errors[0])

    def test_cancel_aborts_in_flight_reply(self):
        # The sidebar "取消" aborts an in-flight reply (closing the client raises on
        # the blocked create) and surfaces ``cancelled`` instead of an error.
        import threading as _th

        closed = _th.Event()

        class _BlockingClient:
            def close(self):
                closed.set()

            @property
            def chat(self):
                return self

            @property
            def completions(self):
                return self

            def create(self, **kwargs):
                closed.wait(5)
                raise RuntimeError("client closed (aborted)")

        seen: list = []
        worker = chat.ChatWorker()
        # DirectConnection: the emit happens on the ''ask'' thread and the test does
        # not run a Qt event loop, so force a synchronous (direct) delivery.
        worker.cancelled.connect(seen.append, Qt.ConnectionType.DirectConnection)
        with mock.patch.object(chat, "OpenAI", lambda **_k: _BlockingClient()):
            t = _th.Thread(target=lambda: worker.ask("你好", _model()), daemon=True)
            t.start()
            _th.Event().wait(0.2)      # let the ask reach the blocking create
            worker.cancel_current()
            t.join(timeout=3)
        self.assertEqual(["已取消"], seen)

    def test_ask_with_document_runs_tools(self):
        # A loaded document hands the chat model its tools; a tool call executes
        # against the real (unit-test) PDF and the result is fed back.
        tmp = Path(tempfile.mkdtemp())
        src = build_sample_pdf(tmp / "cd.pdf", pages=1)
        ctx = DocContext()
        ctx.set_source(str(src))
        responses = [
            _FakeToolResp([_FakeToolCall("get_doc_info", {})], ""),
            _FakeToolResp(None, "文档共 1 页。"),
        ]
        client = _FakeToolClient(responses)
        with mock.patch.object(chat, "OpenAI", lambda **_k: client):
            worker = chat.ChatWorker(ctx=ctx)
            replies: list[str] = []
            worker.reply_ready.connect(replies.append)
            worker.ask("有几页？", _model())
        self.assertEqual(["文档共 1 页。"], replies)
        # The get_doc_info tool actually ran (extracted and read the PDF).
        self.assertIn("tool", [h["role"] for h in worker._session.history])


if __name__ == "__main__":
    unittest.main()
