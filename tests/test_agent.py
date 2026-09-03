"""Tests for the v0.3.0 agent foundation (state skeleton + tool registry)."""
import json
import unittest
from unittest import mock

from translate_app import agent
from translate_app import pdfio
from translate_app import translator
from translate_app.settings import ModelConfig


class WorkflowStateTest(unittest.TestCase):
    def test_defaults(self):
        s = agent.WorkflowState(src_path="a.pdf", lang="English")
        self.assertEqual("a.pdf", s.src_path)
        self.assertEqual("English", s.lang)
        self.assertEqual([], s.pages)
        self.assertEqual([], s.ops)
        self.assertEqual([], s.todo)
        self.assertEqual([], s.requirements)
        self.assertEqual({}, s.user_decisions)
        self.assertIsNone(s.pending_question)
        self.assertEqual(0, s.budget.used_steps)

    def test_page_created_on_access(self):
        s = agent.WorkflowState("a.pdf", "English")
        p = s.page(2)
        self.assertEqual(2, p.index)
        self.assertEqual(agent.STATUS_PENDING, p.status)
        self.assertEqual(3, len(s.pages))   # pages 0, 1, 2 created on demand

    def test_record_op_advances_budget_and_appends(self):
        s = agent.WorkflowState("a.pdf", "English")
        op = s.record_op("verify_number", args={"page": 0, "index": 4},
                         target="page:0 block:4", reason="check a figure")
        self.assertEqual(1, s.budget.used_steps)
        self.assertEqual("verify_number", op.tool)
        self.assertEqual("page:0 block:4", op.target)
        self.assertEqual(1, len(s.ops))
        self.assertTrue(s.budget.remaining_steps() < s.budget.max_steps)

    def test_ask_answer_roundtrip(self):
        s = agent.WorkflowState("a.pdf", "English")
        s.ask("要擦除签字区？", ["确认", "跳过"], target="erase:sig")
        self.assertIsNotNone(s.pending_question)
        self.assertEqual("要擦除签字区？", s.pending_question["question"])
        self.assertEqual(["确认", "跳过"], s.pending_question["options"])
        s.answer("跳过")
        self.assertEqual("跳过", s.user_decisions["erase:sig"])
        self.assertIsNone(s.pending_question)

    def test_budget_exhausted(self):
        b = agent.Budget(max_steps=3)
        b.used_steps = 2
        self.assertFalse(b.exhausted())
        self.assertEqual(1, b.remaining_steps())
        b.used_steps = 3
        self.assertTrue(b.exhausted())

    def test_invalid_status_rejected(self):
        with self.assertRaises(ValueError):
            agent.Goal(kind="t", status="bogus")
        with self.assertRaises(ValueError):
            agent.PageState(index=0, status="nope")


class AgentToolsTest(unittest.TestCase):
    def test_openai_tools_wellformed(self):
        tools = agent.agent_openai_tools()
        self.assertGreater(len(tools), 0)
        for t in tools:
            self.assertEqual("function", t["type"])
            fn = t["function"]
            self.assertIn("name", fn)
            self.assertIn("description", fn)
            params = fn["parameters"]
            self.assertEqual("object", params["type"])
            self.assertIsInstance(params.get("properties"), dict)
            self.assertIsInstance(params.get("required"), list)

    def test_existing_judgment_tools_registered(self):
        names = [t.name for t in agent.AGENT_TOOLS]
        for n in ("classify_block", "detect_table_merge", "verify_number", "retranslate_block"):
            self.assertIn(n, names)
        vn = agent.by_name("verify_number")
        self.assertIsNotNone(vn)
        self.assertIn("is_correct", vn.parameters.get("properties", {}))

    def test_source_read_tools_are_readonly(self):
        # The original is protected: read/observe tools only read it (target source).
        for n in ("read_page", "render_region", "get_layout",
                  "classify_block", "detect_table_merge", "verify_number"):
            t = agent.by_name(n)
            self.assertIsNotNone(t, n)
            self.assertEqual("source", t.target, n)

    def test_write_tools_target_output_and_are_free(self):
        # The translation is fully editable (no confirmation); the original is
        # never a write target.
        write = ("set_text", "rewrite_block", "delete_block", "create_block", "move_block",
                 "cover_region", "erase_text_layer", "drop_element",
                 "delete_page", "create_page", "move_page")
        for n in write:
            t = agent.by_name(n)
            self.assertIsNotNone(t, n)
            self.assertEqual("output", t.target, n)
            self.assertFalse(t.destructive, n)   # translation side: free, no confirm

    def test_no_tool_is_destructive_to_source(self):
        # The unbreakable rule "原文在任何情况下不被删除/篡改": no exposed tool may be
        # marked destructive (i.e. mutate the immutable source).
        for t in agent.AGENT_TOOLS:
            self.assertFalse(t.destructive, t.name)

    def test_page_and_block_tools_registered(self):
        names = [t.name for t in agent.AGENT_TOOLS]
        for n in ("delete_page", "create_page", "move_page",
                  "delete_block", "create_block", "move_block"):
            self.assertIn(n, names)
        self.assertIn("page", agent.by_name("delete_page").parameters["required"])
        self.assertIn("at", agent.by_name("create_page").parameters["required"])
        self.assertEqual("output", agent.by_name("delete_page").target)

    def test_preview_tool_is_interactive_and_never_destructive(self):
        # A user "让我看看译文第 3 页" → the controller calls ``preview_page``; it
        # only pops a (non-blocking) window and must never be destructive.
        t = agent.by_name("preview_page")
        self.assertIsNotNone(t)
        self.assertFalse(t.destructive)
        self.assertEqual("ui", t.category)
        schema = t.parameters
        self.assertIn("page", schema["required"])
        self.assertIn("what", schema["properties"])

    def test_tool_filtering_by_name(self):
        names = ["read_page", "verify_number"]
        tools = agent.agent_openai_tools(names=names)
        self.assertEqual(2, len(tools))
        self.assertEqual({"read_page", "verify_number"}, {t["function"]["name"] for t in tools})

    def test_make_agent_tools_none_for_non_vision(self):
        model = ModelConfig.from_dict(dict(
            id="x", name="x", type="llama-server", endpoint="http://127.0.0.1:9", model="m"))
        self.assertEqual({}, agent.make_agent_tools(model))

    def test_make_agent_tools_binds_vision_tools(self):
        # Patch the OpenAI client so no network / real client is built at bind time.
        class _FakeOpenAI:
            def __init__(self, **_kwargs):
                pass

        model = ModelConfig.from_dict(dict(
            id="v", name="v", type="llama-server",
            endpoint="http://127.0.0.1:9/v1/chat/completions", model="qwen", vision=True))
        with mock.patch.object(translator, "OpenAI", _FakeOpenAI):
            tools = agent.make_agent_tools(model)
        for n in ("classify_block", "detect_table_merge", "verify_number", "retranslate_block"):
            self.assertIn(n, tools)
            self.assertTrue(callable(tools[n]), n)


def _scripted(decisions):
    """A fake ``decide`` that yields a fixed sequence of ``Decision``s, then 'done'."""
    it = iter(decisions)

    def decide(*_args):
        try:
            return next(it)
        except StopIteration:
            return agent.Decision(action="done")

    return decide


class FlowAgentTest(unittest.TestCase):
    """The orchestration skeleton drives a page through scripted tool_calls."""

    def _state(self):
        s = agent.WorkflowState(src_path="a.pdf", lang="English")
        s.todo.append(agent.Goal(kind="translate", page=0))
        s.page(0)
        s.requirements.append("保留报表编号")
        return s

    def _tools(self, calls):
        def read_page(page):
            calls.append(("read_page", page))
            return {"blocks": [{"index": 0, "text": "总资产"}]}

        def classify_block(index, text):
            calls.append(("classify_block", index))
            return {"action": "translate", "confidence": 0.9}

        def verify_number(index, value):
            calls.append(("verify_number", index))
            return {"is_correct": True}

        return {"read_page": read_page,
                "classify_block": classify_block,
                "verify_number": verify_number}

    def test_scripted_calls_run_in_order_and_mutate_state(self):
        s = self._state()
        calls: list = []
        decide = _scripted([
            agent.Decision(action="call", tool="read_page", arguments={"page": 0}),
            agent.Decision(action="call", tool="classify_block",
                           arguments={"index": 0, "text": "总资产"}),
            agent.Decision(action="call", tool="verify_number",
                           arguments={"index": 0, "value": "32,613,779.11"}),
            agent.Decision(action="done", summary="page done"),
        ])
        agent.run_agent_run(s, self._tools(calls), decide)
        # Tool calls in the scripted order.
        self.assertEqual([("read_page", 0), ("classify_block", 0), ("verify_number", 0)], calls)
        # Provenance recorded, budget advanced, last action is done (no pending question).
        self.assertEqual(["read_page", "classify_block", "verify_number"],
                         [op.tool for op in s.ops])
        self.assertEqual(3, s.budget.used_steps)
        self.assertEqual({"page": 0}, s.ops[0].args)
        self.assertIsNone(s.pending_question)

    def test_ask_pauses_loop_and_records_question(self):
        s = self._state()
        decide = _scripted([
            agent.Decision(action="ask", question="要擦除签字区？",
                           options=["确认", "跳过"], target="erase:sig"),
        ])
        agent.run_agent_run(s, self._tools([]), decide)
        self.assertEqual("要擦除签字区？", s.pending_question["question"])
        self.assertEqual(["确认", "跳过"], s.pending_question["options"])
        self.assertEqual("erase:sig", s.pending_question["target"])
        self.assertEqual(0, s.budget.used_steps)   # nothing ran yet

    def test_answer_handler_resumes_after_ask(self):
        # With an answer handler, the agent's question is surfaced and the answer is
        # stored; the loop *resumes* instead of pausing (no pending_question left).
        s = self._state()
        decide = _scripted([
            agent.Decision(action="ask", question="保留该块？",
                           options=["keep", "translate"], target="t"),
            agent.Decision(action="done"),
        ])
        answers: list = []

        def answer_handler(question, options, target):
            answers.append((question, options, target))
            return "keep"

        agent.run_agent_run(s, self._tools([]), decide, answer_handler=answer_handler)
        self.assertEqual([("保留该块？", ["keep", "translate"], "t")], answers)
        self.assertEqual("keep", s.user_decisions["t"])
        self.assertIsNone(s.pending_question)   # answered, not left pending

    def test_unknown_tool_is_fail_closed_not_crash(self):
        s = self._state()
        decide = _scripted([
            agent.Decision(action="call", tool="nope", arguments={}),
            agent.Decision(action="done"),
        ])
        agent.run_agent_run(s, self._tools([]), decide)
        self.assertEqual(1, len(s.ops))
        self.assertEqual("nope", s.ops[0].tool)
        self.assertTrue(any("unknown tool: nope" in m for m in s.log), s.log)

    def test_budget_exhausted_stops_loop(self):
        s = self._state()

        def always_call(*_args):
            return agent.Decision(action="call", tool="read_page", arguments={"page": 0})

        agent.run_agent_run(s, self._tools([]), always_call, max_steps=2)
        self.assertEqual(2, s.budget.used_steps)
        self.assertTrue(s.budget.exhausted())

    def test_run_agent_run_returns_state_and_caps_rounds(self):
        s = self._state()
        decide = _scripted([
            agent.Decision(action="call", tool="read_page", arguments={"page": 0}),
            agent.Decision(action="call", tool="read_page", arguments={"page": 0}),
            agent.Decision(action="call", tool="read_page", arguments={"page": 0}),
        ])
        out = agent.run_agent_run(s, self._tools([]), decide, max_rounds=2)
        self.assertIs(out, s)
        self.assertEqual(2, s.budget.used_steps)   # capped at max_rounds


def _vision_model() -> ModelConfig:
    return ModelConfig.from_dict(dict(
        id="v", name="v", type="llama-server",
        endpoint="http://127.0.0.1:9/v1/chat/completions", model="qwen", vision=True))


class _FakeToolCall:
    class _Func:
        def __init__(self, name, args):
            self.name = name
            self.arguments = args

    def __init__(self, name, args_str):
        self.id = "call_1"
        self.type = "function"
        self.function = self._Func(name, args_str)


class _FakeMsg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeResp:
    def __init__(self, msg):
        self.choices = [type("_C", (), {"message": msg})()]


class _FakeCompletions:
    def __init__(self, queue, seen):
        self._queue = list(queue)
        self.seen = seen

    def create(self, **kwargs):
        self.seen.append(kwargs)
        return self._queue.pop(0)


class _FakeClient:
    def __init__(self, queue, seen):
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(queue, seen)})()


class LlmDecideAndPageLoopTest(unittest.TestCase):
    """P3: real-LLM ``decide`` (vision + tools + result feedback) and the page loop."""

    def test_make_llm_decide_parses_tool_call_then_done(self):
        seen: list = []
        queue = [
            _FakeResp(_FakeMsg(content="", tool_calls=[_FakeToolCall("read_page", '{"page":0}')])),
            _FakeResp(_FakeMsg(content="done!")),
        ]
        client = _FakeClient(queue, seen)
        with mock.patch.object(translator, "OpenAI", lambda **_k: client):
            decide = agent.make_llm_decide(_vision_model(), task="observe the page",
                                           image_provider=lambda _s: b"IMG")
        s = agent.WorkflowState("a.pdf", "English")
        s.page(0)
        d1 = decide("obs1", s)
        self.assertEqual("call", d1.action)
        self.assertEqual("read_page", d1.tool)
        self.assertEqual({"page": 0}, d1.arguments)
        # The source page image is in the first user message (vision observation).
        self.assertIn("data:image/png;base64", json.dumps(seen[0]["messages"]))
        self.assertEqual("auto", seen[0]["tool_choice"])
        # Feed the executed tool result back → the next model answer is 'done'.
        last = agent.AgentResult(ok=True, op_tool="read_page", op_args={"page": 0},
                                 result={"blocks": []})
        d2 = decide("obs2", s, last)
        self.assertEqual("done", d2.action)
        self.assertEqual("done!", d2.summary)
        roles = [m["role"] for m in seen[1]["messages"]]
        self.assertIn("assistant", roles)   # the tool_call was echoed back
        self.assertIn("tool", roles)        # the result was fed back

    def test_make_source_tools_read_page_is_read_only(self):
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("你好", page=0, x0=0, y0=0, x1=10, y1=10,
                                in_table=True, is_chart=False)]],
            blocks=["你好"], block_pages=[0])
        tools = agent.make_source_tools(s)
        out = tools["read_page"](0)
        self.assertEqual(1, len(out["blocks"]))
        self.assertEqual("你好", out["blocks"][0]["text"])
        self.assertTrue(out["blocks"][0]["in_table"])
        self.assertFalse(out["blocks"][0]["is_chart"])

    def test_ask_user_registered_and_prompt_guides_model_to_ask(self):
        # ``ask_user`` is a declared tool; the system prompt tells the model when
        # to call it (the "该问就问" trigger).
        t = agent.by_name("ask_user")
        self.assertIsNotNone(t)
        self.assertIn("question", t.parameters["required"])
        seen: list = []
        queue = [_FakeResp(_FakeMsg(content="ok"))]
        client = _FakeClient(queue, seen)
        with mock.patch.object(translator, "OpenAI", lambda **_k: client):
            decide = agent.make_llm_decide(_vision_model(), task="translate the page",
                                           image_provider=lambda _s: b"")
        s = agent.WorkflowState("a.pdf", "English")
        s.page(0)
        decide("obs", s)
        system = [m["content"] for m in seen[0]["messages"] if m["role"] == "system"][0]
        self.assertIn("ask_user", system)
        self.assertIn("【与用户交互】", system)
        self.assertIn("ask_user", [f["function"]["name"] for f in seen[0]["tools"]])

    def test_llm_decide_reinjects_preview_region_image(self):
        # After a ``preview_page`` call, the user-framed region is injected as a
        # *new* image in the next user message (true "send to AI to see").
        seen: list = []
        queue = [
            _FakeResp(_FakeMsg(content="", tool_calls=[_FakeToolCall("preview_page", '{"page":0}')])),
            _FakeResp(_FakeMsg(content="ok")),
        ]
        client = _FakeClient(queue, seen)
        with mock.patch.object(translator, "OpenAI", lambda **_k: client):
            decide = agent.make_llm_decide(_vision_model(), task="t",
                                           image_provider=lambda _s: b"SRC")
        s = agent.WorkflowState("a.pdf", "English")
        s.page(0)
        d1 = decide("obs1", s)
        self.assertEqual("call", d1.action)
        self.assertEqual("preview_page", d1.tool)
        # Feed the preview result (a framed region) back → the region image appears
        # in the next user message's image content.
        last = agent.AgentResult(ok=True, op_tool="preview_page",
                                 op_args={"page": 0},
                                 result={"image": b"REGION", "rect": [0, 0, 10, 10]})
        d2 = decide("obs2", s, last)
        self.assertEqual("done", d2.action)
        # base64("REGION") = "UkVHSU9O"; it must be in the last user message.
        last_user = [m for m in seen[1]["messages"] if m["role"] == "user"][-1]
        self.assertIn("UkVHSU9O", json.dumps(last_user))

    def test_get_layout_clusters_rows(self):
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("A", page=0, x0=0, y0=0, x1=10, y1=10),
                    pdfio.Block("B", page=0, x0=20, y0=0, x1=30, y1=10),
                    pdfio.Block("C", page=0, x0=0, y0=20, x1=10, y1=30)]],
            blocks=["A", "B", "C"], block_pages=[0, 0, 0])
        layout = agent.make_source_tools(s)["get_layout"](0)
        self.assertEqual(2, layout["rows"])
        self.assertIn(["A", "B"], layout["grid"])

    def test_run_page_visual_skips_when_model_has_no_vision(self):
        model = ModelConfig.from_dict(dict(id="x", name="x", type="llama-server",
                                           endpoint="http://127.0.0.1:9", model="m"))
        s = agent.WorkflowState("missing.pdf", "English")
        s.page(0)
        out = agent.run_page_visual(s, 0, model, task="t")
        self.assertIs(out, s)
        self.assertEqual(0, len(s.ops))
        self.assertTrue(any("不支持视觉" in m for m in s.log), s.log)


def _dummy_model() -> ModelConfig:
    return ModelConfig.from_dict(dict(id="x", name="x", type="llama-server",
                                      endpoint="http://127.0.0.1:9", model="m"))


class PageExecutorsTest(unittest.TestCase):
    """The deterministic content/verify executors write only to ``out_doc``."""

    def _state(self):
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("总资产", page=0, x0=0, y0=0, x1=50, y1=10),
                    pdfio.Block("17,485,938,749.91", page=0, x0=60, y0=0, x1=160, y1=10,
                                in_table=True),
                    pdfio.Block("总负债", page=0, x0=0, y0=20, x1=50, y1=30)]],
            blocks=["总资产", "17,485,938,749.91", "总负债"], block_pages=[0, 0, 0])
        return s

    def test_set_text_writes_to_out_doc(self):
        s = self._state()
        tools = agent.make_page_executors(s, _dummy_model())
        self.assertTrue(tools["set_text"](0, 0, "Total assets")["ok"])
        self.assertEqual("Total assets", s.out_doc[0]["text"])

    def test_set_text_refuses_numeric_block(self):
        s = self._state()
        tools = agent.make_page_executors(s, _dummy_model())
        res = tools["set_text"](0, 1, "1,234,567.89")
        self.assertFalse(res["ok"])
        self.assertIn("不可被 AI 改写", res["error"])
        self.assertNotIn(1, (s.out_doc or {}))

    def test_check_residual_finds_untranslated(self):
        s = self._state()
        tools = agent.make_page_executors(s, _dummy_model())
        # Only block 0 translated; block 2 (总负债) still empty → residual CJK.
        s.out_doc = {0: {"text": "Total assets"}}
        residual = tools["check_residual"]()
        self.assertTrue(any(r["index"] == 2 for r in residual["residual"]), residual)

    def test_set_font_clamps(self):
        s = self._state()
        tools = agent.make_page_executors(s, _dummy_model())
        self.assertEqual(40.0, tools["set_font"](0, 0, 999.0)["size"])
        self.assertEqual(6.0, tools["set_font"](0, 1, 0.5)["size"])

    def test_translate_retries_on_transient_error(self):
        class _FlakyEngine:
            def __init__(self, model):
                self.calls = 0
                self.model = model

            def translate_blocks(self, blocks, lang, **kw):
                self.calls += 1
                if self.calls == 1:
                    # First call: transient failure -> keeps source + reports errors.
                    return type("R", (), {"translated": [str(blocks[0])], "errors": ["conn"]})()
                return type("R", (), {"translated": ["TRANSLATED"], "errors": []})()

        s = self._state()
        with mock.patch.object(translator, "TranslationEngine", _FlakyEngine):
            tools = agent.make_page_executors(s, _dummy_model())
        res = tools["translate_block"](0, "总资产")
        self.assertTrue(res["ok"])
        self.assertEqual("TRANSLATED", res["translated"])
        self.assertEqual("TRANSLATED", s.out_doc[0]["text"])

    def test_preview_page_invokes_handler_and_returns_bytes(self):
        s = self._state()
        got = {}

        def handler(page, what, region):
            got.update(page=page, what=what, region=region)
            return {"png": b"PNG", "rect": [0, 0, 10, 10]}

        tools = agent.make_page_executors(s, _dummy_model(), preview_handler=handler)
        res = tools["preview_page"](0, "source", None)
        self.assertTrue(res["ok"])
        self.assertEqual({"page": 0, "what": "source", "region": None}, got)
        self.assertEqual(b"PNG", res["image"])
        self.assertEqual([0, 0, 10, 10], res["rect"])

    def test_preview_page_accepts_raw_png_from_handler(self):
        s = self._state()
        tools = agent.make_page_executors(
            s, _dummy_model(), preview_handler=lambda page, what, region: b"RAW")
        res = tools["preview_page"](0, "source")
        self.assertEqual(b"RAW", res["image"])
        self.assertIsNone(res["rect"])

    def test_preview_page_without_handler_is_fail_closed(self):
        s = self._state()
        tools = agent.make_page_executors(s, _dummy_model())
        res = tools["preview_page"](0, "source")
        self.assertFalse(res["ok"])
        self.assertIn("预览通道未接线", res["error"])

    def test_ask_user_routes_through_answer_handler(self):
        s = self._state()
        got: dict = {}

        def answer_handler(question, options, target):
            got.update(question=question, options=options, target=target)
            return {"value": "keep", "target": target}

        tools = agent.make_page_executors(s, _dummy_model(), answer_handler=answer_handler)
        res = tools["ask_user"]("保留该块？", ["keep", "translate"], "t")
        self.assertTrue(res["ok"])
        self.assertEqual("保留该块？", got["question"])
        self.assertEqual(["keep", "translate"], got["options"])
        self.assertEqual("keep", res["answer"])

    def test_ask_user_without_handler_is_fail_closed(self):
        s = self._state()
        tools = agent.make_page_executors(s, _dummy_model())
        res = tools["ask_user"]("q")
        self.assertFalse(res["ok"])
        self.assertIn("问答通道未接线", res["error"])


if __name__ == "__main__":
    unittest.main()
