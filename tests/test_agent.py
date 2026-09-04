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
        # ``classify_block`` is a standalone classifier (via ``make_agent_tools``),
        # not part of the per-page ``AGENT_TOOLS`` registry that ``run_page_visual``
        # advertises; ``retranslate_block`` remains a registry tool.
        names = [t.name for t in agent.AGENT_TOOLS]
        self.assertIn("retranslate_block", names)
        self.assertNotIn("classify_block", names)
        self.assertIsNotNone(agent.by_name("retranslate_block"))

    def test_source_read_tools_are_readonly(self):
        # The original is protected: read/observe tools only read it (target source).
        for n in ("read_page", "get_layout", "get_doc_info", "classify_page", "render_page"):
            t = agent.by_name(n)
            self.assertIsNotNone(t, n)
            self.assertEqual("source", t.target, n)

    def test_write_tools_target_output_and_are_free(self):
        # The translation is fully editable (no confirmation); the original is
        # never a write target.
        write = ("set_text", "delete_block", "apply_annotation", "apply_terminology",
                 "translate_block", "retranslate_block")
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

    def test_block_edits_target_output_page_tools_absent(self):
        # Only the effective block-edit tools are exposed; the layout / page-level ops
        # (font/align/create/move block, page create/delete/move, draw/erase) are not
        # wired to the renderer and are deliberately absent so the model never sees a
        # tool that silently does nothing.
        names = [t.name for t in agent.AGENT_TOOLS]
        self.assertIn("delete_block", names)
        self.assertEqual("output", agent.by_name("delete_block").target)
        for n in ("delete_page", "create_page", "move_page", "create_block", "move_block",
                  "set_font", "set_align", "draw_table", "merge_cells", "grid_rule",
                  "cover_region", "erase_text_layer", "drop_element", "qa_render",
                  "render_region", "audit"):
            self.assertNotIn(n, names)

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
        names = ["read_page", "delete_block"]
        tools = agent.agent_openai_tools(names=names)
        self.assertEqual(2, len(tools))
        self.assertEqual({"read_page", "delete_block"}, {t["function"]["name"] for t in tools})


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

        def translate_block(index, text, target_lang):
            calls.append(("translate_block", index))
            return {"ok": True, "translated": "Total assets"}

        return {"read_page": read_page,
                "classify_block": classify_block,
                "translate_block": translate_block}

    def test_scripted_calls_run_in_order_and_mutate_state(self):
        s = self._state()
        calls: list = []
        decide = _scripted([
            agent.Decision(action="call", tool="read_page", arguments={"page": 0}),
            agent.Decision(action="call", tool="classify_block",
                           arguments={"index": 0, "text": "总资产"}),
            agent.Decision(action="call", tool="translate_block",
                           arguments={"index": 0, "text": "总资产", "target_lang": "English"}),
            agent.Decision(action="done", summary="page done"),
        ])
        agent.run_agent_run(s, self._tools(calls), decide)
        # Tool calls in the scripted order.
        self.assertEqual([("read_page", 0), ("classify_block", 0), ("translate_block", 0)], calls)
        # Provenance recorded, budget advanced, last action is done (no pending question).
        self.assertEqual(["read_page", "classify_block", "translate_block"],
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

    def test_llm_decide_uses_translation_params(self):
        # ``decide`` is TRANSLATION-side (by definition only the text chat is
        # interaction-side), so it uses the model's translation temperature and
        # ``reasoning_effort`` (models.json) — NOT ``interaction_*``.
        seen: list = []
        queue = [
            _FakeResp(_FakeMsg(content="", tool_calls=[_FakeToolCall("read_page", '{"page":0}')])),
        ]
        client = _FakeClient(queue, seen)
        model = ModelConfig.from_dict(dict(
            id="v", name="v", type="llama-server",
            endpoint="http://127.0.0.1:9/v1/chat/completions", model="qwen",
            vision=True, temperature=0.3, reasoning_effort="low",
            interaction_temperature=0.9, interaction_reasoning_effort="high",
        ))
        with mock.patch.object(translator, "OpenAI", lambda **_k: client):
            decide = agent.make_llm_decide(model, task="t",
                                           image_provider=lambda _s: b"")
        s = agent.WorkflowState("a.pdf", "English")
        s.page(0)
        decide("obs", s)
        kw = seen[0]
        # Translation-side params, not the interaction overrides.
        self.assertEqual(0.3, kw["temperature"])
        self.assertEqual({"reasoning_effort": "low"}, kw["extra_body"])
        self.assertEqual("auto", kw["tool_choice"])

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

    def test_render_page_invokes_render_handler_and_returns_image(self):
        s = self._state()
        got: dict = {}

        def render(page, what):
            got.update(page=page, what=what)
            return b"PNG"

        tools = agent.make_page_executors(s, _dummy_model(), render_handler=render)
        res = tools["render_page"](0, "translation")
        self.assertTrue(res["ok"])
        self.assertEqual({"page": 0, "what": "translation"}, got)
        self.assertEqual(b"PNG", res["image"])

    def test_render_page_no_handler_fails_closed(self):
        s = self._state()
        tools = agent.make_page_executors(s, _dummy_model())
        res = tools["render_page"](0)
        self.assertFalse(res["ok"])
        self.assertIn("渲染通道", res["error"])

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


class MultipageAndFlatIndexTest(unittest.TestCase):
    """Regressions for the multi-page agent loop (budget + index + tool surface)."""

    class _AlwaysReadPageCompletions:
        def __init__(self, seen):
            self.seen = seen

        def create(self, **kwargs):
            self.seen.append(kwargs)
            return _FakeResp(_FakeMsg(content="",
                                      tool_calls=[_FakeToolCall("read_page", '{"page":0}')]))

    class _AlwaysReadPageClient:
        def __init__(self, seen):
            self.chat = type("_Chat", (),
                             {"completions": MultipageAndFlatIndexTest._AlwaysReadPageCompletions(seen)})()

    def _two_page_state(self):
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[
                [pdfio.Block("A", page=0, x0=0, y0=0, x1=10, y1=10),
                 pdfio.Block("B", page=0, x0=0, y0=20, x1=10, y1=30)],
                [pdfio.Block("C", page=1, x0=0, y0=0, x1=10, y1=10)],
            ],
            blocks=["A", "B", "C"], block_pages=[0, 0, 1])
        return s

    def test_per_page_budget_is_reset_across_pages(self):
        # Regression: the worker drives *every* page through the same WorkflowState,
        # but the budget counters used to accumulate — so page 2 started already
        # exhausted and never ran.  Each ``run_page_visual`` must reset them.
        seen: list = []
        state = self._two_page_state()   # one shared state, two pages (as the worker)

        def make_client():
            return self._AlwaysReadPageClient(seen)

        with mock.patch.object(translator, "OpenAI", lambda **_k: make_client()):
            for page in (0, 1):
                agent.run_page_visual(state, page, _vision_model(),
                                      task="t", max_steps=2)
        # Page 0 and page 1 each made 2 read_page calls (fresh budget each).
        self.assertEqual(2, state.budget.used_steps)
        self.assertEqual(4, len(state.ops))
        # The model was only offered tools that are actually bound — the judgment /
        # draw tools (verify_number / draw_table / cover_region) must not surface.
        for kw in seen:
            names = [f["function"]["name"] for f in kw["tools"]]
            self.assertIn("read_page", names)
            self.assertIn("translate_block", names)
            self.assertNotIn("verify_number", names)
            self.assertNotIn("draw_table", names)

    def test_read_page_reports_flat_global_index(self):
        # Regression: read_page used to report a *per-page local* index, which
        # silently addresses a different block once page > 0 (the write tools all
        # key on the flat document index).
        s = self._two_page_state()
        tools = agent.make_source_tools(s)
        page1 = tools["read_page"](1)
        self.assertEqual(1, len(page1["blocks"]))
        self.assertEqual("C", page1["blocks"][0]["text"])
        self.assertEqual(2, page1["blocks"][0]["index"])   # offset 2 (page 0 had 2 blocks)

    def test_cancel_fn_stops_the_loop(self):
        s = agent.WorkflowState("a.pdf", "English")
        s.todo.append(agent.Goal(kind="page", page=0))
        calls: list = []

        def decide(*_a):
            return agent.Decision(action="call", tool="read_page", arguments={"page": 0})

        def read_page(page):
            calls.append(page)
            return {"blocks": []}

        agent.run_agent_run(s, {"read_page": read_page}, decide, max_steps=100,
                            cancel_fn=lambda: len(calls) >= 2)
        self.assertEqual(2, len(calls))
        self.assertEqual(2, s.budget.used_steps)


class FeedbackTest(unittest.TestCase):
    """The model-won't-be-overwhelmed feedback channel (concise text + images only)."""

    def test_observe_reports_tool_status_without_dumping_result(self):
        from translate_app.agent import flow as af

        s = agent.WorkflowState("a.pdf", "English")
        agent_flow = af.FlowAgent(s, {}, lambda *_a: agent.Decision(action="done"))
        agent_flow.last_result = agent.AgentResult(
            ok=True, op_tool="read_page",
            result={"page": 0, "blocks": [{"index": 0, "text": "x"*400, "bbox": [0, 0, 1, 1]}]})
        obs = agent_flow.observe()
        self.assertNotIn("result=", obs)
        self.assertIn("last=read_page ok=True", obs)

    def test_result_for_message_strips_image_and_bytes(self):
        from translate_app.agent import flow as af

        r = af._result_for_message({"ok": True, "page": 0, "image": b"PNG", "bytes": b"xx"})
        self.assertNotIn("image", r)
        self.assertNotIn("bytes", r)
        self.assertEqual(b"", r.get("bytes", b""))

    def test_summarize_result_truncates_and_strips(self):
        from translate_app.agent import flow as af

        s = af._summarize_result({"ok": True, "block": "y"*1000, "image": b"PNG"})
        self.assertNotIn("image", s)
        self.assertLess(len(s), 200)
        self.assertIn("…", s)

    def test_llm_decide_tool_message_strips_image(self):
        # A tool result carrying image bytes must NOT dump them into the text
        # ``tool`` message — the image is injected only as an ``image_url``.
        from translate_app.agent import flow as af

        s = agent.WorkflowState("a.pdf", "English")
        seen: list = []
        queue = [
            _FakeResp(_FakeMsg(content="", tool_calls=[
                _FakeToolCall("render_page", '{"page": 0}')])),
            _FakeResp(_FakeMsg(content="done")),
        ]
        client = _FakeClient(queue, seen)
        with mock.patch.object(translator, "OpenAI", lambda **_k: client):
            decide = af.make_llm_decide(_vision_model(), task="t",
                                        tool_names=["render_page"])
            d1 = decide("obs1", s, None)
            self.assertEqual("render_page", d1.tool)
            d2 = decide("obs2", s, agent.AgentResult(
                ok=True, op_tool="render_page",
                result={"ok": True, "page": 0, "image": b"PNG"}))
            self.assertEqual("done", d2.summary)
        msgs = seen[1]["messages"]
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        self.assertTrue(tool_msgs)
        self.assertNotIn("PNG", tool_msgs[0]["content"])
        self.assertNotIn('"image"', tool_msgs[0]["content"])
        user = [m for m in msgs if m.get("role") == "user"][-1]
        self.assertTrue(any(p.get("type") == "image_url" for p in user["content"]))


if __name__ == "__main__":
    unittest.main()
