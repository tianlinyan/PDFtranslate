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

    def test_record_op_increments_budget_atomically_under_threads(self):
        # ``used_steps += 1`` is a read-modify-write; the state lock must prevent
        # parallel page loops (page_concurrency > 1) from losing increments.
        import threading

        s = agent.WorkflowState("a.pdf", "English")
        n_threads, per_thread = 8, 2000
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            for _ in range(per_thread):
                s.record_op("t")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(n_threads * per_thread, s.budget.used_steps)
        self.assertEqual(n_threads * per_thread, len(s.ops))

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

    def test_openai_tools_description_includes_returns(self):
        # Regression: the model-facing description must include each tool's ``returns``
        # (result shape / index semantics).  Without it the model never learns e.g.
        # read_page's "每块 index 为全文档扁平索引" or check_table's field names, so it
        # can't reliably use the returned fields.
        descs = {t["function"]["name"]: t["function"]["description"]
                 for t in agent.agent_openai_tools()}
        self.assertIn("扁平", descs["read_page"])
        self.assertIn("complete", descs["check_table"])
        self.assertIn("missing", descs["check_numbers"])
        self.assertTrue(descs["set_text"].startswith("把某块文本直接置为指定值"))
        # The retranslate_block description must be truthful: it returns text only.
        self.assertIn("只返回译文", descs["retranslate_block"])


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

    def test_run_agent_run_requires_decide(self):
        # Passing ``decide=None`` must fail fast with a clear error, not crash on
        # the first step when the loop tries to ask the model.
        with self.assertRaises(ValueError):
            agent.run_agent_run(agent.WorkflowState("a.pdf", "English"), {}, None)


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

    def test_llm_decide_raises_translation_cancelled_on_cancel(self):
        # A cancel during an in-flight request surfaces as ``TranslationCancelled``
        # (a control signal), not a masked "decide failed → done" that would let the
        # page keep running.  The watchdog closes the client → ``create`` raises →
        # ``cancel()`` is true → re-raise.
        class _Completions:
            def create(self, **_k):
                raise RuntimeError("connection aborted")
        class _Chat:
            completions = _Completions()
        class _Client:
            def __init__(self, **_k):
                self.chat = _Chat()
            def close(self):
                pass
        with mock.patch.object(translator, "OpenAI", lambda **_k: _Client()):
            decide = agent.make_llm_decide(_vision_model(), task="observe the page",
                                           image_provider=lambda _s: b"",
                                           cancel=lambda: True)
        s = agent.WorkflowState("a.pdf", "English")
        s.page(0)
        with self.assertRaises(translator.TranslationCancelled):
            decide("obs", s)

    def test_flow_agent_call_reraises_translation_cancelled(self):
        # A per-tool ``TranslationCancelled`` must propagate (not be swallowed into an
        # ``ok=False`` result), so the worker stops the run immediately.
        s = agent.WorkflowState("a.pdf", "English")
        def boom(**_k):
            raise translator.TranslationCancelled()
        fa = agent.FlowAgent(s, {"boom": boom},
                             decide=lambda *a, **k: agent.Decision(action="call", tool="boom"))
        with self.assertRaises(translator.TranslationCancelled):
            fa._call("boom", {})

    def test_tool_call_logs_start_and_elapsed(self):
        # Each tool call logs its name when it starts and the elapsed time when it
        # completes, so the main log shows what is running and how long it took.
        s = agent.WorkflowState("a.pdf", "English")
        logs: list[str] = []
        fa = agent.FlowAgent(s, {"demo": lambda **kw: {"ok": True}},
                             decide=lambda *a, **k: agent.Decision(action="call", tool="demo"),
                             log=logs.append)
        fa.step()
        self.assertTrue(any("工具开始：demo" in m for m in logs), logs)
        self.assertTrue(any("demo" in m and "用时" in m and "s" in m for m in logs), logs)

    def test_window_messages_bounds_and_keeps_user_boundary(self):
        # Scheme 2: the decide request is windowed to a bounded, protocol-valid tail
        # (system + last messages, cut at a *user* boundary so an assistant tool_call
        # and its tool result are never split), so a page with many tool rounds does
        # not re-send O(n²) history.
        from translate_app.agent.flow import _window_messages
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(30):
            msgs.append({"role": "user", "content": f"u{i}"})
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": f"t{i}", "type": "function",
                                         "function": {"name": "f", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": f"r{i}"})
        msgs.append({"role": "user", "content": "final obs"})
        w = _window_messages(msgs, cap=12)
        self.assertLess(len(w), len(msgs))
        self.assertLessEqual(len(w), 12 + 3)          # back-off may keep ≤2 extra
        self.assertEqual("system", w[0]["role"])
        self.assertEqual("user", w[1]["role"])         # cut at a user boundary
        self.assertEqual("final obs", w[-1]["content"])  # freshest obs retained
        # No tool message without its assistant, and no assistant before its tool.
        roles = [m["role"] for m in w]
        self.assertNotIn("assistant", roles[-1])        # last msg is the fresh obs
        # A short list passes through unchanged.
        short = [{"role": "system", "content": "s"}, {"role": "user", "content": "x"}]
        self.assertEqual(short, _window_messages(short))

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
        # The system prompt carries the tool function + usage reference, so the model
        # reads every tool's contract here (not just one-line descriptions in the
        # tools array): grouped, with the gotchas it must not guess.
        self.assertIn("工具功能与用法", system)
        self.assertIn("扁平 index", system)          # read_page index semantics
        self.assertIn("只返回译文", system)           # retranslate_block: returns text only
        self.assertIn("check_table", system)
        self.assertIn("ask_user(question", system)
        # The system prompt also carries the general wording WORKFLOW: how to chain
        # the tools to build / edit / verify, so the method is in the prompt too.
        self.assertIn("翻译通用工作法", system)
        self.assertIn("构建", system)
        self.assertIn("校验", system)

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
        # An out-of-range page fails closed (returns an empty grid, not an IndexError).
        empty = agent.make_source_tools(s)["get_layout"](99)
        self.assertEqual(0, empty["rows"])
        self.assertEqual([], empty["grid"])

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

    def test_check_residual_honors_page(self):
        # Regression: check_residual used to scan the whole document regardless of
        # the required ``page`` — a per-page request now only reports that page.
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("总资产", page=0, x0=0, y0=0, x1=50, y1=10)],
                   [pdfio.Block("营业收入", page=1, x0=0, y0=0, x1=50, y1=10)]],
            blocks=["总资产", "营业收入"], block_pages=[0, 1])
        s.out_doc = {}
        tools = agent.make_page_executors(s, _dummy_model())
        page1 = tools["check_residual"](1)
        self.assertEqual(1, page1["page"])
        self.assertEqual([1], [r["index"] for r in page1["residual"]])
        page0 = tools["check_residual"](0)
        self.assertEqual([0], [r["index"] for r in page0["residual"]])

    def test_check_tools_ignore_verbatim_number_blocks(self):
        # Regression: a pure-number cell (17,485,938,749.91) is kept verbatim and must
        # NOT be flagged as "empty / missing" — otherwise the review loop tries to
        # "fix" a numeric cell it is forbidden to rewrite and gets stuck.
        s = self._state()
        tools = agent.make_page_executors(s, _dummy_model())
        s.out_doc = {}   # nothing translated yet
        missing = [m["index"] for m in tools["check_missing"]()["missing"]]
        residual = [r["index"] for r in tools["check_residual"]()["residual"]]
        # The numeric block (index 1) is excluded from both.
        self.assertNotIn(1, missing)
        self.assertNotIn(1, residual)
        # The translatable text labels (0 总资产, 2 总负债) are still flagged as missing.
        self.assertIn(0, missing)
        self.assertIn(2, missing)

    def test_review_check_tools_registered(self):
        # The three new audit tools are part of the registry (and thus advertised).
        for n in ("check_numbers", "check_table", "check_layout"):
            t = agent.by_name(n)
            self.assertIsNotNone(t, n)
            self.assertEqual("verify", t.category, n)
            self.assertFalse(t.destructive, n)

    def test_audit_page_tool_registered_and_bound(self):
        # ``audit_page`` is the module-level aggregator: advertised in the registry
        # AND bound in ``make_page_executors`` (so the agent can both see and call it).
        t = agent.by_name("audit_page")
        self.assertIsNotNone(t)
        self.assertEqual("verify", t.category)
        self.assertFalse(t.destructive)
        tools = agent.make_page_executors(self._state(), _dummy_model())
        self.assertIn("audit_page", tools)

    def test_audit_page_reports_missing_and_skips_protected(self):
        s = self._state()
        tools = agent.make_page_executors(s, _dummy_model())
        # Nothing translated yet → the two text labels are missing/residual; the
        # numeric cell (flat index 1) is protected and must never be reported.
        res = tools["audit_page"](0)
        self.assertFalse(res["clean"])
        kinds = {iss["check"] for iss in res["issues"]}
        self.assertIn("missing", kinds)
        self.assertIn("residual", kinds)
        reported_idx = [iss.get("index") for iss in res["issues"] if "index" in iss]
        self.assertNotIn(1, reported_idx)

    def test_audit_page_subset_checks(self):
        s = self._state()
        tools = agent.make_page_executors(s, _dummy_model())
        s.out_doc = {0: {"text": "Total assets"}, 2: {"text": "Total liabilities"}}
        res = tools["audit_page"](0, checks=["numbers", "table"])
        self.assertEqual(["numbers", "table"], res["checks_requested"])
        self.assertTrue(res["clean"])

    def test_audit_page_flags_number_discrepancy(self):
        # A translation that drops a digit is flagged by the aggregate audit.
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("2023 年营收 3.14 亿元", page=0, x0=0, y0=0,
                                x1=100, y1=10)]],
            blocks=["2023 年营收 3.14 亿元"], block_pages=[0])
        s.out_doc = {0: {"text": "In 2023 revenue was 3.1 hundred million yuan"}}
        res = agent.audit_page(s, 0, checks=["numbers"])
        self.assertFalse(res["clean"])
        self.assertTrue(any(iss["check"] == "numbers" and iss.get("missing")
                            for iss in res["issues"]), res)

    def test_check_numbers_detects_altered_number(self):
        # A translation that drops/alters a digit must be flagged.
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("2023 年营收 3.14 亿元", page=0, x0=0, y0=0, x1=100, y1=10)]],
            blocks=["2023 年营收 3.14 亿元"], block_pages=[0])
        s.out_doc = {0: {"text": "In 2023 revenue was 3.1 hundred million yuan"}}
        tools = agent.make_page_executors(s, _dummy_model())
        res = tools["check_numbers"](0)
        self.assertTrue(any(r["index"] == 0 and r["missing"] for r in res["numbers"]), res)

    def test_check_numbers_ignores_separator_format(self):
        # Re-formatting thousands separators is NOT a defect — the value is equal.
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("总资产 17,485,938,749.91 元", page=0, x0=0, y0=0, x1=100, y1=10)]],
            blocks=["总资产 17,485,938,749.91 元"], block_pages=[0])
        s.out_doc = {0: {"text": "Total assets 17485938749.91 yuan"}}
        tools = agent.make_page_executors(s, _dummy_model())
        self.assertEqual([], tools["check_numbers"](0)["numbers"])
        # A wrong value (transposed digits) IS flagged.
        s.out_doc = {0: {"text": "Total assets 17485938749.19 yuan"}}
        self.assertTrue(tools["check_numbers"](0)["numbers"])

    def test_check_numbers_ignores_unit_and_date_forms(self):
        # A report date in any spelling ("December 31, 2023" / "2023-12-31" /
        # "2023年12月31日") is the same value set, and "3.14 亿元" is the same value
        # as "314 million yuan" — a correct translation of either must not be
        # flagged.  The old string-reformat compare merged "December 31, 2023"
        # into one giant token and could not attach a Latin unit multiplier.
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("2023 年 12 月 31 日，公司营收 3.14 亿元", page=0,
                                x0=0, y0=0, x1=100, y1=10)]],
            blocks=["2023 年 12 月 31 日，公司营收 3.14 亿元"], block_pages=[0])
        tools = agent.make_page_executors(s, _dummy_model())
        s.out_doc = {0: {"text": "On December 31, 2023 revenue was 314 million yuan"}}
        self.assertEqual([], tools["check_numbers"](0)["numbers"])
        # A dropped digit (31 million vs the source's 314 million) IS flagged.
        s.out_doc = {0: {"text": "As of May 2023 revenue was 31 million yuan"}}
        self.assertTrue(tools["check_numbers"](0)["numbers"])

    def test_check_residual_ignores_cjk_translation_for_cjk_target(self):
        # Regression (fix 1): the residual check scanned the language NAME for CJK
        # glyphs, so the default Chinese target ("Simplified Chinese" is ASCII) was
        # judged "Latin-only" and every Chinese translation was reported as
        # residual.  The target's script is decided by IDENTITY now, so a CJK
        # target never flags its own translations.
        s = agent.WorkflowState("a.pdf", "Simplified Chinese")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("Unchecked fixed assets", page=0, x0=0, y0=0,
                                x1=100, y1=10)]],
            blocks=["Unchecked fixed assets"], block_pages=[0])
        s.out_doc = {0: {"text": "未审定的固定资产"}}
        tools = agent.make_page_executors(s, _dummy_model())
        self.assertEqual([], tools["check_residual"](0)["residual"])

    def test_check_residual_flags_untranslated_latin_prose_for_cjk_target(self):
        # Symmetric residual window for a CJK target: an untranslated English
        # SENTENCE is residual; a code / unit run (GB/T 33436-2016) is not.
        s = agent.WorkflowState("a.pdf", "Simplified Chinese")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("group revenue", page=0, x0=0, y0=0, x1=100, y1=10),
                    pdfio.Block("标准", page=0, x0=0, y0=20, x1=100, y1=30)]],
            blocks=["group revenue", "标准"], block_pages=[0])
        s.out_doc = {0: {"text": "The revenue of the group grew strongly this year"},
                     1: {"text": "GB/T 33436-2016"}}
        tools = agent.make_page_executors(s, _dummy_model())
        self.assertEqual([0], [r["index"] for r in tools["check_residual"](0)["residual"]])

    def test_numeric_cells_refused_on_review_path(self):
        # Regression (fix 9): only set_text (not translate_block/retranslate_block)
        # was write-protected — the AI "correcting" a figure through those tools
        # could change it.  Both now refuse a numeric source outright.
        s = self._state()
        tools = agent.make_page_executors(s, _dummy_model())
        res = tools["translate_block"](1, "1,234,567.89")
        self.assertFalse(res["ok"])
        self.assertIn("不可被 AI 改写", res["error"])
        self.assertNotIn(1, (s.out_doc or {}))
        res = tools["retranslate_block"]("1,234,567.89")
        self.assertFalse(res["ok"])
        self.assertIn("不可被 AI 改写", res["error"])

    def test_check_table_reports_empty_cells(self):
        # A translatable table cell with no translation and an empty prose block are
        # reported; verbatim numerics are excluded from the counts.
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("营业利润", page=0, x0=0, y0=0, x1=50, y1=10, in_table=True),
                    pdfio.Block("总资产", page=0, x0=0, y0=20, x1=50, y1=30)]],
            blocks=["营业利润", "总资产"], block_pages=[0])
        s.out_doc = {0: {"text": "Operating profit"}}
        tools = agent.make_page_executors(s, _dummy_model())
        res = tools["check_table"](0)
        self.assertEqual(1, res["source_cells"])
        self.assertEqual(1, res["translated_cells"])
        self.assertEqual([], res["empty_cells"])
        self.assertEqual([1], res["empty_text"])
        self.assertFalse(res["complete"])

    def test_check_table_complete_when_all_translated(self):
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("营业利润", page=0, x0=0, y0=0, x1=50, y1=10, in_table=True),
                    pdfio.Block("总资产", page=0, x0=0, y0=20, x1=50, y1=30)]],
            blocks=["营业利润", "总资产"], block_pages=[0])
        s.out_doc = {0: {"text": "Operating profit"}, 1: {"text": "Total assets"}}
        tools = agent.make_page_executors(s, _dummy_model())
        res = tools["check_table"](0)
        self.assertTrue(res["complete"])
        self.assertEqual([], res["empty_cells"])
        self.assertEqual([], res["empty_text"])

    def test_check_layout_flags_overflowing_prose_block(self):
        # A much-longer prose translation that cannot fit its box is flagged (fs hit
        # the readable floor and still overflows) — the exporter draws it past the box.
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("Total assets were 17,485,938,749.91 yuan at end of 2023",
                                page=0, x0=0, y0=0, x1=120, y1=14, size=10)]],
            blocks=["Total assets were 17,485,938,749.91 yuan at end of 2023"],
            block_pages=[0])
        s.out_doc = {0: {"text": "The total assets of the group as at 31 December 2023 amounted to "
                                 "one hundred seventy four billion eight hundred fifty nine million "
                                 "three hundred ninety thousand seven hundred forty nine point "
                                 "nine one yuan for the consolidated and parent entities combined"}}
        tools = agent.make_page_executors(s, _dummy_model())
        kinds = {i["kind"] for i in tools["check_layout"](0)["issues"]}
        self.assertTrue(kinds, "expected the long translation to be flagged")
        self.assertTrue({"overflow", "too_small", "crowding"}.intersection(kinds))

    def test_check_layout_clean_for_short_translation(self):
        # A translation that fits its box is NOT flagged (no false positives).
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("总资产", page=0, x0=0, y0=0, x1=50, y1=10, size=10)]],
            blocks=["总资产"], block_pages=[0])
        s.out_doc = {0: {"text": "Total assets"}}
        tools = agent.make_page_executors(s, _dummy_model())
        self.assertEqual([], tools["check_layout"](0)["issues"])

    def test_check_layout_legal_small_source_not_flagged(self):
        # Regression (fix 3): the flat 7pt readability floor flagged a legal 5.4pt
        # footnote, but the exporter's fit floor is min(block size, READABLE) so it
        # draws the block at its own (small) size — exporter-parity floors do not
        # report it as too_small.
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("注释", page=0, x0=0, y0=0, x1=60, y1=8, size=5.4)]],
            blocks=["注释"], block_pages=[0])
        s.out_doc = {0: {"text": "Notes"}}
        tools = agent.make_page_executors(s, _dummy_model())
        kinds = {i["kind"] for i in tools["check_layout"](0)["issues"]}
        self.assertNotIn("too_small", kinds)

    def test_check_layout_wrapped_table_cell_not_too_small(self):
        # A scanned grid cell that legitimately wraps below the 6pt readability
        # floor but above the exporter's 3pt hard floor is a physical-limits case,
        # not a defect — the old check flagged every such cell as too_small.
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("长期负债", page=0, x0=0, y0=0, x1=45, y1=10,
                                size=9, in_table=True, fit_width=45.0, fit_height=9.0)]],
            blocks=["长期负债"], block_pages=[0])
        s.out_doc = {0: {"text": "Long-term liabilities"}}
        tools = agent.make_page_executors(s, _dummy_model())
        kinds = {i["kind"] for i in tools["check_layout"](0)["issues"]}
        self.assertNotIn("too_small", kinds)

    def test_check_layout_ignores_other_column_blocks(self):
        # Regression (fix 4): on a two-column page the sibling at the same y on the
        # OTHER column must not be the "next block" (no x-overlap) — the old code
        # measured the gap to the nearest block on the whole page and flag-ged the
        # left-column paragraph as crowding.
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("The board reviewed the strategic options", page=0,
                                x0=20, y0=0, x1=150, y1=12, size=10),
                    pdfio.Block("右栏标题", page=0, x0=200, y0=11, x1=280, y1=23, size=10)]],
            blocks=["The board reviewed the strategic options", "右栏标题"],
            block_pages=[0])
        s.out_doc = {0: {"text": "The board reviewed the strategic options at length "
                                 "during the year was not yet public"},
                     1: {"text": "Right column"}}
        tools = agent.make_page_executors(s, _dummy_model())
        kinds = {i["kind"] for i in tools["check_layout"](0)["issues"]}
        self.assertNotIn("crowding", kinds)

    def test_check_layout_skips_vertical_labels(self):
        # An org-chart label is rotated by the exporter, so the horizontal fit
        # rules do not apply: a 8×30 label read vertically must never be measured
        # through the wrap path (per-character 5pt shards) or flagged.
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("党政办公室", page=0, x0=0, y0=0, x1=8, y1=30,
                                size=10, single_line=True)]],
            blocks=["党政办公室"], block_pages=[0])
        s.out_doc = {0: {"text": "Party and government office"}}
        tools = agent.make_page_executors(s, _dummy_model())
        self.assertEqual([], tools["check_layout"](0)["issues"])


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

    def test_translate_blocks_batches_and_writes(self):
        # The batch tool translates many blocks in ONE ``engine.translate_blocks``
        # call (the engine batches internally), writes each result to out_doc, and
        # skips numeric / code cells — a whole page in a fraction of the per-block
        # requests that a block-by-block loop would need.
        seen: list = []

        class _Eng:
            def __init__(self, model):
                self.model = model

            def translate_blocks(self, blocks, lang, **kw):
                seen.append(list(blocks))
                return type("R", (), {
                    "translated": [f"TR-{b}" for b in blocks], "errors": []})()

        s = self._state()
        with mock.patch.object(translator, "TranslationEngine", _Eng):
            tools = agent.make_page_executors(s, _dummy_model())
        res = tools["translate_blocks"](0)   # all translatable blocks on page 0
        self.assertTrue(res["ok"])
        self.assertEqual(2, res["count"])     # 总资产 + 总负债 (numeric cell skipped)
        self.assertEqual(1, len(seen))        # ONE batched request, not per-block
        self.assertEqual(["总资产", "总负债"], seen[0])
        self.assertEqual("TR-总资产", s.out_doc[0]["text"])
        self.assertEqual("TR-总负债", s.out_doc[2]["text"])
        self.assertNotIn(1, (s.out_doc or {}))     # numeric cell untouched
        self.assertEqual([], res["failed"])

    def test_translate_blocks_accepts_explicit_indices(self):
        seen: list = []

        class _Eng:
            def __init__(self, model):
                self.model = model

            def translate_blocks(self, blocks, lang, **kw):
                seen.append(list(blocks))
                return type("R", (), {
                    "translated": [f"TR-{b}" for b in blocks], "errors": []})()

        s = self._state()
        with mock.patch.object(translator, "TranslationEngine", _Eng):
            tools = agent.make_page_executors(s, _dummy_model())
        res = tools["translate_blocks"](0, indices=[0, 99, 1, 2])
        self.assertTrue(res["ok"])
        self.assertEqual(["总资产", "总负债"], seen[0])   # bogus 99 / numeric 1 dropped
        self.assertEqual(2, res["count"])

    def test_translate_blocks_nothing_to_translate_errors(self):
        class _Eng:
            def __init__(self, model):
                self.model = model

        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("12345", page=0, x0=0, y0=0, x1=50, y1=10)]],
            blocks=["12345"], block_pages=[0])
        with mock.patch.object(translator, "TranslationEngine", _Eng):
            tools = agent.make_page_executors(s, _dummy_model())
        res = tools["translate_blocks"](0)
        self.assertFalse(res["ok"])
        self.assertIn("没有可翻译的块", res["error"])

    def test_retranslate_blocks_batches_and_writes(self):
        # The self-check's batch re-translate: ONE request, bypasses the cache,
        # writes the fresh translations to out_doc (numeric cells skipped).
        seen: list = []

        def fake_batch(texts, lang):
            seen.append((list(texts), lang))
            return [f"RT-{t}" for t in texts]

        s = self._state()
        with mock.patch.object(translator, "make_retranslate_batch_fn",
                               lambda model, log=None: fake_batch):
            tools = agent.make_page_executors(s, _dummy_model())
        res = tools["retranslate_blocks"](0, indices=[0, 1, 2])
        self.assertTrue(res["ok"])
        self.assertEqual(["总资产", "总负债"], seen[0][0])      # numeric cell 1 dropped
        self.assertEqual("RT-总资产", s.out_doc[0]["text"])
        self.assertEqual("RT-总负债", s.out_doc[2]["text"])
        self.assertNotIn(1, (s.out_doc or {}))                   # numeric untouched
        self.assertEqual([], res["failed"])

    def test_retranslate_blocks_untouched_source_is_failed(self):
        # A block whose retranslation comes back as the source (model declined /
        # failed) is NOT written and is reported in ``failed`` (fail-closed).
        def fake_batch(texts, lang):
            return list(texts)   # model returns the source verbatim

        s = self._state()
        with mock.patch.object(translator, "make_retranslate_batch_fn",
                               lambda model, log=None: fake_batch):
            tools = agent.make_page_executors(s, _dummy_model())
        res = tools["retranslate_blocks"](0, indices=[0])
        self.assertTrue(res["ok"])
        self.assertEqual([0], res["failed"])
        self.assertNotIn(0, (s.out_doc or {}))

    def test_retranslate_blocks_nothing_to_translate_errors(self):
        s = agent.WorkflowState("a.pdf", "English")
        s.src_doc = pdfio.DocumentText(
            pages=[[pdfio.Block("12345", page=0, x0=0, y0=0, x1=50, y1=10)]],
            blocks=["12345"], block_pages=[0])
        with mock.patch.object(translator, "make_retranslate_batch_fn",
                               lambda model, log=None: lambda texts, lang: texts):
            tools = agent.make_page_executors(s, _dummy_model())
        res = tools["retranslate_blocks"](0)
        self.assertFalse(res["ok"])
        self.assertIn("没有可重译的块", res["error"])

    def test_detect_page_skew_asks_and_records(self):
        # A noticeable skew → the tool files a question, records the decision, and
        # reports it (low-risk: it never modifies the PDF).
        s = self._state()
        s.src_path = "a.pdf"
        asked = []

        def ah(question, options, target):
            asked.append((question, options, target))
            return {"value": "校正", "target": target}

        with mock.patch.object(pdfio, "detect_page_skew",
                               return_value={"page": 0, "skew_degrees": 1.2,
                                             "recommended": True, "reason": "倾斜"}):
            tools = agent.make_page_executors(s, _dummy_model(), answer_handler=ah)
            res = tools["detect_page_skew"](0)
        self.assertTrue(res["ok"])
        self.assertEqual(1.2, res["skew_degrees"])
        self.assertEqual("校正", res["decision"])
        self.assertTrue(asked and "几何校正" in asked[0][0])
        self.assertEqual(["校正", "忽略"], asked[0][1])
        self.assertIn("ocr_skew:0", s.user_decisions)
        self.assertTrue(s.user_decisions["ocr_skew:0"]["apply"])
        self.assertTrue(any(op.tool == "detect_page_skew" and op.user_confirmed
                            for op in s.ops))

    def test_detect_page_skew_clean_does_not_ask(self):
        s = self._state()
        s.src_path = "a.pdf"
        with mock.patch.object(pdfio, "detect_page_skew",
                               return_value={"page": 0, "skew_degrees": 0.1,
                                             "recommended": False, "reason": "平正"}):
            tools = agent.make_page_executors(s, _dummy_model(),
                                              answer_handler=lambda *a: None)
            res = tools["detect_page_skew"](0)
        self.assertTrue(res["ok"])
        self.assertEqual(0.1, res["skew_degrees"])
        self.assertFalse(res["recommended"])
        self.assertIsNone(res["decision"])     # no question was asked

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


class ReviewPageTaskPromptTest(unittest.TestCase):
    """The M4 review task is now parameterized: findings injected + read-only mode."""

    def test_default_still_mentions_review_and_fix_tools(self):
        from translate_app import prompts

        task = prompts.review_page_task(2)
        self.assertIn("复核", task)
        self.assertIn("set_text", task)
        self.assertIn("retranslate_block", task)

    def test_review_page_task_guides_batch_retranslate(self):
        # The review task should steer the model to batch re-translate several findings
        # (retranslate_blocks) instead of one retranslate_block per problem — the
        # scheme-1 analogue for the AI self-check.
        from translate_app import prompts

        task = prompts.review_page_task(2, auto_fix=True)
        self.assertIn("retranslate_blocks", task)
        self.assertIn("批量重译", task)

    def test_findings_are_injected_as_concrete_data(self):
        from translate_app import prompts

        findings = {"page": 2, "checks_requested": ["numbers"], "checks": {},
                    "issues": [{"check": "numbers", "index": 7, "source": "3.14 亿元",
                                "translation": "3.1 billion yuan",
                                "missing": ["4"], "extra": []}], "clean": False}
        task = prompts.review_page_task(2, findings=findings)
        self.assertIn("复核", task)
        self.assertIn("块#7", task)
        self.assertIn("3.14", task)
        self.assertIn("missing=['4']", task)

    def test_auto_fix_false_is_read_only(self):
        from translate_app import prompts

        task = prompts.review_page_task(1, auto_fix=False)
        self.assertIn("只读复核", task)
        self.assertIn("不要修改任何译文", task)


class LlmInterpretTest(unittest.TestCase):
    """The M3 special-page answer is AI-interpreted, fail-closed to the matcher."""

    def test_make_llm_interpret_classifies_answer(self):
        seen: list = []
        queue = [_FakeResp(_FakeMsg(content="translate"))]
        client = _FakeClient(queue, seen)
        with mock.patch.object(translator, "OpenAI", lambda **_k: client):
            interpret = agent.make_llm_interpret(_vision_model())
        self.assertIsNotNone(interpret)
        self.assertEqual("translate", interpret("用OCR识别一下", "scan"))
        # The model got the classification prompt as a translation-side, temperature-0 call.
        call = seen[0]
        self.assertEqual(0.0, call["temperature"])
        self.assertIn("特殊页", call["messages"][0]["content"])

    def test_make_llm_interpret_fails_closed_to_matcher(self):
        client = _FakeClient([_FakeResp(_FakeMsg(content=""))], [])

        def boom(**kw):
            raise RuntimeError("net down")

        client.chat.completions.create = boom
        with mock.patch.object(translator, "OpenAI", lambda **_k: client):
            interpret = agent.make_llm_interpret(_vision_model())
        # A network error never raises / hangs — it degrades to the flexible matcher.
        self.assertEqual("keep", interpret("保留原文", "scan"))
        self.assertEqual("translate", interpret("OCR并翻译", "scan"))

    def test_make_llm_interpret_no_client_returns_none(self):
        with mock.patch.object(translator, "OpenAI", side_effect=RuntimeError("no cfg")):
            self.assertIsNone(agent.make_llm_interpret(_vision_model()))


class TranslationPromptScriptTest(unittest.TestCase):
    """Russian / Japanese / Korean are now target languages."""

    def test_is_cjk_language_classes_scripts(self):
        from translate_app import prompts

        self.assertTrue(prompts._is_cjk_language("Japanese"))
        self.assertTrue(prompts._is_cjk_language("Korean"))
        self.assertTrue(prompts._is_cjk_language("Simplified Chinese"))
        self.assertFalse(prompts._is_cjk_language("Russian"))   # Cyrillic, not CJK

    def test_example_output_in_target_script(self):
        import re
        from translate_app import prompts

        # The batch-translation example output must be in the target script, so a
        # Russian / Japanese / Korean translation is anchored to that script, not English.
        for lang, marker in (("Russian", "й"), ("Japanese", "押"), ("Korean", "저장")):
            p = prompts.translation_system_prompt(lang)
            m = re.search(r"Output:\n\[1\]\n(.+?)\n\[2\]\n(.+)\Z", p, re.S)
            self.assertIsNotNone(m, lang)
            self.assertIn(marker, m.group(1) + (m.group(2) or ""), lang)


if __name__ == "__main__":
    unittest.main()
