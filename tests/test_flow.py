"""Phase 2: the Flow skeleton — step model, run_flow executor and the registry.

The executor is deliberately decoupled from ``WorkflowState`` (the caller binds a
tool map / agent runner / user channel), so every step type is unit-testable here
with injected fakes — no model / network / Qt needed.
"""
from __future__ import annotations

import unittest

from translate_app import agent
from translate_app.agent import flow_steps as fs


class StepModelTest(unittest.TestCase):
    def test_default_tool_label_is_tool_name(self):
        s = fs.ToolStep("audit_page", {"page": 1})
        self.assertEqual("audit_page", s.label)
        self.assertEqual("tool", s.kind)

    def test_step_kinds(self):
        self.assertEqual({"tool", "agent", "user", "loop", "if", "foreach_page"},
                         set(fs.STEP_KINDS))

    def test_resolve_preserves_type_for_single_placeholder(self):
        self.assertEqual(3, fs._resolve("{{page}}", {"page": 3}))
        self.assertEqual("translate 3", fs._resolve("translate {{page}}", {"page": 3}))


class RunFlowTest(unittest.TestCase):
    def _flow(self, *steps, **kw):
        return fs.Flow(name=kw.get("name", "t"), description="", steps=list(steps),
                       params=kw.get("params", {}))

    def test_tool_step_executes_in_order(self):
        calls: list = []
        tools = {
            "a": lambda v: calls.append(("a", v)) or {"r": v},
            "b": lambda: calls.append(("b",)) or {"r": "b"},
        }
        rs = agent.run_flow(self._flow(fs.ToolStep("a", {"v": 1}), fs.ToolStep("b")),
                            tools=tools)
        self.assertTrue(rs.ok)
        self.assertEqual([("a", 1), ("b",)], calls)
        self.assertEqual(["a", "b"], rs.applied)
        self.assertEqual({"r": 1}, rs.result["a"])

    def test_tool_step_logs_start_and_elapsed(self):
        # A deterministic ``ToolStep`` logs its name when it starts and the elapsed
        # time when it completes (so an audit_page / export step is visible).
        logs: list = []
        flow = self._flow(fs.ToolStep("demo", {"v": 1}))
        rs = agent.run_flow(flow, tools={"demo": lambda v: {"r": v}}, log=logs.append)
        self.assertTrue(rs.ok)
        self.assertTrue(any("工具开始：demo" in m for m in logs), logs)
        self.assertTrue(any("工具完成：demo" in m and "用时" in m for m in logs), logs)

    def test_agent_step_delegates_and_resolves_task_and_page(self):
        received: dict = {}

        def run_agent(*, task, page, max_steps, image):
            received.update(task=task, page=page, max_steps=max_steps, image=image)
            return {"ok": True}

        flow = self._flow(fs.AgentStep(task="翻译第 {{page}} 页", page="{{page}}"))
        rs = agent.run_flow(flow, run_agent=run_agent, params={"page": 2})
        self.assertEqual("翻译第 2 页", received["task"])
        self.assertEqual(2, received["page"])   # type-preserving placeholder
        self.assertEqual({"ok": True}, rs.result["agent:2"])

    def test_user_step_delegates(self):
        asked: list = []

        def ask(question, options, target):
            asked.append((question, options, target))
            return "keep"

        rs = agent.run_flow(
            self._flow(fs.UserStep("如何处理？", ["a", "b"], "page:0")), ask=ask)
        self.assertEqual([("如何处理？", ["a", "b"], "page:0")], asked)
        self.assertEqual("keep", rs.result["user:page:0"])

    def test_loop_repeats_until_condition(self):
        state = {"n": 0}

        def inc():
            state["n"] += 1
            return {"n": state["n"]}

        rs = agent.run_flow(
            self._flow(fs.LoopStep(
                until=lambda rs: rs.result.get("inc", {}).get("n", 0) >= 3,
                max_iter=10, body=[fs.ToolStep("inc")])),
            tools={"inc": inc})
        self.assertTrue(rs.ok)
        self.assertEqual(3, state["n"])
        self.assertEqual(3, len(rs.applied))

    def test_loop_respects_max_iter(self):
        state = {"n": 0}

        def inc():
            state["n"] += 1
            return {"n": state["n"]}

        rs = agent.run_flow(
            self._flow(fs.LoopStep(until=lambda rs: False, max_iter=3,
                                   body=[fs.ToolStep("inc")])),
            tools={"inc": inc})
        self.assertTrue(rs.ok)
        self.assertEqual(3, state["n"])

    def test_if_branches_on_condition(self):
        calls: list = []

        def done(x):
            calls.append(x)

        flow = self._flow(fs.IfStep(lambda rs: True,
                                    then=[fs.ToolStep("done", {"x": 1})],
                                    else_=[fs.ToolStep("done", {"x": 2})]))
        agent.run_flow(flow, tools={"done": done})
        self.assertEqual([1], calls)

        flow2 = self._flow(fs.IfStep(lambda rs: False,
                                     then=[fs.ToolStep("done", {"x": 1})],
                                     else_=[fs.ToolStep("done", {"x": 2})]))
        agent.run_flow(flow2, tools={"done": done})
        self.assertEqual([1, 2], calls)

    def test_budget_stops_and_fails_closed(self):
        tools = {"a": lambda: None}
        rs = agent.run_flow(
            self._flow(fs.ToolStep("a"), fs.ToolStep("a"), fs.ToolStep("a")),
            tools=tools, max_steps=2)
        self.assertFalse(rs.ok)
        self.assertIn("预算", rs.error)
        self.assertEqual(2, len(rs.applied))   # the 3rd step triggered the budget

    def test_cancel_raises_control_signal(self):
        calls: list = []

        def cancel():
            return len(calls) >= 1

        flow = self._flow(fs.ToolStep("a"), fs.ToolStep("a"))
        with self.assertRaises(agent.FlowCancelled):
            agent.run_flow(flow, tools={"a": lambda: calls.append(1)}, cancel=cancel)

    def test_unknown_tool_fails_closed(self):
        rs = agent.run_flow(self._flow(fs.ToolStep("nope")), tools={})
        self.assertFalse(rs.ok)
        self.assertIn("unknown tool", rs.error)

    def test_tool_step_re_raises_translation_cancelled(self):
        # A deterministic ToolStep that raises the translation engine's cancellation
        # signal must propagate it as a control signal, NOT swallow it as a tool error.
        from translate_app.translator import TranslationCancelled

        def boom():
            raise TranslationCancelled()

        with self.assertRaises(TranslationCancelled):
            agent.run_flow(self._flow(fs.ToolStep("boom")), tools={"boom": boom})

    def test_control_signal_hierarchy(self):
        from translate_app.control import ControlSignal
        from translate_app.translator import TranslationCancelled

        self.assertTrue(issubclass(fs.FlowCancelled, ControlSignal))
        self.assertTrue(issubclass(TranslationCancelled, ControlSignal))


class RegistryTest(unittest.TestCase):
    def test_standard_flows_are_registered(self):
        self.assertEqual({"preprocess", "translate_doc", "translate_page", "translate_normal",
                          "special_page", "special_pages", "self_check_page",
                          "ai_self_check", "export"},
                         set(agent.STANDARD_FLOWS))

    def test_translate_doc_declares_phase_order(self):
        # The top-level phase ORDER is data (declarative), which ``DocumentSession.run``
        # dispatches over — not a hardcoded call sequence.
        phases = agent.STANDARD_FLOWS["translate_doc"].scope["phases"]
        # The review/self-check is decoupled: the standard translate flow ends at
        # "completed" and reports; a review is triggered on demand (custom flow).
        self.assertEqual(["preprocess", "translate_normal", "special_pages", "completed"],
                         phases)

    def test_report_hook_called_per_page(self):
        # A ``ForEachPage`` reports per-page progress to an orchestrator.
        seen: list = []

        def read(page):
            return {"page": page}

        flow = fs.Flow(name="f", description="", params={"pages": [0, 3]},
                       steps=[fs.ForEachPage(pages="{{pages}}",
                                             body=[fs.ToolStep("read", {"page": "{{page}}"})])])
        agent.run_flow(flow, tools={"read": read},
                       report=lambda _phase, done, total, page: seen.append((done, total, page)))
        self.assertEqual([(1, 2, 0), (2, 2, 3)], seen)

    def test_foreach_page_iterates_and_binds_page(self):
        seen: list = []

        def read(page):
            seen.append(page)
            return {"page": page}

        flow = fs.Flow(name="loop", description="",
                       params={"pages": [0, 2, 5]},
                       steps=[fs.ForEachPage(pages="{{pages}}",
                                             body=[fs.ToolStep("read", {"page": "{{page}}"})])])
        rs = agent.run_flow(flow, tools={"read": read})
        self.assertTrue(rs.ok)
        self.assertEqual([0, 2, 5], seen)
        self.assertEqual(5, rs.result["read"]["page"])   # last iteration's result

    def test_user_step_callable_question_returns_tuple(self):
        asked: list = []

        def q(rs, params):
            return ("第几页？", ["a", "b"])

        def ask(question, options, target):
            asked.append((question, options, target))
            return {"value": "a"}

        flow = fs.Flow(name="t", description="",
                       params={"page": 3},
                       steps=[fs.UserStep(question=q, target="page:{{page}}")])
        rs = agent.run_flow(flow, ask=ask)
        self.assertEqual([("第几页？", ["a", "b"], "page:3")], asked)
        self.assertEqual({"value": "a"}, rs.result.get("user:page:3"))

    def test_special_page_flow_asks_and_records_answer(self):
        # P4 unit flow is now the negotiation UserStep (the decision + execution live in
        # the ``DocumentSession`` phase, which injects ``interpret``).
        asked: list = []

        def ask(question, options, target):
            asked.append((question, options, target))
            return {"value": "OCR并翻译", "target": target}

        rs = agent.run_flow(
            agent.STANDARD_FLOWS["special_page"],
            ask=ask, run_agent=lambda *a, **kw: self.fail("flow no longer runs the agent"),
            params={"page": 4, "kind": "scan"})
        self.assertTrue(rs.ok)
        self.assertEqual([("page:4",)], [(a[2],) for a in asked])
        self.assertEqual("OCR并翻译", rs.result["user:page:4"]["value"])

    def test_interpret_decision_maps_free_text(self):
        # The special-page decision is a flexible matcher (an injected AI ``interpret``
        # can override it): buttons, synonyms and free text map to translate/keep/skip.
        cases = {"OCR并翻译": "translate", "翻译一下这页": "translate", "用OCR识别": "translate",
                 "保留原文": "keep", "不动，保留": "keep", "不翻译": "keep",
                 "跳过": "skip", "skip this page": "skip"}
        for answer, want in cases.items():
            self.assertEqual(want, agent.interpret_decision(answer), answer)

    def test_interpret_decision_keep_but_translate_captions(self):
        # "保留公式/图表并翻译说明/图注" -> translate (the engine keeps structural
        # blocks verbatim, so translate = translate the prose/caption only).  This must
        # NOT be swallowed by the lone "保留" keep-match.
        cases = {"保留公式并翻译说明": "translate",
                 "保留图表并翻译图注": "translate",
                 "保留公式并译文字": "translate",
                 "保留原文": "keep",
                 "翻译图注": "translate",
                 "整页翻译": "translate",
                 "跳过": "skip"}
        for answer, want in cases.items():
            self.assertEqual(want, agent.interpret_decision(answer), answer)

    def test_self_check_page_step_kinds_in_order(self):
        # The audit→fix→re-audit rounds live inside ONE ``LoopStep`` (until clean, up to
        # ``max_iter``): a fix that leaves residual issues is fixed again on the next
        # round rather than abandoned after a single fix pass (the old structure ran one
        # fix outside the loop and only re-audited).
        flow = agent.STANDARD_FLOWS["self_check_page"]
        self.assertEqual(["loop"], [s.kind for s in flow.steps])
        loop = flow.steps[0]
        self.assertIsInstance(loop, fs.LoopStep)
        self.assertEqual(["tool", "if"], [s.kind for s in loop.body])
        self.assertEqual("audit_page", loop.body[0].tool)
        self.assertEqual("agent", loop.body[1].then[0].kind)

    def test_self_check_page_runs_audit_then_fix_then_reaudit(self):
        # A realistic run of the registered P6 flow with injected fakes: the first
        # audit finds issues → the agent fix pass runs → the re-audit is clean → the
        # loop ends, without ever exhausting the budget.
        state = {"audits": 0}

        def audit_page(page, checks=None):
            state["audits"] += 1
            if state["audits"] == 1:   # first audit finds a finding
                return {"page": page, "issues": [{"check": "missing", "index": 5}],
                        "clean": False}
            return {"page": page, "issues": [], "clean": True}

        def run_agent(*, task, page, **kw):
            return {"ok": True, "page": page}

        rs = agent.run_flow(agent.STANDARD_FLOWS["self_check_page"],
                            tools={"audit_page": audit_page}, run_agent=run_agent,
                            params={"page": 3})
        self.assertTrue(rs.ok)
        self.assertEqual(2, state["audits"])            # initial + re-audit
        self.assertIn("agent:3", rs.applied)            # the AI fix pass ran once
        self.assertEqual("agent:3", rs.applied[1])

    def test_self_check_page_retries_fix_until_clean(self):
        # The fix step is INSIDE the ``LoopStep``: a fix that leaves residual issues
        # is fixed again on the next round (up to ``max_iter``) rather than abandoned
        # after a single pass (the old structure only re-audited without re-fixing).
        state = {"audits": 0}

        def audit_page(page, checks=None):
            state["audits"] += 1
            if state["audits"] <= 2:   # first two audits still find problems
                return {"page": page, "issues": [{"check": "missing", "index": state["audits"]}],
                        "clean": False}
            return {"page": page, "issues": [], "clean": True}

        def run_agent(*, task, page, **kw):
            return {"ok": True, "page": page}

        rs = agent.run_flow(agent.STANDARD_FLOWS["self_check_page"],
                            tools={"audit_page": audit_page}, run_agent=run_agent,
                            params={"page": 3})
        self.assertTrue(rs.ok)
        # audit(dirty)→fix→audit(dirty)→fix→audit(clean): 3 audits, 2 fix passes.
        self.assertEqual(3, state["audits"])
        self.assertEqual(2, rs.applied.count("agent:3"))


if __name__ == "__main__":
    unittest.main()
