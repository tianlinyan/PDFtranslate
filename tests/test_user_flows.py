"""U1: user-custom flows — requirement→FlowSpec, build, binding consistency, persistence."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from translate_app import agent, translator
from translate_app.agent import flow_steps as fs, user_flows as uf
from translate_app.settings import ModelConfig


class CompileFromUserTest(unittest.TestCase):
    def test_parses_checks_scope_kept_and_readonly(self):
        spec = agent.compile_from_user("自检只查数字和表格，第3到第8页，把保留页也算进去，不修改")
        self.assertEqual("self_check_page", spec.base)
        self.assertEqual({"numbers", "table"}, set(spec.checks))
        self.assertEqual([2, 3, 4, 5, 6, 7], spec.scope)          # pages 3..8, 0-based
        self.assertTrue(spec.include_kept)
        self.assertFalse(spec.auto_fix)                            # read-only

    def test_single_page_scope(self):
        spec = agent.compile_from_user("检查第5页")
        self.assertEqual([4], spec.scope)

    def test_auto_fix_default_when_not_specified(self):
        self.assertIsNone(agent.compile_from_user("自检").auto_fix)

    def test_remap_base_export_and_retranslate(self):
        self.assertEqual("export", agent.compile_from_user("重新导出").base)
        self.assertEqual("translate_page", agent.compile_from_user("重译第2页").base)

    def test_unrecognised_keeps_defaults(self):
        spec = agent.compile_from_user("你好世界")
        self.assertEqual("self_check_page", spec.base)
        self.assertIsNone(spec.checks)
        self.assertIsNone(spec.scope)

    def test_aliases_do_not_match_unrelated_words(self):
        # Bare single-char aliases ("数"/"表") were removed: 数据/代表 etc. must NOT
        # be misread as a numbers/table audit.
        self.assertIsNone(agent.compile_from_user("自检数据完整性").checks)
        self.assertIsNone(agent.compile_from_user("检查数据的准确性").checks)
        self.assertEqual(["table"], agent.compile_from_user("只查表格").checks)
        self.assertEqual(["numbers"], agent.compile_from_user("只查数字").checks)

    def test_ai_slot_filler_builds_spec(self):
        # AI-driven: an injected LLM interprets the requirement into FlowSpec fields,
        # replacing hardcoded keyword rules.
        seen: list[str] = []

        def llm(req):
            seen.append(req)
            return {"base": "self_check_page", "checks": ["numbers", "table"],
                    "scope": [2, 3, 4], "include_kept": True, "auto_fix": False}

        spec = agent.compile_from_user("自检只查数字和表格，第3到第8页，把保留页也算，不修改",
                                       llm=llm)
        self.assertEqual(["自检只查数字和表格，第3到第8页，把保留页也算，不修改"], seen)
        self.assertEqual({"numbers", "table"}, set(spec.checks))
        self.assertEqual([2, 3, 4], spec.scope)
        self.assertTrue(spec.include_kept)
        self.assertFalse(spec.auto_fix)

    def test_ai_slot_filler_falls_back_on_error(self):
        def llm(_req):
            raise RuntimeError("model down")
        # A failing LLM must not crash — it degrades to the default spec (no AI fields).
        spec = agent.compile_from_user("自检第3页", llm=llm)
        self.assertEqual("self_check_page", spec.base)
        self.assertIsNone(spec.scope)
        self.assertIsNone(spec.checks)

    def test_ai_slot_filler_unknown_base_falls_back(self):
        # A bad/unknown base must not crash or produce a flow nobody can run.
        spec = agent.compile_from_user("自检", llm=lambda _r: {"base": "nope"})
        self.assertEqual("self_check_page", spec.base)
        self.assertIn(spec.base, agent.STANDARD_FLOWS)


class BuildFlowTest(unittest.TestCase):
    def test_single_page_self_check_overrides_knobs(self):
        spec = agent.FlowSpec(base="self_check_page", checks=["numbers"],
                              auto_fix=False, page=4)
        flow = agent.build_flow(spec)
        self.assertEqual(["numbers"], flow.params["checks"])
        self.assertFalse(flow.params["auto_fix"])
        self.assertEqual(4, flow.params["page"])

    def test_multi_page_self_check_wraps_in_foreach(self):
        spec = agent.FlowSpec(base="self_check_page", checks=["table"], scope=[2, 3, 4])
        flow = agent.build_flow(spec)
        self.assertEqual(["foreach_page"], [s.kind for s in flow.steps])
        self.assertEqual([2, 3, 4], flow.params["pages"])

    def test_unknown_base_raises(self):
        with self.assertRaises(ValueError):
            agent.build_flow(agent.FlowSpec(base="nope"))


class ToolBindingConsistencyTest(unittest.TestCase):
    def test_standard_flows_reference_only_bound_or_deterministic_tools(self):
        # The "先绑定后暴露" gate: every flow's ToolStep tools must be in the bound
        # agent registry (AGENT_TOOLS aligns with the bindings) or a known
        # deterministic tool.  No flow may reference a tool the pipeline would
        # answer "unknown tool".
        available = {t.name for t in agent.AGENT_TOOLS}
        for name, flow in agent.STANDARD_FLOWS.items():
            unbound = agent.validate_flow_tools(flow, available)
            self.assertEqual([], unbound, f"{name} references unbound tools: {unbound}")

    def test_validate_flags_unknown_tool(self):
        flow = fs.Flow(name="bad", description="", steps=[fs.ToolStep("nope")])
        self.assertEqual(["nope"], agent.validate_flow_tools(flow, {"read_page"}))


class PersistenceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get(uf._FLOW_DIR_ENV)
        os.environ[uf._FLOW_DIR_ENV] = self._tmp.name
        # Keep the in-memory registry out of the way so we exercise a clean round-trip.
        self._saved = dict(uf.USER_FLOW_SPECS)
        uf.USER_FLOW_SPECS.clear()

    def tearDown(self):
        uf.USER_FLOW_SPECS.clear()
        uf.USER_FLOW_SPECS.update(self._saved)
        if self._old is None:
            os.environ.pop(uf._FLOW_DIR_ENV, None)
        else:
            os.environ[uf._FLOW_DIR_ENV] = self._old
        self._tmp.cleanup()

    def test_save_load_roundtrip_returns_built_flow(self):
        spec = agent.FlowSpec(base="self_check_page", checks=["numbers"],
                              scope=[0, 1], include_kept=True)
        agent.save_flow_spec("my_audit", spec)
        # A fresh load (from disk) reconstructs the spec and compiles a runnable flow.
        loaded = agent.load_user_flow_specs()
        self.assertIn("my_audit", loaded)
        flow = agent.get_user_flow("my_audit")
        self.assertEqual(["numbers"], flow.params["checks"])
        self.assertEqual([0, 1], flow.params["pages"])
        # The spec was written to the env-gated dir as JSON.
        self.assertTrue(list(Path(self._tmp.name).glob("*.json")))

    def test_load_ignores_corrupt_file(self):
        (Path(self._tmp.name) / "broken.json").write_text("{not json", encoding="utf-8")
        self.assertEqual({}, agent.load_user_flow_specs())   # no crash, skipped


class MemoryOnlyPersistenceTest(unittest.TestCase):
    """Without ``PDFTRANSLATE_FLOWS_DIR``, promotion is memory-only (no implicit disk write)."""

    def setUp(self):
        self._old = os.environ.pop(uf._FLOW_DIR_ENV, None)
        self._saved = dict(uf.USER_FLOW_SPECS)
        uf.USER_FLOW_SPECS.clear()

    def tearDown(self):
        uf.USER_FLOW_SPECS.clear()
        uf.USER_FLOW_SPECS.update(self._saved)
        if self._old is not None:
            os.environ[uf._FLOW_DIR_ENV] = self._old

    def test_save_is_memory_only_and_load_returns_empty(self):
        spec = agent.FlowSpec(base="self_check_page", checks=["numbers"])
        agent.save_flow_spec("mem_only", spec)
        self.assertIn("mem_only", uf.USER_FLOW_SPECS)
        self.assertEqual({}, agent.load_user_flow_specs())   # no disk read when unconfigured
        self.assertIsNotNone(agent.get_user_flow("mem_only"))   # still resolvable in memory


class FlowCompilerTest(unittest.TestCase):
    """AI slot-filling (``make_llm_flow_compiler``) — Path A's flexible branch."""

    def test_parse_flow_json_strips_fences_and_prose(self):
        self.assertEqual(
            {"base": "self_check_page", "checks": ["numbers"]},
            uf._parse_flow_json('```json\n{"base":"self_check_page","checks":["numbers"]}\n```'))
        self.assertEqual({"scope": [2, 3]}, uf._parse_flow_json('好的：{"scope": [2, 3]}'))
        self.assertEqual({}, uf._parse_flow_json("no json here"))
        self.assertEqual({}, uf._parse_flow_json('["not", "an", "object"]'))

    def test_make_llm_flow_compiler_fills_spec(self):
        class _Msg:
            content = '{"base": "self_check_page", "checks": ["numbers", "table"], "scope": [2, 3]}'

        class _Resp:
            choices = [type("_C", (), {"message": _Msg()})()]

        class _Client:
            def __init__(self):
                self.calls = []

            @property
            def chat(self):
                return self

            @property
            def completions(self):
                return self

            def create(self, **kw):
                self.calls.append(kw)
                return _Resp()

        client = _Client()
        model = ModelConfig(id="m", name="m", type="openai",
                            endpoint="http://127.0.0.1:9/v1", model="mock")
        with mock.patch.object(translator, "OpenAI", lambda **_k: client):
            compiler = uf.make_llm_flow_compiler(model)
        self.assertIsNotNone(compiler)
        data = compiler("自检第3到第4页只查数字和表格")
        self.assertEqual(["numbers", "table"], data["checks"])
        self.assertEqual([2, 3], data["scope"])
        # A temperature-0 translation-side call.
        self.assertEqual(0.0, client.calls[0]["temperature"])

    def test_make_llm_flow_compiler_fails_closed(self):
        class _Client:
            @property
            def chat(self):
                return self

            @property
            def completions(self):
                return self

            def create(self, **kw):
                raise RuntimeError("net down")

        model = ModelConfig(id="m", name="m", type="openai",
                            endpoint="http://127.0.0.1:9/v1", model="mock")
        with mock.patch.object(translator, "OpenAI", lambda **_k: _Client()):
            compiler = uf.make_llm_flow_compiler(model)
        self.assertIsNotNone(compiler)
        self.assertEqual({}, compiler("自检只查数字"))   # network error → defaults

    def test_make_llm_flow_compiler_none_without_client(self):
        model = ModelConfig(id="m", name="m", type="openai",
                            endpoint="http://127.0.0.1:9/v1", model="mock")
        with mock.patch.object(translator, "OpenAI", side_effect=RuntimeError("no cfg")):
            self.assertIsNone(uf.make_llm_flow_compiler(model))


if __name__ == "__main__":
    unittest.main()
