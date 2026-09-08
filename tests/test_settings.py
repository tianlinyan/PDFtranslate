"""Tests for model configuration parsing and validation."""
import os
import unittest

from translate_app.settings import ModelConfig, substitute_env


class SettingsTest(unittest.TestCase):
    def test_client_kwargs_strips_chat_completions(self):
        m = ModelConfig(
            id="m", name="m", type="openai",
            endpoint="http://127.0.0.1:9/v1/chat/completions", model="mod",
        )
        kwargs = m.client_kwargs()
        self.assertEqual(kwargs["base_url"], "http://127.0.0.1:9/v1")
        self.assertEqual(kwargs["api_key"], "not-needed")

    def test_unresolved_env_not_sent_as_key(self):
        os.environ.pop("PDFT_TEST_KEY", None)
        m = ModelConfig(
            id="m", name="m", type="openai",
            endpoint="http://x/v1/chat/completions", model="mod",
            api_key="${PDFT_TEST_KEY}",
        )
        self.assertIn("环境变量未设置", "\n".join(m.validate()))
        # Even though api_key is set to a placeholder, the client must not send it.
        self.assertEqual(m.client_kwargs()["api_key"], "not-needed")

    def test_resolved_env_used_as_key(self):
        os.environ["PDFT_TEST_KEY"] = "secret-123"
        m = ModelConfig(
            id="m", name="m", type="openai",
            endpoint="http://x/v1", model="mod", api_key="${PDFT_TEST_KEY}",
        )
        self.assertEqual(m.validate(), [])
        self.assertEqual(m.client_kwargs()["api_key"], "secret-123")

    def test_reserved_keys_not_overridden_by_extra(self):
        m = ModelConfig(
            id="m", name="m", type="openai",
            endpoint="http://x/v1", model="mod", api_key="k",
        )
        m.extra = {"base_url": "http://evil", "api_key": "bad", "timeout": 30}
        kwargs = m.client_kwargs()
        self.assertEqual(kwargs["base_url"], "http://x/v1")
        self.assertEqual(kwargs["api_key"], "k")
        self.assertEqual(kwargs["timeout"], 30)

    def test_substitute_env_missing_keeps_placeholder(self):
        os.environ.pop("PDFT_MISSING_VAR", None)
        self.assertEqual(substitute_env("${PDFT_MISSING_VAR}"), "${PDFT_MISSING_VAR}")
        os.environ["PDFT_MISSING_VAR"] = "ok"
        self.assertEqual(substitute_env("${PDFT_MISSING_VAR}"), "ok")

    def test_validate_missing_endpoint_model(self):
        m = ModelConfig(id="m", name="m", type="openai", endpoint="", model="")
        issues = "\n".join(m.validate())
        self.assertIn("缺少 endpoint", issues)
        self.assertIn("缺少 model", issues)

    def test_new_request_fields_parsed_with_defaults(self):
        m = ModelConfig.from_dict(
            {"id": "m", "endpoint": "http://x/v1", "model": "mod"}
        )
        self.assertIsNone(m.temperature)
        self.assertIsNone(m.max_tokens)
        self.assertEqual(m.concurrency, 1)
        self.assertEqual(m.batch_size, 4000)

        m2 = ModelConfig.from_dict(
            {
                "id": "m", "endpoint": "http://x/v1", "model": "mod",
                "temperature": 0.7, "max_tokens": 1234, "concurrency": 2,
                "batch_size": 8000,
            }
        )
        self.assertEqual(m2.temperature, 0.7)
        self.assertEqual(m2.max_tokens, 1234)
        self.assertEqual(m2.concurrency, 2)
        self.assertEqual(m2.batch_size, 8000)
        # Known keys must not leak into ``extra``.
        self.assertNotIn("temperature", m2.extra)
        self.assertNotIn("concurrency", m2.extra)
        self.assertNotIn("batch_size", m2.extra)

    def test_page_concurrency_parsed_with_default(self):
        # ``page_concurrency`` is the opt-in parallel-page knob for the agent path;
        # it defaults to 1 (sequential) and must not leak into ``extra``.
        m = ModelConfig.from_dict({"id": "m", "endpoint": "http://x/v1", "model": "mod"})
        self.assertEqual(m.page_concurrency, 1)
        m2 = ModelConfig.from_dict(
            {"id": "m", "endpoint": "http://x/v1", "model": "mod", "page_concurrency": 4}
        )
        self.assertEqual(m2.page_concurrency, 4)
        self.assertNotIn("page_concurrency", m2.extra)

    def test_interaction_params_defaults_and_override(self):
        # The AI-interaction (agent) config is a *separate* set from translation:
        # it defaults to reasoning_effort=medium / temperature=0.6 so the
        # orchestrator explores a little, while translation keeps models.json's
        # own (lower, deterministic) values.
        m = ModelConfig.from_dict({"id": "m", "endpoint": "http://x/v1", "model": "mod"})
        self.assertEqual(0.6, m.interaction_temperature)
        self.assertEqual("medium", m.interaction_reasoning_effort)
        self.assertEqual({"reasoning_effort": "medium"}, m.interaction_request_params())
        self.assertEqual({}, m.request_params())   # translation sends nothing unless set

        m2 = ModelConfig.from_dict({
            "id": "m", "endpoint": "http://x/v1", "model": "mod",
            "interaction_temperature": 0.9, "interaction_reasoning_effort": "high",
        })
        self.assertEqual(0.9, m2.interaction_temperature)
        self.assertEqual("high", m2.interaction_reasoning_effort)
        self.assertEqual({"reasoning_effort": "high"}, m2.interaction_request_params())
        # Known interaction keys do not leak into ``extra``.
        self.assertNotIn("interaction_temperature", m2.extra)
        self.assertNotIn("interaction_reasoning_effort", m2.extra)

    def test_client_kwargs_default_timeout(self):
        m = ModelConfig(id="m", name="m", type="openai",
                        endpoint="http://x/v1", model="mod")
        self.assertEqual(m.client_kwargs()["timeout"], 300.0)

    def test_endpoint_warnings_for_missing_chat_completions_suffix(self):
        m = ModelConfig(id="m", name="m", type="openai",
                        endpoint="http://host:8888", model="mod")
        warns = m.endpoint_warnings()
        self.assertEqual(len(warns), 1)
        self.assertIn("http://host:8888/chat/completions", warns[0])

    def test_endpoint_warnings_empty_for_full_chat_completions_url(self):
        m = ModelConfig(id="m", name="m", type="openai",
                        endpoint="http://host:8888/v1/chat/completions", model="mod")
        self.assertEqual(m.endpoint_warnings(), [])

    def test_endpoint_warnings_empty_when_endpoint_blank(self):
        m = ModelConfig(id="m", name="m", type="openai", endpoint="", model="mod")
        self.assertEqual(m.endpoint_warnings(), [])

    def test_enable_thinking_sent_in_both_params(self):
        m = ModelConfig.from_dict({
            "id": "q", "name": "q", "type": "llama-server",
            "endpoint": "http://x/v1/chat/completions", "model": "qwen3.8-27b",
            "reasoning_effort": "low", "enable_thinking": False,
        })
        self.assertFalse(m.enable_thinking)
        # The toggle is a first-class field (not dumped into ``extra``) and is carried
        # by BOTH the translation and interaction parameter sets.
        self.assertEqual(m.request_params()["enable_thinking"], False)
        self.assertIn("enable_thinking", m.interaction_request_params())
        self.assertIn("reasoning_effort", m.request_params())
        self.assertEqual(m.extra, {})

    def test_enable_thinking_absent_omits(self):
        m = ModelConfig.from_dict({
            "id": "q", "name": "q", "type": "llama-server",
            "endpoint": "http://x/v1", "model": "q", "reasoning_effort": "low",
        })
        self.assertIsNone(m.enable_thinking)
        self.assertNotIn("enable_thinking", m.request_params())
        self.assertNotIn("enable_thinking", m.interaction_request_params())


class PrefsTest(unittest.TestCase):
    """``load_prefs`` / ``save_prefs``: atomic write, failure reason, corrupt tolerance.

    The GUI hard-exits the process on window close, so a plain overwrite could leave a
    truncated ``prefs.json`` behind — and a silently lost preference is invisible until
    the next launch.  Both had zero coverage.
    """

    def setUp(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        from translate_app import settings

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "prefs.json"
        patcher = mock.patch.object(settings, "APP_PREFS_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.settings = settings

    def test_roundtrip(self):
        self.assertIsNone(self.settings.save_prefs({"language": "English"}))
        self.assertEqual({"language": "English"}, self.settings.load_prefs())

    def test_corrupt_file_reads_as_empty(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual({}, self.settings.load_prefs())

    def test_failure_returns_a_reason_and_leaves_no_temp_file(self):
        from unittest import mock

        with mock.patch.object(self.settings.os, "replace",
                               side_effect=OSError("disk full")):
            reason = self.settings.save_prefs({"language": "German"})
        self.assertIsNotNone(reason)
        self.assertIn("disk full", reason)
        self.assertEqual([], list(self.path.parent.glob("*.tmp")))

    def test_write_is_atomic_and_overwrites(self):
        self.settings.save_prefs({"a": 1})
        self.settings.save_prefs({"a": 2})
        self.assertEqual({"a": 2}, self.settings.load_prefs())
        self.assertEqual([], list(self.path.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
