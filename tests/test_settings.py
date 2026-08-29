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

    def test_client_kwargs_default_timeout(self):
        m = ModelConfig(id="m", name="m", type="openai",
                        endpoint="http://x/v1", model="mod")
        self.assertEqual(m.client_kwargs()["timeout"], 300.0)


if __name__ == "__main__":
    unittest.main()
