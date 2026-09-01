"""Tests for the translation engine (batching, alignment, progress, cache)."""
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import httpx
from openai import AuthenticationError, BadRequestError

from translate_app import translator
from translate_app.settings import ModelConfig
from translate_app.translator import (
    TranslationAborted,
    TranslationEngine,
    _cache_dir,
    _cache_key,
)

from tests._helpers import MockServer

BLOCKS = [
    "First paragraph to translate.",
    "Second paragraph with more words here.",
    "这是一个测试段落。",
    "Fourth paragraph to translate.",
    "Fifth paragraph to translate.",
    "Sixth paragraph to translate.",
]


class TranslatorTest(unittest.TestCase):
    def setUp(self):
        # Keep the cache out of the developer's real ``~/.pdftranslate/cache``:
        # every run used to leave files there forever.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache_dir = Path(tmp.name)
        patcher = mock.patch.dict(
            os.environ, {"PDFTRANSLATE_CACHE_DIR": str(self.cache_dir)}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _engine(self, server: MockServer) -> TranslationEngine:
        model = ModelConfig(
            id="mock", name="mock", type="openai",
            endpoint=server.endpoint, model="mock-model",
        )
        return TranslationEngine(model)

    def test_alignment_and_local_numbering(self):
        with MockServer() as server:
            engine = self._engine(server)
            res = engine.translate_blocks(BLOCKS, "Chinese", doc_path=Path("_fake.pdf"))
            self.assertEqual(len(res.translated), len(BLOCKS))
            for src, tr in zip(BLOCKS, res.translated):
                self.assertTrue(tr.startswith("MOCK:" + src), tr)

    def test_progress_starts_at_cached_count(self):
        with MockServer() as server:
            # Unique model id so the cache starts empty on the first run.
            model = ModelConfig(
                id=f"mock-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model",
            )
            doc_path = Path("_cache_test.pdf")
            progress: list[tuple[int, int]] = []

            engine1 = TranslationEngine(model)
            engine1.translate_blocks(
                BLOCKS, "Chinese", doc_path=doc_path,
                on_progress=lambda d, t: progress.append((d, t)),
            )
            # First run: nothing cached yet -> starts at 0.
            self.assertEqual(progress[0], (0, len(BLOCKS)))

            progress2: list[tuple[int, int]] = []
            engine2 = TranslationEngine(model)
            engine2.translate_blocks(
                BLOCKS, "Chinese", doc_path=doc_path,
                on_progress=lambda d, t: progress2.append((d, t)),
            )
            # Second run: everything cached -> starts at total (no re-translate).
            self.assertEqual(progress2[0], (len(BLOCKS), len(BLOCKS)))

    def test_reasoning_effort_is_sent(self):
        with MockServer() as server:
            # Unique model id so a warm cache cannot satisfy the request (and the
            # request actually reaches the mock server, setting ``last_body``).
            model = ModelConfig(
                id=f"mock-rs-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model",
                reasoning_effort="low",
            )
            engine = TranslationEngine(model)
            engine.translate_blocks(["Hello."], "Chinese", doc_path=Path("_fake.pdf"))
            self.assertIsNotNone(server.last_body)
            self.assertEqual(server.last_body.get("reasoning_effort"), "low")

    def test_tools_choice_is_sent(self):
        with MockServer() as server:
            # Unique model id so a warm cache cannot satisfy the request.
            model = ModelConfig(
                id=f"mock-tc-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model", tools_choice="auto",
            )
            engine = TranslationEngine(model)
            engine.translate_blocks(["Hello."], "Chinese", doc_path=Path("_fake.pdf"))
            self.assertIsNotNone(server.last_body)
            # ``tools_choice`` is sent as the API field ``tool_choice``.
            self.assertEqual(server.last_body.get("tool_choice"), "auto")

    def test_parse_failure_falls_back_to_original(self):
        with MockServer() as server:
            # Unique model id so the cache is empty and a batch is actually sent.
            model = ModelConfig(
                id=f"mock-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model",
            )
            engine = TranslationEngine(model)

            def boom(_prompt, _system):
                raise RuntimeError("simulated connection failure")

            engine._request_locked = boom  # type: ignore[method-assign]
            res = engine.translate_blocks(
                BLOCKS, "Chinese", doc_path=Path("_fake.pdf"),
                retry_delays=(0.0, 0.0),
            )
            # On persistent failure the source text is preserved ...
            self.assertEqual(res.translated, BLOCKS)
            # ... the failure is recorded ...
            self.assertTrue(res.errors)
            # ... and the failed batch must NOT be written into the cache,
            # otherwise a transient outage would poison future resume runs.
            cache_path = _cache_dir() / _cache_key(
                Path("_fake.pdf"), "Chinese", model.id
            )
            self.assertEqual(json.loads(cache_path.read_text("utf-8")), {})

    def test_unwritable_cache_dir_falls_back_to_temp(self):
        # ``mkdir(exist_ok=True)`` succeeds on an existing directory even when
        # writing into it is denied (read-only home, sandbox).  Only a real
        # probe write catches that, so the fallback must still kick in.
        home_cache = Path.home() / ".pdftranslate" / "cache"
        real_write_text = Path.write_text

        def fake_write_text(self, *args, **kwargs):
            if self.name.startswith(".write_probe_") and home_cache in self.parents:
                raise PermissionError("denied by test")
            return real_write_text(self, *args, **kwargs)

        # This test is about the home -> temp fallback, so the explicit
        # ``PDFTRANSLATE_CACHE_DIR`` override set in setUp must be out of the way.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PDFTRANSLATE_CACHE_DIR", None)
            with mock.patch.object(Path, "write_text", fake_write_text):
                resolved = _cache_dir()
        self.assertEqual(resolved, Path(tempfile.gettempdir()) / "pdftranslate_cache")

    def test_cache_dir_env_override_is_used(self):
        self.assertEqual(self.cache_dir, _cache_dir())

    def test_cache_write_is_atomic(self):
        # The GUI hard-exits the process on close, so a plain overwrite could
        # leave truncated JSON behind — which reads back as "no cache" and
        # silently re-translates the whole document next time.
        path = self.cache_dir / "trans_v9_atomic.json"
        path.write_text('{"old": "value"}', "utf-8")

        real_replace = os.replace
        seen: list[str] = []

        def failing_replace(src, dst):
            seen.append(str(src))
            raise OSError("disk full")

        with mock.patch.object(os, "replace", failing_replace):
            reason = translator._write_cache(path, {"new": "value"})
        self.assertIsNotNone(reason)
        # The previous cache is untouched, and no temp file is left behind.
        self.assertEqual({"old": "value"}, json.loads(path.read_text("utf-8")))
        self.assertEqual([], list(self.cache_dir.glob("*.tmp")))
        self.assertTrue(seen)

        # And the successful path really swaps the new content in.
        self.assertIsNone(translator._write_cache(path, {"new": "value"}))
        self.assertEqual({"new": "value"}, json.loads(path.read_text("utf-8")))
        self.assertEqual([], list(self.cache_dir.glob("*.tmp")))
        self.assertIs(os.replace, real_replace)

    def test_corrupted_cache_file_is_ignored(self):
        doc_path = Path("_corrupt.pdf")
        cache_path = _cache_dir() / _cache_key(doc_path, "Chinese", "mock-corrupt")
        cache_path.write_text('["not", "a", "mapping"]', "utf-8")
        self.assertEqual({}, translator.load_translation_cache(
            doc_path, "Chinese", "mock-corrupt"
        ))

    def test_clear_cache_also_removes_temp_leftovers(self):
        (self.cache_dir / "trans_v3_abc.json").write_text("{}", "utf-8")
        (self.cache_dir / "trans_v3_abc.json.123.tmp").write_text("{", "utf-8")
        removed = translator.clear_translation_cache()
        self.assertEqual(2, removed)
        self.assertEqual([], list(self.cache_dir.glob("trans_*")))


    def test_cache_write_failure_is_logged_once(self):
        # An unwritable cache must not break the translation, but it must be
        # reported — otherwise every re-run silently re-translates everything.
        with MockServer() as server:
            model = ModelConfig(
                id=f"mock-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model",
            )
            engine = TranslationEngine(model)
            logs: list[str] = []
            with mock.patch.object(
                translator, "_write_cache", return_value="PermissionError: denied"
            ):
                res = engine.translate_blocks(
                    BLOCKS, "Chinese", doc_path=Path("_fake.pdf"), log=logs.append
                )
            # Translation still succeeds ...
            for src, tr in zip(BLOCKS, res.translated):
                self.assertTrue(tr.startswith("MOCK:" + src), tr)
            # ... and the failure is reported exactly once, even though the
            # cache is persisted per batch *and* once at the end.
            warnings = [m for m in logs if "缓存写入失败" in m]
            self.assertEqual(len(warnings), 1, logs)

    def test_multiline_response_is_parsed(self):
        # A model may wrap a long translation across several lines; everything
        # up to the next [n] marker belongs to that block.
        raw = (
            "[1]\nFirst line.\nsecond line continued.\n"
            "[2]\nSecond block."
        )
        parsed = TranslationEngine._parse_response(raw, [0, 1])
        self.assertEqual(
            parsed, ["First line. second line continued.", "Second block."]
        )

    def test_incomplete_numbered_output_is_rejected(self):
        # A response that echoes only some of the requested block numbers must be
        # rejected (treated as a malformed/transient reply for the caller to
        # retry) instead of silently filling the gaps with the original text.
        raw = "[1]\nFirst block.\n[2]\nSecond block."  # third block missing
        with self.assertRaises(ValueError):
            TranslationEngine._parse_response(raw, [0, 1, 2])

        # Out-of-range marker (echoes [4] for a 3-block batch) is also rejected.
        with self.assertRaises(ValueError):
            TranslationEngine._parse_response("[1]\na\n[4]\nd", [0, 1, 2])

    def test_unnumbered_reply_must_match_the_block_count(self):
        # Regression: with no [n] markers at all the parser used to map reply
        # lines onto blocks positionally and pad the rest with the source text.
        # A one-line refusal therefore became block 1's "translation" — and,
        # counting as success, was written to the cache and reused forever.
        with self.assertRaises(ValueError):
            TranslationEngine._parse_response(
                "Sorry, I cannot translate this.", [0, 1, 2]
            )
        # A chatty preamble ahead of the translations is rejected too.
        with self.assertRaises(ValueError):
            TranslationEngine._parse_response(
                "Sure! Here you go:\nFirst.\nSecond.", [0, 1]
            )

    def test_unnumbered_reply_with_one_line_per_block_is_accepted(self):
        parsed = TranslationEngine._parse_response("First.\nSecond.", [0, 1])
        self.assertEqual(["First.", "Second."], parsed)

    def test_unnumbered_single_block_reply_is_folded(self):
        # A single requested block owns the whole reply, however it is wrapped.
        parsed = TranslationEngine._parse_response("A long\nwrapped reply.", [7])
        self.assertEqual(["A long wrapped reply."], parsed)

    def test_unparseable_reply_keeps_source_and_skips_cache(self):
        # End-to-end: a model that answers without markers and with the wrong
        # line count must leave the document untranslated *and* uncached.
        with MockServer() as server:
            model = ModelConfig(
                id=f"mock-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model",
            )
            engine = TranslationEngine(model)
            engine._request_locked = lambda _p, _s: "Sorry, I cannot translate this."
            doc_path = Path("_unparseable.pdf")
            res = engine.translate_blocks(
                BLOCKS, "Chinese", doc_path=doc_path, retry_delays=(0.0, 0.0),
            )
            self.assertEqual(BLOCKS, res.translated)
            self.assertTrue(res.errors)
            cache_path = _cache_dir() / _cache_key(doc_path, "Chinese", model.id)
            self.assertEqual({}, json.loads(cache_path.read_text("utf-8")))


    def test_symbol_only_blocks_are_skipped(self):
        with MockServer() as server:
            model = ModelConfig(
                id=f"mock-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model",
            )
            engine = TranslationEngine(model)
            blocks = ["1234", "§§ --", "Hello world."]
            progress: list[tuple[int, int]] = []
            res = engine.translate_blocks(
                blocks, "Chinese", doc_path=Path("_fake.pdf"),
                on_progress=lambda d, t: progress.append((d, t)),
            )
            # Symbol-only blocks are kept verbatim; the rest is translated.
            self.assertEqual(res.translated[0], "1234")
            self.assertEqual(res.translated[1], "§§ --")
            self.assertEqual(res.translated[2], "MOCK:Hello world.")
            # Progress starts at the two skipped blocks (no request for them)...
            self.assertEqual(progress[0], (2, 3))
            # ... and the request body only contains the block worth translating.
            user = server.last_body["messages"][1]["content"]
            self.assertNotIn("1234", user)
            self.assertIn("Hello world.", user)

    def test_concurrent_batches_keep_order(self):
        with MockServer() as server:
            model = ModelConfig(
                id=f"mock-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model", concurrency=4,
            )
            engine = TranslationEngine(model)
            # ~20 long blocks exceed the 4000-char budget several times, so
            # multiple batches are submitted concurrently.  No trailing space:
            # the mock echoes the block text stripped.
            blocks = [
                f"Block number {i}. " + " ".join(["word"] * 80) for i in range(20)
            ]
            progress: list[tuple[int, int]] = []
            res = engine.translate_blocks(
                blocks, "Chinese", doc_path=Path("_fake.pdf"),
                on_progress=lambda d, t: progress.append((d, t)),
            )
            # Completion order may differ from submission order, but the output
            # must stay aligned with the input blocks.
            self.assertEqual(len(res.translated), len(blocks))
            for src, tr in zip(blocks, res.translated):
                self.assertTrue(tr.startswith("MOCK:" + src), tr)
            self.assertEqual(progress[-1], (len(blocks), len(blocks)))

    def test_batch_size_config_controls_chunking(self):
        with MockServer() as server:
            model = ModelConfig(
                id=f"mock-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model", batch_size=30,
            )
            engine = TranslationEngine(model)
            # A 30-char budget is smaller than every block (~35+), so each
            # block gets its own chunk.
            chunks = engine._make_chunks(BLOCKS, index_filter=lambda _i: True)
            self.assertEqual(len(chunks), len(BLOCKS))
            for chunk in chunks:
                self.assertEqual(len(chunk), 1)

    def test_temperature_and_max_tokens_sent(self):
        with MockServer() as server:
            model = ModelConfig(
                id=f"mock-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model",
                temperature=0.7, max_tokens=1234,
            )
            engine = TranslationEngine(model)
            engine.translate_blocks(["Hello."], "Chinese", doc_path=Path("_fake.pdf"))
            self.assertEqual(server.last_body["temperature"], 0.7)
            self.assertEqual(server.last_body["max_tokens"], 1234)

    def test_default_temperature_sent(self):
        with MockServer() as server:
            model = ModelConfig(
                id=f"mock-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model",
            )
            engine = TranslationEngine(model)
            engine.translate_blocks(["Hello."], "Chinese", doc_path=Path("_fake.pdf"))
            self.assertEqual(server.last_body["temperature"], 0.2)
            self.assertNotIn("max_tokens", server.last_body)


class FatalErrorTest(unittest.TestCase):
    """A wrong key / model must stop the run, not "translate" into the source."""

    def _auth_error(self) -> AuthenticationError:
        request = httpx.Request("POST", "http://x/v1/chat/completions")
        return AuthenticationError(
            "invalid api key", response=httpx.Response(401, request=request), body=None
        )

    def _engine(self, endpoint: str) -> TranslationEngine:
        return TranslationEngine(
            ModelConfig(
                id=f"mock-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=endpoint, model="mock-model",
            )
        )

    def test_auth_error_aborts_the_whole_run(self):
        with MockServer() as server:
            engine = self._engine(server.endpoint)
            err = self._auth_error()

            def boom(_prompt, _system):
                raise err

            engine._request_locked = boom  # type: ignore[method-assign]
            with self.assertRaises(TranslationAborted) as ctx:
                engine.translate_blocks(
                    BLOCKS, "Chinese", doc_path=Path("_fatal.pdf"),
                    retry_delays=(0.0, 0.0),
                )
        # The message must point at the actual cause (the API key).
        self.assertIn("api_key", str(ctx.exception))

    def test_fatal_error_does_not_retry_or_run_further_batches(self):
        with MockServer() as server:
            model = ModelConfig(
                id=f"mock-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model", batch_size=30,
            )
            engine = TranslationEngine(model)
            calls: list[str] = []
            err = self._auth_error()

            def boom(prompt, _system):
                calls.append(prompt)
                raise err

            engine._request_locked = boom  # type: ignore[method-assign]
            # A 30-char budget puts every block in its own batch ...
            with self.assertRaises(TranslationAborted):
                engine.translate_blocks(
                    BLOCKS, "Chinese", doc_path=Path("_fatal2.pdf"),
                    retry_delays=(0.0, 0.0),
                )
            # ... yet exactly one doomed request is issued: no retries for a
            # fatal error, and the batches queued behind it bail out.
            self.assertEqual(1, len(calls), calls)

    def test_bad_request_is_not_fatal(self):
        # 400 is usually data-dependent (one oversized batch), so the other
        # batches must still get their chance: keep the source, carry on.
        with MockServer() as server:
            engine = self._engine(server.endpoint)
            request = httpx.Request("POST", "http://x/v1/chat/completions")
            err = BadRequestError(
                "context length exceeded",
                response=httpx.Response(400, request=request), body=None,
            )

            def boom(_prompt, _system):
                raise err

            engine._request_locked = boom  # type: ignore[method-assign]
            res = engine.translate_blocks(
                BLOCKS, "Chinese", doc_path=Path("_badreq.pdf"),
                retry_delays=(0.0, 0.0),
            )
        self.assertEqual(BLOCKS, res.translated)
        self.assertTrue(res.errors)


if __name__ == "__main__":
    unittest.main()

