"""Tests for the translation engine (batching, alignment, progress, cache)."""
import json
import unittest
import uuid
from pathlib import Path

from translate_app.settings import ModelConfig
from translate_app.translator import TranslationEngine, _cache_dir, _cache_key

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

    def test_multiline_response_is_parsed(self):
        # A model may wrap a long translation across several lines; everything
        # up to the next [n] marker belongs to that block.
        raw = (
            "[1]\nFirst line.\nsecond line continued.\n"
            "[2]\nSecond block."
        )
        parsed = TranslationEngine._parse_response(raw, ["a", "b"], [0, 1])
        self.assertEqual(
            parsed, ["First line. second line continued.", "Second block."]
        )

    def test_incomplete_numbered_output_is_rejected(self):
        # A response that echoes only some of the requested block numbers must be
        # rejected (treated as a malformed/transient reply for the caller to
        # retry) instead of silently filling the gaps with the original text.
        raw = "[1]\nFirst block.\n[2]\nSecond block."  # third block missing
        with self.assertRaises(ValueError):
            TranslationEngine._parse_response(raw, ["a", "b", "c"], [0, 1, 2])

        # Out-of-range marker (echoes [4] for a 3-block batch) is also rejected.
        with self.assertRaises(ValueError):
            TranslationEngine._parse_response("[1]\na\n[4]\nd", ["a", "b", "c"], [0, 1, 2])

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


if __name__ == "__main__":
    unittest.main()
