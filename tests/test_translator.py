"""Tests for the translation engine (batching, alignment, progress, cache)."""
import json
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from translate_app import translator as translator_mod
from translate_app.settings import ModelConfig
from translate_app.translator import (
    TranslationCancelled,
    TranslationEngine,
    _OUTPUT_HEADROOM,
    _block_hash,
    _cache_dir,
    _cache_key,
    _estimate_tokens,
    load_translation_cache,
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

#: Temp dir the tests redirect the on-disk translation cache to; created in
#: ``setUpModule`` so the suite never touches (or is blocked by) the real user
#: cache and cleans up after itself.
_TEST_CACHE_DIR: str | None = None


def setUpModule():
    global _TEST_CACHE_DIR
    _TEST_CACHE_DIR = tempfile.mkdtemp(prefix="pdftranslate_test_cache")
    os.environ["PDFTRANSLATE_CACHE_DIR"] = _TEST_CACHE_DIR


def tearDownModule():
    global _TEST_CACHE_DIR
    if _TEST_CACHE_DIR:
        shutil.rmtree(_TEST_CACHE_DIR, ignore_errors=True)
        _TEST_CACHE_DIR = None
    os.environ.pop("PDFTRANSLATE_CACHE_DIR", None)


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
            cache = load_translation_cache(Path("_fake.pdf"), "Chinese", model.id)
            self.assertEqual(cache, {})

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

    def test_token_aware_batching_respects_max_tokens(self):
        # A tiny ``max_tokens`` must carve batches so a chunk's estimated output
        # stays under the reserved headroom — otherwise the reply is truncated
        # and the engine retries a (wasted) request.
        model = ModelConfig(
            id="m", name="m", type="openai", endpoint="http://x/v1", model="mod",
            batch_size=1_000_000, max_tokens=40,
        )
        engine = TranslationEngine(model)
        blocks = [f"Block {i} text" + ("a" * 20) for i in range(10)]
        chunks = engine._make_chunks(blocks, index_filter=lambda _i: True)
        self.assertGreater(len(chunks), 1)
        cap = int(40 * _OUTPUT_HEADROOM)
        for chunk in chunks:
            est = sum(_estimate_tokens(blocks[i]) + 4 for i in chunk)
            self.assertLessEqual(est, cap)
        # Without ``max_tokens`` the token cap is ignored (char budget only).
        model_plain = ModelConfig(
            id="m2", name="m2", type="openai", endpoint="http://x/v1", model="mod",
            batch_size=1_000_000, max_tokens=None,
        )
        engine_plain = TranslationEngine(model_plain)
        chunks_plain = engine_plain._make_chunks(blocks, index_filter=lambda _i: True)
        self.assertEqual(len(chunks_plain), 1)

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


    def test_cancel_flushes_buffered_translations(self):
        with MockServer() as server:
            model = ModelConfig(
                id=f"mock-cf-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model", batch_size=10,
            )
            engine = TranslationEngine(model)
            # batch_size=10 makes each block its own chunk (serial, in order).
            blocks = ["A block one.", "B block two.", "C block three."]
            calls = [0]

            def cancel():
                calls[0] += 1
                return calls[0] > 1  # let the first batch finish, then cancel

            with self.assertRaises(TranslationCancelled):
                engine.translate_blocks(
                    blocks, "Chinese", doc_path=Path("_fake.pdf"),
                    cancel=cancel, retry_delays=(0.0, 0.0),
                )
            # The completed first batch was held in memory; the cancel must have
            # flushed it, so a resume run reuses rather than re-translates it.
            cache = load_translation_cache(Path("_fake.pdf"), "Chinese", model.id)
            self.assertEqual(set(cache.values()), {"MOCK:A block one."})

    def test_glossary_injected_and_alignment_kept(self):
        with MockServer() as server:
            model = ModelConfig(
                id=f"mock-gl-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model",
            )
            # A glossary passed straight to the engine is injected into the batch
            # prompt for the terms that appear, so concurrent chunks agree on
            # domain terms — while keeping the block numbering aligned.
            blocks = [
                "The key must be kept secret.",
                "The protocol is standard here.",
                "A paragraph without glossary terms.",
            ]
            engine = TranslationEngine(model, glossary={"key": "密钥", "protocol": "协议"})
            res = engine.translate_blocks(blocks, "Chinese", doc_path=Path("_fake.pdf"))
            self.assertEqual(len(res.translated), len(blocks))
            for src, tr in zip(blocks, res.translated):
                self.assertTrue(tr.startswith("MOCK:" + src), tr)
            # The system message (messages[0]) carries the term mapping.
            system = server.last_body["messages"][0]["content"]
            self.assertIn("Glossary", system)
            self.assertIn("key => 密钥", system)
            self.assertIn("protocol => 协议", system)

    def test_glossary_only_injects_terms_present_in_batch(self):
        with MockServer() as server:
            model = ModelConfig(
                id=f"mock-gf-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model",
            )
            blocks = ["The key is here.", "No glossary term."]
            engine = TranslationEngine(model, glossary={"key": "密钥", "zebra": "斑马"})
            engine.translate_blocks(blocks, "Chinese", doc_path=Path("_fake.pdf"))
            system = server.last_body["messages"][0]["content"]
            self.assertIn("key => 密钥", system)
            # ``zebra`` does not appear anywhere in this batch: it must not be
            # injected, so a large glossary is not re-sent verbatim to every chunk.
            self.assertNotIn("zebra", system)

    def test_cache_journal_is_merged(self):
        import json

        p = Path("_fake.pdf")
        model_id = f"mock-j-{uuid.uuid4().hex[:8]}"
        base = _cache_dir() / _cache_key(p, "Chinese", model_id)
        journal = base.with_suffix(".jsonl")
        try:
            base.write_text(json.dumps({"a": "A"}), "utf-8")
            journal.write_text(
                json.dumps({"b": "B"}, ensure_ascii=False) + "\n"
                + json.dumps({"c": "C"}, ensure_ascii=False) + "\n",
                "utf-8",
            )
            # The snapshot and the append-only journal are both read back, so a
            # cancel (which skips the final snapshot) still resumes correctly.
            cache = load_translation_cache(p, "Chinese", model_id)
            self.assertEqual(cache, {"a": "A", "b": "B", "c": "C"})
            # A torn trailing line must be skipped, not reject the whole file.
            journal.write_text(json.dumps({"d": "D"}, ensure_ascii=False) + "\n" + "{broken", "utf-8")
            self.assertEqual(
                load_translation_cache(p, "Chinese", model_id),
                {"a": "A", "d": "D"},
            )
        finally:
            base.unlink(missing_ok=True)
            journal.unlink(missing_ok=True)

    def test_glossary_changes_cache_namespace(self):
        p = Path("_fake.pdf")
        base = _cache_key(p, "Chinese", "m")
        self.assertEqual(base, _cache_key(p, "Chinese", "m", None))
        self.assertEqual(base, _cache_key(p, "Chinese", "m", {}))
        # A different glossary (or a changed mapping) must not reuse the cache.
        self.assertNotEqual(base, _cache_key(p, "Chinese", "m", {"key": "密钥"}))
        self.assertNotEqual(
            _cache_key(p, "Chinese", "m", {"key": "密钥"}),
            _cache_key(p, "Chinese", "m", {"key": "password"}),
        )

    def test_unnumbered_line_fallback_maps_in_order(self):
        # A model that ignores the ``[n]`` protocol entirely still works via
        # positional line matching when it returns one line per block.
        parsed = TranslationEngine._parse_response(
            "译文一\n译文二", ["a", "b"], [0, 1]
        )
        self.assertEqual(parsed, ["译文一", "译文二"])

    def test_unnumbered_short_reply_is_rejected(self):
        # Without ``[n]`` markers AND with fewer lines than blocks, the reply
        # must be rejected — padding the missing blocks with the source text
        # would cache untranslated text and poison resume.
        with self.assertRaises(ValueError):
            TranslationEngine._parse_response("only line", ["a", "b", "c"], [0, 1, 2])

    def test_unnumbered_extra_line_is_rejected(self):
        # A model that ignores the ``[n]`` protocol and prepends an intro line
        # would shift every translation by one if we blindly took the first N
        # lines; any count ABOVE the block count must be rejected and retried
        # (never silently misaligned into the cache).
        with self.assertRaises(ValueError):
            TranslationEngine._parse_response(
                "译者前言\n译文一\n译文二\n译文三", ["a", "b", "c"], [0, 1, 2]
            )

    def test_unnumbered_short_reply_is_not_cached(self):
        # End-to-end: a persistently short unnumbered reply is retried, then
        # the source is preserved, the failure recorded, and NOTHING is
        # written to the cache (otherwise a resume would skip these blocks
        # forever, treating the source as translated).
        with MockServer() as server:
            model = ModelConfig(
                id=f"mock-nm-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model",
            )
            engine = TranslationEngine(model)
            engine._request_locked = lambda _p, _s: "only one line"  # type: ignore[method-assign]
            blocks = ["First block.", "Second block."]
            res = engine.translate_blocks(
                blocks, "Chinese", doc_path=Path("_fake.pdf"),
                retry_delays=(0.0, 0.0),
            )
            self.assertEqual(res.translated, blocks)
            self.assertTrue(res.errors)
            cache = load_translation_cache(Path("_fake.pdf"), "Chinese", model.id)
            self.assertEqual(cache, {})

    def test_compact_triggers_on_journal_total_size(self):
        # A fully-cached run adds nothing new, but the journal ALREADY holds
        # (patched) threshold-many entries from earlier runs: it must still be
        # compacted into the snapshot, so the journal cannot grow unbounded
        # across small incremental runs.
        p = Path("_compact_test.pdf")
        model_id = f"mock-cp-{uuid.uuid4().hex[:8]}"
        base = _cache_dir() / _cache_key(p, "Chinese", model_id)
        journal = base.with_suffix(".jsonl")
        try:
            blocks = ["Compact me one.", "Compact me two."]
            with journal.open("w", encoding="utf-8") as fh:
                for b in blocks:
                    fh.write(
                        json.dumps({_block_hash(b): "MOCK:" + b}, ensure_ascii=False)
                        + "\n"
                    )
            model = ModelConfig(
                id=model_id, name="mock", type="openai",
                endpoint="http://x/v1", model="m",
            )
            engine = TranslationEngine(model)
            with mock.patch.object(translator_mod, "_COMPACT_ENTRIES", 2):
                res = engine.translate_blocks(blocks, "Chinese", doc_path=p)
            # Served from the journal, then the journal compacted away.
            self.assertEqual(
                res.translated, ["MOCK:Compact me one.", "MOCK:Compact me two."]
            )
            self.assertFalse(journal.exists())
            snap = json.loads(base.read_text("utf-8"))
            self.assertEqual(snap[_block_hash(blocks[0])], "MOCK:Compact me one.")
            self.assertEqual(snap[_block_hash(blocks[1])], "MOCK:Compact me two.")
            # The atomic-write temp file must not linger.
            self.assertFalse(base.with_name(base.name + ".tmp").exists())
        finally:
            base.unlink(missing_ok=True)
            journal.unlink(missing_ok=True)

    def test_compact_cache_roundtrip(self):
        base = _cache_dir() / _cache_key(Path("_compact_unit.pdf"), "Chinese", "unit-c")
        journal = base.with_suffix(".jsonl")
        try:
            journal.write_text('{"k1": "v1"}\n', "utf-8")
            translator_mod._compact_cache(base, journal, {"k1": "v1", "k2": "v2"})
            self.assertEqual(
                json.loads(base.read_text("utf-8")), {"k1": "v1", "k2": "v2"}
            )
            self.assertFalse(journal.exists())
            self.assertFalse(base.with_name(base.name + ".tmp").exists())
        finally:
            base.unlink(missing_ok=True)
            journal.unlink(missing_ok=True)

    def test_cache_write_failure_is_logged(self):
        # A silently failing cache is worse than no cache at all (every run
        # restarts from zero with no hint why): the user must see one warning.
        with MockServer() as server:
            model = ModelConfig(
                id=f"mock-wf-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
                endpoint=server.endpoint, model="mock-model",
            )
            engine = TranslationEngine(model)
            messages: list[str] = []
            with mock.patch.object(
                translator_mod, "_append_cache_journal", return_value=False
            ):
                res = engine.translate_blocks(
                    ["Write failure."], "Chinese",
                    doc_path=Path("_fake.pdf"), log=messages.append,
                )
            # Translation itself still succeeds; only durability was lost.
            self.assertEqual(res.translated, ["MOCK:Write failure."])
            self.assertTrue(any("缓存" in m and "失败" in m for m in messages))


if __name__ == "__main__":
    unittest.main()
