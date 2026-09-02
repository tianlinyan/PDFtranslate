"""Tests for the background worker's signal contract.

The worker drives the whole extract → translate → export pipeline and is the
only thing the GUI listens to, so its *exit* behaviour matters as much as its
output: ``main_window`` quits the ``QThread`` from ``stopped``, and a run that
forgets to emit it leaves the window permanently unable to start again.

These tests call ``run()`` directly on the calling thread (no ``QThread``, no
``QApplication`` needed) and assert on the emitted signals.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from translate_app import pdfio
from translate_app import worker as worker_module
from translate_app.settings import ModelConfig
from translate_app.translator import TranslationAborted, TranslationResult
from translate_app.worker import TranslateWorker, format_duration

from tests._helpers import MockServer, build_sample_pdf


class _WorkerTestBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        # Keep the translation cache out of the user's real home directory.
        patcher = mock.patch.dict(
            os.environ, {"PDFTRANSLATE_CACHE_DIR": str(self.tmp / "cache")}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _model(self, endpoint: str) -> ModelConfig:
        return ModelConfig(
            id=f"mock-{uuid.uuid4().hex[:8]}", name="mock", type="openai",
            endpoint=endpoint, model="mock-model",
        )

    def _run(self, worker: TranslateWorker) -> list[str]:
        """Run the worker and return the ordered names of the signals it emits."""
        events: list[str] = []
        worker.finished.connect(lambda _p: events.append("finished"))
        worker.error.connect(lambda _m: events.append("error"))
        worker.stopped.connect(lambda: events.append("stopped"))
        worker.run()
        return events


class WorkerSignalTest(_WorkerTestBase):
    def test_cancelled_run_still_emits_stopped(self):
        """Regression: a cancelled run emitted no signal at all, so the GUI's
        ``QThread`` never quit and "开始翻译" stayed disabled forever."""
        worker = TranslateWorker(
            str(self.tmp / "missing.pdf"), self._model("http://127.0.0.1:9/v1"),
            "Chinese", "plain_text", str(self.tmp / "out.txt"),
        )
        worker.cancel()
        events = self._run(worker)
        # Cancelling is not an error, so only ``stopped`` may fire — and it must.
        self.assertEqual(["stopped"], events)

    def test_failed_run_emits_error_then_stopped(self):
        worker = TranslateWorker(
            str(self.tmp / "missing.pdf"), self._model("http://127.0.0.1:9/v1"),
            "Chinese", "plain_text", str(self.tmp / "out.txt"),
        )
        events = self._run(worker)
        self.assertEqual(["error", "stopped"], events)

    def test_successful_run_emits_finished_then_stopped(self):
        src = build_sample_pdf(self.tmp / "sample.pdf", pages=1)
        out = self.tmp / "out.txt"
        with MockServer() as server:
            worker = TranslateWorker(
                str(src), self._model(server.endpoint), "Chinese",
                "plain_text", str(out),
            )
            events = self._run(worker)
        self.assertEqual(["finished", "stopped"], events)
        self.assertTrue(out.exists())
        self.assertIn("MOCK:", out.read_text("utf-8"))


class WorkerFailureReportingTest(_WorkerTestBase):
    """A run that could not really translate must say so, not report success."""

    def _stub_engine(self, translate):
        """Patch the engine the worker builds, keeping its call signature."""

        class _StubEngine:
            def __init__(self, _model):
                pass

            def translate_blocks(self, blocks, _lang, **kwargs):
                return translate(blocks, kwargs)

        return mock.patch.object(worker_module, "TranslationEngine", _StubEngine)

    def test_fatal_config_error_is_reported_as_an_error(self):
        """Regression: a wrong API key used to produce a "finished" document
        whose every block was the untranslated source."""
        src = build_sample_pdf(self.tmp / "sample.pdf", pages=1)
        out = self.tmp / "out.txt"

        def translate(_blocks, _kwargs):
            raise TranslationAborted("API 认证失败，请检查 models.json 中的 api_key")

        messages: list[str] = []
        worker = TranslateWorker(
            str(src), self._model("http://127.0.0.1:9/v1"), "Chinese",
            "plain_text", str(out),
        )
        worker.error.connect(messages.append)
        with self._stub_engine(translate):
            events = self._run(worker)

        self.assertEqual(["error", "stopped"], events)
        self.assertIn("api_key", messages[0])
        # Nothing may be exported from an aborted run.
        self.assertFalse(out.exists())

    def test_partial_failures_are_summarised_in_the_log(self):
        src = build_sample_pdf(self.tmp / "sample.pdf", pages=1)
        out = self.tmp / "out.txt"

        def translate(blocks, _kwargs):
            result = TranslationResult(blocks=list(blocks), translated=list(blocks))
            result.errors = ["块 1 翻译失败，保留原文", "块 2 翻译失败，保留原文"]
            return result

        logs: list[str] = []
        worker = TranslateWorker(
            str(src), self._model("http://127.0.0.1:9/v1"), "Chinese",
            "plain_text", str(out),
        )
        worker.log.connect(logs.append)
        with self._stub_engine(translate):
            events = self._run(worker)

        self.assertEqual(["finished", "stopped"], events)
        warnings = [m for m in logs if "个文本块翻译失败" in m]
        self.assertEqual(1, len(warnings), logs)
        self.assertIn("2 个文本块翻译失败", warnings[0])


class FormatDurationTest(unittest.TestCase):
    def test_formats(self):
        self.assertEqual("0.0 秒", format_duration(-1))
        self.assertEqual("12.3 秒", format_duration(12.34))
        self.assertEqual("1 分 05 秒", format_duration(65))
        self.assertEqual("1 小时 01 分 01 秒", format_duration(3661))


class KeepOriginalDirectionTest(_WorkerTestBase):
    """Name cells stay source only for a CJK target; Latin targets romanize.

    A ``姓名`` column keeps its original text so Chinese names survive in a
    Chinese document — but in an English output the kept Chinese names would be
    mixed-language leftovers, so there they must be handed to the model (whose
    prompt rule romanizes them with a consistent spelling).  The block flags
    themselves come from ``pdfio._mark_name_column`` (covered by the pdfio
    tests); here the worker's direction gate is the unit under test, so the
    extraction is stubbed with one name block already flagged.
    """

    def _run_capture(self, lang: str) -> tuple[dict, list[str]]:
        name_block = pdfio.Block(
            text="汪建法", page=0, x0=60, y0=120, x1=100, y1=135,
            size=9.0, keep_original=True,
        )
        doc = pdfio.DocumentText(
            pages=[[pdfio.Block(text="姓名", page=0, x0=60, y0=100, x1=100, y1=115, size=9.0), name_block]],
            blocks=["姓名", "汪建法", "Body text to translate."],
            block_pages=[0, 0, 0],
            title="names",
        )
        captured: dict = {}

        class _StubEngine:
            def __init__(self, _model):
                pass

            def translate_blocks(self, blocks, _target, **kwargs):
                captured["keep"] = kwargs.get("keep_original")
                captured["lang"] = _target
                return TranslationResult(
                    blocks=list(blocks),
                    translated=[f"MOCK:{b}" for b in blocks],
                )

        logs: list[str] = []
        worker = TranslateWorker(
            str(self.tmp / "names.pdf"),
            self._model("http://127.0.0.1:9/v1"), lang,
            "plain_text", str(self.tmp / f"out_{lang}.txt"),
        )
        worker.log.connect(logs.append)
        with mock.patch.object(pdfio, "extract_document_text", return_value=doc):
            with mock.patch.object(worker_module, "TranslationEngine", _StubEngine):
                self._run(worker)
        return captured, logs

    def test_name_cells_kept_for_cjk_target(self):
        captured, logs = self._run_capture("简体中文")
        self.assertEqual("简体中文", captured["lang"])
        self.assertEqual({1}, captured["keep"])
        self.assertTrue(any("保留原文" in m for m in logs), logs)

    def test_name_cells_romanized_for_latin_target(self):
        captured, logs = self._run_capture("English")
        self.assertEqual("English", captured["lang"])
        self.assertEqual(set(), captured["keep"])
        self.assertTrue(any("罗马化" in m for m in logs), logs)


class ChartNodeDirectionTest(_WorkerTestBase):
    """Org-chart / architecture-diagram node labels are never translated.

    Unlike a ``姓名`` name cell (kept only for a CJK target), a diagram's box
    label is structural: it must survive a language switch, so the worker feeds
    it into ``keep_original`` for both a CJK and a Latin target.
    """

    def _run_capture(self, lang: str) -> tuple[dict, list[str]]:
        chart_block = pdfio.Block(
            text="党群工作部", page=0, x0=50, y0=50, x1=57.4, y1=79.5,
            size=24.0, align="center", bold=False, is_chart=True,
        )
        doc = pdfio.DocumentText(
            pages=[[chart_block, pdfio.Block(text="标题", page=0, x0=60, y0=20, x1=150, y1=35, size=12.0)]],
            blocks=["党群工作部", "标题"],
            block_pages=[0, 0],
            title="chart",
        )
        captured: dict = {}

        class _StubEngine:
            def __init__(self, _model):
                pass

            def translate_blocks(self, blocks, _target, **kwargs):
                captured["keep"] = kwargs.get("keep_original")
                return TranslationResult(
                    blocks=list(blocks),
                    translated=[f"MOCK:{b}" for b in blocks],
                )

        logs: list[str] = []
        worker = TranslateWorker(
            str(self.tmp / "chart.pdf"),
            self._model("http://127.0.0.1:9/v1"), lang,
            "plain_text", str(self.tmp / f"out_{lang}.txt"),
        )
        worker.log.connect(logs.append)
        with mock.patch.object(pdfio, "extract_document_text", return_value=doc):
            with mock.patch.object(worker_module, "TranslationEngine", _StubEngine):
                self._run(worker)
        return captured, logs

    def test_chart_nodes_kept_for_cjk_target(self):
        captured, _logs = self._run_capture("简体中文")
        self.assertEqual({0}, captured["keep"])

    def test_chart_nodes_kept_for_latin_target(self):
        # A Western target still keeps the diagram's Chinese node labels — they
        # are structural, not prose, so they must not be romanized.
        captured, logs = self._run_capture("English")
        self.assertEqual({0}, captured["keep"])
        self.assertTrue(any("组织结构图/架构图节点" in m for m in logs), logs)


if __name__ == "__main__":
    unittest.main()
