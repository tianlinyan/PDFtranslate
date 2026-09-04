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

from translate_app import agent as agent_module
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

    def test_successful_pdf_run_records_last_pdf_for_preview(self):
        # Regression: after a PDF export the preview's "译文" side must be able to
        # render the real output; the worker records the exported PDF path so the
        # main window can keep it (the worker reference is dropped in ``_cleanup``).
        src = build_sample_pdf(self.tmp / "sample.pdf", pages=1)
        out = self.tmp / "out.pdf"
        with MockServer() as server:
            worker = TranslateWorker(
                str(src), self._model(server.endpoint), "Chinese",
                "translated_pdf", str(out),
            )
            self._run(worker)
        self.assertTrue(out.exists())
        self.assertEqual(str(out), worker._last_pdf)
        self.assertEqual("translated_pdf", worker._output_type)

    def test_markdown_run_leaves_last_pdf_none(self):
        # A non-PDF output has nothing to render in the preview's "译文" side.
        src = build_sample_pdf(self.tmp / "sample.pdf", pages=1)
        out = self.tmp / "out.md"
        with MockServer() as server:
            worker = TranslateWorker(
                str(src), self._model(server.endpoint), "Chinese",
                "markdown", str(out),
            )
            self._run(worker)
        self.assertTrue(out.exists())
        self.assertIsNone(worker._last_pdf)


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


class AgentModeWiringTest(_WorkerTestBase):
    """v0.3.0 default: ``agent_mode`` routes translation through the agent loop."""

    def test_agent_mode_routes_to_agent_and_passes_preview_handler(self):
        src = build_sample_pdf(self.tmp / "agent.pdf", pages=1)
        out = self.tmp / "out.txt"
        doc = pdfio.DocumentText(
            pages=[[pdfio.Block(text="你好", page=0, x0=40, y0=40, x1=120, y1=60, size=10.0)]],
            blocks=["你好"], block_pages=[0], title="agent",
        )
        model = ModelConfig.from_dict(dict(
            id="v", name="v", type="llama-server",
            endpoint="http://127.0.0.1:9/v1/chat/completions", model="qwen", vision=True))
        captured: dict = {}
        handler = object()
        tasks: list[str] = []

        def fake_run_page_visual(state, page, _model, **kw):
            tasks.append(kw.get("task"))
            captured["preview_handler"] = kw.get("preview_handler")
            state.out_doc = {0: {"text": "Hello from agent"}}
            return state

        class _StubEngine:
            def __init__(self, _model):
                pass

            def translate_blocks(self, blocks, _target, **kwargs):
                raise AssertionError("deterministic engine must not run in agent mode")

        logs: list[str] = []
        worker = TranslateWorker(str(src), model, "English", "plain_text", str(out),
                                 preview_handler=handler, agent_mode=True)
        worker.log.connect(logs.append)
        with mock.patch.object(pdfio, "extract_document_text", return_value=doc):
            with mock.patch.object(agent_module, "run_page_visual",
                                   side_effect=fake_run_page_visual):
                with mock.patch.object(worker_module, "TranslationEngine", _StubEngine):
                    events = self._run(worker)
        # The agent path handed the preview_handler through, ran the translate task
        # AND the M4 review pass (a separate '复核' task), and the resulting
        # out_doc fed the export (not the deterministic engine).
        self.assertEqual(handler, captured["preview_handler"])
        self.assertTrue(any("翻译成 English" in t and "set_text" in t for t in tasks), tasks)
        self.assertTrue(any("复核" in t for t in tasks), tasks)   # M4 AI self-check ran
        self.assertIn("Hello from agent", Path(out).read_text("utf-8"))
        self.assertIn("finished", events)
        self.assertTrue(any("已启用 AI 编排" in m for m in logs), logs)

    def test_non_vision_model_falls_back_to_deterministic(self):
        # agent_mode on but the model has no vision → the deterministic pipeline runs.
        src = build_sample_pdf(self.tmp / "agent2.pdf", pages=1)
        out = self.tmp / "out2.txt"
        model = self._model("http://127.0.0.1:9/v1")   # non-vision
        ran: dict = {}

        class _StubEngine:
            def __init__(self, _model):
                pass

            def translate_blocks(self, blocks, _target, **kwargs):
                ran["translated"] = True
                return TranslationResult(blocks=list(blocks),
                                         translated=[f"MOCK:{b}" for b in blocks])

        worker = TranslateWorker(str(src), model, "English", "plain_text", str(out),
                                 agent_mode=True)
        with mock.patch.object(worker_module, "TranslationEngine", _StubEngine):
            events = self._run(worker)
        self.assertTrue(ran.get("translated"))
        self.assertIn("MOCK:", Path(out).read_text("utf-8"))
        self.assertIn("finished", events)


class WorkerAgentPreviewTest(_WorkerTestBase):
    """The agent-state helpers surfaced to the GUI (translation preview + sidebar msg)."""

    def test_add_user_requirement_appends_to_agent_state(self):
        worker = TranslateWorker(
            str(self.tmp / "x.pdf"), self._model("http://127.0.0.1:9/v1"),
            "English", "plain_text", str(self.tmp / "out.txt"),
        )
        state = agent_module.WorkflowState(src_path="x.pdf", lang="English")
        worker._agent_state = state
        worker.add_user_requirement("保留英文货币单位")
        self.assertEqual(["保留英文货币单位"], state.requirements)
        # A blank message is a no-op (never appends an empty requirement).
        worker.add_user_requirement("   ")
        self.assertEqual(["保留英文货币单位"], state.requirements)
        # No active agent run -> no-op, and the message still lands in the log.
        logs: list[str] = []
        worker.log.connect(logs.append)
        worker._agent_state = None
        worker.add_user_requirement("x")
        self.assertIsNone(worker._agent_state)

    def test_render_translation_overlays_translated_blocks(self):
        src = build_sample_pdf(self.tmp / "sample.pdf", pages=1)
        worker = TranslateWorker(
            str(src), self._model("http://127.0.0.1:9/v1"), "English",
            "translated_pdf", str(self.tmp / "out.pdf"),
        )
        state = agent_module.WorkflowState(src_path=str(src), lang="English")
        state.src_doc = pdfio.DocumentText(
            pages=[[
                pdfio.Block("Page 1 heading text.", page=0, x0=72, y0=74, x1=420, y1=88, size=12.0),
                pdfio.Block("This is a sample body paragraph to translate.", page=0,
                            x0=72, y0=114, x1=460, y1=128, size=12.0),
            ]],
            blocks=["Page 1 heading text.", "This is a sample body paragraph to translate."],
            block_pages=[0, 0],
        )
        state.out_doc = {
            0: {"text": "Page 1 heading translated."},
            1: {"text": "This body paragraph is translated."},
        }
        worker._agent_state = state
        png = worker.render_translation(0)
        self.assertIsNotNone(png)
        self.assertGreater(len(png), 100)   # a real, non-trivial PNG
        # PNG magic bytes.
        self.assertEqual(b"\x89PNG", png[:4])

    def test_render_translation_none_when_no_agent_state(self):
        src = build_sample_pdf(self.tmp / "sample.pdf", pages=1)
        worker = TranslateWorker(
            str(src), self._model("http://127.0.0.1:9/v1"), "English",
            "translated_pdf", str(self.tmp / "out.pdf"),
        )
        worker._agent_state = None
        self.assertIsNone(worker.render_translation(0))

    def test_render_page_for_agent_falls_back_to_source(self):
        # The ``render_page`` tool returns a PNG even when there is no agent run /
        # no translation yet (falls back to the raw source page).
        src = build_sample_pdf(self.tmp / "sample.pdf", pages=1)
        worker = TranslateWorker(
            str(src), self._model("http://127.0.0.1:9/v1"), "English",
            "translated_pdf", str(self.tmp / "out.pdf"),
        )
        worker._agent_state = None
        png = worker.render_page_for_agent(0, "translation")
        self.assertIsNotNone(png)
        self.assertEqual(b"\x89PNG", png[:4])


class OverlayApplyTest(_WorkerTestBase):
    """The protected chat/manual overlay always wins over what the run produced."""

    def test_overlay_overrides_export(self):
        src = build_sample_pdf(self.tmp / "ov.pdf", pages=1)
        out = self.tmp / "out.txt"

        class _StubEngine:
            def __init__(self, _model):
                pass

            def translate_blocks(self, blocks, _target, **kwargs):
                return TranslationResult(
                    blocks=list(blocks),
                    translated=list(blocks),
                )

        logs: list[str] = []
        worker = TranslateWorker(
            str(src), self._model("http://127.0.0.1:9/v1"), "Chinese",
            "plain_text", str(out),
            overlay={0: {"text": "被保护的译文"}},
        )
        worker.log.connect(logs.append)
        with mock.patch.object(worker_module, "TranslationEngine", _StubEngine):
            events = self._run(worker)
        self.assertEqual(["finished", "stopped"], events)
        text = out.read_text("utf-8")
        # The protected edit replaced the run's translation of block 0.
        self.assertIn("被保护的译文", text)
        self.assertIn("已应用 1 处", "\n".join(logs))

    def test_blank_overlay_entry_is_skipped(self):
        # A blank / whitespace overlay text must never wipe content to nothing.
        src = build_sample_pdf(self.tmp / "ov2.pdf", pages=1)
        out = self.tmp / "out2.txt"

        class _StubEngine:
            def __init__(self, _model):
                pass

            def translate_blocks(self, blocks, _target, **kwargs):
                return TranslationResult(
                    blocks=list(blocks),
                    translated=list(blocks),
                )

        worker = TranslateWorker(
            str(src), self._model("http://127.0.0.1:9/v1"), "Chinese",
            "plain_text", str(out),
            overlay={0: {"text": "   "}},
        )
        with mock.patch.object(worker_module, "TranslationEngine", _StubEngine):
            events = self._run(worker)
        self.assertEqual(["finished", "stopped"], events)
        # The source text survives (nothing was blanked).
        self.assertIn("Page 1 heading text.", out.read_text("utf-8"))

    def test_re_export_applies_overlay_without_translation(self):
        # "重新导出": re-write the last committed translation with the current
        # overlay edits — no extraction-only re-translation, just export.  The edited
        # block must win over what the last run produced.
        src = build_sample_pdf(self.tmp / "re.pdf", pages=1)
        out = self.tmp / "re.txt"
        doc = pdfio.extract_document_text(str(src), ocr=False, log=lambda m: None)
        last = list(doc.blocks)   # pretend the last run echoed source text
        worker = TranslateWorker(
            str(src), self._model("http://127.0.0.1:9/v1"), "Chinese",
            "plain_text", str(out),
            overlay={0: {"text": "重新导出的编辑"}},
            re_export=True, last_translated=last,
        )
        logs: list[str] = []
        worker.log.connect(logs.append)
        events = self._run(worker)
        self.assertEqual(["finished", "stopped"], events)
        text = out.read_text("utf-8")
        self.assertIn("重新导出的编辑", text)
        self.assertIn("已应用 1 处", "\n".join(logs))

    def test_re_export_with_no_last_translation_errors(self):
        # Re-export needs a previous committed translation; otherwise it must report
        # an error rather than silently doing nothing.
        src = build_sample_pdf(self.tmp / "re2.pdf", pages=1)
        out = self.tmp / "re2.txt"
        worker = TranslateWorker(
            str(src), self._model("http://127.0.0.1:9/v1"), "Chinese",
            "plain_text", str(out),
            re_export=True,
        )
        events: list[str] = []
        worker.finished.connect(lambda _p: events.append("finished"))
        worker.error.connect(lambda _m: events.append("error"))
        worker.stopped.connect(lambda: events.append("stopped"))
        worker.run()
        self.assertEqual(["error", "stopped"], events)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
