"""Tests for the M2/M3 AI-backend gates (parser + OCR selection, with greyout fallback).

These verify the *wiring* (which backend is selected for a given parser/OCR request and
what ``structure_parser`` reports) without requiring a real DocLayout-YOLO model, a
VLM OCR backend, or a network call.  ``extract_structured`` is patched so no PDF is
actually opened; caches are untouched.
"""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from translate_app import pdfio


class _Capture:
    def __init__(self, structure_fn, parser):
        self.structure_fn = structure_fn
        self.parser = parser
        self.structure_parser = parser  # what the honest-override may rewrite to "geo"


def _capture_structured(path, structure_fn, *, parser, **_kw):
    return _Capture(structure_fn, parser)


class ExtractStructuredBackendTest(unittest.TestCase):
    def test_geo_default_uses_geometric_and_reports_geo(self):
        with patch.object(pdfio, "extract_structured", side_effect=_capture_structured):
            res = pdfio.extract_document_structured("x.pdf")
        self.assertEqual("geo", res.structure_parser)

    def test_doclayout_not_installed_degrades_and_reports_geo(self):
        # Simulate DocLayout-YOLO absent so ``make_doclayout_structure_fn`` degrades to
        # the geometric backend and ``structure_parser`` is honest about it.
        with patch.dict(sys.modules, {"doclayout_yolo": None}), \
             patch.object(pdfio, "extract_structured", side_effect=_capture_structured):
            res = pdfio.extract_document_structured("x.pdf", parser="doclayout")
        self.assertEqual("geo", res.structure_parser)

    def test_doclayout_importable_keeps_label_when_regions_produced(self):
        # When DocLayout-YOLO is importable AND real regions flowed, the label stays
        # ``doclayout`` (honest "actually used").  ``extract_structured`` is stubbed, so
        # mark ``_used_doclayout`` on the function to simulate that regions were produced.
        def sf(*a):
            return []
        sf._used_doclayout = True
        with patch.object(pdfio, "make_doclayout_structure_fn", return_value=sf) as dl, \
             patch.dict("sys.modules", {"doclayout_yolo": SimpleNamespace(YOLOv10=object)}), \
             patch.object(pdfio, "extract_structured", side_effect=_capture_structured):
            res = pdfio.extract_document_structured("x.pdf", parser="doclayout")
        dl.assert_called_once()
        self.assertEqual("doclayout", res.structure_parser)

    def test_doclayout_importable_but_no_real_regions_reports_geo(self):
        # Importable but the model produced no region (download/predict failure) → the
        # honest ``structure_parser`` is "geo", so "包能导入" is NOT "真正用了 DocLayout".
        def sf(*a):  # never sets ``_used_doclayout`` → degraded
            return []
        with patch.object(pdfio, "make_doclayout_structure_fn", return_value=sf), \
             patch.dict("sys.modules", {"doclayout_yolo": SimpleNamespace(YOLOv10=object)}), \
             patch.object(pdfio, "extract_structured", side_effect=_capture_structured):
            res = pdfio.extract_document_structured("x.pdf", parser="doclayout")
        self.assertEqual("geo", res.structure_parser)


class SelectOcrFnTest(unittest.TestCase):
    def test_default_backends_return_none(self):
        self.assertIsNone(pdfio.select_ocr_fn(None))
        self.assertIsNone(pdfio.select_ocr_fn(""))
        self.assertIsNone(pdfio.select_ocr_fn("geo"))
        self.assertIsNone(pdfio.select_ocr_fn("rapidocr"))

    def test_vlm_unregistered_returns_none(self):
        self.assertIsNone(pdfio.select_ocr_fn("vlm"))

    def test_vlm_registered_gate_returns_backend(self):
        with patch.object(pdfio, "make_vlm_ocr_fn", return_value="VLM_FAKE") as mv:
            self.assertEqual("VLM_FAKE", pdfio.select_ocr_fn("vlm"))
        mv.assert_called_once_with(name="vlm", log=None)


class _Tensor:
    """A minimal stand-in for a torch tensor exposing ``.cpu().numpy()``."""

    def __init__(self, data):
        self.a = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __len__(self):
        return len(self.a)


class DoclayoutParseTest(unittest.TestCase):
    def _result(self, xyxy, cls, names):
        return SimpleNamespace(
            boxes=SimpleNamespace(xyxy=_Tensor(xyxy), cls=_Tensor(cls)),
            names=names,
        )

    def test_parses_and_converts_pixels_to_pdf_points(self):
        # dpi=150 → scale = 72/150 = 0.48; a figure at pixels [10,20,210,220].
        r = self._result([[10, 20, 210, 220]], [2],
                         {0: "text", 1: "table", 2: "figure", 3: "equation"})
        regions = pdfio._doclayout_parse_results(r, r.names, dpi=150)
        self.assertEqual(regions, [{"kind": "figure",
                                    "bbox": [4.8, 9.6, 100.8, 105.6]}])

    def test_kind_mapping_normalises_categories(self):
        r = self._result([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]], [1, 3, 9],
                         {0: "text", 1: "equation", 3: "unknown_cat", 9: "table_caption"})
        regions = pdfio._doclayout_parse_results(r, r.names, dpi=150)
        kinds = [x["kind"] for x in regions]
        self.assertEqual(kinds, ["formula", "text", "caption"])  # unknown → text

    def test_empty_detection_returns_empty(self):
        # A real "no detections" result has a (0,4) boxes tensor.
        self.assertEqual([], pdfio._doclayout_parse_results(
            self._result(np.zeros((0, 4)), [], {0: "table"}), {0: "table"}, dpi=150))

    def test_missing_boxes_returns_empty(self):
        r = SimpleNamespace(boxes=SimpleNamespace(xyxy=None), names={})
        self.assertEqual([], pdfio._doclayout_parse_results(r, {}, dpi=150))

    def test_device_default_cpu_and_env_override(self):
        self.assertEqual("cpu", pdfio._doclayout_device())
        with patch.dict("os.environ", {"PDFTRANSLATE_DOCLAYOUT_DEVICE": "cuda:0"}):
            self.assertEqual("cuda:0", pdfio._doclayout_device())


class DocLayoutFactoryBranchTest(unittest.TestCase):
    def test_doclayout_importable_returns_doclayout_structure_fn(self):
        # Fake the ``doclayout_yolo`` package so ``make_doclayout_structure_fn`` takes the
        # DocLayout branch (instead of degrading to geometric), then verify the produced
        # ``structure_fn`` returns the fused regions.  ``_resolve_doclayout_model`` is
        # stubbed too: without it the factory would try to download the model (and, with
        # no ``huggingface_hub`` installed, silently degrade to geometric — making this
        # test depend on the machine's optional deps / HF cache instead of the branch).
        with patch.dict(sys.modules, {
                "doclayout_yolo": SimpleNamespace(YOLOv10=lambda *_a, **_k: object()),
            }), \
            patch.object(pdfio, "_resolve_doclayout_model",
                         return_value="fake-doclayout.onnx"), \
            patch.object(pdfio, "_render_page_png", return_value=b"png"), \
            patch.object(pdfio, "_doclayout_regions",
                         return_value=[{"kind": "table", "bbox": [0, 0, 10, 10]}]) as dr:
            sf = pdfio.make_doclayout_structure_fn()
            out = sf(0, object(), [])
        self.assertEqual([{"kind": "table", "bbox": [0, 0, 10, 10]}], out)
        dr.assert_called_once()


if __name__ == "__main__":
    unittest.main()
