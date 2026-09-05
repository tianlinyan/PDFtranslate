"""Tests for the preview window's pure helpers (crop / coordinate mapping)."""
import unittest

import pymupdf as fitz

from translate_app import preview
from translate_app import sidebar


def _make_png(w=200, h=200):
    """A PNG with a blue background and a red square in the middle."""
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.draw_rect(fitz.Rect(0, 0, w, h), color=None, fill=(0, 0, 1))
    page.draw_rect(fitz.Rect(50, 50, 150, 150), color=None, fill=(1, 0, 0))
    png = page.get_pixmap(dpi=72).tobytes("png")
    doc.close()
    return png


class CropRegionTest(unittest.TestCase):
    def test_crop_region_cuts_to_the_requested_box(self):
        png = _make_png()
        out = preview.crop_region(png, (60, 60, 140, 140))
        self.assertTrue(out)
        doc = fitz.open(stream=out, filetype="png")
        self.assertEqual((80.0, 80.0), (doc[0].rect.width, doc[0].rect.height))
        self.assertEqual(b"\xff\x00\x00", doc[0].get_pixmap().samples[:3])  # red
        doc.close()

    def test_crop_region_degrades_to_empty_on_zero_area(self):
        png = _make_png()
        self.assertEqual(b"", preview.crop_region(png, (100, 100, 100, 100)))

    def test_crop_region_clamps_out_of_bounds(self):
        png = _make_png(200, 200)
        out = preview.crop_region(png, (190, 190, 250, 250))  # partly outside
        self.assertTrue(out)
        doc = fitz.open(stream=out, filetype="png")
        self.assertLessEqual(doc[0].rect.width, 10.0)
        doc.close()


class ScaleRectTest(unittest.TestCase):
    def test_scales_display_coords_to_image_coords(self):
        # A display region [10, 20, 30, 50] on a 100x200 canvas with a 200x400 image.
        out = preview.scale_rect((10, 20, 30, 50), 0, 0, 100, 200, 200, 400)
        self.assertAlmostEqual(20.0, out[0])
        self.assertAlmostEqual(40.0, out[1])
        self.assertAlmostEqual(60.0, out[2])
        self.assertAlmostEqual(100.0, out[3])

    def test_offsets_the_display_origin(self):
        # The fit-rect is offset by (10, 20); a region at display (15, 30) maps
        # to image (5, 10) at 2x scale.
        out = preview.scale_rect((15, 30, 25, 40), 10, 20, 100, 100, 200, 200)
        self.assertAlmostEqual(10.0, out[0])
        self.assertAlmostEqual(20.0, out[1])


class ClampPanTest(unittest.TestCase):
    """Left-drag pan clamp: the zoomed image stays reachable / on-screen."""

    def test_no_pan_when_image_fits(self):
        # Image exactly fits the viewport (e.g. at fit-to-window) → no pan room.
        self.assertEqual((0.0, 0.0), preview.clamp_pan(100, -50, 800, 600, 800, 600))

    def test_pan_clamped_to_overshoot(self):
        # Zoomed image 1200x900 in an 800x600 viewport → pan limited to ±200 x, ±150 y.
        dx, dy = preview.clamp_pan(500, -500, 1200, 900, 800, 600)
        self.assertEqual(200.0, dx)
        self.assertEqual(-150.0, dy)

    def test_pan_within_bounds_is_preserved(self):
        # A small drag inside the allowed overshoot is kept as-is.
        self.assertEqual((50.0, -30.0), preview.clamp_pan(50, -30, 1200, 900, 800, 600))

    def test_one_axis_fits_other_overflows(self):
        # Viewport 800x600; zoomed image 800x900 → no x pan, y pans by ±150.
        self.assertEqual((0.0, 100.0), preview.clamp_pan(40, 100, 800, 900, 800, 600))


class PreviewBridgeTest(unittest.TestCase):
    """The worker↔GUI channel: payload round-trip and timeout."""

    def test_get_region_times_out_when_no_gui_responds(self):
        b = preview.PreviewBridge(timeout=0.05)
        self.assertIsNone(b.get_region(0, "source"))   # no GUI slot → None

    def test_on_region_stores_payload(self):
        b = preview.PreviewBridge()
        b.on_region(b"PNG", [0, 0, 10, 10])
        self.assertEqual({"png": b"PNG", "rect": [0, 0, 10, 10]}, b._payload)

    def test_clear_resets_payload(self):
        b = preview.PreviewBridge()
        b.on_region(b"PNG")
        b.clear()
        self.assertIsNone(b._payload)


class AnswerBridgeTest(unittest.TestCase):
    """The worker↔GUI channel for the agent's questions."""

    def test_ask_times_out_when_no_gui_answers(self):
        b = sidebar.AnswerBridge(timeout=0.05)
        self.assertIsNone(b.ask("保留该块？", ["keep", "translate"], "t"))

    def test_answer_stores_payload(self):
        b = sidebar.AnswerBridge()
        b.answer("keep", "t")
        self.assertEqual({"value": "keep", "target": "t"}, b._value)

    def test_clear_resets_value(self):
        b = sidebar.AnswerBridge()
        b.answer("keep", "t")
        b.clear()
        self.assertIsNone(b._value)


if __name__ == "__main__":
    unittest.main()
