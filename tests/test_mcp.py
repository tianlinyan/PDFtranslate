"""Tests for the A-① MCP bridge (``translate_app/mcp.py``)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pymupdf as fitz

from translate_app import mcp as mcp_mod

from tests._helpers import build_sample_pdf


def _mock_translate(texts, *, lang, extra_glossary=None):
    return ["T|" + t for t in texts]


class McpToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(__import__("os").environ, {
            "PDFTRANSLATE_CACHE_DIR": str(Path(self.tmp.name) / "cache"),
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        self.src = build_sample_pdf(Path(self.tmp.name) / "mcp_src.pdf", pages=2)

    def test_tools_are_callables(self):
        tools = mcp_mod.make_mcp_tools(_mock_translate)
        self.assertEqual({"translate_file", "get_doc_info", "get_structure"}, set(tools))
        self.assertTrue(all(callable(f) for f in tools.values()))

    def test_translate_file_end_to_end(self):
        tools = mcp_mod.make_mcp_tools(_mock_translate)
        out = Path(self.tmp.name) / "mcp_out.pdf"
        res = tools["translate_file"](str(self.src), output_path=str(out))
        self.assertTrue(res["ok"], res)
        self.assertTrue(out.exists())
        self.assertGreaterEqual(res["pages"], 1)

    def test_translate_file_rejects_bad_output_type(self):
        tools = mcp_mod.make_mcp_tools(_mock_translate)
        res = tools["translate_file"](str(self.src), output_type="bogus")
        self.assertFalse(res["ok"])

    def test_get_doc_info_and_structure(self):
        tools = mcp_mod.make_mcp_tools(_mock_translate)
        info = tools["get_doc_info"](str(self.src))
        self.assertTrue(info["ok"])
        self.assertGreater(info["pages"], 0)
        struct = tools["get_structure"](str(self.src))
        self.assertTrue(struct["ok"])
        self.assertIn("structure_parser", struct)

    def test_create_mcp_server_degrades_when_mcp_missing(self):
        server = mcp_mod.create_mcp_server(_mock_translate)
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.assertIsNone(server)   # no mcp -> degrade, never crash
        else:
            self.assertIsNotNone(server)  # mcp installed -> a real FastMCP


if __name__ == "__main__":
    unittest.main()
