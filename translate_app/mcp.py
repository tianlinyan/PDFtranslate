"""MCP server exposing the PDFtranslate engine as a callable tool (A-①).

Provides a *headless* bridge so any MCP-capable agent (Claude Desktop, an IDE
agent, other document tools) can translate a PDF without the GUI:

``translate_file(path, target_language, output_type, output_path)``
    extract → IR → translate (via an injected ``translate_fn`` bound to a real
    ``TranslationEngine``) → save; returns the output path + counts.

The tool logic lives in :func:`make_mcp_tools` (plain callables, unit-testable with
a mock ``translate_fn``); :func:`create_mcp_server` lazily imports ``mcp`` and wraps
them.  When ``mcp`` is not installed it returns ``None`` (the caller reports "MCP
未安装"), so importing this module never fails and nothing dead is exposed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence


def make_mcp_tools(translate_fn: Callable[..., Sequence[str]]) -> dict[str, Callable]:
    """The headless PDFtranslate tools exposed over MCP (plain callables).

    ``translate_fn`` is bound by the caller (e.g. ``ir.make_ir_translate_fn`` around
    a real ``TranslationEngine``) and receives ``(texts, *, lang, extra_glossary)``.
    """
    from . import pdfio
    from .ir_pipeline import run_ir_pipeline

    def translate_file(path: str, target_language: str = "English",
                       output_type: str = "translated_pdf",
                       output_path: str | None = None) -> dict:
        if output_type not in ("translated_pdf", "bilingual_pdf"):
            return {"ok": False, "error": f"未知输出格式：{output_type!r}"}
        try:
            out, doc_ir, translated = run_ir_pipeline(
                Path(path), lang=target_language, translate_fn=translate_fn,
                mode=output_type, out_path=output_path)
        except Exception as exc:  # noqa: BLE001 — fail-closed, never crash
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "output": str(out), "block_count": doc_ir.block_count,
                "pages": len(doc_ir.pages), "lang": target_language}

    def get_doc_info(path: str) -> dict:
        try:
            dt = pdfio.extract_document_text(Path(path), ocr=False, log=lambda m: None)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, **pdfio.get_doc_info(dt)}

    def get_structure(path: str) -> dict:
        try:
            dt = pdfio.extract_document_structured(Path(path), parser="geo")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "structure_parser": dt.structure_parser,
                "summary": pdfio.get_structure_summary(dt)}

    return {"translate_file": translate_file, "get_doc_info": get_doc_info,
            "get_structure": get_structure}


def create_mcp_server(translate_fn: Callable[..., Sequence[str]], *,
                      name: str = "pdftranslate", version: str = "0.3.8"):
    """Wrap :func:`make_mcp_tools` in an MCP ``FastMCP`` server (A-①).

    Lazily imports ``mcp``; when unavailable (or the ``mcp`` package is missing),
    returns ``None`` so the caller can report "MCP 未安装" instead of failing.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        return None
    mcp = FastMCP(name, version=version)
    for tool_name, fn in make_mcp_tools(translate_fn).items():
        mcp.add_tool(fn, name=tool_name)
    return mcp
