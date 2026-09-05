"""End-to-end IR pipeline: extract (optionally with structure) → IR → translate → export.

This ties the C-⑥ pieces into one callable so the whole IR loop can be run and
tested headlessly without touching ``worker.py`` / ``DocumentSession`` (which live
in separate WIP).  A ``translate_fn`` is injected so the pipeline is offline-testable
with a mock and, in production, is bound by ``ir.make_ir_translate_fn`` around a real
``TranslationEngine``.

The returned ``(out_path, ir_doc, translated)`` gives callers the produced PDF, the
IR (with ``terms`` for later re-export / audit), and the per-block translation map —
enough to feed ``worker``'s ``out_doc`` / overlay and the eval harness later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from . import ir as ir_mod
from . import pdfio


def run_ir_pipeline(
    src: str | Path,
    *,
    lang: str,
    translate_fn: Callable[..., Sequence[str]],
    mode: str = "translated_pdf",
    out_path: str | Path | None = None,
    structure_fn: Callable | None = None,
    parser: str = "structure",
    log: Callable[[str], None] | None = None,
) -> tuple[Path, "ir_mod.IRDoc", dict[int, str]]:
    """Run the whole IR pipeline and return ``(out_path, ir, translated)``.

    ``structure_fn`` (a B-④ backend) is optional: when given the document is parsed
    semantically (formula / figure / table roles), otherwise the geometry path is
    used and every block is treated as prose.  ``translate_fn`` receives
    ``(texts, *, lang, extra_glossary)`` and returns the same number of translations;
    pass a real engine's ``ir.make_ir_translate_fn`` for production.
    """
    if structure_fn is not None:
        dt = pdfio.extract_structured(str(src), structure_fn, parser=parser, log=log)
    else:
        dt = pdfio.extract_document_text(str(src), ocr=False, log=log)

    doc_ir = ir_mod.build_ir(dt, lang=lang)
    translated = ir_mod.translate_ir(doc_ir, translate_fn, lang=lang, log=log)

    if out_path is None:
        out_path = Path(str(src)).with_name(f"{Path(str(src)).stem}-ir.pdf")
    ir_mod.save_ir(str(src), str(out_path), doc_ir, translated, lang=lang, mode=mode, log=log)
    return Path(out_path), doc_ir, translated
