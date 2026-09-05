"""Intermediate Representation (IR) for document-level translation (C-⑥).

The IR decouples *semantic content* from the *visual layout metadata* that the
exporter still needs: each ``IRBlock`` keeps its ``anchor`` (the extracted
:class:`pdfio.Block` carrying the bbox / fit hints) plus a ``role``
(paragraph / heading / table cell / formula / figure / caption) and a
``group_id`` that binds several anchors into one logical unit (a paragraph the
extractor split, a whole table, a formula).  Translation then happens on the IR
(``IRDoc.terms``, grouping, cross-page context) and a later re-typesetting pass
re-anchors the result onto the original layout.

This is *metadata only* and never alters the deterministic export geometry: the
exporters keep consuming ``Block`` directly, and any IR element that is not
applicable falls back to the block path.  ``build_ir`` is offline and pure.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

from . import pdfio
from .pdfio import Block, DocumentText
from .translator import _needs_translation


@dataclass
class IRBlock:
    """A semantic translation unit anchored to one extracted ``Block``."""

    anchor: Block                  # the visual/metadata anchor (bbox, fit hints)
    text: str                      # source text
    role: str = "text"             # text / heading / table_cell / formula / figure / caption / note
    level: int = 0                 # heading level (1..6); 0 = not a heading
    group_id: int = 0              # logical unit (paragraph/table/formula) shared by >1 anchor
    table_ref: int = 0             # index into IRPage.tables; 0 = not in a semantic table
    style: dict = field(default_factory=dict)   # {size, align, bold, color}
    src_id: int = 0                # flat block index (fallback to the geometry path)


@dataclass
class IRPage:
    """One page of the IR."""

    page: int
    blocks: list[IRBlock] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)   # semantic tables (rows/cols/cells/bbox)
    reading_order: list[int] = field(default_factory=list)  # src_id sequence


@dataclass
class IRDoc:
    """The document-level IR."""

    title: str = ""
    lang: str = ""
    parser: str = ""               # structure backend; "" = geometry
    pages: list[IRPage] = field(default_factory=list)
    terms: dict = field(default_factory=dict)     # document-level glossary (Stage 2)
    block_count: int = 0


#: Roles that are structurally protected (never translated as prose).
_STRUCTURAL_ROLES = ("formula", "figure")
#: Tunable knobs for :func:`infer_terms` (document-level terminology extraction).
_INFER_MAX_TERMS = 80    # cap on candidate terms translated once
_INFER_MIN_FREQ = 2      # a term must appear at least this many times to be a candidate


def is_structural_role(role: str) -> bool:
    """True when ``role`` marks structurally-protected content (formula/figure)."""
    return role in _STRUCTURAL_ROLES


def _style_of(block: Block) -> dict:
    return {"size": block.size, "align": block.align,
            "bold": block.bold, "color": block.color}


def _role_of(src_id: int, structure: pdfio.PageStructure | None) -> tuple[str, int]:
    """``(role, level)`` for a block, derived from the structure when present.

    Joins by *flat index membership* in an element's ``block_indices`` (the same the
    ``read_page`` tool uses), so the two readers never disagree on a block's kind.
    Without a structure the role is ordinary ``text``.
    """
    if structure is None:
        return "text", 0
    for el in structure.elements:
        if src_id in el.get("block_indices", []):
            return str(el.get("kind", "text")), int(el.get("level", 0) or 0)
    return "text", 0


def _table_index_of(src_id: int, structure: pdfio.PageStructure | None, page: int) -> int:
    """The index into ``structure.tables`` claiming ``src_id`` (0 = none)."""
    if structure is None:
        return 0
    for ti, tbl in enumerate(structure.tables):
        if src_id in tbl.block_ref:
            return ti + 1
    return 0


def build_ir(doc: DocumentText, *, lang: str = "", parser: str | None = None) -> IRDoc:
    """Build an ``IRDoc`` from a ``DocumentText`` (optionally with its structure).

    Offline and pure.  Each page walks its blocks, assigns roles from the page
    structure (when present), binds cells to semantic tables, and groups anchors
    into logical units (a table = one group; a run of same-style prose = one
    group).  ``terms`` stays empty — document-level terminology extraction is
    Stage 2 (``translate_ir``).
    """
    ir = IRDoc(title=doc.title, lang=lang or pdfio.detect_language(doc.blocks),
               parser=parser if parser is not None else doc.structure_parser)
    group_counter = 0
    table_group: dict[tuple, int] = {}   # (page, table_ref) -> unique group id

    for p, blocks in enumerate(doc.pages):
        structure = doc.page_structure[p] if p < len(doc.page_structure) else None
        ipage = IRPage(page=p)
        offset = sum(len(pg) for pg in doc.pages[:p])
        tables = []
        # First pass: attach semantic tables (advisory, read-only).
        if structure is not None:
            for tbl in structure.tables:
                tables.append({"rows": tbl.rows, "cols": tbl.cols,
                               "bbox": list(tbl.bbox), "cells": tbl.cells,
                               "block_ref": sorted(tbl.block_ref)})
        ipage.tables = tables

        prev_style: dict | None = None
        prev_group = 0
        last_group_was_table = False
        for i, b in enumerate(blocks):
            src_id = offset + i
            role, level = _role_of(src_id, structure)
            table_ref = _table_index_of(src_id, structure, p)
            style = _style_of(b)
            in_table = table_ref > 0 or bool(getattr(b, "in_table", False))

            if in_table:
                # Every cell of a table shares that table's group; group ids are
                # unique per (page, table) so two pages' tables never collide.
                key = (p, table_ref) if table_ref else (p, id(b))
                if key not in table_group:
                    group_counter += 1
                    table_group[key] = group_counter
                group_id = table_group[key]
                last_group_was_table = True
                if role in ("text", "table"):
                    role = "table_cell"
            else:
                # Group a run of same-style prose (a paragraph the extractor split).
                if (prev_style is not None and role == "text" and not last_group_was_table
                        and _same_style(prev_style, style)):
                    group_id = prev_group
                else:
                    group_counter += 1
                    group_id = group_counter
                last_group_was_table = False

            ipage.blocks.append(IRBlock(
                anchor=b, text=str(b.text), role=role, level=level,
                group_id=group_id, table_ref=table_ref, style=style, src_id=src_id))
            ipage.reading_order.append(src_id)
            prev_style = style
            prev_group = group_id

        ir.pages.append(ipage)

    ir.block_count = len(doc.blocks)
    return ir


def _same_style(a: dict, b: dict) -> bool:
    """True when two blocks share the layout style (a paragraph run)."""
    return (a.get("size") == b.get("size") and a.get("bold") == b.get("bold")
            and a.get("align") == b.get("align") and a.get("color") == b.get("color"))


def structural_groups(ir: IRDoc) -> list[list[IRBlock]]:
    """Group blocks by ``group_id`` for IR-level translation (C-⑥ Stage 2).

    Returns lists of :class:`IRBlock` that form one logical unit, in reading order
    (prose runs, tables, formulas).  A single-anchor group is still a group.
    """
    by_group: dict[int, list[IRBlock]] = {}
    order: list[int] = []
    for ipage in ir.pages:
        for blk in ipage.blocks:
            if blk.group_id not in by_group:
                by_group[blk.group_id] = []
                order.append(blk.group_id)
            by_group[blk.group_id].append(blk)
    return [by_group[g] for g in order]


def set_terms(ir: IRDoc, glossary: dict[str, str]) -> IRDoc:
    """Store a document-level glossary on the IR (C-⑥ Stage 2).

    ``translate_ir`` passes this through as ``extra_glossary`` so every group in
    the document uses the same terminology (cross-page consistency).
    """
    ir.terms = {str(k): str(v) for k, v in (glossary or {}).items() if str(k).strip()}
    return ir


def _is_verbatim(block: Block) -> bool:
    """True when a block is expected to stay byte-identical (engine-skipped/numeric)."""
    return not _needs_translation(str(block.text)) or pdfio._is_numeric_cell(str(block.text))


#: A ``translate_fn`` maps a list of source texts to a same-length list of target
#: texts, honouring a glossary: ``fn(texts, *, lang, extra_glossary) -> list[str]``.
#: The default bound by :func:`make_ir_translate_fn` calls
#: ``translation_engine.translate_blocks``; tests inject a mock.
def infer_terms(ir: IRDoc, *, max_terms: int | None = None) -> list[str]:
    """Conservative document-level terminology candidates (C-⑥).

    Source-only (no translation): call :func:`translate_ir` with ``infer=True`` to
    translate them once and inject the result as ``IRDoc.terms``, giving every
    occurrence the same target across pages.  Candidates are CJK phrases (2–4 chars)
    and TitleCase / ALL-CAPS Latin words that appear >=2 times, capped at
    ``max_terms``.  Deliberately conservative: a noisy candidate is just translated
    once extra, never a correctness risk.
    """
    cjk: Counter[str] = Counter()
    latin: Counter[str] = Counter()
    for ipage in ir.pages:
        for b in ipage.blocks:
            if is_structural_role(b.role) or _is_verbatim(b.anchor):
                continue
            t = str(b.text)
            for m in re.finditer(r"[\u4e00-\u9fff]+", t):
                run = m.group(0)
                if 2 <= len(run) <= 4:   # a term is short; a long CJK run is prose
                    cjk[run] += 1
            for w in re.findall(r"\b[A-Z][a-z]{2,}\b|\b[A-Z]{3,}\b", t):
                latin[w] += 1
    cands = [t for t, n in cjk.items() if n >= _INFER_MIN_FREQ]
    cands += [w for w, n in latin.items() if n >= _INFER_MIN_FREQ]
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= (max_terms or _INFER_MAX_TERMS):
            break
    return out


def translate_ir(
    ir: IRDoc,
    translate_fn: Callable[..., Sequence[str]],
    *,
    lang: str,
    extra_glossary: dict[str, str] | None = None,
    log: Callable[[str], None] | None = None,
    infer: bool = False,
) -> dict[int, str]:
    """IR-level translation → ``{src_id: translated_text}``.

    Structural blocks (``formula`` / ``figure``) and verbatim blocks (numeric /
    engine-skipped) are kept as their source; everything else goes through
    ``translate_fn``, which receives every translatable text in one call so the
    whole document shares one glossary (``IRDoc.terms`` unless overridden) for
    terminology consistency.  ``infer=True`` (and no ``ir.terms`` / not overridden)
    first extracts candidate terms via :func:`infer_terms`, translates them once and
    injects the result as the doc glossary — so cross-page terminology is pinned
    before the main pass.  The per-``src_id`` output aligns with the flat block
    list, so it drops straight into the existing ``out_doc`` / eval harness.
    """
    glossary = dict(ir.terms) if extra_glossary is None else dict(extra_glossary)
    blocks = [b for ipage in ir.pages for b in ipage.blocks]
    if infer and not glossary:
        terms = infer_terms(ir)
        if terms:
            got = translate_fn(terms, lang=lang, extra_glossary={})
            if len(got) != len(terms):
                if log:
                    log(f"[ir] 术语批量返回 {len(got)} 条，与 {len(terms)} 条候选不符，跳过术语注入。")
            else:
                # Only keep a term whose translation actually differs (a model that
                # fails and echoes the source would otherwise pin `s -> s`, suppressing
                # that term's normal translation in the main batch).
                glossary = {s: str(t) for s, t in zip(terms, got)
                            if str(t).strip() and str(t).strip() != s}
                ir.terms = glossary
    requests: list[tuple[int, str]] = []
    out: dict[int, str] = {}
    for b in blocks:
        if is_structural_role(b.role) or _is_verbatim(b.anchor):
            out[b.src_id] = b.text      # formula/figure/numeric → keep source
        else:
            requests.append((b.src_id, b.text))
    if not requests:
        return out
    got = translate_fn([t for _, t in requests], lang=lang, extra_glossary=glossary)
    if len(got) != len(requests):
        raise ValueError(
            f"translate_fn 返回 {len(got)} 条译文，与 {len(requests)} 条请求不一致")
    for (src_id, _), t in zip(requests, got):
        out[src_id] = str(t)
    return out


class _BoundTranslate:
    """A callable ``translate_fn`` that also surfaces the last batch's ``errors``.

    ``translate_ir`` only sees a callable, but the worker needs to know which blocks
    failed every retry (so it can warn instead of silently exporting the source).
    """

    def __init__(self, fn: Callable[..., Sequence[str]], errors: list):
        self._fn = fn
        self._errors = errors

    def __call__(self, texts: Sequence[str], *, lang: str,
                 extra_glossary: dict[str, str] | None = None):
        return self._fn(texts, lang=lang, extra_glossary=extra_glossary)

    @property
    def last_errors(self) -> list:
        return list(self._errors)


def make_ir_translate_fn(engine, *, doc_path: "Path | None" = None,
                         log: Callable[[str], None] | None = None,
                         cancel: Callable[[], bool] | None = None,
                         on_progress: Callable[[int, int], None] | None = None,
                         resume: bool = True,
                         keep_original: "set[int] | None" = None):
    """Bind a real ``TranslationEngine`` as a ``translate_fn`` for :func:`translate_ir`.

    The returned callable does one ``translate_blocks`` for all texts in a single
    batch (with ``extra_glossary`` merged over the on-disk glossary), sharing the
    engine's client / concurrency / cache.  Unlike the deterministic path it used to
    drop ``cancel`` / ``on_progress`` / ``errors`` / ``resume`` / ``keep_original``
    — this now forwards them so a user cancellation interrupts the in-flight request
    (not just after a full batch), progress is reported, and failed batches surface
    via :attr:`_BoundTranslate.last_errors`.
    """
    errors: list = []

    def _translate(texts: Sequence[str], *, lang: str,
                   extra_glossary: dict[str, str] | None = None):
        result = engine.translate_blocks(
            list(texts), lang, log=log or (lambda m: None),
            on_progress=on_progress, cancel=cancel, doc_path=doc_path,
            resume=resume, keep_original=keep_original, extra_glossary=extra_glossary,
        )
        errors[:] = list(getattr(result, "errors", None) or [])
        return list(result.translated)

    return _BoundTranslate(_translate, errors)


def per_page_from_ir(ir: IRDoc, translated: dict[int, str]) -> tuple[list[list[Block]], list[list[str]]]:
    """Rebuild ``(pages, per_page)`` (block anchors + translations) from the IR.

    This is the bridge from the IR back to the existing exporter: every block's
    anchor (its ``Block``) is reused as the page's block list, and the IR-level
    translation (keyed by ``src_id``) becomes that block's translation.  A block
    with no entry (e.g. a structural / numeric block) falls back to its source.
    """
    pages: list[list[Block]] = []
    per_page: list[list[str]] = []
    for ipage in ir.pages:
        anchors = [b.anchor for b in ipage.blocks]
        texts = [str(translated.get(b.src_id, b.text)) for b in ipage.blocks]
        pages.append(anchors)
        per_page.append(texts)
    return pages, per_page


def save_ir(
    src_path: str | Path,
    out_path: str | Path,
    ir: IRDoc,
    translated: dict[int, str],
    *,
    lang: str,
    mode: str = "translated_pdf",
    log: Callable[[str], None] | None = None,
) -> str:
    """Re-export the IR translation back to a PDF (C-⑥ Stage 3).

    The IR-level ``translated`` (``{src_id: text}``) is mapped back onto the
    original anchors and handed to the *existing* exporter
    (:func:`pdfio.save_translated_pdf` / :func:`pdfio.save_interleaved_pdf`),
    which already does the adaptive re-anchoring — table row expansion, scan
    band-fit, fit-to-column — so the IR path inherits every carefully-tuned fit
    rule instead of re-implementing a typesetter.  ``mode`` is one of the
    ``OUTPUT_TYPES`` PDF keys (``translated_pdf`` | ``bilingual_pdf``).
    """
    if mode not in ("translated_pdf", "bilingual_pdf"):
        raise ValueError(f"save_ir 仅支持 PDF 输出，收到 {mode!r}")
    pages, per_page = per_page_from_ir(ir, translated)
    if mode == "bilingual_pdf":
        pdfio.save_interleaved_pdf(str(src_path), per_page, str(out_path), lang, pages=pages)
    else:
        pdfio.save_translated_pdf(str(src_path), pages, per_page, str(out_path), lang, log=log)
    return str(out_path)
