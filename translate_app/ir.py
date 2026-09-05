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

from dataclasses import dataclass, field
from typing import Sequence

from . import pdfio
from .pdfio import Block, DocumentText


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


def is_structural_role(role: str) -> bool:
    """True when ``role`` marks structurally-protected content (formula/figure)."""
    return role in _STRUCTURAL_ROLES


def _style_of(block: Block) -> dict:
    return {"size": block.size, "align": block.align,
            "bold": block.bold, "color": block.color}


def _role_of(block: Block, structure: pdfio.PageStructure | None) -> tuple[str, int]:
    """``(role, level)`` for a block, derived from the structure when present.

    Without a structure the role is an ordinary ``text`` (a heading is *not*
    guessed here — that stays heuristic and risk-free; the structure layer is the
    authoritative source for formula / figure / caption).
    """
    if structure is None:
        return "text", 0
    # Join by flat index: the block's role comes from the element that claimed it.
    for el in structure.elements:
        if block is not None:
            # ``block`` carries its own page; match via the element's block_indices
            # is done by src_id below (block itself is the anchor).  We re-derive
            # role from the element whose bbox holds this block's centre.
            if pdfio._point_in_bbox(block, el["bbox"]):
                return str(el.get("kind", "text")), int(el.get("level", 0) or 0)
    return "text", 0


def _table_index_of(block: Block, structure: pdfio.PageStructure | None, page: int) -> int:
    """The index into ``structure.tables`` claiming ``block`` (0 = none)."""
    if structure is None:
        return 0
    for ti, tbl in enumerate(structure.tables):
        # Match by the block's centre against the table bbox.
        if pdfio._point_in_bbox(block, tbl.bbox):
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
            role, level = _role_of(b, structure)
            table_ref = _table_index_of(b, structure, p)
            style = _style_of(b)
            in_table = table_ref > 0

            if in_table:
                # Every cell of a table shares that table's group (a table is one unit).
                group_id = 1000 + table_ref  # table groups stay above prose groups
                group_counter = max(group_counter, group_id)
                last_group_was_table = True
                if role == "text":
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
