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

import os
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
        prev_verbatim = False
        for i, b in enumerate(blocks):
            src_id = offset + i
            role, level = _role_of(src_id, structure)
            table_ref = _table_index_of(src_id, structure, p)
            style = _style_of(b)
            in_table = table_ref > 0 or bool(getattr(b, "in_table", False))
            # A verbatim block (a pure figure / numeric cell) is a hard boundary of a
            # prose run: ``translate_ir`` keeps it out of the request, so merging the
            # blocks around it handed the model a sentence with the amount missing
            # ("...revenue of million yuan..."), inviting it to invent or move one.
            verbatim = _is_verbatim(b)

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
                        and not verbatim and not prev_verbatim
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
            prev_verbatim = verbatim

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


# --------------------------------------------------------------------------- #
# Stage 4 — prose grouping: translate the *paragraph*, re-anchor onto its blocks
# --------------------------------------------------------------------------- #

#: A paragraph is only joined while the merged text stays below this length.
#: Beyond it the "group" is a section, not a paragraph: it would blow the batch
#: budget and make the back-distribution meaningless (one line's bbox cannot hold
#: a third of a page of text).
_MAX_PROSE_UNIT = 2400

#: Sentence-ending punctuation — the preferred cut point when re-splitting a unit.
_SENT_END = "。！？；.!?;"
#: Characters that must never end up alone at a cut (a hyphen split reads as a
#: broken word, a leading punctuation fragment as noise).
_BAD_EDGE = "-－—（(《“\"'，、。．．：:；;）)》”\""

#: Minimum characters a fragment keeps after the split (a fragment shorter than
#: this is a rounding artefact, not a piece of the paragraph).
_MIN_PIECE = 1


def _group_prose_enabled() -> bool:
    """Env kill-switch for prose grouping (``PDFTRANSLATE_IR_GROUP=0``)."""
    return (os.environ.get("PDFTRANSLATE_IR_GROUP") or "1") != "0"


def _is_cjk(ch: str) -> bool:
    """True for a CJK ideograph (a line break between two of them carries no space)."""
    return bool(ch) and "\u4e00" <= ch <= "\u9fff"


def _joinable(b: IRBlock) -> bool:
    """True when ``b`` may be merged with its same-group neighbours (plain prose).

    Headings, captions, table cells, scanned (OCR) blocks, chart labels and
    verbatim name cells all keep their own translation unit: their geometry is
    pinned (a cell, a band, a node box), so giving the model a merged paragraph
    would produce a text it cannot place back.

    A list / enumeration entry (bullet, ``1.``, ``（一）``, ``Label:``, ``联系地址：``)
    is also excluded — those short items translate to a *different* length than
    their source, so re-splitting a merged paragraph back onto them by length
    drifts the numbers / addresses across the item boundary (a real regression
    measured on the annual-report A/B: 168 / 126 / 317500 landed on the wrong
    block).  A short entry must be translated on its own.
    """
    a = b.anchor
    return (b.role == "text" and b.group_id > 0
            and not getattr(a, "in_table", False)
            and not getattr(a, "ocr", False)
            and not getattr(a, "is_chart", False)
            and not getattr(a, "keep_original", False)
            and not pdfio._is_entry(str(b.text)))


def join_texts(texts: Sequence[str]) -> str:
    """Join extracted fragments into one natural source string.

    A CJK line break carries no space — joining with one injects a gap the model
    may echo into the output — while a Latin one does.  So the separator depends
    on the two characters the join lands between.
    """
    out = ""
    for raw in texts:
        t = str(raw)
        if not out:
            out = t
            continue
        if not t:
            continue
        out += "" if (_is_cjk(out[-1]) and _is_cjk(t[0])) else " "
        out += t
    return out


def prose_units(blocks: Sequence[IRBlock]) -> list[list[IRBlock]]:
    """Split translatable blocks into translation units (C-⑥ Stage 4).

    Consecutive blocks that build one logical paragraph — the same ``group_id``
    :func:`build_ir` assigns to a run of same-style prose — become a *single* unit,
    so the model sees the whole paragraph instead of one line at a time (articles,
    number agreement, anaphora and "如下表所示" style connectives only resolve with
    the sentence in front of it).  Anything that must stay alone (see
    :func:`_joinable`) is a unit of its own.

    Units stay in flat block order and every block lands in exactly one unit, so
    the result maps back onto the block indices the exporter, the overlay and the
    audit tools key on.
    """
    units: list[list[IRBlock]] = []
    for b in blocks:
        cur = units[-1] if units else None
        if (cur is not None and _group_prose_enabled() and _joinable(b)
                and _joinable(cur[0]) and b.group_id == cur[0].group_id
                and len(join_texts([x.text for x in cur] + [b.text])) <= _MAX_PROSE_UNIT):
            cur.append(b)
        else:
            units.append([b])
    return units


def _number_split(t: str, i: int) -> bool:
    """True when cutting ``t`` at ``i`` would break a numeric/token atom.

    ``1,234.56`` split across two blocks is a corruption no exporter can undo —
    the two halves land in different bboxes.  Same rule the drawing-time
    ``pdfio._num_atom`` enforces, applied to the back-distribution.
    """
    if i <= 0 or i >= len(t):
        return False
    prev, nxt = t[i - 1], t[i]
    if prev.isdigit() and nxt.isdigit():
        return True
    if prev.isdigit() and nxt in ".,%/‰－-":
        return True
    if prev in ".,%/－-" and nxt.isdigit():
        return True
    return False


def _cut_cost(t: str, i: int, target: int) -> tuple[int, int]:
    """Cost of cutting ``t`` at ``i``: (total cost, distance to the ideal offset).

    Proximity to the proportional target comes first, but a cut that is a few
    characters off *at a sentence end* beats one that lands exactly on target in
    the middle of a word — the pieces are drawn into separate boxes, so a clean
    boundary matters more than an exact one.
    """
    prev = t[i - 1]
    nxt = t[i]
    cost = abs(i - target)
    if prev in _SENT_END:
        cost -= 6
    if prev.isspace() or nxt.isspace():
        cost -= 3
    if _is_cjk(prev) and _is_cjk(nxt):
        cost -= 1                      # CJK can break anywhere; no penalty needed
    elif prev.isalpha() and nxt.isalpha():
        cost += 6                      # never prefer cutting inside a Latin word
    if prev in _BAD_EDGE or nxt in _BAD_EDGE:
        cost += 8                      # never prefer a dangling hyphen / punctuation
    return (cost, abs(i - target))


def _best_cut(t: str, target: int, lo: int, hi: int) -> int:
    """The cheapest cut index in ``[lo, hi]`` that does not break a number."""
    lo = max(1, lo)
    hi = min(hi, len(t) - 1)
    if lo > hi:
        return min(max(1, target), max(1, len(t) - 1))
    pool = [i for i in range(lo, hi + 1) if not _number_split(t, i)] or list(range(lo, hi + 1))
    return min(pool, key=lambda i: _cut_cost(t, i, target))


def _norm_ws(s: str) -> str:
    """Whitespace-insensitive comparison key."""
    return "".join(s.split())


def split_translation(sources: Sequence[str], translated: str) -> list[str]:
    """Back-distribute one unit's translation onto its source fragments.

    The paragraph translation is cut proportionally to each fragment's source
    length, snapped to the nearest clean boundary (sentence end > space > any
    position) and never inside a number, so every block keeps roughly its own
    share of the text and the existing per-block fit/export rules still apply.
    Invariants: the pieces are non-empty (a fragment that could not get any text keeps
    its SOURCE — never a blank, which would export as an empty box), their concatenation
    loses no content of ``translated``, and a failed batch that echoed the source hands
    the fragments back unchanged (re-splitting the source would scramble the original
    line breaks).
    """
    src = [str(s) for s in sources]
    k = len(src)
    t = str(translated)
    if k == 0:
        return []
    if k == 1:
        return [t]
    if not t.strip():
        return src                      # never blank a paragraph because the model gave up
    if _norm_ws(t) == _norm_ws(join_texts(src)):
        return src                      # the batch failed and echoed the source
    n = len(t)
    if n < k:
        # Too short to give every fragment something: keep the whole translation in the
        # longest fragment and leave the others with their SOURCE text.  An empty
        # fragment exports as a blank box (the exporter redacts the original and draws
        # nothing) — i.e. silently lost content — while the source text stays visible
        # and is caught by ``check_residual``.
        longest = max(range(k), key=lambda idx: len(src[idx]))
        out = list(src)
        out[longest] = t
        return out
    weights = [max(1, len(s)) for s in src]
    total = sum(weights)
    cuts: list[int] = [0]
    for idx in range(1, k):
        target = int(round(n * sum(weights[:idx]) / total))
        lo = cuts[-1] + _MIN_PIECE
        hi = n - _MIN_PIECE * (k - idx)          # leave room for the remaining pieces
        cuts.append(_best_cut(t, target, lo, hi))
    cuts.append(n)
    raw = [t[cuts[i]:cuts[i + 1]] for i in range(k)]
    # A whitespace-only slice would blank its block: fall back to that block's source
    # (same reasoning as the n < k branch — never lose content to a blank).
    pieces = [p.strip() if p.strip() else str(src[i]) for i, p in enumerate(raw)]
    # Re-insert the single space that separated two fragments across the cut —
    # each piece is stripped independently, so a boundary space is lost and the
    # words would run together ("revenue"|"1,234.56" → "revenue1,234.56").
    for i in range(k - 1):
        bi = cuts[i + 1]
        if (pieces[i] and pieces[i + 1]
                and not pieces[i][-1].isspace() and not pieces[i + 1][0].isspace()
                and (t[bi - 1].isspace() or t[bi].isspace())):
            pieces[i + 1] = " " + pieces[i + 1]
    return pieces


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

#: Function / generic words that are NEVER terminology.  A title-cased heading makes
#: them look like proper nouns to the ``[A-Z][a-z]+`` regex, and a glossary entry that
#: pins "The" (or "While"/"Figure") to one rendering is injected into every batch as a
#: must-use term — corrupting the prose it appears in.
_INFER_STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here when while whilst
as at by for from in into of on onto to with without within about above after again
against all also am among any are because been before being below between both can
cannot could did do does doing done down during each either else few first further
had has have having he her his how however is it its just last least less like may
me might more most much must my no nor not now off often once only other our out over
own please rather same she should since so some such their them they though through
thus too under until up upon us very was we were what where whether which who whom why
will would you your figure figures table tables note notes data type types full blue
red green dashed solid small large late early high low new old next previous left
right top bottom
""".split())

#: Generic CJK phrases that are not terminology (a report boilerplate word pinned as a
#: glossary entry just forces a fixed rendering of something that has no single one).
_INFER_STOPWORDS_CJK = frozenset({
    "本报告", "本公司", "本公司及", "年度", "单位", "项目", "其中", "合计", "本期", "上年",
})

#: Candidate n-gram length range for CJK terms (2–12 characters).  Counting runs up to
#: ``_INFER_CJK_COUNT`` so a phrase's longer context is known; only the shorter ones are
#: reported (a 20-char sentence fragment is not a useful glossary entry).
_INFER_CJK_MIN = 2
_INFER_CJK_MAX = 12
_INFER_CJK_COUNT = 20

#: Chinese numerals / 附注 markers: a phrase made only of these is numbering, not a term.
_INFER_CJK_NUMERALS = frozenset("零一二三四五六七八九十百千")
#: Particles/auxiliaries that do not start or end a *term* (a phrase is not cut at a
#: content character like 对/以/为/所/无, which legitimately begin terms).
_INFER_CJK_EDGE = frozenset("的了与及和之等也都就而则且但很更最较又并或若因故即")


def _cjk_candidates(texts: Sequence[str], min_freq: int) -> list[str]:
    """Repeated *maximal* CJK n-grams (2–10 chars) that look like terminology.

    ``infer_terms`` used to keep the *maximal* CJK runs (``[\\u4e00-\\u9fff]+``) whose
    length is 2–4 — in real Chinese prose a sentence is ONE long run, so that condition
    almost never held and the document-level glossary was silently empty for every
    Chinese report (while the Latin side happily pinned ``Revenue``/``Total``).

    Instead, count every 2–16 char window over each run, keep the frequent ones, and
    keep only the **maximal** ones: a gram is dropped when the gram extended by the
    character that actually precedes/follows it is *also* frequent.  That is what
    turns sliding windows of one phrase (``动产生的现金`` / ``活动产生的现`` …) into the
    single phrase (``经营活动产生的现金流量净额``) and keeps ``其他综合收益`` instead of
    its 5-char tail ``他综合收益``.
    """
    counts: Counter[str] = Counter()
    before: dict[str, Counter[str]] = {}
    after: dict[str, Counter[str]] = {}
    for t in texts:
        for m in re.finditer(r"[\u4e00-\u9fff]{2,}", t):
            run = m.group(0)
            for n in range(_INFER_CJK_MIN, min(_INFER_CJK_COUNT, len(run)) + 1):
                for i in range(len(run) - n + 1):
                    gram = run[i:i + n]
                    counts[gram] += 1
                    if i > 0:
                        before.setdefault(gram, Counter())[run[i - 1]] += 1
                    if i + n < len(run):
                        after.setdefault(gram, Counter())[run[i + n]] += 1
    frequent = {g for g, c in counts.items() if c >= min_freq}
    out: list[str] = []
    for gram in frequent:
        if len(gram) > _INFER_CJK_MAX:
            continue
        if all(ch in _INFER_CJK_NUMERALS for ch in gram):
            continue                      # 附注编号（三十、十四…）不是术语
        if gram[0] in _INFER_CJK_EDGE or gram[-1] in _INFER_CJK_EDGE:
            continue                      # 以虚词起止的片段（"的现金净额"、"支付其他与"）
        if any(prev + gram in frequent for prev in before.get(gram, ())):
            continue
        if any(gram + nxt in frequent for nxt in after.get(gram, ())):
            continue
        out.append(gram)
    # Frequent first, then longer (a longer phrase is the more useful entry).
    out.sort(key=lambda g: (-counts[g], -len(g), g))
    return out


def infer_terms(ir: IRDoc, *, max_terms: int | None = None) -> list[str]:
    """Conservative document-level terminology candidates (C-⑥).

    Source-only (no translation): call :func:`translate_ir` with ``infer=True`` to
    translate them once and inject the result as ``IRDoc.terms``, giving every
    occurrence the same target across pages.  Candidates are repeated CJK phrases
    (2–6 chars, maximal and trimmed at function characters) and repeated TitleCase /
    ALL-CAPS Latin words, capped at ``max_terms``.  Stopwords (function words,
    "Figure", "Type", …) are filtered out: a pinned "The" is injected as a must-use
    term into every request, so a noisy candidate is not harmless — it actively
    distorts the translation.
    """
    texts = [str(b.text) for ipage in ir.pages for b in ipage.blocks
             if not is_structural_role(b.role) and not _is_verbatim(b.anchor)]
    cands = [t for t in _cjk_candidates(texts, _INFER_MIN_FREQ)
             if t not in _INFER_STOPWORDS_CJK]
    latin: Counter[str] = Counter()
    for t in texts:
        for w in re.findall(r"\b[A-Z][a-z]{2,}\b|\b[A-Z]{3,}\b", t):
            latin[w] += 1
    cands += [w for w, n in latin.items()
              if n >= _INFER_MIN_FREQ and w.casefold() not in _INFER_STOPWORDS]
    # ``cands`` is already unique (the CJK and Latin key sets are disjoint), so
    # de-duplication is redundant; just cap the count.
    return cands[:(max_terms or _INFER_MAX_TERMS)]


def infer_glossary(
    ir: IRDoc,
    translate_fn: Callable[..., Sequence[str]],
    *,
    lang: str,
    log: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Extract document-level terms and translate them once → a ``{source: target}`` glossary.

    The single source of the "infer terms → translate once → keep only the terms
    that actually changed" logic, shared by :func:`translate_ir` (``infer=True``)
    and by the interactive agent's preprocess (which injects the same glossary into
    ``state.user_decisions["terminology"]`` so per-page block translation stays
    terminology-consistent across pages).  A model that fails and echoes a source
    term would otherwise pin ``s -> s`` (suppressing that term's normal translation
    in the main pass), so only terms whose translation differs are kept.
    """
    terms = infer_terms(ir)
    if not terms:
        # Say so: an empty candidate list used to look exactly like "terminology
        # injection ran and found nothing to do", which hid the broken CJK
        # extraction for every Chinese report.
        if log:
            log("[ir] 未抽到重复术语候选，本次不注入文档级术语。")
        return {}
    got = translate_fn(terms, lang=lang, extra_glossary={})
    if len(got) != len(terms):
        if log:
            log(f"[ir] 术语批量返回 {len(got)} 条，与 {len(terms)} 条候选不符，跳过术语注入。")
        return {}
    return {s: str(t) for s, t in zip(terms, got)
            if str(t).strip() and str(t).strip() != s}


def translate_ir(
    ir: IRDoc,
    translate_fn: Callable[..., Sequence[str]],
    *,
    lang: str,
    extra_glossary: dict[str, str] | None = None,
    log: Callable[[str], None] | None = None,
    infer: bool = False,
    group_prose: bool | None = None,
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

    ``group_prose`` (default: on, env ``PDFTRANSLATE_IR_GROUP=0`` to disable) sends
    one *paragraph* per request instead of one extracted line — see
    :func:`prose_units` — and re-splits the answer back onto the blocks with
    :func:`split_translation`, so block indices (the exporter / overlay / audit
    primary keys) are untouched.
    """
    glossary = dict(ir.terms) if extra_glossary is None else dict(extra_glossary)
    blocks = [b for ipage in ir.pages for b in ipage.blocks]
    if infer and not glossary:
        glossary = infer_glossary(ir, translate_fn, lang=lang, log=log)
        if glossary:
            ir.terms = glossary
    translatable: list[IRBlock] = []
    out: dict[int, str] = {}
    for b in blocks:
        # Fail-safe for the structural gate: a block whose text was OCR'd out of a
        # raster figure (``Block.in_image``) is *text* and must be translated, even
        # if some backend still hands it a structural role.  ``build_structure``
        # already keeps such blocks out of a figure/formula region's members; this
        # guards the decision point itself (a chart's labels otherwise export
        # verbatim — see the fix note there).
        structural = is_structural_role(b.role) and not getattr(b.anchor, "in_image", False)
        if structural or _is_verbatim(b.anchor):
            out[b.src_id] = b.text      # formula/figure/numeric → keep source
        else:
            translatable.append(b)
    if not translatable:
        return out
    use_groups = _group_prose_enabled() if group_prose is None else bool(group_prose)
    units = prose_units(translatable) if use_groups else [[b] for b in translatable]
    got = translate_fn([join_texts([x.text for x in u]) for u in units],
                       lang=lang, extra_glossary=glossary)
    if len(got) != len(units):
        raise ValueError(
            f"translate_fn 返回 {len(got)} 条译文，与 {len(units)} 个翻译单元不一致")
    merged = [u for u in units if len(u) > 1]
    if log and merged:
        log(f"[ir] 段落级组批：{len(units)} 个翻译单元，其中 {len(merged)} 段由 "
            f"{sum(len(u) for u in merged)} 个文本块合并翻译后回填。")
    blanks = 0
    for u, t in zip(units, got):
        text = str(t)
        pieces = split_translation([x.text for x in u], text) if len(u) > 1 else [text]
        for x, p in zip(u, pieces):
            if not str(p).strip():
                blanks += 1
            out[x.src_id] = p
    if log and blanks:
        log(f"[ir] 段落回填后有 {blanks} 个块分到的内容为空（整段译文已并入同段其他块）。")
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
