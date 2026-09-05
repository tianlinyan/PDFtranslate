"""Deterministic evaluation metrics for the PDF translation pipeline.

This module is the *measurement baseline* behind direction **A-②**.  It reuses the
exact fit / number / residual primitives that drive the production exporter and the
audit engine (``pdfio._fit_block``, ``agent.flow._number_signature``, …) so the
metrics agree with what is actually drawn — it never re-implements a measurement.
Two layers:

* **Hard, offline metrics** (no model): layout fit (overflow / crowding /
  too_small / band violation / tiny font), number fidelity, and completeness
  (missing / residual).  These gate whether a change broke the layout.
* **Soft, optional metrics** (LLM-as-judge) live in ``eval_harness.py`` and are
  disabled by default.

Typical use is A/B: run the pipeline twice (baseline vs candidate), capture each
``out_doc``, then ``compare`` the aggregated reports.  All inputs are plain
``Block`` lists + flat per-block translated strings, so the module is unit-testable
without a ``QApplication`` or a real model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

from . import pdfio
from .translator import _needs_translation
from .prompts import _is_cjk_language
from .agent.flow import _has_cjk, _is_latin_prose, _number_signature


@dataclass
class EvalReport:
    """Layout measurement for one page."""

    page: int
    total: int                       # blocks actually measured (had a translation)
    counts: dict[str, int]           # too_small/overflow/crowding/band_violation/...
    buckets: dict[str, int]          # font-size buckets <4 / 4-5 / 5-6 / >=6
    issues: list[dict] = field(default_factory=list)   # per-block detail


#: Default weights for the aggregate layout score.  Adjusting this is how you
#: change the tolerance for a layout defect — never hard-code per-issue weights
#: into the judgment.
LAYOUT_WEIGHTS = {
    "overflow": 2.0,
    "band_violation": 2.0,
    "too_small": 1.5,
    "crowding": 1.5,
    "severe_tiny": 2.0,
    "too_many_lines": 1.0,
}


def _is_measurable(block) -> bool:
    """True when ``block`` carries translatable content (not a keepsake/code).

    Mirrors ``agent.flow._audit_protected`` (engine-skipped or numeric cell) plus
    the structure skips (org-chart node, vertical label): such a block is expected
    to stay verbatim, so it must never show up as "missing" or a layout defect.
    """
    if getattr(block, "is_chart", False):
        return False
    if pdfio._is_vertical_label(block):
        return False
    if not _needs_translation(str(block.text)):
        return False
    if pdfio._is_numeric_cell(str(block.text)):
        return False
    return True


def _issue(index: int, block, text: str, kind: str, detail: str, fs: float) -> dict:
    return {"index": index, "kind": kind, "detail": detail,
            "fs": round(fs, 2), "text": str(block.text)[:40], "translation": text[:40]}


def measure_layout(blocks: Sequence, translated_texts: Sequence[str], *,
                   page: int = 0, font=None) -> EvalReport:
    """Layout hard metrics for one page of source ``blocks`` / their ``translated_texts``.

    Each measured block is fit with ``pdfio._fit_block`` (the same measurement the
    exporter uses) and classified into the defect buckets.  Columns mirror
    ``agent.flow._check_layout`` + ``verify_real_run`` font-bucket/band statistics.
    """
    font = font or pdfio._CJK_FONT
    counts = {k: 0 for k in ("too_small", "overflow", "crowding", "band_violation",
                             "single_over", "severe_tiny", "too_many_lines")}
    buckets = {"<4": 0, "4-5": 0, "5-6": 0, ">=6": 0}
    issues: list[dict] = []
    total = 0
    for i, b in enumerate(blocks):
        if not _is_measurable(b):
            continue
        t = translated_texts[i] if i < len(translated_texts) else ""
        t = str(t or "").strip()
        if not t:
            continue  # handled by measure_complete
        try:
            lines, fs = pdfio._fit_block(b, font, t)
        except Exception:  # noqa: BLE001 — a block we cannot measure is skipped
            continue
        total += 1
        in_table = bool(getattr(b, "in_table", False))
        leading = pdfio._line_leading(font, in_table=in_table, n_lines=len(lines))
        height = pdfio._wrapped_height(font, lines, fs, leading)
        box_h = max(0.5, b.y1 - b.y0)
        start_fs = max(5.0, min(b.size, pdfio._MAX_FONT))
        floor = pdfio._MIN_TABLE_FLOOR if in_table else min(start_fs, pdfio._MIN_READABLE)

        if fs < 4:
            buckets["<4"] += 1
        elif fs < 5:
            buckets["4-5"] += 1
        elif fs < 6:
            buckets["5-6"] += 1
        else:
            buckets[">=6"] += 1
        if fs < 5:
            counts["severe_tiny"] += 1

        if fs + 1e-9 < floor:
            counts["too_small"] += 1
            issues.append(_issue(i, b, t, "too_small",
                                 f"译文字号 {fs:.2f}pt 低于可读下限 {floor:.2f}pt", fs))
        if height > box_h + 2.0:
            counts["overflow"] += 1
            issues.append(_issue(i, b, t, "overflow",
                                 f"译文高度 {height:.1f}pt 超过自身框 {box_h:.1f}pt", fs))
        below = [nb for nb in blocks
                 if nb is not b and nb.y0 >= b.y1 - 0.5
                 and nb.x0 < b.x1 and nb.x1 > b.x0]
        if below:
            gap = min(nb.y0 for nb in below) - b.y1
            if height > box_h + gap + 2.0:
                counts["crowding"] += 1
                issues.append(_issue(i, b, t, "crowding",
                                     f"译文高 {height:.1f}pt 会压入下一块（剩余 {gap:.1f}pt）", fs))
        if in_table and b.fit_height > 0:
            if len(lines) > 1 and height > b.fit_height + 0.05:
                counts["band_violation"] += 1
                issues.append(_issue(i, b, t, "band_violation",
                                     f"{len(lines)} 行越过行带 {b.fit_height:.1f}pt", fs))
            elif len(lines) == 1 and height > b.fit_height + 0.05:
                counts["single_over"] += 1
        if len(lines) > 1 and fs + 1e-9 <= floor:
            counts["too_many_lines"] += 1
    return EvalReport(page=page, total=total, counts=counts, buckets=buckets, issues=issues)


def measure_numbers(blocks: Sequence, translated_texts: Sequence[str], *,
                    page: int = 0) -> dict:
    """Number fidelity per block: does each amount survive value-for-value?

    Reuses ``agent.flow._number_signature`` so full-width glyphs, unit multipliers
    and month/date expansion compare ``3,702,726,474.45`` vs ``３０７２７２６４７４．４５``
    by value and still flag a dropped/altered digit.
    """
    out = []
    for i, b in enumerate(blocks):
        t = translated_texts[i] if i < len(translated_texts) else ""
        t = str(t or "").strip()
        if not t:
            continue  # untranslated → completeness, not number fidelity
        src = _number_signature(str(b.text))
        trans = _number_signature(t)
        missing = [str(x) for x in (src - trans).elements()]
        extra = [str(x) for x in (trans - src).elements()]
        if missing or extra:
            out.append({"index": i, "source": str(b.text), "translation": t,
                        "missing": missing, "extra": extra})
    return {"page": page, "numbers": out, "count": len(out)}


def measure_complete(blocks: Sequence, translated_texts: Sequence[str], *,
                     lang: str = "English", page: int = 0) -> dict:
    """Completeness: blocks that need translation but got none (missing) plus
    residual source-language content left in a translated block.

    Mirrors ``agent.flow._check_missing`` / ``_check_residual``: a translated
    block still holding CJK (Western target) or untranslated Latin prose (CJK
    target) is a residual.
    """
    cjk_target = _is_cjk_language(lang)
    missing, residual = [], []
    for i, b in enumerate(blocks):
        if not _is_measurable(b):
            continue
        t = translated_texts[i] if i < len(translated_texts) else ""
        t = str(t or "").strip()
        if not t:
            missing.append({"index": i, "text": str(b.text)})
        elif cjk_target:
            if _is_latin_prose(t):
                residual.append({"index": i, "text": str(b.text), "reason": "residual_latin"})
        elif _has_cjk(t):
            residual.append({"index": i, "text": str(b.text), "reason": "residual_cjk"})
    return {"page": page, "missing": missing, "residual": residual,
            "missing_count": len(missing), "residual_count": len(residual)}


def aggregate(reports: Sequence[EvalReport], *, weights: dict[str, float] | None = None) -> dict:
    """Aggregate a list of page layout reports into a summary + a 0–100 score.

    ``score = 100 × (1 − min(1, weighted_issues / total_blocks))``.
    ``total`` is only the blocks that were actually measured (had a translation),
    so a page with no translatable content does not drag the score down.
    """
    w = dict(LAYOUT_WEIGHTS)
    if weights:
        w.update(weights)
    total = sum(r.total for r in reports)
    counts: dict[str, int] = {}
    buckets: dict[str, int] = {}
    for r in reports:
        for k, v in r.counts.items():
            counts[k] = counts.get(k, 0) + v
        for k, v in r.buckets.items():
            buckets[k] = buckets.get(k, 0) + v
    weighted = sum(w.get(k, 0.0) * v for k, v in counts.items())
    issue_ratio = min(1.0, weighted / total) if total else 0.0
    return {
        "pages": len(reports),
        "total": total,
        "counts": counts,
        "buckets": buckets,
        "weighted_issues": round(weighted, 2),
        "issue_ratio": round(issue_ratio, 4),
        "score": round(100 * (1 - issue_ratio), 2),
    }


def eval_pages(pages_blocks: Sequence[Sequence], pages_translated: Sequence[Sequence[str]],
               *, lang: str = "English", weights: dict[str, float] | None = None) -> dict:
    """Run all hard metrics over every page and compose a final summary.

    ``pages_blocks``: per-page ``Block`` lists (a ``DocumentText.pages``).
    ``pages_translated``: per-page flat translated text lists, aligned to the
    blocks.  Returns ``{score, layout, numbers, complete, per_page}``.
    """
    reports, numbers, complete = [], [], []
    for p, (blocks, trans) in enumerate(zip(pages_blocks, pages_translated)):
        reports.append(measure_layout(blocks, trans, page=p))
        numbers.append(measure_numbers(blocks, trans, page=p))
        complete.append(measure_complete(blocks, trans, lang=lang, page=p))
    layout = aggregate(reports, weights=weights)
    n_missing = sum(c["missing_count"] for c in complete)
    n_residual = sum(c["residual_count"] for c in complete)
    n_numeric = sum(n["count"] for n in numbers)
    # Fold completeness/number defects into the final score with clear weights.
    defect_blocks = layout["total"] + n_missing + n_residual
    defect_w = (layout["weighted_issues"] + n_missing * 2.0 + n_residual * 2.0
                + n_numeric * 2.0)
    ratio = min(1.0, defect_w / defect_blocks) if defect_blocks else 0.0
    return {
        "lang": lang,
        "score": round(100 * (1 - ratio), 2),
        "layout": layout,
        "numbers": {"total": n_numeric},
        "complete": {"missing": n_missing, "residual": n_residual},
        "per_page": [{"layout": asdict(r), "numbers": n, "complete": c}
                     for r, n, c in zip(reports, numbers, complete)],
    }


def compare(base: dict, cand: dict) -> dict:
    """A/B: delta between a baseline and candidate ``eval_pages`` result.

    Returns aggregate deltas plus the pages whose score changed most.  A negative
    delta on a hard metric means the candidate made it *worse*.
    """
    def _delta(b, c):
        return round(c - b, 2) if isinstance(b, (int, float)) and isinstance(c, (int, float)) else None

    out: dict = {"score_delta": _delta(base["score"], cand["score"]),
                 "layout": {}, "numbers": {}, "complete": {}}
    for k in ("total", "weighted_issues", "issue_ratio"):
        out["layout"][k] = _delta(base["layout"].get(k), cand["layout"].get(k))
    for k in set(list(base["layout"]["counts"]) + list(cand["layout"]["counts"])):
        bc = base["layout"]["counts"].get(k, 0)
        cc = cand["layout"]["counts"].get(k, 0)
        if bc or cc:
            out["layout"]["counts_delta_" + k] = cc - bc
    out["numbers"]["total_delta"] = _delta(base["numbers"]["total"], cand["numbers"]["total"])
    out["complete"]["missing_delta"] = _delta(base["complete"]["missing"], cand["complete"]["missing"])
    out["complete"]["residual_delta"] = _delta(base["complete"]["residual"], cand["complete"]["residual"])
    return out
