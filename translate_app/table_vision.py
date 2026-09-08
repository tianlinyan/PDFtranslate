"""table_vision.py — 用 vision model 识别扫描页表格的**排版样式**。

面向「重建 OCR 表格为非 OCR（矢量）表格」：OCR 块几何（:func:`pdfio._reconstruct_ocr_grid`）
重建不了超密集扫描报表的真实表格线（295 个 col_edges 里大量是同一列不同块的噪点
边缘）。让 vision model 看扫描页 PNG，输出表格的**排版样式**——行/列边界、合并
单元格、表头、对齐——再据此重建为矢量表格并正常填充译文。

非确定性的（模型在线，识别可能有误）：失败时调用方回退 OCR 几何或禁用重建。
默认关闭（``rebuild_table`` + vision model 在线才启用）。
"""
from __future__ import annotations

import base64
import json
import re
from typing import Callable, Sequence

#: 让 vision model 输出表格排版样式（0..1 归一化边界 + 合并/表头/对齐）。
_PROMPT = (
    "你是扫描报表表格排版样式识别专家。看这张扫描页，识别表格的排版样式，"
    "只输出 JSON，格式：\n"
    '{"rows": [每行上边界 y(0..1 归一化)，从上到下递增，...], '
    '"cols": [每列左边界 x(0..1 归一化)，从左到右递增，...], '
    '"merged": [{"r": 行索引, "c": 列索引, "row_span": 1, "col_span": 2}, ...], '
    '"header_rows": [表头行索引...], '
    '"header_cols": [表头列索引...], '
    '"align": [{"col": 列索引, "dir": "right"}, ...], '
    '"non_text": [{"x0": 0, "y0": 0, "x1": 1, "y1": 1, "kind": "signature|stamp"}, ...]}\n'
    "识别规则：\n"
    "1. 只取最大的一张表；无表格则 rows/cols 为空数组；多表只输出最大；跨页只输出本页部分。\n"
    "2. 行/列边界只算表格横线/竖线的位置，单调递增；边界用 0..1 归一化（左上 0,0 右下 1,1）。"
    "每列都要画出来（含最右侧的母公司列），**最右列右边界=该列数字右对齐后的实际右缘**；"
    "不要为了撑满页面把表格边界拉到接近 1.0 的页面右缘——那会让最后一列过宽、数字贴边/超宽。\n"
    "3. merged 标跨行/跨列的合并单元格（如「合并」/「母公司」表头跨两列、"
    "首行表头跨两行）；正文普通单元格不标。\n"
    "4. 表头：标题/表头所在的行、列下标放进 header_rows/header_cols。\n"
    "5. 对齐：数字含量高的列 dir=right，文字列 left（默认）。\n"
    "6. non_text 标手写体签名、盖章等非文本内容的区域（归一化 bbox + kind=[signature|stamp]）；"
    "这些不是表格内容，单独标出、不要算进行列边界。\n"
    "7. 只输出 JSON，不要解释，不要 Markdown 代码块。"
)


def _parse_json(text: str) -> dict | None:
    """Extract the first JSON object from the model reply (tolerates ``` fences)."""
    t = str(text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{"):
        s = t.find("{")
        e = t.rfind("}")
        if s >= 0 and e > s:
            t = t[s:e + 1]
    try:
        data = json.loads(t)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _parse_table_style(data: dict, page_width: float, page_height: float) -> dict | None:
    """Normalise a model reply into a TableStyle dict, or ``None`` if unusable.

    Returns ``{"rows_pts", "cols_pts", "merged", "header_rows", "header_cols", "align"}``
    where rows/cols are PDF points.
    """
    rows = data.get("rows")
    cols = data.get("cols")
    if not isinstance(rows, list) or not isinstance(cols, list):
        return None
    try:
        rows_pts = [min(1.0, max(0.0, float(v))) * page_height for v in rows]
        cols_pts = [min(1.0, max(0.0, float(v))) * page_width for v in cols]
    except (TypeError, ValueError):
        return None
    if len(rows_pts) < 2 or len(cols_pts) < 2:
        return None
    return {
        "rows_pts": sorted(rows_pts),
        "cols_pts": sorted(cols_pts),
        "merged": list(data.get("merged") or []),
        "header_rows": list(data.get("header_rows") or []),
        "header_cols": list(data.get("header_cols") or []),
        "align": list(data.get("align") or []),
        "non_text": _parse_non_text(data.get("non_text"), page_width, page_height),
    }


def _parse_non_text(raw, page_width: float, page_height: float) -> list[tuple]:
    """Parse ``non_text`` regions (handwritten signatures / stamps) into PDF points.

    Each entry is ``(x0, y0, x1, y1, kind)`` (PDF points).  These are NOT table
    content — the rebuild pass must leave them untranslated and uncovered.
    """
    out: list[tuple] = []
    for item in (raw or []):
        if not isinstance(item, dict):
            continue
        try:
            x0 = min(1.0, max(0.0, float(item.get("x0", 0)))) * page_width
            y0 = min(1.0, max(0.0, float(item.get("y0", 0)))) * page_height
            x1 = min(1.0, max(0.0, float(item.get("x1", 0)))) * page_width
            y1 = min(1.0, max(0.0, float(item.get("y1", 0)))) * page_height
        except (TypeError, ValueError):
            continue
        kind = str(item.get("kind") or "signature")
        out.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), kind))
    return out


def make_llm_table_structure(
    model,
    *,
    page_width: float,
    page_height: float,
    client=None,
    log: Callable[[str], None] | None = None,
    n_samples: int = 3,
) -> Callable[[bytes], list[dict]] | None:
    """Return a ``png -> [TableStyle, ...]`` detector, or ``None`` if non-vision.

    Layer-② robustness (revised): the detector runs the vision model ``n_samples``
    times and returns *every* parsed sample.  It deliberately does NOT mode-average
    here — the samples genuinely disagree (notably on row count, 60 vs 52 on the
    same page) and averaging hides that; the caller scores them with
    :func:`pdfio.score_rebuild` and picks the best, falling back to plain OCR when
    even the best fails :func:`pdfio.valid_rebuild`.
    """
    if not getattr(model, "vision", False):
        return None
    from openai import OpenAI

    client = client or OpenAI(**model.client_kwargs())
    body = model.request_params() if hasattr(model, "request_params") else {}

    def _single(png: bytes) -> dict | None:
        b64 = base64.b64encode(png).decode()
        try:
            resp = client.chat.completions.create(
                model=model.model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]}],
                temperature=0.0,
                max_tokens=4096,
                extra_body=body or None,
            )
            content = resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 — a vision outage falls back
            if log:
                log(f"  [table_vision] 识别失败：{type(exc).__name__}: {exc}")
            return None
        data = _parse_json(content)
        if not data:
            if log:
                log("  [table_vision] 无法解析模型返回（非 JSON）。")
            return None
        style = _parse_table_style(data, page_width, page_height)
        if style is None:
            if log:
                log("  [table_vision] 返回缺少可用行列边界。")
        return style

    def _detect(png: bytes) -> list[dict]:
        out: list[dict] = []
        for _ in range(max(1, int(n_samples))):
            r = _single(png)
            if r:
                out.append(r)
        return out

    return _detect


def tables_from_grid(rows_pts: Sequence[float], cols_pts: Sequence[float]) -> list[dict]:
    """Build a ``tables`` structure (rows of cell rects + col_edges) from grid lines.

    Mirrors the dict shape :func:`pdfio._extract_tables` / :func:`pdfio._reconstruct_ocr_tables`
    return, so :func:`pdfio._map_blocks_to_table_cells` and :func:`pdfio._compute_table_layout`
    consume it unchanged.  ``rows_pts`` / ``cols_pts`` are PDF-point boundaries.
    """
    rows: list[list] = []
    for r in range(len(rows_pts) - 1):
        row = [
            _rect(cols_pts[c], rows_pts[r], cols_pts[c + 1], rows_pts[r + 1])
            for c in range(len(cols_pts) - 1)
        ]
        rows.append(row)
    bbox = _rect(cols_pts[0], rows_pts[0], cols_pts[-1], rows_pts[-1])
    return [{"bbox": bbox, "rows": rows, "col_edges": list(cols_pts)}]


def _rect(x0: float, y0: float, x1: float, y1: float):
    import fitz
    return fitz.Rect(x0, y0, x1, y1)
