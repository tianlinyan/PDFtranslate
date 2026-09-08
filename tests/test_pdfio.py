"""Tests for PDF text extraction and the export writers."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pymupdf as fitz

from translate_app import pdfio

from tests._helpers import (
    build_sample_pdf,
    build_two_column_pdf,
    build_two_column_pdf_with_heading,
    build_list_table_pdf,
)

_OUT = Path(__file__).resolve().parent / "_out"


class _OcrCacheIsolated(unittest.TestCase):
    """Redirect the OCR / translation caches so integration tests never touch the
    developer's ``~/.pdftranslate`` (and never hit a hot cache that would turn a
    test green without exercising the review/OCR path)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.dict(
            os.environ,
            {
                "PDFTRANSLATE_OCR_CACHE_DIR": str(tmp.name),
                "PDFTRANSLATE_CACHE_DIR": str(tmp.name),
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)


def _text_lines(page):
    """Return ``(bbox, text)`` for every rendered text line on ``page``."""
    out = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if text:
                out.append((fitz.Rect(line["bbox"]), text))
    return out


def setUpModule():  # noqa: N802
    _OUT.mkdir(exist_ok=True)


def tearDownModule():  # noqa: N802
    shutil.rmtree(_OUT, ignore_errors=True)


class PdfioTest(unittest.TestCase):
    def test_extract_and_exports(self):
        src = _OUT / "sample.pdf"
        build_sample_pdf(src, pages=2)

        doc = pdfio.extract_document_text(src)
        self.assertEqual(doc.page_count, 2)
        self.assertTrue(doc.blocks)
        self.assertEqual(len(doc.blocks), len(doc.block_pages))

        per_page = pdfio.group_by_page(doc.block_pages, doc.blocks, doc.page_count)
        self.assertEqual(len(per_page), 2)

        # Plain text + markdown
        txt = _OUT / "out.txt"
        pdfio.save_plain_text(per_page, txt)
        self.assertTrue(txt.exists() and txt.stat().st_size > 0)

        md = _OUT / "out.md"
        pdfio.save_markdown(per_page, doc.blocks, doc.block_pages, md, "Chinese", "T")
        self.assertTrue(md.exists() and md.stat().st_size > 0)
        self.assertIn("# T", md.read_text("utf-8"))

    def test_bilingual_pdf(self):
        src = _OUT / "sample_b.pdf"
        build_sample_pdf(src, pages=2)
        doc = pdfio.extract_document_text(src)
        per_page = pdfio.group_by_page(doc.block_pages, doc.blocks, doc.page_count)
        out = _OUT / "bilingual.pdf"
        pdfio.save_interleaved_pdf(src, per_page, out, "Chinese")
        d = fitz.open(str(out))
        self.assertGreaterEqual(d.page_count, 4)  # 2 original + 2 translation pages
        d.close()

    def test_translated_pdf_removes_original_text(self):
        src = _OUT / "sample_t.pdf"
        build_sample_pdf(src, pages=2)
        doc = pdfio.extract_document_text(src)
        per_page = pdfio.group_by_page(doc.block_pages, doc.blocks, doc.page_count)
        fake = [
            ["TRANSLATED-" + str(idx) for idx, _t in enumerate(pg)]
            for pg in per_page
        ]
        out = _OUT / "translated.pdf"
        pdfio.save_translated_pdf(src, doc.pages, fake, out, "Chinese")
        d = fitz.open(str(out))
        # Original English body must be gone; placeholder must be present.
        all_text = "\n".join(d[p].get_text() for p in range(d.page_count))
        self.assertNotIn("sample body paragraph", all_text)
        self.assertIn("TRANSLATED-", all_text)
        d.close()

    def test_translated_pdf_covers_ocr_block_with_white(self):
        # An OCR block sits on a raster image, not a text layer, so the
        # in-place exporter must cover the original (scanned) pixels with a
        # white rectangle instead of redacting text — otherwise the
        # translation overprints the original.
        src = _OUT / "scan_src.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        # A dark pixmap standing in for a scanned region containing text.
        pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 300, 70), 0)
        pix.set_rect(fitz.IRect(0, 0, 300, 70), (30,))  # dark gray "scan"
        page.insert_image(fitz.Rect(50, 50, 350, 120), pixmap=pix)
        doc.save(str(src))
        doc.close()

        ocr_block = pdfio.Block(
            text="scanned text", page=0, x0=50, y0=50, x1=350, y1=120,
            size=12.0, align="left", bold=False, single_line=False, ocr=True,
        )
        out = _OUT / "translated_ocr.pdf"
        pdfio.save_translated_pdf(src, [[ocr_block]], [["translated text"]], out, "Chinese")

        d = fitz.open(str(out))
        page = d[0]
        # The translation must be present.
        self.assertIn("translated text", page.get_text())
        # A white-filled rectangle must cover the OCR block's bbox (the block
        # rect expanded by 0.5 on each side).
        found_white = False
        for dr in page.get_drawings():
            fill = dr.get("fill")
            if fill is None:
                continue
            r = dr["rect"]
            if (
                all(abs(c - 1.0) < 0.01 for c in fill)
                and abs(r.x0 - 49.5) < 1.0
                and abs(r.y0 - 49.5) < 1.0
                and abs(r.x1 - 350.5) < 1.0
                and abs(r.y1 - 120.5) < 1.0
            ):
                found_white = True
        d.close()
        self.assertTrue(found_white, "expected a white cover rectangle over the OCR block")

    def test_translated_pdf_keeps_readable_font(self):
        # A translation far longer than its box used to be font-trimmed down to
        # an illegible ~4pt sliver so it would stay inside the box.  The
        # readability floor now wins: the rendered size never drops below
        # ``_MIN_READABLE`` (the text may extend a little past the source box,
        # which is redacted, rather than become unreadable), and it must not
        # overflow the page width.
        src = _OUT / "sample_over.pdf"
        build_sample_pdf(src, pages=1)
        doc = pdfio.extract_document_text(src)
        # Give the first block a long translation that needs many wrapped lines.
        long_cjk = (
            "本飞机专为单飞行员操作而开发，并经过相应调整以更好地模拟真实环境。"
            "其制作综合了多个真实世界的数据点和来自不同时期、不同来源的手册，"
            "并通过对各类组件的修改来让它们更容易在微软飞行模拟器中管理。"
        )
        per_page = [[""] * len(doc.pages[0])]
        per_page[0][0] = long_cjk
        out = _OUT / "translated_over.pdf"
        pdfio.save_translated_pdf(src, doc.pages, per_page, out, "Chinese")
        d = fitz.open(str(out))
        page = d[0]
        info = page.get_text("dict")
        rects, sizes = [], []
        for b in info.get("blocks", []):
            if b.get("type") != 0:
                continue
            for l in b.get("lines", []):
                t = "".join(s["text"] for s in l["spans"]).strip()
                if t:
                    rects.append(fitz.Rect(l["bbox"]))
                    sizes.extend(s["size"] for s in l["spans"])
        self.assertTrue(rects, "expected rendered translation text")
        # Readability floor: no rendered span is smaller than _MIN_READABLE.
        self.assertTrue(sizes)
        self.assertGreaterEqual(min(sizes), pdfio._MIN_READABLE)
        # And no line spills past the page width.
        for r in rects:
            self.assertLessEqual(r.x1, page.rect.x1 + 1.0)
        d.close()


    def test_two_column_reading_order_and_metadata(self):
        src = _OUT / "two_col.pdf"
        build_two_column_pdf(src)
        doc = pdfio.extract_document_text(src)
        self.assertEqual(doc.page_count, 1)
        # Column-major reading order: the whole left column first, then the
        # right column (including the right-aligned footer line).
        texts = doc.blocks
        left_idx = [i for i, t in enumerate(texts) if t.startswith("Left column")]
        right_idx = [i for i, t in enumerate(texts) if t.startswith("Right column")]
        self.assertTrue(left_idx and right_idx)
        self.assertLess(max(left_idx), min(right_idx))
        self.assertEqual(texts[-1], "Page 1 of 9")

        # Layout hints are captured: every block carries a size, an alignment
        # and a single-line flag, and the footer is detected as right-aligned.
        for block in doc.pages[0]:
            self.assertGreater(block.size, 0)
            self.assertIn(block.align, ("left", "center", "right"))
            self.assertTrue(block.single_line)
        footer = doc.pages[0][-1]
        self.assertEqual(footer.text, "Page 1 of 9")
        self.assertEqual(footer.align, "right")

    def test_full_width_heading_does_not_merge_columns(self):
        # Regression: a heading spanning both columns used to become a
        # "column" whose right edge was the page width, so every line of both
        # real columns "overlapped" it and the two columns were merged into
        # one — left and right lines interleaved by y into single blocks.
        # The heading must read first, then the whole left column, then the
        # whole right column.
        src = _OUT / "two_col_heading.pdf"
        build_two_column_pdf_with_heading(src)
        doc = pdfio.extract_document_text(src)
        texts = doc.blocks
        self.assertEqual(
            texts[0], "FULL WIDTH HEADING ACROSS BOTH COLUMNS OF THE PAGE"
        )
        left_idx = [i for i, t in enumerate(texts) if t.startswith("Left column")]
        right_idx = [i for i, t in enumerate(texts) if t.startswith("Right column")]
        self.assertTrue(left_idx and right_idx)
        # The whole left column still precedes the whole right column.
        self.assertLess(max(left_idx), min(right_idx))
        self.assertEqual(
            [texts[i] for i in left_idx],
            [
                "Left column first line.",
                "Left column second line.",
                "Left column third line.",
            ],
        )
        self.assertEqual(
            [texts[i] for i in right_idx],
            ["Right column first line.", "Right column second line."],
        )

    def test_list_and_table_entries_are_not_collapsed(self):
        # Regression: PyMuPDF merges a close-spaced list / table into one block,
        # which used to collapse the whole thing into a single run-on paragraph.
        # The line-aware extractor must keep every numbered / ``Label:`` entry as
        # its own single-line block.
        src = _OUT / "list_table.pdf"
        build_list_table_pdf(src)
        doc = pdfio.extract_document_text(src)
        blocks = doc.pages[0]
        item_texts = [b.text for b in blocks]
        # Each numbered item is a distinct block, not merged with its neighbours.
        for txt in ("1. First item", "2. Second item", "3. Third item"):
            self.assertIn(txt, item_texts)
        # Label/table rows stay on their own lines (single-line blocks).
        rows = [b for b in blocks if b.text.startswith(
            ("Powerplant:", "Brand:", "Model:"))]
        self.assertEqual(len(rows), 3)
        for b in rows:
            self.assertTrue(b.single_line, f"table row collapsed: {b.text!r}")
            self.assertIn(b.text, ("Powerplant:", "Brand: Pratt & Whitney",
                                   "Model: PT6A-140"))

    def test_bilingual_pdf_mirrors_block_positions(self):
        src = _OUT / "sample_m.pdf"
        build_sample_pdf(src, pages=1)
        doc = pdfio.extract_document_text(src)
        per_page = pdfio.group_by_page(doc.block_pages, doc.blocks, doc.page_count)
        fake = [[f"TR-{i}" for i, _t in enumerate(pg)] for pg in per_page]
        out = _OUT / "bilingual_mirror.pdf"
        pdfio.save_interleaved_pdf(src, fake, out, "Chinese", doc.pages)
        d = fitz.open(str(out))
        self.assertEqual(d.page_count, 2)  # original page + translation page
        trans_lines = _text_lines(d[1])
        self.assertEqual(len(trans_lines), len(doc.pages[0]))
        # Each translated block must sit at its source block's y position.
        for block, (rect, _t) in zip(doc.pages[0], trans_lines):
            self.assertGreaterEqual(rect.y0, block.y0 - 2.0)
            self.assertLessEqual(rect.y0, block.y1 + 2.0)
        d.close()

    def test_single_line_block_vertically_centered(self):
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=50, y0=100, x1=350, y1=200,
            size=12.0, align="left", bold=False, single_line=True,
        )
        pdfio._draw_translated_block(page, font, block, "居中文本")
        lines = _text_lines(page)
        self.assertEqual(len(lines), 1)
        (rect, text), = lines
        self.assertEqual(text, "居中文本")
        # The rendered line's vertical centre sits in the box's centre.
        self.assertAlmostEqual(
            (rect.y0 + rect.y1) / 2, (100 + 200) / 2, delta=2.0,
        )

    def test_multiline_table_cell_is_top_anchored(self):
        # A table cell whose translation must wrap to >1 line: it should hug the
        # cell's top rule instead of being vertically centred, so the wrapped
        # block stays inside the cell rather than drifting toward the bottom line.
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=50, y0=100, x1=110, y1=130,
            size=12.0, align="left", bold=False, single_line=True,
            in_table=True,
        )
        pdfio._draw_translated_block(
            page, font, block, "Net Assets Per Share Attributable (Yuan)"
        )
        lines = _text_lines(page)
        self.assertGreater(len(lines), 1)
        # The first (top) line's glyph top sits at the cell's top rule, not centred.
        first = min(lines, key=lambda it: it[0].y0)
        self.assertLess(first[0].y0 - block.y0, 6.0)

    def test_right_aligned_block_hugs_right_edge(self):
        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=50, y0=100, x1=350, y1=130,
            size=12.0, align="right", bold=False, single_line=True,
        )
        pdfio._draw_translated_block(page, font, block, "右对齐")
        (rect, _t), = _text_lines(page)
        self.assertAlmostEqual(rect.x1, 350.0, delta=1.5)

    def test_bold_block_renders_text(self):
        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=50, y0=100, x1=350, y1=130,
            size=12.0, align="left", bold=True, single_line=True,
        )
        pdfio._draw_translated_block(page, font, block, "粗体标题")
        (rect, text), = _text_lines(page)
        self.assertEqual(text, "粗体标题")
        # Bold must NOT pull in a second font: every rendered span uses the
        # standard cjk font (a mixed Heiti/Droid look was a visible defect).
        fonts = {
            s["font"]
            for b in page.get_text("dict")["blocks"] if b.get("type") == 0
            for l in b["lines"] for s in l["spans"]
        }
        self.assertEqual(fonts, {"Droid Sans Fallback Regular"})

    def test_color_restores_original_color(self):
        # Regression: the exporter flattened every heading to black.  A block's
        # captured source colour must be reproduced (same single CJK font).
        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=50, y0=100, x1=300, y1=130,
            size=12.0, align="left", bold=False, single_line=True, color=0xCC0000,
        )
        pdfio._draw_translated_block(page, font, block, "红色标题")
        span = page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]
        self.assertEqual(span["color"], 0xCC0000)
        # Still the one consistent CJK font (no Heiti creep).
        fonts = {
            s["font"] for b in page.get_text("dict")["blocks"] if b.get("type") == 0
            for l in b["lines"] for s in l["spans"]
        }
        self.assertEqual(fonts, {"Droid Sans Fallback Regular"})
        doc.close()

    def test_table_row_expansion_shifts_rows_down(self):
        # Regression: a translated table row that no longer fits its cell used to
        # overlap the row beneath it.  Rows must be pushed down (and their height
        # enlarged) so they stay separate.
        font = fitz.Font("cjk")
        rows = []
        cells = []
        for r in range(3):
            row = [fitz.Rect(x, 200 + r * 20, x + 90, 220 + r * 20) for x in (60, 160)]
            rows.append(row)
            cells.extend(row)
        tables = [{"bbox": fitz.Rect(60, 200, 250, 260), "rows": rows,
                   "col_edges": [60, 150, 250]}]
        blocks, trans = [], []
        for r in range(3):
            for c in range(2):
                cell = rows[r][c]
                blocks.append(pdfio.Block(
                    text="t", page=0, x0=cell.x0, y0=cell.y0, x1=cell.x1, y1=cell.y1,
                    size=9.0, single_line=True,
                ))
                # The middle row gets a translation that needs several lines.
                trans.append(
                    "A considerably longer translated string that wraps across "
                    "many lines and therefore needs extra row height."
                    if r == 1 else "Short"
                )
        mapping = pdfio._map_blocks_to_table_cells(blocks, tables)
        self.assertEqual(len(mapping), len(blocks))
        shifts, new_bottoms, grid, bboxes = pdfio._compute_table_layout(
            tables, mapping, blocks, trans, font
        )
        row0 = next(bi for bi, (_ti, r, _c) in mapping.items() if r == 0)
        row1 = next(bi for bi, (_ti, r, _c) in mapping.items() if r == 1)
        row2 = next(bi for bi, (_ti, r, _c) in mapping.items() if r == 2)
        # Top row is undisturbed; the lower rows move down because row 1 grows.
        self.assertEqual(shifts[row0], 0.0)
        self.assertGreaterEqual(shifts[row1], 0.0)
        self.assertGreater(shifts[row2], 0.0)
        # The grown row 1's new bottom must not cross row 2's shifted top.
        self.assertLessEqual(new_bottoms[row1], 240.0 + shifts[row2] + 0.5)
        # A redrawn grid (horizontal + vertical rules) is produced.
        self.assertTrue(any(g[0] == "h" for g in grid))
        self.assertTrue(any(g[0] == "v" for g in grid))
        # The whole original table is flagged for removal (stale grid lines).
        self.assertEqual(len(bboxes), 1)

    def test_rebalance_table_columns_widens_long_column_and_keeps_total_width(self):
        # C-⑥ reflow（保守层）：长译文列借用相邻短列的空白，表格总宽不变。
        font = fitz.Font("cjk")
        rows = [[fitz.Rect(60, 200, 150, 220), fitz.Rect(150, 200, 250, 220)]]
        tables = [{"bbox": fitz.Rect(60, 200, 250, 220), "rows": rows,
                   "col_edges": [60, 150, 250]}]
        blocks, trans = [], []
        blocks.append(pdfio.Block(text="项目", page=0, x0=60, y0=200, x1=150, y1=220, size=9.0))
        trans.append("A very long translated header that definitely exceeds the narrow column width")
        blocks.append(pdfio.Block(text="备注", page=0, x0=150, y0=200, x1=250, y1=220, size=9.0))
        trans.append("ok")
        mapping = pdfio._map_blocks_to_table_cells(blocks, tables)
        col_boxes, new_edges = pdfio._rebalance_table_columns(tables, mapping, blocks, trans, font)
        # Long column（列 0）widens: its bbox right edge moves past the old 150.
        self.assertGreater(col_boxes[0][1], 150.0)
        # Short column（列 1）narrows: its bbox left edge moves right past 150.
        self.assertGreater(col_boxes[1][0], 150.0)
        # Total width preserved.
        self.assertAlmostEqual(new_edges[0][-1] - new_edges[0][0], 190.0, delta=0.5)

    def test_rebalance_table_columns_never_shrinks_numeric_column(self):
        font = fitz.Font("cjk")
        rows = [[fitz.Rect(60, 200, 150, 220), fitz.Rect(150, 200, 250, 220)]]
        tables = [{"bbox": fitz.Rect(60, 200, 250, 220), "rows": rows,
                   "col_edges": [60, 150, 250]}]
        blocks, trans = [], []
        blocks.append(pdfio.Block(text="项目", page=0, x0=60, y0=200, x1=150, y1=220, size=9.0))
        trans.append("A very long translated header that definitely exceeds the narrow column width")
        blocks.append(pdfio.Block(text="1,234.56", page=0, x0=150, y0=200, x1=250, y1=220, size=9.0))
        trans.append("1,234.56")
        mapping = pdfio._map_blocks_to_table_cells(blocks, tables)
        col_boxes, _new_edges = pdfio._rebalance_table_columns(tables, mapping, blocks, trans, font)
        # 数字列不缩：bbox 保持原列边界（150+2, 250-2）。
        self.assertAlmostEqual(col_boxes[1][0], 152.0, delta=0.5)
        self.assertAlmostEqual(col_boxes[1][1], 248.0, delta=0.5)

    def test_rebuild_ocr_table_reanchors_blocks_to_cells(self):
        # C-⑥ 重建为矢量表格：把 OCR 块 bbox 重排到 AI 识别的单元格边界，
        # 走文本层表格管线的行高扩展 + 矢线重绘 + 正常填充译文。
        blocks = [
            pdfio.Block(text="A", page=0, x0=1, y0=1, x1=10, y1=9, size=9.0, ocr=True),
            pdfio.Block(text="B", page=0, x0=51, y0=1, x1=60, y1=9, size=9.0, ocr=True),
            pdfio.Block(text="C", page=0, x0=1, y0=11, x1=10, y1=19, size=9.0, ocr=True),
            pdfio.Block(text="D", page=0, x0=51, y0=11, x1=60, y1=19, size=9.0, ocr=True),
        ]
        style = {"rows_pts": [0, 10, 20], "cols_pts": [0, 50, 100],
                 "merged": [], "header_rows": [], "header_cols": [], "align": []}
        rebuilt, tables, mapping = pdfio._rebuild_ocr_table_blocks(blocks, style)
        # 每个块的 bbox 重排到对应 AI 单元格（含 _TABLE_CELL_PAD）。
        self.assertAlmostEqual(rebuilt[0].x0, 2.0)     # 0 + 2
        self.assertAlmostEqual(rebuilt[0].y0, 0.0)
        self.assertAlmostEqual(rebuilt[0].x1, 48.0)    # 50 - 2
        self.assertAlmostEqual(rebuilt[1].x0, 52.0)    # 50 + 2
        self.assertEqual(len(tables), 1)
        self.assertEqual(len(tables[0]["rows"]), 2)    # 2 row gaps
        self.assertEqual(len(mapping), 4)
        # 每个块都映射到 AI 网格的单元格。
        self.assertEqual(mapping[0], (0, 0, 0))
        self.assertEqual(mapping[1], (0, 0, 1))
        self.assertEqual(mapping[2], (0, 1, 0))
        self.assertEqual(mapping[3], (0, 1, 1))

    def test_rebuild_semantics_composes_grid(self):
        # Layer-③: 重建后构建语义 StructTable（cells/merged/header），供审计。
        blocks = [
            pdfio.Block(text="A", page=0, x0=1, y0=1, x1=10, y1=9, size=9.0, ocr=True),
            pdfio.Block(text="B", page=0, x0=51, y0=1, x1=60, y1=9, size=9.0, ocr=True),
        ]
        style = {"rows_pts": [0, 10], "cols_pts": [0, 50, 100],
                 "merged": [{"r": 0, "c": 0, "row_span": 1, "col_span": 2}],
                 "header_rows": [0], "header_cols": [], "align": [], "non_text": []}
        sem = pdfio.rebuild_semantics(blocks, style)
        self.assertEqual(sem["rows"], 1)
        self.assertEqual(sem["cols"], 2)
        self.assertEqual(sem["header_rows"], [0])
        self.assertEqual(sem["merged"], [{"r": 0, "c": 0, "row_span": 1, "col_span": 2}])
        # 块 A/B 映射到 (0,0)/(0,1)。
        self.assertEqual(sem["cells"][0][0], 0)
        self.assertEqual(sem["cells"][0][1], 1)

    def test_valid_rebuild_accepts_when_numerics_stay_in_cells(self):
        # Layer-④: 数字块都映射到重建单元格 → valid_rebuild True。
        blocks = [
            pdfio.Block(text="1,234.56", page=0, x0=1, y0=1, x1=20, y1=9, size=9.0, ocr=True),
            pdfio.Block(text="label", page=0, x0=51, y0=1, x1=90, y1=9, size=9.0, ocr=True),
        ]
        style = {"rows_pts": [0, 10], "cols_pts": [0, 50, 100],
                 "merged": [], "header_rows": [], "header_cols": [], "align": [], "non_text": []}
        self.assertTrue(pdfio.valid_rebuild(blocks, style))

    def test_valid_rebuild_rejects_when_numerics_flee(self):
        # Layer-④: 数字块无法映射到任何重建单元格 → 回退（fail-closed）。
        blocks = [
            pdfio.Block(text="1,234.56", page=0, x0=1, y0=1, x1=20, y1=9, size=9.0, ocr=True),
        ]
        style = {"rows_pts": [0, 10], "cols_pts": [0, 50, 100],
                 "merged": [], "header_rows": [], "header_cols": [], "align": [], "non_text": []}
        with mock.patch.object(pdfio, "_rebuild_ocr_table_blocks",
                               return_value=([], [], {})):
            self.assertFalse(pdfio.valid_rebuild(blocks, style))

    def test_in_non_text_detects_signature_region(self):
        # 非文本区域（签名/印章）内的块应被识别为保持原样（不重建、不翻译）。
        non_text = [(50, 690, 550, 750, "signature")]
        sig = pdfio.Block(text="sig", page=0, x0=100, y0=700, x1=500, y1=740, size=9.0, ocr=True)
        self.assertTrue(pdfio._in_non_text(sig, non_text))
        data = pdfio.Block(text="data", page=0, x0=100, y0=100, x1=200, y1=120, size=9.0, ocr=True)
        self.assertFalse(pdfio._in_non_text(data, non_text))

    def test_calibrate_table_grid_reports_prepended_shift(self):
        # Layer-① 回归：几何校准在最左补了一列时，必须报告平移量（1），调用方据此
        # 平移 AI 的 align/merged/header 索引；只补右侧列则不平移。
        blocks = [
            pdfio.Block(text="a", page=0, x0=50, y0=5, x1=90, y1=15, size=9.0, ocr=True),
            pdfio.Block(text="b", page=0, x0=210, y0=5, x1=290, y1=15, size=9.0, ocr=True),
        ]
        cols, rows, col_shift, row_shift = pdfio._calibrate_table_grid(
            blocks, [100.0, 200.0, 300.0], [0.0, 10.0, 20.0])
        self.assertEqual(col_shift, 1)                 # 50 < 100 - 4 → 前插一列
        self.assertEqual(cols, [50.0, 100.0, 200.0, 294.0])  # 右界收到内容右缘 290+4
        self.assertEqual(row_shift, 0)                 # y 在 AI 行跨度内
        self.assertEqual(rows, [0.0, 10.0, 19.0])      # 底行界收到内容底缘 15+4

    def test_calibrate_table_grid_trims_overwide_right_boundary(self):
        # 回归「表格超宽」：AI 把最右列画到页面右缘（如 592.7），而内容实际到 533，
        # 几何校准必须把右界收回到内容右缘，而不是一味向外扩。
        blocks = [
            pdfio.Block(text="a", page=0, x0=50, y0=1, x1=90, y1=9, size=9.0, ocr=True),
            pdfio.Block(text="456", page=0, x0=400, y0=1, x1=533, y1=9, size=9.0, ocr=True),
        ]
        cols, _rows, col_shift, _row_shift = pdfio._calibrate_table_grid(
            blocks, [50.0, 300.0, 592.7], [0.0, 10.0, 20.0])
        self.assertEqual(cols[-1], 537.0)      # 533 + gap(4)，不是 592.7
        self.assertEqual(col_shift, 0)         # 右侧收回不平移列号
        # 数字块仍映射到最右列，数字不丢。
        style = {"rows_pts": [0.0, 10.0, 20.0], "cols_pts": cols, "merged": [],
                 "header_rows": [], "header_cols": [], "align": [], "non_text": []}
        rebuilt, _t, mapping = pdfio._rebuild_ocr_table_blocks(blocks, style)
        self.assertIn(1, mapping)              # 数字块仍落在网格内
        self.assertTrue(any(c.isdigit() for c in rebuilt[1].text))

    def test_calibrate_table_grid_handles_empty_blocks(self):
        # 无 OCR 块时不得因 min() 空序列抛异常：原样返回 AI 网格、平移为 0。
        cols, rows, col_shift, row_shift = pdfio._calibrate_table_grid(
            [], [1.0, 2.0], [3.0, 4.0])
        self.assertEqual(cols, [1.0, 2.0])
        self.assertEqual(rows, [3.0, 4.0])
        self.assertEqual((col_shift, row_shift), (0, 0))

    def test_rebuild_align_follows_column_when_calibration_prepends(self):
        # 回归：几何校准在最左补列后，AI 说的「第 1 列右对齐」必须仍落到真正含该
        # 数字块的那一列（重建后的第 2 列），而不是错位到相邻列。
        blocks = [
            pdfio.Block(text="甲乙", page=0, x0=50, y0=1, x1=90, y1=9, size=9.0, ocr=True),
            pdfio.Block(text="456", page=0, x0=210, y0=1, x1=290, y1=9, size=9.0, ocr=True),
        ]
        style = {"rows_pts": [0.0, 10.0, 20.0], "cols_pts": [100.0, 200.0, 300.0],
                 "align": [{"col": 1, "dir": "right"}], "merged": [],
                 "header_rows": [0], "header_cols": [], "non_text": []}
        rebuilt, _tables, mapping = pdfio._rebuild_ocr_table_blocks(blocks, style)
        self.assertEqual(mapping[1], (0, 0, 2))        # 数字块落到重建的第 2 列
        self.assertEqual(rebuilt[1].align, "right")    # 仍右对齐（曾错到 col 1）
        self.assertEqual(rebuilt[0].align, "left")     # 前插列不是 AI 指定的右对齐列

    def test_rebuild_semantics_shifts_header_and_merge_after_prepend(self):
        # 回归：几何校准前插列后，merged/header_cols 索引必须一起平移，否则审计
        # 看到的合并范围/表头位置会错位一位。
        blocks = [
            pdfio.Block(text="x", page=0, x0=50, y0=5, x1=90, y1=15, size=9.0, ocr=True),
            pdfio.Block(text="y", page=0, x0=210, y0=5, x1=290, y1=15, size=9.0, ocr=True),
        ]
        style = {"rows_pts": [0.0, 10.0, 20.0], "cols_pts": [100.0, 200.0, 300.0],
                 "merged": [{"r": 0, "c": 1, "row_span": 1, "col_span": 2}],
                 "header_rows": [0], "header_cols": [1], "align": [], "non_text": []}
        sem = pdfio.rebuild_semantics(blocks, style)
        self.assertEqual(sem["cols"], 3)
        self.assertEqual(sem["header_cols"], [2])              # 1 + col_shift(1)
        self.assertEqual(sem["merged"][0]["c"], 2)             # 1 + col_shift(1)
        self.assertEqual(sem["header_rows"], [0])              # 无行前插

    def test_rebuild_pages_does_not_whiteout_non_ocr_page(self):
        # 防御回归：rebuild_pages 若含非 OCR 表格页的索引，不得整页涂白（会抹掉
        # 文本层内容）。白矩形只在「本页确实是 OCR 位图表格」时才画。
        src = _OUT / "rebuild_guard_src.pdf"
        build_sample_pdf(src, pages=1)
        doc = pdfio.extract_document_text(str(src), ocr=False, log=lambda m: None)
        trans = [b.text for b in doc.pages[0]]
        out = _OUT / "rebuild_guard_out.pdf"
        style = {"rows_pts": [0.0, 100.0], "cols_pts": [0.0, 300.0],
                 "merged": [], "header_rows": [], "header_cols": [], "align": [],
                 "non_text": []}
        pdfio.save_translated_pdf(str(src), doc.pages, [trans], str(out), "English",
                                  rebuild_pages={0: style})
        d = fitz.open(str(out))
        try:
            page = d[0]
            full_white = [
                dr for dr in page.get_drawings()
                if dr.get("type") == "f" and dr.get("fill")
                and all(abs(c - 1.0) < 0.01 for c in dr["fill"])
                and dr["rect"].width > page.rect.width * 0.9
                and dr["rect"].height > page.rect.height * 0.9
            ]
            self.assertEqual(full_white, [])
        finally:
            d.close()

    def test_score_rebuild_counts_cell_collisions(self):
        # 确定性打分：删掉一条内竖线（两列合并）→ 两个数字挤进同一格 → collisions 上升。
        blocks = [
            pdfio.Block(text="1", page=0, x0=1, y0=1, x1=20, y1=9, size=9.0, ocr=True),
            pdfio.Block(text="2", page=0, x0=55, y0=1, x1=75, y1=9, size=9.0, ocr=True),
        ]
        good = {"rows_pts": [0, 10], "cols_pts": [0, 50, 100], "merged": [],
                "header_rows": [], "header_cols": [], "align": [], "non_text": []}
        merged = {"rows_pts": [0, 10], "cols_pts": [0, 100], "merged": [],
                  "header_rows": [], "header_cols": [], "align": [], "non_text": []}
        self.assertEqual(pdfio.score_rebuild(blocks, good)[0], 0)
        self.assertEqual(pdfio.score_rebuild(blocks, merged)[0], 1)
        # 数字落格率都是 1.0 —— 说明单靠 valid_rebuild 分不出好坏，必须靠 collisions。
        self.assertEqual(pdfio.score_rebuild(blocks, good)[2], 1.0)
        self.assertEqual(pdfio.score_rebuild(blocks, merged)[2], 1.0)

    def test_best_rebuild_prefers_fewer_collisions(self):
        # 多个候选确定性选优：合并列的候选 collisions 更高，应被淘汰。
        blocks = [
            pdfio.Block(text="1", page=0, x0=1, y0=1, x1=20, y1=9, size=9.0, ocr=True),
            pdfio.Block(text="2", page=0, x0=55, y0=1, x1=75, y1=9, size=9.0, ocr=True),
        ]
        good = {"rows_pts": [0, 10], "cols_pts": [0, 50, 100], "merged": [],
                "header_rows": [], "header_cols": [], "align": [], "non_text": []}
        merged = {"rows_pts": [0, 10], "cols_pts": [0, 100], "merged": [],
                  "header_rows": [], "header_cols": [], "align": [], "non_text": []}
        best, score = pdfio.best_rebuild(blocks, [merged, good])
        self.assertIs(best, good)
        self.assertEqual(score[0], 0)
        self.assertEqual(pdfio.best_rebuild(blocks, [])[0], None)

    def test_score_rebuild_ranks_unusable_style_last(self):
        # 结构不可用时打分应为最差（供选优淘汰），而不是抛异常。
        blocks = [pdfio.Block(text="1", page=0, x0=1, y0=1, x1=20, y1=9, size=9.0, ocr=True)]
        with mock.patch.object(pdfio, "_rebuild_ocr_table_blocks",
                               side_effect=ValueError("boom")):
            collisions, overflow, ratio = pdfio.score_rebuild(blocks, {})
        self.assertGreater(collisions, 10 ** 6)
        self.assertEqual(ratio, 0.0)

    def test_rebalance_rebuild_widens_numeric_column(self):
        # 数字感知列宽自适应：数字列过窄（数字溢出）→ 加宽到容纳最宽数字，总表宽不变。
        blocks = [
            pdfio.Block(text="项目", page=0, x0=2, y0=1, x1=300, y1=9, size=9.0, ocr=True),
            pdfio.Block(text="12,345,678,901.12", page=0, x0=310, y0=1, x1=345, y1=9,
                        size=6.0, ocr=True),
        ]
        style = {"rows_pts": [0, 10], "cols_pts": [0, 305, 350], "merged": [],
                 "header_rows": [], "header_cols": [], "align": [], "non_text": []}
        rebuilt, tables, mapping = pdfio._rebuild_ocr_table_blocks(blocks, style)
        font = fitz.Font("cjk")
        col_boxes, new_col_edges = pdfio._rebalance_rebuild_columns(
            tables, mapping, rebuilt, ["项目", "12,345,678,901.12"], font)
        edges = new_col_edges[0]
        src_edges = tables[0]["col_edges"]   # 校准后的总宽（AI 350 被校准收到 349）
        self.assertAlmostEqual(edges[0], src_edges[0], delta=0.01)
        self.assertAlmostEqual(edges[-1], src_edges[-1], delta=0.01)  # 总表宽不变
        self.assertGreater(edges[2] - edges[1],
                           src_edges[2] - src_edges[1])   # 数字列被加宽
        # 数字列宽度 ≥ 最宽数字（单行，不破行）。
        avail_w = col_boxes[1][1] - col_boxes[1][0]
        need_w = font.text_length("12,345,678,901.12", fontsize=6.0)
        self.assertGreaterEqual(avail_w, need_w)

    def test_unique_path_appends_number_when_exists(self):
        # 导出不覆盖重名文件：test_English.pdf 已存在 → test_English(1).pdf。
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            base = d / "test_English.pdf"
            self.assertEqual(pdfio.unique_path(base), base)
            base.write_bytes(b"x")
            self.assertEqual(pdfio.unique_path(base), d / "test_English(1).pdf")
            (d / "test_English(1).pdf").write_bytes(b"x")
            self.assertEqual(pdfio.unique_path(base), d / "test_English(2).pdf")


class TableCellFitTest(unittest.TestCase):
    """A table cell's translation shrinks onto ONE line (instead of wrapping and
    growing the row) when it fits the widened cell; a cell that cannot fit at
    the readability floor falls back to wrapping, and non-table blocks still
    wrap as before."""

    def _line(self, x0, x1, y0, y1, text):
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "size": 9.0, "bold": False, "color": 0, "text": text}

    def test_build_table_blocks_widens_cells_and_sets_in_table(self):
        # Native ruled-table cells are flagged in_table and widened to the whole
        # cell (minus a small gutter) so a longer translation can use the column.
        cell_rects = [fitz.Rect(100, 100, 300, 120)]
        lines = [
            self._line(150, 210, 105, 115, "名称"),
            self._line(150, 250, 105, 115, "1,234,567"),
        ]
        blocks = pdfio._build_table_blocks(lines, [], 0, 600, 0, cell_rects)
        self.assertEqual(len(blocks), 2)
        by_text = {b.text: b for b in blocks}
        name = by_text["名称"]
        num = by_text["1,234,567"]
        self.assertTrue(name.in_table)
        self.assertTrue(num.in_table)
        # Widened to the cell width (with the small gutter), not the text extent.
        self.assertAlmostEqual(name.x0, 100 + pdfio._TABLE_CELL_PAD, delta=0.01)
        self.assertAlmostEqual(name.x1, 300 - pdfio._TABLE_CELL_PAD, delta=0.01)
        # Text cells centre; figure cells right-align.
        self.assertEqual(name.align, "center")
        self.assertEqual(num.align, "right")

    def _line_with_spans(self, x0, x1, y0, y1, text, spans):
        d = self._line(x0, x1, y0, y1, text)
        d["spans"] = spans
        return d

    def test_build_table_blocks_splits_label_and_value_cells(self):
        # A source row that reports label+value on ONE visual line is really two
        # table cells (a label column and a value column).  It must come out as
        # two blocks, each pinned to its own cell — not the whole line stuffed
        # into whichever cell its centre happens to land in (which merged the
        # two columns and dropped the label column).
        cell_rects = [
            fitz.Rect(100, 100, 200, 120),   # label column
            fitz.Rect(200, 100, 400, 120),   # value column
        ]
        line = self._line_with_spans(
            150, 300, 105, 115,
            "Capacity: Pilot + Copilot + Two Passengers",
            [
                (150.0, 105.0, 195.0, 115.0, "Capacity:"),
                (210.0, 105.0, 300.0, 115.0, "Pilot + Copilot + Two Passengers"),
            ],
        )
        blocks = pdfio._build_table_blocks([line], [], 0, 600, 0, cell_rects)
        self.assertEqual(len(blocks), 2)
        by_text = {b.text: b for b in blocks}
        label = by_text["Capacity:"]
        value = by_text["Pilot + Copilot + Two Passengers"]
        self.assertTrue(label.in_table)
        self.assertTrue(value.in_table)
        # Each block is widened to its own cell, not to the merged line's extent.
        self.assertAlmostEqual(label.x0, 100 + pdfio._TABLE_CELL_PAD, delta=0.01)
        self.assertAlmostEqual(label.x1, 200 - pdfio._TABLE_CELL_PAD, delta=0.01)
        self.assertAlmostEqual(value.x0, 200 + pdfio._TABLE_CELL_PAD, delta=0.01)
        self.assertAlmostEqual(value.x1, 400 - pdfio._TABLE_CELL_PAD, delta=0.01)

    def test_build_table_blocks_ignores_narrow_gutter_cell(self):
        # A thin gutter between the label and value columns must never become a
        # cell a long label is fitted into.  The pre-fix code pinned a long
        # label row (whose whole-line centre lands in the gutter) to the ~1pt
        # gutter and rendered the translation as a vertical stack.
        cell_rects = [
            fitz.Rect(100, 100, 200, 120),   # label column
            fitz.Rect(200, 100, 205, 120),   # narrow gutter (sliver, < MIN_WIDTH)
            fitz.Rect(205, 100, 400, 120),   # value column
        ]
        line = self._line_with_spans(
            150, 300, 105, 115,
            "Max Gross Weight: 4,000 lbs / 1,814 kg",
            [
                (170.0, 105.0, 199.0, 115.0, "Max Gross Weight:"),
                (240.0, 105.0, 300.0, 115.0, "4,000 lbs / 1,814 kg"),
            ],
        )
        blocks = pdfio._build_table_blocks([line], [], 0, 600, 0, cell_rects)
        self.assertEqual(len(blocks), 2)
        by_text = {b.text: b for b in blocks}
        label = by_text["Max Gross Weight:"]
        value = by_text["4,000 lbs / 1,814 kg"]
        # Neither is crushed into the ~5pt gutter.
        self.assertGreater(label.x1 - label.x0, 50.0)
        self.assertGreater(value.x1 - value.x0, 50.0)

    def test_fit_block_puts_table_cell_on_one_line(self):
        # A long English translation used to wrap inside the narrow cell and push
        # the rows below it down.  An in_table cell shrinks the font instead.
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=100, y0=100, x1=300, y1=120,
            size=9.0, align="left", bold=False, single_line=True, in_table=True,
        )
        long_name = "Wenling Municipal State-owned Assets Management Co., Ltd."
        lines, fs = pdfio._fit_block(block, font, long_name)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines, [long_name])
        # The font shrank (never below the table floor) to fit the cell width.
        self.assertLess(fs, 9.0)
        self.assertGreaterEqual(fs, pdfio._MIN_TABLE_READABLE)
        self.assertLessEqual(font.text_length(long_name, fontsize=fs), 200.0)

    def test_fit_block_too_narrow_table_cell_wraps_at_readable_floor(self):
        # A cell so narrow that even the readability floor cannot hold a single
        # line used to stay ONE line at the 3pt floor — the slug-cell soft spot
        # (44 cells < 5pt on the scan, 5 at 3.0).  The cell now wraps at the
        # readability floor instead of collapsing to an illegible single line.
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=100, y0=100, x1=160, y1=120,
            size=9.0, align="left", bold=False, single_line=True, in_table=True,
        )
        long_name = "Wenling Municipal State-owned Assets Management Co., Ltd."
        lines, fs = pdfio._fit_block(block, font, long_name)
        self.assertGreater(len(lines), 1)
        self.assertEqual(fs, pdfio._MIN_TABLE_READABLE)
        self.assertEqual("".join(lines).replace(" ", ""), long_name.replace(" ", ""))

    def test_fit_block_keep_font_keeps_size_and_wraps(self):
        # Rebuilt vector-table cell (keep_font): keep the original font size and
        # grow the row to the wrapped line count instead of shrinking.
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=100, y0=100, x1=250, y1=120,
            size=9.0, align="left", bold=False, single_line=True, in_table=True,
        )
        block = pdfio.replace(block, keep_font=True)
        long_name = "Wenling Municipal State-owned Assets Management Co., Ltd."
        lines, fs = pdfio._fit_block(block, font, long_name)
        self.assertEqual(fs, 9.0)           # original font size kept, not shrunk
        self.assertGreater(len(lines), 1)   # wrapped instead of collapsed to 3pt
        self.assertEqual("".join(lines).replace(" ", ""), long_name.replace(" ", ""))

    def test_fit_block_band_wraps_at_readable_floor_when_band_is_plenty(self):
        # A scanned grid cell carries a fit_height band down to the row below
        # (the raster table lines cannot move, so the wrap must stay inside it).
        # A wide-enough band means 2-3 lines at the full readability floor.
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=100, y0=100, x1=160, y1=120,
            size=9.0, align="left", bold=False, single_line=True, in_table=True,
            fit_height=40.0,
        )
        long_name = "Wenling Municipal State-owned Assets Management Co., Ltd."
        lines, fs = pdfio._fit_block(block, font, long_name)
        self.assertGreater(len(lines), 1)
        self.assertEqual(fs, pdfio._MIN_TABLE_READABLE)
        self.assertLessEqual(
            pdfio._wrapped_height(
                font, lines, fs, pdfio._line_leading(font, in_table=True, n_lines=len(lines))
            ),
            40.0,
        )
        self.assertEqual("".join(lines).replace(" ", ""), long_name.replace(" ", ""))

    def test_fit_block_band_descends_when_the_band_is_tight(self):
        # A hole as tight as the scan's real row pitch (~18pt) holds only 2
        # lines at 6pt: the fit downsizes until the wrapped height fits the
        # band (but never below the absolute floor) rather than crossing the
        # grid line to keep a bigger font.
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=100, y0=100, x1=160, y1=120,
            size=9.0, align="left", bold=False, single_line=True, in_table=True,
            fit_height=17.5,
        )
        long_name = "Wenling Municipal State-owned Assets Management Co., Ltd."
        lines, fs = pdfio._fit_block(block, font, long_name)
        self.assertGreater(len(lines), 1)
        self.assertLessEqual(
            round(pdfio._wrapped_height(
                font, lines, fs, pdfio._line_leading(font, in_table=True, n_lines=len(lines))
            ), 2),
            17.5,
        )
        self.assertGreaterEqual(fs, pdfio._MIN_TABLE_FLOOR)
        self.assertLess(fs, pdfio._MIN_TABLE_READABLE)
        self.assertEqual("".join(lines).replace(" ", ""), long_name.replace(" ", ""))

    def test_non_table_block_still_wraps(self):
        # A flowing paragraph (not a table cell) is unaffected: it wraps and keeps
        # its own box rather than being forced onto one line.
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="", page=0, x0=50, y0=100, x1=300, y1=140,
            size=11.0, align="left", bold=False, single_line=False, in_table=False,
        )
        text = ("本飞机专为单飞行员操作而开发，并经过相应调整以更好地模拟真实环境。"
                "其制作综合了多个真实世界的数据点。")
        lines, _fs = pdfio._fit_block(block, font, text)
        self.assertGreater(len(lines), 1)
        self.assertEqual("".join(lines), text)

    def test_fit_block_preserves_two_line_source_count(self):
        # A cell whose source spans exactly TWO lines keeps exactly two lines:
        # the fitter splits and rebalances so the count matches the source (and
        # the model's output is not re-wrapped into a different line count).
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="长期投资\n合计", page=0, x0=100, y0=100, x1=300, y1=140,
            size=9.0, align="left", bold=False, single_line=False, in_table=True,
        )
        text = ("Wenling Municipal State-owned Assets Management Co., Ltd. "
                "Total long-term investments")
        lines, fs = pdfio._fit_block(block, font, text)
        self.assertEqual(len(lines), 2)
        # No character was lost or fabricated across the two lines.
        self.assertEqual("".join(lines).replace(" ", ""), text.replace(" ", ""))
        self.assertGreaterEqual(fs, pdfio._MIN_TABLE_READABLE)

    def test_fit_block_two_lines_shrink_below_readable_floor_if_needed(self):
        # Line count wins over the readability floor: a 2-line source stays two
        # lines even in a too-narrow cell (the exact-n rebalance keeps the count
        # and lets the font drop to the absolute floor).
        font = fitz.Font("cjk")
        block = pdfio.Block(
            text="净亏损\n小计", page=0, x0=100, y0=100, x1=180, y1=140,
            size=9.0, align="left", bold=False, single_line=False, in_table=True,
        )
        text = ("Net loss per share attributable to the shareholders of the "
                "Company after taking into account the discontinued operations")
        lines, fs = pdfio._fit_block(block, font, text)
        self.assertEqual(len(lines), 2)
        self.assertEqual("".join(lines).replace(" ", ""), text.replace(" ", ""))
        self.assertLess(fs, pdfio._MIN_TABLE_READABLE)
        self.assertGreaterEqual(fs, pdfio._MIN_TABLE_FLOOR)

    def test_multiline_table_cell_uses_min_line_leading(self):
        # A cell whose translation wraps to more than one line renders with the
        # *minimum* line spacing — ``_TABLE_CELL_LEADING`` (1.0× the font size)
        # — instead of the loose 1.35× leading that paragraphs keep.  A long
        # translation in a narrow column must stay compact so the wrap fits the
        # row band without growing the row or pushing the rows below it down.
        font = fitz.Font("cjk")
        tight = pdfio._TABLE_CELL_LEADING
        self.assertLess(tight, pdfio._LOOSE_LEADING)
        # Multi-line table cell -> the tight leading.
        self.assertEqual(pdfio._line_leading(font, in_table=True, n_lines=2), tight)
        self.assertEqual(pdfio._line_leading(font, in_table=True, n_lines=3), tight)
        # A single line has no inter-line gap, so it keeps the loose value.
        self.assertEqual(pdfio._line_leading(font, in_table=True, n_lines=1),
                         pdfio._LOOSE_LEADING)
        # Paragraph (non-table) blocks keep the loose leading.
        self.assertEqual(pdfio._line_leading(font, in_table=False, n_lines=3),
                         pdfio._LOOSE_LEADING)
        # A wrapped cell measured through ``_fit_block`` is also measured tight,
        # so the row-height re-layout agrees with the drawing pass.
        cell = pdfio.Block(
            text="净亏损\n小计", page=0, x0=100, y0=100, x1=180, y1=140,
            size=9.0, align="left", bold=False, single_line=False, in_table=True,
        )
        wrapped = ("Net loss per share attributable to the shareholders of the "
                   "Company after taking into account the discontinued operations")
        lines, fs = pdfio._fit_block(cell, font, wrapped)
        self.assertGreater(len(lines), 1)
        # The tight leading packs the lines tighter than the loose default.
        self.assertLess(
            pdfio._measure_block_height(cell, font, wrapped),
            pdfio._wrapped_height(font, lines, fs),
        )


class OcrGridTest(unittest.TestCase):
    """Scanned financial tables (OCR blocks, no text layer / vector rules) must
    be rebuilt into a grid instead of collapsing into a jumble."""

    def _items(self):
        # A miniature balance sheet: a label column, a "行次" number column and
        # two numeric columns (合并 / 母公司).  Item tuple is (y0, x0, x1, y1, text).
        return [
            # header row
            (100, 78, 180, 112, "项目"), (100, 190, 212, 112, "行次"),
            (100, 240, 292, 112, "合并"), (100, 320, 372, 112, "母公司"),
            # data row 1 — label + two figures
            (130, 84, 164, 148, "现金及存放中央银行款项"),
            (130, 226, 290, 148, "17,485,938,749.91"),
            (130, 306, 370, 148, "14,944,565,492.79"),
            # data row 2
            (165, 84, 128, 180, "存放同业款项"),
            (165, 230, 289, 180, "3,702,726,474.45"),
            (165, 310, 369, 180, "1,386,040,370.31"),
            # data row 3 — a label, a line number, one figure
            (200, 84, 141, 212, "发放贷款和垫款"),
            (200, 190, 212, 212, "8"),
            (200, 222, 289, 212, "224,464,860,917.53"),
        ]

    def test_is_numeric_cell(self):
        self.assertTrue(pdfio._is_numeric_cell("17,485,938,749.91"))
        # OCR mis-reads can inject spaces inside a figure; still numeric.
        self.assertTrue(pdfio._is_numeric_cell("65, 334, 085.99"))
        self.assertTrue(pdfio._is_numeric_cell("(1,234.56)"))
        self.assertTrue(pdfio._is_numeric_cell("-789,702,296.83"))
        self.assertTrue(pdfio._is_numeric_cell("5.6"))
        # Labels / ordinals are not numeric.
        self.assertFalse(pdfio._is_numeric_cell("营业收入"))
        self.assertFalse(pdfio._is_numeric_cell("(四)"))
        self.assertFalse(pdfio._is_numeric_cell("现金及存放中央银行款项"))

    def test_reconstruct_ocr_grid_is_row_major_and_right_aligns_numbers(self):
        blocks, tables = pdfio._reconstruct_ocr_grid(self._items())
        self.assertTrue(blocks)
        self.assertEqual(len(tables), 1)
        texts = [b.text for b in blocks]
        # Row-major: the label column is not all emitted before the figures.
        self.assertLess(texts.index("现金及存放中央银行款项"), texts.index("17,485,938,749.91"))
        # Numeric cells are right-aligned; labels stay left.
        by_text = {b.text: b for b in blocks}
        self.assertEqual(by_text["17,485,938,749.91"].align, "right")
        self.assertEqual(by_text["14,944,565,492.79"].align, "right")
        self.assertEqual(by_text["现金及存放中央银行款项"].align, "left")
        # The numeric cell keeps its own right edge (aligned within its column),
        # while the label keeps its own glyph box as the bbox (the cover/redact
        # step erases exactly the printed label pixels — never a neighbouring
        # cell) and carries the whole column as draw room in ``fit_width``.
        self.assertAlmostEqual(by_text["17,485,938,749.91"].x1, 290.0, delta=1.0)
        self.assertAlmostEqual(by_text["现金及存放中央银行款项"].x1, 164.0, delta=1.0)
        self.assertGreater(
            by_text["现金及存放中央银行款项"].fit_width,
            by_text["现金及存放中央银行款项"].x1 - by_text["现金及存放中央银行款项"].x0,
        )

    def test_reconstruct_ocr_tables_maps_all_cells(self):
        blocks, _ = pdfio._reconstruct_ocr_grid(self._items())
        tables = pdfio._reconstruct_ocr_tables(blocks)
        self.assertEqual(len(tables), 1)
        self.assertEqual(len(tables[0]["rows"]), 4)  # header + 3 data rows
        mapping = pdfio._map_blocks_to_table_cells(blocks, tables)
        self.assertEqual(len(mapping), len(blocks))

    def test_grid_subcolumn_header_gets_row_gap_as_fit_width(self):
        # The "合并"/"母公司" header cells live inside a figure sub-column whose
        # own OCR box only encloses the two printed characters; the translation
        # ("Consolidated") must be able to use the empty gap up to the next cell
        # in the row, or it is crushed to the 3pt floor.
        blocks, _ = pdfio._reconstruct_ocr_grid(self._items())
        by_text = {b.text: b for b in blocks}
        header = by_text["合并"]
        self.assertGreater(header.fit_width, header.x1 - header.x0)
        self.assertAlmostEqual(header.fit_width, 320.0 - 2.0 - 240.0, delta=1.0)

    def test_grid_cells_get_row_pitch_as_fit_height(self):
        # Each cell of a grid row may draw down to the *next* row's top (minus a
        # margin for the raster line): a translation too long for a readable
        # single line wraps into the row gap instead of crossing the table line
        # below (scan raster lines cannot move).  The last row has no band.
        blocks, _ = pdfio._reconstruct_ocr_grid(self._items())
        by_text = {b.text: b for b in blocks}
        cell = by_text["现金及存放中央银行款项"]     # row top 130, next row top 165
        self.assertAlmostEqual(cell.fit_height, 165.0 - 130.0 - 1.5, delta=0.01)
        self.assertGreater(cell.fit_height, cell.y1 - cell.y0)
        self.assertEqual(by_text["发放贷款和垫款"].fit_height, 0.0)  # last row

    def test_grid_label_fit_width_stops_before_the_note_marker(self):
        # A row with the 附注 "(二)" band inside the label column: the label
        # keeps its own glyph box as bbox, its draw room stops 2pt before the
        # note (a long translation must not slide under "(二)" and have its tail
        # sliced by the note's white cover), and the note keeps its own box,
        # centred where the source printed it.
        items = self._items() + [
            (140, 84, 141, 152, "拆出资金"),
            (140, 170, 188, 152, "(二)"),
            (140, 230, 289, 152, "3,702,726,474.45"),
        ]
        blocks, _ = pdfio._reconstruct_ocr_grid(items)
        by_text = {b.text: b for b in blocks}
        label, note = by_text["拆出资金"], by_text["(二)"]
        self.assertAlmostEqual(label.x1, 141.0, delta=1.0)
        self.assertAlmostEqual(label.fit_width, 170.0 - 2.0 - 84.0, delta=1.0)
        self.assertEqual(label.align, "left")
        self.assertAlmostEqual(note.x1, 188.0, delta=1.0)
        self.assertEqual(note.align, "center")

    def test_grid_merges_split_label_fragments(self):
        # RapidOCR splits one printed label (营业利润（亏损以"一"号填列）) into two
        # close items; the fragments must become ONE cell so the model receives
        # the whole label and the fitter sees the whole run — while a "(十六)"
        # note marker stays its own cell.
        items = [
            (100, 78, 180, 112, "项目"), (100, 320, 372, 112, "母公司"),
            (130, 84, 131, 148, "营业利润（亏损以"),
            (130, 148, 168, 148, "号填列）"),
            (130, 230, 289, 148, "207,098,342.00"),
            (165, 84, 128, 180, "利息净收入"),
            (165, 170, 188, 180, "（十六）"),
            (165, 230, 289, 180, "84,528,349.88"),
        ]
        blocks, _ = pdfio._reconstruct_ocr_grid(items)
        texts = [x.text for x in blocks]
        self.assertIn("营业利润（亏损以号填列）", texts)
        self.assertNotIn("号填列）", texts)
        self.assertIn("利息净收入", texts)
        self.assertIn("（十六）", texts)
        merged = next(b for b in blocks if b.text == "营业利润（亏损以号填列）")
        # The merged run's box spans both fragments, so the fit sees the run.
        self.assertAlmostEqual(merged.x1, 168.0, delta=1.0)

    def test_grid_does_not_merge_a_note_marker_into_the_label(self):
        # The marker guard is symmetric enough that 拆出资金 + (二) never merge,
        # even though "(二)" sits 30pt from the label (a 40pt+ marker would too).
        items = [
            (100, 78, 180, 112, "项目"), (100, 320, 372, 112, "母公司"),
            (130, 84, 128, 148, "拆出资金"),
            (130, 170, 188, 148, "(二)"),
            (130, 230, 289, 148, "97,923,282.04"),
        ]
        blocks, _ = pdfio._reconstruct_ocr_grid(items)
        texts = [x.text for x in blocks]
        self.assertIn("拆出资金", texts)
        self.assertIn("(二)", texts)
        self.assertNotEqual(texts.index("拆出资金"), texts.index("(二)"))


class NumberAtomicityTest(unittest.TestCase):
    """Figures are never split mid-number by the wrap machinery.

    Regression: on the statement pages the value 292,712,933,925.17 was drawn
    as ``292,712,933,925.1`` on one line and ``7`` on the next — a reader sees
    a decimal point that grew or lost a digit, and an amount that is wrong.
    """

    def test_number_in_narrow_box_stays_whole(self):
        font = fitz.Font("cjk")
        text = "292,712,933,925.17"
        lines = pdfio._wrap(font, text, 25.0, 11.0)  # far narrower than the text
        self.assertEqual([text], lines)

    def test_number_inside_prose_wraps_at_word_boundaries(self):
        font = fitz.Font("cjk")
        text = "总资产达 292,712,933,925.17 元，较上年末增长。"
        lines = pdfio._wrap(font, text, 40.0, 11.0)
        # No character was lost (the wrap drops inter-word spaces at line
        # breaks, so compare content digit-for-digit), no line holds a
        # *partial* amount, and the whole figure sits on one line.
        self.assertEqual(text.replace(" ", ""), "".join(lines))
        self.assertTrue(any("292,712,933,925.17" in line for line in lines))
        for line in lines:
            with self.subTest(line=line):
                if "292,712,933,925.17" not in line:
                    self.assertNotIn("292,712", line)

    def test_stray_spaces_around_separators_are_merged(self):
        # ``65, 334, 085.99`` was one amount OCR padded; splitting it at the
        # spaces would turn the value back into three numbers.
        font = fitz.Font("cjk")
        lines = pdfio._wrap(font, "65, 334, 085.99", 25.0, 11.0)
        self.assertEqual(["65,334,085.99"], lines)

    def test_space_thousand_separator_not_merged(self):
        # ``10 000`` uses a space as the group separator: it must survive as-is.
        font = fitz.Font("cjk")
        lines = pdfio._wrap(font, "10 000", 585.0, 11.0)
        self.assertEqual(["10 000"], lines)

    def test_break_latin_word_minimum_two_char_pieces(self):
        # The org chart shards (``P- ar- ty a- n- d ...``) came from one-char
        # pieces; a piece must carry at least two characters or the rest is
        # kept whole (a dangling ``a-`` would suggest the word continues).
        font = fitz.Font("cjk")
        lines: list[str] = []
        rest = pdfio._break_latin_word(font, "Innovative", 12.0, 11.0, lines)
        pieces = lines + [rest]
        self.assertEqual("".join(p.rstrip("-") for p in pieces), "Innovative")
        for piece in pieces:
            self.assertGreaterEqual(len(piece.rstrip("-")), 2, piece)


class VerticalLabelTest(unittest.TestCase):
    """Narrow-tall boxes (org-chart labels) get a rotated 90° translation."""

    def _block(self, x0=50.0, y0=50.0, x1=57.4, y1=79.5, **kw):
        # 7.4 x 29.5pt: the shape of the real 党群工作部 box on page 5.
        return pdfio.Block(
            text="", page=0, x0=x0, y0=y0, x1=x1, y1=y1,
            size=24.0, align="center", bold=False, **kw
        )

    def test_detection_only_for_narrow_tall_single_line(self):
        block = self._block()
        self.assertTrue(pdfio._is_vertical_label(block))
        # Wide box, squat box, table cell and multi-line: not vertical labels.
        self.assertFalse(pdfio._is_vertical_label(self._block(x1=150.0)))
        self.assertFalse(pdfio._is_vertical_label(self._block(y1=60.0)))
        self.assertFalse(pdfio._is_vertical_label(self._block(in_table=True)))
        self.assertFalse(
            pdfio._is_vertical_label(self._block(single_line=False))
        )

    def test_extraction_marks_rotated_labels_single_line_and_chart(self):
        # Regression: the old ``single_line`` heuristic used ``rect.height <= 1.5*size``,
        # which is always False for a vertically-rotated label (its bbox is *tall*), so
        # ``_is_vertical_label`` never fired on real extraction — the page was classified
        # ``normal`` and labels were drawn as horizontal one-char shards.  This drives the
        # real extraction path (not a hand-built ``Block``) and asserts the labels come
        # back single-line + the page triages as ``chart``.
        src = _OUT / "orgchart.pdf"
        doc = fitz.open()
        page = doc.new_page()
        for y in (100, 200, 300):
            page.insert_text(fitz.Point(60, y), "公司的架构", fontsize=8,
                             fontname="china-s", rotate=90)
        doc.save(str(src))
        doc.close()
        extracted = pdfio.extract_document_text(str(src))
        labels = [b for b in extracted.pages[0] if pdfio._is_vertical_label(b)]
        self.assertGreaterEqual(len(labels), 3, "rotated org-chart labels not detected")
        for b in labels:
            self.assertTrue(b.single_line, f"vertical label not single_line: {b.text!r}")
        self.assertEqual(pdfio.classify_page(extracted.pages[0]), pdfio.PAGE_CHART)

    def test_rotation_keeps_label_inside_the_box(self):
        font = fitz.Font("cjk")
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        # An 8 x 120pt vertical label box (multi-char column): the phrase fits
        # along the height when rotated, which the horizontal draw cannot do.
        block = pdfio.Block(
            text="", page=0, x0=50.0, y0=50.0, x1=58.0, y1=170.0,
            size=24.0, align="center", bold=False,
        )
        pdfio._draw_translated_block(page, font, block, "Party and Mass Work")
        # Extract while the document is still open.
        spans = [s for s in page.get_text("dict")["blocks"] if s.get("type") == 0]
        text = "".join(sp["text"] for s in spans for l in s["lines"] for sp in l["spans"])
        doc.close()
        # The line must be whole (no hyphenated shards) and the glyph column
        # centred inside the box (wider boxes than the source keep the word).
        self.assertNotIn("-", text)
        self.assertTrue("PartyandMassWork" in text.replace(" ", ""))
        bbox = fitz.Rect(0, 0, 0, 0)
        for s in spans:
            for l in s["lines"]:
                bbox |= fitz.Rect(l["bbox"])
        self.assertAlmostEqual((bbox.x0 + bbox.x1) / 2, (block.x0 + block.x1) / 2, delta=2.0)
        self.assertGreater(bbox.y0, block.y0 - 2.0)
        self.assertLess(bbox.y1, block.y1 + 2.0)

    def test_label_longer_than_box_height_is_still_whole(self):
        # A label that cannot fit the box at any readable size is drawn rotated
        # at the column size (extending beyond the box, centred on it) — the
        # alternative, horizontal one-char shards, never reads at all.
        font = fitz.Font("cjk")
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        block = self._block()  # 7.4 x 29.5pt
        long_label = "The Party Committee and Administrative Department"
        pdfio._draw_vertical_label(page, font, block, long_label)
        # Extract while the document is still open.
        spans = [s for s in page.get_text("dict")["blocks"] if s.get("type") == 0]
        text = "".join(sp["text"] for s in spans for l in s["lines"] for sp in l["spans"])
        doc.close()
        self.assertNotIn("-", text)
        self.assertTrue(
            long_label.replace(" ", "") in text.replace(" ", ""), text
        )


class OcrTableNoExpansionTest(unittest.TestCase):
    """OCR-reconstructed tables keep the scan's own geometry.

    Regression: the row-height expansion measured translated cells against the
    reconstructed rows and pushed the lower rows hundreds of points down (on
    the report's statement pages a value moved ~355 pt, several rows below its
    own cell).  Scan pages must draw at the positions the OCR grid produced.
    """

    def _scan_source(self, path: Path) -> Path:
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        # A fake scan: a raster band and no text layer, so find_tables finds no
        # ruled grid and the OCR-table reconstruction path is taken.
        page.draw_rect(fitz.Rect(40, 40, 555, 300), color=None, fill=(0.9, 0.9, 0.9))
        doc.save(str(path))
        doc.close()
        return path

    def _cells(self):
        def cell(x0, y0, x1, y1, text):
            return pdfio.Block(
                text=text, page=0, x0=x0, y0=y0, x1=x1, y1=y1,
                size=10.0, align="left", bold=False,
                single_line=True, ocr=True, in_table=True,
            )

        return cell

    def test_ocr_grid_in_place_keeps_rows_below_the_wrapping_cell(self):
        """A scanned table page stays in-place and the rows are not expanded.

        Six cells where the translation of one cell wraps to several lines
        while the scan row is only 12pt tall: the exporter must keep the scan
        geometry and NOT push the rows below the wrapping cell down — that is
        the ~355pt misplacement regression.
        """
        src = _OUT / "scan_grid_in_place.pdf"
        self._scan_source(src)
        cell = self._cells()

        def row(y0, left, right):
            return cell(50, y0, 260, y0 + 12, left), cell(300, y0, 520, y0 + 12, right)

        b0, b1 = row(100, "总资产", "总负债")
        b2, b3 = row(140, "净资产", "现金")
        b4, b5 = row(180, "净利润", "总成本")
        blocks = [b0, b1, b2, b3, b4, b5]
        trans = [
            "292,712,933,925.17",
            "The consolidated financial statements of the Group and its "
            "subsidiaries were prepared under the principles of going concern",
            "24,321,445,868.48",
            "268,719,676,841.77",
            "12,345,678,901.23",
            "4,567,892,100.00",
        ]
        out = _OUT / "ocr_grid_in_place_out.pdf"
        pdfio.save_translated_pdf(src, [blocks], [trans], out, "English")
        doc = fitz.open(out)
        self.assertEqual(doc.page_count, 1)
        spans = _text_lines(doc[0])
        doc.close()

        def find(text: str) -> fitz.Rect:
            matches = [bb for bb, t in spans if text in t]
            self.assertGreater(len(matches), 0, text)
            return matches[0]

        for b, t in zip(blocks, trans):
            rect = find(t[:20])  # the first words of the cell
            self.assertLess(abs(rect.y0 - b.y0), 8.0, t)
            self.assertLess(abs(rect.x0 - b.x0), 5.0, t)


class WrapTest(unittest.TestCase):
    """``_wrap`` must wrap space-less CJK text (and keep Latin words intact)."""

    def _font(self):
        return fitz.Font("cjk")

    def test_cjk_long_paragraph_wraps_into_multiple_lines(self):
        font = self._font()
        text = (
            "本飞机专为单飞行员操作而开发，并经过相应调整以更好地模拟真实环境。"
            "其制作综合了多个真实世界的数据点和来自不同时期、不同来源的手册，"
            "并通过对各类组件的修改来让它们更容易在微软飞行模拟器中管理。"
        )
        lines = pdfio._wrap(font, text, 585.1, 11.0)
        # Fix regression: a single space-less CJK "word" must be broken, not kept
        # as one overflowing line.
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(
                font.text_length(line, fontsize=11.0), 585.1,
                f"line too wide: {line!r}",
            )
        # Wrapping must not lose or duplicate characters.
        self.assertEqual("".join(lines), text)

    def test_latin_words_stay_intact(self):
        font = self._font()
        text = "This aircraft has been developed for single pilot operations and has been adapted."
        lines = pdfio._wrap(font, text, 300.0, 11.0)
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(font.text_length(line, fontsize=11.0), 300.0)
        # The content must be preserved exactly (words joined by single spaces).
        self.assertEqual(" ".join(lines).replace("  ", " "), text)

    def test_short_cjk_fits_on_one_line(self):
        font = self._font()
        lines = pdfio._wrap(font, "仅供模拟使用", 585.1, 11.0)
        self.assertEqual(len(lines), 1)


class MultilineCellAnchorTest(unittest.TestCase):
    """A table cell whose translation wraps to >1 line is anchored at the cell's
    top border (using the row's full height), so it stays inside the cell."""

    def _grid(self) -> Path:
        path = _OUT / "cellgrid.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        cols = [60, 200, 360]
        tops = [80, 120, 160]
        for x in cols:
            page.draw_line(fitz.Point(x, tops[0]), fitz.Point(x, tops[-1]),
                           color=(0, 0, 0), width=0.6)
        for y in tops:
            page.draw_line(fitz.Point(cols[0], y), fitz.Point(cols[-1], y),
                           color=(0, 0, 0), width=0.6)
        page.insert_text((70, 112), "Net Assets Per Share Attributable", fontsize=9)
        page.insert_text((210, 112), "0.21", fontsize=9)
        doc.save(str(path))
        doc.close()
        return path

    def test_multiline_cell_stays_inside_cell(self):
        src = self._grid()
        dt = pdfio.extract_document_text(src, log=lambda _m: None)
        label = next(b for b in dt.pages[0] if "Net Assets" in b.text)
        self.assertTrue(label.in_table)
        cell = pdfio._extract_tables(fitz.open(str(src))[0])[0]["rows"][0][0]
        cell_top, cell_bot = cell.y0, cell.y1
        per = [
            ("Net Assets Per Share Attributable to Shareholders of the Parent Company"
             if b is label else b.text)
            for b in dt.pages[0]
        ]
        out = _OUT / "cellgrid_out.pdf"
        pdfio.save_translated_pdf(src, [dt.pages[0]], [per], str(out), "English")
        doc = fitz.open(str(out))
        spans = []
        for bb in doc[0].get_text("dict")["blocks"]:
            if bb.get("type") != 0:
                continue
            for l in bb.get("lines", []):
                for s in l.get("spans", []):
                    b = s["bbox"]
                    if s["text"].strip() and b[0] >= cell.x0 - 4 and b[2] <= cell.x1 + 4 \
                            and b[1] >= cell_top - 6 and b[3] <= cell_bot + 6:
                        spans.append(s["bbox"])
        doc.close()
        self.assertGreater(len(spans), 1)  # the label wrapped to >1 line
        # It must sit on the cell's top border (not the source glyph top / centred)
        # and stay inside the cell rather than overflowing the bottom rule.
        self.assertLess(min(s[1] for s in spans) - cell_top, 8.0)
        self.assertLessEqual(max(s[3] for s in spans), cell_bot + 1.0)


class SkewDetectTest(unittest.TestCase):
    """The low-risk OCR skew detector's pure angle math."""

    def test_angle_median_wraps_to_45(self):
        # Angles fold into [-45, 45): a 0° and a 179° line are the same baseline.
        self.assertEqual(0.0, pdfio._angle_median([0.0, 0.5, -0.5]))
        self.assertAlmostEqual(10.0, pdfio._angle_median([9.0, 10.0, 11.0]))
        self.assertAlmostEqual(-1.0, pdfio._angle_median([179.0, 181.0, -1.0]), places=5)
        self.assertIsNone(pdfio._angle_median([]))

    def test_deskew_affine_roundtrips_points(self):
        # A point in the original image, rotated forward to the deskewed frame, then
        # mapped back with ``_map_pt_back`` must land exactly on the original point —
        # this is the geometry that keeps OCR boxes on the real page after a deskew.
        import cv2
        import numpy as np

        img = np.zeros((300, 400, 3), np.uint8)     # H=300, W=400
        for skew in (1.5, -2.0):
            rotated, inv_m, pad = pdfio._deskew_affine(img, skew)
            m = cv2.invertAffineTransform(inv_m)
            for (x, y) in ((5, 5), (37, 123), (395, 295)):
                v = m @ np.array([float(x + pad), float(y + pad), 1.0])
                ox, oy = pdfio._map_pt_back(inv_m, pad, float(v[0]), float(v[1]))
                self.assertAlmostEqual(float(x), ox, places=4)
                self.assertAlmostEqual(float(y), oy, places=4)

    def test_estimate_skew_from_gray_detects_angle(self):
        import cv2
        import numpy as np

        for want in (1.5, -2.0):
            img = np.full((400, 500), 255, np.uint8)
            ang = np.radians(want)
            cx, cy = 250.0, 200.0
            dx, dy = 200.0, 0.0
            x1 = cx + dx * np.cos(ang) - dy * np.sin(ang)
            y1 = cy + dx * np.sin(ang) + dy * np.cos(ang)
            x2 = cx - dx * np.cos(ang) + dy * np.sin(ang)
            y2 = cy - dx * np.sin(ang) - dy * np.cos(ang)
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), 0, 3)
            got = pdfio._estimate_skew_from_gray(img)
            self.assertIsNotNone(got)
            self.assertAlmostEqual(float(want), float(got), delta=1.2)

    def test_detect_page_skew_flat_page_is_not_recommended(self):
        # A page with straight horizontal rules reads as ~0° (no geometry correction
        # is recommended).  --- a smoke test that the CV path runs without error.
        import pymupdf as fitz
        src = _OUT / "skew_flat.pdf"
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        for y in (40, 80, 120, 160, 200, 240):
            page.draw_line(fitz.Point(30, y), fitz.Point(270, y),
                           color=(0, 0, 0), width=1)
        doc.save(str(src))
        doc.close()
        res = pdfio.detect_page_skew(str(src), 0)
        self.assertIsInstance(res["skew_degrees"], float)
        self.assertIn(res["reason"], ("版面基本平正", "未检测到文本线"))


class StructureTest(unittest.TestCase):
    """B-④ semantic-structure layer: build_structure fuser + get_doc_info enrichment."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = build_sample_pdf(Path(self.tmp.name) / "struct_src.pdf", pages=2)

    def _doc(self):
        return pdfio.extract_document_text(str(self.src), ocr=False, log=lambda m: None)

    def _structure_fn(self, page_index, page, blocks):
        # A mock backend: page 0's first block is a formula; page 1's first block is a figure.
        if not blocks:
            return []
        b = blocks[0]
        if page_index == 0:
            return [{"kind": "formula", "bbox": [b.x0, b.y0, b.x1, b.y1]}]
        if page_index == 1:
            return [{"kind": "figure", "bbox": [b.x0, b.y0, b.x1, b.y1],
                     "level": 0}]
        return []

    def test_build_structure_populates_page_structure(self):
        dt = self._doc()
        pdfio.build_structure(str(self.src), dt, self._structure_fn, parser="mock")
        self.assertEqual(dt.structure_parser, "mock")
        self.assertEqual(len(dt.page_structure), dt.page_count)
        self.assertEqual(dt.page_structure[0].elements[0]["kind"], "formula")
        # The first block of page 0 (flat index 0) is contained in its own bbox.
        self.assertIn(0, dt.page_structure[0].elements[0]["block_indices"])

    def test_get_doc_info_includes_structure_counts_when_present(self):
        dt = self._doc()
        pdfio.build_structure(str(self.src), dt, self._structure_fn, parser="mock")
        info = pdfio.get_doc_info(dt)
        self.assertEqual(info["structure_parser"], "mock")
        self.assertEqual(info["formula_pages"], 1)
        self.assertEqual(info["figure_pages"], 1)

    def test_get_doc_info_unchanged_without_structure(self):
        dt = self._doc()
        info = pdfio.get_doc_info(dt)
        # No formula/figure pages when no structure backend ran.
        self.assertEqual(info["formula_pages"], 0)
        self.assertEqual(info["figure_pages"], 0)
        self.assertNotIn("structure_parser", info)
        # Existing keys are unaffected.
        self.assertEqual(info["pages"], 2)

    def test_structure_dominant_kind(self):
        ps = pdfio.PageStructure(page=0, parser="mock",
                                 elements=[{"kind": "formula", "bbox": [0, 0, 1, 1],
                                            "level": 0, "block_indices": [0], "parser": "mock"},
                                           {"kind": "formula", "bbox": [0, 0, 1, 1],
                                            "level": 0, "block_indices": [1], "parser": "mock"}])
        self.assertEqual(pdfio._structure_dominant_kind(ps), "formula")
        # A single formula among text is not dominant.
        ps2 = pdfio.PageStructure(page=1, parser="mock",
                                  elements=[{"kind": "formula", "bbox": [0, 0, 1, 1],
                                             "level": 0, "block_indices": [0], "parser": "mock"},
                                            {"kind": "text", "bbox": [0, 0, 1, 1],
                                             "level": 0, "block_indices": [1], "parser": "mock"}])
        self.assertIsNone(pdfio._structure_dominant_kind(ps2))

    def test_classify_page_formula_and_figure_kinds(self):
        dt = self._doc()
        pdfio.build_structure(str(self.src), dt, self._structure_fn, parser="mock")
        p0 = dt.page_structure[0]
        # Page 0 has a structure with formula (more regions than text blocks it
        # claims), so it is classified as a formula page when structure is supplied.
        self.assertIn(pdfio.classify_page(dt.pages[0], structure=p0),
                      (pdfio.PAGE_FORMULA, pdfio.PAGE_NORMAL))
        # Without a structure, classification is unchanged (no formula/figure).
        self.assertEqual(pdfio.classify_page(dt.pages[0]), pdfio.PAGE_NORMAL)

    def test_extract_structured_populates_and_degrades(self):
        # A backend that works populates page_structure.
        dt = pdfio.extract_structured(str(self.src), self._structure_fn, parser="mock")
        self.assertEqual(dt.structure_parser, "mock")
        self.assertTrue(dt.page_structure)

        # A backend that raises degrades to a plain extraction (no crash).
        def _boom(page_index, page, blocks):
            raise RuntimeError("backend down")
        dt2 = pdfio.extract_structured(str(self.src), _boom, parser="boom",
                                       log=lambda m: None)
        self.assertEqual(dt2.structure_parser, "")
        # A backend that raises on every page leaves element-less structure entries.
        self.assertTrue(all(not ps.elements for ps in dt2.page_structure))

    def test_table_region_builds_struct_table(self):
        def table_fn(page_index, page, blocks):
            if page_index != 0 or not blocks:
                return []
            # A table region over the whole page; cells are flat block indices.
            bbox = [blocks[0].x0, blocks[0].y0, blocks[-1].x1, blocks[-1].y1]
            return [{"kind": "table", "bbox": bbox,
                     "cells": [[0, 1], [None, 2]]}]
        dt = pdfio.extract_structured(str(self.src), table_fn, parser="mock")
        self.assertEqual(dt.structure_parser, "mock")
        ps = dt.page_structure[0]
        self.assertEqual(len(ps.tables), 1)
        tbl = ps.tables[0]
        self.assertEqual((tbl.rows, tbl.cols), (2, 2))
        self.assertEqual(tbl.cells[0], [0, 1])
        self.assertEqual(tbl.cells[1], [-1, 2])
        self.assertTrue({0, 1, 2} <= tbl.block_ref)
        gt = pdfio.get_table(dt, 0, 0)
        self.assertIsNotNone(gt)
        self.assertEqual(gt["rows"], 2)
        # Page 1 has no table → None.
        self.assertIsNone(pdfio.get_table(dt, 1, 0))


class FormulaProtectionTest(unittest.TestCase):
    """B-⑤: a detected math-expression block is never translated (kept verbatim)."""

    def test_formula_detector(self):
        self.assertTrue(pdfio._is_formula_block("x^2 + y^2 = z^2"))
        self.assertTrue(pdfio._is_formula_block("∫_0^∞ e^{-x^2} dx = √π"))
        self.assertFalse(pdfio._is_formula_block("This is an ordinary English sentence."))
        self.assertFalse(pdfio._is_formula_block("3,702,726,474.45"))
        self.assertFalse(pdfio._is_formula_block("营业收入 合计"))

    def test_formula_detector_recognises_display_equation_fragments(self):
        # B-⑤ regression: a LaTeX paper splits a display equation into short line
        # fragments (``P_orig(y|x;T)=``, ``exp(z_i/T)``, ``argsort(z)≡…``) that the
        # old ``≥3 math chars`` test missed (they carry only 1–2 operators).
        self.assertTrue(pdfio._is_formula_block("Porig(yi|x; T) ="))
        self.assertTrue(pdfio._is_formula_block("exp(zi/T)"))
        self.assertTrue(pdfio._is_formula_block("argsort(z) ≡ argsort(z/T)"))
        # Prose / data that merely contains a relation is NOT a formula.
        self.assertFalse(pdfio._is_formula_block("T = 1.0, TTR = 0.400"))

    def test_formula_not_reported_missing(self):
        from translate_app.eval import measure_complete
        from translate_app.pdfio import Block
        res = measure_complete([Block("x^2 + y^2 = z^2", 0, 0, 0, 100, 20)], [""])
        self.assertEqual(res["missing_count"], 0)


class ColumnGapBreakTest(unittest.TestCase):
    """A 2-column page must not merge left and right lines into one full-width block."""

    @staticmethod
    def _ln(x0, x1, y0, y1, text):
        return {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "text": text,
                "size": 10.0, "bold": False, "color": 0}

    def test_cross_column_lines_break_even_when_y_close(self):
        # A right-column line followed by a left-column line at a *close* y (PyMuPDF's
        # line stream interleaves columns by y) must start a new block — otherwise they
        # merge into a full-width block whose translation is drawn across the page.
        base = self._ln(55, 289, 69, 79, "left one")
        prev = self._ln(307, 541, 123, 133, "right col line")
        cur = self._ln(55, 289, 128, 138, "left col next")
        self.assertTrue(pdfio._break_between(base, prev, cur))

    def test_same_column_lines_join(self):
        base = self._ln(55, 289, 80, 90, "left one")
        prev = self._ln(55, 289, 92, 102, "left two")
        cur = self._ln(55, 289, 104, 114, "left three")
        self.assertFalse(pdfio._break_between(base, prev, cur))

    def test_no_full_width_blocks_on_two_column_page(self):
        # Use the two-column fixture end-to-end: extraction must not yield a block
        # spanning both columns.
        with tempfile.TemporaryDirectory() as d:
            pdf = build_two_column_pdf(Path(d) / "two.pdf")
            dt = pdfio.extract_document_text(pdf, ocr=False, log=lambda m: None)
            # page 0 is a clean 2-column page (left ~60..215, right ~315..480).
            blocks = dt.pages[0]
            page_w = 595.0
            for b in blocks:
                self.assertLess((b.x1 - b.x0), page_w * 0.6,
                                f"block {b.text[:30]!r} spans too wide: {b.x0:.0f}..{b.x1:.0f}")


class GeometricStructureTest(unittest.TestCase):
    """The deterministic geometric structure backend (B-④/B-⑤ 'actually runs' offline)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _blocks(self):
        # formula (2 math symbols), bold heading, one table with a label + an amount.
        return [
            pdfio.Block("x^2 + y^2 = z^2", page=0, x0=0, y0=0, x1=100, y1=20),
            pdfio.Block("资产负债表", page=0, x0=0, y0=20, x1=100, y1=36,
                        bold=True, single_line=True),
            pdfio.Block("资产", page=0, x0=0, y0=40, x1=50, y1=60, in_table=True),
            pdfio.Block("1,234.56", page=0, x0=50, y0=40, x1=100, y1=60, in_table=True),
        ]

    def _rich_doc(self):
        blocks = self._blocks()
        return pdfio.DocumentText(pages=[blocks], blocks=[b.text for b in blocks],
                                  block_pages=[0, 0, 0, 0])

    def test_geometric_structure_fn_detects_kinds(self):
        sf = pdfio.make_geometric_structure_fn()
        regions = sf(0, None, self._blocks())
        kinds = {r["kind"] for r in regions}
        self.assertIn("formula", kinds)
        self.assertIn("heading", kinds)
        self.assertIn("table", kinds)
        tbl = next(r for r in regions if r["kind"] == "table")
        self.assertTrue(tbl["cells"])   # a row/col grid was built

    def test_build_structure_and_get_table_run_geometric(self):
        dt = self._rich_doc()
        src = build_sample_pdf(Path(self.tmp.name) / "geo.pdf", pages=1)
        pdfio.build_structure(str(src), dt, pdfio.make_geometric_structure_fn(), parser="geo")
        self.assertEqual(dt.structure_parser, "geo")
        self.assertTrue(dt.page_structure and dt.page_structure[0].elements)
        # The semantic table is now readable through get_table.
        gt = pdfio.get_table(dt, 0, 0)
        self.assertIsNotNone(gt)
        self.assertEqual(gt["rows"], 1)
        self.assertEqual(len(gt["cells"][0]), 2)
        # get_doc_info reports the structure parser once present.
        info = pdfio.get_doc_info(dt)
        self.assertEqual(info["structure_parser"], "geo")

    def test_extract_document_structured_runs_on_real_pdf(self):
        # A real PDF with a formula-ish line: the one-call entry finds it.
        import pymupdf as fitz
        src = Path(self.tmp.name) / "formula.pdf"
        d = fitz.open()
        p = d.new_page(width=300, height=200)
        p.insert_text((72, 80), "x^2 + y^2 = z^2", fontsize=12)
        d.save(str(src))
        d.close()
        dt = pdfio.extract_document_structured(str(src), parser="geo")
        self.assertEqual(dt.structure_parser, "geo")
        kinds = {e["kind"] for ps in dt.page_structure for e in ps.elements}
        self.assertIn("formula", kinds)

    def test_doclayout_structure_fn_degrades_to_geometric(self):
        # DocLayout-YOLO isn't installed here → the factory falls back to the
        # geometric backend, so it still produces real structure (never empty/crash).
        sf = pdfio.make_doclayout_structure_fn(log=lambda m: None)
        regions = sf(0, None, self._blocks())
        kinds = {r["kind"] for r in regions}
        self.assertIn("formula", kinds)
        self.assertIn("table", kinds)

    def test_make_vlm_ocr_fn_degrades_and_register(self):
        # No VLM backend registered → None (built-in RapidOCR used).
        self.assertIsNone(pdfio.make_vlm_ocr_fn())
        # Registering one makes the injectable ocr_fn available.
        pdfio.register_ocr_backend("vlm", lambda: lambda page_index, page: [])
        try:
            fn = pdfio.make_vlm_ocr_fn()
            self.assertTrue(callable(fn))
            self.assertEqual(fn(0, None), [])   # a wrapped backend callable
        finally:
            pdfio._OCR_BACKENDS.pop("vlm", None)

    def test_caption_regex_requires_a_numeral(self):
        # A bare "图" / "Fig" is a prose lead-in, not a caption; a labelled numeral is.
        self.assertRegex("图3 收入构成", pdfio._CAPTION_RE)
        self.assertRegex("Fig. 1: revenue trend", pdfio._CAPTION_RE)
        self.assertRegex("Table 2 主要指标", pdfio._CAPTION_RE)
        self.assertIsNone(pdfio._CAPTION_RE.match("图"))
        self.assertIsNone(pdfio._CAPTION_RE.match("Fig"))

    def test_get_table_page1_uses_flat_offset(self):
        # A table on page 1 must produce *flat* cells (offset added): page-local
        # index 0/1 become flat 2/3 when page 0 has 2 blocks.
        def structure_fn(page_index, page, blocks):
            if page_index != 1 or not blocks:
                return []
            bbox = [blocks[0].x0, blocks[0].y0, blocks[-1].x1, blocks[-1].y1]
            return [{"kind": "table", "bbox": bbox, "cells": [[0, 1]]}]
        p0 = [pdfio.Block("a", page=0, x0=0, y0=0, x1=10, y1=10),
              pdfio.Block("b", page=0, x0=0, y0=10, x1=10, y1=20)]
        p1 = [pdfio.Block("c", page=1, x0=0, y0=0, x1=50, y1=20, in_table=True),
              pdfio.Block("d", page=1, x0=50, y0=0, x1=100, y1=20, in_table=True)]
        dt = pdfio.DocumentText(pages=[p0, p1], blocks=["a", "b", "c", "d"],
                                block_pages=[0, 0, 1, 1])
        src = build_sample_pdf(Path(self.tmp.name) / "off.pdf", pages=2)
        pdfio.build_structure(str(src), dt, structure_fn, parser="mock")
        gt = pdfio.get_table(dt, 1, 0)
        self.assertIsNotNone(gt)
        self.assertEqual(gt["cells"][0], [2, 3])   # page-local 0,1 + offset 2


if __name__ == "__main__":
    unittest.main()
