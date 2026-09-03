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

    def test_name_column_keeps_original(self):
        # A "姓名 / Name"-headed table column must be flagged keep-original so the
        # names are not transliterated; the header and other columns are not.
        blocks = [
            pdfio.Block(text="姓名", page=0, x0=60, y0=100, x1=100, y1=115, size=9),
            pdfio.Block(text="汪建法", page=0, x0=60, y0=120, x1=100, y1=135, size=9),
            pdfio.Block(text="钱水土", page=0, x0=60, y0=140, x1=100, y1=155, size=9),
            pdfio.Block(text="职务", page=0, x0=160, y0=100, x1=200, y1=115, size=9),
            pdfio.Block(text="董事长", page=0, x0=160, y0=120, x1=200, y1=135, size=9),
        ]
        pdfio._mark_name_column(blocks)
        flagged = {b.text for b in blocks if b.keep_original}
        self.assertEqual(flagged, {"汪建法", "钱水土"})
        # The header cell and the other (non-name) column are left untouched.
        self.assertIn("姓名", {b.text for b in blocks if not b.keep_original})
        self.assertIn("董事长", {b.text for b in blocks if not b.keep_original})

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
        row0 = next(bi for bi, (_ti, r) in mapping.items() if r == 0)
        row1 = next(bi for bi, (_ti, r) in mapping.items() if r == 1)
        row2 = next(bi for bi, (_ti, r) in mapping.items() if r == 2)
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

    def test_is_org_chart_page_requires_a_cluster(self):
        # A page is an org chart / architecture diagram only when it holds a
        # *cluster* of node boxes with at least one narrow-tall anchor: a lone
        # narrow box (or a set of undifferentiated squares) is not a diagram.
        self.assertFalse(
            pdfio._is_org_chart_page([self._block()])  # one box
        )
        self.assertFalse(
            pdfio._is_org_chart_page([self._block(), self._block(y0=100, y1=129.5)])
            # two boxes
        )
        cluster = [
            self._block(y0=50, y1=79.5),
            self._block(y0=100, y1=129.5),
            self._block(y0=150, y1=179.5),
        ]
        self.assertTrue(pdfio._is_org_chart_page(cluster))
        # Two vertical boxes plus a wide box / prose block is not a cluster
        # (only two nodes total, and the wide box is not a node).
        mixed = [
            self._block(y0=50, y1=79.5), self._block(y0=100, y1=129.5),
            self._block(x1=300.0, y0=150, y1=179.5),  # a wide (non-node) block
            pdfio.Block(text="正文文字", page=0, x0=40, y0=190, x1=300, y1=205,
                        size=11.0, single_line=True),
        ]
        self.assertFalse(pdfio._is_org_chart_page(mixed))
        # Squares join the cluster only with a narrow-tall anchor: three plain
        # squares alone are not a diagram (they could be headings/list items).
        squares = [
            self._square(x0=200, y0=50), self._square(x0=260, y0=50),
            self._square(x0=320, y0=50),
        ]
        self.assertFalse(pdfio._is_org_chart_page(squares))

    def _square(self, x0=200.0, y0=50.0, **kw):
        """A compact roughly-square node box: a short single-line label."""
        return pdfio.Block(
            text="系统架构", page=0, x0=x0, y0=y0, x1=x0 + 55.0, y1=y0 + 18.0,
            size=12.0, align="center", single_line=True, **kw
        )

    def test_is_chart_node_recognizes_compact_square_boxes(self):
        # A node is either a narrow-tall box or a compact square box; a
        # full-width text line, long prose, or a table cell are not nodes.
        self.assertTrue(pdfio._is_chart_node(self._block()))   # narrow-tall
        self.assertTrue(pdfio._is_chart_node(self._square()))  # compact square
        # Full-width single line (a heading): not a node.
        wide = pdfio.Block(text="合并资产负债表", page=0, x0=40, y0=50, x1=400,
                           y1=68, size=12.0, single_line=True)
        self.assertFalse(pdfio._is_chart_node(wide))
        # Long single-line label: prose, not a node.
        longlbl = pdfio.Block(text="自然资源管理信息系统与数据交换平台", page=0,
                              x0=200, y0=50, x1=255, y1=68, size=12.0,
                              single_line=True)
        self.assertFalse(pdfio._is_chart_node(longlbl))
        # A square box that is a table cell (a ruled column value): not a node.
        cell = pdfio.Block(text="系统架构", page=0, x0=200, y0=50, x1=255, y1=68,
                           size=12.0, single_line=True, in_table=True)
        self.assertFalse(pdfio._is_chart_node(cell))

    def test_is_org_chart_page_accepts_mixed_narrow_and_square(self):
        # A narrow-tall anchor plus compact square nodes together form a chart.
        anchor = [self._block(y0=50, y1=79.5)]
        squares = [self._square(x0=200, y0=50), self._square(x0=260, y0=50)]
        self.assertTrue(pdfio._is_org_chart_page(anchor + squares))
        self.assertTrue(pdfio._is_org_chart_page(
            [self._block(y0=50, y1=79.5)] * pdfio._CHART_NODE_MIN
        ))

    def test_flag_chart_nodes_marks_only_node_boxes(self):
        # On a diagram page the node labels (narrow-tall *and* square) are
        # flagged; headings / prose on the same page still translate.
        intro = pdfio.Block(text="组织机构图", page=0, x0=40, y0=20, x1=300, y1=35,
                            size=14.0, single_line=True)
        nodes = [
            self._block(y0=50, y1=79.5),
            self._block(y0=100, y1=129.5),
            self._square(x0=200, y0=50),
        ]
        page_blocks = [intro] + nodes
        pdfio._flag_chart_nodes(page_blocks)
        self.assertTrue(all(b.is_chart for b in nodes))
        self.assertFalse(intro.is_chart)
        # A non-diagram page leaves every block untouched.
        plain = [self._block(), intro]  # only one node — not a cluster
        pdfio._flag_chart_nodes(plain)
        self.assertTrue(all(not b.is_chart for b in plain))

    def test_compact_section_heading_is_not_a_chart_node(self):
        # A compact ORDINAL-LEADING heading on a diagram page (二、公司组织架构图, a
        # wide-flat box that matches the square-node *shape*) is a heading, not a
        # diagram node: it must still be translated, not kept as a chart label.
        anchor = [self._block(y0=50, y1=79.5)]
        squares = [self._square(x0=200, y0=50), self._square(x0=260, y0=50)]
        heading = pdfio.Block(
            text="二、公司组织架构图", page=0, x0=90, y0=78, x1=201.5, y1=91.4,
            size=12.0, align="left", single_line=True,
        )
        blocks = [heading] + anchor + squares
        self.assertTrue(pdfio._is_org_chart_page(blocks))
        self.assertFalse(pdfio._is_chart_node(heading))
        pdfio._flag_chart_nodes(blocks)
        self.assertFalse(heading.is_chart)
        self.assertTrue(all(b.is_chart for b in anchor + squares))


class ChartPageDetectionIntegrationTest(unittest.TestCase):
    """``extract_document_text`` flags org-chart / architecture node labels.

    End to end: a page whose text layer is a cluster of narrow-tall stacked
    labels (a Chinese org chart) comes back with every node box marked
    ``is_chart``, while a normal page is untouched.
    """

    def _chart_pdf(self) -> Path:
        doc = fitz.open()
        page = doc.new_page(width=400, height=600)
        font = fitz.Font("cjk")
        # Three narrow-tall node boxes, each built by stacking its label into a
        # narrow column — the shape of a vertical org-chart label.
        for x, word in ((60, "党群工作部"), (110, "综合管理部"), (160, "财务部")):
            tw = fitz.TextWriter(page.rect)
            for i, ch in enumerate(word):
                tw.append(fitz.Point(x, 60 + i * 20), ch, font=font, fontsize=22)
            tw.write_text(page)
        path = _OUT / "org_chart.pdf"
        doc.save(str(path))
        doc.close()
        return path

    def test_chart_page_nodes_are_flagged_is_chart(self):
        dt = pdfio.extract_document_text(self._chart_pdf())
        self.assertEqual(1, dt.page_count)
        nodes = dt.pages[0]
        self.assertGreaterEqual(len(nodes), pdfio._CHART_NODE_MIN)
        self.assertTrue(pdfio._is_org_chart_page(nodes))
        # Every node label is kept verbatim; a rotated run that extraction
        # reports with a tall bbox (single_line=False) is still a node.
        self.assertTrue(all(b.is_chart for b in nodes))

    def test_normal_page_has_no_chart_nodes(self):
        src = _OUT / "plain.pdf"
        build_sample_pdf(src, pages=1)
        dt = pdfio.extract_document_text(src)
        for page_blocks in dt.pages:
            for b in page_blocks:
                self.assertFalse(b.is_chart)

    def test_inplace_export_keeps_chart_labels_verbatim(self):
        # The in-place exporter never touches an ``is_chart`` node label: it
        # neither redacts nor redraws it (a fed-in "translation" is ignored),
        # so the original diagram label survives the export exactly as-is.
        src = self._chart_pdf()
        dt = pdfio.extract_document_text(src)
        per_page = [["TRANSLATED ONE"] * len(dt.pages[0])]
        out = _OUT / "org_chart_out.pdf"
        pdfio.save_translated_pdf(src, dt.pages, per_page, out, "English")
        doc = fitz.open(str(out))
        text = "".join(
            s["text"]
            for b in doc[0].get_text("dict")["blocks"] if b.get("type") == 0
            for l in b["lines"] for s in l["spans"]
        )
        doc.close()
        self.assertIn("党", text)
        self.assertIn("群", text)
        self.assertNotIn("TRANSLATED", text)


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


class WholePageReviewTest(unittest.TestCase):
    """Conservative whole-page review: correct unreadable text, log structure."""

    def _page(self):
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        self.addCleanup(doc.close)
        return page

    def _block(self, text, x0, y0, x1, y1):
        return pdfio.Block(
            text=text, page=0, x0=x0, y0=y0, x1=x1, y1=y1,
            size=10.0, align="left", bold=False, single_line=True, ocr=True,
        )

    def test_applies_text_fix_and_logs_structure(self):
        page = self._page()
        blocks = [self._block("现金", 40, 50, 90, 66),
                  self._block("12,34,56", 120, 50, 280, 66)]  # unreadable figure
        z = pdfio._REVIEW_DPI / 72.0

        def review(_i, _o, _r):
            return {
                "text_fixes": [{"bbox": [120 * z, 50 * z, 280 * z, 66 * z],
                                "text": "12,345,600"}],
                "structure_flags": [{"message": "列对齐略有偏差"}],
            }

        logs: list[str] = []
        out = pdfio._apply_page_review(page, blocks, review, 0, logs.append)
        self.assertEqual(out[1].text, "12,345,600")
        self.assertEqual((out[1].x0, out[1].x1), (120.0, 280.0))  # geometry untouched
        self.assertTrue(any("布局提示" in m for m in logs), logs)
        self.assertTrue(any("采纳 1 处文字修正" in m for m in logs), logs)

    def test_does_not_overwrite_a_clean_amount(self):
        page = self._page()
        blocks = [self._block("现金", 40, 50, 90, 66),
                  self._block("17,485,938,749.91", 120, 50, 280, 66)]
        z = pdfio._REVIEW_DPI / 72.0
        logs: list[str] = []

        def review(_i, _o, _r):
            return {"text_fixes": [{"bbox": [120 * z, 50 * z, 280 * z, 66 * z],
                                    "text": "17,485,938,749.9"}]}

        out = pdfio._apply_page_review(page, blocks, review, 0, logs.append)
        # A clean OCR amount is never replaced; the disagreement is surfaced.
        self.assertEqual(out[1].text, "17,485,938,749.91")
        self.assertTrue(any("请人工复核" in m for m in logs), logs)

    def test_review_failure_is_noop(self):
        page = self._page()
        blocks = [self._block("现金", 40, 50, 90, 66)]
        logs: list[str] = []

        def review(_i, _o, _r):
            raise RuntimeError("boom")

        out = pdfio._apply_page_review(page, blocks, review, 0, logs.append)
        self.assertEqual(out[0].text, "现金")
        self.assertTrue(any("整页审查失败" in m for m in logs), logs)

    def test_adopts_corrected_statement_code(self):
        # A statement / subject code OCR misread (会企01表-1 -> 公司01目-1): the
        # reviewer's re-read is adopted — codes are exact identifiers, unlike a
        # clean amount which would never be overwritten.
        b = self._block("公司01目-1", 40, 50, 90, 66)
        self.assertTrue(pdfio._looks_like_code_token(b.text))
        changed = pdfio._apply_review_fix(b, "会企01表-1", 0, None)
        self.assertTrue(changed)
        self.assertEqual(b.text, "会企01表-1")

    def test_does_not_treat_title_with_year_as_code(self):
        # A document title that happens to contain a year is prose, not a code —
        # the reviewer's reading must not be adopted for it.
        b = self._block("2025年年度报告（摘要）", 40, 50, 300, 66)
        self.assertFalse(pdfio._looks_like_code_token(b.text))
        changed = pdfio._apply_review_fix(b, "MTB 2025 Annual Report", 0, None)
        self.assertFalse(changed)
        self.assertEqual(b.text, "2025年年度报告（摘要）")

    def test_does_not_treat_long_prose_as_code(self):
        b = self._block("编制单位：浙江民泰商业银行股份有限公司", 40, 50, 300, 66)
        self.assertFalse(pdfio._looks_like_code_token(b.text))
        changed = pdfio._apply_review_fix(b, "Prepared by: Mintai bank", 0, None)
        self.assertFalse(changed)

    def test_merges_adjacent_split_label(self):
        page = self._page()
        blocks = [self._block("党", 40, 50, 90, 66),
                  self._block("群", 94, 50, 144, 66)]  # adjacent fragments
        z = pdfio._REVIEW_DPI / 72.0
        logs: list[str] = []

        def review(_i, _o, _r):
            return {"structure_flags": [{
                "action": "merge_cells",
                "cells": [[40 * z, 50 * z, 90 * z, 66 * z],
                          [94 * z, 50 * z, 144 * z, 66 * z]],
                "confidence": 0.9, "message": "拆分的标签应合并",
            }]}

        out = pdfio._apply_page_review(page, blocks, review, 0, logs.append)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "党群")                 # CJK joins w/o space
        self.assertEqual((out[0].x0, out[0].x1), (40.0, 144.0))  # bbox = union
        self.assertTrue(any("自动合并" in m for m in logs), logs)

    def test_merge_rejected_when_block_is_figure(self):
        page = self._page()
        blocks = [self._block("现金", 40, 50, 90, 66),
                  self._block("5,000.00", 94, 50, 144, 66)]
        z = pdfio._REVIEW_DPI / 72.0
        logs: list[str] = []

        def review(_i, _o, _r):
            return {"structure_flags": [{
                "action": "merge_cells",
                "cells": [[40 * z, 50 * z, 90 * z, 66 * z],
                          [94 * z, 50 * z, 144 * z, 66 * z]],
                "confidence": 0.95, "message": "标签与数字不应合并",
            }]}

        out = pdfio._apply_page_review(page, blocks, review, 0, logs.append)
        self.assertEqual(len(out), 2)                          # not merged
        self.assertTrue(any("布局提示" in m for m in logs), logs)  # prompted

    def test_merge_rejected_low_confidence(self):
        page = self._page()
        blocks = [self._block("党", 40, 50, 90, 66), self._block("群", 94, 50, 144, 66)]
        z = pdfio._REVIEW_DPI / 72.0
        logs: list[str] = []

        def review(_i, _o, _r):
            return {"structure_flags": [{
                "action": "merge_cells",
                "cells": [[40 * z, 50 * z, 90 * z, 66 * z],
                          [94 * z, 50 * z, 144 * z, 66 * z]],
                "confidence": 0.5,
            }]}

        out = pdfio._apply_page_review(page, blocks, review, 0, logs.append)
        self.assertEqual(len(out), 2)

    def test_merge_rejected_not_adjacent(self):
        page = self._page()
        blocks = [self._block("党", 40, 50, 90, 66), self._block("群", 300, 50, 350, 66)]
        z = pdfio._REVIEW_DPI / 72.0
        logs: list[str] = []

        def review(_i, _o, _r):
            return {"structure_flags": [{
                "action": "merge_cells",
                "cells": [[40 * z, 50 * z, 90 * z, 66 * z],
                          [300 * z, 50 * z, 350 * z, 66 * z]],
                "confidence": 0.9,
            }]}

        out = pdfio._apply_page_review(page, blocks, review, 0, logs.append)
        self.assertEqual(len(out), 2)

    def test_non_action_structure_flag_hints_only(self):
        page = self._page()
        blocks = [self._block("党", 40, 50, 90, 66), self._block("群", 94, 50, 144, 66)]
        z = pdfio._REVIEW_DPI / 72.0
        logs: list[str] = []

        def review(_i, _o, _r):
            return {"structure_flags": [{"message": "列对齐略有偏差"}]}

        out = pdfio._apply_page_review(page, blocks, review, 0, logs.append)
        self.assertEqual(len(out), 2)
        self.assertTrue(any("布局提示" in m for m in logs), logs)


class WholePageReviewIntegrationTest(_OcrCacheIsolated):
    """``extract_document_text`` runs the review on a rebuilt OCR page."""

    def test_extract_applies_review_fix(self):
        src = _OUT / "review_scan.pdf"
        doc = fitz.open()
        page = doc.new_page(width=595, height=300)
        page.draw_rect(fitz.Rect(40, 40, 555, 200), color=None, fill=(0.9, 0.9, 0.9))
        doc.save(str(src))
        doc.close()

        def quad(x0, y0, x1, y1):
            return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

        def ocr_fn(_i, _p):
            return [
                (quad(60, 60, 130, 80), "12,34,56"),
                (quad(60, 100, 130, 120), "5,000.00"),
            ]

        def review(_i, _o, _r):
            z = pdfio._REVIEW_DPI / 72.0
            return {"text_fixes": [{"bbox": [60 * z, 60 * z, 130 * z, 80 * z],
                                    "text": "12,345,600"}]}

        dt = pdfio.extract_document_text(
            src, ocr=True, ocr_fn=ocr_fn, review_fn=review, log=lambda _m: None
        )
        texts = [b.text for b in dt.pages[0]]
        self.assertIn("12,345,600", texts)
        self.assertNotIn("12,34,56", texts)

    def test_extract_merges_split_label(self):
        src = _OUT / "review_merge.pdf"
        doc = fitz.open()
        page = doc.new_page(width=595, height=300)
        page.draw_rect(fitz.Rect(40, 40, 555, 200), color=None, fill=(0.9, 0.9, 0.9))
        doc.save(str(src))
        doc.close()

        def quad(x0, y0, x1, y1):
            return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

        def ocr_fn(_i, _p):
            return [
                (quad(60, 60, 110, 80), "党"),     # split label fragments
                (quad(114, 60, 164, 80), "群"),
                (quad(60, 100, 130, 120), "5,000.00"),
            ]

        def review(_i, _o, _r):
            z = pdfio._REVIEW_DPI / 72.0
            return {"structure_flags": [{
                "action": "merge_cells",
                "cells": [[60 * z, 60 * z, 110 * z, 80 * z],
                          [114 * z, 60 * z, 164 * z, 80 * z]],
                "confidence": 0.92,
            }]}

        dt = pdfio.extract_document_text(
            src, ocr=True, ocr_fn=ocr_fn, review_fn=review, log=lambda _m: None
        )
        texts = [b.text for b in dt.pages[0]]
        self.assertEqual(len(dt.pages[0]), 2)   # two fragments merged, cell kept
        self.assertIn("党群", texts)


class RenderedQaTest(unittest.TestCase):
    """The report-only rendered-output QA (``review_rendered_pages``)."""

    def _pair(self, pages: int = 1):
        src = _OUT / "qa_src.pdf"
        build_sample_pdf(src, pages=pages)
        out = _OUT / "qa_out.pdf"
        doc = fitz.open()
        doc.insert_pdf(fitz.open(str(src)))
        doc.save(str(out))
        doc.close()
        return src, out

    def test_review_rendered_pages_reports_issues(self):
        src, out = self._pair()
        logs: list[str] = []
        pdfio.review_rendered_pages(
            src, out,
            lambda _i, _o, _r: {"issues": [{"kind": "残余中文", "message": "仍有中文"}]},
            logs.append,
        )
        self.assertTrue(any("残余中文" in m and "仍有中文" in m for m in logs), logs)
        self.assertTrue(any("共 1 处待确认" in m for m in logs), logs)

    def test_review_rendered_pages_fail_closed(self):
        src, out = self._pair()
        logs: list[str] = []

        def boom(_i, _o, _r):
            raise RuntimeError("model down")

        # A reviewer error must never abort the export-side QA nor leave a log.
        pdfio.review_rendered_pages(src, out, boom, logs.append)
        self.assertEqual([], logs)
        # A ``None`` reviewer (vision model disabled) is a no-op.
        pdfio.review_rendered_pages(src, out, None, logs.append)
        self.assertEqual([], logs)

    def test_review_rendered_pages_ignores_malformed_result(self):
        src, out = self._pair()
        logs: list[str] = []
        pdfio.review_rendered_pages(src, out, lambda _i, _o, _r: "not a dict", logs.append)
        pdfio.review_rendered_pages(src, out, lambda _i, _o, _r: {"issues": "bad"}, logs.append)
        self.assertEqual([], logs)

    def test_review_rendered_pages_returns_correctable_pages(self):
        src, out = self._pair()
        logs: list[str] = []
        flagged = pdfio.review_rendered_pages(
            src, out,
            lambda _i, _o, _r: {"issues": [{"kind": "残余中文", "message": "仍有中文", "confidence": 0.9}]},
            logs.append,
        )
        # A correctable (残余中文) issue marks the page for correction.
        self.assertIn(0, flagged)
        # A layout-only issue is NOT correctable → not returned.
        flagged2 = pdfio.review_rendered_pages(
            src, out,
            lambda _i, _o, _r: {"issues": [{"kind": "文本越线", "message": "越线", "confidence": 0.95}]},
            logs.append,
        )
        self.assertNotIn(0, flagged2)
        # A low-confidence residual is filtered (below the floor) → not returned.
        flagged3 = pdfio.review_rendered_pages(
            src, out,
            lambda _i, _o, _r: {"issues": [{"kind": "残余中文", "message": "x", "confidence": 0.3}]},
            logs.append,
        )
        self.assertNotIn(0, flagged3)


class ClassifyKeepBlocksTest(unittest.TestCase):
    """Vision second opinion: release a rule-kept chart node the model says should translate."""

    def _chart(self, text: str):
        return pdfio.Block(text=text, page=0, x0=90, y0=78, x1=201.5, y1=91.4,
                           size=12.0, single_line=True, is_chart=True)

    def _src(self, name: str):
        p = _OUT / name
        build_sample_pdf(p, pages=1)
        return p

    def test_classify_releases_confident_translate(self):
        src = self._src("cls_src.pdf")
        pages = [[self._chart("二、公司组织架构图"), self._chart("股东大会")]]
        keep = {0, 1}

        def classify(_i, _o, _candidates):
            return {"classifications": [
                {"index": 0, "kind": "translate_heading", "confidence": 0.9},
            ]}

        released = pdfio.classify_keep_blocks(src, pages, classify, keep, None)
        self.assertEqual(released, {0})
        self.assertEqual(keep - released, {1})

    def test_classify_does_not_release_low_conf_or_keep_kind(self):
        src = self._src("cls_src2.pdf")
        pages = [[self._chart("二、公司组织架构图"), self._chart("股东大会")]]
        keep = {0, 1}

        def classify(_i, _o, _candidates):
            return {"classifications": [
                {"index": 0, "kind": "translate_heading", "confidence": 0.5},  # low
                {"index": 1, "kind": "keep_chart_node", "confidence": 0.9},     # keep kind
            ]}

        self.assertEqual(pdfio.classify_keep_blocks(src, pages, classify, keep, None), set())

    def test_classify_fail_closed(self):
        src = self._src("cls_src3.pdf")
        pages = [[self._chart("x")]]
        keep = {0}

        def boom(_i, _o, _candidates):
            raise RuntimeError("down")

        self.assertEqual(pdfio.classify_keep_blocks(src, pages, boom, keep, None), set())
        # A None classifier (vision disabled) is a no-op.
        self.assertEqual(pdfio.classify_keep_blocks(src, pages, None, keep, None), set())


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


class OcrTableRedrawTest(unittest.TestCase):
    """``redraw_ocr`` regenerates a scanned table page as a clean table."""

    def test_redraw_ocr_table_drops_raster_and_draws_cells(self):
        src = _OUT / "redraw_src.pdf"
        build_sample_pdf(src, pages=1)  # used only for the page size
        blocks = [
            pdfio.Block(text="总资产", page=0, x0=60, y0=100, x1=200, y1=112,
                        size=6.0, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="1,234,567.89", page=0, x0=210, y0=100, x1=360, y1=112,
                        size=6.0, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="总负债", page=0, x0=60, y0=120, x1=200, y1=132,
                        size=6.0, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="9,876,543.21", page=0, x0=210, y0=120, x1=360, y1=132,
                        size=6.0, single_line=True, ocr=True, in_table=True),
        ]
        trans = ["Total assets", "1,234,567.89", "Total liabilities", "9,876,543.21"]
        out = _OUT / "redraw_out.pdf"
        pdfio.save_translated_pdf(src, [blocks], [trans], str(out), "English", redraw_ocr=True)
        doc = fitz.open(str(out))
        page = doc[0]
        self.assertEqual(0, len(page.get_images(full=True)))  # no raster background
        self.assertGreater(len(page.get_drawings()), 0)       # grid rules drawn
        text = page.get_text("text")
        self.assertIn("Total assets", text)
        self.assertIn("9,876,543.21", text)
        doc.close()

    def test_redraw_ocr_off_keeps_inplace(self):
        src = _OUT / "redraw_src2.pdf"
        build_sample_pdf(src, pages=1)
        blocks = [
            pdfio.Block(text="总资产", page=0, x0=60, y0=100, x1=200, y1=112,
                        size=9.0, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="1,234,567.89", page=0, x0=210, y0=100, x1=360, y1=112,
                        size=9.0, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="总负债", page=0, x0=60, y0=120, x1=200, y1=132,
                        size=9.0, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="9,876,543.21", page=0, x0=210, y0=120, x1=360, y1=132,
                        size=9.0, single_line=True, ocr=True, in_table=True),
        ]
        trans = ["Total assets", "1,234,567.89", "Total liabilities", "9,876,543.21"]
        out = _OUT / "redraw_out2.pdf"
        # redraw_ocr=False (default) → the source page (with its raster/page content) is kept.
        pdfio.save_translated_pdf(src, [blocks], [trans], str(out), "English", redraw_ocr=False)
        doc = fitz.open(str(out))
        self.assertEqual(doc[0].rect.height, fitz.open(str(src))[0].rect.height)
        doc.close()

    def test_redraw_skips_chart_page(self):
        # A diagram (org chart) page has node labels, not data cells: redraw must
        # NOT blank it into an empty table. The page falls through to in-place.
        src = _OUT / "redraw_chart.pdf"
        build_sample_pdf(src, pages=1)
        chart_blocks = [
            pdfio.Block(text="董事会", page=0, x0=60, y0=100, x1=88, y1=130,
                        size=6.0, single_line=True, ocr=True, is_chart=True),
            pdfio.Block(text="监事会", page=0, x0=60, y0=140, x1=88, y1=170,
                        size=6.0, single_line=True, ocr=True, is_chart=True),
            pdfio.Block(text="委员会", page=0, x0=110, y0=100, x1=138, y1=130,
                        size=6.0, single_line=True, ocr=True, is_chart=True),
            pdfio.Block(text="3", page=0, x0=300, y0=270, x1=310, y1=282,
                        size=8.0, single_line=True, ocr=True),
        ]
        trans = [b.text for b in chart_blocks]
        out = _OUT / "redraw_chart_out.pdf"
        pdfio.save_translated_pdf(src, [chart_blocks], [trans], str(out), "English", redraw_ocr=True)
        doc = fitz.open(str(out))
        # Not redrawn: the original (source) page is kept, so its text survives.
        self.assertEqual(1, doc.page_count)
        self.assertIn("Page 1 heading", doc[0].get_text("text"))
        doc.close()

    def test_ai_table_rebuild_draws_regular_table(self):
        # With a model-derived grid, the OCR table page is drawn as a clean,
        # regular N x M table (no raster, regular grid, translated cells).
        src = _OUT / "ai_table_src.pdf"
        build_sample_pdf(src, pages=1)
        blocks = [
            pdfio.Block(text="总资产", page=0, x0=60, y0=100, x1=200, y1=112,
                        size=6.0, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="1,234,567.89", page=0, x0=210, y0=100, x1=360, y1=112,
                        size=6.0, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="总负债", page=0, x0=60, y0=120, x1=200, y1=132,
                        size=6.0, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="9,876,543.21", page=0, x0=210, y0=120, x1=360, y1=132,
                        size=6.0, single_line=True, ocr=True, in_table=True),
        ]
        trans = [b.text for b in blocks]
        grid = [["Item", "2025", "2024"],
                ["Total assets", "1,234,567.89", "999,999.99"],
                ["Total liabilities", "9,876,543.21", "888,888.88"]]
        out = _OUT / "ai_table_out.pdf"
        logs: list[str] = []
        pdfio.save_translated_pdf(src, [blocks], [trans], str(out), "English",
                                  redraw_ocr=True, table_rebuild_fn=lambda _i, _png: grid,
                                  log=logs.append)
        doc = fitz.open(str(out))
        page = doc[0]
        self.assertEqual(0, len(page.get_images(full=True)))   # clean, no raster
        self.assertGreater(len(page.get_drawings()), 0)        # regular grid
        text = page.get_text("text")
        self.assertIn("Total assets", text)
        self.assertIn("999,999.99", text)
        self.assertIn("Item", text)
        doc.close()
        # The AI-table-rebuild progress is surfaced in the log.
        self.assertTrue(any("正在 AI 表格重建" in m for m in logs), logs)
        self.assertTrue(any("AI 表格重建完成" in m for m in logs), logs)

    def test_ai_table_rebuild_invalid_falls_back(self):
        # An invalid / implausible rebuilt grid falls back to the geometric redraw.
        src = _OUT / "ai_table_src2.pdf"
        build_sample_pdf(src, pages=1)
        blocks = [
            pdfio.Block(text="总资产", page=0, x0=60, y0=100, x1=200, y1=112,
                        size=6.0, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="1,234,567.89", page=0, x0=210, y0=100, x1=360, y1=112,
                        size=6.0, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="总负债", page=0, x0=60, y0=120, x1=200, y1=132,
                        size=6.0, single_line=True, ocr=True, in_table=True),
            pdfio.Block(text="9,876,543.21", page=0, x0=210, y0=120, x1=360, y1=132,
                        size=6.0, single_line=True, ocr=True, in_table=True),
        ]
        trans = ["Total assets", "1,234,567.89", "Total liabilities", "9,876,543.21"]
        out = _OUT / "ai_table_out2.pdf"
        # table_rebuild_fn returns None → geometric redraw still produces content.
        pdfio.save_translated_pdf(src, [blocks], [trans], str(out), "English",
                                  redraw_ocr=True, table_rebuild_fn=lambda _i, _png: None)
        doc = fitz.open(str(out))
        self.assertIn("Total assets", doc[0].get_text("text"))
        doc.close()


if __name__ == "__main__":
    unittest.main()
