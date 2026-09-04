"""Real-model self-check walkthrough on test_mintai_sample.pdf.

Phase A: batch-translate the content pages via the configured model, populate
``out_doc``, and run the five deterministic audit tools on REAL translations.

Phase B: drive the AI review loop (``run_page_visual`` + ``prompts.review_page_task``)
per page so the agent actually calls the audit tools (and render_page) through the
real model, then report the agent log + the ground-truth audit.

Run: python verify_selfcheck.py
"""

import json
import os
import sys
import time
from pathlib import Path

import pymupdf as fitz

sys.path.insert(0, ".")
from translate_app import pdfio, prompts
from translate_app.agent import flow, WorkflowState
from translate_app.settings import ModelConfig
from translate_app.translator import TranslationEngine, _needs_translation

SRC = "test_mintai_sample.pdf"
PAGES = [1, 2]              # content pages (text-layer); p0 is a cover, p3/p4 are scans
TARGET = "English"
MAX_STEPS = 20              # bounded per-page review loop
OUT = Path(os.environ.get("TEMP", r"C:\Users\tly00\AppData\Local\Temp")) / "selfcheck"

model = None
state = None


def _load():
    global model, state
    raw = json.load(open("models.json", encoding="utf-8"))["models"]
    model = ModelConfig.from_dict(next(m for m in raw if m["id"] == "qwen3.8"))
    print("model:", model.name, model.endpoint, "vision=", bool(getattr(model, "vision", False)))
    dt = pdfio.extract_document_text(SRC, ocr=False, log=lambda m: print("  [ext]", m))
    state = flow.WorkflowState(SRC, TARGET)
    state.src_doc = dt
    state.out_doc = {}   # the agent's writable translation overlay
    print("pages:", len(dt.pages), "lang:", state.lang)
    return dt


def render_page(page: int, what: str = "translation"):
    """Standalone render handler for the agent's ``render_page`` (mirrors worker)."""
    pages = state.src_doc.pages
    if not (0 <= page < len(pages)):
        return None
    doc = fitz.open(SRC)
    try:
        p = doc[page]
        if what == "translation":
            offset = sum(len(pg) for pg in pages[:page])
            to_draw = [(b, str(e.get("text", "")).strip())
                       for i, b in enumerate(pages[page])
                       if isinstance(e := state.out_doc.get(offset + i), dict)
                       and str(e.get("text", "")).strip()]
            if not to_draw:
                return None
            for b, _t in to_draw:
                if not getattr(b, "ocr", False) and not getattr(b, "is_chart", False):
                    p.add_redact_annot(fitz.Rect(b.x0, b.y0, b.x1, b.y1))
            p.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                               graphics=fitz.PDF_REDACT_LINE_ART_NONE)
            for b, t in to_draw:
                if getattr(b, "ocr", False):
                    p.draw_rect(fitz.Rect(b.x0 - 0.5, b.y0 - 0.5, b.x1 + 0.5, b.y1 + 0.5),
                                color=None, fill=(1, 1, 1))
                pdfio._draw_translated_block(p, pdfio._CJK_FONT, b, t)
        return pdfio._render_page_png(p, dpi=200)
    finally:
        doc.close()


def phase_a_translate_and_audit(dt):
    """Batch-translate content pages through the real engine, populate out_doc, audit."""
    engine = TranslationEngine(model)
    print("\n##### PHASE A: real batch translation + deterministic audit #####")
    for page in PAGES:
        blocks = dt.pages[page]
        offset = sum(len(pg) for pg in dt.pages[:page])
        texts = [b.text for b in blocks]
        t0 = time.time()
        res = engine.translate_blocks(texts, TARGET, log=lambda m: None,
                                      doc_path=Path(SRC), resume=False)
        print(f"\n== page {page + 1}: {len(blocks)} blocks, "
              f"translated {sum(1 for i, t in enumerate(res.translated)
                                if i < len(texts) and t and t != texts[i])} in "
              f"{time.time() - t0:.1f}s, errors={res.errors}")
        for i, t in enumerate(res.translated):
            idx = offset + i
            src = texts[i]
            if i < len(texts) and t and t.strip() and t != src and _needs_translation(src):
                state.out_doc[idx] = {"text": str(t)}
        # Deterministic audit on the real in-progress translation.
        tools = flow.make_page_executors(state, model, log=lambda m: None,
                                         render_handler=render_page)
        for name in ("check_residual", "check_missing", "check_numbers",
                     "check_table", "check_layout"):
            try:
                r = tools[name](page)
            except Exception as exc:  # noqa: BLE001
                print(f"   [audit] {name} ERROR: {type(exc).__name__}: {exc}")
                continue
            summary = json.dumps(r, ensure_ascii=False)
            if len(summary) > 700:
                summary = summary[:700] + "…"
            print(f"   [audit] {name}: {summary}")


def phase_b_review_loop(dt):
    """Run the AI self-check (review_page_task) per page through the real model."""
    print("\n##### PHASE B: AI-driven self-check (run_page_visual + review_page_task) #####")
    OUT.mkdir(parents=True, exist_ok=True)
    for page in PAGES:
        print(f"\n== review page {page + 1} (max_steps={MAX_STEPS}) ==")
        state.budget.used_steps = 0
        task = prompts.review_page_task(page)
        t0 = time.time()
        try:
            flow.run_page_visual(
                state, page, model, task=task, log=lambda m: print("      " + m),
                max_steps=MAX_STEPS, cancel=lambda: False,
                render_handler=render_page,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"   review failed: {type(exc).__name__}: {exc}")
        print(f"   (page {page + 1} review took {time.time() - t0:.1f}s, "
              f"{state.budget.used_steps}/{state.budget.max_steps} steps)")
        for line in state.log:
            print("      LOG:", line)
        # Ground-truth audit after the review (what the tools conclude now).
        tools = flow.make_page_executors(state, model, log=lambda m: None,
                                         render_handler=render_page)
        resi = tools["check_residual"](page)
        miss = tools["check_missing"](page)
        nums = tools["check_numbers"](page)
        tab = tools["check_table"](page)
        lay = tools["check_layout"](page)
        print(f"   POST residual={len(resi['residual'])} missing={len(miss['missing'])} "
              f"numbers={len(nums['numbers'])} table_complete={tab['complete']} "
              f"layout_issues={lay['count']}")
        with open(OUT / f"page{page + 1}.json", "w", encoding="utf-8") as fh:
            json.dump({"residual": resi, "missing": miss, "numbers": nums,
                       "table": tab, "layout": lay,
                       "log": state.log[-40:]}, fh, ensure_ascii=False, indent=1)
    print("\nwrote:", OUT)


def main():
    dt = _load()
    phase_a_translate_and_audit(dt)
    phase_b_review_loop(dt)
    print("\nDONE")


if __name__ == "__main__":
    main()
