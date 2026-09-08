"""eval_ir.py — A/B：IR 文档级翻译（段落组批 + 文档级术语）vs 默认逐块翻译。

用途：量化 C-⑥ IR *翻译侧*（非重排版侧）相对默认「逐块翻译」的差异，回答
「把翻译单元从视觉块升级为段落、并注入文档级术语，是否更好 / 是否引入回归」。

两层：

* **离线层（默认，不调模型）**——对比 ``build_ir`` 的分组结构：总块数 vs 段落
  单元数、多少块被合并成一段、孤儿块、表格/标题/图注/OCR/公式 是否正确隔离、
  ``infer_terms`` 的术语候选数、以及按 ``max_blocks_per_batch`` 估算的请求数。
* **在线层（``--run``，需模型在线）**——用真实模型跑两遍：基线 ``translate_blocks``
  （逐块）vs 候选 ``translate_ir(infer=True)``（段落组批 + 术语），再用
  ``translate_app.eval.eval_pages`` 的硬指标（版式 fit / 数字保真 / 完整性）对比，
  并给出计时与术语注入情况。

用法::

    C:\\pv\\Scripts\\python.exe eval_ir.py --pdf test_sample_5pages.pdf          # 离线
    C:\\pv\\Scripts\\python.exe eval_ir.py --pdf test_sample_5pages.pdf --run    # 在线（需模型）

退出码：0 = 完成；2 = 用法/文件错误。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
# Windows console may use a legacy code page; force UTF-8 so the Chinese output
# is not mojibake'd when run from PowerShell / cmd.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from translate_app import ir as ir_mod
from translate_app import pdfio
from translate_app import translator
from translate_app import eval as ev


def _load_model(model_id: str):
    from translate_app.settings import load_models

    models = load_models()
    try:
        return next(m for m in models if m.id == model_id)
    except StopIteration:
        ids = ", ".join(m.id for m in models) or "(none)"
        raise SystemExit(f"找不到模型 {model_id!r}；可用：{ids}")


def _selected_pages(doc: pdfio.DocumentText, pages_spec: str | None) -> list[int]:
    """0-based pages to evaluate; ``None``/``""`` = all."""
    if not pages_spec:
        return list(range(doc.page_count))
    out: set[int] = set()
    for part in pages_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(p for p in out if 0 <= p < doc.page_count)


def offline(doc: pdfio.DocumentText, lang: str) -> None:
    """Grouping / terminology statistics that need no model."""
    doc_ir = ir_mod.build_ir(doc, lang=lang)
    blocks = [b for ipage in doc_ir.pages for b in ipage.blocks]
    units = ir_mod.prose_units([b for b in blocks if not ir_mod.is_structural_role(b.role)])

    n_blocks = len(blocks)
    n_units = len(units)
    n_merged = sum(1 for u in units if len(u) > 1)
    n_in_merged = sum(len(u) for u in units if len(u) > 1)
    n_orphan = sum(1 for u in units if len(u) == 1)
    terms = ir_mod.infer_terms(doc_ir)

    # role isolation: which blocks stayed their own unit (never prose-merged)
    by_role: dict[str, int] = {}
    for b in blocks:
        by_role[b.role] = by_role.get(b.role, 0) + 1
    n_table = sum(1 for b in blocks if b.role == "table_cell" or b.table_ref > 0)
    n_ocr = sum(1 for b in blocks if getattr(b.anchor, "ocr", False))
    n_structural = sum(1 for b in blocks if ir_mod.is_structural_role(b.role))

    print("=== IR 离线层：分组结构（不调模型）===")
    print(f"语言: {doc_ir.lang}  解析后端: {doc_ir.parser or 'geo(几何/无结构)'}")
    print(f"块角色分布: {by_role}")
    print(f"总块数: {n_blocks}  段落单元数: {n_units}")
    print(f"  合并段落: {n_merged} 段（含 {n_in_merged} 块，即 {n_in_merged}/{n_blocks} 块走段落级翻译）")
    print(f"  孤儿单元(单块): {n_orphan}")
    print(f"  表格单元格: {n_table}   OCR 块: {n_ocr}   结构保护(公式/图): {n_structural}")
    print(f"术语候选(infer_terms): {len(terms)} 条")
    if terms:
        print(f"  示例: {terms[:15]}")
    # request-count estimate at max_blocks_per_batch=40
    for cap in (25, 40):
        est_blocks = (n_blocks + cap - 1) // cap
        est_units = (n_units + cap - 1) // cap
        print(f"请求数估算(max_blocks_per_batch={cap}): 逐块={est_blocks}  段落组批={est_units}  "
              f"(省 {est_blocks - est_units} 次)")


def online(doc: pdfio.DocumentText, model, lang: str, pages: list[int]) -> None:
    """Real-model A/B: baseline (per-block) vs candidate (IR paragraph+terms)."""
    print(f"\n=== IR 在线层：真实模型 A/B（{model.name}）===")

    def log(msg):
        print("  [log]", msg)

    # A self-contained sub-document over just the selected pages, so IR's ``src_id``
    # (0-based within the sub-doc) lines up with the flat block list.
    sel_pages = [doc.pages[p] for p in pages]
    sel_blocks = [b for pg in sel_pages for b in pg]
    sel_block_pages = [k for k, pg in enumerate(sel_pages) for _ in pg]
    sub = pdfio.DocumentText(
        title=doc.title,
        blocks=[b.text for b in sel_blocks],
        block_pages=sel_block_pages,
        pages=sel_pages,
    )

    # ---- baseline: per-block -------------------------------------------------
    t0 = time.time()
    engine_b = translator.TranslationEngine(model)
    res_b = engine_b.translate_blocks(
        sub.blocks, lang, log=log, resume=False, block_pages=sub.block_pages)
    dt_b = time.time() - t0
    trans_b = list(res_b.translated)

    # ---- candidate: IR paragraph grouping + document terms -------------------
    t0 = time.time()
    engine_c = translator.TranslationEngine(model)
    doc_ir = ir_mod.build_ir(sub, lang=lang)
    fn = ir_mod.make_ir_translate_fn(engine_c, log=log, resume=False)
    translated_map = ir_mod.translate_ir(doc_ir, fn, lang=lang, log=log, infer=True)
    dt_c = time.time() - t0
    trans_c = [str(translated_map.get(k, sub.blocks[k])) for k in range(len(sub.blocks))]

    # ---- hard metrics --------------------------------------------------------
    pages_b = pdfio.group_by_page(sub.block_pages, trans_b, sub.page_count)
    pages_c = pdfio.group_by_page(sub.block_pages, trans_c, sub.page_count)
    ev_b = ev.eval_pages(sub.pages, pages_b, lang=lang)
    ev_c = ev.eval_pages(sub.pages, pages_c, lang=lang)
    delta = ev.compare(ev_b, ev_c)

    print(f"\n基线(逐块)        {dt_b:.1f}s  score={ev_b['score']}  layout={ev_b['layout']['counts']}")
    print(f"候选(IR段落+术语) {dt_c:.1f}s  score={ev_c['score']}  layout={ev_c['layout']['counts']}")
    print(f"  完整指标 基线: numbers={ev_b['numbers']['total']}  "
          f"missing={ev_b['complete']['missing']}  residual={ev_b['complete']['residual']}")
    print(f"  完整指标 候选: numbers={ev_c['numbers']['total']}  "
          f"missing={ev_c['complete']['missing']}  residual={ev_c['complete']['residual']}")
    print(f"\nA/B delta (负=候选更差): {delta}")
    # Number-error detail — the decisive dimension for a financial report.
    for label, pages_x in (("基线", pages_b), ("候选", pages_c)):
        nums = [ev.measure_numbers(sub.pages[k], pages_x[k], page=k)
                for k in range(len(sub.pages))]
        total_err = sum(n["count"] for n in nums)
        print(f"\n[{label}] 数字错误详情（共 {total_err} 处，显示前 12）:")
        shown = 0
        for k, n in enumerate(nums):
            for e in n["numbers"]:
                if shown >= 12:
                    break
                print(f"  页{pages[k]} 块{e['index']}: {e['source'][:38]!r} -> "
                      f"{e['translation'][:38]!r}  缺失={e['missing']} 多出={e['extra']}")
                shown += 1
            if shown >= 12:
                break
    moved = []
    for k, (pb, pc) in enumerate(zip(ev_b["per_page"], ev_c["per_page"])):
        sb = pb["layout"].get("counts", {})
        sc = pc["layout"].get("counts", {})
        if sb != sc:
            moved.append((pages[k], sb, sc))
    if moved:
        print("\n版式计数变化的页:")
        for p, sb, sc in moved:
            print(f"  原页{p}: {sb} -> {sc}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="A/B: IR document translation vs per-block")
    ap.add_argument("--pdf", default="test_sample_5pages.pdf")
    ap.add_argument("--run", action="store_true", help="在线真实模型 A/B（需模型在线）")
    ap.add_argument("--model", default="qwen3.8-local")
    ap.add_argument("--lang", default="English")
    ap.add_argument("--pages", default=None, help="逗号分隔 0-based 页号，如 0-4 或 0,2")
    args = ap.parse_args(argv)

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"找不到 {pdf}")
        return 2

    t0 = time.time()
    doc = pdfio.extract_document_text(str(pdf), ocr=False, log=lambda m: None)
    print(f"提取 {pdf.name}：{doc.page_count} 页 {len(doc.blocks)} 块（{time.time()-t0:.1f}s）")
    pages = _selected_pages(doc, args.pages)
    print(f"评估页: {pages}\n")

    offline(doc, args.lang)

    if args.run:
        model = _load_model(args.model)
        # Only the selected pages' blocks go to the model; rebuild a sub-doc.
        online(doc, model, args.lang, pages)
    else:
        print("\n（未加 --run：跳过在线 A/B。加 --run 需模型在线。）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
