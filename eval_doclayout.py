#!/usr/bin/env python3
"""eval_doclayout.py — 评估 DocLayout 的真实作用（语义结构 vs 几何）。

DocLayout 并不改块 bbox/几何（那是提取层），它加的是 **语义 kind**（formula/figure/
table/heading/caption），供 ``classify_page``、agent 的 ``read_page``、IR ``_role_of``、
审计 ``_check_*`` 使用。所以对它的评估要看：**geo 和 doclayout 谁识别的结构更准**，
以及这能否让"公式保留原文 / 表格按表处理 / 特殊页分型"更好。

用法::

    C:\\pv\\Scripts\\python.exe eval_doclayout.py [--pdf test_sample_5pages.pdf]

输出：逐页对比 geo / doclayout 的 formula/table/figure/heading/caption 区域数、
``classify_page`` 分型、被标记为 formula（将保留原文）的块数，以及 doclayout 推理耗时。
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from translate_app import pdfio


def _kind_counts(structure):
    if structure is None:
        return Counter()
    return Counter(str(el.get("kind", "text")) for el in getattr(structure, "elements", None) or [])


def compare(pdf: Path):
    print(f"=== DocLayout vs geo 语义结构对比（{pdf.name}）===\n")
    t0 = time.time()
    dt_geo = pdfio.extract_document_structured(str(pdf), parser="geo", ocr=False, log=lambda m: None)
    t_geo = time.time() - t0
    t0 = time.time()
    dt_dl = pdfio.extract_document_structured(str(pdf), parser="doclayout", ocr=False, log=lambda m: None)
    t_dl = time.time() - t0

    print(f"提取+结构耗时: geo={t_geo:.1f}s  doclayout={t_dl:.1f}s   (doclayout 额外 ~{max(0, t_dl-t_geo):.1f}s)\n")

    hdr = (f"{'页':>3} {'kind(geo)':<34} {'classify(geo)':<12} "
           f"{'kind(doclayout)':<40} {'classify(dl)':<12} {'formula保留':>9}")
    print(hdr)
    print("-" * (len(hdr) + 10))
    tot_g = Counter()
    tot_d = Counter()
    for p in range(len(dt_geo.pages)):
        sg = dt_geo.page_structure[p] if dt_geo.page_structure else None
        sd = dt_dl.page_structure[p] if dt_dl.page_structure else None
        kg = _kind_counts(sg)
        kd = _kind_counts(sd)
        tot_g.update(kg)
        tot_d.update(kd)
        cg = pdfio.classify_page(dt_geo.pages[p], sg)
        cd = pdfio.classify_page(dt_dl.pages[p], sd)
        fg = sum(1 for b in dt_geo.pages[p] if pdfio._is_formula_block(str(b.text)))
        fd = sum(1 for b in dt_dl.pages[p] if pdfio._is_formula_block(str(b.text)))
        print(f"{p:>3} {str(dict(kg)):<34} {str(cg):<12} "
              f"{str(dict(kd)):<40} {str(cd):<12} {fg:>7}/{fd:>3}")

    print("\n=== 聚合（全文档区域 kind 计数）===")
    print(f"geo      : {dict(tot_g)}")
    print(f"doclayout: {dict(tot_d)}")
    print(f"\n关键差异：")
    print(f"  table 区域:   geo={tot_g.get('table',0)}  doclayout={tot_d.get('table',0)}")
    print(f"  formula 区域: geo={tot_g.get('formula',0)}  doclayout={tot_d.get('formula',0)}")
    print(f"  figure 区域:  geo={tot_g.get('figure',0)}  doclayout={tot_d.get('figure',0)}")
    print(f"  caption 区域: geo={tot_g.get('caption',0)}  doclayout={tot_d.get('caption',0)}")
    # page-triage差异
    sup = []
    for p in range(len(dt_geo.pages)):
        cg = pdfio.classify_page(dt_geo.pages[p], dt_geo.page_structure[p] if dt_geo.page_structure else None)
        cd = pdfio.classify_page(dt_dl.pages[p], dt_dl.page_structure[p] if dt_dl.page_structure else None)
        if cg != cd:
            sup.append((p, cg, cd))
    print(f"  classify_page 不同的页: {sup if sup else '无（分型一致）'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="DocLayout vs geo structure comparison")
    ap.add_argument("--pdf", default="test_sample_5pages.pdf")
    args = ap.parse_args(argv if argv is not None else None)
    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"找不到 {pdf}")
        return 2
    compare(pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
