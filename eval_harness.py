"""A-② 评估 harness：给「版面保真 + 译文保真」建立可复用、可对比、离线可复现的度量。

两种输入模式::

    python eval_harness.py --pdf-only 原文.pdf 译文.pdf [--lang English] [--json out.json]
    python eval_harness.py --outdoc   原文.pdf out_doc.json [--lang English]
                            [--json out.json] [--baseline base.json] [--compare]

* ``--pdf-only`` —— 导出后**文本层后验**（复用 ``check_translation``：数字一致性 /
  残留中文 / 章节编号 / 页数）。适合「已导出的成品」。
* ``--outdoc``  —— **逐块版面度量**。out_doc.json 是 worker/agent 的扁平
  ``{flat_index: {"text": ...}}`` 译文叠加层；据此重建每页块 + 译文，跑
  ``translate_app.eval`` 的版式/数字/完整性硬指标并聚合评分。适合 **A/B 对比**
  （基线与候选各出一份 out_doc，``--compare`` 看 delta）。

退出码：0 = 通过；1 = 检测到数字不一致 / 漏译 / 残留（或 --compare 下硬指标回归）；
2 = 用法错误。度量逻辑与产线/审计共用一套原语，绝不另写一套（见 translate_app/eval.py）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from translate_app import eval as ev
from translate_app import pdfio

#: 硬门槛：--outdoc 模式下任一数字/漏译/残留都会判为失败。版面分仅作趋势。
FAIL_ON_DEFECTS = True


def _pdf_only(src: str, tgt: str, lang: str, skip: str | None,
              json_out: str | None) -> int:
    import check_translation as ct
    skip_set = ct._parse_page_spec(skip) if skip else set()
    checker = ct.run_checks(Path(src), Path(tgt), lang=lang, skip=skip_set)
    report = {
        "mode": "pdf-only",
        "lang": lang,
        "numeric": checker.numeric,
        "cjk": checker.cjk,
        "numbering": checker.numbering,
        "pages": checker.pages,
        "all_clear": checker.all_clear(),
    }
    if json_out:
        Path(json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for msg in checker.numeric:
        _say(f"[致命] 数字不一致 — {msg}")
    for msg in checker.cjk:
        _say(f"[残留] 中文残留 — {msg}")
    for msg in checker.numbering:
        _say(f"[编号] 章节编号 — {msg}")
    for msg in checker.pages:
        _say(f"[页数] 页数问题 — {msg}")
    _say(f"体检：{'通过' if checker.all_clear() else '有问题'} → {json_out or '(stdout)'}")
    return 0 if checker.all_clear() else 1


def _outdoc(src: str, outdoc: str, lang: str, json_out: str | None,
            baseline: str | None, compare: bool, judge: str | None = None) -> int:
    dt = pdfio.extract_document_text(src, ocr=False, log=lambda m: None)
    try:
        out = json.loads(Path(outdoc).read_text("utf-8"))
    except Exception as exc:  # noqa: BLE001
        _say(f"无法读取 out_doc：{type(exc).__name__}: {exc}")
        return 2
    flat = {int(k): v for k, v in out.items()}

    pages_blocks = dt.pages
    pages_trans: list[list[str]] = []
    offset = 0
    for page in pages_blocks:
        page_trans = []
        for j in range(len(page)):
            page_trans.append(str((flat.get(offset + j) or {}).get("text", "")))
        pages_trans.append(page_trans)
        offset += len(page)

    summary = ev.eval_pages(pages_blocks, pages_trans, lang=lang)
    if judge:
        try:
            summary["judge"] = ev.judge_pages(pages_blocks, pages_trans, lang=lang,
                                              judge_fn=_make_judge_fn(judge))
        except Exception as exc:  # noqa: BLE001 — judge outage degrades, never aborts
            _say(f"[warn] LLM-as-judge 不可用：{type(exc).__name__}: {exc}")
    if baseline and compare:
        try:
            base = json.loads(Path(baseline).read_text("utf-8"))
            summary["delta"] = ev.compare(base, summary)
        except Exception as exc:  # noqa: BLE001
            _say(f"[warn] 无法读取 baseline 或对比：{type(exc).__name__}: {exc}")

    if json_out:
        Path(json_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    _say(f"score={summary['score']} layout.total={summary['layout']['total']} "
         f"numbers={summary['numbers']['total']} "
         f"missing={summary['complete']['missing']} "
         f"residual={summary['complete']['residual']} → {json_out or '(stdout)'}")

    defects = (summary["numbers"]["total"] or summary["complete"]["missing"]
               or summary["complete"]["residual"])
    if defects and FAIL_ON_DEFECTS:
        return 1
    if compare and summary.get("delta", {}).get("score_delta", 0) < -1e-9:
        # 候选比基线差：硬指标上有回归。
        return 1
    return 0


def _say(msg: str) -> None:
    print(msg, file=sys.stderr)


def _make_judge_fn(model_id: str):
    """Build a real ``judge_fn(page, src_text, tgt_text) -> 0..100`` (LLM-as-judge).

    Uses the OpenAI-compatible model ``model_id`` from ``models.json`` to score one
    page (fidelity / terminology / readability).  Only the score number is kept.
    Requires the model to be online — calling it is opt-in (``--judge``).
    """
    import json
    import re as _re

    from openai import OpenAI

    from translate_app.settings import ModelConfig

    raw = json.load(open("models.json", encoding="utf-8"))["models"]
    model = ModelConfig.from_dict(next(m for m in raw if m["id"] == model_id))
    client = OpenAI(**model.client_kwargs())
    sys_prompt = ("你是译文质量评审。比较源代码与译文，就【忠实度/术语一致/可读性】给 0-100 分。"
                  "只输出一个 0-100 的数字，不要任何其它文字。")
    request_params = model.request_params() if hasattr(model, "request_params") else {}
    temperature = getattr(model, "temperature", 0.0)

    def judge_fn(page: int, src_text: str, tgt_text: str) -> float:
        resp = client.chat.completions.create(
            model=model.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"源文：\n{src_text[:4000]}\n\n译文：\n{tgt_text[:4000]}"},
            ],
            temperature=temperature,
            **request_params,
        )
        content = (resp.choices[0].message.content or "0")
        m = _re.search(r"(\d{1,3}(?:\.\d+)?)", content)
        return float(m.group(1)) if m else 0.0
    return judge_fn


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pdf-only", action="store_true",
                   help="导出后文本层后验（原文+译文两个 PDF）")
    p.add_argument("--outdoc", action="store_true",
                   help="逐块版面度量（原文 PDF + out_doc.json）")
    p.add_argument("file1", nargs="?", help="原文 PDF")
    p.add_argument("file2", nargs="?", help="译文 PDF 或 out_doc.json")
    p.add_argument("--lang", default="English", help="目标语言（默认 English）")
    p.add_argument("--skip", default=None, help="跳过的页，如 '24-27'")
    p.add_argument("--json", default=None, help="机器可读报告输出路径")
    p.add_argument("--baseline", default=None, help="baseline 报告（--compare 用）")
    p.add_argument("--compare", action="store_true", help="与 --baseline 对比并输出 delta")
    p.add_argument("--judge", default=None, metavar="MODEL_ID",
                   help="用 models.json 里的模型做 LLM-as-judge 译文保真软评分（可选、需模型在线）")
    args = p.parse_args(argv)

    if not args.file1 or not args.file2:
        p.print_help()
        return 2
    if args.pdf_only and args.outdoc:
        _say("--pdf-only 与 --outdoc 互斥，只能选其一。")
        return 2
    if args.pdf_only:
        return _pdf_only(args.file1, args.file2, args.lang, args.skip, args.json)
    if args.outdoc:
        return _outdoc(args.file1, args.file2, args.lang, args.json,
                       args.baseline, args.compare, judge=args.judge)
    _say("未指定模式：需要 --pdf-only 或 --outdoc。")
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
