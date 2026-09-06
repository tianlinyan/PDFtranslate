#!/usr/bin/env python3
"""verify_doclayout.py — 一键确认 DocLayout-YOLO 是否真正生效。

用法（在项目根目录，用带 DocLayout 的 venv python 运行）::

    C:\\pv\\Scripts\\python.exe verify_doclayout.py [doc.pdf] [--probe]

* 默认：调用 ``extract_document_structured(parser="doclayout")`` 拿到**诚实**的
  ``structure_parser``。

  - ``structure_parser == "doclayout"``  ⇒ DocLayout 模型**真正产出区域**，生效了；
  - ``structure_parser in ("geo", "")``  ⇒ 降级为几何后端（"包能导入 ≠ 真用了"）。
  同时打印 per-page 的语义 kind（caption/heading/table/figure/formula…）。

* ``--probe``：再直接加载模型并对第 1 页跑一次真实 ``model.predict``，打印区域数与
  类别——这是"DocLayout 推理链路确实通了"的最确凿证据。

一条命令即可判断：返回文件；``structure_parser == 'doclayout'`` 即为生效。模型缺失/
预测失败时，项目会自动降级几何、绝不崩，此脚本用来确认"到底走没走 DocLayout"。
"""
import sys
import time
from pathlib import Path


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    probe = "--probe" in sys.argv
    pdf = args[0] if args else "test.pdf"
    if not Path(pdf).exists():
        print(f"找不到 PDF：{pdf}")
        sys.exit(2)

    print(f"=== 检查 DocLayout 是否生效：{pdf} ===")
    t0 = time.time()

    from translate_app import pdfio

    dt = pdfio.extract_document_structured(
        pdf, parser="doclayout", ocr=False, log=lambda m: print("  [log]", m))
    sp = dt.structure_parser
    print("elapsed:", round(time.time() - t0, 1), "s")
    print("structure_parser =", repr(sp))
    print("summary =", pdfio.get_structure_summary(dt))
    for i, ps in enumerate(dt.page_structure):
        kinds = sorted({e["kind"] for e in ps.elements})
        if kinds:
            print(f"  page {i}: {kinds}")

    print("=>", "DocLayout 真正生效" if sp == "doclayout"
          else "未生效（已降级/几何后端）")

    if probe:
        _probe(pdf)


def _probe(pdf: str) -> None:
    """直接加载模型并对第 1 页跑真实推理 —— 最确凿的证据。"""
    print("\n=== 直接模型推理（确凿证据）===")
    import cv2
    import pymupdf as fitz  # noqa: N813 — avoid the deprecated ``fitz`` import warning
    import numpy as np
    from doclayout_yolo import YOLOv10

    from translate_app import pdfio as _p

    local = _p._resolve_doclayout_model(None)
    print("model file:", local)
    m = YOLOv10(local, task="detect")

    doc = fitz.open(pdf)
    img = doc[0].get_pixmap(dpi=150).tobytes("png")
    arr = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    res = m.predict(arr, imgsz=1024, conf=0.2, task="detect",
                    device=_p._doclayout_device())[0]
    names = getattr(res, "names", None) or {}
    n = len(res.boxes) if res.boxes is not None else 0
    print("model.predict 区域数:", n)
    if n:
        cls = res.boxes.cls.cpu().numpy().astype(int)
        xy = res.boxes.xyxy.cpu().numpy()
        for c, b in zip(cls, xy):
            print("  ", names.get(int(c)), [round(float(v), 1) for v in b])


if __name__ == "__main__":
    main()
