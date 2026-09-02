"""译文体检：导出后逐项核对原文与译文的数字、残留中文与章节编号。

用法::

    python check_translation.py 原文.pdf 译文.pdf [--lang English] [--strict]
                                 [--skip 24-27]

检查项
------
1. **数字一致性**（对正确性最致命的一类，发现即 exit 1）：按页提取双方
   文本中的所有数字 token，剥离标点得到数字序列，逐位比较。抓得出
   ``3,702.726,474.45`` vs ``3,702,726,474.45`` 这类千分位错乱、丢小数点
   与拆行错位。源页没有文本层（扫描页，数字来自 OCR/重排，不能作为基准）
   时该页自动跳过；``--skip`` 可额外排除混版式的页。
2. **残留中文**：目标语言为西文时，扫描译文中的 CJK 字符（人名列漏译、
   报表页残留等）。
3. **章节编号**：比对双方「行首编号」（``1.`` / ``1.1`` / ``第4章`` /
   ``Chapter 4``）的数字序列，顺序或数量不一致即告警；译文全文的编号
   风格（点分层次 / 中文 / 单词前缀）也应统一。
4. **页数合理性**：译文页数不应少于原文；少了即告警。

退出码：0 = 全部通过；1 = 数字不一致（或 ``--strict`` 下任意告警）；
2 = 用法错误。
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import pymupdf as fitz

#: A numeric token: digits with grouped separators / a percentage sign.  The
#: token must END in a digit (or %) so a trailing sentence period (``2025.``)
#: is not swallowed into the number.
_NUM_TOKEN_RE = re.compile(r"[+-]?[0-9][0-9,.，．]*[0-9%]|[+-]?[0-9]")

#: Arabic hierarchy at the start of a line: ``1.``、``1.1``、``1.1.2``.  The
#: marker must be followed by a non-digit (a word), so table rows that merely
#: begin with a number (``0.21 0.22 …``) are not mistaken for headings.
_ARABIC_HEADING_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})*)[\.、．]\s*(\D)")

#: Chinese-style headings: ``第4章``、``第四十七条``、``第三篇``。
_CN_HEADING_RE = re.compile(r"第\s*([0-9]+|[一二三四五六七八九十百\d]+)\s*[章节条篇节]")

#: English word prefixes for headings / clauses.
_WORD_HEADING_RE = re.compile(
    r"^\s*(?:Section|Chapter|Part|Article|Item|Clause)\s+([0-9]+)\b",
    re.IGNORECASE,
)

_CN_NUMERALS = "零一二三四五六七八九"


def _cn_to_int(text: str) -> int | None:
    """Normalize a Chinese numeral (``四``→4, ``十二``→12); None if unsupported."""
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    total = 0
    current = 0
    for ch in text:
        if ch.isdigit():
            current = current * 10 + int(ch)
        elif ch == "十":
            total += current * 10 if current else 10
            current = 0
        elif ch in ("百", "千"):
            return None  # hundreds/thousands: out of scope for section numbers
        else:
            idx = _CN_NUMERALS.find(ch)
            if idx < 0:
                return None
            current = current * 10 + idx
    return total + current


def _page_numbers(text: str) -> Counter[str]:
    """All numeric tokens of a page, as canonical number strings.

    Each token keeps its digits *and* the role of each separator
    (``,`` -> C, ``.`` -> D): ``3,702,726,474.45`` becomes
    ``3C702C726C474D45``, so a comma/dot swap like ``3,702.726,474.45``
    is caught even though the digit sequences are identical, and the
    leading zero is kept so a lost ``0.`` (a dropped decimal point)
    shows up as well.
    """
    out: Counter[str] = Counter()
    for tok in _NUM_TOKEN_RE.findall(text):
        canonical: list[str] = []
        neg = False
        for ch in tok:
            if ch.isdigit():
                canonical.append(ch)
            elif ch in ".．":
                canonical.append("D")
            elif ch in ",，":
                canonical.append("C")
            elif ch == "-":
                neg = True
        if canonical:
            out[("M" if neg else "") + "".join(canonical)] += 1
    return out


def _section_numbers(text: str) -> list[tuple[str, int]]:
    """Heading numbers of a page, in order of appearance.

    Each entry is ``(raw, value)`` where ``raw`` is the token as written
    (``"1.1"`` / ``"第四"`` / ``"Chapter 4"``) and ``value`` its int value,
    so the sequence compares across language styles.
    """
    found: list[tuple[str, int]] = []
    for line in text.splitlines():
        m = _ARABIC_HEADING_RE.match(line)
        if m:
            parts = m.group(1).split(".")
            value = 0
            for p in parts:
                value = value * 100 + int(p)
            found.append((m.group(1), value))
            continue
        m = _CN_HEADING_RE.search(line)
        if m:
            num = m.group(1)
            value = _cn_to_int(num)
            if value is not None:
                found.append((num, value))
                continue
        m = _WORD_HEADING_RE.match(line)
        if m:
            found.append((f"word {m.group(1)}", int(m.group(1))))
    return found


def _style_of(tokens: Sequence[tuple[str, int]]) -> str:
    """One style tag for a page's headings: ``dot`` / ``plain`` / ``cn`` / ``word``."""
    styles: set[str] = set()
    for raw, _v in tokens:
        if raw.startswith("word"):
            styles.add("word")
        elif "." in raw:
            styles.add("dot")
        elif raw.isdigit():
            styles.add("plain")
        elif _has_cjk(raw):
            styles.add("cn")
        else:
            styles.add("paren")
    return next(iter(styles)) if len(styles) == 1 else "mixed"


def _snippet(text: str, width: int = 60) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= width:
        return flat
    return flat[:width] + "…"


def _has_cjk(text: str) -> bool:
    return any("一" <= c <= "鿿" for c in text)


class Checker:
    """Collects issues while walking both documents page by page."""

    def __init__(self, lang: str, skip: set[int] | None = None):
        self.lang = lang
        self.skip = skip or set()
        self.numeric: list[str] = []
        self.cjk: list[str] = []
        self.numbering: list[str] = []
        self.pages: list[str] = []
        self.max_reported = 20

    def numeric_ok(self) -> bool:
        return not self.numeric

    def all_clear(self) -> bool:
        return not (self.numeric or self.cjk or self.numbering or self.pages)

    def check(self, source: Path, target: Path, skip_scan: bool = True) -> None:
        src = fitz.open(str(source))
        try:
            tgt = fitz.open(str(target))
            try:
                self._check_page_counts(src, tgt)
                n = min(src.page_count, tgt.page_count)
                tgt_styles: list[str] = []
                for i in range(n):
                    if i + 1 in self.skip:
                        continue
                    style = self._check_page(src[i], tgt[i], i, skip_scan)
                    if style and style != "mixed":
                        tgt_styles.append(style)
                if len(set(tgt_styles)) > 1:
                    self.numbering.append(
                        f"译文章节编号风格全文不一致（"
                        f"{sorted(set(tgt_styles))}），建议统一。"
                    )
            finally:
                tgt.close()
        finally:
            src.close()

    # -- individual checks --------------------------------------------------

    def _check_page_counts(self, src: fitz.Document, tgt: fitz.Document) -> None:
        if tgt.page_count < src.page_count:
            self.pages.append(
                f"译文页数（{tgt.page_count}）少于原文（{src.page_count}）——"
                "可能漏页。"
            )

    def _check_page(
        self, src_page: fitz.Page, tgt_page: fitz.Page, i: int, skip_scan: bool
    ) -> str:
        where = f"第 {i + 1} 页"
        src_text = src_page.get_text("text") or ""
        tgt_text = tgt_page.get_text("text") or ""
        if skip_scan and not src_text.strip():
            return "mixed"  # 无文本层的扫描页：数字来自 OCR，不能作为基准

        # 1) 数字一致性：逐位比较数字序列。
        src_nums = _page_numbers(src_text)
        tgt_nums = _page_numbers(tgt_text)
        only_src = src_nums - tgt_nums
        only_tgt = tgt_nums - src_nums
        if only_src or only_tgt:
            diffs: list[str] = []
            for d, n in only_src.most_common():
                diffs.append(f"原文 {n} 次「{d}」在译文中缺失")
                if len(diffs) >= self.max_reported:
                    break
            for d, n in only_tgt.most_common():
                if len(diffs) >= self.max_reported:
                    break
                diffs.append(f"译文多出 {n} 次「{d}」（不在原文中出现）")
            self.numeric.append(
                f"{where}: 数字序列不一致：" + "；".join(diffs)
            )

        # 2) 残留中文（仅西文目标）。
        if not _has_cjk(self.lang):
            residual = [ch for ch in tgt_text if "一" <= ch <= "鿿"]
            if residual:
                self.cjk.append(
                    f"{where}: 残留 {len(residual)} 个中文字符"
                    f"（如 {''.join(residual[:8])}…）：{_snippet(tgt_text)}"
                )

        # 3) 章节编号：序列对比（值）＋ 风格标签（本页）。
        src_sections = _section_numbers(src_text)
        tgt_sections = _section_numbers(tgt_text)
        src_vals = [v for _raw, v in src_sections]
        tgt_vals = [v for _raw, v in tgt_sections]
        if src_vals and src_vals != tgt_vals:
            self.numbering.append(
                f"{where}: 章节编号不一致 "
                f"原文 {[r for r, _v in src_sections]} ≠ "
                f"译文 {[r for r, _v in tgt_sections]}"
            )
        return _style_of(tgt_sections)


def run_checks(
    source: Path,
    target: Path,
    lang: str = "English",
    skip: set[int] | None = None,
) -> Checker:
    checker = Checker(lang=lang, skip=skip)
    checker.check(Path(source), Path(target))
    return checker


def _parse_page_spec(text: str) -> set[int]:
    pages: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            try:
                pages.update(range(int(lo_s), int(hi_s) + 1))
            except ValueError:
                raise SystemExit(f"无法解析页数范围：{part}")
        else:
            try:
                pages.add(int(part))
            except ValueError:
                raise SystemExit(f"无法解析页数：{part}")
    return pages


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    lang = "English"
    skip_text: str | None = None
    strict = False
    files: list[str] = []

    it = iter(argv)
    for arg in it:
        if arg == "--lang":
            lang = next(it, "")
        elif arg == "--strict":
            strict = True
        elif arg == "--skip":
            skip_text = next(it, "")
        elif arg in ("-h", "--help"):
            print(__doc__)
            return 0
        elif arg.startswith("-"):
            print(f"未知参数：{arg}", file=sys.stderr)
            return 2
        else:
            files.append(arg)

    if len(files) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    skip = _parse_page_spec(skip_text) if skip_text else set()
    checker = run_checks(Path(files[0]), Path(files[1]), lang=lang, skip=skip)

    print(f"原文：{files[0]}")
    print(f"译文：{files[1]}")
    print(f"目标语言：{lang}\n")

    for msg in checker.numeric:
        print(f"[致命] 数字不一致 — {msg}")
    for msg in checker.cjk:
        print(f"[残留] 中文残留 — {msg}")
    for msg in checker.numbering:
        print(f"[编号] 章节编号 — {msg}")
    for msg in checker.pages:
        print(f"[页数] 页数问题 — {msg}")

    print()
    if checker.all_clear():
        print("体检通过：未发现问题。")
        return 0
    if not checker.numeric_ok():
        print("结论：存在数字不一致，请修正后重新导出。")
        return 1
    if strict:
        print("结论：--strict 模式下存在其它问题，请人工复核。")
        return 1
    print("结论：数字一致；存在需要人工复核的告警。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
