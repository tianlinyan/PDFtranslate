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
4. **页数合理性**：译文页数不应少于原文；少了即告警。**双语（交错）产物
   按「源页 i ↔ 译文页 2i+1」配对**检查（页数为 2×原文时），不再把原文页当
   译文页比对——那会让真正的译文页从不被检查（错误的数字也判「体检通过」）。
   **扩页（``译文扩页``）产物**按导出器写进 PDF（XMP ``pageMap``）的「源页 →
   首个输出页」映射，把一页源文的所有续页一起比对；没有映射而页数又多于原文
   时明确告警「配对可能错位」，不静默错配。

退出码：0 = 全部通过；1 = 数字不一致（或 ``--strict`` 下任意告警）；
2 = 用法错误。
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

import pymupdf as fitz

# Let the script run from the repo root without installing anything (same trick as
# ``check_layout.py``): ``translate_app`` is a sibling package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from translate_app import pdfio  # noqa: E402

#: A numeric token: digits with grouped separators / a percentage sign.  The
#: token must END in a digit (or %) so a trailing sentence period (``2025.``)
#: is not swallowed into the number.
_NUM_TOKEN_RE = re.compile(r"[+-]?[0-9][0-9,.，．]*[0-9%]|[+-]?[0-9]")

#: Full-width digits / separators / signs → ASCII.  Without this a full-width
#: source (``１，２３４．５６``) yielded NO tokens while its ASCII translation did,
#: so a *correct* translation was reported as "译文多出 …".
_FULLWIDTH_MAP = str.maketrans("０１２３４５６７８９，．％－＋", "0123456789,.%-+")
#: Unicode minus / dashes → ASCII hyphen-minus.
_MINUS_MAP = str.maketrans("−–—", "---")

#: Unit words that legitimately rescale a value (``1,234.56 万元`` ==
#: ``12,345,600 yuan``).  Kept in sync with the agent's number audit
#: (``translate_app.agent.flow._unit_multiplier``); ``test_check_translation``
#: pins the same sample table so the two cannot drift apart silently.
_UNIT_EN_RE = re.compile(r"^\s*((?:ten|hundred)\s+)?(thousand|million|billion|trillion)\b",
                         re.IGNORECASE)
_UNIT_EN_POWER = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12}
_UNIT_CN_RE = re.compile(r"^\s*(万亿|亿|万)\s*")
_UNIT_CN_POWER = {"万亿": 12, "亿": 8, "万": 4}
#: Currency words carry no multiplier but DO mark the token as a value that may be
#: rendered with a different surface form (``1,234.56 万元`` → ``12,345,600 yuan``).
_CURRENCY_RE = re.compile(
    r"^\s*(?:元|人民币|RMB|yuan|USD|CNY|dollars?)(?![A-Za-z])", re.IGNORECASE)


def _unit_multiplier(window: str) -> Decimal:
    """The scale a unit word right after a number applies to it (``1`` when none)."""
    m = _UNIT_EN_RE.match(window)
    if m:
        power = _UNIT_EN_POWER[m.group(2).lower()]
        prefix = (m.group(1) or "").lower().strip()
        return Decimal(10) ** (power + (2 if prefix == "hundred" else
                                        1 if prefix == "ten" else 0))
    m = _UNIT_CN_RE.match(window)
    if m:
        return Decimal(10) ** _UNIT_CN_POWER[m.group(1)]
    return Decimal(1)

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

#: A Chinese ordinal enumeration heading on its own line: ``一、`` ``二、`` ``（四）`` ``十二、``.
#: These are translated to ``1.`` ``2.`` ``(4)`` ``12.`` for a Latin target, so they
#: must be recognised here (the source carries no digit for them) or the section
#: sequences would be thought to differ.
_CN_ORD_HEADING_RE = re.compile(
    r"^\s*(?:[（(]\s*([一二三四五六七八九十]{1,3})\s*[)）]|([一二三四五六七八九十]{1,3})\s*[、．])"
)

#: A parenthesised ordinal / note marker at the start of a line: ``(4)`` / ``（4）``.
#: For a Latin target the model renders （四） as (4), so the translation side must
#: recognise this form too or its headings would look absent next to the source's
#: recognised （四） → 4.
_PAREN_HEADING_RE = re.compile(r"^\s*[（(]\s*([0-9]{1,4})\s*[)）]\s*\S")

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


#: Chinese ordinal markers that the model renders as Arabic digits for a
#: Latin-script target: ``一、二、…十、`` and ``（一）（二）…（十）`` plus ``第X节/章/条/篇``.
#: The digit comparator normalizes these in BOTH texts so converting them
#: (``一、``→``1、``, ``（四）``→``(4)``, ``第二节``→``2节``) is not mistaken for a
#: figure the source lacks — those are section numbers, not amounts.
#:
#: The marker's own delimiter is **preserved**: dropping it glued the ordinal digit
#: onto a following amount (``一、1,234.56`` → ``11,234.56``, ``1,234（一）`` →
#: ``1,2341``), which made a perfectly correct translation report a fatal
#: "数字不一致" and exit 1.
_CN_ORD_PATTERN = re.compile(
    r"第\s*(?P<num_ch>[一二三四五六七八九十]{1,3})\s*(?P<suffix>[章节条篇])"
    r"|(?P<num_sep>[一二三四五六七八九十]{1,3})\s*(?P<sep>[、．])"
    r"|(?P<open>[（(])\s*(?P<num_paren>[一二三四五六七八九十]{1,3})\s*(?P<close>[)）])"
)


def _normalize_cjk_ordinals(text: str) -> str:
    """Turn Chinese ordinal markers into their Arabic digit value, keeping the marker.

    Only applies to enumeration markers (a numeral followed by ``、``/``．``,
    inside ``()``/``（）``, or after ``第``).  An unrecognisable numeral (containing
    ``百``/``千``) or one outside the ordinal range is left untouched, so prose
    amounts in words are never rewritten.  The surrounding punctuation is kept
    (``一、``→``1、``, ``（四）``→``(4)``, ``第二节``→``2节``) so the substituted digits
    can never merge with an adjacent figure.
    """

    def repl(m: re.Match[str]) -> str:
        num = m.group("num_ch") or m.group("num_sep") or m.group("num_paren")
        val = _cn_to_int(num) if num else None
        if val is None or not 1 <= val <= 99:
            return m.group(0)
        if m.group("suffix"):
            return f"{val}{m.group('suffix')}"
        if m.group("sep"):
            return f"{val}{m.group('sep')}"
        return f"({val})"

    return _CN_ORD_PATTERN.sub(repl, text)


def _amounts(text: str) -> tuple[Counter[str], Counter[Decimal]]:
    """The amount-shaped numbers of a page as ``(role keys, value keys)``.

    Two views of the same tokens, because the two failure modes need different
    evidence:

    * **role keys** keep the digits *and* the role of each separator (``,`` → C,
      ``.`` → D): ``3,702,726,474.45`` becomes ``3C702C726C474D45``, so a
      comma/dot swap (``3,702.726,474.45``, same digits) is caught even though the
      values would look alike, and a dropped ``0.`` still shows up.
    * **value keys** are the numeric values, with unit multipliers applied
      (``1,234.56 万元`` and ``12,345,600 yuan`` are both ``12345600``).  Without
      this, every unit-converted figure — the normal case for a CJK→Latin annual
      report — was reported as a fatal number mismatch.

    A bare integer (``1``, ``2025``, ``22``) is usually a section/ordinal marker, a
    year or a ``Tier 1`` term, not a figure a misread separator could corrupt, so
    it only counts when a unit word attaches to it (``314 million``).
    """
    roles: Counter[str] = Counter()
    values: Counter[Decimal] = Counter()
    t = str(text).translate(_FULLWIDTH_MAP).translate(_MINUS_MAP)
    spans = list(_NUM_TOKEN_RE.finditer(t))
    for i, m in enumerate(spans):
        tok = m.group(0)
        if tok in ("", "-", "+", "."):
            continue
        unit_win = (t[m.end():spans[i + 1].start()] if i + 1 < len(spans)
                    else t[m.end():m.end() + 16])
        mult = _unit_multiplier(unit_win)
        currency = bool(_CURRENCY_RE.match(unit_win))
        has_sep = any(ch in ",.%" for ch in tok)
        # A leading parenthesis is an accounting negative (``（1,234.56）``): the
        # sign lives outside the token, so without this a lost negative compared
        # equal to the positive value (and ``%`` was dropped for the same reason —
        # ``Decimal("92.5%")`` raises and the token got *no* value at all, so a
        # percentage that changed by 10× was reported as "consistent").
        paren_neg = m.start() > 0 and t[m.start() - 1] in "(（"
        if not (has_sep or mult != 1 or currency):
            continue                      # bare integer without a unit: not a figure
        pct = tok.endswith("%")
        canonical: list[str] = []
        neg = paren_neg
        for ch in tok:
            if ch.isdigit():
                canonical.append(ch)
            elif ch == ".":
                canonical.append("D")
            elif ch == ",":
                canonical.append("C")
            elif ch == "-":
                neg = True
        if canonical:
            roles[("M" if neg else "") + "".join(canonical)] += 1
        number = tok[:-1] if pct else tok
        try:
            value = abs(Decimal(number.replace(",", ""))) * mult
        except InvalidOperation:
            # A separator-swapped token (``3,702.726,474.45`` → ``3702.726474.45``)
            # has no value: it stays unmatched at the value level too, so the swap
            # is still reported (and never silently "explained away").
            continue
        # ``neg`` already covers the token's own ``-`` *and* an accounting
        # parenthesis, so apply it once to the absolute value.
        values[-value if neg else value] += 1
    return roles, values


def _numeric_diff(src_text: str, tgt_text: str) -> list[str]:
    """Human-readable differences between the two pages' amounts (empty = clean).

    The comparison is **by value**: ``3.14 亿元`` and ``314 million yuan`` are the
    same figure, ``1,234.56`` and ``1234.56`` differ only in style, and a
    full-width ``１，２３４．５６`` is the same number — none of those is a defect.
    A dropped/altered/invented digit, a lost unit multiplier (``1,234.56 万元`` vs
    ``1,234.56 yuan``), a lost sign (``（1,234.56）`` vs ``1,234.56``) or a
    separator swap (which leaves the token unparsable) changes the value and is
    reported.

    The separator-role view is deliberately *not* a gate any more: the roles of
    ``1,234.56 万元`` and ``1,234.56 yuan`` are identical (the unit is not part of
    the token), so the old "roles equal → clean" short-circuit skipped the value
    check and reported a translation that dropped the 万 multiplier as consistent.
    """
    _src_roles, src_vals = _amounts(src_text)
    _tgt_roles, tgt_vals = _amounts(tgt_text)
    only_src_vals = src_vals - tgt_vals
    only_tgt_vals = tgt_vals - src_vals
    if not only_src_vals and not only_tgt_vals:
        return []                        # same values, different surface form
    diffs: list[str] = []
    for value, n in only_src_vals.most_common():
        diffs.append(f"原文 {n} 次「{value}」在译文中缺失")
    for value, n in only_tgt_vals.most_common():
        diffs.append(f"译文多出 {n} 次「{value}」（不在原文中出现）")
    return diffs


def _page_numbers(text: str) -> Counter[str]:
    """Amount-shaped tokens of a page as separator-role keys (see :func:`_amounts`)."""
    return _amounts(text)[0]


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
            continue
        m = _CN_ORD_HEADING_RE.match(line)
        if m:
            num = m.group(1) or m.group(2)
            value = _cn_to_int(num)
            if value is not None:
                found.append((num, value))
                continue
        m = _PAREN_HEADING_RE.match(line)
        if m:
            found.append((m.group(1), int(m.group(1))))
            continue
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
    # A value that literally contains CJK chars, OR a language NAME denoting a CJK
    # script (``Simplified Chinese`` is ASCII but its script is CJK).  Without the
    # name branch, a Chinese-target run was treated as Western and every intended
    # Chinese translation was falsely reported as "残留中文".
    t = str(text or "").strip().lower()
    if any("一" <= c <= "鿿" for c in t):
        return True
    # Short language *codes* are matched exactly (``zh`` must not match inside
    # another word); ``--lang zh`` / ``cn`` used to be treated as a Western
    # target, so a Chinese translation was falsely reported as residual CJK.
    code = re.sub(r"[^a-z]", "", t)
    if code in {"zh", "cn", "zho", "chi", "zhcn", "zhhans", "zhtw",
                "zhhant", "ja", "jp", "jpn", "ko", "kor"}:
        return True
    return any(k in t for k in (
        "chinese", "中文", "汉语", "普通话", "简体", "繁体",
        "japanese", "日语", "korean", "韩语", "한국어", "日本語",
    ))


def _is_scan_like_text(text: str) -> bool:
    """True when a page's text layer carries no real content.

    A scanned statement often leaves only a page number (``22``) in the text
    layer; that is still a scan, and its figures live in the raster image so the
    digit check has nothing to stand on.  A page with any letter-like character
    (Latin *or* CJK — ``str.isalpha`` covers both) is real content and stays.

    A bare digit page is a scan only when it holds no *amount-shaped* number
    (a thousands/decimal separator or ``%``): a sparse but genuine text page can
    contain just a figure such as ``1,234.56``, which must still be checked.
    """
    if not text:
        return True
    if any(c.isalpha() for c in text):
        return False
    for tok in _NUM_TOKEN_RE.findall(text):
        if any(ch in ",，.．%" for ch in tok):
            return False  # an amount, not a page number → real content
    return True


#: A short structured statement / subject code (``会商银02表``, ``会企01表-1``): mostly
#: CJK plus digits and separators.  These are deliberately not translated (codes
#: are exact identifiers), so their CJK must not be counted as residual prose.
_CODE_TOKEN_RE = re.compile(
    r"^[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9\-－（）()、\s]{0,15}$"
)
#: A run of code-ish characters (CJK + digits + separators) used to locate
#: candidate tokens inside a page's text.
_CODE_CAND_RE = re.compile(r"[\u4e00-\u9fff0-9A-Za-z\-－（）()、]+")


def _is_statement_code(text: str) -> bool:
    """True when ``text`` is a kept statement / subject code, not prose.

    Mirrors ``translate_app.pdfio._looks_like_code_token``: short (≤16), contains
    a digit, and no more than four CJK chars — so a document title with a year
    (``2025年年度报告``) or a long label is not mistaken for a code.
    """
    t = str(text).strip()
    if not (1 <= len(t) <= 16) or not any(c.isdigit() for c in t):
        return False
    cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    if cjk == 0 or cjk > 4:
        return False
    return bool(_CODE_TOKEN_RE.match(t))


def _cjk_residual(text: str) -> list[str]:
    """CJK chars in ``text`` that are not part of a kept statement code."""
    excluded: set[int] = set()
    for m in _CODE_CAND_RE.finditer(text):
        if _is_statement_code(m.group(0).strip()):
            excluded.update(range(m.start(), m.end()))
    return [ch for i, ch in enumerate(text) if "一" <= ch <= "鿿" and i not in excluded]


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
                # A bilingual product interleaves each source page with its
                # translation page (``src[i]`` ↔ ``tgt[2i+1]``).  Pairing
                # ``tgt[i]`` there compared a source page with the *next*
                # pair's source page, so the real translation page was never
                # looked at: a wrong figure in it passed as clean (measured —
                # a bilingual export whose translation page read ``999,999.99``
                # where the source said ``123,456.78`` reported no numeric
                # issue at all).
                interleaved = (src.page_count > 0
                               and tgt.page_count == 2 * src.page_count)
                # 扩页产物：一页源文可能占多个输出页，页序不再一一对应。导出器把
                # 「源页 → 首个输出页」映射写进 PDF 的 XMP（pdfio.document_page_map），
                # 这里优先用它；否则整篇错位配对（实测把源第 2 页配到第 1 页的续页，
                # 报出满屏假「数字不一致」）。
                page_map = pdfio.document_page_map(tgt)
                if page_map is None and not interleaved \
                        and tgt.page_count > src.page_count:
                    self.pages.append(
                        f"译文页数（{tgt.page_count}）多于原文（{src.page_count}）"
                        "且文件里没有扩页映射（XMP pageMap）——按页序配对可能错位，"
                        "结论仅供参考；请用「译文扩页」导出的原始产物重跑。"
                    )
                n = (src.page_count if interleaved
                     else min(src.page_count, tgt.page_count))
                tgt_styles: list[str] = []
                ranges = (pdfio.mapped_page_ranges(page_map, src.page_count,
                                                   tgt.page_count)
                          if page_map is not None else None)
                for i in range(n):
                    if i + 1 in self.skip:
                        continue
                    if ranges is not None:
                        tis = [t for t in ranges[i] if t < tgt.page_count]
                    else:
                        ti = 2 * i + 1 if interleaved else i
                        tis = [ti] if ti < tgt.page_count else []
                    if not tis:
                        continue
                    # 一页源文的续页一起参与比对：被搬到续页的表格行只在那里出现，
                    # 只比首个输出页会把它们误报成「缺失」。
                    tgt_text = "\n".join((tgt[t].get_text("text") or "") for t in tis)
                    style = self._check_page(src[i], tgt_text, i, skip_scan)
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

    def _check_residual(self, tgt_text: str, where: str) -> None:
        """Report CJK left in a Western-target translation (needs only text)."""
        if _has_cjk(self.lang):
            return
        residual = _cjk_residual(tgt_text)
        if residual:
            self.cjk.append(
                f"{where}: 残留 {len(residual)} 个中文字符"
                f"（如 {''.join(residual[:8])}…）：{_snippet(tgt_text)}"
            )

    def _check_page(
        self, src_page: fitz.Page, tgt_text: str, i: int, skip_scan: bool
    ) -> str:
        where = f"第 {i + 1} 页"
        src_text = src_page.get_text("text") or ""
        if skip_scan and _is_scan_like_text(src_text):
            # 扫描页（文本层仅页码/无内容）：数字来自 OCR / 重排，不能作为基准。
            # 但「残留中文」只需要译文文本 —— 整页早退曾把它一起跳过（真机样例
            # 译文第 2 页 750 个残留汉字未被报出、exit 0）。
            self._check_residual(tgt_text, where)
            return "mixed"

        # 1) 数字一致性：按「值」比较（单位倍率、全角数字、千分位风格差异不算错），
        #    并保留分隔符角色比较以抓千分位/小数点错乱。先归一化中文序数（一、→1.、
        #    二、→2.、（四）→(4)、第X节→Section X），否则这些预期转换会被当作
        #    “译文多出的数字”。
        diffs = _numeric_diff(_normalize_cjk_ordinals(src_text),
                              _normalize_cjk_ordinals(tgt_text))
        if diffs:
            self.numeric.append(
                f"{where}: 数字序列不一致：" + "；".join(diffs[:self.max_reported])
            )

        # 2) 残留中文（仅西文目标）。报表/科目号（会商银02表、会企01表-1）按规则保留
        #    不翻译，对其 CJK 不报残留（它们是精确标识，不是漏译）。
        self._check_residual(tgt_text, where)

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


def _parse_page_spec(text: str) -> set[int] | None:
    """Parse ``"24-27,30"`` into ``{24,...,27,30}``; ``None`` = a usage error.

    A bad value used to raise ``SystemExit("字符串")``, whose exit code is **1** —
    the same code as "数字不一致" — while the script's own docs (and
    ``check_layout.py``) reserve 2 for usage errors.  An empty / inverted range
    (``5-1``) also silently skipped nothing, so a typo looked like a clean run.
    """
    pages: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                if hi < lo:
                    return None
                pages.update(range(lo, hi + 1))
            else:
                pages.add(int(part))
        except ValueError:
            return None
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
    if skip is None:
        print(f"--skip 参数无法解析：{skip_text!r}（示例：--skip 24-27,30）",
              file=sys.stderr)
        return 2
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
