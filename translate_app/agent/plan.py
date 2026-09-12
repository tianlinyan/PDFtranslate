"""Document-level translation plan (v0.6.6 M1, opt-in).

One model call *before* the per-page loop decides the things the deterministic tools
cannot express but that must stay consistent across the whole document:

* extra / overriding **terminology** (``glossary``);
* the document's **conventions** — register, forms of address, units, numbering, how
  personal names are written (``style``);
* which of the ambiguous blocks **keep their source** (``keep``);
* which **normal** pages may take the cheap deterministic batch pass instead of the
  per-page agent loop (``page_strategy``, M2) — every such page still has to pass
  the deterministic audit gate afterwards, and falls back to the agent when it does
  not.

Everything a deterministic tool already knows (page count, page kinds, language, block
count) stays in :func:`pdfio.get_doc_info` — a plan never re-derives a fact.

Hard rules, all enforced here so no caller has to remember them:

* **validated against the document**: a glossary key that does not occur in the source
  text, an out-of-range block index, an over-long style — each is dropped **and
  reported** in :attr:`TranslationPlan.dropped`.  A dropped item must never look like
  a plan that was followed.
* **a default, not a lock**: the plan only ever *adds* a keep, never clears one, so it
  cannot widen a ``page_scope`` or undo the content policy.
* **fail-open**: no client, a network error, bad JSON or nothing valid left yields an
  empty plan, so the run behaves exactly as if the feature were off.

Design: ``docs/0.6.6-文档级翻译方案设计.md``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

#: Block excerpts the plan prompt may carry, per page.
_PLAN_BLOCKS_PER_PAGE = 3
#: Hard character budget for the assembled prompt body (a long report is never pasted
#: whole: the plan needs the *shape* of the document, not its full text).
_PLAN_INPUT_BUDGET = 8000
#: Caps that keep a malformed or hostile reply from bloating the run.
_PLAN_MAX_GLOSSARY = 40
#: Accepted ``page_strategy`` values (M2).  ``agent`` is the default behaviour,
#: ``batch`` is the one that actually saves calls; anything else is dropped.
_PAGE_STRATEGIES = ("agent", "batch")
_PLAN_MAX_KEEP = 200
_PLAN_MAX_STYLE = 300
_PLAN_MAX_TERM = 80
_PLAN_MAX_TARGET = 120
#: Below this many pages a document-level pass costs more than it can save.
PLAN_MIN_PAGES = 3
#: Reply budget for the one plan call.  Measured on a real local reasoning model
#: (qwen3.8-27b, ``reasoning_effort=low``): its thinking is billed against the same
#: budget and grows with the document — a 5-page sample needed ~1064 tokens, while a
#: 28-page report and a 51-page report both hit a 3072 cap and truncated the JSON,
#: which makes every plan degrade to "no plan" (a silent no-op feature).  The prompt
#: also bounds the answer (see ``document_plan_task``); this is the safety net.
_PLAN_MAX_TOKENS = 8192


@dataclass
class TranslationPlan:
    """A validated, document-bound plan (empty when nothing usable came back)."""

    glossary: dict[str, str] = field(default_factory=dict)
    style: str = ""
    keep: set[int] = field(default_factory=set)
    #: M2 only (``batch`` / ``agent`` / ...): parsed by the design, not used yet —
    #: M1 deliberately leaves the page schedule to the phase machine.
    page_strategy: dict[int, str] = field(default_factory=dict)
    notes: str = ""
    #: Human-readable reasons an item from the reply was NOT used.
    dropped: list[str] = field(default_factory=list)
    #: ``"llm"`` when the plan carries anything usable, else ``""``.
    source: str = ""

    def has_content(self) -> bool:
        """True when at least one channel would actually change the run."""
        return bool(self.glossary or self.style or self.keep or self.page_strategy)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"glossary": dict(self.glossary), "style": self.style,
                               "keep": sorted(self.keep), "notes": self.notes}
        if self.page_strategy:
            out["page_strategy"] = {str(p): v for p, v in sorted(
                self.page_strategy.items())}
        if self.dropped:
            out["dropped"] = list(self.dropped)
        return out

    def style_lines(self) -> list[str]:
        """The convention as ``state.requirements`` entries (the channel the agent reads)."""
        text = str(self.style or "").strip()
        return [f"文档级翻译约定：{text}"] if text else []


def parse_plan_json(text: str) -> dict:
    """Extract the JSON object from a model reply (fences / surrounding prose dropped).

    The same tolerant idiom as ``policy.parse_policy_json``: a malformed reply degrades
    to ``{}`` — i.e. no plan — never to a half-parsed one.
    """
    m = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except Exception:          # noqa: BLE001 — bad JSON -> no plan
        return {}
    return data if isinstance(data, dict) else {}


def document_summary(doc, doc_info, triage=None, *, terms=(), requirements=(),
                     budget: int = _PLAN_INPUT_BUDGET) -> str:
    """The bounded input the plan prompt sees.

    Not the document: an annual report cannot be pasted into one call.  A
    document-level decision needs the *shape* (counts and page kinds), the candidate
    terms and a few block excerpts per page — enough to see the register of the prose
    without reading all of it.  Special pages (scan / chart / uncertain) come first:
    they are the ones whose treatment can differ page to page.
    """
    lines: list[str] = []
    pages = list(getattr(doc, "pages", None) or [])
    n_blocks = sum(len(p) for p in pages)
    lines.append(f"页数={len(pages)} 语言={getattr(doc_info, 'language', '') or 'unknown'} "
                 f"块数={n_blocks}")
    lines.append(f"正常={getattr(doc_info, 'text_pages', 0)} "
                 f"扫描={getattr(doc_info, 'scan_pages', 0)} "
                 f"图表={getattr(doc_info, 'chart_pages', 0)} "
                 f"待确认={getattr(doc_info, 'uncertain_pages', 0)}")
    if requirements:
        lines.append("用户要求=" + "；".join(str(r) for r in requirements))
    if terms:
        lines.append("候选术语=" + "、".join(str(t) for t in terms))
    kinds = list(getattr(doc_info, "kinds", None) or [])
    order = sorted(range(len(pages)),
                   key=lambda i: (0 if (i < len(kinds) and kinds[i] != "normal") else 1, i))
    for i in order:
        kind = kinds[i] if i < len(kinds) else "normal"
        blocks = pages[i]
        excerpts = [" ".join(str(getattr(b, "text", "")).split()) for b in blocks]
        excerpts = [e for e in excerpts if e][:_PLAN_BLOCKS_PER_PAGE]
        head = f"第{i + 1}页[{kind}] {len(blocks)}块"
        if excerpts:
            head += "：" + " / ".join(e[:60] for e in excerpts)
        lines.append(head)
    text = "\n".join(lines)
    if len(text) > budget:
        text = text[:budget] + "\n…（已截断）"
    return text


def validate_plan(raw: dict | None, *, n_blocks: int, source_text: str = "",
                  n_pages: int = 0, kinds=None) -> TranslationPlan:
    """Bind a model reply to the document; drop (and report) everything unusable."""
    plan = TranslationPlan()
    data = raw if isinstance(raw, dict) else {}
    hay = str(source_text or "")

    glossary = data.get("glossary")
    if isinstance(glossary, dict):
        for key, value in glossary.items():
            k, v = str(key).strip(), str(value).strip()
            if not k or not v:
                plan.dropped.append(f"glossary:{k or '(空)'}(键或值为空)")
                continue
            if len(k) > _PLAN_MAX_TERM or len(v) > _PLAN_MAX_TARGET:
                plan.dropped.append(f"glossary:{k[:20]}(过长)")
                continue
            if hay and k not in hay:
                # A term the document does not contain is a hallucination: pinning it
                # would only add an entry no block can ever match.
                plan.dropped.append(f"glossary:{k[:20]}(原文中不存在)")
                continue
            if len(plan.glossary) >= _PLAN_MAX_GLOSSARY:
                plan.dropped.append("glossary:超出条数上限")
                break
            plan.glossary[k] = v
    elif glossary is not None:
        plan.dropped.append("glossary(不是对象)")

    style = data.get("style")
    if style:
        text = re.sub(r"\s+", " ", str(style)).strip()
        if len(text) > _PLAN_MAX_STYLE:
            plan.dropped.append("style(过长，已截断)")
            text = text[:_PLAN_MAX_STYLE]
        plan.style = text

    keep = data.get("keep")
    if isinstance(keep, (list, tuple, set)):
        for raw_i in keep:
            try:
                i = int(raw_i)
            except (TypeError, ValueError):
                plan.dropped.append(f"keep:{raw_i!r}(不是整数)")
                continue
            if not (0 <= i < int(n_blocks)):
                plan.dropped.append(f"keep:{i}(越界)")
                continue
            if len(plan.keep) >= _PLAN_MAX_KEEP:
                plan.dropped.append("keep:超出条数上限")
                break
            plan.keep.add(i)
    elif keep is not None:
        plan.dropped.append("keep(不是数组)")
    strategy = data.get("page_strategy")
    if isinstance(strategy, dict):
        kind_list = list(kinds or [])
        for raw_p, raw_v in strategy.items():
            value = str(raw_v).strip().lower()
            try:
                p = int(raw_p)
            except (TypeError, ValueError):
                plan.dropped.append(f"page_strategy:{raw_p!r}(不是页号)")
                continue
            if not (0 <= p < int(n_pages)):
                plan.dropped.append(f"page_strategy:{p}(越界)")
                continue
            if value not in _PAGE_STRATEGIES:
                plan.dropped.append(f"page_strategy:{p}={raw_v!r}(未知策略)")
                continue
            kind = kind_list[p] if p < len(kind_list) else "normal"
            if value == "batch" and kind != "normal":
                # A scan / chart / uncertain page needs the visual loop: a batch
                # pass cannot look at the page, and the pixels of a scan are not a
                # text decision.
                plan.dropped.append(f"page_strategy:{p}=batch({kind} 页不支持)")
                continue
            plan.page_strategy[p] = value
    elif strategy is not None:
        plan.dropped.append("page_strategy(不是对象)")
    notes = data.get("notes")
    if notes:
        plan.notes = re.sub(r"\s+", " ", str(notes)).strip()[:200]

    if plan.has_content():
        plan.source = "llm"
    return plan


def make_llm_plan(model, client: Any = None,
                  log: Callable[[str], None] | None = None):
    """Return the plan callback, or ``None`` when no client is usable.

    The returned ``fn(summary, *, terms, requirements, lang) -> dict`` performs the one
    document-level request and returns the *parsed* JSON (``{}`` on any failure).
    Binding it to the document happens in :func:`validate_plan`, so this layer owns
    only the request and the fail-open.
    """
    from .. import prompts
    from .. import translator as _tr

    if client is None:
        try:
            client = _tr.OpenAI(**model.client_kwargs())
        except Exception:      # noqa: BLE001 — no client -> no plan
            return None

    def decide(summary: str, *, terms=(), requirements=(), lang: str = "") -> dict:
        try:
            kwargs: dict[str, Any] = {
                "model": model.model,
                "temperature": 0.0,
                "max_tokens": _PLAN_MAX_TOKENS,
                "messages": [{
                    "role": "user",
                    "content": prompts.document_plan_task(
                        summary, terms, requirements, lang),
                }],
            }
            body = model.request_params()
            if body:
                kwargs["extra_body"] = body
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            text = (getattr(choice.message, "content", "") or "").strip()
            if str(getattr(choice, "finish_reason", "") or "") == "length" and log:
                # A truncated reply is a *budget* failure, not "the model chose to
                # say nothing": say which one it was, or the log implies the feature
                # ran and had nothing to do.
                log(f"  文档级方案：回复被 max_tokens（{_PLAN_MAX_TOKENS}）截断，"
                    f"本次不生成（可减少页数或提高该值）。")
            return parse_plan_json(text)
        except Exception as exc:   # noqa: BLE001 — fail-open to "no plan"
            if log:
                log(f"  文档级方案：请求失败（{type(exc).__name__}: {exc}），本次不生成。")
            return {}

    return decide


def plan_summary(plan: TranslationPlan) -> str:
    """One log line stating what the plan will *do* (never what it claimed)."""
    parts = []
    if plan.glossary:
        parts.append(f"术语 {len(plan.glossary)} 条")
    if plan.style:
        parts.append(f"约定 {len(plan.style)} 字")
    if plan.keep:
        parts.append(f"保留 {len(plan.keep)} 块")
    batch = [p for p, v in plan.page_strategy.items() if v == "batch"]
    if batch:
        parts.append(f"批量页 {len(batch)} 页")
    body = "、".join(parts) or "无有效内容"
    drop = f"（已丢弃 {len(plan.dropped)} 条无法校验的项）" if plan.dropped else ""
    return f"文档级方案：{body}{drop}。"
