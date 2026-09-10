"""AI content policy: which blocks keep their source text (IR / direct paths).

The pipeline's *hard* rules are not decisions and are never offered to the model: a
numeric / amount block, a formula, a table cell and a pure marker never leave the
pipeline (see ``ir.translate_ir``).  What *is* a decision — and what the user asked to
have decided by content rather than by a geometric rule — is:

* a ``figure`` region the structure backend detected (a chart / diagram): kept by
  default, because redrawing its narrow labels overlaps and shrinks them;
* a scanned (OCR) block: translated by default, but its pixels may *be* the content
  (a seal, a stamp, a handwritten signature, a barcode).

This module asks the model about exactly those blocks and nothing else, and applies
the answer through the one channel every other layer already reads
(``Block.keep_original`` for keeps, the IR release set for releases).

Failure always degrades to "no decision" = the deterministic default.  That
direction matters: a keep the model failed to make costs a translation the user can
see is missing, while a *wrong* release erases content with no trace — so an absent
model must never move the dial.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from . import pdfio, prompts

#: Decision values (the JSON keys are ``release`` / ``keep``; a decision is stored
#: keyed by flat block index).
KEEP = "keep"
TRANSLATE = "translate"

#: How many ambiguous blocks are offered per page / in total.  The list is a
#: **sample**, not the document: whatever is not listed keeps its deterministic
#: default, so a longer prompt only spends tokens without widening the decision.
_MAX_PER_PAGE = 6
_MAX_CANDIDATES = 120


@dataclass(frozen=True)
class Candidate:
    """One block whose keep/translate state is genuinely ambiguous."""

    index: int            #: flat block index (the id the model replies with)
    page: int             #: 0-based page
    kind: str             #: "figure" (default kept) | "ocr" (default translated)
    default_keep: bool    #: the deterministic default the model may override
    text: str


def _verbatim(block) -> bool:
    """True when the block never leaves the pipeline (numeric / no letters).

    Imported lazily from the engine so the *same* predicate decides here and in
    ``translate_blocks`` — a second, drifting copy would offer the model blocks it
    must never see.
    """
    from .translator import _needs_translation

    return (not _needs_translation(str(getattr(block, "text", "")))
            or pdfio._is_numeric_cell(str(getattr(block, "text", ""))))


def candidates(doc, *, max_per_page: int = _MAX_PER_PAGE,
               max_blocks: int = _MAX_CANDIDATES) -> list[Candidate]:
    """The blocks the model may decide about, in document order.

    Two kinds only:

    * ``figure`` — the structure backend put this block in a figure region and it is
      not text OCR'd out of an *image on a text-layer page* (``in_image``, which is
      FigureText: it must be translated, so it is not ambiguous at all);
    * ``ocr`` — a scanned block that is not already kept and not a table cell (a
      cell's geometry is pinned by the reconstructed grid; its pixels are not a
      free choice).

    Formulas, numbers and table cells are excluded here, which is what makes the red
    lines unreachable from the model's answer.  At most ``max_per_page`` per page and
    ``max_blocks`` in total are returned, so the prompt stays bounded on a long
    scanned report.
    """
    from . import ir as ir_mod

    pages = list(getattr(doc, "pages", None) or [])
    structure = list(getattr(doc, "page_structure", None) or [])
    out: list[Candidate] = []
    offset = 0
    for p, blocks in enumerate(pages):
        st = structure[p] if p < len(structure) else None
        page_cands: list[Candidate] = []
        for i, b in enumerate(blocks):
            if len(page_cands) >= max_per_page:
                break
            if getattr(b, "is_chart", False) or _verbatim(b):
                continue
            if bool(getattr(b, "in_table", False)):
                continue
            role = ir_mod._role_of(offset + i, st)[0] if st is not None else "text"
            in_image = bool(getattr(b, "in_image", False))
            if role == "figure" and not in_image:
                kind, default_keep = "figure", True
            elif bool(getattr(b, "ocr", False)) and not getattr(b, "keep_original", False):
                kind, default_keep = "ocr", False
            else:
                continue
            page_cands.append(Candidate(index=offset + i, page=p, kind=kind,
                                        default_keep=default_keep,
                                        text=str(getattr(b, "text", ""))))
        out.extend(page_cands)
        offset += len(blocks)
        if len(out) >= max_blocks:
            break
    return out[:max_blocks]


def policy_entries(cands: Sequence[Candidate]) -> list[tuple]:
    """``(index, 1-based page, kind, default_keep, text)`` tuples for the prompt."""
    return [(c.index, c.page + 1, c.kind, c.default_keep, c.text) for c in cands]


def parse_policy_json(text: str, *, allowed: set[int] | None = None
                      ) -> tuple[dict[int, str], str]:
    """Read a policy decision out of a model reply.

    Fail-closed to ``({}, "")`` on anything malformed (prose, no JSON, a JSON
    non-object).  Indices are validated: a boolean, a non-integral float, a
    non-number and — when ``allowed`` is given — any index outside the offered
    candidate set is dropped, so a hallucinated index can never mark a block.
    """
    match = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
    if not match:
        return {}, ""
    try:
        data = json.loads(match.group(0))
    except Exception:  # noqa: BLE001 — bad JSON = no decision
        return {}, ""
    if not isinstance(data, dict):
        return {}, ""
    out: dict[int, str] = {}
    for key, value in (("release", TRANSLATE), ("keep", KEEP)):
        raw = data.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, bool):
                continue
            if isinstance(item, float) and not item.is_integer():
                continue
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if allowed is not None and idx not in allowed:
                continue
            out[idx] = value
    reason = str(data.get("reason") or "").strip()
    return out, reason


def make_llm_policy_fn(model, client: Any = None,
                       log: Callable[[str], None] | None = None):
    """Return an AI content-policy callback, or ``None`` when no client is usable.

    The returned ``fn(candidates, *, lang, requirement)`` answers
    ``(decision, reason)`` — the same shape ``user_flows.ai_scope`` uses.  Any
    failure (no client, network, malformed reply) yields ``({}, "")`` and a log
    line, so the caller keeps the deterministic defaults.
    """
    from . import translator as _tr

    if client is None:
        try:
            client = _tr.OpenAI(**model.client_kwargs())
        except Exception:  # noqa: BLE001 — no client → no AI policy
            return None

    def decide(cands: Sequence[Candidate], *, lang: str, requirement: str = ""
               ) -> tuple[dict[int, str], str]:
        if not cands:
            return {}, ""
        try:
            kwargs: dict[str, Any] = {
                "model": model.model,
                "temperature": 0.0,
                "max_tokens": 512,
                "messages": [{
                    "role": "user",
                    "content": prompts.content_policy_task(
                        policy_entries(cands), lang, requirement),
                }],
            }
            body = model.request_params()
            if body:
                kwargs["extra_body"] = body
            resp = client.chat.completions.create(**kwargs)
            text = (getattr(resp.choices[0].message, "content", "") or "").strip()
            return parse_policy_json(text, allowed={c.index for c in cands})
        except Exception as exc:  # noqa: BLE001 — fail-open to the default policy
            if log:
                log(f"  内容策略判定失败：{type(exc).__name__}: {exc}（沿用默认策略）。")
            return {}, ""

    return decide


def apply_policy(doc, decision: dict[int, str],
                 log: Callable[[str], None] | None = None) -> dict:
    """Write a decision onto the document's blocks and report what it did.

    Returns ``{"kept": n, "released": [indices], "skipped": n}``.  A ``keep`` sets
    ``Block.keep_original`` — the channel the exporter (no cover / no redraw), the
    audit (``_audit_protected``) and the IR prose grouper all read, so one write
    reaches every consumer.  A ``release`` is **not** written on the block (the block
    has no "kept by a rule" flag to clear) but handed back for
    ``ir.translate_ir(release=...)``; ``skipped`` counts decisions naming an index the
    document does not have.
    """
    flat = [b for pg in (getattr(doc, "pages", None) or []) for b in pg]
    if not flat:
        flat = list(getattr(doc, "blocks", None) or [])
    kept = 0
    released: list[int] = []
    skipped = 0
    for index, action in sorted(decision.items()):
        if not (0 <= int(index) < len(flat)):
            skipped += 1
            continue
        block = flat[int(index)]
        if action == KEEP:
            if not getattr(block, "keep_original", False):
                kept += 1
            block.keep_original = True
        elif action == TRANSLATE:
            released.append(int(index))
    if skipped and log:
        log(f"  内容策略：{skipped} 条决策的块索引不在文档内，已忽略。")
    return {"kept": kept, "released": released, "skipped": skipped}
