"""P2: the ``FlowAgent`` orchestration skeleton.

A bounded **observe → plan → call → read-result → verify → recurse** loop that
drives a ``WorkflowState`` through the deterministic tool layer.  The LLM is
plugged in as a ``decide(observation, state) -> Decision`` callback, so a test can
script a fake model returning a fixed sequence of ``tool_calls`` and assert the
call order and the state changes.

The loop is conservative and fail-closed:

* an unknown tool or a tool that raises returns ``ok=False`` (never crashes);
* ``ask`` pauses the loop and records a question for the UI;
* the loop stops on ``done`` / ``ask`` / budget exhausted / a round cap.

See ``docs/0.3.0-设计.md`` §4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable
import re
from collections import Counter
import threading
import time

from .state import (
    DocInfo,
    Goal,
    PageTriage,
    PHASE_COMPLETED,
    PHASE_DONE,
    PHASE_PREPROCESS,
    PHASE_REVIEW,
    PHASE_SPECIAL_PAGES,
    PHASE_TRANSLATE_NORMAL,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_NEEDS_USER,
    WorkflowState,
)
from .tools import agent_openai_tools


@dataclass
class Decision:
    """One controller step: call a tool, ask the user, or finish."""

    action: str                       # "call" | "done" | "ask"
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    question: str = ""
    options: list[str] = field(default_factory=list)
    target: str = ""
    summary: str = ""


@dataclass
class AgentResult:
    """The structured outcome of one tool call (the "read result" step)."""

    ok: bool
    op_tool: str = ""
    op_args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str = ""


class FlowAgent:
    """Drive a bounded agent loop over a ``WorkflowState``.

    ``tools`` is ``{name: callable(**args) -> Any}`` (the deterministic hands).
    ``decide`` is ``(observation: str, state) -> Decision`` — the LLM (or, in
    tests, a scripted fake).  ``observe`` feeds the previous tool result back so
    the controller can verify an action before recursing.
    """

    def __init__(self, state: WorkflowState, tools: dict[str, Callable],
                 decide: Callable[[str, WorkflowState, Any], Decision], *,
                 log: Callable[[str], None] | None = None,
                 max_steps: int | None = None,
                 answer_handler: Callable[[str, list[str], str], Any] | None = None,
                 cancel_fn: Callable[[], bool] | None = None) -> None:
        self.state = state
        self.tools = tools
        self.decide = decide
        self.log = log or (lambda m: None)
        if max_steps is not None:
            state.budget.max_steps = max_steps
        self.last_result: AgentResult | None = None
        #: Optional worker↔GUI channel: ``answer_handler(question, options, target)``
        #: blocks until the user answers; the answer is stored and the loop resumes.
        self.answer_handler = answer_handler
        #: Optional cancel polled at the top of each loop iteration so a user
        #: cancellation stops the agent promptly instead of only between pages
        #: (the worker raises ``TranslationCancelled`` once the page returns).
        self.cancel_fn = cancel_fn

    def observe(self) -> str:
        """Serialize the current state + the last result for the controller."""
        s = self.state
        lines = [f"src={s.src_path}", f"lang={s.lang}"]
        if s.requirements:
            lines.append("requirements=" + "; ".join(s.requirements))
        if s.todo:
            lines.append("todo=" + ",".join(
                g.kind + (f"@p{g.page}" if g.page is not None else "") for g in s.todo))
        else:
            lines.append("todo=")
        for p in s.pages:
            lines.append(f"page{p.index}[{p.status}] keep={sorted(p.keep_indices)} "
                         f"translated={len(p.translated)} issues={p.issues}")
        if self.last_result is not None:
            r = self.last_result
            # Only report tool + success (and any error) here — the full result is
            # already in the ``tool`` message, so re-dumping it (or any image bytes)
            # into the observation would bloat the context each step.
            line = f"last={r.op_tool} ok={r.ok}"
            if r.error:
                line += f" error={r.error}"
            lines.append(line)
        return "\n".join(lines)

    def _call(self, name: str, args: dict) -> AgentResult:
        fn = self.tools.get(name)
        if fn is None:
            return AgentResult(ok=False, op_tool=name, op_args=args,
                               error=f"unknown tool: {name}")
        try:
            return AgentResult(ok=True, op_tool=name, op_args=args, result=fn(**args))
        except _tr.TranslationCancelled:
            raise                     # a user cancel must propagate, not become ok=False
        except Exception as exc:  # noqa: BLE001 — fail-closed, never crash the loop
            return AgentResult(ok=False, op_tool=name, op_args=args,
                               error=f"{type(exc).__name__}: {exc}")

    def _apply(self, dec: Decision, res: AgentResult, *, elapsed: float = 0.0) -> None:
        self.state.record_op(dec.tool, args=dec.arguments, reason=dec.summary or "agent")
        # Log the tool + outcome concisely; never dump the raw result (it may hold
        # a huge block list or image bytes) into the audit log.
        self.state.log.append(
            f"{dec.tool} -> ok={res.ok} {res.error or _summarize_result(res.result)}"
            f"（{elapsed:.2f}s）")
        self.log(
            f"  [{dec.tool}] ok={res.ok}{' ' + res.error if res.error else ''}"
            f"（用时 {elapsed:.2f}s）")

    def step(self) -> Decision:
        """Run one controller step and return the decision it made."""
        obs = self.observe()
        dec = self.decide(obs, self.state, self.last_result)
        if dec.action == "call":
            # Surface the tool call as soon as it starts (so blocking tools like a
            # slow local request are visible) and report the elapsed time on done.
            self.log(f"  工具开始：{dec.tool}")
            started = time.monotonic()
            res = self._call(dec.tool, dec.arguments)
            elapsed = time.monotonic() - started
            self.last_result = res
            self._apply(dec, res, elapsed=elapsed)
        elif dec.action == "ask":
            if self.answer_handler is not None:
                value = self.answer_handler(dec.question, dec.options, dec.target)
                self.state.user_decisions[dec.target] = value
                self.state.log.append(f"ask(answer): {dec.question} -> {value}")
                self.log(f"  ? {dec.question} => {value}")
            else:
                self.state.ask(dec.question, dec.options, target=dec.target)
                self.state.log.append(f"ask: {dec.question}")
                self.log(f"  ? {dec.question}")
        return dec

    def run(self, max_rounds: int | None = None) -> WorkflowState:
        """Loop until done / unanswered ask / budget exhausted / round cap."""
        i = 0
        while not self.state.budget.exhausted():
            if max_rounds is not None and i >= max_rounds:
                break
            if self.cancel_fn is not None and self.cancel_fn():
                break   # a user cancellation stops the loop; the caller raises
            dec = self.step()
            i += 1
            if dec.action == "done":
                break
            if dec.action == "ask" and self.answer_handler is None:
                break   # an unanswered ask pauses the agent at this question
        return self.state


def run_agent_run(
    state: WorkflowState,
    tools: dict[str, Callable] | None = None,
    decide: Callable[[str, WorkflowState, Any], Decision] | None = None,
    *,
    max_steps: int | None = None,
    max_rounds: int | None = None,
    answer_handler: Callable[[str, list[str], str], Any] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
    log: Callable[[str], None] | None = None,
) -> WorkflowState:
    """Convenience wrapper: build a ``FlowAgent`` and run it on ``state``.

    ``tools`` / ``decide`` are required for a real run (the LLM).  ``decide`` must
    not be ``None`` — the loop would otherwise crash on the first step asking the
    model to decide.
    """
    if decide is None:
        raise ValueError("run_agent_run requires a ``decide`` callable (the LLM).")
    agent = FlowAgent(state, tools or {}, decide, log=log, max_steps=max_steps,
                      answer_handler=answer_handler, cancel_fn=cancel_fn)
    return agent.run(max_rounds=max_rounds)


# --------------------------------------------------------------------------
# P3: real-LLM ``decide`` + a single-page visual closed loop.
# --------------------------------------------------------------------------

import base64
import json

from .. import prompts
from .. import translator as _tr
from .flow_steps import FlowCancelled, STANDARD_FLOWS, interpret_decision, run_flow


def _image_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _result_for_message(result: Any) -> Any:
    """A JSON-serialisable copy of a tool result, with image/bytes payloads stripped.

    Images are fed to the model *only* as ``image_url`` (see ``make_llm_decide``);
    dumping the raw PNG bytes into the text ``tool`` message would bloat the context
    enormously.  Recursively drops ``image`` keys and replaces any ``bytes`` value
    with a short placeholder.
    """
    if isinstance(result, dict):
        return {k: _result_for_message(v) for k, v in result.items()
                if k != "image" and not isinstance(v, (bytes, bytearray))}
    if isinstance(result, (list, tuple)):
        return [_result_for_message(v) for v in result]
    if isinstance(result, (bytes, bytearray)):
        return f"<bytes {len(result)}>"
    return result


def _summarize_result(result: Any, limit: int = 120) -> str:
    """A short textual summary of a tool result (for the ``WorkflowState`` log).

    Never serialises full blobs / images into the audit log — that is pure memory
    + log bloat.  Truncates to ``limit`` characters with a note when longer.
    """
    s = json.dumps(_result_for_message(result), ensure_ascii=False, default=str)
    if len(s) > limit:
        return s[:limit] + f"…<{len(s)} chars>"
    return s


#: The agent's interaction rules live in ``translate_app.prompts`` (``agent_interaction_rules``)
#: so the "该问就问" trigger is reviewable/tunable in one place.


def _render_source_page(src_path, page_index: int, dpi: int = 150) -> bytes:
    """Render one page of the source PDF to PNG (read-only, never modifies it)."""
    import pymupdf as fitz
    from .. import pdfio

    doc = fitz.open(str(src_path))
    try:
        if not (0 <= page_index < doc.page_count):
            return b""
        return pdfio._render_page_png(doc[page_index], dpi=dpi)
    finally:
        doc.close()


def _page_translation_counts(state: WorkflowState, page_index: int) -> tuple[int, int]:
    """Return ``(translatable, translated)`` for one page.

    ``translatable`` counts the page's blocks that *should* be translated (contain
    letters and are not a numeric/amount cell); ``translated`` counts how many of
    those have a non-empty AI translation in ``state.out_doc``.  A page with
    ``translatable > 0`` but ``translated == 0`` is the "the AI answered with prose
    / never called a translate tool" case — it must not be recorded ``STATUS_DONE``
    (that is a fail-open: the export would carry the untranslated source).
    """
    doc = getattr(state, "src_doc", None)
    pages = getattr(doc, "pages", None) if doc is not None else None
    if not pages or not (0 <= page_index < len(pages)):
        return 0, 0
    from ..translator import _needs_translation as _needs

    from .. import pdfio as _pdfio  # local import to avoid a load-time cycle
    page = pages[page_index]
    offset = sum(len(p) for p in pages[:page_index])
    out = state.out_doc or {}
    translatable = translated = 0
    for i, b in enumerate(page):
        text = str(getattr(b, "text", ""))
        if not _needs(text) or _pdfio._is_numeric_cell(text):
            continue
        translatable += 1
        entry = out.get(offset + i)
        if isinstance(entry, dict) and str(entry.get("text", "")).strip():
            translated += 1
    return translatable, translated


#: Max chat-completions messages kept per agent ``decide`` request.  The agent appends a
#: (user + assistant tool_call + tool result) triple per tool round; if every one were
#: re-sent the per-request payload would grow with the round count, so a grid-heavy page
#: would pay roughly O(rounds²) in total tokens.  Window to a bounded, protocol-valid
#: tail so the model keeps the recent context without re-reading all prior rounds.
_DECIDE_HISTORY_CAP = 60


def _window_messages(messages: list[dict], cap: int = _DECIDE_HISTORY_CAP) -> list[dict]:
    """Bound ``messages`` to a protocol-valid tail for one ``decide`` request.

    The conversation is ``[system, user, assistant(tool_calls), tool, user, ...]``.  Drop
    the OLDEST complete rounds, keeping ``system`` + the last ``cap`` messages, and back
    off so the cut starts at a ``user`` message — an assistant tool_call and its tool
    results are never split and no ``tool`` message is orphaned (OpenAI requires a tool
    message to reference a tool_call_id from an immediately preceding assistant message).
    The freshest observation and the current tool response are always retained.
    """
    if len(messages) <= cap:
        return messages
    keep = messages[:1]                       # the system prompt always stays
    tail = messages[1:]
    start = max(0, len(tail) - (cap - 1))
    while start > 0 and tail[start]["role"] != "user":
        start -= 1
    return keep + tail[start:]


def make_llm_decide(model, *, task: str, image_provider=None,
                    log: Callable[[str], None] | None = None,
                    max_tokens: int = 4096,
                    tool_names: Sequence[str] | None = None,
                    cancel: Callable[[], bool] | None = None):
    """Return a real-LLM ``decide(obs, state, last_result) -> Decision``.

    Wraps an OpenAI-compatible ``chat.completions`` call with the agent's tool
    schemas (``tool_choice="auto"``), a system prompt (``task``) and an optional
    page image (``image_provider(state) -> png``) so the model can *see* the
    source page.  Tool results are fed back as ``tool`` role messages so the loop
    is a genuine multi-turn tool-use conversation.  A model that stops making
    tool calls yields ``Decision(action="done")``.

    ``tool_names`` filters the advertised ``tools`` array to only the tools that
    are actually bound in the loop — a schema for a tool with no bound callable
    would make every call to it come back ``unknown tool``.

    ``cancel`` (optional) makes an in-flight request abortable: a watchdog thread
    closes the underlying HTTP client the moment ``cancel()`` turns true, so a
    slow llama.cpp generation does not freeze the worker thread until the client
    timeout.  On abort the call re-raises :class:`TranslationCancelled` (a
    control signal), exactly like the translate engine's own watchdog.
    """
    if not getattr(model, "vision", False):
        return None
    client = _tr.OpenAI(**model.client_kwargs())
    specs = agent_openai_tools(names=list(tool_names) if tool_names is not None else None)
    messages: list[dict] = [{"role": "system", "content": str(task) + prompts.agent_interaction_rules() + prompts.agent_tool_policy() + prompts.agent_workflow()}]
    sent_image = False
    pending_assistant: dict | None = None
    pending_call_id: str = ""

    def _decision(obs: str, state: WorkflowState, last_result: AgentResult | None = None) -> Decision:
        nonlocal sent_image, pending_assistant, pending_call_id
        if pending_assistant is not None:
            messages.append(pending_assistant)
            pending_assistant = None
            if last_result is not None:
                messages.append({
                    "role": "tool",
                    "tool_call_id": pending_call_id,
                    "content": json.dumps(
                        {"ok": last_result.ok,
                         "result": _result_for_message(last_result.result),
                         "error": last_result.error},
                        ensure_ascii=False, default=str),
                })
        # Build the user observation: text + (on the first call) the source page
        # image, and re-inject a *user-framed region* as a new visual observation
        # (so a "send this area to the AI" truly reaches the model as an image).
        content: list[dict] = [{"type": "text", "text": obs}]
        if not sent_image:
            if image_provider is not None:
                png = image_provider(state)
                if png:
                    content.append({"type": "image_url",
                                    "image_url": {"url": _image_url(png)}})
            sent_image = True
        # Any tool that returns an ``image`` (a preview page, or a user-drawn
        # annotation region) feeds it back as a fresh visual observation, so the
        # model sees the framed region before it edits the block.
        if last_result is not None and isinstance(last_result.result, dict):
            img = last_result.result.get("image")
            if img:
                content.append({"type": "image_url",
                                "image_url": {"url": _image_url(img)}})
        messages.append({"role": "user", "content": content})
        # A watchdog that closes the HTTP client the moment the user cancels, so an
        # in-flight request is aborted rather than blocking the worker thread until
        # the (300s) client timeout.  It mirrors the translate engine's own watchdog
        # and only runs while this request is outstanding; ``cancel_fn()`` is polled
        # merely between agent steps (see ``FlowAgent.run``), which cannot reach a
        # request already in flight.
        watchdog_stop = threading.Event()
        watchdog: threading.Thread | None = None
        if cancel is not None:
            def _watchdog() -> None:
                while not watchdog_stop.is_set():
                    if cancel():
                        try:
                            client.close()
                        except Exception:  # noqa: BLE001 — best effort on abort
                            pass
                        return
                    time.sleep(0.05)
            watchdog = threading.Thread(target=_watchdog, daemon=True)
            watchdog.start()
        try:
            # ``decide`` is TRANSLATION-side: by definition only the text chat is
            # interaction-side, everything else (incl. the agent's tool-planning
            # calls) uses the model's translation request parameters.  So take
            # ``model.temperature`` / ``model.reasoning_effort`` (models.json),
            # not ``interaction_*``.  ``tool_choice="auto"`` and the tool schemas
            # stay so the agent can still plan tool calls.
            kwargs: dict[str, Any] = {
                "model": model.model,
                "temperature": (
                    model.temperature if model.temperature is not None else 0.2
                ),
                "max_tokens": max_tokens,
                "tools": specs,
                "tool_choice": "auto",
                # Bounded to a protocol-valid tail (scheme 2): a page with many tool
                # rounds must not re-send the whole growing history each decide call.
                "messages": _window_messages(messages),
            }
            body: dict[str, Any] = {}
            # ``reasoning_effort`` (translation-side, e.g. "low" for llama.cpp) is a
            # model-specific body extra; send it via ``extra_body``.
            if model.reasoning_effort:
                body["reasoning_effort"] = model.reasoning_effort
            if body:
                kwargs["extra_body"] = body
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — fail-closed
            # A user cancellation aborts the request via the watchdog closing the
            # client; that must surface as a control signal, not be masked as a
            # one-off "decide failed" (which would let the page run report "done"
            # and keep going).
            if cancel is not None and cancel():
                raise _tr.TranslationCancelled() from exc
            if log:
                log(f"  decide 调用失败：{type(exc).__name__}: {exc}")
            return Decision(action="done", summary=f"error: {exc}")
        finally:
            if watchdog is not None:
                watchdog_stop.set()
        msg = resp.choices[0].message
        tcs = getattr(msg, "tool_calls", None)
        if not tcs:
            return Decision(action="done", summary=str(getattr(msg, "content", "") or ""))
        tc = tcs[0]
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            if not isinstance(args, dict):
                args = {}
        except Exception:
            args = {}
        pending_assistant = {
            "role": "assistant",
            "content": str(getattr(msg, "content", "") or ""),
            "tool_calls": [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }],
        }
        pending_call_id = tc.id
        return Decision(action="call", tool=tc.function.name, arguments=args)

    return _decision


def make_llm_interpret(model, log: Callable[[str], None] | None = None):
    """Return an ``interpret(answer, kind) -> 'translate'|'keep'|'skip'`` backed by the LLM.

    This is the AI-driven reading of a special-page answer (the M3 negotiation lets the
    user pick a button OR type free text; the model classifies whatever they said).  It
    is a translation-side call (``model.client_kwargs`` + ``request_params``), a tiny
    temperature-0 classification.  Fail-closed: on a bad client / network error / an
    unparseable reply it degrades to :func:`interpret_decision`, so the negotiation
    never hangs.
    """
    from .. import translator as _tr

    try:
        client = _tr.OpenAI(**model.client_kwargs())
    except Exception:  # noqa: BLE001 — no client → no AI interpretation, use the matcher
        return None

    def interpret(answer, kind):
        try:
            kwargs: dict[str, Any] = {
                "model": model.model,
                "temperature": 0.0,
                "max_tokens": 4,
                "messages": [{"role": "user",
                              "content": prompts.interpret_special_answer(str(answer or ""), str(kind or ""))}],
            }
            body = model.request_params()
            if body:
                kwargs["extra_body"] = body
            resp = client.chat.completions.create(**kwargs)
            text = (getattr(resp.choices[0].message, "content", "") or "").strip().lower()
            for action in ("translate", "keep", "skip"):
                if action in text:
                    return action
        except Exception as exc:  # noqa: BLE001 — fail-closed to the matcher
            if log:
                log(f"  特殊页回答 AI 解读失败：{type(exc).__name__}: {exc}（用规则匹配落回）。")
        return interpret_decision(answer)

    return interpret


def make_source_tools(state: WorkflowState, *, src_path=None) -> dict[str, Callable]:
    """Read-only tools bound to the immutable source (never write to it).

    ``read_page`` / ``get_layout`` return JSON-serialisable observations so the
    controller can reason about a page; neither mutates the original.
    """
    src_path = src_path or state.src_path

    def read_page(page: int, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
        src = state.src_doc
        if src is None or not (0 <= page < len(src.pages)):
            return {"page": page, "blocks": []}
        # The write tools (``translate_block`` / ``set_text`` / ...) all address
        # blocks by a *flat, document-global* index (``state.out_doc`` keys), so
        # the model must be told that same flat index here — a per-page local
        # index would silently address a different block once page > 0.  ``page``
        # is carried along only as context.  ``offset``/``limit`` let the model page
        # through a huge page without one massive tool message (bounded feedback).
        base = sum(len(p) for p in src.pages[:page])
        blocks = src.pages[page]
        total = len(blocks)
        start = max(0, int(offset or 0))
        end = total if limit is None else min(total, start + max(1, int(limit)))
        return {"page": page, "total": total,
                "offset": start, "limit": end - start,
                "truncated": end < total,
                "blocks": [
                    {"index": base + i, "text": b.text, "bbox": [b.x0, b.y0, b.x1, b.y1],
                     "in_table": bool(getattr(b, "in_table", False)),
                     "is_chart": bool(getattr(b, "is_chart", False))}
                    for i, b in enumerate(blocks[start:end])
                ]}

    def get_layout(page: int) -> dict[str, Any]:
        src = state.src_doc
        if src is None or not (0 <= page < len(src.pages)):
            return {"rows": 0, "cols": 0, "grid": []}
        blocks = src.pages[page]
        # Coarse reading-order grid: rows by y-centre cluster, columns by x.
        rows: list[list] = []
        for b in blocks:
            cy = (b.y0 + b.y1) / 2.0
            placed = False
            for row in rows:
                if any(abs(((x[1] + x[3]) / 2.0) - cy) <= 3.0 for x in row):
                    row.append((b.x0, b.y0, b.x1, b.y1, b.text))
                    placed = True
                    break
            if not placed:
                rows.append([(b.x0, b.y0, b.x1, b.y1, b.text)])
        out = []
        for row in rows:
            row.sort(key=lambda it: it[0])
            out.append([str(it[4]) for it in row])
        return {"rows": len(out), "cols": max((len(r) for r in out), default=0), "grid": out}

    def get_doc_info() -> dict[str, Any]:
        src = state.src_doc
        if src is None:
            return {}
        from .. import pdfio

        return pdfio.get_doc_info(src)

    def classify_page(page: int) -> dict[str, Any]:
        src = state.src_doc
        if src is None or not (0 <= page < len(src.pages)):
            return {"kind": "uncertain"}
        from .. import pdfio

        return {"kind": pdfio.classify_page(src.pages[page])}

    return {"read_page": read_page, "get_layout": get_layout,
            "get_doc_info": get_doc_info, "classify_page": classify_page}


def _has_cjk(text: Any) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text))


#: Full-width digit / decimal-separator rune → ASCII (accounts for OCR/PDF runs that
#: mix full-width glyphs so the number-fidelity check compares equal values).
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９．，", "0123456789.,")
#: Unicode minus (U+2212) and dashes used in PDF runs → ASCII hyphen-minus.
_UNICODE_MINUS = str.maketrans("−–—", "---")
#: A numeric token: an integer (optionally comma-grouped) or a decimal run; sign kept.
_NUM_TOKEN_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
#: Month names in date runs map to their number (``December 31, 2023`` must compare
#: equal to ``2023年12月31日``).  Capitalized only — "may be" must stay prose, never
#: become the digit 5.
_MONTH_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b")
_MONTH_NUM = {"January": "1", "February": "2", "March": "3", "April": "4",
              "May": "5", "June": "6", "July": "7", "August": "8",
              "September": "9", "October": "10", "November": "11", "December": "12"}
#: Unit multipliers that legitimately change a value's appearance: 万/亿 in a CJK
#: source, and thousand/million/billion/trillion (with ten/hundred prefixes) in a
#: Latin translation — ``3.14 亿元`` and ``314 million yuan`` are the same value.
_UNIT_EN_RE = re.compile(
    r"^\s*((?:ten|hundred)\s+)?(thousand|million|billion|trillion)\b", re.IGNORECASE)
_UNIT_EN_POWER = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12}
_UNIT_CN_RE = re.compile(r"^\s*(万亿|亿|万)\s*")
_UNIT_CN_POWER = {"万亿": 12, "亿": 8, "万": 4}
#: Latin words used by the residual-prose detector (a code like ``GB/T 33436-2016``
#: is not prose, an untranslated sentence is).
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:[’'-][A-Za-z]+)*")


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


def _number_signature(text: Any) -> "Counter[Decimal]":
    """The multiset of *values* (not token strings) that ``text`` contains.

    Full-width glyphs, Unicode minus and hyphen-separated runs (``2023-12-31``)
    are normalized first; month names in dates count as their number; and a unit
    multiplier directly attached to a value is applied.  So ``3,702,726,474.45``,
    ``３０７２７２６４７４．４５`` and ``December 31, 2023``
    all compare against their exact source forms *by value*, while a translation
    that drops / alters / invents a digit still does not.
    """
    t = str(text).translate(_FULLWIDTH_DIGITS).translate(_UNICODE_MINUS)
    # Month names first — the month-substituted digits must stay SEPARATE tokens
    # from the day/year ("December 31, 2023" == "2023年12月31日"), so no space
    # normalization may merge them.
    t = _MONTH_RE.sub(lambda m: _MONTH_NUM[m.group(0)], t)
    t = t.replace(",", "").replace("，", "")
    # Only the sign-adjacent space is an artifact: "− 7,326.50" keeps its sign
    # while "3.14 亿元" / "314 million yuan" keep the space the unit matcher
    # needs to attach the multiplier to the right value.
    t = re.sub(r"(?<=[-+])\s+(?=\d)", "", t)
    # A hyphen between digits is a run separator (2023-12-31), not a sign: putting
    # spaces around it keeps the tokenizer from reading "-12" as a negative number.
    t = re.sub(r"(?<=\d)-(?=\d)", " - ", t)
    out: Counter[Decimal] = Counter()
    spans = list(_NUM_TOKEN_RE.finditer(t))
    for i, m in enumerate(spans):
        tok = m.group(0)
        if tok in ("", "-", "+", "."):
            continue
        try:
            v = Decimal(tok)
        except InvalidOperation:
            continue
        # The unit that belongs to THIS value sits between the token and the next
        # number (a unit cannot skip over another figure).
        unit_win = (t[m.end():spans[i + 1].start()] if i + 1 < len(spans)
                    else t[m.end():m.end() + 16])
        out[v * _unit_multiplier(unit_win)] += 1
    return out


def _is_latin_prose(text: Any) -> bool:
    """True when ``text`` looks like untranslated Latin *prose*, not a code/acronym.

    A CJK target legitimately keeps Latin codes / units / acronyms (``GB/T
    33436-2016``, ``GDP``, ``USD 100 million``), so at least three Latin words and
    no CJK glyph at all are required before a block is reported as residual — the
    review loop otherwise burns its budget fixing a genuine figure.
    """
    t = str(text)
    if any("一" <= ch <= "鿿" for ch in t):
        return False
    return len(_LATIN_WORD_RE.findall(t)) >= 3


# --------------------------------------------------------------------------
# Deterministic audit engine.
#
# The four-area review checks live here as module-level functions over a
# ``WorkflowState`` (single source of truth), shared by the bound ``check_*``
# tools in ``make_page_executors`` AND by ``DocumentSession`` — so the review
# loop can run a deterministic audit *before/after* an ``AgentStep`` without
# going through the agent loop.  Previously the audit lived only inside the
# per-page closures, so the review pass relied on the model "remembering" to run
# all four checks (the source of false green).
# --------------------------------------------------------------------------

#: The review-check names, in the order the audit runs them.
_AUDIT_DEFAULT_CHECKS = ("layout", "residual", "missing", "numbers", "table")

#: How many audit→fix→re-audit rounds a single page may take before the review
#: loop gives up (fail-closed to the best translation so far) and moves on.
REVIEW_MAX_ATTEMPTS = 3


def _audit_blocks(state, page):
    """``(blocks, flat_offset)`` for ``page`` (``None`` = whole doc)."""
    src = state.src_doc
    if src is None or not getattr(src, "pages", None):
        return [], 0
    if page is None:
        return [b for pg in src.pages for b in pg], 0
    if not (0 <= page < len(src.pages)):
        return [], 0
    offset = sum(len(pg) for pg in src.pages[:page])
    return src.pages[page], offset


def _audit_read(state, idx):
    """The translated text for flat index ``idx`` (``""`` when none)."""
    out = state.out_doc or {}
    return str((out.get(idx) or {}).get("text", ""))


def _audit_protected(state, block):
    """True when a block must stay byte-identical (engine-skipped or numeric cell)."""
    from .. import pdfio as _pdfio
    return (not _tr._needs_translation(str(block.text))
            or _pdfio._is_numeric_cell(str(block.text)))


def _check_residual(state, page=None):
    """Untranslated content left on ``page`` (``None`` = whole doc)."""
    blocks, offset = _audit_blocks(state, page)
    cjk_target = prompts._is_cjk_language(state.lang)
    out = []
    for i, b in enumerate(blocks):
        if b.is_chart or _audit_protected(state, b):
            continue
        idx = offset + i
        t = _audit_read(state, idx)
        if not t.strip():
            out.append({"index": idx, "text": str(b.text), "reason": "empty"})
        elif cjk_target:
            if _is_latin_prose(t):
                out.append({"index": idx, "text": str(b.text), "reason": "residual_latin"})
        elif _has_cjk(t):
            out.append({"index": idx, "text": str(b.text), "reason": "residual_cjk"})
    return {"residual": out, "page": page}


def _check_missing(state, page=None):
    blocks, offset = _audit_blocks(state, page)
    out = []
    for i, b in enumerate(blocks):
        if _audit_protected(state, b):
            continue
        idx = offset + i
        if str(b.text).strip() and not _audit_read(state, idx).strip():
            out.append({"index": idx, "text": str(b.text)})
    return {"missing": out, "page": page}


def _check_numbers(state, page=None):
    """Number fidelity: are the page's numerals preserved in the translation?"""
    blocks, offset = _audit_blocks(state, page)
    out = []
    for i, b in enumerate(blocks):
        idx = offset + i
        t = _audit_read(state, idx)
        if not t.strip():
            continue   # untranslated → handled by check_missing, not number fidelity
        src = _number_signature(str(b.text))
        trans = _number_signature(t)
        missing = list((src - trans).elements())
        extra = list((trans - src).elements())
        if missing or extra:
            out.append({"index": idx, "source": str(b.text), "translation": t,
                        "missing": missing, "extra": extra})
    return {"numbers": out, "page": page}


def _check_table(state, page=None):
    """Table / text completeness for ``page`` (``None`` = whole doc)."""
    blocks, offset = _audit_blocks(state, page)
    cells: list[tuple[int, str]] = []
    texts: list[tuple[int, str]] = []
    for i, b in enumerate(blocks):
        if _audit_protected(state, b) or getattr(b, "is_chart", False):
            continue
        idx = offset + i
        (cells if getattr(b, "in_table", False) else texts).append((idx, str(b.text)))

    def _empty(cands: list[tuple[int, str]]) -> list[int]:
        return [idx for idx, _ in cands if not _audit_read(state, idx).strip()]

    empty_cells = _empty(cells)
    empty_text = _empty(texts)
    return {
        "page": page,
        "source_cells": len(cells),
        "translated_cells": len(cells) - len(empty_cells),
        "empty_cells": empty_cells,
        "empty_text_count": len(empty_text),
        "empty_text": empty_text,
        "complete": not empty_cells and not empty_text,
    }


def _check_layout(state, page=None):
    """Structure / layout integrity for ``page`` using the exporter's own fit rules."""
    from .. import pdfio as _pdfio
    blocks, offset = _audit_blocks(state, page)
    font = _pdfio._CJK_FONT
    issues: list[dict[str, Any]] = []
    for i, b in enumerate(blocks):
        if _audit_protected(state, b) or _pdfio._is_vertical_label(b):
            continue
        idx = offset + i
        t = _audit_read(state, idx)
        if not t.strip():
            continue
        try:
            lines, fs = _pdfio._fit_block(b, font, t)
        except Exception:  # noqa: BLE001 — a block we cannot measure is skipped
            continue
        in_table = bool(getattr(b, "in_table", False))
        leading = _pdfio._line_leading(font, in_table=in_table, n_lines=len(lines))
        height = _pdfio._wrapped_height(font, lines, fs, leading)
        box_h = max(0.5, b.y1 - b.y0)
        start_fs = max(5.0, min(b.size, _pdfio._MAX_FONT))
        floor = (_pdfio._MIN_TABLE_FLOOR if in_table
                 else min(start_fs, _pdfio._MIN_READABLE))
        if fs + 1e-9 < floor:
            issues.append({"index": idx, "kind": "too_small",
                           "detail": f"译文字号 {fs:.2f}pt 低于可读下限 {floor:.2f}pt"})
        if height > box_h + 2.0:
            issues.append({"index": idx, "kind": "overflow",
                           "detail": f"译文高度 {height:.1f}pt 超过自身框 {box_h:.1f}pt"})
        below = [nb for nb in blocks
                 if nb is not b and nb.y0 >= b.y1 - 0.5
                 and nb.x0 < b.x1 and nb.x1 > b.x0]
        if below:
            gap = min(nb.y0 for nb in below) - b.y1
            if height > box_h + gap + 2.0:
                issues.append({"index": idx, "kind": "crowding",
                               "detail": f"译文高 {height:.1f}pt 会压入下一块（剩余 {gap:.1f}pt）"})
    return {"page": page, "count": len(issues), "issues": issues[:60],
            "truncated": len(issues) > 60}


#: name → check function (the audit aggregator fans out over this).
_AUDIT_CHECKS = {
    "layout": _check_layout,
    "residual": _check_residual,
    "missing": _check_missing,
    "numbers": _check_numbers,
    "table": _check_table,
}


def _audit_normalize(name: str, res: dict) -> tuple[list[dict], bool]:
    """Convert one check result into ``(issues, is_clean)`` with a ``check`` tag."""
    if name == "table":
        if not res.get("complete"):
            return ([{"check": name, "empty_cells": res.get("empty_cells", []),
                      "empty_text": res.get("empty_text", []),
                      "source_cells": res.get("source_cells", 0),
                      "translated_cells": res.get("translated_cells", 0),
                      "complete": False}], False)
        return [], True
    key = {"layout": "issues", "residual": "residual", "missing": "missing",
           "numbers": "numbers"}[name]
    items = res.get(key, [])
    return [{"check": name, **item} for item in items], not items


def audit_page(state, page=None, checks=None) -> dict[str, Any]:
    """Run a deterministic multi-check audit over ``state``.

    ``checks`` is a subset of ``_AUDIT_DEFAULT_CHECKS`` (default all).  Returns
    ``{"page", "checks_requested", "checks", "issues", "clean"}`` where ``issues``
    is a flat, ``check``-tagged list suitable both for machine gating (``clean``)
    and for injecting into ``prompts.review_page_task`` as concrete findings.
    """
    names = [n for n in (checks or _AUDIT_DEFAULT_CHECKS) if n in _AUDIT_CHECKS]
    all_issues: list[dict] = []
    per_check: dict[str, Any] = {}
    for name in names:
        issues, _clean = _audit_normalize(name, _AUDIT_CHECKS[name](state, page))
        per_check[name] = {"clean": not issues, "count": len(issues)}
        all_issues.extend(issues)
    return {"page": page, "checks_requested": names, "checks": per_check,
            "issues": all_issues, "clean": not all_issues}


def make_page_executors(state: WorkflowState, model, log: Callable[[str], None] | None = None,
                        preview_handler: Callable[..., bytes] | None = None,
                        answer_handler: Callable[..., Any] | None = None,
                        cancel: Callable[[], bool] | None = None,
                        render_handler: Callable[[int, str], bytes | None] | None = None) -> dict[str, Callable]:
    """Bound deterministic content / verify / draw tools that operate on ``out_doc``.

    These are the agent's *hands* for a single page (the source is read-only via
    ``make_source_tools``).  ``state.out_doc`` is a mutable ``{index: {...}}``
    overlay holding the translation text (+ optional size/align).  Guards:

    * ``set_text`` / ``translate_block`` refuse a numeric / code source block
      (financial figures are never rewritten by AI).
    * everything that changes a block writes only to ``out_doc``, never to the
      immutable source.
    * ``preview_handler`` (optional) lets ``preview_page`` show a page to the user
      and return the framed region the user drew (see ``preview.PreviewBridge``).
    * ``cancel`` (optional) is polled by per-block translation so a user
      cancellation aborts an in-flight request instead of running to its timeout.
    """
    from pathlib import Path

    from .. import pdfio
    from .. import translator as _tr

    engine = _tr.TranslationEngine(model)
    retranslate = _tr.make_retranslate_fn(model, log)
    retranslate_batch = _tr.make_retranslate_batch_fn(model, log)

    def _flat_blocks() -> list:
        if state.src_doc is None:
            return []
        return [b for page in state.src_doc.pages for b in page]

    def _block(index: int):
        blocks = _flat_blocks()
        return blocks[index] if 0 <= index < len(blocks) else None

    def _flat_of(b) -> int | None:
        for i, x in enumerate(_flat_blocks()):
            if x is b:
                return i
        return None

    def _out() -> dict:
        if state.out_doc is None:
            state.out_doc = {}
        return state.out_doc

    def _write(index: int, text: str) -> None:
        _out().setdefault(index, {})["text"] = str(text)

    def _read(index: int) -> str:
        return str((_out().get(index) or {}).get("text", ""))

    def _translate(source: str, lang: str, attempts: int = 2) -> str:
        # The engine itself retries transient failures internally; an outer loop
        # widens the window so a briefly-hiccuping local llama-server is more
        # likely to land on a good run (the single-page loop fails closed to the
        # source otherwise).
        import time

        last: str = source
        for i in range(attempts):
            try:
                result = engine.translate_blocks([source], lang, log=log,
                                                 cancel=cancel,
                                                 doc_path=Path(state.src_path),
                                                 extra_glossary=(
                                                     state.user_decisions.get("terminology") or {}))
            except _tr.TranslationCancelled:
                raise                     # a user cancel is a control signal, not a failure
            except Exception as exc:  # noqa: BLE001 — retry, then keep source
                if i < attempts - 1:
                    time.sleep(0.5 * (i + 1))
                    continue
                if log:
                    log(f"  翻译重试失败：{type(exc).__name__}: {exc}")
                return source
            # A batch that hit a transient failure reports its errors and keeps the
            # source; only a clean result is trusted.
            if not getattr(result, "errors", None) and result.translated:
                return str(result.translated[0])
            if i < attempts - 1:
                time.sleep(0.5 * (i + 1))
        return last

    def translate_block(index: int, text: str | None = None, target_lang: str | None = None):
        b = _block(index)
        if b is None:
            return {"ok": False, "error": f"bad index {index}"}
        if pdfio._is_numeric_cell(str(b.text)):
            return {"ok": False, "error": "数字/代码块不可被 AI 改写（保真）"}
        src = str(text) if text is not None else str(b.text)
        translated = _translate(src, target_lang or state.lang)
        _write(index, translated)
        return {"ok": True, "index": index, "translated": translated}

    def translate_blocks(page: int, indices: list[int] | None = None,
                         target_lang: str | None = None):
        """Translate several blocks in ONE batched request (or every one on ``page``).

        ``indices`` are flat block indices (from ``read_page``); when omitted, every
        translatable block on ``page`` is picked.  ``engine.translate_blocks`` batches the
        picked source texts by the model's character budget and runs them concurrently
        (``model.concurrency``), so a whole table / row translates in a *fraction* of the
        per-block requests — the dominant cost in a block-by-block loop.  Numeric / code /
        non-letter blocks are skipped (never rewritten); a failed block keeps its source
        and is reported.
        """
        lang = target_lang or state.lang
        src_doc = state.src_doc
        if src_doc is None or not (0 <= int(page) < len(src_doc.pages)):
            return {"ok": False, "error": f"bad page {page}"}
        if indices is None:
            base = sum(len(p) for p in src_doc.pages[: int(page)])
            candidates = [base + i for i in range(len(src_doc.pages[int(page)]))]
        else:
            candidates = [int(i) for i in indices]
        picked: list[int] = []
        seen: set[int] = set()
        for idx in candidates:
            if idx in seen:
                continue
            seen.add(idx)
            b = _block(idx)
            if b is None:
                continue
            if not _tr._needs_translation(str(b.text)) or pdfio._is_numeric_cell(str(b.text)):
                continue
            picked.append(idx)
        if not picked:
            return {"ok": False, "error": "没有可翻译的块（全部为数字/代码/空块）"}
        sources = [str(_block(i).text) for i in picked]
        result = engine.translate_blocks(
            sources, lang, log=log, cancel=cancel,
            doc_path=Path(state.src_path),
            extra_glossary=state.user_decisions.get("terminology") or {},
        )
        failed: set[int] = set()
        for err in (result.errors or []):
            m = re.search(r"块 (\d+) 翻译失败", err)
            if m:
                failed.add(int(m.group(1)) - 1)
        written = 0
        for i, text in enumerate(result.translated or []):
            if i in failed:
                continue
            _write(picked[i], str(text))
            written += 1
        return {"ok": True, "page": int(page), "count": written,
                "indices": picked,
                "translated": {str(i): _read(i) for i in picked if _read(i)},
                "failed": sorted(picked[i] for i in failed) if failed else []}

    def set_text(page: int, index: int, text: str):
        b = _block(index)
        if b is None:
            return {"ok": False, "error": f"bad index {index}"}
        if pdfio._is_numeric_cell(str(b.text)):
            return {"ok": False, "error": "数字/代码块不可被 AI 改写（保真）"}
        _write(index, text)
        return {"ok": True, "index": index}

    def delete_block(page: int, index: int):
        if index in _out():
            _out()[index].pop("text", None)
            if not _out()[index]:
                _out().pop(index, None)
            return {"ok": True, "index": index}
        return {"ok": False, "error": f"no translation for index {index}"}

    def apply_annotation(page: int, bbox, text: str | None = None, action: str = "set"):
        """M6: edit the block under a user-drawn region on the preview.

        ``bbox`` is ``[x0, y0, x1, y1]`` in PDF points; the nearest source block is
        found and its translation is set to ``text`` (``action="set"``) or deleted
        (``action="delete"``).  The edit is recorded as a user-confirmed ``Op``.
        """
        src = state.src_doc
        if src is None or not (0 <= page < len(src.pages)):
            return {"ok": False, "error": f"bad page {page}"}
        block = pdfio.nearest_block(src.pages[page], bbox)
        if block is None:
            return {"ok": False, "error": "标注区域未命中任何块"}
        flat = _flat_of(block)
        if flat is None:
            return {"ok": False, "error": "块定位失败"}
        action = (action or "set").strip().lower()
        if action in ("delete", "void"):
            out = _out()
            if flat in out:
                out[flat].pop("text", None)
                if not out[flat]:
                    out.pop(flat, None)
            state.record_op(tool="apply_annotation",
                            args={"page": page, "bbox": list(bbox), "action": action},
                            target=f"page:{page} block:{flat}", reason="用户标注：删除该块",
                            user_confirmed=True)
            return {"ok": True, "page": page, "index": flat, "action": action}
        if text is None:
            return {"ok": False, "error": "action=set 需要 text"}
        _write(flat, str(text))
        state.record_op(tool="apply_annotation",
                        args={"page": page, "bbox": list(bbox), "action": action,
                              "text": str(text), "source": str(block.text)},
                        target=f"page:{page} block:{flat}", reason="用户标注：改写该块",
                        user_confirmed=True)
        return {"ok": True, "page": page, "index": flat, "action": action,
                "text": _read(flat)}

    def retranslate_block(text: str, target_lang: str | None = None):
        if pdfio._is_numeric_cell(str(text)):
            return {"ok": False, "error": "数字/代码块不可被 AI 改写（保真）"}
        if retranslate is None:
            return {"ok": False, "error": "重译不可用（非视觉模型）"}
        return {"ok": True, "text": retranslate(str(text), target_lang or state.lang)}

    def retranslate_blocks(page: int, indices: list[int] | None = None,
                           target_lang: str | None = None):
        """Re-translate several blocks in ONE request (bypasses the cache).

        Like ``translate_blocks`` but never reuses a cached translation: it forces a
        fresh model translation of each source, so the AI self-check can re-translate a
        batch of problem blocks (residual / missing / number-mismatched) at once instead
        of one ``chat.completions`` per block (``retranslate_block``).  Numeric / code /
        non-letter blocks are skipped; a block the model misses keeps its source.
        """
        lang = target_lang or state.lang
        src_doc = state.src_doc
        if src_doc is None or not (0 <= int(page) < len(src_doc.pages)):
            return {"ok": False, "error": f"bad page {page}"}
        if retranslate_batch is None:
            return {"ok": False, "error": "批量重译不可用（非视觉模型）"}
        if indices is None:
            base = sum(len(p) for p in src_doc.pages[: int(page)])
            candidates = [base + i for i in range(len(src_doc.pages[int(page)]))]
        else:
            candidates = [int(i) for i in indices]
        picked: list[int] = []
        seen: set[int] = set()
        for idx in candidates:
            if idx in seen:
                continue
            seen.add(idx)
            b = _block(idx)
            if b is None:
                continue
            if not _tr._needs_translation(str(b.text)) or pdfio._is_numeric_cell(str(b.text)):
                continue
            picked.append(idx)
        if not picked:
            return {"ok": False, "error": "没有可重译的块（全部为数字/代码/空块）"}
        sources = [str(_block(i).text) for i in picked]
        results = retranslate_batch(sources, lang)
        written = 0
        failed: list[int] = []
        for i, txt in zip(picked, results):
            if txt.strip() and txt != str(_block(i).text):
                _write(i, str(txt))
                written += 1
            else:
                failed.append(i)
        return {"ok": True, "page": int(page), "count": written,
                "indices": picked,
                "translated": {str(i): _read(i) for i in picked if _read(i)},
                "failed": failed}

    def apply_terminology(source: str, target: str):
        state.user_decisions.setdefault("terminology", {})[str(source)] = str(target)
        return {"ok": True}

    def render_page(page: int, what: str = "translation"):
        """Render the in-progress page so the model can visually self-check.

        ``what="translation"`` renders the current in-place translation for ``page``
        (redacting original text the agent has translated); ``what="source"`` renders
        the original.  Falls back to the source page when there is no translation yet
        (or no render channel).  The PNG is returned as the tool ``image`` and fed
        back as a fresh visual observation by ``make_llm_decide``.
        """
        if render_handler is None:
            return {"ok": False, "error": "渲染通道未接线"}
        try:
            png = render_handler(int(page), str(what or "translation"))
        except Exception as exc:  # noqa: BLE001 — fail-closed
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "page": int(page), "what": what or "translation",
                "image": png or b""}

    def check_residual(page: int | None = None):
        """Untranslated content left on ``page`` (``None`` = whole doc)."""
        return _check_residual(state, page)

    def check_missing(page: int | None = None):
        return _check_missing(state, page)

    def check_numbers(page: int | None = None):
        """Number fidelity: are the page's numerals preserved in the translation?"""
        return _check_numbers(state, page)

    def check_table(page: int | None = None):
        """Table / text completeness for ``page`` (``None`` = whole doc)."""
        return _check_table(state, page)

    def check_layout(page: int | None = None):
        """Structure / layout integrity for ``page`` using the exporter's own fit rules."""
        return _check_layout(state, page)

    def audit_tool(page: int | None = None, checks: list[str] | None = None):
        """Deterministic multi-check audit (aggregates the ``check_*`` results)."""
        return audit_page(state, page, checks)

    def preview_page(page: int, what: str = "translation", region=None, **_kw):
        if preview_handler is None:
            return {"ok": False, "error": "预览通道未接线"}
        try:
            res = preview_handler(page, what, region)
        except Exception as exc:  # noqa: BLE001 — fail-closed
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        # The handler returns either the cropped PNG bytes or ``{"png": .., "rect": ..}``.
        image = rect = None
        if isinstance(res, dict):
            image = res.get("png")
            rect = res.get("rect")
        else:
            image = res
        return {"ok": True, "page": page, "image": image or b"", "rect": rect}

    def ask_user(question: str, options=None, target: str = ""):
        if answer_handler is None:
            return {"ok": False, "error": "问答通道未接线"}
        try:
            ans = answer_handler(str(question), list(options or []), str(target or ""))
        except Exception as exc:  # noqa: BLE001 — fail-closed
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        value = (ans or {}).get("value") if isinstance(ans, dict) else ans
        return {"ok": True, "answer": value}

    def detect_page_skew(page: int):
        """Low-risk geometry check for a scanned page: report the skew and, when it is
        noticeable and a question channel is wired, ask the user whether to run a
        (targeted) geometry correction.  Never modifies the PDF — it only detects and
        records the decision (``state.user_decisions["ocr_skew:<page>"]``)."""
        if not state.src_path:
            return {"ok": False, "error": "没有源文件"}
        d = pdfio.detect_page_skew(state.src_path, int(page))
        decision = None
        if d.get("recommended") and answer_handler is not None:
            q = (f"第 {int(page) + 1} 页（扫描件）检测到约 {d.get('skew_degrees', 0.0):.1f}° 文本倾斜，"
                 f"是否做几何校正？")
            try:
                ans = answer_handler(q, ["校正", "忽略"], f"ocr_skew:{int(page)}")
            except Exception as exc:  # noqa: BLE001 — fail-closed to "no correction"
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            val = (ans.get("value") if isinstance(ans, dict) else ans)
            raw = str(val or "").strip()
            decision = "校正" if any(k in raw for k in ("校正", "纠正", "修正")) else "忽略"
            state.user_decisions[f"ocr_skew:{int(page)}"] = {
                "skew": d.get("skew_degrees"), "apply": decision == "校正",
            }
            state.record_op(tool="detect_page_skew", args={"page": int(page)},
                            target=f"page:{int(page)}",
                            reason=f"OCR 几何检测：{d.get('skew_degrees', 0.0)}°",
                            user_confirmed=True)
        return {"ok": True, "page": int(page), "skew_degrees": d.get("skew_degrees"),
                "recommended": d.get("recommended"), "decision": decision,
                "reason": d.get("reason")}

    return {
        "translate_block": translate_block,
        "translate_blocks": translate_blocks,
        "retranslate_block": retranslate_block,
        "retranslate_blocks": retranslate_blocks,
        "set_text": set_text,
        "delete_block": delete_block,
        "apply_annotation": apply_annotation,
        "apply_terminology": apply_terminology,
        "check_residual": check_residual,
        "check_missing": check_missing,
        "check_numbers": check_numbers,
        "check_table": check_table,
        "check_layout": check_layout,
        "audit_page": audit_tool,
        "render_page": render_page,
        "preview_page": preview_page,
        "detect_page_skew": detect_page_skew,
        "ask_user": ask_user,
    }


def run_page_visual(state: WorkflowState, page_index: int, model,
                    tools: dict[str, Callable] | None = None, *,
                    task: str, log: Callable[[str], None] | None = None,
                    max_steps: int | None = None, max_rounds: int | None = None,
                    src_path=None, preview_handler: Callable[..., bytes] | None = None,
                    answer_handler: Callable[[str, list[str], str], Any] | None = None,
                    cancel: Callable[[], bool] | None = None,
                    render_handler: Callable[[int, str], bytes | None] | None = None) -> WorkflowState:
    """Run the AI-orchestrated loop on a single page (the visual closed loop).

    Renders the source page once, hands it to a real-LLM ``decide`` (image +
    tool schemas), and drives ``FlowAgent``: the model observes the page, calls
    tools (read the source / classify / verify / translate), reads the results,
    verifies and recurses until it stops or the budget is spent.  ``state.src_doc``
    should already be populated (extracted blocks) so ``read_page`` works.

    ``preview_handler`` (e.g. ``preview.PreviewBridge.get_region``) lets the agent's
    ``preview_page`` tool show a page to the user and receive the framed region back.

    ``cancel`` (optional) is polled by the loop and by per-block translation, so a
    user cancellation stops this page promptly rather than only between pages.
    """
    state.page(page_index)
    if not any(g.page == page_index for g in state.todo):
        state.todo.append(Goal(kind="page", page=page_index))
    src_path = src_path or state.src_path

    # ``state.budget`` is shared across pages when the worker drives every page
    # through this call (``_run_agent``), so its counters must be reset per page —
    # otherwise ``used_steps`` accumulates and every page after the first starts
    # already exhausted and does nothing.
    state.budget.used_steps = 0

    def image_provider(_s: WorkflowState) -> bytes:
        try:
            return _render_source_page(src_path, page_index)
        except Exception:  # noqa: BLE001 — a bad source page must not crash the loop
            return b""

    if not getattr(model, "vision", False):
        state.log.append("模型不支持视觉，无法执行 AI 编排（回退确定性流水线）。")
        if log:
            log("  [agent] 模型不支持视觉，单页编排跳过。")
        return state

    all_tools = dict(make_source_tools(state, src_path=src_path))
    all_tools.update(make_page_executors(state, model, log=log,
                                         preview_handler=preview_handler,
                                         answer_handler=answer_handler,
                                         cancel=cancel,
                                         render_handler=render_handler))
    if tools:
        all_tools.update(tools)
    decide = make_llm_decide(model, task=task,
                             image_provider=image_provider, log=log,
                             tool_names=list(all_tools), cancel=cancel)
    agent = FlowAgent(state, all_tools, decide, log=log, max_steps=max_steps,
                      answer_handler=answer_handler, cancel_fn=cancel)
    return agent.run(max_rounds=max_rounds)


class DocumentSession:
    """M2: document-level interactive-session controller.

    Drives a :class:`WorkflowState` through the phase state machine
    ``preprocess → translate_normal → special_pages → completed``.  It owns the
    *ordering* (translate ALL normal pages first, then handle special pages) and
    reports progress; the per-page translation is delegated to ``translate_page``
    (an injected callable, normally :func:`run_page_visual`), so a test can script
    it and assert the order.

    In ``special_pages`` each non-normal page is shown in the preview (``show_preview``)
    and the user is asked a per-kind question (``answer_handler``); the decision
    (translate / keep / skip) is honoured and recorded with ``user_confirmed``.  A
    page that fails is fail-closed to its source text and its ``PageState`` is set
    ``needs_user`` so the REVIEW/MODIFY phase can revisit.
    """

    def __init__(
        self,
        state: WorkflowState,
        doc,
        model,
        *,
        log: Callable[[str], None] | None = None,
        translate_page: Callable[..., Any] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
        preview_handler: Callable[..., Any] | None = None,
        answer_handler: Callable[..., Any] | None = None,
        show_preview: Callable[[int, str], None] | None = None,
        render_handler: Callable[[int, str], bytes | None] | None = None,
        audit: Callable[..., dict[str, Any]] | None = None,
        interpret: Callable[[str, str], str] | None = None,
        include_kept: bool = False,
        max_steps_per_page: int = 24,
    ) -> None:
        self.state = state
        self.doc = doc
        self.model = model
        self.log = log or (lambda m: None)
        self.translate_page = translate_page or run_page_visual
        self.progress = progress or (lambda _d, _t, _s: None)
        self.cancel = cancel or (lambda: False)
        self.preview_handler = preview_handler
        self.answer_handler = answer_handler
        self.show_preview = show_preview
        self.render_handler = render_handler
        #: M4 review gate: ``audit(page[, checks]) -> findings`` runs the deterministic
        #: check before/after an AgentStep.  Defaults to ``audit_page`` over the live
        #: state; a test injects a fake to control which pages flow to the AI fix pass.
        self.audit = audit or (lambda page=None, checks=None: audit_page(self.state, page, checks))
        #: M3 negotiation: ``interpret(answer, kind) -> 'translate'|'keep'|'skip'`` reads the
        #: user's special-page answer (incl. free text) — an AI interpretation; defaults to
        #: the flexible ``interpret_decision`` matcher when not injected.
        self.interpret = interpret
        #: M4 (U1 knob): when True the AI self-check also reviews pages the user chose to
        #: keep/skip (default False — those carry the source verbatim, so re-checking them
        #: would wrongly try to translate the intentionally-kept original).
        self.include_kept = include_kept
        self.max_steps_per_page = max_steps_per_page

    #: phase-name → phase constant, used to set ``state.phase`` as the plan advances.
    _PHASE_OF = {
        "preprocess": PHASE_PREPROCESS,
        "translate_normal": PHASE_TRANSLATE_NORMAL,
        "special_pages": PHASE_SPECIAL_PAGES,
        "review": PHASE_REVIEW,
        "completed": PHASE_COMPLETED,
    }


    def run(self) -> WorkflowState:
        """Run the phase state machine to completion (or raise on cancel).

        The top-level phase ORDER is declared as data in ``translate_doc.scope["phases"]``
        (a standard flow), so this is a thin dispatcher over the declared plan instead of
        a hardcoded call sequence.  Each phase method remains the command implementation
        (and is itself flow-driven at the per-page unit level).
        """
        plan = list(STANDARD_FLOWS["translate_doc"].scope.get(
            "phases", ["preprocess", "translate_normal", "special_pages", "review", "completed"]))
        for name in plan:
            self.state.phase = self._PHASE_OF.get(name, self.state.phase)
            method = getattr(self, f"_{name}", None)
            if method is None:
                raise ValueError(f"未知阶段：{name}")
            method()
        self.state.phase = PHASE_DONE
        return self.state

    def _preprocess(self) -> None:
        from .. import pdfio

        info = pdfio.get_doc_info(self.doc)
        self.state.doc_info = DocInfo.from_dict(info)
        self.state.triage = {
            i: PageTriage(page=i, kind=kind)
            for i, kind in enumerate(info.get("kinds", []))
        }
        d = self.state.doc_info
        self.log(
            f"  文档信息：共 {d.pages} 页，语言 {d.language}；正常 {d.text_pages} 页，"
            f"扫描 {d.scan_pages} 页，图表 {d.chart_pages} 页，表格 {d.table_pages} 页，"
            f"待确认 {d.uncertain_pages} 页。"
        )
        self.progress(0, d.pages, "预处理")

    def _page_agent(self, page: int):
        """Bind one page's agent channel (``translate_page``) for a flow's AgentStep.

        ``run_flow`` calls ``run_agent(task=..., page=..., max_steps=...)``; this wires
        it to the phase handler (normally ``run_page_visual``) with the session's
        channels, and converts the pipeline's cancellation control signal into
        ``FlowCancelled`` so it propagates (not swallowed by the flow's fail-closed
        ``except``).
        """
        def run_agent(*, task, page, max_steps=None, image=False):
            try:
                return self.translate_page(
                    self.state, page, self.model, task=task,
                    max_steps=max_steps or self.max_steps_per_page, log=self.log,
                    preview_handler=self.preview_handler, answer_handler=self.answer_handler,
                    cancel=self.cancel, render_handler=self.render_handler,
                )
            except _tr.TranslationCancelled:
                raise FlowCancelled()
        return run_agent

    def _translate_normal(self) -> None:
        normal = [i for i, t in self.state.triage.items() if t.kind == "normal"]
        total = len(normal)
        for done, i in enumerate(normal, start=1):
            if self.cancel():
                raise _tr.TranslationCancelled()
            ps = self.state.page(i)
            ps.status = STATUS_IN_PROGRESS
            try:
                rs = run_flow(STANDARD_FLOWS["translate_page"],
                              run_agent=self._page_agent(i), cancel=self.cancel,
                              log=self.log,
                              params={"page": i, "lang": self.state.lang})
            except FlowCancelled:
                raise _tr.TranslationCancelled()
            except Exception as exc:  # noqa: BLE001 — fail-closed per page
                ps.status = STATUS_NEEDS_USER
                ps.issues.append(f"翻译失败：{type(exc).__name__}")
                self.log(f"  第 {i + 1} 页翻译失败：{type(exc).__name__}: {exc}（保留原文）。")
            else:
                if rs.ok:
                    # Fail-closed: a page with translatable blocks but NO AI translation
                    # (the model answered with prose / never called a translate tool) must
                    # not be recorded DONE — the export would carry the untranslated
                    # source.  ``_page_translation_counts`` returns 0,0 for an all-
                    # numeric/protected page, which falls through to DONE correctly.
                    translatable, translated = _page_translation_counts(self.state, i)
                    if translatable > 0 and translated == 0:
                        ps.status = STATUS_NEEDS_USER
                        ps.issues.append("该页未产生任何译文（AI 未执行翻译）")
                        self.log(f"  第 {i + 1} 页未产生任何译文（AI 未执行翻译），保留原文。")
                    else:
                        ps.status = STATUS_DONE
                else:
                    ps.status = STATUS_NEEDS_USER
                    ps.issues.append(f"翻译失败：{rs.error}")
                    self.log(f"  第 {i + 1} 页翻译失败：{rs.error}（保留原文）。")
            self.progress(done, total, "翻译正常页")
        if not total:
            self.log("  未发现正常文本页，跳过批量翻译。")

    def _special_pages(self) -> None:
        special = [i for i, t in self.state.triage.items() if t.kind != "normal"]
        if not special:
            self.log("  无特殊页。")
            return
        self.log(f"  [特殊页] 共 {len(special)} 页，将逐页与用户协商处理。")
        for i in special:
            if self.cancel():
                raise _tr.TranslationCancelled()
            t = self.state.triage[i]
            ps = self.state.page(i)
            kind = t.kind
            question, options = prompts.special_page_question(i, kind)
            # Show the pending original page so the user can look at it while the
            # question is up (non-blocking; the answer below is what blocks).
            self.log(f"  [特殊页] 第 {i + 1} 页（{kind}）请在预览查看，并在【侧栏】选择处理方式…")
            if self.show_preview is not None:
                try:
                    self.show_preview(i, "source")
                except Exception:  # noqa: BLE001 — preview is cosmetic
                    pass
            # Drive the negotiation through the registered ``special_page`` flow (a
            # UserStep ask); the decision is then interpreted by the phase (AI-injectable
            # via ``self.interpret``, else the flexible ``interpret_decision`` matcher),
            # which also executes the translate/keep/skip.
            decision, raw = "keep", ""
            if self.answer_handler is not None:
                try:
                    rs = run_flow(STANDARD_FLOWS["special_page"],
                                  ask=self.answer_handler, cancel=self.cancel,
                                  log=self.log,
                                  params={"page": i, "kind": kind, "lang": self.state.lang})
                    if rs.ok:
                        ans = rs.result.get(f"user:page:{i}") or {}
                        raw = ans.get("value") if isinstance(ans, dict) else ans
                        decision = self._interpret_answer(raw, kind)
                    else:
                        self.log(f"  特殊页问答未完成：{rs.error}（按保留原文处理）。")
                except FlowCancelled:
                    raise _tr.TranslationCancelled()
                except Exception as exc:  # noqa: BLE001 — fail-closed to retain
                    self.log(f"  特殊页问答失败：{type(exc).__name__}: {exc}（按保留原文处理）。")
            else:
                self.log(f"  （无问答通道，第 {i + 1} 特殊页按保留原文处理。）")
            # A pending question is interruptible by "取消" (the answer bridge polls it);
            # a cancelled ask returns ``None`` — treat that as a real cancellation, not as
            # "保留原文", so the whole run aborts instead of silently continuing.
            if self.cancel():
                raise _tr.TranslationCancelled()
            t.decided = True
            t.decision = decision
            ps.issues.append(f"特殊页（{kind}）按用户意见：{decision}")
            self.state.record_op(
                tool="ask_user", args={"question": question, "options": options, "answer": raw},
                target=f"page:{i}", reason=f"特殊页 {kind} 协商",
                user_confirmed=self.answer_handler is not None,
            )
            if decision == "translate":
                ok = self._translate_special_page(i, kind)
                if ok:
                    ps.status = STATUS_DONE
                    self.state.record_op(
                        tool="translate_block", args={"page": i}, target=f"page:{i}",
                        reason=f"特殊页 {kind} 按用户意见翻译", user_confirmed=True,
                    )
                else:
                    ps.status = STATUS_NEEDS_USER
                    ps.issues.append(f"特殊页（{kind}）翻译失败，保留原文。")
            else:
                ps.status = STATUS_NEEDS_USER   # keep / skip → left as the source
        self.log(f"  [特殊页] 全部 {len(special)} 页处理完毕。")

    def _interpret_answer(self, answer: Any, kind: str) -> str:
        """Interpret a special-page answer into ``translate`` / ``keep`` / ``skip``.

        An injected ``self.interpret`` (the AI reading the user's free text) wins;
        otherwise the flexible ``interpret_decision`` matcher handles the button values
        and common phrasings.  A malformed AI result degrades to the matcher.
        """
        v = str(answer or "").strip()
        if self.interpret is not None:
            try:
                d = str(self.interpret(v, kind)).strip().lower()
                if d in ("translate", "keep", "skip"):
                    return d
            except Exception:  # noqa: BLE001 — a bad AI read degrades to the matcher
                pass
        return interpret_decision(v)

    def _translate_special_page(self, page_index: int, kind: str) -> bool:
        """Translate a special page; return ``True`` on success (or ``False`` on failure/cancel).

        Returns success so ``_special_pages`` can mark the page ``needs_user`` rather
        than silently ``done`` when the translation actually failed (fail-closed to
        the source text).
        """
        try:
            rs = run_flow(STANDARD_FLOWS["translate_page"],
                          run_agent=self._page_agent(page_index), cancel=self.cancel,
                          log=self.log,
                          params={"page": page_index, "lang": self.state.lang, "kind": kind})
        except FlowCancelled:
            raise _tr.TranslationCancelled()
        except Exception as exc:  # noqa: BLE001 — fail-closed to source
            self.log(f"  第 {page_index + 1} 页翻译失败：{type(exc).__name__}: {exc}（保留原文）。")
            return False
        if rs.ok:
            # Same fail-closed guard as ``_translate_normal``: a page with
            # translatable blocks but no AI translation is a failed translation, not
            # "done" (the special page would export as the untranslated source).
            translatable, translated = _page_translation_counts(self.state, page_index)
            if translatable > 0 and translated == 0:
                self.log(f"  第 {page_index + 1} 页未产生任何译文（AI 未执行翻译），保留原文。")
                return False
            return True
        self.log(f"  第 {page_index + 1} 页翻译失败：{rs.error}（保留原文）。")
        return False

    def _completed(self) -> None:
        d = self.state.doc_info
        if d is not None:
            self.state.summary = (
                f"已完成正常页 {d.text_pages}/{d.pages}，特殊页 {d.special_pages} 页待协商。"
            )
            self.log(f"  [完成] {self.state.summary}")

    def _answer_value(self, val) -> str:
        """Normalise an ``answer_handler`` result to a trimmed string."""
        value = (val or {}).get("value") if isinstance(val, dict) else val
        return str(value or "").strip()

    def _review(self) -> None:
        """M4 REVIEW: pick AI self-check / manual check, then confirm export.

        After every page is drafted the session surfaces the review-mode question
        (``review_mode_question``).  ``ai`` runs a per-page self-check-fix pass;
        ``user`` lets the user inspect via the preview / sidebar (M5/M6 tools already
        exist).  Either way the session then confirms export before completing —
        a "继续检查" answer keeps the session in manual-review mode (the worker just
        exports afterwards; a non-exporting review can be resumed from the chat).
        """
        mode = "ai"
        if self.answer_handler is not None:
            try:
                q, opts = prompts.review_mode_question()
                ans = self._answer_value(self.answer_handler(q, list(opts), "review_mode"))
                if ans in ("我手动检查", "user"):
                    mode = "user"
            except Exception as exc:  # noqa: BLE001 — fail-closed to AI self-check
                self.log(f"  复核模式询问失败：{type(exc).__name__}: {exc}（按 AI 自检处理）。")
        self.state.review_mode = mode
        if mode == "ai":
            self._ai_self_check()
        else:
            self.log("  [复核] 手动检查：请在预览/侧边栏查看并标注，检查完成后确认导出。")
        # Confirm export (fail-closed to export).
        export = True
        if self.answer_handler is not None:
            try:
                q, opts = prompts.review_export_question()
                ans = self._answer_value(self.answer_handler(q, list(opts), "export"))
                if ans in ("继续检查", "continue"):
                    export = False
            except Exception as exc:  # noqa: BLE001 — fail-closed to export
                self.log(f"  复核导出确认失败：{type(exc).__name__}: {exc}（按导出处理）。")
        if not export:
            self.state.review_mode = "user"
            self.log("  [复核] 用户选择继续检查：可在侧边栏/预览继续，然后重新导出。")

    def _ai_self_check(self) -> None:
        """M4 AI_SELFCHECK: deterministic audit first, then fix findings via the agent.

        Only pages that were actually translated are re-checked.  Pages the user chose
        to keep / skip (``triage.decision`` in keep/skip) carry the source text verbatim
        — there is no AI translation to review there, and re-checking them would
        (wrongly) try to translate the intentionally-kept original.  Honouring the
        user's manual choice, those pages are skipped.

        A page is first audited deterministically (``self.audit``); if it is already
        clean by the four review criteria, the page is done without spending the AI's
        budget.  Otherwise the findings are injected as concrete data into the agent
        task (``prompts.review_page_task(..., findings=...)``) so the model fixes
        exactly what was reported, then the page is re-audited — the loop repeats up to
        ``REVIEW_MAX_ATTEMPTS`` until the audit comes back clean (fail-closed to keep the
        best translation).  This removes the old "model must remember to run every check"
        failure mode.
        """
        if self.include_kept:
            kept: set[int] = set()
        else:
            kept = {i for i, t in self.state.triage.items()
                    if t.decided and t.decision in ("keep", "skip")}
        pages = [i for i in self.state.triage if i not in kept]
        total = len(pages)
        if not total:
            self.log("  [复核] 无待复核页（全部为保留/跳过页）。")
            return
        if kept:
            self.log(f"  [复核] AI 自检：跳过 {len(kept)} 页用户选择保留/跳过（{sorted(kept)}）。")
        self.log(f"  [复核] AI 自检：先对 {total} 页做确定性审计，再对有问题页逐页修正。")
        for done, i in enumerate(pages, start=1):
            if self.cancel():
                raise _tr.TranslationCancelled()
            try:
                rs = run_flow(
                    STANDARD_FLOWS["self_check_page"],
                    tools={"audit_page": lambda page=None, checks=None: self.audit(page, checks)},
                    run_agent=self._page_agent(i), cancel=self.cancel, log=self.log,
                    params={"page": i, "checks": None, "auto_fix": True,
                            "max_iter": REVIEW_MAX_ATTEMPTS},
                )
            except FlowCancelled:
                raise _tr.TranslationCancelled()
            except Exception as exc:  # noqa: BLE001 — a failed review is not fatal
                self.log(f"  第 {i + 1} 页自检失败：{type(exc).__name__}: {exc}（保留译文）。")
                self.state.page(i).issues.append(f"自检失败：{type(exc).__name__}")
                self.progress(done, total, "复核")
                continue
            # ``run_flow`` fail-closed: a non-``ok`` result (e.g. the audit tool itself
            # failed) must NOT be treated as "clean" — that would silently pass a page
            # whose review never actually ran.
            if not rs.ok:
                self.log(f"  第 {i + 1} 页自检未完成：{rs.error}（保留译文）。")
                self.state.page(i).issues.append(f"自检未完成：{rs.error}")
                self.progress(done, total, "复核")
                continue
            audit = rs.result.get("audit_page", {})
            if audit.get("clean", True):
                self.log(f"  第 {i + 1} 页复核通过。")
                self.state.page(i).issues.append("已复核")
            else:
                n = len(audit.get("issues", []))
                self.log(f"  第 {i + 1} 页仍有 {n} 处问题（达到复核上限）。")
                self.state.page(i).issues.append(f"已复核（仍 {n} 处问题）")
            self.progress(done, total, "复核")
