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
from typing import Any, Callable

from .state import Goal, WorkflowState
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
                 answer_handler: Callable[[str, list[str], str], Any] | None = None) -> None:
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
            lines.append(f"last={r.op_tool} ok={r.ok} result={r.result} error={r.error}")
        return "\n".join(lines)

    def _call(self, name: str, args: dict) -> AgentResult:
        fn = self.tools.get(name)
        if fn is None:
            return AgentResult(ok=False, op_tool=name, op_args=args,
                               error=f"unknown tool: {name}")
        try:
            return AgentResult(ok=True, op_tool=name, op_args=args, result=fn(**args))
        except Exception as exc:  # noqa: BLE001 — fail-closed, never crash the loop
            return AgentResult(ok=False, op_tool=name, op_args=args,
                               error=f"{type(exc).__name__}: {exc}")

    def _apply(self, dec: Decision, res: AgentResult) -> None:
        self.state.record_op(dec.tool, args=dec.arguments, reason=dec.summary or "agent")
        self.state.log.append(f"{dec.tool} -> ok={res.ok} {res.error or res.result}")
        self.log(f"  [{dec.tool}] ok={res.ok}{' ' + res.error if res.error else ''}")

    def step(self) -> Decision:
        """Run one controller step and return the decision it made."""
        obs = self.observe()
        dec = self.decide(obs, self.state, self.last_result)
        if dec.action == "call":
            res = self._call(dec.tool, dec.arguments)
            self.last_result = res
            self._apply(dec, res)
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
    log: Callable[[str], None] | None = None,
) -> WorkflowState:
    """Convenience wrapper: build a ``FlowAgent`` and run it on ``state``.

    ``tools`` / ``decide`` are required for a real run (the LLM).  Kept optional
    here so ``run_agent_run(state)`` can still express the intent; a caller that
    omits them must set them via the returned agent otherwise the loop does
    nothing (empty tools → every call is ``unknown tool``).
    """
    agent = FlowAgent(state, tools or {}, decide, log=log, max_steps=max_steps,
                      answer_handler=answer_handler)
    return agent.run(max_rounds=max_rounds)


# --------------------------------------------------------------------------
# P3: real-LLM ``decide`` + a single-page visual closed loop.
# --------------------------------------------------------------------------

import base64
import json

from .. import translator as _tr


def _image_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


#: Rules appended to the agent's system prompt so the model knows *when* to ask
#: the user (``ask_user``).  This is the "该问就问" trigger — the model calls the
#: ``ask_user`` tool at these decision points and the sidebar answers it.
_INTERACTION_RULES = (
    "\n\n【与用户交互】在以下情况，必须先调用 ask_user 获取用户决定，再继续：\n"
    "1) 术语/专有名词/报表科目名翻译不确定时（给出候选译文让用户选）。\n"
    "2) 某块该保留原文还是翻译、判断不确定时（给出『保留/翻译』）。\n"
    "3) 需要删除/覆盖/擦除译文或页（不可逆/破坏性）时（先确认）。\n"
    "4) 接近预算上限、或当前无法推进时（询问是否收尾或继续）。\n"
    "没有上述情况不要打扰用户。ask_user 的结果会作为工具结果返回，请据此继续。"
)


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


def make_llm_decide(model, *, task: str, image_provider=None,
                    log: Callable[[str], None] | None = None,
                    max_tokens: int = 4096):
    """Return a real-LLM ``decide(obs, state, last_result) -> Decision``.

    Wraps an OpenAI-compatible ``chat.completions`` call with the agent's tool
    schemas (``tool_choice="auto"``), a system prompt (``task``) and an optional
    page image (``image_provider(state) -> png``) so the model can *see* the
    source page.  Tool results are fed back as ``tool`` role messages so the loop
    is a genuine multi-turn tool-use conversation.  A model that stops making
    tool calls yields ``Decision(action="done")``.
    """
    if not getattr(model, "vision", False):
        return None
    client = _tr.OpenAI(**model.client_kwargs())
    specs = agent_openai_tools()
    messages: list[dict] = [{"role": "system", "content": str(task) + _INTERACTION_RULES}]
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
                        {"ok": last_result.ok, "result": last_result.result,
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
        if last_result is not None and last_result.op_tool == "preview_page":
            result = last_result.result
            img = result.get("image") if isinstance(result, dict) else None
            if img:
                content.append({"type": "image_url",
                                "image_url": {"url": _image_url(img)}})
        messages.append({"role": "user", "content": content})
        try:
            resp = client.chat.completions.create(
                model=model.model, temperature=0.0, max_tokens=max_tokens,
                tools=specs, tool_choice="auto", messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 — fail-closed
            if log:
                log(f"  decide 调用失败：{type(exc).__name__}: {exc}")
            return Decision(action="done", summary=f"error: {exc}")
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


def make_source_tools(state: WorkflowState, *, src_path=None) -> dict[str, Callable]:
    """Read-only tools bound to the immutable source (never write to it).

    ``read_page`` / ``get_layout`` return JSON-serialisable observations so the
    controller can reason about a page; neither mutates the original.
    """
    src_path = src_path or state.src_path

    def read_page(page: int) -> dict[str, Any]:
        src = state.src_doc
        blocks = [] if src is None else src.pages[page]
        return {"blocks": [
            {"index": i, "text": b.text, "bbox": [b.x0, b.y0, b.x1, b.y1],
             "in_table": bool(getattr(b, "in_table", False)),
             "is_chart": bool(getattr(b, "is_chart", False))}
            for i, b in enumerate(blocks)
        ]}

    def get_layout(page: int) -> dict[str, Any]:
        src = state.src_doc
        blocks = [] if src is None else src.pages[page]
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

    return {"read_page": read_page, "get_layout": get_layout}


def _has_cjk(text: Any) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text))


def _target_is_cjk(state: WorkflowState) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in state.lang)


def make_page_executors(state: WorkflowState, model, log: Callable[[str], None] | None = None,
                        preview_handler: Callable[..., bytes] | None = None,
                        answer_handler: Callable[..., Any] | None = None) -> dict[str, Callable]:
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
    """
    from pathlib import Path

    from .. import pdfio
    from .. import translator as _tr

    engine = _tr.TranslationEngine(model)
    retranslate = _tr.make_retranslate_fn(model, log)

    def _flat_blocks() -> list:
        if state.src_doc is None:
            return []
        return [b for page in state.src_doc.pages for b in page]

    def _block(index: int):
        blocks = _flat_blocks()
        return blocks[index] if 0 <= index < len(blocks) else None

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
                                                 doc_path=Path(state.src_path))
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
        src = str(text) if text is not None else str(b.text)
        translated = _translate(src, target_lang or state.lang)
        _write(index, translated)
        return {"ok": True, "index": index, "translated": translated}

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

    def create_block(page: int, index: int, text: str, bbox=None):
        _write(index, text)
        return {"ok": True, "index": index}

    def move_block(page: int, index: int, to_index: int):
        o = _out()
        if index in o:
            entry = o.pop(index)
            o[to_index] = entry
            return {"ok": True, "index": to_index}
        return {"ok": False, "error": f"no translation for index {index}"}

    def retranslate_block(text: str, target_lang: str | None = None):
        if retranslate is None:
            return {"ok": False, "error": "重译不可用（非视觉模型）"}
        return {"ok": True, "text": retranslate(str(text), target_lang or state.lang)}

    def rewrite_block(index: int, text: str | None = None, instruction: str = ""):
        b = _block(index)
        if b is None:
            return {"ok": False, "error": f"bad index {index}"}
        src = str(text) if text is not None else str(b.text)
        if pdfio._is_numeric_cell(str(b.text)):
            return {"ok": False, "error": "数字/代码块不可被 AI 改写（保真）"}
        new = _translate(src, state.lang)
        _write(index, new)
        return {"ok": True, "index": index, "text": new}

    def apply_terminology(source: str, target: str):
        state.user_decisions.setdefault("terminology", {})[str(source)] = str(target)
        return {"ok": True}

    def set_font(page: int, index: int, size: float):
        try:
            size = float(size)
        except (TypeError, ValueError):
            return {"ok": False, "error": "size 不是数字"}
        size = max(6.0, min(40.0, size))
        _out().setdefault(index, {})["size"] = size
        return {"ok": True, "size": size}

    def set_align(page: int, index: int, align: str):
        if align not in ("left", "center", "right"):
            return {"ok": False, "error": "bad align"}
        _out().setdefault(index, {})["align"] = align
        return {"ok": True, "align": align}

    def check_residual(page: int | None = None):
        out = []
        for idx, b in enumerate(_flat_blocks()):
            if b.is_chart:
                continue
            t = _read(idx)
            if not t.strip():
                out.append({"index": idx, "text": str(b.text), "reason": "empty"})
            elif not _target_is_cjk(state) and _has_cjk(t):
                out.append({"index": idx, "text": str(b.text), "reason": "residual_cjk"})
        return {"residual": out}

    def check_missing(page: int | None = None):
        out = []
        for idx, b in enumerate(_flat_blocks()):
            if str(b.text).strip() and not _read(idx).strip():
                out.append({"index": idx, "text": str(b.text)})
        return {"missing": out}

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

    return {
        "translate_block": translate_block,
        "retranslate_block": retranslate_block,
        "rewrite_block": rewrite_block,
        "set_text": set_text,
        "delete_block": delete_block,
        "create_block": create_block,
        "move_block": move_block,
        "apply_terminology": apply_terminology,
        "set_font": set_font,
        "set_align": set_align,
        "check_residual": check_residual,
        "check_missing": check_missing,
        "preview_page": preview_page,
        "ask_user": ask_user,
    }


def run_page_visual(state: WorkflowState, page_index: int, model,
                    tools: dict[str, Callable] | None = None, *,
                    task: str, log: Callable[[str], None] | None = None,
                    max_steps: int | None = None, max_rounds: int | None = None,
                    src_path=None, preview_handler: Callable[..., bytes] | None = None,
                    answer_handler: Callable[[str, list[str], str], Any] | None = None) -> WorkflowState:
    """Run the AI-orchestrated loop on a single page (the visual closed loop).

    Renders the source page once, hands it to a real-LLM ``decide`` (image +
    tool schemas), and drives ``FlowAgent``: the model observes the page, calls
    tools (read the source / classify / verify / translate), reads the results,
    verifies and recurses until it stops or the budget is spent.  ``state.src_doc``
    should already be populated (extracted blocks) so ``read_page`` works.

    ``preview_handler`` (e.g. ``preview.PreviewBridge.get_region``) lets the agent's
    ``preview_page`` tool show a page to the user and receive the framed region back.
    """
    state.page(page_index)
    if not any(g.page == page_index for g in state.todo):
        state.todo.append(Goal(kind="page", page=page_index))
    src_path = src_path or state.src_path

    def image_provider(_s: WorkflowState) -> bytes:
        try:
            return _render_source_page(src_path, page_index)
        except Exception:  # noqa: BLE001 — a bad source page must not crash the loop
            return b""

    decide = make_llm_decide(model, task=task,
                             image_provider=image_provider, log=log)
    if decide is None:
        state.log.append("模型不支持视觉，无法执行 AI 编排（回退确定性流水线）。")
        if log:
            log("  [agent] 模型不支持视觉，单页编排跳过。")
        return state
    all_tools = dict(make_source_tools(state, src_path=src_path))
    all_tools.update(make_page_executors(state, model, log=log,
                                         preview_handler=preview_handler,
                                         answer_handler=answer_handler))
    if tools:
        all_tools.update(tools)
    agent = FlowAgent(state, all_tools, decide, log=log, max_steps=max_steps,
                      answer_handler=answer_handler)
    return agent.run(max_rounds=max_rounds)
