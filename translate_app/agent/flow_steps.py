"""Flow: a composable, named sequence of steps (Phase 2 skeleton).

A **Flow** is a first-class composite: one or more **steps**, each being a tool
call (``ToolStep``), a bounded sub-agent loop (``AgentStep``), a user interaction
(``UserStep``) or a loop/branch (``LoopStep`` / ``IfStep``).  This is the
formalisation of "流程 = 由一个或多个工具调用组合而成的复合功能" — the AI (or the
app) executes the steps to complete a relatively independent compound function.

Two sources of truth (see ``docs/0.3.4-流程设计.md``):

* **Standard flows** come from the app config — :data:`STANDARD_FLOWS` — a fixed,
  reproducible skeleton with the AI only filling the decision points.
* **User-custom flows** are compiled from a user requirement (parameterise a
  standard flow, or freely compose tools) and may be promoted to a named standard
  flow.

This module is the **skeleton only**: it defines the step model, the executor
(:func:`run_flow`) and the registry.  It is deliberately decoupled from
``WorkflowState`` / ``make_page_executors`` — the *caller* binds a tool map
(``tools``), an agent runner (``run_agent``) and a user channel (``ask``) into
``run_flow``.  Nothing in the existing pipeline is force-migrated yet (Phase 2 is
"register, don't migrate"); later phases wire ``DocumentSession`` through this.

Step semantics
--------------

* ``ToolStep`` — one deterministic tool call (``tools[name](**params)``).
* ``AgentStep`` — one bounded sub-agent loop (delegated to ``run_agent``).
* ``UserStep`` — block on a user decision (``ask(question, options, target)``).
* ``LoopStep`` — run ``body`` until ``until(FlowRunState)`` is true or ``max_iter``.
* ``IfStep`` — run ``then`` / ``else`` by ``cond(FlowRunState)``.

``{{name}}`` placeholders in string params / tasks are resolved from ``params``
(``flow.params`` merged with the per-run ``params``) so a parameterised flow
(e.g. ``self_check_page``) can be reused for different pages / checks.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..control import ControlSignal

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([\w.-]+)\s*\}\}")


def _resolve(value: Any, params: dict[str, Any]) -> Any:
    """Recursively substitute ``{{name}}`` placeholders from ``params``.

    When a string is exactly one placeholder, the raw (type-preserving) param value is
    returned — so ``{"page": "{{page}}"}`` with ``params["page"]=3`` yields ``3``, not
    ``"3"``.  A placeholder embedded in prose becomes ``str(value)``.
    """
    if isinstance(value, str):
        m = _PLACEHOLDER_RE.fullmatch(value.strip())
        if m:
            return params.get(m.group(1), value)
        return _PLACEHOLDER_RE.sub(lambda m: str(params.get(m.group(1), m.group(0))),
                                   value)
    if isinstance(value, dict):
        return {k: _resolve(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, params) for v in value]
    return value


# --------------------------------------------------------------------------
# Step model.
# --------------------------------------------------------------------------

class FlowStep:
    """Marker base for one executable flow step.

    A plain (non-dataclass) base so each concrete step declares its own fields with
    the required (default-less) ones first — a dataclass base with defaulted fields
    would break subclass field ordering (non-default must precede default).
    """


@dataclass
class ToolStep(FlowStep):
    """One deterministic tool call (the default ``label`` is the tool name)."""

    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    kind: str = "tool"

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.tool


@dataclass
class AgentStep(FlowStep):
    """One bounded sub-agent loop.  ``task`` may be a string (resolved from
    ``params``) or a ``callable(FlowRunState, params) -> str`` so a step can build
    the task from the results accumulated so far (e.g. inject audit findings)."""

    task: str | Callable[[Any, dict[str, Any]], str]
    page: int | None = None
    max_steps: int | None = None
    image: bool = False
    label: str = ""
    kind: str = "agent"


@dataclass
class UserStep(FlowStep):
    """Block on a user decision (delegated to ``ask``)."""

    question: str
    options: list[str] | None = None
    target: str = ""
    label: str = ""
    kind: str = "user"


@dataclass
class LoopStep(FlowStep):
    """Run ``body`` until ``until`` is true or ``max_iter`` is reached."""

    body: list[FlowStep]
    until: Callable[["FlowRunState"], bool] | None = None
    max_iter: int = 3
    label: str = ""
    kind: str = "loop"


@dataclass
class IfStep(FlowStep):
    """Run ``then`` when ``cond`` is true, else ``else_``."""

    cond: Callable[["FlowRunState"], bool]
    then: list[FlowStep]
    else_: list[FlowStep] | None = None
    label: str = ""
    kind: str = "if"


@dataclass
class ForEachPage(FlowStep):
    """Run ``body`` once per page in ``pages``, binding ``{{page}}`` to each page.

    ``pages`` is either a ``list[int]`` (or ``"{{pages}}"`` resolved from ``params``)
    — the page indices the body iterates over.  Each iteration runs ``body`` with
    ``params["page"]`` set to that page, so a per-page flow (e.g. ``self_check_page``)
    can be driven across a whole document with one declarative loop.
    """

    body: list[FlowStep]
    pages: list[int] | str = "{{pages}}"
    label: str = ""
    kind: str = "foreach_page"


@dataclass
class Flow:
    """A named, composable sequence of steps (the "流程" itself)."""

    name: str
    description: str
    steps: list[FlowStep]
    params: dict[str, Any] = field(default_factory=dict)
    guards: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowRunState:
    """What a run executed (for branches/loops, audit and tests)."""

    applied: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    steps: int = 0
    error: str = ""

    def record(self, label: str, data: Any = None) -> None:
        self.applied.append(label)
        self.steps += 1
        if data is not None:
            self.result[label] = data

    @property
    def ok(self) -> bool:
        return not self.error


class FlowCancelled(ControlSignal):
    """Raised when a user cancellation stops a flow run (control signal — the
    caller catches it and reports "已取消", mirroring ``TranslationCancelled``)."""


class FlowBudgetExceeded(Exception):
    """Raised when a run exceeds its ``max_steps`` budget (fail-closed stop)."""


# --------------------------------------------------------------------------
# Executor.
# --------------------------------------------------------------------------

class _Executor:
    """Walk a step list, delegating work to the injected channels."""

    def __init__(self, *, tools=None, run_agent=None, ask=None,
                 log: Callable[[str], None] | None = None,
                 cancel: Callable[[], bool] | None = None,
                 max_steps: int | None = None,
                 report: Callable[[str, int, int, int], None] | None = None) -> None:
        self.tools = tools or {}
        self.run_agent = run_agent
        self.ask = ask
        self.log = log or (lambda _m: None)
        self.cancel = cancel
        self.max_steps = max_steps
        #: ``report(phase, done, total, page)`` is called once per page by a
        #: ``ForEachPage`` so an orchestrator can surface per-page progress / status.
        self.report = report

    def run(self, steps: list[FlowStep], rs: FlowRunState,
            params: dict[str, Any]) -> FlowRunState:
        for step in steps:
            if self.cancel is not None and self.cancel():
                raise FlowCancelled()
            if self.max_steps is not None and rs.steps >= self.max_steps:
                raise FlowBudgetExceeded()
            self._step(step, rs, params)
        return rs

    def _step(self, step: FlowStep, rs: FlowRunState, params: dict[str, Any]) -> None:
        if isinstance(step, ToolStep):
            self._tool(step, rs, params)
        elif isinstance(step, AgentStep):
            self._agent(step, rs, params)
        elif isinstance(step, UserStep):
            self._user(step, rs, params)
        elif isinstance(step, LoopStep):
            self._loop(step, rs, params)
        elif isinstance(step, IfStep):
            self._if(step, rs, params)
        elif isinstance(step, ForEachPage):
            self._foreach_page(step, rs, params)
        else:  # A bare FlowStep / unknown — never crash the flow
            rs.error = f"unknown step: {type(step).__name__}"

    def _tool(self, step: ToolStep, rs: FlowRunState, params: dict[str, Any]) -> None:
        args = _resolve(step.params, params)
        fn = self.tools.get(step.tool)
        label = step.label or step.tool
        if fn is None:
            rs.record(label)
            rs.error = f"unknown tool: {step.tool}"
            return
        # Surface the tool call as it starts and report its elapsed time when done,
        # so a deterministic step (audit_page / export) is visible in the main log.
        self.log(f"  工具开始：{step.tool}")
        started = time.monotonic()
        try:
            data = fn(**args)
        except ControlSignal:
            raise                     # a control signal must never be swallowed
        except Exception as exc:  # noqa: BLE001 — fail-closed, never crash the flow
            self.log(f"  工具失败：{step.tool}（用时 {time.monotonic() - started:.2f}s）："
                     f"{type(exc).__name__}: {exc}")
            rs.record(label)
            rs.error = f"tool {step.tool} failed: {type(exc).__name__}: {exc}"
            return
        self.log(f"  工具完成：{step.tool}（用时 {time.monotonic() - started:.2f}s）")
        rs.record(label, data)

    def _agent(self, step: AgentStep, rs: FlowRunState, params: dict[str, Any]) -> None:
        # Resolve ``page`` FIRST so the default label is keyed on the real page, not a
        # leftover "{{page}}" placeholder.
        page = _resolve(step.page, params)
        label = step.label or f"agent:{page}"
        if self.run_agent is None:
            rs.record(label)
            rs.error = "agent_step unbound"
            return
        task = step.task(rs, params) if callable(step.task) else _resolve(step.task, params)
        try:
            data = self.run_agent(task=task, page=page, max_steps=step.max_steps,
                                  image=step.image)
        except ControlSignal:
            raise                     # a control signal must never be swallowed
        except Exception as exc:  # noqa: BLE001 — fail-closed
            rs.record(label)
            rs.error = f"agent step failed: {type(exc).__name__}: {exc}"
            return
        rs.record(label, data)

    def _user(self, step: UserStep, rs: FlowRunState, params: dict[str, Any]) -> None:
        target = _resolve(step.target, params)
        label = step.label or (f"user:{target}" if target else "user")
        if self.ask is None:
            rs.record(label)
            rs.error = "user_step unbound"
            return
        # ``question`` may be a resolved string or a callable returning either a string
        # or a ``(question, options)`` tuple (so e.g. a special-page negotiation can
        # build its per-kind question/options on the fly).
        question = step.question(rs, params) if callable(step.question) else _resolve(step.question, params)
        options = step.options
        if isinstance(question, tuple):
            question, options = question[0], list(question[1] or options or [])
        try:
            value = self.ask(question, options, target)
        except ControlSignal:
            raise                     # a control signal must never be swallowed
        except Exception as exc:  # noqa: BLE001 — fail-closed
            rs.record(label)
            rs.error = f"user step failed: {type(exc).__name__}: {exc}"
            return
        rs.record(label, value)

    def _loop(self, step: LoopStep, rs: FlowRunState, params: dict[str, Any]) -> None:
        iterations = int(_resolve(step.max_iter, params))
        i = 0
        while i < iterations:
            if step.until is not None and step.until(rs):
                break
            self.run(step.body, rs, params)
            i += 1

    def _if(self, step: IfStep, rs: FlowRunState, params: dict[str, Any]) -> None:
        branch = step.then if step.cond(rs) else (step.else_ or [])
        self.run(branch, rs, params)

    def _foreach_page(self, step: ForEachPage, rs: FlowRunState,
                      params: dict[str, Any]) -> None:
        pages = _resolve(step.pages, params)
        if isinstance(pages, (int, float)):
            pages = [int(pages)]
        if not isinstance(pages, list):
            rs.error = f"foreach_page needs a page list, got {type(pages).__name__}"
            return
        pages = [int(p) for p in pages]
        for done, page in enumerate(pages, start=1):
            page_params = dict(params)
            page_params["page"] = page
            self.run(step.body, rs, page_params)
            if self.report is not None:
                self.report(step.label or step.kind, done, len(pages), page)


#: The exported step kinds (for introspection / tests / a future UI).
STEP_KINDS = ("tool", "agent", "user", "loop", "if", "foreach_page")


def run_flow(flow: Flow, *, tools: dict[str, Callable] | None = None,
             run_agent: Callable[..., Any] | None = None,
             ask: Callable[..., Any] | None = None,
             log: Callable[[str], None] | None = None,
             cancel: Callable[[], bool] | None = None,
             max_steps: int | None = None,
             report: Callable[[str, int, int, int], None] | None = None,
             params: dict[str, Any] | None = None) -> FlowRunState:
    """Execute a :class:`Flow` with the caller-supplied channels and return the run state.

    ``tools`` is a ``{name: callable(**args)}`` bound set (normally
    ``make_source_tools`` + ``make_page_executors`` over the live ``WorkflowState``);
    ``run_agent`` is the ``AgentStep`` executor (normally ``run_page_visual``); ``ask``
    is the ``UserStep`` channel (normally ``answer_handler``).  ``params`` override
    ``flow.params`` for ``{{name}}`` placeholder resolution.  ::func:`run_flow` never
    writes to the source and never crashes the pipeline — a bad tool/step sets
    ``FlowRunState.error`` (fail-closed), and ``FlowCancelled`` propagates to the caller
    as a control signal.
    """
    merged = dict(flow.params)
    if params:
        merged.update(params)
    rs = FlowRunState()
    executor = _Executor(tools=tools, run_agent=run_agent, ask=ask, log=log,
                         cancel=cancel, max_steps=max_steps, report=report)
    try:
        executor.run(flow.steps, rs, merged)
    except ControlSignal:
        raise                      # control signal — the caller decides
    except FlowBudgetExceeded:
        rs.error = "流程执行超出预算上限"
    return rs


# --------------------------------------------------------------------------
# The standard-flow registry (Phase 2: register as data — nothing is migrated yet).
# --------------------------------------------------------------------------

def _has_findings(rs: FlowRunState) -> bool:
    audit = rs.result.get("audit_page")
    return bool(audit and audit.get("issues"))


def _clean(rs: FlowRunState) -> bool:
    audit = rs.result.get("audit_page")
    return bool(audit) and not audit.get("issues")


def _review_task(rs: FlowRunState, params: dict[str, Any]) -> str:
    """Build the M4 review task, injecting the audit findings accumulated so far."""
    from .. import prompts

    page = int(params.get("page", 0))
    findings = rs.result.get("audit_page")
    return prompts.review_page_task(page, findings=findings, auto_fix=bool(params.get("auto_fix", True)))


def make_self_check_page() -> Flow:
    """P6 self_check_page: deterministic audit → fix findings → re-audit (see docs/0.3.4).

    The fix (``AgentStep``) lives INSIDE the ``LoopStep``: each round audits, fixes any
    reported findings, and the next round re-audits — so a fix that leaves residual
    issues is fixed again rather than abandoned after a single pass (the old structure
    ran one fix then only re-audited, spending ``max_iter`` re-checking without ever
    re-fixing).  ``max_iter`` bounds the number of audit→fix rounds.
    """
    return Flow(
        name="self_check_page",
        description="对一页做确定性审计，并把发现的问题交给 AI 就地修正；只复核真正翻译过的页。",
        params={"page": 0, "checks": None, "auto_fix": True, "max_iter": 3},
        steps=[
            LoopStep(until=_clean, max_iter="{{max_iter}}", body=[
                ToolStep("audit_page", {"page": "{{page}}", "checks": "{{checks}}"}),
                IfStep(cond=_has_findings, then=[
                    AgentStep(task=_review_task, page="{{page}}", image=True),
                ]),
            ]),
        ],
        guards={"protect": True, "scope": "translated_pages"},
    )


def make_preprocess() -> Flow:
    """P1 preprocess: source-document info + per-page triage."""
    return Flow(
        name="preprocess",
        description="文档信息分型：页数/类型/块数。",
        params={"page": 0},
        steps=[
            ToolStep("get_doc_info", {}),
            ToolStep("classify_page", {"page": "{{page}}"}),
        ],
    )


def make_export() -> Flow:
    """P10 export: write the translated output (a deterministic ToolStep flow)."""
    return Flow(
        name="export",
        description="把当前译文导出为指定格式（确定性地写文件）。",
        params={"output_type": "translated_pdf", "output_path": ""},
        steps=[
            ToolStep("export", {"output_type": "{{output_type}}", "output_path": "{{output_path}}"}),
        ],
        guards={"deterministic": True},
    )


def _page_task_builder(rs: FlowRunState, params: dict[str, Any]) -> str:
    """The per-page translation task (``prompts.page_task``) for the bound page/kind."""
    from .. import prompts

    page = int(params.get("page", 0))
    return prompts.page_task(page, str(params.get("lang", "")), kind=params.get("kind"))


def _special_question_builder(rs: FlowRunState, params: dict[str, Any]) -> tuple[str, list[str]]:
    """Per-kind special-page question + options (``prompts.special_page_question``)."""
    from .. import prompts

    page = int(params.get("page", 0))
    kind = str(params.get("kind", "scan"))
    return prompts.special_page_question(page, kind)


def _answered(rs: FlowRunState):
    """The last recorded user answer (the value dict from ``answer_handler``)."""
    for v in rs.result.values():
        if isinstance(v, dict) and "value" in v:
            return v
    return None


def interpret_decision(answer: Any) -> str:
    """Interpret a user's special-page answer into ``translate`` / ``keep`` / ``skip``.

    Replaces the old exact-string match with a flexible matcher that also handles the
    free-text answers typed into the sidebar field (an injected ``interpret`` channel can
    override this with a real LLM reading).  Order matters: negation / keep-intent is
    recognised before the bare "翻译" token (so "不翻译" → keep, not translate).
    """
    v = str(answer or "").strip().lower()
    if any(k in v for k in ("跳过", "略过", "停")) or v.startswith("skip") or v in ("skipped", "none"):
        return "skip"
    # "保留公式/图表并翻译说明/图注/文字" explicitly asks to translate the prose /
    # captions while KEEPING the structural blocks.  On a special page that is
    # ``translate`` (the engine keeps formula / figure / chart blocks verbatim), so it
    # must NOT be swallowed by the lone "保留" keep-match below.
    if "保留" in v and any(k in v for k in ("翻译", "译", "说明", "图注", "表注", "文字", "标题")):
        return "translate"
    if any(k in v for k in ("保留", "原样", "不动", "不翻", "别译", "不用", "维持")):
        return "keep"
    if any(k in v for k in ("翻译", "ocr", "转成", "译成", "换成")) or v.startswith("translate"):
        return "translate"
    # A bare fallback: don't translate what the user did not clearly ask to translate.
    return "keep"


def make_translate_page() -> Flow:
    """P2 translate_page: translate one page's text blocks into the target language."""
    return Flow(
        name="translate_page",
        description="把一页的所有文本块翻译成目标语言（有界 agent 循环）。",
        params={"page": 0, "lang": "", "kind": None},
        steps=[
            AgentStep(task=_page_task_builder, page="{{page}}", image=True),
        ],
    )


def make_translate_normal() -> Flow:
    """P2 translate_normal: translate every ``normal`` page (loop)."""
    return Flow(
        name="translate_normal",
        description="按阅读序逐页翻译所有正常文本页。",
        params={"pages": [], "lang": "", "kind": None},
        steps=[
            ForEachPage(pages="{{pages}}", body=[
                AgentStep(task=_page_task_builder, page="{{page}}", image=True),
            ]),
        ],
    )


def make_special_page() -> Flow:
    """P4 unit: negotiate one special page — ask the user (the decision + execution
    is the ``DocumentSession`` phase's job, so it can inject an AI ``interpret``)."""
    return Flow(
        name="special_page",
        description="对某个特殊页按页型询问用户处理方式（翻译/保留/跳过）。",
        params={"page": 0, "lang": "", "kind": "scan"},
        steps=[
            UserStep(question=_special_question_builder, target="page:{{page}}"),
        ],
    )


def make_special_pages() -> Flow:
    """P4 special_pages: loop over every non-normal page and ask the user."""
    return Flow(
        name="special_pages",
        description="逐特殊页询问用户处理方式（翻译/保留/跳过）。",
        params={"pages": [], "lang": "", "kind": "scan"},
        steps=[
            ForEachPage(pages="{{pages}}", body=[
                UserStep(question=_special_question_builder, target="page:{{page}}"),
            ]),
        ],
        guards={"protect": True},
    )


def make_translate_doc() -> Flow:
    """P3: the standard top-level orchestration flow (the phase ORDER as data).

    Preprocess is a prerequisite (it computes the page sets above) and stays in
    ``DocumentSession``; this flow drives the downstream phases whose page sets are
    known once preprocess ran — normal translation, then special-page negotiation.

    **The review/self-check gate and the export confirmation are deliberately NOT part
    of the standard flow** (decoupled): "开始翻译" runs translation and reports; a
    self-check only happens when the user explicitly asks (a custom audit/fix
    requirement routes to ``ai_self_check`` / ``self_check_page``, or via the chat's
    ``self_check`` / ``run_flow``).  So the standard flow ends at ``completed`` with a
    completion report, not a "是否自检？" prompt.
    """
    return Flow(
        name="translate_doc",
        description="整篇翻译标准流程：正常页→特殊页协商→完成报告。复核/自检按用户自定义要求另行触发。",
        params={"normal_pages": [], "special_pages": [], "lang": ""},
        # The top-level phase ORDER, declared as data so ``DocumentSession.run`` is a thin
        # dispatcher over it (rather than a hardcoded call sequence).  Preprocess stays a
        # prerequisite (it computes the page sets the steps below need).
        scope={"phases": ["preprocess", "translate_normal", "special_pages", "completed"]},
        steps=[
            ForEachPage(pages="{{normal_pages}}", body=[
                AgentStep(task=_page_task_builder, page="{{page}}", image=True),
            ]),
            ForEachPage(pages="{{special_pages}}", body=[
                UserStep(question=_special_question_builder, target="page:{{page}}"),
            ]),
        ],
    )


def make_ai_self_check() -> Flow:
    """P5 ai_self_check: run the self-check gate over every translated page."""
    return Flow(
        name="ai_self_check",
        description="对每个已译页跑确定性审计并就地修正（跳过用户保留/跳过的页）。",
        params={"pages": [], "checks": None, "auto_fix": True, "max_iter": 3},
        steps=[
            ForEachPage(pages="{{pages}}", body=make_self_check_page().steps),
        ],
    )


#: The app's standard flows (data; DocumentSession drives the per-page unit flows
#: through ``run_flow`` once Ph3 wires it up).
STANDARD_FLOWS: dict[str, Flow] = {
    "preprocess": make_preprocess(),
    "translate_doc": make_translate_doc(),
    "translate_page": make_translate_page(),
    "translate_normal": make_translate_normal(),
    "special_page": make_special_page(),
    "special_pages": make_special_pages(),
    "self_check_page": make_self_check_page(),
    "ai_self_check": make_ai_self_check(),
    "export": make_export(),
}
