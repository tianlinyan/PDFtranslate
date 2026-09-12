"""U1: user-custom flows — a parameterised standard flow (Path A) + promotion.

A **user-custom flow** is a user requirement turned into a parameterised standard
flow: the app skeleton stays fixed, only the "knobs" change (which checks, which
pages, whether to include kept pages, whether to only report, ...).  This is the
robust default (Path A); free AI composition (Path B) is a later / opt-in path.

Lifecycle:

1. ``compile_from_user(req) -> FlowSpec``  — parse a Chinese requirement into a
   parameterised :class:`FlowSpec` (deterministic rule-based parser; a model could
   be plugged in for slot-filling, but the rules are testable offline).
2. ``build_flow(spec) -> Flow``            — instantiate a runnable :class:`Flow`
   from the spec (clone the base standard flow, override knobs, wrap a per-page flow
   in ``ForEachPage`` when a multi-page scope is given).
3. ``save_flow_spec(name, spec)`` / ``load_user_flow_specs()`` — persist promoted
   specs so a repeatedly-used custom flow becomes a **named standard flow** (stored
   alongside the built-in ``STANDARD_FLOWS`` and recompiled at use time).
4. ``validate_flow_tools(flow, available)`` — the "先绑定后暴露" gate: a flow's
   ``ToolStep`` tools must be bound (or a known deterministic tool), never a tool the
   pipeline would answer "unknown tool".

Persistence is **env-gated** like the caches: disk writes only when
``PDFTRANSLATE_FLOWS_DIR`` is set (tests set it to a temp dir), so a production run
never writes user-flow files implicitly.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from ..control import ControlSignal
from .flow_steps import (
    Flow, STANDARD_FLOWS, ForEachPage, ToolStep, run_flow,
    registered_flow_tiers,
)
from .tool_catalog import (
    atomic_tool_names, TIER_ATOMIC,
)


@dataclass
class FlowSpec:
    """A parameterised standard flow (the U1 "custom flow" spec).

    ``base`` names the standard flow to clone (e.g. ``"self_check_page"``); the other
    fields override its knobs.  ``scope`` is a 0-based page list; ``checks`` a subset
    of the audit check names; ``include_kept`` whether kept pages are also reviewed.
    """

    base: str = "self_check_page"
    checks: list[str] | None = None        # check_* subset for an audit flow
    scope: list[int] | None = None         # 0-based page indices
    include_kept: bool = False
    auto_fix: bool | None = None           # None = keep the base default; False = read-only
    page: int | None = None                # single page (0-based)
    lang: str = ""
    kind: str | None = None                # special-page kind
    output_type: str | None = None         # export format
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "extra"} | dict(self.extra)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlowSpec":
        data = dict(d or {})
        # ``name`` is the registry key written alongside the spec by ``save_flow_spec``;
        # it is not a flow knob and must not leak into ``extra`` (which ``build_flow``
        # copies straight into the flow's params).
        extra = {k: v for k, v in data.items()
                 if k not in {"name", "base", "checks", "scope", "include_kept",
                              "auto_fix", "page", "lang", "kind", "output_type"}}
        return cls(
            base=str(data.get("base", "self_check_page")),
            checks=list(data["checks"]) if data.get("checks") is not None else None,
            scope=[int(p) for p in data["scope"]] if data.get("scope") is not None else None,
            include_kept=bool(data.get("include_kept", False)),
            auto_fix=data.get("auto_fix"),
            page=int(data["page"]) if data.get("page") is not None else None,
            lang=str(data.get("lang", "")),
            kind=data.get("kind"),
            output_type=data.get("output_type"),
            extra=extra,
        )

#: Tools a flow may legitimately reference that are NOT agent-bound (they are
#: resolved deterministically by the caller / the pipeline's own exporters).
DETERMINISTIC_TOOLS = frozenset({"export", "re_export"})

#: Base standard-flows that translate *one page* — a multi-page scope wraps them in a
#: ``ForEachPage`` so a single spec can drive a whole page range.
_PER_PAGE_BASES = frozenset({"self_check_page", "translate_page", "special_page"})

#: The persistent registry of promoted user-flow specs (in-memory + disk).
USER_FLOW_SPECS: dict[str, FlowSpec] = {}

_FLOW_DIR_ENV = "PDFTRANSLATE_FLOWS_DIR"


def user_flows_dir() -> Path:
    """Where promoted user-flow specs live (env-gated dir, else under ``~/.pdftranslate``)."""
    env = os.environ.get(_FLOW_DIR_ENV)
    if env:
        return Path(env)
    return Path.home() / ".pdftranslate" / "flows"


def _flows_persist_enabled() -> bool:
    """Whether user-flow specs persist to disk.

    Disk persistence is **opt-in** via ``PDFTRANSLATE_FLOWS_DIR`` (mirroring
    ``PDFTRANSLATE_CACHE_DIR`` on the translation side — tests set it to a temp dir
    to activate).  A production run keeps promoted flows in memory only unless the
    dir is configured, so a chat "name this flow" never writes files implicitly.
    """
    return bool(os.environ.get(_FLOW_DIR_ENV))


# --------------------------------------------------------------------------
# 1. Rule-based requirement → FlowSpec (Path A slot-filling).
# --------------------------------------------------------------------------

#: Chinese alias → check name, used to fill ``spec.checks``.  Deliberately NO bare
#: single-char aliases: "数" would also match 数据/数量/次数 and "表" would match
#: 代表/表达/表面, silently turning an unrelated sentence into a numbers/table audit.
_AUDIT_ALIASES: dict[str, str] = {
    "数字": "numbers", "金额": "numbers",
    "表格": "table",
    "版面": "layout", "布局": "layout",
    "漏译": "missing", "残留": "residual",
}

_SCOPE_RANGE_RE = re.compile(r"第?\s*(\d+)\s*页?\s*(?:-|到|~|至)\s*第?\s*(\d+)\s*页")
_SCOPE_SINGLE_RE = re.compile(r"第\s*(\d+)\s*页")

#: Cues that turn a page mention into an explicit *scope restriction*
#: (``只翻第2-5页`` / ``仅翻译第3页`` / ``只第3页``).  Deliberately excludes exclusion
#: words (``跳过``/``除了``/``不要翻``): "跳过第3页" means translate everything *except*
#: page 3, so treating it as a scope would translate only page 3.
#:
#: The cue must be a restriction *phrase*, not a bare 只/仅: 只要 / 不仅 / 不只 are
#: ordinary conjunctions, and matching them narrowed a whole-document request to a
#: single page (``帮我翻译整篇年报，只要第5页的数字没错`` → scope [4]).
#:
#: The verb list covers the phrasings users actually write — including
#: 保留/留/想要/需要/想 (``只保留第3页``): those were absent, so the request was
#: silently widened to the whole document.  ``要`` alone stays out on purpose —
#: it would make the conjunction 只要 match again.
_SCOPE_CUE_RE = re.compile(
    r"(?:只|仅)(?:翻|翻译|译|查|检查|看|审|审计|处理|导出|跑|做|改|重译"
    r"|保留|留|想要|需要|需|想)"
    r"|仅限|限定|范围"
    r"|(?<![不])[只仅]\s*第\s*\d+\s*页"
)


def _parse_checks(req: str) -> list[str] | None:
    # De-duplicate while preserving order: a word like 数字 matches both the
    # "数字" alias and the bare "数" alias, and the duplicate would make
    # ``audit_page`` run the same check twice (and report it twice).
    found: list[str] = []
    for alias, name in _AUDIT_ALIASES.items():
        if alias in req and name not in found:
            found.append(name)
    return found or None


def _parse_scope(req: str) -> list[int] | None:
    m = _SCOPE_RANGE_RE.search(req)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        return list(range(start - 1, end))          # 0-based, inclusive of ``end``
    m = _SCOPE_SINGLE_RE.search(req)
    if m:
        return [int(m.group(1)) - 1]
    return None


def parse_explicit_scope(req: str) -> list[int] | None:
    """Page scope **only** when ``req`` explicitly restricts the pages.

    :func:`_parse_scope` (used by :func:`compile_from_user`) treats *any* "第N页"
    mention as a scope, which is right for an audit request ("检查第3到第8页") but
    wrong for a translation requirement that merely *mentions* a page: a
    requirement like "帮我翻译整篇年报，第5页的图表保留原文" used to narrow the run
    to page 5, so every other page was exported untranslated while the tool still
    reported success.  A scope is therefore only derived when the text carries an
    explicit restriction cue ("只/仅/仅限/限定/范围") or an explicit range
    ("第2到第5页"); a bare single-page mention stays a per-page instruction that
    the translation agent reads from ``state.requirements``.
    """
    if _SCOPE_RANGE_RE.search(req) or _SCOPE_CUE_RE.search(req):
        return _parse_scope(req)
    return None


def _coerce_scope(raw) -> list[int] | None:
    """0-based page list from a model's ``scope`` value; ``None`` = no restriction.

    A non-list / unparsable / all-negative value is *unusable*, not a restriction:
    the caller treats it as "the whole document" (fail-open).
    """
    if not isinstance(raw, list) or not raw:
        return None
    out: list[int] = []
    for x in raw:
        if isinstance(x, bool):            # True would silently become page 1
            return None
        if isinstance(x, float) and not x.is_integer():
            return None                    # 1.5 is off-contract, not page 1
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            return None
    pages = sorted({p for p in out if p >= 0})
    return pages or None


def ai_scope(req: str, llm) -> tuple[list[int] | None, str]:
    """Read the page scope out of a requirement with the **model** (Path A).

    Returns ``(0-based pages or None, reason)``.  ``llm`` is the same AI
    slot-filler ``run_flow`` uses (``make_llm_flow_compiler``) — its JSON contract
    already carries ``scope`` plus an optional ``reason`` for the log echo.
    :func:`parse_explicit_scope` stays the caller's *offline* fallback only.

    Any failure — no client, network, malformed JSON, an unusable ``scope`` —
    degrades to ``(None, "")`` = "no restriction".  That fail-open is deliberate:
    silently translating *fewer* pages than asked for is worse than translating the
    whole document, and the caller logs the decision so the user can correct it.
    """
    try:
        data = llm(str(req or "")) or {}
    except Exception:  # noqa: BLE001 — a failing model is not a restriction
        return None, ""
    if not isinstance(data, dict):
        return None, ""
    return _coerce_scope(data.get("scope")), str(data.get("reason") or "").strip()


def _base_from(req: str, default: str) -> str:
    if "重新导出" in req or "重新生成" in req:
        return "export"
    if "重译" in req:
        return "translate_page"
    if "自检" in req or "检查" in req:
        return "self_check_page"
    return default


def _canonical_check(raw: Any) -> str:
    """Map a model-supplied check name to its registry name (aliases included).

    A Chinese alias (数字/表格/版面/漏译/残留) is the model restating the user's own
    wording, so it must become the registry name (``numbers``/…) for the check to
    actually run.  A name that is NOT recognised is passed through **unchanged** so
    ``agent.flow.audit_page`` reports it as an ``unknown_checks`` issue
    (``clean=false``) — dropping it here would silently audit nothing.
    """
    name = str(raw).strip()
    return _AUDIT_ALIASES.get(name, name.lower())


def _spec_from_ai(data: dict | None, default_base: str) -> FlowSpec:
    """Build a validated :class:`FlowSpec` from an AI slot-filler's dict.

    The AI output is a JSON-serialisable mapping of ``FlowSpec`` fields.  Unknown /
    malformed values degrade to defaults (never raise), so a bad AI response still
    yields a valid spec for the caller to confirm before running.
    """
    data = data or {}
    base = str(data.get("base") or default_base)
    if base not in STANDARD_FLOWS:
        base = default_base   # fail-closed: unknown base -> the safe default
    checks = data.get("checks")
    if isinstance(checks, str):
        checks = [checks]
    checks = ([_canonical_check(c) for c in checks] if isinstance(checks, list) else None)
    scope = data.get("scope")
    if isinstance(scope, (int, float)):
        scope = [int(scope)]
    elif isinstance(scope, list):
        scope = [int(s) for s in scope]
    else:
        scope = None
    return FlowSpec(
        base=base,
        checks=checks,
        scope=scope,
        include_kept=bool(data.get("include_kept", False)),
        auto_fix=data.get("auto_fix"),
        page=int(data["page"]) if data.get("page") is not None else None,
        lang=str(data.get("lang", "")),
        kind=data.get("kind"),
        output_type=data.get("output_type"),
        extra=dict(data.get("extra") or {}),
    )


def compile_from_user(req: str, *, default_base: str = "self_check_page",
                      llm: Callable[[str], dict] | None = None) -> FlowSpec:
    """``compile_from_user("自检只查数字和表格，第3到第8页…")`` -> FlowSpec.

    Two paths (AI-driven by default when an LLM is injected):

    * ``llm`` (optional) is an **AI slot-filler** ``callable(req) -> dict`` that
      interprets the requirement into JSON-serialisable ``FlowSpec`` fields — this
      replaces hardcoded keyword rules with the model reading arbitrary phrasing.
    * No ``llm`` → a deterministic rule parser fills ``checks``/``scope``/
      ``include_kept``/``auto_fix``/``base`` from the known patterns (the robust
      offline fallback); unrecognised text is ignored (keeps defaults).

    Either way the result is validated over defaults, so a partial/bad response
    degrades gracefully instead of failing.
    """
    r = str(req or "").strip()
    if llm is not None:
        try:
            data = llm(r) or {}
        except Exception:  # noqa: BLE001 — a failing/fake LLM degrades to defaults
            data = {}
        return _spec_from_ai(data, default_base)
    return FlowSpec(
        base=_base_from(r, default_base),
        checks=_parse_checks(r),
        scope=_parse_scope(r),
        include_kept=("保留页也算" in r or "保留" in r and "算" in r),
        # ``auto_fix`` is OPT-IN: a plain "自检…" is read-only (the tool description
        # and CLAUDE.md both promise that); the user must ask for a fix explicitly.
        auto_fix=(
            True if any(k in r for k in ("自动改", "自动修正", "自动修", "直接改",
                                         "帮我改", "修一下", "修正一下"))
            else False if any(k in r for k in ("只查", "只读", "不修改", "不改"))
            else None
        ),
    )


#: Prompt that asks the model to fill a :class:`FlowSpec` from free text (Path A's
#: flexible slot-filling branch — the deterministic rule parser is the offline fallback).
_FLOW_COMPILE_PROMPT = (
    "把下面这句话解析成一个 JSON 对象（只输出一个 JSON 对象，不要任何解释、不要 markdown 代码围栏）：\n"
    "字段（都可省略）：base = self_check_page|translate_page|export（默认 self_check_page）；"
    "checks = 字符串数组，取值 layout/residual/missing/numbers/table；"
    "scope = 0 起的页号整数数组；auto_fix = 布尔；include_kept = 布尔；"
    "reason = 一句话说明你的理解（可选，用于日志回显）。\n"
    "要求：{req}"
)


def _parse_flow_json(text: str) -> dict:
    """Extract a JSON object from a model reply (strips code fences / surrounding prose).

    Fail-closed: any malformed or non-object reply yields ``{}`` so the caller
    degrades to defaults instead of crashing.
    """
    m = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 — bad JSON degrades to defaults
        return {}


def make_llm_flow_compiler(model, client: Any = None,
                           log: Callable[[str], None] | None = None):
    """Return an AI slot-filler ``llm(req) -> dict`` for ``compile_from_user``, or ``None``.

    This is Path A's flexible branch: the model reads arbitrary phrasing and fills
    the ``FlowSpec`` fields instead of the deterministic keyword rules.  Fail-closed:
    on a network / parse error the callback returns ``{}`` (``compile_from_user`` then
    degrades to defaults), and when no usable client exists it returns ``None`` (the
    caller falls back to the rule parser).  ``client`` (optional) reuses a shared
    OpenAI client.
    """
    from .. import translator as _tr

    if client is None:
        try:
            client = _tr.OpenAI(**model.client_kwargs())
        except Exception:  # noqa: BLE001 — no client → no AI slot-filling
            return None

    def compile_req(req: str) -> dict:
        try:
            kwargs: dict[str, Any] = {
                "model": model.model,
                "temperature": 0.0,
                "max_tokens": 512,
                "messages": [{"role": "user",
                              "content": _FLOW_COMPILE_PROMPT.format(req=str(req or ""))}],
            }
            body = model.request_params()
            if body:
                kwargs["extra_body"] = body
            resp = client.chat.completions.create(**kwargs)
            text = (getattr(resp.choices[0].message, "content", "") or "").strip()
            return _parse_flow_json(text)
        except Exception as exc:  # noqa: BLE001 — fail-closed (no rule fallback)
            if log:
                log(f"  流程槽填充失败：{type(exc).__name__}: {exc}（以默认流程继续）。")
            return {}

    return compile_req


# --------------------------------------------------------------------------
# 2. FlowSpec → runnable Flow.
# --------------------------------------------------------------------------

def build_flow(spec: FlowSpec) -> Flow:
    """Instantiate a runnable :class:`Flow` from a :class:`FlowSpec` (Path A)."""
    base = STANDARD_FLOWS.get(spec.base)
    if base is None:
        raise ValueError(f"未知标准流程：{spec.base!r}")
    flow = copy.deepcopy(base)
    if spec.checks is not None:
        flow.params["checks"] = list(spec.checks)
    if spec.auto_fix is not None:
        flow.params["auto_fix"] = bool(spec.auto_fix)
    if spec.page is not None:
        flow.params["page"] = int(spec.page)
    if spec.lang:
        flow.params["lang"] = spec.lang
    if spec.kind is not None:
        flow.params["kind"] = spec.kind
    if spec.output_type:
        flow.params["output_type"] = spec.output_type
    # U1 knob: carry ``include_kept`` so a custom review flow explicitly opts into
    # reviewing pages the user chose to keep/skip.  The live M4 driver honours it via
    # ``DocumentSession(include_kept=...)`` (see ``_ai_self_check``); this keeps the
    # compiled flow's params consistent for a caller that builds a session from it.
    if spec.include_kept:
        flow.params["include_kept"] = True
    # Pass any extra knob (e.g. a ``negotiate`` flag for ``special_pages``) straight
    # through to the cloned flow's params.
    for k, v in (spec.extra or {}).items():
        flow.params[k] = v
    if spec.scope is not None:
        pages = [int(p) for p in spec.scope]
        if spec.base in _PER_PAGE_BASES and len(pages) > 1:
            # A single-page flow driven across a page range → wrap in a page loop.
            flow = Flow(
                name=f"{spec.base}_pages",
                description=base.description,
                params={**flow.params, "pages": pages},
                steps=[ForEachPage(pages="{{pages}}", body=flow.steps)],
                guards=dict(flow.guards), scope=dict(flow.scope),
            )
        else:
            flow.scope["pages"] = pages
            if spec.base in ("ai_self_check", "translate_normal", "special_pages"):
                flow.params["pages"] = pages
            elif len(pages) == 1 and spec.base in _PER_PAGE_BASES:
                # A single-page scope ("检查第4页") must drive the per-page base's
                # ``{{page}}``.  Writing only ``flow.scope`` (which the executor never
                # reads) silently audited page 1 instead of the requested page.
                flow.params["page"] = pages[0]
    return flow


# --------------------------------------------------------------------------
# 3. Tool-binding consistency (the "先绑定后暴露" gate).
# --------------------------------------------------------------------------

def flow_tool_names(flow: Flow) -> set[str]:
    """All tool names referenced by any ``ToolStep`` in a flow (recursive)."""

    def _walk(steps):
        for s in steps:
            if isinstance(s, ToolStep):
                yield s.tool
            elif isinstance(s, ForEachPage):
                yield from _walk(s.body)
            else:
                for attr in ("body", "then", "else_"):
                    sub = getattr(s, attr, None)
                    if isinstance(sub, list):
                        yield from _walk(sub)

    return set(_walk(flow.steps))


def validate_flow_tools(flow: Flow, available: set[str]) -> list[str]:
    """Return the tools a flow references that are neither bound nor known-deterministic.

    ``available`` is the real bound tool set (``make_source_tools`` +
    ``make_page_executors`` keys).  A non-empty return means the flow would call a
    tool the pipeline answers "unknown tool" — it must not be exposed to the model.
    """
    return sorted(name for name in flow_tool_names(flow)
                  if name not in available and name not in DETERMINISTIC_TOOLS)


# --------------------------------------------------------------------------
# 4. Promote / persist a user flow as a named standard flow.
# --------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]", "_", str(name))


def _executor_spec_path(name: str) -> Path:
    return user_flows_dir() / f"{_sanitize(name)}.json"


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # PID-unique temp name: a fixed ``.tmp`` suffix made two flows whose sanitized
    # names differ only by extension ("检查.v2" vs "检查") share one temp file, so
    # concurrent writes could swap their contents.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def save_flow_spec(name: str, spec: FlowSpec, *, persist: bool = True) -> None:
    """Stock ``spec`` under ``name`` in memory; persist to disk only when the flows dir is enabled.

    ``persist=False`` is an explicit in-memory-only promotion; the default ``True``
    is still env-gated (``_flows_persist_enabled``) so a run without
    ``PDFTRANSLATE_FLOWS_DIR`` never writes user-flow files implicitly.  The written
    JSON carries the original ``name`` so a reload keeps the registry key the user
    chose (the file name is sanitized, e.g. ``我的 流程`` → ``我的_流程.json``).
    """
    USER_FLOW_SPECS[name] = spec
    if persist and _flows_persist_enabled():
        _atomic_write(_executor_spec_path(name), {**spec.to_dict(), "name": name})


def load_user_flow_specs() -> dict[str, FlowSpec]:
    """Load promoted specs from disk (only when the flows dir is enabled) into ``USER_FLOW_SPECS``."""
    if not _flows_persist_enabled():
        return {}
    d = user_flows_dir()
    out: dict[str, FlowSpec] = {}
    if d.is_dir():
        for f in d.glob("*.json"):
            try:
                data = json.loads(f.read_text("utf-8")) if f.exists() else {}
                name = data.get("name", f.stem)
                spec = FlowSpec.from_dict(data)
                out[name] = spec
            except Exception:  # noqa: BLE001 — a corrupt file is skipped, never fatal
                continue
    USER_FLOW_SPECS.update(out)
    return out


def get_user_flow(name: str) -> Flow:
    """Build the runnable :class:`Flow` for a promoted user flow ``name``."""
    spec = USER_FLOW_SPECS.get(name)
    if spec is None:
        raise KeyError(f"未知用户流程：{name!r}")
    return build_flow(spec)


# --------------------------------------------------------------------------
# 5. Path B — an AI-composed task PLAN (自由分解：要求 → 多个异构任务).
# --------------------------------------------------------------------------

@dataclass
class Task:
    """One planned action the AI decomposed a requirement into.

    ``tier`` selects the execution machinery: ``"atomic"`` (a single catalog tool we
    call directly), ``"process"`` (one independent ``Flow``), or ``"composite"`` (a
    top-level translation flow).  ``name`` is a tool name (atomic) or a registered
    flow name (process/composite); ``params`` are that action's arguments.
    """

    tier: str = TIER_ATOMIC
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"tier": self.tier, "name": self.name, "params": dict(self.params)}


@dataclass
class Plan:
    """An ordered list of :class:`Task` (the AI's decomposition of one requirement)."""

    tasks: list[Task] = field(default_factory=list)
    note: str = ""
    #: Task names the model invented and the registry does not know.  They are
    #: dropped (the registry is authoritative), but the caller must be able to say
    #: so: a plan of three tasks that ran two reported plain success, so the user
    #: believed the dropped step ("彻底检查") had happened.
    dropped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tasks": [t.to_dict() for t in self.tasks],
                               "note": self.note}
        if self.dropped:
            out["dropped"] = list(self.dropped)
        return out


def _parse_plan_json(text: str) -> dict:
    """Extract a JSON object from a model reply (strips fences / surrounding prose)."""
    m = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 — bad JSON degrades to an empty plan
        return {}


def _validate_plan(data: dict | None, available: set[str] | None = None) -> Plan:
    """Validate an AI plan into a :class:`Plan`, dropping unknown/malformed tasks.

    The registry is authoritative: a task's ``tier`` is taken from the known tool /
    flow (whatever the model said is advisory), and a name that is neither a known
    atomic tool nor a registered flow is dropped **and reported** in ``Plan.dropped``
    (a plan of three steps that ran two must not read as "done").

    ``available`` (optional) narrows the accepted names to what the caller can run: the
    chat dispatcher supports a subset of the catalog, and a name it cannot run used to
    be accepted here and then fail the whole plan at that step.
    """
    data = data or {}
    tiers = registered_flow_tiers()
    atomic = atomic_tool_names()
    raw_tasks = data.get("tasks")
    tasks: list[Task] = []
    dropped: list[str] = []
    if isinstance(raw_tasks, list):
        for raw in raw_tasks:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "")
            params = dict(raw.get("params")) if isinstance(raw.get("params"), dict) else {}
            if available is not None and name not in available:
                dropped.append(name or "(未命名)")
                continue
            if name in tiers:
                tier = tiers[name]                 # flow -> its real tier
            elif name in atomic:
                tier = TIER_ATOMIC                 # a single atomic tool
            else:
                dropped.append(name or "(未命名)")   # unknown name -> drop, but report
                continue
            tasks.append(Task(tier=tier, name=name, params=params))
    return Plan(tasks=tasks, note=str(data.get("note", "")), dropped=dropped)


def compile_plan(req: str, *, llm: Callable[[str], dict] | None = None,
                 available: set[str] | None = None) -> Plan:
    """Decompose a requirement into a :class:`Plan` of ordered, mixed-tier tasks.

    This is the **free-composition** (Path B) entry.  It deliberately does NOT degrade
    to a deterministic single task: when ``llm`` is ``None`` (no model / no usable
    client) it returns an empty :class:`Plan`, and the caller must refuse — the chat
    ``run_plan`` tool reports "需要模型在线" instead of silently running a rule-parsed
    fallback.  A model reply that yields no valid tasks also yields an empty plan, so
    the caller refuses rather than inventing a fallback.
    """
    r = str(req or "").strip()
    if llm is None:
        return Plan(note="")
    try:
        data = llm(r) or {}
    except Exception:  # noqa: BLE001 — a bad/failing model reply -> empty plan (refuse)
        data = {}
    return _validate_plan(data, available=available)


def run_plan(plan: Plan, *, dispatch: Callable[[Task], dict],
             log: Callable[[str], None] | None = None,
             cancel: Callable[[], bool] | None = None) -> dict:
    """Execute a :class:`Plan` in order, delegating each task to ``dispatch(task)->dict``.

    ``dispatch`` returns a dict that must carry ``ok`` (the chat/agent channel decides
    how a tier/name runs).  Fail-closed: the first task that reports ``ok=False`` (or
    raises) stops the plan and returns the partial ``results`` so the caller can tell
    the user exactly which step failed.
    """
    results: list[dict[str, Any]] = []
    executed = 0
    note = plan.note
    if plan.dropped:
        extra = f"已忽略无法识别的任务：{'、'.join(plan.dropped)}"
        note = f"{note}；{extra}" if note else extra
    for task in plan.tasks:
        if cancel is not None and cancel():
            return {"ok": False, "error": "已取消", "executed": executed, "results": results}
        label = f"[{task.tier}:{task.name}]"
        if log:
            log(f"  计划执行：{label} {task.params}")
        try:
            out = dict(dispatch(task) or {})
        except ControlSignal:
            # A cancellation is a control signal, not a failed task: swallowing it
            # turned "user pressed 取消" into `ok=False, error="TranslationCancelled"`
            # and the plan looked like a normal failure (flow_steps re-raises; this
            # path did not).
            raise
        except Exception as exc:  # noqa: BLE001 — fail-closed per task
            if log:
                log(f"  计划任务失败：{label} {type(exc).__name__}: {exc}")
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        executed += 1
        # A step that reports an ``error`` without an explicit ``ok`` failed: the old
        # default (``True``) let `read_page(page=99)`'s "页号越界" count as success and
        # the plan kept going as if it had read the page.
        ok = bool(out.get("ok", not out.get("error")))
        results.append({"tier": task.tier, "name": task.name, "params": dict(task.params),
                        "ok": ok, **out})
        if not ok:
            return {"ok": False, "executed": executed, "results": results,
                    "dropped": list(plan.dropped), "note": note,
                    "error": f"任务 {task.name} 失败：{out.get('error', '')}"}
    return {"ok": True, "executed": executed, "results": results,
            "dropped": list(plan.dropped), "note": note}


#: Prompt that asks the model to decompose a requirement into an ordered task plan
#: (Path B's flexible branch — the caller refuses when no model is available, rather
#: than falling back to a deterministic single task).
_PLAN_COMPILE_PROMPT = (
    "把下面这句要求分解成**按顺序执行**的若干任务，输出一个 JSON 对象（只输出一个 JSON 对象，"
    "不要任何解释、不要 markdown 代码围栏）：\n"
    # NOTE: the JSON sample's literal braces must be doubled (``{{``/``}}``) — the prompt
    # is fed through ``str.format`` at the call site, so a single ``{`` would be parsed as
    # a replacement field and raise ``KeyError`` (regression: Path B AI 分解永远失败).
    '{{"tasks":[{{"tier":"atomic|process|composite","name":"<工具或流程名>","params":{{}}}}],"note":"<一句话说明>"}}\n'
    "tier=atomic → 单个工具：read_page/classify_page/get_doc_info/get_structure/get_table/get_settings/"
    "goto_page/set_block_text/delete_block_text/apply_annotation/retranslate/self_check/set_setting/re_export。\n"
    "tier=process 或 composite → 标准流程名：translate_page/translate_normal/special_pages/special_page/"
    "preprocess/export/self_check_page/ai_self_check/translate_doc。\n"
    "params 是该任务所需参数（如 page/indices/checks/scope/target_lang/requirement）。\n"
    "要求：{req}"
)


def make_llm_plan_compiler(model, client: Any = None,
                           log: Callable[[str], None] | None = None):
    """Return an AI plan decompiler ``llm(req) -> dict`` (Path B), or ``None``.

    Fail-closed **without degrading**: on a network / parse error the callback returns
    ``{}`` (``compile_plan`` yields an empty plan, so the caller refuses), and with no
    usable client it returns ``None`` — the chat ``run_plan`` tool then reports
    "需要模型在线" instead of running a rule-parsed fallback.  ``client`` (optional)
    reuses a shared
    OpenAI client.
    """
    from .. import translator as _tr

    if client is None:
        try:
            client = _tr.OpenAI(**model.client_kwargs())
        except Exception:  # noqa: BLE001 — no client → no AI decompilation
            return None

    def compile_req(req: str) -> dict:
        try:
            kwargs: dict[str, Any] = {
                "model": model.model,
                "temperature": 0.0,
                "max_tokens": 768,
                "messages": [{"role": "user",
                              "content": _PLAN_COMPILE_PROMPT.format(req=str(req or ""))}],
            }
            body = model.request_params()
            if body:
                kwargs["extra_body"] = body
            resp = client.chat.completions.create(**kwargs)
            text = (getattr(resp.choices[0].message, "content", "") or "").strip()
            return _parse_plan_json(text)
        except Exception as exc:  # noqa: BLE001 — fail-closed (no rule fallback)
            if log:
                log(f"  计划分解失败：{type(exc).__name__}: {exc}（返回空计划，调用方拒绝）。")
            return {}

    return compile_req
