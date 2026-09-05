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

from .flow_steps import Flow, STANDARD_FLOWS, ForEachPage, ToolStep


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
        extra = {k: v for k, v in data.items()
                 if k not in {"base", "checks", "scope", "include_kept", "auto_fix",
                              "page", "lang", "kind", "output_type"}}
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


# --------------------------------------------------------------------------
# 1. Rule-based requirement → FlowSpec (Path A slot-filling).
# --------------------------------------------------------------------------

#: Chinese alias → check name, used to fill ``spec.checks``.
_AUDIT_ALIASES: dict[str, str] = {
    "数字": "numbers", "金额": "numbers", "数": "numbers",
    "表格": "table", "表": "table",
    "版面": "layout", "布局": "layout",
    "漏译": "missing", "残留": "residual",
}

_SCOPE_RANGE_RE = re.compile(r"第?\s*(\d+)\s*(?:-|到|~|至)\s*第?\s*(\d+)\s*页")
_SCOPE_SINGLE_RE = re.compile(r"第\s*(\d+)\s*页")


def _parse_checks(req: str) -> list[str] | None:
    found = [name for alias, name in _AUDIT_ALIASES.items() if alias in req]
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


def _base_from(req: str, default: str) -> str:
    if "重新导出" in req or "重新生成" in req:
        return "export"
    if "重译" in req:
        return "translate_page"
    if "自检" in req or "检查" in req:
        return "self_check_page"
    return default


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
    checks = [str(c) for c in checks] if isinstance(checks, list) else None
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
        auto_fix=False if any(k in r for k in ("只查", "只读", "不修改", "不改")) else None,
    )


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
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def save_flow_spec(name: str, spec: FlowSpec, *, persist: bool = True) -> None:
    """Stock ``spec`` under ``name``; persist to disk when the flows dir is enabled."""
    USER_FLOW_SPECS[name] = spec
    if persist:
        _atomic_write(_executor_spec_path(name), spec.to_dict())


def load_user_flow_specs() -> dict[str, FlowSpec]:
    """Load all promoted specs from disk (env-gated dir) into ``USER_FLOW_SPECS``."""
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
