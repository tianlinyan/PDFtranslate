"""Workflow state for the v0.3.0 AI-orchestration flow.

The :class:`WorkflowState` is the single source of truth the ``FlowAgent``
controller reads/writes while driving the tool layer.  It records, per page and
per operation, exactly what was done and why, so the whole run is auditable and
revertible.  The AI only ever mutates *decisions* held in this structure — the
state's shape stays deterministic and testable.

See ``docs/0.3.0-设计.md`` for the architecture (this is P1: the skeleton).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Per-page / overall progress statuses the controller reports.
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"
STATUS_NEEDS_USER = "needs_user"

#: ``WorkflowState.phase`` — the interactive session state machine.
PHASE_IDLE = "idle"
PHASE_PREPROCESS = "preprocess"
PHASE_TRANSLATE_NORMAL = "translate_normal"
PHASE_SPECIAL_PAGES = "special_pages"
PHASE_COMPLETED = "completed"
PHASE_REVIEW = "review"
PHASE_EXPORT = "export"
PHASE_DONE = "done"


@dataclass
class Budget:
    """Bounded recursion budget for a run (fail-closed beyond these limits)."""
    max_steps: int = 200          # max total tool calls
    max_passes: int = 6           # max passes (rounds) per page
    cost_cap: float = 30.0        # rough cost budget (arbitrary units)
    used_steps: int = 0
    used_passes: int = 0
    cost: float = 0.0

    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.used_steps)

    def exhausted(self) -> bool:
        return self.used_steps >= self.max_steps


#: ``status`` values a Goal / PageState may take.
_GOAL_STATUSES = frozenset((STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_DONE, STATUS_NEEDS_USER))


@dataclass
class Op:
    """A provenance record of one tool application.

    ``before`` / ``after`` are the relevant slice of state that changed (kept
    shallow/serialisable so a run can be audited or rolled back stepwise).
    """

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    target: str = ""              # e.g. "page:3 block:12"
    before: Any = None
    after: Any = None
    reason: str = ""              # why the tool was applied
    user_confirmed: bool = False


@dataclass
class Goal:
    """One item on the controller's todo queue."""

    kind: str                     # e.g. "translate", "verify_number", "check_residual"
    page: int | None = None
    status: str = STATUS_PENDING
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in _GOAL_STATUSES:
            raise ValueError(f"Unknown goal status: {self.status!r}")


@dataclass
class PageState:
    """Per-page progress for the orchestrator."""

    index: int
    status: str = STATUS_PENDING
    keep_indices: set[int] = field(default_factory=set)   # flat indices kept verbatim
    translated: list[str] = field(default_factory=list)   # per-block translation
    issues: list[str] = field(default_factory=list)       # surfaced QA issues
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in _GOAL_STATUSES:
            raise ValueError(f"Unknown page status: {self.status!r}")


@dataclass
class DocInfo:
    """The source document summary produced by the preprocess phase.

    Mirrors the dict from :func:`pdfio.get_doc_info` so the session can carry it
    as a typed object.
    """

    pages: int
    title: str = ""
    language: str = "unknown"
    text_pages: int = 0
    scan_pages: int = 0
    chart_pages: int = 0
    table_pages: int = 0
    uncertain_pages: int = 0
    special_pages: int = 0
    block_count: int = 0
    kinds: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DocInfo":
        return cls(
            pages=int(d.get("pages", 0)),
            title=str(d.get("title", "")),
            language=str(d.get("language", "unknown")),
            text_pages=int(d.get("text_pages", 0)),
            scan_pages=int(d.get("scan_pages", 0)),
            chart_pages=int(d.get("chart_pages", 0)),
            table_pages=int(d.get("table_pages", 0)),
            uncertain_pages=int(d.get("uncertain_pages", 0)),
            special_pages=int(d.get("special_pages", 0)),
            block_count=int(d.get("block_count", 0)),
            kinds=list(d.get("kinds", []) or []),
        )


@dataclass
class PageTriage:
    """One page's triage verdict (kind) plus whether the user decided it."""

    page: int
    kind: str = "normal"
    decided: bool = False
    decision: str = ""


@dataclass
class WorkflowState:
    """All state for one AI-orchestrated translation run.

    The run holds **two** documents side by side and keeps them separated by
    design (hard invariant, see ``docs/0.3.0-设计.md``):

    * ``src_doc`` — the **immutable original** (read-only).  No tool may ever
      write to it; observe/verify tools only read it.
    * ``out_doc`` — the **mutable translation** (freely editable).  Every
      write tool (edit / create / delete / draw / page ops) mutates this, never
      the original.
    """

    src_path: str
    lang: str
    src_doc: Any = None                     # immutable original (read-only)
    out_doc: Any = None                     # mutable translation (freely edited)
    requirements: list[str] = field(default_factory=list)  # user requirements
    user_decisions: dict[str, Any] = field(default_factory=dict)  # key-point answers
    pages: list[PageState] = field(default_factory=list)
    ops: list[Op] = field(default_factory=list)
    todo: list[Goal] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    pending_question: dict[str, Any] | None = None   # surfaced ask to the GUI
    log: list[str] = field(default_factory=list)
    #: Interactive-session fields (see docs/0.3.1-交互式翻译流程设计.md).
    phase: str = PHASE_IDLE                          # SessionPhase
    doc_info: DocInfo | None = None                  # preprocess summary
    triage: dict[int, PageTriage] = field(default_factory=dict)   # per-page kind
    current_page: int = 0                            # preview navigation pointer
    annotations: list[dict[str, Any]] = field(default_factory=list)  # user marks
    summary: str = ""                                # pipeline summary
    review_mode: str = ""                            # "ai" | "user"

    def page(self, index: int) -> PageState:
        """The ``PageState`` for ``index``, created on first access."""
        while len(self.pages) <= index:
            self.pages.append(PageState(index=len(self.pages)))
        return self.pages[index]

    def record_op(
        self,
        tool: str,
        *,
        args: dict[str, Any] | None = None,
        target: str = "",
        before: Any = None,
        after: Any = None,
        reason: str = "",
        user_confirmed: bool = False,
    ) -> Op:
        """Append a provenance ``Op`` and advance the step counter."""
        op = Op(
            tool=tool,
            args=args or {},
            target=target,
            before=before,
            after=after,
            reason=reason,
            user_confirmed=user_confirmed,
        )
        self.ops.append(op)
        self.budget.used_steps += 1
        return op

    def ask(self, question: str, options: list[str] | None = None, *, target: str = "") -> None:
        """Record a question to surface to the UI (non-blocking side panel)."""
        self.pending_question = {
            "question": question,
            "options": options or [],
            "target": target,
        }

    def answer(self, value: Any, target: str = "") -> None:
        """Store the user's answer to the last (or a target) question."""
        if target:
            self.user_decisions[target] = value
        elif self.pending_question:
            self.user_decisions[self.pending_question.get("target", "")] = value
        self.pending_question = None
