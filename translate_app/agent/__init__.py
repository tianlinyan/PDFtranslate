"""v0.3.0 AI-orchestration foundation (P1).

Package layout:

* :mod:`.state` — the ``WorkflowState`` skeleton the controller reads/writes.
* :mod:`.tools` — the deterministic tool registry, the OpenAI tool schemas, and
  the binder for the existing judgment tools.

P2 adds the ``FlowAgent`` controller that plans and calls these tools; P3 adds
the non-blocking sidebar interaction.  See ``docs/0.3.0-设计.md``.
"""

from .state import (
    Budget,
    DocInfo,
    Goal,
    Op,
    PageState,
    PageTriage,
    PHASE_COMPLETED,
    PHASE_DONE,
    PHASE_IDLE,
    PHASE_PREPROCESS,
    PHASE_SPECIAL_PAGES,
    PHASE_TRANSLATE_NORMAL,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_NEEDS_USER,
    STATUS_PENDING,
    WorkflowState,
)
from .tools import (
    AGENT_TOOLS,
    ToolDef,
    agent_openai_tools,
    by_name,
)
from .flow import (
    AgentResult,
    Decision,
    DocumentSession,
    FlowAgent,
    make_llm_decide,
    make_page_executors,
    make_source_tools,
    run_agent_run,
    run_page_visual,
)

__all__ = [
    "AGENT_TOOLS",
    "AgentResult",
    "Budget",
    "Decision",
    "DocInfo",
    "DocumentSession",
    "FlowAgent",
    "Goal",
    "Op",
    "PageState",
    "PageTriage",
    "PHASE_COMPLETED",
    "PHASE_DONE",
    "PHASE_IDLE",
    "PHASE_PREPROCESS",
    "PHASE_SPECIAL_PAGES",
    "PHASE_TRANSLATE_NORMAL",
    "STATUS_DONE",
    "STATUS_IN_PROGRESS",
    "STATUS_NEEDS_USER",
    "STATUS_PENDING",
    "ToolDef",
    "WorkflowState",
    "agent_openai_tools",
    "by_name",
    "make_llm_decide",
    "make_page_executors",
    "make_source_tools",
    "run_agent_run",
    "run_page_visual",
]
