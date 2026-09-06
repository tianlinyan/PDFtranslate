"""Agent-facing view over the single tool catalog.

The canonical definitions (schema + description + audience) live in
:mod:`translate_app.agent.tool_catalog`; this module is the **translation-agent
side's** view of it.  ``AGENT_TOOLS`` is just the ``"agent"``-audience subset, and
``agent_openai_tools`` builds the OpenAI ``tools`` array for that subset.  Kept as
a thin facade so existing callers (``agent_openai_tools`` / ``by_name`` /
``AGENT_TOOLS``) keep their shape while the one-catalog refactor is in place.
"""

from __future__ import annotations

from typing import Any

from .tool_catalog import (  # noqa: F401  (TOOL_CATALOG re-exported for importers)
    TOOL_CATALOG,
    ToolDef,
)
from .tool_catalog import catalog_for, to_openai_schema

# The agent exposes exactly the tools with "agent" in their audience.
AGENT_TOOLS: list[ToolDef] = catalog_for("agent")


def agent_openai_tools(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Return the OpenAI ``tools`` array for the agent subset (optionally filtered)."""
    out: list[dict[str, Any]] = []
    for t in AGENT_TOOLS:
        if names is not None and t.name not in names:
            continue
        out.append(to_openai_schema(t))
    return out


def by_name(name: str) -> ToolDef | None:
    for t in AGENT_TOOLS:
        if t.name == name:
            return t
    return None
