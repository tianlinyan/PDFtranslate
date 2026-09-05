"""Control-signal exceptions shared across the pipeline.

A user cancellation aborts a run by raising a **control signal** — not a failure.
The translation engine and the flow executor each use their own concrete type
(``TranslationCancelled`` / ``FlowCancelled``), but a common base lets a layer
that must re-raise *any* cancellation (the flow executor's ``ToolStep``) do so
without importing the other layer's module.
"""

from __future__ import annotations


class ControlSignal(Exception):
    """Base for user-cancellation control signals.

    These are NOT failures and must never be swallowed into a fail-closed
    ``error``: they propagate all the way up so the worker reports "已取消" and
    stops.  Concrete subclasses: ``translator.TranslationCancelled`` and
    ``agent.flow_steps.FlowCancelled``.
    """
