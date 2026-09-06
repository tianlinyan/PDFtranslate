"""Free-text intent slot-filling for deterministic decision points (M1).

One reusable, LLM-backed "pick an option from free text" reader with a deterministic
fallback.  Every AI-ified decision point routes through here as an *injected* backend:

* ``make_llm_intent_fill(model, client=None)`` -> ``read(text, choices) -> str`` or
  ``None`` when no usable client exists.  The caller injects the reader and keeps its
  existing deterministic matcher as the fallback, so leaving the reader out is a no-op.

Design constraints (see ``docs/0.3.9-ai化路线图.md``):

* It is a **decision**, not a read-only chat tool.  It is never exposed inside the
  ``ChatSession`` tool loop as a tool; it is wired in by the caller as a parameter.
* It uses the **translation-side** request params (``model.request_params()``) with a
  tiny temperature-0 classification, so it never mixes with the interaction parameter
  set (``interaction_*``).
* It is **fail-closed**: a bad client / network / unparseable reply returns ``""`` and
  the caller falls back to its deterministic matcher (never hangs, never crashes).
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Sequence


def _parse_json_object(text: str | None) -> dict:
    """Extract the first JSON object from a model reply; fail-closed to ``{}``.

    Strips code fences / surrounding prose, and degrades gracefully to an empty dict
    for any malformed or non-object reply so the caller falls back to defaults.
    """
    m = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 — bad JSON degrades to defaults
        return {}


_CHOICE_FILL_PROMPT = (
    "把下面这句话解析成用户意图（只输出一个 JSON 对象，不要任何解释、不要代码围栏）：\n"
    '字段：{"choice": "<一个可选值>"}。可选值（只能从这些里选一个）：[[choices]]\n'
    "用户的话：[[text]]"
)


def make_llm_intent_fill(model, *, client: Any = None,
                         log: Callable[[str], None] | None = None,
                         temperature: float = 0.0, max_tokens: int = 64):
    """Return ``read(text, choices) -> str`` (one of ``choices``) backed by the LLM.

    ``choices`` is an iterable of allowed values.  The reader picks one of them from the
    user's free-text ``text`` and returns it lowercased; on any failure it returns
    ``""`` so the caller keeps its deterministic matcher.  Returns ``None`` when no
    usable client exists — the caller simply stays on the deterministic path.

    ``client`` (optional) reuses a shared OpenAI client; ``log`` (optional) reports a
    one-line failure so a reader outage is visible without interrupting the flow.
    """
    from .. import translator as _tr

    if client is None:
        try:
            client = _tr.OpenAI(**model.client_kwargs())
        except Exception:  # noqa: BLE001 — no client → deterministic path
            return None

    def read(text, choices: Sequence[str]) -> str:
        allowed = {str(c).strip().lower() for c in choices}
        prompt = _CHOICE_FILL_PROMPT.replace("[[choices]]", " / ".join(str(c) for c in choices))
        prompt = prompt.replace("[[text]]", str(text or ""))
        try:
            body = model.request_params()
            kwargs: dict[str, Any] = {
                "model": model.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if body:
                kwargs["extra_body"] = body
            resp = client.chat.completions.create(**kwargs)
            txt = (getattr(resp.choices[0].message, "content", "") or "").strip()
            data = _parse_json_object(txt)
            choice = str(data.get("choice", "")).strip().lower()
            if choice in allowed:
                return choice
            # Fallback: an allowed value appearing verbatim in the reply.
            for c in allowed:
                if c in txt.lower():
                    return c
            return ""
        except Exception as exc:  # noqa: BLE001 — fail-closed to the deterministic matcher
            if log:
                log(f"  意图槽填充失败：{type(exc).__name__}: {exc}（用规则匹配落回）。")
            return ""

    return read
