"""Application configuration: ``models.json`` loading and user preferences.

The AI models available to the translator are declared in ``models.json`` at the
project root.  Each entry describes an OpenAI-compatible ``/chat/completions``
endpoint.  A value of the form ``${ENV_VAR}`` is substituted from the process
environment so secrets never have to be stored in the file.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Path to the models.json located next to the package.
DEFAULT_MODELS_PATH = Path(__file__).resolve().parent.parent / "models.json"

#: Path to the user preferences file.
APP_PREFS_PATH = Path.home() / ".pdftranslate" / "prefs.json"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class ModelConfig:
    """A single AI model entry parsed from ``models.json``."""

    id: str
    name: str
    type: str
    endpoint: str
    model: str
    # ``repr=False``: the dataclass's default ``repr`` would print the raw key
    # (``api_key='sk-...'``) if the config were ever logged or asserted on.
    api_key: str | None = field(default=None, repr=False)
    tools_choice: str | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None   # sampling temperature (None → engine default)
    max_tokens: int | None = None      # per-request max completion tokens (None → server default)
    concurrency: int = 1               # parallel batch requests per translation run
    page_concurrency: int = 1          # parallel PAGE translation in the agent path (1 = sequential)
    batch_size: int = 4000             # source-character budget per batch request
    #: When true, the model may be used to *review* an OCR-rebuilt scanned page:
    #: the original page and the reconstruction are sent to it, and its text
    #: corrections and layout hints are applied / surfaced (geometry untouched).
    vision: bool = False
    #: Sampling temperature for the **AI interaction** / agent orchestration
    #: ``decide`` calls.  This is a *separate* parameter set from the translation
    #: request parameters above: a translation wants a low, deterministic
    #: temperature (``model.temperature``), while the orchestrator that plans tool
    #: calls benefits from a little exploration and a bit more reasoning.
    #: ``models.json`` may override per model via ``interaction_temperature``.
    interaction_temperature: float = 0.6
    #: ``reasoning_effort`` sent to the agent's ``decide`` calls (via ``extra_body``),
    #: defaulting to ``"medium"``.  ``models.json`` may override per model via
    #: ``interaction_reasoning_effort``.
    interaction_reasoning_effort: str | None = "medium"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "ModelConfig":
        return cls(
            id=str(item.get("id", "")),
            name=str(item.get("name", item.get("id", ""))),
            type=str(item.get("type", "openai")),
            endpoint=str(item.get("endpoint", "")),
            model=str(item.get("model", "")),
            # Store the raw value; ``${ENV_VAR}`` substitution happens lazily in
            # ``_resolved_api_key`` / ``validate`` (substituting here *and*
            # there again gave inconsistent results — e.g. an env var set to ""
            # silently became "not-needed" with no validation warning).
            api_key=(str(item["api_key"]) if item.get("api_key") else None),
            tools_choice=item.get("tools_choice"),
            reasoning_effort=(item.get("reasoning_effort") or None),
            temperature=(
                float(item["temperature"])
                if item.get("temperature") not in (None, "")
                else None
            ),
            max_tokens=(
                int(item["max_tokens"]) if item.get("max_tokens") not in (None, "") else None
            ),
            concurrency=int(item.get("concurrency") or 1),
            page_concurrency=int(item.get("page_concurrency") or 1),
            batch_size=int(item.get("batch_size") or 4000),
            vision=bool(item.get("vision", False)),
            interaction_temperature=(
                float(item["interaction_temperature"])
                if item.get("interaction_temperature") not in (None, "")
                else 0.6
            ),
            interaction_reasoning_effort=(item.get("interaction_reasoning_effort") or "medium"),
            extra={k: v for k, v in item.items() if k not in cls._KNOWN_FIELDS},
        )

    #: Keys consumed explicitly by :meth:`from_dict`.
    _KNOWN_FIELDS = {
        "id",
        "name",
        "type",
        "endpoint",
        "model",
        "api_key",
        "tools_choice",
        "reasoning_effort",
        "temperature",
        "max_tokens",
        "concurrency",
        "page_concurrency",
        "batch_size",
        "vision",
        "interaction_temperature",
        "interaction_reasoning_effort",
    }

    def request_params(self) -> dict[str, Any]:
        """Per-request body parameters sent to the server, if any.

        ``reasoning_effort`` and ``tools_choice`` are model-specific extras the
        OpenAI client does not expose as first-class kwargs, so they are sent via
        ``extra_body`` (the SDK merges them into the JSON body verbatim).  The
        config key ``tools_choice`` maps to the API field ``tool_choice``.
        """
        params: dict[str, Any] = {}
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        if self.tools_choice:
            params["tool_choice"] = self.tools_choice
        return params

    def interaction_request_params(self) -> dict[str, Any]:
        """Per-request body parameters for the **AI interaction** (agent) calls.

        Kept separate from :meth:`request_params` so the agent orchestrator can
        reason / sample differently than the deterministic translation pass — e.g.
        the same llama.cpp endpoint needs ``reasoning_effort`` supplied for the
        agent's ``decide`` calls too, and the interaction temperature is higher so
        the model explores a little instead of being pinned to greedy 0.0.
        """
        params: dict[str, Any] = {}
        if self.interaction_reasoning_effort:
            params["reasoning_effort"] = self.interaction_reasoning_effort
        return params

    def _resolved_api_key(self) -> str | None:
        """Return the real API key, or ``None`` if absent / unresolved."""
        if not self.api_key:
            return None
        value = substitute_env(self.api_key)
        # An unresolved ``${VAR}`` placeholder must never be sent as a key.
        if not value or "${" in value:
            return None
        return value

    def client_kwargs(self) -> dict[str, Any]:
        """Return the kwargs used to construct an OpenAI client for this model."""
        base = self.endpoint.rstrip("/")
        # The OpenAI client appends ``/chat/completions`` itself, so drop that
        # suffix from a full chat-completions URL to obtain the base_url.
        for suffix in ("/chat/completions",):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        # Sensible request timeout (the OpenAI SDK defaults to 600s); a model
        # entry may still override it explicitly through ``extra``.
        kwargs: dict[str, Any] = {"base_url": base, "timeout": 300.0}
        key = self._resolved_api_key()
        # llama-server / local endpoints typically do not require a key.
        kwargs["api_key"] = key if key else "not-needed"
        for k, v in self.extra.items():
            if k not in ("base_url", "api_key", "model", "endpoint"):
                kwargs[k] = v
        return kwargs

    def validate(self) -> list[str]:
        """Return a list of configuration problems (empty if the model is usable)."""
        issues: list[str] = []
        label = self.id or self.name or "?"
        if not self.endpoint:
            issues.append(f"模型 {label} 缺少 endpoint")
        if not self.model:
            issues.append(f"模型 {label} 缺少 model")
        if self.api_key:
            resolved = substitute_env(self.api_key)
            if not resolved or "${" in resolved:
                issues.append(f"模型 {label} 的 api_key 环境变量未设置")
        return issues

    def endpoint_warnings(self) -> list[str]:
        """Return advisory (non-blocking) hints about the endpoint string.

        Unlike :meth:`validate` (whose findings block a run), these never stop a
        translation — they only surface a likely misconfiguration in the log, so
        any feature (translation *and* the vision whole-page review, which share
        :meth:`client_kwargs`) keeps working exactly as before.

        The documented convention is a full ``.../chat/completions`` URL.  An
        endpoint without that suffix is passed to the SDK as a ``base_url``,
        which only works when the server mounts the OpenAI API at exactly that
        root (e.g. ``http://host:8888/v1``).  A bare ``http://host:8888`` (no
        versioned path) is the classic llama-server mistake: the SDK would
        request ``http://host:8888/chat/completions``, which llama-server does
        not serve.
        """
        if not self.endpoint:
            return []
        base = self.client_kwargs().get("base_url")
        if not base:
            return []
        if self.endpoint.rstrip("/").endswith("/chat/completions"):
            return []
        effective = f"{base.rstrip('/')}/chat/completions"
        return [
            f"endpoint 未以 /chat/completions 结尾，将按 base_url 处理，"
            f"实际会请求 {effective}。若该服务安装在 /v1 下，请改用形如 "
            "http://host:port/v1/chat/completions 的完整地址。"
        ]


def substitute_env(value: str | None) -> str | None:
    """Replace ``${VAR}`` placeholders in ``value`` with environment values."""
    if not value:
        return value

    def _repl(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(0))

    return _ENV_PATTERN.sub(_repl, value)


def load_models(path: Path | str = DEFAULT_MODELS_PATH) -> list[ModelConfig]:
    """Load and parse ``models.json`` into a list of :class:`ModelConfig`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Models configuration not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data.get("models", [])
    return [ModelConfig.from_dict(e) for e in entries]


# ---------------------------------------------------------------------------
# User preferences
# ---------------------------------------------------------------------------

def load_prefs() -> dict[str, Any]:
    """Load the persisted user preferences (empty dict if none are saved)."""
    try:
        if APP_PREFS_PATH.exists():
            with APP_PREFS_PATH.open("r", encoding="utf-8") as fh:
                return dict(json.load(fh))
    except Exception:
        pass
    return {}


def save_prefs(prefs: dict[str, Any]) -> str | None:
    """Persist user preferences to disk.

    Returns ``None`` on success, or a short reason on failure (the caller
    should surface it — a silently lost preference is invisible until the next
    launch).  The write is atomic (temp file + ``os.replace``), matching the
    translation/OCR caches: the GUI hard-exits on window close, and a plain
    overwrite could leave a truncated ``prefs.json``.
    """
    tmp = APP_PREFS_PATH.with_name(f"{APP_PREFS_PATH.name}.{os.getpid()}.tmp")
    try:
        APP_PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), "utf-8")
        os.replace(tmp, APP_PREFS_PATH)
        return None
    except Exception as exc:  # noqa: BLE001 — prefs are best-effort
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return f"{type(exc).__name__}: {exc}"
