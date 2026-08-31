"""Application configuration: ``models.json`` loading and user preferences.

The AI models available to the translator are declared in ``models.json`` at the
project root.  Each entry describes an OpenAI-compatible ``/chat/completions``
endpoint.  A value of the form ``${ENV_VAR}`` is substituted from the process
environment so secrets never have to be stored in the file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


def resource_dir() -> Path:
    """Base directory for external config files (``models.json``, ``glossary.json``).

    When the app is frozen by PyInstaller it may be moved anywhere, so config
    files are looked up **next to the executable** (the directory that holds the
    ``.exe``).  When running from source, the package's parent directory (the
    project root) is used, matching the developer layout.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


#: Path to the models.json located next to the executable (or project root when
#: running from source) — the user can edit this file to declare their models.
DEFAULT_MODELS_PATH = resource_dir() / "models.json"

#: Path to the default glossary (``glossary.json``) located next to the
#: executable.  A model may point at its own glossary file via the ``glossary``
#: config key.
DEFAULT_GLOSSARY_PATH = resource_dir() / "glossary.json"

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
    api_key: str | None = None
    tools_choice: str | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None   # sampling temperature (None → engine default)
    max_tokens: int | None = None      # per-request max completion tokens (None → server default)
    concurrency: int = 1               # parallel batch requests per translation run
    batch_size: int = 4000             # source-character budget per batch request
    glossary: str | None = None        # path to a per-model glossary file
    extra: dict[str, Any] = field(default_factory=dict)
    #: Human-readable problems found while parsing ``models.json`` (bad values
    #: are degraded to defaults instead of failing the whole file, and reported
    #: here so :meth:`validate` can surface them).
    parse_issues: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "ModelConfig":
        label = str(item.get("id") or item.get("name") or "?")
        issues: list[str] = []

        def _num(key: str, cast: Any) -> Any:
            """Parse a numeric field; a bad hand-edited value degrades to the
            default (with a validation issue) instead of raising — one typo
            must not leave *every* model in models.json unusable."""
            raw = item.get(key)
            if raw in (None, ""):
                return None
            try:
                return cast(raw)
            except (TypeError, ValueError):
                issues.append(f"模型 {label} 的 {key} 值无效：{raw!r}（已改用默认值）")
                return None

        concurrency = _num("concurrency", int)
        batch_size = _num("batch_size", int)
        return cls(
            id=str(item.get("id", "")),
            name=str(item.get("name", item.get("id", ""))),
            type=str(item.get("type", "openai")),
            endpoint=str(item.get("endpoint", "")),
            model=str(item.get("model", "")),
            api_key=substitute_env(item.get("api_key")) if item.get("api_key") else None,
            tools_choice=item.get("tools_choice"),
            reasoning_effort=(item.get("reasoning_effort") or None),
            temperature=_num("temperature", float),
            max_tokens=_num("max_tokens", int),
            concurrency=concurrency if concurrency and concurrency >= 1 else 1,
            batch_size=batch_size if batch_size and batch_size >= 1 else 4000,
            glossary=(item.get("glossary") or None),
            extra={k: v for k, v in item.items() if k not in cls._KNOWN_FIELDS},
            parse_issues=issues,
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
        "batch_size",
        "glossary",
    }

    #: Keys from ``models.json`` that may be forwarded verbatim to the OpenAI
    #: client constructor.  Anything else in ``extra`` is a typo (or a request
    #: body param, which belongs in ``reasoning_effort`` etc.) and would make
    #: ``OpenAI(**kwargs)`` raise ``TypeError`` mid-run; ignore it and report it
    #: from :meth:`validate` instead.
    _CLIENT_KEYS = {
        "organization",
        "timeout",
        "max_retries",
        "default_headers",
        "default_query",
        "http_client",
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
            if k in self._CLIENT_KEYS:
                kwargs[k] = v
        return kwargs

    def validate(self) -> list[str]:
        """Blocking problems: the model cannot be used at all until fixed."""
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

    def warnings(self) -> list[str]:
        """Non-blocking configuration problems (the model still works):
        hand-edit typos that were degraded to defaults, and unknown keys that
        are ignored rather than forwarded to the OpenAI client."""
        label = self.id or self.name or "?"
        issues: list[str] = list(self.parse_issues)
        for key in self.extra:
            if key not in self._CLIENT_KEYS:
                issues.append(
                    f"模型 {label} 含未识别的配置键 {key!r}，已忽略"
                    "（请检查拼写；按请求参数如 reasoning_effort 直接写在条目顶层）"
                )
        return issues


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

def default_model_id() -> str:
    """Return the id of the first declared model, or an empty string."""
    try:
        models = load_models()
    except Exception as exc:  # noqa: BLE001
        _logger.warning("读取默认模型失败: %s", exc)
        return ""
    return models[0].id if models else ""


def load_glossary(path: Path | str | None = None) -> dict[str, str]:
    """Load a glossary of ``source -> target`` term mappings.

    ``path`` defaults to ``DEFAULT_GLOSSARY_PATH``; a relative path resolves
    against :func:`resource_dir` (next to the executable / project root), never
    the current working directory, so a frozen app launched from elsewhere
    still finds its file.  A missing/unreadable file yields an empty glossary
    (never an error, so a file-less install still works unchanged).
    Three JSON shapes are accepted:

    * ``{"transformer": "变换器", "key": "密钥"}``
    * ``{"terms": {"transformer": "变换器"}}``
    * ``[["transformer", "变换器"], ["key", "密钥"]]``

    The returned mappings are injected into every batch prompt so the model
    translates the same domain term identically across all chunks.
    """
    try:
        p = Path(path) if path else DEFAULT_GLOSSARY_PATH
        if not p.is_absolute():
            p = resource_dir() / p
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            if isinstance(data.get("terms"), dict):
                data = data["terms"]
            return {str(k): str(v) for k, v in data.items()}
        if isinstance(data, list):
            out: dict[str, str] = {}
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out[str(item[0])] = str(item[1])
            return out
    except Exception as exc:  # noqa: BLE001 — a missing/unreadable glossary is not fatal
        _logger.debug("读取术语表失败（按空返回）: %s", exc)
    return {}


def save_glossary(path: Path | str, glossary: dict[str, str]) -> None:
    """Write ``{"terms": {source: target}}`` to ``path`` (best effort)."""
    try:
        Path(path).write_text(
            json.dumps({"terms": glossary}, ensure_ascii=False, indent=2),
            "utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning("保存术语表失败: %s", exc)


# ---------------------------------------------------------------------------
# User preferences
# ---------------------------------------------------------------------------

def load_prefs() -> dict[str, Any]:
    """Load the persisted user preferences (empty dict if none are saved)."""
    try:
        if APP_PREFS_PATH.exists():
            with APP_PREFS_PATH.open("r", encoding="utf-8") as fh:
                return dict(json.load(fh))
    except Exception as exc:  # noqa: BLE001
        _logger.debug("读取用户偏好失败（按空返回）: %s", exc)
    return {}


def save_prefs(prefs: dict[str, Any]) -> None:
    """Persist user preferences to disk."""
    try:
        APP_PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with APP_PREFS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(prefs, fh, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("保存用户偏好失败: %s", exc)
