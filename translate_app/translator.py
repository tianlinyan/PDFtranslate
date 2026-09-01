"""AI translation engine.

Turns the extracted text blocks of a PDF into a target language using any of the
OpenAI-compatible ``/chat/completions`` endpoints declared in ``models.json``.

Key behaviours
--------------
* Blocks are grouped into batches that stay within a token budget.
* Batches are translated concurrently (``model.concurrency`` parallel requests).
* Each batch is sent as a numbered list; the model is asked to return numbered
  translations so paragraphs stay aligned.
* Translated blocks are cached (keyed by source text) so re-running the same
  document resumes instead of re-translating; the cache is written after every
  batch and carries a version tag so stale formats are never reused.
* Transient failures are retried with backoff; a block that cannot be
  translated is left as the original text rather than silently dropped, and
  failed batches are never written into the cache.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from openai import (
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
)

from .settings import ModelConfig

#: Matches one ``[n]`` block in a model reply.  The block content may span
#: several lines (some models wrap long translations); everything up to the
#: next ``[n]`` marker (or the end of the reply) belongs to this block.
_MULTI_BLOCK_RE = re.compile(r"(?ms)^\s*\[(\d+)\]\s*(.*?)(?=^\s*\[\d+\]\s*|\Z)")

#: Default source-character budget per batch request.  Kept modest so a single
#: request's output stays within the model's max-token limit, and so progress
#: is reported often.  Each model may override it via ``batch_size`` — larger
#: budgets mean fewer requests, which helps slow local models by amortising
#: the per-request fixed cost (prompt processing + reasoning overhead).
_CHAR_BUDGET = 4000

#: Cache format version — bump when the prompt or response format changes so
#: stale translations from an older run are never reused.
_CACHE_VERSION = 3

#: Delay between batch attempts (seconds); injectable so tests don't sleep.
_TRANSIENT_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0)

#: Letters (CJK, kana, hangul, Greek, Cyrillic, Hebrew, Arabic, Latin) that
#: make a block worth translating.  Blocks without any letters (page numbers,
#: separators, pure symbols) are kept as-is and never sent to the model.
_LETTERS_RE = re.compile(
    r"[぀-ヿㇰ-ㇿ㐀-䶿一-鿿豈-﫿"
    r"가-힯Ͱ-ϿЀ-ӿ֐-׿؀-ۿ"
    r"A-Za-z]"
)

ProgressFn = Callable[[int, int], None]
LogFn = Callable[[str], None]
CancelFn = Callable[[], bool]


class TranslationCancelled(Exception):
    """Raised when a user cancels an in-flight translation."""


@dataclass
class TranslationResult:
    """Result of translating a single document."""

    blocks: list[str] = field(default_factory=list)          # source blocks
    translated: list[str] = field(default_factory=list)      # aligned translations
    errors: list[str] = field(default_factory=list)          # human readable notes

    def __len__(self) -> int:
        return len(self.blocks)


def _cache_dir() -> Path:
    """Return a writable cache directory, falling back to the system temp dir.

    Creating the directory is not enough to prove it is usable: an existing
    directory under a read-only / sandboxed ``$HOME`` lets ``mkdir`` succeed
    (``exist_ok=True`` is a no-op) while every later write is denied.  That
    silently disables resume, so each candidate is verified with a real probe
    write before it is accepted.
    """
    for path in (
        Path.home() / ".pdftranslate" / "cache",
        Path(tempfile.gettempdir()) / "pdftranslate_cache",
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / f".write_probe_{os.getpid()}"
            probe.write_text("", "utf-8")
            probe.unlink()
            return path
        except Exception:
            continue
    return Path.home() / ".pdftranslate" / "cache"


def _cache_key(doc_path: Path, target_lang: str, model_id: str) -> str:
    h = hashlib.sha1(
        f"{doc_path.resolve()}|{target_lang}|{model_id}".encode("utf-8")
    ).hexdigest()[:16]
    return f"trans_v{_CACHE_VERSION}_{h}.json"


def _block_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _needs_translation(text: str) -> bool:
    """Return True if ``text`` contains letters and thus needs translation."""
    return bool(_LETTERS_RE.search(text))


def _sleep_interruptible(seconds: float, cancel: CancelFn) -> None:
    """Sleep in small slices so a cancellation stays responsive."""
    if seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if cancel():
            raise TranslationCancelled()
        time.sleep(min(0.2, deadline - time.monotonic()))


def load_translation_cache(doc_path: Path, target_lang: str, model_id: str) -> dict[str, str]:
    """Load the on-disk translation cache for a doc/lang/model (empty if none)."""
    cache_path = _cache_dir() / _cache_key(doc_path, target_lang, model_id)
    if cache_path.exists():
        try:
            import json

            return json.loads(cache_path.read_text("utf-8"))
        except Exception:
            pass
    return {}


def clear_translation_cache() -> int:
    """Delete all cached translation files; returns the number of files removed."""
    removed = 0
    for path in _cache_dir().glob("trans_*.json"):
        try:
            path.unlink()
            removed += 1
        except Exception:
            pass
    return removed


class TranslationEngine:
    """Wraps one AI model and translates a list of text blocks."""

    def __init__(self, model: ModelConfig):
        self.model = model
        self.client = OpenAI(**model.client_kwargs())
        # Guards the shared translation cache while batches complete concurrently.
        self._cache_lock = threading.Lock()

    @staticmethod
    def _build_prompt(blocks: Sequence[str], indices: Sequence[int]) -> str:
        lines = []
        # Number the blocks 1..k within this batch so the model always receives a
        # contiguous, unambiguous set to echo back (avoids misalignment when the
        # batch's global indices are sparse because some blocks were cached).
        for pos, i in enumerate(indices):
            lines.append(f"[{pos + 1}]\n{blocks[i]}")
        return "\n\n".join(lines)

    @staticmethod
    def _system_prompt(language: str) -> str:
        return (
            "You are a professional document translator. Translate every numbered "
            f"block below into {language}.\n"
            "Rules:\n"
            "- Keep the original meaning, tone and paragraph structure.\n"
            "- Keep the translation similar in length to the source and word it "
            "concisely, so it fits the original document layout.\n"
            "- Keep numbers, units, URLs, code, product names and proper nouns as "
            "in the source unless a standard translation exists in the target "
            "language.\n"
            "- If a block is already entirely in the target language, output it "
            "unchanged.\n"
            "- Preserve numbering exactly: reply as '[n] translated text' per "
            "block, in the same order.\n"
            "- Do not merge or split blocks, and do not add explanations, notes or "
            "any preamble.\n"
            "- Output ONLY the numbered translations, nothing else.\n\n"
            "Example:\n"
            "Input:\n"
            "[1]\n"
            "Press OK to continue.\n"
            "[2]\n"
            "Save the file before exiting.\n"
            "Output:\n"
            "[1]\n"
            "点击“确定”继续。\n"
            "[2]\n"
            "退出前请保存文件。"
        )

    def _request_locked(self, prompt: str, system: str) -> str:
        """Issue one chat-completions request and return the assistant text."""
        kwargs: dict = {
            "model": self.model.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": (
                self.model.temperature if self.model.temperature is not None else 0.2
            ),
        }
        if self.model.max_tokens is not None:
            kwargs["max_tokens"] = self.model.max_tokens
        # Model-specific body params (e.g. llama.cpp ``reasoning_effort``) are sent
        # via ``extra_body`` so they reach the server regardless of client support.
        body_params = self.model.request_params()
        if body_params:
            kwargs["extra_body"] = body_params
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return (choice.message.content or "").strip()

    @staticmethod
    def _parse_response(
        text: str, blocks: Sequence[str], indices: Sequence[int]
    ) -> list[str]:
        """Map a model response back onto the requested blocks.

        Each ``[n]`` block may span several lines; internal line breaks are
        folded into spaces so one reply block becomes one translated block.
        """
        result: list[str] = []
        matched: dict[int, str] = {}
        for m in _MULTI_BLOCK_RE.finditer(text):
            pos = int(m.group(1)) - 1
            matched[pos] = " ".join(m.group(2).split())
        if matched:
            # Every requested block must have been echoed (once each).  A partial
            # reply — missing, duplicated or out-of-range ``[n]`` markers — is a
            # malformed response: do NOT silently fill gaps with the original text
            # and cache it, or a transient truncation would poison resume runs.
            # Raise so the caller treats it as a transient failure and retries
            # (falling back to the source text only after retries are exhausted).
            expected = set(range(len(indices)))
            if expected != set(matched):
                got = ", ".join(str(n + 1) for n in sorted(matched))
                raise ValueError(
                    "模型回复的块编号不完整（期望 "
                    f"{len(indices)} 块，回显 [{got}]），无法对齐"
                )
            result = [matched[p] for p in range(len(indices))]
            return result

        # Fallback: assume output lines correspond in order.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for n, i in enumerate(indices):
            if n < len(lines):
                result.append(lines[n])
            else:
                result.append(blocks[i])
        return result

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """True for errors worth retrying (network, 429, 5xx, parse issues).

        Permanent client errors (400/401/403/404) fail fast without retrying.
        """
        if isinstance(
            exc,
            (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError),
        ):
            return False
        if isinstance(exc, APIStatusError):
            return exc.status_code == 429 or exc.status_code >= 500
        return True

    def _translate_batch(
        self,
        indices: Sequence[int],
        blocks: Sequence[str],
        language: str,
        log: LogFn,
        cancel: CancelFn,
        retry_delays: Sequence[float] = _TRANSIENT_RETRY_DELAYS,
    ) -> tuple[list[str], bool]:
        """Translate one batch; returns ``(translations, ok)``.

        ``ok`` is False when every attempt failed — the source text is then
        preserved (content is never dropped) but must NOT be written to the
        cache, or a transient outage would poison it permanently.
        """
        prompt = self._build_prompt(blocks, indices)
        system = self._system_prompt(language)
        attempts = len(retry_delays) + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            if cancel():
                raise TranslationCancelled()
            try:
                raw = self._request_locked(prompt, system)
                parsed = self._parse_response(raw, blocks, indices)
                if len(parsed) == len(indices):
                    return parsed, True
                last_error = RuntimeError(
                    f"model returned {len(parsed)} results for "
                    f"{len(indices)} blocks"
                )
            except TranslationCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 — network / API errors
                last_error = exc
            if not self._is_transient(last_error):
                break
            if attempt < attempts:
                log(f"  重试 {attempt}/{attempts}: {last_error}")
                _sleep_interruptible(retry_delays[attempt - 1], cancel)
        if last_error:
            log(f"  批次失败，保留原文: {last_error}")
        # Preserve the source text for every block in the failed batch.
        return [blocks[i] for i in indices], False

    def translate_blocks(
        self,
        blocks: Sequence[str],
        target_language: str,
        on_progress: ProgressFn | None = None,
        log: LogFn | None = None,
        cancel: CancelFn | None = None,
        doc_path: Path | None = None,
        resume: bool = True,
        retry_delays: Sequence[float] = _TRANSIENT_RETRY_DELAYS,
    ) -> TranslationResult:
        """Translate ``blocks`` into ``target_language``.

        Batches are sent concurrently (``model.concurrency`` parallel requests)
        and results folded back in completion order; the output still aligns
        with the input block order.
        """
        log = log or (lambda _msg: None)
        cancel = cancel or (lambda: False)
        progress = on_progress or (lambda _d, _t: None)

        n = len(blocks)
        result = TranslationResult(blocks=list(blocks), translated=list(blocks))

        # Blocks without any letters (page numbers, separators, pure symbols)
        # are kept as-is: they need no translation and waste a request.
        skip = {i for i, b in enumerate(blocks) if not _needs_translation(b)}

        # Load the on-disk cache so repeated runs are cheap.
        cache: dict[str, str] = {}
        cache_path: Path | None = None
        if resume and doc_path is not None:
            cache = load_translation_cache(doc_path, target_language, self.model.id)
            cache_path = _cache_dir() / _cache_key(doc_path, target_language, self.model.id)

        # Progress starts at the count already present in the cache plus the
        # blocks skipped outright, so the bar reflects genuinely *done* work.
        done = sum(1 for b in blocks if _block_hash(b) in cache) + len(skip)
        progress(done, n)

        # A cache write may fail (read-only home, sandbox, full disk).  That
        # must not abort a translation, but staying silent is worse: the run
        # looks fine and every re-run re-translates everything.  Warn once.
        cache_warned = False

        def _persist_cache() -> None:
            nonlocal cache_warned
            if cache_path is None:
                return
            reason = _write_cache(cache_path, cache)
            if reason and not cache_warned:
                cache_warned = True
                log(
                    f"  警告：翻译缓存写入失败（{reason}），"
                    f"本次结果不会被缓存，重跑将重新翻译：{cache_path}"
                )

        def _needs_request(i: int) -> bool:
            return i not in skip and _block_hash(blocks[i]) not in cache

        chunks = self._make_chunks(blocks, index_filter=_needs_request)
        if chunks:
            max_workers = max(1, int(self.model.concurrency or 1))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        self._translate_batch,
                        chunk,
                        blocks,
                        target_language,
                        log,
                        cancel,
                        retry_delays,
                    ): chunk
                    for chunk in chunks
                }
                for fut in as_completed(futures):
                    chunk = futures[fut]
                    try:
                        translated, ok = fut.result()
                    except TranslationCancelled:
                        raise
                    except Exception as exc:  # noqa: BLE001 — defensive; the
                        # batch already swallows errors, this catches the rest
                        log(f"  批次异常，保留原文: {exc}")
                        translated, ok = [blocks[i] for i in chunk], False
                    if ok:
                        with self._cache_lock:
                            for i, text in zip(chunk, translated):
                                cache[_block_hash(blocks[i])] = text
                            # Persist after every batch so a cancel/crash keeps
                            # the work completed so far (resume reuses it).
                            _persist_cache()
                    else:
                        for i in chunk:
                            result.errors.append(f"块 {i + 1} 翻译失败，保留原文")
                    done += len(chunk)
                    progress(done, n)

        # Fill the output from the (now fully populated) cache.
        for i, b in enumerate(blocks):
            key = _block_hash(b)
            if key in cache:
                result.translated[i] = cache[key]

        if cache_path is not None:
            _persist_cache()

        progress(n, n)
        return result

    def _make_chunks(
        self,
        blocks: Sequence[str],
        index_filter: Callable[[int], bool] | None = None,
    ) -> list[list[int]]:
        """Split indices into chunks that fit the model's character budget."""
        index_filter = index_filter or (lambda _i: True)
        budget = max(1, int(self.model.batch_size or _CHAR_BUDGET))
        chunks: list[list[int]] = []
        current: list[int] = []
        current_chars = 0
        for i, block in enumerate(blocks):
            if not index_filter(i):
                continue
            size = len(block) + len(str(i)) + 6
            if current and (current_chars + size > budget):
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(i)
            current_chars += size
        if current:
            chunks.append(current)
        return chunks


def _write_cache(path: Path, cache: dict[str, str]) -> str | None:
    """Best-effort persist of the translation cache.

    Returns ``None`` on success, or a short human readable reason on failure.
    A cache write must never abort a translation that already succeeded, but
    the caller should surface the reason once — a silently unwritable cache
    looks exactly like a working one until the next (fully re-translated) run.
    """
    try:
        import json

        path.write_text(json.dumps(cache, ensure_ascii=False), "utf-8")
        return None
    except Exception as exc:  # noqa: BLE001 — the cache is optional by design
        return f"{type(exc).__name__}: {exc}"
