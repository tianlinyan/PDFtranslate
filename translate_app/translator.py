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

from .settings import DEFAULT_GLOSSARY_PATH, ModelConfig, load_glossary

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

#: How many newly translated blocks are kept in memory before one batched write
#: to the on-disk journal.  A WINDOW of up to this many blocks may be lost on a
#: hard crash (a clean cancel flushes the buffer, so it is not lost); pick a
#: larger value to write less often (kinder to an SSD), a smaller one to lose
#: less work if the app crashes mid-run.
_FLUSH_ENTRIES = 200

#: Rewrite the whole cache as a single snapshot (and drop the journal) once the
#: journal has grown past this many lines.  Keeps the journal from growing
#: unbounded without rewriting the snapshot on every run.
_COMPACT_ENTRIES = 5000

#: Fraction of ``max_tokens`` reserved for a batch's *output*.  A batch whose
#: estimated output approaches the model's completion cap is likely to be
#: truncated mid-reply, which the engine treats as a malformed response and
#: retries (wasting a request).  Keeping each batch's output under this many
#: tokens avoids those retries.
_OUTPUT_HEADROOM = 0.8

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
    """Return a writable cache directory, falling back to the system temp dir."""
    for path in (
        Path.home() / ".pdftranslate" / "cache",
        Path(tempfile.gettempdir()) / "pdftranslate_cache",
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            continue
    return Path.home() / ".pdftranslate" / "cache"


def _glossary_fingerprint(glossary: dict[str, str] | None) -> str:
    """A short hash of a glossary, so a glossary change never reuses old cache."""
    if not glossary:
        return ""
    import json

    return hashlib.sha1(
        json.dumps(glossary, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]


def _cache_key(
    doc_path: Path,
    target_lang: str,
    model_id: str,
    glossary: dict[str, str] | None = None,
) -> str:
    # The glossary is part of the key: switching or editing it must not serve
    # translations produced under the previous term mapping.
    h = hashlib.sha1(
        f"{doc_path.resolve()}|{target_lang}|{model_id}|{_glossary_fingerprint(glossary)}".encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return f"trans_v{_CACHE_VERSION}_{h}.json"


def _block_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _needs_translation(text: str) -> bool:
    """Return True if ``text`` contains letters and thus needs translation."""
    return bool(_LETTERS_RE.search(text))


def _estimate_tokens(text: str) -> int:
    """A rough token estimate for batching purposes.

    CJK and other wide scripts are near one token per character; Latin-script
    text is more like four characters per token.  Used only to keep a batch's
    expected output under the model's ``max_tokens`` so replies are not
    truncated into a wasteful retry, so a heuristic is plenty.
    """
    cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
    latin = len(text) - cjk
    return cjk + latin // 4 + (1 if latin % 4 else 0)


def _sleep_interruptible(seconds: float, cancel: CancelFn) -> None:
    """Sleep in small slices so a cancellation stays responsive."""
    if seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if cancel():
            raise TranslationCancelled()
        time.sleep(min(0.2, deadline - time.monotonic()))


def load_translation_cache(
    doc_path: Path,
    target_lang: str,
    model_id: str,
    glossary: dict[str, str] | None = None,
) -> dict[str, str]:
    """Load the on-disk translation cache for a doc/lang/model (empty if none).

    The cache is stored as a base ``.json`` snapshot plus an append-only
    ``.jsonl`` journal of newer entries; both are merged on load.  A batch's
    translations land in the journal instantly (so a cancel/crash never loses
    completed work), while the bulk snapshot is only rewritten on compaction.
    """
    import json

    cache_path = _cache_dir() / _cache_key(doc_path, target_lang, model_id, glossary)
    data: dict[str, str] = {}
    for file in (cache_path, _cache_journal_path(cache_path)):
        if not file.exists():
            continue
        try:
            if file.suffix == ".json":
                data.update(json.loads(file.read_text("utf-8")))
            else:
                for line in file.read_text("utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.update(json.loads(line))
                    except Exception:
                        # A torn line from a crash is skipped, not fatal.
                        continue
        except Exception:
            pass
    return data


def clear_translation_cache() -> int:
    """Delete all cached translation files; returns the number of files removed."""
    removed = 0
    for path in _cache_dir().glob("trans_*"):
        try:
            if path.is_file():
                path.unlink()
                removed += 1
        except Exception:
            pass
    return removed


class TranslationEngine:
    """Wraps one AI model and translates a list of text blocks."""

    def __init__(self, model: ModelConfig, glossary: dict[str, str] | None = None):
        self.model = model
        self.client = OpenAI(**model.client_kwargs())
        # Guards the shared translation cache while batches complete concurrently.
        self._cache_lock = threading.Lock()
        # Glossary of ``source -> target`` terms injected into every batch's
        # prompt so the same domain term stays identical across all chunks.
        # Fall back to the model's own glossary file, then the project default.
        if glossary is None:
            glossary = load_glossary(model.glossary or DEFAULT_GLOSSARY_PATH)
        self._glossary = glossary or {}

    @staticmethod
    def _build_prompt(blocks: Sequence[str], indices: Sequence[int]) -> str:
        lines = []
        # Number the blocks 1..k within this batch so the model always receives a
        # contiguous, unambiguous set to echo back (avoids misalignment when the
        # batch's global indices are sparse because some blocks were cached).
        for pos, i in enumerate(indices):
            lines.append(f"[{pos + 1}]\n{blocks[i]}")
        return "\n\n".join(lines)

    def _batch_glossary(
        self, indices: Sequence[int], blocks: Sequence[str]
    ) -> dict[str, str]:
        """The glossary terms that actually appear in this batch's source blocks.

        Only terms present here are injected into the batch's prompt, so a large
        glossary is never re-sent verbatim to every chunk (which would burn
        tokens on terms that cannot occur).  Consistency is unaffected — a term
        that appears in a chunk is always constrained there, and a term absent
        from a chunk needs no constraint.
        """
        if not self._glossary:
            return {}
        text = "\n".join(blocks[i] for i in indices)
        return {
            src: tgt
            for src, tgt in self._glossary.items()
            if src and src in text
        }

    @staticmethod
    def _system_prompt(language: str, glossary: dict[str, str] | None = None) -> str:
        prompt = (
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
            "- Output ONLY the numbered translations, nothing else.\n"
        )
        if glossary:
            lines = [
                "Glossary — translate the following terms exactly as given. Never "
                "rephrase or paraphrase them, and use the same term throughout the "
                "whole document so the terminology stays consistent:"
            ]
            for src, tgt in glossary.items():
                lines.append(f"- {src} => {tgt}")
            prompt += "\n".join(lines) + "\n"
        return (
            prompt
            + "\nExample:\n"
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
        system = self._system_prompt(language, self._batch_glossary(indices, blocks))
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

        if self._glossary:
            log(f"已应用术语表（{len(self._glossary)} 条），保证专用词汇跨分块一致")

        n = len(blocks)
        result = TranslationResult(blocks=list(blocks), translated=list(blocks))

        # Blocks without any letters (page numbers, separators, pure symbols)
        # are kept as-is: they need no translation and waste a request.
        skip = {i for i, b in enumerate(blocks) if not _needs_translation(b)}

        # Load the on-disk cache so repeated runs are cheap.
        cache: dict[str, str] = {}
        cache_path: Path | None = None
        if resume and doc_path is not None:
            cache = load_translation_cache(doc_path, target_language, self.model.id, self._glossary)
            cache_path = _cache_dir() / _cache_key(
                doc_path, target_language, self.model.id, self._glossary
            )

        # Progress starts at the count already present in the cache plus the
        # blocks skipped outright, so the bar reflects genuinely *done* work.
        done = sum(1 for b in blocks if _block_hash(b) in cache) + len(skip)
        progress(done, n)

        def _needs_request(i: int) -> bool:
            return i not in skip and _block_hash(blocks[i]) not in cache

        chunks = self._make_chunks(blocks, index_filter=_needs_request)
        # Translations are buffered in memory and written to disk in large
        # batches (not per batch), so a long document performs only a handful of
        # writes — sparing the SSD while still bounding crash loss.
        pending: dict[str, str] = {}
        flushed_since_compact = 0

        def _flush_pending() -> None:
            nonlocal flushed_since_compact
            if cache_path is None or not pending:
                return
            _append_cache_journal(_cache_journal_path(cache_path), pending)
            flushed_since_compact += len(pending)
            pending.clear()

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
                        # Keep the work completed so far durable before bailing.
                        _flush_pending()
                        raise
                    except Exception as exc:  # noqa: BLE001 — defensive; the
                        # batch already swallows errors, this catches the rest
                        log(f"  批次异常，保留原文: {exc}")
                        translated, ok = [blocks[i] for i in chunk], False
                    if ok:
                        with self._cache_lock:
                            for i, text in zip(chunk, translated):
                                key = _block_hash(blocks[i])
                                cache[key] = text
                                pending[key] = text
                        if len(pending) >= _FLUSH_ENTRIES:
                            _flush_pending()
                    else:
                        for i in chunk:
                            result.errors.append(f"块 {i + 1} 翻译失败，保留原文")
                    done += len(chunk)
                    progress(done, n)

        # Merge the in-memory cache back into the output, then flush whatever is
        # buffered and compact only when the journal has grown large — so a
        # fully-cached re-run performs no disk writes at all.
        for i, b in enumerate(blocks):
            key = _block_hash(b)
            if key in cache:
                result.translated[i] = cache[key]

        _flush_pending()
        if cache_path is not None and flushed_since_compact >= _COMPACT_ENTRIES:
            _compact_cache(cache_path, _cache_journal_path(cache_path), cache)

        progress(n, n)
        return result

    def _make_chunks(
        self,
        blocks: Sequence[str],
        index_filter: Callable[[int], bool] | None = None,
    ) -> list[list[int]]:
        """Split indices into chunks that fit the model's character budget.

        Besides the character budget, when the model declares a ``max_tokens``
        the chunks are also bounded so a batch's *estimated output* stays under
        a safe fraction of that cap.  A batch whose output would be truncated is
        treated as a malformed reply and retried, costing a request; keeping
        each batch comfortably under the cap avoids that waste.
        """
        index_filter = index_filter or (lambda _i: True)
        budget = max(1, int(self.model.batch_size or _CHAR_BUDGET))
        max_out = (
            int(self.model.max_tokens * _OUTPUT_HEADROOM)
            if self.model.max_tokens
            else None
        )
        chunks: list[list[int]] = []
        current: list[int] = []
        current_chars = 0
        current_tokens = 0
        for i, block in enumerate(blocks):
            if not index_filter(i):
                continue
            size = len(block) + len(str(i)) + 6
            toks = _estimate_tokens(block) + 4  # + label/number overhead
            over_char = current and current_chars + size > budget
            over_token = (
                current
                and max_out is not None
                and current_tokens + toks > max_out
            )
            if over_char or over_token:
                chunks.append(current)
                current = []
                current_chars = 0
                current_tokens = 0
            current.append(i)
            current_chars += size
            current_tokens += toks
        if current:
            chunks.append(current)
        return chunks


def _cache_journal_path(cache_path: Path) -> Path:
    """The append-only ``.jsonl`` sibling of a cache snapshot file."""
    return cache_path.with_suffix(".jsonl")


def _append_cache_journal(journal_path: Path, entries: dict[str, str]) -> None:
    """Append new entries as JSON lines; one small write per batch, no rewrite."""
    if not entries:
        return
    try:
        import json

        with journal_path.open("a", encoding="utf-8") as fh:
            for key, value in entries.items():
                fh.write(json.dumps({key: value}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _compact_cache(
    cache_path: Path, journal_path: Path, cache: dict[str, str]
) -> None:
    """Write the merged cache as a single snapshot and drop the journal."""
    try:
        import json

        cache_path.write_text(json.dumps(cache, ensure_ascii=False), "utf-8")
        if journal_path.exists():
            journal_path.unlink()
    except Exception:
        pass
