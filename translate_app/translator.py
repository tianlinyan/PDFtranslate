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
* A fatal configuration error (wrong API key, unknown model) aborts the run
  instead of quietly handing back the untranslated source as a "result".
"""

from __future__ import annotations

import base64
import hashlib
import json
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

from . import prompts
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
#: stale translations from an older run are never reused.  Bumped to 4 when the
#: unnumbered-reply fallback was tightened: caches written before that could
#: contain a model's refusal / preamble line stored as block 1's "translation",
#: and such an entry would otherwise be reused forever.  Bumped to 5 when the
#: glossary entered the key and the name romanization rule was added; to 6 when
#: the name-order rule changed to given name first (cached entries made under
#: the old surname-first rule would otherwise be reused unchanged); to 7 when the
#: numbering rule was hardened to default to ARABIC (一、二、三 / （二） → 1., 2., (2)),
#: so a cache written under the old, loose style (which let the model emit I., II.,
#: (XXXIII)) is never reused.
_CACHE_VERSION = 7

#: Delay between batch attempts (seconds); injectable so tests don't sleep.
_TRANSIENT_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0)

#: Any Unicode letter makes a block worth translating.  Blocks without any
#: letters (page numbers, separators, pure symbols) are kept as-is and never
#: sent to the model.  ``[^\W\d_]`` matches letters of every script — CJK,
#: kana, hangul, Greek, Cyrillic, Hebrew, Arabic, accented Latin (é/Ü/ñ),
#: … — instead of a hand-maintained range list that silently skipped
#: non-ASCII-Latin text.
_LETTERS_RE = re.compile(r"[^\W\d_]")

#: The few-shot output lines for the prompt's example, keyed by the target
#: language's lowercase name.  If the language is missing here the prompt falls
#: back to the example whose output is in Chinese (the default target).  The
#: *input* lines of the example stay in English for every target — they only
#: illustrate the numbered pair format, and keeping them fixed avoids nudging a
#: translation into the input language.  (These live in ``translate_app.prompts``.)

ProgressFn = Callable[[int, int], None]
LogFn = Callable[[str], None]
CancelFn = Callable[[], bool]


class TranslationCancelled(Exception):
    """Raised when a user cancels an in-flight translation."""


class TranslationAborted(Exception):
    """Raised when a fatal configuration error stops the whole run.

    Unlike a per-batch failure (which keeps the source text and carries on),
    a bad API key or a misspelled model name fails identically for every batch.
    Carrying on would produce a "finished" document that is really just the
    untranslated source, so the run stops and the GUI reports the reason.
    """


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

    ``PDFTRANSLATE_CACHE_DIR`` overrides the location (mirroring
    ``PDFTRANSLATE_OCR_CACHE_DIR`` on the OCR side) so tests — and users with a
    read-only home — can point the cache somewhere else.

    Creating the directory is not enough to prove it is usable: an existing
    directory under a read-only / sandboxed ``$HOME`` lets ``mkdir`` succeed
    (``exist_ok=True`` is a no-op) while every later write is denied.  That
    silently disables resume, so each candidate is verified with a real probe
    write before it is accepted.
    """
    candidates = [
        Path.home() / ".pdftranslate" / "cache",
        Path(tempfile.gettempdir()) / "pdftranslate_cache",
    ]
    override = os.environ.get("PDFTRANSLATE_CACHE_DIR")
    if override:
        candidates.insert(0, Path(override))
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / f".write_probe_{os.getpid()}"
            probe.write_text("", "utf-8")
            probe.unlink()
            return path
        except Exception:
            continue
    return Path.home() / ".pdftranslate" / "cache"


def _cache_key(
    doc_path: Path, target_lang: str, model_id: str, glossary_hash: str = ""
) -> str:
    # ``glossary_hash`` is part of the key: a run with a different glossary must
    # not reuse translations made without it, or the new rules would silently
    # never apply to blocks that already hit the cache.
    h = hashlib.sha1(
        f"{doc_path.resolve()}|{target_lang}|{model_id}|{glossary_hash}".encode("utf-8")
    ).hexdigest()[:16]
    return f"trans_v{_CACHE_VERSION}_{h}.json"


def _load_glossary(doc_path: Path, log: Callable[[str], None]) -> dict[str, str]:
    """Load ``glossary.json`` next to the document, if present.

    The file is a flat JSON object ``{"source term": "target term", ...}``.
    Anything malformed is reported through ``log`` and treated as no glossary
    (a silent skip would make the user believe their terms apply when they
    do not).
    """
    path = doc_path.parent / "glossary.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text("utf-8"))
    except Exception:
        log(f"  警告：术语表无法解析（跳过）：{path}")
        return {}
    if not isinstance(data, dict):
        log(f"  警告：术语表格式错误（应为 JSON 对象）：{path}")
        return {}
    glossary = {str(k): str(v) for k, v in data.items() if str(k).strip()}
    if not glossary:
        log(f"  警告：术语表为空：{path}")
    return glossary


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


def load_translation_cache(
    doc_path: Path,
    target_lang: str,
    model_id: str,
    glossary_hash: str = "",
) -> dict[str, str]:
    """Load the on-disk translation cache for a doc/lang/model (empty if none)."""
    cache_path = _cache_dir() / _cache_key(doc_path, target_lang, model_id, glossary_hash)
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text("utf-8"))
            # Anything but a JSON object is a foreign / corrupted file: ignore
            # it rather than letting ``cache[key] = ...`` blow up mid-run.
            if isinstance(data, dict):
                # Drop empty / whitespace-only translations: caches written
                # before the empty-reply check could hold a poisoned "" entry
                # that would blank the block in every export.  Treating it as
                # "not cached" re-translates the block and self-heals the file.
                return {
                    str(k): str(v) for k, v in data.items() if str(v).strip()
                }
        except Exception:
            pass
    return {}


def clear_translation_cache() -> int:
    """Delete all cached translation files; returns the number of files removed."""
    removed = 0
    # ``trans_*`` (not ``trans_*.json``) also sweeps up any ``.tmp`` file left
    # behind by an interrupted atomic write.
    for path in _cache_dir().glob("trans_*"):
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
        # The cache *file* is rewritten by several finishing batches; serialize
        # the disk I/O separately from the in-memory dict, and drop stale
        # snapshots (by sequence number) so an old batch persisting late can
        # never overwrite a newer one.
        self._persist_lock = threading.Lock()
        self._cache_seq = 0
        self._persisted_seq = 0

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
    def _system_prompt(language: str, glossary: dict[str, str] | None = None) -> str:
        # All translation prompt text lives in ``translate_app.prompts`` so it is
        # easy to review/tune without hunting through the engine.  Kept as a thin
        # wrapper for tests that call ``TranslationEngine._system_prompt`` directly.
        return prompts.translation_system_prompt(language, glossary)

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
    def _parse_response(text: str, indices: Sequence[int]) -> list[str]:
        """Map a model response back onto the requested blocks.

        Each ``[n]`` block may span several lines; internal line breaks are
        folded into spaces so one reply block becomes one translated block.
        A reply that cannot be aligned with certainty — including one with an
        empty translation for any block — raises ``ValueError``: the source
        text is never mixed into a "successful" result, and an empty
        translation is never accepted, because such a result would be cached
        and reused forever (blanking the block in every future export).
        """
        matched: dict[int, str] = {}
        seen: set[int] = set()
        for m in _MULTI_BLOCK_RE.finditer(text):
            pos = int(m.group(1)) - 1
            # A duplicate ``[n]`` (the same number echoed twice) would silently
            # overwrite the earlier translation via ``matched[pos] = ...`` below —
            # and a set comparison could not detect it, so a reply like
            # ``[1]a\n[1]b\n[2]c`` for a 2-block batch would "succeed" and return
            # ``["b", "c"]``, dropping block 1's first translation then caching it
            # forever.  Reject duplicates explicitly instead.
            if pos in seen:
                raise ValueError(f"模型回复中编号 [{pos + 1}] 重复出现，无法对齐")
            seen.add(pos)
            matched[pos] = " ".join(m.group(2).split())
        if matched:
            # Every requested block must have been echoed (once each).  A partial
            # reply — missing, duplicated or out-of-range ``[n]`` markers — is a
            # malformed response: do NOT silently fill gaps with the original text
            # and cache it, or a transient truncation would poison resume runs.
            # Raise so the caller treats it as a transient failure and retries
            # (falling back to the source text only after retries are exhausted).
            expected = set(range(len(indices)))
            if expected != seen:
                got = ", ".join(str(n + 1) for n in sorted(seen))
                raise ValueError(
                    "模型回复的块编号不完整（期望 "
                    f"{len(indices)} 块，回显 [{got}]），无法对齐"
                )
            result = [matched[p] for p in range(len(indices))]
            # An *empty* translation is as bad as a missing one: accepting it
            # would blank out the block in the export AND write the empty string
            # to the cache, where it would be reused forever.  Reject it so the
            # caller retries (and, once retries are exhausted, keeps the source).
            if any(not t for t in result):
                raise ValueError("模型回复中存在空译文块，无法对齐")
            return result


        # Fallback: the reply carries no ``[n]`` marker at all.  Mapping lines
        # onto blocks by position is only defensible when the reply really does
        # have one line per requested block.  Accepting anything else let a
        # single-line refusal ("Sorry, I cannot translate this.") become block
        # 1's translation while the rest silently kept the source text — and
        # because that counts as success, the garbage was written to the cache
        # and reused forever.  Reject instead: the caller retries and, once the
        # retries are exhausted, preserves the source text without caching it.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(indices) == 1:
            # One requested block: the whole reply belongs to it (models often
            # wrap a long translation over several lines).  An all-whitespace
            # reply would fold to "" and blank the block, so reject it.
            folded = " ".join(text.split())
            if not folded:
                raise ValueError("模型回复为空，无法对齐")
            return [folded]
        if len(lines) != len(indices):
            raise ValueError(
                "模型回复既无 [n] 编号，行数也与块数不符"
                f"（期望 {len(indices)} 行，实际 {len(lines)} 行），无法对齐"
            )
        return list(lines)

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """True for errors worth retrying (network, 408, 429, 5xx, parse issues).

        Permanent client errors (400/401/403/404) fail fast without retrying.
        Note: once the SDK's own internal retries are exhausted, a server-side
        408 (request timeout) surfaces as a plain ``APIStatusError`` — it is
        transient by nature and must be retried here, not given up on.
        """
        if isinstance(
            exc,
            (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError),
        ):
            return False
        if isinstance(exc, APIStatusError):
            return exc.status_code in (408, 429) or exc.status_code >= 500
        return True

    @staticmethod
    def _is_fatal(exc: Exception | None) -> bool:
        """True for configuration errors that doom *every* batch equally.

        A wrong API key, a revoked key or a misspelled model / endpoint fails
        the same way for every request in the run.  Continuing would burn one
        doomed request per batch and then hand back a document in which every
        block "kept the original text" — a file that looks like a finished
        translation but is just the source.  Such a run must stop loudly.

        ``BadRequestError`` (400) is deliberately NOT fatal: it is usually
        data-dependent (one oversized batch), so the other batches may succeed.
        """
        return isinstance(
            exc, (AuthenticationError, PermissionDeniedError, NotFoundError)
        )

    @staticmethod
    def _fatal_message(exc: Exception) -> str:
        if isinstance(exc, AuthenticationError):
            return f"API 认证失败，请检查 models.json 中的 api_key：{exc}"
        if isinstance(exc, PermissionDeniedError):
            return f"API 拒绝访问，密钥无权调用该模型：{exc}"
        if isinstance(exc, NotFoundError):
            return f"接口或模型不存在，请检查 endpoint 与 model：{exc}"
        return str(exc)

    def _translate_batch(
        self,
        indices: Sequence[int],
        blocks: Sequence[str],
        language: str,
        log: LogFn,
        cancel: CancelFn,
        retry_delays: Sequence[float] = _TRANSIENT_RETRY_DELAYS,
        abort: threading.Event | None = None,
        glossary: dict[str, str] | None = None,
    ) -> tuple[list[str], bool]:
        """Translate one batch; returns ``(translations, ok)``.

        ``ok`` is False when every attempt failed — the source text is then
        preserved (content is never dropped) but must NOT be written to the
        cache, or a transient outage would poison it permanently.

        A fatal configuration error raises :class:`TranslationAborted` and sets
        ``abort`` so the batches still queued behind it return immediately
        instead of repeating the same doomed request.
        """
        prompt = self._build_prompt(blocks, indices)
        system = self._system_prompt(language, glossary)
        attempts = len(retry_delays) + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            if cancel():
                raise TranslationCancelled()
            if abort is not None and abort.is_set():
                # Another batch already hit a fatal error; this run is over.
                return [blocks[i] for i in indices], False
            try:
                raw = self._request_locked(prompt, system)
                # Either returns exactly ``len(indices)`` translations or raises.
                return self._parse_response(raw, indices), True
            except TranslationCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 — network / API errors
                last_error = exc
            # The watchdog may have aborted this request because a cancel (or a
            # fatal error) fired *mid-flight*.  Treat that as the control signal it
            # represents, not as a transient glitch worth retrying.
            if cancel():
                raise TranslationCancelled()
            if abort is not None and abort.is_set():
                return [blocks[i] for i in indices], False
            if self._is_fatal(last_error):
                if abort is not None:
                    abort.set()
                raise TranslationAborted(self._fatal_message(last_error))
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
        keep_original: set[int] | None = None,
    ) -> TranslationResult:
        """Translate ``blocks`` into ``target_language``.

        Batches are sent concurrently (``model.concurrency`` parallel requests)
        and results folded back in completion order; the output still aligns
        with the input block order.  ``keep_original`` is a set of block indices
        that must be left verbatim (e.g. a personal-name column); those blocks
        are never sent to the model and always export as the source text.
        """
        log = log or (lambda _msg: None)
        cancel = cancel or (lambda: False)
        progress = on_progress or (lambda _d, _t: None)
        keep = keep_original or set()

        n = len(blocks)
        result = TranslationResult(blocks=list(blocks), translated=list(blocks))

        # Blocks without any letters (page numbers, separators, pure symbols)
        # and blocks flagged keep-original (names) are kept as-is: they need no
        # translation and waste a request.
        skip = {i for i, b in enumerate(blocks) if not _needs_translation(b)} | keep

        # A glossary.json next to the document pins the terminology; its content
        # takes part in the cache key so adding one always takes effect.
        glossary = _load_glossary(doc_path, log) if doc_path is not None else {}
        if glossary:
            log(f"  已加载 {len(glossary)} 条术语表：{doc_path.parent / 'glossary.json'}")

        # Load the on-disk cache so repeated runs are cheap.
        cache: dict[str, str] = {}
        cache_path: Path | None = None
        if resume and doc_path is not None:
            # An absent glossary keeps the hash empty, so the cache key is the
            # same as before the glossary feature existed (only the version tag
            # differs); a glossary's content is part of the key.
            glossary_hash = (
                hashlib.sha1(
                    json.dumps(glossary, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()[:16]
                if glossary else ""
            )
            cache = load_translation_cache(
                doc_path, target_language, self.model.id, glossary_hash
            )
            cache_path = _cache_dir() / _cache_key(
                doc_path, target_language, self.model.id, glossary_hash
            )

        # Progress starts at the count already present in the cache plus the
        # blocks skipped outright, so the bar reflects genuinely *done* work.
        done = sum(1 for b in blocks if _block_hash(b) in cache) + len(skip)
        progress(done, n)

        # A cache write may fail (read-only home, sandbox, full disk).  That
        # must not abort a translation, but staying silent is worse: the run
        # looks fine and every re-run re-translates everything.  Warn once.
        cache_warned = False

        def _persist_cache(snapshot: dict[str, str], seq: int) -> None:
            nonlocal cache_warned
            if cache_path is None:
                return
            # The disk write is serialized by ``_persist_lock`` — *not*
            # ``_cache_lock`` (which only guards the in-memory dict) — so one
            # batch's slow write cannot stall every other batch's completion.
            # A snapshot that is no longer the newest is dropped: a batch that
            # finished early but persisted late must not overwrite a newer
            # batch's work.
            with self._persist_lock:
                if seq < self._persisted_seq:
                    return
                self._persisted_seq = seq
                reason = _write_cache(cache_path, snapshot)
            if reason:
                # Warn at most once per run, even though several batches may
                # fail to persist concurrently.
                with self._cache_lock:
                    if cache_warned:
                        return
                    cache_warned = True
                log(
                    f"  警告：翻译缓存写入失败（{reason}），"
                    f"本次结果不会被缓存，重跑将重新翻译：{cache_path}"
                )

        def _needs_request(i: int) -> bool:
            return i not in skip and _block_hash(blocks[i]) not in cache
        # Keep-original blocks must never be pulled from the cache either, or a
        # run *before* this change (when the name was transliterated) would
        # quietly hand back the old transliteration.
        for i in keep:
            cache.pop(_block_hash(blocks[i]), None)

        chunks = self._make_chunks(blocks, index_filter=_needs_request)
        if chunks:
            max_workers = max(1, int(self.model.concurrency or 1))
            # Set as soon as one batch hits a fatal configuration error, so the
            # batches queued behind it give up instead of repeating it.
            abort = threading.Event()

            # A watchdog that closes the underlying HTTP client the moment a
            # cancel (or a fatal error) fires.  ``client.close()`` aborts any
            # request still in flight, so 取消 can interrupt a long-running
            # request instead of letting it run to its timeout — without it the
            # cancel flag is polled only *between* attempts, and a single
            # in-flight request (up to the 300s client timeout) blocks the worker.
            watchdog_stop = threading.Event()

            def _watchdog() -> None:
                while not watchdog_stop.is_set():
                    if cancel() or abort.is_set():
                        try:
                            self.client.close()
                        except Exception:  # noqa: BLE001 — best effort
                            pass
                        return
                    time.sleep(0.05)

            watchdog = threading.Thread(target=_watchdog, daemon=True)
            watchdog.start()
            try:
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
                            abort,
                            glossary,
                        ): chunk
                        for chunk in chunks
                    }
                    for fut in as_completed(futures):
                        # Honour a cancel as soon as the next batch resolves,
                        # rather than draining every queued batch first.
                        if cancel():
                            raise TranslationCancelled()
                        chunk = futures[fut]
                        try:
                            translated, ok = fut.result()
                        except (TranslationCancelled, TranslationAborted):
                            raise
                        except Exception as exc:  # noqa: BLE001 — defensive; the
                            # batch already swallows errors, this catches the rest
                            log(f"  批次异常，保留原文: {exc}")
                            translated, ok = [blocks[i] for i in chunk], False
                        if ok:
                            with self._cache_lock:
                                for i, text in zip(chunk, translated):
                                    cache[_block_hash(blocks[i])] = text
                                snapshot = dict(cache)
                                seq = self._cache_seq
                                self._cache_seq += 1
                            # Persist after every batch so a cancel/crash keeps
                            # the work completed so far (resume reuses it).  The
                            # snapshot is taken under the lock; the disk I/O is
                            # not, so concurrent batches do not serialize on it.
                            _persist_cache(snapshot, seq)
                        else:
                            for i in chunk:
                                result.errors.append(f"块 {i + 1} 翻译失败，保留原文")
                        done += len(chunk)
                        progress(done, n)
            finally:
                watchdog_stop.set()
                # The engine's client is deliberately NOT closed here: the agent
                # layer reuses ONE engine across many ``translate_blocks`` calls
                # in a page, and closing the client after every batch made the
                # 2nd+ call fail with "Connection error" (a closed httpx client
                # is unusable — see test_reusable_engine).  The watchdog still
                # closes it on cancel/abort; otherwise the process hard-exits
                # (os._exit), so an idle connection pool is bounded and harmless.

        # Fill the output from the (now fully populated) cache.  Keep-original
        # blocks stay the source text (their translations were never requested).
        for i, b in enumerate(blocks):
            if i in keep:
                continue
            key = _block_hash(b)
            if key in cache:
                result.translated[i] = cache[key]

        # Final persist (no concurrency left — the pool has exited).
        if cache_path is not None:
            _persist_cache(cache, self._cache_seq)

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
    """Best-effort *atomic* persist of the translation cache.

    Returns ``None`` on success, or a short human readable reason on failure.
    A cache write must never abort a translation that already succeeded, but
    the caller should surface the reason once — a silently unwritable cache
    looks exactly like a working one until the next (fully re-translated) run.

    The write goes to a temp file and is swapped in with :func:`os.replace`:
    the cache is rewritten after every batch while the GUI may hard-exit the
    process at any moment (``os._exit`` on window close), and a half-written
    JSON file is unreadable — which looks like "no cache at all" and quietly
    re-translates the entire document on the next run.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(cache, ensure_ascii=False), "utf-8")
        os.replace(tmp, path)
        return None
    except Exception as exc:  # noqa: BLE001 — the cache is optional by design
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return f"{type(exc).__name__}: {exc}"


def _image_data_url(png: bytes) -> str:
    """Encode a PNG as a ``data:image/png;base64,`` URL for the chat API."""
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _extract_json_object(content: str) -> dict | None:
    """Pull the outer ``{ ... }`` out of a model reply and decode it.

    The reply often wraps the object in prose or markdown fences, so the whole
    reply is scanned for the outermost brace pair.  Anything malformed returns
    ``None`` (the caller treats it as an empty / no-op result).
    """
    if not content:
        return None
    start = content.find("{")
    if start < 0:
        return None
    end = content.rfind("}")
    if end <= start:
        return None
    try:
        data = json.loads(content[start:end + 1])
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _vision_call(client, model: str, prompt: str, images: list[bytes],
                 temperature: float = 0.0) -> str:
    """Send ``prompt`` plus ``images`` to the model and return the assistant text."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": _image_data_url(img)}}
        for img in images
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
    )
    return getattr(resp.choices[0].message, "content", None) or ""


#: Lead-in for the block-classification reviewer.  ``@@CANDIDATES@@`` is replaced
#: with the numbered candidate list before the call (the JSON braces in the
#: output spec would collide with ``str.format``, so a plain replace is used).
_REVIEW_CLASSIFY_PROMPT = (
    "这是原图页面。下面是程序判定为「组织结构图/架构图节点标签，应保留原文不翻译」的候选块。"
    "请结合原图判断每一块真的是节点标签（应保留），还是其实是**应该翻译**的标题/正文/图表说明。\n\n"
    "候选块：\n"
    "@@CANDIDATES@@\n\n"
    "只输出一个 JSON 对象，用 index 对应上面的编号：\n"
    "{\"classifications\": [{\"index\": 0, "
    "\"kind\": \"keep_chart_node|keep_verbatim|signature|translate_heading|translate_prose\", "
    "\"confidence\": 0.9, \"message\": \"一句话说明\"}]}\n"
    "kind 说明：keep_chart_node/keep_verbatim=节点或整体保留；signature=手写签字；"
    "translate_heading/translate_prose=应翻译的标题/正文。只列出你有把握的块；没有则用空数组。"
    "不要输出除此 JSON 之外的文字。"
)


#: ``kind`` values that mean a candidate block should actually be translated.
_REVIEW_KIND_TRANSLATE = frozenset(("translate_heading", "translate_prose"))


def _parse_classify(content: str) -> dict:
    """Parse a block-classification reply into ``{"classifications": [...]}``."""
    data = _extract_json_object(content)
    if data is None:
        return {}
    raw = data.get("classifications")
    return {
        "classifications": [
            c for c in (raw if isinstance(raw, list) else []) if isinstance(c, dict)
        ]
    }


def make_classify_review_fn(
    model: ModelConfig, log: Callable[[str], None] | None = None
) -> Callable[[int, bytes, Sequence[tuple]], dict] | None:
    """Return a block-classification callback for ``model`` (or ``None`` if disabled).

    Signature ``(page_index, original_png, candidates) -> dict``, where
    ``candidates`` is ``[(flat_index, bbox, text), ...]``.  It invoices the model
    to say whether each rule-kept candidate is really a structural label (keep)
    or actually translatable content, returning ``{"classifications": [...]}``.
    Best-effort: any failure returns ``{}``, so the caller keeps the rule's
    decision unchanged (safe).
    """
    if not model.vision:
        return None
    client = OpenAI(**model.client_kwargs())

    def _classify(page_index: int, original_png: bytes, candidates: Sequence[tuple]) -> dict:
        try:
            lines = [
                f"[{idx}] bbox=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}) text={str(text)[:40]!r}"
                for idx, b, text in candidates
            ]
            prompt = _REVIEW_CLASSIFY_PROMPT.replace("@@CANDIDATES@@", "\n".join(lines))
            content = _vision_call(client, model.model, prompt, [original_png])
            return _parse_classify(content)
        except Exception as exc:  # noqa: BLE001 — best-effort, fail-closed
            if log:
                log(f"  第 {page_index + 1} 页保留复核失败：{type(exc).__name__}: {exc}")
            return {}

    return _classify


#: Direct re-translation of a single block, used by the QA→correction pass so a
#: residual-Chinese or empty cell is re-translated *without* hitting the cache
#: (which would return the same stale value).  The model is told to output only
#: the translation in the target language and never leave the source language.
_RETRANSLATE_PROMPT = (
    "请把下面这段文字翻译成 {lang}。只输出译文本身，不要任何解释；"
    "如果它已经是目标语言，原样输出。不要保留任何原文语言：\n{text}"
)


def make_retranslate_fn(
    model: ModelConfig, log: Callable[[str], None] | None = None
) -> Callable[[str, str], str] | None:
    """Return a direct single-text re-translation callback, or ``None`` if disabled.

    Signature ``(text, target_language) -> str``: sends one block to the model
    with a "translate only, never leave the source language" prompt (cache and
    number-protocol bypassed).  Best-effort — any failure returns the original
    text, so a correction never makes a cell worse.
    """
    if not model.vision:
        return None
    client = OpenAI(**model.client_kwargs())

    def _retranslate(text: str, lang: str) -> str:
        try:
            prompt = _RETRANSLATE_PROMPT.format(lang=lang, text=text)
            resp = client.chat.completions.create(
                model=model.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = getattr(resp.choices[0].message, "content", None) or ""
            out = str(content).strip()
            return out if out else text
        except Exception as exc:  # noqa: BLE001 — fail-closed, keep the original
            if log:
                log(f"  质检修正重译失败：{type(exc).__name__}: {exc}")
            return text

    return _retranslate


#: OpenAI ``tools`` schema for the block-classification judgment tool, used by
#: the tool-use POC.  The model calls it per ambiguous block; the deterministic
#: caller applies the decision (release / keep) rather than letting the model
#: draw the table.
_CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_block",
        # Description is prompt text, authored centrally (see ``prompts``).
        "description": prompts.AGENT_TOOL_DESCRIPTIONS["classify_block"],
        "parameters": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "输入块列表中的序号"},
                "text": {"type": "string", "description": "块文本"},
                "action": {"type": "string",
                           "enum": ["keep", "translate", "signature"],
                           "description": "keep=保留原文；translate=翻译；signature=手写签字"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string", "description": "一句话理由"},
            },
            "required": ["index", "text", "action"],
        },
    },
}


def _parse_classify_tool_calls(msg) -> list[dict]:
    """Turn the model's ``classify_block`` tool_calls into decision dicts.

    Each entry is ``{"index", "text", "action", "confidence", "reason"}``; a
    malformed / non-``classify_block`` call is dropped so a noisy reply degrades
    to an empty decision list rather than crashing the caller.
    """
    out: list[dict] = []
    tcs = getattr(msg, "tool_calls", None)
    if not tcs:
        return out
    for tc in tcs:
        fn = getattr(tc, "function", None)
        if getattr(fn, "name", None) != "classify_block":
            continue
        args = getattr(fn, "arguments", None)
        try:
            data = json.loads(args) if args else {}
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        data.setdefault("confidence", 0.5)
        out.append(data)
    return out


def make_classify_tool_fn(
    model: ModelConfig, log: Callable[[str], None] | None = None
) -> Callable[[Sequence[str]], list[dict]] | None:
    """Return a tool-use block-classification callback, or ``None`` if disabled.

    Signature ``(blocks) -> [decision, ...]``: it sends the list of ambiguous
    block texts to ``classify_block`` (OpenAI ``tools``) and parses the returned
    ``tool_calls`` into decisions.  Since the model reasons before emitting a
    tool call, a generous ``max_tokens`` is used.  Best-effort — any failure
    returns ``[]`` so the caller falls back to the rule-based decision.
    """
    if not model.vision:
        return None
    client = OpenAI(**model.client_kwargs())

    def _classify(blocks: Sequence[str]) -> list[dict]:
        items = "\n".join(f"[{i}] {str(b)[:60]!r}" for i, b in enumerate(blocks))
        prompt = (
            "请判断下列每个文本块应保留原文（组织架构/架构图节点标签、报表/科目代码、手写签字）"
            "还是翻译成目标语言，逐一调用 classify_block：\n" + items
        )
        try:
            resp = client.chat.completions.create(
                model=model.model, temperature=0.0, max_tokens=4096,
                tools=[_CLASSIFY_TOOL], tool_choice="auto",
                messages=[{"role": "user", "content": prompt}],
            )
            msg = resp.choices[0].message
            decisions = _parse_classify_tool_calls(msg)
            if not decisions and log:
                log("  分类工具未返回 tool_call：" + str(getattr(msg, "content", "")))
            return decisions
        except Exception as exc:  # noqa: BLE001 — best-effort, fail-closed
            if log:
                log(f"  分类工具调用失败：{type(exc).__name__}: {exc}")
            return []

    return _classify

