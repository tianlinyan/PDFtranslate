"""Persistent document context shared by the interaction-chat AI tools.

The free-text chat AI (the "交互侧") needs a document to act on: it answers
questions about the PDF and edits the translation.  A single translation run only
lives for the duration of the pipeline, so the chat needs a *persistent* context
that survives across runs.

:class:`DocContext` holds:

* ``src_path`` — the current source PDF (set by ``MainWindow`` when the user picks
  a file);
* ``src_doc`` — the lazily-extracted, read-only :class:`pdfio.DocumentText`
  (constants never change after extraction);
* ``overlay`` — the mutable, **protected** translation overlay: a mapping of flat
  block index → ``{"text": ...}``.  This is where the chat AI writes its edits
  (and where ``apply_annotation`` records a user-drawn edit).  The pipeline applies
  the overlay *on top of* whatever the run produced, so a chat edit always wins.

Thread-safety: ``DocContext`` is accessed from the GUI thread (``MainWindow``)
and from the chat-worker thread simultaneously.  Every public accessor takes an
:class:`threading.RLock`; the read-only document is extracted **outside** the lock
(so a slow OCR extraction never blocks ``set_source`` on the GUI thread) and only
cached back under the lock if the source did not change in the meantime.  (The
overlay is a plain ``dict``; individual item mutations are atomic under the GIL,
and the lock keeps logical invariants consistent.)
"""

from __future__ import annotations

import threading
from typing import Any, Callable


class DocContext:
    """Persistent source + protected translation overlay for the chat AI tools."""

    def __init__(self, log: Callable[[str], None] | None = None) -> None:
        self._lock = threading.RLock()
        self._log = log or (lambda m: None)
        self.src_path: str | None = None
        self.lang: str = ""
        self.ocr: bool = False
        self._src_doc: Any = None
        self._overlay: dict[int, dict[str, Any]] = {}
        self._loaded_path: str | None = None
        #: The last run's full aligned translation (flat block index → text).  Stored
        #: by ``MainWindow`` after a successful run so the chat's ``read_page`` can
        #: show the *real* current translation (not just the chat's protected edits),
        #: which stops the AI from re-editing blocks that are already translated.
        self._last_translated: list[str] | None = None

    # -- source ---------------------------------------------------------------
    def set_source(self, src_path: str | None, *, lang: str = "", ocr: bool = False) -> None:
        """Point the context at a new source PDF (``None`` clears it).

        Changing the path invalidates the cached document and resets the overlay —
        a new file has no inherited edits.  ``lang`` / ``ocr`` update the context used
        for lazy extraction and for the ``target`` the run will produce later.
        """
        with self._lock:
            if lang:
                self.lang = lang
            self.ocr = ocr
            self.src_path = src_path
            if self._loaded_path != src_path:
                self._src_doc = None
                self._overlay = {}
                self._last_translated = None
                self._loaded_path = src_path

    # -- last run's translation (so the chat reads the real current output) ----
    def set_last_translated(self, translated: list[str] | None) -> None:
        with self._lock:
            self._last_translated = list(translated) if translated else None

    def get_last_translated(self) -> list[str] | None:
        with self._lock:
            return list(self._last_translated) if self._last_translated is not None else None

    # -- read-only document ---------------------------------------------------
    def ensure_doc(self) -> Any:
        """Return the extracted source document (``None`` if no source / extraction failed).

        The document is extracted lazily on first use and cached.  Extraction may be
        slow on an OCR document, so it runs on the calling (chat-worker) thread and
        logs a hint.  The extraction is performed **outside** the state lock so a
        concurrent ``set_source`` (GUI thread) never blocks on it — only the cheap
        cache read/write takes the lock, and the result is cached only if the source
        did not change in the meantime.
        """
        with self._lock:
            if self.src_path is None:
                return None
            if self._src_doc is not None:
                return self._src_doc
            src_path = self.src_path
            ocr = self.ocr
        self._log("  首次需要文档内容，正在提取（含 OCR 时可能较慢）…")
        from . import pdfio

        try:
            new_doc = pdfio.extract_document_text(
                src_path, ocr=ocr, log=self._log, cancel=lambda: False)
        except Exception as exc:  # noqa: BLE001 — report and degrade
            self._log(f"  文档提取失败：{type(exc).__name__}: {exc}")
            new_doc = None
        with self._lock:
            if self.src_path == src_path:
                self._src_doc = new_doc
        return new_doc

    def has_source(self) -> bool:
        with self._lock:
            return bool(self.src_path)

    # -- protected overlay ----------------------------------------------------
    def overlay(self) -> dict[int, dict[str, Any]]:
        """A shallow copy of the protected overlay (flat index → entry)."""
        with self._lock:
            return dict(self._overlay)

    def get_overlay(self, index: int) -> dict[str, Any] | None:
        with self._lock:
            return self._overlay.get(index)

    def set_overlay(self, index: int, text: str | None, *, action: str = "set") -> bool:
        """Write a protected edit for ``index``.

        ``action`` is ``"set"`` (store ``text``) or ``"delete"``/``"void"`` (remove the
        entry).  A ``None``/blank ``text`` also removes the entry, so a chat can "undo"
        an earlier edit.  Returns ``True`` on success.
        """
        with self._lock:
            if action in ("delete", "void") or text is None or not str(text).strip():
                self._overlay.pop(int(index), None)
                return True
            self._overlay[int(index)] = {"text": str(text)}
            return True
