"""Persistent AI chat: a multi-turn conversation driven by the interaction model.

The sidebar free-text panel talks to the configured model on a background thread
so the GUI never blocks.  It uses the **interaction** parameter set (see
``ModelConfig.interaction_temperature`` / ``interaction_reasoning_effort``) —
the same one the agent orchestrator uses for its ``decide`` calls — not the
deterministic translation temperature.

The interaction model may also call **tools**: when a document context
(:class:`translate_app.doc_context.DocContext`) is available, the worker hands the
session the chat tool set (``chat_tools``) and the session runs a bounded
*model → tool → result → model* loop until the model returns a plain reply.

* :class:`ChatSession` — the (Qt-free) conversation state + LLM call + tool loop.
* :class:`ChatWorker` — a ``QObject`` living on a ``QThread`` that runs ``reply``
  off the GUI thread and emits ``reply_ready`` / ``error`` back.
"""

from __future__ import annotations

import base64
import json
import threading
from types import SimpleNamespace
from typing import Any, Callable

from openai import OpenAI

from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot

from . import prompts
from . import translator as _tr
from .control import ControlSignal
from .settings import ModelConfig

#: Bound on how many model→tool→result round-trips a single chat turn may run
#: before we give up and return what we have (guards against a loop that never
#: stops calling tools).
_MAX_TOOL_ROUNDS = 8

#: Default output cap for a chat reply (the translation config may raise it).
_CHAT_MAX_TOKENS = 1024

#: Shown when the interaction model returns an *entirely empty* reply (no tool call
#: and no text) even after a single gentle nudge — the sidebar must never show a
#: blank bubble.
_EMPTY_REPLY = "（模型未返回内容，请重试或换种说法。）"

#: Max chat messages (incl. the current user turn) handed to the model per call.  The
#: console history grows unbounded across turns, and a small local context (e.g. 50k)
#: overflows once history + image tokens pile up (``exceed_context_size_error``).
#: Window the tail to a recent, protocol-valid budget that always keeps one full
#: tool loop (≤ ``_MAX_TOOL_ROUNDS`` rounds) intact.
_CHAT_HISTORY_CAP = 32

#: A preview "发送" screenshot is a full page at render DPI; a vision model bills
#: large image tokens for it and a small local context can blow past its limit.
#: Downscale the long edge to this many px before attaching / re-injecting.
_CHAT_IMAGE_MAX = 1024


class ChatCancelled(ControlSignal):
    """Raised inside :meth:`ChatSession.reply` when the user cancelled the turn.

    The watchdog can only abort a *blocked request* (it closes the httpx client).
    A cancel that lands while a tool is running left the loop untouched: the next
    round used the refreshed client and the model kept calling tools (measured:
    7 more tools + 8 model calls after "取消"), so the user's edits kept being
    written even though the sidebar said the turn was cancelled.

    A :class:`~translate_app.control.ControlSignal` (like ``TranslationCancelled``
    and ``FlowCancelled``) so no ``except Exception`` may swallow it: the final
    fallback in :meth:`ChatSession.reply` used to catch it and return a stale
    reply instead of propagating the cancel.
    """

#: One-time flag: Pillow is a hard dependency in ``requirements.txt`` but imported
#: lazily, so a missing install must be reported once instead of degrading silently.
_PIL_WARNED = False


def _downscale_png(png: bytes, max_side: int = _CHAT_IMAGE_MAX,
                   log: Callable[[str], None] | None = None) -> bytes:
    """Shrink a PNG to ``max_side`` on the long edge (Pillow), cutting image tokens.

    A preview screenshot can be ~1.9k×2.5k px; a vision model charges a lot of image
    tokens for it, which on a small local context triggers ``exceed_context_size_error``.
    Returns the original bytes on any failure (never crashes the chat turn).

    Pillow is listed in ``requirements.txt`` but imported lazily, so a machine
    installed without it would silently send the full-size screenshot — the exact
    failure this function exists to prevent.  Report that once through ``log``.
    """
    global _PIL_WARNED
    try:
        from io import BytesIO

        from PIL import Image

        im = Image.open(BytesIO(png))
        w, h = im.size
        if w <= max_side and h <= max_side:
            return png
        scale = max_side / max(w, h)
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                       Image.LANCZOS)
        out = BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()
    except ImportError:
        if not _PIL_WARNED:
            _PIL_WARNED = True
            if log:
                log("  警告：未安装 Pillow，聊天图片不会压缩（可能触发上下文超限）；"
                    "请执行 pip install Pillow。")
        return png
    except Exception:  # noqa: BLE001 — a bad image shrinks to the original bytes
        return png


def _png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


class ChatSession:
    """Holds the conversation history and calls the interaction model once per turn.

    A single instance keeps ``history`` across turns so the model has context; the
    worker recreates it whenever the user switches to a different model.
    """

    def __init__(self, model: ModelConfig, log: Callable[[str], None] | None = None) -> None:
        self.model = model
        self.log = log
        self.client = self._new_client()
        self.history: list[dict[str, Any]] = []

    def _new_client(self):
        return OpenAI(**self.model.client_kwargs())

    #: How many messages the console keeps in memory.  ``_CHAT_HISTORY_CAP`` is the
    #: API *window*; a few times that is plenty of context to keep around.
    _HISTORY_KEEP = _CHAT_HISTORY_CAP * 4

    def _prune_history(self) -> None:
        """Bound the in-memory history and drop stale image payloads.

        The history lives for the whole session while only the last
        ``_CHAT_HISTORY_CAP`` messages can ever be sent again, and every preview
        screenshot stays in it as base64 (40 turns with images held ~100 MB of pixels
        that were no longer reachable).  Old messages are dropped, and image parts left
        outside the send window are stripped.
        """
        if len(self.history) > self._HISTORY_KEEP:
            del self.history[:len(self.history) - self._HISTORY_KEEP]
        keep_from = max(0, len(self.history) - _CHAT_HISTORY_CAP)
        for i, msg in enumerate(self.history):
            if i >= keep_from:
                break
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = [
                    part for part in content
                    if not (isinstance(part, dict)
                            and part.get("type") == "image_url")
                ]

    def refresh_client(self) -> None:
        """Close the current client and mint a fresh one.

        Used to abort an in-flight ``chat.completions.create`` (closing the httpx
        client raises on the blocked call) while preserving ``history`` for the next
        turn.
        """
        try:
            self.client.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass
        self.client = self._new_client()

    def reply(
        self,
        message: str,
        tools: list[dict[str, Any]] | None = None,
        executor: Callable[[str, dict[str, Any]], Any] | None = None,
        image: bytes | None = None,
        on_chunk: Callable[[str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> str:
        """Append the user message, ask the interaction model, record and return the reply.

        ``image`` (optional PNG bytes) is attached to the user message as a vision
        input for a multimodal model (ignored for a non-vision model).  When
        ``tools`` / ``executor`` are given, the model may call tools; each tool
        result is fed back (``tool`` role message) and the model is asked again, up to
        :data:`_MAX_TOOL_ROUNDS`, until it returns plain content.

        ``cancel`` is polled before every model call and before every tool call; when
        it returns True the turn raises :class:`ChatCancelled` instead of running the
        remaining rounds (the watchdog alone only aborts a blocked request).  Tool
        calls that will never run are answered with a ``tool`` message first, so the
        recorded history stays valid for the next turn.
        """
        def _check() -> None:
            if cancel is not None and cancel():
                raise ChatCancelled()

        def _answer_pending(rest) -> None:
            """Reply to tool calls that will never execute.

            Every ``tool_call`` of an assistant message must be followed by a
            matching ``tool`` message; leaving one unanswered makes the whole
            history invalid, so the *next* user message would be rejected by the
            server (``400``) — a cancel would break the conversation for good.
            """
            for tc in rest:
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"ok": False, "error": "已取消"},
                                          ensure_ascii=False),
                })

        if image and getattr(self.model, "vision", False):
            content: Any = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": _png_data_url(
                    _downscale_png(image, _CHAT_IMAGE_MAX, log=self.log))}},
            ]
        else:
            content = message
        self.history.append({"role": "user", "content": content})
        empty_rescued = False
        for _ in range(_MAX_TOOL_ROUNDS):
            _check()
            msg = self._call(tools=tools, on_text=on_chunk)
            tcs = getattr(msg, "tool_calls", None)
            if not tcs:
                reply = (getattr(msg, "content", None) or "").strip()
                if reply:
                    self.history.append({"role": "assistant", "content": reply})
                    return reply
                # The model ended this round with neither a tool call nor any text —
                # usually because it treated the last tool result as the answer and
                # "stopped" silently.  Nudge once; if it still stammers blank, fall
                # back to a clear placeholder so the sidebar never shows an empty
                # bubble.
                if not empty_rescued:
                    empty_rescued = True
                    self.history.append({
                        "role": "user",
                        "content": "（你刚才没有返回任何文字。请用一句话总结刚才的检查/修改结果，"
                                   "或告诉我下一步该做什么。）",
                    })
                    continue
                self.history.append({"role": "assistant", "content": _EMPTY_REPLY})
                return _EMPTY_REPLY
            self.history.append({
                "role": "assistant",
                "content": str(getattr(msg, "content", "") or ""),
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tcs
                ],
            })
            pending_image: bytes | None = None
            for pos, tc in enumerate(tcs):
                if cancel is not None and cancel():
                    _answer_pending(tcs[pos:])
                    raise ChatCancelled()
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:  # noqa: BLE001 — a bad arg JSON is not fatal
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                result = {"error": "该模型未执行工具"} if executor is None else executor(name, args)
                shown = result
                img = result.get("image") if isinstance(result, dict) else None
                if isinstance(result, dict):
                    # An image-bearing tool result rides as ``image_url`` (never as
                    # base64 text) — strip it here and re-inject below for the model.
                    shown = {k: v for k, v in result.items()
                             if k != "image" and not isinstance(v, (bytes, bytearray))}
                if img is not None and getattr(self.model, "vision", False) \
                        and isinstance(img, (bytes, bytearray)):
                    pending_image = bytes(img)
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(shown, ensure_ascii=False, default=str),
                })
            if pending_image is not None:
                # Re-inject a tool-returned image as a fresh visual observation so a
                # vision model actually "sees" the rendered/annotated page (mirrors
                # ``make_llm_decide``), rather than getting a base64 blob in text.
                self.history.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "（下面是工具返回的页面图。）"},
                        {"type": "image_url", "image_url": {"url": _png_data_url(
                            _downscale_png(pending_image, _CHAT_IMAGE_MAX, log=self.log))}},
                    ],
                })
        # The model kept calling tools past the cap: give it one last chance to
        # answer plainly (no more tool calls) instead of looping forever — but
        # if that call fails too, fall back to the last assistant text.
        try:
            _check()
            msg = self._call(tools=None, on_text=on_chunk)
            content = (getattr(msg, "content", None) or "").strip()
            if content:
                self.history.append({"role": "assistant", "content": content})
                return content
        except ControlSignal:
            # ``_check()`` above raised ChatCancelled: a control signal, not a
            # failure to degrade from — it must reach the worker as "已取消".
            raise
        except Exception:  # noqa: BLE001 — degrade to the last assistant text
            pass
        last = next(
            (m["content"] for m in reversed(self.history)
             if m.get("role") == "assistant" and m.get("content")),
            "",
        )
        reply = str(last) or "（工具调用次数过多，已停止。）"
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def record_exchange(self, question: str, answer: str, target: str = "") -> None:
        """(b) Note a flow-time agent question + the user's answer into ``history``.

        The flow asks the user through ``AnswerBridge`` (a separate channel from the
        chat); recording the exchange here lets the console's next turn see it and
        keep one coherent thread with the flow, instead of two disconnected surfaces.
        Empty answer (e.g. a cancelled ask) is skipped.
        """
        q = str(question or "").strip()
        a = str(answer or "").strip()
        if not q or not a:
            return
        prefix = f"（流程询问 {target}）" if target else "（流程询问）"
        self.history.append({"role": "assistant", "content": f"{prefix}{q}"})
        self.history.append({"role": "user", "content": a})

    @staticmethod
    def _tool_group_is_complete(msgs: list[dict[str, Any]], i: int) -> bool:
        """True when ``msgs[i]`` is an ``assistant(tool_calls)`` with ALL its replies.

        A window may only start at a message that does not depend on anything before
        it: a ``user`` turn, or a complete tool group.  Starting at the assistant of
        an incomplete group (or at a bare ``tool`` message) makes the OpenAI API
        reject the whole request (each ``tool`` needs its ``assistant`` and every
        ``tool_call_id`` needs exactly one reply).
        """
        calls = msgs[i].get("tool_calls") or []
        if not calls:
            return False
        want = {str(tc.get("id")) for tc in calls}
        got: set[str] = set()
        for m in msgs[i + 1:]:
            if m.get("role") != "tool":
                break
            got.add(str(m.get("tool_call_id")))
        return want <= got

    def _window_history(self) -> list[dict[str, Any]]:
        """A bounded, protocol-valid recent window of ``self.history``.

        The console history grows unbounded across turns; a small local context
        overflows (``exceed_context_size_error``) once history + image tokens pile up.
        Keep the last ``_CHAT_HISTORY_CAP`` messages, then trim to a boundary the
        chat API accepts: a ``user`` turn, or a *complete* tool group
        (``assistant(tool_calls)`` + all of its ``tool`` replies).

        The old "trim to the first ``user``, else fall back to the raw tail" rule
        could hand the model a window that STARTS with a ``tool`` message: one turn
        with more tool rounds than the cap leaves no ``user`` in the tail, so the
        raw fallback began mid-group and the endpoint answered 400 — losing the
        whole turn's work.
        """
        msgs = self.history
        if len(msgs) <= _CHAT_HISTORY_CAP:
            return list(msgs)
        tail = msgs[-_CHAT_HISTORY_CAP:]
        for i, m in enumerate(tail):
            role = m.get("role")
            if role == "user":
                return list(tail[i:])
            if role == "assistant" and self._tool_group_is_complete(tail, i):
                return list(tail[i:])
        # No boundary inside the cap (a single turn with more tool rounds than the
        # cap allows): keep the last user turn whole — correctness beats the bound.
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                return list(msgs[i:])
        return []

    def _call(self, *, tools: list[dict[str, Any]] | None,
              on_text: Callable[[str], None] | None = None) -> Any:
        """One chat-completions call using the interaction parameter set.

        When ``on_text`` is given, the request is streamed and each content delta is
        forwarded to it (so the sidebar types out the reply live); ``tool_calls`` deltas
        are accumulated into a reconstructed message.  With ``on_text=None`` the call is
        non-streaming and returns the response message directly (used by tests).
        """
        system_prompt = prompts.chat_system_prompt()
        if tools:
            system_prompt += prompts.chat_tool_hint()
        self._prune_history()
        messages = [{"role": "system", "content": system_prompt}] + self._window_history()
        kwargs: dict[str, Any] = {
            "model": self.model.model,
            "temperature": float(getattr(self.model, "interaction_temperature", 0.6)),
            "max_tokens": (
                self.model.max_tokens if self.model.max_tokens is not None else _CHAT_MAX_TOKENS
            ),
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        # ``reasoning_effort`` is a model-specific body extra the SDK can't take as a
        # first-class kwarg; send it via ``extra_body`` (defaults to "medium").
        body = self.model.interaction_request_params()
        if body:
            kwargs["extra_body"] = body
        if on_text is not None:
            kwargs["stream"] = True
        resp = self.client.chat.completions.create(**kwargs)
        if on_text is None:
            return resp.choices[0].message
        # Streaming: iterate chunks, emit content deltas live, accumulate tool calls.
        # A test/mock client may return a single non-iterable response — read it
        # directly (streaming only real clients).
        try:
            stream_iter = iter(resp)
        except TypeError:
            _msg = resp.choices[0].message
            _txt = getattr(_msg, "content", None)
            if _txt and on_text:
                on_text(str(_txt))
            return _msg
        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        for chunk in stream_iter:
            for ch in getattr(chunk, "choices", None) or []:
                delta = getattr(ch, "delta", None)
                if delta is None:
                    continue
                part = getattr(delta, "content", None)
                if part:
                    content_parts.append(part)
                    on_text(part)
                for tc in getattr(delta, "tool_calls", None) or []:
                    idx = int(getattr(tc, "index", 0))
                    entry = tool_calls.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if getattr(tc, "id", None):
                        entry["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            entry["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            entry["args"] += fn.arguments
        content = "".join(content_parts) or None
        calls = [
            SimpleNamespace(id=e["id"], type="function",
                            function=SimpleNamespace(name=e["name"], arguments=e["args"]))
            for _, e in sorted(tool_calls.items())
        ]
        return SimpleNamespace(content=content, tool_calls=calls or None)


class ChatWorker(QObject):
    """Run :class:`ChatSession` on a worker thread so the GUI never blocks.

    Callers emit :attr:`ask_requested` (or invoke :meth:`ask` via that signal) from
    the GUI thread; because the worker lives on another ``QThread`` the slot runs
    there and the reply is signalled back to the GUI (queued connection → main thread).
    """

    ask_requested = pyqtSignal(str, object, object)   # text, ModelConfig, image_bytes|None
    reply_ready = pyqtSignal(str)
    reply_chunk = pyqtSignal(str)   # a streamed text chunk of the in-progress reply
    error = pyqtSignal(str)
    cancelled = pyqtSignal(str)   # the in-flight reply was aborted by the user ("取消")
    #: (b) A flow-time agent Q&A, noted into the live session's history (queued).
    record_exchange_requested = pyqtSignal(str, str, str)   # question, answer, target

    def __init__(self, log: Callable[[str], None] | None = None,
                 ctx: Any | None = None,
                 show_preview: Callable[[int, str], None] | None = None,
                 re_export: Callable[[], None] | None = None,
                 start_translate: Callable[[str], None] | None = None,
                 set_setting: Callable[[str, str], None] | None = None) -> None:
        super().__init__()
        self._log = log or (lambda m: None)
        self._session: ChatSession | None = None
        #: Aborts the in-flight reply (the sidebar's "取消").  Cleared on each ask.
        self._cancel_ev = threading.Event()
        #: Optional persistent document context (``DocContext``) whose tools the
        #: interaction model may call.  ``show_preview`` is a thread-safe channel to
        #: open a preview page; ``re_export`` re-exports the last translation with the
        #: current edits; ``start_translate`` / ``set_setting`` drive the translate
        #: entry (start the pipeline / change a setting).  The GUI wires these to
        #: bridges so they run on the GUI thread.
        self._ctx = ctx
        self._show_preview = show_preview
        self._re_export = re_export
        self._start_translate = start_translate
        self._set_setting = set_setting

    def _emit_chunk(self, chunk: str) -> None:
        """Forward a streamed text chunk to the GUI (queued) for the live bubble."""
        c = str(chunk or "")
        if c.strip():
            self.reply_chunk.emit(c)

    def _ensure_session(self, model: ModelConfig) -> ChatSession:
        if self._session is None or self._session.model is not model:
            self._session = ChatSession(model, self._log)
        return self._session

    def _build_tools(self):
        """Return ``(openai_tools_schema, executor)`` for the current document, or ``(None, None)``.

        No source document → no tools (the chat is plain conversation).  The tools
        change whenever the source path changes because the executor binds to the
        live :class:`DocContext`.  The executor is ``(name, args) -> result``: a tool
        that raises (or is unknown) fails closed to ``{"error": ...}`` so it never
        crashes the chat turn.
        """
        if self._ctx is None or not self._ctx.has_source():
            return None, None
        from .chat_tools import chat_openai_tools, make_chat_tools
        from .agent import user_flows as _uf

        # AI slot-filling for ``run_flow``: reuse the chat session's own client so a
        # per-turn flow compilation does not open a second connection pool.  ``None``
        # (no model client) simply falls back to the deterministic rule parser.
        llm = None
        plan_llm = None
        if self._session is not None:
            llm = _uf.make_llm_flow_compiler(self._session.model,
                                             client=self._session.client, log=self._log)
            # Path B: the AI decomposes a requirement into an ordered task plan.  ``None``
            # (no client) falls back to a single deterministic rule-derived task.
            plan_llm = _uf.make_llm_plan_compiler(self._session.model,
                                                  client=self._session.client, log=self._log)

        # Translation-side batch re-translation for ``retranslate`` / an ``auto_fix``
        # user flow.  Reuses the chat session's client, and does NOT require vision (it
        # re-translates text blocks — see translator.make_retranslate_batch_fn).  ``None``
        # when no session exists yet → those tools report "重译通道未接线".
        translate_texts = None
        if self._session is not None:
            translate_texts = _tr.make_retranslate_batch_fn(
                self._session.model, client=self._session.client, log=self._log,
                require_vision=False,
            )

        tools_map = make_chat_tools(
            self._ctx, show_preview=self._show_preview, re_export=self._re_export,
            start_translate=self._start_translate, set_setting=self._set_setting, llm=llm,
            plan_llm=plan_llm, log=self._log, translate_texts=translate_texts,
        )

        def executor(name: str, args: dict[str, Any]) -> Any:
            fn = tools_map.get(name)
            if fn is None:
                return {"error": f"未知工具: {name}"}
            try:
                return fn(**args)
            except Exception as exc:  # noqa: BLE001 — fail-closed
                return {"error": f"{type(exc).__name__}: {exc}"}

        return chat_openai_tools(list(tools_map)), executor

    @pyqtSlot(str, object, object)
    def ask(self, text: str, model: ModelConfig | None, image: bytes | None = None) -> None:
        if model is None:
            self.error.emit("没有可用的 AI 模型，无法对话。")
            return
        self._cancel_ev.clear()
        try:
            session = self._ensure_session(model)
            tools, executor = self._build_tools()
            # Abort an in-flight reply on "取消": a watchdog closes the client (which
            # raises on the blocked create) while preserving history for the next turn.
            done = threading.Event()

            def _watchdog():
                while not done.wait(0.1):
                    if self._cancel_ev.is_set():
                        session.refresh_client()
                        return

            threading.Thread(target=_watchdog, daemon=True).start()
            try:
                reply = session.reply(str(text), tools=tools, executor=executor,
                                      image=image, on_chunk=self._emit_chunk,
                                      cancel=self._cancel_ev.is_set)
            finally:
                done.set()
            if self._cancel_ev.is_set():
                self.cancelled.emit("已取消")
                return
            self.reply_ready.emit(reply)
        except ChatCancelled:
            # A cancel that landed while a tool was running (the watchdog only
            # aborts a blocked request).
            self.cancelled.emit("已取消")
        except Exception as exc:  # noqa: BLE001 — best-effort, never crash the thread
            if self._cancel_ev.is_set():
                self.cancelled.emit("已取消")
            else:
                self._log(f"  对话请求失败：{type(exc).__name__}: {exc}")
                # The GUI resets its busy state / live bubble from ``error``; without
                # this the sidebar stayed on "取消" forever and showed nothing at all
                # (the only trace was a main-window log line).
                self.error.emit(f"{type(exc).__name__}: {exc}")

    def cancel_current(self) -> None:
        """Abort the in-flight reply (triggered by the sidebar's "取消" / Enter)."""
        self._cancel_ev.set()

    @pyqtSlot(str, str, str)
    def _record_exchange(self, question: str, answer: str, target: str = "") -> None:
        """(b) Note a flow-time Q&A into the live session's history (queued)."""
        if self._session is None:
            return
        try:
            self._session.record_exchange(question, answer, target)
        except Exception as exc:  # noqa: BLE001 — never crash the worker on a log-only path
            self.error.emit(f"{type(exc).__name__}: {exc}")


def connect_sidebar_cancel(signal, worker: ChatWorker) -> None:
    """Wire the sidebar's "取消" to :meth:`ChatWorker.cancel_current` **directly**.

    ``ChatWorker`` lives on its own ``QThread``, so the default (Auto) connection
    queues the call into that thread's event loop — which is blocked inside
    :meth:`ChatWorker.ask` for the whole reply, so the queued slot only runs after
    the reply has already finished and "取消" does nothing.  The slot only sets a
    ``threading.Event`` (thread-safe, touches no Qt state), so invoking it straight
    from the GUI thread is correct and is the only wiring that actually aborts an
    in-flight reply.
    """
    signal.connect(worker.cancel_current, Qt.ConnectionType.DirectConnection)
