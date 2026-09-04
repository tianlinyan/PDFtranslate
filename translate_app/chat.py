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
from typing import Any, Callable

from openai import OpenAI

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from . import prompts
from .settings import ModelConfig

#: Bound on how many model→tool→result round-trips a single chat turn may run
#: before we give up and return what we have (guards against a loop that never
#: stops calling tools).
_MAX_TOOL_ROUNDS = 8

#: Default output cap for a chat reply (the translation config may raise it).
_CHAT_MAX_TOKENS = 1024


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
        self.client = OpenAI(**model.client_kwargs())
        self.history: list[dict[str, Any]] = []

    def reply(
        self,
        message: str,
        tools: list[dict[str, Any]] | None = None,
        executor: Callable[[str, dict[str, Any]], Any] | None = None,
        image: bytes | None = None,
    ) -> str:
        """Append the user message, ask the interaction model, record and return the reply.

        ``image`` (optional PNG bytes) is attached to the user message as a vision
        input for a multimodal model (ignored for a non-vision model).  When
        ``tools`` / ``executor`` are given, the model may call tools; each tool
        result is fed back (``tool`` role message) and the model is asked again, up to
        :data:`_MAX_TOOL_ROUNDS`, until it returns plain content.
        """
        if image and getattr(self.model, "vision", False):
            content: Any = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": _png_data_url(image)}},
            ]
        else:
            content = message
        self.history.append({"role": "user", "content": content})
        for _ in range(_MAX_TOOL_ROUNDS):
            resp = self._call(tools=tools)
            msg = resp.choices[0].message
            tcs = getattr(msg, "tool_calls", None)
            if not tcs:
                reply = (getattr(msg, "content", None) or "").strip()
                self.history.append({"role": "assistant", "content": reply})
                return reply
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
            for tc in tcs:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:  # noqa: BLE001 — a bad arg JSON is not fatal
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                result = {"error": "该模型未执行工具"} if executor is None else executor(name, args)
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
        # The model kept calling tools past the cap: give it one last chance to
        # answer plainly (no more tool calls) instead of looping forever — but
        # if that call fails too, fall back to the last assistant text.
        try:
            resp = self._call(tools=None)
            content = (getattr(resp.choices[0].message, "content", None) or "").strip()
            if content:
                self.history.append({"role": "assistant", "content": content})
                return content
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

    def _call(self, *, tools: list[dict[str, Any]] | None) -> Any:
        """One chat-completions call using the interaction parameter set."""
        system_prompt = prompts.chat_system_prompt()
        if tools:
            system_prompt += prompts.chat_tool_hint()
        messages = [{"role": "system", "content": system_prompt}] + self.history
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
        return self.client.chat.completions.create(**kwargs)


class ChatWorker(QObject):
    """Run :class:`ChatSession` on a worker thread so the GUI never blocks.

    Callers emit :attr:`ask_requested` (or invoke :meth:`ask` via that signal) from
    the GUI thread; because the worker lives on another ``QThread`` the slot runs
    there and the reply is signalled back to the GUI (queued connection → main thread).
    """

    ask_requested = pyqtSignal(str, object, object)   # text, ModelConfig, image_bytes|None
    reply_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, log: Callable[[str], None] | None = None,
                 ctx: Any | None = None,
                 show_preview: Callable[[int, str], None] | None = None,
                 re_export: Callable[[], None] | None = None,
                 start_translate: Callable[[str], None] | None = None,
                 set_setting: Callable[[str, str], None] | None = None) -> None:
        super().__init__()
        self._log = log or (lambda m: None)
        self._session: ChatSession | None = None
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

        tools_map = make_chat_tools(
            self._ctx, show_preview=self._show_preview, re_export=self._re_export,
            start_translate=self._start_translate, set_setting=self._set_setting,
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
        try:
            session = self._ensure_session(model)
            tools, executor = self._build_tools()
            reply = session.reply(str(text), tools=tools, executor=executor, image=image)
            self.reply_ready.emit(reply)
        except Exception as exc:  # noqa: BLE001 — best-effort, never crash the thread
            self._log(f"  对话请求失败：{type(exc).__name__}: {exc}")
            self.error.emit(f"{type(exc).__name__}: {exc}")
