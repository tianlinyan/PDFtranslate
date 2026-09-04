"""Central prompt library for the AI-centric PDF Translate app.

This app is fundamentally *prompt-driven*: every AI decision — batched translation,
per-page agent orchestration, special-page negotiation, the persistent chat, tool
use — is governed by a system or user prompt.  All prompt text lives here so it is
easy to review, tune, version and reuse, and so no prompt is inlined deep inside a
translator / agent / chat control path.

Conventions:

* **System prompts are functions** when they depend on runtime data (target
  language, glossary, page index, page kind) so callers pass parameters in.
* **Constant prompts** (chat system prompt, tool hint, greeting) are module-level
  strings or zero-arg functions.
* The translator/agent/chat modules import these **instead of** building prompt
  text inline — see the call sites in ``translator`` / ``agent.flow`` / ``chat``.

Keep everything here in Simplified Chinese where it is shown to the user (page
tasks, negotiation questions, chat text); the English prompt bodies (translation
rules) stay English as they target the model directly.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Translation engine — the batched block-translation system prompt.
# ---------------------------------------------------------------------------

#: Immutable example input lines (kept in the input language) that illustrate the
#: numbered-pair format.  Fixed so the model is not nudged toward the reply
#: language; the *output* is rendered in the target language below.
_EXAMPLE_INPUT = ("Press OK to continue.", "Save the file before exiting.")

#: Example outputs per target language (``lc = language.casefold()``).  The output
#: must be in the target language, so a non-ASCII-script target (e.g. Chinese)
#: declares its own examples while a Latin target reuses an English-style result.
_EXAMPLE_OUTPUTS: dict[str, tuple[str, str]] = {
    "simplified chinese": ("点击“确定”继续。", "退出前请保存文件。"),
    "english": ("Click \u201cOK\u201d to continue.", "Save the file before exiting."),
    "spanish": ("Pulse \u201cAceptar\u201d para continuar.", "Guarde el archivo antes de salir."),
    "french": ("Cliquez sur \u201cOK\u201d pour continuer.", "Enregistrez le fichier avant de quitter."),
    "german": ("Klicken Sie auf \u201cOK\u201d, um fortzufahren.", "Speichern Sie die Datei, bevor Sie beenden."),
    "italian": ("Fare clic su \u201cOK\u201d per continuare.", "Salvare il file prima di uscire."),
}


def translation_system_prompt(language: str, glossary: dict[str, str] | None = None) -> str:
    """The system prompt for translating a numbered batch of blocks into ``language``.

    The name / numbering rules only apply to a Latin-script target (an English
    translation of a Chinese annual report): for a CJK target the source names stay
    as they are and the numbering conventions carry over directly.  ``glossary``
    (source → target) is appended so the model honours the document's own terms.
    """
    latin = not any("一" <= c <= "鿿" for c in language)
    lc = language.strip().casefold()
    is_english = latin and lc == "english"
    prompt = (
        "You are a professional document translator. Translate every numbered "
        f"block below into {language}. Output the whole translation in "
        f"{language} only — never in English or any other language.\n"
        "Rules:\n"
        "- Keep the original meaning, tone and paragraph structure.\n"
        "- Keep the translation similar in length to the source and word it "
        "concisely, so it fits the original document layout.\n"
        "- Keep numbers, units, URLs, codes and product names as in the "
        "source: never reformat thousands separators, decimals or figures. "
        "Keep the original unit but express its name in the target language "
        "— for example translate 万元 as \"ten thousand yuan\" in English, or "
        "\"diez mil yuanes\" in Spanish — so the numeric value itself never "
        "changes.\n"
        "- Keep section numbers and note references in the document's own "
        "numbering style and default to ARABIC digits: Chinese listing "
        "numerals (一、二、三) and Chinese note markers （一）（二）（三十三） render as "
        "1., 2., 3. and (1), (2), (33); 第4条 / 第二节 render as 'Article 4' / "
        "'Section 2'; Arabic digits stay Arabic. Use Roman numerals (I., II., "
        "(III)) only when the source literally uses Roman numerals (Ⅰ. Ⅱ. or I. "
        "II.). Do not renumber or invent a different style.\n"
        "- Keep official statement and report codes as they are: do not "
        "transliterate a code like a statement number; use the standard "
    )
    if is_english:
        prompt += (
            "English name (e.g. \"Consolidated Statement of Cash Flows\") with "
            "its original code.\n"
        )
    else:
        prompt += (
            "name in the target language (never the English name: render "
            "\"Consolidated Statement of Cash Flows\" as \"Estado consolidado "
            "de flujos de efectivo\" in Spanish) with its original code.\n"
        )
    if latin:
        prompt += (
            "- Romanize Chinese personal names with the standard pinyin "
            "spelling, given name first and family name last (e.g. 王晓东 "
            "-> \"Xiaodong Wang\"), and use the same spelling for a person "
            "throughout the document; a personal-name cell must never stay "
            "in Chinese.\n"
        )
    prompt += (
        "- If a block is already entirely in the target language, output it "
        "unchanged.\n"
        "- Preserve numbering exactly: reply as '[n] translated text' per "
        "block, in the same order.\n"
        "- Do not merge or split blocks, and do not add explanations, notes or "
        "any preamble.\n"
        "- Output ONLY the numbered translations, nothing else.\n"
    )
    out1, out2 = _EXAMPLE_OUTPUTS.get(lc, _EXAMPLE_OUTPUTS["simplified chinese"])
    inp1, inp2 = _EXAMPLE_INPUT
    prompt += (
        "Example:\n"
        "Input:\n"
        "[1]\n"
        f"{inp1}\n"
        "[2]\n"
        f"{inp2}\n"
        "Output:\n"
        "[1]\n"
        f"{out1}\n"
        "[2]\n"
        f"{out2}\n\n"
        "Do not write anything except the numbered translations."
    )
    if glossary:
        entries = "\n".join(f"- {src}: {dst}" for src, dst in glossary.items())
        prompt += (
            f"\n\nGlossary: use these translations without change when the "
            f"matching source term appears:\n{entries}"
        )
    return prompt


# ---------------------------------------------------------------------------
# 2. Agent orchestration — interaction rules + per-page / negotiation prompts.
# ---------------------------------------------------------------------------

def agent_interaction_rules() -> str:
    """Rules appended to the agent's system prompt so the model knows *when* to ask
    the user (``ask_user``).  This is the "该问就问" trigger — the model calls the
    ``ask_user`` tool at these decision points and the sidebar answers it.
    """
    return (
        "\n\n【与用户交互】在以下情况，必须先调用 ask_user 获取用户决定，再继续：\n"
        "1) 术语/专有名词/报表科目名翻译不确定时（给出候选译文让用户选）。\n"
        "2) 某块该保留原文还是翻译、判断不确定时（给出『保留/翻译』）。\n"
        "3) 需要删除/覆盖/擦除译文或页（不可逆/破坏性）时（先确认）。\n"
        "4) 接近预算上限、或当前无法推进时（询问是否收尾或继续）。\n"
        "没有上述情况不要打扰用户。ask_user 的结果会作为工具结果返回，请据此继续。"
    )


def page_task(page_index: int, lang: str, kind: str | None = None) -> str:
    """The per-page task handed to ``run_page_visual``.

    ``kind`` (normal / scan / chart / table / uncertain) is appended when given (a
    special page), so the model knows what sort of page it is looking at.
    """
    n = page_index + 1
    head = f"这是文档第 {n} 页" + (f"（{kind} 页）" if kind else "") + "。"
    return (
        f"{head}请阅读页面，把需要翻译的块翻译成 {lang}；数字/代码块保持原样。"
        "用 read_page 观察、用 set_text/translate_block 翻译，最后用 check_residual 校验，"
        "没有可做的事就结束。"
    )


def special_page_question(page_index: int, kind: str) -> tuple[str, list[str]]:
    """The per-kind question + answer options for a special page (M3 negotiation).

    Returns ``(question, options)`` surfaced in the sidebar as buttons; the chosen
    option is mapped to translate / keep / skip by ``DocumentSession._ask_decision``.
    """
    n = page_index + 1
    if kind == "scan":
        return (f"第 {n} 页是扫描件，需 OCR 识别后翻译。如何处理？",
                ["OCR并翻译", "保留原文", "跳过"])
    if kind == "chart":
        return (f"第 {n} 页是组织架构图，节点标签宜保留原文。如何处理？",
                ["保留原文", "翻译", "跳过"])
    if kind == "table":
        return (f"第 {n} 页是报表。如何处理？", ["翻译", "保留原文", "跳过"])
    return (f"第 {n} 页类型不确定。如何处理？", ["翻译", "保留原文", "跳过"])


# ---------------------------------------------------------------------------
# 3. Interaction chat — system prompt + tool hint + greeting.
# ---------------------------------------------------------------------------

def chat_system_prompt() -> str:
    """The base system prompt for the persistent sidebar chat (replies in Chinese)."""
    return (
        "你是 PDF Translate 的 AI 助手，负责协助用户翻译、校对 PDF 文档。"
        "请用中文、简洁直接地回复；涉及翻译、术语、版面布局、数字保真时可给出建议。"
    )


def chat_tool_hint() -> str:
    """Appended to the chat system prompt ONLY when the model is given tools, so it
    knows it can inspect / navigate / edit the loaded PDF.
    """
    return (
        "\n\n【可用工具】当用户要你查看或修改当前 PDF 时，请主动调用工具："
        "get_doc_info 看文档概览，classify_page 判页型，read_page 读某页原文与现译文，"
        "goto_page 打开预览窗口，set_block_text / delete_block_text / apply_annotation "
        "改写或删除译文。只在确实需要文档信息、或用户明确要求改动时调用；纯闲聊不要调用工具。"
    )


#: The message the app sends to the chat on startup so the assistant greets the user.
CHAT_GREETING = "你好"


# ---------------------------------------------------------------------------
# 4. Tool descriptions — every OpenAI tool's ``description`` string is also prompt
# text that steers the model.  The tool modules (``agent/tools.py``,
# ``chat_tools.py``) and ``translator._CLASSIFY_TOOL`` reference these instead of
# authoring their own, so all model-facing text lives in one place.
# ---------------------------------------------------------------------------

#: The translation-agent tool descriptions (``agent/tools.py``), keyed by tool name.
AGENT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_page": "读取指定页的文本块与布局元数据",
    "render_region": "把某页的指定区域渲染成 PNG 供视觉观察",
    "get_layout": "提取某页的行/列聚类、二维网格与单元格跨度",
    "get_doc_info": "返回原文件信息：页数/标题/语言/文本页/扫描页/图表页/表格页/块数",
    "classify_page": "判定某页类型：normal/scan/chart/table/uncertain",
    "translate_block": "把某一块文本翻译成目标语言（走缓存+编号协议）",
    "retranslate_block": "强制重译某块（绕过缓存，常用于残中/空缺修正）",
    "rewrite_block": "改写某块译文的措辞（面向用户预览）",
    "set_text": "把某块文本直接置为指定值（数字/代码块会被拒绝）",
    "apply_annotation": "按预览中用户框选的区域改写/删除对应块（M6）",
    "apply_terminology": "为某源词设定统一的术语译文（后续按术语翻译）",
    "delete_block": "删除译文中某一块（输出侧，自由；原文不受影响）",
    "create_block": "在译文某位置创建一块文本（自由；不修改原文）",
    "move_block": "把译文块移到新位置/重排（自由）",
    "draw_table": "绘制一张干净 N×M 表格（含表头合并）",
    "draw_block": "在指定块的位置绘制其译文",
    "set_font": "设置某块的渲染字号（夹逼到可读下限/上限）",
    "set_align": "设置某块的水平对齐方式",
    "merge_cells": "把表格某格跨列/跨行合并",
    "grid_rule": "画一条表格线/分隔线",
    "cover_region": "用不透明白块覆盖输出页某区域（盖扫描字/杂讯）",
    "erase_text_layer": "删除**输出页**某区域的文本层（保留图片/线条）",
    "drop_element": "移除**输出页**上的某元素（签字区/水印/照片）",
    "classify_block": "判断一个文本块应保留原文（组织架构/架构图节点标签、报表/科目代码、手写签字）"
                     "还是翻译成目标语言",
    "check_residual": "检查某页是否有残留中文/漏译的块",
    "check_missing": "检查某页是否有源有译文空的块（内容缺失）",
    "qa_render": "读回成品页找渲染问题（溢出/越线/压叠/字过小）",
    "audit": "汇总全文档差异报告（数字/术语/编号/残留）",
    "preview_page": "在预览窗口显示指定页面（供用户查看），可聚焦某区域/块",
    "ask_user": "向用户提问并等待回答（关键决策/歧义/术语确认）",
    "delete_page": "删除译文中某一页（输出侧，自由；源页不受影响）",
    "create_page": "在译文某处创建/插入一页（自由）",
    "move_page": "把译文页移动到新位置（重排页序，自由）",
}

#: The interaction-chat tool descriptions (``chat_tools.py``), keyed by tool name.
CHAT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_doc_info": "返回当前 PDF 的信息：页数/标题/语言/文本页/扫描页/图表页/表格页/块数/每页类型。",
    "classify_page": "判定某页类型：normal/scan/chart/table/uncertain。",
    "read_page": "读取某页的原始文本块与当前译文，含扁平块索引（供 set_block_text 使用）与布局元数据。",
    "goto_page": "在预览窗口显示指定页（原文/译文侧）。",
    "set_block_text": "把某块的译文直接置为指定文本（数字/代码块会被拒绝；写的是受保护的译文层）。",
    "delete_block_text": "移除某块的 AI 编辑，恢复为未被覆盖的译文。",
    "apply_annotation": "按预览中用户框选的区域改写/删除对应块（用 read_page 拿到的 bbox）。",
}
