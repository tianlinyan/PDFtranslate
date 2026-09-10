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
    "russian": ("Нажмите \u201cOK\u201d, чтобы продолжить.", "Сохраните файл перед выходом."),
    "japanese": ("「OK」を押して続行します。", "終了する前にファイルを保存してください。"),
    "korean": ("“확인”을 누르세요.", "종료하기 전에 파일을 저장하세요."),
}


def _is_cjk_language(language: str) -> bool:
    """True when the target is a CJK-script language.

    The model-facing language name may be ASCII ("Simplified Chinese", "Japanese"),
    so script detection must be by language *identity*, not by scanning the ASCII
    name for CJK glyphs.  Otherwise the default Chinese target ("Simplified Chinese")
    would be treated as Latin-only and wrongly get the Chinese-name-romanization
    rule prepended to a Chinese translation.
    """
    lc = language.strip().casefold()
    return any(s in lc for s in (
        "simplified chinese", "traditional chinese", "chinese", "japanese",
        "korean", "zh", "ja", "ko", "中文", "简体中文", "繁体中文", "日本語", "한국어", "漢語",
    ))


def translation_system_prompt(language: str, glossary: dict[str, str] | None = None) -> str:
    """The system prompt for translating a numbered batch of blocks into ``language``.

    The name / numbering rules only apply to a Latin-script target (an English
    translation of a Chinese annual report): for a CJK target the source names stay
    as they are and the numbering conventions carry over directly.  ``glossary``
    (source → target) is appended so the model honours the document's own terms.
    """
    cjk_target = _is_cjk_language(language)
    latin = not cjk_target
    lc = language.strip().casefold()
    is_english = latin and lc == "english"
    prompt = (
        "You are a professional document translator. Translate every numbered "
        f"block below into {language}. Output the whole translation in "
        f"{language} only — never in any other language.\n"
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
        "- Preserve numbering exactly: reply as '[[n]] translated text' per "
        "block, in the same order.  Use the double-bracket [[n]] marker, never a "
        "single [n] (a single bracket could collide with a citation/reference "
        "like '[1]' in the document).\n"
        "- Do not merge or split blocks; output ONLY the numbered translations, "
        "with no explanations, notes or preamble.\n"
        "- If the user message starts with a 【上下文参考】 section, those lines are "
        "short fragments of the source text ADJACENT to the numbered blocks, "
        "provided only so you can render pronouns, subjects and terminology "
        "consistently across batch seams: never translate, echo or quote them; "
        "output only the numbered [[n]] blocks.\n"
    )
    # Fall back to a target-script-appropriate example: a CJK target uses a Chinese
    # result (so the model is anchored to Chinese), a Latin target uses an
    # English-style one (never a Chinese result for a Spanish/French/… target).
    fallback = "simplified chinese" if cjk_target else "english"
    out1, out2 = _EXAMPLE_OUTPUTS.get(lc, _EXAMPLE_OUTPUTS[fallback])
    inp1, inp2 = _EXAMPLE_INPUT
    prompt += (
        "Example:\n"
        "Input:\n"
        "[[1]]\n"
        f"{inp1}\n"
        "[[2]]\n"
        f"{inp2}\n"
        "Output:\n"
        "[[1]]\n"
        f"{out1}\n"
        "[[2]]\n"
        f"{out2}\n"
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
    """Rules appended to the agent's system prompt: the ONLY times to ask the user.

    The model calls ``ask_user`` at these decision points and the sidebar answers it;
    outside these it must not interrupt the user.
    """
    return (
        "\n\n【与用户交互】仅在这 4 种情况先调 ask_user 拿用户决定，再继续：\n"
        "① 术语/专有名/报表科目名不确定时（给出候选）。"
        "② 某块保留原文还是翻译、判断不确定时（『保留原文还是翻译？』）。"
        "③ 要删除/覆盖/擦除译文或整页（不可逆，先确认）。"
        "④ 接近预算上限、或当前无法推进时（询问是否收尾/继续）。\n"
        "其余情况不要打扰用户。**「开始翻译/重新翻译」本身不需要任何确认**——用户说了就直接开工，"
        "不要问“确认开始吗/要重新翻译还是重新导出/要不要覆盖上次结果”。**提问必须是清晰的自然语言问题，不要依赖按钮**；"
        "ask_user 的 options 只是可选的提示文字，用户会用一句话自由回答，"
        "你要从这句话里理解其意图并据此继续（拿到的是用户的自由文本回答）。"
    )


def agent_tool_policy() -> str:
    """The per-tool reference (function + usage) appended to the agent system prompt.

    The ``tools`` array carries each tool's schema; this spells out the HOW (index
    semantics, what it returns, gotchas) grouped by category, so the model picks the
    right tool without guessing.
    """
    return (
        "\n\n【工具功能与用法】先观察、再修改、最后校验；读过的页别重读；改完必须校验；"
        "某工具返回 ok=false 先读 error 判断原因、别盲目重试；index 一律用 read_page 的**扁平 index**（写回就靠它）。"
        "**页码规则（必须一致）**：工具参数里的 `page` 是 **0 起**（第 1 页 = page 0）；"
        "但你在任务/回复中向用户提到页码时**一律说「第 N+1 页」**（page 0 → 第 1 页），**绝不说“第 0 页”**。\n"
        "各工具的返回值见其 schema，这里只补关键差异：\n"
        "① 观察（只读原文，绝不改写）：read_page（含扁平 index，超大页可 offset/limit 分页；块可带 kind/level——"
        "formula/figure 是结构内容、应保留原文，caption/heading 照常翻译并按 level 处理）、get_layout、get_doc_info、"
        "classify_page（结构存在时会给出 formula/figure 页分型）、render_page（渲染译文 PNG 供自检）、"
        "preview_page（弹窗给用户看）、"
        "detect_page_skew（检测扫描页文本倾斜角，供几何校正决定——recommended 时会问用户）。\n"
        "② 修改（写可编辑译文，原文不可动）：translate_blocks（**批量**翻译并写入，整行/整表/整页首选——远快于逐块请求）、"
        "translate_block（翻译并**写入**，单块，仅翻少数块时用）、set_text（写已知译名；数字格被拒）、"
        "retranslate_blocks（**批量重译并写入**，绕过缓存，修正多条 finding 时首选）、"
        "retranslate_block（**只返回译文**，需再 set_text 写入，单块）、apply_terminology（锁术语）、delete_block（撤为原文）、"
        "apply_annotation（按用户框选改）。\n"
        "③ 校验（只报告，无副作用）：check_residual（残留+空块；目标为西文看残留中文、中文目标看未译英文成句；"
        "纯代码/缩写/单位/数字不算）、check_missing（源有译文空；纯数字格/公式块不算）、check_numbers（数字按值一致；"
        "千分位/小数/单位拆分格式差异与【序数→数字】如 二→2 不算）、check_table（表格格+文本块完整性）、"
        "check_layout（仿导出器量版面：低于可读下限、溢出自身框、压入**同列**下一块——两栏只比同栏）。\n"
        "④ 交互：ask_user(question, options=None, target)——术语/歧义/保留还是翻译/破坏性操作等拿不准时先问；没有这些情况别打扰。"
    )


def agent_workflow() -> str:
    """The general translation METHOD appended to the agent system prompt.

    This is the order/decision flow — BUILD, EDIT/MODIFY, VERIFY — per page.  Tool
    names/params live in ``agent_tool_policy``; here we only chain them and state the
    hard requirements, so nothing is duplicated.
    """
    return (
        "\n\n【翻译通用工作法】整篇逐页推进；每页固定顺序：观察 → 构建/编辑 → 校验 → 复检 → 完成。"
        "工具名与参数见【工具功能与用法】，这里只讲顺序与硬性要求。\n"
        "① 构建（第 1 遍）：get_doc_info / classify_page 认清整篇与页型；read_page 取该页全部块并记下扁平 index；"
        "术语先 apply_terminology 锁定；**优先用 translate_blocks(page) 一次批量翻译该页所有应翻译块**"
        "（一次请求、自动分批+并发，远快于逐块）；确需单独改个别块时才用 translate_block；纯数字/公式/单位块保持原样、不调用翻译工具。\n"
        "② 编辑/修改（复核或修错）：read_page + render_page 看当前译文与版面；按 "
        "check_residual / check_missing / check_numbers / check_table / check_layout 的问题清单决定改哪里——"
        "确知译名 set_text；需重译 translate_block 或 retranslate_block（后者只返回译文）；按框选 apply_annotation；"
        "恢复原文 delete_block。\n"
        "③ 校验：每改一处就复跑对应 check_*，直到 check_residual / check_missing 无问题、且（本页适用时）"
        "check_numbers / check_table / check_layout 无实质问题，才可结束。扫描表报的「字号偏小/行带受限」是物理极限，可留意但不必强修。\n"
        "④ 收尾：整页确认满足才结束；拿不准/术语/保留还是翻译→ask_user；某工具 ok=false→先读 error 判断原因、别盲目重试。\n"
        "**渲染只看一次**：同一页的 render_page 仅在必要时调用一次（拿到版面/当前译文即可），"
        "不要反复重渲染——每次都会把整页图片重新注入对话、拖慢每一步；看不清某处时优先用 read_page 的 "
        "offset/limit 或放大局部，而不是重渲染整页。"
    )


def page_task(page_index: int, lang: str, kind: str | None = None) -> str:
    """The per-page task handed to ``run_page_visual``.

    ``kind`` (normal / scan / chart / table / uncertain) is appended when given (a
    special page), so the model knows what sort of page it is looking at.  The task
    is deliberately strict about the target language: every text block of the page
    must end up in ``lang`` (only numeric / amount / code / unit blocks stay verbatim),
    and the model may not stop while ``check_residual`` still reports untranslated
    content.  This avoids "翻译不完全" where the model skips blocks it judged not to
    need translating.

    **``kind == "chart"`` is the one deliberate exception**: an org chart /
    architecture diagram keeps its node labels as the source (product decision) —
    redrawing those narrow boxes into the target language overlaps and shrinks
    them.  The model is told to translate only the text *outside* the diagram, and
    that it may still translate diagram blocks when the user explicitly asked for
    it (the export honours a translated block by drawing it).
    """
    n = page_index + 1
    head = f"这是文档第 {n} 页" + (f"（{kind} 页）" if kind else "") + "。"
    if kind == "chart":
        return (
            f"{head}这是**组织结构图 / 架构图**：图内节点标签默认**保留原文**"
            "（这些窄高框把译文重画进去只会压字、缩字号），**不要**翻译图内的块。\n"
            f"只把图**外**的文本（大标题、图注、页眉页脚等）译成 {lang}。\n"
            "① read_page(page) 读本页并记下各块 index；② 图外的可译块用 "
            "translate_blocks(page) 或 set_text 处理；③ 若本轮用户要求里**明确**说要翻译"
            "这张图，才对这些块用 translate_block / translate_blocks——翻译过的块会覆盖"
            "默认、正常绘出；④ check_residual 若把图内保留的中文报成残留，**不要**为了"
            f"消残留去翻译图表；确认图外文本已全部译成 {lang} 即可结束。"
        )
    return (
        f"{head}请把本页**所有文本块**均翻译成 {lang}——整页最终应**全部是 {lang}**；"
        "只有纯数字/金额/公式/单位块保持原样。\n"
        "① read_page(page) 读本页全部块并记下 index；② **优先 translate_blocks(page) 一次批量翻译整页所有可翻译块**"
        "（它自动按字符预算分批+并发，且跳过数字格/公式块，远快于逐块请求）；确知的译名可直接 set_text(page, index, text)；"
        "个别需单独重译的块才用 translate_block(index)；③ 最后**必须** check_residual 校验。"
        f"若仍报有块未译/残留，必须继续翻译——**确认整页都已译成 {lang}、且无残留才结束，不得提前结束**。"
    )


def special_page_question(page_index: int, kind: str) -> tuple[str, list[str]]:
    """The per-kind *natural-language* question for a special page (M3 negotiation).

    Returns ``(question, options)``; the question is an open, plain-language prompt and
    ``options`` is only a hint list (never rendered as buttons) so the free-text answer
    is interpreted by ``DocumentSession._interpret_answer`` (an AI read, else a keyword
    matcher) into translate / keep / skip.
    """
    n = page_index + 1
    if kind == "scan":
        return (f"第 {n} 页是扫描件，需要先识别文字再翻译。请问你希望怎么处理这一页？"
                f"请用一句话告诉我（例如：翻译它 / 保留原文 / 跳过这页）。",
                ["OCR并翻译", "保留原文", "跳过"])
    if kind == "chart":
        return (f"第 {n} 页是组织架构图，节点标签通常保留原文。请问要怎么处理？"
                f"请用一句话告诉我（例如：保留原文 / 翻译 / 跳过）。",
                ["保留原文", "翻译", "跳过"])
    if kind == "formula":
        return (f"第 {n} 页含数学公式。公式是数学内容、不应改写，只有公式说明/表注需要翻译。"
                f"请问怎么处理？请用一句话告诉我（例如：翻译说明、保留公式 / 整页都翻 / 跳过）。",
                ["保留公式并翻译说明", "整页翻译", "跳过"])
    if kind == "figure":
        return (f"第 {n} 页是图表。图内文字是结构/图注，通常只翻译图注、保留图形与图内文字。"
                f"请问怎么处理？请用一句话告诉我（例如：翻译图注 / 保留原文 / 跳过）。",
                ["翻译图注", "保留原文", "跳过"])
    return (f"第 {n} 页类型不确定。请问要怎么处理？请用一句话告诉我"
            f"（例如：翻译 / 保留原文 / 跳过）。",
            ["翻译", "保留原文", "跳过"])


def review_mode_question() -> tuple[str, list[str]]:
    """M4: ask (in natural language) whether the draft goes to AI self-check or user review."""
    return ("全文初稿已生成。请问由我来自动自检，还是你自己手动检查？"
            "请用一句话告诉我（例如：你自检 / 我手动检查）。",
            ["AI 自检", "我手动检查"])


def review_export_question() -> tuple[str, list[str]]:
    """M4: after the review, confirm (in natural language) whether to export."""
    return ("复核进行中。要直接导出成品吗？请用一句话告诉我（例如：导出 / 先别导，我再看看）。",
            ["导出", "继续检查"])


def review_page_task(page_index: int, *, findings: dict | None = None,
                     auto_fix: bool = True) -> str:
    """M4 AI_SELFCHECK: the per-page task that fixes audit findings in place.

    ``findings`` is the JSON structure from ``audit_page`` (``{"issues": [...],
    "clean": bool, ...}``) — a **deterministic** audit run *before* this step.  It is
    injected as concrete data so the model fixes exactly the reported problems
    instead of re-running the four checks itself (which used to be the source of
    "false green": a model that forgot one check area silently passed the page).
    ``auto_fix=False`` makes this a read-only re-check (report only, no edits).

    The agent loads the page (``read_page`` + ``render_page``), fixes each finding
    (``set_text`` / ``retranslate_block`` / ``apply_annotation``) and may only finish
    once the audit is clean.  Numeric / code blocks stay verbatim — they are
    protected, not "missing".
    """
    n = page_index + 1
    lines = [
        f"这是文档第 {n} 页（页号 {page_index}，从 0 起）的【复核】。"
        f"先 read_page({page_index}) + render_page({page_index}) 拿到本页原文与译文，再逐项核对、就地修正。",
    ]
    if findings:
        issue_lines: list[str] = []
        for iss in findings.get("issues", []):
            check = str(iss.get("check", ""))
            idx = iss.get("index", "?")
            if check == "layout":
                issue_lines.append(
                    f"- layout/{iss.get('kind', '')} 块#{idx}：{iss.get('detail', '')}")
            elif check == "numbers":
                issue_lines.append(
                    f"- numbers 块#{idx}：源 {iss.get('source', '')!r} → 译文 "
                    f"{iss.get('translation', '')!r}（missing={iss.get('missing')}, "
                    f"extra={iss.get('extra')}）")
            elif check == "residual":
                issue_lines.append(
                    f"- residual 块#{idx}（{iss.get('reason', '')}）：『{iss.get('text', '')}』")
            elif check == "missing":
                issue_lines.append(f"- missing 块#{idx}：源『{iss.get('text', '')}』未译")
            elif check == "table":
                issue_lines.append(
                    f"- table：单元格缺 {iss.get('empty_cells')}，文本缺 {iss.get('empty_text')}")
            else:
                issue_lines.append(f"- {check} 块#{idx}：{iss}")
        if issue_lines:
            lines.append("已由确定性审计发现以下问题，请逐条核对并修正：")
            lines.append("\n".join(issue_lines))
        else:
            lines.append("（确定性审计未发现可列问题。）")
    lines.append(
        "核对要点：版面 check_layout（压得过小/溢出自身框/压入同列下一块）、漏译 check_residual+check_missing"
        "（**应翻译而未翻译**的普通文本块；纯数字/金额/单位/代码块是刻意保留的原文，**不属于漏译**）、"
        "数字 check_numbers（与原文**按值**一致，千分位/小数/单位拆分等格式差异不算问题）、"
        "完整性 check_table（表格单元格与普通文本块齐全、无空缺）。"
    )
    if auto_fix:
        lines.append(
            "修正：**多条 finding 先 retranslate_blocks(page, indices=[...]) 一次批量重译并写入**"
            "（绕过缓存，不会让旧译文复现，远快于逐条重译）；确知的译名 set_text；"
            "单个块或只想拿到译文再写回的用 retranslate_block；apply_annotation 按用户框选改动。"
            "必须让本页适用的上述各项都无问题才结束，否则继续修正，不得提前结束。"
        )
    else:
        lines.append("（本次为**只读复核**：只报告问题，不要修改任何译文。）")
    return "\n".join(lines)


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
    knows it can inspect / navigate / edit / re-export the loaded PDF.
    """
    return (
        "\n\n【可用工具】用户要看/改/导出/开始翻译当前 PDF 时再调用；纯闲聊不要用。"
        "读/导航/改工具（get_doc_info、classify_page、read_page、goto_page、set_block_text、"
        "delete_block_text、apply_annotation）的功能见各工具说明，这里只讲入口规则：\n"
        "**页码规则（必须一致）**：工具参数里的 `page` 是 **0 起**（第 1 页 = page 0）；"
        "但你向用户说明页码时，**一律说「第 N+1 页」**（如 page 0 → 第 1 页、page 1 → 第 2 页），"
        "**绝不说“第 0 页”**——用户看到的 UI 是 1 起的「第 N 页」。\n"
        "**用户要最新译文 → 直接调 re_export**（用当前修改重新生成，秒级、不重译）。若不可用"
        "（还没翻译过/没加载源文件），提示用户点主界面的「重新导出」按钮，**不要说没有导出功能**——应用有「重新导出」。\n"
        "**「重新翻译」＝全新翻译，别做成「重新导出」、更别反问用户**：用户说「重新翻译/再翻一遍/从头翻译/重跑一遍/"
        "换设置再翻」→ 直接调 **run_translate**（重新提取 + 重新翻译 + 导出，等价于从头重跑，当前界面设置"
        "——含「OCR表格重建」等导出选项——在启动时读取并生效）；`re_export` 只是拿**上一次的译文**重新生成文件"
        "（秒级、不重译），只有在用户明确说「按我刚才的修改重新导出/不用重译」时才用。"
        "**不要问用户「要重新翻译还是重新导出」**——「重新翻译」语义唯一，就是全新翻译。\n"
        "**用户要「检查/自检/核对」（“第N页数字对不对/有没有漏译/翻译得怎么样”）→ 调 self_check**"
        "（只读、秒级返回 findings；page 不传查全文，checks 可传子集如 ['numbers']）。\n"
        "**用户要「局部/定点重译」（“把第3页第2块重译/翻成…”“第5页整体重译”→ 调 retranslate**"
        "（按 flat 块索引或整页重译并写入受保护覆盖层；返回 count/indices/failed——failed 是重译失败而保留原文的块，"
        "应如实告知用户；改完提醒 re_export）。**retranslate 是“改一处就重译那一处”，不是整篇重跑。**\n"
        "**用户要「自定义流程」（“只查第3-8页的数字和表格，不修改”“第5页数字错了自动改”）→ 调 run_flow**"
        "（把整句要求传 requirement：规则/模型编译成 scope+checks+auto_fix。auto_fix=True 且重译通道可用时"
        "会**就地修正**审计发现的问题块并写回覆盖层（返回 fixed/remaining）；否则只读审计。传 name 可命名沉淀为本会话流程）。\n"
        "**用户要「一个完整的多步要求」（“先检查第3-8页数字，再导出”“把第3页第2块翻成Bank并重新导出”）→ 调 run_plan**"
        "（把整句要求**分解成按顺序执行**的多个任务：单个工具/翻译流程/自检/导出，可混合层级；每个任务带 tier+name+params。"
        "某步失败会停止并说明是哪个任务——如实转告用户）。\n"
        "**用户要「开始翻译」（或“翻译这个/帮我翻译/再次开始翻译/重新翻译”）→ 立刻调 run_translate**："
        "**「开始翻译」永远＝全新翻译**（重新提取 + 重新翻译 + 导出，按当前界面设置），所以"
        "**不要先 get_settings 汇报设置、不要问“确认开始吗/要重新翻译还是重新导出”、不要复述“即将开始”**——"
        "用户说了就开始。只有用户明确要求改语言/输出格式时才先 set_setting；用户对某页/某块的具体要求"
        "（如“第3页公司名翻成Bank”“只翻第2-5页”）作为 requirement 传给 run_translate。"
        "④ 运行中需要用户决定时翻译 agent 自己会提问。"
        "**注意：标准流程只做“翻译+导出+完成报告”，不会自动问“是否自检/是否导出”**——"
        "翻译完成后应用直接给出报告；用户要是想“额外核对/自检”，**是另一个动作**，用上面的 self_check/run_flow 单独触发，"
        "**不要假设或复述“翻完会自动自检”**。若没选源文件/模型不可用，提示用户先选好再试。"
    )


def interpret_special_answer(answer: str, kind: str) -> str:
    """Ask the model to classify a user's special-page answer into translate/keep/skip.

    The M3 negotiation surfaces buttons AND a free-text field; the model reads whatever
    the user said (including skips/retains/paraphrases) and returns one of the three
    actions, so the decision is AI-interpreted rather than exact-string matched.
    """
    return (
        f"用户在处理一个特殊页（类型：{kind}）。用户给出的选择是：\n"
        f"{answer}\n"
        "请判断用户想要哪种处理：翻译该页(translate)、保留原文不动(keep)、跳过(skip)。"
        "只回复一个词：translate 或 keep 或 skip，不要任何其他文字。"
    )

