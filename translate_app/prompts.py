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
        "- Preserve numbering exactly: reply as '[n] translated text' per "
        "block, in the same order.\n"
        "- Do not merge or split blocks; output ONLY the numbered translations, "
        "with no explanations, notes or preamble.\n"
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
        "[1]\n"
        f"{inp1}\n"
        "[2]\n"
        f"{inp2}\n"
        "Output:\n"
        "[1]\n"
        f"{out1}\n"
        "[2]\n"
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
        "② 某块保留原文还是翻译、判断不确定时（『保留/翻译』）。"
        "③ 要删除/覆盖/擦除译文或整页（不可逆，先确认）。"
        "④ 接近预算上限、或当前无法推进时（询问是否收尾/继续）。\n"
        "其余情况不要打扰用户。ask_user 返回的是用户的选择，据此继续。"
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
        "各工具的返回值见其 schema，这里只补关键差异：\n"
        "① 观察（只读原文，绝不改写）：read_page（含扁平 index，超大页可 offset/limit 分页）、get_layout、get_doc_info、"
        "classify_page、render_page（渲染译文 PNG 供自检）、preview_page（弹窗给用户看）。\n"
        "② 修改（写可编辑译文，原文不可动）：translate_block（翻译并**写入**，首选）、set_text（写已知译名；数字/代码块被拒）、"
        "retranslate_block（**只返回译文**，需再 set_text 写入）、apply_terminology（锁术语）、delete_block（撤为原文）、"
        "apply_annotation（按用户框选改）。\n"
        "③ 校验（只报告，无副作用）：check_residual（残留+空块；目标为西文看残留中文、中文目标看未译英文成句；"
        "纯代码/缩写/单位/数字不算）、check_missing（源有译文空；纯数字/代码不算）、check_numbers（数字按值一致；"
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
        "术语先 apply_terminology 锁定；对每个**应翻译**的块 translate_block；纯数字/代码/单位块保持原样、不调用翻译工具。\n"
        "② 编辑/修改（复核或修错）：read_page + render_page 看当前译文与版面；按 "
        "check_residual / check_missing / check_numbers / check_table / check_layout 的问题清单决定改哪里——"
        "确知译名 set_text；需重译 translate_block 或 retranslate_block（后者只返回译文）；按框选 apply_annotation；"
        "恢复原文 delete_block。\n"
        "③ 校验：每改一处就复跑对应 check_*，直到 check_residual / check_missing 无问题、且（本页适用时）"
        "check_numbers / check_table / check_layout 无实质问题，才可结束。扫描表报的「字号偏小/行带受限」是物理极限，可留意但不必强修。\n"
        "④ 收尾：整页确认满足才结束；拿不准/术语/保留还是翻译→ask_user；某工具 ok=false→先读 error 判断原因、别盲目重试。"
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
    """
    n = page_index + 1
    head = f"这是文档第 {n} 页" + (f"（{kind} 页）" if kind else "") + "。"
    return (
        f"{head}请把本页**所有文本块**均翻译成 {lang}——整页最终应**全部是 {lang}**；"
        "只有纯数字/金额/代码/单位块保持原样。\n"
        "① read_page(page) 读本页全部块并记下 index；② 对每个**应翻译**的块 translate_block(index)（自动用该块原文），"
        "确知的译名可直接 set_text(page, index, text)；③ 最后**必须** check_residual 校验。"
        f"若仍报有块未译/残留，必须继续翻译——**确认整页都已译成 {lang}、且无残留才结束，不得提前结束**。"
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
    return (f"第 {n} 页类型不确定。如何处理？", ["翻译", "保留原文", "跳过"])


def review_mode_question() -> tuple[str, list[str]]:
    """M4: ask whether the draft is handed to the AI self-check or the user checks it."""
    return ("全文初稿已生成。交由 AI 自检，还是你手动检查？", ["AI 自检", "我手动检查"])


def review_export_question() -> tuple[str, list[str]]:
    """M4: after the review, confirm whether to export the result."""
    return ("复核完成。是否导出？", ["导出", "继续检查"])


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
            "修正：set_text 写入已知译文；retranslate_block 只返回译文、需再 set_text 写入；"
            "apply_annotation 按用户框选改动。必须让本页适用的上述各项都无问题才结束，否则继续修正，不得提前结束。"
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
        "**用户要最新译文 → 直接调 re_export**（用当前修改重新生成，秒级、不重译）。若不可用"
        "（还没翻译过/没加载源文件），提示用户点主界面的「重新导出」按钮，**不要说没有导出功能**——应用有「重新导出」。\n"
        "**用户要「开始翻译」（或“翻译这个/帮我翻译”+要求）→ 这是入口**："
        "① 先 get_settings 看当前设置（源文件名/目标语言/输出格式/模型）；"
        "② 若用户要求改语言/格式，先 set_setting 改；"
        "③ 把用户的具体要求（如“第3页公司名翻成Bank”“只翻第2-5页”）作为 requirement 传给 run_translate，"
        "**不要自己复述“即将开始”而不启动**；"
        "④ 运行中需要用户决定时 AI 会提问。若没选源文件/模型不可用，提示用户先选好再试。"
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


#: The message the app sends to the chat on startup so the assistant greets the user.
CHAT_GREETING = "你好"


# ---------------------------------------------------------------------------
# 4. Tool descriptions — every OpenAI tool's ``description`` string is also prompt
# text that steers the model.  The tool modules (``agent/tools.py``,
# ``chat_tools.py``) and ``translator._CLASSIFY_TOOL`` reference these instead of
# authoring their own, so all model-facing text lives in one place.
# ---------------------------------------------------------------------------

#: The translation-agent tool descriptions (``agent/tools.py``), keyed by tool name.
#: Only tools that are actually bound AND have a real effect on the pipeline/output
#: are exposed (see ``agent/flow.py`` ``run_page_visual``); dead/broken tools are
#: kept out so the model never sees a tool that does nothing.
AGENT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_page": "读取指定页的文本块与布局元数据",
    "get_layout": "提取某页的行/列聚类、二维网格与单元格跨度",
    "get_doc_info": "返回原文件信息：页数/标题/语言/文本页/扫描页/图表页/块数/每页类型（表格页恒为 0）",
    "classify_page": "判定某页类型：normal/scan/chart/uncertain",
    "translate_block": "翻译某块并写入其译文（走缓存+编号协议）",
    "retranslate_block": "避开缓存强制重译一段文本，**只返回译文**（不会写入；需再用 set_text 把它写到目标块）。常用于残中/空缺修正",
    "set_text": "把某块文本直接置为指定值（数字/代码块会被拒绝）",
    "apply_annotation": "按预览中用户框选的区域改写/删除对应块（M6）",
    "apply_terminology": "为某源词设定统一的术语译文（会并入本页翻译所用的术语表）",
    "delete_block": "移除某块的译文覆盖，恢复为保留原文（不再翻译该块）",
    "check_residual": "检查某页是否有未翻译残留（目标为西文时看残留中文；目标为中文时看未译的英文成句；纯代码/缩写/单位不算）与空块",
    "check_missing": "检查某页是否有源有译文空的块（内容缺失；纯数字/代码块不算）",
    "check_numbers": "核对某页译文的数字/金额是否与原文**按值**一致：源里的数值被删或改错才报告（千分位/小数/单位拆分等格式差异不算；中文【序数→数字】如 二→2 不算），返回不一致的块",
    "check_table": "检查某页表格单元格与普通文本块的完整性：源可译单元格数 vs 已译数、空/缺失单元格、遗漏文本块",
    "check_layout": "仿照导出器重新测量某页译文的版面：看是否低于可读下限、溢出自身框、压入同列下一块（两栏页只与同栏比较）",
    "audit_page": "对某页一次性跑指定的确定性审计并合并结果：返回 {checks_requested, checks, issues, clean}（issues 是带 check 标签的列单项，供你逐条修正）；checks 可传子集（如 ['numbers','table']），默认全五类。只读复核（不修任何东西）时把 clean 当作本轮是否达标",
    "render_page": "把当前处理到该页的译文渲染成 PNG 供视觉自检（检查溢出/越线/密度）",
    "preview_page": "在预览窗口显示指定页面（供用户查看），可聚焦某区域/块",
    "ask_user": "向用户提问并等待回答（关键决策/歧义/术语确认）",
}

#: The interaction-chat tool descriptions (``chat_tools.py``), keyed by tool name.
CHAT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_doc_info": "返回当前 PDF 的信息：页数/标题/语言/文本页/扫描页/图表页/块数/每页类型（表格页恒为 0）。",
    "get_settings": "返回当前应用设置快照：源文件名、目标语言、输出格式键与显示名、输出路径、模型名称/id、是否 OCR/智能编排。开始翻译前先看它确认设置。",
    "classify_page": "判定某页类型：normal/scan/chart/uncertain。",
    "read_page": "读取某页的原始文本块与当前译文，含扁平块索引（供 set_block_text 使用）与布局元数据。",
    "goto_page": "在预览窗口显示指定页（原文/译文侧）。",
    "set_block_text": "把某块的译文直接置为指定文本（数字/代码块会被拒绝；写的是受保护的译文层）。",
    "delete_block_text": "移除某块的 AI 编辑，恢复为未被覆盖的译文。",
    "apply_annotation": "按预览中用户框选的区域改写/删除对应块（用 read_page 拿到的 bbox）。",
    "re_export": "用当前已加载 PDF 上一次的成功译文，重新导出（应用本次对话/标注里已有的修改；不重新翻译、秒级）。",
    "run_translate": "用**当前设置**开始翻译（把用户的具体要求作为 requirement 传入，会随运行注入到 AI 编排层）。控制权交给翻译流水线，完成在主窗口日志/进度提示。",
    "set_setting": "修改翻译设置项（key 为 target_language 或 output_type，value 为语言名/输出格式键 translated_pdf|bilingual_pdf|markdown|plain_text），下次运行生效。",
}
