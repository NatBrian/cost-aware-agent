"""Prompt text + parsing. Deterministic regex/XML parsing only — no second
LLM call to interpret the model's own output.
"""

import re

PLANNING_PROMPT = """[COST-AWARE-AGENT — PLANNING]
Before acting, decompose this task into constraints:
- Exploration clues: what needs to be found/expanded to make progress
- Verification clues: what needs to be validated once a candidate answer/fix exists

Output your decomposition in this exact format (nothing else in this block). Replace the
placeholder text below with your own specific wording for this task — copying the placeholder
text verbatim is wrong and produces a useless checklist:

<checklist>
<item type="exploration">PLACEHOLDER — replace with what you need to find/expand</item>
<item type="verification">PLACEHOLDER — replace with what you need to validate</item>
</checklist>

Each item will be tracked for the rest of this session — it is never deleted, only marked
satisfied / contradicted / unverifiable as you gather evidence. Update your checklist's
priority as budget shrinks: narrow exploration breadth and increase verification depth when
budget is low."""

# §6 philosophy (2026-07-03): the harness states measured facts and delegates
# ALL judgment to the model — we measure the MODEL's economics, not our rule
# text. The four prescriptive per-tier sentences that used to live here
# ("Prefer the most promising approach...", "Finalize with current
# evidence...") are gone: tier labels stay as compact state, and one fixed
# delegation line replaces the verdicts. (Deviates from the 2026-07-01
# BATS-mimic decision — superseded by the user-endorsed model-does-the-judgment
# philosophy.)
TIER_DELEGATION = "Decide yourself what these numbers mean for your next step."

_ITEM_RE = re.compile(
    r'<item\s+type="(exploration|verification)">(.*?)</item>', re.DOTALL
)
_VERIF_ITEM_RE = re.compile(
    r'<item\s+id="(\d+)"\s+status="(satisfied|contradicted|unverifiable)"\s*/>'
)
_DECISION_RE = re.compile(r"<decision>\s*(SUCCESS|CONTINUE|PIVOT)\s*</decision>")


def parse_checklist(text: str) -> list[tuple[str, str]]:
    """Returns [(clue_type, clue_text), ...]. Empty list if no well-formed
    <checklist> block — caller proceeds with empty plan."""
    return [(t, body.strip()) for t, body in _ITEM_RE.findall(text)]


def parse_verification(text: str) -> tuple[list[tuple[int, str]], str | None]:
    """Returns ([(item_id, status), ...], decision_or_None)."""
    items = [(int(i), s) for i, s in _VERIF_ITEM_RE.findall(text)]
    m = _DECISION_RE.search(text)
    decision = m.group(1) if m else None
    return items, decision


def render_checklist(plan_rows) -> str:
    if not plan_rows:
        return ""
    lines = ["<checklist>"]
    for row in plan_rows:
        lines.append(
            f'<item id="{row["id"]}" type="{row["clue_type"]}" status="{row["status"]}">'
            f'{row["clue_text"]}</item>'
        )
    lines.append("</checklist>")
    return "\n".join(lines)


LAG_NOTE = ("Note: cost is measured from completed turns; the current turn's "
            "tokens are not included yet, so true spend is slightly higher.")

PRICE_UNKNOWN_NOTE = ("Warning: some LLM calls this session used a model with no "
                      "known price; they are costed at a conservative default "
                      "rate, so true spend may differ from the number above.")


def render_budget_tracker(
    spent_usd: float, budget_estimate_usd: float, tool_calls_used: int | None,
    tier: str, plan_rows, lagging: bool = False, approximate: bool = False,
    price_unknown: bool = False, scope: str = "session estimate",
    burn_usd: float | None = None, burn_window_secs: int = 600,
) -> str:
    """tool_calls_used=None omits the per-call counter line and approximate=True
    prefixes the dollar figures with '~'. Both exist for REBUILT channels
    (OpenCode's system prompt), where the tracker is re-sent on every LLM call:
    any byte that changes per call (an exact running total, a call counter)
    invalidates the provider's prompt cache from that point on. Measured on
    deepseek: exact per-call text halved the cache-hit rate (72% -> 44% of input
    tokens) and nearly DOUBLED session cost (+92%) — the harness taxing the
    session it is meant to cheapen. Callers on rebuilt channels pass spend
    quantized to the injection bucket so the rendered text is byte-stable
    between bucket/tier transitions."""
    remaining_usd = max(0.0, budget_estimate_usd - spent_usd)
    # adaptive precision: a $0.0025 budget rendered at :.2f reads "$0.00 of
    # $0.00" — the model gets tier text but no usable numbers. Sub-cent scales
    # (cheap models priced at retail) need more decimals.
    dp = 2 if budget_estimate_usd >= 0.095 or budget_estimate_usd <= 0 else 4
    approx = "~" if approximate else ""
    lag_line = f"{LAG_NOTE}\n" if lagging else ""
    # flips 0->1 at most once per session, so rebuilt-channel byte-stability holds
    price_unknown_line = f"{PRICE_UNKNOWN_NOTE}\n" if price_unknown else ""
    # measured trailing spend, shown only once non-zero — the model does the
    # extrapolating (rebuilt-channel callers pass burn quantized to the bucket
    # step so the line stays byte-stable between transitions)
    burn_line = ""
    if burn_usd is not None and burn_usd > 0:
        burn_line = (f"Burn rate: {approx}${burn_usd:.{dp}f} spent in the last "
                     f"{max(1, burn_window_secs // 60)} min\n")
    tools_line = (f"Tool calls used: {tool_calls_used}\n"
                  if tool_calls_used is not None else "")
    block = (
        "Budget Tracker <budget>\n"
        f"LLM cost used: {approx}${spent_usd:.{dp}f}, "
        f"remaining (of {scope}): {approx}${remaining_usd:.{dp}f}\n"
        f"{tools_line}"
        f"{burn_line}"
        f"Tier: {tier}\n"
        f"{TIER_DELEGATION}\n"
        f"{lag_line}"
        f"{price_unknown_line}"
        "</budget>"
    )
    checklist = render_checklist(plan_rows)
    return block if not checklist else f"{block}\n\n{checklist}"


def is_milestone(tool_name: str, tool_input: dict, config: dict) -> bool:
    if tool_name in config["milestone_tool_patterns"]:
        return True
    if tool_name.lower() in ("bash", "shell", "run_command"):
        command = ""
        if isinstance(tool_input, dict):
            command = str(tool_input.get("command", ""))
        command_lower = command.lower()
        return any(p in command_lower for p in config["milestone_bash_test_patterns"])
    return False


SELF_VERIFICATION_PROMPT = """[SELF-VERIFICATION]
Check each item in your checklist against what you've learned so far. Respond in this exact
format (nothing else in this block):

<verification>
<item id="1" status="satisfied"/>
<item id="2" status="contradicted"/>
<item id="3" status="unverifiable"/>
<decision>SUCCESS | CONTINUE | PIVOT</decision>
<reason>one line</reason>
</verification>

item ids match the checklist shown in the last Budget Tracker block. status is one of:
satisfied (evidence confirms this constraint is met), contradicted (evidence rules this out),
unverifiable (cannot be determined with current information).

Given your checklist status and the Budget Tracker above, decide:
- SUCCESS: all constraints satisfied → finalize your answer/change
- CONTINUE: trajectory is promising, budget allows deeper work on it
- PIVOT: a constraint is contradicted, or this direction has exhausted its reasonable budget
  share → stop this approach, try an alternative"""


def spend_audit_question(delta_usd: float, budget_usd: float,
                         scope: str = "session estimate") -> str:
    """Model-driven self-audit at spend milestones — replaces rule-based
    dead-end detection (the harness never judges behavior; it states the
    measured number and asks). Appended to the tracker when spend crosses
    another budget slice on an accumulating channel.

    Wallet scope words it as PROJECT spend: the wallet delta includes every
    session in the project, so 'you have spent' would wrongly blame this
    session for a concurrent sibling's spend."""
    dp = 2 if budget_usd >= 0.095 or budget_usd <= 0 else 4
    if scope == "project wallet":
        subject = (f"This project has spent ${delta_usd:.{dp}f} across its "
                   "sessions since the last check")
    else:
        subject = f"You have spent ${delta_usd:.{dp}f} since the last check"
    return (f"[BUDGET CHECKPOINT] {subject}. In one sentence: what did that "
            "spend buy? If it bought nothing new, change course or finalize now.")


def streak_fact(streak: int) -> str:
    return f"[STREAK] You have chosen CONTINUE {streak} times in a row while budget was LOW or CRITICAL."


def session_history_fact(costs: list[float]) -> str | None:
    """Experience without estimation (§6): tell the model what past sessions in
    this project MEASURABLY cost and let it extrapolate. None when there is no
    history — never a guess."""
    if not costs:
        return None
    lo, hi = min(costs), max(costs)
    median = sorted(costs)[len(costs) // 2]
    dp = 2 if hi >= 0.095 else 4
    return (f"[PROJECT HISTORY] Your last {len(costs)} session(s) in this project "
            f"cost ${lo:.{dp}f}–${hi:.{dp}f} (median ${median:.{dp}f}). "
            "Use these measured costs as a baseline for what work here tends to cost.")


def render_receipt(dump: dict) -> str:
    """Human-readable end-of-session receipt from a /session/<id>/dump payload.
    Used by both the daemon (logged at /session/stop) and the CLI
    (`cost-aware-agent receipt <session>`), so the two never drift."""
    session = dump.get("session") or {}
    usage = dump.get("llm_usage") or []
    tools = dump.get("tool_calls") or []
    view = dump.get("budget_view") or {}
    spent = dump.get("spent_usd", 0.0)
    budget = view.get("budget_usd", 0.0)

    by_model: dict[str, tuple[int, float]] = {}
    for row in usage:
        n, c = by_model.get(row.get("model") or "?", (0, 0.0))
        by_model[row.get("model") or "?"] = (n + 1, c + (row.get("cost_usd") or 0.0))
    by_tool: dict[str, int] = {}
    for row in tools:
        by_tool[row.get("tool_name") or "?"] = by_tool.get(row.get("tool_name") or "?", 0) + 1

    dp = 2 if budget >= 0.095 or budget <= 0 else 4
    lines = [f"=== cost-aware-agent receipt — session {session.get('session_id', '?')} ==="]
    times = [r["created_at"] for r in usage if r.get("created_at")]
    if times:
        lines.append(f"duration: {(max(times) - min(times)) // 60}m {(max(times) - min(times)) % 60}s (first to last LLM call)")
    lines.append(f"LLM cost: ${spent:.{dp}f} across {len(usage)} calls")
    for model, (n, c) in sorted(by_model.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"  {model}: ${c:.{dp}f} ({n} calls)")
    if budget > 0:
        pct = view.get("spent_usd", spent) / budget * 100
        lines.append(f"budget: ${budget:.{dp}f} ({view.get('scope', 'session estimate')}) — {pct:.0f}% used")
        if view.get("scope") == "project wallet":
            lines.append(f"wallet remaining: ${max(0.0, budget - view.get('spent_usd', 0.0)):.{dp}f}")
    if by_tool:
        top = ", ".join(f"{t} {n}" for t, n in sorted(by_tool.items(), key=lambda kv: -kv[1]))
        lines.append(f"tool calls: {len(tools)} ({top})")
    lines.append(f"injections delivered: {len(dump.get('injections') or [])}")
    unknown = sum(1 for r in usage if r.get("price_unknown"))
    if unknown:
        lines.append(f"WARNING: {unknown} call(s) used a model with no known price — "
                     "costed at a conservative Sonnet-tier fallback; true total may "
                     "be higher or lower")
    expensive = sorted(usage, key=lambda r: -(r.get("cost_usd") or 0.0))[:3]
    if expensive and (expensive[0].get("cost_usd") or 0.0) > 0:
        tops = ", ".join(f"${(r.get('cost_usd') or 0.0):.{dp}f}" for r in expensive)
        lines.append(f"biggest single calls: {tops}")
    return "\n".join(lines)
