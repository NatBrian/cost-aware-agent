"""Prompt text + parsing. Deterministic regex/XML parsing only — no second
LLM call to interpret the model's own output.
"""

import re

PLANNING_PROMPT = """[COST-AWARE-AGENT — PLANNING]
Before acting, decompose this task into constraints:
- Exploration clues: what needs to be found/expanded to make progress
- Verification clues: what needs to be validated once a candidate answer/fix exists

Output your decomposition in this exact format (nothing else in this block):

<checklist>
<item type="exploration">short description</item>
<item type="verification">short description</item>
</checklist>

Each item will be tracked for the rest of this session — it is never deleted, only marked
satisfied / contradicted / unverifiable as you gather evidence. Update your checklist's
priority as budget shrinks: narrow exploration breadth and increase verification depth when
budget is low."""

TIER_GUIDANCE = {
    "HIGH": "Full exploration budget available. Investigate multiple approaches if the problem is ambiguous.",
    "MEDIUM": "Budget available but not abundant. Prefer the most promising approach over broad exploration.",
    "LOW": "Budget constrained. Commit to current approach unless clearly contradicted. Avoid starting new exploration threads.",
    "CRITICAL": "Budget nearly exhausted. Finalize with current evidence unless it is clearly insufficient.",
}

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


def render_budget_tracker(
    spent_usd: float, budget_estimate_usd: float, tool_calls_used: int, tier: str,
    plan_rows,
) -> str:
    remaining_usd = max(0.0, budget_estimate_usd - spent_usd)
    block = (
        "Budget Tracker <budget>\n"
        f"LLM cost used: ${spent_usd:.2f}, remaining (of session estimate): ${remaining_usd:.2f}\n"
        f"Tool calls used: {tool_calls_used}\n"
        f"Tier: {tier}\n"
        f"{TIER_GUIDANCE[tier]}\n"
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


def streak_fact(streak: int) -> str:
    return f"[STREAK] You have chosen CONTINUE {streak} times in a row while budget was LOW or CRITICAL."
