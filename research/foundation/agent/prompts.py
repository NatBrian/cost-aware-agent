"""Agent prompts + deterministic parsing (F2).

Facts-not-advice: the tracker states measured numbers and one fixed delegation
line — no prescriptive guidance anywhere, in any arm (design precedent:
cost_aware_agent/prompts.py §6; see F2 doc). Arm a0 gets NO budget content.
"""

import re

EMPTY_DRAFT = "EMPTY_DRAFT"

SYSTEM_PROMPT = """You answer questions by searching a Wikipedia corpus, step by step.

Each turn, reply in EXACTLY this format (three lines, nothing else):

THOUGHT: one or two sentences of reasoning.
ACTION: search[your search query] OR answer[your final answer]
BEST ANSWER SO FAR: your current best one-line answer, or EMPTY_DRAFT if none yet.

Rules:
- search[...] runs one search; you will see the top results next turn.
- answer[...] submits your final answer and ends the task.
- The BEST ANSWER SO FAR line is mandatory every turn.{budget_rules}"""

BUDGET_RULES = """
- You have a budget of {B} steps for this task. Each search and the final answer each cost one step."""

TRACKER_TEMPLATE = """<budget>
Steps used: {t} of {B}. Remaining: {remaining}.
Decide yourself what these numbers mean for your next step.
</budget>"""

_ACTION_RE = re.compile(r"ACTION:\s*(search|answer)\s*\[(.*?)\]\s*$",
                        re.MULTILINE | re.DOTALL | re.IGNORECASE)
_DRAFT_RE = re.compile(r"BEST ANSWER SO FAR:\s*(.+?)\s*$", re.MULTILINE)


def system_prompt(with_budget: bool, budget: int | None = None) -> str:
    rules = BUDGET_RULES.format(B=budget) if with_budget else ""
    return SYSTEM_PROMPT.format(budget_rules=rules)


def tracker_block(t: int, budget: int) -> str:
    return TRACKER_TEMPLATE.format(t=t, B=budget, remaining=max(0, budget - t))


def parse_step(text: str) -> dict | None:
    """Returns {"action_type": "search"|"answer", "content": str, "draft": str}
    or None when the ACTION line is malformed (caller retries once)."""
    m = _ACTION_RE.search(text)
    if not m:
        return None
    action_type = m.group(1).lower()
    content = " ".join(m.group(2).split())
    if not content:
        return None
    d = _DRAFT_RE.search(text)
    draft = " ".join(d.group(1).split()) if d else EMPTY_DRAFT
    if draft.upper() == EMPTY_DRAFT:
        draft = EMPTY_DRAFT
    return {"action_type": action_type, "content": content, "draft": draft}
