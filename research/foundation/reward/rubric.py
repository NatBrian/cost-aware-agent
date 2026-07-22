"""rubric_v1 — the reward spec (F3). THE file your review approved; any edit
bumps the version and re-runs calibration.

Design rules (F3 doc): binary bits with anchored criteria + worked examples
(PPTAgent trajectory-eval discipline; retrieval-efficiency wording adapted);
judge NEVER sees gold; exact quantities are computed in code, never judged;
weights designed and frozen, not fitted.
"""

from agent.prompts import EMPTY_DRAFT

RUBRIC_VERSION = "rubric_v1"

STEP_BITS = ("new_info", "not_redundant", "was_needed")
ANSWER_BITS = ("supported", "nothing_left")

STEP_PROMPT = """You are auditing ONE step of an agent that answers questions by \
searching Wikipedia under a step budget. Judge only from what is shown — you do \
NOT know the correct answer.

[QUESTION] {question}
[BUDGET] step {t} of budget {B} ({remaining} remaining after this step)
[HISTORY BEFORE THIS STEP]
{history}
[DRAFT BEFORE THIS STEP] {draft_before}
[THIS STEP] searched: {query}
[RESULT DIGEST] {obs}
[DRAFT AFTER THIS STEP] {draft_after}

Answer three YES/NO questions about THIS step:

1. new_info — Did the result add relevant information that was NOT already in the
   history? YES only if the result contains material relevant to the question and
   absent from earlier results. NO if results are off-topic, empty, or repeat
   known content.
   Example YES: first search for a film returns its director's name, needed next.
   Example NO: results restate a fact already retrieved at an earlier step.

2. not_redundant — Is this query genuinely different from every earlier query?
   NO if it duplicates or merely rephrases an earlier query's target, or re-asks
   what the history already answers. YES if it targets a distinct fact or entity.
   Example YES: history covered the film; this queries the director's birth year.
   Example NO: step 2 searched "X director"; this searches "who directed X".

3. was_needed — BEFORE this step, was more work still needed? YES if the draft
   was empty, incomplete, or unsupported by the history. NO if the draft already
   stated a complete answer that the history's evidence supported (then this step
   was waste even if its result was interesting).
   Example YES: draft was EMPTY_DRAFT.
   Example NO: draft named the correct-looking entity and two earlier results
   both confirmed it.

Reply with exactly this JSON and nothing else:
{{"reasoning": "<one or two sentences>", "new_info": 0 or 1, "not_redundant": 0 or 1, "was_needed": 0 or 1}}"""

ANSWER_PROMPT = """You are auditing the STOP decision of an agent that answers \
questions by searching Wikipedia under a step budget. Judge only from what is \
shown — you do NOT know the correct answer.

[QUESTION] {question}
[BUDGET] stopped at step {t} of budget {B} ({remaining} steps left unused)
[FULL HISTORY]
{history}
[FINAL ANSWER GIVEN] {answer}

Answer two YES/NO questions about stopping NOW with THIS answer:

1. supported — Is the final answer consistent with, and backed by, the evidence
   in the history? YES if retrieved results state or directly imply it. NO if it
   contradicts results, or nothing retrieved supports it (a guess).
   Example YES: answer names the person two results both identify.
   Example NO: answer names an entity no result ever mentioned.

2. nothing_left — Was stopping now the right call, i.e. is further searching
   unlikely to change or improve this answer? YES if the question's parts are all
   resolved, or remaining budget is too small for the missing part. NO if an
   obvious unresolved sub-question remained AND budget remained to pursue it.
   Example YES: both hops of the question are answered and confirmed.
   Example NO: the answer's second half is a guess and 4 steps remained.

Reply with exactly this JSON and nothing else:
{{"reasoning": "<one or two sentences>", "supported": 0 or 1, "nothing_left": 0 or 1}}"""


def _history_digest(steps: list[dict], upto_t: int, max_obs_chars: int = 200) -> str:
    lines = []
    for s in steps[:upto_t]:
        obs = " ".join(s["obs_digest"].split())[:max_obs_chars]
        lines.append(f"step {s['t']}: {s['action_type']}[{s['query_or_answer']}] -> {obs}")
    return "\n".join(lines) if lines else "(no steps yet)"


def render_step_prompt(ep: dict, idx: int) -> str:
    """Serialize the gold-free judging context for working step ep['steps'][idx]."""
    s = ep["steps"][idx]
    draft_before = ep["steps"][idx - 1]["draft"] if idx > 0 else EMPTY_DRAFT
    return STEP_PROMPT.format(
        question=ep_question(ep), t=s["t"], B=ep["budget_B"],
        remaining=max(0, ep["budget_B"] - s["t"]),
        history=_history_digest(ep["steps"], idx),
        draft_before=draft_before, query=s["query_or_answer"],
        obs=" ".join(s["obs_digest"].split())[:400], draft_after=s["draft"])


def render_answer_prompt(ep: dict, idx: int) -> str:
    s = ep["steps"][idx]
    return ANSWER_PROMPT.format(
        question=ep_question(ep), t=s["t"], B=ep["budget_B"],
        remaining=max(0, ep["budget_B"] - s["t"]),
        history=_history_digest(ep["steps"], idx),
        answer=s["query_or_answer"])


def ep_question(ep: dict) -> str:
    q = ep.get("question")
    if not q:
        raise KeyError("episode lacks 'question' — collection must store it "
                       "(judging is gold-free but needs the task text)")
    return q


def step_score(bits: dict, weights: dict[str, float]) -> float:
    return sum(weights[b] * float(bits[b]) for b in weights)


def step_reward(bits: dict, weights: dict[str, float], alpha: float) -> float:
    return alpha * (step_score(bits, weights) - 0.5)
