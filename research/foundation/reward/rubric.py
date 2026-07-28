"""rubric_v1 — the reward spec (F3). THE file your review approved; any edit
bumps the version and re-runs calibration.

Design rules (F3 doc): binary bits with anchored criteria + worked examples
(PPTAgent trajectory-eval discipline; retrieval-efficiency wording adapted);
judge NEVER sees gold; exact quantities are computed in code, never judged;
weights designed and frozen, not fitted.
"""

from agent.prompts import EMPTY_DRAFT

RUBRIC_VERSION = "rubric_v3"

STEP_BITS = ("new_info", "not_redundant", "was_needed")
ANSWER_BITS = ("supported", "nothing_left")

# The prompt is DATA BLOCK + INSTRUCTION BLOCK, split at the anchors below.
# Keeping the split explicit lets a stored v1 judging context be re-rendered
# under v2 instructions without the original episode (see upgrade_context) —
# which is what preserves the 50 human calibration labels across a rubric bump.
STEP_ANCHOR = "Answer three YES/NO questions about THIS step:"
ANSWER_ANCHOR = "Answer two YES/NO questions about stopping NOW with THIS answer:"

STEP_DATA = """You are auditing ONE step of an agent that answers questions by \
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

"""

# v2 changes (2026-07-28), each traceable to measured judge behaviour:
#  - disjointness note: new_info judges the RESULT, not_redundant the QUERY.
#    Without it a judge that likes a step sets both to 1, which is how a
#    rephrase-loop scores 0.7 instead of 0.3.
#  - evidence citation (nearest_prior_step + evidence_quote): a judge that must
#    name the step it compared against cannot wave a rephrase through, and one
#    that must quote the novel span cannot assert new_info from vibes.
#  - tie-breaks pointed AT each bit's measured error direction: gemma was
#    lenient on redundancy (4 false passes), so not_redundant defaults NO;
#    Qwen3.6-27B is strict on the answer bits, so those require a NAMED gap.
STEP_INSTRUCTIONS = """Answer three YES/NO questions about THIS step:

new_info judges the RESULT this step returned. not_redundant judges the QUERY
this step issued. They are independent: a redundant query can still return new
information (new_info=1, not_redundant=0), and a well-chosen query can return
nothing useful (new_info=0, not_redundant=1). Score them separately.

1. new_info — Did the result add relevant information that was NOT already in the
   history? YES only if the result contains material relevant to the question and
   absent from earlier results. NO if results are off-topic, empty, or repeat
   known content. Answer YES only if you can point to a specific new span in the
   result; say which one in reasoning.
   Example YES: first search for a film returns its director's name, needed next.
   Example NO: results restate a fact already retrieved at an earlier step.

2. not_redundant — Is this query genuinely different from every earlier query?
   Name the most similar earlier step in nearest_prior_step (0 if this is the
   first step). NO if it duplicates or merely rephrases an earlier query's
   target, or re-asks what the history already answers. YES if it targets a
   distinct fact or entity.
   Example YES: history covered the film; this queries the director's birth year.
   Example NO: step 2 searched "X director"; this searches "who directed X".

3. was_needed — BEFORE this step, was more work still needed? YES if the draft
   was empty, incomplete, or unsupported by the history. NO if the draft already
   stated a complete answer that the history's evidence supported (then this step
   was waste even if its result was interesting).
   Example YES: draft was EMPTY_DRAFT.
   Example NO: draft named the correct-looking entity and two earlier results
   both confirmed it.

Reply with exactly this JSON and nothing else. Do not use double quotes inside
any string value.
{{"nearest_prior_step": <step number or 0>, "reasoning": "<one or two \
sentences>", "new_info": 0 or 1, "not_redundant": 0 or 1, "was_needed": 0 or 1}}"""

ANSWER_DATA = """You are auditing the STOP decision of an agent that answers \
questions by searching Wikipedia under a step budget. Judge only from what is \
shown — you do NOT know the correct answer.

[QUESTION] {question}
[BUDGET] stopped at step {t} of budget {B} ({remaining} steps left unused)
[FULL HISTORY]
{history}
[FINAL ANSWER GIVEN] {answer}

"""

ANSWER_INSTRUCTIONS = """Answer two YES/NO questions about stopping NOW with THIS answer:

1. supported — Is the final answer consistent with, and backed by, the evidence
   in the history? YES if retrieved results state or directly imply it. NO if it
   contradicts results, or nothing retrieved supports it (a guess).
   Example YES: answer names the person two results both identify.
   Example NO: answer names an entity no result ever mentioned.

2. nothing_left — Was stopping now the right call, i.e. is further searching
   unlikely to change or improve this answer? Work in order: (a) list the facts
   the question requires; (b) for each, say whether the history resolves it;
   (c) answer. Answer NO only if you can NAME a specific required fact that is
   still unresolved AND budget remained to pursue it. Otherwise answer YES —
   including when the remaining budget is too small to close the gap, and when
   you are undecided.
   Example YES: both hops of the question are answered and confirmed.
   Example NO: the answer's second half is a guess and 4 steps remained.

Reply with exactly this JSON and nothing else. Do not use double quotes inside
any string value.
{{"reasoning": "<one or two sentences>", "supported": 0 or 1, \
"nothing_left": 0 or 1}}"""

STEP_PROMPT = STEP_DATA + STEP_INSTRUCTIONS
ANSWER_PROMPT = ANSWER_DATA + ANSWER_INSTRUCTIONS


def upgrade_context(context: str, action_type: str) -> str:
    """Re-render a STORED judging context under the current instructions.

    The calibration sheet keeps the rendered prompt, not the episode, and the
    pilot trajectories were lost in the 2026-07-28 wipe. Splitting at the anchor
    swaps instructions while keeping the identical data block, so the 50 human
    labels — which are about the STEP, not the prompt wording — stay valid
    across a rubric bump. Raises if the anchor is missing rather than silently
    judging a malformed prompt.
    """
    anchor = ANSWER_ANCHOR if action_type == "answer" else STEP_ANCHOR
    instructions = ANSWER_INSTRUCTIONS if action_type == "answer" else STEP_INSTRUCTIONS
    head, sep, _ = context.partition(anchor)
    if not sep:
        raise ValueError(f"context has no {action_type} anchor — cannot upgrade")
    return head + instructions


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
