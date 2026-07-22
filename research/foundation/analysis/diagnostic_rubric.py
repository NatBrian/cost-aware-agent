"""Offline six-dimension diagnostic (F7, optional) — adapted from the lab's
PPTAgent trajectory-eval rubrics (/mnt/src/code/PPT-GEN-Demo/eval_codes/
trajectory_eval/, reviewed 2026-07-22; see F3 'Prior art' for what was adopted
vs dropped). Dimensions that apply to single-agent QA, 1–5 anchored scales.

ANALYSIS ONLY: human-read before/after-training comparison ("did RL improve
only stopping, or also search skill?"). Never a reward; never touches the gate.
Ask the authors before any verbatim reuse in public artifacts.
"""

import json

DIMENSIONS = {
    "query_quality": """Rate the SEARCH QUERY QUALITY of this trajectory 1-5:
5 = queries precise and well-targeted; each targets a distinct needed fact;
    terminology matches the domain; logical sequencing toward the answer.
4 = strong queries, minor vagueness or one suboptimal formulation.
3 = adequate; some queries broad or weakly targeted but progress happens.
2 = mostly vague/overlapping queries; progress despite, not because, of them.
1 = queries unrelated to the information need or near-duplicates throughout.""",
    "retrieval_use": """Rate the RETRIEVAL EFFECTIVENESS of this trajectory 1-5:
5 = retrieved content relevant every step; no duplicate searches; follow-ups
    clearly refined from earlier results; search count fits task complexity.
4 = mostly effective; one redundant or unproductive search.
3 = mixed: useful and wasted searches in similar measure.
2 = majority of searches redundant or off-target.
1 = retrieval essentially unused or entirely off-topic.""",
    "reasoning_integration": """Rate SEARCH-INTEGRATED REASONING 1-5:
5 = drafts update promptly on new evidence; claims grounded in retrieved facts,
    not priors; multi-hop links made explicitly; final answer follows from
    the collected evidence.
4 = solid grounding with a small lag or one unsupported leap.
3 = partial: some retrieved facts used, some ignored; draft updates late.
2 = reasoning mostly detached from what was retrieved.
1 = final answer unrelated to any retrieved evidence (pure prior/guess).""",
}

PROMPT = """You are evaluating one complete agent trajectory (question-answering
with a search tool). You do NOT know the correct answer.

[QUESTION] {question}
[TRAJECTORY]
{trajectory}

{dimension_prompt}

Reply with exactly this JSON and nothing else:
{{"reasoning": "<two sentences>", "score": <integer 1-5>}}"""


def render_trajectory(ep: dict, max_obs_chars: int = 200) -> str:
    lines = []
    for s in ep["steps"]:
        obs = " ".join(s["obs_digest"].split())[:max_obs_chars]
        lines.append(f"step {s['t']}: {s['action_type']}[{s['query_or_answer']}]"
                     f" -> {obs} | draft: {s['draft']}")
    lines.append(f"final answer: {ep['final_answer']}")
    return "\n".join(lines)


def diagnose_episode(ep: dict, judge_complete) -> dict:
    """judge_complete: callable(prompt)->str (e.g. JudgeClient._complete).
    Returns {dimension: {"score": int|None, "reasoning": str}}."""
    out = {}
    for dim, dprompt in DIMENSIONS.items():
        prompt = PROMPT.format(question=ep["question"],
                               trajectory=render_trajectory(ep),
                               dimension_prompt=dprompt)
        try:
            obj = json.loads(_extract_json(judge_complete(prompt)))
            score = int(obj.get("score", 0))
            out[dim] = {"score": score if 1 <= score <= 5 else None,
                        "reasoning": str(obj.get("reasoning", ""))[:300]}
        except (ValueError, json.JSONDecodeError):
            out[dim] = {"score": None, "reasoning": "(parse failure)"}
    return out


def _extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    return text[start:end + 1]
