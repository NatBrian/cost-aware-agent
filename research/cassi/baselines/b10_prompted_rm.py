"""B10 — "RM-P": prompted reward model (paper_plan_v2_1 §5.2 row B10, §17 `prompted_rm`,
§20 changelog — [v2.1] addition).

WHAT: the trained-vs-prompted reward-model comparison, answered in our own tables.
A FROZEN judge (lab vLLM Qwen3.5-30B) reads the SAME §18.1 serialized x_t the
stopper M_θ reads (no ground truth, no executor-stated confidence — §2.1 rule),
does brief chain-of-thought, then emits a designed binary rubric. Two arms:

  (a) monitor  — training-free inference-time stopping: stop when the rubric's
                 continue-score ≤ θ_p (the §5.3 frontier knob, calibrated on dev);
  (b) rl       — executor GRPO with the rubric's state-value as the shaping
                 potential Φ in place of RM-T's V̂ (post-K1 GO only, 1 seed,
                 primary domain — §5.2). Reuses executor.shaping unchanged so the
                 ONLY difference vs CASSI is the potential's source (fair fight,
                 same §5.2 logic that makes B9 delete only the stopper).

The two derived scores deliberately mirror RM-T's heads: continue_score is the
prompted analog of the stop margin Δ̂; state_value is the prompted analog of V̂.

PREDICTED OUTCOME (pre-registered in §5.2): RM-P loses — LLM step-redundancy
judgment is ≤24.9% F1 (RedundancyBench 2605.29893), frozen judges get hacked
under RL (Gao 2210.10760; AgentPRM 82%→70%), and binary bits give near-zero
step-to-step differences so the dense signal r_t collapses. If RM-P wins, we
learn it early and cheaply. Either result strengthens the paper.

DESIGNED, NOT FITTED: rubric weights are fixed by design and documented in the
paper appendix. Fitting them to labels would BE training a reward model and
dissolve the comparison (§20 "considered and rejected").

COST KNOB (§5.3): rubric_threshold θ_p in [0,1] — higher ⇒ stops more often ⇒
cheaper, riskier frontier points.

BILLING (§5.3 symmetry): every judge call is billed under the shared price map
(`bill_judge`); CoT output tokens included. Nothing here is free.

TRAINING: none for the judge, ever (needs_training=False in the registry; the
rl arm trains the EXECUTOR, reusing the P6/P8 GRPO wiring, judge stays frozen).

PRODUCTION CLIENT: wrap the lab vLLM server with `VLLMJudgeAdapter` around
`cassi.executor.vllm_client.VLLMClient` (or anything with the same
`.generate(messages, max_tokens)` shape). CPU tests use a scripted fake.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from cassi.budget.cost import token_cost
from cassi.executor.shaping import shaped_step_rewards

COST_KNOB = "rubric_threshold"

# ----------------------------------------------------------------- the rubric
# §17 prompted_rm.protocol — order is part of the contract with the prompt.
RUBRIC_DIMS = (
    "draft_likely_correct",      # 1 = the current draft already looks correct
    "last_step_necessary",       # 1 = the most recent step added needed information
    "more_work_likely_helps",    # 1 = another step would likely improve the answer
    "budget_healthy",            # 1 = enough wallet remains to afford more work
)

# Designed weights (documented in the paper appendix; NEVER fitted — see docstring).
# continue_score: "how much is continuing still worth" — the Δ̂ analog. A correct-
# looking draft is the strongest reason to stop, hence its complement carries the
# largest weight.
CONTINUE_WEIGHTS = {
    "draft_likely_correct": 0.40,    # applied to (1 - bit): correct draft ⇒ stop
    "more_work_likely_helps": 0.30,
    "last_step_necessary": 0.15,
    "budget_healthy": 0.15,
}
# state_value: "how good is this state" — the V̂ analog (the rl arm's potential Φ).
STATE_VALUE_WEIGHTS = {
    "draft_likely_correct": 0.70,
    "budget_healthy": 0.30,
}

JUDGE_MAX_OUTPUT_TOKENS = 256    # CoT allowed; billed (§5.3)

JUDGE_PROMPT_TEMPLATE = (
    "You are a strict reviewer of a working AI agent. Below is the agent's "
    "current state (task, budget, progress, recent history, current draft "
    "answer).\n\n{serialized_x}\n\n"
    "Think step by step BRIEFLY (2-4 sentences), then end your reply with "
    "EXACTLY one final line of the form:\n"
    "RUBRIC: [a,b,c,d]\n"
    "where each letter is 0 or 1:\n"
    "  a = the current draft answer already looks correct\n"
    "  b = the most recent step added information that was actually needed\n"
    "  c = one more step of work would likely improve the final answer\n"
    "  d = the remaining budget comfortably affords more work\n"
    "Output the RUBRIC line last, with no text after it."
)

# Last match wins — the CoT may mention the format before the final line.
_RUBRIC_RE = re.compile(r"RUBRIC:\s*\[\s*([01])\s*,\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\]")

# Sentinel: no dev threshold met the calibration target ⇒ never stop early
# (score ≤ -inf is impossible). Mirrors b2_probe.NEVER_STOP's fail-open logic.
NEVER_STOP = -math.inf


def build_judge_prompt(serialized_x: str) -> str:
    """The full judge prompt for one §18.1-serialized state."""
    return JUDGE_PROMPT_TEMPLATE.format(serialized_x=serialized_x)


def parse_rubric(judge_output: str) -> tuple[int, int, int, int] | None:
    """Extract the final RUBRIC bits; None if absent/malformed (an unparseable
    judge reply never triggers a stop — fail-open to CONTINUE, like B2)."""
    matches = _RUBRIC_RE.findall(judge_output or "")
    if not matches:
        return None
    return tuple(int(b) for b in matches[-1])  # type: ignore[return-value]


def _bits_dict(bits: Sequence[int]) -> dict[str, int]:
    if len(bits) != len(RUBRIC_DIMS) or any(b not in (0, 1) for b in bits):
        raise ValueError(f"rubric bits must be {len(RUBRIC_DIMS)} values in {{0,1}}, got {bits!r}")
    return dict(zip(RUBRIC_DIMS, bits))


def continue_score(bits: Sequence[int]) -> float:
    """The Δ̂ analog in [0,1]: designed weighted 'value of continuing'.
    draft_likely_correct enters as its complement (a correct draft argues STOP)."""
    d = _bits_dict(bits)
    return (
        CONTINUE_WEIGHTS["draft_likely_correct"] * (1 - d["draft_likely_correct"])
        + CONTINUE_WEIGHTS["more_work_likely_helps"] * d["more_work_likely_helps"]
        + CONTINUE_WEIGHTS["last_step_necessary"] * d["last_step_necessary"]
        + CONTINUE_WEIGHTS["budget_healthy"] * d["budget_healthy"]
    )


def state_value(bits: Sequence[int]) -> float:
    """The V̂ analog in [0,1]: designed weighted 'goodness of this state' — the
    rl arm's shaping potential Φ (feeds executor.shaping unchanged)."""
    d = _bits_dict(bits)
    return (
        STATE_VALUE_WEIGHTS["draft_likely_correct"] * d["draft_likely_correct"]
        + STATE_VALUE_WEIGHTS["budget_healthy"] * d["budget_healthy"]
    )


def should_stop(score: float | None, threshold: float) -> bool:
    """Monitor arm's exit rule: stop iff continue_score ≤ θ_p. None (unparseable
    judge output) never stops — fail-open to CONTINUE."""
    if score is None:
        return False
    return score <= threshold


def calibrate_threshold(
    scores: np.ndarray | list[float],
    correct: np.ndarray | list[bool],
    target_precision: float = 0.9,
) -> float:
    """Calibrate θ_p on dev (§17 prompted_rm: 'calibrated on dev').

    Given dev judge calls — continue_score s_i and whether the state's draft was
    actually correct — return the LARGEST threshold whose stop-set precision
    P(correct | s ≤ thr) reaches `target_precision`. Largest-qualifying keeps the
    most cost savings at the required risk level (same matched lost-correct-risk
    logic as b2_probe.calibrate_threshold, with the inequality inverted because
    B10 stops on LOW scores).

    Monotone by construction: the qualifying set shrinks as `target_precision`
    grows, so the returned threshold is non-increasing in it (tested).
    Returns NEVER_STOP (-inf) when no threshold qualifies — B10's monitor then
    degenerates to B1's single point, itself a reportable frontier endpoint.
    """
    s = np.asarray(scores, dtype=float)
    c = np.asarray(correct, dtype=bool)
    if s.shape != c.shape or s.size == 0:
        raise ValueError("scores and correct must be same-length, non-empty")
    for thr in np.unique(s)[::-1]:                    # descending scan
        sel = s <= thr
        if sel.any() and c[sel].mean() >= target_precision:
            return float(thr)
    return NEVER_STOP


# -------------------------------------------------------------------- rl arm
def judge_potentials(
    bits_per_step: Sequence[Sequence[int] | None], fallback: float = 0.0
) -> np.ndarray:
    """Per-step Φ for the rl arm: state_value at each visited state.

    Fail-neutral handling of unparseable steps (None): carry the previous step's
    value forward (first step falls back to `fallback`), so a parse failure
    contributes zero shaped reward rather than a spurious spike."""
    values: list[float] = []
    prev = float(fallback)
    for bits in bits_per_step:
        if bits is not None:
            prev = state_value(bits)
        values.append(prev)
    return np.asarray(values, dtype=float)


def rl_step_rewards(potentials: np.ndarray | Sequence[float], gamma: float = 1.0) -> np.ndarray:
    """r_t = γ·Φ(x_{t+1}) − Φ(x_t) with Φ(terminal) := 0 — IDENTICAL machinery to
    CASSI (§2.4, executor.shaping); only the potential's source differs. The
    telescoping property (Σ r_t = −Φ(x_1) at γ=1) therefore holds here too, so
    step-level advantage assignment stays mandatory for the rl arm as well."""
    return shaped_step_rewards(np.asarray(potentials, dtype=float), gamma=gamma)


# ------------------------------------------------------------------- billing
@dataclass
class JudgeBill:
    """One judge call's cost — added to the method's auxiliary spend (§5.3;
    feeds eval/overhead.py's probe_monitor_usd line)."""
    input_tokens: int
    output_tokens: int
    dollars: float


def bill_judge(input_tokens: int, output_tokens: int, model: str | None = None) -> JudgeBill:
    """Price one judge call under the shared price map — billing symmetry (§5.3):
    B10 pays for its own auxiliary inference at the judge model's rate."""
    return JudgeBill(
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        dollars=token_cost(int(input_tokens), int(output_tokens), model=model),
    )


# ---------------------------------------------------------------- client glue
@runtime_checkable
class JudgeClient(Protocol):
    """Whatever answers a judge prompt with text (production: VLLMJudgeAdapter
    over the lab vLLM server; CPU tests: a scripted fake)."""

    def complete(self, prompt: str) -> str: ...


class VLLMJudgeAdapter:
    """Adapts cassi.executor.vllm_client.VLLMClient (or anything with the same
    .generate(messages, max_tokens) shape) to the JudgeClient protocol."""

    def __init__(self, vllm_client, max_tokens: int = JUDGE_MAX_OUTPUT_TOKENS):
        self._client = vllm_client
        self._max_tokens = int(max_tokens)

    def complete(self, prompt: str) -> str:
        return self._client.generate(
            [{"role": "user", "content": prompt}], max_tokens=self._max_tokens
        )


@dataclass
class JudgeDecision:
    """One monitor-arm decision: parsed bits (None on parse failure ⇒ stop=False),
    the Δ̂-analog score, and the V̂-analog value (carried for the rl arm / diagnostics)."""
    stop: bool
    bits: tuple[int, int, int, int] | None
    score: float | None
    value: float | None
    threshold: float


def judge_decision(client: JudgeClient, serialized_x: str, threshold: float) -> JudgeDecision:
    """One end-to-end B10 monitor call: prompt → judge → parse → score → decide.
    Fail-open: unparseable output ⇒ CONTINUE (stop=False, score=value=None)."""
    bits = parse_rubric(client.complete(build_judge_prompt(serialized_x)))
    if bits is None:
        return JudgeDecision(stop=False, bits=None, score=None, value=None,
                             threshold=threshold)
    s = continue_score(bits)
    return JudgeDecision(stop=should_stop(s, threshold), bits=bits, score=s,
                         value=state_value(bits), threshold=threshold)
