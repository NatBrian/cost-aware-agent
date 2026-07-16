"""x_t serialization — paper_plan_v2 §11 (schema), §18.1 (stopper input template).

Two consumers, one source of truth:
  * `serialize(x, lam, ...)`  → the §18.1 text block the stopper LLM reads
    (identical at training and inference; λ-conditioning lives here, §2.3)
  * `feature_vector(x)`       → numeric vector for the label regressor Ê (Alg.1)

Deliberately absent (§18.1): ground-truth-derived quality, executor-stated
confidence, gold-based stability judgments. Everything here is computable by the
repo harness at inference time.
"""

from __future__ import annotations

import numpy as np

from cassi.common.schema import EMPTY_DRAFT, StepFeatures, TIERS

# Numeric features for the label regressor (order is the contract — do not reorder,
# append only; a trained regressor is invalidated by reordering).
FEATURE_NAMES = [
    "tokens_used", "tokens_pct", "tool_calls", "tool_pct",
    "dollars", "dollars_pct", "burn_rate",
    "tier_idx",                      # HIGH=0 … CRITICAL=3 (ordinal: tighter budget = larger)
    "step_idx", "steps_since_draft_changed",
    "draft_edit_dist_1", "draft_edit_dist_2", "draft_edit_dist_3",
    "retrieval_overlap_last3", "n_distinct_sources",
    "draft_len", "has_draft",
]


def feature_vector(x: StepFeatures) -> np.ndarray:
    d3 = list(x.draft_edit_distance_last3)[-3:]
    d3 = [0.0] * (3 - len(d3)) + [float(v) for v in d3]
    return np.array([
        float(x.tokens_used), float(x.tokens_pct), float(x.tool_calls), float(x.tool_pct),
        float(x.dollars), float(x.dollars_pct), float(x.burn_rate),
        float(TIERS.index(x.tier) if x.tier in TIERS else 3),
        float(x.step_idx), float(x.steps_since_draft_changed),
        d3[0], d3[1], d3[2],
        float(x.retrieval_overlap_last3), float(x.n_distinct_sources),
        float(x.draft_len), float(x.draft != EMPTY_DRAFT),
    ], dtype=float)


def serialize(
    x: StepFeatures, lam: float, *,
    tokens_max: int, tool_calls_max: int, allowance_dollars: float, t_max: int,
) -> str:
    """The §18.1 stopper input block — identical at training and inference."""
    hist_lines = []
    for h in x.history[-3:]:
        hist_lines.append(f"{h.get('t', '?')}: {h.get('action_type', '?')}: {h.get('obs_digest', '')}")
    history = "\n          ".join(hist_lines) if hist_lines else "(none yet)"
    _d = list(x.draft_edit_distance_last3)[-3:]
    d1, d2, d3 = [0.0] * (3 - len(_d)) + [float(v) for v in _d]
    return (
        "<stopper_input>\n"
        f"[TASK] {x.question}\n"
        f"[BUDGET] tokens {x.tokens_used}/{tokens_max} ({x.tokens_pct * 100:.0f}%)"
        f" | tool calls {x.tool_calls}/{tool_calls_max}"
        f" | ${x.dollars:.3f}/${allowance_dollars:.2f}\n"
        f"         | tier {x.tier} | burn ${x.burn_rate:.4f}/step\n"
        f"[OBJECTIVE] cost-sensitivity λ = {lam:g}\n"
        f"[PROGRESS] step {x.step_idx}/{t_max} | draft unchanged for {x.steps_since_draft_changed} steps\n"
        f"           | draft edit-distance (last 3 steps): {d1:.2f},{d2:.2f},{d3:.2f}\n"
        f"           | retrieval overlap (last 3): {x.retrieval_overlap_last3 * 100:.0f}%"
        f" | distinct sources: {x.n_distinct_sources}\n"
        f"[HISTORY] {history}\n"
        f"[DRAFT] {x.draft}\n"
        "</stopper_input>"
    )
