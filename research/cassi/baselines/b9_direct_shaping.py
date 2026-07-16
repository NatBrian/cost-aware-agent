"""B9 — DASH-style direct shaping, NO stopper: THE pivotal baseline (§5.2 row B9, §2.4, H3).

WHAT: CASSI with the stopper deleted and nothing else changed. The Snell labels'
V* (LabelSet.v_star) is used DIRECTLY as the shaping potential at LABELED states,
fed through CASSI's EXACT step-level machinery — the same shaped_step_rewards and
step_level_group_advantages calls from cassi.executor.shaping, same terminal
R_base economy, same GRPO hygiene. The only difference in the entire pipeline is
Φ's source: labels instead of the trained value model V̂_θ.

REIMPLEMENTS: DASH (2607.00482), adapted per §5.2 — post-hoc labels used as
advantages directly, here with our Snell/cost labels. §5.2's fairness mandate is
honored: trajectory-level shaping is provably inert under the §2.4 telescoping
(shaped terms sum to −Φ(x_0), constant within a group), so B9 MUST use the
step-level machinery — anything less would be a strawman. §2.4 machinery is
imported, never reimplemented.

WHAT IT LACKS (by design — this is the H3 measurement): the stopper's regression
generalizes V̂ to UNLABELED states. Labels exist only for states visited by the
COLLECTION policy; fresh RL rollouts leave that support immediately. Here,
unlabeled states fall back to Φ = `unlabeled_value` (default 0.0 — the absorbing-
terminal convention value, i.e. "no information"), producing zero/garbage shaped
rewards exactly where CASSI's stopper still supplies signal. The designed
equivalence (tested): when the stopper's V̂ equals v_star at labeled states and
all states are labeled, B9's advantage arrays are IDENTICAL to CASSI's — so any
E1 gap between them is attributable to generalization + runtime control, nothing
else.

KILLS THE QUESTION: "does the stopper earn its existence" (H3; kill-switch K1's
third arm). If B9 ties CASSI, the stopper demotes to optional and the paper
becomes "Snell-label direct shaping for agents".

COST KNOB (§5.3): λ (of the label set, matching CASSI's economy).

TRAINING: needs_training=True — step-level GRPO. Training arm launched via
scripts/p8_baselines.sh with reward_fn=cassi.baselines.b9_direct_shaping.reward
(terminal economy) wired through direct_shaping_group_advantages (step credit).
"""

from __future__ import annotations

import numpy as np

from cassi.budget.cost import base_reward
from cassi.common.schema import Trajectory
from cassi.executor.shaping import (
    MIN_COHORT,
    GroupAdvantages,
    shaped_step_rewards,
    step_level_group_advantages,
)
from cassi.labels.snell import LabelSet

COST_KNOB = "lambda"

DEFAULT_UNLABELED_VALUE = 0.0    # Φ at states the labels never saw — the gap H3 measures


def label_potential_lookup(labelset: LabelSet) -> dict[tuple[str, int, int], float]:
    """(task_id, rollout_idx, t) → V*_t from the label set — B9's entire 'value
    model'. No regression, no generalization: a lookup table."""
    return {(l.task_id, l.rollout_idx, l.t): l.v_star for l in labelset.labels}


def potentials_for_trajectory(
    labelset: LabelSet | dict[tuple[str, int, int], float],
    traj: Trajectory,
    *,
    unlabeled_value: float = DEFAULT_UNLABELED_VALUE,
) -> np.ndarray:
    """Φ(x_t) for every step of `traj`: v_star where a label exists, else
    `unlabeled_value`. On fresh RL rollouts most states are unlabeled — that
    hard cliff is precisely what the stopper's regression removes (H3)."""
    lookup = labelset if isinstance(labelset, dict) else label_potential_lookup(labelset)
    return np.array([
        lookup.get((traj.task_id, traj.rollout_idx, t), unlabeled_value)
        for t in range(1, len(traj) + 1)
    ], dtype=float)


def label_coverage(labelset: LabelSet | dict, trajs: list[Trajectory]) -> float:
    """Fraction of the group's states that have a label — the honesty diagnostic
    reported alongside B9 results (coverage collapses as RL moves off-support)."""
    lookup = labelset if isinstance(labelset, dict) else label_potential_lookup(labelset)
    total = sum(len(tr) for tr in trajs)
    if total == 0:
        return 0.0
    hit = sum(
        1 for tr in trajs for t in range(1, len(tr) + 1)
        if (tr.task_id, tr.rollout_idx, t) in lookup
    )
    return hit / total


def direct_shaping_group_advantages(
    trajs: list[Trajectory],
    labelset: LabelSet | dict[tuple[str, int, int], float],
    terminal_rewards: list[float],
    *,
    gamma: float = 1.0,
    unlabeled_value: float = DEFAULT_UNLABELED_VALUE,
    min_cohort: int = MIN_COHORT,
) -> GroupAdvantages:
    """B9's step-level advantages for one GRPO group — IDENTICAL calls to CASSI's
    Algorithm 3 pipeline (shaped_step_rewards → step_level_group_advantages, both
    imported from cassi.executor.shaping), with Φ read from labels instead of the
    stopper. When V̂ ≡ v_star at labeled states and coverage is 1.0, the output
    arrays equal CASSI's exactly (the designed equivalence; tested)."""
    potentials = [
        potentials_for_trajectory(labelset, tr, unlabeled_value=unlabeled_value)
        for tr in trajs
    ]
    shaped = [shaped_step_rewards(v, gamma) for v in potentials]
    return step_level_group_advantages(shaped, terminal_rewards, min_cohort=min_cohort)


def reward(
    terminal_quality: float,
    costs_to_tau: list[float],
    tiers_to_tau: list[str],
    lam: float,
    median_pilot_spend: float,
) -> float:
    """B9's terminal reward — the shared economy's R_base (§2.4), same λ as the
    labels; identical to CASSI's terminal term (the fair fight is exact)."""
    return base_reward(terminal_quality, costs_to_tau, tiers_to_tau, lam, median_pilot_spend)
