"""B6 — single-model GRPO with cost-in-reward, CTA-style (§5.2 row B6).

WHAT: the simplest trained cost-aware arm — ONE model, plain GRPO, and the whole
economy collapsed into a flat trajectory-level scalar:
    R = Q_tau − λ · C̃_tau        (quality minus λ times pilot-normalized cost)
No stopper, no step-level credit, no tier scaling, no adaptivity.

REIMPLEMENTS: the CTA-style cost-discounted GRPO of 2602.16699 — the paper that
SHOWED this recipe under-internalizes (§4 motivation-empirics row); B6 exists to
reproduce that finding against CASSI on our envs.

FORM CHOICE (documented): the published CTA form is a FLAT λ on cost, so the
default here uses m ≡ 1 (rule_table_off=True in the shared economy) rather than
CASSI's tier-scaled multipliers — using CASSI's tier economy would quietly hand B6
part of the method. A `tier_scaled=True` variant is exposed for the fairness
check (same-economy comparison) and reported in the appendix if run.

KILLS THE QUESTION: "two-model necessity" — paired with ablation A2's matched-
parameter comparison (§2.3 point 3, H4): if B6 (plus A2's multi-task variant)
matches CASSI on Pareto, separation isn't buying performance.

COST KNOB (§5.3): λ. Monotone: for positive cost, reward strictly decreases in λ
(tested); λ=0 recovers plain task reward.

TRAINING: needs_training=True — trajectory-level GRPO on the single 9B model.
Training arm launched via scripts/p8_baselines.sh with
reward_fn=cassi.baselines.b6_single_model_cost.reward.
"""

from __future__ import annotations

from cassi.budget.cost import base_reward
from cassi.common.schema import Trajectory

COST_KNOB = "lambda"


def reward(
    terminal_quality: float,
    costs_to_tau: list[float],
    tiers_to_tau: list[str],
    lam: float,
    median_pilot_spend: float,
    *,
    tier_scaled: bool = False,
) -> float:
    """R = Q_tau − λ·C̃_tau (flat λ, CTA-style; m ≡ 1 unless tier_scaled=True).

    Reuses the shared economy (cassi.budget.cost.base_reward) so the ONLY
    difference from CASSI's R_base is the missing tier multipliers by default —
    and, of course, the absence of any shaped step rewards on top."""
    return base_reward(
        terminal_quality, costs_to_tau, tiers_to_tau, lam, median_pilot_spend,
        rule_table_off=not tier_scaled,
    )


def trajectory_reward(traj: Trajectory, lam: float, median_pilot_spend: float,
                      *, tier_scaled: bool = False) -> float:
    """Convenience wrapper over a Trajectory (Q_tau from outcome, costs from steps)."""
    q = float(traj.outcome.get("Q_tau", traj.steps[-1].q if traj.steps else 0.0))
    return reward(
        q, [s.c for s in traj.steps], [s.tier for s in traj.steps],
        lam, median_pilot_spend, tier_scaled=tier_scaled,
    )
