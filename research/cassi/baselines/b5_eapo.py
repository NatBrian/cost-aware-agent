"""B5 — EAPO: published adaptive cost penalty (§5.2 row B5).

WHAT: cost-aware RL with an ADAPTIVE trajectory-level penalty — the cost pressure
scales with how solvable the task currently is (group solve rate), so easy tasks
get squeezed and hard tasks keep their budget. This replaces v5's homemade
"adaptive-alpha" with the published adaptive-penalty family (§14 changelog).

REIMPLEMENTS: EAPO (2606.02132) — PRIMARY per §5.2's decision rule; the same rule
names agentic-ALP (ALP, 2506.05256, NeurIPS'25 spotlight) as the FALLBACK if EAPO
proves unreproducible. Implemented form: the solve-rate-scaled penalty that
defines this family (§1.3: "solve-rate-scaled penalties: ALP, DAST, LASER-D"),
    reward = task_reward − w · solve_rate(group) · normalized_cost,
with solve_rate estimated from the GRPO group's own rollouts (free — no extra
inference). HONESTY NOTE: if reproduction shows EAPO's exact published functional
form differs, this module as written IS the documented agentic-ALP fallback and
the paper reports it as such — either way the row tests the same question.

KILLS THE QUESTION: "is a learned VALUE better than adaptive scalar pressure" —
the strongest published no-stopper cost adaptivity.

COST KNOB (§5.3): the penalty weight w. Monotone: for positive cost and positive
solve rate, reward strictly decreases in w (tested); w=0 recovers plain reward.

TRAINING: needs_training=True — trajectory-level GRPO. Training arm launched via
scripts/p8_baselines.sh with reward_fn=cassi.baselines.b5_eapo.reward.
"""

from __future__ import annotations

import numpy as np

from cassi.common.schema import Trajectory

COST_KNOB = "penalty_weight"

SUCCESS_THRESHOLD = 0.5


def solve_rate(task_rewards: list[float] | np.ndarray,
               success_threshold: float = SUCCESS_THRESHOLD) -> float:
    """Empirical solve rate of one GRPO group — the adaptivity signal: penalty
    pressure rises on tasks the policy already solves, vanishes on hopeless ones."""
    r = np.asarray(task_rewards, dtype=float)
    if r.size == 0:
        raise ValueError("empty group")
    return float((r >= success_threshold).mean())


def normalized_cost(total_dollars: float, median_pilot_spend: float) -> float:
    """C̃ = total spend / median pilot spend (§2.1) — same normalization as the
    CASSI economy so w is commensurable with λ across rows."""
    if median_pilot_spend <= 0:
        raise ValueError("median_pilot_spend must be positive (frozen from P2 pilot)")
    return float(total_dollars) / float(median_pilot_spend)


def trajectory_cost(traj: Trajectory, median_pilot_spend: float) -> float:
    """Normalized total cost of one trajectory."""
    return normalized_cost(sum(s.c for s in traj.steps), median_pilot_spend)


def reward(task_reward: float, norm_cost: float, group_solve_rate: float, w: float) -> float:
    """EAPO reward for one rollout: task_reward − w·solve_rate·normalized_cost.
    This is the reward_fn the GRPO training arm optimizes (trajectory-level)."""
    return float(task_reward) - float(w) * float(group_solve_rate) * float(norm_cost)


def group_rewards(task_rewards: list[float] | np.ndarray,
                  norm_costs: list[float] | np.ndarray,
                  w: float, success_threshold: float = SUCCESS_THRESHOLD) -> np.ndarray:
    """EAPO rewards for one GRPO group: one shared solve rate, per-rollout costs."""
    r = np.asarray(task_rewards, dtype=float)
    c = np.asarray(norm_costs, dtype=float)
    if r.shape != c.shape:
        raise ValueError("task_rewards and norm_costs must be same-length")
    sr = solve_rate(r, success_threshold)
    return r - float(w) * sr * c
