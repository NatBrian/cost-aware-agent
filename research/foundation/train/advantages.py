"""Step-level GRPO advantages (F5): group-normalized returns-to-go with the
min-cohort guard.

For a group of G trajectories (same task, same wallet), step-position t's
cohort is every trajectory still alive at t. Advantage of step t in traj i:
  A_t^i = (R_t^i - mean_cohort(t)) / (std_cohort(t) + eps)
If the cohort is smaller than min_cohort, those steps fall back to the
TRAJECTORY-level baseline (total return z-scored across the full group) —
comparing 2 siblings' late steps is noise, not signal (v2.1 §2.4 guard).
"""

import numpy as np

EPS = 1e-6


def trajectory_returns(rtgs: list[list[float]]) -> np.ndarray:
    """Total return per trajectory = R_1 (returns-to-go at the first step)."""
    return np.array([r[0] for r in rtgs], dtype=float)


def group_step_advantages(rtgs: list[list[float]],
                          min_cohort: int = 3) -> list[list[float]]:
    """rtgs: per-trajectory returns-to-go lists (variable length). Returns
    advantages with identical shapes."""
    if not rtgs:
        return []
    totals = trajectory_returns(rtgs)
    traj_adv = (totals - totals.mean()) / (totals.std() + EPS)
    max_len = max(len(r) for r in rtgs)
    adv: list[list[float]] = [[0.0] * len(r) for r in rtgs]
    for t in range(max_len):
        alive = [i for i, r in enumerate(rtgs) if len(r) > t]
        if len(alive) >= min_cohort:
            vals = np.array([rtgs[i][t] for i in alive])
            mu, sd = vals.mean(), vals.std()
            for i in alive:
                adv[i][t] = float((rtgs[i][t] - mu) / (sd + EPS))
        else:                       # guard: trajectory-level baseline
            for i in alive:
                adv[i][t] = float(traj_adv[i])
    return adv
