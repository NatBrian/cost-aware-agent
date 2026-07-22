"""Potential-based economic shaping + step-level advantages — paper_plan_v2 §2.4, §10 Alg.3.

    Φ(x_t) = V̂_θ(x_t);   Φ(absorbing terminal) := 0     (the invariance convention)
    r_t    = γ·Φ(x_{t+1}) − Φ(x_t)                        (γ = 1)
    R_executor = Σ_t r_t + R_base (+ format term)

Consequence of the convention (§2.4): with γ=1, Φ(terminal)=0 the shaped terms
telescope to −Φ(x_0) — CONSTANT within a GRPO group (same task/start/wallet).
Trajectory-level group advantages are therefore provably unaffected by shaping;
ALL of its effect arrives through STEP-LEVEL advantage assignment, which is
mandatory (per-step returns-to-go with group normalization here; the SHAPE-segment
variant is the K1 alternative). `tests/test_core.py` asserts the telescoping.

Pure functions on arrays — no torch, no verl — so the exact semantics are
CPU-testable and `train_grpo.py` only has to wire them into verl's reward hooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_COHORT = 3   # §17 executor.grpo.min_cohort_guard


def shaped_step_rewards(v_hat: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """r_t = γ·Φ(x_{t+1}) − Φ(x_t) for t = 1..T, with Φ(terminal) := 0.

    `v_hat` = V̂_θ(x_1..x_T), the stopper's value head on each visited state.
    Returns T rewards; the T-th uses the absorbing-terminal convention."""
    v = np.asarray(v_hat, dtype=float)
    nxt = np.append(v[1:], 0.0)          # Φ(x_{T+1}) := 0 — the invariance convention
    return gamma * nxt - v


def returns_to_go(step_rewards: np.ndarray, terminal_reward: float) -> np.ndarray:
    """R_t = Σ_{t'≥t} r_{t'} + R_base  (Alg.3). R_base (and format term) is paid at
    the end, so it appears in every step's return-to-go."""
    r = np.asarray(step_rewards, dtype=float)
    return np.cumsum(r[::-1])[::-1] + terminal_reward


@dataclass
class GroupAdvantages:
    """Per-trajectory, per-step advantages for one GRPO group, plus bookkeeping."""
    advantages: list[np.ndarray]        # one array per trajectory (its own length)
    cohort_sizes: np.ndarray            # alive-count per step index (1-based index-1)
    guarded_steps: int                  # steps that fell back to the trajectory baseline


def step_level_group_advantages(
    step_rewards: list[np.ndarray],
    terminal_rewards: list[float],
    *, min_cohort: int = MIN_COHORT, eps: float = 1e-8,
) -> GroupAdvantages:
    """Per-step returns-to-go with group normalization (Alg.3 'per_step_rtg' variant).

    For each step index t, the cohort = trajectories still alive at t. Advantage of
    trajectory i at step t = (R_t^i − mean_cohort(R_t)) / (std_cohort(R_t) + eps).

    Min-cohort guard (§2.4/§17): if fewer than `min_cohort` group members are alive
    at t, those steps fall back to the TRAJECTORY-level baseline (whole-group
    normalized total return) — for those steps ONLY. Whole-trajectory-level shaping
    is provably inert (telescoping) and is never a fallback.
    """
    G = len(step_rewards)
    if G != len(terminal_rewards):
        raise ValueError("step_rewards and terminal_rewards must be same length")
    rtg = [returns_to_go(step_rewards[i], terminal_rewards[i]) for i in range(G)]
    lengths = np.array([len(r) for r in rtg])
    T_max = int(lengths.max()) if G else 0

    # trajectory-level fallback baseline: group-normalized TOTAL return
    totals = np.array([float(step_rewards[i].sum()) + terminal_rewards[i] for i in range(G)])
    traj_adv = (totals - totals.mean()) / (totals.std() + eps)

    advantages = [np.zeros(int(l)) for l in lengths]
    cohort_sizes = np.zeros(T_max, dtype=int)
    guarded = 0
    for t in range(T_max):
        alive = np.where(lengths > t)[0]
        cohort_sizes[t] = len(alive)
        if len(alive) >= min_cohort:
            vals = np.array([rtg[i][t] for i in alive])
            mu, sd = vals.mean(), vals.std()
            for k, i in enumerate(alive):
                advantages[i][t] = (vals[k] - mu) / (sd + eps)
        else:
            guarded += len(alive)
            for i in alive:
                advantages[i][t] = traj_adv[i]
    return GroupAdvantages(advantages=advantages, cohort_sizes=cohort_sizes, guarded_steps=guarded)


def trajectory_level_group_advantages(
    step_rewards: list[np.ndarray], terminal_rewards: list[float], *, eps: float = 1e-8,
) -> np.ndarray:
    """Plain GRPO trajectory advantages — exists ONLY to demonstrate the telescoping
    inertness in tests/diagnostics (§2.4). Never a production fallback."""
    totals = np.array([float(r.sum()) + terminal_rewards[i] for i, r in enumerate(step_rewards)])
    return (totals - totals.mean()) / (totals.std() + eps)


def hacking_divergence(v_hat_terminalward: np.ndarray, realized_rewards: np.ndarray) -> float:
    """V̂-vs-realized-reward divergence diagnostic (§2.4, feeds F6): mean |V̂(x_τ) −
    realized R_base| over a batch. A rising curve triggers a stopper refresh."""
    return float(np.mean(np.abs(np.asarray(v_hat_terminalward) - np.asarray(realized_rewards))))
