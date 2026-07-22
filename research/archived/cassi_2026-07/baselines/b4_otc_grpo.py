"""B4 — OTC-GRPO: outcome-level tool-count reward shaping (§5.2 row B4).

WHAT: cost-aware RL where the TRAJECTORY-level reward is the task reward scaled by
a tool-call-count factor — cost pressure exists, but only as an outcome-level
scalar; no state-dependent stopping value, no step-level economics.

REIMPLEMENTS: OTC-PO / OTC-GRPO (2504.14870), on the same Search-R1 env family the
paper used (§5.1 keeps the corpus commensurable on purpose).

REWARD SHAPE — which form and why (task spec requires stating this): the paper
gives a cos-style tool penalty and a simpler tool-productivity RATIO built from
the group-minimal tool count m*. We implement the RATIO form,
    reward = task_reward * ((m* + 1) / (m + 1)) ** alpha,
because (a) the cos form's phase needs a per-task maximum tool budget M that our
envs do not fix, while m* — the fewest tool calls among the group's CORRECT
rollouts — is exactly what a GRPO group already provides (the paper's own
OTC-GRPO construction anchors on the group minimum); (b) the +1 smoothing keeps
m*=0 (zero-tool solutions) well-defined. alpha=1 is the paper-style operating
point; alpha generalizes it into the §5.3 frontier knob.

KILLS THE QUESTION: "is step-level economic signal needed at all" — if B4 matches
CASSI, outcome-level tool-count pressure suffices and the bridge is unnecessary.

COST KNOB (§5.3): alpha, the tool-count coefficient. Monotone: for m > m*, the
reward strictly decreases in alpha (tested); alpha=0 recovers plain task reward.

TRAINING: needs_training=True — trajectory-level GRPO (NOT CASSI's step-level
machinery: outcome-level is the point of this row). Training arm launched via
scripts/p8_baselines.sh with reward_fn=cassi.baselines.b4_otc_grpo.reward.
"""

from __future__ import annotations

import numpy as np

from cassi.common.schema import Trajectory

COST_KNOB = "tool_count_coefficient"

SUCCESS_THRESHOLD = 0.5      # task_reward >= this counts as "correct" for m* (EM/success are {0,1})


def tool_count(traj: Trajectory) -> int:
    """Number of tool calls m in a trajectory."""
    return sum(1 for s in traj.steps if s.a == "tool_call")


def otc_scale(m: int, m_star: int, alpha: float) -> float:
    """The tool-count factor ((m*+1)/(m+1))**alpha, capped at 1.0 so rollouts using
    FEWER calls than the group's correct minimum are never paid a bonus above their
    task reward (the paper rewards optimality, not starvation)."""
    if m < 0 or m_star < 0:
        raise ValueError("tool counts must be non-negative")
    return float(min(1.0, ((m_star + 1) / (m + 1)) ** float(alpha)))


def reward(task_reward: float, m: int, m_star: int, alpha: float) -> float:
    """OTC reward for one rollout: task_reward scaled by the tool-count factor.
    This is the reward_fn the GRPO training arm optimizes (trajectory-level)."""
    return float(task_reward) * otc_scale(m, m_star, alpha)


def group_m_star(task_rewards: list[float] | np.ndarray, tool_counts: list[int] | np.ndarray,
                 success_threshold: float = SUCCESS_THRESHOLD) -> int:
    """m* for one GRPO group: the minimum tool count among CORRECT rollouts
    (the group's empirical proxy for the optimal call count, per the paper);
    falls back to the group minimum when no rollout is correct."""
    r = np.asarray(task_rewards, dtype=float)
    m = np.asarray(tool_counts, dtype=int)
    if r.shape != m.shape or r.size == 0:
        raise ValueError("task_rewards and tool_counts must be same-length, non-empty")
    correct = r >= success_threshold
    return int(m[correct].min()) if correct.any() else int(m.min())


def group_rewards(task_rewards: list[float] | np.ndarray, tool_counts: list[int] | np.ndarray,
                  alpha: float, success_threshold: float = SUCCESS_THRESHOLD) -> np.ndarray:
    """OTC-GRPO rewards for one group: compute m* from the group, scale each rollout."""
    m_star = group_m_star(task_rewards, tool_counts, success_threshold)
    return np.array([
        reward(r, int(m), m_star, alpha)
        for r, m in zip(np.asarray(task_rewards, dtype=float), np.asarray(tool_counts, dtype=int))
    ])
