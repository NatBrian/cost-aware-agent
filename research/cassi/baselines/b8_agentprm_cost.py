"""B8 — AgentPRM-cost, honestly implemented (§5.2 row B8).

WHAT: a generic value PRM whose training targets are POOLED return-to-go under a
cost-inclusive reward — the closest published dense-credit recipe, plus our cost
term. It has value semantics ("how good is this state") but NOT stopping-margin
semantics (no Cont−U comparison, no Snell recursion, no STOP/CONTINUE head).

REIMPLEMENTS: AgentPRM (2502.10325). HONESTY REQUIREMENT (§5.2, §4 corrections):
AgentPRM's targets are POOLED return-to-go — v5 misread it as per-state Monte
Carlo, and this module must not repeat that error. "Pooled" here means every
visited state contributes its OWN observed return-to-go and the pairs
{(x_t, RTG_t)} are pooled across all trajectories into ONE regression set; the
value model's generalization does the averaging. There is NO per-state resampling
of continuations. (The optional TD+GAE variant per 2511.08325 is a training-time
choice in the GPU arm, not implemented here.)

COST TERM: the return-to-go is computed under the cost-inclusive reward stream —
per-step −λ·m(tier)·c̃ plus terminal Q_tau — i.e. the same economy as CASSI's
labels, so the comparison isolates SEMANTICS (value vs stopping margin), not the
economy.

KILLS THE QUESTION: "is the STOPPING-VALUE semantics what matters vs generic
value+cost" — if B8 matches CASSI, the Snell machinery is decoration.

COST KNOB (§5.3): its λ (in the RTG targets).

TRAINING (needs_training=True), two stages, documented:
  1. Value model V_ψ: pooled regression on the PRMTarget pairs below —
     conceptually reuses the stopper training machinery (cassi/stopper: same §11
     x_t serialization, same TRL scalar-head SFT wiring), but trains ONE value
     head only (no action/Δ heads — B8 has no stopping margin by construction).
  2. Executor GRPO with V_ψ(x_t) used AS the per-step process reward (AgentPRM's
     usage — PRM score as reward, NOT a potential difference; the PBRS bridge is
     CASSI's contribution, not this row's), plus the cost-inclusive outcome reward.
Both launched via scripts/p8_baselines.sh with
reward_fn=cassi.baselines.b8_agentprm_cost.reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cassi.budget.cost import base_reward, scaled_step_costs
from cassi.common.schema import StepFeatures, Trajectory

COST_KNOB = "lambda"


@dataclass
class PRMTarget:
    """One pooled-regression training pair: state features → cost-inclusive RTG."""
    task_id: str
    group_id: str
    rollout_idx: int
    t: int                    # 1-based step index
    x: StepFeatures
    target: float             # RTG_t = Q_tau − Σ_{i≥t} λ·m(tier_i)·c̃_i
    lam: float


def cost_inclusive_returns_to_go(
    traj: Trajectory, lam: float, median_pilot_spend: float,
    *, rule_table_off: bool = False,
) -> np.ndarray:
    """RTG_t = Q_tau − Σ_{i≥t} λ·m(tier_i)·c̃_i for every step t.

    The observed return-to-go of the cost-inclusive reward stream (per-step cost
    penalties, terminal quality) — one number per visited state, no resampling.
    RTG_1 equals the trajectory's R_base by construction (tested)."""
    if not traj.steps:
        raise ValueError("empty trajectory")
    penal = scaled_step_costs(
        [s.c for s in traj.steps], [s.tier for s in traj.steps],
        lam, median_pilot_spend, rule_table_off=rule_table_off,
    )
    q_tau = float(traj.outcome.get("Q_tau", traj.steps[-1].q))
    tail = np.cumsum(penal[::-1])[::-1]        # Σ_{i≥t} penal_i
    return q_tau - tail


def pooled_rtg_targets(
    trajectories: list[Trajectory], lam: float, median_pilot_spend: float,
    *, rule_table_off: bool = False,
) -> list[PRMTarget]:
    """The pooled training set for V_ψ: EVERY visited state of EVERY trajectory
    contributes (x_t, RTG_t); pairs are pooled into one regression set (AgentPRM's
    pooled return-to-go — NOT per-state MC, see module docstring)."""
    out: list[PRMTarget] = []
    for traj in trajectories:
        rtg = cost_inclusive_returns_to_go(traj, lam, median_pilot_spend,
                                           rule_table_off=rule_table_off)
        for i, step in enumerate(traj.steps):
            out.append(PRMTarget(
                task_id=traj.task_id, group_id=traj.group_id,
                rollout_idx=traj.rollout_idx, t=i + 1,
                x=step.x, target=float(rtg[i]), lam=lam,
            ))
    return out


def prm_step_rewards(v_psi: np.ndarray | list[float]) -> np.ndarray:
    """AgentPRM's usage of the trained PRM at executor-RL time: the PRM score
    V_ψ(x_t) IS the per-step process reward (no potential difference — that PBRS
    bridge is CASSI's §2.4 contribution and deliberately absent here)."""
    return np.asarray(v_psi, dtype=float)


def reward(
    terminal_quality: float,
    costs_to_tau: list[float],
    tiers_to_tau: list[str],
    lam: float,
    median_pilot_spend: float,
) -> float:
    """The outcome component of B8's executor RL — the shared economy's R_base
    (same λ as the RTG targets); V_ψ step rewards are added by the GRPO wiring."""
    return base_reward(terminal_quality, costs_to_tau, tiers_to_tau, lam, median_pilot_spend)
