"""B7 — CaRT + cost: trained self-termination via oracle-truncated SFT (§5.2 row B7).

WHAT: teach the executor to STOP by imitation — build SFT data by truncating
collected forced-continuation trajectories at the Snell-optimal stop τ* and
appending ANSWER (the running draft at τ* becomes the final answer), then
fine-tune the executor to reproduce these oracle-truncated trajectories.

REIMPLEMENTS: CaRT (2510.08517) — trained stop/continue for agents — with the
cost delta §5.2 specifies: CaRT's original targets are γ-discount-derived; ours
come from the λ·cost Snell labels (cassi.labels.snell), which is what makes this
row "CaRT + cost".

TWO ARMS (§5.2, mandatory):
  arm 1 "sft_only"      — imitate the truncated data, NO RL. Answers "is RL even
                          needed, or does imitation suffice?"
  arm 2 "sft_plus_grpo" — the SFT checkpoint further trained with trajectory-level
                          GRPO on the shared economy (reward below).
Both arms launched via scripts/p8_baselines.sh: the SFT arm consumes the JSONL
written from build_cart_sft_dataset; the GRPO arm additionally passes
reward_fn=cassi.baselines.b7_cart_cost.reward.

KILLS THE QUESTION: whether trained self-termination needs RL at all — v5's
primary comparison, retained.

COST KNOB (§5.3): the λ of the truncation labels — regenerating the SFT set from
a different λ's LabelSet moves τ* earlier/later, tracing B7's frontier (each
frontier point is a separate SFT set; §16 P8 budget covers 2-3 points).

CPU PART (this module): the data transformation. Training itself is GPU work.
"""

from __future__ import annotations

from dataclasses import replace

from cassi.budget.cost import base_reward
from cassi.common.schema import EMPTY_DRAFT, Step, Trajectory
from cassi.labels.snell import LabelSet

COST_KNOB = "label_lambda"

ARMS = ("sft_only", "sft_plus_grpo")

TRUNCATION_MARKER = "ANSWER (oracle-truncated at tau*, CaRT+cost SFT)"


def truncate_at_tau_star(traj: Trajectory, tau_star: int) -> Trajectory:
    """Truncate one forced-continuation trajectory at τ* and append ANSWER.

    Steps 1..τ* are kept; step τ*'s action becomes the ANSWER emission, with the
    running draft at τ* as the final answer (the §2.6 draft is exactly "the answer
    the agent would give if stopped here"). Costs of the kept steps are unchanged —
    the imitation target is "spend this much, then answer"."""
    if not 1 <= tau_star <= len(traj):
        raise ValueError(f"tau_star={tau_star} outside trajectory length {len(traj)}")
    steps = [Step.from_dict(s.to_dict()) for s in traj.steps[:tau_star]]  # deep copy
    last = steps[-1]
    steps[-1] = replace(last, a="answer", o=TRUNCATION_MARKER, answered_flag=True)
    outcome = {
        **traj.outcome,
        "tau": tau_star,
        "Q_tau": last.q,
        "final_answer": last.draft,
        "collection_mode": "cart_sft_oracle_truncated",
    }
    return Trajectory(
        task_id=traj.task_id, domain=traj.domain,
        allowance_B=traj.allowance_B, wallet_size=traj.wallet_size,
        group_id=traj.group_id, rollout_idx=traj.rollout_idx,
        steps=steps, outcome=outcome,
    )


def build_cart_sft_dataset(
    trajectories: list[Trajectory],
    labelset: LabelSet,
    *,
    drop_empty_draft: bool = True,
) -> list[Trajectory]:
    """Snell τ* (LabelSet.tau_star) → oracle-truncated SFT trajectories.

    Trajectories without a label entry are skipped; with drop_empty_draft=True
    (default) so are those whose draft at τ* is EMPTY_DRAFT — imitating "answer
    with nothing" would teach degenerate stops. The λ baked into `labelset` is
    this row's frontier knob."""
    out: list[Trajectory] = []
    for traj in trajectories:
        tau = labelset.tau_star.get((traj.task_id, traj.rollout_idx))
        if tau is None or not 1 <= tau <= len(traj):
            continue
        if drop_empty_draft and traj.steps[tau - 1].draft == EMPTY_DRAFT:
            continue
        out.append(truncate_at_tau_star(traj, int(tau)))
    return out


def reward(
    terminal_quality: float,
    costs_to_tau: list[float],
    tiers_to_tau: list[str],
    lam: float,
    median_pilot_spend: float,
) -> float:
    """The +GRPO arm's trajectory-level reward — the shared economy's R_base
    (§2.4), same λ as the truncation labels so both arms optimize one Lagrangian."""
    return base_reward(terminal_quality, costs_to_tau, tiers_to_tau, lam, median_pilot_spend)
