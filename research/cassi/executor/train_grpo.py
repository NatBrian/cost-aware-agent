"""Executor GRPO with economic shaping — paper_plan_v2 §2.4, §10 Algorithm 3,
§16 P6 (verl integration), §17 `executor` config, §19 stack.

Two layers, deliberately separated:

(a) PURE reward semantics (`compute_cassi_rewards`) — numpy only, CPU-testable,
    no verl/torch import. Maps per-step V̂ arrays + terminal outcomes to:
      r_t   = γ·V̂(x_{t+1}) − V̂(x_t), Φ(terminal):=0        (shaped step rewards)
      R_term = R_base + γ_fmt·format                          (same economy as labels)
      → per-step advantages via cassi.executor.shaping (step-level variant from
        config; min-cohort guard) — the telescoping property makes step-level
        assignment MANDATORY (§2.4: trajectory-level advantages are provably
        unaffected by the shaping).

(b) `main()` — builds a verl PPO/GRPO trainer config (Dr.GRPO length norm,
    KL β=0.04, G=8, temps per §17) and registers the custom reward/advantage
    through `VerlCassiAdapter`. verl imports are LAZY with actionable startup
    errors; the adapter carries a TODO(P6) block documenting exactly what to
    wire, because verl's custom-advantage API shifts between versions — the
    pure functions in (a) carry the semantics either way.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np

from cassi.budget.cost import base_reward
from cassi.common.config import load_config
from cassi.common.schema import Trajectory
from cassi.executor.shaping import (
    GroupAdvantages,
    shaped_step_rewards,
    step_level_group_advantages,
)

_VERL_HELP = (
    "verl is not installed: {err}\n"
    "GRPO training needs the GPU stack (paper_plan_v2 §16 P0 / §19 manifest):\n"
    "  pip install -r research/cassi/requirements-gpu.txt\n"
    "  bash scripts/p0_setup.sh     # clones + pins verl >= v0.8.0, verl-tool, verl-agent\n"
)
_CUDA_HELP = (
    "No CUDA device visible. On this machine, acquire GPUs first (never kill "
    "occupier processes):\n"
    "  eval $(/mnt/src/zhanka/gpu_acquire.sh 4)    # 4-8 GPUs for executor GRPO\n"
    "and release them when done: /mnt/src/zhanka/gpu_release.sh"
)


# ======================================================================== (a)
@dataclass
class CassiRewards:
    """Output of `compute_cassi_rewards` for ONE GRPO group (G trajectories of
    the same task under the same wallet, §2.2)."""

    step_rewards: list[np.ndarray]        # r_t per trajectory (shaped, Alg.3)
    base_rewards: list[float]             # R_base per trajectory (§2.4)
    terminal_rewards: list[float]         # R_base + γ_fmt·format (paid at the end)
    advantages: list[np.ndarray]          # per-step advantages (step-level, mandatory)
    cohort_sizes: np.ndarray              # alive-count per step index
    guarded_steps: int                    # steps that hit the min-cohort guard
    telescoped_constants: list[float]     # Σ_t r_t = −Φ(x_1) per traj (§2.4 diagnostic)
    v_hat_terminal: list[float] = field(default_factory=list)  # V̂ at last state (F6 divergence)


def compute_cassi_rewards(
    trajectory_batch: list[Trajectory],
    stopper_values: list[np.ndarray],
    cfg: dict,
    *,
    median_pilot_spend: float | None = None,
    rule_table_off: bool = False,
) -> CassiRewards:
    """Algorithm 3 reward semantics for one GRPO group — PURE, CPU-testable.

    Args:
      trajectory_batch: the G rollouts of one group (RL-mode trajectories:
        terminate at ANSWER or T_max, §2.1). Each needs outcome['Q_tau'] filled
        (terminal quality in the labels' measure — training-time GT is allowed,
        §2.1) and outcome['format_score'] from the scaffold.
      stopper_values: per-trajectory V̂_θ(x_1..x_T) arrays — the stopper's value
        head on each VISITED state (frozen within the iteration, refreshed per
        §2.7). Φ(absorbing terminal) := 0 is applied here, not by the caller.
      cfg: configs/cassi.yaml dict (§17). Uses executor.training_lambda,
        executor.shaping.{gamma,format_weight}, executor.grpo.{min_cohort_guard,
        step_level_variant}, label.cost_normalization.
      median_pilot_spend: override for the C̃ normalization constant; when None
        it is read from cfg (must be pilot-frozen, §2.1).
    """
    if len(trajectory_batch) != len(stopper_values):
        raise ValueError("trajectory_batch and stopper_values must align")
    if not trajectory_batch:
        raise ValueError("empty group")

    ex = cfg["executor"]
    lam = float(ex["training_lambda"])
    gamma = float(ex["shaping"]["gamma"])
    fmt_w = float(ex["shaping"]["format_weight"])
    min_cohort = int(ex["grpo"]["min_cohort_guard"])
    variant = ex["grpo"].get("step_level_variant", "per_step_rtg")
    if variant != "per_step_rtg":
        raise NotImplementedError(
            f"step_level_variant={variant!r}: only 'per_step_rtg' is implemented; "
            "the SHAPE-segment alternative (§16 P6 variant (a), K1 picks) is a "
            "TODO(P6) — see VerlCassiAdapter."
        )

    domain = trajectory_batch[0].domain
    if median_pilot_spend is None:
        median_pilot_spend = cfg["label"]["cost_normalization"][f"{domain}_median_pilot_spend"]
        if median_pilot_spend is None:
            raise RuntimeError(
                f"label.cost_normalization.{domain}_median_pilot_spend is null — "
                "run the P2 pilot and freeze it into configs/cassi.yaml (§2.1/§17)."
            )
    median_pilot_spend = float(median_pilot_spend)

    step_rewards: list[np.ndarray] = []
    base_rewards: list[float] = []
    terminal_rewards: list[float] = []
    telescoped: list[float] = []
    v_terminal: list[float] = []

    for traj, v_hat in zip(trajectory_batch, stopper_values):
        v = np.asarray(v_hat, dtype=float)
        if len(v) != len(traj.steps):
            raise ValueError(
                f"V̂ length {len(v)} != trajectory length {len(traj.steps)} "
                f"(task {traj.task_id}, rollout {traj.rollout_idx})"
            )
        q_tau = traj.outcome.get("Q_tau")
        if q_tau is None:
            raise ValueError(
                f"outcome['Q_tau'] missing on task {traj.task_id} — terminal quality "
                "must be scored before reward computation (§2.4)."
            )
        # RL-mode trajectories end at τ, so ALL logged steps are ≤ τ (§2.1);
        # a forced-continuation trajectory passed here would be a caller bug.
        r_t = shaped_step_rewards(v, gamma=gamma)                       # Alg.3 line 2
        r_base = base_reward(                                           # Alg.3 line 3
            float(q_tau), [s.c for s in traj.steps], [s.tier for s in traj.steps],
            lam, median_pilot_spend, rule_table_off=rule_table_off,
        )
        fmt = float(traj.outcome.get("format_score", 1.0))
        step_rewards.append(r_t)
        base_rewards.append(r_base)
        terminal_rewards.append(r_base + fmt_w * fmt)
        telescoped.append(float(r_t.sum()))                             # = −Φ(x_1), γ=1
        v_terminal.append(float(v[-1]) if len(v) else 0.0)

    group: GroupAdvantages = step_level_group_advantages(               # Alg.3 line 4
        step_rewards, terminal_rewards, min_cohort=min_cohort,
    )
    return CassiRewards(
        step_rewards=step_rewards,
        base_rewards=base_rewards,
        terminal_rewards=terminal_rewards,
        advantages=group.advantages,
        cohort_sizes=group.cohort_sizes,
        guarded_steps=group.guarded_steps,
        telescoped_constants=telescoped,
        v_hat_terminal=v_terminal,
    )


def build_verl_trainer_config(cfg: dict, *, domain: str = "qa") -> dict:
    """§17 `executor.grpo` → a verl-style trainer config dict (pure; the values
    verl actually consumes are wired in VerlCassiAdapter). Dr.GRPO length norm,
    KL β=0.04, G=8, rollout/eval temps — all per §17."""
    g = cfg["executor"]["grpo"]
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            # Dr.GRPO unbiased length handling (§2.4 estimator hygiene, mandatory)
            "norm_adv_by_std_in_grpo": False,
            "use_kl_in_reward": False,
        },
        "actor_rollout_ref": {
            "model": {"path": cfg["executor"]["base_model"]},
            "actor": {
                "optim": {"lr": float(g["lr"])},
                "clip_ratio": float(g["clip_eps"]),
                "kl_loss_coef": float(g["kl_beta"]),
                "use_kl_loss": True,
                # Dr.GRPO: token-level (seq-mean-token-sum) loss normalization
                "loss_agg_mode": "seq-mean-token-sum-norm",
            },
            "rollout": {
                "n": int(g["G"]),
                "temperature": float(g["rollout_temp"]),
                "val_kwargs": {"temperature": float(g["eval_temp"])},
                "multi_turn": {"enable": True, "max_turns": int(cfg["executor"]["horizon"][domain])},
            },
        },
        "cassi": {   # consumed by our reward/advantage hooks, not by verl core
            "training_lambda": float(cfg["executor"]["training_lambda"]),
            "shaping": dict(cfg["executor"]["shaping"]),
            "min_cohort_guard": int(g["min_cohort_guard"]),
            "step_level_variant": g.get("step_level_variant", "per_step_rtg"),
            "domain": domain,
        },
    }


# ======================================================================== (b)
class VerlCassiAdapter:
    """The thin seam between CASSI's pure reward semantics and verl's trainer.

    TODO(P6) — exact wiring, kept in one place because verl's custom-advantage
    API changes between versions (this is the ONLY code that may need updating
    when the §17 `pins.verl` commit is frozen at P0):

      1. REWARD FN: verl ≥0.8 custom reward managers expose
           compute_score(data_source, solution_str, ground_truth, extra_info) -> float
         per SAMPLE. That signature is trajectory-level; CASSI needs per-STEP
         rewards. Register instead at the reward-tensor level (RewardManager
         returning a token-level reward tensor): for each rollout group, call
         `compute_cassi_rewards(group_trajs, group_v_hats, cfg)` and write each
         step's shaped reward onto the LAST TOKEN of that step's assistant
         segment (verl AgentLoop marks turn boundaries in the loss mask /
         `multi_turn` metadata). R_terminal goes on the final token.

      2. ADVANTAGES: bypass verl's `compute_grpo_outcome_advantage` (core_algos)
         — it group-normalizes the trajectory-level scalar, which the §2.4
         telescoping proof shows is BLIND to our shaping. Substitute the
         `CassiRewards.advantages` arrays, broadcast each step's advantage over
         that step's response tokens (GiGPO-style step grouping). Keep
         Dr.GRPO/DAPO hygiene: norm_adv_by_std_in_grpo=False +
         token-level loss aggregation (already set in build_verl_trainer_config).

      3. V̂ SOURCE: the stopper is served alongside training (frozen within the
         iteration, §2.7); query its value head on each visited x_t serialized
         per §18.1 to build `stopper_values`. Refresh per iteration; log
         `hacking_divergence(v_hat_terminal, base_rewards)` every step (F6).

      4. SHAPE-SEGMENT VARIANT (K1 alternative): implement segment-level credit
         per SHAPE (2604.06636) as `step_level_variant='shape_segment'` in
         compute_cassi_rewards, then A/B against per_step_rtg in K1.
    """

    def __init__(self, cfg: dict, *, domain: str = "qa"):
        self.cfg = cfg
        self.domain = domain
        self.trainer_config = build_verl_trainer_config(cfg, domain=domain)

    def compute_group_rewards(self, trajectory_batch: list[Trajectory],
                              stopper_values: list[np.ndarray]) -> CassiRewards:
        """The semantics live in the pure function; verl hooks call this."""
        return compute_cassi_rewards(trajectory_batch, stopper_values, self.cfg)


def _check_gpu_stack() -> None:
    try:
        import verl  # noqa: F401 — lazy: never imported at module level
    except ImportError as e:
        raise RuntimeError(_VERL_HELP.format(err=e)) from e
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(_CUDA_HELP)
    except ImportError as e:
        raise RuntimeError(_VERL_HELP.format(err=e)) from e


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="CASSI executor GRPO (paper_plan_v2 Alg.3, §16 P6)")
    p.add_argument("--config", default=None, help="configs/cassi.yaml (§17)")
    p.add_argument("--domain", choices=["qa", "alfworld"], default="qa")
    p.add_argument("--dry-run", action="store_true",
                   help="print the assembled verl trainer config and exit (CPU-safe)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    adapter = VerlCassiAdapter(cfg, domain=args.domain)
    if args.dry_run:
        import json
        print(json.dumps(adapter.trainer_config, indent=2))
        return 0

    _check_gpu_stack()
    raise NotImplementedError(
        "verl trainer launch is gated on the P0 version pin (§17 pins.verl is "
        "null). Follow the TODO(P6) block in VerlCassiAdapter to register the "
        "reward tensor + step-level advantages against the pinned verl API; "
        "the reward semantics are complete and tested in compute_cassi_rewards()."
    )


if __name__ == "__main__":
    sys.exit(main())
