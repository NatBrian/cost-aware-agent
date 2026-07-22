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

(b) `VerlCassiAdapter` + `main()` — wires (a) into the PINNED verl
    (§17 pins.verl = 7aed6b230776f963fa09509c10d9c3a767d1102c, a v0.8.x line).
    All verl-importing code lives in `cassi.executor.verl_hooks` (referenced
    here by fqn STRINGS so this module stays importable on CPU-only boxes);
    the hook mechanism and its file/line pin citations are documented in that
    module's docstring. Summary of the wiring:

      * rollout    — `CassiReactAgentLoop` (agent-loop YAML via
                     rollout.agent.agent_loop_config_path) runs the shared
                     §2.6 ReAct scaffold in RL mode and records step→token
                     alignment (each step's FINAL response token position).
      * V̂ source   — `StopperValueService` loads the --coach checkpoint via
                     cassi.stopper.model.load_predictor and serves batched
                     (frozen within the iteration, refreshed per §2.7).
      * reward     — `CassiAgentLoopManager` (rollout.agent.agent_loop_manager_class)
                     calls compute_cassi_rewards per GRPO group after rollout
                     and writes the token-level rm_scores tensor (advantages
                     difference-encoded on step-final tokens).
      * advantage  — custom adv_estimator "cassi_step_level" (registered with
                     verl's core_algos.register_adv_est) decodes them exactly;
                     verl's trajectory-level GRPO estimator is bypassed (§2.4:
                     provably shaping-blind).
      * Dr.GRPO    — algorithm.norm_adv_by_std_in_grpo=false + actor
                     loss_agg_mode="seq-mean-token-sum-norm" (this commit's
                     actual key names; §2.4 estimator hygiene, mandatory).

    Run parameters cross into verl's ray actors as a JSON sidecar whose path
    travels in the CASSI_GRPO_SIDECAR env var (verl's typed config rejects
    unknown sections; see verl_hooks.SIDECAR_ENV).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cassi.budget.cost import base_reward
from cassi.common.config import load_config, require_pilot_calibration
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
    "NOTE: `import verl` must resolve to third_party/verl (pin 7aed6b2), NOT the\n"
    "verl-agent fork — if verl.__version__ starts with 0.3, re-run\n"
    "  pip install --no-deps -e third_party/verl\n"
)
_CUDA_HELP = (
    "No CUDA device visible. On this machine, acquire GPUs first (never kill "
    "occupier processes):\n"
    "  eval $(/mnt/src/zhanka/gpu_acquire.sh 4)    # 4-8 GPUs for executor GRPO\n"
    "and release them when done: /mnt/src/zhanka/gpu_release.sh"
)

# fqn strings — resolved only inside verl's ray processes (verl_hooks docstring
# documents each hook's pinned-commit file/line).
HOOKS_MODULE = "cassi.executor.verl_hooks"
AGENT_LOOP_MANAGER_FQN = f"{HOOKS_MODULE}.CassiAgentLoopManager"
AGENT_LOOP_TARGET_FQN = f"{HOOKS_MODULE}.CassiReactAgentLoop"
ADV_ESTIMATOR_NAME = "cassi_step_level"   # == verl_hooks.CASSI_ADV_ESTIMATOR
AGENT_LOOP_NAME = "cassi_react"           # == verl_hooks.CASSI_AGENT_LOOP_NAME

ARMS = ("shaped", "shaped_frozen_coach", "shaped_refreshed_coach", "single_multitask")


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
            "TODO(K1) — add it HERE (pure function), the verl wiring in "
            "cassi.executor.verl_hooks is variant-agnostic."
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
    """§17 `executor.grpo` → the CASSI-owned slice of the verl trainer config
    (pure dict; no verl import — merged over verl's composed defaults by
    `VerlCassiAdapter.build_full_verl_config`). Dr.GRPO length norm, KL β=0.04,
    G=8, rollout/eval temps — all per §17. Key names verified against the pin:

      algorithm.adv_estimator / .norm_adv_by_std_in_grpo
                              — verl/trainer/config/algorithm.py:653-654
      actor.loss_agg_mode="seq-mean-token-sum-norm" (Dr.GRPO token-level norm)
                              — verl/workers/config/actor.py:216,
                                core_algos.py agg_loss:1174-1186
      rollout.multi_turn.{enable,max_assistant_turns}
                              — verl/workers/config/rollout.py:66-80
                                (NOT "max_turns"; deviation from older drafts)
    """
    g = cfg["executor"]["grpo"]
    return {
        "algorithm": {
            # OUR estimator (verl_hooks decodes precomputed step-level advantages);
            # verl's "grpo" branch is trajectory-level = provably shaping-blind (§2.4).
            "adv_estimator": ADV_ESTIMATOR_NAME,
            # Dr.GRPO unbiased length handling (§2.4 estimator hygiene, mandatory).
            # Only read by the built-in grpo branch (ray_trainer.py:1622-1632) but
            # kept false so a fallback experiment can't silently reintroduce std-norm.
            "norm_adv_by_std_in_grpo": False,
            # MUST stay false: exact advantage decode needs token_level_rewards ==
            # rm_scores (ray_trainer.py:1600-1607); the KL anchor is actor-side.
            "use_kl_in_reward": False,
            "gamma": 1.0,
            "lam": 1.0,
        },
        "actor_rollout_ref": {
            "model": {"path": cfg["executor"]["base_model"]},
            "actor": {
                "optim": {"lr": float(g["lr"])},
                "clip_ratio": float(g["clip_eps"]),
                "kl_loss_coef": float(g["kl_beta"]),
                "use_kl_loss": True,
                # Dr.GRPO: token-level (seq-mean-token-sum-norm) loss normalization
                "loss_agg_mode": "seq-mean-token-sum-norm",
            },
            "rollout": {
                "n": int(g["G"]),
                "temperature": float(g["rollout_temp"]),
                "val_kwargs": {"temperature": float(g["eval_temp"])},
                "multi_turn": {"enable": True,
                               "max_assistant_turns": int(cfg["executor"]["horizon"][domain])},
                "agent": {
                    "default_agent_loop": AGENT_LOOP_NAME,
                    # agent_loop_config_path + agent_loop_manager_class are run
                    # artifacts (written under --out) — filled by build_full_verl_config.
                },
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
@dataclass
class GrpoRunSpec:
    """One §16 P5/P6/P7 training run — the CLI surface, resolved."""

    domain: str = "qa"
    tasks: list[str] = field(default_factory=list)   # jsonl task files
    iteration: int = 1
    seed: int = 42
    lam: float | None = None            # None → executor.training_lambda (§17)
    coach: str | None = None            # stopper checkpoint dir (V̂ source, §2.4)
    arm: str = "shaped"                 # §16: shaped | shaped_{frozen,refreshed}_coach | single_multitask
    step_credit: str | None = None      # per_step_rtg | shape_segment (K1 picks, §2.4)
    max_steps: int = -1                 # trainer.total_training_steps (P7 matched compute, E5)
    init: str | None = None             # warm-start executor ckpt (P7 iteration 2)
    out: str = "experiments/grpo/dev"   # checkpoint/artifact dir
    vllm_url: str | None = None         # §16 CLI parity; NOT used by verl rollout
                                        # (verl colocates its own vLLM engines) —
                                        # forwarded to the sidecar for eval tooling.
    retriever_url: str = "http://127.0.0.1:8000/retrieve"


class VerlCassiAdapter:
    """The thin seam between CASSI's pure reward semantics and verl's trainer.

    CPU-safe by construction: this class never imports verl at module scope —
    verl-touching work happens in `build_full_verl_config` / `validate_hooks` /
    `launch` (lazy imports), and the heavy hook classes live in
    `cassi.executor.verl_hooks` (see its docstring for the mechanism choice and
    every pinned-commit file/line reference). What each §16 P6 TODO item became:

      1. REWARD FN      → verl_hooks.CassiAgentLoopManager._apply_cassi_rewards
                          (batch-level rm_scores tensor; per-step values on each
                          step's FINAL response token). A per-sample RewardManager
                          cannot see the GRPO group on this commit's streaming
                          reward path — documented in verl_hooks.
      2. ADVANTAGES     → adv_estimator "cassi_step_level" (registered via
                          core_algos.register_adv_est) decodes the precomputed
                          `CassiRewards.advantages`; compute_grpo_outcome_advantage
                          is never called. Dr.GRPO hygiene via config (see
                          build_verl_trainer_config).
      3. V̂ SOURCE       → verl_hooks.StopperValueService (load_predictor on the
                          --coach dir; batched; refresh = new --coach per
                          iteration, §2.7). F6 divergence → <out>/divergence.csv.
      4. SHAPE-SEGMENT  → TODO(K1): implement as step_level_variant='shape_segment'
                          inside compute_cassi_rewards (pure layer); the verl
                          wiring is variant-agnostic.
    """

    def __init__(self, cfg: dict, *, domain: str = "qa"):
        self.cfg = cfg
        self.domain = domain
        self.trainer_config = build_verl_trainer_config(cfg, domain=domain)

    def compute_group_rewards(self, trajectory_batch: list[Trajectory],
                              stopper_values: list[np.ndarray]) -> CassiRewards:
        """The semantics live in the pure function; verl hooks call this."""
        return compute_cassi_rewards(trajectory_batch, stopper_values, self.cfg)

    # ---------------------------------------------------------- run artifacts
    def write_run_files(self, run: GrpoRunSpec) -> dict:
        """Write the per-run artifacts under --out:
          agent_loop.yaml — consumed by AgentLoopWorker at init
                            (pin: agent_loop.py:443-449, hydra _target_);
          sidecar.json    — CASSI params for the ray actors (verl_hooks.SIDECAR_ENV).
        """
        out = Path(run.out)
        out.mkdir(parents=True, exist_ok=True)
        agent_loop_yaml = out / "agent_loop.yaml"
        agent_loop_yaml.write_text(
            f"- name: {AGENT_LOOP_NAME}\n  _target_: {AGENT_LOOP_TARGET_FQN}\n")
        sidecar = out / "cassi_sidecar.json"
        sidecar.write_text(json.dumps({
            "config_path": str(self._config_path or ""),
            "domain": run.domain,
            "lam": float(run.lam if run.lam is not None
                         else self.cfg["executor"]["training_lambda"]),
            "coach_dir": run.coach,
            "arm": run.arm,
            "iteration": run.iteration,
            "seed": run.seed,
            "out_dir": str(out),
            "retriever_url": run.retriever_url,
            "vllm_url": run.vllm_url,
            "stopper_device": os.environ.get("CASSI_STOPPER_DEVICE", "cpu"),
            "terminal_quality_metric": "em",   # §2.4 QA headline (F1 variant reported)
        }, indent=2))
        return {"agent_loop_yaml": agent_loop_yaml, "sidecar": sidecar}

    _config_path: str | None = None   # set by main() so the sidecar can point at it

    def build_task_parquet(self, run: GrpoRunSpec) -> Path:
        """--tasks jsonl file(s) → verl RLHFDataset parquet (data.train_files).

        Columns beyond verl's standard (prompt / data_source / reward_model /
        agent_name) ride the non-tensor batch into the agent loop's kwargs
        (pin: agent_loop.py:548-551): task_id, question, gold, allowance_B,
        wallet_size. One wallet per (task, GRPO group): the row is repeated
        rollout.n times interleaved (ray_trainer.py:1447), so all G rollouts
        share the drawn wallet (§2.2)."""
        import pandas as pd

        require_pilot_calibration(self.cfg, run.domain)     # §17: refuse past P2 on nulls
        allowances = self.cfg["label"]["allowances"][run.domain]
        rng = np.random.default_rng(run.seed)

        # identical scaffold messages to what ReactAgent builds (§2.6)
        from cassi.executor.react_agent import SYSTEM_TEMPLATE
        from cassi.labels.drafts import DRAFT_TEMPLATE_INSTRUCTION
        if run.domain == "qa":
            from cassi.executor.envs.searchr1_qa import SearchR1QAEnv
            tools_text = SearchR1QAEnv(retriever_url=run.retriever_url).tools()
        else:
            raise NotImplementedError(
                "alfworld task staging rides the verl-agent harness (§19) — "
                "see verl_hooks.CassiReactAgentLoop._make_env.")
        system = SYSTEM_TEMPLATE.format(tools=tools_text,
                                        draft_instruction=DRAFT_TEMPLATE_INSTRUCTION)

        rows = []
        for path in run.tasks:
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    t = json.loads(line)
                    question = t.get("question") or t.get("query") or ""
                    gold = t.get("gold") or t.get("answer") or ""
                    if not gold and t.get("golden_answers"):
                        gold = t["golden_answers"][0]
                    wallet = str(rng.choice(["small", "medium", "large"]))
                    rows.append({
                        "prompt": [{"role": "system", "content": system},
                                   {"role": "user", "content": f"Question: {question}"}],
                        "data_source": f"cassi_{run.domain}",
                        "reward_model": {"style": "rule", "ground_truth": gold},
                        "agent_name": AGENT_LOOP_NAME,
                        "task_id": str(t.get("task_id") or t.get("id") or f"t{len(rows)}"),
                        "question": question,
                        "gold": gold,
                        "wallet_size": wallet,
                        "allowance_B": float(allowances[wallet]),
                        "extra_info": {"domain": run.domain, "iteration": run.iteration},
                    })
        if not rows:
            raise RuntimeError(f"no tasks parsed from {run.tasks}")
        out = Path(run.out) / f"tasks_{run.domain}_iter{run.iteration}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out)
        return out

    # ------------------------------------------------------- verl config side
    def build_full_verl_config(self, run: GrpoRunSpec, *, train_files: str,
                               n_gpus: int):
        """Compose verl's own ppo_trainer defaults (hydra, from the PINNED
        checkout's verl/trainer/config) and merge the CASSI overrides.

        Returns the OmegaConf DictConfig that `verl.trainer.main_ppo.run_ppo`
        consumes. The `cassi` section of trainer_config is NOT merged — verl's
        typed configs must not see unknown sections; those values travel in the
        sidecar instead."""
        import verl
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        cfg_dir = str(Path(verl.__file__).parent / "trainer" / "config")
        with initialize_config_dir(config_dir=cfg_dir, version_base=None):
            base = compose(config_name="ppo_trainer")   # pin: main_ppo.py:38 hydra entry
        OmegaConf.set_struct(base, False)

        overrides = {k: v for k, v in self.trainer_config.items() if k != "cassi"}
        run_files = self.write_run_files(run)
        horizon = int(self.cfg["executor"]["horizon"][run.domain])
        g = self.cfg["executor"]["grpo"]

        deep = {
            "data": {
                "train_files": train_files,
                "val_files": train_files,      # dev evals run through eval/run_frontier (§16), not verl
                "train_batch_size": 64,        # prompts/step; G=8 → 512 trajectories/step
                "max_prompt_length": 2048,
                "max_response_length": 1024 * max(4, horizon),   # multi-turn region incl. obs tokens
                "shuffle": True,
                "seed": run.seed,
            },
            "actor_rollout_ref": {
                "model": {"path": run.init or self.cfg["executor"]["base_model"]},
                "actor": {"ppo_mini_batch_size": 64},
                "rollout": {
                    "name": "vllm",            # §19: vllm >= 0.17 (Qwen3.5 GDN kernels)
                    "prompt_length": 2048,
                    "response_length": 1024 * max(4, horizon),
                    "agent": {
                        "default_agent_loop": AGENT_LOOP_NAME,
                        # pin: agent_loop.py:443-449 (worker loads the yaml)
                        "agent_loop_config_path": str(run_files["agent_loop_yaml"]),
                        # pin: ray_trainer.py:929-935 (load_class_from_fqn) +
                        # workers/config/rollout.py:97
                        "agent_loop_manager_class": AGENT_LOOP_MANAGER_FQN,
                    },
                },
            },
            "trainer": {
                "project_name": self.cfg["tracking"]["wandb_project"],
                "experiment_name": f"grpo_{run.domain}_iter{run.iteration}"
                                   f"_{run.arm}_lam{run.lam}_seed{run.seed}",
                "default_local_dir": str(Path(run.out) / "checkpoints"),
                "n_gpus_per_node": n_gpus,
                "nnodes": 1,
                "total_epochs": 1,
                "total_training_steps": run.max_steps if run.max_steps > 0 else None,
                "test_freq": -1,               # frontier evals are eval/run_frontier's job
                "val_before_train": False,
                "logger": ["console", "wandb"],
            },
        }

        def _merge(dst: dict, src: dict) -> dict:
            for k, v in src.items():
                dst[k] = _merge(dst.get(k, {}), v) if isinstance(v, dict) else v
            return dst

        merged_overrides = _merge(overrides, deep)
        cfg = OmegaConf.merge(base, OmegaConf.create(merged_overrides))
        return cfg

    def validate_hooks(self) -> list[str]:
        """Prove the hooks are importable + registered exactly the way verl will
        load them (CPU-safe when verl is installed; the pinned verl imports
        without CUDA). Returns human-readable status lines; raises on failure."""
        lines = []
        import importlib

        hooks = importlib.import_module(HOOKS_MODULE)
        lines.append(f"hooks module import          OK  ({HOOKS_MODULE})")

        # pin: core_algos.py:113-151 — registration happens at hooks import
        from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
        assert hooks.CASSI_ADV_ESTIMATOR == ADV_ESTIMATOR_NAME
        assert ADV_ESTIMATOR_NAME in ADV_ESTIMATOR_REGISTRY, ADV_ESTIMATOR_REGISTRY.keys()
        lines.append(f"adv estimator registered     OK  (algorithm.adv_estimator={ADV_ESTIMATOR_NAME!r})")

        # pin: agent_loop.py:379-390 — decorator path (workers additionally load
        # the agent_loop.yaml at init, agent_loop.py:443-449)
        from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
        assert AGENT_LOOP_NAME in _agent_loop_registry, _agent_loop_registry.keys()
        lines.append(f"agent loop registered        OK  (agent_name={AGENT_LOOP_NAME!r})")

        # pin: ray_trainer.py:929-935 uses load_class_from_fqn on this exact string
        from verl.utils.import_utils import load_class_from_fqn
        cls = load_class_from_fqn(AGENT_LOOP_MANAGER_FQN, "AgentLoopManager")
        from verl.experimental.agent_loop.agent_loop import AgentLoopManager
        assert issubclass(cls, AgentLoopManager)
        lines.append(f"agent loop manager loadable  OK  ({AGENT_LOOP_MANAGER_FQN})")
        return lines

    # ------------------------------------------------------------- execution
    def launch(self, run: GrpoRunSpec) -> int:
        """Assemble everything and hand off to verl's trainer (GPU path)."""
        if run.arm == "single_multitask":
            raise NotImplementedError(
                "--arm single_multitask (K2/A2: one 9B with task+stopping heads) "
                "is separate machinery from the shaped bridge — §5.5 A2 owns it.")
        if run.arm.startswith("shaped") and not run.coach:
            raise ValueError(f"--arm {run.arm} requires --coach <stopper ckpt dir> (§2.4 V̂ source)")
        if not run.tasks:
            raise ValueError("--tasks is required to launch (comma-separated jsonl files)")
        _check_gpu_stack()

        import torch
        n_gpus = torch.cuda.device_count()
        train_files = str(self.build_task_parquet(run))
        config = self.build_full_verl_config(run, train_files=train_files, n_gpus=n_gpus)
        run_files = self.write_run_files(run)

        # Cross-process plumbing: ray actors (TaskRunner, AgentLoopWorker) need
        # (a) `cassi.*` importable and (b) the sidecar path. run_ppo only calls
        # ray.init when not initialized (pin: main_ppo.py:62-83), so initializing
        # here lets us inject the runtime env.
        import ray
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
        research_dir = str(Path(__file__).resolve().parent.parent.parent)
        os.environ["CASSI_GRPO_SIDECAR"] = str(run_files["sidecar"])
        runtime_env = get_ppo_ray_runtime_env()
        runtime_env["env_vars"].update({
            "PYTHONPATH": research_dir + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "CASSI_GRPO_SIDECAR": str(run_files["sidecar"]),
        })
        if not ray.is_initialized():
            ray.init(runtime_env=runtime_env)

        # pin: main_ppo.py:46-48 — legacy reward keys are migrated before run_ppo
        from verl.experimental.reward_loop import migrate_legacy_reward_impl
        from verl.trainer.main_ppo import run_ppo
        config = migrate_legacy_reward_impl(config)
        print(f"[train_grpo] launching verl trainer: arm={run.arm} domain={run.domain} "
              f"iter={run.iteration} λ={run.lam} coach={run.coach} out={run.out}")
        run_ppo(config)
        return 0


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


def _dry_run(adapter: VerlCassiAdapter, run: GrpoRunSpec) -> int:
    """CPU-safe: print the assembled config, validate hook registration, exit 0.
    Never touches CUDA, the retriever, or the task files (§16 P6 contract)."""
    print(json.dumps(adapter.trainer_config, indent=2))
    print("\n[dry-run] §16 CLI resolved:")
    for k, v in vars(run).items():
        print(f"  --{k.replace('_', '-'):<16} {v}")
    try:
        import verl  # noqa: F401
    except ImportError as e:
        print(f"\n[dry-run] hook validation SKIPPED — verl not importable here ({e}).")
        print("[dry-run] install the §16 P0 stack; this dry-run still exits 0 by design.")
        return 0
    for line in adapter.validate_hooks():
        print(f"[dry-run] {line}")
    print(f"[dry-run] verl pin sanity: import resolves to "
          f"{__import__('verl').__file__}")
    print("[dry-run] full-config compose (hydra, pinned defaults) ...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        probe = GrpoRunSpec(**{**vars(run), "out": tmp})
        cfg = adapter.build_full_verl_config(probe, train_files="<built at launch>", n_gpus=8)
        print(f"[dry-run]   algorithm.adv_estimator          = {cfg.algorithm.adv_estimator}")
        print(f"[dry-run]   algorithm.norm_adv_by_std_in_grpo = {cfg.algorithm.norm_adv_by_std_in_grpo}")
        print(f"[dry-run]   algorithm.use_kl_in_reward        = {cfg.algorithm.use_kl_in_reward}")
        print(f"[dry-run]   actor.loss_agg_mode               = {cfg.actor_rollout_ref.actor.loss_agg_mode}")
        print(f"[dry-run]   actor.kl_loss_coef / use_kl_loss  = "
              f"{cfg.actor_rollout_ref.actor.kl_loss_coef} / {cfg.actor_rollout_ref.actor.use_kl_loss}")
        print(f"[dry-run]   rollout.n (G)                     = {cfg.actor_rollout_ref.rollout.n}")
        print(f"[dry-run]   rollout.agent.default_agent_loop  = {cfg.actor_rollout_ref.rollout.agent.default_agent_loop}")
        print(f"[dry-run]   rollout.agent.agent_loop_manager_class = "
              f"{cfg.actor_rollout_ref.rollout.agent.agent_loop_manager_class}")
    print("[dry-run] OK — config assembles and all hooks register against the pinned verl.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="CASSI executor GRPO (paper_plan_v2 Alg.3, §16 P5/P6/P7)")
    p.add_argument("--config", default=None, help="configs/cassi.yaml (§17)")
    p.add_argument("--domain", choices=["qa", "alfworld"], default="qa")
    p.add_argument("--tasks", default=None,
                   help="comma-separated jsonl task files (P1 outputs)")
    p.add_argument("--iteration", type=int, default=1, help="loop iteration i (§2.7)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda", dest="lam", type=float, default=None,
                   help="training λ (default: executor.training_lambda, §17)")
    p.add_argument("--coach", default=None,
                   help="stopper checkpoint dir — the V̂ source (§2.4); refresh per iteration (§2.7)")
    p.add_argument("--arm", default="shaped", choices=list(ARMS),
                   help="§16 arm: shaped (P5/P6) | shaped_{frozen,refreshed}_coach (P7/E5) "
                        "| single_multitask (K2/A2, separate machinery)")
    p.add_argument("--step-credit", dest="step_credit", default=None,
                   choices=["per_step_rtg", "shape_segment"],
                   help="step-level credit variant (K1 picks, §2.4); overrides "
                        "executor.grpo.step_level_variant")
    p.add_argument("--max-steps", dest="max_steps", type=int, default=-1,
                   help="trainer.total_training_steps cap (P7 matched compute, E5)")
    p.add_argument("--init", default=None, help="warm-start executor ckpt (P7 iteration 2)")
    p.add_argument("--out", default="experiments/grpo/dev", help="checkpoint/artifact dir")
    p.add_argument("--vllm-url", dest="vllm_url", default=None,
                   help="§16 CLI parity; verl colocates its own vLLM — forwarded to the sidecar")
    p.add_argument("--retriever-url", dest="retriever_url",
                   default=os.environ.get("CASSI_RETRIEVER_URL", "http://127.0.0.1:8000/retrieve"),
                   help="Search-R1 retrieval server (§16 P1)")
    p.add_argument("--dry-run", action="store_true",
                   help="build the full verl config, validate hook registration, print, exit 0 (CPU-safe)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.step_credit:
        cfg["executor"]["grpo"]["step_level_variant"] = args.step_credit
    run = GrpoRunSpec(
        domain=args.domain,
        tasks=[t for t in (args.tasks or "").split(",") if t],
        iteration=args.iteration, seed=args.seed,
        lam=args.lam if args.lam is not None else float(cfg["executor"]["training_lambda"]),
        coach=args.coach, arm=args.arm, step_credit=args.step_credit,
        max_steps=args.max_steps, init=args.init, out=args.out,
        vllm_url=args.vllm_url, retriever_url=args.retriever_url,
    )
    adapter = VerlCassiAdapter(cfg, domain=args.domain)
    adapter._config_path = str(args.config or
                               Path(__file__).resolve().parent.parent / "configs" / "cassi.yaml")

    if args.dry_run:
        return _dry_run(adapter, run)
    return adapter.launch(run)


if __name__ == "__main__":
    sys.exit(main())
