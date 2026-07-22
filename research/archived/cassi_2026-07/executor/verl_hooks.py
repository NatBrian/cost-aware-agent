"""CASSI ↔ verl wiring — the P6 hooks (paper_plan_v2 §2.4, §10 Alg.3, §16 P6, §19).

This module imports verl/torch at module level and therefore lives OUTSIDE
`train_grpo.py` (which must stay importable on CPU-only boxes, §16 P0). It is
loaded only inside verl's ray processes, by name:

  * `CassiAgentLoopManager`  — via config key
    `actor_rollout_ref.rollout.agent.agent_loop_manager_class`
    (pin 7aed6b2: verl/workers/config/rollout.py:97; loaded with
    `load_class_from_fqn` in verl/trainer/ppo/ray_trainer.py:929-935).
  * `CassiReactAgentLoop`    — via the agent-loop YAML written by
    `train_grpo.VerlCassiAdapter` and pointed at by
    `actor_rollout_ref.rollout.agent.agent_loop_config_path`
    (pin: verl/experimental/agent_loop/agent_loop.py:443-449, instantiated
    per-sample with hydra at agent_loop.py:584-597).
  * `compute_cassi_step_level_advantage` — registered as adv_estimator
    "cassi_step_level" at import time (pin: verl/trainer/ppo/core_algos.py:113-151);
    the TaskRunner process imports this module when it loads the manager class,
    so the registration exists exactly where `compute_advantage` dispatches it
    (pin: verl/trainer/ppo/ray_trainer.py:248-283).

WHY manager-override instead of a custom RewardManager (the mechanism choice,
documented per §16 P6):

  At pin 7aed6b2 the trainer runs rewards through the *streaming* reward loop:
  with no reward model, `enable_agent_reward_loop` is always True
  (ray_trainer.py:938-940), so rewards are computed per-trajectory inside the
  agent loop (`_compute_score`, agent_loop.py:839-901) and materialized as ONE
  SCALAR on the final response token (`_postprocess`, agent_loop.py:962-970).
  A reward manager therefore never sees a whole GRPO group and can never emit
  a token-level tensor on this path — the batch-level manager API
  (`assemble_rm_scores`, reward_loop/reward_manager/base.py:62-83 +
  reward_loop.py:323-352) is only reachable with a colocated reward model.
  The one config-exposed, batch-level, post-rollout hook this commit supports
  is a custom AgentLoopManager; `CassiAgentLoopManager.generate_sequences`
  overwrites `batch["rm_scores"]` after rollout, which the trainer then uses
  verbatim: rm_scores → token_level_scores → token_level_rewards (KL-in-reward
  OFF) → compute_advantage (ray_trainer.py:1592-1633).

ADVANTAGE ENCODING (exactness argument):

  `compute_cassi_rewards` (pure, CPU-tested) already produces the final
  per-step advantages A_t (per-step returns-to-go, group-normalized, min-cohort
  guard — Alg.3). verl's custom adv estimators receive ONLY
  (token_level_rewards, response_mask, config, index) (ray_trainer.py:249-258),
  so the advantages are transported inside the reward tensor itself,
  difference-encoded: rm_scores[end_t] = A_t − A_{t+1} (A_{T+1} := 0) on each
  step's FINAL response token. The registered estimator decodes by reverse
  cumulative sum: every token at or before end_t (and after end_{t-1}) sums
  d_t + d_{t+1} + ... = A_t — i.e. each step's advantage is broadcast over
  exactly that step's response tokens (GiGPO-style step grouping, §2.4).
  Tokens between end_t and step t+1's first generated token get A_{t+1}, but
  those are tool-observation tokens with response_mask = 0 and never reach the
  loss (agg_loss masks them; core_algos.py:1138-1197). Exactness requires
  `algorithm.use_kl_in_reward=False` (asserted in the manager) so that
  token_level_rewards == rm_scores bit-for-bit (ray_trainer.py:1600-1607).

  Consequence (documented quirk): verl's logged batch "reward" metrics are
  computed from token_level_scores and now sum to A_1 per trajectory — a
  meaningless number. The TRUE economic rewards + the F6 V̂-vs-realized
  divergence go to `<out>/divergence.csv` instead.

verl's own GRPO estimator (`compute_grpo_outcome_advantage`, dispatched from
the AdvantageEstimator.GRPO branch at ray_trainer.py:233-246) is bypassed
entirely: it group-normalizes the trajectory-level scalar sum, which §2.4's
telescoping proof shows is blind to the shaping.

Every verl symbol used below carries a `# pin:` comment with the
7aed6b230776f963fa09509c10d9c3a767d1102c file/line so the next person can
re-verify after a verl upgrade (§17 pins.verl).
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import threading
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

# pin 7aed6b2: verl/experimental/agent_loop/agent_loop.py:88-146 (AgentLoopOutput),
# :194-228 (AgentLoopBase), :379-390 (register), :1044-1130 (AgentLoopManager)
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopManager,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

# pin 7aed6b2: verl/trainer/ppo/core_algos.py:113-151 (registry + decorator)
from verl.trainer.ppo.core_algos import register_adv_est

from cassi.common.config import load_config
from cassi.common.schema import Trajectory
from cassi.executor.react_agent import ReactAgent
from cassi.executor.shaping import hacking_divergence
from cassi.executor.train_grpo import compute_cassi_rewards
from cassi.labels.quality import qa_quality

logger = logging.getLogger(__name__)

# The one name shared with train_grpo's config assembly (train_grpo references
# it as a string so it never has to import this module on CPU).
CASSI_ADV_ESTIMATOR = "cassi_step_level"
CASSI_AGENT_LOOP_NAME = "cassi_react"

# Cross-process config channel: verl's typed config rejects unknown top-level
# sections, so CASSI's run parameters travel as a JSON sidecar whose path is in
# this env var — propagated to the TaskRunner + agent-loop ray actors through
# ray.init(runtime_env={"env_vars": ...}) in train_grpo.main (run_ppo only
# initializes ray when it isn't already; pin: verl/trainer/main_ppo.py:62-83).
SIDECAR_ENV = "CASSI_GRPO_SIDECAR"


def load_sidecar() -> dict:
    """Read the CASSI run sidecar written by train_grpo.VerlCassiAdapter."""
    path = os.environ.get(SIDECAR_ENV)
    if not path or not Path(path).exists():
        raise RuntimeError(
            f"{SIDECAR_ENV} is unset or points at a missing file ({path!r}). "
            "Launch through `python -m cassi.executor.train_grpo` (§16 P6) — it "
            "writes the sidecar and injects the env var into ray's runtime env."
        )
    return json.loads(Path(path).read_text())


# ===================================================== 1. advantage estimator
@register_adv_est(CASSI_ADV_ESTIMATOR)  # pin: core_algos.py:116-134
def compute_cassi_step_level_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config=None,
    index=None,
    **kwargs,
):
    """Decode CASSI's precomputed step-level advantages (§2.4, Alg.3 line 4).

    Dispatched by `compute_advantage`'s registry fallthrough with exactly these
    kwargs (pin: ray_trainer.py:248-283). `token_level_rewards` carries the
    difference-encoded advantages written by `CassiAgentLoopManager`
    (d_t = A_t − A_{t+1} on step-final tokens); the reverse cumulative sum
    reconstructs A_t on every token of step t — see the module docstring for
    the exactness argument. All GRPO group statistics (per-step RTG, cohort
    normalization, min-cohort guard) were already applied by the CPU-tested
    `compute_cassi_rewards`; this function is a pure decoder and must never
    re-normalize (Dr.GRPO hygiene lives upstream, §2.4).
    """
    rev_cumsum = torch.flip(
        torch.cumsum(torch.flip(token_level_rewards, dims=[-1]), dim=-1), dims=[-1]
    )
    advantages = rev_cumsum * response_mask
    # returns := advantages, matching verl's GRPO outcome convention
    # (pin: core_algos.py compute_grpo_outcome_advantage returns scores, scores).
    return advantages, advantages


def encode_step_values(step_values: np.ndarray, step_ends: list[int], width: int) -> torch.Tensor:
    """Difference-encode per-step scalars onto step-final token positions.

    Inverse of `compute_cassi_step_level_advantage`'s reverse-cumsum on the
    mask-1 positions: row[end_t] = v_t − v_{t+1} with v_{T+1} := 0."""
    v = np.asarray(step_values, dtype=np.float64)
    if len(v) != len(step_ends):
        raise ValueError(f"{len(v)} step values vs {len(step_ends)} step ends")
    row = torch.zeros(width, dtype=torch.float32)
    diffs = v - np.append(v[1:], 0.0)
    for pos, d in zip(step_ends, diffs):
        if not 0 <= pos < width:
            raise ValueError(f"step-final position {pos} outside response width {width}")
        row[pos] = float(d)
    return row


# ========================================================== 2. V̂ source (§2.7)
class StopperValueService:
    """Serves per-step V̂_θ(x_t) from the stopper checkpoint — frozen within the
    iteration, refreshed per §2.7 (P7 relaunches training with the new --coach).

    Loads via `cassi.stopper.model.load_predictor` (lazy) and batches all steps
    of all trajectories through `HFStopperPredictor.predict_batch`. Runs on the
    device named in the sidecar (`stopper_device`, default "cpu": the TaskRunner
    actor holds no GPU in verl's resource accounting — the 2B stopper is served
    CPU-side per batch; TODO(GPU): co-locate on a training-GPU slice if CPU
    latency dominates the step time)."""

    def __init__(self, coach_dir: str, *, device: str = "cpu", batch_size: int = 64):
        self.coach_dir = coach_dir
        self.device = device
        self.batch_size = batch_size
        self._predictor = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._predictor is None:
                from cassi.stopper.model import load_predictor
                self._predictor = load_predictor(self.coach_dir, device=self.device)
            return self._predictor

    def values_for(self, trajectories: list[Trajectory], lam: float) -> list[np.ndarray]:
        """V̂ on every VISITED state of every trajectory (Φ(terminal):=0 is
        applied later by `shaped_step_rewards`, not here — §2.4)."""
        predictor = self._get()
        items = []
        for tr in trajectories:
            for s in tr.steps:
                items.append((s.x, lam, {
                    "task_id": tr.task_id, "group_id": tr.group_id,
                    "rollout_idx": tr.rollout_idx, "domain": tr.domain,
                    "allowance_B": tr.allowance_B, "t": s.x.step_idx,
                }))
        preds = predictor.predict_batch(items, batch_size=self.batch_size)
        out, i = [], 0
        for tr in trajectories:
            n = len(tr.steps)
            out.append(np.array([p.v for p in preds[i:i + n]], dtype=float))
            i += n
        return out


# ====================================================== 3. rollout agent loop
class _TokenBridgeClient:
    """LLMClient adapter: lets the CPU-tested `ReactAgent` scaffold (§2.6 — the
    SAME loop as collection/inference, so training is not a second scaffold)
    drive verl's async LLM server, while recording per-turn token ids for the
    step→token alignment.

    `ReactAgent.run` executes in a worker thread (the agent loop's event loop
    must not block for a whole episode); each generate() submits a coroutine
    back to the loop with `run_coroutine_threadsafe`."""

    def __init__(self, agent_loop: "CassiReactAgentLoop", sampling_params: dict):
        self.agent_loop = agent_loop
        self.sampling_params = sampling_params
        self.turns: list[tuple[list[int], list[int]]] = []   # (prompt_ids, response_ids)

    def generate(self, messages: list[dict], max_tokens: int) -> str:
        # Per-turn max_tokens is intentionally NOT forwarded: at pin 7aed6b2 the
        # bundled ToolAgentLoop passes sampling_params through unchanged and
        # handles length via post-hoc truncation + response_length termination
        # (tool_agent_loop.py:270-276) — we follow the same convention.
        fut = asyncio.run_coroutine_threadsafe(self._one_turn(messages), self.agent_loop.loop)
        return fut.result()

    async def _one_turn(self, messages: list[dict]) -> str:
        # pin: agent_loop.py:272-330 apply_chat_template → prompt token ids
        prompt_ids = await self.agent_loop.apply_chat_template(messages)
        # pin: workers/rollout/replica.py:39-51 TokenOutput; server generate as
        # used by SingleTurnAgentLoop (single_turn_agent_loop.py:60-70)
        out = await self.agent_loop.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=self.sampling_params,
        )
        self.turns.append((list(prompt_ids), list(out.token_ids)))
        return self.agent_loop.tokenizer.decode(out.token_ids, skip_special_tokens=True)


def _align_turns(turns: list[tuple[list[int], list[int]]]) -> tuple[list[int], list[int], list[int]]:
    """Flatten per-turn (prompt_ids, response_ids) into verl's response region.

    Returns (response_ids, response_mask, step_ends): mask 1 for generated
    tokens, 0 for observation/template tokens (the format documented at
    agent_loop.py:102-104 / generate_sequences docstring, pin lines 1102+);
    step_ends[i] = index of step i+1's FINAL generated token in the region.

    Observation segments are recovered as the suffix of the next turn's prompt
    past its longest common prefix with (prev_prompt + prev_response) — chat
    templates are append-only for generation prompts, but EOS re-rendering can
    shift the boundary by a token or two, hence the LCP tolerance (verl's own
    ToolAgentLoop guards the same risk with multi_turn.tokenization_sanity_check_mode)."""
    response_ids: list[int] = []
    response_mask: list[int] = []
    step_ends: list[int] = []
    for i, (p_ids, r_ids) in enumerate(turns):
        if i > 0:
            prev_full = turns[i - 1][0] + turns[i - 1][1]
            lcp = 0
            for a, b in zip(prev_full, p_ids):
                if a != b:
                    break
                lcp += 1
            obs_seg = p_ids[lcp:]
            if lcp < len(turns[i - 1][0]):
                logger.warning("cassi_react: chat template re-rendered history "
                               "(lcp %d < prompt %d); masks may be approximate", lcp, len(turns[i - 1][0]))
            response_ids.extend(obs_seg)
            response_mask.extend([0] * len(obs_seg))
        response_ids.extend(r_ids)
        response_mask.extend([1] * len(r_ids))
        step_ends.append(len(response_ids) - 1)
    return response_ids, response_mask, step_ends


@register(CASSI_AGENT_LOOP_NAME)  # pin: agent_loop.py:382-390 (also reachable via the YAML _target_)
class CassiReactAgentLoop(AgentLoopBase):
    """Multi-turn ReAct rollout for CASSI GRPO — RL mode: terminate at ANSWER or
    T_max (§2.1), the shared §2.6 scaffold via `ReactAgent`, Search-R1 retriever
    env (§5.1). Instantiated per sample with the dataset row's non-tensor fields
    as kwargs (pin: agent_loop.py:548-597), including our parquet columns
    task_id / question / gold / allowance_B / wallet_size and verl's uid."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sidecar = load_sidecar()
        self.cassi_cfg = load_config(self.sidecar["config_path"])
        self.domain = self.sidecar["domain"]
        self.t_max = int(self.cassi_cfg["executor"]["horizon"][self.domain])
        self.response_length = self.rollout_config.response_length
        self.terminal_metric = self.sidecar.get("terminal_quality_metric", "em")  # §2.4 headline

    def _make_env(self):
        if self.domain == "qa":
            from cassi.executor.envs.searchr1_qa import SearchR1QAEnv
            return SearchR1QAEnv(
                retriever_url=self.sidecar["retriever_url"],
                quality_metric=self.cassi_cfg["label"]["quality_scoring"]["qa"],
            )
        # DEVIATION (documented): §16 P6 trains both domains, but the ALFWorld
        # rollout rides verl-agent's harness (§19), which forks verl 0.3.x and
        # cannot share this pin's AgentLoop API — wire it as its own loop when
        # the verl-agent env is staged (P1.3).
        raise NotImplementedError(
            f"domain {self.domain!r}: only 'qa' rolls out through this agent loop; "
            "alfworld runs on the verl-agent harness (§19 manifest)."
        )

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        task = {
            "task_id": str(kwargs.get("task_id", "")),
            "question": str(kwargs.get("question", "")),
            "gold": kwargs.get("gold"),
        }
        uid = str(kwargs.get("uid", ""))
        bridge = _TokenBridgeClient(self, sampling_params)
        agent = ReactAgent(bridge)
        env = self._make_env()

        metrics = {}
        from verl.utils.profiler import simple_timer  # pin: same helper the bundled loops use
        with simple_timer("generate_sequences", metrics):
            # RL mode, NO monitor: "exploration must be free, economics reach the
            # policy only through rewards" (§2.1). Runs in a thread so the worker
            # event loop keeps serving the other trajectories.
            result = await self.loop.run_in_executor(None, lambda: agent.run(
                task, env, mode="rl", t_max=self.t_max,
                allowance_dollars=float(kwargs.get("allowance_B", 0.0)),
                wallet_size=str(kwargs.get("wallet_size", "medium")),
                group_id=uid, rollout_idx=0,
                seed=self.sidecar.get("seed"), iteration=int(self.sidecar.get("iteration", 1)),
            ))
        metrics.setdefault("num_preempted", -1)

        traj = result.trajectory
        response_ids, response_mask, step_ends = _align_turns(bridge.turns)

        # Post-hoc truncation to rollout.response_length (same convention as
        # ToolAgentLoop, pin tool_agent_loop.py:187-199): drop steps whose final
        # token fell outside the window and clip the trajectory to match.
        n_keep = sum(1 for e in step_ends if e < self.response_length)
        truncated = n_keep < len(traj.steps)
        if truncated:
            traj.steps = traj.steps[:n_keep]
            step_ends = step_ends[:n_keep]
            traj.outcome["tau"] = None
            traj.outcome["stopped_by"] = "response_length"
        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]

        # Terminal quality in the labels' measure (§2.1/§2.4): EM headline for
        # QA; scored HERE (training-time GT is allowed) from the final answer —
        # or the last kept draft when the window truncated the episode.
        gold = str(task.get("gold") or "")
        answer = result.final_answer if not truncated else (
            traj.steps[-1].draft if traj.steps else "")
        traj.outcome["Q_tau"] = qa_quality(answer, gold, self.terminal_metric)
        traj.outcome["Q_tau_f1"] = qa_quality(answer, gold, "f1")
        traj.outcome["success"] = bool(traj.outcome["Q_tau"] >= 0.5)

        output = AgentLoopOutput(
            prompt_ids=list(bridge.turns[0][0]) if bridge.turns else [],
            response_ids=response_ids,
            response_mask=response_mask,
            num_turns=2 * len(bridge.turns) + 1,
            metrics=AgentLoopMetrics(**{k: v for k, v in metrics.items()
                                        if k in AgentLoopMetrics.model_fields}),
        )
        # reward_score=0.0 makes the streaming reward worker a no-op (it only
        # fires when reward_score is None — pin: agent_loop.py:843-845); the
        # scalar it would place at agent_loop.py:962-970 is overwritten wholesale
        # by CassiAgentLoopManager.
        output.reward_score = 0.0
        output.extra_fields["cassi"] = {
            "trajectory": traj.to_dict(),
            "step_ends": [int(e) for e in step_ends],
            "uid": uid,
            "truncated": truncated,
        }
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})  # schema parity
        return output


# ================================================ 4. group reward/adv manager
class CassiAgentLoopManager(AgentLoopManager):
    """Batch-level post-rollout hook: computes Algorithm 3's rewards/advantages
    per GRPO group and writes them into `batch["rm_scores"]` in the encoding the
    registered estimator decodes (module docstring).

    Loaded by fqn from `actor_rollout_ref.rollout.agent.agent_loop_manager_class`
    (pin: ray_trainer.py:929-935) inside the TaskRunner actor — importing this
    module there is also what registers the adv estimator in the process that
    later runs `compute_advantage` (ray_trainer.py:1625-1632, "executed on the
    driver process")."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sidecar = load_sidecar()
        self.cassi_cfg = load_config(self.sidecar["config_path"])
        algo = self.config.algorithm
        # Exact decode requires token_level_rewards == rm_scores
        # (pin: ray_trainer.py:1600-1607).
        if algo.get("use_kl_in_reward", False):
            raise ValueError("CASSI requires algorithm.use_kl_in_reward=false — "
                             "the KL anchor is the actor-side kl_loss (β=0.04, §17).")
        if algo.get("adv_estimator") != CASSI_ADV_ESTIMATOR:
            raise ValueError(f"algorithm.adv_estimator must be {CASSI_ADV_ESTIMATOR!r}, "
                             f"got {algo.get('adv_estimator')!r} (§2.4: verl's grpo "
                             "estimator is trajectory-level and provably shaping-blind).")
        self.lam = float(self.sidecar["lam"])
        self.stopper = StopperValueService(
            self.sidecar["coach_dir"],
            device=self.sidecar.get("stopper_device", "cpu"),
        )
        out_dir = Path(self.sidecar["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        self.divergence_csv = out_dir / "divergence.csv"   # F6 dashboard (§2.4/§16 P6)

    # Parent method is async under @auto_await (pin: agent_loop.py:1101-1127 +
    # verl/utils/ray_utils.py:97-130) — calling it from this sync override runs
    # it to completion in both loop/no-loop contexts.
    def generate_sequences(self, prompts):
        output = super().generate_sequences(prompts)
        self._apply_cassi_rewards(
            output,
            validate=bool(prompts.meta_info.get("validate", False)),
            global_steps=int(prompts.meta_info.get("global_steps", -1)),
        )
        return output

    def _apply_cassi_rewards(self, output, *, validate: bool, global_steps: int) -> None:
        payloads = output.non_tensor_batch.get("cassi")
        if payloads is None:
            raise RuntimeError(
                "rollout batch carries no 'cassi' extra fields — the agent loop must "
                f"be {CASSI_AGENT_LOOP_NAME!r} (check rollout.agent.agent_loop_config_path "
                "and the dataset's agent_name column, §16 P6)."
            )
        trajectories = [Trajectory.from_dict(p["trajectory"]) for p in payloads]
        response_mask = output.batch["response_mask"]
        width = int(response_mask.shape[1])
        rm_scores = torch.zeros(response_mask.shape, dtype=torch.float32)

        v_hats = self.stopper.values_for(trajectories, self.lam)

        # GRPO groups = shared prompt uid: assigned per prompt then repeated
        # n times interleaved (pin: ray_trainer.py:1437-1447) and kept in the
        # gen batch by _get_gen_batch (pin: ray_trainer.py:575-590).
        groups: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(payloads):
            groups[p["uid"]].append(i)

        divergences, base_rewards_all, q_taus, guarded = [], [], [], 0
        for idxs in groups.values():
            group_trajs = [trajectories[i] for i in idxs]
            group_vhats = [v_hats[i] for i in idxs]
            rewards = compute_cassi_rewards(group_trajs, group_vhats, self.cassi_cfg)
            for k, i in enumerate(idxs):
                ends = payloads[i]["step_ends"]
                if validate:
                    # Validation batches feed metrics, not advantages: report the
                    # TRUE economic terminal reward on the final step token so
                    # val_reward stays interpretable.
                    row = torch.zeros(width, dtype=torch.float32)
                    if ends:
                        row[ends[-1]] = float(rewards.terminal_rewards[k])
                    rm_scores[i] = row
                else:
                    rm_scores[i] = encode_step_values(rewards.advantages[k], ends, width)
            divergences.append(hacking_divergence(
                np.array(rewards.v_hat_terminal), np.array(rewards.base_rewards)))
            base_rewards_all.extend(rewards.base_rewards)
            q_taus.extend(float(t.outcome["Q_tau"]) for t in group_trajs)
            guarded += rewards.guarded_steps

        output.batch["rm_scores"] = rm_scores
        if not validate:
            self._log_divergence(global_steps, divergences, base_rewards_all, q_taus,
                                 guarded, len(trajectories))

    def _log_divergence(self, step, divergences, base_rewards, q_taus, guarded, n) -> None:
        """F6: V̂-vs-realized-reward divergence, logged from step 0 (§2.4/§16 P6);
        a rising curve triggers the stopper refresh (§2.7)."""
        new = not self.divergence_csv.exists()
        with self.divergence_csv.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["global_step", "hacking_divergence", "mean_R_base",
                            "mean_Q_tau", "guarded_steps", "n_trajectories"])
            w.writerow([step, float(np.mean(divergences)), float(np.mean(base_rewards)),
                        float(np.mean(q_taus)), guarded, n])
