"""Collection round runner — paper_plan_v2 §16 P2/P7, §2.1 (forced continuation),
§2.2 (group-shared wallets), §11 (trajectory JSONL schema).

For each task:
  * draw ONE wallet per (task, GRPO group), SHARED by all G=8 rollouts of the
    group (`cassi.budget.cost.draw_wallet` with a seeded rng — §2.2: "group
    advantages must compare behavior under the same wallet, never confound
    behavior with wallet luck");
  * run the shared ReAct scaffold in FORCED-CONTINUATION mode (§2.1: ANSWER is
    logged as a draft event with answered_flag=True and the rollout runs to
    T_max — no censoring; the logged positions double as the free self-stop
    measurement);
  * score per-step draft quality vs gold / env subgoals at collection time ONLY
    (§2.6 — q_t never enters x_t);
  * write trajectories as §11-schema JSONL and report the running-draft token
    share + forced-continuation overhead in dollars (T4 accounting, §5.3).

`run_pilot` is the §16 P2 200-task unconstrained pilot whose spend list feeds
`cassi.budget.cost.calibrate_wallets` (wallet sizes + the C̃ normalization
median are frozen into configs/cassi.yaml afterwards).

CLI:
    python -m cassi.executor.collect --config configs/cassi.yaml --domain qa \
        --tasks tasks.jsonl --out rollouts.jsonl [--pilot]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cassi.budget.cost import calibrate_wallets, draw_wallet
from cassi.common.config import load_config, require_pilot_calibration
from cassi.common.schema import Trajectory, save_trajectories
from cassi.executor.react_agent import EpisodeResult, LLMClient, ReactAgent
from cassi.labels.quality import qa_quality

UNCONSTRAINED_ALLOWANCE = 1e9      # pilot wallets: effectively infinite (tier stays HIGH)


# ------------------------------------------------------------- scoring helpers
def score_trajectory(result: EpisodeResult, task: dict, env, *, domain: str) -> None:
    """Fill q_t on every step and Q_τ/success in the outcome — collection time
    ONLY (§2.1/§2.6). q_t scores the RUNNING DRAFT after step t; Q_τ is the
    terminal quality at the first ANSWER step (or the last step at T_max),
    in the labels' own measure (§2.4: QA headline EM; ALFWorld success/subgoals)."""
    traj = result.trajectory
    for step, info in zip(traj.steps, result.step_infos):
        step.q = env.step_quality(step.draft, task, info)

    tau = traj.outcome.get("tau")
    idx = (tau - 1) if tau else (len(traj.steps) - 1)
    if not traj.steps:
        traj.outcome["Q_tau"], traj.outcome["success"] = 0.0, False
        return
    if domain == "qa":
        gold = task.get("gold")
        answer = traj.outcome.get("final_answer") or traj.steps[idx].draft
        q_tau = qa_quality(answer, gold, metric="em") if gold is not None else 0.0
        traj.outcome["Q_tau"] = q_tau
        traj.outcome["success"] = bool(q_tau >= 1.0)
    else:                                # alfworld: env subgoal fraction / win flag
        traj.outcome["Q_tau"] = traj.steps[idx].q
        won = any(bool(i.get("won")) for i in result.step_infos[: idx + 1])
        traj.outcome["success"] = won or traj.steps[idx].q >= 1.0


def forced_continuation_overhead(traj: Trajectory) -> float:
    """Dollars spent strictly AFTER the first answered_flag step — the collection
    overhead the forced-continuation mode buys labels with (T4, §5.3)."""
    tau = traj.outcome.get("tau")
    if not tau:
        return 0.0
    return float(sum(s.c for s in traj.steps[tau:]))


# ------------------------------------------------------------------ pilot (P2)
def run_pilot(
    tasks: list[dict], llm: LLMClient, env, cfg: dict, *,
    domain: str = "qa", t_max: int | None = None, seed: int = 0, iteration: int = 0,
) -> list[float]:
    """§16 P2 unconstrained pilot: natural ("rl"-mode) rollouts, one per task,
    effectively infinite wallet. Returns per-task total spends — feed them to
    `cassi.budget.cost.calibrate_wallets` and freeze the result into §17."""
    t_max = t_max or int(cfg["executor"]["horizon"][domain])
    agent = ReactAgent(llm)
    spends: list[float] = []
    for task in tasks:
        res = agent.run(
            task, env, mode="rl", t_max=t_max,
            allowance_dollars=UNCONSTRAINED_ALLOWANCE, wallet_size="large",
            group_id=f"pilot:{task.get('task_id', '')}", rollout_idx=0,
            seed=seed, iteration=iteration,
        )
        spends.append(float(sum(s.c for s in res.trajectory.steps)))
    return spends


# --------------------------------------------------------- collection round (P2/P7)
def collect_round(
    tasks: list[dict], llm: LLMClient, env, cfg: dict, *,
    domain: str = "qa", out_path: str | Path,
    allowances: dict | None = None, G: int | None = None,
    t_max: int | None = None, seed: int = 0, iteration: int = 0,
) -> dict:
    """One P2/P7 collection round. Writes §11-schema JSONL to `out_path` and
    returns the T4 accounting report. `allowances` overrides the config's
    pilot-frozen wallets (tests pass them explicitly; production reads §17)."""
    if allowances is None:
        require_pilot_calibration(cfg, domain)      # refuse to run un-calibrated (§17)
        allowances = cfg["label"]["allowances"][domain]
    G = G or int(cfg["data"]["collection"]["rollouts_per_task_G"])
    t_max = t_max or int(cfg["executor"]["horizon"][domain])

    rng = np.random.default_rng(seed)               # seeded: wallet draws reproducible
    agent = ReactAgent(llm)
    trajectories: list[Trajectory] = []
    wallet_counts = {"small": 0, "medium": 0, "large": 0}
    total_dollars = 0.0
    total_output_tokens = 0
    total_draft_tokens = 0
    overhead_dollars = 0.0

    for task in tasks:
        task_id = str(task.get("task_id", ""))
        group_id = f"{task_id}:iter{iteration}:g0"
        # ONE wallet per (task, GRPO group) — shared by all G rollouts (§2.2)
        wallet_size, allowance = draw_wallet(rng, allowances)
        wallet_counts[wallet_size] += 1
        for g in range(G):
            res = agent.run(
                task, env, mode="forced_continuation", t_max=t_max,
                allowance_dollars=allowance, wallet_size=wallet_size,
                group_id=group_id, rollout_idx=g, seed=seed, iteration=iteration,
            )
            score_trajectory(res, task, env, domain=domain)
            trajectories.append(res.trajectory)
            total_dollars += float(sum(s.c for s in res.trajectory.steps))
            total_output_tokens += res.output_tokens_total
            total_draft_tokens += res.draft_line_tokens
            overhead_dollars += forced_continuation_overhead(res.trajectory)

    n = save_trajectories(trajectories, out_path)
    return {
        "n_tasks": len(tasks),
        "n_trajectories": n,
        "G": G,
        "t_max": t_max,
        "domain": domain,
        "iteration": iteration,
        "wallet_counts": wallet_counts,
        "total_dollars": total_dollars,
        # T4 accounting (§2.6/§5.3): the label machinery's honest price tags
        "draft_token_share": (total_draft_tokens / total_output_tokens
                              if total_output_tokens else 0.0),
        "forced_continuation_overhead_dollars": overhead_dollars,
        "overhead_pct_of_total": (overhead_dollars / total_dollars
                                  if total_dollars else 0.0),
        "out_path": str(out_path),
    }


# -------------------------------------------------------------------------- CLI
def _load_tasks(path: str | Path) -> list[dict]:
    tasks = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def _build_env(domain: str, args: argparse.Namespace):
    if domain == "qa":
        from cassi.executor.envs.searchr1_qa import SearchR1QAEnv
        return SearchR1QAEnv(retriever_url=args.retriever_url, topk=args.topk)
    if domain == "alfworld":
        from cassi.executor.envs.alfworld import ALFWorldEnv
        return ALFWorldEnv()
    raise ValueError(f"unknown domain {domain!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="CASSI collection round (paper_plan_v2 §16 P2/P7)")
    p.add_argument("--config", default=None, help="configs/cassi.yaml (§17)")
    p.add_argument("--domain", choices=["qa", "alfworld"], default="qa")
    p.add_argument("--tasks", required=True, help="tasks JSONL: {task_id, question, gold}")
    p.add_argument("--out", required=True, help="output trajectories JSONL (§11 schema)")
    p.add_argument("--pilot", action="store_true",
                   help="run the 200-task unconstrained pilot instead (§16 P2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iteration", type=int, default=0, help="loop iteration i (§2.7)")
    p.add_argument("--G", type=int, default=None)
    p.add_argument("--t-max", type=int, default=None)
    p.add_argument("--vllm-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--smoke", action="store_true",
                   help="P0 smoke run: provisional flat wallets, bypasses the P2 "
                        "calibration guard (§16 P0 precedes the pilot by design)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    env = _build_env(args.domain, args)
    from cassi.executor.vllm_client import VLLMClient
    llm = VLLMClient.from_config(cfg, base_url=args.vllm_url)
    tasks = _load_tasks(args.tasks)

    if args.pilot:
        spends = run_pilot(tasks, llm, env, cfg, domain=args.domain,
                           t_max=args.t_max, seed=args.seed, iteration=args.iteration)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(str(s) for s in spends) + "\n")
        calib = calibrate_wallets(spends)
        print(json.dumps({"pilot_n": len(spends), "calibration": calib,
                          "note": "freeze these into configs/cassi.yaml "
                                  "label.allowances + cost_normalization (§17)"},
                         indent=2))
        return 0

    smoke_allowances = ({"small": 0.05, "medium": 0.05, "large": 0.05}
                        if args.smoke else None)   # provisional; NEVER for real collection
    report = collect_round(tasks, llm, env, cfg, domain=args.domain,
                           out_path=args.out, G=args.G, t_max=args.t_max,
                           seed=args.seed, iteration=args.iteration,
                           allowances=smoke_allowances)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
