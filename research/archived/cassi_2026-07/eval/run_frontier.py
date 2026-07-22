"""Frontier runner CLI — the evaluation entry point for P5–P9 (paper_plan_v2 §5.3).

Rolls a policy over a frozen task list (one operating point = one arm at one knob
setting), with an optional stopper monitor (Alg. 4), and APPENDS one summary row

    arm,lambda_dial,accuracy,cost_dollars[,extras...]

to --out-csv, plus per-instance rows to <out-csv stem>_instances.csv — the stats
layer (§5.6) consumes per-instance matrices, never aggregates.

Billing symmetry (§5.3): cost_dollars is ALL-INCLUSIVE — the episode's own c_t
(which already contains the running-draft tokens, §2.6) PLUS the stopper/monitor's
own inference, priced with the same map (budget/cost.py). Regret replays are billed
to the analysis line (T4), never to the method.

Modes:
  * normal        — one RL-mode episode per task (ANSWER terminates)
  * --with-regret — adds a forced-continuation REPLAY per task (dual-run protocol
                    §5.3) and reports utility regret vs the replay's U_t frontier
  * --regret-from-replays — offline aggregation over stored replay JSONLs

The heavy lifting is in `evaluate_arm(...)`, which takes constructed objects and
is CPU-testable with mocks; `main()` only wires vLLM/retriever/stopper endpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from cassi.budget.cost import stopping_utilities, token_cost
from cassi.common.config import load_config, require_pilot_calibration
from cassi.labels.quality import qa_quality

# one stopper query ≈ serialized x_t prefill + tiny decode, at 2B-model prices ≈
# executor reference prices (conservative: same price map — §5.3 billing symmetry)
STOPPER_QUERY_INPUT_TOKENS = 600
STOPPER_QUERY_OUTPUT_TOKENS = 8

SUMMARY_FIELDS = [
    "arm", "lambda_dial", "accuracy", "cost_dollars",
    "domain", "seed", "n_tasks", "wallet", "stopper_every_k",
    "self_termination_rate", "monitor_stop_rate", "mean_regret",
    "stopper_cost_dollars", "replay_cost_dollars_analysis_line",
]
INSTANCE_FIELDS = [
    "arm", "lambda_dial", "domain", "seed", "task_id",
    "correct", "accuracy", "cost_dollars", "tau", "stopped_by", "regret",
]


def _append_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _stopper_query_cost(n_queries: int) -> float:
    return n_queries * token_cost(STOPPER_QUERY_INPUT_TOKENS, STOPPER_QUERY_OUTPUT_TOKENS)


def _episode_accuracy(result, task: dict, domain: str) -> tuple[float, float]:
    """(correct ∈ {0,1} by the headline metric, graded accuracy). QA headline = EM,
    graded = F1 (§2.4/§17 quality_scoring); ALFWorld = env success both ways."""
    if domain == "qa":
        answer = result.final_answer or (result.trajectory.steps[-1].draft
                                         if result.trajectory.steps else "")
        gold = task.get("gold", "")
        return qa_quality(answer, gold, "em"), qa_quality(answer, gold, "f1")
    success = float(result.trajectory.outcome.get("success", 0.0))
    return success, success


def _utility_regret(replay_traj, task: dict, domain: str, method_utility: float,
                    lam: float, median_spend: float) -> float:
    """§5.3 stopping regret (utility gap): replay-frontier max_t U_t − the method's
    actual-stop utility. The replay supplies the counterfactual U curve; the
    method's own run supplies where it actually stopped and what that was worth.

    Rollouts log q=0 by design (§2.1: quality is label machinery) — the replay is
    an ANALYSIS run, so we score its per-step drafts vs gold here (GT allowed)."""
    if domain == "qa":
        q = [qa_quality(s.draft, task.get("gold", ""), "f1") for s in replay_traj.steps]
    else:
        q = [s.q for s in replay_traj.steps]     # ALFWorld: env fills subgoal fraction
    u = stopping_utilities(
        q, [s.c for s in replay_traj.steps],
        [s.tier for s in replay_traj.steps], lam, median_spend,
    )
    return float(u.max() - method_utility)


def evaluate_arm(
    *, agent, env, tasks: list[dict], monitor=None, monitor_factory=None,
    arm: str, lambda_dial: float, domain: str, seed: int,
    t_max: int, allowance_dollars: float, wallet: str = "medium",
    stopper_every_k: int = 1, self_termination: bool = False,
    with_regret: bool = False, lam_economy: float = 1.0,
    median_pilot_spend: float | None = None,
) -> tuple[dict, list[dict]]:
    """Run one operating point. Returns (summary_row, instance_rows).

    `monitor_factory()` (if given) builds a fresh monitor per episode so
    per-episode stats don't leak; `monitor` is used as-is otherwise (tests)."""
    inst_rows: list[dict] = []
    accs, costs, regrets = [], [], []
    self_term, mon_stop, stopper_cost, replay_cost = 0, 0, 0.0, 0.0

    for i, task in enumerate(tasks):
        mon = monitor_factory() if monitor_factory else monitor
        result = agent.run(
            task, env, mode="rl", t_max=t_max,
            allowance_dollars=allowance_dollars, wallet_size=wallet,
            group_id=f"eval_{arm}_{i}", rollout_idx=0, monitor=mon, seed=seed,
        )
        correct, acc = _episode_accuracy(result, task, domain)
        ep_cost = sum(s.c for s in result.trajectory.steps)
        sc = _stopper_query_cost(getattr(mon, "n_queries", 0)) if mon else 0.0
        stopper_cost += sc
        total_cost = ep_cost + sc                       # billing symmetry (§5.3)
        stopped_by = result.stopped_by or ""
        if stopped_by == "answer":
            self_term += 1
        elif stopped_by == "monitor":
            mon_stop += 1

        regret = ""
        if with_regret:
            if median_pilot_spend is None:
                raise ValueError("--with-regret needs the P2-calibrated median spend (§2.1)")
            replay = agent.run(
                task, env, mode="forced_continuation", t_max=t_max,
                allowance_dollars=allowance_dollars, wallet_size=wallet,
                group_id=f"replay_{arm}_{i}", rollout_idx=0, monitor=None, seed=seed,
            )
            replay_cost += sum(s.c for s in replay.trajectory.steps)
            tau = len(result.trajectory.steps)
            # method utility = terminal quality − tier-scaled normalized spend (§2.4)
            u_method = stopping_utilities([correct] * tau,
                                          [s.c for s in result.trajectory.steps],
                                          [s.tier for s in result.trajectory.steps],
                                          lam_economy, median_pilot_spend)
            method_u = float(u_method[-1])
            regret = _utility_regret(replay.trajectory, task, domain, method_u,
                                     lam_economy, median_pilot_spend)
            regrets.append(regret)

        accs.append(acc)
        costs.append(total_cost)
        inst_rows.append({
            "arm": arm, "lambda_dial": lambda_dial, "domain": domain, "seed": seed,
            "task_id": task.get("task_id", str(i)), "correct": correct,
            "accuracy": acc, "cost_dollars": round(total_cost, 6),
            "tau": len(result.trajectory.steps), "stopped_by": stopped_by,
            "regret": regret,
        })

    n = len(tasks)
    summary = {
        "arm": arm, "lambda_dial": lambda_dial,
        "accuracy": round(float(np.mean(accs)), 4) if n else "",
        "cost_dollars": round(float(np.mean(costs)), 6) if n else "",
        "domain": domain, "seed": seed, "n_tasks": n, "wallet": wallet,
        "stopper_every_k": stopper_every_k,
        "self_termination_rate": round(self_term / n, 4) if n and self_termination else "",
        "monitor_stop_rate": round(mon_stop / n, 4) if n else "",
        "mean_regret": round(float(np.mean(regrets)), 4) if regrets else "",
        "stopper_cost_dollars": round(stopper_cost, 6),
        "replay_cost_dollars_analysis_line": round(replay_cost, 6),  # T4 analysis line, NOT method cost
    }
    return summary, inst_rows


# --------------------------------------------------------------------- wiring
def _load_tasks(path: str, seed: int, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).open() if l.strip()]
    if limit:
        import random
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def _build_monitor_factory(args, cfg):
    if args.monitor in (None, "", "none", "off"):
        return None
    from cassi.executor.monitor import monitor_from_config
    from cassi.stopper.model import load_predictor  # lazy: torch
    predictor = load_predictor(args.monitor)
    def factory():
        return monitor_from_config(
            predictor, cfg, lam=args.lambda_dial,
            t_max=cfg["executor"]["horizon"][args.domain], domain=args.domain,
            ablation_a8=args.rule_table,
        )
    return factory


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--policy", required=True, help="model name/path served by vLLM (--vllm-url)")
    ap.add_argument("--monitor", default="none", help="stopper checkpoint dir | none")
    ap.add_argument("--lambda-dial", type=float, required=True)
    ap.add_argument("--domain", choices=["qa", "alfworld"], default="qa")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--wallet", default="medium", choices=["small", "medium", "large"])
    ap.add_argument("--stopper-every-k", type=int, default=1)
    ap.add_argument("--self-termination", action="store_true")
    ap.add_argument("--with-regret", action="store_true")
    ap.add_argument("--rule-table", action="store_true", help="ablation A8 δ(tier) mode")
    ap.add_argument("--limit", type=int, default=None, help="subsample tasks (seeded)")
    ap.add_argument("--vllm-url", default=os.environ.get("CASSI_VLLM_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--retriever-url", default=os.environ.get("CASSI_RETRIEVER_URL", "http://127.0.0.1:8001/retrieve"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--regret-from-replays", nargs="*", metavar="REPLAY_JSONL",
                    help="offline mode: aggregate stored replay trajectories into regret rows")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.regret_from_replays is not None:
        return _regret_from_replays(args, cfg)

    require_pilot_calibration(cfg, args.domain)
    allowances = cfg["label"]["allowances"][args.domain]
    median_spend = cfg["label"]["cost_normalization"][f"{args.domain}_median_pilot_spend"]

    from cassi.executor.react_agent import ReactAgent
    from cassi.executor.vllm_client import VLLMClient
    if args.domain == "qa":
        from cassi.executor.envs.searchr1_qa import SearchR1QAEnv
        env = SearchR1QAEnv(retriever_url=args.retriever_url)
    else:
        from cassi.executor.envs.alfworld import AlfworldEnv
        env = AlfworldEnv()

    agent = ReactAgent(VLLMClient(base_url=args.vllm_url, model=args.policy,
                                  temperature=float(cfg["executor"]["grpo"]["eval_temp"])))
    tasks = (_load_tasks(args.tasks, args.seed, args.limit)
             if args.tasks != "verl_agent_test" else [{"task_id": f"alf_{i}"} for i in range(134)])

    summary, inst = evaluate_arm(
        agent=agent, env=env, tasks=tasks,
        monitor_factory=_build_monitor_factory(args, cfg),
        arm=args.arm, lambda_dial=args.lambda_dial, domain=args.domain,
        seed=args.seed, t_max=int(cfg["executor"]["horizon"][args.domain]),
        allowance_dollars=float(allowances[args.wallet]), wallet=args.wallet,
        stopper_every_k=args.stopper_every_k, self_termination=args.self_termination,
        with_regret=args.with_regret,
        lam_economy=float(cfg["executor"]["training_lambda"]),
        median_pilot_spend=float(median_spend),
    )
    out = Path(args.out_csv)
    _append_csv(out, SUMMARY_FIELDS, [summary])
    _append_csv(out.with_name(out.stem + "_instances.csv"), INSTANCE_FIELDS, inst)
    print(f"[run_frontier] {args.arm} λ={args.lambda_dial}: "
          f"acc={summary['accuracy']} cost=${summary['cost_dollars']} → {out}")
    return 0


def _regret_from_replays(args, cfg) -> int:
    """Offline: read stored replay JSONLs (§5.3 dual-run protocol, P9) and emit
    per-trajectory regret rows keyed by task_id for joining with method instances."""
    from cassi.common.schema import load_trajectories
    require_pilot_calibration(cfg, args.domain)
    median = cfg["label"]["cost_normalization"][f"{args.domain}_median_pilot_spend"]
    lam = float(cfg["executor"]["training_lambda"])
    rows = []
    for p in args.regret_from_replays:
        for tr in load_trajectories(p):
            u = stopping_utilities([s.q for s in tr.steps], [s.c for s in tr.steps],
                                   [s.tier for s in tr.steps], lam, float(median))
            rows.append({"arm": args.arm, "lambda_dial": args.lambda_dial,
                         "domain": args.domain, "seed": args.seed,
                         "task_id": tr.task_id, "correct": "", "accuracy": "",
                         "cost_dollars": "", "tau": int(np.argmax(u)) + 1,
                         "stopped_by": "replay_frontier", "regret": float(u.max())})
    _append_csv(Path(args.out_csv), INSTANCE_FIELDS, rows)
    print(f"[run_frontier] wrote {len(rows)} replay-frontier rows → {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
