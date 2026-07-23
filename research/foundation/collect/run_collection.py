"""Trajectory collection CLI (F2) — THE trajectory script.

Examples:
  # pilot (forced continuation, no budget pressure recorded as large-B)
  .venv/bin/python -m collect.run_collection --task-file data/hotpotqa_train_300.jsonl \
      --limit 50 --arm a1 --mode forced_continuation --g 4 --out experiments/pilot.jsonl
  # baseline eval
  .venv/bin/python -m collect.run_collection --task-file data/hotpotqa_dev_200.jsonl \
      --arm a2 --mode enforce --budget medium --g 1 --temperature 0 --out ...

Budget policy: --budget {small,medium,large} fixes one wallet; --budget draw
draws one per (task, group) from all three, seeded (v2.1 §2.2 rule: shared
across the group's G rollouts). Resumable: completed (task_id, rollout) pairs
are skipped on rerun.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.harness import EpisodeSpec, run_episode
from agent.llm_client import OpenAIChat
from collect.sampling import load_jsonl
from collect.schema import validate_episode
from common import FOUNDATION_ROOT, config_hash, load_config, write_run_stamp
from envs.retrieval_client import RetrievalClient


def draw_budget(policy: str, budgets: dict[str, int], task_id: str, seed: int) -> int:
    if policy in budgets:
        return budgets[policy]
    if policy == "draw":
        rng = random.Random(f"{seed}:{task_id}")
        return budgets[rng.choice(sorted(budgets))]
    raise ValueError(f"unknown budget policy: {policy}")


def completed_keys(out_path: Path) -> set[tuple[str, int]]:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    ep = json.loads(line)
                    done.add((ep["task_id"], ep.get("rollout", 0)))
    return done


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--arm", required=True, choices=["a0", "a1", "a2", "a3"])
    ap.add_argument("--mode", required=True,
                    choices=["none", "enforce", "forced_continuation"])
    ap.add_argument("--budget", default="draw",
                    help="small|medium|large|draw (per task+group, seeded)")
    ap.add_argument("--g", type=int, default=1, help="rollouts per task")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--train", action="store_true",
                    help="capture messages+logprobs for the trainer (F5)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = load_config()
    ep_cfg, ex_cfg = cfg["episode"], cfg["executor"]
    temp = (args.temperature if args.temperature is not None
            else ex_cfg["rollout_temperature"] if args.g > 1
            else ex_cfg["eval_temperature"])
    tasks = load_jsonl(args.task_file)[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(out_path)
    write_run_stamp(out_path.parent, cfg,
                    {"cli": vars(args), "n_tasks": len(tasks)})

    llm = OpenAIChat(ex_cfg["endpoint"], ex_cfg["model"],
                     max_tokens=ex_cfg["max_tokens_per_step"],
                     extra_body={"chat_template_kwargs":
                                 {"enable_thinking": ex_cfg["enable_thinking"]}})
    retr = RetrievalClient(cfg["retrieval"]["endpoint"], cfg["retrieval"]["top_k"])
    chash = config_hash(cfg)

    n_run = n_skip = 0
    with open(out_path, "a") as fout:
        for task in tasks:
            budget = draw_budget(args.budget, ep_cfg["budgets"],
                                 task["id"], cfg["seed"])
            for r in range(args.g):
                if (task["id"], r) in done:
                    n_skip += 1
                    continue
                spec = EpisodeSpec(
                    task_id=task["id"], question=task["question"],
                    golds=task["answers"], arm=args.arm, mode=args.mode,
                    budget=budget, t_max=ep_cfg["t_max"], temperature=temp,
                    seed=cfg["seed"], config_hash=chash,
                    draft_retry=ep_cfg["draft_retry"], train_mode=args.train)
                ep = run_episode(spec, llm, retr)
                ep["rollout"] = r
                validate_episode(ep)
                fout.write(json.dumps(ep, ensure_ascii=False) + "\n")
                fout.flush()
                n_run += 1
    print(f"collected={n_run} skipped(resume)={n_skip} -> {out_path}")


if __name__ == "__main__":
    main()
