"""GRPO training launcher (F5).

Modes:
  --dry-run   CPU, no verl/GPU: fabricated rollout batch flows through the FULL
              reward path (judge mocked neutral) -> advantages + divergence log.
              Gate 1 of F5; must stay green in CI.
  (default)   Real training: requires the GPU env (requirements-gpu.txt, verl
              pinned) and live executor/judge servers. Rollouts run harness mode
              'none' (real termination economics); groups share wallets; step
              advantages from train/advantages.py feed verl's policy-gradient
              update with Dr. GRPO length normalization and KL anchor from
              config. Verified at the micro-run (E-d), not before.

Integration note (verl >= 0.8): we drive verl's AgentLoop with our episode
loop via its custom-rollout hook and inject per-token advantages by
broadcasting each step's advantage over that step's tokens. The glue lives in
_run_real() and is deliberately isolated here — everything above it is
verl-free and fully tested on CPU.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config, write_run_stamp
from train.reward_adapter import DivergenceLog, batch_rewards


class NeutralJudge:
    """Dry-run stand-in: every bit 1 for odd steps, 0 for even — non-degenerate
    on purpose so the dry-run catches all-identical-reward wiring bugs."""

    def __init__(self):
        self.n = 0

    def judge(self, prompt, bit_names):
        self.n += 1
        return {b: self.n % 2 for b in bit_names}


def _fake_group(task_id: str, g: int, budget: int) -> list[dict]:
    eps = []
    for r in range(g):
        n_steps = 2 + (r % 3)
        steps = [{"t": t + 1, "action_type": "search",
                  "query_or_answer": f"q{t}", "obs_digest": f"obs {t}",
                  "draft": "d", "draft_f1_vs_gold": 0.3}
                 for t in range(n_steps - 1)]
        steps.append({"t": n_steps, "action_type": "answer",
                      "query_or_answer": "ans", "obs_digest": "",
                      "draft": "ans", "draft_f1_vs_gold": 0.5})
        eps.append({"task_id": task_id, "question": "fake?", "arm": "a3",
                    "mode": "none", "budget_B": budget, "seed": 42,
                    "config_hash": "dry", "steps": steps, "answered_at": n_steps,
                    "forced_stop": False, "final_answer": "ans",
                    "final_f1": 0.4 + 0.1 * r, "final_em": 0.0,
                    "steps_used": n_steps, "total_steps_run": n_steps,
                    "rollout": r})
    return eps


def dry_run(cfg: dict) -> dict:
    div = DivergenceLog()
    groups = [_fake_group(f"task{i}", cfg["grpo"]["group_size"],
                          cfg["episode"]["budgets"]["medium"]) for i in range(3)]
    out = batch_rewards(groups, NeutralJudge(), cfg, div, train_step=0)
    n_eps = sum(len(g) for g in out)
    all_advs = [a for g in out for e in g for a in e["advantages"]]
    assert n_eps == 3 * cfg["grpo"]["group_size"]
    assert any(abs(a) > 1e-9 for a in all_advs), "degenerate advantages"
    for g in out:
        for e in g:
            assert len(e["advantages"]) == len(e["steps"]) == len(e["step_rewards"])
    return {"episodes": n_eps, "nonzero_advantages": True,
            "divergence_rows": len(div.batches)}


def _run_real(cfg: dict, args) -> None:
    try:
        import verl  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "verl not installed — real training needs the GPU env "
            "(requirements-gpu.txt; pin at install). Dry-run works without it."
        ) from e
    raise SystemExit(
        "real-training glue is wired at the E-d micro-run against the pinned "
        "verl version (see F5 doc gates). Run --dry-run for the CPU path.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="experiments/results/train")
    args = ap.parse_args()
    cfg = load_config()
    if args.dry_run:
        result = dry_run(cfg)
        print(f"DRY-RUN OK: {result}")
        return
    write_run_stamp(args.out, cfg)
    _run_real(cfg, args)


if __name__ == "__main__":
    main()
