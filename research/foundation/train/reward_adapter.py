"""Batch reward computation for RL rollouts (F5) + the hacking diagnostic.

Sits between rollout collection and the optimizer: episodes (our schema) in,
per-step advantages out. Also accumulates the judge-score-vs-realized-F1
divergence series — the run's most important health curve (frozen judge!).
"""

import json
from pathlib import Path

import numpy as np

from reward.rewards import episode_rewards
from reward.rubric import STEP_BITS, step_score
from train.advantages import group_step_advantages


class DivergenceLog:
    """Per-batch mean judge step-score vs mean realized F1. Rising judge score
    with flat/falling F1 = the agent is learning to please the judge (Goodhart);
    plotted as Fig 3 in F7."""

    def __init__(self):
        self.batches: list[dict] = []

    def add(self, judge_scores: list[float], f1s: list[float], step: int) -> dict:
        row = {"step": step,
               "judge_score_mean": float(np.mean(judge_scores)) if judge_scores else 0.5,
               "f1_mean": float(np.mean(f1s)) if f1s else 0.0}
        self.batches.append(row)
        return row

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for row in self.batches:
                f.write(json.dumps(row) + "\n")


def batch_rewards(groups: list[list[dict]], judge, cfg: dict,
                  divergence: DivergenceLog | None = None,
                  train_step: int = 0) -> list[list[dict]]:
    """groups: per-task lists of G episode dicts (same wallet within a group).

    Returns the same nesting where each episode gains:
      step_rewards, r_final, returns_to_go, advantages (aligned with steps).
    """
    judge_scores: list[float] = []
    f1s: list[float] = []
    out: list[list[dict]] = []
    for group in groups:
        enriched = []
        for ep in group:
            rew = episode_rewards(ep, judge, cfg)
            enriched.append({**ep, **rew})
            f1s.append(ep["final_f1"])
            for s, bits in zip(ep["steps"], rew["bits"]):
                if s["action_type"] == "search":
                    judge_scores.append(step_score(bits, cfg["rubric"]["step_bits"]))
        advs = group_step_advantages([e["returns_to_go"] for e in enriched],
                                     cfg["grpo"]["min_cohort"])
        for e, a in zip(enriched, advs):
            e["advantages"] = a
        out.append(enriched)
    if divergence is not None:
        divergence.add(judge_scores, f1s, train_step)
    return out
