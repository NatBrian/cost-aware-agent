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

    def add(self, judge_scores: list[float], f1s: list[float], step: int,
            scope: str = "round") -> dict:
        row = {"step": step, "scope": scope,
               "judge_score_mean": float(np.mean(judge_scores)) if judge_scores else 0.5,
               "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
               "n": len(judge_scores)}
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
    from concurrent.futures import ThreadPoolExecutor
    judge_scores: list[float] = []
    f1s: list[float] = []
    flat = [(gi, ep) for gi, group in enumerate(groups) for ep in group]
    with ThreadPoolExecutor(max_workers=12) as pool:   # judge is the bottleneck
        rewards = list(pool.map(lambda t: episode_rewards(t[1], judge, cfg), flat))
    enriched_by_group: dict[int, list[dict]] = {}
    for (gi, ep), rew in zip(flat, rewards):
        enriched_by_group.setdefault(gi, []).append({**ep, **rew})
        f1s.append(ep["final_f1"])
        for s, bits in zip(ep["steps"], rew["bits"]):
            if s["action_type"] == "search":
                judge_scores.append(step_score(bits, cfg["rubric"]["step_bits"]))
    out: list[list[dict]] = []
    for gi in sorted(enriched_by_group):
        enriched = enriched_by_group[gi]
        advs = group_step_advantages([e["returns_to_go"] for e in enriched],
                                     cfg["grpo"]["min_cohort"])
        for e, a in zip(enriched, advs):
            e["advantages"] = a
        out.append(enriched)
        # Per-group rows too: batch_rewards is called ONCE per round, so
        # round-level rows alone give Fig 3 one point per round (3 total for the
        # whole run) — not a curve. Group rows give ~300/round, enough to see
        # judge score and realized F1 diverge WITHIN a round (audit 2026-07-28).
        if divergence is not None:
            g_scores = [step_score(bits, cfg["rubric"]["step_bits"])
                        for e in enriched
                        for s, bits in zip(e["steps"], e["bits"])
                        if s["action_type"] == "search"]
            divergence.add(g_scores, [e["final_f1"] for e in enriched],
                           train_step, scope=f"group:{gi}")
    if divergence is not None:
        divergence.add(judge_scores, f1s, train_step, scope="round")
    return out
