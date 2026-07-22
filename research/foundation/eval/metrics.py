"""Shared metrics (F4+F6): one code path scores every arm.

Episode JSONL -> per-task rows -> aggregates with bootstrap CIs and per-task
paired deltas. Sanity checks (row counts, utility recompute) live here so both
F4 and F6 run the identical guards.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def episode_row(ep: dict, lam: float) -> dict:
    self_stopped = (ep.get("answered_at") is not None and not ep["forced_stop"]
                    and ep["answered_at"] <= ep["budget_B"])
    return {
        "task_id": ep["task_id"], "arm": ep["arm"], "mode": ep["mode"],
        "budget_B": ep["budget_B"], "rollout": ep.get("rollout", 0),
        "f1": ep["final_f1"], "em": ep["final_em"],
        "steps_used": ep["steps_used"],
        "utility": ep["final_f1"] - lam * (ep["steps_used"] / max(1, ep["budget_B"])),
        "self_stopped": float(self_stopped),
        "hit_cap": float(ep.get("answered_at") is None and not ep["forced_stop"]),
        "config_hash": ep["config_hash"],
    }


def rows_from_jsonl(path: str | Path, lam: float) -> pd.DataFrame:
    rows = [episode_row(json.loads(l), lam) for l in open(path) if l.strip()]
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10000,
                 seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(n_resamples, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(df: pd.DataFrame, n_resamples: int = 10000,
              seed: int = 42) -> pd.DataFrame:
    """Mean + 95% CI per (arm, budget) for every metric column."""
    out = []
    for (arm, b), g in df.groupby(["arm", "budget_B"]):
        row = {"arm": arm, "budget_B": b, "n": len(g)}
        for m in ("f1", "em", "steps_used", "utility", "self_stopped", "hit_cap"):
            lo, hi = bootstrap_ci(g[m].to_numpy(), n_resamples, seed)
            row[m] = g[m].mean()
            row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
        out.append(row)
    return pd.DataFrame(out).sort_values(["arm", "budget_B"]).reset_index(drop=True)


def paired_delta(df: pd.DataFrame, arm_a: str, arm_b: str, budget: int,
                 metric: str = "utility", n_resamples: int = 10000,
                 seed: int = 42) -> dict:
    """Per-task paired A-minus-B on the SAME frozen tasks (valid because every
    arm ran the identical dev list)."""
    a = df[(df.arm == arm_a) & (df.budget_B == budget)].set_index("task_id")[metric]
    b = df[(df.arm == arm_b) & (df.budget_B == budget)].set_index("task_id")[metric]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError(f"no shared tasks between {arm_a} and {arm_b} at B={budget}")
    d = (a.loc[common] - b.loc[common]).to_numpy()
    lo, hi = bootstrap_ci(d, n_resamples, seed)
    return {"arms": f"{arm_a}-{arm_b}", "budget_B": budget, "metric": metric,
            "n": len(common), "mean_delta": float(d.mean()),
            "ci_lo": lo, "ci_hi": hi,
            "frac_tasks_a_wins": float((d > 0).mean())}


# ---------- sanity checks (F6: must pass before any number is reported) -----

def check_row_counts(df: pd.DataFrame, expected_tasks: int) -> None:
    for (arm, b), g in df.groupby(["arm", "budget_B"]):
        if g.task_id.duplicated().any():
            dup = g[g.task_id.duplicated()].task_id.iloc[0]
            raise AssertionError(f"{arm}/B={b}: duplicated task {dup}")
        if len(g) != expected_tasks:
            raise AssertionError(f"{arm}/B={b}: {len(g)} rows, want {expected_tasks}")


def check_utility_recompute(df: pd.DataFrame, lam: float) -> None:
    u = df.f1 - lam * (df.steps_used / df.budget_B)
    bad = (u - df.utility).abs() > 1e-9
    if bad.any():
        raise AssertionError(f"stale utility in {int(bad.sum())} rows "
                             "(lambda/budget mismatch between file and config?)")
