"""Shared metrics (F4+F6): one code path scores every arm.

Episode JSONL -> per-task rows -> aggregates with bootstrap CIs and per-task
paired deltas. Sanity checks (row counts, utility recompute) live here so both
F4 and F6 run the identical guards.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# The mode each arm is DEFINED by. F6 collects A3 in several modes (harness-off,
# harness-on, oracle replay) and they all carry arm="a3", so any selection on arm
# alone silently averages three populations. gate_check had this bug; report.py
# and figures.py had it too (the A3 row read F1 .530 / self-stop 39% instead of
# the harness-off .560 / .775). One helper, used everywhere. (2026-07-29)
GATE_MODE = {"a0": "none", "a1": "none", "a2": "enforce", "a3": "none"}


def canonical_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only each arm's defining mode — the population every headline
    number, figure and verdict must be computed on."""
    keep = [(df.arm == a) & (df["mode"] == m) for a, m in GATE_MODE.items()]
    mask = keep[0]
    for k in keep[1:]:
        mask = mask | k
    return df[mask].copy()


def episode_row(ep: dict, lam: float) -> dict:
    # self_stopped means the agent stopped ITSELF with nothing armed to stop it.
    # Requiring mode == "none" is load-bearing: in "enforce" the harness is armed
    # even on episodes it never had to cut, so counting those as self-stops
    # credits A2 with internalization it does not have; in
    # "forced_continuation" answered_at is logged while the episode keeps
    # running, so it is not a stop at all. (audit 2026-07-28)
    self_stopped = (ep["mode"] == "none" and ep.get("answered_at") is not None
                    and not ep["forced_stop"]
                    and ep["answered_at"] <= ep["budget_B"])
    return {
        "task_id": ep["task_id"], "arm": ep["arm"], "mode": ep["mode"],
        "budget_B": ep["budget_B"], "rollout": ep.get("rollout", 0),
        "answered_at": ep.get("answered_at"),   # kept so A0 can be re-scored
                                                # at budgets it never ran under
        "f1": ep["final_f1"], "em": ep["final_em"],
        "steps_used": ep["steps_used"],
        "utility": ep["final_f1"] - lam * (ep["steps_used"] / max(1, ep["budget_B"])),
        "self_stopped": float(self_stopped),
        "hit_cap": float(ep.get("answered_at") is None and not ep["forced_stop"]),
        "config_hash": ep["config_hash"],
        # --- FOUNDATION-2 primary estimand (plan v2.2 §7.5) -------------------
        # W = steps spent on an episode that returned NOTHING. Averaged over ALL
        # episodes (never conditioned on failure — conditioning would let a policy
        # look good by failing more often). "How much of my budget went to
        # nothing." 52.7% of all steps in the pilot were of this kind.
        # W is only improvable two ways: succeed more often, or abandon failures
        # faster — the target behaviour. It CAN be gamed by quitting everything,
        # which is why the gate pairs it with a paired F1 floor.
        "wasted_spend": float(ep["steps_used"]) if ep["final_f1"] <= 0 else 0.0,
        "failed": float(ep["final_f1"] <= 0),
        # abandonment = self-stopped early AND came back with nothing
        "abandoned": float(self_stopped and ep["final_f1"] <= 0),
        "tokens": float(sum(s.get("prompt_tokens", 0) + s.get("completion_tokens", 0)
                            for s in ep.get("steps", []))),
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
        for m in ("f1", "em", "steps_used", "utility", "self_stopped", "hit_cap",
                  "wasted_spend", "failed", "abandoned", "tokens"):
            if m not in g.columns:
                continue          # FOUNDATION-1 CSVs predate the new columns
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
    # A duplicated task_id (e.g. a3 present in two modes) makes the aligned
    # subtraction fan out to a many-to-many join: it does NOT raise, it silently
    # returns a mean over the wrong population while "n" reports the small
    # number. Refuse instead. (audit 2026-07-28)
    if not a.index.is_unique or not b.index.is_unique:
        raise ValueError(f"duplicate task rows for {arm_a}/{arm_b} at B={budget} "
                         "— filter by mode before pairing")
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
    # Group by mode as well as (arm, budget): F6's legitimate CSV holds a3 in
    # two modes, and grouping without mode would false-fail it — while grouping
    # WITH mode is exactly what catches a3 harness-on rows leaking into the
    # harness-off population. (audit 2026-07-28)
    for (arm, mode, b), g in df.groupby(["arm", "mode", "budget_B"]):
        if g.task_id.duplicated().any():
            dup = g[g.task_id.duplicated()].task_id.iloc[0]
            raise AssertionError(f"{arm}/{mode}/B={b}: duplicated task {dup}")
        if len(g) != expected_tasks:
            raise AssertionError(f"{arm}/{mode}/B={b}: {len(g)} rows, "
                                 f"want {expected_tasks}")


def check_utility_recompute(df: pd.DataFrame, lam: float) -> None:
    u = df.f1 - lam * (df.steps_used / df.budget_B.clip(lower=1))
    bad = (u - df.utility).abs() > 1e-9
    if bad.any():
        raise AssertionError(f"stale utility in {int(bad.sum())} rows "
                             "(lambda/budget mismatch between file and config?)")
