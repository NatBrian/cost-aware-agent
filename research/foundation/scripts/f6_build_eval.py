"""F6 — assemble the evaluation CSV the gate reads.

The A0/A1/A2 baselines survived the wipe as PER-TASK ROWS
(`experiments/results/baselines/baseline_rows.csv`), not as episodes, so they
cannot go through `eval.build_rows` (which expects episode JSONL). This script
merges the surviving rows with freshly-built A3 rows, re-scores A0 across
budgets, and runs the same guards `build_rows` runs.

Usage:
  .venv/bin/python scripts/f6_build_eval.py \
      --baselines experiments/results/baselines/baseline_rows.csv \
      --a3 experiments/results/eval/a3_*.jsonl \
      --out experiments/results/foundation_eval.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import load_config
from eval.metrics import check_row_counts, check_utility_recompute, rows_from_jsonl


def rescore_a0(df: pd.DataFrame, budgets: dict, lam: float) -> pd.DataFrame:
    """A0 sees no budget, so its behaviour cannot depend on B — plan §3 re-scores
    one run under every budget's utility. The surviving CSV has A0 at B=8 only.

    The CSV predates the `answered_at` column, so self_stopped is reconstructed:
    A0 runs with the harness disarmed, so an episode that did not hit the cap
    answered at its last step, i.e. answered_at == steps_used. Hence
    self_stopped(B) = (hit_cap == 0) and (steps_used <= B).
    """
    a0, rest = df[df.arm == "a0"], df[df.arm != "a0"]
    if a0.empty:
        return df
    out = []
    for B in sorted(set(budgets.values())):
        r = a0.copy()
        r["budget_B"] = B
        r["utility"] = r.f1 - lam * (r.steps_used / max(1, B))
        r["self_stopped"] = ((r.hit_cap == 0) & (r.steps_used <= B)).astype(float)
        out.append(r)
    return pd.concat([rest] + out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--a3", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config()
    lam = cfg["economy"]["lambda"]

    base = pd.read_csv(args.baselines)
    print(f"baselines: {len(base)} rows {sorted(base.arm.unique())}")
    a3 = pd.concat([rows_from_jsonl(p, lam) for p in args.a3], ignore_index=True)
    print(f"a3: {len(a3)} rows, modes {sorted(a3['mode'].unique())}, "
          f"budgets {sorted(a3.budget_B.unique())}")

    df = pd.concat([base, a3], ignore_index=True)
    df = rescore_a0(df, cfg["episode"]["budgets"], lam)

    check_utility_recompute(df, lam)
    check_row_counts(df, cfg["data"]["dev_size"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} rows -> {args.out}")
    print(df.groupby(["arm", "mode", "budget_B"]).size().to_string())


if __name__ == "__main__":
    main()
