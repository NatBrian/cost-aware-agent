"""Episodes JSONL -> combined per-task rows CSV (feeds aggregate/gate/figures).

Usage: .venv/bin/python -m eval.build_rows --out rows.csv ep1.jsonl ep2.jsonl ...
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import load_config
from eval.metrics import (check_row_counts, check_utility_recompute,
                          rows_from_jsonl)


def rescore_a0(df: pd.DataFrame, budgets: dict, lam: float) -> pd.DataFrame:
    """A0 sees no budget, so its behaviour cannot depend on B — one run is
    re-scored under every budget's utility (plan §3, F4 'three numbers for
    free'). Without this A0 is absent from the gate budget entirely, and the
    frontier figure has a single A0 point. Re-running A0 per budget instead
    would add sampling noise to an arm whose behaviour is by construction
    identical. (audit 2026-07-28)
    """
    a0, rest = df[df.arm == "a0"], df[df.arm != "a0"]
    if a0.empty:
        return df
    out = []
    for B in sorted(set(budgets.values())):
        r = a0.copy()
        r["budget_B"] = B
        r["utility"] = r.f1 - lam * (r.steps_used / max(1, B))
        r["self_stopped"] = ((r["mode"] == "none") & r.answered_at.notna()
                             & (r.answered_at <= B)).astype(float)
        out.append(r)
    return pd.concat([rest] + out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config()
    lam = cfg["economy"]["lambda"]
    df = pd.concat([rows_from_jsonl(p, lam) for p in args.inputs],
                   ignore_index=True)
    check_utility_recompute(df, lam)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} rows ({sorted(df.arm.unique())}) -> {args.out}")


if __name__ == "__main__":
    main()
