"""The pre-registered GO/NO-GO gate as code (plan §6; thresholds from config).

Usage: .venv/bin/python -m eval.gate_check --rows experiments/results/foundation_eval.csv
Input: the per-task rows CSV (all arms; A3 rows = harness-off runs).
Prints each condition itemized, then the verdict. Exit code 0=GO, 1=NO-GO.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import load_config


def evaluate_gate(df: pd.DataFrame, cfg: dict) -> dict:
    g = cfg["gate"]
    B = cfg["episode"]["budgets"][g["budget"]]
    at = df[df.budget_B == B]
    mean = lambda arm, m: float(at[at.arm == arm][m].mean())
    for arm in ("a1", "a2", "a3"):
        if at[at.arm == arm].empty:
            raise ValueError(f"no rows for arm {arm} at gate budget B={B}")

    u3, u1, u2 = (mean(a, "utility") for a in ("a3", "a1", "a2"))
    cond1 = u3 > u1 and u3 > u2
    self_stop = mean("a3", "self_stopped")
    cond2 = self_stop >= g["min_self_stop"]
    f13, f12 = mean("a3", "f1"), mean("a2", "f1")
    cond3 = f13 >= f12 - g["f1_margin"]

    return {
        "budget_B": B,
        "cond1_utility": {"passed": cond1, "a3": round(u3, 4),
                          "a1": round(u1, 4), "a2": round(u2, 4)},
        "cond2_self_stop": {"passed": cond2, "a3": round(self_stop, 4),
                            "threshold": g["min_self_stop"]},
        "cond3_no_collapse": {"passed": cond3, "a3_f1": round(f13, 4),
                              "a2_f1": round(f12, 4), "margin": g["f1_margin"]},
        "verdict": "GO" if (cond1 and cond2 and cond3) else "NO-GO",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    args = ap.parse_args()
    cfg = load_config()
    result = evaluate_gate(pd.read_csv(args.rows), cfg)
    for k, v in result.items():
        print(f"{k}: {v}")
    sys.exit(0 if result["verdict"] == "GO" else 1)


if __name__ == "__main__":
    main()
