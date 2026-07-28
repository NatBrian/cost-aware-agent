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
from eval.metrics import check_row_counts, check_utility_recompute

# The gate is about A3 with the harness OFF (plan §6). F6 also collects A3
# harness-ON and an oracle forced-continuation replay, and ALL of them carry
# arm="a3" — selecting on arm alone silently averages the three populations
# into the headline claim. Pin the mode per arm.
GATE_MODE = {"a1": "none", "a2": "enforce", "a3": "none"}


def evaluate_gate(df: pd.DataFrame, cfg: dict) -> dict:
    g = cfg["gate"]
    B = cfg["episode"]["budgets"][g["budget"]]
    # Never render a verdict from an unguarded CSV (stale lambda, hand edits,
    # duplicated rows): re-run the same guards F6 runs at build time.
    check_utility_recompute(df, cfg["economy"]["lambda"])
    check_row_counts(df, cfg["data"]["dev_size"])

    at = df[df.budget_B == B]
    sel = lambda arm: at[(at.arm == arm) & (at["mode"] == GATE_MODE[arm])]
    mean = lambda arm, m: float(sel(arm)[m].mean())
    for arm in ("a1", "a2", "a3"):
        rows = sel(arm)
        if rows.empty:
            raise ValueError(f"no rows for arm {arm} (mode {GATE_MODE[arm]}) "
                             f"at gate budget B={B}")
        n_tasks = rows.task_id.nunique()
        if n_tasks != cfg["data"]["dev_size"]:
            raise ValueError(f"arm {arm} at B={B}: {n_tasks} unique tasks, "
                             f"want {cfg['data']['dev_size']} — the gate must "
                             "be computed on the whole frozen dev set")

    u3, u1, u2 = (mean(a, "utility") for a in ("a3", "a1", "a2"))
    cond1 = u3 > u1 and u3 > u2
    self_stop = mean("a3", "self_stopped")
    cond2 = self_stop >= g["min_self_stop"]
    f13, f12 = mean("a3", "f1"), mean("a2", "f1")
    cond3 = f13 >= f12 - g["f1_margin"]

    return {
        "budget_B": B,
        "modes": GATE_MODE,
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
