"""T2 — all baselines × the two training domains (paper_plan_v2 §9): raw
accuracy and all-inclusive dollar cost at each method's headline operating
point (3 seeds, §5.6).

Expected CSV schema (experiments/results/t2_baselines.csv — one row per
method × domain at the headline operating point):
    method    : str (cassi, b1_react … b9_direct_shaping, oracle)
    domain    : str (qa | alfworld)
    accuracy  : float in [0,1]
    acc_ci_lo, acc_ci_hi   : 95% bootstrap CI (§5.6)
    cost_usd  : float — mean all-inclusive dollars per task (§5.3 accounting)
    cost_ci_lo, cost_ci_hi : 95% bootstrap CI
    n_seeds   : int

Emits one row per method with (accuracy, cost) column pairs per domain.

Usage: python analysis/tables/t2_baselines.py --results experiments/results/t2_baselines.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # research/ → cassi.*

from cassi.analysis.plotstyle import load_results  # noqa: E402
from cassi.analysis.texutils import fmt_ci, tex_escape, write_tabular  # noqa: E402

SCRIPT = "t2_baselines"


def build(df, out: str, source: str) -> None:
    domains = sorted(df["domain"].unique())
    header = ["Method"]
    for d in domains:
        header += [rf"{tex_escape(d)} acc.", rf"{tex_escape(d)} cost (USD)"]
    rows = []
    for method in sorted(df["method"].unique()):
        row = [tex_escape(method)]
        for d in domains:
            g = df[(df["method"] == method) & (df["domain"] == d)]
            if g.empty:
                row += ["--", "--"]
            else:
                r = g.iloc[0]
                row += [
                    fmt_ci(r["accuracy"], r.get("acc_ci_lo"), r.get("acc_ci_hi"), digits=3),
                    fmt_ci(r["cost_usd"], r.get("cost_ci_lo"), r.get("cost_ci_hi"),
                           digits=3, dollar=True),
                ]
        rows.append(row)
    write_tabular(out, "l" + "rr" * len(domains), header, rows,
                  script=SCRIPT, source_csv=source)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/results/t2_baselines.csv")
    ap.add_argument("--out", default="paper/tables/t2_baselines.tex")
    args = ap.parse_args()
    df = load_results(args.results, SCRIPT,
                      required_cols=("method", "domain", "accuracy", "cost_usd"))
    if df is None:
        return
    build(df, args.out, args.results)


if __name__ == "__main__":
    main()
