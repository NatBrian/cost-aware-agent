"""T1 — headline cost@iso-accuracy / accuracy@iso-cost (paper_plan_v2 §9;
3 seeds, 95% bootstrap CIs, §5.3 frontier protocol).

Expected CSV schema (experiments/results/t1_headline.csv — one row per
method × domain, computed at the headline operating point by
cassi.eval.metrics.{cost_at_iso_accuracy, accuracy_at_iso_cost, pareto_auc}):
    domain               : str
    method               : str (cassi, b1_react … b9_direct_shaping, oracle)
    cost_at_iso_acc_usd  : float — blank/NaN for knobless methods (excluded
                           from iso-claims, §5.3)
    cost_ci_lo, cost_ci_hi : 95% bootstrap CI (§5.6)
    acc_at_iso_cost      : float in [0,1] — blank/NaN for knobless methods
    acc_ci_lo, acc_ci_hi : 95% bootstrap CI
    pareto_auc           : float — shared cost_range stated in the caption
    n_seeds              : int (3 at headline operating points, §5.6)

Usage: python analysis/tables/t1_headline.py --results experiments/results/t1_headline.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # research/ → cassi.*

from cassi.analysis.plotstyle import load_results  # noqa: E402
from cassi.analysis.texutils import fmt_ci, fmt_num, tex_escape, write_tabular  # noqa: E402

SCRIPT = "t1_headline"


def build(df, out: str, source: str) -> None:
    rows, midrules = [], []
    for domain in sorted(df["domain"].unique()):
        sub = df[df["domain"] == domain].sort_values("method")
        # bold the best (lowest) iso-accuracy cost among knobbed methods
        costs = sub["cost_at_iso_acc_usd"]
        best = costs.idxmin() if costs.notna().any() else None
        midrules.append(len(rows))
        for idx, r in sub.iterrows():
            cost = fmt_ci(r["cost_at_iso_acc_usd"], r.get("cost_ci_lo"),
                          r.get("cost_ci_hi"), digits=3, dollar=True)
            if idx == best and cost != "--":
                cost = rf"\textbf{{{cost}}}"
            rows.append([
                tex_escape(domain), tex_escape(r["method"]), cost,
                fmt_ci(r["acc_at_iso_cost"], r.get("acc_ci_lo"),
                       r.get("acc_ci_hi"), digits=3),
                fmt_num(r["pareto_auc"], digits=3),
                fmt_num(r.get("n_seeds"), digits=0),
            ])
    write_tabular(
        out, "llrrrr",
        ["Domain", "Method", r"Cost@iso-acc (USD)", "Acc@iso-cost",
         "Pareto AUC", "Seeds"],
        rows, script=SCRIPT, source_csv=source,
        midrules_before=tuple(m for m in midrules if m > 0),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/results/t1_headline.csv")
    ap.add_argument("--out", default="paper/tables/t1_headline.tex")
    args = ap.parse_args()
    df = load_results(args.results, SCRIPT,
                      required_cols=("domain", "method", "cost_at_iso_acc_usd",
                                     "acc_at_iso_cost", "pareto_auc"))
    if df is None:
        return
    build(df, args.out, args.results)


if __name__ == "__main__":
    main()
