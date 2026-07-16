"""F5 — label study E4 (paper_plan_v2 §9, §5.4): Snell vs prophet-argmax vs
TD/GAE vs MC labels, at matched label-compute — stopper regret and downstream
executor cost, side by side.

Expected CSV schema (experiments/results/f5_label_study.csv — one row per
label type, aggregated over held-out trajectories / the frozen eval subsample):
    label_type        : str (snell | prophet | td_gae | mc)
    stopper_regret    : float — held-out stopping regret (utility gap, §5.3)
    regret_ci_lo, regret_ci_hi : 95% bootstrap CI (§5.6)
    executor_cost_usd : float — downstream executor mean cost per task at
                        matched accuracy (E4 downstream arm)
    cost_ci_lo, cost_ci_hi     : 95% bootstrap CI

Usage: python analysis/figures/f5_label_study.py --results experiments/results/f5_label_study.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # research/ → cassi.*

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cassi.analysis.plotstyle import PALETTE, apply_style, finish, load_results, yerr_from_ci  # noqa: E402

SCRIPT = "f5_label_study"
ORDER = ["snell", "prophet", "td_gae", "mc"]          # fixed order, fixed hues
LABEL_COLOR = {"snell": PALETTE[0], "prophet": PALETTE[7],
               "td_gae": PALETTE[4], "mc": PALETTE[3]}
PRETTY = {"snell": "Snell (ours)", "prophet": "prophet argmax",
          "td_gae": "TD/GAE", "mc": "MC"}


def _bars(ax, df, value, lo, hi, ylabel, title):
    rows = [df[df["label_type"] == t].iloc[0] for t in ORDER
            if t in set(df["label_type"])]
    x = np.arange(len(rows))
    names = [str(r["label_type"]) for r in rows]
    g = df.set_index("label_type").loc[names].reset_index()
    yerr = yerr_from_ci(g, value, lo, hi) if {lo, hi}.issubset(g.columns) else None
    ax.bar(x, g[value], 0.6, yerr=yerr,
           color=[LABEL_COLOR.get(n, "#52514e") for n in names],
           edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, [PRETTY.get(n, n) for n in names], rotation=12)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def build(df, out: str) -> None:
    apply_style()
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(6.0, 2.6),
                                     gridspec_kw={"wspace": 0.42})
    _bars(ax_r, df, "stopper_regret", "regret_ci_lo", "regret_ci_hi",
          "held-out stopping regret (utility)", "stopper quality")
    _bars(ax_c, df, "executor_cost_usd", "cost_ci_lo", "cost_ci_hi",
          "executor cost per task (USD)", "downstream cost, matched acc.")
    fig.text(0.995, -0.075,
             "matched label-compute incl. draft tokens + forced continuation (E4); "
             "error bars: 95% bootstrap CI",
             ha="right", fontsize=6.0, color="#52514e")
    finish(fig, out, SCRIPT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/results/f5_label_study.csv")
    ap.add_argument("--out", default="paper/figures/f5_label_study.pdf")
    args = ap.parse_args()
    df = load_results(args.results, SCRIPT,
                      required_cols=("label_type", "stopper_regret", "executor_cost_usd"))
    if df is None:
        return
    build(df, args.out)


if __name__ == "__main__":
    main()
