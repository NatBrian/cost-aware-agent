"""F3 — cost/accuracy Pareto frontiers per domain (paper_plan_v2 §9, §5.3
frontier protocol): every method swept over ITS OWN cost knob (3–5 points);
knobless methods (B1 ReAct, oracle) drawn as single points, visually flagged as
excluded from iso-claims.

Expected CSV schema (experiments/results/f3_pareto.csv — one row per
method × domain × knob setting, aggregated over the frozen eval subsample):
    domain      : str  (qa | alfworld | ...)
    method      : str  (cassi, b1_react, ..., b9_direct_shaping, oracle)
    knob        : float — the method's own knob value (λ, threshold, coeff);
                  knobless methods have a single row (knob may be blank)
    cost_usd    : float — mean all-inclusive dollars per task (§5.3 accounting)
    accuracy    : float in [0,1]
    acc_ci_lo, acc_ci_hi : 95% bootstrap CI on accuracy (§5.6) — optional but
                  expected from P9 (cassi.eval.stats.bootstrap_ci)

Usage: python analysis/figures/f3_pareto.py --results experiments/results/f3_pareto.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # research/ → cassi.*

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cassi.analysis.plotstyle import (  # noqa: E402
    apply_style, finish, load_results, series_style, yerr_from_ci,
)

SCRIPT = "f3_pareto"


def build(df, out: str) -> None:
    apply_style()
    domains = sorted(df["domain"].unique())
    fig, axes = plt.subplots(1, len(domains), figsize=(3.4 * len(domains), 2.9),
                             squeeze=False)
    has_ci = {"acc_ci_lo", "acc_ci_hi"}.issubset(df.columns)

    fallback = len(df["method"].unique())  # unknown methods get trailing slots
    for ax, domain in zip(axes[0], domains):
        sub = df[df["domain"] == domain]
        for j, method in enumerate(sorted(sub["method"].unique())):
            g = sub[sub["method"] == method].sort_values("cost_usd")
            st = series_style(method, fallback_idx=fallback + j)
            yerr = yerr_from_ci(g, "accuracy", "acc_ci_lo", "acc_ci_hi") if has_ci else None
            if len(g) == 1:  # knobless: single point, excluded from iso-claims (§5.3)
                ax.errorbar(g["cost_usd"], g["accuracy"], yerr=yerr,
                            color=st["color"], marker=st["marker"], linestyle="none",
                            markerfacecolor="white", label=f"{method} (no knob)")
            else:
                ax.errorbar(g["cost_usd"], g["accuracy"], yerr=yerr,
                            color=st["color"], marker=st["marker"],
                            linestyle=st["linestyle"], label=method)
        ax.set_title(domain)
        ax.set_xlabel("cost per task (USD, all-inclusive)")
    axes[0][0].set_ylabel("accuracy")
    # one legend for the figure — identity via color+marker (CVD-safe order)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        h2, l2 = ax.get_legend_handles_labels()
        for h, l in zip(h2, l2):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14),
               ncols=min(4, len(labels)))
    fig.text(0.995, -0.075, "error bars: 95% bootstrap CI over test instances (§5.6)",
             ha="right", fontsize=6.2, color="#52514e")
    finish(fig, out, SCRIPT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/results/f3_pareto.csv")
    ap.add_argument("--out", default="paper/figures/f3_pareto.pdf")
    args = ap.parse_args()
    df = load_results(args.results, SCRIPT,
                      required_cols=("domain", "method", "cost_usd", "accuracy"))
    if df is None:
        return
    build(df, args.out)


if __name__ == "__main__":
    main()
