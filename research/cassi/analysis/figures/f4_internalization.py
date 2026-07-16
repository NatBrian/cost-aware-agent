"""F4 — internalization across loop iterations (paper_plan_v2 §9, §2.5/§5.3):
monitor-on vs monitor-off cost & accuracy bars per iteration i = 0,1,2, plus
the % of episodes self-terminated before the monitor fires.

Expected CSV schema (experiments/results/f4_internalization.csv — one row per
iteration × monitor arm, aggregated over the frozen eval subsample):
    iteration      : int (0, 1, 2)
    monitor        : str ("on" | "off")
    cost_usd       : float — mean all-inclusive dollars per task
    cost_ci_lo, cost_ci_hi : 95% bootstrap CI (§5.6)
    accuracy       : float in [0,1]
    acc_ci_lo, acc_ci_hi   : 95% bootstrap CI
    self_stop_rate : float in [0,1] — % episodes self-terminated pre-monitor
                     (meaningful on monitor="on" rows; ignored on "off" rows)

Usage: python analysis/figures/f4_internalization.py --results experiments/results/f4_internalization.csv
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

SCRIPT = "f4_internalization"
ARM_COLOR = {"on": PALETTE[0], "off": PALETTE[5]}


def _grouped_bars(ax, df, value, lo, hi, ylabel):
    iters = sorted(df["iteration"].unique())
    width = 0.36
    x = np.arange(len(iters))
    for k, arm in enumerate(("on", "off")):
        g = (df[df["monitor"] == arm].set_index("iteration").reindex(iters).reset_index())
        yerr = yerr_from_ci(g, value, lo, hi) if {lo, hi}.issubset(g.columns) else None
        ax.bar(x + (k - 0.5) * width, g[value], width * 0.92, yerr=yerr,
               color=ARM_COLOR[arm], edgecolor="white", linewidth=0.8,
               label=f"monitor {arm}")
    ax.set_xticks(x, [f"i={i}" for i in iters])
    ax.set_ylabel(ylabel)


def build(df, out: str) -> None:
    apply_style()
    fig, (ax_c, ax_a, ax_s) = plt.subplots(
        1, 3, figsize=(7.4, 2.5), gridspec_kw={"wspace": 0.42})

    _grouped_bars(ax_c, df, "cost_usd", "cost_ci_lo", "cost_ci_hi",
                  "cost per task (USD)")
    ax_c.set_title("cost: enforced vs internalized")
    ax_c.legend(loc="upper right")

    _grouped_bars(ax_a, df, "accuracy", "acc_ci_lo", "acc_ci_hi", "accuracy")
    ax_a.set_title("accuracy, monitor off")

    on = df[df["monitor"] == "on"].sort_values("iteration")
    ax_s.plot(on["iteration"], 100.0 * on["self_stop_rate"], color=PALETTE[1],
              marker="o")
    for _, r in on.iterrows():
        ax_s.annotate(f"{100 * r['self_stop_rate']:.0f}%",
                      (r["iteration"], 100 * r["self_stop_rate"]),
                      textcoords="offset points", xytext=(0, 5),
                      ha="center", fontsize=7, color=PALETTE[1])
    ax_s.set_xticks(on["iteration"], [f"i={i}" for i in on["iteration"]])
    ax_s.set_ylim(0, 105)
    ax_s.set_ylabel("% self-terminated pre-monitor")
    ax_s.set_title("self-termination across the loop")

    fig.text(0.995, -0.075, "error bars: 95% bootstrap CI over test instances (§5.6)",
             ha="right", fontsize=6.2, color="#52514e")
    finish(fig, out, SCRIPT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/results/f4_internalization.csv")
    ap.add_argument("--out", default="paper/figures/f4_internalization.pdf")
    args = ap.parse_args()
    df = load_results(args.results, SCRIPT,
                      required_cols=("iteration", "monitor", "cost_usd",
                                     "accuracy", "self_stop_rate"))
    if df is None:
        return
    build(df, args.out)


if __name__ == "__main__":
    main()
