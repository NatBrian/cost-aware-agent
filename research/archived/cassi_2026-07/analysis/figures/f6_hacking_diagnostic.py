"""F6 — reward-hacking diagnostic (paper_plan_v2 §9, §2.4): stopper-predicted
V̂ vs realized task reward across executor training; divergence rising ⇒ the
executor is gaming the frozen coach ⇒ stopper refresh (the per-iteration
refresh boundaries are drawn).

Expected CSV schema (experiments/results/f6_hacking.csv — one row per logged
training step, from the P6 divergence dashboard):
    training_step        : int — global GRPO step
    iteration            : int — loop iteration the step belongs to (0,1,2…)
    v_hat_mean           : float — batch-mean V̂_θ(x_τ) (stopper's prediction
                           at the terminal-ward states)
    realized_reward_mean : float — batch-mean realized R_base (same economy)
    divergence           : float — mean |V̂ − realized| over the batch
                           (executor/shaping.py::hacking_divergence)

Usage: python analysis/figures/f6_hacking_diagnostic.py --results experiments/results/f6_hacking.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # research/ → cassi.*

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cassi.analysis.plotstyle import PALETTE, apply_style, finish, load_results  # noqa: E402

SCRIPT = "f6_hacking_diagnostic"


def build(df, out: str) -> None:
    apply_style()
    df = df.sort_values("training_step")
    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(3.8, 3.2), sharex=True, height_ratios=[1.6, 1.0],
        gridspec_kw={"hspace": 0.12},
    )

    ax.plot(df["training_step"], df["v_hat_mean"], color=PALETTE[0],
            label="$\\hat{V}_\\theta$ (coach prediction)")
    ax.plot(df["training_step"], df["realized_reward_mean"], color=PALETTE[5],
            linestyle="--", label="realized $R_{base}$")
    ax.set_ylabel("reward (utility units)")
    ax.legend(loc="best")
    ax.set_title("coach prediction vs realized reward")

    axd.plot(df["training_step"], df["divergence"], color=PALETTE[7])
    axd.set_ylabel("$|\\hat{V}-R|$")
    axd.set_xlabel("GRPO training step")

    # stopper-refresh boundaries: where the loop iteration increments (§2.7)
    if "iteration" in df.columns:
        it = df["iteration"].to_numpy()
        for k in range(1, len(it)):
            if it[k] != it[k - 1]:
                for a in (ax, axd):
                    a.axvline(df["training_step"].iloc[k], color="#52514e",
                              linewidth=0.9, linestyle=":")
                ax.annotate("stopper refresh", (df["training_step"].iloc[k],
                                                ax.get_ylim()[1]),
                            xytext=(3, -9), textcoords="offset points",
                            fontsize=6.6, color="#52514e")
    finish(fig, out, SCRIPT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/results/f6_hacking.csv")
    ap.add_argument("--out", default="paper/figures/f6_hacking_diagnostic.pdf")
    args = ap.parse_args()
    df = load_results(args.results, SCRIPT,
                      required_cols=("training_step", "v_hat_mean",
                                     "realized_reward_mean", "divergence"))
    if df is None:
        return
    build(df, args.out)


if __name__ == "__main__":
    main()
