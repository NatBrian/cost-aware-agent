"""F7 figures: CSV in -> PDF out, no hand-edited numbers.

Usage:
  .venv/bin/python -m analysis.figures --rows rows.csv --divergence div.jsonl --out-dir figs/

Style (dataviz skill, validated 2026-07-22): entity-fixed CVD-safe palette
(validator ALL PASS, worst adjacent dE 9.6), color follows the arm everywhere,
one axis per plot, thin marks, direct labels + distinct markers (the required
secondary encoding for the two contrast-WARN hues), 95% bootstrap CI bars.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eval.metrics import aggregate

ARM_COLOR = {"a0": "#CC79A7", "a1": "#0072B2", "a2": "#E69F00", "a3": "#009E73"}
ARM_MARKER = {"a0": "s", "a1": "o", "a2": "^", "a3": "D"}
ARM_LABEL = {"a0": "A0 plain ReAct", "a1": "A1 prompted budget",
             "a2": "A2 harness-enforced", "a3": "A3 RL-internalized"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)


def fig1_frontier(rows: pd.DataFrame, out: Path, resamples: int = 2000) -> Path:
    """Steps (x) vs F1 (y), one line per arm across budgets — the headline."""
    agg = aggregate(rows, n_resamples=resamples)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for arm, g in agg.groupby("arm"):
        g = g.sort_values("steps_used")
        ax.errorbar(g.steps_used, g.f1,
                    yerr=[g.f1 - g.f1_lo, g.f1_hi - g.f1],
                    color=ARM_COLOR[arm], marker=ARM_MARKER[arm],
                    linewidth=2, markersize=6, capsize=2, label=ARM_LABEL[arm])
        last = g.iloc[-1]
        ax.annotate(ARM_LABEL[arm].split()[0], (last.steps_used, last.f1),
                    xytext=(4, 4), textcoords="offset points", fontsize=8,
                    color="#444444")
    ax.set_xlabel("mean steps used")
    ax.set_ylabel("F1")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    p = out / "fig1_frontier.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig2_internalization(rows: pd.DataFrame, out: Path,
                         resamples: int = 2000) -> Path:
    """(a) self-stop rate per arm; (b) stop-step histograms per budget for a3."""
    agg = aggregate(rows, n_resamples=resamples)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    ax = axes[0]
    arms = [a for a in ("a0", "a1", "a2", "a3") if a in set(agg.arm)]
    sub = (agg.groupby("arm", as_index=False)
           .agg(self_stopped=("self_stopped", "mean")))
    sub = sub.set_index("arm").loc[arms].reset_index()
    ax.bar(sub.arm, sub.self_stopped,
           color=[ARM_COLOR[a] for a in sub.arm], width=0.55)
    for _, r in sub.iterrows():
        ax.annotate(f"{r.self_stopped:.0%}", (r.arm, r.self_stopped),
                    ha="center", xytext=(0, 3), textcoords="offset points",
                    fontsize=8)
    ax.set_ylabel("self-terminated episodes")
    ax.set_ylim(0, 1.05)
    _style(ax)

    ax = axes[1]
    a3 = rows[rows.arm == "a3"]
    src = a3 if not a3.empty else rows[rows.arm == rows.arm.iloc[0]]
    for b, g in src.groupby("budget_B"):
        ax.hist(g.steps_used, bins=range(0, 12), histtype="step",
                linewidth=2, label=f"B={b}")
    ax.set_xlabel("stop step (A3, harness off)")
    ax.set_ylabel("episodes")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    fig.tight_layout()
    p = out / "fig2_internalization.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig3_divergence(div_jsonl: Path, out: Path) -> Path:
    """Judge score vs realized F1 over training — the hacking diagnostic."""
    rows = [json.loads(l) for l in open(div_jsonl) if l.strip()]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(df.step, df.judge_score_mean, color="#0072B2", linewidth=2,
            label="judge step score")
    ax.plot(df.step, df.f1_mean, color="#009E73", linewidth=2,
            label="realized F1")
    ax.set_xlabel("training step")
    ax.set_ylabel("batch mean")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    p = out / "fig3_divergence.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--divergence", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.rows)
    print(fig1_frontier(rows, out))
    print(fig2_internalization(rows, out))
    if args.divergence and Path(args.divergence).exists():
        print(fig3_divergence(Path(args.divergence), out))


if __name__ == "__main__":
    main()
