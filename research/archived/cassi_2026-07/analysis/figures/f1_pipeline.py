"""F1 — CASSI pipeline schematic (paper_plan_v2 §9): labels → stopper → shaped
GRPO → loop, plus the inference-monitor branch. Pure matplotlib boxes/arrows —
no data, no CSV (the only figure with nothing to measure).

Usage: python analysis/figures/f1_pipeline.py [--out paper/figures/f1_pipeline.pdf]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # research/ → cassi.*

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from cassi.analysis.plotstyle import PALETTE, TEXT_PRIMARY, apply_style, finish  # noqa: E402

SCRIPT = "f1_pipeline"

# (x, y, w, h, title, subtitle, palette slot)
BOXES = {
    "collect": (0.015, 0.56, 0.215, 0.36,
                "Forced-continuation\nrollouts",
                "$\\pi_i$, G=8, run to $T_{max}$;\nANSWER logged as draft\n(§2.1)", 0),
    "labels":  (0.275, 0.56, 0.215, 0.36,
                "Snell-envelope\nlabels",
                "$V_t=\\max(U_t, \\hat{E}[V_{t+1}|x_t])$\nper $\\lambda$; Alg. 1 (§2.2)", 4),
    "stopper": (0.535, 0.56, 0.20, 0.36,
                "Stopper $M_\\theta$ (2B)",
                "3 heads: $a^*$ / $\\hat{\\Delta}$ / $\\hat{V}$\n$\\lambda$-conditioned (§2.3)", 6),
    "grpo":    (0.78, 0.56, 0.205, 0.36,
                "Shaped GRPO",
                "$r_t=\\Phi(x_{t+1})-\\Phi(x_t)$\n$\\Phi=\\hat{V}_\\theta$; step-level\nadvantages (§2.4)", 1),
    "monitor": (0.535, 0.03, 0.20, 0.30,
                "Inference monitor",
                "STOP when $\\hat{\\Delta}_t\\leq 0$;\n$\\lambda$ dial, no retraining\n(§2.5)", 3),
    "policy":  (0.78, 0.03, 0.205, 0.30,
                "Executor $\\pi_{i+1}$ (9B)",
                "internalized stopping:\nself-terminates\npre-monitor", 1),
}


def _box(ax, key):
    x, y, w, h, title, sub, ci = BOXES[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                linewidth=1.4, edgecolor=PALETTE[ci],
                                facecolor="white"))
    n_title = title.count("\n")
    ax.text(x + w / 2, y + h - 0.05 - 0.045 * n_title, title, ha="center", va="center",
            fontsize=7.6, fontweight="bold", color=TEXT_PRIMARY)
    ax.text(x + w / 2, y + 0.33 * h, sub, ha="center", va="center",
            fontsize=6.2, color="#52514e")


def _arrow(ax, xy_from, xy_to, label="", rad=0.0, color="#52514e",
           label_dy=0.035, label_va="bottom"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.2, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2
        ax.text(mx, my + label_dy, label, ha="center", va=label_va,
                fontsize=6.2, color="#52514e", style="italic")


def build(out: str) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.12)
    ax.axis("off")

    for k in BOXES:
        _box(ax, k)

    # top-row data flow, labels above the arrows (clear of box titles)
    y_mid = 0.74
    _arrow(ax, (0.230, y_mid), (0.275, y_mid), "$(x_t,q_t,c_t)$")
    _arrow(ax, (0.490, y_mid), (0.535, y_mid), "$(a^*,\\Delta^*,V^*)$")
    _arrow(ax, (0.735, y_mid), (0.780, y_mid), "$\\Phi=\\hat{V}_\\theta$")
    # the measured loop (§2.7): arcs OVER the top row, no text crossings
    _arrow(ax, (0.883, 0.925), (0.122, 0.925), rad=-0.22)
    ax.text(0.50, 1.075, "loop $\\geq 2$ iterations: re-collect with $\\pi_{i+1}$, "
                         "refresh $M_\\theta$ (§2.7, E5)",
            ha="center", fontsize=6.6, color="#52514e", style="italic")
    # inference branch
    _arrow(ax, (0.635, 0.56), (0.635, 0.335), "deploy", label_dy=-0.11)
    _arrow(ax, (0.883, 0.56), (0.883, 0.335), "deploy", label_dy=-0.11)
    _arrow(ax, (0.735, 0.18), (0.780, 0.18), "$\\hat{\\Delta}_t$")

    finish(fig, out, SCRIPT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper/figures/f1_pipeline.pdf")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
