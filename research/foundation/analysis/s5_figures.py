"""FOUNDATION-2 Step-1 figures: the two plots the pre-registered result needs.

  fig_s5a  dose-response — paired Δsteps (treatment − control) per budget, with
           95% CIs, the pre-registered threshold, and the binding fraction. The
           prediction (S3 §6) is that the effect concentrates where the budget
           binds; this is the figure that shows it or refutes it.
  fig_s5b  H2 — stop-step distribution split by the CONTROL arm's outcome. If
           the saving is abandonment, the shift lives in the "would have failed"
           panel and not in the other one. A uniform shift means the policy just
           got hastier, and the figure makes that impossible to miss.

Same discipline as analysis/figures.py: CSV/JSONL in, PDF out, no hand-edited
numbers, CVD-safe palette, direct labels, 95% bootstrap CIs.

Usage:
  .venv/bin/python -m analysis.s5_figures --dir experiments/results/s5_eval \
      --out-dir experiments/reports/figs
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_config
from eval.metrics import bootstrap_ci

THRESHOLD = -0.119          # pre-registered (S3 §4); drawn, never recomputed
C_TRT, C_CTL, C_THR = "#009E73", "#0072B2", "#D55E00"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)


def _load(d: Path, arm: str, bname: str) -> dict:
    p = d / f"{arm}_{bname}.jsonl"
    if not p.exists():
        return {}
    return {e["task_id"]: e for e in (json.loads(l) for l in open(p) if l.strip())}


def _paired(d: Path, bname: str, key):
    c, t = _load(d, "control", bname), _load(d, "treatment", bname)
    common = sorted(set(c) & set(t))
    if not common:
        return None, None, None
    return (common,
            np.array([key(t[k]) for k in common], float),
            np.array([key(c[k]) for k in common], float))


def fig_s5a(d: Path, out: Path, seed: int, budgets: dict, gate: int) -> Path | None:
    steps = lambda e: e["steps_used"]
    xs, means, los, his, labels = [], [], [], [], []
    for i, (bname, B) in enumerate(budgets.items()):
        common, s_t, s_c = _paired(d, bname, steps)
        if common is None:
            continue
        dd = s_t - s_c
        lo, hi = bootstrap_ci(dd, 10000, seed)
        xs.append(i); means.append(dd.mean()); los.append(lo); his.append(hi)
        labels.append(f"B={B}" + ("\n(gate, binds)" if B == gate else "\n(slack)"))
    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axhline(0, color="#666666", linewidth=0.9)
    ax.axhline(THRESHOLD, color=C_THR, linewidth=1.2, linestyle="--")
    # label INSIDE the axes: an annotation past the last tick gets clipped or
    # silently widens the figure, and neither is reproducible across backends
    ax.text(-0.42, THRESHOLD, f"pre-registered threshold {THRESHOLD}",
            color=C_THR, fontsize=8, va="bottom", ha="left")
    for x, m, lo, hi in zip(xs, means, los, his):
        ax.plot([x, x], [lo, hi], color=C_TRT, linewidth=1.6, solid_capstyle="round")
        ax.plot([x], [m], marker="D", color=C_TRT, markersize=7, zorder=3)
        ax.annotate(f"{m:+.3f}", (x, m), textcoords="offset points",
                    xytext=(11, -3), fontsize=9, color=C_TRT)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Δ steps  (treatment − control), paired")
    # NEUTRAL title. An earlier draft read "Cost pressure moves stopping only
    # where the budget binds" — which asserts the pre-registered prediction as a
    # finding before the data exists, and would have been a false caption on any
    # other outcome. The figure states what it plots; the reader draws the
    # conclusion.
    ax.set_title("Paired Δ steps by budget, with 95% CIs",
                 fontsize=11, loc="left")
    _style(ax)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def fig_s5b(d: Path, out: Path, gate_name: str) -> Path | None:
    c, t = _load(d, "control", gate_name), _load(d, "treatment", gate_name)
    common = sorted(set(c) & set(t))
    if not common:
        return None
    # partition by the CONTROL's outcome — fixed, never chosen by the treatment
    doomed = [k for k in common if c[k]["final_f1"] <= 0]
    ok = [k for k in common if c[k]["final_f1"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, keys, title in ((axes[0], doomed, "control FAILED (doomed work)"),
                            (axes[1], ok, "control SUCCEEDED")):
        if not keys:
            ax.set_visible(False); continue
        sc = np.array([c[k]["steps_used"] for k in keys], float)
        st = np.array([t[k]["steps_used"] for k in keys], float)
        bins = np.arange(0.5, max(sc.max(), st.max()) + 1.5, 1.0)
        ax.hist(sc, bins=bins, density=True, histtype="step", linewidth=1.8,
                color=C_CTL, label=f"control  (mean {sc.mean():.2f})")
        ax.hist(st, bins=bins, density=True, histtype="step", linewidth=1.8,
                color=C_TRT, label=f"treatment  (mean {st.mean():.2f})")
        ax.set_title(f"{title}   n={len(keys)}   Δ={st.mean()-sc.mean():+.3f}",
                     fontsize=10, loc="left")
        ax.set_xlabel("steps used")
        ax.legend(frameon=False, fontsize=8.5)
        _style(ax)
    axes[0].set_ylabel("fraction of episodes")
    fig.suptitle("H2: does the saving concentrate on work that was going to fail?",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    d, outd = Path(args.dir), Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    budgets = cfg["episode"]["budgets"]
    gate_name = cfg["episode"].get("gate_budget", "small")
    for p in (fig_s5a(d, outd / "fig_s5a_dose_response.pdf", cfg["seed"],
                      budgets, budgets[gate_name]),
              fig_s5b(d, outd / "fig_s5b_h2_split.pdf", gate_name)):
        print(f"wrote {p}" if p else "skipped a figure (missing arm data)")


if __name__ == "__main__":
    main()
