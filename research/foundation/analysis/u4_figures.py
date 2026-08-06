"""U4 — the paper's figure set. CSV/JSONL in, PDF out, no hand-edited numbers.

  fig1 selectivity — Δsteps on doomed vs successful work, every dataset. This is
       the core evidence that the policy abandons rather than hurries.
  fig2 steps vs tokens — relative savings side by side. Shows the step figure
       understates the cost saving ~3x, and carries the wider token CIs honestly.
  fig3 forest — every paired comparison with its CI on one axis, so the reader
       sees the whole evidence base and its spread at once.

Titles state what is plotted, never the conclusion: an earlier draft titled a
figure with its own hypothesis, which would have been a false caption on any
other outcome.

Usage: .venv/bin/python -m analysis.u4_figures --out-dir experiments/reports/figs
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

from common import FOUNDATION_ROOT
from eval.metrics import bootstrap_ci

RES = 10000
SEED = 42
C_DOOM, C_OK, C_TOK, C_STEP = "#D55E00", "#0072B2", "#009E73", "#666666"

SETS = [
    ("HotpotQA", "s5_eval/control_small.jsonl", "s5_eval/treatment_small.jsonl"),
    ("SimpleQA\n(unseen)", "t2_simpleqa/control.jsonl", "t2_simpleqa/treatment.jsonl"),
    ("MuSiQue\nseed 42", "t4_musique/control_small.jsonl", "t4_musique/mqtrtmatched_small.jsonl"),
    ("MuSiQue\nseed 123", "t3_seeds/s123_ctrl.jsonl", "t3_seeds/s123_trt.jsonl"),
    ("MuSiQue\nseed 789", "t3_seeds/s789_ctrl.jsonl", "t3_seeds/s789_trt.jsonl"),
]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)


def load(root, cp, tp):
    c, t = root / cp, root / tp
    if not (c.exists() and t.exists()):
        return None
    C = {e["task_id"]: e for e in (json.loads(l) for l in open(c) if l.strip())}
    T = {e["task_id"]: e for e in (json.loads(l) for l in open(t) if l.strip())}
    ks = sorted(set(C) & set(T))
    if not ks:
        return None
    tok = lambda e: sum(s.get("prompt_tokens", 0) + s.get("completion_tokens", 0)
                        for s in e["steps"])
    return dict(
        ks=ks,
        ds=np.array([T[k]["steps_used"] - C[k]["steps_used"] for k in ks], float),
        dt=np.array([tok(T[k]) - tok(C[k]) for k in ks], float),
        cs=np.array([C[k]["steps_used"] for k in ks], float),
        ct=np.array([tok(C[k]) for k in ks], float),
        failed=np.array([C[k]["final_f1"] <= 0 for k in ks]))


def fig1(data, out):
    labels = [l for l, d in data]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axhline(0, color="#444444", linewidth=0.9)
    for i, (lab, d) in enumerate(data):
        for off, mask, col, nm in ((-0.16, d["failed"], C_DOOM, "doomed"),
                                   (0.16, ~d["failed"], C_OK, "successful")):
            v = d["ds"][mask]
            if len(v) < 5:
                continue
            lo, hi = bootstrap_ci(v, RES, SEED)
            ax.plot([i + off, i + off], [lo, hi], color=col, linewidth=1.8,
                    solid_capstyle="round")
            ax.plot([i + off], [v.mean()], marker="o" if col == C_DOOM else "s",
                    color=col, markersize=6.5, zorder=3,
                    label=nm if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Δ steps  (treatment − control), paired")
    ax.set_title("Δ steps split by whether the control answered the question",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax); fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def fig2(data, out):
    labels = [l for l, d in data]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axhline(0, color="#444444", linewidth=0.9)
    for i, (lab, d) in enumerate(data):
        rs = 100 * d["ds"].mean() / d["cs"].mean()
        rt = 100 * d["dt"].mean() / d["ct"].mean() if d["ct"].mean() > 0 else np.nan
        slo, shi = bootstrap_ci(d["ds"], RES, SEED)
        tlo, thi = bootstrap_ci(d["dt"], RES, SEED)
        ax.plot([i - 0.16] * 2, [100 * slo / d["cs"].mean(), 100 * shi / d["cs"].mean()],
                color=C_STEP, linewidth=1.8, solid_capstyle="round")
        ax.plot([i - 0.16], [rs], marker="s", color=C_STEP, markersize=6.5, zorder=3,
                label="steps" if i == 0 else None)
        if not np.isnan(rt):
            ax.plot([i + 0.16] * 2, [100 * tlo / d["ct"].mean(), 100 * thi / d["ct"].mean()],
                    color=C_TOK, linewidth=1.8, solid_capstyle="round")
            ax.plot([i + 0.16], [rt], marker="D", color=C_TOK, markersize=6.5, zorder=3,
                    label="tokens" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("relative saving (%)")
    ax.set_title("Relative saving in steps vs in tokens, with 95% CIs",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax); fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def fig3(data, out):
    rows = []
    for lab, d in data:
        lo, hi = bootstrap_ci(d["ds"], RES, SEED)
        rows.append((lab.replace("\n", " "), d["ds"].mean(), lo, hi, len(d["ks"])))
    # MuSiQue seeds share the SAME 600 eval questions, so concatenating them
    # would double-count and narrow the CI. Average the seeds PER QUESTION first,
    # then pool over UNIQUE questions. (corrected 2026-08-06)
    by_q, order = {}, []
    for lab, d in data:
        for k, v in zip(d["ks"], d["ds"]):
            key = (lab.split("seed")[0].strip(), k)      # collapse seeds of a dataset
            if key not in by_q:
                by_q[key] = []; order.append(key)
            by_q[key].append(v)
    pooled = np.array([np.mean(by_q[k]) for k in order], float)
    plo, phi = bootstrap_ci(pooled, RES, SEED)
    rows.append(("POOLED (unique Qs)", pooled.mean(), plo, phi, len(pooled)))

    fig, ax = plt.subplots(figsize=(7.0, 0.55 * len(rows) + 1.6))
    ax.axvline(0, color="#444444", linewidth=0.9)
    for i, (lab, m, lo, hi, n) in enumerate(rows):
        y = len(rows) - 1 - i
        bold = lab.startswith("POOLED")
        ax.plot([lo, hi], [y, y], color="#111111" if bold else "#0072B2",
                linewidth=2.4 if bold else 1.7, solid_capstyle="round")
        ax.plot([m], [y], marker="D", markersize=8 if bold else 6.5,
                color="#111111" if bold else "#0072B2", zorder=3)
        ax.text(0.06, y, f"  {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={n}",
                fontsize=8.5, va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9)
    ax.set_xlabel("Δ steps  (treatment − control), paired")
    ax.set_title("All paired comparisons", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    root = FOUNDATION_ROOT / "experiments/results"
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)

    data = []
    for lab, cp, tp in SETS:
        d = load(root, cp, tp)
        if d:
            data.append((lab, d))
        else:
            print(f"  skipped {lab.replace(chr(10),' ')} (not available)")
    if not data:
        print("no data"); sys.exit(1)

    for fn, name in ((fig1, "u4_fig1_selectivity.pdf"),
                     (fig2, "u4_fig2_steps_vs_tokens.pdf"),
                     (fig3, "u4_fig3_forest.pdf")):
        print(f"wrote {fn(data, outd / name)}")


if __name__ == "__main__":
    main()
