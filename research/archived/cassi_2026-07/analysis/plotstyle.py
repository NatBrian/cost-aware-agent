"""Shared figure style + CSV plumbing for F1–F6 — paper_plan_v2 §9, §16 P10.

Rules enforced here so every figure inherits them:
* colorblind-safe categorical palette, validated (CVD ΔE ≥ 8 adjacent, dataviz
  six-checks) — hues assigned to METHODS in fixed order, never cycled by rank;
  markers/linestyles ride along as secondary encoding (three palette slots sit
  below 3:1 contrast on white, so identity is never color-alone);
* error bars = 95% bootstrap CI (precomputed *_ci_lo/_ci_hi columns in the
  results CSVs — produced by cassi.eval.stats.bootstrap_ci at P9);
* CSV in → PDF out, no hand-edited numbers: every script exits gracefully with
  a message when its results CSV is missing (only f2's --demo may synthesize).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — figures are files, never windows

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Validated colorblind-safe categorical palette (fixed order — never re-rank).
PALETTE = [
    "#2a78d6",  # 1 blue
    "#008300",  # 2 green
    "#e87ba4",  # 3 magenta
    "#eda100",  # 4 yellow
    "#1baf7a",  # 5 aqua
    "#eb6834",  # 6 orange
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

# Color follows the ENTITY: canonical method → (palette slot, marker idx),
# stable across every figure regardless of which methods a CSV contains.
# 10 methods on an 8-hue palette: the two hue-sharing pairs (b3/b8, b7/b9)
# carry distinct markers as secondary encoding; `oracle` is a reference bound,
# not a competing series — it renders neutral gray.
CANONICAL_METHODS = {
    "cassi": (0, 0), "b1_react": (1, 1), "b2_probe": (2, 2),
    "b3_supervisor": (3, 3), "b4_otc": (4, 4), "b5_eapo": (5, 5),
    "b6_single_model": (6, 6), "b7_cart": (7, 7),
    "b8_agentprm_cost": (3, 0), "b9_direct_shaping": (7, 2),
}
ORACLE_COLOR = "#52514e"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dddddd"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "errorbar.capsize": 2.0,
        "pdf.fonttype": 42,   # embed TrueType — camera-ready requirement
    })


def series_style(method: str, fallback_idx: int = 0) -> dict:
    """Stable (color, marker, linestyle) for a method — canonical methods keep
    their slot everywhere; unknown methods take `fallback_idx` (order of first
    appearance in the calling script, offset past the canonical range)."""
    key = str(method).strip().lower()
    if key.startswith("oracle"):
        return {"color": ORACLE_COLOR, "marker": "D", "linestyle": ":"}
    for canon, (ci, mi) in CANONICAL_METHODS.items():
        if key == canon or key.startswith(canon):
            break
    else:
        ci = mi = fallback_idx % len(PALETTE)
    return {"color": PALETTE[ci % len(PALETTE)], "marker": MARKERS[mi % len(MARKERS)],
            "linestyle": LINESTYLES[mi % len(LINESTYLES)]}


def load_results(csv_path: str | Path, script: str,
                 required_cols: tuple[str, ...] = ()) -> pd.DataFrame | None:
    """Read a results CSV; on a missing file print the §16 P10 skip message and
    return None (caller exits 0 — `make figures tables` must not fail on a
    partially-run P9). Missing REQUIRED columns are a hard error: that is a
    schema bug, not a not-yet-run experiment."""
    p = Path(csv_path)
    if not p.exists():
        print(f"[{script}] results CSV not found: {p} — skipping "
              "(run P9 to produce experiments/results/*.csv; see the script "
              "docstring for the expected schema).")
        return None
    df = pd.read_csv(p)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[{script}] {p} is missing required columns {missing} "
                         f"(have: {list(df.columns)}) — fix the P9 emitter, "
                         "never hand-edit results.")
    return df


def yerr_from_ci(df: pd.DataFrame, value: str, lo: str, hi: str):
    """Asymmetric matplotlib yerr from precomputed 95% bootstrap CI columns."""
    import numpy as np
    return np.vstack([
        (df[value] - df[lo]).clip(lower=0).to_numpy(),
        (df[hi] - df[value]).clip(lower=0).to_numpy(),
    ])


def finish(fig, out_path: str | Path, script: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[{script}] wrote {out}")


def bootstrap_research_path() -> None:  # pragma: no cover — used by __main__ scripts
    """Standalone-script hook: make `import cassi.*` resolve when a figure/table
    script is run directly (conftest.py does this for pytest only)."""
    research = str(Path(__file__).resolve().parents[2])
    if research not in sys.path:
        sys.path.insert(0, research)
