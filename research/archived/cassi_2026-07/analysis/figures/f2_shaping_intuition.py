"""F2 — shaping intuition on ONE trajectory (paper_plan_v2 §9): U_t, Cont(x_t),
the stop margin Δ_t = Cont − U_t, and τ* = min{t : U_t ≥ Cont(x_t)} drawn on a
single real trajectory from the P3 label run.

Expected CSV schema (experiments/results/f2_trajectory.csv — one trajectory):
    t     : int, 1-based step index (rows sorted by t)
    u_t   : float, stopping utility U_t = q_t − Σ λ·m(tier)·c̃ (§2.2)
    cont  : float, fitted continuation value Ê[V_{t+1}|x_t] (NaN at t = T)
τ* is recomputed here from the two curves (never a hand-entered column).

Usage:
    python analysis/figures/f2_shaping_intuition.py --results experiments/results/f2_trajectory.csv
    python analysis/figures/f2_shaping_intuition.py --demo          # synthetic trajectory,
                                                                    # ONLY with the flag (§16 P10)
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
import pandas as pd  # noqa: E402

from cassi.analysis.plotstyle import PALETTE, apply_style, finish, load_results  # noqa: E402

SCRIPT = "f2_shaping_intuition"


def demo_trajectory(T: int = 10, lam_cost: float = 0.055, seed: int = 7) -> pd.DataFrame:
    """Synthetic single trajectory with the canonical inverted-U economics:
    quality saturates, cost accrues linearly, so U_t peaks mid-trajectory.
    Cont is built by the same backward recursion the Snell labels use (§2.2),
    with mild noise standing in for regression error. Demo data ONLY — the
    figure renders it exclusively under --demo."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, T + 1)
    q = 0.92 * (1.0 - np.exp(-t / 2.8))                  # saturating draft quality
    u = q - lam_cost * t                                  # U_t = q_t − λ·C̃_t
    v = np.empty(T)
    cont = np.full(T, np.nan)
    v[-1] = u[-1]
    for i in range(T - 2, -1, -1):                        # V_t = max(U_t, Cont_t)
        cont[i] = v[i + 1] - 0.004 + rng.normal(0, 0.004)
        v[i] = max(u[i], cont[i])
    return pd.DataFrame({"t": t, "u_t": u, "cont": cont})


def tau_star(df: pd.DataFrame) -> int:
    """τ* = min{t : U_t ≥ Cont(x_t)}; at t = T, Cont is undefined ⇒ STOP (§2.2)."""
    for _, row in df.iterrows():
        if np.isnan(row["cont"]) or row["u_t"] >= row["cont"]:
            return int(row["t"])
    return int(df["t"].iloc[-1])


def build(df: pd.DataFrame, out: str, demo: bool) -> None:
    apply_style()
    df = df.sort_values("t").reset_index(drop=True)
    ts = tau_star(df)
    t, u, cont = df["t"].to_numpy(), df["u_t"].to_numpy(), df["cont"].to_numpy()
    delta = cont - u                                       # stop margin (>0 ⇒ continue)

    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(3.6, 3.4), sharex=True, height_ratios=[2.0, 1.0],
        gridspec_kw={"hspace": 0.12},
    )
    ax.plot(t, u, color=PALETTE[0], marker="o", label="$U_t$ (stop now)")
    ax.plot(t, cont, color=PALETTE[5], marker="s", linestyle="--",
            label="$\\mathrm{Cont}(x_t)=\\hat{E}[V_{t+1}|x_t]$")
    for a in (ax, axd):
        a.axvline(ts, color="#52514e", linewidth=1.0, linestyle=":")
    ax.annotate("$\\tau^{*}$: first $U_t \\geq \\mathrm{Cont}$",
                xy=(ts, 0.97), xycoords=("data", "axes fraction"),
                xytext=(4, -2), textcoords="offset points",
                va="top", fontsize=7.2, color="#52514e")
    ax.set_ylabel("utility ($q - \\lambda\\tilde{C}$)")
    ax.legend(loc="lower right")
    title = "Snell stopping on one trajectory"
    ax.set_title(title + (" (DEMO DATA)" if demo else ""))

    # Δ margin — same units (utility), separate panel: no dual axis.
    axd.bar(t, np.nan_to_num(delta), width=0.62,
            color=[PALETTE[1] if d > 0 else PALETTE[7] for d in np.nan_to_num(delta)])
    axd.axhline(0.0, color="#52514e", linewidth=0.8)
    axd.set_ylabel("$\\Delta^{*}_t$")
    axd.set_xlabel("step $t$")
    axd.text(0.97, 0.82, "$\\Delta>0$: continue", transform=axd.transAxes,
             ha="right", va="top", fontsize=6.8, color=PALETTE[1])
    axd.text(0.97, 0.18, "$\\Delta\\leq 0$: stop", transform=axd.transAxes,
             ha="right", va="bottom", fontsize=6.8, color=PALETTE[7])

    finish(fig, out, SCRIPT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/results/f2_trajectory.csv")
    ap.add_argument("--out", default="paper/figures/f2_shaping_intuition.pdf")
    ap.add_argument("--demo", action="store_true",
                    help="render a synthetic trajectory (the ONLY path that may "
                         "fabricate data — never used for the paper)")
    args = ap.parse_args()

    if args.demo:
        build(demo_trajectory(), args.out, demo=True)
        return
    df = load_results(args.results, SCRIPT, required_cols=("t", "u_t", "cont"))
    if df is None:
        return  # graceful skip — no demo fallback without the flag (§16 P10)
    build(df, args.out, demo=False)


if __name__ == "__main__":
    main()
