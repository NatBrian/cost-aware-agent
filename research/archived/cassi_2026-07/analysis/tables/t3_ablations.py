"""T3 — ablations A1–A9 (paper_plan_v2 §9, §5.5): each ablation arm vs the
CASSI default, single seed (per §7 tiering — the caption must say so, §5.6).

Expected CSV schema (experiments/results/t3_ablations.csv — one row per
ablation arm on the primary domain):
    ablation    : str (A1 … A9)
    arm         : str — the arm's short name (e.g. "stopper_0.8B",
                  "additive_delta", "rule_table", "random_coach")
    accuracy    : float in [0,1]
    cost_usd    : float — mean all-inclusive dollars per task
    delta_acc   : float — accuracy minus the CASSI-default accuracy (pp/100)
    delta_cost_usd : float — cost minus the CASSI-default cost
    n_seeds     : int (1 for ablations, §5.6)

Usage: python analysis/tables/t3_ablations.py --results experiments/results/t3_ablations.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # research/ → cassi.*

from cassi.analysis.plotstyle import load_results  # noqa: E402
from cassi.analysis.texutils import fmt_num, tex_escape, write_tabular  # noqa: E402

SCRIPT = "t3_ablations"


def _signed(x, digits=3, dollar=False) -> str:
    s = fmt_num(abs(x) if x is not None else None, digits, dollar)
    if s == "--":
        return s
    try:
        return ("$+$" if float(x) >= 0 else "$-$") + s
    except (TypeError, ValueError):
        return s


def build(df, out: str, source: str) -> None:
    rows, midrules = [], []
    for ab in sorted(df["ablation"].unique()):
        sub = df[df["ablation"] == ab]
        midrules.append(len(rows))
        for _, r in sub.iterrows():
            rows.append([
                tex_escape(ab), tex_escape(r["arm"]),
                fmt_num(r["accuracy"], 3),
                fmt_num(r["cost_usd"], 3, dollar=True),
                _signed(r.get("delta_acc"), 3),
                _signed(r.get("delta_cost_usd"), 3, dollar=True),
                fmt_num(r.get("n_seeds"), 0),
            ])
    write_tabular(
        out, "llrrrrr",
        ["Abl.", "Arm", "Acc.", "Cost (USD)", r"$\Delta$acc",
         r"$\Delta$cost", "Seeds"],
        rows, script=SCRIPT, source_csv=source,
        midrules_before=tuple(m for m in midrules if m > 0),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/results/t3_ablations.csv")
    ap.add_argument("--out", default="paper/tables/t3_ablations.tex")
    args = ap.parse_args()
    df = load_results(args.results, SCRIPT,
                      required_cols=("ablation", "arm", "accuracy", "cost_usd"))
    if df is None:
        return
    build(df, args.out, args.results)


if __name__ == "__main__":
    main()
