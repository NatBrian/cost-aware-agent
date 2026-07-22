"""T5 — transfer by role (paper_plan_v2 §9, §5.1/E2): trained executor →
BrowseComp-Plus / Bamboogle / 2Wiki; stopper-as-monitor → GAIA-103 over a
frozen live-web agent; stopper transferred across executors (cross-family /
cross-scale).

§5.6 small-n policy is ENFORCED in the rendering: rows with n < 500 are
labeled "transfer indicator" — point estimate + bootstrap CI, never a
hypothesis test (GAIA-103, Bamboogle-125).

Expected CSV schema (experiments/results/t5_transfer.csv — one row per
role × target × metric):
    role     : str (executor_transfer | stopper_as_monitor | executor_swap)
    target   : str — eval set or receiving executor
             (browsecomp_plus, bamboogle, 2wiki, gaia103, ministral_3_8b, qwen3.5_4b)
    metric   : str (accuracy | cost_usd | savings_pct | self_stop_rate ...)
    value    : float
    ci_lo, ci_hi : 95% bootstrap CI (§5.6 — required; CIs are the ONLY
             uncertainty statement allowed at small n)
    n        : int — number of test instances

Usage: python analysis/tables/t5_transfer.py --results experiments/results/t5_transfer.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # research/ → cassi.*

from cassi.analysis.plotstyle import load_results  # noqa: E402
from cassi.analysis.texutils import fmt_ci, fmt_num, tex_escape, write_tabular  # noqa: E402
from cassi.eval.stats import small_n_guard  # noqa: E402

SCRIPT = "t5_transfer"


def build(df, out: str, source: str) -> None:
    rows, midrules = [], []
    for role in sorted(df["role"].unique()):
        sub = df[df["role"] == role]
        midrules.append(len(rows))
        for _, r in sub.iterrows():
            n = int(r["n"])
            note = "" if small_n_guard(n) else r"transfer indicator\textsuperscript{\dag}"
            rows.append([
                tex_escape(role), tex_escape(r["target"]), tex_escape(r["metric"]),
                fmt_ci(r["value"], r.get("ci_lo"), r.get("ci_hi"), digits=3),
                fmt_num(n, 0), note,
            ])
    write_tabular(
        out, "lllrrl",
        ["Role", "Target", "Metric", "Value [95\\% CI]", "$n$", ""],
        rows, script=SCRIPT, source_csv=source,
        midrules_before=tuple(m for m in midrules if m > 0),
    )
    # footnote line for the paper source to place under the table
    print(f"[{SCRIPT}] \\dag rows have n<500: point estimate + bootstrap CI only, "
          "no hypothesis tests (§5.6 small-n policy).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/results/t5_transfer.csv")
    ap.add_argument("--out", default="paper/tables/t5_transfer.tex")
    args = ap.parse_args()
    df = load_results(args.results, SCRIPT,
                      required_cols=("role", "target", "metric", "value", "n"))
    if df is None:
        return
    build(df, args.out, args.results)


if __name__ == "__main__":
    main()
