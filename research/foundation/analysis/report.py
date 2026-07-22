"""F7 report generator: numbers in -> plain-language markdown skeleton out.

Fills every table/number from CSVs (no hand-edited values); prose sections are
scaffolded with TODO markers for the human pass. Section order fixed by the F7
doc. Usage:
  .venv/bin/python -m analysis.report --rows rows.csv [--divergence div.jsonl] --out report.md
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import git_hash, load_config
from eval.gate_check import evaluate_gate
from eval.metrics import aggregate, paired_delta


def build_report(rows: pd.DataFrame, cfg: dict, run_date: str,
                 divergence: list[dict] | None = None) -> str:
    gate = evaluate_gate(rows, cfg)
    B = gate["budget_B"]
    agg = aggregate(rows[rows.budget_B == B], n_resamples=2000)
    lines = [
        f"# Foundation run report — {run_date} (git {git_hash()})",
        "",
        "## 1. What we tested",
        "TODO(3 sentences): trained stopping vs prompt-only vs enforcement on "
        "HotpotQA under step budgets.",
        "",
        f"## 2. Gate verdict: **{gate['verdict']}** (medium budget B={B})",
    ]
    for k in ("cond1_utility", "cond2_self_stop", "cond3_no_collapse"):
        lines.append(f"- {k}: {'PASS' if gate[k]['passed'] else 'FAIL'} — {gate[k]}")
    lines += ["", f"## 3. Arm results at B={B}",
              "", "| arm | F1 (95% CI) | steps | utility | self-stop |",
              "|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r.arm} | {r.f1:.3f} ({r.f1_lo:.3f}–{r.f1_hi:.3f}) "
            f"| {r.steps_used:.2f} | {r.utility:.3f} | {r.self_stopped:.0%} |")
    for other in ("a1", "a2"):
        try:
            d = paired_delta(rows, "a3", other, B, "utility", n_resamples=2000)
            lines.append(f"\nPaired a3−{other} utility: {d['mean_delta']:+.3f} "
                         f"(CI {d['ci_lo']:+.3f}…{d['ci_hi']:+.3f}, "
                         f"a3 wins {d['frac_tasks_a_wins']:.0%} of tasks)")
        except (ValueError, KeyError):
            lines.append(f"\nPaired a3−{other}: (arm missing — fill after runs)")
    lines += ["", "## 4. Surprises & qualitative examples",
              "TODO: 2–3 quoted trajectories (incl. six-dimension diagnostic "
              "verdict: did RL improve only stopping, or also search skill?)",
              "", "## 5. Judge behavior"]
    if divergence:
        first, last = divergence[0], divergence[-1]
        lines.append(
            f"- divergence: judge {first['judge_score_mean']:.3f}→"
            f"{last['judge_score_mean']:.3f}, F1 {first['f1_mean']:.3f}→"
            f"{last['f1_mean']:.3f} over {len(divergence)} batches "
            "(TODO: reading — parallel rise = healthy; judge-up/F1-flat = hacked)")
    else:
        lines.append("- TODO: calibration table + divergence curve reading")
    lines += ["", "## 6. What this means for paper_plan_v2_1",
              "TODO: adjustment list — confirmed / contradicted / investigate.",
              "", "## 7. Run costs",
              "TODO: GPU-hours, judge calls (JudgeStats), wall-clock per stage."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--divergence", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    div = None
    if args.divergence and Path(args.divergence).exists():
        div = [json.loads(l) for l in open(args.divergence) if l.strip()]
    text = build_report(pd.read_csv(args.rows), load_config(),
                        date.today().isoformat(), div)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
