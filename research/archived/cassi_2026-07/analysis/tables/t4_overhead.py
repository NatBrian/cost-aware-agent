"""T4 — end-to-end overhead accounting per serving regime (paper_plan_v2 §9,
§5.3): every dollar a method's pipeline spends — rollouts, running-draft
template tokens, forced-continuation collection, stopper training (amortized),
stopper inference (KV-fork vs re-prefill), probe/monitor calls — plus the
analysis-only replay line, which is NEVER counted in a method total.

Billing symmetry (§5.3) is ENFORCED here: the script rebuilds
cassi.eval.overhead.MethodLedger objects and runs assert_billing_symmetry —
a CSV mixing price maps refuses to render.

Expected CSV schema (experiments/results/t4_overhead.csv — one row per
method × serving regime; *_usd totals over the frozen eval set, built from
MethodLedger.to_row()):
    method, regime (kv_fork | re_prefill)
    rollout_tokens_usd, draft_line_tokens_usd, forced_continuation_usd,
    stopper_training_usd, stopper_inference_usd, probe_monitor_usd,
    replay_analysis_usd
    price_map_input_per_1m, price_map_output_per_1m  : the price map used —
        must be IDENTICAL on every row (billing symmetry)

Usage: python analysis/tables/t4_overhead.py --results experiments/results/t4_overhead.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # research/ → cassi.*

from cassi.analysis.plotstyle import load_results  # noqa: E402
from cassi.analysis.texutils import fmt_num, tex_escape, write_tabular  # noqa: E402
from cassi.eval.overhead import MethodLedger, assert_billing_symmetry  # noqa: E402

SCRIPT = "t4_overhead"

_COMPONENTS = [
    ("rollout_tokens_usd", "Rollout"),
    ("draft_line_tokens_usd", "Draft line"),
    ("forced_continuation_usd", "Forced cont."),
    ("stopper_training_usd", "Stopper train (amort.)"),
    ("stopper_inference_usd", "Stopper infer."),
    ("probe_monitor_usd", "Probe/monitor"),
]


def build(df, out: str, source: str) -> None:
    ledgers = [
        MethodLedger(
            method=str(r["method"]), regime=str(r["regime"]),
            price_map={"input": float(r["price_map_input_per_1m"]),
                       "output": float(r["price_map_output_per_1m"])},
            **{k: float(r[k]) for k, _ in _COMPONENTS},
            replay_analysis_usd=float(r.get("replay_analysis_usd", 0.0)),
        )
        for _, r in df.iterrows()
    ]
    assert_billing_symmetry(ledgers)   # §5.3 — refuse to render asymmetric billing

    rows, midrules = [], []
    for regime in ("kv_fork", "re_prefill"):
        block = [l for l in ledgers if l.regime == regime]
        if not block:
            continue
        midrules.append(len(rows))
        for l in sorted(block, key=lambda x: x.method):
            rows.append(
                [tex_escape(regime), tex_escape(l.method)]
                + [fmt_num(getattr(l, k), 2, dollar=True) for k, _ in _COMPONENTS]
                + [rf"\textbf{{{fmt_num(l.method_total_usd(), 2, dollar=True)}}}",
                   fmt_num(l.replay_analysis_usd, 2, dollar=True)]
            )
    write_tabular(
        out, "ll" + "r" * (len(_COMPONENTS) + 2),
        ["Regime", "Method"] + [name for _, name in _COMPONENTS]
        + [r"\textbf{Method total}", "Replay (analysis)"],
        rows, script=SCRIPT, source_csv=source,
        midrules_before=tuple(m for m in midrules if m > 0),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/results/t4_overhead.csv")
    ap.add_argument("--out", default="paper/tables/t4_overhead.tex")
    args = ap.parse_args()
    df = load_results(
        args.results, SCRIPT,
        required_cols=("method", "regime", "price_map_input_per_1m",
                       "price_map_output_per_1m") + tuple(k for k, _ in _COMPONENTS),
    )
    if df is None:
        return
    build(df, args.out, args.results)


if __name__ == "__main__":
    main()
