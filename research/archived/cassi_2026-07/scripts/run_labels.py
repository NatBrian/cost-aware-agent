#!/usr/bin/env python3
"""P3 driver — Algorithm 1 (Snell-envelope labels) per lambda, plus QC + memo
(paper_plan_v2 §2.2, §10 Alg.1, §16 P3).

Per domain and per lambda in {0.1, 0.5, 1, 2, 5} (§17 label.lambda_values):
  * run cassi.labels.snell.snell_labels on the forced-continuation collection round
    (tier-scaled marginal costing by default; --plain-lambda gives the A8 arm, m == 1);
  * tanh scale s: per domain, fit ONCE on round-0 data at the default lambda, then
    reused across lambdas — if configs/cassi.yaml label.delta_scale.<domain> is set,
    that frozen value wins (§17);
  * write labels JSONL + tau* map + backup residuals (fitted-VI error check, §5.3).

QC (§16 P3):
  (a) export 100 random trajectories with full per-step label context for manual review;
  (b) label-noise sensitivity: re-run with step-subsampled draft scoring (every 2nd
      step, carry-forward) and report the tau* shift;
  (c) sanity: higher lambda => earlier tau* (cassi.labels.snell.qc_lambda_monotonicity).

Also computes the prophet-argmax tau (E4 comparison arm — never a training target).
Writes the one-page label-quality memo (P3 done-criterion artifact).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CASSI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CASSI_ROOT.parent))

from cassi.common.config import load_config, require_pilot_calibration  # noqa: E402
from cassi.common.schema import Trajectory, load_trajectories  # noqa: E402
from cassi.labels.snell import (  # noqa: E402
    prophet_labels,
    qc_lambda_monotonicity,
    qc_mean_tau,
    snell_labels,
)
from cassi.stopper.dataset import save_labelset  # noqa: E402  (meta-header JSONL — the stopper's loader format)


def subsample_quality(tr: Trajectory) -> Trajectory:
    """QC (b): pretend the draft was only scored every 2nd step (carry-forward)."""
    t2 = Trajectory.from_dict(tr.to_dict())
    last_q = 0.0
    for i, s in enumerate(t2.steps):
        if i % 2 == 0:
            last_q = s.q
        else:
            s.q = last_q
    return t2


def run_domain(domain: str, trajs: list[Trajectory], cfg: dict, out_dir: Path,
               plain_lambda: bool, review_n: int, seed: int) -> dict:
    med = cfg["label"]["cost_normalization"][f"{domain}_median_pilot_spend"]
    lambdas = [float(l) for l in cfg["label"]["lambda_values"]]
    default_lam = float(cfg["label"]["default_lambda"])
    frozen_s = (cfg["label"].get("delta_scale") or {}).get(domain)

    # --- fit s once per domain (on the default lambda), unless frozen in config ---
    first = snell_labels(trajs, default_lam, med, rule_table_off=plain_lambda,
                         scale_s=frozen_s, seed=seed)
    scale_s = first.scale_s
    if frozen_s is None:
        print(f"[{domain}] fitted tanh scale s = {scale_s:.6f} — write into configs/cassi.yaml "
              f"label.delta_scale.{domain} and FREEZE (§17)")

    tau_by_lambda: dict[float, dict] = {}
    stats: dict[float, dict] = {}
    for lam in lambdas:
        ls = first if lam == default_lam else snell_labels(
            trajs, lam, med, rule_table_off=plain_lambda, scale_s=scale_s, seed=seed)
        suffix = "_plainlam" if plain_lambda else ""
        path = out_dir / f"{domain}_lambda{lam:g}{suffix}.jsonl"
        save_labelset(ls, path)  # meta-header format read by cassi.stopper.dataset.load_labelset
        tau_by_lambda[lam] = {f"{k[0]}|{k[1]}": v for k, v in ls.tau_star.items()}
        stats[lam] = {
            "n_step_labels": len(ls.labels),
            "mean_tau_star": qc_mean_tau(ls),
            "stop_fraction": float(np.mean([l.a_star == "STOP" for l in ls.labels])),
            "mean_backup_residual": float(np.mean(ls.backup_residuals)) if ls.backup_residuals else None,
            "labels_file": str(path),
        }
        print(f"[{domain}] lambda={lam:g}: {stats[lam]['n_step_labels']} labels, "
              f"mean tau*={stats[lam]['mean_tau_star']:.2f}")

    (out_dir / f"{domain}_tau_star.json").write_text(json.dumps(
        {str(l): t for l, t in tau_by_lambda.items()}, indent=2))

    # --- QC (c): lambda-monotonicity (raw-keyed maps for the checker) ---
    raw_tau = {lam: {tuple(k.split("|")): v for k, v in m.items()} for lam, m in tau_by_lambda.items()}
    mono = qc_lambda_monotonicity(raw_tau)
    print(f"[{domain}] QC(c) lambda-monotonicity: {mono['n_violations']}/{mono['n_pairs']} "
          f"violating pairs ({mono['violation_rate']:.2%})")

    # --- QC (b): noise sensitivity via step-subsampled draft scoring ---
    sub = snell_labels([subsample_quality(t) for t in trajs], default_lam, med,
                       rule_table_off=plain_lambda, scale_s=scale_s, seed=seed)
    shifts = [abs(sub.tau_star[k] - first.tau_star[k]) for k in first.tau_star if k in sub.tau_star]
    noise = {"mean_abs_tau_shift": float(np.mean(shifts)) if shifts else None,
             "p90_abs_tau_shift": float(np.percentile(shifts, 90)) if shifts else None}
    print(f"[{domain}] QC(b) noise sensitivity: mean |dtau*| = {noise['mean_abs_tau_shift']:.3f}")

    # --- QC (a): 100-trajectory manual-review export ---
    rng = random.Random(seed)
    sample = rng.sample(trajs, min(review_n, len(trajs)))
    lab_by_step = {(l.task_id, l.rollout_idx, l.t): l
                   for l in first.labels}
    review_path = out_dir / f"{domain}_review_{len(sample)}.jsonl"
    with review_path.open("w") as f:
        for tr in sample:
            steps = []
            for i, s in enumerate(tr.steps, start=1):
                l = lab_by_step.get((tr.task_id, tr.rollout_idx, i))
                steps.append({"t": i, "action": s.a, "draft": s.draft, "q": s.q,
                              "c": s.c, "tier": s.tier,
                              "U": l.u_t if l else None, "delta_raw": l.delta_raw if l else None,
                              "a_star": l.a_star if l else None,
                              "answered_flag": s.answered_flag})
            f.write(json.dumps({
                "task_id": tr.task_id, "rollout_idx": tr.rollout_idx,
                "wallet": tr.wallet_size, "tau_star": first.tau_star[(tr.task_id, tr.rollout_idx)],
                "outcome": tr.outcome, "steps": steps,
            }) + "\n")
    print(f"[{domain}] QC(a) review export -> {review_path}")

    # --- E4 comparison arm: prophet-argmax tau (never trained on) ---
    tprop = prophet_labels(trajs, default_lam, med, rule_table_off=plain_lambda)
    prophet_gap = float(np.mean([tprop[k] - first.tau_star[k] for k in first.tau_star]))
    print(f"[{domain}] prophet-argmax stops {prophet_gap:+.2f} steps later than Snell tau* "
          f"on average (foresight bias, §2.2)")

    return {"scale_s": scale_s, "frozen_s_used": frozen_s is not None,
            "per_lambda": {str(l): s for l, s in stats.items()},
            "monotonicity": mono, "noise_sensitivity": noise,
            "prophet_mean_tau_gap": prophet_gap, "review_file": str(review_path),
            "n_trajectories": len(trajs)}


MEMO_TEMPLATE = """# Label-quality memo — collection round {round}, {date}
(one page; P3 done-criterion artifact, paper_plan_v2 §16 P3)

Economy: {economy} (tier multipliers {tiers}; cost pilot-normalized, §2.1-2.2)
Lambda grid: {lambdas}

## Per-domain summary
{domain_blocks}

## QC (c) — lambda-monotonicity sanity (higher lambda => earlier tau*)
{mono_lines}
Violations above ~2% need investigation before P4 (regressor variance vs economy bug).

## QC (b) — label-noise sensitivity (step-subsampled draft scoring)
{noise_lines}
Snell regression pools G=8 rollouts, so shifts should be small (<1 step mean, §8 risks).

## QC (a) — manual review (100 random trajectories per domain)
Review files: {review_files}
TODO(manual): open each file, check ~20 trajectories per domain — does tau* sit where a
human would stop (draft stabilized, marginal step not worth its cost)? Note failure
patterns here:
- [ ] qa: ...
- [ ] alfworld: ...

## Prophet-bias check (E4 preview)
{prophet_lines}
(argmax labels stopping LATER than Snell is the predicted foresight bias, §2.2.)

## Verdict
- [ ] labels approved for P4 stopper training
- [ ] issues found -> fix before P4 (list):
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0, help="collection round (0 = P2, 1 = P7)")
    ap.add_argument("--domains", nargs="+", default=["qa", "alfworld"])
    ap.add_argument("--collect-dir", type=Path, default=None,
                    help="override trajectories dir (default experiments/collect/round<N>)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="override output dir (default experiments/labels/round<N>)")
    ap.add_argument("--plain-lambda", action="store_true",
                    help="A8 ablation arm: m(tier) == 1 (plain-lambda labels)")
    ap.add_argument("--review-sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=Path, default=None,
                    help="override configs/cassi.yaml (tests only — the runbook uses the default)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    collect_dir = args.collect_dir or CASSI_ROOT / "experiments" / "collect" / f"round{args.round}"
    out_dir = args.out_dir or CASSI_ROOT / "experiments" / "labels" / f"round{args.round}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for domain in args.domains:
        require_pilot_calibration(cfg, domain)
        src = collect_dir / f"{domain}.jsonl"
        if not src.exists():
            print(f"PENDING: {src} missing — run the collection phase first", file=sys.stderr)
            return 75
        trajs = list(load_trajectories(src))
        forced = [t for t in trajs if t.outcome.get("collection_mode") == "forced_continuation"]
        if len(forced) < len(trajs):
            print(f"[warn] {domain}: {len(trajs)-len(forced)} non-forced-continuation trajectories "
                  f"EXCLUDED (RL-mode rollouts re-introduce censoring, §2.1)")
        results[domain] = run_domain(domain, forced or trajs, cfg, out_dir,
                                     args.plain_lambda, args.review_sample, args.seed)

    (out_dir / "qc_summary.json").write_text(json.dumps(results, indent=2))

    memo = MEMO_TEMPLATE.format(
        round=args.round, date=datetime.now(timezone.utc).date(),
        economy="plain-lambda (A8 arm)" if args.plain_lambda else "tier-scaled marginal costing",
        tiers=cfg["label"]["tier_multipliers"], lambdas=cfg["label"]["lambda_values"],
        domain_blocks="\n".join(
            f"- **{d}**: {r['n_trajectories']} trajectories; s={r['scale_s']:.4g}"
            f" ({'frozen' if r['frozen_s_used'] else 'fitted this round — freeze in §17'});"
            f" mean tau* by lambda: "
            + ", ".join(f"{l}->{s['mean_tau_star']:.2f}" for l, s in r["per_lambda"].items())
            for d, r in results.items()),
        mono_lines="\n".join(
            f"- {d}: {r['monotonicity']['n_violations']}/{r['monotonicity']['n_pairs']} pairs "
            f"({r['monotonicity']['violation_rate']:.2%})" for d, r in results.items()),
        noise_lines="\n".join(
            f"- {d}: mean |dtau*| = {r['noise_sensitivity']['mean_abs_tau_shift']:.3f}, "
            f"p90 = {r['noise_sensitivity']['p90_abs_tau_shift']:.3f}" for d, r in results.items()),
        review_files=", ".join(r["review_file"] for r in results.values()),
        prophet_lines="\n".join(
            f"- {d}: prophet argmax stops {r['prophet_mean_tau_gap']:+.2f} steps later than Snell"
            for d, r in results.items()),
    )
    memo_path = out_dir / "label_quality_memo.md"
    memo_path.write_text(memo)
    print(f"\nMemo -> {memo_path}  (complete the manual-review TODOs before P4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
