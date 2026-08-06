"""Apply the PRE-REGISTERED rule to the λ ablation and print the verdict.

The rule (experiments/reports/ablation_preregistration.md, committed before any
ablation run, quoted verbatim):

    the step-cost term is declared EFFECTIVE iff
      (1) mean_steps(λ=0) − mean_steps(λ=1.0) >= 0.5 at B=4, and
      (2) their 95% bootstrap CIs (10,000 resamples, paired per task) do not overlap.
    Secondary (supporting, not decisive): self-stop rate rises with λ; the
    steps-vs-λ curve is monotone.

Nothing here is tunable. The thresholds come from the pre-registration, the
scoring λ comes from config (the fixed yardstick, NOT whatever λ an arm was
trained with), and the raw metrics are λ-independent by construction.

Usage: .venv/bin/python scripts/analyse_lambda_ablation.py --dir experiments/results/lambda_eval
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common import load_config
from eval.metrics import bootstrap_ci

ARMS = [("lam0", 0.0), ("lam03", 0.3), ("lam10", 1.0), ("lam10_r2", 1.0)]
# lam10_r2 = λ=1.0 at its last HEALTHY checkpoint (round 3 failed its probe).
# Reported for sensitivity; the pre-registered rule uses lam10 (round 3),
# which is protocol-matched to the other arms.
# Budgets come from the config, never hard-coded. This literal held the
# FOUNDATION-1 values {2,4,8} while the config moved to {2,3,4} on
# 2026-07-31, so any re-run on post-redesign data computed self-stop and
# utility with the wrong B. (audit 2026-08-06)
BUDGETS = load_config()["episode"]["budgets"]
THRESHOLD = 0.5          # pre-registered
RESAMPLES = 10000        # pre-registered


def load(d: Path, tag: str, budget: str) -> list[dict]:
    p = d / f"{tag}_{budget}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    cfg = load_config()
    lam_eval = cfg["economy"]["lambda"]      # fixed yardstick for utility
    seed = cfg["seed"]

    print(f"Scoring every arm on the SAME yardstick: utility = F1 - {lam_eval}*(steps/B)")
    print("Raw metrics (steps, F1, self-stop) are λ-independent.\n")

    per_arm = {}
    for tag, lam in ARMS:
        rows = []
        for bname, B in BUDGETS.items():
            eps = load(d, tag, bname)
            if not eps:
                continue
            steps = np.array([e["steps_used"] for e in eps], float)
            f1 = np.array([e["final_f1"] for e in eps], float)
            stop = np.array([
                float(e["mode"] == "none" and e.get("answered_at") is not None
                      and not e["forced_stop"] and e["answered_at"] <= B)
                for e in eps])
            rows.append(dict(B=B, n=len(eps), steps=steps, f1=f1, stop=stop,
                             task=[e["task_id"] for e in eps]))
        per_arm[tag] = {r["B"]: r for r in rows}

    # ---- the table -----------------------------------------------------------
    print(f"{'arm':<8}{'train λ':>9}{'B':>4}{'n':>5}{'steps':>8}{'F1':>8}"
          f"{'self-stop':>11}{'utility':>9}")
    for tag, lam in ARMS:
        for B in sorted(per_arm.get(tag, {})):
            r = per_arm[tag][B]
            u = r["f1"] - lam_eval * (r["steps"] / B)
            print(f"{tag:<8}{lam:>9}{B:>4}{r['n']:>5}{r['steps'].mean():>8.3f}"
                  f"{r['f1'].mean():>8.3f}{r['stop'].mean():>11.3f}{u.mean():>9.3f}")
        if not per_arm.get(tag):
            print(f"{tag:<8}{lam:>9}   -- ABSENT --")

    # ---- the pre-registered rule --------------------------------------------
    print("\n" + "=" * 62)
    print("PRE-REGISTERED RULE @ B=4")
    a0 = per_arm.get("lam0", {}).get(4)
    a10 = per_arm.get("lam10", {}).get(4)
    if not (a0 and a10):
        print("cannot evaluate: one of the λ=0 / λ=1.0 arms is missing at B=4")
        sys.exit(2)

    # paired per task, as pre-registered
    common = sorted(set(a0["task"]) & set(a10["task"]))
    i0 = {t: k for k, t in enumerate(a0["task"])}
    i1 = {t: k for k, t in enumerate(a10["task"])}
    s0 = np.array([a0["steps"][i0[t]] for t in common])
    s1 = np.array([a10["steps"][i1[t]] for t in common])
    delta = s0 - s1

    lo0, hi0 = bootstrap_ci(s0, RESAMPLES, seed)
    lo1, hi1 = bootstrap_ci(s1, RESAMPLES, seed)
    dlo, dhi = bootstrap_ci(delta, RESAMPLES, seed)

    print(f"paired tasks: {len(common)}")
    print(f"  mean steps λ=0.0 : {s0.mean():.3f}  95% CI [{lo0:.3f}, {hi0:.3f}]")
    print(f"  mean steps λ=1.0 : {s1.mean():.3f}  95% CI [{lo1:.3f}, {hi1:.3f}]")
    print(f"  paired Δ (0 − 1.0): {delta.mean():+.3f}  95% CI [{dlo:+.3f}, {dhi:+.3f}]")

    cond1 = delta.mean() >= THRESHOLD
    cond2 = not (lo0 <= hi1 and lo1 <= hi0)          # CIs disjoint
    print(f"\n  cond1  Δ >= {THRESHOLD}                : {'PASS' if cond1 else 'FAIL'}"
          f"  ({delta.mean():+.3f})")
    print(f"  cond2  95% CIs non-overlapping   : {'PASS' if cond2 else 'FAIL'}")
    verdict = "EFFECTIVE" if (cond1 and cond2) else "NOT EFFECTIVE"
    print(f"\n  VERDICT: the step-cost term is {verdict}")

    # ---- secondary, supporting only -----------------------------------------
    print("\nsecondary (supporting, not decisive):")
    PROTOCOL = {"lam0", "lam03", "lam10"}   # sensitivity arm excluded: it is a
                                            # second point at λ=1.0 and would
                                            # break a monotonicity test by design
    steps_by_lam = [(lam, per_arm[tag][4]["steps"].mean())
                    for tag, lam in ARMS
                    if tag in PROTOCOL and per_arm.get(tag, {}).get(4)]
    mono = all(b[1] <= a[1] + 1e-9 for a, b in zip(steps_by_lam, steps_by_lam[1:]))
    print("  steps-vs-λ @ B=4: " + "  ".join(f"λ={l}:{s:.3f}" for l, s in steps_by_lam)
          + f"   monotone decreasing: {mono}")
    stop_by_lam = [(lam, per_arm[tag][4]["stop"].mean())
                   for tag, lam in ARMS
                   if tag in PROTOCOL and per_arm.get(tag, {}).get(4)]
    rising = all(b[1] >= a[1] - 1e-9 for a, b in zip(stop_by_lam, stop_by_lam[1:]))
    print("  self-stop-vs-λ  : " + "  ".join(f"λ={l}:{s:.3f}" for l, s in stop_by_lam)
          + f"   rising with λ: {rising}")

    # ---- power note ---------------------------------------------------------
    n = len(common)
    half = (dhi - dlo) / 2
    print(f"\npower: n={n} paired tasks; the Δ CI half-width is {half:.3f} steps. "
          f"An effect smaller than about that is not resolvable here — say so "
          f"rather than calling it zero.")
    if not cond1 and delta.mean() > 0.15:
        print("NOTE: Δ is positive and non-trivial but under threshold — this is the "
              "'real but weak' case the pre-registration flagged as warranting a "
              "4th λ point (1.5), not a flat null.")
    sys.exit(0 if (cond1 and cond2) else 1)


if __name__ == "__main__":
    main()
