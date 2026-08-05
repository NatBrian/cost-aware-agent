"""T3 — is the MuSiQue result stable across seeds?

Seed 42 comes from T4 (round-matched, both arms at r1). Seeds 123 and 789 are
trained here under an identical protocol and evaluated at their own matched round.

WHAT COUNTS AS REPLICATION, fixed before the numbers:
  - all three seeds negative, and
  - the pooled mean across seeds negative with its CI excluding zero.
A single seed flipping sign does not void the result, but it must be reported and
it caps how strongly the effect can be stated.

WHAT WOULD NOT COUNT: quoting the mean of three seeds while hiding the spread.
The per-seed values are printed individually for exactly that reason.

Note the seeds may be matched at DIFFERENT rounds — the λ=0 control fails its
health gate at a seed-dependent point on MuSiQue. That is itself reported, since
"how long the control survives" is one of the findings.

Usage: .venv/bin/python scripts/t3_analyse.py --dir experiments/results/t3_seeds
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common import FOUNDATION_ROOT, load_config
from eval.metrics import bootstrap_ci

RESAMPLES = 10000
SEED42_DIR = "experiments/results/t4_musique"      # ctrl vs mqtrtmatched, both r1
SEED42_LABEL = 42


def paired(cpath: Path, tpath: Path):
    if not (cpath.exists() and tpath.exists()):
        return None
    c = {e["task_id"]: e for e in (json.loads(l) for l in open(cpath) if l.strip())}
    t = {e["task_id"]: e for e in (json.loads(l) for l in open(tpath) if l.strip())}
    ks = sorted(set(c) & set(t))
    if not ks:
        return None
    return (np.array([t[k]["steps_used"] - c[k]["steps_used"] for k in ks], float),
            np.array([t[k]["final_f1"] - c[k]["final_f1"] for k in ks], float),
            np.array([c[k]["final_f1"] <= 0 for k in ks]), ks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    root = FOUNDATION_ROOT
    seed = load_config()["seed"]

    runs = []
    r42 = paired(root / SEED42_DIR / "control_small.jsonl",
                 root / SEED42_DIR / "mqtrtmatched_small.jsonl")
    if r42:
        runs.append((SEED42_LABEL, *r42))
    for s in (123, 789):
        r = paired(d / f"s{s}_ctrl.jsonl", d / f"s{s}_trt.jsonl")
        if r:
            runs.append((s, *r))

    if not runs:
        print("no seed runs found"); sys.exit(2)

    print(f"T3 — seed replication on MuSiQue, gate budget B=2, "
          f"round-matched, {len(runs)} seeds\n")
    print(f"{'seed':>6}{'n':>6}{'Δsteps':>10}{'95% CI':>22}{'ΔF1':>9}{'matched at':>12}")
    per = []
    for s, ds, df, failed, ks in runs:
        lo, hi = bootstrap_ci(ds, RESAMPLES, seed)
        prov = d / f"provenance_s{s}.json"
        rm = json.loads(prov.read_text())["match_round"] if prov.exists() else 1
        print(f"{s:>6}{len(ds):>6}{ds.mean():>10.3f}"
              f"{f'[{lo:+.3f},{hi:+.3f}]':>22}{df.mean():>+9.3f}{f'r{rm}':>12}"
              + ("  *" if hi < 0 or lo > 0 else ""))
        per.append(ds.mean())

    print(f"\n{'-'*60}")
    print("ACROSS SEEDS")
    per = np.array(per, float)
    print(f"   per-seed Δsteps: " + ", ".join(f"{v:+.3f}" for v in per))
    print(f"   mean {per.mean():+.3f}   sd {per.std(ddof=1) if len(per) > 1 else 0:.3f}"
          f"   range [{per.min():+.3f}, {per.max():+.3f}]")
    all_neg = bool((per < 0).all())
    print(f"   all seeds negative: {all_neg}")

    # pooled across every episode of every seed
    pooled = np.concatenate([r[1] for r in runs])
    plo, phi = bootstrap_ci(pooled, RESAMPLES, seed)
    print(f"   pooled (n={len(pooled)}): {pooled.mean():+.3f}  [{plo:+.3f},{phi:+.3f}]"
          + ("  *" if phi < 0 else ""))

    print(f"\n   REPLICATION: "
          + ("CONFIRMED — all seeds negative and the pooled CI excludes zero"
             if all_neg and phi < 0 else
             "PARTIAL — see the per-seed spread above; state the effect no more "
             "strongly than the weakest seed supports"))

    # selectivity pooled across seeds
    print(f"\n{'-'*60}")
    print("H2 selectivity, pooled across seeds")
    dsa = np.concatenate([r[1] for r in runs])
    fa = np.concatenate([r[3] for r in runs])
    for name, m in (("control FAILED", fa), ("control SUCCEEDED", ~fa)):
        v = dsa[m]
        lo, hi = bootstrap_ci(v, RESAMPLES, seed)
        print(f"   {name:<22} n={len(v):>5}  {v.mean():+.3f}  [{lo:+.3f},{hi:+.3f}]"
              + ("  *" if hi < 0 or lo > 0 else ""))


if __name__ == "__main__":
    main()
