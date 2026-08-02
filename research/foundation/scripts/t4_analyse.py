"""T4 — does the cost-aware effect grow with horizon?

Step 1 gave −0.167 steps on HotpotQA, a ~3-step task: about 6%. Whether that is
interesting depends entirely on whether it **scales**. MuSiQue is matched on every
axis except the dataset (same 300 train tasks, G=8, 3 rounds, λ pair, 600-question
eval, gate budget B=2), so a difference is attributable to the task, not to volume.

Two independent readings of "does it grow":

  BETWEEN datasets — |Δsteps| on MuSiQue vs HotpotQA's −0.167.
  WITHIN MuSiQue   — |Δsteps| by hop count (2 / 3 / 4). This is the stronger
                     test: it holds dataset, policy and training fixed and varies
                     only the required horizon, so it cannot be explained by any
                     between-dataset difference.

Also re-runs the H2 selectivity test (saving must concentrate on doomed work) and
the T1/T2 quality decomposition, because the regularisation confound found in T2
must be checked here too rather than assumed absent.

Usage: .venv/bin/python scripts/t4_analyse.py --dir experiments/results/t4_musique
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
HOTPOT_DSTEPS = -0.167          # Step 1, B=2, for the between-dataset comparison
HOTPOT_DF1_UNCHANGED = 0.092


def load(d: Path, arm: str, bname: str) -> dict:
    p = d / f"{arm}_{bname}.jsonl"
    if not p.exists():
        return {}
    return {e["task_id"]: e for e in (json.loads(l) for l in open(p) if l.strip())}


def ci(v, seed, label, w=38):
    v = np.asarray(v, float)
    if len(v) < 5:
        print(f"   {label:<{w}} n={len(v)} — too few")
        return None
    lo, hi = bootstrap_ci(v, RESAMPLES, seed)
    sig = lo > 0 or hi < 0
    print(f"   {label:<{w}} {v.mean():+.3f}  [{lo:+.3f},{hi:+.3f}]" + ("  *" if sig else ""))
    return v.mean(), lo, hi, sig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    cfg = load_config()
    seed = cfg["seed"]
    budgets = cfg["episode"]["budgets"]
    gate = cfg["episode"].get("gate_budget", "small")
    B = budgets[gate]

    hops = {}
    hp = FOUNDATION_ROOT / "data/musique_eval_600.jsonl"
    if hp.exists():
        for line in open(hp):
            r = json.loads(line)
            hops[r["id"]] = r.get("hops", 2)

    c, t = load(d, "control", gate), load(d, "treatment", gate)
    ks = sorted(set(c) & set(t))
    if not ks:
        print("missing an arm at the gate budget"); sys.exit(2)
    ds = np.array([t[k]["steps_used"] - c[k]["steps_used"] for k in ks], float)
    df = np.array([t[k]["final_f1"] - c[k]["final_f1"] for k in ks], float)

    print(f"T4 — MuSiQue, gate budget B={B}, n={len(ks)}\n")
    print(f"   control   steps {np.mean([c[k]['steps_used'] for k in ks]):.3f}  "
          f"F1 {np.mean([c[k]['final_f1'] for k in ks]):.3f}")
    print(f"   treatment steps {np.mean([t[k]['steps_used'] for k in ks]):.3f}  "
          f"F1 {np.mean([t[k]['final_f1'] for k in ks]):.3f}\n")

    print("1. BETWEEN datasets — does the effect grow?")
    r = ci(ds, seed, "Δsteps on MuSiQue")
    print(f"   {'Δsteps on HotpotQA (Step 1)':<38} {HOTPOT_DSTEPS:+.3f}")
    if r:
        print(f"   -> ratio {abs(r[0]) / abs(HOTPOT_DSTEPS):.2f}x")

    print("\n2. WITHIN MuSiQue — Δsteps by required hops (the stronger test)")
    print("   holds dataset/policy/training fixed and varies only the horizon")
    if hops:
        for h in (2, 3, 4):
            m = np.array([hops.get(k, 2) == h for k in ks])
            if m.sum() >= 5:
                ci(ds[m], seed, f"{h}-hop (n={int(m.sum())})")
    else:
        print("   (hop metadata unavailable)")

    print("\n3. H2 selectivity — must concentrate on doomed work")
    failed = np.array([c[k]["final_f1"] <= 0 for k in ks])
    ci(ds[failed], seed, f"control FAILED (n={int(failed.sum())})")
    ci(ds[~failed], seed, f"control SUCCEEDED (n={int((~failed).sum())})")

    print("\n4. the T2 regularisation confound — is it here too?")
    unchanged = ds >= 0
    ci(df[unchanged], seed, f"ΔF1 | steps not reduced (n={int(unchanged.sum())})")
    print(f"   {'HotpotQA equivalent':<38} {HOTPOT_DF1_UNCHANGED:+.3f}")
    print(f"   {'SimpleQA equivalent':<38} {0.086:+.3f}")
    print("   A similar value means the generic quality effect reproduces here")
    print("   as well, and remains a confound to disclose rather than a benefit.")

    print("\n5. all budgets")
    for bn, bb in budgets.items():
        cc, tt = load(d, "control", bn), load(d, "treatment", bn)
        kk = sorted(set(cc) & set(tt))
        if not kk:
            continue
        v = np.array([tt[k]["steps_used"] - cc[k]["steps_used"] for k in kk], float)
        f = np.array([tt[k]["final_f1"] - cc[k]["final_f1"] for k in kk], float)
        lo, hi = bootstrap_ci(v, RESAMPLES, seed)
        flo, fhi = bootstrap_ci(f, RESAMPLES, seed)
        print(f"   B={bb}  Δsteps {v.mean():+.3f} [{lo:+.3f},{hi:+.3f}]"
              f"   ΔF1 {f.mean():+.3f} [{flo:+.3f},{fhi:+.3f}]")


if __name__ == "__main__":
    main()
