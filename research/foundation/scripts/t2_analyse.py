"""T2 — the SimpleQA negative control: is the quality gain generic?

A NOTE ON WHAT THIS TEST ACTUALLY DISCRIMINATES (revised after the power check).

The plan justified SimpleQA as "single-hop, so no discretionary work exists, so a
cost-awareness effect cannot appear." The power check showed that premise is only
half right: the agent spends **3.34 steps** on these single-hop questions and
leaves **62% unanswered**. There is therefore plenty of doomed work to abandon, and
a NEGATIVE Δsteps here would be perfectly consistent with cost-awareness rather
than a falsification of it.

So Δsteps is NOT the discriminating quantity. **ΔF1 is.** SimpleQA answers are
single short facts: there is no multi-hop reasoning to do better. If the treatment
answers them better, the improvement cannot be about spending steps wisely — it is
generic.

  ΔF1 ≈ 0            -> the HotpotQA quality gain is specific to multi-hop work
  ΔF1 > 0, CI off 0  -> λ is a general regulariser; the quality gain is a
                        CONFOUND and must be reported as one, not as a benefit

The sharpest form is ΔF1 restricted to episodes whose step count did NOT change —
the same slice where T1 found the HotpotQA gain lived (+0.092, CI excluding zero).

Usage: .venv/bin/python scripts/t2_analyse.py --dir experiments/results/t2_simpleqa
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common import load_config
from eval.metrics import bootstrap_ci

RESAMPLES = 10000
# T1's HotpotQA result, for side-by-side comparison. Not a threshold.
HOTPOT_DF1_UNCHANGED = 0.092


def load(d: Path, arm: str) -> dict:
    p = d / f"{arm}.jsonl"
    if not p.exists():
        return {}
    return {e["task_id"]: e for e in (json.loads(l) for l in open(p) if l.strip())}


def stat(v, seed, label, width=34):
    v = np.asarray(v, float)
    lo, hi = bootstrap_ci(v, RESAMPLES, seed)
    sig = lo > 0 or hi < 0
    print(f"   {label:<{width}} {v.mean():+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]"
          + ("  *" if sig else ""))
    return float(v.mean()), lo, hi, sig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    seed = load_config()["seed"]

    c, t = load(d, "control"), load(d, "treatment")
    tasks = sorted(set(c) & set(t))
    if not tasks:
        print("missing an arm"); sys.exit(2)

    cs = np.array([c[k]["steps_used"] for k in tasks], float)
    ts = np.array([t[k]["steps_used"] for k in tasks], float)
    cf = np.array([c[k]["final_f1"] for k in tasks], float)
    tf = np.array([t[k]["final_f1"] for k in tasks], float)
    ds, df = ts - cs, tf - cf

    print(f"T2 — SimpleQA negative control (single-hop), B=2, n={len(tasks)}\n")
    print(f"   control   steps {cs.mean():.3f}  F1 {cf.mean():.3f}  "
          f"nonzero {100*(cf>0).mean():.0f}%")
    print(f"   treatment steps {ts.mean():.3f}  F1 {tf.mean():.3f}  "
          f"nonzero {100*(tf>0).mean():.0f}%\n")

    print("1. overall (paired)")
    stat(ds, seed, "Δsteps")
    f1_all = stat(df, seed, "ΔF1  <- the discriminating quantity")

    print("\n2. ΔF1 by what the treatment did with its steps"
          "\n   (the same decomposition T1 ran on HotpotQA)")
    out = {}
    for name, m in (("spent FEWER steps", ds < 0),
                    ("spent the SAME", ds == 0),
                    ("spent MORE", ds > 0)):
        if m.sum() < 5:
            print(f"   {name:<34} n={int(m.sum())} — too few to report")
            continue
        out[name] = stat(df[m], seed, f"{name} (n={int(m.sum())})")

    print("\n3. THE COMPARISON THAT DECIDES")
    unchanged = ds >= 0
    if unchanged.sum() >= 5:
        m, lo, hi, sig = stat(df[unchanged], seed,
                              f"ΔF1 | steps not reduced (n={int(unchanged.sum())})")
        print(f"\n   HotpotQA equivalent (T1): +{HOTPOT_DF1_UNCHANGED:.3f}, CI excluded zero")
        print()
        if sig and m > 0:
            print("   VERDICT: REGULARISER. The treatment answers single-hop fact")
            print("   lookups better, where there is no multi-hop reasoning to improve.")
            print("   The quality gain is GENERIC and must be reported as a confound,")
            print("   not as a benefit of cost-aware training.")
            print("   The STOPPING result is unaffected -- T1 showed Δsteps and ΔF1 are")
            print("   independent -- but the paper may claim only the stopping effect.")
        elif not sig:
            print("   VERDICT: NOT GENERIC. The quality gain does not reproduce on")
            print("   single-hop questions, so it appears specific to multi-hop work.")
            print("   That is compatible with cost-awareness, though this test does")
            print("   not establish the mechanism -- it only fails to falsify it.")
        else:
            print("   VERDICT: treatment is WORSE here; report as-is.")

    print("\n4. note on Δsteps, which this test does NOT hinge on")
    print("   The agent spends >3 steps on single-hop questions and leaves most")
    print("   unanswered, so doomed work exists here too and a negative Δsteps is")
    print("   consistent with cost-awareness rather than evidence against it.")


if __name__ == "__main__":
    main()
