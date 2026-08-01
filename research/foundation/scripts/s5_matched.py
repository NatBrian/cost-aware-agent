"""S5 supplementary — the ROUND-MATCHED comparison, run only if a health gate
stopped one arm early.

This is deliberately NOT part of scripts/s3_analyse.py. That script was committed
before any S4 data existed and must stay byte-identical to what was
pre-registered; the rule it applies is the rule that decides. This file adds a
robustness check alongside it and can never change the verdict.

Why it exists: if the treatment breaches the temp-1.0 health gate at round 1 or 2
while the control reaches round 3, then "treatment vs control" compares a
1-round policy against a 3-round one, and any Δ mixes the λ effect with an
amount-of-training effect. FOUNDATION-1 hit exactly this with λ=1.0 and the only
honest response was to report both checkpoints. Here the control is additionally
evaluated AT THE TREATMENT'S ROUND, so the two comparisons bracket the answer:

    protocol-matched  (decides)   treatment@rT  vs  control@r3
    round-matched     (robustness) treatment@rT  vs  control@rT

If they disagree, the report must say so and neither is quietly preferred.

Usage: .venv/bin/python scripts/s5_matched.py --dir experiments/results/s5_eval
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


def load(d: Path, arm: str, bname: str) -> dict:
    p = d / f"{arm}_{bname}.jsonl"
    if not p.exists():
        return {}
    return {e["task_id"]: e for e in (json.loads(l) for l in open(p) if l.strip())}


def delta(a: dict, b: dict, key, seed: int):
    """b - a, paired on shared task ids."""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    v = np.array([key(b[k]) - key(a[k]) for k in common], float)
    lo, hi = bootstrap_ci(v, RESAMPLES, seed)
    return {"n": len(common), "mean": float(v.mean()), "lo": lo, "hi": hi}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    cfg = load_config()
    seed = cfg["seed"]
    budgets = cfg["episode"]["budgets"]
    gate_name = cfg["episode"].get("gate_budget", "small")

    prov_p = d / "arm_provenance.json"
    prov = json.loads(prov_p.read_text()) if prov_p.exists() else {}
    if not prov.get("matched"):
        print("No round mismatch — both arms reached the same round. "
              "The protocol-matched comparison in s3_analyse.py is the only one "
              "needed, and this check is a no-op.")
        return

    print("ROUND-MATCHED ROBUSTNESS CHECK")
    print(f"  control round {prov.get('control_round')}, "
          f"treatment round {prov.get('treatment_round')} "
          f"-- a health gate stopped an arm early.\n")

    steps = lambda e: e["steps_used"]
    f1 = lambda e: e["final_f1"]
    out = {"provenance": prov, "comparisons": {}}

    # The early-stopping arm can be either one. Whichever sat at the higher round
    # was re-evaluated at the lower round, so substitute that arm's matched file.
    m_arm = prov.get("matched_arm", "controlmatched")
    if m_arm == "controlmatched":
        pairs = (("protocol-matched (decides)", "control", "treatment"),
                 ("round-matched (robustness)", "controlmatched", "treatment"))
    else:
        pairs = (("protocol-matched (decides)", "control", "treatment"),
                 ("round-matched (robustness)", "control", "treatmentmatched"))

    for label, ctrl_arm, trt_arm in pairs:
        print(f"{label}   [{trt_arm} vs {ctrl_arm}]")
        out["comparisons"][label] = {}
        for bname, B in budgets.items():
            c, t = load(d, ctrl_arm, bname), load(d, trt_arm, bname)
            if not c or not t:
                continue
            ds = delta(c, t, steps, seed)
            df = delta(c, t, f1, seed)
            star = "*" if ds and (ds["hi"] < 0 or ds["lo"] > 0) else " "
            mark = " <- GATE" if bname == gate_name else ""
            print(f"   B={B}  n={ds['n']:>4}  Δsteps {ds['mean']:+.3f}{star} "
                  f"[{ds['lo']:+.3f},{ds['hi']:+.3f}]   "
                  f"ΔF1 {df['mean']:+.3f} [{df['lo']:+.3f},{df['hi']:+.3f}]{mark}")
            out["comparisons"][label][B] = {"steps": ds, "f1": df}
        print()

    # do the two comparisons agree at the gate budget?
    gb = budgets[gate_name]
    a = out["comparisons"]["protocol-matched (decides)"].get(gb, {}).get("steps")
    b = out["comparisons"]["round-matched (robustness)"].get(gb, {}).get("steps")
    if a and b:
        same_sign = (a["mean"] < 0) == (b["mean"] < 0)
        print(f"At the gate budget the two comparisons "
              f"{'AGREE' if same_sign else 'DISAGREE'} in sign "
              f"({a['mean']:+.3f} vs {b['mean']:+.3f}).")
        if not same_sign:
            print("DISAGREEMENT: the effect is confounded with amount of "
                  "training. Neither comparison may be reported alone.")
        out["agree_at_gate"] = bool(same_sign)

    (d / "s5_matched.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {d/'s5_matched.json'}")


if __name__ == "__main__":
    main()
