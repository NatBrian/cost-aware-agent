"""S0 analysis — does pricing already work once the budget BINDS?

Reads the S0 re-scoring JSONLs and reports, per (arm, budget):

    W  = E[steps x 1(F1=0)]     the FOUNDATION-2 primary estimand
    F1                          the guard: a W win bought with quality is not a win
    steps, self-stop, abandonment rate

and the paired per-task deltas against the λ=0 control.

This is DIAGNOSTIC, not the gate. The Step-1 gate is pre-registered at S3 with a
threshold derived from S2-measured headroom, and is applied by its own committed
script. Nothing here is tunable.

Usage: .venv/bin/python scripts/s0_analyse.py --dir experiments/results/s0_rescore
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common import load_config
from eval.metrics import bootstrap_ci

ARMS = [("lam0", 0.0), ("lam03", 0.3), ("lam10", 1.0)]
RESAMPLES = 10000


def load(d: Path, tag: str, bname: str) -> list[dict]:
    p = d / f"{tag}_{bname}.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def metrics(eps: list[dict]) -> dict:
    steps = np.array([e["steps_used"] for e in eps], float)
    f1 = np.array([e["final_f1"] for e in eps], float)
    # W: unconditional. Never divide by the failure count — conditioning would let
    # a policy look thrifty by failing more often.
    W = np.where(f1 <= 0, steps, 0.0)
    stop = np.array([float(e["mode"] == "none" and e.get("answered_at") is not None
                           and not e["forced_stop"]
                           and e["answered_at"] <= e["budget_B"]) for e in eps])
    aband = np.where((stop > 0) & (f1 <= 0), 1.0, 0.0)
    return {"n": len(eps), "steps": steps, "f1": f1, "W": W, "stop": stop,
            "aband": aband, "task": [e["task_id"] for e in eps]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    cfg = load_config()
    budgets = cfg["episode"]["budgets"]
    seed = cfg["seed"]

    print("S0 — existing λ arms re-scored at BINDING budgets "
          f"{dict(budgets)}, val-50, temp 0, harness off\n")
    print("W = mean steps spent on episodes that returned NOTHING (lower better)")
    print("F1 is the guard: a W win bought with quality is not a win.\n")

    per = {}
    print(f"{'arm':<7}{'λ':>5}{'B':>4}{'n':>5}{'W':>8}{'F1':>8}{'steps':>8}"
          f"{'fail%':>8}{'aband%':>9}{'stop%':>8}")
    for tag, lam in ARMS:
        per[tag] = {}
        for bname, B in budgets.items():
            eps = load(d, tag, bname)
            if not eps:
                continue
            m = metrics(eps)
            per[tag][B] = m
            print(f"{tag:<7}{lam:>5}{B:>4}{m['n']:>5}{m['W'].mean():>8.3f}"
                  f"{m['f1'].mean():>8.3f}{m['steps'].mean():>8.3f}"
                  f"{100*(m['f1']<=0).mean():>8.1f}{100*m['aband'].mean():>9.1f}"
                  f"{100*m['stop'].mean():>8.1f}")
        if not per[tag]:
            print(f"{tag:<7}{lam:>5}   -- ABSENT --")

    print("\n" + "=" * 74)
    print("PAIRED per-task deltas vs the λ=0 control  (* = 95% CI excludes zero)")
    print(f"{'B':>3}{'arm':>8}{'ΔW':>9}{'95% CI':>20}{'ΔF1':>9}{'95% CI':>20}")
    ctrl = per.get("lam0", {})
    for bname, B in budgets.items():
        c = ctrl.get(B)
        if not c:
            continue
        ci = {t: k for k, t in enumerate(c["task"])}
        for tag, lam in ARMS[1:]:
            t = per.get(tag, {}).get(B)
            if not t:
                continue
            ti = {t_: k for k, t_ in enumerate(t["task"])}
            common = sorted(set(ci) & set(ti))
            if not common:
                continue
            dW = np.array([t["W"][ti[x]] - c["W"][ci[x]] for x in common])
            dF = np.array([t["f1"][ti[x]] - c["f1"][ci[x]] for x in common])
            wlo, whi = bootstrap_ci(dW, RESAMPLES, seed)
            flo, fhi = bootstrap_ci(dF, RESAMPLES, seed)
            star = "*" if (wlo > 0) or (whi < 0) else " "
            fstar = "*" if (flo > 0) or (fhi < 0) else " "
            print(f"{B:>3}{tag:>8}{dW.mean():>+8.3f}{star}"
                  f"{f'[{wlo:+.3f},{whi:+.3f}]':>20}"
                  f"{dF.mean():>+8.3f}{fstar}{f'[{flo:+.3f},{fhi:+.3f}]':>20}")

    print("\nReading this: a NEGATIVE ΔW with a CI excluding zero and a ΔF1 whose "
          "CI contains zero\nis pricing working — less spent on nothing, quality "
          "intact. That is the Step-1 hypothesis,\nvisible on checkpoints we "
          "already have.")


if __name__ == "__main__":
    main()
