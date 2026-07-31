"""S3 — apply the PRE-REGISTERED Step-1 rule. Committed before the data exists.

The rule (experiments/reports/s3_preregistration.md §4, quoted verbatim):

    H1 PASSES iff all three hold, at the gate budget B=2, paired per task:
      1. mean(Δsteps) <= -0.119            Δ = treatment - control
      2. the 95% bootstrap CI for Δsteps lies entirely below zero
      3. mean(ΔF1) >= -0.02                the quality guard

    H2 (mechanism, reported either way) is supported iff
      mean(Δsteps | control failed) < mean(Δsteps | control succeeded)

NOTHING HERE IS TUNABLE. The threshold comes from S2's measured achievable effect
(0.238 steps) halved; the guard, the resample count and the seed come from the
pre-registration and the config. Editing this file after S4 data exists is a
protocol deviation and must be logged as one.

Usage:
  .venv/bin/python scripts/s3_analyse.py --dir experiments/results/s5_eval
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common import load_config
from eval.metrics import bootstrap_ci

# --- pre-registered constants (S3 §4). Do not parameterise. ------------------
THRESHOLD_STEPS = -0.119     # 50% of S2's measured achievable 0.238
F1_GUARD = -0.02
RESAMPLES = 10000
ARMS = ("control", "treatment")


def load(d: Path, arm: str, bname: str) -> list[dict]:
    p = d / f"{arm}_{bname}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def pair(a: list[dict], b: list[dict], key):
    """Align two arms on shared task_ids. Refuses duplicates: a repeated task_id
    silently fans out to a many-to-many join and reports a mean over the wrong
    population (the FOUNDATION-1 audit bug)."""
    ia = {e["task_id"]: e for e in a}
    ib = {e["task_id"]: e for e in b}
    if len(ia) != len(a) or len(ib) != len(b):
        raise ValueError("duplicate task_id in an arm — filter by mode first")
    common = sorted(set(ia) & set(ib))
    if not common:
        raise ValueError("no shared tasks between arms")
    return (common,
            np.array([key(ib[t]) for t in common], float),   # treatment
            np.array([key(ia[t]) for t in common], float))   # control


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    cfg = load_config()
    budgets = cfg["episode"]["budgets"]
    gate_name = cfg["episode"].get("gate_budget", "small")
    gate_B = budgets[gate_name]
    seed = cfg["seed"]
    lam_report = cfg["economy"]["lambda"]

    steps = lambda e: e["steps_used"]
    f1 = lambda e: e["final_f1"]
    W = lambda e: e["steps_used"] if e["final_f1"] <= 0 else 0.0

    print("FOUNDATION-2 Step 1 — pre-registered analysis")
    print(f"gate budget B={gate_B} ({gate_name});  threshold {THRESHOLD_STEPS} "
          f"steps;  F1 guard {F1_GUARD}\n")

    # ---- descriptive table ------------------------------------------------
    print(f"{'arm':<10}{'B':>3}{'n':>6}{'steps':>8}{'F1':>8}{'W':>8}"
          f"{'fail%':>8}{'self-stop':>11}{'U':>8}")
    for arm in ARMS:
        for bname, B in budgets.items():
            eps = load(d, arm, bname)
            if not eps:
                continue
            s = np.array([steps(e) for e in eps], float)
            f = np.array([f1(e) for e in eps], float)
            w = np.array([W(e) for e in eps], float)
            st = np.array([float(e["mode"] == "none"
                                 and e.get("answered_at") is not None
                                 and not e["forced_stop"]
                                 and e["answered_at"] <= e["budget_B"]) for e in eps])
            u = f - lam_report * (s / max(B, 1))
            print(f"{arm:<10}{B:>3}{len(eps):>6}{s.mean():>8.3f}{f.mean():>8.3f}"
                  f"{w.mean():>8.3f}{100*(f<=0).mean():>8.1f}"
                  f"{100*st.mean():>11.1f}{u.mean():>8.3f}")

    # ---- H1: the gate -----------------------------------------------------
    ctrl = load(d, "control", gate_name)
    trt = load(d, "treatment", gate_name)
    if not ctrl or not trt:
        print(f"\ncannot evaluate: missing an arm at the gate budget "
              f"(control={len(ctrl)}, treatment={len(trt)})")
        sys.exit(2)

    tasks, s_t, s_c = pair(ctrl, trt, steps)
    _, f_t, f_c = pair(ctrl, trt, f1)
    d_steps, d_f1 = s_t - s_c, f_t - f_c
    slo, shi = bootstrap_ci(d_steps, RESAMPLES, seed)
    flo, fhi = bootstrap_ci(d_f1, RESAMPLES, seed)

    print("\n" + "=" * 66)
    print(f"H1 — PRE-REGISTERED GATE @ B={gate_B},  n={len(tasks)} paired tasks")
    print(f"  control   steps {s_c.mean():.3f}   F1 {f_c.mean():.3f}")
    print(f"  treatment steps {s_t.mean():.3f}   F1 {f_t.mean():.3f}")
    print(f"  Δsteps {d_steps.mean():+.3f}  95% CI [{slo:+.3f}, {shi:+.3f}]")
    print(f"  ΔF1    {d_f1.mean():+.3f}  95% CI [{flo:+.3f}, {fhi:+.3f}]")

    c1 = d_steps.mean() <= THRESHOLD_STEPS
    c2 = shi < 0
    c3 = d_f1.mean() >= F1_GUARD
    print(f"\n  cond1  Δsteps <= {THRESHOLD_STEPS}      : "
          f"{'PASS' if c1 else 'FAIL'}  ({d_steps.mean():+.3f})")
    print(f"  cond2  95% CI entirely below 0 : {'PASS' if c2 else 'FAIL'}  "
          f"(upper {shi:+.3f})")
    print(f"  cond3  ΔF1 >= {F1_GUARD} (guard)    : "
          f"{'PASS' if c3 else 'FAIL'}  ({d_f1.mean():+.3f})")
    h1 = c1 and c2 and c3
    print(f"\n  H1 VERDICT: {'PASS' if h1 else 'FAIL'}")

    # ---- H2: mechanism, reported either way -------------------------------
    cf = {e["task_id"]: (e["final_f1"] <= 0) for e in ctrl}   # control's outcome
    failed = np.array([cf[t] for t in tasks])
    print("\n" + "=" * 66)
    print("H2 — does the saving concentrate on work that was going to fail?")
    out = {}
    for name, mask in (("control FAILED", failed), ("control SUCCEEDED", ~failed)):
        if mask.sum() == 0:
            print(f"  {name:<20} (empty)")
            continue
        v = d_steps[mask]
        lo, hi = bootstrap_ci(v, RESAMPLES, seed)
        out[name] = float(v.mean())
        print(f"  {name:<20} n={int(mask.sum()):>4}  Δsteps {v.mean():+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    h2 = (out.get("control FAILED", 0.0) < out.get("control SUCCEEDED", 0.0))
    print(f"\n  H2 {'SUPPORTED' if h2 else 'NOT SUPPORTED'} — the saving is "
          f"{'larger' if h2 else 'NOT larger'} on doomed work")
    if h1 and not h2:
        print("  NOTE: H1 passed but H2 did not. The economic claim survives; the "
              "abandonment\n  mechanism does NOT, and the report must say so.")

    # ---- W: descriptive only, underpowered by design ----------------------
    _, w_t, w_c = pair(ctrl, trt, W)
    dW = w_t - w_c
    wlo, whi = bootstrap_ci(dW, RESAMPLES, seed)
    print("\n" + "=" * 66)
    print(f"W (economic reading; NOT gated on — S2: needs n~2289, we have {len(tasks)})")
    print(f"  ΔW {dW.mean():+.3f}  95% CI [{wlo:+.3f}, {whi:+.3f}]  "
          f"-- underpowered, do not claim significance")

    # ---- dose-response, supporting only -----------------------------------
    print("\n" + "=" * 66)
    print("Dose-response (pre-registered prediction: |Δsteps| largest at B=2)")
    print("  B=3 and B=4 are UNDERPOWERED by S2 (n~751 needed at B=4);")
    print("  a null there is uninformative and is not evidence of absence.")
    for bname, B in budgets.items():
        c, t = load(d, "control", bname), load(d, "treatment", bname)
        if not c or not t:
            continue
        _, x, y = pair(c, t, steps)
        dd = x - y
        lo, hi = bootstrap_ci(dd, RESAMPLES, seed)
        print(f"  B={B}  Δsteps {dd.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    res = {"gate_budget": gate_B, "n": len(tasks),
           "d_steps": float(d_steps.mean()), "d_steps_ci": [slo, shi],
           "d_f1": float(d_f1.mean()), "d_f1_ci": [flo, fhi],
           "cond1": bool(c1), "cond2": bool(c2), "cond3": bool(c3),
           "H1": bool(h1), "H2": bool(h2), "h2_detail": out,
           "dW": float(dW.mean()), "dW_ci": [wlo, whi]}
    (d / "s3_verdict.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {d/'s3_verdict.json'}")
    sys.exit(0 if h1 else 1)


if __name__ == "__main__":
    main()
