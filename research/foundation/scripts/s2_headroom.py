"""S2 — headroom audit at scale, and the economy calibration that follows from it.

Answers three questions on rollouts already on disk, BEFORE any threshold is
written down (plan v2.2 §7.7 rule 1 — the ordering IS the pre-registration):

  1. How much wasted spend W is there, and how much of it is removable?
     Two rules are measured, and the difference between them matters:
       ORACLE  — quit if the draft is still worthless at step k. Uses GOLD, so it
                 is an UPPER BOUND, not an achievable target.
       LEARNED — quit if the S1 gold-free classifier says P(success) < theta.
                 This is what a policy could actually implement, and it is the
                 number the Step-1 threshold must be derived from.
  2. Do the new budgets {2,3,4} actually bind against the current policy?
  3. What lambda makes the best rule worth >= 0.05 utility (cap 0.6)?

FOUNDATION-1's threshold was 0.5 steps against a ceiling of 0.31. Never again:
the ceiling is measured first and the threshold is derived from it.

Usage:
  .venv/bin/python scripts/s2_headroom.py \
      --rollouts experiments/results/train/lam0_round1/rollouts.jsonl \
      --out experiments/results/s2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from common import load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_predictability import build, features  # noqa: E402  (shared feature spec)

TARGET_DU = 0.05          # plan v2.2 §7.4: the oracle rule must be worth this
SEED = 42


def episode_curve(ep: dict) -> list[float]:
    return [s.get("draft_f1_vs_gold", 0.0) for s in ep["steps"]]


def oracle_rule(eps: list[dict], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Quit at step k if the draft is still worthless. Returns (steps, f1)."""
    S, F = [], []
    for ep in eps:
        c = episode_curve(ep)
        a = ep["steps_used"]
        if a > k and len(c) >= k and all(x <= 1e-9 for x in c[:k]):
            S.append(k)
            F.append(c[k - 1])          # deliver what we have (zero, by the test)
        else:
            S.append(a)
            F.append(ep["final_f1"])
    return np.array(S, float), np.array(F, float)


def learned_rule(eps: list[dict], k: int, proba: np.ndarray, idx_of: dict,
                 theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Quit at step k if the GOLD-FREE classifier says P(success) < theta."""
    S, F = [], []
    for i, ep in enumerate(eps):
        c = episode_curve(ep)
        a = ep["steps_used"]
        j = idx_of.get(i)
        if a > k and j is not None and proba[j] < theta:
            S.append(k)
            F.append(c[k - 1] if len(c) >= k else 0.0)
        else:
            S.append(a)
            F.append(ep["final_f1"])
    return np.array(S, float), np.array(F, float)


def W_of(steps: np.ndarray, f1: np.ndarray) -> float:
    return float(np.where(f1 <= 0, steps, 0.0).mean())


def oof_proba(eps: list[dict], k: int):
    """Out-of-fold P(success) at step k, grouped by task_id so no question is
    ever scored by a model that saw its siblings (the S1 leakage rule)."""
    X, y, g, _ = build(eps, k)
    if X is None or len(set(y)) < 2:
        return None, None
    keep = [i for i, ep in enumerate(eps) if len(ep["steps"]) >= k]
    p = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, g):
        m = GradientBoostingClassifier(random_state=SEED, n_estimators=200,
                                       max_depth=3, learning_rate=0.05)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p, {ep_i: j for j, ep_i in enumerate(keep)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    budgets = cfg["episode"]["budgets"]
    cap = cfg["economy"].get("lambda_cap", 0.6)

    allrows = [json.loads(l) for l in open(args.rollouts) if l.strip()]
    report = {"rollouts": args.rollouts, "n": len(allrows), "by_budget": {}}

    # ---- 2. do the new budgets bind? -------------------------------------
    stops = np.array([e["steps_used"] for e in allrows], float)
    print(f"episodes {len(allrows)}   mean stop {stops.mean():.2f}\n")
    print("BUDGET BINDING CHECK — % of episodes that would overspend each budget")
    binding = {}
    for name, B in budgets.items():
        frac = float((stops > B).mean())
        binding[name] = {"B": B, "overspend_frac": frac,
                         "binds": bool(frac >= 0.35)}
        print(f"  {name:<7} B={B}  {100*frac:>5.1f}%  "
              f"{'BINDS' if frac >= 0.35 else 'slack'}")
    report["binding"] = binding

    # ---- 1. headroom, per budget the data actually contains --------------
    for B in sorted({e["budget_B"] for e in allrows}):
        eps = [e for e in allrows if e["budget_B"] == B]
        base_S = np.array([e["steps_used"] for e in eps], float)
        base_F = np.array([e["final_f1"] for e in eps], float)
        W0 = W_of(base_S, base_F)
        print(f"\n{'='*70}\nB={B}   n={len(eps)}   baseline W={W0:.3f}  "
              f"steps={base_S.mean():.2f}  F1={base_F.mean():.3f}  "
              f"fail={100*(base_F<=0).mean():.1f}%")
        print(f"{'rule':<26}{'k':>3}{'W':>8}{'ΔW':>8}{'Δsteps':>9}{'ΔF1':>8}")

        best = {"dW": 0.0}
        for k in range(1, min(6, B + 2)):
            S, F = oracle_rule(eps, k)
            print(f"{'oracle (GOLD, upper bnd)':<26}{k:>3}{W_of(S,F):>8.3f}"
                  f"{W_of(S,F)-W0:>+8.3f}{S.mean()-base_S.mean():>+9.3f}"
                  f"{F.mean()-base_F.mean():>+8.3f}")

        for k in range(2, min(5, B + 2)):
            p, idx = oof_proba(eps, k)
            if p is None:
                continue
            for theta in (0.2, 0.3, 0.4, 0.5):
                S, F = learned_rule(eps, k, p, idx, theta)
                dW, dS, dF = W_of(S, F) - W0, S.mean() - base_S.mean(), \
                    F.mean() - base_F.mean()
                print(f"{'learned θ=' + str(theta):<26}{k:>3}{W_of(S,F):>8.3f}"
                      f"{dW:>+8.3f}{dS:>+9.3f}{dF:>+8.3f}")
                # best = biggest W reduction that keeps F1 within the guard
                if dW < best["dW"] and dF >= -0.02:
                    best = {"dW": dW, "dS": dS, "dF": dF, "k": k, "theta": theta}
        report["by_budget"][B] = {
            "n": len(eps), "W0": W0, "steps": float(base_S.mean()),
            "f1": float(base_F.mean()), "best_learned": best,
        }
        if best.get("k"):
            print(f"  -> best ACHIEVABLE (F1 guard -0.02): ΔW={best['dW']:+.3f} "
                  f"at k={best['k']}, θ={best['theta']}  "
                  f"(Δsteps {best['dS']:+.3f}, ΔF1 {best['dF']:+.3f})")
        else:
            print("  -> no learned rule improves W within the F1 guard")

    # ---- 2b. POWER: is the achievable effect even detectable? -------------
    # FOUNDATION-1 set a 0.5-step threshold against a 0.31-step ceiling. The
    # mirror-image mistake is a threshold BELOW the noise floor, which is just as
    # unpassable. Both are checked here, before any threshold is written.
    print(f"\n{'='*70}\nPOWER — paired within-task SD, and the n each estimand needs")
    print("rule: threshold = 50% of the achievable effect, and the 95% CI")
    print("half-width at the planned n must be SMALLER than that threshold.\n")
    print(f"{'B':>3}{'estimand':>10}{'effect':>9}{'pairedSD':>10}{'thresh':>9}{'n needed':>10}")
    power = {}
    for B, d in report["by_budget"].items():
        eps_b = [e for e in allrows if e["budget_B"] == B]
        by = {}
        for e in eps_b:
            by.setdefault(e["task_id"], {"S": [], "W": []})
            by[e["task_id"]]["S"].append(float(e["steps_used"]))
            by[e["task_id"]]["W"].append(float(e["steps_used"]) if e["final_f1"] <= 0 else 0.0)
        best = d.get("best_learned", {})
        eff = {"steps": abs(best.get("dS", 0.0)), "W": abs(best.get("dW", 0.0))}
        power[B] = {}
        for key, name in (("S", "steps"), ("W", "W")):
            v = [np.var(x[key]) for x in by.values() if len(x[key]) > 1]
            # paired differencing removes the between-task component; what is
            # left is twice the within-task variance
            sd = float(np.sqrt(2 * np.mean(v))) if v else float("nan")
            e = eff[name]
            thr = e / 2
            n_need = float((1.96 * sd / thr) ** 2) if thr > 1e-9 else float("inf")
            power[B][name] = {"effect": e, "paired_sd": sd, "threshold": thr,
                              "n_needed": n_need}
            print(f"{B:>3}{name:>10}{e:>9.3f}{sd:>10.3f}{thr:>9.3f}"
                  f"{n_need:>10.0f}" + ("  FEASIBLE" if n_need <= 1000 else "  too big"))
    report["power"] = power

    # ---- 3. lambda calibration -------------------------------------------
    gate_b = budgets[cfg["episode"].get("gate_budget", "small")]
    g = report["by_budget"].get(gate_b, {}).get("best_learned", {})
    print(f"\n{'='*70}\nLAMBDA CALIBRATION at the gate budget B={gate_b}")
    prim = power.get(gate_b, {})
    if prim:
        ok = [k for k, v in prim.items() if v["n_needed"] <= 1000]
        choice = min(ok, key=lambda k: prim[k]["n_needed"]) if ok else None
        print(f"  PRIMARY ESTIMAND: {choice or 'NONE FEASIBLE'}"
              + (f"  (n >= {prim[choice]['n_needed']:.0f}, "
                 f"threshold {prim[choice]['threshold']:.3f})" if choice else ""))
        report["primary_estimand"] = choice
        report["required_n"] = prim[choice]["n_needed"] if choice else None

    if g.get("k"):
        dS, dF = -g["dS"], -g["dF"]        # steps saved, F1 given up (positive)
        need = (TARGET_DU + dF) * gate_b / max(dS, 1e-9)
        lam = min(need, cap)
        print(f"  achievable: {dS:.3f} steps saved for {dF:.3f} F1 given up")
        print(f"  need λ ≥ ({TARGET_DU} + {dF:.3f})·{gate_b}/{dS:.3f} = {need:.3f}")
        print(f"  cap {cap}  ->  λ* = {lam:.3f}"
              + ("   ** CAPPED: the achievable rule cannot be made worth "
                 f"{TARGET_DU} without exceeding the health-safe λ **"
                 if need > cap else ""))
        report["lambda"] = {"needed": float(need), "cap": cap,
                            "chosen": float(lam), "capped": bool(need > cap),
                            "target_dU": TARGET_DU, "gate_budget": gate_b}
    else:
        print("  cannot calibrate: no achievable rule at the gate budget")
        report["lambda"] = None

    (outdir / "s2_headroom.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {outdir/'s2_headroom.json'}")


if __name__ == "__main__":
    main()
