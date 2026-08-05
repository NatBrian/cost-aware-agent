"""U3 — concrete trajectory pairs. A behavioural claim should show the behaviour.

Everything so far is aggregate. A reader is entitled to see what the policy
actually does differently on a specific question, and — just as important — what
it gets wrong.

Extracts four categories from matched control/treatment pairs on the same task:

  WIN   treatment quit early, control ground on, neither answered.
        The behaviour we claim: cost saved, nothing lost.
  COST  treatment quit early, but the CONTROL eventually answered.
        Abandonment's real price: a winnable question given up. These exist and
        must be shown, not just counted.
  REALLOC treatment spent MORE and got a better answer where control failed.
        The reallocation behaviour seen on MuSiQue.
  NEUTRAL identical step counts, to show the policy is not uniformly hastier.

Prints the question, both trajectories' queries, stop points and outcomes.

Usage: .venv/bin/python scripts/u3_examples.py --n 2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import FOUNDATION_ROOT

PAIRS = [
    ("MuSiQue s123", "t3_seeds/s123_ctrl.jsonl", "t3_seeds/s123_trt.jsonl"),
    ("HotpotQA", "s5_eval/control_small.jsonl", "s5_eval/treatment_small.jsonl"),
]


def load(p: Path):
    if not p.exists():
        return None
    return {e["task_id"]: e for e in (json.loads(l) for l in open(p) if l.strip())}


def render(ep, tag, max_q=3):
    qs = [s["query_or_answer"][:70] for s in ep["steps"] if s["action_type"] == "search"]
    shown = qs[:max_q] + ([f"... (+{len(qs)-max_q} more)"] if len(qs) > max_q else [])
    print(f"      {tag}: {ep['steps_used']} steps, F1={ep['final_f1']:.2f}")
    for i, q in enumerate(shown, 1):
        print(f"        {i}. {q}")
    ans = (ep.get("final_answer") or "")[:70]
    print(f"        -> \"{ans}\"" if ans else "        -> (no answer)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()
    root = FOUNDATION_ROOT / "experiments/results"

    for label, cp, tp in PAIRS:
        c, t = load(root / cp), load(root / tp)
        if not c or not t:
            continue
        ks = sorted(set(c) & set(t))
        cats = {"WIN": [], "COST": [], "REALLOC": [], "NEUTRAL": []}
        for k in ks:
            ds = t[k]["steps_used"] - c[k]["steps_used"]
            cf, tf = c[k]["final_f1"], t[k]["final_f1"]
            if ds <= -2 and cf <= 0 and tf <= 0:
                cats["WIN"].append((k, ds))
            elif ds <= -1 and cf > 0.5 and tf <= 0:
                cats["COST"].append((k, ds))
            elif ds >= 2 and cf <= 0 and tf > 0.5:
                cats["REALLOC"].append((k, ds))
            elif ds == 0 and abs(tf - cf) < 1e-9:
                cats["NEUTRAL"].append((k, ds))

        print("=" * 78)
        print(f"{label}   counts: " + "  ".join(f"{n}={len(v)}" for n, v in cats.items()))
        print("=" * 78)
        for name, desc in (
            ("WIN", "treatment quit, control ground on, NEITHER answered — cost saved, nothing lost"),
            ("COST", "treatment quit but control ANSWERED — the price of abandonment"),
            ("REALLOC", "treatment spent MORE and answered where control failed"),
        ):
            items = cats[name][: args.n]
            if not items:
                print(f"\n[{name}] none found")
                continue
            print(f"\n[{name}] {desc}")
            for k, ds in items:
                q = c[k]["question"][:95]
                print(f"\n   Q: {q}")
                print(f"      (Δsteps {ds:+d})")
                render(c[k], "control  ")
                render(t[k], "treatment")

        n_cost = len(cats["COST"])
        print(f"\n   >> On {label}, abandonment cost a winnable answer in "
              f"{n_cost}/{len(ks)} cases ({100*n_cost/len(ks):.1f}%). "
              f"That is the honest downside.")


if __name__ == "__main__":
    main()
