"""U1 — measure the saving in TOKENS, the unit the paper is actually about.

Everything so far reports Δsteps. But a "step" is not a billable unit: you pay for
tokens. And steps are not interchangeable — every step re-reads the whole
conversation, so on this harness step 10 costs ~9.7x step 1 (measured
2026-07-31, no-cache upper bound).

That matters for the direction of the result. The policy abandons doomed episodes,
and doomed episodes are the LONG ones — so the steps it cuts are the expensive
tail. If so, **the relative token saving should exceed the relative step saving**,
and the headline is currently understated in the unit that counts.

Reports, paired per task, for every dataset and seed:
  Δsteps and Δtokens, in absolute and relative terms
  the ratio of relative savings (>1 means cutting expensive steps)
  prompt vs completion split (is it context re-reading or generation?)

CACHING CAVEAT, stated because it changes the number a reader would compute:
these are tokens PROCESSED as reported by the server. With prefix caching a
provider may bill re-read context at a discount, which would shrink the absolute
dollar figure but not the paired difference between two arms running under the
identical serving regime.

Usage: .venv/bin/python scripts/u1_token_cost.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common import FOUNDATION_ROOT
from eval.metrics import bootstrap_ci

RESAMPLES = 10000
SEED = 42

# (label, control path, treatment path)
PAIRS = [
    ("HotpotQA (Step 1)", "s5_eval/control_small.jsonl", "s5_eval/treatment_small.jsonl"),
    ("SimpleQA (OOD)", "t2_simpleqa/control.jsonl", "t2_simpleqa/treatment.jsonl"),
    ("MuSiQue seed 42 (r1)", "t4_musique/control_small.jsonl", "t4_musique/mqtrtmatched_small.jsonl"),
    ("MuSiQue seed 123 (r3)", "t3_seeds/s123_ctrl.jsonl", "t3_seeds/s123_trt.jsonl"),
    ("MuSiQue seed 789 (r?)", "t3_seeds/s789_ctrl.jsonl", "t3_seeds/s789_trt.jsonl"),
]


def load(p: Path):
    if not p.exists():
        return None
    return {e["task_id"]: e for e in (json.loads(l) for l in open(p) if l.strip())}


def toks(ep, which=None):
    if which == "prompt":
        return sum(s.get("prompt_tokens", 0) for s in ep["steps"])
    if which == "completion":
        return sum(s.get("completion_tokens", 0) for s in ep["steps"])
    return sum(s.get("prompt_tokens", 0) + s.get("completion_tokens", 0)
               for s in ep["steps"])


def main() -> None:
    root = FOUNDATION_ROOT / "experiments/results"
    print("U1 — the saving measured in TOKENS, not step counts\n")
    print("A step is not a billable unit and steps are not equal: every step "
          "re-reads the\nwhole conversation, so late steps cost far more. "
          "Abandonment cuts the long\nepisodes, i.e. the expensive tail.\n")
    print(f"{'dataset':<24}{'Δsteps':>18}{'Δtokens':>20}{'rel steps':>11}"
          f"{'rel tokens':>12}{'ratio':>8}")

    rows = []
    for label, cp, tp in PAIRS:
        c, t = load(root / cp), load(root / tp)
        if not c or not t:
            print(f"{label:<24}   -- not available --")
            continue
        ks = sorted(set(c) & set(t))
        cs = np.array([c[k]["steps_used"] for k in ks], float)
        ts = np.array([t[k]["steps_used"] for k in ks], float)
        ct = np.array([toks(c[k]) for k in ks], float)
        tt = np.array([toks(t[k]) for k in ks], float)
        if ct.sum() == 0:
            print(f"{label:<24}   -- no token data --")
            continue
        ds, dt = ts - cs, tt - ct
        slo, shi = bootstrap_ci(ds, RESAMPLES, SEED)
        tlo, thi = bootstrap_ci(dt, RESAMPLES, SEED)
        rel_s = ds.mean() / cs.mean()
        rel_t = dt.mean() / ct.mean()
        ratio = rel_t / rel_s if abs(rel_s) > 1e-9 else float("nan")
        print(f"{label:<24}{f'{ds.mean():+.3f}':>9}{f'[{slo:+.2f},{shi:+.2f}]':>9}"
              f"{f'{dt.mean():+.0f}':>10}{f'[{tlo:+.0f},{thi:+.0f}]':>10}"
              f"{100*rel_s:>10.1f}%{100*rel_t:>11.1f}%{ratio:>8.2f}")
        rows.append((label, ds, dt, cs, ct, ks, c, t))

    print("\n  ratio > 1 means the tokens saved outrun the steps saved — i.e. the")
    print("  abandoned steps were the expensive ones. ratio ~1 means steps are a")
    print("  fair proxy after all.")

    # ---- prompt vs completion: context re-reading or generation? ----------
    print("\n" + "=" * 74)
    print("Where the tokens go: prompt (context re-read) vs completion (generated)")
    print(f"{'dataset':<24}{'Δprompt':>14}{'Δcompletion':>16}{'prompt share':>15}")
    for label, ds, dt, cs, ct, ks, c, t in rows:
        dp = np.array([toks(t[k], "prompt") - toks(c[k], "prompt") for k in ks], float)
        dc = np.array([toks(t[k], "completion") - toks(c[k], "completion") for k in ks], float)
        share = dp.mean() / (dp.mean() + dc.mean()) if abs(dp.mean() + dc.mean()) > 1e-9 else float("nan")
        print(f"{label:<24}{dp.mean():>14.0f}{dc.mean():>16.0f}{100*share:>14.1f}%")
    print("\n  A high prompt share means the saving is mostly avoided context")
    print("  re-reading, which is exactly what skipping a late step buys.")

    # ---- selectivity in token terms --------------------------------------
    print("\n" + "=" * 74)
    print("Selectivity in tokens (partition by the CONTROL's outcome)")
    for label, ds, dt, cs, ct, ks, c, t in rows:
        failed = np.array([c[k]["final_f1"] <= 0 for k in ks])
        out = []
        for nm, m in (("doomed", failed), ("successful", ~failed)):
            if m.sum() < 5:
                continue
            v = dt[m]
            lo, hi = bootstrap_ci(v, RESAMPLES, SEED)
            out.append(f"{nm} {v.mean():+.0f} [{lo:+.0f},{hi:+.0f}]"
                       + ("*" if hi < 0 or lo > 0 else ""))
        print(f"   {label:<24} " + "   ".join(out))

    print("\nCaveat: tokens PROCESSED as reported by the server. Prefix caching")
    print("would reduce the billed dollar figure but not the paired difference,")
    print("since both arms ran under the identical serving regime.")


if __name__ == "__main__":
    main()
