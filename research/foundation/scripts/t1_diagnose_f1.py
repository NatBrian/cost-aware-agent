"""T1 — why did F1 go UP? The confound that decides what the paper can claim.

Step 1 passed its gate, but F1 rose by +0.080 (CI excluding zero) — an effect that
was NOT predicted. Two explanations fit the numbers equally well:

  (A) COST-AWARENESS. The policy abandons doomed work, and the reward's economic
      structure incidentally improves how it uses the steps it does take.
  (B) REGULARISER. λ is simply a better-conditioned training objective, and the
      policy got better at everything. Cost-awareness is then the wrong story
      even though the gate passed.

The λ=0 control degrading to 20.5% malformed while the priced arm recovered to
3.1% is consistent with (B) and must be taken seriously.

This script decomposes the F1 gain on the 9 eval files already on disk. It cannot
settle (A) vs (B) on its own — SimpleQA (T2) is the decisive test — but it says
WHERE the gain lives, which shapes what to look for.

Checks:
  1. Is the F1 gain on episodes the treatment ABANDONED (stopped earlier) or on
     ones where it spent the same/more? If the gain is where it spent the SAME,
     the saving and the quality gain are independent effects -> leans (B).
  2. Does per-episode Δsteps predict per-episode ΔF1? A null correlation means
     they are separate phenomena.
  3. Is the treatment better FORMED (fewer malformed steps, fewer cap-outs at
     temp 0) or better RESEARCHED (more distinct retrieved titles, higher
     retrieval productivity)?
  4. On which episodes does the treatment win/lose outright?

Usage: .venv/bin/python scripts/t1_diagnose_f1.py --dir experiments/results/s5_eval
"""

import argparse
import json
import re
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


def ci(v, seed):
    lo, hi = bootstrap_ci(np.asarray(v, float), RESAMPLES, seed)
    return f"{np.mean(v):+.3f} [{lo:+.3f},{hi:+.3f}]", (lo > 0 or hi < 0)


def titles(ep):
    t = set()
    for s in ep["steps"]:
        t.update(re.findall(r"\[\d+\]\s*([^:]{1,80}):", s.get("obs_digest") or ""))
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    cfg = load_config()
    seed = cfg["seed"]
    gate = cfg["episode"].get("gate_budget", "small")
    B = cfg["episode"]["budgets"][gate]

    c, t = load(d, "control", gate), load(d, "treatment", gate)
    tasks = sorted(set(c) & set(t))
    print(f"T1 — decomposing the F1 gain at the gate budget B={B}, n={len(tasks)}\n")

    ds = np.array([t[k]["steps_used"] - c[k]["steps_used"] for k in tasks], float)
    df = np.array([t[k]["final_f1"] - c[k]["final_f1"] for k in tasks], float)

    # ---- 1. where does the F1 gain live? ---------------------------------
    print("1. F1 gain by what the treatment did with its STEPS")
    print(f"   {'bucket':<26}{'n':>5}{'ΔF1':>22}{'Δsteps':>10}")
    for name, m in (("treatment spent FEWER", ds < 0),
                    ("treatment spent SAME", ds == 0),
                    ("treatment spent MORE", ds > 0)):
        if m.sum() == 0:
            continue
        s, sig = ci(df[m], seed)
        print(f"   {name:<26}{int(m.sum()):>5}{s:>22}{ds[m].mean():>+10.3f}"
              + ("  *" if sig else ""))
    print("   (* = 95% CI excludes zero)")
    print("   If the gain is significant in the SAME bucket, the quality effect is")
    print("   INDEPENDENT of the saving -- which leans regulariser, not cost-awareness.\n")

    # ---- 2. do the two effects correlate per episode? --------------------
    if ds.std() > 0 and df.std() > 0:
        r = float(np.corrcoef(ds, df)[0, 1])
        rs = []
        rng = np.random.default_rng(seed)
        for _ in range(2000):
            i = rng.integers(0, len(ds), len(ds))
            if ds[i].std() > 0 and df[i].std() > 0:
                rs.append(np.corrcoef(ds[i], df[i])[0, 1])
        lo, hi = np.quantile(rs, [0.025, 0.975])
        print(f"2. per-episode corr(Δsteps, ΔF1) = {r:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]")
        print("   Near zero => saving and quality are separate phenomena.\n")

    # ---- 3. better formed, or better researched? -------------------------
    print("3. mechanism: formatting vs research quality")
    for label, fn in (
        ("malformed steps / episode",
         lambda e: sum(1 for s in e["steps"] if s["action_type"] == "malformed")),
        ("hit t_max (no ANSWER)", lambda e: float(e.get("answered_at") is None)),
        ("distinct titles retrieved", lambda e: len(titles(e))),
        ("distinct titles per step", lambda e: len(titles(e)) / max(len(e["steps"]), 1)),
        ("chars emitted / step",
         lambda e: np.mean([s.get("raw_len", 0) for s in e["steps"]]) if e["steps"] else 0),
        ("empty/failed retrievals",
         lambda e: sum(1 for s in e["steps"]
                       if "No results found" in (s.get("obs_digest") or ""))),
    ):
        cv = np.array([fn(c[k]) for k in tasks], float)
        tv = np.array([fn(t[k]) for k in tasks], float)
        s, sig = ci(tv - cv, seed)
        print(f"   {label:<28} ctrl {cv.mean():>7.3f}  trt {tv.mean():>7.3f}   "
              f"Δ {s}" + ("  *" if sig else ""))

    # ---- 4. win/loss decomposition ---------------------------------------
    print("\n4. outright wins and losses")
    win = (df > 1e-9).sum(); loss = (df < -1e-9).sum(); tie = len(df) - win - loss
    print(f"   treatment better on {win}, worse on {loss}, tied on {tie}")
    both = ((df > 1e-9) & (ds < 0)).sum()
    print(f"   better AND cheaper on {both} episodes "
          f"({100*both/len(df):.1f}%) -- the Pareto cell")

    # ---- 5. does the gain survive on non-abandoned work? -----------------
    kept = ds >= 0     # treatment did NOT spend fewer steps
    s, sig = ci(df[kept], seed)
    print(f"\n5. F1 gain restricted to episodes the treatment did NOT shorten: "
          f"{s}" + ("  * SIGNIFICANT" if sig else "  (n.s.)"))
    print("   A significant gain here is the clearest single sign of a general")
    print("   quality effect rather than a cost-aware one. T2 (SimpleQA) decides.")


if __name__ == "__main__":
    main()
