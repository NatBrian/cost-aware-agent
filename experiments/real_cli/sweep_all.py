#!/usr/bin/env python3
"""Full real-CLI sweep: calibration -> budget rule -> eval, both platforms.

BIAS CONTROL (pre-registered, decided before any eval run):
  budget = median OFF cost per question on the CALIBRATION set (data/calib.json,
  disjoint from data/eval.json). The earlier HotpotQA experiment picked its
  $0.30 tier after seeing OFF costs on the same questions it evaluated — an
  audit finding. Here the rule is fixed up front and computed from held-out
  questions only.

Arms are strictly serial: the daemon condition (inject_enabled, budget) is
global, so OFF and ON can never run concurrently.
"""
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args):
    print(f"\n>>> run_real.py {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_real.py")] + args,
                       cwd=HERE)
    if r.returncode != 0:
        print(f"!!! arm failed rc={r.returncode} — continuing", flush=True)


def median_off_cost(platform, tag="caliboff"):
    costs = []
    base = os.path.join(HERE, "runs", platform)
    for d in os.listdir(base):
        if not d.startswith(tag + "-s"):
            continue
        p = os.path.join(base, d, "results.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if not r.get("failed") and r.get("cost_usd"):
                costs.append(r["cost_usd"])
    if not costs:
        raise SystemExit(f"no calibration costs for {platform}")
    return statistics.median(costs), len(costs)


def main():
    seeds_calib, seeds_eval = 2, 3

    # 1) calibration (OFF only, held-out questions)
    run(["--platform", "claude", "--tag", "caliboff", "--budget", "0",
         "--seeds", str(seeds_calib), "--data", "data/calib.json"])
    run(["--platform", "opencode", "--tag", "caliboff", "--budget", "0",
         "--seeds", str(seeds_calib), "--data", "data/calib.json"])

    # 2) pre-registered budget rule: median OFF calibration cost per question
    cc_med, cc_n = median_off_cost("claude")
    oc_med, oc_n = median_off_cost("opencode")
    cc_budget = round(cc_med, 2)
    oc_budget = round(oc_med, 4)
    budgets = {"rule": "budget = median OFF cost/question on held-out calibration set",
               "claude": {"median": cc_med, "n": cc_n, "budget": cc_budget},
               "opencode": {"median": oc_med, "n": oc_n, "budget": oc_budget}}
    json.dump(budgets, open(os.path.join(HERE, "budgets.json"), "w"), indent=2)
    print(f"\n=== budgets (pre-registered rule): {budgets}", flush=True)

    # 3) eval sweep
    run(["--platform", "claude", "--tag", "off", "--budget", "0",
         "--seeds", str(seeds_eval), "--data", "data/eval.json"])
    run(["--platform", "claude", "--tag", "on", "--budget", str(cc_budget),
         "--seeds", str(seeds_eval), "--data", "data/eval.json"])
    run(["--platform", "opencode", "--tag", "off", "--budget", "0",
         "--seeds", str(seeds_eval), "--data", "data/eval.json"])
    run(["--platform", "opencode", "--tag", "on", "--budget", str(oc_budget),
         "--seeds", str(seeds_eval), "--data", "data/eval.json"])

    print("\n=== SWEEP COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
