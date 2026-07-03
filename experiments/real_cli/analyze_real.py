#!/usr/bin/env python3
"""Paired OFF-vs-ON analysis for the real-CLI experiment.

Money is the metric: per-question mean dollar cost (real Anthropic billing for
Claude Code; daemon retail-priced dollars for OpenCode's free deepseek), paired
per question across seeds, with paired t + 95% CI printed so non-significant
savings can't masquerade as results. Also reports tool calls, F1/EM, injection
delivery, tier escalation, and audit cleanliness.
"""
import json
import math
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def paired_stats(diffs):
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf")
    tcrit = _T95[min(n - 2, len(_T95) - 1)]
    return {"n": n, "mean": mean, "sd": sd, "t": t, "tcrit": tcrit,
            "ci": (mean - tcrit * se, mean + tcrit * se),
            "significant": abs(t) > tcrit}


def load(platform, tag):
    """rows per question id, averaged over seeds later."""
    rows = defaultdict(list)
    base = os.path.join(HERE, "runs", platform)
    if not os.path.isdir(base):
        return rows
    for d in sorted(os.listdir(base)):
        if not re.fullmatch(rf"{re.escape(tag)}-s\d+", d):
            continue
        p = os.path.join(base, d, "results.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if not r.get("failed"):
                rows[r["id"]].append(r)
    return rows


def qmean(rows, q, key):
    vals = [r.get(key) or 0 for r in rows[q]]
    return sum(vals) / len(vals)


def analyze(platform):
    off, on = load(platform, "off"), load(platform, "on")
    if not off or not on:
        print(f"\n[{platform}] missing off/on runs — skipping")
        return None
    common = sorted(set(off) & set(on))
    budgets = json.load(open(os.path.join(HERE, "budgets.json")))
    budget = budgets[platform if platform in budgets else "claude"]["budget"]

    def arm(rows):
        return {k: sum(qmean(rows, q, k) for q in common) / len(common)
                for k in ("cost_usd", "tool_calls", "f1", "em",
                          "injections_delivered")}

    a_off, a_on = arm(off), arm(on)
    n_off = sum(len(v) for v in off.values())
    n_on = sum(len(v) for v in on.values())
    clean = all(r.get("audit_clean", True)
                for rows in (off, on) for v in rows.values() for r in v)
    caps = {"off": sum(1 for v in off.values() for r in v if r.get("timed_out")),
            "on": sum(1 for v in on.values() for r in v if r.get("timed_out"))}
    tiers_on = sorted({t for v in on.values() for r in v
                       for t in (r.get("tiers_seen") or [])})

    print(f"\n{'='*70}\nPLATFORM: {platform} (ON budget = ${budget}, "
          f"pre-registered: median OFF calibration cost)")
    print(f"paired questions: {len(common)}, runs: OFF {n_off} / ON {n_on}, "
          f"audit_clean: {clean}, timeouts: {caps}")
    hdr = (f"{'arm':6} {'$/q':>10} {'tools':>6} {'F1':>6} {'EM':>6} {'inj/q':>6}")
    print(hdr); print("-" * len(hdr))
    for name, a in (("OFF", a_off), ("ON", a_on)):
        print(f"{name:6} {a['cost_usd']:10.5f} {a['tool_calls']:6.2f} "
              f"{a['f1']:6.3f} {a['em']:6.3f} {a['injections_delivered']:6.1f}")

    diffs = [qmean(off, q, "cost_usd") - qmean(on, q, "cost_usd") for q in common]
    ps = paired_stats(diffs)
    saved = a_off["cost_usd"] - a_on["cost_usd"]
    pct = 100 * saved / a_off["cost_usd"] if a_off["cost_usd"] else 0
    tdiffs = [qmean(off, q, "tool_calls") - qmean(on, q, "tool_calls") for q in common]
    pt = paired_stats(tdiffs)
    print(f"\nON vs OFF: ${saved:+.5f}/q ({pct:+.1f}%), "
          f"ΔF1 {a_on['f1']-a_off['f1']:+.3f}, ΔEM {a_on['em']-a_off['em']:+.3f}")
    if ps:
        print(f"  money  : paired t={ps['t']:.2f} (crit {ps['tcrit']:.2f}), "
              f"95% CI of saving [{ps['ci'][0]:+.5f}, {ps['ci'][1]:+.5f}] "
              f"-> {'SIGNIFICANT' if ps['significant'] else 'not significant'}")
    if pt:
        print(f"  tools  : Δ={sum(tdiffs)/len(tdiffs):+.2f}/q, t={pt['t']:.2f} "
              f"-> {'SIGNIFICANT' if pt['significant'] else 'not significant'}")
    print(f"  tiers seen in ON injections: {tiers_on}")
    per_q = [(q, round(d, 5)) for q, d in zip(common, diffs)]
    print(f"  per-question $ diff (OFF−ON): {per_q}")
    return {"platform": platform, "budget": budget, "off": a_off, "on": a_on,
            "saved": saved, "pct": pct, "money_stats": ps, "tool_stats": pt,
            "clean": clean, "common": common, "per_q_diff": per_q}


def main():
    out = [r for p in ("claude", "opencode") if (r := analyze(p))]
    json.dump(out, open(os.path.join(HERE, "analysis.json"), "w"), indent=2,
              default=str)
    print(f"\nsaved -> analysis.json")


if __name__ == "__main__":
    main()
