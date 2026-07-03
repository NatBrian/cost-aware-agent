#!/usr/bin/env python3
"""Paired OFF-vs-ON analysis for the SWE-bench A/B (money, tools, success).

Pairs on (instance, seed). Paired t on cost delta (OFF - ON: positive = ON
cheaper), success-rate comparison (the budget must not buy savings with
failures), audit summary. Writes analysis.json.
"""
import glob
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

# two-sided t critical values, alpha=0.05, by df
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
         13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
         19: 2.093, 20: 2.086, 23: 2.069, 29: 2.045}


def tcrit(df):
    if df in TCRIT:
        return TCRIT[df]
    bigger = [k for k in TCRIT if k >= df]
    return TCRIT[min(bigger)] if bigger else 1.96


def load(platform, arm):
    rows = {}
    for path in glob.glob(os.path.join(HERE, "runs", platform, f"{arm}-s*", "results.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            # exclude platform failures AND timeouts: a killed run reports no
            # total_cost_usd (cost_usd is None), so it cannot enter the paired
            # cost math — it is a reliability event, counted separately
            if (r.get("failed") or r.get("timed_out") or r.get("cost_usd") is None
                    or r.get("contaminated")):
                continue
            rows[(r["iid"], r["seed"])] = r
    return rows


def paired_t(diffs):
    n = len(diffs)
    if n < 2:
        return {"n": n}
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0:
        return {"n": n, "mean": mean, "sd": 0.0, "t": None, "significant": False}
    t = mean / (sd / math.sqrt(n))
    tc = tcrit(n - 1)
    half = tc * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "t": round(t, 3), "tcrit": tc,
            "ci95": [mean - half, mean + half],
            "significant": abs(t) > tc}


def analyze(platform):
    off, on = load(platform, "off"), load(platform, "on")
    common = sorted(set(off) & set(on))
    if not common:
        return {"platform": platform, "error": "no common pairs"}
    d_cost = [off[k]["cost_usd"] - on[k]["cost_usd"] for k in common]
    d_tools = [off[k]["tool_calls"] - on[k]["tool_calls"] for k in common]
    mean_off = statistics.mean(off[k]["cost_usd"] for k in common)
    mean_on = statistics.mean(on[k]["cost_usd"] for k in common)
    per_inst = {}
    for (iid, seed) in common:
        per_inst.setdefault(iid, []).append(
            round(off[(iid, seed)]["cost_usd"] - on[(iid, seed)]["cost_usd"], 5))
    return {
        "platform": platform,
        "pairs": len(common),
        "off": {"cost": mean_off,
                "tools": statistics.mean(off[k]["tool_calls"] for k in common),
                "success": statistics.mean(1.0 * bool(off[k].get("success")) for k in common),
                "injections": statistics.mean(off[k].get("injections_delivered", 0) for k in common)},
        "on": {"cost": mean_on,
               "tools": statistics.mean(on[k]["tool_calls"] for k in common),
               "success": statistics.mean(1.0 * bool(on[k].get("success")) for k in common),
               "injections": statistics.mean(on[k].get("injections_delivered", 0) for k in common)},
        "saved_usd_per_task": mean_off - mean_on,
        "saved_pct": 100 * (mean_off - mean_on) / mean_off if mean_off else None,
        "money_stats": paired_t(d_cost),
        "tool_stats": paired_t(d_tools),
        "per_instance_saving": per_inst,
        "audit_clean_all": all(r.get("audit_clean") for r in list(off.values()) + list(on.values())),
        "timeouts": sum(1 for r in list(off.values()) + list(on.values()) if r.get("timed_out")),
    }


def main():
    out = [analyze(p) for p in ("claude", "opencode")]
    json.dump(out, open(os.path.join(HERE, "analysis.json"), "w"), indent=2)
    for a in out:
        if a.get("error"):
            print(a["platform"], a["error"])
            continue
        ms = a["money_stats"]
        print(f"\n=== {a['platform']} (n={a['pairs']} pairs)")
        print(f"  OFF: ${a['off']['cost']:.4f}/task  tools={a['off']['tools']:.1f}  "
              f"success={a['off']['success']:.2f}")
        print(f"  ON : ${a['on']['cost']:.4f}/task  tools={a['on']['tools']:.1f}  "
              f"success={a['on']['success']:.2f}  inj/run={a['on']['injections']:.1f}")
        print(f"  saved ${a['saved_usd_per_task']:.4f}/task ({a['saved_pct']:.1f}%)  "
              f"t={ms.get('t')} sig={ms.get('significant')} ci95={ms.get('ci95')}")
        print(f"  audit_clean_all={a['audit_clean_all']} timeouts={a['timeouts']}")


if __name__ == "__main__":
    main()
