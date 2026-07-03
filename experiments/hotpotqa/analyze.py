#!/usr/bin/env python3
"""Accuracy-vs-cost analysis for the money-budget HotpotQA experiment.

Reads the full run tree in runs/ and aggregates ACROSS SEEDS per money tier,
bucketed by GATEWAY (claude-cli vs opencode-cli) so the two agents — which run
different models at different cost scales — are never mixed in one table.

The claim under test: injecting a dollar budget makes the session spend less real
money ($) while roughly holding accuracy. OFF (no injection) is the control. Cost
is MONEY — real (Claude) or retail-priced (OpenCode's free deepseek) LLM dollars —
never a count of tool calls or iterations.

Per tier we report, over the question set present & non-failed in EVERY tier of
that gateway (a fair paired comparison), the per-question means averaged over all
seeds: $ spent, tool calls, output tokens, F1, EM. Then the headline: each budget
tier's $ saving vs OFF and its accuracy delta.

The budget shown for a tier is read from the run's recorded daemon condition
(session_budget_estimate_usd), NOT parsed from the tag — so any budget scale is
labelled correctly. Seed labels (`off-s0`, ...) collapse to their base tier.
"""
import glob
import json
import math
import os
import re
from collections import defaultdict

# two-sided 95% critical values of Student's t, df 1..30 — dependency-free
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def paired_stats(diffs):
    """Paired mean diff with t statistic and 95% CI. Returns None when n < 2.
    This is the self-policing layer: a '% cheaper' headline without a
    significant paired diff is noise, and the table now says so itself."""
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf")
    tcrit = _T95[min(n - 2, len(_T95) - 1)]
    ci = (mean - tcrit * se, mean + tcrit * se)
    return {"n": n, "mean": mean, "sd": sd, "t": t, "tcrit": tcrit,
            "ci": ci, "significant": abs(t) > tcrit}

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

# Real cheat-capable tools (can read data/questions.json off disk). Phantom
# tool_use blocks named after our ReAct verbs (SEARCH/READ/ANSWER) are NOT cheats
# — the CLI rejects them and the model re-emits the line as text. Recompute audit
# here so rows written before run_claude's audit was tightened are judged correctly.
_DANGER = {"Bash", "Read", "Write", "Edit", "Glob", "Grep",
           "WebSearch", "WebFetch", "Task", "NotebookEdit"}


def row_clean(r):
    uses = r.get("cli_tool_uses") or []
    return not any(n in _DANGER for n in uses) and r.get("web_requests", 0) == 0


def base_tier(tag):
    return re.sub(r"-s\d+$", "", tag)


def load_runs():
    """Return list of {label, tier, gateway, budget, inject, run_id, rows},
    keeping the latest run per label."""
    latest = {}
    for meta_path in glob.glob(os.path.join(RUNS, "*", "meta.json")):
        run_dir = os.path.dirname(meta_path)
        meta = json.load(open(meta_path))
        label = meta.get("tag")
        run_id = meta.get("run_id", os.path.basename(run_dir))
        res_path = os.path.join(run_dir, "results.jsonl")
        if not label or not os.path.exists(res_path):
            continue
        cond = meta.get("condition", {}) or {}
        rec = {
            "label": label, "tier": base_tier(label),
            "gateway": meta.get("gateway", "claude-cli"),
            "budget": cond.get("session_budget_estimate_usd"),
            "inject": cond.get("inject_enabled", False),
            "run_id": run_id,
            "rows": [json.loads(l) for l in open(res_path) if l.strip()],
        }
        if label not in latest or run_id > latest[label]["run_id"]:
            latest[label] = rec
    return list(latest.values())


def analyze_gateway(gw, recs):
    tier_q = defaultdict(lambda: defaultdict(list))
    tier_seeds = defaultdict(set)
    tier_budget = {}       # tier -> (inject, budget)
    tier_raw = defaultdict(lambda: [0, 0, 0])  # tier -> [total rows, failed, capped]
    for rec in recs:
        t = rec["tier"]
        tier_seeds[t].add(rec["label"])
        tier_budget[t] = (rec["inject"], rec["budget"])
        for r in rec["rows"]:
            tier_raw[t][0] += 1
            if r.get("failed"):
                tier_raw[t][1] += 1
            else:
                tier_q[t][r["id"]].append(r)
            if r.get("hit_cap"):
                tier_raw[t][2] += 1

    def order(t):
        inject, budget = tier_budget[t]
        return (1 if inject else 0, budget or 0.0)
    tiers = sorted(tier_q, key=order)
    common = sorted(set.intersection(*[set(tier_q[t]) for t in tiers])) if tiers else []

    print(f"\n{'='*64}\nGATEWAY: {gw}")
    print(f"tiers: {tiers}")
    print(f"seeds/tier: {{{', '.join(f'{t}:{len(tier_seeds[t])}' for t in tiers)}}}")
    print(f"paired questions (present & non-failed in all tiers): {len(common)} {common}")
    # Per-tier honesty counts: if a tier crashed (failed) more questions its pool
    # shrinks and the paired intersection silently drops those ids for EVERY tier.
    # Print raw totals so a reader can see any asymmetric attrition, not just the
    # surviving paired set.
    print("per-tier rows [total / failed / capped]: "
          + ", ".join(f"{t} {tier_raw[t][0]}/{tier_raw[t][1]}/{tier_raw[t][2]}" for t in tiers))
    if not common:
        print("no common questions"); return

    def tmean(t, k):
        return sum(sum(r.get(k, 0) for r in tier_q[t][q]) / len(tier_q[t][q])
                   for q in common) / len(common)

    def tclean(t):
        return all(row_clean(r) for q in common for r in tier_q[t][q])

    hdr = f"{'tier':8} {'budget':>8} {'cost_usd':>10} {'calls':>6} {'out_tok':>8} {'F1':>6} {'EM':>6} {'clean':>6}"
    print(hdr); print("-" * len(hdr))
    stats = {}
    off_tier = None
    for t in tiers:
        inject, budget = tier_budget[t]
        stats[t] = {"cost": tmean(t, "cost_usd"), "f1": tmean(t, "f1"),
                    "em": tmean(t, "em"), "calls": tmean(t, "tool_calls")}
        if not inject:
            off_tier = t
        b = "  —" if not inject else f"${budget:.3f}".rstrip("0").rstrip(".")
        print(f"{t:8} {b:>8} {stats[t]['cost']:10.5f} {stats[t]['calls']:6.2f} "
              f"{tmean(t,'out_tok'):8.0f} {stats[t]['f1']:6.3f} "
              f"{stats[t]['em']:6.3f} {str(tclean(t)):>6}")

    if off_tier:
        off = stats[off_tier]

        def qmean(t, q, k):  # per-question mean over seeds
            return sum(r.get(k, 0) for r in tier_q[t][q]) / len(tier_q[t][q])

        print(f"\nvs OFF (money is the metric; paired per-question, "
              f"95% CI over n={len(common)} questions):")
        print(f"{'tier':8} {'$ saved':>10} {'% cheaper':>10} {'ΔF1':>8} {'ΔEM':>8} "
              f"{'t':>6} {'95% CI of $ saved':>22} {'sig?':>5}")
        for t in tiers:
            if t == off_tier:
                continue
            s = stats[t]
            saved = off["cost"] - s["cost"]
            pct = 100 * saved / off["cost"] if off["cost"] else 0
            diffs = [qmean(off_tier, q, "cost_usd") - qmean(t, q, "cost_usd")
                     for q in common]
            ps = paired_stats(diffs)
            sig = "YES" if ps and ps["significant"] else "no"
            ci = f"[{ps['ci'][0]:+.4f}, {ps['ci'][1]:+.4f}]" if ps else "—"
            print(f"{t:8} {saved:10.5f} {pct:9.1f}% {s['f1']-off['f1']:+8.3f} "
                  f"{s['em']-off['em']:+8.3f} {ps['t'] if ps else 0:6.2f} {ci:>22} {sig:>5}")
        print("(sig? = paired t exceeds the two-sided 95% critical value; "
              "a 'no' means the $ saving is within noise at this sample size)")

    dirty = [(t, q) for t in tiers for q in common
             for r in tier_q[t][q] if not row_clean(r)]
    web = sum(r.get("web_requests", 0) for t in tiers for q in common
              for r in tier_q[t][q])
    print(f"audit: web_requests={web}, dirty(tier,q)={dirty if dirty else 'none'}")


def main():
    recs = load_runs()
    if not recs:
        print("no runs/ found — run the sweep first"); return
    by_gw = defaultdict(list)
    for r in recs:
        by_gw[r["gateway"]].append(r)
    for gw in sorted(by_gw):
        analyze_gateway(gw, by_gw[gw])


if __name__ == "__main__":
    main()
