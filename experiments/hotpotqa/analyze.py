#!/usr/bin/env python3
"""Accuracy-vs-cost analysis for the money-budget HotpotQA experiment.

Discovers every result tag present in results/ (off, usd30, usd60, usd120, ...)
and prints, per tier, the metrics that matter for the cost-aware claim: real $
spent, tool calls, F1, EM. The question is whether a tighter money budget cuts
spend while holding accuracy. OFF (no injection) is the control.

Tags are NOT hardcoded — any results/<tag>.jsonl is picked up — so it works
whatever dollar tiers the orchestrator ran. Rows are matched by question id
across tiers for a fair paired comparison.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def load(tag):
    rows = {}
    for line in open(os.path.join(RESULTS, f"{tag}.jsonl")):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def tier_order(tag):
    """Sort OFF first, then dollar tiers ascending by their embedded amount."""
    if tag == "off":
        return (0, 0.0)
    digits = "".join(c for c in tag if c.isdigit() or c == ".")
    return (1, float(digits) if digits else 9e9)


def main():
    tags = sorted((os.path.basename(p)[:-6] for p in glob.glob(os.path.join(RESULTS, "*.jsonl"))),
                  key=tier_order)
    if not tags:
        print("no results/*.jsonl found — run the sweep first"); return
    arms = {t: load(t) for t in tags}

    # questions present & non-failed in EVERY arm — the fair common set
    common = sorted(set.intersection(*[{i for i, r in a.items() if not r.get("failed")}
                                       for a in arms.values()]))
    print(f"tags: {tags}")
    print(f"paired questions (present & non-failed in all arms): {len(common)} {common}\n")
    if not common:
        print("no common questions across arms"); return

    def mean(tag, k):
        return sum(arms[tag][i].get(k, 0) for i in common) / len(common)

    hdr = f"{'tier':8} {'cost_usd':>9} {'calls':>6} {'F1':>6} {'EM':>6} {'clean':>6}"
    print(hdr)
    print("-" * len(hdr))
    for t in tags:
        clean = all(arms[t][i].get("audit_clean", True) for i in common)
        print(f"{t:8} {mean(t,'cost_usd'):9.4f} {mean(t,'tool_calls'):6.2f} "
              f"{mean(t,'f1'):6.3f} {mean(t,'em'):6.3f} {str(clean):>6}")

    # cheat-audit rollup across all arms
    dirty = [(t, i) for t in tags for i in common if not arms[t][i].get("audit_clean", True)]
    web = sum(arms[t][i].get("web_requests", 0) for t in tags for i in common)
    print(f"\naudit: web_requests={web}, dirty(tier,q)={dirty if dirty else 'none'}")


if __name__ == "__main__":
    main()
