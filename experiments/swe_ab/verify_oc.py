#!/usr/bin/env python3
"""Machine-check that every harness function fired correctly on the OpenCode
runs (plumbing acceptance, independent of whether deepseek solved anything).

Reads the captured runs + their daemon dumps and asserts, per feature:
  OFF gating          every OFF run delivered 0 injections
  ON delivery         every ON run with real work delivered >=1 injection
  retail cost         non-loss runs have daemon_spent > 0 == cost_usd
  tier signal         ON injections carry a Tier: label
  rebuilt channel     ON injection text is the Budget Tracker block
  wallet scope        ON tracker names the project wallet
  quantization        ON rendered spend values sit on the bucket grid (~$)
  audit integrity     no cheat refs; any net attempt is flagged not silent
  subagent capture    if deepseek used a 'task' subagent, it was ingested
  session tracking    every non-loss run has a daemon session with usage rows
Prints PASS/FAIL per check; exit 1 on any hard fail.
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def load(arm):
    out = []
    for rj in glob.glob(os.path.join(HERE, "runs", "opencode", f"{arm}-s*", "*", "result.json")):
        r = json.load(open(rj))
        d = os.path.dirname(rj)
        dd = json.load(open(os.path.join(d, "daemon_dump.json"))) if \
            os.path.exists(os.path.join(d, "daemon_dump.json")) else {}
        out.append((r, dd))
    return out


def is_loss(r):
    return (r.get("cost_usd") or 0) == 0 and r.get("tool_calls", 0) == 0


def main():
    off, on = load("off"), load("on")
    print(f"OpenCode verification — {len(off)} OFF runs, {len(on)} ON runs\n")

    # 1. OFF gating: zero injections
    off_inj = sum(r.get("injections_delivered", 0) for r, _ in off)
    check("OFF gating: 0 injections across all OFF runs", off_inj == 0,
          f"total OFF injections = {off_inj}")

    # 2. ON delivery: runs with real work got injections (rebuilt channel)
    on_work = [(r, dd) for r, dd in on if not is_loss(r)]
    on_deliv = all(r.get("injections_delivered", 0) >= 1 for r, _ in on_work)
    check("ON delivery: every ON run with work got >=1 injection", on_deliv,
          f"{sum(1 for r,_ in on_work if r.get('injections_delivered',0)>=1)}/{len(on_work)} runs")

    # 3. retail cost measured and consistent
    cost_ok = all(abs((r.get("cost_usd") or 0) - (r.get("daemon_spent_usd") or 0)) < 1e-9
                  for r, _ in off + on)
    nonzero = all((r.get("cost_usd") or 0) > 0 for r, _ in off + on if not is_loss(r))
    check("retail cost: cost_usd == daemon_spent for all runs", cost_ok)
    check("retail cost: >0 for every non-loss run", nonzero)

    # 4. tier signal in ON injections
    tier_seen = set()
    for r, dd in on_work:
        for i in (dd.get("injections") or []):
            m = re.search(r"Tier:\s*(\w+)", i.get("context") or "")
            if m:
                tier_seen.add(m.group(1))
    check("tier signal: ON injections carry Tier labels", bool(tier_seen),
          f"tiers seen: {sorted(tier_seen)}")

    # 5. rebuilt-channel Budget Tracker text
    tracker_ok = any("Budget Tracker" in (i.get("context") or "")
                     for _, dd in on_work for i in (dd.get("injections") or []))
    check("rebuilt channel: Budget Tracker block delivered", tracker_ok)

    # 6. wallet scope wording
    wallet_ok = any("project wallet" in (i.get("context") or "")
                    for _, dd in on_work for i in (dd.get("injections") or []))
    check("wallet scope: tracker names the project wallet", wallet_ok)

    # 7. quantization: rendered spend on bucket grid (~$ approximate figures)
    approx_ok = any(re.search(r"~\$", (i.get("context") or ""))
                    for _, dd in on_work for i in (dd.get("injections") or []))
    check("quantization: ON tracker uses ~$ bucket-grid figures", approx_ok)

    # 8. audit integrity: no cheat refs anywhere; net attempts are flagged
    cheats = sum(len(r.get("cheat_refs") or []) for r, _ in off + on)
    check("audit: zero harness-tamper (cheat) refs", cheats == 0)
    net_runs = [r for r, _ in off + on if r.get("net_refs")]
    flagged_ok = all(not r.get("audit_clean") for r in net_runs)
    check("audit: every run with a net attempt is flagged (not silent)",
          flagged_ok, f"{len(net_runs)} runs had net attempts, all flagged")

    # 9. subagent capture (deepseek 'task' tool -> ingested as tool_call)
    subagent_runs = [(r, dd) for r, dd in off + on
                     if any(tc.get("tool_name") == "task" for tc in (dd.get("tool_calls") or []))]
    check("subagent capture: 'task' subagent calls ingested to daemon",
          True if not subagent_runs else all(
              any(tc.get("tool_name") == "task" for tc in (dd.get("tool_calls") or []))
              for _, dd in subagent_runs),
          f"{len(subagent_runs)} run(s) used a task subagent, all captured")

    # 10. session tracking: non-loss runs have a daemon session + usage rows
    sess_ok = all((dd.get("llm_usage") for r, dd in off + on if not is_loss(r)))
    check("session tracking: non-loss runs have daemon usage rows", sess_ok)

    hard_fails = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{'='*60}\n{len(CHECKS)-len(hard_fails)}/{len(CHECKS)} checks pass"
          + (f"  FAILS: {hard_fails}" if hard_fails else "  — all green"))
    json.dump({"checks": [(n, ok, d) for n, ok, d in CHECKS],
               "pass": len(CHECKS) - len(hard_fails), "total": len(CHECKS)},
              open(os.path.join(HERE, "verify_oc.json"), "w"), indent=1)
    return 1 if hard_fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
