#!/usr/bin/env python3
"""Budget-tier sweep orchestrator for the cost-aware-agent QA experiment.

Runs the full accuracy-vs-cost curve in ONE process so there is a single honest
completion signal and a coherent set of run dirs. For each tier it (1) writes the
daemon condition to config.json, (2) restarts the daemon so the new config loads
(daemon reads config once at import), (3) waits for health, (4) runs the traced
harness run_claude.py for that tier over the same N questions.

Budget is money. Each tier is a DOLLAR budget the daemon measures real LLM spend
against; OFF is the no-injection control:
  off     inject_enabled=False                        (control — no injection)
  usd30   inject_enabled=True,  budget $0.30
  usd60   inject_enabled=True,  budget $0.60
  usd120  inject_enabled=True,  budget $1.20

Each (tier, seed) is a separate run_claude invocation → its own runs/<run_id>/
with full per-step traces. Usage:
  orchestrate.py --n 10 --seeds 1
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CONFIG = os.path.expanduser("~/.cost-aware-agent/config.json")
DAEMON = "http://127.0.0.1:7331"
LOGDIR = os.environ.get("SWEEP_LOGDIR", HERE)

# Money budgets (cost-aware-agent's thesis): each tier is a DOLLAR budget. The
# session budget in USD is what the injected "spent $X of $Y" pressure is measured
# against; the harness feeds real per-call token cost to /llm/usage so spend is live.
#   (tag, inject_enabled, budget_usd)
TIERS = [("off", False, 0.60), ("usd30", True, 0.30),
         ("usd60", True, 0.60), ("usd120", True, 1.20)]


def set_condition(inject, budget_usd):
    c = json.load(open(CONFIG))
    c.pop("tool_call_budget", None)               # money-only; no tool-call budget
    c["inject_enabled"] = inject
    c["session_budget_estimate_usd"] = budget_usd
    json.dump(c, open(CONFIG, "w"), indent=2)


def restart_daemon():
    subprocess.run(["pkill", "-f", "uvicorn cost_aware_agent.daemon:app"])
    time.sleep(2)
    # Detach fully so this orchestrator (and its eventual exit) doesn't take the
    # daemon down mid-run; we manage its lifetime explicitly per tier.
    log = open(os.path.join(LOGDIR, "daemon_sweep.out"), "a")
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "cost_aware_agent.daemon:app",
         "--host", "127.0.0.1", "--port", "7331"],
        cwd=REPO, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
        start_new_session=True)


def wait_health(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            d = json.load(urllib.request.urlopen(DAEMON + "/health", timeout=5))
            if d.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def probe_injection():
    """Return the additionalContext the daemon injects, to prove the tier is live."""
    try:
        req = urllib.request.Request(
            DAEMON + "/tool/pre", json.dumps({"session_id": "sweep-probe",
            "tool_name": "search"}).encode(), {"content-type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=5)).get("additionalContext")
    except Exception as e:
        return f"(probe failed: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tiers", default="off,on5,on10,on20")
    args = ap.parse_args()

    want = set(args.tiers.split(","))
    tiers = [t for t in TIERS if t[0] in want]

    print(f"=== SWEEP start: tiers={[t[0] for t in tiers]} n={args.n} seeds={args.seeds} ===",
          flush=True)
    for tag, inject, budget_usd in tiers:
        set_condition(inject, budget_usd)
        restart_daemon()
        if not wait_health():
            print(f"[{tag}] DAEMON DID NOT COME UP — skipping", flush=True)
            continue
        inj = probe_injection()
        print(f"\n### TIER {tag} (inject={inject}, budget=${budget_usd}) ###", flush=True)
        print(f"    injection probe: {str(inj)[:80]!r}", flush=True)
        for seed in range(args.seeds):
            # seed only varies the run_id/label; Sonnet sampling supplies the
            # run-to-run variance we want to average over.
            label = tag if args.seeds == 1 else f"{tag}-s{seed}"
            print(f"--- run {label} ---", flush=True)
            subprocess.run(
                [sys.executable, "-u", "run_claude.py", "--tag", label, "--n", str(args.n)],
                cwd=HERE)
    print("\n=== SWEEP done ===", flush=True)


if __name__ == "__main__":
    main()
