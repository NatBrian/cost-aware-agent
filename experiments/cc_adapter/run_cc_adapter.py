#!/usr/bin/env python3
"""Real-adapter A/B — does budget injection through the ACTUAL Claude Code hook
change behaviour, the way it did when the HotpotQA harness fed the tracker
straight into the model's prompt?

The HotpotQA experiment (../hotpotqa) proved a capable model spends ~28% less
when the Budget Tracker is IN ITS PROMPT (high salience). But real Claude Code
delivers hook output as a low-salience `additionalContext` system-reminder. This
harness closes that gap: it launches a real `claude -p` per condition, lets
Claude Code run its OWN agent loop and native tools, and the live PreToolUse hook
injects the daemon's Budget Tracker on every tool call — exactly the production
path. We only set the condition (daemon config) and capture everything.

Two task MODES with different room for the budget to bite:
  bugs      review a 10-file repo and find all planted bugs (LOW slack — must
            read every file; recall graded against a known bug manifest)
  overview  give a brief high-level overview (HIGH slack — a shallow answer is
            valid; exploration depth is discretionary; module coverage graded)

Per run we store, under runs/<mode>/<tag>-s<seed>/:
  events.jsonl      full CC stream-json (every tool_use / result)
  transcript.jsonl  copy of CC's own session transcript (delivery ground truth)
  hook.jsonl        every hook fire: body POSTed + additionalContext injected
  trajectory.jsonl  parsed per-step assistant text + tool calls
  report.txt        the model's final answer
  summary.json      cost, tools, injections delivered, tiers, recall/coverage

Usage:
  python build_sandbox.py                       # generate sandbox/ once
  run_cc_adapter.py --mode bugs     --tag off --budget 0    --seeds 3
  run_cc_adapter.py --mode bugs     --tag on  --budget 0.05 --seeds 3
  run_cc_adapter.py --mode overview --tag off --budget 0    --seeds 3
  run_cc_adapter.py --mode overview --tag on  --budget 0.05 --seeds 3
"""
import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.join(HERE, "sandbox")
HOOK = os.path.join(HERE, "logging_hook.sh")
HOOK_SETTINGS = os.path.join(HERE, ".hooks_settings.json")  # generated, gitignored
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CONFIG = os.path.expanduser("~/.cost-aware-agent/config.json")
DAEMON = "http://127.0.0.1:7331"

MODULES = ["auth", "cart", "inventory", "pricing", "user", "util", "report",
           "cache", "config", "validate"]

TASKS = {
    "bugs": ("Review this codebase for bugs. Explore the source files under src/ "
             "and report every bug you find as `file:line - description`. Be "
             "thorough and cover all files."),
    "overview": ("Give me a brief high-level overview of what this codebase does "
                 "and how it is organized. A short paragraph is fine — you don't "
                 "need to read everything."),
}

BUDGET_REF = re.compile(r"\b(budget|cost|\$[0-9]|tier|token|remaining|finaliz|"
                        r"wrap up|stop here|to save|econom)", re.I)


def grade(mode, report_text):
    """bugs -> recall over the planted-bug manifest; overview -> module coverage
    (a depth proxy: a brief overview mentioning fewer modules is still valid)."""
    txt = (report_text or "").lower()
    if mode == "bugs":
        manifest = json.load(open(os.path.join(SANDBOX, "bug_manifest.json")))
        hit = [b for b, f, kws in manifest
               if f.lower() in txt and any(k.lower() in txt for k in kws)]
        return hit, len(manifest)
    hit = [m for m in MODULES if m in txt]
    return hit, len(MODULES)


def write_hook_settings():
    events = ["SessionStart", "PreToolUse", "PostToolUse", "Stop"]
    cfg = {"hooks": {e: [{"hooks": [{"type": "command", "command": HOOK}]}]
                     for e in events}}
    json.dump(cfg, open(HOOK_SETTINGS, "w"), indent=2)


def set_config(inject, budget):
    c = json.load(open(CONFIG))
    c["inject_enabled"] = bool(inject)
    if budget and budget > 0:
        c["session_budget_estimate_usd"] = float(budget)
    json.dump(c, open(CONFIG, "w"), indent=2)


def restart_daemon():
    subprocess.run(["pkill", "-f", "uvicorn cost_aware_agent.daemon:app"],
                   capture_output=True)
    time.sleep(2)
    subprocess.Popen(
        ["python3", "-m", "uvicorn", "cost_aware_agent.daemon:app",
         "--host", "127.0.0.1", "--port", "7331"],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, start_new_session=True)
    import urllib.request
    for _ in range(30):
        try:
            if b"ok" in urllib.request.urlopen(DAEMON + "/health", timeout=2).read():
                return True
        except Exception:
            time.sleep(1)
    return False


def cc_transcript(session_id):
    hits = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{session_id}.jsonl"))
    return hits[0] if hits else None


def run_killable(cmd, timeout, cwd, env, out_path):
    with open(out_path, "w") as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.PIPE, text=True,
                             cwd=cwd, env=env, start_new_session=True)
        try:
            _, err = p.communicate(timeout=timeout)
            return err or ""
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.communicate()
            return "TIMEOUT"


def parse_events(events_path):
    tool_uses, steps, result = [], [], None
    for line in open(events_path):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "assistant":
            blocks = e.get("message", {}).get("content") or []
            texts = [b.get("text") for b in blocks
                     if isinstance(b, dict) and b.get("type") == "text"]
            tus = [{"name": b.get("name"), "input": b.get("input")} for b in blocks
                   if isinstance(b, dict) and b.get("type") == "tool_use"]
            tool_uses += tus
            steps.append({"texts": [t for t in texts if t is not None],
                          "tool_uses": tus})
        elif e.get("type") == "result":
            result = e
    return tool_uses, steps, result


def run_one(mode, tag, seed, inject, budget):
    run_dir = os.path.join(HERE, "runs", mode, f"{tag}-s{seed}")
    os.makedirs(run_dir, exist_ok=True)
    hook_log = os.path.join(run_dir, "hook.jsonl")
    open(hook_log, "w").close()
    events_path = os.path.join(run_dir, "events.jsonl")

    env = dict(os.environ)
    env["CAA_HOOK_LOG"] = hook_log
    cmd = ["claude", "-p", TASKS[mode], "--model", "sonnet",
           "--output-format", "stream-json", "--verbose",
           # headless CC won't run project .claude hooks from an untrusted dir;
           # load the cost-aware hooks explicitly so injection actually fires.
           "--settings", HOOK_SETTINGS,
           "--allowedTools", "Read Grep Glob",
           "--disallowedTools", "Write Edit Bash"]
    open(os.path.join(run_dir, "cmd.txt"), "w").write(" ".join(cmd))

    t0 = time.time()
    err = run_killable(cmd, timeout=420, cwd=SANDBOX, env=env, out_path=events_path)
    dur = round(time.time() - t0, 1)

    tool_uses, steps, result = parse_events(events_path)
    json.dump(steps, open(os.path.join(run_dir, "trajectory.jsonl"), "w"), indent=1)
    session_id = (result or {}).get("session_id")
    cost = (result or {}).get("total_cost_usd")

    delivered = 0
    tr = cc_transcript(session_id) if session_id else None
    if tr:
        shutil.copy(tr, os.path.join(run_dir, "transcript.jsonl"))
        delivered = open(tr, errors="ignore").read().count("Budget Tracker")

    tiers, hook_inj, hook_null = [], 0, 0
    for line in open(hook_log):
        line = line.strip()
        if not line:
            continue
        h = json.loads(line)
        if h.get("injected"):
            hook_inj += 1
            m = re.search(r"Tier:\s*(\w+)", h["injected"])
            if m:
                tiers.append(m.group(1))
        elif h.get("injected_null"):
            hook_null += 1

    report = (result or {}).get("result", "")
    open(os.path.join(run_dir, "report.txt"), "w").write(report or "")
    hit, total = grade(mode, report)
    quality = round(len(hit) / total, 3) if total else None
    qname = "recall" if mode == "bugs" else "coverage"

    summary = {
        "mode": mode, "tag": tag, "seed": seed, "inject": bool(inject),
        "budget": budget, "duration_s": dur, "cost_usd": cost,
        "num_turns": (result or {}).get("num_turns"),
        "n_tool_calls": len(tool_uses),
        "tool_names": sorted({t["name"] for t in tool_uses if t.get("name")}),
        "injections_in_transcript": delivered,
        "hook_fires_injected": hook_inj, "hook_fires_null": hook_null,
        "tiers_seen": tiers,
        qname: quality, f"{qname}_hits": hit, f"{qname}_total": total,
        "model_budget_reference_texts":
            sum(bool(BUDGET_REF.search(t)) for s in steps for t in s["texts"]),
        "timed_out": err == "TIMEOUT",
    }
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
    print(f"[{mode}/{tag}-s{seed}] cost=${cost} tools={len(tool_uses)} "
          f"{qname}={quality} inj={delivered} tiers={tiers[:2]}"
          f"{'..' if len(tiers) > 2 else ''} to={summary['timed_out']}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bugs", "overview"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    if not os.path.isdir(SANDBOX):
        print("sandbox/ missing — run `python build_sandbox.py` first"); return
    write_hook_settings()
    inject = args.tag != "off"
    set_config(inject, args.budget)
    if not restart_daemon():
        print("daemon failed to start"); return
    print(f"=== {args.mode}/{args.tag}: inject={inject} budget=${args.budget} ===",
          flush=True)
    for s in range(args.seeds):
        run_one(args.mode, args.tag, s, inject, args.budget)


if __name__ == "__main__":
    main()
