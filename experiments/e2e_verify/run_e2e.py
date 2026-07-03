#!/usr/bin/env python3
"""E2E feature-verification experiment — real Claude Code + real OpenCode.

NOT an A/B statistics experiment (that's ../real_cli). This is an acceptance
test of the whole harness after the 2026-07-04 backlog + audit work: 5 runs
per agent, each run deliberately exercising specific features, with every
artifact captured so verify.py can machine-check that each feature actually
fired — and that the model never touched the daemon, the DB, or the wallet.

Run matrix (5 per agent, HotpotQA questions from ../real_cli/data/eval.json):

  run  platform  q    project      wallet   exercises
  cc1  claude    q0   cc_off       -        OFF arm: measurement w/o injection, receipt
  cc2  claude    q3   cc_main      $0.50    tracker inject, wallet scope, on_change, burn
  cc3  claude    q9   cc_main      (reuse)  WALLET DEPLETION across sessions + [PROJECT HISTORY]
  cc4  claude    q2   cc_subagent  $0.50    Task-subagent transcript capture (rglob, pull-subagent)
  cc5  claude    q1   cc_tight     $0.06    checkpoint questions, tier escalation, advisory-only
  oc1  opencode  q0   oc_off       -        OFF arm: usage push, retail pricing, receipt
  oc2  opencode  q3   oc_main      $0.006   rebuilt-channel tracker, byte-stability, wallet scope
  oc3  opencode  q9   oc_main      (reuse)  depletion + [PROJECT HISTORY] (harness closes oc2)
  oc4  opencode  q5   oc_tight     $0.002   tight budget: tier escalation on rebuilt channel
  oc5  opencode  q7   oc_over      $0.001   over-budget from early on: CRITICAL + advisory-only

Wallets are set through the real CLI (`cost-aware-agent budget set`) so the
wallet write path under test is the production one. Budgets scale from the
real_cli calibration medians ($0.18/q claude, $0.0025/q opencode).

Capture per run, runs/<stamp>/<run>/:
  task.txt, events.jsonl(+.stderr), transcript.jsonl (CC), subagents/ (cc4),
  hook.jsonl (CC hook fires incl. exact injected text), daemon_dump.json,
  result.json
Plus runs/<stamp>/meta.json (config snapshot, git sha, wallet ledger).
"""
import datetime
import json
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "real_cli"))
import run_real  # reuse: grading, event parsers, killable exec, sandbox audit

HOOK = os.path.join(HERE, "hook.sh")
PLUGIN_SRC = os.path.join(REPO, "cost_aware_agent", "data", "opencode-plugin.ts")
CONFIG = os.path.expanduser("~/.cost-aware-agent/config.json")
DAEMON = "http://127.0.0.1:7331"
CC_MODEL = "sonnet"
OC_MODEL = "opencode/deepseek-v4-flash-free"

CC_ALLOWED = "Read Grep Glob"
CC_DENIED = ["Bash", "Write", "Edit", "WebSearch", "WebFetch", "Task",
             "Agent", "NotebookEdit", "TodoWrite"]

SUBAGENT_EXTRA = (
    "\n\nDelegate the retrieval work: use the Task tool to launch a read-only "
    "subagent that searches corpus/ and reports the relevant passage contents "
    "back to you. Then combine the facts yourself and answer."
)

RUNS = [
    dict(run="cc1", platform="claude", qid="q0", phase="off", wallet=None, project="cc_off"),
    dict(run="cc2", platform="claude", qid="q3", phase="on", wallet=0.50, project="cc_main"),
    dict(run="cc3", platform="claude", qid="q9", phase="on", wallet=None, project="cc_main"),
    dict(run="cc4", platform="claude", qid="q2", phase="on", wallet=0.50, project="cc_subagent", subagent=True),
    dict(run="cc5", platform="claude", qid="q1", phase="on", wallet=0.06, project="cc_tight"),
    dict(run="oc1", platform="opencode", qid="q0", phase="off", wallet=None, project="oc_off"),
    dict(run="oc2", platform="opencode", qid="q3", phase="on", wallet=0.006, project="oc_main"),
    dict(run="oc3", platform="opencode", qid="q9", phase="on", wallet=None, project="oc_main"),
    dict(run="oc4", platform="opencode", qid="q5", phase="on", wallet=0.002, project="oc_tight"),
    dict(run="oc5", platform="opencode", qid="q7", phase="on", wallet=0.001, project="oc_over"),
]


def set_inject(enabled: bool):
    c = json.load(open(CONFIG))
    c["inject_enabled"] = bool(enabled)
    c["inject_mode"] = "on_change"
    json.dump(c, open(CONFIG, "w"), indent=2)


def post(path, body):
    req = urllib.request.Request(
        DAEMON + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=10))
    except Exception as e:
        return {"error": str(e)}


def get_session_state(sid):
    try:
        d = json.load(urllib.request.urlopen(f"{DAEMON}/session/{sid}/dump", timeout=10))
        return (d.get("session") or {}).get("state")
    except Exception:
        return None


def make_project(root, name, q, platform):
    """Persistent sandbox project — SAME directory across runs that share it
    (the project dir is the wallet key), corpus swapped per question."""
    proj = os.path.join(root, "projects", name)
    os.makedirs(proj, exist_ok=True)
    if not os.path.isdir(os.path.join(proj, ".git")):
        subprocess.run(["git", "init", "-q", proj], capture_output=True)
    corpus = os.path.join(proj, "corpus")
    shutil.rmtree(corpus, ignore_errors=True)
    os.makedirs(corpus)
    import re
    for title, text in q["passages"].items():
        slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") or "passage"
        with open(os.path.join(corpus, f"{slug}.txt"), "w") as f:
            f.write(f"TITLE: {title}\n\n{text}\n")
    if platform == "opencode":
        pl = os.path.join(proj, ".opencode", "plugins")
        os.makedirs(pl, exist_ok=True)
        shutil.copy(PLUGIN_SRC, os.path.join(pl, "cost-aware-agent.ts"))
        json.dump({
            "$schema": "https://opencode.ai/config.json",
            "tools": {"bash": False, "edit": False, "write": False,
                      "webfetch": False, "patch": False},
            "permission": {"edit": "deny", "bash": "deny", "webfetch": "deny"},
        }, open(os.path.join(proj, "opencode.json"), "w"), indent=2)
    return proj


def cc_settings(run_dir):
    path = os.path.join(run_dir, "hook_settings.json")
    events = ["SessionStart", "PreToolUse", "PostToolUse", "Stop", "SessionEnd"]
    json.dump({"hooks": {e: [{"hooks": [{"type": "command", "command": HOOK}]}]
                         for e in events}}, open(path, "w"), indent=2)
    return path


def drive_claude(spec, task, run_dir, proj):
    hook_log = os.path.join(run_dir, "hook.jsonl")
    open(hook_log, "w").close()
    env = dict(os.environ, CAA_HOOK_LOG=hook_log, PWD=proj)
    events_path = os.path.join(run_dir, "events.jsonl")
    allowed, denied = CC_ALLOWED, list(CC_DENIED)
    if spec.get("subagent"):
        allowed = "Read Grep Glob Task Agent"
        denied.remove("Task")
        denied.remove("Agent")
    cmd = ["claude", "-p", task, "--model", CC_MODEL,
           "--output-format", "stream-json", "--verbose",
           "--settings", cc_settings(run_dir),
           "--allowedTools", allowed,
           "--disallowedTools", " ".join(denied)]
    timeout = 600 if spec.get("subagent") else 480
    err = run_real.run_killable(cmd, timeout=timeout, cwd=proj, env=env,
                                out_path=events_path)
    tool_uses, result = run_real.parse_cc_events(events_path)
    sid = (result or {}).get("session_id")
    if sid:
        import glob as g
        hits = g.glob(os.path.expanduser(f"~/.claude/projects/*/{sid}.jsonl"))
        if hits:
            shutil.copy(hits[0], os.path.join(run_dir, "transcript.jsonl"))
            # subagent transcripts live in a sibling DIR named after the session
            subdir = os.path.join(os.path.dirname(hits[0]), sid, "subagents")
            if os.path.isdir(subdir):
                shutil.copytree(subdir, os.path.join(run_dir, "subagents"),
                                dirs_exist_ok=True)
    usage = (result or {}).get("usage") or {}
    stu = usage.get("server_tool_use") or {}
    return {
        "session_id": sid,
        "final_text": (result or {}).get("result", ""),
        "cli_cost_usd": (result or {}).get("total_cost_usd"),
        "num_turns": (result or {}).get("num_turns"),
        "tool_uses": tool_uses,
        "web_requests": (stu.get("web_search_requests", 0)
                         + stu.get("web_fetch_requests", 0)),
        "timed_out": err == "TIMEOUT",
    }


def drive_opencode(spec, task, run_dir, proj):
    events_path = os.path.join(run_dir, "events.jsonl")
    cmd = ["opencode", "run", "--format", "json", "--print-logs",
           "--log-level", "ERROR", "-m", OC_MODEL, task]
    err = run_real.run_killable(cmd, timeout=480, cwd=proj,
                                env=dict(os.environ, PWD=proj),
                                out_path=events_path)
    final_text, tool_uses, sid = run_real.parse_oc_events(events_path)
    return {
        "session_id": sid,
        "final_text": final_text,
        "cli_cost_usd": None,  # free tier bills $0; money = daemon retail cost
        "num_turns": None,
        "tool_uses": tool_uses,
        "web_requests": sum(1 for t in tool_uses if t.get("name") == "webfetch"),
        "timed_out": err == "TIMEOUT",
    }


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = os.path.join(HERE, "runs", stamp)
    os.makedirs(root, exist_ok=True)
    questions = {q["id"]: q for q in json.load(
        open(os.path.join(HERE, "..", "real_cli", "data", "eval.json")))}

    meta = {
        "started": stamp,
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True).stdout.strip(),
        "cc_model": CC_MODEL, "oc_model": OC_MODEL,
        "runs": RUNS, "wallets": {},
    }

    current_phase = None
    for spec in RUNS:
        if spec["phase"] != current_phase:
            set_inject(spec["phase"] == "on")
            if not run_real.restart_daemon():
                raise SystemExit("daemon failed to restart")
            current_phase = spec["phase"]
            print(f"--- phase {current_phase}: daemon restarted, "
                  f"inject_enabled={current_phase == 'on'}")

        q = questions[spec["qid"]]
        run_dir = os.path.join(root, spec["run"])
        os.makedirs(run_dir, exist_ok=True)
        proj = make_project(root, spec["project"], q, spec["platform"])

        if spec.get("wallet"):
            r = subprocess.run(["cost-aware-agent", "budget", "set",
                                str(spec["wallet"]), "--project-dir", proj],
                               capture_output=True, text=True)
            meta["wallets"][spec["project"]] = {
                "amount": spec["wallet"], "cli_stdout": r.stdout.strip(),
                "cli_rc": r.returncode}

        task = run_real.TASK_TMPL.format(question=q["question"])
        if spec.get("subagent"):
            task += SUBAGENT_EXTRA
        open(os.path.join(run_dir, "task.txt"), "w").write(task)

        drive = drive_claude if spec["platform"] == "claude" else drive_opencode
        out = drive(spec, task, run_dir, proj)
        if not out.get("final_text") and not out.get("timed_out"):
            print(f"[{spec['run']}] empty output, retrying once...")
            import time
            time.sleep(20)
            out = drive(spec, task, run_dir, proj)

        sid = out.get("session_id")
        # Ensure the session is CLOSED before any later run reads history from
        # it. CC should close itself via SessionEnd; OpenCode has no such event
        # (sessions normally finish via the 24h abandoned rule), so the harness
        # closes it — record which path it was, verify.py checks CC did its own.
        state_before_close = get_session_state(sid) if sid else None
        forced_stop = False
        if sid and state_before_close != "ended":
            post("/session/stop", {"session_id": sid})
            forced_stop = True
        dump = run_real.daemon_dump(sid) if sid else {}
        json.dump(dump, open(os.path.join(run_dir, "daemon_dump.json"), "w"), indent=1)

        pred, had_line = run_real.extract_answer(out["final_text"])
        em, f1 = run_real.score(pred, q["answer"])
        tool_names = [t.get("name") for t in out["tool_uses"]]
        escapes = run_real.outside_sandbox_refs(out["tool_uses"], proj)
        # CC >= 2.1.x surfaces the subagent tool as "Agent" (older: "Task")
        danger = set(run_real.DANGER) | {"Agent"}
        if spec.get("subagent"):
            danger -= {"Task", "Agent"}
        row = {
            "run": spec["run"], "platform": spec["platform"], "qid": spec["qid"],
            "phase": spec["phase"], "project": proj, "wallet": spec.get("wallet"),
            "question": q["question"], "gold": q["answer"],
            "answer": pred, "had_answer_line": had_line, "em": em, "f1": round(f1, 3),
            "session_id": sid,
            "cli_cost_usd": out.get("cli_cost_usd"),
            "daemon_spent_usd": dump.get("spent_usd"),
            "budget_view": dump.get("budget_view"),
            "num_turns": out.get("num_turns"),
            "tool_calls": len(tool_names), "tool_names": tool_names,
            "injections_delivered": len(dump.get("injections") or []),
            "state_before_harness_close": state_before_close,
            "forced_stop": forced_stop,
            "web_requests": out.get("web_requests", 0),
            "outside_sandbox_refs": escapes,
            "danger_tools_used": sorted(set(tool_names) & danger),
            "timed_out": out.get("timed_out", False),
            "failed": not out.get("final_text"),
        }
        json.dump(row, open(os.path.join(run_dir, "result.json"), "w"), indent=2)
        print(f"[{spec['run']}] em={em} f1={row['f1']} "
              f"cli_cost={row['cli_cost_usd']} daemon={row['daemon_spent_usd']} "
              f"tools={row['tool_calls']} inj={row['injections_delivered']} "
              f"state={state_before_close} forced={forced_stop} "
              f"ans={pred[:40]!r}", flush=True)

    json.dump(meta, open(os.path.join(root, "meta.json"), "w"), indent=2)
    print(f"\nDONE. artifacts: {root}\nnext: python3 verify.py {root}")


if __name__ == "__main__":
    main()
