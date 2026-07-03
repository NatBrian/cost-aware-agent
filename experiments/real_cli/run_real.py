#!/usr/bin/env python3
"""Real-CLI HotpotQA A/B — the production harness on real agent loops.

This is the experiment the earlier two couldn't be: REAL Claude Code and REAL
OpenCode running their OWN agent loops with their NATIVE tools over an actual
benchmark dataset (HotpotQA distractor), with the cost-aware-agent daemon
injecting the money budget through each CLI's production channel:

  claude    `claude -p` + live hooks via --settings (PreToolUse injects the
            Budget Tracker as additionalContext — accumulating channel, so the
            daemon's on_change anti-tax policy is what's under test)
  opencode  `opencode run` + the production plugin copied into the sandbox
            project (system-transform injection — rebuilt channel)

Per question the sandbox contains that question's OWN 10 distractor passages
(2 gold + 8 distractors, straight from HotpotQA) as corpus/*.txt files. The
agent must retrieve with Read/Grep/Glob (CC) or read/grep/glob (OpenCode).
Gold answers never enter the sandbox.

FULL CAPTURE per run, runs/<platform>/<tag>-s<seed>/<qid>/:
  events.jsonl      the CLI's complete stream-json/json event stream
  transcript.jsonl  Claude Code's own session transcript (delivery ground truth)
  hook.jsonl        (claude) every hook fire: body POSTed + context injected
  daemon_dump.json  daemon-side export: llm_usage, tool_calls, every delivered
                    injection, plan rows (GET /session/<sid>/dump)
  result.json       answer, EM/F1, cost, tools, audit
  task.txt          the exact prompt given to the CLI

Usage:
  run_real.py --platform claude   --tag off --budget 0     --seeds 3 --data data/eval.json
  run_real.py --platform claude   --tag on  --budget 0.10  --seeds 3 --data data/eval.json
  run_real.py --platform opencode --tag off --budget 0     --seeds 3 --data data/eval.json
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import signal
import string
import subprocess
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
HOOK = os.path.join(HERE, "hook.sh")
PLUGIN_SRC = os.path.join(REPO, "cost_aware_agent", "data", "opencode-plugin.ts")
CONFIG = os.path.expanduser("~/.cost-aware-agent/config.json")
DAEMON = "http://127.0.0.1:7331"

CC_MODEL = "sonnet"
OC_MODEL = "opencode/deepseek-v4-flash-free"

# Tools the agent may use (retrieval only). Everything else is hard-denied on
# CC; on OpenCode write/edit/bash/web are denied via sandbox opencode.json.
CC_ALLOWED = "Read Grep Glob"
CC_DENIED = ["Bash", "Write", "Edit", "WebSearch", "WebFetch", "Task",
             "NotebookEdit", "TodoWrite"]

TASK_TMPL = (
    "Answer the following multi-hop question using ONLY the passage files in "
    "the corpus/ directory of this project. The answer requires combining facts "
    "from more than one passage. Do not use outside knowledge; every fact must "
    "come from the corpus files.\n\n"
    "QUESTION: {question}\n\n"
    "When you are confident, end your reply with exactly one final line:\n"
    "ANSWER: <short answer — a name, date, or phrase>"
)


# --- grading (SQuAD/HotpotQA normalization, same as ../hotpotqa) ---

def _norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def score(pred, gold):
    np_, ng = _norm(pred), _norm(gold)
    em = float(np_ == ng)
    pt, gt = np_.split(), ng.split()
    if not pt or not gt:
        return em, float(np_ == ng)
    common = {}
    for w in pt:
        if w in gt:
            common[w] = min(pt.count(w), gt.count(w))
    nsame = sum(common.values())
    if nsame == 0:
        return em, 0.0
    prec, rec = nsame / len(pt), nsame / len(gt)
    return em, 2 * prec * rec / (prec + rec)


def extract_answer(text):
    """Last 'ANSWER:' line wins; flag when the protocol line is missing."""
    matches = re.findall(r"^\s*ANSWER:\s*(.+?)\s*$", text or "", re.MULTILINE)
    if matches:
        return matches[-1], True
    return (text or "").strip(), False


# --- daemon condition ---

def set_condition(inject, budget_usd):
    c = json.load(open(CONFIG))
    c["inject_enabled"] = bool(inject)
    c["session_budget_estimate_usd"] = float(budget_usd) if budget_usd else 1.0
    c["inject_mode"] = "on_change"  # the production default is what's under test
    json.dump(c, open(CONFIG, "w"), indent=2)


def restart_daemon():
    subprocess.run(["pkill", "-f", "uvicorn cost_aware_agent[.]daemon"],
                   capture_output=True)
    time.sleep(2)
    log = open(os.path.join(HERE, "daemon_sweep.out"), "a")
    subprocess.Popen(
        ["python3", "-m", "uvicorn", "cost_aware_agent.daemon:app",
         "--host", "127.0.0.1", "--port", "7331"],
        cwd=REPO, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
        start_new_session=True)
    for _ in range(30):
        try:
            if b"ok" in urllib.request.urlopen(DAEMON + "/health", timeout=2).read():
                return True
        except Exception:
            time.sleep(1)
    return False


def daemon_dump(session_id):
    try:
        return json.load(urllib.request.urlopen(
            f"{DAEMON}/session/{session_id}/dump", timeout=10))
    except Exception as e:
        return {"error": str(e)}


# --- sandbox ---

def make_sandbox(q, platform):
    sb = tempfile.mkdtemp(prefix=f"real_cli_{q['id']}_")
    # git init makes the sandbox an unambiguous project root. Without it,
    # OpenCode does NOT bind the cwd as its project: a live calibration run
    # got the REPO as project root — the sandbox opencode.json (tool denies)
    # was ignored, bash ran, and the model globbed its way to data/calib.json
    # and read the gold answer (caught by the path audit, em invalidated).
    # With .git present, tools:false is honored and the project path the model
    # sees is the sandbox (verified live).
    subprocess.run(["git", "init", "-q", sb], capture_output=True)
    corpus = os.path.join(sb, "corpus")
    os.makedirs(corpus)
    for title, text in q["passages"].items():
        slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") or "passage"
        with open(os.path.join(corpus, f"{slug}.txt"), "w") as f:
            f.write(f"TITLE: {title}\n\n{text}\n")
    if platform == "opencode":
        pl = os.path.join(sb, ".opencode", "plugins")
        os.makedirs(pl)
        shutil.copy(PLUGIN_SRC, os.path.join(pl, "cost-aware-agent.ts"))
        json.dump({
            "$schema": "https://opencode.ai/config.json",
            # retrieval-only sandbox. `tools: false` REMOVES the tool from the
            # model entirely — verified live: a forced bash attempt becomes an
            # "invalid" tool error and the model reports TOOL-UNAVAILABLE.
            # `permission: deny` alone did NOT block bash in a live test
            # (a calibration run executed `find /home/...` through it), so the
            # tools map is the load-bearing layer and permission is belt+braces.
            "tools": {"bash": False, "edit": False, "write": False,
                      "webfetch": False, "patch": False},
            "permission": {"edit": "deny", "bash": "deny", "webfetch": "deny"},
        }, open(os.path.join(sb, "opencode.json"), "w"), indent=2)
    return sb


def run_killable(cmd, timeout, cwd, env, out_path):
    """stderr goes to <out_path>.stderr — an OpenCode session that dies on a
    provider error (rate limit etc.) exits 0 with a truncated event stream and
    the only evidence lands on stderr; it must be kept, not piped to nowhere."""
    with open(out_path, "w") as out, open(out_path + ".stderr", "w") as errf:
        p = subprocess.Popen(cmd, stdout=out, stderr=errf, text=True,
                             cwd=cwd, env=env, start_new_session=True)
        try:
            p.communicate(timeout=timeout)
            return ""
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
            p.communicate()
            return "TIMEOUT"


# --- platform drivers -------------------------------------------------------

def cc_hook_settings(run_dir):
    path = os.path.join(run_dir, "hook_settings.json")
    events = ["SessionStart", "PreToolUse", "PostToolUse", "Stop"]
    json.dump({"hooks": {e: [{"hooks": [{"type": "command", "command": HOOK}]}]
                         for e in events}}, open(path, "w"), indent=2)
    return path


def parse_cc_events(path):
    tool_uses, result = [], None
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "assistant":
            for b in (e.get("message", {}).get("content") or []):
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tool_uses.append({"name": b.get("name"), "input": b.get("input")})
        elif e.get("type") == "result":
            result = e
    return tool_uses, result


def drive_claude(q, run_dir, sandbox):
    task = TASK_TMPL.format(question=q["question"])
    open(os.path.join(run_dir, "task.txt"), "w").write(task)
    hook_log = os.path.join(run_dir, "hook.jsonl")
    open(hook_log, "w").close()
    # PWD must match the sandbox: subprocess cwd= changes the real cwd but
    # inherits the parent's stale PWD env var, and OpenCode resolves its
    # project root from PWD — a live run bound to the REPO instead of the
    # sandbox and grep'd the gold answers out of data/calib.json (caught by
    # the path audit). Set for both platforms; harmless where unused.
    env = dict(os.environ, CAA_HOOK_LOG=hook_log, PWD=sandbox)
    events_path = os.path.join(run_dir, "events.jsonl")
    cmd = ["claude", "-p", task, "--model", CC_MODEL,
           "--output-format", "stream-json", "--verbose",
           "--settings", cc_hook_settings(run_dir),
           "--allowedTools", CC_ALLOWED,
           "--disallowedTools", " ".join(CC_DENIED)]
    err = run_killable(cmd, timeout=420, cwd=sandbox, env=env, out_path=events_path)

    tool_uses, result = parse_cc_events(events_path)
    session_id = (result or {}).get("session_id")
    # copy CC's own transcript — delivery ground truth for injections
    if session_id:
        hits = glob.glob(os.path.expanduser(
            f"~/.claude/projects/*/{session_id}.jsonl"))
        if hits:
            shutil.copy(hits[0], os.path.join(run_dir, "transcript.jsonl"))
    usage = (result or {}).get("usage") or {}
    stu = usage.get("server_tool_use") or {}
    return {
        "session_id": session_id,
        "final_text": (result or {}).get("result", ""),
        "cost_usd": (result or {}).get("total_cost_usd"),
        "num_turns": (result or {}).get("num_turns"),
        "tool_uses": tool_uses,
        "web_requests": (stu.get("web_search_requests", 0)
                         + stu.get("web_fetch_requests", 0)),
        "permission_denials": (result or {}).get("permission_denials") or [],
        "timed_out": err == "TIMEOUT",
    }


def parse_oc_events(path):
    """opencode run --format json emits JSONL part events. Collect assistant
    text, tool parts (name + state), session id."""
    texts, tool_uses, session_id = [], [], None
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        part = e.get("part") if isinstance(e, dict) else None
        if not isinstance(part, dict):
            continue
        session_id = part.get("sessionID") or session_id
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif part.get("type") == "tool":
            tool_uses.append({"name": part.get("tool"),
                              "status": (part.get("state") or {}).get("status"),
                              "input": (part.get("state") or {}).get("input")})
    return "".join(texts), tool_uses, session_id


def drive_opencode(q, run_dir, sandbox):
    task = TASK_TMPL.format(question=q["question"])
    open(os.path.join(run_dir, "task.txt"), "w").write(task)
    events_path = os.path.join(run_dir, "events.jsonl")
    cmd = ["opencode", "run", "--format", "json", "--print-logs",
           "--log-level", "ERROR", "-m", OC_MODEL, task]
    # PWD=sandbox is load-bearing here — see the note in drive_claude
    err = run_killable(cmd, timeout=420, cwd=sandbox,
                       env=dict(os.environ, PWD=sandbox),
                       out_path=events_path)
    final_text, tool_uses, session_id = parse_oc_events(events_path)
    return {
        "session_id": session_id,
        "final_text": final_text,
        "cost_usd": None,       # free tier bills $0 — money = daemon retail cost
        "num_turns": None,
        "tool_uses": tool_uses,
        "web_requests": sum(1 for t in tool_uses if t.get("name") == "webfetch"),
        "permission_denials": [],
        "timed_out": err == "TIMEOUT",
    }


DANGER = {"Bash", "Write", "Edit", "WebSearch", "WebFetch", "Task",
          "NotebookEdit", "bash", "write", "edit", "webfetch", "patch"}

_ABS_PATH_RE = re.compile(r"/(?:home|etc|root|var|proc|usr|opt)[^\s\"']*")


def outside_sandbox_refs(tool_uses, sandbox):
    """Both CLIs' native Read/read accept ABSOLUTE paths, so a curious model
    could reach outside the sandbox (one calibration run probed
    `find /home/... -name corpus` before bash was removed) — where the gold
    answers live on disk. Flag any tool input referencing an absolute path
    that is not inside the sandbox. /tmp paths are not flagged: the sandbox
    itself lives there and tempfile prefixes vary."""
    hits = []
    for t in tool_uses:
        blob = json.dumps(t.get("input") or t or {})
        for m in _ABS_PATH_RE.findall(blob):
            if not m.startswith(sandbox):
                hits.append({"tool": t.get("name"), "path": m[:120]})
    return hits


def run_question(platform, q, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    sandbox = make_sandbox(q, platform)
    retried = False
    try:
        drive = drive_claude if platform == "claude" else drive_opencode
        out = drive(q, run_dir, sandbox)
        # Two observed transient failure shapes, both retried once:
        # (a) CLI wedge: zero output until timeout (daemon-restart race);
        # (b) OpenCode session dying silently on a provider error — exit 0,
        #     truncated event stream, no final text.
        if not out.get("final_text"):
            retried = True
            time.sleep(20)
            out = drive(q, run_dir, sandbox)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    dump = daemon_dump(out["session_id"]) if out.get("session_id") else {}
    json.dump(dump, open(os.path.join(run_dir, "daemon_dump.json"), "w"), indent=1)

    pred, had_answer_line = extract_answer(out["final_text"])
    em, f1 = score(pred, q["answer"])
    tool_names = [t.get("name") for t in out["tool_uses"]]
    escapes = outside_sandbox_refs(out["tool_uses"], sandbox)
    injections = dump.get("injections") or []
    daemon_spent = dump.get("spent_usd")
    row = {
        "id": q["id"], "question": q["question"], "gold": q["answer"],
        "answer": pred, "had_answer_line": had_answer_line,
        "em": em, "f1": round(f1, 3),
        "session_id": out.get("session_id"),
        # money: CC = real Anthropic billing for the session; OpenCode free
        # tier bills $0, so money = the daemon's retail-priced cost of the
        # same tokens (the identical number the budget tier is computed from)
        "cost_usd": out.get("cost_usd") if out.get("cost_usd") is not None
                    else daemon_spent,
        "daemon_spent_usd": daemon_spent,
        "tool_calls": len(tool_names), "tool_names": tool_names,
        "num_turns": out.get("num_turns"),
        "injections_delivered": len(injections),
        "tiers_seen": [m.group(1) for i in injections
                       for m in [re.search(r"Tier:\s*(\w+)", i["context"] or "")] if m],
        "web_requests": out.get("web_requests", 0),
        "permission_denials": len(out.get("permission_denials") or []),
        "outside_sandbox_refs": escapes,
        "audit_clean": (not any(n in DANGER for n in tool_names)
                        and out.get("web_requests", 0) == 0
                        and not escapes),
        "timed_out": out.get("timed_out", False),
        "retried": retried,
        # a run that ended with no model text at all is a platform failure
        # (provider error / silent truncation), not a graded answer — exclude
        # from the paired analysis, count it in reliability stats instead
        "failed": not out.get("final_text"),
    }
    json.dump(row, open(os.path.join(run_dir, "result.json"), "w"), indent=2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["claude", "opencode"], required=True)
    ap.add_argument("--tag", required=True, help="off | on-<label>")
    ap.add_argument("--budget", type=float, default=0.0, help="0 = OFF (no injection)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--data", default="data/eval.json")
    ap.add_argument("--n", type=int, default=0, help="limit #questions (0 = all)")
    args = ap.parse_args()

    questions = json.load(open(os.path.join(HERE, args.data)))
    if args.n:
        questions = questions[: args.n]

    inject = args.budget > 0
    set_condition(inject, args.budget)
    if not restart_daemon():
        raise SystemExit("daemon failed to start")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = os.path.join(HERE, "runs", args.platform)
    meta = {"platform": args.platform, "tag": args.tag, "budget": args.budget,
            "inject": inject, "model": CC_MODEL if args.platform == "claude" else OC_MODEL,
            "data": args.data, "questions": [q["id"] for q in questions],
            "seeds": args.seeds, "started": stamp,
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                      capture_output=True, text=True).stdout.strip(),
            "daemon_config": json.load(open(CONFIG))}

    all_rows = []
    for seed in range(args.seeds):
        arm_dir = os.path.join(sweep_dir, f"{args.tag}-s{seed}")
        os.makedirs(arm_dir, exist_ok=True)
        json.dump(meta, open(os.path.join(arm_dir, "meta.json"), "w"), indent=2)
        results_path = os.path.join(arm_dir, "results.jsonl")
        with open(results_path, "a") as f:
            done = set()
            if os.path.getsize(results_path) > 0:
                # resume skips only questions with a SUCCESSFUL prior row —
                # failed rows (silent truncation etc.) get re-attempted
                done = {r["id"] for l in open(results_path) if l.strip()
                        for r in [json.loads(l)] if not r.get("failed")}
            for q in questions:
                if q["id"] in done:
                    continue  # resumable
                try:
                    row = run_question(args.platform, q,
                                       os.path.join(arm_dir, q["id"]))
                except Exception as e:
                    row = {"id": q["id"], "gold": q["answer"], "em": 0.0, "f1": 0.0,
                           "answer": f"(ERROR: {str(e)[:100]})", "failed": True,
                           "cost_usd": 0.0, "tool_calls": 0}
                row["seed"] = seed
                all_rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"[{args.platform}/{args.tag}-s{seed}] {row['id']} "
                      f"em={row['em']} f1={row['f1']} cost={row.get('cost_usd')} "
                      f"tools={row.get('tool_calls')} inj={row.get('injections_delivered')} "
                      f"clean={row.get('audit_clean')} to={row.get('timed_out')} "
                      f"ans={str(row['answer'])[:40]!r}", flush=True)

    ok = [r for r in all_rows if not r.get("failed")]
    if ok:
        n = len(ok)
        print(f"\n=== {args.platform}/{args.tag}: n={n} "
              f"mean_cost={sum(r.get('cost_usd') or 0 for r in ok)/n:.4f} "
              f"mean_tools={sum(r.get('tool_calls', 0) for r in ok)/n:.2f} "
              f"mean_f1={sum(r['f1'] for r in ok)/n:.3f} "
              f"mean_em={sum(r['em'] for r in ok)/n:.3f} "
              f"clean={all(r.get('audit_clean', True) for r in ok)}")


if __name__ == "__main__":
    main()
