#!/usr/bin/env python3
"""SWE-bench Lite A/B — does the money budget reduce cost on real coding tasks?

The prior A/B (../real_cli, HotpotQA) answered "no savings on low-slack
retrieval": ~4 required tool calls per question leaves nothing to trim. This
experiment moves to a REAL dataset with slack: SWE-bench Lite instances (real
GitHub issues, graded by the dataset's own FAIL_TO_PASS/PASS_TO_PASS tests).
Open-ended debugging has genuine discretionary spend — exploration breadth,
verification loops, "one more check" — which is exactly where the earlier
experiments (q5, cc_adapter overview task) showed advisory budgets act.

Arms (identical prompts, identical sandboxes; the ONLY difference is daemon
config `inject_enabled` + a project wallet):

  OFF  inject_enabled=false — daemon still measures every dollar, injects nothing
  ON   inject_enabled=true (on_change policy) + project wallet set through the
       real CLI (`cost-aware-agent budget set <w> --project-dir <proj>`),
       wallet = WALLET_FRACTION x median OFF cost of the SAME instance
       (rule pre-registered here, before any ON run)

Platforms: Claude Code (sonnet, production hook channel) and OpenCode
(deepseek-v4-flash-free, production plugin channel; money = daemon retail).

Sandbox per run: fresh checkout of the repo at base_commit with HISTORY
STRIPPED (.git removed, re-init, single import commit — a full clone would
contain the gold fix in future commits), plus .venv with the project installed
editable. Gold patch and test_patch never enter the sandbox.

Grading (harness-side, after the agent exits — SWE-bench semantics):
  agent_patch = git diff of the sandbox
  test files touched by test_patch are restored to base, test_patch applied,
  then FAIL_TO_PASS + a 5-test PASS_TO_PASS sample run in the sandbox venv.
  success = all pass.

Usage:
  run_swe.py --arm off --seeds 2                # both platforms
  run_swe.py --arm on  --seeds 2                # requires OFF results (wallet rule)
  run_swe.py --arm off --platforms claude --instances pytest-dev__pytest-11143 --seeds 1
"""
import argparse
import datetime
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "real_cli"))
import run_real  # run_killable, parsers, restart_daemon, daemon_dump

HOOK = os.path.join(HERE, "hook.sh")
PLUGIN_SRC = os.path.join(REPO, "cost_aware_agent", "data", "opencode-plugin.ts")
CONFIG = os.path.expanduser("~/.cost-aware-agent/config.json")
CC_MODEL = "sonnet"
OC_MODEL = "opencode/deepseek-v4-flash-free"

CC_ALLOWED = "Read Grep Glob Bash Write Edit"
CC_DENIED = ["WebSearch", "WebFetch", "Task", "Agent", "NotebookEdit", "TodoWrite"]

WALLET_FRACTION = 0.5     # pre-registered: wallet = 0.5 x per-instance OFF median
P2P_SAMPLE = 3            # regression sample (first N — the same N screening verified)
MAX_CC_SPEND_USD = 25.0   # hard safety backstop (OFF+ON ~ $18 expected); this
                          # is a runaway guard, not a per-arm budget
TIMEOUT_S = 900       # Claude Code
OC_TIMEOUT_S = 480    # OpenCode/deepseek loops on hard tasks — fail faster

# --- network egress block for the AGENT subprocess only (venv built before the
# agent runs, so its editable install is unaffected). Closes a real hole found
# in the first run: bash `pip download pytest==8.0.0` reached PyPI and one run
# read the released (fixed) source. WebSearch/WebFetch tool denial does not
# cover bash.
#
# Two policies, because the two CLIs have different infra needs:
#  - Claude Code only phones api.anthropic.com, so a black-hole proxy with
#    Anthropic whitelisted breaks curl/wget/pip/git while the CLI still runs.
#  - OpenCode fetches models.dev, its zen gateway, auth etc. at startup through
#    the SAME proxy env, and NO_PROXY host-matching is unreliable across its
#    node HTTP stack — a black-hole proxy hangs it at startup (verified: 900s
#    timeout, zero output). So for OpenCode use NO proxy and block the actual
#    demonstrated vector — pip — with PIP_NO_INDEX (needs no proxy). Any other
#    bash egress (curl/git) is caught by the audit's net_refs and invalidated
#    post-hoc, exactly as the CC contamination was.
NET_BLOCK_CC = {
    "HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1",
    "http_proxy": "http://127.0.0.1:1", "https_proxy": "http://127.0.0.1:1",
    "PIP_NO_INDEX": "1", "PIP_INDEX_URL": "http://127.0.0.1:1/simple",
    "no_proxy": ".anthropic.com,anthropic.com,127.0.0.1,localhost",
    "NO_PROXY": ".anthropic.com,anthropic.com,127.0.0.1,localhost",
}
NET_BLOCK_OC = {
    "PIP_NO_INDEX": "1", "PIP_INDEX_URL": "http://127.0.0.1:1/simple",
}

REPO_CACHE = os.environ.get(
    "SWE_REPO_CACHE",
    "/tmp/claude-1006/-home-liangsheng-brian-agent-budget/"
    "f8171908-94d7-4f46-80af-e3a4917be3a9/scratchpad/repos")

EXTRA_DEPS = {
    "pytest-dev/pytest": [],
    "pylint-dev/pylint": ["pytest==7.4.4", "pytest-timeout", "typing-extensions"],
    "sympy/sympy": ["pytest==7.4.4"],
    "sphinx-doc/sphinx": [],
}

TASK_TMPL = """You are working in a Python repository checkout. Fix the following real issue from the project's tracker.

<issue>
{issue}
</issue>

Rules:
- The fix belongs in the repository source code. Do NOT modify test files.
- A virtualenv with this project installed in editable mode is at .venv/ — run tests with `.venv/bin/python -m pytest <path>`.
- Do not use the network. Work only inside this directory.
- When you consider the issue fixed and verified, stop and reply with a final line: DONE
"""

# tool-input strings that would indicate tampering with the harness or
# reaching for the network / the gold fix
CHEAT_RE = re.compile(
    r"cost-aware-agent|cost_aware_agent|127\.0\.0\.1:7331|localhost:7331|"
    r"db\.sqlite|budget set|\.cost-aware-agent", re.I)
NET_RE = re.compile(
    r"\b(curl|wget|pip3? install(?! -e \.)|pip3? download|git fetch|git pull|"
    r"git clone|git remote|nc |ssh |scp )|https?://", re.I)


def sh(cmd, cwd=None, timeout=1800, check=False):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       timeout=timeout, shell=isinstance(cmd, str))
    if check and p.returncode != 0:
        raise RuntimeError(f"{cmd}: {p.stderr[-1000:]}")
    return p


def set_inject(enabled: bool):
    c = json.load(open(CONFIG))
    c["inject_enabled"] = bool(enabled)
    c["inject_mode"] = "on_change"
    json.dump(c, open(CONFIG, "w"), indent=2)


# --- sandbox ----------------------------------------------------------------

def make_sandbox(inst, proj):
    """Fresh checkout at base_commit, history stripped, venv installed."""
    shutil.rmtree(proj, ignore_errors=True)
    os.makedirs(os.path.dirname(proj), exist_ok=True)
    cache = os.path.join(REPO_CACHE, inst["repo"].replace("/", "__"))
    sh(["git", "clone", "--local", "--no-hardlinks", cache, proj], check=True)
    sh(["git", "checkout", "-q", inst["base_commit"]], cwd=proj, check=True)
    # venv install MUST happen while the real git history is still present:
    # setuptools-scm derives the package version from tags at build time, and
    # an -e install inside the stripped repo records 0.1.dev1 — which trips
    # pytest's own pyproject `minversion` gate at grade time (seen live).
    sh([sys.executable, "-m", "venv", os.path.join(proj, ".venv")], check=True)
    pip = os.path.join(proj, ".venv", "bin", "pip")
    sh([pip, "-q", "install", "-U", "pip", "setuptools", "wheel"], check=True)
    deps = EXTRA_DEPS.get(inst["repo"], ["pytest==7.4.4"])
    sh([pip, "-q", "install", "-e", "."] + deps, cwd=proj, check=True, timeout=1800)
    # strip history — a full clone contains the gold fix in future commits
    shutil.rmtree(os.path.join(proj, ".git"))
    sh(["git", "init", "-q"], cwd=proj, check=True)
    with open(os.path.join(proj, ".git", "info", "exclude"), "w") as f:
        f.write(".venv/\n.pytest_cache/\n__pycache__/\n*.pyc\n.opencode/\nopencode.json\n")
    sh(["git", "add", "-A"], cwd=proj, check=True)
    sh(["git", "-c", "user.email=swe@ab", "-c", "user.name=swe-ab",
        "commit", "-q", "-m", "import"], cwd=proj, check=True)
    return proj


def add_oc_config(proj):
    pl = os.path.join(proj, ".opencode", "plugins")
    os.makedirs(pl, exist_ok=True)
    shutil.copy(PLUGIN_SRC, os.path.join(pl, "cost-aware-agent.ts"))
    json.dump({
        "$schema": "https://opencode.ai/config.json",
        "tools": {"webfetch": False},
        "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
    }, open(os.path.join(proj, "opencode.json"), "w"), indent=2)


# --- grading (SWE-bench semantics) ------------------------------------------

def grade(inst, proj, run_dir):
    """Restore test files, apply test_patch, run F2P + P2P sample in the venv."""
    agent_patch = sh(["git", "diff", "HEAD"], cwd=proj).stdout
    open(os.path.join(run_dir, "agent.patch"), "w").write(agent_patch)
    tfiles = re.findall(r"^diff --git a/(\S+)", inst["test_patch"], re.M)
    existing = [t for t in tfiles if os.path.exists(os.path.join(proj, t))]
    if existing:
        sh(["git", "checkout", "HEAD", "--"] + existing, cwd=proj)
    tp = os.path.join(run_dir, "test.patch")
    open(tp, "w").write(inst["test_patch"])
    ap = sh(["git", "apply", tp], cwd=proj)
    if ap.returncode != 0:
        return {"success": False, "grade_error": "test_patch_apply_failed",
                "detail": ap.stderr[-500:], "agent_patch_lines": len(agent_patch.splitlines())}
    py = os.path.join(proj, ".venv", "bin", "python")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    def run_ids(ids, tag):
        p = subprocess.run([py, "-m", "pytest", "-q", "--no-header",
                            "-p", "no:cacheprovider"] + ids,
                           cwd=proj, capture_output=True, text=True, timeout=900, env=env)
        open(os.path.join(run_dir, f"grade_{tag}.txt"), "w").write(
            p.stdout[-8000:] + "\n--- stderr ---\n" + p.stderr[-2000:])
        return p.returncode

    try:
        rc_f2p = run_ids(inst["ids_f2p"], "f2p")
        rc_p2p = run_ids(inst["ids_p2p_sample"], "p2p")
    except subprocess.TimeoutExpired:
        return {"success": False, "grade_error": "grade_timeout",
                "agent_patch_lines": len(agent_patch.splitlines())}
    return {"success": rc_f2p == 0 and rc_p2p == 0,
            "f2p_pass": rc_f2p == 0, "p2p_pass": rc_p2p == 0,
            "agent_patch_lines": len(agent_patch.splitlines()),
            "agent_touched_tests": any(t in agent_patch for t in tfiles)}


# --- drivers ----------------------------------------------------------------

def cc_settings(run_dir):
    path = os.path.join(run_dir, "hook_settings.json")
    events = ["SessionStart", "PreToolUse", "PostToolUse", "Stop", "SessionEnd"]
    json.dump({"hooks": {e: [{"hooks": [{"type": "command", "command": HOOK}]}]
                         for e in events}}, open(path, "w"), indent=2)
    return path


def drive_claude(task, run_dir, proj):
    hook_log = os.path.join(run_dir, "hook.jsonl")
    open(hook_log, "w").close()
    env = dict(os.environ, CAA_HOOK_LOG=hook_log, PWD=proj, **NET_BLOCK_CC)
    events_path = os.path.join(run_dir, "events.jsonl")
    cmd = ["claude", "-p", task, "--model", CC_MODEL,
           "--output-format", "stream-json", "--verbose",
           "--settings", cc_settings(run_dir),
           "--allowedTools", CC_ALLOWED,
           "--disallowedTools", " ".join(CC_DENIED)]
    err = run_real.run_killable(cmd, timeout=TIMEOUT_S, cwd=proj, env=env,
                                out_path=events_path)
    tool_uses, result = run_real.parse_cc_events(events_path)
    sid = (result or {}).get("session_id")
    if not sid:
        # A killed run (timeout) never emits the final `result` event that
        # carries session_id — but every hook fire logged it. Recover it so the
        # daemon dump (which measured the spend perfectly, even mid-run) is
        # still fetchable. Verified: recovered $2.05 spend on a 900s-timeout run.
        for line in open(hook_log, errors="replace"):
            if line.strip():
                s = json.loads(line).get("session_id")
                if s:
                    sid = s
                    break
    if sid:
        import glob as g
        hits = g.glob(os.path.expanduser(f"~/.claude/projects/*/{sid}.jsonl"))
        if hits:
            shutil.copy(hits[0], os.path.join(run_dir, "transcript.jsonl"))
    return {"session_id": sid,
            "final_text": (result or {}).get("result", ""),
            "cli_cost_usd": (result or {}).get("total_cost_usd"),
            "num_turns": (result or {}).get("num_turns"),
            "tool_uses": tool_uses,
            "timed_out": err == "TIMEOUT"}


def _newest_oc_session(since_ts):
    """OpenCode buffers its --format json stdout and flushes only at exit, so a
    SIGKILL (timeout) loses events, session_id and trajectory entirely. But the
    plugin has been pushing to the daemon the whole time. Recover the session by
    taking the newest opencode session created at/after this run started."""
    try:
        import sqlite3
        db = os.path.expanduser("~/.cost-aware-agent/db.sqlite")
        c = sqlite3.connect(db)
        row = c.execute(
            "SELECT session_id FROM sessions WHERE cli='opencode' "
            "AND created_at >= ? ORDER BY created_at DESC LIMIT 1",
            (int(since_ts),)).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def drive_opencode(task, run_dir, proj):
    events_path = os.path.join(run_dir, "events.jsonl")
    cmd = ["opencode", "run", "--format", "json", "--print-logs",
           "--log-level", "ERROR", "-m", OC_MODEL, task]
    started = time.time()
    err = run_real.run_killable(cmd, timeout=OC_TIMEOUT_S, cwd=proj,
                                env=dict(os.environ, PWD=proj, **NET_BLOCK_OC),
                                out_path=events_path)
    final_text, tool_uses, sid = run_real.parse_oc_events(events_path)
    recovered = False
    if not sid:
        # killed/timed-out run: recover the session from the daemon so its
        # measured spend + tool trail + injections are not lost
        sid = _newest_oc_session(started - 5)
        recovered = bool(sid)
        if sid and err == "TIMEOUT" and not final_text:
            # a timed-out run has no model-emitted answer; mark it so grading
            # and the audit read from the recovered daemon rows, not empty text
            final_text = ""
    return {"session_id": sid, "final_text": final_text,
            "cli_cost_usd": None,  # free tier — money = daemon retail cost
            "num_turns": None, "tool_uses": tool_uses,
            "session_recovered": recovered,
            "timed_out": err == "TIMEOUT"}


# --- audit ------------------------------------------------------------------

DANGER = {"WebSearch", "WebFetch", "Task", "Agent", "webfetch"}


def audit(tool_uses, proj):
    cheats, netrefs, escapes = [], [], []
    for t in tool_uses:
        blob = json.dumps(t.get("input") or {})
        if CHEAT_RE.search(blob):
            cheats.append({"tool": t.get("name"), "snippet": blob[:200]})
        if t.get("name") in ("Bash", "bash") and NET_RE.search(blob):
            netrefs.append({"tool": t.get("name"), "snippet": blob[:200]})
        for m in re.findall(r"/(?:home|root|etc)[^\s\"']*", blob):
            if not m.startswith(proj):
                escapes.append({"tool": t.get("name"), "path": m[:120]})
    danger = sorted({t.get("name") for t in tool_uses} & DANGER)
    return {"cheat_refs": cheats, "net_refs": netrefs,
            "outside_sandbox_refs": escapes, "danger_tools_used": danger,
            "audit_clean": not (cheats or netrefs or escapes or danger)}


# --- wallets ----------------------------------------------------------------

def off_costs(platform):
    """instance -> list of OFF costs from all off-s* results files."""
    import collections
    out = collections.defaultdict(list)
    for path in sorted(
            __import__("glob").glob(os.path.join(HERE, "runs", platform, "off-s*", "results.jsonl"))):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("failed") and r.get("cost_usd"):
                out[r["iid"]].append(r["cost_usd"])
    return out


def compute_wallets():
    wallets = {}
    for platform in ("claude", "opencode"):
        costs = off_costs(platform)
        wallets[platform] = {
            iid: round(WALLET_FRACTION * statistics.median(v), 6)
            for iid, v in costs.items() if v}
    wallets["_rule"] = f"wallet = {WALLET_FRACTION} x per-instance OFF median"
    json.dump(wallets, open(os.path.join(HERE, "wallets.json"), "w"), indent=2)
    return wallets


# --- main loop ----------------------------------------------------------------

def cc_spend_so_far():
    total = 0.0
    import glob as g
    for path in g.glob(os.path.join(HERE, "runs", "claude", "*", "results.jsonl")):
        for line in open(path):
            if line.strip():
                total += json.loads(line).get("cost_usd") or 0.0
    return total


def run_one(platform, inst, arm, seed, wallet, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    proj = os.path.join(HERE, "runs", "projects",
                        f"{platform}-{inst['iid']}-{arm}-s{seed}")
    make_sandbox(inst, proj)
    if platform == "opencode":
        add_oc_config(proj)
    if wallet:
        r = sh(["cost-aware-agent", "budget", "set", str(wallet),
                "--project-dir", proj])
        open(os.path.join(run_dir, "wallet_set.txt"), "w").write(
            f"rc={r.returncode}\n{r.stdout}\n{r.stderr}")
    task = TASK_TMPL.format(issue=inst["problem_statement"])
    open(os.path.join(run_dir, "task.txt"), "w").write(task)

    drive = drive_claude if platform == "claude" else drive_opencode
    out = drive(task, run_dir, proj)

    def _no_usable_data(o):
        # a run that produced no session AND no answer captured nothing; on
        # OpenCode a transient provider "stream error" also manifests as a run
        # with a session but ZERO real work (deepseek free tier is flaky —
        # verified: health OK seconds later, injection delivered but the model
        # completion errored mid-stream). Treat both as retryable.
        if not o.get("session_id") and not o.get("final_text"):
            return True
        if platform == "opencode":
            d = run_real.daemon_dump(o["session_id"]) if o.get("session_id") else {}
            spent = d.get("spent_usd") or 0
            ntools = len(d.get("tool_calls") or [])
            if o.get("timed_out") and spent <= 0 and ntools == 0:
                return True
        return False

    if _no_usable_data(out):
        make_sandbox(inst, proj)
        if platform == "opencode":
            add_oc_config(proj)
        time.sleep(20)
        out = drive(task, run_dir, proj)

    sid = out.get("session_id")
    # close the session daemon-side before reading the dump (OpenCode has no
    # end event; CC's SessionEnd should have fired — record which)
    state = None
    if sid:
        try:
            import urllib.request
            d = json.load(urllib.request.urlopen(
                f"http://127.0.0.1:7331/session/{sid}/dump", timeout=10))
            state = (d.get("session") or {}).get("state")
        except Exception:
            pass
        if state != "ended":
            import urllib.request
            req = urllib.request.Request(
                "http://127.0.0.1:7331/session/stop",
                data=json.dumps({"session_id": sid}).encode(),
                headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=10)
            except Exception:
                pass
    dump = run_real.daemon_dump(sid) if sid else {}
    json.dump(dump, open(os.path.join(run_dir, "daemon_dump.json"), "w"), indent=1)

    g = grade(inst, proj, run_dir)
    # OpenCode's buffered stdout is unreliable (truncation/partial flush even on
    # clean exit — seen: 1 tool parsed vs 20 in the daemon), so audit + count
    # tools from the daemon dump, which the plugin pushed live and in full. This
    # is load-bearing for the cheat audit: auditing only the parsed subset could
    # miss a tool that reached the network or escaped the sandbox.
    daemon_tools = [{"name": tc.get("tool_name"), "input": tc.get("tool_input")}
                    for tc in (dump.get("tool_calls") or [])]
    if platform == "opencode" and len(daemon_tools) >= len(out["tool_uses"]):
        audit_tools = daemon_tools
    else:
        audit_tools = out["tool_uses"]
    a = audit(audit_tools, proj)
    daemon_spent = dump.get("spent_usd")
    injections = dump.get("injections") or []
    row = {
        "iid": inst["iid"], "repo": inst["repo"], "platform": platform,
        "arm": arm, "seed": seed, "wallet": wallet,
        "session_id": sid,
        "cost_usd": out.get("cli_cost_usd") if out.get("cli_cost_usd") is not None
                    else daemon_spent,
        "cli_cost_usd": out.get("cli_cost_usd"),
        "daemon_spent_usd": daemon_spent,
        "num_turns": out.get("num_turns"),
        "tool_calls": len(audit_tools),
        "tool_names": [t.get("name") for t in audit_tools],
        "tools_source": ("daemon" if audit_tools is daemon_tools else "cli"),
        "injections_delivered": len(injections),
        "tiers_seen": sorted({m.group(1) for i in injections
                              for m in [re.search(r"Tier:\s*(\w+)", i.get("context") or "")] if m}),
        "state_before_harness_close": state,
        "timed_out": out.get("timed_out", False),
        "failed": (not out.get("final_text")) and not out.get("timed_out"),
        **g, **a,
    }
    json.dump(row, open(os.path.join(run_dir, "result.json"), "w"), indent=2)
    # keep the sandbox out of the repo tree growth — the agent patch, grade
    # logs and daemon dump are the durable artifacts
    shutil.rmtree(proj, ignore_errors=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["off", "on"], required=True)
    ap.add_argument("--platforms", nargs="+", default=["claude", "opencode"],
                    choices=["claude", "opencode"])
    ap.add_argument("--instances", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "instances.json"))
    args = ap.parse_args()

    instances = json.load(open(args.data))
    if args.instances:
        instances = [i for i in instances if i["iid"] in args.instances]

    wallets = {}
    if args.arm == "on":
        wallets = compute_wallets()
        for p in args.platforms:
            missing = [i["iid"] for i in instances if i["iid"] not in wallets.get(p, {})]
            if missing:
                raise SystemExit(f"no OFF costs yet for {p}: {missing} — run --arm off first")

    set_inject(args.arm == "on")
    if not run_real.restart_daemon():
        raise SystemExit("daemon failed to restart")
    print(f"=== arm={args.arm} inject_enabled={args.arm == 'on'} "
          f"config={json.load(open(CONFIG))}", flush=True)

    meta = {"arm": args.arm, "seeds": args.seeds, "wallet_fraction": WALLET_FRACTION,
            "cc_model": CC_MODEL, "oc_model": OC_MODEL, "timeout_s": TIMEOUT_S,
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_sha": sh(["git", "rev-parse", "HEAD"], cwd=REPO).stdout.strip(),
            "wallets": wallets}

    for platform in args.platforms:
        for seed in range(args.seeds):
            arm_dir = os.path.join(HERE, "runs", platform, f"{args.arm}-s{seed}")
            os.makedirs(arm_dir, exist_ok=True)
            json.dump(meta, open(os.path.join(arm_dir, "meta.json"), "w"), indent=2)
            results_path = os.path.join(arm_dir, "results.jsonl")
            done = set()
            if os.path.exists(results_path):
                done = {r["iid"] for l in open(results_path) if l.strip()
                        for r in [json.loads(l)] if not r.get("failed")}
            with open(results_path, "a") as f:
                for inst in instances:
                    if inst["iid"] in done:
                        continue
                    if platform == "claude":
                        spent = cc_spend_so_far()
                        if spent > MAX_CC_SPEND_USD:
                            print(f"!!! CC spend guard hit (${spent:.2f} > "
                                  f"${MAX_CC_SPEND_USD}) — stopping CC runs", flush=True)
                            break
                    wallet = wallets.get(platform, {}).get(inst["iid"]) \
                        if args.arm == "on" else None
                    run_dir = os.path.join(arm_dir, inst["iid"])
                    try:
                        row = run_one(platform, inst, args.arm, seed, wallet, run_dir)
                    except Exception as e:
                        row = {"iid": inst["iid"], "platform": platform,
                               "arm": args.arm, "seed": seed, "failed": True,
                               "success": False, "cost_usd": 0.0,
                               "error": str(e)[:300]}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"[{platform}/{args.arm}-s{seed}] {inst['iid']} "
                          f"success={row.get('success')} cost={row.get('cost_usd')} "
                          f"tools={row.get('tool_calls')} inj={row.get('injections_delivered')} "
                          f"clean={row.get('audit_clean')} to={row.get('timed_out')}",
                          flush=True)
    print("ARM DONE", flush=True)


if __name__ == "__main__":
    main()
