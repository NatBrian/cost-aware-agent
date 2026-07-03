"""`cost-aware-agent` CLI. Subcommands: daemon start/stop/status, install, status."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from cost_aware_agent.config import HOME_DIR

DAEMON_URL = "http://127.0.0.1:7331"
PID_FILE = HOME_DIR / "daemon.pid"
LOG_FILE = HOME_DIR / "daemon.log"
HOOK_SCRIPT_PATH = HOME_DIR / "hooks" / "claude-code.sh"
OPENCODE_PLUGIN_SRC = Path(__file__).parent / "data" / "opencode-plugin.ts"

HOOK_SCRIPT = f"""#!/bin/bash
# cost-aware-agent Claude Code adapter hook. Installed by `cost-aware-agent install --for claude-code`.
DAEMON="{DAEMON_URL}"
INPUT=$(cat)
EVENT=$(echo "$INPUT" | jq -r '.hook_event_name')
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // empty')
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')

if ! curl -sf -m 1 "$DAEMON/health" > /dev/null 2>&1; then
  cost-aware-agent daemon start > /dev/null 2>&1 &
  sleep 0.5
fi

# Stop fires after EVERY assistant response (NOT session end) -> /turn/end.
# SessionEnd is the real end-of-session event -> /session/stop (receipt).
case "$EVENT" in
  SessionStart) URL="$DAEMON/session/start" ;;
  PreToolUse)   URL="$DAEMON/tool/pre" ;;
  PostToolUse)  URL="$DAEMON/tool/post" ;;
  Stop)         URL="$DAEMON/turn/end" ;;
  SessionEnd)   URL="$DAEMON/session/stop" ;;
  *)            exit 0 ;;
esac

BODY=$(echo "$INPUT" | jq \\
  --arg tp "$TRANSCRIPT" \\
  --arg sid "$SESSION_ID" \\
  --arg cli "claude-code" \\
  --arg tool "$(echo "$INPUT" | jq -r '.tool_name // empty')" \\
  '{{
    session_id: $sid,
    cli: $cli,
    task: (.prompt // ""),
    model: (.model // ""),
    tool_name: $tool,
    tool_input: (.tool_input // {{}}),
    tool_result: ((.tool_response // .tool_result // "") | tostring),
    transcript_path: $tp,
    project_dir: (.cwd // ""),
    source: (.source // "")
  }}')

CONTEXT=$(curl -sf -m 5 -X POST "$URL" -H "Content-Type: application/json" -d "$BODY" 2>/dev/null | \\
  jq -r '.additionalContext // empty')

# additionalContext must be wrapped in hookSpecificOutput with hookEventName, or
# Claude Code silently ignores it (confirmed against code.claude.com/docs/en/hooks.md
# 2026-07-01 — a flat {{"additionalContext": ...}} at top level is NOT read by the model).
if [ -n "$CONTEXT" ]; then
  jq -n --arg event "$EVENT" --arg ctx "$CONTEXT" \\
    '{{hookSpecificOutput: {{hookEventName: $event, additionalContext: $ctx}}}}'
fi
exit 0
"""

CLAUDE_HOOK_EVENTS = ["SessionStart", "PreToolUse", "PostToolUse", "Stop", "SessionEnd"]


def _daemon_healthy() -> bool:
    try:
        urllib.request.urlopen(f"{DAEMON_URL}/health", timeout=1)
        return True
    except Exception:
        return False


def cmd_daemon_start(args):
    if _daemon_healthy():
        print("daemon already running")
        return
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "a")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "cost_aware_agent.daemon:app",
         "--host", "127.0.0.1", "--port", "7331"],
        stdout=log, stderr=log, start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    for _ in range(20):
        if _daemon_healthy():
            print(f"daemon started, pid {proc.pid}")
            return
        time.sleep(0.25)
    print(f"daemon spawned (pid {proc.pid}) but health check did not pass — see {LOG_FILE}", file=sys.stderr)


def cmd_daemon_stop(args):
    if not PID_FILE.exists():
        print("no pidfile, nothing to stop")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, 15)
        print(f"sent SIGTERM to pid {pid}")
    except ProcessLookupError:
        print("process already gone")
    PID_FILE.unlink(missing_ok=True)


def cmd_daemon_status(args):
    print("running" if _daemon_healthy() else "not running")


def _merge_hooks_into_settings(settings_path: Path) -> None:
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
    else:
        settings = {}
    hooks = settings.setdefault("hooks", {})
    command = str(HOOK_SCRIPT_PATH)
    for event in CLAUDE_HOOK_EVENTS:
        entries = hooks.setdefault(event, [])
        already = any(
            h.get("command") == command
            for entry in entries for h in entry.get("hooks", [])
        )
        if not already:
            entries.append({"hooks": [{"type": "command", "command": command}]})
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def cmd_install(args):
    if args.for_cli == "claude-code":
        HOOK_SCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        HOOK_SCRIPT_PATH.write_text(HOOK_SCRIPT)
        HOOK_SCRIPT_PATH.chmod(0o755)

        if args.global_install:
            settings_path = Path.home() / ".claude" / "settings.json"
        else:
            settings_path = Path(args.project_dir) / ".claude" / "settings.json"
        _merge_hooks_into_settings(settings_path)
        print(f"installed hook script at {HOOK_SCRIPT_PATH}")
        print(f"merged hooks into {settings_path}")
    elif args.for_cli == "opencode":
        if args.global_install:
            dest_dir = Path.home() / ".config" / "opencode" / "plugin"
        else:
            dest_dir = Path(args.project_dir) / ".opencode" / "plugins"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "cost-aware-agent.ts"
        dest.write_text(OPENCODE_PLUGIN_SRC.read_text())
        print(f"installed plugin at {dest}")
    else:
        print(f"unsupported --for value: {args.for_cli} (only claude-code, opencode are implemented)", file=sys.stderr)
        sys.exit(1)


def cmd_budget_set(args):
    import math

    from cost_aware_agent import db
    if not math.isfinite(args.amount) or args.amount <= 0:
        # zero/negative/NaN wallet = permanent no-pressure tier; refuse loudly
        # instead of silently disarming the budget
        print(f"invalid budget amount: {args.amount} (must be a positive dollar figure)",
              file=sys.stderr)
        sys.exit(1)
    # the daemon normalizes the hook's cwd the same way — both writers MUST
    # agree or one project silently splits into two wallets
    project_dir = db.normalize_project_dir(args.project_dir)
    db.init_db()
    with db.get_conn() as conn:
        db.set_wallet(conn, project_dir, args.amount)
    print(f"wallet set: ${args.amount:.2f} for {project_dir}")
    print("depletes across all sessions in this project until exhausted (advisory only)")


def cmd_budget_show(args):
    from cost_aware_agent import db
    project_dir = db.normalize_project_dir(args.project_dir)
    db.init_db()
    with db.get_conn() as conn:
        wallet = db.get_wallet(conn, project_dir)
        if wallet is None:
            print(f"no wallet for {project_dir} — set one with: cost-aware-agent budget set <amount>")
            return
        spent = db.wallet_spent_usd(conn, project_dir)
    budget = wallet["budget_usd"]
    print(f"project:   {project_dir}")
    print(f"budget:    ${budget:.2f}")
    print(f"spent:     ${spent:.4f}")
    print(f"remaining: ${max(0.0, budget - spent):.4f}")


def cmd_init(args):
    # "one number from the user, maximum": init IS budget-set (plus a hint to
    # install the adapter if that hasn't happened yet)
    cmd_budget_set(args)
    settings = Path(args.project_dir) / ".claude" / "settings.json"
    if not settings.exists() and not (Path(args.project_dir) / ".opencode").exists():
        print("hint: no adapter detected in this project — run "
              "`cost-aware-agent install --for claude-code --project-dir .`")


def cmd_status(args):
    try:
        with urllib.request.urlopen(f"{DAEMON_URL}/status/{args.session_id}", timeout=2) as resp:
            print(json.dumps(json.loads(resp.read()), indent=2))
    except urllib.error.URLError as e:
        print(f"could not reach daemon: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_receipt(args):
    from cost_aware_agent.prompts import render_receipt
    try:
        with urllib.request.urlopen(f"{DAEMON_URL}/session/{args.session_id}/dump", timeout=2) as resp:
            print(render_receipt(json.loads(resp.read())))
    except urllib.error.URLError as e:
        print(f"could not reach daemon: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="cost-aware-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    daemon_p = sub.add_parser("daemon")
    daemon_sub = daemon_p.add_subparsers(dest="daemon_command", required=True)
    daemon_sub.add_parser("start").set_defaults(func=cmd_daemon_start)
    daemon_sub.add_parser("stop").set_defaults(func=cmd_daemon_stop)
    daemon_sub.add_parser("status").set_defaults(func=cmd_daemon_status)

    install_p = sub.add_parser("install")
    install_p.add_argument("--for", dest="for_cli", required=True)
    install_p.add_argument("--project-dir", default=".")
    install_p.add_argument("--global", dest="global_install", action="store_true")
    install_p.set_defaults(func=cmd_install)

    status_p = sub.add_parser("status")
    status_p.add_argument("session_id")
    status_p.set_defaults(func=cmd_status)

    receipt_p = sub.add_parser("receipt", help="human-readable cost receipt for a session")
    receipt_p.add_argument("session_id")
    receipt_p.set_defaults(func=cmd_receipt)

    init_p = sub.add_parser("init", help="set the project's dollar budget (one number)")
    init_p.add_argument("--budget", dest="amount", type=float, required=True)
    init_p.add_argument("--project-dir", default=".")
    init_p.set_defaults(func=cmd_init)

    budget_p = sub.add_parser("budget")
    budget_sub = budget_p.add_subparsers(dest="budget_command", required=True)
    budget_set_p = budget_sub.add_parser("set")
    budget_set_p.add_argument("amount", type=float)
    budget_set_p.add_argument("--project-dir", default=".")
    budget_set_p.set_defaults(func=cmd_budget_set)
    budget_show_p = budget_sub.add_parser("show")
    budget_show_p.add_argument("--project-dir", default=".")
    budget_show_p.set_defaults(func=cmd_budget_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
