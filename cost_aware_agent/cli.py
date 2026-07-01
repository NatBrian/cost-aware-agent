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

case "$EVENT" in
  SessionStart) URL="$DAEMON/session/start" ;;
  PreToolUse)   URL="$DAEMON/tool/pre" ;;
  PostToolUse)  URL="$DAEMON/tool/post" ;;
  Stop)         URL="$DAEMON/session/stop" ;;
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
    transcript_path: $tp
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

CLAUDE_HOOK_EVENTS = ["SessionStart", "PreToolUse", "PostToolUse", "Stop"]


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


def cmd_status(args):
    try:
        with urllib.request.urlopen(f"{DAEMON_URL}/status/{args.session_id}", timeout=2) as resp:
            print(json.dumps(json.loads(resp.read()), indent=2))
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
