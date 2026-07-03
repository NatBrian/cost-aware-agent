#!/bin/bash
# Instrumented cost-aware-agent Claude Code hook for the E2E feature-verification
# experiment. Mirrors the PRODUCTION hook (cost_aware_agent/cli.py HOOK_SCRIPT,
# post-audit: Stop -> /turn/end, SessionEnd -> /session/stop, project_dir + source
# fields) PLUS appends a JSONL record of every fire (event, exact body POSTed,
# exact additionalContext returned) to $CAA_HOOK_LOG so injection delivery is
# auditable from the hook side, independent of Claude Code's own transcript.
DAEMON="http://127.0.0.1:7331"
LOG="${CAA_HOOK_LOG:-/tmp/caa_hook.log}"
INPUT=$(cat)
EVENT=$(echo "$INPUT" | jq -r '.hook_event_name')
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // empty')
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')

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

BODY=$(echo "$INPUT" | jq \
  --arg tp "$TRANSCRIPT" --arg sid "$SESSION_ID" --arg cli "claude-code" \
  --arg tool "$(echo "$INPUT" | jq -r '.tool_name // empty')" \
  '{session_id:$sid, cli:$cli, task:(.prompt // ""), model:(.model // ""),
    tool_name:$tool, tool_input:(.tool_input // {}),
    tool_result:((.tool_response // .tool_result // "") | tostring),
    transcript_path:$tp,
    project_dir:(.cwd // ""),
    source:(.source // "")}')

RESP=$(curl -sf -m 5 -X POST "$URL" -H "Content-Type: application/json" -d "$BODY" 2>/dev/null)
CONTEXT=$(echo "$RESP" | jq -r '.additionalContext // empty')

# Full audit record of this fire (one JSON line).
jq -nc --arg event "$EVENT" --arg sid "$SESSION_ID" \
  --arg tool "$(echo "$INPUT" | jq -r '.tool_name // empty')" \
  --arg ctx "$CONTEXT" --argjson body "$BODY" \
  '{ts:now, event:$event, session_id:$sid, tool:$tool,
    injected:(if $ctx=="" then null else $ctx end), injected_null:($ctx==""),
    body:$body}' >> "$LOG"

if [ -n "$CONTEXT" ]; then
  jq -n --arg event "$EVENT" --arg ctx "$CONTEXT" \
    '{hookSpecificOutput:{hookEventName:$event, additionalContext:$ctx}}'
fi
exit 0
