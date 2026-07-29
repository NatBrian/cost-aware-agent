#!/usr/bin/env bash
# Wait for the remote judge to come back, then (re)launch a λ arm.
#
# Usage: bash scripts/wait_for_judge_then_run.sh <train_lambda> <tag>
#
# WHY: the judge is a shared server we do not control (Qwen3.6-27B at
# 122.11.227.227:6101). It went down mid-round on 2026-07-29 and every judge call
# returned the neutral 0.5 fallback — which is *correct* client behaviour (never
# crash the reward pipeline) but silently removes the per-step reward signal. A
# round trained through an outage is not the experimental condition; it is a
# terminal-reward-only run wearing the arm's label.
#
# The neutral fallback is deliberately NOT cached (audit 2026-07-28), so once the
# server returns, re-running re-queries exactly the failed steps and reuses the
# good ones. Recovery costs training compute, not judgements.
#
# Readiness = a real judged completion, not just /v1/models: the API can answer
# while the engine is dead.
set -uo pipefail
cd "$(dirname "$0")/.."
LAM=${1:?usage: wait_for_judge_then_run.sh <train_lambda> <tag>}
TAG=${2:?usage: wait_for_judge_then_run.sh <train_lambda> <tag>}
EP=$(grep -oP '^\s*endpoint: \Khttp://[^ ]+' configs/foundation.yaml | grep 6101 | head -1)
EP=${EP:-http://122.11.227.227:6101/v1}
MODEL=$(grep -oP '^\s*model: \KQwen3\.6-27B' configs/foundation.yaml | head -1)
MODEL=${MODEL:-Qwen3.6-27B}
LOG=/mnt/src/liangsheng/cassi_foundation/judge_wait.log

echo "[$(date +%H:%M)] waiting for judge $MODEL at $EP" | tee -a "$LOG"
for i in $(seq 1 2880); do          # up to ~48h at 60s
  OK=$(timeout 25 curl -s -m 20 "$EP/chat/completions" -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with only: OK\"}],\"max_tokens\":8,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
        2>/dev/null | grep -c '"content"' || true)
  if [ "${OK:-0}" -ge 1 ]; then
    echo "[$(date +%H:%M)] judge BACK after $((i)) checks — relaunching arm $TAG" | tee -a "$LOG"
    exec bash scripts/run_lambda_arm.sh "$LAM" "$TAG"
  fi
  [ $((i % 15)) -eq 0 ] && echo "[$(date +%H:%M)] still down (check $i)" | tee -a "$LOG"
  sleep 60
done
echo "[$(date +%H:%M)] GAVE UP after 48h — judge still down" | tee -a "$LOG"
exit 1
