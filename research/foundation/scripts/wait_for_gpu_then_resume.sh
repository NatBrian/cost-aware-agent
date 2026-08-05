#!/usr/bin/env bash
# Wait for a STABLE free-GPU window, then restart the judge and resume T3.
#
# Usage: bash scripts/wait_for_gpu_then_resume.sh
#
# Context: on 2026-08-05 every card on the box was cleared and user `yongyue`
# immediately claimed all 8 with an 8-way tensor-parallel job (133G each). The
# project rule is absolute -- never kill a GPU occupier's processes, and never
# race or co-locate mid-load. So this waits.
#
# STABILITY REQUIREMENT: two consecutive clear checks, 10 minutes apart. A single
# sample can catch the gap between one job exiting and the next loading, and
# grabbing memory in that window is exactly the race the rule forbids.
#
# Needs TWO cards: the judge is ~55G and the executor ~120G, and putting them on
# one card OOMs (learned 2026-07-30).
set -uo pipefail
cd "$(dirname "$0")/.."
NEED_GB=110          # per card, free
STABLE=2             # consecutive clear checks required
INTERVAL=600         # seconds between checks

clear_cards() {      # count cards with >= NEED_GB free
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
    | awk -v need="$NEED_GB" '{ if (($2-$3)/1024 >= need) n++ } END { print n+0 }'
}

echo "[wait] need 2 cards with >= ${NEED_GB}G free, stable across $STABLE checks ${INTERVAL}s apart"
ok=0
while true; do
  n=$(clear_cards)
  if [ "$n" -ge 2 ]; then
    ok=$((ok + 1))
    echo "[wait] $(date -u +%H:%M) $n cards free (${ok}/${STABLE} stable checks)"
  else
    [ "$ok" -gt 0 ] && echo "[wait] $(date -u +%H:%M) window closed ($n free) — resetting"
    ok=0
  fi
  [ "$ok" -ge "$STABLE" ] && break
  sleep "$INTERVAL"
done

echo "[wait] stable window confirmed — restarting judge"
pgrep -u "$(whoami)" -f "[s]erve_retrieval.py" >/dev/null || {
  echo "[wait] retrieval also down; restarting it"
  CUDA_VISIBLE_DEVICES=0 nohup bash scripts/serve_retrieval.sh > experiments/results/serve_retrieval.log 2>&1 &
  sleep 60
}
nohup bash scripts/serve_judge_local.sh > experiments/results/serve_judge.log 2>&1 &

for i in $(seq 1 120); do
  curl -s -m 10 http://127.0.0.1:6101/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3.6-27B","messages":[{"role":"user","content":"ok"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    2>/dev/null | grep -q '"content"' && { echo "[wait] judge UP"; break; }
  [ "$i" = 120 ] && { echo "[wait] judge failed to start even with a free window"; exit 1; }
  sleep 20
done

echo "[wait] resuming T3 (restartable: seed 123 complete, seed 789 resumes from its shards)"
setsid nohup bash scripts/t3_seeds_run.sh >> experiments/results/t3_run.log 2>&1 &
sleep 5
for p in $(ls /proc | grep -E '^[0-9]+$'); do
  [ -r /proc/$p/cmdline ] || continue
  [ "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | sed 's/ *$//')" = "bash scripts/t3_seeds_run.sh" ] && echo "$p" > .t3.pid
done
echo "[wait] T3 resumed, pid $(cat .t3.pid 2>/dev/null)"
