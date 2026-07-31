#!/usr/bin/env bash
# S0 — re-score the EXISTING λ arms at BINDING budgets with the wasted-spend
# estimand. Eval only: no training, no new checkpoints.
#
# Why this runs before anything else (plan v2.2 §8): FOUNDATION-1 evaluated at
# {2,4,8} and gated at B=4, where the budget was slack for 67% of episodes and
# irrelevant to 94% at B=8. The one binding budget, B=2, is the only place the
# cost term ever moved behaviour. If pricing works when the budget binds, the
# already-trained checkpoints should show it at {2,3,4} under a metric that looks
# at the right behaviour — for the cost of an evaluation instead of a week of
# training.
#
# val-50 ONLY. dev-200 is not touched: FOUNDATION-1 left 1 of 3 looks and
# FOUNDATION-2 opens a fresh ledger; a diagnostic never spends a look.
#
# Restartable: an arm whose three JSONLs already exist is skipped, so a killed
# run resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=experiments/results/s0_rescore
mkdir -p "$OUT"

# Second held card: the first runs the retrieval server's E5 encoder.
GPU=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold | cut -d, -f2)
BUDGETS="small medium large"          # -> 2, 3, 4 from configs/foundation.yaml

declare -A ARMS=(
  [lam0]="$PWD/experiments/results/train/lam0_round3/checkpoint"
  [lam03]="$PWD/experiments/results/train/round3/checkpoint"
  [lam10]="$PWD/experiments/results/train/lam10_round3/checkpoint"
)

stop_executor() {
  # ONLY our executor on ONLY its port. A bare "vllm serve" pattern has killed
  # the judge before, and an unscoped pgrep once nearly killed another user's
  # job. Port + user, never anything looser.
  local pid
  pid=$(pgrep -u "$(whoami)" -f "[v]llm serve.*port 8378" | head -1) || true
  [ -n "${pid:-}" ] && { kill "$pid"; sleep 20; } || true
}
trap stop_executor EXIT

for TAG in lam0 lam03 lam10; do
  CKPT="${ARMS[$TAG]}"
  if [ ! -f "$CKPT/config.json" ]; then
    echo "!! $TAG checkpoint missing ($CKPT) — skipping"; continue
  fi
  DONE=1
  for B in $BUDGETS; do [ -s "$OUT/${TAG}_${B}.jsonl" ] || DONE=0; done
  if [ "$DONE" = 1 ]; then echo "== $TAG already complete — skipping"; continue; fi

  echo "=== serving $TAG on GPU ${GPU:-1} ==="
  stop_executor
  CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv-gpu3/bin/vllm serve "$CKPT" \
    --served-model-name Qwen/Qwen3.5-9B --port 8378 --dtype auto \
    --max-model-len 16384 --gpu-memory-utilization 0.85 \
    > "$OUT/serve_$TAG.log" 2>&1 &

  READY=0
  for i in $(seq 1 80); do
    # a REAL completion: /v1/models answers even when the engine core is dead
    if curl -s -m 10 http://127.0.0.1:8378/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
         2>/dev/null | grep -q '"content"'; then READY=1; echo "$TAG serving"; break; fi
    sleep 15
  done
  [ "$READY" = 1 ] || { echo "!! $TAG failed to serve"; tail -30 "$OUT/serve_$TAG.log"; exit 1; }

  for B in $BUDGETS; do
    [ -s "$OUT/${TAG}_${B}.jsonl" ] && { echo "-- $TAG @ $B done"; continue; }
    echo "--- $TAG @ B=$B ---"
    $PY -m collect.run_collection --task-file data/hotpotqa_val_50.jsonl \
        --arm a3 --mode none --budget "$B" --g 1 --temperature 0 \
        --out "$OUT/${TAG}_${B}.jsonl.part"
    mv "$OUT/${TAG}_${B}.jsonl.part" "$OUT/${TAG}_${B}.jsonl"
  done
done

stop_executor
echo "=== S0 COLLECTION COMPLETE ==="
$PY scripts/s0_analyse.py --dir "$OUT"
