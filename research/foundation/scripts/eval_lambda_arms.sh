#!/usr/bin/env bash
# Evaluate every λ arm on the VALIDATION slice and emit the rows the
# pre-registered rule reads.
#
# Usage: bash scripts/eval_lambda_arms.sh
#
# val-50, budgets {2,4,8}, temperature 0, harness OFF. dev-200 is NOT touched —
# one of the three permitted looks remains and it is reserved for a final
# headline method, not a diagnostic.
#
# Every arm is scored later on ONE fixed yardstick (economy.lambda = 0.3) plus the
# λ-independent raw metrics (mean steps, F1, self-stop). Whichever λ an arm was
# TRAINED with is irrelevant to how it is measured — that separation is what makes
# the arms comparable at all.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=experiments/results/lambda_eval
mkdir -p "$OUT"
GPU=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold | cut -d, -f2)

# tag -> checkpoint. λ=0.3 is the ORIGINAL foundation run's round-3 checkpoint.
declare -A ARMS=(
  [lam0]="$PWD/experiments/results/train/lam0_round3/checkpoint"
  [lam03]="$PWD/experiments/results/train/round3/checkpoint"
  [lam10]="$PWD/experiments/results/train/lam10_round3/checkpoint"
)

for TAG in lam0 lam03 lam10; do
  CKPT="${ARMS[$TAG]}"
  if [ ! -f "$CKPT/config.json" ]; then
    echo "!! $TAG checkpoint missing ($CKPT) — skipping, will show as absent in the analysis"
    continue
  fi
  echo "=== serving $TAG ==="
  # ONLY the executor may be stopped. A bare "vllm serve" pattern would match the
  # judge on GPU 0 and has killed it before.
  PID=$(pgrep -u "$(whoami)" -f "[v]llm serve.*port 8378" | head -1) && [ -n "$PID" ] && kill "$PID" && sleep 20 || true
  CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv-gpu3/bin/vllm serve "$CKPT" \
    --served-model-name Qwen/Qwen3.5-9B --port 8378 --dtype auto \
    --max-model-len 16384 --gpu-memory-utilization 0.85 \
    > "$OUT/serve_$TAG.log" 2>&1 &
  for i in $(seq 1 80); do
    # a REAL completion; /v1/models answers even when the engine core is dead
    curl -s -m 10 http://127.0.0.1:8378/v1/chat/completions -H "Content-Type: application/json" \
      -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
      2>/dev/null | grep -q '"content"' && { echo "$TAG serving"; break; }
    sleep 15
    [ "$i" = 80 ] && { echo "!! $TAG failed to serve"; exit 1; }
  done

  for B in small medium large; do
    echo "--- $TAG @ B=$B ---"
    rm -f "$OUT/${TAG}_${B}.jsonl"
    $PY -m collect.run_collection --task-file data/hotpotqa_val_50.jsonl \
        --arm a3 --mode none --budget "$B" --g 1 --temperature 0 \
        --out "$OUT/${TAG}_${B}.jsonl"
  done
done

echo
echo "=== APPLYING THE PRE-REGISTERED RULE ==="
$PY scripts/analyse_lambda_ablation.py --dir "$OUT"
