#!/usr/bin/env bash
# One GRPO round (E-d micro / E-e full): collect -> stop server -> train ->
# restart server on new checkpoint. Usage:
#   bash scripts/e5_round.sh <round_dir> <n_tasks> <G> [max_steps]
# Server model path is read from .serve_model (defaults to config model).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROUND=$1; N=$2; G=$3; MAX=${4:-}
mkdir -p "$ROUND"

echo "== collect ($N tasks x G=$G, train mode) =="
$PY -m collect.run_collection --task-file data/hotpotqa_train_300.jsonl \
    --limit "$N" --arm a3 --mode none --budget draw --g "$G" --train \
    --out "$ROUND/rollouts.jsonl"

echo "== stop executor server (frees GPU for training) =="
pkill -f "vllm serve" || true
sleep 20

echo "== train =="
MS=""; [ -n "$MAX" ] && MS="--max-steps $MAX"
.venv-gpu/bin/python -m train.grpo_trainer --episodes "$ROUND/rollouts.jsonl" \
    --out "$ROUND" $MS

echo "== restart executor on new checkpoint =="
CKPT="$PWD/$ROUND/checkpoint"
CUDA_VISIBLE_DEVICES=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold) \
  nohup .venv-gpu/bin/vllm serve "$CKPT" --served-model-name Qwen/Qwen3.5-9B \
  --port 8378 --dtype auto --max-model-len 16384 --gpu-memory-utilization 0.85 \
  > "$ROUND/vllm_restart.log" 2>&1 &
echo "$CKPT" > .serve_model
for i in $(seq 1 80); do
  curl -s -m 5 http://127.0.0.1:8378/v1/models 2>/dev/null | grep -q "Qwen3.5-9B" && { echo "server UP on $CKPT"; exit 0; }
  sleep 15
done
echo "server failed to restart"; tail -5 "$ROUND/vllm_restart.log"; exit 1
