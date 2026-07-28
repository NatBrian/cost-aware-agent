#!/usr/bin/env bash
# One GRPO round: sharded collect -> stop server -> train -> serve checkpoint.
# Usage: bash scripts/e5_round.sh <round_dir> <n_tasks> <G> [max_steps] [init_ckpt]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROUND=$1; N=$2; G=$3; MAX=${4:-}; INIT=${5:-}
SHARDS=6
mkdir -p "$ROUND"

echo "== collect ($N tasks x G=$G, $SHARDS shards, train mode) =="
PER=$(( (N + SHARDS - 1) / SHARDS ))
for s in $(seq 0 $((SHARDS - 1))); do
  $PY -m collect.run_collection --task-file data/hotpotqa_train_300.jsonl \
      --skip $((s * PER)) --limit $PER --arm a3 --mode none --budget draw \
      --g "$G" --train --out "$ROUND/rollouts_shard$s.jsonl" &
done
wait
cat "$ROUND"/rollouts_shard*.jsonl > "$ROUND/rollouts.jsonl"
wc -l "$ROUND/rollouts.jsonl"

echo "== stop OUR executor server =="
pkill -u "$(whoami)" -f "vllm serve.*8378" || true
sleep 25

echo "== train =="
ARGS=""
[ -n "$MAX" ] && ARGS="$ARGS --max-steps $MAX"
[ -n "$INIT" ] && ARGS="$ARGS --init-from $INIT"
GPUS=$(grep -oP "CUDA_VISIBLE_DEVICES=\\K[0-9,]+" .gpu_hold)
CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv-gpu3/bin/python -m train.grpo_trainer --episodes "$ROUND/rollouts.jsonl" \
    --out "$ROUND" $ARGS

echo "== serve new checkpoint =="
CKPT="$PWD/$ROUND/checkpoint"
GPUS=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold)
CUDA_VISIBLE_DEVICES=$GPUS nohup .venv-gpu3/bin/vllm serve "$CKPT" \
  --served-model-name Qwen/Qwen3.5-9B --port 8378 --dtype auto \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  > "$ROUND/vllm_restart.log" 2>&1 &
echo "$CKPT" > .serve_model
for i in $(seq 1 80); do
  curl -s -m 5 http://127.0.0.1:8378/v1/models 2>/dev/null | grep -q "Qwen3.5-9B" && { echo "server UP on $CKPT"; exit 0; }
  sleep 15
done
echo "server failed to restart"; tail -5 "$ROUND/vllm_restart.log"; exit 1
