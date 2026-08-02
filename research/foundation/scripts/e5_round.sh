#!/usr/bin/env bash
# One GRPO round: sharded collect -> stop server -> train -> serve checkpoint.
# Usage: bash scripts/e5_round.sh <round_dir> <n_tasks> <G> [max_steps] [init_ckpt]
#
# TASK_FILE env var selects the training set (default HotpotQA, so every existing
# caller is unchanged). T4 sets it to the MuSiQue split to test whether the effect
# grows with horizon.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROUND=$1; N=$2; G=$3; MAX=${4:-}; INIT=${5:-}
TASK_FILE=${TASK_FILE:-data/hotpotqa_train_300.jsonl}
SHARDS=6
mkdir -p "$ROUND"

echo "== collect ($N tasks x G=$G, $SHARDS shards, train mode) from $TASK_FILE =="
PER=$(( (N + SHARDS - 1) / SHARDS ))
for s in $(seq 0 $((SHARDS - 1))); do
  $PY -m collect.run_collection --task-file "$TASK_FILE" \
      --skip $((s * PER)) --limit $PER --arm a3 --mode none --budget draw \
      --g "$G" --train --out "$ROUND/rollouts_shard$s.jsonl" &
done
wait
cat "$ROUND"/rollouts_shard*.jsonl > "$ROUND/rollouts.jsonl"
wc -l "$ROUND/rollouts.jsonl"

echo "== stop OUR executor server =="
# Bracketed so the pattern cannot match the shell running it -- an unbracketed
# `pkill -f` matched its own process three times in this project and once killed
# the very script issuing it. (2026-08-01)
pkill -u "$(whoami)" -f "[v]llm serve.*8378" || true
sleep 25

echo "== train =="
ARGS=""
[ -n "$MAX" ] && ARGS="$ARGS --max-steps $MAX"
[ -n "$INIT" ] && ARGS="$ARGS --init-from $INIT"
# GPU 0 hosts the JUDGE (~90G) + retrieval; only the SECOND held card is free for
# training and serving. Passing the whole hold made torch pick cuda:0 and OOM
# against the judge (2026-07-30). Pin to the second card everywhere.
GPUS=$(grep -oP "CUDA_VISIBLE_DEVICES=\\K[0-9,]+" .gpu_hold | cut -d, -f2)
# PYTHONUNBUFFERED: python block-buffers stdout when redirected, so ~30 update
# lines (1.8K) sit unseen in a 4K buffer until the process exits — a 2h round
# looks identical to a hung one. Flush per line so progress is observable.
CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  .venv-train/bin/python -m train.grpo_trainer --episodes "$ROUND/rollouts.jsonl" \
    --out "$ROUND" $ARGS

echo "== serve new checkpoint =="
CKPT="$PWD/$ROUND/checkpoint"
GPUS=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold | cut -d, -f2)
CUDA_VISIBLE_DEVICES=$GPUS nohup .venv-gpu3/bin/vllm serve "$CKPT" \
  --served-model-name Qwen/Qwen3.5-9B --port 8378 --dtype auto \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  > "$ROUND/vllm_restart.log" 2>&1 &
echo "$CKPT" > .serve_model
for i in $(seq 1 80); do
  # a REAL completion, not /v1/models: the API server answers that even when the
  # engine core is dead, which once reported a 500-ing executor as healthy
  curl -s -m 10 http://127.0.0.1:8378/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    2>/dev/null | grep -q '"content"' && { echo "server UP on $CKPT"; exit 0; }
  sleep 15
done
echo "server failed to restart"; tail -5 "$ROUND/vllm_restart.log"; exit 1
