#!/usr/bin/env bash
# Run one complete λ arm of the ablation, start to finish, unattended.
#
# Usage: bash scripts/run_lambda_arm.sh <train_lambda> <tag>
#   e.g. bash scripts/run_lambda_arm.sh 0.0 lam0
#
# Sets economy.train_lambda (the REWARD's λ — economy.lambda, the evaluation
# yardstick, is deliberately left alone), serves the base model, then runs three
# rounds with the temp-1.0 health probe as a HARD gate between each. Stops on a
# failed probe rather than training the next round on a damaged policy — that is
# precisely what wasted the first attempt at this experiment.
#
# Arms must run SEQUENTIALLY: they share one config file and one pair of GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
LAM=${1:?usage: run_lambda_arm.sh <train_lambda> <tag>}
TAG=${2:?usage: run_lambda_arm.sh <train_lambda> <tag>}
BACKUP=/mnt/src/liangsheng/cassi_foundation
GPU=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold | cut -d, -f2)

echo "=== ARM $TAG : train_lambda=$LAM ==="
sed -i "s/^  train_lambda: .*/  train_lambda: $LAM                # ablation arm $TAG/" configs/foundation.yaml
grep -E "^  (lambda|train_lambda):" configs/foundation.yaml

echo "=== serve BASE model (round 1 must collect from base) ==="
PID=$(pgrep -u "$(whoami)" -f "[v]llm serve" | head -1) && [ -n "$PID" ] && kill "$PID" && sleep 20 || true
CUDA_VISIBLE_DEVICES=${GPU:-1} nohup bash scripts/serve_executor.sh > "$BACKUP/${TAG}_serve_base.log" 2>&1 &
for i in $(seq 1 80); do
  curl -s -m 5 http://127.0.0.1:8378/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    2>/dev/null | grep -q '"content"' && { echo "base model serving"; break; }
  sleep 15
  [ "$i" = 80 ] && { echo "ARM $TAG FAILED: base model would not serve"; exit 1; }
done

INIT=""
for R in 1 2 3; do
  DIR="experiments/results/train/${TAG}_round$R"
  echo "=== $TAG round $R ==="
  if [ -z "$INIT" ]; then
    bash scripts/e5_round.sh "$DIR" 300 8 150
  else
    bash scripts/e5_round.sh "$DIR" 300 8 150 "$INIT"
  fi
  grep -q "server UP on" "$BACKUP/${TAG}_round$R.log" 2>/dev/null || true

  echo "=== $TAG round $R HEALTH GATE ==="
  if ! bash scripts/probe_policy_health.sh "/tmp/probe_${TAG}_r$R.jsonl"; then
    echo "ARM $TAG STOPPED: health probe FAILED after round $R — refusing to train on a damaged policy"
    exit 1
  fi

  mkdir -p "$BACKUP/checkpoints" "$BACKUP/trajectories"
  cp "$DIR/rollouts.jsonl" "$BACKUP/trajectories/${TAG}_round${R}_rollouts.jsonl" 2>/dev/null || true
  cp "$DIR"/{round_summary.json,train_log.jsonl,divergence.jsonl} "$BACKUP/trajectories/" 2>/dev/null || true
  for f in round_summary train_log divergence; do
    [ -f "$DIR/$f.json" ] && cp "$DIR/$f.json" "$BACKUP/trajectories/${TAG}_r${R}_$f.json" || true
    [ -f "$DIR/$f.jsonl" ] && cp "$DIR/$f.jsonl" "$BACKUP/trajectories/${TAG}_r${R}_$f.jsonl" || true
  done
  cp -r "$DIR/checkpoint" "$BACKUP/checkpoints/${TAG}_round$R" 2>/dev/null || true
  INIT="$PWD/$DIR/checkpoint"
  # keep only the newest local checkpoint: 19G each, all are on /mnt/src
  [ "$R" -gt 1 ] && rm -rf "experiments/results/train/${TAG}_round$((R-1))/checkpoint" || true
done

echo "=== ARM $TAG COMPLETE — final checkpoint: $INIT ==="
