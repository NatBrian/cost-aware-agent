#!/usr/bin/env bash
# F6 — evaluate A3 on the frozen dev-200 and run the pre-registered gate.
#
# Usage: bash scripts/f6_eval.sh <trained_ckpt_abs_path>
#
# This spends ONE of the <=3 permitted dev-200 looks. Log it in PROGRESS.md with
# the date and reason before running.
#
# A3 is evaluated with the harness OFF (the plan's claim) AND ON (reported for
# the internalization gap), at every budget, at temperature 0. The oracle
# forced-continuation replay gives the "what would more steps have bought"
# counterfactual for the analysis.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=${1:?usage: f6_eval.sh <trained_ckpt_abs_path>}
PY=.venv/bin/python
OUT=experiments/results/eval
mkdir -p "$OUT"

echo "== serving trained checkpoint =="
PID=$(pgrep -u "$(whoami)" -f "[v]llm serve" | head -1) && [ -n "$PID" ] && kill "$PID" && sleep 20 || true
GPUS=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold | cut -d, -f2)
CUDA_VISIBLE_DEVICES=${GPUS:-1} nohup .venv-gpu3/bin/vllm serve "$CKPT" \
  --served-model-name Qwen/Qwen3.5-9B --port 8378 --dtype auto \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  > "$OUT/vllm_eval.log" 2>&1 &
for i in $(seq 1 80); do
  curl -s -m 5 http://127.0.0.1:8378/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    2>/dev/null | grep -q '"content"' && { echo "checkpoint serving"; break; }
  sleep 15
  [ "$i" = 80 ] && { echo "checkpoint failed to serve"; exit 1; }
done

for B in small medium large; do
  for M in none enforce; do
    echo "== A3 mode=$M budget=$B =="
    $PY -m collect.run_collection --task-file data/hotpotqa_dev_200.jsonl \
        --arm a3 --mode "$M" --budget "$B" --g 1 --temperature 0 \
        --out "$OUT/a3_${M}_${B}.jsonl"
  done
done

echo "== oracle forced-continuation replay (medium) =="
$PY -m collect.run_collection --task-file data/hotpotqa_dev_200.jsonl \
    --arm a3 --mode forced_continuation --budget medium --g 1 --temperature 0 \
    --out "$OUT/a3_oracle_medium.jsonl"

echo "== build eval rows =="
# the oracle replay is analysis-only: it is NOT a stop decision, so it must not
# enter the gate population (its answered_at is logged while the episode runs on)
$PY scripts/f6_build_eval.py \
    --baselines experiments/results/baselines/baseline_rows.csv \
    --a3 "$OUT"/a3_none_*.jsonl "$OUT"/a3_enforce_*.jsonl \
    --out experiments/results/foundation_eval.csv

echo "== PRE-REGISTERED GATE =="
$PY -m eval.gate_check --rows experiments/results/foundation_eval.csv
