#!/usr/bin/env bash
# T3 — seed replication of the MuSiQue result (the headline configuration).
#
# Usage: bash scripts/t3_seeds_run.sh
#
# Seed 42 already exists (T4). This adds 123 and 789, giving the three seeds a
# reviewer will expect for a headline number. One seed with a CI excluding zero
# is suggestive; three is evidence.
#
# WHAT THE SEED CHANGES: `seed` drives the per-task budget draw and the GRPO
# update order. It does NOT touch `sampling_seed`, so the frozen train-300 and
# eval-600 splits are identical across seeds -- the comparison isolates training
# stochasticity, not data.
#
# ROUND-MATCHED BY CONSTRUCTION. On MuSiQue the λ=0 control fails its health gate
# early (seed 42: round 2 at 29.4% malformed), so each seed is evaluated with BOTH
# arms at min(last healthy round). That is the primary comparison per the T4
# amendment: matching training amount so only λ differs.
#
# Evaluated at the GATE BUDGET ONLY (B=2). The other budgets are already reported
# for seed 42 and would triple the eval cost for a robustness check.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BACKUP=/mnt/src/liangsheng/cassi_foundation
OUT=experiments/results/t3_seeds
mkdir -p "$OUT"
GPU=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold | cut -d, -f2)
export TASK_FILE=data/musique_train_300.jsonl
EVAL_FILE=data/musique_eval_600.jsonl

judge_up() { curl -s -m 10 http://127.0.0.1:6101/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B","messages":[{"role":"user","content":"ok"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  2>/dev/null | grep -q '"content"'; }
echo "=== waiting for the judge ==="
for i in $(seq 1 120); do judge_up && { echo "judge ready"; break; }
  [ "$i" = 120 ] && { echo "FATAL: judge never came up"; exit 1; }; sleep 20; done

ckpt_at() { for C in "$PWD/experiments/results/train/${1}_round$2/checkpoint" \
                     "$BACKUP/checkpoints/${1}_round$2"; do
              [ -f "$C/config.json" ] && { echo "$C"; return; }; done; }
healthy_at() { for M in "$PWD/experiments/results/train/${1}_round$2/HEALTHY" \
                        "$BACKUP/checkpoints/${1}_round$2/HEALTHY"; do
                 [ -f "$M" ] && { echo yes; return; }; done; }
last_round() { local b=0; for R in 1 2 3; do
                 [ -n "$(healthy_at "$1" $R)" ] && [ -n "$(ckpt_at "$1" $R)" ] && b=$R; done; echo $b; }
stop_executor() { local p; p=$(pgrep -u "$(whoami)" -f "[v]llm serve.*port 8378" | head -1) || true
                  [ -n "${p:-}" ] && { kill "$p"; sleep 20; } || true; }

for SEED in 123 789; do
  echo ""
  echo "################## SEED $SEED ##################"
  sed -i "s/^seed: .*/seed: $SEED                             # T3 seed replication/" configs/foundation.yaml
  grep -E "^seed:" configs/foundation.yaml

  for SPEC in "s${SEED}ctrl:0.0" "s${SEED}trt:0.568"; do
    TAG=${SPEC%%:*}; LAM=${SPEC##*:}
    echo "--- arm $TAG (λ=$LAM) ---"
    TASK_FILE=$TASK_FILE bash scripts/run_lambda_arm.sh "$LAM" "$TAG" 2>&1 || \
      echo "!! $TAG ended non-zero (health gate) — using its last healthy checkpoint"
  done

  RC=$(last_round "s${SEED}ctrl"); RT=$(last_round "s${SEED}trt")
  RM=$(( RC < RT ? RC : RT ))
  echo "seed $SEED: ctrl last healthy r$RC, trt r$RT -> MATCHED at r$RM"
  if [ "$RM" -lt 1 ]; then echo "!! seed $SEED produced no matched pair — skipping eval"; continue; fi
  printf '{"seed":%s,"ctrl_round":%s,"trt_round":%s,"match_round":%s}\n' \
    "$SEED" "$RC" "$RT" "$RM" > "$OUT/provenance_s${SEED}.json"

  for SPEC in "ctrl:$(ckpt_at "s${SEED}ctrl" "$RM")" "trt:$(ckpt_at "s${SEED}trt" "$RM")"; do
    ARM=${SPEC%%:*}; CKPT=${SPEC#*:}
    F="$OUT/s${SEED}_${ARM}.jsonl"
    [ -s "$F" ] && { echo "== s${SEED}_${ARM} done"; continue; }
    [ -n "$CKPT" ] || { echo "!! no checkpoint for s${SEED}_${ARM} at r$RM"; continue; }
    echo "=== serving s${SEED}_${ARM} (r$RM) ==="
    stop_executor
    CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv-gpu3/bin/vllm serve "$CKPT" \
      --served-model-name Qwen/Qwen3.5-9B --port 8378 --dtype auto \
      --max-model-len 16384 --gpu-memory-utilization 0.85 > "$OUT/serve.log" 2>&1 &
    OK=0
    for i in $(seq 1 80); do
      curl -s -m 10 http://127.0.0.1:8378/v1/chat/completions -H "Content-Type: application/json" \
        -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
        2>/dev/null | grep -q '"content"' && { OK=1; break; }
      sleep 15
    done
    [ "$OK" = 1 ] || { echo "FATAL: s${SEED}_${ARM} would not serve"; exit 1; }
    $PY -m collect.run_collection --task-file "$EVAL_FILE" \
        --arm a3 --mode none --budget small --g 1 --temperature 0 \
        --out "$F.part" || exit 1
    mv "$F.part" "$F"
    cp "$F" "$BACKUP/" 2>/dev/null || true
  done
done
stop_executor

# restore the original seed so the repo is not left mid-experiment
sed -i "s/^seed: .*/seed: 42/" configs/foundation.yaml
echo "config seed restored to 42"

$PY scripts/t3_analyse.py --dir "$OUT" | tee "$OUT/t3_verdict.txt"
echo "=== T3 COMPLETE ==="
