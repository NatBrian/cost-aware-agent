#!/usr/bin/env bash
# S4 + S5 — train both Step-1 arms, evaluate on the frozen eval-600, and apply
# the PRE-REGISTERED rule. Designed to run unattended for ~20 hours.
#
# Usage: bash scripts/s4_s5_run.sh
#
# Arms (S3 §2): control λ=0.0 and treatment λ=0.568, trained from the SAME base
# on the SAME data with the SAME seed, rounds and budgets. Only λ differs. The
# FOUNDATION-1 λ=0 checkpoint is deliberately NOT reused: it was trained under
# budgets {2,4,8}, so reusing it would confound the λ change with the budget
# change and make the result unattributable.
#
# RESTARTABLE at every level. run_lambda_arm.sh skips rounds whose checkpoint
# exists; collection resumes from its shards; eval skips completed JSONLs. Re-run
# this command after any interruption — that is the intended recovery.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BACKUP=/mnt/src/liangsheng/cassi_foundation
OUT=experiments/results/s5_eval
mkdir -p "$OUT" "$BACKUP/checkpoints" "$BACKUP/trajectories"
GPU=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold | cut -d, -f2)

LAM_CTRL=0.0
LAM_TRT=0.568          # S2-calibrated, frozen. Cap 0.6. NEVER retuned after eval.

judge_up() {
  curl -s -m 10 http://127.0.0.1:6101/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3.6-27B","messages":[{"role":"user","content":"ok"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    2>/dev/null | grep -q '"content"'
}

echo "=== waiting for the judge (training rewards need it) ==="
for i in $(seq 1 120); do
  judge_up && { echo "judge ready"; break; }
  [ "$i" = 120 ] && { echo "FATAL: judge never came up"; exit 1; }
  sleep 20
done

# ---------------------------------------------------------------- S4: train --
# A failed health probe is a RESULT, not a crash: per S3 §8 the arm stops at its
# last healthy checkpoint and the deviation is reported. So a non-zero exit from
# run_lambda_arm.sh must not kill the whole run.
for SPEC in "ctrl:$LAM_CTRL" "trt:$LAM_TRT"; do
  TAG=${SPEC%%:*}; LAM=${SPEC##*:}
  echo ""
  echo "############ S4 ARM $TAG (λ=$LAM) ############"
  bash scripts/run_lambda_arm.sh "$LAM" "$TAG" 2>&1 || \
    echo "!! ARM $TAG ended non-zero (health gate or error) — continuing with its last healthy checkpoint"
done

# ---- resolve each arm's final usable checkpoint ------------------------------
final_ckpt() {  # newest round with a checkpoint, local or backed up
  local tag=$1 best=""
  for R in 1 2 3; do
    for C in "$PWD/experiments/results/train/${tag}_round$R/checkpoint" \
             "$BACKUP/checkpoints/${tag}_round$R"; do
      [ -f "$C/config.json" ] && best="$C"
    done
  done
  echo "$best"
}
CKPT_CTRL=$(final_ckpt ctrl); CKPT_TRT=$(final_ckpt trt)
echo ""
echo "control   checkpoint: ${CKPT_CTRL:-MISSING}"
echo "treatment checkpoint: ${CKPT_TRT:-MISSING}"
[ -n "$CKPT_CTRL" ] && [ -n "$CKPT_TRT" ] || { echo "FATAL: an arm produced no checkpoint"; exit 1; }

# ----------------------------------------------------------------- S5: eval --
stop_executor() {
  local pid; pid=$(pgrep -u "$(whoami)" -f "[v]llm serve.*port 8378" | head -1) || true
  [ -n "${pid:-}" ] && { kill "$pid"; sleep 20; } || true
}
EVAL_FILE=$($PY -c "import sys;sys.path.insert(0,'.');from common import load_config;print(load_config()['data']['eval_file'])")
echo ""
echo "############ S5 EVALUATION on $EVAL_FILE ############"

for SPEC in "control:$CKPT_CTRL" "treatment:$CKPT_TRT"; do
  ARM=${SPEC%%:*}; CKPT=${SPEC#*:}
  DONE=1
  for B in small medium large; do [ -s "$OUT/${ARM}_${B}.jsonl" ] || DONE=0; done
  [ "$DONE" = 1 ] && { echo "== $ARM already evaluated — skipping"; continue; }

  echo "=== serving $ARM ==="
  stop_executor
  CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv-gpu3/bin/vllm serve "$CKPT" \
    --served-model-name Qwen/Qwen3.5-9B --port 8378 --dtype auto \
    --max-model-len 16384 --gpu-memory-utilization 0.85 \
    > "$OUT/serve_$ARM.log" 2>&1 &
  READY=0
  for i in $(seq 1 80); do
    curl -s -m 10 http://127.0.0.1:8378/v1/chat/completions -H "Content-Type: application/json" \
      -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
      2>/dev/null | grep -q '"content"' && { READY=1; echo "$ARM serving"; break; }
    sleep 15
  done
  [ "$READY" = 1 ] || { echo "FATAL: $ARM would not serve"; tail -30 "$OUT/serve_$ARM.log"; exit 1; }

  for B in small medium large; do
    [ -s "$OUT/${ARM}_${B}.jsonl" ] && continue
    echo "--- $ARM @ B=$B ---"
    $PY -m collect.run_collection --task-file "$EVAL_FILE" \
        --arm a3 --mode none --budget "$B" --g 1 --temperature 0 \
        --out "$OUT/${ARM}_${B}.jsonl.part" || { echo "collection failed"; exit 1; }
    mv "$OUT/${ARM}_${B}.jsonl.part" "$OUT/${ARM}_${B}.jsonl"
    cp "$OUT/${ARM}_${B}.jsonl" "$BACKUP/trajectories/s5_${ARM}_${B}.jsonl" || true
  done
done
stop_executor

echo ""
echo "############ APPLYING THE PRE-REGISTERED RULE ############"
$PY scripts/s3_analyse.py --dir "$OUT" | tee "$OUT/s3_verdict.txt"
cp "$OUT/s3_verdict.txt" "$OUT/s3_verdict.json" "$BACKUP/" 2>/dev/null || true
echo "=== S4+S5 COMPLETE ==="
