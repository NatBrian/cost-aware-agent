#!/usr/bin/env bash
# T4 — does the cost-aware effect grow with horizon? Train both arms on MuSiQue
# and evaluate on the matched MuSiQue eval set.
#
# Usage: bash scripts/t4_musique_run.sh
#
# Everything is matched to the HotpotQA run except the dataset: same 300 train
# tasks, same G=8, same 3 rounds, same health gates, same λ pair (0 and 0.568),
# same 600-question eval, same gate budget B=2. If MuSiQue got more data or more
# rounds, a bigger effect could be volume rather than horizon.
#
# Restartable: rounds with a checkpoint are skipped, collection resumes from its
# shards, completed eval JSONLs are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BACKUP=/mnt/src/liangsheng/cassi_foundation
OUT=experiments/results/t4_musique
mkdir -p "$OUT" "$BACKUP/checkpoints"
GPU=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold | cut -d, -f2)
export TASK_FILE=data/musique_train_300.jsonl
EVAL_FILE=data/musique_eval_600.jsonl

judge_up() {
  curl -s -m 10 http://127.0.0.1:6101/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3.6-27B","messages":[{"role":"user","content":"ok"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    2>/dev/null | grep -q '"content"'
}
echo "=== waiting for the judge ==="
for i in $(seq 1 120); do judge_up && { echo "judge ready"; break; }
  [ "$i" = 120 ] && { echo "FATAL: judge never came up"; exit 1; }; sleep 20; done

# ---------------------------------------------------------------- train ------
for SPEC in "mqctrl:0.0" "mqtrt:0.568"; do
  TAG=${SPEC%%:*}; LAM=${SPEC##*:}
  echo ""
  echo "############ T4 ARM $TAG (λ=$LAM) on MuSiQue ############"
  TASK_FILE=$TASK_FILE bash scripts/run_lambda_arm.sh "$LAM" "$TAG" 2>&1 || \
    echo "!! ARM $TAG ended non-zero (health gate or error) — continuing with its last healthy checkpoint"
done

# ---- last HEALTHY checkpoint per arm (never just the newest that exists) -----
ckpt_at() { for C in "$PWD/experiments/results/train/${1}_round$2/checkpoint" \
                     "$BACKUP/checkpoints/${1}_round$2"; do
              [ -f "$C/config.json" ] && { echo "$C"; return; }; done; }
healthy_at() { for M in "$PWD/experiments/results/train/${1}_round$2/HEALTHY" \
                        "$BACKUP/checkpoints/${1}_round$2/HEALTHY"; do
                 [ -f "$M" ] && { echo yes; return; }; done; }
last_round() { local b=0; for R in 1 2 3; do
                 [ -n "$(healthy_at "$1" $R)" ] && [ -n "$(ckpt_at "$1" $R)" ] && b=$R; done; echo $b; }

R_C=$(last_round mqctrl); R_T=$(last_round mqtrt)
C_C=$(ckpt_at mqctrl "$R_C"); C_T=$(ckpt_at mqtrt "$R_T")
echo ""
echo "mqctrl round $R_C: ${C_C:-MISSING}"
echo "mqtrt  round $R_T: ${C_T:-MISSING}"
[ -n "$C_C" ] && [ -n "$C_T" ] || { echo "FATAL: an arm produced no healthy checkpoint"; exit 1; }
R_M=$(( R_C < R_T ? R_C : R_T ))
MATCH=""; MNAME=""
if [ "$R_C" != "$R_T" ]; then
  if [ "$R_C" -gt "$R_T" ]; then MNAME=mqctrlmatched; MATCH=$(ckpt_at mqctrl "$R_M")
  else MNAME=mqtrtmatched; MATCH=$(ckpt_at mqtrt "$R_M"); fi
  echo "!! round mismatch ($R_C vs $R_T) — adding $MNAME at r$R_M"
fi
printf '{"control_round":%s,"treatment_round":%s,"match_round":%s,"matched":%s,"matched_arm":"%s","dataset":"musique"}\n' \
  "$R_C" "$R_T" "$R_M" "$([ -n "$MATCH" ] && echo true || echo false)" "$MNAME" \
  > "$OUT/arm_provenance.json"

# ---------------------------------------------------------------- eval -------
stop_executor() { local p; p=$(pgrep -u "$(whoami)" -f "[v]llm serve.*port 8378" | head -1) || true
                  [ -n "${p:-}" ] && { kill "$p"; sleep 20; } || true; }
SPECS="control:$C_C treatment:$C_T"
[ -n "$MATCH" ] && SPECS="$SPECS ${MNAME}:$MATCH"

echo ""
echo "############ T4 EVALUATION on $EVAL_FILE ############"
for SPEC in $SPECS; do
  ARM=${SPEC%%:*}; CKPT=${SPEC#*:}
  DONE=1; for B in small medium large; do [ -s "$OUT/${ARM}_${B}.jsonl" ] || DONE=0; done
  [ "$DONE" = 1 ] && { echo "== $ARM already evaluated"; continue; }
  echo "=== serving $ARM ==="
  stop_executor
  CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv-gpu3/bin/vllm serve "$CKPT" \
    --served-model-name Qwen/Qwen3.5-9B --port 8378 --dtype auto \
    --max-model-len 16384 --gpu-memory-utilization 0.85 > "$OUT/serve_$ARM.log" 2>&1 &
  OK=0
  for i in $(seq 1 80); do
    curl -s -m 10 http://127.0.0.1:8378/v1/chat/completions -H "Content-Type: application/json" \
      -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
      2>/dev/null | grep -q '"content"' && { OK=1; echo "$ARM serving"; break; }
    sleep 15
  done
  [ "$OK" = 1 ] || { echo "FATAL: $ARM would not serve"; tail -20 "$OUT/serve_$ARM.log"; exit 1; }
  for B in small medium large; do
    [ -s "$OUT/${ARM}_${B}.jsonl" ] && continue
    echo "--- $ARM @ B=$B ---"
    $PY -m collect.run_collection --task-file "$EVAL_FILE" \
        --arm a3 --mode none --budget "$B" --g 1 --temperature 0 \
        --out "$OUT/${ARM}_${B}.jsonl.part" || exit 1
    mv "$OUT/${ARM}_${B}.jsonl.part" "$OUT/${ARM}_${B}.jsonl"
  done
done
stop_executor
cp -r "$OUT" "$BACKUP/" 2>/dev/null || true
$PY scripts/t4_analyse.py --dir "$OUT" | tee "$OUT/t4_verdict.txt"
echo "=== T4 COMPLETE ==="
