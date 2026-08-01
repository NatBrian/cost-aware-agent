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
ckpt_at() {     # checkpoint for a specific round, local or backed up
  local tag=$1 R=$2
  for C in "$PWD/experiments/results/train/${tag}_round$R/checkpoint" \
           "$BACKUP/checkpoints/${tag}_round$R"; do
    [ -f "$C/config.json" ] && { echo "$C"; return; }
  done
}
# A round counts only if it PASSED its health probe. Checkpoints are written and
# backed up BEFORE the probe runs (deliberately: a failed probe is evidence worth
# keeping), so "a checkpoint exists" does not mean "this policy is usable".
# ctrl_round3 exists and scored 20.5% malformed against a 10% gate; selecting the
# newest existing checkpoint would have silently evaluated a damaged policy and
# called it the control. (2026-08-01)
healthy_at() {
  local tag=$1 R=$2
  for M in "$PWD/experiments/results/train/${tag}_round$R/HEALTHY" \
           "$BACKUP/checkpoints/${tag}_round$R/HEALTHY"; do
    [ -f "$M" ] && { echo yes; return; }
  done
}
last_round() { local tag=$1 best=0
  for R in 1 2 3; do
    [ -n "$(healthy_at "$tag" "$R")" ] && [ -n "$(ckpt_at "$tag" "$R")" ] && best=$R
  done
  echo "$best"; }

R_CTRL=$(last_round ctrl); R_TRT=$(last_round trt)
CKPT_CTRL=$(ckpt_at ctrl "$R_CTRL"); CKPT_TRT=$(ckpt_at trt "$R_TRT")
echo ""
echo "control   round $R_CTRL: ${CKPT_CTRL:-MISSING}"
echo "treatment round $R_TRT: ${CKPT_TRT:-MISSING}"
[ -n "$CKPT_CTRL" ] && [ -n "$CKPT_TRT" ] || { echo "FATAL: an arm produced no checkpoint"; exit 1; }

# If a health gate stopped one arm early, comparing round 3 against round 1 is an
# ASYMMETRIC comparison — the exact trap FOUNDATION-1 hit with λ=1.0, where the
# only honest fix was to report both. So when the rounds differ we additionally
# evaluate the control AT THE TREATMENT'S ROUND and report both comparisons; the
# protocol-matched one is the one the pre-registered rule reads.
# The mismatch can go EITHER WAY. This was written assuming the treatment would
# breach first (λ=0.568 being the untested value); in the event the λ=0 CONTROL
# breached at round 3 while the treatment may reach 3. So: bring whichever arm sits
# at the HIGHER round back down to the lower one, and evaluate that as the
# round-matched comparison.
MATCHED=""; MATCHED_NAME=""
R_MATCH=$(( R_CTRL < R_TRT ? R_CTRL : R_TRT ))
if [ "$R_CTRL" != "$R_TRT" ]; then
  echo "!! ROUND MISMATCH (control r$R_CTRL vs treatment r$R_TRT) — a health gate"
  echo "!! stopped an arm early. Round-matched comparison will use r$R_MATCH."
  if [ "$R_CTRL" -gt "$R_TRT" ]; then
    MATCHED_NAME=controlmatched;   MATCHED=$(ckpt_at ctrl "$R_MATCH")
  else
    MATCHED_NAME=treatmentmatched; MATCHED=$(ckpt_at trt  "$R_MATCH")
  fi
  [ -n "$MATCHED" ] || echo "!! no checkpoint at round $R_MATCH — matched arm unavailable"
fi
printf '{"control_round":%s,"treatment_round":%s,"match_round":%s,"matched":%s,"matched_arm":"%s","lambda_treatment":%s}\n' \
  "$R_CTRL" "$R_TRT" "$R_MATCH" "$([ -n "$MATCHED" ] && echo true || echo false)" \
  "$MATCHED_NAME" "$LAM_TRT" > "$OUT/arm_provenance.json"
cat "$OUT/arm_provenance.json"

# ----------------------------------------------------------------- S5: eval --
stop_executor() {
  local pid; pid=$(pgrep -u "$(whoami)" -f "[v]llm serve.*port 8378" | head -1) || true
  [ -n "${pid:-}" ] && { kill "$pid"; sleep 20; } || true
}
EVAL_FILE=$($PY -c "import sys;sys.path.insert(0,'.');from common import load_config;print(load_config()['data']['eval_file'])")
echo ""
echo "############ S5 EVALUATION on $EVAL_FILE ############"

EVAL_SPECS="control:$CKPT_CTRL treatment:$CKPT_TRT"
[ -n "$MATCHED" ] && EVAL_SPECS="$EVAL_SPECS ${MATCHED_NAME}:$MATCHED"

for SPEC in $EVAL_SPECS; do
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
# Robustness only; cannot change the verdict. No-op unless a health gate stopped
# an arm early and left the two arms at different rounds.
$PY scripts/s5_matched.py --dir "$OUT" 2>&1 | tee -a "$OUT/s3_verdict.txt"
$PY -m analysis.s5_figures --dir "$OUT" --out-dir experiments/reports/figs 2>&1 | tail -3
cp "$OUT/s3_verdict.txt" "$OUT/s3_verdict.json" "$BACKUP/" 2>/dev/null || true
echo "=== S4+S5 COMPLETE ==="
