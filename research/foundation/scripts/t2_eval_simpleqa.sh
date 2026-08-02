#!/usr/bin/env bash
# T2b — SimpleQA negative control. Evaluation only; both arms already trained.
#
# Usage: bash scripts/t2_eval_simpleqa.sh [pilot|full]
#
#   pilot : control only, 50 questions -- the POWER CHECK. SimpleQA is
#           adversarially obscure and our index is a 2018 Wikipedia dump, so if
#           the baseline F1 is ~0 then ΔF1 ≈ 0 is trivially true and the control
#           is UNINFORMATIVE rather than passed. Never skip this.
#   full  : both arms, the full set, at the gate budget B=2.
#
# Restartable: completed JSONLs are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODE=${1:-pilot}
OUT=experiments/results/t2_simpleqa
mkdir -p "$OUT"
BACKUP=/mnt/src/liangsheng/cassi_foundation
GPU=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold | cut -d, -f2)

CKPT_CTRL=$(ls -d "$PWD/experiments/results/train/ctrl_round2/checkpoint" \
                  "$BACKUP/checkpoints/ctrl_round2" 2>/dev/null | head -1)
CKPT_TRT=$(ls -d "$PWD/experiments/results/train/trt_round3/checkpoint" \
                 "$BACKUP/checkpoints/trt_round3" 2>/dev/null | head -1)

stop_executor() {
  local pid; pid=$(pgrep -u "$(whoami)" -f "[v]llm serve.*port 8378" | head -1) || true
  [ -n "${pid:-}" ] && { kill "$pid"; sleep 20; } || true
  # the engine core outlives the API server and keeps the GPU (2026-08-01)
  local eng; eng=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  for p in $eng; do
    [ -d /proc/$p ] || continue
    [ "$(stat -c %U /proc/$p 2>/dev/null)" = "$(whoami)" ] || continue
    grep -q "EngineCore" /proc/$p/comm 2>/dev/null && \
      [ "$p" != "$(pgrep -u "$(whoami)" -f '[s]erve_judge_local' | head -1)" ] && true
  done
}

serve() {  # $1 = checkpoint
  stop_executor
  CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv-gpu3/bin/vllm serve "$1" \
    --served-model-name Qwen/Qwen3.5-9B --port 8378 --dtype auto \
    --max-model-len 16384 --gpu-memory-utilization 0.85 \
    > "$OUT/serve.log" 2>&1 &
  for i in $(seq 1 80); do
    curl -s -m 10 http://127.0.0.1:8378/v1/chat/completions -H "Content-Type: application/json" \
      -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
      2>/dev/null | grep -q '"content"' && { echo "serving"; return 0; }
    sleep 15
  done
  echo "FAILED to serve $1"; return 1
}

if [ "$MODE" = "pilot" ]; then
  echo "=== POWER CHECK: control on 50 SimpleQA questions ==="
  serve "$CKPT_CTRL" || exit 1
  $PY -m collect.run_collection --task-file data/simpleqa_eval_pilot.jsonl \
      --arm a3 --mode none --budget small --g 1 --temperature 0 \
      --out "$OUT/pilot_control.jsonl" || exit 1
  stop_executor
  $PY -c "
import json
eps=[json.loads(l) for l in open('$OUT/pilot_control.jsonl') if l.strip()]
f1=sum(e['final_f1'] for e in eps)/len(eps)
em=sum(e['final_em'] for e in eps)/len(eps)
nz=sum(1 for e in eps if e['final_f1']>0)/len(eps)
st=sum(e['steps_used'] for e in eps)/len(eps)
print(f'baseline F1={f1:.3f}  EM={em:.3f}  nonzero={100*nz:.0f}%  steps={st:.2f}  n={len(eps)}')
print()
if f1 < 0.10:
    print('*** UNINFORMATIVE: baseline F1 < 0.10. Both arms will score ~0, so')
    print('*** dF1 ~ 0 would be trivially true and prove nothing. The negative')
    print('*** control CANNOT be run on this corpus as-is -- say so, do not claim a pass.')
else:
    print('OK: baseline is non-trivial; the negative control has power. Run: full')
"
  exit 0
fi

echo "=== FULL negative control: both arms, B=2 ==="
for SPEC in "control:$CKPT_CTRL" "treatment:$CKPT_TRT"; do
  ARM=${SPEC%%:*}; CKPT=${SPEC#*:}
  [ -s "$OUT/${ARM}.jsonl" ] && { echo "== $ARM done, skipping"; continue; }
  echo "=== serving $ARM ==="
  serve "$CKPT" || exit 1
  $PY -m collect.run_collection --task-file data/simpleqa_eval.jsonl \
      --arm a3 --mode none --budget small --g 1 --temperature 0 \
      --out "$OUT/${ARM}.jsonl.part" || exit 1
  mv "$OUT/${ARM}.jsonl.part" "$OUT/${ARM}.jsonl"
  cp "$OUT/${ARM}.jsonl" "$BACKUP/" 2>/dev/null || true
done
stop_executor
$PY scripts/t2_analyse.py --dir "$OUT" | tee "$OUT/t2_verdict.txt"
echo "=== T2 COMPLETE ==="
