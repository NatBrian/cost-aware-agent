#!/usr/bin/env bash
# Choose which trained checkpoint goes to the dev-200 evaluation.
#
# WHY THIS EXISTS: "use the last round" is only safe if the last round is the
# best round. After round 3 the temp-1.0 probe showed malformed 2.0% -> 9.0% and
# hit_cap 0.0% -> 2.5% — passing the gate, but trending toward exactly the
# sampling-distribution damage that wasted the first run. Picking a checkpoint is
# a tuning decision, so it reads the VALIDATION slice (val-50), never dev-200
# (anti-overfitting policy, PROGRESS.md).
#
# Selection metric is UTILITY, not F1: U = F1 - lambda*(steps/B) is what the run
# optimises, and choosing on F1 alone would reward exactly the over-continuation
# the method is supposed to cure.
#
# Usage: bash scripts/select_checkpoint.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=experiments/results/select
mkdir -p "$OUT"
GPU=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold | cut -d, -f2)

for R in 1 2 3; do
  CKPT="$PWD/experiments/results/train/round$R/checkpoint"
  [ -d "$CKPT" ] || { echo "skip round$R (no checkpoint)"; continue; }
  echo "== serving round$R =="
  PID=$(pgrep -f "[v]llm serve" | head -1) && [ -n "$PID" ] && kill "$PID" && sleep 20 || true
  CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv-gpu3/bin/vllm serve "$CKPT" \
    --served-model-name Qwen/Qwen3.5-9B --port 8378 --dtype auto \
    --max-model-len 16384 --gpu-memory-utilization 0.85 \
    > "$OUT/serve_r$R.log" 2>&1 &
  for i in $(seq 1 80); do
    curl -s -m 5 http://127.0.0.1:8378/v1/chat/completions -H "Content-Type: application/json" \
      -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
      2>/dev/null | grep -q '"content"' && break
    sleep 15
    [ "$i" = 80 ] && { echo "round$R failed to serve"; exit 1; }
  done
  echo "== val-50 @ B=medium, temp 0, harness off =="
  rm -f "$OUT/val_r$R.jsonl"
  $PY -m collect.run_collection --task-file data/hotpotqa_val_50.jsonl \
      --arm a3 --mode none --budget medium --g 1 --temperature 0 \
      --out "$OUT/val_r$R.jsonl"
done

echo
echo "== SELECTION (validation slice, utility = F1 - lambda*steps/B) =="
$PY - <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from common import load_config
cfg = load_config(); lam = cfg["economy"]["lambda"]
B = cfg["episode"]["budgets"]["medium"]
best = None
for r in (1, 2, 3):
    p = Path(f"experiments/results/select/val_r{r}.jsonl")
    if not p.exists():
        continue
    eps = [json.loads(l) for l in open(p) if l.strip()]
    n = len(eps)
    f1 = sum(e["final_f1"] for e in eps) / n
    steps = sum(e["steps_used"] for e in eps) / n
    u = f1 - lam * (steps / B)
    stop = sum(1 for e in eps if e.get("answered_at") is not None
               and not e["forced_stop"] and e["answered_at"] <= B) / n
    mal = sum(1 for e in eps for s in e["steps"] if s["action_type"] == "malformed")
    tot = sum(len(e["steps"]) for e in eps)
    print(f"round{r}: n={n} U={u:.4f} F1={f1:.4f} steps={steps:.2f} "
          f"self_stop={stop:.2f} malformed={mal/max(1,tot):.1%}")
    if best is None or u > best[1]:
        best = (r, u)
print(f"\nSELECTED: round{best[0]} (highest validation utility U={best[1]:.4f})")
Path("experiments/results/select/selected.txt").write_text(str(best[0]))
PYEOF
