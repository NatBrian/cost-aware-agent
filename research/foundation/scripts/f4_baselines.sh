#!/usr/bin/env bash
# F4: baseline runs on the frozen dev-200 (needs executor + retrieval servers up;
# GPU ritual per CLAUDE.md). A0 once (budget recorded=large, rescored later);
# A1/A2 at all three wallets. Resumable — rerun after any crash.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DEV=data/hotpotqa_dev_200.jsonl
OUT=../../experiments/results/foundation/baselines
mkdir -p "$OUT"

$PY -m collect.run_collection --task-file $DEV --arm a0 --mode none \
    --budget large --g 1 --temperature 0 --out "$OUT/a0.jsonl"
for B in small medium large; do
  $PY -m collect.run_collection --task-file $DEV --arm a1 --mode none \
      --budget $B --g 1 --temperature 0 --out "$OUT/a1_$B.jsonl"
  $PY -m collect.run_collection --task-file $DEV --arm a2 --mode enforce \
      --budget $B --g 1 --temperature 0 --out "$OUT/a2_$B.jsonl"
done
$PY -m eval.build_rows --out "$OUT/baseline_rows.csv" "$OUT"/*.jsonl
echo "baseline rows -> $OUT/baseline_rows.csv"
