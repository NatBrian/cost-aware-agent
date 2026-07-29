#!/usr/bin/env bash
# Serve Qwen3.6-27B locally as the judge, replacing the shared remote server.
#
# WHY: the judge at 122.11.227.227:6101 is a machine we do not control, and it
# went down mid-round on 2026-07-29, stopping the ablation. Serving it ourselves
# removes the last external dependency in the pipeline.
#
# CRITICAL — `--served-model-name Qwen3.6-27B` must match the remote server's
# model id exactly. The judge cache key includes the model name (audit
# 2026-07-28), so keeping the id identical means every judgement already cached
# from the remote server stays valid and is reused. Change the name and you
# silently re-judge ~18,000 steps and, worse, mix two judges' scores inside one
# experiment.
#
# Shares GPU 0 with the retrieval server: 27B bf16 is ~54G, retrieval uses ~2G,
# and the card has 143G. The executor keeps GPU 1 for training/serving.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=/mnt/src/liangsheng/cassi_foundation/models/Qwen3.6-27B
PORT=${1:-6101}
GPU=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold | cut -d, -f1)
[ -f "$MODEL/config.json" ] || { echo "judge weights not downloaded yet: $MODEL"; exit 1; }
echo "[judge] serving $MODEL on port $PORT, gpu ${GPU:-0}"
CUDA_VISIBLE_DEVICES=${GPU:-0} exec .venv-gpu3/bin/vllm serve "$MODEL" \
  --served-model-name Qwen3.6-27B \
  --port "$PORT" --dtype auto --max-model-len 32768 \
  --gpu-memory-utilization 0.55 "${@:2}"
