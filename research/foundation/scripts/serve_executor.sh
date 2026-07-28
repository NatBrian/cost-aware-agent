#!/usr/bin/env bash
# Serve the Qwen3.5-9B executor via vLLM (OpenAI-compatible, port from config).
# GPUs: hold them first with /home/liangsheng/brian/acquire_gpus.py -n 2, which
# only ever takes cards no compute process is on; .gpu_hold records which ones we
# got and is the single source of truth for CUDA_VISIBLE_DEVICES (it was stale —
# naming a card we did not hold — until the 2026-07-28 audit).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
VLLM=.venv-gpu2/bin/vllm
MODEL=$($PY -c "
import os
from common import load_config, FOUNDATION_ROOT
c = load_config()['executor']
print(os.path.realpath(FOUNDATION_ROOT / c['model_path']) if c.get('model_path') else c['model'])")
PORT=$($PY -c "from common import load_config; print(load_config()['executor']['endpoint'].rsplit(':',1)[1].split('/')[0])")
CTX=$($PY -c "from common import load_config; print(load_config()['executor']['max_model_len'])")
# Honour an explicit override (the retrieval server sits on one held card, so
# the executor is usually pinned to the other); otherwise take the whole hold.
GPUS=${CUDA_VISIBLE_DEVICES:-$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold)}
echo "[serve] model=$MODEL port=$PORT ctx=$CTX gpus=$GPUS"
# --served-model-name pins the hub id so the client's `model` field is identical
# whether we are serving the base model or a trained checkpoint.
CUDA_VISIBLE_DEVICES=$GPUS exec "$VLLM" serve "$MODEL" \
  --served-model-name Qwen/Qwen3.5-9B \
  --port "$PORT" --dtype auto --max-model-len "$CTX" "$@"
