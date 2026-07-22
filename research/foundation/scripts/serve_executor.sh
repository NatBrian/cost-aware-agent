#!/usr/bin/env bash
# Serve the Qwen3.5-9B executor via vLLM (OpenAI-compatible, port from config).
# GPU ritual (CLAUDE.md): eval $(/mnt/src/zhanka/gpu_acquire.sh 2)  ...run...
#                         /mnt/src/zhanka/gpu_release.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$(.venv/bin/python -c "from common import load_config; print(load_config()['executor']['model'])")
PORT=$(.venv/bin/python -c "from common import load_config; print(load_config()['executor']['endpoint'].rsplit(':',1)[1].split('/')[0])")
exec vllm serve "$MODEL" --port "$PORT" --dtype auto --max-model-len 16384 "$@"
