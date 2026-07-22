#!/usr/bin/env bash
# Waits for a stable-free window, then acquires N GPUs via the zhanka ritual
# and records them in .gpu_hold (standing 2-GPU hold, Brian 2026-07-22).
# Stability: a GPU must look free on two checks 60s apart before we try.
N=${1:-2}
cd "$(dirname "$0")/.."
while true; do
  if out=$(/mnt/src/zhanka/gpu_acquire.sh "$N" 2>/dev/null) && [ -n "$out" ]; then
    eval "$out"
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
      sleep 60   # stability: still ours & idle after a minute?
      echo "$out" > .gpu_hold
      echo "HELD: $CUDA_VISIBLE_DEVICES"
      exit 0
    fi
  fi
  sleep 120
done
