#!/usr/bin/env bash
# Control the GPU reservation monitor.
#
#   gpu_reserve_ctl.sh start    launch the supervisor, detached
#   gpu_reserve_ctl.sh status   print the status file and the live nvidia-smi view
#   gpu_reserve_ctl.sh log      tail the log
#   gpu_reserve_ctl.sh stop     release every card and stop the monitor
#   gpu_reserve_ctl.sh handoff  release the cards, then print CUDA_VISIBLE_DEVICES
#
# Use `handoff` before a training run: the monitor holds most of the memory of
# each card, so it must let go before the training job can allocate. handoff
# prints the device list so the next command can use it at once.
set -uo pipefail

STATE_DIR="${GPU_RESERVE_STATE_DIR:-/mnt/src/liangsheng/cassi_foundation/gpu_reserve}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUP="$HERE/gpu_reserve_supervisor.sh"
LOG="$STATE_DIR/reserve.log"
STATUS="$STATE_DIR/status.json"
READY="$STATE_DIR/READY"
SUP_PID="$STATE_DIR/supervisor.pid"

case "${1:-status}" in

  start)
    if [ -f "$SUP_PID" ] && kill -0 "$(cat "$SUP_PID")" 2>/dev/null; then
      echo "already running (pid $(cat "$SUP_PID"))"; exit 0
    fi
    mkdir -p "$STATE_DIR"
    rm -f "$STATE_DIR/STOP"
    setsid nohup "$SUP" >/dev/null 2>&1 < /dev/null &
    sleep 3
    echo "started; supervisor pid $(cat "$SUP_PID" 2>/dev/null || echo '?')"
    ;;

  status)
    if [ -f "$SUP_PID" ] && kill -0 "$(cat "$SUP_PID")" 2>/dev/null; then
      echo "supervisor: RUNNING (pid $(cat "$SUP_PID"))"
    else
      echo "supervisor: NOT RUNNING"
    fi
    echo "--- status.json ---"
    cat "$STATUS" 2>/dev/null || echo "(no status file yet)"
    if [ -f "$READY" ]; then
      echo "--- READY: CUDA_VISIBLE_DEVICES=$(cat "$READY") ---"
    fi
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
               --format=csv,noheader 2>/dev/null
    ;;

  log)
    tail -n "${2:-40}" "$LOG" 2>/dev/null || echo "(no log yet)"
    ;;

  stop)
    touch "$STATE_DIR/STOP"
    if [ -f "$SUP_PID" ] && kill -0 "$(cat "$SUP_PID")" 2>/dev/null; then
      kill -TERM "$(cat "$SUP_PID")"
      for _ in $(seq 1 40); do
        kill -0 "$(cat "$SUP_PID" 2>/dev/null)" 2>/dev/null || break
        sleep 1
      done
    fi
    rm -f "$READY"
    echo "stopped; cards released"
    ;;

  handoff)
    if [ ! -f "$READY" ]; then
      echo "not holding the full target yet; see: $0 status" >&2; exit 1
    fi
    devs="$(cat "$READY")"
    touch "$STATE_DIR/STOP"
    [ -f "$SUP_PID" ] && kill -TERM "$(cat "$SUP_PID")" 2>/dev/null
    for _ in $(seq 1 40); do
      kill -0 "$(cat "$SUP_PID" 2>/dev/null)" 2>/dev/null || break
      sleep 1
    done
    echo "export CUDA_VISIBLE_DEVICES=$devs"
    ;;

  *)
    echo "usage: $0 {start|status|log|stop|handoff}" >&2; exit 2 ;;
esac
