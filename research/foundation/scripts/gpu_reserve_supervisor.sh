#!/usr/bin/env bash
# Keeps gpu_reserve.py alive across crashes, and across Claude sessions.
#
# Start it detached, so it survives the terminal that launched it:
#   setsid nohup .../gpu_reserve_supervisor.sh >/dev/null 2>&1 < /dev/null &
#
# Stop it with gpu_reserve_ctl.sh stop, which creates the STOP file and sends
# SIGTERM. The trap below forwards SIGTERM to gpu_reserve.py, so every card is
# released cleanly instead of being stranded by a held CUDA context.
#
# Only one copy can run: the flock below makes a second copy exit at once.
set -uo pipefail

STATE_DIR="${GPU_RESERVE_STATE_DIR:-/mnt/src/liangsheng/cassi_foundation/gpu_reserve}"
NUM_GPUS="${GPU_RESERVE_NUM:-2}"
STABLE_SECONDS="${GPU_RESERVE_STABLE:-300}"
POLL_INTERVAL="${GPU_RESERVE_POLL:-30}"
HOLD_FRAC="${GPU_RESERVE_HOLD_FRAC:-0.90}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESERVE="$HERE/gpu_reserve.py"
mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/reserve.log"
STOP="$STATE_DIR/STOP"
SUP_PID="$STATE_DIR/supervisor.pid"
SUP_LOCK="$STATE_DIR/supervisor.lock"

exec 200>"$SUP_LOCK"
if ! flock -n 200; then
  echo "supervisor already running (lock $SUP_LOCK held); this copy exits" >&2
  exit 0
fi

rm -f "$STOP"
echo $$ > "$SUP_PID"

child=""
shutdown() {
  touch "$STOP"
  if [ -n "$child" ]; then
    echo "[$(date '+%F %T')] supervisor: SIGTERM -> gpu_reserve.py ($child)" >> "$LOG"
    kill -TERM "$child" 2>/dev/null
    # Give the CUDA contexts time to close before the supervisor exits.
    for _ in $(seq 1 30); do
      kill -0 "$child" 2>/dev/null || break
      sleep 1
    done
  fi
  echo "[$(date '+%F %T')] supervisor: stopped" >> "$LOG"
  rm -f "$SUP_PID"
  exit 0
}
trap shutdown TERM INT

echo "[$(date '+%F %T')] supervisor: start (pid $$), want ${NUM_GPUS} GPU(s), \
stable ${STABLE_SECONDS}s, poll ${POLL_INTERVAL}s, hold_frac ${HOLD_FRAC}" >> "$LOG"

while [ ! -f "$STOP" ]; do
  python3 "$RESERVE" \
    --num-gpus "$NUM_GPUS" \
    --poll-interval "$POLL_INTERVAL" \
    --stable-seconds "$STABLE_SECONDS" \
    --hold-frac "$HOLD_FRAC" \
    --state-dir "$STATE_DIR" >> "$LOG" 2>&1 &
  child=$!
  wait "$child"
  rc=$?
  child=""
  [ -f "$STOP" ] && break
  echo "[$(date '+%F %T')] supervisor: gpu_reserve.py exited rc=$rc; \
restart in 60s" >> "$LOG"
  # Sleep in slices so a stop request is noticed quickly.
  for _ in $(seq 1 60); do
    [ -f "$STOP" ] && break
    sleep 1
  done
done

echo "[$(date '+%F %T')] supervisor: STOP file present; exit" >> "$LOG"
rm -f "$SUP_PID"
