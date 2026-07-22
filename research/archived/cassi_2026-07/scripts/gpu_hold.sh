#!/usr/bin/env bash
# =============================================================================
# gpu_hold.sh — keep acquired GPUs held BETWEEN experiment phases.
#
# Why: scripts/common.sh acquire_gpus() releases on EXIT, so every phase gives
# the cards back and the next phase races other users to re-acquire. This
# script starts a tiny detached "holder" process that owns the zhanka locks
# (lock validity is tied to a live PID — gpu_acquire.sh reaps locks whose PID
# died) and keeps them until an explicit `stop`. Phase scripts then REUSE the
# held cards (common.sh detects an active hold and skips acquire/release).
#
# Safety properties (deliberate):
#   * acquisition goes ONLY through /mnt/src/zhanka/gpu_acquire.sh — flock'd,
#     skips other people's locks and busy GPUs, never preempts, never kills;
#   * the hold is VISIBLE to everyone (standard /tmp/gpu_lock_<id> files with
#     our username + the holder PID) — the normal etiquette mechanism;
#   * `stop` releases ONLY the ids this hold owns (explicit args to
#     gpu_release.sh) — NEVER the bare user-wide release: the OS account is
#     shared with another person whose locks a bare release would also drop;
#   * the holder itself is just `sleep infinity` — zero GPU/CPU footprint.
#
# Usage:
#   scripts/gpu_hold.sh start N     # acquire N GPUs and keep them held
#   scripts/gpu_hold.sh status      # what is held, is the holder alive
#   scripts/gpu_hold.sh stop        # release the held GPUs (explicit ids)
# =============================================================================
set -euo pipefail

GPU_ACQUIRE=/mnt/src/zhanka/gpu_acquire.sh
GPU_RELEASE=/mnt/src/zhanka/gpu_release.sh
PID_FILE=/tmp/cassi_gpu_hold.pid
DEV_FILE=/tmp/cassi_gpu_hold.devices
LOG_FILE=/tmp/cassi_gpu_hold.log

hold_alive() {
    [ -f "${PID_FILE}" ] || return 1
    local pid; pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null
}

cmd_start() {
    local n="${1:?usage: gpu_hold.sh start <N>}"
    if hold_alive; then
        echo "Already holding GPUs $(cat "${DEV_FILE}" 2>/dev/null || echo '?') (holder pid $(cat "${PID_FILE}")). Use 'stop' first." >&2
        exit 1
    fi
    rm -f "${PID_FILE}" "${DEV_FILE}"; : > "${LOG_FILE}"
    # Holder: acquire runs as a DIRECT child of this shell, so the lock's
    # recorded PID ($PPID inside gpu_acquire.sh) is the holder's own live PID.
    # The output is PARSED, never eval'd — the zhanka script prints non-shell
    # lines ("Ready: ...") on stdout that break eval (verified 2026-07-21).
    # exec sleep keeps the PID alive with zero footprint.
    setsid bash -c '
        echo $$ > '"${PID_FILE}"'
        bash '"${GPU_ACQUIRE}"' '"${n}"' > '"${LOG_FILE}"'.out 2>> '"${LOG_FILE}"'
        devs=$(sed -n "s/^export CUDA_VISIBLE_DEVICES=//p" '"${LOG_FILE}"'.out | tail -1)
        if [ -n "${devs}" ]; then
            echo "${devs}" > '"${DEV_FILE}"'
            exec sleep infinity
        else
            rm -f '"${PID_FILE}"'
            exit 1
        fi
    ' >> "${LOG_FILE}" 2>&1 &
    # Wait for acquire to succeed (it may wait up to ~60s for memory to free).
    # Break early ONLY on a confirmed-dead holder (PID file present, pid gone) —
    # never on a missing PID file, which just means the holder hasn't written
    # it yet (that race caused a premature-failure + state-file cleanup while
    # the real acquire was still running, 2026-07-21).
    for _ in $(seq 1 45); do
        [ -s "${DEV_FILE}" ] && break
        if [ -f "${PID_FILE}" ] && ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
            break   # holder started and died → acquire genuinely failed
        fi
        sleep 2
    done
    if [ -s "${DEV_FILE}" ] && hold_alive; then
        echo "HOLDING GPUs $(cat "${DEV_FILE}") (holder pid $(cat "${PID_FILE}"); locks: /tmp/gpu_lock_*)."
        echo "Phase scripts will reuse them automatically. Release with: scripts/gpu_hold.sh stop"
    else
        rm -f "${PID_FILE}" "${DEV_FILE}"
        echo "HOLD FAILED — acquire did not succeed (not enough free GPUs?). Log:" >&2
        tail -5 "${LOG_FILE}" >&2 || true
        exit 1
    fi
}

cmd_status() {
    if hold_alive && [ -s "${DEV_FILE}" ]; then
        local devs; devs="$(cat "${DEV_FILE}")"
        echo "HOLDING: CUDA_VISIBLE_DEVICES=${devs} (holder pid $(cat "${PID_FILE}"))"
        for gid in ${devs//,/ }; do
            printf "  gpu %s lock: %s\n" "${gid}" "$(cat "/tmp/gpu_lock_${gid}" 2>/dev/null || echo 'MISSING — lock lost!')"
        done
    else
        echo "No active hold."
        rm -f "${PID_FILE}" "${DEV_FILE}" 2>/dev/null || true
    fi
}

cmd_stop() {
    if ! hold_alive && [ ! -s "${DEV_FILE}" ]; then
        echo "No active hold to stop."; rm -f "${PID_FILE}" "${DEV_FILE}"; exit 0
    fi
    local devs=""; [ -s "${DEV_FILE}" ] && devs="$(cat "${DEV_FILE}")"
    if [ -f "${PID_FILE}" ]; then
        kill "$(cat "${PID_FILE}")" 2>/dev/null || true
    fi
    if [ -n "${devs}" ]; then
        # Explicit ids ONLY — never the bare user-wide release (shared account)
        # shellcheck disable=SC2086
        "${GPU_RELEASE}" ${devs//,/ } || true
    fi
    rm -f "${PID_FILE}" "${DEV_FILE}"
    echo "Hold stopped; GPUs ${devs:-<none>} released (occupier will re-take them)."
}

case "${1:-}" in
    start)  cmd_start "${2:?usage: gpu_hold.sh start <N>}" ;;
    status) cmd_status ;;
    stop)   cmd_stop ;;
    *)      echo "usage: gpu_hold.sh {start N|status|stop}" >&2; exit 1 ;;
esac
