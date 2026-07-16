#!/usr/bin/env bash
# =============================================================================
# launch_grpo.sh — assemble + launch one CASSI executor-GRPO run on verl
# (paper_plan_v2 §10 Algorithm 3, §16 P5/P6/P7, §17 executor.grpo, §19 stack).
#
# verl at pin 7aed6b2 launches through ray, not torchrun: the python entry
# (cassi.executor.train_grpo) composes verl's hydra ppo_trainer config, does
# ray.init with the CASSI runtime env (PYTHONPATH + CASSI_GRPO_SIDECAR), and
# hands off to verl.trainer.main_ppo.run_ppo, which spawns the TaskRunner and
# worker actors on this node's GPUs. So this script's job is: prerequisites,
# GPU acquisition (CLAUDE.md protocol), and the single trainer invocation.
#
# Usage:
#   bash scripts/launch_grpo.sh --domain qa --tasks f1.jsonl,f2.jsonl \
#        --coach experiments/stopper/round0 --arm shaped --lambda 1.0 \
#        --seed 42 --out experiments/grpo/iter1/qa [--gpus 8] [--max-steps N] \
#        [--iteration 1] [--init <ckpt>] [--step-credit per_step_rtg] \
#        [--retriever-url http://127.0.0.1:8000/retrieve] [--dry-run]
#
# The phase scripts (p5_killswitch.sh / p6_grpo_iter1.sh / p7_loop_iter2.sh)
# may call this instead of invoking train_grpo directly.
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ------------------------------------------------------------------ arguments
GPUS=8                       # §16 P0 note: 4-8 for executor GRPO
DRY_RUN=0
PASSTHRU=()                  # forwarded verbatim to cassi.executor.train_grpo
DOMAIN="qa"
TASKS=""
RETRIEVER_URL="${CASSI_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)          GPUS="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --domain)        DOMAIN="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
        --tasks)         TASKS="$2"; PASSTHRU+=("$1" "$2"); shift 2 ;;
        --retriever-url) RETRIEVER_URL="$2"; shift 2 ;;
        --coach|--arm|--lambda|--seed|--out|--max-steps|--iteration|--init|--step-credit|--vllm-url|--config)
                         PASSTHRU+=("$1" "$2"); shift 2 ;;
        *) pending "unknown argument '$1' — see the usage block in $0" ;;
    esac
done
PASSTHRU+=(--retriever-url "${RETRIEVER_URL}")

banner "launch_grpo — domain=${DOMAIN} gpus=${GPUS} dry_run=${DRY_RUN}"

# -------------------------------------------------------------- prerequisites
require_impl executor/train_grpo.py "executor agent (Algorithm 3)"
require_impl executor/verl_hooks.py "executor agent (P6 verl wiring)"
require_pins
require_module_cli cassi.executor.train_grpo "P6 trainer entry"

# `import verl` must resolve to the PINNED third_party/verl (7aed6b2, v0.8.x
# line) — p0_setup.sh installs verl-agent editable LAST, and both distributions
# claim the `verl` package name, so the fork can shadow the pin (train_grpo's
# _VERL_HELP documents the fix).
python3 - <<'PY' || pending "verl import resolves to the wrong checkout — run: pip install --no-deps -e third_party/verl"
import verl, pathlib, sys
p = pathlib.Path(verl.__file__).resolve()
assert "third_party/verl/" in str(p).replace("\\", "/"), f"verl resolves to {p}"
PY

if [ "${DRY_RUN}" = "1" ]; then
    banner "launch_grpo — dry-run (CPU-safe; no GPUs acquired, no retriever needed)"
    python -m cassi.executor.train_grpo --dry-run "${PASSTHRU[@]}"
    exit 0
fi

# Past-P2 gates: launch (not dry-run) needs pilot-frozen wallets + the coach.
require_pilot_calibration "${DOMAIN}"
[ -n "${TASKS}" ] || pending "--tasks is required to launch (P1 jsonl outputs)"

# Retriever must answer before we burn GPU-days (§16 P1; qa domain only).
if [ "${DOMAIN}" = "qa" ]; then
    curl -sf -m 5 -X POST "${RETRIEVER_URL}" \
        -H 'Content-Type: application/json' \
        -d '{"queries": ["healthcheck"], "topk": 1}' >/dev/null 2>&1 \
        || pending "Search-R1 retriever not answering at ${RETRIEVER_URL} — run scripts/p1_data.sh first"
fi

# ----------------------------------------------------------------- GPU launch
[ "${GPUS}" -ge 4 ] && [ "${GPUS}" -le 8 ] || pending "executor GRPO wants 4-8 GPUs (got ${GPUS}) — §16 P0"
acquire_gpus "${GPUS}"        # sets CUDA_VISIBLE_DEVICES; release trapped on EXIT

export CASSI_RETRIEVER_URL="${RETRIEVER_URL}"
# Stopper V̂ serving device for verl_hooks.StopperValueService (default cpu;
# TODO(GPU): move to a training-GPU slice if CPU V̂ latency dominates step time).
export CASSI_STOPPER_DEVICE="${CASSI_STOPPER_DEVICE:-cpu}"

# TODO(GPU): the actual training run (~1-3 days on 8xH200 per §7). Everything
# above is real; this is the verl trainer invocation (ray is initialized
# in-process by train_grpo with the CASSI runtime env — no torchrun, §19).
python -m cassi.executor.train_grpo "${PASSTHRU[@]}"

release_gpus
banner "launch_grpo DONE"
