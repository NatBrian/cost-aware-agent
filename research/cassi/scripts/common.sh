#!/usr/bin/env bash
# common.sh — shared helpers for the CASSI phase scripts (paper_plan_v2 §16 runbook).
# Source me from every pN_*.sh:   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# Conventions:
#   * pending "<msg>"  — prerequisite missing: print a clear PENDING line and exit 75
#     (EX_TEMPFAIL) so callers can distinguish "rerun later" from real failure.
#     Scripts must NEVER half-run (task rule / §16: don't start a phase before the
#     previous phase's done-criterion is met).
#   * GPU protocol (CLAUDE.md + §16 P0 note): acquire_gpus N / release on EXIT trap.
#     N=2 for stopper SFT & collection, N=4–8 for executor GRPO.

CASSI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESEARCH_DIR="$(dirname "${CASSI_ROOT}")"
REPO_ROOT="$(dirname "${RESEARCH_DIR}")"

# Dedicated venv — NEVER install the GPU stack into the shared machine python.
# p0_setup.sh creates it; every later phase auto-activates it when present.
CASSI_VENV="${CASSI_ROOT}/.venv"
if [ -f "${CASSI_VENV}/bin/activate" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "${CASSI_VENV}/bin/activate"
fi

CONFIG="${CASSI_ROOT}/configs/cassi.yaml"
THIRD_PARTY="${CASSI_ROOT}/third_party"
DATA_DIR="${CASSI_ROOT}/data"
EXP_DIR="${CASSI_ROOT}/experiments"
GO_NO_GO_LOG="${CASSI_ROOT}/GO_NO_GO.log"

# `import cassi.*` resolves against research/ (see conftest.py)
export PYTHONPATH="${RESEARCH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

GPU_ACQUIRE=/mnt/src/zhanka/gpu_acquire.sh
GPU_RELEASE=/mnt/src/zhanka/gpu_release.sh
_GPUS_HELD=0

banner() { echo; echo "================================================================"; echo "== $*"; echo "================================================================"; }

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

pending() {
    echo
    echo "PENDING: $*" >&2
    echo "PENDING: nothing was half-run — rerun this script once the prerequisite is met." >&2
    exit 75
}

require_cmd() {  # require_cmd <cmd> <why>
    command -v "$1" >/dev/null 2>&1 || pending "requires command '$1' — $2"
}

require_file() {  # require_file <path> <why>
    [ -e "$1" ] || pending "missing $1 — $2"
}

# Modules being added by the other agents (stopper/train_sft.py, executor/collect.py,
# executor/train_grpo.py, baselines/, eval/, analysis/) are referenced by module path;
# gate on file existence so this script PENDs cleanly instead of import-crashing.
require_impl() {  # require_impl <relative/path.py> <who provides it>
    [ -f "${CASSI_ROOT}/$1" ] || pending "requires ${CASSI_ROOT}/$1 (provided by: $2) — not present yet"
}

# Some concurrent modules landed as LIBRARIES without a CLI (`python -m` on a
# module with no __main__ block silently no-ops). Gate on a main() before looping.
require_module_cli() {  # require_module_cli <module> <why>
    python3 -c "
import importlib, sys
try:
    m = importlib.import_module('$1')
except Exception as e:
    print(f'import failed: {e}', file=sys.stderr); sys.exit(1)
sys.exit(0 if hasattr(m, 'main') else 1)
" || pending "module '$1' has no runnable main() CLI yet — $2"
}

require_gpu_tooling() {
    [ -x "${GPU_ACQUIRE}" ] || pending "requires GPU: ${GPU_ACQUIRE} not found (CLAUDE.md GPU protocol)"
    command -v nvidia-smi >/dev/null 2>&1 || pending "requires GPU: nvidia-smi not found on this machine"
}

acquire_gpus() {  # acquire_gpus <N>
    local n="$1"
    require_gpu_tooling
    banner "GPU acquire: ${n} GPU(s) (release is trapped on EXIT)"
    # shellcheck disable=SC2046
    eval $("${GPU_ACQUIRE}" "${n}") || pending "GPU acquire failed (not enough free GPUs?) — try fewer GPUs or wait"
    _GPUS_HELD=1
    trap release_gpus EXIT
    echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
}

release_gpus() {
    if [ "${_GPUS_HELD}" = "1" ]; then
        "${GPU_RELEASE}" || true
        _GPUS_HELD=0
        echo "[gpu] released — occupier will re-take the GPUs"
    fi
}

# Read a dotted key from configs/cassi.yaml (e.g. cfg_get executor.grpo.G)
cfg_get() {
    python3 -c "
import functools, sys, yaml
cfg = yaml.safe_load(open('${CONFIG}'))
print(functools.reduce(lambda d, k: d[k], sys.argv[1].split('.'), cfg))
" "$1"
}

# §17 comment in cassi.yaml: 'scripts must refuse to proceed past P2 while
# calibration fields are null'. Wraps cassi.common.config.require_pilot_calibration.
require_pilot_calibration() {  # require_pilot_calibration <domain>
    python3 -c "
from cassi.common.config import load_config, require_pilot_calibration
require_pilot_calibration(load_config(), '$1')
" || pending "configs/cassi.yaml pilot-calibration fields are null for domain '$1' — run scripts/p2_pilot_and_collect.sh (P2) and write the printed values into configs/cassi.yaml first (§16 P2, §17)"
}

require_pins() {
    python3 -c "
import sys, yaml
pins = yaml.safe_load(open('${CONFIG}'))['pins']
missing = [k for k, v in pins.items() if v is None]
sys.exit(1 if missing else 0)
" || pending "configs/cassi.yaml 'pins:' still has null entries — run scripts/p0_setup.sh (P0) first (§16 P0 / §19)"
}
