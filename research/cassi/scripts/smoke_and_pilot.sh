#!/usr/bin/env bash
# =============================================================================
# smoke_and_pilot.sh — the FIRST GPU session (P0 done-criterion + P2 pilot).
#
# Everything is pre-staged (2026-07-16): models (Qwen3.5-9B/2B in HF cache),
# wiki-18 corpus + assembled e5_Flat.index (data/searchr1_index/), datasets +
# decontamination + manifest (P1), pinned stack in .venv.
#
# Run when GPUs are actually free. IMPORTANT FOOTGUN: gpu_acquire.sh can grant
# LOCKS while a foreign job still holds the memory (it warns "timeout waiting").
# This script verifies memory really freed and aborts otherwise — never kill
# other users' GPU processes (CLAUDE.md).
#
# Sequence: acquire 2 GPUs → retriever server (GPU a) → vLLM Qwen3.5-9B (GPU b)
#           → P0 smoke rollout + verify → 200-task unconstrained pilot (P2)
#           → print wallet calibration to freeze into configs/cassi.yaml.
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

banner "smoke_and_pilot — P0 smoke + P2 pilot"

acquire_gpus 2
IFS=',' read -r GPU_A GPU_B <<< "${CUDA_VISIBLE_DEVICES}"

# --------- verify the memory ACTUALLY freed (locks != free memory, see header)
for g in ${GPU_A} ${GPU_B}; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${g}")
    if [ "${USED}" -gt 2000 ]; then
        release_gpus
        pending "GPU ${g} still holds ${USED} MiB (foreign job — check 'nvidia-smi'; never kill it). Retry when the machine is free."
    fi
done

mkdir -p "${EXP_DIR}/smoke" "${EXP_DIR}/pilot" "${EXP_DIR}/logs"

# ------------------------------------------------------- retriever (GPU_A)
banner "launch retriever server (E5, GPU ${GPU_A})"
( cd "${THIRD_PARTY}/Search-R1" && \
  CUDA_VISIBLE_DEVICES="${GPU_A}" nohup "${CASSI_VENV}/bin/python" \
    search_r1/search/retrieval_server.py \
    --index_path "${DATA_DIR}/searchr1_index/e5_Flat.index" \
    --corpus_path "${DATA_DIR}/searchr1_index/wiki-18.jsonl" \
    --retriever_name e5 --retriever_model intfloat/e5-base-v2 --topk 3 \
    --port 8000 > "${EXP_DIR}/logs/retriever.log" 2>&1 & echo $! > "${EXP_DIR}/retriever.pid" )
echo "retriever pid $(cat "${EXP_DIR}/retriever.pid") — index load takes several minutes (64GB)"

# --------------------------------------------------------- vLLM (GPU_B)
banner "launch vLLM Qwen3.5-9B (GPU ${GPU_B})"
CUDA_VISIBLE_DEVICES="${GPU_B}" nohup "${CASSI_VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$(cfg_get executor.base_model)" --port 8001 --gpu-memory-utilization 0.85 \
    > "${EXP_DIR}/logs/vllm.log" 2>&1 & echo $! > "${EXP_DIR}/vllm.pid"
echo "vllm pid $(cat "${EXP_DIR}/vllm.pid")"

cleanup() {
    kill "$(cat "${EXP_DIR}/vllm.pid" 2>/dev/null)" 2>/dev/null || true
    kill "$(cat "${EXP_DIR}/retriever.pid" 2>/dev/null)" 2>/dev/null || true
    release_gpus
}
trap cleanup EXIT

# ------------------------------------------------ wait for both servers
banner "waiting for servers"
for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:8001/v1/models" >/dev/null 2>&1 && VOK=1 || VOK=0
    curl -sf -X POST "http://127.0.0.1:8000/retrieve" -H 'Content-Type: application/json' \
         -d '{"queries":["test"],"topk":1}' >/dev/null 2>&1 && ROK=1 || ROK=0
    [ "${VOK}${ROK}" = "11" ] && break
    sleep 10
done
[ "${VOK}${ROK}" = "11" ] || { tail -20 "${EXP_DIR}/logs/vllm.log" "${EXP_DIR}/logs/retriever.log"; pending "servers did not come up in 20 min — see logs above"; }
echo "both servers up"

# ------------------------------------------------- P0 smoke (done-criterion)
banner "P0 smoke rollout"
python3 "${CASSI_ROOT}/scripts/make_task_file.py" \
    --in "${DATA_DIR}/hotpotqa_dev.jsonl" --out "${EXP_DIR}/smoke/qa_task1.jsonl" --n 1 --seed 42
"${CASSI_VENV}/bin/python" -m cassi.executor.collect --smoke --domain qa \
    --tasks "${EXP_DIR}/smoke/qa_task1.jsonl" --G 1 --seed 42 \
    --vllm-url http://127.0.0.1:8001/v1 --retriever-url http://127.0.0.1:8000/retrieve \
    --out "${EXP_DIR}/smoke/qa.jsonl"
python3 "${CASSI_ROOT}/scripts/verify_smoke.py" "${EXP_DIR}/smoke/qa.jsonl"
banner "P0 SMOKE PASSED (qa)"

# ------------------------------------------------------ P2 pilot (200 tasks)
banner "P2 pilot — 200 tasks, unconstrained (calibrates wallets, §16 P2)"
python3 "${CASSI_ROOT}/scripts/make_task_file.py" \
    --in "${DATA_DIR}/hotpotqa_train.decontaminated.jsonl" \
    --out "${EXP_DIR}/pilot/tasks200.jsonl" --n 200 --seed 42
"${CASSI_VENV}/bin/python" -m cassi.executor.collect --pilot --domain qa \
    --tasks "${EXP_DIR}/pilot/tasks200.jsonl" --seed 42 \
    --vllm-url http://127.0.0.1:8001/v1 --retriever-url http://127.0.0.1:8000/retrieve \
    --out "${EXP_DIR}/pilot/spends_qa.txt" | tee "${EXP_DIR}/pilot/calibration_qa.json"

banner "DONE — freeze the printed calibration into configs/cassi.yaml (label.allowances.qa"
echo "+ label.cost_normalization.qa_median_pilot_spend), commit, then p2_pilot_and_collect.sh"
echo "runs the full round-0 collection. ALFWorld smoke/pilot: stage the verl-agent task"
echo "list (P1.3) and repeat with --domain alfworld."
