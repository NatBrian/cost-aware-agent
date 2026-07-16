#!/usr/bin/env bash
# =============================================================================
# P0 — Environment setup (paper_plan_v2 §16 P0, week 1)
#
# Goal: clone + pin the reuse stack (§19 manifest) into third_party/ (gitignored),
# install the GPU requirements, record every commit hash / lib version into
# configs/cassi.yaml `pins:`, and print the smoke-run recipe.
#
# Done-criterion (§16 P0, quoted):
#   "✅ Done: one ReAct rollout runs end-to-end on HotpotQA dev and one on
#    ALFWorld, with the running-draft template line present in every step and
#    per-step cost logging into the trajectory JSONL schema (§11)."
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

banner "P0 — environment setup"

# ---------------------------------------------------------------- prerequisites
require_cmd git "needed to clone the §19 reuse stack"
require_cmd python3 "needed for pin recording and smoke verification"
require_cmd pip "needed to install requirements-gpu.txt"
git ls-remote --exit-code https://github.com/verl-project/verl HEAD >/dev/null 2>&1 \
    || pending "requires network access to github.com (clone of the §19 stack)"

mkdir -p "${THIRD_PARTY}" "${EXP_DIR}/results"
touch "${EXP_DIR}/results/.gitkeep"

# --------------------------------------------------- 1. clone + pin (§19 reuse)
# "clone, pin, do not modify internally — our additions are wrapper hooks" (§19)
clone_pin() {  # clone_pin <url> <dir> [<preferred_ref>]
    local url="$1" dir="${THIRD_PARTY}/$2" ref="${3:-}"
    if [ -d "${dir}/.git" ]; then
        echo "[clone] $2 already present — keeping existing checkout (pins record its HEAD)"
    else
        git clone --recurse-submodules "${url}" "${dir}"
        if [ -n "${ref}" ]; then
            git -C "${dir}" checkout "${ref}" \
                || echo "[clone] WARNING: ref '${ref}' not found in $2 — staying on default branch"
        fi
    fi
    echo "[pin ] $2 @ $(git -C "${dir}" rev-parse HEAD)"
}

# verl >= v0.8.0 (GRPO backend + AgentLoop multi-turn; Dr.GRPO/DAPO hygiene — §7/§19).
# Checkout the newest v0.8.x tag if one exists, else stay on default branch and verify >= 0.8.0.
clone_pin https://github.com/verl-project/verl            verl
VERL_TAG="$(git -C "${THIRD_PARTY}/verl" tag -l 'v0.8*' --sort=-v:refname | head -n1 || true)"
if [ -n "${VERL_TAG}" ]; then
    git -C "${THIRD_PARTY}/verl" checkout "${VERL_TAG}"
    echo "[pin ] verl checked out at tag ${VERL_TAG} (>= v0.8.0 required, §19)"
else
    echo "[pin ] WARNING: no v0.8.x tag found in verl — verify the default branch is >= v0.8.0 (§19)"
fi
clone_pin https://github.com/TIGER-AI-Lab/verl-tool       verl-tool     # Search-R1 env on modern verl (§19)
clone_pin https://github.com/langfengQ/verl-agent         verl-agent    # ALFWorld / GiGPO official harness (§19)
clone_pin https://github.com/PeterGriffinJin/Search-R1    Search-R1     # retriever recipe + data (§19)

# ------------------------------------------- 2. install requirements-gpu.txt
# §19 stack pins for Qwen3.5: transformers v5, vLLM >= 0.17 (GDN kernels), TRL v1.8
# scalar head. flash-attn needs the CUDA toolchain; if its build fails, install it
# separately with --no-build-isolation and rerun this script.
banner "P0.2 — pip install requirements-gpu.txt"
pip install -r "${CASSI_ROOT}/requirements-gpu.txt" \
    || pending "pip install of requirements-gpu.txt failed (flash-attn usually needs 'pip install flash-attn --no-build-isolation' on this machine) — fix and rerun"
# Editable installs of the cloned stack (kept unmodified internally — §19):
pip install -e "${THIRD_PARTY}/verl"       --no-deps
pip install -e "${THIRD_PARTY}/verl-tool"  --no-deps || echo "[pip] verl-tool editable install failed — check its README for extras"
pip install -e "${THIRD_PARTY}/verl-agent" --no-deps || echo "[pip] verl-agent editable install failed — check its README for ALFWorld extras (alfworld, textworld)"

# ------------------------------------- 3. record pins into configs/cassi.yaml
banner "P0.3 — write commit hashes + lib versions into configs/cassi.yaml pins:"
python3 "${CASSI_ROOT}/scripts/update_pins.py" \
    || echo "[pins] some pins unresolved — see messages above; rerun after fixing"

# ------------------------------------------------------- 4. smoke-run recipe
# The rollout engine (cassi/executor/collect.py + envs/) is being added by the
# executor agent; the P0 done-criterion is exercised through it.
banner "P0.4 — smoke run (P0 done-criterion)"
cat <<EOF
The P0 done-criterion (§16 P0) is verified by ONE ReAct rollout per domain
(collect.py CLI: --tasks {task_id, question, gold} JSONL; G/T_max from config).
Prereqs: vLLM serving Qwen3.5-9B (--vllm-url) + the P1 retriever server
(--retriever-url; BM25 mode is fine for a smoke test):

  eval \$(${GPU_ACQUIRE} 2)     # N=2: collection-class job (§16 P0 note)
  python3 scripts/make_task_file.py --in ${DATA_DIR}/hotpotqa_dev.jsonl \\
      --out ${EXP_DIR}/smoke/qa_task1.jsonl --n 1 --seed 42
  python -m cassi.executor.collect --domain qa --tasks ${EXP_DIR}/smoke/qa_task1.jsonl \\
      --G 1 --seed 42 --out ${EXP_DIR}/smoke/qa.jsonl
  python -m cassi.executor.collect --domain alfworld --tasks <alfworld task list, P1.3> \\
      --G 1 --seed 42 --out ${EXP_DIR}/smoke/alfworld.jsonl
  ${GPU_RELEASE}

Then verify draft line + per-step cost in the §11 JSONL schema:

  python3 scripts/verify_smoke.py ${EXP_DIR}/smoke/qa.jsonl ${EXP_DIR}/smoke/alfworld.jsonl
EOF

if [ ! -f "${CASSI_ROOT}/executor/collect.py" ] || [ ! -f "${DATA_DIR}/hotpotqa_dev.jsonl" ]; then
    echo
    echo "TODO(GPU): smoke prerequisites missing (executor/collect.py from the executor"
    echo "           agent, and/or P1 data) — run the commands above once they exist."
    echo "           Setup steps 1-3 themselves are done."
    exit 0
fi
if command -v nvidia-smi >/dev/null 2>&1 && [ -x "${GPU_ACQUIRE}" ]; then
    acquire_gpus 2
    mkdir -p "${EXP_DIR}/smoke"
    python3 "${CASSI_ROOT}/scripts/make_task_file.py" \
        --in "${DATA_DIR}/hotpotqa_dev.jsonl" --out "${EXP_DIR}/smoke/qa_task1.jsonl" --n 1 --seed 42
    # TODO(GPU): requires a running vLLM server + retriever server (see the recipe above)
    python -m cassi.executor.collect --domain qa --tasks "${EXP_DIR}/smoke/qa_task1.jsonl" \
        --G 1 --seed 42 --out "${EXP_DIR}/smoke/qa.jsonl"
    echo "TODO(GPU): ALFWorld smoke rollout once the verl-agent task list is staged (P1.3)"
    release_gpus
    python3 "${CASSI_ROOT}/scripts/verify_smoke.py" "${EXP_DIR}/smoke/qa.jsonl"
    banner "P0 DONE (qa) — rerun with ALFWorld staged to complete the done-criterion"
else
    echo "PENDING: requires GPU for the smoke rollouts — setup steps 1-3 completed; run the commands above on the GPU node."
fi
