#!/usr/bin/env bash
# =============================================================================
# P4 — Stopper v0 (paper_plan_v2 §16 P4, week 2)
#
# Goal: Algorithm 2 SFT of the λ-conditioned three-head stopper (Qwen3.5-2B;
# action CE + Δ̂ MSE + V̂ MSE; early stop on held-out stopping REGRET, not CE),
# then evaluate held-out stopping regret vs (i) majority-class and (ii) a
# calibrated confidence probe, plus STOP/CONTINUE F1 and the RedundancyBench
# external check.
#
# Done-criterion (§16 P4, quoted) — THIS IS A GATE:
#   "✅ Done: stopper beats (i) majority-class and (ii) a calibrated confidence
#    probe on held-out stopping regret; if it cannot, STOP — fix features/labels
#    before touching RL."
#
# GPU: N=2 (stopper SFT, §16 P0 note).
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ROUND="${1:-0}"
LABEL_DIR="${EXP_DIR}/labels/round${ROUND}"
STOPPER_DIR="${EXP_DIR}/stopper/round${ROUND}"
EVAL_JSON="${STOPPER_DIR}/regret_eval.json"

banner "P4 — stopper v0 (round ${ROUND})"
require_file "${LABEL_DIR}/qa_lambda1.jsonl" "run scripts/p3_labels.sh first (P3)"
require_impl stopper/train_sft.py "stopper agent (Algorithm 2 SFT)"
require_impl stopper/eval_regret.py "stopper agent (held-out regret evaluation)"
mkdir -p "${STOPPER_DIR}"

# label files: ALL λ sets, BOTH domains, EXCLUDING the A8 plain-λ arm (§2.3:
# ONE λ-conditioned stopper across all λ label sets)
mapfile -t LABEL_FILES < <(ls "${LABEL_DIR}"/*_lambda*.jsonl | grep -v plainlam)
[ "${#LABEL_FILES[@]}" -gt 0 ] || pending "no label files in ${LABEL_DIR} — run p3 first"
TRAJ_FILES=()
for D in qa alfworld; do
    [ -f "${EXP_DIR}/collect/round${ROUND}/${D}.jsonl" ] \
        && TRAJ_FILES+=("${EXP_DIR}/collect/round${ROUND}/${D}.jsonl")
done

# ------------------------------------------------ 1. Algorithm 2 SFT (2 GPUs)
acquire_gpus 2
banner "P4.1 — Algorithm 2 SFT (3 epochs, lr 2e-5, early stop on held-out regret — §17 stopper.sft)"
# TODO(GPU): long run (~hours). train_sft pools all λ label sets (λ-conditioned, §2.3)
python -m cassi.stopper.train_sft \
    --labels "${LABEL_FILES[@]}" \
    --trajectories "${TRAJ_FILES[@]}" \
    --out "${STOPPER_DIR}" --seed 42

# ------- 2+3. held-out regret vs majority-class + calibrated probe = THE GATE
banner "P4.2 — GATE: compare_p4_baselines on held-out tasks (same split as training)"
DEFAULT_LAM_FILES=()
for D in qa alfworld; do
    [ -f "${LABEL_DIR}/${D}_lambda1.jsonl" ] && DEFAULT_LAM_FILES+=("${LABEL_DIR}/${D}_lambda1.jsonl")
done
if python3 "${CASSI_ROOT}/scripts/p4_gate.py" \
    --stopper-dir "${STOPPER_DIR}" \
    --labels "${DEFAULT_LAM_FILES[@]}" \
    --trajectories "${TRAJ_FILES[@]}" \
    --heldout-frac 0.2 --seed 42 \
    --out "${EVAL_JSON}"; then
    release_gpus
else
    release_gpus
    echo "P4 GATE: FAIL — §16 P4: 'if it cannot, STOP — fix features/labels before touching RL.'"
    echo "Do NOT proceed to P5/P6. Suspects: feature families (A4 grid), label noise (P3 memo QC b),"
    echo "regressor variance (backup residuals in experiments/labels/round*/qc_summary.json)."
    exit 1
fi

# ------------------------------------------ 4. external validity (non-gating)
echo
echo "TODO(GPU): external check — RedundancyBench step-redundancy F1 of the stopper"
echo "(2605.29893; §5.3 external validity — feeds the Analysis section, not this gate)."
banner "P4 DONE (gate passed) — report: ${EVAL_JSON}"
