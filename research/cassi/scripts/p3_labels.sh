#!/usr/bin/env bash
# =============================================================================
# P3 — Label construction (paper_plan_v2 §16 P3, week 2)
#
# Goal: run Algorithm 1 (Snell-envelope labels, cassi.labels.snell) per
# λ ∈ {0.1, 0.5, 1, 2, 5} with tier-scaled marginal costing (§2.2; the plain-λ
# variant is kept for ablation A8), fit the tanh scale s per domain, and run the
# three QC checks: (a) 100-trajectory manual review export, (b) label-noise
# sensitivity (step-subsampled draft scoring), (c) λ-monotonicity sanity.
#
# Done-criterion (§16 P3, quoted):
#   "✅ Done: labeled datasets per λ + a one-page label-quality memo."
#
# CPU-only (LightGBM regressor) — no GPU acquisition needed.
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ROUND="${1:-0}"   # 0 = P2 collection; p7_loop_iter2.sh reruns this with 1

banner "P3 — Snell-envelope labels, collection round ${ROUND}"
require_pilot_calibration qa
require_pilot_calibration alfworld
require_file "${EXP_DIR}/collect/round${ROUND}/qa.jsonl" "run scripts/p2_pilot_and_collect.sh (or p7 for round 1) first"
python3 -c "import lightgbm" 2>/dev/null \
    || echo "[p3] note: lightgbm not importable — Algorithm 1 falls back to the sklearn MLP regressor (§17 label.regressor.fallback)"

# ---- Algorithm 1 per λ, both domains + QC (a)/(b)/(c) + memo ----------------
python3 "${CASSI_ROOT}/scripts/run_labels.py" \
    --round "${ROUND}" --domains qa alfworld --review-sample 100 --seed 42

# ---- A8 arm: plain-λ labels (m ≡ 1) — cheap, produced now, consumed at P9 ---
banner "P3 — A8 plain-λ label arm (rule-table comparator economy, §2.2/§5.5)"
python3 "${CASSI_ROOT}/scripts/run_labels.py" \
    --round "${ROUND}" --domains qa alfworld --review-sample 20 --seed 42 --plain-lambda

banner "P3 DONE — labeled datasets in experiments/labels/round${ROUND}/"
echo "Done-criterion artifact: experiments/labels/round${ROUND}/label_quality_memo.md"
echo "ACTION REQUIRED before P4: complete the memo's manual-review TODO checkboxes"
echo "(§16 P3 QC (a): 'does τ* look right?') and freeze the printed tanh scale s"
echo "into configs/cassi.yaml label.delta_scale if this is round 0."
