#!/usr/bin/env bash
# =============================================================================
# P5 — KILL-SWITCH GATE, K1 + K2 (paper_plan_v2 §12, §16 P5, weeks 2-3)
#
# §12 (quoted):
#   "K1 (bridge test): HotpotQA 1K-task subset, 2B stopper + 9B executor, 1 seed:
#    Δ-shaped GRPO vs controller-only vs B9-direct-shaping. Proceed iff shaped-GRPO
#    beats controller-only by ≥3 points cost-at-iso-accuracy AND ≥ B9 (1 seed —
#    read as direction + magnitude, not significance). Else pivot per H2/H3 fallbacks.
#    K2 (separation test): same subset: 9B single multi-task (task+stopping heads)
#    vs 9B+2B (params counted in reporting). Any outcome is publishable via H4.
#    GO/NO-GO review after K1+K2 before building anything else."
#
# Done-criterion (§16 P5, quoted):
#   "✅ GO if K1 passes per §12 thresholds. NO-GO → pivot per H2/H3/H4 fallback
#    framings; write the decision log either way (feeds appendix)."
#
# GPU: N=4 (small-scale GRPO on the 1K subset; full runs use 8 — §16 P0 note: 4-8 for GRPO).
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

KS_DIR="${EXP_DIR}/killswitch"
RESULTS="${KS_DIR}/results"
SUBSET="${DATA_DIR}/killswitch_hotpotqa_1k.jsonl"
DEV="${DATA_DIR}/hotpotqa_dev_frozen.jsonl"
STOPPER="${EXP_DIR}/stopper/round0"
K1_CSV="${RESULTS}/k1_frontier.csv"
K2_CSV="${RESULTS}/k2_frontier.csv"
SEED=42          # 1 seed (§12)
DIALS="0.5 1.0 2.0"   # inference-time λ dial points for the mini-frontier (§5.3 protocol)

banner "P5 — kill-switch gate K1 + K2"
require_file "${STOPPER}" "run scripts/p4_stopper.sh first (P4 gate must have passed)"
require_file "${DEV}" "run scripts/p1_data.sh first (frozen dev subsample, §5.6)"
require_impl executor/train_grpo.py "executor agent (Algorithm 3 shaped GRPO)"
# NOTE: train_grpo currently exposes --config/--domain/--dry-run only; the flags
# used below (--tasks/--seed/--lambda/--coach/--arm/--out) are the §16 contract
# to be wired when the verl GRPO launch lands — the call fails loudly until then.
require_impl baselines/b9_direct_shaping.py "baselines agent (§5.2 B9)"
require_module_cli cassi.baselines.b9_direct_shaping "baselines agent — b9 landed as a library; add a main() CLI (flags used in K1 arm 3 below)"
require_impl eval/run_frontier.py "eval agent — frontier CLI runner over cassi.eval.metrics (landed as a library; the runner must roll out --policy on --tasks with billing symmetry and append arm,lambda_dial,accuracy,cost_dollars rows to --out-csv)"
mkdir -p "${RESULTS}"

# ---------------------------------------------- GO_NO_GO.log (created once, §5.6)
if [ ! -f "${GO_NO_GO_LOG}" ]; then
    cat > "${GO_NO_GO_LOG}" <<'EOF'
# CASSI GO/NO-GO decision log
#
# APPEND-ONLY. Never edit or delete entries. paper_plan_v2 §5.6 no-cherry-picking
# clause (quoted): "kill-switch GO/NO-GO decisions (§12) are logged with dates in
# the repo." This file is committed and ships in the paper appendix (§9), so every
# pivot, NO-GO, and fallback-framing switch is on the record — a NO-GO here is a
# documented pivot per H2/H3/H4 (§6), not a deleted experiment.
#
# Entries are appended by scripts/killswitch_decision.py and by hand for any later
# scope decision (e.g., E5 loop verdicts, baseline drops). Date every entry (UTC).
EOF
    echo "[p5] created ${GO_NO_GO_LOG}"
fi

# --------------------------------------------------- 1K HotpotQA subset (seed 42)
if [ ! -f "${SUBSET}" ]; then
    python3 - "${DATA_DIR}/hotpotqa_train.decontaminated.jsonl" "${SUBSET}" <<'PY'
import json, random, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
random.Random(42).shuffle(rows)
with open(sys.argv[2], "w") as f:
    for r in rows[:1000]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"killswitch subset: {min(1000, len(rows))} tasks -> {sys.argv[2]}")
PY
fi

# Eval contract (cassi/eval/run_frontier.py): each call appends one frontier row
#   arm,lambda_dial,accuracy,cost_dollars
# to --out-csv, evaluating --policy on --tasks with billing symmetry (§5.3:
# stopper/monitor calls are billed into cost_dollars).
eval_point() {  # eval_point <arm> <policy> <monitor> <dial> <out_csv>
    python -m cassi.eval.run_frontier \
        --arm "$1" --policy "$2" --monitor "$3" --lambda-dial "$4" \
        --tasks "${DEV}" --seed "${SEED}" --out-csv "$5" --append
}

acquire_gpus 4

# ============================== K1 — bridge test ==============================
banner "K1 arm 1/3 — Δ-shaped GRPO (the bridge, Algorithm 3)"
SHAPED_CKPT="${KS_DIR}/shaped_grpo"
if [ ! -d "${SHAPED_CKPT}" ]; then
    # TODO(GPU): long run (~1 day at 1K tasks) — potential-based shaping from the
    # round-0 stopper's V̂, step-level advantages, Dr.GRPO hygiene (§2.4/§10 Alg.3)
    python -m cassi.executor.train_grpo \
        --domain qa --tasks "${SUBSET}" --iteration 1 --seed "${SEED}" \
        --lambda 1.0 --coach "${STOPPER}" --arm shaped \
        --out "${SHAPED_CKPT}"
fi
for DIAL in ${DIALS}; do
    eval_point shaped "${SHAPED_CKPT}" "${STOPPER}" "${DIAL}" "${K1_CSV}"
done

banner "K1 arm 2/3 — controller-only (same stopper, NO executor training — H2's comparator)"
BASE_MODEL="$(cfg_get executor.base_model)"
for DIAL in ${DIALS}; do
    eval_point controller_only "${BASE_MODEL}" "${STOPPER}" "${DIAL}" "${K1_CSV}"
done

banner "K1 arm 3/3 — B9 direct shaping (Snell labels as advantages, stopper DELETED — §5.2)"
B9_CKPT="${KS_DIR}/b9_direct"
if [ ! -d "${B9_CKPT}" ]; then
    # TODO(GPU): long run — CASSI's exact step-level machinery, only the stopper removed
    python -m cassi.baselines.b9_direct_shaping \
        --domain qa --tasks "${SUBSET}" --labels "${EXP_DIR}/labels/round0" \
        --lambda 1.0 --seed "${SEED}" --out "${B9_CKPT}"
fi
eval_point b9 "${B9_CKPT}" none 1.0 "${K1_CSV}"

# ============================ K2 — separation test ============================
banner "K2 — 9B single multi-task (task+stopping heads) vs 9B+2B (params counted)"
SINGLE_CKPT="${KS_DIR}/single_multitask"
if [ ! -d "${SINGLE_CKPT}" ]; then
    # TODO(GPU): long run — A2's machinery at kill-switch scale (§5.5 A2 / §12 K2)
    python -m cassi.executor.train_grpo \
        --domain qa --tasks "${SUBSET}" --iteration 1 --seed "${SEED}" \
        --lambda 1.0 --arm single_multitask \
        --out "${SINGLE_CKPT}"
fi
eval_point single_multitask "${SINGLE_CKPT}" self 1.0 "${K2_CSV}"
# two-model arm REUSES the K1 shaped run (no extra training; §7 tiering)
eval_point two_model "${SHAPED_CKPT}" "${STOPPER}" 1.0 "${K2_CSV}"

release_gpus

# ======================= GO/NO-GO decision (§12 thresholds) ===================
banner "P5 — GO/NO-GO decision -> ${GO_NO_GO_LOG}"
if python3 "${CASSI_ROOT}/scripts/killswitch_decision.py"; then
    banner "P5: GO — proceed to P6 (executor GRPO iteration 1)"
else
    RC=$?
    [ "${RC}" -eq 75 ] && pending "kill-switch inputs incomplete — see messages above"
    banner "P5: NO-GO — STOP. Pivot per H2/H3 fallback framings (§6); decision logged."
    exit 1
fi
