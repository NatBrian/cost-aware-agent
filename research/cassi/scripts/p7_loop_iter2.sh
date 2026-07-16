#!/usr/bin/env bash
# =============================================================================
# P7 — Loop iteration 2 (paper_plan_v2 §16 P7, weeks 7-8; experiment E5)
#
# Goal: re-collect with the iteration-1 executor IN FORCED-CONTINUATION MODE
# again (§2.1 — the internalized executor stops early; without forcing, late-step
# label data would vanish exactly because the method worked) → rerun P3-P4
# (label + stopper refresh) → GRPO iteration 2 TWICE at MATCHED COMPUTE:
#   arm (a) frozen iteration-1 coach   — the "more RL steps" control;
#   arm (b) refreshed coach            — the loop;
# the (b)−(a) delta IS the loop's contribution (E5 — without it, no loop language
# survives review, §2.7).
#
# Done-criterion (§16 P7, quoted):
#   "✅ Done: per-iteration deltas table (cost, accuracy, stopping regret at
#    i=0,1,2) + the (b)−(a) loop-contribution delta — this IS E5."
#
# GPU: N=2 for re-collection, N=8 for the two GRPO arms (§16 P0 note).
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ITER1_DIR="${EXP_DIR}/grpo/iter1"
COLLECT1="${EXP_DIR}/collect/round1"
ITER2_DIR="${EXP_DIR}/grpo/iter2"
E5_CSV="${EXP_DIR}/results/e5_loop_iterations.csv"
SEED=42
LAM="$(cfg_get executor.training_lambda)"
STEP_CREDIT="$(cfg_get executor.grpo.step_level_variant)"
G="$(cfg_get data.collection.rollouts_per_task_G)"
MATCHED_STEPS=500   # TODO(GPU): set to iteration-1's actual optimizer-step count — arms (a)/(b) MUST match (E5)

banner "P7 — loop iteration 2 (E5)"
require_file "${ITER1_DIR}/qa" "run scripts/p6_grpo_iter1.sh first"
require_impl executor/collect.py "executor agent"
require_impl executor/train_grpo.py "executor agent"
mkdir -p "${COLLECT1}" "${ITER2_DIR}"

# ---------------- 1. re-collection with the iter-1 executor (forced continuation)
acquire_gpus 2
for DOMAIN in qa alfworld; do
    OUT="${COLLECT1}/${DOMAIN}.jsonl"
    if [ -f "${OUT}" ]; then
        echo "[p7] ${DOMAIN}: ${OUT} exists — skipping recollection"
        continue
    fi
    TASKFILE="${EXP_DIR}/collect/round0/qa_tasks.jsonl"          # reuse the round-0 task list
    [ "${DOMAIN}" = alfworld ] && TASKFILE="verl_agent_builtin"  # TODO(GPU): verl-agent list (P1.3)
    banner "P7.1 — re-collection round 1: ${DOMAIN} (forced continuation, §2.1)"
    # TODO(GPU): long run — the policy must be the ITERATION-1 executor, not the base
    # model: serve ${ITER1_DIR}/${DOMAIN} with vLLM and point --vllm-url at it
    # (collect.py has no --policy flag; the model identity lives in the server).
    python -m cassi.executor.collect \
        --domain "${DOMAIN}" --iteration 1 \
        --tasks "${TASKFILE}" \
        --G "${G}" --seed "${SEED}" --out "${OUT}"
done
release_gpus

# ---------------------------- 2. label refresh (P3 machinery, round 1) + memo
banner "P7.2 — label refresh: rerun P3 on round-1 trajectories"
bash "${CASSI_ROOT}/scripts/p3_labels.sh" 1

# --------------------------------- 3. stopper refresh (P4 machinery, round 1)
banner "P7.3 — stopper refresh: rerun P4 on round-1 labels (anti-hacking, §2.4)"
bash "${CASSI_ROOT}/scripts/p4_stopper.sh" 1

# ------------- 4. GRPO iteration 2, TWO ARMS at matched compute (E5's control)
acquire_gpus 8
for DOMAIN in qa alfworld; do
    for ARM_COACH in "frozen_coach:${EXP_DIR}/stopper/round0" "refreshed_coach:${EXP_DIR}/stopper/round1"; do
        IFS=':' read -r ARM COACH <<< "${ARM_COACH}"
        CKPT="${ITER2_DIR}/${DOMAIN}_${ARM}"
        if [ -d "${CKPT}" ]; then
            echo "[p7] ${DOMAIN}/${ARM}: exists — skipping"
            continue
        fi
        banner "P7.4 — GRPO iteration 2, ${DOMAIN}, arm=${ARM} (matched compute: ${MATCHED_STEPS} steps)"
        # TODO(GPU): long run x4 (2 domains x 2 arms). --init resumes FROM the
        # iteration-1 executor; --max-steps enforces matched compute (E5).
        python -m cassi.executor.train_grpo \
            --domain "${DOMAIN}" --iteration 2 --seed "${SEED}" --lambda "${LAM}" \
            --init "${ITER1_DIR}/${DOMAIN}" --coach "${COACH}" \
            --arm "shaped_${ARM}" --step-credit "${STEP_CREDIT}" \
            --max-steps "${MATCHED_STEPS}" \
            --out "${CKPT}"
    done
done

# -------------------- 5. per-iteration deltas table (done-criterion; feeds E5/F4)
banner "P7.5 — per-iteration eval: i=0 (base), i=1, i=2(a), i=2(b)"
for DOMAIN in qa; do
    DEVSET="${DATA_DIR}/hotpotqa_dev_frozen.jsonl"
    for ARM_POLICY in \
        "iter0_base:$(cfg_get executor.base_model)" \
        "iter1:${ITER1_DIR}/${DOMAIN}" \
        "iter2_frozen_coach:${ITER2_DIR}/${DOMAIN}_frozen_coach" \
        "iter2_refreshed_coach:${ITER2_DIR}/${DOMAIN}_refreshed_coach"; do
        IFS=':' read -r ARM POLICY <<< "${ARM_POLICY}"
        python -m cassi.eval.run_frontier \
            --arm "${ARM}" --policy "${POLICY}" --monitor "${EXP_DIR}/stopper/round1" \
            --lambda-dial "${LAM}" --domain "${DOMAIN}" --tasks "${DEVSET}" \
            --seed "${SEED}" --with-regret --out-csv "${E5_CSV}" --append
    done
done
# TODO(GPU): repeat the block above for alfworld dev, and add stopping-regret columns
# via the 500-task dual-run protocol at P9 (§5.3) — this table is E5's skeleton.
release_gpus

echo
echo "E5 verdict = (iter2_refreshed_coach − iter2_frozen_coach) at matched compute (§2.7)."
echo "Consider appending the E5 outcome to ${GO_NO_GO_LOG} (loop language in the paper"
echo "depends on it — §5.6 no-cherry-picking)."
banner "P7 DONE — deltas table at ${E5_CSV}"
