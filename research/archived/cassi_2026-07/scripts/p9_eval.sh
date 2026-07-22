#!/usr/bin/env bash
# =============================================================================
# P9 — Full evaluation + ablations (paper_plan_v2 §16 P9, weeks 9-10)
#
# Goal: E1-E6 grids (§5.4), ablations A1-A9 (§5.5 — inference-only ablations
# REUSE trained models at near-zero cost, §7 tiering), the 500-task
# forced-continuation regret replays (dual-run protocol §5.3), 3 seeds on
# headline tables, statistics per §5.6, serving-regime overhead (KV-fork vs
# re-prefill).
#
# Done-criterion (§16 P9, quoted):
#   "✅ Done: all numbers for T1-T5 and F3-F6 exist in `experiments/results/`
#    as CSVs with a generation script per figure/table."
#
# GPU: N=2 for inference/eval sweeps; N=8 for the few remaining TRAINING
#      ablations (A1 sizes, A3, A6, A7, A9 — §7 budget tiering).
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RES="${EXP_DIR}/results"
SEEDS_HEADLINE="42 123 789"          # §5.6 — headline numbers only
SEED=42                               # everything else is 1-seed (§5.6 tiering)
LAM_DIALS="0.1 0.5 1.0 2.0 5.0"       # inference-time λ dial (E3; §17 label.lambda_values)
STOPPER0="${EXP_DIR}/stopper/round0"
STOPPER1="${EXP_DIR}/stopper/round1"
CASSI_QA="${EXP_DIR}/grpo/iter2/qa_refreshed_coach"       # headline executor (post-E5)
CASSI_ALF="${EXP_DIR}/grpo/iter2/alfworld_refreshed_coach"
BASE_MODEL="$(cfg_get executor.base_model)"

banner "P9 — full evaluation + ablations"
require_file "${CASSI_QA}" "run scripts/p7_loop_iter2.sh first"
require_impl eval/run_frontier.py "eval agent — frontier CLI runner (see contract in p5_killswitch.sh)"
require_impl eval/stats.py "eval agent (§5.6 statistics)"
require_impl eval/overhead.py "eval agent (serving regimes)"
mkdir -p "${RES}"

evalm() { python -m cassi.eval.run_frontier "$@"; }

# ========================= E1 — full grid, 3 seeds headline ==================
banner "E1 — CASSI vs B1-B9 on both training domains (headline: 3 seeds)"
acquire_gpus 2
for DOMAIN in qa alfworld; do
    POLICY="${CASSI_QA}"; [ "${DOMAIN}" = alfworld ] && POLICY="${CASSI_ALF}"
    DEVSET="${DATA_DIR}/hotpotqa_dev_frozen.jsonl"; [ "${DOMAIN}" = alfworld ] && DEVSET="verl_agent_test"
    for S in ${SEEDS_HEADLINE}; do
        # TODO(GPU): headline operating point x 3 seeds; baseline headline points
        # (B2-B9 at their dev-chosen knob) also get 3 seeds here — frontier points
        # stay 1-seed from P8 (§5.3).
        evalm --arm cassi --policy "${POLICY}" --monitor "${STOPPER1}" --lambda-dial 1.0 \
              --domain "${DOMAIN}" --tasks "${DEVSET}" --seed "${S}" \
              --out-csv "${RES}/e1_grid.csv" --append
    done
done

# ==================== E2 — internalization + transfer by role ================
banner "E2 — monitor-off, self-termination, OOD transfer by role (§5.1)"
# (i) monitor OFF at test time + self-termination rate — the internalization metric (§2.5)
for MODE in monitor_on monitor_off; do
    MON="${STOPPER1}"; [ "${MODE}" = monitor_off ] && MON=none
    evalm --arm "cassi_${MODE}" --policy "${CASSI_QA}" --monitor "${MON}" --lambda-dial 1.0 \
          --domain qa --tasks "${DATA_DIR}/hotpotqa_dev_frozen.jsonl" --seed "${SEED}" \
          --self-termination --out-csv "${RES}/e2_internalization.csv" --append
done
# (ii) trained-executor transfer: local-retrieval OOD sets (tool stack matches training)
for SET in 2wikimultihopqa_dev bamboogle; do
    evalm --arm cassi_transfer --policy "${CASSI_QA}" --monitor "${STOPPER1}" --lambda-dial 1.0 \
          --domain qa --tasks "${DATA_DIR}/${SET}.jsonl" --seed "${SEED}" \
          --out-csv "${RES}/e2_transfer_executor.csv" --append
done
echo "TODO(GPU): BrowseComp-Plus (830 Qs) once its corpus+index are staged (P1.4)."
# (iii) stopper-as-monitor transfer: GAIA-103 over a FROZEN live-web agent
#       (SupervisorAgent-comparable setup; point estimate + bootstrap CI only — n<500, §5.6)
echo "TODO(GPU): GAIA-103 stopper-as-monitor run — frozen live-web agent + ${STOPPER1} as monitor;"
echo "           search-time-contamination caveat (2606.05241) reported (§5.1)."
# (iv) stopper transferred across executors: cross-family + cross-scale (§5.4 E2)
for XFER in "cross_family:$(cfg_get executor.transfer_models.cross_family)" \
            "cross_scale:$(cfg_get executor.transfer_models.cross_scale)"; do
    IFS=':' read -r TAG MODEL <<< "${XFER}"
    evalm --arm "stopper_xfer_${TAG}" --policy "${MODEL}" --monitor "${STOPPER1}" --lambda-dial 1.0 \
          --domain qa --tasks "${DATA_DIR}/hotpotqa_dev_frozen.jsonl" --seed "${SEED}" \
          --out-csv "${RES}/e2_transfer_stopper.csv" --append
done

# ============ E3 — λ-frontier via inference dial + three-wallets study =======
banner "E3 — λ dial frontier on the FIXED executor + same-task-three-wallets (§5.4)"
for DIAL in ${LAM_DIALS}; do
    evalm --arm cassi_dial --policy "${CASSI_QA}" --monitor "${STOPPER1}" --lambda-dial "${DIAL}" \
          --domain qa --tasks "${DATA_DIR}/hotpotqa_dev_frozen.jsonl" --seed "${SEED}" \
          --out-csv "${RES}/e3_lambda_frontier.csv" --append
done
for WALLET in small medium large; do
    evalm --arm "cassi_wallet_${WALLET}" --policy "${CASSI_QA}" --monitor "${STOPPER1}" \
          --lambda-dial 1.0 --wallet "${WALLET}" \
          --domain qa --tasks "${DATA_DIR}/hotpotqa_dev_frozen.jsonl" --seed "${SEED}" \
          --out-csv "${RES}/e3_three_wallets.csv" --append
done
echo "TODO(GPU): ONE λ=0.3 executor training as the dial-tracks-trained-frontier spot check"
echo "           (E3: 'never 5 executor trainings') — python -m cassi.executor.train_grpo --lambda 0.3 ..."

# ================ E4 — label-semantics study at matched compute ===============
banner "E4 — Snell vs prophet vs TD/GAE vs MC labels (stopper + downstream)"
echo "TODO(GPU): per label family {snell, prophet, td_gae, mc}: retrain the 2B stopper on"
echo "           round-0 data (scripts/run_labels.py emits snell+prophet; td_gae/mc from"
echo "           cassi.baselines.b8_agentprm_cost machinery), eval held-out regret"
echo "           (cassi.stopper.eval_regret), + ONE downstream shaped-GRPO run each on qa."
echo "           Results -> ${RES}/e4_label_study.csv (feeds F5)."

# ===================== E5 — assembled from P7 (no new runs) ===================
banner "E5 — per-iteration loop deltas: already produced by p7_loop_iter2.sh"
require_file "${RES}/e5_loop_iterations.csv" "run scripts/p7_loop_iter2.sh first"

# ============ E6 — difficulty-stopping consistency check (no new runs) ========
banner "E6 — 2-hop vs 4-hop MuSiQue stop-step correlation (consistency check only)"
evalm --arm cassi --policy "${CASSI_QA}" --monitor "${STOPPER1}" --lambda-dial 1.0 \
      --domain qa --tasks "${DATA_DIR}/musique_dev_frozen.jsonl" --seed "${SEED}" \
      --by-difficulty --out-csv "${RES}/e6_difficulty.csv" --append

# ============== 500-task stopping-regret replays (dual-run, §5.3) =============
banner "Regret replays — 500-task dual-run protocol (billed to T4's analysis line)"
require_impl executor/collect.py "executor agent"
# TODO(GPU): for each headline method: (1) normal eval run already exists above;
# (2) forced-continuation REPLAY to T_max on the fixed 500-task subsample:
python3 - "${DATA_DIR}/hotpotqa_dev_frozen.jsonl" "${DATA_DIR}/regret_replay_500.jsonl" <<'PY'
import json, random, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
random.Random(42).shuffle(rows)
out = rows[:500]
with open(sys.argv[2], "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"fixed regret-replay subsample: {len(out)} tasks (chosen once, seed 42)")
PY
echo "TODO(GPU): python -m cassi.executor.collect --domain qa --iteration 2 \\"
echo "             --tasks ${DATA_DIR}/regret_replay_500.jsonl --out ${EXP_DIR}/replays/<method>.jsonl"
echo "           (forced continuation is collect.py's default mode; serve each method's"
echo "            checkpoint via vLLM and point --vllm-url at it)"
echo "           then python -m cassi.eval.run_frontier --regret-from-replays ... -> ${RES}/regret_500.csv"
release_gpus

# =============== Ablations A1-A9 (§5.5, with §7 reuse tiering) ================
banner "Ablations A1-A9"
acquire_gpus 8
# A1 stopper size 0.8B/2B/4B — TRAINING (2B exists; controller part reuses eval stack)
for SIZE in 0.8B 4B; do
    echo "TODO(GPU): A1 — python -m cassi.stopper.train_sft --base-model Qwen/Qwen3.5-${SIZE} \\"
    echo "             --labels ${EXP_DIR}/labels/round0 --out ${EXP_DIR}/stopper/a1_${SIZE}"
done
# A2 single- vs two-model at matched params — REUSES K2 (p5) at kill-switch scale;
#    full-scale rerun only if K2 was borderline (log the choice in GO_NO_GO.log).
echo "A2: reuse ${EXP_DIR}/killswitch/single_multitask (K2) — no new run unless borderline"
# A3 potential-based vs additive Δ reward — TRAINING (predicted: additive dawdles, §2.4)
echo "TODO(GPU): A3 — python -m cassi.executor.train_grpo --arm additive_delta --domain qa ..."
# A4 stopper input families — cheap 2B RETRAINS on feature-masked labels
for FAM in budget_only budget_history full; do
    echo "TODO(GPU): A4 — python -m cassi.stopper.train_sft --feature-family ${FAM} ..."
done
# A5 eval frequency + legacy-probe check — INFERENCE-ONLY (reuses headline executor)
for K in 1 2 3; do
    evalm --arm "a5_every_${K}" --policy "${CASSI_QA}" --monitor "${STOPPER1}" \
          --lambda-dial 1.0 --stopper-every-k "${K}" \
          --domain qa --tasks "${DATA_DIR}/hotpotqa_dev_frozen.jsonl" --seed "${SEED}" \
          --out-csv "${RES}/a5_frequency.csv" --append
done
# A6 SFT vs SFT+RL stopper — TRAINING (stopper GRPO stage, demoted to ablation §2.3)
echo "TODO(GPU): A6 — python -m cassi.stopper.train_sft --plus-rl ..."
# A7 rationale on/off — stopper RETRAIN with rationale head (§2.3)
echo "TODO(GPU): A7 — python -m cassi.stopper.train_sft --rationale ..."
# A8 learned allowance-conditioning vs rule table — INFERENCE-ONLY over the
#    plain-λ stopper (labels already emitted by p3 --plain-lambda) + δ(tier) table (§17)
echo "TODO(GPU): A8 — train stopper on *_plainlam labels, eval with"
echo "           --rule-table \"\$(cfg key inference.ablation_A8_rule_table)\" vs learned (§2.5)"
# A9 negative controls — TRAINING x2, 1 seed, primary domain (cheap, decisive §5.5)
for CTRL in random_coach shuffled_label_coach; do
    echo "TODO(GPU): A9 — python -m cassi.executor.train_grpo --arm ${CTRL} --domain qa --seed 42 ..."
done
release_gpus

# ===================== statistics + overhead accounting =======================
banner "Statistics (§5.6) + serving-regime overhead (T4)"
python -m cassi.eval.stats \
    --results-dir "${RES}" --seeds ${SEEDS_HEADLINE} \
    --bootstrap 10000 --holm-bonferroni --min-n-hypothesis-test 500 \
    --out "${RES}/stats_summary.csv"
python -m cassi.eval.overhead \
    --regimes kv_fork,re_prefill \
    --collect-dirs "${EXP_DIR}/collect/round0,${EXP_DIR}/collect/round1" \
    --out "${RES}/t4_overhead.csv"

banner "P9 — done-criterion check: T1-T5 / F3-F6 inputs exist as CSVs"
# Two layers: the raw E-grids above, and the per-figure/table aggregates the
# analysis/ scripts read (exact names hard-coded in analysis/{figures,tables}/*.py).
# The aggregates are distilled from the E-grids by cassi.eval.stats at this phase's
# end — TODO(GPU): wire stats to emit them once all E-grids exist.
MISSING=0
for F in e1_grid e2_internalization e3_lambda_frontier e5_loop_iterations stats_summary \
         t1_headline t2_baselines t3_ablations t4_overhead t5_transfer \
         f3_pareto f4_internalization f5_label_study f6_hacking; do
    if [ ! -f "${RES}/${F}.csv" ]; then echo "  missing: ${RES}/${F}.csv"; MISSING=1; fi
done
[ "${MISSING}" -eq 0 ] && banner "P9 DONE — proceed to P10 (figures/tables)" \
    || { echo "P9 INCOMPLETE — finish the TODO(GPU) blocks above"; exit 1; }
