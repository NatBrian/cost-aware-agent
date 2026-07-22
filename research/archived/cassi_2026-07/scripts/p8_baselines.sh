#!/usr/bin/env bash
# =============================================================================
# P8 — Baselines B2–B9 (paper_plan_v2 §5.2, §16 P8; overlaps W4-8)
#
# Goal: every §5.2 baseline on both training domains, each swept over ITS OWN
# cost knob to a 2-3 point frontier (§5.3 frontier protocol; B1 ReAct is the
# knobless single point and is already evaluated in P6). Training baselines
# B4-B9 are ~1-3 day runs each on 8xH200 (§16 P8) — schedule accordingly.
#
# Done-criterion (§16 P8, quoted):
#   "✅ Done: every baseline evaluated on the same frozen test sets with the
#    same cost accounting."
#
# GPU: N=2 for inference-only baselines (B2/B3), N=8 for training baselines (B4-B9).
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BL_DIR="${EXP_DIR}/baselines"
FRONTIER_CSV="${EXP_DIR}/results/baselines_frontier.csv"
SEED=42          # frontier points are 1-seed; headline operating points get 3 seeds at P9 (§5.3)
BASE_MODEL="$(cfg_get executor.base_model)"

banner "P8 — baselines B2-B9"
require_impl baselines/b9_direct_shaping.py "baselines agent (one module per §5.2 row)"
require_impl eval/run_frontier.py "eval agent — frontier CLI runner (see contract in p5_killswitch.sh)"
mkdir -p "${BL_DIR}"

declare -A TASKS=(
    [qa]="${DATA_DIR}/hotpotqa_train.decontaminated.jsonl,${DATA_DIR}/nq_train.decontaminated.jsonl,${DATA_DIR}/musique_train.decontaminated.jsonl"
    [alfworld]="verl_agent_builtin"
)
declare -A DEV=( [qa]="${DATA_DIR}/hotpotqa_dev_frozen.jsonl" [alfworld]="verl_agent_dev" )
DOMAINS="qa alfworld"

# --- registry: baseline -> module | knob-name | frontier values (§5.2 + §5.3 knobs)
# B2 zero-training self-eval + calibrated confidence exit (Dynasor-style probe)
# B3 SupervisorAgent-style training-free monitor (protocol ADAPTATION — disclosed, §5.2)
# B4 OTC-GRPO (2504.14870)          knob: tool-count coefficient
# B5 EAPO (2606.02132; primary)     knob: penalty weight   [agentic-ALP only if unreproducible]
# B6 single-model GRPO cost-in-reward (CTA-style)          knob: λ
# B7 CaRT+cost — TWO arms: sft_only and grpo (§5.2)        knob: λ
# B8 AgentPRM-cost (pooled return-to-go + cost)            knob: λ
# B9 DASH-style direct shaping (Snell labels, no stopper)  knob: λ
# B10 RM-P prompted judge, monitor arm (v2.1 §5.2)         knob: rubric threshold θ_p
#     (B10's rl arm is NOT run here — post-K1 GO only, launched manually per §5.2)
declare -A MODULE=(
    [b2]=cassi.baselines.b2_probe              [b3]=cassi.baselines.b3_supervisor_monitor
    [b4]=cassi.baselines.b4_otc_grpo           [b5]=cassi.baselines.b5_eapo
    [b6]=cassi.baselines.b6_single_model_cost  [b7]=cassi.baselines.b7_cart_cost
    [b8]=cassi.baselines.b8_agentprm_cost      [b9]=cassi.baselines.b9_direct_shaping
    [b10]=cassi.baselines.b10_prompted_rm
)
declare -A KNOB=(
    [b2]=confidence-threshold [b3]=trigger-sensitivity [b10]=rubric-threshold
    [b4]=tool-coef [b5]=penalty-weight [b6]=lambda [b7]=lambda [b8]=lambda [b9]=lambda
)
declare -A POINTS=(
    [b2]="0.5 0.7 0.9"  [b3]="low medium high"  [b10]="0.2 0.4 0.6"
    [b4]="0.5 1.0 2.0"  [b5]="0.5 1.0 2.0"  [b6]="0.5 1.0 2.0"
    [b7]="0.5 1.0 2.0"  [b8]="0.5 1.0 2.0"  [b9]="0.5 1.0 2.0"
)
INFERENCE_ONLY="b2 b3 b10"    # b10 monitor arm needs the lab vLLM 30B endpoint up (config prompted_rm.base_url)
TRAINING="b4 b5 b6 b7 b8 b9"

# The baselines landed as reward/stopping-rule LIBRARIES (reward fns, probes,
# billing helpers); the runnable CLI over them is pending — gate up front rather
# than let `python -m` silently import-and-exit.
for B in ${INFERENCE_ONLY} ${TRAINING}; do
    require_module_cli "${MODULE[${B}]}" "baselines agent — add a main() CLI wiring the library into rollouts/training (flags used below)"
done

eval_point() {  # eval_point <arm> <policy> <monitor> <dial> <domain>
    python -m cassi.eval.run_frontier \
        --arm "$1" --policy "$2" --monitor "$3" --lambda-dial "$4" \
        --domain "$5" --tasks "${DEV[$5]}" --seed "${SEED}" \
        --out-csv "${FRONTIER_CSV}" --append
}

# ===================== inference-only baselines (B2, B3) =====================
acquire_gpus 2
for B in ${INFERENCE_ONLY}; do
    for DOMAIN in ${DOMAINS}; do
        for V in ${POINTS[${B}]}; do
            banner "P8 — ${B} (${MODULE[${B}]}) ${DOMAIN} ${KNOB[${B}]}=${V} [inference-only]"
            # Billing symmetry (§5.3): the probe/monitor's own inference is billed
            # into cost_dollars by the module — same price map as everything else.
            python -m "${MODULE[${B}]}" \
                --domain "${DOMAIN}" --tasks "${DEV[${DOMAIN}]}" \
                --"${KNOB[${B}]}" "${V}" --policy "${BASE_MODEL}" --seed "${SEED}" \
                --out-csv "${FRONTIER_CSV}" --append --arm "${B}"
        done
    done
done
release_gpus

# ======================= training baselines (B4-B9) ==========================
acquire_gpus 8
for B in ${TRAINING}; do
    ARMS="default"
    [ "${B}" = "b7" ] && ARMS="sft_only grpo"     # §5.2 B7: BOTH arms
    for ARM in ${ARMS}; do
        for DOMAIN in ${DOMAINS}; do
            for V in ${POINTS[${B}]}; do
                TAG="${B}"
                if [ "${ARM}" != default ]; then TAG="${B}_${ARM}"; fi
                CKPT="${BL_DIR}/${TAG}/${DOMAIN}_${KNOB[${B}]}${V}"
                if [ -d "${CKPT}" ]; then
                    echo "[p8] ${TAG}/${DOMAIN}/${V}: exists — skipping training"
                else
                    banner "P8 — TRAIN ${TAG} ${DOMAIN} ${KNOB[${B}]}=${V}"
                    # TODO(GPU): long run (~1-3 days on 8xH200 per point, §16 P8).
                    # b9 at qa/lambda=1.0 can REUSE the K1 checkpoint
                    # (experiments/killswitch/b9_direct) — symlink instead of retraining.
                    EXTRA=""
                    [ "${B}" = "b7" ] && EXTRA="--arm-variant ${ARM}"
                    [ "${B}" = "b8" ] || [ "${B}" = "b9" ] && EXTRA="${EXTRA} --labels ${EXP_DIR}/labels/round0"
                    # shellcheck disable=SC2086
                    python -m "${MODULE[${B}]}" \
                        --domain "${DOMAIN}" --tasks "${TASKS[${DOMAIN}]}" \
                        --"${KNOB[${B}]}" "${V}" --seed "${SEED}" ${EXTRA} \
                        --out "${CKPT}"
                fi
                eval_point "${TAG}" "${CKPT}" none "${V}" "${DOMAIN}"
            done
        done
    done
done
release_gpus

banner "P8 — done-criterion check"
python3 - "${FRONTIER_CSV}" <<'PY'
import csv, sys
from collections import Counter
rows = list(csv.DictReader(open(sys.argv[1])))
counts = Counter((r["arm"].split("_")[0], r.get("domain", "qa")) for r in rows)
missing = []
for b in ["b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10"]:
    for d in ["qa", "alfworld"]:
        n = counts.get((b, d), 0)
        print(f"{b}/{d}: {n} frontier point(s)")
        if n < 2:
            missing.append(f"{b}/{d}")
print("P8 done-criterion:", "PASS" if not missing else f"INCOMPLETE — need >=2 points for {missing}")
sys.exit(0 if not missing else 1)
PY
banner "P8 DONE"
