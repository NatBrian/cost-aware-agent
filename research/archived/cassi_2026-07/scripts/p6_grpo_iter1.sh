#!/usr/bin/env bash
# =============================================================================
# P6 — Executor GRPO, iteration 1 (paper_plan_v2 §16 P6, weeks 4-6)
#
# Goal: Algorithm 3 on both training domains — potential-based economic shaping
# from the round-0 stopper's V̂ (Φ(terminal)=0, γ=1), STEP-LEVEL advantage
# assignment (mandatory — trajectory-level shaping telescopes to a constant,
# §2.4), Dr.GRPO length hygiene, KL β=0.04, V̂-vs-realized-reward divergence
# dashboard logged from step 0 (feeds F6).
#
# Done-criterion (§16 P6, quoted):
#   "✅ Done: iteration-1 executor beats B1 on cost@iso-accuracy on dev in both
#    domains."
#
# GPU: N=8 (full executor GRPO — §16 P0 note: 4-8 for GRPO; ~1-3 days/run, §7).
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STOPPER="${EXP_DIR}/stopper/round0"
GRPO_DIR="${EXP_DIR}/grpo/iter1"
P6_CSV="${EXP_DIR}/results/p6_iter1_dev.csv"
SEED=42
LAM="$(cfg_get executor.training_lambda)"                       # 1.0 headline (§17)
STEP_CREDIT="$(cfg_get executor.grpo.step_level_variant)"       # K1 picks: per_step_rtg | shape_segment (§2.4)

banner "P6 — GRPO iteration 1 (λ=${LAM}, step-credit=${STEP_CREDIT}, seed ${SEED})"
require_file "${STOPPER}" "run scripts/p4_stopper.sh first"
grep -q "VERDICT: GO" "${GO_NO_GO_LOG}" 2>/dev/null \
    || pending "no GO verdict in ${GO_NO_GO_LOG} — run scripts/p5_killswitch.sh first (§16: P5 gates P6)"
require_impl executor/train_grpo.py "executor agent (Algorithm 3)"
require_impl eval/run_frontier.py "eval agent — frontier CLI runner (see contract in p5_killswitch.sh)"
mkdir -p "${GRPO_DIR}"

acquire_gpus 8

declare -A TASKS=(
    [qa]="${DATA_DIR}/hotpotqa_train.decontaminated.jsonl,${DATA_DIR}/nq_train.decontaminated.jsonl,${DATA_DIR}/musique_train.decontaminated.jsonl"
    [alfworld]="verl_agent_builtin"
)
declare -A DEV=( [qa]="${DATA_DIR}/hotpotqa_dev_frozen.jsonl" [alfworld]="verl_agent_dev" )

for DOMAIN in qa alfworld; do
    CKPT="${GRPO_DIR}/${DOMAIN}"
    if [ -d "${CKPT}" ]; then
        echo "[p6] ${DOMAIN}: ${CKPT} exists — skipping training"
    else
        banner "P6 — Algorithm 3, domain=${DOMAIN}"
        # TODO(GPU): long run (~1-3 days on 8xH200, §7). Divergence dashboard (V̂ vs
        # realized reward) is logged by train_grpo into wandb + ${CKPT}/divergence.csv (F6).
        python -m cassi.executor.train_grpo \
            --domain "${DOMAIN}" --tasks "${TASKS[${DOMAIN}]}" \
            --iteration 1 --seed "${SEED}" --lambda "${LAM}" \
            --coach "${STOPPER}" --arm shaped --step-credit "${STEP_CREDIT}" \
            --out "${CKPT}"
    fi
    banner "P6 — dev eval: ${DOMAIN} (CASSI iter-1 vs B1 ReAct, done-criterion)"
    for ARM_POLICY in "cassi_iter1:${CKPT}:${STOPPER}" "b1_react:$(cfg_get executor.base_model):none"; do
        IFS=':' read -r ARM POLICY MON <<< "${ARM_POLICY}"
        python -m cassi.eval.run_frontier \
            --arm "${ARM}" --policy "${POLICY}" --monitor "${MON}" --lambda-dial "${LAM}" \
            --domain "${DOMAIN}" --tasks "${DEV[${DOMAIN}]}" --seed "${SEED}" \
            --out-csv "${P6_CSV}" --append
    done
done

release_gpus

banner "P6 — done-criterion check (beats B1 on cost@iso-accuracy, both domains)"
python3 - "${P6_CSV}" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
ok = True
for domain in ("qa", "alfworld"):
    d = [r for r in rows if r.get("domain", "qa") == domain]
    cassi = [r for r in d if r["arm"] == "cassi_iter1"]
    b1 = [r for r in d if r["arm"] == "b1_react"]
    if not cassi or not b1:
        print(f"[{domain}] rows missing — eval incomplete"); ok = False; continue
    ca, cc = float(cassi[-1]["accuracy"]), float(cassi[-1]["cost_dollars"])
    ba, bc = float(b1[-1]["accuracy"]), float(b1[-1]["cost_dollars"])
    # B1 has no cost knob -> single point (§5.3): require accuracy >= B1 at lower cost
    passed = ca >= ba and cc < bc
    print(f"[{domain}] CASSI acc={ca:.4f} ${cc:.4f} | B1 acc={ba:.4f} ${bc:.4f} -> "
          f"{'PASS' if passed else 'FAIL'}")
    ok &= passed
print("P6 done-criterion:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
banner "P6 DONE"
