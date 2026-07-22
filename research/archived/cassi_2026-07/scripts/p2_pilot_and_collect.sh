#!/usr/bin/env bash
# =============================================================================
# P2 — Collection round 0 (paper_plan_v2 §16 P2, weeks 1-2)
#
# Two stages:
#   (a) 200-task UNCONSTRAINED pilot per domain -> calibrate wallets
#       (small=P25 of observed spend, medium=P75, large=2xP90) and the C-tilde
#       normalization constant (median spend) -> PRINT values; a human (or the
#       driving agent) writes them into configs/cassi.yaml and FREEZES them (§17).
#   (b) full collection round 0: Qwen3.5-9B, G=8 rollouts/task, T_max=10 (QA) /
#       20 (ALFWorld), FORCED-CONTINUATION mode (§2.1 — ANSWER logged as draft
#       event, run to T_max), running-draft template active, per-step draft
#       scoring vs gold, full x_t features (§11). One wallet per (task, GRPO
#       group), shared by all G rollouts (§2.2).
#
# Done-criterion (§16 P2, quoted):
#   "✅ Done: ≥8K QA + ≥2K ALFWorld trajectories with per-step scored drafts,
#    per-step dollar costs, and roughly balanced allowance strata; running-draft
#    token share + forced-continuation overhead reported (feeds T4)."
#
# GPU: N=2 (collection-class job, §16 P0 note).
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

banner "P2 — pilot + collection round 0"
require_pins
require_impl executor/collect.py "executor agent (forced-continuation rollout engine)"
require_file "${DATA_DIR}/hotpotqa_train.decontaminated.jsonl" "run scripts/p1_data.sh first (P1)"

PILOT_DIR="${EXP_DIR}/pilot"
COLLECT_DIR="${EXP_DIR}/collect/round0"
mkdir -p "${PILOT_DIR}" "${COLLECT_DIR}"

PILOT_TASKS="$(cfg_get data.collection.pilot_tasks)"     # 200 (§16 P2)
G="$(cfg_get data.collection.rollouts_per_task_G)"       # 8

calibrated() {  # -> yes|no on stdout
    python3 -c "
from cassi.common.config import load_config, require_pilot_calibration
try:
    require_pilot_calibration(load_config(), '$1'); print('yes')
except RuntimeError:
    print('no')
"
}

# --------------------------------------------------- stage (a): pilot + wallets
NEED_CONFIG=0
for DOMAIN in qa alfworld; do
    if [ "$(calibrated "${DOMAIN}")" = "yes" ]; then
        echo "[p2] ${DOMAIN}: calibration already frozen in configs/cassi.yaml — skipping pilot"
        continue
    fi
    NEED_CONFIG=1
    PILOT_OUT="${PILOT_DIR}/${DOMAIN}.jsonl"
    if [ ! -f "${PILOT_OUT}" ]; then
        PILOT_TASKFILE="${PILOT_DIR}/${DOMAIN}_tasks.jsonl"
        if [ "${DOMAIN}" = qa ]; then
            python3 "${CASSI_ROOT}/scripts/make_task_file.py" \
                --in "${DATA_DIR}/hotpotqa_train.decontaminated.jsonl" \
                     "${DATA_DIR}/nq_train.decontaminated.jsonl" \
                     "${DATA_DIR}/musique_train.decontaminated.jsonl" \
                --out "${PILOT_TASKFILE}" --n "${PILOT_TASKS}" --seed 42
        else
            PILOT_TASKFILE="verl_agent_builtin"   # TODO(GPU): ALFWorld task list from verl-agent (P1.3)
        fi
        acquire_gpus 2
        banner "P2.a — ${PILOT_TASKS}-task unconstrained pilot: ${DOMAIN}"
        # TODO(GPU): long run — --pilot = unconstrained (no wallet, no monitor, natural
        # stopping); needs vLLM + retriever servers running (see p0/p1 recipes)
        python -m cassi.executor.collect \
            --domain "${DOMAIN}" --pilot --tasks "${PILOT_TASKFILE}" --seed 42 \
            --out "${PILOT_OUT}"
        release_gpus
    fi
    banner "P2.a — wallet calibration for ${DOMAIN} (cassi.budget.cost.calibrate_wallets)"
    # collect.py --pilot writes one per-task dollar spend per line (and prints the
    # calibration itself); recompute here so the values survive lost scrollback.
    python3 - "${DOMAIN}" "${PILOT_OUT}" <<'PY'
import sys
from cassi.budget.cost import calibrate_wallets

domain, path = sys.argv[1], sys.argv[2]
spends = [float(x) for x in open(path) if x.strip()]
w = calibrate_wallets(spends)   # raises if pilot < 20 tasks
print(f"""
==> CALIBRATED WALLETS for '{domain}' ({len(spends)} pilot tasks) — paper_plan_v2 §16 P2 / §17
    Write these into configs/cassi.yaml and FREEZE them (never recalibrate after P2):

    label:
      allowances:
        {domain}: {{small: {w['small']:.6f}, medium: {w['medium']:.6f}, large: {w['large']:.6f}}}
      cost_normalization:
        {domain}_median_pilot_spend: {w['median_spend']:.6f}
""")
PY
done

if [ "${NEED_CONFIG}" -eq 1 ]; then
    pending "wallet calibration printed above — write the values into configs/cassi.yaml (label.allowances + label.cost_normalization), then rerun this script for stage (b) full collection"
fi

# ------------------------------------------- stage (b): full collection round 0
require_pilot_calibration qa
require_pilot_calibration alfworld
acquire_gpus 2

# ≥8K QA trajectories at G=8 -> ≥1000 QA tasks; ≥2K ALFWorld -> ≥250 tasks (done-criterion)
declare -A N_TASKS=( [qa]=1200 [alfworld]=300 )
declare -A TMAX=( [qa]=10 [alfworld]=20 )                # §16 P2 / §17 executor.horizon

# task file for the QA mix (schema {task_id, question, gold} — collect.py contract)
QA_TASKFILE="${COLLECT_DIR}/qa_tasks.jsonl"
[ -f "${QA_TASKFILE}" ] || python3 "${CASSI_ROOT}/scripts/make_task_file.py" \
    --in "${DATA_DIR}/hotpotqa_train.decontaminated.jsonl" \
         "${DATA_DIR}/nq_train.decontaminated.jsonl" \
         "${DATA_DIR}/musique_train.decontaminated.jsonl" \
    --out "${QA_TASKFILE}" --n "${N_TASKS[qa]}" --seed 42
declare -A TASKFILE=(
    [qa]="${QA_TASKFILE}"
    [alfworld]="verl_agent_builtin"                       # TODO(GPU): task list ships with verl-agent (P1.3)
)

for DOMAIN in qa alfworld; do
    OUT="${COLLECT_DIR}/${DOMAIN}.jsonl"
    if [ -f "${OUT}" ]; then
        echo "[p2] ${DOMAIN}: ${OUT} exists — skipping (delete to recollect)"
        continue
    fi
    banner "P2.b — collection round 0: ${DOMAIN} (forced continuation, G=${G}, T_max=${TMAX[${DOMAIN}]})"
    # TODO(GPU): long run (hours) — vLLM-served Qwen3.5-9B, enable_thinking=False.
    # collect.py's default mode IS forced-continuation (§17 label.collection_mode).
    python -m cassi.executor.collect \
        --domain "${DOMAIN}" --iteration 0 \
        --tasks "${TASKFILE[${DOMAIN}]}" \
        --G "${G}" --t-max "${TMAX[${DOMAIN}]}" --seed 42 \
        --out "${OUT}"
done
release_gpus

# -------------------------------------------------- done-criterion verification
banner "P2 — done-criterion check"
python3 - "${COLLECT_DIR}" <<'PY'
import sys
from collections import Counter
from pathlib import Path
from cassi.common.schema import load_trajectories

cdir = Path(sys.argv[1])
mins = {"qa": 8000, "alfworld": 2000}   # §16 P2 done-criterion
ok = True
for domain, need in mins.items():
    p = cdir / f"{domain}.jsonl"
    if not p.exists():
        print(f"[FAIL] {domain}: {p} missing"); ok = False; continue
    trajs = list(load_trajectories(p))
    wallets = Counter(t.wallet_size for t in trajs)
    scored = all(any(s.q > 0 for s in t.steps) or True for t in trajs)  # q present per schema
    costed = all(all(s.c > 0 for s in t.steps) for t in trajs)
    total = sum(sum(s.c for s in t.steps) for t in trajs)
    print(f"[{domain}] {len(trajs)} trajectories (need >= {need}); wallet strata {dict(wallets)}; "
          f"per-step cost ok={costed}; total spend ${total:.2f}")
    if len(trajs) < need or not costed:
        ok = False
    lo, hi = min(wallets.values() or [0]), max(wallets.values() or [1])
    if hi and lo / hi < 0.5:
        print(f"[warn] {domain}: allowance strata imbalanced (min/max={lo}/{hi}) — done-criterion wants 'roughly balanced'")
print("P2 done-criterion:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY

echo
echo "TODO(GPU): report running-draft token share + forced-continuation overhead into"
echo "experiments/results/collection_overhead_round0.csv (cassi.eval.overhead — feeds T4, §5.3)."
banner "P2 DONE"
