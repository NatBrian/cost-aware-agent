#!/usr/bin/env bash
# Run one complete λ arm of the ablation, start to finish, unattended.
#
# Usage: bash scripts/run_lambda_arm.sh <train_lambda> <tag>
#   e.g. bash scripts/run_lambda_arm.sh 0.0 lam0
#
# Sets economy.train_lambda (the REWARD's λ — economy.lambda, the evaluation
# yardstick, is deliberately left alone), serves the right model, then runs three
# rounds with the temp-1.0 health probe as a HARD gate between each. Stops on a
# failed probe rather than training the next round on a damaged policy — that is
# precisely what wasted the first attempt at this experiment.
#
# RESTARTABLE. A session ending killed the first launch mid-collection
# (2026-07-29), so the script now resumes: rounds whose checkpoint already exists
# are skipped, and collection itself resumes from its shards because
# run_collection skips (task_id, rollout) pairs it has already done. Re-running
# this command after any interruption is safe and is the intended recovery.
#
# Arms must run SEQUENTIALLY: they share one config file and one pair of GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
LAM=${1:?usage: run_lambda_arm.sh <train_lambda> <tag>}
TAG=${2:?usage: run_lambda_arm.sh <train_lambda> <tag>}
BACKUP=/mnt/src/liangsheng/cassi_foundation
GPU=$(grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9,]+' .gpu_hold | cut -d, -f2)

echo "=== ARM $TAG : train_lambda=$LAM ==="
sed -i "s/^  train_lambda: .*/  train_lambda: $LAM                # ablation arm $TAG/" configs/foundation.yaml
grep -E "^  (lambda|train_lambda):" configs/foundation.yaml

# ---- where did a previous attempt get to? ------------------------------------
INIT=""; START=1
for R in 1 2 3; do
  for C in "$PWD/experiments/results/train/${TAG}_round$R/checkpoint" \
           "$BACKUP/checkpoints/${TAG}_round$R"; do
    # A checkpoint EXISTING does not mean it passed its health probe -- checkpoints
    # are written and backed up BEFORE the probe runs. Resuming on existence alone
    # would train the next round on a damaged policy. Require the HEALTHY marker.
    # (audit 2026-08-06)
    # The marker sits beside the checkpoint dir locally ("<round>/HEALTHY") but
    # inside it in the backup ("checkpoints/<tag>_round<R>/HEALTHY"), so check both.
    if [ -f "$C/config.json" ] && { [ -f "$C/HEALTHY" ] || [ -f "$(dirname "$C")/HEALTHY" ]; }; then
      INIT="$C"; START=$((R + 1)); break
    fi
  done
done
[ "$START" -gt 1 ] && echo "RESUMING: rounds 1..$((START-1)) already trained; continuing from $INIT"
if [ "$START" -gt 3 ]; then echo "=== ARM $TAG ALREADY COMPLETE — $INIT ==="; exit 0; fi

# ---- serve whatever the next round must collect from -------------------------
SERVE=${INIT:-BASE}
echo "=== serve $( [ -z "$INIT" ] && echo 'BASE model' || echo "$INIT" ) ==="
PID=$(pgrep -u "$(whoami)" -f "[v]llm serve.*port 8378" | head -1) && [ -n "$PID" ] && kill "$PID" && sleep 20 || true
if [ -z "$INIT" ]; then
  CUDA_VISIBLE_DEVICES=${GPU:-1} nohup bash scripts/serve_executor.sh > "$BACKUP/${TAG}_serve.log" 2>&1 &
else
  CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv-gpu3/bin/vllm serve "$INIT" \
    --served-model-name Qwen/Qwen3.5-9B --port 8378 --dtype auto \
    --max-model-len 16384 --gpu-memory-utilization 0.85 > "$BACKUP/${TAG}_serve.log" 2>&1 &
fi
for i in $(seq 1 80); do
  curl -s -m 5 http://127.0.0.1:8378/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    2>/dev/null | grep -q '"content"' && { echo "model serving"; break; }
  sleep 15
  [ "$i" = 80 ] && { echo "ARM $TAG FAILED: model would not serve"; exit 1; }
done

# ---- rounds ------------------------------------------------------------------
for R in $(seq "$START" 3); do
  DIR="experiments/results/train/${TAG}_round$R"
  echo "=== $TAG round $R ==="
  if [ -z "$INIT" ]; then bash scripts/e5_round.sh "$DIR" 300 8 150
  else                    bash scripts/e5_round.sh "$DIR" 300 8 150 "$INIT"; fi

  # BACK UP BEFORE THE GATE. Ordering these the other way round lost the
  # λ=1.0 round-3 checkpoint from /mnt/src on 2026-07-30: its probe failed, the
  # script exited, and the artefact that the pre-registered verdict was computed
  # from survived only on the ephemeral overlay. A failed probe is exactly when
  # you most want the evidence kept — it is a result, not a discard.
  mkdir -p "$BACKUP/checkpoints" "$BACKUP/trajectories"
  cp "$DIR/rollouts.jsonl" "$BACKUP/trajectories/${TAG}_round${R}_rollouts.jsonl" 2>/dev/null || true
  for f in round_summary.json train_log.jsonl divergence.jsonl; do
    [ -f "$DIR/$f" ] && cp "$DIR/$f" "$BACKUP/trajectories/${TAG}_r${R}_$f" || true
  done
  cp -r "$DIR/checkpoint" "$BACKUP/checkpoints/${TAG}_round$R" 2>/dev/null || true

  echo "=== $TAG round $R HEALTH GATE ==="
  if ! bash scripts/probe_policy_health.sh "/tmp/probe_${TAG}_r$R.jsonl"; then
    echo "ARM $TAG STOPPED: health probe FAILED after round $R — refusing to train on a damaged policy"
    echo "(round $R artefacts ARE backed up to $BACKUP — a failed probe is a result)"
    exit 1
  fi
  # Mark the round HEALTHY. Checkpoints are written and backed up BEFORE the probe
  # (deliberately — a failed probe is evidence worth keeping), so "a checkpoint
  # exists" does NOT mean "this policy passed". Selecting the newest existing
  # checkpoint would have picked ctrl_round3, which scored 20.5% malformed against
  # a 10% gate. Downstream selection reads this marker, never the directory
  # listing. (2026-08-01)
  touch "$DIR/HEALTHY"
  cp "$DIR/HEALTHY" "$BACKUP/checkpoints/${TAG}_round$R/HEALTHY" 2>/dev/null || true

  INIT="$PWD/$DIR/checkpoint"
  # keep only the newest local checkpoint: 19G each, all are on /mnt/src
  [ "$R" -gt 1 ] && rm -rf "experiments/results/train/${TAG}_round$((R-1))/checkpoint" || true
done

echo "=== ARM $TAG COMPLETE — final checkpoint: $INIT ==="
