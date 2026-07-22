# F5 — RL training (GRPO with prompted-RM rewards)

**Goal:** the method arm A3: GRPO-train the executor on the 300 train tasks with
per-step judge rewards + the exact terminal economy, 1 seed, on verl.

## Reward wiring (the only novel plumbing)

- Rollouts: verl AgentLoop multi-turn on the F2 agent/env, harness mode `none`
  (episodes end at ANSWER or T_max — the policy must experience real termination
  economics), budget B drawn per (task, group), shared across the group's G=8.
- After each rollout batch: score all steps with the F3 judge (batched, cached),
  compute `r_t` per step and `R_final` per trajectory, feed verl step-level
  rewards.
- **Credit assignment:** per-step returns-to-go `R_t = Σ_{t'≥t} r_{t'} + R_final`,
  group-normalized; min-cohort guard (fewer than 3 group members alive at step t →
  trajectory-level baseline for those steps). Dr. GRPO length normalization ON.

## Config (initial; all in `configs/foundation.yaml`)

G=8, lr 5e-6, KL β=0.04, clip 0.2, rollout temp 1.0, ~2–3 epochs over the 300
tasks (≈ enough gradient steps to move behavior at this scale; adjust from the
micro-run), seed 42. Format reward weight 0.1. Judge α=0.2, headline λ from the
F2 pilot memo.

## Ordered gates (cheap failures first)

1. **`--dry-run`:** config parses, model loads, one fake batch flows through the
   reward path with a mocked judge. CPU-able.
2. **Micro-run:** 10 tasks, G=4, ~20 gradient steps, real judge. Checks: rewards
   are non-degenerate (per-step scores vary), KL sane, no crash, wandb dashboard
   (reward, entropy, steps-used, judge-vs-F1 divergence) populates.
3. **Full run:** 300 tasks, 1 seed. GPUs: acquire 4–8 via the ritual; judge load
   stays on the vLLM server.

**Algorithm choice (pre-registered): GRPO only, no PPO comparison.** The
foundation varies one thing (the reward signal), not two; PPO adds a critic
network + its hyperparameters for no answer we need now. v2.1's own A10 already
concluded algorithm comparisons are ablation-grade (100-task gaps are seed
noise; 2512.07611, 2504.11343) and scheduled its GRPO-vs-PPO smoke pilot for
the full plan. **Contingency:** if the micro-run shows visible GRPO collapse
despite the hygiene settings, PPO is the pre-agreed fallback — logged as a
config change, still not a comparison claim.

## Watch-items during the full run (logged, reported either way)

- **Judge-score vs realized-F1 divergence** — the frozen prompted judge is the
  most likely thing to be hacked (v2.1's own prediction for RM-P). Rising judge
  scores + flat/falling F1 = hacking; the curve is a foundation deliverable
  regardless of outcome.
- Steps-used distribution over training (should drift toward earlier ANSWER
  without F1 collapse).
- Entropy collapse / length pathologies (Dr. GRPO hygiene should prevent;
  still watched).

## Done criterion

Full run completes; checkpoint saved + config/git-hash snapshot; dashboard
exported; a 5-trajectory before/after qualitative diff written into the run log.

Depends on: F2, F3 (calibration gate MUST be green first), F1.
Feeds: F6.
