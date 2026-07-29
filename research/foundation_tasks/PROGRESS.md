# Foundation implementation progress

> Living tracker — updated at every stage boundary. Stage = one commit.
> Decisions locked 2026-07-22: GPUs allowed via ritual; judge = config placeholder
> (URL to be provided before F3 calibration); commit per stage, push to main
> (NatBrian); all five design defaults accepted as documented.

## Implementation stages (code-complete before experiments)

- [x] **I0 — Restructure (F0)** ✅ 2026-07-22: plan docs committed (8aef0ee) ·
  79G data artifacts → `research/data_shared/` · run-1 reports + paid-measurement
  records (dataset_manifest.csv, RM-P heldout, trivial baselines) →
  `research/reports_run1/` · cassi archived → `research/archived/cassi_2026-07` ·
  `foundation/` skeleton + configs/foundation.yaml + common.py + venv
  (python3.12 via /mnt/src, get-pip bootstrap — system 3.10 lacks ensurepip) ·
  Makefile `make venv`/`make test` · 5 skeleton tests green · CLAUDE.md rewritten
- [x] **I1 — Data (F1)** ✅ 2026-07-22: `collect/sampling.py` (stratified, seeded,
  deterministic) + `scripts/f1_data.py` → frozen train-300 (194 med/55 easy/51 hard)
  + dev-200 + SHA256 manifest; rerun verified byte-identical; overlap check passes ·
  `envs/retrieval_client.py` + `scripts/serve_retrieval.py` (E5+FAISS over rescued
  64G index — dims verified 768/21M rows; live check deferred to I2 GPU smoke) ·
  14 tests green. **Finding logged:** HotpotQA dev split is 100% hard-level by
  design → dev-200 is all-hard while train is mixed (matches F1 limitation #3;
  report must mention it)
- [x] **I2 — Agent/harness/collection (F2)** ✅ 2026-07-22 (code): prompts +
  facts-not-advice tracker (a0 budget-blind verified by test) · episode loop w/
  3 harness modes, draft retry/persistence, per-step draft F1 · vLLM client +
  serve_executor.sh · schema validator · resumable collection CLI w/ seeded
  per-(task,group) wallet draw · 27 CPU tests green (commit 4bb6458).
  **Pending:** live 2-GPU smoke (20-task batch per mode) — scheduled as the
  pre-pilot checkpoint alongside retrieval-server first start
- [x] **I3 — Reward (F3)** ✅ 2026-07-22: rubric_v1 prompts (anchored bits +
  worked examples, gold-free — tested) · judge client (parse retry → neutral 0.5
  logged; disk cache; call/token stats; placeholder endpoint refused w/ hint) ·
  rewards (8-level table, malformed=worst, terminal economy, returns-to-go —
  plan worked example reproduced) · calibration sheet + agreement gate
  (mean≥.80, floor .70) · 36 tests green (commit b48ff3c)
- [x] **I4 — Metrics/eval (F4+F6)** ✅ 2026-07-22: `eval/metrics.py` (one scorer
  path, bootstrap CIs, per-task paired deltas, row-count + stale-utility guards) ·
  `eval/gate_check.py` (plan §6 itemized, thresholds in config `gate:`) ·
  `eval/build_rows.py` + `scripts/f4_baselines.sh` · 43 tests (commit 8395c82)
- [x] **I5 — Training (F5)** ✅ 2026-07-22 (CPU side): `train/advantages.py`
  (min-cohort guard tested incl. lone-survivor) · `train/reward_adapter.py`
  (+DivergenceLog = Fig-3 hacking curve) · `train/grpo_runner.py --dry-run`
  green (24 eps, nonzero advantages). **Pending:** verl glue in `_run_real`,
  wired + verified at the E-d micro-run against pinned verl (commit babfc2b)
- [x] **I6 — Analysis (F7)** ✅ 2026-07-22: 3 figure scripts (dataviz-validated
  entity-fixed palette, ALL PASS) · report generator (numbers auto-filled,
  prose TODOs) · adapted PPTAgent diagnostic (offline) · 52 tests (commit f915d73)
- [x] **I7 — Readiness** ✅ 2026-07-22: EXPERIMENT_CHECKLIST.md (ordered E-a…E-g
  w/ gates + Brian's 3 inputs) · make test/dry-run/data targets · 52 tests green,
  dry-run green → **CODE-COMPLETE; READY TO START EXPERIMENTS** (pre-flight GPU
  smoke is checklist step 1)

## Experiment sequence (after I7 — each gated per the F-docs)

- [x] E-a ✅ 2026-07-22: 200-episode pilot; knee at step 3; overthinking in
  24/200; budgets {2,4,8} + λ=0.3 FROZEN from data (memo:
  foundation/experiments/reports/pilot_memo.md; auto-approved under mandate)
- [x] E-b ✅ 2026-07-22: calibration gate PASSED (mean .848; all bits ≥.70;
  was_needed 1.0). Label-review round documented (4 author-label fixes, 4
  judge attention-misses kept+logged). Judge biases for divergence read:
  lenient-on-redundancy, late-step supported noise
  (report: calibration_report.md)
- [x] E-c ✅ 2026-07-22 (**dev-200 look #1 of 3**): 1400 episodes. A1 beats A2
  everywhere (paired); A2 B=2 ceiling confirmed (F1 .221 vs .478); A1
  budget-adapts ~1 step. RL bar at B=4: U>.205, F1≥.361, self-stop≥.70
  (report: baselines_report.md)
- [x] E-d ✅ 2026-07-23 (GATE PASSED): verl replaced by custom lean GRPO trainer
  (rationale in commits e4fe6b6/18542f1). Micro-round end-to-end: 124/124
  samples zero token mismatch; update-1 ratio 1.003 / KL 6e-4 (math verified);
  KL-blowup tuned (shuffle + accum 32); multimodal-tensor merge at save;
  trained checkpoint SERVES ('READY')
- [x] E-e ✅ 2026-07-28/29: 3 rounds × (300 tasks × G=8), all health gates passed,
  no reward hacking (judge flat .842/.841/.843 while realized F1 rose
  .591/.611/.622). Checkpoint selected on val-50 utility: round 3.
  (An earlier attempt on 2026-07-23 was lost to the container wipe; restarted
  from base.)
- [ ] **λ ABLATION (post-verdict, 2026-07-29)** — the GO's mechanism is not the
  predicted one (steps did not fall), so arms at train_lambda ∈ {0, 0.3, 1.0}
  test whether the step-cost term does anything. Pre-registered decision rule in
  experiments/reports/ablation_preregistration.md. λ=0 arm running.
- [x] E-f ✅ 2026-07-29 (**dev-200 look #2 of 3**): A3 harness-off/on × 3
  budgets + oracle replay, merged with surviving baseline rows → gate_check
- [x] E-g ✅ 2026-07-29: figures + report + **GO verdict logged** → tagged
  `foundation-run-1`

## Autonomy mandate (Brian, 2026-07-22)

Full autonomy to run E-a…E-g end-to-end and REFINE UNTIL STABLE (RL beats
baselines; PPO fallback if GRPO collapses; fix any issue; websearch when
knowledge-limited). Standing 2-GPU hold at all times, even between experiments.

**Anti-overfitting policy (self-imposed, active from now):**
1. dev-200 stays untouched during ALL refinement. Tuning/iteration decisions use
   a 50-task VALIDATION slice carved from train-300's remainder pool (never the
   frozen dev set, never the 300 training tasks themselves where avoidable).
2. Every dev-200 evaluation is logged in this file with date + reason; target
   ≤3 total dev evaluations (baselines once, headline A3 once, +1 reserve).
   Iterating against dev until the gate passes would BE overfitting — the gate
   must pass on the first or second dev look, driven by validation-slice signals.
3. Refinement order on failure (cheapest first): rubric/judge → reward scales
   (alpha, lambda) → RL hyperparams (lr, KL, epochs) → algorithm (PPO fallback)
   → data size (300→500). One change per iteration, logged.

## Log

- 2026-07-29 23:45 **JUDGE SERVER OUTAGE — λ=0 round 1 training STOPPED
  deliberately.** The shared Qwen3.6-27B at 122.11.227.227:6101 went down
  mid-round: 791 logged failures, all `transport: ConnectionError`, confirmed by
  three direct probes (the old gemma judge on :6102 still answers, so it is that
  server, not our network).
  **Why this had to be stopped rather than ridden out:** the judge client returns
  a neutral 0.5 on transport failure — correct behaviour, it must never crash the
  reward pipeline — but that silently removes the per-step reward. A round
  trained through the outage is NOT the λ=0 condition (judge shaping present,
  step COST absent); it is a terminal-reward-only run wearing the arm's label,
  and it would have quietly invalidated the whole ablation. Killed the trainer
  and the arm script.
  **Cost of the interruption is small, by earlier design:** the 2400 collected
  rollouts are judge-independent and preserved; good judgements are cached
  (18,262 entries); and the 2026-07-28 audit fix means **neutral fallbacks are
  never cached**, so a rerun re-queries exactly the failed steps and reuses the
  rest. We lose training compute, not judgements — this is the concrete payoff of
  that fix.
  **Recovery is automatic:** `scripts/wait_for_judge_then_run.sh` polls the judge
  once a minute for up to 48h and `exec`s the (restartable) arm the moment a real
  judged completion succeeds. Readiness is a genuine completion, not /v1/models,
  which can answer while an engine is dead — the same false-readiness trap found
  on our own executor.
  **Fallbacks if the outage is long** (decide at the next wake-up, in order):
  (1) keep waiting — cheapest, preserves the judge that rubric_v4 was calibrated
  against; (2) serve Qwen3.6-27B locally on GPU 0 alongside retrieval (27B bf16
  ~54G, GPU 0 has 143G and retrieval uses ~2G) — costs a ~54G download but
  removes the external dependency entirely; (3) switching to the gemma judge on
  :6102 is REJECTED — rubric_v4 was calibrated against Qwen and the λ=0.3 arm was
  trained with it, so changing judges mid-ablation would confound every arm.

- 2026-07-29 **SESSION TEARDOWN KILLED THE RUN — and the GPU hold with it.**
  The λ=0 arm died mid-collection at 1167/2400 episodes when the previous session
  ended; `setsid nohup` did NOT protect it. Retrieval server, executor and the
  `acquire_gpus.py` hold were all gone too. Recovered: re-acquired GPUs 0+1
  immediately (**the other five cards are now at 124-143G under other users — we
  took the last free pair**, which is exactly why Brian wants the hold standing),
  restarted retrieval, relaunched the arm.
  **Fix so this costs less next time: `run_lambda_arm.sh` is now RESTARTABLE.**
  It detects which rounds already have a checkpoint (local or on /mnt/src),
  resumes from the next one, serves the correct model for that round rather than
  always the base, and exits cleanly if the arm is already complete. Collection
  was already resumable (run_collection skips finished (task_id, rollout) pairs),
  so the 1167 episodes were not wasted. Re-running the same command after any
  interruption is now the intended recovery path.
  Standing lesson: background work launched from a session is not durable, so
  every long run needs (a) resumability and (b) a scheduled wake-up that
  re-checks and relaunches, not just a watcher that dies with its parent.

- 2026-07-29 **SAFETY NEAR-MISS: my scripts tried to kill another user's vLLM.**
  `run_lambda_arm.sh`, `f6_eval.sh` and `select_checkpoint.sh` all used
  `pgrep -f "[v]llm serve" | head -1` to find "the" server to stop before
  serving a new checkpoint. On this shared box that pattern matches ANY user's
  vLLM. It selected PID 2105447, owned by **yongyue** — the user CLAUDE.md
  explicitly says never to interrupt — and only Linux permissions stopped it
  ("Operation not permitted"). Fixed in all three: `pgrep -u "$(whoami)" -f`.
  `e5_round.sh` was already correctly scoped with `pkill -u "$(whoami)"`, which
  is where the pattern should have been copied from.
  Two lessons: (1) any process-matching command on a shared machine must be
  user-scoped, always; (2) I wrote the unscoped version three times because I
  copied my own earlier line instead of the safe one that already existed in the
  repo. Same failure shape as the arm-blending bug — a defect propagated by
  copy-paste rather than reasoned about once.
- 2026-07-29 **λ ABLATION LAUNCHED (arm lam0, train_lambda=0.0).** Pre-registered
  first (experiments/reports/ablation_preregistration.md, committed before the
  run). `economy.train_lambda` now drives the reward while `economy.lambda` stays
  at 0.3 as the fixed evaluation yardstick, so arms trained under different λ are
  scored on one scale. scripts/run_lambda_arm.sh runs a whole arm unattended:
  serve base -> 3 rounds -> health probe as a HARD gate after each -> back up
  checkpoint/rollouts to /mnt/src -> keep only the newest local checkpoint (19G
  each). Local round1/round2 checkpoints from the λ=0.3 run deleted after
  verifying the /mnt/src copies carry weights + config.

- 2026-07-29 **POST-VERDICT ANALYSIS — the GO is real but the MECHANISM is not
  the predicted one. Report §4 corrected.**
  Steps did NOT fall: A3 uses 2.96/3.61/4.32 vs A1's 2.96/3.54/3.98. Decomposing
  the utility gain: B=4 dU +.084 = quality +.089 + step-cost −.005. **All of the
  gain is answer quality; the step term works slightly AGAINST A3.** Stop-step
  distributions are near-identical (A1 47/79/31/31 vs A3 48/77/30/32 at steps
  2/3/4/5) and self-stop is slightly LOWER after training (−.010 at B=4, −.060
  at B=8).
  **I had to withdraw a claim I made hours earlier.** The first report draft
  presented the harness-off/on gap as internalization evidence. The control —
  A1 vs A2 is the SAME untrained policy with the harness off vs on — shows
  untrained gaps +.114/+.026/+.046 against trained +.185/+.023/+.005. At B=4 and
  B=8 the trained gap is no larger. The gap measures "cutting an agent off
  hurts", true of any agent. Not internalization. Report §4 rewritten, §8
  amended.
  **Diagnosis of why stopping did not move:** at B=4, λ=0.3 prices a step at
  .075 utility while the pilot's 4th step buys ≈+.06 F1 — continuing is roughly
  break-even, so there was almost no gradient toward stopping earlier. λ was set
  to place the optimum at the knee; that is not the same as making the pull
  toward it strong. **λ=0.3 is too weak to change behaviour.**
  **Next experiments, in priority order (in FOUNDATION_EXPLAINED.md §11):**
  (1) λ=0 ablation — if F1 and steps are unchanged, the economic term did
  nothing and this is task-skill RL; (2) λ sweep to 1.0/1.5 — if behaviour moves,
  that is the dose-response evidence the claim actually needs; (3) more seeds.
  DO NOT spend the last dev-200 look until a genuinely new method is ready.
- 2026-07-29 **Wrote research/FOUNDATION_EXPLAINED.md** — plain-language end-to-end
  explanation for Brian as author: setup, how GRPO training works, results,
  the step-count finding, the anti-overfitting design, the anti-hacking evidence,
  what is solid vs shaky, and what to run next.

- 2026-07-29 **🎯 FOUNDATION VERDICT: GO** (dev-200, B=4, pre-registered gate).
  | condition | requirement | result |
  | 1 utility | A3 > A1 and A3 > A2 | **.2894** vs .2052 / .1796 PASS |
  | 2 self-stop | >= .70 | **.775** PASS |
  | 3 no collapse | F1 >= .361 | **.560** PASS |
  A3 wins utility at EVERY budget (.116/.289/.386 vs A1 .035/.205/.352). Paired
  deltas at B=4 exclude zero: vs A1 U +.084 [.025,.147], F1 +.089; vs A2 U +.110
  [.048,.171], F1 +.149. A3's F1 (.560) is the HIGHEST of any arm — it improves
  quality AND efficiency, not one at the other's expense.
  **Tight-budget result is the headline:** at B=2 A2's cutoff collapses F1 to
  .221 while A3 reaches .559 on the SAME 2.96 mean steps as A1. Enforcement
  cannot make a draft ready early; changed behaviour can.
  **Internalization:** A3 harness-off beats harness-on at every budget, gap
  growing as the budget tightens (+.185 / +.023 / +.005). Arming the harness on
  the trained policy at B=2 destroys F1 (.559 -> .231).
  **Reward hacking ruled out:** judge flat .842/.841/.843 while realized F1 rose
  .591/.611/.622. The failure calibration predicted did not occur.
  Report: experiments/reports/foundation_report.md (§7 carries all limitations).
- 2026-07-29 **THIRD instance of the arm-blending bug, found in E-g.** The first
  generated report showed A3 at F1 .530 / self-stop 39% — it was averaging
  harness-off, harness-on and oracle rows, exactly the bug the audit fixed in
  `gate_check.py`. `report.py` and `figures.py` had it too and were never
  checked. Fixed properly this time: one shared `eval.metrics.canonical_rows()`
  used by the gate, the report and the figures, plus a regression test (61 green).
  Lesson: when an audit finds a bug of a given SHAPE, grep for the shape, not
  just fix the instance. Selecting on `arm` alone was wrong in three files.

- 2026-07-29 **CHECKPOINT SELECTED: round 3** (validation slice, utility).
  | ckpt | U | F1 | steps | self_stop | malformed(temp 0) |
  | r1 | .3884 | .664 | 3.68 | .78 | 3.8% |
  | r2 | .3850 | .640 | 3.40 | .84 | 6.5% |
  | **r3** | **.4738** | .733 | 3.46 | .82 | 5.2% |
  Selection CONFIRMED the default rather than overriding it. The round-3 alarm
  (temp-1.0 malformed 9.0%) did not reproduce at temp 0, which is the evaluation
  setting: 5.2% there. Real signal, checked rather than assumed — assuming either
  way is the mistake that cost the first run.
  NOTE for the report: these val-50 numbers must NOT be read as a preview of the
  verdict. val-50 is mixed easy/medium/hard; dev-200 is 100% hard by construction
  (HotpotQA's dev split is all hard-level), so F1 .733 on val says nothing direct
  about the .361 F1 floor on dev.
- 2026-07-29 **E-f LAUNCHED — dev-200 look #2 of <=3.** Reason: the headline A3
  evaluation, per plan §6, on the round-3 checkpoint selected above. A3
  harness-off AND harness-on x budgets {2,4,8} at temp 0, plus the oracle
  forced-continuation replay (analysis only, EXCLUDED from the gate population —
  its answered_at is logged while the episode keeps running, so it is not a stop
  decision). Merged with the surviving baseline rows via scripts/f6_build_eval.py
  (they are per-task rows, not episodes, so re-collecting them would spend a
  third dev look and break comparability with the bar).
  Bar at B=4: U > .205 (A1) and .180 (A2), F1 >= .361, self-stop >= .70.

- 2026-07-29 **E-e COMPLETE (3 rounds). Reward hacking DID NOT occur.**
  Round 3: 150 updates, 8001 samples kept / 4 dropped, mean KL 0.0027.
  **Three-round divergence trend — the predicted failure did not materialise:**
  | round | judge | realized F1 | gap |
  | 1 | .842 | .591 | +.251 |
  | 2 | .841 | .611 | +.230 |
  | 3 | .843 | .622 | +.221 |
  Calibration showed the judge over-approves stopping, which predicts judge score
  climbing against flat/falling F1. Observed: judge FLAT across all three rounds,
  F1 monotonically RISING, gap monotonically NARROWING. The policy improved at
  the task, not at pleasing the judge. This was the most likely route to a
  MISLEADING GO, and it is closed.
- 2026-07-29 **Round-3 probe PASSED but is trending wrong — checkpoint selection
  moved to the validation slice.**
  | probe | malformed | hit_cap | F1 | steps |
  | after r1 | 4.3% | 0.0% | .612 | 3.48 |
  | after r2 | 2.0% | 0.0% | .572 | 3.67 |
  | after r3 | **9.0%** | **2.5%** | .598 | 3.62 |
  Malformed nearly quintupled from round 2 and sits just under the 10% gate;
  hit_cap went non-zero for the first time. That is the early signature of the
  sampling-distribution damage that wasted the first run — passing, but pointed
  the wrong way. So "evaluate the last round" is no longer a safe default.
  **DECISION (logged): pick the checkpoint by UTILITY on val-50**
  (`scripts/select_checkpoint.sh`, all three rounds, B=medium, temp 0, harness
  off). Choosing a checkpoint IS a tuning decision, so it reads val-50 and never
  dev-200 (anti-overfitting policy). Selection is on U = F1 - lambda*(steps/B),
  the quantity the run optimises — selecting on F1 alone would reward exactly the
  over-continuation the method exists to cure. The choice, and every round's
  validation numbers, go in the final report so the selection is visible rather
  than implied.

- 2026-07-29 **E-e ROUND 2 COMPLETE — probe PASSED, round 3 launched.**
  150 updates, 8203 samples kept / 2 dropped, mean KL 0.081 (r1: 0.052) —
  rising slightly but an order of magnitude inside trouble. Entropy mean .412,
  min .079, no decay toward collapse.
  **HARD GATE — probe at temp 1.0 on val-50: PASS.** malformed 2.0% (r1 4.3%),
  hit_cap 0.0%, F1 .572, steps 3.67.
  **REWARD-HACKING DIAGNOSTIC — the predicted failure is NOT happening.**
  Calibration showed the judge over-approves stopping, which predicts: judge
  score climbs while realized F1 stays flat or falls. Observed instead:
  | round | judge | realized F1 | gap |
  | 1 | .842 | .591 | +.251 |
  | 2 | .841 | .611 | +.230 |
  Judge FLAT, F1 RISING, gap NARROWING — the policy is improving at the task,
  not at pleasing the judge. Two points is not a trend; round 3 is where a
  hacking signal has the most opportunity to appear, so the read stays open.
  **Counter-signal to watch, logged now so it is not rationalised later:** the
  probe's own F1 fell .612 -> .572 while mean steps rose 3.48 -> 3.67, i.e. the
  opposite direction from the training-set trend. n=40 episodes at temp 1.0 makes
  this well inside noise, but if round 3's probe repeats it, that is a real
  divergence between train and validation behaviour and must be reported.
  Round 2 rollouts + checkpoint backed up to /mnt/src.

- 2026-07-28 **E-e ROUND 1 COMPLETE — health probe PASSED, round 2 launched.**
  150 updates (hit the cap), **8160 samples kept / 1 dropped**, judge 7473 calls
  with 1 parse failure and 0 transport failures (6.3M tokens).
  **mean KL 0.052** — against the first run's round-1 blowup to ~626. One
  transient spike is visible (a sampled update at KL 1.42, ratio 2.5, 30%
  clipped); the log-ratio clamp bounded it and it did not propagate, which is
  exactly what that clamp was added for.
  **Entropy shows no collapse**: noisy per-update (single-sample logging) but
  ends higher than it starts (0.13 -> 0.95), not decaying toward a degenerate
  policy.
  **HARD GATE — probe_policy_health at temp 1.0 on val-50: PASS.**
  malformed 4.3% (gate <10%), hit_cap 0.0% (gate <15%), F1 .612, mean steps 3.48.
  This is the gate the first run lacked: there, lr 5e-6 left 71.6% malformed at
  temp 1.0 while temp-0 checks looked fine, and round 2 trained on the wreckage.
  **Divergence baseline for the hacking read:** judge score mean .842 vs realized
  F1 mean .591 across 300 group rows. A level gap is not evidence of hacking (the
  two are not on one scale — the judge grades process, F1 grades the answer);
  the diagnostic is the TREND over rounds. Given calibration showed this judge
  over-approves stopping, the shape to watch in rounds 2-3 is judge score rising
  while F1 stays flat or falls.
  Round 1 rollouts + logs + checkpoint backed up to /mnt/src. Round 2 launched
  from the round-1 checkpoint.

- 2026-07-28 **MICRO-ROUND GATE PASSED — full training loop validated end to
  end** in the rebuilt environment. collect(--train, logprobs) -> judge ->
  advantages -> train -> save+merge -> SERVE, all green:
  48 episodes / 12 groups; **177 samples kept, 0 dropped** (no chat-template
  drift across the transformers 4 -> 5 boundary, which was the risk the venv
  re-split reintroduced); judge 0 calls / 159 cache hits (persistent cache
  working); 5 updates at ratio 1.009, mean KL 0.0025, 5% clipped — against the
  first run's round-1 KL blowup to ~626, so the gentler hypers + log-ratio clamp
  hold; **`merged 348 untrained tensors into checkpoint (427 trained, 775
  total)`** — the audit's sharded-save fix works and reproduces the base model's
  exact tensor count; trained checkpoint serves and generates.
  This is the gate F5 mandates, and it earned its cost twice over: it caught the
  venv merge AND validated a save path that had never run, either of which would
  otherwise have surfaced only after a full 300-task collect+judge+train pass.

- 2026-07-28 **MICRO-ROUND CAUGHT A BREAK I INTRODUCED — the two GPU venvs
  cannot be merged.** Collection and judging passed (48 episodes / 12 groups,
  177 samples kept, 0 dropped — the >5% drop guard clean), then training died at
  model load: `ValueError: checkpoint has model type qwen3_5 but Transformers
  does not recognize this architecture`. transformers 4.57.6 cannot load Qwen3.5;
  vLLM serves it through its OWN registry, so serving never noticed. The chain is
  forced: driver 570.x = CUDA 12.8 -> every wheel cu128 -> vllm pinned to 0.17.1
  -> torch 2.10.0 + transformers 4.57.6 -> cannot train. Training needs
  transformers >= 5.
  **This was my error.** On 2026-07-28 I consolidated `.venv-gpu` and
  `.venv-train` into one venv, reasoning that a single environment removes
  chat-template drift between rollout and training time. e5_round.sh's original
  `.venv-train/bin/python` for training and `.venv-gpu/bin/vllm` for serving was
  LOAD-BEARING, not incidental clutter. Re-split: `.venv-gpu3` serves,
  `.venv-train` (transformers 5) trains; e5_round.sh points training back at it;
  requirements-gpu-pinned.txt now documents BOTH stacks and why merging fails.
  Second time this session that "tidying" an artifact from the first run cost a
  cycle (the other was rewriting the pin file). The first run's odd-looking
  choices deserve a `git log` before they are improved.
  Residual risk to watch at the re-run: the tokenizer now renders under
  transformers 5 while rollouts were produced by vLLM's transformers 4 — if the
  chat template drifts, samples get dropped, and the trainer aborts above 5%
  rather than training on a biased subset.

- 2026-07-28 **E-b RUN 2: GATE PASSED (mean+floor), FAILED strict per-bit.**
  New instrument: 150 steps from the run-2 pilot, labeled by 10 FRESH no-context
  subagents (data block + neutral bit definitions only — never the rubric prompt,
  judge output, or session context). n=70-80/bit, CI ~+-0.09 vs +-0.17 before.
  rubric_v4: mean .847, bits — not_redundant .957, supported .863, new_info .843,
  was_needed .800, nothing_left .775. Implemented gate (mean>=.80, floor .70)
  PASSES; the plan's literal >=.80-per-bit FAILS on nothing_left.
  **v3 -> v4 was one edit and it corrected MY error:** v3 told the judge to
  answer YES on nothing_left when undecided, because the old anchored 50-row
  sheet made it look too strict. On a trustworthy instrument the bias is the
  opposite. v4 requires positive evidence; the bit moved .688 -> .775 and false
  approvals halved (23 -> 12).
  **Residual bias is the dangerous direction:** the judge still over-approves
  stopping (nothing_left h0j1 12 vs h1j0 6; was_needed h1j0 14 vs h0j1 0). Since
  nothing_left IS the stop decision, a reward that over-approves stopping teaches
  premature stopping and would produce a GO for the wrong reason. Guards: gate
  cond3 (A3 F1 >= A2 F1 - .05) and the per-group divergence curve.
  **DECISION (logged): proceed to E-e on rubric_v4.** Run 1 was accepted at mean
  .848 with nothing_left .769 — it also failed the strict reading, and was worse
  on this bit; we are better calibrated on far better evidence. .775 vs .80 at
  CI +-0.09 is not distinguishable from the line, and further prompt revision
  would fit the sheet (already demonstrated at n=25/bit). BOTH readings go into
  every downstream artifact. Report: experiments/reports/calibration_report_v2.md
  **Honesty item: these are MODEL labels, not Brian's — the gate measures
  judge-labeler consistency and F3's human-label requirement is formally unmet.**

- 2026-07-28 **SMOKE PASSED (20 tasks x 3 modes), with one finding held open.**
  Schema 0 errors in all three modes; retrievals sane (0 thin observations);
  F1 .462/.414/.454 and mean steps 3.90/3.05/4.00 — close to the surviving
  E-c baselines (A1 B=4: F1 .471, steps 3.55), so the rebuilt pipeline
  reproduces prior behaviour.
  **Finding: 20.5% of steps are malformed in mode=none** (16/78; 0% in enforce,
  8% in forced_continuation). Root cause is NOT format disobedience: the
  malformed steps have `raw_len` 1961–2228 chars against `max_tokens_per_step:
  512`, i.e. the model writes a long multi-sentence THOUGHT, gets truncated
  mid-sentence, and the required ACTION / BEST ANSWER lines never arrive.
  **Decision: do NOT change the scaffold yet.** Raising max_tokens or tightening
  the prompt would give A3 a scaffold the surviving A0/A1/A2 baselines never had,
  and those baselines are the most valuable asset we still hold — re-collecting
  them would spend one of the ≤3 dev looks. The scaffold must be a constant
  across arms (plan §2). Re-decide from the E-a pilot's 200 episodes, where n is
  large enough to tell a real rate from small-sample noise; if it stays ~20%,
  weigh a scaffold change + baseline re-run against living with it (the reward's
  format term already penalises malformed steps, so A3 should learn to avoid
  them — which is a legitimate result, not a confound, as long as the arms share
  the scaffold).
- 2026-07-28 **TWO ROOT CAUSES FOUND (both had bitten the first run; both now
  understood rather than hand-patched).**
  1. **`wiki-18.jsonl.gz` from HuggingFace is a TAR.GZ, not a plain gzip.** It
     holds one member (`data00/.../wiki_dump.jsonl`), so `gunzip -c` yields the
     tar stream: a 512-byte header fused onto record 0 and NUL padding at the
     tail. The retrieval server's `ntotal == lines` assertion caught it again
     (21,015,324 index rows vs 21,015,325 corpus lines). This is *exactly* the
     "botched tar extraction" the first run diagnosed on 2026-07-22 and repaired
     by hand — it was never a bad extraction on our side, it is what the file
     is. Fix: `tar -xzOf`, recorded in `/mnt/src/.../extract.sh`. Note the
     earlier `wc -l` "verification" was misleading: it counts newlines, so the
     unterminated final padding line made a corrupt corpus look correct.
  2. **The box cannot run cu130 builds.** Driver 570.172.08 reports CUDA 12.8;
     `torch 2.11.0+cu130` (pip's default variant) aborts with "NVIDIA driver on
     your system is too old (found version 12080)" and vLLM's engine core dies
     with it. cu128 wheels are forward-compatible across 12.x, so the stack must
     be pinned to cu128 via `--index-url .../whl/cu128`. **This vindicates the
     ORIGINAL `requirements-gpu-pinned.txt` (torch 2.10.0+cu128) that I had
     overwritten** on the strength of this log's "vllm 0.25.1/torch 2.11.0" line
     — the log recorded versions without the CUDA variant, which is the part
     that actually matters here. Lesson logged: verify the environment
     empirically; neither the pin file nor the log was trustworthy alone.
     **Swapping torch alone is not enough:** vllm 0.25.1's PyPI wheel is itself
     compiled for CUDA 13 (`import vllm` -> `ImportError: libcudart.so.13`), and
     no `+cu128` release wheel exists for 0.25.x (the documented GitHub release
     URLs 404 — matching vllm issue #37847). Since `vllm==0.17.1` requires
     exactly `torch==2.10.0`, the ORIGINAL pin set (vllm 0.17.1 / torch
     2.10.0+cu128 / transformers 4.57.6 / faiss-cpu 1.14.3 / fastapi 0.136.3 /
     uvicorn 0.51.0) is a coherent environment that empirically ran E-a and E-c
     on this box. **Restored it verbatim via `git checkout` and rebuilt the venv
     from it** (`.venv-gpu3`). My rewrite of that file was the error, not the
     file. `requirements-gpu.txt` independently notes "vllm>=0.17 ... GDN kernels
     for Qwen3.5", so 0.17.1 supports the executor.
     **The first run had already solved all of this** — commit `28b2728`
     ("executor serving SOLVED — full dependency saga resolved") records a chain
     of five failures: (1) torch 2.11 needs a CUDA>=12.9 driver, box has 12.8 ->
     pin vllm 0.17.1 + torch 2.10.0+cu128; (2) stale flashinfer-cubin 0.6.13 vs
     flashinfer 0.6.4 -> align them; (3) Qwen3.5's GDN prefill kernel JIT needs
     nvcc and there is no /usr/local/cuda on the box; (4) the pip nvcc wheels are
     both useless (the 12.9 wheel ships nvcc 13.2 = header clash, the 12.8 wheel
     ships no nvcc at all); (5) resolution: patch the venv's `qwen3_next.py` to
     take vLLM's own Triton FLA path (`forward_native`, the standard non-Hopper
     implementation, which self-compiles). I burned two turns rediscovering (1)
     because I trusted the prose log over the lockfile and never read the commit
     history for the file I was rewriting. **Read `git log -- <file>` before
     declaring a committed artifact stale.** The venv is gitignored so the
     `qwen3_next.py` patch itself was lost with it and must be re-applied; the
     `.cuda_home/` shim in the repo is a survivor of failure (4).
- 2026-07-28 **AUTONOMY MANDATE v2 (Brian):** "I will not prompt you. You must
  complete everything of the foundation paper plan until the end... You must
  advance and continue automatically and decide the best solution. Only ask me
  if you need clarification/questions." Self-scheduled loop keeps the run moving;
  every judgement call is logged here.
- 2026-07-28 **DATA REBUILT + VERIFIED.** `/mnt/src/liangsheng/cassi_foundation`
  (persistent) holds corpus + index + weights; `research/data_shared` symlinks to
  it. Assembly verified byte-exact against the values recorded in
  `serve_retrieval.py`: `e5_Flat.index` = 64,559,075,373 bytes and
  `wiki-18.jsonl` = 21,015,324 lines, so the index/corpus row-count assertion
  holds and the tar-corruption that bit the first run did NOT recur.
  Splits: **dev-200 EXACT MATCH by id to the pre-registered set** (verified
  against baseline_rows.csv — the gate keeps its questions and the surviving
  A0/A1/A2 numbers stay comparable); train-300 re-derived
  (189 medium/60 easy/51 hard vs the lost run's 194/55/51 — logged deviation);
  val-50 generated for the first time. Overlaps train/dev, val/dev, val/train
  all zero; decontamination dropped 0 rows (HotpotQA train and dev are disjoint
  splits, so the lost `*.decontaminated.jsonl` was belt-and-braces).
- 2026-07-28 **E-b RE-RUN: GATE NOT PASSED with the new judge.** Qwen3.6-27B
  scores mean .769 (rubric_v1) / .764 (v2) / .792 (v3) against the surviving
  50-label sheet — below .80 on both the strict per-bit reading the plan
  specifies and the looser mean+floor the code had drifted to. v2 applied the
  audit's rubric recommendations and was MEASURED, not assumed: `nothing_left`
  +.116, `supported` −.115, so v3 keeps what helped and reverts what hurt, and
  drops a verbatim-quote field that emitted invalid JSON. v3 runs with 0 parse
  failures and 0 neutral fallbacks. **Blocking finding:** at n=24–26 per bit the
  95% CI is ±0.17, so the v1/v2/v3 spread is noise — further prompt-tuning
  against this sheet fits noise. Report: `calibration_report_qwen36.md`.
- 2026-07-28 **DECISION (Brian): rebuild the calibration instrument, labeled by
  a FRESH SUBAGENT with no session context.** ~150 steps drawn from the Phase 3
  pilot; each batch goes to a newly spawned agent that sees only the bit
  definitions and the step context — never any judge output, never the rubric
  version history, never this conversation. Rationale: it removes the anchoring
  that inflated the original .848 (labels there were revised while reading the
  old judge's reasoning), it is independent of the judge under test
  (Qwen3.6-27B, a different model family), and n=150 shrinks the per-bit CI from
  ±0.17 to about ±0.07 — enough to actually arbitrate the gate. Honesty note for
  the report: these are model labels, not Brian's, so the gate is evidence about
  judge–labeler *consistency*, and the F3 requirement for human labels is
  formally unmet.
- 2026-07-28 **CONTAINER WIPE — all gitignored runtime state lost.** The pod came
  back fresh (every file dated 14:39); nothing was running and all 8 GPUs were
  idle, despite the E-e line above claiming RUNNING. `/home/liangsheng` is an
  ephemeral overlay; only committed files survived.
  - **Lost:** `research/data_shared/` (79G — HotpotQA sources incl. the
    *decontaminated* train file, 21M-passage corpus, E5/FAISS index) ·
    `foundation/data/` (frozen train-300/dev-200 + SHA256 manifest) · `.venv`
    and the GPU venv (`.venv-gpu` is now an empty shell: pip only) · every
    trajectory JSONL (pilot, baselines, E-e rollouts) · the judge cache · the
    **round-1b checkpoint**.
  - **Survived (committed):** all code and docs · all three reports ·
    `baseline_rows.csv` (all 1,400 A0/A1/A2 rows) · `sheet_v1.csv` (the 50
    calibration labels *with* each row's exact judge context).
  - **Consequences:** (1) training restarts from base — E-d/E-e survive only as
    code and commit messages; (2) **dev-200 is exactly recoverable** from the 200
    HotpotQA ids in `baseline_rows.csv`, so the surviving baseline numbers stay
    valid and comparable, and the gate is unaffected; (3) **train-300 is NOT
    reproducible** — its decontaminated source is gone and the generator lives in
    the read-banned archive, so it will be re-derived (same seed, same
    stratification, own overlap check vs dev-200) and A3 will train on a slightly
    different 300 questions than the lost run. Logged as a deviation.
  - **Policy fixes adopted (Brian, 2026-07-28):** everything regenerable now
    lives on `/mnt/src/liangsheng/` (persistent CPFS) and is symlinked into the
    repo, so a future wipe costs one symlink; commit and push continuously rather
    than at stage boundaries; GPU holds via `/home/liangsheng/brian/acquire_gpus.py`
    (never takes a card another process is on), 2 held at all times.
  - Dev-look ledger unchanged: **1 of ≤3 used** (E-c baselines).
- 2026-07-28 **Judge changed: gemma-4-31B-it (:6102) → Qwen3.6-27B (:6101)**
  (Brian). This invalidates the E-b gate, which was measured against gemma — so
  calibration re-runs against the new judge before any RL. Cheap: `sheet_v1.csv`
  carries the labels and the exact context, so `agreement()` re-judges from the
  sheet (50 calls, no GPU, no pilot rerun). **Verified integration hazard:**
  Qwen3.6-27B emits chain-of-thought into `message.content` (not
  `reasoning_content`) — a bare "output only JSON" request returned "Here's a
  thinking process: …" and hit the token cap. Strict-JSON parsing would fail on
  every step and fall back to neutral 0.5, silently zeroing the per-step reward
  while training continued on the terminal reward alone. Fix:
  `chat_template_kwargs: {"enable_thinking": false}`, which returns clean JSON.
  Also note `max_model_len` 32768 (gemma had 262144) — late-step prompt lengths
  must be checked against it.
- 2026-07-23 REFINEMENT ITERATION 1 (ladder tier: RL hyperparams): round-1
  training (lr 5e-6, 250 updates, full-FT) damaged the SAMPLING distribution —
  temp-0 val-50 healthy (F1 .607) but temp-1.0 rollouts 71.6% malformed, U
  -0.29; round-2 was training on that garbage -> killed. Changes: lr 2e-6,
  kl_beta 0.1, 150-update cap, KL log-ratio clamp, + NEW post-round temp-1.0
  health probe gate (scripts/probe_policy_health.sh: malformed<10%, cap<15%).
  Round-1b retraining from base on the original healthy rollouts (cached).
  Judge client hardened (transport retries + neutral fallback) after a
  connection-reset crash under 12-thread load.
- 2026-07-22: plan docs reviewed doc-by-doc with Brian; implementation started.
- 2026-07-22 (E-phase): judge = gemma-4-31B-it at 122.11.227.227:6102 (docs said
  Qwen 27B/35B — server reality wins); live rubric sanity PASSED (good search
  1,1,1 vs redundant rephrase 0,0,0). Standing 2-GPU-hold rule active (watcher
  polling; all 8 cards busy). **Corpus corruption found & repaired:** rescued
  wiki-18.jsonl was a botched tar extraction (512-byte header fused to record 0,
  padding tail) — caught by the retrieval server's ntotal==lines assertion;
  streaming repair validated all 21,015,324 records (ids 0…21015323), atomic
  swap, offsets rebuilt. GPU stack pinned (vllm 0.25.1/torch 2.11.0).
