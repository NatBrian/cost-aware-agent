# Foundation implementation progress

> ## 2026-08-02 — FOUNDATION-2 Step 1 COMPLETE: **H1 PASS, H2 SUPPORTED**
>
> Decided by `scripts/s3_analyse.py`, committed before the data existed and run
> unmodified. Frozen eval-600, temp 0, harness off, paired, 10k bootstrap.
>
> | | control (λ=0) | treatment (λ=0.568) | Δ @ B=2 | 95% CI |
> |---|---|---|---|---|
> | steps | 2.977 | **2.810** | **−0.167** | [−0.280, −0.057] |
> | F1 | 0.433 | **0.513** | **+0.080** | [+0.053, +0.108] |
>
> All three gate conditions PASS (threshold −0.119, guard −0.02). **F1 rose** —
> a Pareto improvement, not a trade.
>
> **H2 SUPPORTED — abandonment, not haste:** doomed work −0.486 [−0.832, −0.151];
> successful work −0.031 [−0.093, +0.038]. 16× larger on work that was going
> nowhere, and only that partition excludes zero.
>
> **Robustness:** the control failed its round-3 health gate (20.5% malformed) and
> stopped at r2 while the treatment passed all three, so the treatment was
> re-evaluated at r2. Round-matched −0.162 vs protocol −0.167 — same sign and
> magnitude, so the effect is not an artefact of extra training.
>
> **Pre-registered prediction that FAILED:** |Δsteps| was predicted largest at B=2
> (the only binding budget). It is largest at **B=3** (−0.303), and the effect is
> significant at all three budgets without tracking the binding fraction. Recorded
> rather than dropped; it undercuts the "binding budget is the determinant" story
> and needs investigation before being repeated.
>
> **W stays null** (−0.018, CI contains zero) exactly as S2 predicted at n=600 vs
> the ~2289 needed. Switching the estimand at S2, *before* the data existed, is
> what made this answerable.
>
> Report: `experiments/reports/s5_verdict.md`. Next: Step 3 scale-up (§10 of the
> v2.2 plan). Step 2 (Snell continuation value) is **not triggered** — its trigger
> was an H1 failure.

> **2026-07-31 — FOUNDATION-1 IS COMPLETE; the active plan is now
> `research/paper_plan_v2_2_foundation.md` (FOUNDATION-2, the redesign).**
> Everything below this banner is the FOUNDATION-1 record (stages I0–I7, E-a…E-g)
> and stays as history. FOUNDATION-2 stages are G0–G7 and start a fresh log.
>
> **Why the redesign** (full diagnosis in the v2.2 plan §2–§4): FOUNDATION-1
> passed its gate but the λ ablation showed it passed for the wrong reason —
> better answers, not cheaper ones (λ 0→1.0 moved stopping 0.04 steps). Three
> design errors, none a property of the method: (1) we targeted *stop-when-done*,
> which has 0.31 steps of headroom against a 0.5-step threshold, while
> *quit-when-hopeless* has ~0.94 and accounts for 52.7% of all steps spent;
> (2) a scalar per-step price is state-independent and cannot express a
> state-dependent stopping rule, so no λ could work; (3) the objective was scaled
> so even a perfect oracle quit rule was worth +0.012 utility, and 2 of 3 budgets
> could not bind.
>
> **New empirical basis, measured 2026-07-31 on existing rollouts (no new
> compute):** 43% of episodes end with F1=0 after 5.16 steps = 52.7% of all steps
> buy zero quality; P(eventual success | no progress by step k) decays
> .574→.511→.315→.241→.213; an oracle quit rule saves 0.94 steps. Source:
> `pilot.jsonl`, n=195, arm A1 untrained, `forced_continuation`. **Must be
> replicated on the 2400 λ=0 rollouts before anything is pre-registered (G1).**
>
> **Plan rewritten 2026-07-31 as a STAIRCASE, not a big bang.** The first v2.2
> draft proposed ~12 simultaneous changes — over-scoped, and it reproduced at
> larger scale the attribution error that already cost us a claim. Step 1 now
> changes **four things** (budgets, λ, headline metric, threshold-derivation rule)
> and holds the dataset, reward form, judge, trainer and executor **byte-identical
> to what already ran**. Steps 2 and 3 are conditional, each with a named trigger.
>
> Key re-reading that makes minimal viable: the λ ablation does **not** condemn
> the scalar price. It worked at the one budget that bound (B=2: −0.70 steps, CI
> excludes zero, quality intact) and did nothing at the two that did not. Measured
> against the policy's own stop distribution: B=2 binds for 75% of episodes, B=3
> for 41%, B=4 for 33%, **B=8 for 6%**. We gated at B=4.
>
> **Next actions, in order — both on data already on disk:**
> - **S0** (~½ day, eval only): re-score existing λ={0,0.3,1.0} checkpoints at
>   binding budgets with the wasted-spend metric. Is there already a signal?
> - **S1** (~1 day, CPU only): gold-free predictability check. Gate: held-out
>   **AUC ≥ 0.65**. **This gate can end the redesign in a day** — if hopelessness
>   is visible only in hindsight, the dataset change is promoted from Step 3 to
>   mandatory.
>
> Nothing downstream starts until both clear.

> ## FOUNDATION-2 Step 1 — live status (2026-07-31)
>
> - **S1 gold-free predictability — PASS.** Held-out AUC **0.813** at k=3
>   (gate 0.65), replicated on three arms (0.813 / 0.815 / 0.798), base rate .75.
>   Strongest feature is `logprob_last` — the model's own confidence. The agent
>   already knows when it is lost; it is not trained to act on it. Split is by
>   `task_id`, never episode. Report: `s1_predictability.md`.
> - **Code fixes — DONE**, 70 tests green. Retriever scores were being discarded
>   in *two* places (server dropped the FAISS distances, client kept only
>   title/text); token counts now recorded (only `raw_len` chars existed before);
>   `W` estimand + tests added; schema v2 keeps FOUNDATION-1 rollouts loadable.
> - **S2 headroom + calibration — DONE, and it changed the design:**
>   - **Only B=2 binds** (64.8%) against the *trained* policy; B=3 is 25.5% and
>     B=4 is 17.2%. My earlier "B=3 binds 41%" came from the **pilot's untrained**
>     policy. Gate moved to **B=2**.
>   - **`W` is not runnable** — 72% exact zeros, paired SD 1.64, needs n≈2289 at
>     B=2 and n≈108k at B=4. Primary estimand switched to **paired Δsteps**
>     (n≈479 at B=2). W stays as the economic reading, explicitly underpowered.
>   - **The original ablation was ~15× underpowered**: it ran at B=4 with n=50
>     where n≈751 is needed. "NOT EFFECTIVE against the pre-registered 0.5-step
>     rule" stands; "the effect is 0.04, essentially zero" does **not** — that CI
>     could not resolve the 0.164 effect either way.
>   - **λ\* = 0.568** (cap 0.6), **threshold 0.119 steps**, **eval-600 built**
>     (disjoint from train-300/dev-200/val-50 by id and normalized question).
> - **S3 pre-registration — COMMITTED** before any S4 data existed, together with
>   the analysis script that will be run unmodified (`s3_analyse.py`).
> - **S0 / S4 / S5 — RUNNING.** S0 re-scores the FOUNDATION-1 λ arms at the new
>   budgets; a chained job starts S4 (both arms, ~14h) then S5 (eval-600, ~6h) as
>   soon as S0 releases the GPU.
>
> **Dev-look ledger:** FOUNDATION-1's dev-200 **not touched** by any of the
> above — S0 uses val-50, S1/S2 use training rollouts, S5 uses the new eval-600.
>
> ### 2026-07-31 — 8 HOURS LOST: the watchdog blocked the job it was watching
>
> **S4 did not start for ~8 hours.** GPUs were held and idle; nothing ran.
>
> **Cause — the fifth process-matching bug in this project, and I wrote it after
> documenting the previous four.** The handoff chain waited for S0 to exit with
> `pgrep -f "[s]0_rescore.sh"`. The *watchdog* I armed to catch stalls contained
> the literal string `s0_rescore.sh` in its own command line (`for p in
> s0_rescore.sh chain_s4.sh …`). So the chain matched the watchdog and waited
> forever on a process that was never going to exit.
>
> **The watchdog also could not detect the stall it existed to catch**, for the
> same reason: it counted its own command line as a live pipeline process, so its
> "nothing is alive" branch was unreachable. It was self-satisfying.
>
> Compounding it: while clearing up, a `pkill -f "chain_s4.sh"` matched **its own
> shell** and killed it — the same class of bug a second time in ten minutes.
>
> **Fixes adopted:**
> 1. **Never wait on `pgrep` for a pipeline handoff.** Waiting on a *pattern* is
>    waiting on anything that happens to mention the pattern, including the
>    watcher. Wait on a **PID file** (`[ -d /proc/$(cat .s4.pid) ]`) or a
>    **marker file**, both of which are exact.
> 2. **A watcher must never be able to match itself.** If it must use patterns,
>    exclude its own PID explicitly.
> 3. **Kill by exact PID**, after reading `/proc/<pid>/cmdline` to confirm the
>    target — never `pkill -f`.
> 4. **Liveness is a positive signal, not the absence of an error.** The monitor
>    reported nothing for 8 hours and I read silence as progress. A watcher must
>    emit a periodic heartbeat, or "no news" is indistinguishable from "dead".
>
> S4 relaunched directly at 20:50 UTC (control arm λ=0.0 serving, config verified:
> budgets {2,3,4}, gate small, train_lambda 0.0). Watchdog is now PID-file based
> with no pattern matching anywhere.
>
> ### 2026-08-01 — the λ=0 CONTROL failed its round-3 health gate
>
> | round | malformed | hit_cap | F1 | steps | gate |
> |---|---|---|---|---|---|
> | 1 | 3.6% | 2.5% | .500 | 3.45 | PASS |
> | 2 | 6.7% | 2.5% | .456 | 3.35 | PASS |
> | **3** | **20.5%** | 10.0% | .502 | 3.65 | **FAIL** |
>
> **This is the control — λ=0, no cost pressure at all.** The damage is from
> training itself, not from pricing. FOUNDATION-1's λ=0 arm ran 2.2 / 3.5 / 5.8%
> and passed; the only design difference is the budgets ({2,3,4} vs {2,4,8}), so
> **training under tighter budgets appears to degrade the policy faster.** That is
> a finding in its own right and belongs in the report.
>
> Per pre-registration §8 the arm stops at its **last healthy checkpoint, round 2**.
>
> **Bug this exposed in my own runner (fixed before it could corrupt the result).**
> Checkpoints are written and backed up *before* the probe runs — deliberately,
> because a failed probe is evidence worth keeping. So `ctrl_round3/checkpoint`
> exists. My `last_round` selected the newest *existing* checkpoint, which would
> have made the **20.5%-malformed policy the control arm** and silently invalidated
> the whole comparison. Fixed: `run_lambda_arm.sh` now writes a `HEALTHY` marker
> only after a probe passes, and selection reads the marker, never the directory
> listing. Verified: selection returns ctrl round 2.
>
> **Second fix:** the round-mismatch guard assumed the *treatment* would breach
> first (λ=0.568 being the untested value). It went the other way. The guard now
> brings whichever arm sits at the higher round down to the lower one, in either
> direction, so the round-matched comparison always exists.
>
> ### 2026-08-01 — DATA CORRUPTION: two collection trees wrote the same shards
>
> **Killing the parent script did not kill its children.** `run_lambda_arm.sh`
> was reparented to init (PPID 1) and kept collecting. On relaunch, **two complete
> collection trees ran concurrently against the same
> `trt_round1/rollouts_shard*.jsonl`**: 3800 lines containing only **1915 unique**
> (task_id, rollout, budget) triples — nearly every episode written twice.
>
> **Nothing downstream would have caught it.** GRPO groups on
> (task_id, budget_B), so duplicates silently inflate group sizes, double-count
> trajectories in the advantage baseline, and bias the update toward whichever
> episodes happened to be written twice. The run would have completed and produced
> a number. Caught only because the progress heartbeat showed `collected=3771`
> against a hard maximum of 2400 — a sanity bound I had added for a different
> reason.
>
> Also orphaned: the **vLLM EngineCore survived its API server being killed**,
> holding 130 GB on GPU 1 and blocking the relaunch. It has to be killed
> separately, by PID, after confirming ownership.
>
> **Damage:** contained to `trt_round1`, which was deleted and recollected.
> **`scripts/check_rollout_integrity.py --all` confirms every other round is
> clean** — ctrl 1–3 and the FOUNDATION-1 lam0/lam10 arms all have exactly 2400
> lines, 2400 unique keys, uniform group sizes of 8×300. That retroactively
> validates the S1/S2 analyses built on those rollouts.
>
> **Fixes adopted:**
> 1. **Launch long runs with `setsid`** so the run owns a process group, and kill
>    with `kill -- -PGID`. Killing a bare parent PID orphans the tree.
> 2. **`scripts/check_rollout_integrity.py`** — duplicates, malformed JSON, group
>    sizes, expected count. Run it before any round's data is trusted.
> 3. **Identify processes by exact `/proc/<pid>/cmdline` equality**, never
>    `pgrep -f`. `pgrep -f "s4_s5_run.sh"` matched my own shell for the third time
>    in this session and wrote the wrong PID into `.s4.pid`.
> 4. **After killing a serving process, check the GPU** — `nvidia-smi
>    --query-compute-apps` — because the engine core outlives the API server.
>
> **Dev-look ledger:** FOUNDATION-1's dev-200 has 1 of ≤3 looks remaining.
> FOUNDATION-2 starts a fresh ledger on its own dev set.

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

- 2026-07-31 **PRE-REDESIGN DIAGNOSTICS — and they found that the ablation's
  threshold exceeded the physically available headroom.**
  Run on existing rollouts + the committed dev CSV. No new compute, dev-200 NOT
  re-read.
  **D1 — step slack at equal quality (the decisive one).** For each question,
  among the attempts reaching the group's BEST F1, cheapest vs average steps:
  **mean 0.313, MEDIAN 0.000**; 41% of groups have any slack, only 14% have ≥1
  step. **More than half of all questions have literally none.**
  **=> our pre-registered threshold was 0.5 steps against a ceiling of 0.313.
  The ablation was unpassable by construction.** No λ, reward design or model
  could have cleared it. This is a flaw in the experiment, not only the method,
  and it must go in the write-up. It also softens the ablation's reading: the λ
  term captured 13% of a tiny headroom — "not effective" stands, "incapable in
  principle" does NOT, because the ceiling was never above 0.31.
  **D2 — verbosity slack: at equal best quality the cheapest attempt uses 28%
  FEWER CHARACTERS.** Difficulty is held constant and the outcome is identical,
  so this is pure controllable waste — the strongest clean signal in the data,
  and it corroborates the earlier malformed-step finding (2000-char thoughts hit
  the 512-token cap).
  **D3 — redundant search: 6.8% of all searches** (2.5% exact + 4.3% near-dup).
  Real and controllable but small (~0.23 steps/episode).
  **D4 — the 9B DOES read the budget: telling it moves 1.02 steps** (A0 3.92 flat
  vs A1 2.96→3.98). That is 3x the total equal-quality slack and 25x the λ effect.
  **Capability is NOT the binding constraint here** — which contradicts my own
  earlier ranking of "bigger executor" as priority #2, based on the harness
  result (Sonnet −29%, weak model nil). Corrected on data.
  **Revised priorities for Brian's four proposals:** (1) dataset → long-horizon,
  now PROVEN necessary; (2) token cost kept as its OWN dimension — 28% headroom,
  difficulty-free — do NOT collapse tokens/tool-calls/steps into one dollar
  figure or the clean signal hides under two confounded ones; (3) rubric →
  controllable behaviours (redundancy, verbosity) rather than "was stopping
  right", which was our weakest calibration bit (.775) and least actionable;
  (4) bigger executor DEFERRED until tasks are long-horizon.
  **Process rule adopted: run D1 FIRST on any new dataset and set the detection
  threshold below the measured ceiling.** Ours was picked as "~14% of baseline"
  and exceeded what was achievable.
  Report: `experiments/reports/pre_redesign_diagnostics.md`.

- 2026-07-31 **BACKUP GAP FOUND AND CLOSED — the λ=1.0 final checkpoint was never
  on /mnt/src.** Audit of "is everything backed up?" turned up 8 of 9 checkpoints:
  `lam10_round3` was missing — **the very artefact the pre-registered verdict was
  computed from.**
  Cause: `run_lambda_arm.sh` backed up AFTER the health probe, and λ=1.0's round-3
  probe FAILED (malformed 11.0%), so the script exited before reaching the backup.
  The one round whose result was most unusual was the one round left unprotected.
  It survived only because the ephemeral overlay had not been wiped.
  **Fixed in the script: backup now runs BEFORE the gate**, with the reasoning
  in-file — a failed probe is a RESULT, not a discard, and is exactly when you
  most want the evidence kept. The failure message now states that artefacts are
  backed up.
  Also mirrored all 14 reports (+ PROGRESS.md, README.md, FOUNDATION_EXPLAINED.md)
  to `/mnt/src/.../reports/`. They were already safe on GitHub; the mirror means a
  reader with only the storage mount gets the findings alongside the artefacts.
  **Verified final backup state: 9/9 checkpoints (19G each, weights + config
  present), 32 trajectory files, 12 eval outputs, 14 reports — 372G total.**
  Lesson, third of its kind this run: a safety mechanism placed AFTER the thing it
  protects is not a safety mechanism. (Cf. the neutral-judge fallback that would
  have mislabelled an arm, and the cache that could fail a run.)

- 2026-07-31 **RUN CLOSED — both experiments complete, autonomous loop stopped.**
  Verified: verdict recorded, `ablation_report.md` (211 lines) +
  `ablation_preregistration.md` + `FOUNDATION_EXPLAINED.md` §14 all written;
  0 unpushed commits, 0 dirty files; tags `foundation-run-1` and
  `lambda-ablation-1`; 12/12 eval files backed up to /mnt/src (354G total).
  **Idle GPU memory released, hold retained.** The evaluation left the executor
  (~124G) and judge (~90G) allocated and idle. `acquire_gpus.py` (2 cards, 552M
  each) is the agreed reservation and stays up per Brian's standing rule, but
  sitting on 217G of idle allocations on a box where other users are queueing is
  a different thing. Freed both; kept the retrieval server warm (~2G, and a cold
  start re-reads the 64G index for ~10 min). GPUs 0+1 remain held and are
  instantly usable for follow-ups.
  **Loop stopped deliberately.** With no work outstanding, hourly wake-ups would
  burn tokens and produce nothing. Restarting is one command — the servers'
  start-up lines are all in this file and in the report.
  **State for whoever picks this up:** dev-200 has **1 of 3 looks unused**,
  reserved for a final headline method. The two follow-up designs in
  `ablation_report.md` §8 (price steps relative to the per-question minimum;
  or reward the stop decision against an oracle continuation) are cheap because
  the pipeline, the trajectories and the judge are all already built.

- 2026-07-31 **λ ABLATION COMPLETE — PRE-REGISTERED VERDICT: NOT EFFECTIVE.**
  At B=4: mean steps 3.500 (λ=0) vs 3.460 (λ=1.0), **paired Δ +0.040**, CI
  [−0.140, +0.300]. Threshold was 0.5 with disjoint CIs — **both conditions
  FAIL**. Power: Δ CI half-width 0.220 steps, and the observed effect is far
  below even that.
  Utility on the fixed yardstick *falls* with λ at B=4 (.431 → .424 → .400) and
  B=8 (.569 → .545 → .541) — not a significant harm (CIs overlap), but certainly
  not a benefit. The sensitivity arm (λ=1.0 round 2, last healthy checkpoint) is
  worse on every metric, so the verdict does not depend on the checkpoint choice.
  **The one significant behavioural effect in the sweep is at B=2**: λ=1.0 cuts
  **−0.700 steps, CI [−1.26,−0.24] (excludes zero), F1 intact**. EXPLORATORY —
  the rule was specified at B=4 — and labelled as such everywhere. It locates the
  mechanism: with the harness off, the λ=0 policy takes 3.28 steps at a budget of
  2, i.e. it OVERSPENDS, so the cost term has something to pull against. At B=4
  (3.5 of 4) and B=8 (3.8 of 8) it is already inside budget and no λ has leverage.
  **The determinant is whether the budget binds, not the size of λ.**
  **λ=1.5 NOT RUN.** The pre-registered "real but weak" flag needed Δ>0.15; Δ is
  0.040. Also λ=1.0 already breached the health gate (11.0% malformed, only arm
  to do so, lowest F1 at B=4), and the B=2 result shows the right variable is
  budget-bindingness, not λ. ~10 GPU-hours avoided. Note: earlier in this run I
  pre-justified skipping λ=1.5 from a mechanism argument and a later round
  undercut it — this time the numeric pre-registered condition decides and the
  mechanism only corroborates.
  Reports: `experiments/reports/ablation_report.md` (full),
  `research/FOUNDATION_EXPLAINED.md` §14 (plain language). Tag
  `lambda-ablation-1`.

- 2026-07-30 23:05 **lam10 (λ=1.0) round 3: HEALTH PROBE FAILED — and that is a
  result, not just an incident.** malformed **11.0%** against the 10% gate
  (hit_cap 5.0%, within its 15% gate). The arm stopped itself: "refusing to train
  on a damaged policy". Round 3 was the final round, so nothing trained on the
  damage.
  Malformed across the arm: **5.8% -> 2.4% -> 11.0%**. Compare λ=0 (2.2 -> 3.5 ->
  5.8) and λ=0.3 (4.3 -> 2.0 -> 9.0). The λ=1.0 arm ends worst, and it is the arm
  whose steps were falling — consistent with an aggressive cost term buying step
  reductions partly by degrading output quality rather than by better stopping.
  **Evaluation design decision (logged):** evaluating λ=1.0 only at its round-2
  checkpoint would be asymmetric — every other arm is measured at round 3. So the
  harness now evaluates BOTH:
  * `lam10` = round 3 (protocol-matched) → this is what the PRE-REGISTERED RULE
    reads, unchanged;
  * `lam10_r2` = round 2, the arm's last HEALTHY checkpoint → reported as a
    labelled SENSITIVITY arm, excluded from the monotonicity tests (it is a second
    point at the same λ and would break them by construction).
  Reporting both is the honest option: picking whichever checkpoint favours a
  conclusion is exactly the freedom pre-registration exists to remove.
  Evaluation launched: val-50, budgets {2,4,8}, temp 0, harness off, 4 arms.

- 2026-07-30 19:50 **lam10 round 2 done — and it PARTLY CONTRADICTS my logged
  prediction. Recording that before the final data, as I did the prediction.**
  Round-2 probes (val-50, temp 1.0, n=40), same protocol across arms:
  | arm | r1 steps | r2 steps | direction |
  | λ=0.0 | 3.48 | 3.55 | up |
  | λ=0.3 | 3.48 | 3.67 | up |
  | λ=1.0 | 3.45 | **3.17** | **down** |
  λ=1.0 is the ONLY arm whose step count is falling, and the λ=0 − λ=1.0 gap is
  now **0.38 steps** — under the pre-registered 0.5 threshold, but no longer the
  ~0.03 that made me predict a flat null. Round-2 probe was healthy (malformed
  2.4%, hit_cap 0.0%, F1 .471).
  **What this does to the earlier reasoning:** the difficulty-confound analysis
  stands on its own (it is measured, not inferred) — stop step IS substantially
  question-determined, between/within SD ratio 1.83. But "therefore no λ can move
  mean steps" was my *extrapolation* from that, and the λ=1.0 trend is evidence
  against the strong form of it. A confound can suppress an effect without
  abolishing it. The honest position: difficulty explains why the effect is SMALL
  and hard to price, not that it must be exactly zero.
  **Consequences I am holding to:**
  * the pre-registered rule still decides the verdict, unchanged, on final
    checkpoints at temp 0 with paired bootstrap CIs — not on these probes;
  * **λ=1.5 is now back ON the table.** I logged a decision to skip it because
    "the mechanism explains why no λ will work". That justification is weakened by
    this trend. Decide after the final val-50 numbers: if the λ=0→1.0 slope is
    real but under-threshold, a 4th point at 1.5 becomes the difference between
    "no effect" and "a real but weak dose-response", which is worth 10h.
  Noting for my own discipline: I had a satisfying mechanism and used it to
  pre-justify skipping an experiment. The trend arrived one round later and
  undercut it. Explanations should not retire experiments that are already
  scheduled.

- 2026-07-30 18:50 **MECHANISM FOUND — and it is not the one I predicted. Three
  diagnostics on already-collected rollouts, no new compute.**
  I suspected GRPO's group-relative advantage was cancelling the cost term (any
  reward component roughly constant within a group vanishes in the mean
  subtraction). **That hypothesis is WRONG:**
  | arm | within-group SD of steps | cost term as % of the F1 term in the advantage |
  | λ=0.3 | 0.638 | 24.5% |
  | λ=1.0 | 0.703 | **81.7%** |
  Siblings differ by ~0.7 steps, so at λ=1.0 the cost term carries 82% of the
  advantage signal. The incentive is present and strong. "Signal too weak" is
  ELIMINATED as the explanation.
  **Is stopping earlier actually rewarded? Yes, enormously.** On the λ=1.0
  policy's OWN rollouts at B=4: U(stop@2)=+0.131, U(stop@3)=−0.211,
  U(stop@4)=−0.442. Utility-optimal stop is step 2; the policy stops at 3.29. So
  a +0.34 utility gain sits unclaimed after 150 updates x 3 rounds.
  **Why it cannot claim it — the actual mechanism.** F1 is HIGHEST among episodes
  that stopped at 2 (.631, vs .539 at step 3). Early stops are not quality
  sacrifices; they are the EASY QUESTIONS. Decomposing stop-step variance over
  109 groups of 8 rollouts on the same question:
  * WITHIN-question SD (what the policy varies) = **0.666**
  * BETWEEN-question SD (what the question dictates) = **1.220**
  * ratio 1.83
  **Stop step is substantially a CONSEQUENCE of task difficulty, not a free choice
  the policy makes.** Stopping earlier on a hard question means answering it
  wrong, and the model cannot know difficulty before searching. No value of λ can
  price away a variable the policy only partly controls.
  **This is the paper's real finding**, and it is a much better one than "λ was
  too small": *in a step-budget formulation, "when to stop" is confounded with
  "how hard the question turned out to be", so a per-step cost term cannot move
  mean steps however strongly it is priced.* It also explains the original
  result cleanly — RL improved F1 because F1 is learnable, and left steps alone
  because steps are mostly not the policy's to choose.
  Implication for v2.1: the cost signal needs to attach to something the policy
  actually controls, or the metric needs to condition on difficulty (e.g. steps
  relative to the minimum needed for THAT question, or a stop-decision reward
  evaluated against an oracle continuation). Logged for the report.

- 2026-07-30 17:45 **lam10 (λ=1.0) round 1 done, probe PASS — and the
  accumulating signal now points hard at H0.**
  Round-1 probes, identical protocol, val-50 @ temp 1.0, n=40:
  | train λ | step price at B=4 | probe steps | probe F1 | malformed |
  | 0.0 | free | 3.48 | .467 | 2.2% |
  | 0.3 | .075 | 3.48 | .612 | 4.3% |
  | 1.0 | **.25** | **3.45** | .574 | 5.8% |
  At λ=1.0 a step costs .25 utility while a 4th step buys ~.06 F1 — more than 4x
  unprofitable — and the policy takes the SAME number of steps as the arm with no
  step price at all. Across a 0 -> 1.0 sweep of the cost coefficient, mean steps
  moves by 0.03. Round 1 health fine (mean_kl .1173, 8028 samples kept, judge
  0 parse / 0 transport / 0 corrupt).
  **This is not the verdict and must not be used as one.** n=40, temp 1.0, round 1
  of 3; the pre-registered rule compares FINAL checkpoints on val-50 at temp 0
  with paired bootstrap CIs. But it is now three independent arms spanning the
  whole λ range landing within 0.03 steps of each other, so I am recording the
  prediction in advance of the data: **I expect the pre-registered rule to FAIL
  and the honest outcome to be a NEGATIVE result — per-step economic rewards at
  this scale do not change stopping behaviour.** Writing that down now so the
  eventual report cannot be accused of post-hoc framing either way.
  If that is the outcome, the useful contributions become: (a) prompting beats
  enforcement (A1 > A2 at every budget); (b) a reward term can be correctly
  calibrated to place the optimum and still exert no behavioural pull — a
  concrete warning about reward design; (c) a working, reproducible pipeline plus
  a reward-hacking protocol that made a falsifiable prediction and checked it.

- 2026-07-30 11:20 **ARM lam0 (train_lambda=0.0) COMPLETE — 3 rounds, every
  health probe PASSED.**
  | round | mean_kl | probe malformed | hit_cap | probe F1 | probe steps |
  | 1 | .3837 | 2.2% | 0.0% | .467 | 3.48 |
  | 2 | .0032 | 3.5% | 0.0% | .540 | 3.55 |
  | 3 | — | 5.8% | 2.5% | .672 | 3.88 |
  Round 1's elevated KL (.3837 vs the λ=0.3 arm's .052) settled to .0032 by round
  2 — consistent with larger early advantages from a reward carrying no
  step-cost term, not with policy damage, and the probes confirm health
  throughout. Judge stayed clean: 0 parse failures, 0 transport failures, 0
  corrupt cache reads across the arm (the atomic-write fix holding).
  All three checkpoints + rollouts backed up to /mnt/src.
  **Still NOT the verdict.** The probe numbers are n=40 at temp 1.0; the
  pre-registered rule compares final checkpoints on val-50 at temp 0 against the
  λ=1.0 arm. Recording only that λ=0's probe steps (3.48/3.55/3.88) sit in the
  same band as λ=0.3's (3.48/3.67/3.62).
- 2026-07-30 11:22 **ARM lam10 (train_lambda=1.0) LAUNCHED.** Steps priced ~3x
  the original: at B=4 a step costs .25 utility against the ~.06 F1 a 4th step
  buys, so continuing should be clearly unprofitable — the regime the λ=0.3 run
  never tested. Evaluation λ stays pinned at 0.3 so all three arms are scored on
  one yardstick. ~10h expected.

- 2026-07-30 04:25 **A CORRUPT JUDGE-CACHE ENTRY killed lam0 round 2 after all
  2400 episodes were collected AND judged.** `json.decoder.JSONDecodeError: Extra
  data: line 1 column 388`. Root cause: `judge_client` wrote cache entries with a
  plain `write_text`, and with 12 judging threads two of them can hit the SAME
  cache key concurrently — the interleaved writes left two JSON objects
  concatenated in one file. One bad file out of 23,836 took down a round that had
  already cost ~45 min of collection and ~30 min of judging.
  **Two fixes, both tested (65 tests green):**
  1. **Atomic writes** — temp file + `os.replace` (atomic on POSIX), so
     concurrent writers can never interleave.
  2. **Corrupt entries degrade to a cache MISS**, never an exception: the entry is
     deleted, counted in a new `cache_corrupt` stat, and re-queried. A cache is an
     optimisation; it must never be able to fail a run.
  Scanned all 23,836 entries: exactly 1 corrupt, 0 stray temp files. Removed it.
  Regression tests cover both the corrupt-read path and the no-.tmp-left-behind
  property.
  Reflection: this is the third failure of the same shape — a mechanism that
  should only ever make things *faster* (cache) or *safer* (neutral fallback,
  process kill) was able to make the run *fail* or *lie*. Worth auditing the
  remaining helpers for that property rather than waiting for the next one.

- 2026-07-30 03:50 **I corrupted a RUNNING shell script by editing it in place.**
  `e5_round.sh: line 37: hung: command not found` killed the arm right after
  round-1 training finished. The file was fine (`bash -n` clean) — bash reads
  scripts LAZILY BY BYTE OFFSET, so inserting three comment lines while it was
  executing shifted every subsequent offset and dropped the running shell into
  the middle of a comment. **Never edit a shell script that is currently
  running**; edit a copy, or wait for it to exit.
  Cost was small: round-1 training had already completed and saved
  (150 updates, 7994 samples, mean_kl .3837, final_loss .6958). The restartable
  runner resumed cleanly — "RESUMING: rounds 1..1 already trained" — and is
  serving the round-1 checkpoint for round-2 collection.
  **Gap this exposed and I closed by hand:** the arm died INSIDE e5_round.sh, i.e.
  before the health-probe gate, and on resume the script jumps straight to round
  2 — so round 1's policy would never have been health-checked. Ran the probe
  manually against the served round-1 checkpoint. This matters more than usual
  here: **λ=0's round-1 mean_kl is 0.3837 against 0.052 for the λ=0.3 arm** (~7x),
  which is what a reward with no step-cost term and therefore larger, less
  constrained advantages would look like. Higher KL is not automatically bad, but
  it is exactly the regime where the first run's policy damage happened, so the
  probe is not optional.
  TODO for the runner (do NOT edit while running): on resume, re-run the health
  probe for the last completed round before starting the next.
  **Probe result (run by hand): PASS** — malformed 2.2%, hit_cap 0.0%, F1 .467,
  steps 3.48. So the 7x KL is larger advantages from an unconstrained reward, not
  policy damage. Early and NOT to be over-read: λ=0 round 1 gives mean steps 3.48
  at temp 1.0, identical to the λ=0.3 arm's round-1 probe (3.48) — directionally
  consistent with H0 (cost term inert), but n=40, temp 1.0, round 1 of 3. The
  verdict comes from final checkpoints on val-50 at temp 0 under the
  pre-registered rule.

- 2026-07-30 00:35 **OOM: training landed on the judge's GPU.** Round-1 training
  died with `torch.OutOfMemoryError` — `e5_round.sh` passed the whole 2-GPU hold
  (`0,1`) as CUDA_VISIBLE_DEVICES, so torch took `cuda:0`, which now holds the
  local judge (~90G) plus retrieval. Harmless when GPU 0 carried only retrieval;
  fatal once the judge moved there.
  Fixed: training AND checkpoint-serving in `e5_round.sh` are pinned to the
  SECOND held card (`cut -d, -f2`). GPU 0 = judge + retrieval, GPU 1 = executor
  and training — and the executor is stopped before training, so GPU 1 is free
  when the trainer needs it.
  **Hardened at the same time:** `e5_round.sh`'s post-training readiness check
  used `/v1/models`, which vLLM's API server answers even when the engine core is
  dead — the same false-readiness trap that once reported a 500-ing executor as
  healthy. It now requires a real completion.
  Pattern worth naming: moving the judge on-box changed a *global* resource
  assumption, and three separate places still encoded the old one (process
  matching, GPU selection, readiness). A change of that shape needs a sweep of
  everything that touched the old assumption, not a fix at the point of failure.

- 2026-07-30 00:20 **The arm killed our own judge — third instance of the same
  process-matching bug.** Once the judge moved local, we run TWO of our own vLLM
  servers (judge :6101 on GPU 0, executor :8378 on GPU 1). The arm script's
  "stop the executor before serving a new checkpoint" step used
  `pgrep -u $(whoami) -f "[v]llm serve" | head -1` and killed whichever matched
  first — the judge. GPU 0 dropped 57G -> 2.2G seconds after the judge came up.
  Fixed in `run_lambda_arm.sh`, `f6_eval.sh`, `select_checkpoint.sh`: the pattern
  is now `[v]llm serve.*port 8378`, which cannot match the judge.
  `e5_round.sh` was already port-scoped — the third time this run that the
  correct version already existed in the repo and I copied the wrong one.
  **Progression of this same defect:** unscoped across users (nearly killed
  yongyue's job) -> scoped to user but not to service (killed our own judge) ->
  scoped to the specific service. I had even written the two-servers hazard into
  my own wake-up notes an hour earlier and still did not fix the scripts. Writing
  a warning is not the same as removing the failure mode.
- 2026-07-30 00:15 **Local judge needed `--max-num-seqs 32`.** First start died at
  engine init: `assert num_cache_lines >= batch` in causal_conv1d.py via
  qwen3_next.py. Qwen3.6-27B is the same Qwen3-Next hybrid (mamba + attention)
  family as the executor, and its conv-state cache is per sequence slot — the
  default slot count outnumbered the cache lines available while sharing GPU 0
  with retrieval. Bounded concurrency instead of only raising memory (the judge
  client drives at most 12 concurrent requests). Also confirms the model needs the
  same `patch_vllm_qwen3next.py` Triton-path patch as the executor: that patch is
  why init reached the cache assertion rather than dying earlier in flashinfer.

- 2026-07-30 00:10 **SWITCHED TO A LOCAL JUDGE — external dependency removed.**
  The remote Qwen3.6-27B (122.11.227.227:6101) stayed down through the whole
  wait, so the pre-decided fallback fired: 52G of weights downloaded to
  /mnt/src, served on GPU 0 beside the retrieval server via
  `scripts/serve_judge_local.sh`, `judge.endpoint` repointed to
  `http://127.0.0.1:6101/v1`. Poller stopped; the arm relaunches against the
  local judge.
  **Why this is safe for the experiment:** the served model id stays
  `Qwen3.6-27B`, byte-identical to the remote, and the judge cache keys on model
  name — so all ~18,000 judgements already computed against the remote server
  remain valid and get reused rather than silently recomputed by a different
  judge. Same weights, same rubric_v4, same temp 0 + enable_thinking=false, so
  the λ=0 arm is judged by the same function the λ=0.3 arm was.
  GPU layout now: GPU 0 = retrieval (~2G) + judge (~55G) of 143G; GPU 1 = free
  for the executor and training. No contention.
  This also permanently removes the pipeline's last dependence on a machine we do
  not control — worth the 52G on its own, independent of this outage.

- 2026-07-30 00:25 **Judge still down ~40min. Preparing the local-judge fallback
  IN PARALLEL rather than waiting out the clock.** The auto-resume poller
  (`wait_for_judge_then_run.sh`) is alive and will relaunch the arm the instant
  the remote returns, so nothing is lost by also getting ready to replace it.
  Downloading `Qwen/Qwen3.6-27B` (~54G) to /mnt/src and added
  `scripts/serve_judge_local.sh`: serves it on GPU 0 alongside retrieval (27B bf16
  ~54G + retrieval ~2G on a 143G card; the executor keeps GPU 1), at
  gpu-memory-utilization 0.55 so the two coexist.
  **The load-bearing detail is `--served-model-name Qwen3.6-27B`** — identical to
  the remote id. The judge cache key includes the model name (audit 2026-07-28),
  so keeping the id byte-identical means all ~18,000 cached judgements stay valid
  and are reused; changing it would silently re-judge everything AND mix two
  judges' scores inside one experiment.
  Rationale for doing this now rather than at the 4h mark: ~6h of ablation work
  remains (2 arms × 3 rounds), and the judge is the pipeline's only external
  dependency — it has already failed once mid-round. A 54G download costs nothing
  but bandwidth and buys permanent independence. If the remote returns first, the
  poller uses it and the local copy is simply insurance.

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
