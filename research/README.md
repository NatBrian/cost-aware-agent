# research/ — the CASSI research project

This directory is a **separate project** from the cost-metering harness in the
repository root ([`../README.md`](../README.md)). The harness *measures* what an
agent spends and reports it back; this project asks the next question:

> Can an agent be **trained** so the economics live inside the policy — so it
> stops itself at the right time, instead of being told about a budget (prompt)
> or forced to stop (harness)?

Everything research-related belongs here: plans, literature, code, configs, tests,
experiment outputs, reports. The root `experiments/` and `tests/` directories are
the **harness's** — never put research runs or research tests there.

**What we are building right now:** the plan is
[`paper_plan_v2_1_foundation.md`](paper_plan_v2_1_foundation.md) and the code that
implements it is [`foundation/`](foundation/).

---

## Start here (read in this order)

1. [`paper_plan_v2_1_foundation.md`](paper_plan_v2_1_foundation.md) — **the active
   plan.** A deliberately small, end-to-end version of the full paper: step-count
   budgets, HotpotQA only, GRPO with prompted-judge rewards, three arms, one
   pre-registered GO/NO-GO gate. Reviewed doc-by-doc with the author 2026-07-22.
2. [`foundation_tasks/PROGRESS.md`](foundation_tasks/PROGRESS.md) — **the live
   tracker.** Which stage we are in, what is next, decisions locked, and the
   anti-overfitting policy. Always the current truth; this README is not.
3. [`foundation_tasks/F0…F7*.md`](foundation_tasks/) — the stage spec for whatever
   stage is current (one doc per stage, each with its own done-criteria gate).
4. [`foundation/README.md`](foundation/README.md) → the code.
5. [`paper_plan_v2_1.md`](paper_plan_v2_1.md) — only when full-paper context is
   needed. It remains the source of truth for the eventual ICLR paper; the
   foundation run de-risks it first. **Where the two conflict, the foundation doc
   governs foundation work.**

---

## Directory map

### The active work

| Path | What it is |
|---|---|
| [`paper_plan_v2_1_foundation.md`](paper_plan_v2_1_foundation.md) | The foundation plan: the simplifications vs. the full plan (each with its reason), the three arms, the reward design, the stage sequence, the pre-registered gate, and the metrics. |
| [`foundation/`](foundation/) | **The code.** Python package implementing the plan end-to-end — agent, harness, collection, rubric/judge, rewards, GRPO trainer, eval, figures. See [the code map](#the-foundation-codebase) below. |
| [`foundation_tasks/`](foundation_tasks/) | One spec doc per stage — `F0_repo_restructure`, `F1_data`, `F2_harness_trajectories`, `F3_rubric_reward_model`, `F4_baselines`, `F5_rl_training`, `F6_evaluation`, `F7_analysis_report` — plus `PROGRESS.md`, the living tracker (implementation stages I0–I7, then experiment stages E-a…E-g). |

### Paper plans (the full paper)

| Path | What it is |
|---|---|
| [`paper_plan_v2_1.md`](paper_plan_v2_1.md) | **The full ICLR 2027 paper plan** — 2 domains, ~10 baselines, trained reward model, dollar-denominated costs, full statistics. Additive revision of v2 (every change tagged `[v2.1]`, §20 changelog). |
| [`paper_plan_v2.md`](paper_plan_v2.md) | Superseded by v2.1 — kept for its history only. |
| [`paper_plan_v2_simple.md`](paper_plan_v2_simple.md) | Plain-language companion to v2: same ground, nothing removed, reasoning spelled out. The best single document for understanding the project end to end. |
| [`architecture_comparison.md`](architecture_comparison.md) | Prompted big reward model vs. trained small reward model — the design comparison that produced v2.1, in simple language with a citation or a code file behind every "why". Illustrated by `proposed_rl_training_architecture.png`. |

### Positioning (why this paper is publishable)

| Path | What it is |
|---|---|
| [`lit_review/`](lit_review/) | 10 research areas, ~94 papers read at PDF level: overthinking empirics, token-efficient reasoning RL, learned stopping, budget-aware agents, agent PRMs, monitor/executor metareasoning, optimal-stopping theory, agentic RL, adaptive compute routing, hindsight loops. `00_overview.md` is the master synthesis. |
| [`novelty_check/`](novelty_check/) | Five independent adversarial reviews of the proposal (skeptical area chair, RL expert, efficient-reasoning expert, agent-systems expert, prior-art hunter). |
| [`competitor_analysis.md`](competitor_analysis.md) | CASSI vs. the field (July 2026), sourced entirely from `lit_review/` + `novelty_check/`. |

### History and artifacts

| Path | What it is |
|---|---|
| [`progress_report_2026-07-22.md`](progress_report_2026-07-22.md) | Narrative report of everything up to the foundation decision: the architecture choice, the first implementation, the first experiments and results. |
| [`reports_run1/`](reports_run1/) | Plain-language reports + paid-measurement records rescued from the first CASSI run *before* it was archived (smoke pilot, collection, labels, dataset manifest, RM-P heldout, trivial baselines). Kept because re-buying those measurements costs money. |
| `requirements-cpu.txt` | Leftover top-level pin from run 1; the foundation's own requirements live in `foundation/`. |
| `archived/` | The previous CASSI codebase and superseded docs. **Read-banned** by [`../CLAUDE.md`](../CLAUDE.md): the foundation was built fresh on purpose so its correctness never depends on code we no longer trust. |
| `papers/` | The 101 reference PDFs behind `lit_review/`. Gitignored — present only where they were downloaded. |
| `data_shared/` | Rescued regenerable data artifacts (~79G: HotpotQA source splits, the 21M-passage Wikipedia corpus, the E5/FAISS index). Gitignored — not in a fresh checkout; `foundation/scripts/` regenerates what depends on it. |

---

## The foundation run in one page

**Task.** HotpotQA multi-hop QA. One tool: search over a local Wikipedia index.
A *step* = one ReAct iteration (thought → search → observation); emitting an
answer ends the episode. Hard cap `t_max = 10`.

**Cost.** Number of steps — no dollars, no price map. A *step budget* `B` is drawn
per episode from `{small: 2, medium: 4, large: 8}` (frozen from the F2 pilot);
within a GRPO group all G rollouts share the same `B`, so group advantages are
never confounded with budget luck.

**The economy.** `U = F1(answer) − λ·(steps_used / B)`, λ = 0.3 (frozen from the
pilot: it makes the observed knee at step 3 an interior optimum at `B = 4`).

**Four arms**, one shared scaffold so the scaffold is a constant, not an advantage:

| Arm | What it is | What it answers |
|---|---|---|
| **A0** | Plain ReAct, no budget information anywhere | The no-cost-signal floor |
| **A1** | Budget tracker injected each step, nothing enforced | Is *telling* it enough? |
| **A2** | Harness hard-stops at `B` and forces an answer from the draft | Is *forcing* it enough? |
| **A3** | Same executor, GRPO-trained on per-step judge rewards + terminal economy, evaluated **with the harness off** | Does training move the economics into the policy? |

**The reward.** Anything computable exactly is computed exactly, never judged:
`R_final = F1 − λ·(steps/B) + 0.1·format_ok` from gold. On top, a frozen prompted
judge scores every step against a fixed, versioned binary rubric (`rubric_v1`:
`new_info`, `not_redundant`, `was_needed`; answer bits `supported`,
`nothing_left`) → `r_t = 0.2·(score − 0.5)`. The judge never sees gold, and it
must clear a **calibration gate** (≥80% per-bit agreement with 50 hand labels)
before any RL run consumes its scores.

**The gate (pre-registered, in code as `eval/gate_check.py`).** GO iff on the
frozen dev-200 at the medium budget: A3-harness-off beats both A1 and A2 on
utility, self-terminates in ≥70% of episodes, and stays within 5 F1 points of A2
(savings must not come from answer-quality collapse).

---

## The foundation codebase

`foundation/` is a plain Python package (no install step; modules are run with
`python -m` from `foundation/`). Every constant comes from
**`configs/foundation.yaml` — the single source of truth**; no module may
hard-code an experiment constant.

The pipeline, in the order data flows:

```
 configs/foundation.yaml ──────── every constant, loaded via common.load_config()
          │
 scripts/f1_data.py ──────────▶ data/  frozen train-300 + dev-200 + SHA256 manifest
          │                            (collect/sampling.py: stratified, seeded)
          ▼
 collect/run_collection.py ───▶ episodes JSONL  ◀── agent/harness.py  (the episode loop)
   the trajectory script                        ◀── agent/prompts.py  (facts-not-advice tracker)
   resumable · G rollouts                       ◀── agent/llm_client.py → vLLM executor
   · seeded wallet draw                         ◀── envs/retrieval_client.py → search server
          │                        collect/schema.py = the contract every stage validates
          ├──────────────▶ reward/  judge scores each step (rubric_v1) → per-step rewards
          │                  rubric.py · judge_client.py (cached, neutral-on-parse-failure)
          │                  rewards.py (r_t, R_final, returns-to-go) · calibration.py (the gate)
          │                        │
          │                        ▼
          │                train/  grpo_trainer.py (lean round-synced on-policy GRPO)
          │                        advantages.py (group-normalized step returns-to-go)
          │                        reward_adapter.py (+ judge-vs-F1 divergence log)
          │                        grpo_runner.py --dry-run (CPU smoke gate)
          │                        → checkpoint → served → next round
          ▼
 eval/  build_rows.py → rows CSV → metrics.py (bootstrap CIs, paired deltas)
                                 → gate_check.py (GO/NO-GO, exit 0/1)
          ▼
 analysis/  figures.py (CSV in → PDF out) · report.py (numbers auto-filled)
            → experiments/reports/ + experiments/results/
```

| Package | Main modules | Responsibility |
|---|---|---|
| `configs/` | `foundation.yaml` | **Single source of truth** for every constant: budgets, λ, rubric bits and weights, GRPO hyperparameters, endpoints, gate thresholds. Changing an experiment means changing this file. |
| — | `common.py` | Config loading + run stamping (config snapshot, seed, git hash written next to every run's outputs). |
| `agent/` | `harness.py`, `prompts.py`, `llm_client.py` | The episode loop with three harness modes (`none` / `enforce` / `forced_continuation`), the ReAct prompt and its deterministic parser, the running `BEST ANSWER SO FAR` draft, and the budget tracker — *facts, not advice*, and arm A0 gets no budget content at all (verified by test). |
| `envs/` | `retrieval_client.py` | The agent's one tool: HTTP client for the local E5+FAISS retrieval server (`scripts/serve_retrieval.py`). |
| `collect/` | `run_collection.py`, `schema.py`, `sampling.py` | **The trajectory script** — resumable, seeded, G rollouts per task, per-step draft scoring — plus the JSONL schema every later stage validates against, and the stratified dataset sampler. |
| `reward/` | `rubric.py`, `judge_client.py`, `rewards.py`, `calibration.py` | The reward spec (`rubric_v1`, versioned — any edit re-runs calibration), the judge client (strict JSON, one reprompt, neutral 0.5 on failure, disk cache, call accounting), the reward math, and the calibration gate. |
| `train/` | `grpo_trainer.py`, `advantages.py`, `reward_adapter.py`, `grpo_runner.py` | Round-synced on-policy GRPO written in-house (verl was dropped at E-d — rationale in the module docstring and commits `e4fe6b6`/`18542f1`): step-level group-normalized advantages with a min-cohort guard, Dr. GRPO length hygiene, KL anchor, and the judge-score-vs-realized-F1 divergence log (the reward-hacking curve). |
| `eval/` | `metrics.py`, `qa_metrics.py`, `build_rows.py`, `gate_check.py` | One scorer path for every arm and stage (`qa_metrics.py` is the *only* answer scorer in the codebase), aggregates with bootstrap CIs and paired per-task deltas, and the pre-registered gate as executable code. |
| `analysis/` | `figures.py`, `report.py`, `diagnostic_rubric.py` | Figures regenerated from CSVs (no hand-edited numbers, CVD-safe entity-fixed palette), the plain-language report generator, and an offline six-dimension trajectory diagnostic (analysis only — never a reward, never touches the gate). |
| `scripts/` | `f1_data.py`, `serve_retrieval.py`, `serve_executor.sh`, `e5_round.sh`, `probe_policy_health.sh`, `f4_baselines.sh`, `gpu_watch.sh` | Everything with side effects: dataset regeneration, the two servers, one training round end-to-end, the post-round temp-1.0 policy-health probe, the baseline sweep, the GPU hold watcher. |
| `tests/` | `test_f1…f6_*.py` | CPU-only test suite (~52 tests) covering every stage — must stay green. |
| `experiments/` | `reports/`, `results/` | This project's experiment outputs: `reports/` = plain-language memos (`pilot_memo.md`, `calibration_report.md`, `baselines_report.md`), `results/` = the CSVs figures and gates are computed from. Trajectories, checkpoints and training dirs are gitignored; reports and CSVs are committed. |

Also in `foundation/`: `Makefile` (`venv`, `test`, `dry-run`, `data`),
`requirements-cpu.txt` / `requirements-gpu*.txt` (the CPU test path is separate
from the pinned GPU stack), and `EXPERIMENT_CHECKLIST.md` — the ordered E-a…E-g
runbook with the gate between each stage.

### Running it

```bash
cd research/foundation
make venv          # python3.12 venv + CPU requirements
make test          # the full CPU suite — must be green before anything else
make data          # regenerate the frozen train-300 / dev-200 + manifest
make dry-run       # CPU smoke: fabricated rollouts through the FULL reward path
```

GPU work (collection, judging, training) additionally needs the two servers up
(`scripts/serve_retrieval.py`, `scripts/serve_executor.sh`) and follows the lab's
GPU ritual — acquire before, release after; see [`../CLAUDE.md`](../CLAUDE.md).

---

## Rules that apply to everything in here

- **Research code stays in `research/`.** Not in the root `experiments/`, not in
  the root `tests/` — those belong to the harness project.
- **`configs/foundation.yaml` is the only place a constant may live.**
- **Never read `archived/`** — the foundation is deliberately built fresh.
- **Data, indices and checkpoints are not in git** (`data_shared/`,
  `foundation/data/`, trajectories, training dirs); `foundation/scripts/`
  regenerates them. Reports and result CSVs *are* committed.
- **Every run stamps its config, seed and git hash** next to its outputs, and every
  figure and table regenerates from CSVs by script — no hand-edited numbers.
- **dev-200 is frozen.** Refinement uses a 50-task validation slice; every dev-200
  evaluation is logged in `PROGRESS.md` with a date and a reason (target: ≤3 total).
- **Git identity is NatBrian only** — `git config user.name` must print
  "Nathanael Brian" before committing, and commits carry no AI co-author lines.
