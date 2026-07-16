# CASSI Project Guide — read this to fully understand the project

**Audience:** an AI agent (or human) with ZERO prior context. This is the master
explanation: what the research is, why every piece of code exists, how the documents
relate, what's done, what's next, and what must never be changed. Reading order:
this file → `HANDOFF.md` (operational status + next commands) → `../paper_plan_v2.md`
(the authoritative spec) as needed.

**Hard rule inherited from the user:** never read `research/archived*` (stale outputs).

**Contents:** §1 what the research is · §2 glossary · §3 pipeline · §4 repo map ·
§5 document map · §6 status · §7 verification · §8 invariants · §9 worked example
(numbers through the whole pipeline) · §10 experiments E1–E6 explained · §11 baselines
B1–B9 explained · §12 ablations A1–A9 · §13 hypotheses & what to do when results
disappoint · §14 config walkthrough · §15 data schema rationale · §16 how the verl
integration works · §17 troubleshooting FAQ · §18 results→paper mapping ·
§19 project history & why key decisions were made · §20 reviewer landmines.

---

## 1. What this research is (plain language)

LLM agents work in loops: think → call a tool → look at the result → repeat. Nothing
in their training ever teaches them that **each extra step costs money and that at some
point the next step is no longer worth its price**. So they overwork: measured studies
show extra steps beyond the optimum actually REDUCE accuracy while burning cost.

Today's fixes are all external: a supervisor process watches the agent and cuts it off
(the agent itself never learns), or a blunt cost penalty is added to the final reward
(too coarse — it can't tell WHICH step stopped being useful).

**CASSI's idea, in one sentence:** train a small "coach" model that estimates, at every
step, whether continuing is still worth it economically — then use the coach's estimate
as a dense, step-by-step training signal for the big "worker" agent, so the worker
*internalizes* when to stop instead of being externally throttled.

- **Coach** = the stopper, Qwen3.5-2B. Learns from hindsight: after collecting many
  finished attempts, we can compute in retrospect where stopping would have been optimal
  (the "Snell envelope" — see glossary). The coach is trained to predict that.
- **Worker** = the executor, Qwen3.5-9B. Trained with RL (GRPO) where the coach's value
  estimate shapes the per-step reward (potential-based shaping — provably doesn't change
  what the best policy is, only makes credit assignment dense).
- **The loop:** better worker → fresh data → recompute optimal stopping → refresh coach
  → retrain worker. At least 2 iterations, with a control arm proving the loop itself
  (not just "more training") is what helps.

**The one claim the paper lives or dies on:** a worker TRAINED with coach-derived
rewards beats (i) the same coach used only as an inference-time supervisor, (ii) using
the hindsight labels directly as rewards with no coach (B9), and (iii) training-free
monitors — on cost at equal accuracy, across 2 domains. Kill-switch experiments K1/K2
test this cheaply BEFORE the expensive pipeline (paper_plan_v2 §12).

**Why it's novel (verified 2026-07-16 by a 94-paper review + 5 independent audits):**
every ingredient exists separately in the 2026 literature — hindsight stop labels
(inference-time only), monitor-as-reward (quality-only, no cost), cost-aware RL
(trajectory-level only). NOBODY converts an explicit quality−λ·cost hindsight optimum
into a trained stopping-value model that trains the executor at the step level, and
nobody measures whether stopping economics can be internalized into weights. That
composition + measurement is the paper.

Target: ICLR 2027 (submission ~Sept 2026). Fallback venues NeurIPS/ICML 2027.

---

## 2. Glossary — paper terms → code symbols

| Term | Meaning (plain) | In code |
|---|---|---|
| `q_t` | how good the agent's current draft answer is at step t (F1 vs gold; ALFWorld: fraction of subgoals done). Ground-truth-derived → label machinery ONLY | `labels/quality.py`; stored on `Step.q` (0.0 during live rollouts by design — filled at collection scoring) |
| `c_t`, tier, wallet | dollar cost of step t; budget tier = how much of the wallet (allowance) REMAINS — tier names describe remaining budget, not urgency: HIGH = >60% left (m=0.5), MEDIUM = 30–60% (m=1.0), LOW = 10–30% (m=2.0), CRITICAL = <10% (m=5.0). m(tier) multiplies the cost penalty, so spending when nearly broke hurts 10× more than when flush. Wallet drawn per (task, GRPO group) | `budget/cost.py` (`tier_from_remaining`, `TIER_MULTIPLIERS`, `draw_wallet`) |
| `U_t` | stopping utility = quality so far minus tier-scaled, normalized, λ-weighted spend: `q_t − Σ λ·m(tier_i)·c̃_i` | `budget/cost.py: stopping_utilities` |
| λ (lambda) | the cost-sensitivity dial. Higher λ = cost matters more = stop earlier. The stopper is λ-CONDITIONED (λ is in its input text), so one model serves the whole dial | config `label.lambda_values`; `stopper/features.py: serialize` |
| Snell envelope | the mathematically correct "should I stop now?" value computed backward over collected trajectories (max of stop-now vs expected-value-of-continuing). Avoids the "prophet bias" of just taking the best-in-hindsight step | `labels/snell.py: snell_labels` (Algorithm 1); prophet version kept only as E4 comparison |
| Δ* / Δ̂ | stop margin: continuation value − stop-now value. >0 ⇒ continue. Δ* = label; Δ̂ = stopper's prediction. Inference stops at Δ̂ ≤ 0 | `StepLabel.delta_*`; `executor/monitor.py` |
| V* / V̂ | the stopping VALUE (unnormalized). V̂ is the shaping potential Φ | third head of `stopper/model.py`; consumed by `executor/shaping.py` |
| PBRS / telescoping | potential-based reward shaping: r_t = Φ(next) − Φ(now). Provably doesn't change the optimal policy. With γ=1 the sum telescopes to −Φ(start) ⇒ trajectory-level advantages CANNOT see it ⇒ step-level credit is mandatory | `executor/shaping.py` (tests prove both facts) |
| min-cohort guard | at late steps few group members are still alive; below 3, fall back to trajectory baseline for those steps only | `shaping.py: step_level_group_advantages` |
| forced continuation | label-collection rollouts IGNORE the agent's ANSWER (log it, keep going to T_max) so we can observe what continuing would have yielded — otherwise labels are censored by the very behavior we train | `executor/react_agent.py` mode=`forced_continuation`; `answered_flag` on Step |
| running draft | every step, the agent must end with `BEST ANSWER SO FAR: ...` — makes per-step quality a free string comparison and gives the stopper stability features. ALL methods share it (fairness) | `labels/drafts.py`; `react_agent.py` scaffold |
| R_base | the worker's terminal reward, SAME economy as the labels (one Lagrangian): `Q_τ − Σ λ·m(tier)·c̃` | `budget/cost.py: base_reward` |
| GRPO / Dr.GRPO | the RL algorithm (group of G=8 rollouts per task, advantages relative to the group); Dr.GRPO = bias-hygiene settings so token savings aren't a length-bias artifact | `executor/train_grpo.py` + `verl_hooks.py` |
| frontier protocol | you can't compare methods at "equal accuracy" unless EVERY method is swept over its own cost knob → 3–5 point frontier, interpolate | `eval/metrics.py: Frontier`; `eval/run_frontier.py` |
| stopping regret | utility gap between where the method stopped and the Snell-optimal stop, measured via a second forced-continuation replay (dual-run protocol) | `eval/run_frontier.py`, `stopper/eval_regret.py` |
| matched lost-correct risk | the LearnStop fairness protocol: sweep each method's own stop threshold and compare cost savings at EQUAL fractions of correct answers sacrificed (1%, 2%, 5%) — "how much do you save per correct answer you're willing to lose" | `eval/metrics.py: matched_lost_correct_risk`; used in §5.3/§5.6 reporting |
| internalization | THE headline measurement: turn the monitor OFF at test time — a worker that still stops well has internalized the economics; also % episodes self-stopped before the monitor fires | `executor/monitor.py` stats; E2 in the plan |
| K1 / K2 | kill-switches. K1: does the coach-as-training-signal beat coach-as-supervisor and labels-without-coach? K2: two models vs one at matched params. Run FIRST; NO-GO → pre-planned pivots | `scripts/p5_killswitch.sh`, plan §12, fallbacks §6 |

---

## 3. The pipeline, end to end (what runs, in order, and where)

```
P0  install/pin stack        scripts/p0_setup.sh          [DONE except GPU smoke]
P1  data + decontamination   scripts/p1_data.sh           [DONE]
──── first GPU session: scripts/smoke_and_pilot.sh ─────────────────────────────
P0' smoke rollout            collect.py --smoke → verify_smoke.py
P2  pilot → wallets          collect.py --pilot → WRITE VALUES INTO configs/cassi.yaml
P2' round-0 collection       scripts/p2_pilot_and_collect.sh (forced continuation, G=8)
P3  Snell labels per λ       scripts/p3_labels.sh (Algorithm 1 + QC memo)
P4  train coach v0           scripts/p4_stopper.sh (SFT; GATE: beat majority+probe on regret)
P5  KILL-SWITCHES K1/K2      scripts/p5_killswitch.sh → GO_NO_GO.log  ← decision point
P6  worker GRPO iteration 1  scripts/p6_grpo_iter1.sh (train_grpo.py + verl_hooks.py)
P7  loop iteration 2         scripts/p7_loop_iter2.sh (frozen-coach control arm = E5)
P8  baselines B2–B9          scripts/p8_baselines.sh
P9  full eval E1–E6 + A1–A9  scripts/p9_eval.sh (everything lands in experiments/results/*.csv)
P10 figures/tables           make figures tables (CSV in → PDF/tex out, no hand edits)
P11 write the paper          paper/ (skeleton compiles; writing order in plan §16)
```

Data flow: rollout JSONLs (schema `common/schema.py`, §11) → `labels/snell.py` →
labeled sets → `stopper/train_sft.py` → coach checkpoint → (a) `executor/monitor.py`
at inference, (b) V̂ into `verl_hooks.py` rewards at training → eval CSVs →
`analysis/` scripts → `paper/figures|tables` → `paper/main.tex`.

---

## 4. Repo map (research/cassi/)

```
common/    schema.py (§11 trajectory JSONL — THE data contract), config.py (loads §17 YAML;
           blocks post-P2 phases until pilot values are filled)
budget/    cost.py — THE economy. One module owns every dollar/tier/U_t/R_base computation
           so coach, worker, and labels provably optimize the same objective.
labels/    snell.py (Algorithm 1 + QC), quality.py (F1/EM/subgoals), drafts.py (draft parse
           + stability features + legacy probe kept only for ablation A5)
stopper/   features.py (§18.1 input text + numeric vector), dataset.py (SFT examples,
           task-level splits, LabelSet persistence), model.py (3-head model, MockStopper,
           load_predictor), train_sft.py (Alg.2; early-stop on held-out REGRET), eval_regret.py
executor/  react_agent.py (shared scaffold, both rollout modes), collect.py (P2/P7 + --pilot
           + --smoke), monitor.py (Alg.4 + internalization stats), shaping.py (PBRS + step
           advantages — the math core, fully CPU-tested), train_grpo.py (full §16 CLI +
           --dry-run), verl_hooks.py (ALL verl-touching code, pinned refs), vllm_client.py,
           envs/ (searchr1_qa.py, alfworld.py, base.py with MockSearchEnv)
baselines/ b1_react … b9_direct_shaping + oracle; registry in __init__.py; each docstring
           states the paper it reimplements, its cost knob, and what question it kills
eval/      metrics.py (frontier/regret/matched-risk), stats.py (§5.6, guards ENFORCED),
           overhead.py (T4 ledger, serving regimes, billing symmetry), run_frontier.py (CLI)
analysis/  figures/f1..f6, tables/t1..t5 (CSV → PDF/tex; Makefile at repo level)
scripts/   p0..p9 phase runners, smoke_and_pilot.sh (first GPU session), launch_grpo.sh,
           killswitch_decision.py, decontaminate.py, download_data.py, run_labels.py, ...
paper/     main.tex (compiles TODAY with article fallback), sections/01..08 (each headed by
           page budget + which plan §s feed it), references.bib (verify entries at P11)
tests/     114 CPU tests (~5s). test_core.py = the math invariants. Run before/after any change.
configs/   cassi.yaml — single source of truth (§17). `pins:` = installed stack versions.
data/      (gitignored) datasets, frozen dev subsamples, searchr1_index/ (64GB E5 + corpus)
third_party/ (gitignored) pinned clones: verl, verl-tool, verl-agent, Search-R1
.venv/     dedicated python env — scripts auto-activate it (scripts/common.sh)
```

---

## 5. Document map — which file answers which question

| Question | Read |
|---|---|
| What exactly is the method/experiments/claims? | `../paper_plan_v2.md` — THE spec. §2 method, §5 experiments, §10 algorithms, §12 kill-switches, §16 runbook, §17 config, §19 sourcing. Where anything disagrees, it wins. |
| What's done / what's next / what commands? | `HANDOFF.md` (+ §3b adjustment map: what to tune after real data, freeze rules) |
| Why is the field-positioning what it is? | `../lit_review/00_overview.md` (+ 10 area files); `../competitor_analysis.md` (beginner-friendly) |
| How harsh were the pre-rewrite reviews? | `../novelty_check/` (5 audit verdicts vs the OLD v5 plan — v2 already answers them; see paper_plan_v2 §14 changelog) |
| What was decided during experiments? | `GO_NO_GO.log` (append-only; created at P5) |
| Machine rules (GPUs, git identity)? | repo-root `AGENTS.md`/`CLAUDE.md`; `/root/dataDisk/liangsheng/Brian/AGENTS.md`; `/root/dataDisk/liangsheng/CLAUDE.md` |

---

## 6. Status snapshot (2026-07-16, end of build session)

DONE: all code (both former wiring gaps closed), 114 CPU tests green, verl dry-run green,
P0 installs+pins, P1 data (decontaminated, manifest), wiki-18 corpus + 64GB E5 index
assembled, Qwen3.5-9B/2B prefetched, paper skeleton compiles, everything pushed to
github.com/NatBrian/cost-aware-agent (private).

NOT DONE (all GPU-gated): smoke rollout, pilot calibration (config `null`s), collection,
labels-on-real-data, coach training, K1/K2, all RL, all experiments, all real numbers, the
paper text. **GPUs are currently off-limits** (machine held by another user's job; user
instruction: do not use or wait for GPUs). First command when that changes:
`bash scripts/smoke_and_pilot.sh`.

Deliberate stubs (documented, will raise NotImplementedError with instructions):
ALFWorld agent-loop under verl (package-name clash — needs own env), K2's
single-multitask arm (**must be implemented BEFORE P5 — K1 alone does not gate a GO;
ideal work during the no-GPU period**, HANDOFF §6), BrowseComp-Plus staging, GAIA
exact-103 filter. The full sanctioned no-GPU work queue: HANDOFF §6.

## 7. How to verify nothing rotted (THE canonical ritual — run anytime, no GPU)

```bash
cd research/cassi && source .venv/bin/activate     # manual shells must activate; scripts self-activate
python -m pytest tests/ -q                                    # expect 114 passed, 1 skipped
python -m cassi.executor.train_grpo --dry-run \
    --config configs/cassi.yaml --domain qa                   # expect "[dry-run] OK"
(cd paper && make)                                            # expect main.pdf builds
git config user.name                                          # expect: Nathanael Brian
```

Never "verify" by re-running `scripts/p0_setup.sh` — it is an installer, and P0 is done.

## 8. The ten commandments (violating these voids the paper)

1. paper_plan_v2.md outranks every other document and any memory of past sessions.
2. Never read `research/archived*`.
3. x_t (stopper input) must NEVER contain ground truth or executor-stated confidence.
4. Label collection = forced continuation; RL rollouts = natural termination. Never mix.
5. One economy: labels, R_base, and eval utilities all come from `budget/cost.py`.
6. Step-level advantages are mandatory (trajectory-level shaping is provably inert).
7. Frozen things stay frozen: eval subsamples, scaffold after first baseline, economy
   after P3, headline λ chosen on dev, GO_NO_GO.log append-only.
8. Every method pays for its own auxiliary inference (billing symmetry) — enforced in code.
9. Kill-switches gate the expensive phases; a NO-GO is a documented pivot, not a failure.
10. Git identity: NatBrian only (enforced by folder config); no AI co-author lines;
    GPUs only via acquire/release, never kill foreign jobs.

---

## 9. A worked example — one task through the entire pipeline, with numbers

Follow ONE HotpotQA task through every stage. All numbers are illustrative but to scale.

**Collection (P2).** Task: "What is the capital of the country where the Eiffel Tower
is?" Gold: "Paris". The group draws wallet B = $0.02 (medium), shared by all G=8
rollouts. One rollout, forced-continuation mode, T_max=10:

| t | action | draft after step | q_t (F1 vs gold) | c_t ($) | spent | tier |
|---|---|---|---|---|---|---|
| 1 | search[eiffel tower country] | EMPTY_DRAFT | 0.00 | 0.002 | 0.002 | HIGH |
| 2 | search[capital of France] | Paris | 1.00 | 0.002 | 0.004 | HIGH |
| 3 | ANSWER "Paris" → logged, forced on | Paris | 1.00 | 0.001 | 0.005 | HIGH |
| 4–10 | more searches (nothing new) | Paris | 1.00 | 0.002/ea | 0.019 | →LOW |

`answered_flag=True` at t=3 (free self-stop measurement). q_t is scored at collection
time only (string compare vs gold) — it never enters x_t.

**Utilities (λ=1, median pilot spend = $0.01 → c̃_t = c_t/0.01).**
U_t = q_t − Σ λ·m(tier_i)·c̃_i. With m(HIGH)=0.5: U_1 = 0 − 0.10 = −0.10;
U_2 = 1.0 − (0.10+0.10) = 0.80; U_3 = 1.0 − (0.10+0.10+0.05) = 0.75 (still paying, no
quality gain); by t=10 (tier multipliers rising as the wallet empties) U_10 ≈ −0.4.
**The curve rises to t=2 then strictly falls — the economics of overwork, in one row.**

**Snell labels (P3, Algorithm 1).** Backward over ALL trajectories: at t=9,
Cont(x_9)=Ê[V_10|x_9]≈−0.4 < U_9 ⇒ stop-region. ... At t=2, the regressor sees states
like x_2 (fresh draft, HIGH tier, overlap low) across the whole batch; some continuations
improved (multi-hop tasks), so Cont(x_2)≈0.71 < U_2=0.80 ⇒ **τ*=2, a*_2=STOP,
Δ*_2 = 0.71−0.80 = −0.09**; at t=1, Cont(x_1)≈0.65 > U_1=−0.10 ⇒ CONTINUE, Δ*_1=+0.75.
Per step we emit (a*, tanh(Δ*/s), V*) — V*_1 = max(U_1, Cont) = 0.65 is the VALUE label.

**Coach training (P4).** Input = the §18.1 text block for x_t (with "λ = 1" inside);
targets = the three labels. One coach for all λ, because λ is in the text.

**Worker training (P6).** RL-mode rollout stops at ANSWER (t=3 here). Coach's value head
gives Φ = V̂: say V̂(x_1)=0.65, V̂(x_2)=0.78, V̂(x_3)=0.74, terminal Φ:=0.
Shaped rewards r_t = Φ(x_{t+1})−Φ(x_t): r_1 = +0.13 (that search moved the state toward
stop-worthy — paid immediately), r_2 = −0.04 (past the frontier — mildly punished),
r_3 = −0.74 (cash out: the telescoping settlement). R_base = 1.0 − (tier-scaled spend)
≈ 0.75 paid at the end. Returns-to-go per step + group normalization per step index
(cohort = rollouts still alive) = the advantages. The group's lazy rollout that searched
10 times gets negative advantage at steps 3–10 — *step-resolved* pressure to stop, which
a trajectory-level penalty cannot deliver (it telescopes away — proven in tests).

**Inference (Alg.4).** User sets λ. Coach reads x_t each step; stop at Δ̂ ≤ 0. Same
state under a tighter wallet serializes with tier=LOW → coach learned to output lower
Δ̂ → earlier stop, no rule table. **Internalization check:** turn the coach OFF —
a well-trained worker still answers at t≈2–3 by itself.

---

## 10. The experiments E1–E6 — what each proves, in plain terms

| ID | Question it answers | Design in one line | Output |
|---|---|---|---|
| E1 | Does CASSI beat everything on cost-at-equal-accuracy? | full grid: CASSI vs B1–B9, 2 domains, 3 seeds at headline points, frontiers per method | T1/T2, F3 |
| E2 | Did the economics move INTO the policy? + does it transfer? | monitor-off eval; % self-stopped; trained worker on OOD sets (BrowseComp-Plus/Bamboogle/2Wiki); coach supervising a frozen live-web agent on GAIA-103; coach across executors (Ministral-3-8B, Qwen3.5-4B) | F4, T5 |
| E3 | Is the λ dial real? Is wallet-awareness learned? | sweep λ at INFERENCE on one fixed worker; PLUS one additional worker actually TRAINED at λ=0.3 (R_base at λ=0.3; coach queried at λ=0.3 via its λ-conditioning — no new labels, the point of the λ-in-input design) to check the inference dial tracks the trained frontier; same tasks under small/medium/large wallets → stop-steps should shift | F3 overlay, E3 CSVs |
| E4 | Are Snell labels actually better than alternatives? | same pipeline, label source swapped: Snell vs prophet-argmax vs TD/GAE vs MC at matched label compute | F5 |
| E5 | Does the LOOP help (not just more training)? | iteration 2 run twice at matched compute: frozen coach vs refreshed coach; the DELTA between arms is the loop's contribution | per-iteration table |
| E6 | Sanity: does stopping track difficulty? | 2-hop vs 4-hop MuSiQue stop-step correlation (consistency check only, no novelty claim) | one appendix figure |

**The two metrics that carry the paper:** (1) dollar cost at iso-accuracy, read off each
method's own knob-swept frontier by interpolation (never compare single points);
(2) stopping regret = utility gap vs the Snell-optimal stop, measured by the dual-run
replay protocol on a fixed 500-task subsample (replay costs billed to the analysis line).

## 11. The baselines B1–B9 — who they are and what question each kills

| # | What it is | Kills the question | Knob |
|---|---|---|---|
| B1 | plain ReAct, no cost signal | how much slack exists at all | none (excluded from iso-claims) |
| B2 | self-eval prompt + calibrated confidence probe (Dynasor-style) | "why not just ask the model if it's done?" — DANGEROUS: literature says it can win | confidence threshold |
| B3 | SupervisorAgent-style training-free monitor (ICLR'26, −29.7% GAIA tokens) | "why train anything?" — the strongest training-free bar | trigger sensitivity |
| B4 | OTC-GRPO (tool-count-scaled outcome reward) | is STEP-level signal needed vs outcome-level? | tool-count coefficient |
| B5 | EAPO (solve-rate-adaptive penalty) | is a learned VALUE better than adaptive scalar pressure? | penalty weight |
| B6 | single model, cost-in-reward (CTA-style) | is the two-model split earning anything? | λ |
| B7 | CaRT+cost, TWO arms (SFT-only / +GRPO) | is RL needed, or does imitating truncated trajectories suffice? | truncation λ |
| B8 | AgentPRM + cost term (pooled return-to-go — NOT per-state MC) | is the STOPPING semantics what matters vs generic value+cost? | λ |
| B9 | **the pivotal one**: our Snell labels as advantages directly, NO coach, same step-level machinery | does the coach earn its existence (generalization to unlabeled states)? | λ |
| oracle | stop at Snell τ* using ground truth | headroom upper bound | none |

Fairness invariants: all share the same scaffold/draft template, the same envs and
price map, and every method's auxiliary inference is billed (B2's probes, B3's triggers,
our coach's queries).

## 12. Ablations A1–A9 (most reuse trained models at near-zero cost)

A1 coach size 0.8B/2B/4B (watch: ~1B meta-models can collapse — ReMA warning) ·
A2 single-model multitask vs two-model at MATCHED total params (the honest version of
"do we need two models") · A3 potential-based vs v5's additive reward (prediction:
additive teaches dawdling) · A4 coach input families (budget-only / +history / +draft
stability) · A5 coach eval frequency every k-th step + legacy-probe vs running-draft
label-source check · A6 SFT-only vs SFT+RL coach · A7 rationale text on/off ·
A8 learned wallet-conditioning vs v5's hand-written δ(tier) rule table (beating it is
a supporting result) · A9 **negative controls**: random coach (noise Δ̂/V̂) and
shuffled-label coach — if the worker still improves, gains were generic dense-reward
regularization, not economics. Cheap and decisive; reviewers love these.

## 13. Hypotheses and the pre-planned "what if it fails" moves (plan §6)

| H | Claim | If it FAILS → the paper becomes |
|---|---|---|
| H1 | CASSI Pareto-beats training-free B2/B3 on ≥2 domains | "when does learned stopping help" regime study (still publishable) |
| H2 | bridge (training) > controller-only (same coach) | **KILL-SWITCH: stop/pivot per §12** |
| H3 | CASSI > B9 direct shaping | coach demoted to optional; paper = "Snell-label shaping for agents" |
| H4 | two models ≥ one at matched params | claim rests on transfer + privileged-info hygiene + runtime control |
| H5 | ≥70% of savings retained with monitor OFF | "partial internalization" framing |
| H6 | savings hold in both serving regimes | regime-conditional recommendation |

A failed hypothesis is a REPORTED RESULT with its fallback framing — never buried.

## 14. Config walkthrough (configs/cassi.yaml — §17)

- `label.lambda_values [0.1,0.5,1,2,5]`: the λ grid for LABELS (one coach learns all).
  `default_lambda 1.0`: headline; confirmed on dev before test (§5.6).
- `label.tier_multipliers {0.5,1,2,5}`: m(tier) — the discretized "shadow price" of
  spending when the wallet is fuller/emptier. `m≡1` = ablation A8's plain economy.
- `label.allowances / cost_normalization`: **null until the P2 pilot** — every later
  phase refuses to run while null (deliberate gate). small=P25, medium=P75, large=2×P90
  of unconstrained pilot spend; median = the c̃ normalizer that makes λ dimensionless.
- `label.regressor lightgbm`: the CROSS-SECTIONAL regressor inside Algorithm 1 (NOT the
  coach — separate on purpose, to avoid label-model coupling).
- `stopper.sft early_stop_metric heldout_stopping_regret`: the coach is selected by the
  metric that matters (regret), not CE loss.
- `stopper.heads action/delta/value = CE 1.0 / MSE 0.5 / MSE 0.5`: three heads; delta is
  tanh-normalized (decisions), value is UNnormalized (the shaping potential must live in
  quality units).
- `executor.grpo`: G=8; `length_norm dr_grpo` + step-level advantage + `min_cohort_guard 3`
  are correctness requirements, not tuning knobs. `advantage step_level` variant
  (`per_step_rtg` vs `shape_segment`) — K1 picks the winner.
- `executor.horizon {qa:10, alfworld:20}`: T_max. Changing it invalidates labels.
- `inference.delta_threshold 0.0`: FIXED — wallet-sensitivity lives in the weights;
  `ablation_A8_rule_table` is the comparator, never the default.
- `cost_model`: reference local token prices + §17 tool fees; API models via the repo
  harness price map. Draft-line tokens are charged to EVERY method.
- `pins:`: the installed stack's exact commits/versions — the ground truth for "what
  APIs am I coding against"; update ONLY at a deliberate re-pin + rerun of dry-run+tests.

## 15. Data schema rationale (common/schema.py — §11)

`StepFeatures` (x_t) is THE anti-hacking boundary. Everything in it is computable by the
deployment harness at inference: budget arithmetic (tokens/calls/dollars/% of wallet/
burn rate/tier), progress signals (step index, how long the draft has been unchanged,
edit distances of recent drafts, retrieval overlap = "are searches finding anything
new?", distinct sources), the draft text itself, task text, and digested recent history.
**Deliberately absent:** anything derived from ground truth, and the executor's own
stated confidence (self-reported confidence is the classic hacking channel). The honest
caveat (plan §2.4): the executor authors the draft, so within an iteration it could
freeze a wrong draft to fake stability — the defense is cross-iteration (label refresh
re-grounds everything in GT) plus the V̂-vs-reward divergence diagnostic.

`Step` adds what only training may see: q (GT quality), c/tier (billing),
answered_flag (forced-continuation ANSWER event). `Trajectory` adds the wallet
(allowance_B, wallet_size, group_id — the WALLET IS PER (task, GROUP), so group members
are comparable) and the outcome dict. JSONL round-trip is tested.

## 16. How the verl integration works (executor/verl_hooks.py — for whoever debugs P6)

verl (pinned commit, see `configs/cassi.yaml pins:`) runs GRPO with an "agent loop"
that rolls multi-turn episodes and, by default, puts ONE scalar reward on the final
token and computes trajectory-level group advantages. CASSI needs per-step rewards and
OUR advantages. Three hooks, all registered from one module import:

1. `CassiReactAgentLoop` — a verl agent loop that internally drives our CPU-tested
   `ReactAgent` scaffold (so the §2.6 template/features are identical in training and
   eval) and records which token ends each step.
2. `CassiAgentLoopManager` — after each rollout wave, groups trajectories by task
   (verl's `uid`), queries the coach checkpoint for V̂ on every visited state, runs the
   untouched `compute_cassi_rewards` (shaped rewards → step returns-to-go → cohort
   advantages), and writes the result into the batch's reward tensor.
3. Registered adv estimator `cassi_step_level` — verl's estimator API only hands over
   the reward tensor, so the manager DIFFERENCE-ENCODES advantages on step-final tokens
   (`A_t − A_{t+1}`) and the estimator's reverse-cumsum reconstructs A_t exactly on
   every token of step t. This bypasses verl's own trajectory-level GRPO advantage
   (which is provably blind to our shaping — §2.4). A round-trip unit test guards the
   encoding; `--dry-run` verifies all three registrations against the pinned source.

Quirks to know: verl's logged "reward" metric shows A₁ per trajectory (artifact of the
encoding — IGNORE it); true economic rewards stream to `<out>/divergence.csv` (feeds
figure F6); validation batches carry real terminal rewards so val metrics stay
readable. Coach V̂ serving defaults to CPU (`CASSI_STOPPER_DEVICE=cuda:N` to change).

## 17. Troubleshooting FAQ (errors you WILL meet, and what they mean)

- `RuntimeError: Pilot calibration missing ...` → by design. Run the pilot
  (smoke_and_pilot.sh) and write the printed values into configs/cassi.yaml. Never
  bypass by inventing numbers.
- `NotImplementedError` in alfworld.py / `--arm single_multitask` / BrowseComp-Plus →
  deliberate walls with instructions inside; see HANDOFF "deliberate stubs".
- `Permission denied (publickey)` on git push → the NatBrian SSH key needs re-adding
  (github.com/settings/keys). NEVER switch to another credential (Brian/AGENTS.md).
- `gpu_acquire.sh` prints "Acquired" but nvidia-smi shows ~100GB used → locks ≠ free
  memory; a foreign job holds the GPUs. Release and stop. Never kill foreign processes.
- `import verl` resolves to verl-agent's 0.3.x fork → the two packages share a name;
  reinstall `third_party/verl` editable LAST (p0_setup.sh now enforces order; the
  dry-run asserts the resolution).
- vLLM serving Qwen3.5 emits `<think>` or drops the draft line → check
  `enable_thinking=False` reaches the chat template (vllm_client.py extra_body) and the
  §19 token-in-token-out note.
- lightgbm "X does not have valid feature names" warning → cosmetic, suppressed in
  snell.py; ignore elsewhere.
- Retriever returns nothing / connection refused → the E5 server isn't up; see
  smoke_and_pilot.sh (index load takes minutes; log at experiments/logs/retriever.log).
- Tests suddenly failing after a verl/transformers upgrade → you changed the pinned
  stack; either re-pin deliberately (update pins:, rerun dry-run + full tests) or
  restore the pin. Never "fix" tests to match an accidental upgrade.

## 18. Where numbers go — results → figures/tables → paper sections

Every experiment writes CSVs to `experiments/results/` (frontier summary + per-instance
files from `eval/run_frontier.py`). `make figures tables` regenerates everything —
NO hand-edited numbers anywhere. Mapping: F1 pipeline schematic (no data) · F2 shaping
intuition (one real trajectory's U/Cont/Δ/τ*) · F3 ← e1_grid + e3_lambda_frontier ·
F4 ← e2_internalization (monitor-off bars, self-stop % across iterations) ·
F5 ← label study CSVs (E4) · F6 ← divergence.csv from training · T1/T2 ← e1_grid with
stats.py CIs · T3 ← ablation CSVs · T4 ← overhead ledger (billing symmetry re-checked
at render — asymmetric CSVs are REFUSED) · T5 ← e2_transfer_*. Paper sections and page
budgets: paper/sections/0*.tex headers quote their sources; writing order and the
claims-audit greps are plan §16 P11 (banned phrases from §14 must not reappear).

## 19. Project history — how we got here and why key decisions were made

1. **Original idea (user's):** budget-aware agents; a monitor that tracks budget and
   judges good-enough; "monitor as reward model" as a new training paradigm.
2. **v5 plan** (research/archived/ — do not read) made strong claims; **15 independent
   research agents** (94 papers read at PDF level) + **5 novelty audits** scored it
   4–5.5/10 and falsified several claims (a "first self-reinforcing cycle" that wasn't
   first; a complexity story based on misreading AgentPRM; a strawman "static penalty"
   framing; a foresight-biased oracle; a reward with a dawdling incentive).
3. **paper_plan_v2** was rewritten around the surviving, verified gap (the cost-aware
   stopping bridge + internalization measurement) and then hardened by FOUR internal
   adversarial review rounds (11+10+12 repairs + 19 consistency fixes — all logged with
   evidence in its §14 changelog). Headline repairs worth knowing: Snell labels replaced
   argmax (prophet bias); potential-based shaping replaced additive rewards (dawdling);
   forced-continuation collection fixed a flaw where the method's own success would
   have destroyed iteration-2 training data; the frontier protocol fixed an
   iso-accuracy metric that was uncomputable as written; the E5 frozen-coach arm made
   the loop claim defensible; model/benchmark choices were web-verified against
   July-2026 SoTA with pre-written rebuttal lines (§19 of the plan).
4. **Implementation session (2026-07-16):** everything in §6 above; four subagent
   builders + one integrator; every module CPU-tested before any GPU exists.

Moral for the next agent: the plan is not a draft — it is the survivor of a deliberate
attempt to kill it. When something looks odd, the §14 changelog usually explains why
it is the way it is. Check there before "simplifying".

## 20. Reviewer landmines (for the agent that writes the paper)

Pre-answered in plan §15 — keep them pre-answered: "why not just prompt/probe?" (B2 is
in every table; RedundancyBench ≤24.9% F1 citation) · "isn't this AgentPRM + cost?"
(B8 exists; delta = stopping semantics + Snell targets + bridge + internalization) ·
"learned monitors get hacked" (objective features, per-iteration refresh, F6 curves) ·
"overhead?" (T4 end-to-end dollars incl. draft tokens + collection + serving, both
regimes) · "contamination?" (§5.6 protocol; headline metric is cost at MATCHED accuracy
— uniform inflation cancels) · "n=103 significance?" (never claimed; CIs only, labeled
transfer indicators) · "cherry-picking?" (frozen subsamples, dev-chosen λ, append-only
GO_NO_GO.log, all seeds/λ in appendix). NEVER let these reappear: "first
self-reinforcing cycle", O(K×T²) complexity claims, "static instance-blind penalties"
strawman, "representation conflict" as theory.
