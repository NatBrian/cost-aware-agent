# Token-Efficient Reasoning via RL

> Research area: length penalties, budget-conditioned training, and difficulty-adaptive rewards for
> efficient reasoning (2024–2026). Critical question for CASSI: does the field still consist of
> "static, instance-blind penalties," or do adaptive / per-instance methods already exist?
> **Answer up front: instance-adaptive penalties are now the dominant 2025–2026 paradigm.**
> Sources verified against arXiv PDFs downloaded to `research/papers/` (dates as of 2026-07-16).

## Area overview

Efficient-reasoning RL exploded after DeepSeek-R1 (Jan 2025) exposed the "overthinking" problem:
RLVR-trained models inflate chain-of-thought (CoT) length even on trivial prompts. The first wave
(Dec 2024 – Mar 2025) used *uniform* mechanisms: a global length penalty applied to correct
responses with a fixed trade-off coefficient α (Arora & Zanette; Kimi k1.5 long2short), a
user-supplied target budget in the prompt (L1/LCPO; TALE-EP; s1 budget forcing), hard truncation
with progressively tighter limits (ThinkPrune), or offline length-preference optimization
(O1-Pruner). Even in this wave, "static" is only half-true: Arora & Zanette normalize the penalty
*per prompt* (mean/std of correct-response lengths for that prompt), and Kimi k1.5 uses per-prompt
min/max normalization — the penalty *strength* is static but the signal is already query-relative.

The second wave (Mar–Aug 2025) made difficulty adaptation the explicit design goal, using the same
free signal CASSI's oracle relies on — per-problem statistics of sampled rollouts. DAST builds a
per-problem Token Length Budget from sampling accuracy; ALP scales the per-token penalty inversely
with each prompt's online solve rate; LASER-D assigns dynamic difficulty-bucketed target lengths
re-estimated during training; AdaptThink/AutoThink learn a per-instance binary think/no-think
decision; AdaCtrl trains the model to self-assess difficulty (with a calibration reward) and
penalizes length only when it self-tags a problem easy; HAPO keeps a per-problem history state
(shortest correct response so far) as an evolving reward reference; GFPO adds an
adaptive-difficulty variant that modulates its filtering strength per prompt. Several of these are
now peer-reviewed at top venues (ALP: NeurIPS 2025 spotlight; Arora & Zanette: NeurIPS 2025;
AdaptThink: EMNLP 2025; AutoThink: NeurIPS 2025; L1: COLM 2025; HAPO: AAAI 2026; TALE: ACL 2025
Findings). The papers themselves use CASSI's exact critique as *their* motivation — ALP:
prior methods "treat all problems alike regardless of difficulty"; HAPO: "universal rewards are
suboptimal due to their lack of adaptivity."

A third, still-thin wave (late 2025 – 2026) pushes toward CASSI's territory: e1 trains models to
follow a continuous effort parameter; TAB (CMU, Apr 2026) trains a *separate small budget-allocator
policy* (1.7B) via GRPO to assign per-turn budgets to a frozen 8B solver in multi-turn reasoning;
2026 arXiv follow-ups (LEAD, TRiMS, AALC, EntroCut, "Think Dense Not Long") continue refining
adaptive length rewards. What remains genuinely unoccupied: heterogeneous agentic costs (tools,
dollars — everything here is CoT tokens on math), step-level process rewards for economic judgment
(all rewards here are outcome-level per response/trajectory), post-hoc oracle *stopping* labels,
and any co-training loop between a stopping/budget model and the executor.

## Core papers

### L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning (LCPO) (Aggarwal & Welleck, 2025, COLM 2025; arXiv 2503.04697)

- **Read from:** PDF pages 1–5 (`2503.04697_l1-lcpo.pdf`)
- **Problem:** Reasoning models cannot allocate a target test-time compute budget; S1 budget forcing degrades accuracy via mid-thought truncation / "Wait" insertion.
- **Method:** Prompt-conditioned length control. Each training prompt is augmented with "Think for n_gold tokens," n_gold ~ U(100, 4000). LCPO-Exact reward: `r = I(y = y_gold) − α·|n_gold − n_y|` with **fixed global α = 0.0003**. L1-Max variant: `r = I(y = y_gold) · clip(α·(n_gold − n_y) + δ, 0, 1)`, δ = 0.5 (soft max-budget constraint). Penalty is **budget-conditioned** (target comes from the prompt, chosen by the user) with a **static α**; the model *learns* to satisfy arbitrary budgets, but the budget itself is exogenous, not difficulty-derived. **Not instance-adaptive by itself** — the model does not decide the budget (though within L1-Max it can undershoot).
- **Training / RL usage:** GRPO (VeRL), 700 steps, base DeepScaleR-1.5B-Preview, 4K train context, batch 128, lr 1e-6.
- **Experiments & benchmarks:** AIME 2025, MATH, AMC, Olympiad-Bench, GPQA, LSAT, MMLU; target lengths {512, 1024, 2048, 3600}; 16 seeds.
- **Key results:** Beats S1 by 100–150% relative / 20–25% absolute accuracy at 512–1024 token budgets; log-linear accuracy-vs-length slope 0.24 (vs 0.37 for S1); mean length deviation ~3%; "short-CoT" L1-1.5B matches GPT-4o at equal token budgets (up to +10% over the non-reasoning counterpart).
- **Limitations:** Requires the user to specify the budget a priori (ALP's Pareto analysis shows L1-Exact has adaptation ratio 1.01× — enforced uniformity; L1-Max only 1.36×); single-turn math CoT; token cost only.
- **Relation to CASSI:** CASSI's planned baseline; CASSI's characterization of L1 as a fixed/exogenous-budget method is **accurate**. THREAT: **LOW-MEDIUM** — it does not adapt per instance, but it is the standard budget-conditioned baseline CASSI must beat on the Pareto frontier, and its models are reused as controllable solvers by later systems (TAB).

### Training Language Models to Reason Efficiently (Arora & Zanette, 2025, NeurIPS 2025; arXiv 2502.04463)

- **Read from:** PDF pages 1–6 (`2502.04463_reason-efficiently.pdf`)
- **Problem:** Long-CoT deployment cost; want a *family* of models trading compute for accuracy via one knob.
- **Method:** Maximize `E[1{y = y*}·(1 − α·f(LEN(y)))]` where `f(LEN(y)) = σ((LEN(y) − MEAN(x)) / STD(x))` and MEAN/STD are **per-prompt** statistics of *correct* responses estimated from online rollouts. α ∈ [0,1) is a **single static hyperparameter** (one model per α). Penalty applies **only to correct responses**. Important nuance: the paper explicitly states per-prompt normalization "ensures that longer chains of thought on hard problems are not disproportionately penalized" — so the penalty *signal* is query-relative even though the *strength* α is uniform. **Partially instance-adaptive** (per-prompt normalized, correct-only gating creates emergent difficulty-dependent compression) but no explicit difficulty term. Includes Propositions 4.3/4.4: the population-level maximizer preserves accuracy and yields the shortest correct solution per prompt.
- **Training / RL usage:** PPO with RLOO advantage estimator (not GRPO), only ~100 RL steps (~200 gradient updates) on DeepSeek-R1-Distill-Qwen-1.5B/7B.
- **Experiments & benchmarks:** GSM8K, MATH500, AIME2024 (+ CommonsenseQA, LogiQA in appendix); α sweep {0.05, 0.1, 0.2, 0.4}.
- **Key results:** 7B: ~50% token reduction with <5% accuracy loss overall; reduction is difficulty-dependent — 16% fewer tokens on AIME (−3.3 pts), 37% on MATH500 (−2.2), 65% on GSM8K (−1.7).
- **Limitations:** α is static and picking it is manual (each α = one training run); no explicit difficulty estimate; response-level (outcome) reward only; math-only; ALP shows its uniform penalty ("R1-Alpha") has the *worst* efficiency score on hard mixtures.
- **Relation to CASSI:** A named CASSI baseline. Calling it a "static penalty" is defensible for α but misleading if presented as instance-*blind* — the per-prompt normalization is instance-relative and the trained models already exhibit difficulty-dependent length. THREAT: **MEDIUM** — weakens the strawman's wording; CASSI must characterize it precisely or reviewers who know the σ-normalization will object.

### Just Enough Thinking: Efficient Reasoning with Adaptive Length Penalties Reinforcement Learning (ALP) (Xiang, Blagden, Rafailov, Lile, Truong, Finn, Haber, 2025, NeurIPS 2025 spotlight; arXiv 2506.05256)

- **Read from:** PDF pages 1–7 (`2506.05256_alp-adaptive-length-penalty.pdf`)
- **Problem:** Explicitly the same framing as CASSI's H5: SFT-on-short-traces, user budgets, and "RL with uniform penalties … treat all problems alike regardless of difficulty."
- **Method:** **Per-prompt difficulty-adaptive penalty.** During training, empirical solve rate `p_solved(q) = (1/K)·Σ 1[answer(y^(k)) = y*]` is computed from the K rollouts already sampled for group-based RL. Reward: `r(y,q) = 1[answer(y) = y*] − β·N·max(p_solved(q), 1/K) · (per-token cost 1/N)` — i.e., the per-token penalty **weight scales inversely with prompt difficulty** (high solve rate → strong penalty; hard prompts nearly unpenalized; clip at 1/K keeps some penalty for unsolved prompts). β global (1e-7), N = max trace length. **Explicitly instance-adaptive** ("difficulty-conditioned penalty that enables models to learn instance-specific computation allocation"), zero extra compute since GRPO/RLOO/Reinforce++ already sample K rollouts.
- **Training / RL usage:** GRPO (VeRL) on DeepScaleR-1.5B, 100 steps, batch 512, 16,384 context.
- **Experiments & benchmarks:** MATH-500, OlympiadBench, AIME 2024+2025 combined; inference budgets {512, 1024, 2048, 4096}; baselines L1-Exact, L1-Max, ThinkPrune-2K, "R1-Alpha" (Arora & Zanette, α = 0.2); difficulty-mixture stress tests (0→60% AIME content).
- **Key results:** ~50% average token reduction at maintained Pass@1; at 1024-token budget, +40% Pass@1 over base on MATH; uses only 21% of tokens on the easiest 50% of problems; **adaptation ratio 5.35×** (tokens hard/easy) vs 1.01× L1-Exact, 4.57× R1-Alpha; best efficiency score 0.68; most robust accuracy under difficulty shift (≥0.75 accuracy up to 12% AIME mix).
- **Limitations:** Outcome-level reward (no step-level signal); difficulty = online solve rate (needs K rollouts and verifiable answers); single-model, single-turn math CoT; token cost only; no explicit stopping decision.
- **Relation to CASSI:** **The single strongest counterexample to CASSI's "static, instance-blind penalties" claim.** ALP is per-instance, difficulty-adaptive, trains the executor with RL, and is peer-reviewed (NeurIPS 2025 spotlight) with the exact "adaptive computation allocation" narrative and even the same H5 evidence (difficulty–token correlation). THREAT: **HIGH** — CASSI cannot claim adaptive penalties don't exist; it must reposition against agentic/heterogeneous-cost/step-level/separate-stopper axes. **Instance-adaptive: YES (continuous, per-prompt solve rate).**

### DAST: Difficulty-Adaptive Slow-Thinking for Large Reasoning Models (Shen et al., 2025, arXiv 2503.04472, v3 Jan 2026; China Unicom)

- **Read from:** PDF pages 1–6 (`2503.04472_dast.pdf`)
- **Problem:** Uniform CoT compression degrades hard-task performance; want length that scales with problem difficulty.
- **Method:** **Per-problem continuous budget.** Token Length Budget `L_budget = p·L̄_correct + (1−p)·L_max`, where `p = c/N` is the per-problem sampling accuracy over N rollouts (p=0 → budget = max length; easy → budget ≈ mean correct length). Reward calibration around the budget with `λ = (L_i − L_budget)/L_budget`: `reward = max(−0.5λ + 0.5, 0.1)` if correct, `min(0.9λ − 0.1, −0.1)` if incorrect (incorrect-but-short is pushed to think *longer*, up to the TLB). Ranked pairs (Dual-Correct: prefer much shorter correct; Dual-Incorrect: prefer *longer* attempt) train via SimPO. **Instance-adaptive: YES (continuous TLB per problem)** — though implemented as *offline preference optimization*, not online RL.
- **Training / RL usage:** Not online RL — SimPO preference training (1 epoch, lr 5e-6) on 20 sampled responses/question from MATH train; 10,295 / 9,813 pairs for DS-7B / DS-32B.
- **Experiments & benchmarks:** DeepSeek-R1-Distill-Qwen-7B and 32B; MATH-500, AIME 2024, GPQA; baselines CCoT, CoD, SFT-Shortest, SimPO-Shortest/Cosine/LenPenalty.
- **Key results:** >30% average token reduction while preserving accuracy on hard tasks (per abstract); AdaptThink's reproduction: DAST-1.5B = −42% length at −0.6 acc; DAST-7B = −33.4% length but −3.2 acc, illustrating its trade-off; one of the first (Mar 2025) explicitly difficulty-adaptive budget papers, widely used as a baseline.
- **Limitations:** Offline (budget frozen at data-construction time — no co-evolution with the policy); needs N rollouts per problem to estimate p; TLB is a length proxy for difficulty, not a value estimate; math-only.
- **Relation to CASSI:** DAST's TLB is conceptually the closest existing thing to CASSI's per-instance budget tiers, derived from the same rollout-success signal CASSI's oracle uses. THREAT: **HIGH** for the "per-instance dynamic cost adaptation beats static penalties" contribution (their headline claim since March 2025); LOW for architecture (no separate model, no stopping, no loop).

### Learn to Reason Efficiently with Adaptive Length-based Reward Shaping (LASER, LASER-D, LASER-DE) (Liu et al., 2025, arXiv 2505.15612; HKUST + Apple)

- **Read from:** PDF pages 1–8 (`2505.15612_laser-d-adaptive-reward-shaping.pdf`)
- **Problem:** Unifies prior efficient-reasoning RL as length-based reward shaping `R̂(x,y) = C(y) + λ(y)·S(y)` and asks what shape S should take; argues rewards must be **dynamic** (evolve during training) and **difficulty-aware**.
- **Method:** LASER: step-function bonus `S(y) = α·1(L(y) ≤ L_T)` for correct responses (α = 0.5). **LASER-D**: queries bucketed into easy/medium/hard **by correctness rate within the rollout batch** (thresholds k/3, 2k/3); three target lengths, each **automatically re-tuned every N (=20) steps** on a 500-sample monitoring set via the Expected-Correct-Responses metric (smallest L_A with ECR_d = P_{l,d}·|C_d| ≥ 1); +3.5% compute overhead. LASER-DE: weaker penalty for incorrect-and-over-budget responses to preserve exploration. Their Table 2 formally classifies prior work: truncation (ThinkPrune), group-based (Arora & Zanette `−α·σ((L−Mean)/STD)`; Kimi k1.5 `0.5 − (L−L_min)/(L_max−L_min)`), budget-based (L1). **Instance-adaptive: PARTIALLY (3 difficulty buckets per query, dynamically re-estimated) — difficulty-aware and dynamic by construction.**
- **Training / RL usage:** GRPO on DeepSeek-R1-Distill-Qwen-1.5B/7B/32B, DeepScaleR-Preview 40K dataset.
- **Experiments & benchmarks:** MATH500, AIME2024, AMC2023, OlympiadBench; Pareto frontiers over hyperparameter sweeps.
- **Key results:** 1.5B: LASER-D 60.3% avg accuracy @ 3,520 tokens vs original 56.9% @ 10,177 (**+6.1 pts on AIME24 with −63% tokens**); beats truncation (T_4096: 49.4% @ 1,646), group-based (best 55.3%), and L1-Max; fewer redundant "self-reflections" in qualitative analysis.
- **Limitations:** Bucketed (not continuous) difficulty; target lengths shared within bucket; outcome-level; math-only; single model.
- **Relation to CASSI:** Second-strongest reward-design counterexample: difficulty-aware + dynamically re-calibrated budgets with a monitoring set — an automated version of what CASSI's λ-tier budget state hand-designs. Also supplies the taxonomy CASSI's related-work section needs. THREAT: **HIGH** for the strawman; its unified `C(y) + λ(y)·S(y)` framing makes "all prior penalties are static" unwritable.

### AdaptThink: Reasoning Models Can Learn When to Think (Zhang, Lin, Hou, Feng, Li, 2025, EMNLP 2025 main; arXiv 2505.13417)

- **Read from:** PDF pages 1–6 (`2505.13417_adaptthink.pdf`)
- **Problem:** NoThinking (empty `<think></think>`) beats Thinking on easy problems in both accuracy and cost; models should choose mode per instance.
- **Method:** **Per-instance binary thinking-mode selection via constrained RL.** Objective: maximize P(NoThinking) subject to accuracy ≥ reference model; penalty-form advantage `A(x,y) = 1(y₁ = </think>)·δ + R(x,y) − R̄_ref(x)` with δ = 0.05, where R̄_ref(x) is the **per-prompt** mean reward of the frozen reference (K = 16 pre-sampled). PPO-style clipped loss, no KL. Cold-start solved by importance sampling: half of each batch forced to NoThinking mode. Analysis (Eq. 10–11): NoThinking wins only when `R̄_nothink + δ > max(R̄_ref, R̄_think)` per prompt — i.e., mode choice is decided by per-instance difficulty. **Instance-adaptive: YES (binary mode per instance; per-prompt reference baseline).**
- **Training / RL usage:** VeRL, DeepScaleR 40K dataset, 1 epoch (314 steps), 16K context, DS-R1-Distill-Qwen-1.5B (one 8×H800 node, 32h) and 7B (four nodes, 28h).
- **Experiments & benchmarks:** GSM8K, MATH500, AIME2024; baselines DPO-Shortest, OverThink, DAST, O1-Pruner, TLMRE, ModelMerging, RFT-MixThinking.
- **Key results:** 1.5B: **−53.0% average length, +2.4 accuracy** (GSM8K 83.1 +4.1 @ 480 tokens; MATH500 82.0 +1.4 @ 1,782; AIME 31.0 +1.6 @ 6,679); 7B: −40.1% length, +2.3 acc. NoThinking ratio decreases monotonically with MATH difficulty level: 97.7% (L1) → 50.7% (L5).
- **Limitations:** Binary (think vs not) — no graded budget; reference pre-sampling K responses/prompt before training; math-focused; single model deciding for itself (self-evaluation entangled with execution — relevant to CASSI's "representation conflict" claim, untested here).
- **Relation to CASSI:** Learned per-instance "should I think at all" is a coarse cousin of CASSI's learned per-step "should I continue." THREAT: **MEDIUM-HIGH** — instance-adaptive compute decision exists at EMNLP 2025; but it is a one-shot pre-generation decision, not sequential stopping over an agent trajectory.

### AdaCtrl: Towards Adaptive and Controllable Reasoning via Difficulty-Aware Budgeting (Huang et al., 2025, arXiv 2505.18822 v2 Dec 2025; HKUST + CUHK + PKU, preprint)

- **Read from:** PDF pages 1–5 (`2505.18822_adactrl.pdf`)
- **Problem:** Want both self-adaptive budgets (model estimates difficulty) and explicit user control over reasoning depth.
- **Method:** Two stages. (1) Cold-start SFT on data tagged "[Easy]"/"[Hard]" (from DeepMATH difficulty annotations; easy answers generated concisely by Qwen2.5-7B-Instruct, hard by R1) so the model emits a difficulty self-assessment tag then a correspondingly sized response. (2) **Difficulty-aware GRPO** with three rewards: outcome `r_o ∈ {+1, −1}`; **difficulty-estimation calibration reward** `r_f` (+1 if the generated tag matches the label derived from rollout solve frequency > δ, −1 if missing); **difficulty-aware length reward** `r_l = 1 − (1 − cos((l/L_i)·π))/2` applied **only when self-tagged [Easy]** (L_i = max group rollout length). Total `r = r_o + α·r_f + β·r_l`. **Instance-adaptive: YES (self-assessed difficulty gates the penalty; calibration reward aligns self-assessment with empirical solve rate).**
- **Training / RL usage:** GRPO with group-normalized advantages + KL; rollout-frequency difficulty labels recomputed online.
- **Experiments & benchmarks:** AIME2024, AIME2025, MATH500, GSM8K vs SFT+RL baselines.
- **Key results:** vs standard SFT+RL baseline: accuracy +up to 10.14%, response length **−10.06% (AIME24), −12.14% (AIME25), −62.05% (MATH500), −91.04% (GSM8K)** — near-monotone budget-vs-difficulty allocation; accurate self-difficulty estimation reported; "[Easy]"/"[Hard]" tags double as a user-control interface.
- **Limitations:** Two-bucket difficulty; still ongoing work (repo "will be released"); math-only; single model (self-assessment shares parameters with execution — again an untested single-model version of CASSI's conflict hypothesis).
- **Relation to CASSI:** Closest to CASSI's "budget state + tier" *representation* idea, and its calibration reward parallels CASSI's stopper-accuracy reward. THREAT: **HIGH** for the claim that difficulty-aware budgeting is novel; LOW for the separate-model/process-reward architecture.

### HAPO: Training Language Models to Reason Concisely via History-Aware Policy Optimization (Huang, Zhang, Cardie, 2025, AAAI 2026; arXiv 2505.11225)

- **Read from:** PDF pages 1–5 (`2505.11225_hapo.pdf`)
- **Problem:** Universal budgets and query-level in-batch comparisons can't leverage *cross-epoch history* — the model should beat its own best concise solution per problem.
- **Method:** **Per-problem history state** h_i = min length of previously generated *correct* responses for problem i (initialized Null, updated each encounter). Length reward `rl = max(f(|y|, h_i), c)` if correct, `min(f(|y|, h_i), 0)` if incorrect, 0 if h_i Null, with `f = cos(min(π/2·|y|/h_i, π))`, clip c = −0.7; short-but-incorrect gets neutral 0 (exploration); total `r = 1(correct) + w·rl`, w = 1.0. Their taxonomy: universal budgets (L1, ThinkPrune) vs query-level rewards (Kimi k1.5, Arora & Zanette in-batch; O1-Pruner reference-based) vs their history-based rewards; argues "universal rewards are suboptimal due to their lack of adaptivity." **Instance-adaptive: YES (per-problem evolving reference; adapts across training epochs).**
- **Training / RL usage:** GRPO (PPO in appendix), 5 epochs on only 2,000 DeepScaleR problems; DS-R1-1.5B, DeepScaleR-1.5B, Qwen2.5-1.5B-Instruct.
- **Experiments & benchmarks:** GSM8K, MATH500, AIME2024 (Pass@1 over multiple samples; 32K context); baselines L1-Exact, L1-Max (re-trained), Query-Opt (= Arora & Zanette), O1-Pruner.
- **Key results:** DS-R1-1.5B: **−49% length, −2% accuracy** (GSM8K −68%, MATH500 −54%, AIME −44%); DeepScaleR-1.5B −33%/−5%; Qwen-Inst −59%/−2%; on AIME uses 19% fewer tokens than the best prior baseline at similar accuracy.
- **Limitations:** Requires repeated encounters with the same problems (multi-epoch, small train set); h_i is length-based, not value-based; outcome-level; math-only.
- **Relation to CASSI:** Peer-reviewed (AAAI 2026) proof that per-instance *adaptive* length rewards are established; its universal-vs-query-level taxonomy is the vocabulary reviewers will use against a "static penalties" strawman. THREAT: **HIGH** for the framing; no overlap with stopping models or agents.

### Sample More to Think Less: Group Filtered Policy Optimization for Concise Reasoning (GFPO) (Shrivastava, Awadallah, Balachandran, Garg, Behl, Papailiopoulos, 2025, arXiv 2508.09726; Microsoft Research, preprint)

- **Read from:** PDF pages 1–5 (`2508.09726_gfpo.pdf`)
- **Problem:** GRPO inflates length (Phi-4-reasoning-plus: 4K→14K tokens in 100 steps); on AIME25, in 72% of questions longer responses are more likely wrong.
- **Method:** **Data-filtering as implicit reward shaping.** Sample a larger group G (16–24), retain top-k by a metric — response length or token efficiency (reward/length) — and compute group-normalized advantages **only within the retained subset** (rejected responses get zero advantage). No explicit penalty coefficient at all. **Adaptive Difficulty GFPO:** online difficulty estimate = mean reward of the group; a streaming t-digest tracks the difficulty distribution; retained k set per difficulty bucket (easy 4, medium 6, hard/very-hard 8 of 16) — "first algorithm to dynamically adapt the effective group size based on question difficulty." **Instance-adaptive: YES in the adaptive-difficulty variant (per-prompt online difficulty modulates filtering strength).**
- **Training / RL usage:** GRPO variant (DAPO token-level normalization, verl), Phi-4-reasoning (14B).
- **Experiments & benchmarks:** AIME 24/25, GPQA, Omni-MATH, LiveCodeBench (also OOD).
- **Key results:** Cuts GRPO's length inflation **46–71%** (length metric) and **71–85%** (token-efficiency metric) while matching accuracy; Adaptive Difficulty GFPO best balances accuracy on hard questions; e.g., AIME25 inflation −46.1% (Shortest-8/24), −70.9% (token efficiency).
- **Limitations:** Higher training compute (larger G); filtering metric hand-chosen; outcome-level; single-turn; preprint (under review).
- **Relation to CASSI:** Shows a third mechanism family (selection, not reward) achieving difficulty-adaptive brevity — evidence that the design space is crowded beyond penalty shaping. THREAT: **MEDIUM** — different mechanism, same headline benefit; also from a major industry lab, so reviewers will know it.

### Not All Turns Are Equally Hard: Adaptive Thinking Budgets for Efficient Multi-Turn Reasoning (TAB) (Jali, Nayak, Joshi, 2026, arXiv 2604.05164 v2, Apr 2026; CMU, preprint)

- **Read from:** PDF pages 1–6 (`2604.05164_tab-multiturn-budgets.pdf`)
- **Problem:** All the above are single-turn; in multi-turn reasoning, early verbosity compounds serving cost and budget decisions have temporal dependency, delayed feedback, and credit assignment — a *sequential compute allocation* problem.
- **Method:** Formulated as a multi-objective MDP. **A separate small budgeter policy** π_φ (Qwen3-1.7B, LoRA r=64) observes conversation history + current sub-question and picks a per-turn token budget b_t ∈ {256, 512, 1024, 2048, 4096} for a **frozen solver** (L1-Qwen3-8B-Exact, reused precisely because it obeys prompted budgets). Terminal trajectory reward `r = acc(x) − λ·max(0, Σ_t b_t − B)` (hinge on the **global per-problem budget** B; λ = 0.001); GRPO with the same trajectory advantage assigned to all turns (no value network). Variant TAB All-SubQ conditions on all past+future sub-questions when a plan exists. **Instance-adaptive: YES — per-turn, per-trajectory budget allocation conditioned on history (turn-level difficulty).**
- **Training / RL usage:** GRPO on the 1.7B budgeter only (solver and user-decomposer frozen), 125 steps, batch 64, MATH Level-5 training problems, budget penalties B ∈ {3k, 5k, 8k, 10k}.
- **Experiments & benchmarks:** MATH-500, AMC23, MATH Level-5, OlympiadBench, AIME25; baselines: static per-turn budgets, off-the-shelf LLM-judge budgeter (individual and multi-turn).
- **Key results:** Up to **35% token savings** at maintained/improved accuracy vs static and LLM-judge baselines (macro average across five benchmarks); TAB All-SubQ up to **40%**; superior accuracy-token frontier across all benchmarks.
- **Limitations:** Solver frozen (no executor training, no self-reinforcing loop); allocates budgets but never decides to *stop* — number of turns is fixed by the decomposition; math sub-questions, not tool-using agents; terminal reward only (they note the credit-assignment problem but sidestep it with trajectory-level advantages); no oracle labels — pure RL exploration over budget choices.
- **Relation to CASSI:** **Structurally the closest paper to CASSI found in this area**: a small RL-trained economic controller (1.7B) supervising a larger (8B) reasoner per step under a cost-accuracy objective. CASSI's remaining deltas: stopping (not just sizing), heterogeneous costs, post-hoc O(T) oracle labels instead of pure RL exploration, the stopper-as-process-reward bridge that *trains the executor*, and the closed loop. THREAT: **HIGH** for "small model supervises large model per-step compute" novelty; CASSI must cite and differentiate explicitly.

## Peripheral papers

**TALE: Token-Budget-Aware LLM Reasoning (Han et al., arXiv 2412.18547, ACL 2025 Findings)** — *(read: PDF pages 1–4, `2412.18547_tale-token-budget.pdf`)* The earliest per-problem budget paper (Dec 2024). TALE-EP has an LLM zero-shot estimate a budget per question and injects it into the prompt; TALE-PT internalizes budgets via SFT/DPO post-training. Introduces "token elasticity": under-sized budgets *increase* actual token use, so minimal-feasible budget is found by binary search (~90.9% of GSM8K samples satisfy the monotonic feasibility assumption). Results: TALE-EP −67% output tokens, <3% accuracy drop (GPT-4o-mini); TALE-PT ~−50% vs vanilla CoT. Not RL and estimation is heuristic, but it is per-instance budget estimation predating CASSI's plan; CASSI already cites it. Instance-adaptive: YES (estimated). Threat: MEDIUM.

**Kimi k1.5: Scaling Reinforcement Learning with LLMs (Kimi Team, arXiv 2501.12599, Jan 2025)** — Frontier-lab report whose long2short section includes an RL **length penalty with per-prompt min/max normalization**: reward `0.5 − (L − L_min)/(L_max − L_min)` for correct responses, `min(0, ·)` for incorrect (as formalized in LASER-D Table 2), plus model merging, shortest-rejection-sampling, and long2short DPO. Static weight, query-relative signal. Notable as the industrial origin of the group-normalized length penalty. Instance-adaptive: signal per-prompt-relative, strength static. Threat: LOW-MEDIUM.

**O1-Pruner: Length-Harmonizing Fine-Tuning (Luo et al., arXiv 2501.12570, Jan 2025)** — Offline RL-style objective: pre-sample the reference model's per-problem accuracy/length baseline, then optimize shorter-than-reference solutions under an accuracy constraint (PPO-style surrogate). Per-problem *reference-based* comparison (HAPO's taxonomy: "query-level"). Up to ~50% inference-time reduction on math with accuracy maintained; in AdaptThink's table, 1.5B: −33.9% length, −1.0 acc. Instance-adaptive: partially (per-problem reference, static weights). Threat: LOW-MEDIUM.

**ThinkPrune: Pruning Long Chain-of-Thought of LLMs via RL (Hou et al., arXiv 2504.01296, Apr 2025)** — Iterative RL with a hard token-limit clip: any response unfinished at the limit gets zero reward; the limit tightens over rounds (4k→3k→2k). Simple, uniform (same limit for all problems), effective: DS-R1-1.5B keeps AIME24 performance with −~50% tokens (ALP measures its adaptation ratio at 2.81× — moderate, likely from truncation not learned adaptation). A named CASSI-adjacent baseline; the "static cutoff" description fits it. Instance-adaptive: NO. Threat: LOW.

**ShorterBetter: Guiding Reasoning Models to Find Optimal Inference Length (Yi et al., arXiv 2504.21370, Apr 2025)** — Defines Sample Optimal Length (SOL) = length of the *shortest correct* response among n rollouts per problem (falls back to max if none correct), and rewards proximity to SOL. A **dynamic per-problem target length** recomputed every step — effectively a cheap online oracle for "how long does this problem need." Up to 80% output-length reduction on DeepSeek-Distill-Qwen-1.5B in- and out-of-domain with maintained accuracy. Conceptually notable for CASSI: SOL is a hindsight per-instance optimum, a 1-D cousin of CASSI's post-hoc oracle t*. Instance-adaptive: YES. Threat: MEDIUM-HIGH (hindsight per-instance optimal-length labels already exist).

**ConciseRL: Conciseness-Guided RL (Dumitru et al., arXiv 2505.17250, May 2025)** — Replaces token counting with a hyperparameter-free **LLM-judge conciseness score** as the reward signal, making the penalty context-dependent per trace. MATH: up to 31× fewer tokens on easy problems, +7.5% accuracy with 3.6× fewer tokens on the hardest tier. Shows semantic (non-length) cost signals; judge cost is the catch. Instance-adaptive: YES (per-trace judged). Threat: MEDIUM.

**AutoThink / Learning When to Think (Tu et al., arXiv 2505.10832, NeurIPS 2025)** — Multi-stage RL over an ellipsis-triggered stochastic think/no-think switch; stage-wise reward shaping stabilizes mode selection then optimizes accuracy and brevity. DS-R1-1.5B: +6.4% relative accuracy with −52% tokens. With AdaptThink, establishes learned binary per-instance compute decisions at NeurIPS 2025. Instance-adaptive: YES (binary). Threat: MEDIUM (same axis as AdaptThink).

**SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning (Li et al., arXiv 2505.11274, May 2025)** — The model first **predicts its own token budget** for the query, then reasons within it; budget-guided GRPO reward penalizes deviation from the self-predicted budget while rewarding correctness. ~61% average response-length compression on math with accuracy roughly maintained; supports user-specified budget override. Single model doing self-budgeting (vs CASSI's separate stopper). Instance-adaptive: YES (self-predicted budget). Threat: MEDIUM.

**Scalable Chain of Thoughts via Elastic Reasoning (Xu et al., arXiv 2505.05315, May 2025; Salesforce)** — Splits generation into thinking and solution phases with **independently allocated budgets**; budget-constrained rollout (truncated thinking during GRPO training) teaches robustness to interruption, so any inference-time budget yields a complete solution. Trained once, generalizes across budgets; E1-Math-1.5B matches L1-Exact-style control with far less training. Inference-time budget enforcement rather than learned per-instance allocation. Instance-adaptive: NO (budgets exogenous; robustness is what's learned). Threat: LOW-MEDIUM.

**BudgetThinker: Budget-Aware LLM Reasoning with Control Tokens (Wen et al.?, arXiv 2508.17196, Aug 2025)** — Periodically inserts special control tokens during generation announcing **remaining budget**, trained by SFT then RL with a budget-adherence + accuracy reward; improves budget compliance across curriculum of budgets. A CASSI-named baseline; budget is exogenous, adherence is learned. Instance-adaptive: NO. Threat: LOW.

**LAPO: Internalizing Reasoning Efficiency via Length-Adaptive Policy Optimization (Wu et al., arXiv 2507.15758, Jul 2025)** — Two-stage RL: stage 1 discovers each problem's distribution of *successful solution lengths*; stage 2 feeds these statistics back into the model's context as meta-cognitive guidance (the model plans its own budget), turning length control "from external constraint into intrinsic capability." Up to −40.9% tokens with +2.3% accuracy on math benchmarks (per abstract). Another per-problem hindsight-length-statistics method. Instance-adaptive: YES. Threat: MEDIUM.

**e1: Learning Adaptive Control of Reasoning Effort (arXiv 2510.27042, Oct 2025)** — Trains models to obey a **continuous effort parameter** interpreted as a fraction of the current average CoT length per query; RL makes allocation proportional to difficulty within any effort setting. 2–3× CoT reduction at maintained/improved accuracy from 1.5B to 32B. Continuous-dial successor to L1. Instance-adaptive: PARTIAL (dial exogenous, allocation within dial adaptive). Threat: MEDIUM.

**2026 rapid follow-ups (spot-checked via search, abstracts only):** LEAD "Length-Efficient Adaptive and Dynamic reasoning" (2605.09806); TRiMS "Real-Time Tracking of Minimal Sufficient Length via RL" (2603.17449); AALC "adaptive accuracy-length control" — length reward weight scheduled by online validation accuracy (2506.20160); DiffAdapt — difficulty-adaptive Easy/Normal/Hard inference strategies (2510.19669); EntroCut — entropy-guided adaptive truncation (2601.22617); "Think Dense, Not Long" — decoupled conditional advantage (2602.02099); ADaPT token-level decoupling (2606.19919); "Not All Errors Are Equal: Consequence-Aware Reasoning Compute Allocation" (2606.04402). The stream is active and uniformly *adaptive* in branding; none found trains a separate stopping model or touches tool-cost agents. Surveys covering this taxonomy: "Stop Overthinking" (2503.16419), "Harnessing the Reasoning Economy" (2503.24377), "Towards Concise and Adaptive Thinking" (2507.09662, has a dedicated difficulty-aware-reward section incl. LASER-D, ALP), "Reasoning on a Budget" (2507.02076), "Don't Overthink It: Efficient R1-style LRMs" (2508.02120).

## Synthesis

### Landscape

Three mechanism families now coexist: (1) **reward shaping** (penalties/bonuses on length: Arora & Zanette, Kimi k1.5, ALP, DAST, LASER-D, HAPO, AdaCtrl, ShorterBetter, ConciseRL, AALC); (2) **budget conditioning** (budget in prompt/control tokens, learned adherence: L1/LCPO, TALE, SelfBudgeter, BudgetThinker, Elastic Reasoning, e1, TAB); (3) **selection/filtering and mode switching** (GFPO; AdaptThink, AutoThink, DiffAdapt). The field's own vocabulary (HAPO, LASER-D, surveys) already distinguishes "universal/static" from "query-level/adaptive" rewards — and by mid-2025 the adaptive column is where all new work lands. Difficulty is operationalized almost universally as **online per-prompt solve rate over the RL group's rollouts** (ALP, LASER-D, DAST, AdaCtrl, GFPO) or **hindsight per-problem length statistics** (ShorterBetter's SOL, HAPO's h_i, LAPO) — i.e., cheap post-hoc/rollout-derived labels, the same information class as CASSI's oracle t*, but 1-dimensional (length), outcome-level, and never turned into a reusable stopping controller.

### Classification of each method's penalty/budget

| Method (venue) | Mechanism | Static coefficient? | Budget-conditioned? | Difficulty-adaptive? | Per-instance signal | Instance-adaptive verdict |
|---|---|---|---|---|---|---|
| L1/LCPO (COLM'25) | prompt-target penalty `−α·|n_gold−n_y|` | α static | YES (exogenous) | NO | none (user budget) | NO |
| Arora & Zanette (NeurIPS'25) | `−α·σ((L−μ_x)/σ_x)`, correct-only | α static | NO | implicit (correct-only + per-prompt norm) | per-prompt length stats | PARTIAL |
| Kimi k1.5 (arXiv'25) | per-prompt min-max length reward | static | NO | implicit | per-prompt length stats | PARTIAL |
| ThinkPrune (arXiv'25) | hard clip, iterative 4k→2k | limit static/scheduled | global limit | NO | none | NO |
| O1-Pruner (arXiv'25) | offline ref-based length RL | static | NO | partial (per-problem reference) | ref accuracy/length | PARTIAL |
| TALE (ACL'25 F) | estimated budget in prompt / PT | n/a | YES (estimated) | YES (est.) | LLM-estimated budget | YES (heuristic) |
| DAST (arXiv v3 '26) | TLB-calibrated reward → SimPO | shape static | per-problem TLB | **YES** | solve rate p = c/N | **YES (continuous)** |
| ALP (NeurIPS'25 spotlight) | per-token penalty × solve-rate | β static, weight adaptive | NO | **YES** | online p_solved(q) | **YES (continuous)** |
| LASER-D (arXiv'25) | step reward at dynamic target | α static, L_A adaptive | bucket budgets | **YES** | rollout correctness buckets | YES (bucketed, dynamic) |
| AdaptThink (EMNLP'25) | think/no-think constrained RL | δ static | NO | **YES** | per-prompt ref reward R̄_ref(x) | YES (binary) |
| AutoThink (NeurIPS'25) | multi-stage mode RL | staged | NO | YES | rollout stats | YES (binary) |
| AdaCtrl (arXiv'25) | self-tag-gated cosine penalty + calibration | α,β static | tags as UI | **YES** | self-assessed + rollout freq | YES (self-assessed) |
| HAPO (AAAI'26) | history-ref cosine reward | w static | NO | YES (via history) | per-problem min correct len h_i | **YES (evolving)** |
| ShorterBetter (arXiv'25) | distance to SOL | static | NO | YES | shortest-correct-of-n | **YES (hindsight)** |
| ConciseRL (arXiv'25) | LLM-judge conciseness reward | hyperparam-free | NO | YES (semantic) | per-trace judge score | YES |
| GFPO (arXiv'25, MSR) | top-k filtering (+adaptive k) | k static / bucketed | NO | YES (variant) | group mean reward, t-digest | YES (variant) |
| SelfBudgeter (arXiv'25) | self-predicted budget + GRPO | static | YES (self) | YES | self-estimate | YES (self-predicted) |
| Elastic Reasoning (arXiv'25) | separate think/solve budgets, trunc.-robust GRPO | n/a | YES (exogenous) | NO | none | NO |
| BudgetThinker (arXiv'25) | remaining-budget control tokens | static | YES (exogenous) | NO | none | NO |
| e1 (arXiv'25) | continuous effort dial | dial exogenous | YES (relative) | YES (within dial) | avg-length-relative | PARTIAL |
| TAB (arXiv'26) | separate budgeter policy, hinge on Σb_t | λ static | YES (global B) | **YES (per turn)** | conversation history | **YES (sequential)** |

### Gaps (what nobody in this area does)

1. **Heterogeneous, non-token costs.** Every method prices CoT tokens; none prices tool calls, API dollars, or wall-clock in an agent loop. "Cost" = length everywhere.
2. **Step-level / process-level economic rewards.** All rewards are outcome-level per response (or per trajectory in TAB). No method emits a per-step continue/stop value signal usable as a process reward (ALP's related work explicitly notes most RL frameworks lack step-level reward support).
3. **A separate, transferable stopping model.** Only TAB has a second model, and it sizes budgets for a frozen solver rather than deciding termination or training the executor. Nobody tests CASSI's "representation conflict" hypothesis — all others are single-model self-regulation.
4. **Sequential stopping in open-ended agent trajectories.** AdaptThink/AutoThink decide *before* generating; L1-family decides *how much*; nothing decides "stop now" mid-trajectory based on marginal value of continuing.
5. **Closed training loops.** No method uses its budget/stopping signal to re-train the executor and then re-derive labels (no oracle→controller→process-reward→executor cycle anywhere).

### Top threats to CASSI's "static penalty" claim (ranked)

1. **ALP (2506.05256, NeurIPS 2025 spotlight)** — a per-instance, difficulty-adaptive RL length penalty with exactly CASSI's H5 narrative (adaptation ratio 5.35×, easy problems get 21% of tokens). The claim "L1/Reason Efficiently/BudgetThinker are single-model *static* penalties" survives only if scoped to those three named systems; as a claim about the field it is false since June 2025.
2. **TAB (2604.05164, Apr 2026)** — small RL-trained budget-allocation policy supervising a larger frozen solver turn-by-turn under a cost-accuracy reward: overlaps CASSI's "small model supervises large executor" architecture in the multi-turn regime. Must be cited and differentiated (stopping vs sizing; executor training vs frozen; oracle labels vs pure RL; agentic tool costs vs math sub-questions).
3. **DAST + LASER-D + AdaCtrl (2503.04472 / 2505.15612 / 2505.18822)** — the difficulty-adaptive budget trio: continuous per-problem TLB, dynamically re-calibrated difficulty-bucketed targets, and calibrated self-assessed difficulty gating. Together they own "per-instance dynamic budget adaptation beats static penalties" (CASSI contribution 4 / H5) in the CoT regime since March 2025.
4. **HAPO (AAAI 2026) + ShorterBetter** — peer-reviewed per-problem *hindsight* reward references (min/shortest correct length so far ≈ 1-D post-hoc oracle). Weakens the "oracle labels from completed trajectories are new" framing unless CASSI stresses its labels are *value-based stopping times over multi-step trajectories with cumulative heterogeneous cost*, not scalar lengths.
5. **AdaptThink / AutoThink (EMNLP/NeurIPS 2025)** — learned per-instance compute decisions (binary). Undercut "no training signal for 'good enough, stop now'" if left unqualified; CASSI must say "no *sequential, mid-trajectory* stopping signal in *agents*."

### Opportunities for CASSI

- **Reframe, don't strawman.** Replace "existing penalties are static and instance-blind" with the field's own taxonomy (universal vs query-adaptive, per HAPO/LASER-D) and position CASSI as the *agentic, step-level, heterogeneous-cost* generalization. Cite ALP/DAST/LASER-D as *supporting evidence* for H5 (difficulty↔compute correlation is established in CoT; CASSI extends it to trajectory stopping with r > 0.5 claims).
- **Own the axes nobody occupies:** (a) stop/continue as a *sequential* decision with a learned margin Δ(s_t); (b) cost = tokens+tools+dollars; (c) O(T) post-hoc *value-based* oracle labels (vs solve-rate or shortest-length proxies); (d) stopper-as-PRM that trains the executor (vs TAB's frozen solver); (e) the closed loop with monotone-improvement analysis.
- **Adopt the strongest baselines from this area:** ALP and LASER-D (adaptive penalties), TAB-style budgeter (separate-model control), L1-Max and Elastic Reasoning (budget-conditioned), AdaptThink (binary adaptive). Beating only L1/Arora & Zanette will be judged as beating 2024-era strawmen.
- **Test the representation-conflict claim against self-regulating single models** (SelfBudgeter, AdaCtrl) — these are the real ablation comparators for "separate stopper is necessary," and none of them reports such an ablation, so CASSI can claim that experiment.

*Not found despite search:* no 2025–2026 paper was found that trains a separate stopping model on post-hoc oracle stopping labels or uses a cost-aware stopping value as a process reward for executor RL — CASSI's core loop appears unoccupied in this literature. All seed papers were located; none was un-findable.
