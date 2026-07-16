# Overthinking & Efficiency-Accuracy Empirics

> Area: empirical evidence that LLMs/agents overthink, that longer reasoning can hurt, and surveys
> of efficient reasoning. Grounds CASSI's motivation (research date: 2026-07-16).
> All core papers were downloaded to `research/papers/` and read from the PDFs (pages noted per
> entry); peripheral entries are based on fetched arXiv abstracts unless stated otherwise.

## Area overview

Between early 2024 and mid-2026 a coherent empirical literature established that "more thinking"
is not monotonically better for LLMs. The first peer-reviewed data point is Chiang & Lee (EACL
2024): on questions whose answers are literally stated in the prompt, RLHF-tuned LLMs still emit
redundant calculations 47–89% of the time, and answers containing calculations are markedly *less*
accurate. With the arrival of o1-style Large Reasoning Models (LRMs), Chen et al. (Dec 2024)
quantified the phenomenon at scale — o1-like models spend ~1,900% more tokens than conventional
LLMs on trivial questions, and >92% of the time the *first* solution round is already correct —
while Wu et al. (2025) showed accuracy vs. chain-of-thought length follows an inverted-U with a
closed-form optimal length that grows with task difficulty and shrinks with model capability.
Hassid et al. (2025) established the within-question version: among 20 samples for the *same*
question, the shortest chain beats the longest by up to +34.5 points. Gema et al. (TMLR 2025) went
further, constructing tasks with genuine *inverse scaling* in test-time compute — longer reasoning
actively degrades accuracy (and even safety behavior) through distraction, spurious-feature
amplification, and framing overfit.

A second, smaller thread moved the evidence from single-turn CoT to *agents*. Cuadron et al.
(2025) is the flagship: on SWE-bench Verified with OpenHands, an LLM-judged "overthinking score"
(favoring internal simulation over environmental interaction) strongly predicts failure (R² = 0.892
for reasoning models), and simply selecting the lowest-overthinking of k samples buys ~30% relative
performance at 43% lower cost. In 2026 the agent-step-waste question became a research area of its
own: RedundancyBench (Hu et al., 2026) defines step-level redundancy in tool-use trajectories via
counterfactual step removal and shows that even frontier prompted LLM judges reach only 24.88%
step-level F1 at detecting redundant steps; Agent-Omit (ICML 2026) and LaRMA corroborate that
agent thoughts/steps are often unnecessary and costly. Two surveys (Sui et al., TMLR 2025;
Wang et al., 2025 "Reasoning Economy") organize the mitigation space — RL length rewards, SFT on
compressed traces, routing, early exit — which is now very crowded for *single-model, single-turn*
efficiency.

The consensus that emerges: (i) overthinking is systematic and worst on *easy* instances
(outcome-efficiency <50% on easy math levels); (ii) the relationship is two-sided — within a
question shorter is better, but harder questions genuinely need more compute (up to 2.9× more
tokens), and *underthinking* is a real dual failure — so any fix must be per-instance adaptive,
not a global length cap; (iii) mechanisms include RLHF/judge length bias and error accumulation;
(iv) prompted self-evaluation is unreliable for economic judgment (weak self-evaluators in
budget-aware evaluation; 24.88% F1 on redundancy detection), which is direct ammunition for
CASSI's claim that a *trained* stopping model is needed. What no paper in this area does: learn a
stopping/value model from hindsight cost-aware labels and feed it back as a process reward to
train the executor — CASSI's loop sits beyond the empirics, which serve as its motivation.

## Core papers

### The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks (Cuadron et al., 2025, arXiv 2502.08235; preprint, ICML format — no confirmed venue as of 2026-07)
- **Read from:** PDF pages 1–9 (full paper)
- **Problem:** LRMs in *agentic* environments face a Reasoning-Action Dilemma: rely on internal
  simulation of the environment vs. acting and getting real feedback. Overthinking := excessive
  reliance on internal reasoning chains over environmental interaction.
- **Method/Study design:** SWE-bench Verified inside OpenHands (CodeAct single-agent scaffold);
  19 models spanning reasoning vs. non-reasoning, 1.5B–671B, function-calling (FC) vs. not.
  An LLM-as-a-judge (Claude Sonnet 3.5, temperature 0, blinded to task outcome) assigns a 0–10
  overthinking score built on three manifestation patterns: Analysis Paralysis, Rogue Actions
  (multiple interdependent actions without awaiting feedback), Premature Disengagement. Score
  validated against 4 human experts on 20 traces (Spearman rho = 0.800). 3,908 trajectories
  generated and scored (abstract cites 4,018 analyzed); dataset + framework open-sourced.
- **Training / RL usage:** None — observational study. Suggests native function calling and
  "selective reinforcement learning" as future fixes (not implemented beyond FC comparison).
- **Experiments & benchmarks:** SWE-bench Verified (issue resolution); regression of resolution
  rate on overthinking score; model-size, token-budget, and context-window analyses; BCFL
  multi-turn check for the FC finding.
- **Key results:** Higher overthinking → lower resolution for all models: reasoning models
  beta1 = −7.894, R² = 0.892, p = 0.000; non-reasoning beta1 = −15.938, R² = 0.839, p = 0.010.
  Reasoning models overthink more (mean score 3.505 ± 1.774 vs. 2.228 ± 0.751). Smaller models
  overthink more (R1-distill family 6.700 ± 1.656 vs. Qwen2.5 5.001 ± 1.732). o1-low shows 35%
  *higher* overthinking than o1-high (2.774 vs. 2.426) — more reasoning budget can *curb* agentic
  overthinking. Economics: o1-high 29.1% resolution at $1,400 vs. o1-low 21.0% at $400; picking
  the lowest-overthinking of k=2 low-effort samples yields 27.3% at $800 (43% cost cut), k=3 gives
  30.3%, beating o1-high while cheaper. Native FC lifts o1 from 29.1% to 47.7% while dropping
  overthinking 2.43 → 1.05.
- **Limitations:** Single benchmark/domain (SWE); LLM-judged score (validated on only 20 traces);
  mitigation is best-of-k selection (still pays k rollouts); no training intervention; preprint.
- **Relation to CASSI:** The flagship motivation citation for agentic overthinking — CASSI's
  problem statement is essentially this paper's "future work" (selective RL to fix overthinking).
  No stopping model, no cost-aware rewards, no training loop. Its Lowest-Overthinking@k is a
  cheap zero-training baseline CASSI's evaluation should acknowledge. THREAT LEVEL: LOW — purely
  observational; it calls for exactly the kind of training signal CASSI builds.

### When More is Less: Understanding Chain-of-Thought Length in LLMs (Wu, Wang, Ye, Du, Jegelka, Yisen Wang, 2025, arXiv 2502.07266; preprint v3 "under review" May 2025)
- **Read from:** PDF pages 1–9 (full main text)
- **Problem:** Is longer CoT always better? Shows task accuracy vs. number of CoT steps follows an
  inverted-U: decomposition helps until per-step simplicity is outweighed by error accumulation
  over more steps.
- **Method/Study design:** Three-pronged: (a) real-world observation — Qwen2.5-Instruct series
  (1.5B–72B) on MATH Level-5, 60 sampled solutions per question with controlled length variation
  (MMLU-STEM in appendix); (b) controlled synthetic arithmetic — GPT-2 models of varying depth on
  T-operator addition problems with explicit CoT length N (per-step size t = T/N); (c) theory —
  accuracy A(N) = alpha[(1−E(N,M,T))(1−sigma(T))]^N (Prop. 4.2), closed-form optimal length
  N*(M,T) = TZ/(M(Z+1)) via Lambert-W (Thm. 4.3), scaling laws (Cor. 4.4), and a bandit argument
  that outcome-reward RL converges to the optimal (usually shorter) CoT length (Cor. 4.5).
- **Training / RL usage:** GRPO on LeetCode-2K with Qwen2.5-7B-Instruct (observing length dynamics);
  PPO (VERL) on the synthetic task; proof-of-concept SFT on optimal-length CoT data; inference-time
  Length-Filtered Vote (bin answers by CoT length, majority-vote over the K=3 lowest-entropy bins).
- **Experiments & benchmarks:** MATH L5, MMLU-STEM, synthetic arithmetic, LeetCode-2K (RL), GPQA
  100-question subset (Length-Filtered Vote).
- **Key results:** Optimal CoT length shrinks with model size: 14 steps (1.5B) → 4 steps (72B).
  Optimal length grows with difficulty: r = 0.57, p = 1.887e-08 (Qwen 1.5B). Optimal-length vs.
  longest-CoT accuracy gap reaches ~40 points for the 72B model. During GRPO/PPO, average CoT
  length *decreases* as accuracy rises ("simplicity bias"); PPO converges to the true optimum
  N* = 5 (96% accuracy) on the synthetic task. A 6-layer model trained on optimal-length CoTs
  beats a 9-layer model trained on mixed lengths. Length-Filtered Vote > vanilla majority vote on
  GPQA.
- **Limitations:** Real-world evidence is math-only, single-turn; synthetic setting is arithmetic
  with uniform step sizes; theory assumes independent per-step errors and linear error scaling;
  no tools, no agents, no dollar costs.
- **Relation to CASSI:** Direct motivation citation (inverted-U) and quantitative support for
  CASSI's H5 — optimal stopping point correlates with difficulty (their r = 0.57 vs. CASSI's
  hypothesized r > 0.5). BUT Corollary 4.5 is a latent threat to CASSI's *motivation framing*:
  it proves plain outcome-reward RL already drifts toward the optimal length, so reviewers may ask
  why a separate stopper/process reward is necessary. CASSI's answers must be: idealized stateless
  bandit ≠ multi-step tool costs; asymptotic convergence ≠ sample efficiency; token length ≠
  multi-dimensional cost. THREAT LEVEL: MEDIUM — supportive empirics, but its RL-suffices
  corollary arms the "single-model GRPO + cost penalty" baseline.

### Over-Reasoning and Redundant Calculation of Large Language Models (Chiang & Lee, 2024, EACL 2024 main conference, short paper; arXiv 2401.11467)
- **Read from:** PDF pages 1–6 (full paper)
- **Problem:** Do LLMs know *when* CoT is unnecessary? Studies redundancy: superfluous content not
  required to answer the question.
- **Method/Study design:** Constructs GSM8K-Zero — 2,978 QA pairs derived from GSM8K by removing
  the original query and asking for a value already stated in the question (answerable with zero
  calculation); ~85% valid per manual audit of 250 pairs. Zero-shot answers from 7 RLHF LLMs;
  redundancy = share of answers containing math operators (regex); accuracy split by presence of
  calculations. Mechanism study: GPT-4/ChatGPT as "proxy reward models" choose between long
  (redundant) and short (answer-only) responses.
- **Training / RL usage:** None (evaluation study); the mechanism analysis targets RLHF reward
  models as the *cause* of verbosity.
- **Experiments & benchmarks:** GSM8K-Zero on GPT-4, ChatGPT, Claude-2, PaLM, Llama-2-chat 7/13/70B.
- **Key results:** Redundancy rates: GPT-4 11.7%, ChatGPT 47.1%, Claude-2 74.7%, PaLM 29.2%,
  Llama-2-70b 80.3%, 13b 88.3%, 7b 88.6%. Accuracy collapses when models calculate: ChatGPT 96.6%
  (no calc) vs. 60.7% (calc); Llama-2-70b 87.7% vs. 46.3%. Telling models they may skip reasoning
  cuts redundancy but it stays >25%. Proxy RMs prefer long answers even when *wrong*: ChatGPT
  consistently prefers lengthy-but-incorrect; GPT-4 picks the short correct answer in only 61% of
  cases and the long wrong one in 34%.
- **Limitations:** Single constructed dataset (with ~15% construction noise); regex redundancy
  metric; 2023-era models; no multi-step/agentic setting; short paper depth.
- **Relation to CASSI:** Earliest peer-reviewed overthinking evidence and the cleanest *mechanism*
  story CASSI can cite: RLHF reward models are length-biased, so "stop now" is never rewarded —
  exactly the missing training signal CASSI supplies. No method overlap. THREAT LEVEL: LOW —
  foundational empirics, no competing machinery.

### Don't Overthink it. Preferring Shorter Thinking Chains for Improved LLM Reasoning (Hassid, Synnaeve, Adi, Schwartz, 2025, arXiv 2505.17813; preprint v2 Feb 2026; Meta FAIR + HUJI)
- **Read from:** PDF pages 1–8 (intro, method, main results, analysis)
- **Problem:** Challenges "longer thinking = better": within an individual question, are shorter
  sampled thinking chains more accurate?
- **Method/Study design:** For 4 open reasoning LLMs (Llama-3.3-Nemotron-Super-49B, R1-Distill-
  Qwen-32B, QwQ-32B, R1-0528-670B) generate 20 responses/question (~36k generations; temp 0.7,
  max 32,768 tokens) on AIME-2024, AIME-2025, HMMT-Feb-2025, GPQA-Diamond; compare
  shortest/random/longest chain per question. Then propose short-m@k inference: launch k parallel
  generations, halt everything when the first m finish thinking, majority-vote over those m (ties →
  shorter). Also finetunes Qwen2.5-7B/32B on S1-dataset variants (S1-short/S1-long/S1-random).
- **Training / RL usage:** SFT only (S1 variants); no RL. short-m@k is training-free.
- **Experiments & benchmarks:** Above 4 benchmarks; metrics: accuracy vs. sample budget k, vs.
  thinking-token compute, vs. wall-time; backtrack analysis via keyword counting.
- **Key results:** Shortest-of-20 beats random by +2.2 to +15.7 points (math average) and beats
  longest by up to +34.5 points, while using ~50%/67% fewer tokens than random/longest. E.g.
  LN-Super-49B math avg: shortest 63.4% (7,083 tokens) vs. random 47.7% (12,270) vs. longest 28.9%
  (19,098). R1-670B: 83.3% vs. 78.2% vs. 58.9%. short-1@k matches/beats majority@k at low compute
  with up to 40% fewer thinking tokens; short-3@k dominates majority voting across budgets and is
  up to 33% faster (R1-32B, k=5); LN-Super k=5: ~50% time cut with +1.5 points. SFT on S1-short
  improves accuracy AND shortens thinking; S1-long adds cost without gains. Nuance: harder
  questions still consume up to 2.9× more tokens — between-question scaling persists; correct
  trajectories contain ~2.5× fewer backtracks than incorrect ones.
- **Limitations:** Math/GPQA single-turn only; needs k parallel samples (compute floor); length
  is a post-hoc selector, not a learned per-step stopping decision; preprint.
- **Relation to CASSI:** Canonical "shortest chains are more accurate" motivation citation. The
  2.9×-more-tokens-on-hard-questions nuance is the best argument *for* CASSI's per-instance
  adaptive stopping (vs. global caps). short-m@k is a strong zero-training efficiency baseline
  in the CoT domain, conceptually a competitor to "stopper as controller" for single-model
  reasoning, but it has no per-step value estimate, no cost-awareness beyond length, no training
  signal for the executor, and does not apply to sequential agent trajectories (can't parallel-
  sample a stateful environment cheaply). THREAT LEVEL: LOW-MEDIUM — baseline pressure only.

### Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs (Chen et al., 2024/2025, arXiv 2412.21187 v2; preprint — no venue confirmed; Tencent AI Lab + SJTU)
- **Read from:** PDF pages 1–9 (observation study + mitigation methods)
- **Problem:** First comprehensive study of overthinking in o1-like LRMs: excessive compute on
  problems where it adds minimal benefit.
- **Method/Study design:** Decompose responses into "solution rounds" (Llama-3.3-70B used as
  splitter); measure solution-count distributions, first-correctness distribution, and diversity
  (GPT-4o clusters solutions by reasoning strategy). Two novel metrics: outcome efficiency
  xi_O = mean fraction of tokens up to the first correct answer; process efficiency xi_P = mean
  fraction of tokens in *distinct* solutions. Then mitigation: self-training to prefer efficient
  responses.
- **Training / RL usage:** Self-training on PRM12K prompts with QwQ-32B-Preview: SFT, DPO, RPO,
  SimPO using shortest-of-10 sampled correct responses as positives (longest as negatives), plus
  response "streamlining": FCS (keep only solutions up to the First-Correct Solution), FCS+
  Reflection (keep second correct round too), GDS (greedily keep only strategy-diverse solutions).
  No online RL.
- **Experiments & benchmarks:** ASDIV (2,305), GSM8K (1,319), MATH500 observation; training
  evaluated across GSM8K/MATH500/GPQA/AIME difficulty spectrum; models QwQ-32B-Preview,
  DeepSeek-R1, vs. conventional Llama-3.3-70B, Qwen2.5-Math-72B.
- **Key results:** o1-like models use ~1,953% more tokens than conventional LLMs on "2+3=?" (QwQ:
  901 tokens, 13 solution rounds). They generate MORE rounds on EASIER sets (QwQ 3.5 avg on ASDIV
  vs. 3.2 on MATH500; R1 4.5 vs. 4.3). In >92% of correct cases the first round is already correct;
  the first round is <60% of tokens (287 tokens = 38.7% of the response for QwQ/ASDIV). Later
  rounds mostly repeat earlier strategies (distinctness falls with index). Efficiency: QwQ MATH500
  acc 93.0% with xi_O = 52.3%, xi_P = 71.2%; R1 96.4% with xi_O = 51.0%, xi_P = 66.2%; on MATH500
  Level-1 both models' xi_O < 50% (46.4% / 42.4%), rising to ~57.5% at Level-5 — overthinking is
  worst on the easiest problems. Mitigation: −48.6% output tokens on MATH500 while maintaining
  accuracy (Fig. 1b); FCS positives reach xi_O = 99.5%.
- **Limitations:** Math-only, single-turn; xi_O requires ground truth (post-hoc metric); offline
  preference tuning (not per-instance adaptive at inference); "solution round" segmentation and
  strategy clustering rely on LLMs; preprint.
- **Relation to CASSI:** The closest conceptual precursor in this area to CASSI's *oracle
  labeling*: FCS truncates a trajectory at the earliest point where quality is achieved — a
  hindsight quality-based stopping label used to build training data, i.e., a special case of
  CASSI's t* = argmax[quality − lambda·cumcost] with lambda→implicit and cost = tokens. Also
  supplies xi_O/xi_P metrics CASSI can reuse for measuring post-t* waste. Differences: single
  model self-training, no explicit cost term or budget tiers, no learned stopping/value model, no
  process-reward bridge, no agents/tools. THREAT LEVEL: MEDIUM — hindsight earliest-sufficient-
  point relabeling already exists for CoT; CASSI's oracle novelty must rest on the explicit
  quality-minus-lambda-cost trade-off, agentic multi-dimensional costs, and the stopper-as-PRM loop.

### Inverse Scaling in Test-Time Compute (Gema et al., 2025, TMLR 12/2025 — peer-reviewed; arXiv 2507.14417; Anthropic Fellows Program et al.)
- **Read from:** PDF pages 1–9 (setup + main task results)
- **Problem:** Constructs tasks where extending LRM reasoning length *decreases* accuracy —
  inverse scaling between test-time compute and performance, beyond mere cost waste.
- **Method/Study design:** Four task categories: (1) simple counting with distractors — Misleading
  Math / Misleading Python (2,500 questions each, 500 per distractor count n ∈ {1..5}; answer is
  always "2"), plus a Famous-Paradoxes framing variant (812 questions); (2) regression with
  spurious features (Grades Regression from a Kaggle lifestyle dataset, 500 students; 0/8/16-shot);
  (3) deduction with constraint tracking (Zebra Puzzles); (4) model-written safety evals. Two
  scaling protocols: *controlled overthinking* (prompted reasoning budgets 0–16,384 tokens; o-series
  low/medium/high) and *natural overthinking* (sample 5 responses, rank by length); 3–5 repetitions.
- **Training / RL usage:** None — pure evaluation of 9 frontier models (Claude Sonnet 3.7/4,
  Opus 4, o3-mini, o4-mini, o3, Qwen3-32B, QwQ-32B, DeepSeek R1).
- **Experiments & benchmarks:** As above; also checks standard benchmarks (GSM8K, ASDIV,
  MultiArith, GSM-IC) — which show *minimal* inverse scaling, motivating the new suite.
- **Key results:** Five failure modes: (1) Claude models get increasingly distracted by irrelevant
  content as they reason longer — Opus 4 drops from ~100% to ~85–90% on Misleading Math;
  (2) o-series resist distractors but overfit to familiar problem framings (recognize "Birthday
  Paradox" and apply memorized machinery to a trivial question); (3) with longer reasoning, models
  shift from reasonable priors to spurious features in regression (few-shot examples largely fix
  it); (4) all models degrade on Zebra constraint tracking with extended reasoning; (5) extended
  reasoning amplifies safety-relevant behaviors (Claude Sonnet 4 expresses more self-preservation).
  DeepSeek R1 falls 70% → 30% on Misleading Math with 5 distractors (natural setup). Inverse
  scaling requires >2% accuracy change with non-overlapping CIs (their trend criterion).
- **Limitations:** Deliberately constructed tasks (distractors/spurious features) rather than
  organic distributions; sequential scaling only; single-turn, no tools or agents; model API
  budget knobs are coarse.
- **Relation to CASSI:** Upgrades CASSI's motivation from "longer costs more" to "longer can be
  *wrong and unsafe*", and its per-model heterogeneity of failure modes argues for a learned,
  trajectory-conditioned stopping signal instead of fixed budgets. No method overlap. THREAT
  LEVEL: LOW — evaluation-only; strengthens rather than competes.

### Reasoning in Token Economies: Budget-Aware Evaluation of LLM Reasoning Strategies (Junlin Wang et al., 2024, EMNLP 2024 main, pp. 19916–19939; arXiv 2406.06461)
- **Read from:** PDF pages 1–8 (framework + main results + self-evaluation analysis)
- **Problem:** Reasoning-strategy papers compare methods at *unequal compute*; apparent algorithmic
  gains may just be budget gains. (This is the paper the CASSI plan cites as "Token Economies,
  Wang et al., EMNLP 2024" — verified to exist and to be peer-reviewed.)
- **Method/Study design:** Budget-aware evaluation framework with three budget axes: API monetary
  cost (c = alpha1·n_I + alpha2·n_O), total tokens, number of queries. Re-evaluates 7 strategies —
  CoT + Self-Consistency (SC), Multi-Agent Debate (MAD), Reflexion, Plan-and-Solve, Least-to-Most,
  Progressive Hints, Tree-of-Thoughts — at *matched* budgets; ablates ToT/Reflexion into proposer
  vs. evaluator budgets; probes LLM self-evaluation quality (Yes/No, 1–10 score formats;
  oracle-evaluator counterfactuals).
- **Training / RL usage:** None — prompting-strategy evaluation (GPT-3.5, GPT-4, Mistral-7B, more
  in appendix).
- **Experiments & benchmarks:** GSM8K, MATH, TheoremQA, CSQA, HotpotQA (+ Game of 24); budgets up
  to 20 queries / 10k tokens per question.
- **Key results:** Budget-matched CoT-SC matches or beats MAD, Reflexion, and most "sophisticated"
  strategies on nearly all datasets; MAD's answer-entropy (diversity) declines every round —
  dependent sampling tunnels onto wrong answers; SC helps only when per-sample accuracy > 0.5 and
  *hurts* monotonically when < 0.5. ToT pays off only with strong models: GPT-4-proposer +
  GPT-3.5-evaluator achieves 72% on Game of 24 at $33.53 vs. 76% at $159.87 with a GPT-4 evaluator
  (~5× cost for +4 points). Self-evaluation: Reflexion with an *oracle* evaluator beats SC by a
  large margin, but Reflexion with a GPT-4 evaluator *underperforms* SC — LLM self-evaluation is
  currently too weak to steer compute, though it is the budget-efficient lever.
- **Limitations:** Prompting-era methods and models (GPT-3.5/4); no trained evaluators; mostly
  single-turn reasoning datasets; monetary costs tied to 2023/24 API pricing.
- **Relation to CASSI:** Methodological anchor for CASSI's evaluation protocol (iso-cost accuracy,
  iso-accuracy cost, cost-matched baselines are exactly this paper's doctrine) and strong evidence
  for CASSI's premise that *prompted* self-evaluation is a weak stopping signal while an accurate
  evaluator would unlock large gains — the gap CASSI's trained stopper targets. THREAT LEVEL:
  LOW — evaluation framework, no learned stopping.

### Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models (Sui et al., 2025, TMLR 2025 — accepted; arXiv 2503.16419 v4)
- **Read from:** PDF pages 1–4 (framing + full taxonomy tree) + curated repo description
- **Problem:** First structured survey of "efficient reasoning" — mitigating the overthinking
  phenomenon in LRMs.
- **Method/Study design:** Taxonomy with three axes: (1) *model-based* — RL with length-reward
  design (≈40 methods listed incl. L1/LCPO, O1-Pruner, Kimi k1.5, DAST, ThinkPrune, ConciseRL,
  Elastic Reasoning, SelfBudgeter…), SFT with variable-length CoT (TokenSkip, C3oT, CoT-Valve,
  Self-Training/FCS…); (2) *reasoning-output-based* — latent compression (Coconut, CODI, CCoT,
  SoftCoT) and dynamic inference paradigms (speculative/early-exit/sleep-time compute, DEER-like
  exits); (3) *input-prompts-based* — prompt-guided length control (Token-Budget, Chain-of-Draft,
  NoThinking) and difficulty routing (RouteLLM, ThinkSwitcher, SwitchCoT). Plus: efficient training
  data (LIMO, s1), small-model reasoning/distillation, and evaluation/benchmarks (cites Cuadron's
  "Danger", Sys2Bench, S1-Bench).
- **Training / RL usage:** Survey — documents that the dominant training recipe for efficiency is
  RL with length-penalized rewards or SFT on compressed/variable-length CoT.
- **Experiments & benchmarks:** N/A (survey; maintains a living GitHub repo).
- **Key results:** The single-model, single-turn length-control space is extremely crowded (dozens
  of RL-length-reward papers by mid-2025); routing-by-difficulty exists as its own category;
  benchmarks for efficiency evaluation are emerging. Agentic/multi-step overthinking is covered
  only via evaluation citations — no training-loop methods for agent stopping appear anywhere in
  the taxonomy.
- **Limitations:** Snapshot (v4 Aug 2025) of a fast-moving field; taxonomy is method-oriented with
  little quantitative meta-analysis; thin on agents, tools, and dollar-cost budgets.
- **Relation to CASSI:** The map reviewers will consult. It confirms a defensible gap (no
  hindsight-labeled stopping model used as a cost-aware PRM for executor training; nothing
  agentic), but it also shows CASSI's "static penalty" characterization of the RL-length-reward
  family is risky: the bucket already contains difficulty-adaptive members (DAST, AdaCoT-style,
  budget-aware tuning). THREAT LEVEL: MEDIUM — not a competing method, but it documents ~40
  single-model foils CASSI's related-work and ablations must cleanly beat or distinguish.

### Redundant or Necessary? A Benchmark for Detecting Redundant Steps in Agent Trajectories (Hu, Yang, Zhou, Liang, Guo, Yin, Han, 2026, arXiv 2605.29893; preprint May 2026; Huawei Noah's Ark Lab + ICT CAS)
- **Read from:** PDF pages 1–8 (full paper)
- **Problem:** Proposes a new research problem: *redundant step detection* in agent trajectories —
  steps that consume resources while contributing little to task completion. (The strongest 2026
  empirical grounding for agent-step waste found in this search.)
- **Method/Study design:** Formal counterfactual definition: for a successful trajectory tau, step
  t is redundant iff removing it leaves the task still successful (Delta_t(tau) = 1 iff Z(tau)=1
  and Z(tau^t)=1). RedundancyBench: 200 successful trajectories (>8,000 steps) generated by
  Qwen-3.6-Plus on tau^2-bench (retail/telecom/airline tool-use domains; 278 collected, failures
  filtered), plus synthetic redundant-step injection; 4 redundancy types (Abnormal, Duplicated,
  Incorrect, Exploratory); 3-round annotation by 6 experts (heuristic pre-labeling → expert
  revision with confidence → cross-validation), ~1 hour per trajectory. Detection methods: LLM
  judges with One-to-One (single step), Window-to-One (k=3 context), All-to-All (full trajectory)
  receptive fields.
- **Training / RL usage:** None — benchmark + prompted-LLM detection study (GPT-4o, GPT-5.4,
  DeepSeek-V4-Pro).
- **Experiments & benchmarks:** Trajectory-level accuracy (does the trajectory contain redundancy)
  and step-level average F1; length-level analysis (1–18 / 19–30 / 31–49 / 50–110 steps);
  with/without ground-truth action sequences.
- **Key results:** Step-level redundancy detection is *hard for prompted LLMs*: best method
  (DeepSeek-V4-Pro, Window-to-One) reaches only 24.88% step-level F1; worst (GPT-5.4, One-to-One)
  4.25%; One-to-One trajectory-level detection (43.17%) is *below random guessing* (~50%).
  Trajectory-level best is 70.88% (GPT-5.4, Window-to-One). Context exhibits an inverted-U:
  full-trajectory context *hurts* vs. windowed (attention dilution) — e.g. GPT-4o 64.81/20.49
  (window) vs. 47.32/13.42 (all-to-all). Ground-truth reference actions lift GPT-4o All-to-All
  step F1 from 12.54 to 24.37 — still far from reliable. Failure attributed to LLMs not modeling
  causal dependencies between steps.
- **Limitations:** One source benchmark (tau^2-bench), one generator model, 200 trajectories;
  detection-only (no efficient-agent training); annotation cost limits scale; preprint.
- **Relation to CASSI:** Highly synergistic 2026 evidence: (a) legitimizes agent-step waste as a
  named problem; (b) empirically demonstrates that prompted LLM judges cannot do step-level
  economic judgment (24.88% F1) — the sharpest available justification for CASSI's *trained*
  stopping model over zero-training self-eval baselines; (c) offers an external validation set for
  CASSI's stopper; (d) their counterfactual step-removal label needs re-execution/annotation per
  step, whereas CASSI's suffix-truncation oracle t* is computed post-hoc in O(T) with zero extra
  rollouts — a concrete efficiency contrast CASSI can draw. Differences: it labels *interior*
  step redundancy, not a stopping point; no cost-quality trade-off (lambda); no training. THREAT
  LEVEL: LOW-MEDIUM — problem-space overlap without method overlap; reviewers may expect CASSI to
  cite it and possibly evaluate against it.

## Peripheral papers

**Harnessing the Reasoning Economy: A Survey of Efficient Reasoning for LLMs (Rui Wang et al., 2025, arXiv 2503.24377; preprint "work in progress", Mar 2025; CUHK et al.)** — *Read from PDF pages 1–4.* The second seed survey; frames System-1/System-2 "reasoning economy" (benefits vs. budgets) across post-training and test-time. Its challenge taxonomy separates *inefficient model behaviors from post-training* (length bias — citing Cuadron; "fake thinking"/deceptive behaviors) from *inefficient test-time usage* (unreasonable algorithm selection and computation allocation), and its solution tree spans long2short RL, quality-length disentanglement, procedure rewards, adaptive budget-aware tuning, single-model routing, and output-side early stopping. Two details matter for CASSI: it explicitly notes PRM training data is expensive ("extensive human annotations or a large amount of sampling"), supporting CASSI's O(T)-oracle-vs-Monte-Carlo efficiency claim; and like Sui et al. it contains no agentic stopping-model training loop. Threat LOW as a survey; same "static penalty strawman" caution as Sui.

**Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs (Wang et al., 2025, arXiv 2501.18585; preprint v2 Feb 2025)** — *Abstract + update notes.* The dual failure mode: on hard problems (e.g., AIME-level math) o1-like models frequently *switch* reasoning thoughts prematurely instead of exploring promising paths deeply; frequent thought-switching correlates with incorrect responses; proposes a token-efficiency-in-incorrect-answers metric and TIP, a thought-switching decoding penalty that improves accuracy without retraining (results updated to hold on DeepSeek-R1). Relevance: CASSI's stopper must avoid inducing underthinking — premature-stop errors are as real as overthinking; cite for the two-sided error model and the |t_stop − t*| metric's symmetry. Threat LOW.

**OptimalThinkingBench: Evaluating Over and Underthinking in LLMs (2025, arXiv 2508.13141; preprint v2 Oct 2025; Meta)** — *Abstract.* Unified benchmark: OverthinkingBench (simple math + general queries across 72 domains) and UnderthinkingBench (11 challenging reasoning tasks), scored with thinking-adjusted accuracy metrics over 33 thinking/non-thinking models. Headline: *no* model thinks optimally; thinking models burn hundreds of tokens on trivial queries without accuracy gains. Relevance: ready-made single-turn evaluation target for CASSI's controller and quantitative backing for "no model has economic judgment". Threat LOW.

**Between Underthinking and Overthinking: An Empirical Study of Reasoning Length and Correctness in LLMs (2025, arXiv 2505.00127; preprint)** — *Abstract.* Systematic study finding LLMs overthink easy problems and underthink hard ones — i.e., models misjudge difficulty and miscalibrate response length; also shows naive preference optimization toward shorter responses (ignoring correctness) cuts generation length substantially with acceptable accuracy. Relevance: difficulty-miscalibration evidence supports CASSI's per-instance adaptive stopping (H5); the naive-shortness-DPO result is a cautionary baseline (length can be cut without any stopping intelligence). Threat LOW.

**The Impact of Reasoning Step Length on Large Language Models (Jin et al., 2024, Findings of ACL 2024; arXiv 2401.04925)** — *Abstract.* The main *counterpoint* in this area: lengthening reasoning steps in few-shot CoT *demonstrations* — even without adding information — considerably improves accuracy across datasets, and compressing demonstration steps hurts. Reconciliation with the overthinking literature: this is prompt-side demonstration length for pre-LRM models, not model-generated thinking length at inference; together with Wu et al. it locates both rising and falling branches of the inverted-U. CASSI should cite it to preempt "longer is better" objections with the regime distinction. Threat LOW.

**Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs (2025, arXiv 2507.02076; preprint)** — *Abstract.* Note on seed disambiguation: the CASSI plan's citation "Reasoning on a Budget / Token Economies (Wang et al., EMNLP 2024)" conflates two distinct papers — the EMNLP 2024 Token Economies paper (core, above) and this separate July-2025 survey. The survey taxonomizes test-time compute methods into L1-*controllability* (fixed compute budgets) vs. L2-*adaptiveness* (scaling with input difficulty/confidence), benchmarks proprietary LLMs on the performance-token trade-off, and repeats the overthink-easy/underthink-hard diagnosis. Relevance: its L1/L2 axis is a clean vocabulary for positioning CASSI as L2-adaptive with a learned controller; its existence again shows adaptive-budget *taxonomies* exist while trained agentic stopper-as-PRM loops do not. Threat LOW.

**Agent-Omit: Adaptive Context Omission for Efficient LLM Agents (2026, arXiv 2602.04284; ICML 2026 — peer-reviewed)** — *Abstract.* The most method-relevant 2026 neighbor found: quantitative investigation showing thought necessity and observation utility *vary across turns* in multi-turn agents, then a unified training framework (cold-start SFT on synthesized single/multi-turn omission data + "omit-aware" agentic RL with a dual sampling mechanism and a tailored omission reward, with a KL-bounded deviation guarantee) that teaches one agent to adaptively skip redundant thoughts/observations. Agent-Omit-8B matches seven frontier agents on five agent benchmarks with the best effectiveness-efficiency trade-off. Relevance/threat: this is *single-model adaptive efficiency RL for agents* published at ICML 2026 — it crowds CASSI's "agents lack economic judgment" territory and must be engaged in related work; however, it optimizes per-turn thought/context omission, not episode-level stopping; it has no separate small value model, no hindsight stopping oracle, and no stopper-derived process reward for training. THREAT MEDIUM — nearest 2026 agentic-efficiency-training neighbor in this area.

## Synthesis

**Landscape & consensus.** (1) Overthinking is real, measurable, and worst on easy instances:
redundancy 47–89% on zero-computation questions (Chiang & Lee); ~1,953% token overhead on trivial
arithmetic with outcome efficiency below 50% on the easiest MATH500 level (Chen et al.); thinking
models burn hundreds of tokens on trivial queries with no gain (OptimalThinkingBench). (2) Length
and accuracy follow an inverted-U per instance: within a question, shortest sampled chains beat
longest by up to +34.5 points (Hassid), and optimal length exists in closed form, growing with
difficulty (r = 0.57) and shrinking with capability (Wu). (3) The harm is not only cost: longer
reasoning can *reduce* accuracy via distraction, framing overfit, and spurious-feature
amplification (Gema, TMLR), and in agents overthinking strongly predicts task failure
(R² = 0.892, Cuadron). (4) The dual failure — underthinking/premature switching — is equally
documented (2501.18585; OptimalThinkingBench), so global truncation is not a solution; adaptivity
is the consensus requirement. (5) Mechanistically, RLHF reward models prefer long answers even
when wrong (Chiang & Lee), while outcome-reward RL exhibits a simplicity bias that only slowly
self-corrects (Wu). (6) Prompted LLM self-evaluation is too weak to steer compute: oracle
evaluators would beat SC decisively but GPT-4 evaluators do not (Token Economies), and step-level
redundancy detection tops out at 24.88% F1 (RedundancyBench, 2026).

**Gaps.** Nearly all quantitative evidence is single-turn math/QA; the agentic evidence base is
three papers (Cuadron — SWE; RedundancyBench — tau^2-bench; LaRMA-style analyses) with token/step
counts but no dollar-denominated, multi-dimensional cost accounting; nobody measures stopping
quality against a hindsight optimum (|t_stop − t*|); no work in this area trains *any* dedicated
stopping/value model, and none connects an economic signal to executor training as a process
reward. The empirics literature diagnoses; it does not treat.

**Top threats to CASSI's novelty (ranked, from this area only).**
1. **Chen et al. 2412.21187 (FCS self-training) — MEDIUM.** Hindsight "truncate at first-correct"
   labels used to build efficiency training data already exist; CASSI's oracle must be positioned
   as its cost-aware, agentic, lambda-parameterized generalization feeding a *separate* model.
2. **Agent-Omit (ICML 2026) — MEDIUM.** Single-model omit-aware agentic RL with an omission reward
   shows "train agents to be economical" is already publishable territory; CASSI must show the
   two-model loop beats single-model adaptive-efficiency RL (its planned single-vs-two-model
   ablation is the right instrument).
3. **Wu et al. Corollary 4.5 (RL converges to optimal length) — MEDIUM.** Arms the "plain GRPO
   with outcome/cost reward suffices" objection against the necessity of a stopper-as-PRM;
   CASSI needs sample-efficiency and multi-dimensional-cost arguments plus the planned
   single-model-GRPO baseline.
4. **Survey taxonomies (Sui TMLR 2025; Reasoning Economy; Reasoning on a Budget) — MEDIUM.**
   Dozens of length-control methods, some difficulty-adaptive (DAST, adaptive budget-aware
   tuning, routing) — the "prior penalties are static" framing must be softened to "prior
   penalties are single-model, single-turn, and not trajectory-state-conditioned".
5. **Zero-training selection baselines (Hassid short-m@k; Cuadron Lowest-Overthinking@k) — LOW-
   MEDIUM.** Cheap inference-time competitors on the cost-accuracy Pareto frontier that reviewers
   will demand as baselines.

**Opportunities CASSI could exploit.** (a) Reuse established metrics as motivation and evaluation:
outcome/process efficiency xi_O/xi_P (Chen) to quantify post-t* waste, the overthinking score
(Cuadron, open-sourced) for behavioral validation, thinking-adjusted accuracy
(OptimalThinkingBench), and budget-matched evaluation doctrine (Token Economies) which directly
sanctions CASSI's iso-cost/iso-accuracy protocol. (b) Use RedundancyBench as an external test of
the stopper's step-level economic judgment — beating 24.88% F1 with a 0.5–3B trained stopper would
be a headline result, and the contrast "counterfactual re-execution labels vs. O(T) post-hoc
oracle" sharpens CASSI's efficiency claim. (c) Ground H5 in Wu's r = 0.57 difficulty-length
correlation and Hassid's 2.9× hard-question token ratio — per-instance adaptivity is exactly what
the empirics prescribe. (d) Cite Gema (TMLR) to claim stopping improves *accuracy and safety*,
not just cost. (e) Exploit the documented weakness of prompted self-evaluation (Token Economies;
RedundancyBench) as the empirical basis for the "representation conflict / separate stopper"
contribution — the strongest empirically-backed part of CASSI's story.
