# Learned Stopping, Self-Termination & Early Exit

> Area 03 of the CASSI literature review. Researched 2026-07-16. Sources: arXiv API, Semantic
> Scholar, web search; core papers read from PDFs in `research/papers/`. Focus questions:
> (a) does any work train a SEPARATE model to decide stopping? (b) does any work use stopping
> decisions as TRAINING rewards for the main policy?

## Area overview

"When to stop" has gone from a niche heuristic question (2023–24: early-stopping self-consistency,
adaptive retrieval gating) to a crowded 2025–26 subfield with three distinct waves. **Wave 1 —
training-free signals (2024–mid-2025):** monitor the model's own behavior and stop on a threshold:
answer-consistency probes (Certaindex/Dynasor), trial-answer confidence at reasoning transition
points (DEER), answer convergence, entropy monitors. These are same-model, heuristic, inference-only,
and quality-only (cost appears only as the thing being saved, never inside a decision objective).
**Wave 2 — learned stoppers (mid-2025–2026):** train a lightweight separate module — hidden-state
probes (Reasoning-Models-Know-When-They're-Right, Thought Calibration, LYNX, TERMINATOR), feature
classifiers (LearnStop), or even a small LM (CaRT's termination model, FlashThink's verifier LLM) —
on labels computed post hoc from completed trajectories (forced/hindsight exits). By 2026 this wave
has its own theory (OS-Pruner's optimal-stopping formulation with an explicit `accuracy − λ·tokens`
utility and a proof that threshold classifiers can be arbitrarily suboptimal), its own statistical
tooling (conformal / risk-controlled stopping: LYNX, MARS, LearnStop), and even its own meta-study
asking when learning to stop beats calibrated scalar exits (Dong et al. 2026).

**Wave 3 — stopping enters training (late 2025–2026):** the newest and thinnest wave moves stopping
signals from inference into the policy's training loss. Same-model variants train the policy to
choose thinking modes (Thinkless, AdaptThink) or shape per-segment advantages by whether the model
has already passed a correct intermediate answer (DASH). Agentic variants relabel stop/continue
decisions post hoc via causal interventions and align the search agent with DPO (DAS, WWW 2026), or
RL-train explore/commit behavior under cost-discounted rewards with calibrated priors injected
(Calibrate-Then-Act). Crucially for CASSI: every component of its pipeline now exists somewhere in
this literature — post-hoc oracle stop labels (TERMINATOR, LYNX, OS-Pruner, CaRT), a separate small
stopper supervising a frozen executor (CaRT-medical, TERMINATOR, SWE-PRM), a cost-aware stopping
value (OS-Pruner), and stop-derived training signal for the executor (DASH, DAS, SAGE-RL) — but **no
paper combines them**: nobody trains a separate cost-aware stopping model and then uses its value
estimate as a process reward to RL-train the executor, let alone iterates that loop. The gap is
real but compositional, and several 2026 papers are one step away from it.

## Core papers

### CaRT: Teaching LLM Agents to Know When They Know Enough (Grace Liu, Yuxiao Qu, Jeff Schneider, Aarti Singh, Aviral Kumar; 2025; arXiv 2510.08517, CMU, preprint)

- **Read from:** PDF pages 1–11 (`research/papers/2510.08517_cart-know-when-enough.pdf`)
- **Problem:** Strategic information gathering requires knowing not just how to acquire information
  but *when to stop* and commit; off-the-shelf LLMs badly mis-time termination in multi-turn
  information seeking (medical diagnosis) and long CoT (math). Objective formalized as
  `max E[Σ_t γ^t 1{a_t=terminate}·r(x,y_t)]` — cost enters only as an implicit discount γ.
- **Method:** WHO decides — in the medical domain, an explicitly **separate termination model**
  (Qwen2.5-3B) watches a *frozen* question-asking model; in math, the same model decides at episode
  boundaries. Trained, not heuristic: each conversation/reasoning prefix is labeled post hoc with
  external success rate (Llama-3.1-8B judges diagnostic accuracy per prefix; math prefixes labeled
  "terminate" if stopping early yields higher success than continuing). Two ingredients: (1) hard
  negative **counterfactual pairs** — minimally perturb a correct-termination trajectory (swap the
  final QA pair so success drops <30%, or delete a span of reasoning steps) so termination becomes
  wrong; (2) GPT-4o-generated verbal rationales for each decision ("verbalized value function").
  SFT on these pairs; inference-only control of the executor.
- **Training / RL usage:** SFT is the main method; a CaRT+RL variant runs GRPO *on the termination
  decision itself* (binary reward: +1 for correctly terminating when success ≥0.5 / continuing when
  <0.5, −1 otherwise). RL "tends towards longer conversations" and gave no gains in math. The
  stopping decisions are **never used as rewards to train the executor** — the question-asker stays
  frozen.
- **Experiments & benchmarks:** Interactive medical diagnosis built from craft-MD (MedQA-USMLE +
  MedMCQA, 1,233 problems, GPT-4o-simulated dialogues; 100 ID + 200 OOD dermatology test); math on
  2,000 DeepScaleR problems with Qwen3-1.7B, evaluated on AIME 2025. Metrics: FRQ success rate,
  success-rate difference vs. fixed-mean-index baseline, optimal termination rate.
- **Key results:** Medical: optimal termination rate ≈0.32 (CaRT) vs ≈0.17 (SFT) vs ≈0.04 (base);
  best FRQ-SR-vs-mean difference (≈+0.027); holds OOD. Math (AIME25): higher success (~33 vs ~32.5)
  with ~14.5K vs ~18K response tokens; optimal termination rate ≈0.37 vs ≈0.20 base. Ablations:
  counterfactuals matter most; rationales improve generalization (probe LR test acc 0.774 vs 0.645
  without); an auxiliary verbalized-confidence head adds nothing on top of full CaRT.
- **Limitations:** No explicit cost in labels or objective (γ only); no per-instance budget/λ
  adaptation; termination model never trains the executor; RL variant unhelpful; small models;
  two domains only; no tool-cost accounting.
- **Relation to CASSI:** The closest published *training-based termination* work and CASSI's
  declared primary baseline; it already has post-hoc per-prefix quality labels, a separate stopper
  watching a frozen executor, and counterfactual data. CASSI's deltas: cost-aware oracle
  (quality − λ·cumcost), continuous Δ value, budget conditioning, and the process-reward bridge
  into executor GRPO. THREAT LEVEL: **HIGH** — reviewers will ask "CaRT + cost term + your GRPO
  bridge = CASSI?"; the plan's CaRT+cost+GRPO baseline is mandatory.

### OS-Pruner: Pruning Chains-of-Thought of Reasoning Models via Optimal Stopping (Mohammed Ehab, Aymane El Gadarri, Vivek Farias, Adam Jozefiak, Ciamac C. Moallemi; 2026; arXiv 2607.11089, MIT/Columbia, preprint, July 2026)

- **Read from:** PDF pages 1–5 (`research/papers/2607.11089_os-pruner-optimal-stopping.pdf`)
- **Problem:** Existing early exits reduce stopping to correctness classification with a manual
  threshold; CoT pruning is "more naturally a sequential decision problem": continue only while
  expected improvement outweighs token cost.
- **Method:** WHO decides — a **separate lightweight stopping policy** attached to the frozen base
  LRM: a linear head on the last hidden state plus light fine-tuning of the last n=2 self-attention
  layers, invoked at paragraph boundaries. Trained, not heuristic. Stopping reward is **explicitly
  cost-aware**: `r(y_≤i|x) = A(y_≤i|x) − λ·L(y_≤i)` (per-prefix accuracy minus λ × tokens) — the
  same functional form as CASSI's oracle `quality_t − λ·cumcost_t`. Labels: force a final answer
  after *every* reasoning step of frozen-model traces (SGLang fork), grade with Math-Verify; because
  the base model is frozen the per-prefix rewards are precomputed once and the stopping objective's
  expectation is optimized exactly — "no repeated on-policy rollouts". λ is a user knob sweeping the
  accuracy–length frontier. Includes a value-function/Bellman view — stop when
  `A(y_≤i|x) ≥ C_λ(x,y_≤i)` (continuation value) — and **Theorem 1**: for any λ and any K>0 there
  is a stopping problem where the optimal policy beats the *best* fixed-threshold correctness
  classifier by factor K, even with A known exactly.
- **Training / RL usage:** trains only the stopping policy (exact-expectation optimization of the
  stopping objective); the reasoning model is frozen; stopping is never a reward for the generator.
- **Experiments & benchmarks:** math reasoning benchmarks over multiple base reasoning models
  (details in later pages; headline results below); baselines include budget forcing/guidance,
  HALT-CoT, DEER, answer-convergence classifiers, FlashThink.
- **Key results:** 20–60% generation-length reduction with minimal accuracy sacrifice across
  benchmarks and base models; still yields gains stacked on top of a brevity-fine-tuned model;
  single scalar λ gives fine-grained frontier control.
- **Limitations:** Single-model math CoT only; token cost only (no tools/dollars/multi-dim budget);
  inference-time plug-in — never feeds back into generator training; tied to the base model's
  hidden states (not executor-agnostic).
- **Relation to CASSI:** The **same stopping objective and much of the theory**: quality−λ·cost
  argmax labels, value-margin stopping (their A vs C_λ ≈ CASSI's Δ = Q_continue − Q_stop),
  λ-frontier sweeps, and a suboptimality theorem for threshold stoppers that CASSI's "formal
  properties" section would partially duplicate. CASSI's O(T)-labeling claim is also mirrored
  (precomputed per-prefix rewards, no extra on-policy rollouts). What OS-Pruner lacks: agents,
  multi-dimensional cost, any executor training, the closed loop. THREAT LEVEL: **HIGH** — it
  occupies the cost-aware-optimal-stopping formulation ground; CASSI's theory contributions must be
  positioned as extensions to agent trajectories + the reward bridge, and OS-Pruner's Theorem 1
  should be cited *in support* of value-based over classifier stopping.

### TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning (Alliot Nagle, Jakhongir Saydaliev, Dhia Garbaya, Michael Gastpar, Ashok Vardhan Makkuva, Hyeji Kim; 2026; arXiv 2603.12529 v2 (May 2026), UT Austin/EPFL, preprint; on OpenReview)

- **Read from:** PDF pages 1–7 (`research/papers/2603.12529_terminator-optimal-exit.pdf`)
- **Problem:** LRMs keep reasoning long after their final answer has logically arrived; optimal CoT
  lengths are task- and model-dependent, so fixed budgets and calibrated thresholds are brittle.
- **Method:** Defines **hindsight-optimal reasoning length (HORL)**: the earliest position in a
  *completed* CoT at which the model's own final answer â has first logically arrived — a purely
  retrospective, post-hoc label needing no ground truth. An LLM-based Extract–Identify–Verify
  pipeline (Qwen3-30B-A3B) locates â's earliest span reliably at scale. WHO decides — a **separate
  binary probe classifier**: one transformer block (weights copied from the LRM's final block) + a
  prediction head on final-layer hidden states, predicting token-level "has the final answer
  arrived?" with class-weighted BCE. At inference a sliding window of the 10 most recent predictions
  injects `</think>` on majority vote (threshold 0.7) — no per-dataset calibrated threshold needed.
  Also documents observable signatures of answer arrival (token-confidence spike; "thinking token"
  frequency shifts, e.g. "hmm"/"okay" before vs "another" after).
- **Training / RL usage:** trains only the probe on hindsight labels; LRM frozen; no RL; stopping
  never used as generator reward.
- **Experiments & benchmarks:** Qwen3-8B/14B, Ministral-3-8B/14B-Reasoning; train mix AIME 1983–2024
  + MATH + OpenCoder-SFT + OpenScience (3 CoTs/problem); eval MATH-500, AIME 2025, HumanEval, GPQA.
  Baselines: Vanilla, NoThinking, DEER, Dynasor, Thought Calibration (supervised probes).
- **Key results:** 14–55% average CoT-length reduction across the four datasets; >2× inference
  latency reduction vs the original LRM; defines the Pareto frontier on 14/16 (LRM, benchmark)
  pairs; best or second-best on 28/32 metrics.
- **Limitations:** "Optimal" = first arrival of the model's *own* final answer (self-consistency
  proxy, no correctness or cost in the label); single-model CoT; inference-only; probe tied to the
  base model's hidden states.
- **Relation to CASSI:** Directly precedents CASSI's "oracle stopping labels computed post-hoc from
  completed trajectories, no extra rollouts → train a small stopper." The difference is the label
  semantics (first-answer-arrival vs quality−λ·cost argmax), the state (CoT tokens vs agent
  tool-use trajectories with budget state), and no executor training. THREAT LEVEL: **HIGH** for
  contribution 3's framing — "post-hoc optimal-exit dataset + small learned stopper" is no longer
  novel on its own; CASSI must lead with cost-awareness + the training bridge.

### When Does Learning to Stop Help? A Cost-Aware Study of Early Exits in Reasoning Models (Zhe Dong, Fang Qin, Manish Shah; 2026; arXiv 2606.30852 v2 (July 2026), preprint)

- **Read from:** PDF pages 1–8 (`research/papers/2606.30852_when-does-learning-to-stop-help.pdf`)
- **Problem:** Early-exit rules (confidence thresholds, entropy monitors, stability checks, learned
  stoppers) are evaluated under incomparable protocols; a deployed policy must commit to a threshold
  before test time, be compared at *equal lost-correct risk*, and pay for its own probes.
- **Method:** WHO decides — **LearnStop**, a deliberately simple separate learned stopper: at token
  checkpoints, fork the prefix, force a ≤48-token answer, extract 8 prefix-observable features
  (normalized budget, checkpoint index, answer log-prob, answer entropy, answer-match-previous, run
  length, prefix vote share, backtracking-marker density) and stop when a **logistic regression**
  predicts the forced answer is correct (label: match vs gold). Hidden-state-free → portable across
  model families. All policies (learned + scalar exits) calibrated on the same split with a
  finite-grid lost-correct UCB (Eq. 6) at matched risk α=0.15.
- **Training / RL usage:** trains only the classifier; no RL anywhere; inference-only.
- **Experiments & benchmarks:** 18 task–model settings: GSM8K, MATH-500, MMLU-Pro, AIME-90,
  GPQA-Diamond × Qwen3-8B/32B + DeepSeek-R1-Distill (Qwen-7B, Llama-8B). Baselines: confidence,
  entropy, confidence-leap, run-stability exits, DEER-style, EAT-style, PUMA-style,
  TERMINATOR-light, under one checkpoint protocol. Costs three serving regimes: KV-fork,
  prefix-cache, black-box API.
- **Key results:** A three-regime map. Learned stopping wins on free-form math: certifies 33.8–51.4%
  total-token savings at α=0.15 on Qwen3 GSM8K/MATH-500, beating the best calibrated scalar by +3.2
  to +21.2 pp. Calibrated scalar exits win on MMLU-Pro (scalar saves 2.7–3.9 pp more). On small hard
  sets (AIME-90 with 67.8% never-correct trajectories, GPQA) *no* aggressive policy is certifiable.
  Trajectory decomposition (early-solved / beneficial / harmful / unsolved / oscillating) predicts
  the regime: oscillation (9.4–14% on GSM8K) favors multi-feature learning. **Probe overhead can
  erase savings**: the same GSM8K-8B policy saves +32.2% tokens under KV-fork, −4.3% under
  prefix-cache, and *costs +120.9%* under black-box repeated prefilling. Feature complementarity:
  GSM8K-32B peak gain 0.157 combined vs 0.136 entropy-only.
- **Limitations:** Math/MC QA only, no agents/tools; correctness-only labels (cost enters evaluation,
  not the learned objective); logistic stopper by design (information-content study, not SOTA).
- **Relation to CASSI:** Not a competing method but a **referee**: it operationalizes exactly the
  evaluation CASSI will face (matched-risk comparison vs calibrated scalar exits, probe-overhead
  accounting per serving regime) and shows learned stopping is *not* uniformly better — which cuts
  against CASSI's "separate stopping model is necessary" claim unless CASSI wins under this
  protocol on agent workloads. Its overhead result pressures the "<3% monitor overhead" claim.
  THREAT LEVEL: **MEDIUM** — evaluation-level threat and a must-cite; not a method rival.

### LYNX: Learning Dynamic Exits for Confidence-Controlled Reasoning (Ömer Faruk Akgül, Yusuf Hakan Kalaycı, Rajgopal Kannan, Willie Neiswanger, Viktor Prasanna; 2025; arXiv 2512.05325 (Dec 2025), USC/DEVCOM ARL, preprint)

- **Read from:** PDF pages 1–6 (`research/papers/2512.05325_lynx-learning-dynamic-exits.pdf`)
- **Problem:** Existing early exits either manipulate decoding with extra sampling/heuristics, rely
  on auxiliary verifier LLMs, or are post-hoc probing pipelines without formal guarantees.
- **Method:** WHO decides — a **separate lightweight probe** on the frozen model's hidden states at
  naturally occurring cue tokens ("hmm", "wait", "alternatively"). Trained, not heuristic:
  supervision via **counterfactual forced exits** — at each cue, cut the CoT, append the native
  answer cue, generate a short answer-only continuation, compare to gold: label "safe to exit now."
  Probe scores are wrapped in **split conformal prediction**, giving a distribution-free bound on
  the premature-exit rate at user-chosen confidence 1−α. Trained + calibrated *once* per base model
  on a generic math corpus, then frozen and reused across benchmarks, temperatures, and even
  non-math tasks.
- **Training / RL usage:** probe-only training; no RL; base model unchanged; inference-only.
- **Experiments & benchmarks:** DeepSeek-R1-Distill-Qwen-1.5B, QwQ-32B, Llama-3.1-Nemotron-8B on
  GSM8K, MATH-500, AIME 2024, CommonsenseQA. Compared (Table 1) against DEER, Dynasor-CoT, NoWait,
  FlashThink, Think-or-Not, Zhang et al. probes on axes: online / no-proxy-LLM / self-contained
  data / statistical guarantee / cue-triggered / multi-benchmark — LYNX alone checks all six.
- **Key results:** GSM8K: matches or improves accuracy with 60–65% fewer tokens; MATH-500: up to
  +12 accuracy points with ~35–60% fewer tokens (e.g., DeepSeek-R1-1.5B 68.0→74.6–76.2); AIME 2024:
  baseline accuracy with >50% token savings; CommonsenseQA zero-shot: modest accuracy gains, up to
  70% fewer tokens; overall 1.5–3.3× efficiency at confidence levels 0.97→0.70.
- **Limitations:** Quality-only labels (no cost in objective — cost sensitivity only via α);
  hidden-state probe is executor-specific; label generation needs one forced short continuation per
  cue (cheap but not free); math-trained; single-model CoT; inference-only.
- **Relation to CASSI:** Another strong instance of "separate learned stopper on post-hoc
  counterfactual exit labels," with statistical guarantees CASSI lacks. Its forced-exit labeling
  also shows CASSI's O(T) oracle is standard practice in this family (the claimed contrast should
  target AgentPRM's O(K×T²) Monte Carlo, not early-exit labeling). THREAT LEVEL: **MEDIUM** —
  crowded-component threat; no cost objective, no agents, no executor training.

### Know When to Stop: Segment-Level Credit Assignment for Reducing Overthinking — DASH (Chia-Hsuan Lee, Sihui Dai, Mingyang Zhou, Isha Slavin, Shi-Xiong Zhang, Sambit Sahu, William Campbell; 2026; arXiv 2607.00482 (July 2026), Capital One, preprint)

- **Read from:** PDF pages 1–7 (`research/papers/2607.00482_know-when-to-stop-segment-credit.pdf`)
- **Problem:** Overthinking behaviors (hedging, approach abandonment, self-contradiction) are not
  just length: even length-controlled, incorrect traces show more unproductive self-reflection.
  Standard GRPO broadcasts one scalar advantage to all tokens, penalizing the valid prefix of a
  trace that found the right answer and then drifted away ("GRPO's blind spot").
- **Method:** WHO decides — no separate model and no inference-time stopper: a cheap **post-hoc
  proxy inside training**. Extract intermediate answer commitments (\boxed{}, "the answer is X")
  from each rollout, compare each to ground truth, detect **answer drift** (correct → incorrect).
  **DASH (Drift-Aware advantage SHaping)**: split each rollout into segments bounded by answer
  checkpoints; positive segments (leading to a correct checkpoint) get +|A|·α₊; negative segments
  get −|A|·α₋·w(t) with an *escalating* length penalty w(t)=1+α·(t−t_start)/(t_end−t_start) capped
  at w_max — literally encoding "the longer you continue past a correct answer, the worse." Drift
  traces also get shaped reward r_drift = r_incorrect + δ(1 − L_postdrift/L_total).
- **Training / RL usage:** This IS training-side stopping signal: segment-level stop/continue
  quality directly shapes GRPO advantages for the reasoning policy (n=16, 4×8 H100). No learned
  stopper — labels come from ground-truth checks on intermediate answers.
- **Experiments & benchmarks:** Llama-3.1-Nemotron-Nano-4B-v1.1 (high drift, ~20% on AIME-level),
  16.5K OpenR1-Math-220K problems; eval OlympiadBench, AMC23, AIME24, AIME25 (avg@32). Baselines:
  GRPO, DR-GRPO, GRPO+Brevity Bonus (shortest-correct-length bonus).
- **Key results:** AIME25 (highest drift prevalence): DASH 50.8% vs GRPO 45.4% vs base 46.1% — GRPO
  *degrades* AIME25 (−0.7) while DASH improves it (+4.7). Averages: DASH 67.22 vs GRPO 66.83 (but
  DR-GRPO 68.38 wins the average). DASH's correct traces show 2× more contradiction-then-resolution
  (0.92 vs 0.47/trace) with fewer blind abandonments — longer but more productive reasoning; traces
  11.8% longer yet lowest overthinking-signal profile. Ablation: removing the escalating length
  penalty on negative segments costs −2.2 on AIME25.
- **Limitations:** Needs ground truth at every intermediate checkpoint (RLVR-style domains only);
  same-model, math-only, no tools/agents; no explicit cost/λ (drift, not dollars); no inference-time
  controller; no separate stopper; DR-GRPO beats it on average.
- **Relation to CASSI:** The closest existing instance of "stopping-relevant signal as a *process
  reward inside GRPO*" — it partially occupies CASSI's contribution 1 bridge, but from ground-truth
  drift checks rather than a learned cost-aware stopping model, with no cost formalism, no agent
  costs, and no loop. Also independently validates CASSI's oracle intuition (post-t* continuation
  should be penalized increasingly). THREAT LEVEL: **HIGH** for the reward-bridge novelty in the
  single-model reasoning setting; CASSI must differentiate via the learned stopper as reward source,
  cost-awareness, agentic tasks, and the controller+trainer dual role.

### To Search or Not to Search: Aligning the Decision Boundary of Deep Search Agents via Causal Intervention — DAS (Wenlin Zhang, Kuicai Dong, Junyi Li, Yingyi Zhang, Xiaopeng Li, Pengyue Jia, Yi Wen, Derong Xu, Maolin Wang, Yichao Wang, Yong Liu, Xiangyu Zhao; 2026; **WWW 2026** (peer-reviewed), arXiv 2602.03304, CityU HK/Huawei)

- **Read from:** PDF pages 1–7 (`research/papers/2602.03304_to-search-or-not-to-search.pdf`)
- **Problem:** Deep search agents mis-place the **decision boundary** — the threshold where
  accumulated knowledge suffices to answer: **over-search** (redundant searches despite sufficient
  knowledge; efficiency failure) and **under-search** (premature answering; accuracy failure).
  Outcome-centric RL training prioritizes final results over per-round decision optimality.
- **Method:** WHO decides — the agent itself after alignment (no separate stopper at inference).
  **Causal-intervention diagnosis** on completed trajectories: at each decision point apply
  do(A_t:=Answer) — if the counterfactual answer is correct, knowledge was Sufficient → the factual
  Search was over-search; when the agent answered wrongly, apply do(A_t:=Search) to build the
  corrective counterfactual trajectory (under-search). Post-hoc, per-decision-point stop/continue
  relabeling — a hindsight oracle for the search-vs-answer boundary (requires interventional
  continuations, roughly O(T) extra partial rollouts per trajectory).
- **Training / RL usage:** **Yes — stopping decisions become training signal for the executor**:
  preference pairs (shorter counterfactual answer ≻ over-searching factual trace; counterfactual
  search ≻ premature factual answer) fine-tune the agent policy via **DPO** (β=0.3, LoRA r=64,
  20K pairs from NQ+HotpotQA, 3 epochs) on Search-R1-trained Qwen2.5-7B/14B.
- **Experiments & benchmarks:** NQ, HotpotQA, 2WikiMultiHopQA; agents: Search-R1 (Qwen2.5-7B/14B
  base and RL) and Search-O1 (GPT-4o, GPT-4o-mini, Gemini-2.5-Flash, DeepSeek-V3); metrics: EM,
  inference time, avg search queries (ASQ), Over-Search Rate, Under-Search Rate.
- **Key results:** Boundary errors are pervasive: GPT-4o USR 0.670 on NQ; outcome RL (Search-R1)
  lifts NQ EM 0.190→0.410 and cuts USR 0.672→0.505 but *raises* OSR 0.196→0.263 with ASQ +60% —
  "incentivizing correctness without penalizing cost alleviates under-search only to exacerbate
  over-search." DAS on 7B-RL: NQ EM 0.381→0.394 with time 464→432s, ASQ 2.014→1.896, both OSR and
  USR down; 14B NQ EM 0.440→0.454, ASQ 1.016→0.955; 2Wiki 7B EM 0.457→0.478. Over-search ≈19% of
  all search actions. Under- and over-search preference data act antagonistically; both are needed.
- **Limitations:** Binary sufficiency (no continuous value, no explicit cost/λ or budget state);
  DPO on trajectory pairs, not step-level process rewards; no separate stopping model (no reusable
  monitor, no inference-time control); knowledge-sufficiency only (search agents), not general
  tool cost; modest absolute gains.
- **Relation to CASSI:** In CASSI's own task family (multi-hop search QA), DAS already
  (i) computes post-hoc stop/continue oracle labels via counterfactual interventions and
  (ii) trains the executor on them. Its RL finding (outcome reward → over-search) is strong
  motivational evidence *for* CASSI's cost-aware process rewards. THREAT LEVEL: **HIGH** — closest
  agentic "stopping supervision trains the executor" work; CASSI must beat DAS-style preference
  alignment with its Δ-as-PRM bridge and show the separate stopper + cost formalism add value.

### Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents (Wenxuan Ding, Nicholas Tomlin, Greg Durrett; 2026; arXiv 2602.16699 v3 (May 2026), NYU/UT Austin, preprint)

- **Read from:** PDF pages 1–5, 7–8 (`research/papers/2602.16699_calibrate-then-act.pdf`)
- **Problem:** Agents must trade exploration cost against uncertainty — when to stop exploring and
  commit. Formalized as a POMDP with reward `R = 1[task completed]·D_θ(a_{1:T})` where D_θ is a
  cost-discount over actions taken (Pandora's Box: γ^t; QA with optional retrieval; file-reading
  coding with unit-test/code-execution discounts d_u, d_c).
- **Method:** WHO decides — the agent, but informed by an **external calibrated prior**: CTA
  estimates p̂(Z|x) (verbalized confidence calibrated by isotonic regression — ECE 0.618→0.029 on
  PopQA — or a trained environment-state predictor) and injects it into the prompt so the model can
  explicitly compute the explore/commit tradeoff. CTA-Prompted (zero-shot) and CTA-RL (GRPO
  fine-tuning end-to-end with the cost-discounted reward, conditioned on priors).
- **Training / RL usage:** CTA-RL trains the executor with a **cost-discounted terminal reward**
  (GRPO); the calibrated estimator is an *input feature*, not a reward source; no separate stopping
  model issuing process rewards.
- **Experiments & benchmarks:** Pandora's Box (Qwen3-8B, 100 instances, K=3, γ∈[0,1] sweeps), QA
  with optional retrieval (PopQA-based), FileReading with relative costs ρ∈{0.5,1,2,4}.
- **Key results:** Pandora's Box: CTA-Prompted 94.0% optimal-policy match rate (avg reward 0.625 vs
  oracle 0.649) vs Prompted 23.0% (0.476). QA: CTA-Prompted reward 0.293 vs Prompted 0.283 vs
  Prompted-NonThink 0.244 (which retrieves 97.7% of the time — cost-insensitive). FileReading:
  plain RL and prompting both **collapse to static policies across cost regimes** (0% code-first
  traces); CTA-RL 0.268 vs RL 0.259 (+3.5%). Headline: "LLMs struggle to make calibrated,
  cost-sensitive decisions when prior estimates are implicit" — even after RL.
- **Limitations:** Small/synthetic settings (single-round retrieval decision; toy file schemas);
  fixed multiplicative cost discounts; no learned stopping model; no process rewards; no
  multi-step budget state.
- **Relation to CASSI:** Strongest *supporting evidence* for CASSI's H-claims: plain GRPO with
  cost-discounted rewards fails to internalize cost-sensitivity (backs the "single-model
  cost-penalty GRPO is insufficient / representation conflict" ablation hypothesis), and an
  external calibrated estimate fixes it (analogous to a stopper informing the executor — though via
  prompt, not reward). Also stakes out "cost-aware exploration in LLM agents" terminology.
  THREAT LEVEL: **MEDIUM** — conceptual overlap on cost-aware stopping in agents, but toy scale, no
  learned stopper, no process-reward bridge.

### Efficiently Scaling LLM Reasoning with Certaindex (+ Dynasor) (Yichao Fu, Junda Chen, Siqi Zhu, Zheyu Fu, Zhongdongming Dai, Yonghao Zhuang, Yian Ma, Aurick Qiao, Tajana Rosing, Ion Stoica, Hao Zhang; 2024/2025; arXiv 2412.20993 v2 (May 2025), UCSD/Tsinghua/CMU/Snowflake/Berkeley, preprint)

- **Read from:** PDF pages 1–4 (`research/papers/2412.20993_certaindex-dynasor.pdf`)
- **Problem:** Test-time algorithms (CoT, SC, MCTS, REBASE) waste tokens after answers stabilize;
  serving systems lack a progress signal to allocate/deallocate compute.
- **Method:** WHO decides — the serving system, heuristically. **Probe-In-The-Middle**: periodically
  ("every 64 tokens") append "Oh, I suddenly got the answer... Final Answer: boxed{" to force an
  intermediate answer; **Certaindex** = sliding-window answer-consistency C_k = (1/w)Σ1[y_j=y_k]
  (plus hesitation-marker filtering; generalizes to SC vote share / MCTS statistics); terminate when
  C_k ≥ τ. **Dynasor** integrates certaindex into a reasoning-aware serving system (early exit,
  dynamic token allocation, gang scheduling). Training-free, same-model probing, threshold-based.
- **Training / RL usage:** none — pure inference/serving-time control.
- **Experiments & benchmarks:** math (AMC23, AIME24, MATH-500), multiple reasoning algorithms and
  models; batch and online serving with SLO attainment.
- **Key results:** DeepSeek-R1-style models spend a median 2.7K tokens on AMC23 where 830 suffice;
  Dynasor saves up to 50% compute in batch inference at equal accuracy; online: 3.3× higher
  sustainable request rate or 4.7× tighter latency SLOs. (Follow-on "Dynasor-CoT" applies this to
  CoT decoding with 29% token savings, per later papers' comparisons.)
- **Limitations:** Heuristic thresholds (needs calibration); consistency can lock in confident
  wrong answers; quality-only signal (no cost–quality objective; cost handled by the scheduler);
  no learning, no agents (tool costs invisible), no training feedback.
- **Relation to CASSI:** The canonical system-level stopping baseline and the source of the
  "answer stabilization" signal CASSI's stopper consumes as an input feature ("stability
  indicators"). CASSI's inference-controller mode must beat certaindex-style thresholds at matched
  risk. THREAT LEVEL: **LOW-MEDIUM** — heuristic and single-model; but its serving-cost framing
  (probe overhead, scheduling) is the right accounting standard.

### Dynamic Early Exit in Reasoning Models — DEER (Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Minghui Chen, Zheng Lin, Weiping Wang; 2025; arXiv 2504.15895 v3 (Sep 2025), CAS-IIE/Huawei, preprint; on OpenReview)

- **Read from:** PDF pages 1–4 (`research/papers/2504.15895_deer-dynamic-early-exit.pdf`)
- **Problem:** "Pearl reasoning" — the critical point where reasoning information becomes just
  sufficient: ~75% of AIME samples contain such an early-exit point; 36.7% need less than half the
  original CoT; some answers are *only* correct with early exit.
- **Method:** WHO decides — the same model, heuristically. Monitor **action transition points**
  ("Wait"/"Alternatively", or entropy-based); at each ATP, induce a trial answer via "final answer"
  prompt injection; compute confidence = geometric-mean token probability of the trial answer;
  exit if C > λ (empirical threshold ~0.95), else revoke the trial branch and continue. DEER-Pro:
  parallel answer inductions at candidate exits with aggregated calibration. Training-free,
  plug-and-play.
- **Training / RL usage:** none.
- **Experiments & benchmarks:** 11 reasoning LLMs (DeepSeek-R1-Distill 1.5B–32B, QwQ, etc.), 10
  benchmarks: GSM8K, MATH-500, AMC23, AIME24, GPQA-Diamond, BigCodeBench, LiveCodeBench, etc.
- **Key results:** CoT length −19.1% to −80.1% on average while accuracy +0.3% to +5.0% across
  benchmarks/models.
- **Limitations:** Confidence thresholds are brittle across models/temperatures (LYNX, LearnStop
  critiques); trial-answer branches add probe cost; quality-only; single-model; no learning; no
  agents; no training feedback.
- **Relation to CASSI:** The standard training-free early-exit baseline (CASSI plan already cites
  it). Its confidence signal is subsumable as a stopper input feature. THREAT LEVEL: **LOW** —
  functional neighbor, methodologically distant; must appear as baseline.

## Peripheral papers

**s1: Simple test-time scaling — budget forcing (Muennighoff, Yang, Shi, Li, Fei-Fei, Hajishirzi,
Zettlemoyer, Liang, Candès, Hashimoto; 2025; arXiv 2501.19393; read pp.1–3).** SFT of
Qwen2.5-32B-Instruct on 1K curated traces (26 min on 16 H100s), then **budget forcing**: forcibly
append the end-of-thinking delimiter to cap thinking, or suppress it and append "Wait" to extend.
s1-32B beats o1-preview by up to 27% on MATH/AIME24; extrapolation lifts AIME24 50%→57%. Stopping
is *externally imposed* (a length knob), not decided from state — the anti-adaptive pole of this
design space and a mandatory baseline (already in CASSI's plan). LOW threat; used by every 2026
stopping paper as the "static budget" strawman.

**RL mode-selection family — Thinkless / AdaptThink / Learning-When-to-Think / Switch-Reasoner.**
Thinkless (Fang, Ma, Wang; NUS; arXiv 2505.13379; read pp.1–4): DeGRPO decouples the control-token
(<think>/<short>) loss from response loss (reward 1.0 correct-short, 1−γ correct-long, −1 wrong);
cuts long-form reasoning 50–90% on Minerva Algebra/MATH-500/GSM8K. AdaptThink (Zhang, Lin, Hou,
Feng, Li; Tsinghua; arXiv 2505.13417; read p.1): constrained-optimization RL + importance sampling
for Thinking/NoThinking; −53% response length and +2.4% accuracy for R1-Distill-1.5B on three math
sets. "Learning When to Think" (2505.10832) does the same via multi-stage RL; Switch-Reasoner
(2607.08572, July 2026) extends to multitask MLLM mixtures. All are **same-model, pre-hoc binary
mode choices trained with static cost preferences** — they adapt *whether* to think, not *when to
stop mid-trajectory*, and none uses a separate stopper or per-step process rewards. MEDIUM-LOW
threat: they undercut the "static penalty" strawman if CASSI overstates it (these are learned,
input-adaptive policies), but they don't touch the stopper-as-PRM bridge.

**Hidden-state probing family — Reasoning Models Know When They're Right (Zhang, Chen, Pan, Zhao,
Panda, Li, He; NYU; arXiv 2504.05419; read pp.1–4) + Thought Calibration (Menon et al.; EMNLP 2025;
arXiv 2505.18404) + Thinking Out Loud (2504.06564).** 2504.05419: chunk CoT by intermediate
answers (Gemini-labeled correctness), train a 2-layer MLP probe on hidden states — ROC-AUC >0.7,
ECE <0.1, and *look-ahead*: correctness is predictable before the answer is articulated; used as a
verifier for early exit → 24% token reduction without accuracy loss (R1-Distill 1.5B–70B, QwQ-32B;
GSM8K/MATH/AIME/KnowLogic). Thought Calibration: linear probes over reasoning-tree structure with
Learn-then-Test calibration; up to 60% thinking-token reduction in-distribution, 20% OOD, across
3 LRMs/4 datasets. Together they establish that a **separate tiny learned model deciding stopping is
mainstream** — but always probes on the executor's own hidden states, quality-only, inference-only.
MEDIUM threat to "separate stopper" novelty; no cost, no agents, no training bridge.

**Early-stopping self-consistency family — ESC (Li et al.; ICLR 2024; arXiv 2401.10480), Adaptive
Self-Consistency (Aggarwal et al.; EMNLP 2023; arXiv 2305.11860), Difficulty-Adaptive SC
(2408.13457), CGES (2511.02603), MARS (2606.12935).** ESC stops sampling when a window of SC
samples is unanimous: −33.8% samples on MATH, −80.1% GSM8K, −76.8% StrategyQA, −78.5% CSQA at
comparable accuracy. ASC uses a lightweight Beta-mixture stopping criterion per question (~2–4×
fewer samples). CGES replaces majority count with calibrated confidence aggregation; MARS (2026)
adds margin-based risk-controlled stopping for parallel scaling with per-instance guarantees.
This is the oldest "learned/adaptive stopping of sampling" line — orthogonal axis (how many
samples, not how many steps) but the same cost-quality tradeoff; classical sequential-testing
framing. LOW threat; cite as the sampling-dimension analogue.

**Answer Convergence as a Signal for Early Stopping in Reasoning (2025; arXiv 2506.02536).** Finds
models converge to their final answer after ~60% of reasoning steps; proposes (1) answer-consistency
stopping, (2) boosting end-of-reasoning token probability, (3) a **supervised classifier on internal
activations that learns when to stop**. On NaturalQuestions, answer consistency cuts >40% of tokens
while *improving* accuracy; five benchmarks, five open models. LOW-MEDIUM threat: another learned
stopper instance (quality-only, single-model, inference-only).

**Adaptive retrieval — when to retrieve: Self-RAG (Asai et al.; ICLR 2024; arXiv 2310.11511),
Adaptive-RAG (Jeong et al.; NAACL 2024; arXiv 2403.14403), FLARE (Jiang et al.; EMNLP 2023; arXiv
2305.06983), SeaKR (Yao et al.; 2024; arXiv 2406.19215), Adapt-LLM (Labruna et al.; RANLP 2025;
arXiv 2404.19705), When to Retrieve During Reasoning (2026; arXiv 2604.26649).** The retrieval
literature solved a sibling problem years earlier: Self-RAG trains special reflection tokens
(including [Retrieve]) via critic-labeled SFT — a same-model learned decision; Adaptive-RAG trains
a **separate small classifier** routing queries to no/single/multi-step retrieval by predicted
complexity; FLARE retrieves when next-token confidence is low (heuristic); SeaKR triggers retrieval
from internal-state self-aware uncertainty; Adapt-LLM trains generation of a ⟨RET⟩ token (beats
always/never-retrieve on PopQA). 2604.26649 brings this to LRMs (when to retrieve mid-reasoning).
Pattern: separate-or-same-model *learned gating of a costly action* — structurally identical to
stopping ("stop gathering") but with binary per-call costs and no trajectory-level termination or
training-reward bridge. LOW threat individually; collectively they preempt "first learned decision
about costly actions" phrasing, so CASSI's claims must be scoped to termination + process rewards.

**When Agents go Astray: Course-Correcting SWE Agents with PRMs — SWE-PRM (Jain et al.; 2025;
arXiv 2509.02360, v2).** An **inference-time Process Reward Model** intervenes during SWE-agent
execution to correct trajectory-level inefficiencies from a taxonomy including "failure to
terminate once a solution is reached," redundant exploration, and looping. On SWE-bench Verified,
closed-source PRM feedback lifts resolution 40.0%→50.6% (+10.6pp), largest on medium/hard, while
*reducing* trajectory length; added cost as low as $0.2/task; policy unchanged. MEDIUM threat: a
separate monitor already polices agent termination in SWE-bench (one of CASSI's benchmarks) — but
it is feedback-injection at inference, not a trained stopper, not cost-objective-based, and never a
training reward.

**2026 miscellany — stopping in/around training and agents.** SAGE / "Does Your Reasoning Model
Implicitly Know When to Stop Thinking?" (arXiv 2602.08354 v5): shows LRMs *implicitly* know the
right stop time but sampling obscures it; SAGE sampling exposes it and **SAGE-RL folds the
discovered efficient stopping patterns into group-based RL training**, improving accuracy and
efficiency on math — a same-model, self-signal analogue of CASSI's bridge (no separate stopper, no
cost formalism). Semantic Early-Stopping for Iterative LLM Agent Loops (2606.27009): replaces
max-iterations kill-switches in Writer–Critic loops with semantic improvement detection. Knowing
When to Quit (2604.18419): dynamic abstention mid-reasoning as principled risk control. RLStop
(Stevenson & Bin-Hezam; 2024; arXiv 2405.02525): an RL-trained stopping policy for technology-
assisted review — an early precedent of "train a stopping policy with RL" outside LLM reasoning.
Optimal Stopping vs Best-of-N (2510.01394): Pandora's-Box (Weitzman) framing of inference-time
sampling with per-sample costs — the classical economics CASSI's oracle inherits. Agentic
Abstention (2606.28733) and VLAA-GUI (2604.21375): agent-side premature-stop/loop failure analyses.
Collectively LOW-MEDIUM threat, but they show the 2026 frontier is converging on CASSI's territory
from several directions at once.

## Synthesis

### Landscape classification

| Method (year) | Stop decider: same vs separate model | Heuristic vs learned | Inference-only vs trains-policy | Cost-aware vs quality-only |
|---|---|---|---|---|
| s1 budget forcing (2025) | external knob (neither) | heuristic (fixed budget) | inference-only | cost-only (budget), state-blind |
| ESC / ASC / DSC / CGES (2023–25) | same (output votes) | heuristic / lightweight stats | inference-only | quality-only (cost saved, not optimized) |
| DEER (2025) | same (own confidence) | heuristic threshold | inference-only | quality-only |
| Certaindex / Dynasor (2024–25) | same signals, external scheduler | heuristic threshold | inference-only (serving) | quality signal; cost via scheduler |
| Answer Convergence (2025) | same + small classifier | both variants | inference-only | quality-only |
| Probing (2504.05419) / Thought Calibration (2025) | **separate probe** (on own hidden states) | learned (+calibration) | inference-only | quality-only |
| LYNX (2025) | **separate probe** | learned + conformal | inference-only | quality-only (α knob) |
| TERMINATOR (2026) | **separate probe** (1 block + head) | learned (hindsight labels) | inference-only | quality-only (first-arrival) |
| LearnStop study (2026) | **separate classifier** (features) | learned, risk-calibrated | inference-only | cost-aware *evaluation*, quality-only labels |
| OS-Pruner (2026) | **separate stopping policy** (head + 2 layers) | learned (optimal stopping) | inference-only | **cost-aware objective: A − λ·L** |
| CaRT (2025) | separate (medical) / same (math) LM | learned (counterfactual SFT, +RL on stop) | trains stopper only; executor frozen | quality-only labels (γ implicit) |
| FlashThink (2025) | separate verifier LLM | learned/prompted | inference-only | quality-only |
| Thinkless / AdaptThink / LWT / Switch-Reasoner (2025–26) | same model (mode token) | learned (RL) | **trains policy** (mode choice) | static cost preference in reward |
| DASH (2026) | none at inference (GT drift check in training) | learned policy via shaped GRPO | **trains policy** (segment advantages) | drift/length-aware, no explicit λ·cost |
| DAS (2026, WWW) | agent itself after alignment | learned (causal counterfactual labels → DPO) | **trains policy** (executor DPO) | efficiency-framed, no explicit cost objective |
| SAGE-RL (2026) | same model (implicit signal) | learned (RL on SAGE data) | **trains policy** | quality+efficiency, no explicit cost |
| CTA (2026) | agent + external calibrated prior | learned prior + GRPO | **trains policy** (CTA-RL) | **cost-discounted reward** |
| SWE-PRM (2025) | **separate PRM** | learned/prompted PRM | inference-only (feedback injection) | efficiency taxonomy, no cost objective |
| Self-RAG / Adaptive-RAG / SeaKR / Adapt-LLM (2023–24) | same (token) / separate classifier | learned | trains gating (not termination) | quality-only (retrieval cost implicit) |
| Adaptive-Rag'26 agent works (GRASP, 2604.26649) | same agent | learned/RL | mixed | mostly quality-first |
| **CASSI (proposed)** | **separate small LM stopper** | **learned (oracle t* = argmax q−λc)** | **trains policy via stopper-Δ process rewards + controller at inference** | **explicitly cost-aware, per-instance λ/budget tiers** |

### Where the gaps actually are

1. **No stopper-as-PRM bridge.** No paper trains a stopping/value model and reuses its cost-aware
   margin Δ as a *step-level reward* for RL-training the executor. The near misses: DASH (stop-
   relevant segment advantages, but from ground-truth drift, same-model, no stopper), DAS
   (stop/continue labels → DPO, trajectory-level, no separate model, no continuous value), SAGE-RL
   (self-signal → RL data), CaRT+RL (RL on the stop decision only), CTA-RL (cost-discounted
   terminal reward, prior as input). The specific composition is open.
2. **No closed loop.** Nobody iterates executor-improvement → relabel oracle → retrain stopper →
   better process rewards. All pipelines are one-shot.
3. **Cost-aware stopping objectives exist only for tokens, single-model.** OS-Pruner has
   quality − λ·tokens; nobody has multi-dimensional agent costs (tool fees, latency, dollars,
   budget tiers with dynamic λ) inside the stopping objective, and no agentic benchmark (GAIA,
   SWE-bench, WebWalker) has a learned cost-aware stopper.
4. **Executor-agnostic stoppers are rare.** Most learned stoppers are hidden-state probes welded to
   one base model; text-level stoppers that transfer across executor families (LearnStop's features,
   CaRT's 3B LM) are the exception — CASSI's structured-text small-LM stopper is in the less
   crowded cell.
5. **Evaluation rigor is now table stakes.** Matched-risk calibration, probe-overhead accounting per
   serving regime (LearnStop), conformal premature-exit guarantees (LYNX), and OSR/USR-style
   decision-error metrics (DAS) define how 2026 reviewers will judge stopping claims.

### Top threats to CASSI's novelty (ranked)

1. **OS-Pruner (2607.11089)** — same stopping objective (quality − λ·cost), optimal-stopping/Bellman
   theory, learned stopper trained without extra on-policy rollouts, λ-frontier control, plus a
   threshold-suboptimality theorem. Directly erodes contribution 3's oracle formulation and the
   formal-properties section (uniqueness/λ-monotonicity claims will look incremental). CASSI must
   scope its theory to *agent trajectories with heterogeneous costs* and lean on the training
   bridge.
2. **DAS / To Search or Not to Search (WWW 2026)** — post-hoc counterfactual stop/continue oracle +
   executor training (DPO) in deep-search agents, with over/under-search metrics and the "outcome RL
   causes over-search" finding. Erodes "first to use stopping decisions as executor training signal
   in agents"; CASSI's edge narrows to *process-level rewards from a learned value model* + cost
   formalism + closed loop.
3. **CaRT (2510.08517)** — separate termination model over a frozen executor, counterfactual
   training, post-hoc success-rate prefix labels, GRPO-on-stop variant, medical *agentic* domain.
   The plan's characterization ("CaRT lacks executor training") is verified and correct — that is
   precisely the remaining wedge, plus cost.
4. **DASH (2607.00482)** — "know when to stop" as segment-level credit assignment *inside GRPO*
   occupies the training-side stopping-reward idea in single-model reasoning; also note DR-GRPO
   beats it on average — evidence this bridge is hard to make pay off.
5. **TERMINATOR (2603.12529) + LYNX (2512.05325) + Thought Calibration (2505.18404) +
   probing (2504.05419)** — collectively saturate "post-hoc exit labels train a small separate
   stopper (O(T), no extra rollouts)"; only the cost term and the agent state distinguish CASSI's
   stopper training.
6. **LearnStop study (2606.30852)** — an adversarial evaluation standard: learned stoppers lose to
   calibrated scalars on some regimes and probe overhead can exceed savings (+120.9% under
   black-box serving), directly pressuring CASSI's "<3% overhead" and "separate model necessary"
   claims. CASSI should pre-empt by adopting matched-risk protocol and reporting per-serving-regime
   costs.
7. **CTA (2602.16699)** — owns "cost-aware exploration in LLM agents" phrasing; but its negative
   RL result is ammunition for CASSI's motivation.

### Opportunities

- **The composition is still open and now well-motivated by others' findings:** DAS shows outcome
  RL mis-sets the stopping boundary; CTA shows cost-discounted GRPO alone doesn't internalize cost;
  LearnStop shows scalar signals miss oscillating trajectories; OS-Pruner's Theorem 1 proves
  value-margin stopping strictly dominates threshold classification. Each is a citable argument
  for CASSI's exact design (learned cost-aware value Δ, used both as controller and process reward).
- **Agentic, multi-dimensional cost is virgin territory:** every learned stopper above optimizes or
  saves *tokens of one model*; none prices tool calls, API dollars, or tiered budgets, and none runs
  on GAIA/WebWalker/SWE-bench with a trained stopper (SWE-PRM is the lone, non-cost, inference-only
  neighbor).
- **The closed loop is unclaimed** — but reviewers will demand evidence it actually converges
  (CaRT's RL hurting termination length and DASH losing to DR-GRPO on average are cautionary
  precedents to address).
- **Positioning language:** avoid "first learned stopping model" (false), "first post-hoc oracle
  labels" (false — TERMINATOR/LYNX/OS-Pruner), "first cost-aware stopping objective" (false —
  OS-Pruner). Defensible: "first to train a separate cost-aware stopping model on hindsight
  economic-optimality labels from agent trajectories *and* to close the loop by using its value
  estimates as process rewards for executor RL," with DAS/DASH/CaRT/OS-Pruner as the four
  boundary-markers to beat empirically.
