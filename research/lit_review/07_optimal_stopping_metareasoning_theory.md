# Optimal Stopping Theory, Rational Metareasoning & Value of Computation

> Area 07. Researched 2026-07-16. Question: where does CASSI's oracle
> `t* = argmax_t [quality_t − λ·cumcost_t]`, its margin `Δ(s_t)`, and its trained stopping model sit
> in the theoretical lineage of optimal stopping / rational metareasoning / value of computation
> (VOC) — and has this lineage already been applied to LLM agents with *trained* stopping?

## Area overview

The idea that an agent should stop deliberating when the expected gain of further computation no
longer covers its cost is one of the oldest formalized ideas in AI. Simon (1958) posed "the global
optimization problem is to find the least-cost or best-return decision, *net of computational
costs*"; I. J. Good (1968/1971) called optimally solving this meta-problem "Type II rationality."
The 1980s–90s produced two near-identical formal programs: Horvitz's *flexible computation* and
expected value of computation (EVC), and Russell & Wefald's *rational metareasoning*, in which
computations are treated as actions whose value — the **value of computation, VOC** — is the
expected improvement in decision quality minus the cost of the computation; deliberation stops when
no computation has positive VOC. Dean & Boddy's *anytime algorithms* supplied the object being
controlled: an interruptible process whose solution quality improves with time. This line
culminates in two load-bearing formal results for CASSI: Hansen & Zilberstein (2001) define the
optimal stopping point of an anytime process as **t\* = argmax_t E[U(q,t)] with time-separable
utility U(q,t) = U_I(q) − U_C(t)** — syntactically CASSI's oracle — and solve run-time monitoring
(continue/stop from observed quality, with monitoring itself costing something) by dynamic
programming; Hay, Russell, Tolpin & Shimony (2012) formalize the general problem as a **metalevel
MDP** whose value function is exactly `E[−c·N + quality-at-stop]`, prove myopic-vs-optimal stopping
relations, and show metalevel problems are *not* index-solvable in general. In economics, Weitzman
(1979) solved the parallel "which alternative to probe next and when to stop" problem (Pandora's
box) with reservation-price indices — the canonical optimal-stopping-with-inspection-costs result.

Two later strands matter for CASSI's *learning* claims. First, **learning the metalevel policy**
instead of deriving it: Harada & Russell (1998/99) proposed RL on the metalevel MDP; Callaway,
Lieder et al. (UAI 2018) built the first general "metacognitive RL" method (BMPS), learning a VOC
approximation (a weighted mix of myopic VOI, VPI and cost — with per-computation cost literally
denoted λ) that decides, among other things, *when to terminate deliberation*; Svegliato et
al. (IJCAI 2018; ICRA 2020) learn stopping policies for anytime algorithms online, without
performance profiles, via model-free RL. Second, **learning to stop inside deep models**: Chen et
al. (ICML 2020) train a stopping policy for iterative deep models with a two-stage recipe —
compute a *closed-form oracle stopping distribution post-hoc from the completed forward pass using
label information* (`q*(t|y,x) ∝ exp(−ℓ(y,x_t)/β)`, MAP = argmax over steps), then *imitate* the
oracle with a policy that only sees runtime features. This is structurally CASSI's oracle-labeling
trick, eight years early, minus the explicit λ·cost term and minus agents.

In the LLM era (2024–2026) the lineage has been picked up on three distinct axes. (i) *Training a
single model with a VOC reward*: De Sabbata, Sumers, Griffiths et al. (2024/25) explicitly import
Russell & Wefald's VOC into LLM post-training (reward = utility-gain of the CoT minus γ·length,
optimized by Expert Iteration) — the closest thing to direct prior art for "cost-aware training
signal," though single-turn, single-model, and not agentic. (ii) *Training-free optimal-stopping
controllers at inference*: Kalayci et al. (2025) port Weitzman's Pandora's box to best-of-N
sampling with a UCB stopping rule; Huang et al. (2026) do Bayesian optimal stopping for
self-consistency sampling; Fang et al. (2026) score LLM *search-agent* actions with a hand-built
task-level VOI-per-unit-budget controller (explicitly *not* a learned value model). (iii)
*Framing/surveys*: computational rationality (Gershman, Horvitz, Tenenbaum 2015) and
resource-rational analysis (Lieder & Griffiths 2020) as the cognitive-science umbrella, and
2025–26 surveys of adaptive test-time compute. The specific combination CASSI claims — a
*separate, trained* stopping model, supervised by *post-hoc cost-aware oracle labels* over agent
trajectories, then re-used as a *process reward to train the executor* — does not appear in any of
the above; multiple 2026 sources still describe RL-trained stopping decisions for LLM
agents/orchestration as an open gap. But nearly every ingredient exists separately, and the formal
"properties" CASSI states are substantially classical.

## Core papers

### Principles of Metareasoning (Stuart Russell & Eric Wefald, 1991, Artificial Intelligence 49:361–395; and *Do the Right Thing: Studies in Limited Rationality*, MIT Press 1991)
- **Read from:** abstract + secondary (AIJ paper is paywalled on ScienceDirect). Formal content
  cross-verified from three primary PDFs read in full: Hansen & Zilberstein 2001 (§1, §3.2, refs
  [28–30]), Hay et al. 2012 (Def. 6 and §1), Callaway et al. 2018 (§2.1–2.2), De Sabbata et
  al. 2025 (Eq. 1).
- **Problem:** How should a bounded agent choose *which computations to perform and when to stop
  computing and act*, given that computation improves decisions but costs time?
- **Method/Theory:** Computations are meta-level actions. The **value of computation** VOC(c,b) =
  expected improvement in decision quality from performing computation c in belief state b (and
  continuing optimally) minus cost(c). Optimal meta-policy: perform argmax_c VOC(c,b); **stop and
  act when no computation has positive VOC** (VOC(⊥,b)=0). Because exact VOC is intractable
  (requires reasoning over all future computation sequences), they introduce the **meta-greedy /
  myopic approximation with the single-step assumption**: value each computation as if it were the
  last, using estimates of how it would change the current best action's utility. Time cost is
  handled via a *time-separable* comprehensive utility (intrinsic value minus cost of time — the
  terminology Hansen & Zilberstein adopt). Maps to CASSI: Δ(s_t) = Q_continue − Q_stop is a
  1-step VOC estimate; "stop when Δ<0" is exactly R&W's stopping rule; λ is the cost-of-time
  weight.
- **Training / RL usage:** None (analytical estimates from search statistics; pre-learning era).
- **Experiments:** Applied to game-tree search control (MGSS*/othello in *Do the Right Thing*),
  large speedups over fixed-depth search at equal decision quality.
- **Key results:** General decision-theoretic account of meta-level control; myopic VOC control of
  search; the field's standard vocabulary (object level vs meta level).
- **Limitations:** Myopic/single-step assumptions can stop too early; VOC estimation hand-derived
  per domain; no learned components.
- **Relation to CASSI:** The conceptual ancestor of the entire proposal — CASSI's Δ(s_t) *is* an
  amortized VOC estimate and its oracle objective is R&W's comprehensive utility. Not a competing
  method. **THREAT: MEDIUM** — no overlap in machinery, but any theory-aware reviewer will expect
  CASSI's framing ("agents lack economic judgment") and Property 1-style claims to be positioned
  relative to R&W; failure to cite would look like re-invention.

### Optimal Search for the Best Alternative (Martin L. Weitzman, 1979, Econometrica 47(3):641–654)
- **Read from:** PDF pages 1–7 of the MIT-EL 78-008WP working-paper version
  (`research/papers/weitzman-1979_pandoras-box-optimal-search.pdf`).
- **Problem:** "Pandora's box": n boxes with independent reward distributions F_i, opening cost
  c_i (and time lag t_i). Which box to open next, and when to stop and take the best reward found?
- **Method/Theory:** Full characterization by a **reservation-price index policy**: each source i
  gets a reservation price z_i solving c_i = E[(x_i − z_i)+] (the cost equals the expected excess
  reward over z_i — Kalayci et al.'s "fair-cap value"). *Selection rule:* open boxes in decreasing
  z_i. *Stopping rule:* stop when the max sampled reward exceeds every unopened box's z_i. z_i
  depends only on (F_i, c_i), not on history. His opening example shows the optimal order is not
  by expected value — high-variance options are probed first. z_i is decreasing in c_i
  (comparative statics analogous to CASSI's Property 2: higher cost weight ⇒ stop earlier).
- **Training / RL usage:** None (closed-form economics).
- **Experiments:** None (theory paper).
- **Key results:** One of the two canonical optimal-stopping-with-search-costs results (with
  Gittins indices); foundation of a large 2019–2024 algorithmic literature (see Beyhaghi & Cai
  2023 survey).
- **Limitations:** Requires known independent distributions; alternatives are exchangeable
  one-shot draws, *not* sequential improvement of one artifact (an agent trajectory is a
  correlated process, not independent boxes).
- **Relation to CASSI:** Lineage citation and the formal backbone of the *repeated-sampling* axis
  (Best-of-N) rather than CASSI's *trajectory-length* axis. Also a caution: Hay et al. prove
  index policies can *fail to exist* for metalevel problems, so CASSI should not describe its
  oracle as an index rule. **THREAT: LOW** — different problem structure; cite for lineage.

### Monitoring and Control of Anytime Algorithms: A Dynamic Programming Approach (Eric A. Hansen & Shlomo Zilberstein, 2001, Artificial Intelligence 126(1–2):139–157)
- **Read from:** PDF pages 1–19 (complete;
  `research/papers/hansen-zilberstein-2001_monitoring-anytime-algorithms.pdf`).
- **Problem:** When to stop an anytime algorithm (quality improves stochastically with time), and
  how often to monitor its progress, when monitoring itself costs.
- **Method/Theory:** The formal skeleton CASSI's oracle lives in. (1) *Performance profile*
  Pr(q_j|t); *time-dependent utility* U(q,t); Def. 3 **time-separable utility U(q,t) = U_I(q) −
  U_C(t)** (intrinsic value minus cost of time) — CASSI's `quality_t − λ·cumcost_t` is exactly
  this with linear U_C. (2) Def. 4 **optimal fixed allocation t\* = argmax_t Σ_i Pr(q_i|t)
  U(q_i,t)** — CASSI's oracle is the *realized, per-trajectory* (post-hoc, deterministic) version
  of this argmax. Their §3.6 example even uses U(q,t) = 100q − 20t. (3) Def. 6–7 **myopic
  stopping**: MEVC(Δt) = Σ_j Pr(q_j|q_i,Δt)U(q_j,t+Δt) − U(q_i,t); continue iff MEVC > 0 —
  CASSI's Δ(s_t) > 0 rule. (4) **Theorem 1 + Corollary 1**: the myopic rule is *optimal* when the
  expected marginal increase in intrinsic value is non-increasing in quality and the marginal cost
  of time is non-decreasing — i.e., a one-sided-crossing condition under diminishing returns +
  convex cost. This is, in expectation form, CASSI's Property 1 (uniqueness/monotone threshold
  crossing of t\*). (5) Def. 8 + Theorem 2: **non-myopic monitoring policy** π(q_i,t_k) →
  {stop, continue} computed *offline* by DP on V(q,t) (optimal under Markov quality improvement) —
  what Horvitz calls "compilation of metareasoning"; a learned stopping model is the amortized
  analog. (6) §3.4 **cost-sensitive monitoring**: DP that also chooses *whether/when to monitor*
  given monitoring cost C — the classical treatment of CASSI's "<3% monitor overhead" concern.
  (7) §3.5: when quality is not observable at run time, condition the policy on **features f
  correlated with quality**, Pr(q|f,t) — anticipating CASSI's stopper inputs (confidence,
  stability indicators) and noting predictions conditioned on (f, t) dominate those on t alone.
- **Training / RL usage:** None here (tabular DP from compiled profiles), but the conclusion
  explicitly points to Harada & Russell (1999): treat meta-level control as an MDP and solve by
  **reinforcement learning with value-function approximation** — the learned-stopper idea in 1999.
- **Experiments:** Lin-Kernighan TSP tour improvement; DP monitoring policy (expected value 303.3)
  beats optimal fixed allocation (269.2) even paying monitoring costs; value degrades gracefully
  as quality must be estimated from features (282.3).
- **Key results:** Optimal stopping + monitoring of an improving computation solved by DP;
  sufficient conditions for myopic-rule optimality; cost of monitoring internalized.
- **Limitations:** Tabular/discretized quality and time; needs a compiled performance profile per
  algorithm-domain pair; single quality scalar; no learning; no multi-dimensional budgets.
- **Relation to CASSI:** The single most important theory citation. Overlap: t\* argmax objective,
  Δ-rule, feature-conditioned monitoring, monitoring-cost accounting, even λ-style linear time
  cost. Differences: CASSI *learns* the monitor from post-hoc labels on its own trajectories,
  handles tool-using LLM agents, and feeds the monitor back as a training reward — none of which
  is here. **THREAT: HIGH** (to the formal-novelty claims, not the system) — Property 1 and the
  "dynamic monitoring beats static allocation" claim (H5's spirit) are essentially known results
  here; presenting them as new mathematics would be flagged by any senior reviewer.

### Selecting Computations: Theory and Applications (Nicholas Hay, Stuart Russell, David Tolpin, Solomon E. Shimony, 2012, UAI 2012; arXiv:1207.5879)
- **Read from:** PDF pages 1–6 of 10 (`research/papers/1207.5879_selecting-computations.pdf`).
- **Problem:** Rigorous foundation for metalevel decisions: which simulations/computations to run
  and when to stop, e.g., in Monte Carlo tree search.
- **Method/Theory:** **Metalevel probability model** (U_1..U_k, ℰ) and **metalevel MDP**: states =
  sequences of computation outcomes; actions = computations E (reward −c each) or terminate ⊥
  (reward max_i μ_i(s), the posterior-expected utility of the best object-level action).
  **Theorem 4: V^π(s) = E[−c·N + max_i μ_i(S_N)]** — expected quality at stopping minus
  cumulative computation cost, i.e., the population version of CASSI's per-trajectory objective.
  Theorem 5: optimal expected number of computations ≤ (value of perfect information)/c. Def. 6 +
  Theorem 7: the **myopic policy** (= R&W's meta-greedy with single-step assumption) stops no
  later than... precisely: if myopic computes in s, so does the optimal policy (myopic stops *too
  early*); Theorems 8–9: on transition-closed sets where myopic stops, optimal stops too.
  **Section 3 (non-indexability):** unlike bandits (Gittins), metalevel problems admit *no* index
  policy in general — context (the value of the fixed alternative λ, their notation) changes which
  computation is optimal in ways indices cannot express; also bandit algorithms (UCB/UCT) lack any
  native notion of stopping. Blinkered policy as a tractable middle ground.
- **Training / RL usage:** None (exact/DP analyses and closed forms for Bernoulli sampling).
- **Experiments:** Bernoulli selection problems (25 arms): blinkered policy dominates UCB1
  variants and myopic; Go/MCTS experiments in later sections.
- **Key results:** First finite sampling bounds for optimal metalevel policies; counterexample —
  an optimal metalevel policy may *compute forever* with positive probability; non-indexability.
- **Limitations:** i.i.d.-sample computations with known priors; small discrete problems; no
  learned components; object-level action fixed at stop time.
- **Relation to CASSI:** Supplies the correct formal container for CASSI's setting (agent steps =
  metalevel actions with cost; stopping = ⊥) and the exact quality−cost value function. Also a
  useful *defensive* citation: stopping in sequential computation is provably not a bandit/index
  problem, distinguishing CASSI from router/bandit baselines (Ares/SeqRoute). **THREAT: MEDIUM**
  — theory overlap with the oracle objective; no learned stopper, no LLMs.

### Learning to Select Computations (Frederick Callaway, Sayan Gul, Paul M. Krueger, Thomas L. Griffiths, Falk Lieder, 2018, UAI 2018; arXiv:1711.06892)
- **Read from:** PDF pages 1–4 of 12 (`research/papers/1711.06892_learning-to-select-computations.pdf`).
- **Problem:** Optimal metareasoning is intractable (Hay et al. 2012); can a *general, learned*
  policy approximate it — including deciding **when to terminate deliberation**?
- **Method/Theory:** **Bayesian Metalevel Policy Search (BMPS)** — "the first general approach to
  metacognitive RL." Metalevel MDP with uniform computation cost **cost(c) = λ** (their notation!)
  and r_meta(b,⊥) = max_π E[U_π(θ)]. Key insight: VOC(c,b) is sandwiched between the myopic value
  of information VOI_1(c,b) and the value of perfect information VPI(b); so approximate
  **V̂OC = w1·VOI_1 + w2·VPI + w3·VPI_sub − w4·cost(c)** (convex weights, w4 ∈ [1,h]) and fit w by
  direct policy search (Bayesian optimization on expected metalevel return). Policy: argmax_c
  V̂OC; terminate when best V̂OC ≤ 0. Meta-greedy (R&W) shown optimal only under diminishing
  returns — again the Property-1-style condition.
- **Training / RL usage:** Yes — metalevel policy *learned* from experience (direct policy
  search); no object-level training (the object level is fixed).
- **Experiments:** (i) when to stop deliberating (beta-Bernoulli evidence, horizon 30);
  (ii) allocating computation across options; (iii) planning (Mouselab MDP); emergency-management
  scenario including metareasoning overhead.
- **Key results:** Near-optimal on all three; beats meta-greedy and blinkered baselines; feature
  regression shows VOI_1/VPI features suffice to capture exact VOC.
- **Limitations:** Toy metalevel MDPs with known Bayesian belief updates; features hand-derived;
  linear scalarization; no deep networks, no language models, no training of the object level.
- **Relation to CASSI:** Direct precedent for "a *learned* model that estimates cost-aware value
  of continuing and decides termination." Differences: features not learned from trajectories, no
  post-hoc oracle labels, object level untouched (no PRM feedback loop). **THREAT: MEDIUM** —
  weakens any claim that *learning* the stopping decision is itself new; strengthens the case that
  CASSI's novelty is the label source (post-hoc oracle) + executor-training loop + LLM-agent scale.

### Learning to Stop While Learning to Predict (Xinshi Chen, Hanjun Dai, Yu Li, Xin Gao, Le Song, 2020, ICML 2020, PMLR 119:1520–1530; arXiv:2006.05082)
- **Read from:** PDF pages 1–8 of 15 (`research/papers/2006.05082_learning-to-stop-while-learning-to-predict.pdf`).
  (Note: task brief said "Xiao et al."; correct authors are Chen et al.)
- **Problem:** Algorithm-inspired deep models (unrolled ISTA, MAML, DnCNN…) use a fixed depth for
  every input; the optimal number of iterations is instance-dependent ("avoid over-thinking").
  Learn a per-instance stopping policy jointly with the predictive model.
- **Method/Theory:** Predictive model F_θ produces states x_1..x_T; stopping policy π_φ(x, x_t) ∈
  [0,1] halts sequentially. Variational-Bayes view: stop time t is a latent variable; β-VAE
  objective. **The two-stage trick (CASSI-relevant):** *Stage I (oracle):* for fixed θ the optimal
  stop distribution has closed form **q*_θ(t|y,x) ∝ exp(−ℓ(y,x_t;θ)/β)** — computed **post-hoc
  from a completed forward pass using the label y**, O(T) per instance; MAP variant **t̂(x,y) =
  argmax_t q*** (they report the argmax/MAP version trains better). Train θ against the oracle.
  *Stage II (imitation):* train π_φ (which sees only runtime features, no label) to match the
  oracle via forward KL = cross-entropy on oracle stop labels. They also derive the equivalence of
  reverse-KL imitation to **max-entropy RL** with reward −βℓ(y,x_t) on stop, 0 on continue. A
  non-uniform prior p(t|x) can penalize deeper (costlier) stopping, though experiments use uniform
  (no explicit λ·cost term).
- **Training / RL usage:** Supervised/variational training of a *separate stopping head/policy*
  from post-hoc oracle labels; RL connection shown but not used; the predictive model is trained
  *jointly with the oracle in the loop* (Stage I) — a static two-stage analog of CASSI's cycle.
- **Experiments:** Sparse recovery (LISTA-stop), task-imbalanced few-shot meta-learning
  (MAML-stop), image denoising (DnCNN-stop), Tiny-ImageNet early exit.
- **Key results:** LISTA-stop with ≤20 iterations beats ISTA/FISTA with 100; MAML-stop +3.5% on
  MiniImagenet; DnCNN-stop generalizes to unseen noise levels; oracle-based two-stage training
  beats joint variational (AEVB) training — mode-collapse of jointly-trained stop distributions.
- **Limitations:** Fixed-depth unrolled architectures, not open-ended agents; quality = supervised
  loss at every layer (dense labels); no explicit computation-cost term in the objective; Stage II
  imitation can fail when the oracle is hard to mimic (their image-recognition case).
- **Relation to CASSI:** **The closest ML precedent for CASSI's oracle-labeling trick** (question
  (c) answered: yes, post-hoc argmax labeling for training a stopper exists): compute optimal stop
  per instance after the run using ground truth, then distill into a runtime stopping policy.
  CASSI adds: explicit λ·cumulative-cost in the argmax, tokens/tools/dollars costs, LLM agent
  trajectories, GRPO on the stopper, and — the genuinely absent piece — using the stopper's value
  as a process reward to train the executor. **THREAT: HIGH** — if uncited, a reviewer can claim
  the "O(T) oracle labels" contribution is a known technique re-instantiated.

### Rational Metareasoning for Large Language Models (C. Nicolò De Sabbata, Theodore R. Sumers, Badr AlKhamissi, Antoine Bosselut, Thomas L. Griffiths, 2024, arXiv:2410.05563; v3 June 2025, preprint under review — also seen as OpenReview submission)
- **Read from:** PDF pages 1–9 of 19 (`research/papers/2410.05563_rational-metareasoning-llm.pdf`).
- **Problem:** CoT raises inference cost on *all* queries regardless of difficulty; train an LLM
  to reason only when beneficial.
- **Method/Theory:** **Directly imports Russell & Wefald's VOC into LLM training** (their Eq. 1 is
  the VOC definition). Reward for a reasoning chain z on input x with target y:
  **R_π(x,y,z) = U_π(z|x,y) − C(z)**, with utility U_π = log π_θ(y|z,x) − log π_θ(y|x)
  (likelihood gain of the answer given the chain — the policy itself is the reward model, no
  external RM) and cost **C(z) = γ·l(z)** (token count, γ=0.1). Verified: **yes, it uses a
  VOC-style reward penalizing compute inside an RL-style loop** — specifically **Expert
  Iteration** (their Algorithm 1): sample K=4 chains per question, compute advantages a_ik = r_ik
  − mean_k(r_ik), reject negative-advantage chains, SFT on survivors, iterate (5 iterations);
  incorrect-answer chains can also be discarded.
- **Training / RL usage:** Expert Iteration (rejection-sampling RL) on Llama-3.2-3B and
  Llama-3.1-8B; no PPO/GRPO (left as future work); single model — no separate monitor.
- **Experiments:** ARC, CommonsenseQA, GSM8K, ProofWriter (mixed 4×1024 training set); OOD on
  MMLU-CF; baselines: Direct/CoT few-shot, STaR, instruct models.
- **Key results:** 23–32% fewer output tokens than STaR (35–42% vs CoT few-shot) at equal or
  better accuracy; adaptivity: hard/easy length gap grows (up to 50.3% length reduction on easy
  split); OOD: 28–36% fewer tokens than STaR on MMLU-CF at similar accuracy.
- **Limitations (authors' own §7):** **not agentic** — "adapting our method to this context would
  require incorporating the cost of tool use (e.g., API calls) into the reward function"; Expert
  Iteration only; trajectory-level reward (whole chain scored once — no per-step stop/continue
  decision, no mid-trajectory stopping, no value margin); no controller at inference.
- **Relation to CASSI:** The seed question (a) is answered: **yes, rational metareasoning/VOC has
  been applied to LLM training with a compute-penalizing reward** — but to single-turn CoT with a
  single model and no trained stopping policy. Overlap: cost-aware training signal grounded in
  VOC; "reasoning adapts to difficulty" claim (CASSI's H5 analog exists here at chain level).
  Differences: no separate stopper, no per-step Δ, no oracle t\* labels, no process rewards, no
  tools/budgets, no cycle. **THREAT: HIGH** — the single biggest framing threat in this area:
  CASSI's "LLMs have no training signal for 'good enough, stop now'" motivation and any "first
  cost-aware training" phrasing are falsified by this paper; CASSI must claim the *loop* and the
  *agentic per-step stopping*, not cost-aware training per se. It also hands CASSI a ready-made
  baseline (VOC-reward single-model training ≈ CASSI's "single-model GRPO+cost penalty" arm).

### Optimal Stopping vs Best-of-N for Inference Time Optimization (Yusuf Kalayci, Vinod Raman, Shaddin Dughmi, 2025, arXiv:2510.01394)
- **Read from:** PDF pages 1–7 of 24 (`research/papers/2510.01394_optimal-stopping-vs-best-of-n.pdf`).
- **Problem:** Best-of-N wastes generations (N fixed in advance). Decide *adaptively when to stop
  sampling* candidate responses, without knowing the reward distribution.
- **Method/Theory:** Frames repeated LLM generation as **Weitzman's Pandora's box** with
  i.i.d. boxes: each generation = opening a box at cost c with reward from unknown D. Weitzman's
  optimal rule stops when max observed reward ≥ fair-cap value τ solving E[(v−τ)+] = c. Their
  **UCB Pandora's Box** maintains anytime-valid upper confidence bounds τ+ on τ from samples and
  stops when max reward exceeds it; Theorem 5 bounds the additive suboptimality gap vs Weitzman
  with known D; instantiated for exponential tails (gap Õ(1/λ_rate)). Practical layer:
  Bradley–Terry acceptance-rate transformation to normalize rewards across prompts; stopping
  thresholds learned online per prompt.
- **Training / RL usage:** None — an online statistical stopping rule; reward model assumed given;
  no model is trained.
- **Experiments:** AlpacaFarm and HH-RLHF, multiple LLM–RM pairs; matches non-adaptive Best-of-N
  reward with **15–35% fewer generations**.
- **Key results:** First stopping strategy for Pandora's box with unknown distributions
  (vanishing regret to Weitzman); principled bridge "optimal stopping theory ↔ inference-time
  scaling."
- **Limitations:** Parallel-sampling axis only (independent draws; no notion of a trajectory
  improving over steps); needs a numeric reward model at inference; i.i.d. box assumption.
- **Relation to CASSI:** Proves the "optimal stopping for LLM inference" territory is being
  actively claimed (Oct 2025), but on an orthogonal axis (how many samples) with no learning and
  no agents. Useful citation to delimit CASSI's axis (sequential steps of one trajectory).
  **THREAT: MEDIUM** — a reviewer may ask why CASSI doesn't compare against adaptive-sampling
  stopping; conceptual overlap in "stop when marginal value < cost."

### Inference-Time Budget Control for LLM Search Agents (Zhengru Fang, Senkang Hu, Zhonghao Chang, Yu Guo, Yihang Tao, Hongyao Liu, Mengzhe Ruan, Jun Huang, Yuguang Fang, 2026, arXiv:2605.05701, CityU HK / Tsinghua / Ant Group)
- **Read from:** PDF pages 1–8 of 36 (`research/papers/2605.05701_inference-time-budget-control-voi.pdf`).
- **Problem:** LLM search agents under hard dual budgets (tool calls B_tool, output tokens B_tok):
  which action should get the next budget unit, and when is evidence sufficient to commit an
  answer?
- **Method/Theory:** Two-stage **training-free** controller on a tree-search (BAVT-style)
  backbone. Stage 1: each feasible action k ∈ {SEARCH, DECOMPOSE, ANSWER} gets a **task-level VOI
  score** r_t(k) = [u_t(k)]+/(d_t(k;b_t)+ε) — marginal task value per unit budget — with utility
  u_t(k) = Δ̂_t(k) (critic-derived progress signal) + Ψ_t(k) (structural signals) − Π_t(k;b_t) (a
  **budget-dependent penalty serving as a tractable proxy for the oracle budget shadow-cost
  λ*ᵀg_t(k)**); deterministic guards; pick argmax. Explicitly: "it is not Shannon information
  gain, Bayesian posterior value, **or a learned value model**" — fixed coefficients over explicit
  features. Stage 2: risk-controlled answer finalization F(z) = G(z) − ηH(z) ≥ τ with a safe set
  (rewrite only on low-risk answer-form errors). Appendix theory: the utility locally approximates
  an oracle budget-charged one-step lookahead value (ranking consistency under a margin
  condition); no global optimality claim.
- **Training / RL usage:** None — training-free by design; no executor training, no learned
  stopper.
- **Experiments:** HotpotQA, 2WikiMultihopQA, MuSiQue, Bamboogle; Qwen3-32B, Qwen3.5-122B,
  GPT-5.4-Mini; four dual-budget levels; baselines BAVT, BATS, AFlow, Search-o1 under a hard
  budget audit.
- **Key results:** Best-F1 in 7/16 (+2 tied) cells on Qwen3-32B; beats BATS in 10/16, BAVT in
  15/16; biggest gains at low budgets; ablations: the budget-dependent penalty is the dominant
  component; explicitly "positive but not a dominance claim" (BATS strong at high budgets).
- **Limitations:** Hand-designed features/coefficients per action taxonomy; QA-only; critic is a
  prompted signal, not calibrated; backbone-dependent (mixed results on Qwen3.5-122B); answers
  committed by threshold rules.
- **Relation to CASSI:** Same task family (multi-hop QA search agents; MuSiQue/HotpotQA overlap
  with CASSI's benchmarks), same vocabulary (VOI, marginal value per budget, shadow price λ*),
  same decision points (continue vs commit). But it is exactly what CASSI's contribution 1 says
  is missing: no learning anywhere — no oracle labels, no trained stopper, no process rewards, no
  executor RL. **THREAT: MEDIUM-HIGH** — as of May 2026 the "cost-aware per-step controller for
  search agents" exists in training-free form; CASSI must cite it, likely as a baseline
  (BATS/BAVT-family), and its novelty must rest on *learning* the controller and closing the
  training loop, not on having a controller at all.

## Peripheral papers

- **Zilberstein (1996), "Using Anytime Algorithms in Intelligent Systems," AI Magazine 17(3);
  Zilberstein & Russell (1996), "Optimal Composition of Real-Time Systems," AIJ 82(1–2):181–213.**
  (Abstract/secondary; verified via Hansen & Zilberstein's reference list.) The anytime-algorithm
  program: performance profiles, contract vs interruptible algorithms, compilation of composed
  anytime systems. Establishes the quality-vs-time trade-off vocabulary CASSI inherits; cite
  alongside Dean & Boddy (1988) "An Analysis of Time-Dependent Planning" (AAAI-88), which
  introduced deliberation scheduling, and Horvitz (1987/1990) for EVC/flexible computation. LOW
  threat, expected citations.

- **Harada & Russell (1998/99), "Learning Search Strategies" (AAAI Spring Symp. extended
  abstract).** (Secondary, via Hansen & Zilberstein §5 and Callaway §2.3.) Earliest proposal to
  treat meta-level control as an MDP and solve it with RL + value-function approximation — the
  "learned stopping policy" idea predates deep learning by two decades. LOW threat; one-line
  lineage citation.

- **Svegliato, Wray & Zilberstein (2018), "Meta-Level Control of Anytime Algorithms with Online
  Performance Prediction," IJCAI 2018; Svegliato, Sharma & Zilberstein (2020), "A Model-Free
  Approach to Meta-Level Control of Anytime Algorithms," ICRA 2020.** (Abstracts + search;
  ijcai.org/proceedings/2018/208.) Learn when to stop an anytime algorithm *online, without a
  precompiled performance profile* — the ICRA version uses model-free RL over (quality, time)
  states. Modern learned instantiation of Hansen–Zilberstein monitoring; also Bhatia, Svegliato &
  Zilberstein (ICAPS 2022) apply deep RL to metareasoning about anytime planners, and "Stop!
  Planner Time" (AAAI 2024) continues the line. MEDIUM-LOW threat: learned stopping monitors for
  computational processes exist; not LLMs, not label-based, no executor training.

- **Gershman, Horvitz & Tenenbaum (2015), "Computational Rationality: A Converging Paradigm for
  Intelligence in Brains, Minds, and Machines," Science 349(6245):273–278.** (Abstract +
  secondary; paywalled.) Position piece unifying AI/cogsci/neuroscience around maximizing expected
  utility *net of computation costs*: meta-level architectures, EVC, bounded optimality. The
  standard umbrella citation for CASSI's "economic judgment" framing. LOW threat.

- **Lieder & Griffiths (2020), "Resource-Rational Analysis," Behavioral and Brain Sciences 43:e1.**
  (Abstract + secondary.) Methodology: model cognition as the optimal use of *limited*
  computational resources — derive the algorithm that optimally trades resource cost against
  accuracy, compare to behavior, iterate. Parent framework of Callaway's metacognitive RL and of
  De Sabbata's LLM work (Griffiths is senior author on both). LOW threat; framing citation.
  Related 2026 curiosity: "Humans Disengage, Reasoning Models Persist" (arXiv:2606.26502) argues
  LRMs lack the human coupling between difficulty registration and deliberation allocation —
  empirical motivation for CASSI-style external stopping.

- **Goldstein, McAfee, Suri & Wright (2020), "Learning When to Stop Searching," Management Science
  66(3):1375–1394.** (Abstract + author PDF page skim.) Repeated secretary problem with humans:
  with repeated exposure to a fixed distribution people learn near-optimal *threshold* stopping
  rules. Behavioral, not ML — relevant only as evidence that threshold policies are learnable from
  experience in optimal-stopping tasks. LOW threat. (The classical secretary problem itself —
  stop-after-n/e — is a *rank-based, no-recall* stopping problem, structurally unlike CASSI's.)

- **Becker, Cheridito & Jentzsch (2019), "Deep Optimal Stopping," JMLR 20(74):1–25.** (Verified
  via Chen et al. 2020's related-work discussion + search.) Learns stopping decisions for
  high-dimensional stochastic processes (Bermudan option pricing) by backward recursion over
  per-step stop/continue networks trained on simulated paths — the canonical deep-learning
  optimal-stopping method; with Herrera et al. (2021) "Optimal Stopping via Randomized Neural
  Networks" and the RL treatments (e.g., arXiv:2105.08877) it forms the "optimal stopping with
  deep RL" cluster (2019–2024). Finance-process setting, dense simulable rewards, no agents/LLMs.
  LOW-MEDIUM threat: shows per-step learned stop/continue value networks are standard elsewhere.

- **Huang, Ma & Zhou (2026), "Optimal Bayesian Stopping for Efficient Inference of Consistent LLM
  Answers," arXiv:2602.05395.** (Abstract via WebFetch.) Bayesian optimal stopping for
  self-consistency sampling: track only the top L−1 answer counts ("L-aggregated" policy; L=3
  suffices for asymptotic optimality), stop when posterior consistency is sufficient; up to ~50%
  fewer LLM calls, nothing trained. Sampling-axis sibling of Kalayci et al. LOW threat.

- **Bilal et al. (2026), "What If We Allocate Test-Time Compute Adaptively?" arXiv:2602.01070; and
  "Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs"
  (arXiv:2507.02076).** (Abstracts.) The former: verifier/PRM-guided adaptive allocation across
  iterative trajectories (inference-time control, PRM as signal — no trained stopper, no cost
  term in training). The survey is the field map for adaptive test-time compute and notes that
  rational-metareasoning-style utility is usually "collapsed into a uniform accuracy objective" —
  a quotable gap statement aligned with CASSI. LOW threat, useful citations.

- **Oh & Gobet (2025), "Monitor-Generate-Verify: Formalising Metacognitive Theory for Language
  Model Reasoning," NeurIPS 2025 Workshop on Foundations of Reasoning in LMs (arXiv:2511.04341).**
  (Abstract via WebFetch; already in CASSI's related-work list as "MGV.") Position paper
  translating Flavell / Nelson–Narens metacognition into a Monitor→Generate→Verify architecture;
  **no empirical validation, nothing trained**, does not engage the VOC formalism. Confirms the
  monitor-architecture idea is in the air; CASSI's trained, cost-aware monitor goes beyond it.
  LOW threat.

- **"Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning"
  (arXiv:2607.05458, July 2026).** (Abstract via WebFetch.) Trains a lightweight controller over
  "structural execution actions" of a frozen LLM's harness via offline RL (advantage-weighted
  regression) from historical rollouts with terminal task reward; improves verification behavior
  on tau-bench retail / AgentBench DB. Stopping decisions not explicitly among the controlled
  actions; no cost-aware objective, no oracle labels, no executor training. Adjacent evidence
  that "small learned controller supervises big frozen LLM" is emerging *now*. MEDIUM-LOW threat
  — watch this line; a follow-up adding cost-aware stopping would collide with CASSI.
  Relatedly, the multi-agent orchestration-traces line (arXiv:2605.02801) explicitly notes that
  as of May 2026 **no curated method trains the stopping decision itself by RL** — stop signals
  come from verifiers or step caps — which independently corroborates CASSI's gap.

## Synthesis

### Landscape

Three generations answer "when should computation stop?": (1) *Analytical* (1979–2012): Weitzman's
indices for independent alternatives; Horvitz/Russell–Wefald VOC and myopic control;
Hansen–Zilberstein DP monitoring of anytime processes; Hay et al.'s metalevel MDPs with
quality-minus-cost value functions and non-indexability. (2) *Learned metalevel control*
(1999–2022): Harada–Russell RL proposal; Callaway's BMPS (learned VOC features, includes
termination); Svegliato's model-free stopping monitors; deep optimal stopping in finance; Chen et
al.'s oracle-then-imitate stopping for unrolled deep models — the direct ancestor of post-hoc
argmax stop-labeling. (3) *LLM era* (2024–2026): De Sabbata's VOC reward inside Expert Iteration
(single model, single-turn); training-free stopping controllers — Pandora's-box sampling rules
(Kalayci; Huang) on the samples axis, VOI budget controllers (Fang; BATS/BAVT) on the agent-steps
axis; learned harness controllers without cost objectives (2607.05458). Across all three
generations, no work trains a stopping model on cost-aware post-hoc oracle labels over LLM-agent
trajectories, and none feeds a stopping/value model back as a process reward to train the
executor. Two independent 2026 sources describe RL-trained stopping for LLM agents/orchestration
as an open problem.

### Mapping table: CASSI concept ↔ classical construct ↔ source

| CASSI concept | Classical construct | Source |
|---|---|---|
| Oracle `t* = argmax_t [quality_t − λ·cumcost_t]` | Optimal (fixed) allocation `t* = argmax_t Σ_i Pr(q_i|t)U(q_i,t)` under time-separable `U = U_I(q) − U_C(t)`; realized per-trajectory version | Hansen & Zilberstein 2001, Defs. 3–4 (their §3.6 example: `U = 100q − 20t`); Simon 1958 ("net of computational costs") |
| Trajectory objective (quality at stop minus summed step costs) | Metalevel MDP value `V^π = E[−c·N + max_i μ_i(S_N)]` | Hay et al. 2012, Thm. 4 |
| Margin `Δ(s_t) = Q_continue − Q_stop`; stop iff Δ < 0 | (Myopic) expected value of computation; `MEVC(Δt) > 0` ⇒ continue; VOC(c,b) > 0 ⇒ compute | Horvitz 1987/90 (EVC); Russell & Wefald 1991 (VOC, meta-greedy); H&Z 2001 Defs. 6–7 |
| λ and budget-tier multipliers (HIGH/MED/LOW/CRITICAL) | Cost of time `U_C(t)` / per-computation cost λ / budget shadow price λ* | H&Z 2001 Def. 3; Callaway et al. 2018 (cost(c)=λ); Fang et al. 2026 (Π_t as shadow-cost proxy) |
| Small stopping model M_θ as runtime controller | Compiled (DP) monitoring policy π(q_i,t_k) ∈ {stop, continue}; "compilation of metareasoning"; learned metalevel policy | H&Z 2001 Def. 8, Thm. 2; Horvitz; Callaway 2018 (BMPS); Svegliato 2018/2020 |
| Post-hoc oracle STOP/CONTINUE labels + margin from finished trajectories, O(T) | Closed-form oracle stop distribution `q*(t|y,x) ∝ exp(−ℓ(y,x_t)/β)` computed after the forward pass with labels; MAP `t̂ = argmax_t`; then imitation | Chen et al. 2020 (ICML), Stages I–II |
| Stopper input features (confidence, stability, budget state) when true quality unobservable | Monitoring conditioned on run-time features `f` correlated with quality, `Pr(q|f,t,Δt)` | H&Z 2001 §3.5 |
| "<3% monitor overhead" | Cost-sensitive monitoring: monitoring cost C inside the DP; monitor only when worth it | H&Z 2001 §3.4, Def. 9, Thm. 3 |
| Two-model split (executor vs stopper) | Object level vs meta level | Russell & Wefald 1991; Cox & Raja 2011 (*Metareasoning: Thinking about Thinking*) |
| Cost-aware reward for LLM training | `R = U_π(z|x,y) − γ·l(z)` (VOC reward) inside Expert Iteration | De Sabbata et al. 2024/25 |
| Adaptive per-instance stopping beats static budgets (H5) | Run-time monitoring beats optimal fixed allocation (269.2 → 303.3 in their TSP example) | H&Z 2001 §2 vs §3; Dean & Boddy 1988 |
| Stop-sampling axis (not CASSI's, but adjacent) | Pandora's box reservation price / fair-cap `E[(v−τ)+] = c` | Weitzman 1979; Kalayci et al. 2025; Huang et al. 2026 |

### Are CASSI's Properties 1–3 already known math?

- **Property 1 (existence/uniqueness–monotone crossing of t\*):** substantially known *in
  expectation form*. H&Z Theorem 1 + Corollary 1 give exactly the one-sided-crossing condition —
  myopic stopping is optimal iff once MEVC goes non-positive it stays non-positive, guaranteed by
  non-increasing marginal intrinsic value (diminishing returns of quality) + non-decreasing
  marginal cost of time. Callaway et al. repeat it ("meta-greedy optimal under diminishing
  returns"); Hay et al. give the myopic-vs-optimal stopping ordering (Thms. 7–9). CASSI's
  realized, per-trajectory version (observed quality_t need not be concave; argmax of a finite
  sequence trivially exists, uniqueness needs tie-breaking) is a small delta. Presenting Property
  1 as a new theorem invites a "known since 2001" review comment; present it as a *transfer* of
  H&Z's condition to realized trajectories.
- **Property 2 (λ ↑ ⇒ t\* earlier):** standard monotone comparative statics of `argmax_t [q_t −
  λc_t]` with c_t increasing in t (Topkis-style single-crossing argument); the same monotonicity
  appears as reservation prices decreasing in opening cost in Weitzman 1979 and steeper `U_C`
  shifting t\* left in the anytime literature. Not novel as mathematics; fine as a sanity lemma.
- **Property 3 (oracle quality improves as the executor policy improves — the "cycle"
  property):** the least classical of the three. Closest antecedents: Chen et al.'s Stage I oracle
  `q*_θ` explicitly depends on and co-improves with θ, and Expert-Iteration/STaR-style
  policy-improvement arguments (De Sabbata iterate reward-filtered distillation five times). No
  prior work states or proves a monotone-improvement property for a *cost-aware stopping oracle
  coupled to executor RL* — this is where genuinely new formal work is possible, and also where a
  rigorous statement is hardest (the claim needs assumptions about off-policy drift of the
  stopper as the executor's trajectory distribution shifts).

### Gaps CASSI can legitimately claim

1. No prior work computes **cost-aware (quality − λ·cumcost) stopping labels post-hoc from LLM
   agent trajectories** and trains a separate stopping/value model on them (Chen et al. = the
   trick without cost or agents; H&Z = the objective without learning).
2. No prior work **feeds a learned stopping margin back as a process reward for executor RL** —
   classical metareasoning never trains the object level; RaM trains the object level but has no
   meta-model; training-free VOI controllers train nothing. The "self-reinforcing cycle" is
   unclaimed territory (corroborated by 2026 orchestration-RL sources naming trained stopping an
   open problem).
3. Amortizing metareasoning into a **small LM supervising a large LM** with quantified overhead —
   the cost-sensitive-monitoring question (H&Z §3.4) has no LLM-era learned instantiation.
4. Multi-dimensional budget state (tokens/tools/dollars + tiers) vs the single time axis of the
   classical theory.

### Top threats (ranked)

1. **De Sabbata et al. 2024/25 (RaM)** — HIGH. VOC-based, compute-penalizing reward already used
   to train LLMs; kills "first cost-aware training signal" phrasing; senior-author overlap with
   the metareasoning school makes an uncited collision embarrassing. CASSI survives by claiming
   the agentic per-step stopping + separate stopper + executor-PRM loop (all absent in RaM, and
   RaM's own limitations section calls the agentic extension future work).
2. **Chen et al. 2020 (ICML)** — HIGH. Post-hoc oracle argmax labeling + imitation for a learned
   stopping policy is a known ML technique; CASSI's "O(T) oracle labels, zero extra rollouts"
   contribution must be framed as *extending* it (explicit λ·cost, agent trajectories, RL
   fine-tuning, reward-bridge), not inventing it.
3. **Hansen & Zilberstein 2001 (+ Hay et al. 2012)** — HIGH for formal claims, LOW for the
   system. t\*-argmax, Δ>0 rule, feature-conditioned monitoring, monitoring-cost accounting, and
   Property-1-type conditions are all here; CASSI's Properties 1–2 are transfers, not theorems.
4. **Fang et al. 2026 (VOI budget control)** — MEDIUM-HIGH. Same benchmarks, same per-step
   cost-aware value scoring, May 2026 — but deliberately training-free; must be cited and ideally
   beaten as a baseline (it is BAVT-adjacent, already in CASSI's baseline list).
5. **Kalayci et al. 2025 / Huang et al. 2026** — MEDIUM. "Optimal stopping for LLM inference" is
   being claimed on the sampling axis; CASSI should explicitly scope its axis (sequential
   trajectory steps) and cite these to preempt "isn't this just Pandora's box?"
6. **Callaway 2018 / Svegliato 2018–2020** — MEDIUM. Learned termination policies exist outside
   LLMs; weakens "learning to stop is new," strengthens "no one did it for LLM agents with
   post-hoc labels + executor feedback."
7. **2607.05458 (harness offline RL, July 2026)** — MEDIUM-LOW now, rising. Nearest active
   trajectory toward a colliding result; monitor for v2s adding cost-aware stopping.

### What CASSI must cite to survive theory-aware reviewers

Minimum set: Simon (1955/1958) and Good (1968) one-liners; Horvitz (1987, 1990 thesis — EVC,
flexible computation, compilation of metareasoning); **Russell & Wefald (1991, AIJ + book)**;
Dean & Boddy (1988) + Zilberstein (1996) for anytime algorithms; **Hansen & Zilberstein (2001)**
when stating the oracle and Properties 1–2 (attribute the argmax construct and the myopic-rule
optimality conditions); **Hay et al. (2012)** for the metalevel-MDP formalization and
non-indexability (also useful against bandit-baseline confusion); Callaway et al. (2018) +
Svegliato et al. (2018/2020) for learned metalevel/stopping policies; **Chen et al. (ICML 2020)**
when introducing post-hoc oracle labels; Becker et al. (2019) for deep optimal stopping;
Weitzman (1979) + Kalayci et al. (2025) + Huang et al. (2026) to scope the sampling axis;
**De Sabbata et al. (2024/25)** as the closest LLM-training prior art and a baseline; Fang et
al. (2026) as the training-free agent-controller counterpart; Gershman et al. (2015) and Lieder &
Griffiths (2020) for framing. Recommended narrative: "CASSI operationalizes the
Hansen–Zilberstein/Russell–Wefald stopping problem for LLM agents by *learning* the monitor from
post-hoc realized-VOC labels (à la Chen et al.) and, unlike all prior metareasoning work, closes
the loop by using the monitor's VOC estimate as a process reward to train the object level."
