# Adaptive Test-Time Compute, Difficulty-Aware Budgeting & Routing

> Area researcher notes for CASSI (contribution #4: per-instance dynamic cost adaptation; also touches #5,
> small controller supervising large executor). Researched 2026-07-16. All core papers read from PDFs in
> `research/papers/`. Venues verified against PDF headers / ACL Anthology / ICLR proceedings where possible.

## Area overview

This literature answers one question in many guises: **how much compute should THIS problem get?**
It began with upfront, single-shot decisions: model cascades (FrugalGPT, 2023; Mixture-of-Thoughts
cascades, ICLR 2024) and preference-trained routers (RouteLLM, ICLR 2025) that pick a cheap or
expensive model per query; Snell et al. (ICLR 2025 Oral) formalized the "compute-optimal" view —
the best test-time strategy (revisions vs. parallel search, search depth) varies with question
difficulty, and binning difficulty yields ~4x efficiency over best-of-N. A second wave (2025) moved
the knob inside a single model: small learned switchers choose short vs. long CoT per query
(ThinkSwitcher, EMNLP 2025 Findings; Thinkless; Adaptive Deep Reasoning), budget predictors
estimate token allowances before reasoning (TALE, SelfBudgeter, Plan-and-Budget at ICLR 2026),
and joint model+strategy routers optimize accuracy-per-token (Route-to-Reason). Almost all of this
wave is **upfront**: difficulty is estimated once, before generation, and the allocation is fixed.

The 2026 wave — the one that matters most for CASSI — moves adaptation **mid-trajectory and into
agents**. Ares (arXiv 2603.07915) trains a 1.7B router with SFT+GRPO to pick a discrete reasoning-effort
level for a frozen gpt-oss-20b agent **at every step** of TAU-Bench/WebArena trajectories, using
hindsight "minimum sufficient effort" labels mined from completed trajectories. BAAR (arXiv
2602.21227, Microsoft) trains a per-step model router (small vs. large executor) for 20+-step agent
tasks with cheapest-successful-trajectory SFT labels plus GRPO under soft and hard budget
constraints. SeqRoute (arXiv 2605.25424) shows **hindsight budget relabeling** by name: it replays
unconstrained conversation logs under hypothetical budgets to synthesize bankruptcy-annotated
transitions, trains a conservative Q-function offline, and sweeps a Lagrangian λ at inference to trace
the cost-quality Pareto frontier zero-shot. Training-free variants exist too (TrACE per-step sample
allocation via action agreement; Dynasor/Certaindex certainty-based token allocation; VoI-based
budget control for search agents).

The clear structural gap: **every method in this area controls a frozen executor**. Routers,
switchers, budget predictors, and controllers allocate compute *around* the policy; none of them
feeds its cost-aware value estimates back as process rewards to *train* the executor, and none
makes learned economic stop/continue decisions (they choose effort levels or models, not
termination). That closed loop — controller-as-PRM training the executor — is CASSI's remaining
white space; "per-instance adaptation" and "small controller supervises large executor" on their
own are no longer novel claims.

## Core papers

### Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters (Snell, Lee, Xu, Kumar; 2024; ICLR 2025 Oral; arXiv 2408.03314)

- **Read from:** PDF pages 1–6 (`2408.03314_snell-test-time-compute.pdf`)
- **Problem:** Given a fixed inference budget, how should test-time compute be spent (parallel best-of-N vs. sequential revisions vs. PRM tree search), and can it substitute for pretraining scale?
- **Method:** Defines the "test-time compute-optimal scaling strategy" θ*(q, N) = argmax_θ E[correctness] per prompt q and budget N (their Eq. 1). Difficulty is the allocator: questions binned into 5 difficulty levels by base-model pass@1 over 2048 samples ("oracle difficulty") or by averaged verifier scores over the same samples ("model-predicted difficulty"); the best strategy/hyperparameters are picked per bin on a validation fold. Allocation is **upfront, per-prompt, discrete (5 bins), and selected by lookup** — no learned controller network. Two mechanisms studied: PRM-guided search (best-of-N weighted, beam, lookahead) and sequential self-revisions.
- **Training / RL usage:** PRM trained without human labels (MC estimation per Wang et al.); revision model fine-tuned on constructed multi-attempt data. No RL for allocation itself.
- **Experiments & benchmarks:** MATH (500-question Lightman et al. split), PaLM 2-S* (Codey) base.
- **Key results:** Compute-optimal allocation beats best-of-N with **~4x less compute** (both revisions and search). FLOPs-matched comparison: test-time compute beats a **~14x larger** pretrained model on easy/medium questions (but pretraining wins on the hardest questions / high inference-to-pretraining token ratios).
- **Limitations:** Single-turn math only; difficulty estimation itself needs 2048 samples (cost acknowledged but not charged); bins not a deployable controller; no notion of stopping or sequential budget state.
- **Relation to CASSI:** The canonical citation for "harder problems deserve more compute" — CASSI's H5 (stopping step correlates with difficulty) is this paper's thesis transplanted to agent stopping. No overlap in mechanism (no stopping model, no executor training, not agentic). **THREAT LEVEL: LOW-MEDIUM** — motivational prior art that reviewers will expect cited, not a competing method.

### RouteLLM: Learning to Route LLMs with Preference Data (Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous, Stoica; 2024; ICLR 2025; arXiv 2406.18665)

- **Read from:** PDF pages 1–3 (`2406.18665_routellm.pdf`)
- **Problem:** Per-query binary routing between a strong model (GPT-4) and a weak model (Mixtral-8x7B) to minimize cost at a quality target.
- **Method:** A **win-prediction model** P(win_strong | q) trained on 80k Chatbot Arena human-preference battles (+ augmentation with MMLU golden labels and GPT-4-judge labels), thresholded by α to make the route. Four router parameterizations: similarity-weighted Elo ranking, matrix factorization, BERT classifier, causal-LLM classifier. **Upfront, one-shot per query; the router is small and separate from the answering models; no budget state, no sequential reasoning.**
- **Training / RL usage:** Supervised (MLE on preference labels). No RL.
- **Experiments & benchmarks:** MT-Bench, MMLU, GSM8K; metrics CPT (call-performance threshold) and APGR. Generalization to unseen model pairs (Claude 3 Opus / Llama-3-8B).
- **Key results:** Cost reductions **>2x** without substantial quality loss (e.g., large drops in %-GPT-4-calls needed to reach 95% of GPT-4 quality on MT-Bench); routers transfer across model pairs without retraining.
- **Limitations:** Single-turn, two-way, quality-only labels (no token/length cost modeling); no difficulty adaptation within a response; assumes preference data availability.
- **Relation to CASSI:** Establishes "small learned router allocates expensive compute per instance" as a well-known paradigm — CASSI's novelty cannot rest on that framing. Zero overlap with stopping, process rewards, or executor training. **THREAT LEVEL: LOW** — different problem (model selection), but the reference point every reviewer knows.

### ThinkSwitcher: When to Think Hard, When to Think Fast (Liang, Zhong, Yang, Quan; 2025; EMNLP 2025 Findings; arXiv 2505.14183)

- **Read from:** PDF pages 1–5 (`2505.14183_thinkswitcher.pdf`)
- **Problem:** One LRM overthinks easy queries; deploying two models (reasoner + fast model) is wasteful. Can a single LRM switch between short-CoT and long-CoT modes per query?
- **Method:** Observation: an empty `<think></think>` block reliably elicits short CoT from R1-distill models. A **lightweight switcher (regression head on the LRM's query embedding)** predicts the empirical pass rate of each mode, ŷ_SC and ŷ_LC; choose long CoT iff ŷ_LC − ŷ_SC ≥ τ. **Upfront, per-query, binary decision, continuous pass-rate outputs; controller separate from (and much smaller than) the frozen backbone.**
- **Training / RL usage:** Self-supervised regression: for each training query sample k responses per mode, empirical pass rates become targets; MSE + margin-aware loss on the predicted SC/LC difference. No RL.
- **Experiments & benchmarks:** DeepSeek-R1-Distill-Qwen-1.5B/7B/14B; GSM8K, MATH-500, AIME, LiveAoPSBench, Omni-MATH, OlympiadBench; baselines SC-only, LC-only, random switcher, ModernBERT router.
- **Key results:** **20–30% average token reduction** at competitive accuracy; GSM8K ~30% fewer tokens with <1% loss; AIME 38% reduction with ~2% loss. Their Fig. 2: prompt-induced short CoT alone cuts tokens 75–84% on MATH-500 with accuracy dropping 91.7→78.9 (R1-7B) — motivating selective use.
- **Limitations:** Binary (two modes); single-turn math; label construction needs k samples per mode per query (sampling-heavy hindsight labels); threshold τ hand-tuned per deployment; no budget state.
- **Relation to CASSI:** Closest single-model analogue of a "small learned difficulty-aware controller": trained on hindsight outcome statistics of completed generations — rhetorically adjacent to CASSI's oracle labels, but labels cost k extra rollouts per query per mode (vs. CASSI's O(T) claim). No stopping mid-trajectory, no executor training, no cost term in the objective beyond mode choice. **THREAT LEVEL: MEDIUM** — undercuts "novel small controller predicts compute needs per instance"; leaves stopping/agents/executor-RL open.

### Plan and Budget: Effective and Efficient Test-Time Scaling on LLM Reasoning (Lin, Zeng, Zhu, Wang, Shun, Wu, Zhou; 2025→2026; **ICLR 2026**; arXiv 2505.16122)

- **Read from:** PDF pages 1–6, v3 (`2505.16122_plan-and-budget.pdf`; header "Published as a conference paper at ICLR 2026")
- **Problem:** "Reasoning miscalibration": overthinking on trivial-but-ambiguous queries and underthinking under hard token caps. How to allocate a per-query token budget across reasoning stages by uncertainty?
- **Method:** Theory: **BAM (Budget Allocation Model)** — reasoning is a sequence of sub-questions; epistemic uncertainty decays as inverse power law c_ij/b_ij^β; Lagrangian solution (their Eq. 6) gives unimodal allocation — moderately hard sub-questions get the most tokens, hopeless ones get few. Practice: **Plan-and-Budget** — a lightweight planner LLM decomposes the query into sub-questions with complexity scores; the global budget B_i is split by normalized complexity weights and **decay-based schedules** (polynomial/cosine) that front-load tokens to early, high-uncertainty stages; budgets enforced via prompts. Introduces **E3 = A²/T** metric. **Upfront estimation + fixed within-trajectory schedule; training-free; model-agnostic; no learned controller.**
- **Training / RL usage:** None (test-time framework only).
- **Experiments & benchmarks:** DS-R1-Qwen-32B, QwQ-32B, DS-R1-LLaMA-70B, o4-mini; math reasoning, instruction following, **agentic planning (NaturalPlan-style tasks)**.
- **Key results:** Up to **+70% accuracy**, **−39% tokens**, **+193.8% E3**; DS-Qwen-32B E3 0.16→0.47, nearly closing the gap to DS-LLaMA-70B (0.50) without retraining.
- **Limitations:** Requires an extra planner LLM call; decay schedules are heuristic surrogates for BAM (uncertainty parameters unobservable); budget adherence via prompting is soft; no feedback from actual mid-trajectory progress.
- **Relation to CASSI:** The strongest published (ICLR 2026) instance of **per-instance, difficulty-aware budget allocation** — reviewers will ask how CASSI differs. Differences: P&B is upfront/open-loop and training-free; CASSI is a closed-loop learned stopper with budget state that also trains the executor. λ-style trade-off appears in their Lagrangian too. **THREAT LEVEL: MEDIUM** — claims the "adaptive budgeting beats static budgets" territory at CASSI's target venue, but with a completely different (training-free, single-model, non-stopping) mechanism.

### Route to Reason: Adaptive Routing for LLM and Reasoning Strategy Selection (Pan, Zhang, Zhao, Han; 2025; arXiv 2505.19435, preprint — no verified acceptance)

- **Read from:** PDF pages 1–5 (`2505.19435_route-to-reason.pdf`)
- **Problem:** Model routing alone ignores that the reasoning *strategy* (CoT, PAL, CoD, vanilla) co-determines cost/accuracy; jointly pick (model, strategy) per query under budget concerns.
- **Method:** Encode query with a pretrained encoder; each model and strategy gets a dual representation (frozen text-description embedding + learnable embedding). Two MLP heads predict, for every (model, strategy) pair, (a) success probability (BCE on correctness labels) and (b) output token count (MSE). A routing table is scored: score = λ·â − (1−λ)·l̂; argmax wins. **Upfront, one-shot; discrete combinatorial choice; small trained predictors; λ is a user-set trade-off knob (not learned, not budget-state-dependent).**
- **Training / RL usage:** Supervised only (correctness + length regression from offline sweeps of all pairs on training queries). No RL.
- **Experiments & benchmarks:** 7 open models (Qwen2.5-3B/7B/14B, R1-distill-7B/14B, QwQ-32B), 4 strategies; GSM8K, MMLU, MATH, OlympiadBench, +OOD sets.
- **Key results:** Higher accuracy than the best single model while **cutting tokens >60%**; dominates single-model and single-strategy baselines on accuracy-cost trade-off.
- **Limitations:** Requires exhaustive offline evaluation of all model×strategy pairs for training labels (expensive hindsight data); single-turn; static λ; no sequential/budget dynamics.
- **Relation to CASSI:** Expands routing's action space (strategy as compute knob) but stays upfront and frozen-executor. Its trained token-count predictor is a "budget predictor" precedent. **THREAT LEVEL: LOW-MEDIUM** — same claim-space as RouteLLM/ThinkSwitcher, no mid-trajectory or training-loop overlap.

### Ares: Adaptive Reasoning Effort Selection for Efficient LLM Agents (Yang, Hou, Wei, Bao, Chang; 2026; arXiv 2603.07915, preprint, UCSB+Accenture; code UCSB-NLP-Chang/Ares)

- **Read from:** PDF pages 1–7 (`2603.07915_ares-effort-router.pdf`)
- **Problem:** Thinking-LLM agents accrue reasoning tokens at every step of multi-turn trajectories; fixed effort levels (high/medium/low, as exposed by gpt-oss / GPT-5 / Gemini-3) are wasteful or damaging (gpt-oss-20b drops ~20% accuracy when forced high→low everywhere).
- **Method:** **Per-step dynamic reasoning-effort selection — a mid-trajectory controller.** A lightweight router (Qwen3-1.7B) receives interaction history h_t + current observation o_t at every turn and predicts the *lowest sufficient* effort level e_t ∈ {low, mid, high}, which configures the frozen agent's thinking mode for that step (KV cache preserved across levels, unlike model routing). Objective: max E[V(τ,x) − λ Σ_t cost(e_t)] (their Eq. 2). Router outputs a 3–5 sentence rationale then a discrete label.
- **Training / RL usage:** Three-phase hindsight label pipeline: (1) sample N successful trajectories at e_high, keep the most concise as reference τ*; (2) for each step and each effort level, sample the agent K=3 times and mark the level *sufficient* if it reproduces the ground-truth action a*_t in ≥M of K trials (functional-equivalence checks; LLM judge for search queries/messages); label = cheapest sufficient level; (3) teacher-generated rationales; SFT next-token loss. Then **GRPO on the router** (executor frozen): R = R_out (+5 task success) + R_cost (per-turn −0.2/−0.5/−1.0 for low/mid/high, trajectory-averaged, applied **only to successful trajectories** to avoid fail-cheap degeneracy) + R_form (−1 format). RL data filtering: drop zero-success prompts; keep top-30% reward-variance prompts among SR=100% ones.
- **Experiments & benchmarks:** TAU-Bench (retail/airline, GPT-4o user sim), BrowseComp-Plus (BM25, deep research), WebArena (AgentOccam scaffold); backbone gpt-oss-20b.
- **Key results (their Table 1):** TAU-Retail: 54.8% acc = fixed-High's 54.8% at **652k vs 1007k total tokens (−35%)**; Airline 36.0 vs High 38.0 (678k vs 873k); BrowseComp-Plus 41.3 vs 42.7 (1071k vs 1841k, **−42%**); WebArena **46.5 vs 45.0 (better than High)** at 1512k vs 2763k (**−45%**). Abstract claims **up to 52.7% reasoning-token reduction** (thinking tokens) with minimal success degradation; beats prompting-based routing (GPT-5, Gemini-3 Pro as router) on all four suites.
- **Limitations:** Effort levels are the only knob — cannot terminate the episode, cannot switch models, cannot alter budget; labeling needs K×|E|≈9 extra single-step executions per step (cheaper than full-rollout MC but not O(T)); cost reward hand-set per level; router sees no explicit remaining-budget state; executor untouched.
- **Relation to CASSI:** **The single closest system to CASSI's controller half.** Small model (1.7B ≈ CASSI's 0.5–3B), per-step mid-trajectory decisions, hindsight labels from completed trajectories ("minimum sufficient effort" ↔ CASSI's t*), SFT→GRPO training recipe (same as CASSI's stopper recipe), agentic benchmarks, quality-minus-λ·cost objective. What Ares does NOT do: stop/continue decisions, budget-state conditioning, value margin Δ output, and — decisively — **its router never trains the executor; there is no process-reward bridge and no self-reinforcing cycle.** CASSI's plan already cites "Ares-style discrete effort router" as a baseline — correct move; it must be implemented faithfully. **THREAT LEVEL: HIGH** — contribution #4 ("per-instance dynamic adaptation") and #5 ("small controller supervises large executor, <3% overhead") are both anticipated; CASSI's defensible deltas are stopping semantics, budget state, O(T) labels, and executor co-training.

### SeqRoute: Global Budget-Aware Sequential LLM Routing via Offline Reinforcement Learning (Xu, Zheng, Wang; 2026; arXiv 2605.25424, preprint, UT Austin ORIE)

- **Read from:** PDF pages 1–7 (`2605.25424_seqroute-budget-routing.pdf`)
- **Problem:** Routers treat queries independently, but real sessions have a **global budget**; myopic routing causes "budget bankruptcy" — spending on early easy turns and failing later hard ones.
- **Method:** Finite-horizon MDP (T_max=4 conversation turns): state = 384-d MiniLM embedding of history ∥ normalized remaining budget b_t (385-d); actions = {weak Llama-3.1-8B (1x cost), strong Llama-3.1-70B-AWQ (10x)}; reward r_t = ArmoRM quality − 5.0·1[bankrupt], γ=0.99. **Hindsight Budget Relabeling (HBR):** replay each unconstrained log under 5 hypothetical budgets {500,1500,3000,5000,8000}, deterministically depleting b_t and truncating at bankruptcy — 10k seed sessions → **2.38M budget-annotated transitions at zero extra API cost**; Proposition 1 proves validity (budget transition is bookkeeping, conditionally independent of conversation dynamics). Train **discrete CQL** (3-layer MLP [512,256,256]) — conservatism suppresses Q(low-budget, 70B), yielding emergent "delayed gratification." Deployment: **λ-sweep** a*_t = argmax_a[Q(s_t,a) − λ·c(a)] traces the whole cost-quality Pareto frontier zero-shot (Lagrangian duality).
- **Training / RL usage:** Offline RL (CQL) on relabeled logs; BC baseline same architecture. No online exploration; no LLM is trained.
- **Experiments & benchmarks:** Synthetic-but-systematic: ShareGPT/WildChat/Chatbot-Arena seeds, counterfactual tree rollouts with an 8B user simulator, ArmoRM-Llama3-8B scoring; 10k held-out episodes, eval budget 5000.
- **Key results (their Table 1):** CQL λ=0 strictly dominates BC: cost 3275 vs 3482 (**−6.0%**), bankruptcy 24.3% vs 31.8% (**−7.5pp**), return −3.44 vs −3.65. λ=5×10⁻⁴: cost 921, **bankruptcy 0.3%**, 0.1% 70B usage (**−73.5% cost vs BC**); budget-aware heuristic needs 4.2x the cost (3837) for similar safety. Behavioral probe: P(70B) slope vs remaining budget +0.13 for CQL, +0.02 for BC.
- **Limitations:** Two-model routing on 4-turn chat with a reward-model judge (no ground-truth correctness, no tools, no reasoning benchmarks); MLP-over-embeddings controller; budget is dollars/tokens for model choice, not reasoning depth; nothing is stopped early (sessions always run to T or bankruptcy); executor frozen.
- **Relation to CASSI:** Proves **hindsight relabeling with budget constraints for sequential routing already exists** — CASSI cannot claim post-hoc budget-conditioned relabeling of completed trajectories as new in routing-land. But the label semantics differ fundamentally: HBR re-simulates *budget bookkeeping* to expose bankruptcy signals; CASSI's oracle computes *optimal stopping* t* = argmax(quality − λ·cumcost) from per-step quality, a supervised target no routing paper produces. λ-sweep also anticipates CASSI's λ-tier mechanism (HIGH/MED/LOW multipliers) as inference-time Lagrangian control. **THREAT LEVEL: MEDIUM-HIGH** — the "hindsight relabeling + budget + λ" vocabulary is taken; the stopping-oracle content and agentic/executor-training scope are not.

### Budget-Aware Agentic Routing via Boundary-Guided Training (Zhang, Xia, Zhang, Madrigal, Mallick, Kessler, Rühle, Rajmohan; 2026; arXiv 2602.21227, preprint, Cambridge + Microsoft M365)

- **Read from:** PDF pages 1–4 (`2602.21227_budget-aware-agentic-routing.pdf`)
- **Problem:** Agentic routing is sequential and path-dependent (early mistakes compound; feedback only terminal); deployments impose strict per-task budgets. Step-wise choice between a cheap and an expensive model over long-horizon (T>20) agent trajectories.
- **Method:** POMDP: router π_θ(a_t | s_t), s_t = interaction history; a_t ∈ {M_small, M_large} **at every step (mid-trajectory)**. Two paradigms: soft-budget J = E[success − λ·Σcost] and hard-budget CMDP (max success s.t. Σc ≤ B_max) enforced at inference via **Budget-Constrained Decoding** (prune actions that would exceed remaining budget). Training pipeline: (1) **difficulty taxonomy** — run boundary policies always-small / always-large K=5 times → Easy / Hard / Intractable; (2) **BoSFT** — Easy and Intractable get always-small labels (deliberate "fail cheaply" on intractable!); for Hard, **stratified sampling** sweeps P(large)=k/N over N=10–20 rollouts and takes the **cheapest successful trajectory τ* = argmin cost** as the SFT expert (hindsight selection); (3) **BoPO** — GRPO variant with boundary-relative reward R = r_success + r_hard − λ·C_norm and reference anchors from the two boundary policies to prevent always-small collapse under sparse terminal reward.
- **Training / RL usage:** SFT (BoSFT) + online GRPO-style RL (BoPO) on the router; executors frozen.
- **Experiments & benchmarks:** Three long-horizon suites — scientific discovery, embodied instruction following, tool-using coding (avg ~20 steps).
- **Key results:** BoPO consistently improves the cost-success efficiency frontier, **matching always-large success at a fraction of its cost** across all three environments; transfers to strict hard-budget caps without budget-specific training. (Read from intro/method; numeric tables are in later pages of the PDF.)
- **Limitations (self-stated):** Trained policies embody one fixed soft-budget trade-off and **do not condition on remaining budget** — adapting to different hard caps is explicitly left to future work; binary model choice; no stopping action; no executor training; difficulty taxonomy costs 10 boundary rollouts per task plus up to N stratified rollouts for hard tasks.
- **Relation to CASSI:** Second-closest system after Ares: mid-trajectory, learned, cost-penalized GRPO router for agents with hindsight cheapest-success SFT labels. Crucially, its confessed gap — no remaining-budget conditioning — is exactly what CASSI's multi-dim budget-state input (tier + λ multipliers) provides, and its "fail cheaply on intractable" is a coarse cousin of CASSI's learned early STOP. **THREAT LEVEL: HIGH** — "budget-aware sequential decision-making for agents, SFT+GRPO, per-step" is now occupied; CASSI must differentiate on stopping semantics, budget-state conditioning, O(T) labels (BoSFT needs ~10–25 rollouts/task), and the process-reward bridge into executor training.

### Don't Overthink It: Inter-Rollout Action Agreement as a Free Adaptive-Compute Signal for LLM Agents (TrACE) (Sethi; 2026; arXiv 2604.08369, preprint, single-author)

- **Read from:** PDF pages 1–4 (`2604.08369_trace-action-agreement.pdf`)
- **Problem:** Deployed agents apply uniform per-step compute (greedy or fixed self-consistency k); easy steps waste samples, hard steps are starved.
- **Method:** **Training-free per-timestep controller**: at each step draw k_init=2 candidate actions; compute agreement α_t = fraction matching the plurality action (canonicalized); if α_t ≥ τ_high=0.75 commit immediately, else add samples one at a time up to k_max ∈ {4,8} and commit to plurality. Compute knob = number of samples per step, conditioned on the model's own behavioral consistency (argued better-calibrated than verbalized confidence). No learned components, no verifier, no labels. Claims to be "the first training-free, per-timestep adaptive-compute controller for LLM agents evaluated on multi-step sequential decision tasks."
- **Training / RL usage:** None.
- **Experiments & benchmarks:** Qwen2.5-3B-Instruct (quantized, CPU); GSM8K n=50; MiniHouse (custom text household navigation) n=30.
- **Key results:** TrACE-4 matches SC-4 accuracy with **33% fewer LLM calls** (GSM8K) / **39% fewer** (MiniHouse); TrACE-8 matches SC-8 with **55% / 65% fewer calls**; agreement shown to be a leading indicator of step success.
- **Limitations:** Tiny scale (n=50/30, one 3B model), custom toy environment, self-acknowledged "no SOTA claims"; controls sampling width only; heuristic threshold; cannot stop the episode or vary reasoning depth.
- **Relation to CASSI:** Occupies the rhetorical territory "per-step adaptive compute for agents" from the training-free side (as Dynasor does for reasoning programs). Its existence sharpens the bar: CASSI's learned stopper must beat certainty/agreement heuristics (CASSI's plan does include a zero-training self-eval baseline — keep it). **THREAT LEVEL: MEDIUM** — weak evidence but a direct "first" claim adjacent to CASSI's #4; easy to out-experiment, must be cited.

## Peripheral papers

- **FrugalGPT: How to Use LLMs While Reducing Cost and Improving Performance** (Chen, Zaharia, Zou; 2023; arXiv 2305.05176). The cascade ur-paper: send each query through a learned LLM sequence (cheap→expensive), with a scoring function deciding whether to accept the current answer or escalate; plus prompt adaptation and completion caching. Matches GPT-4 at up to ~98% cost reduction on tested tasks (their headline). Escalation-on-low-confidence is per-instance compute adaptation avant la lettre — upfront/sequential-cascade, heuristic-threshold, frozen models. LOW threat; must appear in related work as the cascade lineage root.

- **LLM Cascades with Mixture-of-Thoughts Representations for Cost-Efficient Reasoning** (Yue et al.; ICLR 2024; arXiv 2310.03094). Weak-LLM "answer consistency" across sampled CoT and PoT solutions is the difficulty signal: consistent → accept weak answer, inconsistent → escalate to GPT-4. 40% of GPT-4 cost at comparable accuracy on math/causal reasoning. Notable as the difficulty-estimation-by-consistency precedent (ancestor of Dynasor and TrACE signals). LOW threat.

- **System-1.x: Learning to Balance Fast and Slow Planning with LLMs** (Saha, Prasad, Chen, Hase, Stengel-Eskin, Bansal; ICLR 2025; arXiv 2407.14414). A **controller** decomposes a planning problem into sub-goals classified easy/hard and delegates them to a System-1 (direct plan) or System-2 (explicit search) planner, governed by a **user-specified hybridization factor x** (fraction of compute in System-2). All three components are fine-tuned LLMs (Llama-3-8B); on Maze/Blocksworld it beats pure System-1 and System-2 at matched search budgets. Sub-goal-level (coarser than per-step), user-set rather than learned budget, non-agentic planning domains. LOW-MEDIUM threat; the "controller allocates deliberation per sub-problem" precedent.

- **Efficiently Scaling LLM Reasoning with Certaindex / Dynasor** (Fu, Chen, Zhu, Fu et al.; 2024–25; arXiv 2412.20993). Defines **Certaindex**, an algorithm-agnostic mid-generation certainty statistic (e.g., empirical answer-agreement / entropy) that signals when further compute won't change the final answer; used inside the Dynasor serving system for early exit, **dynamic token allocation across concurrent reasoning programs**, and gang scheduling (up to ~50% fewer tokens claimed at matched accuracy in follow-up reporting). This is *mid-trajectory, continuous, heuristic, serving-level* allocation — the strongest training-free cross-reference for CASSI's confidence/stability input features. MEDIUM threat to the "dynamic per-instance adaptation" wording (not to the learned-stopper or executor-training claims).

- **xRouter: Training Cost-Aware LLMs Orchestration System via RL** (Qian, Liu, Kokane, Prabhakar et al., Salesforce; 2025; arXiv 2510.08439). An orchestrator LLM is trained **end-to-end with RL under an explicit cost-aware reward** to answer directly or invoke external models (tool-calling routing, including multi-model orchestration). Strong cost-performance trade-offs across benchmarks; shows "RL with cost-encoding rewards" applied to *routing* (not to the executor's own reasoning policy). MEDIUM threat to the framing "cost-aware RL signal from economics of the task" — but it trains the router-as-policy, not a stopper, and no process rewards.

- **Adaptive LLM Routing under Budget Constraints (PILOT)** (Panda, Magazine, Devaguptapu, Takemori et al.; EMNLP 2025 Findings; arXiv 2508.21141). Routing as a **contextual bandit** with preference-aligned query-LLM embeddings (warm-started LinUCB) and an **online multi-choice knapsack** policy for spend pacing under a global budget — adaptation from bandit feedback rather than full supervision; ~93% of GPT-4 quality at 25% of its cost (their headline). Single-turn, budget-paced, frozen models. LOW-MEDIUM threat; strengthens the "budget-constrained routing is crowded" picture.

- **Inference-Time Budget Control for LLM Search Agents** (Fang, Hu, Chang, Guo et al.; 2026; arXiv 2605.05701). Multi-hop QA agents under **dual hard budgets** (tool calls + tokens): a controller scores each feasible action (retrieve / decompose / **commit answer**) by an operational **Value-of-Information estimate — marginal task value per unit budget given the current state and remaining budgets** — then a selective evidence-grounded finalizer commits. Training-free, mid-trajectory, includes an answer-commitment (stopping-flavored) decision. MEDIUM-HIGH threat as the nearest *budget-state-conditioned mid-trajectory controller for search agents*; but heuristic VoI, no learning, no executor training. (Also relevant to the stopping-area review; PDF already in `research/papers/` as `2605.05701_inference-time-budget-control-voi.pdf`.)

- Briefly noted (same cluster, thinner overlap): **Thinkless** (2505.13379; RL learns `<short>`/`<think>` control token — in-policy binary effort choice); **Adaptive Deep Reasoning** (2505.20101; RL long-short group-wise rewards + logit-based mode-switch loss, single model); **Z1** (2504.00810, EMNLP 2025 Industry; trains on short+long code-reasoning trajectories with a Shifted Thinking Window — Z1-7B matches R1-distill-7B at ~30% of thinking tokens); **Speculative Thinking** (2504.12329; training-free big-model-takes-over-reflection for small reasoners, +6–12% accuracy with shorter outputs); **Steering LLM Thinking with Budget Guidance** (2506.13752, ACL 2026 Findings; lightweight Gamma-distribution predictor of remaining thinking length softly steers generation toward a target budget — a *learned mid-generation budget tracker*); **ORBIT** (2601.08310; on-policy RL for multi-budget-conditioned reasoning); **DART** (2606.23181; training-free draft-agreement thinking-budget router for hybrid models); **ODAR** (2602.23681; active-inference difficulty estimator routing fast/slow agents); **Prompt Difficulty Prediction** (2511.03808; lightweight per-query compute allocators for o-style models); **Anytime Verified Agents** (OpenReview JMDCMf7mlF, 2026; controller reallocates search/sampling/verification within a user budget by uncertainty and marginal-reliability estimates); surveys **2507.02076** ("Reasoning on a Budget"), **2511.10788** ("From Efficiency to Adaptivity"), **2603.04445** (routing/cascading survey) — useful taxonomies confirming the upfront-vs-adaptive framing used here.

## Synthesis

### Landscape table

| System | Allocator | Timing | Decision space | Trained how | Budget state input? | Controls frozen policy or trains policy? | Domain |
|---|---|---|---|---|---|---|---|
| Snell et al. 2024 | difficulty bins (lookup) | upfront | strategy + hyperparams (discrete) | none (validation binning) | no | controls (frozen) | single-turn math |
| FrugalGPT / MoT-cascade | cascade threshold | sequential escalation | model chain | small scorer / consistency heuristic | no | controls | single-turn |
| RouteLLM | small router | upfront | 2 models | supervised (preference) | no | controls | single-turn |
| PILOT (2508.21141) | bandit router | upfront + spend pacing | k models | contextual bandit + knapsack | global spend pacing | controls | single-turn stream |
| ThinkSwitcher | small switcher | upfront | 2 CoT modes | supervised regression (hindsight pass rates) | no | controls | single-turn math |
| Route to Reason | MLP predictors | upfront | model × strategy | supervised (correctness + length) | λ knob only | controls | single-turn |
| Plan-and-Budget | planner LLM + decay schedule | upfront + fixed schedule | continuous token budgets per sub-question | none (training-free) | per-query budget | controls | single-turn + planning |
| Dynasor / Certaindex | certainty statistic | **mid-generation** | continuous token allocation / exit | none (heuristic) | serving-level | controls | reasoning programs |
| TrACE | agreement heuristic | **per-step** | sample count (2→k_max) | none | no | controls | toy agent tasks |
| VoI budget control (2605.05701) | VoI scores | **per-step** | retrieve/decompose/**commit** | none (operational estimates) | **yes (dual remaining budgets)** | controls | search agents |
| **Ares** | **1.7B router** | **per-step** | 3 discrete effort levels | **SFT (hindsight min-effort labels) + GRPO** | no | controls | **TAU-Bench, BrowseComp+, WebArena** |
| **BAAR** | router policy | **per-step** | 2 models | **BoSFT (cheapest-success hindsight) + BoPO/GRPO** | only via decode-time pruning (self-stated gap) | controls | long-horizon agents |
| **SeqRoute** | MLP Q-function | per-turn | 2 models | **offline CQL on hindsight-budget-relabeled logs** | **yes (remaining budget in state)** | controls | 4-turn chat |
| xRouter | orchestrator LLM | per-call | k models / direct answer | online RL, cost-aware reward | no | trains the *router* LLM | mixed benchmarks |
| **CASSI (planned)** | 0.5–3B stopper | **per-step** | STOP/CONTINUE/ADJUST + continuous Δ | SFT on O(T) oracle t* + GRPO | **yes (multi-dim tier + λ)** | **controls AND trains executor (Δ as process reward)** | tool agents |

### Gaps the literature leaves open

1. **Nobody trains the executor.** Every allocator above — heuristic or learned, upfront or per-step — treats the task-solving policy as frozen. No paper converts a controller's cost-aware value estimate into a process reward for executor RL, and none claims a self-reinforcing controller↔executor cycle. CASSI's contribution #1 (the bridge/cycle) is untouched by this area.
2. **Stopping is not in the action space.** Ares picks effort, BAAR/SeqRoute/RouteLLM pick models, Plan-and-Budget picks token counts. Only the VoI paper (heuristic) includes answer-commitment, and BAAR's "fail cheaply on intractable" is a static label, not a learned per-state STOP. A *learned, budget-conditioned* stop/continue margin Δ(s_t) remains open in this literature (the stopping-literature review must confirm from its side).
3. **Label efficiency.** Hindsight labels here cost extra sampling: ThinkSwitcher k samples × 2 modes per query; Ares K=3 × 3 levels per step; BAAR ~10 boundary + up to 20 stratified rollouts per task; only SeqRoute's HBR is zero-extra-cost bookkeeping (but computes bankruptcy, not optimal stopping). CASSI's O(T) argmax-over-existing-steps oracle is genuinely cheaper — *provided per-step quality_t is measurable*, which is CASSI's own strong assumption to defend.
4. **Remaining-budget conditioning is rare and shallow.** SeqRoute has it (385-d state), VoI has it (heuristic), BAAR explicitly lacks it and names it future work. CASSI's multi-dimensional budget tiers with λ multipliers go further than anything read here.
5. **Mid-trajectory budget *re-adjustment*** (CASSI's ADJUST action; re-planning λ or budget as the trajectory unfolds) has no learned precedent — Plan-and-Budget's schedules are fixed at t=0; Budget Guidance tracks but does not re-plan.

### Top threats to CASSI's per-instance-adaptation claim (ranked)

1. **Ares (2603.07915) — HIGH.** A small SFT+GRPO controller already performs per-step, per-instance compute adaptation for large frozen agents on TAU-Bench/WebArena with hindsight minimal-effort labels. Kills any claim that "a small learned controller adapting compute per instance/step for agents" is new. CASSI must (a) implement the Ares-style baseline faithfully, (b) reframe #4 as *budget-state-conditioned stopping* + (c) emphasize the executor-training bridge Ares lacks.
2. **BAAR (2602.21227) — HIGH.** "Budget-aware sequential decision-making for agentic routing" with SFT-from-hindsight-cheapest-trajectory + GRPO is Microsoft-branded and formalized (soft/hard budget CMDP). Overlaps CASSI's framing, vocabulary, and training recipe; differs in action space (models, not stopping) and no budget-state conditioning.
3. **SeqRoute (2605.25424) — MEDIUM-HIGH.** "Hindsight budget relabeling" + inference-time λ-sweep exist by name. CASSI's oracle-labeling story must be positioned as *optimal-stopping* relabeling (quality-aware t*), not budget bookkeeping, and its λ-tier mechanism should cite the Lagrangian λ-sweep precedent.
4. **Plan-and-Budget (2505.16122, ICLR 2026) — MEDIUM.** Per-instance, difficulty-aware budget allocation is already at CASSI's target venue, training-free. CASSI's H5 (difficulty-correlated stopping) must beat this class, not just static penalties — the "static penalty" strawman is outdated.
5. **Dynasor/Certaindex + TrACE + VoI control — MEDIUM.** Training-free mid-trajectory allocation signals (certainty, agreement, VoI) are established; CASSI's learned stopper must demonstrably beat calibrated heuristics, not only fixed budgets.

### Opportunities for CASSI

- **Sharpen contribution #4's wording**: not "per-instance adaptation" (taken) but "learned budget-state-conditioned optimal-stopping control, mid-trajectory, with continuous value margin Δ" — no paper here has that combination; BAAR's self-stated limitation is direct ammunition.
- **The efficiency table almost writes itself**: label-construction cost per trajectory — AgentPRM O(K×T) rollouts, Ares O(K·|E|·T) step re-executions, BAAR O(N+K) trajectories, ThinkSwitcher O(k·modes) per query, SeqRoute O(T) bookkeeping (no quality oracle), **CASSI O(T) with quality oracle**. Position against Ares/BAAR explicitly, not just AgentPRM.
- **Adopt the λ-sweep evaluation**: SeqRoute's zero-shot Pareto navigation via inference-time λ is a strong protocol CASSI's tiered-λ design can reproduce and extend (report frontier traversal without retraining).
- **Baselines to add** beyond the plan's current list: BAAR-style per-step model routing under equal budgets; TrACE/Certaindex-style agreement heuristics as the zero-training controller; VoI-scored commitment. Beating Ares-style effort routing at iso-cost on TAU-Bench/WebArena would be the single most persuasive head-to-head.
- **The unclosed loop is the paper.** Every reviewer in this area will accept "routers exist"; the sale is that CASSI's stopper *feeds back* into executor GRPO (process-reward bridge) and demonstrably improves the executor itself — an ablation (controller-only vs. +bridge) is the pivotal experiment, exactly as the plan's "CASSI w/o process-reward bridge" foresees.
