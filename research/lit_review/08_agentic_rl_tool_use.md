# Agentic RL for Tool Use & Long-Horizon Tasks

> Area: the GRPO-family RL training infrastructure that CASSI's executor training builds on.
> Focus questions: (a) reward designs — do any already include cost/efficiency/stopping terms?
> (b) standard benchmarks/executors for RL-trained agents (feasibility of CASSI's experiment plan);
> (c) known GRPO pathologies for agents.
> Researched 2026-07-16. All PDFs in `research/papers/`. Citation counts are Semantic Scholar, 2026-07.

## Area overview

Between early 2024 and 2026, "agentic RL" consolidated into a standard recipe: an LLM policy
interleaves `<think>` reasoning with tool calls (search engine, browser, code sandbox), receives an
**outcome-level verifiable reward** at trajectory end, and is optimized with a critic-free
group-based policy-gradient method — GRPO (DeepSeekMath, Shao et al. 2024) or a descendant (DAPO,
Dr. GRPO, RLOO, REINFORCE++) — with tool/observation tokens **masked from the loss**. Search-R1,
R1-Searcher, ReTool, DeepResearcher, ToolRL and WebAgent-R1 established this template on Qwen2.5
3B–32B backbones; frameworks (verl and its derivatives, RAGEN, AgentGym-RL, SkyRL, rLLM) turned it
into infrastructure. The TMLR 2026 survey ("The Landscape of Agentic Reinforcement Learning for
LLMs") formalizes the shift as moving from degenerate single-step MDPs to POMDPs with temporally
extended interaction, and synthesizes 500+ works.

For CASSI the crucial fact is that **reward design in this literature is overwhelmingly outcome
(+ format) only, and deliberately so** — Search-R1 and ReTool explicitly avoid process rewards to
prevent reward hacking. Where economics enters at all, it enters in one of four thin forms:
(1) a **multiplicative tool-count coefficient** on the outcome reward with a per-question optimal
call count estimated in hindsight from the rollout group (OTC-PO); (2) a **gamma-decay** on the
outcome reward favoring shorter trajectories (Kimi-Researcher); (3) **step-level redundancy
penalties** against reference search trajectories (StepSearch); and (4) **difficulty-aware shaping**
penalizing tool calls on easy queries (EAPO, 2026). None of these learns a *state-dependent
quality-vs-cost margin*, none trains a *separate* stopping/value model, and none feeds a cost-aware
signal back as a *process reward* for executor RL — the specific combination CASSI proposes.
Notably, several papers document the opposite pressure: Search-R1's agent learns to call search
*more* over training; ARPO adds a +0.1 *bonus* for using more tool types; OTC names the resulting
failure "cognitive offloading" and shows it worsens with model scale.

The second crucial fact is that **credit assignment and stability, not economy, are this
literature's obsession** — and its solutions overlap CASSI's efficiency claims. GiGPO obtains
per-step advantages with zero extra rollouts by grouping actions taken at recurring "anchor states"
across the rollout group, explicitly solving the same O(K×T²)-extra-rollout problem CASSI cites
against Monte-Carlo PRM training, and reports OTC-level tool efficiency emerging *without any cost
term* (discounted step returns implicitly penalize wasted steps). RAGEN documents the "Echo Trap"
(reward-std collapse → entropy collapse → gradient spikes) in multi-turn GRPO; DAPO and Dr. GRPO
document entropy collapse, length inflation and normalization biases in the single-turn regime;
ARPO shows token entropy spikes right after tool feedback and branches rollouts there. Any CASSI
experiment section will be reviewed against these known pathologies and their fixes.

## Core papers

### Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (Jin et al., 2025, COLM 2025; arXiv 2503.09516)
- **Read from:** PDF pages 1–9 (`2503.09516_search-r1.pdf`)
- **Problem:** LLMs prompted to use search engines interact with them suboptimally; SFT on
  trajectories doesn't scale; how to RL-train interleaved reason+search?
- **Method:** Multi-turn rollout with `<think>/<search>/<information>/<answer>` tokens; the search
  engine is modeled as part of the environment (π_θ(·|x; R)); **retrieved-token loss masking**
  (policy gradient computed only on LLM-generated tokens); works with both PPO and GRPO (PPO is the
  paper's default); hard cap of B actions per rollout (Algorithm 1), plus a "My action is not
  correct. Let me rethink." injection on malformed actions.
- **Training / RL usage:** PPO (default) and GRPO. Reward verbatim: "we adopt a rule-based reward
  system that consists solely of **final outcome rewards**… r_φ(x,y) = EM(a_pred, a_gold)". No
  format reward ("our learned model already demonstrates strong structural adherence"), **no
  cost/length/tool-count term**, explicitly avoids process rewards and neural reward models.
- **Experiments & benchmarks:** Qwen2.5-3B/7B (Base+Instruct), E5 retriever over 2018 Wikipedia,
  train on NQ+HotpotQA; eval on NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle.
- **Key results:** 7B-base PPO avg EM 0.431 vs RAG 0.304 (paper headline: +24%/+20% relative for
  7B/3B; the contribution bullet states 41%/20% vs RAG under same setup). GRPO converges faster but
  **collapses after many steps**; PPO more stable. Response length: sharp drop in first ~100 steps,
  then grows as the model learns to search; **number of valid search calls monotonically increases
  over training** (Fig. 2d). Retrieved-token masking: 0.431 vs 0.343 without.
- **Limitations:** EM-only reward invites verbosity insensitivity; static local corpus; no notion
  of cost — more search is always weakly better for reward.
- **Relation to CASSI:** This is the canonical executor CASSI would train on HotpotQA/MuSiQue; its
  "search calls increase over training" curve is direct motivating evidence for CASSI's
  no-economic-signal claim. THREAT LEVEL: **LOW** — zero cost-awareness; purely complementary
  baseline.

### ToolRL: Reward is All Tool Learning Needs (Qian et al., 2025, NeurIPS 2025; arXiv 2504.13958)
- **Read from:** PDF pages 1–13 (`2504.13958_toolrl.pdf`)
- **Problem:** First systematic study of *reward design* for general tool selection/invocation
  (TIR) under RL; SFT on distilled traces overthinks and fails to generalize.
- **Method:** GRPO (KL removed) over single-step-decomposed tool-call instances from ToolACE,
  Hammer, xLAM (4K samples). Studies 4 reward axes: type, scale, granularity, dynamics.
- **Training / RL usage:** Reward verbatim: `R_final = R_format + R_correct`, R_format ∈ {0,1};
  R_correct ∈ [−3,3] fine-grained decomposition = tool-**name** match (Jaccard) + parameter-**name**
  match + parameter-**value** match, normalized `R_correct = 6·R_max/S_max − 3`. **Length-reward
  ablation:** `R_length = min(L_think/L_target, 1)`, L_target = 512, plus a dynamic variant
  `L_target·(1+p)` growing with training progress. **No tool-count/cost term.**
- **Experiments & benchmarks:** Qwen2.5-1.5B/3B/7B-Instruct, Llama-3.2-3B; BFCL V3, API-Bank,
  Bamboogle. veRL, batch 512, G=4, 15 epochs, temp 1.0.
- **Key results:** BFCL overall: 3B 52.98% (GRPO cold start) vs 33.04% raw, 7B 58.38% vs 41.97%;
  ~+17% over base, +15% over SFT. **Takeaway 1 (verbatim): "length rewards encourage longer
  reasoning traces [but] do not consistently improve task performance and may even harm it in
  smaller models"** — 1.5B drops 46.20%→33.23% with length reward. Smooth dynamic reward-scale
  transitions beat abrupt two-stage switches (3B: 53.81 dynamic vs 50.66 two-stage vs 52.98
  original). Finer-grained (partial-credit) rewards beat coarse all-or-nothing. On Bamboogle their
  model gets best accuracy (7B: 72.0%) with moderate tool calls (1.63 avg) — efficiency *observed*,
  not rewarded.
- **Limitations:** Single-step decomposition (needs ground-truth tool calls per step ≈ dense
  supervision); no multi-turn credit assignment; no cost objective.
- **Relation to CASSI:** Strongest evidence that naive length shaping *hurts* — supports CASSI's
  claim that scalar static penalties are crude, but also warns that any added reward term (incl.
  CASSI's α·Δ(s_t)) must beat the "fine-grained decomposition is what matters" explanation. THREAT
  LEVEL: **LOW-MEDIUM** — no cost term, but its reward-design-ablation framing is exactly what
  reviewers will demand of CASSI's R_t = α·Δ + β·progress + γ·format.

### Acting Less is Reasoning More! Teaching Model to Act Efficiently (OTC-PO) (Wang, Qian et al., 2025, arXiv preprint 2504.14870, v2 May 2025; 68 cites)
- **Read from:** PDF pages 1–9 (`2504.14870_otc-acting-less.pdf`)
- **Problem:** RL agents optimized only for final correctness over-call tools ("**cognitive
  offloading**"), which worsens as model size grows; wants correct answers with *minimal* tool
  calls. Introduces **tool productivity** TP = #correct answers / #tool calls.
- **Method:** Optimal Tool Call-controlled Policy Optimization (OTC-PO), plug-in for PPO/GRPO.
  Assumes for each (question, model) there is an optimal (minimal) tool-call count n; since n is
  unknown, **approximates n in hindsight as the minimum tool calls among correct trajectories in
  the GRPO group** (k = min(C), updated across epochs toward a global optimum).
- **Training / RL usage:** Reward verbatim: final reward `r_φ^tool(q,y) = α · r_tool · r_φ(q,y)` —
  a **multiplicative efficiency coefficient** on the base correctness reward. OTC-PPO:
  `r_tool = cos(mπ/(2m+c))`. OTC-GRPO: r_tool = 1 if m=n=0; cos(mπ/(2m+c)) if n=0;
  `sin(f(m,n)·π/(2n))` otherwise, with remap `f(m,n) = 2nm/(m+n)` (peak reward exactly at m=n,
  lower for both over- and under-calling). Multiplicative rather than additive by design: additive
  "was unstable and sub-optimal" (model earns reward by dropping tools *without* being correct);
  when the answer is wrong r_φ = 0 kills the efficiency bonus, preventing hacking.
- **Experiments & benchmarks:** Search setting = Search-R1's (NQ+HotpotQA train, Qwen2.5-3B/7B-Base,
  eval incl. TriviaQA, PopQA, 2Wiki, MuSiQue, Bamboogle); code setting = ToRL (Qwen2.5-Math-1.5B/7B,
  AIME24/25).
- **Key results:** Up to **68.3% fewer tool calls and up to 215.4% higher TP at comparable
  accuracy**: e.g. 7B NQ — Search-R1-PPO EM 0.449 / TC 3.282 vs OTC-PPO EM 0.446 / TC 1.040;
  OTC-GRPO EM 0.444 / TC 0.990. OOD (7B, OTC-PPO) beats Search-R1-PPO EM on all 5 QA sets with far
  fewer calls. Also: 3B Search-R1 uses fewer calls than 7B — over-reliance grows with scale.
- **Limitations (self-stated & observed):** cost = tool-call *count* only (no token/latency/dollar
  costs, no quality-cost margin); trajectory-level coefficient (no per-step signal, no stopping
  decision); hindsight n is only defined when some group trajectory is correct; QA/math tools only;
  future work "extend… to more complex agentic tasks… and longer-horizon reasoning".
- **Relation to CASSI:** Closest reward-design competitor in this area. Shares (i) economics
  beyond correctness, (ii) hindsight per-question optimal computed from completed trajectories at
  zero extra rollout cost — structurally analogous to CASSI's post-hoc t* oracle. Differs: no
  learned stopping model, no per-step process reward, no continuous quality_t−λ·cost_t margin, no
  budget tiers, no inference-time controller, single-model. CASSI's plan already lists it implicitly
  ("single-model GRPO+cost penalty" baseline) but must cite and compare to OTC-PO *specifically*.
  THREAT LEVEL: **HIGH** — a reviewer's first "hasn't this been done?" citation for cost-aware
  agent RL rewards; CASSI must beat OTC-GRPO, not just Search-R1.

### ReTool: Reinforcement Learning for Strategic Tool Use in LLMs (Feng, Huang et al., ByteDance Seed, 2025, arXiv preprint 2504.11536; 330 cites)
- **Read from:** PDF pages 1–7 (`2504.11536_retool.pdf`)
- **Problem:** Text-only reasoning RL is weak at exact computation; train models to decide *when
  and how* to invoke a code interpreter inside long CoT.
- **Method:** Cold-start SFT on automatically code-augmented reasoning traces (D_CI), then PPO
  (veRL, KL=0) with interleaved sandbox execution; `<interpreter>` feedback masked from loss;
  KV-cache reuse across code pauses; asynchronous sandbox pool.
- **Training / RL usage:** Reward verbatim: `R(a,â) = 1 if is_equivalent(a,â) else −1` — accuracy
  only; "we simplify the reward design… based on mere outcome feedback **without considering code
  executability reward**" (anti-hacking rationale). **No cost/length/tool-count term.**
- **Experiments & benchmarks:** Qwen2.5-32B-Instruct (16k ctx, batch 512); AIME 2024/2025 (pass@1
  avg of 32).
- **Key results:** AIME24 **67.0% in 400 RL steps vs text-RL 40.0% at 1080 steps**; on
  DeepSeek-R1-Distill-Qwen-32B backbone 72.5% AIME24 / 54.3% AIME25 (o1-preview +27.9%). Behavior:
  **response length drops ~40% (10k→6k tokens) vs pre-RL** as code replaces verbose arithmetic;
  code ratio →98%; invocation timing shifts earlier; emergent code self-correction ("aha moment").
- **Limitations:** Single tool, math only; efficiency is an emergent side-effect, never optimized
  or measured against cost budgets.
- **Relation to CASSI:** Standard tool-RL executor recipe (cited in CASSI plan); its 40%
  token-reduction-without-cost-reward result is a confound CASSI ablations must control for (tool
  integration alone shortens trajectories). THREAT LEVEL: **LOW**.

### DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments (Zheng et al., 2025, EMNLP 2025; arXiv 2504.03160)
- **Read from:** PDF pages 1–8 (`2504.03160_deepresearcher.pdf`)
- **Problem:** RAG-based RL trains against sanitized local corpora; train agents end-to-end against
  the live, noisy web instead.
- **Method:** GRPO with observation masking; multi-agent architecture — the policy calls
  `web_search` (top-10 title/URL/snippet) and `web_browse`, where a *separate frozen browsing
  agent* reads pages segment-by-segment, maintains short-term memory, and returns distilled info.
  50-node crawler cluster, retry + 7-day cache for API limits/anti-crawling.
- **Training / RL usage:** GRPO, G=16 rollouts × 256 prompts, ≤10 tool calls per rollout (hard
  cap). Reward verbatim: "**−1 if format is incorrect; word-level F1 if format is correct**". No
  cost/length/tool term.
- **Experiments & benchmarks:** Qwen2.5-7B-Instruct; 80K contamination-filtered examples
  (NQ:TQ:HotpotQA:2Wiki = 1:1:3:3); ID eval NQ/TQ/HotpotQA/2Wiki, OOD MuSiQue, Bamboogle, PopQA
  (model-based judging).
- **Key results:** Up to **+28.9 points over prompt-engineered baselines and +7.2 over RAG-RL
  agents** (e.g. beats Search-R1-instruct and R1-Searcher on all 7 sets, Fig. 1). Emergent
  planning, cross-source validation, reflection, honesty.
- **Limitations:** 7B + short-answer F1; hard 10-call cap substitutes for any economic policy;
  real-web infra cost is the bottleneck (their own motivation for caching).
- **Relation to CASSI:** Best evidence that real-web RL at 7B is feasible (relevant to CASSI's
  WebWalkerQA/GAIA ambitions) and that infra, not algorithm, is the constraint. THREAT LEVEL:
  **LOW** — no economics.

### RAGEN / StarPO: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning (Wang et al., 2025, arXiv preprint 2504.20073; 257 cites)
- **Read from:** PDF pages 1–8 (`2504.20073_ragen.pdf`)
- **Problem:** What makes multi-turn agent RL stable? Single-turn recipes (PPO/GRPO) collapse when
  ported to interactive, stochastic environments.
- **Method:** StarPO — trajectory-level objective J = E[R(τ)] over full interaction sequences with
  PPO/GRPO instantiations; RAGEN = modular system on 4 environments (Bandit, Sokoban, FrozenLake,
  WebShop). **StarPO-S** stabilized variant: (1) *uncertainty-based instance filtering* — train only
  on top-p% prompts by reward std (default keep 25%); (2) KL-term removal; (3) asymmetric
  clip-higher (both adopted from DAPO).
- **Training / RL usage:** Qwen2.5-0.5B/3B-Instruct; P=8 prompts × N=16 rollouts, ≤5 turns/10
  actions; GRPO trajectory-score normalization or PPO+GAE (γ=λ=1.0); entropy bonus 0.001; reward =
  environment return **plus a −0.1 response-format penalty**. No cost term.
- **Experiments & benchmarks:** The 4 environments; metrics incl. success, rollout entropy,
  in-group reward std, gradient norm.
- **Key results:** **"Echo Trap"** failure mode: early diverse reasoning → locally-rewarded
  templates; observable as reward-std collapse and entropy drop *before* performance collapse, then
  gradient spikes (e.g. FrozenLake-PPO std collapses at step ~40, reward at ~90). PPO outlasts GRPO
  on Bandit/Sokoban; GRPO better on FrozenLake/WebShop. Filtering to high-variance prompts delays or
  eliminates collapse. Finding 3 (verbatim): without "fine-grained, reasoning-aware reward signals,
  agent reasoning hardly emerge[s] through multi-turn RL" — models drift to shallow/hallucinated
  reasoning when only task success is rewarded.
- **Limitations:** Small symbolic environments + 0.5B models for the main analysis; no fix offered
  beyond filtering/clipping (their "meticulous reward design" is future work).
- **Relation to CASSI:** (i) catalog of multi-turn GRPO pathologies CASSI training must anticipate;
  (ii) its explicit call for fine-grained reward signals is the gap CASSI's stopper-as-PRM fills;
  (iii) uncertainty-filtering is orthogonal machinery CASSI could reuse. THREAT LEVEL: **LOW** as
  competition; HIGH as a reviewer checklist for training-stability reporting.

### GiGPO: Group-in-Group Policy Optimization for LLM Agent Training (Feng et al., NTU/Skywork, 2025, NeurIPS 2025; arXiv 2505.10978)
- **Read from:** PDF pages 1–9 (`2505.10978_gigpo.pdf`)
- **Problem:** Episode-level GRPO advantages are uninformative over long horizons (ALFWorld ~50
  steps, 20k+ tokens); per-state Monte-Carlo re-rollouts are prohibitively expensive. Goal:
  fine-grained credit assignment that stays critic-free with **zero extra rollouts**.
- **Method:** Two-level advantages. Episode level: standard GRPO normalization of total returns
  (F_norm = std or 1, the latter = unbiased leave-one-out). Step level: **anchor state grouping** —
  retroactively hash-match identical (or ≥0.9-similar) environment states recurring across the N
  trajectories; for every anchor state s̄, group all actions taken there, score each by discounted
  return `R_t = Σ_k γ^{k−t} r_k`, and normalize within the group. Combined advantage
  `A = A^E + ω·A^S` (ω=1). Entirely offline hashing; "<0.002% extra time", same memory.
- **Training / RL usage:** GRPO-style clipped objective with KL β; reward = environment
  success/score (outcome), **no explicit cost term** — but γ<1 discounting inside step returns
  implicitly prefers actions that reach reward sooner (their WebShop example explicitly ranks
  "1st-item-now" over "2nd-item-then-backtrack" over "next-page").
- **Experiments & benchmarks:** ALFWorld, WebShop (Qwen2.5-1.5B/7B-Instruct, N=8); search QA à la
  Search-R1 (NQ+HotpotQA train; N=5, ≤4 turns; E5).
- **Key results:** ALFWorld 1.5B: 96.0 vs GRPO 85.3; 7B: 98.8 vs 90.8. WebShop success 1.5B: 67.4
  vs 56.8; 7B: 75.2 vs 66.1 (>12%/>9% over GRPO). QA avg **42.1 (3B) / 47.2 (7B) vs Search-R1 32.5
  / 38.5**. Tool efficiency: 7B needs ~0.9 calls (single-hop) / ~1.6 (multi-hop), "matching OTC…
  which achieves ~1.0 and ~1.7" — **without any efficiency reward**; step-level credit suppresses
  redundant repeated queries. Also notes std-normalization "difficulty bias"; F_norm=1 helps on
  hard tasks.
- **Limitations:** Anchor grouping needs *recurring identical states* (natural in ALFWorld/WebShop;
  needs fuzzy matching already for QA; questionable for open-web or SWE states); discount-based
  step returns are still outcome-derived, no quality-vs-cost tradeoff, no stopping semantics.
- **Relation to CASSI:** The most direct challenge to CASSI's "O(T) oracle vs O(K×T²) MC-PRM"
  efficiency framing: GiGPO already delivers per-step credit at O(T) hashing cost with zero extra
  rollouts *and* gets OTC-level tool economy for free. CASSI's rebuttal must be that GiGPO assigns
  credit for *task success*, not for *economic stopping* (no λ, no budget state, no controller),
  and fails when states don't recur. THREAT LEVEL: **MEDIUM-HIGH** — undercuts the efficiency
  novelty argument and part of the "process signal without extra rollouts" story.

### DAPO: An Open-Source LLM Reinforcement Learning System at Scale (Yu et al., ByteDance Seed & Tsinghua AIR, 2025, NeurIPS 2025; arXiv 2503.14476; ~2.2k cites)
- **Read from:** PDF pages 1–9 (`2503.14476_dapo.pdf`)
- **Problem:** Naive GRPO on Qwen2.5-32B reaches only 30 AIME vs DeepSeek's 47 — diagnoses and
  fixes the failure modes of large-scale GRPO.
- **Method / pathologies → fixes:** (1) **entropy collapse** → Clip-Higher (decouple ε_low=0.2,
  ε_high=0.28); (2) **vanishing gradients when a group is all-correct/all-wrong (zero advantage)**
  → Dynamic Sampling (oversample, drop acc∈{0,1} prompts); (3) **sample-level loss underweights
  long responses → gibberish/repetition growth and "unhealthy" length increase** → token-level
  policy-gradient loss; (4) **reward noise from truncated overlong samples** → Overlong Filtering
  and **Soft Overlong Punishment**, verbatim: `R_length(y) = 0 if |y| ≤ L_max−L_cache;
  ((L_max−L_cache)−|y|)/L_cache if in the interval; −1 if |y| > L_max` (L_max=16384,
  L_cache=4096), *added to* the ±1 correctness reward. KL term removed.
- **Training / RL usage:** GRPO-derived; rule-based `R = 1/−1` on integer-answer equivalence
  (DAPO-Math-17K).
- **Experiments & benchmarks:** Qwen2.5-32B base, verl; AIME24 avg@32.
- **Key results:** Naive GRPO 30 → +overlong filtering 36 → +clip-higher 38 → +soft overlong
  punishment 41 → +token-level loss 42 → +dynamic sampling **50** (beats R1-Zero-Qwen-32B's 47 at
  50% of steps).
- **Limitations:** Single-turn math; the length penalty is a *truncation-noise* device (fixed
  budget hygiene), not an economic policy — it fires only near the context limit.
- **Relation to CASSI:** DAPO's tricks are now default in agent RL (RAGEN, Nebius-SWE, ARPO
  baselines); CASSI's GRPO training should adopt/report them. Its soft overlong punishment is the
  most-cited "length penalty" in the family and the right citation for "static penalties exist but
  aren't per-instance economics". THREAT LEVEL: **LOW**.

### SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution (Wei et al., Meta AI, 2025, NeurIPS 2025; arXiv 2502.18449)
- **Read from:** PDF pages 1–7 (`2502.18449_swe-rl.pdf`)
- **Problem:** Scale rule-based RL to real-world software engineering without executable
  environments.
- **Method:** *Single-turn* (non-interactive) GRPO: prompt = issue + full file contexts; policy
  emits reasoning + search/replace edits. **Reward verbatim:** `R(o) = −1 if wrong format, else
  compare(patch_pred, patch_gt)` where compare = Python `difflib.SequenceMatcher` similarity ∈
  [0,1] against the oracle patch — continuous similarity, no execution, no cost term. KL kept.
- **Experiments & benchmarks:** 273k GitHub PR seeds; Llama-3.3-70B-Instruct, G=16 over 32
  problems/batch (batch 512), 1,600 steps, 16k ctx, **512 H100s ≈ 32 hours**. Eval: SWE-bench
  Verified via Agentless Mini scaffold (500 repair samples, rerank by 30 reproduction tests).
- **Key results:** **41.0% SWE-bench Verified** — then-best <100B open model, > GPT-4o (38.8 w/
  Agentless). Repair-only oracle-context: 34.8 vs base 5.4 / SFT 29.6. OOD gains on HumanEval+,
  CRUXEval, MATH, MMLU while SFT regresses. Sample-scaling: 33.6→41.0 as repair samples 40→500.
- **Limitations (for CASSI's purposes):** pipeline-not-agent (no multi-turn env interaction), 70B
  scale, similarity reward is noisy w.r.t. semantic correctness.
- **Relation to CASSI:** Feasibility anchor for the SWE-bench Verified plank of CASSI's plan — see
  synthesis: *interactive* SWE RL results cluster at 32B–72B (DeepSWE/SkyRL/Nebius), and even
  non-interactive needs 70B+massive compute for 41%. THREAT LEVEL: **LOW** (no economics), but a
  **feasibility red flag** for CASSI's 7B SWE-bench ambitions.

### ARPO: Agentic Reinforced Policy Optimization (Dong et al., RUC & Kuaishou, 2025, arXiv preprint 2507.19849; 116 cites)
- **Read from:** PDF pages 1–8 (`2507.19849_arpo.pdf`)
- **Problem:** Trajectory-level RL (GRPO/DAPO) undersamples the states that matter: token entropy
  **spikes in the first 10–50 tokens after every tool-call feedback** (search feedback > Python
  feedback). Sample there.
- **Method:** **Entropy-based adaptive rollout**: N global trajectories + budget M−N for partial
  branches; monitor step entropy change ΔH_t; branch probability P_t = α + β·ΔH_t, branch Z
  partial paths when P_t > τ. **Advantage attribution:** hard (average advantage over shared
  prefix tokens) vs soft (GRPO importance ratios naturally handle shared vs branched tokens; soft
  wins). GPG theorem provided as justification for macro-action segmentation. Claimed rollout
  complexity drops from O(n²) toward O(n log n).
- **Training / RL usage:** Reward verbatim (hierarchical, follows Tool-Star):
  `R = max(Acc + r_M, Acc) if format good & Acc>0; 0 if format good & Acc=0; −1 otherwise`, with
  **r_M = 0.1 if both <search> and <python> are used** — i.e. an explicit *multi-tool usage bonus*,
  the opposite sign of a cost penalty. Correctness = token-level F1. Cold-start SFT (Tool-Star 54K
  + STILL 0.8K), RL on 10K (reasoning) / **1K hard search samples** (SimpleDeepSearcher+WebSailor
  data).
- **Experiments & benchmarks:** 13 datasets: AIME24/25, MATH500, GSM8K…; HotpotQA, 2Wiki, MuSiQue,
  Bamboogle, WebWalker[QA]; deep search: **GAIA, HLE, xbench-DeepSearch** (WebSailor split).
  Qwen2.5-7B, Llama-3.1-8B, Qwen3-8B/14B. Baselines GRPO, DAPO, REINFORCE++.
- **Key results:** Beats trajectory-level RL across the 13 sets and — headline — "**achieves
  improved performance using only half of the tool-call budget required by existing methods**"
  (training-time tool-call consumption, Fig. 1 right: ~420 vs ~980 calls/step for GRPO at similar
  or better reward). Only 1K RL samples for deep-search competence (Qwen3-14B ARPO leads GAIA/HLE
  bars vs GRPO in Fig. 1).
- **Limitations:** "Tool-call budget" here = *training* rollout economy, not an inference-time
  stopping/cost policy; the reward actively encourages tool diversity; entropy branching adds
  rollout-scheduler complexity.
- **Relation to CASSI:** Shows the field's notion of "efficiency" is training-compute economy, not
  task economics — a framing gap CASSI can occupy. Its post-tool-call entropy finding independently
  supports CASSI's premise that post-observation states are the decision-critical ones the stopper
  should read. THREAT LEVEL: **LOW-MEDIUM** (terminology collision on "tool-call budget"; no
  inference-time cost policy).

### AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning (Xi et al., Fudan & ByteDance Seed, 2025, arXiv preprint 2509.08755)
- **Read from:** PDF pages 1–8 (`2509.08755_agentgym-rl.pdf`)
- **Problem:** No unified, SFT-free, end-to-end multi-turn RL framework across realistic
  environments; naive long-horizon training collapses.
- **Method:** Modular env/agent/training framework (HTTP server-client envs; built on AgentGym +
  verl) covering web navigation (WebArena), deep search (SearchQA-style), games (TextCraft),
  embodied (BabyAI), science (SciWorld); PPO/GRPO/RLOO/REINFORCE++ supported. **ScalingInter-RL**:
  progressive *interaction-horizon curriculum* — cap turns K ≤ h_t, small h_0 first (exploitation,
  master basics), then raise h in stages (exploration, richer behaviors), because starting with
  long horizons yields "model collapse… futile exploration… hallucinated tools".
- **Training / RL usage:** POMDP with **outcome reward r(τ) ∈ [0,1]** from the environment at
  episode end; no cost/efficiency term; GRPO default in experiments.
- **Experiments & benchmarks:** Qwen2.5-3B/7B mainly; 27 tasks / 5 scenarios.
- **Key results:** 7B + ScalingInter-RL: BabyAI 96.7, TextCraft 91.0, SciWorld 57.0, WebArena 26.0,
  Deep Search 38.2 — avg **+33.65 points over base**, "matching or surpassing" o3/Gemini-2.5-Pro on
  those suites (Fig. 1).
- **Limitations:** Horizon schedule is hand-designed and global (same cap for all instances — no
  per-instance difficulty adaptation); outcome-only reward; WebArena scores still low in absolute
  terms.
- **Relation to CASSI:** (i) candidate training infrastructure for CASSI's executor; (ii)
  ScalingInter-RL is the nearest thing to "horizon control as a training lever" — but it is a fixed
  curriculum, not a learned per-state stopping policy, and it optimizes learnability, not cost.
  CASSI's H5 (per-instance stopping correlates with difficulty) is a claim this line never tests.
  THREAT LEVEL: **LOW-MEDIUM**.

## Peripheral papers

- **DeepSeekMath (Shao et al., 2024; arXiv 2402.03300)** — origin of **GRPO**: PPO variant deleting
  the critic, baselining advantages by group mean/std over G samples per prompt; motivation was
  *memory efficiency* for 7B math training (51.7% MATH). All agentic work above inherits this
  estimator and its quirks (std normalization, per-sequence averaging). Read: abstract + prior
  knowledge; peripheral because non-agentic.

- **DeepSeek-R1 (DeepSeek-AI, 2025; arXiv 2501.12948; published in Nature, Sept 2025)** — showed
  pure outcome-reward RL (GRPO, accuracy + format rewards; no process rewards, explicitly rejecting
  PRMs for reward-hacking/scalability reasons) elicits long-CoT reasoning ("aha moments"). Its
  R1-Zero recipe is the ideological template for every "outcome-only" agent paper above; also the
  source of the response-length-inflation phenomenon that Dr. GRPO dissects. Abstract + prior
  knowledge.

- **Dr. GRPO / Understanding R1-Zero-Like Training (Liu et al., 2025; arXiv 2503.20783; ~1.2k
  cites)** — identifies an **optimization bias in GRPO: the 1/|o| length normalization and std
  normalization artificially inflate response length (especially for incorrect outputs)**; removing
  both (Dr. GRPO) preserves accuracy with far better token efficiency (43.3% AIME24 at 7B). The
  canonical citation for "GRPO itself manufactures length growth" — CASSI's cost analyses must
  control for estimator-induced length effects vs genuine overthinking. Abstract only.

- **R1-Searcher (Song et al., 2025; arXiv 2503.05592; 256 cites, preprint)** — two-stage
  outcome-based RL for search: stage 1 rewards *invoking retrieval correctly* (retrieval-count +
  format reward, no answer reward), stage 2 rewards answers; REINFORCE++-style, no process rewards,
  no cost term. Bing + Wikipedia variant. Peripheral canonical search-RL baseline. Abstract only.

- **StepSearch (Wang et al., 2025, EMNLP 2025; arXiv 2505.15107)** — step-wise PPO for search
  agents: token-level process supervision with **"information gain and redundancy penalties"** per
  search step, built from a 19k dataset with sub-question-level golden search trajectories; +11.2%
  (3B) / +4.2% (7B) absolute over global-reward RL baselines on multi-hop QA. The redundancy
  penalty is a genuine per-step efficiency-flavored term, but it needs *reference trajectories*
  (supervision CASSI's oracle avoids) and targets retrieval redundancy, not stopping economics.
  Abstract only. THREAT: MEDIUM-LOW.

- **WebAgent-R1 (Wei et al., 2025, EMNLP 2025; arXiv 2505.16421)** — end-to-end multi-turn RL for
  GUI/web agents with **binary task-success rewards** (rule-based) and asynchronous trajectory
  generation; WebArena-Lite success: Qwen2.5-3B 6.1%→33.9%, Llama-3.1-8B 8.5%→44.8%, beating o3.
  Notes test-time scaling with *more* interactions helps — again, no pressure toward economy.
  Abstract only.

- **WebSailor (Li et al., Alibaba/Tongyi, 2025; arXiv 2507.02592)** — post-training for
  BrowseComp-level web agents: obfuscated high-uncertainty task synthesis (SailorFog-QA), RFT cold
  start, and **DUPO** (Duplicating Sampling Policy Optimization — duplicates high-variance samples
  in-batch to densify agentic RL). Outcome rewards; no cost term. Peripheral: the "hard web tasks
  need data+RL, and RL is expensive per rollout" datapoint. Abstract only.

- **Kimi-Researcher (Moonshot AI, June 2025; tech report/blog, no arXiv)** — end-to-end agentic RL
  (REINFORCE, strictly on-policy, negative-sample filtering) on an internal Kimi model; ~23
  reasoning steps and 200+ URLs per task; HLE 26.9% Pass@1, xbench-DeepSearch 69%. **Reward =
  format + correctness + a gamma-decay efficiency term: outcome reward discounted as r·γ^(T−i)
  (γ∈(0,1)) so shorter correct trajectories earn more** — the clearest published *trajectory-length
  economy term* in frontier agent RL, plus turn-level partial rollouts and context management. No
  learned stopper, no per-state cost-value estimates, closed-source. Read via WebFetch of
  https://moonshotai.github.io/Kimi-Researcher/. THREAT: MEDIUM — establishes "efficiency-decayed
  outcome reward" as known practice; CASSI must position Δ(s_t) as strictly richer (per-step,
  quality-aware, budget-conditioned).

- **EAPO — Learning When Not to Act: Mitigating Tool Abuse in Agentic Reinforcement Learning (Chen
  et al., June 2026; arXiv 2606.02132)** — 2026 successor to OTC: an "Efficient Agentic Policy
  Optimization" framework learning *selective* tool use via (1) injected tool-free trajectories,
  (2) **difficulty-aware reward shaping penalizing redundant calls on easier queries**, (3)
  confidence-aware token reweighting. Reports +10.45/+7.27/+9.69 accuracy with 18.3–24.6% fewer
  tool calls on Qwen2.5-3B/7B, Llama-3.1-8B. Directly attacks the "static penalties ignore
  difficulty" gap CASSI claims (contribution 4). Single-model, reward-shaping only — still no
  learned stopping model, oracle labels, process-reward bridge, or inference controller. Abstract
  only (WebFetch). THREAT: **HIGH** for the per-instance-adaptivity contribution; must be cited and
  ablated against.

- **The Landscape of Agentic Reinforcement Learning for LLMs: A Survey (Zhang, Geng, Yu, Yin et
  al., 2025; arXiv 2509.02547; TMLR 01/2026)** — 500+-work survey; formalizes agentic RL as POMDPs
  vs degenerate single-step MDPs; taxonomy over capabilities (planning, tool use, memory,
  reasoning, self-improvement, perception) and domains; consolidates environments/frameworks. Use
  as the citation for field framing and for benchmark/framework tables. Read: PDF pages 1–2 +
  abstract.

- **From Reasoning to Agentic: Credit Assignment in RL for LLMs (Zhang, 2026; arXiv 2604.09459)** —
  2026 survey of **47 credit-assignment methods (2024–early 2026)**, taxonomized by granularity
  (token/segment/step/turn/multi-agent) × methodology (MC, TD, model-based, game-theoretic,
  information-theoretic); notes agentic horizons of 100+ turns / 100K–1M tokens make episode-level
  credit "increasingly uninformative". Useful map to check CASSI's O(T)-vs-O(K×T²) claim against
  TD-style and implicit-PRM alternatives. Abstract only.

- **SWE-agent RL at scale cluster (feasibility datapoints):**
  (i) *Nebius: Training Long-Context, Multi-Turn SWE Agents with RL* (Golubev et al., 2025; arXiv
  2508.03501) — RFT then **DAPO** on **Qwen2.5-72B-Instruct**, 131k ctx: SWE-bench Verified
  11%→**39%** Pass@1 (RFT baseline 20%);
  (ii) *SkyRL-Agent* (Cao et al., Berkeley, Nov 2025; arXiv 2511.16108) — SA-SWE-**32B** (from
  Qwen3-32B): 24.4%→**39.4%** Verified with >2× cost reduction vs comparable pipelines, async
  dispatcher 1.55× speedup;
  (iii) *DeepSWE-Preview* (Agentica/Together, July 2025; blog) — pure RL (rLLM/GRPO-variant) on
  **Qwen3-32B** over R2E-Gym: ~42.2% Verified (59% with hybrid TTS);
  (iv) *RepoNavigator* (Zhang et al., Dec 2025; arXiv 2512.20957) — single-tool (jump-to-definition)
  repo **localization** agent RL-trained from base models; 7B beats 14B baselines — the main
  evidence that *sub-task* SWE RL works at 7B;
  (v) *The Rollout Infrastructure Tax in Coding-Agent RL* (Graviet et al., July 2026; arXiv
  2607.01415) — 110× cold-start latency spread and 1.8× worker-hour spread across four execution
  substrates for 150-step SWE rollouts; execution infrastructure dominates coding-agent RL cost.
  All abstract-only reads. Collectively: **no published interactive SWE-bench-Verified RL success
  at 7B**; credible results start at 32B.

## Synthesis

### Landscape

The training stack is standardized: **Qwen2.5-Instruct/Base (1.5B–32B) or Qwen3 + verl-family
trainer + GRPO-variant + observation masking + rule-based outcome reward**, cold-start SFT/RFT
optional but increasingly common for hard domains (ReTool, ARPO, WebSailor, Nebius). PPO persists
where stability matters (Search-R1 default, RAGEN's Bandit/Sokoban); GiGPO/ARPO/DUPO/StarPO-S are
the 2025 wave of *agent-specific* GRPO surgery (credit assignment, exploration allocation,
variance filtering); ScalingInter-RL adds horizon curricula. Economy of action is an afterthought:
hard caps (B actions in Search-R1, 10 calls in DeepResearcher, ≤5 turns in RAGEN) plus at most one
of {multiplicative tool-count coefficient (OTC), gamma decay (Kimi-Researcher), redundancy penalty
(StepSearch), difficulty-aware shaping (EAPO), truncation shaping (DAPO)}.

### Reward designs (outcome / format / efficiency terms)

| Paper | Outcome term | Format term | Efficiency/cost term | Process/step-level? |
|---|---|---|---|---|
| Search-R1 (COLM'25) | EM(a_pred, a_gold) | none (deliberate) | **none** (hard action cap B) | no |
| R1-Searcher | stage-2 answer reward | stage-1 retrieval+format | none | no (two-stage schedule) |
| ToolRL (NeurIPS'25) | R_correct ∈ [−3,3], fine-grained tool/param match | R_format ∈ {0,1} | none; **length reward tested and rejected** (min(L_think/512,1) hurts small models) | partial credit within a call, not across steps |
| OTC-PO | r_φ (EM etc.) | optional | **multiplicative r_tool: cos/sin ramp peaking at hindsight-optimal call count n from group min** | trajectory-level coefficient only |
| ReTool | ±1 equivalence | in prompt only | none (length drop emergent) | no |
| DeepResearcher (EMNLP'25) | word-level F1 | −1 if malformed | none (10-call cap) | no |
| RAGEN/StarPO | env trajectory return | **−0.1 format penalty** | none; entropy bonus 0.001 | no (trajectory-level; calls for finer rewards) |
| GiGPO (NeurIPS'25) | env success/score | — | **implicit: γ-discounted step returns penalize detours** | **yes — anchor-state step advantages, zero extra rollouts** |
| DAPO (NeurIPS'25) | ±1 correctness | — | **soft overlong punishment** (linear −1 ramp in last 4k tokens before L_max) | no (token-level loss ≠ step reward) |
| SWE-RL (NeurIPS'25) | difflib similarity ∈[0,1] to oracle patch | −1 if malformed | none | no |
| ARPO | max(Acc+r_M, Acc), F1-based | −1 if malformed | **anti-efficiency: +0.1 bonus for using both tool types**; "half tool-call budget" = training rollout economy | soft advantage attribution over branch points |
| AgentGym-RL | env r(τ) ∈ [0,1] | — | none (ScalingInter horizon curriculum is training-side) | no |
| WebAgent-R1 (EMNLP'25) | binary task success | rule-embedded | none | no |
| Kimi-Researcher (blog) | correctness | format penalties | **γ-decay: r·γ^(T−i) favors shorter correct trajectories** | no |
| StepSearch (EMNLP'25) | global answer reward | yes | **step redundancy penalty + information gain (needs golden sub-trajectories)** | yes (token-level process supervision) |
| EAPO (2026) | correctness | yes | **difficulty-aware shaping penalizing redundant calls on easy queries; tool-free trajectory injection** | confidence-aware token reweighting |

Bottom line for CASSI Q(a): efficiency terms exist (OTC, Kimi, StepSearch, EAPO, DAPO-overlong),
but all are **reward-shaping on a single model**; no work trains a **separate learned
stopping/value model**, none produces a **per-step cost-aware value margin Δ(s_t)** as a process
reward, and none closes an oracle→stopper→executor loop. "Stopping bonuses" per se do not appear
anywhere in this literature — stopping is handled by hard caps or emergent `<answer>` emission.

### Standard base models & benchmarks (feasibility for CASSI)

- **Base models:** Qwen2.5-3B/7B(-Instruct) is the de-facto standard for search/QA agents
  (Search-R1, DeepResearcher, OTC, GiGPO, ToolRL, EAPO); Qwen2.5-32B for math-tool RL (ReTool,
  DAPO); Qwen3-8B/14B appearing in 2025H2 (ARPO); SWE RL requires 32B–72B today. Llama-3.1-8B /
  3.2-3B as the secondary family. CASSI's 7B–32B Qwen2.5 executor plan is squarely standard.
- **HotpotQA / MuSiQue:** universally used (Search-R1, OTC, GiGPO, DeepResearcher, ARPO train/eval;
  MuSiQue usually OOD-eval). Local E5+Wikipedia executor stack is fully reproducible at 7B on
  modest GPU counts. **Fully feasible; must include Search-R1, OTC-GRPO and GiGPO as baselines.**
- **WebWalkerQA:** used as an *eval* benchmark by ARPO and the WebSailor/Tongyi line (also
  reported by AgentGym-RL-adjacent deep-search work); training against it requires live-web
  serving à la DeepResearcher (50-node crawler cluster, caching, anti-crawl handling) — feasible
  but infra-heavy. Common pattern: train on ~1K–80K curated web-search samples, eval on
  WebWalkerQA/GAIA.
- **GAIA:** standard *evaluation-only* target for RL-trained deep-research agents (ARPO, WebSailor,
  Kimi-Researcher, AgentGym-RL deep-search); typically the text-subset split from WebThinker/HiRA.
  Nobody RL-trains *on* GAIA (163-task dev set, too small); CASSI should treat it as OOD eval of a
  web-trained executor.
- **SWE-bench Verified:** **RL training of interactive SWE agents at 7B is not established.**
  Published successes: 32B (DeepSWE 42.2%, SA-SWE-32B 39.4%), 70B–72B (SWE-RL 41.0% non-interactive
  on 512 H100s; Nebius 39% with DAPO at 131k ctx). 7B evidence exists only for sub-tasks
  (RepoNavigator localization). The 2026 "rollout infrastructure tax" paper shows execution
  substrate dominates cost (110× cold-start spread). **Recommendation: CASSI should either (i)
  demonstrate SWE at 14B–32B with a small task subset (R2E-Gym/SWE-Gym environments), (ii) use a
  localization/repair sub-task at 7B, or (iii) demote SWE-bench Verified to a stopper-as-controller
  (no executor RL) experiment — claiming full 7B SWE RL training would strain credibility.**
- **MATH-500 / BFCL:** MATH-500 standard everywhere (ARPO, ReTool-line); BFCL V3 is ToolRL/EAPO
  territory with per-step verifiable tool calls — both low-risk.

### Known GRPO pathologies for agents (checklist for CASSI's training section)

1. **Length/verbosity inflation from the estimator itself** — GRPO's 1/|o| and std normalization
   inflate length, especially of incorrect responses (Dr. GRPO); sample-level loss lets
   gibberish grow in long outputs (DAPO → token-level loss). Any "cost saving" CASSI reports must
   be measured against a Dr.GRPO/DAPO-hygienic baseline, or reviewers will attribute savings to
   fixing estimator bias.
2. **Entropy collapse / exploration death** (DAPO → clip-higher; ARPO branches at post-tool-call
   entropy spikes instead).
3. **Zero-advantage vanishing gradients** when all G rollouts tie (all-correct/all-wrong) — acute
   for agents where success is sparse (DAPO dynamic sampling; RAGEN/StarPO-S variance-based prompt
   filtering; WebSailor DUPO duplicates high-variance samples).
4. **Echo Trap in multi-turn RL** — reward-std collapse → repetitive templates → gradient spikes →
   irreversible collapse (RAGEN); GRPO reward collapse also observed by Search-R1 (PPO chosen as
   default for stability) and OTC ("GRPO less stable than PPO… our method can delay the early
   collapse").
5. **Credit-assignment dilution over horizons** — episode-level advantages uninformative at 30–50+
   steps (GiGPO, credit-assignment survey 2604.09459); fixed by anchor-state step groups (GiGPO),
   entropy-targeted branching (ARPO), or step process rewards (StepSearch) — CASSI's stopper-PRM is
   a fourth mechanism and should be compared to GiGPO explicitly.
6. **Truncation reward noise** — punishing overlong-truncated-but-sound trajectories destabilizes
   training (DAPO overlong filtering/soft punishment); directly relevant to CASSI's budget-capped
   rollouts (a trajectory stopped by the budget is *not* evidence the policy was wrong).
7. **Reward hacking of shaped terms** — additive tool penalties get hacked (OTC found additive
   unstable: model drops tools while wrong; multiplicative gating on correctness required); length
   rewards backfire (ToolRL); format-only compliance without genuine reasoning ("hallucinated
   reasoning", RAGEN). CASSI's additive R_t = α·Δ + β·progress + γ·format sits in the risk zone
   OTC explicitly abandoned — worth an additive-vs-multiplicative ablation.
8. **Tool-call explosion / cognitive offloading** absent explicit pressure (Search-R1 Fig. 2d; OTC;
   EAPO) — worsens with model scale (OTC).

### Gaps

1. No work trains a **separate small model to make stop/continue economic decisions** for an agent
   — stopping is hard caps or emergent; the monitor/stopper-as-PRM bridge is unoccupied.
2. Efficiency terms are **cost-only and quality-blind**: OTC/Kimi/EAPO penalize counts/length
   irrespective of the *marginal quality gain* of the next step; nobody computes a per-step
   quality_t − λ·cost_t frontier from finished trajectories (the closest, OTC's hindsight-min-n,
   collapses it to a single integer per question).
3. **No per-instance, budget-conditioned adaptivity at inference**: nothing conditions the policy
   or reward on a live budget state (HIGH/MED/LOW/CRITICAL tiers) — ScalingInter's horizon caps are
   global training curricula. (EAPO's difficulty-awareness is the nearest 2026 encroachment.)
4. **No self-reinforcing loop**: no paper feeds an improving learned evaluator back into executor
   RL for economics (iterative self-improvement exists for *task success* only, e.g. AgentEvol-style
   rejection tuning).
5. Step-level signals that do exist (GiGPO, StepSearch, ARPO) are outcome-derived or
   supervision-dependent; none transfer across executors the way a standalone 0.5B–3B stopper could.

### Top threats (ranked)

1. **OTC-PO (2504.14870)** — HIGH. Already does cost-aware agent RL with hindsight per-question
   optimal tool calls at zero extra rollouts, on CASSI's own benchmarks (NQ/HotpotQA + OOD
   MuSiQue/Bamboogle) and models (Qwen2.5-3B/7B). CASSI's novelty must rest on the per-step
   quality-aware margin, the separate stopper, the PRM bridge, and the loop — not on "cost in the
   reward" per se.
2. **EAPO (2606.02132, June 2026)** — HIGH. Difficulty-aware cost shaping directly undercuts
   CASSI's contribution 4 ("per-instance dynamic cost adaptation beats static penalties") as sole
   differentiator; CASSI needs it as a baseline (it is single-model reward shaping — exactly the
   arm CASSI's two-model ablation must beat).
3. **GiGPO (2505.10978)** — MEDIUM-HIGH. Neutralizes the "step-level signal without extra rollouts
   is novel/expensive-otherwise" framing (O(T) hashing, <0.002% overhead) and reports OTC-level
   tool economy with no cost term; CASSI must argue economic stopping ≠ success credit and cover
   non-recurring-state regimes.
4. **Kimi-Researcher gamma decay** — MEDIUM. Frontier-lab precedent for trajectory-shortening
   reward decay; blog-only, so citable but not a benchmark competitor.
5. **StepSearch (2505.15107)** — MEDIUM-LOW. Per-step redundancy penalties exist, but require
   golden sub-trajectories; CASSI's oracle labels are self-supervised.
6. **ARPO / AgentGym-RL** — LOW-MEDIUM. Occupy the "efficiency" and "horizon control" *terminology*
   (training-compute budget, horizon curricula); CASSI must sharply distinguish inference-time task
   economics from training-rollout economy in its writing.

### Opportunities

- **Baseline suite is ready-made and cheap at 7B:** Search-R1 (outcome-only), OTC-GRPO
  (multiplicative cost), GiGPO (implicit step credit), EAPO (difficulty-aware shaping), + DAPO
  hygiene — all on the same NQ/HotpotQA→MuSiQue/Bamboogle stack CASSI already plans. A Pareto
  (accuracy × cost) comparison across these five, which no paper has published, would itself be a
  contribution.
- **Motivating-evidence goldmine:** Search-R1's rising search-call curve, OTC's scale-worsens-
  offloading finding, ARPO's post-tool-call entropy spikes, and RAGEN's fine-grained-reward call
  can all be cited as the field's own diagnosis of the gap CASSI fills.
- **The stopper-as-PRM bridge remains unoccupied**, and the credit-assignment survey (2604.09459)
  gives a taxonomy slot ("step-granular, model-based, learned evaluator") where CASSI would be the
  first cost-aware entry.
- **Infrastructure:** verl / AgentGym-RL / SkyRL make the executor-RL half of CASSI reproducible;
  DeepResearcher shows a frozen auxiliary agent (their browsing agent) inside an RL loop is
  practical — precedent for CASSI's frozen-then-updated stopper in the loop.
- **SWE scope adjustment:** pivoting the SWE plank to 14B–32B on R2E-Gym/SWE-Gym subsets (or
  stopper-as-controller-only at 7B) both de-risks the plan and aligns with where the field
  demonstrably is (DeepSWE, SkyRL-Agent, Nebius).
