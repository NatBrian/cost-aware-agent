# Process Reward Models for Agents & Step-Level Credit Assignment

> Area owner: PRMs for agents / step-level credit assignment. Focus: CASSI contribution #3 —
> "O(T) oracle labels vs. O(K×T²) Monte Carlo PRM training" — plus (a) do cheap PRM-training
> methods already exist, (b) does any PRM encode cost, (c) do step-credit methods for agents
> already avoid extra rollouts. All core papers read from downloaded PDFs (research/papers/).
> Date of research: 2026-07-16.

## Area overview

Process reward models (PRMs) assign per-step scores to multi-step LLM outputs. The field started
in math reasoning with human step labels (Lightman et al. 2023) and moved to automatic labels via
Monte Carlo (MC) completion: from every prefix, roll out K completions and use the empirical
success rate as the step's Q-value (Math-Shepherd, ACL 2024; MiPS; OmegaPRM's MCTS variant). This
MC-completion family is the source of CASSI's O(K×T²) framing: K extra episode-continuations per
step × T steps per trajectory. Two waves of 2024–2026 work have attacked exactly this cost. First,
*implicit/TD-style labels*: Implicit PRM (ICML 2025) shows a PRM emerges for free from ORM
training with a log-likelihood-ratio parameterization (38.6× cheaper than Math-Shepherd-style
collection); PRIME makes this an online RL recipe (PRM updated from the same rollouts used for
policy training, zero extra cost); TDRM trains PRMs by temporal-difference bootstrapping; and —
critically for CASSI — the *other* AgentPRM (Fudan, WWW 2026) applies TD+GAE label estimation to
agent tasks explicitly to avoid per-state MC rollouts. Second, *rollout-free step credit inside
agent RL*: GiGPO (NeurIPS 2025) computes step-level advantages by retroactively grouping repeated
environment states across the existing group rollouts (zero extra rollouts); SPA-RL learns a
return-decomposition progress estimator by regression to the outcome; SALT builds a trajectory
graph; "Neglected Free Lunch" (Jun 2026) derives an agent step-advantage from the log-prob ratio
of the RL-trained policy vs. its reference — no reward model trained at all.

For agents specifically, the landscape splits by *how the step signal is obtained*: (i) fresh
per-state MC rollouts or search trees (Math-Shepherd-style ports, AgentRM's MCTS, Zhai et al.
Q-value models, IPR) — genuinely O(K×T) extra episodes per trajectory; (ii) pooled/reused
rollouts (Choudhury's AgentPRM computes MC return-to-go over a hashed state-action dictionary
built from the 10k–70k trajectories collected anyway for RL; GiGPO's anchor grouping is the same
trick applied directly to advantages); (iii) bootstrapped/implicit labels needing no extra
rollouts (WWW-2026 AgentPRM, PRIME, TDRM, progress-advantage); (iv) regression/redistribution
from outcome (SPA-RL, SALT). An important correction to CASSI's current plan: **Choudhury's
AgentPRM (2502.10325), the plan's primary comparison, does NOT launch K rollouts from every
state** — it reuses complete trajectories and averages discounted return-to-go over repeated
(s,a) visits. The "~160 extra executions per 20-step trajectory" characterization fits the
MC-completion family (and AgentRM/IPR/Zhai et al.), not AgentPRM as implemented.

On cost-awareness: across everything surveyed, **no trained PRM, value model, or step-credit
method encodes token/tool/dollar cost in its target or label**. All step signals estimate success
probability, progress toward success, or preference between actions. The closest neighbors are
HiPRAG (ICLR 2026: rule/LLM-checked over-search/under-search bonuses inside the RL reward of an
agentic-RAG system — an efficiency-shaped process *reward function*, not a learned cost-aware
value model), OTC (outcome-level tool-call penalty, covered by another area file), and emergent
effects (GiGPO reports its agents learn to suppress redundant tool calls without any cost term;
discounting γ<1 in AgentPRM/GiGPO is a weak implicit brevity pressure). A cost-aware stopping
margin Δ(s_t) = Q_continue − Q_stop used as a process reward remains unoccupied — but the
"cheap labels" part of CASSI's contribution #3 is no longer novel on its own.

## Core papers

### Process Reward Models for LLM Agents: Practical Framework and Directions (Sanjiban Choudhury, Cornell, 2025, arXiv 2502.10325, preprint "under review")
- **Read from:** PDF pages 1–16 (entire paper).
- **Problem:** How can LLM agents improve through interaction without extensive human supervision?
  Large-scale RL is impractical (long horizons, sparse rewards); PRMs offer dense turn-level
  supervision but are underexplored for agents acting in external environments (beam search over
  known transitions, as in math PRMs, is impossible).
- **Method (label acquisition — read carefully):** Turn-level MDP; the PRM is defined as the
  Q-function Q^π(s_t,a_t)=E[Σ γ^{k−t} r_k]. Three-stage loop, iterated K=3 times: (1) roll out
  current policy π_{i−1} many times per task (10k trajectories per iteration on 3,257 ALFWorld
  games; a 70k-rollout setting is used to study reward hacking), store every visited (s,a) in a
  hashed dictionary G(s,a); compute targets as the *average discounted return-to-go over all
  stored trajectories passing through (s,a)* (Eq. 1) — i.e., pooled first/every-visit MC from
  complete episodes, **not fresh rollouts from each state**. The paper explicitly rejects MCTS as
  "requires synchronous exploration and is difficult to scale," collecting rollouts
  asynchronously instead. (2) Train PRM with soft binary cross-entropy on Q̂ (BT preference loss
  ablated — similar performance; the relative-loss dataset is small because "there are far fewer
  states that are visited multiple times"). (3) Policy update via Online DPO maximizing Q while
  KL-regularized to π_{i−1} (conservative policy iteration argument). **No cost anywhere**:
  reward is terminal task success in [0,1]; no stopping decision; γ discounting is the only
  (implicit) length pressure. Also proposes **InversePRM**: learn process rewards directly from
  expert demonstrations via IRL — PRM parameterized as Q-difference Q(s,a)−γQ(s′,a′)
  discriminates expert transitions D+ from learner transitions D− (no outcome reward needed).
- **Training / RL usage:** PRM+policy both Llama-3.2-3B; Online DPO; Best-of-N (N=16) at
  inference with the PRM as ranker. Reward shaping variant: blend PRM target with reference-policy
  advantage A^μ (α=0.5) to stabilize low-sample training.
- **Experiments & benchmarks:** ALFWorld only (134 out-of-distribution games, max 30 actions);
  baselines: BUTLER, ReAct (gpt-4o 65.7%, claude-3.5-sonnet 76.1%), Autogen, ExpeL, Reflexion,
  AdaPlanner.
- **Key results:** π_2 85.8%, π_3 88.1% success; BoN 91.0% — a 3B model beating claude-3.5-sonnet
  (76.1%). Average #actions 12.0–12.7 vs. 19–25 for ReAct baselines (efficiency emerges but is
  not optimized). InversePRM: 82.8% after ONE iteration vs. AgentPRM 73.9% (with 70k rollouts),
  near expert (91%). Reward hacking demonstrated when PRM trained on only 10k rollouts: success
  falls 82%→70% while the PRM's own reward keeps rising; mitigated at 70k.
- **Limitations:** single benchmark (ALFWorld) with a discrete, revisitable state space — the
  hashed-dictionary trick needs repeated (s,a) visits and would degrade in open-ended web/SWE
  settings; 10k–70k rollouts per iteration is still a large sample cost; no cost/economy, no
  stopping semantics; preprint (not peer-reviewed).
- **Relation to CASSI:** CASSI's primary comparison and the anchor of contribution #3. Overlap:
  small PRM supervises/trains an executor with RL; iterative co-training loop (its 3-stage
  iteration is arguably already a "self-reinforcing cycle" for success-only rewards — CASSI's
  cycle novelty must rest on the *cost-aware oracle*, not the loop shape). Differences: no cost,
  no stopping model, success-only Q. **THREAT LEVEL: HIGH** — not because it undercuts the cost
  angle (it doesn't), but because CASSI's plan currently *mischaracterizes* its label cost
  ("K rollouts from every state, ~160 extra executions") — reviewers who know the paper will
  reject the O(K×T²) contrast as a strawman unless it is re-aimed at MC-completion methods
  (Math-Shepherd-style, AgentRM, IPR, Zhai et al.) and the residual claim vs. AgentPRM is
  restated as "0 extra rollouts and no state-revisit requirement vs. 10k–70k pooled rollouts."

### AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress (Xi, Liao, Li et al., Fudan/Ant Group, 2025/2026, WWW 2026, arXiv 2511.08325)
- **Read from:** PDF pages 1–14 (method, experiments, related work).
- **Problem:** (same name, different paper!) PRMs for agent tasks face: no clear-cut step
  "correctness"; steps are interdependent (a login detour is locally regressive but globally
  necessary); and "previous methods for training PRMs often depend on either expert annotations
  or extensive Monte Carlo-based sampling for estimation, both of which are costly."
- **Method (label acquisition):** PRM = two heads/objectives: **promise** (Q-value: expected
  future success, MSE loss L_Q) and **progress** (advantage A = Q(s_t,a_t) − Q(s_{t−1},a_{t−1})
  under sparse deterministic-transition assumptions, MSE loss L_A); total L = L_Q + β·L_A (β=1).
  Labels via **TD-based estimation with GAE**: sample N_TD=16 plain trajectories per query from
  the policy; bootstrap targets from the PRM's own predictions — δ_t = r_t + γM_φ(s_t,a_t) −
  M_φ(s_{t−1},a_{t−1}), Â_t = Σ(γλ)^k δ_{t+k} (λ=0.95), Q̂_t = Â_t + M_φ(s_{t−1},a_{t−1});
  terminal Q̂(s_T,a_T) = r(u,τ); iterate batch-sample → estimate with current model → update.
  Explicit claim: "TD-based estimation with GAE does not require additional rollouts from each
  state like MC-based method." The MC baseline they implement is exactly the O(K×T²) scheme:
  N_Traj=1 seed trajectory, N_mc=16 fresh rollouts from *every* step's successor state.
- **Training / RL usage:** PRMs on Qwen2.5-0.5B/3B (also 7B, Llama-3.1-8B policies); used for
  (a) Best-of-N and step-level beam search (@N×M) at inference, (b) **PPO training of the
  executor** with AgentPRM as the reward (§5.2, BabyAI + TextCraft, Qwen2.5-3B) — more stable
  and higher task score than ORM/PVM rewards.
- **Experiments & benchmarks:** WebShop (max 6 turns), BabyAI (20), TextCraft (20) on AgentGym;
  GSM8K transfer. Baselines: SFT, RFT, ORM, PVM (per-step value), Math-Shepherd MC estimation.
- **Key results:** Qwen2.5-3B beam search 8×8: WebShop 76.0 vs. PVM 54.5 / ORM 57.0; BabyAI 89.8;
  TextCraft 56.7 vs. ORM 43.3. Claimed **8× more compute-efficient** than ORM/PVM baselines at
  matched Best-of-N performance. **Label-cost comparison (Table 3): MC-based estimation uses
  1.9× (WebShop), 2.8× (BabyAI), 1.5× (math) more sampled tokens than their TD-based labels,
  and TD still scores higher** (e.g., WebShop BoN@64 74.0 vs. 72.0; beam 8×8 76.0 vs. 70.5).
- **Limitations:** TD targets are bootstrapped from the PRM's own (initially wrong) predictions —
  biased, needs iterative re-estimation; assumes deterministic transitions and sparse terminal
  reward for the advantage identity; short horizons (≤20 turns); no ground-truth per-step
  quality; **no cost/economy, no stopping**; peer-reviewed (WWW 2026).
- **Relation to CASSI:** the single most direct occupant of the "cheap agent-PRM labels"
  territory: an agent PRM whose *stated contribution* is avoiding per-state MC rollouts, with
  token-cost tables, published at WWW 2026 (before CASSI submission). CASSI's O(T)-vs-O(K×T²)
  framing cannot be presented as first; the defensible residual is: oracle labels are
  *ground-truth-anchored* (per-step quality vs. gold answer, no bootstrap bias, no iterative
  re-estimation) and *economic* (encode λ·cost and a stopping margin), neither of which TD+GAE
  provides. **THREAT LEVEL: HIGH** — directly attacks contribution #3's efficiency novelty;
  must be cited and differentiated on label semantics, not label cost.

### Free Process Rewards without Process Labels (Yuan, Li, Chen, Cui et al., UIUC/Tsinghua, 2024, ICML 2025 — PMLR v267, arXiv 2412.01981)
- **Read from:** PDF pages 1–5 (theory, setup); abstract/figures for the rest.
- **Problem:** PRM training requires per-step labels; MC-completion collection (Math-Shepherd
  style: 10-step rollouts × 8 completions per step = 80 trajectories per instruction) costs ~80×
  an ORM's data and is noisy (hard estimation overestimates Q, soft underestimates).
- **Method (label acquisition):** **zero step labels, zero extra rollouts.** Parameterize the
  outcome reward as r_θ(y) = β log π_θ(y)/π_ref(y) and train an ORM on response-level labels
  (any loss: DPO, KTO, NCA, CE). Proposition 3.1: q_θ^t = Σ_{i≤t} β log ratio is then an exact
  expectation of the outcome reward at step t (a Q-function); process reward r^t = q^t − q^{t−1}
  falls out at inference for free, at token granularity. Prop 3.2: the implicit Q is sandwiched
  between MC hard/soft estimates, mitigating both biases.
- **Training / RL usage:** train on 33K math instructions × 8 sampled solutions (needed anyway
  for the ORM); evaluate as Best-of-N ranker on MATH-500 across Mistral-7B, Llama-3.1-8B/70B
  generators. CE instantiation is most data-efficient (works with one response per instruction).
- **Key results:** outperforms Math-Shepherd reimplementation using **less than 1/38 of the
  training FLOPs** (data collection + training); Figure 1: 38.6× overhead reduction with +2.9%
  accuracy; extra Math-Shepherd step labels bring *no further improvement* on top of outcome-only
  training. Reference model can be dropped at inference with little accuracy loss.
- **Limitations:** math reasoning; states are text prefixes generated by the model itself — in
  agent settings prefixes contain environment observations not generated by the policy, so the
  log-ratio telescoping needs re-derivation (done later by "Neglected Free Lunch," see
  peripherals); rewards are success-based; no cost.
- **Relation to CASSI:** the canonical proof that "PRMs need expensive step labels" is false in
  general. CASSI must scope its efficiency claim to *agentic, environment-coupled* PRMs and
  argue why implicit rewards are insufficient there (observation-conditioned prefixes, need for
  calibrated absolute stopping margins rather than relative rankers, cost integration).
  **THREAT LEVEL: HIGH** (conceptual): any reviewer can ask "why not an implicit PRM trained on
  cost-penalized outcome labels?" — CASSI needs an explicit answer/baseline.

### PRIME: Process Reinforcement through Implicit Rewards (Cui, Yuan, Wang et al., Shanghai AI Lab/Tsinghua/UIUC, 2025, arXiv 2502.01456, preprint v2 Sep 2025)
- **Read from:** PDF pages 1–4 (method); abstract/intro results.
- **Problem:** dense rewards for online RL are blocked by: step labels are prohibitively
  expensive to collect online ("estimation-based methods require about 10× more rollouts for
  each step"); PRMs go stale/hackable unless updated online; explicit RM training adds cost.
- **Method (label acquisition):** **zero extra rollouts, online.** Initialize policy and Implicit
  PRM from the same SFT model; each RL step: sample rollouts, grade with ground-truth outcome
  verifier, **update the Implicit PRM online with CE loss on those same rollouts + outcome
  labels**, compute token-level process rewards r_φ(y_t) = β log π_φ(y_t|y_<t)/π_ref(y_t|y_<t),
  fuse with outcome reward in any MC advantage estimator (they choose RLOO-style leave-one-out;
  no value network).
- **Training / RL usage:** Qwen2.5-Math-7B-Base after light SFT; RL on competition math + code.
- **Key results:** +15.1% avg over SFT across reasoning benchmarks; **2.5× sample efficiency and
  +6.9% final performance vs. outcome-reward-only RL (RLOO)**; Eurus-2-7B-PRIME surpasses
  Qwen2.5-Math-7B-Instruct on 5 math benchmarks using 10% of its data; online PRM update shown
  essential (§5.1); works with REINFORCE/RLOO/PPO/GRPO.
- **Limitations:** single-turn verifiable reasoning (math/code), not multi-turn agents with
  environment observations; token-level log-ratio rewards are relative and policy-coupled (not a
  calibrated stopping value); no cost.
- **Relation to CASSI:** shows "dense process rewards at zero extra label cost, updated online
  inside the RL loop" is established practice — including the "PRM improves as policy improves"
  co-evolution that CASSI's cycle claim echoes. CASSI's distinct pieces remain the cost-aware
  oracle target and the stopping controller. **THREAT LEVEL: HIGH** (framing) / MEDIUM
  (technical overlap, different domain).

### Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations (Wang, Li, Shao et al., PKU/DeepSeek, 2024, ACL 2024, arXiv 2312.08935)
- **Read from:** PDF pages 1–4 (method, formulation).
- **Problem:** PRMs beat ORMs but human step annotation (PRM800K) is unaffordable.
- **Method (label acquisition — the O(K×T²) anchor):** define step quality as "potential to
  deduce the correct answer": from each step s_i of each solution, a completer decodes **N
  subsequent full completions** (N=8 in experiments); label = 1 if any completion reaches the
  gold answer (hard) or the fraction that do (soft). Per solution of T steps this is N×T extra
  completions, each up to T steps long — the K×T² step-generation cost CASSI cites; the Implicit
  PRM paper measures it as 38.8× ORM FLOPs.
- **Training / RL usage:** PRM (7B) trained with per-step BCE; used (a) as Best-of-N verifier,
  (b) as step-by-step reward in PPO.
- **Key results:** PPO with Math-Shepherd: Mistral-7B 77.9→84.1% GSM8K, 28.6→33.0% MATH;
  +verification best-of-256: 89.1% GSM8K / 43.5% MATH; DeepSeek-67B reaches 93.3% / 48.1%.
- **Limitations:** label noise (false positives: bad step rescued by strong completer;
  hard-estimation overestimates Q); enormous sampling cost; math-only; no cost in reward.
- **Relation to CASSI:** the legitimate target of the O(K×T²) contrast — CASSI should aim the
  efficiency claim here (and at its agent ports: Zhai et al. Q-value models, IPR, AgentRM
  explicit RM) rather than at Choudhury's AgentPRM. **THREAT LEVEL: LOW** (it is the strawman
  that makes CASSI look good; but its very age (2023) means "we're cheaper than Math-Shepherd"
  is a 2024-era contribution — implicit/TD methods already made that point).

### OmegaPRM: Improve Mathematical Reasoning in Language Models by Automated Process Supervision (Luo, Liu, Liu et al., Google DeepMind, 2024, arXiv 2406.06592, preprint v2 Dec 2024)
- **Read from:** PDF pages 1–3 (intro, related work, algorithm sketch).
- **Problem:** per-step MC estimation (Math-Shepherd/MiPS) is too expensive at scale — "their
  efficiency remains limited due to the vast search space."
- **Method (label acquisition):** divide-and-conquer MCTS: for each question build a search tree;
  **binary search locates the first error** in a CoT (a rollout with correct answer certifies all
  earlier steps; incorrect certifies an error exists later/earlier half), balancing positive and
  negative examples; per-node statistics (visit counts, MC accuracy) cached and reused. Cost per
  labeled solution drops from O(N×T) completions to roughly O(N×log T), though still rollout-based.
  Collected **1.5M process annotations** fully automatically.
- **Training / RL usage:** PRM used with weighted self-consistency at inference (no RL of the
  generator in this paper).
- **Key results:** Gemini Pro: 51%→69.4% MATH500, 86.4%→93.6% GSM8K; Gemma2-27B: 42.3%→58.2%
  MATH500, 74.0%→92.2% GSM8K.
- **Limitations:** still needs many fresh rollouts (binary search reduces the multiplier, not the
  paradigm); assumes a verifiable final answer and a resumable, deterministic "environment"
  (text prefix) — binary-search resumption from arbitrary prefixes is exactly what real agent
  environments make hard/irreversible; no cost in reward.
- **Relation to CASSI:** weakens a naive "MC labeling must be O(K×T²)" statement — the best MC
  methods are already sub-quadratic. CASSI's complexity table should include the O(K×T×logT)
  middle ground. **THREAT LEVEL: MEDIUM** — a reviewer citing OmegaPRM can call the T² framing
  outdated even within the MC family.

### GiGPO: Group-in-Group Policy Optimization for LLM Agent Training (Feng, Xue, Liu, An, NTU/Skywork, 2025, NeurIPS 2025, arXiv 2505.10978)
- **Read from:** PDF pages 1–9 (method, experiments, ablations, group dynamics).
- **Problem:** group-based RL (GRPO/RLOO) gives one trajectory-level advantage — no step-level
  credit for long-horizon agents; per-state rollouts for step credit (Figure 1 middle) are
  "prohibitively expensive."
- **Method (step credit without a PRM and without rollouts):** two-level advantages from the SAME
  N=8 rollouts GRPO already collects. Episode level: standard group-normalized return A^E.
  Step level: **anchor state grouping** — hash every environment state; identical states
  recurring across the group's trajectories become anchors; all actions taken from an anchor
  form a step-group; each action's discounted return-to-go R_t = Σ γ^{k−t} r_k is normalized
  within the step-group to give A^S. Final advantage A = A^E + ω·A^S (ω=1). "Entirely offline…
  lightweight key-based grouping using hashmaps"; for open-ended states, similarity grouping
  (longest matching subsequence ≥ 0.9). Overhead < 0.002% of training time. Critic-free, no
  auxiliary model at all.
- **Training / RL usage:** direct advantage for PPO-style clipped objective; Qwen2.5-1.5B/3B/7B.
- **Experiments & benchmarks:** ALFWorld, WebShop, search-augmented QA (NQ, TriviaQA, PopQA,
  HotpotQA, 2Wiki, MuSiQue, Bamboogle) vs. PPO, RLOO, GRPO, Search-R1, ZeroSearch, StepSearch.
- **Key results:** ALFWorld 1.5B: 86.7% vs. GRPO 72.8% (+13.9); 7B: 90.8%; WebShop success 1.5B:
  67.4% vs. 56.8%; QA avg 42.1% (3B) / 47.2% (7B) vs. Search-R1 38.5% (7B). **Tool-efficiency
  emerges without any cost term**: ~0.9 tool calls/query single-hop, ~1.6 multi-hop (matching
  OTC's explicitly optimized numbers), attributed to grouping exposing redundant repeated calls.
  >65% of states recur across the group early in training (group-size distribution analysis).
- **Limitations:** needs identical initial state + task per group and *state recurrence* within
  the group — fine for ALFWorld/WebShop, brittle for open-web/SWE where states rarely repeat
  (similarity hack is a partial fix); no persistent value model (signal exists only where groups
  overlap, only during training); success-only; no stopping semantics.
- **Relation to CASSI:** the flagship "step-level credit for agents with ZERO extra rollouts"
  paper at a top venue — kills any blanket claim that agent step-credit requires extra rollouts,
  and its emergent tool-call reduction weakens (slightly) the motivation that agents can't learn
  economy without a cost signal. CASSI's differentiators: a persistent, deployable stopping/value
  model (GiGPO's credit is ephemeral, in-batch), ground-truth per-step quality, explicit
  cost-awareness, per-instance budgets. **THREAT LEVEL: HIGH** — must be cited as the
  rollout-free credit baseline; ideally compared against (CASSI's plan currently does not list
  GiGPO as a baseline).

### SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution (Wang, Leong, Wang, Wang, Li, HK PolyU, 2025, arXiv 2505.20732, preprint "under review")
- **Read from:** PDF pages 1–7 (method, experiments).
- **Problem:** delayed sparse rewards make PPO ineffective for long-horizon agents (exponentially
  vanishing GAE weights (γλ)^{n−1−t} on early actions; degenerate value learning); existing
  step-supervision (StepAgent, PRM4A) optimizes locally/greedily.
- **Method (label acquisition):** **reward redistribution via regression — no per-state
  rollouts.** (1) BC on expert trajectories → base agent. (2) Base agent does M=10 plain rollouts
  per training task → D_explore. (3) **Progress estimator** (Llama-3.2-3B + MLP head on last
  hidden state) outputs per-step contribution ĉ_t; trained with MSE so that Σ_t ĉ_t ≈ final
  outcome reward R (RUDDER-style return decomposition for LLM agents). (4) Per-step reward
  fused with a grounding signal g_t (action executability): r_t = α·ĉ_t + β·g_t (α=1, β=0.5).
- **Training / RL usage:** PPO (LoRA) with the fused dense reward replacing the sparse terminal
  reward; GAE over dense rewards.
- **Experiments & benchmarks:** ALFWorld, WebShop, VirtualHome; baselines SFT, RFT, PPO, ArCHer,
  StepAgent, RAGEN, PRM4A.
- **Key results:** ALFWorld unseen 79.1% success (PPO 73.9, StepAgent 75.4, RAGEN 75.4, PRM4A
  73.9); grounding accuracy 91.7–93.7%; WebShop 64.1% (RAGEN 63.0); VirtualHome 53.4%. Ablation:
  w/o stepwise progress 77.6, w/o grounding 77.6.
- **Limitations:** progress estimator learned from only M=10 rollouts/task and outcome MSE —
  credit is a learned attribution, not verified; sum constraint forces conservation but not
  correctness; 3B models, short-to-medium horizons; no cost term; preprint.
- **Relation to CASSI:** an *agent-specific process-reward-like signal trained with zero extra
  rollouts* — its "progress toward completion" is the success-side analog of CASSI's per-step
  quality_t; CASSI adds the cost axis and stopping margin, and grounds quality in gold answers
  rather than learned attribution. **THREAT LEVEL: MEDIUM-HIGH** — occupies "dense agent rewards
  without MC rollouts" and should be a baseline for the process-reward-bridge ablation.

### AgentRM: Enhancing Agent Generalization with Reward Modeling (Xia, Fan, Chen et al., Tsinghua, 2025, ACL 2025 main (2025.acl-long.945), arXiv 2502.18407)
- **Read from:** PDF pages 1–5 (method, experiments).
- **Problem:** finetuning the policy on diverse tasks degrades held-out tasks; finetuning a
  *reward model* to guide test-time search generalizes better.
- **Method (label acquisition — MC/tree-search based):** compares three RM constructions:
  (1) **Explicit RM (best):** per task instruction, build an MCTS-style search tree (UCB
  selection; expansion samples k actions, merging identical ones; **simulation: n complete
  rollouts from each expanded node**; backprop of V and visit counts, ω iterations); extract
  state-values V(s_t), filter low-visit states, train LM + value head with MSE. (2) Implicit RM
  (Yuan et al. parameterization, 16 complete trajectories per instruction, MSE on progress
  rate). (3) LLM-as-a-judge (training-free). Trained on 3 held-in tasks (WebShop, ALFWorld,
  SciWorld) from LLaMA-3-8B SFT-agent explorations.
- **Training / RL usage:** no policy RL — Best-of-5 and step-level beam search (W1=5, W2=5) at
  test time, guiding both untuned and finetuned policies, incl. weak-to-strong (8B RM guiding
  LLaMA-3-70B).
- **Key results:** Explicit RM Best-of-5: +8.8 avg over greedy across 9 tasks (4 types),
  surpassing top general agent by 4.0; beam search: 63.3 overall; held-in Webshop 71.0/ALFWorld
  94.8/SciWorld 76.1 (Best-of-5), 75.3/96.3/82.6 (beam); +12.6 on LLaMA-3-70B; beats specialized
  agents (QLASS, Agent-R) on all three held-in tasks. Notably: "the efficiency of explicit
  reward modeling is not necessarily inferior to that of implicit reward modeling" (their data-
  scaling analysis) — a counterpoint to the implicit-PRM narrative *within agent tasks*.
- **Limitations:** tree construction needs resettable/replayable environments and many rollouts
  per instruction (k expansions × n simulations × ω iterations); value filtered by visit count;
  test-time-search only (no executor training); no cost.
- **Relation to CASSI:** the best current example of the *expensive* (search-tree/MC) agent-RM
  pipeline — a legitimate target of CASSI's O(K×T²) critique, and evidence that in agents the
  MC/tree route still wins over implicit RM (useful for defending why cheap-but-grounded labels
  matter). **THREAT LEVEL: LOW-MEDIUM** — supports rather than undermines CASSI's framing, but
  its implicit-RM comparison must be acknowledged (implicit was competitive on held-in tasks).

### CSO: Verified Critical Step Optimization for LLM Agents (Li, Zeng, Fang et al., Tencent AI Lab/HKU, 2026, ACL 2026 Findings, arXiv 2602.03412)
- **Read from:** PDF pages 1–6 (method, experiments).
- **Problem:** outcome-only preference training (ETO/RFT) has coarse credit; step-level PRM
  scoring is noisy; full per-step MC (IPR) is "prohibitively expensive"; only a small fraction of
  steps are pivotal (high-entropy-token principle).
- **Method (label acquisition — selective, verified branch rollouts):** start from **failed**
  policy trajectories. At each step: expert model (Claude-3.7-Sonnet) proposes k=5 alternative
  actions; a **prompted PRM** (Claude-3.7-Sonnet with rubric — not a trained model) scores policy
  action and alternatives; candidate critical step iff policy score < γ_low=0.45 and best
  alternative > γ_high=0.65. For each candidate: **branch rollout** — splice in the expert
  alternative, let the *policy itself* continue to termination, verify against ground-truth
  outcome. Only outcome-verified flips (fail→success) become DPO pairs (s_t, a+, a−). Iterative
  (2 rounds), π_ref updated each round.
- **Training / RL usage:** DPO on verified critical-step pairs (β=0.5); policy CK-Pro-8B
  (Qwen3-8B SFT) in Cognitive Kernel Pro framework.
- **Experiments & benchmarks:** GAIA-Text-103, XBench-DeepSearch (100 tasks); baselines GPT-4.1,
  Claude-3.7-Sonnet, RFT, ETO, Step-wise DPO (implemented per Choudhury 2025), IPR.
- **Key results:** GAIA-Text 49.5% overall (+37% relative over SFT; matches GPT-4.1);
  XBench +26% relative; ≥ +5.0 points over all post-training baselines (IPR 44.6); supervision
  needed at only **16% of trajectory steps**. Ablation: expert-positive + policy-negative pairs
  best.
- **Limitations:** still needs branch rollouts (extra executions) plus k=5 expert-API proposals
  and prompted-PRM scores at every step of every failed trajectory — cheaper than IPR but far
  from free; relies on a frontier closed model as expert/PRM; DPO-offline (no online RL); no
  cost-awareness; success-flip-only definition of criticality.
- **Relation to CASSI:** CASSI's plan cites CSO as related agent-RL work; its economics
  (selective supervision at 16% of steps) shows the field is already compressing step-label
  cost along a different axis (selectivity rather than post-hoc computation). No overlap with
  stopping/cost. **THREAT LEVEL: LOW-MEDIUM** — mainly a framing constraint: "extra rollouts"
  in 2026 are targeted, not exhaustive, so CASSI's cost table should include the selective
  regime.

## Peripheral papers

**Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents (2026, arXiv
2606.26080, preprint).** Argues that for agents, both human annotation and MC estimation of step
values are "infeasible at scale," then shows RL post-training already yields a step-level score
for free: under a general *stochastic* MDP, the log-probability ratio between the RL-trained
policy and its reference policy recovers the optimal advantage function ("progress advantage") —
extending the implicit-PRM identity from text prefixes to environment-coupled agent trajectories.
Annotation-free, domain-agnostic, a byproduct of standard RL; validated on test-time scaling,
uncertainty quantification, and failure attribution over five benchmarks / four model families,
beating confidence baselines and *dedicated trained reward models*. THREAT: HIGH for the claim
that agent step-values require any dedicated labeling at all; CASSI's answer must be that a
policy-derived relative advantage is neither calibrated for absolute stop/continue decisions nor
cost-aware. (Read from abstract; June 2026, very recent.)

**HiPRAG: Hierarchical Process Rewards for Efficient Agentic RAG (Wu et al., 2025, ICLR 2026,
arXiv 2510.07794).** The closest existing thing to a *cost-aware process reward*: decomposes
agentic-RAG trajectories into parsable steps and adds hierarchical bonuses (on top of outcome +
format rewards) for the proportion of "optimal" search/non-search decisions, where each step's
necessity is checked on-the-fly (knowledge-grounded rule/LLM check for over-search: searching
when the model already knows; under-search: answering without needed evidence). No trained PRM,
no extra rollouts. Qwen2.5/Llama-3.2: 65.4% (3B) and 67.2% (7B) avg EM over seven QA benchmarks
with over-search rate cut to 2.3%. THREAT: MEDIUM-HIGH to "no PRM incorporates cost" — CASSI must
narrow its claim to *learned, budget-conditioned economic value* (HiPRAG penalizes a binary
redundancy rule, has no budget state, no dollars/tokens, no stopping model). (Read from abstract
+ fetched summary.)

**TDRM: Smooth Reward Models with Temporal Difference for LLM RL and Inference (Tsinghua/THUDM,
2025, arXiv 2509.15110, preprint).** Trains PRMs by minimizing temporal differences along
trajectories (n-step TD with a target network), yielding temporally consistent, smooth reward
landscapes without per-step MC labeling. Best-of-N +6.6%, tree search +23.7%; combined with RLVR
reaches baseline performance with 2.5k vs. 50.1k data (≈20× data efficiency) across 8 models.
Math-reasoning domain; shows TD-style PRM training is an established, general alternative to MC.
THREAT: MEDIUM (same mechanism as WWW-AgentPRM but non-agentic).

**SALT: Step-level Advantage Assignment for Long-horizon Agents via Trajectory Graph (2025,
arXiv 2510.20022, preprint).** Plug-and-play module for group-based RL (GRPO/RLOO): builds a
graph from the group's trajectories for the same prompt (merging identical states, like a
GiGPO variant), quantifies per-step quality from outcome rewards propagated on the graph, and
reweights advantages — "no modifications to the rollout procedure and negligible computational
overhead." Evaluated on WebShop, ALFWorld, AppWorld. Confirms rollout-free step credit for
agents is now a crowded design space. THREAT: MEDIUM (redundant with GiGPO for CASSI purposes).

**ARPO: Agentic Reinforced Policy Optimization (2025, arXiv 2507.19849, preprint).** Observes
token-entropy spikes right after tool calls; proposes entropy-based *adaptive rollout*: branch
extra partial rollouts only at high-entropy post-tool steps (global + step-level sampling mix)
plus advantage attribution so stepwise tool-use differences are internalized. Across 13
reasoning/deep-search benchmarks beats trajectory-level RL **using only half the tool-call
budget** of baselines. Notable for CASSI: it *does* care about tool-call cost, but as a training
sampling-budget optimization, not as a task-time reward signal or stopping decision. THREAT:
LOW-MEDIUM (efficiency-of-training, not economy-of-behavior).

**CARL: Criticality-Aware Agentic Reinforcement Learning (Shen et al.?, 2025, arXiv 2512.04949,
preprint; authors per arXiv listing).** Uses entropy as a proxy for state criticality: assigns
rewards/updates only to actions from high-criticality states and drops low-criticality ones from
the update, avoiding noisy credit and redundant computation in long-horizon agentic RL; reports
stronger performance and higher training efficiency. Same family as ARPO/CSO's "only a few steps
matter." No PRM, no extra rollouts, no cost in reward. THREAT: LOW (orthogonal mechanism;
verify author list before citing — arXiv listing did not clearly show "Shen").

**Agent-RRM: Exploring Reasoning Reward Model for Agents (Fan et al., 2026, arXiv 2601.22154,
preprint).** Generative reward model for agent trajectories producing a reasoning trace, a
targeted critique, and an overall process score; three integration schemes (Reagent-C/R/U) for
refinement, reward-augmented guidance, and unified feedback. Reagent-U reaches 43.7% GAIA and
46.2% WebWalkerQA. Trajectory-level generative judging (LLM-as-judge lineage), not step-value
estimation; no rollout-based labels, no cost. Relevant to CASSI only as the "generative RM"
alternative for the stopper's rationale output. THREAT: LOW.

**Q♯: Provably Optimal Distributional RL for LLM Post-Training (Zhou et al., 2025, arXiv
2502.20548, preprint).** Value-based KL-regularized post-training: learns the optimal regularized
Q via distributional RL on an aggregated online dataset and uses it to guide the reference
policy; provably optimal for KL-regularized RL, variance-dependent bounds; beats value baselines
on math with smaller KL. Shows principled Q-learning (bootstrapped, no per-state MC) for LLMs;
single-turn, no cost. THREAT: LOW (theoretical support for TD-style labels' legitimacy).

**SWE-Search: MCTS + Iterative Refinement for Software Agents (Antoniades et al., 2024, arXiv
2410.20285 v6, preprint/ICLR-era).** Multi-agent MCTS over repo-level SWE tasks with a *hybrid
value function* — LLM-estimated numerical value + qualitative explanation per node — plus
discriminator debate; +23% relative over baseline agents on SWE-bench across five models. The
value function is prompted, not trained, and search multiplies execution cost at *inference*
(the opposite of CASSI's cheap-supervision goal); no training-time credit assignment. THREAT:
LOW (background for "value functions for agents").

**A Survey of Process Reward Models (2025, arXiv 2510.08049, preprint).** Systematizes the full
PRM loop — process-data generation (human, MC estimation, tree search, implicit), PRM
architectures (discriminative, generative), and uses (test-time scaling, RL) — across math,
code, multimodal, robotics, and agents; catalogs benchmarks (ProcessBench, PRMBench). Its
taxonomy confirms: label-generation methods are exactly {human, MC/tree rollouts, implicit/TD};
*none* of the surveyed reward targets includes action/tool/token cost. Useful citation for the
"no cost-aware PRM exists" gap claim. Related recent benchmark: ToolPRMBench (2601.12294)
evaluates PRMs for tool-using agents (fine-grained action-level labels), again success-only.

## Synthesis

### Landscape

Three regimes of step-signal acquisition now coexist. (1) **Rollout-heavy**: MC completion
(Math-Shepherd: N=8 fresh completions per step; ≈38.8× ORM FLOPs), MCTS trees (OmegaPRM
sub-quadratic via binary search; AgentRM's UCB trees with n simulations per node; SWE-Search at
inference). (2) **Rollout-reuse**: Choudhury's AgentPRM (pooled return-to-go over hashed (s,a)
revisits from 10k–70k trajectories), GiGPO/SALT (anchor-state or graph grouping of the existing
GRPO group), SPA-RL (M=10 rollouts/task + regression redistribution), CSO (branch rollouts at
only ~16% of steps). (3) **Rollout-free/bootstrapped**: Implicit PRM/PRIME (log-ratio ORM ⇒
token-level process rewards; online updates from policy rollouts), TDRM and WWW-2026 AgentPRM
(TD(+GAE) bootstrapping), progress advantage (free byproduct of RL post-training). The field's
direction of travel 2024→2026 is unambiguous: away from per-state MC, toward implicit/TD/grouped
signals — *in both reasoning and agent settings*.

### Label-acquisition cost per method

| Method (year, venue) | Extra env rollouts for labels? | Label cost per trajectory (T steps, K samples) | Signal type | Cost/economy in target? |
|---|---|---|---|---|
| Math-Shepherd (ACL'24) | Yes — N completions from every step | O(N·T) episodes ≈ O(N·T²) step-gens (≈38.8× ORM FLOPs) | success prob (hard/soft) | No |
| OmegaPRM (2024) | Yes — MCTS binary search | ≈O(N·T·log T) step-gens | success prob + tree stats | No |
| AgentRM explicit (ACL'25) | Yes — UCB tree, n sims/expanded node, ω iters | O(ω·n·T) episodes per task | state value V(s) | No |
| IPR (2024, per CSO) | Yes — MC from each step | O(N·T) episodes | step reward | No |
| CSO (ACL'26 F) | Partial — branches at ~16% of steps of failed trajs | O(0.16·T) partial episodes + k=5 expert calls/step | verified preference pairs | No |
| AgentPRM Choudhury (2025 preprint) | No fresh per-state; needs 10k–70k pooled trajectories w/ (s,a) revisits | O(K·T) total (K = revisits needed for stable Q̂); labels O(1) post-hoc per visit | Q (success) | No (γ only) |
| GiGPO (NeurIPS'25) | **No** (group of N=8 reused) | O(N·T) hashing, <0.002% time | in-batch step advantage | No (efficiency emerges) |
| SALT (2025) | **No** | O(N·T) graph build | in-batch step advantage | No |
| SPA-RL (2025) | **No** (M=10 plain rollouts/task) | O(M·T) + regression | learned progress ĉ_t (Σĉ=R) | No |
| Implicit PRM (ICML'25) | **No** | 0 beyond the K responses already sampled (1/38 of Math-Shepherd FLOPs) | implicit Q / token reward | No |
| PRIME (2025) | **No** (online, reuses RL rollouts) | 0 extra | implicit token reward | No |
| TDRM (2025) | **No** | O(K·T) TD passes | smoothed PRM | No |
| AgentPRM WWW'26 | **No** (N_TD=16 plain trajectories) | O(N·T); 1.5–2.8× fewer tokens than MC | Q ("promise") + advantage ("progress") | No |
| Progress advantage (2026) | **No** | 0 (byproduct of RL) | implicit optimal advantage | No |
| HiPRAG (ICLR'26) | **No** (on-the-fly step checks) | O(T) LLM/rule checks | rule bonus for search necessity | **Efficiency yes** (over/under-search), no budget/λ |
| **CASSI oracle (proposed)** | **No** | O(T) argmax + O(T) per-step quality evals vs. gold | cost-aware stopping margin Δ, t* | **Yes** (λ·cumcost, budget tiers) |

### Does any PRM incorporate cost?

No trained PRM/value model encodes cost in its *target*: every learned step signal above
estimates success probability, progress, or preference. The near-misses: **HiPRAG** puts search-
necessity (an efficiency proxy) into a hand-designed hierarchical process *reward function*
(ICLR 2026 — must be cited); **OTC** (covered in the tool-use file) penalizes tool calls at the
*outcome* level; **ARPO/CARL** spend the *training* rollout budget cost-adaptively (entropy-
targeted) without task-time cost semantics; **GiGPO** shows tool-call frugality emerging with no
cost term at all (a result CASSI's motivation section must reckon with); γ-discounting in
AgentPRM/GiGPO is an implicit uniform step tax. Nothing conditions on remaining budget, λ, or
per-instance cost state, and nothing models the *stop-vs-continue margin*.

### Gaps (defensible space for CASSI)

1. **Cost-aware label semantics**: no existing label scheme computes "marginal economic value of
   continuing" (quality − λ·cumcost margin). All rollout-free competitors estimate success-side
   quantities only.
2. **Ground-truth anchoring without bootstrap bias**: TD/implicit labels inherit the current
   model's biases and are relative/policy-coupled; CASSI's oracle uses measured per-step quality
   vs. gold — calibrated absolute targets, no iterative re-estimation, no state-revisit
   requirement (vs. Choudhury/GiGPO hashing).
3. **A persistent, deployable stopping model**: GiGPO/SALT credit is ephemeral (in-batch);
   implicit PRMs are rankers; nobody ships a small budget-conditioned controller usable at
   inference *and* as a process reward.
4. **Per-instance budget conditioning** (tiers, λ multipliers) is absent from every PRM.

### Top threats to the O(T)-vs-O(K×T²) claim (ranked)

1. **AgentPRM (WWW 2026, 2511.08325)** — an *agent* PRM whose headline is TD+GAE labels avoiding
   per-state MC rollouts, with measured 1.5–2.8× token savings vs. MC and 8× inference-compute
   efficiency; also used as PPO reward. The efficiency contribution as currently worded is
   pre-empted; CASSI must pivot to label *semantics* (cost, ground truth, stopping).
2. **GiGPO (NeurIPS 2025)** — step-level credit for agent RL with zero extra rollouts and
   emergent tool-frugality; direct counterexample to "step credit needs rollouts"; missing from
   CASSI's baseline list.
3. **Implicit PRM (ICML 2025) + PRIME** — "free process rewards from outcome labels" is now
   textbook; reviewers will demand an implicit-PRM(-with-cost-penalized-outcome) baseline and an
   argument why it fails for stopping (calibration, observation-coupled prefixes).
4. **Progress advantage (2606.26080)** — extends the implicit identity to stochastic agent MDPs,
   annotation-free, beats trained RMs on failure attribution/test-time scaling; newest and
   least-known, but reviewers scanning 2026 arXiv will find it.
5. **Choudhury's AgentPRM itself** — the plan's "~160 extra executions per 20-step trajectory"
   description is factually wrong for that paper (pooled reuse, not per-state MC); an informed
   reviewer reads this as a strawman. Redirect the O(K×T²) contrast to Math-Shepherd-style/
   AgentRM/IPR, and state the honest AgentPRM contrast: 10k–70k trajectories + revisit
   requirement vs. 0 extra and no revisit requirement.
6. **OmegaPRM** — even within the MC family, binary search already gives O(K·T·logT); the
   quadratic strawman is outdated without qualification.
7. **HiPRAG (ICLR 2026)** — falsifies an unqualified "no process reward considers efficiency";
   narrow the claim to *learned, budget-conditioned economic value models*.

### Opportunities

- Reframe contribution #3 as a 2×2: {rollout-free vs. rollout-heavy} × {success-only vs.
  cost-aware} — CASSI is alone in the (rollout-free, cost-aware) cell, and additionally offers
  ground-truth-anchored calibration that TD/implicit methods lack. Include the cost table above
  (with OmegaPRM's logT middle ground) to preempt "strawman" reviews.
- Add baselines: GiGPO (rollout-free step credit), PRIME-style implicit PRM with cost-penalized
  outcome reward (the strongest cheap ablation of CASSI's oracle), SPA-RL-style progress
  regression, and HiPRAG-style rule bonus — beating these isolates the value of the economic
  margin Δ.
- Exploit AgentPRM's reward-hacking finding (10k-rollout PRM hacks; 70k needed) as evidence that
  success-only pooled-MC labels are data-hungry, whereas oracle labels are exact given the
  trajectory — a *quality*, not just cost, argument.
- GiGPO's emergent tool-frugality and AgentPRM's low #act suggest part of CASSI's cost savings
  may come for free from success-optimizing RL — a control experiment (cost metrics of
  success-only-trained agents) is needed to attribute savings to the cost-aware signal.
