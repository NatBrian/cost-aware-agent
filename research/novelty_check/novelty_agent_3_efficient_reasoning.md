# Novelty Check — Agent 3: Efficient-Reasoning Expert

> Reviewer persona: efficient-reasoning subfield specialist (length control, overthinking mitigation,
> token budgets, adaptive thinking, 2024–2026). Target: `research/paper_plan.md` (CASSI, v5).
> Date: 2026-07-16. Independence: did not read `research/archived_*`, other `lit_review/` files
> (except the sanctioned `00_paper_plan_summary.md`), or other `novelty_check/` files.

---

## My understanding of the proposal

CASSI trains a small (0.5B–3B) "stopping model" on labels computed post-hoc from completed agent
trajectories: `t* = argmax_t [quality_t − λ·cumulative_cost_{1..t}]`, an O(T) computation requiring
zero extra rollouts. Steps before t* are labeled CONTINUE, steps at/after t* STOP, with a continuous
margin Δ(s_t) ∈ [−1,1]. The stopping model (SFT + GRPO) then serves two roles: (a) during executor
training, its Δ(s_t) is injected as a step-level *cost-aware process reward* into GRPO training of a
7B–72B tool-using executor; (b) at inference, it is enforced as a stop/continue/adjust controller
with claimed <3% overhead. The authors call the resulting loop (better executor → better trajectories
→ better oracle labels → better stopper → better process rewards) a "self-reinforcing cycle" and
claim it is the first of its kind. Benchmarks: GAIA, WebWalkerQA, HotpotQA, MuSiQue, SWE-bench
Verified, MATH-500 (control), BFCL.

From my subfield's perspective, the proposal stakes four efficiency-relevant claims: (1) existing
length-penalty RL (L1, "Training LMs to Reason Efficiently", BudgetThinker) is "static,
instance-blind" with "no mid-trajectory adaptation"; (2) a *separate* learned model is structurally
necessary because execution and economic self-evaluation representations conflict; (3) per-instance
dynamic cost adaptation ("spend more on hard, less on easy", H5: r > 0.5 difficulty correlation) is
a contribution; (4) a small learned monitor supervising a large executor with negligible overhead is
new. The plan's related-work Section 6.5 cites only L1, Reason Efficiently, TALE, SelfBudgeter,
BudgetThinker, DiffAdapt from my subfield — a February–May 2025 snapshot of a field that moved
substantially between March 2025 and mid-2026.

---

## Core claims extracted (efficiency lens)

| # | Claim (as stated in plan) | Where |
|---|---|---|
| C1 | First self-reinforcing cycle: learned stopping model's Δ used as cost-aware process reward to train executor RL | §4.2, §13.1-1 |
| C2 | Separate stopping model is *necessary* ("representation conflict"); single-model cost-penalty RL fundamentally fails | §4.1, §13.1-2 |
| C3 | Oracle O(T) post-hoc labels vs O(K×T²) MC-rollout PRM training; zero extra executions | §8, §13.1-3, §20 |
| C4 | Static penalties (L1, Reason Efficiently, BudgetThinker) are "instance-blind", "no mid-trajectory adaptation"; CASSI's dynamic per-instance adaptation outperforms them; H5 difficulty–stopping-point correlation r>0.5 is "the defining characteristic of per-instance adaptation" | §4.1, §5.2, §11.1, §13.1-4 |
| C5 | Small stopper (0.5B–3B) supervises large executor (7B–72B), <3% overhead, 20–40% savings | §13.1-5 |
| C6 | Baseline set (L1, Reason Efficiently, s1, BATS, CaRT, AgentPRM-cost, BudgetThinker, Ares-style P2) represents the current field | §11.4 |

---

## Search log (queries + notable hits)

All papers named below were actually fetched (abstract via arXiv API/WebFetch/WebSearch) or read
from PDFs in `research/papers/`. Local PDFs read: 2502.04463 (Reason Efficiently), 2506.05256 (ALP),
2503.04472 (DAST), 2505.13417 (AdaptThink), 2505.18822 (AdaCtrl), 2505.11225 (HAPO), 2505.15612
(LASER-D), 2508.09726 (GFPO), 2604.05164 (TAB), 2605.29893 (RedundancyBench), 2511.08325
(AgentPRM-Fudan), 2504.14870 (OTC), 2603.07915 (Ares). Downloaded this session: 2511.09158 (CRM),
2505.02811 (SIM-RAG).

| Query (tool) | Notable hits |
|---|---|
| "AutoThink Thinkless adaptive thinking RL difficulty" (WebSearch) | AutoThink 2505.10832; AdaptThink 2505.13417; Omni-AutoThink 2512.03783; overthinking deviation-monitoring 2603.14251 (Mar 2026); survey 2507.09662 |
| "ConciseRL LLM judge conciseness reward RL" (WebSearch) | ConciseRL 2505.17250 (EMNLP-Findings 2025); **"Efficient Reasoning via Reward Model" 2511.09158**; Concise Reasoning via RL 2504.05185 |
| WebFetch 2511.09158 abs + html | CRM: Qwen2.5-3B conciseness reward model, SFT on 72B-judge labels; reward R̂=R°·[1+α·c·(s+d_q)], d_q per-question difficulty coefficient |
| WebFetch 2603.14251 | training-free mid-generation overthinking termination via path-deviation index |
| "OTC-PO tool productivity RL" (WebSearch) + local PDF | **OTC 2504.14870**: "pioneering RL-based framework that explicitly optimizes for both the efficiency and effectiveness of tool-integrated reasoning"; per-question optimal tool calls approximated by min calls among correct rollouts |
| "MRT meta reinforcement fine-tuning progress reward" (WebSearch) | **MRT 2503.07572**: dense per-episode progress rewards (Δ success probability) for token-efficient test-time compute; SHAPE 2604.06636 |
| "step-level length reward SmartThinker" (WebSearch + WebFetch) | **SmartThinker 2507.04348**: SCPO = online step-importance estimator + step-level length rewards + S-GAE + difficulty-adaptive clipping; ExpThink 2605.07501; step-level advantage selection 2604.24003 |
| "efficient agents RL tool call cost 2026" (WebSearch) | Efficient Agents 2508.02694; Tool-R1 2509.12867; Trajectory Reduction 2509.23586; efficiency survey 2601.14192; ToolOrchestra 2511.21689 |
| arXiv API `"when to stop" AND "LLM agent"` | **Agentic Abstention 2606.28733 (Jun 2026)**; **Semantic Early-Stopping for Iterative LLM Agent Loops 2606.27009 (Jun 2026)**; Calibrate-Then-Act 2602.16699; CaRT (Oct 2025); AgentAbstain 2607.10059 (Jul 2026) |
| arXiv API `"cost-aware" AND "process reward"` | **zero hits** — supports the specific gap |
| arXiv API `"value model" AND "early stopping"` | zero relevant hits |
| arXiv API SIM-RAG / sufficiency critic | **SIM-RAG 2505.02811 (SIGIR 2025)**: trained lightweight per-round information-sufficiency critic |
| "BudgetThinker SelfBudgeter Elastic Reasoning" (WebSearch) | BudgetThinker 2508.17196 (control tokens announcing remaining budget mid-generation); SelfBudgeter 2505.11274 (self-predicts budget up-front) |
| "Ares controller discrete effort" (WebSearch + local PDF) | **Ares 2603.07915 (Mar 2026)**: Qwen3-1.7B per-step reasoning-effort router for agents (TAU-Bench, BrowseComp-Plus, WebArena), SFT on min-effort labels mined from trajectories; 52.7% reasoning-cost cut at iso-success |
| S2 API "adaptive length penalty" | ALP (44 citations); **Leash** (ACL 2025, adaptive length penalty + reward shaping); "Thinking Fast and Right" (adaptive rewards) |
| arXiv API `ti:"Thinkless"` | Thinkless 2505.13379 ("LLM Learns When to Think"); ThinkLess 2505.15684 (training-free); FROST 2601.19001 |

---

## Subfield accuracy audit: is CASSI's characterization of length-control RL fair?

**Verdict: fair for February 2025; false for 2026.** The plan's blanket statement — "Existing
cost-aware training adds a length penalty R = R_task − α·len(a). This is a static, instance-blind
scalar" (§4.1) — describes only the earliest generation. Two independent errors:

**(a) The strawman mischaracterizes even its named targets.** Arora & Zanette's own paper (local
PDF, NeurIPS 2025) states "our framework allows models to adapt computational resources based on
the difficulty of the problem" and contains a dedicated "Difficulty based analysis" appendix
(bigger reductions on GSM8K than AIME). A static penalty *coefficient* does not produce
instance-blind *behavior*, because the policy conditions on the instance — trained models shorten
easy problems more than hard ones. CASSI conflates the two throughout.

**(b) The field fixed instance-blindness in mid-2025 and moved to step-level and agent-level
signals by 2026.** Full audit table (all entries verified from PDFs or fetched abstracts):

| Method (arXiv, date) | Penalty/control type | Instance-adaptive? | Mid-trajectory? | Learned signal? |
|---|---|---|---|---|
| L1/LCPO (2503.04697, 3/25) | RL penalty vs. prompt-specified target length | Per-prompt target (user-set, not difficulty-derived) | No | No |
| Reason Efficiently (2502.04463, NeurIPS'25) | Fixed-α length reward on correct answers | Coefficient: no. **Trained behavior: yes (their own App. I)** | No | No |
| DAST (2503.04472, v3 1/26) | Per-problem Token-Length-Budget from sampled solve rate; budget-aware reward + SimPO | **Yes** (difficulty via solve rate) | No | No (rollout formula) |
| ALP (2506.05256, 6/25) | Penalty magnitude ∝ inverse online per-prompt solve rate | **Yes** — "5.35× more tokens on hard vs easy" | No | No |
| LASER-D (2505.15612, 5/25) | Dynamic difficulty-aware step-function length rewards | **Yes** (difficulty buckets, adapts during training) | No | No |
| HAPO (2505.11225, 5/25) | Reward vs. per-question historical min correct length | **Yes** (per-question history state) | No | No |
| AdaptThink (2505.13417, 5/25) | Constrained RL choosing Think vs NoThink | **Yes** (mode per instance) | No (chosen at start) | No |
| Thinkless (2505.13379, 5/25) | DeGRPO over `<short>`/`<think>` control tokens | **Yes** | No | No |
| AutoThink (2505.10832, 5/25) | Multi-stage RL, stage-wise reward shaping | **Yes** | No | No |
| AdaCtrl (2505.18822, v2 12/25) | Self-estimated difficulty tags + difficulty-aware RL budgets (−91% GSM8K, −10% AIME24 length) | **Yes** (self-assessed) | Tag at generation start | Self-assessment |
| SelfBudgeter (2505.11274, 5/25) | Model predicts own budget, then adheres | **Yes** (pre-commit) | No | Own prediction |
| BudgetThinker (2508.17196, 8/25) | Control tokens announcing remaining budget inserted periodically | Budget per instance | **Yes** (periodic mid-generation budget state) | No |
| GFPO (2508.09726, 8/25) | Group filtering by length & reward-per-token; Adaptive-Difficulty variant | **Yes** | No | No |
| ConciseRL (2505.17250, EMNLP-F'25) | LLM-judge conciseness score as RL reward | Context-aware per trace | No (response-level) | **Yes (prompted judge)** |
| **CRM (2511.09158, 11/25)** | **Trained** 3B conciseness reward model (SFT on 72B-judge labels) multiplying outcome reward; difficulty coefficient d_q=exp(\|correct\|/G) | **Yes** (d_q per question) | No (response-level) | **Yes (trained RM)** |
| MRT (2503.07572, 3/25) | Dense progress reward per thinking episode (Δ prob. of eventual success) for budget-agnostic efficiency | Progress-dependent | **Yes** (episode segments within the trace) | No (rollout estimate) |
| SmartThinker (2507.04348, 7/25) | Step-level length rewards modulated by online step-importance estimator (SCPO, S-GAE, difficulty-adaptive clipping) | **Yes** | **Yes (step-level)** | Partially (online estimator) |
| Leash (ACL 2025, via S2) | Adaptive length penalty + reward shaping | **Yes** | — | No |
| **TAB (2604.05164, 4/26)** | **Separate GRPO-trained budget-allocation policy** choosing per-turn budgets for a solver LLM under global constraint | **Yes** | **Yes (sequential, per turn)** | **Yes (learned policy)** |
| **Ares (2603.07915, 3/26)** | **Separate 1.7B router predicting per-step reasoning-effort level for tool/web/research agents**; SFT on min-effort labels mined post-hoc from trajectories | **Yes** | **Yes (per step, agent loops)** | **Yes (trained router)** |
| **OTC-PO (2504.14870, 4/25)** | RL reward scaling correctness by tool-call efficiency vs per-question hindsight-optimal call count | **Yes** (min calls among correct rollouts per question) | No (trajectory-level) | No — but **trains the agent policy for cost** |
| **SIM-RAG (2505.02811, SIGIR'25)** | **Trained lightweight sufficiency critic** deciding continue/stop each retrieval round (self-practice data, outcome-labeled) | **Yes** | **Yes (per round)** | **Yes (trained critic)** |
| DEER (cited in plan), Semantic Early-Stopping (2606.27009, 6/26), deviation monitoring (2603.14251, 3/26) | Training-free confidence/embedding-drift mid-generation stopping | Yes | **Yes** | No |

Rows in bold are the ones that break specific CASSI claims. Summary: *instance-adaptive penalties*
are the 2025 norm, not a gap; *mid-trajectory* adaptation exists at step level (SmartThinker, MRT,
BudgetThinker) and at agent-step level (Ares, SIM-RAG, CaRT, TAB); *learned* efficiency signals
exist both as judges (ConciseRL) and as trained reward models (CRM). What the table does NOT
contain: any method that converts a learned, cost-aware stopping *value* into a *step-level process
reward used to train* a tool-using executor. That cell is genuinely empty.

---

## Closest prior work table

| Paper | Year | Venue | Overlap with CASSI | Key Difference |
|---|---|---|---|---|
| **OTC: Optimal Tool Calls via RL** (2504.14870) | 2025 | arXiv/OpenReview | RL-trains a tool-using agent to be cost-efficient on QA benchmarks (CASSI's exact domain); per-question "optimal" cost derived post-hoc from completed rollouts (hindsight, like CASSI's oracle); self-described "pioneering RL framework explicitly optimizing efficiency+effectiveness of tool use" | Single model; trajectory-level reward scaling, no step-level Δ; tool-call count only (no tokens/$); no separate stopping model; no stop/continue decision |
| **Ares: Adaptive Reasoning Effort Selection** (2603.07915) | 2026 | arXiv (Mar) | Small learned model (1.7B) supervising a large agent's per-step compute in tool/web/deep-research loops; labels mined post-hoc from trajectories; 52.7% cost cut at iso-success = CASSI's contribution #5 almost verbatim | Discrete effort levels, not stop/continue value; SFT router, no RL; executor never trained (inference-time only); no cost-λ oracle |
| **TAB: Turn-Adaptive Budgets** (2604.05164) | 2026 | arXiv (Apr) | Separate budget policy π_φ trained with GRPO making *sequential, mid-trajectory* compute decisions for a solver under a global budget; reward = accuracy − λ·overage | Multi-turn math, not tool agents; allocates budgets, doesn't stop; solver frozen (no process-reward feedback); no oracle labels |
| **SIM-RAG** (2505.02811) | 2025 | SIGIR | Trained lightweight critic makes per-round continue/stop sufficiency decisions in multi-round retrieval; self-practice trajectory data, no human labels | Labels from outcome success only (no cost term in labels); critic controls inference, never becomes a reward for training the generator |
| **CRM: Efficient Reasoning via Reward Model** (2511.09158) | 2025 | arXiv (Nov) | **Trained reward model providing efficiency signal inside RLVR training of the policy**; per-question difficulty coefficient | Response-level conciseness score, not step-level stopping value; single-shot math, not agents; no stopping decision, no controller role |
| **MRT** (2503.07572) | 2025 | arXiv (Mar) | Dense mid-trajectory *progress* rewards explicitly for token-efficiency of test-time compute ("progress = Δ probability of eventual success" ≈ CASSI's Q_continue−Q_stop intuition) | Progress estimated via extra rollouts (not O(T), not a learned model); single-shot math; no stopping model |
| **AgentPRM (Fudan): Step-Wise Promise and Progress** (2511.08325) | 2025 | arXiv (Nov) | Step-level PRM for *agents* trained with **TD+GAE labels (no per-state MC rollouts), "8× more compute-efficient"**; applied to RL of LLM agents | Quality/progress only — no cost term, no stopping; but it already removes the O(K×T²) bottleneck CASSI positions against |
| **ALP** (2506.05256) | 2025 | arXiv | "Spend more on hard, less on easy" as a *trained* behavior with explicit allocation analysis (21% of budget on easy half; 5.35× hard/easy ratio) = H5's phenomenon | Token-level, single-shot; penalty formula from solve rates, no learned monitor, no agents |
| **CaRT** (Oct 2025, cited in plan) | 2025 | arXiv | Trains agent termination ("know when they know enough") via SFT + counterfactuals | Same-model self-termination; no executor RL from stopping signal (plan correctly notes this) |
| **Semantic Early-Stopping for Agent Loops** (2606.27009) | 2026 | arXiv (Jun) | Mid-loop stopping for iterative agents on HotpotQA; oracle-stop analysis; 38% token cut at parity quality with a *training-free* embedding-drift stopper | No learning at all — which is exactly why it is the most dangerous cheap baseline for CASSI's savings numbers |
| **Agentic Abstention** (2606.28733) / **AgentAbstain** (2607.10059) | 2026 | arXiv (Jun/Jul) | Define/benchmark sequential when-to-stop-acting for tool agents; distilled stopping rules (CONVOLVE) | Evaluation + prompting-level; no trained stopper, no RL — but shows the agent-stopping space is being claimed *now* |
| **RedundancyBench** (2605.29893) | 2026 | arXiv (May) | Step-level redundant/necessary labels for agent trajectories — the exact quantity CASSI's monitor must detect | Benchmark only; best detector scores 24.88%, signaling CASSI's monitor task is much harder than the plan assumes |

---

## Per-claim novelty verdicts

**C1 — Self-reinforcing cycle (stopper-as-cost-aware-PRM → executor GRPO): MEDIUM.**
Closest: CRM (2511.09158) — a trained efficiency reward model inside policy RL, response-level;
AgentPRM-Fudan (2511.08325) — step-level agent PRM with cheap labels applied to agent RL,
quality-only; SIM-RAG/Ares — learned stopping/effort controllers, inference-only. Delta: nobody
composes "learned stopping-value" + "step-level process reward" + "executor RL" + "agents". That
composition survives my search (the `"cost-aware" AND "process reward"` query returns zero). But
every edge of the "cycle" exists separately, and the "self-reinforcing cycle" framing is rhetorical
inflation of one iteration of standard actor-critic alternation (the plan's own §10.1 admits it
follows "the PRM training pattern from AgentPRM"). Novel composition, not a novel paradigm.

**C2 — Representation conflict / separate model necessary: LOW (as claimed), MEDIUM (as ablation).**
My subfield's evidence points the other way: AdaptThink, Thinkless, AutoThink, AdaCtrl,
SelfBudgeter, BudgetThinker, CaRT all show a *single* model can learn economic self-assessment
(difficulty estimation, budget adherence, when-to-think, when-to-terminate) alongside task
execution, at least at the token/turn level. No theory in the plan supports "structurally
necessary" — it is an ablation prediction dressed as an argument. If the single-model
CaRT+cost+GRPO baseline matches CASSI, contribution #2 collapses; the plan itself rates this risk
"Medium/High" (§15). State it as a hypothesis, never as a "proof."

**C3 — O(T) oracle vs O(K×T²): LOW-MEDIUM.**
Post-hoc/hindsight labeling from completed trajectories is an established pattern in this subfield:
HAPO's per-question historical minimum length, OTC's minimal-tool-calls-among-correct-rollouts,
Ares's mined minimum-effort labels, SeqRoute's hindsight relabeling (cited in plan). And the
comparison target is stale: AgentPRM-Fudan (Nov 2025) already trains *agent* PRMs with TD+GAE at
"8× more compute-efficient" — no O(K×T²) MC needed. The specific per-step STOP/CONTINUE labels from
`argmax[quality−λ·cumcost]` plus the three formal properties are a real but thin refinement
(Properties 1–2 are elementary comparative statics). This claim survives only if framed as
"cost-aware stopping labels for free," not "we beat MC-PRM complexity."

**C4 — Per-instance dynamic adaptation beats static; H5 difficulty correlation: LOW (token level), MEDIUM (agent level).**
The phenomenon "trained model spends more on hard, less on easy, with explicit
difficulty-allocation analysis" is *already published many times over*: ALP (5.35× hard/easy token
ratio, Pareto analysis), AdaCtrl ("accurately estimates problem difficulty and allocates budgets in
alignment"), DAST, AdaptThink (mode ratio vs difficulty), GFPO-AD, CRM's d_q, and — fatally for the
framing — Arora & Zanette's own difficulty appendix. H5 (r>0.5) is a replication of a known effect
in a new setting, not "the defining characteristic" of anything novel. As an agent-domain result
(stopping *step* vs task difficulty in tool loops) it retains some value — as a sanity check.

**C5 — Small stopper supervises large executor, <3% overhead: LOW-MEDIUM.**
Ares does precisely "tiny model supervises big agent's per-step compute with negligible overhead,
52.7% savings" in March 2026, on harder agent benchmarks (TAU-Bench, WebArena, BrowseComp-Plus)
than most of CASSI's. TAB does it for multi-turn budgets. The delta is only the *kind* of signal
(continuous stopping value vs discrete effort/budget) and the process-reward reuse (C1).

**C6 — Baseline currency: FAIL as drafted.**
Missing from §11.4 and §6.5 entirely: OTC-PO (the direct agent-cost-RL competitor, on
HotpotQA-style QA — its absence would be flagged by any informed reviewer), ALP/DAST/AdaptThink/
AdaCtrl/HAPO/LASER-D (the published adaptive-penalty family — the plan substitutes a homemade
"Adaptive-α Reason Efficiently" P0 baseline, which reviewers will read as a weakened stand-in for
methods the authors didn't cite), TAB, AgentPRM-Fudan, SIM-RAG, MRT, ConciseRL/CRM. Ares is cited
but parked at P2; given it's the closest architecture, it belongs at P0/P1. Note the PDFs for most
of these already sit in `research/papers/` — the plan text simply hasn't absorbed them.

---

## Overall assessment

**Score: 5/10. Recommendation: PROCEED WITH CAUTION.**

**Key differentiator (the one defensible gap):** a learned, cost-aware stopping-value model whose
Δ(s_t) is used as a *step-level process reward to train* a tool-using executor — the ④→⑤ "bridge"
plus enforcement at inference. My searches found no paper occupying that exact cell, and the
plan's own ablation (bridge vs controller-only) is the right experiment to prove the cell matters.

**Biggest risk:** the motivation and two of five contributions are built on a 2025-vintage strawman
("static, instance-blind, no mid-trajectory adaptation") that an efficient-reasoning reviewer will
recognize as false within the first two pages — ALP/DAST/AdaCtrl/SmartThinker/TAB/Ares/OTC
collectively falsify it. Second-biggest: the 2026 agent-stopping space is crowding fast (Ares Mar,
TAB Apr, RedundancyBench May, Semantic Early-Stopping Jun, Agentic Abstention Jun, AgentAbstain
Jul) — six months of review latency could produce a direct collision on the remaining gap.

---

## Weaknesses & likely rejection reasons (ranked)

1. **Strawman framing.** "Penalty is instance-blind — same pressure on easy and hard problems"
   (§4.1, §6.8) is contradicted by ≥8 published adaptive-penalty methods (DAST, ALP, LASER-D,
   AdaCtrl, HAPO, AdaptThink, Thinkless, GFPO-AD, Leash) and by the difficulty analysis *inside*
   the very paper CASSI names as its foil. Reviewer 2 in my subfield kills the paper on this alone.
2. **Missing direct competitor: OTC-PO.** An RL method that trains tool agents for cost efficiency
   on overlapping benchmarks, with hindsight per-question optimal cost — neither cited nor
   baselined. This is the single most conspicuous omission.
3. **Contribution #4 is not a contribution.** Difficulty-correlated compute allocation as a trained
   behavior, with correlation/allocation analysis, is established (ALP, AdaCtrl, DAST, GFPO-AD).
   Claiming it as novel invites "已有工作" citations in every review.
4. **Contribution #2 overclaims.** "Structurally necessary" / "fundamentally fail" with no theory,
   against a subfield full of single-model successes at economic self-assessment. If ablation (B)
   multi-task ≈ (C) two-model, the paper's architecture story collapses.
5. **O(K×T²) target is stale.** TD/GAE-labeled agent PRMs (AgentPRM-Fudan, Nov 2025) already claim
   8× compute efficiency; "we avoid MC rollouts" is a 2025 problem statement in a 2026 submission.
6. **Cheap training-free stoppers may match the savings.** Semantic Early-Stopping reports 38%
   token reduction at parity quality on HotpotQA with an embedding-drift heuristic; DEER-style
   confidence exits are free. CASSI's 20–40% target savings must beat these to justify two training
   phases; the plan's zero-training self-eval baseline is weaker than these published heuristics.
7. **Monitor feasibility risk.** RedundancyBench shows SOTA methods detect redundant agent steps at
   24.88% — the plan's assumption that a 0.5B–3B model reliably predicts t* from noisy mid-answer
   quality is optimistic; intermediate F1/quality-at-step-t is also ill-defined for SWE-bench.
8. **Scope non-credibility.** 7 benchmarks × ~17 baselines × 9 ablations × 3 seeds in an 8-week
   plan reads as a wish list; reviewers will suspect the delivered subset was cherry-picked.

---

## Concrete improvement suggestions

1. **Rewrite the positioning sentence.** Replace "static, instance-blind" with the accurate 2026
   gap: "Existing adaptive methods set cost pressure *before* generation (solve-rate penalties,
   difficulty tags, budget pre-commitment) or shape trajectory-level rewards (tool-call counts);
   none learns a *state-dependent economic stopping value* from observed mid-trajectory progress
   and reuses it as a process reward to train the agent." That sentence is true; the current one
   is not.
2. **Fix the baseline set.** Promote: OTC-PO (P0, agent-level cost RL — the real primary
   comparison alongside CaRT+cost+GRPO), one *published* adaptive penalty (ALP or DAST) instead of
   the homemade adaptive-α stand-in, Ares (P0/P1, closest architecture), a TAB-style learned budget
   allocator, and AgentPRM-Fudan-TD+cost (replacing or complementing MC-based AgentPRM-cost).
   Add a training-free semantic/confidence stopper (embedding-drift, DEER-style) as the mandatory
   cheap-stopping control.
3. **Demote H5** from "P0 load-bearing contribution" to a consistency check, citing ALP/AdaCtrl as
   the token-level precedents; the new content is only that the effect transfers to tool-step
   allocation in agents.
4. **Reframe contribution #2** as an empirical question ("when does decoupling help in long-horizon
   agents?") and pre-register the possibility that multi-task single-model wins at small horizons —
   the honest version is more publishable than the "necessity proof" version, which is one ablation
   away from self-refutation.
5. **Reposition the O(T) claim** around *cost-aware stopping labels for free*, and benchmark label
   quality against TD/GAE labels (AgentPRM-Fudan), not only against MC rollouts.
6. **Cut to 3 benchmarks** (one multi-hop QA, one web, SWE-bench Verified) with the full baseline
   grid, rather than 7 benchmarks with a sparse grid.
7. **Use RedundancyBench** as an external validation set for the stopping model's step-redundancy
   detection — turning a threat into a differentiating evaluation.

---

## Is this strong enough for ICLR? (honest verdict)

**Not as written.** An efficient-reasoning reviewer will (i) falsify the "static, instance-blind"
premise from memory, (ii) ask where OTC-PO, ALP/DAST/AdaptThink, Ares, TAB, and TD-based agent PRMs
are, and (iii) observe that two of five claimed contributions (per-instance adaptation; small
supervisor with low overhead) restate published results. The proposal's genuine asset — the
stopper-as-cost-aware-process-reward bridge, with the bridge-necessity ablation — is real,
currently unoccupied, and sits in a hot problem space (agent cost is the 2026 topic). If the
framing is rewritten against the actual 2025–2026 literature, the baseline set is modernized, and
the headline result shows CASSI beating OTC-PO, Ares, and a training-free semantic stopper on
cost-at-iso-accuracy in agents, this becomes a credible ICLR submission (accept probability maybe
30–40%, execution permitting). On the current trajectory — strawman framing, 2025 baselines,
confirmatory H5 as a headline — it is a likely reject (3s and 5s) regardless of experimental
effort. The window is also closing: at the observed rate of 2026 agent-stopping publications, the
remaining gap may not survive two more review cycles. Move fast, claim narrow.
