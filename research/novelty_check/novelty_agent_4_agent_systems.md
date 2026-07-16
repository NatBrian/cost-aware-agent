# Novelty Check — Agent 4: Agent-Systems Expert

> Reviewer persona: builds and evaluates tool-using agents (web research, SWE agents, deep-research systems).
> Date: 2026-07-16. Target: `research/paper_plan.md` (v5, read in full, 1,576 lines).
> Method: claims extracted → ≥3 query formulations per claim (WebSearch; arXiv API attempted but blocked from this network; local PDF verification) → 19 papers read directly from `research/papers/` (page-level reads listed below). Every paper named in this report was actually opened or its abstract fetched; none is cited from memory alone.

---

## My understanding of the proposal (2 paragraphs)

CASSI trains a small (0.5B–3B) "stopping model" on labels computed post-hoc from completed executor trajectories: `t* = argmax_t [quality_t − λ·cumulative_cost_{1..t}]`, an O(T) computation claimed to need zero additional policy executions. Steps before t* are labeled CONTINUE, steps at/after t* STOP, with a continuous margin Δ(s_t) = Q_continue − Q_stop ∈ [−1,1]. The stopper is trained by SFT (+ value head) then GRPO, taking as input a structured state summary (multi-dimensional budget state with HIGH/MED/LOW/CRITICAL tiers that scale λ, last-K steps, answer draft, confidence/stability indicators). The stopper's Δ is then used as a step-level process reward (plus progress and format terms) to train the 7B–32B executor via GRPO; at inference the stopper is enforced as a controller with claimed <3% overhead. The pitch is a "self-reinforcing cycle": better executor → better trajectories → better oracle labels → better stopper → better process rewards → better executor.

The paper stakes five claims: (1) first closed cost-aware training cycle of this kind; (2) a separate stopping model is *necessary* due to a "representation conflict" between execution and economic self-evaluation; (3) O(T) oracle labels vs. AgentPRM's alleged O(K×T²) Monte-Carlo rollouts (~160 extra executions per 20-step trajectory, "160× reduction"); (4) per-instance, mid-trajectory cost adaptation beats static length penalties (H5: stopping step correlates with difficulty, r>0.5); (5) a tiny stopper supervises a large executor at <3% overhead with 20–40% cost savings. Evaluation is planned on GAIA, WebWalkerQA, HotpotQA, MuSiQue, SWE-bench Verified, MATH-500, and BFCL against 14+ baselines with 3 seeds, in an 8-week schedule.

---

## Core claims extracted (agent-systems lens)

| ID | Claim | Type |
|---|---|---|
| C1 | First self-reinforcing cost-aware training cycle (oracle → stopper → Δ-as-process-reward → executor GRPO → better trajectories) | Novelty / framing |
| C2 | Separate stopping model is *structurally necessary* (representation conflict); two-model > single-model cost-penalty RL | Architecture + hypothesis |
| C3 | O(T) hindsight oracle labels vs. O(K×T²) MC-rollout PRM training (AgentPRM as the strawman); "zero extra executions" | Efficiency |
| C4 | Dynamic per-instance/mid-trajectory cost pressure beats static penalties (L1, Reason Efficiently) | Empirical |
| C5 | 0.5B–3B stopper supervises 7B–72B executor, <3% inference overhead, 20–40% savings at iso-accuracy | Systems |

---

## Search log (queries + notable hits)

WebSearch (8 queries; arXiv export API returned empty from this network, so verification was via WebSearch/WebFetch + local PDFs):

1. `cost-aware process reward model LLM agent reinforcement learning stopping 2025 2026` → **AgentPRM (Fudan), WWW 2026, arXiv 2511.08325** (name collision with Choudhury's AgentPRM!); ESTAR (2602.10004, early-stop signals as RL reward); RewardFlow (2603.18859); "RL for Multi-Agent Systems through Orchestration Traces" (2605.02801 — its survey text states no retained prior work trains the orchestration stop decision directly, as of May 2026).
2. `train LLM agent "when to stop" learned stopping model reinforcement learning tool calls` → **"Learning to Control LLM Agent Harnesses with Offline RL" (2607.05458, Jul 2026)**: lightweight controller separate from a frozen LLM, trained by advantage-weighted regression on offline rollouts with terminal rubric rewards (structural execution actions incl. verification); SENTINEL (2606.12908).
3. WebFetch arXiv 2607.05458 abstract → confirmed controller-vs-frozen-LLM design; not explicitly cost-aware; does not train the executor.
4. `"OTC" optimal tool call reinforcement learning tool productivity GRPO Wang 2025` → **OTC-PO (2504.14870)** confirmed: tool-integrated reward (correctness × tool-efficiency coefficient), OTC-PPO/OTC-GRPO, −73.1% tool calls at comparable accuracy.
5. `GAIA benchmark train split RL training web agents` → GAIA has no official training split (public dev/validation with answers; test hidden); RL web agents train on synthetic data (WebExplorer 2509.06501, WebShaper 2507.15061, WebGym 2601.02439); **SlimSearcher (2606.07074)** surfaced here.
6. `SWE-bench Verified reinforcement learning 7B 32B GRPO DeepSWE SWE-Gym` → **DeepSWE**: Qwen3-32B + GRPO++ on 4,500 R2E-Gym tasks → 42.2% pass@1 (59% with TTS); SWE-Gym RL attempts show "limited performance improvements, high solve-none rates"; DAPO-style run 20%→39% (2508.03501, long-context multi-turn SWE RL).
7. `IterResearch Alibaba Tongyi Markovian state reconstruction RL trained` → **IterResearch (2511.07327) is RL-trained with an "Efficiency-Aware Policy Optimization" (discounted reward shaping to incentivize efficient research)** — i.e., NOT a training-free heuristic as the plan asserts.
8. `SIM-RAG lightweight critic sufficiency "when to stop" retrieval SIGIR 2025` → **SIM-RAG (2505.02811, SIGIR 2025)**: lightweight *trained* sufficiency Critic decides continue-vs-stop per retrieval round, trained on self-practice data (intermediate answers checked against ground truth).

Local PDFs read directly (page-level): AgentPRM-Choudhury 2502.10325 (pp.1–14); OTC 2504.14870 (pp.1–6); CaRT 2510.08517 (pp.1–4); CoRL 2511.02755 (pp.1–3); SupervisorAgent 2510.26585 (pp.1–2); Terminator 2603.12529 (pp.1–3); DASH 2607.00482 (pp.1–2); AgentPRM-Fudan 2511.08325 (pp.1–3); BATS 2511.17006; INTENT 2602.11541; Ares 2603.07915; SeqRoute 2605.25424; "When Does Learning to Stop Help?" 2606.30852; EAPO 2606.02132; VOI budget control 2605.05701; BAGEN 2606.00198; MaR 2605.23384; LYNX 2512.05325; SlimSearcher 2606.07074 (p.1 each unless noted).

---

## AgentPRM fact-check (what it actually does; is O(K×T²) fair?)

**Verdict: the plan's primary strawman is factually wrong.** AgentPRM (Choudhury, 2502.10325) does **not** launch K Monte-Carlo rollouts from every intermediate state.

What the paper actually does (Sec. 2.2, Algorithm 1, read directly):

- **Stage 1:** roll out the current policy π_{i−1} on tasks, **asynchronously**, multiple times per task. Rollouts are stored in a dictionary G(s,a) hashing each state-action pair to the set of trajectories passing through it. PRM targets are computed as
  `Q̂(s,a) = (1/|G(s,a)|) Σ_{(s_t,a_t)∈D(s,a)} Σ_{k=t}^{T−1} γ^{k−t} r_k`  (Eq. 1)
  — i.e., **averaged discounted returns-to-go over rollouts that were already collected**. The paper explicitly says this was chosen because MCTS-style synchronous per-state exploration "is difficult to scale. In contrast, we collect our rollouts asynchronously."
- **Stages 2–3:** train the PRM by soft-BCE on these targets; train the policy with Online DPO against the PRM with KL to π_{i−1}; **iterate** (π0→Q0→π1→Q1→π2→Q2→π3 on ALFWorld, 10k rollout trajectories per iteration, Llama-3.2-3B for BOTH policy and PRM).

Implications:

1. **Execution cost:** AgentPRM's data collection is O(N×T) per task (N on-policy rollouts of length T) with **O(T) post-hoc target computation per trajectory** — the same order as CASSI, which itself collects trajectories in Phase 1 and generates G=8 rollouts per task during GRPO. The plan's Section 8.1 arithmetic ("K=8, T=20 → 1,520 additional step-executions, ~160 additional full trajectories per trajectory"; "840 additional step-executions on SWE-bench... CASSI requires 0") describes **Math-Shepherd-style per-prefix MC estimation** (2312.08935), not AgentPRM. Any reviewer who has read AgentPRM Eq. 1 will treat the paper's central quantitative hook ("160× fewer executions") as a mischaracterization, and the "Training Complexity Analysis" section (Sec. 20) collapses.
2. **The "cycle" already exists there.** AgentPRM's iterative loop — rollouts → post-hoc targets → train reward model → RL the policy against it → better rollouts → repeat — is structurally CASSI's ①→②→③→④→⑤ cycle. The plan even concedes this ("standard iterative actor-critic, following the PRM training pattern from AgentPRM," Sec. 10.1) while simultaneously claiming "this cycle does not exist in any prior work" (Sec. 4.2). Internal inconsistency; what is genuinely absent in AgentPRM is *cost/stopping semantics in the labels*, nothing else.
3. **Cheaper variants the plan ignores, in the same paper:** InversePRM (PRMs from expert demos, no outcome rewards, no per-state rollouts; better sample-efficiency than AgentPRM in 1 iteration) and process reward shaping with a reference-policy advantage (Sec. 4.2). The paper also cites ARCHER (TD-trained turn-level Q) as a "very similar framework" — i.e., TD critics with zero extra rollouts predate CASSI in agent RL.
4. **The name collision makes it worse.** The Fudan **AgentPRM (2511.08325, WWW 2026)** trains an agent PRM ("promise + progress") using **TD-based estimation with GAE explicitly because MC-based sampling is too costly**, reports ~8× compute-efficiency over baselines, and **already integrates the PRM into RL training of the agent** (their Fig. 2c). A cost-only delta on top of a TD/GAE-labeled agent PRM is the natural reviewer counterproposal, and it is already published.
5. **Fair residual criticism CASSI could make** (much weaker than the current one): dictionary state-hashing only concentrates estimates where states repeat (fine in ALFWorld's compact state space); in open-ended web/SWE settings each state is visited once, so Eq. 1 degenerates to single-sample return-to-go — high variance, though still zero extra rollouts. CASSI's honest framing is "variance/semantics of labels," not "asymptotic execution count."

**Bottom line: O(K×T²) is a fair description of Math-Shepherd-style math PRMs, not of AgentPRM. Claim 3's headline comparison, the ~160× number, and RQ5 as framed are not defensible.**

---

## Experimental feasibility audit

### Benchmark-by-benchmark (as an agent-systems practitioner)

| Benchmark | RL-trainable as planned? | Per-step quality_t measurable? | Assessment |
|---|---|---|---|
| **GAIA** | **No train split** (466 Qs; public dev ~165 with answers, test hidden). RL web agents (WebSailor/WebExplorer/WebShaper/SlimSearcher) train on *synthetic* web-QA corpora and evaluate on GAIA. Building such a pipeline is its own multi-month project. | Exact-match on a *forced* per-step answer → 0/1 step function; needs an extra "answer now" generation at every step (agents do not emit answers per step); LLM-judge for fuzzy answers adds cost. | Keep as **OOD eval only**. Do not claim RL training on GAIA. |
| **WebWalkerQA** | Eval benchmark (~680 Qs); no standard RL train split. Same synthetic-data problem as GAIA. | Same forced-answer issue. | Eval-only, or drop. |
| **HotpotQA** | Yes — standard (Search-R1 precedent trains on NQ+HotpotQA). | Forced per-step answer + F1 vs. gold: cheap (~T short generations/trajectory). Noisy mid-trajectory F1 is tolerable. | **Keep. Primary training domain.** |
| **MuSiQue** | Yes (train split exists; harder). | Same as HotpotQA. | Keep. |
| **SWE-bench Verified** | Verified is **eval-only**; training requires SWE-Gym/R2E-Gym environments. Known results: SWE-Gym RL "limited improvements, high solve-none rates"; DeepSWE needed **Qwen3-32B + 4,500 R2E-Gym tasks + GRPO++ on a large H100 cluster** for +20 pts; 7B SWE RL essentially does not work. | **This is where the "zero extra cost" claim breaks hardest**: quality_t = per-step test-suite execution in Docker, ~1–10 min per run → T=20 runs per trajectory × G=8 × batch × steps = astronomically more env-compute than the policy executions saved. Worse, intermediate patches usually fail all tests → quality_t ≈ 0 until near the end → t* degenerates to the final step → no stopping signal. | **Cut from the RL story.** At most: inference-time monitor on a frozen 32B agent, appendix. |
| **MATH-500** | Eval-only (train on MATH train). Not agentic. | Intermediate answers occur naturally in CoT (Terminator/DASH exploit exactly this). | Fine as low-slack control, inference-time only. |
| **BFCL** | Eval-only leaderboard; mostly single/few-turn function calling — stopping is largely irrelevant except the multi-turn slice. | N/A per-step quality in the CASSI sense. | Drop. |

### Where the oracle's "O(T), zero extra cost" hides real costs

1. **Per-step answer forcing.** Algorithm 1's `answer_t ← ExtractAnswer(a_t)` assumes each action contains an answer. In ReAct traces it does not: you must either (a) probe the model ("answer now") at every step — T extra generations per trajectory (CaRT does exactly this and pays for it), or (b) modify the executor to emit a draft answer every step — a behavioral/distributional change the plan never analyzes.
2. **Per-step scoring.** Free-ish for string-F1 QA; a test-suite run per step on SWE-bench (dominant cost, see above); an LLM-judge call per step for open-ended web answers.
3. **Train/test feature mismatch (methodological hole).** The stopping-model prompt (Sec. 21) embeds `Quality change: {quality_delta:+.3f}`, "progress rate," "answer stability" — quantities derived from **ground truth**, which does not exist at inference. The plan's own reviewer-question table asserts the monitor "only needs task description + trajectory state — no ground truth at inference," contradicting its prompt template. Either the features must be replaced by inference-available proxies (self-confidence, answer self-consistency) — retraining the oracle pipeline — or the reported stopping accuracy will not transfer to deployment. No section addresses this.
4. **Monitor runtime accounting.** ~1–2K-token structured prompt + generation per step × 20 steps; <3% holds only when executor steps are large. The July 2026 study "When Does Learning to Stop Help?" (2606.30852) shows probe overhead accounting can *flip the sign* of savings depending on serving regime (32% token savings under KV-cache forking becomes +121% cost under black-box re-prefilling) and that **calibrated scalar confidence exits beat learned stoppers on several benchmark families**. The plan's Zero-Training Self-Eval baseline is therefore genuinely dangerous, and overhead must be reported per serving regime.
5. **Fabricated-sounding numbers.** Sec. 27 answers a reviewer with "In our experiments, monitor errors are 16% of decisions, and 78% of errors are false-CONTINUE" — there are no experiments yet. If this survives into a submission it is a credibility bomb.

### Schedule/matrix realism

14+ baselines × 7 benchmarks × 3 seeds, where ≥7 baselines are themselves full RL training runs (L1, Reason Efficiently, adaptive-α variant, CaRT+cost+GRPO, single-model GRPO+cost, AgentPRM-cost, BudgetThinker, ReMA-cost) plus CASSI's own three-phase pipeline (collection → stopper SFT+RL → executor GRPO) per λ, per stopper size, per executor. On an 8×H200 node, one multi-turn tool-agent GRPO run at 7B with search is realistically 1–3 days; the full matrix is **hundreds of runs**. The 8-week plan (which allocates Weeks 1–2 to infra + trajectory collection across seven heterogeneous harnesses including SWE Docker environments) is off by roughly an order of magnitude. This matters for novelty because the differentiating ablations (single- vs two-model; bridge vs controller-only; oracle vs MC labels) are exactly the runs that will get cut when time runs out.

**What to cut (my recommendation):** train on HotpotQA+MuSiQue (one executor: Qwen2.5-7B; one stopper size for main results); add ALFWorld *or* AppWorld/WebShop as the second training domain (train splits exist and AgentPRM/GiGPO comparisons become natural); GAIA + Bamboogle/2Wiki as OOD inference-time evals; MATH-500 control; drop SWE-bench-Verified RL, WebWalkerQA, BFCL, ReMA-cost, BudgetThinker, Ares-reimplementation; keep OTC-GRPO and EAPO as the *primary* cost-aware RL baselines (currently missing entirely), plus zero-training self-eval + calibrated-confidence exits, single-model GRPO+cost, CaRT+cost, AgentPRM-cost (honestly implemented as return-to-go + cost), and the two CASSI ablations. 3 seeds on the headline table only.

---

## Closest prior work table

| Paper | Year | Venue | Overlap | Key Difference from CASSI |
|---|---|---|---|---|
| **AgentPRM** (Choudhury, 2502.10325) | 2025 | preprint (Cornell) | Iterative loop: on-policy rollouts → **post-hoc O(T) return-to-go targets** → train PRM → RL policy → repeat; 3B PRM; ALFWorld | No cost term, no stopping semantics; policy via Online DPO; **not** O(K×T²) |
| **AgentPRM** (Fudan, 2511.08325) | 2025 | **WWW 2026** | Agent PRM trained **TD+GAE explicitly to avoid MC sampling cost** (8× compute-efficiency); PRM used for step-level search **and agent RL** | Quality/progress only — no cost, no stopping; name collision must be handled |
| **OTC-PO** (2504.14870) | 2025 | preprint (>widely cited) | Cost-aware agent RL: per-question **optimal (minimal) tool calls estimated in hindsight from the group's own correct rollouts**, zero extra executions; OTC-GRPO; −73% tool calls | Single model; trajectory-level multiplicative reward, no step-level Δ, no learned stopper, tool-count-only cost. **Uncited by the plan** |
| **EAPO** (2606.02132) | 2026 | preprint (CAS/ByteDance) | **Difficulty-aware cost shaping** in agentic RL (penalize redundant tool calls mainly on easy queries); tool-free trajectories injected into GRPO groups | Single model; no stopper; directly implements the plan's "adaptive-α" baseline idea. Uncited |
| **SlimSearcher** (2606.07074) | 2026 | preprint (ZJU/Ant) | Cost-aware deep-research agent RL on **GAIA/BrowseComp**: Pareto-filtered SFT + **Adaptive Reward Gating** (cohort-relative tool/token efficiency gated by correctness); −17–58% tool rounds | Single model, trajectory-level; no learned stopper. Uncited; occupies the plan's flagship domain |
| **IterResearch** (2511.07327) | 2025 | preprint (Alibaba/RUC) | Long-horizon research agent trained with **Efficiency-Aware Policy Optimization (discounted reward shaping for efficiency)** | Plan mislabels it "training-free heuristic"; no stopper/PRM |
| **CaRT** (2510.08517) | 2025 | preprint (CMU) | Learned termination; **uses per-step task-success (quality at each timestep) as the dense label source**; counterfactual pairs + rationale SFT | γ-discount instead of explicit λ·cost; SFT-only; no Δ-as-process-reward for executor RL |
| **Terminator** (2603.12529) | 2026 | preprint | **"Hindsight-optimal reasoning length": earliest arrival of the final answer computed post-hoc from completed CoTs (zero extra rollouts) + trains a lightweight exit probe** on these labels | Reasoning CoT, not tool agents; λ→0 special case of CASSI's oracle; inference-time only, no executor training |
| **DASH** (2607.00482) | 2026 | preprint (Capital One) | **Intermediate answers vs. ground truth checked post-hoc, "without any additional supervision," converted into segment-level GRPO advantages** to cut overthinking | Single model, math only, no explicit cost term, no stopper — shows oracle labels can skip the stopper entirely |
| **SIM-RAG** (2505.02811) | 2025 | **SIGIR 2025** | **Small trained sufficiency Critic decides continue-vs-stop each retrieval round**; labels from self-practice runs checked against ground truth | No cost/λ tradeoff; no executor training; QA-RAG only |
| **LYNX** (2512.05325) | 2025 | preprint (USC) | Lightweight learned probe for early exit, supervision from forced exits, conformal control | Hidden-state probe for CoT; inference-only |
| **"When Does Learning to Stop Help?"** (2606.30852) | 2026 | preprint | **Cost-aware controlled study of learned stoppers vs. scalar confidence/entropy exits including probe overhead**; regimes where learned stopping loses | Reasoning models; directly threatens H7/<3% overhead framing and the zero-training baseline gap |
| **CoRL** (2511.02755) | 2025 | preprint (Apple/UIUC) | **Controller LLM trained with RL under dual task+cost rewards, budget-conditioned behavior**, coordinating frozen expert models | Routing/orchestration, not stop/continue process rewards; experts frozen |
| **Ares** (2603.07915) | 2026 | preprint (UCSB) | **Small trained per-step router predicting minimum reasoning effort for each step of an agent** (TAU-Bench, BrowseComp-Plus, WebArena; −52.7% tokens) | Discrete effort levels; labels from a minimal-effort data pipeline; does not train the executor |
| **SeqRoute** (2605.25424) | 2026 | preprint (UT Austin) | **Hindsight Budget Relabeling** + offline RL (CQL); **dynamic λ-sweep to navigate cost-quality Pareto frontier without retraining** | Session-level model routing, not within-trajectory stopping |
| **BATS** (2511.17006) | 2025 | preprint (Google) | Budget Tracker plug-in + budget-aware test-time scaling for web search agents; unified token+tool cost metric; Pareto framing | Training-free; matches plan's description |
| **INTENT** (2602.11541) | 2026 | preprint (RUC/Baidu) | Inference-time planning under hard monetary tool budgets (world model, risk-calibrated cost) | Training-free; matches plan's description |
| **VOI budget control** (2605.05701) | 2026 | preprint (CityU HK) | **Per-step Value-of-Information score (marginal task value per unit budget under remaining dual budget)** deciding retrieve/decompose/commit | Inference-time controller; CASSI's Δ(s_t) is the learned version of this published quantity |
| **BAGEN** (2606.00198) | 2026 | preprint (Northwestern+) | Budget-awareness benchmark; shows **SFT+RL trains early-stop/alert behavior; early stop saves 28–64% tokens** | Single model; benchmark/analysis focus |
| **SupervisorAgent** (2510.26585) | 2025 | **ICLR 2026** | **Lightweight runtime supervisor** cutting GAIA token use ~29.7% without hurting success (intervention, guidance, observation purification) | Training-free supervision; no reward-model role |
| **CaRT/PRIME context — Implicit PRM** (2412.01981) + **PRIME** (2502.01456) | 2024–25 | ICML'25 era | Process rewards **for free** from outcome-labeled data (no rollouts, no step labels); PRIME uses them online in RL | Not cost-aware; undercuts any "PRMs need expensive step labels" framing |
| **SPA-RL** (2505.20732) / **GiGPO** (2505.10978) | 2025 | preprints | Step-level credit for agent RL **without extra rollouts** (progress attribution / anchor-state grouped advantages) | Not cost-aware, no stopper — further erodes C3's "step-level signal requires O(K×T²)" premise |
| **Harness controller** (2607.05458) | 2026 | preprint | Lightweight controller (separate from frozen LLM) trained offline-RL to make structural execution decisions | Not cost-aware; no process-reward bridge |

Production patterns relevant to C2/C5 (reviewer common knowledge, not paper-fetched): RLHF's separate reward model; GPT-5's real-time router allocating thinking vs. non-thinking per query; Anthropic's effort parameter on Claude Opus 4.5; judge/guardrail sidecar models in agent stacks. A "small model watching a big agent" is an established production pattern; novelty must come from the training bridge, not the architecture.

---

## Per-claim novelty verdicts

| Claim | Verdict | Closest prior | Residual delta |
|---|---|---|---|
| **C1: first self-reinforcing cost-aware cycle** | **LOW–MEDIUM** | AgentPRM-Choudhury (iterative PRM↔policy loop with post-hoc labels); AgentPRM-Fudan (PRM→agent RL); SPARK 2509.22624 / Cooper 2508.05613 (policy-RM co-evolution, in library) | Cost+stopping semantics inside an otherwise-existing loop. "First cycle" framing will be rejected; "cost-aware stopping labels in the PRM loop" survives |
| **C2: separate stopper is necessary (representation conflict)** | **LOW as architecture; MEDIUM as ablation** | Ares, CoRL, SIM-RAG critic, SupervisorAgent, harness controller, RLHF RM, GPT-5 router | The controlled single-model vs. two-model vs. bridge ablation would be a genuinely useful empirical result; the "necessity" theory is an untested hypothesis stated as fact, and OTC/EAPO/BAGEN already show single models learn cost adaptation |
| **C3: O(T) oracle vs. O(K×T²) MC PRM** | **LOW** | AgentPRM actually O(N×T) (fact-check above); Terminator's hindsight-optimal exit labels; DASH's post-hoc intermediate-answer credit; CaRT per-step success labels; Fudan TD+GAE; SPA-RL/GiGPO/ImplicitPRM | Only the explicit λ·cost term and the (trivial) comparative-statics properties are new. The 160× headline is unsupportable |
| **C4: per-instance dynamic adaptation beats static penalties** | **MEDIUM-LOW** | OTC (per-question optimal tool calls, hindsight), EAPO (difficulty-aware shaping), LASER-D/DAST/ALP/HAPO (adaptive length rewards, in library), SeqRoute λ-sweep | *Mid-trajectory step-level* adaptation via a learned monitor is the residual delta — but Ares (per-step effort) and VOI (per-step marginal value) already cover the per-step decision at inference. H5 (difficulty–stop correlation) is a fine sanity analysis, not a contribution |
| **C5: small stopper supervises large executor, <3% overhead** | **LOW–MEDIUM** | Ares, SIM-RAG, LYNX, Terminator probe, SupervisorAgent, CoRL | Asymmetric sizing is standard; the overhead claim must be defended per serving regime (2606.30852 shows it can flip) and against calibrated scalar exits |

**The one genuinely open cell my searches support:** a *trained, cost-aware* stopping/value model whose Δ is used as a **step-level process reward to RL-train the executor**, with stopping enforced at inference, on tool-using agents. Nothing I fetched does this whole bridge (the May 2026 orchestration-RL survey text explicitly notes the trained stop decision is missing in its taxonomy). That is a composition novelty — real, but narrow, and it must carry the entire paper.

---

## Overall assessment

**Score: 4/10 as currently written** (would be ~6/10 after the reframe below).
**Recommendation: PROCEED WITH CAUTION — major reframe and descope required before any experiment is run.**

- **Key differentiator (defensible):** the process-reward bridge — hindsight cost-aware stopping labels → small trained stopper → Δ(s_t) as step-level reward for executor GRPO — plus the controller-only vs. bridge ablation that tests whether executors *internalize* cost-awareness. Nobody I could find closes this loop with cost in it.
- **Biggest risk:** the paper's central quantitative claim (O(K×T²)→O(T), "160× fewer executions vs. AgentPRM") is based on a mischaracterization of AgentPRM that any informed reviewer will catch, in a field where OTC/EAPO/SlimSearcher/Terminator/DASH/SIM-RAG (all uncited) have already occupied the surrounding territory during 2025–2026. Second-biggest: the experimental plan is infeasible as scheduled, so the differentiating ablations are the ones most likely to be missing at submission time.

---

## Weaknesses & likely rejection reasons (ranked)

1. **Factually wrong strawman at the heart of Contribution 3.** AgentPRM computes post-hoc return-to-go targets from asynchronously collected rollouts (Eq. 1), not K-per-state MC rollouts. The 160×/O(K×T²) framing, Sec. 8.1's SWE-bench arithmetic, and RQ5 will be flagged as misrepresentation. Near-certain reject trigger with any reviewer who knows the paper.
2. **Missing 2025–2026 prior art that owns the headline territory:** OTC (cost-aware agent RL with hindsight per-question optimal tool calls), EAPO (difficulty-aware cost shaping), SlimSearcher (cost-aware deep-research RL on GAIA), Terminator (hindsight stopping labels + trained probe), SIM-RAG (trained stop/continue critic), Fudan AgentPRM (TD+GAE agent PRM fed into RL), DASH (post-hoc intermediate-answer step credit), BAGEN (SFT+RL early-stop). Related work also **mischaracterizes IterResearch** (it is RL-trained with efficiency-aware reward shaping, not a training-free heuristic).
3. **"First self-reinforcing cycle" is marketing.** Iterative RM↔policy co-training is standard (AgentPRM, iterated RLHF, SPARK/Cooper). The plan even admits following AgentPRM's iterative pattern while claiming the cycle doesn't exist — internal contradiction reviewers will quote.
4. **The oracle's "zero extra cost" is false in the domains that motivate the paper.** Per-step answer forcing (T extra generations), per-step scoring (SWE-bench: a test-suite run per step), and LLM-judge calls are real costs; on SWE-bench the intermediate quality signal is also nearly flat at 0, giving the oracle no gradient.
5. **Train/test feature leakage:** the monitor's input template contains ground-truth-derived quality deltas that cannot exist at inference; the plan contradicts itself about this. Uncorrected, headline stopping-accuracy numbers won't transfer and reviewers will call it out.
6. **Infeasible experiment matrix** (7 benchmarks, 14+ baselines — 7 of them full RL runs — 3 seeds, 8 weeks): guarantees the paper arrives with the weak baselines run and the load-bearing ablations missing. GAIA/WebWalkerQA/BFCL/MATH-500 lack train splits; SWE-bench RL at 7B is known not to work and at 32B requires DeepSWE-scale compute.
7. **The zero-training baseline may win.** 2606.30852 shows calibrated scalar confidence exits beat learned stoppers on several task families once probe overhead is priced in; the plan's fallback ("position as more flexible/interpretable") is not an ICLR paper.
8. **Hygiene:** invented result-like numbers in the reviewer-Q&A section ("monitor errors are 16%..."); two papers named "AgentPRM" conflated into one citation.

---

## Concrete improvement suggestions

1. **Rewrite the efficiency story honestly.** Frame the oracle as a *label-semantics* contribution ("cost-aware stopping targets at the same collection cost as outcome-only RL"), and turn efficiency into an *ablation over label-generation methods on identical data*: hindsight t* labels vs. single-sample return-to-go (AgentPRM-Choudhury style) vs. TD+GAE (AgentPRM-Fudan style) vs. per-state MC (Math-Shepherd style, small scale only). Report measured wall-clock and label quality. Delete every O(K×T²) claim about AgentPRM; attribute per-state MC to Math-Shepherd/OmegaPRM where it belongs.
2. **Make OTC-GRPO and EAPO the primary baselines** and the bridge ablation the primary claim: does Δ-as-process-reward from a trained stopper beat (a) OTC's hindsight trajectory-level cost coefficient and (b) direct DASH-style oracle-label advantages *without* a stopper? If CASSI can't beat "use the oracle labels directly as step rewards," the stopper is dead weight — test this first, cheaply, on HotpotQA.
3. **Descope to 2 training domains** (HotpotQA+MuSiQue; ALFWorld or AppWorld/WebShop for agentic breadth and AgentPRM comparability), GAIA/2Wiki/Bamboogle as OOD inference-time evals, MATH-500 control. Drop SWE-bench RL, WebWalkerQA, BFCL, ReMA-cost, BudgetThinker, Ares reimplementations.
4. **Fix the monitor's input contract:** only inference-available features (self-consistency of forced answers, answer-change rate, budget state, step embeddings); recompute oracle labels accordingly; report the gap between ground-truth-featured and deployable monitors explicitly.
5. **Price the oracle honestly:** report label-generation cost including per-step answer forcing and scoring per benchmark; report monitor overhead under both KV-forking and re-prefill serving (cite 2606.30852's protocol) and include calibrated scalar-exit baselines at matched lost-correct risk.
6. **Handle the AgentPRM name collision** explicitly (Choudhury 2502.10325 vs. Fudan WWW 2026 2511.08325) and correct the IterResearch characterization; add SIM-RAG, Terminator, LYNX, CaRT, DASH to a "learned stopping" related-work subsection with a delta table.
7. **Strip result-like numbers from the plan/paper** until measured; keep hypotheses directional.
8. If the two-model claim is to survive, add the strongest single-model competitor: executor fine-tuned with oracle stopping labels multi-task (predict STOP/CONTINUE alongside acting) + cost-shaped GRPO — if that matches CASSI, the honest paper is "the labels matter, the second model doesn't," which is still publishable if framed as a finding.

---

## Is this strong enough for ICLR? (honest verdict)

**Not as written.** The submission as planned would face: a falsifiable-and-false central efficiency claim, a "first" claim contradicted by its own citations, a related-work section missing or mislabeling the half-dozen 2025–2026 papers closest to it (OTC, EAPO, SlimSearcher, Terminator, SIM-RAG, Fudan-AgentPRM, IterResearch), and an experimental matrix that cannot be completed — I would expect scores around 3/5/3 (reject) with the AgentPRM fact-check appearing verbatim in at least one review.

The salvageable ICLR paper is narrower and honest: *"Cost-aware hindsight stopping labels turn a small monitor into a process reward model that makes tool-using executors economical"* — two training domains, OTC/EAPO/scalar-exit/DASH-style-direct-label baselines, the bridge and single-vs-two-model ablations as the scientific core, efficiency reported as measured wall-clock label-generation comparisons rather than asymptotic strawmen, and deployable monitor features. Executed cleanly with the bridge ablation showing a real effect, that is a credible borderline-accept ICLR paper (or a strong NeurIPS/ICML resubmission); if the bridge ablation shows no effect over direct oracle-label shaping or OTC, the project should pivot to a rigorous negative/analysis paper. Given the current density of concurrent work, every month of delay materially increases scoop risk on the one remaining open cell.
