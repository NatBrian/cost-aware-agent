# Harsh Novelty Assessment: CASSI — Reviewer-Style Evaluation

> **Assessed by:** Two independent ICLR/NeurIPS-level subagent reviewers
> **Date:** 2026-07-15
> **Status:** Pre-experiment. Paper plan evaluated; zero empirical results.

---

## Overall Score

| Dimension | Score (1–5) | Verdict |
|---|---|---|
| **Scientific Novelty** | **2.0** | Core ideas (separate process reward model, two-agent RL, cost penalties) are established. The combination is incremental. |
| **Technical Depth** | **2.0** | Theoretical analysis proves trivial statements. Training pipeline chains existing components without algorithmic innovation. |
| **Experimental Credibility** | **1.0** | Zero experimental results. All numbers are hypothesized for illustration. |
| **Significance** | **3.0** | If it works as claimed, practically significant. But gains are hypothesized to be incremental over much simpler methods. |
| **Presentation Clarity** | **4.0** | Well-structured plan, explicit pseudocode, detailed experimental design. |
| **Composite** | **2.4 / 5.0** | **Below ICLR/NeurIPS acceptance threshold in current form.** |

---

## Claim-by-Claim Assessment

### Claim 1: "First work to propose a learned, separate monitor for cost-aware stopping"

**Verdict: OVERCLAIMED — FALSE**

- **AgentPRM (Choudhury, 2025)** already has a separate Process Reward Model trained via supervised learning that provides step-level rewards for policy training via PPO. CASSI's monitor is architecturally identical to AgentPRM's PRM — both are separate models evaluating states and producing training signals. Adding a cost term to the reward (`quality − λ*cost` instead of just `quality`) is a 1-line change in the Monte Carlo return computation, not a new contribution.
- **ReMA (Wan et al., NeurIPS 2025)** already has a separate high-level meta-thinking agent training a low-level reasoning agent via multi-agent RL. The two-agent hierarchical RL architecture is identical.
- **CaRT (Liu et al., 2025)** already trains models to decide when to stop gathering information. CASSI adds a cost term and moves the termination model to a separate process — this is an architectural refactoring, not a new idea.

**Recommended fix:** Drop "first to propose" language. Frame as: "We specialize process reward models for cost-aware stopping, demonstrating that cost-aware rewards outperform quality-only rewards for agent efficiency."

### Claim 2: "The soft stopping curve — a continuous value function Δ(s_t)"

**Verdict: OVERCLAIMED — REBRANDING**

- Δ(s_t) = Q_continue(s_t) − Q_stop(s_t) is mathematically identical to the advantage function A(s, stop) in standard RL, or the Q-value difference. The term "soft stopping curve" is new branding, not a new concept.
- **BAVT (Li et al., 2026)** already produces a continuous, soft budget-conditioned decision function using the remaining budget ratio as an exponent. It's non-learned but conceptually identical — a continuous function mapping budget state to stopping propensity.
- **Budget Guidance (Li et al., 2025)** already models remaining thinking length as a continuous Gamma distribution.

**Recommended fix:** Drop "soft stopping curve" as a named contribution. Frame as: "We train a cost-aware value function Δ(s_t) that estimates the expected net benefit of continuing, and use it to provide dynamic stopping signals."

### Claim 3: "Three-phase training pipeline as a contribution"

**Verdict: OVERCLAIMED — STANDARD METHODOLOGY**

- **AgentPRM (2025)** describes exactly: collect trajectories → train PRM on Monte Carlo targets → train policy via PPO with PRM as reward. The pipeline is textbook iterative actor-critic.
- **CaRT (2025)** describes: generate counterfactual data → SFT → optional GRPO-based RL. Same structure.

**Recommended fix:** Remove from contributions list. Describe as "our training methodology" not as a contribution.

### Claim 4: "Monitor generalizes across executor model sizes and domains"

**Verdict: PARTIALLY NOVEL BUT UNPROVEN AND FRAGILE**

- No existing cost-aware stopping work tests cross-model transfer because no existing work has a separate monitor.
- But the paper plan's own hypothesized numbers show **35% degradation** on domain transfer (GAIA → SWE-bench: 84.3% → 65.2% stopping accuracy). This is not generalization — it's failure.
- The claim is a double-edged sword: if transfer fails, it undermines the entire "separate monitor" argument (why not embed cost-awareness in the executor like L1?).

**Recommended fix:** Narrow to "moderate transfer within related task families." Drop the strong generalization claim. Add a failure analysis section on when transfer breaks.

### Claim 5: "Multi-dimensional budget representation"

**Verdict: NOT NOVEL — ENGINEERING**

- **BATS (2025)** already uses "a unified cost metric jointly accounting for token and tool consumption."
- **INTENT (2026)** already tracks dollar costs explicitly.
- **BAVT (2026)** already operates on multi-dimensional budget concepts.

**Recommended fix:** Remove from contributions list. Describe as a design choice in the architecture section.

### Claim 6: "Monitor as reward model"

**Verdict: NOT NOVEL — DEFINITION OF A PRM**

- Using an evaluation model to generate training signals for a policy IS the definition of a Process Reward Model. AgentPRM, Agent-RRM, and CSO all do this.
- The paper plan itself acknowledges: "Our monitor IS a process reward model specialized for cost-aware stopping" (Section 11.1).

**Recommended fix:** Remove as a separate claim. It's already covered by the cost-aware PRM contribution.

### Claim 7: "Dynamic, per-instance, mid-trajectory adaptation vs. static penalties"

**Verdict: GENUINELY NOVEL — THE STRONGEST CLAIM**

- **L1** and **Reason Efficiently** use static length penalties — same α for all instances. The model cannot say "this problem is harder, I need more tokens."
- **TALE, SelfBudgeter, BudgetThinker** pre-commit to budgets before generation. They don't adapt mid-trajectory.
- **BAVT** adapts dynamically but via a fixed mathematical function, not a learned model.
- CASSI's monitor observes the *actual reasoning trajectory* and adapts in real-time. This is genuinely different.

**BUT:** This claim depends entirely on the experiment showing that CASSI's stopping point *varies with task difficulty* (H5: r > 0.6). If the monitor just learns a fixed threshold, it's equivalent to L1 with extra parameters. **This is the make-or-break experiment.**

---

## Top 5 Closest Competitor Papers — Detailed Comparison

### 1. AgentPRM (Choudhury, 2025) — "Process Reward Models for LLM Agents"

| Aspect | AgentPRM | CASSI |
|---|---|---|
| Architecture | Separate PRM → Q(s,a) → reward for policy PPO | Separate Monitor → Δ(s_t) → reward for executor GRPO |
| Training | Monte Carlo rollouts → supervised PRM training | Oracle labels → SFT → monitor GRPO |
| Reward signal | Quality Q-values | Cost-quality trade-off Δ |
| Benchmarks | ALFWorld (embodied) | GAIA, HotpotQA, SWE-bench, MATH |

**Is CASSI different?** The architecture is nearly identical. Both train a separate model to evaluate states and provide training signals for a policy. The only difference is the reward function: AgentPRM uses quality-only Q-values; CASSI uses cost-penalized Δ values. This is a **reward engineering** difference, not an architectural one.

**Critical missing experiment:** Compare CASSI against "AgentPRM-cost" — AgentPRM with `R = quality − λ*cost` as the Monte Carlo return. If CASSI outperforms AgentPRM-cost, the specialized monitor design (structured Δ output, budget tiers, SFT+RL) matters beyond the cost term. If they perform equivalently, CASSI is AgentPRM with a renamed output.

**Verdict: CASSI is a special case of AgentPRM applied to cost-aware stopping. The paper must demonstrate that the specialized design produces better outcomes than a generic AgentPRM with cost-augmented rewards.**

---

### 2. CaRT (Liu et al., 2025) — "Teaching LLM Agents to Know When They Know Enough"

| Aspect | CaRT | CASSI |
|---|---|---|
| Goal | Learn when to stop gathering information | Learn when to stop, with cost-awareness |
| Who terminates? | Same model (self-termination) | Separate monitor |
| Training | SFT on counterfactual pairs + optional GRPO-RL | SFT on oracle labels + GRPO-RL |
| Decision | Binary (terminate/continue) | Categorical (STOP/CONTINUE/ADJUST) + continuous Δ |
| Cost modeling | None | Multi-dimensional (tokens, tools, dollars) |

**Is CASSI different?** CaRT is the closest prior work in spirit — same goal of learning when to stop. CASSI adds: (a) separate model, (b) cost-awareness, (c) continuous value signal. But:

- **"Separate model"** is an architectural choice, not proven better. CaRT shows the same model CAN do both reasoning and termination. The paper plan's capacity argument (Section 18.5) is hand-waving.
- **"Cost-awareness"** could be added to CaRT with a 3-line reward modification: include `−λ*cost` in the binary reward.
- **"Continuous Δ"** is a Q-value difference. CaRT's internal reasoning traces already encode a continuous assessment of termination appropriateness.

**Critical missing experiment:** (a) CaRT + cost penalty in reward, (b) CaRT + GRPO (to control for training algorithm), (c) CASSI monitor but with SFT only (to control for RL). These disentangle architectural contribution from training algorithm contribution from reward design contribution.

**Verdict: CASSI = CaRT + separate model + cost term − counterfactuals. The "separate model" aspect is what needs proving.**

---

### 3. L1 / Reason Efficiently (Aggarwal & Welleck, 2025; Arora & Zanette, NeurIPS 2025)

| Aspect | L1 / Reason Efficiently | CASSI |
|---|---|---|
| Cost penalty | Static: R = 1{correct} − α × length | Dynamic: per-step Δ(s_t) from separate monitor |
| Adaptation | Same α for all instances | Per-instance, mid-trajectory |
| Training | 200 gradient updates (Reason Efficiently) | Full SFT + GRPO pipeline for monitor + executor |
| Where is cost? | Embedded in policy model | Externalized in monitor model |

**Is CASSI different?** The "dynamic, per-instance adaptation" is CASSI's strongest differentiator. Reason Efficiently shows you can get 50% token reduction with 200 gradient updates and a 2-line reward change. CASSI requires orders of magnitude more computation for what might be a marginal improvement.

**Critical missing experiment:** Compare CASSI against Reason Efficiently with per-instance adaptive α (α chosen by a lightweight difficulty classifier, no separate monitor). This tests whether the monitor's mid-trajectory signal adds value beyond instance-level adaptation. Also: correlation analysis between task difficulty and stopping point (H5). If r < 0.3, the monitor is not adapting — it's learning a fixed threshold.

**Verdict: The dynamic adaptation claim is the ONE potentially genuine advance. But it must be proven, and the bar is high: CASSI must significantly outperform a well-tuned static α on mixed-difficulty tasks.**

---

### 4. ReMA (Wan et al., NeurIPS 2025) — "Learning to Meta-Think for LLMs with Multi-Agent RL"

| Aspect | ReMA | CASSI |
|---|---|---|
| Architecture | High-level meta-agent + low-level reasoning agent | Monitor + executor |
| Signal | Strategic plans + consistency rewards | STOP/CONTINUE/ADJUST + Δ |
| Training | Iterative MARL (PPO/REINFORCE++) | Sequential SFT+GRPO |
| Domain | Math reasoning, LLM-as-Judge | Web agents, QA, coding |

**Is CASSI different?** Architecturally, ReMA and CASSI are nearly identical — two-agent hierarchical RL. The difference is in *what* the high-level agent evaluates: ReMA's evaluates strategies; CASSI's evaluates stopping. CASSI is a **special case of ReMA applied to cost-aware stopping** — narrowing a general framework to a specific decision type.

**Critical missing experiment:** Adapt ReMA to include cost-awareness (+ `−λ*cost` in reward) and a "STOP and submit" action in the meta-agent's strategy space. If CASSI outperforms "ReMA-cost," the specialized architecture matters. If not, CASSI is a special case of ReMA.

**Verdict: CASSI is ReMA specialized for stopping. The paper must demonstrate that the specialization produces better outcomes than a general ReMA-style meta-agent with cost-awareness.**

---

### 5. BATS (Liu et al., 2025) — "Budget-Aware Tool-Use Enables Effective Agent Scaling"

| Aspect | BATS | CASSI |
|---|---|---|
| Budget tracking | Yes (prompt-level) | Yes (learned monitor) |
| Verification | Heuristic "dig deeper" vs. "pivot" | Learned STOP/CONTINUE/ADJUST |
| Training | Training-free | Full SFT+RL pipeline |
| Cost model | Unified token + tool metric | Multi-dimensional (tokens, tools, dollars, iterations) |

**Is CASSI different?** BATS and CASSI make the same type of decision (dig deeper/pivot vs. stop/continue/adjust). The difference is heuristic vs. learned. Whether learning provides a meaningful advantage over well-designed heuristics is an **empirical question with no answer**.

**Critical missing experiment:** Compare CASSI against BATS with heuristics optimized on the same training data. If CASSI significantly outperforms optimized BATS (>5% on any metric), learning matters. If the gap is narrow (<3%), the complexity is not justified.

**Verdict: Potentially better, but hypothesized margins (1.9% accuracy gain on GAIA) are too small to justify the complexity.**

---

## What Is Genuinely Novel vs. Incremental vs. Overclaimed

### Genuinely Novel (1 item)

1. **Training a cost-aware process reward model for dynamic, mid-trajectory stopping decisions.** No existing PRM evaluates cost-quality trade-offs. No existing stopping method adapts mid-trajectory based on observed reasoning progress. This is the ONE claim that survives scrutiny — but it must be proven empirically.

### Incremental (3 items)

2. **The separate monitor architecture.** The architectural pattern exists in ReMA and AgentPRM. CASSI applies it to cost-aware stopping, which is a specialization, not an invention.
3. **The cost-aware oracle label formulation.** `quality_t − λ × cumulative_cost` is a standard constrained optimization objective. The novelty is in applying it to agent trajectory stopping labels.
4. **Empirical demonstration across diverse agent benchmarks.** If the experiments work, showing that cost-aware stopping works across web search, QA, and coding is a useful empirical contribution.

### Overclaimed (4 items)

5. **"First to propose a learned, separate monitor for cost-aware stopping."** False. CaRT proposed learning to stop. AgentPRM proposed separate process reward models. Stop Wasting Your Tokens proposed separate supervisors. CASSI combines them and adds cost-awareness.
6. **"The soft stopping curve as a contribution."** Standard RL value function rebranded. Drop as a named contribution.
7. **"Three-phase training pipeline as a contribution."** Standard iterative actor-critic. Drop.
8. **"Multi-dimensional budget representation."** BATS, INTENT, and BAVT already do this. Drop.

---

## Revised Contributions List (Recommended)

Strip down to these 3 claims:

1. **Cost-aware process reward modeling.** We train a process reward model to evaluate cost-quality trade-offs (not just quality) using oracle labels that balance task correctness with cumulative resource consumption. This produces a learned cost-aware value function Δ(s_t) that estimates whether continuing is worth its cost.

2. **Dynamic, per-instance stopping outperforms static length penalties.** We demonstrate that a monitor providing instance-adaptive, mid-trajectory cost feedback enables agents to spend more compute on hard problems and less on easy ones — outperforming static length penalties (L1, Reason Efficiently) on benchmarks with heterogeneous task difficulty.

3. **Practical finding: small monitors effectively supervise large executors.** A 0.5B–3B parameter monitor can supervise 7B–72B parameter executor agents, with the monitor's inference overhead (<3% of total cost) far outweighed by the savings it produces (35–46% cost reduction), enabling net-positive deployment.

---

## Critical Experiments That Must Succeed

These are the experiments that will make or break the paper:

### Must-Pass (Paper fails without these)

1. **CASSI vs. Well-Tuned Static α on Mixed-Difficulty Tasks.** Compare CASSI against Arora & Zanette (NeurIPS 2025) with the best single α found via grid search on the same data. If CASSI doesn't significantly outperform, the "dynamic adaptation" claim is false. *Expected threshold: p < 0.05 on cost reduction at iso-accuracy.*

2. **Stopping Point Variance Explained by Task Difficulty (H5).** Compute correlation between task difficulty (e.g., number of hops in MuSiQue, question complexity in GAIA) and CASSI's stopping point. If r < 0.4, the monitor is not adapting — it's learning a fixed threshold. *Expected threshold: r > 0.5.*

3. **Monitor Overhead < Savings (H8).** Measure total monitor inference cost vs. total savings vs. unconstrained ReAct. If net savings are negative or marginal (<5%), the approach is not practical. *Expected threshold: net savings > 15%.*

### Should-Pass (Paper is significantly weaker without these)

4. **CASSI vs. AgentPRM-cost.** Train AgentPRM with `R = quality − λ*cost` in the Monte Carlo return. Compare on the same benchmarks. This isolates the value of CASSI's specialized monitor design (structured Δ output, SFT+RL, budget tiers) vs. a generic PRM.

5. **CASSI vs. CaRT + cost.** Train CaRT with `−λ*cost` in the binary reward. Compare on stopping accuracy. This isolates the value of a separate monitor vs. self-termination.

6. **Ablation: SFT-only Monitor vs. SFT+RL Monitor.** If RL doesn't improve the monitor beyond SFT, the GRPO phase is unnecessary complexity.

7. **Ablation: Monitor-Δ as Reward vs. Monitor as Stopping Controller Only.** If using the monitor as a reward model for RL training doesn't outperform using it only for inference-time stopping, the "reward model" framing is unsupported.

### Nice-to-Have

8. **Cross-Model Transfer.** Monitor trained on 7B executor supervising 32B executor. Shows practical reusability. But if transfer is poor, drop this claim — it's not essential.

9. **Comparison with BATS Optimized Heuristics.** Shows whether learning beats engineering. But if the gap is small, don't claim this as a win.

---

## What a Skeptical Reviewer Would Say (Simulated Rejection)

> "This paper proposes CASSI, a 'new training paradigm' where a separate monitor agent learns to make cost-aware stopping decisions. The idea is reasonable, but the claimed novelty is substantially overstated.
>
> **First, the 'separate monitor' architecture is not new.** AgentPRM (Choudhury, 2025) already proposed a separate Process Reward Model providing step-level Q-values for policy training — this is structurally identical to CASSI's monitor. ReMA (Wan et al., NeurIPS 2025) already demonstrated two-agent hierarchical RL with a high-level meta-agent. The paper's claim to be the 'first' is contradicted by its own literature review.
>
> **Second, the 'soft stopping curve' is a relabeling of standard RL value functions.** Δ(s_t) = Q_continue − Q_stop is the advantage function for the stop action. The theoretical analysis (Section 18) proves trivial statements (Theorem 1: maximizing over a finite set has a maximum). This is padding, not contribution.
>
> **Third, the paper's hypothesized results tables** (e.g., CASSI achieves 37.5% on GAIA) are not real results. They are fabricated for illustration. At ICLR, reviewers expect empirical validation.
>
> **Fourth, the comparisons to prior work are unfair.** The paper compares CASSI's RL-trained separate monitor against CaRT's SFT-only self-termination — conflating training algorithm with architecture. It does not compare against AgentPRM with a cost term, which would be the most direct baseline.
>
> **The ONE potentially interesting idea** — dynamic, per-instance, mid-trajectory cost adaptation — is not isolated. The paper conflates three things: (a) having a separate model, (b) training it with RL, and (c) adding cost to the reward. Any one of these applied to CaRT or AgentPRM could account for the hypothesized gains. Without careful ablations that disentangle these factors, the paper cannot claim which of its design choices actually matters.
>
> **Recommendation: Reject.** The paper repackages existing ideas under new names without demonstrating that the repackaging produces meaningfully better outcomes. The authors should (1) run the experiments, (2) add fair architectural baselines, (3) strip the overclaimed novelty statements, and (4) resubmit when they have empirical evidence for their central claim: that dynamic, mid-trajectory cost adaptation outperforms simpler alternatives."

---

## Action Items Before Submission

| Priority | Action | Rationale |
|---|---|---|
| **P0** | Run experiments. No paper without results. | Minimum bar for any submission. |
| **P0** | Add AgentPRM-cost baseline. | Most direct architectural comparison. |
| **P0** | Add CaRT+cost+GRPO baseline. | Controls for training algorithm + cost-awareness. |
| **P0** | Test H5 (stopping point vs. difficulty correlation). | Proves or disproves the "dynamic adaptation" claim. |
| **P1** | Compare against best static α on mixed-difficulty data. | Proves CASSI beats simpler methods. |
| **P1** | Rewrite contributions list to 3 claims (see above). | Removes overclaiming. |
| **P1** | Rename "soft stopping curve" to "cost-aware value function." | Removes branding-as-contribution. |
| **P1** | Remove "three-phase pipeline" and "multi-dimensional budget" from contributions. | These are methodology, not contributions. |
| **P2** | Add ReMA-cost baseline. | Controls for two-agent architecture. |
| **P2** | Add BATS optimized heuristics baseline. | Tests learning vs. engineering. |
| **P2** | Add total-cost-of-ownership analysis (training + inference). | Addresses "complexity vs. gain" concern. |
| **P2** | Strengthen or remove the "capacity argument" (Section 18.5). | Currently hand-waving; either formalize or drop. |
| **P3** | Cross-model transfer experiments. | Nice-to-have practical contribution. |
| **P3** | Domain transfer experiments. | Only if results are positive; if transfer is poor, drop this claim. |

---

## Final Verdict

**The paper identifies a real and important problem.** LLM agents overthink, over-search, and waste resources. Teaching them cost-aware stopping is valuable.

**The proposed solution is architecturally sound but not novel.** CASSI's monitor-executor architecture is a standard two-agent hierarchical RL setup. The training pipeline is standard iterative actor-critic. The "soft stopping curve" is a standard value function.

**The genuine contribution is narrower than claimed.** The paper's real contribution — if the experiments succeed — is demonstrating that (1) cost-aware process rewards produce better agent efficiency than quality-only rewards, and (2) dynamic, per-instance cost adaptation outperforms static penalties on heterogeneous tasks.

**The paper is currently unpublishable** because it has zero experimental results, overclaims its novelty, and lacks fair comparisons with the closest prior work (AgentPRM-cost, CaRT+cost, ReMA-cost).

**The path to acceptance:** Run experiments → add fair baselines → strip overclaimed novelty → demonstrate that dynamic adaptation meaningfully outperforms static penalties. If the experiments confirm the hypotheses, this is a solid paper. If they don't, the idea needs rethinking.