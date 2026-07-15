# Novelty Check v2 — Consolidated Meta-Review

> **Paper:** "Scalable Cost-Aware Agent Training via Oracle-Guided Stopping Rewards" (CASSI v2)
> **Reviewers:** 3 independent ICLR-level reviewers (Algorithmic, Architecture & Training, Meta-Review)
> **Date:** 2026-07-15

---

## Aggregate Scores

| Dimension | R1 (Algorithm) | R2 (Arch+Training) | R3 (Meta) | **Average** |
|---|---|---|---|---|
| Scientific Novelty | 2.0 | 2.0 | 3.5 | **2.5** |
| Technical Depth | 2.0 | 2.0 | 3.0 | **2.3** |
| Experimental Credibility | — | — | 1.0 | **1.0** (no results) |
| Significance | — | — | 3.5 | **3.5** |
| Presentation/Coherence | — | — | 4.0 | **4.0** |
| **Composite** | — | — | — | **≈2.7 / 5.0** (with results: ≈3.2) |

**Verdict: Weak Reject (current) → Borderline/Weak Accept (with experiments)**

The v2 reframe is a substantial improvement over v1 (2.4 → ≈2.7 without experiments, ≈3.2 with). But three structural issues remain that no amount of reframing can fix without changing the method itself.

---

## What Improved (v1 → v2)

| Aspect | v1 | v2 | Verdict |
|---|---|---|---|
| **Framing** | "First separate monitor for cost-aware stopping" | "Scalable cost-aware agent training via oracle-guided stopping rewards" | Big improvement — honest, differentiated |
| **Primary contribution** | 7 claims, 4 overclaimed | 3 claims, all defensible | Big improvement |
| **AgentPRM positioning** | Mentioned as related work | Central contrast + dedicated "Why Not" section | Good idea, execution has issues |
| **Terminology** | "Monitor agent," "soft stopping curve" | "Stopping model," "cost-aware value function" | Honest, no rebranding |
| **Architecture claims** | Claimed as contribution | Demoted to "design choice" | Correct and necessary |
| **Baselines** | Missing AgentPRM-cost, CaRT+cost | Added 4 critical baselines + RQ7 | Comprehensive in principle |
| **Theoretical analysis** | Trivial theorems (max exists, error bounded) | Complexity analysis O(T) vs O(K×T²) | More useful, but thin |

---

## What Still Needs Fixing

### Critical Issue #1: The "Why Not AgentPRM?" Section Is Weak (R1, R3)

The v2 reframe adds Section 8.0 arguing why AgentPRM-cost fails. But 2 of 3 arguments are wrong or misleading:

| Argument | R1 Verdict | Why |
|---|---|---|
| **Computational intractability** ($80/trajectory) | **Not convincing** | Rollouts can be parallelized on GPUs. If the paper's own experiment budget is $10K+, $80/trajectory is negligible. The "intractability" claim is hyperbolic. |
| **Action space explosion** (AgentPRM must evaluate all actions) | **Mostly wrong** | AgentPRM trains on sampled (s,a) pairs, not all possible actions. A PRM is a function approximator that generalizes from observed actions. |
| **Credit assignment dilution** (focusing on stop/continue avoids diluting signal) | **Partially convincing** | This is a genuine hypothesis but unproven. It should be presented as a hypothesis to test, not as self-evident truth. |

**What to do:** Either (a) rewrite this section to present the arguments as hypotheses backed by the complexity analysis (which IS rigorous), or (b) drop the weak arguments and lead exclusively with the complexity analysis: "AgentPRM's MC rollouts cost O(K×T²) per trajectory; our oracle labels cost O(T). For T=20, K=8, this is 160× fewer additional executions. Whether this matters for final performance is an empirical question we address in Section 9."

### Critical Issue #2: The Iterative Refinement Loop Contradicts Efficiency Claims (R2, R3)

The v2 plan keeps Algorithm 5's iterative refinement loop:

```
for iteration in 1..K:
    Collect trajectories from current executor   // Cost: O(T)
    Retrain monitor on new trajectories           // Cost: SFT + GRPO
    Retrain executor with updated monitor         // Cost: GRPO
```

If K=5 iterations, total trajectory collection cost is 5×T, not T. The honest total cost is O(K×T), not O(T). This undermines the central efficiency claim.

**What to do:** Pick one:
- **Option A:** Drop iterative refinement entirely. Claim single-pass training is sufficient. This is the cleanest option and strengthens the O(T) efficiency claim.
- **Option B:** Keep it but do honest total-cost math: "Single-pass CASSI: O(T). Iterative CASSI with K=5: O(5×T). AgentPRM with K=8 MC rollouts: O(8×T²). For T=20: CASSI single-pass = 20, CASSI iterative = 100, AgentPRM = 3200." The math still favors CASSI, but acknowledge it honestly.

### Critical Issue #3: H5 Is the Load-Bearing Wall (R3)

All three reviewers agree: the paper lives or dies on H5 — the correlation between task difficulty and stopping point. If r < 0.3:

- The "dynamic, per-instance adaptation" claim is **false**
- CASSI is learning a fixed threshold, equivalent to L1 with extra parameters
- The paper has no contribution at any top venue

**What to do:** Elevate H5 to P0 status. It must be the first figure in the results section. Before showing any cost-accuracy tables, show the scatter plot of stopping step vs. task difficulty, with correlation coefficient and p-value. If r > 0.5, the paper has a story. If r < 0.3, abandon the paper or reframe completely.

---

## Issues Unresolved by the v2 Reframe

### 1. The Oracle Label Is Noisy and Path-Dependent (R1)

`t* = argmax_t [quality_t − λ × cost_{1..t}]` only considers states actually visited. If the executor took a bad action at step 2 and never recovered, t* is suboptimal. Quality at intermediate steps is noisy (a partial answer may look correct but be wrong). The oracle label has no variance reduction — unlike MC rollouts which average over multiple completions.

**CaRT's counterfactual approach is more principled.** CaRT isolates the *causal* signal for termination by constructing minimally-modified trajectories where continuation becomes necessary. CASSI's oracle labels conflate correlation (quality happened to be high) with causation (this is where we should stop).

**Missed synthesis:** Combine CaRT's counterfactuals with CASSI's oracle labels. Use oracle labels to identify candidate t*, then construct counterfactual pairs (stop at t* vs. continue past t*) for training. This would be genuinely novel — neither paper alone does it.

**What to do:** Add a paragraph in the discussion acknowledging this limitation and proposing the counterfactual+oracle synthesis as future work. If time permits, add a "CASSI+CaRT" hybrid as an additional experiment.

### 2. Single-Insight Paper (R2, R3)

The entire contribution reduces to: "compute t* from completed trajectories." Everything else — two-agent architecture, GRPO training, budget tracking, iterative refinement — is standard machinery. Accepted ICLR/NeurIPS papers typically contribute 2-3 interconnected insights. CASSI has one.

**What to do:** This is not fixable without adding new insights. Accept it and position accordingly:
- The paper is an **efficiency contribution to PRM training**, not a fundamental advance in agent learning
- Target venues that value practical efficiency improvements (EMNLP, AAAI, CoLM) alongside ICLR/NeurIPS
- If the experiments are strong, the narrowness is offset by empirical impact

### 3. Missing Zero-Training Baseline (R3)

No baseline for: "Give the executor a prompt asking it to self-evaluate whether to stop." This is the first thing a reviewer will think of. If prompt-based stopping gets 80% of CASSI's savings with zero training, learning isn't justified.

**What to do:** Add as P0 baseline. Prompt: "Evaluate your current answer. If you are confident it is correct and complete, output [FINAL ANSWER]. If you need more information, continue searching."

### 4. The "Separate Model" Justification Is Now Just "Design Choice" (R2)

After demoting architecture from contribution to design choice, the question "why a separate model?" has a weak answer. The v2's implicit justification is: "our labeling trick requires recording trajectories and computing t* post-hoc, which you can't do during generation with a single model." But you COULD do it with a single model trained to output its own Δ value after each step. The separation is a design choice, not a requirement.

**What to do:** Be explicit: "We use a separate stopping model because (1) it can be much smaller than the executor, minimizing inference overhead, (2) it can be trained on offline trajectory data while the executor is deployed, (3) it can supervise multiple executor models. None of these require architectural novelty — they're practical engineering benefits of separation." This is honest and reviewers will accept it.

### 5. Domain Transfer Numbers Are Self-Defeating (R3)

The v1 plan's hypothesized numbers show 35% degradation (84.3% → 65.2%) on domain transfer. That's not generalization — it's failure. If v2 still makes transfer claims with similar numbers, reviewers will destroy them.

**What to do:** Either (a) drop transfer claims entirely (the paper doesn't need them), or (b) reframe as "we find that transfer is limited; the stopping model is best trained per-domain, and the cost of doing so is low because training is O(T) per trajectory."

---

## Revised Score Card (Post-Fixes)

If the critical issues above are addressed:

| Dimension | Current v2 | After Fixes | What Changed |
|---|---|---|---|
| Scientific Novelty | 2.5 | **3.0** | Honest positioning + counterfactual synthesis idea + acknowledged limitations |
| Technical Depth | 2.3 | **3.0** | Fixed "Why Not AgentPRM" section + honest complexity math without hyperbole |
| Experimental Credibility | 1.0 | **3.5** | Experiments run + P0 baselines added + H5 as lead figure |
| Significance | 3.5 | **3.5** | Unchanged — already the strongest dimension |
| Presentation | 4.0 | **4.0** | Unchanged — already strong |
| **Composite** | **2.7** | **3.4** | **Weak Accept territory** |

---

## Revised Contributions List (v3)

After incorporating all reviewer feedback:

### Primary Contributions (Claimed)

1. **Oracle-guided stopping rewards for scalable cost-aware PRM training.** We show that optimal stopping labels can be computed analytically from completed trajectories as `t* = argmax_t [quality_t − λ × cost_{1..t}]`, reducing the training computation for cost-aware process reward models from O(K×T²) (Monte Carlo rollouts from every intermediate state) to O(T) (single trajectory with post-hoc labeling). For T=20, K=8, this eliminates ~160 additional policy executions per trajectory — making cost-aware PRM training tractable on long-horizon benchmarks where prior methods were computationally prohibitive.

2. **Empirical demonstration that dynamic, per-instance cost adaptation outperforms static penalties on heterogeneous agent tasks.** Our stopping model adapts cost pressure based on observed reasoning progress — spending more compute on hard problems and less on easy ones. We show (a) significant correlation between task difficulty and stopping point (r > 0.5), (b) better cost-accuracy Pareto frontiers than static length penalties (L1, Reason Efficiently) and quality-only PRMs (AgentPRM).

3. **A small stopping model (0.5B–3B) effectively supervises large executor agents (7B–72B)** with inference overhead <3% of total cost, producing net savings of 35–46% on web search, multi-hop QA, and software engineering benchmarks.

### Design Choices (Not Claimed as Contributions)

- Two-model architecture (following the PRM pattern from AgentPRM and ReMA)
- Multi-dimensional budget encoding (following BATS and INTENT)
- Cost-aware value function Δ(s_t) (standard advantage formulation specialized for stopping)
- Iterative refinement loop (standard actor-critic, included only if shown to improve over single-pass)

---

## P0 Action Items (Must Do Before Submission)

| # | Action | Rationale |
|---|---|---|
| 1 | **Run all experiments.** No results = no paper. | Minimum bar. |
| 2 | **Make H5 the lead result.** Scatter plot: stopping step vs. task difficulty. r > 0.5 required. | Load-bearing wall. If this fails, abandon the paper. |
| 3 | **Add zero-training baseline.** Prompt-based self-evaluation of stopping. | First thing reviewers think of. |
| 4 | **Add AgentPRM-cost baseline and run it.** If you claim AgentPRM is intractable, you must run it to prove it. | Central contrast. Without this, the efficiency claim is theoretical only. |
| 5 | **Fix or drop "Why Not AgentPRM" section.** Either rewrite with honest, hypothesis-framed arguments, or replace with pure complexity analysis. | Currently weak (2/3 arguments wrong). |
| 6 | **Resolve iterative refinement contradiction.** Either drop iterative loop or do honest O(K×T) math. | Contradicts O(T) efficiency claim. |
| 7 | **Drop domain transfer claims** (or reframe as "we find transfer is limited; stopping models are cheap to train per-domain"). | 35% degradation = failure, not contribution. |

## P1 Action Items (Should Do)

| # | Action | Rationale |
|---|---|---|
| 8 | **Add CaRT+cost+GRPO baseline.** Controls for training algorithm + cost-awareness + architecture. | Disentangles which factor matters. |
| 9 | **Add adaptive-α Reason Efficiently baseline.** Difficulty classifier picks α per instance. Tests whether monitor adds value beyond instance-level classification. | Critical for "dynamic adaptation" claim. |
| 10 | **Add SFT-only monitor ablation.** If SFT alone achieves 95% of SFT+RL, the RL phase is unjustified complexity. | Proves RL adds value. |
| 11 | **Add BATS-optimized-heuristics baseline.** Grid-search BATS parameters on training data. Tests whether learning beats engineering. | If BATS-optimized ≈ CASSI, learning isn't worth it. |
| 12 | **Acknowledge oracle label limitations** (path-dependence, noise) and propose counterfactual+oracle synthesis as future work. | Shows awareness of limitations. |

## P2 Action Items (Nice to Have)

| # | Action | Rationale |
|---|---|---|
| 13 | **Add CASSI+CaRT hybrid experiment** (oracle labels + counterfactual pairs). | Genuinely novel combination. |
| 14 | **Add ReMA-cost baseline.** Controls for two-agent architecture. | If ReMA-cost ≈ CASSI, the stopping specialization doesn't add value. |
| 15 | **Total-cost-of-ownership analysis.** Training cost + inference cost + monitor overhead vs. savings. | Addresses "complexity vs. gain" concern. |
| 16 | **Cross-model transfer** (monitor trained on 7B supervising 32B). | If it works, practical contribution. If it fails, drop it. |

---

## Final Verdict

**The v2 reframe is a necessary and substantial improvement.** The shift from "separate monitor architecture" to "scalable stopping rewards via oracle-guided labeling" is the right move — it's more honest, more differentiated from prior work, and gives the paper a clean narrative arc.

**But the paper is still not ready for submission.** Three structural issues remain:
1. The oracle labeling algorithm, while clever, is a single insight wrapped in standard machinery
2. The "Why Not AgentPRM" section is weak and damages credibility
3. Most importantly: **zero experimental results and H5 is unproven**

**If experiments succeed** (especially H5), this is a **Weak Accept at ICLR** — a solid empirical contribution demonstrating that cost-aware stopping can be trained efficiently through oracle-guided labels.

**If experiments fail** (H5 r < 0.3), the paper has no contribution at any top venue and should be abandoned or reframed entirely.

**If experiments aren't run**, this remains a well-structured research proposal, not a conference paper.

---

## Comparison: v1 vs. v2 vs. Target

| | v1 (Old) | v2 (Current) | v3 (Target with Fixes) |
|---|---|---|---|
| **Title** | "Monitor Agent as Cost-Aware Reward Model" | "Scalable Cost-Aware Agent Training via Oracle-Guided Stopping Rewards" | Same |
| **Architecture claim** | "First separate monitor" (overclaimed) | "Design choice" (honest) | Same |
| **Algorithmic claim** | Buried in Section 8.3.1 | Central contrast with AgentPRM | Strengthened: honest complexity math, fixed "Why Not" section |
| **Contributions** | 7 (4 overclaimed) | 3 (all defensible) | 3 (all defensible) |
| **Baselines** | Missing AgentPRM, CaRT+cost | Added 4 critical baselines | Added zero-training + SFT-only ablation |
| **H5** | Listed as research question | Not elevated | **P0 — lead figure in results** |
| **Iterative refinement** | Included | Included (unresolved contradiction) | **Dropped or honestly costed** |
| **Domain transfer** | Claimed as contribution | Still present | **Dropped or reframed as limitation** |
| **Overall Score** | 2.4/5 | 2.7/5 | **3.4/5** |
| **Verdict** | Reject | Weak Reject (no results) | **Weak Accept (with results)** |
