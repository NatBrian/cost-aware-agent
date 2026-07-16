# Final Novelty Assessment — Converged (v4)

> **Paper:** "Scalable Cost-Aware Agent Training via Oracle-Guided Stopping Rewards" (CASSI)
> **Iterations:** 5 rounds (v1 → v2 → v3 → v4)
> **Status:** Refinement ceiling reached. Further improvement requires experiments.

---

## Score Evolution Across Iterations

| Dimension | v1 | v2 | v3 | v4 | Δ Total |
|---|---|---|---|---|---|
| Scientific Novelty | 2.0 | 2.5 | 2.8 | **3.5** | +1.5 |
| Technical Depth | 2.0 | 2.3 | 2.8 | **3.5** | +1.5 |
| Experimental Credibility | 1.0 | 1.0 | 2.5 | **4.0** | +3.0 |
| Significance | 3.0 | 3.5 | 3.0 | **3.5** | +0.5 |
| Presentation | 4.0 | 4.0 | 2.5 | **3.5** | −0.5 |
| **Composite** | **2.4** | **2.7** | **2.8** | **3.6** | **+1.2** |

---

## What Changed (v1 → v4)

| Iteration | Key Changes | Score Impact |
|---|---|---|
| **v1 → v2** | Reframed from "separate monitor architecture" to "scalable stopping rewards." Dropped 4 overclaimed contributions. Added complexity analysis O(T) vs O(K×T²). Added "Why Not AgentPRM" section. Terminology sweep. | +0.3 |
| **v2 → v3** | Applied P0 baselines (Zero-Training Self-Eval, Adaptive-α, AgentPRM-cost, CaRT+cost+GRPO). Elevated H5 to load-bearing P0. Rewrote contributions (3 claims + design choices section). Replaced trivial theorems with complexity analysis. Dropped iterative refinement. Added oracle label limitations. | +0.1 |
| **v3 → v4** | Fixed section numbering chaos (29 consecutive sections). Defined task difficulty per benchmark. Removed fabricated expected-results tables. Unified H5 threshold (r > 0.5). Dropped transfer claims. Removed backronym. Expanded "Why Not AgentPRM" with concrete SWE-bench example. Added oracle label quality validation experiment. | +0.8 |

---

## Final Score Justification

### Scientific Novelty — 3.5/5

**Strong:** Oracle-guided stopping labels (O(T) vs. O(K×T²)) is a genuine algorithmic insight. Dynamic per-instance adaptation distinguishes from static penalties. Honest about what ISN'T novel (design choices section).

**Limited:** Single insight. No theoretical guarantees (convergence, regret bounds). The "argmax over recorded values" operation is trivial — the contribution is the application, not the math.

### Technical Depth — 3.5/5

**Strong:** Rigorous complexity analysis with concrete numbers. Multi-dimensional cost model with adaptive λ tiers. Preemptive "Why Not AgentPRM" section. Honest caveats about oracle label limitations and MC parallelization.

**Limited:** No formal learning theory. No convergence proofs. The "focus = better sample efficiency" claim is a hypothesis, not a theorem.

### Experimental Credibility (Design) — 4.0/5

**Strong:** 13 baselines including 4 P0 must-pass. H5 elevated with operationalized difficulty per benchmark. Oracle label quality validation (human annotators, Cohen's κ). Control condition (MATH-500). Edge-case testing. 3-seed protocol. Ablation suite covers 6 axes.

**Limited:** No results. AgentPRM-cost baseline may be computationally prohibitive. Adaptive-α baseline is underspecified. All scores are design quality, not execution.

### Significance — 3.5/5

**Strong:** Enables cost-aware PRM training on benchmarks where it was intractable (SWE-bench). Real economic impact for deployed agent systems. The O(K×T²) → O(T) reduction is practically meaningful.

**Limited:** Efficiency contribution, not a capability breakthrough. The method doesn't make agents more capable — it makes them more efficient at the same capability level. Per-domain training requirement limits deployment breadth.

### Presentation — 3.5/5

**Strong:** Clean narrative. Honest positioning. Clear contributions vs. design choices separation. Strong elevator pitch. Well-structured sections.

**Limited:** Sub-section numbering still has minor errors (orphaned 8.6, off-by-one in subsections 18-25). Internal cross-references may be stale.

---

## ICLR Verdict

**If experiments match expectations: 6.5/10 — Weak Accept**

The paper tells a clean, honest story: it doesn't claim to invent a new architecture or paradigm — it makes an existing approach (cost-aware PRM training) tractable via a simple but effective trick (oracle labels from completed trajectories). The honesty about what is and isn't novel will earn reviewer respect.

### What Prevents a Strong Accept (8+)

1. **Single insight.** One algorithmic trick wrapped in standard machinery. Strong ICLR papers typically contribute 2-3 interconnected insights.
2. **No results.** The design is comprehensive but execution is everything. H5 (r > 0.5) is the load-bearing wall.
3. **Efficiency, not capability.** The paper makes training cheaper, not agents smarter. This limits its intellectual contribution.
4. **No theoretical guarantees.** The complexity analysis is correct but doesn't prove anything about convergence, optimality, or sample complexity.

### What Could Push It to Accept (7+)

1. **H5: r > 0.6** (exceeding the r > 0.5 threshold) would show strong dynamic adaptation.
2. **CASSI significantly (>10%) outperforms adaptive-α Reason Efficiently** on mixed-difficulty tasks.
3. **CASSI's stopping model achieves >80% oracle agreement** with <3% inference overhead.
4. **Training time grows linearly with T** while AgentPRM-cost grows quadratically — confirming the core efficiency claim.

---

## Remaining Unfixable Issues (Require Experiments)

| Issue | Why Experiments Are Required |
|---|---|
| Oracle label signal quality vs. MC rollouts | Only empirical comparison can show O(T) labels match O(K×T²) signal |
| H5: r > 0.5 correlation with difficulty | The entire dynamic adaptation claim depends on this |
| Beating zero-training and adaptive-α baselines | If simple methods work, the separate model is unjustified |
| Cross-model transfer | Stopping patterns may be model-specific — only data can tell |
| Actual cost savings on GAIA/SWE-bench | Aspirational 30-50% range — real savings depend on trajectory slack |
| AgentPRM-cost baseline feasibility | May be too expensive to run at scale |

---

## Refinement Ceiling

**Further refinement without experiments will yield diminishing returns.** The paper plan has reached the quality ceiling for a pre-experiment document. The remaining variance (3.6 → ~6.5/10 at ICLR) requires:

1. Running the experiments
2. Proving H5 (r > 0.5)
3. Demonstrating significant improvement over adaptive-α baselines
4. Showing linear training-time scaling for CASSI

The paper's design is sound. The framing is honest. The baselines are comprehensive. **Time to build.**
