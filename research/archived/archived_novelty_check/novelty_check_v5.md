# CASSI Novelty Assessment for ICLR

> **Conducted by:** 3 independent subagent reviewers, each reading actual papers in full
> **Papers reviewed in depth:** 15 most threatening competitor papers
> **Overall Score:** 3.0/5 (borderline-accept for ICLR)
> **Verdict:** Publishable with strong empirical results + careful positioning, but incremental combination of known techniques

---

## 1. Executive Summary

CASSI proposes a three-part innovation: (1) compute optimal stopping labels from completed trajectories via `t* = argmax_t [quality_t − λ × cumulative_cost]` — O(T), zero extra executions; (2) train a small stopping model on these labels; (3) use the stopping model's cost-aware value estimates as process rewards to train the executor via GRPO.

**The bad news:** None of the individual components are novel. GRPO is from DeepSeekMath (2024). Post-hoc trajectory labeling exists in CSO and SeqRoute. Small-controller-large-executor architectures exist in Ares and ReMA. Budget-aware RL training exists in BudgetThinker, SelfBudgeter, and Reason Efficiently. The underlying phenomenon (shorter chains can be better) is well-documented by When More is Less and Don't Overthink It.

**The good news:** The *specific combination* of these components — oracle-derived stopping labels → trained stopping model → cost-aware process rewards for executor GRPO → self-reinforcing training cycle — appears in **no single paper**. The `t* = argmax[quality − λ×cumulative_cost]` formulation applied to both labeling AND process rewards is genuinely new.

**The verdict:** 3.0/5 — borderline ICLR acceptance. The paper needs very strong empirical results and careful positioning against the closest competitors to survive review.

---

## 2. Threat Level Summary (All 15 Papers)

| Paper | Threat | Reason | CASSI's Defense |
|---|---|---|---|
| **CaRT** | 🔴 HIGH | Trains a stopper with cost penalty (γ). Already does "learn when to stop." CASSI = CaRT + process rewards for executor | CaRT's discussion lists "unified exploration + termination" as FUTURE WORK. CASSI does exactly that. CaRT uses binary threshold labels; CASSI uses continuous quality−λ×cost. Must include CaRT as baseline |
| **SeqRoute** | 🔴 HIGH | Hindsight Budget Relabeling (HBR) retroactively labels data with budget annotations. λ-sweep for Pareto navigation. Very similar to CASSI's oracle labeling + λ | HBR is for session-level model routing; CASSI is for within-trajectory stopping. HBR relabels with budget constraints; CASSI computes quality-cost optimality. Different problem structures. HBR is a general technique (from HER, 2017) — neither paper invented it |
| **Ares** | 🔴 HIGH | Same architecture: small router (1.7B) + large agent (20B). Same pipeline: collect trajectories → oracle labels → SFT → GRPO. Only difference: effort levels vs stop/continue | CASSI's oracle label formula is continuous (argmax quality−λ×cost), not discrete trial-and-error. Ares does NOT train the executor — the process reward bridge is CASSI's unique contribution. Ares limited to models with configurable effort levels |
| **CSO** | 🟡 MEDIUM | Post-hoc critical step identification from trajectories. Selective step-level supervision. Uses PRM for candidate identification | CSO identifies WHAT actions to take; CASSI determines WHEN to stop. Different RL problems. CSO's labels are for action correction; CASSI's are for termination decisions |
| **When More is Less** | 🟡 MEDIUM | Inverted-U curve, optimal CoT length, trains on optimally-lengthed CoTs. Pre-describes the phenomenon CASSI addresses | Characterizes the problem, doesn't solve it. No per-trajectory stopping objective. No stopping model. No process rewards. No cost-quality tradeoff parameter. CASSI is the prescriptive algorithm for what they describe |
| **BudgetThinker** | 🟡 MEDIUM | SFT + curriculum RL with budget-aware rewards. Control tokens for continuous budget awareness | Single-model approach; CASSI is two-model. CASSI's stopper is a plug-in, not an embedded pipeline modification. CASSI's oracle labels are post-hoc; BudgetThinker's are curriculum-based |
| **AgentPRM** | 🟡 MEDIUM | PRM → policy training framework. Most similar overall paradigm | No cost-awareness. No stopping. O(K×T²) vs CASSI's O(T). CASSI's cost-aware value + stopping model are orthogonal additions |
| **TALE** | 🟡 MEDIUM | Post-trains with per-instance budget labels. Two-stage pipeline | Global token budget per problem vs. per-step stopping. CASSI's oracle is per-timestep, not per-problem |
| **SelfBudgeter** | 🟡 MEDIUM | Self-budgeting with GRPO. Cold-start + RL pipeline | Single model, self-referential labels. No two-model architecture. No process rewards. CASSI's stopper is a separate learned component |
| **Don't Overthink It** | 🟡 MEDIUM | Shorter chains 34.5% more accurate. Trains on short chains | Inference heuristic + simple SFT comparison. No oracle objective, no stopping model, no process rewards, no agent tasks |
| **BATS** | 🟢 LOW | Training-free budget prompts. Different paradigm | CASSI is learning-based; BATS is prompt engineering. BATS validates the problem, not the solution |
| **BAVT** | 🟢 LOW | Budget-conditioned tree search. λ-like parameter | Training-free inference heuristic. CASSI is a training method. Different mechanisms |
| **Reason Efficiently** | 🟢 LOW | Length-penalized RL for math reasoning | Different problem domain (math CoT vs agent trajectories). No stopping model. No process rewards. No per-step decisions |
| **L1** | 🟢 LOW | Length-controlled RL with prompt targets | Completely different mechanism. User-specified budget vs CASSI's automatic oracle |
| **INTENT** | 🟢 LOW | Budget-constrained tool selection. Different problem | Plans WHICH tools to use under budget; CASSI decides WHEN to stop. Orthogonal |

---

## 3. The Three Biggest Threats (Detailed)

### Threat #1: CaRT (Liu et al., Oct 2025)
**Why it's dangerous:** CaRT already trains a model to decide when to stop information gathering. It uses a cost penalty (γ discount factor) that is conceptually equivalent to CASSI's λ. It computes oracle-style termination labels from completed trajectories. It uses GRPO as optional post-training. A reviewer will say: "CASSI is just CaRT + using the stopper's outputs as process rewards for the executor."

**CASSI's defense:**
- CaRT's discussion section EXPLICITLY lists "unified exploration and termination" as future work. CASSI does exactly this — CaRT's authors recognized it was non-trivial.
- CaRT uses BINARY threshold labels (success rate jump ≥50%). CASSI uses CONTINUOUS `quality − λ×cumulative_cost` optimization. This enables cost-quality Pareto navigation that CaRT cannot do.
- CaRT doesn't model cumulative cost — it uses per-step discounting γ^t. CASSI's Σcost_{1..t} handles variable per-step costs (tool calls differ in cost).
- Must include CaRT as a direct baseline comparison.

### Threat #2: SeqRoute + HBR (2026)
**Why it's dangerous:** Hindsight Budget Relabeling retroactively annotates historical data with budget signals — the same conceptual move as CASSI's oracle labeling. The λ-sweep deployment mechanism is nearly identical to CASSI's λ-based Pareto navigation. A reviewer will say: "CASSI's oracle labeling is just HBR applied to stopping."

**CASSI's defense:**
- HBR is a general technique dating back to Hindsight Experience Replay (Andrychowicz et al., 2017). Neither SeqRoute nor CASSI invented it.
- SeqRoute applies HBR to session-level MODEL SELECTION (discrete choice: weak vs. strong model). CASSI applies it to within-trajectory STOPPING (binary decision per step). These are fundamentally different problem structures.
- CASSI's oracle objective is `quality − λ×cumulative_cost` — a quality-cost tradeoff. SeqRoute's HBR relabels with budget constraints — a feasibility check. Different mathematical formulations.
- SeqRoute uses offline CQL; CASSI uses online SFT+GRPO with a separate stopping model.

### Threat #3: Ares (2026)
**Why it's dangerous:** Same architecture (small controller + large agent). Same pipeline (trajectory collection → oracle labeling → SFT → GRPO). A reviewer will say: "CASSI is Ares with 'stop/continue' instead of '{low,medium,high}'."

**CASSI's defense:**
- Ares' oracle labeling is DISCRETE trial-and-error: "test each effort level K=3 times, keep minimum that works." CASSI's oracle is a single-pass CONTINUOUS optimization: `argmax[quality − λ×cost]`. CASSI's labeling is O(T) and principled; Ares' is O(K×E) per step and heuristic.
- **Ares does NOT train the executor.** The agent is frozen. CASSI's stopper-to-executor process reward bridge is entirely absent from Ares.
- Ares requires models with configurable "thinking levels" (GPT-oss, Gemini). CASSI is model-agnostic.
- Ares' action space is reasoning effort; CASSI's is termination. Different problem domains.

---

## 4. Dangerous Combinations

Some combinations of papers could collectively threaten CASSI:

### Combination A: "When More is Less" + CSO + Agent-RRM
- When More is Less: optimal-length reasoning exists
- CSO: post-hoc critical step identification from trajectories
- Agent-RRM: separate RM + GRPO training pipeline
- **Reviewer argument:** "We already know shorter CoTs can be better (WMiL), we already identify critical steps from trajectories (CSO), and we already train executors with separate RMs (Agent-RRM). CASSI just recombines these."
- **Defense:** WMiL doesn't define a per-trajectory stopping objective. CSO identifies actions, not stopping points. Agent-RRM's RM is quality-only, not cost-aware. CASSI's specific combination of oracle objective + stopping model + cost-aware process rewards is new.

### Combination B: SeqRoute HBR + BudgetThinker
- SeqRoute: hindsight relabeling with λ-sweep
- BudgetThinker: budget-aware RL training
- **Reviewer argument:** "We already relabel data with budget signals (SeqRoute) and train models with budget-aware rewards (BudgetThinker). CASSI's two-model architecture is the only novelty, and that's not enough."
- **Defense:** HBR + budget-aware RL ≠ CASSI's oracle-guided stopping rewards. Neither SeqRoute nor BudgetThinker train a separate stopping model whose outputs become process rewards for the executor. The closed-loop training cycle (oracle → stopper → executor → better trajectories) is absent from both.

---

## 5. CASSI's Irreducible Novelty

**The single thing that no paper (or combination) does:**

> **A closed-loop training pipeline where an oracle-derived stopping objective (`t* = argmax_t [quality_t − λ × cumulative_cost_{1..t}]`) generates labels to train a stopping model, whose cost-aware value estimates Δ(s_t) become process rewards for executor GRPO training, creating a self-reinforcing cycle.**

Breaking this down into testable claims:

| Claim | Covered by any paper? | Verdict |
|---|---|---|
| `t* = argmax_t [quality_t − λ × cumulative_cost]` as oracle for stopping | No — CaRT uses binary thresholds, SeqRoute uses budget constraints, Ares uses discrete trial-and-error | **Novel** |
| Train small model on oracle stopping labels via SFT+GRPO | Ares does SFT+GRPO, but for effort levels, not stopping | **Partially novel** |
| Use stopping model's Δ(s_t) as cost-aware process rewards for executor GRPO | No paper uses a stopper's predictions as process rewards for the executor | **Novel** |
| Self-reinforcing cycle: executor → trajectories → oracle → stopper → process rewards → better executor | No paper closes this loop | **Novel** |
| Two-model architecture (small stopper + large executor) | Ares, ReMA, CORL already do this | **Not novel** |
| Per-instance dynamic adaptation | CaRT, SelfBudgeter, Reason Efficiently do this implicitly | **Not novel** |
| O(T) post-hoc labeling (zero extra executions) | CaRT and CSO also do post-hoc labeling, but with different objectives | **Partially novel** |

---

## 6. Reviewer Questions to Prepare For

| Question | Preparation |
|---|---|
| "CASSI = CaRT + process rewards for executor. Is that enough?" | CaRT's discussion lists this as future work. CASSI's continuous oracle + Σcost formulation is more principled than CaRT's binary threshold. Must show ablation: CaRT alone vs. CASSI full pipeline |
| "Ares already uses small-controller-large-executor + SFT+GRPO. What's new?" | Ares doesn't train the executor. CASSI's oracle is continuous (argmax), not discrete trial-and-error. CASSI's process reward bridge is the novel contribution |
| "SeqRoute's HBR is the same as your oracle labeling." | HBR is a general technique (HER, 2017). CASSI applies it to stopping with a novel quality-cost objective. Different problem structures (routing vs. stopping) |
| "Why not add a cost penalty to the executor's reward and skip the stopping model?" | Must show ablation: single-model GRPO with cost penalty vs. CASSI's two-model approach. Hypothesis: stopper provides specialized, cost-aware process rewards that a single model cannot generate |
| "When More is Less already shows optimal length exists. What's novel?" | They characterize the phenomenon; CASSI operationalizes it as a training algorithm. They don't define a per-trajectory oracle, train a stopping model, or use it for process rewards |
| "Why λ and not just a fixed budget?" | Must show Pareto frontier across multiple λ values. Must demonstrate that the optimal λ varies per task, making a fixed budget suboptimal |

---

## 7. What Must Be Included for ICLR Acceptance

### Absolutely Required (P0):
1. **CaRT as a direct baseline** — without this, the paper is incomplete
2. **Ablation: single-model GRPO with cost penalty vs. CASSI two-model** — proves the stopper adds value beyond "just add a cost penalty"
3. **Ablation: SFT-only stopper vs. SFT+GRPO stopper** — proves RL training of the stopper matters
4. **Ablation: stopper as inference-only controller vs. stopper as process reward for executor** — proves the process reward bridge matters
5. **H5 (stopping point correlates with difficulty, r > 0.5)** — this is the distinctive claim. If it fails, CASSI is just "train shorter" like When More is Less

### Strongly Recommended (P1):
6. **Comparison with BudgetThinker** — the closest single-model budget-aware RL approach
7. **Training wall-clock time comparison: CASSI vs. AgentPRM-cost** — proves the O(T) vs O(K×T²) advantage
8. **Human validation of oracle labels** — proves the oracle is reasonable
9. **Explicit positioning against the HBR/SeqRoute argument** — acknowledge the general technique, distinguish the specific application

### Nice to Have (P2):
10. **Stopper transfer across executor models** — shows the stopper is reusable
11. **Qualitative examples of coach behavior** — makes the paper compelling

---

## 8. Final Score and Recommendation

| Dimension | Score | Notes |
|---|---|---|
| **Problem importance** | 4.5/5 | Cost-aware agents are a critical practical problem |
| **Technical novelty** | 2.5/5 | Individual components are known; the combination + specific oracle formulation is new |
| **Empirical ambition** | 4.0/5 | 6 benchmarks, 12 baselines, 7 RQs, 6 ablations — comprehensive |
| **Clarity/elegance** | 4.0/5 | Clean 3-phase pipeline, intuitive motivation, simple oracle formula |
| **Overall** | **3.0/5** | Borderline ICLR accept |

### Recommendation: **Conditional Accept**

CASSI is a solid engineering contribution that solves a real problem. The individual pieces are not novel, but the specific combination — particularly the oracle objective `t* = argmax[quality − λ×cumulative_cost]` applied to both labeling AND process rewards — has not been done before.

**The paper will survive ICLR review IF:**
1. The empirical results are strong (20-40% cost reduction at iso-accuracy)
2. The ablation studies clearly show each component matters (especially the process reward bridge)
3. The paper includes CaRT as a direct baseline and beats it
4. H5 (correlation r > 0.5) is confirmed — this is the distinctive signal
5. The paper explicitly positions against the SeqRoute HBR / BudgetThinker / Ares axis

**The paper will be rejected IF:**
1. The results are weaker than expected (<20% cost reduction)
2. H5 fails (no correlation between stopping and difficulty)
3. The ablation shows the two-model architecture doesn't matter (single-model performs similarly)
4. The paper doesn't include CaRT as a baseline
5. A reviewer identifies an even closer prior art we missed

**Conservative reviewer prediction:** "This paper applies known techniques (GRPO, process rewards, post-hoc labeling) to a new problem (cost-aware stopping). The integration is clean and the results are strong, contributing a practical tool for efficient agent deployment. I vote weak accept."

**Aggressive reviewer prediction:** "The core idea — compute optimal stopping points from completed trajectories — is a straightforward application of hindsight relabeling (HER, 2017) to the stopping problem. The two-model architecture is from Ares. The budget-aware RL is from BudgetThinker. Nothing here is fundamentally new. I vote reject."

The AC's decision will likely depend on how well the authors handle the aggressive reviewer's concerns in rebuttal.

---

*Novelty assessment completed by 3 independent subagent reviewers, each reading papers in full. All 15 most threatening papers analyzed.*