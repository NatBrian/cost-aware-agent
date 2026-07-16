# Novelty Check — Agent 2: RL Methods Expert

> Reviewer persona: RL methods specialist (policy optimization, reward modeling, credit assignment).
> Target: `research/paper_plan.md` (CASSI, v5, read in full — 1576 lines). Date: 2026-07-16.
> Independence: did not read `research/archived_*`, other `lit_review/` files (except the sanctioned brief `00_paper_plan_summary.md`), or other `novelty_check/` files.
> Every paper named below was fetched: read as a local PDF in `research/papers/`, downloaded there by me, or verified via WebSearch/WebFetch of its arXiv page. No citation is from memory alone.

---

## My understanding of the proposal

CASSI trains LLM agents to be "economically rational" about when to stop. Stage 1: a ReAct-style executor (7B–72B) runs tasks to completion, recording per-step cost and per-step intermediate-answer quality (F1/accuracy vs. ground truth). Stage 2: for each completed trajectory, an "oracle" stopping point is computed post-hoc as `t* = argmax_t [quality_t − λ·cumcost_{1..t}]` (O(T), zero extra rollouts); steps before t* are labeled CONTINUE, after STOP, and a continuous margin `Δ_oracle(s_t) = Q_continue_oracle − Q_stop_oracle` is derived, where `Q_continue_oracle` is the best realized future net value *on that same trajectory*. A small stopping model (0.5B–3B) is trained on these labels via SFT + MSE value head + GRPO. Stage 3: the stopping model's Δ(s_t) is used as a dense per-step process reward (`R_t = α·Δ + β·progress + γ·format`, plus terminal outcome reward) to train the executor with GRPO (G=8). The stages can be iterated, which the plan brands "the self-reinforcing cycle" and claims as the primary contribution.

The five claimed contributions are: (1) the first closed loop oracle→stopper→process-rewards→executor-RL; (2) a "representation conflict" argument that a *separate* stopping model is structurally necessary; (3) an O(T)-vs-O(K×T²) efficiency claim against Monte-Carlo PRM training (AgentPRM), plus three "formal properties" of the oracle; (4) per-instance dynamic cost adaptation beating static length penalties; (5) a tiny stopper supervising a large executor at <3% overhead. From an RL standpoint, the proposal is an instance of generalized policy iteration with a learned, cost-augmented critic that is specialized to the binary stop/continue decision, trained on hindsight labels rather than bootstrapped or Monte-Carlo value targets.

---

## Core claims extracted (RL-technical framing)

- **C1 (cycle):** Iterated {rollout → fit critic on relabeled data → policy improvement against critic} with a cost-aware stopping critic is a novel training paradigm ("this cycle does not exist in any prior work").
- **C2 (separate critic necessity):** Policy and stopping-value objectives conflict at the representation level, so a single network cannot do both well; a separate small critic is *necessary*, with predicted strict Pareto ordering (two-model) > (multi-task) > (single-model penalty).
- **C3a (label efficiency):** Step-level training signals for PRMs require O(K×T²) extra executions (attributed to AgentPRM); CASSI's hindsight argmax label costs O(T) with zero extra executions — a ~160× reduction.
- **C3b (oracle soundness):** `t*` / `Δ_oracle` computed from a single realized trajectory is a valid supervised target for the stop/continue Q-gap.
- **C3c (formal grounding):** Uniqueness of t*, λ-monotonicity of t*(λ), and "oracle improves as policy improves" are meaningful formal properties distinguishing CASSI from empirical work.
- **C4 (adaptivity):** Static length penalties are instance-blind; a learned stopper provides per-instance, mid-trajectory adaptation (H5: stop step correlates with difficulty, r > 0.5).
- **C5 (asymmetric supervision):** A 0.5–3B model can supervise a 7–72B executor as a reward model with <3% inference overhead.
- **C6 (stability, implicit):** Training the executor via GRPO against a frozen-then-retrained learned Δ reward is stable and converges ("why the cycle converges", Section 4.3).

---

## Search log (queries + notable hits)

Tools: WebSearch (~14 queries), arXiv API curl (several; two returned empty due to encoding/no match), Semantic Scholar curl (rate-limited 429 — fell back to WebSearch), WebFetch (arXiv abs pages), plus full-text reads of 20+ PDFs in `research/papers/` (74 already cached; I added 3).

| # | Query (abbrev.) | Notable hits |
|---|---|---|
| 1 | hindsight optimal stopping label single trajectory LLM | **TERMINATOR (2603.12529)** "hindsight-optimal reasoning length"; Answer Convergence early stopping (2506.02536); hindsight trajectory rewriting (2510.10304) |
| 2 | learned stopping value as process reward train agent cost-aware | **AgentPRM-Fudan (2511.08325)**; QLASS (2502.02584); **CoRL (2511.02755)**; xRouter (2510.08439); CATP-LLM (2411.16313); ATLAS (2606.01667) |
| 3 | dichotomy of control / RCSL bias stochastic hindsight | **DoC (2210.13435, ICLR 2023)**; **Brandfonbrener et al. RCSL (NeurIPS 2022)** |
| 4 | implicit PRM free process rewards no extra rollouts | **Implicit PRM (2412.01981, ICML 2025)** — "additional process labels and extra rollouts are unnecessary" |
| 5 | GiGPO step-level advantages no rollouts critic-free | **GiGPO (2505.10978, NeurIPS 2025)** — anchor-state grouping, step advantages, "without auxiliary models or additional rollouts" |
| 6 | self-reinforcing loop policy↔reward model iterated | Self-Rewarding LMs (2401.10020); **Co-Evolution of Policy and Internal Reward (2604.03098)**; theory of iterative self-rewarding (2601.22513) |
| 7 | phasic policy gradient policy/value interference | **PPG (2009.04416, ICML 2021)** — interference exists, solved by phased distillation, not separation |
| 8 | reward model overoptimization Goodhart PRM hacking | **Gao et al. (2210.10760, ICML 2023)**; reward-hacking survey (2604.13602) |
| 9 | MRT progress reward test-time compute | **MRT (2503.07572)** — dense progress reward (Δ success prob.) for token efficiency |
| 10 | SWEET-RL step-wise critic asymmetric information | **SWEET-RL (2503.15478)** — separate critic w/ privileged training-time info gives per-turn rewards for agent RL |
| 11 | Longstaff-Schwartz continuation value foresight bias | LSMC neural extensions (2104.13669, 1907.06474) — continuation value must be a *conditional expectation regression*, not pathwise max |
| 12 | cost/budget-aware process reward model 2026 | Budget-Aware Agentic Routing (2602.21227); **BAGEN (2606.00198)** — cost estimation is a *distinct capability* from execution |
| 13 | adaptive length penalty difficulty-aware RL | **ALP (2506.05256, NeurIPS 2025)**; DLER/DA-DLER (2510.15110); AdaCtrl (2505.18822); LASER-D (2505.15612) |
| 14 | monotone comparative statics argmax penalty | Topkis/Milgrom-Shannon standard; "Comparative Statics for Optimal Stopping Problems" (2402.06999) |
| 15 | prophet inequality hindsight max vs online | Krengel–Sucheston 1/2-competitive bound; survey hits — online rules provably below hindsight max |
| 16 | Ng-Harada-Russell 1999; Setlur PAV | **Policy Invariance (ICML 1999)** verified; **Rewarding Progress/PAV (2410.08146)** — process reward should be an advantage, not a value |
| 17 | LLM self-evaluation interference same model | Self-[In]Correct (2404.04298); LLM evaluators favor own generations (2404.13076) |

PDFs read (local, `research/papers/`): 2502.10325 AgentPRM-Choudhury (method + Alg. 1 + reward-hacking section), 2511.08325 AgentPRM-Fudan (TD+GAE labeling), 2412.01981 ImplicitPRM, 2502.01456 PRIME, 2505.10978 GiGPO, 2505.20732 SPA-RL, 2510.08517 CaRT, 2603.12529 TERMINATOR, 2606.30852 When-Does-Learning-to-Stop-Help, 2605.23384 MaR, 2511.02755 CoRL, 2603.07915 Ares, 2605.25424 SeqRoute, 2510.01394 OptStop-vs-BoN, 2006.05082 Learning-to-Stop-While-Learning-to-Predict, 2602.16165 HiPER, 2605.05701 VOI-budget-control, 1207.5879 Selecting-Computations, 2412.21187 Do-NOT-Think-That-Much, 2312.08935 Math-Shepherd, 2504.14870 OTC-PO. Downloaded by me: 2503.07572 MRT, 2503.15478 SWEET-RL, 2604.03098 Co-Evolution.

---

## Closest prior work table

| Paper | Year | Venue | Overlap | Key Difference |
|---|---|---|---|---|
| AgentPRM (Choudhury, 2502.10325) | 2025 | preprint (Cornell) | **Iterative loop is identical in structure**: Alg. 1 = for i: rollout π_{i−1} → fit PRM Q_i → RL π_i. Also empirically documents PRM reward hacking | No cost term, no stopping decision; Q targets from shared-rollout MC averaging (NOT per-state restarts — see soundness Q2) |
| TERMINATOR (2603.12529) | 2026 | preprint | **Hindsight-optimal stopping labels from completed trajectories** ("first arrival of final answer") train a learned exit model; Pareto frontiers vs. DEER/Dynasor | Inference-time exit only; no executor training, no explicit λ·cost, single-turn CoT not tool agents |
| SPA-RL (2505.20732) | 2025 | NeurIPS | Trains a **progress estimator from completed trajectories** that redistributes final reward into per-step rewards used to RL-train the agent — same "learned step-reward → agent GRPO/PPO" bridge, O(T) labeling | Progress toward completion, not cost-quality stopping; no stop/continue head; no budget state |
| SWEET-RL (2503.15478) | 2025 | preprint (Meta) | Separate **step-wise critic with privileged training-time info** provides per-turn rewards for multi-turn agent RL (small critic supervises actor) | No cost/stopping objective; critic is advantage-style, avoiding CASSI's shaping bug |
| Implicit PRM (2412.01981) | 2025 | ICML | Process rewards **without extra rollouts or step labels** (log-ratio parameterization over outcome data) — kills the "PRMs need O(K×T²)" premise | Not cost-aware, not a stopping model |
| PRIME (2502.01456) | 2025 | preprint (widely cited) | **Online-updated implicit PRM inside RL** — dense step rewards, no dedicated PRM phase, explicitly to mitigate reward hacking | Same as above; also highlights the hacking risk CASSI ignores |
| AgentPRM (Fudan, 2511.08325) | 2026 | WWW | Agent PRM labels via **TD estimation + GAE**, "8× more compute-efficient than [MC] baselines", applied to agent RL | No cost/stopping; directly obsoletes the O(K×T²) framing for agent PRMs |
| GiGPO (2505.10978) | 2025 | NeurIPS | **Step-level advantages with zero extra rollouts** via anchor-state grouping, critic-free | No cost/stopping; advantage estimation not a learned monitor |
| CaRT (2510.08517) | 2025 | preprint (CMU) | Learned termination for information gathering; counterfactual pair labels + rationale | SFT of the *agent itself* (plan's claim "no executor training" is wrong — it fine-tunes the executor, just not via RL); no separate reward model |
| MRT (2503.07572) | 2025 | preprint (CMU) | **Dense progress rewards (Δ success probability) explicitly to optimize the compute-performance tradeoff** (cumulative regret over the thinking trace) | Single model; progress not net-value-of-continuing; math CoT not tool agents |
| OTC-PO (2504.14870) | 2025 | preprint | **Cost-aware agent RL**: reward scales correctness by tool-call efficiency toward per-question optimal (fewest observed) calls — group-hindsight optimal-cost labels | Single model, reward shaping only; no learned stopper, no Δ value |
| ALP (2506.05256) | 2025 | NeurIPS | **Per-instance adaptive length penalty** (scales with online solve rate) — refutes "penalties are instance-blind" as a blanket claim | Not mid-trajectory, not agentic, no learned monitor |
| Ares (2603.07915) | 2026 | preprint | Lightweight router trained on labels of **minimum reasoning effort sufficient per step** (hindsight-style minimal-compute labels) | Discrete effort levels, router controls executor at inference; no executor RL |
| Co-Evolution (2604.03098) | 2026 | preprint | **Co-evolving loop**: self-generated guidance reused as dense step-level internal reward for GRPO; "better policy → better guidance → better reward → better policy" | No cost/budget/stopping; internal (same-model) reward |
| Learning to Stop While Learning to Predict (2006.05082) | 2020 | ICML | **Oracle stopping distribution + imitation stage** to train a stopping policy jointly with a predictive model | Algorithmic depth setting; establishes oracle-label + imitation as an old recipe |
| SeqRoute (2605.25424) | 2026 | preprint | Hindsight **Budget** Relabeling + budget-in-state + λ-sweep Pareto navigation (offline RL) | Model routing across queries, not within-trajectory stopping |
| When Does Learning to Stop Help? (2606.30852) | 2026 | preprint | Controlled study: learned stoppers vs. calibrated scalar exits **at matched lost-correct risk with probe-overhead accounting**; learned stopping wins only in some regimes; hard benchmarks admit no aggressive stopping | Evaluation study; defines the protocol CASSI's <3%-overhead and savings claims will be judged against |
| Selecting Computations (1207.5879) / Rational metareasoning | 2012 | UAI | Δ(s) = value-of-computation (VOC): stop when expected improvement < cost — the *concept* behind Δ(s_t) is textbook metareasoning (Russell–Wefald line) | Bayesian/analytic, not learned from trajectories |

---

## Technical soundness analysis (main value-add)

### Q1. Is `t* = argmax_t [quality_t − λ·cumcost_t]` from a single trajectory sound as a Q-value target?

**No — as specified it is a "prophet" label, not a Q-target, and the plan does not recognize the dominant bias direction.** Three distinct problems:

1. **Non-measurability / foresight bias.** `Q_continue_oracle(s_t) = max_{t'>t} [quality_{t'} − λ·cost_{t+1..t'}]` is a function of the *realized future*, i.e., not measurable w.r.t. the information at step t. Training a regressor on these labels makes it approximate `E[max_{t'>t}(...) | x_t]` — the *expected pathwise maximum*. By the prophet-inequality literature (Krengel–Sucheston; surveys verified in search #15), the expected hindsight max strictly upper-bounds the value of **every implementable (non-anticipating) stopping rule**, with worst-case gap a factor of 2. The learned Δ therefore systematically **overestimates the value of continuing**, biasing the stopper toward late stopping — the exact failure mode the paper is trying to eliminate. This is precisely why Longstaff–Schwartz-style methods in optimal stopping regress *conditional expectations of the continuation value* rather than using pathwise maxima (search #11): pathwise-max targets are known to produce foresight-biased exercise policies. The plan's own caveat (§8.3) only discusses the *opposite* direction (trajectory suboptimality → labels too pessimistic) and never mentions the optimism from taking a max over the future.
2. **Stochasticity confound (hindsight-relabeling critique).** In stochastic environments (tool results, sampled generations), labeling a state by its realized future outcome credits luck. This is the formal core of "When does return-conditioned supervised learning work?" (Brandfonbrener et al., NeurIPS 2022) and Dichotomy of Control (Yang et al., ICLR 2023): hindsight-outcome-conditioned supervision yields policies inconsistent with achievable expectations in stochastic MDPs. For a prefix where 1 of 8 continuations finds the answer, CASSI labels CONTINUE on that trajectory and STOP on the others; the SFT cross-entropy average over G=8 partially smooths this, but the *value* target Δ remains a mean-of-maxima, and mean-of-max ≠ max-of-mean.
3. **The correct construction costs the same O(T).** The measurable analogue is the Snell envelope computed by backward recursion with cross-sectional regression over the collected trajectories: `V_T = q_T − λC_T; V_t = max(q_t − λC_t, E_hat[V_{t+1} | x_t])`, where `E_hat` is fitted by regression (fitted-Q / TD / Longstaff–Schwartz). Zero extra rollouts, same data, unbiased in the limit of a well-specified regressor. That the plan chose the pathwise argmax when the textbook-correct target is equally cheap is the single most damaging technical observation for an RL-expert reviewer: **the claimed innovation (O(T) labels) is real only in the trivial sense, and the specific label chosen is the biased variant of a known estimator family.**

**Has the exact trick been used?** Yes, in adjacent forms: TERMINATOR (2026) builds datasets of hindsight-optimal exit points ("first arrival of the final answer") from completed CoTs and trains an exit predictor; "Do NOT Think That Much" (2412.21187) uses first-correct-solution positions for efficiency preference pairs; Ares (2026) labels each step with the minimal sufficient reasoning effort discovered offline; SeqRoute (2026) does Hindsight Budget Relabeling; "Learning to Stop While Learning to Predict" (ICML 2020) trains a stopping policy from an oracle stopping distribution + imitation. The λ-parameterized quality-minus-cost argmax over an *agent* trajectory with multi-dimensional cost is an incremental generalization, not a new mechanism.

### Q2. Is the O(T) vs O(K×T²) comparison fair?

**No — it is wrong on the specific citation and obsolete as a general premise.**

- **Mis-attribution.** AgentPRM (Choudhury, 2502.10325), the named O(K×T²) baseline, does *not* run K fresh rollouts from every intermediate state. Its Eq. (1) computes `Q̂(s,a)` by averaging discounted returns over the *already-collected* rollouts that pass through the hashed pair (s,a) (multiple rollouts per task for state coverage). That is O(N·T) per task — the same order as CASSI's own data collection (which also needs G=8 trajectories per task for GRPO). The O(K×T²) figure describes Math-Shepherd-style per-step completion labeling (2312.08935 — verified: N completions launched from *each* step), which is a math-PRM annotation scheme, not the cited agent baseline. A reviewer who has read AgentPRM will flag the central complexity table as factually incorrect.
- **Obsolete premise.** Even granting the MC strawman, at least five published lines already produce step-level signals without per-state rollouts: Implicit PRM (ICML 2025: process rewards free from outcome-only training; "extra rollouts unnecessary" verified), PRIME (online implicit PRM inside RL), AgentPRM-Fudan (WWW 2026: TD+GAE labels, 8× compute efficiency, explicitly replacing MC estimation, applied to agent RL), GiGPO (NeurIPS 2025: anchor-state step advantages, zero extra rollouts), SPA-RL (NeurIPS 2025: learned per-step progress redistribution of the final reward), plus classical TD(λ)/GAE critics generally and HiPER's hierarchical advantage estimation (2026). The plan cites none of these except GiGPO-adjacent work indirectly. **The efficiency contribution reduces to: "for the specific stop/continue decision, a closed-form label exists" — true, minor, and preceded (Q1).**
- **Scope inflation.** The abstract says "training computation is reduced from O(K×T²) to O(T)". Only the *label-generation* step is O(T); stopper GRPO (G=8 responses per state over thousands of states) and executor GRPO (G=8 trajectories per task) dominate and are unchanged. At minimum the claim must be renamed "label-collection cost."

### Q3. Is the "self-reinforcing cycle" new?

**No — it is generalized policy iteration with a learned critic, and the named baseline already implements the loop.** AgentPRM's Algorithm 1 is literally `for i = 1..K: rollout π_{i−1} → train PRM Q_i → update π_i via RL against Q_i` with a conservative-policy-iteration justification. The plan's own §4.2 concedes AgentPRM has ②→③→④ but claims the cycle "does not exist in prior work" because prior work lacks the cost/stopping content — that is a content difference, not a cycle difference. Beyond AgentPRM: expert-iteration/ReST/Self-Rewarding LMs (2401.10020) all iterate policy↔evaluator, and Co-Evolution of Policy and Internal Reward (2604.03098, Jan 2026) explicitly markets the same loop ("better policy produces better guidance, better guidance further improves policy as internal reward") with step-level dense rewards under GRPO. An RL reviewer will call C1 standard.

Worse, the plan's *evidence* for the cycle is not evidence for a cycle: the "stopper-as-controller-only" ablation shows that *using the reward to train the executor matters* (arrow ④→⑤) — it cannot show the loop ①→⑤→① is "necessary," which would require ≥2 full iterations with per-iteration gains. The implementation plan (Step 6) lists iterative refinement as "*optional*". As written, the headline contribution is a diagram, not an experiment.

### Q4. Does the "representation conflict" argument hold?

**It is a legitimate empirical hypothesis stated as a theorem, and the plan contradicts it internally.**

- *Supporting evidence exists:* PPG (2009.04416) documents interference between policy and value objectives in shared networks; Self-[In]Correct (2404.04298) shows LLMs discriminate their own generations worse than they generate; BAGEN (2606.00198) finds cost-estimation and task-execution are distinct capabilities. The plan cites none of these — its best available support is missing.
- *Counter-evidence also exists:* PPG *solves* interference with phased distillation while keeping shared features (separation is not "structurally necessary"); CaRT shows a single fine-tuned model can learn termination; the whole adaptive-penalty line (ALP, AdaCtrl, BudgetThinker) trains single models with cost sensitivity.
- *Internal contradiction:* Contribution 1/5 and §4.3(5) claim the executor "internalizes cost-awareness" and "generates better actions even before the stopper intervenes" after training on Δ rewards — i.e., a single model demonstrably *can* represent both capabilities once given a dense signal. Then the conflict is about *training-signal engineering* (dense, privileged-information labels vs. sparse penalties), not representation capacity. SWEET-RL makes the honest version of this argument: the critic must be separate because it consumes *privileged training-time information* — which is also the true reason CASSI's stopper can be trained on ground-truth quality while the executor cannot see it at inference.
- *Pre-registered result:* Contribution 2 states the ablation "produce[s] strictly ordered Pareto frontiers (C) > (B) > (A)" in the indicative mood before any experiment. Reviewers notice this.
- The claim *is* testable as stated (the three-way ablation is well designed); it is the "structurally necessary / theoretical justification" framing that is indefensible.

### Q5. Stability: frozen-then-updated learned reward + GRPO

**Two documented failure modes apply, and one design bug is specific to CASSI.**

1. **Reward-model overoptimization.** Gao et al. (2210.10760) give the scaling laws; AgentPRM itself *measured* the agentic version: with a PRM trained on 10k rollouts, policy success fell 82%→70% while the PRM score kept rising, and their RM-ensemble detector failed. PRIME's stated motivation for online PRM updates is exactly that frozen PRMs get hacked. CASSI trains the executor against a frozen 0.5B–3B stopper whose inputs (confidence statements, answer drafts, stability indicators) are *generated by the executor being optimized* — a wide-open Goodhart channel (inflate stated confidence / stabilize the draft text to farm Δ). The risk table (§15) lists "GRPO training instability" but never reward hacking of the learned Δ. This is a major omission given the executor-training stage is the paper's centerpiece.
2. **Non-potential-based shaping bias.** The executor return includes `Σ_t α·Δ(s_t)`. Δ is a state-value-like quantity, not a potential difference (Ng, Harada & Russell, ICML 1999), so this changes the optimal policy — concretely: every pre-t* step contributes positive Δ, so trajectories that take *more* "promising" steps accumulate *more* reward. The dense term therefore pays the executor to dawdle in high-Δ states — an incentive directly opposed to cost reduction, only partially offset by the terminal cost-free outcome reward and negative Δ after t*. Two standard fixes exist: use potential-based shaping `γΦ(s_{t+1}) − Φ(s_t)`, or define the step reward as *progress/advantage* rather than value — which is exactly the finding of Rewarding Progress/PAV (2410.08146) and the design of MRT (2503.07572) and SWEET-RL. The plan shows no awareness of this literature.
3. **Nonstationarity of the iterated loop.** Retraining the stopper on the improved policy's trajectories moves the executor's reward landscape each iteration. Iterated RM retraining is manageable (standard RLHF practice, AgentPRM's conservative-policy-iteration note), but §4.3's "why the cycle converges" is a narrative, not an argument — no monotone-improvement condition, no trust-region coupling, nothing. Given the prophet bias (Q1) persists at every iteration, the loop's fixed point — if it exists — imitates a clairvoyant stopper on the current policy's paths, which is *not* the optimal online stopper for any policy.

### Q6. The three "formal properties"

- **Property 1 (existence/uniqueness of t*):** On a finite trajectory, existence of an argmax is trivial (finite set). Uniqueness is **false as stated**: with non-decreasing quality and strictly increasing cost, ties occur whenever a quality gain exactly offsets λ·Δcost, and plateau-jump-plateau quality profiles give multiple local maxima; nothing in the "proof sketch" (which only shows f eventually decreases) delivers uniqueness. Trivial-or-wrong.
- **Property 2 (λ-monotonicity):** Correct (with a smallest-argmax tie-break), and a direct instance of monotone comparative statics — the objective has decreasing differences in (t, λ) since cumcost_t is increasing — i.e., Topkis/Milgrom–Shannon; there is even a dedicated paper on comparative statics for optimal stopping (2402.06999). Fine as a one-line remark; not a contribution. SeqRoute already uses a λ-sweep for Pareto navigation in the budget setting.
- **Property 3 ("oracle convergence under policy improvement"):** Not a theorem. "Labels are always correct for the trajectory they label" is circular; expected-return improvement of π implies nothing about convergence of per-trajectory argmax labels to the optimal stopping rule of π (no metric, no statement, no proof — and Q1's foresight bias is policy-independent, so the labels do *not* converge to any implementable rule's values). An RL theory reviewer will treat §8.4 as padding, and the genuinely interesting question — what does regression-on-prophet-labels converge to, and how far is it from the Snell-optimal rule — is never posed.

---

## Per-claim novelty verdicts

| Claim | Verdict | Closest paper | Delta |
|---|---|---|---|
| C1 self-reinforcing cycle | **LOW** | AgentPRM (2502.10325) Alg. 1; Co-Evolution (2604.03098) | Cycle structure identical (generalized policy iteration); only the critic's content (cost-aware stopping) is new. Plan doesn't even commit to running >1 iteration |
| C2 separate stopper necessary | **LOW as theory / MEDIUM as ablation** | PPG (2009.04416); SWEET-RL (2503.15478); BAGEN (2606.00198) | No theory offered; internally contradicted by "executor internalizes cost-awareness"; the honest asymmetric-information version is SWEET-RL's. The 3-way ablation itself would be a useful empirical datapoint |
| C3a O(T) vs O(K×T²) | **LOW** | Implicit PRM (2412.01981); AgentPRM-Fudan (2511.08325); GiGPO; SPA-RL | Baseline complexity mis-attributed (AgentPRM uses shared-rollout MC, and Math-Shepherd is the real O(K×T²) method); rollout-free step signals already exist in ≥5 published forms |
| C3b oracle label soundness | **LOW (and unsound as specified)** | TERMINATOR (2603.12529); Do-NOT-Think (2412.21187); Ares (2603.07915); Learning-to-Stop (2006.05082) | Hindsight stop labels well precedented; CASSI's λ-cost generalization is incremental and carries an unacknowledged prophet/foresight bias; Snell-envelope regression at the same cost is the correct known alternative |
| C3c formal properties | **LOW** | Topkis/Milgrom–Shannon; 2402.06999 | P1 trivial-or-false, P2 standard comparative statics, P3 vacuous |
| C4 per-instance adaptivity | **MEDIUM-LOW** | ALP (2506.05256); AdaCtrl; DA-DLER; MRT | "Static penalty" strawman already false for 2025 reasoning literature; *mid-trajectory* adaptation in *tool agents with dollar/tool costs* is a genuinely less-occupied cell; H5 (difficulty–stop correlation) is a nice measurable |
| C5 small stopper supervises large executor, <3% overhead | **MEDIUM** | SWEET-RL; AgentRM (2502.18407); When-Does-Learning-to-Stop-Help (2606.30852) | Small-critic-supervises-large exists; the overhead claim must survive matched-risk + probe-overhead accounting (2606.30852 shows overhead can flip sign by serving regime) |
| Full combination (cost-aware stopping value → dense process reward → executor RL) | **MEDIUM** | SPA-RL + OTC-PO + TERMINATOR (jointly) | No single paper occupies this exact cell per my searches — but it is a recombination whose parts all exist, and its current technical execution (Q1, Q5 bugs) is below the bar |

---

## Overall assessment

**Score: 4/10** (novelty as currently framed; the underlying empirical program could reach ~5.5–6/10 if reframed and technically repaired).

**Recommendation: PROCEED WITH CAUTION — major reframing and two technical repairs required before any experiment is worth running.**

**Key differentiator (the real one):** nobody, per my searches, trains an executor agent with RL against a *learned, cost-aware stopping value* on long-horizon tool-use tasks, and the per-instance difficulty-adaptivity hypothesis (H5) with multi-dimensional costs (tokens/tools/dollars) on GAIA/SWE-bench is a defensible empirical niche. The paper's viable identity is an *empirical systems-RL paper about cost-aware credit assignment for agents*, not a new training paradigm, not a complexity result, not theory.

**Biggest risk:** an informed reviewer needs one read of AgentPRM Eq. (1) to see the O(K×T²) table is wrong, one read of Implicit PRM/PRIME to see the premise is obsolete, and one graduate optimal-stopping course to see the oracle is a foresight-biased prophet label. Any of the three sinks the current headline claims.

---

## Weaknesses & likely rejection reasons (ranked)

1. **Central efficiency claim is factually wrong about its own baseline.** AgentPRM does not restart K rollouts per state; O(K×T²) belongs to Math-Shepherd-style annotation. The 160× number will not survive review. (Fatal as written.)
2. **Oracle label is a non-measurable prophet target with unacknowledged optimism bias**; the measurable Snell/fitted-Q construction costs the same O(T). Reviewers from the RL-theory or quantitative-finance-adjacent pool will consider this a soundness flaw, not a nitpick.
3. **"First self-reinforcing cycle" is generalized policy iteration**, already instantiated by the cited baseline (AgentPRM) and by Co-Evolution (2026); the plan's own experiment design treats iteration as optional, so the headline contribution is unfalsifiable as planned.
4. **Reward-hacking exposure unaddressed**: executor is optimized against a frozen small RM whose input features the executor itself generates; AgentPRM already measured exactly this failure (82%→70%), PRIME exists precisely to fix it. No mitigation in the plan.
5. **Shaping bug**: `Σ_t α·Δ(s_t)` is non-potential-based, pays the agent to accumulate promising steps pre-t*, i.e., rewards the pathology it claims to cure. PAV/MRT already established that dense process rewards should be progress/advantages.
6. **Missing baselines/citations that reviewers will demand**: Implicit PRM, PRIME, AgentPRM-Fudan (TD+GAE), GiGPO, SPA-RL, SWEET-RL, MRT, OTC-PO, ALP/DA-DLER, PAV. The related-work section as drafted would read as a 2025-H1 snapshot.
7. **"Formal properties" are trivial (P1-existence), false (P1-uniqueness), standard (P2), or vacuous (P3)** — including them as a contribution invites theory-reviewer hostility.
8. **Evaluation-protocol risk**: 2606.30852 shows learned stoppers beat calibrated scalar exits only in specific regimes and that probe overhead can exceed savings under some serving regimes; CASSI's <3% overhead and 20–40% savings claims must be made under matched lost-correct risk with explicit serving-regime accounting, which the plan does not specify.
9. **CaRT mischaracterized** ("no executor training" — CaRT fine-tunes the executor via SFT; the correct statement is "no RL and no separate reward model"), plus text says "monitor errors are 16%..." in §27 as if measured — pre-experiment numbers in the anticipated-questions table look fabricated.

---

## Concrete improvement suggestions

1. **Fix the oracle (highest leverage).** Replace the pathwise argmax target with a backward-recursion regression target: `V_t = max(q_t − λC_t, E_hat[V_{t+1}|x_t])`, fitting `E_hat` across trajectories (fitted-Q / Longstaff–Schwartz over the G=8×tasks pool). Same O(T) per trajectory, measurable, and — bonus — the comparison "prophet-argmax labels vs. Snell-regression labels vs. MC labels" is itself a genuinely novel, cheap, publishable ablation nobody has run in the LLM-agent setting. Cite prophet inequalities to *quantify* the gap you eliminate; that would be real formal grounding, unlike §8.4.
2. **Re-scope the efficiency claim.** Delete O(K×T²)-vs-O(T) as a contribution. State honestly: "stopping labels are closed-form post-hoc; unlike implicit PRMs (which give quality-only signals from outcome labels) they encode an explicit, tunable λ cost-quality tradeoff." Add Implicit PRM/PRIME and AgentPRM-Fudan (TD+GAE) as baselines, not just AgentPRM-Feb-2025.
3. **Fix the reward plumbing.** Use Δ as potential-based shaping (`γΔ(s_{t+1}) − Δ(s_t)`) or convert to a progress/advantage form (PAV/MRT-style: change in predicted net value attributable to the step). Feed the stopper *verified* features (actual cost counters, measured answer-change) rather than executor-authored confidence text, or adversarially train against feature inflation. Add a reward-hacking probe (held-out true-success vs. Δ during executor RL) and PRIME-style online stopper updates as the mitigation.
4. **Make the cycle claim earn its name or drop it.** Commit to K≥2 full iterations with per-iteration metrics (label quality vs. human review, stopper accuracy, executor Pareto), or retitle the contribution "a learned cost-aware stopping critic for agent RL" and present iteration as standard practice (cite AgentPRM's conservative policy iteration).
5. **Reframe C2 as the asymmetric-information argument** (stopper is trained on privileged ground-truth quality; the executor never sees it), cite SWEET-RL/PPG/BAGEN, keep the 3-way ablation, and delete "structurally necessary"/pre-registered orderings.
6. **Adopt the 2606.30852 evaluation protocol**: matched lost-correct risk, probe-overhead accounting per serving regime, calibrated scalar-exit baselines (confidence/entropy/answer-convergence) in addition to the RL baselines — plus ALP/DA-DLER as the published adaptive-penalty baselines instead of only a self-constructed "Adaptive-α" variant.
7. **Tighten theory or cut it.** Keep P2 as a one-line remark with a Topkis citation. Replace P1/P3 with the one theorem worth having: a bound on the gap between the prophet-label policy and the Snell-optimal policy (or empirical estimate thereof).

---

## Is this strong enough for ICLR? (honest verdict)

**Not as written.** The three headline contributions (cycle, complexity reduction, formal properties) are respectively standard, factually wrong about the cited baseline, and trivial — and the core label construction has a soundness hole that RL-literate reviewers will find. Under the current framing I would expect scores clustering at 3–5 (reject) at ICLR 2027, with at least one reviewer producing the AgentPRM Eq.-(1) rebuttal and one producing the implicit-PRM/PRIME rebuttal.

There *is* a viable paper underneath: a carefully executed empirical study of **learned cost-aware stopping critics as process rewards for tool-using agents**, with the Snell-vs-prophet-vs-MC label ablation, hacking-aware training, matched-risk evaluation, and honest positioning as a recombination targeting an unoccupied cell (cost-aware + stopping + executor RL + long-horizon agents). That paper is publishable at ICLR *if the empirical wins are real and large* (SWE-bench-scale results at 20–40% cost reduction with iso-accuracy would carry it). But it is a different, humbler paper than the one this plan describes, and the plan should be revised before implementation effort is spent — the two technical repairs (oracle target, reward form) change what gets built, not just what gets written.
