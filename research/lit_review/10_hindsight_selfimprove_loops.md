# Hindsight Relabeling, Post-Hoc Oracle Labels & Self-Improvement Loops

> Area: prior art for CASSI's two framing pillars — (a) oracle stopping labels computed post-hoc
> from completed trajectories (a hindsight-relabeling move), and (b) the claimed "self-reinforcing
> cycle" (policy → trajectories → oracle labels → stopping/reward model → process rewards → policy).
> Researched 2026-07-16. All core papers read from downloaded PDFs in `research/papers/`.

## Area overview

Two literatures converge on CASSI's framing. The first is **hindsight relabeling**: after a
trajectory is complete, recompute labels/rewards/goals for its prefixes using information only
available at the end — at zero extra environment cost. Hindsight Experience Replay (HER, NeurIPS
2017) established the pattern in goal-conditioned RL; HIR (ICML 2023) ported it to LLM instruction
alignment; AgentHER (2026) ports it to LLM-agent goal relabeling. Critically for CASSI, 2026 work
has already specialized hindsight labels to *stopping*: TERMINATOR defines a "hindsight-optimal
reasoning length" (earliest arrival of the final answer in a completed CoT) and trains a small
probe on those labels to early-exit LRMs; CaRT (covered in the stopping-area review, but its
labeling is hindsight-flavored) labels episode prefixes "terminate" when stopping early would have
yielded higher success; SeqRoute retrospectively relabels recorded multi-turn routing logs under
counterfactual budgets, explicitly citing HER, and exposes a test-time λ-sweep over quality−λ·cost.
So the answer to critical question (i) is **yes**: post-hoc stopping/budget labeling from recorded
trajectories already exists in CoT early-exit (TERMINATOR), agent termination (CaRT), and budget
routing (SeqRoute) — although none of these computes CASSI's specific
`t* = argmax_t [quality_t − λ·cumcost_t]` margin from measured per-step answer quality, and none
feeds the labels back as a process reward to train the executor.

The second literature is **self-improvement loops in which the policy and the reward/judge signal
co-evolve**. The lineage is long and by 2025–2026 extremely crowded: Expert Iteration (2017,
planner↔network), STaR/ReST/ReST-EM (2022–2023, policy→filtered data→policy), Self-Rewarding and
Meta-Rewarding LMs (2024, the model judges and thereby retrains itself, with the *judge itself
improving across iterations*), SPIN (2024, self-play), Cooper and Mutual-Taught (2025, a *separate*
reward model co-updated with the policy inside the RL loop, motivated by RM staleness/reward
hacking), SPARK (2025, policy and generative reward unified in one model, claiming a "positive
co-evolving feedback loop"), R-Zero / Absolute Zero (2025, co-evolving proposer/solver from zero
data; R-Zero is ICLR 2026), and — closest to CASSI — Self-Guide (2026), which trains language
agents with GRPO on a composite of environment reward plus step-level internal reward derived from
the agent's own self-guidance, and *verbatim* claims "this creates a self-reinforcing loop: better
policy produces better guidance, and better guidance further improves policy." The answer to
critical question (ii) is therefore also **yes**: policy↔reward-model co-evolution is established
prior art, including for agents and including CASSI's exact "self-reinforcing" vocabulary. What
remains unclaimed in this literature is the *cost-aware, optimal-stopping* instantiation: no prior
loop trains a separate small stopping/value model on quality-minus-cost oracle labels and uses its
margin as a process reward for executor RL. CASSI's cycle framing is defensible only if narrowed to
that composition and if the cycle is actually demonstrated over ≥2 iterations (most co-evolution
papers run and measure multiple iterations; a one-shot pipeline would compare unfavorably).

## Core papers

### Hindsight Experience Replay (Andrychowicz, Wolski, Ray, Schneider, Fong, Welinder, McGrew, Tobin, Abbeel, Zaremba — NeurIPS 2017, arXiv 1707.01495)
- **Read from:** PDF pages 1–6
- **Problem:** RL with sparse binary rewards is nearly untrainable in large state spaces (robotics manipulation); reward shaping is brittle and domain-specific.
- **Method:** After each episode `s_0..s_T`, store every transition in the replay buffer not only with the original goal g but also with substituted goals (e.g., the goal actually achieved at the end, `m(s_T)`), recomputing the reward `r' = r(s_t, a_t, g')` post-hoc. Works with any off-policy algorithm (DQN, DDPG) on UVFA-style goal-conditioned policies. Zero extra environment interaction — labels are a pure O(T) recomputation over the recorded trajectory.
- **Training / RL usage:** Relabeled transitions are ordinary replay data; a form of implicit curriculum.
- **Experiments & benchmarks:** Bit-flipping toy environment; 7-DOF Fetch arm (pushing, sliding, pick-and-place) in MuJoCo with binary success rewards; sim-to-real deployment.
- **Key results:** Vanilla DQN solves bit-flipping only for n ≤ 13; DQN+HER solves n up to 50. Makes all three robotic tasks trainable from sparse binary rewards where vanilla DDPG fails; HER with sparse rewards even beats HER with shaped rewards.
- **Limitations:** Requires goal-parameterized tasks and off-policy replay; relabels *goals*, not decision-quality or cost.
- **Relation to CASSI:** The canonical citation for "compute labels post-hoc from completed trajectories at zero extra rollout cost" — CASSI's oracle labeling is an instance of this move applied to stopping. Different domain, no cost model, no reward-model bridge. THREAT LEVEL: LOW — foundational framing precedent, not competing prior art, but must be cited as the origin of pillar (a).

### The Wisdom of Hindsight Makes Language Models Better Instruction Followers — HIR (Zhang, Liu, Wong, Abbeel, Gonzalez — ICML 2023, arXiv 2302.05206)
- **Read from:** PDF pages 1–7
- **Problem:** RLHF-style alignment (PPO + reward/value networks) is complex; Final-Answer RL discards failure data.
- **Method:** Views instruction following as goal-conditioned RL (instruction = goal, LLM = policy and world model). Two phases: online sampling of instruction-output pairs at temperature 1; offline **hindsight instruction relabeling** — replace the instruction p with p* generated from a (scripted) feedback function so that it matches what the model actually produced, including *sub-output relabeling* at intermediate timesteps; then plain supervised seq2seq training plus a contrastive instruction loss and entropy regularization. Explicitly adopts HER's relabeling strategy; no reward or value networks, no extra parameters.
- **Training / RL usage:** Converts RL into iterated supervised learning on hindsight-relabeled data (RL-free loop of sample → relabel → fine-tune).
- **Experiments & benchmarks:** 12 BigBench reasoning tasks with FLAN-T5-base/large; baselines PPO and Final-Answer RL (FARL).
- **Key results:** Average 67.3% vs PPO 34.7%, FARL 56.1%, no-training 26.8% (FLAN-T5-large): +32.6 points over PPO, +11.2 over FARL. E.g., Tracking Shuffled Objects (3) 100.0 vs PPO 35.0; Date Understanding 98.0.
- **Limitations:** Scripted correctness feedback; single-step generation tasks (no multi-step agent, no tools); relabels goals, not stopping decisions; no cost notion.
- **Relation to CASSI:** Establishes hindsight relabeling *for LLMs* and the "turn RL into SFT on post-hoc labels" recipe that CASSI's stopper SFT stage follows. No cost, no stopping, no reward model. THREAT LEVEL: LOW — framing precedent to cite, orthogonal task.

### Self-Rewarding Language Models (Yuan, Pang, Cho, Li, Sukhbaatar, Xu, Weston — ICML 2024, arXiv 2401.10020)
- **Read from:** PDF pages 1–6
- **Problem:** Frozen reward models trained from fixed human preferences bottleneck alignment: the RM cannot improve as the policy improves.
- **Method:** One model both (i) follows instructions and (ii) judges its own candidate responses via LLM-as-a-Judge prompting (additive 5-point rubric). Iterative loop: model M_t generates new prompts, N=4 responses each, self-scores them, forms preference pairs (best vs worst), trains M_{t+1} with DPO. Both instruction-following *and judging ability* improve across iterations — the reward signal is explicitly not frozen.
- **Training / RL usage:** Iterative DPO (M1 → M2 → M3) on self-generated AIFT preference data (3,964 pairs for M2, 6,942 for M3).
- **Experiments & benchmarks:** Llama-2-70B seed (Open Assistant IFT+EFT); AlpacaEval 2.0, MT-Bench, 9 NLP benchmarks; reward-modeling correlation with human rankings.
- **Key results:** AlpacaEval 2.0 win rate over GPT-4-Turbo: 9.94% (iter 1) → 15.38% (iter 2) → 20.44% (iter 3), surpassing Claude 2, Gemini Pro, and GPT-4-0613. Head-to-head: M2 beats M1 55.5% vs 11.7%; M3 beats M2 47.7% vs 12.5%.
- **Limitations:** Same-model judge (reward hacking / saturation risk, later addressed by Meta-Rewarding); alignment domain; trajectory-level preference, no process rewards, no cost, no agents.
- **Relation to CASSI:** The canonical modern "policy → data → (self-)reward → policy" loop where the reward signal improves as the policy improves — exactly the abstract structure of CASSI's cycle claim. CASSI differs in every component (separate small model, oracle labels from measured quality/cost, process-level GRPO reward, agents), but "self-reinforcing cycle" as a concept is established here. THREAT LEVEL: MEDIUM — makes an *unqualified* "first self-reinforcing training cycle" claim indefensible; CASSI must scope the claim to cost-aware stopping.

### Cooper: Co-Optimizing Policy and Reward Models in Reinforcement Learning for Large Language Models (Hong, Yan, Wu, Hou, Zhang, Lu, Shen, Xiao — arXiv preprint 2508.05613, Aug 2025)
- **Read from:** PDF pages 1–7
- **Problem:** In RLVR/RLHF, rule-based rewards lack robustness (Math-Verify: 96% precision but only 63% recall) while fixed model-based rewards get hacked as the policy evolves.
- **Method:** Each training step has two stages: (1) GRPO policy update scored by a reference-based reward model (VerifyRM, 1.5B, trained on 58.7K hybrid rule+LLM-judge-labeled triples, 89.42% on VerifyBench); (2) **reward-model update** via contrastive loss on pairs mined from the policy's own fresh rollouts — positives selected by the high-precision rule verifier, negatives generated by an assistant LLM corrupting positives (verified incorrect). The RM thus tracks the moving policy distribution.
- **Training / RL usage:** Simultaneous, synchronized policy-GRPO + RM contrastive updates in one loop ("dynamically adjusting the parameters of the reward model during RL").
- **Experiments & benchmarks:** Math reasoning (Qwen2.5-1.5B-Instruct policy); VerifyBench for the RM.
- **Key results:** VerifyRM 89.42% on VerifyBench (best at its scale, vs xVerify-0.5B-I 70.68, Math-Verify rule 79.93); Cooper avoids the training collapse observed with a fixed RM and yields +0.54% average end-to-end accuracy over baselines.
- **Limitations:** Outcome-level correctness reward only (no process rewards, no cost, no stopping, no agents); small measured end-to-end gain; preprint.
- **Relation to CASSI:** Direct prior art for "reward model retrained from the evolving policy's own rollouts inside the RL loop" — the mechanism CASSI's cycle needs when it refreshes the stopper from new trajectories. Differences: correctness vs cost-aware stopping value; verification labels vs optimal-stopping oracle labels. THREAT LEVEL: MEDIUM — the co-optimization loop is prior art; also useful *supporting* citation for why the stopper must be re-trained across iterations (RM staleness/hacking).

### SPARK: Synergistic Policy And Reward Co-Evolving Framework (Liu, Zang, Ding, Cao, Dong, Duan, Lin, Wang — arXiv preprint 2509.22624, Sep 2025)
- **Read from:** PDF pages 1–6
- **Problem:** RLVR discards rollouts and correctness signals after each update; RLHF needs costly preference data and a separate RM that lags the policy (reward–policy mismatch, hacking).
- **Method:** During GRPO-style RLVR, **recycle** the rollouts and their verifier-assigned correctness into auxiliary training of the *same* model as a generative reward model: pointwise judgment samples, pairwise comparison samples, and reflection samples (fix an incorrect response conditioned on the judgment). One unified model = policy + reward + reflector; at test time it self-judges and self-refines (test-time scaling without external RM).
- **Training / RL usage:** Single model jointly optimized with verifiable-reward GRPO plus the recycled judgment/reflection objectives; the paper claims "a positive co-evolving feedback loop: improved reward accuracy yields better policy gradients, which produce higher-quality rollouts that further refine the reward model."
- **Experiments & benchmarks:** Qwen2.5 / Qwen2.5-VL backbones (7B, 32B); 7 reasoning benchmarks (MathVista, MathVision, WeMath, DynaMath, LogicVista, MMK12...), RewardBench2/VL-RewardBench, 8 general benchmarks.
- **Key results:** SPARK-VL-7B: +9.7% average on 7 reasoning benchmarks, +12.1% on 2 reward benchmarks, +1.5% on 8 general benchmarks; overall average 57.1 vs 46.5 for Qwen2.5-VL-7B (+10.6); beats GRPO policy-only and policy&reward variants.
- **Limitations:** Verifiable single-turn reasoning (not tool-using agents); judgment reward is correctness-based, not cost-aware; no stopping; unified model means no small-supervises-large economics; preprint.
- **Relation to CASSI:** Second strong precedent for policy↔reward co-evolution with feedback-loop language. Also *directly relevant to CASSI Claim 2* (representation conflict): SPARK finds that unifying policy and reward in ONE model produces positive synergy for correctness judging — CASSI's claim that execution and economic self-evaluation conflict in a single model must confront this opposite finding (different signal type is CASSI's escape hatch, and the single- vs two-model ablation becomes load-bearing). THREAT LEVEL: MEDIUM — cycle prior art + counter-evidence to the separation argument.

### Co-Evolution of Policy and Internal Reward for Language Agents — "Self-Guide" (Wang, Wu, Song, Zhang, Zhang, Kong, Kwok, Chang, Luo, Wu, Liu — arXiv preprint 2604.03098, Apr 2026, under review)
- **Read from:** PDF pages 1–8
- **Problem:** Long-horizon agent RL is bottlenecked by sparse, delayed rewards; external PRMs add overhead and drift from the policy's evolving rollout distribution; post-hoc credit assignment guides training but not inference.
- **Method:** At each step the agent generates a short verbal self-guidance signal z_t, then acts conditioned on it (inference-time steering). At training time the *same* z_t is mapped to a scalar step reward (polarity mapping: positive/neutral/negative → +0.1/0/−0.1), aggregated into R_sg(τ), and combined as R(τ;u) = R_env(τ) + λ(u)·Σ r_t^sg inside a single GRPO objective that jointly trains guiding and acting. A trapezoid **stage-wise trust schedule** λ(u) resolves the bootstrap (chicken-and-egg) problem: Phase I guidance-only (λ=0, steps <40), Phase II ramp 0→1 (40–50), Phase III full internal reward (50–70), Phase IV anneal 1→0 (70–80) because the shaping is not potential-based.
- **Training / RL usage:** GRPO (also verified with DAPO) on composite reward; the paper states verbatim that this "creates a self-reinforcing loop: a stronger policy generates more coherent trajectories, which produce more reliable self-guidance signals, which in turn yield more accurate internal rewards and further strengthen the policy," and shows the loop "in action" when SG&GR separates from SG-only at reward activation.
- **Experiments & benchmarks:** ALFWorld, ScienceWorld, WebShop; Qwen3-1.7B, Qwen3-4B, Qwen2.5-7B-Instruct; baselines ReAct, Reflexion, ReFlAct, GRPO.
- **Key results:** ~8% average improvement over GRPO with Qwen3-4B: ALFWorld 96.9 vs 86.7 (+10.2), ScienceWorld success 65.0 vs 59.3 (+5.7), WebShop success 78.1 vs 71.9 (+6.2). Qwen3-1.7B ALFWorld 89.8 vs 72.7 (+17.1). Ablations: immediate full internal reward (39.7) is *worse than no internal reward*; offline-distilled self-guidance does not transfer — "policy and internal reward must improve together online."
- **Limitations:** Internal (same-model) reward, coarse ±0.1 polarity signal; no cost/budget dimension; no stopping decision; no separate small supervisor; no inference-cost accounting.
- **Relation to CASSI:** The closest existing "self-reinforcing cycle" — language agents, GRPO, step-level reward derived from the system's own trajectory assessments, dual use at inference and training, identical loop vocabulary. CASSI's distinctions: signal = cost-aware stopping margin from *objective* oracle computation (not self-generated verbal judgment), a *separate* 0.5B–3B model, and cost/stopping as the target behavior. Self-Guide's trust-schedule and its "immature signal destabilizes training" finding are also methodological lessons CASSI's reward bridge must address. THREAT LEVEL: HIGH — a reviewer can cite this as an existing policy↔reward self-reinforcing GRPO loop for agents; CASSI's contribution 1 must be rescoped as the first *cost-aware / optimal-stopping* such loop.

### SeqRoute: Global Budget-Aware Sequential LLM Routing via Offline Reinforcement Learning (Xu, Zheng, Wang — arXiv preprint 2605.25424, May 2026)
- **Read from:** PDF pages 1–7
- **Problem:** LLM routers treat queries independently, ignoring session-level budgets → "budget bankruptcy" (resources exhausted before late hard queries).
- **Method:** Formulates multi-turn routing (weak 8B vs strong 70B) as a finite-horizon MDP whose state appends normalized remaining budget b_t to a 384-d MiniLM embedding. **Hindsight Budget Relabeling (HBR)** — "inspired by Hindsight Experience Replay" — retrospectively replays each recorded unconstrained trajectory under 5 hypothetical initial budgets {500...8000}, truncating where the counterfactual budget depletes and assigning bankruptcy penalty −η=−5; expands 10K sessions into 2.38M budget-annotated transitions at zero extra API cost, with a validity proposition (budget transition is deterministic bookkeeping, conditionally independent of conversation dynamics). Trains Discrete CQL (3-layer MLP); at deployment a **λ-sweep** a* = argmax_a [Q(s_t,a) − λ·c(a)] traces the cost–quality Pareto frontier zero-shot (Lagrangian duality).
- **Training / RL usage:** Offline conservative Q-learning on relabeled logs; no LM is trained; reward = ArmoRM quality − bankruptcy penalty.
- **Experiments & benchmarks:** 10K seed queries (ShareGPT/WildChat/Chatbot Arena), counterfactual tree rollouts with Llama-3.1-8B vs 70B-AWQ, 4-turn sessions; baselines Always-8B/70B, Random, budget heuristic, behavior cloning.
- **Key results:** CQL at λ=0 strictly dominates BC: cost 3275 vs 3482 (−6.0%), bankruptcy 24.3% vs 31.8% (−7.5pp); at λ=5×10⁻⁴ cost 921 with 0.3% bankruptcy (−73.5% cost vs BC); BR≤5% achievable at −67.6% cost; heuristic needs 4.2× the cost at equivalent safety. Micro-analysis shows learned "delayed gratification" (P(70B) slope +0.13 in remaining budget).
- **Limitations:** Discrete two-model routing, not stopping/continuation of a reasoning process; relabels *budget states*, not optimal stopping points from quality traces; tiny MLP Q-function, no reward model, no executor training, no loop; 4-turn horizon; preprint.
- **Relation to CASSI:** The clearest existing "hindsight relabeling for cost" — plus the same Lagrangian λ appearing as an explicit inference-time cost-quality knob (cf. CASSI's tiered λ multipliers). CASSI's paper-plan characterization of SeqRoute as "discrete budget routing only" is accurate. THREAT LEVEL: MEDIUM — pillar (a) is not novel in the cost/budget domain; CASSI keeps the distinct mechanism (quality−λ·cumcost argmax over steps) and the RL bridge.

### TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning (Nagle, Saydaliev, Garbaya, Gastpar, Makkuva, Kim — arXiv preprint 2603.12529, v2 May 2026)
- **Read from:** PDF pages 1–8
- **Problem:** LRMs overthink: they keep reasoning long after the eventual final answer has already appeared in the CoT; optimal CoT length is task- and model-dependent, so fixed thresholds/heuristics are suboptimal.
- **Method:** Defines **hindsight-optimal reasoning length (HORL)**: `HORL(x,r,s,â) = min{i : r_≤i contains the earliest logical arrival of â}` — "a retrospective property of the realized CoT and final answer," computed post-hoc from completed trajectories with **zero additional rollouts**. An LRM-based extract–identify–verify pipeline (Qwen3-30B-A3B) locates the exact token index i* at scale; token-level binary labels y_i = 1(i < i*) then train **a small separate stopping model** — a single transformer block (copied from the LRM's final layer) + prediction head — on the LRM's final-layer hidden states with class-weighted BCE. At inference, a sliding window of 10 token-level predictions with majority voting injects `</think>` to stop. Also shows answer arrival is marked by measurable token-confidence spikes and thinking-token frequency shifts.
- **Training / RL usage:** Supervised probe training only — the stopper is a pure inference-time controller; the LRM itself is never retrained; no RL, no reward model, no loop.
- **Experiments & benchmarks:** Qwen3-8B/14B, Ministral-3-8B/14B-Reasoning; train on AIME 1983–2024, MATH, OpenCoder-SFT, OpenScience; eval on MATH-500, AIME 2025, HumanEval, GPQA; baselines Vanilla, NoThinking, DEER, Dynasor, Thought Calibration.
- **Key results:** 14%–55% average CoT-length reduction across the four datasets; defines the accuracy-compression Pareto frontier on 14 of 16 (LRM, benchmark) pairs and is best/second-best on 28/32 metrics; >2× latency reduction (Qwen3-8B MATH-500: 14.10s vs 32.68s) with 10.8% (8B) / 7.5% (14B) compute overhead; empirically close to the (unachievable) HORL upper bound on all datasets.
- **Limitations:** Single-model CoT only (no tools, no agents, no multi-step environment); the label is "first arrival of the final answer" — a pure quality-plateau criterion with **no cost term and no λ tradeoff** (implicitly λ→0⁺: stop as soon as quality stops improving); no continuous value margin; probe requires hidden-state access (white-box); stopper never becomes a reward; no cycle; preprint.
- **Relation to CASSI:** The single most direct precedent for CASSI's oracle-labeling pillar: post-hoc *optimal-stopping* labels from completed trajectories, an O(T)-style zero-extra-rollout efficiency argument, and a small learned stopper trained on those labels used as an inference-time controller. CASSI's remaining deltas: explicit quality−λ·cumcost objective with budget tiers, agentic multi-step tool trajectories, a continuous Δ margin, and above all the process-reward bridge that trains the executor. THREAT LEVEL: HIGH — "hindsight-optimal stopping labels + small trained stopper" already exists; CASSI must cite it and lean on cost-awareness + executor RL + cycle as the contribution.

### Retrospective Progress-Aware Self-Refinement for LLM Agent Training — RePro (Ma, Zheng, Qiu, Hong, Yao, Qu, Yin, Lou, Wang, Liu, Zhang, Zhang, Zhao — arXiv preprint 2606.14302, Jun 2026)
- **Read from:** PDF pages 1–8
- **Problem:** RL-trained LLM agents optimize step-wise actions but lack metacognitive awareness of task progress; sparse outcome rewards give no intermediate credit; a pilot shows *online* progress prompting hurts (−8.6% average success on WebShop with DeepSeek-V4/GPT-5.1) while *retrospective* progress demonstrations anchored on known outcomes help (+7.9%).
- **Method:** **Forward-then-reflect**: the agent executes with online progress estimates p̃_t; after the trajectory completes and the outcome is known, the agent retrospectively re-assesses per-step progress p_1..p_T conditioned on the full trajectory and outcome (post-hoc labels — successful endings anchor 100%). Stage 1: Retrospection Warmup (SFT on DeepSeek-V4 retrospection demos). Stage 2: **RePro-PO** — composite step reward r_t = r_env + r_p + r_align + r_format where r_p = β(p_{t+1} − p_t) shapes progress deltas and r_align pulls online estimates toward retrospective quality; hierarchical episode+step advantages following GiGPO, PPO-style clipped objective. Progress estimation is deliberately *internalized* — "without relying on an additional reward model."
- **Training / RL usage:** Retrospective (post-hoc) per-step labels are converted into dense process-level rewards inside agent RL — structurally the same bridge as CASSI's "oracle labels → process rewards → executor training."
- **Experiments & benchmarks:** WebShop, ALFWorld, Sokoban; Qwen2.5-1.5B/3B/7B, Qwen3-4B; baselines GRPO, GiGPO, Meta-Prompt, L1/L2 reward-shaping variants.
- **Key results:** WebShop absolute success gains +8.98 / +11.57 / +5.82 pp over the Meta-Prompt baseline (best SR 81.64 / 83.59 / 84.38 for 1.5B/3B/7B); ALFWorld +11.72 pp (1.5B, best 99.22) and +4.69 pp (7B, best 100.0); Sokoban +3.12 pp (Qwen3-4B). Retrospective progress deltas correlate positively with step-level advantages in 91.5% of rollouts; intermediate discrimination (success vs failure) jumps from 1.86 to 6.37 (1.5B) and 1.11 to 31.62 (3B). Warmup-only crashes (−23 to −81 pp), showing outcome-grounded RL on the retrospective signal is essential.
- **Limitations:** Progress ≠ cost: no budget, no λ, no notion of overthinking (an agent can be at 100% progress and keep polishing); labels are the agent's own generated retrospections, not objective quality measurements; no stopping decision or controller; no separate model; preprint.
- **Relation to CASSI:** Closes "post-hoc trajectory labels → per-step process rewards → agent RL (GRPO-family)" — the exact reward-bridge pattern of CASSI steps 2+4 — and even shows the online/retrospective asymmetry CASSI's stopper implicitly exploits. CASSI's deltas: cost-aware optimal-stopping objective, oracle labels computed from measured quality/cost (objective, not self-generated), a separate small stopper reusable as an inference-time controller, and the closed cycle. THREAT LEVEL: HIGH — the reward-bridge novelty is substantially narrowed; only the cost/stopping content and the two-model architecture remain distinctive.

## Peripheral papers

**STaR: Bootstrapping Reasoning With Reasoning (Zelikman, Wu, Mu, Goodman — NeurIPS 2022, arXiv 2203.14465).** The template self-improvement loop: sample rationales, keep those yielding correct answers, fine-tune, repeat; for failures, "rationalization" generates a rationale *given the correct answer* — a hindsight move (condition on the outcome to create training data you couldn't produce forward). GPT-J with STaR reaches 72.5% on CommonsenseQA, comparable to a 30×-larger GPT-3 (73.0%). Policy→data→policy only — the filter (answer checking) never learns, so it is a fixed-oracle loop, not reward-model co-evolution. LOW threat; lineage citation.

**ReST (Gulcehre et al. — arXiv 2308.08998, 2023, preprint) and ReST-EM / Beyond Human Data (Singh et al. — TMLR 2024, arXiv 2312.06585).** Grow/Improve iterations: sample from the policy, filter/weight by a reward (ReST: learned MT reward; ReST-EM: binary correctness), fine-tune offline, repeat — cast as expectation-maximization. ReST-EM with PaLM-2 significantly surpasses human-data-only fine-tuning on MATH and APPS and scales favorably with model size. The reward/verifier is fixed across iterations — again half the CASSI cycle (policy evolves, judge does not). LOW-MEDIUM threat; establishes iterated self-training as standard.

**Expert Iteration (Anthony, Tian, Barber — NeurIPS 2017, arXiv 1705.08439).** Decomposes RL into planning (MCTS expert) and generalization (neural apprentice); the apprentice imitates the expert, and the improved apprentice strengthens the expert's search — the original policy↔evaluator mutual-improvement loop (contemporaneous with AlphaGo Zero). Tabula-rasa ExIt beats MoHex 1.0 at Hex and outperforms REINFORCE. LOW threat, but the honest ancestor of every "component A improves component B improves component A" claim.

**SPIN: Self-Play Fine-Tuning (Chen, Deng, Yuan, Ji, Gu — ICML 2024, arXiv 2401.01335).** Iterative self-play where the model discriminates its own previous-iteration generations from human SFT data (DPO-like objective); provably converges when the policy matches the target distribution. Strengthens zephyr-7B across HuggingFace Open LLM benchmarks without new human data. Policy-only loop, no reward model, no agents. LOW threat.

**Meta-Rewarding Language Models (Wu, Yuan, Dwivedi-Yu, Pang, Weston, Sukhbaatar et al. — arXiv preprint 2407.19594, 2024).** Extends Self-Rewarding with a meta-judge: the model judges its own *judgments*, so the reward signal itself is explicitly trained each iteration (addressing judge saturation). Llama-3-8B-Instruct AlpacaEval 2 win rate 22.9% → 39.4%, Arena-Hard 20.6% → 29.1%, without human labels. MEDIUM threat to the cycle framing: explicit evidence that making the *reward side* improve across iterations is known and beneficial.

**RISE: Recursive Introspection (Qu, Zhang, Garg, Kumar — NeurIPS 2024, arXiv 2407.18219).** Fine-tunes LLMs to improve their own answers over sequential attempts by casting single-turn problems as multi-turn MDPs and distilling improvement behavior from (self- or expert-generated) revision data; enables monotone 5-turn improvement on GSM8K/MATH for Llama2/Mistral where prompting-only self-correction fails. Self-improvement *at inference over turns*, not a policy↔reward training loop; no cost objective (it spends more compute to gain accuracy — the opposite tradeoff CASSI manages). LOW threat.

**Absolute Zero (Zhao et al. — arXiv 2505.03335, 2025) and R-Zero (Huang et al. — ICLR 2026, arXiv 2508.05004).** Zero-external-data self-play: a proposer/Challenger generates tasks at the edge of the solver's ability (rewarded for ~50% solver success) while the Solver is trained on them (pseudo-labels via majority vote / code execution). R-Zero boosts Qwen3-4B-Base +6.49 on math and +7.54 on general reasoning benchmarks. Two co-evolving roles with a closed data loop — further evidence that co-evolution loops are a crowded 2025–2026 genre (see also Multi-Agent Evolve, arXiv 2510.23595, Proposer/Solver/Judge from one LLM). MEDIUM threat to framing novelty, LOW to substance (no cost, no stopping, no PRM).

**AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling (arXiv preprint 2603.21357, 2026).** Adapts HER to natural-language agent trajectories: failed trajectories are relabeled as correct demonstrations for an achievable alternative goal via a four-stage pipeline (failure classification, outcome extraction, LLM-guided relabeling with confidence gating, packaging into SFT/DPO data). On WebArena and ToolBench, +7.6–11.4% over success-only SFT across four model families (GPT-4o, Qwen2.5-72B/7B, Llama-3.1-8B) with 2× sample efficiency. Occupies the "agent trajectory relabeling" name; relabels *goals*, not stopping points or costs; offline data augmentation, no reward model, no RL loop. MEDIUM threat to vocabulary/framing ("hindsight relabeling for agents" is taken), LOW to mechanism.

**Mutual-Taught (Shi et al. — arXiv 2506.06292, 2025) and other policy↔RM co-adaptation works.** Mutual-Taught runs EM-style alternation: E-step updates the policy under the current RM; M-step updates the RM from pseudo-preference pairs built from policy outputs before vs after the E-step — explicitly fixing RM distribution shift without human labels. Related contemporaries: Self-Evolved Reward Learning (arXiv 2411.00418), SPARK (core above), and 2026 entries that even use CASSI's vocabulary — EAPO (arXiv 2601.10306) advertises an "Adaptive Reward-Policy Co-Evolution mechanism that establishes a self-reinforcing loop" for long-context reasoning, and RLAnything / HCAPO (cited in Self-Guide's related work) jointly optimize policy+reward or refine step-level Q-values via *post-hoc hindsight critics*. Collectively MEDIUM-HIGH threat to the "first self-reinforcing cycle" wording: by 2026 "co-evolution/self-reinforcing loop" is a recognized genre with at least a dozen members; none is cost-aware or stopping-centric.

## Synthesis

**Landscape.** The two pillars CASSI stands on are both mature, and their 2026 frontier has moved
close to CASSI on separate flanks without occupying its intersection. Pillar (a), post-hoc labels
from completed trajectories: HER (2017) → HIR (2023, LLMs) → AgentHER (2026, LLM agents) for goal
relabeling; and now *stopping-specific* hindsight labels — TERMINATOR's hindsight-optimal exit
positions for CoT (small probe stopper, zero extra rollouts), CaRT's counterfactual
terminate-vs-continue prefix labels for agents, SeqRoute's counterfactual budget relabeling for
routing with an explicit quality−λ·cost inference rule, and RePro's outcome-anchored retrospective
per-step progress labels. Pillar (b), policy↔reward loops: ExIt → STaR/ReST/ReST-EM (fixed judge) →
Self-Rewarding/Meta-Rewarding (self-judge that improves) → Cooper/Mutual-Taught (separate RM
co-updated in-loop) → SPARK (unified model) → Self-Guide (agents, step-level internal reward inside
GRPO, verbatim "self-reinforcing loop") → R-Zero/Absolute Zero (co-evolving task proposer/solver).

**How established is each pillar?** (a) Hindsight/post-hoc labeling: fully established as a
general move, and *specifically for stopping* since 2026 (TERMINATOR; CaRT) — though never with
an explicit cost-per-step tradeoff objective computed from measured per-step quality
(TERMINATOR's label is cost-blind "first answer arrival"; CaRT's is success-based; SeqRoute's is
budget bookkeeping, not stopping quality). (b) Policy↔reward-model co-evolution: firmly
established prior art across alignment (Self-Rewarding, Meta-Rewarding, Mutual-Taught), RLVR
(Cooper, SPARK), agents (Self-Guide, RLAnything), and data generation (R-Zero, Absolute Zero).

**Is the "self-reinforcing cycle" framing defensible?** As stated ("first self-reinforcing
cost-aware training cycle"), only the qualifier "cost-aware" saves it, and the plan's current
argument leans too hard on the cycle itself being new. Recommended repositioning: (1) claim the
first cycle whose exchanged signal is an *economic* (cost-aware optimal-stopping) value rather
than correctness/preference/progress; (2) cite Self-Rewarding, Cooper, SPARK, Mutual-Taught, and
especially Self-Guide as cycle prior art and differentiate by signal type, separate small-model
architecture, and inference-time controller reuse; (3) actually *demonstrate* the cycle (≥2
iterations, measured oracle-label and stopper-accuracy improvement per iteration, as
Self-Rewarding and Meta-Rewarding do) — a single-pass pipeline would make the cycle claim
decorative; (4) engage Self-Guide's finding that immature internal rewards destabilize training
(trust scheduling) — CASSI's oracle labels being *objective* (computed from ground-truth quality
traces, not self-generated judgments) is a genuine advantage worth foregrounding here, since it
sidesteps both reward hacking (Cooper's motivation) and immature-signal instability (Self-Guide's).

**Gaps CASSI can own.** (1) No prior work computes optimal-stopping labels via
`argmax_t [quality_t − λ·cumcost_t]` with per-step measured quality and a tunable λ — TERMINATOR
is the λ→0 special case restricted to single-model CoT. (2) No prior work uses a stopping/value
model's margin Δ as a *process reward* to RL-train the executor — RePro is closest but its signal
is self-generated progress with no cost semantics and no separate model. (3) No cost-aware
policy↔reward co-evolution exists anywhere in the loop literature. (4) The small-stopper-supervises
-large-executor economics (<3% overhead) is unclaimed — TERMINATOR's probe is analogous at
inference but never supervises training.

**Top threats (ranked).**
1. **TERMINATOR (2603.12529)** — hindsight-optimal stopping labels from completed trajectories + a
   small trained stopper + the zero-extra-rollout efficiency argument already exist; HIGH.
2. **Self-Guide (2604.03098)** — an explicit "self-reinforcing loop" of policy and step-level
   internal reward, trained with GRPO, on agents; HIGH.
3. **RePro (2606.14302)** — retrospective post-hoc per-step labels turned into dense process
   rewards inside agent RL (GiGPO-family); HIGH.
4. **Cooper / SPARK / Mutual-Taught / Meta-Rewarding** — policy↔reward co-optimization loops are a
   genre, and SPARK's single-model synergy finding pressures the "representation conflict" claim; MEDIUM.
5. **SeqRoute (2605.25424)** — HER-inspired hindsight *budget* relabeling plus an inference-time
   λ·cost Lagrangian knob; MEDIUM.
6. **Self-Rewarding LMs (2401.10020)** — the canonical improving-judge loop that makes unqualified
   "first cycle" claims indefensible; MEDIUM.
7. **AgentHER (2603.21357)** — owns the "HER for LLM agent trajectories" branding; MEDIUM (naming),
   LOW (mechanism).

**Opportunities.** Frame CASSI's oracle step as "hindsight-optimal stopping under an explicit cost
model," citing HER/TERMINATOR/SeqRoute, and its loop as "cost-aware policy–stopper co-evolution,"
citing Self-Rewarding/Cooper/Self-Guide — then claim only the intersection. Borrow: trust
scheduling (Self-Guide) for when Δ enters the executor reward; RM-refresh cadence and
hacking-resistance arguments (Cooper); iteration-wise judge-quality metrics (Self-Rewarding,
Meta-Rewarding) to *measure* the cycle; SeqRoute's λ-sweep as a baseline/knob comparison; and the
online-vs-retrospective asymmetry evidence (RePro's pilot) as independent motivation for training
a dedicated stopper on retrospective labels instead of prompting for online self-assessment.
