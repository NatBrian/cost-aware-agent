# Category 5: Token-Efficient Reasoning & Budget-Aware Inference

Papers about making LLM reasoning more token-efficient, budget-constrained generation, and diminishing returns of test-time compute. These papers provide the empirical foundation for why token budgets matter and demonstrate that efficiency can be trained.

---

### Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters

- **Authors:** Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar
- **Year/Venue:** 2024, ICLR 2025
- **URL:** https://arxiv.org/abs/2408.03314

**Background:** Prior work had shown that scaling model parameters improves performance, but the complementary axis of scaling test-time computation remained underexplored. Existing test-time methods (best-of-N, majority voting) had shown only modest or negative returns.

**Motivation:** If an LLM can use a fixed but non-trivial amount of inference-time compute, how much can it improve its performance? This has deep implications for whether to spend compute on pretraining larger models vs. spending it at inference time.

**Approach:** Studies two primary mechanisms: **(1) Search against verifier reward models:** Using a process-based reward model (PRM) to guide best-of-N or beam search, where the PRM scores intermediate reasoning steps. **(2) Adaptive revision:** Iteratively updating the model's output distribution by allowing the model to sequentially revise its answer. The key innovation is a **compute-optimal scaling strategy** that adaptively selects per-prompt between sequential revision and parallel search based on the prompt's estimated difficulty. Easy prompts get single-shot generation; hard prompts get more search budget. Difficulty is estimated via the PRM's confidence on early samples. The optimal policy is determined offline by benchmarking different allocations on a development set and fitting cost-performance curves.

**Training Method:** No RL for the policy model. PaLM 2-S* base model. PRM trained via supervised learning on Math-Shepherd process-level annotations.

**Key Results:** 4× more efficient test-time compute scaling vs. best-of-N baseline. Small model with test-time compute can outperform a 14× larger model when the inference-to-pretraining token ratio is low. Per-prompt adaptive allocation is critical — strategy effectiveness varies significantly by prompt difficulty.

**Weakness:** Limited to math reasoning tasks. Difficulty prediction overhead excluded from compute accounting. PRM bias can skew results. Only studied with PaLM 2 family.

**Relevance to Our Work:** **Foundational.** This paper demonstrates that per-prompt adaptive allocation of inference compute is both necessary and highly effective. Our monitor agent extends this paradigm by providing a learned, generalizable stopping signal rather than relying on task-specific PRMs.

---

### Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference

- **Authors:** Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, Yiming Yang
- **Year/Venue:** 2024, ICLR 2025 (Poster)
- **URL:** https://arxiv.org/abs/2408.00724

**Background:** While training scaling laws were well-studied, optimal inference configurations remained unexplored. The community lacked systematic understanding of how inference strategies trade off cost and performance across model sizes.

**Motivation:** Understand inference scaling laws — how does LLM problem-solving performance improve with increased inference-time compute, and what is the optimal model size and strategy for a given budget?

**Approach:** Systematic empirical study across model sizes (Pythia family: 160M to 6.9B, plus Llemma-7B/34B, Mistral-7B), inference strategies (greedy search, majority voting, best-of-n, weighted voting), and search algorithms (two tree search variants including REBASE). Compute is controlled via FLOPs by varying generated tokens and sample count. A Llemma-34B reward model is finetuned on Math-Shepherd to output scalar step-level rewards. REBASE uses this reward model to guide tree search via learned backtracking. Key contribution: compute-optimal model size selection — for a given budget, smaller models are optimal at lower budgets.

**Training Method:** Reward model (Llemma-34B) finetuned on Math-Shepherd. Policy models finetuned on MetaMath via full-parameter SFT.

**Key Results:** Accuracy follows power-law improvement with inference compute but eventually plateaus — **diminishing returns observed universally**. Smaller models (Llemma-7B) can outperform larger ones (Llemma-34B) at same compute with advanced inference strategies. Optimal model size shifts toward smaller models as budget decreases. REBASE tree search achieves Pareto-optimal trade-offs.

**Weakness:** Only math reasoning; non-instruction-tuned models; FLOPs may not reflect real-world cost; reward model quality affects tree search.

**Relevance to Our Work:** **High.** Directly demonstrates diminishing returns in inference compute scaling — accuracy plateaus and more compute eventually stops helping. The finding that smaller models can match larger ones at fixed compute is highly relevant for cost-aware deployment.

---

### TALE: Token-Budget-Aware LLM Reasoning

- **Authors:** Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, Zhenyu Chen
- **Year/Venue:** 2024/2025, Findings of ACL 2025
- **URL:** https://arxiv.org/abs/2412.18547

**Background:** CoT reasoning significantly increases token costs (often 10-30× vs. direct answering). Prior work showed LLMs can follow length constraints in prompts, but no systematic method existed for choosing appropriate token budgets per problem.

**Motivation:** Can we dynamically estimate the right token budget per problem and compress CoT reasoning without sacrificing accuracy?

**Approach:** TALE introduces the concept of a token budget as a first-class object. Two phases: **(1) Budget Estimation:** Searches over budget candidates using a small validation set to find the minimal budget that maintains accuracy, then trains an estimator (linear model over problem embeddings) to predict this budget. **(2) Budget-Constrained Reasoning:** The estimated budget is included in the prompt as a constraint. Key finding: LLMs struggle with very small budgets (they fail to compress below a threshold) but can effectively compress when given a reasonable budget. Introduces the concept of **token elasticity** — how much can CoT be compressed at each budget level. A paradoxical finding: a budget of 10 tokens produces MORE output (157 tokens) than a budget of 50 tokens (86 tokens) because extremely tight budgets cause model confusion.

**Training Method:** No model training. Budget estimation uses a simple regression model on problem embeddings (from a sentence transformer). Prompt-based budget enforcement.

**Key Results:** Significant token cost reduction — for GPT-4o-mini on MathBench-College: token usage reduced from ~680 to ~240 tokens (65% reduction) with only 2% accuracy drop. Budget estimator achieves 60.61% in-range accuracy. LLMs sometimes fail to follow very small budgets.

**Weakness:** Prompt-based budget control is imprecise — LLMs don't reliably follow constraints. Budget estimator is a simple regression model. Only tested on math reasoning. Does not apply to models with native thinking modes.

**Relevance to Our Work:** **High.** TALE establishes that token budgets matter and that per-problem budget estimation is valuable. However, their prompt-based approach is fragile. Our proposal replaces prompt-based budget control with a learned monitor agent that observes the reasoning process and decides in real-time when to stop.

---

### SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning

- **Authors:** Zheng Li, Qingxiu Dong, Jingyuan Ma, Di Zhang, Kai Jia, Zhifang Sui
- **Year/Venue:** 2025, arXiv preprint (Peking University / ByteDance)
- **URL:** https://arxiv.org/abs/2505.11274

**Background:** Reasoning models like o1 and DeepSeek-R1 overthink even simple questions — e.g., QwQ-32B provides 13 solutions for "2+3=?" using 100× more tokens than non-reasoning models.

**Motivation:** Can a model learn to autonomously determine an appropriate token budget based on query difficulty, and generate responses of corresponding length?

**Approach:** Dual-phase training paradigm: **(1) Cold-Start SFT:** Model learns to first output an estimated token budget within `<budget>` tags, then generate the solution. The budget is learned from the optimal budget found through grid search on training data. **(2) Budget-Guided GRPO:** RL optimization using GRPO with a composite reward that includes: accuracy reward, length reward (penalizing exceeding budget), and budget estimation accuracy reward. During inference, users can either let the model self-estimate the budget or pre-fill a budget value for explicit control.

**Training Method:** SFT + GRPO. Composite reward: accuracy (binary) + length penalty + budget estimation accuracy bonus.

**Key Results:** 61% average response length compression for 1.5B model, 48% for 7B model, while maintaining near-unchanged accuracy. The model adaptively allocates more budget to harder problems. Budget estimation provides users with predictable wait times.

**Weakness:** Budget estimation may be inaccurate for out-of-distribution problems. Only evaluated on math reasoning. Requires training a specific model variant.

**Relevance to Our Work:** **High.** SelfBudgeter's approach — learning a budget estimator and using RL with length rewards — is closely aligned with our goal. Our monitor agent extends this by being a separate, modular component that can work with any base model and use richer signals for stopping decisions.

---

### BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens

- **Authors:** Hao Wen, Xinrui Wu, Yi Sun, Feifei Zhang, Liye Chen, Jie Wang, Yunxin Liu, Ya-Qin Zhang, Yuanchun Li
- **Year/Venue:** 2025, arXiv preprint (Tsinghua AIR)
- **URL:** https://arxiv.org/abs/2508.17196

**Background:** Existing budget control methods either fail to reliably enforce constraints (prompt-based), lack fine-grained control (thinking/non-thinking toggle), or struggle with strict adherence (SFT/RL approaches).

**Motivation:** How can we precisely control the length of model reasoning while optimizing for both accuracy and budget adherence?

**Approach:** BudgetThinker introduces **control tokens** — special tokens periodically inserted during reasoning that explicitly signal the remaining token budget (e.g., `[BUDGET: 300]`). This provides continuous budget awareness during generation. Two-stage training pipeline: **(1) SFT phase:** Training data is augmented with control tokens at regular intervals. **(2) Curriculum-based RL phase:** Uses a length-aware reward function that rewards correctness and penalizes deviation from target budget. The RL phase uses progressively tighter budgets (curriculum learning) to gradually improve budget adherence. 41k training examples across diverse reasoning datasets.

**Training Method:** SFT with control-token-augmented data, followed by curriculum RL with length-aware rewards. GRPO algorithm. Full parameter training on DeepSeek-R1-Distilled Qwen-2.5 (1.5B and 7B).

**Key Results:** 4.9% average accuracy improvement across all tested budgets compared to baselines. More precise budget adherence — the gap between generated length and target budget is significantly smaller. Control tokens lead to faster and more stable convergence during RL training. Maintains high performance even at very tight budgets (500 tokens).

**Weakness:** Requires modified model architecture/training with special tokens. Control tokens consume some of the budget. Limited to math benchmarks.

**Relevance to Our Work:** **Medium-High.** The idea of continuous budget signaling (control tokens) is complementary to our monitor agent approach. Our agent could observe these signals as part of its state, but instead of hard-coded budget tokens, it would use learned representations of reasoning progress and confidence.

---

### DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference

- **Authors:** Xiang Liu, Xuming Hu, Xiaowen Chu, Eunsol Choi
- **Year/Venue:** 2025, arXiv preprint (HKUST / NYU)
- **URL:** https://arxiv.org/abs/2510.19669

**Background:** LLMs generate uniformly long reasoning traces regardless of problem complexity. Easy problems waste tokens; hard problems may need more tokens than provided.

**Motivation:** Can we selectively apply different inference strategies per problem based on difficulty, achieving token savings without sacrificing accuracy?

**Approach:** DiffAdapt discovers a **U-shaped entropy pattern** in token-level probabilities during CoT: high entropy on easy problems (model is "overthinking" despite high accuracy — 22-25% entropy higher than medium), low entropy on medium difficulty, high entropy on hard problems (genuine uncertainty). Based on this, DiffAdapt creates three inference strategies (Easy/Normal/Hard), each with distinct prompts, temperatures, and max token lengths. The "Easy" prompt tells the model to answer succinctly; the "Hard" prompt encourages careful thinking. A lightweight probe classifier (linear on hidden states) selects the strategy per problem. No LLM fine-tuning needed.

**Training Method:** No LLM training. A small probe classifier is trained to predict difficulty category from model embeddings.

**Key Results:** Up to 22.4% token reduction while maintaining or improving accuracy. Oracle strategy selection achieves 50% token savings with 10% accuracy gain. U-shaped entropy pattern consistent across models — a fundamental signal for overthinking detection.

**Weakness:** Only three discrete difficulty tiers; probe classifier needs per-model training; may not capture very fine-grained difficulty variations.

**Relevance to Our Work:** **Medium.** The U-shaped entropy pattern is a potential feature for our monitor agent — it provides a training-free signal that distinguishes overthinking from genuine deliberation. Our approach could incorporate this as one of many signals the agent monitors.

---

### L1: Controlling How Long a Reasoning Model Thinks With Reinforcement Learning

- **Authors:** Pranjal Aggarwal, Sean Welleck
- **Year/Venue:** 2025, arXiv preprint (Carnegie Mellon University)
- **URL:** https://arxiv.org/abs/2503.04697

**Background:** Reasoning models can improve by "thinking longer" but their CoT length is not controllable — users cannot specify desired compute budgets. Prior methods like s1 attempted length control but were suboptimal.

**Motivation:** Users need fine-grained control over test-time compute allocation — trading off cost and accuracy smoothly. Existing methods cannot produce outputs that satisfy explicit user-specified length constraints.

**Approach:** Length Controlled Policy Optimization (LCPO) — an RL method optimizing for both accuracy and adherence to user-specified length constraints. The length constraint is given in the prompt (e.g., "think for at most 1000 tokens"). LCPO trains the model to respect this constraint while maximizing accuracy. Key finding: models trained with LCPO develop an unexpected **Short Reasoning Model (SRM)** capability — they can generate reasoning patterns similar to full-length models but at lengths comparable to non-reasoning models. At inference, the user specifies a length budget in the prompt and L1 generates a CoT respecting that budget. L1's 1.5B model surpasses GPT-4o at equal reasoning lengths.

**Training Method:** RL with LCPO. Reward function combines answer correctness with a length adherence penalty. Trains on length-constrained prompts. Ground-truth answer verification for correctness.

**Key Results:** L1 enables smooth trading of computational cost vs. accuracy across tasks. Outperforms s1 for length control. 1.5B L1 model surpasses GPT-4o at equal reasoning lengths. SRMs achieve reasoning-model quality at non-reasoning-model cost.

**Weakness:** Length control is approximate — exact token-level adherence not guaranteed. Only tested on reasoning benchmarks, not agent tool-use scenarios.

**Relevance to Our Work:** **High.** L1 shows that models can learn to respect explicit length budgets through RL — parallel to our monitor learning cost-aware stopping. The SRM discovery suggests efficiency and capability are not strongly opposed. L1's prompt-based budget specification contrasts with our approach of having a separate monitor make the stopping decision.

---

### Training Language Models to Reason Efficiently

- **Authors:** Daman Arora, Andrea Zanette
- **Year/Venue:** 2025, NeurIPS 2025 (Carnegie Mellon University)
- **URL:** https://arxiv.org/abs/2502.04463

**Background:** Large reasoning models produce very long chain-of-thoughts, incurring high deployment costs. Diminishing returns from scaling model size necessitates alternative methods for improving capabilities.

**Motivation:** Even for resource-rich organizations, excessive inference costs may mean operating at a loss. Can we train reasoning models to produce shorter CoTs while maintaining accuracy?

**Approach:** Uses policy gradient RL with a modified reward function: `R = 1{answer is correct} - α * (response_length / max_length)`. The α hyperparameter controls the efficiency-accuracy trade-off. The key insight is training with this reward for only ~100 RL steps (~200 gradient updates) — remarkably efficient. Starting from a reasoning model, this produces a family of models at different efficiency levels. The method requires only a couple of lines of changes to any standard RL implementation. The approach systematically navigates the token-accuracy Pareto frontier. No SFT pre-training needed beyond the base reasoning model.

**Training Method:** Policy gradient RL (REINFORCE-style). ~100 RL steps, ~200 gradient updates. Ground-truth scoring function (answer verification for math). No SFT pre-training needed.

**Key Results:** 7B model: 50% token reduction with <5% accuracy drop. 1.5B model: 65% token reduction on GSM8K with 1.7% accuracy drop. 16% token reduction on AIME 2024 with 3.3% accuracy drop. Training efficient reasoning is surprisingly cheap.

**Weakness:** Only tested on math reasoning. Linear length penalty may not be optimal for all tasks. Requires ground-truth answers for reward computation. The α parameter must be chosen empirically.

**Relevance to Our Work:** **Highest.** The simplicity of the RL reward for efficiency (correctness - α * length) directly informs reward design for our monitor agent. The finding that efficiency training requires very few RL steps suggests training a monitor could also be computationally cheap. The Pareto frontier navigation via α is analogous to our monitor learning optimal stop/continue decisions at different cost sensitivity levels.