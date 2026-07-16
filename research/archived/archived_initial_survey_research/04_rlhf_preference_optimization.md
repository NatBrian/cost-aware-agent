# Category 4: RLHF, Preference Optimization & RL Training for Agent Behaviors

Papers about training methods (RL, DPO, GRPO, etc.) for agent behaviors, process reward models, and credit assignment. These provide the algorithmic machinery needed to train both our executor agent and our monitor agent.

---

### GRPO: Group Relative Policy Optimization

- **Authors:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, et al. (DeepSeek-AI)
- **Year/Venue:** 2024, arXiv (foundational for DeepSeek-R1 and DeepSeek-V3)
- **URL:** https://arxiv.org/abs/2402.03300

**Background:** PPO requires training a separate value function model alongside the LLM, which is memory-intensive and unstable in the vast state space of language. RLHF training needed a simpler critic-free alternative.

**Motivation:** Reduce the computational and memory overhead of PPO while maintaining (or improving) training stability and performance for LLM alignment and reasoning tasks.

**Approach:** GRPO removes the critic model entirely. Instead of training a value function to estimate advantages, GRPO samples G responses for each prompt, computes a scalar reward for each (via a reward model or rule-based verifier), and normalizes these within the group to compute a relative advantage: A_i = (r_i - mean(r)) / std(r). The policy gradient then follows a clipped objective similar to PPO but with group-normalized advantages. A KL divergence penalty to a reference policy prevents catastrophic forgetting. The key insight is that for LLM reasoning, relative comparisons within a batch are sufficient to determine which outputs are better, eliminating the need for absolute value estimation.

**Training Method:** Online RL. For each prompt, sample G completions from current policy; compute rewards; compute group-normalized advantages; update policy via clipped surrogate objective with KL penalty to reference model. Subsequently adopted by DeepSeek-R1 for pure RL reasoning training.

**Key Results:** GSM8K: 82.9% → 88.2%, MATH: 46.8% → 51.7% with GRPO on DeepSeekMath-7B. DeepSeek-R1 showed that pure GRPO training could produce emergent chain-of-thought reasoning.

**Weakness:** GRPO assumes all steps in a trajectory contribute equally (uniform credit assignment), which fails in multi-turn agent settings. Performance depends on reward variance within groups — when all samples get the same reward, advantages collapse to zero.

**Relevance to Our Work:** **Foundational.** GRPO is the core RL algorithm used by most recent agent training work. For our proposal, GRPO (or its multi-turn variants like GiGPO) would be the natural choice for training both the monitor and the executor.

---

### GiGPO: Group-in-Group Policy Optimization for LLM Agent Training

- **Authors:** Lang Feng, Zhenghai Xue, Tongxuan Liu, Bo An
- **Year/Venue:** 2025, NeurIPS 2025
- **URL:** https://arxiv.org/abs/2505.10978

**Background:** GRPO has driven frontier LLMs in single-turn tasks like math reasoning but scales poorly to multi-turn agent training. In agent settings, interactions unfold over many steps with sparse/delayed rewards, making credit assignment across individual steps much harder.

**Motivation:** Standard GRPO assigns the same trajectory-level outcome reward to all steps in a rollout. This is fundamentally misaligned with the structure of multi-step agent interactions where some steps are critical and others are trivial.

**Approach:** GiGPO introduces a two-level hierarchical advantage estimation structure. At the **episode level**, it computes macro relative advantages based on groups of complete trajectories (identical to standard GRPO). At the **step level**, it introduces an "anchor state grouping mechanism" that retroactively constructs step-level groups by identifying repeated environment states across different trajectories. Actions stemming from the same environment state are grouped together, enabling micro relative advantage estimation. This hierarchical structure captures both global trajectory quality and local step effectiveness without relying on auxiliary value models or additional rollouts. The step-level advantage is computed by normalizing rewards of actions that share the same parent state, then combining macro and micro advantages for the final policy gradient.

**Training Method:** Online RL with GRPO-style group-based training. No critic model needed. Uses the same rollout data for step-level credit assignment via the anchor state mechanism.

**Key Results:** >12% improvement on ALFWorld and >9% on WebShop over standard GRPO. 42.1% accuracy on QA with 3B model, 47.2% with 7B model. All with same GPU overhead and no additional time cost.

**Weakness:** Requires environment states to be meaningfully comparable for anchor state grouping; environments with continuous/resource-variable states may not support clean state matching.

**Relevance to Our Work:** **Highly relevant.** GiGPO tackles the exact problem of fine-grained credit assignment in multi-step agent trajectories, which is directly applicable to training a monitor agent that needs to learn when to stop based on per-step cost/quality trade-offs.

---

### Training-Free Group Relative Policy Optimization

- **Authors:** Yusong Cai, Siqi Cai, Y. Shi, Zihan Xu, Lin-Zhi Chen, Yao Qin, et al.
- **Year/Venue:** 2025, arXiv preprint
- **URL:** https://arxiv.org/abs/2510.08191

**Background:** Full GRPO training is expensive for large models. Smaller fine-tuned models suffer from poor generalization and overfitting with limited data.

**Motivation:** Can we achieve GRPO-like policy improvement effects without any parameter updates?

**Approach:** Training-Free GRPO operates in multiple epochs with inference-only operations. In each epoch, the LLM generates multiple rollouts for each query. Instead of computing numerical advantages for gradient ascent, an LLM-based introspection step examines each group of rollouts and distills a "semantic advantage" — high-quality experiential knowledge expressed as natural language reflections and error corrections. This knowledge is accumulated across epochs and prepended as a token prior (learned context) during subsequent API calls. The approach preserves GRPO's multi-epoch structure but replaces gradient descent with iterative in-context knowledge accumulation.

**Training Method:** Inference-only, no parameter updates. Multi-epoch: generate rollouts → LLM introspects on relative quality → distill experiential knowledge → use as prefix for next epoch.

**Key Results:** AIME24: 68.6% → 72.6% (direct), 80.0% → 82.7% (ReAct+CI). AIME25: 52.9% → 54.0% (direct), 67.9% → 73.3% (ReAct+CI). Training cost ~$8-18 vs. thousands for full fine-tuning.

**Weakness:** Gains are modest (2-6% absolute); may not match full fine-tuning for large-scale tasks. Depends on LLM's introspection quality.

**Relevance to Our Work:** **Relevant.** Shows that group-relative comparison with LLM introspection can improve agent behavior. The cost-efficiency demonstration supports the feasibility of cost-aware agent training where expensive full RL may be unnecessary.

---

### Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning

- **Authors:** Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, Jiaya Jia
- **Year/Venue:** 2024, ICLR 2025
- **URL:** https://arxiv.org/abs/2406.18629

**Background:** Vanilla DPO has shown limited benefits for long-chain mathematical reasoning. Models using DPO struggle to identify specific errors in incorrect answers because DPO compares entire responses holistically.

**Motivation:** Can we adapt DPO to work at the level of individual reasoning steps rather than complete answers, enabling the model to pinpoint and fix specific errors in long reasoning chains?

**Approach:** Step-DPO shifts the optimization unit from entire responses to individual reasoning steps. Given a math problem and preceding correct reasoning steps, it trains the model to prefer a correct next step over an incorrect one. The dataset is constructed via a three-stage pipeline: (1) Error Collection — run the model on math problems and collect incorrect answers; (2) Step Localization — identify the first erroneous step in each incorrect chain; (3) Rectification — have the model generate a corrected step given the correct prefix. Crucially, the correct step is self-generated by the model (not GPT-4 or humans), as in-distribution data proved significantly more effective. Only 10K preference pairs and <500 training steps are needed.

**Training Method:** SFT first on math instruction data, then Step-DPO training with pairwise preference loss at the step level.

**Key Results:** Qwen2-72B-Instruct + Step-DPO: 70.8% on MATH, 94.0% on GSM8K. ~3% absolute gain over the base model. Surpassed GPT-4-1106, Claude-3-Opus, Gemini-1.5-Pro on MATH. Crucially, vanilla DPO showed minimal improvement on these benchmarks.

**Weakness:** Focused only on mathematical reasoning; step localization requires correct ground-truth answers; pipeline assumes errors can be localized to specific steps.

**Relevance to Our Work:** **Highly relevant.** Step-DPO demonstrates the importance of fine-grained preference optimization at the step level rather than full-trajectory level. For a cost-aware monitor agent, we need exactly this kind of step-level signal to learn when the cost of continuing outweighs the benefit.

---

### EntroPO: Building Coding Agents via Entropy-Enhanced Multi-Turn Preference Optimization

- **Authors:** Jiahao Yu, Cheng Zhang, Xiaolin Wu, Xinyu Xing
- **Year/Venue:** 2025, arXiv preprint
- **URL:** https://arxiv.org/abs/2509.12434

**Background:** Preference optimization (DPO, KTO) for LLMs reduces output diversity as it aligns to preferences, harming test-time scaling (TTS) which depends on diverse outputs. Existing preference optimization is single-turn and doesn't model multi-turn, tool-using agent trajectories.

**Motivation:** Can we extend preference optimization to multi-turn, tool-assisted settings while preserving output diversity essential for test-time scaling?

**Approach:** EntroPO extends entropy-regularized preference optimization from single-turn to multi-turn trajectories. It augments the standard DPO/KTO objective with an explicit entropy bonus term that preserves policy entropy during alignment. The framework provides a general recipe for adapting any preference optimization algorithm to multi-turn settings by optimizing over the full interaction sequence. The multi-turn formulation accounts for tool responses, environment observations, and intermediate reasoning steps. The authors also propose a hybrid best-trajectory selection scheme combining a learned verifier model with model-free approaches for test-time scaling.

**Training Method:** SFT on correct tool-using trajectories, then EntroPO fine-tuning with entropy regularization on multi-turn interactions.

**Key Results:** 30B EntroPO model ranks 1st on SWE-bench Lite and 4th on SWE-bench Verified among open-weight models, surpassed only by 350B+ models. The entropy bonus was critical to maintaining diversity.

**Weakness:** Requires trajectory-level preference pairs (correct vs. incorrect); relies on having successful trajectories for training; verifier model adds complexity.

**Relevance to Our Work:** **Directly relevant.** EntroPO demonstrates that multi-turn preference optimization with entropy preservation is effective for training agents. Our cost-aware stopping model similarly operates over multi-turn trajectories and needs to balance alignment (stopping at the right time) with preserving exploration diversity.

---

### CSO: Verified Critical Step Optimization for LLM Agents

- **Authors:** Mukai Li, Qingcheng Zeng, Tianqing Fang, Zhenwen Liang, Linfeng Song, Qi Liu, et al.
- **Year/Venue:** 2026, arXiv preprint (Tencent AI Lab, HKU)
- **URL:** https://arxiv.org/abs/2602.03412

**Background:** Prior agent post-training faces a trichotomy: (a) outcome-based trajectory optimization gives coarse rewards, (b) step-level process optimization introduces noisy estimated rewards, (c) Monte Carlo sampling for step rewards is computationally prohibitive.

**Motivation:** Can we focus preference learning only on verified critical steps — decision points where alternate actions demonstrably flip task outcomes — achieving precise credit assignment with minimal supervision?

**Approach:** CSO starts from failed policy trajectories (not expert demos), targeting the policy's actual weaknesses. A Process Reward Model (PRM) identifies candidate critical steps — pivotal decisions (tool selection, query formulation). An expert model proposes high-quality alternatives at these steps. The key innovation: these alternatives are verified by continuing execution from the alternative using the policy model itself until task completion. Only alternatives that the policy successfully executes to correct outcomes are used as DPO training data. This ensures both quality (expert proposed) and policy reachability (policy verified). Training uses pairwise DPO on verified critical step preference pairs. Only ~16% of trajectory steps receive supervision.

**Training Method:** Offline/semi-online: run policy to collect failed trajectories → PRM scores steps → expert proposes alternatives → verify by rolling out from alternative with policy → train with DPO on verified pairs.

**Key Results:** 37% and 26% relative improvement over SFT baseline on GAIA and XBench respectively. 8B model matches GPT-4.1 performance. Substantially outperforms all baseline post-training methods (ETO, IPR, AgentRPM, full-step DPO).

**Weakness:** Depends on PRM quality; expert model may introduce bias; verification rollout is expensive; focused on web search agents only.

**Relevance to Our Work:** **Highly relevant.** CSO's core idea of focusing optimization on critical decision steps aligns perfectly with our cost-aware stopping problem. The "stop or continue" decision is exactly the kind of critical step CSO targets. The verification-by-policy-execution concept could be adapted to verify whether stopping at a given point leads to acceptable outcomes.

---

### AgentPRM: Process Reward Models for LLM Agents

- **Authors:** Sanjiban Choudhury
- **Year/Venue:** 2025, arXiv preprint
- **URL:** https://arxiv.org/abs/2502.10325

**Background:** Training LLM agents to improve through interactions requires reward signals beyond sparse outcome feedback. Step-level process rewards can accelerate learning but are expensive to collect and design.

**Motivation:** Create a simple, scalable actor-critic framework for process reward modeling that integrates minimally with existing RLHF pipelines.

**Approach:** AgentPRM uses a lightweight actor-critic paradigm with Monte Carlo rollouts to compute reward targets. Two variants: **(1) AgentPRM:** The standard version where the PRM learns from rollouts of the current policy, computing Q-values as Monte Carlo returns from each state-action pair, then the policy is updated via RL using these step-level rewards. **(2) InversePRM:** Learns process rewards directly from expert demonstrations without any outcome supervision, by training the PRM to predict whether a given (state, action) pair appears in expert trajectories. The framework requires minimal modifications to existing RLHF infrastructure.

**Training Method:** Iterative actor-critic: policy rollout → train PRM via supervised learning on Monte Carlo targets → train policy via standard RL (PPO) using PRM as reward function.

**Key Results:** 3B models trained with AgentPRM and InversePRM outperform strong GPT-4o baselines on ALFWorld, despite the massive parameter disadvantage.

**Weakness:** Requires running many Monte Carlo rollouts for Q-value targets; InversePRM needs high-quality expert demos; tested only on ALFWorld; PRM may still suffer from reward hacking.

**Relevance to Our Work:** **Core relevance.** AgentPRM provides the exact architectural template for our proposed monitor agent as a reward model. The PRM framework (Q(s,a) for each state-action pair) directly maps to a cost-aware stopping model that evaluates Q(stop | trajectory) and Q(continue | trajectory).

---

### Agent-RRM: Exploring Reasoning Reward Model for Agents

- **Authors:** Kaixuan Fan, Kaituo Feng, Manyuan Zhang, Tianshuo Peng, Zhixun Li, Yilei Jiang, et al.
- **Year/Venue:** 2026, arXiv preprint
- **URL:** https://arxiv.org/abs/2601.22154

**Background:** Agentic RL typically relies on sparse outcome-based rewards that fail to differentiate intermediate reasoning quality. Existing scalar PRMs are susceptible to reward hacking and provide no language-based guidance for fixing errors.

**Motivation:** Can we create a reward model that produces structured, reasoning-aware feedback (not just scalar scores) for agentic trajectories?

**Approach:** Agent-RRM is a multi-faceted reward model that generates three outputs for each agent trajectory: (1) an explicit reasoning trace analyzing logical consistency; (2) a focused critique identifying specific flaws and providing refinement guidance; (3) a holistic quality score. Three integration strategies: **Reagent-C** (text-augmented refinement), **Reagent-R** (reward-augmented guidance), and **Reagent-U** (unified feedback integration — harmonizes multi-source rewards with critique-augmented sampling). The RM operates without access to ground-truth answers, relying on its own reasoning to identify flaws.

**Training Method:** Agent-RRM trained on curated trajectory data with reasoning annotations. Three agent training variants use DPO, SFT, or RL depending on feedback type.

**Key Results:** Reagent-U achieves 43.7% on GAIA and 46.2% on WebWalkerQA. The unified integration consistently outperforms text-only and reward-only variants. Textual critiques alone improve performance, confirming that language-based feedback provides complementary gains.

**Weakness:** Training the RM requires trajectory annotation data; RM may inherit biases; three integration variants add complexity; RM quality limits downstream agent performance.

**Relevance to Our Work:** **Directly relevant.** Agent-RRM provides a model for how a monitor agent could produce both a stopping score AND an explanation/critique for its decision. The three integration strategies (text-only, reward-only, unified) offer a roadmap for how to incorporate cost-awareness feedback into agent training.