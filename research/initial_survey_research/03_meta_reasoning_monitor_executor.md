# Category 3: Meta-Reasoning, Monitor-Executor Architectures & Metacognition

Papers about separate monitor/controller architectures, meta-reasoning, self-evaluation, and reflection loops. These provide architectural templates for our monitor-executor split and theoretical grounding from cognitive science.

---

### MGV: Monitor-Generate-Verify — Formalising Metacognitive Theory for Language Model Reasoning

- **Authors:** Nick Oh, Fernand Gobet
- **Year/Venue:** 2025, FoRLM Workshop (socius labs / London School of Economics)
- **URL:** https://arxiv.org/abs/2511.04341

**Background:** Test-time reasoning architectures follow the Generate-Verify paradigm where a model iteratively refines or verifies its own outputs. These architectures prioritize generation and verification but exclude the monitoring processes that determine when and how reasoning should begin. This omission may contribute to the "prefix dominance trap" where models commit early to suboptimal reasoning paths and seldom recover (~20% accuracy loss).

**Motivation:** What architectural vocabulary might help diagnose what Generate-Verify systems lack? Cognitive psychology offers a candidate answer: metacognitive monitoring.

**Approach:** MGV is a formal framework translating Flavell's and Nelson and Narens' metacognitive theories into computational form. The framework extends Generate-Verify by adding explicit **monitoring** that captures metacognitive experiences (difficulty assessments, confidence judgments) before generation begins and refines future monitoring through verification feedback. The framework is grounded in **resource-rational analysis**, formalizing metacognition as a meta-level MDP that derives optimal stopping policies from cost-benefit principles. The paper is primarily a position paper — it formalizes the framework but does not provide empirical validation. Key contributions include: (1) a taxonomy of metacognitive monitoring types (ease-of-processing judgments, feeling-of-knowing, judgment-of-learning, confidence judgments), (2) mapping these to specific LLM reasoning interventions, and (3) connections to optimal stopping theory.

**Training Method:** No empirical training — theoretical framework paper.

**Key Results:** No empirical results. The paper provides a formal vocabulary for diagnosing component-level failures in reasoning systems and suggests specific architectural interventions for future designs.

**Weakness:** Position paper — no empirical validation. The framework is high-level and lacks implementation details. The connection between psychological theory and LLM implementation is suggestive rather than proven.

**Relevance to Our Work:** **High.** MGV provides the theoretical and cognitive science grounding for our monitor-executor architecture. The MGV framework directly maps to our proposal: Monitor = our monitor agent, Generate = our executor agent, Verify = feedback loop. MGV's resource-rational analysis provides theoretical justification for why a separate monitor is necessary and how it should derive stopping policies from cost-benefit principles.

---

### ReMA: Learning to Meta-Think for LLMs with Multi-Agent Reinforcement Learning

- **Authors:** Ziyu Wan, Yunxiang Li, Yan Song, Hanjing Wang, Linyi Yang, Mark Schmidt, Jun Wang, Weinan Zhang, Shuyue Hu, Ying Wen
- **Year/Venue:** 2025, arXiv preprint (Shanghai Jiao Tong University / UBC / UCL)
- **URL:** https://arxiv.org/abs/2503.09501

**Background:** Recent research has sought to enhance LLM reasoning by integrating meta-thinking — enabling models to monitor, evaluate, and control their reasoning processes. However, single-agent RL attempts to learn meta-thinking and reasoning within a single forward pass, struggling to capture complex reasoning structures autoregressively.

**Motivation:** Construction-based supervised approaches (SFT on template reasoning) lack flexibility for OOD generalization. Single-agent RL (R1-style) needs strong base models and extensive fine-tuning. Can multi-agent RL provide a better framework for learning meta-thinking?

**Approach:** ReMA decouples the reasoning process into two hierarchical agents: **(1) High-level meta-thinking agent:** Responsible for generating strategic oversight and plans — it analyzes the problem, identifies the appropriate reasoning strategy, and produces meta-instructions. **(2) Low-level reasoning agent:** Tasked with detailed execution based on the meta-agent's guidance. The two agents are trained via **multi-agent RL** with aligned reward functions. Each agent is updated iteratively — the meta-agent learns to produce better strategies, and the reasoning agent learns to better execute them. The training uses REINFORCE/PPO with reward functions that include correctness, format adherence, and consistency rewards. The key insight is that distributing the exploration space across multiple agents enables each to explore more structurally and efficiently.

**Training Method:** Multi-agent RL with PPO/REINFORCE++. Aligned reward functions for both agents. Training on math reasoning and LLM-as-a-Judge benchmarks.

**Key Results:** ReMA outperforms single-agent RL baselines on competitive-level math and LLM-as-a-Judge benchmarks. The hierarchical design improves OOD generalization. Ablation studies reveal emergent role specialization — the meta-agent learns to identify problem types and select appropriate strategies. Under different reward settings, unexpected role reversal behaviors emerge.

**Weakness:** Multi-agent training adds complexity. The meta-agent's plans are high-level and may not capture fine-grained reasoning. Only tested on math and judge tasks.

**Relevance to Our Work:** **Very High.** ReMA is the closest architectural match to our proposal — a separate high-level agent (meta-thinker) guiding a low-level agent (reasoner), trained via RL. Our proposal extends this by: (a) making the high-level agent specifically a cost-aware monitor, (b) using the monitor as a reward model (not just a strategy provider), and (c) focusing on stopping decisions rather than general strategy selection.

---

### Deep Reasoning in General Purpose Agents via Structured Meta-Cognition

- **Authors:** (Multiple authors, 2025)
- **Year/Venue:** 2025, arXiv preprint
- **URL:** https://arxiv.org/abs/2605.11388

**Background:** General-purpose agents need to reason across diverse domains, but single reasoning strategies don't work universally. The agent needs to reason about which reasoning strategy to use — meta-cognition.

**Motivation:** Can structured meta-cognition enable general-purpose agents to adapt their reasoning strategies dynamically based on task characteristics?

**Approach:** A framework for structured meta-cognition in general-purpose agents. The agent maintains a **meta-cognitive state** that tracks: (1) task characteristics (domain, difficulty, required precision), (2) available reasoning strategies (CoT, ToT, retrieval, tool use), (3) strategy performance history. At each decision point, the meta-cognitive module selects the most appropriate strategy based on the current state. The module is trained via RL with rewards that combine task success with efficiency penalties. The meta-cognitive state is structured — it uses explicit representations of task features and strategy properties rather than implicit learned embeddings.

**Training Method:** RL with combined task success + efficiency reward. The meta-cognitive module is trained as a separate policy that selects reasoning strategies.

**Key Results:** Structured meta-cognition enables better strategy selection than fixed strategies. The agent adapts to different task types effectively. Efficiency penalties produce more cost-effective strategy selection.

**Weakness:** The structured meta-cognitive state requires feature engineering per domain. Strategy selection is at the coarse level (which strategy to use) rather than fine-grained step-level decisions.

**Relevance to Our Work:** **Medium.** This paper applies meta-cognition to strategy selection, while we apply it to stopping decisions. The structured meta-cognitive state approach could be adapted for our monitor to track reasoning progress and cost. The efficiency-penalized RL training is directly applicable.

---

### Re-ReST: Reflection-Reinforced Self-Training for Language Agents

- **Authors:** (Multiple authors, 2024)
- **Year/Venue:** 2024, arXiv preprint
- **URL:** https://arxiv.org/abs/2406.01495

**Background:** Language agents benefit from reflection — analyzing their own outputs to identify errors and improve. However, reflection is typically done at inference time by the same model, which is limited by the model's own blind spots.

**Motivation:** Can we train a separate reflector model that provides better feedback than self-reflection, and use it to improve the language agent through iterative self-training?

**Approach:** Re-ReST has two components: **(1) Language Agent:** The primary agent that performs the task (e.g., multi-hop reasoning, web navigation). **(2) Reflector:** A separate model trained to analyze the agent's trajectories and provide reflection feedback — identifying errors, suggesting alternative approaches, and evaluating the quality of intermediate steps. The training process: (a) Initial generation — the agent generates trajectories; (b) Reflection with environmental feedback — the reflector analyzes trajectories, identifying failures and successes; (c) Model training — the agent is fine-tuned on successful trajectories and reflector-improved trajectories; the reflector is trained on (trajectory, reflection) pairs. This creates an iterative self-training loop: better agent → better trajectories → better reflector → better agent.

**Training Method:** Iterative SFT: agent trained on successful/improved trajectories; reflector trained on (trajectory, reflection) pairs generated by a stronger model. No RL.

**Key Results:** Iterative self-training with a separate reflector improves agent performance over standard self-training. The reflector provides more diverse and useful feedback than self-reflection. The approach works across multi-hop reasoning and web navigation tasks.

**Weakness:** SFT-based training — no RL optimization. The reflector's quality is bounded by the teacher model used to generate reflection data. The iterative loop requires careful data filtering.

**Relevance to Our Work:** **Medium.** Re-ReST's separate reflector model is architecturally similar to our monitor agent. The key difference: the reflector provides general feedback, while our monitor provides cost-aware stopping signals. The iterative training loop (agent → reflector → agent) is a template for our monitor-agent training dynamic.

---

### Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training

- **Authors:** (Multiple authors, 2025)
- **Year/Venue:** 2025, arXiv preprint
- **URL:** https://arxiv.org/abs/2501.11425

**Background:** Language agents make errors in multi-step tasks. Reflection — analyzing past actions to correct course — can improve performance, but current agents either don't reflect or reflect poorly.

**Motivation:** Can we train agents to reflect effectively using MCTS-guided trajectory generation and iterative self-training?

**Approach:** Two-phase approach: **(1) Phase I: Model-Guided Reflection Trajectory Generation:** Uses Monte Carlo Tree Search (MCTS) to explore action spaces and identify "transition points" — moments where the agent should pause and reflect on its progress. At each transition point, the agent generates a reflection (analysis of what went well/poorly, what to do next) and continues from the reflected state. MCTS explores multiple reflection strategies to find the most effective ones. **(2) Phase II: Iterative Self-Training with Revision Trajectories:** The agent is trained on the MCTS-generated reflection trajectories. Successful trajectories (those that led to correct outcomes) are used for SFT; the agent learns to generate both actions and reflections. The training is iterative — the improved agent generates new trajectories, which are fed back into MCTS for further improvement.

**Training Method:** SFT on MCTS-generated reflection trajectories. Iterative self-training: agent → MCTS → improved trajectories → agent. No RL for the agent itself.

**Key Results:** Agent-R improves performance on interactive and agentic environments. The MCTS-guided reflection trajectories are more effective than random or heuristic reflection. The iterative training loop produces consistent improvements.

**Weakness:** MCTS is computationally expensive. The transition point identification is heuristic. Reflection quality depends on the MCTS exploration budget.

**Relevance to Our Work:** **Medium.** Agent-R's "transition points" (moments to pause and reflect) are analogous to our monitor's "stopping points" (moments to evaluate whether to continue). The MCTS-based exploration of reflection strategies could inform how we train our monitor to make stopping decisions.

---

### SELF: Self-Evolution with Language Feedback

- **Authors:** (Multiple authors, 2023)
- **Year/Venue:** 2023, arXiv preprint
- **URL:** https://arxiv.org/abs/2310.00533

**Background:** LLMs can improve through self-play or self-training, but typically rely on scalar rewards or binary feedback. Language feedback — natural language critiques and suggestions — is richer but harder to operationalize for training.

**Motivation:** Can LLMs self-evolve using language feedback as the training signal, learning to improve their outputs through iterative refinement guided by natural language critiques?

**Approach:** SELF has two stages: **(1) Meta-Skill Learning:** The model is trained to perform "meta-skills" — generating critiques, identifying errors, and suggesting improvements. This is done via SFT on a corpus of (input, output, critique, improvement) examples. **(2) Self-Evolution Training:** The model generates initial outputs, critiques them (using the meta-skills), refines them based on the critique, and is trained on the refined outputs. The self-evolution loop runs iteratively: the model's own critiques guide its improvement. The training objective includes both task performance and critique quality. At inference time, the model can optionally perform critique-then-refine to improve its answers.

**Training Method:** SFT on meta-skill corpus, then iterative self-training with language feedback. No RL.

**Key Results:** SELF-trained models show improvement over base models on multiple tasks. The language feedback provides richer training signals than scalar rewards. The meta-skill learning transfers across tasks.

**Weakness:** Self-generated critiques can reinforce errors. The meta-skill corpus requires careful curation. Iterative self-training can lead to mode collapse.

**Relevance to Our Work:** **Medium.** SELF's concept of using language feedback as a training signal is relevant to how our monitor agent could provide rich, interpretable feedback beyond just a scalar stopping score. The monitor could generate critiques like "The agent has gathered sufficient evidence; continuing would only add redundant information."

---

### PACER: A Single Revision Step Improves Token-Efficient LLM Reasoning

- **Authors:** Yingchuan Zhang, Terry Ma, Wenxuan Zhong, Ping Ma
- **Year/Venue:** 2026, arXiv preprint
- **URL:** https://arxiv.org/abs/2602.02828

**Background:** Majority voting (self-consistency) aggregates multiple reasoning traces but treats each in isolation — traces that are confidently wrong cannot be corrected. Confidence-guided early stopping (DeepConf) reduces cost but evaluates traces independently.

**Motivation:** There's a "coordination gap" at test time: self-consistency benefits from diversity but costs N× tokens; early stopping reduces cost but treats traces independently. Can we introduce a minimal coordination step that lets traces "peer-review" each other?

**Approach:** PACER is a training-free, three-stage coordination layer. **(1) Generate:** Sample N reasoning traces under token budget, using confidence-guided early stopping (DeepConf-Online) to screen unreliable traces. **(2) Summarize:** Build a compact "consensus packet" containing unique candidate answers, aggregated confidence scores, and short representative reasoning summaries per answer. This is low-bandwidth — the packet provides set-level evidence without exposing all full traces. **(3) Revise:** Each trace performs a brief self-review conditioned on the packet and optionally revises its final answer. The final prediction is a confidence-weighted vote over revised answers. The key insight: traces that are confidently wrong in isolation can self-correct when shown that alternative answers exist with coherent rationales.

**Training Method:** Training-free — purely inference-time coordination. Uses DeepConf-Online as the underlying sampling mechanism.

**Key Results:** PACER consistently dominates the token-accuracy Pareto frontier. Matches or exceeds accuracy of 256-sample majority voting. Significantly outperforms raw ensemble baselines. The single revision step transforms simple consensus into collaborative logical refinement.

**Weakness:** Only tested on math benchmarks. The consensus packet construction relies on answer extraction which can fail for open-ended answers. The revision step adds latency.

**Relevance to Our Work:** **Medium.** PACER's peer-review mechanism is conceptually related to having a monitor agent evaluate the primary agent's output. The "consensus packet" is a lightweight summary — analogous to our monitor providing a compact cost-benefit assessment. PACER shows that a separate evaluation step (beyond self-evaluation) can significantly improve efficiency.

---

### Budget Guidance: Steering LLM Thinking with Budget Guidance

- **Authors:** Junyan Li, Wenshuo Zhao, Yihao Zhang, Chuang Gan
- **Year/Venue:** 2025, arXiv preprint (UMass Amherst / MIT-IBM Watson AI Lab)
- **URL:** https://arxiv.org/abs/2506.13752

**Background:** Deep-thinking LLMs reason extensively to improve performance but lengthy reasoning incurs excessive costs with disproportionate gains. Controlling reasoning length without sacrificing performance is challenging, especially under tight budgets.

**Motivation:** Existing length-control methods (L1, TALE) require training. Can we achieve budget control at inference time without any fine-tuning?

**Approach:** Budget Guidance is a training-free inference-time method. It introduces a lightweight predictor that models a **Gamma distribution over remaining thinking length** during next-token generation. At each token generation step, the predictor estimates the probability distribution of the remaining CoT length. This signal is used to guide generation in a soft, token-level manner — when the predicted remaining length exceeds the budget, token probabilities are adjusted to favor more concise continuations. The guidance strength is modulated by how much the predicted length exceeds the budget. The predictor itself is small (a few MLP layers) and runs alongside the LLM without modifying its weights. The Gamma distribution is chosen because it naturally models positive continuous quantities (remaining length) and is flexible enough to capture different thinking patterns.

**Training Method:** The length predictor is trained separately (supervised on CoT length data). The LLM itself is not fine-tuned — pure inference-time guidance.

**Key Results:** Up to 26% accuracy gain on MATH-500 under tight budgets vs. baselines. Maintains competitive accuracy with only 63% of thinking tokens used by full-thinking model. The predictor's estimated remaining length correlates with question difficulty — harder questions get longer estimated lengths.

**Weakness:** The length predictor requires training data from the target LLM. Token-level guidance can slow down generation. Gamma distribution assumption may not fit all reasoning patterns. Only tested on text reasoning.

**Relevance to Our Work:** **Medium.** Budget Guidance shows that a separate predictor (analogous to our monitor) can estimate remaining computation and guide behavior. The soft, token-level guidance is a more fine-grained version of our monitor's stopping signal. The finding that the predictor naturally estimates difficulty supports using a monitor for adaptive cost allocation.