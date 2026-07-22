"""Baseline suite — paper_plan_v2_1 §5.2 (rows B1–B10 + oracle), §5.3 frontier protocol, §16 P8.

One module per §5.2 row. Decision/reward logic lives here as pure CPU-testable
functions; anything needing training reuses the executor GRPO wiring and is
launched via scripts/p8_baselines.sh with reward_fn=<module>.reward (each module
documents its own launch line).

Frontier protocol (§5.3): every method is swept over ITS OWN cost knob to produce
a 3–5 point frontier; iso-accuracy cost is read by linear interpolation between
adjacent frontier points. Knobless methods (B1 ReAct, oracle) are reported as
single points and EXCLUDED from iso-claims.

Billing symmetry (§5.3): every method pays for ALL auxiliary inference it uses
under the same price map — B2's probe calls return token counts for exactly this
reason; B3's triggers are harness arithmetic (≈$0, disclosed).
"""

from __future__ import annotations

import importlib
from types import ModuleType

# name -> {module, type, cost_knob, needs_training} (+ provenance extras).
# `cost_knob` is the §5.3 frontier dial; None ⇒ single point, excluded from iso-claims.
BASELINES: dict[str, dict] = {
    "b1_react": {
        "module": "cassi.baselines.b1_react",
        "type": "lower_bound",
        "cost_knob": None,                       # single point — excluded from iso-claims (§5.3)
        "needs_training": False,
        "paper": "2210.03629",
        "kills": "how much slack exists",
    },
    "b2_probe": {
        "module": "cassi.baselines.b2_probe",
        "type": "training_free",
        "cost_knob": "confidence_threshold",
        "needs_training": False,
        "paper": "2412.20993",                   # Dynasor-style scalar probe (§5.2 B2)
        "kills": "why not just ask/probe? (dangerous baseline — LearnStop shows it can win)",
    },
    "b3_supervisor_monitor": {
        "module": "cassi.baselines.b3_supervisor_monitor",
        "type": "training_free_monitor",
        "cost_knob": "trigger_sensitivity",
        "needs_training": False,
        "paper": "2510.26585",                   # SupervisorAgent, ICLR'26
        "kills": "the ICLR'26 bar: -29.7% tokens at parity on GAIA",
    },
    "b4_otc_grpo": {
        "module": "cassi.baselines.b4_otc_grpo",
        "type": "cost_aware_rl_outcome_level",
        "cost_knob": "tool_count_coefficient",
        "needs_training": True,
        "paper": "2504.14870",                   # OTC-PO
        "kills": "is step-level economic signal needed at all",
    },
    "b5_eapo": {
        "module": "cassi.baselines.b5_eapo",
        "type": "adaptive_penalty_rl",
        "cost_knob": "penalty_weight",
        "needs_training": True,
        "paper": "2606.02132",                   # EAPO primary; agentic-ALP 2506.05256 fallback
        "kills": "is a learned VALUE better than adaptive scalar pressure",
    },
    "b6_single_model_cost": {
        "module": "cassi.baselines.b6_single_model_cost",
        "type": "single_model_cost_rl",
        "cost_knob": "lambda",
        "needs_training": True,
        "paper": "2602.16699",                   # CTA-style cost-in-reward
        "kills": "two-model necessity (matched params with ablation A2)",
    },
    "b7_cart_cost": {
        "module": "cassi.baselines.b7_cart_cost",
        "type": "trained_self_termination",
        "cost_knob": "label_lambda",
        "needs_training": True,
        "paper": "2510.08517",                   # CaRT
        "kills": "is RL even needed, or does imitation suffice? (two arms: SFT-only, +GRPO)",
        "arms": ["sft_only", "sft_plus_grpo"],
    },
    "b8_agentprm_cost": {
        "module": "cassi.baselines.b8_agentprm_cost",
        "type": "quality_prm_plus_cost",
        "cost_knob": "lambda",
        "needs_training": True,
        "paper": "2502.10325",                   # AgentPRM (pooled return-to-go, NOT per-state MC)
        "kills": "is the STOPPING-VALUE semantics what matters vs generic value+cost",
    },
    "b9_direct_shaping": {
        "module": "cassi.baselines.b9_direct_shaping",
        "type": "ablation_as_baseline",
        "cost_knob": "lambda",
        "needs_training": True,
        "paper": "2607.00482",                   # DASH, adapted
        "kills": "THE pivotal test: does the stopper earn its existence (H3)",
    },
    "b10_prompted_rm": {
        "module": "cassi.baselines.b10_prompted_rm",
        "type": "prompted_reward_model",
        "cost_knob": "rubric_threshold",         # θ_p — stop when continue-score ≤ θ_p (v2.1 §5.2)
        "needs_training": False,                 # the judge is NEVER trained; the rl arm trains
                                                 # the EXECUTOR (post-K1, reuses P6/P8 GRPO wiring)
        "paper": "2309.00267",                   # RLAIF-style prompted judge (v2.1 B10 "RM-P")
        "kills": "the trained-vs-prompted reward-model question, in our own tables (v2.1)",
        "arms": ["monitor_training_free", "rl_post_k1_trains_executor"],
    },
    "oracle": {
        "module": "cassi.baselines.oracle",
        "type": "upper_bound",
        "cost_knob": None,                       # eval-only, GT-based — excluded from iso-claims
        "needs_training": False,
        "paper": None,                           # Snell tau* with ground truth (§2.2 / §5.2)
        "kills": "headroom",
    },
}

REQUIRED_KEYS = ("module", "type", "cost_knob", "needs_training")


def load(name: str) -> ModuleType:
    """Import a baseline module by registry name."""
    return importlib.import_module(BASELINES[name]["module"])
