"""The CASSI economy — paper_plan_v2 §2.1, §2.2, §17 `cost_model`.

One module owns every dollar computation so coach, worker, and labels provably
optimize the SAME Lagrangian (§2.4 "one Lagrangian, not three"):

- step cost c_t (tokens at reference-local prices + tool fees; draft-line tokens included)
- budget tier from % of allowance remaining (HIGH/MEDIUM/LOW/CRITICAL)
- tier multipliers m(tier) — the discretized shadow price of spend (§2.2)
- pilot normalization  c̃ = c / median_pilot_spend  (λ becomes dimensionless, §2.1)
- wallet calibration from the P2 pilot (small=P25, medium=P75, large=2×P90)
- U_t = q_t − Σ_{i≤t} λ·m(tier_i)·c̃_i   (the stopping utility, §2.2)
- R_base = Q_τ − Σ_{i≤τ} λ·m(tier_i)·c̃_i (the executor outcome reward, §2.4)

API-model pricing reuses the repo harness price map (cost_aware_agent/cost.py) —
do not duplicate it (§17).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- harness reuse
_REPO_ROOT = Path(__file__).resolve().parents[3]   # .../cost-aware-agent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:  # the harness price map for API models (§17 cost_model.api_models)
    from cost_aware_agent.cost import resolve_price as harness_resolve_price
except Exception:  # CPU test environments without the package installed
    harness_resolve_price = None

# ------------------------------------------------------------------- constants
# §17 cost_model.token_prices_per_1M.reference_local — constant across methods
REFERENCE_LOCAL_PRICE_PER_1M = {"input": 0.60, "output": 2.20}

# §17 cost_model.tool_costs
TOOL_COSTS = {
    "web_search": {"per_query": 0.003, "per_result": 0.001},
    "http_fetch": {"per_request": 0.0001},
    "code_exec": {"per_exec": 0.0001, "per_sec": 0.0001},
    "retrieval_local": {"per_query": 0.0001},
}

# §2.2 / §17 label.tier_multipliers — m(tier), the marginal shadow price of spend
TIER_MULTIPLIERS = {"HIGH": 0.5, "MEDIUM": 1.0, "LOW": 2.0, "CRITICAL": 5.0}

# §17 inference.budget_tiers — tier from fraction of allowance REMAINING
TIER_BOUNDS = [("HIGH", 0.60), ("MEDIUM", 0.30), ("LOW", 0.10), ("CRITICAL", 0.0)]


# ------------------------------------------------------------------ step costs
def token_cost(input_tokens: int, output_tokens: int, model: str | None = None) -> float:
    """Dollar cost of one LLM call. Local models use the reference-local rate;
    a named API model uses the harness price map (never $0 — harness guarantees)."""
    if model is not None and harness_resolve_price is not None:
        price, _unknown = harness_resolve_price(model)
        return input_tokens * price["input_cost_per_token"] + output_tokens * price["output_cost_per_token"]
    return (
        input_tokens * REFERENCE_LOCAL_PRICE_PER_1M["input"]
        + output_tokens * REFERENCE_LOCAL_PRICE_PER_1M["output"]
    ) / 1_000_000


def tool_cost(tool: str, *, n_results: int = 0, seconds: float = 0.0) -> float:
    """Dollar cost of one tool invocation per the §17 fee schedule."""
    fees = TOOL_COSTS[tool]
    cost = fees.get("per_query", 0.0) + fees.get("per_request", 0.0) + fees.get("per_exec", 0.0)
    cost += n_results * fees.get("per_result", 0.0)
    cost += seconds * fees.get("per_sec", 0.0)
    return cost


# ------------------------------------------------------------------- tiering
def tier_from_remaining(spent_dollars: float, allowance_dollars: float) -> str:
    """Budget tier from the fraction of the wallet still unspent (§17 inference.budget_tiers)."""
    if allowance_dollars <= 0:
        return "CRITICAL"
    remaining = max(0.0, 1.0 - spent_dollars / allowance_dollars)
    for tier, lo in TIER_BOUNDS:
        if remaining > lo:
            return tier
    return "CRITICAL"


def tier_multiplier(tier: str, *, rule_table_off: bool = False) -> float:
    """m(tier) (§2.2). rule_table_off=True gives the plain-λ ablation A8 economy (m ≡ 1)."""
    if rule_table_off:
        return 1.0
    return TIER_MULTIPLIERS[tier]


# ------------------------------------------------- pilot calibration (P2, §16)
def calibrate_wallets(pilot_spends: list[float]) -> dict:
    """P2 pilot → wallet sizes: small=P25 of unconstrained spend, medium=P75, large=2×P90.
    Returns dollars; freeze the result into configs/cassi.yaml `label.allowances`."""
    s = np.asarray(pilot_spends, dtype=float)
    if len(s) < 20:
        raise ValueError(f"Pilot too small ({len(s)} tasks) — §16 P2 wants a 200-task pilot.")
    return {
        "small": float(np.percentile(s, 25)),
        "medium": float(np.percentile(s, 75)),
        "large": float(2.0 * np.percentile(s, 90)),
        "median_spend": float(np.median(s)),   # the C̃ normalization constant (§2.1)
    }


def draw_wallet(rng: np.random.Generator, allowances: dict) -> tuple[str, float]:
    """Draw one wallet per (task, GRPO group) — SHARED by all G rollouts of the group (§2.2).
    Never draw per-episode within a group: group advantages must compare behavior
    under the same wallet."""
    size = rng.choice(["small", "medium", "large"])
    return str(size), float(allowances[size])


# ------------------------------------------------------------ the one economy
def scaled_step_costs(
    costs: list[float], tiers: list[str], lam: float, median_pilot_spend: float,
    *, rule_table_off: bool = False,
) -> np.ndarray:
    """Per-step penalized normalized cost  λ·m(tier_i)·c̃_i  (§2.2).
    `costs` are raw dollars; `tiers` the tier prevailing when each was spent."""
    if median_pilot_spend <= 0:
        raise ValueError("median_pilot_spend must be positive (frozen from P2 pilot, §2.1)")
    c = np.asarray(costs, dtype=float) / median_pilot_spend
    m = np.asarray([tier_multiplier(t, rule_table_off=rule_table_off) for t in tiers])
    return lam * m * c


def stopping_utilities(
    qualities: list[float], costs: list[float], tiers: list[str],
    lam: float, median_pilot_spend: float, *, rule_table_off: bool = False,
) -> np.ndarray:
    """U_t = q_t − Σ_{i≤t} λ·m(tier_i)·c̃_i  for every step t (§2.2). Vectorized."""
    penal = scaled_step_costs(costs, tiers, lam, median_pilot_spend, rule_table_off=rule_table_off)
    return np.asarray(qualities, dtype=float) - np.cumsum(penal)


def base_reward(
    terminal_quality: float, costs_to_tau: list[float], tiers_to_tau: list[str],
    lam: float, median_pilot_spend: float, *, rule_table_off: bool = False,
) -> float:
    """R_base = Q_τ − Σ_{i≤τ} λ·m(tier_i)·c̃_i  (§2.4) — SAME economy as the labels."""
    penal = scaled_step_costs(costs_to_tau, tiers_to_tau, lam, median_pilot_spend,
                              rule_table_off=rule_table_off)
    return float(terminal_quality - penal.sum())
