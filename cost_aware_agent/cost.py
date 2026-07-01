"""Cost Engine — real dollar cost from token/tool-call counts, not arbitrary units.

Price map field names confirmed against the vendored LiteLLM snapshot,
2026-07-01: input_cost_per_token, output_cost_per_token,
cache_read_input_token_cost, cache_creation_input_token_cost.
"""

import json
from pathlib import Path

_PRICE_MAP_PATH = Path(__file__).parent / "data" / "model_prices_and_context_window.json"
_price_map: dict | None = None


def _load_price_map() -> dict:
    global _price_map
    if _price_map is None:
        _price_map = json.loads(_PRICE_MAP_PATH.read_text())
    return _price_map


def price_for_model(model: str) -> dict | None:
    """Exact-key lookup only for MVP — every model seen in a real transcript
    so far has a direct key. Returns None if the model is unknown (caller
    should cost that row as 0 rather than raise — advisory-only extends to
    pricing gaps too)."""
    return _load_price_map().get(model)


def cost_llm_usage(
    model: str, input_tokens: int, output_tokens: int,
    cache_read_tokens: int, cache_creation_5m_tokens: int = 0,
    cache_creation_1h_tokens: int = 0,
) -> float:
    """Cache-write tokens are NOT one price. Anthropic bills 1-hour ephemeral
    cache writes at a higher per-token rate than the 5-minute default (price
    map: cache_creation_input_token_cost = 5m rate,
    cache_creation_input_token_cost_above_1hr = 1h rate). Confirmed 2026-07-01
    against a real Claude Code session (which uses 1h caching exclusively,
    ephemeral_5m_input_tokens was 0 throughout) — treating all cache-write
    tokens at the 5m rate undercounted spend by ~24% in that session. Callers
    must pass the split, not a single aggregate cache_creation_tokens number."""
    price = price_for_model(model)
    if price is None:
        return 0.0
    return (
        input_tokens * price.get("input_cost_per_token", 0.0)
        + output_tokens * price.get("output_cost_per_token", 0.0)
        + cache_read_tokens * price.get("cache_read_input_token_cost", 0.0)
        + cache_creation_5m_tokens * price.get("cache_creation_input_token_cost", 0.0)
        + cache_creation_1h_tokens * price.get("cache_creation_input_token_cost_above_1hr", price.get("cache_creation_input_token_cost", 0.0))
    )


def cost_tool_call(config: dict) -> float:
    return config["tool_call_price_usd"]


def compute_tier(spent: float, session_budget_estimate_usd: float) -> tuple[str, float]:
    pct_remaining = max(
        0.0,
        (session_budget_estimate_usd - spent) / session_budget_estimate_usd,
    ) * 100
    if pct_remaining < 10:
        tier = "CRITICAL"
    elif pct_remaining < 30:
        tier = "LOW"
    elif pct_remaining < 70:
        tier = "MEDIUM"
    else:
        tier = "HIGH"
    return tier, pct_remaining
