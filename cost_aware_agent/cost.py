"""Cost Engine — real dollar cost from token/tool-call counts, not arbitrary units.

Price map field names confirmed against the vendored LiteLLM snapshot,
2026-07-01: input_cost_per_token, output_cost_per_token,
cache_read_input_token_cost, cache_creation_input_token_cost.
"""

import json
from pathlib import Path

_PRICE_MAP_PATH = Path(__file__).parent / "data" / "model_prices_and_context_window.json"
_price_map: dict | None = None

# Router prefixes that are OpenCode's own routing layer, not a LiteLLM pricing
# key — LiteLLM tracks provider-native pricing (deepseek/, fireworks_ai/, ...),
# never opencode/, so an exact-key lookup on the raw model id always misses.
_ROUTER_PREFIXES = ("opencode/",)

# A genuinely unpriced model must NEVER cost $0 — a zero-cost channel is a hole
# in the budget pressure (route work through the unpriced model and the whole
# measurement is voided). Charge a conservative mid-tier rate instead: Claude
# Sonnet retail as of the 2026-07-01 vendored snapshot. Rows costed this way
# carry price_unknown=1 in the DB and the tracker shows a warning line.
FALLBACK_PRICE = {
    "input_cost_per_token": 3e-06,
    "output_cost_per_token": 1.5e-05,
    "cache_read_input_token_cost": 3e-07,
    "cache_creation_input_token_cost": 3.75e-06,
    "cache_creation_input_token_cost_above_1hr": 6e-06,
}


def _load_price_map() -> dict:
    global _price_map
    if _price_map is None:
        _price_map = json.loads(_PRICE_MAP_PATH.read_text())
    return _price_map


def price_for_model(model: str) -> dict | None:
    """Exact-key lookup first (covers every model seen in a real transcript
    so far). Falls back to stripping a known router prefix, then a trailing
    '-free' suffix, pricing at the underlying paid model's rate. The project
    tracks simulated real-market cost, not the user's actual bill — a $0 API
    tier must still show what the same usage would cost at retail, so the
    agent gets genuine budget pressure regardless of which account it runs
    under. Still returns None for a genuinely unpriced model — callers that
    need a usable rate go through resolve_price(), which substitutes
    FALLBACK_PRICE and flags the row instead of costing it $0."""
    price_map = _load_price_map()
    if not model:
        return None
    if model in price_map:
        return price_map[model]

    stripped = model
    for prefix in _ROUTER_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if stripped in price_map:
        return price_map[stripped]

    if stripped.lower().endswith("-free"):
        base = stripped[: -len("-free")]
        if base in price_map:
            return price_map[base]

    return None


def resolve_price(model: str) -> tuple[dict, bool]:
    """(price_dict, price_unknown). Never returns a zero price: an unpriced
    model gets FALLBACK_PRICE with the flag set, so no usage row can slip
    through the budget as free."""
    price = price_for_model(model)
    if price is None:
        return FALLBACK_PRICE, True
    return price, False


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
    must pass the split, not a single aggregate cache_creation_tokens number.

    An unpriced model is costed at FALLBACK_PRICE, never $0 (see resolve_price)."""
    price, _ = resolve_price(model)
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
    # A non-positive budget means "no budget configured" — treat as no pressure
    # (HIGH) rather than dividing by zero and 500-ing every /tool/pre and /status.
    if session_budget_estimate_usd <= 0:
        return "HIGH", 100.0
    pct_remaining = max(
        0.0,
        (session_budget_estimate_usd - spent) / session_budget_estimate_usd,
    ) * 100
    # epsilon so an exact boundary ((1.00-0.90)/1.00 -> 9.999...8 in floats)
    # lands in the documented tier (LOW is >=10%), not one tier tighter
    eps = 1e-9
    if pct_remaining < 10 - eps:
        tier = "CRITICAL"
    elif pct_remaining < 30 - eps:
        tier = "LOW"
    elif pct_remaining < 70 - eps:
        tier = "MEDIUM"
    else:
        tier = "HIGH"
    return tier, pct_remaining
