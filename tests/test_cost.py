"""Cost Engine unit tests — tier boundaries, cache-split pricing, price lookup.

The cache 5m/1h split test exists because this exact bug shipped once: pricing
all cache-write tokens at the 5m rate undercounted a real session by ~24%.
"""
import pytest

from cost_aware_agent import cost


# --- compute_tier ---

@pytest.mark.parametrize("spent,budget,tier", [
    (0.00, 1.00, "HIGH"),       # 100% remaining
    (0.30, 1.00, "HIGH"),       # 70% remaining — HIGH is >= 70
    (0.31, 1.00, "MEDIUM"),     # 69% remaining
    (0.70, 1.00, "MEDIUM"),     # 30% remaining — MEDIUM is >= 30
    (0.71, 1.00, "LOW"),        # 29% remaining
    (0.90, 1.00, "LOW"),        # 10% remaining — LOW is >= 10
    (0.91, 1.00, "CRITICAL"),   # 9% remaining
    (1.00, 1.00, "CRITICAL"),   # 0% remaining
    (5.00, 1.00, "CRITICAL"),   # over budget: floors at 0, no error state
])
def test_tier_boundaries(spent, budget, tier):
    got, pct = cost.compute_tier(spent, budget)
    assert got == tier
    assert 0.0 <= pct <= 100.0


def test_tier_no_budget_means_no_pressure():
    # non-positive budget = "not configured" — HIGH, never a ZeroDivisionError
    assert cost.compute_tier(0.5, 0.0) == ("HIGH", 100.0)
    assert cost.compute_tier(0.5, -1.0) == ("HIGH", 100.0)


# --- cost_llm_usage ---

SONNET = "claude-sonnet-5"  # vendored map: in 3e-6, out 1.5e-5,
                            # cache read 3e-7, 5m write 3.75e-6, 1h write 6e-6


def test_llm_cost_basic_tokens():
    c = cost.cost_llm_usage(SONNET, 1000, 100, 0, 0, 0)
    assert c == pytest.approx(1000 * 3e-6 + 100 * 1.5e-5)


def test_llm_cost_cache_split_5m_vs_1h():
    # identical token count must cost MORE at the 1h rate — the shipped bug
    # priced both at 5m and undercounted
    c_5m = cost.cost_llm_usage(SONNET, 0, 0, 0, 10_000, 0)
    c_1h = cost.cost_llm_usage(SONNET, 0, 0, 0, 0, 10_000)
    assert c_5m == pytest.approx(10_000 * 3.75e-6)
    assert c_1h == pytest.approx(10_000 * 6e-6)
    assert c_1h > c_5m


def test_llm_cost_1h_rate_falls_back_to_5m_when_absent():
    # deepseek-v4-flash has no cache_creation_input_token_cost_above_1hr key —
    # 1h tokens must fall back to the 5m rate, not crash or price at 0-by-accident
    price = cost.price_for_model("deepseek-v4-flash")
    assert "cache_creation_input_token_cost_above_1hr" not in price
    c = cost.cost_llm_usage("deepseek-v4-flash", 0, 0, 0, 0, 10_000)
    assert c == pytest.approx(10_000 * price.get("cache_creation_input_token_cost", 0.0))


def test_llm_cost_unknown_model_is_zero_not_error():
    assert cost.cost_llm_usage("no-such-model-xyz", 1000, 1000, 0, 0, 0) == 0.0


# --- price_for_model ---

def test_price_exact_key():
    assert cost.price_for_model(SONNET) is not None


def test_price_strips_opencode_router_prefix():
    direct = cost.price_for_model("deepseek-v4-flash")
    routed = cost.price_for_model("opencode/deepseek-v4-flash")
    assert routed == direct


def test_price_free_suffix_priced_at_paid_retail():
    # the project tracks simulated real-market cost: a $0 API tier must still
    # price at the paid model's retail rate so budget pressure stays real
    paid = cost.price_for_model("deepseek-v4-flash")
    free = cost.price_for_model("opencode/deepseek-v4-flash-free")
    assert free == paid


def test_price_unknown_returns_none():
    assert cost.price_for_model("totally-unknown-model") is None
