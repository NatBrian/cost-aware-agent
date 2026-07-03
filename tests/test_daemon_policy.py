"""Injection-policy bucket math — over-budget cadence must coarsen."""
from cost_aware_agent import daemon


def test_bucket_under_budget_10pct_slices():
    assert daemon._spend_bucket(0.00, 1.0) == 0
    assert daemon._spend_bucket(0.05, 1.0) == 0
    assert daemon._spend_bucket(0.15, 1.0) == 1
    assert daemon._spend_bucket(0.95, 1.0) == 9


def test_bucket_over_budget_coarsens_to_half_budget_steps():
    b100 = daemon._spend_bucket(1.00, 1.0)
    b120 = daemon._spend_bucket(1.20, 1.0)   # +20% over -> same over-bucket
    b160 = daemon._spend_bucket(1.60, 1.0)   # +60% over -> next over-bucket
    assert b100 == b120
    assert b160 == b100 + 1


def test_bucket_no_budget():
    assert daemon._spend_bucket(5.0, 0.0) == 0
