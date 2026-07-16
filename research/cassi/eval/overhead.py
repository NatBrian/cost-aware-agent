"""End-to-end honest overhead accounting — paper_plan_v2 §5.3 (feeds T4).

The audit demand: total dollars INCLUDING running-draft template tokens,
forced-continuation collection overhead, stopper training (amortized), stopper
inference, probe/monitor calls — per SERVING REGIME:

* `kv_fork`   — the stopper forks the executor's KV cache, so each stopper call
  pays only its own marginal tokens (the serialized x_t block + decision output).
* `re_prefill` — black-box serving: every stopper call RE-PAYS the shared prefix
  (task + history context) before its marginal tokens. Overhead can flip sign
  between regimes (LearnStop 2606.30852) — both are modeled, both are reported
  (T4, H6).

Billing symmetry (§5.3): every method pays for ALL auxiliary inference it uses —
B2's self-eval prompts, B3's monitor triggers, our stopper's calls, any judge —
under the SAME price map. `assert_billing_symmetry` is the tripwire: run it over
the full ledger set before emitting T4.

Replay/analysis dollars (the §5.3 dual-run regret replays) are carried on the
ledger but NEVER counted in a method's total — they bill to the analysis line.

Reporting follows the HAL harness conventions (princeton-pli/hal-harness,
2510.11977). All CPU, stdlib + the repo price constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from cassi.budget.cost import REFERENCE_LOCAL_PRICE_PER_1M

__all__ = [
    "KV_FORK",
    "RE_PREFILL",
    "SERVING_REGIMES",
    "stopper_inference_cost_usd",
    "amortized_training_usd",
    "MethodLedger",
    "BillingAsymmetryError",
    "assert_billing_symmetry",
]

KV_FORK = "kv_fork"
RE_PREFILL = "re_prefill"
SERVING_REGIMES = (KV_FORK, RE_PREFILL)

# Ledger components that count toward a method's own total (T4 rows).
_METHOD_COST_FIELDS = (
    "rollout_tokens_usd",          # the agent's own generation + tool traffic
    "draft_line_tokens_usd",       # running-draft template line (§2.6 — ALL methods)
    "forced_continuation_usd",     # label-collection overhead past would-have-stopped (§2.1)
    "stopper_training_usd",        # amortized per evaluated episode (zero for training-free)
    "stopper_inference_usd",       # per-step monitor calls, regime-dependent
    "probe_monitor_usd",           # B2 self-eval prompts / B3 triggers / judges (billing symmetry)
)
_ANALYSIS_COST_FIELDS = (
    "replay_analysis_usd",         # §5.3 dual-run regret replays — analysis line, never a method's
)


# --------------------------------------------------------- regime-aware pricing
def stopper_inference_cost_usd(
    n_calls: int,
    prefix_tokens: int,
    input_tokens_per_call: int,
    output_tokens_per_call: int,
    regime: str,
    price_per_1m: dict | None = None,
) -> float:
    """Dollar cost of `n_calls` stopper/monitor evaluations under a serving
    regime (§5.3). Under `kv_fork` each call pays only its marginal input
    (the serialized §18.1 block); under `re_prefill` each call ALSO re-pays the
    `prefix_tokens` shared context — the regime that can flip the overhead sign."""
    if regime not in SERVING_REGIMES:
        raise ValueError(f"unknown serving regime {regime!r}; expected one of {SERVING_REGIMES}")
    p = price_per_1m or REFERENCE_LOCAL_PRICE_PER_1M
    billed_input = input_tokens_per_call + (prefix_tokens if regime == RE_PREFILL else 0)
    per_call = (billed_input * p["input"] + output_tokens_per_call * p["output"]) / 1_000_000
    return float(n_calls * per_call)


def amortized_training_usd(total_training_usd: float, n_episodes: int) -> float:
    """Stopper-training dollars amortized per evaluated episode (§5.3 —
    'amortization vs training-free baselines' zero amortization')."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    return float(total_training_usd) / n_episodes


# ---------------------------------------------------------------------- ledger
@dataclass
class MethodLedger:
    """One method's end-to-end dollar ledger under one serving regime (T4 row).

    All *_usd fields are totals over the same frozen eval set (per-episode
    amortization already applied to training). `price_map` is the token price
    map the auxiliary inference was billed with — billing symmetry (§5.3) is
    the assertion that it is IDENTICAL across every compared ledger.
    """

    method: str
    regime: str
    rollout_tokens_usd: float = 0.0
    draft_line_tokens_usd: float = 0.0
    forced_continuation_usd: float = 0.0
    stopper_training_usd: float = 0.0
    stopper_inference_usd: float = 0.0
    probe_monitor_usd: float = 0.0
    replay_analysis_usd: float = 0.0
    price_map: dict = field(default_factory=lambda: dict(REFERENCE_LOCAL_PRICE_PER_1M))

    def __post_init__(self) -> None:
        if self.regime not in SERVING_REGIMES:
            raise ValueError(f"unknown serving regime {self.regime!r}")
        for f in _METHOD_COST_FIELDS + _ANALYSIS_COST_FIELDS:
            if getattr(self, f) < 0:
                raise ValueError(f"{self.method}: {f} is negative")

    def method_total_usd(self) -> float:
        """Everything the METHOD pays (§5.3) — excludes the analysis line."""
        return float(sum(getattr(self, f) for f in _METHOD_COST_FIELDS))

    def grand_total_usd(self) -> float:
        """Method total + analysis-line replays (bookkeeping only, not a T4
        method column)."""
        return self.method_total_usd() + float(
            sum(getattr(self, f) for f in _ANALYSIS_COST_FIELDS)
        )

    def to_row(self) -> dict:
        """Flat dict for the T4 CSV (analysis/tables/t4_overhead.py schema)."""
        row = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "price_map"}
        row["method_total_usd"] = self.method_total_usd()
        return row


# ------------------------------------------------------------ billing symmetry
class BillingAsymmetryError(AssertionError):
    """Two compared methods priced auxiliary inference under different price
    maps — T4 comparisons are void until fixed (§5.3 billing symmetry)."""


def assert_billing_symmetry(ledgers: list[MethodLedger]) -> None:
    """Assert every method's auxiliary inference is priced with the SAME price
    map (§5.3). Call over the full set of compared ledgers before emitting T4."""
    if not ledgers:
        raise ValueError("no ledgers to check")
    ref = ledgers[0].price_map
    offenders = [l.method for l in ledgers[1:] if l.price_map != ref]
    if offenders:
        raise BillingAsymmetryError(
            f"price map differs from '{ledgers[0].method}' for: {offenders} — "
            "billing symmetry (§5.3) requires ONE price map across all methods' "
            "auxiliary inference (self-eval prompts, monitor triggers, stopper calls, judges)."
        )
