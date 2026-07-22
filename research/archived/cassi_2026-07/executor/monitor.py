"""Algorithm 4 — inference protocol (paper_plan_v2 §2.5, §10).

    each k-th step (k=1 headline): serialize x_t (harness features + user's λ,
    §18.1) → M_θ → Δ̂_t
    stop if Δ̂_t ≤ 0  or  executor emits final answer  or  budget exhausted

Fixed threshold (0.0): allowance-sensitivity is already IN THE WEIGHTS (§2.2
learned conditioning). The v5-style hand-written δ(tier) rule table survives
ONLY as ablation A8 (config `inference.ablation_A8_rule_table`) — a comparator,
never the default.

The monitor also tracks the INTERNALIZATION metric (§2.5): % of episodes the
executor self-terminated (emitted ANSWER) before the monitor fired — the
cleanest evidence that economics moved into the policy.

The stopper behind the monitor is either protocol:
  * text-based `StopperClient` (`evaluate(serialized_x) -> StopperOutput`) —
    the generative variant (§18.3) and this module's `MockStopper`;
  * feature-based `predict(x, lam, meta) -> StopperPrediction` — the value-head
    variant in `cassi.stopper.model` (ThreeHeadStopper via HFStopperPredictor,
    and its MockStopper). The monitor detects `.predict` and passes the raw
    StepFeatures + λ directly; the predictor serializes with the same §18.1
    template internally, so the two paths see identical inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from cassi.common.schema import StepFeatures
from cassi.executor.react_agent import DEFAULT_TOKENS_MAX, EpisodeResult
from cassi.stopper.features import serialize


# ---------------------------------------------------------------------- stopper
@dataclass
class StopperOutput:
    """One stopper query result (§18.3): Δ̂ drives the stop decision, V̂ is the
    shaping potential (unused at inference; carried for diagnostics)."""

    delta: float
    value: float = 0.0
    action: str | None = None            # optional STOP|CONTINUE from the CE head


@runtime_checkable
class StopperClient(Protocol):
    """Whatever answers a serialized §18.1 x_t block with Δ̂ (and V̂).
    `cassi.stopper.model` must expose this interface once implemented."""

    def evaluate(self, serialized_x: str) -> StopperOutput: ...


_STEP_RE = re.compile(r"\[PROGRESS\] step (\d+)/")


class MockStopper:
    """Deterministic stopper for CPU tests. Reads the step index straight out of
    the serialized §18.1 block (honest protocol: text in, decision out) and
    returns a scheduled Δ̂ — or `delta_fn(step_idx)` when provided."""

    def __init__(self, *, delta_by_step: dict[int, float] | None = None,
                 default_delta: float = 1.0,
                 delta_fn: Callable[[int], float] | None = None,
                 value_by_step: dict[int, float] | None = None):
        self.delta_by_step = delta_by_step or {}
        self.default_delta = default_delta
        self.delta_fn = delta_fn
        self.value_by_step = value_by_step or {}
        self.n_queries = 0

    def evaluate(self, serialized_x: str) -> StopperOutput:
        self.n_queries += 1
        m = _STEP_RE.search(serialized_x)
        t = int(m.group(1)) if m else 0
        if self.delta_fn is not None:
            delta = float(self.delta_fn(t))
        else:
            delta = float(self.delta_by_step.get(t, self.default_delta))
        return StopperOutput(delta=delta, value=float(self.value_by_step.get(t, 0.0)),
                             action="STOP" if delta <= 0 else "CONTINUE")


# ---------------------------------------------------------------------- monitor
class StopperMonitor:
    """§10 Algorithm 4. Plugs into `ReactAgent.run(..., monitor=...)`; the agent
    calls `should_stop` on every PRE-ACTION state x_t and `record_episode` at
    episode end.

    Modes:
      * learned (default): STOP when Δ̂ ≤ delta_threshold (fixed 0.0, §2.5);
      * ablation A8: pass `rule_table` (the §17 `inference.ablation_A8_rule_table`
        δ(tier) dict) — STOP when Δ̂ ≤ δ(x_t.tier). Comparator only, never default.

    Budget-exhausted stop applies in both modes and regardless of `every_k`.
    """

    def __init__(
        self,
        stopper: StopperClient,
        lam: float,
        *,
        every_k: int = 1,                        # §17 inference.stopper_eval_every_k (A5: 2, 3)
        delta_threshold: float = 0.0,            # §17 inference.delta_threshold — FIXED
        rule_table: dict[str, float] | None = None,   # ablation A8 ONLY
        tokens_max: int = DEFAULT_TOKENS_MAX,
        tool_calls_max: int | None = None,
        t_max: int = 10,
    ):
        if every_k < 1:
            raise ValueError("every_k must be ≥ 1")
        self.stopper = stopper
        self.lam = lam
        self.every_k = every_k
        self.delta_threshold = delta_threshold
        self.rule_table = dict(rule_table) if rule_table else None
        self.tokens_max = tokens_max
        self.tool_calls_max = tool_calls_max
        self.t_max = t_max
        # bookkeeping (internalization metric + overhead accounting)
        self.n_queries = 0
        self.episodes = 0
        self.self_terminated = 0
        self.monitor_stopped = 0
        self.budget_stopped = 0
        self.last_delta: float | None = None

    # -- Alg.4 per-step decision -------------------------------------------------
    def should_stop(self, x: StepFeatures, allowance_dollars: float) -> str | None:
        """Returns the stop reason ('budget' | 'monitor') or None to continue."""
        if allowance_dollars > 0 and x.dollars >= allowance_dollars:
            return "budget"                       # budget exhausted (Alg.4)
        if x.step_idx % self.every_k != 0:
            return None                           # not a query step (A5 ablation)
        if hasattr(self.stopper, "predict"):      # feature-based value-head variant
            out = self.stopper.predict(x, self.lam)   # (cassi.stopper.model; serializes
        else:                                     #  the same §18.1 template internally)
            serialized = serialize(
                x, self.lam,
                tokens_max=self.tokens_max,
                tool_calls_max=self.tool_calls_max or self.t_max,
                allowance_dollars=allowance_dollars,
                t_max=self.t_max,
            )
            out = self.stopper.evaluate(serialized)
        self.n_queries += 1
        self.last_delta = out.delta
        threshold = (self.rule_table[x.tier] if self.rule_table is not None
                     else self.delta_threshold)
        return "monitor" if out.delta <= threshold else None

    # -- internalization tracking (§2.5) ------------------------------------------
    def record_episode(self, result: EpisodeResult) -> None:
        self.episodes += 1
        if result.stopped_by == "answer":
            self.self_terminated += 1            # executor beat the monitor to it
        elif result.stopped_by == "monitor":
            self.monitor_stopped += 1
        elif result.stopped_by == "budget":
            self.budget_stopped += 1

    def stats(self) -> dict:
        return {
            "episodes": self.episodes,
            "self_terminated": self.self_terminated,
            "monitor_stopped": self.monitor_stopped,
            "budget_stopped": self.budget_stopped,
            "self_termination_rate": (self.self_terminated / self.episodes
                                      if self.episodes else 0.0),
            "stopper_queries": self.n_queries,
            "mode": "rule_table_A8" if self.rule_table is not None else "learned",
        }


def monitor_from_config(stopper: StopperClient, cfg: dict, *, lam: float | None = None,
                        t_max: int | None = None, domain: str = "qa",
                        ablation_a8: bool = False) -> StopperMonitor:
    """Build the §2.5 monitor from configs/cassi.yaml (§17 `inference` block).
    `ablation_a8=True` selects the δ(tier) rule-table comparator (A8)."""
    inf = cfg["inference"]
    return StopperMonitor(
        stopper,
        lam if lam is not None else float(cfg["label"]["default_lambda"]),
        every_k=int(inf["stopper_eval_every_k"]),
        delta_threshold=float(inf["delta_threshold"]),
        rule_table=dict(inf["ablation_A8_rule_table"]) if ablation_a8 else None,
        t_max=t_max if t_max is not None else int(cfg["executor"]["horizon"][domain]),
    )
