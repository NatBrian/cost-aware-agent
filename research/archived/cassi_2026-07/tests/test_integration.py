"""Cross-module integration: the two stopper protocols drive the same monitor.

§2.5/§18.3 define two serving variants — generative (text in) and value-head
(features in). StopperMonitor must stop at the same step under either."""

from __future__ import annotations

from cassi.common.schema import StepFeatures
from cassi.executor.monitor import MockStopper as TextMockStopper, StopperMonitor
from cassi.stopper.model import MockStopper as FeatureMockStopper


def _x(t: int) -> StepFeatures:
    return StepFeatures(step_idx=t, dollars=0.01 * t, tier="HIGH",
                        question="q", domain="qa", draft="d", draft_len=1)


def test_both_protocols_stop_at_same_step():
    # Δ̂ schedule: positive until step 4, ≤0 from step 5
    text_stopper = TextMockStopper(delta_fn=lambda t: 0.5 if t < 5 else -0.1)
    feat_stopper = FeatureMockStopper(
        delta_fn=lambda x, lam, meta: 0.5 if x.step_idx < 5 else -0.1)

    for stopper in (text_stopper, feat_stopper):
        mon = StopperMonitor(stopper, lam=1.0, t_max=10)
        decisions = [mon.should_stop(_x(t), allowance_dollars=1.0) for t in range(1, 8)]
        assert decisions[:4] == [None] * 4
        assert decisions[4] == "monitor"          # step 5: first Δ̂ ≤ 0


def test_feature_protocol_value_head_flows():
    feat = FeatureMockStopper(delta_fn=lambda x, lam, meta: 1.0,
                              v_fn=lambda x, lam, meta: 0.42)
    mon = StopperMonitor(feat, lam=1.0, t_max=10)
    assert mon.should_stop(_x(1), allowance_dollars=1.0) is None
    assert mon.n_queries == 1
