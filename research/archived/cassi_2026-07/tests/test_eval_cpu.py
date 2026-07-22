"""CPU tests for eval/ (§5.3 metrics, §5.6 stats, T4 overhead) and the P10
analysis scripts. Run from research/cassi/:  python -m pytest tests/test_eval_cpu.py -q
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cassi.eval.metrics import (
    Frontier,
    KnoblessFrontierError,
    accuracy_at_iso_cost,
    cost_at_iso_accuracy,
    internalization_metrics,
    matched_lost_correct_risk,
    pareto_auc,
    stopping_regret,
)
from cassi.eval.overhead import (
    KV_FORK,
    RE_PREFILL,
    BillingAsymmetryError,
    MethodLedger,
    amortized_training_usd,
    assert_billing_symmetry,
    stopper_inference_cost_usd,
)
from cassi.eval.stats import (
    SmallSampleError,
    bootstrap_ci,
    effect_sizes,
    holm_bonferroni,
    paired_tests,
    pareto_dominance_bootstrap,
    small_n_guard,
)

CASSI_DIR = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------ stopping regret
def test_stopping_regret_hand_built_curve():
    u = np.array([0.10, 0.50, 0.30, 0.05])        # U_1..U_4
    assert stopping_regret(u, tau_method=3, tau_star=2) == pytest.approx(0.20)
    assert stopping_regret(u, tau_method=2, tau_star=2) == 0.0
    # utility gap, NOT |t − t*|: stopping 2 steps late at similar utility ≠ big regret
    assert stopping_regret(u, tau_method=4, tau_star=2) == pytest.approx(0.45)
    # method may pathwise beat the conditional-expectation optimum (§2.2 note iii)
    assert stopping_regret(u, tau_method=2, tau_star=3) == pytest.approx(-0.20)
    with pytest.raises(ValueError):
        stopping_regret(u, tau_method=0, tau_star=2)
    with pytest.raises(ValueError):
        stopping_regret(u, tau_method=5, tau_star=2)


# ---------------------------------------------------------- frontier protocol
@pytest.fixture
def three_point_frontier() -> Frontier:
    return Frontier([(1.0, 0.5), (2.0, 0.7), (4.0, 0.8)], method="cassi")


def test_frontier_interpolation_exact(three_point_frontier):
    f = three_point_frontier
    assert cost_at_iso_accuracy(f, 0.6) == pytest.approx(1.5)     # midway 0.5↔0.7
    assert cost_at_iso_accuracy(f, 0.75) == pytest.approx(3.0)    # midway 0.7↔0.8
    assert cost_at_iso_accuracy(f, 0.5) == pytest.approx(1.0)     # endpoint
    assert accuracy_at_iso_cost(f, 3.0) == pytest.approx(0.75)
    assert accuracy_at_iso_cost(f, 1.5) == pytest.approx(0.6)
    # outside the swept range → None, never extrapolated (§5.3)
    assert cost_at_iso_accuracy(f, 0.4) is None
    assert cost_at_iso_accuracy(f, 0.9) is None
    assert accuracy_at_iso_cost(f, 0.5) is None
    assert accuracy_at_iso_cost(f, 5.0) is None


def test_frontier_drops_dominated_points():
    f = Frontier([(1.0, 0.5), (2.0, 0.7), (3.0, 0.6), (4.0, 0.8)], method="m")
    assert len(f.eff_cost) == 3                    # (3.0, 0.6) is dominated
    assert accuracy_at_iso_cost(f, 3.0) == pytest.approx(0.75)


def test_knobless_single_point_excluded_from_iso_claims():
    f = Frontier([(2.0, 0.6)], method="b1_react")
    assert f.is_single_point
    for fn, arg in ((cost_at_iso_accuracy, 0.6), (accuracy_at_iso_cost, 2.0)):
        with pytest.raises(KnoblessFrontierError):
            fn(f, arg)
    with pytest.raises(KnoblessFrontierError):
        pareto_auc(f, (1.0, 3.0))


def test_pareto_auc_known_values(three_point_frontier):
    f = three_point_frontier
    # exact support: (0.6·1 + 0.75·2) / 3 = 0.7
    assert pareto_auc(f, (1.0, 4.0)) == pytest.approx(0.7)
    # below support contributes 0: 2.1 / 4
    assert pareto_auc(f, (0.0, 4.0)) == pytest.approx(0.525)
    # above support plateaus at max accuracy: (2.1 + 0.8·2) / 5
    assert pareto_auc(f, (1.0, 6.0)) == pytest.approx(0.74)
    with pytest.raises(ValueError):
        pareto_auc(f, (4.0, 1.0))


# ------------------------------------------------- matched lost-correct risk
def test_matched_lost_correct_risk_synthetic():
    n = 100
    correct_full = np.ones(n)
    cost_full = np.ones(n)
    # threshold sweep: lose {0, 2, 5} correct answers for {10%, 30%, 50%} savings
    cm = np.ones((3, n))
    cm[1, :2] = 0
    cm[2, :5] = 0
    costm = np.stack([np.full(n, 0.9), np.full(n, 0.7), np.full(n, 0.5)])
    res = matched_lost_correct_risk(correct_full, cost_full, cm, costm,
                                    risk_levels=(0.01, 0.02, 0.05, 0.10))
    assert res.savings_at_risk[0.01] == pytest.approx(0.2)   # interp (0→0.1, 0.02→0.3)
    assert res.savings_at_risk[0.02] == pytest.approx(0.3)
    assert res.savings_at_risk[0.05] == pytest.approx(0.5)
    assert res.savings_at_risk[0.10] is None                 # sweep never reached 10%
    assert list(res.lost_fracs) == pytest.approx([0.0, 0.02, 0.05])


# ------------------------------------------------------------ internalization
def test_internalization_metrics():
    out = internalization_metrics(
        self_stop_flags=np.array([True, True, False, False]),
        cost_monitor_on=np.array([2.0, 2.0]), acc_monitor_on=np.array([1.0, 0.0]),
        cost_monitor_off=np.array([3.0, 3.0]), acc_monitor_off=np.array([1.0, 1.0]),
        baseline_cost=10.0,
    )
    assert out["self_termination_rate"] == pytest.approx(0.5)
    assert out["cost_delta_monitor_off"] == pytest.approx(1.0)
    assert out["acc_delta_monitor_off"] == pytest.approx(0.5)
    # H5 retention: (10−3)/(10−2)
    assert out["savings_retention_monitor_off"] == pytest.approx(0.875)


# ----------------------------------------------------------------- bootstrap
def test_bootstrap_ci_coverage_sanity():
    rng = np.random.default_rng(0)
    covered = 0
    for rep in range(100):
        v = rng.normal(0.0, 1.0, size=200)
        ci = bootstrap_ci(v, n_boot=400, seed=rep)
        assert ci.lo <= ci.point <= ci.hi
        covered += ci.lo <= 0.0 <= ci.hi
    assert covered >= 85                      # nominal 95%, generous slack


# ------------------------------------------------------------- paired + guard
def test_small_n_guard_policy():
    assert small_n_guard(500)
    assert not small_n_guard(499)
    assert small_n_guard(103, threshold=100)


def test_paired_tests_enforce_small_n_and_report_both():
    rng = np.random.default_rng(1)
    small = rng.normal(size=103)
    with pytest.raises(SmallSampleError):     # GAIA-103: CIs only (§5.6)
        paired_tests(small, small + 0.1)
    a = rng.normal(size=600)
    b = a - 0.5 - 0.1 * rng.normal(size=600)  # a clearly larger, paired
    res = paired_tests(a, b)
    assert res["n"] == 600
    assert res["t_pvalue"] < 1e-6 and res["wilcoxon_pvalue"] < 1e-6
    assert res["governing"] in ("t", "wilcoxon")
    assert res["governing_pvalue"] == res[f"{'t' if res['governing'] == 't' else 'wilcoxon'}_pvalue"]
    # heavy-tailed diffs → Wilcoxon governs
    c = a + rng.standard_cauchy(size=600)
    assert paired_tests(a, c)["governing"] == "wilcoxon"


# ------------------------------------------------------------ Holm–Bonferroni
def test_holm_bonferroni_known_case():
    res = holm_bonferroni([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    assert res.reject == [True, False, False, True]
    assert res.adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])
    all_reject = holm_bonferroni([0.001, 0.002], alpha=0.05)
    assert all_reject.reject == [True, True]


# ------------------------------------------------- Pareto dominance bootstrap
def test_pareto_dominance_bootstrap_clear_domination():
    n = 400

    def correct_rows(fracs):
        return np.stack([
            np.concatenate([np.ones(int(f * n)), np.zeros(n - int(f * n))])
            for f in fracs
        ])

    correct = correct_rows([0.5, 0.7, 0.9])           # same accuracies both methods
    costs_a = np.stack([np.full(n, c) for c in (1.0, 2.0, 3.0)])
    costs_b = np.stack([np.full(n, c) for c in (3.0, 6.0, 9.0)])
    frac = pareto_dominance_bootstrap(costs_a, correct, costs_b, correct,
                                      n_boot=200, seed=0)
    assert frac >= 0.95                                # A dominates ~always
    # and the reverse direction ~never dominates
    frac_rev = pareto_dominance_bootstrap(costs_b, correct, costs_a, correct,
                                          n_boot=200, seed=0)
    assert frac_rev <= 0.05


# ---------------------------------------------------------------- effect sizes
def test_effect_sizes_known_values():
    res = effect_sizes(np.array([2.0, 4.0, 2.0, 4.0]), np.array([1.0, 3.0, 1.0, 3.0]))
    assert res["cohens_d"] == pytest.approx(np.sqrt(3) / 2)      # 1 / sqrt(4/3)
    assert res["abs_risk_difference"] == pytest.approx(1.0)
    binary = effect_sizes(np.array([1.0, 1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    assert binary["abs_risk_difference"] == pytest.approx(0.5)   # 50 pp


# -------------------------------------------------------------- overhead / T4
def test_overhead_ledger_sums_and_analysis_line():
    led = MethodLedger(
        method="cassi", regime=KV_FORK,
        rollout_tokens_usd=1.0, draft_line_tokens_usd=2.0,
        forced_continuation_usd=3.0, stopper_training_usd=4.0,
        stopper_inference_usd=5.0, probe_monitor_usd=6.0,
        replay_analysis_usd=7.0,
    )
    assert led.method_total_usd() == pytest.approx(21.0)   # replay NOT in the method total
    assert led.grand_total_usd() == pytest.approx(28.0)
    row = led.to_row()
    assert row["method_total_usd"] == pytest.approx(21.0)
    assert "price_map" not in row
    with pytest.raises(ValueError):
        MethodLedger(method="x", regime="serverless")      # unknown regime


def test_stopper_inference_regimes():
    kv = stopper_inference_cost_usd(100, prefix_tokens=1000, input_tokens_per_call=200,
                                    output_tokens_per_call=10, regime=KV_FORK)
    re = stopper_inference_cost_usd(100, prefix_tokens=1000, input_tokens_per_call=200,
                                    output_tokens_per_call=10, regime=RE_PREFILL)
    assert kv == pytest.approx(100 * (200 * 0.60 + 10 * 2.20) / 1e6)
    # re-prefill re-pays the prefix on every call (§5.3)
    assert re - kv == pytest.approx(100 * 1000 * 0.60 / 1e6)
    with pytest.raises(ValueError):
        stopper_inference_cost_usd(1, 0, 0, 0, regime="warm")


def test_billing_symmetry():
    a = MethodLedger(method="cassi", regime=KV_FORK, stopper_inference_usd=1.0)
    b = MethodLedger(method="b2_probe", regime=KV_FORK, probe_monitor_usd=0.5)
    assert_billing_symmetry([a, b])                        # same price map: fine
    c = MethodLedger(method="b3_supervisor", regime=KV_FORK,
                     price_map={"input": 0.30, "output": 2.20})
    with pytest.raises(BillingAsymmetryError):
        assert_billing_symmetry([a, b, c])


def test_amortized_training():
    assert amortized_training_usd(100.0, 200) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        amortized_training_usd(100.0, 0)


# ------------------------------------------------------- P10 analysis scripts
def _run(script_rel: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CASSI_DIR / script_rel), *args],
        capture_output=True, text=True, cwd=CASSI_DIR,
    )


def test_f2_demo_writes_pdf(tmp_path):
    out = tmp_path / "f2_demo.pdf"
    proc = _run("analysis/figures/f2_shaping_intuition.py", "--demo", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:5] == b"%PDF-"


def test_f2_without_demo_skips_gracefully_on_missing_csv(tmp_path):
    out = tmp_path / "f2.pdf"
    proc = _run("analysis/figures/f2_shaping_intuition.py",
                "--results", str(tmp_path / "nope.csv"), "--out", str(out))
    assert proc.returncode == 0, proc.stderr        # graceful skip (§16 P10)
    assert not out.exists()                         # no demo fallback without --demo
    assert "skipping" in proc.stdout


def test_t1_table_from_synthetic_csv(tmp_path):
    csv = tmp_path / "t1_headline.csv"
    csv.write_text(
        "domain,method,cost_at_iso_acc_usd,cost_ci_lo,cost_ci_hi,"
        "acc_at_iso_cost,acc_ci_lo,acc_ci_hi,pareto_auc,n_seeds\n"
        "qa,cassi,0.010,0.008,0.012,0.720,0.700,0.740,0.680,3\n"
        "qa,b4_otc,0.015,0.013,0.017,0.700,0.680,0.720,0.640,3\n"
        "qa,b1_react,,,,,,,,3\n"                    # knobless: no iso numbers (§5.3)
    )
    out = tmp_path / "t1_headline.tex"
    proc = _run("analysis/tables/t1_headline.py", "--results", str(csv), "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    tex = out.read_text()
    assert tex.count(r"\begin{tabular}") == 1 and tex.count(r"\end{tabular}") == 1
    assert r"\toprule" in tex and r"\bottomrule" in tex
    assert r"b4\_otc" in tex                        # underscores escaped
    assert r"\textbf{\$0.010" in tex                # best cost bolded
    assert tex.splitlines()[0].startswith("% AUTO-GENERATED")
    line_b1 = next(l for l in tex.splitlines() if l.startswith("qa & b1\\_react"))
    assert "--" in line_b1                          # knobless renders as '--'
