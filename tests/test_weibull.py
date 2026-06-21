"""
test_weibull.py — Unit tests for Weibull fitting and failure-probability math.

Validates against known closed-form Weibull properties:
  • F(eta) = 1 - e^-1 ≈ 0.632   (definition of characteristic life)
  • Fitted parameters on a *known-generated* sample should recover the
    true (beta, eta) within statistical tolerance.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from scipy.stats import weibull_min

from src.modeling.weibull import (
    fit_weibull,
    failure_probability,
    remaining_useful_life,
    run_weibull_analysis,
)
from src.utils.config import TA_CFG


class TestFitWeibull:
    def test_recovers_known_parameters(self):
        """Fit against a large synthetic sample with known (beta, eta)."""
        true_beta, true_eta = 2.5, 1000.0
        rng = np.random.default_rng(123)
        sample = weibull_min.rvs(true_beta, scale=true_eta, size=5000, random_state=rng)

        beta_hat, eta_hat = fit_weibull(sample)

        # MLE on 5000 points should be within ~10% of truth
        assert abs(beta_hat - true_beta) / true_beta < 0.10
        assert abs(eta_hat - true_eta) / true_eta < 0.10

    def test_handles_too_few_observations(self):
        """With < 3 points, falls back to exponential (beta=1)."""
        beta_hat, eta_hat = fit_weibull(np.array([500.0, 600.0]))
        assert beta_hat == 1.0
        assert eta_hat > 0

    def test_handles_empty_array(self):
        beta_hat, eta_hat = fit_weibull(np.array([]))
        assert beta_hat == 1.0
        assert eta_hat > 0

    def test_filters_nonpositive_values(self):
        """Negative/zero TTFs should be dropped before fitting."""
        rng = np.random.default_rng(7)
        sample = weibull_min.rvs(2.0, scale=800.0, size=200, random_state=rng)
        contaminated = np.concatenate([sample, [-10, 0, -5]])
        beta_hat, eta_hat = fit_weibull(contaminated)
        # Should still produce a sane positive fit, unaffected by the negatives
        assert beta_hat > 0 and eta_hat > 0

    def test_filters_nonfinite_values_regression(self):
        """
        Regression test: a single np.inf in the input array used to survive
        the old `times > 0` filter (inf > 0 is True) and corrupt BOTH the
        primary fit and the fallback mean calculation, producing eta=inf.
        Caught while writing this test suite — guards against reappearing.
        """
        rng = np.random.default_rng(3)
        sample = weibull_min.rvs(2.2, scale=900.0, size=50, random_state=rng)
        contaminated = np.concatenate([sample, [np.inf, -np.inf, np.nan]])
        beta_hat, eta_hat = fit_weibull(contaminated)
        assert np.isfinite(beta_hat), "beta must be finite even with inf/nan contamination"
        assert np.isfinite(eta_hat), "eta must be finite even with inf/nan contamination"
        assert eta_hat > 0

    def test_degenerate_fit_triggers_exception_fallback(self):
        """
        An array dominated by infinite values (after filtering, too few real
        observations remain) must hit the documented exponential fallback
        path rather than propagate a non-finite parameter.
        """
        # Only 2 finite, positive observations after filtering -> < 3 branch
        contaminated = np.array([np.inf, np.inf, np.inf, 500.0, 600.0])
        beta_hat, eta_hat = fit_weibull(contaminated)
        assert beta_hat == 1.0  # documented exponential fallback shape
        assert eta_hat == pytest.approx(550.0)  # mean of the 2 surviving values


class TestFailureProbability:
    def test_zero_horizon_gives_zero_probability(self):
        p = failure_probability(age=500, beta=2.0, eta=1000.0, horizon=0)
        assert p == pytest.approx(0.0, abs=1e-9)

    def test_probability_bounded_zero_one(self):
        for age in [0, 100, 1000, 10000]:
            p = failure_probability(age=age, beta=2.0, eta=1000.0, horizon=365)
            assert 0.0 <= p <= 1.0

    def test_probability_increases_with_age(self):
        """Older assets (closer to wear-out) should show higher near-term risk
        for a wear-out regime (beta > 1)."""
        p_young = failure_probability(age=100, beta=3.0, eta=2000.0, horizon=365)
        p_old = failure_probability(age=2500, beta=3.0, eta=2000.0, horizon=365)
        assert p_old > p_young

    def test_negative_age_treated_as_zero(self):
        p_neg = failure_probability(age=-50, beta=2.0, eta=1000.0, horizon=365)
        p_zero = failure_probability(age=0, beta=2.0, eta=1000.0, horizon=365)
        assert p_neg == pytest.approx(p_zero)

    def test_matches_closed_form_conditional_cdf(self):
        """Cross-check against the textbook conditional-failure formula."""
        age, horizon, beta, eta = 400.0, 200.0, 2.2, 1500.0
        F_a = weibull_min.cdf(age, beta, scale=eta)
        F_ah = weibull_min.cdf(age + horizon, beta, scale=eta)
        expected = (F_ah - F_a) / (1 - F_a)
        actual = failure_probability(age, beta, eta, horizon)
        assert actual == pytest.approx(expected, rel=1e-6)


class TestHorizonDefaultResolvesLiveConfig:
    """
    Regression tests for a real, confirmed bug: failure_probability() and
    run_weibull_analysis() used to declare `horizon: float = TA_CFG.planning_horizon_days`
    directly in the function signature. Python evaluates that default
    exactly ONCE, at import time — so a later `--horizon-days` CLI override
    (or any TA_CFG.planning_horizon_days mutation) was silently ignored.
    Empirically confirmed: running the full CLI with --horizon-days 30
    produced bit-for-bit identical failure_prob values to the 365-day
    default. The fix resolves TA_CFG.planning_horizon_days from inside the
    function body via a None sentinel instead. Every test below mutates
    TA_CFG mid-test (exactly what the CLI flag does) and confirms the very
    next call picks up the change.
    """

    def test_failure_probability_omitted_horizon_resolves_live_config(self):
        original = TA_CFG.planning_horizon_days
        try:
            TA_CFG.planning_horizon_days = 30
            p_short = failure_probability(age=500, beta=2.0, eta=1000.0)  # horizon omitted
            TA_CFG.planning_horizon_days = 3000
            p_long = failure_probability(age=500, beta=2.0, eta=1000.0)  # horizon omitted
            assert p_short != p_long
            assert p_short < p_long  # a longer horizon must never show LOWER failure probability
        finally:
            TA_CFG.planning_horizon_days = original

    def test_failure_probability_explicit_horizon_overrides_config(self):
        original = TA_CFG.planning_horizon_days
        try:
            TA_CFG.planning_horizon_days = 999  # should be ignored below
            p = failure_probability(age=500, beta=2.0, eta=1000.0, horizon=30)
            p_direct = failure_probability(age=500, beta=2.0, eta=1000.0, horizon=30)
            assert p == p_direct
        finally:
            TA_CFG.planning_horizon_days = original

    def test_run_weibull_analysis_omitted_horizon_resolves_live_config(self):
        wos = pd.DataFrame(
            {
                "wo_id": ["WO-1", "WO-2"],
                "asset_class": ["PMP", "PMP"],
                "age_days": [500.0, 600.0],
            }
        )
        rng = np.random.default_rng(1)
        failures = pd.DataFrame(
            {
                "asset_class": ["PMP"] * 50,
                "time_to_failure_d": weibull_min.rvs(2.0, scale=1000.0, size=50, random_state=rng),
            }
        )
        original = TA_CFG.planning_horizon_days
        try:
            TA_CFG.planning_horizon_days = 10
            out_short = run_weibull_analysis(wos, failures)  # horizon_days omitted
            TA_CFG.planning_horizon_days = 5000
            out_long = run_weibull_analysis(wos, failures)  # horizon_days omitted
            assert not out_short["failure_prob"].equals(out_long["failure_prob"])
            assert (out_short["failure_prob"] <= out_long["failure_prob"]).all()
        finally:
            TA_CFG.planning_horizon_days = original


class TestRemainingUsefulLife:
    def test_rul_is_nonnegative(self):
        rul = remaining_useful_life(age=500, beta=2.0, eta=1000.0)
        assert rul >= 0

    def test_rul_decreases_with_age(self):
        rul_young = remaining_useful_life(age=100, beta=2.5, eta=1500.0)
        rul_old = remaining_useful_life(age=1200, beta=2.5, eta=1500.0)
        assert rul_old < rul_young

    def test_rul_at_characteristic_life_matches_reliability_definition(self):
        """
        At t = eta, reliability R(eta) = exp(-1) ≈ 0.368.
        So RUL to reach 36.8% reliability, starting from age=0, should be ≈ eta.
        """
        eta = 1000.0
        rul = remaining_useful_life(age=0, beta=2.0, eta=eta, reliability_target=np.exp(-1))
        assert rul == pytest.approx(eta, rel=0.02)

    def test_rul_falls_back_to_eta_when_target_unreachable(self):
        """
        An impossible reliability_target (>1, since reliability is bounded
        [0,1]) means sf(t) - target never crosses zero anywhere in the
        brentq bracket. The function must catch that and fall back to eta
        (characteristic life) rather than raising or returning garbage.
        """
        rul = remaining_useful_life(age=100, beta=2.0, eta=1000.0, reliability_target=1.5)
        assert rul == pytest.approx(1000.0)


class TestRunWeibullAnalysisBatch:
    """
    Direct tests for the batch orchestrator (`run_weibull_analysis`), which
    contains real branching logic — per-class MLE fitting, asset-inline
    parameter preference, and a fallback path for classes with no failure
    history at all. This logic previously had only indirect coverage via
    the full pipeline / notebooks; these tests pin it down directly.
    """

    @staticmethod
    def _make_wos(classes, n_per_class=5, with_inline_weibull=True):
        rows = []
        rng = np.random.default_rng(0)
        i = 0
        for cls in classes:
            for _ in range(n_per_class):
                row = {
                    "wo_id": f"WO-{i:04d}",
                    "asset_class": cls,
                    "age_days": float(rng.uniform(100, 2000)),
                }
                if with_inline_weibull:
                    row["weibull_beta"] = 2.0
                    row["weibull_eta"] = 1500.0
                rows.append(row)
                i += 1
        return pd.DataFrame(rows)

    @staticmethod
    def _make_failures(classes, beta=2.2, eta=1200.0, n=200, seed=1):
        rng = np.random.default_rng(seed)
        rows = []
        for cls in classes:
            times = weibull_min.rvs(beta, scale=eta, size=n, random_state=rng)
            for t in times:
                rows.append({"asset_class": cls, "time_to_failure_d": float(t)})
        return pd.DataFrame(rows)

    def test_adds_all_expected_columns(self):
        wos = self._make_wos(["PMP", "HX"])
        failures = self._make_failures(["PMP", "HX"])
        out = run_weibull_analysis(wos, failures, horizon_days=365)
        for col in ["fitted_beta", "fitted_eta", "failure_prob", "rul_days", "weibull_source"]:
            assert col in out.columns

    def test_no_missing_failure_probabilities(self):
        """Every work order must end up with a usable failure_prob — including
        classes that have plenty of failure history AND classes that don't."""
        wos = self._make_wos(["PMP", "HX", "GHOST_CLASS"])
        failures = self._make_failures(["PMP", "HX"])  # GHOST_CLASS has NO history
        out = run_weibull_analysis(wos, failures, horizon_days=365)
        assert not out["failure_prob"].isna().any()
        assert not out["fitted_beta"].isna().any()

    def test_class_with_no_failure_history_uses_inline_fallback(self):
        """A class entirely absent from failure_history must fall back to the
        asset's inline Weibull params rather than crash or produce NaN."""
        wos = self._make_wos(["NEVER_FAILED"], with_inline_weibull=True)
        failures = self._make_failures(["PMP"])  # unrelated class only
        out = run_weibull_analysis(wos, failures, horizon_days=365)
        assert (out["weibull_source"] == "asset_inline_fallback").all()
        assert (out["fitted_beta"] == 2.0).all()
        assert (out["fitted_eta"] == 1500.0).all()

    def test_prefers_asset_inline_params_when_present(self):
        """When a WO carries its own inline weibull_beta/eta (more precise,
        asset-specific), it should be preferred over the coarser class-level fit."""
        wos = self._make_wos(["PMP"], with_inline_weibull=True)
        failures = self._make_failures(["PMP"], beta=3.5, eta=900.0)  # different from inline (2.0, 1500)
        out = run_weibull_analysis(wos, failures, horizon_days=365)
        assert (out["weibull_source"] == "asset_inline").all()
        assert (out["fitted_beta"] == 2.0).all()  # inline value wins, not the 3.5 class fit

    def test_failure_prob_within_valid_bounds(self):
        wos = self._make_wos(["PMP", "HX", "VLV"], n_per_class=15)
        failures = self._make_failures(["PMP", "HX", "VLV"])
        out = run_weibull_analysis(wos, failures, horizon_days=365)
        assert out["failure_prob"].between(0, 1).all()
        assert (out["rul_days"] >= 0).all()

    def test_pure_class_fit_when_no_inline_columns_exist(self):
        """
        Third branch of weibull_source: a WO table with NO weibull_beta/eta
        columns at all (common for a real CMMS export that never carried
        per-asset reliability params) must fall back cleanly to the pure
        class-level fit — distinct from both 'asset_inline' and
        'asset_inline_fallback'.
        """
        wos = self._make_wos(["PMP"], with_inline_weibull=False)
        assert "weibull_beta" not in wos.columns
        failures = self._make_failures(["PMP"], beta=2.8, eta=1100.0, n=300)
        out = run_weibull_analysis(wos, failures, horizon_days=365)
        assert (out["weibull_source"] == "class_fit").all()
        # The class-level MLE fit should land reasonably close to the true generating params
        assert out["fitted_beta"].iloc[0] == pytest.approx(2.8, rel=0.25)
        assert out["fitted_eta"].iloc[0] == pytest.approx(1100.0, rel=0.15)
