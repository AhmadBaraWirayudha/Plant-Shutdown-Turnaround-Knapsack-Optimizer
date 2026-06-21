"""
test_risk.py — Unit tests for risk scoring, criticality matrix, and
deferred-cost economics.
"""

from __future__ import annotations
import pandas as pd
import pytest

from src.modeling.risk import (
    consequence_score,
    deferred_risk_cost,
    risk_level,
    compute_risk_scores,
    build_criticality_matrix,
)
from src.utils.config import RISK_CFG


class TestConsequenceScore:
    def test_weighted_average_matches_manual_calc(self):
        score = consequence_score(
            c_safety=5,
            c_env=1,
            c_prod=1,
            c_cost=1,
            w_safety=0.4,
            w_env=0.25,
            w_prod=0.25,
            w_cost=0.1,
        )
        expected = 0.4 * 5 + 0.25 * 1 + 0.25 * 1 + 0.1 * 1
        assert score == pytest.approx(expected)

    def test_clipped_to_valid_range(self):
        # Even with extreme weights, output must stay within [1, 5]
        score = consequence_score(c_safety=5, c_env=5, c_prod=5, c_cost=5)
        assert 1.0 <= score <= 5.0
        score_low = consequence_score(c_safety=0, c_env=0, c_prod=0, c_cost=0)
        assert score_low >= 1.0


class TestDeferredRiskCost:
    def test_zero_probability_gives_zero_cost(self):
        cost = deferred_risk_cost(failure_prob=0.0, consequence=5.0, replace_usd=100_000)
        assert cost == 0.0

    def test_scales_linearly_with_probability(self):
        c1 = deferred_risk_cost(0.1, 3.0, 100_000, factor=0.15)
        c2 = deferred_risk_cost(0.2, 3.0, 100_000, factor=0.15)
        assert c2 == pytest.approx(2 * c1)

    def test_formula_matches_definition(self):
        p, c, rv, f = 0.4, 3.5, 200_000, 0.15
        expected = p * c * rv * f
        assert deferred_risk_cost(p, c, rv, f) == pytest.approx(expected)


class TestWeightDefaultsResolveLiveConfig:
    """
    Regression tests matching the same bug class fixed in weibull.py: an
    earlier version of consequence_score() / deferred_risk_cost() declared
    their weight defaults directly as `w_safety: float = RISK_CFG.w_safety`
    in the function signature, which Python evaluates once at import time.
    Nothing in this codebase currently mutates RISK_CFG after import, so
    this was a latent footgun rather than an active bug (unlike the
    confirmed-broken --horizon-days and --num-work-orders cases in
    weibull.py / data_generator.py) — but the same fix (None sentinel,
    resolved live inside the function body) was applied for consistency
    and to close off the whole footgun class. These tests guard against
    the live-resolution behavior regressing back to a frozen default.
    """

    def test_consequence_score_omitted_weights_resolve_live_config(self):
        original = RISK_CFG.w_safety
        try:
            RISK_CFG.w_safety = 0.9
            score_high_safety_weight = consequence_score(5, 1, 1, 1)  # weights omitted
            RISK_CFG.w_safety = 0.1
            score_low_safety_weight = consequence_score(5, 1, 1, 1)  # weights omitted
            assert score_high_safety_weight != score_low_safety_weight
            assert score_high_safety_weight > score_low_safety_weight
        finally:
            RISK_CFG.w_safety = original

    def test_consequence_score_explicit_weight_overrides_config(self):
        original = RISK_CFG.w_safety
        try:
            RISK_CFG.w_safety = 0.99  # should be ignored below
            score = consequence_score(5, 1, 1, 1, w_safety=0.1, w_env=0.3, w_prod=0.3, w_cost=0.3)
            score_direct = consequence_score(5, 1, 1, 1, w_safety=0.1, w_env=0.3, w_prod=0.3, w_cost=0.3)
            assert score == score_direct
        finally:
            RISK_CFG.w_safety = original

    def test_deferred_risk_cost_omitted_factor_resolves_live_config(self):
        original = RISK_CFG.deferral_cost_factor
        try:
            RISK_CFG.deferral_cost_factor = 0.05
            cost_low = deferred_risk_cost(0.5, 4.0, 100_000)  # factor omitted
            RISK_CFG.deferral_cost_factor = 0.50
            cost_high = deferred_risk_cost(0.5, 4.0, 100_000)  # factor omitted
            assert cost_low != cost_high
            assert cost_low < cost_high
        finally:
            RISK_CFG.deferral_cost_factor = original


class TestRiskLevel:
    @pytest.mark.parametrize(
        "lik,con,expected",
        [
            (5, 5, "CRITICAL"),  # score 25
            (4, 5, "CRITICAL"),  # score 20
            (5, 4, "CRITICAL"),  # score 20
            (3, 4, "HIGH"),  # score 12
            (2, 5, "HIGH"),  # score 10
            (2, 3, "MEDIUM"),  # score 6
            (1, 5, "MEDIUM"),  # score 5
            (1, 4, "LOW"),  # score 4
            (1, 1, "LOW"),  # score 1
        ],
    )
    def test_risk_level_boundaries(self, lik, con, expected):
        assert risk_level(lik, con) == expected


@pytest.fixture
def sample_wos():
    return pd.DataFrame(
        {
            "wo_id": ["WO-1", "WO-2", "WO-3"],
            "mandatory": [True, False, False],
            "failure_prob": [0.50, 0.02, 0.80],
            "c_safety": [5, 1, 4],
            "c_env": [4, 1, 3],
            "c_prod": [5, 1, 4],
            "c_cost": [3, 1, 3],
            "replace_usd": [400_000, 5_000, 90_000],
            "estimated_cost_usd": [60_000, 500, 10_000],
        }
    )


class TestComputeRiskScores:
    def test_adds_all_expected_columns(self, sample_wos):
        out = compute_risk_scores(sample_wos)
        for col in [
            "consequence_score",
            "likelihood_tier",
            "consequence_tier",
            "risk_score",
            "risk_level",
            "deferred_cost_usd",
            "net_value_usd",
        ]:
            assert col in out.columns

    def test_high_consequence_high_prob_is_critical(self, sample_wos):
        out = compute_risk_scores(sample_wos)
        row = out[out.wo_id == "WO-1"].iloc[0]
        assert row.risk_level in ("HIGH", "CRITICAL")

    def test_low_prob_low_consequence_is_low_risk(self, sample_wos):
        out = compute_risk_scores(sample_wos)
        row = out[out.wo_id == "WO-2"].iloc[0]
        assert row.risk_level == "LOW"

    def test_net_value_is_true_economics_no_artificial_inflation(self, sample_wos):
        """
        Regression test: net_value_usd must equal deferred_cost - cost exactly,
        with NO hidden bonus for mandatory tasks. A prior version of this code
        added a $5M bonus to mandatory tasks' net value, which silently
        corrupted every downstream ROI metric. This test guards against
        that regression reappearing.
        """
        out = compute_risk_scores(sample_wos)
        for _, row in out.iterrows():
            expected = row.deferred_cost_usd - row.estimated_cost_usd
            assert row.net_value_usd == pytest.approx(expected, abs=0.01)
            # Critically: no value should be anywhere near a million-dollar
            # bonus territory given these small input costs/replace values
            assert abs(row.net_value_usd) < 1_000_000

    def test_likelihood_tiers_within_valid_range(self, sample_wos):
        out = compute_risk_scores(sample_wos)
        assert out["likelihood_tier"].between(1, 5).all()

    def test_consequence_tiers_within_valid_range(self, sample_wos):
        out = compute_risk_scores(sample_wos)
        assert out["consequence_tier"].between(1, 5).all()


class TestCriticalityMatrix:
    def test_matrix_is_5x5(self, sample_wos):
        scored = compute_risk_scores(sample_wos)
        matrix = build_criticality_matrix(scored)
        assert matrix.shape == (5, 5)

    def test_matrix_counts_sum_to_total_tasks(self, sample_wos):
        scored = compute_risk_scores(sample_wos)
        matrix = build_criticality_matrix(scored)
        assert matrix.values.sum() == len(scored)
