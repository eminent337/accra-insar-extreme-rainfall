"""
Unit Tests: Extreme Value Theory (EVT) Module
=============================================
"""

import math
import numpy as np
import pandas as pd
import pytest

from accra_engine.evt_extremes import (
    EVTExtremesEngine,
    GPDFitResult,
    GoodnessOfFitResult,
    POTResult,
    ReturnLevelResult,
)


@pytest.fixture
def synthetic_extreme_series():
    """Generate 10 years of daily rainfall with known GPD tail."""
    rng = np.random.default_rng(42)
    n = 3652
    base = rng.gamma(shape=1.2, scale=8.0, size=n)

    # Inject extreme Pareto tail above u=35 mm (sigma=20.0, xi=0.20)
    u = 35.0
    true_sigma = 20.0
    true_xi = 0.20
    extreme_mask = rng.random(n) < 0.05
    n_ext = int(extreme_mask.sum())
    unifs = rng.uniform(0.01, 0.99, n_ext)
    excesses = (true_sigma / true_xi) * ((1.0 - unifs) ** (-true_xi) - 1.0)
    base[extreme_mask] = u + excesses

    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    return pd.Series(base, index=dates)


def test_mrl_curve_computation(synthetic_extreme_series):
    evt = EVTExtremesEngine()
    mrl_df = evt.compute_mrl_curve(synthetic_extreme_series, min_threshold=15.0, max_threshold=50.0, num_thresholds=15)

    assert isinstance(mrl_df, pd.DataFrame)
    assert len(mrl_df) > 5
    assert "mean_excess" in mrl_df.columns
    assert "se" in mrl_df.columns
    assert (mrl_df["mean_excess"] > 0).all()
    assert (mrl_df["ci_lower_95"] <= mrl_df["mean_excess"]).all()
    assert (mrl_df["ci_upper_95"] >= mrl_df["mean_excess"]).all()


def test_suggest_optimal_threshold(synthetic_extreme_series):
    evt = EVTExtremesEngine()
    u_opt = evt.suggest_optimal_threshold(synthetic_extreme_series)
    assert isinstance(u_opt, float)
    assert u_opt >= 10.0


def test_pot_extraction_and_declustering():
    dates = pd.date_range("2021-06-01", periods=10, freq="D")
    # Consecutive storm days: Day 2=60mm, Day 3=85mm (same storm)
    rain = np.array([5.0, 60.0, 85.0, 2.0, 0.0, 70.0, 0.0, 0.0, 50.0, 55.0])
    s = pd.Series(rain, index=dates)

    evt = EVTExtremesEngine()
    pot_raw = evt.extract_pot(s, threshold=30.0, decluster_run_length=1)
    assert pot_raw.num_exceedances == 5

    # With run length declustering of 2 days:
    # Days 2 and 3 cluster -> peak 85.0
    # Day 6 alone -> 70.0
    # Days 9 and 10 cluster -> peak 55.0
    pot_declust = evt.extract_pot(s, threshold=30.0, decluster_run_length=2)
    assert pot_declust.num_exceedances == 3
    assert 85.0 in pot_declust.exceedance_values
    assert 70.0 in pot_declust.exceedance_values
    assert 55.0 in pot_declust.exceedance_values


def test_gpd_maximum_likelihood_fit():
    # Generate synthetic excesses from GPD with known sigma=18.0, xi=0.15
    rng = np.random.default_rng(101)
    k = 400
    true_sigma = 18.0
    true_xi = 0.15
    u_vals = rng.uniform(0.001, 0.999, k)
    excesses = (true_sigma / true_xi) * ((1.0 - u_vals) ** (-true_xi) - 1.0)

    evt = EVTExtremesEngine()
    fit = evt._fit_gpd_mle(excesses, threshold=25.0)

    assert isinstance(fit, GPDFitResult)
    assert fit.convergence is True
    # Estimated parameters should be close to true values (within 2 standard errors)
    assert abs(fit.scale - true_sigma) < 3.0
    assert abs(fit.shape - true_xi) < 0.15
    assert fit.scale_se > 0.0
    assert fit.shape_se > 0.0
    assert fit.cov_matrix.shape == (2, 2)
    assert fit.cov_matrix[0, 0] > 0.0  # Var(sigma) > 0
    assert fit.cov_matrix[1, 1] > 0.0  # Var(xi) > 0


def test_return_level_computation(synthetic_extreme_series):
    evt = EVTExtremesEngine(observations_per_year=365.25)
    pot = evt.extract_pot(synthetic_extreme_series, threshold=25.0)
    fit = evt.fit_gpd(pot)

    periods = [5.0, 10.0, 25.0, 50.0, 100.0]
    rl = evt.compute_return_levels(return_periods=periods, fit_result=fit, pot_result=pot)

    assert isinstance(rl, ReturnLevelResult)
    assert len(rl.return_levels) == 5

    # Return levels must be strictly monotonic increasing
    for i in range(len(rl.return_levels) - 1):
        assert rl.return_levels[i] < rl.return_levels[i + 1]

    # Confidence interval must bracket return level
    for i in range(len(rl.return_levels)):
        assert rl.ci_lower[i] <= rl.return_levels[i] <= rl.ci_upper[i]
        assert rl.standard_errors[i] > 0.0


def test_goodness_of_fit_ks_and_ad():
    rng = np.random.default_rng(202)
    excesses = rng.exponential(scale=15.0, size=200)

    evt = EVTExtremesEngine()
    fit = evt._fit_gpd_mle(excesses, threshold=20.0)

    pot_mock = POTResult(
        threshold=20.0,
        exceedances=excesses,
        exceedance_values=excesses + 20.0,
        exceedance_dates=None,
        total_observations=2000,
        num_exceedances=200,
        exceedance_rate=0.10,
        observations_per_year=365.25,
    )

    gof = evt.test_goodness_of_fit(fit, pot_mock, n_bootstraps=50)

    assert isinstance(gof, GoodnessOfFitResult)
    assert 0.0 <= gof.ks_statistic <= 1.0
    assert 0.0 <= gof.ks_pvalue <= 1.0
    assert gof.ad_statistic >= 0.0
    assert 0.0 <= gof.ad_pvalue <= 1.0
