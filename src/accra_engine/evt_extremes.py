"""
Extreme Value Theory (EVT) Module: Accra Hydro-InSAR Engine
============================================================
Rigorous implementation of Peaks-Over-Threshold (POT) sampling,
Mean Residual Life (MRL) plot analysis, Generalized Pareto Distribution (GPD)
Maximum Likelihood Estimation (MLE), delta-method return levels (5, 10, 25, 50, 100-yr),
and Anderson-Darling & Kolmogorov-Smirnov goodness-of-fit testing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Results Data Classes
# ---------------------------------------------------------------------------

@dataclass
class POTResult:
    """Peaks-Over-Threshold sampling result."""
    threshold: float
    exceedances: np.ndarray  # Excesses: y = x - u
    exceedance_values: np.ndarray  # Original values x > u
    exceedance_dates: Optional[pd.DatetimeIndex]
    total_observations: int
    num_exceedances: int
    exceedance_rate: float  # zeta_u = k / N
    observations_per_year: float  # n_y (typically 365.25)


@dataclass
class GPDFitResult:
    """Fitted Generalized Pareto Distribution parameterization."""
    threshold: float
    scale: float  # sigma > 0
    shape: float  # xi (Frechet if >0, Gumbel if =0, Weibull if <0)
    scale_se: float
    shape_se: float
    cov_matrix: np.ndarray  # 2x2 covariance matrix [ [Var(sigma), Cov], [Cov, Var(xi)] ]
    log_likelihood: float
    aic: float
    bic: float
    convergence: bool
    num_exceedances: int


@dataclass
class ReturnLevelResult:
    """Return level estimates across multiple return periods."""
    return_periods: List[float]  # in years (e.g. [5, 10, 25, 50, 100])
    return_levels: List[float]  # x_T in mm
    ci_lower: List[float]  # 95% CI lower bound
    ci_upper: List[float]  # 95% CI upper bound
    standard_errors: List[float]
    summary_df: pd.DataFrame


@dataclass
class GoodnessOfFitResult:
    """Statistical goodness-of-fit tests for the GPD threshold model."""
    ks_statistic: float
    ks_pvalue: float
    ad_statistic: float
    ad_pvalue: float  # Parametric bootstrap or empirical approximation
    passed_at_5pct: bool
    details: Dict[str, Union[float, str, bool]]


# ---------------------------------------------------------------------------
# Core EVT Extremes Engine
# ---------------------------------------------------------------------------

class EVTExtremesEngine:
    """
    Production-grade Extreme Value Theory (EVT) engine for hydrological downpours.
    Implements POT-GPD with Fisher information covariance and delta-method return levels.
    """

    def __init__(self, observations_per_year: float = 365.25) -> None:
        self.obs_per_year = observations_per_year
        self.fitted_gpd: Optional[GPDFitResult] = None
        self.pot_data: Optional[POTResult] = None

    # -----------------------------------------------------------------------
    # 1. Mean Residual Life (MRL) Computation
    # -----------------------------------------------------------------------
    def compute_mrl_curve(
        self,
        series: Union[np.ndarray, pd.Series],
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
        num_thresholds: int = 40,
    ) -> pd.DataFrame:
        """
        Compute the Mean Residual Life (MRL) plot:
        e(u) = E[X - u | X > u] across candidate thresholds u.
        For a valid GPD threshold, e(u) should be linear with slope xi / (1 - xi).
        """
        data = np.asarray(series)
        data = data[np.isfinite(data)]
        rainy_data = data[data > 0.1]  # rainy days

        if min_threshold is None:
            min_threshold = float(np.percentile(rainy_data, 75))
        if max_threshold is None:
            max_threshold = float(np.percentile(rainy_data, 98.5))

        thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
        mrl_records = []

        for u in thresholds:
            excesses = data[data > u] - u
            k = len(excesses)
            if k < 10:
                continue

            mean_excess = float(np.mean(excesses))
            std_excess = float(np.std(excesses, ddof=1))
            se = std_excess / math.sqrt(k)
            ci_low = max(0.0, mean_excess - 1.96 * se)
            ci_high = mean_excess + 1.96 * se

            mrl_records.append({
                "threshold": round(float(u), 3),
                "num_exceedances": k,
                "mean_excess": round(mean_excess, 3),
                "se": round(se, 3),
                "ci_lower_95": round(ci_low, 3),
                "ci_upper_95": round(ci_high, 3),
            })

        return pd.DataFrame(mrl_records)

    def suggest_optimal_threshold(
        self,
        series: Union[np.ndarray, pd.Series],
        min_percentile: float = 90.0,
        max_percentile: float = 98.0,
        candidate_count: int = 15,
    ) -> float:
        """
        Select an optimal POT threshold by evaluating GPD fit stability
        and Anderson-Darling goodness-of-fit across candidate quantiles.
        """
        data = np.asarray(series)
        rainy = data[data > 0.5]

        candidates = np.percentile(rainy, np.linspace(min_percentile, max_percentile, candidate_count))
        best_u = float(candidates[len(candidates) // 2])
        best_score = float("inf")

        for u in candidates:
            excesses = data[data > u] - u
            if len(excesses) < 25:
                continue
            try:
                fit = self._fit_gpd_mle(excesses, u)
                if fit.convergence and fit.scale > 0:
                    # Evaluate goodness of fit via KS statistic
                    ks_stat, _ = self._compute_ks_test(excesses, fit.scale, fit.shape)
                    # Penalize too few exceedances or poor fit
                    score = ks_stat + 15.0 / math.sqrt(len(excesses))
                    if score < best_score:
                        best_score = score
                        best_u = float(u)
            except Exception:
                continue

        return round(best_u, 2)

    # -----------------------------------------------------------------------
    # 2. Peaks-Over-Threshold (POT) Extraction with Runs Declustering
    # -----------------------------------------------------------------------
    def extract_pot(
        self,
        series: Union[pd.Series, np.ndarray],
        threshold: float,
        dates: Optional[pd.DatetimeIndex] = None,
        decluster_run_length: int = 1,
    ) -> POTResult:
        """
        Extract extreme peak values exceeding threshold u.
        Optionally declusters dependent storms using run length (in days).
        """
        if isinstance(series, pd.Series):
            if dates is None and isinstance(series.index, pd.DatetimeIndex):
                dates = series.index
            data = series.to_numpy()
        else:
            data = np.asarray(series)

        total_obs = len(data)
        exceed_idx = np.where(data > threshold)[0]

        if len(exceed_idx) == 0:
            raise ValueError(f"No observations exceed threshold u = {threshold}")

        if decluster_run_length <= 1 or dates is None:
            # Independent observations assumption
            selected_idx = exceed_idx
        else:
            # Runs declustering: cluster consecutive exceedance days separated by < run_length days
            clusters: List[List[int]] = []
            curr_cluster = [exceed_idx[0]]

            for idx in exceed_idx[1:]:
                prev_date = dates[curr_cluster[-1]]
                curr_date = dates[idx]
                day_diff = (curr_date - prev_date).days
                if day_diff <= decluster_run_length:
                    curr_cluster.append(idx)
                else:
                    clusters.append(curr_cluster)
                    curr_cluster = [idx]
            clusters.append(curr_cluster)

            # Pick the peak within each storm cluster
            selected_idx = np.array([max(cl, key=lambda i: data[i]) for cl in clusters])

        excesses = data[selected_idx] - threshold
        original_vals = data[selected_idx]
        sel_dates = dates[selected_idx] if dates is not None else None

        k = len(excesses)
        zeta_u = k / total_obs

        pot_res = POTResult(
            threshold=threshold,
            exceedances=excesses,
            exceedance_values=original_vals,
            exceedance_dates=sel_dates,
            total_observations=total_obs,
            num_exceedances=k,
            exceedance_rate=zeta_u,
            observations_per_year=self.obs_per_year,
        )
        self.pot_data = pot_res
        return pot_res

    # -----------------------------------------------------------------------
    # 3. GPD Maximum Likelihood Estimation (MLE)
    # -----------------------------------------------------------------------
    def fit_gpd(
        self,
        pot_result: Optional[POTResult] = None,
        threshold: Optional[float] = None,
        series: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> GPDFitResult:
        """
        Fit Generalized Pareto Distribution using numerical Maximum Likelihood Estimation.
        Estimates scale sigma and shape xi, and derives the Fisher Information covariance matrix.
        """
        if pot_result is None:
            if series is None or threshold is None:
                raise ValueError("Either pot_result or (series and threshold) must be provided.")
            pot_result = self.extract_pot(series, threshold)

        self.pot_data = pot_result
        excesses = pot_result.exceedances
        u = pot_result.threshold

        fit_res = self._fit_gpd_mle(excesses, u)
        self.fitted_gpd = fit_res
        return fit_res

    def _fit_gpd_mle(self, excesses: np.ndarray, threshold: float) -> GPDFitResult:
        """Internal optimization routine for GPD MLE with analytical Hessian inversion."""
        k = len(excesses)
        y = excesses.astype(np.float64)
        mean_y = float(np.mean(y))
        var_y = float(np.var(y, ddof=1)) if k > 1 else 1.0

        # Method of Moments initial guesses:
        # mean = sigma / (1 - xi)
        # var = sigma^2 / [ (1 - xi)^2 * (1 - 2*xi) ]
        if var_y > 0 and mean_y > 0:
            xi_0 = 0.5 * (1.0 - (mean_y ** 2) / var_y)
            xi_0 = float(np.clip(xi_0, -0.4, 0.4))
            sigma_0 = float(max(1.0, 0.5 * mean_y * (1.0 + (mean_y ** 2) / var_y)))
        else:
            xi_0 = 0.1
            sigma_0 = max(1.0, mean_y)

        def neg_log_likelihood(params: np.ndarray) -> float:
            sigma, xi = params
            if sigma <= 1e-6:
                return 1e12

            # GPD domain condition: 1 + xi * y / sigma > 0 for all y
            arg = 1.0 + xi * y / sigma
            if np.any(arg <= 0):
                return 1e12

            if abs(xi) < 1e-5:
                # Exponential limit as xi -> 0
                nll = k * np.log(sigma) + np.sum(y) / sigma
            else:
                nll = k * np.log(sigma) + (1.0 + 1.0 / xi) * np.sum(np.log(arg))

            return float(nll)

        # Optimize using L-BFGS-B with bounds
        init_params = np.array([sigma_0, xi_0])
        bounds = [(1e-3, 500.0), (-0.8, 1.2)]
        opt_res = optimize.minimize(
            neg_log_likelihood,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
        )

        sigma_hat = float(opt_res.x[0])
        xi_hat = float(opt_res.x[1])
        nll_val = float(opt_res.fun)

        # Calculate observed Fisher Information Hessian via finite differences
        eps = 1e-4
        hess = np.zeros((2, 2), dtype=np.float64)

        f0 = neg_log_likelihood(np.array([sigma_hat, xi_hat]))
        f_sig_p = neg_log_likelihood(np.array([sigma_hat + eps, xi_hat]))
        f_sig_m = neg_log_likelihood(np.array([sigma_hat - eps, xi_hat]))
        f_xi_p = neg_log_likelihood(np.array([sigma_hat, xi_hat + eps]))
        f_xi_m = neg_log_likelihood(np.array([sigma_hat, xi_hat - eps]))

        f_diag_pp = neg_log_likelihood(np.array([sigma_hat + eps, xi_hat + eps]))
        f_diag_pm = neg_log_likelihood(np.array([sigma_hat + eps, xi_hat - eps]))
        f_diag_mp = neg_log_likelihood(np.array([sigma_hat - eps, xi_hat + eps]))
        f_diag_mm = neg_log_likelihood(np.array([sigma_hat - eps, xi_hat - eps]))

        hess[0, 0] = (f_sig_p - 2 * f0 + f_sig_m) / (eps ** 2)
        hess[1, 1] = (f_xi_p - 2 * f0 + f_xi_m) / (eps ** 2)
        hess[0, 1] = (f_diag_pp - f_diag_pm - f_diag_mp + f_diag_mm) / (4 * eps ** 2)
        hess[1, 0] = hess[0, 1]

        # Invert Hessian to obtain parameter covariance matrix
        cov_matrix = np.zeros((2, 2))
        try:
            cov_matrix = np.linalg.inv(hess)
            # Ensure positive variances on diagonal
            if cov_matrix[0, 0] < 0 or cov_matrix[1, 1] < 0:
                raise np.linalg.LinAlgError("Negative variance in Hessian inverse")
            sigma_se = float(math.sqrt(cov_matrix[0, 0]))
            xi_se = float(math.sqrt(cov_matrix[1, 1]))
        except Exception:
            # Asymptotic fallback for regular GPD MLE (Hosking & Wallis 1987)
            if xi_hat > -0.5:
                var_sigma = ((1 + xi_hat) ** 2 * (sigma_hat ** 2)) / k
                var_xi = ((1 + xi_hat) ** 2) / k
                cov_val = (1 + xi_hat) * sigma_hat / k
            else:
                var_sigma = (sigma_hat ** 2) / k
                var_xi = 1.0 / k
                cov_val = 0.0
            cov_matrix = np.array([[var_sigma, cov_val], [cov_val, var_xi]])
            sigma_se = float(math.sqrt(var_sigma))
            xi_se = float(math.sqrt(var_xi))

        aic = 2 * 2 + 2 * nll_val
        bic = 2 * math.log(k) + 2 * nll_val

        return GPDFitResult(
            threshold=threshold,
            scale=sigma_hat,
            shape=xi_hat,
            scale_se=sigma_se,
            shape_se=xi_se,
            cov_matrix=cov_matrix,
            log_likelihood=-nll_val,
            aic=aic,
            bic=bic,
            convergence=bool(opt_res.success),
            num_exceedances=k,
        )

    # -----------------------------------------------------------------------
    # 4. Return Level Curves with Delta-Method Confidence Intervals
    # -----------------------------------------------------------------------
    def compute_return_levels(
        self,
        return_periods: Optional[List[float]] = None,
        fit_result: Optional[GPDFitResult] = None,
        pot_result: Optional[POTResult] = None,
    ) -> ReturnLevelResult:
        """
        Compute T-year return levels x_T:
        x_T = u + (sigma / xi) * [ (n_y * T * zeta_u)^xi - 1 ]  (if xi != 0)
        x_T = u + sigma * ln(n_y * T * zeta_u)                (if xi == 0)

        Calculates standard errors and 95% confidence intervals via the Delta Method.
        """
        fit = fit_result or self.fitted_gpd
        pot = pot_result or self.pot_data

        if fit is None or pot is None:
            raise ValueError("Fitted GPD and POT data must be available.")

        if return_periods is None:
            return_periods = [5.0, 10.0, 25.0, 50.0, 100.0]

        u = fit.threshold
        sigma = fit.scale
        xi = fit.shape
        zeta = pot.exceedance_rate
        ny = pot.observations_per_year
        n_total = pot.total_observations

        # Variance of exceedance rate zeta_u (binomial proportion)
        var_zeta = (zeta * (1.0 - zeta)) / n_total

        # Combined 3x3 covariance matrix: [ [Var(zeta), 0, 0], [0, V_sigma, V_cov], [0, V_cov, V_xi] ]
        cov_3x3 = np.zeros((3, 3))
        cov_3x3[0, 0] = var_zeta
        cov_3x3[1:3, 1:3] = fit.cov_matrix

        levels: List[float] = []
        se_list: List[float] = []
        ci_low: List[float] = []
        ci_high: List[float] = []
        summary_rows = []

        for T in return_periods:
            m = ny * T * zeta  # Expected exceedances in T years

            if m <= 1.0:
                # When T is within or below threshold observation frequency
                x_T = u
                se_T = fit.scale_se
            elif abs(xi) < 1e-5:
                x_T = u + sigma * math.log(m)
                # Gradient: [dx/dzeta, dx/dsigma, dx/dxi]
                grad = np.array([
                    sigma / zeta,
                    math.log(m),
                    0.5 * sigma * (math.log(m) ** 2),
                ])
                var_T = float(grad @ cov_3x3 @ grad)
                se_T = math.sqrt(max(0.0, var_T))
            else:
                m_xi = m ** xi
                x_T = u + (sigma / xi) * (m_xi - 1.0)

                # Analytical gradients for Delta Method
                dx_dzeta = (sigma * (ny * T) ** xi * (zeta ** (xi - 1.0)))
                dx_dsigma = (m_xi - 1.0) / xi
                dx_dxi = (-sigma / (xi ** 2)) * (m_xi - 1.0) + (sigma / xi) * m_xi * math.log(m)

                grad = np.array([dx_dzeta, dx_dsigma, dx_dxi])
                var_T = float(grad @ cov_3x3 @ grad)
                se_T = math.sqrt(max(0.0, var_T))

            low = max(u, x_T - 1.96 * se_T)
            high = x_T + 1.96 * se_T

            levels.append(round(x_T, 2))
            se_list.append(round(se_T, 2))
            ci_low.append(round(low, 2))
            ci_high.append(round(high, 2))

            summary_rows.append({
                "return_period_years": T,
                "return_level_mm": round(x_T, 2),
                "se_mm": round(se_T, 2),
                "ci_lower_95_mm": round(low, 2),
                "ci_upper_95_mm": round(high, 2),
                "annual_exceedance_probability": round(1.0 / T, 4),
            })

        df_summary = pd.DataFrame(summary_rows)
        return ReturnLevelResult(
            return_periods=return_periods,
            return_levels=levels,
            ci_lower=ci_low,
            ci_upper=ci_high,
            standard_errors=se_list,
            summary_df=df_summary,
        )

    # -----------------------------------------------------------------------
    # 5. Goodness-of-Fit Tests (Kolmogorov-Smirnov & Anderson-Darling)
    # -----------------------------------------------------------------------
    def test_goodness_of_fit(
        self,
        fit_result: Optional[GPDFitResult] = None,
        pot_result: Optional[POTResult] = None,
        n_bootstraps: int = 400,
    ) -> GoodnessOfFitResult:
        """
        Evaluate statistical validity of the fitted GPD using:
        1. Kolmogorov-Smirnov test (maximum vertical distance between empirical and fitted CDF).
        2. Anderson-Darling test (heavier weighting on extreme upper tail).
        Uses a parametric bootstrap to derive exact finite-sample p-values.
        """
        fit = fit_result or self.fitted_gpd
        pot = pot_result or self.pot_data

        if fit is None or pot is None:
            raise ValueError("Fitted GPD and POT data must be available.")

        excesses = pot.exceedances
        k = len(excesses)
        sigma = fit.scale
        xi = fit.shape

        ks_stat, ks_pval = self._compute_ks_test(excesses, sigma, xi)
        ad_stat = self._compute_ad_test_statistic(excesses, sigma, xi)

        # Parametric bootstrap for Anderson-Darling p-value under estimated parameters
        rng = np.random.default_rng(101)
        sim_ad_stats = []

        for _ in range(n_bootstraps):
            # Sample synthetic excesses from fitted GPD
            u_unif = rng.uniform(1e-5, 1.0 - 1e-5, k)
            if abs(xi) < 1e-4:
                sim_excess = -sigma * np.log(1.0 - u_unif)
            else:
                sim_excess = (sigma / xi) * ((1.0 - u_unif) ** (-xi) - 1.0)

            # Fit GPD to simulated sample
            try:
                sim_fit = self._fit_gpd_mle(sim_excess, fit.threshold)
                stat_b = self._compute_ad_test_statistic(sim_excess, sim_fit.scale, sim_fit.shape)
                sim_ad_stats.append(stat_b)
            except Exception:
                continue

        if len(sim_ad_stats) > 50:
            ad_pval = float(np.mean(np.array(sim_ad_stats) >= ad_stat))
        else:
            # Fallback to Choulakian & Stephens (2001) empirical GPD critical threshold (~0.95 at 5% level)
            ad_pval = 0.50 if ad_stat < 0.95 else 0.02

        passed = bool(ks_pval >= 0.05 and ad_pval >= 0.05)

        return GoodnessOfFitResult(
            ks_statistic=round(ks_stat, 4),
            ks_pvalue=round(ks_pval, 4),
            ad_statistic=round(ad_stat, 4),
            ad_pvalue=round(ad_pval, 4),
            passed_at_5pct=passed,
            details={
                "num_exceedances": k,
                "fitted_scale": round(sigma, 3),
                "fitted_shape": round(xi, 4),
                "ad_bootstrap_samples": len(sim_ad_stats),
                "threshold_u": fit.threshold,
            },
        )

    def _gpd_cdf(self, y: np.ndarray, sigma: float, xi: float) -> np.ndarray:
        """GPD cumulative distribution function."""
        y_safe = np.maximum(0.0, y)
        if abs(xi) < 1e-5:
            return 1.0 - np.exp(-y_safe / sigma)
        arg = np.maximum(1e-12, 1.0 + xi * y_safe / sigma)
        return 1.0 - (arg ** (-1.0 / xi))

    def _compute_ks_test(self, excesses: np.ndarray, sigma: float, xi: float) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test against fitted GPD."""
        sorted_y = np.sort(excesses)
        k = len(sorted_y)
        emp_cdf = np.arange(1, k + 1) / k
        theo_cdf = self._gpd_cdf(sorted_y, sigma, xi)
        d_plus = np.max(emp_cdf - theo_cdf)
        d_minus = np.max(theo_cdf - (np.arange(0, k) / k))
        ks_stat = float(max(d_plus, d_minus))

        # Asymptotic KS p-value (Kolmogorov distribution)
        ks_pval = float(stats.kstwo.sf(ks_stat, k))
        return ks_stat, ks_pval

    def _compute_ad_test_statistic(self, excesses: np.ndarray, sigma: float, xi: float) -> float:
        """
        Anderson-Darling statistic:
        A^2 = -k - (1/k) * sum_{i=1}^k (2i - 1) * [ ln(z_i) + ln(1 - z_{k+1-i}) ]
        """
        sorted_y = np.sort(excesses)
        k = len(sorted_y)
        z = self._gpd_cdf(sorted_y, sigma, xi)
        # Avoid log(0)
        z = np.clip(z, 1e-8, 1.0 - 1e-8)

        i_vec = np.arange(1, k + 1)
        sum_terms = (2 * i_vec - 1) * (np.log(z) + np.log(1.0 - z[::-1]))
        ad_stat = -k - (1.0 / k) * np.sum(sum_terms)
        return float(ad_stat)
