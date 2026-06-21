"""
weibull.py — Weibull reliability analysis.

For each equipment class:
  1. Fit a two-parameter Weibull distribution to historical time-to-failure data
     using maximum-likelihood estimation (SciPy).
  2. Attach fitted parameters to every work order.
  3. Compute failure probability P(T ≤ planning_horizon) for each asset
     given its current age.
  4. Compute Remaining Useful Life (RUL) at the specified reliability target.

Weibull CDF:  F(t) = 1 - exp(-((t - γ) / η)^β)
  β = shape  (Weibull modulus) — "wear-in / random / wear-out" regime
  η = scale  (characteristic life) — 63.2 % of population fails by η
  γ = 0      (location, fixed at 0 → two-parameter model)
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from scipy.optimize import OptimizeWarning

from src.utils.config import TA_CFG
from src.utils.helpers import get_logger, timed

log = get_logger("modeling.weibull")

# Suppress ill-conditioned fit warnings during batch fitting
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─── Fitting ──────────────────────────────────────────────────────────────────


def fit_weibull(times: np.ndarray) -> tuple[float, float]:
    """
    Fit a two-parameter Weibull model to an array of time-to-failure values.

    Returns
    -------
    (beta, eta) : shape and scale parameters

    Falls back to empirical (beta=1, eta=mean) if there are < 3 observations
    or the optimiser fails to converge.
    """
    times = np.asarray(times, dtype=float)
    # Remove zero/negative TTFs AND non-finite values (inf/-inf/nan). A single
    # corrupted timestamp (e.g. a CMMS export defect producing inf) must not
    # silently poison either the primary MLE fit OR the fallback mean — both
    # operate on this same cleaned array.
    times = times[np.isfinite(times) & (times > 0)]

    if len(times) < 3:
        eta_fallback = float(np.mean(times)) if len(times) > 0 else 1825.0
        log.debug("Too few observations (%d) – using exponential fallback", len(times))
        return 1.0, eta_fallback

    try:
        # floc=0 → fix location at 0 (two-parameter model)
        shape, _, scale = weibull_min.fit(times, floc=0)
        # Sanity check: shape and scale must be positive finite
        if not (np.isfinite(shape) and shape > 0 and np.isfinite(scale) and scale > 0):
            raise ValueError("Non-positive Weibull parameters")
        return float(shape), float(scale)
    except Exception as exc:
        log.debug("Weibull fit failed (%s) – using empirical fallback", exc)
        return 1.0, float(np.mean(times))


# ─── Probability & RUL ───────────────────────────────────────────────────────


def failure_probability(age: float, beta: float, eta: float, horizon: float | None = None) -> float:
    """
    Probability of first failure within `horizon` days starting from `age` days.

    P(T ≤ age + horizon | T > age)  [conditional on survival to current age]
    = [F(age + horizon) - F(age)] / [1 - F(age)]

    Clamps to [0, 1].

    `horizon` defaults to `None`, resolved against the LIVE value of
    TA_CFG.planning_horizon_days inside this function body — NOT baked in
    as a Python default-argument value at import time. A default value of
    `TA_CFG.planning_horizon_days` directly in the signature would freeze
    whatever that value happened to be the moment this module was first
    imported, silently ignoring any later `--horizon-days` CLI override or
    TA_CFG mutation. This was a real, confirmed bug during development —
    see docs/METHODOLOGY.md §1 and tests/test_weibull.py's regression test.
    """
    if horizon is None:
        horizon = TA_CFG.planning_horizon_days
    if age <= 0:
        age = 0.0
    F_age = weibull_min.cdf(age, beta, scale=eta, loc=0)
    F_age_horizon = weibull_min.cdf(age + horizon, beta, scale=eta, loc=0)
    survival = 1.0 - F_age

    if survival < 1e-9:  # already in "certain failure" zone
        return 1.0
    prob = (F_age_horizon - F_age) / survival
    return float(np.clip(prob, 0.0, 1.0))


def remaining_useful_life(age: float, beta: float, eta: float, reliability_target: float = 0.10) -> float:
    """
    Estimate RUL (days) as the additional age at which reliability drops to
    `reliability_target` (default 10 %).

    Reliability R(t) = 1 - F(t)
    We seek  t*  such that  R(t*) = reliability_target, then RUL = t* - age.
    """
    from scipy.optimize import brentq

    def objective(t):
        return weibull_min.sf(t, beta, scale=eta, loc=0) - reliability_target

    # Safe brentq bracketing
    lower, upper = age, max(age + 1, eta * 10)
    try:
        t_star = brentq(objective, lower, upper, xtol=1e-3, maxiter=200)
        return max(0.0, float(t_star - age))
    except Exception:
        return float(eta)  # fallback: characteristic life


# ─── Batch Analysis ───────────────────────────────────────────────────────────


@timed
def run_weibull_analysis(
    work_orders: pd.DataFrame,
    failure_history: pd.DataFrame,
    horizon_days: int | None = None,
) -> pd.DataFrame:
    """
    Fit Weibull models per equipment class, then attach failure probabilities
    and RUL estimates to every work order.

    Parameters
    ----------
    work_orders     : cleaned WO DataFrame (must have asset_class, age_days)
    failure_history : cleaned failure records (must have asset_class, time_to_failure_d)
    horizon_days    : planning window. Defaults to `None`, resolved against
                       the LIVE value of TA_CFG.planning_horizon_days inside
                       this function body — see failure_probability()'s
                       docstring for why this matters.

    Returns
    -------
    work_orders DataFrame with new columns:
        fitted_beta, fitted_eta, failure_prob, rul_days, weibull_source
    """
    if horizon_days is None:
        horizon_days = TA_CFG.planning_horizon_days

    wos = work_orders.copy()

    # Aggregate failure times by equipment class
    class_ttf: dict[str, np.ndarray] = (
        failure_history.groupby("asset_class")["time_to_failure_d"].apply(np.array).to_dict()
    )

    # Fit per class and store params
    class_params: dict[str, tuple[float, float]] = {}
    for cls, times in class_ttf.items():
        beta, eta = fit_weibull(times)
        class_params[cls] = (beta, eta)
        log.info(
            "  %6s → β=%.3f  η=%.1f days  (n=%d TTF records)",
            cls,
            beta,
            eta,
            len(times),
        )

    # Columns to fill
    wos["fitted_beta"] = np.nan
    wos["fitted_eta"] = np.nan
    wos["failure_prob"] = np.nan
    wos["rul_days"] = np.nan
    wos["weibull_source"] = "class_fit"

    for cls, (beta, eta) in class_params.items():
        mask = wos["asset_class"] == cls
        if not mask.any():
            continue

        # If WO already carries inline Weibull params (e.g. from asset master),
        # use the work-order level values for greater precision
        has_inline = (
            "weibull_beta" in wos.columns
            and "weibull_eta" in wos.columns
            and wos.loc[mask, "weibull_beta"].notna().any()
        )

        if has_inline:
            b = wos.loc[mask, "weibull_beta"].fillna(beta)
            e = wos.loc[mask, "weibull_eta"].fillna(eta)
            wos.loc[mask, "fitted_beta"] = b
            wos.loc[mask, "fitted_eta"] = e
            wos.loc[mask, "weibull_source"] = "asset_inline"
        else:
            wos.loc[mask, "fitted_beta"] = beta
            wos.loc[mask, "fitted_eta"] = eta

        # Vectorised probability and RUL
        for idx in wos[mask].index:
            age = float(wos.at[idx, "age_days"])
            b_i = float(wos.at[idx, "fitted_beta"])
            e_i = float(wos.at[idx, "fitted_eta"])
            wos.at[idx, "failure_prob"] = failure_probability(age, b_i, e_i, horizon_days)
            wos.at[idx, "rul_days"] = remaining_useful_life(age, b_i, e_i)

    # For any class not in failure history, fall back to the inline Weibull params
    still_missing = wos["failure_prob"].isna()
    if still_missing.any():
        log.warning(
            "%d WOs have no fitted Weibull – using asset inline params",
            still_missing.sum(),
        )
        for idx in wos[still_missing].index:
            b = float(wos.at[idx, "weibull_beta"]) if "weibull_beta" in wos.columns else 2.0
            e = float(wos.at[idx, "weibull_eta"]) if "weibull_eta" in wos.columns else 1825.0
            age = float(wos.at[idx, "age_days"])
            wos.at[idx, "fitted_beta"] = b
            wos.at[idx, "fitted_eta"] = e
            wos.at[idx, "failure_prob"] = failure_probability(age, b, e, horizon_days)
            wos.at[idx, "rul_days"] = remaining_useful_life(age, b, e)
            wos.at[idx, "weibull_source"] = "asset_inline_fallback"

    log.info(
        "Weibull analysis complete | mean P(failure)=%.3f | mean RUL=%.0f days",
        wos["failure_prob"].mean(),
        wos["rul_days"].mean(),
    )
    return wos
