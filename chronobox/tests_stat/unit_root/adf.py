"""
Augmented Dickey-Fuller test for unit roots.

Tests the null hypothesis that a unit root is present in a time series.

    Delta y_t = alpha + beta*t + gamma*y_{t-1} + sum(delta_i*Delta y_{t-i}) + eps_t

    H0: gamma = 0 (unit root present)
    H1: gamma < 0 (series is stationary)

Three regression models:
    'n'  : no constant, no trend
    'c'  : constant only (default)
    'ct' : constant + linear trend

Lag selection via AIC, BIC, or sequential t-significance.
Critical values from MacKinnon (1996) regression surface.

References
----------
- Dickey, D.A. & Fuller, W.A. (1979). Distribution of the estimators for
  autoregressive time series with a unit root. JASA, 74(366), 427-431.
- MacKinnon, J.G. (1996). Numerical distribution functions for unit root
  and cointegration tests. Journal of Applied Econometrics, 11(6), 601-618.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from chronobox.tests_stat.base import TestResult
from chronobox.tests_stat.critical_values.mackinnon import (
    mackinnon_crit,
    mackinnon_pvalue,
)


def _build_adf_regressors(
    y: NDArray[np.floating[Any]],
    lags: int,
    regression: str,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Build the ADF regression matrices.

    Parameters
    ----------
    y : ndarray, shape (nobs_total,)
        Time series data.
    lags : int
        Number of lagged differences to include.
    regression : str
        Regression model: 'n', 'c', or 'ct'.

    Returns
    -------
    dy : ndarray, shape (nobs_total - 1 - lags,)
        Dependent variable (Delta y_t).
    x_mat : ndarray, shape (nobs_total - 1 - lags, k)
        Regressor matrix: [y_{t-1}, Delta y_{t-1}, ..., Delta y_{t-p}, const, trend].
    """
    nobs_total = len(y)
    dy = np.diff(y)

    # y_{t-1} aligned with dy[lags:]
    y_lag1 = y[lags : nobs_total - 1]

    # Dependent variable
    dy_dep = dy[lags:]

    n = len(dy_dep)

    # Build regressor list
    regressors: list[NDArray[np.floating[Any]]] = [y_lag1.reshape(-1, 1)]

    # Lagged differences
    for i in range(1, lags + 1):
        lag_diff = dy[lags - i : nobs_total - 1 - i]
        regressors.append(lag_diff.reshape(-1, 1))

    # Deterministic terms
    if regression in ("c", "ct"):
        regressors.append(np.ones((n, 1)))
    if regression == "ct":
        regressors.append(np.arange(1, n + 1, dtype=float).reshape(-1, 1))

    x_mat = np.column_stack(regressors)
    return dy_dep, x_mat


def _ols_t_stat(
    y: NDArray[np.floating[Any]],
    x_mat: NDArray[np.floating[Any]],
) -> tuple[float, float, float]:
    """OLS regression returning t-stat, coefficient, and SSR for first regressor.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Dependent variable.
    x_mat : ndarray, shape (n, k)
        Regressor matrix. First column is the variable of interest.

    Returns
    -------
    t_stat : float
        t-statistic of the first coefficient (gamma).
    gamma : float
        Coefficient estimate for first regressor.
    ssr : float
        Sum of squared residuals.
    """
    n, k = x_mat.shape
    # OLS via normal equations
    xtx = x_mat.T @ x_mat
    xty = x_mat.T @ y
    beta = np.linalg.solve(xtx, xty)

    residuals = y - x_mat @ beta
    ssr = float(np.sum(residuals**2))

    # Standard errors
    sigma2 = ssr / (n - k)
    var_beta = sigma2 * np.linalg.inv(xtx)
    se = np.sqrt(np.diag(var_beta))

    gamma = float(beta[0])
    t_stat = gamma / float(se[0])

    return t_stat, gamma, ssr


def adf_test(
    y: NDArray[np.floating[Any]],
    regression: str = "c",
    maxlag: int | None = None,
    autolag: str | None = "AIC",
) -> TestResult:
    """Augmented Dickey-Fuller unit root test.

    Tests H0: unit root present vs H1: series is stationary.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series data. Must be 1-dimensional.
    regression : str, default 'c'
        Regression model:
        - 'n': no constant, no trend
        - 'c': constant only
        - 'ct': constant + linear trend
    maxlag : int or None, default None
        Maximum number of lags for the lagged difference terms.
        If None, uses int(12 * (T/100)^{1/4}).
    autolag : str or None, default 'AIC'
        Method for automatic lag selection:
        - 'AIC': minimize Akaike Information Criterion
        - 'BIC': minimize Bayesian Information Criterion
        - 't-sig': sequential t-significance at 5%
        - None: use maxlag directly (no automatic selection)

    Returns
    -------
    TestResult
        Standardized test result with:
        - statistic: ADF t-statistic
        - pvalue: approximate MacKinnon p-value
        - critical_values: MacKinnon critical values at 1%, 5%, 10%
        - additional_info: {'regression': model, 'nobs': effective sample size,
                            'gamma': coefficient estimate}

    Raises
    ------
    ValueError
        If y is not 1-dimensional or regression model is invalid.
    """
    y = np.asarray(y, dtype=float).ravel()
    nobs_total = len(y)

    if nobs_total < 10:
        raise ValueError(
            f"Series too short: T={nobs_total}, need at least 10 observations."
        )

    if regression not in ("n", "c", "ct"):
        raise ValueError(
            f"Invalid regression model '{regression}'. Choose from 'n', 'c', 'ct'."
        )

    if maxlag is None:
        maxlag = int(12.0 * (nobs_total / 100.0) ** 0.25)
    maxlag = min(maxlag, nobs_total // 2 - 2)

    if autolag is None:
        # Use maxlag directly
        best_lag = maxlag
    elif autolag.upper() in ("AIC", "BIC"):
        # Use consistent sample size (trimmed to maxlag) for comparable IC values
        best_ic = np.inf
        best_lag = 0
        for p in range(0, maxlag + 1):
            try:
                dy_ref, _ = _build_adf_regressors(y, maxlag, regression)
                n_consistent = len(dy_ref)
                # Re-build with actual lag p but trim to same sample
                dy_dep_p, x_p = _build_adf_regressors(y, p, regression)
                # Trim to consistent sample size
                trim = len(dy_dep_p) - n_consistent
                if trim > 0:
                    dy_dep_p = dy_dep_p[trim:]
                    x_p = x_p[trim:]
                n = len(dy_dep_p)
                _, _, ssr = _ols_t_stat(dy_dep_p, x_p)
                k = x_p.shape[1]
                if autolag.upper() == "AIC":
                    ic = n * np.log(ssr / n) + 2 * k
                else:  # BIC
                    ic = n * np.log(ssr / n) + k * np.log(n)
                if ic < best_ic:
                    best_ic = ic
                    best_lag = p
            except (np.linalg.LinAlgError, ValueError):
                continue
    elif autolag.lower() == "t-sig":
        best_lag = 0
        for p in range(maxlag, 0, -1):
            try:
                dy_dep, x_mat = _build_adf_regressors(y, p, regression)
                n, k = x_mat.shape
                beta = np.linalg.solve(x_mat.T @ x_mat, x_mat.T @ dy_dep)
                resid = dy_dep - x_mat @ beta
                ssr = float(np.sum(resid**2))
                sigma2 = ssr / (n - k)
                var_beta = sigma2 * np.linalg.inv(x_mat.T @ x_mat)
                se = np.sqrt(np.diag(var_beta))
                # t-stat of the last lagged difference (index p for p-th lag)
                t_last = abs(float(beta[p])) / float(se[p])
                if t_last > 1.96:
                    best_lag = p
                    break
            except (np.linalg.LinAlgError, ValueError, IndexError):
                continue
    else:
        raise ValueError(
            f"Invalid autolag '{autolag}'. Choose from 'AIC', 'BIC', 't-sig', or None."
        )

    # Final estimation with selected lag
    dy_dep, x_mat = _build_adf_regressors(y, best_lag, regression)
    nobs = len(dy_dep)
    t_stat, gamma, ssr = _ols_t_stat(dy_dep, x_mat)

    # Critical values and p-value
    cv = mackinnon_crit(nobs=nobs, regression=regression, test_type="adf")
    pval = mackinnon_pvalue(stat=t_stat, regression=regression, nobs=nobs)

    reject = t_stat < cv["5%"]

    return TestResult(
        test_name="Augmented Dickey-Fuller",
        statistic=t_stat,
        pvalue=pval,
        critical_values=cv,
        null_hypothesis="Unit root present (gamma = 0)",
        alternative_hypothesis="Series is stationary (gamma < 0)",
        reject_at_5pct=reject,
        lags_used=best_lag,
        additional_info={
            "regression": regression,
            "nobs": nobs,
            "gamma": gamma,
        },
    )
