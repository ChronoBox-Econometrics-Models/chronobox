"""
Phillips-Perron test for unit roots.

Non-parametric correction to the Dickey-Fuller test that accounts for
serial correlation and heteroskedasticity using Newey-West long-run
variance estimation.

    Z_t statistic with Bartlett kernel and bandwidth l = int(4*(T/100)^{2/9})

    H0: rho = 1 (unit root present)
    H1: rho < 1 (series is stationary)

References
----------
- Phillips, P.C.B. & Perron, P. (1988). Testing for a unit root in time series
  regression. Biometrika, 75(2), 335-346.
- Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  Econometrica, 55(3), 703-708.
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


def _newey_west_long_run_variance(
    residuals: NDArray[np.floating[Any]],
    bandwidth: int,
) -> tuple[float, float]:
    """Compute Newey-West long-run variance estimate with Bartlett kernel.

    Parameters
    ----------
    residuals : ndarray, shape (nobs,)
        Residuals from OLS regression.
    bandwidth : int
        Number of lags for the kernel.

    Returns
    -------
    gamma_0 : float
        Sample variance (autocovariance at lag 0).
    lambda_sq : float
        Long-run variance estimate.
    """
    nobs = len(residuals)
    gamma_0 = float(np.sum(residuals**2)) / nobs

    lambda_sq = gamma_0
    for j in range(1, bandwidth + 1):
        w_j = 1.0 - j / (bandwidth + 1.0)  # Bartlett kernel
        gamma_j = float(np.sum(residuals[j:] * residuals[:-j])) / nobs
        lambda_sq += 2.0 * w_j * gamma_j

    return gamma_0, lambda_sq


def pp_test(
    y: NDArray[np.floating[Any]],
    regression: str = "c",
    lags: str | int | None = "short",
) -> TestResult:
    """Phillips-Perron unit root test.

    Tests H0: unit root present vs H1: series is stationary.
    Uses a non-parametric correction for serial correlation.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series data. Must be 1-dimensional.
    regression : str, default 'c'
        Regression model:
        - 'c': constant only
        - 'ct': constant + linear trend
    lags : str or int or None, default 'short'
        Bandwidth for the Newey-West estimator:
        - 'short': int(4*(T/100)^{2/9})
        - 'long': int(12*(T/100)^{2/9})
        - int: use directly as bandwidth
        - None: same as 'short'

    Returns
    -------
    TestResult
        Standardized test result with:
        - statistic: Z_t statistic
        - pvalue: approximate MacKinnon p-value
        - critical_values: MacKinnon critical values at 1%, 5%, 10%
        - additional_info: {'Z_alpha': Z_alpha statistic, 'bandwidth': l,
                            'regression': model, 'nobs': T}

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

    if regression not in ("c", "ct"):
        raise ValueError(
            f"Invalid regression model '{regression}'. Choose from 'c', 'ct'."
        )

    # Bandwidth selection
    if lags is None or lags == "short":
        bandwidth = int(4.0 * (nobs_total / 100.0) ** (2.0 / 9.0))
    elif lags == "long":
        bandwidth = int(12.0 * (nobs_total / 100.0) ** (2.0 / 9.0))
    elif isinstance(lags, int):
        bandwidth = lags
    else:
        raise ValueError(f"Invalid lags '{lags}'. Use 'short', 'long', int, or None.")

    bandwidth = max(1, bandwidth)

    # DF regression (0 lags): Delta y_t = alpha (+ beta*t) + gamma*y_{t-1} + eps_t
    # gamma = rho - 1, so testing gamma=0 is equivalent to testing rho=1
    dy = np.diff(y)
    y_lag = y[:-1]
    n = len(dy)

    # Build regressor matrix: [y_{t-1}, const, trend]
    regressors: list[NDArray[np.floating[Any]]] = [y_lag.reshape(-1, 1)]
    if regression in ("c", "ct"):
        regressors.append(np.ones((n, 1)))
    if regression == "ct":
        regressors.append(np.arange(1, n + 1, dtype=float).reshape(-1, 1))

    x_mat = np.column_stack(regressors)
    k = x_mat.shape[1]

    # OLS
    xtx = x_mat.T @ x_mat
    xty = x_mat.T @ dy
    beta = np.linalg.solve(xtx, xty)
    residuals = dy - x_mat @ beta

    gamma_hat = float(beta[0])
    rho_hat = gamma_hat + 1.0
    ssr = float(np.sum(residuals**2))
    s_sq = ssr / (n - k)

    # Standard error of gamma (first coefficient)
    var_beta = s_sq * np.linalg.inv(xtx)
    se_gamma = float(np.sqrt(var_beta[0, 0]))
    t_gamma = gamma_hat / se_gamma  # DF t-stat testing gamma=0

    # Newey-West long-run variance of residuals
    gamma_0, lambda_sq = _newey_west_long_run_variance(residuals, bandwidth)

    # PP Z_t statistic (correction to the DF t-stat)
    z_t = (
        t_gamma * np.sqrt(gamma_0 / lambda_sq)
        - (n * se_gamma * (lambda_sq - gamma_0))
        / (2.0 * np.sqrt(lambda_sq) * np.sqrt(s_sq))
    )

    # PP Z_alpha statistic
    z_alpha = (
        n * gamma_hat
        - (n**2 * se_gamma**2) / (2.0 * s_sq) * (lambda_sq - gamma_0)
    )

    # Use Z_t as main statistic (same distribution as ADF under H0)
    cv = mackinnon_crit(nobs=n, regression=regression, test_type="adf")
    pval = mackinnon_pvalue(stat=z_t, regression=regression, nobs=n)

    reject = bool(z_t < cv["5%"])

    return TestResult(
        test_name="Phillips-Perron",
        statistic=z_t,
        pvalue=pval,
        critical_values=cv,
        null_hypothesis="Unit root present (rho = 1)",
        alternative_hypothesis="Series is stationary (rho < 1)",
        reject_at_5pct=reject,
        lags_used=bandwidth,
        additional_info={
            "Z_alpha": z_alpha,
            "bandwidth": bandwidth,
            "regression": regression,
            "nobs": n,
            "rho_hat": rho_hat,
            "lambda_sq": lambda_sq,
            "gamma_0": gamma_0,
        },
    )
