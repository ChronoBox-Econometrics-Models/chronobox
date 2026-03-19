"""
Elliott-Rothenberg-Stock (ERS) / DF-GLS unit root test.

GLS detrending followed by ADF, providing higher power than standard ADF
for near-unit-root alternatives.

    Step 1: GLS detrending with quasi-difference parameter
        c_bar = -7 (constant) or c_bar = -13.5 (constant+trend)
    Step 2: ADF test on detrended series

    H0: gamma = 0 (unit root)
    H1: gamma < 0 (stationary)

References
----------
- Elliott, G., Rothenberg, T.J. & Stock, J.H. (1996). Efficient tests
  for an autoregressive unit root. Econometrica, 64(4), 813-836.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from chronobox.tests_stat.base import TestResult

# ERS/DF-GLS asymptotic critical values (Elliott et al. 1996, Table 1)
_ERS_CV: dict[str, dict[str, float]] = {
    "c": {
        "1%": -2.5658,
        "5%": -1.9393,
        "10%": -1.6156,
    },
    "ct": {
        "1%": -3.4800,
        "5%": -2.8900,
        "10%": -2.5700,
    },
}


def _gls_detrend(
    y: NDArray[np.floating[Any]],
    regression: str,
) -> NDArray[np.floating[Any]]:
    """GLS detrending for ERS test.

    Parameters
    ----------
    y : ndarray, shape (T,)
        Original time series.
    regression : str
        'c' (constant) or 'ct' (constant + trend).

    Returns
    -------
    y_detrended : ndarray, shape (T,)
        GLS-detrended series.
    """
    nobs = len(y)

    # Quasi-difference parameter
    c_bar = -7.0 if regression == "c" else -13.5

    alpha_bar = 1.0 + c_bar / nobs

    # Build quasi-differenced y
    y_qd = np.empty(nobs)
    y_qd[0] = y[0]
    y_qd[1:] = y[1:] - alpha_bar * y[:-1]

    # Build quasi-differenced deterministics
    if regression == "c":
        x = np.ones((nobs, 1))
        x_qd = np.empty((nobs, 1))
        x_qd[0, 0] = 1.0
        x_qd[1:, 0] = 1.0 - alpha_bar
    else:  # 'ct'
        x = np.column_stack([np.ones(nobs), np.arange(1, nobs + 1, dtype=float)])
        x_qd = np.empty((nobs, 2))
        x_qd[0, 0] = 1.0
        x_qd[0, 1] = 1.0
        x_qd[1:, 0] = 1.0 - alpha_bar
        x_qd[1:, 1] = (
            np.arange(2, nobs + 1, dtype=float)
            - alpha_bar * np.arange(1, nobs, dtype=float)
        )

    # GLS regression on quasi-differenced data
    beta_gls = np.linalg.solve(x_qd.T @ x_qd, x_qd.T @ y_qd)

    # Detrend original series
    y_detrended: NDArray[np.floating[Any]] = y - x @ beta_gls

    return y_detrended


def ers_test(
    y: NDArray[np.floating[Any]],
    regression: str = "c",
    maxlag: int | None = None,
    autolag: str | None = "AIC",
) -> TestResult:
    """ERS / DF-GLS unit root test.

    More powerful alternative to standard ADF, using GLS detrending.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series data. Must be 1-dimensional.
    regression : str, default 'c'
        Regression model:
        - 'c': constant (c_bar = -7)
        - 'ct': constant + trend (c_bar = -13.5)
    maxlag : int or None, default None
        Maximum number of lags. If None, uses int(12*(T/100)^{1/4}).
    autolag : str or None, default 'AIC'
        Automatic lag selection method: 'AIC', 'BIC', or None.

    Returns
    -------
    TestResult
        Standardized test result with:
        - statistic: DF-GLS t-statistic
        - critical_values: ERS critical values
        - additional_info: {'regression': model, 'nobs': T, 'c_bar': value}
    """
    y = np.asarray(y, dtype=float).ravel()
    nobs = len(y)

    if nobs < 15:
        raise ValueError(
            f"Series too short: T={nobs}, need at least 15 observations."
        )

    if regression not in ("c", "ct"):
        raise ValueError(
            f"Invalid regression model '{regression}'. Choose from 'c', 'ct'."
        )

    # Step 1: GLS detrending
    y_d = _gls_detrend(y, regression)

    # Step 2: ADF on detrended series (no deterministics in ADF)
    if maxlag is None:
        maxlag = int(12.0 * (nobs / 100.0) ** 0.25)
    maxlag = min(maxlag, nobs // 2 - 2)

    dy = np.diff(y_d)  # (nobs-1,)

    best_lag = 0
    best_ic = np.inf

    for p in range(0, maxlag + 1):
        # Build regressors: y_{t-1} and lagged diffs (no constant/trend)
        y_lag1 = y_d[p : nobs - 1]
        dy_dep = dy[p:]
        n = len(dy_dep)

        if n < p + 2:
            continue

        regressors: list[NDArray[np.floating[Any]]] = [y_lag1.reshape(-1, 1)]
        for i in range(1, p + 1):
            lag_diff = dy[p - i : nobs - 1 - i]
            regressors.append(lag_diff.reshape(-1, 1))

        x_mat = (
            np.column_stack(regressors)
            if len(regressors) > 1
            else regressors[0].reshape(-1, 1)
        )
        k = x_mat.shape[1]

        try:
            beta = np.linalg.solve(x_mat.T @ x_mat, x_mat.T @ dy_dep)
            resid = dy_dep - x_mat @ beta
            ssr = float(np.sum(resid**2))

            if autolag is None:
                if p == maxlag:
                    best_lag = p
                    best_ic = ssr
                continue
            elif autolag.upper() == "AIC":
                ic = n * np.log(ssr / n) + 2 * k
            elif autolag.upper() == "BIC":
                ic = n * np.log(ssr / n) + k * np.log(n)
            else:
                ic = ssr  # fallback
            if ic < best_ic:
                best_ic = ic
                best_lag = p
        except np.linalg.LinAlgError:
            continue

    if autolag is None:
        best_lag = maxlag

    # Final estimation
    y_lag1 = y_d[best_lag : nobs - 1]
    dy_dep = dy[best_lag:]
    n = len(dy_dep)

    regressors_final: list[NDArray[np.floating[Any]]] = [y_lag1.reshape(-1, 1)]
    for i in range(1, best_lag + 1):
        lag_diff = dy[best_lag - i : nobs - 1 - i]
        regressors_final.append(lag_diff.reshape(-1, 1))

    x_mat = (
        np.column_stack(regressors_final)
        if len(regressors_final) > 1
        else regressors_final[0].reshape(-1, 1)
    )
    k = x_mat.shape[1]

    beta = np.linalg.solve(x_mat.T @ x_mat, x_mat.T @ dy_dep)
    resid = dy_dep - x_mat @ beta
    ssr = float(np.sum(resid**2))
    sigma2 = ssr / (n - k)
    var_beta = sigma2 * np.linalg.inv(x_mat.T @ x_mat)
    se = np.sqrt(np.diag(var_beta))

    gamma = float(beta[0])
    t_stat = gamma / float(se[0])

    # Critical values
    cv = _ERS_CV[regression]
    c_bar = -7.0 if regression == "c" else -13.5

    reject = t_stat < cv["5%"]

    return TestResult(
        test_name="ERS / DF-GLS",
        statistic=t_stat,
        pvalue=None,
        critical_values=cv,
        null_hypothesis="Unit root present (gamma = 0)",
        alternative_hypothesis="Series is stationary (gamma < 0)",
        reject_at_5pct=reject,
        lags_used=best_lag,
        additional_info={
            "regression": regression,
            "nobs": n,
            "c_bar": c_bar,
            "gamma": gamma,
        },
    )
