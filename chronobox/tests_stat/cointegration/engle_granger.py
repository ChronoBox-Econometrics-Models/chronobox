"""
Engle-Granger two-step cointegration test.

Step 1: OLS cointegrating regression y_t = alpha + beta*x_t + e_t
Step 2: ADF test on residuals e_hat_t

    H0: no cointegration (residuals have unit root)
    H1: cointegration (residuals are stationary)

Critical values from MacKinnon (2010) for residual-based tests,
which differ from standard ADF because e_hat are estimated residuals.

References
----------
- Engle, R.F. & Granger, C.W.J. (1987). Co-integration and error correction:
  representation, estimation, and testing. Econometrica, 55(2), 251-276.
- MacKinnon, J.G. (2010). Critical values for cointegration tests.
  Queen's Economics Department Working Paper No. 1227.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chronobox.tests_stat.base import TestResult
from chronobox.tests_stat.critical_values.mackinnon import mackinnon_crit


def engle_granger_test(
    y: NDArray[np.floating],
    x: NDArray[np.floating],
    trend: str = "c",
    maxlag: int | None = None,
    autolag: str = "AIC",
) -> TestResult:
    """Engle-Granger two-step cointegration test.

    Parameters
    ----------
    y : array_like, shape (T,)
        Dependent variable (assumed I(1)).
    x : array_like, shape (T,) or (T, k)
        Independent variable(s) (assumed I(1)).
        If 1-dimensional, treated as single regressor.
    trend : str, default 'c'
        Deterministic terms in cointegrating regression:
        - 'n': no constant
        - 'c': constant only
        - 'ct': constant + trend
    maxlag : int or None, default None
        Maximum lags for ADF on residuals. If None, uses int(12*(T/100)^{1/4}).
    autolag : str, default 'AIC'
        Lag selection for ADF: 'AIC' or 'BIC'.

    Returns
    -------
    TestResult
        With additional_info containing:
        - 'cointegrating_vector': estimated beta coefficients
        - 'residuals': OLS residuals from cointegrating regression
        - 'n_vars': total number of variables in the system
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    nobs, k = x.shape
    if len(y) != nobs:
        raise ValueError(
            f"y and x must have same length: len(y)={len(y)}, x.shape[0]={nobs}"
        )

    n_vars = k + 1  # total variables in the system

    # Step 1: OLS cointegrating regression
    # y_t = alpha (+ beta*t) + gamma' * x_t + e_t
    regressors: list[NDArray[np.floating]] = [x]
    if trend in ("c", "ct"):
        regressors.append(np.ones((nobs, 1)))
    if trend == "ct":
        regressors.append(np.arange(1, nobs + 1, dtype=float).reshape(-1, 1))

    x_coint = np.column_stack(regressors)
    beta_coint = np.linalg.solve(x_coint.T @ x_coint, x_coint.T @ y)
    residuals = y - x_coint @ beta_coint

    # Cointegrating vector (coefficients on x)
    coint_vector = beta_coint[:k]

    # Step 2: ADF test on residuals (no deterministic terms in ADF)
    if maxlag is None:
        maxlag = int(12.0 * (nobs / 100.0) ** 0.25)
    maxlag = min(maxlag, nobs // 4)

    dy = np.diff(residuals)  # (nobs-1,)

    best_lag = 0
    best_ic = np.inf

    for p in range(0, maxlag + 1):
        e_lag = residuals[p : nobs - 1]
        dy_dep = dy[p:]
        n = len(dy_dep)

        if n < p + 2:
            continue

        # Regressor: e_{t-1} + lagged diffs (no constant in residual ADF)
        regs: list[NDArray[np.floating]] = [e_lag.reshape(-1, 1)]
        for i in range(1, p + 1):
            lag_diff = dy[p - i : nobs - 1 - i]
            regs.append(lag_diff.reshape(-1, 1))

        x_adf = np.column_stack(regs)
        k_adf = x_adf.shape[1]

        try:
            beta_adf = np.linalg.solve(x_adf.T @ x_adf, x_adf.T @ dy_dep)
            resid_adf = dy_dep - x_adf @ beta_adf
            ssr = float(np.sum(resid_adf**2))

            if autolag.upper() == "AIC":
                ic = n * np.log(ssr / n) + 2 * k_adf
            else:
                ic = n * np.log(ssr / n) + k_adf * np.log(n)

            if ic < best_ic:
                best_ic = ic
                best_lag = p
        except np.linalg.LinAlgError:
            continue

    # Final ADF estimation with best lag
    e_lag = residuals[best_lag : nobs - 1]
    dy_dep = dy[best_lag:]
    n = len(dy_dep)

    regs_final: list[NDArray[np.floating]] = [e_lag.reshape(-1, 1)]
    for i in range(1, best_lag + 1):
        lag_diff = dy[best_lag - i : nobs - 1 - i]
        regs_final.append(lag_diff.reshape(-1, 1))

    x_adf = np.column_stack(regs_final)
    k_adf = x_adf.shape[1]

    beta_adf = np.linalg.solve(x_adf.T @ x_adf, x_adf.T @ dy_dep)
    resid_adf = dy_dep - x_adf @ beta_adf
    ssr = float(np.sum(resid_adf**2))
    sigma2 = ssr / (n - k_adf)
    var_beta = sigma2 * np.linalg.inv(x_adf.T @ x_adf)
    se = np.sqrt(np.diag(var_beta))

    gamma = float(beta_adf[0])
    t_stat = gamma / float(se[0])

    # Critical values for residual-based test (MacKinnon EG tables)
    cv = mackinnon_crit(nobs=n, n_vars=n_vars, test_type="eg")

    reject = t_stat < cv["5%"]

    return TestResult(
        test_name="Engle-Granger",
        statistic=t_stat,
        pvalue=None,
        critical_values=cv,
        null_hypothesis="No cointegration (residuals have unit root)",
        alternative_hypothesis="Cointegration (residuals are stationary)",
        reject_at_5pct=reject,
        lags_used=best_lag,
        additional_info={
            "cointegrating_vector": coint_vector.tolist(),
            "residuals": residuals,
            "n_vars": n_vars,
            "trend": trend,
            "nobs": n,
            "gamma": gamma,
        },
    )
