"""
Phillips-Ouliaris residual-based cointegration test.

Similar to Engle-Granger but uses Phillips-Perron style correction
on the cointegrating residuals instead of ADF.

    H0: no cointegration (residuals have unit root)
    H1: cointegration (residuals are stationary)

References
----------
- Phillips, P.C.B. & Ouliaris, S. (1990). Asymptotic properties of residual
  based tests for cointegration. Econometrica, 58(1), 165-193.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chronobox.tests_stat.base import TestResult
from chronobox.tests_stat.critical_values.mackinnon import mackinnon_crit


def _newey_west_lrv(
    residuals: NDArray[np.floating],
    bandwidth: int,
) -> tuple[float, float]:
    """Compute Newey-West long-run variance with Bartlett kernel.

    Returns (gamma_0, lambda_sq).
    """
    nobs = len(residuals)
    gamma_0 = float(np.sum(residuals**2)) / nobs
    lambda_sq = gamma_0
    for j in range(1, bandwidth + 1):
        w_j = 1.0 - j / (bandwidth + 1.0)
        gamma_j = float(np.sum(residuals[j:] * residuals[:-j])) / nobs
        lambda_sq += 2.0 * w_j * gamma_j
    return gamma_0, lambda_sq


def phillips_ouliaris_test(
    y: NDArray[np.floating],
    x: NDArray[np.floating],
    trend: str = "c",
    lags: str | int = "short",
) -> TestResult:
    """Phillips-Ouliaris residual-based cointegration test.

    Parameters
    ----------
    y : array_like, shape (T,)
        Dependent variable.
    x : array_like, shape (T,) or (T, k)
        Independent variable(s).
    trend : str, default 'c'
        Deterministic terms: 'c' (constant) or 'ct' (constant + trend).
    lags : str or int, default 'short'
        Bandwidth: 'short' = int(4*(T/100)^{2/9}), 'long' = int(12*(T/100)^{2/9}),
        or integer.

    Returns
    -------
    TestResult
        With additional_info containing:
        - 'Z_alpha': Z_alpha statistic
        - 'cointegrating_vector': estimated coefficients
        - 'residuals': OLS residuals
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    nobs, k = x.shape
    n_vars = k + 1

    # Step 1: OLS cointegrating regression
    regressors: list[NDArray[np.floating]] = [x]
    if trend in ("c", "ct"):
        regressors.append(np.ones((nobs, 1)))
    if trend == "ct":
        regressors.append(np.arange(1, nobs + 1, dtype=float).reshape(-1, 1))

    x_coint = np.column_stack(regressors)
    beta_coint = np.linalg.solve(x_coint.T @ x_coint, x_coint.T @ y)
    residuals = y - x_coint @ beta_coint
    coint_vector = beta_coint[:k]

    # Step 2: PP test on residuals
    # Bandwidth
    if isinstance(lags, str):
        if lags == "short":
            bandwidth = int(4.0 * (nobs / 100.0) ** (2.0 / 9.0))
        elif lags == "long":
            bandwidth = int(12.0 * (nobs / 100.0) ** (2.0 / 9.0))
        else:
            bandwidth = int(4.0 * (nobs / 100.0) ** (2.0 / 9.0))
    else:
        bandwidth = int(lags)
    bandwidth = max(1, bandwidth)

    # AR(1) regression on residuals (no constant for residuals)
    e_dep = residuals[1:]
    e_lag = residuals[:-1]
    n = len(e_dep)

    # OLS: e_t = rho * e_{t-1} + v_t
    rho_hat = float(np.sum(e_lag * e_dep)) / float(np.sum(e_lag**2))
    v = e_dep - rho_hat * e_lag
    ssr = float(np.sum(v**2))
    s_sq = ssr / (n - 1)

    se_rho = np.sqrt(s_sq / float(np.sum(e_lag**2)))
    t_rho = (rho_hat - 1.0) / se_rho

    # Newey-West long-run variance of residuals v
    gamma_0, lambda_sq = _newey_west_lrv(v, bandwidth)

    # PP z_t statistic
    z_t = float(
        t_rho * np.sqrt(gamma_0 / lambda_sq)
        - (n * se_rho * (lambda_sq - gamma_0))
        / (2.0 * np.sqrt(lambda_sq) * np.sqrt(s_sq))
    )

    # PP z_alpha statistic
    z_alpha = float(
        n * (rho_hat - 1.0)
        - (n**2 * se_rho**2) / (2.0 * s_sq) * (lambda_sq - gamma_0)
    )

    # Critical values (Phillips-Ouliaris)
    try:
        cv = mackinnon_crit(nobs=n, n_vars=n_vars, test_type="po")
    except ValueError:
        # Fallback to EG critical values
        cv = mackinnon_crit(nobs=n, n_vars=n_vars, test_type="eg")

    reject = z_t < cv["5%"]

    return TestResult(
        test_name="Phillips-Ouliaris",
        statistic=z_t,
        pvalue=None,
        critical_values=cv,
        null_hypothesis="No cointegration (residuals have unit root)",
        alternative_hypothesis="Cointegration (residuals are stationary)",
        reject_at_5pct=reject,
        lags_used=bandwidth,
        additional_info={
            "Z_alpha": z_alpha,
            "cointegrating_vector": coint_vector.tolist(),
            "residuals": residuals,
            "bandwidth": bandwidth,
            "trend": trend,
            "n_vars": n_vars,
            "nobs": n,
        },
    )
