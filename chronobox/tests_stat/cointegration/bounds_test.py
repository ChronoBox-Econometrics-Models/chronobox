"""
Pesaran-Shin-Smith (PSS) ARDL Bounds test for cointegration.

Tests for a levels relationship using the ARDL conditional ECM approach.
The test is valid regardless of whether regressors are I(0), I(1), or
mutually cointegrated.

    F-test: H0: pi_yy = 0 AND pi_yx = 0 (no levels relationship)
    t-test: H0: pi_yy = 0

    Decision based on bounds:
    - F > upper bound -> cointegration
    - F < lower bound -> no cointegration
    - between bounds -> inconclusive

References
----------
- Pesaran, M.H., Shin, Y. & Smith, R.J. (2001). Bounds testing approaches
  to the analysis of level relationships. Journal of Applied Econometrics,
  16(3), 289-326.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chronobox.tests_stat.base import TestResult
from chronobox.tests_stat.critical_values.pss_bounds import pss_f_bounds


def _estimate_ecm(
    y: NDArray[np.floating],
    x: NDArray[np.floating],
    lags: int,
    case: int,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    float,
    int,
    int,
]:
    """Estimate the unrestricted ECM for the bounds test.

    Returns
    -------
    beta, residuals, x_mat, ssr, n, k
    """
    nobs = x.shape[0]

    dy = np.diff(y)
    dep = dy[lags:]
    n = len(dep)

    regressors: list[NDArray[np.floating]] = []

    # Levels terms: y_{t-1}, x_{t-1}
    regressors.append(y[lags : nobs - 1].reshape(-1, 1))  # y_{t-1}
    regressors.append(x[lags : nobs - 1])  # x_{t-1}

    # Lagged Delta y
    for j in range(1, lags + 1):
        regressors.append(dy[lags - j : nobs - 1 - j].reshape(-1, 1))

    # Current and lagged Delta x
    dx = np.diff(x, axis=0)
    for j in range(0, lags + 1):
        dx_part = dx[lags:] if j == 0 else dx[lags - j : nobs - 1 - j]
        regressors.append(dx_part)

    # Deterministics
    if case in (3, 5):
        regressors.append(np.ones((n, 1)))
    if case == 5:
        regressors.append(np.arange(lags + 1, nobs, dtype=float).reshape(-1, 1))

    x_mat = np.column_stack(regressors)
    k_total = x_mat.shape[1]

    beta = np.linalg.solve(x_mat.T @ x_mat, x_mat.T @ dep)
    residuals = dep - x_mat @ beta
    ssr = float(np.sum(residuals**2))

    return beta, residuals, x_mat, ssr, n, k_total


def bounds_test(
    y: NDArray[np.floating],
    x: NDArray[np.floating],
    lags: int | None = None,
    case: int = 3,
) -> TestResult:
    """PSS ARDL Bounds test for cointegration.

    Parameters
    ----------
    y : array_like, shape (T,)
        Dependent variable.
    x : array_like, shape (T,) or (T, k)
        Independent variable(s).
    lags : int or None, default None
        Number of lags for the ECM. If None, selects via AIC.
    case : int, default 3
        Deterministic specification:
        - 3: unrestricted intercept, no trend (most common)
        - 5: unrestricted intercept and trend

    Returns
    -------
    TestResult
        With additional_info containing:
        - 'f_statistic': F-test value
        - 't_statistic': t-test value for pi_yy
        - 'lower_bound': I(0) bound at 5%
        - 'upper_bound': I(1) bound at 5%
        - 'decision': 'cointegration', 'no cointegration', or 'inconclusive'
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    nobs, k = x.shape
    if len(y) != nobs:
        raise ValueError("y and x must have same length.")

    # Lag selection via AIC
    if lags is None:
        max_lags = int(4.0 * (nobs / 100.0) ** 0.25)
        max_lags = max(1, min(max_lags, nobs // 4))
        best_lag = 1
        best_ic = np.inf
        for p in range(1, max_lags + 1):
            try:
                _, _, _, ssr_u, n_u, k_u = _estimate_ecm(y, x, p, case)
                ic = n_u * np.log(ssr_u / n_u) + 2 * k_u
                if ic < best_ic:
                    best_ic = ic
                    best_lag = p
            except (np.linalg.LinAlgError, ValueError):
                continue
        lags = best_lag

    # Estimate unrestricted ECM
    beta_u, _resid_u, x_u, ssr_u, n_u, k_u = _estimate_ecm(y, x, lags, case)

    # Number of levels terms: y_{t-1} and x_{t-1} columns
    n_levels = k + 1  # y_{t-1} and x_i_{t-1}

    # Estimate restricted model (without levels terms)
    dy = np.diff(y)
    dep = dy[lags:]
    n = len(dep)

    regs_r: list[NDArray[np.floating]] = []

    # Lagged Delta y
    for j in range(1, lags + 1):
        regs_r.append(dy[lags - j : nobs - 1 - j].reshape(-1, 1))

    # Current and lagged Delta x
    dx = np.diff(x, axis=0)
    for j in range(0, lags + 1):
        dx_part = dx[lags:] if j == 0 else dx[lags - j : nobs - 1 - j]
        regs_r.append(dx_part)

    # Deterministics
    if case in (3, 5):
        regs_r.append(np.ones((n, 1)))
    if case == 5:
        regs_r.append(np.arange(lags + 1, nobs, dtype=float).reshape(-1, 1))

    x_r = np.column_stack(regs_r) if regs_r else np.ones((n, 1))

    beta_r_est = np.linalg.solve(x_r.T @ x_r, x_r.T @ dep)
    resid_r = dep - x_r @ beta_r_est
    ssr_r = float(np.sum(resid_r**2))

    # F-statistic
    q = n_levels  # number of restrictions
    f_stat = ((ssr_r - ssr_u) / q) / (ssr_u / (n_u - k_u))

    # t-statistic for pi_yy (first coefficient in unrestricted)
    sigma2_u = ssr_u / (n_u - k_u)
    var_beta_u = sigma2_u * np.linalg.inv(x_u.T @ x_u)
    se_u = np.sqrt(np.diag(var_beta_u))
    t_stat = float(beta_u[0]) / float(se_u[0])

    # Bounds critical values
    try:
        f_bounds = pss_f_bounds(k=k, case=case)
        lower_5 = f_bounds["5%"][0]
        upper_5 = f_bounds["5%"][1]
    except ValueError:
        # Fallback for unsupported k
        lower_5 = 3.0
        upper_5 = 4.0

    # Decision
    if f_stat > upper_5:
        decision = "cointegration"
        reject = True
    elif f_stat < lower_5:
        decision = "no cointegration"
        reject = False
    else:
        decision = "inconclusive"
        reject = False

    cv_display: dict[str, float] = {
        "5% I(0)": lower_5,
        "5% I(1)": upper_5,
    }

    return TestResult(
        test_name="PSS Bounds",
        statistic=f_stat,
        pvalue=None,
        critical_values=cv_display,
        null_hypothesis="No levels relationship (pi_yy = pi_yx = 0)",
        alternative_hypothesis="Levels relationship exists (cointegration)",
        reject_at_5pct=reject,
        lags_used=lags,
        additional_info={
            "f_statistic": f_stat,
            "t_statistic": t_stat,
            "lower_bound": lower_5,
            "upper_bound": upper_5,
            "decision": decision,
            "case": case,
            "k": k,
            "nobs": n_u,
        },
    )
