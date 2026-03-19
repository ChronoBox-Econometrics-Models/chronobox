"""
Gregory-Hansen cointegration test with structural break.

Tests for cointegration allowing for a single structural break in the
cointegrating relationship at an unknown date.

    Model C:   level shift in intercept
    Model C/T: level shift + trend
    Model C/S: regime shift (coefficients change)

    H0: no cointegration
    H1: cointegration with structural break

References
----------
- Gregory, A.W. & Hansen, B.E. (1996). Residual-based tests for cointegration
  in models with regime shifts. Journal of Econometrics, 70(1), 99-126.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chronobox.tests_stat.base import TestResult

# Gregory-Hansen critical values (Table 1, Gregory & Hansen 1996)
# For bivariate case (m=1 regressor)
_GH_CV: dict[str, dict[str, float]] = {
    "c": {"1%": -5.13, "5%": -4.61, "10%": -4.34},
    "c/t": {"1%": -5.45, "5%": -4.99, "10%": -4.72},
    "c/s": {"1%": -5.47, "5%": -4.95, "10%": -4.68},
}


def _adf_on_residuals(
    residuals: NDArray[np.floating],
    maxlag: int,
) -> float:
    """Run ADF regression on residuals and return t-statistic.

    Parameters
    ----------
    residuals : ndarray, shape (T,)
        Residuals from cointegrating regression.
    maxlag : int
        Maximum number of lags for lag selection.

    Returns
    -------
    t_stat : float
        ADF t-statistic.
    """
    nobs = len(residuals)
    dy = np.diff(residuals)

    best_lag = 0
    best_ic = np.inf

    for p in range(0, maxlag + 1):
        e_lag = residuals[p : nobs - 1]
        dy_dep = dy[p:]
        n = len(dy_dep)
        if n < p + 3:
            continue

        regs: list[NDArray[np.floating]] = [e_lag.reshape(-1, 1)]
        for i in range(1, p + 1):
            lag_diff = dy[p - i : nobs - 1 - i]
            regs.append(lag_diff.reshape(-1, 1))

        x_sel = np.column_stack(regs)
        k_sel = x_sel.shape[1]

        try:
            beta = np.linalg.solve(x_sel.T @ x_sel, x_sel.T @ dy_dep)
            resid = dy_dep - x_sel @ beta
            ssr = float(np.sum(resid**2))
            ic = n * np.log(ssr / n) + 2 * k_sel
            if ic < best_ic:
                best_ic = ic
                best_lag = p
        except np.linalg.LinAlgError:
            continue

    # Final estimation
    e_lag = residuals[best_lag : nobs - 1]
    dy_dep = dy[best_lag:]
    n = len(dy_dep)

    regs_final: list[NDArray[np.floating]] = [e_lag.reshape(-1, 1)]
    for i in range(1, best_lag + 1):
        lag_diff = dy[best_lag - i : nobs - 1 - i]
        regs_final.append(lag_diff.reshape(-1, 1))

    x_fin = np.column_stack(regs_final)
    k_fin = x_fin.shape[1]

    beta = np.linalg.solve(x_fin.T @ x_fin, x_fin.T @ dy_dep)
    resid = dy_dep - x_fin @ beta
    ssr = float(np.sum(resid**2))
    sigma2 = ssr / (n - k_fin)
    var_beta = sigma2 * np.linalg.inv(x_fin.T @ x_fin)
    se = np.sqrt(np.diag(var_beta))

    return float(beta[0]) / float(se[0])


def gregory_hansen_test(
    y: NDArray[np.floating],
    x: NDArray[np.floating],
    model: str = "c",
    maxlag: int | None = None,
    trim: float = 0.15,
) -> TestResult:
    """Gregory-Hansen cointegration test with structural break.

    Parameters
    ----------
    y : array_like, shape (T,)
        Dependent variable.
    x : array_like, shape (T,) or (T, k)
        Independent variable(s).
    model : str, default 'c'
        Model specification:
        - 'c': level shift (break in intercept)
        - 'c/t': level shift + trend
        - 'c/s': regime shift (break in intercept and slope coefficients)
    maxlag : int or None, default None
        Maximum lags for ADF on residuals.
    trim : float, default 0.15
        Fraction to trim from each end.

    Returns
    -------
    TestResult
        With additional_info containing:
        - 'break_date': estimated break date index
        - 'break_fraction': break as fraction of sample
        - 'model': model specification used
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    nobs, _k = x.shape
    if len(y) != nobs:
        raise ValueError("y and x must have same length.")

    model = model.lower()
    if model not in ("c", "c/t", "c/s"):
        raise ValueError(f"Invalid model '{model}'. Choose from 'c', 'c/t', 'c/s'.")

    if maxlag is None:
        maxlag = int(12.0 * (nobs / 100.0) ** 0.25)
    maxlag = min(maxlag, nobs // 4)

    start = int(np.ceil(trim * nobs))
    end = int(np.floor((1.0 - trim) * nobs))

    min_adf = np.inf
    best_break = start

    for tb in range(start, end + 1):
        # Build dummy variable
        du = np.zeros(nobs)
        du[tb:] = 1.0

        # Build regression matrix
        regs: list[NDArray[np.floating]] = []
        regs.append(np.ones((nobs, 1)))  # mu_1
        regs.append(du.reshape(-1, 1))  # mu_2 * du

        if model == "c/t":
            regs.append(np.arange(1, nobs + 1, dtype=float).reshape(-1, 1))  # trend

        regs.append(x)  # alpha' * x_t

        if model == "c/s":
            # Regime shift: x_t * du_t
            x_du = x * du.reshape(-1, 1)
            regs.append(x_du)

        x_coint = np.column_stack(regs)

        try:
            beta_coint = np.linalg.solve(x_coint.T @ x_coint, x_coint.T @ y)
            residuals = y - x_coint @ beta_coint

            adf_stat = _adf_on_residuals(residuals, maxlag)

            if adf_stat < min_adf:
                min_adf = adf_stat
                best_break = tb
        except np.linalg.LinAlgError:
            continue

    cv = _GH_CV.get(model, _GH_CV["c"])
    reject = min_adf < cv["5%"]

    return TestResult(
        test_name="Gregory-Hansen",
        statistic=min_adf,
        pvalue=None,
        critical_values=cv,
        null_hypothesis="No cointegration",
        alternative_hypothesis="Cointegration with structural break",
        reject_at_5pct=reject,
        lags_used=None,
        additional_info={
            "break_date": best_break,
            "break_fraction": best_break / nobs,
            "model": model,
            "nobs": nobs,
        },
    )
