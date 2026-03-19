"""
Zivot-Andrews unit root test with endogenous structural break.

Tests for a unit root allowing for one structural break in the intercept
and/or trend at an unknown date.

    Model A: break in intercept only
    Model B: break in trend only
    Model C: break in both intercept and trend

    H0: unit root (no break)
    H1: trend-stationary with break at unknown date

The break date is selected as the date that gives the strongest evidence
against the unit root null (minimum t-statistic).

References
----------
- Zivot, E. & Andrews, D.W.K. (1992). Further evidence on the great crash,
  the oil-price shock, and the unit-root hypothesis. JBES, 10(3), 251-270.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from chronobox.tests_stat.base import TestResult

# Zivot-Andrews critical values (Table 4, Zivot & Andrews 1992)
_ZA_CV: dict[str, dict[str, float]] = {
    "a": {"1%": -5.34, "5%": -4.80, "10%": -4.58},
    "b": {"1%": -4.93, "5%": -4.42, "10%": -4.11},
    "c": {"1%": -5.57, "5%": -5.08, "10%": -4.82},
}


def zivot_andrews_test(
    y: NDArray[np.floating[Any]],
    model: str = "c",
    maxlag: int | None = None,
    trim: float = 0.15,
    autolag: str = "AIC",
) -> TestResult:
    """Zivot-Andrews unit root test with endogenous structural break.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series data. Must be 1-dimensional.
    model : str, default 'c'
        Break specification:
        - 'a': break in intercept only
        - 'b': break in trend only
        - 'c': break in both intercept and trend
    maxlag : int or None, default None
        Maximum number of lags. If None, uses int(12*(T/100)^{1/4}).
    trim : float, default 0.15
        Fraction of data to trim from each end when searching for break.
    autolag : str, default 'AIC'
        Lag selection method: 'AIC' or 'BIC'.

    Returns
    -------
    TestResult
        Standardized test result with:
        - statistic: minimum t-statistic across all break dates
        - additional_info: {'break_date': int, 'break_fraction': float,
                            'model': str}
    """
    y = np.asarray(y, dtype=float).ravel()
    nobs = len(y)

    if nobs < 20:
        raise ValueError(
            f"Series too short: T={nobs}, need at least 20 observations."
        )

    model = model.lower()
    if model not in ("a", "b", "c"):
        raise ValueError(f"Invalid model '{model}'. Choose from 'a', 'b', 'c'.")

    if maxlag is None:
        maxlag = int(12.0 * (nobs / 100.0) ** 0.25)
    maxlag = min(maxlag, nobs // 4)

    dy = np.diff(y)  # (nobs-1,)

    start = int(np.ceil(trim * nobs))
    end = int(np.floor((1.0 - trim) * nobs))

    min_t_stat = np.inf
    best_break = start
    best_lag_final = 0

    for tb in range(start, end + 1):
        # Find best lag for this break date
        best_lag_tb = 0
        best_ic_tb = np.inf

        for p in range(0, maxlag + 1):
            dy_dep = dy[p:]
            y_lag1 = y[p : nobs - 1]
            n = len(dy_dep)

            if n < p + 5:
                continue

            # Build regressor matrix
            t_trend = np.arange(p + 1, nobs, dtype=float)

            # Break dummies (relative to original series index)
            du = np.zeros(n)
            dt_dummy = np.zeros(n)
            for i in range(n):
                orig_t = p + 1 + i  # original time index
                if orig_t > tb:
                    du[i] = 1.0
                    dt_dummy[i] = float(orig_t - tb)

            regressors: list[NDArray[np.floating[Any]]] = []
            regressors.append(np.ones(n))  # constant
            regressors.append(t_trend)  # trend
            regressors.append(y_lag1)  # y_{t-1}

            if model in ("a", "c"):
                regressors.append(du)
            if model in ("b", "c"):
                regressors.append(dt_dummy)

            # Lagged diffs
            for i in range(1, p + 1):
                lag_diff = dy[p - i : nobs - 1 - i]
                regressors.append(lag_diff)

            x_mat = np.column_stack(regressors)
            k = x_mat.shape[1]

            try:
                beta = np.linalg.solve(x_mat.T @ x_mat, x_mat.T @ dy_dep)
                resid = dy_dep - x_mat @ beta
                ssr = float(np.sum(resid**2))

                if autolag.upper() == "AIC":
                    ic = n * np.log(ssr / n) + 2 * k
                else:
                    ic = n * np.log(ssr / n) + k * np.log(n)

                if ic < best_ic_tb:
                    best_ic_tb = ic
                    best_lag_tb = p
            except np.linalg.LinAlgError:
                continue

        # Estimate with best lag for this break date
        p = best_lag_tb
        dy_dep = dy[p:]
        y_lag1 = y[p : nobs - 1]
        n = len(dy_dep)
        t_trend = np.arange(p + 1, nobs, dtype=float)

        du = np.zeros(n)
        dt_dummy = np.zeros(n)
        for i in range(n):
            orig_t = p + 1 + i
            if orig_t > tb:
                du[i] = 1.0
                dt_dummy[i] = float(orig_t - tb)

        regressors_final: list[NDArray[np.floating[Any]]] = []
        regressors_final.append(np.ones(n))
        regressors_final.append(t_trend)
        regressors_final.append(y_lag1)

        gamma_idx = 2  # index of y_{t-1} coefficient

        if model in ("a", "c"):
            regressors_final.append(du)
        if model in ("b", "c"):
            regressors_final.append(dt_dummy)

        for i in range(1, p + 1):
            lag_diff = dy[p - i : nobs - 1 - i]
            regressors_final.append(lag_diff)

        x_mat = np.column_stack(regressors_final)
        k = x_mat.shape[1]

        try:
            beta = np.linalg.solve(x_mat.T @ x_mat, x_mat.T @ dy_dep)
            resid = dy_dep - x_mat @ beta
            ssr = float(np.sum(resid**2))
            sigma2 = ssr / (n - k)
            var_beta = sigma2 * np.linalg.inv(x_mat.T @ x_mat)
            se = np.sqrt(np.diag(var_beta))

            t_gamma = float(beta[gamma_idx]) / float(se[gamma_idx])

            if t_gamma < min_t_stat:
                min_t_stat = t_gamma
                best_break = tb
                best_lag_final = p
        except np.linalg.LinAlgError:
            continue

    cv = _ZA_CV[model]
    reject = min_t_stat < cv["5%"]

    return TestResult(
        test_name="Zivot-Andrews",
        statistic=min_t_stat,
        pvalue=None,
        critical_values=cv,
        null_hypothesis="Unit root (no structural break)",
        alternative_hypothesis="Trend-stationary with structural break",
        reject_at_5pct=reject,
        lags_used=best_lag_final,
        additional_info={
            "break_date": best_break,
            "break_fraction": best_break / nobs,
            "model": model,
        },
    )
