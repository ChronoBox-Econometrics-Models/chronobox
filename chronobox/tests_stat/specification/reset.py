"""
Ramsey RESET test for functional form misspecification.

    Auxiliary model: y = X*beta + gamma_2*yhat^2 + gamma_3*yhat^3 + ... + eps
    F = ((SSR_0 - SSR_1)/(power-1)) / (SSR_1/(T - n_params))
    F ~ F(power-1, T - n_params)

    H0: model correctly specified
    H1: functional form misspecification (omitted nonlinear terms)

References
----------
- Ramsey, J.B. (1969). Tests for specification errors in classical linear
  least-squares regression analysis. JRSS B, 31(2), 350-371.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats  # type: ignore[reportMissingTypeStubs]

from chronobox.tests_stat.base import TestResult


def reset_test(
    y: NDArray[np.floating],
    x_regressors: NDArray[np.floating],
    power: int = 3,
) -> TestResult:
    """Ramsey RESET test for functional form misspecification.

    Parameters
    ----------
    y : array_like, shape (T,)
        Dependent variable.
    X : array_like, shape (T, k)
        Regressor matrix (should include constant if desired).
    power : int, default 3
        Maximum power of yhat to include (2 means yhat^2 only,
        3 means yhat^2 and yhat^3).

    Returns
    -------
    TestResult
        F statistic with F-distribution p-value.
    """
    y = np.asarray(y, dtype=float).ravel()
    x_regressors = np.asarray(x_regressors, dtype=float)
    if x_regressors.ndim == 1:
        x_regressors = x_regressors.reshape(-1, 1)

    nobs, _ = x_regressors.shape

    # Original model OLS
    beta = np.linalg.solve(x_regressors.T @ x_regressors, x_regressors.T @ y)
    yhat = x_regressors @ beta
    resid = y - yhat
    ssr_0 = float(np.sum(resid**2))

    # Auxiliary model: add yhat^2, yhat^3, ..., yhat^power
    aux_terms: list[NDArray[np.floating]] = []
    for p in range(2, power + 1):
        aux_terms.append((yhat**p).reshape(-1, 1))

    x_aug = np.column_stack([x_regressors, *aux_terms])
    k_aug = x_aug.shape[1]

    beta_aug = np.linalg.solve(x_aug.T @ x_aug, x_aug.T @ y)
    resid_aug = y - x_aug @ beta_aug
    ssr_1 = float(np.sum(resid_aug**2))

    # F-test
    n_restrictions = power - 1
    df1 = n_restrictions
    df2 = nobs - k_aug

    if df2 <= 0 or ssr_1 < 1e-15:
        f_stat = 0.0
        pvalue = 1.0
    else:
        f_stat = ((ssr_0 - ssr_1) / df1) / (ssr_1 / df2)
        pvalue = 1.0 - float(sp_stats.f.cdf(f_stat, df1, df2))  # type: ignore[reportUnknownMemberType]

    reject = pvalue < 0.05

    return TestResult(
        test_name="RESET",
        statistic=f_stat,
        pvalue=pvalue,
        critical_values={
            "5%": float(sp_stats.f.ppf(0.95, df1, max(df2, 1))),  # type: ignore[reportUnknownMemberType]
            "1%": float(sp_stats.f.ppf(0.99, df1, max(df2, 1))),  # type: ignore[reportUnknownMemberType]
        },
        null_hypothesis="Model correctly specified (no omitted nonlinear terms)",
        alternative_hypothesis="Functional form misspecification",
        reject_at_5pct=reject,
        additional_info={
            "F_stat": f_stat,
            "SSR_restricted": ssr_0,
            "SSR_unrestricted": ssr_1,
            "power": power,
            "df1": df1,
            "df2": df2,
            "nobs": nobs,
        },
    )
