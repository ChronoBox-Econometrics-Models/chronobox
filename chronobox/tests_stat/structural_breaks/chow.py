"""
Chow test for structural break at a known break point.

Tests whether the regression coefficients differ between two sub-samples
split at a known break point.

    F = ((SSR - SSR_1 - SSR_2) / k) / ((SSR_1 + SSR_2) / (T - 2k))
    F ~ F(k, T - 2k)

    H0: coefficients are constant (no structural break)
    H1: coefficients change at break point

References
----------
- Chow, G.C. (1960). Tests of equality between sets of coefficients in two
  linear regressions. Econometrica, 28(3), 591-605.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats  # type: ignore[reportMissingTypeStubs]

from chronobox.tests_stat.base import TestResult


def chow_test(
    y: NDArray[np.floating],
    x_mat: NDArray[np.floating],
    break_point: int,
) -> TestResult:
    """Chow test for structural break at a known break point.

    Parameters
    ----------
    y : array_like, shape (T,)
        Dependent variable.
    x_mat : array_like, shape (T, k)
        Regressor matrix (should include constant if desired).
    break_point : int
        Index at which the break occurs. Observations [0, break_point) are
        in sub-sample 1, and [break_point, T) in sub-sample 2.

    Returns
    -------
    TestResult
        With statistic = F, pvalue from F-distribution,
        additional_info containing SSR values.

    Raises
    ------
    ValueError
        If break_point is too close to the edges (need k obs in each sub-sample).
    """
    y = np.asarray(y, dtype=float).ravel()
    x_mat = np.asarray(x_mat, dtype=float)
    if x_mat.ndim == 1:
        x_mat = x_mat.reshape(-1, 1)

    n_obs, k = x_mat.shape
    if len(y) != n_obs:
        raise ValueError(f"y and X must have same length: {len(y)} != {n_obs}")

    if break_point < k or break_point > n_obs - k:
        raise ValueError(
            f"break_point={break_point} too close to edge. "
            f"Need at least k={k} observations in each sub-sample."
        )

    # Full sample
    beta_full = np.linalg.solve(x_mat.T @ x_mat, x_mat.T @ y)
    resid_full = y - x_mat @ beta_full
    ssr = float(np.sum(resid_full**2))

    # Sub-sample 1: [0, break_point)
    y1, x1 = y[:break_point], x_mat[:break_point]
    beta1 = np.linalg.solve(x1.T @ x1, x1.T @ y1)
    resid1 = y1 - x1 @ beta1
    ssr1 = float(np.sum(resid1**2))

    # Sub-sample 2: [break_point, T)
    y2, x2 = y[break_point:], x_mat[break_point:]
    beta2 = np.linalg.solve(x2.T @ x2, x2.T @ y2)
    resid2 = y2 - x2 @ beta2
    ssr2 = float(np.sum(resid2**2))

    # F-statistic
    df1 = k
    df2 = n_obs - 2 * k
    f_stat = ((ssr - ssr1 - ssr2) / df1) / ((ssr1 + ssr2) / df2)

    # p-value from F-distribution
    pvalue = 1.0 - float(sp_stats.f.cdf(f_stat, df1, df2))  # type: ignore[reportUnknownMemberType]

    reject = pvalue < 0.05

    return TestResult(
        test_name="Chow",
        statistic=f_stat,
        pvalue=pvalue,
        critical_values={
            "1%": float(sp_stats.f.ppf(0.99, df1, df2)),  # type: ignore[reportUnknownMemberType]
            "5%": float(sp_stats.f.ppf(0.95, df1, df2)),  # type: ignore[reportUnknownMemberType]
            "10%": float(sp_stats.f.ppf(0.90, df1, df2)),  # type: ignore[reportUnknownMemberType]
        },
        null_hypothesis="Coefficients are constant (no structural break)",
        alternative_hypothesis=f"Structural break at index {break_point}",
        reject_at_5pct=reject,
        additional_info={
            "break_point": break_point,
            "SSR_full": ssr,
            "SSR_1": ssr1,
            "SSR_2": ssr2,
            "T1": break_point,
            "T2": n_obs - break_point,
            "df1": df1,
            "df2": df2,
        },
    )
