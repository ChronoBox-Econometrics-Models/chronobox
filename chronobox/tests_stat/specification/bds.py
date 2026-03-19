"""
BDS test for independence (iid).

Tests the null hypothesis that the data are independently and identically
distributed using the correlation integral.

    BDS_m(eps) = sqrt(T) * (C_m(eps) - C_1(eps)^m) / sigma_m

    H0: data are iid
    H1: dependence (linear or nonlinear)

References
----------
- Brock, W.A., Dechert, W.D. & Scheinkman, J.A. (1996). A test for
  independence based on the correlation dimension. Econometric Reviews,
  15(3), 197-235.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats  # type: ignore[reportMissingTypeStubs]

from chronobox.tests_stat.base import TestResult


def _correlation_integral(
    x: NDArray[np.floating],
    m: int,
    epsilon: float,
) -> float:
    """Compute the correlation integral C_m(epsilon)."""
    n = len(x)
    n_m = n - m + 1

    if n_m < 2:
        return 0.0

    # Build m-dimensional embedded vectors
    embedded = np.zeros((n_m, m))
    for i in range(m):
        embedded[:, i] = x[i : i + n_m]

    # Count pairs with sup-norm < epsilon
    count = 0
    for i in range(n_m):
        diffs = np.abs(embedded[i + 1 :] - embedded[i])
        max_diffs = np.max(diffs, axis=1) if m > 1 else diffs.ravel()
        count += int(np.sum(max_diffs < epsilon))

    c_m = 2.0 * count / (n_m * (n_m - 1))
    return c_m


def bds_test(
    residuals: NDArray[np.floating],
    max_dim: int = 6,
    epsilon: float | None = None,
) -> TestResult:
    """BDS test for independence (iid).

    Parameters
    ----------
    residuals : array_like, shape (T,)
        Residuals or data to test.
    max_dim : int, default 6
        Maximum embedding dimension (tests m=2..max_dim).
    epsilon : float or None, default None
        Distance threshold. If None, uses 0.7 * std(residuals).

    Returns
    -------
    TestResult
        BDS statistic for the default dimension (m=2), with
        additional_info containing statistics for all dimensions.
    """
    residuals = np.asarray(residuals, dtype=float).ravel()
    nobs = len(residuals)

    if epsilon is None:
        epsilon = 0.7 * float(np.std(residuals))

    if epsilon <= 0:
        epsilon = 1.0

    # Standardize
    std = float(np.std(residuals))
    if std < 1e-15:
        std = 1.0
    x = (residuals - np.mean(residuals)) / std
    epsilon_std = epsilon / std

    # C_1
    c_1 = _correlation_integral(x, 1, epsilon_std)

    bds_stats: dict[int, float] = {}
    bds_pvals: dict[int, float] = {}

    for m in range(2, max_dim + 1):
        c_m = _correlation_integral(x, m, epsilon_std)

        # Simplified variance estimate
        sigma2 = (
            4.0
            * m**2
            * max(c_1, 1e-10) ** (2 * (m - 1))
            * max(1.0 - c_1, 1e-10) ** 2
        )

        sigma = np.sqrt(max(sigma2, 1e-15))

        # BDS statistic
        bds_m = (
            np.sqrt(nobs) * (c_m - c_1**m) / sigma if sigma > 1e-10 else 0.0
        )
        bds_stats[m] = float(bds_m)

        # Two-sided p-value (asymptotically normal)
        pval = 2.0 * (1.0 - float(sp_stats.norm.cdf(abs(bds_m))))  # type: ignore[reportUnknownMemberType]
        bds_pvals[m] = pval

    # Main result: m=2
    main_stat = bds_stats.get(2, 0.0)
    main_pval = bds_pvals.get(2, 1.0)

    reject = main_pval < 0.05

    return TestResult(
        test_name="BDS",
        statistic=main_stat,
        pvalue=main_pval,
        critical_values={
            "5%": 1.96,
            "1%": 2.576,
        },
        null_hypothesis="Data are iid (independent and identically distributed)",
        alternative_hypothesis="Dependence present (linear or nonlinear)",
        reject_at_5pct=reject,
        additional_info={
            "bds_stats": bds_stats,
            "bds_pvalues": bds_pvals,
            "epsilon": epsilon,
            "max_dim": max_dim,
            "C_1": c_1,
            "nobs": nobs,
        },
    )
