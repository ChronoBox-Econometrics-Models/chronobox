"""
Jarque-Bera test for normality.

    JB = (T/6) * (S^2 + (K-3)^2/4)
    JB ~ chi2(2)

    H0: data are normally distributed (S=0, K=3)
    H1: data are not normally distributed

References
----------
- Jarque, C.M. & Bera, A.K. (1980). Efficient tests for normality,
  homoscedasticity and serial independence of regression residuals.
  Economics Letters, 6(3), 255-259.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats  # type: ignore[reportMissingTypeStubs]

from chronobox.tests_stat.base import TestResult


def jarque_bera_test(
    residuals: NDArray[np.floating],
) -> TestResult:
    """Jarque-Bera test for normality.

    Parameters
    ----------
    residuals : array_like, shape (T,)
        Data or residuals to test for normality.

    Returns
    -------
    TestResult
        JB statistic with chi-squared(2) p-value.
    """
    residuals = np.asarray(residuals, dtype=float).ravel()
    nobs = len(residuals)

    mean = float(np.mean(residuals))
    std = float(np.std(residuals, ddof=0))

    if std < 1e-15:
        return TestResult(
            test_name="Jarque-Bera",
            statistic=0.0,
            pvalue=1.0,
            critical_values={"5%": 5.991, "1%": 9.210},
            null_hypothesis="Normality (S=0, K=3)",
            alternative_hypothesis="Non-normality",
            reject_at_5pct=False,
        )

    centered = residuals - mean

    # Skewness: E[(x-mu)^3] / sigma^3
    skew = float(np.mean(centered**3)) / (std**3)

    # Kurtosis: E[(x-mu)^4] / sigma^4
    kurt = float(np.mean(centered**4)) / (std**4)

    # JB statistic
    jb_stat = (nobs / 6.0) * (skew**2 + (kurt - 3.0) ** 2 / 4.0)

    pvalue = 1.0 - float(sp_stats.chi2.cdf(jb_stat, 2))  # type: ignore[reportUnknownMemberType]
    reject = pvalue < 0.05

    return TestResult(
        test_name="Jarque-Bera",
        statistic=jb_stat,
        pvalue=pvalue,
        critical_values={
            "5%": float(sp_stats.chi2.ppf(0.95, 2)),  # type: ignore[reportUnknownMemberType]
            "1%": float(sp_stats.chi2.ppf(0.99, 2)),  # type: ignore[reportUnknownMemberType]
        },
        null_hypothesis="Normality (skewness=0, kurtosis=3)",
        alternative_hypothesis="Non-normality",
        reject_at_5pct=reject,
        additional_info={
            "skewness": skew,
            "kurtosis": kurt,
            "excess_kurtosis": kurt - 3.0,
            "nobs": nobs,
        },
    )
