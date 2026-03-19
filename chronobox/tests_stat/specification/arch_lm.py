"""
ARCH-LM test for conditional heteroskedasticity.

    Auxiliary regression: e_t^2 = a0 + a1*e_{t-1}^2 + ... + aq*e_{t-q}^2 + v_t
    LM = T * R^2 ~ chi2(q)

    H0: no ARCH effects (conditional homoskedasticity)
    H1: ARCH effects present (conditional heteroskedasticity)

References
----------
- Engle, R.F. (1982). Autoregressive conditional heteroscedasticity with
  estimates of the variance of United Kingdom inflation. Econometrica,
  50(4), 987-1007.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats  # type: ignore[reportMissingTypeStubs]

from chronobox.tests_stat.base import TestResult


def arch_lm_test(
    residuals: NDArray[np.floating],
    nlags: int = 1,
) -> TestResult:
    """ARCH-LM test for conditional heteroskedasticity.

    Parameters
    ----------
    residuals : array_like, shape (T,)
        Residuals from a fitted model.
    nlags : int, default 1
        Number of ARCH lags to test (q).

    Returns
    -------
    TestResult
        LM statistic with chi-squared p-value.
    """
    residuals = np.asarray(residuals, dtype=float).ravel()
    nobs = len(residuals)

    e2 = residuals**2

    # Build auxiliary regression: e_t^2 = a0 + a1*e_{t-1}^2 + ... + aq*e_{t-q}^2
    dep = e2[nlags:]
    n = len(dep)

    x_aux = np.ones((n, nlags + 1))
    for j in range(1, nlags + 1):
        x_aux[:, j] = e2[nlags - j : nobs - j]

    # OLS
    beta = np.linalg.solve(x_aux.T @ x_aux, x_aux.T @ dep)
    resid = dep - x_aux @ beta

    # R^2
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((dep - np.mean(dep)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    # LM statistic
    lm_stat = n * r_squared

    pvalue = 1.0 - float(sp_stats.chi2.cdf(lm_stat, nlags))  # type: ignore[reportUnknownMemberType]
    reject = pvalue < 0.05

    return TestResult(
        test_name="ARCH-LM",
        statistic=lm_stat,
        pvalue=pvalue,
        critical_values={
            "5%": float(sp_stats.chi2.ppf(0.95, nlags)),  # type: ignore[reportUnknownMemberType]
            "1%": float(sp_stats.chi2.ppf(0.99, nlags)),  # type: ignore[reportUnknownMemberType]
        },
        null_hypothesis=f"No ARCH effects up to order {nlags}",
        alternative_hypothesis="ARCH effects present",
        reject_at_5pct=reject,
        lags_used=nlags,
        additional_info={
            "LM": lm_stat,
            "R_squared": r_squared,
            "nobs": n,
        },
    )
