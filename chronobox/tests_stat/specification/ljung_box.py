"""
Ljung-Box test for serial correlation in residuals.

    Q(m) = T*(T+2) * sum(rho_k^2/(T-k), k=1..m)
    Q(m) ~ chi2(m - model_df)

    H0: no autocorrelation up to lag m
    H1: autocorrelation present at some lag

References
----------
- Ljung, G.M. & Box, G.E.P. (1978). On a measure of lack of fit in time
  series models. Biometrika, 65(2), 297-303.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats  # type: ignore[reportMissingTypeStubs]

from chronobox.tests_stat.base import TestResult


def ljung_box_test(
    residuals: NDArray[np.floating],
    lags: int = 10,
    model_df: int = 0,
) -> TestResult:
    """Ljung-Box test for serial correlation.

    Parameters
    ----------
    residuals : array_like, shape (T,)
        Residuals from a fitted model.
    lags : int, default 10
        Number of lags to test.
    model_df : int, default 0
        Degrees of freedom used by the model (p+q for ARMA(p,q)).
        Adjusts chi-squared df to (lags - model_df).

    Returns
    -------
    TestResult
        Q statistic with chi-squared p-value.
    """
    residuals = np.asarray(residuals, dtype=float).ravel()
    nobs = len(residuals)

    if lags >= nobs:
        lags = nobs - 1

    # Sample autocorrelations
    mean = np.mean(residuals)
    centered = residuals - mean
    var = float(np.sum(centered**2)) / nobs

    if var < 1e-15:
        return TestResult(
            test_name="Ljung-Box",
            statistic=0.0,
            pvalue=1.0,
            critical_values={},
            null_hypothesis="No serial correlation up to lag " + str(lags),
            alternative_hypothesis="Serial correlation present",
            reject_at_5pct=False,
            lags_used=lags,
        )

    rho = np.zeros(lags)
    for k in range(1, lags + 1):
        rho[k - 1] = float(np.sum(centered[k:] * centered[:-k])) / (nobs * var)

    # Ljung-Box Q statistic
    q_stat = nobs * (nobs + 2.0) * float(np.sum(rho**2 / (nobs - np.arange(1, lags + 1))))

    # Degrees of freedom
    df = lags - model_df
    if df <= 0:
        df = 1

    pvalue = 1.0 - float(sp_stats.chi2.cdf(q_stat, df))  # type: ignore[reportUnknownMemberType]

    reject = pvalue < 0.05

    return TestResult(
        test_name="Ljung-Box",
        statistic=q_stat,
        pvalue=pvalue,
        critical_values={
            "5%": float(sp_stats.chi2.ppf(0.95, df)),  # type: ignore[reportUnknownMemberType]
            "1%": float(sp_stats.chi2.ppf(0.99, df)),  # type: ignore[reportUnknownMemberType]
        },
        null_hypothesis=f"No serial correlation up to lag {lags}",
        alternative_hypothesis="Serial correlation present",
        reject_at_5pct=reject,
        lags_used=lags,
        additional_info={
            "Q": q_stat,
            "df": df,
            "model_df": model_df,
            "autocorrelations": rho.tolist(),
        },
    )
