"""
Durbin-Watson test for first-order autocorrelation.

    DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
    DW ~ 2 - 2*rho_1

    DW ~ 2: no autocorrelation
    DW ~ 0: positive autocorrelation
    DW ~ 4: negative autocorrelation

References
----------
- Durbin, J. & Watson, G.S. (1950, 1951). Testing for serial correlation
  in least squares regression. Biometrika, 37-38.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chronobox.tests_stat.base import TestResult


def durbin_watson_test(
    residuals: NDArray[np.floating],
) -> TestResult:
    """Durbin-Watson test for first-order autocorrelation.

    Parameters
    ----------
    residuals : array_like, shape (T,)
        OLS residuals.

    Returns
    -------
    TestResult
        DW statistic (values near 2 = no autocorrelation).
    """
    residuals = np.asarray(residuals, dtype=float).ravel()
    nobs = len(residuals)

    # DW statistic
    diff_resid = np.diff(residuals)
    ss_diff = float(np.sum(diff_resid**2))
    ss_resid = float(np.sum(residuals**2))

    dw_stat = 2.0 if ss_resid < 1e-15 else ss_diff / ss_resid

    # Approximate rho_1
    rho_1 = 1.0 - dw_stat / 2.0

    # Interpretation
    if dw_stat < 1.5:
        interpretation = "Positive autocorrelation likely"
    elif dw_stat > 2.5:
        interpretation = "Negative autocorrelation likely"
    else:
        interpretation = "No strong evidence of autocorrelation"

    # Approximate test: DW significantly different from 2?
    reject = dw_stat < 1.5 or dw_stat > 2.5

    return TestResult(
        test_name="Durbin-Watson",
        statistic=dw_stat,
        pvalue=None,
        critical_values={"lower_bound": 1.5, "upper_bound": 2.5},
        null_hypothesis="No first-order autocorrelation (DW ~ 2)",
        alternative_hypothesis="First-order autocorrelation present",
        reject_at_5pct=reject,
        additional_info={
            "DW": dw_stat,
            "rho_1_approx": rho_1,
            "interpretation": interpretation,
            "nobs": nobs,
        },
    )
