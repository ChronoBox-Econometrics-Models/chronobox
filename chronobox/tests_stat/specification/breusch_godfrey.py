"""
Breusch-Godfrey test for serial correlation (LM test).

    Auxiliary regression: e_t = X_t*gamma + sum(alpha_j*e_{t-j}) + v_t
    LM = T * R^2 ~ chi2(p)

    H0: no serial correlation up to order p
    H1: serial correlation present

References
----------
- Breusch, T.S. (1978). Testing for autocorrelation in dynamic linear models.
  Australian Economic Papers, 17(31), 334-355.
- Godfrey, L.G. (1978). Testing against general autoregressive and moving
  average error models when the regressors include lagged dependent variables.
  Econometrica, 46(6), 1293-1301.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats  # type: ignore[reportMissingTypeStubs]

from chronobox.tests_stat.base import TestResult


def breusch_godfrey_test(
    residuals: NDArray[np.floating],
    x_regressors: NDArray[np.floating],
    nlags: int = 1,
) -> TestResult:
    """Breusch-Godfrey LM test for serial correlation.

    Parameters
    ----------
    residuals : array_like, shape (T,)
        OLS residuals from the original regression.
    X : array_like, shape (T, k)
        Original regressor matrix.
    nlags : int, default 1
        Number of lags to test for serial correlation.

    Returns
    -------
    TestResult
        LM statistic with chi-squared and F p-values.
    """
    residuals = np.asarray(residuals, dtype=float).ravel()
    x_regressors = np.asarray(x_regressors, dtype=float)
    if x_regressors.ndim == 1:
        x_regressors = x_regressors.reshape(-1, 1)

    nobs, k = x_regressors.shape

    # Build auxiliary regression
    # e_t = X_t * gamma + alpha_1*e_{t-1} + ... + alpha_p*e_{t-p} + v_t
    # Use e_{t-j} = 0 for t-j < 0

    lagged_resid = np.zeros((nobs, nlags))
    for j in range(1, nlags + 1):
        lagged_resid[j:, j - 1] = residuals[:-j]

    x_aux = np.column_stack([x_regressors, lagged_resid])

    # OLS auxiliary regression
    beta_aux = np.linalg.solve(x_aux.T @ x_aux, x_aux.T @ residuals)
    resid_aux = residuals - x_aux @ beta_aux

    # R^2 of auxiliary regression
    ss_res = float(np.sum(resid_aux**2))
    ss_tot = float(np.sum((residuals - np.mean(residuals)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    # LM statistic
    lm_stat = nobs * r_squared

    # F statistic
    f_stat = (
        (r_squared / nlags) / ((1.0 - r_squared) / (nobs - k - nlags))
        if (1.0 - r_squared) > 1e-15
        else 0.0
    )

    pvalue_lm = 1.0 - float(sp_stats.chi2.cdf(lm_stat, nlags))  # type: ignore[reportUnknownMemberType]
    pvalue_f = 1.0 - float(sp_stats.f.cdf(f_stat, nlags, nobs - k - nlags))  # type: ignore[reportUnknownMemberType]

    reject = pvalue_lm < 0.05

    return TestResult(
        test_name="Breusch-Godfrey",
        statistic=lm_stat,
        pvalue=pvalue_lm,
        critical_values={
            "5%": float(sp_stats.chi2.ppf(0.95, nlags)),  # type: ignore[reportUnknownMemberType]
            "1%": float(sp_stats.chi2.ppf(0.99, nlags)),  # type: ignore[reportUnknownMemberType]
        },
        null_hypothesis=f"No serial correlation up to order {nlags}",
        alternative_hypothesis="Serial correlation present",
        reject_at_5pct=reject,
        lags_used=nlags,
        additional_info={
            "LM": lm_stat,
            "F_stat": f_stat,
            "pvalue_F": pvalue_f,
            "R_squared": r_squared,
            "nobs": nobs,
        },
    )
