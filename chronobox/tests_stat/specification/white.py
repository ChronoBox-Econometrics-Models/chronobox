"""
White test for heteroskedasticity.

    Auxiliary regression: e^2 = a + b*X + c*X^2 + d*cross_terms + v
    W = T * R^2 ~ chi2(q)

    H0: homoskedasticity
    H1: heteroskedasticity (general form)

References
----------
- White, H. (1980). A heteroskedasticity-consistent covariance matrix
  estimator and a direct test for heteroskedasticity. Econometrica,
  48(4), 817-838.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats  # type: ignore[reportMissingTypeStubs]

from chronobox.tests_stat.base import TestResult


def white_test(
    residuals: NDArray[np.floating],
    x_regressors: NDArray[np.floating],
    cross_terms: bool = True,
) -> TestResult:
    """White test for heteroskedasticity.

    Parameters
    ----------
    residuals : array_like, shape (T,)
        OLS residuals.
    X : array_like, shape (T, k)
        Original regressor matrix (excluding constant).
    cross_terms : bool, default True
        Whether to include cross products of regressors.

    Returns
    -------
    TestResult
        W statistic with chi-squared p-value.
    """
    residuals = np.asarray(residuals, dtype=float).ravel()
    x_regressors = np.asarray(x_regressors, dtype=float)
    if x_regressors.ndim == 1:
        x_regressors = x_regressors.reshape(-1, 1)

    nobs, k = x_regressors.shape
    e2 = residuals**2

    # Build auxiliary regressors: constant + X + X^2 + cross terms
    aux_regs: list[NDArray[np.floating]] = [np.ones((nobs, 1))]
    aux_regs.append(x_regressors)
    aux_regs.append(x_regressors**2)

    if cross_terms and k > 1:
        for i in range(k):
            for j in range(i + 1, k):
                aux_regs.append((x_regressors[:, i] * x_regressors[:, j]).reshape(-1, 1))

    x_aux = np.column_stack(aux_regs)
    q = x_aux.shape[1] - 1  # exclude constant for df

    # OLS
    try:
        beta = np.linalg.solve(x_aux.T @ x_aux, x_aux.T @ e2)
        resid = e2 - x_aux @ beta
    except np.linalg.LinAlgError:
        beta, _, _, _ = np.linalg.lstsq(x_aux, e2, rcond=None)
        resid = e2 - x_aux @ beta

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((e2 - np.mean(e2)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    w_stat = nobs * r_squared

    pvalue = 1.0 - float(sp_stats.chi2.cdf(w_stat, q))  # type: ignore[reportUnknownMemberType]
    reject = pvalue < 0.05

    return TestResult(
        test_name="White",
        statistic=w_stat,
        pvalue=pvalue,
        critical_values={
            "5%": float(sp_stats.chi2.ppf(0.95, q)),  # type: ignore[reportUnknownMemberType]
            "1%": float(sp_stats.chi2.ppf(0.99, q)),  # type: ignore[reportUnknownMemberType]
        },
        null_hypothesis="Homoskedasticity",
        alternative_hypothesis="Heteroskedasticity (general form)",
        reject_at_5pct=reject,
        additional_info={
            "W": w_stat,
            "R_squared": r_squared,
            "df": q,
            "nobs": nobs,
            "cross_terms": cross_terms,
        },
    )
