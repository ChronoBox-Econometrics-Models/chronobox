"""
KPSS test for stationarity.

Tests the null hypothesis that the series is stationary (level or trend)
against the alternative of a unit root.

    H0: sigma_u^2 = 0 (series is stationary)
    H1: sigma_u^2 > 0 (unit root present)

    eta = (1/T^2) * sum(S_t^2) / lambda^2

Note: KPSS has REVERSED hypotheses compared to ADF and PP.
Rejection means evidence of unit root (non-stationarity).

References
----------
- Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. (1992).
  Testing the null hypothesis of stationarity against the alternative
  of a unit root. Journal of Econometrics, 54(1-3), 159-178.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from chronobox.tests_stat.base import TestResult

# KPSS asymptotic critical values from Kwiatkowski et al. (1992)
_KPSS_CV: dict[str, dict[str, float]] = {
    "c": {
        "10%": 0.347,
        "5%": 0.463,
        "2.5%": 0.574,
        "1%": 0.739,
    },
    "ct": {
        "10%": 0.119,
        "5%": 0.146,
        "2.5%": 0.176,
        "1%": 0.216,
    },
}


def kpss_test(
    y: NDArray[np.floating[Any]],
    regression: str = "c",
    nlags: int | None = None,
) -> TestResult:
    """KPSS test for stationarity.

    Tests H0: series is stationary vs H1: unit root present.
    NOTE: This test has REVERSED hypotheses compared to ADF and PP.
    Rejection indicates non-stationarity.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series data. Must be 1-dimensional.
    regression : str, default 'c'
        Regression model:
        - 'c': test for level stationarity
        - 'ct': test for trend stationarity
    nlags : int or None, default None
        Bandwidth for the Newey-West estimator.
        If None, uses int(4*(T/100)^{2/9}).

    Returns
    -------
    TestResult
        Standardized test result with:
        - statistic: KPSS eta statistic
        - pvalue: approximate p-value (interpolated from tables)
        - critical_values: critical values at 1%, 5%, 10%
        - additional_info: {'regression': model, 'nlags': bandwidth, 'nobs': T}

    Notes
    -----
    Unlike ADF and PP, KPSS tests the NULL of stationarity.
    - If reject_at_5pct is True: evidence of unit root (non-stationary)
    - If reject_at_5pct is False: no evidence against stationarity

    Raises
    ------
    ValueError
        If y is not 1-dimensional or regression model is invalid.
    """
    y = np.asarray(y, dtype=float).ravel()
    nobs = len(y)

    if nobs < 10:
        raise ValueError(f"Series too short: T={nobs}, need at least 10 observations.")

    if regression not in ("c", "ct"):
        raise ValueError(
            f"Invalid regression model '{regression}'. Choose from 'c', 'ct'."
        )

    if nlags is None:
        nlags = int(4.0 * (nobs / 100.0) ** (2.0 / 9.0))
    nlags = max(1, nlags)

    # OLS regression to get residuals
    if regression == "c":
        x_mat = np.ones((nobs, 1))
    else:  # 'ct'
        x_mat = np.column_stack([np.ones(nobs), np.arange(1, nobs + 1, dtype=float)])

    beta = np.linalg.solve(x_mat.T @ x_mat, x_mat.T @ y)
    residuals = y - x_mat @ beta

    # Partial sums of residuals
    partial_sums = np.cumsum(residuals)

    # Newey-West long-run variance estimate
    gamma_0 = float(np.sum(residuals**2)) / nobs
    lambda_sq = gamma_0
    for j in range(1, nlags + 1):
        w_j = 1.0 - j / (nlags + 1.0)  # Bartlett kernel
        gamma_j = float(np.sum(residuals[j:] * residuals[:-j])) / nobs
        lambda_sq += 2.0 * w_j * gamma_j

    # KPSS statistic
    eta = float(np.sum(partial_sums**2)) / (nobs**2 * lambda_sq)

    # Critical values
    cv = _KPSS_CV[regression]

    # Approximate p-value via interpolation
    pvalue: float
    if eta > cv["1%"]:
        pvalue = 0.005
    elif eta > cv["2.5%"]:
        frac = (eta - cv["2.5%"]) / (cv["1%"] - cv["2.5%"])
        pvalue = 0.025 - frac * (0.025 - 0.01)
    elif eta > cv["5%"]:
        frac = (eta - cv["5%"]) / (cv["2.5%"] - cv["5%"])
        pvalue = 0.05 - frac * (0.05 - 0.025)
    elif eta > cv["10%"]:
        frac = (eta - cv["10%"]) / (cv["5%"] - cv["10%"])
        pvalue = 0.10 - frac * (0.10 - 0.05)
    else:
        pvalue = 0.15  # > 10%

    # For KPSS, reject H0 (stationarity) if stat > critical value
    reject = eta > cv["5%"]

    # Return critical values as dict[str, float]
    cv_return: dict[str, float] = {
        "1%": cv["1%"],
        "5%": cv["5%"],
        "10%": cv["10%"],
    }

    return TestResult(
        test_name="KPSS",
        statistic=eta,
        pvalue=pvalue,
        critical_values=cv_return,
        null_hypothesis=(
            f"Series is {'level' if regression == 'c' else 'trend'}-stationary"
        ),
        alternative_hypothesis="Unit root present (non-stationary)",
        reject_at_5pct=reject,
        lags_used=nlags,
        additional_info={
            "regression": regression,
            "nlags": nlags,
            "nobs": nobs,
            "lambda_sq": lambda_sq,
        },
    )
