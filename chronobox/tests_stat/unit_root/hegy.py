"""
HEGY test for seasonal unit roots.

Tests for unit roots at seasonal frequencies in quarterly data.

    Frequencies tested:
    - Frequency 0: pi_1 = 0 (non-seasonal unit root)
    - Frequency pi: pi_2 = 0 (semi-annual unit root)
    - Frequency pi/2: pi_3 = pi_4 = 0 (annual unit root)

    Transformations for quarterly data (s=4):
    y_{1t} = (1+L)(1+L^2) * y_t = y_t + y_{t-1} + y_{t-2} + y_{t-3}
    y_{2t} = -(1-L)(1+L^2) * y_t = -y_t + y_{t-1} - y_{t-2} + y_{t-3}
    y_{3t} = -(1-L^2) * y_t = -y_t + y_{t-2}

References
----------
- Hylleberg, S., Engle, R.F., Granger, C.W.J. & Yoo, B.S. (1990).
  Seasonal integration and cointegration. Journal of Econometrics,
  44(1-2), 215-238.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from chronobox.tests_stat.base import TestResult

# HEGY approximate critical values for quarterly data, T=100,
# with constant + seasonal dummies
_HEGY_CV: dict[str, dict[str, float]] = {
    "t_pi1": {"1%": -3.48, "5%": -2.88, "10%": -2.58},
    "t_pi2": {"1%": -3.48, "5%": -2.88, "10%": -2.58},
    "F_pi3pi4": {"1%": 8.65, "5%": 6.07, "10%": 4.79},
}


def hegy_test(
    y: NDArray[np.floating[Any]],
    period: int = 4,
    regression: str = "c",
    maxlag: int | None = None,
) -> TestResult:
    """HEGY test for seasonal unit roots (quarterly data).

    Parameters
    ----------
    y : array_like, shape (T,)
        Quarterly time series. T should be a multiple of 4.
    period : int, default 4
        Seasonal period (currently only 4 for quarterly is supported).
    regression : str, default 'c'
        Deterministic terms:
        - 'n': no deterministics
        - 'c': constant + seasonal dummies
        - 'ct': constant + trend + seasonal dummies
    maxlag : int or None, default None
        Maximum lags of Delta_4 y_t. If None, uses int(4*(T/100)^{1/4}).

    Returns
    -------
    TestResult
        With additional_info containing:
        - 't_pi1': t-stat for frequency 0
        - 't_pi2': t-stat for frequency pi
        - 'F_pi3pi4': F-stat for frequency pi/2
        - 'pi_estimates': (pi1, pi2, pi3, pi4)
    """
    y = np.asarray(y, dtype=float).ravel()
    nobs = len(y)

    if period != 4:
        raise ValueError("Currently only period=4 (quarterly) is supported.")

    if nobs < 20:
        raise ValueError(
            f"Series too short: T={nobs}, need at least 20 observations."
        )

    if regression not in ("n", "c", "ct"):
        raise ValueError(
            f"Invalid regression '{regression}'. Choose 'n', 'c', 'ct'."
        )

    if maxlag is None:
        maxlag = int(4.0 * (nobs / 100.0) ** 0.25)

    # Compute HEGY transformations
    # y_{1t} = y_t + y_{t-1} + y_{t-2} + y_{t-3}
    y1 = np.zeros(nobs)
    for t in range(3, nobs):
        y1[t] = y[t] + y[t - 1] + y[t - 2] + y[t - 3]

    # y_{2t} = -y_t + y_{t-1} - y_{t-2} + y_{t-3}
    y2 = np.zeros(nobs)
    for t in range(3, nobs):
        y2[t] = -y[t] + y[t - 1] - y[t - 2] + y[t - 3]

    # y_{3t} = -y_t + y_{t-2}
    y3 = np.zeros(nobs)
    for t in range(2, nobs):
        y3[t] = -y[t] + y[t - 2]

    # Dependent variable: Delta_4 y_t = y_t - y_{t-4}
    dy4 = np.zeros(nobs)
    for t in range(4, nobs):
        dy4[t] = y[t] - y[t - 4]

    # Effective sample starts after max(4, 4+maxlag)
    start = 4 + maxlag
    n = nobs - start

    if n < 10:
        raise ValueError(
            f"Not enough observations after transformation: n={n}."
        )

    # Build regression matrix
    dep = dy4[start:]

    regressors: list[NDArray[np.floating[Any]]] = []
    # y_{1,t-1}
    regressors.append(y1[start - 1 : nobs - 1])
    # y_{2,t-1}
    regressors.append(y2[start - 1 : nobs - 1])
    # y_{3,t-2}
    regressors.append(y3[start - 2 : nobs - 2])
    # y_{3,t-1}
    regressors.append(y3[start - 1 : nobs - 1])

    # Lagged Delta_4 y
    for j in range(1, maxlag + 1):
        regressors.append(dy4[start - j : nobs - j])

    # Deterministic terms
    if regression in ("c", "ct"):
        regressors.append(np.ones(n))
        # Seasonal dummies
        for s in range(3):
            sd = np.zeros(n)
            for i in range(n):
                if (start + i) % 4 == s:
                    sd[i] = 1.0
            regressors.append(sd)
    if regression == "ct":
        regressors.append(np.arange(start + 1, nobs + 1, dtype=float))

    x_mat = np.column_stack(regressors)
    k = x_mat.shape[1]

    # OLS
    beta = np.linalg.solve(x_mat.T @ x_mat, x_mat.T @ dep)
    resid = dep - x_mat @ beta
    ssr = float(np.sum(resid**2))
    sigma2 = ssr / (n - k)
    var_beta = sigma2 * np.linalg.inv(x_mat.T @ x_mat)
    se = np.sqrt(np.diag(var_beta))

    pi1 = float(beta[0])
    pi2 = float(beta[1])
    pi3 = float(beta[2])
    pi4 = float(beta[3])

    t_pi1 = pi1 / float(se[0])
    t_pi2 = pi2 / float(se[1])

    # F-test for pi3 = pi4 = 0
    # Restricted model (without y3 terms)
    regressors_r: list[NDArray[np.floating[Any]]] = []
    regressors_r.append(y1[start - 1 : nobs - 1])
    regressors_r.append(y2[start - 1 : nobs - 1])
    # Skip y3 terms
    for j in range(1, maxlag + 1):
        regressors_r.append(dy4[start - j : nobs - j])
    if regression in ("c", "ct"):
        regressors_r.append(np.ones(n))
        for s in range(3):
            sd = np.zeros(n)
            for i in range(n):
                if (start + i) % 4 == s:
                    sd[i] = 1.0
            regressors_r.append(sd)
    if regression == "ct":
        regressors_r.append(np.arange(start + 1, nobs + 1, dtype=float))

    x_mat_r = np.column_stack(regressors_r)
    beta_r = np.linalg.solve(x_mat_r.T @ x_mat_r, x_mat_r.T @ dep)
    resid_r = dep - x_mat_r @ beta_r
    ssr_r = float(np.sum(resid_r**2))

    # F = ((SSR_r - SSR_u) / q) / (SSR_u / (n - k))
    q = 2  # number of restrictions
    f_pi3pi4 = ((ssr_r - ssr) / q) / (ssr / (n - k))

    cv = _HEGY_CV

    # Reject any seasonal unit root?
    reject_freq0 = t_pi1 < cv["t_pi1"]["5%"]
    reject_freqpi = t_pi2 < cv["t_pi2"]["5%"]
    reject_freqpi2 = f_pi3pi4 > cv["F_pi3pi4"]["5%"]

    # Overall: the "main" result focuses on frequency 0
    main_stat = t_pi1

    return TestResult(
        test_name="HEGY",
        statistic=main_stat,
        pvalue=None,
        critical_values=cv["t_pi1"],
        null_hypothesis="Seasonal unit roots present",
        alternative_hypothesis="No seasonal unit roots",
        reject_at_5pct=reject_freq0,
        lags_used=maxlag,
        additional_info={
            "t_pi1": t_pi1,
            "t_pi2": t_pi2,
            "F_pi3pi4": f_pi3pi4,
            "pi_estimates": (pi1, pi2, pi3, pi4),
            "reject_freq0": reject_freq0,
            "reject_freqpi": reject_freqpi,
            "reject_freqpi2": reject_freqpi2,
            "cv_t_pi1": cv["t_pi1"],
            "cv_t_pi2": cv["t_pi2"],
            "cv_F_pi3pi4": cv["F_pi3pi4"],
            "nobs": n,
        },
    )
