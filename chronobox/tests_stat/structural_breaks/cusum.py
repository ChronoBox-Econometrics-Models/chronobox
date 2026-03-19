"""
CUSUM and CUSUM of squares tests for parameter stability.

CUSUM: Tests whether regression coefficients are stable over time using
cumulative sums of recursive residuals.

CUSUMSQ: Tests whether the variance of residuals is stable over time.

    H0: parameter stability (coefficients/variance constant)
    H1: parameter instability (structural change)

References
----------
- Brown, R.L., Durbin, J. & Evans, J.M. (1975). Techniques for testing
  the constancy of regression relationships over time. JRSS B, 37(2), 149-192.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chronobox.tests_stat.base import TestResult


def _recursive_residuals(
    y: NDArray[np.floating],
    x_mat: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute recursive residuals (one-step-ahead prediction errors).

    For t = k+1, ..., T:
        w_t = (y_t - x_t' * beta_{t-1}) / sqrt(1 + x_t' * (X_{t-1}'X_{t-1})^{-1} * x_t)

    Parameters
    ----------
    y : ndarray, shape (T,)
    x_mat : ndarray, shape (T, k)

    Returns
    -------
    w : ndarray, shape (T - k,)
        Recursive residuals.
    """
    n_obs, k = x_mat.shape
    w = np.zeros(n_obs - k)

    # Initialize with first k observations
    x_init = x_mat[:k]
    y_init = y[:k]
    xtx_inv = np.linalg.inv(x_init.T @ x_init)
    beta = xtx_inv @ x_init.T @ y_init

    for t in range(k, n_obs):
        x_t = x_mat[t]
        y_t = y[t]

        # Prediction error
        f_t = 1.0 + float(x_t @ xtx_inv @ x_t)
        e_t = y_t - float(x_t @ beta)
        w[t - k] = e_t / np.sqrt(f_t)

        # Update beta and xtx_inv using Sherman-Morrison formula
        gain = xtx_inv @ x_t / f_t
        xtx_inv = xtx_inv - np.outer(gain, x_t @ xtx_inv)
        beta = beta + gain * e_t

    return w


def cusum_test(
    y: NDArray[np.floating],
    x_mat: NDArray[np.floating],
    significance: float = 0.05,
) -> TestResult:
    """CUSUM test for parameter stability.

    Parameters
    ----------
    y : array_like, shape (T,)
        Dependent variable.
    x_mat : array_like, shape (T, k)
        Regressor matrix (should include constant if desired).
    significance : float, default 0.05
        Significance level (0.05 or 0.01).

    Returns
    -------
    TestResult
        With additional_info containing:
        - 'cusum_values': cumulative sum of recursive residuals
        - 'upper_band': upper significance band
        - 'lower_band': lower significance band
        - 'max_departure': maximum departure from zero line
    """
    y = np.asarray(y, dtype=float).ravel()
    x_mat = np.asarray(x_mat, dtype=float)
    if x_mat.ndim == 1:
        x_mat = x_mat.reshape(-1, 1)

    # Compute recursive residuals
    w = _recursive_residuals(y, x_mat)
    n = len(w)  # T - k

    # Standardize
    sigma_hat = np.sqrt(float(np.sum(w**2)) / n)
    if sigma_hat < 1e-15:
        sigma_hat = 1.0

    # Cumulative sum
    w_stat = np.cumsum(w) / sigma_hat

    # Significance bands
    # a = 0.948 for 5%, 1.143 for 1%
    a = 1.143 if significance <= 0.01 else 0.948

    t_vals = np.arange(1, n + 1)
    upper = a * np.sqrt(n) + 2 * a * t_vals / np.sqrt(n)
    lower = -upper

    # Check if CUSUM crosses the bands
    crosses = bool(np.any(upper < w_stat) or np.any(lower > w_stat))

    # Maximum departure
    max_dep = float(np.max(np.abs(w_stat)))

    return TestResult(
        test_name="CUSUM",
        statistic=max_dep,
        pvalue=None,
        critical_values={"5%": float(a * np.sqrt(n) + 2 * a * n / np.sqrt(n))},
        null_hypothesis="Parameter stability (coefficients constant over time)",
        alternative_hypothesis="Parameter instability (structural change)",
        reject_at_5pct=crosses,
        additional_info={
            "cusum_values": w_stat,
            "upper_band": upper,
            "lower_band": lower,
            "max_departure": max_dep,
            "sigma_hat": sigma_hat,
            "n_recursive_resid": n,
        },
    )


def cusumsq_test(
    y: NDArray[np.floating],
    x_mat: NDArray[np.floating],
    significance: float = 0.05,
) -> TestResult:
    """CUSUM of squares test for variance stability.

    Parameters
    ----------
    y : array_like, shape (T,)
        Dependent variable.
    x_mat : array_like, shape (T, k)
        Regressor matrix.
    significance : float, default 0.05
        Significance level.

    Returns
    -------
    TestResult
        With additional_info containing:
        - 'cusumsq_values': cumulative sum of squared recursive residuals
        - 'upper_band': upper significance band
        - 'lower_band': lower significance band
    """
    y = np.asarray(y, dtype=float).ravel()
    x_mat = np.asarray(x_mat, dtype=float)
    if x_mat.ndim == 1:
        x_mat = x_mat.reshape(-1, 1)

    # Recursive residuals
    w = _recursive_residuals(y, x_mat)
    n = len(w)

    # CUSUM of squares
    w2 = w**2
    total_w2 = float(np.sum(w2))
    if total_w2 < 1e-15:
        total_w2 = 1.0

    s_vals = np.cumsum(w2) / total_w2

    # Expected value: t/n where t=1..n
    e_s = np.arange(1, n + 1, dtype=float) / n

    # Significance band: c_alpha from KS distribution
    # c_alpha = 0.1478 for 5%, 0.1756 for 1%
    c_alpha = 0.1756 if significance <= 0.01 else 0.1478

    upper = e_s + c_alpha
    lower = np.maximum(e_s - c_alpha, 0.0)

    # Check crossing
    crosses = bool(np.any(upper < s_vals) or np.any(lower > s_vals))

    # Maximum departure from expected
    max_dep = float(np.max(np.abs(s_vals - e_s)))

    return TestResult(
        test_name="CUSUMSQ",
        statistic=max_dep,
        pvalue=None,
        critical_values={"5%": c_alpha, "1%": 0.1756},
        null_hypothesis="Variance stability (homoskedastic residuals)",
        alternative_hypothesis="Variance instability (heteroskedastic)",
        reject_at_5pct=crosses,
        additional_info={
            "cusumsq_values": s_vals,
            "expected_values": e_s,
            "upper_band": upper,
            "lower_band": lower,
            "max_departure": max_dep,
            "n_recursive_resid": n,
        },
    )
