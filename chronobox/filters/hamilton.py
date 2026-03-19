"""
Hamilton (2018) regression filter for trend-cycle decomposition.

Uses OLS regression of y_{t+h} on p lags of y to separate trend from cycle.
Does not produce spurious cycles on random walks and has no end-of-sample bias.

References
----------
Hamilton, J. D. (2018). Why you should never use the Hodrick-Prescott filter.
    Review of Economics and Statistics, 100(5), 831-843.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class HamiltonFilterResult:
    """Result of Hamilton filter decomposition.

    Attributes
    ----------
    trend : ndarray
        Trend component (fitted values from regression).
        NaN for observations where the filter cannot be applied.
    cycle : ndarray
        Cyclical component (residuals from regression).
        NaN for observations where the filter cannot be applied.
    coefficients : ndarray
        OLS regression coefficients [beta_0, beta_1, ..., beta_p].
    h : int
        Forecast horizon used.
    p : int
        Number of lags used.
    """

    trend: np.ndarray
    cycle: np.ndarray
    coefficients: np.ndarray
    h: int
    p: int


def _hamilton_core(
    ya: np.ndarray,
    h: int,
    p: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core Hamilton filter computation. Returns (trend, cycle, beta)."""
    n_t = len(ya)
    n_reg = n_t - h - (p - 1)  # number of regression observations

    # Dependent variable: y_{t+h}
    y_dep = np.zeros(n_reg)
    # Design matrix: [1, y_t, y_{t-1}, ..., y_{t-p+1}]
    x_mat = np.zeros((n_reg, p + 1))

    for i in range(n_reg):
        t = (p - 1) + i
        y_dep[i] = ya[t + h]
        x_mat[i, 0] = 1.0
        for j in range(p):
            x_mat[i, j + 1] = ya[t - j]

    # OLS estimation: beta = (X'X)^{-1} X'Y
    beta = np.linalg.solve(x_mat.T @ x_mat, x_mat.T @ y_dep)

    # Fitted values and residuals
    y_hat = x_mat @ beta
    residuals = y_dep - y_hat

    # Map back to full-length arrays
    trend = np.full(n_t, np.nan)
    cycle = np.full(n_t, np.nan)

    for i in range(n_reg):
        t = (p - 1) + i
        idx = t + h
        trend[idx] = y_hat[i]
        cycle[idx] = residuals[i]

    return trend, cycle, beta


def hamilton_filter(
    y: ArrayLike,
    h: int = 8,
    p: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the Hamilton (2018) regression filter.

    Decomposes y into trend and cycle using OLS regression:
        y_{t+h} = beta_0 + beta_1 * y_t + ... + beta_p * y_{t-p+1} + v_{t+h}

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series to filter. Must be 1-D.
    h : int
        Forecast horizon. Default is 8 (2 years for quarterly data).
    p : int
        Number of lags in the regression. Default is 4.

    Returns
    -------
    trend : ndarray, shape (T,)
        Trend component. NaN for the first h+p-1 observations.
    cycle : ndarray, shape (T,)
        Cyclical component. NaN for the first h+p-1 observations.

    Raises
    ------
    ValueError
        If y is not 1-D, h < 1, p < 1, or T <= h + p.

    Notes
    -----
    The Hamilton filter regression uses y_{t+h} as the dependent variable
    and [1, y_t, y_{t-1}, ..., y_{t-p+1}] as regressors.

    The trend at time t+h is the fitted value, and the cycle is the residual.
    Re-indexed so that trend[t+h] and cycle[t+h] correspond to time t+h.

    The first h+p-1 observations will be NaN because the filter cannot
    be applied there.

    Advantages over HP filter:
    - Does not produce spurious cycles on random walks
    - No end-of-sample bias
    - Robust to unit roots
    - No arbitrary smoothing parameter

    Examples
    --------
    >>> import numpy as np
    >>> y = np.random.randn(200).cumsum()
    >>> trend, cycle = hamilton_filter(y, h=8, p=4)
    >>> valid = ~np.isnan(cycle)
    >>> np.allclose(trend[valid] + cycle[valid], y[valid])
    True
    """
    ya = np.asarray(y, dtype=np.float64).squeeze()
    if ya.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {ya.shape}")

    n_t = len(ya)
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    if n_t <= h + p:
        raise ValueError(
            f"Series length T={n_t} must be > h+p={h + p}. "
            f"Use a longer series or reduce h/p."
        )

    trend, cycle, _ = _hamilton_core(ya, h, p)
    return trend, cycle


def hamilton_filter_detailed(
    y: ArrayLike,
    h: int = 8,
    p: int = 4,
) -> HamiltonFilterResult:
    """
    Apply Hamilton filter and return detailed results including coefficients.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series to filter.
    h : int
        Forecast horizon. Default is 8.
    p : int
        Number of lags. Default is 4.

    Returns
    -------
    HamiltonFilterResult
        Detailed results including trend, cycle, and regression coefficients.
    """
    ya = np.asarray(y, dtype=np.float64).squeeze()
    if ya.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {ya.shape}")

    n_t = len(ya)
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    if n_t <= h + p:
        raise ValueError(f"Series length T={n_t} must be > h+p={h + p}")

    trend, cycle, beta = _hamilton_core(ya, h, p)

    return HamiltonFilterResult(
        trend=trend,
        cycle=cycle,
        coefficients=beta,
        h=h,
        p=p,
    )
