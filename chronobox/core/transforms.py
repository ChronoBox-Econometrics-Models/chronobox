"""Time series transformations with inverses."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def difference(y: NDArray[np.float64], d: int = 1) -> NDArray[np.float64]:
    """Apply regular differencing: (1-L)^d * y.

    Parameters
    ----------
    y : ndarray
        Input series.
    d : int
        Order of differencing.

    Returns
    -------
    ndarray
        Differenced series (length T - d).
    """
    result = y.copy()
    for _ in range(d):
        result = np.diff(result)
    return result


def undifference(
    dy: NDArray[np.float64], y0: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Invert regular differencing given initial values.

    Parameters
    ----------
    dy : ndarray
        Differenced series.
    y0 : ndarray
        Initial values (length d for d-th order differencing).
        For d=1: y0 = [y[0]], for d=2: y0 = [y[0], y[1]], etc.

    Returns
    -------
    ndarray
        Reconstructed series.
    """
    result = np.concatenate([y0[:1], dy])
    return np.cumsum(result)


def seasonal_difference(
    y: NDArray[np.float64], s: int
) -> NDArray[np.float64]:
    """Apply seasonal differencing: (1-L^s) * y.

    Parameters
    ----------
    y : ndarray
        Input series.
    s : int
        Seasonal period.

    Returns
    -------
    ndarray
        Seasonally differenced series (length T - s).
    """
    return y[s:] - y[:-s]


def seasonal_undifference(
    dy: NDArray[np.float64], y0s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Invert seasonal differencing.

    Parameters
    ----------
    dy : ndarray
        Seasonally differenced series.
    y0s : ndarray
        First s values of the original series.

    Returns
    -------
    ndarray
        Reconstructed series.
    """
    s = len(y0s)
    n = len(dy) + s
    result = np.empty(n, dtype=np.float64)
    result[:s] = y0s
    for t in range(s, n):
        result[t] = dy[t - s] + result[t - s]
    return result


def log_transform(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Natural log transformation.

    Parameters
    ----------
    y : ndarray
        Input series (must be positive).

    Returns
    -------
    ndarray
        Log-transformed series.
    """
    return np.log(y)


def exp_transform(ly: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exponential transformation (inverse of log).

    Parameters
    ----------
    ly : ndarray
        Log-transformed series.

    Returns
    -------
    ndarray
        Exponentiated series.
    """
    return np.exp(ly)


def boxcox(
    y: NDArray[np.float64], lam: float | None = None
) -> tuple[NDArray[np.float64], float]:
    """Box-Cox transformation.

    If lam is None, optimal lambda is estimated via MLE.

    Parameters
    ----------
    y : ndarray
        Input series (must be positive).
    lam : float or None
        Box-Cox parameter. Estimated if None.

    Returns
    -------
    tuple of (ndarray, float)
        Transformed series and lambda used.
    """
    if lam is None:
        transformed, lam = stats.boxcox(y)
        return transformed, float(lam)
    if lam == 0:
        return np.log(y), 0.0
    return (y**lam - 1.0) / lam, lam


def inv_boxcox(
    z: NDArray[np.float64], lam: float
) -> NDArray[np.float64]:
    """Inverse Box-Cox transformation.

    Parameters
    ----------
    z : ndarray
        Box-Cox transformed series.
    lam : float
        Box-Cox parameter used in forward transform.

    Returns
    -------
    ndarray
        Original-scale series.
    """
    if lam == 0:
        return np.exp(z)
    return (z * lam + 1.0) ** (1.0 / lam)


def standardize(
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float]:
    """Standardize to zero mean and unit variance.

    Parameters
    ----------
    y : ndarray
        Input series.

    Returns
    -------
    tuple of (ndarray, float, float)
        Standardized series, mean, and std.
    """
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=1))
    if std == 0:
        return y - mean, mean, 1.0
    return (y - mean) / std, mean, std


def unstandardize(
    z: NDArray[np.float64], mean: float, std: float
) -> NDArray[np.float64]:
    """Reverse standardization.

    Parameters
    ----------
    z : ndarray
        Standardized series.
    mean : float
        Original mean.
    std : float
        Original standard deviation.

    Returns
    -------
    ndarray
        Original-scale series.
    """
    return z * std + mean


def detrend(
    y: NDArray[np.float64], order: int = 1
) -> NDArray[np.float64]:
    """Remove polynomial trend.

    Parameters
    ----------
    y : ndarray
        Input series.
    order : int
        Polynomial order (1 = linear, 2 = quadratic).

    Returns
    -------
    ndarray
        Detrended series.
    """
    t = np.arange(len(y), dtype=np.float64)
    coeffs = np.polyfit(t, y, order)
    trend = np.polyval(coeffs, t)
    return y - trend
