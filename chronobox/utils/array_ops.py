"""Array operations for time series manipulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lag_matrix(y: NDArray[np.float64], lags: int) -> NDArray[np.float64]:
    """Create a matrix of lagged values.

    Parameters
    ----------
    y : ndarray of shape (T,)
        Input series.
    lags : int
        Number of lags.

    Returns
    -------
    ndarray of shape (T - lags, lags)
        Matrix where column j is y lagged by j+1.
    """
    n = len(y)
    result = np.empty((n - lags, lags), dtype=np.float64)
    for j in range(lags):
        result[:, j] = y[lags - j - 1 : n - j - 1]
    return result


def convolve_polynomials(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convolve (multiply) two lag polynomials.

    Parameters
    ----------
    a : ndarray
        Coefficients of first polynomial.
    b : ndarray
        Coefficients of second polynomial.

    Returns
    -------
    ndarray
        Coefficients of the product polynomial.
    """
    return np.convolve(a, b)


def expand_seasonal_polynomial(
    coeffs: NDArray[np.float64], s: int, max_lag: int
) -> NDArray[np.float64]:
    """Expand seasonal polynomial to full lag representation.

    E.g., Phi(L^s) = 1 - Phi_1*L^s with s=12 becomes
    [1, 0, 0, ..., 0, -Phi_1, 0, ...] with non-zero at positions 0 and s.

    Parameters
    ----------
    coeffs : ndarray
        Seasonal polynomial coefficients (including leading 1).
    s : int
        Seasonal period.
    max_lag : int
        Length of the expanded polynomial.

    Returns
    -------
    ndarray
        Expanded polynomial of length max_lag.
    """
    result = np.zeros(max_lag, dtype=np.float64)
    for i, c in enumerate(coeffs):
        pos = i * s
        if pos < max_lag:
            result[pos] = c
    return result
