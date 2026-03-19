"""
Baxter-King (BK) band-pass filter for isolating cyclical components.

The BK filter is a symmetric, finite-order approximation to the ideal
band-pass filter. It isolates frequency components with periods between
`low` and `high`.

References
----------
Baxter, M., & King, R. G. (1999). Measuring business cycles: Approximate
    band-pass filters for economic time series. Review of Economics and
    Statistics, 81(4), 575-593.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def bk_filter(
    y: ArrayLike,
    low: int = 6,
    high: int = 32,
    trunc: int = 12,
) -> np.ndarray:
    """
    Apply the Baxter-King band-pass filter.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series to filter. Must be 1-D.
    low : int
        Minimum period of the cycle (in number of observations).
        Default is 6 (quarterly data, cycles >= 1.5 years).
    high : int
        Maximum period of the cycle (in number of observations).
        Default is 32 (quarterly data, cycles <= 8 years).
    trunc : int
        Truncation parameter K (number of leads/lags).
        Default is 12 for quarterly data.

    Returns
    -------
    cycle : ndarray, shape (T - 2*trunc,)
        Estimated cyclical component. Length is T - 2*trunc because
        trunc observations are lost at each end.

    Raises
    ------
    ValueError
        If y is not 1-D, or T <= 2*trunc, or low >= high, or low < 2.

    Notes
    -----
    The filter weights are adjusted to sum to zero, which removes
    the trend component (zero-frequency) from the filtered series.

    The ideal band-pass filter coefficients are:
        b_k = [sin(omega_H * k) - sin(omega_L * k)] / (pi * k), k != 0
        b_0 = (omega_H - omega_L) / pi

    where omega_L = 2*pi/high, omega_H = 2*pi/low.

    The adjusted weights are:
        a_k = b_k - (1/(2K+1)) * sum_{j=-K}^{K} b_j

    Examples
    --------
    >>> import numpy as np
    >>> y = np.random.randn(200).cumsum()
    >>> cycle = bk_filter(y, low=6, high=32, trunc=12)
    >>> len(cycle) == len(y) - 2 * 12
    True
    """
    y = np.asarray(y, dtype=np.float64).squeeze()
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")

    nobs = len(y)

    if low < 2:
        raise ValueError(f"low must be >= 2, got {low}")
    if high <= low:
        raise ValueError(f"high must be > low, got low={low}, high={high}")
    if trunc < 1:
        raise ValueError(f"K must be >= 1, got {trunc}")
    if nobs <= 2 * trunc:
        raise ValueError(
            f"Series length T={nobs} must be > 2*K={2 * trunc}. "
            f"Reduce K or use a longer series."
        )

    # Frequencies
    omega_low = 2.0 * np.pi / high  # lower cutoff frequency
    omega_high = 2.0 * np.pi / low  # upper cutoff frequency

    # Ideal band-pass filter coefficients
    # b[0] corresponds to lag 0, b[k] to lag k (symmetric, so b[-k] = b[k])
    b = np.zeros(trunc + 1)
    b[0] = (omega_high - omega_low) / np.pi

    for k in range(1, trunc + 1):
        b[k] = (np.sin(omega_high * k) - np.sin(omega_low * k)) / (np.pi * k)

    # Full symmetric weights: b[-K], ..., b[-1], b[0], b[1], ..., b[K]
    # Sum of all weights (ideally should be 0 for band-pass, but truncation breaks this)
    weight_sum = b[0] + 2.0 * np.sum(b[1:])

    # Adjusted weights: subtract mean to force sum = 0
    adjustment = weight_sum / (2 * trunc + 1)
    a = b - adjustment

    # Apply filter: y_t^cycle = sum_{k=-K}^{K} a_k * y_{t-k}
    # Valid range: t = K, K+1, ..., T-K-1
    n_valid = nobs - 2 * trunc
    cycle = np.zeros(n_valid)

    for t in range(trunc, nobs - trunc):
        val = a[0] * y[t]
        for k in range(1, trunc + 1):
            val += a[k] * (y[t - k] + y[t + k])
        cycle[t - trunc] = val

    return cycle


def bk_filter_weights(
    low: int = 6,
    high: int = 32,
    trunc: int = 12,
) -> np.ndarray:
    """
    Compute the BK filter weights (adjusted, summing to zero).

    Parameters
    ----------
    low : int
        Minimum period of the cycle.
    high : int
        Maximum period of the cycle.
    trunc : int
        Truncation parameter K.

    Returns
    -------
    weights : ndarray, shape (2*trunc + 1,)
        Filter weights from lag -K to +K.
        weights[0] = a_{-K}, ..., weights[K] = a_0, ..., weights[2K] = a_K.
    """
    omega_low = 2.0 * np.pi / high
    omega_high = 2.0 * np.pi / low

    b = np.zeros(trunc + 1)
    b[0] = (omega_high - omega_low) / np.pi
    for k in range(1, trunc + 1):
        b[k] = (np.sin(omega_high * k) - np.sin(omega_low * k)) / (np.pi * k)

    weight_sum = b[0] + 2.0 * np.sum(b[1:])
    adjustment = weight_sum / (2 * trunc + 1)
    a = b - adjustment

    # Full symmetric weights
    weights = np.concatenate([a[trunc:0:-1], a])
    return weights
