"""
Christiano-Fitzgerald (CF) band-pass filter.

Full-sample, asymmetric band-pass filter that does not lose observations.
Optimal for I(1) (random walk) series.

References
----------
Christiano, L. J., & Fitzgerald, T. J. (2003). The band pass filter.
    International Economic Review, 44(2), 435-465.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def cf_filter(
    y: ArrayLike,
    low: int = 6,
    high: int = 32,
    drift: bool = False,
) -> np.ndarray:
    """
    Apply the Christiano-Fitzgerald band-pass filter.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series to filter. Must be 1-D.
    low : int
        Minimum period of the cycle. Default is 6 (quarterly).
    high : int
        Maximum period of the cycle. Default is 32 (quarterly).
    drift : bool
        If True, remove linear drift before filtering. Default is False.

    Returns
    -------
    cycle : ndarray, shape (T,)
        Estimated cyclical component. Same length as input (no observation loss).

    Raises
    ------
    ValueError
        If y is not 1-D, low < 2, high <= low, or T < 4.

    Notes
    -----
    The CF filter is asymmetric at the endpoints but symmetric in the interior.
    It is optimal under the assumption that y follows a random walk.

    Unlike the BK filter, no observations are lost. At the endpoints,
    asymmetric weights are used that are optimal under the random walk assumption.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.random.randn(100).cumsum()
    >>> cycle = cf_filter(y, low=6, high=32)
    >>> len(cycle) == len(y)
    True
    """
    ya = np.asarray(y, dtype=np.float64).squeeze()
    if ya.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {ya.shape}")

    n_t = len(ya)
    if n_t < 4:
        raise ValueError(f"Need at least 4 observations, got {n_t}")
    if low < 2:
        raise ValueError(f"low must be >= 2, got {low}")
    if high <= low:
        raise ValueError(f"high must be > low, got low={low}, high={high}")

    # Remove drift if requested
    if drift:
        t_idx = np.arange(n_t, dtype=np.float64)
        slope = (ya[-1] - ya[0]) / (n_t - 1)
        ya = ya - slope * t_idx

    # Frequencies
    omega_low = 2.0 * np.pi / high
    omega_high = 2.0 * np.pi / low

    # Ideal filter coefficients b[j] for j = 0, 1, ..., n_t-1
    b = np.zeros(n_t)
    b[0] = (omega_high - omega_low) / np.pi
    for j in range(1, n_t):
        b[j] = (np.sin(omega_high * j) - np.sin(omega_low * j)) / (np.pi * j)

    # Build the CF filter using asymmetric endpoint adjustment.
    # For the random walk case, the interior weights are the ideal b[k],
    # but the outermost lead/lag weight is adjusted so that all weights
    # (including self = b[0]) sum to zero.

    cycle = np.zeros(n_t)

    for t in range(n_t):
        n_leads = n_t - 1 - t  # observations after t
        n_lags = t  # observations before t

        # Forward (lead) weights: b[1], ..., b[n_leads]
        fwd = np.array([b[j + 1] for j in range(n_leads)])
        # Backward (lag) weights: b[1], ..., b[n_lags]
        bwd = np.array([b[j + 1] for j in range(n_lags)])

        # Adjust the outermost weight on each side so that
        # b[0] + sum(fwd) + sum(bwd) = 0
        # When both sides exist, split the adjustment equally.
        # When only one side exists, put all adjustment there.
        if n_leads > 0 and n_lags > 0:
            fwd[-1] = -b[0] / 2.0 - np.sum(b[1:n_leads])
            bwd[-1] = -b[0] / 2.0 - np.sum(b[1:n_lags])
        elif n_leads > 0:
            fwd[-1] = -b[0] - np.sum(b[1:n_leads])
        elif n_lags > 0:
            bwd[-1] = -b[0] - np.sum(b[1:n_lags])

        # Apply weights
        val = b[0] * ya[t]
        for j in range(n_leads):
            val += fwd[j] * ya[t + j + 1]
        for j in range(n_lags):
            val += bwd[j] * ya[t - j - 1]

        cycle[t] = val

    return cycle
