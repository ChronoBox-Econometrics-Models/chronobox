"""
Hodrick-Prescott (HP) filter for trend-cycle decomposition.

The HP filter decomposes a time series y into trend (tau) and cycle (c):
    y_t = tau_t + c_t

The trend minimizes:
    min sum(y_t - tau_t)^2 + lambda * sum[(tau_{t+1} - tau_t) - (tau_t - tau_{t-1})]^2

Solution: (I + lambda * K'K) * tau = y, where K is the second-difference matrix.

References
----------
Hodrick, R. J., & Prescott, E. C. (1997). Postwar US business cycles:
    An empirical investigation. Journal of Money, Credit and Banking, 1-16.
Ravn, M. O., & Uhlig, H. (2002). On adjusting the Hodrick-Prescott filter
    for the frequency of observations. Review of Economics and Statistics, 84(2), 371-376.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse  # type: ignore[reportMissingTypeStubs]
from scipy.sparse.linalg import spsolve  # type: ignore[reportMissingTypeStubs]

# Lambda defaults by frequency (Ravn & Uhlig, 2002)
LAMBDA_DEFAULTS: dict[str, float] = {
    "annual": 6.25,
    "quarterly": 1600.0,
    "monthly": 129600.0,
}


def hp_filter(
    y: ArrayLike,
    lamb: float | None = None,
    frequency: Literal["annual", "quarterly", "monthly"] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the Hodrick-Prescott filter to decompose y into trend and cycle.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series to filter. Must be 1-D.
    lamb : float or None
        Smoothing parameter lambda. If None, inferred from `frequency`.
        Common values: 6.25 (annual), 1600 (quarterly), 129600 (monthly).
    frequency : {"annual", "quarterly", "monthly"} or None
        Used to set lambda if `lamb` is not given. Default is "quarterly".

    Returns
    -------
    trend : ndarray, shape (T,)
        Estimated trend component.
    cycle : ndarray, shape (T,)
        Estimated cyclical component (y - trend).

    Raises
    ------
    ValueError
        If y is not 1-D or has fewer than 4 observations.
        If both lamb and frequency are None and no default can be chosen.

    Notes
    -----
    Uses sparse matrix algebra for efficiency. The system matrix
    (I + lambda * K'K) is pentadiagonal and positive definite,
    solved via scipy.sparse.linalg.spsolve.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.random.randn(100).cumsum()
    >>> trend, cycle = hp_filter(y, lamb=1600)
    >>> np.allclose(trend + cycle, y)
    True
    """
    y = np.asarray(y, dtype=np.float64).squeeze()
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")

    nobs = len(y)
    if nobs < 4:
        raise ValueError(f"Need at least 4 observations, got {nobs}")

    # Resolve lambda
    if lamb is None:
        if frequency is not None:
            freq_key = frequency.lower()
            if freq_key not in LAMBDA_DEFAULTS:
                raise ValueError(
                    f"Unknown frequency '{frequency}'. "
                    f"Choose from {list(LAMBDA_DEFAULTS.keys())}"
                )
            lamb = LAMBDA_DEFAULTS[freq_key]
        else:
            # Default to quarterly
            lamb = LAMBDA_DEFAULTS["quarterly"]

    if lamb < 0:
        raise ValueError(f"lambda must be non-negative, got {lamb}")

    # Build second-difference matrix K: (T-2) x T, sparse
    # K[i, i] = 1, K[i, i+1] = -2, K[i, i+2] = 1
    e = np.ones(nobs - 2)

    diff_mat = sparse.diags(  # type: ignore[reportUnknownMemberType,reportArgumentType]
        [e, -2 * e, e],
        [0, 1, 2],  # type: ignore[reportArgumentType]
        shape=(nobs - 2, nobs),
        format="csc",
    )

    # System: (I + lambda * K'K) * tau = y
    eye = sparse.eye(nobs, format="csc")  # type: ignore[reportUnknownMemberType]
    system = eye + lamb * (diff_mat.T @ diff_mat)  # type: ignore[reportUnknownMemberType]

    # Solve for trend
    trend: np.ndarray = np.asarray(spsolve(system, y))  # type: ignore[reportUnknownArgumentType]

    # Cycle is residual
    cycle: np.ndarray = y - trend

    return trend, cycle
