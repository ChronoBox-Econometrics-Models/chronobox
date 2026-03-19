"""Input validation utilities for chronobox."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_endog(y: NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
    """Validate and convert endogenous variable to float64 array.

    Parameters
    ----------
    y : array-like
        Input series.

    Returns
    -------
    NDArray[np.float64]
        Validated 1-D array.

    Raises
    ------
    ValueError
        If input is empty, not 1-D, or contains only NaN.
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        msg = f"endog must be 1-D, got shape {arr.shape}"
        raise ValueError(msg)
    if arr.size == 0:
        msg = "endog must not be empty"
        raise ValueError(msg)
    if np.all(np.isnan(arr)):
        msg = "endog contains only NaN values"
        raise ValueError(msg)
    return arr


def validate_order(order: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate ARIMA order (p, d, q).

    Parameters
    ----------
    order : tuple of (int, int, int)
        (p, d, q) order.

    Returns
    -------
    tuple of (int, int, int)
        Validated order.

    Raises
    ------
    ValueError
        If any component is negative or order has wrong length.
    """
    if len(order) != 3:
        msg = f"order must have 3 elements (p, d, q), got {len(order)}"
        raise ValueError(msg)
    p, d, q = order
    if p < 0 or d < 0 or q < 0:
        msg = f"order components must be non-negative, got ({p}, {d}, {q})"
        raise ValueError(msg)
    return (p, d, q)


def validate_seasonal_order(
    seasonal_order: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Validate seasonal order (P, D, Q, s).

    Parameters
    ----------
    seasonal_order : tuple of (int, int, int, int)
        (P, D, Q, s) seasonal order.

    Returns
    -------
    tuple of (int, int, int, int)
        Validated seasonal order.

    Raises
    ------
    ValueError
        If any component is negative or s < 1 when seasonal terms exist.
    """
    if len(seasonal_order) != 4:
        msg = f"seasonal_order must have 4 elements (P, D, Q, s), got {len(seasonal_order)}"
        raise ValueError(msg)
    big_p, big_d, big_q, s = seasonal_order
    if big_p < 0 or big_d < 0 or big_q < 0:
        msg = (
            f"seasonal_order components must be non-negative, "
            f"got ({big_p}, {big_d}, {big_q}, {s})"
        )
        raise ValueError(msg)
    if (big_p > 0 or big_d > 0 or big_q > 0) and s < 2:
        msg = f"seasonal period s must be >= 2 when seasonal terms are used, got s={s}"
        raise ValueError(msg)
    return (big_p, big_d, big_q, s)
