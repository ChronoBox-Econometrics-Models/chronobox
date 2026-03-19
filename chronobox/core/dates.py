"""Date and frequency utilities for time series."""

from __future__ import annotations

import pandas as pd


def detect_frequency(index: pd.DatetimeIndex) -> str | None:
    """Detect frequency of a DatetimeIndex.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Temporal index.

    Returns
    -------
    str or None
        Frequency string, or None if cannot be determined.
    """
    if not isinstance(index, pd.DatetimeIndex):
        return None
    return pd.infer_freq(index)


def infer_periods_per_year(freq: str | None) -> int:
    """Map frequency string to periods per year.

    Parameters
    ----------
    freq : str or None
        Frequency string.

    Returns
    -------
    int
        Periods per year. Returns 1 if frequency is unknown.
    """
    if freq is None:
        return 1
    freq_upper = freq.upper()
    mapping: dict[str, int] = {
        "D": 365,
        "B": 252,
        "W": 52,
        "MS": 12,
        "M": 12,
        "ME": 12,
        "QS": 4,
        "Q": 4,
        "QE": 4,
        "AS": 1,
        "A": 1,
        "YS": 1,
        "Y": 1,
        "YE": 1,
    }
    # Try exact match
    if freq_upper in mapping:
        return mapping[freq_upper]
    # Try prefix match (e.g., "MS" from "MS-JAN")
    for key, val in mapping.items():
        if freq_upper.startswith(key):
            return val
    return 1


def create_forecast_index(
    last_date: pd.Timestamp, steps: int, freq: str | None
) -> pd.DatetimeIndex:
    """Create a DatetimeIndex for forecast periods.

    Parameters
    ----------
    last_date : pd.Timestamp
        Last date in the observed series.
    steps : int
        Number of forecast steps.
    freq : str or None
        Frequency string. If None, uses daily.

    Returns
    -------
    pd.DatetimeIndex
        Index for forecast periods.
    """
    if freq is None:
        freq = "D"
    return pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]


def validate_index(index: pd.DatetimeIndex) -> bool:
    """Check if DatetimeIndex has regular frequency.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Temporal index.

    Returns
    -------
    bool
        True if frequency is regular (can be inferred).
    """
    if not isinstance(index, pd.DatetimeIndex):
        return False
    return pd.infer_freq(index) is not None
