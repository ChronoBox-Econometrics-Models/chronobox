"""Classical time series decomposition (additive and multiplicative).

Decomposes a time series into trend, seasonal, and remainder components
using centered moving averages.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from chronobox.decomposition.stl import DecompositionResult
from chronobox.utils.validation import validate_endog


class ClassicalDecomposition:
    """Classical time series decomposition.

    Supports additive and multiplicative decomposition using centered
    moving averages for trend estimation.

    Parameters
    ----------
    period : int
        Seasonal period.
    model : str
        'additive' or 'multiplicative'.

    Examples
    --------
    >>> import numpy as np
    >>> from chronobox.decomposition.classical import ClassicalDecomposition
    >>> t = np.arange(48)
    >>> y = 100 + 2*t + 20*np.sin(2*np.pi*t/12) + np.random.normal(0, 2, 48)
    >>> cd = ClassicalDecomposition(period=12, model='additive')
    >>> result = cd.fit(y)
    """

    def __init__(
        self,
        period: int,
        model: Literal["additive", "multiplicative"] = "additive",
    ) -> None:
        if period < 2:
            msg = f"period must be >= 2, got {period}"
            raise ValueError(msg)
        if model not in ("additive", "multiplicative"):
            msg = f"model must be 'additive' or 'multiplicative', got '{model}'"
            raise ValueError(msg)
        self.period = period
        self.model = model

    def fit(self, endog: NDArray[np.float64] | list[float]) -> DecompositionResult:
        """Decompose the time series.

        Parameters
        ----------
        endog : array-like
            Time series data.

        Returns
        -------
        DecompositionResult
            Decomposition with trend, seasonal, and remainder.
        """
        y = validate_endog(endog)
        n = len(y)
        m = self.period
        is_mult = self.model == "multiplicative"

        if n < 2 * m:
            msg = f"Need at least 2 seasonal periods ({2 * m} obs), got {n}"
            raise ValueError(msg)

        if is_mult and np.any(y <= 0):
            msg = "Multiplicative decomposition requires all positive data"
            raise ValueError(msg)

        # Step 1: Compute centered moving average (trend)
        trend = self._centered_ma(y, m)

        # Step 2: Detrend
        if is_mult:
            detrended = np.where(
                np.isnan(trend) | (np.abs(trend) < 1e-300),
                np.nan,
                y / trend,
            )
        else:
            detrended = np.where(np.isnan(trend), np.nan, y - trend)

        # Step 3: Compute seasonal component
        seasonal_means = np.zeros(m, dtype=np.float64)
        for s in range(m):
            vals = detrended[s::m]
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                seasonal_means[s] = float(np.mean(valid))
            else:
                seasonal_means[s] = 0.0 if not is_mult else 1.0

        # Normalize
        if is_mult:
            seasonal_means = seasonal_means * m / np.sum(seasonal_means)
        else:
            seasonal_means = seasonal_means - np.mean(seasonal_means)

        # Tile seasonal to full length
        seasonal = np.tile(seasonal_means, (n // m) + 1)[:n]

        # Step 4: Compute remainder
        if is_mult:
            remainder = np.where(
                np.isnan(trend) | (np.abs(trend * seasonal) < 1e-300),
                np.nan,
                y / (trend * seasonal),
            )
        else:
            remainder = np.where(np.isnan(trend), np.nan, y - trend - seasonal)

        return DecompositionResult(
            observed=y.copy(),
            trend=trend,
            seasonal=seasonal,
            remainder=remainder,
            weights=None,
            period=m,
            model=self.model,
        )

    def _centered_ma(self, y: NDArray[np.float64], m: int) -> NDArray[np.float64]:
        """Compute centered moving average.

        For even m: 2xm MA (symmetric filter with half-weights at ends).
        For odd m: simple centered MA.
        """
        n = len(y)
        trend = np.full(n, np.nan, dtype=np.float64)

        if m % 2 == 0:
            # 2xm MA
            half = m // 2
            for i in range(half, n - half):
                val = (
                    0.5 * y[i - half]
                    + np.sum(y[i - half + 1 : i + half])
                    + 0.5 * y[i + half]
                )
                trend[i] = val / m
        else:
            half = m // 2
            for i in range(half, n - half):
                trend[i] = np.mean(y[i - half : i + half + 1])

        return trend
