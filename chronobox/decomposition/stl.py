"""STL - Seasonal and Trend decomposition using LOESS.

Implementation of the STL algorithm by Cleveland et al. (1990).

References
----------
- Cleveland, R.B., Cleveland, W.S., McRae, J.E. & Terpenning, I. (1990).
  STL: A seasonal-trend decomposition procedure based on loess.
  Journal of Official Statistics, 6(1), 3-73.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chronobox.utils.validation import validate_endog


@dataclass
class DecompositionResult:
    """Result of time series decomposition.

    Attributes
    ----------
    observed : NDArray[np.float64]
        Original series.
    trend : NDArray[np.float64]
        Trend component.
    seasonal : NDArray[np.float64]
        Seasonal component.
    remainder : NDArray[np.float64]
        Remainder (residual) component.
    weights : NDArray[np.float64] | None
        Robustness weights (None if not robust).
    period : int
        Seasonal period.
    model : str
        'additive' or 'multiplicative'.
    """

    observed: NDArray[np.float64]
    trend: NDArray[np.float64]
    seasonal: NDArray[np.float64]
    remainder: NDArray[np.float64]
    weights: NDArray[np.float64] | None
    period: int
    model: str

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "=" * 60,
            f"{'Decomposition Results':^60}",
            "=" * 60,
            f"Model:              {self.model}",
            f"Period:             {self.period}",
            f"No. Observations:   {len(self.observed)}",
            "-" * 60,
            f"{'Component':<20} {'Mean':>12} {'Std':>12}",
            "-" * 60,
            f"{'Trend':<20} {np.nanmean(self.trend):>12.4f} {np.nanstd(self.trend):>12.4f}",
            f"{'Seasonal':<20} {np.nanmean(self.seasonal):>12.4f}"
            f" {np.nanstd(self.seasonal):>12.4f}",
            f"{'Remainder':<20} {np.nanmean(self.remainder):>12.4f}"
            f" {np.nanstd(self.remainder):>12.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


def _loess_smooth(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_out: NDArray[np.float64],
    span: int,
    degree: int = 1,
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """LOESS (locally estimated scatterplot smoothing).

    Parameters
    ----------
    x : NDArray
        Predictor values.
    y : NDArray
        Response values.
    x_out : NDArray
        Points at which to evaluate the smooth.
    span : int
        Number of nearest neighbors to use.
    degree : int
        Polynomial degree (0 or 1).
    weights : NDArray or None
        Additional weights for each observation.

    Returns
    -------
    NDArray
        Smoothed values at x_out points.
    """
    n = len(x)
    n_out = len(x_out)
    result = np.zeros(n_out, dtype=np.float64)
    q = min(span, n)

    for i in range(n_out):
        x0 = x_out[i]

        # Find q nearest neighbors
        distances = np.abs(x - x0)
        sorted_idx = np.argsort(distances)
        nn_idx = sorted_idx[:q]
        max_dist = distances[nn_idx[-1]]

        if max_dist == 0:
            max_dist = 1.0

        # Tricube kernel weights
        u = distances[nn_idx] / (max_dist * 1.0001)  # avoid exact 1
        w = (1.0 - u**3) ** 3
        w = np.maximum(w, 0.0)

        if weights is not None:
            w = w * weights[nn_idx]

        # Weighted regression
        x_nn = x[nn_idx]
        y_nn = y[nn_idx]

        if degree == 0:
            # Weighted mean
            w_sum = np.sum(w)
            if w_sum > 0:
                result[i] = np.sum(w * y_nn) / w_sum
            else:
                result[i] = np.mean(y_nn)
        else:
            # Weighted linear regression
            xc = x_nn - x0
            w_mat = np.diag(w)
            x_mat = np.column_stack([np.ones(q), xc])

            try:
                xtwx = x_mat.T @ w_mat @ x_mat
                xtwy = x_mat.T @ w_mat @ y_nn
                beta = np.linalg.solve(xtwx, xtwy)
                result[i] = beta[0]
            except np.linalg.LinAlgError:
                result[i] = np.sum(w * y_nn) / (np.sum(w) + 1e-300)

    return result


def _moving_average(y: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Compute centered moving average."""
    n = len(y)
    result = np.full(n, np.nan, dtype=np.float64)
    half = window // 2

    for i in range(half, n - half):
        if window % 2 == 0:
            # 2xm MA for even window
            result[i] = (
                0.5 * y[i - half]
                + np.sum(y[i - half + 1 : i + half])
                + 0.5 * y[i + half]
            ) / window
        else:
            result[i] = np.mean(y[i - half : i + half + 1])

    return result


def _make_odd(n: int) -> int:
    """Ensure n is odd (round up if even)."""
    return n if n % 2 == 1 else n + 1


class STL:
    """STL - Seasonal and Trend decomposition using LOESS.

    Parameters
    ----------
    period : int
        Seasonal period (e.g., 12 for monthly data).
    seasonal : int
        LOESS window for seasonal extraction. Must be odd >= 7. Default 7.
    trend : int or None
        LOESS window for trend extraction. Default computed from period and seasonal.
    low_pass : int or None
        LOESS window for low-pass filter. Default: period (made odd).
    robust : bool
        Use robustness iterations. Default False.
    seasonal_deg : int
        Degree of LOESS for seasonal. Default 1.
    trend_deg : int
        Degree of LOESS for trend. Default 1.
    low_pass_deg : int
        Degree of LOESS for low-pass. Default 1.
    n_inner : int
        Number of inner loop iterations. Default 2.
    n_outer : int
        Number of outer robustness iterations (0 if not robust). Default 0.

    Examples
    --------
    >>> import numpy as np
    >>> from chronobox.decomposition.stl import STL
    >>> rng = np.random.default_rng(42)
    >>> t = np.arange(144)
    >>> y = 100 + 0.5*t + 20*np.sin(2*np.pi*t/12) + rng.normal(0, 5, 144)
    >>> stl = STL(period=12)
    >>> result = stl.fit(y)
    >>> print(result.trend[:5])
    """

    def __init__(
        self,
        period: int,
        seasonal: int = 7,
        trend: int | None = None,
        low_pass: int | None = None,
        robust: bool = False,
        seasonal_deg: int = 1,
        trend_deg: int = 1,
        low_pass_deg: int = 1,
        n_inner: int = 2,
        n_outer: int = 0,
    ) -> None:
        if period < 2:
            msg = f"period must be >= 2, got {period}"
            raise ValueError(msg)

        self.period = period
        self.seasonal = _make_odd(max(seasonal, 7))
        self.robust = robust
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.n_inner = n_inner

        if robust and n_outer == 0:
            self.n_outer = 15
        else:
            self.n_outer = n_outer

        # Default trend window
        if trend is None:
            raw = math.ceil(1.5 * period / (1.0 - 1.5 / self.seasonal))
            self.trend_window = _make_odd(raw)
        else:
            self.trend_window = _make_odd(trend)

        # Default low-pass window
        if low_pass is None:
            self.low_pass_window = _make_odd(period)
        else:
            self.low_pass_window = _make_odd(low_pass)

    def fit(self, endog: NDArray[np.float64] | list[float]) -> DecompositionResult:
        """Decompose a time series using STL.

        Parameters
        ----------
        endog : array-like
            Time series data.

        Returns
        -------
        DecompositionResult
            Decomposition with trend, seasonal, and remainder components.
        """
        y = validate_endog(endog)
        n = len(y)
        m = self.period

        if n < 2 * m:
            msg = f"Need at least 2 seasonal periods ({2 * m} obs), got {n}"
            raise ValueError(msg)

        # Initialize
        trend = np.zeros(n, dtype=np.float64)
        seasonal = np.zeros(n, dtype=np.float64)
        remainder = np.zeros(n, dtype=np.float64)
        rweights = np.ones(n, dtype=np.float64)

        x_all = np.arange(n, dtype=np.float64)

        n_total_outer = max(1, self.n_outer) if self.robust else 1

        for outer_iter in range(n_total_outer):
            for _inner_iter in range(self.n_inner):
                # Step 1: Detrending
                detrended = y - trend

                # Step 2: Cycle-subseries smoothing
                cycle_smooth = np.zeros(n, dtype=np.float64)
                for s in range(m):
                    # Extract subseries at seasonal position s
                    indices = np.arange(s, n, m)
                    sub_y = detrended[indices]
                    sub_x = indices.astype(np.float64)

                    if len(sub_y) < 3:
                        cycle_smooth[indices] = sub_y
                        continue

                    # LOESS smooth of subseries
                    span = min(self.seasonal, len(sub_y))
                    smoothed = _loess_smooth(
                        sub_x,
                        sub_y,
                        sub_x,
                        span=span,
                        degree=self.seasonal_deg,
                        weights=rweights[indices] if self.robust else None,
                    )
                    cycle_smooth[indices] = smoothed

                # Step 3: Low-pass filter
                # Apply MA(m) + MA(m) + MA(3) + LOESS
                lp = _moving_average(cycle_smooth, m)
                lp2 = _moving_average(lp, m)
                lp3 = _moving_average(lp2, 3)

                # Fill NaN edges with nearest valid value
                valid = ~np.isnan(lp3)
                if np.any(valid):
                    first_valid = int(np.argmax(valid))
                    last_valid = n - 1 - int(np.argmax(valid[::-1]))
                    lp3[:first_valid] = lp3[first_valid]
                    lp3[last_valid + 1 :] = lp3[last_valid]

                    # LOESS smooth of low-pass result
                    valid_mask = ~np.isnan(lp3)
                    x_valid = x_all[valid_mask]
                    lp3_valid = lp3[valid_mask]
                    lp_smooth = _loess_smooth(
                        x_valid,
                        lp3_valid,
                        x_all,
                        span=min(self.low_pass_window, len(x_valid)),
                        degree=self.low_pass_deg,
                    )
                else:
                    lp_smooth = np.zeros(n)

                # Step 4: Deseasonalize
                seasonal = cycle_smooth - lp_smooth

                # Step 5: Trend smoothing
                deseasoned = y - seasonal
                trend = _loess_smooth(
                    x_all,
                    deseasoned,
                    x_all,
                    span=min(self.trend_window, n),
                    degree=self.trend_deg,
                    weights=rweights if self.robust else None,
                )

            # Step 6: Compute remainder and robustness weights
            remainder = y - trend - seasonal

            if self.robust and outer_iter < n_total_outer - 1:
                # Bisquare robustness weights
                abs_resid = np.abs(remainder)
                h = 6.0 * np.median(abs_resid)
                if h > 0:
                    u = abs_resid / h
                    rweights = np.where(u < 1.0, (1.0 - u**2) ** 2, 0.0)
                else:
                    rweights = np.ones(n, dtype=np.float64)

        return DecompositionResult(
            observed=y.copy(),
            trend=trend,
            seasonal=seasonal,
            remainder=remainder,
            weights=rweights if self.robust else None,
            period=m,
            model="additive",
        )
