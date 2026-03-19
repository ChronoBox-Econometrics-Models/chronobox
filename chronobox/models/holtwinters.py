"""Holt-Winters Exponential Smoothing.

Classical Holt-Winters method with additive and multiplicative seasonality.
Optimization via L-BFGS-B over smoothing parameters and initial states.

References
----------
- Winters, P.R. (1960). Forecasting sales by exponentially weighted moving averages.
  Management Science, 6(3), 324-342.
- Holt, C.C. (1957). Forecasting seasonals and trends by exponentially weighted averages.
  ONR Memorandum 52.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from chronobox.utils.validation import validate_endog


@dataclass
class HoltWintersResults:
    """Results from fitting a Holt-Winters model.

    Attributes
    ----------
    method : str
        'additive' or 'multiplicative'.
    seasonal_period : int
        Seasonal period m.
    alpha : float
        Level smoothing parameter.
    beta : float
        Trend smoothing parameter.
    gamma : float
        Seasonal smoothing parameter.
    l0 : float
        Initial level.
    b0 : float
        Initial trend.
    s0 : NDArray[np.float64]
        Initial seasonal indices (length m).
    level : NDArray[np.float64]
        Estimated level at each time point.
    trend : NDArray[np.float64]
        Estimated trend at each time point.
    season : NDArray[np.float64]
        Estimated seasonal component at each time point.
    resid : NDArray[np.float64]
        Residuals (y - fitted).
    fittedvalues : NDArray[np.float64]
        Fitted values (one-step-ahead predictions).
    sse : float
        Sum of squared errors.
    mse : float
        Mean squared error.
    rmse : float
        Root mean squared error.
    nobs : int
        Number of observations.
    endog : NDArray[np.float64]
        Original series.
    """

    method: str
    seasonal_period: int
    alpha: float
    beta: float
    gamma: float
    l0: float
    b0: float
    s0: NDArray[np.float64]
    level: NDArray[np.float64]
    trend: NDArray[np.float64]
    season: NDArray[np.float64]
    resid: NDArray[np.float64]
    fittedvalues: NDArray[np.float64]
    sse: float
    mse: float
    rmse: float
    nobs: int
    endog: NDArray[np.float64]

    def summary(self) -> str:
        """Return a formatted summary."""
        lines = [
            "=" * 60,
            f"{'Holt-Winters Results':^60}",
            "=" * 60,
            f"Method:             {self.method}",
            f"Seasonal Period:    {self.seasonal_period}",
            f"No. Observations:   {self.nobs}",
            f"SSE:                {self.sse:.4f}",
            f"MSE:                {self.mse:.4f}",
            f"RMSE:               {self.rmse:.4f}",
            "-" * 60,
            f"{'Smoothing Parameters':^60}",
            "-" * 60,
            f"  alpha = {self.alpha:.6f}",
            f"  beta  = {self.beta:.6f}",
            f"  gamma = {self.gamma:.6f}",
            "-" * 60,
            f"{'Initial States':^60}",
            "-" * 60,
            f"  l0 = {self.l0:.4f}",
            f"  b0 = {self.b0:.4f}",
        ]
        for i, s in enumerate(self.s0):
            lines.append(f"  s[{i}] = {s:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def forecast(self, steps: int = 1) -> NDArray[np.float64]:
        """Forecast future values.

        Parameters
        ----------
        steps : int
            Number of steps ahead.

        Returns
        -------
        NDArray[np.float64]
            Forecasted values.
        """
        t_last = self.nobs
        m = self.seasonal_period
        l_t = self.level[-1]
        b_t = self.trend[-1]
        s = self.season.copy()

        forecasts = np.empty(steps, dtype=np.float64)
        is_mult = self.method == "multiplicative"

        for h in range(1, steps + 1):
            # Get seasonal index: s_{T + h - m * (floor((h-1)/m) + 1)}
            s_idx = t_last + h - m * (((h - 1) // m) + 1)
            if s_idx < 0:
                s_idx = s_idx % m
            s_h = s[s_idx] if s_idx < len(s) else s[s_idx % m]

            if is_mult:
                forecasts[h - 1] = (l_t + h * b_t) * s_h
            else:
                forecasts[h - 1] = l_t + h * b_t + s_h

        return forecasts


def _hw_filter(
    y: NDArray[np.float64],
    method: str,
    m: int,
    alpha: float,
    beta: float,
    gamma: float,
    l0: float,
    b0: float,
    s0: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Run the Holt-Winters filter.

    Returns (level, trend, season, fitted, resid).
    """
    n = len(y)
    is_mult = method == "multiplicative"

    level = np.zeros(n, dtype=np.float64)
    trend_arr = np.zeros(n, dtype=np.float64)
    season = np.zeros(n + m, dtype=np.float64)
    fitted = np.zeros(n, dtype=np.float64)
    resid = np.zeros(n, dtype=np.float64)

    # Initialize seasonal
    season[:m] = s0

    l_prev = l0
    b_prev = b0

    for t in range(n):
        s_tm = season[t]

        # One-step-ahead prediction
        y_hat = (l_prev + b_prev) * s_tm if is_mult else l_prev + b_prev + s_tm

        fitted[t] = y_hat
        resid[t] = y[t] - y_hat

        # Update level
        if is_mult:
            if abs(s_tm) > 1e-300:
                l_t = alpha * (y[t] / s_tm) + (1 - alpha) * (l_prev + b_prev)
            else:
                l_t = (1 - alpha) * (l_prev + b_prev)
        else:
            l_t = alpha * (y[t] - s_tm) + (1 - alpha) * (l_prev + b_prev)

        # Update trend
        b_t = beta * (l_t - l_prev) + (1 - beta) * b_prev

        # Update seasonal
        if is_mult:
            if abs(l_prev + b_prev) > 1e-300:
                s_t = gamma * (y[t] / (l_prev + b_prev)) + (1 - gamma) * s_tm
            else:
                s_t = (1 - gamma) * s_tm
        else:
            s_t = gamma * (y[t] - l_prev - b_prev) + (1 - gamma) * s_tm

        level[t] = l_t
        trend_arr[t] = b_t
        season[t + m] = s_t

        l_prev = l_t
        b_prev = b_t

    return level, trend_arr, season, fitted, resid


class HoltWinters:
    """Holt-Winters Exponential Smoothing.

    Supports additive and multiplicative seasonal methods.

    Parameters
    ----------
    seasonal : str
        'additive' or 'multiplicative'.
    seasonal_period : int
        Seasonal period m (must be >= 2).

    Examples
    --------
    >>> import numpy as np
    >>> from chronobox.models.holtwinters import HoltWinters
    >>> rng = np.random.default_rng(42)
    >>> t = np.arange(48)
    >>> y = 100 + 2*t + 20*np.sin(2*np.pi*t/12) + rng.normal(0, 3, 48)
    >>> model = HoltWinters(seasonal='additive', seasonal_period=12)
    >>> results = model.fit(y)
    >>> fc = results.forecast(steps=12)
    """

    def __init__(
        self,
        seasonal: Literal["additive", "multiplicative"] = "additive",
        seasonal_period: int = 12,
    ) -> None:
        if seasonal not in ("additive", "multiplicative"):
            msg = f"seasonal must be 'additive' or 'multiplicative', got '{seasonal}'"
            raise ValueError(msg)
        if seasonal_period < 2:
            msg = f"seasonal_period must be >= 2, got {seasonal_period}"
            raise ValueError(msg)
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period

    def fit(
        self,
        endog: NDArray[np.float64] | list[float],
        optimized: bool = True,
    ) -> HoltWintersResults:
        """Fit the Holt-Winters model.

        Parameters
        ----------
        endog : array-like
            Time series data.
        optimized : bool
            If True, optimize parameters via L-BFGS-B.

        Returns
        -------
        HoltWintersResults
            Fitted model results.
        """
        y = validate_endog(endog)
        n = len(y)
        m = self.seasonal_period
        is_mult = self.seasonal == "multiplicative"

        if n < 2 * m:
            msg = f"Need at least 2 seasonal cycles ({2 * m} obs), got {n}"
            raise ValueError(msg)

        if is_mult and np.any(y <= 0):
            msg = "Multiplicative seasonality requires all positive data"
            raise ValueError(msg)

        # Initialize
        l0_init, b0_init, s0_init = self._initialize(y, m, is_mult)

        if not optimized:
            alpha, beta, gamma = 0.2, 0.1, 0.1
            return self._build_results(y, alpha, beta, gamma, l0_init, b0_init, s0_init)

        # Pack: [alpha, beta, gamma, l0, b0, s0_0, ..., s0_{m-2}]
        # s0_{m-1} determined by constraint
        n_params = 3 + 2 + (m - 1)
        x0 = np.zeros(n_params)
        x0[0] = 0.2  # alpha
        x0[1] = 0.1  # beta
        x0[2] = 0.1  # gamma
        x0[3] = l0_init
        x0[4] = b0_init
        x0[5:] = s0_init[: m - 1]

        y_range = float(np.max(y) - np.min(y))
        y_min = float(np.min(y))
        y_max = float(np.max(y))

        bounds: list[tuple[float, float]] = [
            (1e-4, 0.9999),  # alpha
            (1e-4, 0.9999),  # beta
            (1e-4, 0.9999),  # gamma
            (y_min - y_range, y_max + y_range),  # l0
            (-y_range, y_range),  # b0
        ]
        for _ in range(m - 1):
            if is_mult:
                bounds.append((0.01, 5.0))
            else:
                bounds.append((-y_range * 2, y_range * 2))

        seasonal_type = self.seasonal

        def sse_objective(params: NDArray[np.float64]) -> float:
            alpha_v = params[0]
            beta_v = params[1]
            gamma_v = params[2]
            l0_v = params[3]
            b0_v = params[4]
            s0_v = np.empty(m, dtype=np.float64)
            s0_v[: m - 1] = params[5:]
            if is_mult:
                s0_v[m - 1] = m - np.sum(s0_v[: m - 1])
                if s0_v[m - 1] <= 0:
                    return 1e20
            else:
                s0_v[m - 1] = -np.sum(s0_v[: m - 1])

            try:
                _, _, _, _, resid_v = _hw_filter(
                    y, seasonal_type, m, alpha_v, beta_v, gamma_v, l0_v, b0_v, s0_v
                )
                return float(np.sum(resid_v**2))
            except Exception:
                return 1e20

        # Try multiple starting points for robustness
        best_sse = sse_objective(x0)
        best_x = x0.copy()

        alpha_starts = [0.2, 0.5, 0.8]
        beta_starts = [0.01, 0.1, 0.3]
        gamma_starts = [0.01, 0.1, 0.5]

        for a0 in alpha_starts:
            for b0_s in beta_starts:
                for g0 in gamma_starts:
                    x_try = x0.copy()
                    x_try[0] = a0
                    x_try[1] = b0_s
                    x_try[2] = g0
                    try:
                        res = optimize.minimize(
                            sse_objective,
                            x_try,
                            method="L-BFGS-B",
                            bounds=bounds,
                            options={"maxiter": 1000, "ftol": 1e-12, "eps": 1e-6},
                        )
                        if res.fun < best_sse:
                            best_sse = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        continue

        alpha = float(best_x[0])
        beta = float(best_x[1])
        gamma = float(best_x[2])
        l0 = float(best_x[3])
        b0 = float(best_x[4])
        s0 = np.empty(m, dtype=np.float64)
        s0[: m - 1] = best_x[5:]
        if is_mult:
            s0[m - 1] = m - np.sum(s0[: m - 1])
        else:
            s0[m - 1] = -np.sum(s0[: m - 1])

        return self._build_results(y, alpha, beta, gamma, l0, b0, s0)

    def _initialize(
        self,
        y: NDArray[np.float64],
        m: int,
        is_mult: bool,
    ) -> tuple[float, float, NDArray[np.float64]]:
        """Heuristic initialization for Holt-Winters."""
        first_cycle = y[:m]
        second_cycle = y[m : 2 * m]

        l0 = float(np.mean(first_cycle))
        b0 = float(np.mean(second_cycle) - np.mean(first_cycle)) / m

        if is_mult:
            s0 = first_cycle / l0 if abs(l0) > 1e-300 else np.ones(m, dtype=np.float64)
            s0 = s0 * (m / np.sum(s0))
        else:
            s0 = first_cycle - l0
            s0 = s0 - np.mean(s0)

        return l0, b0, s0.copy()

    def _build_results(
        self,
        y: NDArray[np.float64],
        alpha: float,
        beta: float,
        gamma: float,
        l0: float,
        b0: float,
        s0: NDArray[np.float64],
    ) -> HoltWintersResults:
        """Build results from fitted parameters."""
        m = self.seasonal_period
        level, trend_arr, season, fitted, resid = _hw_filter(
            y, self.seasonal, m, alpha, beta, gamma, l0, b0, s0
        )

        n = len(y)
        sse = float(np.sum(resid**2))
        mse = sse / n
        rmse = float(np.sqrt(mse))

        return HoltWintersResults(
            method=self.seasonal,
            seasonal_period=m,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            l0=l0,
            b0=b0,
            s0=s0,
            level=level,
            trend=trend_arr,
            season=season,
            resid=resid,
            fittedvalues=fitted,
            sse=sse,
            mse=mse,
            rmse=rmse,
            nobs=n,
            endog=y,
        )
