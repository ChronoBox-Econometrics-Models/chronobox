"""Theta Method for time series forecasting.

Implements the Theta method of Assimakopoulos & Nikolopoulos (2000)
and its SES-with-drift equivalence (Hyndman & Billah 2003).

References
----------
- Assimakopoulos, V. & Nikolopoulos, K. (2000). The theta model: a decomposition
  approach to forecasting. International Journal of Forecasting, 16(4), 521-530.
- Hyndman, R.J. & Billah, B. (2003). Unmasking the Theta method.
  International Journal of Forecasting, 19(2), 287-290.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt

from chronobox.utils.validation import validate_endog


@dataclass
class ThetaResults:
    """Results from fitting the Theta method.

    Attributes
    ----------
    ses_level : float
        Final SES level (l_T).
    drift : float
        Drift term (half the OLS slope).
    alpha : float
        SES smoothing parameter.
    slope : float
        OLS trend slope.
    intercept : float
        OLS trend intercept.
    fittedvalues : NDArray[np.float64]
        One-step-ahead fitted values.
    resid : NDArray[np.float64]
        Residuals.
    nobs : int
        Number of observations.
    endog : NDArray[np.float64]
        Original series.
    """

    ses_level: float
    drift: float
    alpha: float
    slope: float
    intercept: float
    fittedvalues: NDArray[np.float64]
    resid: NDArray[np.float64]
    nobs: int
    endog: NDArray[np.float64]

    def summary(self) -> str:
        """Return a formatted summary."""
        lines = [
            "=" * 60,
            f"{'Theta Method Results':^60}",
            "=" * 60,
            f"No. Observations:   {self.nobs}",
            f"SES alpha:          {self.alpha:.6f}",
            f"SES level (l_T):    {self.ses_level:.4f}",
            f"OLS slope:          {self.slope:.6f}",
            f"OLS intercept:      {self.intercept:.4f}",
            f"Drift (slope/2):    {self.drift:.6f}",
            f"RMSE:               {np.sqrt(np.mean(self.resid**2)):.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def forecast(self, steps: int = 1) -> NDArray[np.float64]:
        """Forecast future values using SES with drift.

        Parameters
        ----------
        steps : int
            Number of steps ahead.

        Returns
        -------
        NDArray[np.float64]
            Forecasted values.
        """
        h = np.arange(1, steps + 1, dtype=np.float64)
        return self.ses_level + h * self.drift


def _ses_fit(
    y: NDArray[np.float64],
    alpha: float | None = None,
) -> tuple[float, float, NDArray[np.float64]]:
    """Fit Simple Exponential Smoothing.

    If alpha is None, optimize by minimizing SSE.

    Returns (alpha, l_T, fitted).
    """
    n = len(y)

    def ses_sse(alpha_val: float) -> float:
        level = float(y[0])
        sse = 0.0
        for t in range(n):
            sse += (y[t] - level) ** 2
            level = alpha_val * y[t] + (1 - alpha_val) * level
        return sse

    if alpha is None:
        result = opt.minimize_scalar(ses_sse, bounds=(0.001, 0.999), method="bounded")
        alpha = float(result.x)

    # Run SES with chosen alpha
    fitted = np.zeros(n, dtype=np.float64)
    level = float(y[0])
    for t in range(n):
        fitted[t] = level
        level = alpha * y[t] + (1 - alpha) * level
    l_t = level

    return alpha, l_t, fitted


class ThetaMethod:
    """Theta Method for time series forecasting.

    Implements the standard Theta method which decomposes the series into
    two theta-lines (theta=0 and theta=2) and combines their forecasts.

    Equivalent to SES with drift (Hyndman & Billah 2003):
        forecast_h = l_T + h * (slope / 2)

    Parameters
    ----------
    theta : float
        Theta parameter (default 2.0 as in the original method).

    Examples
    --------
    >>> import numpy as np
    >>> from chronobox.models.theta import ThetaMethod
    >>> rng = np.random.default_rng(42)
    >>> y = 100 + 0.5 * np.arange(60) + rng.normal(0, 3, 60)
    >>> model = ThetaMethod()
    >>> results = model.fit(y)
    >>> fc = results.forecast(steps=12)
    """

    def __init__(self, theta: float = 2.0) -> None:
        if theta < 0:
            msg = f"theta must be non-negative, got {theta}"
            raise ValueError(msg)
        self.theta = theta

    def fit(
        self,
        endog: NDArray[np.float64] | list[float],
        alpha: float | None = None,
    ) -> ThetaResults:
        """Fit the Theta method.

        Parameters
        ----------
        endog : array-like
            Time series data.
        alpha : float or None
            SES smoothing parameter. If None, optimized via SSE.

        Returns
        -------
        ThetaResults
            Fitted results.
        """
        y = validate_endog(endog)
        n = len(y)

        # Step 1: Fit linear trend via OLS
        t_idx = np.arange(1, n + 1, dtype=np.float64)
        t_mean = np.mean(t_idx)
        y_mean = np.mean(y)
        slope = float(
            np.sum((t_idx - t_mean) * (y - y_mean)) / np.sum((t_idx - t_mean) ** 2)
        )
        intercept = float(y_mean - slope * t_mean)
        trend_line = intercept + slope * t_idx

        # Step 2: Theta-line for theta=2
        # Z_t(theta) = theta * y_t - (theta - 1) * L_t
        z_theta2 = self.theta * y - (self.theta - 1) * trend_line

        # Step 3: Fit SES to z_theta2
        alpha_opt, _l_t_ses, fitted_ses = _ses_fit(z_theta2, alpha)

        # Step 4: Drift = slope / 2
        drift = slope / 2.0

        # Step 5: Fitted values (in-sample)
        # Combine: 0.5 * linear_fit + 0.5 * SES_fit
        fittedvalues = 0.5 * trend_line + 0.5 * fitted_ses
        resid = y - fittedvalues

        # The SES level for forecasting (applied to original scale)
        # Using the SES+drift equivalence:
        # Refit SES on original series
        _alpha_orig, l_t_orig, _ = _ses_fit(y, alpha_opt)
        ses_level = l_t_orig

        return ThetaResults(
            ses_level=ses_level,
            drift=drift,
            alpha=alpha_opt,
            slope=slope,
            intercept=intercept,
            fittedvalues=fittedvalues,
            resid=resid,
            nobs=n,
            endog=y,
        )
