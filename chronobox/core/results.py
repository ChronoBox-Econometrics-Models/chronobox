"""TimeSeriesResults - Container for model estimation results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


class TimeSeriesResults:
    """Container for time series model estimation results.

    Parameters
    ----------
    params : ndarray
        Estimated parameters.
    param_names : list of str
        Names of the parameters.
    se : ndarray
        Standard errors of the parameters.
    loglike : float
        Maximized log-likelihood.
    nobs : int
        Total number of observations.
    nobs_effective : int
        Effective observations used in estimation (post-differencing).
    residuals : ndarray
        Model residuals.
    fitted_values : ndarray
        Fitted (in-sample) values.
    model_name : str
        Name of the model (e.g., "ARIMA(1,1,1)").
    forecast_func : callable or None
        Function to compute forecasts. Signature: (steps, alpha) -> dict.
    model : object or None
        Reference to the fitted model object.
    """

    def __init__(
        self,
        params: NDArray[np.float64],
        param_names: list[str],
        se: NDArray[np.float64],
        loglike: float,
        nobs: int,
        nobs_effective: int,
        residuals: NDArray[np.float64],
        fitted_values: NDArray[np.float64],
        model_name: str,
        forecast_func: Any = None,
        model: Any = None,
    ) -> None:
        self.params = params
        self.param_names = param_names
        self.se = se
        self.loglike = loglike
        self.nobs = nobs
        self.nobs_effective = nobs_effective
        self.residuals = residuals
        self.fitted_values = fitted_values
        self.model_name = model_name
        self._forecast_func = forecast_func
        self.model = model

        # Derived quantities
        k = len(params)
        n = nobs_effective

        with np.errstate(divide="ignore", invalid="ignore"):
            self.tvalues = np.where(se > 0, params / se, np.zeros_like(params))
        self.pvalues = np.where(
            se > 0,
            2.0 * (1.0 - stats.t.cdf(np.abs(self.tvalues), df=max(n - k, 1))),
            np.ones_like(params),
        )

        self.aic = -2.0 * loglike + 2.0 * k
        self.bic = -2.0 * loglike + k * np.log(n)
        if n > 0 and np.log(n) > 0:
            self.hqic = -2.0 * loglike + 2.0 * k * np.log(np.log(n))
        else:
            self.hqic = np.nan
        if n - k - 1 > 0:
            self.aicc = self.aic + 2.0 * k * (k + 1) / (n - k - 1)
        else:
            self.aicc = np.inf

    def summary(self) -> str:
        """Generate formatted summary table.

        Returns
        -------
        str
            Formatted summary with parameters, SE, t-values, p-values, and ICs.
        """
        width = 70
        lines: list[str] = []
        lines.append("=" * width)
        lines.append(f"{'Model: ' + self.model_name:^{width}}")
        lines.append("=" * width)
        lines.append(
            f"  Nobs: {self.nobs}    Effective Nobs: {self.nobs_effective}"
        )
        lines.append(f"  Log-Likelihood: {self.loglike:.4f}")
        lines.append(
            f"  AIC: {self.aic:.4f}    BIC: {self.bic:.4f}    "
            f"AICc: {self.aicc:.4f}    HQIC: {self.hqic:.4f}"
        )
        lines.append("-" * width)
        lines.append(
            f"  {'Parameter':<15} {'Estimate':>10} {'Std.Err':>10} "
            f"{'t-value':>10} {'p-value':>10}"
        )
        lines.append("-" * width)
        for i, name in enumerate(self.param_names):
            lines.append(
                f"  {name:<15} {self.params[i]:>10.4f} {self.se[i]:>10.4f} "
                f"{self.tvalues[i]:>10.4f} {self.pvalues[i]:>10.4f}"
            )
        lines.append("=" * width)

        # Residual diagnostics
        resid = self.residuals[~np.isnan(self.residuals)]
        if len(resid) > 0:
            lines.append(
                f"  Residual std: {np.std(resid, ddof=1):.4f}    "
                f"Mean: {np.mean(resid):.4f}"
            )

            # Ljung-Box test on residuals (lag 10)
            n = len(resid)
            max_lag = min(10, n // 5)
            if max_lag > 0:
                acf_vals = _acf(resid, max_lag)
                lb_stat = n * (n + 2) * np.sum(
                    acf_vals[1:] ** 2 / (n - np.arange(1, max_lag + 1))
                )
                lb_pvalue = 1.0 - stats.chi2.cdf(lb_stat, df=max_lag)
                lines.append(
                    f"  Ljung-Box(lag={max_lag}): stat={lb_stat:.4f}  "
                    f"p-value={lb_pvalue:.4f}"
                )
        lines.append("=" * width)

        return "\n".join(lines)

    def forecast(
        self, steps: int = 1, alpha: float = 0.05
    ) -> dict[str, NDArray[np.float64]]:
        """Compute out-of-sample forecasts.

        Parameters
        ----------
        steps : int
            Number of forecast steps.
        alpha : float
            Significance level for confidence intervals.

        Returns
        -------
        dict with keys 'forecast', 'lower', 'upper'
            Point forecasts and confidence interval bounds.
        """
        if self._forecast_func is None:
            msg = "No forecast function available for this model"
            raise NotImplementedError(msg)
        return self._forecast_func(steps, alpha)

    def plot_diagnostics(self, figsize: tuple[int, int] = (12, 8)) -> object:
        """Plot diagnostic panel (2x2).

        1. Standardized residuals vs time
        2. ACF of residuals
        3. QQ-plot
        4. Histogram with normal curve

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt

        resid = self.residuals[~np.isnan(self.residuals)]
        std_resid = resid / np.std(resid, ddof=1) if np.std(resid) > 0 else resid

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Standardized residuals
        ax = axes[0, 0]
        ax.plot(std_resid)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        ax.axhline(y=2, color="r", linestyle="--", linewidth=0.5)
        ax.axhline(y=-2, color="r", linestyle="--", linewidth=0.5)
        ax.set_title("Standardized Residuals")
        ax.set_xlabel("Observation")

        # 2. ACF
        ax = axes[0, 1]
        n = len(resid)
        max_lag = min(20, n // 4)
        acf_vals = _acf(resid, max_lag)
        ax.bar(range(max_lag + 1), acf_vals, width=0.3, color="steelblue")
        conf = 1.96 / np.sqrt(n)
        ax.axhline(y=conf, color="r", linestyle="--", linewidth=0.5)
        ax.axhline(y=-conf, color="r", linestyle="--", linewidth=0.5)
        ax.set_title("ACF of Residuals")
        ax.set_xlabel("Lag")

        # 3. QQ-plot
        ax = axes[1, 0]
        stats.probplot(resid, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q Plot")

        # 4. Histogram
        ax = axes[1, 1]
        ax.hist(resid, bins="auto", density=True, alpha=0.7, color="steelblue")
        x = np.linspace(resid.min(), resid.max(), 100)
        ax.plot(x, stats.norm.pdf(x, np.mean(resid), np.std(resid)), "r-")
        ax.set_title("Histogram of Residuals")

        fig.suptitle(f"Diagnostics: {self.model_name}", fontsize=14)
        fig.tight_layout()
        return fig

    def plot_forecast(
        self,
        steps: int = 12,
        alpha: float = 0.05,
        figsize: tuple[int, int] = (12, 5),
    ) -> object:
        """Plot forecast with fan chart.

        Parameters
        ----------
        steps : int
            Number of forecast steps.
        alpha : float
            Significance level.
        figsize : tuple
            Figure size.

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt

        fc = self.forecast(steps, alpha)
        n = len(self.fitted_values)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(n), self.fitted_values, label="Fitted", color="steelblue")
        fc_x = range(n, n + steps)
        ax.plot(fc_x, fc["forecast"], label="Forecast", color="red")
        ax.fill_between(
            fc_x,
            fc["lower"],
            fc["upper"],
            alpha=0.2,
            color="red",
            label=f"{(1 - alpha) * 100:.0f}% CI",
        )
        ax.legend()
        ax.set_title(f"Forecast: {self.model_name}")
        fig.tight_layout()
        return fig

    def to_dataframe(self) -> pd.DataFrame:
        """Convert parameter results to DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: estimate, std_err, t_value, p_value.
        """
        return pd.DataFrame(
            {
                "estimate": self.params,
                "std_err": self.se,
                "t_value": self.tvalues,
                "p_value": self.pvalues,
            },
            index=self.param_names,
        )

    def save(self, path: str) -> None:
        """Save results to file via pickle.

        Parameters
        ----------
        path : str
            Output file path.
        """
        import pickle
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> TimeSeriesResults:
        """Load results from file.

        Parameters
        ----------
        path : str
            Input file path.

        Returns
        -------
        TimeSeriesResults
        """
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return (
            f"TimeSeriesResults(model='{self.model_name}', "
            f"nobs={self.nobs}, aic={self.aic:.2f})"
        )


def _acf(y: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    """Compute autocorrelation function.

    Parameters
    ----------
    y : ndarray
        Input series.
    max_lag : int
        Maximum lag.

    Returns
    -------
    ndarray
        ACF values from lag 0 to max_lag.
    """
    n = len(y)
    y_centered = y - np.mean(y)
    c0 = np.dot(y_centered, y_centered) / n
    if c0 == 0:
        return np.zeros(max_lag + 1)
    acf_vals = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        acf_vals[k] = np.dot(y_centered[: n - k], y_centered[k:]) / n / c0
    return acf_vals
