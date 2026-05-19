"""
Plot helpers for ARIMA diagnostic analysis.

Provides functions for ACF/PACF plots, residual diagnostics,
and cross-validation comparison between Python, R, and Stata results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats


def plot_acf_pacf(
    series: np.ndarray | pd.Series,
    lags: int = 40,
    title: str = "",
    figsize: tuple = (12, 5),
    alpha: float = 0.05,
) -> plt.Figure:
    """Plot ACF and PACF side by side.

    Parameters
    ----------
    series : array-like
        Time series data.
    lags : int
        Number of lags to display.
    title : str
        Title prefix for the plots.
    figsize : tuple
        Figure size (width, height).
    alpha : float
        Significance level for confidence bands.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_acf(series, lags=lags, alpha=alpha, ax=axes[0])
    axes[0].set_title(f"{title} ACF".strip())

    plot_pacf(series, lags=lags, alpha=alpha, ax=axes[1], method="ywm")
    axes[1].set_title(f"{title} PACF".strip())

    fig.tight_layout()
    return fig


def plot_residual_diagnostics(
    residuals: np.ndarray | pd.Series,
    lags: int = 30,
    title: str = "Residual Diagnostics",
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Plot a 4-panel residual diagnostic: time series, histogram, Q-Q, ACF.

    Parameters
    ----------
    residuals : array-like
        Model residuals.
    lags : int
        Number of lags for the ACF plot.
    title : str
        Overall figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    residuals = np.asarray(residuals)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    # 1. Residuals over time
    axes[0, 0].plot(residuals, linewidth=0.8)
    axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=0.5)
    axes[0, 0].set_title("Residuals")
    axes[0, 0].set_xlabel("Observation")

    # 2. Histogram with KDE
    axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor="black")
    x_grid = np.linspace(residuals.min(), residuals.max(), 200)
    axes[0, 1].plot(x_grid, stats.norm.pdf(x_grid, residuals.mean(), residuals.std()),
                    "r-", linewidth=2)
    axes[0, 1].set_title("Histogram + Normal Density")

    # 3. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")

    # 4. ACF of residuals
    plot_acf(residuals, lags=lags, ax=axes[1, 1])
    axes[1, 1].set_title("ACF of Residuals")

    fig.tight_layout()
    return fig


def plot_forecast(
    observed: pd.Series,
    fitted: pd.Series | None = None,
    forecast: pd.Series | None = None,
    ci_lower: pd.Series | None = None,
    ci_upper: pd.Series | None = None,
    title: str = "Forecast",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot observed data with optional fitted values, forecast, and confidence interval.

    Parameters
    ----------
    observed : pd.Series
        Observed time series (with DatetimeIndex).
    fitted : pd.Series or None
        In-sample fitted values.
    forecast : pd.Series or None
        Out-of-sample forecast.
    ci_lower : pd.Series or None
        Lower confidence band for forecast.
    ci_upper : pd.Series or None
        Upper confidence band for forecast.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(observed.index, observed.values, label="Observed", color="black")

    if fitted is not None:
        ax.plot(fitted.index, fitted.values, label="Fitted", color="blue", alpha=0.7)

    if forecast is not None:
        ax.plot(forecast.index, forecast.values, label="Forecast", color="red")

    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(ci_lower.index, ci_lower.values, ci_upper.values,
                        alpha=0.2, color="red", label="95% CI")

    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def compare_results(
    python_results: dict,
    r_results: dict | None = None,
    stata_results: dict | None = None,
    metric_names: list | None = None,
) -> pd.DataFrame:
    """Compare estimation results across Python, R, and Stata.

    Parameters
    ----------
    python_results : dict
        Dict of metric_name -> value from Python estimation.
    r_results : dict or None
        Dict of metric_name -> value from R estimation.
    stata_results : dict or None
        Dict of metric_name -> value from Stata estimation.
    metric_names : list or None
        Specific metrics to compare. If None, uses all keys from python_results.

    Returns
    -------
    pd.DataFrame
        Comparison table with columns for each tool.
    """
    if metric_names is None:
        metric_names = list(python_results.keys())

    data = {"Metric": metric_names, "Python": [python_results.get(m) for m in metric_names]}

    if r_results is not None:
        data["R"] = [r_results.get(m) for m in metric_names]

    if stata_results is not None:
        data["Stata"] = [stata_results.get(m) for m in metric_names]

    df = pd.DataFrame(data).set_index("Metric")
    return df


def ljung_box_summary(residuals: np.ndarray, lags: list | None = None) -> pd.DataFrame:
    """Run Ljung-Box test on residuals and return summary table.

    Parameters
    ----------
    residuals : array-like
        Model residuals.
    lags : list or None
        Specific lags to test. Defaults to [10, 20, 30].

    Returns
    -------
    pd.DataFrame
        Ljung-Box test results with statistic and p-value.
    """
    if lags is None:
        lags = [10, 20, 30]

    result = acorr_ljungbox(residuals, lags=lags, return_df=True)
    return result
