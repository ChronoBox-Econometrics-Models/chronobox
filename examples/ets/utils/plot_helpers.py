"""
Plot helpers for ETS diagnostic analysis.

Provides functions for ETS decomposition plots, component visualization,
residual diagnostics, and cross-validation comparison between Python, R, and Stata.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats


def plot_ets_decomposition(
    observed: pd.Series,
    level: pd.Series | np.ndarray | None = None,
    trend: pd.Series | np.ndarray | None = None,
    seasonal: pd.Series | np.ndarray | None = None,
    residuals: pd.Series | np.ndarray | None = None,
    title: str = "ETS Decomposition",
    figsize: tuple = (12, 12),
) -> plt.Figure:
    """Plot ETS decomposition: observed, level, trend, seasonal, and residuals.

    Parameters
    ----------
    observed : pd.Series
        Observed time series (with DatetimeIndex).
    level : array-like or None
        Estimated level component.
    trend : array-like or None
        Estimated trend component.
    seasonal : array-like or None
        Estimated seasonal component.
    residuals : array-like or None
        Model residuals.
    title : str
        Overall figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_panels = 1 + sum(x is not None for x in [level, trend, seasonal, residuals])
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14)

    idx = observed.index if isinstance(observed, pd.Series) else np.arange(len(observed))
    panel = 0

    axes[panel].plot(idx, observed, color="black", linewidth=0.8)
    axes[panel].set_title("Observed")
    panel += 1

    if level is not None:
        axes[panel].plot(idx[:len(level)], level, color="blue", linewidth=0.8)
        axes[panel].set_title("Level")
        panel += 1

    if trend is not None:
        axes[panel].plot(idx[:len(trend)], trend, color="green", linewidth=0.8)
        axes[panel].set_title("Trend")
        panel += 1

    if seasonal is not None:
        axes[panel].plot(idx[:len(seasonal)], seasonal, color="orange", linewidth=0.8)
        axes[panel].set_title("Seasonal")
        panel += 1

    if residuals is not None:
        axes[panel].plot(idx[:len(residuals)], residuals, color="red", linewidth=0.8)
        axes[panel].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        axes[panel].set_title("Residuals")

    fig.tight_layout()
    return fig


def plot_seasonal_pattern(
    seasonal: np.ndarray | pd.Series,
    seasonal_periods: int = 12,
    labels: list | None = None,
    title: str = "Seasonal Pattern",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Plot seasonal indices as a bar chart or line plot across one cycle.

    Parameters
    ----------
    seasonal : array-like
        Full seasonal component from which the first cycle is extracted,
        or a single cycle of seasonal indices.
    seasonal_periods : int
        Number of periods in one seasonal cycle.
    labels : list or None
        Labels for each period. Defaults to month abbreviations for s=12.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    seasonal = np.asarray(seasonal)
    if len(seasonal) > seasonal_periods:
        # Average across full cycles
        n_full = len(seasonal) // seasonal_periods
        seasonal = seasonal[:n_full * seasonal_periods].reshape(n_full, seasonal_periods).mean(axis=0)
    elif len(seasonal) < seasonal_periods:
        raise ValueError(f"seasonal has {len(seasonal)} values, need at least {seasonal_periods}")

    if labels is None:
        if seasonal_periods == 12:
            labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        elif seasonal_periods == 4:
            labels = ["Q1", "Q2", "Q3", "Q4"]
        else:
            labels = [str(i + 1) for i in range(seasonal_periods)]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, seasonal, color="steelblue", edgecolor="black", alpha=0.8)

    # Color positive/negative differently
    for bar, val in zip(bars, seasonal):
        if val < 0:
            bar.set_color("salmon")

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Seasonal Index")
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

    # 2. Histogram with normal density
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
    title: str = "ETS Forecast",
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
