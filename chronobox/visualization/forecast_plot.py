"""Forecast fan chart for chronobox.

Provides plot_forecast() which generates Bank of England / BCB-style fan charts
with degradee confidence intervals at 50%, 70%, 90%, and 95% levels.

Usage:
    from chronobox.visualization import plot_forecast

    results = model.fit(data)
    fig = plot_forecast(results, steps=24)
    fig = plot_forecast(results, steps=12, ci_levels=[0.5, 0.9, 0.95])
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import stats as sp_stats

from chronobox.visualization.themes import get_confidence_colors, get_theme


def plot_forecast(
    results: Any | None = None,
    steps: int = 12,
    alpha: float = 0.05,
    history: NDArray[np.float64] | None = None,
    forecast_mean: NDArray[np.float64] | None = None,
    forecast_se: NDArray[np.float64] | None = None,
    ci_levels: list[float] | None = None,
    history_periods: int = 50,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot forecast fan chart with confidence bands.

    Creates a fan chart with historical data as a solid line and forecast
    as a dashed line, with degradee confidence bands from dark (tightest)
    to light (widest).

    Parameters
    ----------
    results : model results object or None
        Model results with forecast capabilities. If None, history/forecast_mean/forecast_se
        must be provided directly.
    steps : int
        Number of forecast steps.
    alpha : float
        Significance level (default 0.05 for 95% interval).
    history : ndarray or None
        Historical series values. Extracted from results if None.
    forecast_mean : ndarray or None
        Point forecast. Computed from results if None.
    forecast_se : ndarray or None
        Forecast standard errors. Computed from results if None.
    ci_levels : list[float] or None
        Confidence interval levels (default [0.5, 0.7, 0.9, 0.95]).
    history_periods : int
        Number of historical periods to display.
    title : str or None
        Plot title.
    xlabel : str or None
        X-axis label.
    ylabel : str or None
        Y-axis label.
    figsize : tuple or None
        Figure size.
    ax : Axes or None
        Existing axes.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Figure
        Matplotlib figure with fan chart.
    """
    theme = get_theme()
    ci_colors = get_confidence_colors(theme)

    if ci_levels is None:
        ci_levels = [0.50, 0.70, 0.90, 0.95]

    # Sort levels from widest to narrowest for correct layering
    ci_levels_sorted = sorted(ci_levels, reverse=True)

    if figsize is None:
        figsize = (12, 6)

    # Extract data from results if not provided directly
    if history is None and results is not None:
        history = _extract_history(results)
    if forecast_mean is None and results is not None:
        forecast_mean, forecast_se = _extract_forecast(results, steps)

    if history is None or forecast_mean is None:
        msg = (
            "Either provide a results object or specify history, "
            "forecast_mean, and forecast_se directly."
        )
        raise ValueError(msg)

    history = np.asarray(history, dtype=np.float64)
    forecast_mean = np.asarray(forecast_mean, dtype=np.float64)

    if forecast_se is None:
        # If no SE, create constant SE based on history residual variance
        forecast_se = np.full(len(forecast_mean), np.std(history[-20:], ddof=1))

    forecast_se = np.asarray(forecast_se, dtype=np.float64)

    # Trim history to requested number of periods
    if len(history) > history_periods:
        history = history[-history_periods:]

    # Create x-axis indices
    n_hist = len(history)
    n_fcast = len(forecast_mean)
    x_hist = np.arange(n_hist)
    x_fcast = np.arange(n_hist - 1, n_hist + n_fcast)

    # Forecast line starts from last historical point
    fcast_line = np.concatenate([[history[-1]], forecast_mean])
    fcast_se_line = np.concatenate([[0.0], forecast_se])

    # Create figure
    if ax is None:
        fig, ax_plot = plt.subplots(1, 1, figsize=figsize)
    else:
        ax_plot = ax
        fig = ax_plot.get_figure()

    # Plot confidence bands (widest first, so narrowest is on top)
    for i, level in enumerate(ci_levels_sorted):
        z = sp_stats.norm.ppf(1 - (1 - level) / 2)
        upper = fcast_line + z * fcast_se_line
        lower = fcast_line - z * fcast_se_line

        color_idx = len(ci_levels) - 1 - i
        color = ci_colors[color_idx] if color_idx < len(ci_colors) else ci_colors[-1]

        # Compute alpha for degradee effect
        band_alpha = 0.3 + 0.4 * (1 - i / max(len(ci_levels_sorted) - 1, 1))

        ax_plot.fill_between(
            x_fcast, lower, upper,
            color=color, alpha=band_alpha,
            label=f"{int(level * 100)}% CI",
            zorder=1 + i,
        )

    # Plot historical data
    ax_plot.plot(
        x_hist, history,
        color=theme.colors[0], linewidth=theme.line_width,
        label="Historical", zorder=10,
    )

    # Plot forecast line (dashed)
    ax_plot.plot(
        x_fcast, fcast_line,
        color=theme.colors[0], linewidth=theme.line_width,
        linestyle="--", label="Forecast", zorder=10,
    )

    # Vertical line at forecast origin
    ax_plot.axvline(
        x=n_hist - 1, color=theme.text_color,
        linewidth=0.8, linestyle=":", alpha=0.5, zorder=5,
    )

    # Labels and title
    if title:
        ax_plot.set_title(title, fontsize=theme.title_size, fontweight="bold")
    else:
        ax_plot.set_title("Forecast", fontsize=theme.title_size, fontweight="bold")

    if xlabel:
        ax_plot.set_xlabel(xlabel, fontsize=theme.label_size)
    if ylabel:
        ax_plot.set_ylabel(ylabel, fontsize=theme.label_size)

    ax_plot.legend(loc="upper left", fontsize=theme.tick_size - 1, frameon=theme.legend_frame)
    ax_plot.grid(True, alpha=theme.grid_alpha, color=theme.grid_color, linestyle=theme.grid_style)

    fig.tight_layout()
    return fig


def _extract_history(results: Any) -> NDArray[np.float64]:
    """Extract historical series from results object.

    Parameters
    ----------
    results : Any
        Results object.

    Returns
    -------
    ndarray
        Historical values.
    """
    for attr in ("endog", "y", "data", "fittedvalues"):
        if hasattr(results, attr):
            val = getattr(results, attr)
            if callable(val):
                val = val()
            return np.asarray(val, dtype=np.float64).ravel()

    msg = "Cannot extract historical data from results. Provide 'history' directly."
    raise AttributeError(msg)


def _extract_forecast(
    results: Any, steps: int
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Extract forecast mean and standard errors from results.

    Parameters
    ----------
    results : Any
        Results object.
    steps : int
        Number of forecast steps.

    Returns
    -------
    tuple of (mean, se)
        Forecast mean and standard errors (se may be None).
    """
    # Try forecast method
    if hasattr(results, "forecast"):
        fc = results.forecast(steps=steps)
        if isinstance(fc, tuple):
            if len(fc) >= 2:
                return np.asarray(fc[0], dtype=np.float64), np.asarray(fc[1], dtype=np.float64)
            return np.asarray(fc[0], dtype=np.float64), None  # type: ignore[index]
        return np.asarray(fc, dtype=np.float64), None

    # Try get_forecast method (statsmodels style)
    if hasattr(results, "get_forecast"):
        fc_obj = results.get_forecast(steps=steps)
        mean = np.asarray(fc_obj.predicted_mean, dtype=np.float64)
        se = None
        if hasattr(fc_obj, "se_mean"):
            se = np.asarray(fc_obj.se_mean, dtype=np.float64)
        return mean, se

    msg = "Cannot extract forecast from results. Provide forecast_mean directly."
    raise AttributeError(msg)
