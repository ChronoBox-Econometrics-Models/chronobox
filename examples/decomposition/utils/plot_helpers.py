"""
Plot helpers for decomposition analysis.

Provides functions for visualizing decomposed time series components
(trend, seasonal, residual) and cross-validation comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_decomposition(
    observed: np.ndarray | pd.Series,
    trend: np.ndarray | pd.Series,
    seasonal: np.ndarray | pd.Series,
    residual: np.ndarray | pd.Series,
    dates: pd.DatetimeIndex | None = None,
    title: str = "Time Series Decomposition",
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Plot a 4-panel decomposition: observed, trend, seasonal, residual.

    Parameters
    ----------
    observed : array-like
        Original time series.
    trend : array-like
        Trend component.
    seasonal : array-like
        Seasonal component.
    residual : array-like
        Residual component.
    dates : pd.DatetimeIndex or None
        Date index for the x-axis. If None, uses integer index.
    title : str
        Overall figure title.
    figsize : tuple
        Figure size (width, height).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14)

    x = dates if dates is not None else np.arange(len(observed))

    components = [
        (observed, "Observed", "black"),
        (trend, "Trend", "blue"),
        (seasonal, "Seasonal", "green"),
        (residual, "Residual", "red"),
    ]

    for ax, (data, label, color) in zip(axes, components):
        ax.plot(x, np.asarray(data), color=color, linewidth=0.8)
        ax.set_ylabel(label)
        if label == "Residual":
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    return fig


def plot_seasonal_subseries(
    seasonal: np.ndarray | pd.Series,
    s: int = 12,
    labels: list | None = None,
    title: str = "Seasonal Subseries Plot",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot seasonal subseries (one subplot per season).

    Parameters
    ----------
    seasonal : array-like
        Seasonal component (full length, will be reshaped).
    s : int
        Seasonal period.
    labels : list or None
        Labels for each season (e.g., month names). If None, uses integers.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    seasonal = np.asarray(seasonal)
    n_cycles = len(seasonal) // s

    if labels is None:
        labels = [str(i + 1) for i in range(s)]

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=14)

    for i in range(s):
        values = seasonal[i::s][:n_cycles]
        x_positions = np.arange(len(values)) + i * (n_cycles + 1)
        ax.bar(x_positions, values, width=0.8, alpha=0.7)
        ax.axhline(y=values.mean(), xmin=(i * (n_cycles + 1)) / (s * (n_cycles + 1)),
                    xmax=((i + 1) * (n_cycles + 1) - 1) / (s * (n_cycles + 1)),
                    color="red", linewidth=1.5)

    tick_positions = [i * (n_cycles + 1) + n_cycles / 2 for i in range(s)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels[:s])
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    return fig


def plot_stl_diagnostics(
    observed: np.ndarray | pd.Series,
    trend: np.ndarray | pd.Series,
    seasonal: np.ndarray | pd.Series,
    residual: np.ndarray | pd.Series,
    weights: np.ndarray | None = None,
    dates: pd.DatetimeIndex | None = None,
    title: str = "STL Decomposition Diagnostics",
    figsize: tuple = (12, 12),
) -> plt.Figure:
    """Plot STL decomposition with optional robustness weights.

    Parameters
    ----------
    observed : array-like
        Original time series.
    trend : array-like
        Trend component from STL.
    seasonal : array-like
        Seasonal component from STL.
    residual : array-like
        Residual component from STL.
    weights : array-like or None
        Robustness weights from STL (if robust=True was used).
    dates : pd.DatetimeIndex or None
        Date index for the x-axis.
    title : str
        Overall figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_panels = 5 if weights is not None else 4
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14)

    x = dates if dates is not None else np.arange(len(observed))

    axes[0].plot(x, np.asarray(observed), color="black", linewidth=0.8)
    axes[0].set_ylabel("Observed")

    axes[1].plot(x, np.asarray(trend), color="blue", linewidth=0.8)
    axes[1].set_ylabel("Trend")

    axes[2].plot(x, np.asarray(seasonal), color="green", linewidth=0.8)
    axes[2].set_ylabel("Seasonal")

    axes[3].plot(x, np.asarray(residual), color="red", linewidth=0.8)
    axes[3].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    axes[3].set_ylabel("Residual")

    if weights is not None:
        axes[4].plot(x, np.asarray(weights), color="purple", linewidth=0.8)
        axes[4].set_ylabel("Weights")
        axes[4].set_ylim(-0.05, 1.05)

    fig.tight_layout()
    return fig


def compare_components(
    python_components: dict,
    r_components: dict | None = None,
    stata_components: dict | None = None,
    component_name: str = "trend",
    dates: pd.DatetimeIndex | None = None,
    title: str | None = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Compare a decomposition component across Python, R, and Stata.

    Parameters
    ----------
    python_components : dict
        Dict with component arrays from Python (keys: 'trend', 'seasonal', 'residual').
    r_components : dict or None
        Dict with component arrays from R.
    stata_components : dict or None
        Dict with component arrays from Stata.
    component_name : str
        Which component to compare ('trend', 'seasonal', 'residual').
    dates : pd.DatetimeIndex or None
        Date index for the x-axis.
    title : str or None
        Plot title. If None, auto-generated from component_name.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if title is None:
        title = f"Comparison of {component_name.title()} Component"

    fig, ax = plt.subplots(figsize=figsize)

    x = dates if dates is not None else np.arange(len(python_components[component_name]))

    ax.plot(x, python_components[component_name], label="Python (chronobox)",
            color="blue", linewidth=1.2)

    if r_components is not None and component_name in r_components:
        ax.plot(x, r_components[component_name], label="R",
                color="red", linestyle="--", linewidth=1.2)

    if stata_components is not None and component_name in stata_components:
        ax.plot(x, stata_components[component_name], label="Stata",
                color="green", linestyle=":", linewidth=1.2)

    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
