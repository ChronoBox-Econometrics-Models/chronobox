"""
Plot helpers for statistical tests examples.

Provides visualization functions for unit root tests, cointegration analysis,
structural break detection, and CUSUM diagnostics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple


def plot_rejection_rates(
    rejection_rates: Dict[str, float],
    nominal_size: float = 0.05,
    title: str = "Test Rejection Rates",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Axes:
    """
    Bar chart of rejection rates across different tests.

    Parameters
    ----------
    rejection_rates : dict
        Mapping of test name to rejection rate (0 to 1).
    nominal_size : float
        Nominal significance level (plotted as horizontal reference line).
    title : str
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on. If None, a new figure is created.
    figsize : tuple
        Figure size if creating a new figure.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    names = list(rejection_rates.keys())
    rates = list(rejection_rates.values())

    colors = ["#2ecc71" if r <= nominal_size * 1.5 else "#e74c3c" for r in rates]
    bars = ax.bar(names, rates, color=colors, alpha=0.8, edgecolor="black")

    ax.axhline(y=nominal_size, color="navy", linestyle="--", linewidth=1.5,
               label=f"Nominal size ({nominal_size:.0%})")
    ax.set_ylabel("Rejection Rate")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, max(rates) * 1.3 if rates else 0.2)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{rate:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return ax


def plot_confidence_bands(
    series: pd.Series,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str = "Series with Confidence Bands",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (12, 6),
    label: str = "Series",
    band_label: str = "95% CI",
) -> plt.Axes:
    """
    Plot a time series with confidence bands.

    Parameters
    ----------
    series : pd.Series
        The time series data.
    lower : np.ndarray
        Lower confidence band.
    upper : np.ndarray
        Upper confidence band.
    title : str
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.
    label : str
        Label for the series line.
    band_label : str
        Label for the confidence band.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(series))
    ax.plot(x, series.values, color="steelblue", linewidth=1.5, label=label)
    ax.fill_between(x, lower, upper, alpha=0.2, color="steelblue", label=band_label)
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("Observation")
    ax.set_ylabel("Value")
    plt.tight_layout()
    return ax


def plot_cusum(
    cusum: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    title: str = "CUSUM Test",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Axes:
    """
    Plot CUSUM statistic with critical value boundaries.

    Parameters
    ----------
    cusum : np.ndarray
        CUSUM statistic values.
    upper_bound : np.ndarray
        Upper critical value boundary.
    lower_bound : np.ndarray
        Lower critical value boundary.
    title : str
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(cusum))
    ax.plot(x, cusum, color="steelblue", linewidth=1.5, label="CUSUM")
    ax.plot(x, upper_bound, color="red", linestyle="--", linewidth=1, label="5% Critical Value")
    ax.plot(x, lower_bound, color="red", linestyle="--", linewidth=1)
    ax.fill_between(x, lower_bound, upper_bound, alpha=0.1, color="red")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Observation")
    ax.set_ylabel("CUSUM")
    ax.legend()
    plt.tight_layout()
    return ax


def plot_structural_break(
    series: pd.Series,
    break_index: int,
    title: str = "Series with Structural Break",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (12, 6),
    show_regimes: bool = True,
) -> plt.Axes:
    """
    Plot a time series highlighting the structural break point.

    Parameters
    ----------
    series : pd.Series
        The time series data.
    break_index : int
        Index of the detected structural break.
    title : str
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on.
    figsize : tuple
        Figure size.
    show_regimes : bool
        Whether to color pre/post-break regimes differently.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(series))

    if show_regimes:
        ax.plot(x[:break_index], series.values[:break_index],
                color="steelblue", linewidth=1.2, label="Regime 1")
        ax.plot(x[break_index:], series.values[break_index:],
                color="coral", linewidth=1.2, label="Regime 2")
    else:
        ax.plot(x, series.values, color="steelblue", linewidth=1.2)

    ax.axvline(x=break_index, color="red", linestyle="--", linewidth=2,
               label=f"Break at t={break_index}")

    # Annotate means of each regime
    mean_before = np.mean(series.values[:break_index])
    mean_after = np.mean(series.values[break_index:])
    ax.axhline(y=mean_before, xmin=0, xmax=break_index / len(series),
               color="steelblue", linestyle=":", alpha=0.7)
    ax.axhline(y=mean_after, xmin=break_index / len(series), xmax=1,
               color="coral", linestyle=":", alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    return ax


def plot_unit_root_series(
    series_dict: Dict[str, pd.Series],
    title: str = "Unit Root Test Series Comparison",
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot multiple series for unit root comparison (I(0), I(1), I(2)).

    Parameters
    ----------
    series_dict : dict
        Mapping of series name to pd.Series.
    title : str
        Overall figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
    """
    n_series = len(series_dict)
    fig, axes = plt.subplots(n_series, 1, figsize=figsize, sharex=True)
    if n_series == 1:
        axes = [axes]

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12"]

    for i, (name, series) in enumerate(series_dict.items()):
        color = colors[i % len(colors)]
        axes[i].plot(series.values, color=color, linewidth=1.2)
        axes[i].set_ylabel(name)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Observation")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_cointegration_residuals(
    y: pd.Series,
    x: pd.Series,
    residuals: pd.Series,
    title: str = "Cointegration Analysis",
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot cointegrated series and their equilibrium error.

    Parameters
    ----------
    y : pd.Series
        Dependent variable.
    x : pd.Series
        Independent variable.
    residuals : pd.Series
        Cointegration residuals (equilibrium error).
    title : str
        Overall figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Panel 1: Both series
    axes[0].plot(y.values, label="y", color="steelblue", linewidth=1.2)
    axes[0].plot(x.values, label="x", color="coral", linewidth=1.2)
    axes[0].set_title("Cointegrated Series")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Scatter plot
    axes[1].scatter(x.values, y.values, alpha=0.5, s=15, color="steelblue")
    z = np.polyfit(x.values, y.values, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(x.values)
    axes[1].plot(x_sorted, p(x_sorted), color="red", linewidth=1.5,
                 label=f"y = {z[0]:.3f}x + {z[1]:.3f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Cointegrating Relationship")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Residuals (should be stationary)
    axes[2].plot(residuals.values, color="green", linewidth=1.0)
    axes[2].axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    axes[2].set_title("Equilibrium Error (should be stationary)")
    axes[2].set_xlabel("Observation")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    return fig
