"""Forecast Error Variance Decomposition (FEVD) plots.

Provides plot_fevd() which generates stacked area or stacked bar charts
showing the proportion of forecast error variance attributable to each shock.

Usage:
    from chronobox.visualization.fevd_plot import plot_fevd

    fig = plot_fevd(fevd_results)
    fig = plot_fevd(fevd_results, variable='gdp', plot_type='bar')
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from chronobox.visualization.themes import get_color_cycle, get_theme


def plot_fevd(
    fevd_results: Any | None = None,
    fevd_array: NDArray[np.float64] | None = None,
    var_names: list[str] | None = None,
    variable: str | int | None = None,
    plot_type: str = "area",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot Forecast Error Variance Decomposition.

    Two visualization modes:
    - Stacked area: Shows proportion over horizons for each response variable
    - Stacked bar: Shows bars for selected horizons

    Parameters
    ----------
    fevd_results : FEVD results object or None
        FEVD results. If None, fevd_array must be provided.
    fevd_array : ndarray of shape (H, K, K) or None
        FEVD values. fevd_array[h, i, j] = proportion of variance of var i
        explained by shock j at horizon h.
    var_names : list[str] or None
        Variable/shock names.
    variable : str or int or None
        Plot only for this response variable. If None, plots all.
    plot_type : str
        'area' for stacked area chart, 'bar' for stacked bar chart.
    title : str or None
        Overall figure title.
    figsize : tuple or None
        Figure size.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Figure
        Matplotlib figure with FEVD visualization.
    """
    theme = get_theme()
    colors = get_color_cycle(theme)

    # Extract data
    if fevd_array is None and fevd_results is not None:
        fevd_array, var_names = _extract_fevd_data(fevd_results)

    if fevd_array is None:
        msg = "Either 'fevd_results' or 'fevd_array' must be provided."
        raise ValueError(msg)

    fevd_array = np.asarray(fevd_array, dtype=np.float64)
    n_horizons, k_resp, k_shock = fevd_array.shape

    if var_names is None:
        var_names = [f"Var {i + 1}" for i in range(max(k_resp, k_shock))]

    # Determine which variables to plot
    if variable is not None:
        resp_indices = [var_names.index(variable)] if isinstance(variable, str) else [variable]
    else:
        resp_indices = list(range(k_resp))

    n_panels = len(resp_indices)

    if figsize is None:
        figsize = (12, 4 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, squeeze=False)

    horizons = np.arange(1, n_horizons + 1)

    for panel_idx, resp_idx in enumerate(resp_indices):
        ax = axes[panel_idx, 0]
        resp_name = var_names[resp_idx] if resp_idx < len(var_names) else f"Var {resp_idx + 1}"

        # Get FEVD data for this response variable: shape (H, K_shock)
        fevd_data = fevd_array[:, resp_idx, :]

        if plot_type == "area":
            _plot_stacked_area(ax, horizons, fevd_data, var_names, colors, theme)
        else:
            _plot_stacked_bar(ax, horizons, fevd_data, var_names, colors, theme)

        ax.set_ylabel("Proportion", fontsize=theme.label_size)
        ax.set_xlabel("Horizon", fontsize=theme.tick_size)
        ax.set_title(
            f"FEVD: {resp_name}",
            fontsize=theme.label_size,
            fontweight="bold",
        )
        ax.set_ylim(0, 1.0)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=theme.tick_size - 1,
            frameon=theme.legend_frame,
        )
        ax.grid(True, alpha=theme.grid_alpha * 0.3, color=theme.grid_color, axis="y")

    if title:
        fig.suptitle(title, fontsize=theme.title_size + 2, fontweight="bold", y=1.02)
    else:
        fig.suptitle(
            "Forecast Error Variance Decomposition",
            fontsize=theme.title_size + 2,
            fontweight="bold",
            y=1.02,
        )

    fig.tight_layout()
    return fig


def _plot_stacked_area(
    ax: plt.Axes,
    horizons: NDArray[np.float64],
    fevd_data: NDArray[np.float64],
    var_names: list[str],
    colors: list[str],
    theme: Any,
) -> None:
    """Plot stacked area chart.

    Parameters
    ----------
    ax : Axes
        Target axes.
    horizons : ndarray
        Horizon indices.
    fevd_data : ndarray of shape (H, K)
        FEVD proportions.
    var_names : list[str]
        Shock names.
    colors : list[str]
        Color palette.
    theme : ThemeConfig
        Current theme.
    """
    n_shocks = fevd_data.shape[1]
    labels = [var_names[j] if j < len(var_names) else f"Shock {j + 1}" for j in range(n_shocks)]

    ax.stackplot(
        horizons,
        *[fevd_data[:, j] for j in range(n_shocks)],
        labels=labels,
        colors=[colors[j % len(colors)] for j in range(n_shocks)],
        alpha=0.8,
    )


def _plot_stacked_bar(
    ax: plt.Axes,
    horizons: NDArray[np.float64],
    fevd_data: NDArray[np.float64],
    var_names: list[str],
    colors: list[str],
    theme: Any,
) -> None:
    """Plot stacked bar chart.

    Parameters
    ----------
    ax : Axes
        Target axes.
    horizons : ndarray
        Horizon indices.
    fevd_data : ndarray of shape (H, K)
        FEVD proportions.
    var_names : list[str]
        Shock names.
    colors : list[str]
        Color palette.
    theme : ThemeConfig
        Current theme.
    """
    n_shocks = fevd_data.shape[1]
    bottom = np.zeros(len(horizons))

    for j in range(n_shocks):
        label = var_names[j] if j < len(var_names) else f"Shock {j + 1}"
        color = colors[j % len(colors)]

        ax.bar(
            horizons, fevd_data[:, j],
            bottom=bottom,
            label=label,
            color=color,
            alpha=0.85,
            edgecolor=theme.background,
            linewidth=0.5,
        )
        bottom += fevd_data[:, j]


def _extract_fevd_data(
    fevd_results: Any,
) -> tuple[NDArray[np.float64], list[str] | None]:
    """Extract FEVD data from a results object.

    Parameters
    ----------
    fevd_results : Any
        FEVD results object.

    Returns
    -------
    tuple of (fevd_array, var_names)
    """
    fevd_array = None
    var_names = None

    # Try different attribute names
    for attr in ("decomp", "fevd", "variance_decomp", "fevd_array"):
        if hasattr(fevd_results, attr):
            val = getattr(fevd_results, attr)
            if callable(val):
                val = val()
            fevd_array = np.asarray(val, dtype=np.float64)
            break

    # Variable names
    for attr in ("names", "var_names", "variable_names", "columns"):
        if hasattr(fevd_results, attr):
            var_names = list(getattr(fevd_results, attr))
            break

    if fevd_array is None:
        msg = "Cannot extract FEVD data from results object."
        raise AttributeError(msg)

    return fevd_array, var_names
