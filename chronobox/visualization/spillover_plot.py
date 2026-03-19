"""Spillover visualizations: network graph, heatmap, and rolling total index.

Provides three functions:
    - plot_network(spillover): Directed graph with edge thickness proportional to spillover
    - plot_heatmap(spillover): Heatmap of KxK spillover table
    - plot_rolling(rolling_spillover): Rolling total spillover index over time

Usage:
    from chronobox.visualization.spillover_plot import plot_network, plot_heatmap, plot_rolling

    fig = plot_network(spillover_table, var_names=['GDP', 'CPI', 'Rate'])
    fig = plot_heatmap(spillover_table, var_names=['GDP', 'CPI', 'Rate'])
    fig = plot_rolling(rolling_total, dates=dates)
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from chronobox.visualization.themes import get_color_cycle, get_theme


def plot_network(
    spillover: Any | None = None,
    spillover_table: NDArray[np.float64] | None = None,
    var_names: list[str] | None = None,
    threshold: float = 1.0,
    node_size: float = 2000,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot spillover network graph.

    Creates a directed graph where node positions are arranged in a circle
    and edge thickness is proportional to the net pairwise spillover.

    Parameters
    ----------
    spillover : spillover results object or None
        Spillover results.
    spillover_table : ndarray of shape (K, K) or None
        Spillover table. spillover_table[i, j] = spillover FROM j TO i.
    var_names : list[str] or None
        Variable names.
    threshold : float
        Minimum spillover value to draw an edge.
    node_size : float
        Node size in scatter plot units.
    title : str or None
        Plot title.
    figsize : tuple or None
        Figure size.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Figure
        Matplotlib figure with network graph.
    """
    theme = get_theme()
    colors = get_color_cycle(theme)

    if spillover_table is None and spillover is not None:
        spillover_table, var_names = _extract_spillover_data(spillover)

    if spillover_table is None:
        msg = "Either 'spillover' or 'spillover_table' must be provided."
        raise ValueError(msg)

    spillover_table = np.asarray(spillover_table, dtype=np.float64)
    n_vars = spillover_table.shape[0]

    if var_names is None:
        var_names = [f"Var {i + 1}" for i in range(n_vars)]

    if figsize is None:
        figsize = (10, 10)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Arrange nodes in a circle
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    radius = 3.0
    x_pos = radius * np.cos(angles)
    y_pos = radius * np.sin(angles)

    # Draw edges
    max_spillover = np.max(np.abs(spillover_table))
    if max_spillover == 0:
        max_spillover = 1.0

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            val = spillover_table[i, j]
            if abs(val) < threshold:
                continue

            # Edge from j to i (j is the source of spillover TO i)
            edge_width = max(0.5, 4.0 * abs(val) / max_spillover)
            edge_color = theme.positive_color if val > 0 else theme.negative_color
            edge_alpha = min(0.9, 0.3 + 0.6 * abs(val) / max_spillover)

            ax.annotate(
                "",
                xy=(x_pos[i], y_pos[i]),
                xytext=(x_pos[j], y_pos[j]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": edge_color,
                    "lw": edge_width,
                    "alpha": edge_alpha,
                    "connectionstyle": "arc3,rad=0.1",
                },
            )

    # Draw nodes
    for i in range(n_vars):
        color = colors[i % len(colors)]
        ax.scatter(
            x_pos[i], y_pos[i],
            s=node_size, c=color, zorder=5,
            edgecolors=theme.text_color, linewidth=1.5,
        )
        ax.text(
            x_pos[i], y_pos[i],
            var_names[i],
            ha="center", va="center",
            fontsize=theme.label_size,
            fontweight="bold",
            color="white" if _is_dark(color) else theme.text_color,
            zorder=6,
        )

    ax.set_xlim(-radius * 1.8, radius * 1.8)
    ax.set_ylim(-radius * 1.8, radius * 1.8)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=theme.title_size, fontweight="bold")
    else:
        ax.set_title("Spillover Network", fontsize=theme.title_size, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_heatmap(
    spillover: Any | None = None,
    spillover_table: NDArray[np.float64] | None = None,
    var_names: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str | None = None,
    annotate: bool = True,
    fmt: str = ".1f",
    **kwargs: Any,
) -> Figure:
    """Plot spillover heatmap.

    Parameters
    ----------
    spillover : spillover results object or None
        Spillover results.
    spillover_table : ndarray of shape (K, K) or None
        Spillover table.
    var_names : list[str] or None
        Variable names.
    title : str or None
        Plot title.
    figsize : tuple or None
        Figure size.
    cmap : str or None
        Colormap name. If None, auto-selected based on theme.
    annotate : bool
        Whether to annotate cells with values.
    fmt : str
        Number format for cell annotations.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Figure
        Matplotlib figure with heatmap.
    """
    theme = get_theme()

    if spillover_table is None and spillover is not None:
        spillover_table, var_names = _extract_spillover_data(spillover)

    if spillover_table is None:
        msg = "Either 'spillover' or 'spillover_table' must be provided."
        raise ValueError(msg)

    spillover_table = np.asarray(spillover_table, dtype=np.float64)
    n_vars = spillover_table.shape[0]

    if var_names is None:
        var_names = [f"Var {i + 1}" for i in range(n_vars)]

    if cmap is None:
        cmap = "Greys" if theme.name == "academic" else "YlOrRd"

    if figsize is None:
        figsize = (8, 7)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(spillover_table, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Annotate cells
    if annotate:
        for i in range(n_vars):
            for j in range(n_vars):
                val = spillover_table[i, j]
                text_color = "white" if val > spillover_table.max() * 0.6 else theme.text_color
                ax.text(
                    j, i, f"{val:{fmt}}",
                    ha="center", va="center",
                    fontsize=theme.tick_size,
                    color=text_color,
                )

    # Labels
    ax.set_xticks(range(n_vars))
    ax.set_yticks(range(n_vars))
    ax.set_xticklabels(var_names, fontsize=theme.tick_size, rotation=45, ha="right")
    ax.set_yticklabels(var_names, fontsize=theme.tick_size)
    ax.set_xlabel("From", fontsize=theme.label_size)
    ax.set_ylabel("To", fontsize=theme.label_size)

    if title:
        ax.set_title(title, fontsize=theme.title_size, fontweight="bold")
    else:
        ax.set_title("Spillover Table", fontsize=theme.title_size, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_rolling(
    rolling_spillover: Any | None = None,
    rolling_total: NDArray[np.float64] | None = None,
    dates: Any | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ylabel: str | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot rolling total spillover index over time.

    Parameters
    ----------
    rolling_spillover : rolling spillover results or None
        Rolling spillover results object.
    rolling_total : ndarray or None
        Total spillover index over time.
    dates : array or None
        Date index.
    title : str or None
        Plot title.
    figsize : tuple or None
        Figure size.
    ylabel : str or None
        Y-axis label.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    theme = get_theme()

    if rolling_total is None and rolling_spillover is not None:
        rolling_total, dates = _extract_rolling_data(rolling_spillover)

    if rolling_total is None:
        msg = "Either 'rolling_spillover' or 'rolling_total' must be provided."
        raise ValueError(msg)

    rolling_total = np.asarray(rolling_total, dtype=np.float64)

    if figsize is None:
        figsize = (14, 5)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x = dates if dates is not None else np.arange(len(rolling_total))

    ax.plot(
        x, rolling_total,
        color=theme.colors[0],
        linewidth=theme.line_width,
        **kwargs,
    )
    ax.fill_between(
        x if isinstance(x, np.ndarray) else range(len(rolling_total)),
        rolling_total,
        alpha=0.15,
        color=theme.colors[0],
    )

    # Add mean line
    mean_val = np.nanmean(rolling_total)
    ax.axhline(
        y=mean_val, color=theme.negative_color,
        linewidth=0.8, linestyle="--", alpha=0.7,
        label=f"Mean = {mean_val:.1f}%",
    )

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=theme.label_size)
    else:
        ax.set_ylabel("Total Spillover Index (%)", fontsize=theme.label_size)

    if title:
        ax.set_title(title, fontsize=theme.title_size, fontweight="bold")
    else:
        ax.set_title("Rolling Total Spillover Index", fontsize=theme.title_size, fontweight="bold")

    ax.legend(loc="upper right", fontsize=theme.tick_size, frameon=theme.legend_frame)
    ax.grid(True, alpha=theme.grid_alpha, color=theme.grid_color)

    fig.tight_layout()
    return fig


def _extract_spillover_data(
    spillover: Any,
) -> tuple[NDArray[np.float64], list[str] | None]:
    """Extract spillover table from results.

    Parameters
    ----------
    spillover : Any
        Spillover results.

    Returns
    -------
    tuple of (table, var_names)
    """
    table = None
    var_names = None

    for attr in ("table", "spillover_table", "spillover_matrix", "decomp"):
        if hasattr(spillover, attr):
            table = np.asarray(getattr(spillover, attr), dtype=np.float64)
            break

    for attr in ("var_names", "names", "variable_names"):
        if hasattr(spillover, attr):
            var_names = list(getattr(spillover, attr))
            break

    if table is None:
        msg = "Cannot extract spillover data."
        raise AttributeError(msg)

    return table, var_names


def _extract_rolling_data(
    rolling_spillover: Any,
) -> tuple[NDArray[np.float64], Any]:
    """Extract rolling spillover data.

    Parameters
    ----------
    rolling_spillover : Any
        Rolling spillover results.

    Returns
    -------
    tuple of (total_index, dates)
    """
    total = None
    dates = None

    for attr in ("total_index", "total", "rolling_total", "tsi"):
        if hasattr(rolling_spillover, attr):
            total = np.asarray(getattr(rolling_spillover, attr), dtype=np.float64)
            break

    for attr in ("dates", "index"):
        if hasattr(rolling_spillover, attr):
            dates = getattr(rolling_spillover, attr)
            break

    if total is None:
        msg = "Cannot extract rolling spillover data."
        raise AttributeError(msg)

    return total, dates


def _is_dark(hex_color: str) -> bool:
    """Check if a hex color is dark (for text contrast).

    Parameters
    ----------
    hex_color : str
        Hex color string.

    Returns
    -------
    bool
        True if color is dark.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return True
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return luminance < 0.5
