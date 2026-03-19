"""Time series plot for chronobox.

Provides plot_series() for plotting one or more time series with annotations,
shaded recession bars, and secondary Y-axis support.

Usage:
    from chronobox.visualization import plot_series

    fig = plot_series(data, title='GDP Growth')
    fig = plot_series(
        [series1, series2],
        labels=['GDP', 'Inflation'],
        secondary_y=['Inflation'],
        recessions=[(datetime(2008,1,1), datetime(2009,6,1))],
    )
"""

from __future__ import annotations

from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray

from chronobox.visualization.themes import get_color_cycle, get_theme


def plot_series(
    data: pd.Series | pd.DataFrame | NDArray[np.float64] | list[Any],
    labels: list[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    secondary_y: list[str] | None = None,
    secondary_ylabel: str | None = None,
    annotations: list[dict[str, Any]] | None = None,
    recessions: list[tuple[Any, Any]] | None = None,
    recession_color: str = "#cccccc",
    recession_alpha: float = 0.3,
    figsize: tuple[float, float] | None = None,
    legend: bool = True,
    grid: bool = True,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot one or more time series.

    Parameters
    ----------
    data : Series, DataFrame, ndarray, or list
        Time series data. If a list, each element is a separate series.
        If a DataFrame, each column is a separate series.
    labels : list[str] or None
        Labels for each series. Inferred from column names if DataFrame.
    title : str or None
        Plot title.
    xlabel : str or None
        X-axis label.
    ylabel : str or None
        Y-axis label.
    secondary_y : list[str] or None
        List of series labels to plot on secondary Y-axis.
    secondary_ylabel : str or None
        Label for secondary Y-axis.
    annotations : list[dict] or None
        List of annotation dicts with keys: 'x', 'y', 'text', and optional
        'arrowprops', 'fontsize', 'color'.
    recessions : list[tuple] or None
        List of (start, end) tuples for shaded recession bars.
    recession_color : str
        Color for recession shading.
    recession_alpha : float
        Alpha for recession shading.
    figsize : tuple or None
        Figure size (width, height) in inches.
    legend : bool
        Whether to show legend.
    grid : bool
        Whether to show grid.
    ax : Axes or None
        Existing axes to plot on. If None, creates new figure.
    **kwargs
        Additional keyword arguments passed to plt.plot().

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    theme = get_theme()
    colors = get_color_cycle(theme)

    if figsize is None:
        figsize = (12, 5)

    # Normalize data to list of (index, values, label) tuples
    series_list = _normalize_data(data, labels)

    # Create figure and axes
    if ax is None:
        fig, ax_main = plt.subplots(1, 1, figsize=figsize)
    else:
        ax_main = ax
        fig = ax_main.get_figure()

    ax_sec = None
    secondary_y_set = set(secondary_y) if secondary_y else set()

    if secondary_y_set:
        ax_sec = ax_main.twinx()

    # Plot each series
    for i, (index, values, label) in enumerate(series_list):
        color = colors[i % len(colors)]
        target_ax = ax_sec if label in secondary_y_set else ax_main

        plot_kwargs: dict[str, Any] = {
            "color": color,
            "label": label,
            "linewidth": theme.line_width,
        }
        plot_kwargs.update(kwargs)

        if index is not None:
            target_ax.plot(index, values, **plot_kwargs)
        else:
            target_ax.plot(values, **plot_kwargs)

    # Add recession bars
    if recessions:
        for start, end in recessions:
            ax_main.axvspan(
                start, end,
                color=recession_color,
                alpha=recession_alpha,
                zorder=0,
            )

    # Add annotations
    if annotations:
        for ann in annotations:
            ax_main.annotate(
                ann.get("text", ""),
                xy=(ann["x"], ann["y"]),
                xytext=ann.get("xytext", (ann["x"], ann["y"] * 1.05)),
                arrowprops=ann.get("arrowprops", {"arrowstyle": "->", "color": theme.text_color}),
                fontsize=ann.get("fontsize", theme.tick_size),
                color=ann.get("color", theme.text_color),
            )

    # Labels and title
    if title:
        ax_main.set_title(title, fontsize=theme.title_size, fontweight="bold")
    if xlabel:
        ax_main.set_xlabel(xlabel, fontsize=theme.label_size)
    if ylabel:
        ax_main.set_ylabel(ylabel, fontsize=theme.label_size)
    if secondary_ylabel and ax_sec is not None:
        ax_sec.set_ylabel(secondary_ylabel, fontsize=theme.label_size)

    # Grid
    ax_main.grid(grid, alpha=theme.grid_alpha, color=theme.grid_color, linestyle=theme.grid_style)

    # Legend
    if legend and len(series_list) > 0:
        lines_main, labels_main = ax_main.get_legend_handles_labels()
        if ax_sec is not None:
            lines_sec, labels_sec = ax_sec.get_legend_handles_labels()
            ax_main.legend(
                lines_main + lines_sec,
                labels_main + labels_sec,
                loc="best",
                frameon=theme.legend_frame,
            )
        else:
            ax_main.legend(loc="best", frameon=theme.legend_frame)

    # Format dates if index is datetime
    if series_list and series_list[0][0] is not None:
        idx = series_list[0][0]
        if hasattr(idx, "dtype") and np.issubdtype(getattr(idx, "dtype", None), np.datetime64):
            ax_main.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
            fig.autofmt_xdate()

    fig.tight_layout()
    return fig


def _normalize_data(
    data: pd.Series | pd.DataFrame | NDArray[np.float64] | list[Any],
    labels: list[str] | None = None,
) -> list[tuple[Any, NDArray[np.float64], str]]:
    """Normalize input data to a list of (index, values, label) tuples.

    Parameters
    ----------
    data : various
        Input data.
    labels : list[str] or None
        Labels for series.

    Returns
    -------
    list of (index, values, label) tuples.
    """
    result: list[tuple[Any, NDArray[np.float64], str]] = []

    if isinstance(data, pd.DataFrame):
        for i, col in enumerate(data.columns):
            label = labels[i] if labels and i < len(labels) else str(col)
            values = data[col].to_numpy(dtype=np.float64)
            index = data.index if not isinstance(data.index, pd.RangeIndex) else None
            result.append((index, values, label))
    elif isinstance(data, pd.Series):
        label = labels[0] if labels else (data.name or "Series")
        values = data.to_numpy(dtype=np.float64)
        index = data.index if not isinstance(data.index, pd.RangeIndex) else None
        result.append((index, values, str(label)))
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            label = labels[0] if labels else "Series"
            result.append((None, data.astype(np.float64), label))
        elif data.ndim == 2:
            for i in range(data.shape[1]):
                label = labels[i] if labels and i < len(labels) else f"Series {i + 1}"
                result.append((None, data[:, i].astype(np.float64), label))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            label = labels[i] if labels and i < len(labels) else f"Series {i + 1}"
            if isinstance(item, pd.Series):
                values = item.to_numpy(dtype=np.float64)
                index = item.index if not isinstance(item.index, pd.RangeIndex) else None
                result.append((index, values, label))
            elif isinstance(item, np.ndarray):
                result.append((None, item.astype(np.float64), label))
            else:
                result.append((None, np.asarray(item, dtype=np.float64), label))
    else:
        label = labels[0] if labels else "Series"
        result.append((None, np.asarray(data, dtype=np.float64), label))

    return result
