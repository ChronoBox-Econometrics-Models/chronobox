"""
Plot helpers for filters examples.

Provides convenience plotting functions for trend/cycle decomposition,
bandpass filter comparison, and spillover heatmaps.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_trend_cycle(
    dates: pd.Series,
    observed: np.ndarray,
    trend: np.ndarray,
    cycle: np.ndarray,
    title: str = "Trend-Cycle Decomposition",
    figsize: tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot observed series with trend overlay and separate cycle panel.

    Parameters
    ----------
    dates : pd.Series
        Date index.
    observed : np.ndarray
        Observed (raw) series.
    trend : np.ndarray
        Estimated trend component.
    cycle : np.ndarray
        Estimated cyclical component.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(dates, observed, label="Observed", alpha=0.7, linewidth=0.8)
    axes[0].plot(dates, trend, label="Trend", linewidth=2)
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dates, cycle, label="Cycle", color="tab:red", linewidth=1)
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_title("Cyclical Component")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_bandpass_comparison(
    dates: pd.Series,
    cycles: dict[str, np.ndarray],
    title: str = "Bandpass Filter Comparison",
    figsize: tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare cyclical components from multiple bandpass filters.

    Parameters
    ----------
    dates : pd.Series
        Date index.
    cycles : dict
        Mapping of filter name to cyclical component array.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, cycle in cycles.items():
        valid_idx = ~np.isnan(cycle) if hasattr(cycle, "__len__") else slice(None)
        ax.plot(dates[valid_idx], np.asarray(cycle)[valid_idx], label=name, linewidth=1.2)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_spillover_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str = "Spillover Table",
    figsize: tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a spillover/connectedness matrix as a heatmap.

    Parameters
    ----------
    matrix : np.ndarray
        Square spillover matrix (k x k).
    labels : list of str
        Variable labels.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=9)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
