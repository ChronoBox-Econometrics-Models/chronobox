"""Test visualization plots for structural break and stability tests.

Provides:
    - plot_cusum: CUSUM path with significance bands
    - plot_bai_perron: Series with vertical break lines
    - plot_zivot_andrews: Series with estimated break point
    - plot_recursive_coefs: Recursive coefficient estimates over time

Usage:
    from chronobox.visualization.test_plot import plot_cusum, plot_bai_perron

    fig = plot_cusum(cusum_result)
    fig = plot_bai_perron(bp_result, y)
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from chronobox.visualization.themes import get_theme


def plot_cusum(
    cusum_result: Any | None = None,
    cusum_path: NDArray[np.float64] | None = None,
    upper_band: NDArray[np.float64] | None = None,
    lower_band: NDArray[np.float64] | None = None,
    significance: float = 0.05,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    dates: Any | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot CUSUM test path with significance bands.

    Parameters
    ----------
    cusum_result : CUSUM results or None
        CUSUM test results object.
    cusum_path : ndarray or None
        CUSUM statistic path.
    upper_band : ndarray or None
        Upper significance band.
    lower_band : ndarray or None
        Lower significance band.
    significance : float
        Significance level.
    title : str or None
        Plot title.
    figsize : tuple or None
        Figure size.
    dates : array or None
        Date index.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    theme = get_theme()

    # Extract from results
    if cusum_path is None and cusum_result is not None:
        cusum_path = np.asarray(
            getattr(cusum_result, "cusum", getattr(cusum_result, "path", None)),
            dtype=np.float64,
        )
        if hasattr(cusum_result, "upper"):
            upper_band = np.asarray(cusum_result.upper, dtype=np.float64)
        if hasattr(cusum_result, "lower"):
            lower_band = np.asarray(cusum_result.lower, dtype=np.float64)

    if cusum_path is None:
        msg = "Either 'cusum_result' or 'cusum_path' must be provided."
        raise ValueError(msg)

    n = len(cusum_path)
    x = dates if dates is not None else np.arange(n)

    # Generate default bands if not provided: +-a*sqrt(n) + 2*a*(t/n)
    if upper_band is None or lower_band is None:
        a = 0.948  # 5% significance
        if significance == 0.01:
            a = 1.143
        elif significance == 0.10:
            a = 0.850
        t = np.arange(1, n + 1, dtype=np.float64)
        upper_band = a * np.sqrt(n) + 2 * a * (t / n)
        lower_band = -upper_band

    if figsize is None:
        figsize = (12, 5)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # CUSUM path
    ax.plot(x, cusum_path, color=theme.colors[0], linewidth=theme.line_width, label="CUSUM")

    # Significance bands
    ax.plot(
        x, upper_band, color=theme.negative_color,
        linewidth=1.0, linestyle="--", label=f"{int((1 - significance) * 100)}% band",
    )
    ax.plot(x, lower_band, color=theme.negative_color, linewidth=1.0, linestyle="--")
    ax.fill_between(
        x if isinstance(x, np.ndarray) else range(n),
        lower_band, upper_band,
        color=theme.negative_color, alpha=0.05,
    )

    ax.axhline(y=0, color=theme.text_color, linewidth=0.5, linestyle="-")

    if title:
        ax.set_title(title, fontsize=theme.title_size, fontweight="bold")
    else:
        ax.set_title("CUSUM Test", fontsize=theme.title_size, fontweight="bold")

    ax.set_xlabel("Observation", fontsize=theme.label_size)
    ax.set_ylabel("CUSUM", fontsize=theme.label_size)
    ax.legend(loc="upper left", fontsize=theme.tick_size, frameon=theme.legend_frame)
    ax.grid(True, alpha=theme.grid_alpha, color=theme.grid_color)

    fig.tight_layout()
    return fig


def plot_bai_perron(
    bp_result: Any | None = None,
    y: NDArray[np.float64] | None = None,
    break_dates: list[int] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    dates: Any | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot Bai-Perron structural break test results.

    Shows the original series with vertical lines at each estimated break point.

    Parameters
    ----------
    bp_result : Bai-Perron results or None
        Test results.
    y : ndarray or None
        Original time series.
    break_dates : list[int] or None
        Break point indices.
    title : str or None
        Plot title.
    figsize : tuple or None
        Figure size.
    dates : array or None
        Date index.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    theme = get_theme()

    if bp_result is not None:
        if y is None and hasattr(bp_result, "y"):
            y = np.asarray(bp_result.y, dtype=np.float64)
        if break_dates is None:
            for attr in ("breaks", "break_dates", "break_indices", "breakpoints"):
                if hasattr(bp_result, attr):
                    break_dates = list(getattr(bp_result, attr))
                    break

    if y is None:
        msg = "Series 'y' must be provided."
        raise ValueError(msg)

    if break_dates is None:
        break_dates = []

    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    x = dates if dates is not None else np.arange(n)

    if figsize is None:
        figsize = (12, 5)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(x, y, color=theme.colors[0], linewidth=theme.line_width, label="Series")

    # Vertical break lines
    for i, bp in enumerate(break_dates):
        label = "Break" if i == 0 else None
        x_val = dates[bp] if dates is not None and bp < len(dates) else bp
        ax.axvline(
            x=x_val, color=theme.negative_color,
            linewidth=1.5, linestyle="--", alpha=0.8, label=label,
        )

    # Shade different regimes
    all_breaks = [0, *list(break_dates), n - 1]
    regime_colors = [theme.colors[i % len(theme.colors)] for i in range(len(all_breaks) - 1)]
    for i in range(len(all_breaks) - 1):
        start = all_breaks[i]
        end = all_breaks[i + 1]
        x_start = dates[start] if dates is not None and start < len(dates) else start
        x_end = dates[end] if dates is not None and end < len(dates) else end
        ax.axvspan(x_start, x_end, alpha=0.05, color=regime_colors[i])

    if title:
        ax.set_title(title, fontsize=theme.title_size, fontweight="bold")
    else:
        ax.set_title(
            f"Bai-Perron Test ({len(break_dates)} breaks)",
            fontsize=theme.title_size,
            fontweight="bold",
        )

    ax.legend(loc="best", fontsize=theme.tick_size, frameon=theme.legend_frame)
    ax.grid(True, alpha=theme.grid_alpha, color=theme.grid_color)

    fig.tight_layout()
    return fig


def plot_zivot_andrews(
    za_result: Any | None = None,
    y: NDArray[np.float64] | None = None,
    break_index: int | None = None,
    za_stat: float | None = None,
    critical_values: dict[str, float] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    dates: Any | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot Zivot-Andrews unit root test with structural break.

    Shows the series with a vertical line at the estimated break point.

    Parameters
    ----------
    za_result : ZA results or None
        Test results.
    y : ndarray or None
        Original series.
    break_index : int or None
        Estimated break point index.
    za_stat : float or None
        ZA test statistic.
    critical_values : dict or None
        Critical values.
    title : str or None
        Plot title.
    figsize : tuple or None
        Figure size.
    dates : array or None
        Date index.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    theme = get_theme()

    if za_result is not None:
        if y is None and hasattr(za_result, "y"):
            y = np.asarray(za_result.y, dtype=np.float64)
        if break_index is None:
            for attr in ("break_index", "break_point", "breakpoint", "bp"):
                if hasattr(za_result, attr):
                    break_index = int(getattr(za_result, attr))
                    break
        if za_stat is None and hasattr(za_result, "statistic"):
            za_stat = float(za_result.statistic)

    if y is None:
        msg = "Series 'y' must be provided."
        raise ValueError(msg)

    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    x = dates if dates is not None else np.arange(n)

    if figsize is None:
        figsize = (12, 5)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(x, y, color=theme.colors[0], linewidth=theme.line_width, label="Series")

    if break_index is not None:
        x_break = (
            dates[break_index]
            if dates is not None and break_index < len(dates)
            else break_index
        )
        ax.axvline(
            x=x_break, color=theme.negative_color,
            linewidth=2.0, linestyle="--", alpha=0.8, label="Break",
        )

        # Annotation
        stat_text = f"Break at t={break_index}"
        if za_stat is not None:
            stat_text += f"\nZA stat = {za_stat:.3f}"
        ax.text(
            0.95, 0.05, stat_text,
            transform=ax.transAxes,
            fontsize=theme.tick_size,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": theme.background,
                  "alpha": 0.8, "edgecolor": theme.grid_color},
        )

    if title:
        ax.set_title(title, fontsize=theme.title_size, fontweight="bold")
    else:
        ax.set_title("Zivot-Andrews Test", fontsize=theme.title_size, fontweight="bold")

    ax.legend(loc="upper left", fontsize=theme.tick_size, frameon=theme.legend_frame)
    ax.grid(True, alpha=theme.grid_alpha, color=theme.grid_color)

    fig.tight_layout()
    return fig


def plot_recursive_coefs(
    recursive_results: Any | None = None,
    coefs: NDArray[np.float64] | None = None,
    coef_se: NDArray[np.float64] | None = None,
    coef_names: list[str] | None = None,
    dates: Any | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot recursive coefficient estimates over time.

    Parameters
    ----------
    recursive_results : results or None
        Recursive estimation results.
    coefs : ndarray of shape (T, n_coefs) or None
        Recursive coefficient estimates.
    coef_se : ndarray of shape (T, n_coefs) or None
        Recursive coefficient standard errors.
    coef_names : list[str] or None
        Coefficient names.
    dates : array or None
        Date index.
    title : str or None
        Overall title.
    figsize : tuple or None
        Figure size.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    theme = get_theme()

    if coefs is None and recursive_results is not None:
        for attr in ("recursive_coefs", "coefs", "params"):
            if hasattr(recursive_results, attr):
                coefs = np.asarray(getattr(recursive_results, attr), dtype=np.float64)
                break
        for attr in ("recursive_se", "coef_se", "bse"):
            if hasattr(recursive_results, attr):
                coef_se = np.asarray(getattr(recursive_results, attr), dtype=np.float64)
                break
        for attr in ("coef_names", "param_names", "names"):
            if hasattr(recursive_results, attr):
                coef_names = list(getattr(recursive_results, attr))
                break
        for attr in ("dates", "index"):
            if hasattr(recursive_results, attr):
                dates = getattr(recursive_results, attr)
                break

    if coefs is None:
        msg = "Either 'recursive_results' or 'coefs' must be provided."
        raise ValueError(msg)

    coefs = np.asarray(coefs, dtype=np.float64)
    if coefs.ndim == 1:
        coefs = coefs[:, np.newaxis]

    n_obs, n_coefs = coefs.shape

    if coef_names is None:
        coef_names = [f"Coef {i + 1}" for i in range(n_coefs)]

    x = dates if dates is not None else np.arange(n_obs)

    n_cols = min(2, n_coefs)
    n_rows = int(np.ceil(n_coefs / n_cols))

    if figsize is None:
        figsize = (6 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for idx in range(n_coefs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        ax.plot(
            x, coefs[:, idx],
            color=theme.colors[0],
            linewidth=theme.line_width,
        )

        if coef_se is not None:
            se = np.asarray(coef_se, dtype=np.float64)
            if se.ndim == 2 and idx < se.shape[1]:
                upper = coefs[:, idx] + 1.96 * se[:, idx]
                lower = coefs[:, idx] - 1.96 * se[:, idx]
                ax.fill_between(
                    x if isinstance(x, np.ndarray) else range(n_obs),
                    lower, upper,
                    alpha=0.15, color=theme.colors[0],
                )

        ax.set_title(
            coef_names[idx] if idx < len(coef_names) else f"Coef {idx + 1}",
            fontsize=theme.label_size,
            fontweight="bold",
        )
        ax.grid(True, alpha=theme.grid_alpha * 0.5)

    # Hide empty
    for idx in range(n_coefs, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=theme.title_size + 2, fontweight="bold", y=1.02)
    else:
        fig.suptitle(
            "Recursive Coefficients",
            fontsize=theme.title_size + 2,
            fontweight="bold",
            y=1.02,
        )

    fig.tight_layout()
    return fig
