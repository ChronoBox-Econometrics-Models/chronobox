"""Time-varying coefficient plots for TVP-VAR models.

Provides plot_tvp_coefs() which shows estimated coefficients over time
with confidence bands and OLS constant reference lines.

Usage:
    from chronobox.visualization.coef_plot import plot_tvp_coefs

    fig = plot_tvp_coefs(tvpvar_results)
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from chronobox.visualization.themes import get_confidence_colors, get_theme


def plot_tvp_coefs(
    tvp_results: Any | None = None,
    coefs: NDArray[np.float64] | None = None,
    coef_lower: NDArray[np.float64] | None = None,
    coef_upper: NDArray[np.float64] | None = None,
    ols_coefs: NDArray[np.float64] | None = None,
    coef_names: list[str] | None = None,
    dates: Any | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    max_coefs: int | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot time-varying parameter coefficients.

    Creates one subplot per coefficient showing the filtered/smoothed
    estimate over time with confidence bands, and a horizontal dashed
    line for the OLS constant estimate.

    Parameters
    ----------
    tvp_results : TVP-VAR results or None
        TVP-VAR results object.
    coefs : ndarray of shape (T, n_coefs) or None
        Time-varying coefficient estimates.
    coef_lower : ndarray of shape (T, n_coefs) or None
        Lower confidence band.
    coef_upper : ndarray of shape (T, n_coefs) or None
        Upper confidence band.
    ols_coefs : ndarray of shape (n_coefs,) or None
        OLS constant coefficient estimates for reference.
    coef_names : list[str] or None
        Coefficient names.
    dates : array or None
        Date index.
    title : str or None
        Overall title.
    figsize : tuple or None
        Figure size.
    max_coefs : int or None
        Maximum number of coefficients to plot.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    theme = get_theme()
    ci_colors = get_confidence_colors(theme)

    # Extract data
    if coefs is None and tvp_results is not None:
        coefs, coef_lower, coef_upper, ols_coefs, coef_names, dates = _extract_tvp_data(
            tvp_results
        )

    if coefs is None:
        msg = "Either 'tvp_results' or 'coefs' array must be provided."
        raise ValueError(msg)

    coefs = np.asarray(coefs, dtype=np.float64)
    if coefs.ndim == 1:
        coefs = coefs[:, np.newaxis]

    n_obs, n_coefs = coefs.shape

    if max_coefs is not None:
        n_coefs = min(n_coefs, max_coefs)

    if coef_names is None:
        coef_names = [f"Coef {i + 1}" for i in range(n_coefs)]

    x = dates if dates is not None else np.arange(n_obs)

    n_cols = min(3, n_coefs)
    n_rows = int(np.ceil(n_coefs / n_cols))

    if figsize is None:
        figsize = (5 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for idx in range(n_coefs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Plot TVP estimate
        ax.plot(
            x, coefs[:, idx],
            color=theme.colors[0],
            linewidth=theme.line_width,
            label="TVP",
            zorder=5,
        )

        # Confidence bands
        if coef_lower is not None and coef_upper is not None:
            lower = np.asarray(coef_lower, dtype=np.float64)
            upper = np.asarray(coef_upper, dtype=np.float64)
            if lower.ndim == 2 and upper.ndim == 2:
                ax.fill_between(
                    x if isinstance(x, np.ndarray) else range(n_obs),
                    lower[:, idx], upper[:, idx],
                    color=ci_colors[1] if len(ci_colors) > 1 else ci_colors[0],
                    alpha=0.2,
                    label="CI",
                    zorder=2,
                )

        # OLS reference
        if ols_coefs is not None:
            ols_vals = np.asarray(ols_coefs, dtype=np.float64)
            if idx < len(ols_vals):
                ax.axhline(
                    y=ols_vals[idx],
                    color=theme.negative_color,
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.7,
                    label="OLS",
                    zorder=4,
                )

        ax.set_title(
            coef_names[idx] if idx < len(coef_names) else f"Coef {idx + 1}",
            fontsize=theme.label_size,
            fontweight="bold",
        )
        ax.grid(True, alpha=theme.grid_alpha * 0.5, color=theme.grid_color)
        if idx == 0:
            ax.legend(fontsize=theme.tick_size - 2, loc="best", frameon=theme.legend_frame)

    # Hide empty subplots
    for idx in range(n_coefs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=theme.title_size + 2, fontweight="bold", y=1.02)
    else:
        fig.suptitle(
            "Time-Varying Coefficients",
            fontsize=theme.title_size + 2,
            fontweight="bold",
            y=1.02,
        )

    fig.tight_layout()
    return fig


def _extract_tvp_data(
    tvp_results: Any,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    list[str] | None,
    Any,
]:
    """Extract TVP data from results.

    Parameters
    ----------
    tvp_results : Any
        TVP-VAR results.

    Returns
    -------
    tuple of (coefs, lower, upper, ols_coefs, names, dates)
    """
    coefs = None
    lower = None
    upper = None
    ols_coefs = None
    names = None
    dates = None

    for attr in ("smoothed_state", "filtered_state", "coefs", "coefficients"):
        if hasattr(tvp_results, attr):
            coefs = np.asarray(getattr(tvp_results, attr), dtype=np.float64)
            break

    for attr_l, attr_u in [("conf_int_lower", "conf_int_upper"), ("lower", "upper")]:
        if hasattr(tvp_results, attr_l) and hasattr(tvp_results, attr_u):
            lower = np.asarray(getattr(tvp_results, attr_l), dtype=np.float64)
            upper = np.asarray(getattr(tvp_results, attr_u), dtype=np.float64)
            break

    for attr in ("ols_coefs", "ols_params", "constant_coefs"):
        if hasattr(tvp_results, attr):
            ols_coefs = np.asarray(getattr(tvp_results, attr), dtype=np.float64)
            break

    for attr in ("coef_names", "param_names", "names"):
        if hasattr(tvp_results, attr):
            names = list(getattr(tvp_results, attr))
            break

    for attr in ("dates", "index"):
        if hasattr(tvp_results, attr):
            dates = getattr(tvp_results, attr)
            break

    if coefs is None:
        msg = "Cannot extract TVP coefficient data."
        raise AttributeError(msg)

    return coefs, lower, upper, ols_coefs, names, dates
