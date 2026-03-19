"""Historical Decomposition plots for SVAR models.

Provides plot_hd() which generates stacked bars by shock with the
observed series as an overlay line.

Usage:
    from chronobox.visualization.hd_plot import plot_hd

    fig = plot_hd(hd_results)
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from chronobox.visualization.themes import get_color_cycle, get_theme


def plot_hd(
    hd_results: Any | None = None,
    hd_array: NDArray[np.float64] | None = None,
    observed: NDArray[np.float64] | None = None,
    var_names: list[str] | None = None,
    shock_names: list[str] | None = None,
    variable: str | int | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    dates: Any | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot Historical Decomposition.

    Shows stacked bars representing the contribution of each structural shock
    to the observed variable, with the actual observed series overlaid as a line.

    Parameters
    ----------
    hd_results : HD results object or None
        Historical decomposition results.
    hd_array : ndarray of shape (T, K_var, K_shock) or None
        HD contributions. hd_array[t, i, j] = contribution of shock j
        to variable i at time t.
    observed : ndarray of shape (T,) or (T, K) or None
        Observed series (deviations from baseline).
    var_names : list[str] or None
        Response variable names.
    shock_names : list[str] or None
        Shock names.
    variable : str or int or None
        Which response variable to plot. If None, plots all.
    title : str or None
        Overall figure title.
    figsize : tuple or None
        Figure size.
    dates : array or None
        Date index for x-axis.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Figure
        Matplotlib figure with HD stacked bars.
    """
    theme = get_theme()
    colors = get_color_cycle(theme)

    # Extract data
    if hd_array is None and hd_results is not None:
        hd_array, observed, var_names, shock_names = _extract_hd_data(hd_results)

    if hd_array is None:
        msg = "Either 'hd_results' or 'hd_array' must be provided."
        raise ValueError(msg)

    hd_array = np.asarray(hd_array, dtype=np.float64)

    # Handle 2D case: (T, K_shock) for single variable
    if hd_array.ndim == 2:
        hd_array = hd_array[:, np.newaxis, :]

    n_obs, k_var, k_shock = hd_array.shape

    if var_names is None:
        var_names = [f"Var {i + 1}" for i in range(k_var)]
    if shock_names is None:
        shock_names = [f"Shock {j + 1}" for j in range(k_shock)]

    # Determine variables to plot
    if variable is not None:
        var_indices = (
            [var_names.index(variable)] if isinstance(variable, str) else [variable]
        )
    else:
        var_indices = list(range(k_var))

    n_panels = len(var_indices)
    if figsize is None:
        figsize = (14, 4 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, squeeze=False)

    x = dates if dates is not None else np.arange(n_obs)

    for panel_idx, var_idx in enumerate(var_indices):
        ax = axes[panel_idx, 0]
        var_name = var_names[var_idx] if var_idx < len(var_names) else f"Var {var_idx + 1}"

        # Separate positive and negative contributions for proper stacking
        contributions = hd_array[:, var_idx, :]  # (n_obs, k_shock)

        # Stack bars
        pos_bottom = np.zeros(n_obs)
        neg_bottom = np.zeros(n_obs)

        for j in range(k_shock):
            values = contributions[:, j]
            shock_label = shock_names[j] if j < len(shock_names) else f"Shock {j + 1}"
            color = colors[j % len(colors)]

            pos_vals = np.where(values > 0, values, 0)
            neg_vals = np.where(values < 0, values, 0)

            ax.bar(
                x, pos_vals, bottom=pos_bottom,
                color=color, alpha=0.8, label=shock_label,
                width=0.8, edgecolor="none",
            )
            ax.bar(
                x, neg_vals, bottom=neg_bottom,
                color=color, alpha=0.8,
                width=0.8, edgecolor="none",
            )

            pos_bottom += pos_vals
            neg_bottom += neg_vals

        # Overlay observed line
        if observed is not None:
            obs = np.asarray(observed, dtype=np.float64)
            obs_line = obs[:, var_idx] if obs.ndim == 2 else obs

            ax.plot(
                x, obs_line,
                color="black", linewidth=theme.line_width * 1.2,
                label="Observed", zorder=10,
            )

        ax.axhline(y=0, color=theme.text_color, linewidth=0.5, linestyle="-")
        ax.set_ylabel(var_name, fontsize=theme.label_size, fontweight="bold")
        ax.set_title(
            f"Historical Decomposition: {var_name}",
            fontsize=theme.label_size,
            fontweight="bold",
        )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=theme.tick_size - 1,
            frameon=theme.legend_frame,
        )
        ax.grid(True, alpha=theme.grid_alpha * 0.3, axis="y")

    if title:
        fig.suptitle(title, fontsize=theme.title_size + 2, fontweight="bold", y=1.02)

    fig.tight_layout()
    return fig


def _extract_hd_data(
    hd_results: Any,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64] | None,
    list[str] | None,
    list[str] | None,
]:
    """Extract HD data from results object.

    Parameters
    ----------
    hd_results : Any
        HD results.

    Returns
    -------
    tuple of (hd_array, observed, var_names, shock_names)
    """
    hd_array = None
    observed = None
    var_names = None
    shock_names = None

    for attr in ("hd", "historical_decomposition", "contributions", "decomposition"):
        if hasattr(hd_results, attr):
            hd_array = np.asarray(getattr(hd_results, attr), dtype=np.float64)
            break

    for attr in ("observed", "y", "endog", "actual"):
        if hasattr(hd_results, attr):
            observed = np.asarray(getattr(hd_results, attr), dtype=np.float64)
            break

    for attr in ("var_names", "variable_names", "names"):
        if hasattr(hd_results, attr):
            var_names = list(getattr(hd_results, attr))
            break

    for attr in ("shock_names", "shocks"):
        if hasattr(hd_results, attr):
            shock_names = list(getattr(hd_results, attr))
            break

    if hd_array is None:
        msg = "Cannot extract HD data from results."
        raise AttributeError(msg)

    return hd_array, observed, var_names, shock_names
