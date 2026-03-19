"""Impulse Response Function (IRF) plots for VAR/SVAR models.

Provides plot_irf() which generates a KxK grid (or selected subset) showing
impulse responses with confidence bands.

Usage:
    from chronobox.visualization.irf_plot import plot_irf

    fig = plot_irf(irf_results)
    fig = plot_irf(irf_results, impulse='shock_1', response='gdp')
    fig = plot_irf(irf_results, cumulative=True)
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from chronobox.visualization.themes import get_confidence_colors, get_theme


def plot_irf(
    irf_results: Any | None = None,
    irf_array: NDArray[np.float64] | None = None,
    irf_lower: NDArray[np.float64] | None = None,
    irf_upper: NDArray[np.float64] | None = None,
    var_names: list[str] | None = None,
    impulse: str | int | None = None,
    response: str | int | None = None,
    cumulative: bool = False,
    sigs: float = 0.95,
    periods: int | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot Impulse Response Functions.

    Generates a grid of IRF plots. Each cell (i, j) shows the response
    of variable i to a shock in variable j.

    Parameters
    ----------
    irf_results : IRF results object or None
        IRF results with arrays and variable names.
        If None, irf_array must be provided.
    irf_array : ndarray of shape (H, K, K) or None
        IRF values. irf_array[h, i, j] = response of var i to shock j at horizon h.
    irf_lower : ndarray of shape (H, K, K) or None
        Lower confidence band.
    irf_upper : ndarray of shape (H, K, K) or None
        Upper confidence band.
    var_names : list[str] or None
        Variable names. If None, uses generic names.
    impulse : str or int or None
        Plot only IRFs for this impulse (shock) variable. If None, all.
    response : str or int or None
        Plot only IRFs for this response variable. If None, all.
    cumulative : bool
        Whether to plot cumulative IRFs.
    sigs : float
        Significance level for confidence bands (default 0.95).
    periods : int or None
        Number of horizons to plot. If None, uses all available.
    title : str or None
        Overall figure title.
    figsize : tuple or None
        Figure size. Auto-computed if None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Figure
        Matplotlib figure with IRF grid.
    """
    theme = get_theme()
    ci_colors = get_confidence_colors(theme)

    # Extract data from results object
    if irf_array is None and irf_results is not None:
        irf_array, irf_lower, irf_upper, var_names = _extract_irf_data(
            irf_results, cumulative=cumulative
        )

    if irf_array is None:
        msg = "Either 'irf_results' or 'irf_array' must be provided."
        raise ValueError(msg)

    irf_array = np.asarray(irf_array, dtype=np.float64)
    n_horizons, k_resp, k_imp = irf_array.shape

    if periods is not None:
        n_horizons = min(n_horizons, periods)
        irf_array = irf_array[:n_horizons]
        if irf_lower is not None:
            irf_lower = irf_lower[:n_horizons]
        if irf_upper is not None:
            irf_upper = irf_upper[:n_horizons]

    if var_names is None:
        var_names = [f"Var {i + 1}" for i in range(max(k_resp, k_imp))]

    # Apply cumulative sum if requested
    if cumulative and irf_results is None:
        irf_array = np.cumsum(irf_array, axis=0)
        if irf_lower is not None:
            irf_lower = np.cumsum(irf_lower, axis=0)
        if irf_upper is not None:
            irf_upper = np.cumsum(irf_upper, axis=0)

    # Determine which IRFs to plot
    imp_indices, resp_indices = _get_indices(
        impulse, response, var_names, k_imp, k_resp
    )

    n_rows = len(resp_indices)
    n_cols = len(imp_indices)

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    horizons = np.arange(n_horizons)

    for row_idx, resp_idx in enumerate(resp_indices):
        for col_idx, imp_idx in enumerate(imp_indices):
            ax = axes[row_idx, col_idx]

            # Plot IRF line
            irf_line = irf_array[:, resp_idx, imp_idx]
            ax.plot(
                horizons, irf_line,
                color=theme.colors[0],
                linewidth=theme.line_width,
                zorder=5,
            )

            # Plot confidence bands
            if irf_lower is not None and irf_upper is not None:
                lower = irf_lower[:, resp_idx, imp_idx]
                upper = irf_upper[:, resp_idx, imp_idx]
                ax.fill_between(
                    horizons, lower, upper,
                    color=ci_colors[1] if len(ci_colors) > 1 else ci_colors[0],
                    alpha=0.25,
                    zorder=2,
                )

            # Zero line
            ax.axhline(
                y=0, color=theme.text_color,
                linewidth=0.8, linestyle="--", alpha=0.5,
            )

            # Title for each subplot
            resp_name = var_names[resp_idx] if resp_idx < len(var_names) else f"Var {resp_idx + 1}"
            imp_name = var_names[imp_idx] if imp_idx < len(var_names) else f"Var {imp_idx + 1}"

            if row_idx == 0:
                ax.set_title(
                    f"Shock: {imp_name}",
                    fontsize=theme.label_size,
                    fontweight="bold",
                )
            if col_idx == 0:
                ax.set_ylabel(
                    f"Response: {resp_name}",
                    fontsize=theme.tick_size,
                )

            ax.grid(True, alpha=theme.grid_alpha * 0.5, color=theme.grid_color)
            ax.set_xlabel("Horizon", fontsize=theme.tick_size - 1)

    if title:
        fig.suptitle(title, fontsize=theme.title_size + 2, fontweight="bold", y=1.02)
    else:
        irf_type = "Cumulative IRF" if cumulative else "Impulse Response Functions"
        fig.suptitle(irf_type, fontsize=theme.title_size + 2, fontweight="bold", y=1.02)

    fig.tight_layout()
    return fig


def _extract_irf_data(
    irf_results: Any,
    cumulative: bool = False,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    list[str] | None,
]:
    """Extract IRF data from a results object.

    Parameters
    ----------
    irf_results : Any
        IRF results object.
    cumulative : bool
        Whether to extract cumulative IRFs.

    Returns
    -------
    tuple of (irf_array, lower, upper, var_names)
    """
    irf_array = None
    irf_lower = None
    irf_upper = None
    var_names = None

    # Try cumulative first if requested
    if cumulative:
        for attr in ("cum_effects", "irfs_cumulative", "cumulative_irf"):
            if hasattr(irf_results, attr):
                irf_array = np.asarray(getattr(irf_results, attr), dtype=np.float64)
                break

    # Regular IRF
    if irf_array is None:
        for attr in ("irfs", "irf", "irf_array", "effects"):
            if hasattr(irf_results, attr):
                val = getattr(irf_results, attr)
                if callable(val):
                    val = val()
                irf_array = np.asarray(val, dtype=np.float64)
                break

    # Confidence bands
    for attr_l, attr_u in [
        ("lower", "upper"),
        ("ci_lower", "ci_upper"),
        ("irf_lower", "irf_upper"),
        ("conf_int_lower", "conf_int_upper"),
    ]:
        if hasattr(irf_results, attr_l) and hasattr(irf_results, attr_u):
            irf_lower = np.asarray(getattr(irf_results, attr_l), dtype=np.float64)
            irf_upper = np.asarray(getattr(irf_results, attr_u), dtype=np.float64)
            break

    # Variable names
    for attr in ("names", "var_names", "variable_names", "columns"):
        if hasattr(irf_results, attr):
            var_names = list(getattr(irf_results, attr))
            break

    if irf_array is None:
        msg = "Cannot extract IRF data from results object."
        raise AttributeError(msg)

    return irf_array, irf_lower, irf_upper, var_names


def _get_indices(
    impulse: str | int | None,
    response: str | int | None,
    var_names: list[str],
    k_imp: int,
    k_resp: int,
) -> tuple[list[int], list[int]]:
    """Convert impulse/response selection to index lists.

    Parameters
    ----------
    impulse : str, int, or None
        Impulse selection.
    response : str, int, or None
        Response selection.
    var_names : list[str]
        Variable names.
    k_imp : int
        Number of impulse variables.
    k_resp : int
        Number of response variables.

    Returns
    -------
    tuple of (imp_indices, resp_indices)
    """
    if impulse is not None:
        imp_indices = [var_names.index(impulse)] if isinstance(impulse, str) else [impulse]
    else:
        imp_indices = list(range(k_imp))

    if response is not None:
        resp_indices = [var_names.index(response)] if isinstance(response, str) else [response]
    else:
        resp_indices = list(range(k_resp))

    return imp_indices, resp_indices
