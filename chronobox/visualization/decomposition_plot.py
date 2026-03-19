"""Decomposition plots for time series components.

Provides plot_decomposition() which generates vertical panels showing
observed, trend, seasonal, cycle, and remainder components.

Supports STL, classical, ETS decomposition, and filter-based decomposition
(HP, BK, CF, Hamilton, BN).

Usage:
    from chronobox.visualization.decomposition_plot import plot_decomposition

    fig = plot_decomposition(results)
    fig = plot_decomposition(results, which=['trend', 'seasonal'])
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from chronobox.visualization.themes import get_color_cycle, get_theme

# Standard component names and labels
_COMPONENT_LABELS: dict[str, str] = {
    "observed": "Observed",
    "trend": "Trend",
    "seasonal": "Seasonal",
    "cycle": "Cycle",
    "remainder": "Remainder",
    "irregular": "Irregular",
    "resid": "Residual",
    "level": "Level",
}


def plot_decomposition(
    results: Any | None = None,
    components: dict[str, NDArray[np.float64]] | None = None,
    which: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    share_x: bool = True,
    dates: NDArray[np.float64] | Any | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot time series decomposition as vertical panels.

    Creates a vertically stacked set of subplots, one for each component
    (observed, trend, seasonal, cycle, remainder).

    Parameters
    ----------
    results : decomposition results object or None
        Results with component attributes (trend, seasonal, resid, etc.).
        If None, `components` dict must be provided.
    components : dict[str, ndarray] or None
        Dictionary mapping component names to arrays. Used if results is None.
        Keys should be from: 'observed', 'trend', 'seasonal', 'cycle', 'remainder'.
    which : list[str] or None
        Which components to plot. If None, plots all available.
    title : str or None
        Overall figure title.
    figsize : tuple or None
        Figure size. If None, auto-computed based on number of panels.
    share_x : bool
        Whether to share x-axis across panels.
    dates : array or None
        Date index for x-axis. If None, uses integer index.
    **kwargs
        Additional keyword arguments passed to plot().

    Returns
    -------
    Figure
        Matplotlib figure with vertical panels.

    Raises
    ------
    ValueError
        If no components can be extracted.
    """
    theme = get_theme()
    colors = get_color_cycle(theme)

    # Extract components
    if components is None and results is not None:
        components = _extract_components(results)
    if components is None or len(components) == 0:
        msg = "No components found. Provide 'results' or 'components' dict."
        raise ValueError(msg)

    # Extract dates from results if available
    if dates is None and results is not None:
        dates = _extract_dates(results)

    # Filter components
    if which is not None:
        components = {k: v for k, v in components.items() if k in which}

    n_panels = len(components)
    if n_panels == 0:
        msg = "No components to plot after filtering."
        raise ValueError(msg)

    if figsize is None:
        figsize = (12, 3 * n_panels)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=figsize,
        sharex=share_x,
        squeeze=False,
    )

    for i, (name, values) in enumerate(components.items()):
        ax = axes[i, 0]
        values = np.asarray(values, dtype=np.float64)
        label = _COMPONENT_LABELS.get(name, name.capitalize())
        color = colors[i % len(colors)]

        x = dates if dates is not None else np.arange(len(values))

        if name in ("seasonal", "remainder", "irregular", "resid", "cycle"):
            # Bar-like for oscillating components
            ax.plot(x, values, color=color, linewidth=theme.line_width * 0.8, **kwargs)
            ax.axhline(y=0, color=theme.text_color, linewidth=0.5, linestyle="-")
            # Fill between for visual effect
            ax.fill_between(
                x if isinstance(x, np.ndarray) else range(len(values)),
                0, values,
                color=color, alpha=0.15,
            )
        else:
            # Line for trend, observed, level
            ax.plot(x, values, color=color, linewidth=theme.line_width, **kwargs)

        ax.set_ylabel(label, fontsize=theme.label_size, fontweight="bold")
        ax.grid(True, alpha=theme.grid_alpha * 0.5, color=theme.grid_color)

        # Remove x-tick labels except on last panel
        if i < n_panels - 1 and share_x:
            ax.tick_params(labelbottom=False)

    if title:
        fig.suptitle(title, fontsize=theme.title_size + 2, fontweight="bold", y=1.01)
    else:
        fig.suptitle("Time Series Decomposition", fontsize=theme.title_size + 2,
                      fontweight="bold", y=1.01)

    fig.tight_layout()
    return fig


def _extract_components(results: Any) -> dict[str, NDArray[np.float64]]:
    """Extract decomposition components from a results object.

    Parameters
    ----------
    results : Any
        Results object with component attributes.

    Returns
    -------
    dict[str, ndarray]
        Dictionary of component names to arrays.
    """
    components: dict[str, NDArray[np.float64]] = {}

    # Common attribute names for different decomposition methods
    attr_map = {
        "observed": ["observed", "endog", "y", "data"],
        "trend": ["trend", "trend_component", "level"],
        "seasonal": ["seasonal", "seasonal_component"],
        "cycle": ["cycle", "cyclical", "cycle_component"],
        "remainder": ["resid", "residual", "remainder", "irregular"],
    }

    for comp_name, attr_names in attr_map.items():
        for attr in attr_names:
            if hasattr(results, attr):
                val = getattr(results, attr)
                if val is not None:
                    arr = np.asarray(val, dtype=np.float64).ravel()
                    if len(arr) > 0 and not np.all(np.isnan(arr)):
                        components[comp_name] = arr
                        break

    return components


def _extract_dates(results: Any) -> Any | None:
    """Try to extract date index from results.

    Parameters
    ----------
    results : Any
        Results object.

    Returns
    -------
    dates or None
    """
    for attr in ("dates", "index", "observed"):
        if hasattr(results, attr):
            val = getattr(results, attr)
            if hasattr(val, "index"):
                return val.index
    return None
