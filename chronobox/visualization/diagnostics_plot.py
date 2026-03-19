"""Diagnostic plots for time series model residuals.

Provides plot_diagnostics() which generates a 2x2 panel:
    - Top-left: Standardized residuals vs time
    - Top-right: ACF and PACF of residuals
    - Bottom-left: QQ-plot against Normal distribution
    - Bottom-right: Histogram + Normal curve + JB statistic

Usage:
    from chronobox.visualization import plot_diagnostics

    results = model.fit(data)
    fig = plot_diagnostics(results)
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import stats as sp_stats

from chronobox.visualization.themes import get_theme


def plot_diagnostics(
    results: Any | None = None,
    residuals: NDArray[np.float64] | None = None,
    lags: int = 20,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot model diagnostic panel (2x2).

    Creates a 2x2 grid with:
    - Standardized residuals over time
    - ACF/PACF of residuals
    - QQ-plot
    - Histogram with Normal overlay and Jarque-Bera test

    Parameters
    ----------
    results : model results object or None
        Model results with a `resid` attribute. If None, `residuals` must be provided.
    residuals : ndarray or None
        Raw residuals. Used if results is None.
    lags : int
        Number of lags for ACF/PACF.
    figsize : tuple or None
        Figure size.
    title : str or None
        Overall figure title.

    Returns
    -------
    Figure
        Matplotlib figure with 4 subplots.

    Raises
    ------
    ValueError
        If neither results nor residuals are provided.
    """
    theme = get_theme()

    if residuals is None and results is not None:
        residuals = _extract_residuals(results)
    if residuals is None:
        msg = "Either 'results' with a .resid attribute or 'residuals' array must be provided."
        raise ValueError(msg)

    residuals = np.asarray(residuals, dtype=np.float64)
    # Remove NaNs
    mask = ~np.isnan(residuals)
    residuals = residuals[mask]

    if figsize is None:
        figsize = (12, 10)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Standardize residuals
    std_resid = (residuals - np.mean(residuals)) / np.std(residuals, ddof=1)

    # Panel 1: Standardized Residuals
    _plot_standardized_residuals(axes[0, 0], std_resid, theme)

    # Panel 2: ACF / PACF
    _plot_acf_pacf(axes[0, 1], residuals, lags, theme)

    # Panel 3: QQ-plot
    _plot_qqplot(axes[1, 0], std_resid, theme)

    # Panel 4: Histogram + Normal + JB
    _plot_histogram(axes[1, 1], std_resid, theme)

    if title:
        fig.suptitle(title, fontsize=theme.title_size + 2, fontweight="bold", y=1.02)
    else:
        fig.suptitle("Model Diagnostics", fontsize=theme.title_size + 2, fontweight="bold", y=1.02)

    fig.tight_layout()
    return fig


def _extract_residuals(results: Any) -> NDArray[np.float64]:
    """Extract residuals from a results object.

    Parameters
    ----------
    results : Any
        Results object with resid, residuals, or resids attribute.

    Returns
    -------
    ndarray
        Residuals array.
    """
    for attr in ("resid", "residuals", "resids"):
        if hasattr(results, attr):
            r = getattr(results, attr)
            if callable(r):
                r = r()
            return np.asarray(r, dtype=np.float64)
    msg = "Results object has no 'resid', 'residuals', or 'resids' attribute."
    raise AttributeError(msg)


def _plot_standardized_residuals(
    ax: plt.Axes,
    std_resid: NDArray[np.float64],
    theme: Any,
) -> None:
    """Plot standardized residuals over time.

    Parameters
    ----------
    ax : Axes
        Target axes.
    std_resid : ndarray
        Standardized residuals.
    theme : ThemeConfig
        Current theme.
    """
    n = len(std_resid)
    ax.scatter(
        range(n), std_resid,
        s=8, color=theme.colors[0], alpha=0.7, zorder=3,
    )
    ax.axhline(y=0, color=theme.text_color, linewidth=0.8, linestyle="-")
    ax.axhline(y=2, color=theme.negative_color, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(y=-2, color=theme.negative_color, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.fill_between(
        range(n), -2, 2,
        color=theme.colors[0], alpha=0.05,
    )
    ax.set_title("Standardized Residuals", fontsize=theme.label_size, fontweight="bold")
    ax.set_xlabel("Observation", fontsize=theme.tick_size)
    ax.set_ylabel("Std. Residual", fontsize=theme.tick_size)
    ax.grid(True, alpha=theme.grid_alpha * 0.5)


def _plot_acf_pacf(
    ax: plt.Axes,
    residuals: NDArray[np.float64],
    lags: int,
    theme: Any,
) -> None:
    """Plot ACF and PACF on the same axes.

    Parameters
    ----------
    ax : Axes
        Target axes.
    residuals : ndarray
        Raw residuals.
    lags : int
        Number of lags.
    theme : ThemeConfig
        Current theme.
    """
    n = len(residuals)
    lags = min(lags, n // 2 - 1)

    # Compute ACF
    acf_vals = _compute_acf(residuals, lags)

    # Compute PACF (Durbin-Levinson)
    pacf_vals = _compute_pacf(residuals, lags)

    # Significance bands
    sig_bound = 1.96 / np.sqrt(n)

    lag_range = np.arange(1, lags + 1)
    width = 0.35

    # ACF bars
    ax.bar(
        lag_range - width / 2, acf_vals[1:],
        width=width, color=theme.colors[0], alpha=0.7, label="ACF",
    )
    # PACF bars
    ax.bar(
        lag_range + width / 2, pacf_vals[1:],
        width=width, color=theme.colors[1] if len(theme.colors) > 1 else theme.colors[0],
        alpha=0.7, label="PACF",
    )

    # Significance lines
    ax.axhline(y=sig_bound, color=theme.negative_color, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(y=-sig_bound, color=theme.negative_color, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(y=0, color=theme.text_color, linewidth=0.5)

    ax.set_title("ACF / PACF of Residuals", fontsize=theme.label_size, fontweight="bold")
    ax.set_xlabel("Lag", fontsize=theme.tick_size)
    ax.legend(loc="upper right", fontsize=theme.tick_size - 1)
    ax.grid(True, alpha=theme.grid_alpha * 0.5)


def _plot_qqplot(
    ax: plt.Axes,
    std_resid: NDArray[np.float64],
    theme: Any,
) -> None:
    """Plot QQ-plot against Normal distribution.

    Parameters
    ----------
    ax : Axes
        Target axes.
    std_resid : ndarray
        Standardized residuals.
    theme : ThemeConfig
        Current theme.
    """
    sorted_resid = np.sort(std_resid)
    n = len(sorted_resid)
    theoretical = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    ax.scatter(
        theoretical, sorted_resid,
        s=15, color=theme.colors[0], alpha=0.7, edgecolors="none", zorder=3,
    )

    # Reference line (45-degree)
    lim_min = min(theoretical.min(), sorted_resid.min()) - 0.5
    lim_max = max(theoretical.max(), sorted_resid.max()) + 0.5
    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max],
        color=theme.negative_color, linewidth=1.0, linestyle="--", alpha=0.8,
    )

    ax.set_title("Normal Q-Q Plot", fontsize=theme.label_size, fontweight="bold")
    ax.set_xlabel("Theoretical Quantiles", fontsize=theme.tick_size)
    ax.set_ylabel("Sample Quantiles", fontsize=theme.tick_size)
    ax.grid(True, alpha=theme.grid_alpha * 0.5)


def _plot_histogram(
    ax: plt.Axes,
    std_resid: NDArray[np.float64],
    theme: Any,
) -> None:
    """Plot histogram of residuals with Normal overlay and JB statistic.

    Parameters
    ----------
    ax : Axes
        Target axes.
    std_resid : ndarray
        Standardized residuals.
    theme : ThemeConfig
        Current theme.
    """
    n = len(std_resid)

    # Histogram
    n_bins = max(10, int(np.sqrt(n)))
    counts, bins, patches = ax.hist(
        std_resid, bins=n_bins, density=True,
        color=theme.colors[0], alpha=0.5, edgecolor=theme.text_color, linewidth=0.5,
    )

    # Normal curve
    x_grid = np.linspace(std_resid.min() - 0.5, std_resid.max() + 0.5, 200)
    mu, sigma = np.mean(std_resid), np.std(std_resid, ddof=1)
    normal_pdf = sp_stats.norm.pdf(x_grid, mu, sigma)
    ax.plot(x_grid, normal_pdf, color=theme.negative_color, linewidth=1.5, label="N(0,1)")

    # Jarque-Bera test
    jb_stat, jb_pval = sp_stats.jarque_bera(std_resid)
    jb_text = f"JB = {jb_stat:.2f}\np = {jb_pval:.4f}"
    ax.text(
        0.95, 0.95, jb_text,
        transform=ax.transAxes, fontsize=theme.tick_size - 1,
        verticalalignment="top", horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": theme.background, "alpha": 0.8,
              "edgecolor": theme.grid_color},
    )

    ax.set_title("Histogram + Normal", fontsize=theme.label_size, fontweight="bold")
    ax.set_xlabel("Std. Residual", fontsize=theme.tick_size)
    ax.set_ylabel("Density", fontsize=theme.tick_size)
    ax.legend(loc="upper left", fontsize=theme.tick_size - 1)
    ax.grid(True, alpha=theme.grid_alpha * 0.5)


def _compute_acf(y: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    """Compute sample autocorrelation function.

    Parameters
    ----------
    y : ndarray
        Time series.
    max_lag : int
        Maximum lag.

    Returns
    -------
    ndarray
        ACF values from lag 0 to max_lag.
    """
    n = len(y)
    y_demean = y - np.mean(y)
    var = np.sum(y_demean**2) / n
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for k in range(1, max_lag + 1):
        acf[k] = np.sum(y_demean[:n - k] * y_demean[k:]) / (n * var)
    return acf


def _compute_pacf(y: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    """Compute partial autocorrelation function via Durbin-Levinson.

    Parameters
    ----------
    y : ndarray
        Time series.
    max_lag : int
        Maximum lag.

    Returns
    -------
    ndarray
        PACF values from lag 0 to max_lag.
    """
    acf = _compute_acf(y, max_lag)
    pacf = np.zeros(max_lag + 1)
    pacf[0] = 1.0

    if max_lag == 0:
        return pacf

    # Durbin-Levinson algorithm
    phi = np.zeros((max_lag + 1, max_lag + 1))
    phi[1, 1] = acf[1]
    pacf[1] = acf[1]

    for k in range(2, max_lag + 1):
        num = acf[k] - sum(phi[k - 1, j] * acf[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * acf[j] for j in range(1, k))
        if abs(den) < 1e-15:
            break
        phi[k, k] = num / den
        pacf[k] = phi[k, k]
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

    return pacf
