"""
Plot helpers for advanced model analysis (FAVAR, TVP-VAR, GVAR).

Provides functions for plotting factors, time-varying parameters,
spillover networks, and historical decompositions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_factors(
    factors: np.ndarray,
    factor_names: list[str] | None = None,
    dates: pd.DatetimeIndex | None = None,
    title: str = "Estimated Factors",
    figsize: tuple | None = None,
) -> plt.Figure:
    """Plot estimated latent factors from a FAVAR model.

    Parameters
    ----------
    factors : np.ndarray
        Factor array of shape (n, n_factors).
    factor_names : list of str or None
        Names for each factor. If None, uses generic labels.
    dates : pd.DatetimeIndex or None
        Date index for the x-axis.
    title : str
        Figure title.
    figsize : tuple or None
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n, n_factors = factors.shape

    if factor_names is None:
        factor_names = [f"Factor {i+1}" for i in range(n_factors)]

    if figsize is None:
        figsize = (12, 2.5 * n_factors)

    x = dates if dates is not None else np.arange(n)

    fig, axes = plt.subplots(n_factors, 1, figsize=figsize, sharex=True, squeeze=False)
    fig.suptitle(title, fontsize=14)

    for i in range(n_factors):
        ax = axes[i, 0]
        ax.plot(x, factors[:, i], linewidth=1.2, color="steelblue")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_ylabel(factor_names[i])
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Date" if dates is not None else "Observation")
    fig.tight_layout()
    return fig


def plot_factor_loadings(
    loadings: np.ndarray,
    series_names: list[str] | None = None,
    factor_names: list[str] | None = None,
    title: str = "Factor Loadings",
    figsize: tuple | None = None,
) -> plt.Figure:
    """Plot factor loadings as a heatmap.

    Parameters
    ----------
    loadings : np.ndarray
        Loading matrix of shape (n_series, n_factors).
    series_names : list of str or None
        Names for each observable series.
    factor_names : list of str or None
        Names for each factor.
    title : str
        Figure title.
    figsize : tuple or None
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_series, n_factors = loadings.shape

    if series_names is None:
        series_names = [f"Series {i+1}" for i in range(n_series)]
    if factor_names is None:
        factor_names = [f"Factor {i+1}" for i in range(n_factors)]

    if figsize is None:
        figsize = (max(6, n_factors * 1.5), max(4, n_series * 0.4))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(loadings, aspect="auto", cmap="RdBu_r", vmin=-np.max(np.abs(loadings)),
                   vmax=np.max(np.abs(loadings)))
    ax.set_xticks(range(n_factors))
    ax.set_xticklabels(factor_names)
    ax.set_yticks(range(n_series))
    ax.set_yticklabels(series_names)
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_tvp_coefficients(
    A_path: np.ndarray,
    variable_names: list[str] | None = None,
    dates: pd.DatetimeIndex | None = None,
    title: str = "Time-Varying VAR Coefficients",
    figsize: tuple | None = None,
) -> plt.Figure:
    """Plot time-varying VAR coefficient paths.

    Parameters
    ----------
    A_path : np.ndarray
        Coefficient path of shape (n, k, k).
    variable_names : list of str or None
        Variable names for labeling.
    dates : pd.DatetimeIndex or None
        Date index for the x-axis.
    title : str
        Figure title.
    figsize : tuple or None
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n, k, _ = A_path.shape

    if variable_names is None:
        variable_names = [f"Var {i+1}" for i in range(k)]

    if figsize is None:
        figsize = (12, 3 * k)

    x = dates if dates is not None else np.arange(n)

    fig, axes = plt.subplots(k, k, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14)

    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            ax.plot(x, A_path[:, i, j], linewidth=1.0, color="darkred")
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax.set_title(f"A[{variable_names[i]},{variable_names[j]}]", fontsize=9)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_spillover_network(
    spillover_matrix: np.ndarray,
    variable_names: list[str] | None = None,
    threshold: float = 0.05,
    title: str = "Spillover Network",
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """Plot spillover connectedness as a network/heatmap.

    Parameters
    ----------
    spillover_matrix : np.ndarray
        Spillover table of shape (k, k), where entry (i, j) is the
        contribution of variable j to the forecast error of variable i.
    variable_names : list of str or None
        Variable or country names.
    threshold : float
        Minimum spillover value to display.
    title : str
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    k = spillover_matrix.shape[0]

    if variable_names is None:
        variable_names = [f"Var {i+1}" for i in range(k)]

    fig, ax = plt.subplots(figsize=figsize)

    # Mask small values for clarity
    masked = spillover_matrix.copy()
    masked[masked < threshold] = 0

    im = ax.imshow(masked, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(k))
    ax.set_xticklabels(variable_names, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels(variable_names)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("From")
    ax.set_ylabel("To")

    # Annotate cells
    for i in range(k):
        for j in range(k):
            val = spillover_matrix[i, j]
            if val >= threshold:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Spillover (%)")
    fig.tight_layout()
    return fig


def plot_historical_decomposition(
    decomposition: np.ndarray,
    variable_names: list[str] | None = None,
    shock_names: list[str] | None = None,
    dates: pd.DatetimeIndex | None = None,
    target_var: int = 0,
    title: str = "Historical Decomposition",
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Plot historical decomposition of a variable into structural shocks.

    Parameters
    ----------
    decomposition : np.ndarray
        Decomposition array of shape (n, k_shocks) for the target variable,
        or (n, k_response, k_shocks) for all variables.
    variable_names : list of str or None
        Names of response variables.
    shock_names : list of str or None
        Names of structural shocks.
    dates : pd.DatetimeIndex or None
        Date index for the x-axis.
    target_var : int
        Index of the target variable (used if decomposition is 3D).
    title : str
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if decomposition.ndim == 3:
        decomp = decomposition[:, target_var, :]
    else:
        decomp = decomposition

    n, k_shocks = decomp.shape

    if shock_names is None:
        shock_names = [f"Shock {i+1}" for i in range(k_shocks)]

    x = dates if dates is not None else np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.Set2
    colors = [cmap(i / max(k_shocks - 1, 1)) for i in range(k_shocks)]

    # Stacked bar chart
    pos = np.maximum(decomp, 0)
    neg = np.minimum(decomp, 0)

    bottom_pos = np.zeros(n)
    bottom_neg = np.zeros(n)

    for j in range(k_shocks):
        ax.bar(x, pos[:, j], bottom=bottom_pos, label=shock_names[j],
               color=colors[j], alpha=0.8, width=1.0)
        ax.bar(x, neg[:, j], bottom=bottom_neg, color=colors[j], alpha=0.8, width=1.0)
        bottom_pos += pos[:, j]
        bottom_neg += neg[:, j]

    # Actual series (sum of contributions)
    actual = decomp.sum(axis=1)
    ax.plot(x, actual, color="black", linewidth=1.5, label="Actual")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date" if dates is not None else "Observation")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def compare_results(
    python_results: dict,
    r_results: dict | None = None,
    stata_results: dict | None = None,
    metric_names: list | None = None,
) -> pd.DataFrame:
    """Compare estimation results across Python, R, and Stata.

    Parameters
    ----------
    python_results : dict
        Dict of metric_name -> value from Python estimation.
    r_results : dict or None
        Dict of metric_name -> value from R estimation.
    stata_results : dict or None
        Dict of metric_name -> value from Stata estimation.
    metric_names : list or None
        Specific metrics to compare. If None, uses all keys from python_results.

    Returns
    -------
    pd.DataFrame
        Comparison table with columns for each tool.
    """
    if metric_names is None:
        metric_names = list(python_results.keys())

    data = {"Metric": metric_names, "Python": [python_results.get(m) for m in metric_names]}

    if r_results is not None:
        data["R"] = [r_results.get(m) for m in metric_names]

    if stata_results is not None:
        data["Stata"] = [stata_results.get(m) for m in metric_names]

    df = pd.DataFrame(data).set_index("Metric")
    return df
