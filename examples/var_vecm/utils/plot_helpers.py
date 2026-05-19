"""
Plot helpers for VAR/VECM analysis.

Provides functions for plotting impulse response functions (IRF),
forecast error variance decomposition (FEVD), and multivariate time series.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_irf(
    irf_array: np.ndarray,
    variable_names: list[str] | None = None,
    shock_names: list[str] | None = None,
    ci_lower: np.ndarray | None = None,
    ci_upper: np.ndarray | None = None,
    title: str = "Impulse Response Functions",
    figsize: tuple | None = None,
) -> plt.Figure:
    """Plot impulse response functions in a grid.

    Parameters
    ----------
    irf_array : np.ndarray
        IRF array of shape (periods, k_response, k_shock).
    variable_names : list of str or None
        Names of the response variables. If None, uses generic labels.
    shock_names : list of str or None
        Names of the shock variables. If None, uses variable_names.
    ci_lower : np.ndarray or None
        Lower confidence band, same shape as irf_array.
    ci_upper : np.ndarray or None
        Upper confidence band, same shape as irf_array.
    title : str
        Overall figure title.
    figsize : tuple or None
        Figure size. If None, auto-calculated.

    Returns
    -------
    matplotlib.figure.Figure
    """
    periods, k_resp, k_shock = irf_array.shape

    if variable_names is None:
        variable_names = [f"Var {i+1}" for i in range(k_resp)]
    if shock_names is None:
        shock_names = variable_names[:k_shock]

    if figsize is None:
        figsize = (4 * k_shock, 3 * k_resp)

    fig, axes = plt.subplots(k_resp, k_shock, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14)

    x = np.arange(periods)

    for i in range(k_resp):
        for j in range(k_shock):
            ax = axes[i, j]
            ax.plot(x, irf_array[:, i, j], color="blue", linewidth=1.5)
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

            if ci_lower is not None and ci_upper is not None:
                ax.fill_between(
                    x, ci_lower[:, i, j], ci_upper[:, i, j],
                    alpha=0.2, color="blue",
                )

            if i == 0:
                ax.set_title(f"Shock: {shock_names[j]}", fontsize=10)
            if j == 0:
                ax.set_ylabel(variable_names[i], fontsize=10)
            if i == k_resp - 1:
                ax.set_xlabel("Periods")

    fig.tight_layout()
    return fig


def plot_fevd(
    fevd_array: np.ndarray,
    variable_names: list[str] | None = None,
    title: str = "Forecast Error Variance Decomposition",
    figsize: tuple | None = None,
    colors: list[str] | None = None,
) -> plt.Figure:
    """Plot forecast error variance decomposition as stacked area charts.

    Parameters
    ----------
    fevd_array : np.ndarray
        FEVD array of shape (periods, k_response, k_source).
        Each row for a given response variable should sum to 1.
    variable_names : list of str or None
        Names of the variables. If None, uses generic labels.
    title : str
        Overall figure title.
    figsize : tuple or None
        Figure size. If None, auto-calculated.
    colors : list of str or None
        Colors for each source variable.

    Returns
    -------
    matplotlib.figure.Figure
    """
    periods, k_resp, k_source = fevd_array.shape

    if variable_names is None:
        variable_names = [f"Var {i+1}" for i in range(max(k_resp, k_source))]

    if figsize is None:
        figsize = (10, 3 * k_resp)

    if colors is None:
        cmap = plt.cm.Set2
        colors = [cmap(i / max(k_source - 1, 1)) for i in range(k_source)]

    fig, axes = plt.subplots(k_resp, 1, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14)

    x = np.arange(1, periods + 1)

    for i in range(k_resp):
        ax = axes[i, 0]
        bottom = np.zeros(periods)

        for j in range(k_source):
            ax.fill_between(
                x, bottom, bottom + fevd_array[:, i, j],
                label=variable_names[j], color=colors[j], alpha=0.8,
            )
            bottom += fevd_array[:, i, j]

        ax.set_ylabel(variable_names[i])
        ax.set_ylim(0, 1)
        ax.set_xlim(1, periods)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        if i == k_resp - 1:
            ax.set_xlabel("Forecast Horizon")

    fig.tight_layout()
    return fig


def plot_multivariate_series(
    df: pd.DataFrame,
    date_col: str = "date",
    title: str = "Multivariate Time Series",
    figsize: tuple | None = None,
) -> plt.Figure:
    """Plot multiple time series in vertically stacked subplots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a date column and one or more numeric columns.
    date_col : str
        Name of the date column.
    title : str
        Overall figure title.
    figsize : tuple or None
        Figure size. If None, auto-calculated.

    Returns
    -------
    matplotlib.figure.Figure
    """
    numeric_cols = [c for c in df.columns if c != date_col]
    k = len(numeric_cols)

    if figsize is None:
        figsize = (12, 2.5 * k)

    fig, axes = plt.subplots(k, 1, figsize=figsize, sharex=True, squeeze=False)
    fig.suptitle(title, fontsize=14)

    dates = pd.to_datetime(df[date_col])

    for i, col in enumerate(numeric_cols):
        ax = axes[i, 0]
        ax.plot(dates, df[col], linewidth=1.0)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Date")
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
