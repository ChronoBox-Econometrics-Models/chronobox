"""
Plot helpers for SVAR/BVAR analysis.

Provides functions for plotting structural impulse response functions,
confidence bands, sign restriction results, and historical decompositions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_structural_irf(
    irf_array: np.ndarray,
    variable_names: list[str] | None = None,
    shock_names: list[str] | None = None,
    ci_lower: np.ndarray | None = None,
    ci_upper: np.ndarray | None = None,
    title: str = "Structural Impulse Response Functions",
    figsize: tuple | None = None,
    ci_alpha: float = 0.2,
) -> plt.Figure:
    """Plot structural IRFs with confidence bands.

    Parameters
    ----------
    irf_array : np.ndarray
        IRF array of shape (periods, k_response, k_shock).
    variable_names : list of str or None
        Names of the response variables.
    shock_names : list of str or None
        Names of the structural shocks.
    ci_lower : np.ndarray or None
        Lower confidence band, same shape as irf_array.
    ci_upper : np.ndarray or None
        Upper confidence band, same shape as irf_array.
    title : str
        Overall figure title.
    figsize : tuple or None
        Figure size. If None, auto-calculated.
    ci_alpha : float
        Transparency for confidence band shading.

    Returns
    -------
    matplotlib.figure.Figure
    """
    periods, k_resp, k_shock = irf_array.shape

    if variable_names is None:
        variable_names = [f"Var {i+1}" for i in range(k_resp)]
    if shock_names is None:
        shock_names = [f"Shock {j+1}" for j in range(k_shock)]

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
                    alpha=ci_alpha, color="blue",
                )

            if i == 0:
                ax.set_title(f"Shock: {shock_names[j]}", fontsize=10)
            if j == 0:
                ax.set_ylabel(variable_names[i], fontsize=10)
            if i == k_resp - 1:
                ax.set_xlabel("Periods")

    fig.tight_layout()
    return fig


def plot_irf_comparison(
    irf_true: np.ndarray,
    irf_estimated: np.ndarray,
    variable_names: list[str] | None = None,
    shock_names: list[str] | None = None,
    ci_lower: np.ndarray | None = None,
    ci_upper: np.ndarray | None = None,
    title: str = "IRF: True vs Estimated",
    figsize: tuple | None = None,
) -> plt.Figure:
    """Compare true and estimated structural IRFs.

    Parameters
    ----------
    irf_true : np.ndarray
        True IRF array, shape (periods, k_response, k_shock).
    irf_estimated : np.ndarray
        Estimated IRF array, same shape.
    variable_names : list of str or None
        Names of the response variables.
    shock_names : list of str or None
        Names of the structural shocks.
    ci_lower : np.ndarray or None
        Lower confidence band for estimated IRF.
    ci_upper : np.ndarray or None
        Upper confidence band for estimated IRF.
    title : str
        Overall figure title.
    figsize : tuple or None
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    periods, k_resp, k_shock = irf_true.shape

    if variable_names is None:
        variable_names = [f"Var {i+1}" for i in range(k_resp)]
    if shock_names is None:
        shock_names = [f"Shock {j+1}" for j in range(k_shock)]

    if figsize is None:
        figsize = (4 * k_shock, 3 * k_resp)

    fig, axes = plt.subplots(k_resp, k_shock, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14)

    x = np.arange(periods)

    for i in range(k_resp):
        for j in range(k_shock):
            ax = axes[i, j]
            ax.plot(x, irf_true[:, i, j], color="red", linewidth=1.5,
                    linestyle="--", label="True")
            ax.plot(x, irf_estimated[:, i, j], color="blue", linewidth=1.5,
                    label="Estimated")
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

            if ci_lower is not None and ci_upper is not None:
                ax.fill_between(
                    x, ci_lower[:, i, j], ci_upper[:, i, j],
                    alpha=0.15, color="blue",
                )

            if i == 0:
                ax.set_title(f"Shock: {shock_names[j]}", fontsize=10)
            if j == 0:
                ax.set_ylabel(variable_names[i], fontsize=10)
            if i == k_resp - 1:
                ax.set_xlabel("Periods")
            if i == 0 and j == k_shock - 1:
                ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    return fig


def plot_sign_restriction_irfs(
    accepted_irfs: list[np.ndarray],
    variable_names: list[str] | None = None,
    shock_names: list[str] | None = None,
    quantiles: tuple[float, float] = (0.16, 0.84),
    title: str = "Sign-Restriction Identified IRFs",
    figsize: tuple | None = None,
) -> plt.Figure:
    """Plot IRFs from sign-restriction identification (median + bands).

    Parameters
    ----------
    accepted_irfs : list of np.ndarray
        List of accepted IRF arrays, each of shape (periods, k_resp, k_shock).
    variable_names : list of str or None
        Names of the response variables.
    shock_names : list of str or None
        Names of the structural shocks.
    quantiles : tuple of float
        Lower and upper quantiles for the bands (default: 68% band).
    title : str
        Overall figure title.
    figsize : tuple or None
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    stacked = np.stack(accepted_irfs, axis=0)  # (n_accepted, periods, k_resp, k_shock)
    median_irf = np.median(stacked, axis=0)
    lower = np.quantile(stacked, quantiles[0], axis=0)
    upper = np.quantile(stacked, quantiles[1], axis=0)

    periods, k_resp, k_shock = median_irf.shape

    if variable_names is None:
        variable_names = [f"Var {i+1}" for i in range(k_resp)]
    if shock_names is None:
        shock_names = [f"Shock {j+1}" for j in range(k_shock)]

    if figsize is None:
        figsize = (4 * k_shock, 3 * k_resp)

    fig, axes = plt.subplots(k_resp, k_shock, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14)

    x = np.arange(periods)

    for i in range(k_resp):
        for j in range(k_shock):
            ax = axes[i, j]
            ax.plot(x, median_irf[:, i, j], color="blue", linewidth=1.5,
                    label="Median")
            ax.fill_between(
                x, lower[:, i, j], upper[:, i, j],
                alpha=0.2, color="blue",
                label=f"{int((quantiles[1]-quantiles[0])*100)}% band",
            )
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

            if i == 0:
                ax.set_title(f"Shock: {shock_names[j]}", fontsize=10)
            if j == 0:
                ax.set_ylabel(variable_names[i], fontsize=10)
            if i == k_resp - 1:
                ax.set_xlabel("Periods")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    return fig


def plot_historical_decomposition(
    hd: np.ndarray,
    dates: pd.DatetimeIndex | np.ndarray,
    variable_names: list[str] | None = None,
    shock_names: list[str] | None = None,
    title: str = "Historical Decomposition",
    figsize: tuple | None = None,
    colors: list[str] | None = None,
) -> plt.Figure:
    """Plot historical decomposition as stacked bar charts.

    Parameters
    ----------
    hd : np.ndarray
        Historical decomposition of shape (T, k_response, k_shock).
    dates : pd.DatetimeIndex or np.ndarray
        Date index for the x-axis.
    variable_names : list of str or None
        Names of the response variables.
    shock_names : list of str or None
        Names of the structural shocks.
    title : str
        Overall figure title.
    figsize : tuple or None
        Figure size.
    colors : list of str or None
        Colors for each shock contribution.

    Returns
    -------
    matplotlib.figure.Figure
    """
    T, k_resp, k_shock = hd.shape

    if variable_names is None:
        variable_names = [f"Var {i+1}" for i in range(k_resp)]
    if shock_names is None:
        shock_names = [f"Shock {j+1}" for j in range(k_shock)]

    if figsize is None:
        figsize = (12, 3 * k_resp)

    if colors is None:
        cmap = plt.cm.Set2
        colors = [cmap(i / max(k_shock - 1, 1)) for i in range(k_shock)]

    fig, axes = plt.subplots(k_resp, 1, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14)

    for i in range(k_resp):
        ax = axes[i, 0]
        pos_bottom = np.zeros(T)
        neg_bottom = np.zeros(T)

        for j in range(k_shock):
            vals = hd[:, i, j]
            pos_vals = np.where(vals >= 0, vals, 0)
            neg_vals = np.where(vals < 0, vals, 0)

            ax.bar(dates, pos_vals, bottom=pos_bottom, color=colors[j],
                   label=shock_names[j], alpha=0.8, width=60)
            ax.bar(dates, neg_vals, bottom=neg_bottom, color=colors[j],
                   alpha=0.8, width=60)

            pos_bottom += pos_vals
            neg_bottom += neg_vals

        actual = hd[:, i, :].sum(axis=1)
        ax.plot(dates, actual, color="black", linewidth=1.0, label="Actual")

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_ylabel(variable_names[i])

        if i == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=k_shock + 1)
        if i == k_resp - 1:
            ax.set_xlabel("Date")

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
