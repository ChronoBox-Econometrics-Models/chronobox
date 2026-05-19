"""
Plot helpers for ARDL analysis.

Provides functions for plotting ARDL lag structures, ECM adjustment paths,
bounds test results, and long-run equilibrium relationships.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_ardl_lag_structure(
    ar_coefs: np.ndarray,
    x_coefs: np.ndarray,
    var_names: tuple[str, str] = ("y", "x"),
    title: str = "ARDL Lag Structure",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Plot ARDL lag structure as bar charts showing coefficient magnitudes.

    Parameters
    ----------
    ar_coefs : np.ndarray
        AR coefficients [phi_1, ..., phi_p].
    x_coefs : np.ndarray
        Coefficients for x [theta_0, theta_1, ..., theta_q].
    var_names : tuple of str
        Names of (dependent, regressor) variables.
    title : str
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    # AR lags
    p = len(ar_coefs)
    lags_ar = np.arange(1, p + 1)
    colors_ar = ["steelblue" if c >= 0 else "salmon" for c in ar_coefs]
    axes[0].bar(lags_ar, ar_coefs, color=colors_ar, edgecolor="black", alpha=0.8)
    axes[0].axhline(y=0, color="black", linewidth=0.5)
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title(f"AR lags of {var_names[0]}")
    axes[0].set_xticks(lags_ar)

    # Distributed lags of x
    q = len(x_coefs)
    lags_x = np.arange(0, q)
    colors_x = ["steelblue" if c >= 0 else "salmon" for c in x_coefs]
    axes[1].bar(lags_x, x_coefs, color=colors_x, edgecolor="black", alpha=0.8)
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Coefficient")
    axes[1].set_title(f"Distributed lags of {var_names[1]}")
    axes[1].set_xticks(lags_x)

    fig.tight_layout()
    return fig


def plot_ecm_adjustment(
    y: np.ndarray,
    x: np.ndarray,
    long_run_const: float,
    long_run_coef: float,
    dates: pd.DatetimeIndex | None = None,
    var_names: tuple[str, str] = ("y", "x"),
    title: str = "ECM Adjustment Path",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plot ECM adjustment dynamics: actual vs equilibrium and error correction term.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (levels).
    x : np.ndarray
        Regressor variable (levels).
    long_run_const : float
        Long-run intercept (a in y = a + b*x).
    long_run_coef : float
        Long-run slope coefficient (b).
    dates : pd.DatetimeIndex or None
        Date index for x-axis. If None, uses integer index.
    var_names : tuple of str
        Names of (dependent, regressor) variables.
    title : str
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    equilibrium = long_run_const + long_run_coef * x
    ec_term = y - equilibrium

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14)

    t = dates if dates is not None else np.arange(len(y))

    # Panel 1: actual y vs equilibrium y
    axes[0].plot(t, y, label=f"Actual {var_names[0]}", linewidth=1.2)
    axes[0].plot(t, equilibrium, label="Long-run equilibrium", linestyle="--",
                 color="red", linewidth=1.0)
    axes[0].set_ylabel(var_names[0])
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: x series
    axes[1].plot(t, x, color="green", linewidth=1.0)
    axes[1].set_ylabel(var_names[1])
    axes[1].grid(True, alpha=0.3)

    # Panel 3: error correction term
    axes[2].plot(t, ec_term, color="purple", linewidth=1.0)
    axes[2].axhline(y=0, color="black", linewidth=0.5)
    axes[2].fill_between(
        t if dates is not None else range(len(ec_term)),
        ec_term, 0, alpha=0.2, color="purple",
    )
    axes[2].set_ylabel("Equilibrium Error")
    axes[2].set_xlabel("Date" if dates is not None else "Observation")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_bounds_test(
    f_statistic: float,
    i0_bound: float,
    i1_bound: float,
    significance: str = "5%",
    title: str = "Pesaran-Shin-Smith Bounds Test",
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """Visualize bounds test result showing F-statistic relative to critical bounds.

    Parameters
    ----------
    f_statistic : float
        Computed F-statistic from the bounds test.
    i0_bound : float
        Lower bound (I(0) critical value).
    i1_bound : float
        Upper bound (I(1) critical value).
    significance : str
        Significance level label (e.g., "5%", "10%").
    title : str
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Draw the three zones
    x_min = 0
    x_max = max(f_statistic, i1_bound) * 1.5

    ax.axhspan(0, 0.33, facecolor="green", alpha=0.15, label="Cointegration (F > I(1))")
    ax.axhspan(0.33, 0.67, facecolor="yellow", alpha=0.15, label="Inconclusive")
    ax.axhspan(0.67, 1.0, facecolor="red", alpha=0.15, label="No cointegration (F < I(0))")

    # Draw bounds and F-stat on horizontal axis
    ax.barh(0.5, i0_bound, height=0.08, color="red", alpha=0.8,
            label=f"I(0) bound = {i0_bound:.3f}")
    ax.barh(0.5, i1_bound, height=0.08, color="orange", alpha=0.6,
            label=f"I(1) bound = {i1_bound:.3f}")
    ax.barh(0.5, f_statistic, height=0.08, color="blue", alpha=0.9,
            label=f"F-statistic = {f_statistic:.3f}")

    # Decision marker
    if f_statistic > i1_bound:
        decision = "Reject H0: Cointegration exists"
        marker_color = "green"
    elif f_statistic < i0_bound:
        decision = "Cannot reject H0: No cointegration"
        marker_color = "red"
    else:
        decision = "Inconclusive"
        marker_color = "orange"

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Test Statistic")
    ax.set_yticks([])
    ax.set_title(f"{title} ({significance} level)\n{decision}", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.axvline(x=i0_bound, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=i1_bound, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=f_statistic, color="blue", linestyle="-", linewidth=2, alpha=0.8)

    fig.tight_layout()
    return fig


def plot_long_run_relationship(
    y: np.ndarray,
    x: np.ndarray,
    long_run_const: float | None = None,
    long_run_coef: float | None = None,
    var_names: tuple[str, str] = ("y", "x"),
    title: str = "Long-Run Relationship",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Scatter plot of y vs x with fitted long-run relationship line.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable.
    x : np.ndarray
        Regressor variable.
    long_run_const : float or None
        Long-run intercept. If None, fits via OLS.
    long_run_coef : float or None
        Long-run slope. If None, fits via OLS.
    var_names : tuple of str
        Names of (dependent, regressor) variables.
    title : str
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if long_run_const is None or long_run_coef is None:
        # Simple OLS fit
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        long_run_const = beta[0]
        long_run_coef = beta[1]

    ax.scatter(x, y, alpha=0.4, s=20, color="steelblue", label="Observations")

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = long_run_const + long_run_coef * x_line
    ax.plot(x_line, y_line, color="red", linewidth=2,
            label=f"{var_names[0]} = {long_run_const:.2f} + {long_run_coef:.2f}*{var_names[1]}")

    ax.set_xlabel(var_names[1])
    ax.set_ylabel(var_names[0])
    ax.set_title(title)
    ax.legend(fontsize=10)
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
