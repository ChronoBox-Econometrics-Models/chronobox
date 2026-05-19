"""
Plot helpers for complete workflow pipeline summaries.

Provides high-level plotting functions that visualize the output of
univariate_pipeline() and multivariate_pipeline().
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Any


# ---------------------------------------------------------------------------
# Univariate pipeline summary
# ---------------------------------------------------------------------------

def plot_pipeline_summary(
    series: pd.Series,
    pipeline_results: dict[str, Any],
    title: str = "Univariate Pipeline Summary",
    figsize: tuple = (16, 14),
) -> plt.Figure:
    """Create a multi-panel summary figure from univariate pipeline results.

    Panels
    ------
    1. Original series + trend (HP filter)
    2. Decomposition components (trend, seasonal, residual)
    3. Cyclical component from filters
    4. Forecast comparison across models

    Parameters
    ----------
    series : pd.Series
        Original time series.
    pipeline_results : dict
        Output of ``univariate_pipeline()``.
    title : str
        Overall figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 1, figure=fig, hspace=0.35)

    # Panel 1 — Original + HP trend
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(series.index, series.values, label="Original", color="steelblue")
    hp = pipeline_results.get("filters", {}).get("hp", {})
    if "trend" in hp:
        trend = np.array(hp["trend"])
        idx = series.dropna().index
        if len(trend) == len(idx):
            ax1.plot(idx, trend, label="HP trend", color="firebrick", ls="--")
    ax1.set_title("Original Series + HP Trend")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2 — Decomposition
    ax2 = fig.add_subplot(gs[1])
    dec = pipeline_results.get("decomposition", {})
    if "trend" in dec and dec["trend"] is not None:
        t = np.array(dec["trend"])
        ax2.plot(t, label="Trend", color="darkgreen")
    if "seasonal" in dec and dec["seasonal"] is not None:
        s = np.array(dec["seasonal"])
        ax2.plot(s, label="Seasonal", color="darkorange", alpha=0.7)
    if "residual" in dec and dec["residual"] is not None:
        r = np.array(dec["residual"])
        ax2.plot(r, label="Residual", color="gray", alpha=0.5)
    method = dec.get("method", "")
    ax2.set_title(f"Decomposition ({method})")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Panel 3 — Cyclical components
    ax3 = fig.add_subplot(gs[2])
    if "cycle" in hp:
        cycle_hp = np.array(hp["cycle"])
        ax3.plot(cycle_hp, label="HP cycle", color="steelblue")
    cf = pipeline_results.get("filters", {}).get("cf", {})
    if "cycle" in cf:
        cycle_cf = np.array(cf["cycle"])
        ax3.plot(cycle_cf, label="CF cycle", color="firebrick", ls="--")
    ax3.axhline(0, color="black", lw=0.5)
    ax3.set_title("Cyclical Components (HP vs CF)")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Panel 4 — Forecast comparison
    ax4 = fig.add_subplot(gs[3])
    forecasts = pipeline_results.get("forecasts", {})
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(forecasts), 1)))
    for i, (name, info) in enumerate(forecasts.items()):
        if "values" in info:
            vals = info["values"]
            ax4.plot(
                range(len(vals)),
                vals,
                label=f"{name} (RMSE={info.get('rmse', 0):.3f})",
                color=colors[i],
            )
    # Plot actual
    h = len(next((v["values"] for v in forecasts.values() if "values" in v), []))
    if h > 0:
        actual = series.values[-h:]
        ax4.plot(range(h), actual, "k--", label="Actual", lw=2)
    best = pipeline_results.get("best_model", "")
    ax4.set_title(f"Forecast Comparison (best: {best})")
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    return fig


# ---------------------------------------------------------------------------
# Multivariate pipeline summary
# ---------------------------------------------------------------------------

def plot_multivariate_summary(
    data: pd.DataFrame,
    pipeline_results: dict[str, Any],
    title: str = "Multivariate Pipeline Summary",
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """Create a multi-panel summary from multivariate pipeline results.

    Panels
    ------
    1. Raw variables (standardised)
    2. Unit-root test summary table
    3. VAR forecasts
    4. IRF preview (first shock → all variables)

    Parameters
    ----------
    data : pd.DataFrame
        Original multivariate data.
    pipeline_results : dict
        Output of ``multivariate_pipeline()``.
    title : str
        Overall figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1 — Standardised variables
    ax1 = fig.add_subplot(gs[0, 0])
    for col in data.columns:
        s = data[col].dropna()
        ax1.plot(s.index, (s - s.mean()) / s.std(), label=col)
    ax1.set_title("Variables (standardised)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2 — Test summary
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    tests = pipeline_results.get("tests", {})
    cell_text = []
    row_labels = []
    for var_name, var_tests in tests.items():
        for test_name, res in var_tests.items():
            stat = res.get("statistic", "—")
            pval = res.get("p_value", "—")
            rej = res.get("reject_h0", "—")
            if isinstance(stat, float):
                stat = f"{stat:.3f}"
            if isinstance(pval, float):
                pval = f"{pval:.3f}"
            cell_text.append([str(stat), str(pval), str(rej)])
            row_labels.append(f"{var_name}/{test_name}")
    if cell_text:
        table = ax2.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=["Statistic", "p-value", "Reject H0"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.2)
    ax2.set_title("Unit-Root Tests", pad=20)

    # Panel 3 — VAR forecasts
    ax3 = fig.add_subplot(gs[1, 0])
    fc = pipeline_results.get("forecasts", {})
    if "values" in fc:
        fc_arr = np.array(fc["values"])
        for j, col in enumerate(data.columns):
            if j < fc_arr.shape[1]:
                ax3.plot(fc_arr[:, j], label=f"{col} forecast")
    ax3.set_title("VAR Forecasts")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4 — placeholder for IRF
    ax4 = fig.add_subplot(gs[1, 1])
    irf_data = pipeline_results.get("irf", {})
    if "irf" in irf_data and hasattr(irf_data["irf"], "__len__"):
        ax4.text(
            0.5, 0.5,
            "IRF computed — use dedicated\nIRF plot for full detail",
            ha="center", va="center", fontsize=10, transform=ax4.transAxes,
        )
    else:
        err = irf_data.get("error", "Not computed")
        ax4.text(
            0.5, 0.5, f"IRF: {err}",
            ha="center", va="center", fontsize=10, transform=ax4.transAxes,
        )
    ax4.set_title("Impulse Response Functions")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    return fig


# ---------------------------------------------------------------------------
# Cross-validation comparison
# ---------------------------------------------------------------------------

def plot_cross_validation(
    comparison_df: pd.DataFrame,
    title: str = "Cross-Validation: Python vs R vs Stata",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Bar chart comparing metrics across Python, R, and Stata.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of ``cross_validation_report()``.
    title : str
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = comparison_df.copy()
    numeric_cols = ["python", "r", "stata"]
    df = df.dropna(subset=numeric_cols, how="all")

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))
    width = 0.25

    for i, (col, color) in enumerate(
        zip(numeric_cols, ["steelblue", "firebrick", "forestgreen"])
    ):
        vals = df[col].fillna(0).values
        ax.bar(x + i * width, vals, width, label=col.capitalize(), color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["metric"], rotation=45, ha="right")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_forecast_comparison(
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    title: str = "Forecast Comparison",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Line plot comparing forecasts from multiple sources against actuals.

    Parameters
    ----------
    actual : array-like
        Actual values.
    forecasts : dict
        ``{label: forecast_array}``.
    title : str
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(actual, "k-o", label="Actual", lw=2, markersize=4)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(forecasts), 1)))
    for i, (label, fc) in enumerate(forecasts.items()):
        ax.plot(fc, "--s", label=label, color=colors[i], markersize=3)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
