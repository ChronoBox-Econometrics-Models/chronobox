"""Visualization module for chronobox.

Provides 11 chart types, 4 themes, and multi-format export capabilities.

Chart types:
    - plot_series: Multiple time series with annotations
    - plot_diagnostics: 2x2 residual diagnostics panel
    - plot_forecast: Fan chart with confidence intervals
    - plot_decomposition: Vertical panels (trend, seasonal, cycle, remainder)
    - plot_irf: Impulse response functions (KxK grid)
    - plot_fevd: Forecast error variance decomposition
    - plot_hd: Historical decomposition
    - plot_network / plot_heatmap / plot_rolling: Spillover visualizations
    - plot_tvp_coefs: Time-varying parameter coefficients
    - plot_cusum / plot_bai_perron / plot_zivot_andrews: Test visualizations

Themes:
    - professional: Blue/gray, Helvetica
    - academic: B&W, Computer Modern
    - presentation: Vibrant colors, large fonts
    - bcb: Banco Central do Brasil institutional style
"""

from __future__ import annotations

from chronobox.visualization.coef_plot import plot_tvp_coefs
from chronobox.visualization.decomposition_plot import plot_decomposition
from chronobox.visualization.diagnostics_plot import plot_diagnostics
from chronobox.visualization.fevd_plot import plot_fevd
from chronobox.visualization.forecast_plot import plot_forecast
from chronobox.visualization.hd_plot import plot_hd
from chronobox.visualization.irf_plot import plot_irf
from chronobox.visualization.spillover_plot import plot_heatmap, plot_network, plot_rolling
from chronobox.visualization.test_plot import (
    plot_bai_perron,
    plot_cusum,
    plot_recursive_coefs,
    plot_zivot_andrews,
)
from chronobox.visualization.themes import get_theme, list_themes, set_theme
from chronobox.visualization.ts_plot import plot_series

__all__ = [
    "get_theme",
    "list_themes",
    "plot_bai_perron",
    "plot_cusum",
    "plot_decomposition",
    "plot_diagnostics",
    "plot_fevd",
    "plot_forecast",
    "plot_hd",
    "plot_heatmap",
    "plot_irf",
    "plot_network",
    "plot_recursive_coefs",
    "plot_rolling",
    "plot_series",
    "plot_tvp_coefs",
    "plot_zivot_andrews",
    "set_theme",
]
