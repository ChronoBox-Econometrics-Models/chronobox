"""Utilities for ARIMA examples."""

from .data_generators import (
    generate_arima_process,
    generate_sarima_process,
    generate_arfima_process,
)


def _import_plot_helpers():
    """Lazy import of plot_helpers (requires statsmodels/matplotlib)."""
    from . import plot_helpers
    return plot_helpers


__all__ = [
    "generate_arima_process",
    "generate_sarima_process",
    "generate_arfima_process",
    "plot_acf_pacf",
    "plot_residual_diagnostics",
    "plot_forecast",
    "compare_results",
    "ljung_box_summary",
]


def __getattr__(name):
    plot_names = {
        "plot_acf_pacf",
        "plot_residual_diagnostics",
        "plot_forecast",
        "compare_results",
        "ljung_box_summary",
    }
    if name in plot_names:
        mod = _import_plot_helpers()
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
