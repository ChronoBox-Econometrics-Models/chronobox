"""Utilities for ETS examples."""

from .data_generators import (
    generate_ets_process,
    generate_multiplicative_seasonal,
    generate_damped_trend,
)


def _import_plot_helpers():
    """Lazy import of plot_helpers (requires statsmodels/matplotlib)."""
    from . import plot_helpers
    return plot_helpers


__all__ = [
    "generate_ets_process",
    "generate_multiplicative_seasonal",
    "generate_damped_trend",
    "plot_ets_decomposition",
    "plot_seasonal_pattern",
    "plot_residual_diagnostics",
    "plot_forecast",
    "compare_results",
    "ljung_box_summary",
]


def __getattr__(name):
    plot_names = {
        "plot_ets_decomposition",
        "plot_seasonal_pattern",
        "plot_residual_diagnostics",
        "plot_forecast",
        "compare_results",
        "ljung_box_summary",
    }
    if name in plot_names:
        mod = _import_plot_helpers()
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
