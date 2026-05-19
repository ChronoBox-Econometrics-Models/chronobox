"""Utilities for advanced model examples (FAVAR, TVP-VAR, GVAR)."""

from .data_generators import (
    generate_factor_model,
    generate_tvp_var,
    generate_gvar_data,
)


def _import_plot_helpers():
    """Lazy import of plot_helpers (requires matplotlib)."""
    from . import plot_helpers
    return plot_helpers


__all__ = [
    "generate_factor_model",
    "generate_tvp_var",
    "generate_gvar_data",
    "plot_factors",
    "plot_factor_loadings",
    "plot_tvp_coefficients",
    "plot_spillover_network",
    "plot_historical_decomposition",
    "compare_results",
]


def __getattr__(name):
    plot_names = {
        "plot_factors",
        "plot_factor_loadings",
        "plot_tvp_coefficients",
        "plot_spillover_network",
        "plot_historical_decomposition",
        "compare_results",
    }
    if name in plot_names:
        mod = _import_plot_helpers()
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
