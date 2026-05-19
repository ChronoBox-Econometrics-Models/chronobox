"""Utilities for VAR/VECM examples."""

from .data_generators import (
    generate_var_process,
    generate_vecm_process,
    generate_granger_causal,
)


def _import_plot_helpers():
    """Lazy import of plot_helpers (requires matplotlib)."""
    from . import plot_helpers
    return plot_helpers


__all__ = [
    "generate_var_process",
    "generate_vecm_process",
    "generate_granger_causal",
    "plot_irf",
    "plot_fevd",
    "plot_multivariate_series",
    "compare_results",
]


def __getattr__(name):
    plot_names = {
        "plot_irf",
        "plot_fevd",
        "plot_multivariate_series",
        "compare_results",
    }
    if name in plot_names:
        mod = _import_plot_helpers()
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
