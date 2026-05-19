"""Utilities for ARDL examples."""

from .data_generators import (
    generate_ardl_process,
    generate_ecm_data,
    generate_mixed_integration,
)


def _import_plot_helpers():
    """Lazy import of plot_helpers (requires matplotlib)."""
    from . import plot_helpers
    return plot_helpers


__all__ = [
    "generate_ardl_process",
    "generate_ecm_data",
    "generate_mixed_integration",
    "plot_ardl_lag_structure",
    "plot_ecm_adjustment",
    "plot_bounds_test",
    "plot_long_run_relationship",
    "compare_results",
]


def __getattr__(name):
    plot_names = {
        "plot_ardl_lag_structure",
        "plot_ecm_adjustment",
        "plot_bounds_test",
        "plot_long_run_relationship",
        "compare_results",
    }
    if name in plot_names:
        mod = _import_plot_helpers()
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
