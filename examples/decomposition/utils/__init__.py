"""Utilities for decomposition examples."""

from .data_generators import (
    generate_additive_seasonal,
    generate_multiplicative_seasonal,
    generate_complex_seasonal,
)


def _import_plot_helpers():
    """Lazy import of plot_helpers (requires matplotlib)."""
    from . import plot_helpers
    return plot_helpers


__all__ = [
    "generate_additive_seasonal",
    "generate_multiplicative_seasonal",
    "generate_complex_seasonal",
    "plot_decomposition",
    "plot_seasonal_subseries",
    "plot_stl_diagnostics",
    "compare_components",
]


def __getattr__(name):
    plot_names = {
        "plot_decomposition",
        "plot_seasonal_subseries",
        "plot_stl_diagnostics",
        "compare_components",
    }
    if name in plot_names:
        mod = _import_plot_helpers()
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
