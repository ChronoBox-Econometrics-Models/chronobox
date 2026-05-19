"""Utilities for SVAR/BVAR examples."""

from .data_generators import (
    generate_svar_process,
    generate_demand_supply,
    generate_monetary_policy,
    generate_blanchard_quah,
    generate_sign_restriction_dgp,
)


def _import_plot_helpers():
    """Lazy import of plot_helpers (requires matplotlib)."""
    from . import plot_helpers
    return plot_helpers


__all__ = [
    "generate_svar_process",
    "generate_demand_supply",
    "generate_monetary_policy",
    "generate_blanchard_quah",
    "generate_sign_restriction_dgp",
    "plot_structural_irf",
    "plot_irf_comparison",
    "plot_sign_restriction_irfs",
    "plot_historical_decomposition",
    "compare_results",
]


def __getattr__(name):
    plot_names = {
        "plot_structural_irf",
        "plot_irf_comparison",
        "plot_sign_restriction_irfs",
        "plot_historical_decomposition",
        "compare_results",
    }
    if name in plot_names:
        mod = _import_plot_helpers()
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
