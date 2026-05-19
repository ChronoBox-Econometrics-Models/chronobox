"""
Utility functions for statistical tests examples.

Provides data generators for unit root, cointegration, and structural break tests,
plus plot helpers for visualizing test results.
"""

from .data_generators import (
    generate_unit_root_process,
    generate_cointegrated_pair,
    generate_structural_break,
    generate_trend_stationary,
)
from .plot_helpers import (
    plot_rejection_rates,
    plot_confidence_bands,
    plot_cusum,
    plot_structural_break,
    plot_unit_root_series,
    plot_cointegration_residuals,
)

__all__ = [
    "generate_unit_root_process",
    "generate_cointegrated_pair",
    "generate_structural_break",
    "generate_trend_stationary",
    "plot_rejection_rates",
    "plot_confidence_bands",
    "plot_cusum",
    "plot_structural_break",
    "plot_unit_root_series",
    "plot_cointegration_residuals",
]
