"""
Complete workflow utilities for chronobox examples.

Provides pipeline orchestration, plotting, and report generation
for end-to-end time series analysis workflows.
"""

from .pipeline import (
    univariate_pipeline,
    multivariate_pipeline,
    cross_validation_report,
)
from .plot_helpers import (
    plot_pipeline_summary,
    plot_multivariate_summary,
    plot_cross_validation,
    plot_forecast_comparison,
)
from .report_generator import generate_comparison_report

__all__ = [
    "univariate_pipeline",
    "multivariate_pipeline",
    "cross_validation_report",
    "plot_pipeline_summary",
    "plot_multivariate_summary",
    "plot_cross_validation",
    "plot_forecast_comparison",
    "generate_comparison_report",
]
