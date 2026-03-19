"""Tests for basic visualization plots (ts_plot, diagnostics, forecast).

Tests:
    - plot_series generates figure
    - plot_diagnostics has 4 subplots
    - plot_forecast has growing bands
    - 4 themes apply without error
    - export_png generates file > 0 bytes
    - export_html generates file with content
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from chronobox.visualization.diagnostics_plot import plot_diagnostics
from chronobox.visualization.export import figure_to_html, save_figure
from chronobox.visualization.forecast_plot import plot_forecast
from chronobox.visualization.themes import get_theme, list_themes, set_theme
from chronobox.visualization.ts_plot import plot_series


@pytest.fixture
def sample_series() -> np.ndarray:
    """Generate a sample time series."""
    rng = np.random.default_rng(42)
    n = 200
    t = np.arange(n, dtype=np.float64)
    y = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 5, n)
    return y


@pytest.fixture
def sample_residuals() -> np.ndarray:
    """Generate sample residuals."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, 200)


class TestPlotSeries:
    """Tests for plot_series."""

    def test_generates_figure(self, sample_series: np.ndarray) -> None:
        """plot_series returns a matplotlib Figure."""
        fig = plot_series(sample_series)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiple_series(self, sample_series: np.ndarray) -> None:
        """plot_series handles multiple series."""
        fig = plot_series(
            [sample_series, sample_series * 1.1],
            labels=["Original", "Scaled"],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_title_and_labels(self, sample_series: np.ndarray) -> None:
        """plot_series applies title and labels."""
        fig = plot_series(
            sample_series,
            title="Test Series",
            xlabel="Time",
            ylabel="Value",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_annotations(self, sample_series: np.ndarray) -> None:
        """plot_series handles annotations."""
        fig = plot_series(
            sample_series,
            annotations=[{"x": 50, "y": sample_series[50], "text": "Event"}],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_secondary_y(self, sample_series: np.ndarray) -> None:
        """plot_series supports secondary Y axis."""
        fig = plot_series(
            [sample_series, sample_series * 0.01],
            labels=["GDP", "Rate"],
            secondary_y=["Rate"],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotDiagnostics:
    """Tests for plot_diagnostics."""

    def test_has_4_subplots(self, sample_residuals: np.ndarray) -> None:
        """plot_diagnostics creates exactly 4 subplots."""
        fig = plot_diagnostics(residuals=sample_residuals)
        assert isinstance(fig, Figure)
        axes = fig.get_axes()
        assert len(axes) == 4
        plt.close(fig)

    def test_with_custom_lags(self, sample_residuals: np.ndarray) -> None:
        """plot_diagnostics works with custom lag count."""
        fig = plot_diagnostics(residuals=sample_residuals, lags=10)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_raises_without_data(self) -> None:
        """plot_diagnostics raises ValueError without data."""
        with pytest.raises(ValueError):
            plot_diagnostics()


class TestPlotForecast:
    """Tests for plot_forecast."""

    def test_generates_figure(self, sample_series: np.ndarray) -> None:
        """plot_forecast returns a figure with growing bands."""
        np.random.default_rng(42)
        n_fcast = 24
        forecast_mean = np.full(n_fcast, sample_series[-1]) + np.arange(n_fcast) * 0.5
        # Growing standard errors
        forecast_se = np.sqrt(np.arange(1, n_fcast + 1)) * 2.0

        fig = plot_forecast(
            history=sample_series,
            forecast_mean=forecast_mean,
            forecast_se=forecast_se,
            steps=n_fcast,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_growing_bands(self, sample_series: np.ndarray) -> None:
        """Confidence bands should grow over time (SE increases)."""
        n_fcast = 12
        forecast_mean = np.full(n_fcast, sample_series[-1])
        forecast_se = np.sqrt(np.arange(1, n_fcast + 1)) * 3.0

        # SE should be increasing
        assert np.all(np.diff(forecast_se) > 0)

        fig = plot_forecast(
            history=sample_series,
            forecast_mean=forecast_mean,
            forecast_se=forecast_se,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_ci_levels(self, sample_series: np.ndarray) -> None:
        """plot_forecast works with custom CI levels."""
        n_fcast = 12
        forecast_mean = np.full(n_fcast, sample_series[-1])
        forecast_se = np.ones(n_fcast) * 5.0

        fig = plot_forecast(
            history=sample_series,
            forecast_mean=forecast_mean,
            forecast_se=forecast_se,
            ci_levels=[0.5, 0.9, 0.99],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestThemes:
    """Tests for theme system."""

    def test_4_themes_apply_without_error(self) -> None:
        """All 4 themes apply without raising errors."""
        themes = list_themes()
        assert len(themes) == 4
        for name in themes:
            set_theme(name)
            theme = get_theme()
            assert theme.name == name

    def test_professional_theme(self) -> None:
        """Professional theme has expected properties."""
        set_theme("professional")
        theme = get_theme()
        assert theme.name == "professional"
        assert len(theme.colors) >= 4

    def test_invalid_theme_raises(self) -> None:
        """Unknown theme name raises ValueError."""
        with pytest.raises(ValueError):
            set_theme("nonexistent")

    def test_list_themes(self) -> None:
        """list_themes returns sorted theme names."""
        themes = list_themes()
        assert "professional" in themes
        assert "academic" in themes
        assert "presentation" in themes
        assert "bcb" in themes


class TestExport:
    """Tests for figure export."""

    def test_export_png(self, sample_series: np.ndarray) -> None:
        """Export PNG generates file with content."""
        fig = plot_series(sample_series)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_figure(fig, Path(tmpdir) / "test.png", fmt="png")
            assert path.exists()
            assert path.stat().st_size > 0
        plt.close(fig)

    def test_export_svg(self, sample_series: np.ndarray) -> None:
        """Export SVG generates file with content."""
        fig = plot_series(sample_series)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_figure(fig, Path(tmpdir) / "test.svg", fmt="svg")
            assert path.exists()
            assert path.stat().st_size > 0
        plt.close(fig)

    def test_export_pdf(self, sample_series: np.ndarray) -> None:
        """Export PDF generates file with content."""
        fig = plot_series(sample_series)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_figure(fig, Path(tmpdir) / "test.pdf", fmt="pdf")
            assert path.exists()
            assert path.stat().st_size > 0
        plt.close(fig)

    def test_export_html(self, sample_series: np.ndarray) -> None:
        """Export HTML generates file with HTML content."""
        fig = plot_series(sample_series)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_figure(fig, Path(tmpdir) / "test.html", fmt="html")
            assert path.exists()
            assert path.stat().st_size > 0
            content = path.read_text(encoding="utf-8")
            assert "<html>" in content.lower() or "<img" in content.lower()
        plt.close(fig)

    def test_figure_to_html_string(self, sample_series: np.ndarray) -> None:
        """figure_to_html returns a non-empty HTML string."""
        fig = plot_series(sample_series)
        html = figure_to_html(fig)
        assert isinstance(html, str)
        assert len(html) > 100
        assert "base64" in html
        plt.close(fig)

    def test_infer_format_from_extension(self, sample_series: np.ndarray) -> None:
        """Format can be inferred from file extension."""
        fig = plot_series(sample_series)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_figure(fig, Path(tmpdir) / "test.png")  # No fmt arg
            assert path.exists()
        plt.close(fig)
