"""Integration tests for the visualization module.

Tests the complete pipeline: create plot -> apply theme -> export to file.
Covers all 11 chart types with all 4 themes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from chronobox.visualization import (
    list_themes,
    plot_bai_perron,
    plot_cusum,
    plot_decomposition,
    plot_diagnostics,
    plot_fevd,
    plot_forecast,
    plot_hd,
    plot_heatmap,
    plot_irf,
    plot_network,
    plot_recursive_coefs,
    plot_rolling,
    plot_series,
    plot_tvp_coefs,
    plot_zivot_andrews,
    set_theme,
)
from chronobox.visualization.export import figure_to_html, save_figure


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def sample_series(rng: np.random.Generator) -> np.ndarray:
    n = 200
    t = np.arange(n, dtype=np.float64)
    return 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 5, n)


class TestAllPlotsAllThemes:
    """Test that all 11 chart types work with all 4 themes."""

    @pytest.mark.parametrize("theme_name", ["professional", "academic", "presentation", "bcb"])
    def test_plot_series_all_themes(self, sample_series: np.ndarray, theme_name: str) -> None:
        set_theme(theme_name)
        fig = plot_series(sample_series, title=f"Test - {theme_name}")
        assert isinstance(fig, Figure)
        plt.close(fig)

    @pytest.mark.parametrize("theme_name", ["professional", "academic", "presentation", "bcb"])
    def test_plot_diagnostics_all_themes(self, theme_name: str, rng: np.random.Generator) -> None:
        set_theme(theme_name)
        resid = rng.normal(0, 1, 200)
        fig = plot_diagnostics(residuals=resid)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 4
        plt.close(fig)

    @pytest.mark.parametrize("theme_name", ["professional", "academic", "presentation", "bcb"])
    def test_plot_forecast_all_themes(
        self, sample_series: np.ndarray, theme_name: str
    ) -> None:
        set_theme(theme_name)
        n_fc = 12
        fc_mean = np.full(n_fc, sample_series[-1])
        fc_se = np.sqrt(np.arange(1, n_fc + 1)) * 3.0
        fig = plot_forecast(history=sample_series, forecast_mean=fc_mean, forecast_se=fc_se)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_decomposition_integration(self, sample_series: np.ndarray) -> None:
        n = len(sample_series)
        t = np.arange(n, dtype=np.float64)
        components = {
            "observed": sample_series,
            "trend": 100 + 0.5 * t,
            "seasonal": 10 * np.sin(2 * np.pi * t / 12),
            "remainder": sample_series - (100 + 0.5 * t) - 10 * np.sin(2 * np.pi * t / 12),
        }
        fig = plot_decomposition(components=components)
        assert len(fig.get_axes()) == 4
        plt.close(fig)

    def test_irf_integration(self, rng: np.random.Generator) -> None:
        H, K = 20, 3
        irf = np.zeros((H, K, K))
        for i in range(K):
            for j in range(K):
                irf[:, i, j] = np.exp(-np.arange(H) * 0.2) * rng.normal(0, 0.5, H)
                if i == j:
                    irf[0, i, j] = 1.0
        fig = plot_irf(irf_array=irf, var_names=["Y1", "Y2", "Y3"])
        assert len(fig.get_axes()) == K * K
        plt.close(fig)

    def test_fevd_integration(self, rng: np.random.Generator) -> None:
        H, K = 20, 3
        fevd = rng.dirichlet(np.ones(K), size=(H, K))
        fig = plot_fevd(fevd_array=fevd, var_names=["Y1", "Y2", "Y3"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_hd_integration(self, rng: np.random.Generator) -> None:
        T, K = 100, 3
        hd = rng.normal(0, 5, (T, K, K))
        observed = np.sum(hd, axis=2)
        fig = plot_hd(hd_array=hd, observed=observed, var_names=["Y1", "Y2", "Y3"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_spillover_integration(self, rng: np.random.Generator) -> None:
        K = 4
        table = rng.uniform(5, 30, (K, K))
        np.fill_diagonal(table, rng.uniform(50, 80, K))
        names = ["A", "B", "C", "D"]

        fig_net = plot_network(spillover_table=table, var_names=names)
        assert isinstance(fig_net, Figure)
        plt.close(fig_net)

        fig_heat = plot_heatmap(spillover_table=table, var_names=names)
        assert isinstance(fig_heat, Figure)
        plt.close(fig_heat)

        rolling = 50 + 10 * np.sin(np.arange(200) * 0.05) + rng.normal(0, 2, 200)
        fig_roll = plot_rolling(rolling_total=rolling)
        assert isinstance(fig_roll, Figure)
        plt.close(fig_roll)

    def test_tvp_coefs_integration(self, rng: np.random.Generator) -> None:
        T, n = 200, 4
        coefs = np.cumsum(rng.normal(0, 0.01, (T, n)), axis=0) + np.array([1.0, -0.5, 0.3, 0.0])
        ols = np.mean(coefs, axis=0)
        fig = plot_tvp_coefs(coefs=coefs, ols_coefs=ols)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_cusum_integration(self, rng: np.random.Generator) -> None:
        cusum = np.cumsum(rng.normal(0, 1, 100))
        fig = plot_cusum(cusum_path=cusum)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_bai_perron_integration(self, rng: np.random.Generator) -> None:
        y = np.concatenate([rng.normal(10, 1, 80), rng.normal(15, 1, 60), rng.normal(8, 1, 60)])
        fig = plot_bai_perron(y=y, break_dates=[80, 140])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_zivot_andrews_integration(self, rng: np.random.Generator) -> None:
        y = np.concatenate([np.cumsum(rng.normal(0, 1, 100)), np.cumsum(rng.normal(0.5, 1, 100))])
        fig = plot_zivot_andrews(y=y, break_index=100)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_recursive_coefs_integration(self, rng: np.random.Generator) -> None:
        coefs = np.cumsum(rng.normal(0, 0.1, (150, 3)), axis=0)
        fig = plot_recursive_coefs(coefs=coefs)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestExportIntegration:
    """Test export pipeline: plot -> export to file in all formats."""

    def test_export_all_formats(self, sample_series: np.ndarray) -> None:
        """Export a figure to PNG, SVG, PDF, and HTML."""
        fig = plot_series(sample_series)
        with tempfile.TemporaryDirectory() as tmpdir:
            for fmt in ["png", "svg", "pdf", "html"]:
                path = save_figure(fig, Path(tmpdir) / f"test.{fmt}", fmt=fmt)
                assert path.exists(), f"Export to {fmt} failed"
                assert path.stat().st_size > 0, f"Export to {fmt} produced empty file"
        plt.close(fig)

    def test_figure_to_html_contains_image(self, sample_series: np.ndarray) -> None:
        """figure_to_html produces HTML with embedded base64 image."""
        fig = plot_series(sample_series)
        html = figure_to_html(fig)
        assert "<img" in html
        assert "base64" in html
        plt.close(fig)

    def test_theme_then_export(self, sample_series: np.ndarray) -> None:
        """Apply theme then export works correctly."""
        for theme in list_themes():
            set_theme(theme)
            fig = plot_series(sample_series)
            with tempfile.TemporaryDirectory() as tmpdir:
                path = save_figure(fig, Path(tmpdir) / "test.png")
                assert path.exists()
                assert path.stat().st_size > 0
            plt.close(fig)

    def test_figure_to_base64(self, sample_series: np.ndarray) -> None:
        """figure_to_base64 produces valid base64 string."""
        from chronobox.visualization.export import figure_to_base64

        fig = plot_series(sample_series)
        b64 = figure_to_base64(fig, fmt="png", dpi=100)
        assert len(b64) > 100
        plt.close(fig)

    def test_figures_to_html_gallery(self, sample_series: np.ndarray, rng: np.random.Generator) -> None:
        """figures_to_html_gallery creates multi-figure HTML page."""
        from chronobox.visualization.export import figures_to_html_gallery

        fig1 = plot_series(sample_series)
        resid = rng.normal(0, 1, 200)
        fig2 = plot_diagnostics(residuals=resid)
        html = figures_to_html_gallery([("Series", fig1), ("Diagnostics", fig2)])
        assert "<h1>" in html
        assert "Series" in html
        assert "Diagnostics" in html
        assert "base64" in html
        plt.close(fig1)
        plt.close(fig2)

    def test_figure_to_html_not_full(self, sample_series: np.ndarray) -> None:
        """figure_to_html with full_html=False returns just the div."""
        fig = plot_series(sample_series)
        html = figure_to_html(fig, full_html=False)
        assert "<img" in html
        assert "<!DOCTYPE" not in html
        plt.close(fig)

    def test_figure_to_html_with_div_id(self, sample_series: np.ndarray) -> None:
        """figure_to_html with custom div_id."""
        fig = plot_series(sample_series)
        html = figure_to_html(fig, div_id="my-chart")
        assert 'id="my-chart"' in html
        plt.close(fig)

    def test_save_figure_infers_format(self, sample_series: np.ndarray) -> None:
        """save_figure infers format from file extension."""
        fig = plot_series(sample_series)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_figure(fig, Path(tmpdir) / "test.png")
            assert path.exists()
        plt.close(fig)

    def test_save_figure_with_dpi(self, sample_series: np.ndarray) -> None:
        """save_figure respects dpi parameter."""
        fig = plot_series(sample_series)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_figure(fig, Path(tmpdir) / "test.png", dpi=72)
            assert path.exists()
            assert path.stat().st_size > 0
        plt.close(fig)

    def test_plot_series_with_options(self, sample_series: np.ndarray) -> None:
        """plot_series works with various kwargs."""
        fig = plot_series(
            sample_series,
            title="My Chart",
            xlabel="Time",
            ylabel="Value",
            grid=True,
            legend=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_series_dataframe(self, rng: np.random.Generator) -> None:
        """plot_series works with pandas DataFrame input."""
        import pandas as pd

        df = pd.DataFrame({
            "GDP": rng.normal(100, 10, 100),
            "CPI": rng.normal(50, 5, 100),
        })
        fig = plot_series(df, title="Multi-Series")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_series_with_secondary_y(self, rng: np.random.Generator) -> None:
        """plot_series with secondary y-axis."""
        import pandas as pd

        df = pd.DataFrame({
            "GDP": rng.normal(100, 10, 100),
            "Rate": rng.normal(5, 1, 100),
        })
        fig = plot_series(df, secondary_y=["Rate"], ylabel="GDP", secondary_ylabel="Rate")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_series_with_annotations(self, sample_series: np.ndarray) -> None:
        """plot_series with annotations."""
        annotations = [
            {"text": "Peak", "x": 50, "y": float(sample_series[50])},
        ]
        fig = plot_series(sample_series, annotations=annotations)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_series_with_recessions(self, sample_series: np.ndarray) -> None:
        """plot_series with recession bars."""
        fig = plot_series(sample_series, recessions=[(30, 50), (100, 120)])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_series_pandas_series(self, rng: np.random.Generator) -> None:
        """plot_series with pandas Series input."""
        import pandas as pd

        s = pd.Series(rng.normal(0, 1, 100), name="TestSeries")
        fig = plot_series(s)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_forecast_with_results_params(self, sample_series: np.ndarray) -> None:
        """plot_forecast with explicit ci_levels."""
        n_fc = 12
        fc_mean = np.full(n_fc, sample_series[-1])
        fc_se = np.sqrt(np.arange(1, n_fc + 1)) * 3.0
        fig = plot_forecast(
            history=sample_series,
            forecast_mean=fc_mean,
            forecast_se=fc_se,
            ci_levels=[0.5, 0.9],
            title="Forecast",
            xlabel="Time",
            ylabel="Value",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_irf_with_ci(self, rng: np.random.Generator) -> None:
        """plot_irf with confidence intervals."""
        H, K = 20, 2
        irf = np.zeros((H, K, K))
        for i in range(K):
            for j in range(K):
                irf[:, i, j] = np.exp(-np.arange(H) * 0.2) * (1.0 if i == j else 0.3)
        irf_lower = irf - 0.5
        irf_upper = irf + 0.5
        fig = plot_irf(
            irf_array=irf,
            irf_lower=irf_lower,
            irf_upper=irf_upper,
            var_names=["X", "Y"],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_irf_cumulative(self, rng: np.random.Generator) -> None:
        """plot_irf with cumulative=True."""
        H, K = 20, 2
        irf = rng.normal(0, 0.3, (H, K, K))
        fig = plot_irf(irf_array=irf, var_names=["X", "Y"], cumulative=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_irf_with_periods(self, rng: np.random.Generator) -> None:
        """plot_irf with periods limit."""
        H, K = 30, 2
        irf = rng.normal(0, 0.3, (H, K, K))
        fig = plot_irf(irf_array=irf, var_names=["X", "Y"], periods=15)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_fevd_bar(self, rng: np.random.Generator) -> None:
        """plot_fevd with bar plot type."""
        H, K = 20, 3
        fevd = rng.dirichlet(np.ones(K), size=(H, K))
        fig = plot_fevd(fevd_array=fevd, var_names=["Y1", "Y2", "Y3"], plot_type="bar")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_hd_specific_variable(self, rng: np.random.Generator) -> None:
        """plot_hd for a specific variable."""
        T, K = 100, 3
        hd = rng.normal(0, 5, (T, K, K))
        observed = np.sum(hd, axis=2)
        fig = plot_hd(hd_array=hd, observed=observed, var_names=["Y1", "Y2", "Y3"], variable=0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_decomposition_subset(self, sample_series: np.ndarray) -> None:
        """plot_decomposition with which parameter."""
        n = len(sample_series)
        t = np.arange(n, dtype=np.float64)
        components = {
            "observed": sample_series,
            "trend": 100 + 0.5 * t,
            "seasonal": 10 * np.sin(2 * np.pi * t / 12),
            "remainder": sample_series - (100 + 0.5 * t) - 10 * np.sin(2 * np.pi * t / 12),
        }
        fig = plot_decomposition(components=components, which=["trend", "seasonal"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_tvp_coefs_with_ci(self, rng: np.random.Generator) -> None:
        """plot_tvp_coefs with confidence intervals."""
        T, n = 200, 3
        coefs = np.cumsum(rng.normal(0, 0.01, (T, n)), axis=0)
        coef_lower = coefs - 0.5
        coef_upper = coefs + 0.5
        fig = plot_tvp_coefs(coefs=coefs, coef_lower=coef_lower, coef_upper=coef_upper, coef_names=["a", "b", "c"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_tvp_coefs_1d(self, rng: np.random.Generator) -> None:
        """plot_tvp_coefs with 1D coefs array."""
        coefs = np.cumsum(rng.normal(0, 0.01, 200))
        fig = plot_tvp_coefs(coefs=coefs)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_tvp_coefs_max_coefs(self, rng: np.random.Generator) -> None:
        """plot_tvp_coefs with max_coefs limit."""
        T = 200
        coefs = np.cumsum(rng.normal(0, 0.01, (T, 6)), axis=0)
        fig = plot_tvp_coefs(coefs=coefs, max_coefs=3)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_tvp_coefs_with_title(self, rng: np.random.Generator) -> None:
        """plot_tvp_coefs with custom title and 5 coefs (unused subplots)."""
        T = 200
        coefs = np.cumsum(rng.normal(0, 0.01, (T, 5)), axis=0)
        fig = plot_tvp_coefs(coefs=coefs, title="TVP Estimates")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_forecast_no_se(self, sample_series: np.ndarray) -> None:
        """plot_forecast without explicit forecast_se."""
        fc_mean = np.full(12, sample_series[-1])
        fig = plot_forecast(history=sample_series, forecast_mean=fc_mean)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_network_with_threshold(self, rng: np.random.Generator) -> None:
        """plot_network with high threshold."""
        K = 4
        table = rng.uniform(5, 30, (K, K))
        np.fill_diagonal(table, rng.uniform(50, 80, K))
        fig = plot_network(spillover_table=table, var_names=["A", "B", "C", "D"], threshold=20.0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_heatmap_custom_cmap(self, rng: np.random.Generator) -> None:
        """plot_heatmap with custom colormap."""
        K = 3
        table = rng.uniform(5, 30, (K, K))
        fig = plot_heatmap(spillover_table=table, var_names=["A", "B", "C"], cmap="viridis")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_rolling_with_dates(self, rng: np.random.Generator) -> None:
        """plot_rolling with dates parameter."""
        import pandas as pd

        rolling = 50 + 10 * np.sin(np.arange(200) * 0.05) + rng.normal(0, 2, 200)
        dates = pd.date_range("2010-01-01", periods=200, freq="ME")
        fig = plot_rolling(rolling_total=rolling, dates=dates)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_bai_perron_with_title(self, rng: np.random.Generator) -> None:
        """plot_bai_perron with custom title."""
        y = np.concatenate([rng.normal(10, 1, 80), rng.normal(15, 1, 60)])
        fig = plot_bai_perron(y=y, break_dates=[80], title="Break Test")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_zivot_andrews_with_stats(self, rng: np.random.Generator) -> None:
        """plot_zivot_andrews with stat and critical values."""
        y = np.concatenate([np.cumsum(rng.normal(0, 1, 100)), np.cumsum(rng.normal(0.5, 1, 100))])
        fig = plot_zivot_andrews(
            y=y,
            break_index=100,
            za_stat=-4.5,
            critical_values={"1%": -5.34, "5%": -4.80, "10%": -4.58},
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_fevd_single_variable(self, rng: np.random.Generator) -> None:
        """plot_fevd for a single variable."""
        H, K = 20, 3
        fevd = rng.dirichlet(np.ones(K), size=(H, K))
        fig = plot_fevd(fevd_array=fevd, var_names=["Y1", "Y2", "Y3"], variable=0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_hd_2d_array(self, rng: np.random.Generator) -> None:
        """plot_hd with 2D array (single variable)."""
        T, K = 100, 3
        hd = rng.normal(0, 5, (T, K))
        observed = np.sum(hd, axis=1)
        fig = plot_hd(hd_array=hd, observed=observed, shock_names=["S1", "S2", "S3"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_hd_with_title(self, rng: np.random.Generator) -> None:
        """plot_hd with custom title."""
        T, K = 100, 3
        hd = rng.normal(0, 5, (T, K, K))
        observed = np.sum(hd, axis=2)
        fig = plot_hd(hd_array=hd, observed=observed, var_names=["Y1", "Y2", "Y3"], title="HD Test")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_forecast_on_existing_ax(self, sample_series: np.ndarray) -> None:
        """plot_forecast on an existing axes."""
        fig_outer, ax = plt.subplots()
        fc_mean = np.full(12, sample_series[-1])
        fc_se = np.sqrt(np.arange(1, 13)) * 3.0
        fig = plot_forecast(history=sample_series, forecast_mean=fc_mean, forecast_se=fc_se, ax=ax)
        assert fig is fig_outer
        plt.close(fig)

    def test_plot_series_on_existing_ax(self, sample_series: np.ndarray) -> None:
        """plot_series on an existing axes."""
        fig_outer, ax = plt.subplots()
        fig = plot_series(sample_series, ax=ax)
        assert fig is fig_outer
        plt.close(fig)

    def test_plot_decomposition_with_title(self, sample_series: np.ndarray) -> None:
        """plot_decomposition with title and dates."""
        import pandas as pd

        n = len(sample_series)
        t = np.arange(n, dtype=np.float64)
        components = {
            "observed": sample_series,
            "trend": 100 + 0.5 * t,
        }
        dates = pd.date_range("2000-01-01", periods=n, freq="ME")
        fig = plot_decomposition(components=components, title="Decomposition", dates=dates)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_series_list_of_arrays(self, rng: np.random.Generator) -> None:
        """plot_series with list of arrays."""
        data = [rng.normal(0, 1, 100), rng.normal(1, 1, 100)]
        fig = plot_series(data, labels=["Series A", "Series B"])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_cusum_with_bands(self, rng: np.random.Generator) -> None:
        """plot_cusum with explicit upper/lower bands."""
        n = 100
        cusum = np.cumsum(rng.normal(0, 1, n))
        upper = np.linspace(1, 3, n)
        lower = -upper
        fig = plot_cusum(cusum_path=cusum, upper_band=upper, lower_band=lower)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_recursive_coefs_with_se(self, rng: np.random.Generator) -> None:
        """plot_recursive_coefs with standard errors."""
        coefs = np.cumsum(rng.normal(0, 0.1, (150, 3)), axis=0)
        coef_se = np.abs(rng.normal(0.1, 0.02, (150, 3)))
        fig = plot_recursive_coefs(coefs=coefs, coef_se=coef_se, coef_names=["a", "b", "c"])
        assert isinstance(fig, Figure)
        plt.close(fig)
