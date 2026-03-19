"""Tests for HD, Spillover, Coef, and Test plots.

Tests:
    - plot_hd generates stacked bars + observed line
    - plot_network generates graph without error
    - plot_heatmap generates heatmap
    - plot_rolling generates line chart
    - plot_tvp_coefs generates panels
    - plot_cusum generates path with bands
    - plot_bai_perron marks break points
    - plot_zivot_andrews marks single break
    - plot_recursive_coefs generates panels
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from chronobox.visualization.coef_plot import plot_tvp_coefs
from chronobox.visualization.hd_plot import plot_hd
from chronobox.visualization.spillover_plot import plot_heatmap, plot_network, plot_rolling
from chronobox.visualization.test_plot import (
    plot_bai_perron,
    plot_cusum,
    plot_recursive_coefs,
    plot_zivot_andrews,
)


@pytest.fixture
def sample_hd_data() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Generate sample HD data: (T, K_var, K_shock)."""
    rng = np.random.default_rng(42)
    T, K = 100, 3
    hd = rng.normal(0, 5, (T, K, K))
    observed = np.sum(hd, axis=2)  # Observed = sum of all shock contributions
    var_names = ["GDP", "Inflation", "Rate"]
    shock_names = ["Supply", "Demand", "Monetary"]
    return hd, observed, var_names, shock_names


@pytest.fixture
def sample_spillover_table() -> tuple[np.ndarray, list[str]]:
    """Generate sample spillover table (K x K)."""
    rng = np.random.default_rng(42)
    K = 4
    table = rng.uniform(0, 30, (K, K))
    np.fill_diagonal(table, rng.uniform(40, 80, K))
    var_names = ["GDP", "CPI", "Rate", "Exchange"]
    return table, var_names


class TestPlotHD:
    """Tests for plot_hd."""

    def test_stacked_bars_and_observed(self, sample_hd_data: tuple) -> None:
        """plot_hd generates figure with stacked bars and observed line."""
        hd, observed, var_names, shock_names = sample_hd_data
        fig = plot_hd(
            hd_array=hd,
            observed=observed,
            var_names=var_names,
            shock_names=shock_names,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_single_variable(self, sample_hd_data: tuple) -> None:
        """plot_hd can plot a single variable."""
        hd, observed, var_names, shock_names = sample_hd_data
        fig = plot_hd(
            hd_array=hd,
            observed=observed,
            var_names=var_names,
            shock_names=shock_names,
            variable="GDP",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_raises_without_data(self) -> None:
        """plot_hd raises ValueError without data."""
        with pytest.raises(ValueError):
            plot_hd()


class TestSpilloverPlots:
    """Tests for spillover visualizations."""

    def test_network_generates_graph(self, sample_spillover_table: tuple) -> None:
        """plot_network generates graph without error."""
        table, var_names = sample_spillover_table
        fig = plot_network(spillover_table=table, var_names=var_names)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_heatmap_generates(self, sample_spillover_table: tuple) -> None:
        """plot_heatmap generates annotated heatmap."""
        table, var_names = sample_spillover_table
        fig = plot_heatmap(spillover_table=table, var_names=var_names)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_rolling_generates(self) -> None:
        """plot_rolling generates line chart."""
        rng = np.random.default_rng(42)
        rolling = 50 + 10 * np.sin(np.arange(200) * 0.05) + rng.normal(0, 2, 200)
        fig = plot_rolling(rolling_total=rolling)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotTVPCoefs:
    """Tests for plot_tvp_coefs."""

    def test_generates_panels(self) -> None:
        """plot_tvp_coefs generates coefficient panels."""
        rng = np.random.default_rng(42)
        T, n_coefs = 200, 4
        coefs = rng.normal(0, 1, (T, n_coefs))
        coefs = np.cumsum(coefs * 0.01, axis=0) + np.array([1.0, -0.5, 0.3, 0.0])
        ols_coefs = np.mean(coefs, axis=0)
        fig = plot_tvp_coefs(
            coefs=coefs,
            ols_coefs=ols_coefs,
            coef_names=["beta_1", "beta_2", "phi_1", "const"],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_confidence_bands(self) -> None:
        """plot_tvp_coefs shows confidence bands."""
        rng = np.random.default_rng(42)
        T, n_coefs = 100, 2
        coefs = rng.normal(0, 1, (T, n_coefs)).cumsum(axis=0) * 0.01
        lower = coefs - 0.5
        upper = coefs + 0.5
        fig = plot_tvp_coefs(coefs=coefs, coef_lower=lower, coef_upper=upper)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestTestPlots:
    """Tests for statistical test visualizations."""

    def test_cusum_generates(self) -> None:
        """plot_cusum generates path with bands."""
        rng = np.random.default_rng(42)
        n = 100
        cusum = np.cumsum(rng.normal(0, 1, n))
        fig = plot_cusum(cusum_path=cusum)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_bai_perron_marks_breaks(self) -> None:
        """plot_bai_perron shows break points."""
        rng = np.random.default_rng(42)
        y = np.concatenate([
            rng.normal(10, 1, 70),
            rng.normal(15, 1, 60),
            rng.normal(8, 1, 70),
        ])
        fig = plot_bai_perron(y=y, break_dates=[70, 130])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_zivot_andrews_marks_break(self) -> None:
        """plot_zivot_andrews shows single break."""
        rng = np.random.default_rng(42)
        y = np.concatenate([
            np.cumsum(rng.normal(0, 1, 100)),
            np.cumsum(rng.normal(0.5, 1, 100)),
        ])
        fig = plot_zivot_andrews(y=y, break_index=100, za_stat=-4.5)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_recursive_coefs_generates(self) -> None:
        """plot_recursive_coefs generates panels."""
        rng = np.random.default_rng(42)
        T, n_coefs = 150, 3
        coefs = np.cumsum(rng.normal(0, 0.1, (T, n_coefs)), axis=0)
        fig = plot_recursive_coefs(coefs=coefs, coef_names=["a", "b", "c"])
        assert isinstance(fig, Figure)
        plt.close(fig)
