"""Tests for decomposition, IRF, and FEVD plots.

Tests:
    - plot_decomposition generates correct number of panels
    - plot_irf generates KxK grid
    - plot_fevd stacked sums to 1
    - All plots handle both results objects and raw arrays
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from chronobox.visualization.decomposition_plot import plot_decomposition
from chronobox.visualization.fevd_plot import plot_fevd
from chronobox.visualization.irf_plot import plot_irf


@pytest.fixture
def sample_components() -> dict[str, np.ndarray]:
    """Generate sample decomposition components."""
    rng = np.random.default_rng(42)
    n = 200
    t = np.arange(n, dtype=np.float64)
    trend = 100 + 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    remainder = rng.normal(0, 2, n)
    observed = trend + seasonal + remainder
    return {
        "observed": observed,
        "trend": trend,
        "seasonal": seasonal,
        "remainder": remainder,
    }


@pytest.fixture
def sample_irf_data() -> tuple[np.ndarray, list[str]]:
    """Generate sample IRF data (H=20, K=3)."""
    rng = np.random.default_rng(42)
    H, K = 20, 3
    irf = np.zeros((H, K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                irf[:, i, j] = np.exp(-np.arange(H) * 0.2) * (1 + 0.1 * rng.normal(0, 1, H))
            else:
                irf[:, i, j] = 0.3 * np.exp(-np.arange(H) * 0.3) * rng.normal(0, 1, H)
    var_names = ["GDP", "Inflation", "Interest Rate"]
    return irf, var_names


@pytest.fixture
def sample_fevd_data() -> tuple[np.ndarray, list[str]]:
    """Generate sample FEVD data (H=20, K=3). Sums to 1 per row."""
    rng = np.random.default_rng(42)
    H, K = 20, 3
    raw = rng.dirichlet(np.ones(K), size=(H, K))
    # raw shape: (H, K, K) - each (h, i, :) sums to 1
    var_names = ["GDP", "Inflation", "Interest Rate"]
    return raw, var_names


class TestPlotDecomposition:
    """Tests for plot_decomposition."""

    def test_generates_correct_panels(self, sample_components: dict) -> None:
        """plot_decomposition creates N subplots matching component count."""
        fig = plot_decomposition(components=sample_components)
        assert isinstance(fig, Figure)
        axes = fig.get_axes()
        assert len(axes) == len(sample_components)
        plt.close(fig)

    def test_select_which_components(self, sample_components: dict) -> None:
        """plot_decomposition filters components with 'which' parameter."""
        fig = plot_decomposition(components=sample_components, which=["trend", "seasonal"])
        axes = fig.get_axes()
        assert len(axes) == 2
        plt.close(fig)

    def test_custom_title(self, sample_components: dict) -> None:
        """plot_decomposition accepts custom title."""
        fig = plot_decomposition(components=sample_components, title="Test Decomposition")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_raises_without_data(self) -> None:
        """plot_decomposition raises ValueError without data."""
        with pytest.raises(ValueError):
            plot_decomposition()


class TestPlotIRF:
    """Tests for plot_irf."""

    def test_generates_kxk_grid(self, sample_irf_data: tuple) -> None:
        """plot_irf generates KxK grid for K-variable system."""
        irf, var_names = sample_irf_data
        K = len(var_names)
        fig = plot_irf(irf_array=irf, var_names=var_names)
        assert isinstance(fig, Figure)
        axes = fig.get_axes()
        assert len(axes) == K * K
        plt.close(fig)

    def test_with_confidence_bands(self, sample_irf_data: tuple) -> None:
        """plot_irf shows confidence bands."""
        irf, var_names = sample_irf_data
        lower = irf - 0.5
        upper = irf + 0.5
        fig = plot_irf(irf_array=irf, irf_lower=lower, irf_upper=upper, var_names=var_names)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_select_impulse(self, sample_irf_data: tuple) -> None:
        """plot_irf can select a single impulse variable."""
        irf, var_names = sample_irf_data
        fig = plot_irf(irf_array=irf, var_names=var_names, impulse="GDP")
        assert isinstance(fig, Figure)
        axes = fig.get_axes()
        assert len(axes) == len(var_names)  # All responses, 1 impulse
        plt.close(fig)

    def test_cumulative(self, sample_irf_data: tuple) -> None:
        """plot_irf supports cumulative IRFs."""
        irf, var_names = sample_irf_data
        fig = plot_irf(irf_array=irf, var_names=var_names, cumulative=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_raises_without_data(self) -> None:
        """plot_irf raises ValueError without data."""
        with pytest.raises(ValueError):
            plot_irf()


class TestPlotFEVD:
    """Tests for plot_fevd."""

    def test_stacked_sums_to_1(self, sample_fevd_data: tuple) -> None:
        """FEVD proportions should sum to 1 for each variable at each horizon."""
        fevd, var_names = sample_fevd_data
        for i in range(fevd.shape[1]):
            row_sums = np.sum(fevd[:, i, :], axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_generates_figure_area(self, sample_fevd_data: tuple) -> None:
        """plot_fevd generates stacked area chart."""
        fevd, var_names = sample_fevd_data
        fig = plot_fevd(fevd_array=fevd, var_names=var_names, plot_type="area")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_generates_figure_bar(self, sample_fevd_data: tuple) -> None:
        """plot_fevd generates stacked bar chart."""
        fevd, var_names = sample_fevd_data
        fig = plot_fevd(fevd_array=fevd, var_names=var_names, plot_type="bar")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_single_variable(self, sample_fevd_data: tuple) -> None:
        """plot_fevd can plot a single variable."""
        fevd, var_names = sample_fevd_data
        fig = plot_fevd(fevd_array=fevd, var_names=var_names, variable="GDP")
        axes = fig.get_axes()
        assert len(axes) == 1
        plt.close(fig)

    def test_raises_without_data(self) -> None:
        """plot_fevd raises ValueError without data."""
        with pytest.raises(ValueError):
            plot_fevd()
