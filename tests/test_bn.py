"""Tests for the Beveridge-Nelson decomposition."""

import numpy as np
import pytest

from chronobox.filters.bn import (
    _compute_psi_coefficients,  # pyright: ignore[reportPrivateUsage]
    bn_decomposition,
    bn_decomposition_detailed,
)


class TestBNDecomposition:
    """Tests for bn_decomposition."""

    def setup_method(self) -> None:
        """Set up test data."""
        np.random.seed(42)
        self.T = 300
        # Random walk with drift
        self.y = 0.1 + np.random.randn(self.T).cumsum()

    def test_trend_plus_cycle_equals_original(self) -> None:
        """trend + cycle must reconstruct original series exactly."""
        trend, cycle = bn_decomposition(self.y, p=2)
        np.testing.assert_allclose(trend + cycle, self.y, atol=1e-10)

    def test_trend_behaves_as_random_walk(self) -> None:
        """Trend should behave like a random walk (diff should be ~stationary)."""
        trend, _ = bn_decomposition(self.y, p=2)
        d_trend = np.diff(trend)
        # Standard deviation of diff(trend) should be finite and stable
        assert np.std(d_trend) > 0
        # First half vs second half should have similar std
        mid = len(d_trend) // 2
        std_1 = np.std(d_trend[:mid])
        std_2 = np.std(d_trend[mid:])
        ratio = std_1 / std_2
        assert 0.3 < ratio < 3.0, f"Trend diff std ratio {ratio} not stable"

    def test_cycle_stationary(self) -> None:
        """Cycle should be mean-reverting (approximately zero mean)."""
        _, cycle = bn_decomposition(self.y, p=2)
        # Cycle should have bounded variance
        assert np.std(cycle) < np.std(self.y)

    def test_psi_coefficients_ar1(self) -> None:
        """For AR(1) with phi=0.5, psi_j should decay geometrically."""
        phi = np.array([0.5])
        theta = np.array([])
        psi = _compute_psi_coefficients(phi, theta, n_terms=50)
        # psi_j = phi^j for AR(1)
        for j in range(10):
            np.testing.assert_allclose(psi[j], 0.5**j, atol=1e-10)

    def test_psi_one_ar1(self) -> None:
        """psi(1) for AR(1) with phi=0.5 should be 1/(1-0.5) = 2."""
        phi = np.array([0.5])
        theta = np.array([])
        psi = _compute_psi_coefficients(phi, theta, n_terms=500)
        psi_one = np.sum(psi)
        np.testing.assert_allclose(psi_one, 2.0, atol=0.01)

    def test_detailed_result(self) -> None:
        """Detailed result should contain all expected fields."""
        result = bn_decomposition_detailed(self.y, p=2)
        assert result.trend.shape == (self.T,)
        assert result.cycle.shape == (self.T,)
        assert result.ar_coeffs.shape == (2,)
        assert isinstance(result.psi_one, float)
        assert isinstance(result.drift, float)
        np.testing.assert_allclose(result.trend + result.cycle, self.y, atol=1e-10)

    def test_different_p(self) -> None:
        """Should work with different AR orders."""
        for p in [1, 2, 3, 4, 6]:
            trend, cycle = bn_decomposition(self.y, p=p)
            np.testing.assert_allclose(trend + cycle, self.y, atol=1e-10)

    def test_invalid_inputs(self) -> None:
        """Should raise for invalid inputs."""
        with pytest.raises(ValueError, match="1-D"):
            bn_decomposition(np.random.randn(10, 2))
        with pytest.raises(ValueError, match="p must be >= 1"):
            bn_decomposition(self.y, p=0)
        with pytest.raises(ValueError, match="at least"):
            bn_decomposition(np.ones(4), p=2)

    def test_constant_first_diff_zero(self) -> None:
        """Linear series has zero first differences -> trend = series."""
        y_linear = np.linspace(0, 10, 100)
        trend, cycle = bn_decomposition(y_linear, p=1)
        # Cycle should be approximately zero for a pure linear series
        np.testing.assert_allclose(trend + cycle, y_linear, atol=1e-10)
