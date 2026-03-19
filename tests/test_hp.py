"""Tests for the HP filter."""

import numpy as np
import pytest

from chronobox.filters.hp import hp_filter


class TestHPFilter:
    """Tests for hp_filter."""

    def setup_method(self) -> None:
        """Set up test data."""
        np.random.seed(42)
        self.T = 200
        # Random walk + noise (realistic macro series)
        self.y = np.random.randn(self.T).cumsum()

    def test_trend_plus_cycle_equals_original(self) -> None:
        """trend + cycle must reconstruct the original series exactly."""
        trend, cycle = hp_filter(self.y, lamb=1600)
        np.testing.assert_allclose(trend + cycle, self.y, atol=1e-10)

    def test_output_gap_zero_mean(self) -> None:
        """The cycle (output gap) should have approximately zero mean."""
        _trend, cycle = hp_filter(self.y, lamb=1600)
        assert abs(np.mean(cycle)) < 1.0  # not strict, but should be small

    def test_trend_smooth(self) -> None:
        """The trend should be smoother than the original series."""
        trend, _cycle = hp_filter(self.y, lamb=1600)
        # Second differences of trend should be smaller than original
        d2_trend = np.diff(trend, n=2)
        d2_y = np.diff(self.y, n=2)
        assert np.std(d2_trend) < np.std(d2_y)

    def test_lambda_quarterly_default(self) -> None:
        """Default lambda for quarterly data should be 1600."""
        trend1, _ = hp_filter(self.y, lamb=1600)
        trend2, _ = hp_filter(self.y, frequency="quarterly")
        np.testing.assert_allclose(trend1, trend2)

    def test_lambda_annual(self) -> None:
        """Annual lambda should be 6.25."""
        trend, _ = hp_filter(self.y, lamb=6.25)
        trend_a, _ = hp_filter(self.y, frequency="annual")
        np.testing.assert_allclose(trend, trend_a)

    def test_lambda_monthly(self) -> None:
        """Monthly lambda should be 129600."""
        trend, _ = hp_filter(self.y, lamb=129600)
        trend_m, _ = hp_filter(self.y, frequency="monthly")
        np.testing.assert_allclose(trend, trend_m)

    def test_lambda_zero_trend_equals_series(self) -> None:
        """With lambda=0, trend should equal the original series."""
        trend, cycle = hp_filter(self.y, lamb=0)
        np.testing.assert_allclose(trend, self.y, atol=1e-10)
        np.testing.assert_allclose(cycle, 0.0, atol=1e-10)

    def test_large_lambda_linear_trend(self) -> None:
        """With very large lambda, trend should approach a linear trend."""
        trend, _ = hp_filter(self.y, lamb=1e12)
        # Check that the trend is approximately linear
        x = np.arange(len(self.y))
        coeffs = np.polyfit(x, trend, 1)
        fitted_linear = np.polyval(coeffs, x)
        np.testing.assert_allclose(trend, fitted_linear, atol=0.1)

    def test_output_shapes(self) -> None:
        """Output shapes should match input."""
        trend, cycle = hp_filter(self.y, lamb=1600)
        assert trend.shape == (self.T,)
        assert cycle.shape == (self.T,)

    def test_invalid_1d(self) -> None:
        """Should raise ValueError for non-1D input."""
        with pytest.raises(ValueError, match="1-D"):
            hp_filter(np.random.randn(10, 2), lamb=1600)

    def test_too_short(self) -> None:
        """Should raise ValueError for series shorter than 4."""
        with pytest.raises(ValueError, match="at least 4"):
            hp_filter(np.array([1.0, 2.0, 3.0]), lamb=1600)

    def test_negative_lambda(self) -> None:
        """Should raise ValueError for negative lambda."""
        with pytest.raises(ValueError, match="non-negative"):
            hp_filter(self.y, lamb=-1.0)

    def test_unknown_frequency(self) -> None:
        """Should raise ValueError for unknown frequency string."""
        with pytest.raises(ValueError, match="Unknown frequency"):
            hp_filter(self.y, frequency="weekly")  # type: ignore[reportArgumentType]

    def test_vs_known_values(self) -> None:
        """Compare with a simple known case: constant series."""
        y_const = np.ones(50) * 5.0
        trend, cycle = hp_filter(y_const, lamb=1600)
        np.testing.assert_allclose(trend, 5.0, atol=1e-10)
        np.testing.assert_allclose(cycle, 0.0, atol=1e-10)

    def test_linear_series(self) -> None:
        """For a linear series, trend should equal the series (no cycle)."""
        y_linear = np.linspace(0, 10, 100)
        trend, cycle = hp_filter(y_linear, lamb=1600)
        # Linear series has zero second differences, so penalty is zero
        np.testing.assert_allclose(trend, y_linear, atol=1e-8)
        np.testing.assert_allclose(cycle, 0.0, atol=1e-8)
