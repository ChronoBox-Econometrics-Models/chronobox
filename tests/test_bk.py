"""Tests for the Baxter-King band-pass filter."""

import numpy as np
import pytest

from chronobox.filters.bk import bk_filter, bk_filter_weights


class TestBKFilter:
    """Tests for bk_filter."""

    def setup_method(self) -> None:
        """Set up test data."""
        np.random.seed(42)
        self.T = 200
        self.y = np.random.randn(self.T).cumsum()

    def test_zero_sum_weights(self) -> None:
        """BK filter weights must sum to zero (removes trend)."""
        weights = bk_filter_weights(low=6, high=32, trunc=12)
        np.testing.assert_allclose(np.sum(weights), 0.0, atol=1e-14)

    def test_loses_observations(self) -> None:
        """BK filter loses K observations at each end."""
        k = 12
        cycle = bk_filter(self.y, low=6, high=32, trunc=k)
        assert len(cycle) == self.T - 2 * k

    def test_business_cycle_isolation(self) -> None:
        """Should isolate business cycle frequencies."""
        # Create series with known cycle: period=16 quarters (4 years)
        t = np.arange(500)
        cycle_true = np.sin(2 * np.pi * t / 16)  # period 16
        trend = 0.01 * t
        noise = 0.1 * np.random.randn(500)  # high frequency
        y = trend + cycle_true + noise

        cycle_bk = bk_filter(y, low=6, high=32, trunc=12)

        # The filtered cycle should correlate with the true cycle
        # (after trimming K observations from each end)
        true_trimmed = cycle_true[12:-12]
        corr = np.corrcoef(cycle_bk, true_trimmed)[0, 1]
        assert corr > 0.8, f"Correlation {corr} too low"

    def test_symmetric_weights(self) -> None:
        """BK filter weights should be symmetric."""
        weights = bk_filter_weights(low=6, high=32, trunc=12)
        n = len(weights)
        for i in range(n // 2):
            np.testing.assert_allclose(weights[i], weights[n - 1 - i], atol=1e-14)

    def test_output_shape(self) -> None:
        """Output shape check."""
        k = 8
        cycle = bk_filter(self.y, low=6, high=32, trunc=k)
        assert cycle.shape == (self.T - 2 * k,)

    def test_invalid_1d(self) -> None:
        """Should raise for non-1D input."""
        with pytest.raises(ValueError, match="1-D"):
            bk_filter(np.random.randn(10, 2))

    def test_series_too_short(self) -> None:
        """Should raise if T <= 2*K."""
        with pytest.raises(ValueError, match="must be >"):
            bk_filter(np.ones(20), trunc=12)

    def test_low_ge_high(self) -> None:
        """Should raise if low >= high."""
        with pytest.raises(ValueError, match="must be > low"):
            bk_filter(self.y, low=32, high=6)

    def test_low_less_than_2(self) -> None:
        """Should raise if low < 2."""
        with pytest.raises(ValueError, match="low must be >= 2"):
            bk_filter(self.y, low=1, high=32)

    def test_constant_series_zero_cycle(self) -> None:
        """Constant series should have zero cycle."""
        y_const = np.ones(100) * 3.0
        cycle = bk_filter(y_const, low=6, high=32, trunc=12)
        np.testing.assert_allclose(cycle, 0.0, atol=1e-12)

    def test_monthly_defaults(self) -> None:
        """Test with monthly-appropriate parameters."""
        y_monthly = np.random.randn(500).cumsum()
        cycle = bk_filter(y_monthly, low=18, high=96, trunc=36)
        assert len(cycle) == 500 - 72
