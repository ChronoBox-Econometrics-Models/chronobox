"""Tests for the Christiano-Fitzgerald band-pass filter."""

import numpy as np
import pytest

from chronobox.filters.bk import bk_filter
from chronobox.filters.cf import cf_filter


class TestCFFilter:
    """Tests for cf_filter."""

    def setup_method(self) -> None:
        """Set up test data."""
        np.random.seed(42)
        self.T = 200
        self.y = np.random.randn(self.T).cumsum()

    def test_full_sample_no_observation_loss(self) -> None:
        """CF filter should return same length as input."""
        cycle = cf_filter(self.y, low=6, high=32)
        assert len(cycle) == self.T

    def test_similar_to_bk_in_interior(self) -> None:
        """CF and BK should be highly correlated in the interior."""
        trunc = 12
        cycle_cf = cf_filter(self.y, low=6, high=32)
        cycle_bk = bk_filter(self.y, low=6, high=32, trunc=trunc)

        # Compare interior: CF[K:-K] vs BK (which already lost K at each end)
        cf_interior = cycle_cf[trunc : self.T - trunc]
        corr = np.corrcoef(cf_interior, cycle_bk)[0, 1]
        assert corr > 0.90, f"Correlation {corr} too low (expected > 0.90)"

    def test_output_shape(self) -> None:
        """Output shape should match input."""
        cycle = cf_filter(self.y, low=6, high=32)
        assert cycle.shape == (self.T,)

    def test_constant_series_zero_cycle(self) -> None:
        """Constant series should have zero cycle."""
        y_const = np.ones(100) * 5.0
        cycle = cf_filter(y_const, low=6, high=32)
        np.testing.assert_allclose(cycle, 0.0, atol=1e-10)

    def test_known_cycle_extraction(self) -> None:
        """Should extract a known cycle component."""
        t = np.arange(300)
        cycle_true = np.sin(2 * np.pi * t / 16)  # period=16
        trend = 0.01 * t
        y = trend + cycle_true

        cycle_cf = cf_filter(y, low=6, high=32)
        # In the interior, should correlate well with true cycle
        margin = 20
        corr = np.corrcoef(
            cycle_cf[margin:-margin],
            cycle_true[margin:-margin],
        )[0, 1]
        assert corr > 0.85, f"Correlation {corr} too low"

    def test_invalid_inputs(self) -> None:
        """Should raise for invalid inputs."""
        with pytest.raises(ValueError, match="1-D"):
            cf_filter(np.random.randn(10, 2))
        with pytest.raises(ValueError, match="low must be >= 2"):
            cf_filter(self.y, low=1)
        with pytest.raises(ValueError, match="high must be > low"):
            cf_filter(self.y, low=32, high=6)
        with pytest.raises(ValueError, match="at least 4"):
            cf_filter(np.array([1.0, 2.0, 3.0]))

    def test_with_drift_option(self) -> None:
        """Drift removal should not crash and should return valid output."""
        cycle = cf_filter(self.y, low=6, high=32, drift=True)
        assert len(cycle) == self.T
        assert np.all(np.isfinite(cycle))
