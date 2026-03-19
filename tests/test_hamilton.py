"""Tests for the Hamilton regression filter."""

import numpy as np
import pytest

from chronobox.filters.hamilton import hamilton_filter, hamilton_filter_detailed


class TestHamiltonFilter:
    """Tests for hamilton_filter."""

    def setup_method(self) -> None:
        """Set up test data."""
        np.random.seed(42)
        self.T = 200
        self.y = np.random.randn(self.T).cumsum()

    def test_default_h8_p4(self) -> None:
        """Default should be h=8, p=4."""
        result = hamilton_filter_detailed(self.y)
        assert result.h == 8
        assert result.p == 4

    def test_no_spurious_cycles_random_walk(self) -> None:
        """Should not create spurious cycles on a random walk."""
        np.random.seed(123)
        # Pure random walk
        rw = np.random.randn(500).cumsum()
        _trend, cycle = hamilton_filter(rw, h=8, p=4)

        valid = ~np.isnan(cycle)
        cycle_valid = cycle[valid]

        # Hamilton filter on a RW should not produce cycles with variance
        # much smaller than the h-step innovation variance. For a RW with
        # unit innovations, the h-step forecast error has variance ~ h.
        # Check that the cycle std is in a reasonable range (not artificially
        # smooth, which would indicate spurious cycles).
        cycle_std = np.std(cycle_valid)
        # For a RW the cycle should scale roughly as sqrt(h)
        assert cycle_std > 0.5, f"Cycle std {cycle_std} suspiciously low"
        # The cycle mean should be near zero
        assert abs(np.mean(cycle_valid)) < 1.0, "Cycle mean too far from zero"

    def test_trend_plus_cycle_equals_original(self) -> None:
        """trend + cycle should equal original where both are valid."""
        trend, cycle = hamilton_filter(self.y, h=8, p=4)
        valid = ~np.isnan(cycle)
        np.testing.assert_allclose(
            trend[valid] + cycle[valid], self.y[valid], atol=1e-10
        )

    def test_nan_at_start(self) -> None:
        """First h+p-1 observations should be NaN."""
        h, p = 8, 4
        trend, cycle = hamilton_filter(self.y, h=h, p=p)
        # First h+p-1 should be NaN
        assert np.all(np.isnan(trend[: h + p - 1]))
        assert np.all(np.isnan(cycle[: h + p - 1]))
        # After that, should be valid
        assert np.all(np.isfinite(trend[h + p - 1 :]))
        assert np.all(np.isfinite(cycle[h + p - 1 :]))

    def test_vs_hp_correlation(self) -> None:
        """Hamilton and HP cycles should be correlated (both capture cycles)."""
        from chronobox.filters.hp import hp_filter as hp

        _trend_ham, cycle_ham = hamilton_filter(self.y, h=8, p=4)
        _, cycle_hp = hp(self.y, lamb=1600)

        valid = ~np.isnan(cycle_ham)
        corr = np.corrcoef(cycle_ham[valid], cycle_hp[valid])[0, 1]
        # Should be positively correlated (both capture cycles)
        assert corr > 0.3, f"Correlation with HP {corr} too low"

    def test_output_shapes(self) -> None:
        """Output shapes should match input."""
        trend, cycle = hamilton_filter(self.y, h=8, p=4)
        assert trend.shape == (self.T,)
        assert cycle.shape == (self.T,)

    def test_coefficients_returned(self) -> None:
        """Detailed result should include regression coefficients."""
        result = hamilton_filter_detailed(self.y, h=8, p=4)
        # p+1 coefficients: constant + p lags
        assert result.coefficients.shape == (5,)  # 1 + 4

    def test_invalid_inputs(self) -> None:
        """Should raise for invalid inputs."""
        with pytest.raises(ValueError, match="1-D"):
            hamilton_filter(np.random.randn(10, 2))
        with pytest.raises(ValueError, match="h must be >= 1"):
            hamilton_filter(self.y, h=0)
        with pytest.raises(ValueError, match="p must be >= 1"):
            hamilton_filter(self.y, p=0)
        with pytest.raises(ValueError, match="must be >"):
            hamilton_filter(np.ones(10), h=8, p=4)

    def test_different_h_p(self) -> None:
        """Should work with different h and p values."""
        for h, p in [(2, 1), (4, 2), (12, 6), (24, 12)]:
            if h + p < self.T:
                trend, cycle = hamilton_filter(self.y, h=h, p=p)
                valid = ~np.isnan(cycle)
                np.testing.assert_allclose(
                    trend[valid] + cycle[valid], self.y[valid], atol=1e-10
                )
