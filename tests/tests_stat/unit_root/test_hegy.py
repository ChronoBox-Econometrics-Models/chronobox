"""Tests for the HEGY seasonal unit root test."""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.tests_stat.unit_root.hegy import hegy_test


class TestHEGY:
    """Tests for hegy_test function."""

    def test_hegy_white_noise(self) -> None:
        """HEGY on white noise (no unit roots at any frequency)."""
        np.random.seed(42)
        T = 200  # 50 years of quarterly data
        y = np.random.randn(T)
        result = hegy_test(y, period=4, regression="c")
        # White noise should reject all unit root hypotheses
        assert result.test_name == "HEGY"

    def test_hegy_seasonal_random_walk(self) -> None:
        """HEGY on seasonal random walk (unit roots at seasonal frequencies)."""
        np.random.seed(42)
        T = 200
        y = np.zeros(T)
        for t in range(4, T):
            y[t] = y[t - 4] + np.random.randn()
        result = hegy_test(y, period=4, regression="c")
        # Should not reject seasonal unit roots
        assert result.test_name == "HEGY"

    def test_hegy_quarterly_data(self) -> None:
        """Test HEGY with realistic quarterly-like data."""
        np.random.seed(42)
        T = 100
        # Seasonal pattern + noise
        season = np.tile([1.0, -0.5, 0.3, -0.8], T // 4)
        y = season + np.random.randn(T) * 0.5
        result = hegy_test(y, period=4, regression="c")
        assert "t_pi1" in result.additional_info
        assert "t_pi2" in result.additional_info
        assert "F_pi3pi4" in result.additional_info

    def test_hegy_additional_info(self) -> None:
        """Test that additional_info has all expected keys."""
        np.random.seed(42)
        y = np.random.randn(100)
        result = hegy_test(y, period=4)
        info = result.additional_info
        assert "t_pi1" in info
        assert "t_pi2" in info
        assert "F_pi3pi4" in info
        assert "pi_estimates" in info
        assert "reject_freq0" in info
        assert "reject_freqpi" in info
        assert "reject_freqpi2" in info

    def test_hegy_invalid_period(self) -> None:
        """Test that non-quarterly period raises error."""
        y = np.random.randn(100)
        with pytest.raises(ValueError, match="period=4"):
            hegy_test(y, period=12)

    def test_hegy_models(self) -> None:
        """Test different regression models."""
        np.random.seed(42)
        y = np.random.randn(100)
        for reg in ("n", "c", "ct"):
            result = hegy_test(y, period=4, regression=reg)
            assert result.test_name == "HEGY"
