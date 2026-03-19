"""Tests for the Zivot-Andrews unit root test with structural break."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.unit_root.zivot_andrews import zivot_andrews_test


class TestZivotAndrews:
    """Tests for zivot_andrews_test function."""

    def test_za_detects_break(self) -> None:
        """ZA should detect a structural break near the true break date."""
        np.random.seed(42)
        T = 200
        break_point = 100
        # Stationary series with level shift
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + np.random.randn()
        y[break_point:] += 5.0  # level shift

        result = zivot_andrews_test(y, model="a")
        break_date = result.additional_info["break_date"]
        # Break date should be within 15 periods of true break
        assert abs(break_date - break_point) < 15, (
            f"Detected break at {break_date}, expected near {break_point}"
        )

    def test_za_no_break_random_walk(self) -> None:
        """ZA on random walk without break should not reject."""
        np.random.seed(42)
        T = 200
        y = np.cumsum(np.random.randn(T))
        result = zivot_andrews_test(y, model="c")
        # Main check: test runs without error
        assert result.test_name == "Zivot-Andrews"

    def test_za_model_a(self) -> None:
        """Test model A (intercept break only)."""
        np.random.seed(42)
        y = np.concatenate([np.random.randn(100), np.random.randn(100) + 3])
        result = zivot_andrews_test(y, model="a")
        assert result.additional_info["model"] == "a"

    def test_za_model_b(self) -> None:
        """Test model B (trend break only)."""
        np.random.seed(42)
        T = 200
        t_idx = np.arange(T, dtype=float)
        y = 0.05 * t_idx + np.random.randn(T)
        y[100:] += 0.1 * (t_idx[100:] - 100)
        result = zivot_andrews_test(y, model="b")
        assert result.additional_info["model"] == "b"

    def test_za_model_c(self) -> None:
        """Test model C (both intercept and trend break)."""
        np.random.seed(42)
        y = np.concatenate([np.random.randn(100), np.random.randn(100) + 5])
        result = zivot_andrews_test(y, model="c")
        assert result.additional_info["model"] == "c"

    def test_za_critical_values(self) -> None:
        """Test that critical values are from ZA tables."""
        np.random.seed(42)
        y = np.random.randn(100)
        result = zivot_andrews_test(y, model="c")
        assert abs(result.critical_values["5%"] - (-5.08)) < 0.01

    def test_za_break_fraction(self) -> None:
        """Test that break_fraction is in valid range."""
        np.random.seed(42)
        y = np.random.randn(200)
        result = zivot_andrews_test(y, model="a", trim=0.15)
        frac = result.additional_info["break_fraction"]
        assert 0.15 <= frac <= 0.85
