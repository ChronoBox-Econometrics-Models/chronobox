"""Tests for CUSUM and CUSUMSQ tests."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.structural_breaks.cusum import cusum_test, cusumsq_test


class TestCUSUM:
    """Tests for cusum_test function."""

    def test_cusum_stable(self) -> None:
        """CUSUM should not reject for stable model."""
        np.random.seed(42)
        T = 200
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(T) * 0.5
        result = cusum_test(y, X)
        assert not result.reject_at_5pct

    def test_cusum_unstable(self) -> None:
        """CUSUM should reject for model with parameter change."""
        np.random.seed(42)
        T = 200
        bp = 100
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = np.zeros(T)
        y[:bp] = X[:bp] @ np.array([1.0, 2.0]) + np.random.randn(bp) * 0.5
        y[bp:] = X[bp:] @ np.array([5.0, -2.0]) + np.random.randn(T - bp) * 0.5
        result = cusum_test(y, X)
        # Should detect instability
        assert result.test_name == "CUSUM"
        assert "cusum_values" in result.additional_info

    def test_cusum_bands(self) -> None:
        """Test that significance bands are returned."""
        np.random.seed(42)
        T = 100
        X = np.ones((T, 1))
        y = np.random.randn(T)
        result = cusum_test(y, X)
        assert "upper_band" in result.additional_info
        assert "lower_band" in result.additional_info


class TestCUSUMSQ:
    """Tests for cusumsq_test function."""

    def test_cusumsq_stable(self) -> None:
        """CUSUMSQ should not reject for constant variance."""
        np.random.seed(42)
        T = 200
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(T) * 0.5
        result = cusumsq_test(y, X)
        assert not result.reject_at_5pct

    def test_cusumsq_values(self) -> None:
        """Test that CUSUMSQ values are between 0 and 1."""
        np.random.seed(42)
        T = 100
        X = np.ones((T, 1))
        y = np.random.randn(T)
        result = cusumsq_test(y, X)
        S = result.additional_info["cusumsq_values"]
        assert np.all(S >= -0.01)
        assert np.all(S <= 1.01)
