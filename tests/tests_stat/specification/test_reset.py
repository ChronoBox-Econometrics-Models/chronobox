"""Tests for the Ramsey RESET test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.specification.reset import reset_test


class TestRESET:
    """Tests for reset_test function."""

    def test_reset_linear(self) -> None:
        """RESET should not reject for correctly specified linear model."""
        np.random.seed(42)
        T = 300
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(T) * 0.5
        result = reset_test(y, X, power=3)
        assert not result.reject_at_5pct

    def test_reset_nonlinear(self) -> None:
        """RESET should reject for linear model fit to quadratic DGP."""
        np.random.seed(42)
        T = 300
        x = np.random.randn(T)
        y = 1.0 + 2.0 * x + 3.0 * x**2 + np.random.randn(T) * 0.5
        X = np.column_stack([np.ones(T), x])  # Linear model (missing x^2)
        result = reset_test(y, X, power=3)
        assert result.reject_at_5pct

    def test_reset_pvalue(self) -> None:
        """Test that p-value is in [0, 1]."""
        np.random.seed(42)
        T = 200
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(T) * 0.5
        result = reset_test(y, X)
        assert 0.0 <= result.pvalue <= 1.0

    def test_reset_power_parameter(self) -> None:
        """Test with different power settings."""
        np.random.seed(42)
        T = 200
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(T)
        for p in (2, 3, 4):
            result = reset_test(y, X, power=p)
            assert result.additional_info["power"] == p
