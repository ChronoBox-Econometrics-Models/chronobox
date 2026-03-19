"""Tests for the White heteroskedasticity test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.specification.white import white_test


class TestWhite:
    """Tests for white_test function."""

    def test_white_homoskedastic(self) -> None:
        """White should not reject for homoskedastic data."""
        np.random.seed(42)
        T = 200
        X = np.random.randn(T, 2)
        resid = np.random.randn(T) * 1.0
        result = white_test(resid, X)
        assert not result.reject_at_5pct

    def test_white_heteroskedastic(self) -> None:
        """White should reject for heteroskedastic data."""
        np.random.seed(42)
        T = 500
        X = np.random.randn(T, 1)
        # Variance depends on X: eps ~ N(0, X^2)
        resid = np.random.randn(T) * np.abs(X[:, 0]) * 3.0
        result = white_test(resid, X)
        assert result.reject_at_5pct

    def test_white_no_cross(self) -> None:
        """Test without cross terms."""
        np.random.seed(42)
        T = 200
        X = np.random.randn(T, 2)
        resid = np.random.randn(T)
        result = white_test(resid, X, cross_terms=False)
        assert not result.reject_at_5pct

    def test_white_pvalue(self) -> None:
        """Test that p-value is in [0, 1]."""
        np.random.seed(42)
        T = 200
        X = np.random.randn(T, 1)
        resid = np.random.randn(T)
        result = white_test(resid, X)
        assert 0.0 <= result.pvalue <= 1.0
