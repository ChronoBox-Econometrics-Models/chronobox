"""Tests for the Chow structural break test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.structural_breaks.chow import chow_test


class TestChow:
    """Tests for chow_test function."""

    def test_chow_break_detected(self) -> None:
        """Chow should detect a structural break when present."""
        np.random.seed(42)
        T = 200
        bp = 100
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = np.zeros(T)
        y[:bp] = X[:bp] @ np.array([1.0, 2.0]) + np.random.randn(bp) * 0.5
        y[bp:] = X[bp:] @ np.array([5.0, -1.0]) + np.random.randn(T - bp) * 0.5

        result = chow_test(y, X, break_point=bp)
        assert result.reject_at_5pct, (
            f"Should detect break, F={result.statistic:.4f}, p={result.pvalue:.4f}"
        )

    def test_chow_no_break(self) -> None:
        """Chow should not reject when no break exists."""
        np.random.seed(42)
        T = 200
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(T) * 0.5

        result = chow_test(y, X, break_point=100)
        assert not result.reject_at_5pct, (
            f"Should not reject, F={result.statistic:.4f}, p={result.pvalue:.4f}"
        )

    def test_chow_pvalue(self) -> None:
        """Test that p-value is reasonable."""
        np.random.seed(42)
        T = 200
        X = np.ones((T, 1))
        y = np.random.randn(T)
        result = chow_test(y, X, break_point=100)
        assert result.pvalue is not None
        assert 0.0 <= result.pvalue <= 1.0

    def test_chow_ssr_values(self) -> None:
        """Test that SSR values are in additional_info."""
        np.random.seed(42)
        T = 200
        X = np.ones((T, 1))
        y = np.random.randn(T)
        result = chow_test(y, X, break_point=100)
        assert "SSR_full" in result.additional_info
        assert "SSR_1" in result.additional_info
        assert "SSR_2" in result.additional_info
        # SSR_full >= SSR_1 + SSR_2
        assert result.additional_info["SSR_full"] >= (
            result.additional_info["SSR_1"] + result.additional_info["SSR_2"] - 0.01
        )
