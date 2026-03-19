"""Tests for the QLR (sup-Wald) test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.structural_breaks.qlr import qlr_test


class TestQLR:
    """Tests for qlr_test function."""

    def test_qlr_break_detected(self) -> None:
        """QLR should detect a structural break."""
        np.random.seed(42)
        T = 200
        bp = 100
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = np.zeros(T)
        y[:bp] = X[:bp] @ np.array([1.0, 2.0]) + np.random.randn(bp) * 0.5
        y[bp:] = X[bp:] @ np.array([5.0, -1.0]) + np.random.randn(T - bp) * 0.5

        result = qlr_test(y, X, trim=0.15)
        assert result.reject_at_5pct

    def test_qlr_no_break(self) -> None:
        """QLR should not reject for stable model."""
        np.random.seed(42)
        T = 200
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(T) * 0.5

        result = qlr_test(y, X, trim=0.15)
        assert not result.reject_at_5pct

    def test_qlr_break_date(self) -> None:
        """QLR break date should be near true break."""
        np.random.seed(42)
        T = 200
        bp = 100
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        y = np.zeros(T)
        y[:bp] = X[:bp] @ np.array([0.0, 3.0]) + np.random.randn(bp) * 0.5
        y[bp:] = X[bp:] @ np.array([5.0, -3.0]) + np.random.randn(T - bp) * 0.5

        result = qlr_test(y, X, trim=0.15)
        break_date = result.additional_info["break_date"]
        assert abs(break_date - bp) < 20, (
            f"Break at {break_date}, expected near {bp}"
        )

    def test_qlr_wald_sequence(self) -> None:
        """Test that Wald sequence is returned."""
        np.random.seed(42)
        T = 100
        X = np.ones((T, 1))
        y = np.random.randn(T)
        result = qlr_test(y, X)
        assert "wald_sequence" in result.additional_info
        assert len(result.additional_info["wald_sequence"]) == T
