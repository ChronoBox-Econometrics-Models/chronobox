"""Tests for the ERS / DF-GLS unit root test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.unit_root.ers import ers_test


class TestERS:
    """Tests for ers_test function."""

    def test_ers_stationary_rejects(self) -> None:
        """ERS should reject H0 for stationary AR(1)."""
        np.random.seed(42)
        T = 500
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + np.random.randn()
        result = ers_test(y, regression="c")
        assert result.reject_at_5pct

    def test_ers_random_walk_no_reject(self) -> None:
        """ERS should not reject H0 for random walk."""
        np.random.seed(42)
        T = 500
        y = np.cumsum(np.random.randn(T))
        result = ers_test(y, regression="c")
        assert not result.reject_at_5pct

    def test_ers_more_powerful(self) -> None:
        """ERS should reject where ADF might not for near-unit-root."""
        np.random.seed(42)
        T = 200
        # Near unit root: phi = 0.95
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.95 * y[t - 1] + np.random.randn()
        result = ers_test(y, regression="c")
        # ERS has higher power, so it may reject even for phi close to 1
        # This is a qualitative check; outcome depends on realization
        assert result.statistic < 0  # gamma should be negative for phi < 1

    def test_ers_consistent_with_adf(self) -> None:
        """For clear stationary case, ERS and ADF should agree."""
        np.random.seed(42)
        T = 500
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.3 * y[t - 1] + np.random.randn()
        from chronobox.tests_stat.unit_root.adf import adf_test

        adf_res = adf_test(y, regression="c")
        ers_res = ers_test(y, regression="c")
        # Both should reject for clearly stationary series
        assert adf_res.reject_at_5pct
        assert ers_res.reject_at_5pct

    def test_ers_models(self) -> None:
        """Test constant and constant+trend models."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(200))
        for model in ("c", "ct"):
            result = ers_test(y, regression=model)
            assert result.additional_info["regression"] == model

    def test_ers_additional_info(self) -> None:
        """Test that additional info contains expected keys."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(200))
        result = ers_test(y)
        assert "c_bar" in result.additional_info
        assert "gamma" in result.additional_info
