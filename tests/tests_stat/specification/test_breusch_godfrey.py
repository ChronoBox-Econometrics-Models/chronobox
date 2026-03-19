"""Tests for the Breusch-Godfrey test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.specification.breusch_godfrey import breusch_godfrey_test


class TestBreuschGodfrey:
    """Tests for breusch_godfrey_test function."""

    def test_bg_no_autocorrelation(self) -> None:
        """BG should not reject for iid residuals."""
        np.random.seed(42)
        T = 200
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        resid = np.random.randn(T)
        result = breusch_godfrey_test(resid, X, nlags=4)
        assert not result.reject_at_5pct

    def test_bg_autocorrelated(self) -> None:
        """BG should reject for autocorrelated residuals."""
        np.random.seed(42)
        T = 200
        X = np.column_stack([np.ones(T), np.random.randn(T)])
        resid = np.zeros(T)
        for t in range(1, T):
            resid[t] = 0.7 * resid[t - 1] + np.random.randn()
        result = breusch_godfrey_test(resid, X, nlags=4)
        assert result.reject_at_5pct

    def test_bg_f_stat(self) -> None:
        """Test that F-statistic is returned."""
        np.random.seed(42)
        T = 200
        X = np.ones((T, 1))
        resid = np.random.randn(T)
        result = breusch_godfrey_test(resid, X, nlags=2)
        assert "F_stat" in result.additional_info
        assert "pvalue_F" in result.additional_info
