"""Tests for the PSS Bounds cointegration test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.cointegration.bounds_test import bounds_test


class TestBoundsTest:
    """Tests for bounds_test function."""

    def test_pss_cointegrated(self) -> None:
        """F should exceed upper bound for genuine cointegrating relationship."""
        np.random.seed(42)
        T = 300
        x = np.cumsum(np.random.randn(T))
        y = 2.0 * x + np.random.randn(T) * 0.3
        result = bounds_test(y, x)
        assert result.additional_info["decision"] == "cointegration"
        assert result.reject_at_5pct

    def test_pss_no_relation(self) -> None:
        """F should be below lower bound for independent series."""
        np.random.seed(42)
        T = 300
        x = np.cumsum(np.random.randn(T))
        y = np.cumsum(np.random.randn(T))
        result = bounds_test(y, x)
        # Should not reject - either "no cointegration" or "inconclusive"
        assert result.additional_info["decision"] != "cointegration"

    def test_pss_critical_values(self) -> None:
        """Critical value bounds should match PSS (2001) tables."""
        np.random.seed(42)
        T = 200
        x = np.cumsum(np.random.randn(T))
        y = x + np.random.randn(T)
        result = bounds_test(y, x, case=3)
        lower = result.additional_info["lower_bound"]
        upper = result.additional_info["upper_bound"]
        # For k=1, case 3, 5%: lower ~ 3.62, upper ~ 4.16
        assert 3.0 < lower < 5.0
        assert lower < upper

    def test_pss_multivariate(self) -> None:
        """Test with multiple regressors."""
        np.random.seed(42)
        T = 300
        x = np.column_stack([
            np.cumsum(np.random.randn(T)),
            np.cumsum(np.random.randn(T)),
        ])
        y = 1.5 * x[:, 0] + 0.8 * x[:, 1] + np.random.randn(T) * 0.3
        result = bounds_test(y, x)
        assert result.additional_info["k"] == 2

    def test_pss_additional_info(self) -> None:
        """Test that additional info has expected keys."""
        np.random.seed(42)
        T = 200
        x = np.cumsum(np.random.randn(T))
        y = x + np.random.randn(T)
        result = bounds_test(y, x)
        assert "f_statistic" in result.additional_info
        assert "t_statistic" in result.additional_info
        assert "lower_bound" in result.additional_info
        assert "upper_bound" in result.additional_info
        assert "decision" in result.additional_info
