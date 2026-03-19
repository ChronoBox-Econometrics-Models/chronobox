"""Tests for the Jarque-Bera normality test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.specification.jarque_bera import jarque_bera_test


class TestJarqueBera:
    """Tests for jarque_bera_test function."""

    def test_jb_normal(self) -> None:
        """JB should not reject for Normal data."""
        np.random.seed(42)
        data = np.random.randn(1000)
        result = jarque_bera_test(data)
        assert not result.reject_at_5pct

    def test_jb_skewed(self) -> None:
        """JB should reject for skewed data."""
        np.random.seed(42)
        # Exponential distribution (positively skewed)
        data = np.random.exponential(1.0, size=500)
        result = jarque_bera_test(data)
        assert result.reject_at_5pct

    def test_jb_heavy_tails(self) -> None:
        """JB should reject for heavy-tailed data."""
        np.random.seed(42)
        # t-distribution with 3 df (excess kurtosis)
        data = np.random.standard_t(3, size=500)
        result = jarque_bera_test(data)
        assert result.reject_at_5pct

    def test_jb_skewness_kurtosis(self) -> None:
        """Test that skewness and kurtosis are returned."""
        np.random.seed(42)
        data = np.random.randn(500)
        result = jarque_bera_test(data)
        assert "skewness" in result.additional_info
        assert "kurtosis" in result.additional_info
        # For Normal: S ~ 0, K ~ 3
        assert abs(result.additional_info["skewness"]) < 0.5
        assert abs(result.additional_info["kurtosis"] - 3.0) < 1.0

    def test_jb_pvalue_range(self) -> None:
        """Test that p-value is in [0, 1]."""
        np.random.seed(42)
        data = np.random.randn(200)
        result = jarque_bera_test(data)
        assert 0.0 <= result.pvalue <= 1.0
