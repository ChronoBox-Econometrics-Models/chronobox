"""Tests for the Ljung-Box test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.specification.ljung_box import ljung_box_test


class TestLjungBox:
    """Tests for ljung_box_test function."""

    def test_lb_white_noise(self) -> None:
        """LB should not reject for white noise."""
        np.random.seed(42)
        resid = np.random.randn(500)
        result = ljung_box_test(resid, lags=10)
        assert not result.reject_at_5pct

    def test_lb_ar1_residuals(self) -> None:
        """LB should reject for autocorrelated residuals."""
        np.random.seed(42)
        T = 500
        resid = np.zeros(T)
        for t in range(1, T):
            resid[t] = 0.7 * resid[t - 1] + np.random.randn()
        result = ljung_box_test(resid, lags=10)
        assert result.reject_at_5pct

    def test_lb_pvalue_range(self) -> None:
        """Test that p-value is in [0, 1]."""
        np.random.seed(42)
        resid = np.random.randn(200)
        result = ljung_box_test(resid, lags=5)
        assert 0.0 <= result.pvalue <= 1.0

    def test_lb_model_df(self) -> None:
        """Test with model degrees of freedom adjustment."""
        np.random.seed(42)
        resid = np.random.randn(200)
        result = ljung_box_test(resid, lags=10, model_df=2)
        assert result.additional_info["model_df"] == 2
        assert result.additional_info["df"] == 8
