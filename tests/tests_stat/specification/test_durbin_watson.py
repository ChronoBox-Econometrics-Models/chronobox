"""Tests for the Durbin-Watson test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.specification.durbin_watson import durbin_watson_test


class TestDurbinWatson:
    """Tests for durbin_watson_test function."""

    def test_dw_no_autocorrelation(self) -> None:
        """DW should be near 2 for iid residuals."""
        np.random.seed(42)
        resid = np.random.randn(500)
        result = durbin_watson_test(resid)
        assert 1.5 < result.statistic < 2.5

    def test_dw_positive_autocorrelation(self) -> None:
        """DW should be near 0 for positive autocorrelation."""
        np.random.seed(42)
        T = 500
        resid = np.zeros(T)
        for t in range(1, T):
            resid[t] = 0.9 * resid[t - 1] + np.random.randn() * 0.1
        result = durbin_watson_test(resid)
        assert result.statistic < 1.0

    def test_dw_range(self) -> None:
        """DW statistic should be in [0, 4]."""
        np.random.seed(42)
        resid = np.random.randn(100)
        result = durbin_watson_test(resid)
        assert 0.0 <= result.statistic <= 4.0
