"""Tests for the Engle-Granger cointegration test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.cointegration.engle_granger import engle_granger_test


class TestEngleGranger:
    """Tests for engle_granger_test function."""

    def test_eg_cointegrated(self) -> None:
        """EG should reject H0 for cointegrated series."""
        np.random.seed(42)
        T = 300
        x = np.cumsum(np.random.randn(T))
        y = 2.0 * x + np.random.randn(T) * 0.5
        result = engle_granger_test(y, x)
        assert result.reject_at_5pct, (
            f"Should reject no-cointegration, stat={result.statistic:.4f}"
        )

    def test_eg_not_cointegrated(self) -> None:
        """EG should not reject H0 for independent random walks."""
        np.random.seed(42)
        T = 300
        x = np.cumsum(np.random.randn(T))
        y = np.cumsum(np.random.randn(T))
        result = engle_granger_test(y, x)
        assert not result.reject_at_5pct, (
            f"Should not reject for independent RWs, stat={result.statistic:.4f}"
        )

    def test_eg_vector(self) -> None:
        """Test that cointegrating vector is approximately correct."""
        np.random.seed(42)
        T = 500
        x = np.cumsum(np.random.randn(T))
        y = 2.0 * x + np.random.randn(T) * 0.3
        result = engle_granger_test(y, x)
        beta = result.additional_info["cointegrating_vector"]
        assert abs(beta[0] - 2.0) < 0.2, (
            f"Cointegrating vector should be ~2.0, got {beta[0]:.4f}"
        )

    def test_eg_multivariate(self) -> None:
        """Test with multiple regressors."""
        np.random.seed(42)
        T = 300
        x1 = np.cumsum(np.random.randn(T))
        x2 = np.cumsum(np.random.randn(T))
        y = 1.5 * x1 + 0.8 * x2 + np.random.randn(T) * 0.5
        x = np.column_stack([x1, x2])
        result = engle_granger_test(y, x)
        assert result.reject_at_5pct
        assert result.additional_info["n_vars"] == 3

    def test_eg_residuals(self) -> None:
        """Test that residuals are returned."""
        np.random.seed(42)
        T = 200
        x = np.cumsum(np.random.randn(T))
        y = x + np.random.randn(T) * 0.5
        result = engle_granger_test(y, x)
        resid = result.additional_info["residuals"]
        assert len(resid) == T

    def test_eg_critical_values_more_negative(self) -> None:
        """EG critical values should be more negative than standard ADF."""
        np.random.seed(42)
        T = 200
        x = np.cumsum(np.random.randn(T))
        y = x + np.random.randn(T)
        result = engle_granger_test(y, x)
        # EG 5% CV for n=2 should be around -3.34 (more negative than ADF -2.86)
        assert result.critical_values["5%"] < -3.0
