"""Tests for the Phillips-Ouliaris cointegration test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.cointegration.phillips_ouliaris import phillips_ouliaris_test


class TestPhillipsOuliaris:
    """Tests for phillips_ouliaris_test function."""

    def test_po_cointegrated(self) -> None:
        """PO should reject H0 for cointegrated series."""
        np.random.seed(42)
        T = 300
        x = np.cumsum(np.random.randn(T))
        y = 2.0 * x + np.random.randn(T) * 0.5
        result = phillips_ouliaris_test(y, x)
        assert result.reject_at_5pct

    def test_po_not_cointegrated(self) -> None:
        """PO should not reject H0 for independent random walks."""
        np.random.seed(42)
        T = 300
        x = np.cumsum(np.random.randn(T))
        y = np.cumsum(np.random.randn(T))
        result = phillips_ouliaris_test(y, x)
        assert not result.reject_at_5pct

    def test_po_z_alpha(self) -> None:
        """Test that Z_alpha statistic is returned."""
        np.random.seed(42)
        T = 200
        x = np.cumsum(np.random.randn(T))
        y = x + np.random.randn(T)
        result = phillips_ouliaris_test(y, x)
        assert "Z_alpha" in result.additional_info

    def test_po_cointegrating_vector(self) -> None:
        """Test that cointegrating vector is returned."""
        np.random.seed(42)
        T = 300
        x = np.cumsum(np.random.randn(T))
        y = 1.5 * x + np.random.randn(T) * 0.3
        result = phillips_ouliaris_test(y, x)
        beta = result.additional_info["cointegrating_vector"]
        assert abs(beta[0] - 1.5) < 0.2

    def test_po_models(self) -> None:
        """Test constant and constant+trend models."""
        np.random.seed(42)
        T = 200
        x = np.cumsum(np.random.randn(T))
        y = x + np.random.randn(T) * 0.5
        for trend in ("c", "ct"):
            result = phillips_ouliaris_test(y, x, trend=trend)
            assert result.additional_info["trend"] == trend
