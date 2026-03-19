"""Tests for the Augmented Dickey-Fuller unit root test."""

from __future__ import annotations

import numpy as np
import pytest

from chronobox.tests_stat.unit_root.adf import adf_test


class TestADF:
    """Tests for adf_test function."""

    def test_adf_random_walk_no_reject(self) -> None:
        """ADF should not reject H0 for a random walk (unit root present)."""
        np.random.seed(42)
        T = 500
        y = np.cumsum(np.random.randn(T))
        result = adf_test(y, regression="c", autolag="AIC")
        assert not result.reject_at_5pct, (
            f"Should not reject H0 for random walk, stat={result.statistic:.4f}"
        )
        assert result.test_name == "Augmented Dickey-Fuller"

    def test_adf_stationary_rejects(self) -> None:
        """ADF should reject H0 for a stationary AR(1) process."""
        np.random.seed(42)
        T = 500
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + np.random.randn()
        result = adf_test(y, regression="c", autolag="AIC")
        assert result.reject_at_5pct, (
            f"Should reject H0 for stationary AR(1), stat={result.statistic:.4f}"
        )

    def test_adf_nile(self) -> None:
        """ADF on Nile dataset (classic test case)."""
        nile = np.array([
            1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140,
            995, 935, 1110, 994, 1020, 960, 1180, 799, 958, 1140,
            1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100, 774, 840,
            874, 694, 940, 833, 701, 916, 692, 1020, 1050, 969,
            831, 726, 456, 824, 702, 1120, 1100, 832, 764, 821,
            768, 845, 864, 862, 698, 845, 744, 796, 1040, 759,
            781, 865, 845, 944, 984, 897, 822, 1010, 771, 676,
            649, 846, 812, 742, 801, 1040, 860, 874, 848, 890,
            744, 749, 838, 1050, 918, 986, 797, 923, 975, 815,
            1020, 906, 901, 1170, 912, 746, 919, 718, 714, 740,
        ], dtype=float)
        result = adf_test(nile, regression="c", autolag="AIC")
        assert result.statistic < -2.5, (
            f"ADF stat for Nile should be strongly negative, got {result.statistic:.4f}"
        )

    def test_adf_lag_selection_aic(self) -> None:
        """Test that AIC lag selection picks a reasonable lag."""
        np.random.seed(123)
        T = 200
        y = np.zeros(T)
        for t in range(2, T):
            y[t] = 0.5 * y[t - 1] + 0.2 * y[t - 2] + np.random.randn()
        result = adf_test(y, regression="c", autolag="AIC")
        assert result.lags_used is not None
        assert 0 <= result.lags_used <= 15

    def test_adf_lag_selection_bic(self) -> None:
        """Test BIC lag selection."""
        np.random.seed(123)
        T = 200
        y = np.cumsum(np.random.randn(T))
        result = adf_test(y, regression="c", autolag="BIC")
        assert result.lags_used is not None

    def test_adf_lag_selection_tsig(self) -> None:
        """Test t-significance lag selection."""
        np.random.seed(123)
        T = 200
        y = np.cumsum(np.random.randn(T))
        result = adf_test(y, regression="c", autolag="t-sig")
        assert result.lags_used is not None

    def test_adf_no_autolag(self) -> None:
        """Test with fixed lag (no automatic selection)."""
        np.random.seed(42)
        T = 200
        y = np.cumsum(np.random.randn(T))
        result = adf_test(y, regression="c", maxlag=4, autolag=None)
        assert result.lags_used == 4

    def test_adf_models(self) -> None:
        """Test all three regression models."""
        np.random.seed(42)
        T = 300
        y = np.cumsum(np.random.randn(T))
        for model in ("n", "c", "ct"):
            result = adf_test(y, regression=model)
            assert result.additional_info["regression"] == model

    def test_adf_critical_values_ordered(self) -> None:
        """Test that critical values are properly ordered."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(200))
        result = adf_test(y, regression="c")
        cv = result.critical_values
        assert cv["1%"] < cv["5%"] < cv["10%"]

    def test_adf_summary(self) -> None:
        """Test that summary() produces readable output."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(200))
        result = adf_test(y, regression="c")
        summary = result.summary()
        assert "Augmented Dickey-Fuller" in summary
        assert "H0:" in summary

    def test_adf_invalid_regression(self) -> None:
        """Test that invalid regression model raises error."""
        y = np.random.randn(100)
        with pytest.raises(ValueError, match="Invalid regression"):
            adf_test(y, regression="invalid")

    def test_adf_short_series(self) -> None:
        """Test that very short series raises error."""
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="too short"):
            adf_test(y)

    def test_adf_pvalue_stationary(self) -> None:
        """Test that p-value is small for stationary series."""
        np.random.seed(42)
        T = 500
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.3 * y[t - 1] + np.random.randn()
        result = adf_test(y, regression="c")
        assert result.pvalue is not None
        assert result.pvalue < 0.05

    def test_adf_pvalue_unit_root(self) -> None:
        """Test that p-value is large for unit root series."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(500))
        result = adf_test(y, regression="c")
        assert result.pvalue is not None
        assert result.pvalue > 0.05
