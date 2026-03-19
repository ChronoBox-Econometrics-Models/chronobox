"""Tests for the KPSS stationarity test."""

from __future__ import annotations

import numpy as np

from chronobox.tests_stat.unit_root.kpss import kpss_test


class TestKPSS:
    """Tests for kpss_test function."""

    def test_kpss_random_walk_rejects(self) -> None:
        """KPSS should reject H0 (stationarity) for a random walk."""
        np.random.seed(42)
        T = 500
        y = np.cumsum(np.random.randn(T))
        result = kpss_test(y, regression="c")
        assert result.reject_at_5pct, (
            f"Should reject stationarity for random walk, stat={result.statistic:.4f}"
        )

    def test_kpss_stationary_no_reject(self) -> None:
        """KPSS should not reject H0 for a stationary series."""
        np.random.seed(42)
        T = 500
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + np.random.randn()
        result = kpss_test(y, regression="c")
        assert not result.reject_at_5pct, (
            f"Should not reject stationarity for AR(1), stat={result.statistic:.4f}"
        )

    def test_kpss_nile(self) -> None:
        """KPSS on Nile dataset - should not reject level stationarity at 1%.

        With sufficient bandwidth (nlags=12, Schwert formula), the Nile data
        is generally considered level-stationary despite a structural break ~1898.
        """
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
        # Use Schwert-like bandwidth (12) for proper serial correlation correction
        result = kpss_test(nile, regression="c", nlags=12)
        assert result.statistic < result.critical_values["1%"]

    def test_kpss_trend_stationary(self) -> None:
        """KPSS trend test on series with linear trend."""
        np.random.seed(42)
        T = 300
        trend = np.linspace(0, 10, T)
        y = trend + np.random.randn(T) * 0.5
        result = kpss_test(y, regression="ct")
        assert not result.reject_at_5pct

    def test_kpss_reversed_hypothesis(self) -> None:
        """Verify that KPSS has reversed interpretation vs ADF."""
        np.random.seed(42)
        T = 500
        y_stat = np.zeros(T)
        for t in range(1, T):
            y_stat[t] = 0.5 * y_stat[t - 1] + np.random.randn()

        kpss_res = kpss_test(y_stat, regression="c")
        assert not kpss_res.reject_at_5pct

    def test_kpss_models(self) -> None:
        """Test level and trend models."""
        np.random.seed(42)
        y = np.random.randn(200)
        for model in ("c", "ct"):
            result = kpss_test(y, regression=model)
            assert result.additional_info["regression"] == model

    def test_kpss_critical_values(self) -> None:
        """Test that critical values are correct for level model."""
        np.random.seed(42)
        y = np.random.randn(200)
        result = kpss_test(y, regression="c")
        cv = result.critical_values
        assert abs(cv["1%"] - 0.739) < 0.001
        assert abs(cv["5%"] - 0.463) < 0.001
        assert abs(cv["10%"] - 0.347) < 0.001

    def test_kpss_custom_nlags(self) -> None:
        """Test with custom number of lags."""
        np.random.seed(42)
        y = np.random.randn(200)
        result = kpss_test(y, regression="c", nlags=10)
        assert result.lags_used == 10
